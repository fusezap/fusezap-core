"""fusezap.serve on CUDA (P4-CUDA-SERVE): KV kernels, the tensor-core linears, W8A8 FP8, the bf16/fp8 model and engine.

Runs on a CUDA device (skips elsewhere, except the source-rendering checks). The model tests use Qwen/Qwen2.5-0.5B-Instruct
from the HF cache; tests marked slow build LLMEngines (FZ_SLOW=1). The 7B serving smoke test (Qwen/Qwen2.5-7B-Instruct over
the OpenAI API, bf16 and fp8) runs only with FZ_7B=1 (it downloads 15 GB when FZ_HF_DOWNLOAD=1 and needs a ~40 GB+ GPU).
FZ_MODEL7=<hf id> runs every FZ_7B test on another 7-8B checkpoint (MODEL2: mistralai/Mistral-7B-Instruct-v0.3). On a
48 GB GPU run the two HF-reference tests (top1_vs_hf_spec, f32_greedy) each in its own pytest process: a 7B engine
built by an earlier test keeps its device memory for the life of the process. Qwen/Qwen2.5-32B-Instruct runs them on an
H100 (docs/status/H100-32B.md).
The f32 token-identical-to-HF and 32-concurrent gates are tests/test_serve_model.py and tests/test_scheduler.py with
FZ_DEVICE=CUDA.
"""
from __future__ import annotations
import math, os, socket, subprocess, sys, time
import numpy as np
import pytest
from fusezap.tensor import Tensor
from fusezap.dtype import dtypes
from fusezap.device import Device, Buffer
from fusezap.jit import Jit
from fusezap.serve import kv_cache as K
from fusezap.serve import cuda as cu
from fusezap.serve.weights import bf16_to_f32, f32_to_bf16
from fusezap.serve.quant import fp8_e4m3_decode, fp8_e4m3_encode, quantize_weight, QuantLinear

HAS_CUDA = Device.available("CUDA")
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA GPU")
DEV = "CUDA"
MODEL, MODEL7 = "Qwen/Qwen2.5-0.5B-Instruct", os.environ.get("FZ_MODEL7") or "Qwen/Qwen2.5-7B-Instruct"

def _model_path(m: str = MODEL) -> str | None:
  try:
    from fusezap.serve.weights import resolve_model
    return resolve_model(m, allow_download=False)
  except Exception: return None
needs_model = pytest.mark.skipif(not HAS_CUDA or _model_path() is None, reason=f"needs CUDA and {MODEL} in the HF cache")

def _host(a: np.ndarray, dt) -> np.ndarray:
  """float32 values -> fusezap's host format for dt."""
  return f32_to_bf16(a) if dt == dtypes.bfloat16 else a.astype(np.float16 if dt == dtypes.float16 else np.float32)
def _f32(a: np.ndarray, dt) -> np.ndarray: return bf16_to_f32(a) if dt == dtypes.bfloat16 else np.asarray(a, np.float32)

DTS = [dtypes.float32, dtypes.float16, dtypes.bfloat16]

# ---------------------------------------------------------------- sources (no GPU needed)

def test_cuda_sources_render():
  for T in ("float", "unsigned short"):
    for f, nm in ((K.cu_kv_write, "a"), (K.cu_kv_write_rows, "b"), (K.cu_kv_gather, "c"), (K.cu_rows, "d")):
      src = f(nm, T)
      assert f'extern "C" __global__' in src and f" {nm}(" in src and T in src
  for dt in DTS: assert "cvt.rn.satfinite.e4m3x2.f32" in cu.render_rowquant("q", dt)

# ---------------------------------------------------------------- KV kernels == the numpy references

@needs_cuda
@pytest.mark.parametrize("dt", DTS, ids=["f32", "f16", "bf16"])
def test_kv_kernels_match_reference(dt):
  rng = np.random.default_rng(0)
  H, hd, bs, NB = 2, 16, 4, 12
  P0 = _host(rng.standard_normal(2 * NB * H * bs * hd).astype(np.float32), dt)
  # kv_write: 7 token rows (row stride xs, k at xk, v at xv), slots scattered, two padding rows (-1)
  n, xs = 7, 3 * H * hd + 5
  X = _host(rng.standard_normal(n * xs).astype(np.float32), dt)
  M = np.array([41, 3, -1, 17, 0, -1, 30], np.int32)
  p = dict(n=n, H=H, hd=hd, bs=bs, NB=NB, xs=xs, xk=H * hd, xv=2 * H * hd + 1, moff=0)
  pool, xb, mb = (Tensor(a, device=DEV, dtype=d).realize() for a, d in ((P0, dt), (X, dt), (M, dtypes.int32)))
  K.kv_write(K._buf(pool), K._buf(xb), K._buf(mb), *p.values())
  ref = P0.copy(); K._np_kv_write([ref, X, M], p)
  np.testing.assert_array_equal(pool.numpy(), ref)
  # kv_write_rows + kv_gather through PagedKVCache.update (B=2 rows, interleaved block tables)
  c = K.PagedKVCache(1, H, hd, NB, bs, dt, DEV, max_seq_len=16)
  c.ensure(5, 4); c.ensure(6, 8); c.ensure(5, 11)
  c.set_rows([5, 6])
  T = 3
  k_, v_ = (_host(rng.standard_normal((2, H, T, hd)).astype(np.float32), dt) for _ in range(2))
  Kt, Vt = c.update(0, Tensor(k_, device=DEV, dtype=dt), Tensor(v_, device=DEV, dtype=dt), 4, T)
  Kt, Vt = Kt.numpy(), Vt.numpy()
  for b, s in enumerate((5, 6)):
    np.testing.assert_array_equal(Kt[b, :, 4:7], k_[b]); np.testing.assert_array_equal(Vt[b, :, 4:7], v_[b])
    blocks = c.tables[s]
    pl = c.pools[0].numpy().reshape(2, NB, H, bs, hd)
    for j in range(16):
      want = 0 if j // bs >= len(blocks) else pl[:, blocks[j // bs], :, j % bs]
      np.testing.assert_array_equal(np.stack([Kt[b, :, j], Vt[b, :, j]]), np.broadcast_to(want, (2, H, hd)))
  # row gather
  R, D = 5, 24
  Xr = _host(rng.standard_normal(9 * D).astype(np.float32), dt)
  Mr = np.array([-1, -1, 8, 0, -1, 3, 3], np.int32)
  out = K.gather_rows(Tensor(Xr.reshape(9, D), device=DEV, dtype=dt), K._buf(Tensor(Mr, device=DEV, dtype=dtypes.int32).realize()), R, 2).numpy()
  ref = np.zeros((R, D), Xr.dtype); K._np_rows([ref, Xr, Mr], dict(R=R, D=D, moff=2))
  np.testing.assert_array_equal(out, ref)

@needs_cuda
@pytest.mark.parametrize("dt", [dtypes.float16, dtypes.bfloat16], ids=["f16", "bf16"])
def test_paged_attend_16bit_vs_float64(dt):
  """attend() on a 16-bit pool (bf16 has no numpy reference): the k/v rows land in their slots bit-exactly and the
  attention equals float64 attention over each token's own sequence."""
  rng = np.random.default_rng(1)
  Hq, Hkv, hd, bs, NB = 14, 2, 64, 16, 32
  c = K.PagedKVCache(1, Hkv, hd, NB, bs, dt, DEV, max_seq_len=NB * bs)
  lens = {0: 37, 1: 5, 2: 1}
  for s, n in lens.items(): c.ensure(s, n)
  toks = [(s, p) for s, n in lens.items() for p in range(n)]
  N = len(toks) + 2
  qkv = _host(rng.standard_normal((N, Hq + 2 * Hkv, hd)).astype(np.float32), dt)
  c.set_step(K.StepLayout(N, 3, c.max_blocks), Tensor(c.step_meta(N, 3, toks, []), device=DEV, dtype=dtypes.int32))
  o = _f32(c.attend(0, Tensor(qkv, device=DEV, dtype=dt), Hq, 1 / math.sqrt(hd)).numpy(), dt).reshape(N, Hq, hd)
  q = _f32(qkv, dt).astype(np.float64)
  pool = c.pools[0].numpy()
  for i, (s, p) in enumerate(toks):
    sl = c.slot(s, p)
    np.testing.assert_array_equal(pool[:, sl // bs, :, sl % bs], qkv[i, Hq:].reshape(2, Hkv, hd))
    rows = [j for j, (s2, p2) in enumerate(toks) if s2 == s and p2 <= p]
    Kh, Vh = q[rows, Hq:Hq + Hkv], q[rows, Hq + Hkv:]
    for h in range(Hq):
      sc = Kh[:, h // (Hq // Hkv)] @ q[i, h] / math.sqrt(hd)
      e = np.exp(sc - sc.max())
      np.testing.assert_allclose(o[i, h], (e / e.sum()) @ Vh[:, h // (Hq // Hkv)], rtol=2e-2, atol=2e-2)
  assert np.all(o[len(toks):] == 0)

def test_paged_flash_source_renders():
  from fusezap.device.cuda_kernels.attention import render_paged_flash
  for dt in (dtypes.float16, dtypes.bfloat16):
    for hd, G in ((64, 7), (128, 7), (128, 2)):
      src = render_paged_flash("pf", dt, hd, G)
      assert 'extern "C" __global__' in src and " pf(" in src and "mma.sync.aligned.m16n8k16" in src and "${" not in src

@needs_cuda
@pytest.mark.parametrize("dt", [dtypes.float16, dtypes.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("Hq,Hkv,hd", [(14, 2, 64), (28, 4, 128)])
def test_paged_prefill_flash(dt, Hq, Hkv, hd):
  """attend() on a prefill layout (the tensor-core paged prefill): prompt chunks after cached context, several sequences
  sharing 64-row tiles, padding rows. Equals float64 attention over each token's own sequence, is close to the split-K
  decode kernel, and a token's output is bit-identical whatever rows share its tile (row order, padding, other chunks)."""
  from fusezap.device.cuda_kernels.attention import use_paged_flash
  if not use_paged_flash(Device[DEV], dt, hd, (Hq + 2 * Hkv) * hd): pytest.skip("needs sm_80+")
  rng = np.random.default_rng(hd + Hq)
  bs, NB = 16, 64
  ctx, chunk = {0: 100, 1: 0, 2: 0, 3: 5, 4: 63}, {0: 70, 1: 37, 2: 1, 3: 86, 4: 2}
  c = K.PagedKVCache(1, Hkv, hd, NB, bs, dt, DEV, max_seq_len=256)
  qkvt = {s: _host(rng.standard_normal((ctx[s] + chunk[s], Hq + 2 * Hkv, hd)).astype(np.float32), dt) for s in ctx}
  for s in ctx: c.ensure(s, ctx[s] + chunk[s])
  def step(toks, N, prefill=True):
    qkv = np.zeros((N, Hq + 2 * Hkv, hd), qkvt[0].dtype)
    for i, t in enumerate(toks):
      if t is not None: qkv[i] = qkvt[t[0]][t[1]]
    real = [t for t in toks if t is not None]
    m = c.step_meta(N, 5, real, [])
    idx = [i for i, t in enumerate(toks) if t is not None]
    m2 = np.full_like(m, -1)   # place the real tokens at their (possibly gapped) rows
    L = K.StepLayout(N, 5, c.max_blocks, prefill)
    for j, i in enumerate(idx): m2[L.row + i], m2[L.pos + i], m2[L.slot + i] = m[L.row + j], m[L.pos + j], m[L.slot + j]
    m2[L.bt:] = m[L.bt:]
    c.set_step(L, Tensor(m2, device=DEV, dtype=dtypes.int32))
    o = _f32(c.attend(0, Tensor(qkv, device=DEV, dtype=dt), Hq, 1 / math.sqrt(hd)).numpy(), dt).reshape(N, Hq, hd)
    return {t: o[i] for i, t in enumerate(toks) if t is not None}, o
  step([(s, p) for s in ctx for p in range(ctx[s])], 256)                    # the cached context
  toks = [(s, p) for s in ctx for p in range(ctx[s], ctx[s] + chunk[s])]
  N = len(toks) + 7
  out, o = step(toks + [None] * 7, N)
  assert np.all(o[len(toks):] == 0)
  pool = _f32(c.pools[0].numpy(), dt).astype(np.float64)
  for (s, p), r in out.items():
    sl = [c.slot(s, j) for j in range(p + 1)]
    Kh, Vh = (np.stack([pool[a, x // bs, :, x % bs] for x in sl]) for a in (0, 1))
    q = _f32(qkvt[s][p, :Hq], dt).astype(np.float64)
    for h in range(Hq):
      sc = Kh[:, h // (Hq // Hkv)] @ q[h] / math.sqrt(hd)
      e = np.exp(sc - sc.max())
      np.testing.assert_allclose(r[h], (e / e.sum()) @ Vh[:, h // (Hq // Hkv)], rtol=2e-2, atol=2e-2)
  split, _ = step(toks + [None] * 7, N, prefill=False)
  for t in toks: np.testing.assert_allclose(out[t], split[t], rtol=2e-2, atol=2e-2)
  # other tile compositions: reversed sequence order with padding in front and between, one sequence's chunk alone
  rev = [t for s in reversed(list(ctx)) for t in toks if t[0] == s]
  alt = [None] * 13 + rev[:60] + [None] * 3 + rev[60:]
  for tt in (alt, [t for t in toks if t[0] == 3], [t for t in toks if t[0] == 0][5:]):
    got, _ = step(tt, len(tt))
    for t, r in got.items(): np.testing.assert_array_equal(r, out[t], err_msg=str(t))

# ---------------------------------------------------------------- tensor-core linears

@needs_cuda
@pytest.mark.parametrize("dt", [dtypes.float16, dtypes.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("M,N,K_", [(1, 1000, 896), (32, 4864, 896), (300, 896, 4864), (5, 136, 64)])
def test_linear16_vs_reference(dt, M, N, K_):
  rng = np.random.default_rng(M + N)
  x, w, b = (_host(rng.standard_normal(s).astype(np.float32) * sc, dt) for s, sc in (((M, K_), 1.0), ((N, K_), 0.05), ((N,), 0.1)))
  y = cu.linear(Tensor(x, device=DEV, dtype=dt), Tensor(w, device=DEV, dtype=dt), Tensor(b, device=DEV, dtype=dt))
  assert y.dtype == dt and y.shape == (M, N)
  ref = _f32(x, dt).astype(np.float64) @ _f32(w, dt).astype(np.float64).T
  # one rounding of the f32 sum + b to dt (cuBLAS's bias epilogue, as HF transformers / vLLM)
  r16 = _f32(_host((ref + _f32(b, dt)).astype(np.float32), dt), dt)
  np.testing.assert_allclose(_f32(y.numpy(), dt), r16, rtol=2e-2, atol=2e-2)
  # == the composed Tensor.linear (which rounds before the bias add) within a few 16-bit ulps
  comp = Tensor(x, device=DEV, dtype=dt).linear(Tensor(w, device=DEV, dtype=dt), Tensor(b, device=DEV, dtype=dt)).numpy()
  np.testing.assert_allclose(_f32(y.numpy(), dt), _f32(comp, dt), rtol=1e-2, atol=1e-2)

@needs_cuda
@pytest.mark.parametrize("dt", [dtypes.float16, dtypes.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("M,N,K_", [(1, 1152, 896), (32, 4608, 3584), (300, 1152, 896)])
def test_linear16_bias_single_rounding(dt, M, N, K_):
  """With a bias, cu.linear rounds once: its bits == round(f32 accumulator + b) computed on the host from the same GEMM's
  f32 output; and the old double rounding round(round(acc) + b) differs somewhere (the bias is large enough to matter)."""
  rng = np.random.default_rng(M * 7 + N)
  x, w, b = (_host(rng.standard_normal(s).astype(np.float32) * sc, dt) for s, sc in (((M, K_), 1.0), ((N, K_), 0.05), ((N,), 3.0)))
  X, W, B = Tensor(x, device=DEV, dtype=dt), Tensor(w, device=DEV, dtype=dt), Tensor(b, device=DEV, dtype=dt)
  acc = cu.linear(X, W, out_dtype=dtypes.float32).numpy()
  got = cu.linear(X, W, B)
  assert got.dtype == dt and got.shape == (M, N)
  want = _host((acc + _f32(b, dt)).astype(np.float32), dt)
  np.testing.assert_array_equal(got.numpy().view(np.uint16), want.view(np.uint16))
  double = _host(_f32(_host(acc, dt), dt) + _f32(b, dt), dt)
  assert np.any(double.view(np.uint16) != want.view(np.uint16))
  # out_dtype float32 with a bias: acc + b unrounded
  np.testing.assert_array_equal(cu.linear(X, W, B, dtypes.float32).numpy(), (acc + _f32(b, dt)).astype(np.float32))
  # == torch's F.linear (cuBLAS, bias in the epilogue: HF transformers' and vLLM's qkv projection), when torch has CUDA
  try:
    import torch
    tcuda = torch.cuda.is_available()
  except ImportError: tcuda = False
  if tcuda:
    tdt = torch.bfloat16 if dt == dtypes.bfloat16 else torch.float16
    def tt(a): return torch.from_numpy(_f32(a, dt)).to("cuda", tdt)
    th = torch.nn.functional.linear(tt(x), tt(w), tt(b)).float().cpu().numpy()
    eq, eq2 = np.mean(_f32(got.numpy(), dt) == th), np.mean(_f32(double, dt) == th)
    # the f32 sums may differ in order from cuBLAS's (split-K, gemv), so not every output need be equal
    assert eq > 0.97 and eq > eq2, f"bit-equal to torch F.linear on {100 * eq:.3f}% of outputs (double rounding: {100 * eq2:.3f}%)"
  # jitted (as in the model's steps) == eager
  jf = Jit(lambda x_: cu.linear(x_ * 1.0, W, B).float())
  for _ in range(3): np.testing.assert_array_equal(jf(X).numpy(), cu.linear(X * 1.0, W, B).float().numpy())

@needs_cuda
@pytest.mark.parametrize("dt", [dtypes.float16, dtypes.bfloat16], ids=["f16", "bf16"])
def test_linear16_f32_out_unrounded(dt):
  """out_dtype=float32 (the lm_head's logits) returns the f32 accumulator, not a 16-bit rounding of it."""
  rng = np.random.default_rng(11)
  M, N, K_ = 7, 1000, 896
  x, w = (_host(rng.standard_normal(s).astype(np.float32) * sc, dt) for s, sc in (((M, K_), 1.0), ((N, K_), 0.05)))
  y = cu.linear(Tensor(x, device=DEV, dtype=dt), Tensor(w, device=DEV, dtype=dt), out_dtype=dtypes.float32)
  assert y.dtype == dtypes.float32 and y.shape == (M, N)
  got, ref = y.numpy(), _f32(x, dt).astype(np.float64) @ _f32(w, dt).astype(np.float64).T
  np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)
  assert np.mean(np.abs(got - ref)) < 0.1 * np.mean(np.abs(_f32(_host(ref.astype(np.float32), dt), dt) - ref))   # far below 16-bit rounding

@needs_cuda
def test_linear16_jit_replay():
  rng = np.random.default_rng(3)
  dt, M, N, K_ = dtypes.bfloat16, 16, 512, 256
  w = Tensor(f32_to_bf16(rng.standard_normal((N, K_)).astype(np.float32) * 0.05), device=DEV, dtype=dt).realize()
  calls = []
  def step(x: Tensor) -> Tensor:
    calls.append(1)
    return (cu.linear(x * 2.0, w) + 1.0).float()
  jf = Jit(step)
  for _ in range(4):
    x = f32_to_bf16(rng.standard_normal((M, K_)).astype(np.float32))
    got = jf(Tensor(x, device=DEV, dtype=dt)).numpy()
    want = (cu.linear(Tensor(x, device=DEV, dtype=dt) * 2.0, w) + 1.0).float().numpy()
    np.testing.assert_array_equal(got, want)
  assert len(calls) == 2

def test_rmsnorm_source_renders():
  for dt in (dtypes.float16, dtypes.bfloat16):
    for res in (False, True):
      src = cu.render_rmsnorm("n", dt, 3584, 1e-6, res)
      assert 'extern "C" __global__' in src and " n(" in src and ("XO" in src) == res and "uint4" in src

def _composed_norm(x: Tensor, w: Tensor, eps: float) -> Tensor:   # Llama._rmsnorm
  xf = x.float()
  return w * (xf * ((xf * xf).mean(-1, keepdim=True) + eps).rsqrt()).cast(x.dtype)

@needs_cuda
@pytest.mark.parametrize("dt", [dtypes.float16, dtypes.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("M,K_", [(1, 3584), (32, 3584), (5, 896), (3, 72), (2, 4096 + 8)])
def test_rmsnorm_fused_vs_composed(dt, M, K_):
  """cu.rmsnorm: the residual stream x + d is bit-identical to the composed add; the norm uses the composed kernels'
  ops and roundings, so it differs only where the f32 sum's order moves a 16-bit rounding (<= 1 ulp, rarely); row 0
  alone == row 0 of the batch; a jit replay (CUDA graph) == the eager call."""
  rng = np.random.default_rng(M * K_)
  x, d, w = (_host(rng.standard_normal(s).astype(np.float32) * sc, dt) for s, sc in (((M, K_), 2.0), ((M, K_), 1.0), ((K_,), 0.5)))
  X, D, W = (Tensor(a, device=DEV, dtype=dt).realize() for a in (x, d, w))
  xo, h = cu.rmsnorm(X, W, 1e-6, D)
  assert xo.shape == h.shape == (M, K_) and xo.dtype == h.dtype == dt
  xs = (X + D).realize()
  np.testing.assert_array_equal(xo.numpy(), xs.numpy())
  for got in (h, cu.rmsnorm(xs, W, 1e-6)[1]):
    g, c = _f32(got.numpy(), dt), _f32(_composed_norm(xs, W, 1e-6).numpy(), dt)
    np.testing.assert_allclose(g, c, rtol=2 ** -6, atol=1e-6)
    assert np.mean(g == c) >= 0.99
    xf = _f32(xs.numpy(), dt).astype(np.float64)
    ref = _f32(w, dt) * xf / np.sqrt((xf * xf).mean(-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(g, ref, rtol=2e-2, atol=2e-2)
  np.testing.assert_array_equal(cu.rmsnorm(X[:1], W, 1e-6, D[:1])[1].numpy()[0], h.numpy()[0])
  jf = Jit(lambda a, b: [t.float() for t in cu.rmsnorm(a * 1.0, W, 1e-6, b)])
  for _ in range(4):
    x2, d2 = (_host(rng.standard_normal((M, K_)).astype(np.float32), dt) for _ in range(2))
    A, B = Tensor(x2, device=DEV, dtype=dt), Tensor(d2, device=DEV, dtype=dt)
    got = [t.numpy() for t in jf(A, B)]
    want = [t.float().numpy() for t in cu.rmsnorm(A * 1.0, W, 1e-6, B)]
    for a, b in zip(got, want): np.testing.assert_array_equal(a, b)

def test_argmax_source_renders():
  src = cu.render_argmax("am", 152064)
  assert 'extern "C" __global__' in src and " am(" in src and "152064" in src and "__shfl_xor_sync" in src

@needs_cuda
@pytest.mark.parametrize("R,V", [(1, 152064), (32, 151936), (33, 1000), (3, 7), (2, 1025)])
def test_argmax_rows_vs_numpy(R, V):
  """cu.argmax_rows == np.argmax per row, including ties (first index), -inf rows, rows with NaNs (the first NaN) and a
  jit replay (CUDA graph)."""
  rng = np.random.default_rng(R * V)
  x = rng.standard_normal((R, V)).astype(np.float32)
  x[0, [V // 3, V - 1, V // 2]] = 50.0                           # a three-way tie
  if R > 1: x[1] = -np.inf
  if R > 2: x[2, [V - 1, 1 % V]] = np.nan
  if R > 3: x[3] = np.round(x[3])                                # many ties at the max
  got = cu.argmax_rows(Tensor(x, device=DEV)).numpy()
  assert got.dtype == np.int32 and got.shape == (R,)
  np.testing.assert_array_equal(got, np.argmax(x, -1))
  jf = Jit(lambda a: cu.argmax_rows(a * 1.0))
  for i in range(4):
    y = rng.standard_normal((R, V)).astype(np.float32) * (i + 1)
    np.testing.assert_array_equal(jf(Tensor(y, device=DEV)).numpy(), np.argmax(y, -1))

def test_silu_rope_sources_render():
  for dt in (dtypes.float16, dtypes.bfloat16):
    src = cu.render_silu_mul("sm", dt, 18944)
    assert 'extern "C" __global__' in src and " sm(" in src and "exp2f" in src and "uint4" in src
    for bias in (False, True):
      for yf32 in (False, True):
        src = cu.render_rope_qkv("rq", dt, 36, 32, 128, bias, yf32)
        assert 'extern "C" __global__' in src and " rq(" in src and (" B," in src) == bias and ("const float* __restrict__ Y" in src) == yf32

def test_rope_kvw_sources_render():
  """I60: rope_qkv's kvw form also stores the k / v heads into the paged pool (the slot from the step metadata, padding
  rows skipped); its buffers come in the launch order (O, P, Y, CS, SN, [B], M), then the ints (n, moff, bs, NB)."""
  for dt in (dtypes.float16, dtypes.bfloat16):
    for bias in (False, True):
      for yf32 in (False, True):
        src = cu.render_rope_qkv("rk", dt, 40, 36, 128, bias, yf32, (32, 4))
        sig = src[src.index(" rk("):src.index(") {", src.index(" rk("))]
        order = [sig.index(s) for s in ("O,", " P,", " Y,", " CS,", " SN,", " M,", "int n,", "int moff,", "int bs,", "int NB")]
        assert order == sorted(order) and (" B," in sig) == bias, sig
        assert "M[moff + r]" in src and "if (slot >= 0)" in src and "hh >= 32" in src and "kv * 4" in src
        assert "M[moff" not in cu.render_rope_qkv("rq", dt, 40, 36, 128, bias, yf32)
  with pytest.raises(AssertionError): cu.render_rope_qkv("rk", dtypes.bfloat16, 40, 36, 128, False, False, (32, 2))
  import re, shutil
  if shutil.which("clang++") is not None:   # the CUDA C++ parses (stand-ins for the CUDA built-ins; NVRTC runs on GPUs only)
    pre = ("#define __global__\n#define __device__\n#define __forceinline__ inline\n#define __launch_bounds__(x)\n#define __restrict__\n"
           "#define FZ_ASM(...) 0\nstruct fz_dim3 { unsigned x, y, z; }; extern fz_dim3 threadIdx, blockIdx;\n"
           "struct uint4 { unsigned x, y, z, w; }; struct float4 { float x, y, z, w; };\nfloat __uint_as_float(unsigned); unsigned __float_as_uint(float);\n")
    for dt in (dtypes.float16, dtypes.bfloat16):
      s = re.sub(r"\basm\s+volatile\s*\(|\basm\s*\(", "FZ_ASM(", cu.render_rope_qkv("rk", dt, 40, 36, 128, True, True, (32, 4)))
      p = subprocess.run(["clang++", "-fsyntax-only", "-x", "c++", "-std=c++17", "-"], input=pre + s, text=True, capture_output=True)
      assert p.returncode == 0, p.stderr[:3000]
  assert cu.fuse_rows() == int(os.environ.get("FZ_FUSE_ROWS", "") or cu.FUSE_ROWS)

@needs_cuda
@pytest.mark.parametrize("N,Hq,Hkv,bias,yf32", [(1, 32, 4, False, False), (32, 32, 4, False, False), (5, 28, 4, True, True), (300, 40, 8, True, False)])
def test_rope_qkv_kvw_vs_kv_write(N, Hq, Hkv, bias, yf32):
  """I60: rope_qkv(kvw=) gives the same q / k / v rows as rope_qkv alone, and leaves the pool exactly as kv_write of
  those rows does (padding rows, slot -1, untouched), eager and in a jit replay (CUDA graph)."""
  from fusezap.serve.models.llama import LlamaConfig, rope_tables
  from fusezap.serve.weights import to_dtype
  rng = np.random.default_rng(N + Hq)
  dt, hd, bs, NB = dtypes.bfloat16, 128, 16, 64
  H, h2 = Hq + 2 * Hkv, hd // 2
  ct, st = rope_tables(LlamaConfig(Hq * hd, 4 * Hq * hd, 1, Hq, Hkv, hd, 100, rope_theta=5e5), 4096)
  def inputs():
    pos = rng.integers(0, 4096, N)
    CS, SN = (Tensor(to_dtype(t[pos], dt), device=DEV, dtype=dt).realize() for t in (ct, st))
    a = rng.standard_normal((N, H * hd)).astype(np.float32) * 2.0
    Y = (Tensor(a, device=DEV, dtype=dtypes.float32) if yf32 else Tensor(_host(a, dt), device=DEV, dtype=dt)).realize()
    slots = rng.permutation(NB * bs)[:N].astype(np.int32)
    slots[rng.random(N) < 0.2] = -1
    meta = Tensor(np.concatenate([np.full(7, -1, np.int32), slots]), device=DEV, dtype=dtypes.int32).realize()
    return Y, CS, SN, meta
  B = Tensor(_host(rng.standard_normal(H * hd).astype(np.float32), dt), device=DEV, dtype=dt).realize() if bias else None
  pool0 = _host(rng.standard_normal(2 * NB * Hkv * bs * hd).astype(np.float32), dt)
  def ref(Y, CS, SN, meta):
    pool = Tensor(pool0, device=DEV, dtype=dt).realize()
    o = cu.rope_qkv(Y, B, CS, SN, H, Hq + Hkv, hd).realize()
    K.kv_write(K._buf(pool), K._buf(o), K._buf(meta), N, Hkv, hd, bs, NB, H * hd, Hq * hd, (Hq + Hkv) * hd, 7)
    return _bits(o), _bits(pool)
  def fused(Y, CS, SN, meta, pool):
    return cu.rope_qkv(Y, B, CS, SN, H, Hq + Hkv, hd, K.KVDest(K._buf(pool), K._buf(meta), 7, Hkv, hd, bs, NB))
  args = inputs()
  pool = Tensor(pool0, device=DEV, dtype=dt).realize()
  o = fused(*args, pool)
  ro, rp = ref(*args)
  assert np.array_equal(_bits(o), ro) and np.array_equal(_bits(pool), rp)
  jf = Jit(lambda y, c, s, m, p: (fused(y, c, s, m, p).float(), p))
  for _ in range(3):
    args = inputs()
    pool = Tensor(pool0, device=DEV, dtype=dt).realize()
    o, p2 = jf(*args, pool)
    ro, rp = ref(*args)
    assert np.array_equal(to_dtype(o.numpy(), dt), ro) and np.array_equal(_bits(p2), rp)

def test_step_inputs_source_renders():
  """Speed run 4: the device-built decode inputs (cuda.DeviceSteps) and where they apply."""
  src = cu.render_step_inputs("si", 3584, 64)
  assert 'extern "C" __global__' in src and " si(" in src and "IDS[s]" in src and "PK[3 * N + i]" in src
  assert cu.DeviceSteps.ok(3584, 128, dtypes.bfloat16) and cu.DeviceSteps.ok(896, 64, dtypes.float16)
  assert not cu.DeviceSteps.ok(3584, 128, dtypes.float32) and not cu.DeviceSteps.ok(3584, 8, dtypes.bfloat16)

@needs_cuda
def test_device_step_inputs_match_host():
  """DeviceSteps builds x / cos / sin / meta bit-identical to the host-built inputs, with tokens from the host and from a
  device ids buffer (src rows), and its async ids read-back returns the device values."""
  from fusezap.serve.weights import to_dtype
  from fusezap.tensor import Tensor
  rng = np.random.default_rng(0)
  V, D, L, H2, dt = 1000, 64, 300, 16, dtypes.bfloat16
  E = to_dtype(rng.standard_normal((V, D)).astype(np.float32), dt)
  C, S = to_dtype(rng.standard_normal((L, H2)).astype(np.float32), dt), to_dtype(rng.standard_normal((L, H2)).astype(np.float32), dt)
  ds = cu.DeviceSteps("CUDA", E, C, S, dt)
  prev = Tensor(np.array([7, 999, 3, 0, 512], np.int32), device="CUDA", dtype=dtypes.int32).realize()
  for _ in range(6):   # more steps than the pinned ring has slots
    N = 8
    tok = rng.integers(0, V, N); src = np.where(rng.random(N) < 0.5, rng.integers(0, 5, N), -1); pos = rng.integers(0, L, N)
    meta = rng.integers(-1, 1000, 123).astype(np.int32)
    x, c, s, m = ds.inputs(tok, src, pos, meta, cu._buf(prev))
    want = np.where(src >= 0, np.array([7, 999, 3, 0, 512])[np.maximum(src, 0)], tok)
    assert np.array_equal(_bits(x), E[want])
    assert np.array_equal(_bits(c), C[pos]) and np.array_equal(_bits(s), S[pos]) and np.array_equal(m.numpy(), meta)
    rb = ds.readback(cu._buf(prev))
    assert list(rb.get()) == [7, 999, 3, 0, 512]

@needs_cuda
@pytest.mark.parametrize("N,K", [(3584, 3584), (3584, 18944), (4608, 3584), (896, 4864)], ids=["o", "down", "qkv", "05b_down"])
@pytest.mark.parametrize("out", [None, dtypes.float32], ids=["bf16", "f32"])
def test_linear_mixed_rows_equal_decode(N, K, out):
  """Speed run 4: in a mixed forward (prompt chunk rows 0 .. C-1, decode rows C ..), cuda.linear(redo_rows=(C, S)) gives
  the decode rows exactly the bits of the decode forward's GEMM on those rows alone (M = S, and at M = 1 per row): the
  prefill tile matches the decode kernel without split-K, and split-K shapes are recomputed with the decode kernel."""
  if tuple(Device[DEV].cc) != (9, 0): pytest.skip("sm_90 (swap-AB decode GEMM)")
  rng = np.random.default_rng(1)
  C, S = 128, 32
  dt = dtypes.bfloat16
  x = Tensor(rng.standard_normal((C + S, K)).astype(np.float32) * 0.5, device=DEV).cast(dt).realize()
  w = Tensor(rng.standard_normal((N, K)).astype(np.float32) * 0.02, device=DEV).cast(dt).realize()
  full = cu.linear(x, w, None, out, redo_rows=(C, S)).float().numpy()
  mixed = full[C:]
  pre = cu.linear(x[:C].contiguous().realize(), w, None, out).float().numpy()   # prompt rows: the prefill GEMM's bits
  assert np.array_equal(full[:C], pre)
  dec = cu.linear(x[C:].contiguous().realize(), w, None, out).float().numpy()
  assert np.array_equal(mixed, dec)
  one = cu.linear(x[C + 3:C + 4].contiguous().realize(), w, None, out).float().numpy()
  assert np.array_equal(mixed[3:4], one)

def _bits(t) -> np.ndarray:
  """bf16 tensor -> its uint16 bit patterns (fusezap's host format for bf16)."""
  from fusezap.serve.weights import to_dtype
  return to_dtype(t.float().numpy(), dtypes.bfloat16)

@needs_cuda
@pytest.mark.parametrize("dt", [dtypes.float16, dtypes.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("M,I", [(1, 18944), (32, 4864), (5, 16), (2048, 18944)])
def test_silu_mul_vs_composed(dt, M, I):
  """cu.silu_mul is bit-identical to the composed g.silu() * u (inf / NaN / large inputs included), row 0 alone == row 0
  of the batch, and a jit replay (CUDA graph) == the eager call."""
  rng = np.random.default_rng(M * I)
  a = rng.standard_normal((M, 2 * I)).astype(np.float32) * 4.0
  a[0, :4] = [np.inf, -np.inf, np.nan, 80.0]
  a[0, I:I + 4] = [1.0, 1.0, 1.0, -3.0]
  GU = Tensor(_host(a, dt), device=DEV, dtype=dt).realize()
  got = cu.silu_mul(GU)
  assert got.shape == (M, I) and got.dtype == dt
  want = (GU.shrink((None, (0, I))).silu() * GU.shrink((None, (I, 2 * I)))).numpy()
  np.testing.assert_array_equal(got.numpy().view(np.uint16), want.view(np.uint16))
  np.testing.assert_array_equal(cu.silu_mul(GU[:1]).numpy()[0], got.numpy()[0])
  jf = Jit(lambda x: cu.silu_mul(x * 1.0).float())
  for _ in range(3):
    X = Tensor(_host(rng.standard_normal((M, 2 * I)).astype(np.float32), dt), device=DEV, dtype=dt)
    np.testing.assert_array_equal(jf(X).numpy(), cu.silu_mul(X * 1.0).float().numpy())

@needs_cuda
@pytest.mark.parametrize("dt", [dtypes.float16, dtypes.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("N,Hq,Hkv,hd,bias", [(1, 28, 4, 128, True), (32, 28, 4, 128, True), (2048, 28, 4, 128, True),
                                              (7, 14, 2, 64, False), (3, 40, 8, 128, True)])
@pytest.mark.parametrize("yf32", [False, True], ids=["y16", "y32"])
def test_rope_qkv_vs_composed(dt, N, Hq, Hkv, hd, bias, yf32):
  """cu.rope_qkv is bit-identical to Llama.forward_tokens' composed bias add + RoPE + cat (the engine's cos / sin rows
  in the model dtype), row 0 alone == row 0 of the batch, and a jit replay (CUDA graph) == the eager call. yf32: y is
  the GEMM's f32 accumulator and the composed reference is cu.linear's single rounding, (y + float(b)).cast(dt)."""
  from fusezap.serve.models.llama import LlamaConfig, rope_tables
  from fusezap.serve.weights import to_dtype
  rng = np.random.default_rng(N * Hq + hd)
  H, h2 = Hq + 2 * Hkv, hd // 2
  cfg = LlamaConfig(Hq * hd, 4 * Hq * hd, 1, Hq, Hkv, hd, 100, rope_theta=1e6)
  ct, st = rope_tables(cfg, 4096)
  pos = rng.integers(0, 4096, N)
  CS, SN = (Tensor(to_dtype(t[pos], dt), device=DEV, dtype=dt).realize() for t in (ct, st))
  def mk_y(scale: float) -> Tensor:
    a = rng.standard_normal((N, H * hd)).astype(np.float32) * scale
    return Tensor(a, device=DEV, dtype=dtypes.float32) if yf32 else Tensor(_host(a, dt), device=DEV, dtype=dt)
  Y = mk_y(3.0).realize()
  B = Tensor(_host(rng.standard_normal(H * hd).astype(np.float32), dt), device=DEV, dtype=dt).realize() if bias else None
  def composed(y, cs, sn):
    n = y.shape[0]
    if yf32: q = ((y + B.cast(dtypes.float32)) if B is not None else y).cast(dt).reshape(n, H, hd)
    else: q = (y + B if B is not None else y).reshape(n, H, hd)
    cs, sn = cs.reshape(n, 1, h2), sn.reshape(n, 1, h2)
    qk, v = q.shrink((None, (0, Hq + Hkv), None)), q.shrink((None, (Hq + Hkv, H), None))
    x1, x2 = qk.shrink((None, None, (0, h2))), qk.shrink((None, None, (h2, hd)))
    return (x1 * cs + (-x2) * sn).cat(x2 * cs + x1 * sn, axis=-1).cat(v, axis=1)
  got = cu.rope_qkv(Y, B, CS, SN, H, Hq + Hkv, hd)
  assert got.shape == (N, H, hd) and got.dtype == dt
  np.testing.assert_array_equal(got.numpy().view(np.uint16), composed(Y, CS, SN).numpy().view(np.uint16))
  np.testing.assert_array_equal(cu.rope_qkv(Y[:1], B, CS[:1], SN[:1], H, Hq + Hkv, hd).numpy()[0], got.numpy()[0])
  jf = Jit(lambda y, c, s: cu.rope_qkv(y * 1.0, B, c, s, H, Hq + Hkv, hd).float())
  for _ in range(3):
    Y2 = mk_y(1.0)
    p2 = rng.integers(0, 4096, N)
    C2, S2 = (Tensor(to_dtype(t[p2], dt), device=DEV, dtype=dt) for t in (ct, st))
    np.testing.assert_array_equal(jf(Y2, C2, S2).numpy(), cu.rope_qkv(Y2 * 1.0, B, C2, S2, H, Hq + Hkv, hd).float().numpy())

def _graph_env(monkeypatch, on: bool) -> None:
  from fusezap.helpers import getenv
  monkeypatch.setenv("FZ_GRAPH", "1" if on else "0")
  getenv.cache_clear()

@needs_cuda
@pytest.mark.parametrize("swap_pool", [False, True], ids=["same_pool", "new_pool"])
def test_jit_cuda_graph_replay(monkeypatch, swap_pool):
  """A var-free CUDA jit replays as one CUDA graph: read-only inputs are staged, the in-place pool input is written in
  place, every output is a fresh buffer, and the results equal launch-by-launch replay (FZ_GRAPH=0) bit for bit. A pool
  that arrives as a different buffer rebuilds the graph."""
  rng = np.random.default_rng(5)
  dt, H, hd, bs, NB, n = dtypes.bfloat16, 2, 16, 4, 8, 5
  xs = 2 * H * hd
  w = Tensor(f32_to_bf16(rng.standard_normal((64, xs)).astype(np.float32) * 0.1), device=DEV, dtype=dt).realize()
  def step(x: Tensor, meta: Tensor, pool: Tensor) -> list[Tensor]:
    K.kv_write(K._buf(pool), K._buf(x), K._buf(meta), n, H, hd, bs, NB, xs, 0, H * hd, 0)
    y = cu.linear(x * 0.5, w)
    return [(y + 1.0).float(), y.sum(1).float()]
  def run(on: bool) -> tuple[list, list]:
    _graph_env(monkeypatch, on)
    r = np.random.default_rng(9)
    pool = Tensor.zeros(2 * NB * H * bs * hd, dtype=dt, device=DEV).contiguous().realize()
    jf, outs, pools = Jit(step), [], []
    for i in range(7):
      if swap_pool and i == 5: pool = Tensor(pool.numpy(), device=DEV, dtype=dt).realize()
      x = Tensor(f32_to_bf16(r.standard_normal((n, xs)).astype(np.float32)), device=DEV, dtype=dt)
      meta = Tensor(r.permutation(NB * bs)[:n].astype(np.int32), device=DEV, dtype=dtypes.int32)
      a, b = jf(x, meta, pool)
      outs.append((a.numpy(), b.numpy()))
      pools.append(pool.numpy())
    return outs, pools, jf
  try:
    (o1, p1, j1), (o0, p0, j0) = run(True), run(False)
  finally: _graph_env(monkeypatch, True)
  assert j1.captured.graph is not None and j0.captured.graph is None
  for (a1, b1), (a0, b0) in zip(o1, o0): np.testing.assert_array_equal(a1, a0); np.testing.assert_array_equal(b1, b0)
  for q1, q0 in zip(p1, p0): np.testing.assert_array_equal(q1, q0)
  assert o1[-1][0] is not o1[-2][0] and not np.array_equal(o1[-1][0], o1[-2][0])

# ---------------------------------------------------------------- FP8

def _fp8_ok() -> bool: return HAS_CUDA and cu.fp8_ok(DEV, 16)
needs_fp8 = pytest.mark.skipif(not _fp8_ok(), reason="needs CUDA sm_89+ (fp8 tensor cores)")

@needs_fp8
@pytest.mark.parametrize("dt", DTS, ids=["f32", "f16", "bf16"])
def test_rowquant_matches_host_encoder(dt):
  rng = np.random.default_rng(4)
  x = rng.standard_normal((7, 96)).astype(np.float32) * np.array([1, 1e-3, 300, 0, 1, 5e3, 2e-7], np.float32)[:, None]
  xh = _host(x, dt)
  q, s = cu.quantize_rows_fp8(Tensor(xh, device=DEV, dtype=dt))
  q, s = q.numpy(), s.numpy()[:, 0]
  xf = _f32(xh, dt)
  amax = np.abs(xf).max(1)
  np.testing.assert_array_equal(s, np.where(amax > 0, amax / np.float32(448), np.float32(1)).astype(np.float32))
  np.testing.assert_array_equal(q, fp8_e4m3_encode(xf / s[:, None]))

@needs_fp8
@pytest.mark.parametrize("M", [1, 32, 257])
def test_fp8_linear_vs_dequantized(M):
  rng = np.random.default_rng(M)
  N, K_ = 640, 896
  x = (rng.standard_normal((M, K_)) * rng.uniform(0.1, 10, (M, 1))).astype(np.float32)
  w = (rng.standard_normal((N, K_)) * 0.05).astype(np.float32)
  b = rng.standard_normal(N).astype(np.float32) * 0.1
  qw = quantize_weight(w, "fp8")
  ql = QuantLinear(qw, dtypes.bfloat16, DEV)
  assert ql.ws is not None
  xt = Tensor(f32_to_bf16(x), device=DEV, dtype=dtypes.bfloat16)
  y = bf16_to_f32(ql.apply(xt, Tensor(f32_to_bf16(b), device=DEV, dtype=dtypes.bfloat16)).numpy())
  # reference: the same per-row activation quantization, dequantized, in float64
  xf = bf16_to_f32(f32_to_bf16(x))
  sx = np.abs(xf).max(1, keepdims=True) / np.float32(448)
  xq = fp8_e4m3_decode(fp8_e4m3_encode(xf / sx)).astype(np.float64) * sx
  wd = fp8_e4m3_decode(qw.q).astype(np.float64) * float(qw.scale[0, 0])
  ref = xq @ wd.T
  sc = np.abs(ref).max()
  np.testing.assert_allclose(y - bf16_to_f32(f32_to_bf16(b)), ref, rtol=2e-2, atol=2e-3 * sc)
  # and close to the unquantized product (fp8 W8A8 error)
  full = xf.astype(np.float64) @ w.astype(np.float64).T
  assert np.abs(y - b - full).max() < 0.08 * np.abs(full).max()

@needs_fp8
def test_device_fp8_weight_quant_matches_host():
  rng = np.random.default_rng(5)
  w = (rng.standard_normal((384, 512)) * 0.02).astype(np.float32)
  wb = f32_to_bf16(w)
  q, s = cu.quantize_fp8_weight(Tensor(wb, device=DEV, dtype=dtypes.bfloat16))
  host = quantize_weight(bf16_to_f32(wb), "fp8")
  assert float(s.numpy()[0]) == float(host.scale[0, 0])
  np.testing.assert_array_equal(q.numpy(), host.q)
  ql = QuantLinear.from_device_fp8(q, s, dtypes.bfloat16)
  np.testing.assert_array_equal(ql.host().q, host.q)
  np.testing.assert_allclose(ql.host().scale, host.scale, rtol=1e-7)

@needs_fp8
def test_fp8_linear_jit_and_batch_invariance():
  """Rows are independent: a row's output does not depend on the other rows of the step (per-row activation scales)."""
  rng = np.random.default_rng(6)
  ql = QuantLinear.from_float((rng.standard_normal((256, 512)) * 0.05).astype(np.float32), "fp8", dtypes.bfloat16, DEV)
  jf = Jit(lambda x: ql.apply(x).float())
  rows = f32_to_bf16(rng.standard_normal((32, 512)).astype(np.float32))
  outs = []
  for k in range(4):
    x = rows.copy()
    if k: x[1:] = f32_to_bf16(rng.standard_normal((31, 512)).astype(np.float32) * 10 ** k)   # other rows change scale
    outs.append(jf(Tensor(x, device=DEV, dtype=dtypes.bfloat16)).numpy()[0])
  for o in outs[1:]: np.testing.assert_array_equal(o, outs[0])

# ---------------------------------------------------------------- the model and engine on CUDA (Qwen2.5-0.5B)

@pytest.fixture(scope="module")
def hf():
  torch = pytest.importorskip("torch")
  tr = pytest.importorskip("transformers")
  return tr.AutoModelForCausalLM.from_pretrained(_model_path(), dtype=torch.float32).eval()

@pytest.fixture(scope="module")
def prompt_ids():
  from fusezap.serve.tokenizer import Tokenizer
  return Tokenizer.from_pretrained(_model_path()).apply_chat_template([{"role": "user", "content": "Write a short poem about the sea."}], True)

@pytest.fixture(scope="module")
def m16():
  from fusezap.serve.models import load_model
  return load_model(MODEL, dtype=dtypes.bfloat16, device=DEV, max_len=512, prefill_chunk=128)

def _teacher(hfm, ids: list[int], n: int) -> tuple[list[int], np.ndarray]:
  import torch
  with torch.no_grad():
    out = hfm.generate(torch.tensor([ids]), max_new_tokens=n, do_sample=False, num_beams=1)[0].tolist()
    return out, hfm(torch.tensor([out])).logits[0].numpy()

def _top1(model, seq: list[int], start: int) -> np.ndarray:
  x = Tensor(model.embed(np.array([seq])), device=model.device, dtype=model.dtype)
  lg = model.forward(x, model.cache.bind(model.cache.state()), 0, len(seq), last_only=False).float().numpy()[0]
  return lg.argmax(-1)[start - 1:-1]

def _hf_bf16_top1(seq: list[int], start: int) -> np.ndarray:
  """HF transformers itself in bf16 (torch, on the GPU when torch has CUDA): the baseline for what bf16 costs."""
  import torch, transformers as tr
  d = "cuda" if torch.cuda.is_available() else "cpu"
  m = tr.AutoModelForCausalLM.from_pretrained(_model_path(), dtype=torch.bfloat16).to(d).eval()
  with torch.no_grad(): lg = m(torch.tensor([seq], device=d)).logits[0].float().cpu().numpy()
  del m
  return lg.argmax(-1)[start - 1:-1]

@needs_model
def test_lm_head_f32_logits(m16, prompt_ids):
  """On CUDA bf16 the lm_head writes float32 logits (== the final hidden @ lm_head^T in float64, to f32 accumulation
  error), not bf16-rounded ones: bf16 logits (~0.1 apart near 20) turn near-ties into top-1 flips."""
  x = Tensor(m16.embed(np.array([prompt_ids])), device=DEV, dtype=dtypes.bfloat16)
  hs: list[Tensor] = []
  lg = m16.forward(x, m16.cache.bind(m16.cache.state()), 0, len(prompt_ids), last_only=False, hiddens=hs)
  assert lg.dtype == dtypes.float32
  got = lg.numpy()[0]
  h = bf16_to_f32(hs[-1].numpy()[0]).astype(np.float64)
  ref = h @ bf16_to_f32(m16.lm_head.numpy()).astype(np.float64).T
  assert np.abs(got - ref).max() < 2e-3 * np.abs(ref).max(), np.abs(got - ref).max()
  assert np.any(got.view(np.uint32) & 0xffff), "logits are bf16-representable: the lm_head output was rounded"

@needs_model
def test_bf16_top1_vs_hf(hf, m16, prompt_ids):
  """0.5B regression check (the SPEC §0 >= 99% gate is test_7b_bf16_top1_vs_hf_spec): bf16 (flash prefill attention +
  tensor-core linears, f32 logits) vs the f32 HF reference, teacher-forced over a 128-token greedy continuation. At
  0.5B HF transformers' own bf16 is below 99% too (printed baseline), so this asserts fusezap stays within 2 points of
  it and >= 0.95."""
  seq, R = _teacher(hf, prompt_ids, 128)
  assert m16.cuda16
  ref = R.argmax(-1)[len(prompt_ids) - 1:-1]
  agree = float((_top1(m16, seq, len(prompt_ids)) == ref).mean())
  base = float((_hf_bf16_top1(seq, len(prompt_ids)) == ref).mean())
  print(f"bf16 top-1 agreement vs HF f32 over {len(ref)} positions: fusezap {agree:.4f}, HF transformers bf16 {base:.4f}")
  assert agree >= min(base - 0.02, 0.99) and agree >= 0.95, (agree, base)

@needs_model
def test_bf16_jitted_prefill_decode_vs_forward(m16, prompt_ids):
  """Llama.prefill (jitted, composed attention, symbolic start) + decode replays == the flash forward's last logits."""
  cont = [785, 9396, 11, 264, 12767, 323]
  m16.reset()
  lgs = [m16.prefill([prompt_ids])[0]] + [m16.decode([[t]])[0] for t in cont]
  seq = prompt_ids + cont
  x = Tensor(m16.embed(np.array([seq])), device=DEV, dtype=dtypes.bfloat16)
  ref = m16.forward(x, m16.cache.bind(m16.cache.state()), 0, len(seq), last_only=False).float().numpy()[0]
  for i, lg in enumerate(lgs):
    r = ref[len(prompt_ids) - 1 + i]
    assert r[lg.argmax()] >= r.max() - 0.02 * abs(r.max()) and np.abs(lg - r).max() < 0.15 * np.abs(r).max(), f"step {i}"

@needs_model
def test_fp8_model_top1_vs_bf16(hf, prompt_ids):
  """fp8 W8A8 (device-quantized e4m3 weights, per-row e4m3 activations, the e4m3 GEMM) agrees with HF's top-1 at >= 75%
  of teacher-forced positions (0.5B is the sensitive case; 7B perplexity is in docs/quant.md)."""
  if not _fp8_ok(): pytest.skip("needs sm_89+")
  from fusezap.serve.models import load_model
  from fusezap.serve.quant import quantize_model
  m = quantize_model(load_model(MODEL, dtype=dtypes.bfloat16, device=DEV, max_len=256, prefill_chunk=64), "fp8")
  assert all(l.wqkv.ws is not None for l in m.layers)
  seq, R = _teacher(hf, prompt_ids, 128)
  agree = float((_top1(m, seq, len(prompt_ids)) == R.argmax(-1)[len(prompt_ids) - 1:-1]).mean())
  print(f"fp8 top-1 agreement vs HF f32 over {len(seq) - len(prompt_ids)} positions: {agree:.4f}")
  assert agree >= 0.75, agree

def _engine_alone_vs_batched(model, n: int = 12):
  from fusezap.serve.engine import LLMEngine, SamplingParams
  rng = np.random.default_rng(7)
  lens = [1, 3, 40, 129, 17, 255, 64, 5, 300, 90, 33, 2][:n]
  prompts = [rng.integers(100, 30000, k).tolist() for k in lens]
  maxt = rng.integers(8, 24, n).tolist()
  eng = LLMEngine(model, num_blocks=512, block_size=16, max_num_seqs=16, max_prefill_tokens=256)
  ps = [SamplingParams(max_tokens=m, ignore_eos=True) for m in maxt]
  alone = [eng.generate([p], q)[0] for p, q in zip(prompts, ps)]
  together = eng.generate(prompts, ps)
  for i, (a, t) in enumerate(zip(alone, together)): assert a == t, f"request {i} ({lens[i]} prompt tokens) differs batched"
  return alone

@needs_model
@pytest.mark.slow
def test_engine_bf16_batched_matches_alone(m16):
  _engine_alone_vs_batched(m16)

@needs_model
@pytest.mark.slow
def test_engine_cuda_graph_tokens_identical(m16, monkeypatch):
  """The engine's decode and prefill forwards replay as CUDA graphs, with the same greedy tokens as without them."""
  from fusezap.serve.engine import LLMEngine, SamplingParams
  rng = np.random.default_rng(11)
  prompts = [rng.integers(100, 30000, k).tolist() for k in (3, 70, 300, 9, 400, 250)]   # >= 8 prefill steps
  ps = SamplingParams(max_tokens=12, ignore_eos=True)
  res = {}
  try:
    for on in (True, False):
      _graph_env(monkeypatch, on)
      eng = LLMEngine(m16, num_blocks=256, block_size=16, max_num_seqs=6, max_prefill_tokens=128)
      res[on] = eng.generate(prompts, ps)
      dec = [f.jit.captured for f in eng._decodes.values() if f.jit.captured is not None]   # decode buckets (speed run 4)
      # the prefill side: whichever prefill or mixed forwards ran (with FZ_MIX=1 a prompt that arrives while others decode
      # runs in a mixed forward, so the largest prefill size may never run)
      pre = [f.jit.captured for f in [*eng._prefills.values(), *eng._mixes.values()] if f.jit.captured is not None]
      assert any(c.graph is not None for c in dec) == on and pre and any(c.graph is not None for c in pre) == on
  finally: _graph_env(monkeypatch, True)
  assert res[True] == res[False]

@needs_model
@pytest.mark.slow
def test_engine_rope_kvw_and_fuse_rows_tokens_identical(m16, monkeypatch):
  """I60: RoPE + KV-cache write in one kernel (FZ_ROPE_KVW, default on) and the fused SwiGLU / RoPE kernels at decode
  rows (FZ_FUSE_ROWS=1) give the greedy tokens of the old path (separate kv_write, composed decode RoPE / SwiGLU), and
  the decode graphs lose one kv_write launch per layer."""
  from fusezap.serve.engine import LLMEngine, SamplingParams
  from fusezap.helpers import getenv
  rng = np.random.default_rng(13)
  prompts = [rng.integers(100, 30000, k).tolist() for k in (3, 70, 300, 9, 140)]
  ps = SamplingParams(max_tokens=10, ignore_eos=True)
  res, kvw = {}, {}
  try:
    for name, env in (("old", {"FZ_ROPE_KVW": "0"}), ("kvw", {}), ("rows1", {"FZ_FUSE_ROWS": "1"})):
      for k in ("FZ_ROPE_KVW", "FZ_FUSE_ROWS"): monkeypatch.delenv(k, raising=False)
      for k, v in env.items(): monkeypatch.setenv(k, v)
      getenv.cache_clear()
      eng = LLMEngine(m16, num_blocks=256, block_size=16, max_num_seqs=8, max_prefill_tokens=128)
      res[name] = eng.generate(prompts, ps)
      c = next(f.jit.captured for f in eng._decodes.values() if f.jit.captured is not None)
      names = [it.prg.name if it.prg is not None else getattr(it.si.ast[0], "name", "") for it in c.items]
      kvw[name] = (sum("kv_write" in n for n in names), sum("_kvw_" in n for n in names))
  finally:
    for k in ("FZ_ROPE_KVW", "FZ_FUSE_ROWS"): monkeypatch.delenv(k, raising=False)
    getenv.cache_clear()
  assert res["old"] == res["kvw"] == res["rows1"]
  L = m16.cfg.n_layers
  assert kvw["old"] == (L, 0) and kvw["kvw"] == (0, L) and kvw["rows1"] == (0, L), kvw

@needs_model
@pytest.mark.slow
def test_engine_prefill_sizes_tokens_identical(m16):
  """CUDA prefill forwards come in several row counts (the smallest that holds the step's prompt rows): greedy tokens
  equal a one-size engine's at either end, batched equals alone, and the steps really used several sizes."""
  from fusezap.serve.engine import LLMEngine, SamplingParams
  rng = np.random.default_rng(13)
  prompts = [rng.integers(100, 30000, k).tolist() for k in (3, 300, 70, 129, 400, 9, 256, 33)]
  ps = SamplingParams(max_tokens=10, ignore_eos=True)
  eng = LLMEngine(m16, num_blocks=512, block_size=16, max_num_seqs=8, max_prefill_tokens=1024)
  assert eng.prefill_sizes == [256, 512, 1024]
  used: list[int] = []
  orig = type(eng._prefill).__call__   # called through the class, so the spy wraps it there
  def spy(self, tokens, sel, *a, **k):   # decode forwards pass more arguments (lookahead, speed run 4)
    if self.eng is eng and self.prefill: used.append(self.N)
    return orig(self, tokens, sel, *a, **k)
  type(eng._prefill).__call__ = spy
  try: together = eng.generate(prompts, ps)
  finally: type(eng._prefill).__call__ = orig
  assert len(set(used)) >= 2, used
  for P in (256, 1024):
    one = LLMEngine(m16, num_blocks=512, block_size=16, max_num_seqs=8, max_prefill_tokens=P, prefill_buckets=[P])
    assert one.generate(prompts, ps) == together, f"one {P}-row prefill size"
  assert [eng.generate([p], ps)[0] for p in prompts] == together

def _first_logits(eng, prompts: list[list[int]]) -> tuple[list[np.ndarray], list[int]]:
  """Each prompt's first-token logits row, read off the prefill forwards, and the forwards' row counts."""
  from fusezap.serve.engine import SamplingParams
  got: dict[int, np.ndarray] = {}
  sizes: list[int] = []
  cls = type(eng._prefill)
  orig = cls.__call__
  def spy(self, tokens, sel, *a, **k):
    out = orig(self, tokens, sel, *a, **k)
    if self.eng is eng and self.prefill:
      sizes.append(self.N)
      lg = out.lg.numpy().reshape(self.R, -1)
      for i, r in enumerate(sel): got[tokens[r][0]] = lg[i].copy()
    return out
  cls.__call__ = spy
  try:
    rids = [eng.add_request(p, SamplingParams(max_tokens=1, ignore_eos=True)) for p in prompts]
    while eng.has_unfinished(): eng.step()
  finally: cls.__call__ = orig
  for r in rids: eng.release(r)
  return [got[r] for r in rids], sizes

@needs_model
@pytest.mark.slow
def test_engine_f32_prefill_sizes_bit_identical():
  """Review fix (run 2): without the paged flash prefill (float32 here; also FZ_PAGED_FLASH=0, other head dims, sm < 80)
  prefill attention is the split-K kernel, whose partition count set a token's sum order and used to follow the
  forward's row count. A 400-token prompt alone (512-row forward) and batched with a 1500-token one (2048 rows) then gave
  bitwise different logits; the prefill sizes now share the largest size's partition count."""
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine
  m = load_model(MODEL, dtype=dtypes.float32, device=DEV, max_len=2048, prefill_chunk=128)
  eng = LLMEngine(m, num_blocks=512, block_size=16, max_num_seqs=4, max_prefill_tokens=2048)
  assert eng.prefill_sizes == [256, 512, 1024, 2048]
  rng = np.random.default_rng(21)
  short, long_ = rng.integers(100, 30000, 400).tolist(), rng.integers(100, 30000, 1500).tolist()
  for _ in range(3):   # the third run of each size replays its capture
    (alone,), n1 = _first_logits(eng, [short])
    (_, both), n2 = _first_logits(eng, [long_, short])
    assert n1 == [512] and n2 == [2048], (n1, n2)
    np.testing.assert_array_equal(alone, both)

@needs_model
@pytest.mark.slow
def test_engine_prefill_memory_planned(m16):
  """Review fix (run 2): each CUDA prefill size is its own jit and used to keep every intermediate of its forward alive
  (7B bf16 with 2048 prefill tokens then ran out of memory at warmup on a 24 GB card). Their intermediates now share
  memory where live ranges allow; tokens stay as before (test_engine_prefill_sizes_tokens_identical)."""
  from fusezap.serve.engine import LLMEngine, SamplingParams
  from fusezap.serve.api import warmup
  eng = LLMEngine(m16, num_blocks=512, block_size=16, max_num_seqs=8, max_prefill_tokens=1024)
  warmup(eng)   # captures the larger sizes; two more short steps capture the 256-row one
  for _ in range(2): eng.generate([[1000 + i for i in range(40)]], SamplingParams(max_tokens=1))
  for n, f in eng._prefills.items():
    c = f.jit.captured
    assert c is not None and c.arenas, n
    mid = [b for b in c.persist if b is not None]
    held, unplanned = sum(a.nbytes for a in c.arenas), sum(b.nbytes for b in mid)
    print(f"prefill {n}: intermediates {unplanned / 2**20:.0f} MiB unplanned, {held / 2**20:.0f} MiB planned")
    assert held * 6 < unplanned, (n, held, unplanned)
  assert all(f.jit.plan for f in eng._decodes.values())   # decode buckets are planned too since speed run 4

@needs_model
@pytest.mark.slow
def test_engine_device_argmax_tokens_identical(m16):
  """Greedy rows take their token from the device argmax (only R ints read back); sampled and repetition-penalty rows
  still read the logits. Every request's tokens equal the all-host engine's (device_argmax off)."""
  from fusezap.serve.engine import LLMEngine, SamplingParams
  rng = np.random.default_rng(12)
  prompts = [rng.integers(100, 30000, k).tolist() for k in (5, 140, 300, 17, 260, 64)]
  ps = [SamplingParams(max_tokens=10, ignore_eos=True), SamplingParams(max_tokens=10, temperature=0.8, top_k=40, seed=1),
        SamplingParams(max_tokens=10, ignore_eos=True, repetition_penalty=1.3), SamplingParams(max_tokens=10, top_k=1, temperature=0.5),
        SamplingParams(max_tokens=10, temperature=1.0, top_p=0.9, seed=2), SamplingParams(max_tokens=10, ignore_eos=True)]
  res = {}
  for dev_am in (True, False):
    eng = LLMEngine(m16, num_blocks=256, block_size=16, max_num_seqs=6, max_prefill_tokens=128)
    assert eng.device_argmax
    eng.device_argmax = dev_am
    res[dev_am] = (eng.generate(prompts, ps), eng.generate(prompts[:1] * 3, ps[:1] * 3))
    # speed run 4: the lookahead feeds the next decode from device ids, so it runs only with device_argmax (it crashed
    # without them); with them it runs on the greedy-only stretches
    assert (eng.stats["lookahead_steps"] > 0) == (dev_am and eng.lookahead), (dev_am, eng.stats)
  assert res[True] == res[False]

@needs_model
@pytest.mark.slow
def test_engine_fp8_batched_matches_alone():
  if not _fp8_ok(): pytest.skip("needs sm_89+")
  from fusezap.serve.models import load_model
  from fusezap.serve.quant import quantize_model
  _engine_alone_vs_batched(quantize_model(load_model(MODEL, dtype=dtypes.bfloat16, device=DEV, max_len=512, prefill_chunk=128), "fp8"))

@needs_model
@pytest.mark.slow
def test_engine_int8_prefill_dequant_gemm():
  """int8 on CUDA: prefill (>= 64 rows) dequantizes into a scratch weight and runs the tensor-core GEMM, decode keeps
  the fused composed GEMV; batched == alone."""
  from fusezap.serve.models import load_model
  from fusezap.serve.quant import quantize_model
  _engine_alone_vs_batched(quantize_model(load_model(MODEL, dtype=dtypes.bfloat16, device=DEV, max_len=512, prefill_chunk=128), "int8"), 6)

# ---------------------------------------------------------------- 7B smoke: fusezap serve + the openai client (FZ_7B=1)

def _free_port() -> int:
  with socket.socket() as s: s.bind(("127.0.0.1", 0)); return s.getsockname()[1]

def _wait(proc: subprocess.Popen, port: int, timeout: float = 1800.0) -> None:
  import httpx
  t0 = time.time()
  while time.time() - t0 < timeout:
    if proc.poll() is not None: raise RuntimeError(f"server exited with {proc.returncode}:\n{proc.stdout.read()}")
    try:
      if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200: return
    except httpx.HTTPError: pass
    time.sleep(1.0)
  raise TimeoutError("server did not come up")

@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("FZ_7B"), reason="7B smoke: set FZ_7B=1 (H100 / >= 40 GB GPU)")
@pytest.mark.parametrize("quant", [None, "fp8"], ids=["bf16", "fp8"])
def test_7b_serves_openai_chat(quant):
  from openai import OpenAI
  if quant == "fp8" and not _fp8_ok(): pytest.skip("needs sm_89+")
  port = _free_port()
  cmd = [sys.executable, "-m", "fusezap", "serve", MODEL7, "--device", DEV, "--port", str(port), "--max-model-len", "2048",
         "--log-level", "warning"] + (["--quant", quant] if quant else [])
  t0 = time.time()
  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ, "FZ_HF_DOWNLOAD": "1"})
  try:
    _wait(proc, port)
    up = time.time() - t0
    c = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="unused")
    msgs = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
    r = c.chat.completions.create(model=MODEL7, messages=msgs, max_tokens=16, temperature=0)
    text = r.choices[0].message.content
    print(f"7B {quant or 'bf16'}: up in {up:.0f}s, answer {text!r}, usage {r.usage}")
    assert "paris" in text.lower()
    msgs = [{"role": "user", "content": "Name three primary colors."}]
    chunks = list(c.chat.completions.create(model=MODEL7, messages=msgs, max_tokens=48, temperature=0, stream=True))
    text = "".join(ch.choices[0].delta.content or "" for ch in chunks if ch.choices)
    print(f"7B {quant or 'bf16'} stream: {text!r}")
    assert "red" in text.lower() and "blue" in text.lower()
  finally:
    proc.terminate()
    try: out, _ = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired: proc.kill(); out, _ = proc.communicate()
    print((out or "")[-3000:])

@pytest.mark.slow
@pytest.mark.skipif(not (HAS_CUDA and os.environ.get("FZ_7B")), reason="7B serve memory: set FZ_7B=1 (>= 24 GB GPU)")
def test_7b_serve_defaults_warm_up():
  """Review fix (run 2): `fusezap serve Qwen/Qwen2.5-7B-Instruct` with the defaults (2048-token max_model_len, 1024 KV
  blocks, 32 seqs) builds and warms up all its forwards on a 24 GB card (RTX 4090 / L4 / A10 class). With a fixed
  2048-token prefill budget, warmup ran out of memory there: the budget now follows the free memory (1024 on 24 GB) and
  the prefill jits share their intermediates."""
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine, SamplingParams
  from fusezap.serve.api import warmup
  from fusezap.serve.cli import build_parser
  a = build_parser().parse_args(["serve", MODEL7])
  m = load_model(MODEL7, dtype=dtypes.bfloat16, device=DEV, max_len=a.max_model_len, prefill_chunk=128)
  eng = LLMEngine(m, num_blocks=a.num_blocks, block_size=a.block_size, max_num_seqs=a.max_num_seqs, max_prefill_tokens=a.max_prefill_tokens)
  free = Device[DEV].mem_free()
  print(f"7B serve defaults: prefill sizes {eng.prefill_sizes}, {free / 2**30:.1f} GiB free before warmup")
  assert eng.prefill_sizes[0] == 256 and eng.prefill_sizes[-1] >= 1024
  warmup(eng)   # captures every size above 256 (the 256-row one captures on the first short requests)
  assert all(eng._prefills[n].jit.captured is not None for n in eng.prefill_sizes[1:])
  rng = np.random.default_rng(4)   # then serve: a full largest-size step and short ones, replayed
  out = eng.generate([rng.integers(100, 30000, k).tolist() for k in (eng.prefill_sizes[-1] - 40, 30, 7)], SamplingParams(max_tokens=8, ignore_eos=True))
  assert [len(o) for o in out] == [8, 8, 8]

# ---------------------------------------------------------------- 7B bf16 top-1 vs HF f32: the SPEC §0 item 3 gate (FZ_7B=1)

PROMPTS7 = ["Write a short poem about the sea.", "Explain how a hash table works, with an example in Python.",
            "What were the main causes of the French Revolution?", "Give me a recipe for a vegetarian lasagna.",
            "Summarize the theory of evolution by natural selection in a paragraph.",
            "Translate into French: The weather is nice today, so we are going to the park.",
            "A train leaves at 3 pm going 60 km/h; another at 4 pm going 90 km/h. When does the second catch up? Show the steps.",
            "List five tips for writing clean code and explain each briefly."]

def _f32_linear(lin):
  """Stand-in for a bf16 torch Linear that computes like HF float32: the weight stays bf16 in memory and is widened
  exactly to f32 per call (bf16 -> f32 is exact, so the GEMM is HF f32's GEMM on identical f32 values). Lets a model
  whose f32 weights exceed the GPU (32B: 131 GB) still get the f32 reference."""
  import torch
  class F32Linear(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.register_buffer("w", lin.weight.data, persistent=False)
      self.register_buffer("b", None if lin.bias is None else lin.bias.data.float(), persistent=False)
    def forward(self, x): return torch.nn.functional.linear(x, self.w.float(), self.b)
  return F32Linear()

def _hf_model(path: str, mode: str, dev: str = "cuda"):
  """HF transformers model on the GPU. mode: "f32" (float32 weights), "bf16", or "f32c" (float32 compute, bf16-stored
  linear weights via _f32_linear; embeddings, norms and RoPE in float32, so every activation is float32 as in "f32")."""
  import torch, transformers as tr
  if mode != "f32c": return tr.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32 if mode == "f32" else torch.bfloat16).to(dev).eval()
  m = tr.AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()
  for name, mod in list(m.named_modules()):
    for cn, ch in list(mod.named_children()):
      if isinstance(ch, torch.nn.Linear): setattr(mod, cn, _f32_linear(ch))
  for prm in m.parameters(): prm.data = prm.data.float()
  for mod in m.modules():   # RoPE inverse frequencies recomputed in float32, as HF's own init does
    if hasattr(mod, "inv_freq") and hasattr(mod, "compute_default_rope_parameters") and getattr(mod, "rope_type", "default") == "default":
      inv = mod.compute_default_rope_parameters(mod.config)[0].float()
      mod.inv_freq, mod.original_inv_freq = inv, inv.clone()
  return m.to(dev)

def _f32_fits(path: str) -> bool:
  """The float32 copy of the checkpoint (2x its bf16 safetensors) fits in 85% of the GPU."""
  import torch
  st = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".safetensors"))
  return 2 * st < 0.85 * torch.cuda.get_device_properties(0).total_memory

@pytest.mark.slow
@pytest.mark.skipif(not (HAS_CUDA and os.environ.get("FZ_7B")), reason="7B top-1 gate: set FZ_7B=1 (H100 / >= 48 GB GPU)")
def test_7b_bf16_top1_vs_hf_spec():
  """SPEC §0 item 3: in bf16 the top-1 token matches HF transformers at >= 99% of positions. Qwen2.5-7B-Instruct (or
  FZ_MODEL7), teacher-forced: HF in float32 (on the GPU) greedily continues 8 chat prompts by 128 tokens; fusezap
  bf16 (flash attention, tensor-core linears) runs the same sequences in one full forward each and its argmax is
  compared with the f32 reference's argmax at every continuation position (8 x 128 = 1024). When the f32 weights do not
  fit the GPU (32B), the reference is HF with f32 compute over bf16-stored linear weights (_hf_model "f32c"). The same
  fusezap model is then quantized to fp8 W8A8 and scored against the same reference (printed; not part of the gate)."""
  torch = pytest.importorskip("torch")
  pytest.importorskip("transformers")
  if not torch.cuda.is_available(): pytest.skip("needs torch with CUDA for the f32 reference")
  import gc
  from fusezap.serve.weights import resolve_model
  from fusezap.serve.tokenizer import Tokenizer
  from fusezap.serve.models import load_model
  from fusezap.serve.quant import quantize_model
  path = resolve_model(MODEL7, allow_download=True)
  tok = Tokenizer.from_pretrained(path)
  prompts = [tok.apply_chat_template([{"role": "user", "content": p}], True) for p in PROMPTS7]
  N = 128
  def hf_top1(mode: str, seqs: list[list[int]] | None = None) -> tuple[list[list[int]], list[np.ndarray]]:
    """HF in `mode`: greedy-continue each prompt by N tokens (unless seqs is given), then the top-1 at each continuation position."""
    m = _hf_model(path, mode)
    with torch.no_grad():
      if seqs is None: seqs = [m.generate(torch.tensor([ids], device="cuda"), max_new_tokens=N, min_new_tokens=N, do_sample=False, num_beams=1)[0].tolist() for ids in prompts]
      tops = [m(torch.tensor([q], device="cuda")).logits[0].float().argmax(-1).cpu().numpy()[len(ids) - 1:-1] for q, ids in zip(seqs, prompts)]
    del m; gc.collect(); torch.cuda.empty_cache()
    return seqs, tops
  fmode = os.environ.get("FZ_TOP1_REF") or ("f32" if _f32_fits(path) else "f32c")   # FZ_TOP1_REF=f32c forces the bf16-stored reference
  seqs_ref, ref = hf_top1(fmode)
  _, hfb = hf_top1("bf16", seqs_ref)
  m = load_model(path, dtype=dtypes.bfloat16, device=DEV, max_len=512, prefill_chunk=128)
  assert m.cuda16
  n = sum(len(r) for r in ref)
  def score(model) -> tuple[float, list[str]]:
    ours = [_top1(model, seq, len(ids)) for seq, ids in zip(seqs_ref, prompts)]
    return sum(int((o == r).sum()) for o, r in zip(ours, ref)) / n, [f"{float((o == r).mean()):.3f}" for o, r in zip(ours, ref)]
  agree, per = score(m)
  base = sum(int((b == r).sum()) for b, r in zip(hfb, ref)) / n
  print(f"{MODEL7} bf16 top-1 agreement vs HF {fmode} over {n} positions: fusezap {agree:.4f} (per prompt {per}), HF transformers bf16 {base:.4f}")
  if _fp8_ok():
    a8, per8 = score(quantize_model(m, "fp8"))
    print(f"{MODEL7} fp8 W8A8 top-1 agreement vs HF {fmode} over {n} positions: fusezap {a8:.4f} (per prompt {per8})")
  assert n == len(PROMPTS7) * N
  assert agree >= 0.99, (agree, base)

# ---------------------------------------------------------------- 7B f32 greedy == HF f32, and the engine (FZ_7B=1)

@pytest.mark.slow
@pytest.mark.skipif(not (HAS_CUDA and os.environ.get("FZ_7B")), reason="7B f32 greedy gate: set FZ_7B=1 (>= 48 GB GPU)")
def test_7b_f32_greedy_64_tokens_match_hf():
  """test_serve_model's greedy gate at 7B: fusezap float32 (composed kernels) greedily continues the PROMPTS7 chat
  prompts by 64 tokens and must produce HF transformers' float32 greedy tokens (on the GPU, TF32 off). Prints the
  first mismatch per prompt; a prompt whose HF logits have a near-tie (top-2 gap < 1e-3) at the first mismatch is
  reported as such, but still fails."""
  torch = pytest.importorskip("torch")
  tr = pytest.importorskip("transformers")
  if not torch.cuda.is_available(): pytest.skip("needs torch with CUDA for the f32 7B reference")
  import gc
  from fusezap.serve.weights import resolve_model
  from fusezap.serve.tokenizer import Tokenizer
  from fusezap.serve.models import load_model
  from fusezap.serve.sampling import Sampler
  assert not torch.backends.cuda.matmul.allow_tf32
  path = resolve_model(MODEL7, allow_download=True)
  tok = Tokenizer.from_pretrained(path)
  prompts = [tok.apply_chat_template([{"role": "user", "content": p}], True) for p in PROMPTS7]
  N = 64
  m = tr.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to("cuda").eval()
  refs, gaps = [], []
  with torch.no_grad():
    for ids in prompts:
      out, past, inp = [], None, torch.tensor([ids], device="cuda")
      g = []
      for _ in range(N):
        o = m(inp, past_key_values=past, use_cache=True)
        past, lg = o.past_key_values, o.logits[0, -1].double()
        t2 = lg.topk(2).values
        t = int(lg.argmax()); out.append(t); g.append(float(t2[0] - t2[1]))
        inp = torch.tensor([[t]], device="cuda")
      refs.append(out); gaps.append(g)
  del m, past, o; gc.collect(); torch.cuda.empty_cache()
  fz = load_model(path, dtype=dtypes.float32, device=DEV, max_len=256, prefill_chunk=128)
  assert not fz.cuda16
  bad = []
  for i, (ids, ref) in enumerate(zip(prompts, refs)):
    mine = list(fz.generate(ids, N, Sampler(), stop=[]))
    j = next((k for k, (a, b) in enumerate(zip(mine, ref)) if a != b), None)
    print(f"prompt {i} ({len(ids)} tokens): " + ("64/64 identical" if j is None else f"first mismatch at {j}, HF top-2 gap {gaps[i][j]:.2e}"))
    if j is not None: bad.append((i, j, gaps[i][j]))
  assert not bad, bad

@pytest.mark.slow
@pytest.mark.skipif(not (HAS_CUDA and os.environ.get("FZ_7B")), reason="7B engine: set FZ_7B=1 (>= 24 GB GPU)")
def test_7b_engine_bf16_batched_matches_alone():
  """The continuous-batching engine on the 7B checkpoint (paged KV, CUDA graphs, bf16): every request batched with 11
  others gives the same greedy tokens as alone."""
  from fusezap.serve.weights import resolve_model
  from fusezap.serve.models import load_model
  _engine_alone_vs_batched(load_model(resolve_model(MODEL7, allow_download=True), dtype=dtypes.bfloat16, device=DEV, max_len=512, prefill_chunk=128))
