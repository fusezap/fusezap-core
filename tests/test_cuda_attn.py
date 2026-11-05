"""P3-ATTN: CUDA flash attention (fusezap/device/cuda_kernels/attention.py).

Every kernel is checked against the composed equivalent (the same expression with FZ_CUSTOM=0 on CUDA) and against torch,
with the Metal attention tests' tolerances (tests/test_metal_kernels.py): f32 rtol 1e-4 / atol 1e-5, f16 1e-2, bf16 1e-2
(2e-2 against the composed bf16 path, which rounds its own intermediates).
  - scheduler hook (float32 sdpa -> one kind="custom" item -> the SIMT kernel), also under fz.jit replay
  - direct `attention`: tensor-core prefill (f16/bf16, head dim 64/128, causal, GQA, ragged L/S) and the SIMT path
  - paged decode on fusezap/serve/kv_cache.py's pool layout: ragged lengths, scattered block tables, padding rows, a
    chunk of prefill tokens in one step, NaN in every slot no token may read; drop-in for kv_cache.paged_attention
    inside PagedKVCache.attend and inside a fz.jit step.
The renderer tests run everywhere; the rest need an NVIDIA GPU (FZ_TEST_TIERS=CUDA makes an unavailable GPU a failure).
"""
from __future__ import annotations
import math, os
import numpy as np
import pytest
import torch
from fusezap import Tensor, dtypes, helpers
from fusezap.device import Device, Buffer
from fusezap.device.cuda_kernels import attention as A

F32, F16, BF16 = dtypes.float32, dtypes.float16, dtypes.bfloat16
TDT = {F32: torch.float32, F16: torch.float16, BF16: torch.bfloat16}
TOL = {F32: dict(rtol=1e-4, atol=1e-5), F16: dict(rtol=1e-2, atol=1e-2), BF16: dict(rtol=1e-2, atol=1e-2)}
COMPOSED_TOL = {F32: TOL[F32], F16: TOL[F16], BF16: dict(rtol=2e-2, atol=2e-2)}
WANT_CUDA = "CUDA" in [t.strip() for t in os.getenv("FZ_TEST_TIERS", "").split(",")]
HAVE_CUDA = Device.available("CUDA")
gpu = pytest.mark.skipif(not HAVE_CUDA, reason="needs an NVIDIA GPU + cuda-bindings")
R = np.random.default_rng(11)

@pytest.fixture(autouse=True)
def env(monkeypatch):
  if hasattr(helpers.getenv, "cache_clear"): helpers.getenv.cache_clear()
  for k in ("FZ_INTERP", "FZ_CUSTOM", "FZ_BEAM", "FZ_OPT", "FZ_FLASH", "FZ_FLASH_NW", "FZ_FLASH_BN", "FZ_FLASH_SPLITKV", "FZ_PAGED_SPLIT", "FZ_PAGED_SPLIT_ROWS", "FZ_PAGED_DECODE", "FZ_PAGED_DECODE_PART", "FZ_PAGED_DECODE_ROWS", "FZ_PAGED_DECODE_U", "FZ_PAGED_MMA", "FZ_PAGED_MMA_PART", "FZ_PAGED_KPT", "FZ_PAGED_KPT_VPRE", "FZ_PAGED_KPT_CTAS", "FZ_PAGED_TC_CTAS"): monkeypatch.delenv(k, raising=False)
  # fz_paged_tcd is the hd-128 decode default since round 15: the tests of the other decode kernels pin them by turning it
  # off here; the tcd tests (and test_paged_decode_default) turn it on
  monkeypatch.setenv("FZ_PAGED_TC", "0")
  if HAVE_CUDA:
    monkeypatch.setattr(Device, "DEFAULT", "CUDA", raising=False)
    A.install(Device["CUDA"])
  yield
  if hasattr(helpers.getenv, "cache_clear"): helpers.getenv.cache_clear()

def _clear() -> None:
  if hasattr(helpers.getenv, "cache_clear"): helpers.getenv.cache_clear()

def setenv(monkeypatch, k: str, v: str) -> None:
  monkeypatch.setenv(k, v)
  _clear()

def rnd(*shape) -> np.ndarray: return R.standard_normal(shape).astype(np.float32)
def T(a: np.ndarray, dt=F32) -> Tensor: return Tensor(a, device="CUDA").cast(dt) if dt != F32 else Tensor(a, device="CUDA")
def f32(t: Tensor) -> np.ndarray: return t.cast(F32).numpy() if t.dtype != F32 else t.numpy()
def tt(a: np.ndarray, dt=F32) -> torch.Tensor: return torch.tensor(a).to(TDT[dt])

def sdpa_ref(q, k, v, causal=False, dt=F32, scale=None, coff=0) -> np.ndarray:
  """torch float32 sdpa on the dt-rounded inputs; GQA by repeating kv heads (dim -3); causal: s <= l + coff."""
  q, k, v = [tt(x, dt).float() for x in (q, k, v)]
  if q.ndim >= 3 and k.shape[-3] != q.shape[-3]:
    g = q.shape[-3] // k.shape[-3]
    k, v = k.repeat_interleave(g, dim=-3), v.repeat_interleave(g, dim=-3)
  mask = None
  if causal:
    L, S = q.shape[-2], k.shape[-2]
    mask = torch.arange(S)[None, :] <= torch.arange(L)[:, None] + coff
  return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale).numpy()

def composed(q, k, v, causal, dt, monkeypatch) -> np.ndarray:
  """fusezap's composed sdpa (FZ_CUSTOM=0) on CUDA, kv heads repeated for GQA."""
  setenv(monkeypatch, "FZ_CUSTOM", "0")
  tq, tk, tv = T(q, dt), T(k, dt), T(v, dt)
  if q.ndim >= 3 and k.shape[-3] != q.shape[-3]:
    g = q.shape[-3] // k.shape[-3]
    rep = lambda t: t.unsqueeze(-3).expand(*t.shape[:-2], g, *t.shape[-2:]).reshape(*t.shape[:-3], t.shape[-3] * g, *t.shape[-2:])
    tk, tv = rep(tk), rep(tv)
  out = f32(tq.scaled_dot_product_attention(tk, tv, is_causal=causal))
  monkeypatch.delenv("FZ_CUSTOM")
  _clear()
  return out

# ------------------------------------------------------------------ renderer (no GPU needed)

def test_render_flash_source():
  src = A.render_flash("k", BF16, 128, True, 4, 64)
  assert 'extern "C" __global__ void __launch_bounds__(128) k(' in src and "#include" not in src
  assert "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32" in src and "ldmatrix.sync.aligned.m8n8.x4.trans" in src
  assert "cp.async.cg.shared.global" in src and "cvt.rn.bf16x2.f32" in src
  assert "f16.f16.f32" in A.render_flash("k", F16, 64, False)
  assert A.flash_smem(128, 64) == 65536
  with pytest.raises(AssertionError): A.render_flash("k", F32, 128, False)
  with pytest.raises(AssertionError): A.render_flash("k", F16, 96, False)

def test_render_simt_source():
  src = A.render_simt("k", F16, 160, 136, 4, True)
  assert "fz_h2f" in src and "fz_f2h(acc[c] / den)" in src and "const int coff" in src and "acc[5]" in src
  p = A.render_simt("kp", BF16, 128, 128, 7, paged=True)
  assert "__launch_bounds__(224)" in p and "bt[j / bs]" in p and "const int* __restrict__ M" in p
  assert ", ".join(f"const int {x}" for x in A.PAGED_PARAMS) in p

def test_paged_params_match_kv_cache():
  from fusezap.serve import kv_cache
  assert A.PAGED_PARAMS == kv_cache.ATTN_PARAMS

def test_kernel_choice():
  class D: cc = (9, 0)
  assert A.use_flash(D, BF16, 128, 128, 0.1) and A.use_flash(D, F16, 64, 64, 0.1)
  assert not A.use_flash(D, F32, 128, 128, 0.1) and not A.use_flash(D, BF16, 96, 96, 0.1) and not A.use_flash(D, BF16, 128, 64, 0.1)
  assert not A.use_flash(D, BF16, 128, 128, -1.0)
  class Old: cc = (7, 5)
  assert not A.use_flash(Old, BF16, 128, 128, 0.1)

def test_split_scratch_bounded():
  """Review fix: the split-K scratch used to be N * Hq * ceil(max_seq_len / PART) rows for every call, so a 256-token
  prefill chunk of Qwen2.5-7B at max_seq_len 32768 held 448 MB of f32 acc per layer under fz.jit. NP is now capped so
  N * Hq * NP <= SPLIT_ROWS (8 MB of acc at hd 128) and CTAs loop over chunks; small steps keep one CTA per chunk."""
  Hkv, G, hd, bs = 4, 7, 128, 16
  for N, msl in ((256, 32768), (32, 32768), (256, 4096), (1024, 32768), (4096, 4096)):
    a, c = A.split_calls(BF16, N, Hkv, G, hd, bs, 4096, msl // bs, Hkv * G * hd, 0, N, 2 * N, 1.0)
    NP = a.params[-1]
    assert NP >= 1 and c.params[-1] == NP and a.global_size[0] == N * Hkv * NP
    assert N * Hkv * G * NP <= max(A.SPLIT_ROWS, N * Hkv * G), (N, msl, NP)
    assert N * Hkv * G * NP * hd * 4 <= max(8 << 20, N * Hkv * G * hd * 4)   # f32 acc bytes per call: 8 MB unless N * Hq > cap
  assert A.split_np(256, 28, 32768 // 16, 16) == 2 and A.split_np(32, 28, 32768 // 16, 16) == 18
  # the bench's decode shapes (max_seq_len == ctx) keep one CTA per 256-key chunk
  for N, ctx in ((1, 1024), (1, 8192), (8, 2048), (32, 1024), (32, 4096)): assert A.split_np(N, 28, ctx // 16, 16) == ctx // A.PART
  assert A.split_np(4, 8, 4, 16, rows=1) == 1 and A.split_np(1, 1, 1, 1) == 1
  part, comb = A.render_split("k", BF16, 128, 7)
  assert "c += NP" in part and "min(NP," in comb and "once" not in part and "c += NP" not in A.render_split("k", BF16, 128, 7, False)[0]
  assert A.split_calls(BF16, 32, 4, 7, 128, 16, 64, 256, 3584, 0, 32, 64, 1.0)[0].name.endswith("_g7")             # NP = 16 chunks
  assert A.split_calls(BF16, 256, 4, 7, 128, 16, 64, 2048, 3584, 0, 256, 512, 1.0)[0].name.endswith("_g7_loop")   # NP = 2 < 128

def test_split_np_fixed_across_row_counts():
  """Review fix (run 2): prefill forwards of 256..2048 rows must sum a token's keys in one order, so they size NP for the
  largest (np_n) instead of their own N; the scratch then stays within the cap at every N <= np_n."""
  Hkv, G, hd, bs, MB = 2, 7, 64, 16, 128
  nps = {N: A.split_calls(F32, N, Hkv, G, hd, bs, 4096, MB, Hkv * G * hd, 0, N, 2 * N, 1.0, np_n=2048)[0].params[-1] for N in (256, 512, 1024, 2048)}
  assert set(nps.values()) == {A.split_np(2048, Hkv * G, MB, bs)}, nps
  assert A.split_np(256, Hkv * G, MB, bs) != A.split_np(2048, Hkv * G, MB, bs)   # the bug: each size had its own NP
  with pytest.raises(AssertionError): A.split_calls(F32, 512, Hkv, G, hd, bs, 4096, MB, Hkv * G * hd, 0, 512, 1024, 1.0, np_n=256)

# ------------------------------------------------------------------ GPU

def test_cuda_tier_is_live():
  if WANT_CUDA: assert HAVE_CUDA, "FZ_TEST_TIERS names CUDA but the CUDA device cannot start"
  else: pytest.skip("CUDA tier not requested")

def names(t: Tensor) -> list[str]:
  from fusezap.schedule import create_schedule
  return [si.ast[0].kernel.name if si.kind == "custom" else si.kind for si in create_schedule([t.lazydata]) if si.kind != "copy"]

def custom_and_composed(fn, monkeypatch, *arrs):
  outs = []
  for flag in ("1", "0"):
    setenv(monkeypatch, "FZ_CUSTOM", flag)
    if flag == "1": ks = names(fn(*[T(a) for a in arrs]))
    outs.append(fn(*[T(a) for a in arrs]).numpy())
  monkeypatch.delenv("FZ_CUSTOM")
  _clear()
  return outs[0], outs[1], ks

ATT = [((2, 3, 50, 32), (2, 3, 50, 32), (2, 3, 50, 32)),   # L, S not multiples of anything
       ((1, 4, 128, 64), (1, 4, 128, 64), (1, 4, 128, 64)),
       ((2, 33, 16), (2, 70, 16), (2, 70, 24)),              # L != S, Dv != D
       ((1, 2, 9, 13), (1, 2, 40, 13), (1, 2, 40, 7)),       # odd head dims
       ((3, 64, 128), (3, 64, 128), (3, 64, 128)),
       ((1, 8, 40), (1, 300, 40), (1, 300, 40)),
       ((2, 17, 160), (2, 23, 160), (2, 23, 136))]
ATT_IDS = ["x".join(map(str, s[0])) + "_" + str(s[1][-2]) + "_" + str(s[2][-1]) for s in ATT]

@gpu
@pytest.mark.parametrize("shapes", ATT, ids=ATT_IDS)
def test_attention_hook(shapes, monkeypatch):
  q, k, v = [rnd(*s) for s in shapes]
  c, p, ks = custom_and_composed(lambda q, k, v: q.scaled_dot_product_attention(k, v), monkeypatch, q, k, v)
  assert ks == ["fz_attention"], ks
  np.testing.assert_allclose(c, p, **TOL[F32])
  np.testing.assert_allclose(c, sdpa_ref(q, k, v), **TOL[F32])

@gpu
def test_attention_hook_softmax_form(monkeypatch):
  q, k, v = rnd(2, 40, 32), rnd(2, 56, 32), rnd(2, 56, 32)
  c, p, ks = custom_and_composed(lambda q, k, v: ((q @ k.transpose(-1, -2)) * 0.3).softmax(-1) @ v, monkeypatch, q, k, v)
  assert ks == ["fz_attention"], ks
  np.testing.assert_allclose(c, p, **TOL[F32])

@gpu
def test_attention_hook_large_scores_stable(monkeypatch):
  q, k, v = rnd(1, 2, 64, 32) * 30, rnd(1, 2, 96, 32) * 30, rnd(1, 2, 96, 32)
  c, p, _ = custom_and_composed(lambda q, k, v: q.scaled_dot_product_attention(k, v), monkeypatch, q, k, v)
  assert np.isfinite(c).all()
  np.testing.assert_allclose(c, p, rtol=1e-4, atol=1e-4)

@gpu
def test_attention_hook_transposed_inputs(monkeypatch):
  """Strided operands (k given as (D, S) and transposed): the hook's general strides."""
  q, kt, v = rnd(2, 37, 24), rnd(2, 24, 45), rnd(2, 45, 24)
  c, p, ks = custom_and_composed(lambda q, kt, v: q.scaled_dot_product_attention(kt.transpose(-1, -2), v), monkeypatch, q, kt, v)
  assert ks == ["fz_attention"], ks
  np.testing.assert_allclose(c, p, **TOL[F32])

@gpu
def test_attention_hook_jit_replay():
  from fusezap.jit import Jit
  calls = []
  def f(q, k, v):
    calls.append(1)
    return q.scaled_dot_product_attention(k, v) * 2.0
  jf = Jit(f)
  for _ in range(4):
    q, k, v = rnd(2, 3, 20, 16), rnd(2, 3, 30, 16), rnd(2, 3, 30, 16)
    np.testing.assert_allclose(jf(T(q), T(k), T(v)).numpy(), 2 * sdpa_ref(q, k, v), **TOL[F32])
  assert len(calls) == 2

# direct entry: (q, k, v) shapes; dim -3 is heads (GQA when k's differ)
DIRECT = [((2, 4, 50, 64), (2, 4, 50, 64), (2, 4, 50, 64)),
          ((1, 4, 128, 128), (1, 4, 128, 128), (1, 4, 128, 128)),
          ((2, 8, 77, 128), (2, 2, 77, 128), (2, 2, 77, 128)),        # GQA 4:1, ragged L
          ((1, 28, 200, 128), (1, 4, 200, 128), (1, 4, 200, 128)),    # Qwen2.5-7B heads, 7:1
          ((1, 6, 33, 64), (1, 2, 150, 64), (1, 2, 150, 64)),         # L < S
          ((1, 2, 300, 64), (1, 2, 70, 64), (1, 2, 70, 64)),          # L > S
          ((3, 1, 65, 128), (3, 1, 129, 128), (3, 1, 129, 128)),      # S = 2*64+1
          ((2, 4, 40, 13), (2, 2, 40, 13), (2, 2, 40, 7)),            # SIMT: odd dims, GQA
          ((1, 2, 17, 160), (1, 2, 23, 160), (1, 2, 23, 136))]        # SIMT: big dims
DIRECT_IDS = ["q" + "x".join(map(str, s[0])) + "_kv" + str(s[1][-3]) + "x" + str(s[1][-2]) + "_dv" + str(s[2][-1]) for s in DIRECT]

@gpu
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("dt", [F32, F16, BF16], ids=["f32", "f16", "bf16"])
@pytest.mark.parametrize("shapes", DIRECT, ids=DIRECT_IDS)
def test_attention_direct(shapes, dt, causal, monkeypatch):
  q, k, v = [rnd(*s) for s in shapes]
  out = A.attention(T(q, dt), T(k, dt), T(v, dt), causal=causal)
  assert out.dtype == dt and tuple(out.shape) == shapes[0][:-1] + (shapes[2][-1],)
  got = f32(out)
  np.testing.assert_allclose(got, sdpa_ref(q, k, v, causal, dt), **TOL[dt])
  np.testing.assert_allclose(got, composed(q, k, v, causal, dt, monkeypatch), **COMPOSED_TOL[dt])

@gpu
@pytest.mark.parametrize("dt", [F16, BF16], ids=["f16", "bf16"])
@pytest.mark.parametrize("D", [64, 128])
def test_flash_path_taken_and_matches_simt(dt, D, monkeypatch):
  """The tensor-core kernel is the one used for f16/bf16 at D 64/128, and it agrees with the SIMT kernel (FZ_FLASH=0)."""
  dev = Device["CUDA"]
  if dev.cc < (8, 0): pytest.skip("tensor-core prefill needs sm_80+")
  q, k, v = rnd(2, 8, 190, D), rnd(2, 2, 190, D), rnd(2, 2, 190, D)
  st = lambda H, L: (H * L * D, L * D, D, 1)
  c = A.dense_call(dev, dt, 2, 8, 4, 190, 190, D, D, st(8, 190), st(2, 190), st(2, 190), st(8, 190), True, 0.1)
  assert c.name.startswith("fz_flash_") and c.smem == A.flash_smem(D, A.flash_config(D)[1])
  fl = f32(A.attention(T(q, dt), T(k, dt), T(v, dt), causal=True))
  setenv(monkeypatch, "FZ_FLASH", "0")
  si = f32(A.attention(T(q, dt), T(k, dt), T(v, dt), causal=True))
  np.testing.assert_allclose(fl, si, **TOL[dt])

@gpu
@pytest.mark.parametrize("split", [0, 1], ids=["onegroup", "splitkv"])
@pytest.mark.parametrize("nw,bn", [(4, 32), (4, 64), (4, 128), (8, 32), (8, 64), (8, 128), (2, 64)])
def test_flash_tile_configs(nw, bn, split, monkeypatch):
  """Every tile config and pipeline the bench may pick is correct (block edges at every multiple)."""
  if Device["CUDA"].cc < (8, 0): pytest.skip("needs sm_80+")
  setenv(monkeypatch, "FZ_FLASH_NW", str(nw)); setenv(monkeypatch, "FZ_FLASH_BN", str(bn)); setenv(monkeypatch, "FZ_FLASH_SPLITKV", str(split))
  for D in (64, 128):
    if (bn * D // 8) % (32 * nw) or A.flash_smem(D, bn) > A.max_smem(Device["CUDA"]): continue
    q, k, v = rnd(1, 4, 257, D), rnd(1, 2, 300, D), rnd(1, 2, 300, D)
    for causal in (False, True):
      got = f32(A.attention(T(q, BF16), T(k, BF16), T(v, BF16), causal=causal))
      np.testing.assert_allclose(got, sdpa_ref(q, k, v, causal, BF16), **TOL[BF16])

@gpu
@pytest.mark.parametrize("dt", [F32, BF16], ids=["f32", "bf16"])
def test_attention_causal_offset(dt):
  """coff = S - L: the last query row sees every key (a prefill chunk after cached context)."""
  q, k, v = rnd(1, 4, 40, 128), rnd(1, 2, 170, 128), rnd(1, 2, 170, 128)
  got = f32(A.attention(T(q, dt), T(k, dt), T(v, dt), causal=True, coff=130))
  np.testing.assert_allclose(got, sdpa_ref(q, k, v, True, dt, coff=130), **TOL[dt])

@gpu
def test_attention_scale_broadcast_and_2d():
  q, k, v = rnd(2, 3, 20, 64), rnd(1, 3, 30, 64), rnd(1, 3, 30, 64)
  for dt in (F32, F16):
    got = f32(A.attention(T(q, dt), T(k, dt), T(v, dt), causal=True, scale=0.7))
    ref = sdpa_ref(q, np.broadcast_to(k, (2, 3, 30, 64)), np.broadcast_to(v, (2, 3, 30, 64)), True, dt, scale=0.7)
    np.testing.assert_allclose(got, ref, **TOL[dt])
  q2, k2, v2 = rnd(9, 32), rnd(14, 32), rnd(14, 8)
  got = A.attention(T(q2), T(k2), T(v2))
  assert tuple(got.shape) == (9, 8)
  np.testing.assert_allclose(got.numpy(), sdpa_ref(q2, k2, v2), **TOL[F32])

@gpu
def test_attention_empty_keys():
  got = A.attention(T(rnd(2, 1, 5, 8)), T(rnd(2, 1, 0, 8)), T(rnd(2, 1, 0, 4)))
  np.testing.assert_array_equal(got.numpy(), np.zeros((2, 1, 5, 4), np.float32))

@gpu
def test_attention_long_bf16_float_accum():
  """f32 accumulation over long key ranges (S = 4000) in bf16, causal and full, with an offset in v."""
  q, k, v = rnd(1, 2, 64, 128), rnd(1, 2, 4000, 128), rnd(1, 2, 4000, 128) + 3.0
  for causal, coff in ((False, 0), (True, 3936)):
    got = f32(A.attention(T(q, BF16), T(k, BF16), T(v, BF16), causal=causal, coff=coff))
    np.testing.assert_allclose(got, sdpa_ref(q, k, v, causal, BF16, coff=coff), **TOL[BF16])

@gpu
def test_attention_jit_replay():
  from fusezap.jit import Jit
  calls = []
  def f(q, k, v):
    calls.append(1)
    return A.attention(q, k, v, causal=True) + 1.0
  jf = Jit(f)
  for _ in range(4):
    q, k, v = rnd(1, 4, 70, 64), rnd(1, 2, 70, 64), rnd(1, 2, 70, 64)
    np.testing.assert_allclose(f32(jf(T(q, F16), T(k, F16), T(v, F16))), sdpa_ref(q, k, v, True, F16) + 1.0, **TOL[F16])
  assert len(calls) == 2

# ------------------------------------------------------------------ paged decode (fusezap/serve/kv_cache.py layout)

def _to_dt(a: np.ndarray, dt) -> np.ndarray:
  """float32 values -> the host representation of dt (bf16: uint16 bits)."""
  if dt == F32: return a.astype(np.float32)
  if dt == F16: return a.astype(np.float16)
  return torch.tensor(a).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)

def _from_dt(a: np.ndarray, dt) -> np.ndarray:
  if dt == BF16: return (a.astype(np.uint32) << 16).view(np.float32)
  return a.astype(np.float32)

def _paged_case(dt, hd: int, G: int, lens: dict[int, int], new: dict[int, int], bs: int = 16, Hkv: int = 2, pad: int = 3, seed: int = 0):
  """A PagedKVCache on CUDA whose sequences hold lens[s] cached tokens plus new[s] tokens in this step, with block tables
  scattered over the pool. The pool is random where tokens live and NaN everywhere else. Returns the cache, the step's
  rows (seq, pos), q (N, Hq*hd) as float32, the pool as float32 (dt-rounded) and the meta array."""
  from fusezap.serve.kv_cache import PagedKVCache
  rng = np.random.default_rng(seed)
  total = {s: lens[s] + new[s] for s in lens}
  NB = sum(-(-n // bs) for n in total.values()) + 7
  c = PagedKVCache(1, Hkv, hd, NB, bs, dt, "CUDA", max_seq_len=max(total.values()) + bs)
  c.ensure(999, 5 * bs)                                     # scatter: interleave allocations, then free a hole
  for step in range(max(-(-n // bs) for n in total.values())):
    for s, n in total.items():
      if step * bs < n: c.ensure(s, min(n, (step + 1) * bs))
  c.free(999)
  pool = np.full((2, NB, Hkv, bs, hd), np.nan, np.float32)
  for s, n in total.items():
    for p in range(n):
      sl = c.slot(s, p)
      pool[:, sl // bs, :, sl % bs] = rng.standard_normal((2, Hkv, hd))
  pool = _from_dt(_to_dt(pool, dt), dt)
  c.pools[0] = Tensor(_to_dt(pool, dt).reshape(-1), device="CUDA", dtype=dt).reshape(2, NB, Hkv, bs, hd).contiguous().realize()
  toks = [(s, lens[s] + i) for s in lens for i in range(new[s])]
  N, R = len(toks) + pad, len(lens)
  meta = c.step_meta(N, R, toks, [])
  q = _from_dt(_to_dt(rng.standard_normal((N, Hkv * G * hd)).astype(np.float32), dt), dt)
  return c, toks, q, pool, meta

def _paged_ref(toks, q, pool, c, G: int, scale: float) -> np.ndarray:
  Hkv, hd = c.n_kv, c.hd
  out = np.zeros((q.shape[0], Hkv * G, hd), np.float64)
  for i, (s, p) in enumerate(toks):
    sl = np.array([c.slot(s, j) for j in range(p + 1)])
    K, V = pool[0, sl // c.bs, :, sl % c.bs].astype(np.float64), pool[1, sl // c.bs, :, sl % c.bs].astype(np.float64)   # (S, Hkv, hd)
    for h in range(Hkv * G):
      sc = K[:, h // G] @ q[i].reshape(-1, hd)[h].astype(np.float64) * scale
      e = np.exp(sc - sc.max())
      out[i, h] = (e / e.sum()) @ V[:, h // G]
  return out.reshape(q.shape[0], -1)

PAGED = [(F32, 64, 1, 1), (F16, 128, 7, 1), (BF16, 128, 7, 1), (BF16, 64, 4, 1), (F16, 80, 2, 1), (BF16, 128, 7, 0), (F32, 32, 3, 0)]

@gpu
@pytest.mark.parametrize("dt,hd,G,split", PAGED, ids=[f"{A._tag(d)}_hd{h}_g{g}{'' if sp else '_nosplit'}" for d, h, g, sp in PAGED])
def test_paged_decode_ragged(dt, hd, G, split, monkeypatch):
  """Decode tokens of sequences with ragged lengths (block edges 15/16/17, 1, partition edges 255/256/257, long), a
  5-token chunk of one sequence in the same step, and padding rows. split=0: the one-CTA-per-head kernel (FZ_PAGED_SPLIT=0);
  hd=80 always takes it."""
  setenv(monkeypatch, "FZ_PAGED_SPLIT", str(split))
  assert A.split_ok(hd, G, hd * G * 2) == (bool(split) and hd != 80)
  lens = {0: 0, 1: 14, 2: 15, 3: 16, 4: 100, 5: 700, 6: 31, 7: 255, 8: 256}
  new = {0: 1, 1: 1, 2: 1, 3: 1, 4: 5, 5: 1, 6: 2, 7: 1, 8: 1}
  c, toks, q, pool, meta = _paged_case(dt, hd, G, lens, new)
  scale = 1.0 / math.sqrt(hd)
  N = q.shape[0]
  x = Tensor(_to_dt(q, dt).reshape(-1), device="CUDA", dtype=dt)._realized_contiguous().realized
  mb = Tensor(meta, device="CUDA", dtype=dtypes.int32)._realized_contiguous().realized
  pb = c.pools[0]._realized_contiguous().realized
  from fusezap.serve.kv_cache import StepLayout
  L = StepLayout(N, len(lens), c.max_blocks)
  o = A.paged_attention(x, pb, mb, N, c.n_kv, G, hd, c.bs, c.num_blocks, L.MB, c.n_kv * G * hd, L.row, L.pos, L.bt, scale)
  got = _from_dt(o.numpy().reshape(N, -1), dt)
  assert np.isfinite(got).all(), "read a slot outside the sequences' positions"
  ref = _paged_ref(toks, q, pool, c, G, scale)
  np.testing.assert_allclose(got, ref, **TOL[dt])
  assert np.all(got[len(toks):] == 0)                        # padding rows
  # composed: fusezap sdpa (FZ_CUSTOM=0 path is the default for this graph) on each token's gathered K/V
  for i in (0, 4, len(toks) - 1):
    s, p = toks[i]
    sl = np.array([c.slot(s, j) for j in range(p + 1)])
    K = np.repeat(pool[0, sl // c.bs, :, sl % c.bs].transpose(1, 0, 2), G, axis=0)   # (Hq, S, hd)
    V = np.repeat(pool[1, sl // c.bs, :, sl % c.bs].transpose(1, 0, 2), G, axis=0)
    cq = q[i].reshape(-1, 1, hd)
    comp = f32(T(cq, dt).scaled_dot_product_attention(T(K, dt), T(V, dt))).reshape(-1)
    np.testing.assert_allclose(got[i], comp, **COMPOSED_TOL[dt])

@gpu
@pytest.mark.parametrize("dt", [F32, F16], ids=["f32", "f16"])
def test_paged_dropin_attend(dt, monkeypatch):
  """PagedKVCache.attend with this kernel in place of kv_cache.paged_attention equals attend with kv_cache's own (numpy
  reference) kernel on the same CUDA buffers: same pool layout, same metadata."""
  from fusezap.serve import kv_cache as KV
  Hq, Hkv, hd, bs = 8, 2, 64, 4
  outs = []
  for impl in (KV.paged_attention, A.paged_attention):
    monkeypatch.setattr(KV, "paged_attention", impl)
    c = KV.PagedKVCache(1, Hkv, hd, 40, bs, dt, "CUDA", max_seq_len=64)
    rng = np.random.default_rng(5)
    hist = {0: 0, 1: 0, 2: 0}
    for n_new in ({0: 9, 1: 3, 2: 1}, {0: 1, 1: 1, 2: 6}, {0: 1, 1: 1, 2: 1}):
      toks = []
      for s, n in n_new.items():
        c.ensure(s, hist[s] + n)
        toks += [(s, p) for p in range(hist[s], hist[s] + n)]
        hist[s] += n
      N = len(toks) + 2
      L = KV.StepLayout(N, 3, c.max_blocks)
      c.set_step(L, Tensor(c.step_meta(N, 3, toks, []), device="CUDA", dtype=dtypes.int32).realize())
      qkv = rng.standard_normal((N, Hq + 2 * Hkv, hd)).astype(np.float32)
      o = c.attend(0, Tensor(qkv, device="CUDA").cast(dt), Hq, 1.0 / math.sqrt(hd))
      outs.append(f32(o))
  for a, b in zip(outs[:3], outs[3:]): np.testing.assert_allclose(b, a, **TOL[dt])

@gpu
def test_paged_jit_replay(monkeypatch):
  """attend() inside a fz.jit step with this kernel: recorded with the composed kernels, replayed with new metadata."""
  from fusezap.jit import Jit
  from fusezap.serve import kv_cache as KV
  monkeypatch.setattr(KV, "paged_attention", A.paged_attention)
  Hq, Hkv, hd, bs, NB = 4, 2, 32, 4, 32
  c = KV.PagedKVCache(1, Hkv, hd, NB, bs, dtypes.float32, "CUDA", max_seq_len=64)
  N, R = 8, 4
  L = KV.StepLayout(N, R, c.max_blocks)
  calls = []
  def step(x: Tensor, meta: Tensor, pools: list[Tensor]) -> Tensor:
    calls.append(1)
    b = c.bind(pools, [meta]); b.layout = L
    o = b.attend(0, (x * 2.0).reshape(N, Hq + 2 * Hkv, hd), Hq, 0.25)
    return b.select(o + 1.0)
  jf = Jit(step)
  rng = np.random.default_rng(3)
  hist: dict[int, list[np.ndarray]] = {0: [], 1: []}
  for it in range(5):
    s_lens = {0: 1 if it else 5, 1: 2}
    toks = []
    for s, n in s_lens.items():
      a = len(hist[s])
      c.ensure(s, a + n)
      toks += [(s, p) for p in range(a, a + n)]
    x = rng.standard_normal((N, (Hq + 2 * Hkv) * hd)).astype(np.float32)
    sel = [len(toks) - 1, 0]
    out = jf(Tensor(x, device="CUDA"), Tensor(c.step_meta(N, R, toks, sel), device="CUDA", dtype=dtypes.int32), c.state()).numpy()
    q = (2 * x).reshape(N, Hq + 2 * Hkv, hd)
    for i, (s, p) in enumerate(toks): hist[s].append(q[i, Hq:])
    for r, i in enumerate(sel):
      s, p = toks[i]
      kv = np.stack(hist[s][:p + 1])
      for h in range(Hq):
        sc = kv[:, h // 2] @ q[i, h] * 0.25
        e = np.exp(sc - sc.max())
        np.testing.assert_allclose(out[r, h * hd:(h + 1) * hd], (e / e.sum()) @ kv[:, Hkv + h // 2] + 1.0, rtol=1e-4, atol=1e-5)
  assert len(calls) == 2

@gpu
@pytest.mark.parametrize("np_", [1, 2, 3, 4], ids=lambda n: f"np{n}")
def test_paged_split_capped_np(np_, monkeypatch):
  """The scratch cap (FZ_PAGED_SPLIT_ROWS) forces NP below the chunk count of max_seq_len: each CTA then runs an online
  softmax over chunks p, p + NP, ... and the result still matches float64. Lengths cross the 256-key chunk edges and reach
  6 chunks; the f32 scratch the kernel gets is exactly N * Hq * NP rows."""
  dt, hd, G = BF16, 128, 7
  lens = {0: 0, 1: 255, 2: 256, 3: 700, 4: 1500, 5: 40, 6: 511}
  new = {0: 1, 1: 1, 2: 1, 3: 3, 4: 1, 5: 1, 6: 2}
  c, toks, q, pool, meta = _paged_case(dt, hd, G, lens, new, seed=np_)
  N, Hq = q.shape[0], c.n_kv * G
  setenv(monkeypatch, "FZ_PAGED_SPLIT_ROWS", str(N * Hq * np_))
  setenv(monkeypatch, "FZ_PAGED_DECODE", "0")   # the split kernel (decode now takes the one-pass kernel by default)
  setenv(monkeypatch, "FZ_PAGED_MMA", "0")      # ... or the tensor-core one
  setenv(monkeypatch, "FZ_PAGED_KPT", "0")      # ... or the key-per-thread one
  from fusezap.serve.kv_cache import StepLayout
  L = StepLayout(N, len(lens), c.max_blocks)
  assert -(-L.MB * c.bs // A.PART) > np_ and A.split_np(N, Hq, L.MB, c.bs) == np_
  seen = []
  run = A._run
  monkeypatch.setattr(A, "_run", lambda call, bufs, n_out=1: (seen.append((call.name, [b.size for b in bufs])), run(call, bufs, n_out))[1])
  x = Tensor(_to_dt(q, dt).reshape(-1), device="CUDA", dtype=dt)._realized_contiguous().realized
  mb = Tensor(meta, device="CUDA", dtype=dtypes.int32)._realized_contiguous().realized
  pb = c.pools[0]._realized_contiguous().realized
  scale = 1.0 / math.sqrt(hd)
  o = A.paged_attention(x, pb, mb, N, c.n_kv, G, hd, c.bs, c.num_blocks, L.MB, Hq * hd, L.row, L.pos, L.bt, scale)
  got = _from_dt(o.numpy().reshape(N, -1), dt)
  assert [n for n, _ in seen] == ["fz_paged_split_bf16_d128_g7_loop", "fz_paged_split_bf16_d128_g7_loop_combine"]
  assert seen[0][1][:2] == [N * Hq * np_ * hd, N * Hq * np_ * 2]
  assert np.isfinite(got).all(), "read a slot outside the sequences' positions"
  np.testing.assert_allclose(got, _paged_ref(toks, q, pool, c, G, scale), **TOL[dt])
  assert np.all(got[len(toks):] == 0)

# ------------------------------------------------------------------ one-pass paged decode (speed run 4)

def test_decode_calls_render():
  """Mac: the one-pass decode's partition count is fixed by np_n (never by the step's rows) and covers max_seq_len unless
  the scratch cap forces the chunk loop; both sources render."""
  Hkv, G, hd, bs = 4, 7, 128, 16
  a, c = A.decode_calls(BF16, 32, Hkv, G, hd, bs, 4096, 2048 // bs, Hkv * G * hd, 0, 32, 64, 1.0)
  assert a.name == "fz_paged_decode_bf16_d128_g7_p64" and a.params[-1] == 32 and a.global_size[0] == 32 * Hkv * 32
  assert c.name == a.name + "_combine" and c.params[-1] == 32 and "c * 64 <= pos" not in a.src and "once" in a.src
  nps = {N: A.decode_calls(BF16, N, Hkv, G, hd, bs, 4096, 32768 // bs, Hkv * G * hd, 0, N, 2 * N, 1.0, np_n=32)[0].params[-1] for N in (1, 2, 8, 32)}
  assert len(set(nps.values())) == 1 and nps[1] == A.DECODE_ROWS // (32 * 28)
  assert A.decode_calls(BF16, 1, Hkv, G, hd, bs, 4096, 32768 // bs, Hkv * G * hd, 0, 1, 2, 1.0, np_n=32)[0].name.endswith("_loop")
  for dt, hd_, G_ in ((F32, 64, 1), (F16, 128, 7), (BF16, 64, 4), (F32, 32, 3)):
    src, comb = A.render_decode("k", dt, hd_, G_, 64)
    assert "__shfl_xor_sync" in src and "exp2f" in comb
  with pytest.raises(AssertionError): A.decode_calls(F32, 8, 2, 2, 64, 16, 64, 8, 256, 0, 8, 16, 1.0, np_n=4)
  assert A.decode_ok(128, 7, 3584) and A.decode_ok(64, 11, 1408) and not A.decode_ok(128, 12, 1536) and not A.decode_ok(80, 2, 160)

DECODE = [(BF16, 128, 7, 64, 2), (BF16, 128, 7, 16, 2), (BF16, 128, 7, 256, 2), (F16, 128, 7, 32, 2), (F32, 64, 1, 64, 2), (BF16, 64, 4, 64, 2),
          (F32, 32, 3, 48, 2), (BF16, 128, 7, 64, 4), (BF16, 128, 7, 128, 4), (F16, 64, 7, 32, 1)]

@gpu
@pytest.mark.parametrize("dt,hd,G,part,U", DECODE, ids=[f"{A._tag(d)}_hd{h}_g{g}_p{p}_u{u}" for d, h, g, p, u in DECODE])
def test_paged_decode_onepass(dt, hd, G, part, U, monkeypatch):
  """The one-pass decode against float64 on ragged lengths (block, partition and chunk edges), a 5-token chunk of one
  sequence, padding rows, NaN in every slot no token may read; then the same with the scratch cap forcing the chunk loop."""
  setenv(monkeypatch, "FZ_PAGED_DECODE_PART", str(part))
  setenv(monkeypatch, "FZ_PAGED_DECODE_U", str(U))
  setenv(monkeypatch, "FZ_PAGED_MMA", "0")   # the one-pass kernel (not the tensor-core or the key-per-thread one)
  setenv(monkeypatch, "FZ_PAGED_KPT", "0")
  lens = {0: 0, 1: 14, 2: 15, 3: 16, 4: 100, 5: 700, 6: 31, 7: 255, 8: 256, 9: 63, 10: 64}
  new = {0: 1, 1: 1, 2: 1, 3: 1, 4: 5, 5: 1, 6: 2, 7: 1, 8: 1, 9: 1, 10: 1}
  c, toks, q, pool, meta = _paged_case(dt, hd, G, lens, new)
  scale, N, Hq = 1.0 / math.sqrt(hd), q.shape[0], c.n_kv * G
  from fusezap.serve.kv_cache import StepLayout
  L = StepLayout(N, len(lens), c.max_blocks)
  x = Tensor(_to_dt(q, dt).reshape(-1), device="CUDA", dtype=dt)._realized_contiguous().realized
  mb = Tensor(meta, device="CUDA", dtype=dtypes.int32)._realized_contiguous().realized
  pb = c.pools[0]._realized_contiguous().realized
  ref = _paged_ref(toks, q, pool, c, G, scale)
  outs = []
  for rows in (None, N * Hq * 2):   # one CTA per chunk, then NP = 2 with each CTA looping over chunks
    if rows: setenv(monkeypatch, "FZ_PAGED_DECODE_ROWS", str(rows))
    seen = []
    run = A._run
    monkeypatch.setattr(A, "_run", lambda call, bufs, n_out=1: (seen.append(call.name), run(call, bufs, n_out))[1])
    o = A.paged_attention(x, pb, mb, N, c.n_kv, G, hd, c.bs, c.num_blocks, L.MB, Hq * hd, L.row, L.pos, L.bt, scale)
    monkeypatch.setattr(A, "_run", run)
    assert seen[0].startswith(f"fz_paged_decode_{A._tag(dt)}_d{hd}_g{G}_p{part}") and seen[0].endswith("_loop") == bool(rows), seen
    got = _from_dt(o.numpy().reshape(N, -1), dt)
    assert np.isfinite(got).all(), "read a slot outside the sequences' positions"
    np.testing.assert_allclose(got, ref, **TOL[dt])
    assert np.all(got[len(toks):] == 0)
    outs.append(got)

@gpu
def test_paged_decode_rows_independent(monkeypatch):
  """A token's one-pass decode output is bit-identical whichever rows share its step (alone, reordered, with padding), as
  long as np_n is the same: the decode buckets rely on it."""
  dt, hd, G = BF16, 128, 7
  lens = {0: 3, 1: 700, 2: 64, 3: 129}
  new = {s: 1 for s in lens}
  c, toks, q, pool, meta = _paged_case(dt, hd, G, lens, new, pad=0, seed=4)
  from fusezap.serve.kv_cache import StepLayout
  scale, Hq = 1.0 / math.sqrt(hd), c.n_kv * G
  pb = c.pools[0]._realized_contiguous().realized
  def run(order: list[int], pad: int) -> np.ndarray:
    tk = [toks[i] for i in order]
    N = len(tk) + pad
    m = c.step_meta(N, len(tk), tk, [])
    L = StepLayout(N, len(tk), c.max_blocks)
    qq = np.zeros((N, q.shape[1]), np.float32); qq[:len(tk)] = q[order]
    x = Tensor(_to_dt(qq, dt).reshape(-1), device="CUDA", dtype=dt)._realized_contiguous().realized
    mb = Tensor(m, device="CUDA", dtype=dtypes.int32)._realized_contiguous().realized
    o = A.paged_attention(x, pb, mb, N, c.n_kv, G, hd, c.bs, c.num_blocks, L.MB, Hq * hd, L.row, L.pos, L.bt, scale, np_n=32)
    return o.numpy().reshape(N, -1)[:len(tk)]
  full = run([0, 1, 2, 3], 0)
  for i in range(4): assert np.array_equal(run([i], 0)[0], full[i])
  assert np.array_equal(run([3, 2, 1, 0], 5)[::-1], full)


# ------------------------------------------------------------------ tensor-core GQA paged decode (speed run 4, A/B 3)

def test_mdec_calls_render(monkeypatch):
  setenv(monkeypatch, "FZ_PAGED_MMA", "1")   # off by default since A/B 3
  class D: cc = (9, 0)
  assert A.mdec_ok(D, BF16, 128, 7, 3584) and A.mdec_ok(D, F16, 64, 16, 2048) and not A.mdec_ok(D, F32, 128, 7, 3584)
  assert not A.mdec_ok(D, BF16, 80, 2, 320) and not A.mdec_ok(D, BF16, 128, 17, 3584)
  class Old: cc = (7, 5)
  assert not A.mdec_ok(Old, BF16, 128, 7, 3584)
  setenv(monkeypatch, "FZ_PAGED_MMA", "0")
  assert not A.mdec_ok(D, BF16, 128, 7, 3584)
  a, c = A.mdec_calls(BF16, 32, 4, 7, 128, 16, 4096, 2048 // 16, 3584, 0, 32, 64, 0.1)
  assert a.name == "fz_paged_mdec_bf16_d128_g7_p128" and a.params[-2] == 16 and a.params[-1] == 0 and a.global_size == (32 * 4 * 16, 1, 1) and a.local_size == (32, 1, 1)
  assert "mma.sync" in a.src and "for (int jb = jb0; jb < nb; jb++)" in a.src and c.name == a.name + "_combine"
  nps = {N: A.mdec_calls(BF16, N, 4, 7, 128, 16, 4096, 32768 // 16, 3584, 0, N, 2 * N, 0.1, np_n=32)[0].params[-2] for N in (1, 8, 32)}
  assert len(set(nps.values())) == 1

MDEC = [(BF16, 128, 7, 128), (BF16, 128, 7, 64), (F16, 128, 7, 256), (BF16, 64, 4, 64), (F16, 64, 1, 128), (BF16, 128, 16, 128)]

@gpu
@pytest.mark.parametrize("dt,hd,G,part", MDEC, ids=[f"{A._tag(d)}_hd{h}_g{g}_p{p}" for d, h, g, p in MDEC])
def test_paged_decode_mma(dt, hd, G, part, monkeypatch):
  """The tensor-core decode against float64 on ragged lengths (block, 64-key and partition edges), a 5-token chunk of one
  sequence, padding rows, NaN in every slot no token may read; then with the scratch cap forcing the partition loop."""
  setenv(monkeypatch, "FZ_PAGED_MMA", "1")
  setenv(monkeypatch, "FZ_PAGED_KPT", "0")   # kpt (the default since round 5) would take hd 128 first
  setenv(monkeypatch, "FZ_PAGED_MMA_PART", str(part))
  lens = {0: 0, 1: 14, 2: 15, 3: 16, 4: 100, 5: 700, 6: 31, 7: 255, 8: 256, 9: 63, 10: 64, 11: 127, 12: 128}
  new = {s: 1 for s in lens}; new[4] = 5
  c, toks, q, pool, meta = _paged_case(dt, hd, G, lens, new)
  scale, N, Hq = 1.0 / math.sqrt(hd), q.shape[0], c.n_kv * G
  from fusezap.serve.kv_cache import StepLayout
  L = StepLayout(N, len(lens), c.max_blocks)
  x = Tensor(_to_dt(q, dt).reshape(-1), device="CUDA", dtype=dt)._realized_contiguous().realized
  mb = Tensor(meta, device="CUDA", dtype=dtypes.int32)._realized_contiguous().realized
  pb = c.pools[0]._realized_contiguous().realized
  ref = _paged_ref(toks, q, pool, c, G, scale)
  for rows in (None, N * Hq * 2):
    if rows: setenv(monkeypatch, "FZ_PAGED_DECODE_ROWS", str(rows))
    seen = []
    run = A._run
    monkeypatch.setattr(A, "_run", lambda call, bufs, n_out=1: (seen.append((call.name, call.params[-2])), run(call, bufs, n_out))[1])
    o = A.paged_attention(x, pb, mb, N, c.n_kv, G, hd, c.bs, c.num_blocks, L.MB, Hq * hd, L.row, L.pos, L.bt, scale)
    monkeypatch.setattr(A, "_run", run)
    assert seen[0][0] == f"fz_paged_mdec_{A._tag(dt)}_d{hd}_g{G}_p{part}" and (seen[0][1] == 2) == bool(rows), seen
    got = _from_dt(o.numpy().reshape(N, -1), dt)
    assert np.isfinite(got).all(), "read a slot outside the sequences' positions"
    np.testing.assert_allclose(got, ref, **TOL[dt])
    assert np.all(got[len(toks):] == 0)


# ------------------------------------------------------------------ 4-warp tensor-core GQA decode (speed run 4, round 14)

TCD = [(BF16, 7), (F16, 7), (BF16, 1), (BF16, 4), (BF16, 8), (BF16, 16)]

def test_tcd_calls_render(monkeypatch):
  """No GPU: the launch (a fixed grid of FZ_PAGED_TC_CTAS CTAs looping over the partition-major work items), the scratch
  it is given; on by default (FZ_PAGED_TC=0 falls back to kpt)."""
  monkeypatch.delenv("FZ_PAGED_TC"); _clear()
  assert A.tcd_ok(type("D", (), {"cc": (9, 0)})(), BF16, 128, 7, 3584)
  setenv(monkeypatch, "FZ_PAGED_TC", "0")
  assert not A.tcd_ok(type("D", (), {"cc": (9, 0)})(), BF16, 128, 7, 3584)
  setenv(monkeypatch, "FZ_PAGED_TC", "1")
  assert A.tcd_ok(type("D", (), {"cc": (9, 0)})(), BF16, 128, 7, 3584) and not A.tcd_ok(type("D", (), {"cc": (7, 5)})(), BF16, 128, 7, 3584)
  assert not A.tcd_ok(type("D", (), {"cc": (9, 0)})(), BF16, 64, 7, 3584) and not A.tcd_ok(type("D", (), {"cc": (9, 0)})(), BF16, 128, 17, 3584)
  a, c = A.tcd_calls(BF16, 32, 4, 7, 128, 16, 4096, 2048 // 16, 3584, 0, 32, 64, 0.1)
  assert a.name == "fz_paged_tcd_bf16_d128_g7" and A.call_np(a) == 16 and a.global_size == (A.TCD_CTAS, 1, 1) and a.local_size == (128, 1, 1)
  assert A.tcd_calls(BF16, 1, 4, 7, 128, 16, 4096, 2048 // 16, 3584, 0, 1, 2, 0.1)[0].global_size == (4 * 16, 1, 1)   # fewer items than CTAs
  setenv(monkeypatch, "FZ_PAGED_TC_CTAS", "0")
  assert A.tcd_calls(BF16, 32, 4, 7, 128, 16, 4096, 2048 // 16, 3584, 0, 32, 64, 0.1)[0].global_size == (32 * 4 * 16, 1, 1)
  monkeypatch.delenv("FZ_PAGED_TC_CTAS"); _clear()
  assert a.smem == A.tcd_smem(128) <= 227 * 1024 and "mma.sync" in a.src and "cp.async" in a.src and c.params[4] == 16
  a2, c2 = A.tcd_calls(BF16, 32, 4, 7, 128, 16, 4096, 2048 // 16, 3584, 0, 544, 1088, 0.1, row0=512)
  assert a2.params[6:8] == (512, 1056) and a2.params[-1] == 512 and c2.params[-1] == 512

@gpu
@pytest.mark.parametrize("dt,G", TCD, ids=[f"{A._tag(d)}_g{g}" for d, g in TCD])
def test_paged_decode_tcd(dt, G, monkeypatch):
  """The tensor-core decode against float64 on ragged lengths (block, 32-key warp slice and 128-key partition edges, 700
  keys), a 5-token chunk of one sequence, padding rows, NaN in every slot no token may read; a token's output
  bit-identical alone (another batch, another row count) and with the rows reversed."""
  setenv(monkeypatch, "FZ_PAGED_TC", "1")
  hd = 128
  lens = {0: 0, 1: 14, 2: 15, 3: 16, 4: 100, 5: 700, 6: 31, 7: 255, 8: 256, 9: 127, 10: 128, 11: 129, 12: 33, 13: 1}
  new = {s: 1 for s in lens}; new[4] = 5
  c, toks, q, pool, meta = _paged_case(dt, hd, G, lens, new)
  scale, N, Hq = 1.0 / math.sqrt(hd), q.shape[0], c.n_kv * G
  from fusezap.serve.kv_cache import StepLayout
  L = StepLayout(N, len(lens), c.max_blocks)
  x = Tensor(_to_dt(q, dt).reshape(-1), device="CUDA", dtype=dt)._realized_contiguous().realized
  mb = Tensor(meta, device="CUDA", dtype=dtypes.int32)._realized_contiguous().realized
  pb = c.pools[0]._realized_contiguous().realized
  seen = []
  run = A._run
  monkeypatch.setattr(A, "_run", lambda call, bufs, n_out=1: (seen.append(call.name), run(call, bufs, n_out))[1])
  o = A.paged_attention(x, pb, mb, N, c.n_kv, G, hd, c.bs, c.num_blocks, L.MB, Hq * hd, L.row, L.pos, L.bt, scale, np_n=N)
  monkeypatch.setattr(A, "_run", run)
  assert seen[0] == f"fz_paged_tcd_{A._tag(dt)}_d{hd}_g{G}", seen
  got = _from_dt(o.numpy().reshape(N, -1), dt)
  assert np.isfinite(got).all(), "read a slot outside the sequences' positions"
  np.testing.assert_allclose(got, _paged_ref(toks, q, pool, c, G, scale), **TOL[dt])
  assert np.all(got[len(toks):] == 0)
  raw = o.numpy().reshape(N, -1)
  for i in (0, 5, len(toks) - 1):   # row i alone (same np_n): the same bits
    m1 = c.step_meta(1, 1, [toks[i]], [])
    L1 = StepLayout(1, 1, c.max_blocks)
    x1 = Tensor(_to_dt(q[i:i + 1], dt).reshape(-1), device="CUDA", dtype=dt)._realized_contiguous().realized
    mb1 = Tensor(m1, device="CUDA", dtype=dtypes.int32)._realized_contiguous().realized
    o1 = A.paged_attention(x1, pb, mb1, 1, c.n_kv, G, hd, c.bs, c.num_blocks, L1.MB, Hq * hd, L1.row, L1.pos, L1.bt, scale, np_n=N)
    assert np.array_equal(o1.numpy().reshape(1, -1)[0], raw[i]), i
  rv = list(reversed(range(len(toks))))   # the rows reversed: each row's bits unchanged
  Lr = StepLayout(len(rv), len(lens), c.max_blocks)
  mr = c.step_meta(len(rv), len(lens), [toks[i] for i in rv], [])
  xr = Tensor(_to_dt(q[rv], dt).reshape(-1), device="CUDA", dtype=dt)._realized_contiguous().realized
  mbr = Tensor(mr, device="CUDA", dtype=dtypes.int32)._realized_contiguous().realized
  orr = A.paged_attention(xr, pb, mbr, len(rv), c.n_kv, G, hd, c.bs, c.num_blocks, Lr.MB, Hq * hd, Lr.row, Lr.pos, Lr.bt, scale, np_n=N)
  assert np.array_equal(orr.numpy().reshape(len(rv), -1), raw[rv])
  for ctas in ("0", "1", "7", "1000"):   # grid of 1 CTA to one per item: the same bits (an item's arithmetic is the CTA's alone)
    setenv(monkeypatch, "FZ_PAGED_TC_CTAS", ctas)
    o2 = A.paged_attention(x, pb, mb, N, c.n_kv, G, hd, c.bs, c.num_blocks, L.MB, Hq * hd, L.row, L.pos, L.bt, scale, np_n=N)
    assert np.array_equal(o2.numpy().reshape(N, -1), raw), ctas

# ------------------------------------------------------------------ key-per-thread paged decode (speed run 4, round 4)

def test_kpt_calls_render(monkeypatch):
  assert A.kpt_ok(BF16, 128, 7, 3584, 1000)   # on by default since round 4
  setenv(monkeypatch, "FZ_PAGED_KPT", "0")
  assert not A.kpt_ok(BF16, 128, 7, 3584, 1000)
  setenv(monkeypatch, "FZ_PAGED_KPT", "1")
  assert A.kpt_ok(BF16, 128, 7, 3584, 1000) and A.kpt_ok(F16, 128, 8, 4096, 1000) and not A.kpt_ok(F32, 128, 7, 3584, 1000)
  assert not A.kpt_ok(BF16, 64, 7, 1792, 1000) and not A.kpt_ok(BF16, 128, 9, 4608, 1000) and not A.kpt_ok(BF16, 128, 7, 3584, 1 << 30)
  a, c = A.kpt_calls(BF16, 32, 4, 7, 128, 16, 4096, 2048 // 16, 3584, 0, 32, 64, 0.1)
  # NP = 16 partitions of max_seq_len 2048; by default (FZ_PAGED_KPT_CTAS = 0) one CTA per partition (NPL = NP)
  assert a.name == "fz_paged_kpt_bf16_d128_g7" and a.params[-3] == 16 and a.params[-2] == 0 and a.params[-1] == 16 and a.global_size == (32 * 4 * 16, 1, 1) and a.local_size == (128, 1, 1)
  setenv(monkeypatch, "FZ_PAGED_KPT_CTAS", "512")   # opt-in: NPL = 4 CTAs per (token, kv head) at 32 rows, NP for one row
  assert A.kpt_calls(BF16, 32, 4, 7, 128, 16, 4096, 2048 // 16, 3584, 0, 32, 64, 0.1)[0].global_size == (32 * 4 * 4, 1, 1)
  assert A.kpt_calls(BF16, 1, 4, 7, 128, 16, 4096, 2048 // 16, 3584, 0, 1, 2, 0.1)[0].params[-1] == 16
  setenv(monkeypatch, "FZ_PAGED_KPT_CTAS", str(A.KPT_CTAS))
  assert "cp.async" in a.src and "fz_up8" in a.src and c.name == a.name + "_combine"
  # NP follows max_seq_len only: the same for every row count (bucket)
  assert len({A.kpt_calls(BF16, N, 4, 7, 128, 16, 4096, 2048 // 16, 3584, 0, N, 2 * N, 0.1, np_n=32)[0].params[-3] for N in (1, 8, 32)}) == 1
  a2, c2 = A.kpt_calls(BF16, 32, 4, 7, 128, 16, 4096, 2048 // 16, 3584, 0, 544, 1088, 0.1, row0=512)   # a mixed forward's decode rows
  assert a2.params[6:8] == (512, 1056) and a2.params[-2] == 512 and c2.params[2:4] == (512, 1056) and c2.params[-1] == 512 and "O0" in c2.src

@pytest.mark.parametrize("N,Hkv,G,ctas", [(32, 4, 7, "512"), (1, 1, 8, "512"), (3, 1, 8, "0"), (5, 4, 7, "1"), (32, 4, 7, "0")])
def test_kpt_scratch_size(N, Hkv, G, ctas, monkeypatch):
  """No GPU needed: paged_attention's decode path sizes the Acc / ML scratch for every partition of max_seq_len (NP), and
  the combine reads NP slots; NPL (CTAs per token and kv head) must not change either. Round 12's regression took X0 for NP
  (a zero-sized scratch) and every kpt test on the H100 returned inf / NaN; this catches it on any machine."""
  setenv(monkeypatch, "FZ_PAGED_KPT", "1")
  setenv(monkeypatch, "FZ_PAGED_KPT_CTAS", ctas)
  hd, bs, MB, NB = 128, 16, 2048 // 16, 4096
  seen = []
  monkeypatch.setattr(A, "_run", lambda call, bufs, n_out=1: seen.append((call, [b.size for b in bufs])))
  x = Buffer("CPU", N * Hkv * (G + 2) * hd, BF16)
  pool = Buffer("CPU", 2 * NB * Hkv * bs * hd, BF16)
  meta = Buffer("CPU", 3 * N + N + N * MB, dtypes.int32)
  A.paged_attention(x, pool, meta, N, Hkv, G, hd, bs, NB, MB, Hkv * (G + 2) * hd, 0, N, 2 * N, 0.1, np_n=N)
  (a, sa), (c, sc) = seen
  NP = -(-MB * bs // A.KPT_PART)
  assert a.name.startswith("fz_paged_kpt") and A.call_np(a) == NP == c.params[4]
  assert sa[0] == N * Hkv * G * NP * hd and sa[1] == N * Hkv * G * NP * 2 and sc[1:3] == sa[:2]
  npl = a.params[-1]
  assert 1 <= npl <= NP and a.global_size == (N * Hkv * npl, 1, 1)

KPT = [(BF16, 7, 1), (F16, 7, 1), (BF16, 1, 1), (BF16, 8, 1), (BF16, 4, 1), (BF16, 7, 0)]

@gpu
@pytest.mark.parametrize("dt,G,vpre", KPT, ids=[f"{A._tag(d)}_g{g}{'' if v else '_vlate'}" for d, g, v in KPT])
def test_paged_decode_kpt(dt, G, vpre, monkeypatch):
  """The key-per-thread decode against float64 on ragged lengths (block and 128-key partition edges, 700 keys), a 5-token
  chunk of one sequence, padding rows, NaN in every slot no token may read; and a token's output bit-identical alone,
  reordered and with padding (the decode buckets rely on it)."""
  setenv(monkeypatch, "FZ_PAGED_KPT", "1")
  setenv(monkeypatch, "FZ_PAGED_KPT_VPRE", str(vpre))
  hd = 128
  lens = {0: 0, 1: 14, 2: 15, 3: 16, 4: 100, 5: 700, 6: 31, 7: 255, 8: 256, 9: 127, 10: 128, 11: 129}
  new = {s: 1 for s in lens}; new[4] = 5
  c, toks, q, pool, meta = _paged_case(dt, hd, G, lens, new)
  scale, N, Hq = 1.0 / math.sqrt(hd), q.shape[0], c.n_kv * G
  from fusezap.serve.kv_cache import StepLayout
  L = StepLayout(N, len(lens), c.max_blocks)
  x = Tensor(_to_dt(q, dt).reshape(-1), device="CUDA", dtype=dt)._realized_contiguous().realized
  mb = Tensor(meta, device="CUDA", dtype=dtypes.int32)._realized_contiguous().realized
  pb = c.pools[0]._realized_contiguous().realized
  seen = []
  run = A._run
  monkeypatch.setattr(A, "_run", lambda call, bufs, n_out=1: (seen.append(call.name), run(call, bufs, n_out))[1])
  o = A.paged_attention(x, pb, mb, N, c.n_kv, G, hd, c.bs, c.num_blocks, L.MB, Hq * hd, L.row, L.pos, L.bt, scale, np_n=N)
  monkeypatch.setattr(A, "_run", run)
  assert seen[0] == f"fz_paged_kpt_{A._tag(dt)}_d{hd}_g{G}{'' if vpre else '_vlate'}", seen
  got = _from_dt(o.numpy().reshape(N, -1), dt)
  assert np.isfinite(got).all(), "read a slot outside the sequences' positions"
  np.testing.assert_allclose(got, _paged_ref(toks, q, pool, c, G, scale), **TOL[dt])
  assert np.all(got[len(toks):] == 0)
  raw = o.numpy().reshape(N, -1)
  for i in (0, 5, len(toks) - 1):   # row i alone (same np_n): the same bits
    s_, p_ = toks[i]
    m1 = c.step_meta(1, 1, [toks[i]], [])
    L1 = StepLayout(1, 1, c.max_blocks)
    x1 = Tensor(_to_dt(q[i:i + 1], dt).reshape(-1), device="CUDA", dtype=dt)._realized_contiguous().realized
    mb1 = Tensor(m1, device="CUDA", dtype=dtypes.int32)._realized_contiguous().realized
    o1 = A.paged_attention(x1, pb, mb1, 1, c.n_kv, G, hd, c.bs, c.num_blocks, L1.MB, Hq * hd, L1.row, L1.pos, L1.bt, scale, np_n=N)
    assert np.array_equal(o1.numpy().reshape(1, -1)[0], raw[i]), i
  for ctas in ("0", "1", "8", "100000"):   # CTAs per (token, kv head) from 1 to every partition: the same bits
    setenv(monkeypatch, "FZ_PAGED_KPT_CTAS", ctas)
    o2 = A.paged_attention(x, pb, mb, N, c.n_kv, G, hd, c.bs, c.num_blocks, L.MB, Hq * hd, L.row, L.pos, L.bt, scale, np_n=N)
    assert np.array_equal(o2.numpy().reshape(N, -1), raw), ctas


@gpu
def test_paged_decode_default_is_tcd(monkeypatch):
  """With nothing set, hd-128 decode rows take fz_paged_tcd (sm_80+), and FZ_PAGED_TC=0 takes kpt: both close to the
  float64 reference on the 7B shape (g7)."""
  dt, hd, G = BF16, 128, 7
  lens = {0: 600, 1: 31, 2: 128, 3: 129, 4: 0}
  c, toks, q, pool, meta = _paged_case(dt, hd, G, lens, {s: 1 for s in lens}, Hkv=4)
  scale, N, Hq = 1.0 / math.sqrt(hd), q.shape[0], c.n_kv * G
  from fusezap.serve.kv_cache import StepLayout
  L = StepLayout(N, len(lens), c.max_blocks)
  x = Tensor(_to_dt(q, dt).reshape(-1), device="CUDA", dtype=dt)._realized_contiguous().realized
  mb = Tensor(meta, device="CUDA", dtype=dtypes.int32)._realized_contiguous().realized
  pb = c.pools[0]._realized_contiguous().realized
  ref = _paged_ref(toks, q, pool, c, G, scale)
  for tc, name in (("", "fz_paged_tcd"), ("0", "fz_paged_kpt")):
    if tc: setenv(monkeypatch, "FZ_PAGED_TC", tc)
    else: monkeypatch.delenv("FZ_PAGED_TC"); _clear()
    seen = []
    run = A._run
    monkeypatch.setattr(A, "_run", lambda call, bufs, n_out=1: (seen.append(call.name), run(call, bufs, n_out))[1])
    o = A.paged_attention(x, pb, mb, N, c.n_kv, G, hd, c.bs, c.num_blocks, L.MB, Hq * hd, L.row, L.pos, L.bt, scale, np_n=N)
    monkeypatch.setattr(A, "_run", run)
    assert seen[0].startswith(name), seen
    np.testing.assert_allclose(_from_dt(o.numpy().reshape(N, -1), dt), ref, **TOL[dt])
