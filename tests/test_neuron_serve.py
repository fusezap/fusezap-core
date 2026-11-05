"""N5: serving on NEURON (fusezap/serve/neuron_kv.py), checked on the Mac without a device:
  * the NKI kernels it adds (paged KV write, RMSNorm + residual, keyrange attention with two-level rows) through NKISHIM
    against numpy (the KV write bit for bit) and against the composed kernels (RMSNorm bit for bit);
  * LLMEngine and Llama.prefill/decode on NKISHIM against CPU on a tiny random Qwen2-architecture checkpoint: the same
    greedy tokens (also with preemption), and the per-token launch count;
  * the same engine on the persistent NEURON path with the fake nrtpy of tests/test_neuron_nrt.py (NEFF binding of the
    in-place and two-output kernels, launches batched into one runtime-worker request per replay);
  * Qwen2.5-0.5B-Instruct on NKISHIM against the float32 reference (docs/validate/examples/refs), when it is cached.
"""
from __future__ import annotations
import json, os
import numpy as np
import pytest
from fusezap import Tensor
from fusezap.device import Buffer, Device
from fusezap.dtype import dtypes
from fusezap.codegen import nki_kernels as nk
from test_tp import tiny_model_dir
from test_neuron_nrt import nrt, neuron_dev   # noqa: F401  (fixtures)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = np.random.default_rng(7)
# the NEURON tier these tests run on: NKISHIM on the Mac; FZ_TEST_TIERS=NEURON (scripts/nki_n5_inf2.sh) or NKISIM selects that one
TIER = next((t for t in os.getenv("FZ_TEST_TIERS", "").split(",") if t.strip() in ("NEURON", "NKISIM")), "NKISHIM").strip()

def _launch(sp, arrays: list[np.ndarray]) -> list[np.ndarray]:
  """Run a NeuronCall spec's kernel on TIER over device copies of the flat arrays; returns them read back."""
  dev = Device[TIER]
  bufs = [Buffer(TIER, a.size, dtypes.float32 if a.dtype == np.float32 else dtypes.int32).copyin(memoryview(np.ascontiguousarray(a)).cast("B")) for a in arrays]
  dev.run_program(dev.spec_program(sp), tuple(bufs))
  return [b.numpy() for b in bufs]

# ---------------------------------------------------------------- kernels

def test_kv_write_kernel_is_an_exact_scatter():
  """pool rows at the step's slots take the token's [k | v] exactly; every other row is untouched. 130 tokens (two K
  chunks), 600 columns (two PSUM column tiles), 300 slots (a ragged last tile), padding tokens (slot -1)."""
  P, N, C2, xoff = 300, 130, 600, 64
  xs = xoff + C2 + 1 + 7
  pool = R.standard_normal(P * C2).astype(np.float32)
  slots = np.full(N, -1)
  slots[:120] = R.permutation(P)[:120]
  X = R.standard_normal((N, xs)).astype(np.float32)
  X[:, xoff + C2] = 1.0
  WT = (slots[:, None] == np.arange(P)[None, :]).astype(np.float32)
  want = pool.reshape(P, C2).copy()
  for t in np.flatnonzero(slots >= 0): want[slots[t]] = X[t, xoff:xoff + C2]
  pool = _launch(nk.KVWriteSpec(P, N, C2, xs, xoff), [pool, WT.reshape(-1), X.reshape(-1)])[0]
  assert np.array_equal(pool.reshape(P, C2), want)

@pytest.mark.parametrize("add", [True, False])
def test_rmsnorm_kernel_matches_the_composed_kernels(add):
  """NormSpec (one launch, also storing x + d) gives the composed Llama._rmsnorm kernels' bits (same instructions, same order)."""
  from fusezap.serve.neuron_kv import rmsnorm
  N, D, eps = 130, 896, 1e-6
  x, d, w = (R.standard_normal(s).astype(np.float32) * 3 for s in ((N, D), (N, D), (D,)))
  xs, ds, ws = (Tensor(a, device=TIER) for a in (x, d, w))
  xo, h = rmsnorm(xs, ws, eps, ds if add else None)
  xr = xs + ds if add else xs
  ref = ws * (xr * ((xr * xr).mean(-1, keepdim=True) + eps).rsqrt())
  assert np.array_equal(xo.numpy(), (x + d) if add else x)
  assert np.array_equal(h.numpy(), ref.numpy())
  x64 = (x + d if add else x).astype(np.float64)
  np.testing.assert_allclose(h.numpy(), w * x64 / np.sqrt((x64 * x64).mean(-1, keepdims=True) + eps), rtol=2e-6, atol=1e-6)

def test_attention_keyrange_two_level_rows():
  """Rows (token, q head of the group) with two strides, keys = every pool slot, key t visible iff lo[t] <= kid <= hi[t];
  a token with no visible key gets zeros (padding rows), never NaN."""
  N, Hkv, G, hd, P = 5, 2, 3, 16, 700
  Hq, C = Hkv * G, Hkv * hd
  xs = (Hq + 2 * Hkv + 1) * hd   # q read from the [q | k | v | 1] rows, as NeuronKVCache does
  X = R.standard_normal(N * xs).astype(np.float32)
  pool = R.standard_normal(P * 2 * C).astype(np.float32)
  kid = np.full(P, -1, np.int32); kid[100:140] = np.arange(40); kid[600:650] = 4096 + np.arange(50)
  lo, hi = np.array([0, 0, 4096, 0, 1], np.int32), np.array([10, 39, 4096 + 49, 0, 0], np.int32)
  meta = np.concatenate([lo, hi, kid]).astype(np.int32)
  sp = nk.AttnSpec((Hkv, 1), (N, G), P, hd, hd, (G * hd, 0, hd, 1, 0), (hd, 0, 2 * C, 1, 0), (hd, 0, 2 * C, 1, C), (G * hd, 0, hd, 1, 0),
                   0.25, "keyrange", (1, 0, 0, N, 2 * N), rs=(xs, Hq * hd), kv_one=True)
  out = _launch(sp, [np.full(N * Hq * hd, np.nan, np.float32), X, pool, meta])[0]
  ref = np.zeros(N * Hq * hd)
  sp.ref([ref, X, pool, pool, meta])
  np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
  assert not np.isnan(out).any() and not out.reshape(N, -1)[4].any()

def _run_prog(prog, arrays: list[np.ndarray]) -> list[np.ndarray]:
  """A rendered NKIProgram on TIER over device copies of the flat arrays (float32 / int32); returns them read back."""
  dev = Device[TIER]
  bufs = [Buffer(TIER, a.size, dtypes.float32 if a.dtype == np.float32 else dtypes.int32).copyin(memoryview(np.ascontiguousarray(a)).cast("B")) for a in arrays]
  dev.run_program(dev.runtime(prog, dev.compiler.compile(prog.src)), tuple(bufs))
  return [b.numpy() for b in bufs]

def _same(x: np.ndarray, y: np.ndarray, what: str) -> None:
  """Bit for bit on NKISHIM / NKISIM (the transpose is exact there); on NEURON also the share of equal bits, printed (the
  N6 hardware run reads it: TensorE's float32 transpose of finite values is expected to be exact too)."""
  if TIER == "NEURON":
    print(f"{what} on NEURON: {np.mean(x.view(np.int32) == y.view(np.int32)):.6f} of the outputs bitwise equal, max |diff| {np.abs(x - y).max():.3g}")
    np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6)
  else: assert np.array_equal(x, y), what

@pytest.mark.parametrize("M,K", [(130, 300), (128, 16640), (1, 896)], ids=["two_row_blocks", "A_reloaded_per_tile", "decode_row"])
def test_transposed_operand_loads_give_the_same_bits(monkeypatch, M, K):
  """N6: Aᵀ (matmul) and kᵀ (attention) loaded as contiguous rows and transposed on TensorE instead of a strided DMA of
  single elements: the same values, so the same bits (both kernels, every layout they take)."""
  N = 520
  a, b = R.standard_normal(M * K).astype(np.float32), R.standard_normal(K * N).astype(np.float32)
  params = (1, M, N, K, 0, K, 1, 0, N, 1, 0, N, 1, 0, 0)
  outs = []
  for rows in (0, 1):
    monkeypatch.setattr(nk, "AT_ROWS", rows)
    prog = nk.render_matmul(params)
    assert ("nc_transpose" in prog.src) == bool(rows)
    outs.append(_run_prog(prog, [np.full(M * N, np.nan, np.float32), a, b])[0])
  _same(outs[0], outs[1], f"matmul Aᵀ by rows {M}x{N}x{K}")
  np.testing.assert_allclose(outs[1].reshape(M, N), a.reshape(M, K).astype(np.float64) @ b.reshape(K, N), rtol=2e-4, atol=2e-4)
  Nt, Hkv, G, hd, P = 3, 2, 7, 64, 1100
  Hq, C = Hkv * G, Hkv * hd
  xs = (Hq + 2 * Hkv + 1) * hd
  X, pool = R.standard_normal(Nt * xs).astype(np.float32), R.standard_normal(P * 2 * C).astype(np.float32)
  kid = np.full(P, -1, np.int32); kid[:300] = np.arange(300); kid[700:1000] = 4096 + np.arange(300)
  meta = np.concatenate([np.int32([0, 4096, 10]), np.int32([250, 4096 + 299, 9]), kid]).astype(np.int32)
  sp = nk.AttnSpec((Hkv, 1), (Nt, G), P, hd, hd, (G * hd, 0, hd, 1, 0), (hd, 0, 2 * C, 1, 0), (hd, 0, 2 * C, 1, C), (G * hd, 0, hd, 1, 0),
                   0.125, "keyrange", (1, 0, 0, Nt, 2 * Nt), rs=(xs, Hq * hd), kv_one=True)
  outs = []
  for kt in (False, True):
    monkeypatch.setattr(nk, "KT_ROWS", kt)
    outs.append(_run_prog(nk.render_attention(sp), [np.full(Nt * Hq * hd, np.nan, np.float32), X, pool, meta])[0])
  _same(outs[0], outs[1], "attention kᵀ by rows")
  assert not np.isnan(outs[1]).any()
  ref = np.zeros(Nt * Hq * hd)
  sp.ref([ref, X, pool, pool, meta])
  np.testing.assert_allclose(outs[1], ref, rtol=1e-5, atol=1e-6)

@pytest.mark.parametrize("head", ["v1", "v2", "v3"])
@pytest.mark.parametrize("rows,V", [(1, 151936), (4, 151936), (3, 1000)])
def test_device_argmax_is_np_argmax(rows, V, head, monkeypatch):
  """N7: the greedy token on the device (neuron_kv.argmax_rows: the head's last kernels, so a greedy step reads R int32
  back, not R x vocab logits) is np.argmax of each row: the lowest index of the maximum on ties (several maxima, a row of
  equal values, the maximum last, -0.0 vs 0.0), Qwen2.5's 151936-entry vocabulary, several rows at once. Both heads
  (FZ_NEURON_HEAD=v1: the composed kernels; v2 and v3: ArgmaxSpec, the vocabulary on the partitions)."""
  from fusezap.serve.neuron_kv import argmax_rows
  monkeypatch.setenv("FZ_NEURON_HEAD", head)
  rng = np.random.default_rng(rows * V)
  x = (rng.standard_normal((rows, V)) * 8).astype(np.float32)
  x[0, [7, V // 2, V - 1]] = x[0].max() + 1                    # a three-way tie: index 7
  if rows > 1: x[1] = np.float32(0.25)                         # all equal: index 0
  if rows > 2: x[2, V - 1] = x[2].max() + 1                    # the last index alone
  if rows > 3: x[3] = -1.0; x[3, V - 2], x[3, V - 1] = -0.0, 0.0   # -0.0 == 0.0: the first of the two
  got = argmax_rows(Tensor(x, device=TIER)).numpy()
  assert got.dtype == np.int32 and got.tolist() == np.argmax(x, -1).tolist()

@pytest.mark.parametrize("V", [151936, 128256, 64128, 1000, 1001, 200, 77])   # 128256 / 64128: Llama-3.1-8B, and its TP=2 half (I13)
def test_argmax_ties_straddle_partitions(V):
  """N7 ArgmaxSpec: ties across the per-partition runs (c = ceil(V / 128) entries each), the 512-column tiles and the
  padded tail partition resolve as np.argmax does (the lowest index); -0.0 == 0.0 across a run boundary; a maximum on the
  very last entry; V not a multiple of 128 (a short tail partition, padded with -FLT_MAX) and V < 128."""
  c = -(-V // 128)
  rng = np.random.default_rng(V)
  rows = []
  def row(*idx, val=None, base=None):
    x = (rng.standard_normal(V) * 4).astype(np.float32) if base is None else np.full(V, base, np.float32)
    top = x.max() + 3 if val is None else val
    for i in idx: x[i % V] = top
    rows.append(x)
  row(c, c - 1)                                   # the last entry of run 0 and the first of run 1: c - 1
  row(5 * c, 3 * c - 1, 7 * c + 1)               # three runs, the middle one's last entry is the lowest index
  row(511, 512, 1023)                            # across a 512-column boundary
  row(V - 1, V - 2)                              # the tail of the last (possibly short) run
  row(V - 1)                                     # the very last entry alone
  row(c - 1, c, val=0.0, base=-1.0); rows[-1][c - 1] = -0.0   # -0.0 then 0.0 across a run boundary: c - 1
  row(base=-2.5)                                 # all equal: 0
  x = np.stack(rows)
  sp = nk.ArgmaxSpec(*x.shape)
  got = _launch(sp, [np.full(x.shape[0], -7, np.int32), x.reshape(-1)])[0]
  assert got.tolist() == np.argmax(x, -1).tolist()

@pytest.mark.parametrize("M,V,K", [(1, 151936, 896), (4, 151936, 896), (3, 1000, 200), (17, 700, 64), (130, 520, 300)])
def test_head_v2_matches_the_matmul_bitwise(M, V, K):
  """N7 HeadSpec (FZ_NEURON_HEAD=v2) over head_tiles' layout gives render_matmul's logits (v1: h @ kmajor(lm_head)) bit
  for bit: the same nc_matmul per 512-column tile and K chunk in the same order, only the weight loads differ. Covers a K
  tail (K % 128 != 0), K < 128, a narrow last block, two row blocks and Aᵀ by rows (M >= 16)."""
  rng = np.random.default_rng(M * V + K)
  h, w = rng.standard_normal((M, K)).astype(np.float32), (rng.standard_normal((V, K)) * 0.1).astype(np.float32)
  v1 = _run_prog(nk.render_matmul((1, M, V, K, 0, K, 1, 0, V, 1, 0, V, 1, 0, 0)),
                 [np.full(M * V, np.nan, np.float32), h.reshape(-1), np.ascontiguousarray(w.T).reshape(-1)])[0]
  for W, d in ((nk.HEAD_W, nk.HEAD_DMAS),) + (((512, 4),) if TIER != "NEURON" else ()):   # hardware: the default only (compiles)
    sp = nk.HeadSpec(M, V, K, W, d)
    v2 = _launch(sp, [np.full(M * V, np.nan, np.float32), h.reshape(-1), nk.head_tiles(w, W)])[0]
    if TIER == "NEURON": print(f"head v2 {sp.name} W={W} on NEURON: {np.mean(v1.view(np.int32) == v2.view(np.int32)):.6f} of the logits bitwise equal to v1")
    assert np.array_equal(v1.view(np.int32), v2.view(np.int32)), f"{sp.name} W={W}: {np.mean(v1 == v2):.6f} equal"
  np.testing.assert_allclose(v2.reshape(M, V), h.astype(np.float64) @ w.T, rtol=2e-4, atol=2e-4)

@pytest.mark.parametrize("M,V,K", [(1, 151936, 896), (4, 20000, 896), (3, 1000, 200), (17, 700, 64), (130, 520, 300)])
def test_head_v4_weight_stationary(M, V, K, monkeypatch):
  """N7 step 4 HeadSpec(ws=True) (FZ_NEURON_HEAD=v4): each 128-vocab slice of the relaid weight stationary, hᵀ moving, the
  same K chunks accumulated in PSUM in the same order, logitsᵀ back to rows by exact transposes: render_matmul's products and
  sums with the operands swapped. Bit for bit with the order-independent shim matmul (FZ_NKISHIM_MM=f64, both kernels);
  BLAS's float32 kernels (the default shim) round the swapped form differently, so there only close. On NEURON: the share of
  equal bits (whether TensorE's float32 matmul is operand-symmetric), printed."""
  rng = np.random.default_rng(M * V + K + 1)
  h, w = rng.standard_normal((M, K)).astype(np.float32), (rng.standard_normal((V, K)) * 0.1).astype(np.float32)
  if TIER == "NKISHIM": monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  v1 = _run_prog(nk.render_matmul((1, M, V, K, 0, K, 1, 0, V, 1, 0, V, 1, 0, 0)),
                 [np.full(M * V, np.nan, np.float32), h.reshape(-1), np.ascontiguousarray(w.T).reshape(-1)])[0]
  for W in (nk.HEAD_W,) + ((512,) if TIER != "NEURON" else ()):
    sp = nk.HeadSpec(M, V, K, W, 1, ws=True)
    assert sp.name.startswith("fz_headw_")
    v4 = _launch(sp, [np.full(M * V, np.nan, np.float32), h.reshape(-1), nk.head_tiles(w, W)])[0]
    eq = np.mean(v1.view(np.int32) == v4.view(np.int32))
    if TIER == "NEURON": print(f"head v4 {sp.name} W={W} on NEURON: {eq:.6f} of the logits bitwise equal to v1")
    else: assert eq == 1.0, f"{sp.name} W={W}: {eq:.6f} equal"
    np.testing.assert_allclose(v4.reshape(M, V), h.astype(np.float64) @ w.T, rtol=2e-4, atol=2e-4)

# ---------------------------------------------------------------- N7 step 3: a decoder layer as one kernel (LayerSpec)

def _layer_pair(D, I, Hq, Hkv, hd, rows, P, bs, ctx, add, obias, variant, monkeypatch, qbias: bool = True) -> dict:
  """One decoder layer on TIER through the members (FZ_NEURON_LAYER=v1: Llama.forward_tokens' ten kernels) and through
  NeuronKVCache.layer (LayerSpec, `variant`) from the same state: {output: (v1, v2)} for the residual, the down
  projection, the pool, X = [q | k | v | 1] and the attention output."""
  import math
  from fusezap import jit as J
  from fusezap.serve.kv_cache import StepLayout
  from fusezap.serve.models.llama import Llama, LlamaConfig, rope_tables
  from fusezap.serve.neuron_kv import NeuronKVCache, rope_rows
  rng = np.random.default_rng(D * 7 + I + rows * 31 + P)
  c = {"model_type": "qwen2", "hidden_size": D, "intermediate_size": I, "num_hidden_layers": 1, "num_attention_heads": Hq,
       "num_key_value_heads": Hkv, "head_dim": hd, "vocab_size": 64, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
       "tie_word_embeddings": False, "hidden_act": "silu", "max_position_embeddings": 4096, "eos_token_id": 63}
  r = lambda *s, sc=0.1: (rng.standard_normal(s) * sc).astype(np.float32)
  p = "model.layers.0."
  ws = {"model.embed_tokens.weight": r(64, D), "model.norm.weight": 1 + r(D), "lm_head.weight": r(64, D),
        p + "input_layernorm.weight": 1 + r(D), p + "post_attention_layernorm.weight": 1 + r(D), p + "self_attn.q_proj.weight": r(Hq * hd, D),
        p + "self_attn.k_proj.weight": r(Hkv * hd, D), p + "self_attn.v_proj.weight": r(Hkv * hd, D, sc=0.3), p + "self_attn.o_proj.weight": r(D, Hq * hd),
        p + "mlp.gate_proj.weight": r(I, D), p + "mlp.up_proj.weight": r(I, D), p + "mlp.down_proj.weight": r(D, I),
        **({p + "self_attn.q_proj.bias": r(Hq * hd, sc=0.3), p + "self_attn.k_proj.bias": r(Hkv * hd, sc=0.3), p + "self_attn.v_proj.bias": r(Hkv * hd)}
           if qbias else {})}   # qbias False: Llama's layer (I13), no q/k/v bias
  if obias: ws[p + "self_attn.o_proj.bias"] = r(D, sc=0.3)
  m = Llama(LlamaConfig.from_hf(c), ws, dtypes.float32, TIER, max_len=64, prefill_chunk=16)
  l, scale = m.layers[0], 1 / math.sqrt(hd)
  cache = NeuronKVCache(1, Hkv, hd, P // bs, bs, dtypes.float32, TIER, max_seq_len=P)
  lim = (ctx or P) // rows
  pos = rng.integers(0, lim - bs, rows)   # every block of the step's sequences below the prefix (lowest blocks first)
  for b in range(rows): cache.ensure(b, int(pos[b]) + 1)
  meta = cache.step_meta_rows(rows, rows, np.arange(rows), pos, list(range(rows)))
  cos, sin = (a[pos] for a in rope_rows(*rope_tables(m.cfg, P)))
  pool0, x, d = r(P, 4 * Hkv * hd, sc=1.0), r(rows, D, sc=1.0), r(rows, D, sc=1.0)
  seen: list = []
  real = J.run_item
  monkeypatch.setattr(J, "run_item", lambda si, *a, **k: (seen.append(si), real(si, *a, **k))[1])
  out: dict = {}
  for mode in ("v1", variant):
    for k, v in nk.LAYER_VARIANTS[mode].items(): monkeypatch.setenv(k, v)
    pools = [Tensor(pool0.copy(), device=TIER).realize()]
    cc = cache.bind(pools, [Tensor(meta, device=TIER).realize()])
    cc.layout, cc.ctx = StepLayout(rows, rows, cache.max_blocks), ctx
    xt, dt, ct, st = (Tensor(a, device=TIER).realize() for a in (x, d, cos, sin))
    seen.clear()
    if mode == "v1":
      x1, h = m._add_norm(xt, dt if add else None, l.ln1)
      o = cc.attend_rope(0, m.linear(h, l.wqkv, l.bqkv), ct, st, Hq, scale)
      x2, h2 = m._add_norm(x1, m.linear(o, l.wo, l.bo), l.ln2)
      gu = m.linear(h2, l.wgu)
      dn = m.linear(gu.shrink((None, (0, I))).silu() * gu.shrink((None, (I, 2 * I))), l.wd)
      att = next(s for s in seen if s.ast[0].name.startswith("fz_attention"))
      X, O = att.bufs[1].numpy(), o.numpy()
    else:
      assert (res := cc.layer(0, xt, dt if add else None, l, ct, st, Hq, scale, m.cfg.rms_eps)) is not None
      x2, dn = res
      si = next(s for s in seen if s.ast[0].name.startswith("fz_layer"))
      assert si.ast[0].spec.heads == nk.LAYER_VARIANTS[variant]["FZ_NEURON_LAYER_HEADS"]
      sp_ = si.ast[0].spec
      while not hasattr(sp_, "tile"): sp_ = sp_.base   # N8 (LayerBF16 wraps the LayerSpec): a tiled variant really ran the tiled layer (and its weight columns)
      assert sp_.tile == (nk.LAYER_VARIANTS[variant].get("FZ_NEURON_LAYER_TILE") == "1")
      assert not sp_.tile or sp_.wcols == int(nk.LAYER_VARIANTS[variant].get("FZ_NEURON_LAYER_WCOLS") or nk.LAYER_WCOLS)
      names = si.ast[0].spec.args()
      X, O = si.bufs[names.index("xs")].numpy(), si.bufs[names.index("os")].numpy()
    out[mode] = {"residual": x2.numpy(), "down": dn.numpy(), "pool": pools[0].numpy(), "X": X, "attention": O}
  return {k: (out["v1"][k].reshape(-1), out[variant][k].reshape(-1)) for k in out["v1"]}

LAYER_CASES = {   # D, I, Hq, Hkv, hd, rows, pool slots, block size, prefix (None: the pool), residual term, o_proj bias
  "qwen_1row_512": (896, 4864, 14, 2, 64, 1, 512, 16, None, True, False),
  "qwen_4rows_768_of_1024": (896, 4864, 14, 2, 64, 4, 1024, 16, 768, True, False),
  "odd_shapes": (200, 300, 6, 2, 16, 3, 300, 4, None, True, True),          # D, I not multiples of 128; a ragged pool tile and key chunk
  "first_layer_two_blocks": (200, 300, 6, 2, 16, 2, 700, 4, None, False, True),   # no residual term; two key blocks, the last ragged
  "one_kv_head_5rows": (64, 96, 4, 1, 32, 5, 320, 16, None, True, False),
}

@pytest.mark.parametrize("variant", ["v2", "v2s", "v2a", "v2sa", "v2e", "v2se", "v2sw", "v2swe", "v2swep"])
@pytest.mark.parametrize("case", list(LAYER_CASES))
def test_layer_kernel_matches_the_members(case, variant, monkeypatch):
  """N7 step 3: LayerSpec (one kernel per decoder layer: intermediates in SBUF, attention's exact exp on the transposed
  block, both RoPE and both attention-head layouts) gives the ten member kernels' bits: the residual stream, the down
  projection, every pool row (the KV write), X and the attention output. Rows 1 to 5, pool prefixes 300 to 768 of 1024.
  N7 step 4: early (e: g-major attention rows, qᵀ / oᵀ by transposes, the first key block without the rescale) and
  weight-stationary (w) move rows and swap matmul operands, which BLAS's float32 kernels round differently, so on NKISHIM
  they are checked with the order-independent matmul (FZ_NKISHIM_MM=f64: the members run with it too)."""
  if variant[2:].strip("sa") and TIER == "NKISHIM": monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  for k, (a, b) in _layer_pair(*LAYER_CASES[case], variant, monkeypatch).items():
    if TIER == "NEURON":   # hardware: the share of equal bits (scripts/nki_n7_layer_bench.py decides the serving path on it)
      print(f"layer {case} {variant} {k} on NEURON: {np.mean(a.view(np.int32) == b.view(np.int32)):.6f} bitwise equal, max |diff| {np.abs(a - b).max():.3g}")
      np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
    else: assert np.array_equal(a.view(np.int32), b.view(np.int32)), f"{case} {variant} {k}: max |diff| {np.abs(a - b).max()}"

def test_layer_spec_limits():
  """LayerSpec takes a step when a kv head's attention rows (Hq/Hkv * rows) fit the 128 partitions (stacked heads: in
  32-row quadrants); bigger steps (prefill) keep the members. FZ_NEURON_LAYER selects the path (v2 default)."""
  assert nk.layer_ok(4, 14, 2, 64) and nk.layer_ok(18, 14, 2, 64) and not nk.layer_ok(19, 14, 2, 64) and not nk.layer_ok(128, 14, 2, 64)
  assert nk.layer_ok(4, 14, 2, 64, "stack") and nk.layer_ok(9, 14, 2, 64, "stack") and not nk.layer_ok(10, 14, 2, 64, "stack")
  assert nk.layer_mode() == "v2" and set(nk.LAYER_VARIANTS) == {"v1", "v2", "v2s", "v2a", "v2sa", "v2e", "v2se", "v2sw", "v2swe", "v2swep", "v2set", "v2et", "v2set2", "v2set4"}
  assert nk.LAYER_BITWISE == {"v1", "v2", "v2s", "v2a", "v2sa", "v2e", "v2se", "v2set", "v2et", "v2set2", "v2set4"}
  assert all(set(e) == set(nk.LAYER_ENV) for v, e in nk.LAYER_VARIANTS.items() if v != "v1")
  with pytest.raises(nk.NKIUnsupported): nk.LayerSpec(19, 896, 14, 2, 64, 4864, 512, 1e-6, 0.125, 0).render()

# ---------------------------------------------------------------- the engine on NKISHIM vs CPU

@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> str:
  return tiny_model_dir(str(tmp_path_factory.mktemp("tiny_qwen")))

PROMPTS = [[1, 2, 3, 4, 5, 6, 7], [9, 8, 7], [5] * 20, list(range(30, 60))]

def _serve(path: str, device: str, num_blocks: int = 64, max_num_seqs: int = 4) -> tuple[dict, object]:
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine, SamplingParams
  m = load_model(path, dtype=dtypes.float32, device=device, max_len=128, prefill_chunk=128)
  eng = LLMEngine(m, num_blocks=num_blocks, block_size=16, max_num_seqs=max_num_seqs, max_prefill_tokens=32, max_model_len=128, decode_beam=0)
  rids = [eng.add_request(p, SamplingParams(max_tokens=10, ignore_eos=True)) for p in PROMPTS]
  out = {r: [] for r in rids}
  while eng.has_unfinished():
    for o in eng.step(): out[o.request_id] += o.new_token_ids
  return out, eng

def _decode_launches(eng) -> int:
  f = eng._decodes[max(eng._decodes)]
  c = next(j.captured for j in [f.jit, *f.ctx_jits.values()] if j.captured is not None)   # N6: a jit per pool prefix
  return sum(not it.const for it in c.items)

@pytest.mark.parametrize("fuse", ["0", "1", "2"])
@pytest.mark.parametrize("num_blocks", [64, 6], ids=["roomy", "preempting"])
def test_engine_tokens_match_cpu(tiny, num_blocks, fuse, monkeypatch):
  monkeypatch.setenv("FZ_NEURON_FUSE", fuse)
  monkeypatch.setenv("FZ_NEURON_HEAD_FUSE", "0")   # the head its own launch (N8 default: fused, test_head_fused_into_the_last_layers)
  v2 = nk.head_argmax()
  cpu, _ = _serve(tiny, "CPU", num_blocks)
  shim, eng = _serve(tiny, TIER, num_blocks)
  assert shim == cpu
  assert type(eng.cache).__name__ == "NeuronKVCache" and eng.device_argmax
  if num_blocks == 6: assert eng.scheduler.num_preemptions > 0
  # FZ_NEURON_FUSE=0 (N5): per layer norm, qkv, rope (q|k|v|1), KV write, attention, o_proj, o bias (this checkpoint has
  # one), norm, gate_up, silu*mul, down; per step the one-hot slot matrix, final norm, lm_head, argmax (N7 v2/v3: 1, v1: 2).
  # N6: one fused kernel per FZ_NEURON_FUSE layers, then the head (final norm, lm_head, argmax) as one more. N7 step 3
  # (FZ_NEURON_LAYER=v2, default): a layer is one LayerSpec launch
  per = 11 if nk.layer_mode() == "v1" else 1
  assert _decode_launches(eng) == {"0": 2 * per + 5 - v2, "1": 2 + 1, "2": 1 + 1}[fuse]

def test_pool_prefix_buckets(tiny, monkeypatch):
  """N6: each step's KV write and attention touch only a pool prefix that holds every block of the step's sequences
  (FZ_NEURON_CTX_MIN=16 here, so the steps move through several prefixes as the sequences grow): CPU's tokens, and
  FZ_NEURON_CTX=0 runs every step on the whole pool. Without prefill-first, so the first decodes run before the longer
  prompts are in (with it every decode step holds all four sequences)."""
  monkeypatch.setenv("FZ_NEURON_CTX_MIN", "16")
  monkeypatch.setenv("FZ_PREFILL_FIRST", "0")
  cpu, _ = _serve(tiny, "CPU")
  got, eng = _serve(tiny, TIER)
  assert got == cpu
  assert eng.cache.ctx_buckets() == [16, 32, 64, 128, 256, 512, 1024]
  used = {c for f in eng._decodes.values() for c, j in f.ctx_jits.items() if j.cnt}
  assert len(used) >= 2 and max(used) < 1024
  monkeypatch.setenv("FZ_NEURON_CTX", "0")
  got, eng = _serve(tiny, TIER)
  assert got == cpu and eng.cache.ctx_buckets() == [1024] and not any(f.ctx_jits for f in eng._decodes.values())

def test_prefill_pool_prefix_of_its_own(tiny, monkeypatch):
  """I75: FZ_NEURON_PREFILL_CTX sets the prefill forwards' pool prefixes (the same forms as FZ_NEURON_CTX), so decode can
  stay on the whole pool (FZ_NEURON_CTX=0: one decode NEFF set) while a prompt's attention scans only its prefix; unset,
  the prefill follows FZ_NEURON_CTX. CPU's tokens."""
  from fusezap.serve.neuron_kv import NeuronKVCache
  monkeypatch.delenv("FZ_NEURON_PREFILL_CTX", raising=False)
  c = NeuronKVCache(1, 2, 16, 256, 16, dtypes.float32, TIER)
  monkeypatch.setenv("FZ_NEURON_CTX", "0")
  assert c.ctx_buckets() == c.ctx_buckets(prefill=True) == [4096]
  monkeypatch.setenv("FZ_NEURON_PREFILL_CTX", "1")
  assert c.ctx_buckets() == [4096] and c.ctx_buckets(prefill=True) == [512, 1024, 2048, 4096]
  monkeypatch.setenv("FZ_NEURON_PREFILL_CTX", "256,512")
  assert c.ctx_buckets(prefill=True) == [256, 512, 4096]
  monkeypatch.setenv("FZ_NEURON_PREFILL_CTX", "1"); monkeypatch.setenv("FZ_NEURON_CTX_MIN", "16"); monkeypatch.setenv("FZ_PREFILL_FIRST", "0")
  cpu, _ = _serve(tiny, "CPU")
  got, eng = _serve(tiny, TIER)
  assert got == cpu and eng.cache.ctx_buckets() == [1024] and not any(f.ctx_jits for f in eng._decodes.values())
  used = {c for f in eng._prefills.values() for c, j in f.ctx_jits.items() if j.cnt}
  assert used and max(used) < 1024, used

def test_lowest_blocks_keep_the_pool_prefix_small(tiny, monkeypatch):
  """N7 serve: after requests that finished at different times, a lone request gets the lowest block ids
  (FZ_NEURON_LOW_BLOCKS, default on), so its steps run at the smallest pool prefix; FZ_NEURON_LOW_BLOCKS=0 (LIFO, as
  before) handed it blocks above that. Tokens equal CPU's either way, and prefill-first (on by default on NEURON) ran."""
  from fusezap.serve.engine import SamplingParams
  monkeypatch.setenv("FZ_NEURON_CTX_MIN", "16")
  cpu, _ = _serve(tiny, "CPU")
  lone = {}
  for low in ("1", "0"):
    monkeypatch.setenv("FZ_NEURON_LOW_BLOCKS", low)
    got, eng = _serve(tiny, TIER)
    assert got == cpu and eng.scheduler.num_prefill_first > 0
    rid = eng.add_request(PROMPTS[1], SamplingParams(max_tokens=10, ignore_eos=True))
    eng.step()
    blocks = list(eng.cache.tables[rid])
    toks = list(eng.seqs[rid].output)
    while eng.has_unfinished(): eng.step()
    lone[low] = (blocks, toks + [t for t in eng.seqs[rid].output[len(toks):]])
  assert lone["1"][0] == [0] and max(lone["0"][0]) > 0
  assert lone["1"][1] == lone["0"][1]

def test_fused_forward_is_bit_identical(tiny, monkeypatch):
  """N6: a fused replay (one kernel per layer, the head as one more) gives the unfused replay's logits bit for bit, prefill
  and decode (Llama.prefill / decode on NEURON: NeuronSteps' jitted ragged forwards, replayed from the third call)."""
  from fusezap.serve.models import load_model
  out = {}
  monkeypatch.setenv("FZ_NEURON_HEAD_FUSE", "0")
  for fuse in ("0", "1"):
    monkeypatch.setenv("FZ_NEURON_FUSE", fuse)
    m = load_model(tiny, dtype=dtypes.float32, device=TIER, max_len=128, prefill_chunk=16)
    lg = [m.prefill([list(range(5, 21))], chunk=16)] + [m.decode([[3 + i]]) for i in range(4)]
    m.reset()
    lg += [m.prefill([list(range(40, 56))], chunk=16)] + [m.decode([[7 + i]]) for i in range(2)]
    out[fuse] = lg
    c = m._neuron_steps.jits[(1, 1)].captured
    assert sum(not it.const for it in c.items) == (2 * (11 if nk.layer_mode() == "v1" else 1) + 3 if fuse == "0" else 3)
  for a, b in zip(out["0"], out["1"]):
    if TIER == "NKISHIM": assert np.array_equal(a, b)
    else:   # hardware: the fused NEFF is compiled as a whole (engine choice per copy may differ); report the share of equal bits
      print(f"fused vs unfused on {TIER}: {np.mean(a == b):.4f} of the logits bitwise equal, max |diff| {np.abs(a - b).max():.3g}")
      np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize("fuse", ["1", "2", "24"])
def test_head_fused_into_the_last_layers(tiny, monkeypatch, fuse):
  """N7 step 4: FZ_NEURON_HEAD_FUSE=1 drops the boundary before the head, so the last layers and the head (final norm,
  lm_head, argmax) are one kernel: one launch fewer per forward, the same logits bit for bit (prefill and decode) and the
  engine's tokens the same as CPU's. FZ_NEURON_FUSE above the layer count (24 here, 2 layers) fuses the whole forward."""
  from fusezap.serve.models import load_model
  monkeypatch.setenv("FZ_NEURON_FUSE", fuse)
  out, n = {}, {}
  for hf in ("0", "1"):
    monkeypatch.setenv("FZ_NEURON_HEAD_FUSE", hf)
    m = load_model(tiny, dtype=dtypes.float32, device=TIER, max_len=128, prefill_chunk=16)
    out[hf] = [m.prefill([list(range(5, 21))], chunk=16)] + [m.decode([[3 + i]]) for i in range(4)]
    n[hf] = sum(not it.const for it in m._neuron_steps.jits[(1, 1)].captured.items)
  assert n == {"0": {"1": 3, "2": 2, "24": 2}[fuse], "1": {"1": 2, "2": 1, "24": 1}[fuse]}
  for a, b in zip(out["0"], out["1"]): assert np.array_equal(a.view(np.int32), b.view(np.int32))
  cpu, _ = _serve(tiny, "CPU")
  got, eng = _serve(tiny, TIER)
  assert got == cpu and _decode_launches(eng) == {"1": 2, "2": 1, "24": 1}[fuse]

def test_n8_defaults(tiny, monkeypatch):
  """N8: trn1 step 4's choice is the code default: layer v2se (heads stacked, RoPE slices, mm=mv, early, no prefetch), head
  v2 (HeadSpec W 1024, 1 DMA per block), FZ_NEURON_FUSE=24 with FZ_NEURON_FUSE_MAX=44 and FZ_NEURON_HEAD_FUSE=1: the tiny
  model's decode forward is one launch, LayerSpec early layers and HeadSpec inside it, CPU's tokens. The weight-stationary
  forms stay off. A step too wide for stacked heads uses one tile per kv head instead of the members."""
  from fusezap.device import neuron
  for k in nk.LAYER_ENV + ("FZ_NEURON_HEAD", "FZ_NEURON_HEAD_W", "FZ_NEURON_HEAD_DMAS", "FZ_NEURON_FUSE", "FZ_NEURON_FUSE_MAX",
                           "FZ_NEURON_HEAD_FUSE", "FZ_NEURON_DTYPE"): monkeypatch.delenv(k, raising=False)
  assert nk.layer_mode() == "v2" and nk.layer_variant() == {"heads": "stack", "rope": "slice", "mm": "mv", "early": True, "pf": 0, "tile": "auto", "wcols": 2048}
  assert nk.head_mode() == "v2" and (nk.HEAD_W, nk.HEAD_DMAS) == (1024, 1)   # (module constants: read at import)
  assert neuron.fuse_layers() == 24 and neuron.fuse_max() == 44 and neuron.head_fuse()
  assert {v for v in nk.LAYER_VARIANTS if "w" in v[2:]} == {"v2sw", "v2swe", "v2swep"} and not nk.LAYER_VARIANTS["v2se"]["FZ_NEURON_LAYER_MM"] == "ws"
  cpu, _ = _serve(tiny, "CPU")
  got, eng = _serve(tiny, TIER)
  assert got == cpu and _decode_launches(eng) == 1 and eng.model.lm_head_t is not None
  c = eng._decodes[max(eng._decodes)]
  names = [p[0] for j in [c.jit, *c.ctx_jits.values()] if j.captured is not None for it in j.captured.items if not it.const
           for p in getattr(getattr(it.si.ast[0], "spec", None), "parts", ())]
  assert any(n.startswith("fz_layerse_") for n in names) and any(n.startswith("fz_head_") for n in names)
  monkeypatch.setenv("FZ_NEURON_FUSE_MAX", "0")
  assert neuron.fuse_max() == 0

def test_stacked_heads_fall_back_to_each(tmp_path, monkeypatch):
  """N8: heads=stack (the default) needs every kv head's rows padded to 32 in one 128-partition tile; a step with more rows
  (7 query heads per kv head as Qwen2.5-0.5B, 10 rows: 2 x 96 > 128) runs LayerSpec with one tile per kv head (70 rows)
  instead of the members, with v1's logits bit for bit."""
  from test_tp import tiny_model_dir
  from fusezap.serve.models import load_model
  assert not nk.layer_ok(10, 14, 2, 64, "stack") and nk.layer_ok(10, 14, 2, 64, "each") and nk.layer_ok(9, 14, 2, 64, "stack")
  p = tiny_model_dir(str(tmp_path / "g7"), seed=5, cfg={"hidden_size": 64, "intermediate_size": 96, "num_attention_heads": 14,
                                                        "num_key_value_heads": 2, "head_dim": 16})
  for k in nk.LAYER_ENV: monkeypatch.delenv(k, raising=False)
  monkeypatch.setenv("FZ_NEURON_FUSE", "1"); monkeypatch.setenv("FZ_NEURON_HEAD_FUSE", "0")
  for rows in (9, 10):
    out = {}
    for v in ("v1", ""):
      if v: monkeypatch.setenv("FZ_NEURON_LAYER", v)
      else: monkeypatch.delenv("FZ_NEURON_LAYER", raising=False)
      m = load_model(p, dtype=dtypes.float32, device=TIER, max_len=64, prefill_chunk=16, max_batch=rows)
      out[v] = [m.prefill([[3 + b, 5 + b, 7] for b in range(rows)], chunk=16)] + [m.decode([[3 + i + b] for b in range(rows)]) for i in range(3)]
      names = []   # every decode launch's kernel, a fused kernel's members included
      for it in m._neuron_steps.jits[(rows, rows)].captured.items:
        if not it.const and (sp := getattr(it.si.ast[0], "spec", None)) is not None: names += [q[0] for q in sp.parts] if hasattr(sp, "parts") else [sp.name]
      want = "fz_layerse_" if rows == 9 else "fz_layere_"   # stacked (s) at 9 rows, one tile per kv head at 10; early (e)
      assert sum(n.startswith("fz_layer") for n in names) == sum(n.startswith(want) for n in names) == (0 if v else m.cfg.n_layers), names
    for a, b in zip(out["v1"], out[""]): assert np.array_equal(a.view(np.int32), b.view(np.int32))

@pytest.mark.parametrize("head", ["v2", "v3", "v4"])
def test_head_model_is_bit_identical(tiny, monkeypatch, head):
  """N7: FZ_NEURON_HEAD=v2 (HeadSpec on the relaid lm_head, ArgmaxSpec) and v3 (the default: render_matmul on
  kmajor(lm_head), ArgmaxSpec) vs v1 (render_matmul, the composed argmax) through the model: prefill and decode logits bit
  for bit (fused, FZ_NEURON_FUSE=1), and the engine's greedy tokens (device argmax) the same."""
  from fusezap.serve.models import load_model
  monkeypatch.setenv("FZ_NEURON_FUSE", "1")
  if head == "v4" and TIER == "NKISHIM": monkeypatch.setenv("FZ_NKISHIM_MM", "f64")   # operands swapped (test_head_v4_weight_stationary)
  out, toks = {}, {}
  for hd in ("v1", head):
    monkeypatch.setenv("FZ_NEURON_HEAD", hd)
    m = load_model(tiny, dtype=dtypes.float32, device=TIER, max_len=128, prefill_chunk=16)
    assert (m.lm_head_t is not None) == (hd in ("v2", "v4")) and (m.lm_head_k is not None) == (hd not in ("v2", "v4"))
    out[hd] = [m.prefill([list(range(5, 21))], chunk=16)] + [m.decode([[3 + i]]) for i in range(4)]
    toks[hd], _ = _serve(tiny, TIER)
  for a, b in zip(out["v1"], out[head]):
    if TIER == "NEURON": print(f"head {head} vs v1 on NEURON: {np.mean(a.view(np.int32) == b.view(np.int32)):.4f} of the logits bitwise equal")
    assert np.array_equal(a.view(np.int32), b.view(np.int32))
  assert toks["v1"] == toks[head]

@pytest.mark.parametrize("variant,fuse,batch,mm", [("v2", "1", 1, ""), ("v2", "0", 2, ""), ("v2s", "1", 3, ""), ("v2sa", "2", 1, ""),
                                                   ("v2se", "1", 1, ""), ("v2se", "2", 3, "f64"), ("v2swe", "1", 2, "f64"), ("v2sw", "2", 1, "f64")])
def test_layer_model_is_bit_identical(tiny, monkeypatch, variant, fuse, batch, mm):
  """N7 step 3: FZ_NEURON_LAYER=v2 (each decoder layer one LayerSpec launch, fused FZ_NEURON_FUSE layers at a time) vs
  v1 (the ten members) through the model: prefill (16 and 32 rows: LayerSpec; 48 rows: the members) and decode logits bit
  for bit at 1 to 3 rows, and the engine's greedy tokens (four requests, two of them preempting) the same as v1's and CPU's.
  N7 step 4: early (e) and weight-stationary (w) too; mm=f64: both paths with the order-independent shim matmul (rows moved
  or operands swapped: see test_layer_kernel_matches_the_members)."""
  from fusezap.serve.models import load_model
  monkeypatch.setenv("FZ_NEURON_FUSE", fuse)
  if mm and TIER == "NKISHIM": monkeypatch.setenv("FZ_NKISHIM_MM", mm)
  out, toks, cpu = {}, {}, _serve(tiny, "CPU")[0]
  for v in ("v1", variant):
    for k, val in nk.LAYER_VARIANTS[v].items(): monkeypatch.setenv(k, val)
    m = load_model(tiny, dtype=dtypes.float32, device=TIER, max_len=128, prefill_chunk=16, max_batch=batch)
    out[v] = [m.prefill([list(range(5 + b, 21 + b)) for b in range(batch)], chunk=16)] + [m.decode([[3 + i + b] for b in range(batch)]) for i in range(4)]
    names = []   # every decode launch's kernel, a fused kernel's members included
    for it in m._neuron_steps.jits[(batch, batch)].captured.items:
      if not it.const and (sp := getattr(it.si.ast[0], "spec", None)) is not None: names += [p[0] for p in sp.parts] if hasattr(sp, "parts") else [sp.name]
    assert sum(n.startswith("fz_layer") for n in names) == (2 if v != "v1" else 0)
    toks[v], _ = _serve(tiny, TIER, num_blocks=6 if batch == 2 else 64)
  for a, b in zip(out["v1"], out[variant]):
    if TIER == "NEURON": print(f"layer {variant} vs v1 on NEURON: {np.mean(a.view(np.int32) == b.view(np.int32)):.4f} of the logits bitwise equal")
    assert np.array_equal(a.view(np.int32), b.view(np.int32))
  assert toks["v1"] == toks[variant] and (batch == 2 or toks[variant] == cpu)

def test_pool_prefix_mid_buckets(tiny, monkeypatch):
  """N7 step 3: FZ_NEURON_CTX=mid adds 3/2 of each power of two (768 between 512 and 1024); a list sets the prefixes
  outright. CPU's tokens, and the engine ran some steps at a 3/2 prefix."""
  from fusezap.serve.neuron_kv import NeuronKVCache
  c = NeuronKVCache(1, 2, 16, 256, 16, dtypes.float32, TIER)
  for mode, want in (("mid", [512, 768, 1024, 1536, 2048, 3072, 4096]), ("1", [512, 1024, 2048, 4096]), ("512,768,1024", [512, 768, 1024, 4096]), ("0", [4096])):
    monkeypatch.setenv("FZ_NEURON_CTX", mode)
    assert c.ctx_buckets() == want, mode
  monkeypatch.setenv("FZ_NEURON_CTX", "mid"); monkeypatch.setenv("FZ_NEURON_CTX_MIN", "32"); monkeypatch.setenv("FZ_PREFILL_FIRST", "0")
  cpu, _ = _serve(tiny, "CPU")
  got, eng = _serve(tiny, TIER)
  assert got == cpu and eng.cache.ctx_buckets() == [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
  used = {c for f in eng._decodes.values() for c, j in f.ctx_jits.items() if j.cnt}
  assert used & {48, 96, 192}

def test_llama_prefill_decode_match_cpu(tiny):
  """Llama.prefill / decode / generate on NEURON go through the ragged forward (positions as data, NeuronSteps)."""
  from fusezap.serve.models import load_model
  from fusezap.serve.sampling import Sampler
  out = {}
  for dev in ("CPU", TIER):
    m = load_model(tiny, dtype=dtypes.float32, device=dev, max_len=128, prefill_chunk=32)
    toks = list(m.generate(list(range(1, 40)), 8, Sampler(), stop=[]))
    m.reset()
    lg = m.prefill([list(range(5, 25))], chunk=16)
    out[dev] = (toks, lg, m.decode([[3]]))
  assert out["CPU"][0] == out[TIER][0]
  for i in (1, 2): np.testing.assert_allclose(out[TIER][i], out["CPU"][i], atol=2e-5, rtol=0)
  assert len(m._neuron_steps.jits) == 3   # prefill buckets 16 and 32 (generate's 39-token prompt: 32 + 7 -> 16), decode

@pytest.mark.skipif(TIER != "NKISHIM", reason="the fake runtime: Mac only")
def test_engine_on_the_persistent_path(tiny, neuron_dev, nrt, monkeypatch):
  """The engine on NEURON's persistent path (fake nrtpy running the kernels through NKISHIM): CPU's tokens, every NEFF
  loaded once, and each forward's launches sent to the runtime worker as one batch."""
  dev, nki = neuron_dev
  cpu, _ = _serve(tiny, "CPU")
  calls0 = dict(dev.allocator.calls)
  got, eng = _serve(tiny, "NEURON")
  assert got == cpu
  c = {k: v - calls0.get(k, 0) for k, v in dev.allocator.calls.items()}
  steps = eng.stats["steps"]
  assert c["exec"] >= steps * 3 and c["batches"] <= c["read"] + c["write"] + c["load"] + 1   # flushed only at host traffic (N6: 3 fused launches per decode)
  assert nrt.session.execs == c["exec"] and nki.neffs == c["load"]
  monkeypatch.setenv("FZ_NEURON_BATCH", "0")
  from fusezap.device.neuron import NeuronAllocator
  assert not NeuronAllocator(worker=object()).batch

@pytest.mark.skipif(TIER != "NKISHIM", reason="the fake runtime: Mac only")
@pytest.mark.parametrize("fast,dargmax", [("1", "1"), ("0", "1"), ("1", "0")], ids=["hostfast", "n6b_host", "logits_read"])
def test_host_paths_on_the_persistent_path(tiny, neuron_dev, nrt, monkeypatch, fast, dargmax):
  """N7: FZ_NEURON_HOSTFAST (writes queued with the launches, the read taking them along, NeffReplay launches, inputs
  uploaded straight into device buffers), N6b's host path (=0) and FZ_NEURON_DEVICE_ARGMAX=0 (the logits read back,
  np.argmax on the host) all give CPU's tokens; the fast path is one runtime-worker round trip per replayed decode step."""
  from fusezap.serve.engine import SamplingParams
  dev, nki = neuron_dev
  monkeypatch.setenv("FZ_NEURON_HOSTFAST", fast); monkeypatch.setenv("FZ_NEURON_DEVICE_ARGMAX", dargmax)
  dev.allocator.fast = fast == "1"   # the allocator was made by the fixture, before the env
  cpu, _ = _serve(tiny, "CPU")
  got, eng = _serve(tiny, "NEURON")
  assert got == cpu and eng.device_argmax == (dargmax == "1") and eng.upload == (fast == "1")
  al, trips = dev.allocator, []
  rid = eng.add_request(PROMPTS[0], SamplingParams(max_tokens=8, ignore_eos=True))
  while eng.has_unfinished():
    t0 = al.trips; eng.step(); trips.append(al.trips - t0)
  assert eng.seqs[rid].output == cpu[0][:8]
  assert trips[-3:] == ([1, 1, 1] if fast == "1" else [6, 6, 6])   # N6b: 4 input writes, the launches, the token read

@pytest.mark.skipif(TIER != "NKISHIM", reason="the fake runtime: Mac only")
def test_profile_and_unfusable_fallback(tiny, neuron_dev, nrt, monkeypatch, capsys):
  """N6 on the persistent path: FZ_NEURON_PROFILE's per-kernel breakdown (fused kernels by name, execute time, nrtpy's
  device-time bench), and a fused kernel that neuronx-cc rejects falls back to its members, launched one by one."""
  from fusezap.device import neuron
  dev, nki = neuron_dev
  cpu, _ = _serve(tiny, "CPU")
  dev.allocator.profile = True
  got, eng = _serve(tiny, "NEURON")
  assert got == cpu
  rep = neuron.neuron_profile(dev.allocator, iters=3)
  fused = {k: v for k, v in rep["kernels"].items() if k.startswith("fz_fused")}
  assert fused and all(v["launches"] > 0 and v["device_ms"] is not None for v in fused.values()) and rep["batches"] > 0
  real = nki.call
  monkeypatch.setattr(nki, "call", lambda req: {"ok": False, "err": "NCC_FAKE: too big"} if req.get("name", "").startswith("fz_fused") else real(req))
  dev._cache.clear(); monkeypatch.setenv("FZ_NEFF_CACHE", str(neuron.neff_cache_dir()) + "_2")
  got, eng = _serve(tiny, "NEURON")
  assert got == cpu and dev._unfusable and "launching its members separately" in capsys.readouterr().err
@pytest.mark.skipif(TIER != "NKISHIM", reason="the fake runtime: Mac only")
def test_head_bench_script_on_the_persistent_path(neuron_dev, nrt, monkeypatch, tmp_path, capsys):
  """scripts/nki_n7_head_bench.py (the N7 step 2 hardware phase) on the fake runtime: each variant's head is one fused
  launch, profiled with a device time, and v2's logits and tokens equal v1's."""
  import importlib.util, sys
  spec = importlib.util.spec_from_file_location("hb", os.path.join(ROOT, "scripts/nki_n7_head_bench.py"))
  hb = importlib.util.module_from_spec(spec); spec.loader.exec_module(hb)
  monkeypatch.setenv("FZ_NEURON_HEAD", "v3"); monkeypatch.setenv("FZ_NEURON_FUSE", "1")   # main() sets both: restored after
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64")   # v4 swaps the matmul operands (test_head_v4_weight_stationary)
  monkeypatch.setattr(sys, "argv", ["hb", "--out", str(tmp_path), "--rows", "1,4", "--vocab", "1500", "--hidden", "200", "--replays", "2",
                                    "--variants", "v1,v3,v2,v2:512:1,v2-logits,v4:512:1"])
  hb.main()
  res = json.load(open(tmp_path / "head_bench.json"))["results"]
  assert len(res) == 12 and all(r["launches"] == 1 and r["device_ms"] is not None and r["logits_bitwise_vs_v1"] == 1.0 for r in res)
  assert all(r["argmax_ok"] and r["tokens_vs_v1"] for r in res if not r["variant"].endswith("-logits"))
  assert "head_bench rows 4 v2:512:1" in capsys.readouterr().out

@pytest.mark.skipif(TIER != "NKISHIM", reason="the fake runtime: Mac only")
def test_layer_bench_script_on_the_persistent_path(neuron_dev, nrt, monkeypatch, tmp_path, capsys):
  """scripts/nki_n7_layer_bench.py (the N7 step 3 hardware phase) on the fake runtime: per rows, pool size and layer path
  the timed layer is one launch with a device time (v1: the members fused, v2*: one LayerSpec), every variant's logits,
  tokens and pool equal v1's, "v2@<ctx>" runs at that pool size only, and --ablate times each v1 member alone."""
  import importlib.util, sys
  spec = importlib.util.spec_from_file_location("lb", os.path.join(ROOT, "scripts/nki_n7_layer_bench.py"))
  lb = importlib.util.module_from_spec(spec); spec.loader.exec_module(lb)
  for k in ("FZ_NEURON_FUSE", "FZ_NEURON_HEAD_FUSE", "FZ_NEURON_DTYPE", "FZ_NEURON_BF16_ATTN") + nk.LAYER_ENV:   # main() sets them: restored after
    if k in os.environ: monkeypatch.setenv(k, os.environ[k])
    else: monkeypatch.setenv(k, ""); monkeypatch.delenv(k)   # absent again after
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64")   # v2swe: rows moved, operands swapped (test_layer_kernel_matches_the_members)
  monkeypatch.setattr(sys, "argv", ["lb", "--out", str(tmp_path), "--rows", "1,2", "--ctx", "64,96", "--hidden", "64", "--inter", "96",
                                    "--heads", "4", "--kv", "2", "--hd", "16", "--replays", "2", "--variants", "v1,v2,v2s@64,v2swe@96", "--ablate"])
  lb.main()
  res = json.load(open(tmp_path / "layer_bench.json"))["results"]
  assert [(r["rows"], r["ctx"], r["variant"]) for r in res] == [(1, 64, "v1"), (1, 64, "v2"), (1, 64, "v2s"), (1, 96, "v1"), (1, 96, "v2"),
                                                                (1, 96, "v2swe"), (2, 64, "v1"), (2, 64, "v2"), (2, 64, "v2s"), (2, 96, "v1"),
                                                                (2, 96, "v2"), (2, 96, "v2swe")]
  assert [r["bitwise_by_construction"] for r in res[:6]] == [True] * 5 + [False]
  assert all(r["launches"] == 4 and r["device_ms"] is not None and r["logits_bitwise_vs_v1"] == 1.0 and r["tokens_vs_v1"]
             and r["pool_bitwise_vs_v1"] == 1.0 for r in res)
  assert all(r["kernel"].startswith("fz_fused10_" if r["variant"] == "v1" else "fz_layer") for r in res)
  assert len(res[0]["ablation"]) == 10 and all(isinstance(v, float) for v in res[0]["ablation"].values())
  assert "layer_bench rows 2 ctx 96 v2" in capsys.readouterr().out

@pytest.mark.skipif(TIER != "NKISHIM", reason="the fake runtime: Mac only")
def test_probe_script_on_the_persistent_path(neuron_dev, nrt, tmp_path, capsys):
  """scripts/nki_n7_probe.py (the N7 step 4 microbenchmarks) on the fake runtime: every probe kernel (TensorE matmul in both
  operand forms, bf16, transposes, the DMA patterns) compiles, launches and gets a device time at both sizes, and the
  per-instruction cost is derived from the difference."""
  import importlib.util, sys
  spec = importlib.util.spec_from_file_location("pr", os.path.join(ROOT, "scripts/nki_n7_probe.py"))
  pr = importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)
  old = sys.argv
  sys.argv = ["pr", "--out", str(tmp_path), "--quick", "--iters", "2"]
  try: pr.main()
  finally: sys.argv = old
  d = json.load(open(tmp_path / "probe.json"))
  assert not d["errors"] and len(d["results"]) == 13
  assert all(None not in r["device_ms"] and "us_per" in r for r in d["results"])
  assert all("gb_per_s" in r for r in d["results"] if r["probe"].startswith("dma"))
  assert "probe ws m=1" in capsys.readouterr().out

def test_probe_kernels_on_the_shim():
  """N7 step 4 ProbeSpec kernels compute what they say on NKISHIM: mv / ws accumulate n products of the resident tiles (ws
  is the transposed form), tr leaves the last transpose, dma reads column 0 of every block."""
  from fusezap.device import nkishim as S
  rng = np.random.default_rng(3)
  for sp in (nk.ProbeSpec("mv", 10, 2), nk.ProbeSpec("ws", 10, 3), nk.ProbeSpec("tr", 5, 4), nk.ProbeSpec("dma", 3, f=512, mode="col4"),
             nk.ProbeSpec("dma", 3, f=1024, mode="run2k"), nk.ProbeSpec("mv", 4, 1, dt="bfloat16")):
    inp = (rng.standard_normal(sp.in_size()) * 0.1).astype(np.float32)
    out = np.zeros(sp.out_size(), np.float32)
    S.execute(sp.render(), [out, inp])
    t = [inp[j * 128 * 512:(j + 1) * 128 * 512].reshape(128, 512).astype(np.float64) for j in range(8)] if sp.kind != "dma" else []
    if sp.kind == "mv" and sp.dt == "float32":
      v = t[0][:, :sp.m]
      np.testing.assert_allclose(out[:sp.m * 512].reshape(sp.m, 512), sum(v.T @ t[r % 8] for r in range(sp.n)), rtol=1e-4, atol=1e-5)
    elif sp.kind == "ws":
      v = t[0][:, :sp.m]
      want = sum(t[(r // 4) % 8][:, (r % 4) * 128:(r % 4 + 1) * 128].T @ v for r in range(sp.n))
      np.testing.assert_allclose(out[:128 * sp.m].reshape(128, sp.m), want, rtol=1e-4, atol=1e-5)
    elif sp.kind == "tr":   # the last transpose's source: n - 1 = 4 is even -> the copied tile (rows 0..m of t0)
      np.testing.assert_array_equal(out[:128 * sp.m].reshape(128, sp.m), t[0][:sp.m, :128].T.astype(np.float32))
    elif sp.kind == "dma":
      blocks = inp[:sp.n * 128 * sp.f].reshape(sp.n, 128, sp.f)
      q = sp.f // 4 if sp.mode == "col4" else 512
      np.testing.assert_array_equal(out[:128 * sp.n].reshape(128, sp.n), blocks[:, :, (sp.f // q - 1) * q].T)
    else: assert np.isfinite(out).all() and np.abs(out).sum() > 0

def test_step4_choices(tmp_path):
  """scripts/nki_n7_step4_choose.py: the fastest variant bitwise == v1 everywhere wins (a faster non-bitwise one does not,
  and is the 24-prompt candidate); the head choice; the fuse choice needs fz_fuse4's c1 tokens."""
  import subprocess, sys
  L = lambda v, r, c, ms, bits=1.0: {"variant": v, "rows": r, "ctx": c, "device_ms": ms, "logits_bitwise_vs_v1": bits, "pool_bitwise_vs_v1": bits, "tokens_vs_v1": True}
  layer = [L("v1", 1, 512, 0.40), L("v1", 4, 512, 0.46), L("v2s", 1, 512, 0.318), L("v2s", 4, 512, 0.329), L("v2se", 1, 512, 0.30), L("v2se", 4, 512, 0.31),
           L("v2se", 1, 1024, 0.34), L("v2sw", 1, 512, 0.25, 0.97), L("v2sw", 4, 512, 0.26, 0.98)]
  H = lambda v, r, ms, bits=1.0, ok=True: {"variant": v, "rows": r, "device_ms": ms, "logits_bitwise_vs_v1": bits, "argmax_ok": ok}
  head = [H("v1-logits", 1, 2.15), H("v3", 1, 2.17), H("v3", 4, 2.25), H("v2:1024:1", 1, 2.06), H("v2:1024:1", 4, 2.13), H("v4:1024:1", 1, 1.5, 0.99),
          H("v4:1024:1", 4, 1.6, 0.99)]
  json.dump({"results": layer}, open(tmp_path / "layer_bench.json", "w")); json.dump({"results": head}, open(tmp_path / "head_bench.json", "w"))
  run = lambda *a: subprocess.run([sys.executable, os.path.join(ROOT, "scripts/nki_n7_step4_choose.py"), *a], capture_output=True, text=True, cwd=ROOT)
  r = run("layer", str(tmp_path))
  assert r.stdout.split() == [f"{k}={v}" for k, v in nk.LAYER_VARIANTS["v2se"].items()] and "v2sw" in r.stderr and "not eligible" in r.stderr
  assert run("head", str(tmp_path)).stdout.strip() == "FZ_NEURON_HEAD=v2 FZ_NEURON_HEAD_W=1024 FZ_NEURON_HEAD_DMAS=1"
  c = run("candidate", str(tmp_path)).stdout
  assert "FZ_NEURON_LAYER_MM=ws" in c and "FZ_NEURON_HEAD=v4" in c
  layer[4]["pool_bitwise_vs_v1"] = 0.5   # v2se not bitwise in one case: v2s
  json.dump({"results": layer}, open(tmp_path / "layer_bench.json", "w"))
  assert "FZ_NEURON_LAYER_HEADS=stack" in run("layer", str(tmp_path)).stdout and "FZ_NEURON_LAYER_EARLY=0" in run("layer", str(tmp_path)).stdout
  S = lambda tp, ids: {"c1": {"tpot_ms_median": tp, "ids": ids}, "launches": {"decode_1_ctx512": {"launches": 7}}}
  for lab, tp, ids in (("fuse4", 11.8, [1, 2]), ("fuse4h", 11.6, [1, 2]), ("fuse8h", 11.0, [1, 3]), ("fuse24h", 11.2, [1, 2])):
    json.dump(S(tp, ids), open(tmp_path / f"serve_{lab}.json", "w"))
  r = run("fuse", str(tmp_path), "fuse4", "fuse4h", "fuse8h", "fuse24h", "fuse12h")
  assert r.stdout.strip() == "fuse24h" and "fuse12h      no result" in r.stderr

def test_layer_cost_model_script():
  """scripts/nki_layer_cost.py: the device-time model of the layer kernels runs (record-only on NKISHIM, Qwen2.5-0.5B's
  shapes), attributes every instruction to a v1 member or a LayerSpec section, reports the critical path per section and
  engine, and prints the weight-stationary predictions under three stationary-load costs (labelled predictions)."""
  import subprocess, sys
  out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/nki_layer_cost.py"), "--rows", "1", "--ctx", "512", "--variants", "v1,v2,v2se,v2swe,v2s+bf16,v2se+bf16"],
                       capture_output=True, text=True, timeout=600, cwd=ROOT, env={**os.environ, "FZ_DEVICE": "NKISHIM"})
  assert out.returncode == 0, out.stderr[-2000:]
  o = out.stdout
  assert "9: fz_matmul_1_1_896_4864" in o and "5. attention (each)" in o and "0. input-only loads" in o and "critical path by engine" in o
  assert "predict rows 1: ctx 512: v1 " in o and ", v2se " in o and "predict v2swe r1 c512" in o and "te_stat = te_col" in o
  assert "v1 r1 c512" in o and "0.3963" in o   # measured next to the model
  assert "v2s+bf16 r1 c512" in o and "0.2395" in o and "fz_layerse_add_1_512_896_4864_bf16" in o   # I68 bf16 variants (LayerBF16)

# ---------------------------------------------------------------- Qwen2.5-0.5B on NKISHIM vs the float32 reference

def _qwen() -> str | None:
  try:
    from fusezap.serve.weights import resolve_model
    return resolve_model("Qwen/Qwen2.5-0.5B-Instruct", allow_download=False)
  except Exception: return None

@pytest.mark.skipif(_qwen() is None or TIER != "NKISHIM", reason="Qwen/Qwen2.5-0.5B-Instruct not in the HF cache (NEURON: scripts/nki_n5_serve.py)")
def test_qwen_tokens_match_the_f32_reference():
  """Two reference prompts batched, 4 greedy tokens each on NKISHIM: the reference's tokens (HF transformers float32)."""
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine, SamplingParams
  ref_path = os.path.join(ROOT, "docs/validate/examples/refs/qwen2.5-0.5b-instruct-f32.json")
  if not os.path.exists(ref_path): pytest.skip("the float32 reference (docs/validate) is not in this tree (public export)")
  ref = json.load(open(ref_path))
  m = load_model("Qwen/Qwen2.5-0.5B-Instruct", dtype=dtypes.float32, device="NKISHIM", max_len=256, prefill_chunk=64)
  eng = LLMEngine(m, num_blocks=16, block_size=16, max_num_seqs=2, max_prefill_tokens=64, max_model_len=256)
  ps = ref["prompts"][:2]
  rids = {eng.add_request(p["prompt_ids"], SamplingParams(max_tokens=4)): p for p in ps}
  out = {r: [] for r in rids}
  while eng.has_unfinished():
    for o in eng.step(): out[o.request_id] += o.new_token_ids
  for r, p in rids.items(): assert out[r] == p["greedy_ids"][:4], (p["id"], out[r])
  assert _decode_launches(eng) == 1   # N8 defaults (FUSE=24, HEAD_FUSE=1): the whole decode forward one kernel (N6 FUSE=1: 24 + 1;
                                      # N5: 24 * 10 + 5; Qwen2.5 has no o_proj bias)

def test_planned_forwards_and_scorer_hold_less(tmp_path, monkeypatch):
  """N8 (I71): FZ_NEURON_PLAN=1 (default) plans the engine's captured forwards and the logprobs scorer's steps on NEURON (fz.jit
  plan; the scorer's trunk now has per-layer fences, so its layers are separate launches whose intermediates share memory):
  hbm_report's bytes for them drop, the tokens and logprobs stay the same, and the scorer keeps at most FZ_LOGPROBS_JITS
  (default 2 on NEURON) buckets captured."""
  from test_tp import tiny_model_dir
  from fusezap.serve.logprobs import scorer
  from fusezap.serve.neuron_kv import hbm_report, hbm_line
  p = tiny_model_dir(str(tmp_path / "q7p"), seed=2, cfg={"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 3,
                                                          "num_attention_heads": 28, "num_key_value_heads": 4, "head_dim": 128})
  monkeypatch.setenv("FZ_NEURON_FUSE", "1")
  res = {}
  for plan in ("0", "1"):
    monkeypatch.setenv("FZ_NEURON_PLAN", plan)
    toks, eng = _serve(p, TIER)
    sc = scorer(eng)
    lps = [sc.score(list(range(3, 3 + n)), 1, 3) for n in (20, 20, 40, 40, 70, 70)]   # each bucket twice (captured on the second)
    r = hbm_report(eng)
    assert len(sc.jits) <= 2 and "NEURON HBM" in hbm_line(eng)
    res[plan] = (toks, lps, sum(r["forwards"].values()), r["scorer"])
  assert res["0"][0] == res["1"][0] and res["0"][1] == res["1"][1]
  assert res["1"][2] < res["0"][2] and res["1"][3] < 0.6 * res["0"][3], (res["0"][2:], res["1"][2:])

@pytest.mark.skipif(TIER != "NKISHIM", reason="the fake runtime: Mac only")
def test_prewarm_compiles_and_loads_every_prefix(tiny, neuron_dev, nrt, monkeypatch):
  """I75: the engine's prewarm (fusezap.serve.neuron_prewarm, run by `fusezap serve`'s warm-up) captures every forward at
  every pool prefix (here the prefill at 16, 32, 64 and the whole pool, decode at the pool), compiles the fused NEFFs in
  parallel (FZ_NEURON_PREWARM_JOBS threads, each with its own compile worker) and loads them; afterwards a prefill at every
  prefix bucket (synthetic steps and real requests) compiles nothing and loads nothing, hbm_report counts every prefix's
  forward, and the tokens are CPU's (prewarm's KV rows in free blocks are never read)."""
  from fusezap.device import neuron
  from fusezap.serve import neuron_prewarm as PW
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine, SamplingParams
  from fusezap.serve.neuron_kv import hbm_report
  dev, nki = neuron_dev
  monkeypatch.setenv("FZ_NEURON_PREFILL_CTX", "16,32,64"); monkeypatch.setenv("FZ_NEURON_CTX", "0")
  monkeypatch.setenv("FZ_NEURON_PREWARM_JOBS", "3")
  workers = []
  monkeypatch.setattr(neuron, "compile_worker", lambda: workers.append(1) or nki)
  cpu, _ = _serve(tiny, "CPU")
  m = load_model(tiny, dtype=dtypes.float32, device="NEURON", max_len=128, prefill_chunk=128)
  eng = LLMEngine(m, num_blocks=64, block_size=16, max_num_seqs=4, max_prefill_tokens=32, max_model_len=128, decode_beam=0)
  assert eng.cache.ctx_buckets(True) == [16, 32, 64, 1024] and eng.cache.ctx_buckets() == [1024]
  rep = PW.prewarm(eng, lambda s: None)
  f = eng._prefills[32]
  assert sorted(x["ctx"] for x in rep["forwards"] if x["forward"] == "prefill_32") == [16, 32, 64, 1024] and not rep["skipped"]
  assert all(j.captured is not None for j in [f.jit, *f.ctx_jits.values()]) and set(f.ctx_jits) == {16, 32, 64}
  nn = rep["neffs"][0]
  assert nn and all(x["ok"] for x in nn) and any(not x["hit"] for x in nn) and workers, (nn, workers)   # parallel compile ran
  assert rep["loads"][0] and rep["seconds"]["total"] >= rep["seconds"]["compile"]
  fw = hbm_report(eng)["forwards"]
  assert fw["prefill_32"] == sum(neuron_kv_bytes(j) for j in [f.jit, *f.ctx_jits.values()]) > 0
  n0, l0, c0 = nki.neffs, dev.allocator.calls["load"], dict(neuron.NEFF_STATS)
  for ctx in eng.cache.ctx_buckets(True):   # a prefill at every prefix bucket
    step = PW.synthetic_step(eng, f, ctx)
    assert eng.cache.meta_ctx(step[3], f.N, f.R, True) == ctx
    f.run(*step)
  rids = [eng.add_request(p, SamplingParams(max_tokens=10, ignore_eos=True)) for p in PROMPTS]
  out = {r: [] for r in rids}
  while eng.has_unfinished():
    for o in eng.step(): out[o.request_id] += o.new_token_ids
  assert out == {r: cpu[k] for r, k in zip(rids, sorted(cpu))}
  assert (nki.neffs, dev.allocator.calls["load"], neuron.NEFF_STATS["compiles"]) == (n0, l0, c0["compiles"]), "a compile or load after prewarm"

def neuron_kv_bytes(j) -> int:
  from fusezap.serve.neuron_kv import jit_device_bytes
  return jit_device_bytes(j)
