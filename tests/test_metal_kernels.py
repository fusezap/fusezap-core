"""P2-MTL: Metal custom kernels (SPEC §11.1): the simdgroup-matrix matmul and flash attention.

Every kernel is checked against the composed equivalent (the same expression with FZ_CUSTOM=0) and against torch, with the
SPEC §10 tolerances. Scheduler-hook path (float32): `a @ b` and `scaled_dot_product_attention` become kind="custom"
items run by METALDevice.run. Direct path (fusezap.device.metal.matmul / attention): float16/bfloat16 and causal.
"""
from __future__ import annotations
import math
import numpy as np
import pytest
import torch
from fusezap import Tensor, dtypes, helpers
from fusezap.device import Device

pytestmark = pytest.mark.skipif(not Device.available("METAL"), reason="needs Metal")
if Device.available("METAL"):
  from fusezap.device import metal as M
  from fusezap.schedule import create_schedule
  from fusezap.codegen.metal import attention_block

R = np.random.default_rng(5)
F32, F16, BF16 = dtypes.float32, dtypes.float16, dtypes.bfloat16
TDT = {F32: torch.float32, F16: torch.float16, BF16: torch.bfloat16}
TOL = {F32: dict(rtol=1e-4, atol=1e-5), F16: dict(rtol=1e-2, atol=1e-2), BF16: dict(rtol=1e-2, atol=1e-2)}

@pytest.fixture(autouse=True)
def metal(monkeypatch):
  if hasattr(helpers.getenv, "cache_clear"): helpers.getenv.cache_clear()
  for k in ("FZ_INTERP", "FZ_CUSTOM", "FZ_BEAM", "FZ_OPT"): monkeypatch.delenv(k, raising=False)
  monkeypatch.setattr(Device, "DEFAULT", "METAL", raising=False)
  yield
  if hasattr(helpers.getenv, "cache_clear"): helpers.getenv.cache_clear()

def rnd(*shape) -> np.ndarray: return R.standard_normal(shape).astype(np.float32)
def T(a: np.ndarray, dt=F32) -> Tensor: return Tensor(a, device="METAL").cast(dt) if dt != F32 else Tensor(a, device="METAL")
def f32(t: Tensor) -> np.ndarray: return t.cast(F32).numpy() if t.dtype != F32 else t.numpy()
def tt(a: np.ndarray, dt=F32) -> torch.Tensor: return torch.tensor(a).to(TDT[dt])
def names(t: Tensor) -> list[str]:
  return [si.ast[0].kernel.name if si.kind == "custom" else si.kind for si in create_schedule([t.lazydata]) if si.kind != "copy"]

def custom_and_composed(fn, monkeypatch, *arrs):
  """(custom output, composed output, kernel names with custom on) of fn on fresh METAL tensors."""
  outs = []
  for flag in ("1", "0"):
    monkeypatch.setenv("FZ_CUSTOM", flag)
    if flag == "1": ks = names(fn(*[T(a) for a in arrs]))
    outs.append(fn(*[T(a) for a in arrs]).numpy())
  monkeypatch.delenv("FZ_CUSTOM")
  return outs[0], outs[1], ks

# ------------------------------------------------------------------ matmul through the scheduler hook (float32)
MM = {
  "2d": (lambda a, b: a @ b, [(5, 7), (7, 3)]),
  "square_256": (lambda a, b: a @ b, [(256, 256), (256, 256)]),
  "edges": (lambda a, b: a @ b, [(65, 33), (33, 130)]),
  "tile_exact": (lambda a, b: a @ b, [(64, 64), (64, 128)]),
  "k_tail": (lambda a, b: a @ b, [(128, 13), (13, 64)]),
  "batched": (lambda a, b: a @ b, [(2, 3, 70, 45), (2, 3, 45, 100)]),
  "broadcast_b": (lambda a, b: a @ b, [(3, 70, 45), (45, 100)]),
  "broadcast_a": (lambda a, b: a @ b, [(70, 45), (4, 45, 100)]),   # M axis split by a batch axis: not one GEMM, composed
  "a_transposed": (lambda a, b: a.T @ b, [(96, 128), (96, 64)]),
  "b_transposed": (lambda a, b: a @ b.T, [(128, 96), (64, 96)]),
  "both_transposed": (lambda a, b: a.T @ b.T, [(40, 129), (66, 40)]),
  "linear": (lambda a, b: a @ b.transpose(-1, -2) + 1, [(2, 33, 64), (80, 64)]),
  "strided": (lambda a, b: a[:, ::2] @ b[::2], [(64, 128), (128, 64)]),
  "vec_mat": (lambda a, b: a @ b, [(40,), (40, 72)]),
  "mat_vec": (lambda a, b: a @ b, [(72, 40), (40,)]),
  "k1": (lambda a, b: a @ b, [(65, 1), (1, 70)]),
  "chain_epilogue": (lambda a, b: ((a @ b).relu() @ b.T).sum(-1), [(64, 48), (48, 96)]),
}
MM_FALLBACK = {"broadcast_a"}

@pytest.mark.parametrize("name", list(MM))
def test_matmul_hook(name, monkeypatch):
  fn, shapes = MM[name]
  arrs = [rnd(*s) for s in shapes]
  c, p, ks = custom_and_composed(fn, monkeypatch, *arrs)
  assert ("fz_matmul" in ks) == (name not in MM_FALLBACK), ks
  np.testing.assert_allclose(c, p, **TOL[F32])
  ref = fn(*[torch.tensor(a) for a in arrs]).numpy()
  np.testing.assert_allclose(c, ref, **TOL[F32])

def test_matmul_hook_large(monkeypatch):
  a, b = rnd(1024, 768), rnd(768, 1000)
  c, p, ks = custom_and_composed(lambda x, y: x @ y, monkeypatch, a, b)
  assert ks == ["fz_matmul"]
  np.testing.assert_allclose(c, p, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(c, a.astype(np.float64) @ b, rtol=1e-4, atol=1e-4)

def test_matmul_backward():
  a, b = rnd(3, 40, 72), rnd(72, 56)
  x, y = Tensor(a, device="METAL", requires_grad=True), Tensor(b, device="METAL", requires_grad=True)
  ((x @ y).relu() * 2).sum().backward()
  tx, ty = torch.tensor(a, requires_grad=True), torch.tensor(b, requires_grad=True)
  ((tx @ ty).relu() * 2).sum().backward()
  np.testing.assert_allclose(x.grad.numpy(), tx.grad.numpy(), **TOL[F32])
  np.testing.assert_allclose(y.grad.numpy(), ty.grad.numpy(), **TOL[F32])

def test_matmul_disabled(monkeypatch):
  monkeypatch.setenv("FZ_CUSTOM", "0")
  assert "fz_matmul" not in names(T(rnd(16, 16)) @ T(rnd(16, 16)))

def test_jit_replay_custom():
  from fusezap.jit import jit
  @jit
  def f(a, b): return (a @ b).relu()
  bm = rnd(48, 80)
  B = T(bm).realize()
  for _ in range(4):
    a = rnd(33, 48)
    np.testing.assert_allclose(f(T(a).realize(), B).numpy(), np.maximum(a @ bm, 0), **TOL[F32])

# ------------------------------------------------------------------ matmul, direct (float32 / float16 / bfloat16)
DMM = [((5, 7), (7, 3)), ((65, 33), (33, 130)), ((256, 256), (256, 256)), ((2, 3, 70, 45), (2, 3, 45, 100)),
       ((3, 70, 45), (45, 100)), ((70, 45), (4, 45, 100)), ((128, 13), (13, 64)), ((1, 9), (9, 1))]

@pytest.mark.parametrize("dt", [F32, F16, BF16], ids=["f32", "f16", "bf16"])
@pytest.mark.parametrize("shapes", DMM, ids=[f"{a}x{b}" for a, b in DMM])
def test_matmul_direct(dt, shapes, monkeypatch):
  a, b = rnd(*shapes[0]), rnd(*shapes[1])
  out = M.matmul(T(a, dt), T(b, dt))
  assert out.dtype == dt and tuple(out.shape) == tuple(np.broadcast_shapes(a.shape[:-2], b.shape[:-2])) + (a.shape[-2], b.shape[-1])
  monkeypatch.setenv("FZ_CUSTOM", "0")
  composed = f32(T(a, dt) @ T(b, dt))
  got = f32(out)
  np.testing.assert_allclose(got, composed, **TOL[dt])
  ref = (tt(a, dt) @ tt(b, dt)).float().numpy()
  np.testing.assert_allclose(got, ref, **TOL[dt])

def test_matmul_direct_views():
  """Non-contiguous inputs are materialized first (transpose, slice)."""
  a, b = rnd(64, 96), rnd(80, 64)
  got = M.matmul(T(a)[:, ::2].T, T(b).T)
  np.testing.assert_allclose(got.numpy(), a[:, ::2].T @ b.T, **TOL[F32])

# ------------------------------------------------------------------ attention
def sdpa_ref(q, k, v, causal, dt=F32) -> np.ndarray:
  if dt == F32: return torch.nn.functional.scaled_dot_product_attention(*map(torch.tensor, (q, k, v)), is_causal=causal).numpy()
  # torch half sdpa on CPU is not reliably available: compute in f32 from the rounded inputs
  q, k, v = [tt(x, dt).float() for x in (q, k, v)]
  return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal).numpy()

ATT = [((2, 3, 50, 32), (2, 3, 50, 32), (2, 3, 50, 32)),   # L, S not multiples of the block
       ((1, 4, 128, 64), (1, 4, 128, 64), (1, 4, 128, 64)),
       ((2, 33, 16), (2, 70, 16), (2, 70, 24)),              # L != S, Dv != D
       ((1, 2, 9, 13), (1, 2, 40, 13), (1, 2, 40, 7)),       # head dims not multiples of 8
       ((3, 64, 128), (3, 64, 128), (3, 64, 128)),           # D = 128 (smallest key block)
       ((1, 8, 40), (1, 300, 40), (1, 300, 40)),
       ((2, 17, 160), (2, 23, 160), (2, 23, 136))]           # too large for threadgroup memory: per-row kernel
ATT_IDS = ["x".join(map(str, s[0])) + "_" + str(s[1][-2]) + "_" + str(s[2][-1]) for s in ATT]

@pytest.mark.parametrize("shapes", ATT, ids=ATT_IDS)
def test_attention_hook(shapes, monkeypatch):
  q, k, v = [rnd(*s) for s in shapes]
  fn = lambda q, k, v: q.scaled_dot_product_attention(k, v)
  c, p, ks = custom_and_composed(fn, monkeypatch, q, k, v)
  assert ks == ["fz_attention"], ks
  np.testing.assert_allclose(c, p, **TOL[F32])
  np.testing.assert_allclose(c, sdpa_ref(q, k, v, False), **TOL[F32])

def test_attention_hook_softmax_form(monkeypatch):
  q, k, v = rnd(2, 40, 32), rnd(2, 56, 32), rnd(2, 56, 32)
  fn = lambda q, k, v: ((q @ k.transpose(-1, -2)) * 0.3).softmax(-1) @ v
  c, p, ks = custom_and_composed(fn, monkeypatch, q, k, v)
  assert ks == ["fz_attention"], ks
  np.testing.assert_allclose(c, p, **TOL[F32])

def test_attention_large_scores_stable(monkeypatch):
  """The online softmax must subtract the running max: scores ~ +-1e3 would overflow exp otherwise."""
  q, k, v = rnd(1, 2, 64, 32) * 30, rnd(1, 2, 96, 32) * 30, rnd(1, 2, 96, 32)
  c, p, _ = custom_and_composed(lambda q, k, v: q.scaled_dot_product_attention(k, v), monkeypatch, q, k, v)
  assert np.isfinite(c).all()
  np.testing.assert_allclose(c, p, rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("dt", [F32, F16, BF16], ids=["f32", "f16", "bf16"])
@pytest.mark.parametrize("shapes", ATT, ids=ATT_IDS)
def test_attention_direct(shapes, dt, causal, monkeypatch):
  q, k, v = [rnd(*s) for s in shapes]
  out = M.attention(T(q, dt), T(k, dt), T(v, dt), causal=causal)
  assert out.dtype == dt and tuple(out.shape) == shapes[0][:-1] + (shapes[2][-1],)
  monkeypatch.setenv("FZ_CUSTOM", "0")
  composed = f32(T(q, dt).scaled_dot_product_attention(T(k, dt), T(v, dt), is_causal=causal))
  got = f32(out)
  # the composed bf16 path rounds its own intermediates: two bf16 results can differ by ~2 ulp (0.0156 near |x|=1).
  # Input data depends on test order (module RNG), so the 1e-2 bound flaked under xdist.
  np.testing.assert_allclose(got, composed, **(dict(rtol=2e-2, atol=2e-2) if dt == BF16 else TOL[dt]))
  np.testing.assert_allclose(got, sdpa_ref(q, k, v, causal, dt), **TOL[dt])

def test_attention_direct_scale_and_broadcast():
  q, k, v = rnd(2, 3, 20, 16), rnd(1, 3, 30, 16), rnd(30, 16)
  got = M.attention(T(q), T(k), T(v), causal=True, scale=0.7)
  tq, tk, tv = map(torch.tensor, (q, k, v))
  ref = torch.nn.functional.scaled_dot_product_attention(tq, tk.expand(2, 3, 30, 16), tv.expand(2, 3, 30, 16), is_causal=True, scale=0.7)
  np.testing.assert_allclose(got.numpy(), ref.numpy(), **TOL[F32])

def test_attention_block_sizes():
  assert attention_block(64, 64) == 32 and attention_block(128, 128) == 8 and attention_block(160, 136) is None

def test_attention_direct_empty_keys():
  got = M.attention(T(rnd(2, 5, 8)), T(rnd(2, 0, 8)), T(rnd(2, 0, 4)))
  assert tuple(got.shape) == (2, 5, 4)
  np.testing.assert_array_equal(got.numpy(), np.zeros((2, 5, 4), np.float32))

@pytest.mark.parametrize("dt,S,D,Dv,off", [(BF16, 2000, 160, 136, 3.0), (F16, 3000, 160, 136, 0.0), (BF16, 500, 200, 200, 0.0),
                                           (F16, 1500, 200, 72, 3.0)], ids=["bf16_2000_off", "f16_3000", "bf16_500_d200", "f16_dv72_off"])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_attention_rows_long_keys_float_accum(dt, S, D, Dv, off, causal):
  """Regression: the per-row fallback (head dims too large for threadgroup memory) must accumulate O in float32, not in
  the f16/bf16 output buffer, else the error grows with S (bf16, S=2000, v+3 was 1.41 max abs error)."""
  assert attention_block(D, Dv) is None
  q, k, v = rnd(1, 4 if not causal else S // 4, D), rnd(1, S, D), rnd(1, S, Dv) + off
  got = f32(M.attention(T(q, dt), T(k, dt), T(v, dt), causal=causal))
  np.testing.assert_allclose(got, sdpa_ref(q, k, v, causal, dt), **TOL[dt])
