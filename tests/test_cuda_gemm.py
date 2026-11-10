"""P3-GEMM: CUDA tensor-core GEMMs (fusezap/device/cuda_kernels).

Hook path: `a @ b` on CUDA becomes a kind="custom" item once cuda_kernels.install(dev) ran: float32 runs 3xTF32 mma.sync,
f16/bf16 operands run the 16-bit mma.sync / wgmma kernel (the matcher looks through the casts to f32). Every form is checked
against the composed kernels (FZ_CUSTOM=0) and torch at the SPEC §10 matmul tolerance.
Direct path: matmul / linear in f16 / bf16 (mma.sync on sm_80+, wgmma + TMA on sm_90) and f32; every tile config the
device can run; fp8 e4m3 (sm_89+) against the dequantized reference. Render tests run anywhere; GPU tests need CUDA.
"""
from __future__ import annotations
import itertools, os
import numpy as np
import pytest
from fusezap import Tensor, dtypes, helpers
from fusezap.device import Device
from fusezap.device.cuda_kernels import mma, wgmma, tf32, fp8
import fusezap.device.cuda_kernels as ck

HAS_CUDA = Device.available("CUDA")
F32, F16, BF16, U8 = dtypes.float32, dtypes.float16, dtypes.bfloat16, dtypes.uint8
TOL = {F32: dict(rtol=1e-4, atol=1e-5), F16: dict(rtol=1e-2, atol=1e-2), BF16: dict(rtol=1e-2, atol=1e-2)}
R = np.random.default_rng(7)

# ------------------------------------------------------------------ render (no GPU)

def test_render_variants():
  names = set()
  for kind, out, nn in itertools.product(["f16", "bf16", "e4m3"], ["f32", "f16", "bf16"], [False, True]):
    for cfg in [mma.MmaCfg(), mma.MmaCfg(128, 256, 2, 4, 64, 3), mma.MmaCfg(64, 64, 2, 2, 128, 4)]:
      if not mma.valid(cfg, kind, nn): continue
      src = mma.render(kind, out, nn, cfg)
      assert mma.name_of(kind, out, nn, cfg) in src and ("ldsm4t(fb" in src) == nn
      assert ("m16n8k32" if kind == "e4m3" else "m16n8k16") in src
      names.add(mma.name_of(kind, out, nn, cfg))
    if kind == "e4m3" and nn: continue
    cfg = ck.wg_default(kind)
    src = wgmma.render(kind, out, nn, cfg)
    assert ("k32.f32.e4m3.e4m3" if kind == "e4m3" else f"k16.f32.{kind}.{kind}") in src and ("setmaxnreg" in src) == cfg.regs
    assert src.count('"+f"(d[') >= cfg.BN // 2 and ("d[i] += dt[i]" in src) == (kind == "e4m3")
  assert not mma.valid(mma.MmaCfg(), "e4m3", True)   # no 8-bit ldmatrix.trans
  for la, lb in itertools.product("kr", "kr"): assert tf32.name_of(la, lb) in tf32.render(la, lb)
  assert tf32.layout(1, 64) == "r" and tf32.layout(64, 1) == "k" and tf32.layout(3, 5) == "k"
  for dt in ("f32", "f16", "bf16"): assert f"fz_fp8_quant_{dt}" in fp8.render(dt)

def test_install_needs_sm80():
  class D:
    cc, custom_kernels = (7, 5), {}
    def run(self, si): return None
  assert not ck.install(D())

def test_cuda_tier_is_live():
  if "CUDA" not in [t.strip() for t in os.getenv("FZ_TEST_TIERS", "").split(",")]: pytest.skip("CUDA tier not requested")
  Device["CUDA"]   # raises if the device cannot start

class _FakeKernel:
  """Records launches instead of running them (grid limits and chunk addresses, no GPU)."""
  def __init__(self, log): self.log = log
  def __call__(self, gs, bs, args, wait=False):
    self.log.append((tuple(gs), dict(enumerate(v for _, v in args))))
    return 1.0 if wait else None
  def bind(self, gs, bs, args): return lambda wait=False: self(gs, bs, args, wait)

@pytest.mark.parametrize("impl", ["mma", "wgmma"])
def test_gemm_splits_batch_over_grid_limit(monkeypatch, impl):
  """Regression: the 16-bit / fp8 kernels put the batch on gridDim.y (<= 65535); a 70000-entry batch must launch in chunks."""
  class D:
    def __init__(self): self.cc, self.gpu_name, self._fz_smem, self._fz_sms = (9, 0), "fake", 227 * 1024, 132
  log: list = []
  monkeypatch.setattr(ck, "kernel", lambda *a, **k: _FakeKernel(log))
  monkeypatch.setattr(ck, "_tmap", lambda dev, addr, *a: addr.to_bytes(8, "little"))
  monkeypatch.delenv("FZ_BEAM", raising=False)
  helpers.getenv.cache_clear()
  Bt, M, N, K = 70000, 1, 64, 64
  C, A, B = 1 << 32, 2 << 32, 3 << 32
  for bind in (False, True):
    log.clear()
    r = ck.gemm(D(), C, A, B, M, N, K, kind="bf16", out="bf16", Bt=Bt, sab=M * K, sbb=0, scb=M * N, impl=impl, bind=bind)
    if bind: r()
    assert [g[1] for g, _ in log] == [65535, Bt - 65535] and all(max(g) <= 65535 or g[1] == 1 and g[2] == 1 for g, _ in log)
    a0 = [int.from_bytes(p[1], "little") if impl == "wgmma" else p[1] for _, p in log]
    assert a0 == [A, A + 65535 * M * K * 2] and [p[0] for _, p in log] == [C, C + 65535 * M * N * 2]
    b0 = [int.from_bytes(p[2], "little") if impl == "wgmma" else p[2] for _, p in log]
    assert b0 == [B, B]   # broadcast operand: every chunk reads the same matrix
  log.clear()
  assert ck.gemm(D(), C, A, B, M, N, K, kind="bf16", out="f32", Bt=Bt, sab=M * K, scb=M * N, impl=impl, wait=True) == 2.0
  assert [p[0] for _, p in log] == [C, C + 65535 * M * N * 4]
  helpers.getenv.cache_clear()

def _sum_base(t: Tensor):
  from fusezap.ops import ReduceOps
  stack = [t.lazydata]
  while stack:
    if (b := stack.pop().base).op is ReduceOps.SUM: return b
    stack.extend(b.srcs)

FORMS16 = {   # name -> (fn, shapes, takes the 16-bit kernel)
  "2d": (lambda a, b: a @ b, [(5, 16), (16, 24)], True),
  "batched": (lambda a, b: a @ b, [(2, 3, 20, 8), (2, 3, 8, 24)], True),
  "broadcast_b": (lambda a, b: a @ b, [(3, 70, 48), (48, 104)], True),
  "b_transposed": (lambda a, b: a @ b.T, [(33, 64), (80, 64)], True),
  "linear": (lambda a, b: a @ b.transpose(-1, -2) + 1, [(2, 33, 64), (80, 64)], True),
  "offset": (lambda a, b: a[3:, 8:] @ b[8:, 8:], [(70, 48), (48, 72)], True),
  "vec_mat": (lambda a, b: a @ b, [(40,), (40, 72)], True),
  "mat_vec": (lambda a, b: a @ b, [(72, 40), (40,)], True),
  "a_transposed": (lambda a, b: a.T @ b, [(96, 128), (96, 64)], False),   # A not K-contiguous: 3xTF32 on f32 copies
  "k_tail": (lambda a, b: a @ b, [(128, 13), (13, 64)], False),            # K % 8 != 0
  "unaligned_n_nn": (lambda a, b: a @ b, [(16, 32), (32, 21)], False),      # (K, N) B with N % 8 != 0
  "odd_offset": (lambda a, b: a[:, 1:] @ b[1:], [(16, 33), (33, 24)], False),
}

@pytest.mark.parametrize("name", list(FORMS16))
def test_match16_params(monkeypatch, name):
  """No GPU: the 16-bit matcher sees through the f32 casts of f16/bf16 operands and yields the same GEMM params the float32
  matcher gives for the same expression on float32 buffers (or keeps the 3xTF32 match when the kernels cannot read it)."""
  from fusezap import schedule
  if ck.HOOK is None: monkeypatch.setattr(ck, "HOOK", ck._hook_kernel()); monkeypatch.setattr(ck, "HOOK16", ck._hook16_kernel())
  monkeypatch.setattr(ck, "_MATCH0", ck._MATCH0 or schedule._match_custom)
  monkeypatch.setattr(schedule, "_device_kernels", lambda d: {"matmul": ck.HOOK, "matmul16": ck.HOOK16})
  fn, shapes, takes = FORMS16[name]
  arrs = [rnd(*s) for s in shapes]
  m32 = ck._match16(_sum_base(fn(*[Tensor(a, device="NPY").realize() for a in arrs])))
  assert m32 is not None and m32.kernel is ck.HOOK
  for dt in (F16, BF16):
    m = ck._match16(_sum_base(fn(*[Tensor(a, device="NPY").cast(dt).realize() for a in arrs])))
    assert m is not None and m.kernel is (ck.HOOK16 if takes else ck.HOOK), (name, dt)
    if takes:
      assert m.params == m32.params and all(x.dtype == dt for x in m.inputs)
      assert ck._mm16(m.params) is not None
  # mixed dtypes stay on the f32 path
  m = ck._match16(_sum_base(fn(Tensor(arrs[0], device="NPY").cast(BF16).realize(), Tensor(arrs[1], device="NPY").realize())))
  assert m is not None and m.kernel is ck.HOOK

# ------------------------------------------------------------------ GPU fixtures

gpu = pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")

@pytest.fixture
def cu(monkeypatch):
  if not HAS_CUDA: pytest.skip("CUDA not available")
  helpers.getenv.cache_clear()
  for k in ("FZ_INTERP", "FZ_CUSTOM", "FZ_BEAM", "FZ_OPT", "FZ_NO_WGMMA"): monkeypatch.delenv(k, raising=False)
  monkeypatch.setattr(Device, "DEFAULT", "CUDA", raising=False)
  dev = Device["CUDA"]
  if tuple(dev.cc) < (8, 0): pytest.skip("tensor-core GEMMs need sm_80+")
  assert ck.install(dev)
  yield dev
  helpers.getenv.cache_clear()

def rnd(*shape) -> np.ndarray: return R.standard_normal(shape).astype(np.float32)
def T(a: np.ndarray, dt=F32) -> Tensor:
  t = Tensor(a, device="CUDA")
  return t.cast(dt).realize() if dt != F32 else t
def f32(t: Tensor) -> np.ndarray: return t.cast(F32).numpy() if t.dtype != F32 else t.numpy()
def rounded(a: np.ndarray, dt) -> np.ndarray: return f32(T(a, dt))   # the values a 16-bit tensor really holds
def names(t: Tensor) -> list[str]:
  from fusezap.schedule import create_schedule
  return [si.ast[0].kernel.name if si.kind == "custom" else si.kind for si in create_schedule([t.lazydata]) if si.kind != "copy"]

# ------------------------------------------------------------------ hook path: custom == composed (float32)
MM = {
  "2d": (lambda a, b: a @ b, [(5, 7), (7, 3)]),
  "square_256": (lambda a, b: a @ b, [(256, 256), (256, 256)]),
  "edges": (lambda a, b: a @ b, [(65, 33), (33, 130)]),
  "tile_exact": (lambda a, b: a @ b, [(128, 64), (64, 128)]),
  "k_tail": (lambda a, b: a @ b, [(128, 13), (13, 64)]),
  "batched": (lambda a, b: a @ b, [(2, 3, 70, 45), (2, 3, 45, 100)]),
  "broadcast_b": (lambda a, b: a @ b, [(3, 70, 45), (45, 100)]),
  "broadcast_a": (lambda a, b: a @ b, [(70, 45), (4, 45, 100)]),   # M axis split by a batch axis: not one GEMM, composed
  "a_transposed": (lambda a, b: a.T @ b, [(96, 128), (96, 64)]),
  "b_transposed": (lambda a, b: a @ b.T, [(128, 96), (64, 96)]),
  "both_transposed": (lambda a, b: a.T @ b.T, [(40, 129), (66, 40)]),
  "linear": (lambda a, b: a @ b.transpose(-1, -2) + 1, [(2, 33, 64), (80, 64)]),
  "strided": (lambda a, b: a[:, ::2] @ b[::2], [(64, 128), (128, 64)]),
  "offset": (lambda a, b: a[3:, 1:] @ b[1:, 5:], [(70, 41), (41, 77)]),
  "vec_mat": (lambda a, b: a @ b, [(40,), (40, 72)]),
  "mat_vec": (lambda a, b: a @ b, [(72, 40), (40,)]),
  "k1": (lambda a, b: a @ b, [(65, 1), (1, 70)]),
  "chain_epilogue": (lambda a, b: ((a @ b).relu() @ b.T).sum(-1), [(64, 48), (48, 96)]),
  "half_inputs": (lambda a, b: (a.cast(F16) @ b.cast(F16)).cast(F32), [(96, 80), (80, 112)]),
  "bf16_inputs": (lambda a, b: (a.cast(BF16) @ b.cast(BF16)).cast(F32), [(96, 80), (80, 112)]),
}
MM_FALLBACK = {"broadcast_a"}

def custom_and_composed(fn, monkeypatch, *arrs):
  outs = []
  for flag in ("1", "0"):
    monkeypatch.setenv("FZ_CUSTOM", flag)
    helpers.getenv.cache_clear()
    if flag == "1": ks = names(fn(*[T(a) for a in arrs]))
    outs.append(fn(*[T(a) for a in arrs]).numpy())
  monkeypatch.delenv("FZ_CUSTOM")
  helpers.getenv.cache_clear()
  return outs[0], outs[1], ks

@pytest.mark.parametrize("name", list(MM))
def test_matmul_hook(cu, name, monkeypatch):
  import torch
  fn, shapes = MM[name]
  arrs = [rnd(*s) for s in shapes]
  c, p, ks = custom_and_composed(fn, monkeypatch, *arrs)
  half = name in ("half_inputs", "bf16_inputs")   # 16-bit operands: the f16/bf16 hook (no f32 copies of the inputs)
  assert (("fz_matmul_mma16" if half else "fz_matmul_tf32x3") in ks) == (name not in MM_FALLBACK), ks   # one f16/bf16 rounding of slightly different f32 sums: 1 ulp apart
  K = max(shapes[0][-1] if len(shapes[0]) > 1 else shapes[0][0], 1)
  # any two f32 summation orders differ by ~sqrt(K) ulps of the partial sums: the f32 tolerance's atol grows with sqrt(K)
  # and a chain's later sums cancel terms as large as the whole output, so atol also scales with max |output| (B2-INT: the
  # module RNG makes inputs depend on xdist order, and chain_epilogue / k_tail failed on one ordering by these margins)
  scale = float(np.abs(p).max())
  tol = TOL[F16] if half else dict(rtol=1e-4, atol=max(1e-5 * max(1, np.sqrt(K) / 8), 2e-6 * scale))
  np.testing.assert_allclose(c, p, **tol)
  if not half:
    exact = fn(*[torch.tensor(a, dtype=torch.float64) for a in arrs]).numpy()
    np.testing.assert_allclose(c, exact, **tol)
    # 3xTF32 is a few f32 ulps from exact; composed f32 may be closer by luck: allow 2x plus a few ulps of the output scale
    assert np.abs(c - exact).max() <= 2 * np.abs(p - exact).max() + 1e-6 + 2**-21 * scale, "custom is less accurate than composed"

def test_matmul_hook_large(cu, monkeypatch):
  a, b = rnd(1024, 768), rnd(768, 1000)
  c, p, ks = custom_and_composed(lambda x, y: x @ y, monkeypatch, a, b)
  assert ks == ["fz_matmul_tf32x3"]
  np.testing.assert_allclose(c, p, rtol=1e-4, atol=1e-4)
  exact = a.astype(np.float64) @ b
  np.testing.assert_allclose(c, exact, rtol=1e-4, atol=1e-4)
  assert np.abs(c - exact).max() <= 2 * np.abs(p - exact).max() + 1e-6, "custom is less accurate than composed"

def test_matmul_hook_special_values(cu, monkeypatch):
  a, b = rnd(40, 24), rnd(24, 36)
  a[3, 5], a[7, :] = np.inf, 1e30
  b[:, 9] = np.nan
  c, p, _ = custom_and_composed(lambda x, y: x @ y, monkeypatch, a, b)
  np.testing.assert_array_equal(np.isnan(c), np.isnan(p))
  np.testing.assert_array_equal(np.isinf(c), np.isinf(p))
  fin = np.isfinite(p)
  np.testing.assert_allclose(c[fin], p[fin], rtol=1e-4, atol=1e-4)

def test_matmul_hook_backward(cu, monkeypatch):
  import torch
  a, b = rnd(33, 47), rnd(47, 29)
  ta, tb = Tensor(a, device="CUDA", requires_grad=True), Tensor(b, device="CUDA", requires_grad=True)
  ((ta @ tb).relu().sum()).backward()
  xa, xb = torch.tensor(a, requires_grad=True), torch.tensor(b, requires_grad=True)
  ((xa @ xb).relu().sum()).backward()
  np.testing.assert_allclose(ta.grad.numpy(), xa.grad.numpy(), **TOL[F32])
  np.testing.assert_allclose(tb.grad.numpy(), xb.grad.numpy(), **TOL[F32])

def test_matmul_hook_jit(cu):
  import fusezap as fz
  @fz.jit
  def f(x: Tensor, y: Tensor) -> Tensor: return (x @ y + 1).relu()
  for _ in range(4):
    a, b = rnd(70, 50), rnd(50, 90)
    np.testing.assert_allclose(f(T(a), T(b)).numpy(), np.maximum(a @ b + 1, 0), rtol=1e-4, atol=1e-4)

# ------------------------------------------------------------------ hook path: f16 / bf16 operands (16-bit kernel)

@pytest.mark.parametrize("dt", [F16, BF16], ids=["f16", "bf16"])
@pytest.mark.parametrize("name", list(FORMS16))
def test_matmul_hook16(cu, name, dt, monkeypatch):
  """16-bit Tensor matmul on CUDA: the scheduled kernels (custom mma16 + the output cast, no input casts) and the values
  against the composed kernels (FZ_CUSTOM=0) and the f64 product of the 16-bit values."""
  fn, shapes, takes = FORMS16[name]
  arrs = [rnd(*s) for s in shapes]
  ks = names(fn(*[T(a, dt) for a in arrs]))
  if takes: assert ks[0] == "fz_matmul_mma16" and "fz_matmul_tf32x3" not in ks, ks
  else: assert "fz_matmul_mma16" not in ks and "fz_matmul_tf32x3" in ks, ks
  if name == "2d": assert ks == ["fz_matmul_mma16", "kernel"], ks   # output cast only
  c = f32(fn(*[T(a, dt) for a in arrs]))
  monkeypatch.setenv("FZ_CUSTOM", "0")
  helpers.getenv.cache_clear()
  p = f32(fn(*[T(a, dt) for a in arrs]))
  monkeypatch.delenv("FZ_CUSTOM")
  helpers.getenv.cache_clear()
  np.testing.assert_allclose(c, p, **TOL[dt])
  import torch
  exact = fn(*[torch.tensor(rounded(a, dt), dtype=torch.float64) for a in arrs]).numpy()
  np.testing.assert_allclose(c, exact, rtol=1e-2, atol=1e-2 * max(1, np.abs(exact).max() / 8))

def test_matmul_hook16_jit_and_disable(cu, monkeypatch):
  import fusezap as fz
  @fz.jit
  def f(x: Tensor, y: Tensor) -> Tensor: return (x @ y + 1).relu()
  for _ in range(4):
    a, b = rnd(70, 64), rnd(64, 96)
    ref = np.maximum(rounded(a, BF16).astype(np.float64) @ rounded(b, BF16) + 1, 0)
    np.testing.assert_allclose(f32(f(T(a, BF16), T(b, BF16))), ref, rtol=1e-2, atol=5e-2)
  monkeypatch.setenv("FZ_MMA16", "0")
  helpers.getenv.cache_clear()
  ck.install(cu)
  try: assert "fz_matmul_mma16" not in names(T(rnd(16, 16), BF16) @ T(rnd(16, 16), BF16))
  finally:
    monkeypatch.delenv("FZ_MMA16")
    helpers.getenv.cache_clear()
    ck.install(cu)

def test_batch_over_grid_limit(cu):
  """Regression: 70000 batch entries (> gridDim.y's 65535) through matmul, the raw mma launcher and the 16-bit hook."""
  a, b = rnd(70000, 1, 64), rnd(64, 64)
  for dt in (BF16, F16, F32):
    ref = np.matmul(rounded(a, dt).astype(np.float64), rounded(b, dt))
    tol = TOL[dt] if dt != F32 else dict(rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(f32(ck.matmul(T(a, dt), T(b, dt))), ref, **tol)
  for impl in ["mma"] + (["wgmma"] if tuple(cu.cc) == (9, 0) else []):
    got, ref = _raw(cu, 3, 72, 64, "bf16", "f32", False, impl, None, Bt=70000)
    np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3 * np.abs(ref).max(), err_msg=impl)
  x, y = rnd(70000, 2, 64), rnd(70000, 64, 16)
  out = T(x, BF16) @ T(y, BF16)
  assert names(T(x, BF16) @ T(y, BF16))[0] == "fz_matmul_mma16"
  np.testing.assert_allclose(f32(out), np.matmul(rounded(x, BF16).astype(np.float64), rounded(y, BF16)), **TOL[BF16])

# ------------------------------------------------------------------ direct path: 16-bit / f32 matmul and linear
SHAPES = [(1, 64, 64, 1), (128, 128, 128, 1), (200, 136, 72, 1), (257, 520, 264, 1), (64, 1000, 4096, 1), (130, 96, 40, 3),
          (1, 4096, 1024, 1), (300, 4, 512, 2)]

@pytest.mark.parametrize("dt", [F16, BF16, F32], ids=["f16", "bf16", "f32"])
@pytest.mark.parametrize("M,N,K,Bt", SHAPES)
def test_direct_matmul(cu, dt, M, N, K, Bt):
  a, b = rnd(Bt, M, K), rnd(Bt, K, N)
  ar, br = rounded(a, dt), rounded(b, dt)
  ref = np.matmul(ar.astype(np.float64), br)
  out = ck.matmul(T(a, dt), T(b, dt))
  assert out.dtype == dt and out.shape == (Bt, M, N)
  tol = TOL[dt] if dt != F32 else dict(rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(f32(out), ref, rtol=tol["rtol"], atol=tol["atol"] * max(1, np.sqrt(K) / 8))
  w = rnd(N, K)
  lin = ck.linear(T(a, dt), T(w, dt))
  np.testing.assert_allclose(f32(lin), np.matmul(ar.astype(np.float64), rounded(w, dt).T), rtol=tol["rtol"], atol=tol["atol"] * max(1, np.sqrt(K) / 8))

@pytest.mark.parametrize("dt", [F16, BF16], ids=["f16", "bf16"])
def test_direct_matches_composed(cu, dt, monkeypatch):
  """The 16-bit kernels against the composed Tensor.matmul (f32 accumulate, one rounding at the end) on the same inputs."""
  monkeypatch.setenv("FZ_CUSTOM", "0")
  helpers.getenv.cache_clear()
  for M, N, K in [(96, 80, 64), (130, 200, 72), (1, 256, 128)]:
    a, b = T(rnd(M, K), dt), T(rnd(K, N), dt)
    np.testing.assert_allclose(f32(ck.matmul(a, b)), f32(a @ b), **TOL[dt])

def test_direct_broadcast_and_unaligned(cu):
  a, b = rnd(3, 70, 48), rnd(48, 100)
  for dt in (F16, BF16):
    np.testing.assert_allclose(f32(ck.matmul(T(a, dt), T(b, dt))), np.matmul(rounded(a, dt), rounded(b, dt)), rtol=1e-2, atol=2e-2)
  a, b = rnd(37, 13), rnd(13, 21)   # K % 8 != 0: 3xTF32 on f32 copies
  np.testing.assert_allclose(f32(ck.matmul(T(a, BF16), T(b, BF16))), rounded(a, BF16) @ rounded(b, BF16), rtol=1e-2, atol=1e-2)

def _raw(cu, M, N, K, kind, out, nn, impl, cfg, Bt=1):
  """Run one kernel config through ck.gemm on raw buffers; returns (result f32, reference f64)."""
  if kind == "e4m3":
    from fusezap.serve.quant import fp8_e4m3_encode, fp8_e4m3_decode
    qa, qb = fp8_e4m3_encode(rnd(Bt, M, K) * 8), fp8_e4m3_encode(rnd(Bt, N, K) * 8)
    A, B = Tensor(qa, device="CUDA", dtype=U8).realize(), Tensor(qb, device="CUDA", dtype=U8).realize()
    ref = np.matmul(fp8_e4m3_decode(qa).astype(np.float64), np.swapaxes(fp8_e4m3_decode(qb), -1, -2)) * 0.25
    alpha = 0.25
  else:
    dt = F16 if kind == "f16" else BF16
    a, b = rnd(Bt, M, K), (rnd(Bt, K, N) if nn else rnd(Bt, N, K))
    A, B = T(a, dt), T(b, dt)
    ref = np.matmul(f32(A).astype(np.float64), f32(B) if nn else np.swapaxes(f32(B), -1, -2))
    alpha = 1.0
  odt = {"f32": F32, "f16": F16, "bf16": BF16}[out]
  C = ck._new("CUDA", Bt * M * N, odt)
  ck.gemm(cu, C.handle, ck._base(A).handle, ck._base(B).handle, M, N, K, kind=kind, out=out, nn=nn, Bt=Bt, sab=M * K,
          sbb=K * N, scb=M * N, alpha=alpha, impl=impl, cfg=cfg)
  return f32(ck._wrap(C, (Bt, M, N))), ref

@pytest.mark.parametrize("kind,nn", [("f16", False), ("bf16", False), ("f16", True), ("bf16", True), ("e4m3", False)])
def test_mma_configs(cu, kind, nn):
  if kind == "e4m3" and tuple(cu.cc) < (8, 9): pytest.skip("fp8 mma needs sm_89+")
  for cfg in ck.mma_candidates(cu, kind, nn):
    got, ref = _raw(cu, 200, 264, 160, kind, "f32", nn, "mma", cfg, Bt=2)
    np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3 * np.abs(ref).max(), err_msg=str(cfg))

@pytest.mark.parametrize("kind,nn", [("f16", False), ("bf16", False), ("f16", True), ("bf16", True), ("e4m3", False)])
def test_mma_small_m_configs(cu, kind, nn):
  """B2-INT: the decode-shape tiles (M <= 64: BM 16 / 32 / 64, BN 32-128, deep K stages) for every candidate, ragged M/N/K,
  plus the dispatch default (impl=None) with outputs in f32 and bf16."""
  if kind == "e4m3" and tuple(cu.cc) < (8, 9): pytest.skip("fp8 mma needs sm_89+")
  for M in (1, 7, 16, 32, 33, 64):
    cs = ck.mma_small_candidates(cu, kind, nn, M)
    assert cs and all(c.BM <= 64 for c in cs)
    for cfg in cs:
      got, ref = _raw(cu, M, 200, 1040, kind, "f32", nn, "mma", cfg, Bt=2)
      np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3 * np.abs(ref).max(), err_msg=f"{cfg} M={M}")
    for out, K in (("f32", 512), ("bf16", 512), ("bf16", 4096)):
      got, ref = _raw(cu, M, 3584, K, kind, out, nn, None, None)
      tol = 2e-3 if out == "f32" else 1e-2
      np.testing.assert_allclose(got, ref, rtol=tol, atol=tol * np.abs(ref).max(), err_msg=f"default M={M} K={K} {out}")
  for M in (1, 32):
    c = ck.mma_small_default(cu, kind, nn, M, 3584, 1, 3584)
    nk = 3584 * ck.ESIZE[kind] // c.BKB   # K stages; each split keeps at least 4
    assert c.BM == (16 if M == 1 else 32) and c.SK > 1 and -(-3584 // c.BN) * c.SK >= min(2 * ck.sm_count(cu), nk // 4 * 3584 // c.BN), c
  cs = [ck.mma_small_default(cu, kind, nn, M, 3584, 1, 3584) for M in (1, 7, 16, 32, 33, 64)]
  assert len({(c.BN, c.BKB, c.SK) for c in cs}) == 1, cs   # the K partition (so each row's sum) never depends on M
  assert ck.mma_small_default(cu, kind, nn, 1, 152064, 1, 3584).SK == 1

@pytest.mark.parametrize("kind,nn", [("bf16", False), ("f16", True), ("e4m3", False)])
def test_mma_split_k(cu, kind, nn):
  """SPEED-2: split-K small-M tiles (grid z, last-arriving CTA sums the f32 partials in split order) for ragged K
  (splits with no K tiles), batches, every output dtype; repeated launches give bit-identical results (the tile
  counters reset themselves), and a row's result does not depend on the other rows of the batch."""
  if kind == "e4m3" and tuple(cu.cc) < (8, 9): pytest.skip("fp8 mma needs sm_89+")
  for M in (1, 7, 32, 33):
    BM = 16 if M <= 16 else 32 if M <= 32 else 64
    for SK in (2, 3, 8, 24):
      cfg = mma.MmaCfg(BM, 64, 1 if BM <= 32 else 2, 2, 128, 4, SK)
      got, ref = _raw(cu, M, 200, 1040, kind, "f32", nn, "mma", cfg, Bt=2)
      np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3 * np.abs(ref).max(), err_msg=f"{cfg} M={M}")
  if kind == "e4m3": return
  dt = F16 if kind == "f16" else BF16
  a, b = rnd(32, 1040), (rnd(1040, 3584) if nn else rnd(3584, 1040))
  A, B = T(a, dt), T(b, dt)
  def run(rows: int, out: str) -> np.ndarray:
    cfg = mma.MmaCfg(16 if rows <= 16 else 32, 128, 1, 4, 128, 4, 6)
    C = ck._new("CUDA", rows * 3584, {"f32": F32, "bf16": BF16}[out])
    ck.gemm(cu, C.handle, ck._base(A).handle, ck._base(B).handle, rows, 3584, 1040, kind=kind, out=out, nn=nn, impl="mma", cfg=cfg)
    return f32(ck._wrap(C, (rows, 3584)))
  for out in ("f32", "bf16"):
    full = run(32, out)
    np.testing.assert_array_equal(full, run(32, out))
    np.testing.assert_array_equal(full[:1], run(1, out))
    ref = np.matmul(f32(A).astype(np.float64), f32(B) if nn else f32(B).T)
    np.testing.assert_allclose(full, ref, rtol=1e-2, atol=1e-2 * np.abs(ref).max())

@pytest.mark.parametrize("kind,nn", [("f16", False), ("bf16", False), ("f16", True), ("bf16", True), ("e4m3", False)])
def test_wgmma_configs(cu, kind, nn):
  if tuple(cu.cc) != (9, 0): pytest.skip("wgmma needs sm_90")
  for cfg in ck.wg_candidates(kind, cu):
    for M, N, K, Bt in [(200, 264, 160, 2), (128, 256, 512, 1), (37, 312, 1040, 1), (1, 72, 64, 3)]:
      got, ref = _raw(cu, M, N, K, kind, "f32", nn, "wgmma", cfg, Bt=Bt)
      np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3 * np.abs(ref).max(), err_msg=f"{cfg} {M}x{N}x{K}x{Bt}")

def test_wgmma_compiles_sm90a(cu):
  """NVRTC compiles every wgmma variant for sm_90a on any NVIDIA machine (catches PTX errors before H100 time)."""
  from fusezap.device.cuda_kernels._rt import compiler
  c = compiler(cu, "sm_90a")
  for kind, out, nn in itertools.product(["f16", "bf16", "e4m3"], ["f32", "bf16"], [False, True]):
    if kind == "e4m3" and nn: continue
    for cfg in ck.wg_candidates(kind): assert len(c.compile_cached(wgmma.render(kind, out, nn, cfg))) > 1000
    if kind != "e4m3" and not nn:
      for cfg in (wgmma.SwapCfg(1, 8, 8, 1), wgmma.SwapCfg(1, 32, 8, 5), wgmma.SwapCfg(2, 40, 4, 3), wgmma.SwapCfg(1, 64, 6, 2)):
        assert len(c.compile_cached(wgmma.render_swap(kind, out, cfg))) > 1000

def test_wg_pick():
  """16-bit M <= 2048 get the WG_TILES entry with the lowest modelled time (waves x area / eff, SPEED-5 / SPEED-R3-step3);
  larger M, fp8 and small shared memory keep wg_default."""
  class D: pass
  d = D()
  d.__dict__.update(_fz_sms=132, _fz_smem=232448)
  W = wgmma.WgCfg
  assert ck.wg_pick(d, "bf16", 256, 4608) == W(192, 1, 6, True, True)                         # qkv: 96 CTAs
  assert ck.wg_pick(d, "bf16", 256, 3584) == W(128, 1, 8, True, True)                         # o / down: 112
  assert ck.wg_pick(d, "bf16", 256, 37888) == ck.wg_pick(d, "bf16", 2048, 4608) == W(192, 2, 4, True)   # 3 full waves; 2.9 vs 2.2
  assert ck.wg_pick(d, "bf16", 512, 3584) == W(128, 2, 5, True)                               # 1 wave of 112 (7B o / down)
  assert ck.wg_pick(d, "f16", 2048, 3584) == ck.wg_pick(d, "bf16", 1024, 37888) == ck.wg_default("bf16")
  assert ck.wg_pick(d, "bf16", 4096, 4608) == ck.wg_default("bf16") and ck.wg_pick(d, "e4m3", 256, 3584) == ck.wg_default("e4m3")
  assert ck.wg_cost(d, W(256, 2, 4, True), 1.0, 128, 256 * 132) == 128 * 256 and ck.wg_cost(d, W(256, 2, 4, True), 1.0, 128, 256 * 66) == 0.8 * 128 * 256
  d.__dict__.update(_fz_smem=101376)
  assert ck.wg_pick(d, "bf16", 256, 3584) == ck.wg_default("bf16")
  assert all(c in ck.WG_CANDIDATES for c in ck.WG_TILES) and len(ck.WG_EFF) == len(ck.WG_TILES)

@pytest.mark.parametrize("nn", [False, True])
def test_wg_pick_bit_identical(cu, nn):
  """Every WG_TILES tile accumulates each element over the same k16 steps in the same order: bit-identical to wg_default."""
  if tuple(cu.cc) != (9, 0): pytest.skip("wgmma needs sm_90")
  for M, N, K in [(256, 3584, 1040), (256, 4608, 512), (200, 1000, 3584), (512, 37888 // 4, 256), (1500, 1152, 320)]:
    a, b = rnd(M, K), (rnd(K, N) if nn else rnd(N, K))
    A, B = T(a, BF16), T(b, BF16)
    def run(cfg, out):
      C = ck._new("CUDA", M * N, {"f32": F32, "bf16": BF16}[out])
      ck.gemm(cu, C.handle, ck._base(A).handle, ck._base(B).handle, M, N, K, kind="bf16", out=out, nn=nn, impl="wgmma", cfg=cfg)
      return f32(ck._wrap(C, (M, N))).view(np.uint32)   # bf16 -> f32 is exact
    for out in ("f32", "bf16"):
      ref = run(ck.wg_default("bf16"), out)
      for cfg in dict.fromkeys(ck.WG_TILES + (ck.wg_pick(cu, "bf16", M, N),)):
        np.testing.assert_array_equal(run(cfg, out), ref, err_msg=f"{cfg} {M}x{N}x{K}")

def test_wg_swap_ab_knobs(monkeypatch):
  """I57 M3's A/B knobs (default off): FZ_WG_SWAP_S deepens the narrow shapes only (as one CTA per SM allows); FZ_WG_SWAP_FILL
  lets the split reach f CTAs per SM; both depend on (N, K) only, never on M (a row's sum stays batch-independent)."""
  from fusezap.helpers import getenv
  class D: pass
  d = D()
  d.__dict__.update(_fz_sms=132, _fz_smem=232448, cc=(9, 0))
  qkv, gu = (5120, 8192), (28672, 8192)   # one TP=2 rank of Llama-3.3-70B
  base = {M: ck.wg_swap_default(d, "bf16", M, *qkv) for M in (1, 8, 32)}
  assert {c.SK for c in base.values()} == {1} and {c.S for c in base.values()} == {8}   # 80 tiles: one CTA on 80 of 132 SMs
  try:
    monkeypatch.setenv("FZ_WG_SWAP_S", "12"); monkeypatch.setenv("FZ_WG_SWAP_FILL", "2"); getenv.cache_clear()
    cs = {M: ck.wg_swap_default(d, "bf16", M, *qkv) for M in (1, 8, 32, 64)}
    assert {c.S for c in cs.values()} == {12} and {c.SK for c in cs.values()} == {3}   # 3 x 80 = 240 CTAs <= 2 x 132
    assert all(wgmma.swap_smem_bytes(c) <= 232448 for c in cs.values())
    assert [ck.wg_swap_default(d, "bf16", M, *gu).S for M in (1, 32)] == [4, 6] and ck.wg_swap_default(d, "bf16", 1, *gu).SK == 1   # wide: unchanged
  finally:
    monkeypatch.delenv("FZ_WG_SWAP_S"); monkeypatch.delenv("FZ_WG_SWAP_FILL"); getenv.cache_clear()
  assert ck.wg_swap_default(d, "bf16", 1, *qkv) == base[1]

def test_wg_swap_default():
  """SPEED-R3: the swap-AB decode tile: BX = M rounded up to 8 (the wgmma n); NC and the split depend only on (N, K)."""
  class D: pass
  d = D()
  d.__dict__.update(_fz_sms=132, _fz_smem=232448, cc=(9, 0))
  cs = {M: ck.wg_swap_default(d, "bf16", M, 3584, 3584) for M in (1, 7, 8, 9, 16, 32, 33, 64)}
  assert [c.BX for c in cs.values()] == [8, 8, 8, 16, 16, 32, 40, 64]
  assert len({(c.NC, c.SK) for c in cs.values()}) == 1 and cs[32].SK == 2 and cs[32].S == 8   # 56 tiles x 2 <= 132 CTAs
  assert ck.wg_swap_default(d, "bf16", 32, 3584, 18944).SK == 2 and ck.wg_swap_default(d, "bf16", 32, 152064, 3584).SK == 1
  assert ck.wg_swap_default(d, "bf16", 32, 4608, 3584).SK == 1 and ck.wg_swap_default(d, "bf16", 1, 896, 4864).SK == 9   # 72 tiles; 14 x 9 <= 132
  # wide shapes (>= 2 waves of tiles) take fewer stages: 7B gate_up / lm_head; qkv / o / down keep 8 (speed run 4)
  assert [ck.wg_swap_default(d, "bf16", M, 37888, 3584).S for M in (1, 8, 32)] == [4, 4, 6] and ck.wg_swap_default(d, "bf16", 32, 152064, 3584).S == 6
  assert ck.wg_swap_default(d, "bf16", 1, 4608, 3584).S == 8 and ck.wg_swap_default(d, "bf16", 32, 3584, 18944).S == 8
  assert all(wgmma.swap_smem_bytes(c) <= 232448 // 2 for c in cs.values())   # two CTAs per SM at every BX
  assert ck.wg_swap_ok(d, "bf16", False, 32) and not ck.wg_swap_ok(d, "bf16", True, 32) and not ck.wg_swap_ok(d, "e4m3", False, 32)
  assert not ck.wg_swap_ok(d, "bf16", False, 65)
  d.cc = (8, 9)
  assert not ck.wg_swap_ok(d, "bf16", False, 32)
  for kind, out in itertools.product(["f16", "bf16"], ["f32", "f16", "bf16"]):
    for c in (cs[1], cs[33], wgmma.SwapCfg(2, 24, 4, 1)):
      src = wgmma.render_swap(kind, out, c)
      assert wgmma.swap_name(kind, out, c) in src and f"m64n{c.BX}k16.f32.{kind}.{kind}" in src and ("atomicAdd" in src) == (c.SK > 1)

@pytest.mark.parametrize("kind", ["f16", "bf16"])
def test_wg_swap_configs(cu, kind):
  """Every swap-AB candidate for ragged M / N / K and batches (both operands batched), f32 out; then the dispatch default
  (impl=None takes the swap kernel on sm_90 for M <= 64) with f32 and 16-bit outputs."""
  if tuple(cu.cc) != (9, 0): pytest.skip("wgmma needs sm_90")
  for M in (1, 7, 16, 32, 33, 64):
    for cfg in ck.wg_swap_candidates(cu, kind, M, 200, 1040, 2)[::3] + [ck.wg_swap_default(cu, kind, M, 200, 1040, 2)]:
      got, ref = _raw(cu, M, 200, 1040, kind, "f32", False, "wgswap", cfg, Bt=2)
      np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3 * np.abs(ref).max(), err_msg=f"{cfg} M={M}")
    for out, N, K in (("f32", 3584, 512), (kind, 3584, 4096), (kind, 4104, 3584)):
      got, ref = _raw(cu, M, N, K, kind, out, False, None, None)
      tol = 2e-3 if out == "f32" else 1e-2
      np.testing.assert_allclose(got, ref, rtol=tol, atol=tol * np.abs(ref).max(), err_msg=f"default M={M} {N}x{K} {out}")

def test_wg_swap_rows_independent(cu):
  """A row's swap-AB result is bit-identical whatever M (so BX) is and over repeated split-K launches (the tile counters
  reset themselves), for the default split and others; the decode shapes of a 7B model."""
  if tuple(cu.cc) != (9, 0): pytest.skip("wgmma needs sm_90")
  for N, K in ((3584, 3584), (4608, 3584), (3584, 18944)):
    A, B = T(rnd(64, K), BF16), T(rnd(N, K), BF16)
    def run(rows: int, out: str, SK: int | None = None) -> np.ndarray:
      cfg = ck.wg_swap_default(cu, "bf16", rows, N, K)
      if SK is not None: cfg = wgmma.SwapCfg(cfg.NC, cfg.BX, cfg.S, SK)
      C = ck._new("CUDA", rows * N, {"f32": F32, "bf16": BF16}[out])
      ck.gemm(cu, C.handle, ck._base(A).handle, ck._base(B).handle, rows, N, K, kind="bf16", out=out, impl="wgswap", cfg=cfg)
      return f32(ck._wrap(C, (rows, N))).view(np.uint32)
    for out in ("f32", "bf16"):
      full = run(32, out)
      np.testing.assert_array_equal(full, run(32, out))
      for rows in (1, 8, 17, 64): np.testing.assert_array_equal(run(rows, out)[:min(rows, 32)], full[:min(rows, 32)], err_msg=f"{N}x{K} M={rows} {out}")
      for SK in (1, 3, 7): np.testing.assert_array_equal(run(1, out, SK), run(32, out, SK)[:1], err_msg=f"{N}x{K} SK={SK} {out}")
    ref = np.matmul(f32(A).astype(np.float64)[:32], f32(B).T)
    np.testing.assert_allclose(full.view(np.float32), ref, rtol=1e-2, atol=1e-2 * np.abs(ref).max())

def test_gemm_table_configs_keep_row_bits(cu, tmp_path, monkeypatch):
  """What the per-shape table (gemm_table, I57 M3b) may vary per BX, the depth S, and per shape, NC, leaves a row's bits
  alone at a given SK, whatever the batch rows; then the autotune on one shape stores an entry that wg_swap_default uses."""
  if tuple(cu.cc) != (9, 0): pytest.skip("wgmma needs sm_90")
  from fusezap.device.cuda_kernels import gemm_table as GT
  N, K = 4096, 4096
  A, B = T(rnd(64, K), BF16), T(rnd(N, K), BF16)
  def run(cfg, rows: int) -> np.ndarray:
    C = ck._new("CUDA", rows * N, F32)
    ck.gemm(cu, C.handle, ck._base(A).handle, ck._base(B).handle, rows, N, K, kind="bf16", out="f32", impl="wgswap", cfg=cfg)
    return f32(ck._wrap(C, (rows, N))).view(np.uint32)
  for SK in (1, 3):
    ref = run(wgmma.SwapCfg(1, 32, 8, SK), 32)
    for NC, S, rows in itertools.product((1, 2), (4, 6, 12), (1, 8, 17, 32)):
      cfg = wgmma.SwapCfg(NC, GT.bx_of(rows), S, SK)
      if wgmma.swap_smem_bytes(cfg) <= ck.smem_limit(cu): np.testing.assert_array_equal(run(cfg, rows), ref[:rows], err_msg=f"{cfg}")
  monkeypatch.setenv("FZ_GEMM_TABLE_FILE", str(tmp_path / "t.json")); GT.reset()
  try:
    rep = GT.autotune(cu, "bf16", [("w", N, K, "f32", [ck._base(B).handle])], [1, 8, 32], log=print, budget_s=300)
    assert rep[0]["status"] == "tuned", rep
    e = rep[0]["entry"]
    assert sum(e["us"].values()) <= 1.02 * sum(e["rule_us"].values()), e   # the rule was a candidate
    c = ck.wg_swap_default(cu, "bf16", 32, N, K)
    assert (c.NC, c.SK, c.S) == (e["NC"], e["SK"], e["S"]["32"])
  finally: GT.reset()

@pytest.mark.parametrize("out", ["f16", "bf16"])
def test_output_dtypes(cu, out):
  impl = "wgmma" if tuple(cu.cc) == (9, 0) else "mma"
  got, ref = _raw(cu, 130, 98, 64, "bf16", out, False, impl, None)
  np.testing.assert_allclose(got, ref, rtol=1e-2, atol=1e-2)

def test_beam_tuning(cu, monkeypatch, tmp_path):
  monkeypatch.setattr(ck, "_TUNE_FILE", tmp_path / "tune.json")
  monkeypatch.setenv("FZ_BEAM", "1")
  helpers.getenv.cache_clear()
  cu.__dict__.pop("_fz_tuned", None)
  got, ref = _raw(cu, 256, 256, 256, "bf16", "f32", False, None, None)
  np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3 * np.abs(ref).max())
  assert (tmp_path / "tune.json").exists() and "|256|256|256" in (tmp_path / "tune.json").read_text()
  cu.__dict__.pop("_fz_tuned", None)

# ------------------------------------------------------------------ fp8 e4m3 (sm_89+)

@pytest.fixture
def cu89(cu):
  if tuple(cu.cc) < (8, 9): pytest.skip("fp8 needs sm_89+")
  return cu

@pytest.mark.parametrize("src", [F32, F16, BF16], ids=["f32", "f16", "bf16"])
def test_quantize_fp8(cu89, src):
  from fusezap.serve.quant import fp8_e4m3_encode, fp8_e4m3_decode
  x = rnd(300, 77) * 5
  xt = T(x, src)
  q, s = ck.quantize_fp8(xt)
  xs = f32(xt)
  sv = float(s.numpy()[0])
  assert sv == pytest.approx(np.abs(xs).max() / 448, rel=1e-6)
  ref = fp8_e4m3_encode((xs / np.float32(sv)).astype(np.float32))
  qn = q.numpy()
  assert q.dtype == U8 and qn.shape == x.shape and (qn == ref).mean() > 0.999
  np.testing.assert_allclose(fp8_e4m3_decode(qn) * sv, xs, rtol=0.07, atol=sv * 2 ** -9)

@pytest.mark.parametrize("M,N,K", [(1, 128, 64), (64, 96, 128), (200, 264, 160), (512, 4608, 3584), (33, 70, 48)])
@pytest.mark.parametrize("out", [F32, BF16], ids=["f32", "bf16"])
def test_fp8_linear(cu89, M, N, K, out):
  from fusezap.serve.quant import fp8_e4m3_encode, fp8_e4m3_decode
  x, w = rnd(M, K) * 3, rnd(N, K) * 0.05
  sx, sw = np.abs(x).max() / 448, np.abs(w).max() / 448
  qx, qw = fp8_e4m3_encode(x / sx), fp8_e4m3_encode(w / sw)
  ref = (fp8_e4m3_decode(qx).astype(np.float64) * sx) @ (fp8_e4m3_decode(qw).astype(np.float64) * sw).T   # dequantized reference
  X, W = Tensor(qx, device="CUDA", dtype=U8), Tensor(qw, device="CUDA", dtype=U8)
  got = f32(ck.fp8_linear(X, W, float(sx), float(sw), out))
  scale = np.abs(ref).max()
  # fp8 tensor cores add with ~14-bit precision inside each promoted K stage: ~1e-4 of the output scale (f32 output)
  np.testing.assert_allclose(got, ref, rtol=1e-3 if out == F32 else 1e-2, atol=(5e-4 if out == F32 else 1e-2) * scale)
  # device scales (no host sync) give the same values
  got2 = f32(ck.fp8_linear(X, W, Tensor(np.array([sx], np.float32), device="CUDA"), Tensor(np.array([sw], np.float32), device="CUDA"), out))
  np.testing.assert_allclose(got2, got, rtol=1e-6, atol=1e-6 * scale)

def test_fp8_end_to_end(cu89):
  """quantize_fp8 on the device, then fp8_linear with the device scales, against the f32 product."""
  x, w = rnd(96, 256), rnd(320, 256) * 0.1
  (qx, sx), (qw, sw) = ck.quantize_fp8(T(x)), ck.quantize_fp8(T(w))
  got = f32(ck.fp8_linear(qx, qw, sx, sw, F32))
  ref = x @ w.T
  assert np.abs(got - ref).max() < 0.08 * np.abs(ref).max()
