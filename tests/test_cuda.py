"""P3-CUDA: CUDARenderer (runs everywhere: source checks) and CUDADevice on hand-built UOp programs, the allocator, timing,
fz.jit replay and FZ_BEAM (need an NVIDIA GPU + cuda-bindings; skipped otherwise). The op suite runs on CUDA with
FZ_TEST_TIERS=CUDA (tests/test_ops.py); test_cuda_tier_is_live fails if that tier was requested but CUDA cannot start."""
from __future__ import annotations
import math, os
import numpy as np
import pytest
import torch
from fusezap.dtype import DType, dtypes, to_np
from fusezap.ops import UnaryOps, BinaryOps, TernaryOps
from fusezap.codegen.uops import UOp, UOps, PtrDType
from fusezap.codegen.cuda import CUDARenderer
from fusezap.device import Device, Buffer
from fusezap.shape.symbolic import Var

F32, I32, F16, BF16, BOOL, I64, I16, I8, U8 = (dtypes.float32, dtypes.int32, dtypes.float16, dtypes.bfloat16, dtypes.bool,
                                              dtypes.int64, dtypes.int16, dtypes.int8, dtypes.uint8)

class B:
  """Tiny UOp program builder (as in tests/test_metal.py)."""
  def __init__(self): self.uops: list[UOp] = []
  def u(self, op, dtype, src=(), arg=None) -> UOp:
    self.uops.append(x := UOp(op, dtype, tuple(src), arg)); return x
  def buf(self, i, dt, out=False): return self.u(UOps.DEFIFZ_GLOBAL, PtrDType(dt), (), (i, f"data{i}", out))
  def var(self, v): return self.u(UOps.DEFIFZ_VAR, I32, (), v)
  def special(self, axis, size, local=False): return self.u(UOps.SPECIAL, I32, (), (axis, f"{'l' if local else 'g'}idx{axis}", size))
  def c(self, v, dt=I32): return self.u(UOps.CONST, dt, (), v)
  def alu(self, op, *src, dt=None): return self.u(UOps.ALU, dt or (BOOL if op in (BinaryOps.CMPLT, BinaryOps.CMPEQ) else src[-1].dtype), src, op)
  def load(self, buf, idx, valid=None, alt=None):
    return self.u(UOps.LOAD, buf.dtype.base, (buf, idx) if valid is None else (buf, idx, valid, alt))
  def store(self, buf, idx, val): return self.u(UOps.STORE, None, (buf, idx, val))
  def cast(self, x, dt): return self.u(UOps.CAST, dt, (x,))
  def bitcast(self, x, dt): return self.u(UOps.BITCAST, dt, (x,))

def bf16_bits(t: torch.Tensor) -> np.ndarray: return t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)

# ---------------------------------------------------------------- renderer (no GPU needed)

def test_kernel_signature_and_specials():
  n = Var("n", 1, 64)
  b = B(); out, x = b.buf(0, F32, True), b.buf(1, F32); b.var(n)
  g, l = b.special(0, n), b.special(0, 32, local=True)
  b.store(out, b.alu(BinaryOps.ADD, g, l), b.alu(UnaryOps.EXP2, b.load(x, g)))
  src = CUDARenderer().render("k", b.uops)
  assert 'extern "C" __global__ void __launch_bounds__(32) k(float* __restrict__ data0, const float* __restrict__ data1, const int n)' in src
  assert "int gidx0 = (int)blockIdx.x;" in src and "int lidx0 = (int)threadIdx.x;" in src and "exp2f(" in src
  assert "#include" not in src and "for (" not in src   # header-free; SPECIALs are hardware indices, not loops

def test_no_launch_bounds_without_locals():
  b = B(); out = b.buf(0, I32, True); g = b.special(1, 7)
  b.store(out, g, g)
  src = CUDARenderer().render("k", b.uops)
  assert "__launch_bounds__" not in src and "(int)blockIdx.y" in src

def test_half_bf16_render():
  b = B(); o16, obf, x16, xbf = b.buf(0, F16, True), b.buf(1, BF16, True), b.buf(2, F16), b.buf(3, BF16)
  g = b.special(0, 4)
  b.store(o16, g, b.alu(BinaryOps.ADD, b.load(x16, g), b.c(0.1, F16)))
  b.store(obf, g, b.alu(BinaryOps.MAX, b.load(xbf, g), b.c(1e9, BF16)))
  src = CUDARenderer().render("k", b.uops)
  assert "unsigned short* __restrict__ data0" in src and "const unsigned short* __restrict__ data3" in src
  assert "fz_h2f(data2[gidx0])" in src and "fz_bf2f(data3[gidx0])" in src
  assert "fz_hround((val0+0.0999755859375f))" in src            # const rounded to f16, the sum rounded to f16
  assert "data1[gidx0] = fz_f2bf(" in src and "998244352.0f" in src   # 1e9 as bf16
  assert "cvt.rn.f16.f32" in src and "fz_f2bf" in src

def test_special_consts_and_int64():
  b = B(); o, oi = b.buf(0, F32, True), b.buf(1, I64, True); g = b.special(0, 1)
  b.store(o, g, b.alu(BinaryOps.ADD, b.c(math.inf, F32), b.alu(BinaryOps.MUL, b.c(-math.inf, F32), b.c(math.nan, F32))))
  b.store(oi, g, b.c(1 << 40, I64))
  src = CUDARenderer().render("k", b.uops)
  assert "((-INFINITY)*NAN)" in src and "(INFINITY+alu0)" in src and "1099511627776ll" in src and "#define INFINITY" in src

def test_bitcast_render():
  b = B(); o, x = b.buf(0, I16, True), b.buf(1, F16); g = b.special(0, 1)
  b.store(o, g, b.bitcast(b.load(x, g), I16))
  assert "fz_bc<short, unsigned short>(fz_f2h(val0))" in CUDARenderer().render("k", b.uops)

def test_warp_reduce_and_shared_hooks():
  r = CUDARenderer()
  b = B(); o, x = b.buf(0, F32, True), b.buf(1, F32); l = b.special(0, 32, local=True)
  b.store(o, l, b.u(UOps.ALU, F32, (b.load(x, l),), "simd_sum"))
  src = r.render("k", b.uops)
  assert "__shfl_xor_sync" in src and "simd_sum(val0)" in src
  assert r.render_local("tile", F16, 64) == "__shared__ unsigned short tile[64];" and r.barrier == "__syncthreads();"

def test_registry_and_tiers(monkeypatch):
  from fusezap.device import _DeviceRegistry
  import conftest
  assert "CUDA" in _DeviceRegistry._names
  monkeypatch.delenv("FZ_TEST_TIERS", raising=False)
  assert "CUDA" not in conftest._selected_tiers()   # the Mac default tier list is unchanged
  monkeypatch.setenv("FZ_TEST_TIERS", "CUDA")
  assert conftest._selected_tiers() == ["CUDA"]

def test_unavailable_raises(monkeypatch):
  import builtins
  from fusezap.device.cuda import CUDADevice
  real = builtins.__import__
  def fake(name, *a, **k):
    if name.startswith("cuda"): raise ImportError("no cuda-bindings")
    return real(name, *a, **k)
  monkeypatch.setattr(builtins, "__import__", fake)
  with pytest.raises(RuntimeError, match="cuda-bindings"): CUDADevice("CUDA")

def _mock_cuda(monkeypatch, ndev=2):
  """A fake cuda.bindings (driver + nvrtc) that tracks the thread's current context and records it at every cuMemAlloc."""
  import sys, types
  drv, nv, cur, log = types.ModuleType("cuda.bindings.driver"), types.ModuleType("cuda.bindings.nvrtc"), [0], []
  class Ctx(int): pass
  def set_cur(c): cur[0] = c; return (0,)
  ns = dict(cuInit=lambda f: (0,), cuDeviceGetCount=lambda: (0, ndev), cuDeviceGet=lambda o: (0, o),
            cuDevicePrimaryCtxRetain=lambda d: (0, Ctx(100 + d)), cuCtxSetCurrent=set_cur, cuCtxGetCurrent=lambda: (0, Ctx(cur[0])),
            cuDeviceGetAttribute=lambda a, d: (0, 8 if a == "MAJOR" else 9), cuDeviceGetName=lambda n, d: (0, b"Mock GPU\0"),
            cuEventCreate=lambda f: (0, object()), cuMemAlloc=lambda n: (log.append(f"ctx{cur[0] - 100}"), (0, 4096 * len(log)))[1],
            cuMemFree=lambda h: (0,), cuCtxSynchronize=lambda: (log.append(f"sync ctx{cur[0] - 100}"), (0,))[1],
            CUdevice_attribute=types.SimpleNamespace(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR="MAJOR",
                                                     CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR="MINOR"))
  for k, v in ns.items(): setattr(drv, k, v)
  nv.nvrtcVersion, nv.nvrtcGetSupportedArchs = lambda: (0, 12, 4), lambda: (0, [80, 89, 90])
  pkg, root = types.ModuleType("cuda.bindings"), types.ModuleType("cuda")
  pkg.driver, pkg.nvrtc, root.bindings = drv, nv, pkg
  for k, m in {"cuda": root, "cuda.bindings": pkg, "cuda.bindings.driver": drv, "cuda.bindings.nvrtc": nv}.items():
    monkeypatch.setitem(sys.modules, k, m)
  return cur, log, set_cur

def test_ctx_switches_between_ordinals(monkeypatch):
  """Regression: the current context is per thread and shared by all devices; CUDA:0 work after CUDA:1 was built (or after
  another library switched contexts) must run in CUDA:0's context, not whichever one was set last."""
  from fusezap.device.cuda import CUDADevice
  cur, log, set_cur = _mock_cuda(monkeypatch)
  d0, d1 = CUDADevice("CUDA:0"), CUDADevice("CUDA:1")
  assert d0.renderer is not None and d0.compiler.arch == "sm_89"
  for d in (d0, d1, d0, d1, d0): d.allocator._alloc(16)
  assert log == ["ctx0", "ctx1", "ctx0", "ctx1", "ctx0"]
  set_cur(100 + 1)   # another library (torch.cuda.set_device, ...) makes GPU 1 current
  d0.synchronize()
  set_cur(0)         # ... or clears the current context
  d1.allocator._alloc(16)
  assert log[-2:] == ["sync ctx0", "ctx1"]

def test_ctx_switches_in_other_threads(monkeypatch):
  import threading
  from fusezap.device.cuda import CUDADevice
  cur, log, set_cur = _mock_cuda(monkeypatch, ndev=1)
  d0 = CUDADevice("CUDA:0")
  set_cur(0)   # a fresh thread starts with no current context (the mock's state is global, so clear it by hand)
  t = threading.Thread(target=lambda: d0.allocator._alloc(16)); t.start(); t.join()
  assert log == ["ctx0"]

# ---------------------------------------------------------------- device (NVIDIA GPU)

def test_cuda_tier_is_live():
  """With FZ_TEST_TIERS naming CUDA, a CUDA that cannot start must fail the run, not skip every op test."""
  if "CUDA" not in [t.strip() for t in os.getenv("FZ_TEST_TIERS", "").split(",")]: pytest.skip("CUDA tier not requested")
  from fusezap.device.cuda import CUDADevice
  d = CUDADevice("CUDA")
  print(f"CUDA device: {d.gpu_name} sm_{d.cc[0]}{d.cc[1]}, NVRTC {d.compiler.version}, target {d.compiler.arch}")

@pytest.fixture(scope="module")
def cu():
  if not Device.available("CUDA"): pytest.skip("CUDA not available")
  return Device["CUDA"]

def mkbuf(a: np.ndarray, dt: DType) -> Buffer:
  b = Buffer("CUDA", a.size, dt)
  return b.copyin(memoryview(np.ascontiguousarray(a.astype(to_np(dt)) if a.dtype != to_np(dt) else a)).cast("B"))

def run(cu, uops, arrays, dtypes_, global_size=(1, 1, 1), local_size=None, out_idx=(0,), vals=(), name="test_kernel"):
  src = cu.renderer.render(name, uops)
  prg = cu.runtime(name, cu.compiler.compile_cached(src))
  bufs = [mkbuf(a, dt) for a, dt in zip(arrays, dtypes_)]
  prg(*[b.handle for b in bufs], global_size=global_size, local_size=local_size, vals=vals)
  return [bufs[i].numpy() for i in out_idx], src

def test_device_props(cu):
  from fusezap.device.cuda import CUDADevice
  assert isinstance(cu, CUDADevice) and isinstance(cu.renderer, CUDARenderer) and cu.renderer.has_local
  assert cu.cc >= (7, 5) and cu.compiler.arch.endswith(f"{cu.cc[0]}{cu.cc[1]}") or cu.compiler.arch.startswith("compute_")
  assert cu.compiler.arch in cu.compiler.cache_key() and "--fmad=false" in cu.compiler.cache_key()

def test_elementwise_three_dims_locals(cu):
  A, R, C, L = 3, 5, 7, 32
  b = B(); out, x = b.buf(0, F32, True), b.buf(1, F32)
  g0, g1, g2, l = b.special(0, A), b.special(1, R), b.special(2, C), b.special(0, L, local=True)
  xi = b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, g0, b.c(R)), g1), b.c(C)), g2), b.c(L)), l)
  b.store(out, xi, b.alu(TernaryOps.WHERE, b.alu(BinaryOps.CMPLT, l, b.c(5)), b.alu(BinaryOps.MUL, b.load(x, xi), b.c(2.0, F32)), b.load(x, xi)))
  xa = np.random.randn(A * R * C * L).astype(np.float32)
  (o,), _ = run(cu, b.uops, [np.zeros_like(xa), xa], [F32] * 2, global_size=(A, R, C), local_size=(L, 1, 1))
  np.testing.assert_array_equal(o, np.where(np.arange(xa.size) % L < 5, xa * 2, xa))

@pytest.mark.parametrize("kind", ["sum", "max"])
def test_reduction_and_guards(cu, kind):
  N, M = 6, 333
  b = B(); out, o2, x = b.buf(0, F32, True), b.buf(1, F32, True), b.buf(2, F32)
  g = b.special(0, N)
  acc = b.u(UOps.DEFIFZ_ACC, F32, (), 0.0 if kind == "sum" else float("-inf"))
  r = b.u(UOps.RANGE, I32, (b.c(0), b.c(M)))
  v = b.load(x, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, g, b.c(M)), r))
  phi = b.u(UOps.PHI, F32, (acc, b.alu(BinaryOps.ADD if kind == "sum" else BinaryOps.MAX, acc, v)))
  b.u(UOps.ENDRANGE, None, (r,))
  b.store(out, g, phi)
  i = b.u(UOps.IF, None, (b.alu(BinaryOps.CMPEQ, b.alu(BinaryOps.MOD, g, b.c(2)), b.c(0)),))
  b.store(o2, g, b.load(x, b.alu(BinaryOps.ADD, g, b.c(3)), b.alu(BinaryOps.CMPLT, g, b.c(N - 3)), b.c(-1.5, F32)))
  b.u(UOps.ENDIF, None, (i,))
  xa = np.random.randn(N, M).astype(np.float32)
  (o, p), _ = run(cu, b.uops, [np.zeros(N, np.float32), np.zeros(N, np.float32), xa.ravel()], [F32] * 3, global_size=(N, 1, 1), out_idx=(0, 1))
  t = torch.from_numpy(xa)
  np.testing.assert_allclose(o, (t.sum(1) if kind == "sum" else t.amax(1)).numpy(), rtol=1e-4, atol=1e-5)
  np.testing.assert_array_equal(p, [xa.ravel()[3], 0, xa.ravel()[5], 0, -1.5, 0])

def test_unary_math_and_special_values(cu):
  N = 256
  ops = [UnaryOps.NEG, UnaryOps.EXP2, UnaryOps.LOG2, UnaryOps.SIN, UnaryOps.SQRT, UnaryOps.RECIP]
  b = B(); outs = [b.buf(i, F32, True) for i in range(len(ops))]; x = b.buf(len(ops), F32)
  g = b.special(0, N)
  v = b.load(x, g)
  for o, op in zip(outs, ops): b.store(o, g, b.alu(op, v))
  xa = np.random.uniform(0.1, 20.0, N).astype(np.float32)
  xa[:5] = [np.inf, -np.inf, np.nan, 0.0, -1.0]
  res, _ = run(cu, b.uops, [np.zeros(N, np.float32)] * len(ops) + [xa], [F32] * (len(ops) + 1), global_size=(N, 1, 1), out_idx=range(len(ops)))
  t = torch.from_numpy(xa)
  for o, ref in zip(res, [-t, torch.exp2(t), torch.log2(t), torch.sin(t), torch.sqrt(t), 1 / t]):
    np.testing.assert_allclose(o, ref.numpy(), rtol=1e-5, atol=1e-6)

def test_casts_ints_bitcast(cu):
  N = 9
  b = B()
  o_i, o_b, o_h, o_bc, o_i8, o_u8, o_l, o_d, x, xi = (b.buf(0, I32, True), b.buf(1, BOOL, True), b.buf(2, F16, True), b.buf(3, I32, True),
                                                      b.buf(4, I8, True), b.buf(5, U8, True), b.buf(6, I64, True), b.buf(7, I32, True), b.buf(8, F32), b.buf(9, I32))
  g = b.special(0, N)
  v, vi = b.load(x, g), b.load(xi, g)
  b.store(o_i, g, b.cast(v, I32)); b.store(o_b, g, b.cast(v, BOOL)); b.store(o_h, g, b.cast(v, F16)); b.store(o_bc, g, b.bitcast(v, I32))
  b.store(o_i8, g, b.cast(b.alu(BinaryOps.MOD, vi, b.c(100)), I8)); b.store(o_u8, g, b.cast(b.alu(BinaryOps.MOD, vi, b.c(100)), U8))
  b.store(o_l, g, b.alu(BinaryOps.MUL, b.cast(vi, I64), b.c(1 << 20, I64)))
  b.store(o_d, g, b.alu(BinaryOps.IDIV, vi, b.c(7)))
  xa = np.array([-2.7, -1.5, -0.5, 0.0, 0.5, 1.5, 2.7, 1e5, -0.0], np.float32)
  xia = np.array([-7, -1, 0, 1, 2, 3, 1 << 20, -(1 << 24) - 1, 123456789], np.int32)
  outs = (I32, BOOL, F16, I32, I8, U8, I64, I32)
  res, _ = run(cu, b.uops, [np.zeros(N, to_np(d)) for d in outs] + [xa, xia], [*outs, F32, I32], global_size=(N, 1, 1), out_idx=range(8))
  t, ti = torch.from_numpy(xa), torch.from_numpy(xia)
  np.testing.assert_array_equal(res[0], t.to(torch.int32).numpy()); np.testing.assert_array_equal(res[1], t.bool().numpy())
  np.testing.assert_array_equal(res[2], t.half().numpy()); np.testing.assert_array_equal(res[3], xa.view(np.int32))
  np.testing.assert_array_equal(res[4], torch.fmod(ti, 100).to(torch.int8).numpy()); np.testing.assert_array_equal(res[5], torch.fmod(ti, 100).to(torch.uint8).numpy())
  np.testing.assert_array_equal(res[6], ti.long().numpy() << 20); np.testing.assert_array_equal(res[7], torch.div(ti, 7, rounding_mode="trunc").numpy())

def test_float16_bfloat16(cu):
  N = 64
  b = B(); o16, obf, ob16, o16bits, x16, xbf = b.buf(0, F16, True), b.buf(1, BF16, True), b.buf(2, BF16, True), b.buf(3, I16, True), b.buf(4, F16), b.buf(5, BF16)
  g = b.special(0, N)
  h, bf = b.load(x16, g), b.load(xbf, g)
  b.store(o16, g, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, h, h), b.alu(UnaryOps.EXP2, h)))
  b.store(obf, g, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, bf, bf), b.c(0.1, BF16)))
  b.store(ob16, g, b.cast(h, BF16)); b.store(o16bits, g, b.bitcast(h, I16))
  xa = np.random.uniform(-3, 3, N).astype(np.float32)
  th, tb = torch.from_numpy(xa).half(), torch.from_numpy(xa).bfloat16()
  res, _ = run(cu, b.uops, [np.zeros(N, np.float16), np.zeros(N, np.uint16), np.zeros(N, np.uint16), np.zeros(N, np.int16), th.numpy(), bf16_bits(tb)],
               [F16, BF16, BF16, I16, F16, BF16], global_size=(N, 1, 1), out_idx=range(4))
  np.testing.assert_allclose(res[0], (th * th + torch.exp2(th)).numpy(), rtol=1e-3, atol=1e-3)
  np.testing.assert_array_equal(res[1], bf16_bits(tb * tb + torch.tensor(0.1).bfloat16()))
  np.testing.assert_array_equal(res[2], bf16_bits(th.bfloat16())); np.testing.assert_array_equal(res[3], th.numpy().view(np.int16))

def test_symbolic_var_and_launch_errors(cu):
  n = Var("n", 1, 1000)
  b = B(); out, x = b.buf(0, F32, True), b.buf(1, F32); nv = b.var(n)
  g = b.special(0, n)
  b.store(out, g, b.alu(BinaryOps.ADD, b.load(x, g), b.cast(nv, F32)))
  src = cu.renderer.render("symk", b.uops)
  prg = cu.runtime("symk", cu.compiler.compile_cached(src))
  xa = np.random.randn(1000).astype(np.float32)
  ob, xb = mkbuf(np.zeros(1000, np.float32), F32), mkbuf(xa, F32)
  for nn in (1, 17, 1000):
    prg(ob.handle, xb.handle, global_size=(nn, 1, 1), vals=(nn,))
    np.testing.assert_array_equal(ob.numpy()[:nn], xa[:nn] + nn)
  t = prg(ob.handle, xb.handle, global_size=(1000, 1, 1), vals=(1000,), wait=True)
  assert isinstance(t, float) and 0 < t < 1
  assert prg(ob.handle, xb.handle, global_size=(0, 1, 1), vals=(0,), wait=True) == 0.0
  with pytest.raises(RuntimeError, match="exceeds"): prg(ob.handle, xb.handle, global_size=(1, 1, 1), local_size=(2048, 1, 1), vals=(1,))
  with pytest.raises(RuntimeError, match="NVRTC compile failed"): cu.compiler.compile('extern "C" __global__ void bad() { nope; }')
  cu.synchronize()

def test_shared_memory_block_reduce(cu):
  """render_local + barrier: a hand-written 256-thread tree reduction through shared memory."""
  r = cu.renderer
  src = f"""extern "C" __global__ void __launch_bounds__(256) blocksum(float* out, const float* x) {{
  {r.render_local("s", F32, 256)}
  int t = threadIdx.x;
  s[t] = x[blockIdx.x * 256 + t];
  {r.barrier}
  for (int k = 128; k > 0; k >>= 1) {{ if (t < k) s[t] += s[t + k]; {r.barrier} }}
  if (t == 0) out[blockIdx.x] = s[0];
}}"""
  prg = cu.runtime("blocksum", cu.compiler.compile(src))
  xa = np.random.randn(8 * 256).astype(np.float32)
  ob, xb = mkbuf(np.zeros(8, np.float32), F32), mkbuf(xa, F32)
  prg(ob.handle, xb.handle, global_size=(8, 1, 1), local_size=(256, 1, 1))
  np.testing.assert_allclose(ob.numpy(), xa.reshape(8, 256).sum(1), rtol=1e-5, atol=1e-5)

def test_warp_reduce(cu):
  b = B(); o, x = b.buf(0, F32, True), b.buf(1, F32); g, l = b.special(0, 4), b.special(0, 32, local=True)
  v = b.load(x, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, g, b.c(32)), l))
  s, m = b.u(UOps.ALU, F32, (v,), "simd_sum"), b.u(UOps.ALU, F32, (v,), "simd_max")
  i = b.u(UOps.IF, None, (b.alu(BinaryOps.CMPEQ, l, b.c(0)),))
  b.store(o, b.alu(BinaryOps.MUL, g, b.c(2)), s); b.store(o, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, g, b.c(2)), b.c(1)), m)
  b.u(UOps.ENDIF, None, (i,))
  xa = np.random.randn(128).astype(np.float32)
  (out,), _ = run(cu, b.uops, [np.zeros(8, np.float32), xa], [F32] * 2, global_size=(4, 1, 1), local_size=(32, 1, 1))
  np.testing.assert_allclose(out[0::2], xa.reshape(4, 32).sum(1), rtol=1e-5, atol=1e-5); np.testing.assert_array_equal(out[1::2], xa.reshape(4, 32).max(1))

def test_allocator_partial_copies_and_lru(cu):
  a = cu.allocator
  h = a.alloc(1 << 20)
  src = np.arange(1 << 18, dtype=np.float32)
  a.copyin(h, memoryview(src).cast("B"))
  part = np.zeros(1000, np.float32)
  a.copyout(memoryview(part).cast("B"), h)          # a prefix, as a jit replay copies
  np.testing.assert_array_equal(part, src[:1000])
  ro = memoryview(np.full(4, 7, np.float32).tobytes())   # read-only source
  a.copyin(h, ro); a.copyout(memoryview(part).cast("B"), h)
  np.testing.assert_array_equal(part[:5], [7, 7, 7, 7, 4])
  a.free(h, 1 << 20)
  assert a.alloc(1 << 20) == h                       # served from the LRU cache
  a.free(h, 1 << 20)

def test_tensor_end_to_end(cu):
  from fusezap import Tensor
  a, w = np.random.randn(33, 70).astype(np.float32), np.random.randn(70, 19).astype(np.float32)
  x = Tensor(a, device="CUDA")
  out = ((x @ Tensor(w, device="CUDA")).relu().softmax(-1) + x.sum(-1, keepdim=True).exp() * 0).numpy()
  ref = torch.softmax(torch.relu(torch.from_numpy(a) @ torch.from_numpy(w)), -1).numpy()
  np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)
  y = Tensor(a, device="CUDA").to("CPU" if Device.available("CPU") else "NPY").numpy()
  np.testing.assert_array_equal(y, a)

def test_jit_replay(cu):
  from fusezap import Tensor
  from fusezap.jit import jit, bind
  import fusezap.schedule
  @jit
  def f(x: Tensor, y: Tensor) -> Tensor: return (x * y + 1).sum(-1).sqrt()
  n_sched, orig = [0], fusezap.schedule.create_schedule
  for i in range(5):
    a, b = np.random.rand(8, 16).astype(np.float32), np.random.rand(8, 16).astype(np.float32)
    xa, xb = Tensor(a, device="CUDA").realize(), Tensor(b, device="CUDA").realize()
    if i == 3:
      fusezap.schedule.create_schedule = lambda *a_, **k: (n_sched.__setitem__(0, n_sched[0] + 1), orig(*a_, **k))[1]
    try: out = f(xa, xb).numpy()
    finally: fusezap.schedule.create_schedule = orig
    np.testing.assert_allclose(out, np.sqrt((a * b + 1).sum(-1)), rtol=1e-5, atol=1e-6)
  assert n_sched[0] == 0   # replays do not schedule
  # symbolic length
  v = Var("n", 1, 64)
  @jit
  def g(x: Tensor, n) -> Tensor: return (x.shrink(((0, n),)) * 2.0).sum().reshape(1)
  base = np.random.rand(64).astype(np.float32)
  xb = Tensor(base, device="CUDA").realize()
  for nn in (5, 9, 13, 64, 1):
    np.testing.assert_allclose(g(xb, bind(v, nn)).numpy(), [base[:nn].sum() * 2], rtol=1e-5)
  # symbolic output shape
  h = jit(lambda x, n: x.shrink(((0, n),)) + 1.0)
  for nn in (4, 4, 2, 64, 7): np.testing.assert_array_equal(h(xb, v, var_vals={v: nn}).numpy(), base[:nn] + 1)

def test_beam_on_cuda(cu, tmp_path, monkeypatch):
  import fusezap.helpers as h
  from fusezap import Tensor
  from fusezap.codegen.lower import _cache, heuristic_opts, check_opts
  monkeypatch.setattr(h, "CACHE_DIR", tmp_path)
  monkeypatch.setenv("FZ_BEAM", "2"); h.getenv.cache_clear()
  _cache.clear()
  try:
    a, b = np.random.randn(64, 96).astype(np.float32), np.random.randn(96, 48).astype(np.float32)
    out = (Tensor(a, device="CUDA") @ Tensor(b, device="CUDA")).relu().sum(axis=0).numpy()
    np.testing.assert_allclose(out, np.maximum(a @ b, 0).sum(0), rtol=1e-4, atol=1e-4)
    assert any((tmp_path / "beam").iterdir())
    monkeypatch.setenv("FZ_BEAM", "0"); monkeypatch.setenv("FZ_OPT", "1"); h.getenv.cache_clear()
    np.testing.assert_allclose((Tensor(a, device="CUDA") @ Tensor(b, device="CUDA")).numpy(), a @ b, rtol=1e-4, atol=1e-4)
  finally:
    _cache.clear(); h.getenv.cache_clear()

def test_ctx_restored_after_foreign_context(cu):
  """Regression on hardware: another context made current in this thread (as another library might) must not capture
  CUDA work: the next fusezap call switches back to the device's primary context, and its buffers stay usable."""
  from fusezap.device.cuda import _check, _ptr
  drv = cu.cu
  a = np.arange(64, dtype=np.float32)
  b = mkbuf(a, F32)
  try: other = _check(drv.cuCtxCreate(None, 0, cu.device), "cuCtxCreate")   # cuda-bindings 13: (params, flags, dev); makes it current
  except TypeError: other = _check(drv.cuCtxCreate(0, cu.device), "cuCtxCreate")
  try:
    assert _ptr(_check(drv.cuCtxGetCurrent())) != _ptr(cu.context)
    np.testing.assert_array_equal(b.numpy(), a)
    assert _ptr(_check(drv.cuCtxGetCurrent())) == _ptr(cu.context)
    c = mkbuf(a * 2, F32)
    _check(drv.cuCtxSetCurrent(other))
    np.testing.assert_array_equal(c.numpy(), a * 2)
    _check(drv.cuCtxSetCurrent(drv.CUcontext(0)))   # no context current at all
    np.testing.assert_array_equal(b.numpy(), a)
    assert _ptr(_check(drv.cuCtxGetCurrent())) == _ptr(cu.context)
  finally:
    _check(drv.cuCtxSetCurrent(cu.context)); drv.cuCtxDestroy(other)

def test_two_gpus_keep_their_buffers():
  """With two GPUs: interleaved CUDA:0 / CUDA:1 allocations and kernels each land on their own device."""
  if not Device.available("CUDA"): pytest.skip("CUDA not available")
  from fusezap.device.cuda import _check
  d0 = Device["CUDA:0"]
  if int(_check(d0.cu.cuDeviceGetCount())) < 2: pytest.skip("needs 2 GPUs")
  from fusezap.tensor import Tensor
  x0 = Tensor(np.arange(8, dtype=np.float32), device="CUDA:0")
  x1 = Tensor(np.arange(8, dtype=np.float32) * 10, device="CUDA:1")
  y0 = (x0 + 1).realize(); y1 = (x1 + 1).realize(); z0 = (y0 * 2).realize()
  for t, dev in ((y0, "CUDA:0"), (y1, "CUDA:1"), (z0, "CUDA:0")):
    attrs = _check(d0.cu.cuPointerGetAttribute(d0.cu.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, t.lazydata.base.realized.handle))
    assert int(attrs) == int(dev.split(":")[1])
  np.testing.assert_array_equal(z0.numpy(), (np.arange(8) + 1) * 2)
  np.testing.assert_array_equal(y1.numpy(), np.arange(8) * 10 + 1)
