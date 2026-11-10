"""WP4a: CStyleRenderer/CRenderer + CPUDevice on hand-built UOp programs (no lowering involved)."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from fusezap.dtype import DType, dtypes, to_np
from fusezap.ops import UnaryOps, BinaryOps, TernaryOps
from fusezap.codegen.uops import UOp, UOps, PtrDType
from fusezap.codegen.cstyle import CStyleRenderer, CRenderer
from fusezap.device import Device, Buffer
from fusezap.device.cpu import CPUDevice, CPUProgram, ClangCompiler

F32, I32, F16, BF16, BOOL = dtypes.float32, dtypes.int32, dtypes.float16, dtypes.bfloat16, dtypes.bool

def make_var(name: str, lo: int, hi: int):
  """Real symbolic Var when WP1 has landed, else a minimal stand-in (the renderer only needs `.name`)."""
  try:
    from fusezap.shape.symbolic import Var
    return Var(name, lo, hi)
  except NotImplementedError:
    class FakeVar:
      def __init__(self, name, vmin, vmax): self.name, self.vmin, self.vmax = name, vmin, vmax
      def __repr__(self): return f"Var({self.name})"
    return FakeVar(name, lo, hi)

class B:
  """Tiny UOp program builder."""
  def __init__(self): self.uops: list[UOp] = []
  def u(self, op, dtype, src=(), arg=None) -> UOp:
    self.uops.append(x := UOp(op, dtype, tuple(src), arg)); return x
  def buf(self, i, dt, out=False): return self.u(UOps.DEFIFZ_GLOBAL, PtrDType(dt), (), (i, f"data{i}", out))
  def var(self, v): return self.u(UOps.DEFIFZ_VAR, I32, (), v)
  def special(self, axis, size): return self.u(UOps.SPECIAL, I32, (), (axis, f"gidx{axis}", size))
  def c(self, v, dt=I32): return self.u(UOps.CONST, dt, (), v)
  def alu(self, op, *src, dt=None): return self.u(UOps.ALU, dt or (BOOL if op in (BinaryOps.CMPLT, BinaryOps.CMPEQ) else src[-1].dtype), src, op)
  def load(self, buf, idx, valid=None, alt=None):
    return self.u(UOps.LOAD, buf.dtype.base, (buf, idx) if valid is None else (buf, idx, valid, alt))
  def store(self, buf, idx, val): return self.u(UOps.STORE, None, (buf, idx, val))
  def cast(self, x, dt): return self.u(UOps.CAST, dt, (x,))
  def bitcast(self, x, dt): return self.u(UOps.BITCAST, dt, (x,))

@pytest.fixture(scope="module")
def cpu() -> CPUDevice: return Device["CPU"]

def run(cpu, uops, arrays: list[np.ndarray], dtypes_: list[DType], out_idx=(0,), vals=(), name="test_kernel"):
  src = cpu.renderer.render(name, uops)
  prg = cpu.runtime(name, cpu.compiler.compile_cached(src))
  bufs = []
  for a, dt in zip(arrays, dtypes_):
    b = Buffer("CPU", a.size, dt)
    b.copyin(memoryview(np.ascontiguousarray(a.astype(to_np(dt)) if a.dtype != to_np(dt) else a)).cast("B"))
    bufs.append(b)
  prg(*[b.handle for b in bufs], global_size=(1, 1, 1), local_size=None, vals=vals)
  return [bufs[i].numpy() for i in out_idx], src

def bf16_bits(t: torch.Tensor) -> np.ndarray: return t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
def from_bf16_bits(a: np.ndarray) -> torch.Tensor: return torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16)

# ---------------------------------------------------------------- tests

def test_device_registry(cpu):
  assert isinstance(cpu, CPUDevice) and isinstance(cpu.renderer, CRenderer)
  assert not cpu.renderer.has_local and cpu.renderer.device == "CPU"
  assert Device.available("CPU")

def test_elementwise_add_mul_where(cpu):
  N = 37
  b = B(); out, x, y = b.buf(0, F32, True), b.buf(1, F32), b.buf(2, F32)
  g = b.special(0, N)
  a, c = b.load(x, g), b.load(y, g)
  b.store(out, g, b.alu(TernaryOps.WHERE, b.alu(BinaryOps.CMPLT, a, c), b.alu(BinaryOps.ADD, a, c), b.alu(BinaryOps.MUL, a, c)))
  xa, ya = np.random.randn(N).astype(np.float32), np.random.randn(N).astype(np.float32)
  (o,), src = run(cpu, b.uops, [np.zeros(N, np.float32), xa, ya], [F32] * 3)
  tx, ty = torch.from_numpy(xa), torch.from_numpy(ya)
  np.testing.assert_allclose(o, torch.where(tx < ty, tx + ty, tx * ty).numpy(), rtol=1e-5, atol=1e-6)
  assert "for (int gidx0 = 0; gidx0 < 37; gidx0++)" in src and "float* restrict data0" in src

def test_two_special_dims(cpu):
  R, C = 5, 7
  b = B(); out, x = b.buf(0, F32, True), b.buf(1, F32)
  g0, g1 = b.special(0, R), b.special(1, C)
  # out[g1*R + g0] = x[g0*C + g1]  (transpose)
  v = b.load(x, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, g0, b.c(C)), g1))
  b.store(out, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, g1, b.c(R)), g0), v)
  xa = np.random.randn(R, C).astype(np.float32)
  (o,), _ = run(cpu, b.uops, [np.zeros(R * C, np.float32), xa.ravel()], [F32] * 2)
  np.testing.assert_array_equal(o.reshape(C, R), xa.T)

@pytest.mark.parametrize("kind", ["sum", "max"])
def test_reduction_acc_range_phi(cpu, kind):
  N, M = 6, 33
  b = B(); out, x = b.buf(0, F32, True), b.buf(1, F32)
  g = b.special(0, N)
  acc = b.u(UOps.DEFIFZ_ACC, F32, (), 0.0 if kind == "sum" else float("-inf"))
  r = b.u(UOps.RANGE, I32, (b.c(0), b.c(M)))
  v = b.load(x, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, g, b.c(M)), r))
  phi = b.u(UOps.PHI, F32, (acc, b.alu(BinaryOps.ADD if kind == "sum" else BinaryOps.MAX, acc, v)))
  b.u(UOps.ENDRANGE, None, (r,))
  b.store(out, g, phi)
  xa = np.random.randn(N, M).astype(np.float32)
  (o,), src = run(cpu, b.uops, [np.zeros(N, np.float32), xa.ravel()], [F32] * 2)
  t = torch.from_numpy(xa)
  ref = t.sum(1) if kind == "sum" else t.amax(1)
  np.testing.assert_allclose(o, ref.numpy(), rtol=1e-4, atol=1e-5)
  if kind == "max": assert "-INFINITY" in src

def test_guarded_load(cpu):
  # out[i] = i < N-3 ? x[i+3] : -1.5   (x has only N-3 elements: the guard must prevent the OOB read)
  N = 20
  b = B(); out, x = b.buf(0, F32, True), b.buf(1, F32)
  g = b.special(0, N)
  valid = b.alu(BinaryOps.CMPLT, g, b.c(N - 3))
  v = b.load(x, b.alu(BinaryOps.ADD, g, b.c(3)), valid, b.c(-1.5, F32))
  b.store(out, g, v)
  xa = np.random.randn(N).astype(np.float32)
  (o,), _ = run(cpu, b.uops, [np.zeros(N, np.float32), xa], [F32] * 2)
  np.testing.assert_array_equal(o, np.concatenate([xa[3:], np.full(3, -1.5, np.float32)]))

def test_if_guarded_store(cpu):
  N = 16
  b = B(); out, x = b.buf(0, F32, True), b.buf(1, F32)
  g = b.special(0, N)
  i = b.u(UOps.IF, None, (b.alu(BinaryOps.CMPEQ, b.alu(BinaryOps.MOD, g, b.c(2)), b.c(0)),))
  b.store(out, g, b.load(x, g))
  b.u(UOps.ENDIF, None, (i,))
  xa = np.random.randn(N).astype(np.float32)
  (o,), _ = run(cpu, b.uops, [np.zeros(N, np.float32), xa], [F32] * 2)
  np.testing.assert_array_equal(o, np.where(np.arange(N) % 2 == 0, xa, 0))

def test_unary_math(cpu):
  N = 64
  ops = [UnaryOps.NEG, UnaryOps.EXP2, UnaryOps.LOG2, UnaryOps.SIN, UnaryOps.SQRT, UnaryOps.RECIP]
  b = B(); outs = [b.buf(i, F32, True) for i in range(len(ops))]; x = b.buf(len(ops), F32)
  g = b.special(0, N)
  v = b.load(x, g)
  for o, op in zip(outs, ops): b.store(o, g, b.alu(op, v))
  xa = np.random.uniform(0.1, 4.0, N).astype(np.float32)
  res, _ = run(cpu, b.uops, [np.zeros(N, np.float32)] * len(ops) + [xa], [F32] * (len(ops) + 1), out_idx=range(len(ops)))
  t = torch.from_numpy(xa)
  refs = [-t, torch.exp2(t), torch.log2(t), torch.sin(t), torch.sqrt(t), 1 / t]
  for o, ref in zip(res, refs): np.testing.assert_allclose(o, ref.numpy(), rtol=1e-5, atol=1e-6)

def test_special_values(cpu):
  N = 4
  b = B(); out, x = b.buf(0, F32, True), b.buf(1, F32)
  g = b.special(0, N)
  v = b.load(x, g)
  b.store(out, g, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MAX, v, b.c(float("-inf"), F32)), b.c(1e-7, F32)))
  xa = np.array([np.inf, -np.inf, np.nan, 2.0], np.float32)
  (o,), _ = run(cpu, b.uops, [np.zeros(N, np.float32), xa], [F32] * 2)
  # MAX is spec'd as `a > b ? a : b`, so MAX(nan, -inf) == -inf (unlike torch.maximum)
  assert o[0] == np.inf and o[1] == -np.inf and o[2] == -np.inf and o[3] == np.float32(2.0) + np.float32(1e-7)

def test_casts(cpu):
  N = 9
  b = B()
  o_i, o_f, o_b, o_bf, o_h, o_bc, x, xi = (b.buf(0, I32, True), b.buf(1, F32, True), b.buf(2, BOOL, True), b.buf(3, F32, True),
                                            b.buf(4, F16, True), b.buf(5, I32, True), b.buf(6, F32), b.buf(7, I32))
  g = b.special(0, N)
  v, vi = b.load(x, g), b.load(xi, g)
  b.store(o_i, g, b.cast(v, I32))                       # float -> int truncates toward zero
  b.store(o_f, g, b.cast(vi, F32))
  bl = b.cast(v, BOOL)
  b.store(o_b, g, bl)                                   # -> bool is x != 0
  b.store(o_bf, g, b.cast(bl, F32))
  b.store(o_h, g, b.cast(v, F16))
  b.store(o_bc, g, b.bitcast(v, I32))
  xa = np.array([-2.7, -1.5, -0.5, 0.0, 0.5, 1.5, 2.7, 1e3, -0.0], np.float32)
  xia = np.array([-7, -1, 0, 1, 2, 3, 1 << 20, -(1 << 24) - 1, 123456789], np.int32)
  res, _ = run(cpu, b.uops, [np.zeros(N, to_np(d)) for d in (I32, F32, BOOL, F32, F16, I32)] + [xa, xia],
               [I32, F32, BOOL, F32, F16, I32, F32, I32], out_idx=range(6))
  t, ti = torch.from_numpy(xa), torch.from_numpy(xia)
  np.testing.assert_array_equal(res[0], t.to(torch.int32).numpy())
  np.testing.assert_array_equal(res[1], ti.to(torch.float32).numpy())
  np.testing.assert_array_equal(res[2], t.to(torch.bool).numpy())
  np.testing.assert_array_equal(res[3], t.to(torch.bool).to(torch.float32).numpy())
  np.testing.assert_array_equal(res[4], t.to(torch.float16).numpy())
  np.testing.assert_array_equal(res[5], xa.view(np.int32))

def test_int_idiv_mod(cpu):
  N = 12
  b = B(); od, om, x, y = b.buf(0, I32, True), b.buf(1, I32, True), b.buf(2, I32), b.buf(3, I32)
  g = b.special(0, N)
  a, c = b.load(x, g), b.load(y, g)
  b.store(od, g, b.alu(BinaryOps.IDIV, a, c)); b.store(om, g, b.alu(BinaryOps.MOD, a, c))
  xa = np.array([7, -7, 7, -7, 0, 1, -1, 100, -100, 13, 2147483647, -2147483647], np.int32)
  ya = np.array([2, 2, -2, -2, 3, 5, 5, 7, 7, -4, 10, 10], np.int32)
  (d, m), _ = run(cpu, b.uops, [np.zeros(N, np.int32)] * 2 + [xa, ya], [I32] * 4, out_idx=(0, 1))
  tx, ty = torch.from_numpy(xa), torch.from_numpy(ya)
  np.testing.assert_array_equal(d, torch.div(tx, ty, rounding_mode="trunc").numpy())
  np.testing.assert_array_equal(m, torch.fmod(tx, ty).numpy())

def test_float16_storage(cpu):
  N = 50
  b = B(); out, osum, x, y = b.buf(0, F16, True), b.buf(1, F32, True), b.buf(2, F16), b.buf(3, F16)
  g = b.special(0, N)
  a, c = b.load(x, g), b.load(y, g)
  b.store(out, g, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, a, c), b.alu(UnaryOps.EXP2, a)))
  b.store(osum, g, b.cast(b.alu(BinaryOps.ADD, a, c), F32))
  xa, ya = np.random.randn(N).astype(np.float16), np.random.randn(N).astype(np.float16)
  (o, s), src = run(cpu, b.uops, [np.zeros(N, np.float16), np.zeros(N, np.float32), xa, ya], [F16, F32, F16, F16], out_idx=(0, 1))
  tx, ty = torch.from_numpy(xa), torch.from_numpy(ya)
  np.testing.assert_allclose(o.astype(np.float32), (tx * ty + torch.exp2(tx)).float().numpy(), rtol=1e-2, atol=1e-2)
  np.testing.assert_array_equal(s, (tx + ty).float().numpy())
  assert "_Float16* restrict data0" in src

def test_bfloat16_storage(cpu):
  N, M = 8, 16
  b = B(); out, osum, x, y = b.buf(0, BF16, True), b.buf(1, F32, True), b.buf(2, BF16), b.buf(3, BF16)
  g = b.special(0, N)
  a, c = b.load(x, g), b.load(y, g)
  b.store(out, g, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, a, c), b.c(0.1, BF16)))
  # f32-accumulated reduction over a bf16 row of y viewed as [N/2, M]... keep it simple: sum_j x[g]*j in f32
  acc = b.u(UOps.DEFIFZ_ACC, F32, (), 0.0)
  r = b.u(UOps.RANGE, I32, (b.c(0), b.c(M)))
  phi = b.u(UOps.PHI, F32, (acc, b.alu(BinaryOps.ADD, acc, b.alu(BinaryOps.MUL, b.cast(a, F32), b.cast(r, F32)))))
  b.u(UOps.ENDRANGE, None, (r,))
  b.store(osum, g, phi)
  tx, ty = torch.randn(N).to(torch.bfloat16), torch.randn(N).to(torch.bfloat16)
  (o, s), src = run(cpu, b.uops, [np.zeros(N, np.uint16), np.zeros(N, np.float32), bf16_bits(tx), bf16_bits(ty)], [BF16, F32, BF16, BF16], out_idx=(0, 1))
  ref = tx * ty + torch.tensor(0.1, dtype=torch.bfloat16)
  ot = from_bf16_bits(o)
  torch.testing.assert_close(ot, ref, rtol=1e-2, atol=1e-2)
  assert (ot == ref).float().mean() >= 0.75  # per-op bf16 rounding mirrors torch closely
  np.testing.assert_allclose(s, (tx.float()[:, None] * torch.arange(M).float()).sum(1).numpy(), rtol=1e-4, atol=1e-5)
  assert "unsigned short* restrict data0" in src

def test_bfloat16_cast_rounding(cpu):
  N = 6
  b = B(); out, back, x = b.buf(0, BF16, True), b.buf(1, F32, True), b.buf(2, F32)
  g = b.special(0, N)
  h = b.cast(b.load(x, g), BF16)
  b.store(out, g, h); b.store(back, g, b.cast(h, F32))
  xa = np.array([1.0, 1.00390625, 1.01171875, -3.14159, 65504.0, 1e-3], np.float32)
  (o, bk), _ = run(cpu, b.uops, [np.zeros(N, np.uint16), np.zeros(N, np.float32), xa], [BF16, F32, F32], out_idx=(0, 1))
  ref = torch.from_numpy(xa).to(torch.bfloat16)
  np.testing.assert_array_equal(o, bf16_bits(ref))
  np.testing.assert_array_equal(bk, ref.float().numpy())

def test_symbolic_define_var(cpu):
  n = make_var("n", 1, 1024)
  b = B(); out, x = b.buf(0, F32, True), b.buf(1, F32)
  vn = b.var(n)
  g = b.special(0, n)
  # out[i] = x[i] + n   (var used both as a loop bound and in arithmetic)
  b.store(out, g, b.alu(BinaryOps.ADD, b.load(x, g), b.cast(vn, F32)))
  src = cpu.renderer.render("sym_kernel", b.uops)
  assert "const int n" in src and "gidx0 < n" in src
  prg = cpu.runtime("sym_kernel", cpu.compiler.compile_cached(src))
  CAP = 100
  for nv in (3, 17, 100):
    xa = np.random.randn(CAP).astype(np.float32)
    ob, xb = Buffer("CPU", CAP, F32), Buffer("CPU", CAP, F32)
    ob.copyin(memoryview(np.full(CAP, -9, np.float32)).cast("B")); xb.copyin(memoryview(xa).cast("B"))
    prg(ob.handle, xb.handle, global_size=(nv, 1, 1), local_size=None, vals=(nv,))
    o = ob.numpy()
    np.testing.assert_array_equal(o[:nv], xa[:nv] + np.float32(nv))
    assert (o[nv:] == -9).all()

def test_compile_cached_and_timing(cpu, tmp_path, monkeypatch):
  import fusezap.helpers as helpers
  monkeypatch.setattr(helpers, "CACHE_DIR", tmp_path)
  monkeypatch.delenv("FZ_NOCACHE", raising=False); helpers.getenv.cache_clear()  # the test asserts a cache hit
  b = B(); out = b.buf(0, I32, True); g = b.special(0, 4); b.store(out, g, b.alu(BinaryOps.MUL, g, g))
  src = cpu.renderer.render("sq", b.uops)
  lib1 = ClangCompiler().compile_cached(src)
  assert any(tmp_path.rglob("*"))
  assert ClangCompiler().compile_cached(src) == lib1
  ob = Buffer("CPU", 4, I32)
  t = CPUProgram("sq", lib1)(ob.handle, global_size=(4, 1, 1), local_size=None, wait=True)
  assert isinstance(t, float) and t >= 0
  np.testing.assert_array_equal(ob.numpy(), [0, 1, 4, 9])

def test_compile_error_reports_source(cpu):
  with pytest.raises(RuntimeError, match="clang failed"): ClangCompiler().compile("void broken( {")

def test_lru_allocator_reuses(cpu):
  a = cpu.allocator
  h = a.alloc(4096); a.free(h, 4096)
  assert a.alloc(4096) is h
  a.free(h, 4096); a.free_cache()
  assert not a.cache

def test_lru_allocator_cache_cap():
  """cache_max bounds the free list (CUDA: FZ_ALLOC_CACHE_MB) by giving the least recently used sizes back to the driver,
  and held / cached / counters() track the driver's share (round 8: traced forwards' cached activations took memory vLLM
  needed)."""
  from fusezap.device.npy import NPYAllocator
  a = NPYAllocator(); a.cache_max = 6000
  hs = [a.alloc(4096), a.alloc(4096), a.alloc(1024)]
  assert a.held == 9216 and a.cached == 0 and a.counters()["allocs"] == 3
  for h, n in zip(hs, (4096, 4096, 1024)): a.free(h, n)
  assert a.cached == 5120 and a.held == 5120 and a.counters()["evicted_bytes"] == 4096   # 8192 > 6000: one 4096 block out
  assert a.alloc(1024) is hs[2] and a.cached == 4096
  a.free_cache()
  assert a.cached == 0 and a.held == 1024 and not a.cache and a.counters()["frees"] == 2

def test_lru_allocator_cap_keeps_the_hot_block():
  """Round 10's regression: the first cap policy refused the freed block when cold blocks (the echo-logprob scorer's
  traces) had filled the list, so a block freed and re-requested every serving step went to the driver every step
  (a cuMemFree + cuMemAlloc each, device synchronizations: -5% at c32 in Invarix). Now the cold size is evicted once and
  the hot loop does no driver call; inside a CUDA graph capture (evict=False) nothing is freed until the next free."""
  from fusezap.device.npy import NPYAllocator
  a = NPYAllocator(); a.cache_max = 6000
  cold = [a.alloc(2500), a.alloc(2500)]
  for h in cold: a.free(h, 2500)
  assert a.cached == 5000
  h = a.alloc(2048); a.free(h, 2048)                 # 7048 > 6000: the cold size goes, the hot block stays
  c0 = a.counters()
  assert a.cached == 4548 and 2048 in a.cache and c0["evicted_bytes"] == 2500
  for _ in range(100):
    h2 = a.alloc(2048); assert h2 is h; a.free(h2, 2048)
  assert a.counters()["allocs"] == c0["allocs"] and a.counters()["frees"] == c0["frees"]
  big = a.alloc(4000); a.free(big, 4000, evict=False)  # a capture: over the cap, nothing freed yet
  assert a.cached == 8548 and a.counters()["frees"] == c0["frees"]
  h3 = a.alloc(2048); a.free(h3, 2048)                 # the next free evicts least recently used first: the 2500 block, then 4000
  assert a.cached <= 6000 and 2048 in a.cache and a.cache[2048][-1] is h3

def test_gpu_style_subclass_surface():
  """A Metal-like subclass only overrides the table; the shared driver emits SPECIAL via render_special."""
  class FakeGPU(CStyleRenderer):
    device, has_local = "FAKE", True
    kernel_prefix, buffer_prefix, arg_int_prefix = "kernel void", "device ", "constant int&"
    def render_special(self, u): return f"gid.{'xyz'[u.arg[0]]}"
  n = make_var("n", 1, 64)
  b = B(); out, x = b.buf(0, F32, True), b.buf(1, F32); b.var(n)
  g = b.special(0, n)
  b.store(out, g, b.alu(UnaryOps.EXP2, b.load(x, g)))
  src = FakeGPU().render("k", b.uops)
  assert "kernel void k(device float* data0, device float* data1, constant int& n)" in src
  assert "int gidx0 = gid.x;" in src and "exp2(val0)" in src and "for (" not in src

# ---------------------------------------------------------------- review regressions

def test_neg_of_negative_const(cpu):
  # NEG(CONST <0) must not render as C pre-decrement `--c` (e.g. `x - (-2)` lowered as ADD(x, NEG(CONST -2)))
  I64 = dtypes.int64
  cases = [(F32, -1.5), (F32, -0.0), (I32, -3), (I64, -3), (BF16, -2.0), (BF16, -0.0), (F16, -2.0)]
  b = B(); outs = [b.buf(i, dt, True) for i, (dt, _) in enumerate(cases)]
  g = b.special(0, 1)
  for o, (dt, v) in zip(outs, cases): b.store(o, g, b.alu(UnaryOps.NEG, b.c(v, dt)))
  dts = [dt for dt, _ in cases]
  res, src = run(cpu, b.uops, [np.zeros(1, np.uint16 if dt == BF16 else to_np(dt)) for dt in dts], dts, out_idx=range(len(cases)))
  assert "--" not in src
  for r, (dt, v) in zip(res, cases):
    if dt == BF16: np.testing.assert_array_equal(r, bf16_bits(-torch.tensor([v])))
    else:
      ref = (-torch.tensor([v], dtype={F32: torch.float32, I32: torch.int32, I64: torch.int64, F16: torch.float16}[dt])).numpy()
      np.testing.assert_array_equal(r, ref); assert np.signbit(r).tolist() == np.signbit(ref).tolist()

def test_bf16_const_above_f32_range(cpu):
  vals = [1e39, -1e39, 3.4028236e38, 1e300]
  b = B(); out = b.buf(0, BF16, True)
  g = b.special(0, 1)
  acc = b.u(UOps.DEFIFZ_ACC, BF16, (), 1e39)
  for i, v in enumerate(vals): b.store(out, b.c(i), b.c(v, BF16))
  b.store(out, b.c(len(vals)), acc)
  (o,), src = run(cpu, b.uops, [np.zeros(len(vals) + 1, np.uint16)], [BF16])
  np.testing.assert_array_equal(o, bf16_bits(torch.tensor(vals + [1e39], dtype=torch.float64)))
  assert "INFINITY" in src

def test_bf16_nan_bits_preserved(cpu):
  I16 = dtypes.int16
  bits = np.array([0x7f81, 0xff81, 0x7fc0, 0x7f80, 0xff80, 0x3f80, 0x8000, 0x7fff, 0x0001], np.uint16)
  N = len(bits)
  b = B(); ocp, obc, orev, x, xi = b.buf(0, BF16, True), b.buf(1, I16, True), b.buf(2, BF16, True), b.buf(3, BF16), b.buf(4, I16)
  g = b.special(0, N)
  v = b.load(x, g)
  b.store(ocp, g, v)                                    # plain copy
  b.store(obc, g, b.bitcast(v, I16))                    # bf16 -> int16 bits
  b.store(orev, g, b.bitcast(b.load(xi, g), BF16))      # int16 bits -> bf16
  (cp, bc, rev), _ = run(cpu, b.uops, [np.zeros(N, np.uint16), np.zeros(N, np.int16), np.zeros(N, np.uint16), bits, bits.view(np.int16)],
                         [BF16, I16, BF16, BF16, I16], out_idx=(0, 1, 2))
  ref = bf16_bits(from_bf16_bits(bits).clone())
  np.testing.assert_array_equal(ref, bits)
  np.testing.assert_array_equal(cp, bits); np.testing.assert_array_equal(bc.view(np.uint16), bits); np.testing.assert_array_equal(rev, bits)
  # arithmetic NaN results are still NaN (never collapse to inf)
  b = B(); out, x = b.buf(0, BF16, True), b.buf(1, BF16)
  g = b.special(0, 2)
  b.store(out, g, b.alu(BinaryOps.ADD, b.load(x, g), b.c(1.0, BF16)))
  (o,), _ = run(cpu, b.uops, [np.zeros(2, np.uint16), bits[:2]], [BF16, BF16])
  assert torch.isnan(from_bf16_bits(o)).all()
