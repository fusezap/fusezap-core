"""WP5: MetalRenderer + METALDevice on hand-built UOp programs (no lowering involved), compared with torch/numpy."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from fusezap.dtype import DType, dtypes, to_np
from fusezap.ops import UnaryOps, BinaryOps, TernaryOps
from fusezap.codegen.uops import UOp, UOps, PtrDType
from fusezap.codegen.metal import MetalRenderer
from fusezap.device import Device, Buffer

F32, I32, F16, BF16, BOOL, I64, I16, I8, U8 = (dtypes.float32, dtypes.int32, dtypes.float16, dtypes.bfloat16, dtypes.bool,
                                              dtypes.int64, dtypes.int16, dtypes.int8, dtypes.uint8)

def make_var(name: str, lo: int, hi: int):
  try:
    from fusezap.shape.symbolic import Var
    return Var(name, lo, hi)
  except NotImplementedError:
    class FakeVar:
      def __init__(self, name, vmin, vmax): self.name, self.vmin, self.vmax = name, vmin, vmax
      def __repr__(self): return f"Var({self.name})"
    return FakeVar(name, lo, hi)

class B:
  """Tiny UOp program builder (same shape as tests/test_cstyle_cpu.py)."""
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

@pytest.fixture(scope="module")
def mtl():
  if not Device.available("METAL"): pytest.skip("METAL not available")
  return Device["METAL"]

def mkbuf(a: np.ndarray, dt: DType) -> Buffer:
  b = Buffer("METAL", a.size, dt)
  return b.copyin(memoryview(np.ascontiguousarray(a.astype(to_np(dt)) if a.dtype != to_np(dt) else a)).cast("B"))

def run(mtl, uops, arrays: list[np.ndarray], dtypes_: list[DType], global_size=(1, 1, 1), local_size=None, out_idx=(0,), vals=(), name="test_kernel"):
  src = mtl.renderer.render(name, uops)
  prg = mtl.runtime(name, mtl.compiler.compile_cached(src))
  bufs = [mkbuf(a, dt) for a, dt in zip(arrays, dtypes_)]
  prg(*[b.handle for b in bufs], global_size=global_size, local_size=local_size, vals=vals)
  return [bufs[i].numpy() for i in out_idx], src

def bf16_bits(t: torch.Tensor) -> np.ndarray: return t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
def from_bf16_bits(a: np.ndarray) -> torch.Tensor: return torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16)

# ---------------------------------------------------------------- device / renderer surface

def test_device_registry(mtl):
  from fusezap.device.metal import METALDevice
  assert isinstance(mtl, METALDevice) and isinstance(mtl.renderer, MetalRenderer)
  assert mtl.renderer.has_local and mtl.renderer.device == "METAL"

def test_unavailable_raises(monkeypatch):
  import builtins
  from fusezap.device.metal import METALDevice
  real = builtins.__import__
  def fake(name, *a, **k):
    if name == "Metal": raise ImportError("no Metal")
    return real(name, *a, **k)
  monkeypatch.setattr(builtins, "__import__", fake)
  with pytest.raises(RuntimeError, match="pyobjc|only available on macOS"): METALDevice("METAL")   # macOS / other hosts

def test_kernel_signature():
  n = make_var("n", 1, 64)
  b = B(); out, x = b.buf(0, F32, True), b.buf(1, F32); b.var(n)
  g = b.special(0, n)
  b.store(out, g, b.alu(UnaryOps.EXP2, b.load(x, g)))
  src = MetalRenderer().render("k", b.uops)
  assert ("kernel void k(device float* data0 [[buffer(0)]], device float* data1 [[buffer(1)]], constant int& n [[buffer(2)]], "
          "uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]])") in src
  assert "#include <metal_stdlib>" in src and "int gidx0 = (int)gid.x;" in src and "for (" not in src

# ---------------------------------------------------------------- numerics

def test_elementwise_add_mul_where(mtl):
  N = 37
  b = B(); out, x, y = b.buf(0, F32, True), b.buf(1, F32), b.buf(2, F32)
  g = b.special(0, N)
  a, c = b.load(x, g), b.load(y, g)
  b.store(out, g, b.alu(TernaryOps.WHERE, b.alu(BinaryOps.CMPLT, a, c), b.alu(BinaryOps.ADD, a, c), b.alu(BinaryOps.MUL, a, c)))
  xa, ya = np.random.randn(N).astype(np.float32), np.random.randn(N).astype(np.float32)
  (o,), _ = run(mtl, b.uops, [np.zeros(N, np.float32), xa, ya], [F32] * 3, global_size=(N, 1, 1))
  tx, ty = torch.from_numpy(xa), torch.from_numpy(ya)
  np.testing.assert_allclose(o, torch.where(tx < ty, tx + ty, tx * ty).numpy(), rtol=1e-5, atol=1e-6)

def test_three_special_dims(mtl):
  A, R, C = 3, 5, 7
  b = B(); out, x = b.buf(0, F32, True), b.buf(1, F32)
  g0, g1, g2 = b.special(0, A), b.special(1, R), b.special(2, C)
  # out[g2, g1, g0] = x[g0, g1, g2]
  xi = b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, g0, b.c(R)), g1), b.c(C)), g2)
  oi = b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, g2, b.c(R)), g1), b.c(A)), g0)
  b.store(out, oi, b.load(x, xi))
  xa = np.random.randn(A, R, C).astype(np.float32)
  (o,), src = run(mtl, b.uops, [np.zeros(xa.size, np.float32), xa.ravel()], [F32] * 2, global_size=(A, R, C))
  np.testing.assert_array_equal(o.reshape(C, R, A), xa.transpose(2, 1, 0))
  assert "gid.y" in src and "gid.z" in src

def test_local_dims(mtl):
  # 2-D launch: groups x threads; out[g*L + l] = x[g*L + l] * 2 + l
  G, L = 6, 32
  b = B(); out, x = b.buf(0, F32, True), b.buf(1, F32)
  g, l = b.special(0, G), b.special(0, L, local=True)
  i = b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, g, b.c(L)), l)
  b.store(out, i, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, b.load(x, i), b.c(2.0, F32)), b.cast(l, F32)))
  xa = np.random.randn(G * L).astype(np.float32)
  (o,), src = run(mtl, b.uops, [np.zeros(G * L, np.float32), xa], [F32] * 2, global_size=(G, 1, 1), local_size=(L, 1, 1))
  np.testing.assert_array_equal(o, xa * 2 + np.tile(np.arange(L, dtype=np.float32), G))
  assert "int lidx0 = (int)lid.x;" in src

@pytest.mark.parametrize("kind", ["sum", "max"])
def test_reduction_acc_range_phi(mtl, kind):
  N, M = 6, 333
  b = B(); out, x = b.buf(0, F32, True), b.buf(1, F32)
  g = b.special(0, N)
  acc = b.u(UOps.DEFIFZ_ACC, F32, (), 0.0 if kind == "sum" else float("-inf"))
  r = b.u(UOps.RANGE, I32, (b.c(0), b.c(M)))
  v = b.load(x, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, g, b.c(M)), r))
  phi = b.u(UOps.PHI, F32, (acc, b.alu(BinaryOps.ADD if kind == "sum" else BinaryOps.MAX, acc, v)))
  b.u(UOps.ENDRANGE, None, (r,))
  b.store(out, g, phi)
  xa = np.random.randn(N, M).astype(np.float32)
  (o,), _ = run(mtl, b.uops, [np.zeros(N, np.float32), xa.ravel()], [F32] * 2, global_size=(N, 1, 1))
  t = torch.from_numpy(xa)
  np.testing.assert_allclose(o, (t.sum(1) if kind == "sum" else t.amax(1)).numpy(), rtol=1e-4, atol=1e-5)

def test_guarded_load_and_if(mtl):
  N = 20
  b = B(); out, o2, x = b.buf(0, F32, True), b.buf(1, F32, True), b.buf(2, F32)
  g = b.special(0, N)
  b.store(out, g, b.load(x, b.alu(BinaryOps.ADD, g, b.c(3)), b.alu(BinaryOps.CMPLT, g, b.c(N - 3)), b.c(-1.5, F32)))
  i = b.u(UOps.IF, None, (b.alu(BinaryOps.CMPEQ, b.alu(BinaryOps.MOD, g, b.c(2)), b.c(0)),))
  b.store(o2, g, b.load(x, b.alu(BinaryOps.IDIV, g, b.c(2))))
  b.u(UOps.ENDIF, None, (i,))
  xa = np.random.randn(N).astype(np.float32)
  (o, p), _ = run(mtl, b.uops, [np.zeros(N, np.float32), np.zeros(N, np.float32), xa], [F32] * 3, global_size=(N, 1, 1), out_idx=(0, 1))
  np.testing.assert_array_equal(o, np.concatenate([xa[3:], np.full(3, -1.5, np.float32)]))
  np.testing.assert_array_equal(p, np.where(np.arange(N) % 2 == 0, xa[np.arange(N) // 2], 0))

def test_unary_math(mtl):
  N = 256
  ops = [UnaryOps.NEG, UnaryOps.EXP2, UnaryOps.LOG2, UnaryOps.SIN, UnaryOps.SQRT, UnaryOps.RECIP]
  b = B(); outs = [b.buf(i, F32, True) for i in range(len(ops))]; x = b.buf(len(ops), F32)
  g = b.special(0, N)
  v = b.load(x, g)
  for o, op in zip(outs, ops): b.store(o, g, b.alu(op, v))
  xa = np.random.uniform(0.1, 20.0, N).astype(np.float32)
  res, _ = run(mtl, b.uops, [np.zeros(N, np.float32)] * len(ops) + [xa], [F32] * (len(ops) + 1), global_size=(N, 1, 1), out_idx=range(len(ops)))
  t = torch.from_numpy(xa)
  for o, ref in zip(res, [-t, torch.exp2(t), torch.log2(t), torch.sin(t), torch.sqrt(t), 1 / t]):
    np.testing.assert_allclose(o, ref.numpy(), rtol=1e-5, atol=1e-6)

def test_special_values(mtl):
  # fast math must be off: inf/nan propagate, log2(0) = -inf, sqrt(-1) = nan, 1/0 = inf
  N = 5
  b = B(); o_add, o_log, o_sqrt, o_rcp, o_cmp, x = b.buf(0, F32, True), b.buf(1, F32, True), b.buf(2, F32, True), b.buf(3, F32, True), b.buf(4, BOOL, True), b.buf(5, F32)
  g = b.special(0, N)
  v = b.load(x, g)
  b.store(o_add, g, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MAX, v, b.c(float("-inf"), F32)), b.c(1e-7, F32)))
  b.store(o_log, g, b.alu(UnaryOps.LOG2, v)); b.store(o_sqrt, g, b.alu(UnaryOps.SQRT, v)); b.store(o_rcp, g, b.alu(UnaryOps.RECIP, v))
  b.store(o_cmp, g, b.alu(BinaryOps.CMPEQ, v, v))
  xa = np.array([np.inf, -np.inf, np.nan, 0.0, -1.0], np.float32)
  res, _ = run(mtl, b.uops, [np.zeros(N, np.float32)] * 4 + [np.zeros(N, bool), xa], [F32] * 4 + [BOOL, F32], global_size=(N, 1, 1), out_idx=range(5))
  a, lg, sq, rc, eq = res
  assert a[0] == np.inf and a[1] == -np.inf and a[2] == -np.inf and a[3] == np.float32(1e-7)
  t = torch.from_numpy(xa)
  np.testing.assert_array_equal(lg, torch.log2(t).numpy()); np.testing.assert_array_equal(sq, torch.sqrt(t).numpy())
  np.testing.assert_array_equal(rc, (1 / t).numpy()); np.testing.assert_array_equal(eq, (t == t).numpy())

def test_casts(mtl):
  N = 9
  b = B()
  o_i, o_f, o_b, o_bf, o_h, o_bc, o_i8, o_u8, x, xi = (b.buf(0, I32, True), b.buf(1, F32, True), b.buf(2, BOOL, True), b.buf(3, F32, True),
                                                       b.buf(4, F16, True), b.buf(5, I32, True), b.buf(6, I8, True), b.buf(7, U8, True), b.buf(8, F32), b.buf(9, I32))
  g = b.special(0, N)
  v, vi = b.load(x, g), b.load(xi, g)
  b.store(o_i, g, b.cast(v, I32)); b.store(o_f, g, b.cast(vi, F32))
  bl = b.cast(v, BOOL)
  b.store(o_b, g, bl); b.store(o_bf, g, b.cast(bl, F32)); b.store(o_h, g, b.cast(v, F16)); b.store(o_bc, g, b.bitcast(v, I32))
  b.store(o_i8, g, b.cast(b.alu(BinaryOps.MOD, vi, b.c(100)), I8)); b.store(o_u8, g, b.cast(b.alu(BinaryOps.MOD, vi, b.c(100)), U8))
  xa = np.array([-2.7, -1.5, -0.5, 0.0, 0.5, 1.5, 2.7, 1e3, -0.0], np.float32)
  xia = np.array([-7, -1, 0, 1, 2, 3, 1 << 20, -(1 << 24) - 1, 123456789], np.int32)
  outs = (I32, F32, BOOL, F32, F16, I32, I8, U8)
  res, _ = run(mtl, b.uops, [np.zeros(N, to_np(d)) for d in outs] + [xa, xia], [*outs, F32, I32], global_size=(N, 1, 1), out_idx=range(8))
  t, ti = torch.from_numpy(xa), torch.from_numpy(xia)
  np.testing.assert_array_equal(res[0], t.to(torch.int32).numpy())
  np.testing.assert_array_equal(res[1], ti.to(torch.float32).numpy())
  np.testing.assert_array_equal(res[2], t.to(torch.bool).numpy())
  np.testing.assert_array_equal(res[3], t.to(torch.bool).to(torch.float32).numpy())
  np.testing.assert_array_equal(res[4], t.to(torch.float16).numpy())
  np.testing.assert_array_equal(res[5], xa.view(np.int32))
  np.testing.assert_array_equal(res[6], torch.fmod(ti, 100).to(torch.int8).numpy())
  np.testing.assert_array_equal(res[7], torch.fmod(ti, 100).to(torch.uint8).numpy())

def test_int_idiv_mod_int64(mtl):
  N = 12
  b = B(); od, om, o64, x, y = b.buf(0, I32, True), b.buf(1, I32, True), b.buf(2, I64, True), b.buf(3, I32), b.buf(4, I32)
  g = b.special(0, N)
  a, c = b.load(x, g), b.load(y, g)
  b.store(od, g, b.alu(BinaryOps.IDIV, a, c)); b.store(om, g, b.alu(BinaryOps.MOD, a, c))
  b.store(o64, g, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, b.cast(a, I64), b.c(1 << 33, I64)), b.c(-3, I64)))
  xa = np.array([7, -7, 7, -7, 0, 1, -1, 100, -100, 13, 2147483647, -2147483647], np.int32)
  ya = np.array([2, 2, -2, -2, 3, 5, 5, 7, 7, -4, 10, 10], np.int32)
  (d, m, w), src = run(mtl, b.uops, [np.zeros(N, np.int32)] * 2 + [np.zeros(N, np.int64), xa, ya], [I32, I32, I64, I32, I32], global_size=(N, 1, 1), out_idx=(0, 1, 2))
  tx, ty = torch.from_numpy(xa), torch.from_numpy(ya)
  np.testing.assert_array_equal(d, torch.div(tx, ty, rounding_mode="trunc").numpy())
  np.testing.assert_array_equal(m, torch.fmod(tx, ty).numpy())
  np.testing.assert_array_equal(w, (tx.long() * (1 << 33) - 3).numpy())
  assert "ll" not in src.replace("lid", "")

def test_float16(mtl):
  N = 50
  b = B(); out, osum, x, y = b.buf(0, F16, True), b.buf(1, F32, True), b.buf(2, F16), b.buf(3, F16)
  g = b.special(0, N)
  a, c = b.load(x, g), b.load(y, g)
  b.store(out, g, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, a, c), b.alu(UnaryOps.EXP2, a)))
  b.store(osum, g, b.cast(b.alu(BinaryOps.ADD, a, c), F32))
  xa, ya = np.random.randn(N).astype(np.float16), np.random.randn(N).astype(np.float16)
  (o, s), src = run(mtl, b.uops, [np.zeros(N, np.float16), np.zeros(N, np.float32), xa, ya], [F16, F32, F16, F16], global_size=(N, 1, 1), out_idx=(0, 1))
  tx, ty = torch.from_numpy(xa), torch.from_numpy(ya)
  np.testing.assert_allclose(o.astype(np.float32), (tx * ty + torch.exp2(tx)).float().numpy(), rtol=1e-2, atol=1e-2)
  np.testing.assert_array_equal(s, (tx + ty).float().numpy())
  assert "device half* data0" in src

def test_bfloat16(mtl):
  N, M = 64, 16
  b = B(); out, osum, oex, x, y = b.buf(0, BF16, True), b.buf(1, F32, True), b.buf(2, BF16, True), b.buf(3, BF16), b.buf(4, BF16)
  g = b.special(0, N)
  a, c = b.load(x, g), b.load(y, g)
  b.store(out, g, b.alu(BinaryOps.ADD, b.alu(BinaryOps.MUL, a, c), b.c(0.1, BF16)))
  b.store(oex, g, b.alu(UnaryOps.EXP2, a))
  acc = b.u(UOps.DEFIFZ_ACC, F32, (), 0.0)
  r = b.u(UOps.RANGE, I32, (b.c(0), b.c(M)))
  phi = b.u(UOps.PHI, F32, (acc, b.alu(BinaryOps.ADD, acc, b.alu(BinaryOps.MUL, b.cast(a, F32), b.cast(r, F32)))))
  b.u(UOps.ENDRANGE, None, (r,))
  b.store(osum, g, phi)
  tx, ty = torch.randn(N).to(torch.bfloat16), torch.randn(N).to(torch.bfloat16)
  (o, s, e), src = run(mtl, b.uops, [np.zeros(N, np.uint16), np.zeros(N, np.float32), np.zeros(N, np.uint16), bf16_bits(tx), bf16_bits(ty)],
                       [BF16, F32, BF16, BF16, BF16], global_size=(N, 1, 1), out_idx=(0, 1, 2))
  ref = tx * ty + torch.tensor(0.1, dtype=torch.bfloat16)
  ot = from_bf16_bits(o)
  torch.testing.assert_close(ot, ref, rtol=1e-2, atol=1e-2)
  assert (ot == ref).float().mean() >= 0.75
  torch.testing.assert_close(from_bf16_bits(e), torch.exp2(tx), rtol=1e-2, atol=1e-2)
  np.testing.assert_allclose(s, (tx.float()[:, None] * torch.arange(M).float()).sum(1).numpy(), rtol=1e-4, atol=1e-5)
  assert "device bfloat* data0" in src

def test_bfloat16_casts(mtl):
  N = 6
  b = B(); out, back, oi, obf, x, xi = b.buf(0, BF16, True), b.buf(1, F32, True), b.buf(2, I32, True), b.buf(3, BF16, True), b.buf(4, F32), b.buf(5, I32)
  g = b.special(0, N)
  h = b.cast(b.load(x, g), BF16)
  b.store(out, g, h); b.store(back, g, b.cast(h, F32)); b.store(oi, g, b.cast(h, I32)); b.store(obf, g, b.cast(b.load(xi, g), BF16))
  xa = np.array([1.0, 1.00390625, 1.01171875, -3.14159, 65504.0, -7.9], np.float32)
  xia = np.array([0, 1, -3, 257, 1 << 20, -12345], np.int32)
  (o, bk, i, fb), _ = run(mtl, b.uops, [np.zeros(N, np.uint16), np.zeros(N, np.float32), np.zeros(N, np.int32), np.zeros(N, np.uint16), xa, xia],
                          [BF16, F32, I32, BF16, F32, I32], global_size=(N, 1, 1), out_idx=(0, 1, 2, 3))
  ref = torch.from_numpy(xa).to(torch.bfloat16)
  np.testing.assert_array_equal(o, bf16_bits(ref)); np.testing.assert_array_equal(bk, ref.float().numpy())
  np.testing.assert_array_equal(i, ref.to(torch.int32).numpy()); np.testing.assert_array_equal(fb, bf16_bits(torch.from_numpy(xia).to(torch.bfloat16)))

def test_neg_of_negative_const(mtl):
  cases = [(F32, -1.5), (F32, -0.0), (I32, -3), (I64, -3), (BF16, -2.0), (BF16, -0.0), (F16, -2.0)]
  b = B(); outs = [b.buf(i, dt, True) for i, (dt, _) in enumerate(cases)]
  g = b.special(0, 1)
  for o, (dt, v) in zip(outs, cases): b.store(o, g, b.alu(UnaryOps.NEG, b.c(v, dt)))
  dts = [dt for dt, _ in cases]
  res, src = run(mtl, b.uops, [np.zeros(1, to_np(dt)) for dt in dts], dts, out_idx=range(len(cases)))
  assert "--" not in src
  for r, (dt, v) in zip(res, cases):
    if dt == BF16: np.testing.assert_array_equal(r, bf16_bits(-torch.tensor([v])))
    else:
      ref = (-torch.tensor([v], dtype={F32: torch.float32, I32: torch.int32, I64: torch.int64, F16: torch.float16}[dt])).numpy()
      np.testing.assert_array_equal(r, ref); assert np.signbit(r).tolist() == np.signbit(ref).tolist()

def test_consts_out_of_f32_range(mtl):
  vals = [1e39, -1e39, 3.4028236e38, 1e300]
  b = B(); out, o32 = b.buf(0, BF16, True), b.buf(1, F32, True)
  b.special(0, 1)
  acc = b.u(UOps.DEFIFZ_ACC, BF16, (), 1e39)
  for i, v in enumerate(vals): b.store(out, b.c(i), b.c(v, BF16)); b.store(o32, b.c(i), b.c(v, F32))
  b.store(out, b.c(len(vals)), acc)
  (o, f), _ = run(mtl, b.uops, [np.zeros(len(vals) + 1, np.uint16), np.zeros(len(vals), np.float32)], [BF16, F32], out_idx=(0, 1))
  np.testing.assert_array_equal(o, bf16_bits(torch.tensor(vals + [1e39], dtype=torch.float64)))
  np.testing.assert_array_equal(f, torch.tensor(vals, dtype=torch.float64).float().numpy())

def test_bf16_nan_bits_preserved(mtl):
  bits = np.array([0x7f81, 0xff81, 0x7fc0, 0x7f80, 0xff80, 0x3f80, 0x8000, 0x7fff, 0x0001], np.uint16)
  N = len(bits)
  b = B(); ocp, obc, orev, x, xi = b.buf(0, BF16, True), b.buf(1, I16, True), b.buf(2, BF16, True), b.buf(3, BF16), b.buf(4, I16)
  g = b.special(0, N)
  v = b.load(x, g)
  b.store(ocp, g, v); b.store(obc, g, b.bitcast(v, I16)); b.store(orev, g, b.bitcast(b.load(xi, g), BF16))
  (cp, bc, rev), _ = run(mtl, b.uops, [np.zeros(N, np.uint16), np.zeros(N, np.int16), np.zeros(N, np.uint16), bits, bits.view(np.int16)],
                         [BF16, I16, BF16, BF16, I16], global_size=(N, 1, 1), out_idx=(0, 1, 2))
  np.testing.assert_array_equal(cp, bits); np.testing.assert_array_equal(bc.view(np.uint16), bits); np.testing.assert_array_equal(rev, bits)

def test_symbolic_var(mtl):
  n = make_var("n", 1, 1024)
  b = B(); out, x = b.buf(0, F32, True), b.buf(1, F32)
  vn = b.var(n)
  g = b.special(0, n)
  b.store(out, g, b.alu(BinaryOps.ADD, b.load(x, g), b.cast(vn, F32)))
  src = mtl.renderer.render("sym_kernel", b.uops)
  prg = mtl.runtime("sym_kernel", mtl.compiler.compile_cached(src))
  CAP = 100
  for nv in (3, 17, 100):
    xa = np.random.randn(CAP).astype(np.float32)
    ob, xb = mkbuf(np.full(CAP, -9, np.float32), F32), mkbuf(xa, F32)
    prg(ob.handle, xb.handle, global_size=(nv, 1, 1), local_size=None, vals=(nv,))
    o = ob.numpy()
    np.testing.assert_array_equal(o[:nv], xa[:nv] + np.float32(nv))
    assert (o[nv:] == -9).all()

# ---------------------------------------------------------------- runtime plumbing

def test_async_launch_timing_synchronize(mtl):
  N = 1 << 16
  b = B(); out, x = b.buf(0, F32, True), b.buf(1, F32)
  g = b.special(0, N)
  b.store(out, g, b.alu(BinaryOps.ADD, b.load(x, g), b.c(1.0, F32)))
  src = mtl.renderer.render("inc", b.uops)
  prg = mtl.runtime("inc", mtl.compiler.compile_cached(src))
  ob, xb = mkbuf(np.zeros(N, np.float32), F32), mkbuf(np.zeros(N, np.float32), F32)
  for _ in range(5):  # ping-pong without waiting: in-order queue
    assert prg(ob.handle, xb.handle, global_size=(N, 1, 1), local_size=None) is None
    assert prg(xb.handle, ob.handle, global_size=(N, 1, 1), local_size=None) is None
  assert mtl.in_flight
  mtl.synchronize()
  assert not mtl.in_flight
  np.testing.assert_array_equal(xb.numpy(), np.full(N, 10, np.float32))
  t = prg(ob.handle, xb.handle, global_size=(N // 64, 1, 1), local_size=(64, 1, 1), wait=True)
  assert isinstance(t, float) and 0 <= t < 10
  prg(ob.handle, xb.handle, global_size=(N, 1, 1), local_size=None)
  np.testing.assert_array_equal(ob.numpy(), np.full(N, 11, np.float32))   # copyout synchronizes

def test_compile_cached_is_source(mtl, tmp_path, monkeypatch):
  import fusezap.helpers as helpers
  monkeypatch.setattr(helpers, "CACHE_DIR", tmp_path)
  b = B(); out = b.buf(0, I32, True); g = b.special(0, 4); b.store(out, g, b.alu(BinaryOps.MUL, g, g))
  src = mtl.renderer.render("sq", b.uops)
  lib = mtl.compiler.compile_cached(src)
  assert lib == src.encode() and any(tmp_path.rglob("*")) and mtl.compiler.compile_cached(src) == lib
  ob = Buffer("METAL", 4, I32)
  mtl.runtime("sq", lib)(ob.handle, global_size=(4, 1, 1), local_size=None, wait=True)
  np.testing.assert_array_equal(ob.numpy(), [0, 1, 4, 9])

def test_compile_error_reports_source(mtl):
  with pytest.raises(RuntimeError, match="Metal compile failed"): mtl.compiler.compile("kernel void broken( {")

def test_too_many_threads(mtl):
  b = B(); out = b.buf(0, I32, True); g = b.special(0, 4); b.store(out, g, g)
  prg = mtl.runtime("tm", mtl.compiler.compile(mtl.renderer.render("tm", b.uops)))
  with pytest.raises(RuntimeError, match="local size"): prg(Buffer("METAL", 4, I32).handle, global_size=(1, 1, 1), local_size=(4096, 1, 1))

def test_lru_allocator_reuses(mtl):
  a = mtl.allocator
  h = a.alloc(4096); a.free(h, 4096)
  assert a.alloc(4096) is h
  assert h.storageMode() == mtl.Metal.MTLStorageModeShared and h.length() >= 4096
  a.free(h, 4096); a.free_cache()
  assert not a.cache

# ---------------------------------------------------------------- review fixes: buffer slot limit, bool bitcast

@pytest.mark.parametrize("nbufs,nvars", [(31, 0), (28, 3), (29, 3), (30, 1), (31, 1), (32, 0), (40, 3), (600, 2)])
def test_many_buffers_and_vars(mtl, nbufs, nvars):
  """Metal has 31 buffer slots: wide kernels must still compile and run (out = sum of inputs + sum of vars)."""
  from fusezap.codegen.metal import metal_layout
  N = 8
  vs = [make_var(f"v{i}", 0, 100) for i in range(nvars)]
  b = B(); out = b.buf(0, F32, True); ins = [b.buf(i, F32) for i in range(1, nbufs)]
  vus = [b.var(v) for v in vs]
  g = b.special(0, N)
  acc = b.load(ins[0], g)
  for x in ins[1:]: acc = b.alu(BinaryOps.ADD, acc, b.load(x, g))
  for vu in vus: acc = b.alu(BinaryOps.ADD, acc, b.cast(vu, F32))
  b.store(out, g, acc)
  arrs = [np.zeros(N, np.float32)] + [np.random.randn(N).astype(np.float32) for _ in ins]
  vals = tuple(range(3, 3 + nvars))
  (o,), src = run(mtl, b.uops, arrs, [F32] * nbufs, global_size=(N, 1, 1), local_size=(1, 1, 1), vals=vals)
  ref = np.zeros(N, np.float32)
  for a in arrs[1:]: ref = ref + a
  for v in vals: ref = ref + np.float32(v)
  np.testing.assert_allclose(o, ref, rtol=1e-5, atol=1e-5)
  slots = [int(x) for x in __import__("re").findall(r"\[\[buffer\((\d+)\)\]\]", src)]
  assert max(slots) <= 30
  assert metal_layout(nbufs, nvars) == ("plain" if nbufs + nvars <= 31 else "packed" if nbufs <= 30 else "argbuf")
  if nvars and nbufs + nvars <= 31: assert "constant int& v0" in src   # SPEC signature kept whenever it fits

@pytest.mark.parametrize("sdt,ddt", [(BOOL, U8), (U8, BOOL), (BOOL, I8), (I8, BOOL)])
def test_bitcast_bool(mtl, sdt, ddt):
  N = 4
  b = B(); out, x = b.buf(0, ddt, True), b.buf(1, sdt)
  g = b.special(0, N)
  b.store(out, g, b.bitcast(b.load(x, g), ddt))
  xa = np.array([1, 0, 1, 1]).astype(to_np(sdt))
  (o,), _ = run(mtl, b.uops, [np.zeros(N, to_np(ddt)), xa], [ddt, sdt], global_size=(N, 1, 1), local_size=(1, 1, 1))
  tref = torch.from_numpy(xa).view(torch.bool if ddt == BOOL else (torch.uint8 if ddt == U8 else torch.int8)).numpy()
  np.testing.assert_array_equal(o, tref)

def _phys_footprint() -> int:
  """This process's phys_footprint (task_info TASK_VM_INFO): what Activity Monitor's Memory column shows."""
  import ctypes, ctypes.util
  class TVI(ctypes.Structure):
    _fields_ = [("v", ctypes.c_uint64), ("rc", ctypes.c_int32), ("ps", ctypes.c_int32)] + [(f"u{i}", ctypes.c_uint64) for i in range(16)] + \
               [("phys_footprint", ctypes.c_uint64), ("pad", ctypes.c_uint64 * 40)]
  libc = ctypes.CDLL(ctypes.util.find_library("c"))
  info, cnt = TVI(), ctypes.c_uint32(ctypes.sizeof(TVI) // 4)
  assert libc.task_info(libc.mach_task_self(), 22, ctypes.byref(info), ctypes.byref(cnt)) == 0
  return info.phys_footprint

def test_freed_buffers_return_memory(mtl):
  """I81: PyObjC keeps newBufferWithLength's +1 on top of its proxy's retain; MetalAllocator._free must release it, or
  every block that leaves the LRU cache leaks (64 MB x 3 rounds here grew the footprint by 768 MB before the fix)."""
  al, n = mtl.allocator, 64 << 20
  src = memoryview(np.ones(n // 4, np.float32)).cast("B")
  def round_trip():
    hs = [al._alloc(n) for _ in range(4)]
    for h in hs: al.copyin(h, src)   # touch every page
    for h in hs: al._release(h, n)
  round_trip()
  base = _phys_footprint()
  for _ in range(3): round_trip()
  assert _phys_footprint() - base < n, f"footprint grew {(_phys_footprint() - base) / 1e6:.0f} MB"

def test_free_waits_for_in_flight_work(mtl):
  """A block freed while a kernel may still read it is released only at the next synchronize()."""
  from fusezap import Tensor
  x = Tensor(np.arange(1 << 16, dtype=np.float32), device="METAL").realize()
  y = (x * 2 + 1).realize()
  pending = len(mtl.pending_release)
  if mtl.in_flight:
    h = mtl.allocator._alloc(1 << 20)
    mtl.allocator._free(h, 1 << 20)
    assert len(mtl.pending_release) == pending + (1 if mtl.allocator.owns_new else 0)
  np.testing.assert_array_equal(y.numpy(), np.arange(1 << 16, dtype=np.float32) * 2 + 1)
  assert not mtl.pending_release and not mtl.in_flight
