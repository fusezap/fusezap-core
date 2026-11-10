"""Lowering (SPEC §7.2) and the compiled run path (run_compiled) on CPU. WP4b."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from fusezap import Tensor, dtypes
from fusezap.codegen import lower as L
from fusezap.codegen.cstyle import CRenderer, CStyleRenderer
from fusezap.codegen.uops import UOps
from fusezap.device import Buffer, Device
from fusezap.ir import ConstBuffer, LazyOp, MemBuffer
from fusezap.lazy import LazyBuffer
from fusezap.ops import BufferOps, BinaryOps, ReduceOps, UnaryOps
from fusezap.realize import run_schedule
from fusezap.schedule import create_schedule
from fusezap.shape.symbolic import Var
from fusezap.shape.view import ShapeTracker

RNG = np.random.default_rng(7)
pytestmark = pytest.mark.skipif(not Device.available("CPU"), reason="CPU backend (clang) not available")

@pytest.fixture
def cpu(monkeypatch):
  from fusezap import helpers
  helpers.getenv.cache_clear()
  monkeypatch.delenv("FZ_INTERP", raising=False)
  monkeypatch.setattr(Device, "DEFAULT", "CPU", raising=False)
  yield
  helpers.getenv.cache_clear()

def T(*shape, dt=np.float32) -> Tensor: return Tensor(RNG.uniform(-2, 2, size=shape).astype(dt))
def kernels(*ts: Tensor): return [si for si in create_schedule([t.lazydata for t in ts]) if si.kind == "kernel"]
def ops(spec, op): return [u for u in spec.uops if u.op is op]

class _GPU(CStyleRenderer):  # GPU-style surface: SPECIALs are thread indices, y/z capped like Metal
  has_local = True
  global_max = (2**31 - 1, 65535, 65535)
  def render_special(self, u): return f"{'lid' if u.arg[1][0] == 'l' else 'gid'}.{'xyz'[u.arg[0]]}"

def load(idx, shape, dt=dtypes.float32, s=None): return LazyOp(BufferOps.LOAD, (), MemBuffer(idx, dt, s or ShapeTracker.from_shape(shape)))
def store(val, shape, dt=dtypes.float32): return (LazyOp(BufferOps.STORE, (val,), MemBuffer(0, dt, ShapeTracker.from_shape(shape))),)

# ------------------------------------------------------------------ structure
def test_elementwise_structure():
  ast = store(LazyOp(BinaryOps.ADD, (load(1, (4, 5)), load(2, (4, 5)))), (4, 5))
  spec = L.lower(ast, CRenderer())
  assert spec.n_bufs == 3 and spec.vars == [] and not ops(spec, UOps.RANGE)
  sp = ops(spec, UOps.SPECIAL)
  assert [u.arg for u in sp] == [(1, "gidx1", 4), (0, "gidx0", 5)]  # outermost first, gidx0 innermost
  assert spec.global_size == (5, 4) and spec.name == "E_4_5"
  assert [u.arg for u in ops(spec, UOps.DEFIFZ_GLOBAL)] == [(0, "data0", True), (1, "data1", False), (2, "data2", False)]

def test_high_rank_collapses_to_3_dims():
  sh = (2, 3, 4, 5, 6)
  spec = L.lower(store(LazyOp(UnaryOps.NEG, (load(1, sh),)), sh), CRenderer())
  assert len(ops(spec, UOps.SPECIAL)) == 3 and spec.global_size == (6, 5, 24)

def test_reduce_structure():
  x = load(1, (4, 5))
  ast = store(LazyOp(ReduceOps.SUM, (x,), (1,)), (4, 1))
  spec = L.lower(ast, CRenderer())
  assert len(ops(spec, UOps.SPECIAL)) == 1 and len(ops(spec, UOps.RANGE)) == 1
  assert len(ops(spec, UOps.DEFIFZ_ACC)) == 1 and len(ops(spec, UOps.PHI)) == 1
  order = [u.op for u in spec.uops]
  assert order.index(UOps.DEFIFZ_ACC) < order.index(UOps.RANGE) < order.index(UOps.PHI) < order.index(UOps.ENDRANGE) < order.index(UOps.STORE)
  assert spec.name == "r_4_5"

def test_masked_load_is_guarded():
  s = ShapeTracker.from_shape((4,)).pad(((2, 1),))
  spec = L.lower(store(load(1, (7,), s=s), (7,)), CRenderer())
  (ld,) = ops(spec, UOps.LOAD)
  assert len(ld.src) == 4 and ld.src[2].dtype == dtypes.bool and ld.src[3].op is UOps.CONST and ld.src[3].arg == 0

def test_gpu_local_split_and_global_max():
  spec = L.lower(store(LazyOp(UnaryOps.NEG, (load(1, (3, 512)),)), (3, 512)), _GPU())
  assert spec.local_size == (256, 1, 1) and spec.global_size == (2, 3)
  assert sorted(u.arg[1] for u in ops(spec, UOps.SPECIAL)) == ["gidx0", "gidx1", "lidx0"]
  # an outer dim above the y limit is merged into x
  spec = L.lower(store(LazyOp(UnaryOps.NEG, (load(1, (70000, 2)),)), (70000, 2)), _GPU())
  assert len(ops(spec, UOps.SPECIAL)) == 2 and spec.global_size == (140000 // 250,) and spec.local_size == (250, 1, 1)

def test_symbolic_defines_var():
  n = Var("n", 1, 16)
  s = ShapeTracker.from_shape((16,)).shrink(((0, n),))
  spec = L.lower(store(LazyOp(BinaryOps.MUL, (load(1, (n,), s=s), load(1, (n,), s=s))), (n,)), CRenderer())
  assert spec.vars == [n] and [u.arg for u in ops(spec, UOps.DEFIFZ_VAR)] == [n]
  assert spec.global_size == (n,)

# ------------------------------------------------------------------ full pipeline on CPU
def test_pipeline_matches_torch(cpu, assert_close):
  a, b = T(5, 7), T(7, 3)
  out = ((a @ b).relu() + a.sum(1, keepdim=True)).softmax(1)
  ta, tb = torch.tensor(a.numpy()), torch.tensor(b.numpy())
  assert_close(out, ((ta @ tb).relu() + ta.sum(1, keepdim=True)).softmax(1), reduce=True)

def test_pipeline_views(cpu, assert_close):
  a = T(4, 6)
  out = a.T.flip(0)[1:5].pad(((1, 2), (0, 0))).exp() + a.T[::2].sum(1, keepdim=True)[:1] + a.T[1:5].max(1, keepdim=True).pad(((1, 2), (0, 0)))
  ta = torch.tensor(a.numpy())
  ref = torch.nn.functional.pad(ta.T.flip(0)[1:5], (0, 0, 1, 2)).exp() + ta.T[::2].sum(1, keepdim=True)[:1] + torch.nn.functional.pad(ta.T[1:5].amax(1, keepdim=True), (0, 0, 1, 2))
  assert_close(out, ref, reduce=True)

def test_multi_axis_reduce_and_rank5(cpu, assert_close):
  a = T(2, 3, 4, 5, 6)
  ta = torch.tensor(a.numpy())
  assert_close(a.sum((1, 3)), ta.sum((1, 3)), reduce=True)
  assert_close(a.max((0, 2, 4)), ta.amax((0, 2, 4)), reduce=True)
  assert_close(a.permute(4, 2, 0, 3, 1) * 2, ta.permute(4, 2, 0, 3, 1) * 2)

def test_max_propagates_nan(cpu):
  a = np.array([[1.0, np.nan, 3.0], [np.nan, -1.0, -2.0], [4.0, 5.0, 6.0]], np.float32)
  np.testing.assert_array_equal(Tensor(a).max(1).numpy(), torch.tensor(a).amax(1).numpy())

def test_int_div_by_zero_is_zero(cpu):
  def host(a):
    return LazyBuffer.from_buffer(Buffer("CPU", a.size, dtypes.int32).copyin(memoryview(a).cast("B")), a.shape)
  a, b = host(np.array([7, -7, 5, -7], np.int32)), host(np.array([2, 0, 0, 2], np.int32))
  q, m = a.e(BinaryOps.IDIV, b).contiguous(), a.e(BinaryOps.MOD, b).contiguous()
  run_schedule(create_schedule([q, m]))
  np.testing.assert_array_equal(q.realized.numpy(), [3, 0, 0, -3])  # C truncation; 0 for a zero divisor (interp semantics)
  np.testing.assert_array_equal(m.realized.numpy(), [1, 0, 0, -1])

def test_half_sum_accumulates_f32(cpu):
  a = np.full(4096, 1.0, np.float16)
  assert float(Tensor(a).sum().numpy()) == 4096.0

def test_symbolic_one_compile(cpu):
  a = np.arange(16, dtype=np.float32)
  buf = Buffer("CPU", 16, dtypes.float32).copyin(memoryview(a).cast("B"))
  lb = LazyBuffer.from_buffer(buf, (16,))
  n = Var("n", 1, 16)
  e = lb.shrink(((0, n),)).e(BinaryOps.MUL, lb.shrink(((0, n),)))
  r = e.r(ReduceOps.SUM, (0,))
  before = None
  for v in (3, 9, 16):
    e2, r2 = e.contiguous(), r.contiguous()
    s = create_schedule([e2, r2], var_vals={n: v})
    run_schedule(s)
    np.testing.assert_array_equal(e2.realized.numpy()[:v], a[:v] ** 2)
    np.testing.assert_array_equal(r2.realized.numpy(), [(a[:v] ** 2).sum()])
    if before is None: before = len(L._cache)
  assert len(L._cache) == before  # new var values reuse the compiled programs

def test_debug4_prints_source(cpu, monkeypatch, capsys):
  from fusezap import helpers
  monkeypatch.setenv("FZ_DEBUG", "4")
  helpers.getenv.cache_clear()
  L._cache.clear()
  (T(3, 4) * 3).realize()
  out = capsys.readouterr().out
  assert "void E_3_4(" in out and "for (int gidx1" in out

def test_cache_key_reuses_program(cpu):
  a = T(6, 6)
  (a + 1).realize()
  n = len(L._cache)
  (a + 1).realize()
  assert len(L._cache) == n

# ------------------------------------------------------------------ regressions (review of WP4b)
def test_cache_key_distinguishes_signed_zero_and_nan(cpu):
  x = Tensor(np.array([1.0, 2.0], np.float32))
  assert np.all(np.isposinf((x * 0.0).reciprocal().numpy()))
  assert np.all(np.isneginf((x * -0.0).reciprocal().numpy()))  # must not reuse the 0.0 program
  (x + float("nan")).realize()
  n = len(L._cache)
  for _ in range(3): assert np.all(np.isnan((x + float("nan")).numpy()))
  assert len(L._cache) == n  # nan consts hit the cache

def test_int32_cast_fold_wraps_like_interp(cpu, monkeypatch):
  from fusezap import helpers
  def run(interp: bool):
    monkeypatch.setenv("FZ_INTERP", "1") if interp else monkeypatch.delenv("FZ_INTERP", raising=False)
    helpers.getenv.cache_clear()
    x = Tensor(np.array([0, 1, 2, 3], np.int32))
    c = Tensor.full((4,), 2**32 + 5, dtype=dtypes.int64).cast(dtypes.int32)
    f = Tensor.full((4,), 3e9, dtype=dtypes.float32).cast(dtypes.int32)
    return [(c < x + 10).numpy(), (c // (x + 1)).numpy(), (f + x * 0).numpy()]
  got, ref = run(False), run(True)
  np.testing.assert_array_equal(got[0], [True] * 4)
  np.testing.assert_array_equal(got[1], [5, 2, 1, 1])
  for g, r in zip(got, ref): np.testing.assert_array_equal(g, r)
  ast = store(LazyOp(UnaryOps.CAST, (LazyOp(BufferOps.CONST, (), ConstBuffer(2**32 + 5, dtypes.int64, ShapeTracker.from_shape((4,)))),), dtypes.int32), (4,), dtypes.int32)
  assert [u.arg for u in ops(L.lower(ast, CRenderer()), UOps.CONST) if u.dtype == dtypes.int32 and u.arg not in (0, 1)] == [5]

def test_multi_root_masked_store_compiles():
  s = ShapeTracker.from_shape((3,)).pad(((1, 1),))
  roots = tuple(LazyOp(BufferOps.STORE, (LazyOp(UnaryOps.NEG, (load(2, (5,)),)),), MemBuffer(i, dtypes.float32, s)) for i in (0, 1))
  spec = L.lower(roots, CRenderer())
  st = ops(spec, UOps.STORE)
  assert len(st) == 2 and st[0].src[1] is st[1].src[1]
  for u in st: assert spec.uops.index(u.src[1]) < min(i for i, v in enumerate(spec.uops) if v.op is UOps.IF)
  Device["CPU"].compiler.compile_cached(spec.src)  # previously: use of undeclared identifier
