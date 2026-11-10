"""Scheduler (SPEC §6.2/§6.3), run_schedule and the NPY device. WP3."""
from __future__ import annotations
import numpy as np
import pytest
from fusezap import Tensor, dtypes
from fusezap.device import Buffer, Device
from fusezap.ir import LazyOp
from fusezap.lazy import LazyBuffer
from fusezap.ops import BufferOps, ReduceOps, BinaryOps
from fusezap.realize import run_schedule
from fusezap.schedule import create_schedule
from fusezap.shape.symbolic import Var

R = np.random.default_rng(3)

@pytest.fixture
def npy(monkeypatch):
  from fusezap import helpers
  helpers.getenv.cache_clear()
  monkeypatch.delenv("FZ_INTERP", raising=False)
  monkeypatch.setattr(Device, "DEFAULT", "NPY", raising=False)
  yield
  helpers.getenv.cache_clear()

def T(*shape, dt=np.float32) -> Tensor: return Tensor(R.uniform(-2, 2, size=shape).astype(dt)).realize()

def sched(*ts: Tensor):
  s = create_schedule([t.lazydata for t in ts])
  return s, [si for si in s if si.kind == "kernel"]

def nkernels(*ts: Tensor) -> int:
  s, ks = sched(*ts)
  run_schedule(s)
  return len(ks)

# ------------------------------------------------------------------ kernel counts (fusion rules)
def test_elementwise_chain_one_kernel(npy):
  a, b, c = T(3, 4), T(3, 4), T(3, 4)
  out = ((a + b) * c - a).relu().sqrt() + 1.0
  assert nkernels(out) == 1
  an, bn, cn = a.numpy(), b.numpy(), c.numpy()
  np.testing.assert_allclose(out.numpy(), np.sqrt(np.maximum((an + bn) * cn - an, 0)) + 1, rtol=1e-6)

def test_elementwise_through_views_one_kernel(npy):
  a, b = T(3, 4), T(4, 3)
  out = (a + 1).T * b.flip(0) + a.T[1:3].sum() * 0  # a.T[1:3].sum() is a separate reduce kernel
  s, ks = sched(out)
  assert len(ks) == 2
  run_schedule(s)
  e = (a.numpy() + 1).T * b.numpy()[::-1]
  np.testing.assert_allclose(out.numpy(), e, rtol=1e-6)

def test_reduce_with_prologue_and_epilogue_one_kernel(npy):
  x, y = T(5, 6), T(5)
  out = ((x * 2 + 1).exp().sum(1) * 3).sqrt() + y
  assert nkernels(out) == 1
  np.testing.assert_allclose(out.numpy(), np.sqrt(np.exp(x.numpy() * 2 + 1).sum(1) * 3) + y.numpy(), rtol=1e-5)

def test_reduce_keepdim_epilogue_one_kernel(npy):
  x = T(4, 6)
  out = x.sum(1, keepdim=True).reshape(2, 2) * 2
  assert nkernels(out) == 1

def test_reduce_through_expand_is_materialized(npy):
  x = T(4, 6)
  out = x - x.max(1, keepdim=True)  # the max is read through an expand
  assert nkernels(out) == 2
  np.testing.assert_allclose(out.numpy(), x.numpy() - x.numpy().max(1, keepdims=True), rtol=1e-6)

def test_softmax_three_kernels(npy):
  x = T(4, 6)
  out = x.softmax(1)
  assert nkernels(out) == 3   # max | sum(exp(x - max)) | exp(x - max) / sum

def test_multi_consumer_reduce_materialized(npy):
  x = T(4, 6)
  r = x.sum(1)
  o1, o2 = r + 1, r * 2
  assert nkernels(o1, o2) == 3
  np.testing.assert_allclose(o1.numpy(), x.numpy().sum(1) + 1, rtol=1e-5)
  np.testing.assert_allclose(o2.numpy(), x.numpy().sum(1) * 2, rtol=1e-5)

def test_reduce_used_twice_in_one_kernel_fuses(npy):
  x = T(4, 6)
  r = x.sum(1)
  out = (r + 1) * (r * 2)
  s, ks = sched(out)
  assert len(ks) == 1
  assert sum(isinstance(n.op, ReduceOps) for n in ks[0].ast[0].walk()) == 1
  run_schedule(s)
  rn = x.numpy().sum(1)
  np.testing.assert_allclose(out.numpy(), (rn + 1) * (rn * 2), rtol=1e-5)

def test_epilogue_with_two_consumer_kernels_materialized(npy):
  x = T(4, 6)
  e = x.sum(1) + 1          # contains a reduce, consumed by two kernels -> e itself becomes a kernel output
  o1, o2 = e * 2, e * 3
  s, ks = sched(o1, o2)
  assert len(ks) == 3
  assert [n.op for n in ks[0].ast[0].walk() if not isinstance(n.op, BufferOps)] == [ReduceOps.SUM, BinaryOps.ADD]
  run_schedule(s)
  np.testing.assert_allclose(o2.numpy(), (x.numpy().sum(1) + 1) * 3, rtol=1e-5)

def test_one_reduce_per_kernel(npy):
  x, y = T(4, 6), T(4, 6)
  assert nkernels(x.sum(1) + y.sum(1)) == 2       # siblings
  assert nkernels(x.sum(1).sum(0)) == 2           # nested
  z = x.max(1) + (x * x).sum(1)
  s, ks = sched(z)
  assert len(ks) == 2 and all(sum(isinstance(n.op, ReduceOps) for n in k.ast[0].walk()) == 1 for k in ks)
  run_schedule(s)
  np.testing.assert_allclose(z.numpy(), x.numpy().max(1) + (x.numpy() ** 2).sum(1), rtol=1e-5)

def test_contiguous_is_materialized(npy):
  x = T(3, 4)
  y = (x * 2).contiguous()
  out = y.T + 1
  assert nkernels(out) == 2

def test_pad_safety(npy):
  x = T(3, 4)
  safe = (x * 2).pad(((1, 0), (0, 2))) + 0               # 0*2 == 0: fuses through the pad
  assert nkernels(safe) == 1
  np.testing.assert_allclose(safe.numpy(), np.pad(x.numpy() * 2, ((1, 0), (0, 2))), rtol=1e-6)
  unsafe = x.exp().pad(((1, 0), (0, 2))) + 0             # exp(0) == 1: must be materialized before the pad
  assert nkernels(unsafe) == 2
  np.testing.assert_allclose(unsafe.numpy(), np.pad(np.exp(x.numpy()), ((1, 0), (0, 2))), rtol=1e-6)
  s = (x + 1).pad(((0, 0), (1, 1))).sum(1)             # pad inside a reduce prologue
  np.testing.assert_allclose(s.numpy(), (x.numpy() + 1).sum(1), rtol=1e-5)
  c = Tensor.ones(2, 3).pad(((0, 1), (0, 0))) * 5      # a padded CONST reads 0 outside the mask
  np.testing.assert_allclose(c.numpy(), np.pad(np.ones((2, 3)), ((0, 1), (0, 0))) * 5)

# ------------------------------------------------------------------ AST conventions (§6.3)
def test_ast_conventions(npy):
  x, y = T(3, 4), T(3)
  out = (x * 2).sum(1) + y
  s, ks = sched(out)
  assert len(ks) == 1
  si = ks[0]
  assert len(si.ast) == 1 and si.ast[0].op is BufferOps.STORE and si.n_outputs == 1
  store = si.ast[0]
  assert store.arg.idx == 0 and store.arg.st.shape == (3, 1) and store.arg.st.contiguous
  assert out.lazydata.base.realized is si.bufs[0]
  leaves = [n for n in store.walk() if n.op in (BufferOps.LOAD, BufferOps.CONST)]
  assert leaves and all(n.arg.st.shape == (3, 4) for n in leaves)
  assert {n.arg.idx for n in leaves if n.op is BufferOps.LOAD} == {1, 2} and len(si.bufs) == 3
  yl = next(n for n in leaves if n.op is BufferOps.LOAD and si.bufs[n.arg.idx] is y.lazydata.base.realized)
  assert yl.arg.st.views[-1].strides[1] == 0   # epilogue leaf: stride 0 on the reduced axis
  assert not si.bufs[0].allocated              # allocation is deferred to run_schedule
  run_schedule(s)
  np.testing.assert_allclose(out.numpy(), (x.numpy() * 2).sum(1) + y.numpy(), rtol=1e-5)

def test_copy_items(npy):
  a = np.arange(6, dtype=np.float32)
  t = Tensor(a)
  s = create_schedule([t.lazydata])
  assert [si.kind for si in s] == ["copy"] and s[0].ast is None
  run_schedule(s)
  np.testing.assert_array_equal(t.numpy(), a)
  if Device.available("CPU"):
    u = (Tensor(a) * 2).to("CPU")
    np.testing.assert_array_equal(u.numpy(), a * 2)

def test_topological_order_and_bufs_exist(npy):
  x = T(4, 4)
  outs = [(x @ x).sum(0), x.softmax(0).max(1)]
  s = create_schedule([o.lazydata for o in outs])
  produced: set[int] = set()
  for si in s:
    for b in si.bufs[si.n_outputs:]:
      assert b.allocated or id(b) in produced or si.kind == "copy", "input not produced by an earlier item"
    produced.update(id(b) for b in si.bufs[:si.n_outputs])
  run_schedule(s)

# ------------------------------------------------------------------ symbolic
def test_symbolic_shapes(npy):
  a = np.arange(8, dtype=np.float32)
  buf = Buffer("NPY", 8, dtypes.float32).copyin(memoryview(a).cast("B"))
  lb = LazyBuffer.from_buffer(buf, (8,))
  n = Var("n", 1, 8)
  e = lb.shrink(((0, n),)).e(BinaryOps.MUL, lb.shrink(((0, n),)))
  r = e.r(ReduceOps.SUM, (0,))
  for v in (3, 5):
    e2, r2 = e.contiguous(), r.contiguous()
    s = create_schedule([e2, r2], var_vals={n: v})
    assert all(si.var_vals == {n: v} for si in s)
    run_schedule(s)
    np.testing.assert_array_equal(e2.realized.numpy(), a[:v] ** 2)
    np.testing.assert_array_equal(r2.realized.numpy(), [(a[:v] ** 2).sum()])

# ------------------------------------------------------------------ debug output
def test_debug_printing(npy, monkeypatch, capsys):
  from fusezap import helpers
  monkeypatch.setenv("FZ_DEBUG", "3")
  helpers.getenv.cache_clear()
  x = T(3, 4)
  (x.sum(1) + 1).realize()
  out = capsys.readouterr().out
  assert "schedule: 1 items (1 kernels)" in out and "STORE" in out and "SUM" in out and "***" in out and " ms" in out

# ------------------------------------------------------------------ NPY == interp, bit for bit
CASES = {
  "chain": lambda a, b: ((a + b) * a - b.exp()).relu(),
  "reduce_epi": lambda a, b: a.sum(1).sqrt() * b.max(1),
  "softmax": lambda a, b: a.softmax(1) + b.log_softmax(0),
  "matmul": lambda a, b: a @ b.T,
  "layernorm": lambda a, b: a.layernorm() * 2 + b,
  "views": lambda a, b: a.T.flip(0)[1:4].pad(((1, 1), (0, 0))).exp() + b.T.flip(0)[1:4].pad(((1, 1), (0, 0))),
  "cat_cumsum": lambda a, b: a.cat(b, axis=1).cumsum(1),
  "minmax": lambda a, b: a.min(0) - b.argmax(1).cast(dtypes.float32).sum() + a.mean() * b.std(),
  "int": lambda a, b: ((a * 10).cast(dtypes.int32) // 3 + (b * 10).cast(dtypes.int32) % 4).sum(0),
  "half": lambda a, b: (a.half() * b.half()).sum(1) + a.half().exp().max(1),
  "where": lambda a, b: (a < b).where(a, b * 2) + (a == a).cast(dtypes.float32),
}

@pytest.mark.parametrize("name", list(CASES))
def test_npy_matches_interp(name, monkeypatch):
  from fusezap import helpers
  an, bn = R.uniform(0.1, 2, size=(5, 6)).astype(np.float32), R.uniform(-2, 2, size=(5, 6)).astype(np.float32)
  outs = {}
  for tier in ("interp", "NPY"):
    helpers.getenv.cache_clear()
    if tier == "interp": monkeypatch.setenv("FZ_INTERP", "1")
    else: monkeypatch.delenv("FZ_INTERP", raising=False)
    monkeypatch.setattr(Device, "DEFAULT", "NPY", raising=False)
    outs[tier] = CASES[name](Tensor(an), Tensor(bn)).numpy()
  helpers.getenv.cache_clear()
  np.testing.assert_array_equal(outs["NPY"], outs["interp"])

# ------------------------------------------------------------------ review fix: offset-0 shrink of a reduce
def test_reduce_leading_shrink(npy):
  xn = np.arange(24, dtype=np.float32).reshape(4, 6)
  x = Tensor(xn)
  np.testing.assert_array_equal((x.sum(1)[:2] * 1).numpy(), xn.sum(1)[:2])
  np.testing.assert_array_equal((x.max(0)[:3] * 1).numpy(), xn.max(0)[:3])
  np.testing.assert_array_equal((x.sum(1, keepdim=True)[:2] + 1).numpy(), xn.sum(1, keepdims=True)[:2] + 1)
  # pad -> max -> shrink (fuzz seed 643 shape)
  pn = np.pad(xn, ((0, 1), (0, 0)))
  np.testing.assert_array_equal((x.pad(((0, 1), (0, 0))).max(1, keepdim=True)[:4] * 2).numpy(), pn.max(1, keepdims=True)[:4] * 2)
  # a same-size reshape of a reduce still fuses into its consumer
  assert nkernels(x.sum(1).reshape(2, 2) + 1) == 1

def test_elementwise_leading_shrink(npy):
  xn = R.uniform(-1, 1, size=(4, 5, 3)).astype(np.float32)
  x = Tensor(xn)
  np.testing.assert_allclose((x.exp()[:2] * 1).numpy(), np.exp(xn)[:2], rtol=1e-6)
  np.testing.assert_allclose((x.T.exp()[:1] + 1).numpy(), np.exp(xn.T)[:1] + 1, rtol=1e-6)
  np.testing.assert_array_equal(((x * 2)[:1] * 1).numpy(), (xn * 2)[:1])
  assert nkernels((x * 2)[:3] + 1) == 1
