"""WP2: LazyBuffer structure and the reference interpreter (FZ_INTERP=1), checked against numpy."""
from __future__ import annotations
import numpy as np
import pytest
from fusezap.dtype import dtypes
from fusezap.ops import LoadOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps
from fusezap.lazy import LazyBuffer
from fusezap.device import Buffer, Device
from fusezap.shape.symbolic import Var
from fusezap import interpret

@pytest.fixture
def dev(monkeypatch):
  from fusezap import helpers
  helpers.getenv.cache_clear()
  monkeypatch.setenv("FZ_INTERP", "1")
  d = "CPU" if Device.available("CPU") else "NPY"
  monkeypatch.setattr(Device, "DEFAULT", d, raising=False)
  yield d
  helpers.getenv.cache_clear()

def host(a: np.ndarray, dev: str) -> LazyBuffer:
  from fusezap.dtype import dtypes as D
  return LazyBuffer.loadop(LoadOps.COPY, a.shape, D.from_np(a.dtype), dev, a.reshape(-1))

def val(lb: LazyBuffer) -> np.ndarray:
  out = lb.contiguous() if lb.base is not lb or lb.op is LoadOps.CONST else lb
  interpret.realize([out])
  return out.realized.numpy().reshape(lb.shape)

def test_base_and_views(dev):
  a = np.arange(12, dtype=np.float32).reshape(3, 4)
  x = host(a, dev)
  assert x.base is x and x.st.contiguous and x.op is LoadOps.COPY and x.realized is None
  v = x.permute((1, 0))
  assert v.base is x and v.shape == (4, 3) and v.op is LoadOps.COPY and v.srcs == x.srcs
  assert v.permute((1, 0)) is x  # identity view returns the base
  assert x.reshape((3, 4)) is x and x.expand((3, 4)) is x and x.pad(((0, 0), (0, 0))) is x
  s = v.shrink(((1, 3), (0, 2))).pad(((1, 0), (0, 1))).flip((True, False))
  e = np.pad(a.T[1:3, 0:2], ((1, 0), (0, 1)))[::-1]
  np.testing.assert_array_equal(val(s), e)
  # movement never adds graph nodes
  assert s.base is x

def test_const_is_expanded_scalar(dev):
  c = LazyBuffer.loadop(LoadOps.CONST, (2, 3), dtypes.float32, dev, 1.5)
  assert c.base.shape == () and c.is_unrealized_const() and c.shape == (2, 3)
  np.testing.assert_array_equal(val(c), np.full((2, 3), 1.5, np.float32))
  assert c.base.realized is None  # consts are never materialized by the interpreter
  i = LazyBuffer.loadop(LoadOps.CONST, (), dtypes.int32, dev, 3.7)
  assert i.arg == 3 and i.shape == ()

def test_elementwise_asserts(dev):
  x = host(np.ones((2, 3), np.float32), dev)
  y = host(np.ones((3,), np.float32), dev)
  with pytest.raises(AssertionError): x.e(BinaryOps.ADD, y)                      # no broadcasting
  with pytest.raises(AssertionError): x.e(BinaryOps.ADD, x.cast(dtypes.float16))  # same dtype
  with pytest.raises(AssertionError): x.e(TernaryOps.WHERE, x, x)                 # bool condition
  with pytest.raises(AssertionError): x.e(BinaryOps.IDIV, x)                      # ints only
  assert x.e(BinaryOps.CMPLT, x).dtype == dtypes.bool
  assert x.e(UnaryOps.CAST, arg=dtypes.int32).dtype == dtypes.int32

def test_reduce_keepdim(dev):
  a = np.random.default_rng(0).standard_normal((2, 3, 4)).astype(np.float32)
  x = host(a, dev)
  r = x.r(ReduceOps.SUM, (0, 2))
  assert r.shape == (1, 3, 1) and r.arg == (0, 2)
  np.testing.assert_allclose(val(r), a.sum((0, 2), keepdims=True), rtol=1e-5)
  np.testing.assert_array_equal(val(x.permute((2, 0, 1)).r(ReduceOps.MAX, (1,))), a.transpose(2, 0, 1).max(1, keepdims=True))

def test_c_semantics_int(dev):
  a, b = np.array([7, -7, 8, -8, 0], np.int32), np.array([2, 2, -3, -3, 4], np.int32)
  x, y = host(a, dev), host(b, dev)
  np.testing.assert_array_equal(val(x.e(BinaryOps.IDIV, y)), np.trunc(a / b).astype(np.int32))
  np.testing.assert_array_equal(val(x.e(BinaryOps.MOD, y)), np.fmod(a, b))

def test_max_semantics(dev):
  a = np.array([1.0, np.nan, 2.0], np.float32); b = np.array([np.nan, 1.0, 2.0], np.float32)
  out = val(host(a, dev).e(BinaryOps.MAX, host(b, dev)))   # a > b ? a : b
  np.testing.assert_array_equal(np.isnan(out), [True, False, False])
  assert out[1] == 1.0 and out[2] == 2.0

def test_casts(dev):
  a = np.array([-2.7, -0.5, 0.0, 0.5, 2.7], np.float32)
  x = host(a, dev)
  np.testing.assert_array_equal(val(x.cast(dtypes.int32)), np.array([-2, 0, 0, 0, 2], np.int32))
  np.testing.assert_array_equal(val(x.cast(dtypes.bool)), a != 0)
  np.testing.assert_array_equal(val(x.cast(dtypes.int32, bitcast=True)), a.view(np.int32))

def test_bf16_storage_is_bits(dev):
  import torch
  a = np.random.default_rng(1).uniform(-100, 100, 64).astype(np.float32)
  x = host(a, dev).cast(dtypes.bfloat16)
  interpret.realize([x])
  bits = x.realized.numpy()
  assert bits.dtype == np.uint16
  np.testing.assert_array_equal(bits, torch.tensor(a).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))
  # arithmetic rounds per op to bf16
  y = x.e(BinaryOps.MUL, x)
  ty = torch.tensor(a).to(torch.bfloat16) * torch.tensor(a).to(torch.bfloat16)
  np.testing.assert_array_equal(val(y), ty.view(torch.int16).numpy().view(np.uint16))

def test_from_buffer_and_realized(dev):
  buf = Buffer(dev, 6, dtypes.float32).ensure_allocated()
  buf.copyin(memoryview(np.arange(6, dtype=np.float32)).cast("B"))
  lb = LazyBuffer.from_buffer(buf, (2, 3))
  assert lb.is_realized() and lb.realized is buf
  np.testing.assert_array_equal(val(lb.permute((1, 0))), np.arange(6, dtype=np.float32).reshape(2, 3).T)

def test_shared_subgraph_computed_once(dev):
  x = host(np.ones(4, np.float32), dev)
  y = x.e(BinaryOps.ADD, x)
  z = y.e(BinaryOps.MUL, y)
  w = z.e(BinaryOps.ADD, y)
  interpret.realize([w, z])
  assert y.realized is not None and z.realized is not None
  np.testing.assert_array_equal(w.realized.numpy(), np.full(4, 6, np.float32))

def test_deep_graph_no_recursion(dev):
  x = host(np.zeros(3, np.float32), dev)
  one = x.const(1.0)
  for _ in range(5000): x = x.e(BinaryOps.ADD, one)
  np.testing.assert_array_equal(val(x), np.full(3, 5000, np.float32))

def test_symbolic_view(dev):
  n = Var("n", 1, 8)
  a = np.arange(8, dtype=np.float32)
  x = host(a, dev)
  v = x.shrink(((0, n),))
  y = v.e(BinaryOps.ADD, v)
  for k in (1, 3, 8):
    y2 = y.base  # fresh evaluation per value: build a new graph each time
    lb = x.shrink(((0, n),)).e(BinaryOps.MUL, x.shrink(((0, n),)))
    interpret.realize([lb], {n: k})
    np.testing.assert_array_equal(lb.realized.numpy(), (a * a)[:k])
    del y2

def test_copy_to_device_noop_and_const(dev):
  x = host(np.ones(2, np.float32), dev)
  assert x.copy_to_device(dev) is x
  c = x.const(2.0)
  assert c.copy_to_device("NPY").is_unrealized_const()
