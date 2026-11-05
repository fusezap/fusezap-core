"""WP2 regression tests for the second review round (one per confirmed issue), against torch."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from fusezap import Tensor, dtypes, interpret
from fusezap.shape.symbolic import Var

def grads(fn, *arrs):
  xs = [Tensor(a, requires_grad=True) for a in arrs]
  ts = [torch.tensor(a, requires_grad=True) for a in arrs]
  fn(*xs).sum().backward(); fn(*ts).sum().backward()
  for x, t in zip(xs, ts): np.testing.assert_allclose(x.grad.numpy(), t.grad.numpy(), rtol=1e-5, atol=1e-6, equal_nan=False)

def test_bool_list_index_rejected(device):
  x = Tensor([0., 1., 2.])
  for idx in ([True, False, True], np.array([True, False, True])):
    with pytest.raises(NotImplementedError): x[idx]
  np.testing.assert_array_equal(x[[2, 0]].numpy(), [2., 0.])
  np.testing.assert_array_equal(x[np.array([1], np.int64)].numpy(), [1.])

@pytest.mark.parametrize("npdt", [np.int8, np.int16, np.int32, np.int64])
def test_min_argmin_signed_dtype_min(device, npdt):
  lo, hi = np.iinfo(npdt).min, np.iinfo(npdt).max
  a = np.array([[3, lo, 5], [hi, 0, lo]], npdt)
  x, t = Tensor(a), torch.tensor(a)
  assert x.min().item() == lo and x.argmin().item() == 1
  np.testing.assert_array_equal(x.min(1).numpy(), t.amin(1).numpy())
  np.testing.assert_array_equal(x.argmin(1).numpy(), t.argmin(1).numpy())
  np.testing.assert_array_equal(x.min(0).numpy(), t.amin(0).numpy())

def test_pow_zero_base_gradients(device):
  z = np.array([0., 1., 2.], np.float32)
  for e in (2.5, 0.5, 1.5, -0.5, 3.5):
    grads(lambda x: x ** e, z)
  b = np.array([0., 0., 0., 0., 2., 3.], np.float32); e = np.array([2., 2.5, 0., 1., 0.5, -1.5], np.float32)
  grads(lambda x, y: x ** y, b, e)
  grads(lambda y: 2.0 ** y, e)
  x = Tensor(np.array([0., 0.], np.float32), requires_grad=True); y = Tensor(np.array([0.5, -0.5], np.float32), requires_grad=True)
  (x ** y).sum().backward()
  np.testing.assert_array_equal(x.grad.numpy(), [np.inf, -np.inf]); np.testing.assert_array_equal(y.grad.numpy(), [0., -np.inf])

@pytest.mark.parametrize("dt,tdt", [(dtypes.bfloat16, torch.bfloat16), (dtypes.float16, torch.float16)])
def test_exp_log_half_upcast(device, dt, tdt):
  a = np.array([-3., 0.5, 5., 7.25, 10.5, 11.], np.float32)
  ref = torch.tensor(a).to(tdt).exp().float().numpy()
  np.testing.assert_allclose(Tensor(a).cast(dt).exp().float().numpy(), ref, rtol=1e-2 if dt == dtypes.bfloat16 else 1e-3)
  p = np.abs(a) + 0.1
  np.testing.assert_allclose(Tensor(p).cast(dt).log().float().numpy(), torch.tensor(p).to(tdt).log().float().numpy(), rtol=1e-2, atol=1e-2)
  assert Tensor(a).cast(dt).exp().dtype == dt

def test_clip_boundary_gradient(device):
  a = np.array([-1., 0., 0.5, 1., 2.], np.float32)
  grads(lambda x: x.clip(0., 1.), a)
  grads(lambda x: x.clamp(min=0.), np.array([0., 1.], np.float32))
  grads(lambda x: x.clamp(max=1.), a)
  np.testing.assert_array_equal(Tensor(np.array([np.nan, 3.], np.float32)).clip(0., 1.).numpy(), [np.nan, 1.])
  np.testing.assert_array_equal(Tensor(np.array([-5, 3, 9], np.int32)).clip(0, 4).numpy(), [0, 3, 4])

@pytest.mark.parametrize("npdt", [np.int8, np.uint8, np.int16, np.bool_])
def test_cumsum_small_int_promotes(device, npdt):
  a = np.array([100, 100, 50], npdt) if npdt != np.bool_ else np.array([True, True, False])
  x = Tensor(a)
  np.testing.assert_array_equal(x.cumsum().numpy(), torch.tensor(a).cumsum(0).numpy())
  assert x.cumsum().dtype == x.sum().dtype == dtypes.default_int

def test_mean_var_symbolic_axis():
  n = Var("n", 1, 10)
  a = np.arange(10, dtype=np.float32) ** 1.5
  for k in (1, 3, 10):
    v = Tensor(a).shrink(((0, n),))
    outs = [v.mean(), v.var(correction=0), v.std(correction=0), v.reshape(1, n).mean(1)]
    interpret.realize([o.lazydata for o in outs], {n: k})
    got = [o.lazydata.base.realized.numpy().reshape(-1)[0] for o in outs]
    np.testing.assert_allclose(got, [a[:k].mean(), a[:k].var(), a[:k].std(), a[:k].mean()], rtol=1e-5, atol=1e-6)
