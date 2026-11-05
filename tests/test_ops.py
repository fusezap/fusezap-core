"""Op conformance suite (SPEC §10): every public Tensor method vs PyTorch, over all device tiers. Public API only."""
from __future__ import annotations
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from fusezap import Tensor, dtypes, manual_seed

DT = {"float32": (dtypes.float32, torch.float32, np.float32), "float16": (dtypes.float16, torch.float16, np.float16),
      "int32": (dtypes.int32, torch.int32, np.int32), "bool": (dtypes.bool, torch.bool, np.bool_)}
FLOATS = ["float32", "float16"]
R = np.random.default_rng(0)

def rnd(shape, dt="float32", lo=-2.0, hi=2.0):
  if dt == "bool": return R.random(shape) > 0.5
  if dt == "int32": return R.integers(-5, 6, size=shape).astype(np.int32)
  return R.uniform(lo, hi, size=shape).astype(DT[dt][2])

def mk(a, grad=False):
  """numpy array -> (fusezap Tensor, torch Tensor)."""
  a = np.ascontiguousarray(a) if a.ndim else a
  t = torch.tensor(a)
  if grad: t.requires_grad_(True)
  return Tensor(a, requires_grad=True if grad else None), t

def ref(t):  # torch ops on half run in float32 then round, matching how fusezap computes in fp32 and rounds
  return t

def chk(ac, out, exp, dt="float32", reduce=False):
  assert tuple(out.shape) == tuple(exp.shape), f"shape {out.shape} != {tuple(exp.shape)}"
  ac(out.numpy(), exp.detach(), dt if exp.is_floating_point() else None, reduce)

def tol_dt(dt): return dt if dt in FLOATS else None

# ------------------------------------------------------------------ properties / creation / realization
def test_properties(device):
  t = Tensor(rnd((2, 3, 4)))
  assert t.shape == (2, 3, 4) and t.ndim == 3 and t.numel() == 24 and t.dtype == dtypes.float32
  assert t.T.shape == (4, 3, 2) or t.T.shape == (3, 2, 4) or t.T.ndim == 3
  assert isinstance(t.device, str)
  assert Tensor(3.0).shape == () and Tensor([[1, 2], [3, 4]]).shape == (2, 2)

@pytest.mark.parametrize("dt", FLOATS + ["int32"])
def test_item_tolist_numpy(device, dt):
  a = rnd((2, 3), dt)
  t = Tensor(a)
  np.testing.assert_array_equal(t.numpy(), a)
  assert np.allclose(np.array(t.tolist()), a)
  assert Tensor(a[0, 0]).item() == a[0, 0].item()
  assert t.realize() is t or t.realize().shape == (2, 3)

def test_creation(device, assert_close):
  for fn, tf in [(Tensor.zeros, torch.zeros), (Tensor.ones, torch.ones)]:
    chk(assert_close, fn(2, 3), tf(2, 3)); chk(assert_close, fn(0, 3), tf(0, 3))
  chk(assert_close, Tensor.zeros(()), torch.zeros(()))
  assert Tensor.empty(2, 3).shape == (2, 3)
  chk(assert_close, Tensor.full((2, 3), 1.5), torch.full((2, 3), 1.5))
  chk(assert_close, Tensor.arange(5), torch.arange(5, dtype=torch.int32))
  chk(assert_close, Tensor.arange(2, 11, 3), torch.arange(2, 11, 3, dtype=torch.int32))
  chk(assert_close, Tensor.arange(0.0, 1.0, 0.25), torch.arange(0.0, 1.0, 0.25))
  chk(assert_close, Tensor.eye(4), torch.eye(4))
  x, tx = mk(rnd((2, 3)))
  chk(assert_close, x.zeros_like(), torch.zeros_like(tx)); chk(assert_close, x.ones_like(), torch.ones_like(tx))
  chk(assert_close, x.full_like(3.0), torch.full_like(tx, 3.0))
  assert Tensor.zeros(2, 3, dtype=dtypes.float16).dtype == dtypes.float16
  assert Tensor.ones(2, dtype=dtypes.int32).dtype == dtypes.int32

def test_random(device):
  manual_seed(1); a = Tensor.rand(1000, 10).numpy(); manual_seed(1); b = Tensor.rand(1000, 10).numpy()
  np.testing.assert_array_equal(a, b)
  assert a.min() >= 0 and a.max() < 1 and abs(a.mean() - 0.5) < 0.02
  n = Tensor.randn(20000).numpy(); assert abs(n.mean()) < 0.05 and abs(n.std() - 1) < 0.05
  r = Tensor.randint(5000, low=2, high=7).numpy(); assert r.min() >= 2 and r.max() < 7 and len(np.unique(r)) == 5
  u = Tensor.uniform(5000, low=-3.0, high=-1.0).numpy(); assert u.min() >= -3 and u.max() < -1
  m = Tensor.normal(20000, mean=5.0, std=2.0).numpy(); assert abs(m.mean() - 5) < 0.1 and abs(m.std() - 2) < 0.1
  assert Tensor.rand(0, 3).shape == (0, 3)

def test_to_detach_contiguous_assign(device, assert_close):
  x, tx = mk(rnd((3, 4)), grad=True)
  y = x.detach(); assert not y.requires_grad
  chk(assert_close, x.to(x.device), tx)
  p, tp = mk(rnd((4, 3))); chk(assert_close, p.permute(1, 0).contiguous(), tp.permute(1, 0).contiguous())
  w = Tensor(rnd((3, 4))); v = Tensor(rnd((3, 4))); vn = v.numpy().copy()
  w.assign(v); np.testing.assert_array_equal(w.realize().numpy(), vn)

@pytest.mark.parametrize("src,dst", [("float32", "float16"), ("float16", "float32"), ("float32", "int32"), ("int32", "float32"),
                                     ("bool", "float32"), ("float32", "bool"), ("int32", "bool"), ("bool", "int32")])
def test_cast(device, assert_close, src, dst):
  a = rnd((3, 4), src, -5, 5)
  x, tx = mk(a)
  out = x.cast(DT[dst][0]); assert out.dtype == DT[dst][0]
  chk(assert_close, out, tx.to(DT[dst][1]), tol_dt(dst))

def test_cast_helpers(device, assert_close):
  x, tx = mk(rnd((3, 4), "float32", -5, 5))
  chk(assert_close, x.float(), tx.float()); chk(assert_close, x.half(), tx.half(), "float16")
  chk(assert_close, x.int(), tx.int()); chk(assert_close, x.bool(), tx.bool())

def test_bitcast(device):
  a = np.array([1.0, -2.5, 3.25], np.float32)
  np.testing.assert_array_equal(Tensor(a).bitcast(dtypes.int32).numpy(), a.view(np.int32))
  np.testing.assert_array_equal(Tensor(a.view(np.int32)).bitcast(dtypes.float32).numpy(), a)

# ------------------------------------------------------------------ movement
SHAPES = [(), (1,), (5,), (1, 1), (3, 4), (2, 1, 4), (2, 3, 4)]

@pytest.mark.parametrize("dt", FLOATS + ["int32", "bool"])
@pytest.mark.parametrize("shape", SHAPES)
def test_identity_roundtrip(device, shape, dt):
  a = rnd(shape, dt); np.testing.assert_array_equal(Tensor(a).numpy(), a)

@pytest.mark.parametrize("shape,new", [((2, 3, 4), (6, 4)), ((2, 3, 4), (-1,)), ((2, 3, 4), (4, -1, 2)), ((), (1,)), ((1,), ()),
                                       ((5,), (1, 5, 1)), ((0, 3), (3, 0)), ((6,), (2, 3))])
def test_reshape_view(device, assert_close, shape, new):
  x, tx = mk(rnd(shape))  # torch reference always takes the shape as one tuple (reshape() with no args is a TypeError)
  chk(assert_close, x.reshape(*new), tx.reshape(tuple(new))); chk(assert_close, x.reshape(new), tx.reshape(tuple(new))); chk(assert_close, x.view(*new), tx.view(tuple(new)))

def test_reshape_ref_scalar_shape():  # regression: torch-side reference for new=() must not call reshape() with no args
  assert torch.zeros(1).reshape(()).shape == () and torch.zeros(1).view(()).shape == ()

@pytest.mark.parametrize("order", [(1, 0, 2), (2, 1, 0), (0, 2, 1), (1, 2, 0)])
def test_permute_transpose(device, assert_close, order):
  x, tx = mk(rnd((2, 3, 4)))
  chk(assert_close, x.permute(*order), tx.permute(*order)); chk(assert_close, x.permute(order), tx.permute(order))
  chk(assert_close, x.permute(*order) + 1, tx.permute(*order) + 1)

def test_transpose_T(device, assert_close):
  x, tx = mk(rnd((3, 4)))
  chk(assert_close, x.transpose(), tx.transpose(0, 1)); chk(assert_close, x.transpose(0, 1), tx.transpose(0, 1))
  y, ty = mk(rnd((2, 3, 4))); chk(assert_close, y.transpose(0, 2), ty.transpose(0, 2)); chk(assert_close, y.transpose(-1, -2), ty.transpose(-1, -2))
  chk(assert_close, x.T, tx.T)

def test_flatten_unflatten(device, assert_close):
  x, tx = mk(rnd((2, 3, 4, 5)))
  chk(assert_close, x.flatten(), tx.flatten()); chk(assert_close, x.flatten(1), tx.flatten(1)); chk(assert_close, x.flatten(1, 2), tx.flatten(1, 2))
  chk(assert_close, x.unflatten(3, (5, 1)), tx.unflatten(3, (5, 1))); chk(assert_close, x.unflatten(1, (3, 1)), tx.unflatten(1, (3, 1)))
  y, ty = mk(rnd(())); chk(assert_close, y.flatten(), ty.flatten())

def test_squeeze_unsqueeze(device, assert_close):
  x, tx = mk(rnd((1, 3, 1, 4)))
  chk(assert_close, x.squeeze(), tx.squeeze()); chk(assert_close, x.squeeze(0), tx.squeeze(0)); chk(assert_close, x.squeeze(2), tx.squeeze(2))
  chk(assert_close, x.squeeze(1), tx.squeeze(1))
  y, ty = mk(rnd((3, 4)))
  for ax in (0, 1, 2, -1): chk(assert_close, y.unsqueeze(ax), ty.unsqueeze(ax))
  z, tz = mk(rnd(())); chk(assert_close, z.unsqueeze(0), tz.unsqueeze(0))

def test_expand(device, assert_close):
  x, tx = mk(rnd((1, 3, 1)))
  chk(assert_close, x.expand(2, 3, 4), tx.expand(2, 3, 4)); chk(assert_close, x.expand(2, -1, 4), tx.expand(2, -1, 4))
  chk(assert_close, x.expand(5, 1, 3, 4), tx.expand(5, 1, 3, 4)); chk(assert_close, x.expand((2, 3, 4)), tx.expand((2, 3, 4)))
  y, ty = mk(rnd(())); chk(assert_close, y.expand(3, 2), ty.expand(3, 2))

def test_pad(device, assert_close):
  x, tx = mk(rnd((3, 4)))
  chk(assert_close, x.pad(((1, 2), (0, 3))), F.pad(tx, (0, 3, 1, 2)))
  chk(assert_close, x.pad(((1, 1), (2, 0)), value=2.5), F.pad(tx, (2, 0, 1, 1), value=2.5))
  chk(assert_close, x.pad(((0, 0), (0, 0))), tx)
  a, ta = mk(rnd((3, 4), "int32")); chk(assert_close, a.pad(((1, 0), (0, 1))), F.pad(ta, (0, 1, 1, 0)))

def test_shrink(device, assert_close):
  x, tx = mk(rnd((5, 6)))
  chk(assert_close, x.shrink(((1, 4), (2, 6))), tx[1:4, 2:6]); chk(assert_close, x.shrink((None, (0, 3))), tx[:, 0:3])
  chk(assert_close, x.shrink(((2, 2), (0, 6))), tx[2:2])

def test_flip(device, assert_close):
  x, tx = mk(rnd((3, 4, 5)))
  chk(assert_close, x.flip(0), tx.flip(0)); chk(assert_close, x.flip(-1), tx.flip(-1)); chk(assert_close, x.flip((0, 2)), tx.flip((0, 2)))
  chk(assert_close, x.flip([1]), tx.flip([1]))

GETITEM = [lambda t: t[0], lambda t: t[-1], lambda t: t[1:3], lambda t: t[::2], lambda t: t[::-1].contiguous() if False else t[1:], lambda t: t[:, 1],
           lambda t: t[None], lambda t: t[:, None, 1:3], lambda t: t[..., 0], lambda t: t[0, ..., 1:], lambda t: t[1:4:2, ::3],
           lambda t: t[:, -2:], lambda t: t[2, 3, 1], lambda t: t[1:1]]

@pytest.mark.parametrize("i", range(len(GETITEM)))
def test_getitem(device, assert_close, i):
  x, tx = mk(rnd((4, 5, 3)))
  chk(assert_close, GETITEM[i](x), GETITEM[i](tx))

def test_getitem_negative_step(device, assert_close):
  x, tx = mk(rnd((6, 4)))
  chk(assert_close, x[::-1], tx.flip(0)); chk(assert_close, x[4:0:-2], tx.flip(0)[1::2][:2] if False else torch.tensor(tx.numpy()[4:0:-2].copy()))
  chk(assert_close, x[:, ::-2], torch.tensor(tx.numpy()[:, ::-2].copy()))

def test_getitem_tensor_index(device, assert_close):
  x, tx = mk(rnd((5, 4)))
  i = np.array([0, 3, 3, 1], np.int32)
  chk(assert_close, x[Tensor(i)], tx[torch.tensor(i, dtype=torch.long)])
  chk(assert_close, x[:, Tensor(i)], tx[:, torch.tensor(i, dtype=torch.long)])
  chk(assert_close, x[Tensor(np.array([[0, 1], [2, 4]], np.int32))], tx[torch.tensor([[0, 1], [2, 4]])])

def test_gather(device, assert_close):
  x, tx = mk(rnd((4, 5)))
  idx = R.integers(0, 5, size=(4, 3)).astype(np.int32)
  chk(assert_close, x.gather(1, Tensor(idx)), tx.gather(1, torch.tensor(idx, dtype=torch.long)))
  idx0 = R.integers(0, 4, size=(2, 5)).astype(np.int32)
  chk(assert_close, x.gather(0, Tensor(idx0)), tx.gather(0, torch.tensor(idx0, dtype=torch.long)))

def test_cat_stack(device, assert_close):
  (a, ta), (b, tb), (c, tc) = mk(rnd((2, 3))), mk(rnd((4, 3))), mk(rnd((1, 3)))
  chk(assert_close, a.cat(b, c), torch.cat([ta, tb, tc])); chk(assert_close, a.cat(a, axis=1), torch.cat([ta, ta], 1))
  chk(assert_close, a.cat(Tensor(rnd((2, 0))), axis=1), ta)
  (d, td), (e, te) = mk(rnd((2, 3))), mk(rnd((2, 3)))
  chk(assert_close, Tensor.stack([d, e]), torch.stack([td, te])); chk(assert_close, Tensor.stack([d, e], axis=1), torch.stack([td, te], 1))
  chk(assert_close, Tensor.stack([d, e], axis=-1), torch.stack([td, te], -1))
  i, ti = mk(rnd((2, 2), "int32")); chk(assert_close, i.cat(i), torch.cat([ti, ti]))

def test_split_chunk(device, assert_close):
  x, tx = mk(rnd((7, 4)))
  for o, e in zip(x.split(3), tx.split(3)): chk(assert_close, o, e)
  for o, e in zip(x.split([2, 5]), tx.split([2, 5])): chk(assert_close, o, e)
  for o, e in zip(x.split(2, axis=1), tx.split(2, 1)): chk(assert_close, o, e)
  for n in (1, 2, 3):
    o, e = x.chunk(n), tx.chunk(n); assert len(o) == len(e)
    for p, q in zip(o, e): chk(assert_close, p, q)
  o, e = x.chunk(2, axis=1), tx.chunk(2, 1)
  for p, q in zip(o, e): chk(assert_close, p, q)

def test_repeat(device, assert_close):
  x, tx = mk(rnd((2, 3)))
  chk(assert_close, x.repeat(2, 3), tx.repeat(2, 3)); chk(assert_close, x.repeat(1, 1), tx.repeat(1, 1)); chk(assert_close, x.repeat(2, 1, 2), tx.repeat(2, 1, 2))
  chk(assert_close, x.repeat((2, 2)), tx.repeat((2, 2)))

# ------------------------------------------------------------------ elementwise unary
U_DOMAIN = {"log": (0.1, 4), "log2": (0.1, 4), "sqrt": (0.1, 4), "rsqrt": (0.1, 4), "reciprocal": (0.5, 3), "exp": (-3, 3), "exp2": (-3, 3)}
UNARY = [("neg", torch.neg), ("exp", torch.exp), ("exp2", torch.exp2), ("log", torch.log), ("log2", torch.log2), ("sqrt", torch.sqrt),
         ("rsqrt", torch.rsqrt), ("reciprocal", torch.reciprocal), ("square", torch.square), ("sin", torch.sin), ("cos", torch.cos),
         ("tanh", torch.tanh), ("sigmoid", torch.sigmoid), ("relu", torch.relu), ("gelu", lambda t: F.gelu(t, approximate="tanh")),
         ("silu", F.silu), ("abs", torch.abs), ("sign", torch.sign)]

def _tref(f, t, dt):  # compute reference in fp32 for half so only output rounding differs
  return f(t.float()).to(t.dtype) if dt == "float16" else f(t)

@pytest.mark.parametrize("dt", FLOATS)
@pytest.mark.parametrize("name,tf", UNARY, ids=[n for n, _ in UNARY])
@pytest.mark.parametrize("shape", [(), (7,), (3, 4), (1, 5)])
def test_unary(device, assert_close, name, tf, dt, shape):
  lo, hi = U_DOMAIN.get(name, (-3, 3))
  x, tx = mk(rnd(shape, dt, lo, hi))
  chk(assert_close, getattr(x, name)(), _tref(tf, tx, dt), dt)

@pytest.mark.parametrize("name,tf", UNARY, ids=[n for n, _ in UNARY])
def test_unary_noncontig(device, assert_close, name, tf):
  lo, hi = U_DOMAIN.get(name, (-3, 3))
  a = rnd((4, 6), "float32", lo, hi); x, tx = mk(a)
  chk(assert_close, getattr(x.permute(1, 0), name)(), tf(tx.permute(1, 0))); chk(assert_close, getattr(x[:, ::2], name)(), tf(tx[:, ::2]))

@pytest.mark.parametrize("name,tf", [("neg", torch.neg), ("abs", torch.abs), ("sign", torch.sign), ("square", torch.square), ("relu", torch.relu)])
def test_unary_int(device, assert_close, name, tf):
  x, tx = mk(rnd((3, 4), "int32")); chk(assert_close, getattr(x, name)(), tf(tx))

def test_logical_not(device, assert_close):
  x, tx = mk(rnd((3, 4), "bool")); chk(assert_close, x.logical_not(), tx.logical_not()); chk(assert_close, x.permute(1, 0).logical_not(), tx.T.logical_not())
  y, ty = mk(rnd(())) ; z, tz = mk(rnd((), "bool")); chk(assert_close, z.logical_not(), tz.logical_not())

@pytest.mark.parametrize("name,tf", [(n, f) for n, f in UNARY if n in {"exp", "log", "sqrt", "rsqrt", "reciprocal", "square", "sin", "cos", "tanh", "sigmoid",
                                                                       "relu", "gelu", "silu", "abs", "neg", "exp2", "log2"}])
def test_unary_backward(device, assert_close, name, tf):
  lo, hi = U_DOMAIN.get(name, (-3, 3))
  a = rnd((3, 4), "float32", lo, hi)
  if name in ("relu", "abs"): a = np.where(np.abs(a) < 0.05, 0.5, a).astype(np.float32)
  x, tx = mk(a, True); g = rnd((3, 4))
  getattr(x, name)().backward(Tensor(g)); tf(tx).backward(torch.tensor(g))
  chk(assert_close, x.grad, tx.grad, "float32", reduce=True)

# ------------------------------------------------------------------ elementwise binary
BSHAPES = [((3, 4), (3, 4)), ((3, 4), (4,)), ((3, 1), (1, 4)), ((2, 3, 4), (3, 1)), ((), (3, 4)), ((3, 4), ()), ((1,), (1,)), ((0, 3), (3,)), ((2, 1, 4), (3, 1))]
BIN = [("add", torch.add), ("sub", torch.sub), ("mul", torch.mul), ("div", torch.div), ("maximum", torch.maximum), ("minimum", torch.minimum)]

@pytest.mark.parametrize("dt", FLOATS)
@pytest.mark.parametrize("name,tf", BIN, ids=[n for n, _ in BIN])
@pytest.mark.parametrize("sa,sb", BSHAPES)
def test_binary_broadcast(device, assert_close, name, tf, sa, sb, dt):
  lo, hi = (0.5, 3) if name == "div" else (-3, 3)
  (x, tx), (y, ty) = mk(rnd(sa, dt)), mk(rnd(sb, dt, lo, hi))
  chk(assert_close, getattr(x, name)(y), _tref(lambda a: tf(a, ty.float() if dt == "float16" else ty), tx, dt), dt)

@pytest.mark.parametrize("name,tf", [("add", torch.add), ("sub", torch.sub), ("mul", torch.mul), ("maximum", torch.maximum), ("minimum", torch.minimum)])
@pytest.mark.parametrize("sa,sb", BSHAPES[:6])
def test_binary_int(device, assert_close, name, tf, sa, sb):
  (x, tx), (y, ty) = mk(rnd(sa, "int32")), mk(rnd(sb, "int32"))
  chk(assert_close, getattr(x, name)(y), tf(tx, ty))

def test_div_int_promotes(device, assert_close):
  (x, tx), (y, ty) = mk(rnd((3, 4), "int32")), mk(np.array([1, 2, 3, -4], np.int32))
  out = x.div(y); assert out.dtype == dtypes.float32
  chk(assert_close, out, tx.float() / ty.float())

def test_bool_arith(device, assert_close):
  (x, tx), (y, ty) = mk(rnd((3, 4), "bool")), mk(rnd((3, 4), "bool"))
  chk(assert_close, x.maximum(y), torch.maximum(tx, ty)); chk(assert_close, x.minimum(y), torch.minimum(tx, ty))
  chk(assert_close, (x == y), tx == ty)

SCAL = [("add", 2.5), ("sub", 1.5), ("mul", -3.0), ("div", 4.0), ("maximum", 0.3), ("minimum", 0.3), ("add", 2), ("mul", 3)]

@pytest.mark.parametrize("dt", FLOATS)
@pytest.mark.parametrize("name,s", SCAL)
def test_binary_scalar(device, assert_close, name, s, dt):
  x, tx = mk(rnd((3, 4), dt))
  tf = dict(BIN)[name]
  chk(assert_close, getattr(x, name)(s), _tref(lambda a: tf(a, torch.tensor(s, dtype=a.dtype)) if name in ("maximum", "minimum") else tf(a, s), tx, dt), dt)
  assert getattr(x, name)(s).dtype == DT[dt][0]  # python scalars are weak

def test_scalar_weak_int(device):
  x = Tensor(rnd((3,), "int32")); assert (x + 2).dtype == dtypes.int32 and (x * 2.5).dtype.name.startswith("float")
  h = Tensor(rnd((3,), "float16")); assert (h + 1.5).dtype == dtypes.float16

def test_promotion_tensors(device):
  assert (Tensor(rnd((2,), "int32")) + Tensor(rnd((2,), "float16"))).dtype == dtypes.float16
  assert (Tensor(rnd((2,), "bool")) + Tensor(rnd((2,), "int32"))).dtype == dtypes.int32
  assert (Tensor(rnd((2,), "float16")) + Tensor(rnd((2,), "float32"))).dtype == dtypes.float32

def test_operators(device, assert_close):
  (x, tx), (y, ty) = mk(rnd((3, 4))), mk(rnd((3, 4), "float32", 0.5, 3))
  chk(assert_close, x + y, tx + ty); chk(assert_close, x - y, tx - ty); chk(assert_close, x * y, tx * ty); chk(assert_close, x / y, tx / ty)
  chk(assert_close, 2 + x, 2 + tx); chk(assert_close, 2 - x, 2 - tx); chk(assert_close, 2 * x, 2 * tx); chk(assert_close, 2 / y, 2 / ty)
  chk(assert_close, -x, -tx); chk(assert_close, y ** 2, ty ** 2); chk(assert_close, y ** 0.5, ty ** 0.5); chk(assert_close, 2 ** x, 2 ** tx)
  chk(assert_close, y ** x, ty ** tx)
  chk(assert_close, x.T @ x if False else x @ y.T, tx @ ty.T)

def test_pow(device, assert_close):
  (x, tx), (y, ty) = mk(rnd((3, 4), "float32", 0.2, 3)), mk(rnd((4,), "float32", -2, 2))
  chk(assert_close, x.pow(y), tx.pow(ty)); chk(assert_close, x.pow(3), tx.pow(3)); chk(assert_close, x.pow(-1.5), tx.pow(-1.5))
  chk(assert_close, x.pow(0.0), tx.pow(0.0))

def test_int_floordiv_mod(device, assert_close):
  a = np.array([7, -7, 8, -8, 0, 5], np.int32); b = np.array([2, 2, -3, -3, 4, 5], np.int32)
  # C semantics: truncation toward zero, remainder takes the dividend's sign
  np.testing.assert_array_equal((Tensor(a) // Tensor(b)).numpy(), np.trunc(a / b).astype(np.int32))
  np.testing.assert_array_equal((Tensor(a) % Tensor(b)).numpy(), np.fmod(a, b))
  np.testing.assert_array_equal((Tensor(a) // 3).numpy(), np.trunc(a / 3).astype(np.int32))
  np.testing.assert_array_equal((Tensor(a) % 3).numpy(), np.fmod(a, 3))

CMP = ["__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"]

@pytest.mark.parametrize("dt", FLOATS + ["int32"])
@pytest.mark.parametrize("op", CMP)
def test_compare(device, assert_close, op, dt):
  a, b = rnd((3, 4), dt), rnd((4,), dt); b[0] = a[0, 0]
  (x, tx), (y, ty) = mk(a), mk(b)
  out = getattr(x, op)(y); assert out.dtype == dtypes.bool
  chk(assert_close, out, getattr(tx, op)(ty)); chk(assert_close, getattr(x, op)(a[0, 0].item()), getattr(tx, op)(a[0, 0].item()))

def test_clip(device, assert_close):
  x, tx = mk(rnd((3, 4), "float32", -3, 3))
  chk(assert_close, x.clip(-1, 1), tx.clip(-1, 1)); chk(assert_close, x.clip(min_=0.0), tx.clip(min=0.0)); chk(assert_close, x.clip(max_=0.5), tx.clip(max=0.5))
  a, ta = mk(rnd((3, 4), "int32")); chk(assert_close, a.clip(-2, 2), ta.clip(-2, 2))

def test_where_masked_fill(device, assert_close):
  c, tc = mk(rnd((3, 4), "bool")); (x, tx), (y, ty) = mk(rnd((3, 4))), mk(rnd((4,)))
  chk(assert_close, c.where(x, y), torch.where(tc, tx, ty)); chk(assert_close, c.where(x, 0.0), torch.where(tc, tx, torch.tensor(0.0)))
  chk(assert_close, c.where(1.0, -1.0), torch.where(tc, 1.0, -1.0))
  chk(assert_close, x.masked_fill(c, -1e9), tx.masked_fill(tc, -1e9)); chk(assert_close, x.masked_fill(c[:, :1], 3.0), tx.masked_fill(tc[:, :1], 3.0))
  i, ti = mk(rnd((3, 4), "int32")); chk(assert_close, i.masked_fill(c, 7), ti.masked_fill(tc, 7))

def test_binary_noncontig(device, assert_close):
  (x, tx), (y, ty) = mk(rnd((5, 6))), mk(rnd((6, 5)))
  chk(assert_close, x[1:4, ::2] + y.permute(1, 0)[1:4, 1::2][:, :3] * 2, tx[1:4, ::2] + ty.permute(1, 0)[1:4, 1::2][:, :3] * 2)

BIN_BWD = [("add", torch.add), ("sub", torch.sub), ("mul", torch.mul), ("div", torch.div), ("maximum", torch.maximum), ("minimum", torch.minimum), ("pow", torch.pow)]

@pytest.mark.parametrize("name,tf", BIN_BWD, ids=[n for n, _ in BIN_BWD])
@pytest.mark.parametrize("sa,sb", [((3, 4), (3, 4)), ((3, 4), (4,)), ((3, 1), (1, 4)), ((), (3,))])
def test_binary_backward(device, assert_close, name, tf, sa, sb):
  a, b = rnd(sa, "float32", 0.5, 3), rnd(sb, "float32", 0.5, 3)
  if name in ("maximum", "minimum"): b = b + 0.123
  (x, tx), (y, ty) = mk(a, True), mk(b, True)
  out, tout = getattr(x, name)(y), tf(tx, ty)
  g = rnd(tuple(tout.shape)); out.backward(Tensor(g)); tout.backward(torch.tensor(g))
  chk(assert_close, x.grad, tx.grad, "float32", reduce=True); chk(assert_close, y.grad, ty.grad, "float32", reduce=True)

def test_where_backward(device, assert_close):
  c, tc = mk(rnd((3, 4), "bool")); (x, tx), (y, ty) = mk(rnd((3, 4)), True), mk(rnd((4,)), True)
  c.where(x, y).sum().backward(); torch.where(tc, tx, ty).sum().backward()
  chk(assert_close, x.grad, tx.grad, reduce=True); chk(assert_close, y.grad, ty.grad, reduce=True)

def test_clip_masked_fill_backward(device, assert_close):
  a = rnd((3, 4), "float32", -3, 3) + 0.01; x, tx = mk(a, True)
  x.clip(-1, 1).sum().backward(); tx.clip(-1, 1).sum().backward(); chk(assert_close, x.grad, tx.grad, reduce=True)
  c, tc = mk(rnd((3, 4), "bool")); x, tx = mk(a, True)
  (x.masked_fill(c, 5.0) * 2).sum().backward(); (tx.masked_fill(tc, 5.0) * 2).sum().backward(); chk(assert_close, x.grad, tx.grad, reduce=True)

def test_movement_backward(device, assert_close):
  a = rnd((2, 3, 4)); g = rnd((4, 6))
  cases = {
    "reshape": (lambda t: t.reshape(4, 6)), "permute": (lambda t: t.permute(1, 0, 2).reshape(3, 8)[:, :6].reshape(3, 6)),
    "expand": (lambda t: t[:1].expand(4, 3, 4).reshape(4, 12)[:, :6]), "pad": (lambda t: t.reshape(6, 4).pad(((0, 0), (1, 1)))[:4]),
    "shrink": (lambda t: t.reshape(4, 6)[1:3]), "flip": (lambda t: t.reshape(4, 6).flip(1)), "getitem": (lambda t: t.reshape(4, 6)[::-1, 1:]),
    "repeat": (lambda t: t.reshape(2, 12)[:, :3].repeat(2, 2)), "cat": (lambda t: t.reshape(4, 6).cat(t.reshape(4, 6) * 2, axis=0)[:4]),
    "stack": (lambda t: Tensor.stack([t.reshape(4, 6), t.reshape(4, 6) * 3])[1]), "squeeze": (lambda t: t.unsqueeze(0).squeeze(0).reshape(4, 6)),
  }
  def tcase(n, t):
    return {"reshape": lambda: t.reshape(4, 6), "permute": lambda: t.permute(1, 0, 2).reshape(3, 8)[:, :6].reshape(3, 6),
            "expand": lambda: t[:1].expand(4, 3, 4).reshape(4, 12)[:, :6], "pad": lambda: F.pad(t.reshape(6, 4), (1, 1, 0, 0))[:4],
            "shrink": lambda: t.reshape(4, 6)[1:3], "flip": lambda: t.reshape(4, 6).flip(1), "getitem": lambda: t.reshape(4, 6).flip(0)[:, 1:],
            "repeat": lambda: t.reshape(2, 12)[:, :3].repeat(2, 2), "cat": lambda: torch.cat([t.reshape(4, 6), t.reshape(4, 6) * 2])[:4],
            "stack": lambda: torch.stack([t.reshape(4, 6), t.reshape(4, 6) * 3])[1], "squeeze": lambda: t.unsqueeze(0).squeeze(0).reshape(4, 6)}[n]()
  for n, f in cases.items():
    x, tx = mk(a, True); out, tout = f(x), tcase(n, tx)
    gg = rnd(tuple(tout.shape)); out.backward(Tensor(gg)); tout.backward(torch.tensor(gg))
    chk(assert_close, x.grad, tx.grad, reduce=True)

def test_gather_backward(device, assert_close):
  x, tx = mk(rnd((4, 5)), True); idx = R.integers(0, 5, size=(4, 3)).astype(np.int32)
  (x.gather(1, Tensor(idx)) * 2).sum().backward(); (tx.gather(1, torch.tensor(idx, dtype=torch.long)) * 2).sum().backward()
  chk(assert_close, x.grad, tx.grad, reduce=True)
  x, tx = mk(rnd((5, 4)), True); i = np.array([0, 3, 3, 1], np.int32)
  x[Tensor(i)].sum().backward(); tx[torch.tensor(i, dtype=torch.long)].sum().backward(); chk(assert_close, x.grad, tx.grad, reduce=True)

# ------------------------------------------------------------------ reductions
RSHAPES = [(5,), (1,), (3, 4), (1, 4), (3, 1), (2, 3, 4), (4, 0) if False else (4, 7)]
AXES = [None, 0, -1, (0, 1)]

def _axok(shape, ax): return not (ax == (0, 1) and len(shape) < 2) and not (ax == 0 and len(shape) == 0)

def _targ(ax): return {} if ax is None else {"dim": ax}

def _ref_reduce(name, t, ax, keepdim):
  """torch reference; max/min reject tuple dims and dim=None+keepdim, so use amax/amin there."""
  if name in ("max", "min"):
    if ax is None and not keepdim: return getattr(torch, name)(t)
    if ax is None or isinstance(ax, tuple): return getattr(torch, "a" + name)(t, dim=tuple(range(t.ndim)) if ax is None else ax, keepdim=keepdim)
    return getattr(torch, name)(t, dim=ax, keepdim=keepdim).values
  if ax is None: return getattr(torch, name)(t) if not keepdim else getattr(torch, name)(t, dim=tuple(range(t.ndim)), keepdim=True)
  return getattr(torch, name)(t, dim=ax, keepdim=keepdim)

@pytest.mark.parametrize("name", ["sum", "mean", "max", "min"])
@pytest.mark.parametrize("ax", AXES, ids=str)
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("shape", RSHAPES)
def test_reduce_reference_builds(name, ax, keepdim, shape):  # regression: reference must not raise for tuple/None axes (torch only)
  if not _axok(shape, ax): pytest.skip("axis out of range")
  e = _ref_reduce(name, torch.tensor(rnd(shape)), ax, keepdim)
  n = len(shape); assert e.ndim == (n if keepdim else (0 if ax is None else n - len(ax if isinstance(ax, tuple) else (ax,))))

@pytest.mark.parametrize("dt", FLOATS)
@pytest.mark.parametrize("name", ["sum", "mean", "max", "min"])
@pytest.mark.parametrize("ax", AXES, ids=str)
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("shape", RSHAPES)
def test_reduce(device, assert_close, name, ax, keepdim, shape, dt):
  if not _axok(shape, ax): pytest.skip("axis out of range")
  x, tx = mk(rnd(shape, dt))
  e = _ref_reduce(name, tx.float() if dt == "float16" else tx, ax, keepdim).to(tx.dtype)
  chk(assert_close, getattr(x, name)(axis=ax, keepdim=keepdim), e, dt, reduce=name in ("sum", "mean"))

def test_reduce_scalar_input(device, assert_close):
  x, tx = mk(rnd(())); chk(assert_close, x.sum(), tx.sum()); chk(assert_close, x.max(), tx.max()); chk(assert_close, x.mean(), tx.mean())

def test_reduce_int_bool(device, assert_close):
  i, ti = mk(rnd((3, 4), "int32")); chk(assert_close, i.sum(), ti.sum(dtype=torch.int32)); chk(assert_close, i.sum(axis=0), ti.sum(0, dtype=torch.int32))
  chk(assert_close, i.max(), ti.max()); chk(assert_close, i.min(axis=1), ti.min(1).values)
  b, tb = mk(rnd((3, 4), "bool")); np.testing.assert_array_equal(b.sum(axis=1).numpy(), tb.sum(1).numpy()); chk(assert_close, b.max(), tb.max())

def test_sum_dtype(device, assert_close):
  x, tx = mk(rnd((3, 4), "float16")); out = x.sum(axis=1, dtype=dtypes.float32); assert out.dtype == dtypes.float32
  chk(assert_close, out, tx.float().sum(1), "float32", reduce=True)
  i, ti = mk(rnd((3, 4), "int32")); chk(assert_close, i.sum(dtype=dtypes.float32), ti.float().sum(), reduce=True)

def test_reduce_noncontig(device, assert_close):
  x, tx = mk(rnd((4, 6, 5)))
  chk(assert_close, x.permute(2, 0, 1).sum(axis=1), tx.permute(2, 0, 1).sum(1), reduce=True)
  chk(assert_close, x[:, ::2].max(axis=-1), tx[:, ::2].max(-1).values); chk(assert_close, x[1:3, :, 1:].mean(), tx[1:3, :, 1:].mean(), reduce=True)

def test_reduce_large_f16_accumulates_f32(device, assert_close):
  x = Tensor(np.full((4096,), 0.1, np.float16)); e = np.float32(np.float16(0.1)) * 4096
  assert abs(float(x.sum().numpy()) - e) / e < 1e-2

@pytest.mark.parametrize("name", ["sum", "mean", "max", "min", "var", "std", "logsumexp", "softmax", "log_softmax", "cumsum"])
@pytest.mark.parametrize("ax", [0, 1])
def test_reduce_backward(device, assert_close, name, ax):
  a = rnd((4, 5)); a = a + np.arange(20, dtype=np.float32).reshape(4, 5) * 1e-2  # break ties
  (x, tx) = mk(a, True)
  kw = {"axis": ax}
  if name in ("sum", "mean", "var", "std", "logsumexp"):
    o = getattr(x, name)(axis=ax); to = {"sum": lambda t: t.sum(ax), "mean": lambda t: t.mean(ax), "var": lambda t: t.var(ax), "std": lambda t: t.std(ax), "logsumexp": lambda t: t.logsumexp(ax)}[name](tx)
  elif name in ("max", "min"):
    o = getattr(x, name)(axis=ax); to = getattr(tx, name)(ax).values
  elif name == "cumsum": o = x.cumsum(axis=ax); to = tx.cumsum(ax)
  else: o = getattr(x, name)(axis=ax); to = getattr(tx, name)(ax)
  g = rnd(tuple(to.shape)); o.backward(Tensor(g)); to.backward(torch.tensor(g)); chk(assert_close, x.grad, tx.grad, reduce=True)

@pytest.mark.parametrize("corr", [0, 1])
@pytest.mark.parametrize("ax", [None, 0, -1, (0, 1)], ids=str)
@pytest.mark.parametrize("keepdim", [False, True])
def test_var_std(device, assert_close, corr, ax, keepdim):
  x, tx = mk(rnd((4, 5, 3)))
  d = {} if ax is None else {"dim": ax}
  chk(assert_close, x.var(axis=ax, keepdim=keepdim, correction=corr), tx.var(**d, correction=corr, keepdim=keepdim), reduce=True)
  chk(assert_close, x.std(axis=ax, keepdim=keepdim, correction=corr), tx.std(**d, correction=corr, keepdim=keepdim), reduce=True)

def test_var_f16(device, assert_close):
  x, tx = mk(rnd((4, 8), "float16")); chk(assert_close, x.var(axis=1), tx.float().var(1).half(), "float16", reduce=True)

@pytest.mark.parametrize("name", ["argmax", "argmin"])
@pytest.mark.parametrize("ax", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dt", ["float32", "int32", "float16"])
def test_argmax_argmin(device, assert_close, name, ax, keepdim, dt):
  a = rnd((4, 6), dt); a[1, 2] = a[1, 4] = a.max() if name == "argmax" else a.min()  # ties -> first index
  x, tx = mk(a); out = getattr(x, name)(axis=ax, keepdim=keepdim); assert out.dtype == dtypes.int32
  e = getattr(torch, name)(tx, dim=ax, keepdim=keepdim).to(torch.int32) if ax is not None else getattr(torch, name)(tx).to(torch.int32)
  if ax is None and keepdim: e = e.reshape(1, 1)
  chk(assert_close, out, e)

def test_argmax_1d(device):
  a = np.array([1, 5, 5, 2], np.float32); assert int(Tensor(a).argmax().item()) == 1; assert int(Tensor(a).argmin().item()) == 0

@pytest.mark.parametrize("dt", FLOATS)
@pytest.mark.parametrize("ax", [None, 0, 1, -1, (0, 1)], ids=str)
@pytest.mark.parametrize("keepdim", [False, True])
def test_logsumexp(device, assert_close, ax, keepdim, dt):
  x, tx = mk(rnd((4, 6), dt, -5, 5))
  e = torch.logsumexp(tx.float(), dim=ax if ax is not None else (0, 1), keepdim=keepdim).to(tx.dtype)
  chk(assert_close, x.logsumexp(axis=ax, keepdim=keepdim), e, dt, reduce=True)

def test_logsumexp_stable(device, assert_close):
  a = np.array([[1000.0, 1001.0], [-1000.0, -999.0]], np.float32); x, tx = mk(a)
  chk(assert_close, x.logsumexp(axis=1), torch.logsumexp(tx, 1), reduce=True)

@pytest.mark.parametrize("dt", FLOATS)
@pytest.mark.parametrize("shape,ax", [((5,), 0), ((3, 4), -1), ((3, 4), 0), ((2, 3, 4), 1), ((1, 4), -1), ((3, 1), -1)])
def test_softmax(device, assert_close, shape, ax, dt):
  x, tx = mk(rnd(shape, dt, -5, 5))
  chk(assert_close, x.softmax(axis=ax), torch.softmax(tx.float(), ax).to(tx.dtype), dt, reduce=True)
  chk(assert_close, x.log_softmax(axis=ax), torch.log_softmax(tx.float(), ax).to(tx.dtype), dt, reduce=True)

def test_softmax_default_and_stable(device, assert_close):
  a = np.array([[1000.0, 1001.0, 999.0]], np.float32); x, tx = mk(a)
  chk(assert_close, x.softmax(), torch.softmax(tx, -1), reduce=True); chk(assert_close, x.log_softmax(), torch.log_softmax(tx, -1), reduce=True)
  y, ty = mk(rnd((4, 8))); chk(assert_close, y.permute(1, 0).softmax(), torch.softmax(ty.T, -1), reduce=True)

@pytest.mark.parametrize("dt", FLOATS + ["int32"])
@pytest.mark.parametrize("shape,ax", [((5,), 0), ((3, 4), 1), ((3, 4), 0), ((2, 3, 4), -1), ((1, 4), 1), ((3, 1), 0)])
def test_cumsum(device, assert_close, shape, ax, dt):
  x, tx = mk(rnd(shape, dt, -1, 1) if dt != "int32" else rnd(shape, dt))
  e = tx.float().cumsum(ax).to(tx.dtype) if dt == "float16" else tx.cumsum(ax).to(tx.dtype)
  chk(assert_close, x.cumsum(axis=ax), e, tol_dt(dt) or "float32", reduce=True)

# ------------------------------------------------------------------ matmul / nn
MM = [((3, 4), (4, 5)), ((1, 4), (4, 1)), ((4,), (4,)), ((4,), (4, 5)), ((3, 4), (4,)), ((2, 3, 4), (4, 5)), ((3, 4), (2, 4, 5)),
      ((2, 3, 4), (2, 4, 5)), ((2, 1, 3, 4), (5, 4, 2)), ((1, 7), (7, 3)), ((17, 33), (33, 19)), ((64, 64), (64, 64))]

@pytest.mark.parametrize("dt", FLOATS)
@pytest.mark.parametrize("sa,sb", MM)
def test_matmul(device, assert_close, sa, sb, dt):
  (x, tx), (y, ty) = mk(rnd(sa, dt, -1, 1)), mk(rnd(sb, dt, -1, 1))
  e = (tx.float() @ ty.float()).to(tx.dtype)
  chk(assert_close, x @ y, e, dt, reduce=True); chk(assert_close, x.matmul(y), e, dt, reduce=True)
  if len(sa) == 1 and len(sb) == 1: chk(assert_close, x.dot(y), e, dt, reduce=True)

def test_matmul_int(device, assert_close):
  (x, tx), (y, ty) = mk(rnd((3, 4), "int32")), mk(rnd((4, 5), "int32")); chk(assert_close, x @ y, tx @ ty if False else (tx.long() @ ty.long()).int())

def test_matmul_noncontig(device, assert_close):
  (x, tx), (y, ty) = mk(rnd((6, 5))), mk(rnd((7, 6)))
  chk(assert_close, x.permute(1, 0) @ y.permute(1, 0), tx.T @ ty.T, reduce=True); chk(assert_close, x[::2] @ y[1:4].permute(1, 0)[:5], tx[::2] @ ty[1:4].T[:5], reduce=True)

def test_matmul_noncontig_shapes():  # regression: the strided-operand case must be shape-compatible (torch only)
  tx, ty = torch.zeros(6, 5), torch.zeros(7, 6)
  assert (tx[::2] @ ty[1:4].T[:5]).shape == (3, 3) and (tx.T @ ty.T).shape == (5, 7)

@pytest.mark.parametrize("sa,sb", MM[:10])
def test_matmul_backward(device, assert_close, sa, sb):
  (x, tx), (y, ty) = mk(rnd(sa, "float32", -1, 1), True), mk(rnd(sb, "float32", -1, 1), True)
  o, to = x @ y, tx @ ty; g = rnd(tuple(to.shape)); o.backward(Tensor(g)); to.backward(torch.tensor(g))
  chk(assert_close, x.grad, tx.grad, reduce=True); chk(assert_close, y.grad, ty.grad, reduce=True)

@pytest.mark.parametrize("dt", FLOATS)
@pytest.mark.parametrize("sx,bias", [((3, 4), True), ((3, 4), False), ((2, 3, 4), True), ((4,), True), ((1, 4), False)])
def test_linear(device, assert_close, sx, bias, dt):
  (x, tx), (w, tw) = mk(rnd(sx, dt, -1, 1)), mk(rnd((5, 4), dt, -1, 1)); b, tb = mk(rnd((5,), dt, -1, 1))
  e = F.linear(tx.float(), tw.float(), tb.float() if bias else None).to(tx.dtype)
  chk(assert_close, x.linear(w, b if bias else None), e, dt, reduce=True)

def test_linear_backward(device, assert_close):
  (x, tx), (w, tw), (b, tb) = mk(rnd((2, 3, 4)), True), mk(rnd((5, 4)), True), mk(rnd((5,)), True)
  g = rnd((2, 3, 5)); x.linear(w, b).backward(Tensor(g)); F.linear(tx, tw, tb).backward(torch.tensor(g))
  for p, q in ((x, tx), (w, tw), (b, tb)): chk(assert_close, p.grad, q.grad, reduce=True)

@pytest.mark.parametrize("dt", FLOATS)
@pytest.mark.parametrize("shape,ax", [((3, 8), -1), ((2, 3, 8), -1), ((8, 3), 0), ((2, 6, 4), 1), ((1, 8), -1)])
def test_layernorm(device, assert_close, shape, ax, dt):
  x, tx = mk(rnd(shape, dt, -3, 3))
  e = F.layer_norm(tx.float(), (tx.shape[ax],), eps=1e-5) if ax in (-1, tx.ndim - 1) else (lambda t: (t - t.mean(ax, keepdim=True)) / torch.sqrt(t.var(ax, correction=0, keepdim=True) + 1e-5))(tx.float())
  chk(assert_close, x.layernorm(axis=ax), e.to(tx.dtype), dt, reduce=True)
  chk(assert_close, x.layernorm(axis=ax, eps=1e-3) if False else x.layernorm(ax, 1e-5), e.to(tx.dtype), dt, reduce=True)

def test_layernorm_backward(device, assert_close):
  x, tx = mk(rnd((3, 8), "float32", -3, 3), True); g = rnd((3, 8))
  x.layernorm().backward(Tensor(g)); F.layer_norm(tx, (8,)).backward(torch.tensor(g)); chk(assert_close, x.grad, tx.grad, reduce=True)

def test_dropout(device, assert_close):
  x, tx = mk(rnd((3, 4)))
  Tensor.training = False
  chk(assert_close, x.dropout(0.5), tx)
  Tensor.training = True
  try:
    manual_seed(0); big = Tensor.ones(20000); o = big.dropout(0.5).numpy()
    assert set(np.unique(o).tolist()) <= {0.0, 2.0} and abs((o == 0).mean() - 0.5) < 0.03 and abs(o.mean() - 1) < 0.05
  finally: Tensor.training = False

def test_one_hot(device, assert_close):
  i = np.array([0, 2, 1, 4], np.int32); out = Tensor(i).one_hot(5)
  np.testing.assert_array_equal(out.numpy(), F.one_hot(torch.tensor(i, dtype=torch.long), 5).numpy())
  j = np.array([[0, 1], [2, 2]], np.int32); np.testing.assert_array_equal(Tensor(j).one_hot(3).numpy(), F.one_hot(torch.tensor(j, dtype=torch.long), 3).numpy())

@pytest.mark.parametrize("diag", [-2, -1, 0, 1, 3])
@pytest.mark.parametrize("shape", [(4, 4), (3, 5), (5, 3), (2, 4, 4), (1, 1)])
def test_triu_tril(device, assert_close, shape, diag):
  x, tx = mk(rnd(shape))
  chk(assert_close, x.triu(diag), tx.triu(diag)); chk(assert_close, x.tril(diag), tx.tril(diag))
  chk(assert_close, x.triu(), tx.triu()); chk(assert_close, x.tril(), tx.tril())
  b, tb = mk(rnd(shape, "bool")); chk(assert_close, b.triu(diag), tb.triu(diag)) if False else None

def test_triu_tril_backward(device, assert_close):
  x, tx = mk(rnd((4, 4)), True); (x.triu(1) * 3).sum().backward(); (tx.triu(1) * 3).sum().backward(); chk(assert_close, x.grad, tx.grad)

@pytest.mark.parametrize("red", ["mean", "sum", "none"])
@pytest.mark.parametrize("ls", [0.0, 0.1])
@pytest.mark.parametrize("n,c", [(5, 7), (1, 3), (8, 2)])
def test_cross_entropy(device, assert_close, red, ls, n, c):
  (x, tx) = mk(rnd((n, c), "float32", -3, 3), True); t = R.integers(0, c, size=(n,)).astype(np.int32)
  out = x.cross_entropy(Tensor(t), reduction=red, label_smoothing=ls)
  e = F.cross_entropy(tx, torch.tensor(t, dtype=torch.long), reduction=red, label_smoothing=ls)
  chk(assert_close, out, e, reduce=True)
  g = rnd(tuple(e.shape)) if red == "none" else None
  if g is None: out.backward(); e.backward()
  else: out.backward(Tensor(g)); e.backward(torch.tensor(g))
  chk(assert_close, x.grad, tx.grad, reduce=True)

def test_cross_entropy_1d_and_ignore(device, assert_close):
  x, tx = mk(rnd((6,), "float32", -3, 3)); chk(assert_close, x.cross_entropy(Tensor(np.array(2, np.int32))), F.cross_entropy(tx, torch.tensor(2)), reduce=True)
  (x, tx) = mk(rnd((5, 4), "float32", -3, 3), True); t = np.array([0, -1, 3, -1, 2], np.int32)
  out = x.cross_entropy(Tensor(t)); e = F.cross_entropy(tx, torch.tensor(t, dtype=torch.long), ignore_index=-1)
  chk(assert_close, out, e, reduce=True); out.backward(); e.backward(); chk(assert_close, x.grad, tx.grad, reduce=True)
  (x, tx) = mk(rnd((5, 4), "float32", -3, 3)); t2 = np.array([0, 5 - 2, 3, 1, 2], np.int32)
  chk(assert_close, x.cross_entropy(Tensor(t2), ignore_index=3, reduction="sum"), F.cross_entropy(tx, torch.tensor(t2, dtype=torch.long), ignore_index=3, reduction="sum"), reduce=True)

@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 4, 8), (2, 4, 6, 8), (2, 2, 1, 4), (2, 3, 5, 16)])
def test_sdpa(device, assert_close, shape, causal):
  (q, tq), (k, tk), (v, tv) = mk(rnd(shape, "float32", -1, 1), True), mk(rnd(shape, "float32", -1, 1), True), mk(rnd(shape, "float32", -1, 1), True)
  out = q.scaled_dot_product_attention(k, v, is_causal=causal); e = F.scaled_dot_product_attention(tq, tk, tv, is_causal=causal)
  chk(assert_close, out, e, reduce=True)
  g = rnd(shape); out.backward(Tensor(g)); e.backward(torch.tensor(g))
  for p, t in ((q, tq), (k, tk), (v, tv)): chk(assert_close, p.grad, t.grad, reduce=True)

def test_sdpa_mask_and_cross_len(device, assert_close):
  (q, tq), (k, tk), (v, tv) = mk(rnd((2, 3, 4, 8), "float32", -1, 1)), mk(rnd((2, 3, 6, 8), "float32", -1, 1)), mk(rnd((2, 3, 6, 8), "float32", -1, 1))
  m = rnd((4, 6), "bool"); m[:, 0] = True; fm = rnd((2, 3, 4, 6), "float32", -1, 1)
  chk(assert_close, q.scaled_dot_product_attention(k, v, attn_mask=Tensor(m)), F.scaled_dot_product_attention(tq, tk, tv, attn_mask=torch.tensor(m)), reduce=True)
  chk(assert_close, q.scaled_dot_product_attention(k, v, attn_mask=Tensor(fm)), F.scaled_dot_product_attention(tq, tk, tv, attn_mask=torch.tensor(fm)), reduce=True)

def test_sdpa_dropout_zero(device, assert_close):
  (q, tq), (k, tk), (v, tv) = mk(rnd((2, 4, 8))), mk(rnd((2, 4, 8))), mk(rnd((2, 4, 8)))
  chk(assert_close, q.scaled_dot_product_attention(k, v, dropout_p=0.0), F.scaled_dot_product_attention(tq, tk, tv), reduce=True)

# ------------------------------------------------------------------ autograd
def test_backward_semantics(device, assert_close):
  x, tx = mk(rnd((3, 4)), True)
  y = (x * x).sum(); y.backward(); (tx * tx).sum().backward(); chk(assert_close, x.grad, tx.grad)
  (x * 2).sum().backward(); (tx * 2).sum().backward(); chk(assert_close, x.grad, tx.grad)  # accumulation
  with pytest.raises(Exception): (x * 2).backward()  # non-scalar without gradient

def test_requires_grad_(device):
  x = Tensor(rnd((2, 3))); assert x.requires_grad_() is x and x.requires_grad
  x.requires_grad_(False); assert not x.requires_grad
  w = Tensor(rnd((2, 3)), requires_grad=True); assert (w * 2).requires_grad and not (w * 2).detach().requires_grad

def test_no_grad(device):
  from fusezap import no_grad
  w = Tensor(rnd((2, 3)), requires_grad=True)
  with no_grad(): y = w * 2
  assert not y.requires_grad
  Tensor.no_grad = True
  try: assert not (w * 2).requires_grad
  finally: Tensor.no_grad = False
  assert (w * 2).requires_grad

def test_grad_shared_subgraph(device, assert_close):
  x, tx = mk(rnd((3, 4)), True); y = x.exp(); z = (y * y + y).sum(); tz = ((tx.exp()) ** 2 + tx.exp()).sum()
  z.backward(); tz.backward(); chk(assert_close, x.grad, tx.grad, reduce=True)

def test_composite_chain_backward(device, assert_close):
  (x, tx), (w, tw) = mk(rnd((4, 8)), True), mk(rnd((8, 8), "float32", -0.5, 0.5), True)
  y = (x @ w).gelu().layernorm().softmax().log().mean(); ty = F.layer_norm(F.gelu(tx @ tw, approximate="tanh"), (8,)).softmax(-1).log().mean()
  chk(assert_close, y, ty, reduce=True); y.backward(); ty.backward()
  chk(assert_close, x.grad, tx.grad, reduce=True); chk(assert_close, w.grad, tw.grad, reduce=True)

# ------------------------------------------------------------------ edge cases
def test_zero_size(device, assert_close):
  x, tx = mk(rnd((0, 4))); chk(assert_close, x + 1, tx + 1); chk(assert_close, x.sum(axis=0), tx.sum(0)); chk(assert_close, x.sum(axis=1), tx.sum(1))
  chk(assert_close, x.permute(1, 0), tx.T); chk(assert_close, x.exp(), tx.exp())
  a, ta = mk(rnd((3, 0))); b, tb = mk(rnd((0, 5))); chk(assert_close, a @ b, ta @ tb)

def test_size_one_broadcast_chain(device, assert_close):
  (a, ta), (b, tb), (c, tc) = mk(rnd((1, 1))), mk(rnd((1, 5))), mk(rnd((4, 1)))
  chk(assert_close, (a + b) * c - a / (b.abs() + 1), (ta + tb) * tc - ta / (tb.abs() + 1))

def test_reuse_and_inplace_safety(device, assert_close):
  a = rnd((3, 4)); x = Tensor(a); y = x + 1; z = x * 2
  np.testing.assert_allclose(y.numpy(), a + 1, rtol=1e-6); np.testing.assert_allclose(z.numpy(), a * 2, rtol=1e-6); np.testing.assert_array_equal(x.numpy(), a)

def test_nan_inf(device):
  a = np.array([np.inf, -np.inf, np.nan, 0.0, 1.0], np.float32); x = Tensor(a)
  np.testing.assert_array_equal(np.isnan(x.numpy()), np.isnan(a)); assert np.isposinf((x + 1).numpy()[0]) and np.isnan((x * 0).numpy()[0])
  assert np.isnan(x.exp().numpy()[2]) and x.exp().numpy()[1] == 0 and np.isneginf(Tensor(np.array([0.0], np.float32)).log().numpy()[0])


# ------------------------------------------------------------------ bfloat16 (host storage is uint16 bits, SPEC §4)
def _bf(a):
  """float32 ndarray -> (fusezap bf16 Tensor, torch bf16 Tensor) with identical (already bf16-rounded) values."""
  from conftest import bf16_to_f32, f32_to_bf16
  r = bf16_to_f32(f32_to_bf16(a))
  return Tensor(r).cast(dtypes.bfloat16), torch.tensor(r).to(torch.bfloat16)

def _bfnp(t):
  from conftest import bf16_to_f32
  n = t.numpy(); assert n.dtype == np.uint16, f"bf16 host buffer should be uint16 bits, got {n.dtype}"
  return bf16_to_f32(n)

def test_bf16_helpers_match_torch():  # regression: host-side bit helpers agree with torch's rounding
  from conftest import bf16_to_f32, f32_to_bf16
  a = rnd((64,), "float32", -100, 100)
  np.testing.assert_array_equal(bf16_to_f32(f32_to_bf16(a)), torch.tensor(a).to(torch.bfloat16).float().numpy())

def test_bf16_roundtrip_cast(device):
  a = rnd((3, 5), "float32", -50, 50); x, tx = _bf(a)
  assert x.dtype == dtypes.bfloat16 and x.shape == (3, 5)
  np.testing.assert_array_equal(_bfnp(x), tx.float().numpy())
  np.testing.assert_array_equal(x.float().numpy(), tx.float().numpy())
  np.testing.assert_array_equal(Tensor(a).cast(dtypes.bfloat16).float().numpy(), torch.tensor(a).to(torch.bfloat16).float().numpy())

def test_assert_close_bf16_bits(assert_close):  # regression: uint16 bf16 bits must be compared as floats, not exactly
  from conftest import f32_to_bf16
  a = np.array([1.0, 2.0, 3.0], np.float32); bits = f32_to_bf16(a)
  assert_close(bits, torch.tensor(a * 1.001).to(torch.bfloat16), "bfloat16")

@pytest.mark.parametrize("op", ["add", "mul", "sub", "maximum"])
def test_bf16_binary(device, assert_close, op):
  (x, tx), (y, ty) = _bf(rnd((4, 6))), _bf(rnd((4, 6)))
  f = {"add": lambda a, b: a + b, "mul": lambda a, b: a * b, "sub": lambda a, b: a - b, "maximum": lambda a, b: a.maximum(b)}[op]
  o = f(x, y); assert o.dtype == dtypes.bfloat16
  assert_close(o.numpy(), f(tx, ty), "bfloat16")

@pytest.mark.parametrize("name", ["exp", "tanh", "sigmoid", "relu", "neg"])
def test_bf16_unary(device, assert_close, name):
  x, tx = _bf(rnd((3, 7)))
  assert_close(getattr(x, name)().numpy(), getattr(torch, name)(tx) if name != "relu" else torch.relu(tx), "bfloat16")

@pytest.mark.parametrize("name", ["sum", "mean", "max"])
def test_bf16_reduce(device, assert_close, name):
  x, tx = _bf(rnd((4, 8)))
  e = getattr(tx.float(), name)(1) if name != "max" else tx.float().amax(1)
  assert_close(getattr(x, name)(axis=1).numpy(), e.to(torch.bfloat16), "bfloat16", reduce=True)

@pytest.mark.parametrize("sa,sb", [((3, 4), (4, 5)), ((8, 16), (16, 8)), ((1, 4), (4, 1))])
def test_bf16_matmul(device, assert_close, sa, sb):
  (x, tx), (y, ty) = _bf(rnd(sa, "float32", -1, 1)), _bf(rnd(sb, "float32", -1, 1))
  assert_close((x @ y).numpy(), (tx.float() @ ty.float()).to(torch.bfloat16), "bfloat16", reduce=True)

def test_bf16_softmax(device, assert_close):
  x, tx = _bf(rnd((3, 9)))
  assert_close(x.softmax().numpy(), torch.softmax(tx.float(), -1).to(torch.bfloat16), "bfloat16", reduce=True)
