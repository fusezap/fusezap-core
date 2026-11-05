"""WP1: View / ShapeTracker (SPEC §5.2). Random movement-op sequences must match numpy on an arange buffer."""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as hst
from fusezap.helpers import prod
from fusezap.shape.symbolic import Node, Var, MulNode, sym_infer
from fusezap.shape.view import View, ShapeTracker

# ---------------------------------------------------------------- numpy reference

def np_apply(ref: np.ndarray, op: str, arg) -> np.ndarray:
  if op == "reshape": return ref.reshape(arg)
  if op == "permute": return ref.transpose(arg)
  if op == "expand": return np.broadcast_to(ref, arg)
  if op == "pad": return np.pad(ref, arg, constant_values=-1)
  if op == "shrink": return ref[tuple(slice(b, e) for b, e in arg)]
  if op == "flip": return np.flip(ref, tuple(i for i, f in enumerate(arg) if f)) if any(arg) else ref
  raise ValueError(op)

def check_st(st: ShapeTracker, ref: np.ndarray, buf_size: int, var_vals: dict | None = None):
  """Every logical element: valid iff ref != -1, and then the buffer index equals ref (values of arange)."""
  var_vals = var_vals or {}
  shape = tuple(sym_infer(s, var_vals) for s in st.shape)
  assert shape == ref.shape, f"{shape} != {ref.shape}"
  n = prod(shape)
  want_valid = (ref.reshape(-1) != -1).astype(np.int64)
  gidx = Var("gidx", 0, max(n - 1, 0))
  idx, valid = st.expr_node(gidx)
  vals = {**var_vals, gidx: np.arange(n)}
  got_idx, got_valid = (np.broadcast_to(np.asarray(sym_infer(x, vals)), (n,)) for x in (idx, valid))
  assert np.array_equal(got_valid, want_valid), f"valid mismatch\n{st}\n{got_valid}\n{want_valid}"
  m = want_valid == 1
  assert np.array_equal(got_idx[m], ref.reshape(-1)[m]), f"index mismatch\n{st}"
  # expr_idxs with one Var per axis agrees
  if n:
    ivs = [Var(f"i{k}", 0, s - 1) for k, s in enumerate(shape)]
    idx2, valid2 = st.expr_idxs(ivs)
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    vals2 = {**var_vals, **dict(zip(ivs, grids))}
    g2i, g2v = (np.broadcast_to(np.asarray(sym_infer(x, vals2)), shape).reshape(-1) for x in (idx2, valid2))
    assert np.array_equal(g2v, want_valid) and np.array_equal(g2i[m], got_idx[m])
    # buffer accesses are in range and real_size covers them
    if not st.vars():
      rs = st.real_size()
      assert rs <= buf_size and (not m.any() or got_idx[m].max() < rs) and (not m.any() or got_idx[m].min() >= 0)
  # a single unmasked view is exactly numpy's as_strided over the buffer
  if len(st.views) == 1 and st.views[0].mask is None and n and not st.vars():
    v, buf = st.views[0], np.arange(buf_size, dtype=np.int64)
    strided = np.lib.stride_tricks.as_strided(buf[v.offset:], v.shape, tuple(s * 8 for s in v.strides))
    assert np.array_equal(strided, ref)

# ---------------------------------------------------------------- random op sequences

def divisors(n): return [d for d in range(1, n + 1) if n % d == 0]

def random_reshape(draw, shape):
  n = prod(shape)
  kind = draw(hst.integers(0, 3))
  if kind == 0 and len(shape) > 1:   # merge two adjacent dims
    k = draw(hst.integers(0, len(shape) - 2)); return shape[:k] + (shape[k] * shape[k + 1],) + shape[k + 2:]
  if kind == 1:                      # split one dim
    k = draw(hst.integers(0, len(shape) - 1)) if shape else 0
    if not shape: return (1,)
    d = draw(hst.sampled_from(divisors(shape[k]))); return shape[:k] + (d, shape[k] // d) + shape[k + 1:]
  if kind == 2:                      # insert / drop a 1
    if 1 in shape and draw(hst.booleans()): k = shape.index(1); return shape[:k] + shape[k + 1:]
    k = draw(hst.integers(0, len(shape))); return shape[:k] + (1,) + shape[k:]
  out, rem = [], n                   # arbitrary refactorization
  while rem > 1 and len(out) < 4:
    d = draw(hst.sampled_from(divisors(rem))); out.append(d); rem //= d
  if rem > 1: out.append(rem)
  return tuple(draw(hst.permutations(out))) if out else ()

def random_op(draw, shape):
  ops = ["reshape", "permute", "expand", "pad", "shrink", "flip"]
  op = draw(hst.sampled_from(ops))
  nd = len(shape)
  if op == "reshape": return op, random_reshape(draw, shape)
  if nd == 0: return "reshape", (1,)
  if op == "permute": return op, tuple(draw(hst.permutations(range(nd))))
  if op == "expand":
    return op, tuple(draw(hst.integers(1, 3)) if s == 1 else s for s in shape)
  if op == "pad": return op, tuple((draw(hst.integers(0, 2)), draw(hst.integers(0, 2))) for _ in shape)
  if op == "shrink":
    arg = []
    for s in shape:
      b = draw(hst.integers(0, max(s - 1, 0))); arg.append((b, draw(hst.integers(b + (1 if s else 0), s))))
    return op, tuple(arg)
  return op, tuple(draw(hst.booleans()) for _ in shape)

@hst.composite
def op_sequences(draw):
  shape = tuple(draw(hst.lists(hst.integers(1, 5), min_size=1, max_size=3)))
  ops, cur = [], shape
  for _ in range(draw(hst.integers(1, 8))):
    op, arg = random_op(draw, cur)
    ops.append((op, arg))
    cur = np_apply(np.empty(cur, dtype=np.int8), op, arg).shape if op != "expand" else arg
    if prod(cur) > 4000: break
  return shape, ops

@settings(max_examples=600, deadline=None)
@given(op_sequences())
def test_random_movement_ops(seq):
  shape, ops = seq
  n = prod(shape)
  st, ref = ShapeTracker.from_shape(shape), np.arange(n, dtype=np.int64).reshape(shape)
  check_st(st, ref, n)
  for op, arg in ops:
    st, ref = getattr(st, op)(arg), np_apply(ref, op, arg)
    check_st(st, ref, n)
  # simplify is idempotent and preserves semantics
  check_st(st.simplify(), ref, n)

# ---------------------------------------------------------------- symbolic shapes + substitution

N = Var("n", 2, 6)

@hst.composite
def sym_sequences(draw, N=N):
  shape = (N,) + tuple(draw(hst.lists(hst.integers(1, 4), min_size=0, max_size=2)))
  ops, cur = [], shape
  for _ in range(draw(hst.integers(1, 6))):
    nd = len(cur)
    op = draw(hst.sampled_from(["reshape", "permute", "expand", "pad", "shrink", "flip"]))
    if op == "permute": arg = tuple(draw(hst.permutations(range(nd))))
    elif op == "flip": arg = tuple(draw(hst.booleans()) for _ in cur)
    elif op == "expand": arg = tuple(draw(hst.integers(1, 3)) if isinstance(s, int) and s == 1 else s for s in cur)
    elif op == "pad": arg = tuple((draw(hst.integers(0, 1)), draw(hst.integers(0, 1))) for _ in cur)
    elif op == "shrink":   # symbolic end points on symbolic dims, int ranges elsewhere
      arg = []
      for s in cur:
        if isinstance(s, Node): arg.append((b := draw(hst.integers(0, 1)) if s.vmin >= 2 else 0, s))
        else: b = draw(hst.integers(0, s - 1)); arg.append((b, draw(hst.integers(b + 1, s))))
      arg = tuple(arg)
    else:
      kind = draw(hst.integers(0, 2))
      if kind == 0 and nd > 1:
        k = draw(hst.integers(0, nd - 2))
        if isinstance(cur[k], Node) and isinstance(cur[k + 1], Node): continue
        arg = cur[:k] + (cur[k] * cur[k + 1],) + cur[k + 2:]
      elif kind == 1 and any(isinstance(s, MulNode) and isinstance(s.b, int) for s in cur):
        k = next(i for i, s in enumerate(cur) if isinstance(s, MulNode) and isinstance(s.b, int))
        arg = cur[:k] + (cur[k].a, cur[k].b) + cur[k + 1:]
      else:
        k = draw(hst.integers(0, nd)); arg = cur[:k] + (1,) + cur[k:]
    ops.append((op, arg))
    cur = getattr(ShapeTracker.from_shape(cur), op)(arg).shape
  return shape, ops

def concrete(arg, vv):
  if isinstance(arg, tuple): return tuple(concrete(a, vv) for a in arg)
  return sym_infer(arg, vv) if isinstance(arg, (int, Node)) and not isinstance(arg, bool) else arg

@settings(max_examples=300, deadline=None)
@given(sym_sequences())
def test_symbolic_movement_ops(seq):
  shape, ops = seq
  st = ShapeTracker.from_shape(shape)
  for op, arg in ops: st = getattr(st, op)(arg)
  assert N in st.vars() or all(isinstance(s, int) for s in st.shape)
  for nv in range(N.vmin, N.vmax + 1):
    vv = {N: nv}
    cshape = concrete(shape, vv)
    ref = np.arange(prod(cshape), dtype=np.int64).reshape(cshape)
    cst = ShapeTracker.from_shape(cshape)
    for op, arg in ops: ref, cst = np_apply(ref, op, concrete(arg, vv)), getattr(cst, op)(concrete(arg, vv))
    check_st(cst, ref, prod(cshape))
    check_st(st, ref, prod(cshape), vv)            # symbolic tracker evaluated with n bound
    check_st(st.subs(vv), ref, prod(cshape))      # symbolic substitution gives a concrete tracker
    assert sym_infer(st.real_size() if not st.vars() else 0, vv) >= 0

# a dim that may be 0 (vmin 0): index math must not assert or divide by 0 (regressions for the review of WP1)
P = Var("p", 0, 5)

@settings(max_examples=200, deadline=None)
@given(sym_sequences(P))
def test_symbolic_movement_ops_zero_dim(seq):
  shape, ops = seq
  st = ShapeTracker.from_shape(shape)
  for op, arg in ops: st = getattr(st, op)(arg)
  for pv in range(P.vmin, P.vmax + 1):
    vv = {P: pv}
    cshape = concrete(shape, vv)
    ref = np.arange(prod(cshape), dtype=np.int64).reshape(cshape)
    for op, arg in ops: ref = np_apply(ref, op, concrete(arg, vv))
    check_st(st, ref, prod(cshape), vv)
    check_st(st.subs(vv), ref, prod(cshape))

def _c_eval(x, env):  # evaluate rendered source with C truncating division; ZeroDivisionError on a 0 divisor
  if isinstance(x, int): return x
  src = x.render().replace("&&", " and ")
  return int(eval(src.replace("/", "//"), {}, {v.name: env[v] for v in x.vars()}))

def _check_zero_dim(st: ShapeTracker, ref_fn, var: Var, flat: bool):
  for pv in range(var.vmin, var.vmax + 1):
    ref = ref_fn(pv)
    check_st(st, ref, ref.size if pv else 0, {var: pv})
  # rendered index math never divides by 0, even when the dim is 0 (the divisor is max(p, 1))
  g = Var("g", 0, 31)
  idx, valid = st.expr_node(g) if flat else st.expr_idxs([Var(f"i{k}", 0, 7) for k in range(len(st.shape))])
  for pv in range(var.vmin, var.vmax + 1):
    env = {var: pv, g: 3, **{Var(f"i{k}", 0, 7): 1 for k in range(len(st.shape))}}
    _c_eval(idx, env), _c_eval(valid, env)

def test_zero_dim_inner_symbolic():
  p = Var("p", 0, 8)
  st = ShapeTracker.from_shape((4, p))
  idx, valid = st.expr_node(Var("g", 0, 31))
  _check_zero_dim(st, lambda pv: np.arange(4 * pv).reshape(4, pv), p, True)

def test_zero_dim_kv_shrink_transpose_flatten():
  p = Var("p", 0, 8)
  st = ShapeTracker.from_shape((8, 4)).shrink(((0, p), (0, 4))).permute((1, 0)).reshape((4 * p,))
  assert len(st.views) == 2
  st.expr_idxs([Var("i", 0, 31)])
  _check_zero_dim(st, lambda pv: np.arange(32).reshape(8, 4)[:pv].T.reshape(-1), p, True)

def test_zero_dim_from_shrink_offset():
  n = Var("n", 1, 5)
  st = ShapeTracker.from_shape((n, 3)).shrink(((1, n), (0, 3)))
  assert st.shape[0].vmin == 0
  st = st.permute((1, 0)).reshape((3 * st.shape[0],))
  _check_zero_dim(st, lambda nv: np.arange(nv * 3).reshape(nv, 3)[1:].T.reshape(-1), n, True)
  st2 = ShapeTracker.from_shape((2, n, 3)).shrink(((0, 2), (1, n), (0, 3))).permute((2, 0, 1))
  _check_zero_dim(st2.reshape((3, 2 * (n - 1))), lambda nv: np.arange(2 * nv * 3).reshape(2, nv, 3)[:, 1:].transpose(2, 0, 1).reshape(3, -1), n, False)

# ---------------------------------------------------------------- unit tests

def test_create_canonical():
  v = View.create((3, 1, 4))
  assert v.strides == (4, 0, 1) and v.contiguous and v.mask is None and v.offset == 0
  assert View.create((2, 3), mask=((0, 2), (0, 3))).mask is None
  assert not View.create((2, 3), (1, 2)).contiguous
  e = View.create((2, 3), (3, 1), 5, ((0, 2), (2, 2)))
  assert e.mask == ((0, 0), (0, 0)) and e.strides == (0, 0) and e.offset == 0
  assert hash(View.create((2, 3))) == hash(View.create((2, 3)))

def test_properties():
  st = ShapeTracker.from_shape((4, 5))
  assert st.contiguous and st.size == 20 and st.real_size() == 20 and st.unit_stride_axes() == [1]
  assert not st.permute((1, 0)).contiguous and st.permute((1, 0)).unit_stride_axes() == [0]
  assert st.shrink(((1, 3), (0, 5))).real_size() == 15
  assert st.expand((4, 5)).contiguous
  assert ShapeTracker.from_shape((1,)).expand((7,)).real_size() == 1
  assert st.pad(((1, 1), (2, 0))).real_size() == 20
  assert ShapeTracker.from_shape((4, 5)).shrink(((0, 0), (0, 5))).size == 0

def test_reshape_merges():
  st = ShapeTracker.from_shape((4, 6))
  assert len(st.reshape((2, 2, 3, 2)).reshape((24,)).views) == 1
  assert len(st.permute((1, 0)).reshape((3, 2, 4)).views) == 1          # split of a permuted dim
  assert len(st.permute((1, 0)).reshape((24,)).views) == 2              # genuinely needs two views
  assert len(st.expand((4, 6)).reshape((4, 2, 3)).views) == 1
  p = st.pad(((1, 1), (0, 0)))
  assert len(p.reshape((36,)).views) == 1 and len(p.reshape((3, 2, 6)).views) == 2   # rows 1..4 are not a box
  assert len(ShapeTracker.from_shape((2, 6)).pad(((2, 2), (0, 0))).reshape((3, 2, 6)).views) == 1   # rows 2..3 are
  assert len(p.reshape((3, 2, 6)).reshape((6, 6)).views) == 1           # merges back
  e = ShapeTracker.from_shape((3, 1)).expand((3, 4)).reshape((12,))
  assert len(e.views) == 2 and e.reshape((3, 4)).views == ShapeTracker.from_shape((3, 1)).expand((3, 4)).views

def test_view_ops_masks():
  st = ShapeTracker.from_shape((3, 4)).pad(((1, 0), (0, 2))).shrink(((0, 2), (3, 6))).flip((True, False))
  ref = np.flip(np.pad(np.arange(12).reshape(3, 4), ((1, 0), (0, 2)), constant_values=-1)[0:2, 3:6], 0)
  check_st(st, ref, 12)
  assert st.views[-1].mask is not None

def test_expr_concrete_ints():
  st = ShapeTracker.from_shape((3, 4)).permute((1, 0))
  assert st.expr_idxs([2, 1]) == (6, 1)
  p = ShapeTracker.from_shape((3,)).pad(((1, 1),))
  assert p.expr_idxs([0])[1] == 0 and p.expr_idxs([2]) == (1, 1)

def test_symbolic_basic():
  T = Var("T", 1, 32)
  st = ShapeTracker.from_shape((T, 4))
  assert st.contiguous and st.size == T * 4 and st.vars() == {T}
  idx, valid = st.expr_idxs([Var("a", 0, 31), Var("b", 0, 3)])
  assert valid == 1 and idx.substitute({Var("a", 0, 31): 5, Var("b", 0, 3): 2}) == 22
  r = st.reshape((T * 4,)).reshape((T, 2, 2))
  assert len(r.views) == 1 and r.views[0].strides == (4, 2, 1)
  e = ShapeTracker.from_shape((T, 1)).expand((T, 8)).shrink(((0, T), (2, 5)))
  assert e.shape == (T, 3) and e.real_size() == 32
  # pad/permute with symbolic args must not crash
  ShapeTracker.from_shape((T, 4)).pad(((0, T), (1, 1))).permute((1, 0)).expr_node(Var("g", 0, 1000))

def test_flip_negative_strides():
  st = ShapeTracker.from_shape((2, 3)).flip((True, True))
  assert st.views[0].strides == (-3, -1) and st.views[0].offset == 5
  check_st(st, np.arange(6).reshape(2, 3)[::-1, ::-1], 6)

def test_invalid_args():
  st = ShapeTracker.from_shape((2, 3))
  with pytest.raises(AssertionError): st.reshape((4,))
  with pytest.raises(AssertionError): st.permute((0, 0))
  with pytest.raises(AssertionError): st.expand((4, 3))
  with pytest.raises(AssertionError): st.shrink(((1, 0), (0, 3)))
