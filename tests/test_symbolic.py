"""WP1: symbolic integer layer (SPEC §5.1)."""
import itertools
import numpy as np
import pytest
from hypothesis import given, settings, strategies as hst
from fusezap.shape.symbolic import (Node, Var, NumNode, MulNode, DivNode, ModNode, SumNode, LtNode, AndNode, RenderOps,
                                    sym_infer, sym_subs, smax, smin, ands)

X, Y, Z = Var("x", 0, 9), Var("y", 0, 3), Var("z", -4, 5)
VARS = [X, Y, Z]

# ---------------------------------------------------------------- simplification minimums (SPEC §5.1)

def test_const_fold():
  assert NumNode(3) + 4 == 7 and isinstance(NumNode(3) + 4, int)
  assert X * 0 == 0 and X + 0 is X and X * 1 is X
  assert (X + 3) - X == 3 and isinstance((X + 3) - X, int)
  assert Var("c", 5, 5) * 2 == 10

def test_combine_terms():
  assert X * 2 + X * 3 == X * 5
  assert X * 2 + Y + X * -2 == Y
  assert (X * 4 + 1) * 3 == X * 12 + 3
  assert X + Y == Y + X and hash(X + Y) == hash(Y + X)

def test_mul_div():
  assert (X * 4) // 4 == X
  assert (X * 8) // 4 == X * 2
  assert (X * 4) // 8 == X // 2
  assert (X * 4 + Y) // 4 == X       # 0 <= y < 4
  assert (X * 4 + Y + 4) // 4 == X + 1
  assert (X // 2) // 5 == X // 10
  assert X // 10 == 0 and X // 1 is X

def test_mod_by_bounds():
  assert X % 10 is X and Y % 4 is Y
  assert isinstance(X % 5, ModNode)
  assert (X * 4 + Y) % 4 == Y
  assert (X * 6) % 3 == 0
  assert ((X * 12 + Y * 4 + 2) // 4) % 3 == Y % 3
  assert (X % 4) % 2 == X % 2

def test_div_mod_recombine():
  assert (X // 4) * 4 + X % 4 == X
  e = X * 3 + Y
  assert (e // 5) * 10 + (e % 5) * 2 == e * 2

def test_lt():
  assert X.lt(10) == 1 and X.lt(0) == 0 and X.ge(0) == 1
  assert (X * 4).lt(8) == X.lt(2)
  assert (X + 3).lt(5) == X.lt(2)
  assert isinstance(X.lt(5), LtNode) and isinstance(X.ge(3), LtNode)
  assert (X < 5) == X.lt(5)   # SPEC: `<` returns a Node

def test_bounds():
  e = X * 2 - Y
  assert (e.vmin, e.vmax) == (-3, 18)
  assert ((X * -1) // 4).vmin == -3 and ((X * -1) // 4).vmax == 0
  assert (Z % 4).vmin >= 0

def test_negative_numerator_normalized():
  # every int-divisor DivNode/ModNode has a nonnegative numerator, so C truncation == floor
  for e in [(X * -1) // 4, Z // 3, Z % 3, (Z * 2 - X) // 5, (Z - 7) % 4]:
    for n in walk(e):
      if isinstance(n, (DivNode, ModNode)) and isinstance(n.b, int): assert n.a.vmin >= 0, e

def test_vars_substitute_render():
  e = (X * 4 + Y) // 3 + Z % 2
  assert e.vars() == {X, Y, Z}
  assert e.substitute({X: 5, Y: 2, Z: -3}) == (5 * 4 + 2) // 3 + (-3) % 2
  with pytest.raises(KeyError): e.substitute({X: 1})
  assert sym_infer(7, {}) == 7 and sym_infer(X * 2, {X: 3}) == 6
  assert sym_subs(X * 4 + Y, {X: 2}) == Y + 8
  assert X.render() == "x"
  ops = type("Ops", (RenderOps,), {"var": staticmethod(lambda v: f"var_{v.name}")})
  assert (X + 1).render(ops) == "(var_x+1)"

def test_substitute_numpy():
  e = ands([((X * 4 + Y) % 7).lt(5), X.ge(2)])
  xs, ys = np.meshgrid(np.arange(10), np.arange(4), indexing="ij")
  got = e.substitute({X: xs, Y: ys})
  assert np.array_equal(got, (((xs * 4 + ys) % 7) < 5) & (xs >= 2))

def test_smax_smin_ands():
  assert smax(3, 5) == 5 and smin(3, 5) == 3
  assert smax(X, -1) is X and smin(X, 20) is X
  for x, y in itertools.product(range(10), range(4)):
    assert smax(X, Y * 3).substitute({X: x, Y: y}) == max(x, 3 * y)
    assert smin(X, Y * 3).substitute({X: x, Y: y}) == min(x, 3 * y)
  assert ands([1, 1]) == 1 and ands([X.lt(3), 0]) == 0 and ands([]) == 1
  a = ands([X.lt(3), Y.lt(2), X.lt(3)])
  assert isinstance(a, AndNode) and len(a.nodes) == 2

def test_hashable_structural():
  assert {X * 2 + 1: 1}[X * 2 + 1] == 1
  assert Var("x", 0, 9) == X and Var("x", 0, 8) != X

def test_bool():
  assert bool(X + 1) and not bool(NumNode(0))
  with pytest.raises(TypeError): bool(Z)

def test_symbolic_divisor():
  n = Var("n", 1, 8)
  assert (X * n) // n == X or ((X * n) // n).substitute({X: 3, n: 5}) == 3
  for x, nv in itertools.product(range(10), range(1, 9)):
    assert (X // n).substitute({X: x, n: nv}) == x // nv
    assert (X % n).substitute({X: x, n: nv}) == x % nv

# ---------------------------------------------------------------- property tests

def walk(n):
  yield n
  if isinstance(n, (MulNode, DivNode, ModNode, LtNode)):
    for c in (n.a, n.b):
      if isinstance(c, Node): yield from walk(c)
  if isinstance(n, (SumNode, AndNode)):
    for c in n.nodes: yield from walk(c)

def cdiv(a, b): q = abs(a) // abs(b); return q if (a >= 0) == (b > 0) else -q
def cmod(a, b): return a - b * cdiv(a, b)
class CRender(RenderOps):
  div = staticmethod(lambda a, b: f"cdiv({a},{b})")
  mod = staticmethod(lambda a, b: f"cmod({a},{b})")
  and_ = staticmethod(lambda xs: "(" + " and ".join(xs) + ")")

@hst.composite
def exprs(draw, depth=0):
  """(node-or-int, python reference function of an env dict)."""
  if depth >= 4 or draw(hst.integers(0, 3)) == 0:
    if draw(hst.booleans()):
      v = draw(hst.sampled_from(VARS)); return v, (lambda env, v=v: env[v])
    c = draw(hst.integers(-6, 6)); return c, (lambda env, c=c: c)
  op = draw(hst.sampled_from(["add", "addn", "mul", "div", "mod", "lt", "ge", "neg", "muln"]))
  a, fa = draw(exprs(depth + 1))
  k = draw(hst.integers(-5, 7))
  if op == "add": return a + k, (lambda env: fa(env) + k)
  if op == "addn":
    b, fb = draw(exprs(depth + 1)); return (a + b if isinstance(a, Node) else b + a), (lambda env: fa(env) + fb(env))
  if op == "mul": return a * k, (lambda env: fa(env) * k)
  if op == "muln":
    b, fb = draw(exprs(depth + 1)); return (a * b if isinstance(a, Node) else b * a), (lambda env: fa(env) * fb(env))
  if op == "neg": return -a, (lambda env: -fa(env))
  d = draw(hst.integers(1, 7))
  if op == "div": return a // d, (lambda env: fa(env) // d)
  if op == "mod": return a % d, (lambda env: fa(env) % d)
  if op == "lt": return (a.lt(k) if isinstance(a, Node) else int(a < k)), (lambda env: int(fa(env) < k))
  return (a.ge(k) if isinstance(a, Node) else int(a >= k)), (lambda env: int(fa(env) >= k))

ALL_ENVS = [dict(zip(VARS, vals)) for vals in itertools.product(*(range(v.vmin, v.vmax + 1) for v in VARS))]

@settings(max_examples=400, deadline=None)
@given(exprs())
def test_random_expr(e):
  node, ref = e
  for env in ALL_ENVS[::7]:
    want = ref(env)
    if isinstance(node, int): assert node == want; continue
    got = node.substitute(env)
    assert got == want, f"{node.render()} at {env}: {got} != {want}"
    assert node.vmin <= want <= node.vmax, f"bounds [{node.vmin},{node.vmax}] of {node.render()} miss {want}"
    # rendered C-like source (truncating division) agrees: numerators of int divisions are nonnegative
    src = node.render(CRender)
    assert int(eval(src, {"cdiv": cdiv, "cmod": cmod, **{v.name: env[v] for v in VARS}})) == want, src
  if isinstance(node, Node):
    assert node == sym_subs(node, {}) or sym_subs(node, {}).substitute(ALL_ENVS[0]) == ref(ALL_ENVS[0])
    part = sym_subs(node, {X: 3})
    for env in ALL_ENVS[::11]:
      if env[X] == 3: assert sym_infer(part, env) == ref(env)

@settings(max_examples=200, deadline=None)
@given(exprs(), exprs())
def test_random_ands(a, b):
  (na, fa), (nb, fb) = a, b
  ca, cb = (na.lt(2) if isinstance(na, Node) else int(na < 2)), (nb.ge(1) if isinstance(nb, Node) else int(nb >= 1))
  n = ands([ca, cb])
  for env in ALL_ENVS[::13]:
    assert sym_infer(n, env) == int(fa(env) < 2 and fb(env) >= 1)

def test_divisor_may_be_zero():
  # a symbolic dim that can be 0 is a legal divisor (regression: this asserted vmin > 0)
  p, g = Var("p", 0, 8), Var("g", 0, 31)
  for e in (g // p, g % p, (g + 3) // (p * 2), g % (p + 0)):
    assert isinstance(e, Node)
    for pv in range(0, 9):
      for gv in (0, 5, 31):
        got = e.substitute({p: pv, g: gv})   # a 0 divisor evaluates as 1 instead of raising
        assert e.vmin <= got <= e.vmax
        if pv: assert got == eval(e.render().replace("/", "//"), {}, {"p": pv, "g": gv})
  assert (g // p).vmin == 0 and (g // p).vmax == 31 and (g % p).vmax == 7
  assert (Var("q", 0, 3) % Var("r", 0, 9)).vmax == 8
  # numpy arrays as values
  assert list((g % p).substitute({p: np.array([0, 3]), g: np.array([5, 5])})) == [0, 2]
  with pytest.raises(AssertionError): g // 0
  with pytest.raises(AssertionError): g // Var("m", -1, 3)
