"""Symbolic integer expressions for shapes and index math (SPEC §5.1). Implemented by WP1.

Contract (other packages rely on exactly this):
  sint = int | Node
  Var(name, vmin, vmax)                       a bounded integer variable
  NumNode(b)                                  a constant (Node form); prefer plain ints in shapes
  Node ops: + - * // % with int or Node, unary -, `lt(other)` -> Node (0/1), `ge` likewise
  node.vmin / node.vmax                       inclusive bounds
  node.vars() -> set[Var]                     free variables
  node.substitute(dict[Var, int]) -> int      evaluate
  node.render(ops: RenderOps) -> str          C-like source; RenderOps lets renderers name variables
  sym_infer(x: sint, var_vals) -> int         int passthrough, Node evaluated
  smax(a, b), smin(a, b), ands([nodes]) -> Node   (ands: logical and of 0/1 nodes)
Nodes are immutable and hashable; equal structure means equal hash.
Simplification minimums are listed in SPEC §5.1.

Node kinds (for code that walks trees, e.g. lowering to UOps). Every op folds constants and returns a
plain int when the result is constant, so trees only contain these classes:
  Var(name, vmin, vmax)       NumNode(b)                (b: int; only appears as the constant term of a SumNode)
  MulNode(a: Node, b: sint)   a*b                       DivNode(a: Node, b: sint)  floor(a/b), b > 0 (*)
  ModNode(a: Node, b: sint)   a mod b (floor), b > 0 (*) LtNode(a: sint, b: sint)   1 if a < b else 0 (one side may be an int)
  SumNode(nodes: tuple)       sum of terms              AndNode(nodes: tuple)      product of 0/1 terms (logical and)
Division semantics: `//` and `%` are floor division/modulo (Python). Whenever the divisor is an int, the simplifier
normalizes the numerator so that its vmin >= 0, so C truncating `/` and `%` give the same result: renderers and
lowering can map DivNode/ModNode straight to IDIV/MOD.
(*) A symbolic (Node) divisor may have vmin == 0: a dim that can be 0. Its value is only meaningful when it is > 0
(a 0 dim means nothing indexes it), so bounds are computed as if vmin were 1 and `substitute` divides by 1 instead
of 0. Index math built by fusezap.shape.view uses `s + (s<1)` for such dims, so rendered code never divides by 0.
An int divisor must be > 0 (int 0 asserts).
`substitute` only uses + * // % < on the values, so it also evaluates with numpy integer arrays as var values
(LtNode yields an int array). `subs` does partial substitution and returns a simplified sint.
"""
from __future__ import annotations
import math
from typing import Callable, Union

def _c(x):  # constant Node (NumNode, or any node with vmin == vmax) -> int; ints pass through
  return x.vmin if isinstance(x, Node) and x.vmin == x.vmax else x
def _vmin(x) -> int: return x if isinstance(x, int) else x.vmin
def _vmax(x) -> int: return x if isinstance(x, int) else x.vmax
def _b2i(x):  # comparison result -> 0/1 (python bool or numpy bool array)
  return int(x) if isinstance(x, bool) or not hasattr(x, "astype") else x.astype("int64")
def _nz(x):  # a symbolic divisor's value, with 0 (a 0-size dim: the result is never used) replaced by 1
  return x + _b2i(x == 0)

class Node:
  vmin: int
  vmax: int
  _key: tuple
  def __eq__(self, o) -> bool:
    if isinstance(o, int) and not isinstance(o, bool): return isinstance(self, NumNode) and self.b == o
    return isinstance(o, Node) and type(o) is type(self) and o._key == self._key
  def __hash__(self) -> int:
    if (h := self.__dict__.get("_h")) is None: object.__setattr__(self, "_h", h := hash((type(self).__name__, self._key)))
    return h
  def __bool__(self) -> bool:
    if self.vmin == self.vmax: return self.vmin != 0
    if self.vmin > 0 or self.vmax < 0: return True
    raise TypeError(f"truth value of symbolic {self!r} is ambiguous")
  def __int__(self) -> int:
    if self.vmin == self.vmax: return self.vmin
    raise TypeError(f"symbolic {self!r} is not a constant")
  def __repr__(self) -> str: return f"<{self.render()}>"
  def _str(self) -> str:
    if (s := self.__dict__.get("_s")) is None: object.__setattr__(self, "_s", s := self.render())
    return s

  def vars(self) -> set["Var"]: raise NotImplementedError
  def substitute(self, var_vals: dict["Var", int]) -> int: raise NotImplementedError
  def subs(self, var_vals: dict["Var", "sint"]) -> "sint": raise NotImplementedError
  def render(self, ops: "RenderOps | None" = None) -> str: raise NotImplementedError

  def __add__(self, b: sint) -> sint: return add(self, b)
  def __radd__(self, b: int) -> sint: return add(b, self)
  def __sub__(self, b: sint) -> sint: return add(self, mul(b, -1))
  def __rsub__(self, b: int) -> sint: return add(b, mul(self, -1))
  def __neg__(self) -> sint: return mul(self, -1)
  def __mul__(self, b: sint) -> sint: return mul(self, b)
  def __rmul__(self, b: int) -> sint: return mul(b, self)
  def __floordiv__(self, b: sint) -> sint: return floordiv(self, b)
  def __rfloordiv__(self, b: int) -> sint: return floordiv(b, self)
  def __mod__(self, b: sint) -> sint: return mod(self, b)
  def __rmod__(self, b: int) -> sint: return mod(b, self)
  def lt(self, b: sint) -> sint: return lt(self, b)
  def ge(self, b: sint) -> sint: return ge(self, b)
  # SPEC: `<` returns a Node (0/1). Decidable comparisons fold to an int.
  def __lt__(self, b: sint) -> sint: return lt(self, b)
  def __ge__(self, b: sint) -> sint: return ge(self, b)
  def __gt__(self, b: sint) -> sint: return lt(b, self)
  def __le__(self, b: sint) -> sint: return lt(self, add(b, 1))

class Var(Node):
  def __init__(self, name: str, vmin: int, vmax: int):
    assert isinstance(vmin, int) and isinstance(vmax, int) and vmin <= vmax, f"bad bounds for {name}: [{vmin}, {vmax}]"
    self.name, self.vmin, self.vmax, self._key = name, vmin, vmax, (name, vmin, vmax)
  def vars(self): return {self}
  def substitute(self, var_vals):
    if self not in var_vals: raise KeyError(f"no value for {self.name}")
    return var_vals[self]
  def subs(self, var_vals): return _c(var_vals.get(self, self))
  def render(self, ops=None): return (ops or RenderOps).var(self)
  def __repr__(self): return f"Var({self.name!r}, {self.vmin}, {self.vmax})"

class NumNode(Node):
  def __init__(self, b: int):
    assert isinstance(b, int)
    self.b = self.vmin = self.vmax = b
    self._key = (b,)
  def vars(self): return set()
  def substitute(self, var_vals): return self.b
  def subs(self, var_vals): return self.b
  def render(self, ops=None): return str(self.b)

sint = Union[int, Node]

class _BinNode(Node):
  def __init__(self, a: sint, b: sint, vmin: int, vmax: int):
    self.a, self.b, self.vmin, self.vmax, self._key = a, b, vmin, vmax, (a, b)
  def vars(self):
    return (self.a.vars() if isinstance(self.a, Node) else set()) | (self.b.vars() if isinstance(self.b, Node) else set())
  def _sa(self, var_vals): return self.a.substitute(var_vals) if isinstance(self.a, Node) else self.a
  def _sb(self, var_vals): return self.b.substitute(var_vals) if isinstance(self.b, Node) else self.b
  def _ra(self, ops): return self.a.render(ops) if isinstance(self.a, Node) else str(self.a)
  def _rb(self, ops): return self.b.render(ops) if isinstance(self.b, Node) else str(self.b)
  def subs(self, var_vals):
    return self._fn(*(x.subs(var_vals) if isinstance(x, Node) else x for x in (self.a, self.b)))

class MulNode(_BinNode):
  _fn = staticmethod(lambda a, b: mul(a, b))
  def substitute(self, var_vals): return self._sa(var_vals) * self._sb(var_vals)
  def render(self, ops=None): return f"({self._ra(ops)}*{self._rb(ops)})"

class DivNode(_BinNode):
  _fn = staticmethod(lambda a, b: floordiv(a, b))
  def substitute(self, var_vals): return self._sa(var_vals) // (_nz(self._sb(var_vals)) if isinstance(self.b, Node) else self.b)
  def render(self, ops=None): return (ops or RenderOps).div(self._ra(ops), self._rb(ops))

class ModNode(_BinNode):
  _fn = staticmethod(lambda a, b: mod(a, b))
  def substitute(self, var_vals): return self._sa(var_vals) % (_nz(self._sb(var_vals)) if isinstance(self.b, Node) else self.b)
  def render(self, ops=None): return (ops or RenderOps).mod(self._ra(ops), self._rb(ops))

class LtNode(_BinNode):
  _fn = staticmethod(lambda a, b: lt(a, b))
  def substitute(self, var_vals): return _b2i(self._sa(var_vals) < self._sb(var_vals))
  def render(self, ops=None): return f"({self._ra(ops)}<{self._rb(ops)})"

class _RedNode(Node):
  def __init__(self, nodes: tuple[Node, ...], vmin: int, vmax: int):
    self.nodes, self.vmin, self.vmax, self._key = tuple(nodes), vmin, vmax, tuple(nodes)
  def vars(self): return set().union(*(n.vars() for n in self.nodes))

class SumNode(_RedNode):
  def substitute(self, var_vals):
    ret = 0
    for n in self.nodes: ret = ret + n.substitute(var_vals)
    return ret
  def subs(self, var_vals):
    ret: sint = 0
    for n in self.nodes: ret = add(ret, n.subs(var_vals))
    return ret
  def render(self, ops=None): return "(" + "+".join(n.render(ops) for n in self.nodes) + ")"

class AndNode(_RedNode):
  def substitute(self, var_vals):
    ret = 1
    for n in self.nodes: ret = ret * n.substitute(var_vals)
    return ret
  def subs(self, var_vals): return ands([n.subs(var_vals) for n in self.nodes])
  def render(self, ops=None): return (ops or RenderOps).and_(n.render(ops) for n in self.nodes)

class RenderOps:
  """Hook for renderers: how to print a Var (default: its name) and integer division/modulo."""
  var: Callable[[Var], str] = staticmethod(lambda v: v.name)
  div: Callable[[str, str], str] = staticmethod(lambda a, b: f"({a}/{b})")
  mod: Callable[[str, str], str] = staticmethod(lambda a, b: f"({a}%{b})")
  and_: Callable = staticmethod(lambda xs: "(" + "&&".join(xs) + ")")

# ---------------------------------------------------------------- simplifying constructors

def _corners(f, a: sint, b: sint) -> tuple[int, int]:
  vals = [f(x, y) for x in (_vmin(a), _vmax(a)) for y in (_vmin(b), _vmax(b))]
  return min(vals), max(vals)

def _const_or(n: Node) -> sint: return n.vmin if n.vmin == n.vmax else n

def _coef(t: Node) -> tuple[Node, int]:
  return (t.a, t.b) if isinstance(t, MulNode) and isinstance(t.b, int) else (t, 1)

def _terms(x: sint) -> tuple[list[Node], int]:
  x = _c(x)
  if isinstance(x, int): return [], x
  if isinstance(x, SumNode):
    ts = [n for n in x.nodes if not isinstance(n, NumNode)]
    return ts, sum(n.b for n in x.nodes if isinstance(n, NumNode))
  return [x], 0

def _build_sum(terms: list[Node], const: int) -> sint:
  coefs: dict[Node, int] = {}
  for t in terms:
    if isinstance(t, NumNode): const += t.b; continue
    if isinstance(t, SumNode):
      ts, c = _terms(t); const += c
      for tt in ts: base, k = _coef(tt); coefs[base] = coefs.get(base, 0) + k
      continue
    base, k = _coef(t)
    coefs[base] = coefs.get(base, 0) + k
  coefs = {b: k for b, k in coefs.items() if k != 0}
  # (x//m)*(k*m) + (x%m)*k -> x*k
  for base, k in coefs.items():
    if isinstance(base, ModNode) and isinstance(base.b, int):
      dn = DivNode(base.a, base.b, 0, 0)
      if coefs.get(dn) == k * base.b:
        rest = [mul(b2, k2) for b2, k2 in coefs.items() if b2 != base and b2 != dn]
        ts, c = _terms(mul(base.a, k))
        return _build_sum(rest + ts, const + c)
  nodes = sorted((_mul_raw(b, k) for b, k in coefs.items()), key=lambda n: n._str())
  if not nodes: return const
  if len(nodes) == 1 and const == 0: return nodes[0]
  vmin, vmax = const + sum(n.vmin for n in nodes), const + sum(n.vmax for n in nodes)
  if vmin == vmax: return vmin
  return SumNode(tuple(nodes) + ((NumNode(const),) if const else ()), vmin, vmax)

def _mul_raw(a: Node, b: int) -> Node:
  if b == 1: return a
  return MulNode(a, b, *sorted((a.vmin * b, a.vmax * b)))

def add(a: sint, b: sint) -> sint:
  a, b = _c(a), _c(b)
  if isinstance(a, int) and isinstance(b, int): return a + b
  ta, ca = _terms(a); tb, cb = _terms(b)
  return _build_sum(ta + tb, ca + cb)

def mul(a: sint, b: sint) -> sint:
  a, b = _c(a), _c(b)
  if isinstance(a, int) and isinstance(b, int): return a * b
  if isinstance(a, int): a, b = b, a
  if isinstance(b, Node):
    if b.vmin == b.vmax: b = b.vmin
    elif a.vmin == a.vmax: a, b = b, a.vmin
  if isinstance(b, int):
    if b == 0: return 0
    if b == 1: return a
    if isinstance(a, MulNode) and isinstance(a.b, int): return _mul_raw(a.a, a.b * b) if a.b * b != 1 else a.a
    if isinstance(a, SumNode):
      ts, c = _terms(a)
      return _build_sum([mul(t, b) for t in ts], c * b)
    return _mul_raw(a, b)
  a, b = sorted((a, b), key=lambda n: n._str())
  return _const_or(MulNode(a, b, *_corners(lambda x, y: x * y, a, b)))

def _gcd_terms(terms: list[Node], *ints: int) -> int:
  return math.gcd(*[_coef(t)[1] for t in terms], *ints)

def _div_terms(terms: list[Node], g: int) -> list[Node]:
  return [mul(base, k // g) for base, k in map(_coef, terms)]

def floordiv(a: sint, b: sint) -> sint:
  a, b = _c(a), _c(b)
  if isinstance(b, Node) and b.vmin == b.vmax: b = b.vmin
  if isinstance(b, Node):
    assert b.vmin >= 0, f"symbolic divisor must be non-negative, got {b!r}"
    if isinstance(a, int) and a == 0: return 0
    if a == b: return 1
    vals = [x // y for x in (_vmin(a), _vmax(a)) for y in (max(b.vmin, 1), b.vmax)]
    return _const_or(DivNode(a, b, min(vals), max(vals)))
  assert b != 0, "division by zero"
  if b < 0: return floordiv(mul(a, -1), -b)
  if b == 1: return a
  if isinstance(a, int): return a // b
  if a.vmin // b == a.vmax // b: return a.vmin // b
  if isinstance(a, MulNode) and isinstance(a.b, int):
    if a.b % b == 0: return mul(a.a, a.b // b)
    if a.b > 0 and b % a.b == 0: return floordiv(a.a, b // a.b)
  if isinstance(a, DivNode) and isinstance(a.b, int): return floordiv(a.a, a.b * b)
  terms, const = _terms(a)
  fact: sint = 0
  rest = []
  for t in terms:
    base, k = _coef(t)
    if k % b == 0: fact = add(fact, mul(base, k // b))
    else: rest.append(t)
  if not rest: return add(fact, const // b)
  rmin = sum(t.vmin for t in rest)
  q = (const + min(rmin, 0)) // b
  const -= q * b
  if (g := _gcd_terms(rest, const, b)) > 1: rest, const, b = _div_terms(rest, g), const // g, b // g
  inner = _build_sum(rest, const)
  if isinstance(inner, int) or inner.vmax < b: return add(fact, q + (inner // b if isinstance(inner, int) else 0))
  return add(add(fact, q), DivNode(inner, b, inner.vmin // b, inner.vmax // b))

def mod(a: sint, b: sint) -> sint:
  a, b = _c(a), _c(b)
  if isinstance(b, Node) and b.vmin == b.vmax: b = b.vmin
  if isinstance(b, Node):
    assert b.vmin >= 0, f"symbolic modulus must be non-negative, got {b!r}"
    if a == b or (isinstance(a, int) and a == 0): return 0
    if _vmin(a) >= 0 and _vmax(a) < max(b.vmin, 1): return a
    return _const_or(ModNode(a, b, 0, b.vmax - 1))
  assert b > 0, f"modulus must be positive, got {b}"
  if b == 1: return 0
  if isinstance(a, int): return a % b
  if a.vmin // b == a.vmax // b: return add(a, -(a.vmin // b) * b)
  if isinstance(a, ModNode) and isinstance(a.b, int) and a.b % b == 0: return mod(a.a, b)
  terms, const = _terms(a)
  rest = []
  for t in terms:
    base, k = _coef(t)
    k2 = k if abs(k) < b else k % b
    if k2 != 0: rest.append(mul(base, k2))
  if not rest: return const % b
  rmin = sum(t.vmin for t in rest)
  const -= ((const + min(rmin, 0)) // b) * b   # same normalization as floordiv, so (e//b)*b + e%b recombines
  if (g := _gcd_terms(rest, const, b)) > 1: return mul(mod(_build_sum(_div_terms(rest, g), const // g), b // g), g)
  inner = _build_sum(rest, const)
  if isinstance(inner, int): return inner % b
  if inner.vmax < b: return inner
  if isinstance(inner, ModNode) and isinstance(inner.b, int) and inner.b % b == 0: return mod(inner.a, b)
  return ModNode(inner, b, 0, min(b - 1, inner.vmax))

def lt(a: sint, b: sint) -> sint:
  a, b = _c(a), _c(b)
  if isinstance(a, int) and isinstance(b, int): return int(a < b)
  if _vmax(a) < _vmin(b): return 1
  if _vmin(a) >= _vmax(b): return 0
  if isinstance(a, Node) and isinstance(b, Node): return lt(add(a, mul(b, -1)), 0)
  if isinstance(b, int):  # node < int
    if isinstance(a, MulNode) and isinstance(a.b, int) and a.b > 0: return lt(a.a, -((-b) // a.b))
    if isinstance(a, SumNode):
      terms, const = _terms(a)
      g = _gcd_terms(terms)
      if const != 0 or g > 1: return lt(_build_sum(_div_terms(terms, g), 0), -((-(b - const)) // g))
    return LtNode(a, b, 0, 1)
  # int < node
  if isinstance(b, MulNode) and isinstance(b.b, int) and b.b > 0: return lt(a // b.b, b.a)
  if isinstance(b, SumNode):
    terms, const = _terms(b)
    g = _gcd_terms(terms)
    if const != 0 or g > 1: return lt((a - const) // g, _build_sum(_div_terms(terms, g), 0))
  return LtNode(a, b, 0, 1)

def ge(a: sint, b: sint) -> sint:
  """a >= b, rendered as a comparison `(b-1 < a)` when b is an int."""
  a, b = _c(a), _c(b)
  if isinstance(b, int): return lt(b - 1, a)
  return lt(add(b, -1), a) if isinstance(a, int) else lt(add(b, mul(a, -1)), 1)

def sym_infer(x: sint, var_vals: dict[Var, int]) -> int:
  return x if isinstance(x, int) else x.substitute(var_vals)

def sym_subs(x: sint, var_vals: dict[Var, sint]) -> sint:
  """Partial substitution: replace the given Vars and re-simplify."""
  return x if isinstance(x, int) else x.subs(var_vals)

def smax(a: sint, b: sint) -> sint:
  a, b = _c(a), _c(b)
  if isinstance(a, int) and isinstance(b, int): return max(a, b)
  if _vmin(a) >= _vmax(b): return a
  if _vmin(b) >= _vmax(a): return b
  return add(a, mul(lt(a, b), add(b, mul(a, -1))))

def smin(a: sint, b: sint) -> sint:
  return mul(smax(mul(a, -1), mul(b, -1)), -1)

def ands(nodes: list[sint]) -> sint:
  out: list[Node] = []
  for n in nodes:
    n = _c(n)
    if isinstance(n, int):
      if n == 0: return 0
      assert n == 1, f"ands expects 0/1 values, got {n}"
      continue
    for x in (n.nodes if isinstance(n, AndNode) else (n,)):
      if x.vmax == 0: return 0
      if x not in out: out.append(x)
  if not out: return 1
  if len(out) == 1: return out[0]
  return AndNode(tuple(sorted(out, key=lambda n: n._str())), 0, 1)
