"""Zero-copy views (SPEC §5.2). Implemented by WP1.

All movement methods are pure and return new objects. `arg` formats match fusezap.ops.MovementOps:
  reshape(new_shape)  permute(order)  expand(new_shape)  pad(((before, after), ...))
  shrink(((start, end), ...))  flip((bool, ...))
A View maps a logical index (i0..in) to buffer index `offset + sum(ik*stride_k)`, valid where every
`mask[k][0] <= ik < mask[k][1]`. A ShapeTracker composes views: views[-1] is the logical view, and each
earlier view interprets the row-major flattened index of the one after it. views[0] indexes the buffer.
A fully masked view (nothing valid) is canonicalized to strides 0, offset 0, mask ((0,0),...).
"""
from __future__ import annotations
from dataclasses import dataclass
from fusezap.helpers import prod
from fusezap.shape.symbolic import (Node, Var, NumNode, MulNode, SumNode, LtNode, AndNode, sint, ands, ge, lt, smax, smin,
                                    sym_subs)

def _is_int(x) -> bool: return isinstance(x, int)
def _vmax(x) -> int: return x if isinstance(x, int) else x.vmax
def _vmin(x) -> int: return x if isinstance(x, int) else x.vmin
def _sym_vars(xs) -> set:
  out: set = set()
  for x in xs:
    if isinstance(x, Node): out |= x.vars()
  return out

def strides_for_shape(shape: tuple[sint, ...]) -> tuple[sint, ...]:
  """Row-major strides, with size-1 dims canonicalized to 0."""
  strides, acc = [], 1
  for s in reversed(shape):
    strides.append(0 if s == 1 else acc)
    acc = acc * s
  return tuple(reversed(strides))

def _safe_dim(s: sint) -> sint:
  """A divisor for dim `s`: s itself, or max(s, 1) (`s + (s<1)` when symbolic) when s may be 0, so index math
  never divides by 0 at runtime. Where s is 0 the dim has no elements, so the quotient is never meaningfully used."""
  if isinstance(s, int): return max(s, 1)
  return s + lt(s, 1) if s.vmin <= 0 else s

def _unflatten(idx: sint, shape: tuple[sint, ...]) -> list[sint]:
  """Row-major multi-index of a flat index; the outermost dim skips `%` (the index is in range when valid)."""
  out, acc = [], 1
  for k in range(len(shape) - 1, -1, -1):
    s = _safe_dim(shape[k])
    if s == 1: out.append(0)
    else: out.append(idx // acc if k == 0 else (idx // acc) % s)
    acc = acc * s
  return out[::-1]

def _empty_mask(mask) -> bool:
  return mask is not None and any(_vmax(e) - _vmin(b) <= 0 or _vmax(e - b) <= 0 for b, e in mask)

@dataclass(frozen=True)
class View:
  shape: tuple[sint, ...]
  strides: tuple[sint, ...]
  offset: sint
  mask: tuple[tuple[sint, sint], ...] | None   # per-axis valid [start, end); None = all valid
  contiguous: bool

  @staticmethod
  def create(shape: tuple[sint, ...], strides: tuple[sint, ...] | None = None, offset: sint = 0,
             mask: tuple[tuple[sint, sint], ...] | None = None) -> "View":
    """Canonicalizes: size-1 dims get stride 0; a mask covering everything becomes None."""
    shape = tuple(shape)
    strides = strides_for_shape(shape) if strides is None else tuple(0 if s == 1 else st for s, st in zip(shape, strides))
    assert len(strides) == len(shape), f"strides {strides} do not match shape {shape}"
    if any(s == 0 for s in shape): return View(shape, (0,) * len(shape), 0, None, True)
    if mask is not None:
      mask = tuple((b, e) for b, e in mask)
      assert len(mask) == len(shape)
      if _empty_mask(mask): return View(shape, (0,) * len(shape), 0, ((0, 0),) * len(shape), False)
      # a dim restricted to a single element contributes a constant: fold it into the offset
      new_strides = []
      for (b, e), st in zip(mask, strides):
        if _is_int(b) and _is_int(e) and e - b == 1 and st != 0: offset, st = offset + b * st, 0
        new_strides.append(st)
      strides = tuple(new_strides)
      if all(b == 0 and e == s for (b, e), s in zip(mask, shape)): mask = None
    contiguous = offset == 0 and mask is None and strides == strides_for_shape(shape)
    return View(shape, strides, offset, mask, contiguous)

  # ---- index math
  def expr(self, idxs: list[sint]) -> tuple[sint, sint]:
    """(buffer index, valid) for one logical multi-index of this view."""
    assert len(idxs) == len(self.shape), f"need {len(self.shape)} indices, got {len(idxs)}"
    idx: sint = self.offset
    for i, st in zip(idxs, self.strides):
      if st != 0: idx = idx + i * st
    valid: list[sint] = []
    if self.mask is not None:
      for i, (b, e), s in zip(idxs, self.mask, self.shape):
        if b != 0: valid.append(ge(i, b))
        if e != s: valid.append(lt(i, e))
    return idx, ands(valid)

  def vars(self) -> set:
    return _sym_vars(self.shape + self.strides + (self.offset,) + (sum(self.mask, ()) if self.mask else ()))

  def is_empty(self) -> bool: return self.mask is not None and _empty_mask(self.mask)

  def subs(self, var_vals: dict) -> "View":
    f = lambda xs: tuple(sym_subs(x, var_vals) for x in xs)
    return View.create(f(self.shape), f(self.strides), sym_subs(self.offset, var_vals),
                       tuple(f(m) for m in self.mask) if self.mask is not None else None)

  # ---- movement
  def permute(self, axis: tuple[int, ...]) -> "View":
    assert sorted(axis) == list(range(len(self.shape))), f"invalid permutation {axis} for shape {self.shape}"
    return View.create(tuple(self.shape[a] for a in axis), tuple(self.strides[a] for a in axis), self.offset,
                       tuple(self.mask[a] for a in axis) if self.mask is not None else None)

  def expand(self, new_shape: tuple[sint, ...]) -> "View":
    assert len(new_shape) == len(self.shape), f"expand {self.shape} -> {new_shape} changes rank"
    assert all(s == ns or s == 1 for s, ns in zip(self.shape, new_shape)), f"cannot expand {self.shape} -> {new_shape}"
    if any(ns == 0 for ns in new_shape): return View.create(tuple(new_shape))
    mask = None
    if self.mask is not None:
      mask = tuple(m if s == ns else ((0, ns) if m != (0, 0) else (0, 0)) for m, s, ns in zip(self.mask, self.shape, new_shape))
    return View.create(tuple(new_shape), tuple(st if s == ns else 0 for st, s, ns in zip(self.strides, self.shape, new_shape)),
                       self.offset, mask)

  def pad(self, arg: tuple[tuple[sint, sint], ...]) -> "View":
    assert len(arg) == len(self.shape) and all(_vmin(b) >= 0 and _vmin(e) >= 0 for b, e in arg), f"invalid pad {arg}"
    if all(b == 0 and e == 0 for b, e in arg): return self
    old_mask = self.mask if self.mask is not None else tuple((0, s) for s in self.shape)
    mask = tuple((mb + b, me + b) for (mb, me), (b, e) in zip(old_mask, arg))
    offset = self.offset
    for (b, _), st in zip(arg, self.strides):
      if b != 0 and st != 0: offset = offset - b * st
    return View.create(tuple(s + b + e for s, (b, e) in zip(self.shape, arg)), self.strides, offset, mask)

  def shrink(self, arg: tuple[tuple[sint, sint], ...]) -> "View":
    assert len(arg) == len(self.shape), f"invalid shrink {arg} for {self.shape}"
    for (b, e), s in zip(arg, self.shape):
      assert _vmin(b) >= 0 and (not (_is_int(b) and _is_int(e)) or b <= e) and (not (_is_int(e) and _is_int(s)) or e <= s), \
        f"invalid shrink {arg} for {self.shape}"
    if all(b == 0 and e == s for (b, e), s in zip(arg, self.shape)): return self
    mask = None
    if self.mask is not None:
      mask = tuple((smax(mb - b, 0), smin(me - b, e - b)) for (mb, me), (b, e) in zip(self.mask, arg))
    offset = self.offset
    for (b, _), st in zip(arg, self.strides):
      if b != 0 and st != 0: offset = offset + b * st
    return View.create(tuple(e - b for b, e in arg), self.strides, offset, mask)

  def flip(self, arg: tuple[bool, ...]) -> "View":
    assert len(arg) == len(self.shape), f"invalid flip {arg} for {self.shape}"
    if not any(arg): return self
    offset, strides = self.offset, []
    for f, s, st in zip(arg, self.shape, self.strides):
      if f and st != 0: offset = offset + (s - 1) * st
      strides.append(-st if f else st)
    mask = None
    if self.mask is not None: mask = tuple((s - me, s - mb) if f else (mb, me) for f, (mb, me), s in zip(arg, self.mask, self.shape))
    return View.create(self.shape, tuple(strides), offset, mask)

  def reshape(self, new_shape: tuple[sint, ...]) -> "View | None":
    """None if this view cannot express the reshape without a new view."""
    new_shape = tuple(new_shape)
    if self.shape == new_shape: return self
    if all(_is_int(s) for s in self.shape + new_shape):
      assert prod(self.shape) == prod(new_shape), f"reshape {self.shape} -> {new_shape} changes size"
    if any(s == 0 for s in self.shape) or any(s == 0 for s in new_shape): return View.create(new_shape)
    if self.contiguous: return View.create(new_shape)
    if self.is_empty(): return View.create(new_shape, mask=((0, 0),) * len(new_shape)) if new_shape else None
    old = [(s, st, (self.mask[k] if self.mask is not None else (0, s))) for k, (s, st) in enumerate(zip(self.shape, self.strides)) if s != 1]
    new = [s for s in new_shape if s != 1]
    # group old/new dims into runs with equal products
    groups: list[tuple[list, list]] = []
    i = j = 0
    while i < len(old) or j < len(new):
      if i >= len(old) or j >= len(new): return None
      if old[i][0] == new[j]: groups.append(([old[i]], [new[j]])); i, j = i + 1, j + 1; continue
      if not (_is_int(old[i][0]) and _is_int(new[j])): return None
      go, gn, po, pn = [old[i]], [new[j]], old[i][0], new[j]
      while po != pn:
        if po < pn:
          i += 1
          if i >= len(old) or not _is_int(old[i][0]): return None
          go.append(old[i]); po *= old[i][0]
        else:
          j += 1
          if j >= len(new) or not _is_int(new[j]): return None
          gn.append(new[j]); pn *= new[j]
      groups.append((go, gn)); i, j = i + 1, j + 1
    offset, new_strides, new_mask = self.offset, [], []
    for go, gn in groups:
      if len(go) == 1 and len(gn) == 1:
        new_strides.append(go[0][1]); new_mask.append(go[0][2]); continue
      if not all(_is_int(x) for d in go for x in (d[0], *d[2])): return None
      # merge the old dims into one dim of size N: index = S*f + adj, valid on flat range [lo, hi)
      sizes = [d[0] for d in go]
      P = [prod(sizes[k + 1:]) for k in range(len(go))]
      p = next((k for k, d in enumerate(go) if d[2][1] - d[2][0] != 1), len(go) - 1)
      if any(d[2] != (0, d[0]) for d in go[p + 1:]): return None
      if any(go[k][1] != go[k + 1][1] * go[k + 1][0] for k in range(p, len(go) - 1)): return None
      S = go[-1][1]
      F0 = sum(go[k][2][0] * P[k] for k in range(p))
      adj: sint = sum(go[k][2][0] * go[k][1] for k in range(p)) - S * F0
      lo, hi = F0 + go[p][2][0] * P[p], F0 + go[p][2][1] * P[p]
      # split the flat range [lo, hi) over the new dims as a box
      Q = [prod(gn[t + 1:]) for t in range(len(gn))]
      box = None
      for t in range(len(gn) - 1, -1, -1):
        if lo % Q[t] == 0 and hi % Q[t] == 0 and (lo // Q[t]) // gn[t] == (hi // Q[t] - 1) // gn[t]:
          box = [((lo // Q[u]) % gn[u], (lo // Q[u]) % gn[u] + 1) for u in range(t)]
          box += [((lo // Q[t]) % gn[t], (hi // Q[t] - 1) % gn[t] + 1)] + [(0, n) for n in gn[t + 1:]]
          break
      if box is None: return None
      offset = offset + adj
      new_strides += [S * q for q in Q]; new_mask += box
    # re-insert size-1 dims of the new shape
    strides, mask, k = [], [], 0
    for s in new_shape:
      if s == 1: strides.append(0); mask.append((0, 1))
      else: strides.append(new_strides[k]); mask.append(new_mask[k]); k += 1
    return View.create(new_shape, tuple(strides), offset, tuple(mask) if self.mask is not None else None)

def _linear(node: sint, vs: list[Var]) -> tuple[list[int], int] | None:
  """node == const + sum(coef_k * vs[k]) with int coefficients, else None."""
  coefs, const = [0] * len(vs), 0
  terms = node.nodes if isinstance(node, SumNode) else (node,)
  for t in terms:
    if isinstance(t, int): const += t
    elif isinstance(t, NumNode): const += t.b
    elif isinstance(t, Var) and t in vs: coefs[vs.index(t)] += 1
    elif isinstance(t, MulNode) and isinstance(t.a, Var) and t.a in vs and isinstance(t.b, int): coefs[vs.index(t.a)] += t.b
    else: return None
  return coefs, const

def _valid_box(valid: sint, vs: list[Var]) -> list[tuple[int, int]] | None:
  """valid == AND of single-variable bounds -> per-var [lo, hi), else None."""
  box = [(v.vmin, v.vmax + 1) for v in vs]
  if valid == 0: return [(0, 0)] * len(vs)
  if isinstance(valid, int): return box
  for c in (valid.nodes if isinstance(valid, AndNode) else (valid,)):
    if not isinstance(c, LtNode): return None
    if isinstance(c.a, Var) and c.a in vs and isinstance(c.b, int):
      k = vs.index(c.a); box[k] = (box[k][0], min(box[k][1], c.b))
    elif isinstance(c.b, Var) and c.b in vs and isinstance(c.a, int):
      k = vs.index(c.b); box[k] = (max(box[k][0], c.a + 1), box[k][1])
    else: return None
  return box

def _merge_views(vi: View, vo: View) -> View | None:
  """One View equal to applying `vo` on top of `vi` (vo indexes vi's flattened logical space), or None."""
  if vo.is_empty() or vi.contiguous: return vo
  if vo.contiguous and prod(vo.shape) == prod(vi.shape) and (r := vi.reshape(vo.shape)) is not None: return r
  if vi.vars() or vo.vars() or any(s == 0 for s in vo.shape): return None
  # a Var per outer dim, bounded by vo's mask: the merged view only has to agree where vo is valid
  mask = vo.mask if vo.mask is not None else tuple((0, s) for s in vo.shape)
  vs = [Var(f"_m{k}", b, e - 1) for k, (b, e) in enumerate(mask)]
  idx: sint = vo.offset
  for v, st in zip(vs, vo.strides): idx = idx + v * st
  idx, valid = vi.expr(_unflatten(idx, vi.shape))
  if (lin := _linear(idx, vs)) is None or (box := _valid_box(valid, vs)) is None: return None
  if not vo.shape and valid == 0: return None   # a 0-d view cannot be masked
  new_mask = tuple(box) if vo.mask is not None or any(b != (0, s) for b, s in zip(box, vo.shape)) else None
  return View.create(vo.shape, tuple(lin[0]), lin[1], new_mask)

@dataclass(frozen=True)
class ShapeTracker:
  views: tuple[View, ...]

  @staticmethod
  def from_shape(shape: tuple[sint, ...]) -> "ShapeTracker":
    return ShapeTracker((View.create(shape),))

  @property
  def shape(self) -> tuple[sint, ...]: return self.views[-1].shape
  @property
  def contiguous(self) -> bool:
    """True iff a single, unmasked, contiguous view with offset 0."""
    return len(self.views) == 1 and self.views[0].contiguous
  @property
  def size(self) -> sint:
    """Number of logical elements."""
    return prod(self.shape)
  def real_size(self) -> int:
    """Minimum number of elements the backing buffer needs (max reachable index + 1)."""
    v = self.views[0]
    if any(s == 0 for s in v.shape) or v.is_empty(): return 0
    mask = v.mask if v.mask is not None else tuple((0, s) for s in v.shape)
    hi: sint = v.offset
    for (b, e), st in zip(mask, v.strides):
      if st != 0: hi = hi + (e - 1) * st if _vmin(st) > 0 else hi + b * st if _vmax(st) < 0 else hi + smax((e - 1) * st, b * st)
    return max(_vmax(hi) + 1, 0)

  def _replace_last(self, v: View) -> "ShapeTracker":
    st = ShapeTracker(self.views[:-1] + (v,))
    return st.simplify() if len(st.views) > 1 else st
  def reshape(self, new_shape: tuple[sint, ...]) -> "ShapeTracker":
    if (v := self.views[-1].reshape(tuple(new_shape))) is not None: return self._replace_last(v)
    return ShapeTracker(self.views + (View.create(tuple(new_shape)),)).simplify()
  def permute(self, axis: tuple[int, ...]) -> "ShapeTracker": return self._replace_last(self.views[-1].permute(tuple(axis)))
  def expand(self, new_shape: tuple[sint, ...]) -> "ShapeTracker": return self._replace_last(self.views[-1].expand(tuple(new_shape)))
  def pad(self, arg: tuple[tuple[sint, sint], ...]) -> "ShapeTracker": return self._replace_last(self.views[-1].pad(tuple(arg)))
  def shrink(self, arg: tuple[tuple[sint, sint], ...]) -> "ShapeTracker": return self._replace_last(self.views[-1].shrink(tuple(arg)))
  def flip(self, arg: tuple[bool, ...]) -> "ShapeTracker": return self._replace_last(self.views[-1].flip(tuple(arg)))

  def simplify(self) -> "ShapeTracker":
    """Merge adjacent views where possible."""
    views = self.views
    while len(views) >= 2 and (m := _merge_views(views[-2], views[-1])) is not None: views = views[:-2] + (m,)
    return ShapeTracker(views)

  def expr_idxs(self, idxs: list[sint]) -> tuple[sint, sint]:
    """Logical multi-index -> (linear buffer index, validity 0/1). The only index math codegen uses."""
    idx, valid = self.views[-1].expr(list(idxs))
    for v in reversed(self.views[:-1]):
      if valid == 0: return 0, 0
      idx, vv = v.expr(_unflatten(idx, v.shape))
      valid = ands([valid, vv])
    return (0, 0) if valid == 0 else (idx, valid)

  def expr_node(self, gidx: sint) -> tuple[sint, sint]:
    """Same as expr_idxs, from one row-major flattened logical index."""
    return self.expr_idxs(_unflatten(gidx, self.shape))

  def unit_stride_axes(self) -> list[int]:
    """Axes whose buffer stride is 1 (for vectorization later)."""
    if len(self.views) == 1: return [k for k, st in enumerate(self.views[0].strides) if st == 1]
    if self.vars() or any(s == 0 for s in self.shape): return []
    vs = [Var(f"_u{k}", 0, s - 1) for k, s in enumerate(self.shape)]
    lin = _linear(self.expr_idxs(vs)[0], vs)
    return [] if lin is None else [k for k, c in enumerate(lin[0]) if c == 1]

  def vars(self) -> set: return set().union(*(v.vars() for v in self.views))

  def subs(self, var_vals: dict) -> "ShapeTracker":
    """Substitute (some) symbolic Vars with values, e.g. to get a concrete tracker for one var_vals."""
    return ShapeTracker(tuple(v.subs(var_vals) for v in self.views)).simplify()
