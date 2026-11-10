"""Kernel scheduler (SPEC §6.2). Implemented by WP3.

create_schedule(outs) -> list[ScheduleItem]
  - topologically ordered; each item's input buffers are realized by earlier items or already exist
  - applies the fusion rules in SPEC §6.2 and builds the kernel AST per §6.3
  - creates (unallocated) output Buffers and assigns them to `lb.base.realized`; allocation is
    deferred to run_schedule
  - FZ_DEBUG>=2 prints the schedule; >=3 also prints each AST (LazyOp.pretty)

Planning (which bases become kernel outputs), on top of the SPEC §6.2 rules:
  - "consumed by more than one kernel": every unmaterialized base gets the kernel root it fuses into;
    a base reached from two roots and containing a reduce is materialized (pure elementwise is recomputed).
  - one reduce per kernel: a reduce below another reduce in the same kernel, or a second sibling reduce
    fusing into a root that already has one, is materialized.
  - non-trivial view: a reduce-containing base read through anything but a contiguous reshape is materialized,
    so the path from a kernel root to its fused reduce is always reshape-only.
  - pad safety: a masked view (pad) read of a fused elementwise tree is only valid if the tree maps all-zero
    inputs to 0 (masked lanes read 0). Otherwise the base is materialized.
Kernel AST conventions (§6.3): F = the reduce input shape (or the output shape without a reduce), R = F with the
reduced axes set to 1. LOAD/CONST sts have shape F (epilogue leaves stride 0 over the reduced axes), the STORE st is
contiguous over R. A CONST st may carry a mask: the value is `val` where valid, 0 elsewhere (like a masked LOAD).

Custom kernels (SPEC §11.1, P2-CUSTOM): when `Device[b.device].custom_kernels` (dict pattern -> CustomKernel) offers a
kernel with source for that device, a matched subgraph becomes one ScheduleItem(kind="custom") whose `ast` is
`(CustomCall(kernel, params),)` and whose bufs are (out, *inputs). The match replaces the base's srcs with the kernel's
inputs in the graph, the inputs are materialized (they are read through plain strided views) and the output is a
contiguous buffer over the reduce's keepdim shape. Patterns (anything else falls back to composed ops; FZ_CUSTOM=0
disables matching):
  - "matmul":    SUM(MUL(x, y), one axis) where every non-reduced axis collapses into batch (both operands vary), M (only
                 x varies) or N (only y varies) groups with single strides, M*N > 1; float32, int shapes, unmasked single views.
  - "attention": softmax(matmul(q, k) [* c]) @ v exactly as Tensor.softmax / scaled_dot_product_attention compose it
                 (no mask, no dropout). The kernel repeats the composed op order, so CPU output is bit-identical.
  - "attention_mask": the same with the scores masked by WHERE(bool mask, scores, -inf) (is_causal, a bool attn_mask,
                 the models' position masks); the mask is a 4th input. Only NEURON offers it (NKI-N3), so other devices
                 keep these graphs composed.
  - "conv":      (I81 D8) a composed conv2d: the matmul weight (O, C*T) @ col (C*T, P) where col is Tensor.stack of T =
                 kh*kw shifted (padded, strided) views of one materialized (1, C, H, W) input X (an ADD of T masked views,
                 shape (C, T, P)). _match_conv proves the views are im2col of X (exact symbolic index check, see there);
                 the kernel reads X directly, so the (C*T, P) im2col buffer is never written. Inputs: the weight and X.
"""
from __future__ import annotations
import math, struct, sys
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from fusezap.device import Buffer
from fusezap.dtype import dtypes, to_np
from fusezap.helpers import getenv, prod
from fusezap.ir import ScheduleItem, LazyOp, MemBuffer, ConstBuffer
from fusezap.lazy import LazyBuffer
from fusezap.ops import BufferOps, LoadOps, ReduceOps, UnaryOps, BinaryOps, TernaryOps
from fusezap.shape.symbolic import Node, Var, sym_infer
from fusezap.shape.view import ShapeTracker

_MULTI = -1  # group marker: reached from more than one kernel root

def _is_leaf(b: LazyBuffer) -> bool: return b.realized is not None or b.op is LoadOps.CONST

def _has_mask(st: ShapeTracker) -> bool: return any(v.mask is not None for v in st.views)

def compose(inner: ShapeTracker, outer: ShapeTracker) -> ShapeTracker:
  """`inner` maps its logical shape S onto a buffer; `outer` maps the kernel shape onto row-major linear indices of S.
  Returns the tracker mapping the kernel shape onto inner's buffer."""
  if any(x == 0 for x in outer.shape): return ShapeTracker.from_shape(outer.shape)  # no elements: never indexed
  if outer.contiguous and outer.size == inner.size: return inner if inner.shape == outer.shape else inner.reshape(outer.shape)
  if inner.contiguous: return outer
  return ShapeTracker(inner.views + outer.views).simplify()

def _single_view(device: str) -> bool:
  """Does the device lower only single-view ShapeTrackers (NEURON: no div/mod index math, docs/NKI_BACKEND.md §2 "Views")."""
  from fusezap.device import Device
  try: return bool(getattr(Device[device], "single_view", False))
  except Exception: return False

def _one_view_reads(x: LazyBuffer) -> LazyBuffer:
  """x read through single views only: each view but the last is materialized by a contiguous copy (whose index space the
  next view addresses), so x = the last view over a chain of copies."""
  if len(x.st.views) <= 1: return x
  cur = x.base
  for v in x.st.views[:-1]: cur = cur._view(ShapeTracker((v,))).contiguous()
  return cur._view(ShapeTracker((x.st.views[-1],)))

class _Graph:
  def __init__(self, outs: list[LazyBuffer]):
    self.keep: list[Any] = []                  # keep synthesized bases alive (ids are dict keys)
    self.copy_src: dict[int, LazyBuffer] = {}  # COPY base id -> contiguous source
    self.custom: dict[int, _Match | None] = {}  # base id -> custom-kernel match (None: no match)
    self.single = _single_view(outs[0].device) if outs else False
    self.view_src: dict[int, tuple] = {}       # single-view devices: base id -> its srcs, multi-view reads made contiguous
    self.order = self._toposort([o.base for o in outs])

  def srcs(self, b: LazyBuffer) -> tuple[LazyBuffer, ...]:
    if b.op is LoadOps.COPY and b.srcs:
      if id(b) not in self.copy_src:
        s = b.srcs[0]
        if s.base is not s or s.base.op is LoadOps.CONST: s = s.contiguous()  # COPY moves whole buffers
        self.copy_src[id(b)] = s
        self.keep.append((b, s))
      return (self.copy_src[id(b)],)
    if isinstance(b.op, ReduceOps) and b.realized is None:
      if id(b) not in self.custom:
        self.custom[id(b)] = _match_custom(b)
        self.keep.append(b)
      if (m := self.custom[id(b)]) is not None: return m.inputs
    if self.single and b.srcs and any(len(x.st.views) > 1 for x in b.srcs):   # a read that is several views by itself (e.g. a
      if id(b) not in self.view_src:                                           # reshape of a permute) goes through copies
        self.view_src[id(b)] = tuple(_one_view_reads(x) for x in b.srcs)
        self.keep.append((b, self.view_src[id(b)]))
      return self.view_src[id(b)]
    return b.srcs

  def _toposort(self, roots: list[LazyBuffer]) -> list[LazyBuffer]:
    order, seen = [], set()
    stack: list[tuple[LazyBuffer, bool]] = [(r, False) for r in reversed(roots)]
    while stack:
      b, done = stack.pop()
      if done:
        order.append(b)
        continue
      if id(b) in seen or b.realized is not None: continue
      seen.add(id(b))
      stack.append((b, True))
      for s in reversed(self.srcs(b)):
        if id(s.base) not in seen and s.base.realized is None: stack.append((s.base, False))
    return order

def _zero_eval(b: LazyBuffer, real: set[int], g: _Graph, memo: dict) -> Any:
  """Value of b's fused tree when every leaf reads 0 (a masked lane). None if the tree is not plain elementwise."""
  from fusezap.interpret import _elementwise, _vdtype
  if id(b) in memo: return memo[id(b)]
  if _is_leaf(b) or id(b) in real: r = np.zeros((), _vdtype(b.dtype))
  elif isinstance(b.op, (UnaryOps, BinaryOps, TernaryOps)):
    xs = [_zero_eval(s.base, real, g, memo) for s in g.srcs(b)]
    r = None if any(x is None for x in xs) else np.asarray(_elementwise(b.op, xs, [s.dtype for s in b.srcs], b.dtype, b.arg))
  else: r = None
  memo[id(b)] = r
  return r

def _forced(g: _Graph, outs: list[LazyBuffer]) -> set[int]:
  real: set[int] = {id(o.base) for o in outs if not _is_leaf(o.base)}
  for b in g.order:
    if b.op in (LoadOps.CONTIGUOUS, LoadOps.COPY, LoadOps.EMPTY): real.add(id(b))
    if b.op is LoadOps.COPY: real.update(id(s.base) for s in g.srcs(b) if not _is_leaf(s.base))
    if g.custom.get(id(b)) is not None:  # custom kernels read materialized inputs and write their own buffer
      real.add(id(b))
      real.update(id(s.base) for s in g.srcs(b) if not _is_leaf(s.base))
  return real

def _reverse_pass(g: _Graph, real: set[int], has_red: dict[int, bool]) -> tuple[set[int], set[int]]:
  """Consumers first. Reduce-level rules are exact; `has_red` (from a previous plan) only chooses to materialize a
  reduce's epilogue instead of the reduce itself. Returns (real, reduces materialized only for being multi-kernel)."""
  real, multi = set(real), set()
  cons: dict[int, list[tuple[LazyBuffer, LazyBuffer]]] = {}
  for b in g.order:
    for s in g.srcs(b): cons.setdefault(id(s.base), []).append((b, s))
  group: dict[int, int] = {}       # kernel root this base fuses into (_MULTI: several)
  red_above: dict[int, bool] = {}  # a reduce sits between this base and its kernel root (or is the root)
  ntv_above: dict[int, bool] = {}  # the path to the kernel root has a non-trivial view
  claimed: dict[int, int] = {}     # kernel root id -> id of the reduce fused into it
  for b in reversed(g.order):
    if _is_leaf(b): continue
    isred = isinstance(b.op, ReduceOps)
    if id(b) not in real:
      grp, ra, ntv = None, False, False
      for c, lb in cons.get(id(b), []):
        cfused = id(c) not in real
        cg = group[id(c)] if cfused else id(c)
        grp = cg if grp is None or grp == cg else _MULTI
        ra |= isinstance(c.op, ReduceOps) or (cfused and red_above[id(c)])
        ntv |= (lb is not b and not (lb.st.contiguous and lb.size == b.size)) or (cfused and ntv_above[id(c)])
      if grp is None: grp = id(b)
      if has_red.get(id(b), False) and not isred and (grp == _MULTI or ntv): real.add(id(b))
      elif isred and (grp == _MULTI or ra or ntv or claimed.get(grp, id(b)) != id(b)):
        real.add(id(b))
        if not (ra or ntv): multi.add(id(b))
      else:
        group[id(b)], red_above[id(b)], ntv_above[id(b)] = grp, ra, ntv
        if isred: claimed[grp] = id(b)
    if id(b) in real:
      group[id(b)], red_above[id(b)], ntv_above[id(b)] = id(b), False, False
      if isred: claimed[id(b)] = id(b)
  return real, multi

def _has_red(g: _Graph, real: set[int]) -> dict[int, bool]:
  """Does the base's fused tree (under `real`) contain a reduce."""
  out: dict[int, bool] = {}
  for b in g.order:
    if _is_leaf(b): continue
    out[id(b)] = isinstance(b.op, ReduceOps) or any(not _is_leaf(s.base) and id(s.base) not in real and out[id(s.base)] for s in g.srcs(b))
  return out

def _plan(g: _Graph, outs: list[LazyBuffer], extra: set[int] | None = None) -> set[int]:
  """ids of the bases that become kernel outputs (or allocations / copies). extra: bases to materialize as well."""
  forced = _forced(g, outs) | (extra or set())
  real, multi = _reverse_pass(g, forced, {})
  real, _ = _reverse_pass(g, forced, _has_red(g, real - multi))  # 2nd pass: materialize such reduces' epilogues instead
  # pad safety (forward: srcs are final before their consumers are judged)
  masked = {id(s.base) for b in g.order for s in g.srcs(b) if s is not s.base and _has_mask(s.st)}
  memo: dict = {}
  for b in g.order:
    if id(b) in masked and not _is_leaf(b) and id(b) not in real:
      z = _zero_eval(b, real, g, memo)
      if z is None or not (z == 0).all():
        real.add(id(b))
        memo.clear()
  return real

class _Kernel:
  def __init__(self, g: _Graph, real: set[int], bufs: dict[int, Buffer], root: LazyBuffer):
    self.g, self.real, self.bufmap = g, real, bufs
    self.bufs: list[Buffer] = [bufs[id(root)]]
    self.memo: dict = {}
    self.red: LazyOp | None = None
    self.split: set[int] = set()   # single-view devices: fused bases whose reads compose into several views
    red = self._find_reduce(root)
    F = red.srcs[0].shape if red is not None else root.shape
    R = red.shape if red is not None else root.shape
    self.full = ShapeTracker.from_shape(F)
    ctx = self.full if red is None else ShapeTracker.from_shape(R).expand(F)
    val = self._node(root, ctx, as_root=True)
    self.ast = (LazyOp(BufferOps.STORE, (val,), MemBuffer(0, root.dtype, ShapeTracker.from_shape(R))),)

  def _fused(self, b: LazyBuffer) -> bool: return not _is_leaf(b) and id(b) not in self.real
  def _find_reduce(self, root: LazyBuffer) -> LazyBuffer | None:
    found, seen, stack = [], set(), [root]
    while stack:
      b = stack.pop()
      if id(b) in seen: continue
      seen.add(id(b))
      if isinstance(b.op, ReduceOps):
        found.append(b)
        continue
      stack.extend(s.base for s in self.g.srcs(b) if self._fused(s.base))
    assert len(found) <= 1, f"scheduler bug: {len(found)} reduces in one kernel"
    return found[0] if found else None

  def _buf_idx(self, b: LazyBuffer) -> int:
    buf = b.realized if b.realized is not None else self.bufmap[id(b)]
    for i, x in enumerate(self.bufs):
      if x is buf: return i
    self.bufs.append(buf)
    return len(self.bufs) - 1

  def _node(self, lb: LazyBuffer, ctx: ShapeTracker, as_root: bool = False, parent: LazyBuffer | None = None) -> LazyOp:
    b = lb.base
    st = ctx if as_root else compose(lb.st, ctx)
    if self.g.single and len(st.views) > 1 and parent is not None and self._fused(parent):
      self.split.add(id(parent))   # the parent's view composed with this read: materialize the parent (its own kernel reads lb.st)
    if not as_root:
      if b.op is LoadOps.CONST and b.realized is None: return LazyOp(BufferOps.CONST, (), ConstBuffer(b.arg, b.dtype, st))
      if not self._fused(b): return LazyOp(BufferOps.LOAD, (), MemBuffer(self._buf_idx(b), b.dtype, st))
    if isinstance(b.op, ReduceOps):
      if self.red is None: self.red = LazyOp(b.op, (self._node(self.g.srcs(b)[0], self.full, parent=b),), b.arg)
      return self.red
    key = (id(b), st)
    if key in self.memo: return self.memo[key]
    srcs = self.g.srcs(b) if self.g.single else b.srcs
    if b.op is LoadOps.CONTIGUOUS: ret = self._node(srcs[0], st, parent=b)
    else:
      assert isinstance(b.op, (UnaryOps, BinaryOps, TernaryOps)), f"cannot fuse {b}"
      ret = LazyOp(b.op, tuple(self._node(s, st, parent=b) for s in srcs), b.arg)
    self.memo[key] = ret
    return ret

def _ast_vars(ast: tuple[LazyOp, ...]) -> set:
  out: set = set()
  for root in ast:
    for n in root.walk():
      if n.op in (BufferOps.LOAD, BufferOps.STORE, BufferOps.CONST):
        out |= n.arg.st.vars()
        if n.op is BufferOps.CONST and isinstance(n.arg.val, Node): out |= n.arg.val.vars()
  return out

def _size(b: LazyBuffer, var_vals: dict) -> int:
  n = sym_infer(prod(b.shape), var_vals)
  assert isinstance(n, int), f"buffer size of {b} is symbolic; pass var_vals"
  return n

def create_schedule(outs: list[LazyBuffer], var_vals: dict[Var, int] | None = None) -> list[ScheduleItem]:
  var_vals = var_vals or {}
  outs = [o for o in outs if not _is_leaf(o.base)]
  if not outs: return []
  if sys.getrecursionlimit() < 20000: sys.setrecursionlimit(20000)
  g = _Graph(outs)
  real = _plan(g, outs)
  if g.single:   # NEURON: materialize the fused bases whose reads would need several views, until every read is one view
    extra: set[int] = set()
    for _ in range(64):
      dry = {id(b): Buffer(b.device, 1, b.dtype) for b in g.order if id(b) in real}
      bad = set()
      for b in g.order:
        if id(b) in real and b.op not in (LoadOps.COPY, LoadOps.EMPTY) and g.custom.get(id(b)) is None and not _is_leaf(b):
          bad |= _Kernel(g, real, dry, b).split
      if not (bad - extra): break
      extra |= bad
      real = _plan(g, outs, extra)
  bufs = {id(b): Buffer(b.device, _size(b, var_vals), b.dtype) for b in g.order if id(b) in real}
  sched: list[ScheduleItem] = []
  for b in g.order:
    if id(b) not in real or b.op is LoadOps.EMPTY: continue
    if b.op is LoadOps.COPY:
      if b.srcs:
        s = g.srcs(b)[0].base
        src = s.realized if s.realized is not None else bufs[id(s)]
      else:
        a = np.ascontiguousarray(np.asarray(b.arg).reshape(-1)) if not isinstance(b.arg, memoryview) else \
            np.frombuffer(b.arg.cast("B"), dtype=to_np(b.dtype))
        src = Buffer("NPY", a.size, b.dtype)
        src._handle = a.view(np.uint8)   # host staging buffer: the NPY allocator's handles are numpy byte arrays
      sched.append(ScheduleItem(None, (bufs[id(b)], src), {}, "copy"))
      continue
    if bufs[id(b)].size == 0: continue   # nothing to compute; the Buffer is allocated on first use
    if (m := g.custom.get(id(b))) is not None:
      ins = tuple(s.base.realized if s.base.realized is not None else bufs[id(s.base)] for s in m.inputs)
      sched.append(ScheduleItem((CustomCall(m.kernel, m.params),), (bufs[id(b)],) + ins, {}, "custom"))
      continue
    k = _Kernel(g, real, bufs, b)
    sched.append(ScheduleItem(k.ast, tuple(k.bufs), {v: var_vals[v] for v in _ast_vars(k.ast)}, "kernel"))
  for b in g.order:
    if id(b) in real: b.realized = bufs[id(b)]
    if b.op is LoadOps.COPY and not b.srcs: b._arg = None   # the copy item holds the host array; a realized base never
    # reads it again, and keeping it held a host copy of every weight for the model's lifetime (a 70B TP rank: ~60 GB)
  if (dbg := getenv("FZ_DEBUG", 0)) >= 2:
    print(f"schedule: {len(sched)} items ({sum(si.kind == 'kernel' for si in sched)} kernels)")
    for i, si in enumerate(sched):
      ins = ", ".join(f"{x.size}{x.dtype.name[0]}{x.dtype.itemsize*8}" for x in si.bufs[1:])
      ops = "" if si.ast is None else si.ast[0].kernel.name if si.kind == "custom" else " ".join(dict.fromkeys(n.op.name for n in si.ast[0].walk() if not isinstance(n.op, BufferOps)))
      print(f"  {i:3d} {si.kind:6s} {si.bufs[0]} <- [{ins}] {ops}")
      if dbg >= 3 and si.ast is not None:
        for r in si.ast: print(r.pretty(2))
  return sched

# ---------------------------------------------------------------- custom kernels (SPEC §11.1)

@dataclass(eq=False)
class CustomKernel:
  """A library kernel. src_per_device: device base name -> C-like source defining `name` (compiled devices: called as
  name(*bufs, *params), every param an int) or a python callable fn(arrays, params) (NPY: arrays are the flat value
  arrays of the bufs, outputs first). signature: the buffer names then the int param names."""
  name: str
  src_per_device: dict[str, Any]
  signature: tuple[str, ...]
  nbufs: int = 0
  dtypes: tuple = field(default_factory=lambda: (dtypes.float32,))
  accept: Any = None   # optional fn(params) -> bool: False leaves that match to the composed kernels (e.g. a device limit)

@dataclass(frozen=True, eq=False)
class CustomCall:
  """The `ast` payload of a kind="custom" ScheduleItem (ScheduleItem has no field for it, see docs/status/P2-CUSTOM.md)."""
  kernel: CustomKernel
  params: tuple[int, ...]
  def pretty(self, indent: int = 0) -> str: return f"{'  ' * indent}CUSTOM {self.kernel.name} {self.params}"

@dataclass
class _Match:
  kernel: CustomKernel
  inputs: tuple[LazyBuffer, ...]
  params: tuple[int, ...]

def _f32_bits(x: float) -> int: return struct.unpack("<i", struct.pack("<f", x))[0]
def _bits_f32(x: int) -> float: return struct.unpack("<f", struct.pack("<i", x))[0]

def _sview(lb: LazyBuffer) -> tuple[tuple[int, ...], int] | None:
  """(strides, offset) of lb over its base's row-major buffer, if that is one unmasked int view."""
  st = lb.st.simplify()
  if len(st.views) != 1 or (v := st.views[0]).mask is not None: return None
  if not all(isinstance(x, int) for x in v.shape + v.strides + (v.offset,)): return None
  return v.strides, v.offset

def _ident(lb: LazyBuffer) -> bool:
  """lb reads its base element for element in row-major order."""
  return lb is lb.base or (lb.st.contiguous and lb.size == lb.base.size)

def _fused_op(lb: LazyBuffer, op) -> LazyBuffer | None:
  b = lb.base
  return b if b.op is op and b.realized is None else None

def _collapse(F: tuple[int, ...], axes: list[int], strides: tuple[int, ...]) -> tuple[int, int] | None:
  """(size, stride) of a group of axes indexed by one row-major counter, or None if their strides do not chain."""
  for a, c in zip(axes, axes[1:]):
    if strides[a] != strides[c] * F[c]: return None
  return prod([F[a] for a in axes]), (strides[axes[-1]] if axes else 0)

def _mm_spec(b: LazyBuffer, strict: bool = True) -> tuple[LazyBuffer, LazyBuffer, tuple[int, ...]] | None:
  """b = SUM(MUL(x, y)) as a batched GEMM: (x, y, (Bt, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, oa, ob)).
  x/y strides are over their bases' row-major buffers, the out strides over b's keepdim shape."""
  if b.op is not ReduceOps.SUM or b.realized is not None or len(b.arg) != 1 or b.dtype != dtypes.float32: return None
  m0 = b.srcs[0]
  if not _ident(m0) or (m := _fused_op(m0, BinaryOps.MUL)) is None or m.dtype != dtypes.float32: return None
  if tuple(m.shape) != tuple(m0.shape): return None   # SUM over a reshape of the MUL: b.arg indexes the reshaped axes
  F, k = m.shape, b.arg[0]
  if not all(isinstance(s, int) and s > 0 for s in F): return None
  x, y = m.srcs
  if any(s.base.realized is None and s.base.op is LoadOps.CONST for s in (x, y)): return None
  if (vx := _sview(x)) is None or (vy := _sview(y)) is None: return None
  (sx, ox), (sy, oy) = vx, vy
  R = tuple(1 if i == k else s for i, s in enumerate(F))
  sc = _canon(R, _row_major(R))
  grp: dict[str, list[int]] = {"b": [], "m": [], "n": []}
  for i, s in enumerate(F):
    if i != k and s != 1: grp["m" if sy[i] == 0 and sx[i] != 0 else "n" if sx[i] == 0 and sy[i] != 0 else "b"].append(i)
  out: list[int] = []
  for g, ts in (("b", (sx, sy, sc)), ("m", (sx, sc)), ("n", (sy, sc))):
    cs = [_collapse(F, grp[g], t) for t in ts]
    if any(c is None for c in cs): return None
    out.append(cs[0][0])
    grp[g] = [c[1] for c in cs]  # type: ignore[index]
  if strict and out[1] == 1 and out[2] == 1: return None  # no operand is broadcast: a fused dot/sum, not a matmul
  (sab, sbb, scb), (sam, scm), (sbn, scn) = grp["b"], grp["m"], grp["n"]
  return x, y, (*out, F[k], sab, sam, sx[k], sbb, sy[k], sbn, scb, scm, scn, ox, oy)

def _canon(shape: tuple[int, ...], strides: tuple[int, ...]) -> tuple[int, ...]: return tuple(0 if s == 1 else st for s, st in zip(shape, strides))
def _row_major(shape: tuple[int, ...]) -> tuple[int, ...]:
  out, acc = [], 1
  for s in reversed(shape):
    out.append(acc)
    acc *= s
  return tuple(reversed(out))

def _levels(lv: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
  """(size, stride) index levels, outer first, merged where one row-major counter covers them; at most 2, padded to 2."""
  out: list[tuple[int, int]] = []
  for n, st in lv:
    if n == 1: continue
    if out and out[-1][1] == st * n: out[-1] = (out[-1][0] * n, st)
    else: out.append((n, st))
  return None if len(out) > 2 else [(1, 0)] * (2 - len(out)) + out

def _mask_params(P: LazyBuffer, Bt: int, mask: LazyBuffer) -> tuple[int, ...] | None:
  """A bool mask over P's shape -> (mb1, mb2, smb1, smb2, mr1, mr2, smr1, smr2, sms, om): its strides as two batch levels
  (b = b1*mb2 + b2) and two query-row levels (r = r1*mr2 + r2), the key stride and the offset."""
  if (sv := _sview(mask)) is None: return None
  ms, om = sv
  acc, nb = 1, 0
  while acc < Bt and nb < len(P.shape) - 1: acc, nb = acc * P.shape[nb], nb + 1   # P's leading axes are the batch, then the rows
  if acc != Bt: return None
  b = _levels([(P.shape[i], ms[i]) for i in range(nb)])
  r = _levels([(P.shape[i], ms[i]) for i in range(nb, len(P.shape) - 1)])
  # NEURON's renderer DMAs the mask rows with an ascending access pattern: a negative row stride (a flipped mask) stays composed
  if b is None or r is None or min(r[0][1], r[1][1]) < 0: return None
  return (b[0][0], b[1][0], b[0][1], b[1][1], r[0][0], r[1][0], r[0][1], r[1][1], ms[-1], om)

def _match_attention(o: LazyBuffer, ker: CustomKernel, masked: bool = False) -> _Match | None:
  """o = SUM(MUL(P, V)), P = E * RECIP(SUM(E)), E = EXP2((T + NEG(MAX(T))) * log2(e)), T = MM [* c], MM = SUM(MUL(Q, K)).
  masked (the "attention_mask" pattern, offered only by devices that list it): T = WHERE(mask, MM [* c], -inf) with a bool
  mask over the scores (is_causal, a bool attn_mask, a position mask); the mask becomes a 4th input and its strides
  (_mask_params) are appended to the params."""
  if (mm2 := _mm_spec(o, False)) is None: return None
  pv, v, (Bt, L, Dv, S, spb, spl, sps, svb, svs, svd, sob, sol, sod, op_, ov) = mm2
  P = pv.base
  if P.realized is not None or P.op is not BinaryOps.MUL or P.dtype != dtypes.float32 or op_ != 0: return None
  n = len(P.shape)
  if n < 2 or prod(P.shape) != Bt * L * S or P.shape[-1] != S: return None
  if any(sz > 1 and st != want for sz, st, want in ((Bt, spb, L * S), (L, spl, S), (S, sps, 1))): return None
  rshape = P.shape[:-1] + (1,)
  bcast_st = (_canon(P.shape, _row_major(rshape)[:-1] + (0,)), 0)
  def bcast(lb: LazyBuffer) -> bool: return lb.shape == P.shape and lb.base.shape == rshape and _sview(lb) == bcast_st
  def unwrap(lb: LazyBuffer, uop, rop) -> LazyBuffer | None:
    """lb = uop(rop(X, last axis, keepdim)) broadcast back to P's shape (the broadcast before or after uop); returns X."""
    if bcast(lb) and (u := _fused_op(lb, uop)) is not None and _ident(u.srcs[0]): r = u.srcs[0]
    elif _ident(lb) and lb.shape == P.shape and (u := _fused_op(lb, uop)) is not None and bcast(u.srcs[0]): r = u.srcs[0]
    else: return None
    if (rb := _fused_op(r, rop)) is None or rb.arg != (n - 1,) or rb.shape != rshape or not _ident(rb.srcs[0]): return None
    return rb.srcs[0].base
  # P = E * RECIP(SUM(E))
  a, c = P.srcs
  if (E := unwrap(c, UnaryOps.RECIP, ReduceOps.SUM)) is None: a, c = c, a
  if (E := unwrap(c, UnaryOps.RECIP, ReduceOps.SUM)) is None or not _ident(a) or a.base is not E or a.shape != P.shape: return None
  if E.op is not UnaryOps.EXP2 or E.realized is not None: return None
  # E = EXP2(MUL(ADD(T, NEG(MAX(T))), log2e))
  if not _ident(E.srcs[0]) or (ml := _fused_op(E.srcs[0], BinaryOps.MUL)) is None: return None
  def split_const(b: LazyBuffer) -> tuple[LazyBuffer, float] | None:
    for u, w in (b.srcs, b.srcs[::-1]):
      # a masked (padded) const view reads 0 outside its mask: only an unmasked view is a uniform scalar
      if w.base.op is LoadOps.CONST and w.base.realized is None and isinstance(w.base.arg, (int, float)) and not _has_mask(w.st) and _ident(u):
        return u, float(w.base.arg)
    return None
  if (sp := split_const(ml)) is None or sp[1] != math.log2(math.e) or (ad := _fused_op(sp[0], BinaryOps.ADD)) is None: return None
  t, ng = ad.srcs
  if (T := unwrap(ng, UnaryOps.NEG, ReduceOps.MAX)) is None: t, ng = ng, t
  if (T := unwrap(ng, UnaryOps.NEG, ReduceOps.MAX)) is None or not _ident(t) or t.base is not T or T.realized is not None: return None
  mask_in: tuple[LazyBuffer, ...] = ()
  if masked:   # T = WHERE(mask, X, -inf): the scores X are what the composed graph scales
    if T.op is not TernaryOps.WHERE or len(T.srcs) != 3: return None
    c, x, y = T.srcs
    if not (y.base.op is LoadOps.CONST and y.base.realized is None and y.base.arg == -math.inf and not _has_mask(y.st)): return None
    if c.dtype != dtypes.bool or (c.base.op is LoadOps.CONST and c.base.realized is None) or not _ident(x) or x.shape != P.shape: return None
    if (mp := _mask_params(P, Bt, c)) is None or (T := x.base).realized is not None: return None
    mask_in = (c,)
  # T = MM * c, or MM itself
  scale, mmb = 1.0, T
  if T.op is BinaryOps.MUL and (sp := split_const(T)) is not None: mmb, scale = sp[0].base, sp[1]
  if mmb.realized is not None or (mm1 := _mm_spec(mmb, False)) is None or mmb.size != P.size: return None
  q, kk, (Bt1, L1, S1, D, sqb, sql, sqd, skb, skd, sks, scb, scl, scs, oq, ok) = mm1
  if (Bt1, L1, S1) != (Bt, L, S) or any(sz > 1 and st != want for sz, st, want in ((Bt, scb, L * S), (L, scl, S), (S, scs, 1))): return None
  if any(s.dtype != dtypes.float32 for s in (q, kk, v)): return None
  params = (Bt, L, S, D, Dv, sqb, sql, sqd, skb, sks, skd, svb, svs, svd, sob, sol, sod, oq, ok, ov, _f32_bits(scale), _f32_bits(math.log2(math.e)))
  return _Match(ker, (q, kk, v) + mask_in, params + (mp if masked else ()))

def _device_kernels(device: str) -> dict[str, CustomKernel]:
  from fusezap.device import Device
  if not getenv("FZ_CUSTOM", 1): return {}
  try: ks = getattr(Device[device], "custom_kernels", None) or {}
  except Exception: return {}
  base = device.split(":")[0].upper()
  return {p: k for p, k in ks.items() if base in k.src_per_device}

def _lin_bounds(valid: Any, vs: list[Var], Wo: int) -> list[tuple[int, int]] | None:
  """valid (an AND of `<` comparisons) as a box over vs = [c, t, ho, wo]: per var [lo, hi), or None if it is not one.
  Besides single-variable bounds it accepts bounds on ho*Wo + wo (the flattened pixel) whose limit is a multiple of Wo:
  with 0 <= wo < Wo they bound ho alone."""
  from fusezap.shape.symbolic import LtNode, AndNode
  from fusezap.shape.view import _linear
  box = [(v.vmin, v.vmax + 1) for v in vs]
  if valid == 0: return None
  if isinstance(valid, int): return box
  for cn in (valid.nodes if isinstance(valid, AndNode) else (valid,)):
    if not isinstance(cn, LtNode): return None
    if isinstance(cn.b, int): expr, lo, hi = cn.a, None, cn.b          # expr < b
    elif isinstance(cn.a, int): expr, lo, hi = cn.b, cn.a + 1, None    # expr >= a + 1
    else: return None
    if (lin := _linear(expr, vs)) is None: return None
    co, k0 = lin
    nz = [i for i, x in enumerate(co) if x]
    if len(nz) == 1:
      i, a = nz[0], co[nz[0]]
      if a != 1: return None
    elif nz == [2, 3] and co[3] == 1 and co[2] == Wo:
      i, a = 2, Wo
    else: return None
    if hi is not None:
      B = hi - k0
      if B % a: return None
      box[i] = (box[i][0], min(box[i][1], B // a))
    else:
      A = lo - k0   # expr >= A
      if A % a: return None
      box[i] = (max(box[i][0], A // a), box[i][1])
  return box

def _conv_leaves(cb: LazyBuffer) -> list[LazyBuffer] | None:
  """The views an unrealized ADD tree sums (Tensor.cat / stack), each of cb's shape."""
  out: list[LazyBuffer] = []
  def walk(b: LazyBuffer) -> bool:
    for sr in b.srcs:
      if sr.shape != cb.shape: return False
      if sr.base is not sr and sr.base.op is BinaryOps.ADD and sr.base.realized is None and _ident(sr): ok = walk(sr.base)
      elif sr.base is sr and sr.op is BinaryOps.ADD and sr.realized is None: ok = walk(sr)
      else: out.append(sr); ok = True
      if not ok: return False
    return True
  return out if walk(cb) else None

def _match_conv(b: LazyBuffer, ker: CustomKernel) -> _Match | None:
  """b = SUM(MUL(weight, col)) (a GEMM, _mm_spec) where col is the im2col of one materialized input X, as Tensor.conv2d
  composes it: col = stack([v_0 .. v_{T-1}], axis=1).reshape(C*T, P), v_t = the (C, P) window view of PAD(X) for tap t =
  (i, j) (row c*T + t of col), P = Ho*Wo pixels. Proof obligation, checked exactly on the symbolic index expressions: for
  every leaf of the ADD tree (one per tap), with Vars c, t, ho, wo, leaf.st.expr_idxs([c, t, ho*Wo + wo]) must be
  (a) valid only at t == t0 (distinct taps 0..T-1), and for t = t0 (b) the buffer index c*H*W + (ho*sh + i*dh - ph)*W +
  wo*sw + j*dw - pw and (c) valid exactly on the box 0 <= ho*sh + i*dh - ph < H, 0 <= wo*sw + j*dw - pw < W (within
  ho < Ho, wo < Wo), with (i, j) = divmod(t0, kw). Then col[c*T + t, ho*Wo + wo] is X[c, ho*sh + i*dh - ph, wo*sw +
  j*dw - pw] or 0, which is what the kernel reads. Anything else (other layouts, a non-contiguous X) stays composed."""
  if (mm := _mm_spec(b)) is None: return None
  x, y, (Bt, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, oa, ob) = mm
  if Bt != 1: return None
  # col as the B operand (out[o, p]) or as A (out[p, o]): normalize to weight w[o, k] and col[k, p]
  for (w, col, Mw, Np, swo, swk, so, sp, ow, sck, scp, oc) in ((x, y, M, N, sam, sak, scm, scn, oa, sbk, sbn, ob), (y, x, N, M, sbn, sbk, scn, scm, ob, sak, sam, oa)):
    cb = col.base
    if cb.realized is not None or cb.op is not BinaryOps.ADD or len(cb.shape) != 3 or oc != 0: continue
    C, T, P = cb.shape
    if P != Np or C * T != K or T < 2 or (sck, scp) != (P, 1): continue
    if (leaves := _conv_leaves(cb)) is None or len(leaves) != T: continue
    X = leaves[0].base
    if any(l.base is not X for l in leaves) or X is cb: continue
    xs = tuple(int(s) for s in X.shape) if all(isinstance(s, int) for s in X.shape) else None
    if xs is None or len(xs) < 3 or prod(xs[:-3]) != 1 or xs[-3] != C: continue
    H, W = xs[-2], xs[-1]
    cands = sorted({v.shape[-1] for l in leaves for v in l.st.views if v.shape and isinstance(v.shape[-1], int) and v.shape[-1] > 0 and P % v.shape[-1] == 0})
    for Wo in cands:
      if (r := _conv_geometry(leaves, C, T, P, H, W, Wo)) is not None:
        kh, kw, sh, sw, dh, dw, ph, pw = r
        params = (Mw, P, K, swo, swk, ow, so, sp, C, H, W, Wo, kh, kw, sh, sw, dh, dw, ph, pw)
        return _Match(ker, (w, X), params)
  return None

def _conv_geometry(leaves: list[LazyBuffer], C: int, T: int, P: int, H: int, W: int, Wo: int) -> tuple[int, ...] | None:
  """(kh, kw, sh, sw, dh, dw, ph, pw) if every leaf is exactly its tap's window of X (see _match_conv), else None."""
  from fusezap.shape.view import _linear
  Ho = P // Wo
  cv, tv, hv, wv = Var("_cc", 0, C - 1), Var("_ct", 0, T - 1), Var("_ch", 0, Ho - 1), Var("_cw", 0, Wo - 1)
  vs = [cv, tv, hv, wv]
  taps: dict[int, tuple[list[int], int, list[tuple[int, int]]]] = {}
  for l in leaves:
    idx, valid = l.st.expr_idxs([cv, tv, hv * Wo + wv])
    if (box := _lin_bounds(valid, vs, Wo)) is None: return None
    t0 = box[1][0]
    if box[1][1] != t0 + 1 or t0 in taps or box[0] != (0, C): return None
    idx, valid = l.st.expr_idxs([cv, t0, hv * Wo + wv])   # the index at its tap
    if (lin := _linear(idx, vs)) is None or (box := _lin_bounds(valid, vs, Wo)) is None or lin[0][1] != 0: return None
    taps[t0] = (lin[0], lin[1], [box[0], (t0, t0 + 1)] + box[2:])
  if sorted(taps) != list(range(T)): return None
  co0, k00, _ = taps[0]
  if co0[0] != H * W or co0[2] % W or co0[2] <= 0 or co0[3] <= 0 or k00 > 0: return None
  sh, sw = co0[2] // W, co0[3]
  ph, pw = (-k00) // W, (-k00) % W
  for kh in range(1, T + 1):
    if T % kh: continue
    kw = T // kh
    dw = taps[1][1] - k00 if kw > 1 else 1
    dh = (taps[kw][1] - k00) // W if kh > 1 else 1
    if dw <= 0 or dh <= 0: continue
    ok = True
    for t, (co, k0, box) in taps.items():
      i, j = divmod(t, kw)
      oh, ow = i * dh - ph, j * dw - pw     # input row = ho*sh + oh, col = wo*sw + ow
      want = ([H * W, 0, sh * W, sw], oh * W + ow,
              [(0, C), (t, t + 1), (max(0, -(oh // sh) if oh < 0 else 0), min(Ho, -(-(H - oh) // sh))),
               (max(0, -(ow // sw) if ow < 0 else 0), min(Wo, -(-(W - ow) // sw)))])
      if (co, k0, box) != (want[0], want[1], want[2]): ok = False; break
    if ok: return kh, kw, sh, sw, dh, dw, ph, pw
  return None

def _match_custom(b: LazyBuffer) -> _Match | None:
  if b.op is not ReduceOps.SUM or not (ks := _device_kernels(b.device)): return None
  def ok(m: _Match | None) -> bool: return m is not None and (m.kernel.accept is None or bool(m.kernel.accept(m.params)))
  if "conv" in ks and ok(m := _match_conv(b, ks["conv"])): return m
  if "attention" in ks and ok(m := _match_attention(b, ks["attention"])): return m
  if "attention_mask" in ks and ok(m := _match_attention(b, ks["attention_mask"], True)): return m
  if "matmul" in ks and (mm := _mm_spec(b)) is not None: return _Match(ks["matmul"], mm[:2], mm[2])
  return None

# ---- reference kernels. Each repeats the composed op order (SUM accumulates k = 0..K-1 from 0; MAX keeps NaN), so it
# ---- equals the composed kernels of the same device.

def _np_strided(a: np.ndarray, off: int, shape: tuple[int, ...], strides: tuple[int, ...]) -> np.ndarray:
  if prod(shape) == 0: return np.zeros(shape, a.dtype)
  return np.lib.stride_tricks.as_strided(a[off:], shape, tuple(s * a.itemsize for s in strides), writeable=False)

def _np_mm(A: np.ndarray, Bm: np.ndarray) -> np.ndarray:
  """(Bt, M, K) x (Bt, K, N) -> (Bt, M, N), summed like the composed (M, K, N) product reduced over K."""
  return np.sum(A[:, :, :, None] * Bm[:, None, :, :], axis=2, dtype=np.float32)

def _np_store(C: np.ndarray, val: np.ndarray, strides: tuple[int, ...]) -> None:
  idx = sum(np.arange(s).reshape([-1 if j == i else 1 for j in range(3)]) * st for i, (s, st) in enumerate(zip(val.shape, strides)))
  C[np.broadcast_to(idx, val.shape).reshape(-1)] = val.reshape(-1)

def _npy_matmul(arrs: list[np.ndarray], p: tuple[int, ...]) -> None:
  C, A, B = arrs
  Bt, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, oa, ob = p
  a, b = _np_strided(A, oa, (Bt, M, K), (sab, sam, sak)), _np_strided(B, ob, (Bt, K, N), (sbb, sbk, sbn))
  _np_store(C, _np_mm(a.astype(np.float32), b.astype(np.float32)), (scb, scm, scn))

def _npy_attention(arrs: list[np.ndarray], p: tuple[int, ...]) -> None:
  O, Q, K, V = arrs
  Bt, L, S, D, Dv, sqb, sql, sqd, skb, sks, skd, svb, svs, svd, sob, sol, sod, oq, ok, ov, scale, log2e = p
  q, k = _np_strided(Q, oq, (Bt, L, D), (sqb, sql, sqd)), _np_strided(K, ok, (Bt, D, S), (skb, skd, sks))
  v = _np_strided(V, ov, (Bt, S, Dv), (svb, svs, svd))
  f = np.float32
  with np.errstate(all="ignore"):
    t = _np_mm(q, k) * f(_bits_f32(scale))
    e = np.exp2((t + -np.max(t, axis=2, keepdims=True)) * f(_bits_f32(log2e)))
    pr = e * np.reciprocal(np.sum(e, axis=2, keepdims=True, dtype=f))
    _np_store(O, _np_mm(pr, v), (sob, sol, sod))

_C_MATMUL = r"""#include <stdlib.h>
void fz_matmul(float* restrict C, const float* A, const float* B, int Bt, int M, int N, int K, int sab, int sam, int sak,
               int sbb, int sbk, int sbn, int scb, int scm, int scn, int oa, int ob) {
  float* acc = (float*)malloc(sizeof(float) * (N > 0 ? N : 1));
  for (long b = 0; b < Bt; b++) for (long m = 0; m < M; m++) {
    const float* a = A + oa + b*sab + m*sam;
    const float* bb = B + ob + b*sbb;
    for (long n = 0; n < N; n++) acc[n] = 0.0f;
    for (long k = 0; k < K; k++) {   /* k-outer, n-inner: each acc[n] still sums k = 0..K-1 in order */
      float av = a[k*sak];
      const float* brow = bb + k*sbk;
      for (long n = 0; n < N; n++) acc[n] = acc[n] + av*brow[n*sbn];
    }
    float* c = C + b*scb + m*scm;
    for (long n = 0; n < N; n++) c[n*scn] = acc[n];
  }
  free(acc);
}
"""

_C_ATTENTION = r"""#include <math.h>
#include <stdlib.h>
#include <string.h>
static float fz_f32(int bits) { float f; memcpy(&f, &bits, 4); return f; }
void fz_attention(float* restrict O, const float* Q, const float* Kb, const float* V, int Bt, int L, int S, int D, int Dv,
                  int sqb, int sql, int sqd, int skb, int sks, int skd, int svb, int svs, int svd, int sob, int sol, int sod,
                  int oq, int ok, int ov, int scale_bits, int log2e_bits) {
  const float scale = fz_f32(scale_bits), log2e = fz_f32(log2e_bits);
  float* p = (float*)malloc(sizeof(float) * (S > 0 ? S : 1));
  float* acc = (float*)malloc(sizeof(float) * (Dv > 0 ? Dv : 1));
  for (long b = 0; b < Bt; b++) for (long l = 0; l < L; l++) {
    const float* q = Q + oq + b*sqb + l*sql;
    float mx = -INFINITY;
    for (long s = 0; s < S; s++) {
      const float* k = Kb + ok + b*skb + s*sks;
      float t = 0.0f;
      for (long d = 0; d < D; d++) t = t + q[d*sqd]*k[d*skd];
      t = t*scale;
      p[s] = t;
      mx = (t == t) ? ((t > mx) ? t : mx) : t;   /* NaN wins, like the composed MAX */
    }
    float den = 0.0f;
    for (long s = 0; s < S; s++) { float e = exp2f((p[s] + (-mx))*log2e); p[s] = e; den = den + e; }
    float r = (1.0f/den);
    for (long d = 0; d < Dv; d++) acc[d] = 0.0f;
    for (long s = 0; s < S; s++) {
      float ps = p[s]*r;
      const float* v = V + ov + b*svb + s*svs;
      for (long d = 0; d < Dv; d++) acc[d] = acc[d] + ps*v[d*svd];
    }
    float* o = O + b*sob + l*sol;
    for (long d = 0; d < Dv; d++) o[d*sod] = acc[d];
  }
  free(p); free(acc);
}
"""

MATMUL = CustomKernel("fz_matmul", {"NPY": _npy_matmul, "CPU": _C_MATMUL},
                      ("C", "A", "B", "Bt", "M", "N", "K", "sab", "sam", "sak", "sbb", "sbk", "sbn", "scb", "scm", "scn", "oa", "ob"), 3)
ATTENTION = CustomKernel("fz_attention", {"NPY": _npy_attention, "CPU": _C_ATTENTION},
                         ("O", "Q", "K", "V", "Bt", "L", "S", "D", "Dv", "sqb", "sql", "sqd", "skb", "sks", "skd", "svb", "svs", "svd",
                          "sob", "sol", "sod", "oq", "ok", "ov", "scale_bits", "log2e_bits"), 4)
REFERENCE_KERNELS: dict[str, CustomKernel] = {"matmul": MATMUL, "attention": ATTENTION}

CONV_PARAMS = ("M", "N", "K", "sam", "sak", "oa", "so", "sp", "C", "H", "W", "Wo", "kh", "kw", "sh", "sw", "dh", "dw", "ph", "pw")

def _npy_conv(arrs: list[np.ndarray], p: tuple[int, ...]) -> None:
  """Reference "conv" kernel (_match_conv): out[o*so + q*sp] = sum_k w[oa + o*sam + k*sak] * col[k, q], col the im2col
  of X (C, H, W), summed over k = 0..K-1 like the composed matmul."""
  out, w, x = arrs
  M, N, K, sam, sak, oa, so, sp, C, H, W, Wo, kh, kw, sh, sw, dh, dw, ph, pw = p
  X = x[:C * H * W].reshape(C, H, W)
  Ho = N // Wo
  col = np.zeros((C, kh * kw, Ho, Wo), np.float32)
  for i in range(kh):
    for j in range(kw):
      hs, ws = np.arange(Ho) * sh + i * dh - ph, np.arange(Wo) * sw + j * dw - pw
      hi, wi = np.flatnonzero((hs >= 0) & (hs < H)), np.flatnonzero((ws >= 0) & (ws < W))
      col[:, i * kw + j, hi[:, None], wi[None, :]] = X[:, hs[hi][:, None], ws[wi][None, :]]
  A = _np_strided(w, oa, (1, M, K), (0, sam, sak))
  val = _np_mm(A, col.reshape(1, K, N))
  _np_store(out, val, (0, so, sp))

CONV = CustomKernel("fz_conv", {"NPY": _npy_conv}, ("C", "A", "X") + CONV_PARAMS, 3)
REFERENCE_KERNELS["conv"] = CONV   # NPY's reference; METAL offers its own (fusezap/device/metal.py)

def run_custom(device: Any, si: ScheduleItem) -> float | None:
  """Execute a kind="custom" item on `device` (NPY: python callable; compiled devices: compile + launch the source)."""
  import time
  assert si.kind == "custom" and si.ast is not None and isinstance(call := si.ast[0], CustomCall), f"not a custom item: {si.kind}"
  ker, base = call.kernel, device.name.split(":")[0].upper()
  src = ker.src_per_device[base]
  st = time.perf_counter()
  if callable(src):
    from fusezap.interpret import from_host, to_host
    arrs = [b.ensure_allocated().handle[:b.nbytes].view(to_np(b.dtype)) if b.size else np.zeros(0, to_np(b.dtype)) for b in si.bufs]
    vals = [from_host(a, b.dtype) for a, b in zip(arrs, si.bufs)]
    src(vals, call.params)
    for o in range(si.n_outputs):
      if vals[o] is not arrs[o]: arrs[o][:] = to_host(vals[o], si.bufs[o].dtype)
  else:
    # cache on the device instance (dies with it) keyed by name and source, so a replaced kernel recompiles
    if (progs := device.__dict__.get("_fz_progs")) is None: device._fz_progs = progs = {}
    if (prg := progs.get((ker.name, src))) is None: progs[(ker.name, src)] = prg = device.runtime(ker.name, device.compiler.compile_cached(src))
    prg(*[b.ensure_allocated().handle for b in si.bufs], global_size=(1,), local_size=None, vals=call.params)
  return time.perf_counter() - st

def install_custom_kernels(device: Any, kernels: dict[str, CustomKernel] | None = None) -> None:
  """Offer `kernels` (default REFERENCE_KERNELS) on a device instance and route its kind="custom" items to run_custom.
  Stopgap for compiled devices until CompiledDevice.run dispatches kind="custom" itself."""
  device.custom_kernels = REFERENCE_KERNELS if kernels is None else kernels
  if getattr(device.run, "_fz_custom", False): return
  orig = device.run
  def run(si: ScheduleItem) -> float | None: return run_custom(device, si) if si.kind == "custom" else orig(si)
  run._fz_custom = True  # type: ignore[attr-defined]
  device.run = run
