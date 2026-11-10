"""Kernel AST -> UOps -> ProgramSpec, plus the shared compiled-device run path (SPEC §7.2). Implemented by WP4.

  lower(ast, renderer, opts=None) -> ProgramSpec
      builds the UOp list (fusezap/codegen/uops.py), calls renderer.render(name, uops),
      and fills global/local sizes and the var order
  run_compiled(device, si) -> float | None
      cache key: (canonical ast, renderer type, device name). Lowers, compiles (device.compiler.compile_cached), builds
      device.runtime(name, lib), and launches it with the bufs' handles, evaluated global size and var values.
      FZ_DEBUG>=4 prints the generated source.

Lowering strategy (MVP, Opts ignored):
  F = kernel (reduce input) shape, R = STORE shape. Output axes (R[i] != 1) are grouped onto at most 3 SPECIAL dims
  (gidx0 = innermost group; outer groups are merged first, and a group above renderer.global_max is merged inward).
  A GPU renderer (has_local) additionally splits gidx0 into gidx0*L + lidx0 with L the largest divisor of its size <= 256.
  Every reduce axis (R[i] == 1 != F[i]) is one RANGE loop; the reduce is a DEFIFZ_ACC updated with PHI inside the loops
  (SUM/MAX of float16/bfloat16 accumulate in float32, like the interpreter; float MAX propagates NaN like np.max).
  Each axis gets a symbolic Var; LOAD/CONST/STORE index and validity come from ShapeTracker.expr_idxs only, and the
  resulting Node trees are lowered to int32 ALU UOps, int64 where a value can leave int32 (CSE'd per loop scope). Masked LOADs are guarded (alt 0), masked
  CONSTs become WHERE(valid, c, 0). Epilogue leaves are read with the reduced axes pinned to 0.
  Value IDIV/MOD return 0 for a zero divisor (interpreter semantics) instead of trapping.
"""
from __future__ import annotations
import copy, dataclasses, itertools, json, math, os, re
import numpy as np
from dataclasses import dataclass
from typing import Any
from fusezap.codegen.uops import ProgramSpec, UOp, UOps, PtrDType
from fusezap.dtype import DType, dtypes, to_np
from fusezap.helpers import getenv, prod
from fusezap.ir import LazyOp, ConstBuffer
from fusezap.ops import BufferOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps
from fusezap.shape.symbolic import Node, Var, NumNode, MulNode, DivNode, ModNode, LtNode, SumNode, AndNode, sym_infer

def _fits32(lo: Any, hi: Any) -> bool: return -2**31 <= lo and hi <= 2**31 - 1

@dataclass(frozen=True)
class Opts:
  """Kernel optimizations (P2-OPT). Axes index the kernel's full shape F (rank of every st in the AST).
  An empty Opts() is the unoptimized MVP lowering."""
  upcast: tuple[tuple[int, int], ...] = ()        # (output axis, amount): each thread computes `amount` adjacent outputs
  unroll: tuple[tuple[int, int], ...] = ()        # (reduce axis, amount): reduce loop body repeated `amount` times
  local: tuple[tuple[int, int], ...] = ()         # (output axis, amount): a threadgroup dim of `amount` (GPU; lidx in order)
  group_reduce: int = 0                           # 32: one SIMD group / warp splits the innermost reduce axis (METAL, CUDA)

def _vmax(x) -> int: return x if isinstance(x, int) else x.vmax
def _is1(x) -> bool: return isinstance(x, int) and x == 1
def _min_val(dt: DType) -> Any:
  if dt.is_float: return -math.inf
  if dt == dtypes.bool: return False
  if dt == dtypes.uint8: return 0
  return -(1 << (8 * dt.itemsize - 1))

class _Lowerer:
  def __init__(self, ast: tuple[LazyOp, ...], renderer: Any, opts: Opts | None = None):
    self.ast, self.renderer, self.opts = ast, renderer, opts or Opts()
    self.uops: list[UOp] = []
    self.consts: dict[tuple, UOp] = {}
    self.scopes: list[dict] = [{}]     # CSE caches; a new one per open RANGE (locals do not outlive their block)
    self.vmap: dict[Var, UOp] = {}

  # ------------------------------------------------------------ uop helpers
  def emit(self, op: UOps, dtype, src: tuple = (), arg: Any = None) -> UOp:
    self.uops.append(u := UOp(op, dtype, tuple(src), arg))
    return u
  def const(self, val: Any, dt: DType) -> UOp:
    if dt == dtypes.bool: val = bool(val)
    elif dt.is_int: val = int(val)
    else: val = float(val)
    key = (dt, repr(val))
    if key not in self.consts: self.consts[key] = self.emit(UOps.CONST, dt, (), val)
    return self.consts[key]
  def cached(self, key: tuple, fn) -> UOp:
    for sc in reversed(self.scopes):
      if key in sc: return sc[key]
    self.scopes[-1][key] = u = fn()
    return u
  def alu(self, op: Any, dt: DType, *src: UOp) -> UOp:
    return self.cached(("alu", op, dt, tuple(id(s) for s in src)), lambda: self.emit(UOps.ALU, dt, src, op))
  def cast(self, x: UOp, dt: DType) -> UOp:
    if x.dtype == dt: return x
    if x.op is UOps.CONST and not (dt == dtypes.bfloat16 or x.dtype == dtypes.bfloat16):
      v = x.arg
      if dt == dtypes.bool: return self.const(v != 0, dt)
      if dt.is_float: return self.const(float(v), dt)
      if dt == dtypes.int32 and x.dtype != dtypes.bool:  # fold with the interpreter's semantics (numpy trunc / int wrap)
        if x.dtype.is_float:
          with np.errstate(invalid="ignore", over="ignore"): return self.const(int(np.trunc(np.asarray(v, to_np(x.dtype))).astype(np.int32)), dt)
        return self.const(((int(v) + 2**31) % 2**32) - 2**31, dt)
    return self.cached(("cast", dt, id(x)), lambda: self.emit(UOps.CAST, dt, (x,)))

  # ------------------------------------------------------------ index math (symbolic Node -> int32 / int64 / bool UOps)
  # int32 unless a value can leave int32's range: a node whose bounds (vmin, vmax) or, for a sum, whose running partial
  # sums can exceed it is computed in int64 (its operands widened), and so is everything built on it. Indexes into
  # buffers over 2**31 elements (an im2col of a 256-channel 1024^2 conv: 2.4e9) thereby get 64-bit math; every kernel
  # whose indexes fit renders exactly as before.
  def iconst(self, v: int) -> UOp: return self.const(v, dtypes.int32 if _fits32(v, v) else dtypes.int64)
  def idx(self, n: Any) -> UOp:
    if isinstance(n, bool): return self.const(int(n), dtypes.int32)
    if isinstance(n, int): return self.iconst(n)
    if isinstance(n, NumNode): return self.iconst(n.b)
    if n.vmin == n.vmax: return self.iconst(n.vmin)
    if isinstance(n, Var):
      assert n in self.vmap, f"lower: unbound variable {n!r}"
      return self.vmap[n]
    return self.cached(("node", n), lambda: self._narrow(n, self._idx(n)))
  def _narrow(self, n: Node, u: UOp) -> UOp:
    """A node computed in int64 whose own value fits int32 goes back to int32, so 64-bit math stays on the overflowing
    part (e.g. the flat im2col index before its division) and the rest of the kernel does not pay for it."""
    return self.cast(u, dtypes.int32) if u.dtype == dtypes.int64 and _fits32(n.vmin, n.vmax) else u
  def _iop(self, op: Any, wide: bool, *src: UOp) -> UOp:
    dt = dtypes.int64 if wide or any(s.dtype == dtypes.int64 for s in src) else dtypes.int32
    return self.alu(op, dt, *[self._widen(s, dt) for s in src])
  def _widen(self, s: UOp, dt: DType) -> UOp:
    if s.dtype != dt and s.op is UOps.CONST and s.dtype == dtypes.int32: return self.const(int(s.arg), dt)
    return self.cast(s, dt)
  def _idx(self, n: Node) -> UOp:
    wide = not _fits32(n.vmin, n.vmax)
    if isinstance(n, MulNode): return self._iop(BinaryOps.MUL, wide, self.idx(n.a), self.idx(n.b))
    if isinstance(n, DivNode): return self._iop(BinaryOps.IDIV, wide, self.idx(n.a), self.idx(n.b))
    if isinstance(n, ModNode): return self._iop(BinaryOps.MOD, wide, self.idx(n.a), self.idx(n.b))
    if isinstance(n, SumNode):
      lo = hi = 0
      for t in n.nodes:
        tl, th = (t, t) if isinstance(t, int) else (t.vmin, t.vmax)
        lo, hi = lo + min(tl, 0), hi + max(th, 0)
      wide = wide or not _fits32(lo, hi)
      ret = None
      for t in n.nodes: ret = self.idx(t) if ret is None else self._iop(BinaryOps.ADD, wide, ret, self.idx(t))
      return ret
    if isinstance(n, (LtNode, AndNode)): return self.cast(self.bool(n), dtypes.int32)
    raise NotImplementedError(f"lower: cannot lower index node {type(n).__name__}")
  def bool(self, n: Any) -> UOp:
    if isinstance(n, int) or (isinstance(n, Node) and n.vmin == n.vmax): return self.const(int(n if isinstance(n, int) else n.vmin) != 0, dtypes.bool)
    return self.cached(("bool", n), lambda: self._bool(n))
  def _bool(self, n: Node) -> UOp:
    if isinstance(n, LtNode): return self._cmplt(self.idx(n.a), self.idx(n.b))
    if isinstance(n, AndNode):
      ret = None
      for t in n.nodes: ret = self.bool(t) if ret is None else self.alu(TernaryOps.WHERE, dtypes.bool, ret, self.bool(t), self.const(False, dtypes.bool))
      return ret
    return self._cmplt(self.const(0, dtypes.int32), self.idx(n))
  def _cmplt(self, a: UOp, b: UOp) -> UOp:
    if dtypes.int64 in (a.dtype, b.dtype): a, b = self._widen(a, dtypes.int64), self._widen(b, dtypes.int64)
    return self.alu(BinaryOps.CMPLT, dtypes.bool, a, b)

  # ------------------------------------------------------------ kernel structure
  def run(self) -> ProgramSpec:
    ast, r, o = self.ast, self.renderer, self.opts
    assert len(ast) >= 1 and all(s.op is BufferOps.STORE for s in ast), "lower: AST roots must be STOREs"
    nodes = [n for root in ast for n in root.walk()]
    leaves = [n for n in nodes if n.op in (BufferOps.LOAD, BufferOps.CONST)]
    reds = [n for n in nodes if isinstance(n.op, ReduceOps)]
    R = tuple(ast[0].arg.st.shape)
    F = tuple(leaves[0].arg.st.shape) if leaves else R
    assert len(F) == len(R), f"lower: rank mismatch {F} vs {R}"
    self.F, self.R = F, R
    self.raxes = [i for i in range(len(F)) if _is1(R[i]) and not _is1(F[i])]
    check_opts(o, ast, r)
    if o.group_reduce: self.renderer = r = _simd_renderer(r)
    up, loc = dict(o.upcast), dict(o.local)
    # buffers and vars
    bdt: dict[int, DType] = {}
    for n in nodes:
      if n.op in (BufferOps.LOAD, BufferOps.STORE): bdt.setdefault(n.arg.idx, n.arg.dtype)
    outs = {s.arg.idx for s in ast}
    self.n_bufs = max(bdt) + 1
    self.bufs = {i: self.emit(UOps.DEFIFZ_GLOBAL, PtrDType(bdt.get(i, dtypes.uint8)), (), (i, f"data{i}", i in outs)) for i in range(self.n_bufs)}
    vs: set = set()
    for n in nodes:
      if n.op in (BufferOps.LOAD, BufferOps.CONST, BufferOps.STORE): vs |= n.arg.st.vars()
      if n.op is BufferOps.CONST and isinstance(n.arg.val, Node): vs |= n.arg.val.vars()
    self.vars = sorted(vs, key=lambda v: (v.name, v.vmin, v.vmax))
    for v in self.vars: self.vmap[v] = self.emit(UOps.DEFIFZ_VAR, dtypes.int32, (), v)
    # global dims: what is left of each output axis after its upcast (registers) and local (threadgroup) parts
    gaxes = [i for i in range(len(R)) if not _is1(R[i])]
    G = {i: R[i] // (up.get(i, 1) * loc.get(i, 1)) for i in gaxes}
    groups = [[i] for i in gaxes if not _is1(G[i])]
    while len(groups) > 3: groups[:2] = [groups[0] + groups[1]]
    gmax = getattr(r, "global_max", None)
    changed = True
    while changed and gmax is not None:
      changed = False
      for p in range(len(groups) - 1):  # position p has hardware axis len-1-p (> 0)
        ax = len(groups) - 1 - p
        if _vmax(prod([G[i] for i in groups[p]])) > gmax[ax]:
          groups[p:p + 2] = [groups[p] + groups[p + 1]]; changed = True; break
    self.local_size: tuple[int, ...] | None = None
    gpart: list[Any] = [0] * len(R)
    gvars = []
    for p, g in enumerate(groups):
      ax, size = len(groups) - 1 - p, prod([G[i] for i in g])
      gv = Var(f"_fzg{ax}", 0, max(_vmax(size) - 1, 0))
      gvars.append((ax, gv, size))
      inner: Any = 1
      for i in reversed(g):
        gpart[i] = (gv // inner) % G[i] if len(g) > 1 else gv
        inner = inner * G[i]
    self.global_size: list = [1] * len(groups)
    auto_local = getattr(r, "has_local", False) and not o.local and not o.group_reduce
    for ax, gv, size in gvars:  # SPECIALs: outermost first, so CPU loops nest outer -> inner
      if ax == 0 and auto_local and isinstance(size, int) and size > 1:
        L = next(d for d in range(min(size, 256), 0, -1) if size % d == 0)
        gu = self.emit(UOps.SPECIAL, dtypes.int32, (), (0, "gidx0", size // L))
        if L > 1:
          lu = self.emit(UOps.SPECIAL, dtypes.int32, (), (0, "lidx0", L))
          gu = self.alu(BinaryOps.ADD, dtypes.int32, self.alu(BinaryOps.MUL, dtypes.int32, gu, self.const(L, dtypes.int32)), lu)
          self.local_size = (L, 1, 1)
        self.global_size[0] = size // L
        self.vmap[gv] = gu
      else:
        self.vmap[gv] = self.emit(UOps.SPECIAL, dtypes.int32, (), (ax, f"gidx{ax}", size))
        self.global_size[ax] = size
    # local dims: the group-for-reduce lanes (one SIMD group) take lidx0, then the local opts in order
    ldims: list[tuple[Any, int]] = ([("lane", o.group_reduce)] if o.group_reduce else []) + [(i, l) for i, l in o.local]
    lpart: dict[int, Any] = {}
    self.lane: Any = None
    for k, (what, l) in enumerate(ldims):
      lv = Var(f"_fzl{k}", 0, l - 1)
      self.vmap[lv] = self.emit(UOps.SPECIAL, dtypes.int32, (), (k, f"lidx{k}", l))
      if what == "lane": self.lane, self.lane_uop = lv, self.vmap[lv]
      else: lpart[what] = lv
    if ldims: self.local_size = tuple(l for _, l in ldims) + (1,) * (3 - len(ldims))
    if getattr(r, "has_local", False) and self.local_size is None and (groups or ldims): self.local_size = (1, 1, 1)
    # output index per upcast combo: idx_i = (gpart_i * local_i + lidx_i) * upcast_i + k_i
    base = list(gpart)
    for i in gaxes:
      if i in loc: base[i] = base[i] * loc[i] + lpart[i]
    self.up_axes = [i for i, _ in o.upcast]
    self.combos = list(itertools.product(*[range(up[i]) for i in self.up_axes]))
    self.out_idxs: dict[tuple, list] = {}
    for c in self.combos:
      ix = list(base)
      for i, k in zip(self.up_axes, c): ix[i] = ix[i] * up[i] + k
      self.out_idxs[c] = ix
    self.memo: dict = {}
    self.red_val: dict = {}
    for n in reds: self.reduce(n)
    for c in self.combos:
      self.cur, self.cur_red = c, None
      for root in ast: self.store(root)
    reduce_tag = "r" if reds else "E"
    name = "_".join([reduce_tag] + [_ident(R[i]) for i in gaxes] + ([_ident(F[i]) for i in self.raxes] if reds else []))
    src = r.render(name, self.uops)
    return ProgramSpec(name, src, tuple(self.global_size), self.local_size, self.n_bufs, list(self.vars), self.uops)

  def store(self, root: LazyOp) -> None:
    mb = root.arg
    val = self.cast(self.value(root.src[0], False), mb.dtype)
    idx, valid = mb.st.expr_idxs(self.out_idxs[self.cur])
    if isinstance(valid, int) and valid == 0: return
    iu = self.idx(idx)  # lowered before the IF: CSE'd index ALUs must not be declared inside the guarded block
    cond = None if isinstance(valid, int) else self.bool(valid)
    if self.lane is not None:  # group-for-reduce: every lane holds the result, lane 0 stores it
      first = self.alu(BinaryOps.CMPEQ, dtypes.bool, self.lane_uop, self.const(0, dtypes.int32))
      cond = first if cond is None else self.alu(TernaryOps.WHERE, dtypes.bool, first, cond, self.const(False, dtypes.bool))
    guard = None if cond is None else self.emit(UOps.IF, None, (cond,))
    self.emit(UOps.STORE, None, (self.bufs[mb.idx], iu, val))
    if guard is not None: self.emit(UOps.ENDIF, None, (guard,))

  def leaf_idxs(self) -> list[Any]:
    out = list(self.out_idxs[self.cur])
    if self.cur_red is not None:
      for i, v in zip(self.raxes, self.cur_red): out[i] = v
    return out

  def value(self, n: LazyOp, inloop: bool) -> UOp:
    key = (id(n), inloop, self.cur, self.cur_red if inloop else None)
    if key not in self.memo: self.memo[key] = self._value(n, inloop)
    return self.memo[key]

  def _value(self, n: LazyOp, inloop: bool) -> UOp:
    dt = n.dtype
    if n.op is BufferOps.LOAD:
      idx, valid = n.arg.st.expr_idxs(self.leaf_idxs())
      if isinstance(valid, int) and valid == 0: return self.const(0, dt)
      buf, iu = self.bufs[n.arg.idx], self.idx(idx)
      if isinstance(valid, int): return self.cached(("load", id(buf), id(iu)), lambda: self.emit(UOps.LOAD, dt, (buf, iu)))
      vu = self.bool(valid)
      return self.cached(("load", id(buf), id(iu), id(vu)), lambda: self.emit(UOps.LOAD, dt, (buf, iu, vu, self.const(0, dt))))
    if n.op is BufferOps.CONST:
      _, valid = n.arg.st.expr_idxs(self.leaf_idxs())
      if isinstance(valid, int) and valid == 0: return self.const(0, dt)
      v = n.arg.val
      c = self.cast(self.idx(v), dt) if isinstance(v, Node) else self.const(v, dt)
      if isinstance(valid, int): return c
      return self.alu(TernaryOps.WHERE, dt, self.bool(valid), c, self.const(0, dt))
    if isinstance(n.op, ReduceOps):
      assert not inloop, "lower: nested reduce in one kernel"
      return self.red_val[(id(n), self.cur)]
    srcs = [self.value(s, inloop) for s in n.src]
    if n.op is UnaryOps.CAST: return self.cast(srcs[0], dt)
    if n.op is UnaryOps.BITCAST: return self.cached(("bitcast", dt, id(srcs[0])), lambda: self.emit(UOps.BITCAST, dt, (srcs[0],)))
    if n.op in (BinaryOps.IDIV, BinaryOps.MOD) and not (srcs[1].op is UOps.CONST and srcs[1].arg != 0):
      zero = self.alu(BinaryOps.CMPEQ, dtypes.bool, srcs[1], self.const(0, srcs[1].dtype))
      safe = self.alu(TernaryOps.WHERE, srcs[1].dtype, zero, self.const(1, srcs[1].dtype), srcs[1])
      return self.alu(TernaryOps.WHERE, dt, zero, self.const(0, dt), self.alu(n.op, dt, srcs[0], safe))
    return self.alu(n.op, dt, *srcs)

  def reduce(self, n: LazyOp) -> None:
    """Emit the reduce loops once for every upcast combo; results land in self.red_val[(id(n), combo)]."""
    dt, raxes, o = n.dtype, self.raxes, self.opts  # by the AST convention every reduced axis belongs to the one reduce
    if not raxes:  # nothing to loop over (all reduced axes have size 1)
      for c in self.combos:
        self.cur, self.cur_red = c, ()
        self.red_val[(id(n), c)] = self.cast(self.value(n.src[0], True), dt)
      self.cur_red = None
      return
    acc_dt = dtypes.float32 if dt in (dtypes.float16, dtypes.bfloat16) else dt
    init = 0 if n.op is ReduceOps.SUM else _min_val(acc_dt)
    accs = {c: self.emit(UOps.DEFIFZ_ACC, acc_dt, (), False if acc_dt == dtypes.bool else (float(init) if acc_dt.is_float else int(init))) for c in self.combos}
    unr = dict(o.unroll)
    parts, ranges = [], []  # per reduce axis: (loop var or 0, unroll, group)
    for k, i in enumerate(raxes):
      u, g = unr.get(i, 1), (o.group_reduce if o.group_reduce and i == raxes[-1] else 1)
      ext = self.F[i] // (u * g) if u * g > 1 else self.F[i]
      if _is1(ext): parts.append((0, u, g)); continue
      rv = Var(f"_fzr{k}", 0, max(_vmax(ext) - 1, 0))
      rng = self.emit(UOps.RANGE, dtypes.int32, (self.const(0, dtypes.int32), self.idx(ext)))
      self.vmap[rv] = rng
      parts.append((rv, u, g)); ranges.append(rng)
      self.scopes.append({})
    for rc in itertools.product(*[range(u) for _, u, _ in parts]):
      ridx = tuple(b * (u * g) + kk * g + (self.lane if g > 1 else 0) for (b, u, g), kk in zip(parts, rc))
      for c in self.combos:
        self.cur, self.cur_red = c, ridx
        x, acc = self.cast(self.value(n.src[0], True), acc_dt), accs[c]
        # accumulator updates are emitted directly (never CSE'd): acc changes between them
        if n.op is ReduceOps.SUM: new = self.emit(UOps.ALU, acc_dt, (acc, x), BinaryOps.ADD)
        else:
          new = self.emit(UOps.ALU, acc_dt, (x, acc), BinaryOps.MAX)
          if acc_dt.is_float: new = self.emit(UOps.ALU, acc_dt, (self.alu(BinaryOps.CMPEQ, dtypes.bool, x, x), new, x), TernaryOps.WHERE)  # NaN wins, like np.max
        self.emit(UOps.PHI, acc_dt, (acc, new))
    for rng in reversed(ranges):
      self.emit(UOps.ENDRANGE, None, (rng,))
      self.scopes.pop()
    self.cur_red = None
    self.memo = {k: v for k, v in self.memo.items() if not k[1]}  # loop-body values are out of scope now
    for c in self.combos:
      v = accs[c]
      if self.lane is not None:  # combine the lanes' partials across the SIMD group
        if n.op is ReduceOps.SUM: v = self.emit(UOps.ALU, acc_dt, (v,), "simd_sum")
        else:
          m = self.emit(UOps.ALU, acc_dt, (v,), "simd_max")
          if acc_dt.is_float:  # simd_max drops NaN: count NaN lanes separately so NaN still wins
            isn = self.emit(UOps.ALU, acc_dt, (self.emit(UOps.ALU, dtypes.bool, (v, v), BinaryOps.CMPEQ), self.const(0.0, acc_dt), self.const(1.0, acc_dt)), TernaryOps.WHERE)
            anyn = self.emit(UOps.ALU, dtypes.bool, (self.const(0.0, acc_dt), self.emit(UOps.ALU, acc_dt, (isn,), "simd_max")), BinaryOps.CMPLT)
            m = self.emit(UOps.ALU, acc_dt, (anyn, self.const(math.nan, acc_dt), m), TernaryOps.WHERE)
          v = m
      self.red_val[(id(n), c)] = self.cast(v, dt)

def _ident(s: Any) -> str:
  if isinstance(s, int): return str(s)
  return re.sub(r"[^0-9A-Za-z_]", "", s.name if isinstance(s, Var) else "s") or "s"

def _simd_renderer(r: Any) -> Any:
  """A copy of a METAL renderer that also renders the SIMD-group reductions used by group_reduce. They travel as ALU
  UOps whose arg is the string "simd_sum"/"simd_max" (the UOp spec has no cross-thread op; see docs/status/P2-OPT.md)."""
  r2 = copy.copy(r)
  r2.code_for_op = {**r.code_for_op, "simd_sum": lambda a, dt: f"simd_sum({a})", "simd_max": lambda a, dt: f"simd_max({a})"}
  return r2

def lower(ast: tuple, renderer: Any, opts: Opts | None = None) -> ProgramSpec:
  return _Lowerer(tuple(ast), renderer, opts).run()

# ---------------------------------------------------------------- Opts: validity, candidate actions, search

class OptError(ValueError): pass

def _kernel_shapes(ast: tuple) -> tuple[tuple, tuple, list, list]:
  nodes = [n for root in ast for n in root.walk()]
  leaves = [n for n in nodes if n.op in (BufferOps.LOAD, BufferOps.CONST)]
  reds = [n for n in nodes if isinstance(n.op, ReduceOps)]
  R = tuple(ast[0].arg.st.shape)
  F = tuple(leaves[0].arg.st.shape) if leaves else R
  return F, R, reds, nodes

MAX_UPCAST, MAX_UNROLL, MAX_LOCAL = 64, 32, 1024
# renderers whose code_for_op renders the "simd_sum"/"simd_max" ALUs as a 32-lane cross-thread reduce (Metal simd_sum,
# CUDA __shfl_xor_sync warp shuffles; the lane dim is local dim 0, so each group of 32 lanes is one SIMD group / warp)
GROUP_REDUCE_DEVICES = ("METAL", "CUDA")

def check_opts(o: Opts, ast: tuple, renderer: Any) -> None:
  """Raise OptError if `o` cannot be applied to this kernel on this renderer."""
  if o == Opts(): return
  F, R, reds, _ = _kernel_shapes(ast)
  gaxes = [i for i in range(len(R)) if not _is1(R[i])]
  raxes = [i for i in range(len(F)) if _is1(R[i]) and not _is1(F[i])]
  def fail(msg: str): raise OptError(f"{msg} (opts={o}, F={F}, R={R})")
  for nm, lst, allowed in (("upcast", o.upcast, gaxes), ("local", o.local, gaxes), ("unroll", o.unroll, raxes if reds else [])):
    axs = [a for a, _ in lst]
    if len(set(axs)) != len(axs): fail(f"{nm}: repeated axis")
    for a, amt in lst:
      if a not in allowed: fail(f"{nm}: axis {a} not allowed")
      if not isinstance(amt, int) or amt < 2: fail(f"{nm}: amount {amt} must be an int >= 2")
      if not isinstance(F[a], int): fail(f"{nm}: symbolic axis {a}")
  up, loc, unr = dict(o.upcast), dict(o.local), dict(o.unroll)
  for i in gaxes:
    if R[i] % (up.get(i, 1) * loc.get(i, 1)): fail(f"axis {i} not divisible")
  if prod(up.values()) > MAX_UPCAST: fail("upcast too large")
  if prod(unr.values()) * prod(up.values()) > MAX_UPCAST * 4: fail("unrolled body too large")
  if prod(unr.values()) > MAX_UNROLL: fail("unroll too large")
  if o.local or o.group_reduce:
    if not getattr(renderer, "has_local", False): fail("renderer has no local dims")
    if len(o.local) + (1 if o.group_reduce else 0) > 3: fail("more than 3 local dims")
    if prod(loc.values()) * max(o.group_reduce, 1) > MAX_LOCAL: fail("threadgroup too large")
  if o.group_reduce:
    if getattr(renderer, "device", "") not in GROUP_REDUCE_DEVICES or o.group_reduce != 32: fail("group_reduce is 32 (one SIMD group / warp) on METAL and CUDA only")
    if not reds or not raxes: fail("group_reduce needs a reduce")
    if not all(n.dtype in (dtypes.float32, dtypes.float16, dtypes.bfloat16, dtypes.int32) for n in reds): fail("group_reduce dtype")
    last = raxes[-1]
    if not isinstance(F[last], int) or F[last] % (o.group_reduce * unr.get(last, 1)): fail("group_reduce axis not divisible")
  for i in raxes:
    if i in unr and F[i] % unr[i]: fail(f"unroll axis {i} not divisible")

def opt_actions(ast: tuple, renderer: Any) -> list[tuple[str, int, int]]:
  """Single-step actions BEAM tries: (kind, axis, amount). Amounts multiply into an existing entry for that axis."""
  F, R, reds, _ = _kernel_shapes(ast)
  gaxes = [i for i in range(len(R)) if not _is1(R[i]) and isinstance(R[i], int)]
  raxes = [i for i in range(len(F)) if _is1(R[i]) and not _is1(F[i]) and isinstance(F[i], int)] if reds else []
  acts = [("upcast", i, a) for i in gaxes for a in (2, 3, 4, 8)]
  acts += [("unroll", i, a) for i in raxes for a in (2, 4, 8)]
  if getattr(renderer, "has_local", False):
    acts += [("local", i, a) for i in gaxes for a in (2, 4, 8, 16, 32)]
    if getattr(renderer, "device", "") in GROUP_REDUCE_DEVICES and raxes: acts.append(("group_reduce", 0, 32))
  return acts

def apply_action(o: Opts, act: tuple[str, int, int]) -> Opts:
  kind, axis, amt = act
  if kind == "group_reduce": return dataclasses.replace(o, group_reduce=amt)
  lst = list(getattr(o, kind))
  for k, (a, v) in enumerate(lst):
    if a == axis: lst[k] = (a, v * amt); break
  else: lst.append((axis, amt))
  return dataclasses.replace(o, **{kind: tuple(lst)})

def heuristic_opts(ast: tuple, renderer: Any) -> Opts:
  """Hand-coded opts (FZ_OPT=1, and a BEAM seed): group-for-reduce for wide reduces, register tiles + locals for matmul-like kernels."""
  F, R, reds, nodes = _kernel_shapes(ast)
  o = Opts()
  def ok(c: Opts) -> bool:
    try: check_opts(c, ast, renderer); return True
    except OptError: return False
  raxes = [i for i in range(len(F)) if _is1(R[i]) and not _is1(F[i])] if reds else []
  gaxes = [i for i in range(len(R)) if not _is1(R[i]) and isinstance(R[i], int)]
  out_n = prod([R[i] for i in gaxes]) if gaxes else 1
  # group-for-reduce when lanes read adjacent elements (reduce axis has stride 1 in some LOAD) or outputs are few
  loads = [n for n in nodes if n.op is BufferOps.LOAD]
  contig = bool(raxes) and any(n.arg.st.views[-1].strides[raxes[-1]] == 1 for n in loads)
  if raxes and getattr(renderer, "device", "") in GROUP_REDUCE_DEVICES and isinstance(F[raxes[-1]], int) and F[raxes[-1]] >= 256 \
     and ((contig and out_n <= 1 << 16) or out_n <= 1024) and ok(c := apply_action(o, ("group_reduce", 0, 32))):
    o = c
    for a in (8, 4, 2):
      if ok(c := apply_action(o, ("unroll", raxes[-1], a))): o = c; break
    if gaxes and ok(c := apply_action(o, ("local", gaxes[-1], 4))): o = c
    return o
  if raxes and len(gaxes) >= 2:  # matmul-like: register tile over the last two output axes, as big as parallelism allows
    i0, i1 = gaxes[-2], gaxes[-1]
    for u0, u1 in ((8, 4), (4, 4), (2, 4), (2, 2)):
      c = apply_action(apply_action(o, ("upcast", i0, u0)), ("upcast", i1, u1))
      if out_n // (u0 * u1) >= 1 << 14 and ok(c): o = c; break
    for a in (8, 4, 2):
      if ok(c := apply_action(o, ("unroll", raxes[-1], a))): o = c; break
    if getattr(renderer, "has_local", False):
      for i, a in ((i1, 16), (i0, 4)):
        if ok(c := apply_action(o, ("local", i, a))): o = c
  return o

def opts_to_json(o: Opts) -> str: return json.dumps(dataclasses.asdict(o))
def opts_from_json(s: str) -> Opts:
  d = json.loads(s)
  return Opts(tuple(map(tuple, d["upcast"])), tuple(map(tuple, d["unroll"])), tuple(map(tuple, d["local"])), int(d["group_reduce"]))

def _beam_cache_path(ast: tuple, device: Any, width: int):
  from fusezap.helpers import CACHE_DIR, stable_hash
  return CACHE_DIR / "beam" / stable_hash(f"v5\0{device.name}\0{type(device.renderer).__name__}\0{width}\0{_ast_key(ast)!r}")

def time_opts(device: Any, ast: tuple, opts: Opts, bufs: list, vals: dict, reps: int = 3, warm: float = 0.002) -> float:
  """Lower+compile `ast` with `opts` and return the best device time (seconds) of `reps` launches on `bufs`."""
  spec = lower(ast, device.renderer, opts)
  prg = device.runtime(spec.name, device.compiler.compile_cached(spec.src))
  gs = tuple(int(sym_infer(s, vals)) for s in spec.global_size)
  hs, vv = [b.ensure_allocated().handle for b in bufs[:spec.n_bufs]], tuple(int(vals[v]) for v in spec.vars)
  call = lambda w: prg(*hs, global_size=gs, local_size=spec.local_size, vals=vv, wait=w)
  first = call(True)
  first = first if first is not None and first > 0 else 1e-3
  for _ in range(min(256, int(warm / first))): call(False)  # keep the GPU busy ~`warm` s first so its clocks are up
  ts = [call(True) for _ in range(reps)]
  device.synchronize()
  return min(t if t is not None else math.inf for t in ts)

def beam_search(device: Any, ast: tuple, bufs: list, vals: dict, width: int, n_out: int = 1, max_rounds: int = 8) -> Opts:
  """FZ_BEAM: time candidate Opts on `device` (outputs go to scratch buffers) and return the fastest; the winner is
  cached per (AST, device) under CACHE_DIR/beam."""
  path = _beam_cache_path(ast, device, width)
  if path.exists() and not os.getenv("FZ_NOCACHE"):
    try:
      o = opts_from_json(path.read_text()); check_opts(o, ast, device.renderer); return o
    except Exception: pass
  from fusezap.device import Buffer
  outs = {s.arg.idx for s in ast}
  tbufs = [Buffer(b.device, b.size, b.dtype) if i in outs or i < n_out else b for i, b in enumerate(bufs)]
  acts = opt_actions(ast, device.renderer)
  def t(o: Opts, reps: int = 3) -> float:
    try: return time_opts(device, ast, o, tbufs, vals, reps)
    except Exception as e:
      if getenv("FZ_DEBUG", 0) >= 2: print(f"beam: {o} failed: {type(e).__name__}: {str(e)[:120]}")
      return math.inf
  seeds = [Opts()]  # the search starts from the unoptimized kernel and the hand-coded opts
  try: check_opts(h := heuristic_opts(ast, device.renderer), ast, device.renderer); seeds.append(h)
  except OptError: pass
  beam = sorted(((t(o), o) for o in dict.fromkeys(seeds)), key=lambda x: x[0])
  best_t, best = beam[0]
  seen, finalists = set(seeds), list(beam)
  for _ in range(max_rounds):
    cands: list[Opts] = []
    for _, o in beam:
      for a in acts:
        c = apply_action(o, a)
        if c in seen: continue
        seen.add(c)
        try: check_opts(c, ast, device.renderer)
        except OptError: continue
        cands.append(c)
    timed = sorted(((tt, c) for c in cands if (tt := t(c)) < math.inf), key=lambda x: x[0])
    finalists += timed[:width]
    if not timed or timed[0][0] >= best_t: break
    best_t, best = timed[0]
    beam = timed[:width]
    if getenv("FZ_DEBUG", 0) >= 2: print(f"beam: {best_t*1e6:.1f} us {best}")
  # timings are noisy (shared GPU, clock changes): re-time the best few with more launches before picking the winner
  top = list(dict.fromkeys([o for _, o in sorted(finalists, key=lambda x: x[0])][:max(3, width)] + seeds[1:]))
  if len(top) > 1:
    score = dict.fromkeys(top, math.inf)
    for _ in range(3):  # interleaved rounds, best of each
      for o in top: score[o] = min(score[o], t(o, 10))
    best = min(top, key=score.__getitem__)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(opts_to_json(best))
  return best

_cache: dict[tuple, tuple[ProgramSpec, Any]] = {}

def _ast_key(ast: tuple) -> tuple:
  """Hashable canonical form of an AST: float consts are keyed by repr, so 0.0 != -0.0 and nan == nan."""
  memo: dict[int, tuple] = {}
  def rec(n: LazyOp) -> tuple:
    if (k := memo.get(id(n))) is None:
      a = n.arg
      if isinstance(a, ConstBuffer): a = ("const", type(a.val).__name__, a.val if isinstance(a.val, Node) else repr(a.val), a.dtype, a.st)
      memo[id(n)] = k = (n.op, tuple(rec(s) for s in n.src), a)
    return k
  return tuple(rec(r) for r in ast)

def _int_env(k: str) -> int:
  try: return int(os.getenv(k, "0") or 0)
  except ValueError: return 0

def run_compiled(device: Any, si: Any) -> float | None:
  assert si.kind == "kernel" and si.ast is not None, f"run_compiled runs kernels only, got {si.kind}"
  bw, hand = _int_env("FZ_BEAM"), _int_env("FZ_OPT")
  key = (_ast_key(si.ast), type(device.renderer), device.name, bw, hand)
  if (ent := _cache.get(key)) is None:
    opts = None
    if bw > 0: opts = beam_search(device, si.ast, list(si.bufs), si.var_vals, bw, getattr(si, "n_outputs", 1))
    elif hand: opts = heuristic_opts(si.ast, device.renderer)
    spec = lower(si.ast, device.renderer, opts)
    if getenv("FZ_DEBUG", 0) >= 4: print(spec.src)
    if getenv("FZ_DEBUG", 0) >= 5:
      for u in spec.uops: print(u)
    lib = device.compiler.compile_cached(spec.src)
    _cache[key] = ent = (spec, device.runtime(spec.name, lib))
  spec, prg = ent
  vv = si.var_vals
  missing = [v for v in spec.vars if v not in vv]
  assert not missing, f"run_compiled: no value for {missing}"
  gs = tuple(int(sym_infer(s, vv)) for s in spec.global_size)
  if any(s <= 0 for s in gs): return None  # empty output: nothing to launch
  bufs = [b.ensure_allocated().handle for b in si.bufs[:spec.n_bufs]]
  return prg(*bufs, global_size=gs, local_size=spec.local_size, vals=tuple(int(vv[v]) for v in spec.vars), wait=getenv("FZ_DEBUG", 0) >= 1)
