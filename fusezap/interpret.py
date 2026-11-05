"""Reference interpreter (SPEC §2). Implemented by WP2.

Evaluates the lazy graph op by op with numpy, with no scheduling or codegen. It is the trusted oracle
the compiled tiers are checked against, so it favors obvious correctness over speed.
Results are stored as realized Buffers on the lazybuffer's own device, so `Tensor.numpy()` works unchanged.
Movement views are evaluated by replaying ShapeTracker semantics, not by reading lb.arg history.

Value representation while computing (host storage is always `to_np(dtype)`):
  bool/ints -> numpy of the same type; float16 -> np.float16; float32 -> np.float32;
  bfloat16  -> np.float32 holding bf16-representable values (stored as uint16 bits, round-to-nearest-even).
Float ops on float16/bfloat16 are computed in float32 and rounded once per op (like the CPU backend).
Every unrealized non-CONST base reached from `lbs` gets a realized Buffer.

Symbolic shapes: `realize(lbs, var_vals)` takes values for the Vars in the graph's ShapeTrackers.
"""
from __future__ import annotations
import functools
import numpy as np
from fusezap.dtype import DType, dtypes, to_np
from fusezap.helpers import prod
from fusezap.ops import LoadOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps
from fusezap.device import Buffer
from fusezap.shape.symbolic import Node, Var, sym_infer
from fusezap.shape.view import ShapeTracker
from fusezap.lazy import LazyBuffer

# ---------------------------------------------------------------- dtype helpers

def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
  return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)

def _f32_to_bf16(a: np.ndarray) -> np.ndarray:
  """float32 -> bf16 bits, round-to-nearest-even; NaN keeps sign/payload (quiet bit set if truncation would give inf)."""
  u = np.ascontiguousarray(a, np.float32).view(np.uint32)
  rounded = ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)
  nan = np.isnan(np.ascontiguousarray(a, np.float32))
  if nan.any(): rounded = np.where(nan, ((u >> np.uint32(16)) | np.uint32(0x40)).astype(np.uint16), rounded)
  return rounded

def _vdtype(dt: DType):
  """numpy dtype of the value representation."""
  return np.dtype(np.float32) if dt == dtypes.bfloat16 else to_np(dt)

def _fix(x, dt: DType) -> np.ndarray:
  """Round/convert a computed array into the value representation of `dt`."""
  x = np.asarray(x)
  if dt == dtypes.bfloat16: return _bf16_to_f32(_f32_to_bf16(x.astype(np.float32)))
  return x.astype(_vdtype(dt), copy=False)

def to_host(x: np.ndarray, dt: DType) -> np.ndarray:
  """value representation -> host storage array."""
  if dt == dtypes.bfloat16: return _f32_to_bf16(x)
  return np.ascontiguousarray(x, to_np(dt))

def from_host(x: np.ndarray, dt: DType) -> np.ndarray:
  if dt == dtypes.bfloat16: return _bf16_to_f32(x)
  return x

def _compute_dtype(dt: DType):
  return np.float32 if dt in (dtypes.float16, dtypes.bfloat16) else _vdtype(dt)

# ---------------------------------------------------------------- views

@functools.lru_cache(maxsize=4096)
def _gather_idx(st: ShapeTracker) -> tuple[np.ndarray, np.ndarray | None]:
  """(flat buffer index, valid mask or None) per row-major logical element of a concrete ShapeTracker."""
  n = prod(st.shape)
  g: int | Var = Var("_gidx", 0, n - 1) if n > 1 else 0
  idx, valid = st.expr_node(g)
  vals = {g: np.arange(n, dtype=np.int64)} if isinstance(g, Var) else {}
  ev = lambda x: x.substitute(vals) if isinstance(x, Node) else x
  idx_a = np.broadcast_to(np.asarray(ev(idx), np.int64), (n,))
  if isinstance(valid, int) and valid == 1: return idx_a, None
  valid_a = np.broadcast_to(np.asarray(ev(valid)).astype(bool), (n,))
  return idx_a, valid_a

def _apply_view(flat: np.ndarray, st: ShapeTracker, var_vals: dict) -> np.ndarray:
  if st.vars(): st = st.subs(var_vals)
  shape = tuple(int(s) for s in st.shape)
  n = prod(shape)
  if n == 0: return np.zeros(shape, flat.dtype)
  if st.contiguous and flat.size == n: return flat.reshape(shape)
  idx, valid = _gather_idx(st)
  if valid is None: return flat[idx].reshape(shape)
  if flat.size == 0: return np.zeros(shape, flat.dtype)
  out = flat[np.clip(idx, 0, flat.size - 1)]
  return np.where(valid, out, np.zeros((), flat.dtype)).reshape(shape)

# ---------------------------------------------------------------- ops

def _idiv(a, b):
  bz = b == 0
  bs = np.where(bz, 1, b).astype(b.dtype)
  return np.where(bz, 0, (a - np.fmod(a, bs)) // bs).astype(a.dtype)

def _mod(a, b):
  bz = b == 0
  return np.where(bz, 0, np.fmod(a, np.where(bz, 1, b).astype(b.dtype))).astype(a.dtype)

def _cast(x: np.ndarray, src: DType, dt: DType) -> np.ndarray:
  if dt == dtypes.bool: return x != 0
  if dt.is_int and src.is_float:
    with np.errstate(invalid="ignore"): return np.trunc(x).astype(to_np(dt))
  if dt == dtypes.bfloat16: return _fix(x.astype(np.float32), dt)
  return x.astype(_vdtype(dt))

def _bitcast(x: np.ndarray, src: DType, dt: DType) -> np.ndarray:
  return from_host(np.ascontiguousarray(to_host(x, src)).view(to_np(dt)), dt)

def _elementwise(op, srcs: list[np.ndarray], src_dts: list[DType], out: DType, arg) -> np.ndarray:
  if op is UnaryOps.CAST: return _cast(srcs[0], src_dts[0], arg)
  if op is UnaryOps.BITCAST: return _bitcast(srcs[0], src_dts[0], arg)
  if op is TernaryOps.WHERE: return np.where(srcs[0], srcs[1], srcs[2]).astype(_vdtype(out))
  cdt = _compute_dtype(src_dts[0])
  xs = [s.astype(cdt, copy=False) for s in srcs]
  with np.errstate(all="ignore"):
    if op is UnaryOps.NEG: r = np.logical_not(xs[0]) if src_dts[0] == dtypes.bool else -xs[0]
    elif op is UnaryOps.EXP2: r = np.exp2(xs[0])
    elif op is UnaryOps.LOG2: r = np.log2(xs[0])
    elif op is UnaryOps.SIN: r = np.sin(xs[0])
    elif op is UnaryOps.SQRT: r = np.sqrt(xs[0])
    elif op is UnaryOps.RECIP: r = np.reciprocal(xs[0])
    elif op is BinaryOps.ADD: r = np.logical_or(*xs) if src_dts[0] == dtypes.bool else xs[0] + xs[1]
    elif op is BinaryOps.MUL: r = np.logical_and(*xs) if src_dts[0] == dtypes.bool else xs[0] * xs[1]
    elif op is BinaryOps.MAX: r = np.where(xs[0] > xs[1], xs[0], xs[1])
    elif op is BinaryOps.CMPLT: r = xs[0] < xs[1]
    elif op is BinaryOps.CMPEQ: r = xs[0] == xs[1]
    elif op is BinaryOps.IDIV: r = _idiv(*xs)
    elif op is BinaryOps.MOD: r = _mod(*xs)
    else: raise NotImplementedError(f"interp: {op}")
  return _fix(r, out)

def _reduce(op: ReduceOps, x: np.ndarray, dt: DType, axis: tuple[int, ...]) -> np.ndarray:
  if not axis: return x
  cdt = _compute_dtype(dt)
  xc = x.astype(cdt, copy=False)
  if op is ReduceOps.SUM: r = np.sum(xc, axis=axis, keepdims=True, dtype=cdt)
  elif op is ReduceOps.MAX:
    if xc.size == 0:
      init = -np.inf if dt.is_float else (False if dt == dtypes.bool else np.iinfo(cdt).min)
      r = np.full(tuple(1 if i in axis else s for i, s in enumerate(x.shape)), init, cdt)
    else: r = np.max(xc, axis=axis, keepdims=True)
  else: raise NotImplementedError(f"interp: {op}")
  return _fix(r, dt)

# ---------------------------------------------------------------- driver

def _toposort(lbs: list[LazyBuffer]) -> list[LazyBuffer]:
  """Unrealized bases reachable from lbs, sources first (iterative DFS: graphs can be deep)."""
  order, seen = [], set()
  stack: list[tuple[LazyBuffer, bool]] = [(lb.base, False) for lb in reversed(lbs)]
  while stack:
    b, done = stack.pop()
    if done:
      order.append(b)
      continue
    if id(b) in seen or b.realized is not None: continue
    seen.add(id(b))
    stack.append((b, True))
    for s in reversed(b.srcs):
      if id(s.base) not in seen and s.base.realized is None: stack.append((s.base, False))
  return order

class _Ctx:
  def __init__(self, var_vals: dict):
    self.var_vals, self.cache = var_vals, {}
  def flat(self, b: LazyBuffer) -> np.ndarray:
    """Flat value array of a base."""
    if id(b) in self.cache: return self.cache[id(b)][1]
    if b.realized is not None: a = from_host(b.realized.numpy(), b.dtype)
    elif b.op is LoadOps.CONST:
      v = b.arg
      if isinstance(v, Node): v = sym_infer(v, self.var_vals)
      a = _fix(np.full(1, v), b.dtype)
    else: raise RuntimeError(f"interp: base {b} is not computed yet")
    self.cache[id(b)] = (b, a)   # keep b alive so its id is not reused
    return a
  def value(self, lb: LazyBuffer) -> np.ndarray:
    """Value array of any LazyBuffer (base or view), shaped lb.shape."""
    b = lb.base
    flat = self.flat(b)
    if lb is b and b.op is not LoadOps.CONST: return flat.reshape(tuple(sym_infer(s, self.var_vals) for s in b.shape))
    return _apply_view(flat, lb.st, self.var_vals)

def _compute(ctx: _Ctx, b: LazyBuffer) -> np.ndarray:
  op = b.op
  shape = tuple(sym_infer(s, ctx.var_vals) for s in b.shape)
  if op is LoadOps.EMPTY: return np.zeros(shape, _vdtype(b.dtype))
  if op is LoadOps.COPY:
    if b.srcs: return ctx.value(b.srcs[0])
    a = np.asarray(b.arg)
    if not isinstance(b.arg, np.ndarray): a = np.frombuffer(memoryview(b.arg).cast("B"), dtype=to_np(b.dtype))
    return from_host(a.reshape(-1), b.dtype).reshape(shape)
  if op is LoadOps.CONTIGUOUS: return ctx.value(b.srcs[0])
  if op is LoadOps.CONST: return ctx.value(b)
  srcs = [ctx.value(s) for s in b.srcs]
  if isinstance(op, ReduceOps): return _reduce(op, srcs[0], b.srcs[0].dtype, b.arg)
  return _elementwise(op, srcs, [s.dtype for s in b.srcs], b.dtype, b.arg)

def realize(lbs: list[LazyBuffer], var_vals: dict | None = None) -> None:
  ctx = _Ctx(var_vals or {})
  for b in _toposort(lbs):
    if b.op is LoadOps.CONST: continue   # consts are never materialized
    val = np.asarray(_compute(ctx, b))
    shape = tuple(sym_infer(s, ctx.var_vals) for s in b.shape)
    assert val.shape == shape, f"interp produced {val.shape} for {b} (expected {shape})"
    host = to_host(val.reshape(-1), b.dtype)
    buf = Buffer(b.device, host.size, b.dtype)
    if host.size: buf.copyin(memoryview(host).cast("B"))
    else: buf.ensure_allocated()
    b.realized = buf
    if b.op is LoadOps.COPY and not b.srcs: b._arg = None   # as create_schedule: no host copy kept
    ctx.cache[id(b)] = (b, val.reshape(-1))
