"""Lazy graph of primitive ops (SPEC §6.1). Implemented by WP2.

A LazyBuffer is a base (op/srcs/arg, maybe realized) or a view of a base through a ShapeTracker.
Movement ops only create views. Every other op creates a new base whose srcs may be views.
The scheduler (WP3) and interpreter (WP2) read exactly these attributes:

  lb.base            the base LazyBuffer (self for a base)
  lb.st              ShapeTracker of this view over base (contiguous for a base)
  lb.shape, lb.dtype, lb.device
  lb.op, lb.srcs, lb.arg       meaningful on a base only (a view forwards them to its base)
  lb.realized        Buffer | None (base only); set once it has been computed
  lb.is_unrealized_const()     a CONST base that was never materialized
  lb.sharding        None in the MVP (SPEC P7)

Bases are never views of anything: `lb.base is lb` and `lb.st` is contiguous over `lb.shape`.
A movement op whose result is the identity view of the base returns the base itself.
"""
from __future__ import annotations
from typing import Any
from fusezap.dtype import DType, dtypes, to_np
from fusezap.helpers import prod
from fusezap.ops import LoadOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps
from fusezap.shape.symbolic import sint
from fusezap.shape.view import ShapeTracker
from fusezap.device import Buffer

def _shape_eq(a, b) -> bool: return len(a) == len(b) and all(x is y or x == y for x, y in zip(a, b))

class LazyBuffer:
  base: "LazyBuffer"
  st: ShapeTracker
  dtype: DType
  device: str
  op: Any
  srcs: tuple["LazyBuffer", ...]
  arg: Any
  realized: Buffer | None
  sharding: Any = None

  def __init__(self, device: str, st: ShapeTracker, dtype: DType, op: Any = None, srcs: tuple["LazyBuffer", ...] = (),
               arg: Any = None, base: "LazyBuffer | None" = None):
    self.device, self.st, self.dtype = device, st, dtype
    if base is None:
      assert st.contiguous, f"a base must have a contiguous ShapeTracker, got {st}"
      self._base, self._op, self._srcs, self._arg, self._realized = None, op, tuple(srcs), arg, None
    else:
      assert base._base is None, "views of views are not allowed; view the base"
      assert base.dtype == dtype and base.device == device
      self._base = base

  # base-only attributes; a view forwards them to its base
  @property
  def base(self) -> "LazyBuffer": return self if self._base is None else self._base
  @property
  def op(self): return self.base._op
  @property
  def srcs(self) -> tuple["LazyBuffer", ...]: return self.base._srcs
  @property
  def arg(self): return self.base._arg
  @property
  def realized(self) -> Buffer | None: return self.base._realized
  @realized.setter
  def realized(self, buf: Buffer | None):
    b = self.base
    assert buf is None or buf.dtype == b.dtype, f"realized dtype {buf.dtype} != {b.dtype}"
    b._realized = buf

  @property
  def shape(self) -> tuple[sint, ...]: return self.st.shape
  @property
  def size(self) -> sint: return self.st.size

  def __repr__(self) -> str:
    if self._base is None:
      return f"<LB {self.device} {self.shape} {self.dtype} op={getattr(self._op, 'name', self._op)}{' realized' if self._realized else ''}>"
    return f"<LB view {self.device} {self.shape} {self.dtype} of {self.base!r} st={self.st}>"

  # ---- construction
  @staticmethod
  def _new(device: str, shape: tuple[sint, ...], dtype: DType, op: Any, srcs: tuple["LazyBuffer", ...] = (), arg: Any = None):
    return LazyBuffer(device, ShapeTracker.from_shape(tuple(shape)), dtype, op, srcs, arg)

  @staticmethod
  def loadop(op: LoadOps, shape: tuple[sint, ...], dtype: DType, device: str, arg: Any = None,
             src: tuple["LazyBuffer", ...] = ()) -> "LazyBuffer":
    """CONST: arg is a scalar and the result is an EXPANDed view of a 1-element base (no allocation).
    EMPTY: an unfilled base. COPY: arg is a host np.ndarray/memoryview or src is (lb,) on another device."""
    shape = tuple(shape)
    if op is LoadOps.CONST:
      base = LazyBuffer._new(device, (), dtype, LoadOps.CONST, (), _const_val(arg, dtype))
      return base.reshape((1,) * len(shape)).expand(shape) if shape else base
    if op is LoadOps.EMPTY:
      assert not src
      return LazyBuffer._new(device, shape, dtype, LoadOps.EMPTY, (), arg)
    if op is LoadOps.COPY:
      if src:
        assert len(src) == 1 and _shape_eq(src[0].shape, shape) and src[0].dtype == dtype, f"bad COPY src {src}"
        return LazyBuffer._new(device, shape, dtype, LoadOps.COPY, tuple(src), None)
      import numpy as np
      a = np.ascontiguousarray(np.asarray(arg).reshape(-1))
      assert a.dtype == to_np(dtype), f"COPY arg host dtype {a.dtype} does not match {dtype} ({to_np(dtype)})"
      assert a.size == prod(shape), f"COPY arg has {a.size} elements, shape {shape} needs {prod(shape)}"
      return LazyBuffer._new(device, shape, dtype, LoadOps.COPY, (), a)
    if op is LoadOps.CONTIGUOUS:
      assert len(src) == 1 and _shape_eq(src[0].shape, shape)
      return LazyBuffer._new(device, shape, dtype, LoadOps.CONTIGUOUS, tuple(src), arg)
    raise ValueError(f"unknown loadop {op}")

  @staticmethod
  def from_buffer(buf: Buffer, shape: tuple[sint, ...]) -> "LazyBuffer":
    """Wrap an already-realized Buffer."""
    assert isinstance(prod(shape), int) and buf.size == prod(shape), f"buffer of {buf.size} elements cannot have shape {shape}"
    lb = LazyBuffer._new(buf.device, tuple(shape), buf.dtype, LoadOps.EMPTY)
    lb.realized = buf
    return lb

  def const(self, val: Any, shape: tuple[sint, ...] | None = None) -> "LazyBuffer":
    """A CONST with this dtype/device and `shape` (default: self.shape)."""
    return LazyBuffer.loadop(LoadOps.CONST, self.shape if shape is None else tuple(shape), self.dtype, self.device, val)

  # ---- compute
  def e(self, op: UnaryOps | BinaryOps | TernaryOps, *srcs: "LazyBuffer", arg: Any = None) -> "LazyBuffer":
    """Elementwise. `self` is the first operand. Shapes must match exactly (no broadcasting here)."""
    allsrc = (self,) + srcs
    for s in srcs:
      assert isinstance(s, LazyBuffer), f"elementwise operand must be a LazyBuffer, got {type(s)}"
      assert _shape_eq(s.shape, self.shape), f"{op}: shape mismatch {self.shape} vs {s.shape}"
      assert s.device == self.device, f"{op}: device mismatch {self.device} vs {s.device}"
    if isinstance(op, UnaryOps):
      assert len(srcs) == 0, f"{op} takes one operand"
      if op in (UnaryOps.CAST, UnaryOps.BITCAST):
        assert isinstance(arg, DType), f"{op} needs a DType arg"
        if op is UnaryOps.BITCAST: assert arg.itemsize == self.dtype.itemsize, f"bitcast {self.dtype} -> {arg} changes itemsize"
        out = arg
      else:
        assert self.dtype != dtypes.bool, f"{op} on bool"
        if op is not UnaryOps.NEG: assert self.dtype.is_float, f"{op} needs a float dtype, got {self.dtype}"
        out = self.dtype
    elif isinstance(op, BinaryOps):
      assert len(srcs) == 1, f"{op} takes two operands"
      assert srcs[0].dtype == self.dtype, f"{op}: dtype mismatch {self.dtype} vs {srcs[0].dtype}"
      if op in (BinaryOps.IDIV, BinaryOps.MOD): assert self.dtype.is_int, f"{op} is ints only, got {self.dtype}"
      out = dtypes.bool if op in (BinaryOps.CMPLT, BinaryOps.CMPEQ) else self.dtype
    elif isinstance(op, TernaryOps):
      assert op is TernaryOps.WHERE and len(srcs) == 2
      assert self.dtype == dtypes.bool, f"WHERE condition must be bool, got {self.dtype}"
      assert srcs[0].dtype == srcs[1].dtype, f"WHERE: dtype mismatch {srcs[0].dtype} vs {srcs[1].dtype}"
      out = srcs[0].dtype
    else: raise ValueError(f"not an elementwise op: {op}")
    return LazyBuffer._new(self.device, self.shape, out, op, allsrc, arg)

  def cast(self, dtype: DType, bitcast: bool = False) -> "LazyBuffer":
    if dtype == self.dtype: return self
    return self.e(UnaryOps.BITCAST if bitcast else UnaryOps.CAST, arg=dtype)

  def r(self, op: ReduceOps, axis: tuple[int, ...]) -> "LazyBuffer":
    """Reduce over `axis`; reduced axes become size 1."""
    assert isinstance(op, ReduceOps)
    axis = tuple(sorted(set(axis)))
    assert all(0 <= a < len(self.shape) for a in axis), f"bad reduce axes {axis} for shape {self.shape}"
    if op is ReduceOps.SUM: assert self.dtype != dtypes.bool, "SUM over bool; cast first"
    new_shape = tuple(1 if i in axis else s for i, s in enumerate(self.shape))
    return LazyBuffer._new(self.device, new_shape, self.dtype, op, (self,), axis)

  def contiguous(self) -> "LazyBuffer":
    if self._base is None and self.op is not LoadOps.CONST and self.realized is not None: return self
    return LazyBuffer.loadop(LoadOps.CONTIGUOUS, self.shape, self.dtype, self.device, src=(self,))

  def copy_to_device(self, device: str) -> "LazyBuffer":
    if device == self.device: return self
    if self.is_unrealized_const():  # consts need no transfer
      return LazyBuffer.loadop(LoadOps.CONST, self.shape, self.dtype, device, self.base.arg)
    return LazyBuffer.loadop(LoadOps.COPY, self.shape, self.dtype, device, src=(self,))

  # ---- movement (views only)
  def _view(self, st: ShapeTracker) -> "LazyBuffer":
    b = self.base
    if st.contiguous and _shape_eq(st.shape, b.shape): return b
    return LazyBuffer(self.device, st, self.dtype, base=b)
  def reshape(self, arg: tuple[sint, ...]) -> "LazyBuffer":
    arg = tuple(arg)
    if _shape_eq(arg, self.shape): return self
    return self._view(self.st.reshape(arg))
  def permute(self, arg: tuple[int, ...]) -> "LazyBuffer":
    arg = tuple(arg)
    if arg == tuple(range(len(self.shape))): return self
    return self._view(self.st.permute(arg))
  def expand(self, arg: tuple[sint, ...]) -> "LazyBuffer":
    arg = tuple(arg)
    if _shape_eq(arg, self.shape): return self
    return self._view(self.st.expand(arg))
  def pad(self, arg: tuple[tuple[sint, sint], ...]) -> "LazyBuffer":
    arg = tuple(tuple(p) for p in arg)
    if all(b == 0 and e == 0 for b, e in arg): return self
    return self._view(self.st.pad(arg))
  def shrink(self, arg: tuple[tuple[sint, sint], ...]) -> "LazyBuffer":
    arg = tuple(tuple(p) for p in arg)
    if all(b == 0 and (e is s or e == s) for (b, e), s in zip(arg, self.shape)): return self
    return self._view(self.st.shrink(arg))
  def flip(self, arg: tuple[bool, ...]) -> "LazyBuffer":
    arg = tuple(bool(a) for a in arg)
    if not any(arg): return self
    return self._view(self.st.flip(arg))

  # ---- queries
  def is_unrealized_const(self) -> bool: return self.base._op is LoadOps.CONST and self.base._realized is None
  def is_realized(self) -> bool: return self.base.realized is not None

def _const_val(val: Any, dtype: DType) -> Any:
  """Normalize a CONST scalar to the python type of its dtype (Vars pass through for symbolic consts)."""
  from fusezap.shape.symbolic import Node
  if isinstance(val, Node): return val
  if dtype == dtypes.bool: return bool(val)
  if dtype.is_int:
    import math
    if isinstance(val, float) and not math.isfinite(val): raise ValueError(f"cannot make an {dtype} const from {val}")
    return int(val)
  return float(val)
