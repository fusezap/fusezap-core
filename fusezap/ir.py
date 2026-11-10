"""Kernel AST and schedule items (SPEC §6.3). Owned by the integrator; builders must not edit."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from fusezap.dtype import DType
from fusezap.ops import BufferOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps
from fusezap.shape.view import ShapeTracker

if TYPE_CHECKING:
  from fusezap.device import Buffer
  from fusezap.shape.symbolic import Var

@dataclass(frozen=True)
class MemBuffer:
  idx: int              # index into ScheduleItem.bufs
  dtype: DType
  st: ShapeTracker

@dataclass(frozen=True)
class ConstBuffer:
  val: Any              # python scalar (or a Var for symbolic constants)
  dtype: DType
  st: ShapeTracker

KernelOp = BufferOps | UnaryOps | BinaryOps | TernaryOps | ReduceOps

@dataclass(frozen=True)
class LazyOp:
  """Node of a kernel AST. arg: MemBuffer (LOAD/STORE), ConstBuffer (CONST), axes tuple (reduce),
  DType (CAST/BITCAST), else None."""
  op: KernelOp
  src: tuple["LazyOp", ...] = ()
  arg: Any = None

  @property
  def dtype(self) -> DType:
    if self.op in (BufferOps.LOAD, BufferOps.CONST, BufferOps.STORE): return self.arg.dtype
    if self.op in (UnaryOps.CAST, UnaryOps.BITCAST): return self.arg
    if self.op in (BinaryOps.CMPLT, BinaryOps.CMPEQ):
      from fusezap.dtype import dtypes
      return dtypes.bool
    return self.src[-1].dtype  # WHERE: dtype of the value operands

  def walk(self) -> list["LazyOp"]:
    """Post-order, deduplicated."""
    seen, out = set(), []
    def rec(n: LazyOp):
      if id(n) in seen: return
      seen.add(id(n))
      for s in n.src: rec(s)
      out.append(n)
    rec(self)
    return out

  def pretty(self, indent: int = 0) -> str:
    pad = "  " * indent
    head = f"{pad}{self.op.name}" + (f" {self.arg}" if self.arg is not None else "")
    return "\n".join([head] + [s.pretty(indent + 1) for s in self.src])

@dataclass
class ScheduleItem:
  ast: tuple[LazyOp, ...] | None       # STORE roots; None for kind="copy"
  bufs: tuple["Buffer", ...]           # outputs first, then inputs, in MemBuffer.idx order
  var_vals: dict["Var", int] = field(default_factory=dict)
  kind: Literal["kernel", "copy", "custom"] = "kernel"   # "custom": SPEC §11.1
  n_outputs: int = 1
