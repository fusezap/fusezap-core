"""The 28 primitive ops (SPEC §3). Owned by the integrator; adding an op requires a spec change."""
from __future__ import annotations
from enum import Enum, auto

class UnaryOps(Enum):
  NEG = auto(); EXP2 = auto(); LOG2 = auto(); SIN = auto(); SQRT = auto(); RECIP = auto()
  CAST = auto()      # arg: DType. float->int truncates toward zero; ->bool is (x != 0)
  BITCAST = auto()   # arg: DType of the same itemsize

class BinaryOps(Enum):
  ADD = auto(); MUL = auto()
  MAX = auto()       # a > b ? a : b
  CMPLT = auto()     # -> bool
  CMPEQ = auto()     # -> bool
  IDIV = auto()      # ints only, C semantics (truncate toward zero)
  MOD = auto()       # ints only, C semantics (sign follows the dividend)

class TernaryOps(Enum):
  WHERE = auto()     # WHERE(cond: bool, a, b)

class ReduceOps(Enum):
  SUM = auto(); MAX = auto()   # arg: tuple of axes; reduced axes stay as size 1

class MovementOps(Enum):
  RESHAPE = auto()   # arg: new shape
  PERMUTE = auto()   # arg: permutation
  EXPAND = auto()    # arg: new shape (only size-1 dims may grow)
  PAD = auto()       # arg: ((before, after), ...), zero fill
  SHRINK = auto()    # arg: ((start, end), ...)
  FLIP = auto()      # arg: tuple[bool, ...] per axis

class LoadOps(Enum):
  CONST = auto()       # arg: python scalar
  EMPTY = auto()       # unfilled allocation
  COPY = auto()        # src: one LazyBuffer on another device, or arg: host memoryview / np.ndarray
  CONTIGUOUS = auto()  # forces materialization of src

class BufferOps(Enum):
  """Kernel-AST-only ops (SPEC §6.3); never appear in the lazy graph."""
  LOAD = auto(); CONST = auto(); STORE = auto()

Op = UnaryOps | BinaryOps | TernaryOps | ReduceOps | MovementOps | LoadOps | BufferOps
ElementwiseOps = (UnaryOps, BinaryOps, TernaryOps)

# Identity element for each reduce, used for accumulator init and for padding inside a reduce.
REDUCE_IDENTITY = {ReduceOps.SUM: 0, ReduceOps.MAX: float("-inf")}
