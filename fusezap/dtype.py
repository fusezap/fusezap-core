"""DTypes and promotion (SPEC §4). Owned by the integrator; builders must not edit."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Final

@dataclass(frozen=True, order=True)
class DType:
  priority: int        # position in the promotion lattice
  itemsize: int        # bytes
  name: str
  fmt: str | None      # numpy/struct format char for host transfer; bfloat16 travels as uint16 bits ("H")
  def __repr__(self): return f"dtypes.{self.name}"
  @property
  def is_float(self) -> bool: return self in dtypes.floats
  @property
  def is_int(self) -> bool: return self in dtypes.ints
  @property
  def is_unsigned(self) -> bool: return self in (dtypes.uint8,)

class dtypes:
  bool: Final[DType] = DType(0, 1, "bool", "?")
  uint8: Final[DType] = DType(1, 1, "uint8", "B")
  int8: Final[DType] = DType(2, 1, "int8", "b")
  int16: Final[DType] = DType(3, 2, "int16", "h")
  int32: Final[DType] = DType(4, 4, "int32", "i")
  int64: Final[DType] = DType(5, 8, "int64", "q")
  float16: Final[DType] = DType(6, 2, "float16", "e")
  bfloat16: Final[DType] = DType(7, 2, "bfloat16", "H")
  float32: Final[DType] = DType(8, 4, "float32", "f")

  ints: Final = (uint8, int8, int16, int32, int64)
  floats: Final = (float16, bfloat16, float32)
  all: Final = (bool, uint8, int8, int16, int32, int64, float16, bfloat16, float32)

  default_float: DType = float32
  default_int: DType = int32

  @staticmethod
  def from_py(x) -> DType:
    if isinstance(x, bool): return dtypes.bool
    if isinstance(x, int): return dtypes.default_int
    if isinstance(x, float): return dtypes.default_float
    raise TypeError(f"no dtype for python value {x!r}")

  @staticmethod
  def from_np(npdtype) -> DType:
    import numpy as np
    m = {np.bool_: dtypes.bool, np.uint8: dtypes.uint8, np.int8: dtypes.int8, np.int16: dtypes.int16,
         np.int32: dtypes.int32, np.int64: dtypes.int64, np.float16: dtypes.float16, np.float32: dtypes.float32,
         np.float64: dtypes.float32}  # float64 is narrowed: the engine has no f64
    return m[np.dtype(npdtype).type]

  @staticmethod
  def by_name(name: str) -> DType:
    return next(d for d in dtypes.all if d.name == name)

def to_np(dt: DType):
  """numpy dtype used for host buffers of `dt` (bfloat16 -> uint16 bit pattern)."""
  import numpy as np
  return np.dtype(dt.fmt)

def promote(a: DType, b: DType) -> DType:
  """Binary promotion: max on the lattice, except float16 with bfloat16 gives float32."""
  if {a, b} == {dtypes.float16, dtypes.bfloat16}: return dtypes.float32
  return max(a, b)

def least_upper_float(dt: DType) -> DType:
  """dtype used for float-only ops (exp, sqrt, ...) on `dt`."""
  return dt if dt.is_float else dtypes.default_float
