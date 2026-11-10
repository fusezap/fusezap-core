"""Linear loop IR shared by lowering and every renderer (SPEC §7.1).
Owned by the integrator; builders must not edit. Lowering (WP4) produces this; renderers (WP4, WP5) consume it.

A program is `list[UOp]` in execution order. Every UOp is SSA: it is defined once, and its `src`
entries appear earlier in the list. `dtype` is None for ops that produce no value.

  op            dtype       src                             arg
  ------------  ----------  ------------------------------  ---------------------------------------
  DEFIFZ_GLOBAL DType(ptr)  ()                              (idx: int, name: str, is_output: bool)   buffer param `dataN`
  DEFIFZ_VAR    int32       ()                              Var                                      symbolic dim param
  SPECIAL       int32       ()                              (axis: 0..2, name: "gidx0"|"lidx0", size: sint)
                                                             GPU: thread/group index; CPU: renderer turns it into a loop
  CONST         DType       ()                              python scalar (renderer prints with a dtype suffix)
  RANGE         int32       (start: UOp, end: UOp)          None                                     opens `for (i=start; i<end; i++)`
  ENDRANGE      None        (range: UOp,)                   None                                     closes the loop
  DEFIFZ_ACC    DType       ()                              initial value (python scalar)            mutable accumulator, declared before its RANGE
  PHI           DType       (acc: UOp, new_value: UOp)      None                                     acc = new_value (accumulator update in the loop)
  LOAD          DType       (buf, idx) or                   None                                     buf[idx]; with valid: valid ? buf[idx] : alt
                            (buf, idx, valid, alt)
  STORE         None        (buf, idx, value)               None                                     buf[idx] = value
  ALU           DType       (a,) | (a, b) | (c, a, b)       UnaryOps|BinaryOps|TernaryOps (not CAST/BITCAST)
  CAST          DType       (a,)                            None                                     value conversion
  BITCAST       DType       (a,)                            None                                     reinterpret bits
  IF            None        (cond: UOp,)                    None
  ENDIF         None        (if: UOp,)                      None

Index math is plain int32 ALU UOps (ADD/MUL/IDIV/MOD/CMPLT) built from ShapeTracker.expr_idxs.
Pointer dtypes are represented as `PtrDType(base)`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from fusezap.dtype import DType

class UOps(Enum):
  DEFIFZ_GLOBAL = auto(); DEFIFZ_VAR = auto(); SPECIAL = auto(); CONST = auto()
  RANGE = auto(); ENDRANGE = auto(); DEFIFZ_ACC = auto(); PHI = auto()
  LOAD = auto(); STORE = auto(); ALU = auto(); CAST = auto(); BITCAST = auto()
  IF = auto(); ENDIF = auto()

@dataclass(frozen=True)
class PtrDType:
  base: DType
  def __repr__(self): return f"ptr({self.base!r})"

@dataclass(eq=False)
class UOp:
  op: UOps
  dtype: DType | PtrDType | None
  src: tuple["UOp", ...] = ()
  arg: Any = None
  def __repr__(self):
    return f"{self.op.name:<14} {str(self.dtype):<18} srcs={len(self.src)} {self.arg!r}"

@dataclass
class ProgramSpec:
  """Output of lowering + rendering, consumed by a device runtime."""
  name: str
  src: str
  global_size: tuple[Any, ...]                # sint per dim; evaluated with var_vals at launch
  local_size: tuple[int, ...] | None
  n_bufs: int
  vars: list[Any] = field(default_factory=list)   # Var order of the trailing int params
  uops: list[UOp] = field(default_factory=list)
