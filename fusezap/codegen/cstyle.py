"""Renderer base + C-like renderer (SPEC §7.3). Implemented by WP4; MetalRenderer (WP5) subclasses CStyleRenderer.

Overridable surface (keep this stable; WP5 depends on it):
  device: str
  has_local: bool                     GPU-style launch (True) vs loops (False)
  global_max: tuple[int, int, int] | None
  supports_dtype(dt) -> bool
  type_map: dict[DType, str]          C type name per dtype
  kernel_prefix: str                  e.g. "kernel void" or "void"
  buffer_prefix: str                  e.g. "device " or ""
  buffer_suffix: str                  e.g. " restrict"
  arg_int_prefix: str                 e.g. "constant int&" or "const int"
  render_special(uop) -> str          expression for SPECIAL
  render_kernel(name, body_lines, bufs, vars) -> str
  code_for_op: dict[op, Callable[..., str]]   ALU rendering; e.g. EXP2 -> "exp2(a)"
  render_cast(dt, x) / render_bitcast(dt, x) / render_const(val, dt)
  render(name, uops) -> str           drives everything above; renderers rarely override it

Additional hooks (optional to override; defaults are plain C):
  buf_type_map: dict[DType, str]      element type of a buffer pointer when it differs from type_map (CPU bf16 -> uint16)
  prelude(uops) -> str                text emitted before the kernel (includes, helpers)
  render_alu(op, srcs, dtype) -> str  wraps code_for_op (CPU rounds bf16 results here)
  render_load(dt, buf, idx) / render_store(dt, buf, idx, val) -> str   memory access (CPU converts bf16 bits here)
  var_name(v) -> str                  C name of a DEFIFZ_VAR (default: Var.name)
  render_size(size) -> str            sint (int | Node) as a C expression (CPU SPECIAL loop bounds)
  self._cur                           the UOp being rendered, readable from any hook (e.g. a bitcast's source dtype)

Conventions: every non-CONST value gets its own SSA local (`alu3`, `val0`, `acc1`, `ridx0`, ...); CONSTs are
inlined. bufs passed to render_kernel are `(name, PtrDType, is_output)`, vars are `(name, Var)`, in uop order.
"""
from __future__ import annotations
import math, struct
from typing import Any, Callable
from fusezap.dtype import DType, dtypes
from fusezap.ops import UnaryOps, BinaryOps, TernaryOps
from fusezap.codegen.uops import UOp, UOps, PtrDType

class Renderer:
  device: str = ""
  has_local: bool = False
  global_max: tuple[int, int, int] | None = None
  def supports_dtype(self, dt: DType) -> bool: return True
  def render(self, name: str, uops: list[UOp]) -> str: raise NotImplementedError

def _base_code_for_op() -> dict[Any, Callable[..., str]]:
  return {
    UnaryOps.NEG: lambda a, dt: f"(!{a})" if dt == dtypes.bool else f"(-({a}))",  # (-(x)): x may be a negative literal
    UnaryOps.EXP2: lambda a, dt: f"exp2({a})", UnaryOps.LOG2: lambda a, dt: f"log2({a})",
    UnaryOps.SIN: lambda a, dt: f"sin({a})", UnaryOps.SQRT: lambda a, dt: f"sqrt({a})",
    UnaryOps.RECIP: lambda a, dt: f"(1/{a})",
    BinaryOps.ADD: lambda a, b, dt: f"({a}+{b})", BinaryOps.MUL: lambda a, b, dt: f"({a}*{b})",
    BinaryOps.MAX: lambda a, b, dt: f"(({a}>{b})?{a}:{b})",
    BinaryOps.CMPLT: lambda a, b, dt: f"({a}<{b})", BinaryOps.CMPEQ: lambda a, b, dt: f"({a}=={b})",
    BinaryOps.IDIV: lambda a, b, dt: f"({a}/{b})", BinaryOps.MOD: lambda a, b, dt: f"({a}%{b})",
    TernaryOps.WHERE: lambda c, a, b, dt: f"({c}?{a}:{b})",
  }

class CStyleRenderer(Renderer):
  kernel_prefix: str = "void"
  buffer_prefix: str = ""
  buffer_suffix: str = ""
  arg_int_prefix: str = "const int"
  type_map: dict[DType, str] = {
    dtypes.bool: "bool", dtypes.uint8: "unsigned char", dtypes.int8: "signed char", dtypes.int16: "short",
    dtypes.int32: "int", dtypes.int64: "long long", dtypes.float16: "half", dtypes.bfloat16: "bfloat16", dtypes.float32: "float"}
  buf_type_map: dict[DType, str] = {}
  code_for_op: dict[Any, Callable[..., str]] = _base_code_for_op()
  infinity: str = "INFINITY"
  nan: str = "NAN"

  # ------------------------------------------------------------ small hooks
  def tname(self, dt: DType) -> str: return self.type_map[dt]
  def buf_tname(self, dt: DType) -> str: return self.buf_type_map.get(dt, self.type_map[dt])
  def prelude(self, uops: list[UOp]) -> str: return ""
  def var_name(self, v: Any) -> str: return v.name
  def render_special(self, uop: UOp) -> str: raise NotImplementedError(f"{type(self).__name__} has no SPECIAL rendering")
  def render_cast(self, dt: DType, x: str) -> str: return f"(({self.tname(dt)})({x}))"
  def render_bitcast(self, dt: DType, x: str) -> str: return f"__builtin_bit_cast({self.tname(dt)}, {x})"
  def render_alu(self, op: Any, srcs: list[str], dt: DType) -> str: return self.code_for_op[op](*srcs, dt)
  def render_load(self, dt: DType, buf: str, idx: str) -> str: return f"{buf}[{idx}]"
  def render_store(self, dt: DType, buf: str, idx: str, val: str) -> str: return f"{buf}[{idx}] = {val};"
  def float_literal(self, x: float) -> str:
    if math.isnan(x): return self.nan
    if math.isinf(x): return self.infinity if x > 0 else f"(-{self.infinity})"
    return f"{float(x)!r}f"
  def render_const(self, val: Any, dt: DType) -> str:
    if dt == dtypes.bool: return "1" if val else "0"
    if dt.is_float:
      lit = self.float_literal(float(val))
      return lit if dt == dtypes.float32 else self.render_cast(dt, lit)
    if dt == dtypes.int32: return str(int(val))
    if dt == dtypes.int64: return f"{int(val)}ll"
    return self.render_cast(dt, str(int(val)))
  def render_size(self, size: Any) -> str:
    if isinstance(size, int): return str(size)
    if size in self._vars_by_obj: return self._vars_by_obj[size]
    from fusezap.shape.symbolic import RenderOps
    ops = RenderOps()
    ops.var = lambda v: self._vars_by_obj.get(v, self.var_name(v))
    return size.render(ops)

  def render_kernel(self, name: str, body_lines: list[str], bufs: list[tuple[str, PtrDType, bool]], vars: list[tuple[str, Any]]) -> str:
    args = [f"{self.buffer_prefix}{self.buf_tname(dt.base)}*{self.buffer_suffix} {n}" for n, dt, _ in bufs]
    args += [f"{self.arg_int_prefix} {n}" for n, _ in vars]
    pre = self.prelude(self._uops)
    return (pre + "\n" if pre else "") + f"{self.kernel_prefix} {name}({', '.join(args)}) {{\n" + "\n".join(body_lines) + "\n}\n"

  # ------------------------------------------------------------ driver
  def render(self, name: str, uops: list[UOp]) -> str:
    self._uops = uops
    r: dict[UOp, str] = {}
    lines: list[str] = []
    bufs: list[tuple[str, PtrDType, bool]] = []
    vars: list[tuple[str, Any]] = []
    self._vars_by_obj: dict[Any, str] = {}
    counts: dict[str, int] = {}
    depth, open_specials = 1, 0
    def ssa(prefix: str) -> str:
      counts[prefix] = counts.get(prefix, -1) + 1
      return f"{prefix}{counts[prefix]}"
    def emit(s: str): lines.append("  " * depth + s)
    for u in uops:
      self._cur, op, dt, src = u, u.op, u.dtype, u.src
      if op is UOps.DEFIFZ_GLOBAL:
        _, nm, is_out = u.arg
        bufs.append((nm, dt, is_out)); r[u] = nm
      elif op is UOps.DEFIFZ_VAR:
        nm = self.var_name(u.arg)
        vars.append((nm, u.arg)); r[u] = nm
        try: self._vars_by_obj[u.arg] = nm
        except TypeError: pass
      elif op is UOps.SPECIAL:
        nm = u.arg[1]; r[u] = nm
        if self.has_local: emit(f"int {nm} = {self.render_special(u)};")
        else:
          emit(f"for (int {nm} = 0; {nm} < {self.render_size(u.arg[2])}; {nm}++) {{")
          depth += 1; open_specials += 1
      elif op is UOps.CONST: r[u] = self.render_const(u.arg, dt)
      elif op is UOps.RANGE:
        nm = r[u] = ssa("ridx")
        emit(f"for (int {nm} = {r[src[0]]}; {nm} < {r[src[1]]}; {nm}++) {{"); depth += 1
      elif op in (UOps.ENDRANGE, UOps.ENDIF): depth -= 1; emit("}")
      elif op is UOps.IF: emit(f"if ({r[src[0]]}) {{"); depth += 1
      elif op is UOps.DEFIFZ_ACC:
        nm = r[u] = ssa("acc")
        emit(f"{self.tname(dt)} {nm} = {self.render_const(u.arg, dt)};")
      elif op is UOps.PHI:
        emit(f"{r[src[0]]} = {r[src[1]]};"); r[u] = r[src[0]]
      elif op is UOps.LOAD:
        ld = self.render_load(dt, r[src[0]], r[src[1]])
        if len(src) == 4: ld = f"({r[src[2]]}) ? ({ld}) : ({r[src[3]]})"
        nm = r[u] = ssa("val")
        emit(f"{self.tname(dt)} {nm} = {ld};")
      elif op is UOps.STORE:
        emit(self.render_store(src[0].dtype.base, r[src[0]], r[src[1]], r[src[2]]))
      elif op is UOps.ALU:
        nm = r[u] = ssa("alu")
        emit(f"{self.tname(dt)} {nm} = {self.render_alu(u.arg, [r[s] for s in src], dt)};")
      elif op is UOps.CAST:
        nm = r[u] = ssa("cast")
        emit(f"{self.tname(dt)} {nm} = {self.render_cast(dt, r[src[0]]) if src[0].dtype != dt else r[src[0]]};")
      elif op is UOps.BITCAST:
        nm = r[u] = ssa("cast")
        emit(f"{self.tname(dt)} {nm} = {self.render_bitcast(dt, r[src[0]])};")
      else: raise NotImplementedError(f"cannot render {op}")
    for _ in range(open_specials): depth -= 1; emit("}")
    assert depth == 1, f"unbalanced RANGE/IF in {name}"
    return self.render_kernel(name, lines, bufs, vars)

# ---------------------------------------------------------------- CPU (clang)

_C_PRELUDE = "#include <math.h>\n#include <stdbool.h>"
_BF16_HELPERS = """static inline float fz_bf2f(unsigned short x) { unsigned int u = (unsigned int)x << 16; float f; __builtin_memcpy(&f, &u, 4); return f; }
static inline unsigned short fz_f2bf(float f) {
  unsigned int u; __builtin_memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) {  /* NaN: keep sign+payload bits; quiet only if truncation would make it inf */
    unsigned short h = (unsigned short)(u >> 16); return (h & 0x7fu) ? h : (unsigned short)(h | 0x40u);
  }
  u += 0x7fffu + ((u >> 16) & 1u); return (unsigned short)(u >> 16);
}
static inline float fz_bfround(float f) { return fz_bf2f(fz_f2bf(f)); }"""

def _bf16_round(x: float) -> float:
  if math.isnan(x) or math.isinf(x): return x
  if abs(x) > 3.4028234663852886e38: return math.copysign(math.inf, x)  # above f32 max: f32 (and so bf16) overflows to inf
  u = struct.unpack("<I", struct.pack("<f", x))[0]
  u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000
  return struct.unpack("<f", struct.pack("<I", u))[0]

def _c_code_for_op() -> dict[Any, Callable[..., str]]:
  d = _base_code_for_op()
  d.update({UnaryOps.EXP2: lambda a, dt: f"exp2f({a})", UnaryOps.LOG2: lambda a, dt: f"log2f({a})",
            UnaryOps.SIN: lambda a, dt: f"sinf({a})", UnaryOps.SQRT: lambda a, dt: f"sqrtf({a})",
            UnaryOps.RECIP: lambda a, dt: f"(1.0f/{a})"})
  return d

class CRenderer(CStyleRenderer):
  """Single-threaded C for clang. SPECIAL dims become loops. bf16 lives in registers as float (rounded to bf16
  after every op) and in memory as uint16 bits; float16 is clang's native `_Float16`."""
  device = "CPU"
  has_local = False
  buffer_suffix = " restrict"
  type_map = {**CStyleRenderer.type_map, dtypes.float16: "_Float16", dtypes.bfloat16: "float"}
  buf_type_map = {dtypes.bfloat16: "unsigned short"}
  code_for_op = _c_code_for_op()

  def prelude(self, uops: list[UOp]) -> str:
    bf = any(u.dtype == dtypes.bfloat16 or (isinstance(u.dtype, PtrDType) and u.dtype.base == dtypes.bfloat16) for u in uops)
    return _C_PRELUDE + ("\n" + _BF16_HELPERS if bf else "")
  def render_const(self, val: Any, dt: DType) -> str:
    if dt == dtypes.bfloat16: return self.float_literal(_bf16_round(float(val)))
    return super().render_const(val, dt)
  def render_alu(self, op: Any, srcs: list[str], dt: DType) -> str:
    s = super().render_alu(op, srcs, dt)
    return f"fz_bfround({s})" if dt == dtypes.bfloat16 else s
  def render_cast(self, dt: DType, x: str) -> str:
    if dt == dtypes.bfloat16: return f"fz_bfround((float)({x}))"
    return super().render_cast(dt, x)
  def render_bitcast(self, dt: DType, x: str) -> str:
    # bf16 values are floats in registers: go through their uint16 bit pattern
    if self._cur.src[0].dtype == dtypes.bfloat16: x = f"fz_f2bf({x})"
    if dt == dtypes.bfloat16: return f"fz_bf2f(__builtin_bit_cast(unsigned short, {x}))"
    if dt == dtypes.bool: return f"(({x}) != 0)"  # a byte other than 0/1 is not a valid _Bool; normalize like Metal
    return super().render_bitcast(dt, x)
  def render_load(self, dt: DType, buf: str, idx: str) -> str:
    return f"fz_bf2f({buf}[{idx}])" if dt == dtypes.bfloat16 else f"{buf}[{idx}]"
  def render_store(self, dt: DType, buf: str, idx: str, val: str) -> str:
    return f"{buf}[{idx}] = fz_f2bf({val});" if dt == dtypes.bfloat16 else f"{buf}[{idx}] = {val};"
