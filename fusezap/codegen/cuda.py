"""CUDA C++ renderer for NVRTC (SPEC §11.2). Implemented by P3-CUDA by subclassing CStyleRenderer.

  extern "C" __global__ void __launch_bounds__(T) name(float* __restrict__ data0, ..., const int var0)

SPECIAL gidxN -> blockIdx.{x,y,z}, lidxN -> threadIdx.{x,y,z}; __launch_bounds__ is the product of the lidx sizes.
The source needs no headers (NVRTC's built-in intrinsics only), so it compiles in milliseconds and needs no toolkit include
directory. float16 and bfloat16 live in memory as their 16-bit patterns (`unsigned short`) and in registers as float,
rounded to the 16-bit type after every rounding op and cast (f16 with PTX cvt.rn, bf16 round-to-nearest-even in software,
like the CPU renderer): the same values as half/bfloat arithmetic that rounds each op. Bitcasts go through a union
(`fz_bc<To, From>`). Transcendentals are the precise libdevice ones: compile with fast math off and `--fmad=false`
(fusezap.device.cuda), so no a*b+c is contracted into an FMA, matching the CPU backend's -ffp-contract=off.

Group-for-reduce ALUs ("simd_sum"/"simd_max", fusezap.codegen.lower._simd_renderer) render as 32-lane warp shuffles.
Shared memory and barriers have no UOp yet: `render_local` / `barrier` are the hooks hand-written kernels (and a future
DEFINE_LOCAL/BARRIER UOp) use. The Metal renderer has no vector loads, so neither does this one.
"""
from __future__ import annotations
from typing import Any, Callable
import numpy as np
from fusezap.dtype import DType, dtypes
from fusezap.ops import UnaryOps, BinaryOps, TernaryOps
from fusezap.helpers import prod
from fusezap.codegen.uops import UOp, UOps, PtrDType
from fusezap.codegen.cstyle import CStyleRenderer, _base_code_for_op
from fusezap.codegen.metal import _f32, _bf16

_HALVES = (dtypes.float16, dtypes.bfloat16)

CUDA_PRELUDE = """#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fc00000))
template <class T, class F> static __device__ __forceinline__ T fz_bc(F x) { union { F f; T t; } u; u.f = x; return u.t; }"""

HALF_HELPERS = """static __device__ __forceinline__ float fz_h2f(unsigned short h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
static __device__ __forceinline__ unsigned short fz_f2h(float f) { unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
static __device__ __forceinline__ float fz_hround(float f) { return fz_h2f(fz_f2h(f)); }"""

BF16_HELPERS = """static __device__ __forceinline__ float fz_bf2f(unsigned short x) { return __uint_as_float(((unsigned int)x) << 16); }
static __device__ __forceinline__ unsigned short fz_f2bf(float f) {
  unsigned int u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) { unsigned short h = (unsigned short)(u >> 16); return (h & 0x7fu) ? h : (unsigned short)(h | 0x40u); }
  u += 0x7fffu + ((u >> 16) & 1u); return (unsigned short)(u >> 16);
}
static __device__ __forceinline__ float fz_bfround(float f) { return fz_bf2f(fz_f2bf(f)); }"""

WARP_HELPERS = """template <class T> static __device__ __forceinline__ T simd_sum(T x) {
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
  return x;
}
template <class T> static __device__ __forceinline__ T simd_max(T x) {
  for (int o = 16; o > 0; o >>= 1) { T y = __shfl_xor_sync(0xffffffffu, x, o); x = (y > x) ? y : x; }
  return x;
}"""

_ROUND = {dtypes.float16: "fz_hround", dtypes.bfloat16: "fz_bfround"}
_LOAD = {dtypes.float16: "fz_h2f", dtypes.bfloat16: "fz_bf2f"}
_STORE = {dtypes.float16: "fz_f2h", dtypes.bfloat16: "fz_f2bf"}
_EXACT = (UnaryOps.NEG, BinaryOps.MAX, BinaryOps.CMPLT, BinaryOps.CMPEQ, TernaryOps.WHERE)   # results need no 16-bit rounding

def _f16(x: float) -> float:
  with np.errstate(over="ignore"): return float(np.float16(x))

def _cuda_code_for_op() -> dict[Any, Callable[..., str]]:
  d = _base_code_for_op()
  d.update({UnaryOps.EXP2: lambda a, dt: f"exp2f({a})", UnaryOps.LOG2: lambda a, dt: f"log2f({a})",
            UnaryOps.SIN: lambda a, dt: f"sinf({a})", UnaryOps.SQRT: lambda a, dt: f"sqrtf({a})",
            UnaryOps.RECIP: lambda a, dt: f"(1.0f/{a})",
            "simd_sum": lambda a, dt: f"simd_sum({a})", "simd_max": lambda a, dt: f"simd_max({a})"})
  return d

class CUDARenderer(CStyleRenderer):
  device = "CUDA"
  has_local = True
  global_max = (2**31 - 1, 65535, 65535)
  kernel_prefix = 'extern "C" __global__ void'
  buffer_suffix = " __restrict__"
  arg_int_prefix = "const int"
  type_map = {dtypes.bool: "bool", dtypes.uint8: "unsigned char", dtypes.int8: "signed char", dtypes.int16: "short",
              dtypes.int32: "int", dtypes.int64: "long long", dtypes.float16: "float", dtypes.bfloat16: "float", dtypes.float32: "float"}
  buf_type_map = {dtypes.float16: "unsigned short", dtypes.bfloat16: "unsigned short"}
  code_for_op = _cuda_code_for_op()
  barrier = "__syncthreads();"
  def __init__(self, arch: str = "sm_80"): self.arch = arch

  # ------------------------------------------------------------ hooks
  def render_local(self, name: str, dt: DType, size: int) -> str:
    """Shared-memory array declaration (16-bit floats are stored as their bits, like global buffers)."""
    return f"__shared__ {self.buf_tname(dt)} {name}[{size}];"
  def prelude(self, uops: list[UOp]) -> str:
    dts = {u.dtype.base if isinstance(u.dtype, PtrDType) else u.dtype for u in uops}
    parts = [CUDA_PRELUDE]
    if dtypes.float16 in dts: parts.append(HALF_HELPERS)
    if dtypes.bfloat16 in dts: parts.append(BF16_HELPERS)
    if any(u.op is UOps.ALU and isinstance(u.arg, str) and u.arg.startswith("simd_") for u in uops): parts.append(WARP_HELPERS)
    return "\n".join(parts)
  def render_special(self, uop: UOp) -> str:
    axis, name, _ = uop.arg
    return f"(int){'threadIdx' if name.startswith('l') else 'blockIdx'}.{'xyz'[axis]}"
  def float_literal(self, x: float) -> str: return super().float_literal(_f32(x))
  def render_const(self, val: Any, dt: DType) -> str:
    if dt == dtypes.float16: return self.float_literal(_f16(float(val)))
    if dt == dtypes.bfloat16: return self.float_literal(_bf16(float(val)))
    return super().render_const(val, dt)
  def render_alu(self, op: Any, srcs: list[str], dt: DType) -> str:
    s = super().render_alu(op, srcs, dt)
    return f"{_ROUND[dt]}({s})" if dt in _HALVES and op not in _EXACT else s
  def render_cast(self, dt: DType, x: str) -> str:
    if dt in _HALVES: return f"{_ROUND[dt]}((float)({x}))"
    return f"(({self.tname(dt)})({x}))"
  def render_bitcast(self, dt: DType, x: str) -> str:
    sdt = self._cur.src[0].dtype
    if sdt in _HALVES: x, sdt_name = f"{_STORE[sdt]}({x})", "unsigned short"   # a 16-bit float's bits
    else: sdt_name = self.tname(sdt)
    if dt == dtypes.bool: return f"(({x}) != 0)"
    if dt in _HALVES: return f"{_LOAD[dt]}(fz_bc<unsigned short, {sdt_name}>({x}))"
    return f"fz_bc<{self.tname(dt)}, {sdt_name}>({x})"
  def render_load(self, dt: DType, buf: str, idx: str) -> str: return f"{_LOAD[dt]}({buf}[{idx}])" if dt in _HALVES else f"{buf}[{idx}]"
  def render_store(self, dt: DType, buf: str, idx: str, val: str) -> str:
    return f"{buf}[{idx}] = {_STORE[dt]}({val});" if dt in _HALVES else f"{buf}[{idx}] = {val};"
  def render_kernel(self, name: str, body_lines: list[str], bufs: list[tuple[str, PtrDType, bool]], vars: list[tuple[str, Any]]) -> str:
    args = [f"{'' if out else 'const '}{self.buf_tname(dt.base)}*{self.buffer_suffix} {n}" for n, dt, out in bufs]
    args += [f"{self.arg_int_prefix} {n}" for n, _ in vars]
    lsz = [u.arg[2] for u in self._uops if u.op is UOps.SPECIAL and u.arg[1].startswith("l")]
    bounds = f" __launch_bounds__({prod(lsz)})" if lsz and all(isinstance(s, int) for s in lsz) else ""
    return self.prelude(self._uops) + "\n" + f"{self.kernel_prefix}{bounds} {name}({', '.join(args)}) {{\n" + "\n".join(body_lines) + "\n}\n"
