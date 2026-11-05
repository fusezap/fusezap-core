"""Per-tensor float8 e4m3 quantization on the GPU (sm_89+ cvt.rn.satfinite.e4m3x2): amax over the tensor (one atomicMax
of the float bits; |x| >= 0 so the unsigned order is the float order), then q = e4m3(x / s) with s = amax / 448
(s = 1 for an all-zero tensor). Codes are uint8 float8_e4m3fn bits, like fusezap.serve.quant; scale stays on the device."""
from __future__ import annotations
from fusezap.device.cuda_kernels._rt import COMMON

LOAD = {"f32": ("float", "x[i]"), "f16": ("u16", "fz_h2f(x[i])"), "bf16": ("u16", "__uint_as_float(((u32)x[i]) << 16)")}

def render(dt: str) -> str:
  ct, ld = LOAD[dt]
  return COMMON + f"""static __device__ __forceinline__ float fz_h2f(u16 h) {{ float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }}
extern "C" __global__ void fz_absmax_{dt}(u32* __restrict__ out, const {ct}* __restrict__ x, const long long n) {{
  float m = 0.f;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) m = fmaxf(m, fabsf({ld}));
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) atomicMax(out, __float_as_uint(m));
}}
extern "C" __global__ void fz_fp8_quant_{dt}(u8* __restrict__ q, float* __restrict__ scale, const {ct}* __restrict__ x, const u32* __restrict__ amax, const long long n) {{
  const float a = __uint_as_float(*amax), s = a > 0.f ? a / 448.f : 1.f;
  if (blockIdx.x == 0 && threadIdx.x == 0) *scale = s;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {{
    u16 r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.f), "f"({ld} / s));
    q[i] = (u8)(r & 0xff);
  }}
}}
"""
