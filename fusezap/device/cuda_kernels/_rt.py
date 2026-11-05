"""Runtime glue for the CUDA library kernels (P3-GEMM): NVRTC at a chosen arch (sm_90a for WGMMA), modules with dynamic
shared memory, launches with arbitrary parameter bytes (TMA descriptors are 128-byte __grid_constant__ structs) and
cuTensorMapEncodeTiled through libcuda (ctypes). Builds on fusezap.device.cuda (its device, context and NVRTC setup)."""
from __future__ import annotations
import ctypes, struct
from typing import Any
from fusezap.device.cuda import CUDADevice, NVRTCCompiler, _check, debug_sync, set_carveout

# shared device helpers (no headers: NVRTC built-ins and inline PTX only)
COMMON = r"""typedef unsigned int u32; typedef unsigned long long u64; typedef unsigned short u16; typedef unsigned char u8;
static __device__ __forceinline__ u32 fz_smem(const void* p) { u32 r; asm("{ .reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a; }" : "=r"(r) : "l"(p)); return r; }
static __device__ __forceinline__ u16 fz_f2h(float f) { u16 h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
static __device__ __forceinline__ u16 fz_f2bf(float f) { u16 h; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
static __device__ __forceinline__ u32 fz_pack_h(float a, float b) { u32 r; asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r; }
static __device__ __forceinline__ u32 fz_pack_bf(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r; }
"""

class ArchCompiler(NVRTCCompiler):
  """NVRTCCompiler for an explicit arch string (e.g. "sm_90a", whose wgmma/setmaxnreg the plain sm_90 target rejects)."""
  def __init__(self, dev: CUDADevice, arch: str):
    super().__init__(dev)
    self.arch, self.cubin = arch, True

_compilers: dict[tuple[int, str], ArchCompiler] = {}
def compiler(dev: CUDADevice, arch: str | None = None) -> NVRTCCompiler:
  if arch is None or arch == dev.compiler.arch: return dev.compiler
  if (c := _compilers.get(key := (id(dev), arch))) is None: _compilers[key] = c = ArchCompiler(dev, arch)
  return c

class Kernel:
  """One compiled library kernel: cuModuleLoadData + the dynamic shared memory opt-in."""
  def __init__(self, dev: CUDADevice, name: str, src: str, arch: str | None = None, smem: int = 0):
    self.dev, self.name, self.src, self.smem = dev, name, src, smem
    cu = dev.cu
    lib = compiler(dev, arch).compile_cached(src)
    dev.ctx()
    self.module = _check(cu.cuModuleLoadData(lib), f"cuModuleLoadData({name})")
    self.fxn = _check(cu.cuModuleGetFunction(self.module, name.encode()), f"cuModuleGetFunction({name})")
    if smem > 48 * 1024:
      _check(cu.cuFuncSetAttribute(self.fxn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem), f"smem {smem} for {name}")
    set_carveout(cu, self.fxn)
    self.regs = int(_check(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, self.fxn)))
    self.spills = int(_check(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, self.fxn)))
  def __call__(self, grid: tuple[int, ...], block: tuple[int, ...], args: list[Any], wait: bool = False) -> float | None:
    """args: ints/handles as ("p", v) 64-bit, ("i", v) int32, ("f", v) float32, ("b", bytes) raw (64-byte aligned copy)."""
    gs = (tuple(grid) + (1, 1, 1))[:3]
    if 0 in gs: return 0.0 if wait else None
    return self.bind(grid, block, args)(wait)
  def bind(self, grid: tuple[int, ...], block: tuple[int, ...], args: list[Any]) -> Any:
    """A launcher with the parameter block built once: f(wait=False) -> seconds | None (benchmarks, replays)."""
    gs, bs = (tuple(grid) + (1, 1, 1))[:3], (tuple(block) + (1, 1, 1))[:3]
    keep, ptrs = [], []
    for kind, v in args:
      if kind == "b":
        raw = ctypes.create_string_buffer(len(v) + 64)
        addr = (ctypes.addressof(raw) + 63) & ~63
        ctypes.memmove(addr, v, len(v))
        keep.append(raw); ptrs.append(addr)
        continue
      c = ctypes.c_uint64(int(v)) if kind == "p" else ctypes.c_int32(int(v)) if kind == "i" else ctypes.c_float(float(v))
      keep.append(c); ptrs.append(ctypes.addressof(c))
    params = (ctypes.c_void_p * max(len(ptrs), 1))(*ptrs)
    keep.append(params)
    dev, cu, fxn, smem, pp = self.dev, self.dev.cu, self.fxn, self.smem, ctypes.addressof(params)
    def launch(wait: bool = False) -> float | None:
      _ = keep
      dev.ctx()
      if wait: _check(cu.cuEventRecord(dev.ev_start, 0), "cuEventRecord")
      _check(cu.cuLaunchKernel(fxn, *gs, *bs, smem, dev.stream, pp, 0), f"cuLaunchKernel({self.name}, grid={gs}, block={bs})")
      debug_sync(dev, self.name, gs, bs)
      if not wait: return None
      _check(cu.cuEventRecord(dev.ev_end, 0), "cuEventRecord")
      _check(cu.cuEventSynchronize(dev.ev_end), f"kernel {self.name}")
      return float(_check(cu.cuEventElapsedTime(dev.ev_start, dev.ev_end), "cuEventElapsedTime")) * 1e-3
    return launch
  def __del__(self):
    try: self.dev.cu.cuModuleUnload(self.module)
    except Exception: pass

# ---------------------------------------------------------------- TMA descriptors

TMA_U8, TMA_U16, TMA_F32 = 0, 1, 7    # CUtensorMapDataType (fp8 travels as bytes, f16/bf16 as 16-bit words)
SWIZZLE_128B, L2_256B = 3, 3

_libcuda: Any = None
def tensor_map(addr: int, dtype_code: int, esize: int, dims: tuple[int, ...], strides_bytes: tuple[int, ...], box: tuple[int, ...],
               swizzle: int = SWIZZLE_128B) -> bytes:
  """cuTensorMapEncodeTiled: dims/box innermost first; strides_bytes for dims[1:] (multiples of 16). OOB reads fill 0."""
  global _libcuda
  if _libcuda is None: _libcuda = ctypes.CDLL("libcuda.so.1")
  r = len(dims)
  assert len(strides_bytes) == r - 1 and len(box) == r and all(s % 16 == 0 for s in strides_bytes), (dims, strides_bytes, box)
  out = ctypes.create_string_buffer(128 + 64)
  p = (ctypes.addressof(out) + 63) & ~63
  u64, u32 = ctypes.c_uint64, ctypes.c_uint32
  err = _libcuda.cuTensorMapEncodeTiled(ctypes.c_void_p(p), ctypes.c_int(dtype_code), u32(r), ctypes.c_void_p(addr),
                                        (u64 * r)(*dims), (u64 * max(r - 1, 1))(*strides_bytes), (u32 * r)(*box), (u32 * r)(*([1] * r)),
                                        ctypes.c_int(0), ctypes.c_int(swizzle), ctypes.c_int(L2_256B), ctypes.c_int(0))
  if err != 0: raise RuntimeError(f"cuTensorMapEncodeTiled failed ({err}): dims={dims} strides={strides_bytes} box={box} esize={esize}")
  return ctypes.string_at(p, 128)

def f32_bits(x: float) -> int: return struct.unpack("<i", struct.pack("<f", x))[0]
