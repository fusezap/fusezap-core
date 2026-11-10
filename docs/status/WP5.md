# WP5 — Metal renderer + device

**Status:** done

**Gate:** `uv run pytest tests/test_metal.py tests/test_skeleton.py` -> `65 passed`
(needs `pyobjc-framework-Metal`. It is the `metal` extra in pyproject and is already in uv.lock. Install it with `uv sync --extra metal --inexact`.)

## Built
- `fusezap/codegen/metal.py`: `MetalRenderer(CStyleRenderer)`
  - Signature: `kernel void name(device T* dataN [[buffer(N)]], ..., constant int& var [[buffer(n_bufs+i)]], uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]])`. The prelude is `#include <metal_stdlib>` / `using namespace metal;`.
  - SPECIAL `gidxK` renders as `(int)gid.{x,y,z}` and `lidxK` as `(int)lid.{x,y,z}`. There are no loops.
  - Types are `bool/uchar/char/short/int/long/half/bfloat/float`. bfloat16 is native `bfloat` in both registers and memory. A plain copy or a bitcast keeps the exact bits, NaN payloads included.
  - On half and bfloat, `exp2/log2/sin/sqrt` are computed in float and cast back. RECIP is `((T)1/(x))`.
  - Conversions between bfloat and int/bool go through float. Bitcasts use `as_type<T>(x)`, except to/from bool (MSL forbids it): `x != 0` into bool, `(T)(x)` out of bool.
  - Buffer slots: Metal allows `[[buffer(0..30)]]` only. `metal_layout(nbufs, nvars)` (shared with the device) picks the layout: `plain` (the SPEC signature) when bufs + vars <= 31; `packed` (bufs at 0..n-1, all vars in one `constant int* fz_vars [[buffer(n)]]`, unpacked into `int var = fz_vars[i];` at the kernel top) when bufs <= 30; `argbuf` (> 30 bufs: a `constant fz_args_<name>&` struct of device pointers at buffer(0), filled from `gpuAddress()` + `useResource`, vars at buffer(1)). No scheduler cap is needed.
  - Constants: f32 literals are rounded to f32 first, so an overflow becomes `INFINITY` instead of an out-of-range literal. bf16 constants are rounded to bf16 on the host (RNE, ±inf above the f32 max). int64 constants render as `((long)v)`, because MSL has no `ll`.
- `fusezap/device/metal.py`
  - `METALDevice`: raises `RuntimeError` when the platform is not macOS, pyobjc is missing, or there is no MTLDevice or command queue. It holds one command queue and an `in_flight` list. `synchronize()` waits on all in-flight command buffers and raises if any reported an error.
  - Libraries are compiled with `newLibraryWithSource:options:error:` in safe math mode with precise float functions (fast math is off). They are LRU-cached per source.
  - `MetalAllocator(LRUAllocator)`: `MTLResourceStorageModeShared` buffers. copyin/copyout call `synchronize()` and then memcpy through `contents()`.
  - `MetalCompiler`: `compile` validates the source with the runtime compiler, so errors include the source, and returns the source bytes as the "binary". `compile_cached` therefore caches the source.
  - `MetalProgram(dev, name, lib)` builds the pipeline state. `device.runtime` is `functools.partial(MetalProgram, device)`, so `device.runtime(name, lib)` works. `__call__` runs `dispatchThreadgroups(global_size, local_size or (1,1,1))`, binding buffers and vars per `metal_layout` (setBytes for vars and for the argument-buffer address table up to 4 KB, a temporary MTLBuffer beyond that).
    - Without `wait` it is asynchronous and queues the command buffer as in flight.
    - With `wait=True` it returns GPU seconds (`GPUEndTime - GPUStartTime`).
    - It rejects local sizes above `maxTotalThreadsPerThreadgroup`.
- `tests/test_metal.py` (37 tests) compares results against torch/numpy on hand-built UOp programs:
  - Launch shapes: elementwise, 3-D SPECIAL, gid + lid launch.
  - Arithmetic: SUM/MAX reductions, a guarded LOAD and IF, f32 unary math, IEEE special values (proves fast math is off).
  - Conversions: casts (including int8/uint8), int32 IDIV/MOD with negative operands and int64 math.
  - Low precision: f16, bf16 (arithmetic, exp2, f32 reduction, casts), NEG of negative constants, out-of-f32-range constants, bf16 NaN bit preservation.
  - Review fixes: wide kernels (31/0, 28/3, 29/3, 30/1, 31/1, 32/0, 40/3, 600/2 buffers/vars) and bool<->uint8/int8 bitcasts vs torch `view`.
  - Runtime: a symbolic var (one compile, three sizes), async launches with synchronize, GPU timing, the compile cache, compile errors, too many threads, LRU reuse with shared storage, and construction failing without pyobjc.

## Known gaps / notes
- The `argbuf` layout needs Metal 3 / argument buffers tier 2 (all Apple silicon). The SPEC §7.3 signature text only holds while bufs + vars <= 31.
- `global_size` counts threadgroups and `local_size` counts threads per group. The SPECIAL sizes emitted by lowering must equal the launch dims, because the renderer adds no bounds guard.
- `METALDevice.run` uses `lower.run_compiled` (WP4b), which is not tested here. Now that Metal is available, the METAL tier of `tests/test_ops.py` runs instead of being skipped.
- Metal has no float64. int64 is `long` (supported, but slow on Apple GPUs).
- bf16 arithmetic uses Metal's native `bfloat` operators. The tests show results close to torch's per-op rounding, but bit-exactness is not guaranteed.
