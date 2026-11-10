"""CUDA backend: NVRTC + the driver API through cuda-bindings (SPEC §11.2). Implemented by P3-CUDA.

CUDADevice(CompiledDevice) for "CUDA" / "CUDA:<ordinal>": the device's primary context, made current before each driver
call when cuCtxGetCurrent shows another one (ordinals and other libraries can interleave in a thread). Kernels run in
issue order on the legacy default stream, so the synchronous H2D/D2H copies order with them and the host never sees a
half-written buffer.
  CUDAAllocator   LRUAllocator over cuMemAlloc / cuMemFree; handles are device addresses (int). copyin/copyout move
                  exactly the memoryview's bytes (a jit replay copies a prefix of a larger buffer). Out of memory -> MemoryError.
  NVRTCCompiler   CUDA C++ -> CUBIN for the device's compute capability (`--gpu-architecture=sm_XY`); when NVRTC is older
                  than the GPU it emits PTX for its newest arch instead and the driver JIT-compiles it. Options: --fmad=false
                  (no FMA contraction), precise div/sqrt, no fast math. The disk cache key holds the arch, NVRTC version
                  and options; cache files are written atomically (tmp + rename) since test workers share them.
  CUDAProgram     cuModuleLoadData + cuModuleGetFunction; __call__ = cuLaunchKernel (grid = global_size in blocks,
                  block = local_size) on dev.stream. wait=True times the launch with CUDA events and returns seconds.
  CUDAGraph       capture_graph(fn) records fn's launches (all of them go to dev.stream, which is a side stream under
                  stream capture) into a graph that fz.jit replays with one cuGraphLaunch (see fusezap/jit.py).
custom_kernels: the constructor installs the library kernels (fusezap.device.cuda_kernels: tensor-core "matmul" /
"matmul16", P3-GEMM; cuda_kernels.attention: "attention", P3-ATTN), which wrap `run` to launch their kind="custom" items
(FZ_CUDA_LIB=0 skips that); other kind="custom" items run through fusezap.schedule.run_custom, whose CUDA source is
compiled and launched like any kernel.
Construction raises (so Device.available("CUDA") is False) without cuda-bindings, a driver or a GPU.
"""
from __future__ import annotations
import ctypes, functools, os, threading
from typing import TYPE_CHECKING, Any
import numpy as np
from fusezap.helpers import getenv
from fusezap.device import CompiledDevice, Compiler, LRUAllocator, Program
from fusezap.codegen.cuda import CUDARenderer
if TYPE_CHECKING:
  from fusezap.ir import ScheduleItem

def _code(err: Any) -> int: return int(getattr(err, "value", err))

def _check(ret: Any, what: str = "") -> Any:
  """Unpack a cuda-bindings result tuple (err, *values), raising on an error code."""
  err, rest = (ret[0], ret[1:]) if isinstance(ret, tuple) else (ret, ())
  if _code(err) != 0:
    from cuda.bindings import driver as cu
    try: msg = cu.cuGetErrorString(err)[1].decode()
    except Exception: msg = str(err)
    if _code(err) == 2: raise MemoryError(f"CUDA out of memory{': ' + what if what else ''} ({msg})")
    raise RuntimeError(f"CUDA error {_code(err)} {getattr(err, 'name', '')}{' in ' + what if what else ''}: {msg}")
  return rest[0] if len(rest) == 1 else rest

def set_carveout(cu: Any, fxn: Any) -> None:
  """FZ_CUDA_CARVEOUT=<0..100> (I60 A/B; unset: the driver's choice): every kernel loaded after it asks for that
  percentage of the L1 / shared memory split, so consecutive kernels of a decode graph (GEMMs with ~200 KB of shared
  memory between elementwise kernels with none) need no reconfiguration of the split in between. Read per load."""
  if (v := os.environ.get("FZ_CUDA_CARVEOUT", "")) == "": return
  _check(cu.cuFuncSetAttribute(fxn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, int(v)), "carveout")

class CUDAAllocator(LRUAllocator):
  def __init__(self, dev: "CUDADevice"):
    super().__init__()
    self.dev = dev
    self.cache_max = getenv("FZ_ALLOC_CACHE_MB", 1024) << 20   # see LRUAllocator
  def free(self, handle: int, nbytes: int, evict: bool = True) -> None:
    # cuMemFree inside a stream capture would invalidate it: keep every block, evict after the capture
    super().free(handle, nbytes, evict=evict and not getattr(self.dev, "capturing", False))
  def _alloc(self, nbytes: int) -> int:
    self.dev.ctx()
    return int(_check(self.dev.cu.cuMemAlloc(nbytes), f"cuMemAlloc({nbytes})"))
  def _free(self, handle: int, nbytes: int) -> None:
    try:
      self.dev.ctx()
      self.dev.cu.cuMemFree(handle)
    except Exception: pass   # interpreter shutdown / dead context: nothing left to free
  def copyin(self, handle: int, src: memoryview) -> None:
    if not (n := src.nbytes): return
    self.dev.ctx()
    _check(self.dev.cu.cuMemcpyHtoD(handle, _addr(src), n), "cuMemcpyHtoD")
  def copyout(self, dst: memoryview, handle: int) -> None:
    if not (n := dst.nbytes): return
    self.dev.ctx()
    _check(self.dev.cu.cuMemcpyDtoH(_addr(dst), handle, n), "cuMemcpyDtoH")

def _ptr(h: Any) -> int:
  """Raw pointer value of a cuda-bindings handle (CUcontext etc.; NULL -> 0)."""
  if h is None: return 0
  try: return int(h)
  except TypeError: return int(h.getPtr())

def _addr(mv: memoryview) -> int:
  """Host address of a C-contiguous memoryview (read-only ones too)."""
  return np.frombuffer(mv.cast("B") if mv.format != "B" or mv.ndim != 1 else mv, np.uint8).ctypes.data

class NVRTCCompiler(Compiler):
  opts = ("--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false", "--std=c++17")
  def __init__(self, dev: "CUDADevice"):
    from cuda.bindings import nvrtc
    self.nvrtc = nvrtc
    self.version = tuple(_check(nvrtc.nvrtcVersion(), "nvrtcVersion"))
    supported = [int(a) for a in _check(nvrtc.nvrtcGetSupportedArchs(), "nvrtcGetSupportedArchs")]
    cc = dev.cc[0] * 10 + dev.cc[1]
    # a GPU newer than this NVRTC: PTX for the newest arch it knows, JIT-compiled by the driver at load
    self.cubin = cc in supported
    self.arch = f"sm_{cc}" if self.cubin else f"compute_{max(a for a in supported if a <= cc)}"
  def cache_key(self) -> str: return f"nvrtc {self.version} {self.arch} {' '.join(self.opts)}"
  def compile_cached(self, src: str) -> bytes:
    """Compiler.compile_cached with an atomic write: parallel test workers share the disk cache, and a reader must never
    load a half-written CUBIN (CUDA_ERROR_INVALID_IMAGE)."""
    from fusezap.helpers import CACHE_DIR, getenv, stable_hash
    path = CACHE_DIR / type(self).__name__ / stable_hash(self.cache_key() + "\0" + src)
    if path.exists() and not getenv("FZ_NOCACHE"): return path.read_bytes()
    lib = self.compile(src)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(lib)
    os.replace(tmp, path)
    return lib
  def compile(self, src: str) -> bytes:
    nv = self.nvrtc
    prog = _check(nv.nvrtcCreateProgram(src.encode(), b"fz_kernel.cu", 0, [], []), "nvrtcCreateProgram")
    try:
      opts = [f"--gpu-architecture={self.arch}".encode()] + [o.encode() for o in self.opts]
      err = nv.nvrtcCompileProgram(prog, len(opts), opts)[0]
      if _code(err) != 0:
        n = _check(nv.nvrtcGetProgramLogSize(prog))
        log = b" " * n
        nv.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"NVRTC compile failed ({self.arch}):\n{log.decode(errors='replace').rstrip(chr(0))}\n--- source ---\n{src}")
      get, size = (nv.nvrtcGetCUBIN, nv.nvrtcGetCUBINSize) if self.cubin else (nv.nvrtcGetPTX, nv.nvrtcGetPTXSize)
      out = b" " * _check(size(prog))
      _check(get(prog, out), "nvrtcGet binary")
      return out if self.cubin else out.rstrip(b"\0") + b"\0"
    finally: nv.nvrtcDestroyProgram(prog)

def debug_sync(dev: Any, name: str, grid: Any = None, block: Any = None) -> None:
  """FZ_CUDA_SYNC=1: synchronize the stream after every kernel launch and raise naming THIS kernel if it faulted (a
  kernel error is asynchronous: without it, it surfaces at some later driver call, e.g. the next cuModuleLoadData).
  Skipped during a stream capture. Slow: for pinpointing a fault only."""
  if not getenv("FZ_CUDA_SYNC", 0) or getattr(dev, "capturing", False): return
  _check(dev.cu.cuStreamSynchronize(dev.stream), f"FZ_CUDA_SYNC: kernel {name} (grid={grid}, block={block}) faulted")

class CUDAProgram(Program):
  def __init__(self, dev: "CUDADevice", name: str, lib: bytes):
    super().__init__(name, lib)
    self.dev, cu = dev, dev.cu
    dev.ctx()
    self.module = _check(cu.cuModuleLoadData(lib), f"cuModuleLoadData({name})")
    self.fxn = _check(cu.cuModuleGetFunction(self.module, name.encode()), f"cuModuleGetFunction({name})")
    set_carveout(cu, self.fxn)
    self.max_threads = int(_check(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, self.fxn)))
  def __call__(self, *bufs: Any, global_size: tuple[int, ...] = (1, 1, 1), local_size: tuple[int, ...] | None = None,
               vals: tuple[int, ...] = (), wait: bool = False) -> float | None:
    gs = (tuple(int(x) for x in global_size) + (1, 1, 1))[:3]
    ls = (tuple(int(x) for x in (local_size or ())) + (1, 1, 1))[:3]
    if ls[0] * ls[1] * ls[2] > self.max_threads: raise RuntimeError(f"local size {ls} exceeds {self.max_threads} threads for {self.name}")
    if 0 in gs: return 0.0 if wait else None
    cu, dev = self.dev.cu, self.dev
    dev.ctx()
    # kernelParams: a void*[] pointing at each argument's value (64-bit device addresses, then 32-bit ints)
    ptrs = (ctypes.c_uint64 * max(len(bufs), 1))(*[int(b) for b in bufs])
    ints = (ctypes.c_int32 * max(len(vals), 1))(*[int(v) for v in vals])
    pa, ia = ctypes.addressof(ptrs), ctypes.addressof(ints)
    params = (ctypes.c_void_p * max(len(bufs) + len(vals), 1))(*[pa + 8 * i for i in range(len(bufs))], *[ia + 4 * i for i in range(len(vals))])
    if wait: _check(cu.cuEventRecord(dev.ev_start, 0), "cuEventRecord")
    _check(cu.cuLaunchKernel(self.fxn, *gs, *ls, 0, dev.stream, ctypes.addressof(params), 0), f"cuLaunchKernel({self.name}, grid={gs}, block={ls})")
    debug_sync(dev, self.name, gs, ls)
    if not wait: return None
    _check(cu.cuEventRecord(dev.ev_end, 0), "cuEventRecord")
    _check(cu.cuEventSynchronize(dev.ev_end), f"kernel {self.name}")
    return float(_check(cu.cuEventElapsedTime(dev.ev_start, dev.ev_end), "cuEventElapsedTime")) * 1e-3
  def __del__(self):
    try: self.dev.cu.cuModuleUnload(self.module)
    except Exception: pass

class CUDAGraph:
  """An instantiated CUDA graph; launch() enqueues it on the legacy default stream (ordered with eager launches and copies)."""
  def __init__(self, dev: "CUDADevice", graph: Any):
    self.dev, self.graph = dev, graph
    self.exe = _check(dev.cu.cuGraphInstantiate(graph, 0), "cuGraphInstantiate")
  def launch(self) -> None:
    self.dev.ctx()
    _check(self.dev.cu.cuGraphLaunch(self.exe, 0), "cuGraphLaunch")
  def __del__(self):
    try: self.dev.cu.cuGraphExecDestroy(self.exe); self.dev.cu.cuGraphDestroy(self.graph)
    except Exception: pass

class CUDADevice(CompiledDevice):
  def __init__(self, name: str):
    try: from cuda.bindings import driver as cu
    except ImportError as e: raise RuntimeError("CUDA backend needs cuda-bindings (pip install cuda-bindings; Linux + NVIDIA driver)") from e
    self.cu, self.name = cu, name
    self.stream, self._capture_stream = 0, None   # launches go to self.stream: the legacy default stream, or a capture
    _check(cu.cuInit(0), "cuInit")
    ordinal = int(name.split(":")[1]) if ":" in name else 0
    if ordinal >= (n := int(_check(cu.cuDeviceGetCount(), "cuDeviceGetCount"))): raise RuntimeError(f"no CUDA device {ordinal} ({n} found)")
    self.device = _check(cu.cuDeviceGet(ordinal), "cuDeviceGet")
    self.context = _check(cu.cuDevicePrimaryCtxRetain(self.device), "cuDevicePrimaryCtxRetain")
    self._ctx_ptr = _ptr(self.context)
    self.ctx()
    A = cu.CUdevice_attribute
    attr = lambda a: int(_check(cu.cuDeviceGetAttribute(a, self.device)))
    self.cc = (attr(A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR), attr(A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR))
    self.gpu_name = _check(cu.cuDeviceGetName(256, self.device)).split(b"\0")[0].decode()
    self.ev_start, self.ev_end = _check(cu.cuEventCreate(0)), _check(cu.cuEventCreate(0))
    self.allocator = CUDAAllocator(self)
    self.compiler = NVRTCCompiler(self)
    self.renderer = CUDARenderer(self.compiler.arch)
    self.runtime = functools.partial(CUDAProgram, self)
    self.custom_kernels: dict[str, Any] = {}
    if getenv("FZ_CUDA_LIB", 1): self._install_library()
  def _install_library(self) -> None:
    """Offer the library kernels to the scheduler hook (SPEC §11.2): tensor-core matmul (P3-GEMM; f32 3xTF32, f16/bf16
    mma.sync / wgmma, sm_80+) and float32 attention (P3-ATTN's SIMT kernel). FZ_CUSTOM=0 still keeps the composed kernels;
    FZ_CUDA_LIB=0 at construction leaves custom_kernels empty."""
    from fusezap.device import cuda_kernels
    from fusezap.device.cuda_kernels import attention
    cuda_kernels.install(self)
    attention.install(self)
  def ctx(self) -> None:
    """Make this device's context current in the calling thread. The current context is per thread and shared by every
    device (and by other libraries such as torch), so check the driver's current context on each call and switch when it
    differs; a per-device "already set" flag would go stale as soon as a second ordinal or another library switched."""
    if _ptr(_check(self.cu.cuCtxGetCurrent(), "cuCtxGetCurrent")) == self._ctx_ptr: return
    _check(self.cu.cuCtxSetCurrent(self.context), "cuCtxSetCurrent")
  def run(self, si: "ScheduleItem") -> float | None:
    if si.kind == "custom":
      from fusezap.schedule import run_custom
      return run_custom(self, si)
    return super().run(si)
  def mem_total(self) -> int:
    """The device's total memory in bytes (cuDeviceTotalMem)."""
    return int(_check(self.cu.cuDeviceTotalMem(self.device), "cuDeviceTotalMem"))
  def mem_free(self) -> int:
    """Bytes the driver can still allocate on this device, plus what the allocator's free list holds."""
    self.ctx()
    return int(_check(self.cu.cuMemGetInfo(), "cuMemGetInfo")[0]) + self.allocator.cached
  def copy_async(self, dst: int, src: int, nbytes: int) -> None:
    """Device-to-device copy, ordered with the kernels on the current stream."""
    if nbytes:
      self.ctx()
      _check(self.cu.cuMemcpyDtoDAsync(dst, src, nbytes, self.stream), "cuMemcpyDtoDAsync")
  def capture_graph(self, fn: Any) -> "CUDAGraph | None":
    """Run fn() with every kernel launch recorded into a CUDA graph instead of executed (stream capture on a side stream,
    thread-local mode, so a stray legacy-stream call or synchronous copy fails the capture instead of running). Returns
    the instantiated graph, or None if the capture failed (then nothing ran: the caller runs fn normally)."""
    cu = self.cu
    self.ctx()
    if self._capture_stream is None: self._capture_stream = _ptr(_check(cu.cuStreamCreate(0), "cuStreamCreate"))
    st = self._capture_stream
    _check(cu.cuStreamBeginCapture(st, cu.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL), "cuStreamBeginCapture")
    self.stream, ok, self.capturing = st, False, True
    try:
      fn()
      ok = True
    except RuntimeError as e:
      if getenv("FZ_DEBUG", 0) >= 1: print(f"CUDA graph capture failed, replaying launch by launch: {e}")
    finally:
      self.stream, self.capturing = 0, False
      err, graph = cu.cuStreamEndCapture(st)
    if not ok or _code(err) != 0 or graph is None: return None
    try: return CUDAGraph(self, graph)
    except RuntimeError: return None
  def synchronize(self) -> None:
    self.ctx()
    _check(self.cu.cuCtxSynchronize(), "cuCtxSynchronize (an earlier kernel failed)")
