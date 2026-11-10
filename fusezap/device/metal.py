"""Metal backend via pyobjc (SPEC §8). Implemented by WP5; custom kernels by P2-MTL.

METALDevice(CompiledDevice): MTLCreateSystemDefaultDevice, one command queue,
MetalAllocator (shared storage mode, LRU), MetalCompiler (runtime newLibraryWithSource; the cached
"bytes" may simply be the source, since there is no offline compiler), MetalProgram (pipeline state,
dispatchThreadgroups, optional wait + GPU timestamps). synchronize() waits on the last command buffer.
Construction raises if Metal is unavailable, so Device.available("METAL") is False.

Custom kernels (SPEC §11.1): METALDevice.custom_kernels offers "matmul" (simdgroup-matrix GEMM) and "attention" (flash
attention) to the P2-CUSTOM scheduler hook (float32; FZ_CUSTOM=0 turns matching off), and METALDevice.run executes
kind="custom" items itself with the right grid (run_custom launches one thread). The kernels take float32, float16 or
bfloat16 inputs (float32 accumulation), chosen from the item's buffer dtypes. The scheduler only matches float32,
non-causal patterns, so `matmul(a, b)` and `attention(q, k, v, causal)` below run the half/bf16 and causal variants on
Tensors directly (inference only: the result has no autograd history).
"""
from __future__ import annotations
import functools, math, struct, sys
from typing import TYPE_CHECKING, Any
from fusezap.device import CompiledDevice, Compiler, LRUAllocator, Program
from fusezap.dtype import dtypes
from fusezap.helpers import getenv, prod
from fusezap.codegen.metal import (MetalRenderer, metal_layout, render_matmul, render_attention, matmul_launch, attention_launch,
                                   MATMUL_PARAMS, ATTN_PARAMS, flash_config, render_attention_flash, attention_flash_launch,
                                   render_conv, conv_launch)
if TYPE_CHECKING:
  from fusezap.ir import ScheduleItem
  from fusezap.tensor import Tensor

def _compile_options(Metal: Any) -> Any:
  """Numerics must match PyTorch: no fast math (precise transcendentals, IEEE inf/nan, no reassociation)."""
  opts = Metal.MTLCompileOptions.new()
  if hasattr(opts, "setMathMode_"):
    opts.setMathMode_(getattr(Metal, "MTLMathModeSafe", 0))
    if hasattr(opts, "setMathFloatingPointFunctions_"): opts.setMathFloatingPointFunctions_(getattr(Metal, "MTLMathFloatingPointFunctionsPrecise", 1))
  else: opts.setFastMathEnabled_(False)
  return opts

def _check(cb: Any) -> None:
  if (err := cb.error()) is not None: raise RuntimeError(f"Metal command buffer failed: {err}")

class MetalAllocator(LRUAllocator):
  """MTLBuffers in shared storage: the host reads/writes `contents()` directly, after waiting for in-flight work."""
  def __init__(self, dev: "METALDevice"):
    super().__init__()
    self.dev = dev
    self.owns_new: bool | None = None   # does newBufferWithLength's +1 need our release()? (decided on the first buffer)
  def _alloc(self, nbytes: int) -> Any:
    buf = self.dev.device.newBufferWithLength_options_(nbytes, self.dev.Metal.MTLResourceStorageModeShared)
    if buf is None: raise MemoryError(f"Metal could not allocate {nbytes} bytes")
    if self.owns_new is None: self.owns_new = buf.retainCount() > 1
    return buf
  def _free(self, handle: Any, nbytes: int) -> None:
    """PyObjC (12.x) keeps the +1 of newBufferWithLength:options: on top of its proxy's own retain (a fresh buffer
    has retainCount 2), so the proxy dying never freed the MTLBuffer: every block that left the LRU cache leaked
    (I81: +16 GB of footprint per 512^2 VAE decode once its cache was trimmed). Balance it here, as tinygrad's
    PyObjC-era allocator did. Argument-buffer kernels bind by gpuAddress (useResource, no retain), so while work is
    in flight the release waits for the next synchronize()."""
    if not self.owns_new: return
    if self.dev.in_flight: self.dev.pending_release.append(handle)
    else: handle.release()
  def free_cache(self) -> None:
    self.dev.synchronize()   # so the cached blocks are released now (an alloc retrying after MemoryError needs that)
    super().free_cache()
  def _view(self, handle: Any, n: int) -> memoryview: return handle.contents().as_buffer(handle.length())[:n]
  def copyin(self, handle: Any, src: memoryview) -> None:
    src = src.cast("B") if src.format != "B" or src.ndim != 1 else src
    self.dev.synchronize()
    self._view(handle, src.nbytes)[:] = src
  def copyout(self, dst: memoryview, handle: Any) -> None:
    dst = dst.cast("B") if dst.format != "B" or dst.ndim != 1 else dst
    self.dev.synchronize()
    dst[:] = self._view(handle, dst.nbytes)

class MetalCompiler(Compiler):
  """There is no offline `metal` compiler here: compile() validates the source with newLibraryWithSource (so errors surface
  at compile time, with the source) and the cached "binary" is the source itself. MetalProgram compiles it again (fast)."""
  def __init__(self, dev: "METALDevice"): self.dev = dev
  def compile(self, src: str) -> bytes:
    self.dev.library(src)
    return src.encode()

class MetalProgram(Program):
  def __init__(self, dev: "METALDevice", name: str, lib: bytes):
    super().__init__(name, lib)
    self.dev = dev
    fxn = dev.library(lib.decode()).newFunctionWithName_(name)
    if fxn is None: raise RuntimeError(f"Metal library has no kernel named {name!r}")
    self.pipeline, err = dev.device.newComputePipelineStateWithFunction_error_(fxn, None)
    if self.pipeline is None: raise RuntimeError(f"Metal pipeline creation failed for {name}: {err}")
    self.max_threads = self.pipeline.maxTotalThreadsPerThreadgroup()
  def __call__(self, *bufs: Any, global_size: tuple[int, ...] = (1, 1, 1), local_size: tuple[int, ...] | None = None,
               vals: tuple[int, ...] = (), wait: bool = False) -> float | None:
    Metal = self.dev.Metal
    gs = (tuple(int(x) for x in global_size) + (1, 1, 1))[:3]
    ls = (tuple(int(x) for x in (local_size or ())) + (1, 1, 1))[:3]
    if ls[0] * ls[1] * ls[2] > self.max_threads: raise RuntimeError(f"local size {ls} exceeds {self.max_threads} threads for {self.name}")
    if 0 in gs: return 0.0 if wait else None
    cb = self.dev.queue.commandBuffer()
    enc = cb.computeCommandEncoder()
    enc.setComputePipelineState_(self.pipeline)
    layout = metal_layout(len(bufs), len(vals))   # must match MetalRenderer.render_kernel
    if layout == "argbuf":
      addrs = struct.pack(f"<{len(bufs)}Q", *[b.gpuAddress() for b in bufs])
      if len(addrs) <= 4096: enc.setBytes_length_atIndex_(addrs, len(addrs), 0)   # setBytes is limited to 4 KB
      else: enc.setBuffer_offset_atIndex_(self.dev.device.newBufferWithBytes_length_options_(addrs, len(addrs), 0), 0, 0)
      rw = Metal.MTLResourceUsageRead | Metal.MTLResourceUsageWrite
      for b in bufs: enc.useResource_usage_(b, rw)
    else:
      for i, b in enumerate(bufs): enc.setBuffer_offset_atIndex_(b, 0, i)
    if layout == "plain":
      for i, v in enumerate(vals): enc.setBytes_length_atIndex_(struct.pack("<i", int(v)), 4, len(bufs) + i)
    elif vals:
      enc.setBytes_length_atIndex_(struct.pack(f"<{len(vals)}i", *[int(v) for v in vals]), 4 * len(vals), 1 if layout == "argbuf" else len(bufs))
    enc.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSizeMake(*gs), Metal.MTLSizeMake(*ls))
    enc.endEncoding()
    cb.commit()
    if not wait:
      self.dev.in_flight.append(cb)
      return None
    cb.waitUntilCompleted()
    _check(cb)
    return cb.GPUEndTime() - cb.GPUStartTime()

class METALDevice(CompiledDevice):
  def __init__(self, name: str):
    if sys.platform != "darwin": raise RuntimeError("Metal is only available on macOS")
    try: import Metal
    except ImportError as e: raise RuntimeError("Metal backend needs pyobjc-framework-Metal (pip install 'fusezap[metal]')") from e
    self.Metal, self.name = Metal, name
    self.device = Metal.MTLCreateSystemDefaultDevice()
    if self.device is None: raise RuntimeError("no Metal device available")
    self.queue = self.device.newCommandQueue()
    if self.queue is None: raise RuntimeError("could not create a Metal command queue")
    self.in_flight: list[Any] = []
    self.pending_release: list[Any] = []   # MTLBuffers MetalAllocator._free releases at the next synchronize()
    self._opts = _compile_options(Metal)
    self.allocator = MetalAllocator(self)
    self.renderer = MetalRenderer()
    self.compiler = MetalCompiler(self)
    self.runtime = functools.partial(MetalProgram, self)
  @functools.lru_cache(maxsize=256)
  def library(self, src: str) -> Any:
    lib, err = self.device.newLibraryWithSource_options_error_(src, self._opts, None)
    if lib is None: raise RuntimeError(f"Metal compile failed:\n{err}\n--- source ---\n{src}")
    return lib
  @functools.cached_property
  def custom_kernels(self) -> dict[str, Any]:
    from fusezap.schedule import CustomKernel
    # "attention" source depends on the head dims, so it is rendered per call by run_custom_call (the entry marks support)
    from fusezap.schedule import CONV_PARAMS
    ks = {"matmul": CustomKernel("fz_matmul", {"METAL": render_matmul()}, ("C", "A", "B") + MATMUL_PARAMS, 3),
          "attention": CustomKernel("fz_attention", {"METAL": render_attention}, ("O", "Q", "K", "V") + ATTN_PARAMS, 4)}
    # I81 D8: implicit-GEMM conv2d for the composed Tensor.conv2d (schedule._match_conv); FZ_METAL_CONV=0 keeps im2col + GEMM.
    # Used whenever the matcher proves im2col, at any size (decided: no size threshold; small convs lose ~2 ms at
    # 128^2, the large VAE levels and conv_out win, and the im2col buffer is never written)
    if getenv("FZ_METAL_CONV", 1): ks["conv"] = CustomKernel("fz_conv", {"METAL": render_conv}, ("C", "A", "X") + CONV_PARAMS, 3)
    return ks
  def run(self, si: "ScheduleItem") -> float | None:
    if si.kind == "custom": return self.run_custom_call(si.ast[0].kernel.name, si.ast[0].params, si.bufs)
    return super().run(si)
  def run_custom_call(self, name: str, params: tuple[int, ...], bufs: tuple[Any, ...], wait: bool | None = None) -> float | None:
    """Launch library kernel `name` ("fz_matmul", "fz_attention", "fz_attention_causal") on Buffers (outputs first)."""
    dts = tuple(b.dtype for b in bufs[:2])
    if name == "fz_matmul":
      Bt, M, N = params[:3]
      key, render = (name, dts), lambda: render_matmul(name, dts[1], dts[0])
      gs, ls = matmul_launch(M, N, Bt)
    elif name in ("fz_attention", "fz_attention_causal"):
      Bt, L, D, Dv = params[0], params[1], params[3], params[4]
      if dts == (dtypes.float32, dtypes.float32) and getenv("FZ_METAL_FLASH", 1) and (cfg := flash_config(D, Dv)) is not None:
        # I81 D8: float32 flash kernel (Q and O slices in registers, direct fragment loads): long sequences, head dims to 512
        key, render = (name, dts, D, Dv, cfg), lambda: render_attention_flash(name, D, Dv, name.endswith("causal"), cfg)
        gs, ls = attention_flash_launch(L, Bt, cfg[0], cfg[1])
      else:
        key, render = (name, dts, D, Dv), lambda: render_attention(name, D, Dv, dts[1], dts[0], name.endswith("causal"))
        gs, ls = attention_launch(L, Bt, D, Dv)
    elif name == "fz_conv":
      from fusezap.codegen.metal import _conv_cfg
      cfg = _conv_cfg(params[0])
      key, render = (name, cfg), lambda: render_conv(name, cfg[0], cfg[1])
      gs, ls = conv_launch(params[0], params[1])
    else: raise RuntimeError(f"METAL has no custom kernel {name!r}")
    if (prg := self._custom_progs.get(key)) is None:
      src = render()
      if getenv("FZ_DEBUG", 0) >= 4: print(src)
      self._custom_progs[key] = prg = self.runtime(name, self.compiler.compile_cached(src))
    if prod(gs) == 0 or bufs[0].size == 0: return None
    return prg(*[b.ensure_allocated().handle for b in bufs], global_size=gs, local_size=ls, vals=tuple(int(p) for p in params),
               wait=getenv("FZ_DEBUG", 0) >= 1 if wait is None else wait)
  @functools.cached_property
  def _custom_progs(self) -> dict[tuple, MetalProgram]: return {}
  def synchronize(self) -> None:
    cbs, self.in_flight = self.in_flight, []
    for cb in cbs: cb.waitUntilCompleted()
    rel, self.pending_release = self.pending_release, []
    for h in rel: h.release()   # MetalAllocator._free deferred these until no kernel could still read them
    for cb in cbs: _check(cb)

# ---------------------------------------------------------------- direct entry points (Tensors in, Tensor out)

_FLOATS = (dtypes.float32, dtypes.float16, dtypes.bfloat16)

def _base(t: "Tensor") -> Any:
  """t's elements as one realized row-major METAL Buffer."""
  assert t.device.split(":")[0].upper() == "METAL", f"Metal custom kernels need METAL tensors, got {t.device}"
  return t._realized_contiguous().realized

def _batch(ts: list["Tensor"], tail: int) -> tuple[list["Tensor"], tuple[int, ...]]:
  """Broadcast the leading (batch) dims of ts (each keeps its last `tail` dims); size-1 batches stay unexpanded."""
  nb = max(t.ndim - tail for t in ts)
  ts = [t.reshape((1,) * (nb - (t.ndim - tail)) + tuple(t.shape)) for t in ts]
  bs = tuple(max(t.shape[i] for t in ts) for i in range(nb))
  out = []
  for t in ts:
    if prod(t.shape[:nb]) != 1 and tuple(t.shape[:nb]) != bs: t = t.expand(bs + tuple(t.shape[nb:]))
    out.append(t)
  return out, bs

def _out(dev: str, shape: tuple[int, ...], dt: Any) -> Any:
  from fusezap.device import Buffer
  return Buffer(dev, prod(shape), dt)

def _wrap(buf: Any, shape: tuple[int, ...]) -> "Tensor":
  from fusezap.lazy import LazyBuffer
  from fusezap.tensor import Tensor
  return Tensor(LazyBuffer.from_buffer(buf, shape))

def matmul(a: "Tensor", b: "Tensor") -> "Tensor":
  """a @ b (..., M, K) x (..., K, N) with the simdgroup GEMM, batch dims broadcast; a and b share a float dtype
  (float32/float16/bfloat16); float32 accumulation, output in that dtype like Tensor.matmul."""
  assert a.ndim >= 2 and b.ndim >= 2 and a.shape[-1] == b.shape[-2], f"matmul shapes {a.shape} @ {b.shape}"
  assert a.dtype == b.dtype and a.dtype in _FLOATS, f"matmul needs one float dtype, got {a.dtype}, {b.dtype}"
  (a, b), bs = _batch([a, b], 2)
  M, K, N, Bt = a.shape[-2], a.shape[-1], b.shape[-1], prod(bs)
  A, B = _base(a), _base(b)
  C = _out(a.device, (Bt, M, N), a.dtype)
  sab, sbb = (0 if A.size == M * K else M * K), (0 if B.size == K * N else K * N)
  if Bt and M and N: _dev(a.device).run_custom_call("fz_matmul", (Bt, M, N, K, sab, K, 1, sbb, N, 1, M * N, N, 1, 0, 0), (C, A, B))
  return _wrap(C, bs + (M, N))

def attention(q: "Tensor", k: "Tensor", v: "Tensor", causal: bool = False, scale: float | None = None) -> "Tensor":
  """Flash attention == scaled_dot_product_attention(q, k, v, is_causal=causal): q (..., L, D), k (..., S, D), v (..., S, Dv),
  batch dims broadcast, one float dtype; float32 softmax and accumulation, output in the input dtype."""
  assert q.ndim >= 2 and k.ndim >= 2 and v.ndim >= 2 and q.shape[-1] == k.shape[-1] and k.shape[-2] == v.shape[-2], \
    f"attention shapes {q.shape}, {k.shape}, {v.shape}"
  assert q.dtype == k.dtype == v.dtype and q.dtype in _FLOATS, f"attention needs one float dtype, got {q.dtype}, {k.dtype}, {v.dtype}"
  from fusezap.codegen.metal import attention_block
  if q.dtype != dtypes.float32 and attention_block(q.shape[-1], v.shape[-1]) is None and getenv("FZ_METAL_FLASH", 1) and \
     flash_config(q.shape[-1], v.shape[-1]) is not None:   # wide heads (VAE 512): the float32 flash kernel on float32 copies
    return attention(q.float(), k.float(), v.float(), causal, scale).cast(q.dtype)
  (q, k, v), bs = _batch([q, k, v], 2)
  L, D, S, Dv, Bt = q.shape[-2], q.shape[-1], k.shape[-2], v.shape[-1], prod(bs)
  Q, Kb, V = _base(q), _base(k), _base(v)
  O = _out(q.device, (Bt, L, Dv), q.dtype)
  sc = 1.0 / math.sqrt(D) if scale is None else scale
  bst = lambda buf, n: 0 if buf.size == n else n
  params = (Bt, L, S, D, Dv, bst(Q, L * D), D, 1, bst(Kb, S * D), D, 1, bst(V, S * Dv), Dv, 1, L * Dv, Dv, 1, 0, 0, 0,
            _bits(sc), _bits(math.log2(math.e)))
  if Bt and L and Dv: _dev(q.device).run_custom_call("fz_attention_causal" if causal else "fz_attention", params, (O, Q, Kb, V))
  return _wrap(O, bs + (L, Dv))

def _bits(x: float) -> int: return struct.unpack("<i", struct.pack("<f", x))[0]
def _dev(name: str) -> "METALDevice":
  from fusezap.device import Device
  return Device[name]  # type: ignore[return-value]
