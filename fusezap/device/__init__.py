"""Device registry, buffers and backend interfaces (SPEC §8). Owned by the integrator.

Backends live in fusezap/device/{npy,cpu,metal,cuda,neuron}.py and each defines a class named `<NAME>Device`
(NPYDevice, CPUDevice, METALDevice, CUDADevice, NEURONDevice) subclassing `DeviceBase` (or `CompiledDevice`).
`_modules` maps a device to a module of another name (the NEURON test tiers NKISHIM/NKISIM live in neuron.py).
"""
from __future__ import annotations
import functools, importlib, weakref
from typing import TYPE_CHECKING, Any
from fusezap.dtype import DType, to_np
from fusezap.helpers import getenv

if TYPE_CHECKING:
  import numpy as np
  from fusezap.ir import ScheduleItem
  from fusezap.codegen.uops import ProgramSpec

# ---------------------------------------------------------------- allocators

class Allocator:
  def alloc(self, nbytes: int) -> Any: raise NotImplementedError
  def free(self, handle: Any, nbytes: int) -> None: raise NotImplementedError
  def copyin(self, handle: Any, src: memoryview) -> None: raise NotImplementedError
  def copyout(self, dst: memoryview, handle: Any) -> None: raise NotImplementedError

class LRUAllocator(Allocator):
  """Wraps _alloc/_free with a size-bucketed free list so steady-state training does no real allocs.
  cache_max (bytes, None: unbounded) caps what the free list holds. CUDA sets it (FZ_ALLOC_CACHE_MB, default 1024):
  tracing a forward allocates all of its intermediates at once and each traced size frees a different set of sizes, so
  an unbounded list kept every traced size's activations (gigabytes for a 2048-row 7B prefill) away from other
  processes on the same GPU (speed run 4 round 8: vLLM died next to FuseZap).
  Over the cap, the least recently used SIZES go back to the driver, never the block just freed (unless that one size
  alone is over the cap). Round 9/10's first policy instead refused the incoming block when the list was full: once the
  echo-logprob scorer's traces had filled the list with cold blocks (Validate's accuracy check), every block a serving
  step freed went to cuMemFree and was cuMemAlloc'ed again the next step, each a device synchronization: -5% at c32 in
  Validate only (the bench never scores, so its list never filled).
  `held` is what the allocator holds from the driver now (in use plus cached), `cached` the free list's share of it;
  `counters()` counts driver allocs / frees and evicted bytes."""
  cache_max: int | None = None
  def __init__(self):
    self.cache: dict[int, list[Any]] = {}   # size -> free blocks; insertion order = least recently used size first
    self.held = self.cached = 0
    self.n_alloc = self.b_alloc = self.n_free = self.b_free = self.b_evict = 0
  def counters(self) -> dict[str, int]:
    return {"allocs": self.n_alloc, "alloc_bytes": self.b_alloc, "frees": self.n_free, "free_bytes": self.b_free,
            "evicted_bytes": self.b_evict, "held": self.held, "cached": self.cached}
  def alloc(self, nbytes: int) -> Any:
    hs = self.cache.pop(nbytes, None)
    if hs:
      self.cached -= nbytes
      h = hs.pop()
      if hs: self.cache[nbytes] = hs   # back in as the most recently used size
      return h
    try: h = self._alloc(nbytes)
    except MemoryError:
      self.free_cache()
      h = self._alloc(nbytes)
    self.held += nbytes
    self.n_alloc += 1; self.b_alloc += nbytes
    return h
  def free(self, handle: Any, nbytes: int, evict: bool = True) -> None:
    if not getenv("FZ_LRU", 1): return self._release(handle, nbytes)
    hs = self.cache.pop(nbytes, None) or []
    hs.append(handle)
    self.cache[nbytes] = hs
    self.cached += nbytes
    if evict and self.cache_max is not None:
      while self.cached > self.cache_max:
        n = next(iter(self.cache))   # the least recently used size (the freed one is last)
        h = self.cache[n].pop()
        if not self.cache[n]: del self.cache[n]
        self.cached -= n
        self.b_evict += n
        self._release(h, n)
  def _release(self, handle: Any, nbytes: int) -> None:
    self._free(handle, nbytes)
    self.held -= nbytes
    self.n_free += 1; self.b_free += nbytes
  def free_cache(self):
    for nbytes, hs in self.cache.items():
      for h in hs: self._release(h, nbytes)
    self.cache.clear()
    self.cached = 0
  def _alloc(self, nbytes: int) -> Any: raise NotImplementedError
  def _free(self, handle: Any, nbytes: int) -> None: pass

# ---------------------------------------------------------------- buffers

class Buffer:
  """A device allocation of `size` elements of `dtype`. Allocated lazily; freed on GC."""
  def __init__(self, device: str, size: int, dtype: DType):
    self.device, self.size, self.dtype = device, size, dtype
    self._handle: Any = None
  @property
  def nbytes(self) -> int: return self.size * self.dtype.itemsize
  @property
  def allocated(self) -> bool: return self._handle is not None
  @property
  def handle(self) -> Any:
    self.ensure_allocated()
    return self._handle
  def ensure_allocated(self) -> "Buffer":
    if self._handle is None:
      alloc = Device[self.device].allocator
      self._handle = alloc.alloc(max(self.nbytes, 1))
      weakref.finalize(self, alloc.free, self._handle, max(self.nbytes, 1))
    return self
  def copyin(self, src: memoryview) -> "Buffer":
    assert src.nbytes == self.nbytes, f"copyin size mismatch {src.nbytes} != {self.nbytes}"
    Device[self.device].allocator.copyin(self.handle, src)
    return self
  def copyout(self, dst: memoryview) -> memoryview:
    assert dst.nbytes == self.nbytes, f"copyout size mismatch {dst.nbytes} != {self.nbytes}"
    Device[self.device].allocator.copyout(dst, self.handle)
    return dst
  def numpy(self) -> "np.ndarray":
    import numpy as np
    out = np.empty(self.size, dtype=to_np(self.dtype))
    self.copyout(memoryview(out).cast("B"))
    return out
  def __repr__(self): return f"<Buffer {self.device} {self.size} {self.dtype}>"

# ---------------------------------------------------------------- devices

class DeviceBase:
  name: str
  allocator: Allocator
  def run(self, si: "ScheduleItem") -> float | None:
    """Execute one ScheduleItem; return elapsed seconds if timing is enabled."""
    raise NotImplementedError
  def synchronize(self) -> None: pass

class Compiler:
  def compile(self, src: str) -> bytes: raise NotImplementedError
  def cache_key(self) -> str: return ""  # compiler identity/flags; part of the disk-cache key so flag changes never reuse stale binaries
  def compile_cached(self, src: str) -> bytes:
    from fusezap.helpers import CACHE_DIR, stable_hash
    path = CACHE_DIR / type(self).__name__ / stable_hash(self.cache_key() + "\0" + src)
    if path.exists() and not getenv("FZ_NOCACHE"): return path.read_bytes()
    lib = self.compile(src)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lib)
    return lib

class Program:
  def __init__(self, name: str, lib: bytes): self.name, self.lib = name, lib
  def __call__(self, *bufs: Any, global_size: tuple[int, ...], local_size: tuple[int, ...] | None,
               vals: tuple[int, ...] = (), wait: bool = False) -> float | None:
    raise NotImplementedError

class CompiledDevice(DeviceBase):
  """Device = renderer + compiler + runtime + allocator. `run` is: lower -> render -> compile -> launch.
  Implemented by WP4 (shared path in fusezap/codegen/lower.py: `run_compiled`)."""
  renderer: Any          # fusezap.codegen.cstyle.Renderer
  compiler: Compiler
  runtime: type[Program]
  def run(self, si: "ScheduleItem") -> float | None:
    from fusezap.codegen.lower import run_compiled
    return run_compiled(self, si)

class _DeviceRegistry:
  _names = ("METAL", "CUDA", "CPU", "NPY")
  _modules = {"NKISHIM": "neuron", "NKISIM": "neuron"}
  @functools.lru_cache(maxsize=None)
  def __getitem__(self, name: str) -> DeviceBase:
    base = name.split(":")[0].upper()
    mod = importlib.import_module(f"fusezap.device.{self._modules.get(base, base.lower())}")
    return getattr(mod, f"{base}Device")(name)
  def available(self, name: str) -> bool:
    try:
      self[name]
      return True
    except Exception:
      return False
  @functools.cached_property
  def DEFAULT(self) -> str:
    if (d := getenv("FZ_DEVICE", "")): return d.upper()
    # NPY (numpy) is always there: a host with neither Metal, CUDA nor clang (the N4 inf2 DLAMI has only gcc) must not raise
    # StopIteration here (run 1: every NEURON test fixture failed with "generator raised StopIteration")
    return next((n for n in ("METAL", "CUDA", "CPU") if self.available(n)), "NPY")
  @staticmethod
  def canonicalize(name: str | None) -> str:
    return (name or Device.DEFAULT).upper()

Device = _DeviceRegistry()
