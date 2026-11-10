"""CPU backend: C via clang, loaded with ctypes (SPEC §8). Implemented by WP4.

CPUDevice(CompiledDevice): allocator = CPUAllocator (LRUAllocator over ctypes/np buffers),
renderer = CRenderer(), compiler = ClangCompiler (src -> .dylib bytes), runtime = CPUProgram.
"""
from __future__ import annotations
import ctypes, os, shutil, subprocess, sys, tempfile, time
from typing import Any
from fusezap.device import CompiledDevice, Compiler, LRUAllocator, Program
from fusezap.codegen.cstyle import CRenderer

class CPUAllocator(LRUAllocator):
  """Host memory as ctypes byte arrays; the handle passes straight to a C kernel as a pointer."""
  def _alloc(self, nbytes: int) -> Any: return (ctypes.c_uint8 * nbytes)()
  def copyin(self, handle: Any, src: memoryview) -> None:
    src = src.cast("B") if src.format != "B" or src.ndim != 1 else src
    memoryview(handle).cast("B")[:src.nbytes] = src
  def copyout(self, dst: memoryview, handle: Any) -> None:
    dst = dst.cast("B") if dst.format != "B" or dst.ndim != 1 else dst
    dst[:] = memoryview(handle).cast("B")[:dst.nbytes]

class ClangCompiler(Compiler):
  args = ["-O2", "-shared", "-fPIC", "-ffp-contract=off", "-fwrapv", "-march=native", "-Wno-unused-function"]
  def __init__(self, cc: str | None = None):
    self.cc = cc or os.getenv("FZ_CC") or shutil.which("clang") or "clang"
  def cache_key(self) -> str: return " ".join([self.cc, *self.args])
  def compile(self, src: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="fz_cc_") as d:
      out = os.path.join(d, "kernel.so")
      p = subprocess.run([self.cc, *self.args, "-x", "c", "-", "-o", out], input=src.encode(), capture_output=True)
      if p.returncode != 0: raise RuntimeError(f"clang failed:\n{p.stderr.decode()}\n--- source ---\n{src}")
      with open(out, "rb") as f: return f.read()

class CPUProgram(Program):
  """Loads the shared-object bytes with ctypes. global/local sizes are ignored: SPECIAL dims are loops in the C."""
  def __init__(self, name: str, lib: bytes):
    super().__init__(name, lib)
    fd, self._path = tempfile.mkstemp(prefix=f"fz_{name}_", suffix=".so")
    with os.fdopen(fd, "wb") as f: f.write(lib)
    self._dll = ctypes.CDLL(self._path)
    self.fxn = getattr(self._dll, name)
    self.fxn.restype = None
    try: os.unlink(self._path)  # the mapping stays valid after unlink
    except OSError: pass
  def __call__(self, *bufs: Any, global_size: tuple[int, ...] = (), local_size: tuple[int, ...] | None = None,
               vals: tuple[int, ...] = (), wait: bool = False) -> float | None:
    st = time.perf_counter()
    self.fxn(*[b if isinstance(b, (ctypes.Array, ctypes.c_void_p)) else ctypes.c_void_p(b) for b in bufs], *[ctypes.c_int(int(v)) for v in vals])
    return time.perf_counter() - st if wait else None

class CPUDevice(CompiledDevice):
  def __init__(self, name: str):
    if sys.platform == "win32": raise RuntimeError("CPU backend needs clang on a POSIX host")
    self.name = name
    self.allocator = CPUAllocator()
    self.renderer = CRenderer()
    self.compiler = ClangCompiler()
    if shutil.which(self.compiler.cc) is None and not os.path.exists(self.compiler.cc): raise RuntimeError("clang not found")
    self.runtime = CPUProgram
