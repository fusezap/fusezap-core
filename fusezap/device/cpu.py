"""CPU backend: C via clang, loaded with ctypes (SPEC §8). Implemented by WP4.

CPUDevice(CompiledDevice): allocator = CPUAllocator (LRUAllocator over ctypes/np buffers),
renderer = CRenderer(), compiler = ClangCompiler (src -> .dylib bytes), runtime = CPUProgram.
"""
from __future__ import annotations
import ctypes, functools, os, platform, shutil, subprocess, sys, tempfile, time
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

@functools.lru_cache(maxsize=None)
def _clang_accepts(cc: str, flag: str) -> bool:
  """Whether this compiler knows `flag` (clang rejects an unknown -m flag outright)."""
  if shutil.which(cc) is None and not os.path.exists(cc): return False
  return subprocess.run([cc, flag, "-x", "c", "-fsyntax-only", "-"], input=b"", capture_output=True, timeout=60).returncode == 0

class ClangCompiler(Compiler):
  args = ["-O2", "-shared", "-fPIC", "-ffp-contract=off", "-fwrapv", "-march=native", "-Wno-unused-function"]
  def __init__(self, cc: str | None = None):
    self.cc = cc or os.getenv("FZ_CC") or shutil.which("clang") or "clang"
    # x86: no native float16 instructions. clang 18.1 (Ubuntu 24.04's) crashes in codegen on a float16 min kernel when
    # -march=native turns them on: AVX512-FP16 on Sapphire/Granite Rapids, and on Granite Rapids also AVX10.1-256 and
    # AVX10.1-512, each of which brings them back even under -mno-avx512fp16. Without them float16 maths is the same on
    # every x86 CPU. Compilers that predate AVX10 (clang < 18) reject its flags, and their -march=native cannot enable it.
    self.args = list(ClangCompiler.args)
    if platform.machine().lower() in ("x86_64", "amd64"):
      self.args += ["-mno-avx512fp16", *(f for f in ("-mno-avx10.1-256", "-mno-avx10.1-512") if _clang_accepts(self.cc, f))]
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
