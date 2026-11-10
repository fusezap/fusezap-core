"""NEURON backend: AWS Inferentia2 / Trainium1 (docs/NKI_BACKEND.md §4). N1: the device skeleton and its NKISHIM tier.

The four parts, as the design maps them (exceptions 1 and 2: UOps are bypassed and there are no runtime vals, so `run`
is its own path, not `lower.run_compiled`): render the kernel AST with the symbolic dims bound (fusezap.codegen.nki),
compile the source, and launch it on flat buffers.

| Tier | Compiler | Runtime | Status |
|---|---|---|---|
| NKISHIM (Mac) | `NKIShimCompiler`: the source itself | `NKIShimProgram`: runs it through fusezap.device.nkishim | N1 |
| NKISIM (CPU pod) | `NKISimCompiler`: the source (nki traces it at launch) | `NKISimProgram`: `nki.simulate` in a `.venv-nki` worker process | N2 |
| NEURON (inf2), default `FZ_NEURON_RT=nrt` | `NeuronNrtCompiler`: neuronx-cc to a NEFF once per (source, buffer sizes), in the `.venv-nki` worker, disk-cached | `NeuronNeffProgram`: the NEFF loaded once into the `.venv-nrt` runtime worker (nrt_worker.py) and executed on device-resident buffers (`NeuronAllocator`) | N4b, needs /dev/neuron* + libnrt |
| NEURON, `FZ_NEURON_RT=jit` (fallback) | `NKIHwCompiler`: the source | `NeuronHwProgram`: nki.jit in the `.venv-nki` worker, host buffers; run 1: ~1.31 s per call (recompile + reload) | N4 |

Programs are cached by (AST, bound var_vals, renderer limits): NKI has no runtime integer arguments, so every Var value
is part of the compile key (§3). Bucketing the values (powers of two >= 16) is the model's job (the prefill buckets),
so a NEURON kernel is compiled for exactly the values it is given.
Buffers: host numpy byte arrays (`NPYAllocator`) on NKISHIM/NKISIM and on NEURON with FZ_NEURON_RT=jit; on the persistent
NEURON path they are device tensors in the runtime worker (`NeuronAllocator`: untyped byte buffers by id, exception 4), so
an intermediate never crosses to the host: only copy items (copyin/copyout) move data.
NKISIM: the nki wheel is linux_x86_64-only and conflicts with nrtpy (exception 5), so nki never loads in this process.
A worker (fusezap/device/nkisim_worker.py) runs under `.venv-nki/bin/python` (FZ_NKI_PYTHON overrides; made by
scripts/nki_env.sh) and talks pickle-5 frames (arrays out of band) over a pipe. `compile_neff` uses the same worker to compile a
rendered kernel to a NEFF with neuronx-cc (no device needed; `scripts/nki_env.sh --cc`). FZ_NKISIM_SHIM_CHECK=<file>
also runs every kernel through NKISHIM and appends each disagreement with the simulator to <file> as a JSON line;
FZ_NKISIM_NEFF=<file> also compiles each distinct kernel to an inf2 NEFF and logs the outcome there.
Custom kernels (N3, design §5 and exception 3): `custom_kernels` offers "matmul", "attention" and the NEURON-only
"attention_mask" pattern (attention with a bool mask: is_causal, a bool attn_mask, a model's position mask) to the scheduler
hook, so those graphs reach TensorE as one kind="custom" item instead of composed MUL+SUM kernels. `run` renders the item
with fusezap.codegen.nki_kernels (params baked in, cached by kernel name + params) and runs it on the tier's runtime.
Direct entry points: `attention(q, k, v, causal=, coff=, pos=)` (flash attention with GQA; causal by `affine_select`,
the decode mask from an int32 position tensor) and `kmajor(w)` (a weight relaid K-major once at load time, so the matmul
kernel's moving tiles are contiguous rows).
A kernel the renderer cannot lower raises `NKIUnsupported`; nothing falls back to another device.
"""
from __future__ import annotations
import atexit, hashlib, itertools, json, math, os, pickle, shutil, struct, subprocess, sys, tempfile, threading, time
from pathlib import Path
from typing import Any
import numpy as np
from fusezap.codegen.nki import NKIProgram, NKIRenderer, NKIUnsupported
from fusezap.device import Compiler, DeviceBase, LRUAllocator, Program
from fusezap.device.npy import NPYAllocator
from fusezap.dtype import to_np
from fusezap.ir import ScheduleItem

class NKIShimCompiler(Compiler):
  """NKISHIM 'compiles' to the source text: the shim executes Python."""
  def cache_key(self) -> str: return "nkishim"
  def compile(self, src: str) -> bytes: return src.encode()

class NKIShimProgram(Program):
  """A rendered kernel executed by the numpy shim of nki 0.6 on flat host arrays, in place."""
  def __init__(self, prog: NKIProgram, lib: bytes):
    super().__init__(prog.name, lib)
    assert lib.decode() == prog.src
    self.prog = prog
  def __call__(self, *arrays: np.ndarray, wait: bool = False) -> float | None:
    from fusezap.device import nkishim
    et = nkishim.execute(self.prog, list(arrays))
    return et if wait else None

def nki_python() -> Path:
  """The .venv-nki interpreter: FZ_NKI_PYTHON, else <repo>/.venv-nki/bin/python."""
  if p := os.getenv("FZ_NKI_PYTHON"): return Path(p)
  return Path(__file__).resolve().parents[2] / ".venv-nki" / "bin" / "python"

def _send(f, obj) -> None:
  """Pickle 5 with out-of-band buffers: header (meta length, buffer count, buffer lengths), meta, then each array's memory
  written to the pipe as is, so a big array is never copied into an in-band pickle (NKI-N3: with in-band pickles a 4 GB
  pod could not simulate a 0.5 GB weight). nkisim_worker.py has the same pair."""
  bufs: list = []
  meta = pickle.dumps(obj, protocol=5, buffer_callback=bufs.append)
  raws = [b.raw() for b in bufs]
  f.write(struct.pack("<QI", len(meta), len(raws)) + b"".join(struct.pack("<Q", r.nbytes) for r in raws) + meta)
  for r in raws: f.write(r)
  f.flush()

def _recv(f):
  h = f.read(12)
  if len(h) < 12: return None
  n, k = struct.unpack("<QI", h)
  lens = struct.unpack(f"<{k}Q", f.read(8 * k)) if k else ()
  meta, bufs = f.read(n), []
  for m in lens:
    b = bytearray(m)
    mv, got = memoryview(b), 0
    while got < m:
      if not (r := f.readinto(mv[got:])): raise EOFError
      got += r
    bufs.append(b)
  return pickle.loads(meta, buffers=bufs)

def nrt_python() -> Path:
  """The .venv-nrt interpreter (numpy + nrtpy; scripts/nki_env.sh --nrt): FZ_NRT_PYTHON, else <repo>/.venv-nrt/bin/python."""
  if p := os.getenv("FZ_NRT_PYTHON"): return Path(p)
  return Path(__file__).resolve().parents[2] / ".venv-nrt" / "bin" / "python"

class NKIWorker:
  """One `.venv-nki` python running nkisim_worker.py, per fusezap process; requests are serialized by a lock."""
  _inst: "NKIWorker | None" = None
  script, what = "nkisim_worker.py", "NKISIM: the nki worker"
  @staticmethod
  def python() -> Path: return nki_python()
  def __init__(self, python: Path):
    worker = Path(__file__).with_name(self.script)
    env = dict(os.environ)
    if (n := os.getenv("FZ_NEURON_XDIST_CORES")) and (w := os.getenv("PYTEST_XDIST_WORKER", "")).startswith("gw"):
      env["NEURON_RT_VISIBLE_CORES"] = str(int(w[2:]) % int(n))   # N4: one NeuronCore per pytest-xdist worker
    if Path("/opt/aws/neuron/lib").is_dir() and "/opt/aws/neuron/lib" not in env.get("LD_LIBRARY_PATH", ""):
      env["LD_LIBRARY_PATH"] = "/opt/aws/neuron/lib" + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    self.proc = subprocess.Popen([str(python), str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, env=env)
    self.lock = threading.Lock()
    atexit.register(self.close)
  @classmethod
  def get(cls) -> "NKIWorker":
    inst = cls.__dict__.get("_inst")   # one per class: NrtWorker has its own
    if inst is None or inst.proc.poll() is not None: cls._inst = inst = cls(cls.python())
    return inst
  def call(self, req: dict) -> dict:
    with self.lock:
      assert self.proc.stdin is not None and self.proc.stdout is not None
      try:
        _send(self.proc.stdin, req)
        if (rep := _recv(self.proc.stdout)) is None: raise EOFError
      except (BrokenPipeError, EOFError) as e:
        raise RuntimeError(f"{self.what} died (exit {self.proc.poll()}); see its stderr above") from e
    return rep
  def close(self) -> None:
    if self.proc.poll() is None:
      try: self.proc.stdin.close(); self.proc.wait(timeout=5)   # type: ignore[union-attr]
      except Exception: self.proc.kill()

class NKISimError(RuntimeError):
  """The real nki simulator (or compiler) rejected or failed a rendered kernel: a renderer bug, never a fallback."""

def _host_arrays(prog: NKIProgram, arrays: list[Any]) -> tuple[list[Any], list[str]]:
  from fusezap.codegen.nki import nl_dtype
  return list(arrays), [nl_dtype(prog.buf_dtypes[i]) if i in prog.buf_dtypes else "uint8" for i in range(len(arrays))]

def compile_neff(prog: NKIProgram, arrays: list[Any], target: str = "inf2", out_dir: str | None = None) -> tuple[bytes, float]:
  """Compile a rendered kernel to a NEFF for `target` with neuronx-cc in .venv-nki (no Neuron device needed): (bytes, seconds).
  The arrays give the argument shapes and dtypes (nki traces with them); `nki.jit` then tries to run the NEFF, which fails
  without libnrt after the NEFF is written. An (elements, numpy dtype) pair instead of an array has the worker allocate
  zeros of that size itself (for weights too big to send)."""
  rep = _compile_neff_rep(prog, arrays, target, out_dir)
  return rep["neff"], rep["seconds"]

def _compile_neff_rep(prog: NKIProgram, arrays: list[Any], target: str = "inf2", out_dir: str | None = None) -> dict:
  """The worker's full reply: neff bytes, seconds, io (the NEFF's input/output names and aliases), how (compile_only | jit_fallback).
  The worker compiles each request in a fresh unique directory (N4 run 1: a reused one failed with NCC_IDRV077)."""
  xs, dts = _host_arrays(prog, list(arrays))
  rep = (getattr(_TLS, "worker", None) or NKIWorker.get()).call({"op": "neff", "src": prog.src, "name": prog.name, "arrays": xs, "dtypes": dts, "target": target, "dir": out_dir})
  if not rep["ok"]: raise NKISimError(f"NKISIM: neuronx-cc failed on {prog.name}: {rep['err']}")
  return rep

# ---------------------------------------------------------------- N4b: the persistent NEURON path (NEFF cache, runtime worker)

def neff_cache_dir() -> Path:
  """FZ_NEFF_CACHE, else <fusezap cache>/neff. One directory per NEFF key, written once (see cached_neff)."""
  if p := os.getenv("FZ_NEFF_CACHE"): return Path(p)
  from fusezap.helpers import CACHE_DIR
  return Path(CACHE_DIR) / "neff"

def neff_key(prog: NKIProgram, specs: list[tuple[int, Any]], target: str, versions: dict) -> str:
  """Content hash of everything a NEFF depends on: compiler versions, target, kernel name and source, and each argument's
  element count and dtype (nki traces with concrete shapes)."""
  body = json.dumps([sorted((str(k), str(v)) for k, v in versions.items()), target, prog.name, prog.src,
                     [(int(n), np.dtype(d).str) for n, d in specs]])
  return hashlib.sha256(body.encode()).hexdigest()[:32]

NEFF_STATS = {"compiles": 0, "hits": 0, "compile_s": 0.0}
_STATS_LOCK = threading.Lock()
_TLS = threading.local()   # I75 prewarm: a compile thread's own nki worker (precompile), else NKIWorker.get()

def compile_worker() -> Any:
  """A fresh `.venv-nki` worker for one precompile thread (I75 prewarm: NEFF compiles are independent neuronx-cc runs)."""
  return NKIWorker(NKIWorker.python())

def prewarm_jobs() -> int:
  """FZ_NEURON_PREWARM_JOBS: parallel NEFF compiles per rank at prewarm (default min(4, CPUs / 4), at least 1). The
  per-key lock (cached_neff) keeps two ranks from compiling the same NEFF twice."""
  v = os.getenv("FZ_NEURON_PREWARM_JOBS", "")
  return max(1, int(v)) if v.strip() else max(1, min(4, (os.cpu_count() or 4) // 4))

DMI_PRODUCT = "/sys/devices/virtual/dmi/id/product_name"   # the EC2 instance type on Nitro instances (inf2.xlarge, trn1.2xlarge)

def neuron_target(product_file: str = DMI_PRODUCT) -> str:
  """The neuronx-cc target the persistent path compiles for: FZ_NEURON_TARGET, else from this host's instance type (trn1* /
  trn1n* -> trn1, anything else -> inf2). Both are NeuronCore-v2; the target is part of the NEFF key, so the two never share
  a cache entry."""
  if t := os.getenv("FZ_NEURON_TARGET"): return t
  try:
    with open(product_file) as f: name = f.read().strip().lower()
  except OSError: name = ""
  return "trn1" if name.startswith("trn1") else "inf2"

def cached_neff(prog: NKIProgram, specs: list[tuple[int, Any]], target: str = "inf2", versions: dict | None = None) -> tuple[Path, dict]:
  """(path of kernel.neff, meta) for a kernel at these argument sizes, compiled at most once per key across processes.
  A compile happens in the worker's own fresh directory; the result is written to a private temp dir next to the cache entry
  and renamed into place, so a key directory, once it exists, is complete and never written again (a concurrent writer that
  loses the rename discards its copy)."""
  versions = versions if versions is not None else _nki_versions()
  key = neff_key(prog, specs, target, versions)
  d = neff_cache_dir() / key
  if (d / "kernel.neff").exists():
    with _STATS_LOCK: NEFF_STATS["hits"] += 1
    try: meta = json.loads((d / "meta.json").read_text())
    except (OSError, ValueError): meta = {}
    return d / "kernel.neff", meta
  # N8: one compile per key across processes (tensor-parallel ranks compile the same kernels at the same moment): the
  # others wait on the key's lock file and read the result, so the ranks leave a cold first step together
  import fcntl
  d.parent.mkdir(parents=True, exist_ok=True)
  with open(lock := d.parent / f".{key}.lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    try:
      if (d / "kernel.neff").exists(): return cached_neff(prog, specs, target, versions)
      return _compile_into(prog, specs, target, versions, key, d)
    finally:
      try: lock.unlink()   # waiters hold the file open and find the entry; later callers never reach the lock
      except OSError: pass
      fcntl.flock(lk, fcntl.LOCK_UN)

def _compile_into(prog: NKIProgram, specs: list, target: str, versions: dict, key: str, d: Path) -> tuple[Path, dict]:
  rep = _compile_neff_rep(prog, list(specs), target)
  with _STATS_LOCK: NEFF_STATS["compiles"] += 1; NEFF_STATS["compile_s"] += rep["seconds"]
  meta = {"key": key, "kernel": prog.name, "target": target, "seconds": rep["seconds"], "io": rep.get("io") or {}, "how": rep.get("how"),
          "versions": versions}
  d.parent.mkdir(parents=True, exist_ok=True)
  tmp = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=d.parent))
  (tmp / "kernel.neff").write_bytes(rep["neff"])
  (tmp / "meta.json").write_text(json.dumps(meta))
  try: os.rename(tmp, d)
  except OSError: shutil.rmtree(tmp, ignore_errors=True)   # another process wrote this key first: keep its copy
  return d / "kernel.neff", meta

_VERSIONS: dict = {}
def _nki_versions() -> dict:
  if not _VERSIONS:
    rep = NKIWorker.get().call({"op": "version"})
    _VERSIONS.update(nki=rep.get("nki"), neuronxcc=rep.get("neuronxcc"))
  return dict(_VERSIONS)

_ALIAS_SUFFIX = ".must_alias_input"
_BUF_IDS = itertools.count(1)   # device buffer ids, never reused within a process

class NeuronBindError(RuntimeError):
  """A NEFF declares a tensor that no kernel argument maps to."""

def neff_binding(inputs: Any, outputs: Any, alias: dict[str, str], n_args: int) -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
  """Map a NEFF's declared I/O tensors to kernel argument indices: (inputs {name: arg}, outputs {name: arg}, alias {output: input}).
  Our kernels take every buffer as an argument named data<i> and return the outputs, so nki makes each output that is also an
  argument a must-alias pair (run 1: inputs data0.must_alias_input, data1, data2; output data0). Every declared input is bound
  (run 1 left out data0.must_alias_input: NRT_INVALID), and an aliased output is executed on a slice of its input's tensor
  (the worker builds it), so the kernel writes the argument buffer itself. An output with no alias binds to its own argument."""
  def arg(name: str) -> int:
    base = name[:-len(_ALIAS_SUFFIX)] if name.endswith(_ALIAS_SUFFIX) else name
    if not (base.startswith("data") and base[4:].isdigit()) or int(base[4:]) >= n_args:
      raise NeuronBindError(f"NEFF tensor {name!r} is not one of the kernel arguments data0..data{n_args - 1}")
    return int(base[4:])
  ins = {str(n): arg(str(n)) for n in inputs}
  al = {str(o): str(i) for o, i in (alias or {}).items()}
  for n in ins:   # nrtpy fills alias_info from the NEFF metadata; the suffix alone also implies the pair
    if n.endswith(_ALIAS_SUFFIX) and (o := n[:-len(_ALIAS_SUFFIX)]) in [str(x) for x in outputs]: al.setdefault(o, n)
  outs, alias_out = {}, {}
  for o in (str(x) for x in outputs):
    if o in al:
      if al[o] not in ins: raise NeuronBindError(f"NEFF output {o!r} must alias input {al[o]!r}, which the NEFF does not declare")
      alias_out[o] = al[o]
    else: outs[o] = arg(o)
  return ins, outs, alias_out

class NrtWorker(NKIWorker):
  """One `.venv-nrt` python running nrt_worker.py per fusezap process: the only process that opens the NeuronCore on the
  persistent path. Device buffers and loaded NEFFs live there for the process's lifetime."""
  _inst: "NrtWorker | None" = None
  script, what = "nrt_worker.py", "NEURON: the runtime worker (nrt_worker.py)"
  @staticmethod
  def python() -> Path: return nrt_python()

class NrtBuf:
  """A device buffer in the runtime worker: an id and a byte size (untyped, exception 4: dtype is fusezap's metadata)."""
  __slots__ = ("id", "nbytes")
  def __init__(self, id: int, nbytes: int): self.id, self.nbytes = id, nbytes
  def __repr__(self): return f"<NrtBuf {self.id} {self.nbytes}B>"

def host_fast() -> bool:
  """FZ_NEURON_HOSTFAST (N7, default on): the persistent path's per-step host work cut down: small writes are queued with the
  launches, a read takes the queued requests with it (one runtime-worker round trip per decode step instead of six),
  replays launch through NeffReplay, LLMEngine uploads its step inputs straight into device buffers. 0: N6b's host path."""
  return os.getenv("FZ_NEURON_HOSTFAST", "1") not in ("", "0")

class NeuronOOM(MemoryError):
  """N8 (I71): a device allocation that would exceed the HBM budget (FZ_NEURON_HBM_GB) after the allocator's free list went
  back to the runtime, or that the runtime refused (NRT_RESOURCE) before any launch of its batch ran. A MemoryError, so the
  scorer's release-and-retry and the serving layer's 503 apply; nothing of the failed batch executed."""

def hbm_budget() -> int:
  """FZ_NEURON_HBM_GB (GiB, default 15.25): the device tensor bytes one runtime worker may hold. I69 run 1 (inf2.8xlarge):
  core 0 failed at 15.85 GiB of tensors next to 0.11 GiB of NEFF code and 0.02 GiB of DMA rings (the probe's 512 MiB chunks
  stop at 15.00); the default leaves ~0.6 GiB for NEFFs, rings and fragmentation. 0: no budget (the runtime's limit only)."""
  v = os.getenv("FZ_NEURON_HBM_GB", "")
  return int(float(v or 15.25) * (1 << 30))

def neff_reserve() -> int:
  """FZ_NEURON_NEFF_RESERVE_MB (default 256): NEFF bytes (code, DMA rings) the HBM budget's margin covers. I75 prewarm loads
  every prefill NEFF up front (a FUSE=all 7B TP=2 prefill NEFF is ~9 MB, four pool prefixes); loaded NEFF bytes beyond this
  reserve come off the tensor budget (NeuronAllocator.limit), so a large NEFF set cannot push the device past the runtime's limit."""
  return int(float(os.getenv("FZ_NEURON_NEFF_RESERVE_MB", "") or 256) * (1 << 20))

def alloc_cache_cap() -> int:
  """FZ_NEURON_ALLOC_CACHE_MB (default 256): the most bytes NeuronAllocator's free list keeps for reuse (I69 run 1: the
  list was unbounded, so every intermediate size a 7B scoring pass had freed stayed allocated on the device)."""
  return int(float(os.getenv("FZ_NEURON_ALLOC_CACHE_MB", "") or 256) * (1 << 20))

def neuron_plan() -> bool:
  """FZ_NEURON_PLAN (default 1; 0 = N7's behaviour): the serving forwards and the scorer's jitted steps on NEURON plan their
  intermediates (fz.jit `plan`: buffers whose live ranges do not overlap share one device allocation, as on CUDA). Unplanned,
  every captured forward kept each intermediate of every layer allocated for good: a 7B prefill of 128 rows ~1.3 GB, a
  512-row scoring bucket ~5 GB (I69 run 1's NRT_RESOURCE). Planning ranges over the launched (fused) items, so buffers inside
  one fused kernel never share; a NEFF tensor bound to a larger arena gets a leading slice (nrt_worker._ref)."""
  return os.getenv("FZ_NEURON_PLAN", "1") not in ("", "0")

def hbm_log() -> bool:
  """FZ_NEURON_HBM_LOG=1: one stderr line per NEFF load and per allocation past each GiB (the HBM accounting, N8)."""
  return os.getenv("FZ_NEURON_HBM_LOG", "0") not in ("", "0")

class NeuronAllocator(LRUAllocator):
  """Device-resident buffers for the persistent NEURON path. `_alloc` only names a buffer; the worker allocates it (zeroed)
  with the next request that carries it, and frees ride along with the next request too, so neither costs a round trip.
  copyin/copyout are the only host<->device transfers (copy items at the graph boundary). N7 (host_fast): a write of at most
  FZ_NEURON_QWRITE_MAX bytes (default 1 MiB, a copy of the data) is queued like a launch, and a read is sent together with
  the queued requests as one batch, so a decode step's inputs, launches and token read are one round trip."""
  def __init__(self, worker: Any = None):
    super().__init__()
    self._worker = worker
    self._ids = _BUF_IDS   # process-wide: two allocators (two NEURONDevice instances) share the one runtime worker
    self.new: dict[int, int] = {}
    self.dead: list[int] = []
    self.calls = {"write": 0, "read": 0, "exec": 0, "load": 0, "batches": 0, "ar": 0}
    self.cc_keys: set = set()   # N8: loaded NEFFs with collectives; cc_count: their launches + "ar" requests (neuron_tp's desync check)
    self.cc_count = 0
    self.batch = os.getenv("FZ_NEURON_BATCH", "1") not in ("", "0")
    self.batch_max = int(os.getenv("FZ_NEURON_BATCH_MAX", "") or 4096)
    self.pending: list[dict] = []
    self.fast, self.qwrite_max = host_fast(), int(os.getenv("FZ_NEURON_QWRITE_MAX", "") or 1 << 20)
    self.trips = 0               # round trips to the runtime worker
    self.wait_s = 0.0            # wall seconds inside them (FZ_SERVE_PROFILE reads it per step)
    # N6 profiling (FZ_NEURON_PROFILE=1, or set .profile): per kernel name [launches, execute seconds in the worker], the
    # worker's batch wall time, and the last binding of each loaded NEFF (for a device-time `bench`, neuron_profile())
    self.profile = os.getenv("FZ_NEURON_PROFILE", "") not in ("", "0")
    self.prof: dict[str, list] = {}
    self.prof_wall = [0, 0.0]   # batches, worker seconds
    self.prof_ipc = [0, 0.0, 0.0]   # N7: round trips, host seconds inside them, the worker's seconds handling them
    self.bindings: dict[str, dict] = {}
    # N8 (I71) HBM: a bounded free list, a budget checked here (allocations ride lazily on the next request, so the runtime's
    # own refusal would only surface mid-batch), and the accounting neuron_hbm_report reads
    self.cache_max, self.budget, self.log = alloc_cache_cap(), hbm_budget(), hbm_log()
    self.peak, self.neffs, self.neff_bytes, self.ooms, self._logged_gib = 0, 0, 0, 0, 0
    self.neff_reserve = neff_reserve()
    self.load_log: list[tuple] = []   # I75 prewarm: (kernel, key, seconds, NEFF bytes) per NEFF load (the last 4096)
  def limit(self) -> int:
    """The tensor bytes this allocator may hold: the budget less the loaded NEFF bytes past their reserve (I75)."""
    return self.budget - max(0, self.neff_bytes - self.neff_reserve)
  @property
  def worker(self) -> Any: return self._worker or NrtWorker.get()
  def _alloc(self, nbytes: int) -> NrtBuf:
    if self.budget and self.held + nbytes > self.limit():   # LRUAllocator.alloc hands the free list back and retries once
      self.ooms += 1
      raise NeuronOOM(f"NEURON: {nbytes / 2**20:.1f} MiB more would exceed the HBM budget: {self.held / 2**30:.3f} GiB held "
                      f"({self.cached / 2**30:.3f} cached) of {self.limit() / 2**30:.2f} (FZ_NEURON_HBM_GB, less "
                      f"{max(0, self.neff_bytes - self.neff_reserve) / 2**20:.0f} MiB of NEFFs past FZ_NEURON_NEFF_RESERVE_MB)")
    b = NrtBuf(next(self._ids), nbytes)
    self.new[b.id] = nbytes
    self.peak = max(self.peak, self.held + nbytes)
    if self.log and (g := int((self.held + nbytes) >> 30)) > self._logged_gib:
      self._logged_gib = g
      print(f"NEURON HBM: tensors held {(self.held + nbytes) / 2**30:.3f} GiB (cached {self.cached / 2**30:.3f}) of budget "
            f"{self.budget / 2**30:.2f}; NEFFs {self.neffs} ({self.neff_bytes / 2**20:.1f} MiB)", file=sys.stderr)
    return b
  def _free(self, handle: NrtBuf, nbytes: int) -> None:
    if self.new.pop(handle.id, None) is None: self.dead.append(handle.id)
  def call(self, req: dict) -> dict:
    """One request to the runtime worker. N5: launches are queued (FZ_NEURON_BATCH, default on) and sent as one "batch"
    request at the next non-launch request (a copy, a load) or flush(), so a jit replay of a few hundred kernels costs one
    pipe round trip instead of one per kernel; the worker runs them in order. A queued launch's error surfaces at that
    flush, naming the launch."""
    req["new"], req["free"] = self.new, self.dead
    self.new, self.dead = {}, []
    if (op := req.get("op")) in self.calls: self.calls[op] += 1
    if op == "ar" or (op == "exec" and req.get("key") in self.cc_keys): self.cc_count += 1
    if op == "exec" and self.profile: self.bindings[req["key"]] = {k: req[k] for k in ("key", "in", "out", "alias", "name") if k in req}
    if self.batch and (op in ("exec", "ar") or (op == "write" and self._queues(req["data"].nbytes))):   # N8: "ar", the shm all-reduce
      self.pending.append(req)
      if len(self.pending) >= self.batch_max: self.flush()
      return {"ok": True, "seconds": [0.0] * int(req.get("iters", 1)), "queued": True}
    if op == "read" and self.fast and self.pending:   # N7: the queued requests and the read, one round trip
      reqs, self.pending = self.pending + [req], []
      self.calls["batches"] += 1
      rep = self._send({"op": "batch", "reqs": reqs, "new": {}, "free": []})
      if self.profile: self._account(reqs, rep)
      return rep
    self.flush()
    rep = self._send(req)
    if op == "exec" and self.profile: self._account([req], rep)
    return rep
  def _account(self, reqs: list[dict], rep: dict) -> None:
    secs = list(rep.get("seconds") or [])
    for r in reqs:
      if r.get("op") != "exec": continue   # queued writes and a trailing read report no execute times
      n = int(r.get("iters", 1))
      ent = self.prof.setdefault(r.get("name", r.get("key", "?")), [0, 0.0])
      ent[0] += n; ent[1] += float(sum(secs[:n])); secs = secs[n:]
    self.prof_wall[0] += 1; self.prof_wall[1] += float(rep.get("wall", 0.0))
  def _queues(self, nbytes: int) -> bool: return self.fast and self.batch and nbytes <= self.qwrite_max
  def _send(self, req: dict) -> dict:
    self.trips += 1
    st = time.perf_counter()
    rep = self.worker.call(req)
    dt = time.perf_counter() - st
    self.wait_s += dt
    if self.profile: self.prof_ipc[0] += 1; self.prof_ipc[1] += dt; self.prof_ipc[2] += float(rep.get("t", 0.0))
    if not rep["ok"]:
      if rep.get("oom"):   # N8: the worker refused the request's allocations before running any of it
        self.ooms += 1
        raise NeuronOOM(f"NEURON: the runtime refused device memory ({rep['err']}); host accounting {self.held / 2**30:.3f} GiB held, "
                        f"{self.cached / 2**30:.3f} cached"
                        + ("" if " ran)" in str(rep["err"]) else "; nothing of the request ran"))
      raise NKISimError(f"NEURON: runtime worker {req.get('op')} failed: {rep['err']}\n{rep.get('tb', '')}")
    return rep
  def flush(self) -> None:
    if not self.pending: return
    reqs, self.pending = self.pending, []
    self.calls["batches"] += 1
    rep = self._send({"op": "batch", "reqs": reqs, "new": {}, "free": []})
    if self.profile: self._account(reqs, rep)
  def copyin(self, handle: NrtBuf, src: memoryview) -> None:
    m = src.cast("B") if src.format != "B" or src.ndim != 1 else src
    a = np.frombuffer(m, np.uint8)
    self.call({"op": "write", "id": handle.id, "data": a.copy() if self._queues(a.nbytes) else a})   # queued: the caller may reuse src
  def copyout(self, dst: memoryview, handle: NrtBuf) -> None:
    rep = self.call({"op": "read", "id": handle.id, "n": dst.nbytes})
    np.frombuffer(dst.cast("B") if dst.format != "B" or dst.ndim != 1 else dst, np.uint8)[:] = np.frombuffer(rep["data"], np.uint8)[:dst.nbytes]

def neuron_profile(alloc: NeuronAllocator, iters: int = 20, bench: bool = True) -> dict:
  """N6: the per-kernel launch breakdown collected while alloc.profile was on: per kernel name the launches, the mean
  execute time in the runtime worker (nrt execute: device time + the runtime's own per-call cost) and, with bench, the
  device time of one launch by nrtpy's benchmark (mode="device", on the binding last seen; it reruns the kernel, so run
  it on a throwaway state), plus the worker's batch wall time. ms throughout."""
  alloc.flush()
  out: dict[str, Any] = {"kernels": {}, "batches": alloc.prof_wall[0], "worker_batch_ms": round(1e3 * alloc.prof_wall[1], 3)}
  dev_ms: dict[str, float] = {}
  if bench:
    for key, b in alloc.bindings.items():
      try: st = alloc.call({"op": "bench", **{k: v for k, v in b.items() if k != "name"}, "iters": iters, "mode": "device"})["stats"]
      except Exception as e: st = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
      if "error" in st: out.setdefault("bench_errors", {})[b.get("name", key)] = st["error"]
      else: dev_ms[b.get("name", key)] = st.get("median_ms", st.get("mean_ms"))
  for nm, (n, s) in sorted(alloc.prof.items(), key=lambda kv: -kv[1][1]):
    out["kernels"][nm] = {"launches": n, "exec_ms_mean": round(1e3 * s / max(n, 1), 4), "exec_ms_total": round(1e3 * s, 3),
                          "device_ms": dev_ms.get(nm)}
  out["exec_ms_total"] = round(sum(v["exec_ms_total"] for v in out["kernels"].values()), 3)
  return out

class NeuronNrtCompiler(Compiler):
  """Persistent NEURON path: the source text here; the NEFF is compiled per argument sizes by `cached_neff` at first launch."""
  def __init__(self, nki: str, nrtpy: str): self.nki, self.nrtpy = nki, nrtpy
  def cache_key(self) -> str: return f"neuron-nrt-{self.nki}-{self.nrtpy}"
  def compile(self, src: str) -> bytes: return src.encode()

class NeuronNeffProgram(Program):
  """A rendered kernel on the persistent path: compiled to a NEFF once per argument signature (cached on disk by content),
  loaded once into the runtime worker, then each launch is one `exec` request binding device buffer ids to the NEFF's
  tensors: no compile, no load and no data in the request (N4 run 1's nki.jit path paid ~1.31 s per call for both)."""
  def __init__(self, prog: NKIProgram, lib: bytes):
    super().__init__(prog.name, lib)
    self.prog = prog
    self.loaded: dict[tuple, tuple[str, dict[str, int], dict[str, int], dict[str, str]]] = {}
  def prepare(self, alloc: NeuronAllocator, specs: tuple) -> tuple[str, dict[str, int], dict[str, int], dict[str, str]]:
    if (ent := self.loaded.get(specs)) is None:
      path, meta = cached_neff(self.prog, list(specs), neuron_target())
      key = meta.get("key") or path.parent.name
      cc = {"cc": {"rank": int(os.getenv("FZ_NEURON_CC_RANK", "0")), "world": int(os.getenv("FZ_NEURON_CC_WORLD", "1"))}} \
           if "nki.collectives" in self.prog.src else {}   # N8: a collective NEFF loads for this tensor-parallel rank
      t0 = time.perf_counter()
      info = alloc.call({"op": "load", "key": key, "path": str(path), **cc})
      if (ll := getattr(alloc, "load_log", None)) is not None:
        if len(ll) >= 4096: del ll[:2048]
        ll.append((self.prog.name, key, time.perf_counter() - t0, path.stat().st_size if path.exists() else 0))
      if cc: alloc.cc_keys.add(key)
      if hasattr(alloc, "neffs"):   # N8: per-NEFF accounting (the runtime keeps each NEFF's code and DMA rings in HBM too)
        nb = path.stat().st_size if path.exists() else 0
        alloc.neffs += 1; alloc.neff_bytes += nb
        if alloc.log: print(f"NEURON HBM: load {self.prog.name} NEFF {nb / 1024:.0f} KiB ({alloc.neffs} NEFFs, {alloc.neff_bytes / 2**20:.1f} MiB); "
                            f"tensors held {alloc.held / 2**30:.3f} GiB (cached {alloc.cached / 2**30:.3f}) of {alloc.budget / 2**30:.2f}", file=sys.stderr)
      ins, outs, al = neff_binding(info["inputs"], info["outputs"], info.get("alias", {}), self.prog.n_bufs)
      self.loaded[specs] = ent = (key, ins, outs, al)
    return ent
  def launch(self, alloc: NeuronAllocator, handles: list[NrtBuf], specs: tuple, iters: int = 1) -> list[float]:
    key, ins, outs, al = self.prepare(alloc, specs)
    rep = alloc.call({"op": "exec", "key": key, "in": {n: handles[i].id for n, i in ins.items()},
                      "out": {n: handles[i].id for n, i in outs.items()}, "alias": al, "iters": iters, "name": self.prog.name})
    return rep["seconds"]
  def __call__(self, *args: Any, wait: bool = False, **kw: Any) -> float | None:
    raise RuntimeError("NeuronNeffProgram runs on device buffers: NEURONDevice.run_program")

class NeffReplay:
  """N7 (host_fast): one captured item's launch at fz.jit replay without NEURONDevice.run's per-launch work (the program
  looked up by its spec, every argument's size and dtype, the binding dict over ~60 arguments of a fused layer): the exec
  request is built once, after a first launch through run(), and re-bound only for the arguments whose buffer changed since
  the last replay (the step's inputs, fresh outputs). Anything else (a size change, a size-0 argument, a fused kernel
  that fell back to its members) goes through run() as before."""
  __slots__ = ("dev", "si", "bufs", "req", "args", "n")
  def __init__(self, dev: "NEURONDevice", si: ScheduleItem):
    self.dev, self.si, self.bufs, self.req, self.args, self.n = dev, ScheduleItem(si.ast, (), si.var_vals, si.kind, si.n_outputs), None, None, None, 0
  def __call__(self, bufs: list) -> None:
    req, last = self.req, self.bufs
    if req is not None and len(bufs) == len(last):
      ins = outs = None
      for i in range(self.n):
        b, o = bufs[i], last[i]
        if b is o: continue
        if b.size != o.size or b.dtype != o.dtype or not b.size: req = None; break
        h = b.ensure_allocated().handle.id
        for out, nm in self.args[i]:
          if out: outs = outs if outs is not None else dict(req["out"]); outs[nm] = h
          else: ins = ins if ins is not None else dict(req["in"]); ins[nm] = h
      if req is not None:
        if ins is not None or outs is not None:
          self.req = req = {**req, "in": ins if ins is not None else req["in"], "out": outs if outs is not None else req["out"]}
        self.bufs = bufs
        self.dev.allocator.call(dict(req))   # call() adds new / free to the dict it is given
        return
    si = self.si
    self.dev.run(ScheduleItem(si.ast, tuple(bufs), si.var_vals, si.kind, si.n_outputs))
    self._build(bufs)
  def _build(self, bufs: list) -> None:
    self.req = None
    if (prg := self.dev._neff_program(self.si)) is None: return
    n = prg.prog.n_bufs
    if len(bufs) < n or any(not b.size for b in bufs[:n]): return   # scratch arguments: run() every time
    if (ent := prg.loaded.get(tuple((b.size, to_np(b.dtype).str) for b in bufs[:n]))) is None: return
    key, ins, outs, al = ent
    self.args = [[] for _ in range(n)]
    for nm, i in ins.items(): self.args[i].append((False, nm))
    for nm, i in outs.items(): self.args[i].append((True, nm))
    self.n, self.bufs = n, bufs
    self.req = {"op": "exec", "key": key, "in": {nm: bufs[i].handle.id for nm, i in ins.items()},
                "out": {nm: bufs[i].handle.id for nm, i in outs.items()}, "alias": al, "iters": 1, "name": prg.prog.name}

class NeuronCall:
  """kind="custom" payload of a NEURON kernel that no scheduler pattern builds (N5 serving, fusezap/serve/neuron_kv.py): a
  launch described by its spec (nki_kernels.AttnSpec: attention; NormSpec: RMSNorm + residual add), rendered by
  `spec.render()` and run by NEURONDevice.run."""
  __slots__ = ("spec",)
  def __init__(self, spec: Any): self.spec = spec
  @property
  def kernel(self) -> "NeuronCall": return self   # CustomCall-like: si.ast[0].kernel.name
  @property
  def name(self) -> str: return self.spec.name
  def pretty(self, indent: int = 0) -> str: return f"{'  ' * indent}NEURON {self.spec.name}"

class FusedCall(NeuronCall):
  """N6: a FusedSpec launch plus its members (each original item with its buffers as indices into the fused item's), so
  NEURONDevice.run can launch them one by one if the fused kernel does not compile (a logged fallback, never wrong).
  subs (I75): the same members as smaller fused kernels of FUSE_FALLBACK layers each (same format as members: an item
  without buffers, its buffers as indices into this item's), which the fallback launches instead of the members when set
  (a TP prefill fused over every layer whose one NEFF does not compile runs as groups of 4 layers, not 300+ launches).
  arg_specs: (elements, numpy dtype str) per argument at capture, for precompile (the NEFF key's sizes)."""
  __slots__ = ("members", "subs", "arg_specs")
  def __init__(self, spec: Any, members: tuple, subs: tuple = (), arg_specs: tuple = ()):
    self.spec, self.members, self.subs, self.arg_specs = spec, members, subs, arg_specs

class NKISimCompiler(Compiler):
  """NKISIM compiles nothing ahead of time: `nki.simulate` traces the source at launch. The key holds the nki version."""
  def __init__(self, version: str): self.version = version
  def cache_key(self) -> str: return f"nkisim-{self.version}"
  def compile(self, src: str) -> bytes: return src.encode()

class NKISimProgram(Program):
  """A rendered kernel run by the real nki 0.6 CPU simulator in the worker; outputs are copied back in place."""
  def __init__(self, prog: NKIProgram, lib: bytes):
    super().__init__(prog.name, lib)
    self.prog = prog
  def __call__(self, *arrays: np.ndarray, wait: bool = False) -> float | None:
    xs, dts = _host_arrays(self.prog, list(arrays))
    check = os.getenv("FZ_NKISIM_SHIM_CHECK")
    before = [a.copy() for a in xs] if check or os.getenv("FZ_NKISIM_NEFF") else None
    rep = NKIWorker.get().call({"op": "sim", "src": self.prog.src, "name": self.prog.name, "arrays": xs, "dtypes": dts,
                                "outs": list(self.prog.outs)})
    if not rep["ok"]: raise NKISimError(f"NKISIM: nki.simulate failed on {self.prog.name}: {rep['err']}\n{rep.get('tb', '')}")
    if check and before is not None: _shim_check(self.prog, before, rep["outs"], check)
    if (neff_log := os.getenv("FZ_NKISIM_NEFF")) and before is not None: _neff_check(self.prog, before, neff_log)
    for i, out in rep["outs"].items(): arrays[i][...] = out.reshape(arrays[i].shape)
    return rep["seconds"] if wait else None

def _shim_check(prog: NKIProgram, inputs: list[np.ndarray], sim: dict[int, np.ndarray], path: str) -> None:
  """Run the same kernel through NKISHIM and log every output that differs from the simulator (bitwise; NaNs equal)."""
  from fusezap.device import nkishim
  xs = [a.copy() for a in inputs]
  try: nkishim.execute(prog, xs)
  except Exception as e:
    rec: dict[str, Any] = {"kernel": prog.name, "shim_error": f"{type(e).__name__}: {str(e)[:500]}"}
  else:
    rec = {}
    for i, s in sim.items():
      a, b = xs[i].reshape(-1), np.asarray(s).reshape(-1)
      same = (a.view(np.uint8) == b.view(np.uint8)).reshape(a.size, -1).all(axis=1) if a.size else np.ones(0, bool)
      if a.dtype.kind == "f": same |= np.isnan(a) & np.isnan(b)
      if not same.all():
        k = int(np.flatnonzero(~same)[0])
        diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))[~same])) if a.dtype.kind in "fiu" else None
        rec.setdefault("outs", []).append({"buf": i, "dtype": str(a.dtype), "n_diff": int((~same).sum()), "n": int(a.size),
                                           "max_abs": diff, "first": k, "shim": a[k].item(), "sim": b[k].item()})
    if not rec: return
    rec["kernel"] = prog.name
  rec["src_sha"] = hashlib.sha256(prog.src.encode()).hexdigest()[:12]
  rec["src"] = prog.src
  fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)   # one write per record: safe across xdist workers
  try: os.write(fd, (json.dumps(rec) + "\n").encode())
  finally: os.close(fd)

_NEFF_SEEN: set[str] = set()

def _neff_check(prog: NKIProgram, inputs: list[np.ndarray], path: str) -> None:
  """FZ_NKISIM_NEFF=<file>: also compile each distinct kernel source to an inf2 NEFF and log the outcome as a JSON line."""
  sha = hashlib.sha256(prog.src.encode()).hexdigest()[:12]
  if sha in _NEFF_SEEN: return
  _NEFF_SEEN.add(sha)
  rec: dict[str, Any] = {"kernel": prog.name, "src_sha": sha}
  try:
    neff, secs = compile_neff(prog, inputs, os.getenv("FZ_NKISIM_NEFF_TARGET", "inf2"))
    rec.update(ok=True, seconds=round(secs, 3), bytes=len(neff))
  except NKISimError as e: rec.update(ok=False, err=str(e)[-600:], src=prog.src)
  fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
  try: os.write(fd, (json.dumps(rec) + "\n").encode())
  finally: os.close(fd)

class NKIHwCompiler(Compiler):
  """NEURON (N4): nki.jit compiles the source with neuronx-cc at the first launch on the device; the key holds the nki version."""
  def __init__(self, version: str): self.version = version
  def cache_key(self) -> str: return f"neuron-{self.version}"
  def compile(self, src: str) -> bytes: return src.encode()

class NeuronHwProgram(Program):
  """A rendered kernel run on the NeuronCore by nki.jit in the `.venv-nki` worker (libnrt from the Neuron DLAMI). Host numpy
  buffers go in and the outputs are copied back, synchronously. [unverified on hardware] (N4 measures it): the per-launch cost
  of this path, and whether nki 0.6 returns outputs or writes them in place (the worker handles both; `out_mode` records it)."""
  out_modes: dict[str, int] = {}
  def __init__(self, prog: NKIProgram, lib: bytes):
    super().__init__(prog.name, lib)
    self.prog = prog
  def __call__(self, *arrays: np.ndarray, wait: bool = False) -> float | None:
    xs, dts = _host_arrays(self.prog, list(arrays))
    rep = NKIWorker.get().call({"op": "hw", "src": self.prog.src, "name": self.prog.name, "arrays": xs, "dtypes": dts,
                                "outs": list(self.prog.outs)})
    if not rep["ok"]: raise NKISimError(f"NEURON: the kernel failed on the device: {self.prog.name}: {rep['err']}\n{rep.get('tb', '')}")
    NeuronHwProgram.out_modes[rep["out_mode"]] = NeuronHwProgram.out_modes.get(rep["out_mode"], 0) + 1
    for i, out in rep["outs"].items(): arrays[i][...] = out.reshape(arrays[i].shape)
    return rep["seconds"] if wait else None

TP_WORLD = [1]   # this process's tensor-parallel world (fusezap.serve.neuron_tp.install sets it, NeuronComm.close resets it)
FUSE_ALL = 1 << 12   # "every layer": a group boundary only at layer 0
FUSE_FALLBACK = 4    # I75: layers per kernel when a fused group of more layers (a fence's `every`) fails to compile

def fuse_layers() -> int:
  """FZ_NEURON_FUSE: decoder layers per fused kernel (N6; default 24 since N8: trn1 step 4's fuse24h, the whole decode
  forward of Qwen2.5-0.5B in one launch with FZ_NEURON_HEAD_FUSE; 1: one per layer; 0 = no fusion, N5's one launch per item).
  Under tensor parallelism (I72 follow-up) the default is every layer: with the fused TP layer (collectives inside) a decode
  step is one NEFF per rank (inf2 7B TP=2: FUSE=28 + head 38.8 ms/token in process vs 41-ish at FUSE=1 / 4, token-identical);
  the halves path's all-reduces split the groups anyway."""
  v = os.getenv("FZ_NEURON_FUSE")
  if v is None: return FUSE_ALL if TP_WORLD[0] > 1 else 24
  return int(v or 0)

def fuse_max() -> int:
  """FZ_NEURON_FUSE_MAX: at most this many items per fused kernel (default 44 since N8, trn1 step 4's fuse24h; 0 = no cap):
  bounds a kernel built from the members (v1 layers, 10 items each; layers too wide for LayerSpec) under a large FZ_NEURON_FUSE.
  Under tensor parallelism the default is no cap (the inf2 run's setting): the members' groups end at every all-reduce launch."""
  v = os.getenv("FZ_NEURON_FUSE_MAX")
  if v is None: return 0 if TP_WORLD[0] > 1 else 44
  return int(v or 0)

def head_fuse() -> bool:
  """FZ_NEURON_HEAD_FUSE (N7 step 4; default on since N8, 0 = off): the head (final norm, lm_head, argmax) joins the last layers' fused kernel
  instead of being a NEFF of its own (one launch fewer per forward: trn1 step 3, ~0.17 ms of runtime per launch). The same
  members in the same order, so the same bits; the head is then compiled once per pool prefix instead of once."""
  return os.getenv("FZ_NEURON_HEAD_FUSE", "1") not in ("", "0")

def neuron_device_present() -> bool:
  return any(Path("/dev").glob("neuron[0-9]*"))

class NEURONDevice(DeviceBase):
  """The NEURON device (N4): kernels run on a NeuronCore of an inf2/trn1 instance through nki.jit in the `.venv-nki` worker
  (scripts/nki_env.sh --cc; libnrt from the Neuron DLAMI). The NKISHIM and NKISIM tiers subclass it with other backends."""
  tier = "NEURON"
  single_view = True   # the NKI renderer lowers single-view ShapeTrackers only: the scheduler materializes the rest (N5)
  def __init__(self, name: str, renderer: NKIRenderer | None = None):
    self.name = name
    self.allocator = NPYAllocator()
    self.renderer = renderer or NKIRenderer()
    self.compiler, self.runtime = self._backend()
    self._cache: dict[tuple, Program] = {}
    from fusezap.codegen.nki_kernels import ATTN_MASK_PARAMS, MATMUL_PARAMS
    from fusezap.schedule import ATTENTION, CustomKernel
    t = name.split(":")[0].upper()   # the scheduler offers a kernel to a device whose base name has a source entry
    self.custom_kernels: dict[str, CustomKernel] = {
      "matmul": CustomKernel("fz_matmul", {t: "nki_kernels.render_matmul"}, ("C", "A", "B") + MATMUL_PARAMS, 3),
      "attention": CustomKernel("fz_attention", {t: "nki_kernels.render_attention"}, ATTENTION.signature, 4),
      "attention_mask": CustomKernel("fz_attention_mask", {t: "nki_kernels.render_attention"},
                                     ATTENTION.signature[:4] + ("M",) + ATTENTION.signature[4:] + ATTN_MASK_PARAMS, 5)}

  def _backend(self) -> tuple[Compiler, Any]:
    if not neuron_device_present():
      raise RuntimeError(f"{self.tier}: needs a Neuron device (/dev/neuron*, an inf2/trn1 instance) and its runtime (N4)")
    py = nki_python()
    if not py.exists(): raise RuntimeError(f"{self.tier}: no nki env at {py}; run scripts/nki_env.sh --cc on the instance (N4)")
    rep = NKIWorker.get().call({"op": "version"})
    if not rep["ok"]: raise RuntimeError(f"{self.tier}: the nki worker cannot start: {rep['err']} (N4)")
    if (mode := os.getenv("FZ_NEURON_RT", "nrt").lower()) == "jit": return NKIHwCompiler(rep["nki"]), NeuronHwProgram
    if mode != "nrt": raise RuntimeError(f"{self.tier}: FZ_NEURON_RT must be nrt (persistent, default) or jit (nki.jit fallback), not {mode!r}")
    if not nrt_python().exists():
      raise RuntimeError(f"{self.tier}: no nrtpy env at {nrt_python()}; run scripts/nki_env.sh --nrt on the instance, or FZ_NEURON_RT=jit (N4)")
    nrt = NrtWorker.get().call({"op": "version"})
    if not nrt["ok"]: raise RuntimeError(f"{self.tier}: the runtime worker cannot start: {nrt['err']} (N4)")
    self.allocator = NeuronAllocator()
    return NeuronNrtCompiler(rep["nki"], nrt["nrtpy"]), NeuronNeffProgram

  def _scratch(self, nbytes: int) -> NrtBuf:
    """A kept device buffer standing in for a size-0 or missing argument (the NEFF still declares 1 element)."""
    sc = self.__dict__.setdefault("_scratch_bufs", {})
    if nbytes not in sc: sc[nbytes] = self.allocator.alloc(nbytes)
    return sc[nbytes]

  def run_neff(self, prg: "NeuronNeffProgram", bufs: tuple, iters: int = 1) -> list[float]:
    """Launch a NEFF program on device-resident buffers: binds buffer ids, no host data moves."""
    n = prg.prog.n_bufs
    hs, specs = [], []
    for b in bufs[:n]:
      npd = to_np(b.dtype)
      if b.size: hs.append(b.ensure_allocated().handle); specs.append((b.size, npd.str))
      else: hs.append(self._scratch(npd.itemsize)); specs.append((1, npd.str))
    for _ in range(n - len(hs)): hs.append(self._scratch(1)); specs.append((1, np.dtype(np.uint8).str))
    return prg.launch(self.allocator, hs, tuple(specs), iters)

  def program(self, si: ScheduleItem) -> Program:
    vv = tuple(sorted(((str(v), int(x)) for v, x in si.var_vals.items())))
    fast = self.__dict__.setdefault("_by_ast", {})   # a jit replay passes the captured AST object again: skip hashing it
    if (hit := fast.get(id(si.ast))) is not None and hit[0] is si.ast and hit[1] == vv: return hit[2]
    from fusezap.codegen.lower import _ast_key
    key = (_ast_key(si.ast), vv, self.renderer.pmax, self.renderer.fmax)
    if (prg := self._cache.get(key)) is None:
      prog = self.renderer.render(si.ast, si.var_vals)
      self._cache[key] = prg = self.runtime(prog, self.compiler.compile(prog.src))
    if len(fast) > 4096: fast.clear()
    fast[id(si.ast)] = (si.ast, vv, prg)   # holds the AST, so its id is not reused while the entry lives
    return prg

  def synchronize(self) -> None:
    if isinstance(self.allocator, NeuronAllocator): self.allocator.flush()

  def replay_launcher(self, si: ScheduleItem) -> "NeffReplay | None":
    """fz.jit asks at capture: a launcher for this item's replays (N7, host_fast, the persistent path), else None (run())."""
    if not isinstance(self.allocator, NeuronAllocator) or not self.allocator.fast or si.var_vals or not si.ast: return None
    if si.kind not in ("kernel", "custom") or (si.kind == "custom" and hasattr(si.ast[0], "launch")): return None
    return NeffReplay(self, si)

  def _neff_program(self, si: ScheduleItem) -> "NeuronNeffProgram | None":
    """The program run() launches for si as one NEFF exec, else None (a fence, a fused kernel launched as its members)."""
    try:
      if si.kind == "kernel": prg = self.program(si)
      elif si.kind == "custom":
        call = si.ast[0]
        if (sp := getattr(call, "spec", None)) is not None:
          if sp.name == "fz_fence" or (isinstance(call, FusedCall) and sp in self.__dict__.get("_unfusable", ())): return None
          prg = self.spec_program(sp)
        elif (nm := getattr(getattr(call, "kernel", None), "name", None)) is not None and hasattr(call, "params"): prg = self.custom_program(nm, call.params)
        else: return None
      else: return None
    except NKIUnsupported: return None
    return prg if isinstance(prg, NeuronNeffProgram) else None

  def custom_program(self, name: str, params: tuple[int, ...]) -> Program:
    """The rendered + compiled NKI kernel of a custom item (fusezap.codegen.nki_kernels), cached by kernel name and params."""
    from fusezap.codegen import nki_kernels as nk
    key = ("custom", name, tuple(int(p) for p in params))
    if (prg := self._cache.get(key)) is None:
      if name == "fz_matmul": prog = nk.render_matmul(params)
      elif name in ("fz_attention", "fz_attention_mask"): prog = nk.render_attention(nk.spec_from_params(params, name == "fz_attention_mask"))
      else: raise NKIUnsupported(f"{self.tier}: no NKI kernel for custom item {name!r}")
      self._cache[key] = prg = self.runtime(prog, self.compiler.compile(prog.src))
    return prg

  def spec_program(self, sp: Any) -> Program:
    """The kernel of a NeuronCall spec (nki_kernels.AttnSpec / NormSpec: `sp.render()`), cached by the spec."""
    if (prg := self._cache.get(key := ("spec", sp))) is None:
      prog = sp.render()
      self._cache[key] = prg = self.runtime(prog, self.compiler.compile(prog.src))
    return prg

  def attention_program(self, sp: Any) -> Program:
    """The flash attention kernel of an AttnSpec (the direct entry point `attention`), cached by the spec."""
    from fusezap.codegen.nki_kernels import render_attention
    if (prg := self._cache.get(key := ("attention", sp))) is None:
      prog = render_attention(sp)
      self._cache[key] = prg = self.runtime(prog, self.compiler.compile(prog.src))
    return prg

  def run_program(self, prg: Program, bufs: tuple) -> float:
    if isinstance(prg, NeuronNeffProgram):
      st = time.perf_counter()
      self.run_neff(prg, bufs)
      return time.perf_counter() - st
    n = prg.prog.n_bufs
    arrays = [b.ensure_allocated().handle[:b.nbytes].view(to_np(b.dtype)) if b.size else np.zeros(1, to_np(b.dtype)) for b in bufs[:n]]
    arrays += [np.zeros(1, np.uint8)] * (n - len(arrays))
    st = time.perf_counter()
    prg(*arrays)
    return time.perf_counter() - st

  # ---- N6: fusion of a captured forward's items into fewer launches (fz.jit hands every capture to fuse_schedule)
  def _member(self, si: ScheduleItem) -> tuple[NKIProgram, tuple] | None:
    """(rendered kernel, its argument buffers) of an item that can be part of a fused kernel, else None."""
    from fusezap.codegen.nki_kernels import FenceSpec
    if si.kind not in ("kernel", "custom") or si.var_vals or not si.ast: return None
    try:
      if si.kind == "kernel": prg = self.program(si)
      else:
        call = si.ast[0]
        if hasattr(call, "launch"): return None
        if (sp := getattr(call, "spec", None)) is not None:
          if isinstance(sp, FenceSpec) or not getattr(sp, "fusable", True): return None   # N8: the all-reduce launches alone
          prg = self.spec_program(sp)
        elif (nm := getattr(getattr(call, "kernel", None), "name", None)) is not None and hasattr(call, "params"): prg = self.custom_program(nm, call.params)
        else: return None
    except NKIUnsupported: return None
    prog = getattr(prg, "prog", None)
    if prog is None or len(si.bufs) < prog.n_bufs: return None
    bufs = tuple(si.bufs[:prog.n_bufs])
    if any(b.size == 0 or b.device != self.name for b in bufs): return None
    return prog, bufs

  def fuse_schedule(self, sched: list[ScheduleItem]) -> list[ScheduleItem]:
    """N6 (docs/NKI_BACKEND.md §4 exception 9): runs of fusable items between fences (neuron_kv.fence: the layer boundaries
    Llama.forward_tokens records on NEURON) become one kind="custom" item each, a FusedSpec kernel over the union of their
    buffers, written buffers first (so fz.jit sees them as the item's outputs). FZ_NEURON_FUSE=k: k layers per fused kernel
    (default 24; 0: no fences, nothing fused, N5's launches); FZ_NEURON_HEAD_FUSE=1 (default): the head's fence is not a boundary (the
    head joins the last layers' kernel); a fence with `every` set (I75: a tensor-parallel prefill, FZ_NEURON_TP_PREFILL_FUSE) groups
    that many layers instead of FZ_NEURON_FUSE; such a group of more than FUSE_FALLBACK layers also carries its split into
    groups of FUSE_FALLBACK layers (FusedCall.subs), the fallback if its NEFF does not compile. Fences never reach a replay.
    Every fused item is remembered (self._fused_seen) for precompile (I75 prewarm)."""
    from fusezap.codegen.nki_kernels import FenceSpec, FusedSpec
    k, cap, hf = fuse_layers(), fuse_max(), head_fuse()
    out: list[ScheduleItem] = []
    group: list[tuple[ScheduleItem, NKIProgram, tuple]] = []
    dts: dict[int, Any] = {}
    splits: list[int] = []
    def flush() -> None:
      if len(group) == 1: out.append(group[0][0])
      elif group: out.append(self._fused(group, FusedSpec, tuple(s for s in splits if 0 < s < len(group))))
      group.clear(); dts.clear(); splits.clear()
    for si in sched:
      sp = getattr(si.ast[0], "spec", None) if si.kind == "custom" and si.ast else None
      if isinstance(sp, FenceSpec):   # I75: a fence may carry its forward's own group size (FenceSpec.every: a TP prefill)
        if not k or (sp.hard and not hf) or (not sp.hard and sp.layer % (sp.every or k) == 0): flush()
        elif sp.every > FUSE_FALLBACK and not sp.hard and sp.layer % FUSE_FALLBACK == 0: splits.append(len(group))
        continue
      m = self._member(si) if k else None
      if m is None: flush(); out.append(si); continue
      prog, bufs = m
      mine = {id(b): prog.buf_dtypes.get(i) for i, b in enumerate(bufs)}
      if (cap and len(group) >= cap) or any(d is not None and dts.get(i) not in (None, d) for i, d in mine.items()): flush()
      for i, d in mine.items():
        if d is not None: dts[i] = d
      group.append((si, prog, bufs))
    flush()
    seen = self.__dict__.setdefault("_fused_seen", {})
    if len(seen) > 4096: seen.clear()
    for si in out:
      if si.kind == "custom" and isinstance(call := si.ast[0], FusedCall): seen.setdefault(call.spec, call)
    return out

  def _fused(self, group: list, FusedSpec: Any, splits: tuple = ()) -> ScheduleItem:
    si = self._fuse_group(group, FusedSpec)
    call = si.ast[0]
    call.arg_specs = tuple((int(b.size), to_np(b.dtype).str) for b in si.bufs)
    if splits:   # I75: the fallback groups, each item's buffers as indices into this one's
      pos = {id(b): n for n, b in enumerate(si.bufs)}
      subs = []
      for a, b in zip((0,) + splits, splits + (len(group),)):
        part = group[a:b]
        if len(part) == 1: sub, sbufs = part[0][0], part[0][2]   # one member: launched as itself
        else:
          sub = self._fuse_group(part, FusedSpec)
          sbufs = sub.bufs
          sub.ast[0].arg_specs = tuple(call.arg_specs[pos[id(x)]] for x in sbufs)
        subs.append((ScheduleItem(sub.ast, (), sub.var_vals, sub.kind, sub.n_outputs), tuple(pos[id(x)] for x in sbufs)))
      call.subs = tuple(subs)
    return si

  @staticmethod
  def _fuse_group(group: list, FusedSpec: Any) -> ScheduleItem:
    args: list = []
    idx: dict[int, int] = {}
    written: list[int] = []
    dts: dict[int, Any] = {}
    maps = []
    for _, prog, bufs in group:
      amap = []
      for i, b in enumerate(bufs):
        if (j := idx.get(id(b))) is None: idx[id(b)] = j = len(args); args.append(b)
        if i in prog.buf_dtypes: dts.setdefault(j, prog.buf_dtypes[i])
        amap.append(j)
      written += [amap[o] for o in prog.outs if amap[o] not in written]
      maps.append(amap)
    order = written + [j for j in range(len(args)) if j not in written]   # outputs first: fz.jit's n_outputs convention
    new = {j: n for n, j in enumerate(order)}
    parts = tuple((prog.name, prog.src, tuple(new[j] for j in amap)) for (_, prog, _), amap in zip(group, maps))
    sp = FusedSpec(parts, len(args), tuple(range(len(written))), tuple(sorted((new[j], d) for j, d in dts.items())))
    pos = {id(args[j]): n for n, j in enumerate(order)}
    members = tuple((ScheduleItem(si.ast, (), si.var_vals, si.kind, si.n_outputs), tuple(pos[id(b)] for b in bufs))
                    for si, _, bufs in group)
    return ScheduleItem((FusedCall(sp, members),), tuple(args[j] for j in order), {}, "custom", len(written))

  def _unfuse(self, call: "FusedCall", err: str) -> None:
    """A fused kernel that failed to compile: launched as its fallback groups (FusedCall.subs, I75) or its members from now on."""
    sp = call.spec
    self.__dict__.setdefault("_unfusable", set()).add(sp)
    how = f"as {len(call.subs)} kernels of up to {FUSE_FALLBACK} layers" if call.subs else "its members separately"
    print(f"NEURON: fused kernel {sp.name} ({len(sp.parts)} kernels) failed, launching {how}: {err[:600]}", file=sys.stderr, flush=True)

  def precompile(self, jobs: int = 1, rank: int = 0, world: int = 1, log: Any = None) -> list[dict]:
    """I75 prewarm: compile the NEFF of every fused kernel captured so far (self._fused_seen) that is not loaded yet, `jobs`
    at a time, each thread with its own nki worker (compile_worker): the compiles are independent neuronx-cc runs, so a
    cold start pays about the longest one instead of their sum. Rank r starts r/world of the way down the list, so two
    ranks compiling the same keys start on different ones (cached_neff's per-key lock makes the other rank wait for, then
    read, each). A kernel that fails to compile is marked unfusable now (logged) and its fallback groups (FusedCall.subs)
    are compiled instead. Returns one record per NEFF: kernel, key, hit (in the disk cache already), seconds, ok, error.
    Compiles only (the runtime load happens at the forward's next launch). Nothing on a tier without NEFFs (NKISHIM)."""
    seen = self.__dict__.get("_fused_seen", {})
    bad = self.__dict__.setdefault("_unfusable", set())
    todo, recs, lock = [], [], threading.Lock()
    def add(call: Any) -> None:
      if call.spec in bad or not call.arg_specs: return
      try: prg = self.spec_program(call.spec)
      except NKISimError as e:   # does not render: the fallback groups (or members) from now on, as run() would
        self._unfuse(call, str(e))
        for m, _ in call.subs:
          if isinstance(m.ast[0], FusedCall): add(m.ast[0])
        return
      if not isinstance(prg, NeuronNeffProgram): return
      n = prg.prog.n_bufs
      specs = tuple(call.arg_specs[:n]) + ((1, np.dtype(np.uint8).str),) * max(0, n - len(call.arg_specs))
      if specs not in prg.loaded: todo.append((call, prg, specs))
    for call in list(seen.values()): add(call)
    if not todo: return recs
    k = (rank * len(todo)) // max(world, 1)
    todo = todo[k:] + todo[:k]
    target, versions = neuron_target(), _nki_versions()
    def one(item: tuple) -> list:
      call, prg, specs = item
      key = neff_key(prg.prog, list(specs), target, versions)
      hit = (neff_cache_dir() / key / "kernel.neff").exists()
      t0 = time.perf_counter()
      try:
        cached_neff(prg.prog, list(specs), target, versions)
        r = {"kernel": prg.prog.name, "key": key, "hit": hit, "seconds": round(time.perf_counter() - t0, 2), "ok": True}
      except NKISimError as e:
        self._unfuse(call, str(e))
        r = {"kernel": prg.prog.name, "key": key, "hit": False, "seconds": round(time.perf_counter() - t0, 2), "ok": False, "error": str(e)[:300]}
      except Exception as e:   # not the compiler's verdict (a worker died, ...): left to the first launch, which compiles it again
        r = {"kernel": prg.prog.name, "key": key, "hit": False, "seconds": round(time.perf_counter() - t0, 2), "ok": False,
             "error": f"{type(e).__name__}: {e}"[:300], "retry": True}
        if log is not None: log(f"NEFF {r['kernel']} not precompiled ({r['error']}); it compiles at its first launch")
        with lock: recs.append(r)
        return []
      if log is not None: log(f"NEFF {r['kernel']} {'cache hit' if r['hit'] else 'compiled' if r['ok'] else 'FAILED'} in {r['seconds']} s")
      more = []
      if not r["ok"]:
        for m, ix in call.subs:
          sub = m.ast[0]
          if isinstance(sub, FusedCall) and sub.arg_specs:
            p2 = self.spec_program(sub.spec)
            n = p2.prog.n_bufs
            more.append((sub, p2, tuple(sub.arg_specs[:n]) + ((1, np.dtype(np.uint8).str),) * max(0, n - len(sub.arg_specs))))
      with lock: recs.append(r)
      return more
    def worker_loop(q: list) -> None:
      w = None
      try:
        while True:
          with lock:
            if not q: return
            item = q.pop(0)
          if w is None and jobs > 1: _TLS.worker = w = compile_worker()
          more = one(item)
          with lock: q.extend(more)
      finally:
        _TLS.worker = None
        if w is not None and w is not NKIWorker.__dict__.get("_inst"):
          try: w.close()
          except Exception: pass
    q = list(todo)
    ts = [threading.Thread(target=worker_loop, args=(q,), daemon=True) for _ in range(max(1, min(jobs, len(q))))]
    for t in ts: t.start()
    for t in ts: t.join()
    if q: worker_loop(q)   # fallback groups queued after every thread left
    return recs

  def run(self, si: ScheduleItem) -> float | None:
    if si.kind == "custom":
      assert si.ast is not None
      call = si.ast[0]
      if (sp := getattr(call, "spec", None)) is not None:   # NeuronCall
        if sp.name == "fz_fence": return 0.0
        if isinstance(call, FusedCall):
          bad = self.__dict__.setdefault("_unfusable", set())
          if sp not in bad:
            try: return self.run_program(self.spec_program(sp), si.bufs)
            except NKISimError as e:   # the fused kernel failed to compile: its members one by one from now on
              self._unfuse(call, str(e))
          return sum(self.run(ScheduleItem(m.ast, tuple(si.bufs[j] for j in ix), m.var_vals, m.kind, m.n_outputs)) or 0.0
                     for m, ix in (call.subs or call.members))
        return self.run_program(self.spec_program(sp), si.bufs)
      name = getattr(getattr(call, "kernel", None), "name", None)
      if name is None or not hasattr(call, "params"): raise NKIUnsupported(f"{self.tier}: custom item {call!r} has no NKI kernel")
      return self.run_program(self.custom_program(name, call.params), si.bufs)
    if si.kind != "kernel": raise NKIUnsupported(f"{self.tier}: {si.kind} items are not supported")
    assert si.ast is not None
    return self.run_program(self.program(si), si.bufs)

class NKISHIMDevice(NEURONDevice):
  """NEURON with the numpy shim as compiler + runtime: generated kernels run on the Mac (docs/NKI_BACKEND.md §6)."""
  tier = "NKISHIM"
  def _backend(self) -> tuple[Compiler, Any]: return NKIShimCompiler(), NKIShimProgram

class NKISIMDevice(NEURONDevice):
  """NEURON with the real nki 0.6 CPU simulator in a `.venv-nki` worker (N2; Linux x86_64 only, e.g. a RunPod CPU pod)."""
  tier = "NKISIM"
  def _backend(self) -> tuple[Compiler, Any]:
    py = nki_python()
    if not py.exists(): raise RuntimeError(f"NKISIM: no nki env at {py}; run scripts/nki_env.sh on a Linux x86_64 host (or set FZ_NKI_PYTHON)")
    rep = NKIWorker.get().call({"op": "version"})
    if not rep["ok"]: raise RuntimeError(f"NKISIM: the nki worker cannot start: {rep['err']}")
    return NKISimCompiler(rep["nki"]), NKISimProgram

# ---------------------------------------------------------------- direct entry points (Tensors in, Tensor out)

def kmajor(w: Any) -> Any:
  """A torch-layout weight (N, K) relaid K-major once at load time: the realized (K, N) transpose. `x @ kmajor(w)` equals
  x.linear(w) without bias, and the matmul kernel's moving tiles B[k0:k0+128, n0:n0+512] are then contiguous rows."""
  return w.transpose(-1, -2).contiguous().realize()

def attention(q: Any, k: Any, v: Any, causal: bool = False, scale: float | None = None, coff: int = 0, pos: Any = None) -> Any:
  """Flash attention on a NEURON tier == scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale, enable_gqa=True)
  in float32: q (..., Hq, L, D), k (..., Hkv, S, D), v (..., Hkv, S, Dv), Hq a multiple of Hkv (q head h reads kv head
  h // (Hq / Hkv)). Leading dims broadcast. Masks:
    causal: key s is visible to query row l iff s <= l + coff (coff = 0 is torch's is_causal; coff = S - L puts the last
            query on the last key), applied with `affine_select`; key blocks past a row tile's last visible key are skipped.
    pos:    an int32 Tensor (L,): key s is visible to query row l iff s <= pos[l] (the decode / chunked-prefill mask of a
            fixed-size KV cache), from `iota` key indices compared with the positions on chip.
  Inference only (no autograd). The q heads of one kv head share its K/V reads (GQA rows are folded), except with causal,
  whose row tiles stay inside one head so the mask stays affine in the partition index."""
  from fusezap.codegen.nki_kernels import AttnSpec
  from fusezap.device import Buffer, Device
  from fusezap.dtype import dtypes
  from fusezap.helpers import prod
  from fusezap.lazy import LazyBuffer
  from fusezap.tensor import Tensor
  assert not (causal and pos is not None), "attention: pass causal or pos, not both"
  assert q.dtype == k.dtype == v.dtype == dtypes.float32, f"NEURON attention is float32 here (bf16 is N5), got {q.dtype}"
  squeeze = q.ndim == 2
  if squeeze: q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
  (Hq, L, D), (Hkv, S, Dk), (Hv, Sv, Dv) = q.shape[-3:], k.shape[-3:], v.shape[-3:]
  assert D == Dk and S == Sv and Hkv == Hv and Hq % Hkv == 0, f"attention shapes {q.shape}, {k.shape}, {v.shape}"
  G = Hq // Hkv
  nbd = max(t.ndim - 3 for t in (q, k, v))
  q, k, v = [t.reshape((1,) * (nbd - (t.ndim - 3)) + tuple(t.shape)) for t in (q, k, v)]
  bs = tuple(max(t.shape[i] for t in (q, k, v)) for i in range(nbd))
  B = prod(bs)
  dev = Device[q.device]
  def base(t: Any) -> Any:   # every input expanded to the full batch, so one stride per batch level
    if tuple(t.shape[:nbd]) != bs: t = t.expand(bs + tuple(t.shape[nbd:]))
    return t._realized_contiguous().realized
  Q, K, V = base(q), base(k), base(v)
  O = Buffer(q.device, B * Hq * L * Dv, dtypes.float32)
  ks, vs = (S * D, 0, D, 1, 0), (S * Dv, 0, Dv, 1, 0)
  if causal:   # b = (batch * kv head, q head in its group); rows = that head's L rows
    nb, nr = (B * Hkv, G), (1, L)
    qs, os_ = (G * L * D, L * D, D, 1, 0), (G * L * Dv, L * Dv, Dv, 1, 0)
  else:        # b = batch * kv head; rows = (q head in the group, l): one stride over the G*L rows of the contiguous q
    nb, nr = (B * Hkv, 1), (G, L)
    qs, os_ = (G * L * D, 0, D, 1, 0), (G * L * Dv, 0, Dv, 1, 0)
  mode, m, bufs = ("causal" if causal else "pos" if pos is not None else "none"), (), [O, Q, K, V]
  if pos is not None:   # positions indexed by l, the inner row level: row (g, l) reads pos[l]
    assert tuple(pos.shape) == (L,) and pos.dtype == dtypes.int32, f"attention pos must be int32 ({L},), got {pos.dtype} {pos.shape}"
    bufs.append(pos._realized_contiguous().realized)
    m = (0, 1)
  sp = AttnSpec(nb, nr, S, D, Dv, qs, ks, vs, os_, 1.0 / math.sqrt(D) if scale is None else float(scale), mode, m, int(coff))
  if B and Hq and L and Dv:
    if S == 0: O.copyin(memoryview(bytearray(O.nbytes)))   # no keys: zeros, like the CUDA and Metal kernels
    else: dev.run_program(dev.attention_program(sp), tuple(bufs))
  out = Tensor(LazyBuffer.from_buffer(O, bs + (Hq, L, Dv)))
  return out.reshape(L, Dv) if squeeze and not bs else out
