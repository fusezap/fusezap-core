"""Tensor parallelism for fusezap.serve (I57 M1): Megatron-style sharding of the Llama decoder over W processes, one per
device, with an all-reduce after each row-parallel projection.

  g = launch(2, "CUDA")                          # rank 0: spawn rank 1 (`python -m fusezap.serve.dist`), init NCCL
  g.post("load", {"model": path, ...})           # every worker loads its own shard while rank 0 loads its own
  m = load_model(path, ..., tp=g)                # Llama(tp=g): this rank's slices of the weights, cfg = the local sizes
  y = all_reduce(x, g)                           # inside a model step: a kind="custom" item, recorded by fz.jit

Processes (as vLLM's multiproc executor): rank 0 runs the HTTP API, tokenizer, scheduler and sampling; every model call
it makes is mirrored (SPMD): `with g.spmd(op, *args):` sends (op, args) over the rank's control connection to every
worker before rank 0 runs the call itself, and the worker loop (`worker_main`) runs the same call on its shard. The
mirrored calls are LLMEngine forwards ("fwd": the step's host arrays: token ids, positions, the paged-cache metadata,
the lookahead source rows), Llama.prefill / decode / reset, the logprobs scorer and the allocator release; workers keep
no scheduler, so a worker's forwards are exactly rank 0's, in the same order, and the all-reduces pair up.

Sharding (Megatron; `tp_slice`): q/k/v rows by head group (rank r holds q heads [r Hq/W, (r+1) Hq/W) and kv heads
[r Hkv/W, (r+1) Hkv/W): GQA groups stay whole, since Hq/Hkv is the same per rank), their biases likewise; gate/up rows
by intermediate dim; o_proj / down_proj columns (row-parallel: each rank's product is a partial sum, all-reduced, then
the bias once). Norms, embedding and lm_head are replicated (lm_head on every rank: a worker's logits are bit-identical
to rank 0's, so CUDA lookahead can feed a worker its own device argmax; 2.1 GB for 70B). Each rank reads only its slices
from the mmap'd safetensors (weights.load_safetensors), so no process materializes the full checkpoint. The rank's
Llama.cfg holds the local head / intermediate counts, so the KV cache holds only the rank's kv heads.

Collectives (one per process group, `Comm.all_reduce(dev, out, inp, n, dtype)`):
  NcclComm   CUDA: libnccl through ctypes (the nvidia-nccl wheel, FZ_NCCL_LIB, or the system's), ncclGetUniqueId on rank 0
             sent over the control connection, ncclAllReduce (sum) on the device's current stream: the legacy default
             stream, or the capture stream while fz.jit records a CUDA graph, so the all-reduce is captured with the step.
  HostComm   any device (tests on a Mac): partials copied to the host, summed on rank 0 in float32 in rank order, the
             result sent back to every rank (identical bits everywhere); sockets with timeouts (FZ_TP_TIMEOUT).

Speed (I57 M3): on CUDA with peer access between every pair of devices (NVLink), all-reduces up to FZ_TP_AR_MAX_MB
(8 MB: every decode step, prefill chunks up to 512 rows of 8192) run fusezap.serve.custom_ar's one-shot kernel instead
of NCCL (TPComm picks per message; NCCL for the rest), and the all-reduce that feeds a residual add + RMSNorm runs fused
with it (all_reduce_add_norm, Llama.forward_tokens on <= 64 rows); FZ_TP_CUSTOM_AR=0 / FZ_TP_AR_FUSE=0 turn them off.
Rank 0's messages go through a shared-memory ring (fusezap.serve.shmchan; FZ_TP_SHM=0: the socket as before) that the
workers poll, so a worker starts a forward microseconds after rank 0 sent it; replies, errors and payloads larger than a
ring slot still use the socket. FZ_TP_PROFILE=<dir> writes a per-forward timeline per rank (fusezap.serve.tpprof).

Shutdown (TPGroup.close, bounded on every path; FZ_TP_DEBUG=1 logs each step with a timestamp on stderr): rank 0 sends
"quit", aborts its own communicator, waits up to FZ_TP_CLOSE_TIMEOUT (10 s) for the workers, then kills them. A worker
on "quit" (or when its control connection closes) aborts its communicator and leaves with os._exit(0) (no interpreter
teardown to wait on). The communicator is released with ncclCommAbort, not ncclCommDestroy: since NCCL 2.18
ncclCommDestroy is an intra-node collective that waits for every rank to destroy too, so rank 0 waiting for the workers
to exit while they sat in ncclCommDestroy waiting for rank 0, then killing them and calling ncclCommDestroy itself
(waiting for a peer that no longer exists), hung `pytest tests/test_tp.py` on CUDA forever after the last test
(I57 M1). ncclCommAbort is local; a synchronize before it is bounded too (a collective stuck on a dead peer is what
ncclCommAbort unblocks). torch's ProcessGroupNCCL.shutdown aborts the same way. NCCL 2.30's ncclCommAbort did not return
within 10 s on 2x H200 (I57 M2), so the abort is waited for FZ_TP_ABORT_WAIT (1 s) only: the process exits right after
(os._exit frees the communicator with the context) and a test process just leaves it behind.

Failure handling: rank 0 runs a watchdog thread; a worker that exits while serving stops rank 0 with a clear message
(os._exit(1): NCCL cannot be unblocked otherwise, and `fusezap serve`'s supervisor restarts it). A worker exits when
its control connection closes or its parent process changes; an error in a worker is reported to rank 0 (a startup
barrier raises it) and the worker exits.
"""
from __future__ import annotations
import contextlib, ctypes, dataclasses, os, secrets, subprocess, sys, threading, time, traceback
from typing import Any, Callable
import numpy as np
from fusezap.dtype import DType, dtypes

TIMEOUT = float(os.environ.get("FZ_TP_TIMEOUT", "") or 600.0)   # seconds a collective or barrier waits for a peer
CLOSE_TIMEOUT = float(os.environ.get("FZ_TP_CLOSE_TIMEOUT", "") or 10.0)   # seconds each shutdown step may take
ABORT_WAIT = float(os.environ.get("FZ_TP_ABORT_WAIT", "") or 1.0)          # seconds ncclCommAbort is waited for

def debug_on() -> bool: return os.environ.get("FZ_TP_DEBUG", "") not in ("", "0")

def dbg(msg: str) -> None:
  """FZ_TP_DEBUG=1: one timestamped line on stderr (wall clock to the ms, rank, pid), so a hang shows its last step."""
  if not debug_on(): return
  t = time.time()
  print(f"[{time.strftime('%H:%M:%S', time.localtime(t))}.{int(t * 1000) % 1000:03d}] fusezap tp rank {os.environ.get('FZ_TP_RANK', '0')} "
        f"pid {os.getpid()}: {msg}", file=sys.stderr, flush=True)

def _bounded(fn: Callable[[], Any], timeout: float, what: str) -> bool:
  """Run fn in a daemon thread; True if it returned within timeout (its exceptions are logged, not raised). A call that
  does not return (a driver or NCCL call waiting for a peer) is left behind in its thread, so shutdown goes on."""
  done = threading.Event()
  def run() -> None:
    try: fn()
    except Exception as e: dbg(f"{what}: {e!r}")
    finally: done.set()
  dbg(f"{what} ...")
  t0 = time.monotonic()
  threading.Thread(target=run, name=f"fusezap-tp-{what.split()[0]}", daemon=True).start()
  ok = done.wait(timeout)
  dbg(f"{what}: {'done' if ok else 'DID NOT RETURN'} in {time.monotonic() - t0:.2f} s")
  return ok

# ---------------------------------------------------------------- sharding

def check_tp(cfg: Any, world: int) -> None:
  """Raise unless cfg splits evenly over `world` ranks (heads, kv heads, intermediate size)."""
  bad = [f"{k} {v}" for k, v in (("num_attention_heads", cfg.n_heads), ("num_key_value_heads", cfg.n_kv_heads),
                                 ("intermediate_size", cfg.intermediate_size)) if v % world]
  if bad: raise ValueError(f"tensor parallel size {world} must divide {', '.join(bad)}")

def local_cfg(cfg: Any, world: int) -> Any:
  """The per-rank config: heads, kv heads and intermediate size divided by world (head_dim, hidden, vocab unchanged)."""
  check_tp(cfg, world)
  return dataclasses.replace(cfg, n_heads=cfg.n_heads // world, n_kv_heads=cfg.n_kv_heads // world,
                             intermediate_size=cfg.intermediate_size // world)

COL = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")   # column-parallel: output rows split (weight and bias)
ROW = ("o_proj", "down_proj")                                    # row-parallel: input columns split (bias unsplit)

def tp_slice(name: str, a: np.ndarray, cfg: Any, rank: int, world: int) -> np.ndarray:
  """Rank `rank`'s slice of HF tensor `name` (a view: an mmap'd array is only read where sliced). cfg: the full config."""
  if world == 1: return a
  parts = name.split(".")
  kind = parts[-2] if len(parts) >= 2 else ""
  hd = cfg.head_dim
  n = {"q_proj": cfg.n_heads * hd, "k_proj": cfg.n_kv_heads * hd, "v_proj": cfg.n_kv_heads * hd,
       "gate_proj": cfg.intermediate_size, "up_proj": cfg.intermediate_size, "o_proj": cfg.n_heads * hd,
       "down_proj": cfg.intermediate_size}.get(kind)
  if n is None: return a
  s = n // world
  if kind in COL:
    assert a.shape[0] == n, f"{name}: {a.shape[0]} rows, expected {n}"
    return a[rank * s:(rank + 1) * s]
  if parts[-1] == "bias": return a        # a row-parallel bias is added once, after the all-reduce
  assert a.shape[1] == n, f"{name}: {a.shape[1]} columns, expected {n}"
  return a[:, rank * s:(rank + 1) * s]

# ---------------------------------------------------------------- host <-> float32 (the host formats of fusezap dtypes)

def _to_f32(b: bytes | memoryview | np.ndarray, dt: DType) -> np.ndarray:
  a = np.frombuffer(b, np.uint8) if not isinstance(b, np.ndarray) else b.view(np.uint8)
  if dt == dtypes.bfloat16: return (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
  if dt == dtypes.float16: return a.view(np.float16).astype(np.float32)
  if dt == dtypes.float32: return a.view(np.float32).copy()
  raise NotImplementedError(f"all-reduce of {dt}")

def _from_f32(x: np.ndarray, dt: DType) -> np.ndarray:
  if dt == dtypes.bfloat16:
    from fusezap.serve.weights import f32_to_bf16
    return f32_to_bf16(x)
  return x.astype(np.float16 if dt == dtypes.float16 else np.float32)

# ---------------------------------------------------------------- collectives

class HostComm:
  """All-reduce through the host (any device): rank 0 sums the partials in float32 in rank order and sends the
  rounded result back, so every rank gets the same bits. peers: rank 0: {rank: Connection}; a worker: {0: Connection}."""
  name = "host"
  def __init__(self, rank: int, world: int, peers: dict[int, Any]):
    self.rank, self.world, self.peers = rank, world, peers
  def _recv(self, r: int) -> bytes:
    c = self.peers[r]
    if not c.poll(TIMEOUT): raise RuntimeError(f"tensor-parallel all-reduce: no data from rank {r} in {TIMEOUT:.0f} s")
    try: return c.recv_bytes()
    except (EOFError, OSError) as e: raise RuntimeError(f"tensor-parallel all-reduce: rank {r} is gone ({e!r})") from None
  def reduce_host(self, a: np.ndarray, dt: DType) -> np.ndarray:
    """a: this rank's partial (host format of dt, any shape) -> the sum over ranks, same format."""
    if self.rank != 0:
      self.peers[0].send_bytes(a)
      return np.frombuffer(self._recv(0), a.dtype).reshape(a.shape)
    acc = _to_f32(np.ascontiguousarray(a), dt)
    for r in range(1, self.world): acc += _to_f32(self._recv(r), dt)
    out = np.ascontiguousarray(_from_f32(acc, dt)).reshape(a.shape)
    for r in range(1, self.world): self.peers[r].send_bytes(out)
    return out
  def all_reduce(self, dev: Any, out: Any, inp: Any, n: int, dt: DType) -> None:
    from fusezap.dtype import to_np
    host = np.empty(n, to_np(dt) if dt != dtypes.bfloat16 else np.uint16)
    inp.copyout(memoryview(host).cast("B"))
    out.copyin(memoryview(np.ascontiguousarray(self.reduce_host(host, dt))).cast("B"))
  def close(self, timeout: float | None = None) -> None:
    for c in self.peers.values():
      with contextlib.suppress(Exception): c.close()

def nccl_lib_path() -> str:
  """libnccl: FZ_NCCL_LIB, else the nvidia-nccl-cu12/cu13 wheel (a torch dependency on Linux), else the system's."""
  if p := os.environ.get("FZ_NCCL_LIB"): return p
  import glob, importlib.util
  spec = importlib.util.find_spec("nvidia")
  for root in (spec.submodule_search_locations or []) if spec else []:
    if hits := sorted(glob.glob(os.path.join(root, "**", "libnccl.so*"), recursive=True)): return hits[0]
  import ctypes.util
  return ctypes.util.find_library("nccl") or "libnccl.so.2"

class _UID(ctypes.Structure): _fields_ = [("internal", ctypes.c_byte * 128)]
NCCL_DT = {dtypes.float16: 6, dtypes.float32: 7, dtypes.bfloat16: 9}
_NCCL: list = []

def _nccl() -> Any:
  if not _NCCL:
    # NCCL's graph-mixing support makes every captured all-reduce cost ~78 us instead of ~10 us in our decode graphs
    # (2x H100 A/B, docs/status/tp-m5-h100: TPOT 27.2 -> 22.0 ms with it off); set before libnccl reads its environment
    os.environ.setdefault("NCCL_GRAPH_MIXING_SUPPORT", "0")
    lib = ctypes.CDLL(nccl_lib_path())
    lib.ncclGetErrorString.restype = ctypes.c_char_p
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UID, ctypes.c_int]
    lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.ncclCommDestroy.argtypes = lib.ncclCommAbort.argtypes = [ctypes.c_void_p]
    _NCCL.append(lib)
  return _NCCL[0]

def _nck(lib: Any, ret: int, what: str) -> None:
  if ret != 0: raise RuntimeError(f"NCCL error {ret} in {what}: {lib.ncclGetErrorString(ret).decode()}")

def nccl_unique_id() -> bytes:
  lib, uid = _nccl(), _UID()
  _nck(lib, lib.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
  return bytes(uid.internal)

def nccl_version() -> str:
  lib, v = _nccl(), ctypes.c_int()
  _nck(lib, lib.ncclGetVersion(ctypes.byref(v)), "ncclGetVersion")
  return f"{v.value // 10000}.{v.value // 100 % 100}.{v.value % 100}"

class NcclComm:
  """One NCCL communicator per rank over the device's primary context (made current first; NCCL takes the device from
  it). all_reduce enqueues on dev.stream: in order with the step's kernels, and into the graph while one is captured."""
  name = "nccl"
  def __init__(self, rank: int, world: int, uid: bytes, device: str):
    from fusezap.device import Device
    self.lib, self.rank, self.world, self.dev = _nccl(), rank, world, Device[device]
    self.dev.ctx()
    self.comm, u = ctypes.c_void_p(), _UID()
    ctypes.memmove(ctypes.addressof(u), uid, 128)
    _nck(self.lib, self.lib.ncclCommInitRank(ctypes.byref(self.comm), world, u, rank), "ncclCommInitRank")
  def all_reduce(self, dev: Any, out: Any, inp: Any, n: int, dt: DType) -> None:
    dev.ctx()
    _nck(self.lib, self.lib.ncclAllReduce(inp.handle, out.handle, n, NCCL_DT[dt], 0, self.comm, dev.stream or None), "ncclAllReduce")
  def close(self, timeout: float | None = None) -> None:
    """Release the communicator without waiting for the other ranks (see the module docstring, Shutdown): synchronize
    the device (bounded: a collective whose peer is gone never finishes), then ncclCommAbort (local; it also stops
    such a collective). Each step waits at most `timeout` (FZ_TP_CLOSE_TIMEOUT)."""
    if not self.comm: return
    comm, self.comm = self.comm, ctypes.c_void_p()
    t = CLOSE_TIMEOUT if timeout is None else timeout
    _bounded(self.dev.synchronize, t, "nccl synchronize")
    if os.environ.get("FZ_TP_NCCL_ABORT", "1") == "0": return dbg("ncclCommAbort skipped (FZ_TP_NCCL_ABORT=0)")
    def abort() -> None:
      self.dev.ctx()
      _nck(self.lib, self.lib.ncclCommAbort(comm), "ncclCommAbort")
    _bounded(abort, min(t, ABORT_WAIT), "nccl ncclCommAbort")

@dataclasses.dataclass(eq=False)
class AllReduceCall:
  """Payload (`ast[0]`) of a kind="custom" item: bufs (out, in), the sum over the group's ranks of `in`, written to out.
  fz.jit records it like any custom kernel (jit._run calls launch at capture and at replay)."""
  comm: Any
  @property
  def kernel(self) -> "AllReduceCall": return self
  @property
  def name(self) -> str: return f"fz_all_reduce_{self.comm.name}"
  def launch(self, dev: Any, bufs: Any, var_vals: dict | None = None) -> None:
    out, inp = bufs[0], bufs[1]
    if out.size: self.comm.all_reduce(dev, out.ensure_allocated(), inp.ensure_allocated(), out.size, out.dtype)
  def pretty(self, indent: int = 0) -> str: return f"{'  ' * indent}ALLREDUCE {self.comm.name}"

class TPComm:
  """The group's all-reduce (TPGroup.comm): the custom one-shot kernel (custom_ar.CustomAR, `car`) for the messages it
  takes, the base collective (NCCL / host) for the rest. mode (FZ_TP_AR_MODE): "auto"; "base" (the base collective
  only); "skip" (CUDA, profiling only: out = in, no communication: a step's time without its all-reduces). fuse
  (FZ_TP_AR_FUSE, default 1): Llama.forward_tokens may fuse an all-reduce with the residual add + RMSNorm after it.
  Every rank decides the same way from the same message, so the ranks' collectives always pair up."""
  def __init__(self, base: Any):
    self.base, self.car = base, None
    self.mode = os.environ.get("FZ_TP_AR_MODE", "") or "auto"
    self.fuse = os.environ.get("FZ_TP_AR_FUSE", "1") != "0"
  @property
  def name(self) -> str: return self.base.name
  @property
  def rank(self) -> int: return self.base.rank
  @property
  def world(self) -> int: return self.base.world
  def custom(self, nbytes: int, dt: DType) -> bool: return self.mode == "auto" and self.car is not None and self.car.takes(nbytes, dt)
  def fuse_ok(self, M: int, K: int, dt: DType) -> bool:
    """Up to FZ_TP_AR_FUSE_ROWS rows (64: the decode buckets): the fused kernel runs a row per block on the all-reduce's
    G blocks, slower than the two kernels on prefill-sized row counts."""
    if self.mode != "auto" or not self.fuse or M > int(os.environ.get("FZ_TP_AR_FUSE_ROWS", "") or 64): return False
    if emulate_fuse(): return True   # tests: the fused call's control flow on any device (all_reduce + composed norm)
    return self.car is not None and self.car.norm_takes(M, K, dt)
  def all_reduce(self, dev: Any, out: Any, inp: Any, n: int, dt: DType) -> None:
    if self.mode == "skip":
      if hasattr(dev, "copy_async"): return dev.copy_async(out.handle, inp.handle, n * dt.itemsize)
      host = memoryview(np.empty(n * dt.itemsize, np.uint8))   # off CUDA (script checks): through the host
      inp.copyout(host); out.copyin(host); return None
    if self.custom(n * dt.itemsize, dt): return self.car.all_reduce(dev, out, inp, n, dt)
    return self.base.all_reduce(dev, out, inp, n, dt)
  def close(self, timeout: float | None = None) -> None:
    if self.car is not None:
      with contextlib.suppress(Exception): self.car.close()
    self.base.close(timeout)

@dataclasses.dataclass(eq=False)
class AllReduceNormCall:
  """Payload of a kind="custom" item: bufs (h, xo, p, x, w): xo = x + all_reduce(p), h = w * rmsnorm(xo) (the custom
  all-reduce's fused kernel; bit-identical to all_reduce followed by fusezap.serve.cuda.rmsnorm)."""
  comm: Any
  M: int
  K: int
  eps: float
  @property
  def kernel(self) -> "AllReduceNormCall": return self
  @property
  def name(self) -> str: return "fz_all_reduce_add_norm"
  def launch(self, dev: Any, bufs: Any, var_vals: dict | None = None) -> None:
    h, xo, p, x, w = (b.ensure_allocated() for b in bufs)
    self.comm.car.all_reduce_add_norm(dev, h, xo, p, x, w, self.M, self.K, self.eps, p.dtype)
  def pretty(self, indent: int = 0) -> str: return f"{'  ' * indent}ALLREDUCE+ADD+RMSNORM custom"

def _stamps() -> Any:
  """fusezap.serve.tpprof when FZ_TP_AR_STAMP=1 (GPU clock stamps around every all-reduce), else None."""
  if os.environ.get("FZ_TP_AR_STAMP", "") in ("", "0"): return None
  from fusezap.serve import tpprof
  return tpprof

def emulate_fuse() -> bool: return os.environ.get("FZ_TP_AR_FUSE_EMULATE", "") == "1"

def fuse_ok(g: "TPGroup | None", x: Any) -> bool:
  """all_reduce_add_norm applies to rows like x (N, K) (CUDA, the custom all-reduce on, 16-bit, the message fits)."""
  if g is None or g.world == 1 or not hasattr(g.comm, "fuse_ok") or len(x.shape) != 2 or not all(isinstance(s, int) for s in x.shape): return False
  return g.comm.fuse_ok(int(x.shape[0]), int(x.shape[1]), x.dtype)

def all_reduce_add_norm(p: Any, x: Any, w: Any, eps: float, g: "TPGroup") -> tuple[Any, Any]:
  """(x + all_reduce(p), w * rmsnorm(x + all_reduce(p))) in one kernel: p, x (N, K) 16-bit, w (K,). See fuse_ok."""
  from fusezap.tensor import Tensor
  from fusezap.lazy import LazyBuffer
  from fusezap.device import Buffer
  from fusezap.ir import ScheduleItem
  from fusezap import jit as J
  M, K = int(p.shape[0]), int(p.shape[1])
  assert tuple(x.shape) == (M, K) and tuple(w.shape) == (K,) and p.dtype == x.dtype == w.dtype
  if g.comm.car is None and emulate_fuse():   # tests (FZ_TP_AR_FUSE_EMULATE): Llama._add_norm's composed ops
    xs = x + all_reduce(p, g)
    xf = xs.float()
    return xs, w * (xf * ((xf * xf).mean(-1, keepdim=True) + eps).rsqrt()).cast(x.dtype)
  pb, xb, wb = (t._realized_contiguous().realized for t in (p, x, w))
  h, xo = Buffer(pb.device, M * K, p.dtype), Buffer(pb.device, M * K, p.dtype)
  if (st := _stamps()) is not None: st.stamp(pb, "pre")
  J.run_item(ScheduleItem((AllReduceNormCall(g.comm, M, K, float(eps)),), (h, xo, pb, xb, wb), {}, "custom", 2))   # type: ignore[arg-type]
  if st is not None: st.stamp(xo, "post")
  return Tensor(LazyBuffer.from_buffer(xo, (M, K))), Tensor(LazyBuffer.from_buffer(h, (M, K)))

def all_reduce(x: Any, g: "TPGroup", fuse: bool = False) -> Any:
  """Sum of x over the tensor-parallel group (a new Tensor of x's shape, dtype and device). x is realized first (inside a
  jitted step that realize is recorded too); the shape must be concrete. fuse (I75, NEURON: a prefill's all-reduce under
  FZ_NEURON_TP_PREFILL=fused): record it as a collective the fused kernels may hold (neuron_tp.NeuronComm.call)."""
  from fusezap.tensor import Tensor
  from fusezap.lazy import LazyBuffer
  from fusezap.device import Buffer
  from fusezap.ir import ScheduleItem
  from fusezap.helpers import prod
  from fusezap import jit as J
  assert all(isinstance(s, int) for s in x.shape), f"all_reduce needs a concrete shape, got {x.shape}"
  src = x._realized_contiguous().realized
  out = Buffer(src.device, prod(x.shape), x.dtype)
  if (st := _stamps()) is not None: st.stamp(src, "pre")
  mk = getattr(getattr(g.comm, "base", None), "call", None)   # N8 NEURON: the collective as a device kernel (neuron_tp.NeuronComm)
  call = (mk(prod(x.shape), x.dtype, fuse=True) if fuse else mk(prod(x.shape), x.dtype)) if mk is not None else None
  J.run_item(ScheduleItem((call or AllReduceCall(g.comm),), (out, src), {}, "custom", 1))   # type: ignore[arg-type]
  if st is not None: st.stamp(out, "post")
  return Tensor(LazyBuffer.from_buffer(out, tuple(x.shape)))

# ---------------------------------------------------------------- process group

def rank_device(device: str | None, rank: int) -> str:
  """Rank r's device: CUDA:<base + r> for CUDA (base: the given ordinal; rank 0 keeps the name it was given, so it is the
  same Device instance as the rest of this process uses), the same device name elsewhere."""
  from fusezap.device import Device
  d = Device.canonicalize(device)
  if d.split(":")[0] != "CUDA" or rank == 0: return d
  base = int(d.split(":")[1]) if ":" in d else 0
  return f"CUDA:{base + rank}"

def default_backend(device: str) -> str:
  return os.environ.get("FZ_TP_BACKEND") or ("nccl" if device.split(":")[0] == "CUDA" else "host")

class TPGroup:
  """This process's view of the tensor-parallel group. Rank 0 holds the workers (procs, ctl connections); a worker
  holds its connection to rank 0. depth > 0 while a mirrored call runs (nested model calls are not sent again)."""
  def __init__(self, rank: int, world: int, device: str, backend: str):
    self.rank, self.world, self.device, self.backend = rank, world, device, backend
    self.comm: Any = None
    self.ctl: dict[int, Any] = {}           # rank 0: {worker rank: Connection}; worker: {0: Connection}
    self.procs: dict[int, subprocess.Popen] = {}
    self.depth, self.closing, self.lock = 0, False, threading.Lock()
    self.on_dead: Callable[[int, int | None, str], None] = _exit_on_dead
    self.ring: Any = None                    # the shared-memory control ring (fusezap.serve.shmchan), None: the socket
    self.t_post = (0.0, 0.0)                 # rank 0: monotonic time before / after the last post (tpprof)
    self.t_msg = self.t_recv = 0.0           # a worker: the last message's post time on rank 0 / its receive time here
    self.ar_info = ""                        # how the all-reduces run (printed by `fusezap serve`)
    self.worker: Any = None                  # a worker: its Worker (functions run by op "call" reach the engine)
    self.engine: Any = None                  # rank 0: an LLMEngine registered for tpprof (bench/tp_step.py)
    self._pending: Any = None
    # N8 (NEURON, fusezap.serve.neuron_tp.install): stamp() on rank 0 travels with every message (the collectives this rank
    # has launched); a worker's check(stamp) compares it with its own before running the message, and end_op() runs after
    # every top-level mirrored call on every rank (NEURON: send the queued launches, so no rank's collectives stay queued)
    self.stamp: Callable[[], Any] | None = None
    self.check: Callable[[Any, str], None] | None = None
    self.end_op: Callable[[], None] | None = None
    self.last_stamp: Any = None
  def __repr__(self) -> str: return f"TPGroup(rank {self.rank}/{self.world}, {self.device}, {self.backend})"
  # ---- rank 0
  def post(self, op: str, *args: Any) -> None:
    """Send (op, args) to every worker (rank 0): through the ring when there is one, else each worker's socket."""
    assert self.rank == 0
    with self.lock:
      t0 = time.monotonic()
      st = self.stamp() if self.stamp is not None else None
      if self.ring is not None:
        def big(data: bytes) -> None:
          for r, c in self.ctl.items():
            try: c.send_bytes(data)
            except (OSError, EOFError) as e: raise RuntimeError(f"tensor-parallel worker rank {r} is gone ({e!r})") from None
        def wake(i: int) -> None:
          with contextlib.suppress(OSError, EOFError): self.ctl[i + 1].send_bytes(b"w")
        def alive() -> None:
          if dead := [r for r in self.procs if self._code(r) is not None]: raise RuntimeError(f"tensor-parallel worker rank {dead[0]} is gone")
        self.ring.send((op, args, t0, st), big=big, wake=wake, alive=alive, timeout=TIMEOUT)
      else:
        for r, c in self.ctl.items():
          try: c.send((op, args, t0, st))
          except (OSError, EOFError) as e: raise RuntimeError(f"tensor-parallel worker rank {r} is gone ({e!r})") from None
      self.t_post = (t0, time.monotonic())
  # ---- a worker
  def recv(self) -> tuple[str, tuple, float]:
    """The next message from rank 0 (a worker): (op, args, rank 0's post time). EOFError / OSError: rank 0 is gone."""
    c = self.ctl[0]
    if self.ring is None: m = c.recv()
    else:
      import collections
      if self._pending is None: self._pending = collections.deque()
      pend = self._pending
      def idle(s: float) -> None:   # asleep: the socket carries wake tokens and payloads too big for a slot
        if c.poll(s) and (m := c.recv_bytes()) != b"w": pend.append(m)
      def big() -> bytes:
        if pend: return pend.popleft()
        while (m := c.recv_bytes()) == b"w": pass
        return m
      m = self.ring.recv(big=big, idle=idle)
    self.t_msg, self.t_recv = (m[2] if len(m) > 2 else 0.0), time.monotonic()
    self.last_stamp = m[3] if len(m) > 3 else None
    return m[0], m[1], self.t_msg
  @contextlib.contextmanager
  def spmd(self, op: str, *args: Any):
    """Rank 0: mirror this call on every worker (unless it runs inside another mirrored call); workers: just run it.
    A mirrored call that raises on rank 0 is fatal (on_dead): the workers may be waiting in a collective it never
    reached, and the next call's collectives would pair with those. Validate requests before a mirrored call."""
    top = self.rank == 0 and self.depth == 0
    if top: self.post(op, *args)
    self.depth += 1
    try:
      yield
      if self.depth == 1 and self.end_op is not None: self.end_op()   # N8: the call's queued launches go out now, on every rank
    except BaseException:
      if top: self.on_dead(0, None, f"'{op}' failed on rank 0 after the workers started it:\n{traceback.format_exc()}")
      raise
    finally: self.depth -= 1
  def barrier(self, tag: str, timeout: float | None = None) -> list:
    """Rank 0: wait until every worker has run every message sent so far; returns their replies (rank order). A worker's
    error (its traceback) or exit raises here."""
    self.post("ping", tag)
    lim = TIMEOUT if timeout is None else timeout
    t0 = time.monotonic()
    out, t_end, t_log = [], t0 + lim, t0
    for r, c in sorted(self.ctl.items()):
      while True:
        if time.monotonic() - t_log > 30: t_log = time.monotonic(); dbg(f"waiting for rank {r} at '{tag}': {t_log - t0:.0f} s of {lim:.0f}")
        if c.poll(0.5):
          try: msg = c.recv()
          except (EOFError, OSError): raise RuntimeError(f"tensor-parallel worker rank {r} exited (code {self._code(r)}) before '{tag}'") from None
          if msg[0] == "error": raise RuntimeError(f"tensor-parallel worker rank {r} failed:\n{msg[2]}")
          if msg[0] == "pong" and msg[1] == tag: out.append(msg[2]); break
          continue
        if (code := self._code(r)) is not None and not c.poll(0): raise RuntimeError(f"tensor-parallel worker rank {r} exited (code {code}) before '{tag}'")
        if time.monotonic() > t_end: raise RuntimeError(f"tensor-parallel worker rank {r}: no reply to '{tag}' in {lim:.0f} s")
    return out
  def _code(self, r: int) -> int | None:
    p = self.procs.get(r)
    return p.poll() if p is not None else None
  def close(self, timeout: float | None = None) -> None:
    """Stop the workers (rank 0) and the collective; returns within about 3 x timeout (FZ_TP_CLOSE_TIMEOUT) whatever
    state the workers are in. Rank 0: "quit" to every worker, its own communicator aborted (it does not wait for the
    workers' collectives), the workers waited for, killed if still running, reaped."""
    if self.closing: return
    self.closing = True
    t = CLOSE_TIMEOUT if timeout is None else timeout
    dbg(f"close {self!r}")
    if self.rank == 0 and self.ctl:
      ok = _bounded(lambda: self.post("quit"), t, "post quit")   # a full socket buffer must not block shutdown
      if not ok: dbg("quit not sent; the workers will be killed")
    if self.comm is not None: self.comm.close(t)
    if self.rank == 0:
      t_end = time.monotonic() + t
      for r, p in self.procs.items():
        with contextlib.suppress(Exception): p.wait(max(0.1, t_end - time.monotonic()))
        if p.poll() is None:
          dbg(f"worker rank {r} (pid {p.pid}) still running after {t:.0f} s: kill")
          with contextlib.suppress(Exception): p.kill()
          with contextlib.suppress(Exception): p.wait(5)
        dbg(f"worker rank {r} exit code {p.poll()}")
    for c in self.ctl.values():
      with contextlib.suppress(Exception): c.close()
    if self.ring is not None:
      with contextlib.suppress(Exception): self.ring.unlink()
    dbg("close done")
  def _watch(self) -> None:
    """Rank 0's watchdog: a worker that exits while the group is open ends this process (on_dead)."""
    while not self.closing:
      for r, p in self.procs.items():
        if (code := p.poll()) is not None and not self.closing:
          msg = ""
          with contextlib.suppress(Exception):
            c = self.ctl[r]
            while c.poll(0):
              m = c.recv()
              if m[0] == "error": msg = m[2]
          self.on_dead(r, code, msg)
          return
      time.sleep(0.5)

def _exit_on_dead(rank: int, code: int | None, msg: str) -> None:
  """Default on_dead: the ranks are out of step (a worker exited, or a mirrored call failed on rank 0): stop."""
  what = f"worker rank {rank} exited (code {code})" if rank else "rank 0 cannot continue"
  print(f"fusezap: tensor-parallel {what}; stopping (a step cannot complete without every rank){chr(10) + msg if msg else ''}",
        file=sys.stderr, flush=True)
  os._exit(1)

def launch(world: int, device: str | None = None, backend: str | None = None, timeout: float = 300.0,
           env: dict[str, str] | None = None, on_dead: Callable[[int, int | None, str], None] | None = None) -> TPGroup:
  """Rank 0: spawn ranks 1..world-1 (`python -m fusezap.serve.dist`, this interpreter, inherited environment), connect
  the control (and, for the host backend, data) connections, then initialize the collective on every rank."""
  from multiprocessing.connection import Listener
  dev0 = rank_device(device, 0)
  backend = backend or default_backend(dev0)
  assert backend in ("host", "nccl"), f"tensor-parallel backend {backend!r}: host or nccl"
  g = TPGroup(0, world, dev0, backend)
  if on_dead is not None: g.on_dead = on_dead
  if world == 1:
    g.comm = TPComm(HostComm(0, 1, {}))
    return g
  key = secrets.token_bytes(16)
  ls = Listener(("127.0.0.1", 0), authkey=key)
  base = {**os.environ, **(env or {}), "FZ_TP_ADDR": f"{ls.address[0]}:{ls.address[1]}", "FZ_TP_AUTH": key.hex(),
          "FZ_TP_WORLD": str(world), "FZ_TP_COMM": backend}
  base.pop("FZ_STEP_PROF", None)
  from fusezap.serve import neuron_tp as NT
  renv = NT.rank_envs(dev0, world) if NT.is_neuron_tier(dev0) else [{} for _ in range(world)]   # N8: a NeuronCore per rank
  if renv[0]: NT.apply_rank0_env(renv[0])
  if base.get("FZ_TP_SHM", "1") != "0":
    from fusezap.serve.shmchan import ShmRing
    try:
      g.ring = ShmRing.create(world - 1)
      base["FZ_TP_SHM_PATH"] = g.ring.path
    except OSError as e: dbg(f"no shared-memory control ring ({e!r}): the socket carries every message")
  for r in range(1, world):
    g.procs[r] = subprocess.Popen([sys.executable, "-m", "fusezap.serve.dist"], env={**base, **renv[r], "FZ_TP_RANK": str(r)})
  need, conns = (world - 1) * (2 if backend == "host" else 1), {}
  got: list = []
  def acc() -> None:
    with contextlib.suppress(OSError):
      while len(got) < need: got.append(ls.accept())
  th = threading.Thread(target=acc, daemon=True)
  th.start()
  t_end = time.monotonic() + timeout
  try:
    while th.is_alive():
      th.join(0.2)
      if dead := [r for r, p in g.procs.items() if p.poll() is not None]: raise RuntimeError(f"tensor-parallel worker rank {dead[0]} exited during start-up")
      if time.monotonic() > t_end: raise RuntimeError(f"tensor-parallel workers did not connect in {timeout:.0f} s")
    for c in got:
      kind, r = c.recv()
      conns[(kind, r)] = c
    g.ctl = {r: conns[("ctl", r)] for r in range(1, world)}
    if backend == "nccl":
      uid = nccl_unique_id()
      for r in range(1, world): g.ctl[r].send(("init", backend, rank_device(device, r), uid))
      threading.Thread(target=g._watch, name="fusezap-tp-watchdog", daemon=True).start()
      g.comm = TPComm(NcclComm(0, world, uid, dev0))
    else:
      for r in range(1, world): g.ctl[r].send(("init", backend, rank_device(device, r), None))
      hc = HostComm(0, world, {r: conns[("data", r)] for r in range(1, world)})
      g.comm = TPComm(NT.NeuronComm(hc, dev0) if NT.is_neuron_tier(dev0) else hc)
      threading.Thread(target=g._watch, name="fusezap-tp-watchdog", daemon=True).start()
    g.barrier("init", timeout)
    if g.ring is not None: g.ring.unlink()   # every worker has it mapped: no file is left behind, whatever happens next
    if NT.is_neuron_tier(dev0) and backend == "host": NT.install(g)   # N8: flush every call's launches; desync check
    g.ar_info = NT.setup(g, timeout) if NT.is_neuron_tier(dev0) and backend == "host" else _setup_custom_ar(g, timeout)
  except BaseException:
    g.closing = True
    for p in g.procs.values():
      with contextlib.suppress(Exception): p.kill()
    if g.ring is not None: g.ring.unlink()
    raise
  finally: ls.close()
  import atexit
  atexit.register(g.close)
  return g

def _setup_custom_ar(g: TPGroup, timeout: float) -> str:
  """Rank 0: set up fusezap.serve.custom_ar on every rank (NCCL backend, FZ_TP_CUSTOM_AR, peer access between every pair
  of devices): each rank allocates its buffer, the IPC handles go round, each rank opens the others'; it is enabled only
  if every rank succeeded, on every rank at once. Returns a line saying how the all-reduces run."""
  from fusezap.serve import custom_ar as CA
  if g.backend != "nccl": return f"{g.backend} all-reduce"
  if not CA.enabled_env(): return "NCCL all-reduce (custom all-reduce off: FZ_TP_CUSTOM_AR=0 or NCCL_P2P_DISABLE=1)"
  if g.world > CA.MAX_W: return f"NCCL all-reduce (custom all-reduce: at most {CA.MAX_W} ranks)"
  ords = [CA.ordinal(rank_device(g.device, r)) for r in range(g.world)]
  try: p2p = all(CA.can_access_peer(a, b) for i, a in enumerate(ords) for b in ords[i + 1:])
  except Exception as e: return f"NCCL all-reduce (peer access check failed: {e!r})"
  if not p2p: return "NCCL all-reduce (no peer access between the GPUs: custom all-reduce off)"
  from fusezap.device import Device
  def fail(why: str) -> str:
    g.post("car_enable", False)
    g.barrier("car_enable", timeout)
    return f"NCCL all-reduce (custom all-reduce setup failed: {why})"
  g.post("car_init")
  mine: Any = None
  try:
    mine = CA.CustomAR(Device[g.device], 0, g.world)
    h0: Any = mine.handle()
  except Exception as e: h0 = f"rank 0: {e!r}"
  hs = [h0] + g.barrier("car_init", timeout)
  if bad := [h for h in hs if not isinstance(h, bytes)]: return fail(str(bad[0]))
  g.post("car_open", hs)
  ok0: Any = True
  try: mine.open(hs)
  except Exception as e: ok0 = f"rank 0: {e!r}"
  oks = [ok0] + g.barrier("car_open", timeout)
  if bad := [o for o in oks if o is not True]: return fail(str(bad[0]))
  g.post("car_enable", True)
  g.comm.car = mine
  g.barrier("car_enable", timeout)
  return f"custom all-reduce up to {mine.slot >> 20} MB ({mine.G} blocks, CUDA IPC + peer access), NCCL above"

# ---------------------------------------------------------------- self-tests (run on every rank: g.spmd("call", ...))

def _rank_data(dt: str, n: int, rank: int) -> np.ndarray:
  return (np.random.default_rng(1234 + rank).standard_normal(n) * (rank + 1)).astype(np.float32)

def selftest_allreduce(g: TPGroup, dt: str, n: int) -> dict:
  """Every rank: all_reduce of a rank-seeded tensor (dtype dt, n elements) on the group's device, through fz.jit.run_item
  like a model step. Then each rank's result is gathered (a second all-reduce of a zero-padded float32 array: x + 0 is
  exact), so rank 0 checks the sum and that every rank holds the same bits. Returns {ok, max_err, same, n, dtype}."""
  from fusezap.tensor import Tensor
  dtype = getattr(dtypes, dt)
  xs = [_rank_data(dt, n, r) for r in range(g.world)]
  y = all_reduce(Tensor(xs[g.rank], device=g.device).cast(dtype), g).float().numpy()
  pad = np.zeros((g.world, n), np.float32)
  pad[g.rank] = y
  allv = all_reduce(Tensor(pad, device=g.device), g).numpy()
  ref = np.zeros(n, np.float32)
  for x in xs: ref += _to_f32(_from_f32(x, dtype), dtype)   # each rank's input as rounded to dtype, summed in rank order
  ref = _to_f32(_from_f32(ref, dtype), dtype)
  tol = {"float32": 1e-6, "float16": 2e-3, "bfloat16": 1.6e-2}[dt] * (np.abs(ref).max() + 1)
  err = float(np.abs(y - ref).max())
  same = bool(all(np.array_equal(allv[r], allv[0]) for r in range(g.world)))
  exact = bool(np.array_equal(y, ref))
  return {"ok": err <= tol and same, "max_err": err, "exact": exact, "same": same, "n": n, "dtype": dt, "backend": g.backend}

def bench_allreduce(g: TPGroup, dt: str, n: int, iters: int = 50, mode: str = "auto", chain: bool = False) -> dict:
  """Every rank: time `iters` all-reduces of n elements inside one fz.jit step (replayed: a CUDA graph on CUDA), each
  followed by a * 0.5 kernel (chain=True: all-reduces back to back). mode: TPComm.mode while the step is captured
  ("base": NCCL). Returns {us, GBps (algorithm bandwidth: bytes / time), path (custom / the base collective)}."""
  from fusezap.tensor import Tensor
  from fusezap.jit import Jit
  from fusezap.device import Device
  dtype = getattr(dtypes, dt)
  def step(x: Tensor) -> Tensor:
    for _ in range(iters): x = all_reduce(x, g) if chain else all_reduce(x, g) * 0.5
    return x
  j = Jit(step, plan=True)
  x = Tensor(_rank_data(dt, n, g.rank) * (0.01 if chain else 1.0), device=g.device).cast(dtype).realize()
  dev = Device[g.device]
  old, g.comm.mode = g.comm.mode, mode
  try:
    path = "custom" if g.comm.custom(n * dtype.itemsize, dtype) else g.comm.name if mode != "skip" else "skip"
    for _ in range(3): j(x).realize()
  finally: g.comm.mode = old
  dev.synchronize()
  t0 = time.perf_counter()
  for _ in range(5): j(x).realize()
  dev.synchronize()
  t = (time.perf_counter() - t0) / (5 * iters)
  return {"us": t * 1e6, "GBps": n * dtype.itemsize / t / 1e9, "n": n, "dtype": dt, "backend": g.backend, "path": path, "chain": chain}

def selftest_custom(g: TPGroup, rows: Any = (1, 8, 32, 64, 256, 512), K: int = 8192, eps: float = 1e-5) -> dict:
  """Every rank (CUDA, custom all-reduce on): per row count, a bf16 (rows, K) all-reduce through the custom kernel vs
  NCCL's (TPComm.mode "base"): the same bits, eager and inside a replayed fz.jit step (CUDA graph, 4 replays on new
  inputs, two all-reduces per step); every rank holds the same bits (gathered by an f32 all-reduce of zero-padded
  copies); up to 64 rows all_reduce_add_norm equals all_reduce + cuda.rmsnorm bit for bit (eager and replayed).
  Returns {ok, rows: [...]}; custom_ar.check() raises if a barrier timed out (FZ_TP_AR_SPIN_S)."""
  from fusezap.tensor import Tensor
  from fusezap.jit import Jit
  from fusezap.serve import cuda as _cu
  c, bf = g.comm, dtypes.bfloat16
  rng = np.random.default_rng(4321 + g.rank)
  rnd = lambda *s, sc=1.0: Tensor((rng.standard_normal(s) * sc).astype(np.float32), device=g.device).cast(bf).realize()
  bits = lambda t: t.float().numpy()
  def base(fn: Callable[[], Any]) -> Any:
    old, c.mode = c.mode, "base"
    try: return fn()
    finally: c.mode = old
  def same_on_ranks(a: np.ndarray) -> bool:
    pad = np.zeros((g.world,) + a.shape, np.float32)
    pad[g.rank] = a
    allv = base(lambda: all_reduce(Tensor(pad, device=g.device), g).numpy())
    return bool(all(np.array_equal(allv[r], allv[0]) for r in range(g.world)))
  out, ok = [], True
  for M in rows:
    x = rnd(M, K, sc=g.rank + 1.0)
    took = c.custom(M * K * 2, bf)
    y1, y0 = bits(all_reduce(x, g)), bits(base(lambda: all_reduce(x, g)))
    r = {"rows": M, "custom": took, "eq_nccl": bool(np.array_equal(y1, y0)), "eq_ranks": same_on_ranks(y1)}
    j = Jit(lambda t: all_reduce(all_reduce(t, g) * 0.5, g), plan=True)
    eq = []
    for k in range(6):
      xk = rnd(M, K, sc=1.0 + k)
      yk = bits(j(xk))
      rk = bits(base(lambda: all_reduce(all_reduce(xk, g) * 0.5, g)))
      if k >= 2: eq.append(bool(np.array_equal(yk, rk)))
    r["eq_graph"] = all(eq)
    if c.fuse_ok(M, K, bf):
      p, xr, w = rnd(M, K, sc=0.5), rnd(M, K), rnd(K, sc=0.2)
      xo, h = all_reduce_add_norm(p, xr, w, eps, g)
      xo2, h2 = _cu.rmsnorm(xr, w, eps, all_reduce(p, g))
      r["fused_eq"] = bool(np.array_equal(bits(xo), bits(xo2)) and np.array_equal(bits(h), bits(h2)))
      jf = Jit(lambda a, b, ww: all_reduce_add_norm(a, b, ww, eps, g)[1], plan=True)
      feq = []
      for k in range(6):
        pk, xk = rnd(M, K, sc=0.5 + k), rnd(M, K)
        hk = bits(jf(pk, xk, w))
        if k >= 2: feq.append(bool(np.array_equal(hk, bits(_cu.rmsnorm(xk, w, eps, all_reduce(pk, g))[1]))))
      r["fused_graph_eq"] = all(feq)
    ok = ok and all(v for k, v in r.items() if k.startswith("eq") or k.startswith("fused"))
    out.append(r)
  if c.car is not None: c.car.check()
  return {"ok": ok, "rank": g.rank, "car": c.car is not None, "rows": out}

# ---------------------------------------------------------------- the worker process

class Worker:
  """A worker's state: the group, its model shard and engine. An op's return value is sent back with the next "ping"
  reply (TPGroup.barrier returns them)."""
  def __init__(self, g: TPGroup): self.g, self.model, self.eng, self.car, self.reply = g, None, None, None, None
  def op_load(self, spec: dict) -> None:
    from fusezap.serve.models import load_model
    from fusezap.dtype import dtypes as D
    if (sh := spec.get("random")) is not None:   # bench/tp_step.py --random: this rank's slices of a random model (same seed)
      from fusezap.serve.models import random_weights
      self.model = random_weights.build(sh, self.g.device, getattr(D, spec["dtype"]), spec["max_len"], spec.get("prefill_chunk", 512),
                                        tp=self.g, seed=spec.get("seed", 0), layers=spec.get("layers", 0))
      return
    self.model = load_model(spec["model"], dtype=getattr(D, spec["dtype"]), device=self.g.device, max_len=spec["max_len"],
                            prefill_chunk=spec.get("prefill_chunk", 512), tp=self.g, **spec.get("kw", {}))
  def op_engine(self, kw: dict, sig: tuple) -> None:
    from fusezap.serve.engine import LLMEngine
    self.eng = LLMEngine(self.model, **kw)
    if (mine := self.eng.tp_signature()) != sig:
      raise RuntimeError(f"rank {self.g.rank}'s engine differs from rank 0's: {mine} != {sig}")
  def op_fwd(self, *a: Any) -> None: self.eng.mirror_forward(*a)
  def op_prefill(self, ids: Any, start: int, chunk: int | None) -> None: self.model.prefill(ids, start, chunk)
  def op_decode(self, ids: Any, start: int) -> None: self.model.decode(ids, start)
  def op_reset(self) -> None: self.model.reset()
  def op_score(self, ids: list[int], start: int, k: int) -> None:
    from fusezap.serve import logprobs as LP
    LP.scorer(self.eng).score(ids, start, k)
  def op_release(self) -> None:
    from fusezap.serve.api import release_cache
    release_cache(self.eng)
  def op_call(self, fn: str, *args: Any) -> Any:
    """A module-level function `module:name`, called as fn(group, *args) (tests and scripts: e.g. selftest_allreduce)."""
    import importlib
    mod, name = fn.split(":")
    return getattr(importlib.import_module(mod), name)(self.g, *args)
  def op_car_init(self) -> Any:
    from fusezap.serve import custom_ar as CA
    from fusezap.device import Device
    try:
      self.car = CA.CustomAR(Device[self.g.device], self.g.rank, self.g.world)
      return self.car.handle()
    except Exception as e: return f"rank {self.g.rank}: {e!r}"
  def op_car_open(self, hs: list) -> Any:
    try:
      self.car.open(hs)
      return True
    except Exception as e: return f"rank {self.g.rank}: {e!r}"
  def op_car_enable(self, on: bool) -> None:
    if on: self.g.comm.car = self.car

def _parent_watch(ppid: int) -> None:
  while True:
    if os.getppid() != ppid: os._exit(1)   # rank 0 is gone: nothing will ever pair with this rank's collectives
    time.sleep(1.0)

def _worker_exit(g: TPGroup, code: int) -> None:
  """Leave the worker: abort the communicator (bounded), flush, os._exit (no interpreter teardown: CUDA / NCCL library
  destructors or a thread stuck in a driver call cannot hold the process)."""
  g.close()
  dbg(f"exit {code}")
  for f in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception): f.flush()
  os._exit(code)

def worker_main() -> int:
  """`python -m fusezap.serve.dist` (spawned by launch): connect to rank 0, initialize the collective, then run rank 0's
  messages in order until "quit" or the connection closes. Any error is sent to rank 0 and ends the process."""
  from multiprocessing.connection import Client
  rank, world, backend = int(os.environ["FZ_TP_RANK"]), int(os.environ["FZ_TP_WORLD"]), os.environ["FZ_TP_COMM"]
  host, port = os.environ["FZ_TP_ADDR"].rsplit(":", 1)
  key = bytes.fromhex(os.environ["FZ_TP_AUTH"])
  threading.Thread(target=_parent_watch, args=(os.getppid(),), daemon=True).start()
  import signal
  signal.signal(signal.SIGINT, signal.SIG_IGN)   # Ctrl-C goes to rank 0, which stops the workers
  ring = None
  if path := os.environ.get("FZ_TP_SHM_PATH"):
    from fusezap.serve.shmchan import ShmRing
    ring = ShmRing.attach(path, rank - 1)   # before connecting: rank 0 unlinks the file once every worker is in
  ctl = Client((host, int(port)), authkey=key)
  ctl.send(("ctl", rank))
  data = None
  if backend == "host":
    data = Client((host, int(port)), authkey=key)
    data.send(("data", rank))
  try:
    _, bk, device, uid = ctl.recv()
    g = TPGroup(rank, world, device, bk)
    g.ctl, g.ring = {0: ctl}, ring
    if bk == "nccl": g.comm = TPComm(NcclComm(rank, world, uid, device))
    else:
      from fusezap.serve import neuron_tp as NT
      hc = HostComm(rank, world, {0: data})
      g.comm = TPComm(NT.NeuronComm(hc, device) if NT.is_neuron_tier(device) else hc)   # N8: NEURON's all-reduce modes
      if NT.is_neuron_tier(device): NT.install(g)
    g.worker = w = Worker(g)
    while True:
      try: op, args, _ = g.recv()
      except (EOFError, OSError): dbg("control connection closed"); _worker_exit(g, 0)   # rank 0 is gone or closed the group
      if op == "quit": dbg("quit"); _worker_exit(g, 0)
      if op == "ping":
        ctl.send(("pong", args[0], w.reply))
        w.reply = None
        continue
      if g.check is not None: g.check(g.last_stamp, op)   # N8: a rank out of step with rank 0 fails here, not in a collective
      with g.spmd(op): w.reply = getattr(w, "op_" + op)(*args)
  except BaseException:
    tb = traceback.format_exc()
    print(f"fusezap: tensor-parallel worker rank {rank} failed:\n{tb}", file=sys.stderr, flush=True)
    with contextlib.suppress(Exception): ctl.send(("error", rank, tb))
    os._exit(1)

if __name__ == "__main__": sys.exit(worker_main())
