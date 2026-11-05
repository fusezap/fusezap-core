"""Tensor parallelism on NEURON (I72, docs/NKI_BACKEND.md §11): Qwen2.5-7B at TP=2 over the two NeuronCores of one
Inferentia2, the shape AWS NxD Inference runs it in.

The structure is fusezap.serve.dist's (I57, the CUDA TP): one fusezap process per rank (rank 0 spawns rank 1 with
`python -m fusezap.serve.dist`), each with its own NEURON device and its own runtime worker (nrt_worker.py), which opens
only that rank's NeuronCore (NEURON_RT_VISIBLE_CORES=<core of the rank>: one runtime process sees one core's 16 GiB of
HBM, run 1's HBM probe). Llama(tp=g) loads the rank's Megatron slices (dist.tp_slice: q/k/v and gate/up by rows,
o_proj/down by columns), bf16 K-major per shard (the relayout runs after the slice), the KV pool holds the rank's kv heads
(Hkv/W), and rank 0 mirrors every forward (dist.TPGroup.spmd), so both ranks launch the same kernels in the same order.

The all-reduce (two per layer: after o_proj and after down_proj) is NeuronComm, the group's base collective, in one of
three modes (FZ_NEURON_TP_AR; `auto`, the default, self-tests them in this order at launch and keeps the first that
passes on every rank):
  cc    an NKI collective kernel (nki_kernels.AllReduceSpec: nki.collectives.all_reduce, compiled into the NEFF by
        neuronx-cc), loaded by each rank's runtime worker with nrt_load_collectives (nrtpy load_from_neff(cc_enabled=True,
        rank_id, world_size); the runtime bootstraps the ranks through NEURON_RT_ROOT_COMM_ID). It is a launch like any
        other: it stays inside the step's batch of launches, so a decode step is still one runtime-worker round trip per
        rank, and the cores exchange the partial sums themselves. On NKISHIM the shim's collective calls the group's host
        all-reduce (COLLECTIVE hook), so the whole path runs on the Mac.
  shm   the runtime workers exchange the partial sums through a shared-memory file (nrt_worker.ShmAR): each reads its
        partial from the device, posts it, waits for the peer's, sums in rank order (float32, the same bits on every
        rank), writes the sum back. Also inside the batch (an "ar" request), no fusezap process involved; costs two
        host<->device copies of the message and a spin wait.
  host  dist.HostComm through the fusezap processes (each all-reduce flushes the batch: a read, a socket exchange, a
        write). The fallback that works anywhere.
FZ_NEURON_TP_AR_KIND picks AllReduceSpec's form (phbm / hbm / sbuf); FZ_NEURON_TP_AR_FUSE=1 lets fuse_schedule put the
collective inside a fused layer kernel (one NEFF per layer group with the collective in it).
FZ_NEURON_TP_LAYER=fused (I72 follow-up; the default since its inf2 run): each shard decode layer is one LayerSpec(part="tp")
kernel with both all-reduces inside it (when the collective passes the launch self-test; else the halves), and under TP
FZ_NEURON_FUSE defaults to every layer with the head fused: one NEFF per rank and step (fused_layer, neuron.fuse_layers).
FZ_NEURON_TP_PREFILL=fused (I75, default): a prefill forward's layers (rows beyond LayerSpec: the member kernels) record
their all-reduces as AllReduceSpec collectives that fuse_schedule may put inside the fused kernels, and the forward is
grouped FZ_NEURON_TP_PREFILL_FUSE layers per NEFF (default `all` since the I75 inf2 run: the whole chunk and the head in
one NEFF per rank; a group that fails to compile runs as groups of 4 layers, neuron.FUSE_FALLBACK):
the same instructions as the members with separate cc launches (so the same bits), without one launch (~0.33 ms with a
collective in it) or one shm exchange (~1.2 ms at 128 rows of 7B, inside the step) per all-reduce. Needs the collective's
self-test to pass, else the members with the separate all-reduce mode. `plan_mix` / NeuronComm.mix state the whole choice.
"""
from __future__ import annotations
import atexit, contextlib, os, secrets, socket, sys, tempfile, time
from typing import Any
import numpy as np
from fusezap.dtype import DType, dtypes

MODES = ("cc", "shm", "host")
SHM_SLOT = int(os.environ.get("FZ_NEURON_TP_SHM_SLOT", "") or 8 << 20)   # bytes per rank and parity: the largest shm message

def tier(device: str) -> str: return device.split(":")[0].upper()
def is_neuron_tier(device: str) -> bool: return tier(device) in ("NEURON", "NKISHIM", "NKISIM")

def _free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]

def _cores(world: int) -> list[str]:
  """Rank r's NeuronCore: the r-th entry of NEURON_RT_VISIBLE_CORES when it lists >= world cores (e.g. "0,1" or "0-1"),
  else FZ_NEURON_TP_CORE0 + r (default 0 + r)."""
  vis = os.environ.get("NEURON_RT_VISIBLE_CORES", "").strip()
  got: list[str] = []
  for part in (p.strip() for p in vis.split(",") if p.strip()):
    if "-" in part:
      a, b = part.split("-", 1)
      got += [str(i) for i in range(int(a), int(b) + 1)]
    else: got.append(part)
  if len(got) >= world: return got[:world]
  c0 = int(os.environ.get("FZ_NEURON_TP_CORE0", "") or 0)
  return [str(c0 + r) for r in range(world)]

def shm_create(world: int, slot: int = SHM_SLOT) -> str:
  """A fresh zeroed shared-memory file for nrt_worker.ShmAR (rank 0, before any worker starts: no stale sequence numbers)."""
  from fusezap.device.nrt_worker import ShmAR
  d = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
  path = os.path.join(d, f"fz_tp_{os.getpid()}_{secrets.token_hex(4)}")
  with open(path, "wb") as f: f.truncate(ShmAR.size(world, slot))
  def rm() -> None:
    with contextlib.suppress(OSError): os.unlink(path)
  atexit.register(rm)
  return path

def rank_envs(device: str, world: int) -> list[dict[str, str]]:
  """Per-rank environment for a NEURON-tier group (rank 0 applies its own to os.environ before its runtime worker starts):
  the rank's NeuronCore, the collectives' bootstrap address, the rank / world the runtime worker loads collective NEFFs
  with, and the shm all-reduce's file."""
  cores = _cores(world)
  root = os.environ.get("NEURON_RT_ROOT_COMM_ID") or f"127.0.0.1:{_free_port()}"
  shm = shm_create(world)
  out = []
  for r in range(world):
    e = {"FZ_NEURON_CC_RANK": str(r), "FZ_NEURON_CC_WORLD": str(world), "NEURON_RT_ROOT_COMM_ID": root, "FZ_NEURON_TP_SHM": shm,
         "FZ_NEURON_TP_SLOT": str(SHM_SLOT)}
    if tier(device) == "NEURON":
      e["NEURON_RT_VISIBLE_CORES"] = cores[r]
      # a collective execute waits inside nrt for its peer, which may still be compiling a NEFF on a cold cache; the
      # runtime's execute timeout defaults to 30 s (nrt-configurable-parameters: NEURON_RT_EXEC_TIMEOUT)
      e["NEURON_RT_EXEC_TIMEOUT"] = os.environ.get("NEURON_RT_EXEC_TIMEOUT") or "600"
    out.append(e)
  return out

def apply_rank0_env(env: dict[str, str]) -> None:
  """Rank 0's environment, before its NEURON device opens the runtime worker (which inherits it). A worker started
  earlier (e.g. the device was used before the group launched) keeps the cores it opened: say so."""
  from fusezap.device.neuron import NrtWorker
  inst = NrtWorker.__dict__.get("_inst")
  if inst is not None and inst.proc.poll() is None and os.environ.get("NEURON_RT_VISIBLE_CORES") != env.get("NEURON_RT_VISIBLE_CORES", os.environ.get("NEURON_RT_VISIBLE_CORES")):
    print("fusezap: NEURON runtime worker started before the tensor-parallel group: rank 0 keeps its cores "
          f"({os.environ.get('NEURON_RT_VISIBLE_CORES', 'all')})", file=sys.stderr, flush=True)
  os.environ.update(env)

def default_mode(device: str) -> str:
  m = (os.environ.get("FZ_NEURON_TP_AR", "") or "auto").strip().lower()
  if m not in MODES + ("auto",): raise ValueError(f"FZ_NEURON_TP_AR={m!r}: one of auto, {', '.join(MODES)}")
  return m

class NeuronComm:
  """The NEURON group's base collective (dist.TPComm(NeuronComm(HostComm))): mode cc / shm / host (module docstring).
  `call(n, dtype)` gives dist.all_reduce the item to record (cc: a NeuronCall of AllReduceSpec, a NEFF launch like any
  kernel, so fz.jit replays it through NeffReplay); `all_reduce` runs it for an AllReduceCall item (shm: one queued "ar"
  request to the runtime worker; host: HostComm)."""
  name = "neuron"
  def __init__(self, base: Any, device: str, mode: str = "host"):
    self.base, self.device, self.mode = base, device, mode
    self.t = tier(device)
    self.layer_cc = False   # the decode layers' all-reduces inside the layer kernels (LayerSpec part "tp", set by setup)
    self.prefill_cc = False # I75: a prefill's all-reduces as collectives inside its fused member kernels (set by setup)
    self.cc_ok = False      # I75: the collective loaded and passed its self-test on every rank (large shm messages take it)
    self.n = 0   # collectives this rank ran outside the runtime worker's queue (shim hook, host), for count()
    if self.t in ("NKISHIM",):   # the shim's collective: the group's host all-reduce (float32, rank order)
      from fusezap.device import nkishim
      def peer(a: np.ndarray, op: str, groups: list) -> np.ndarray:
        if op != "add": raise NotImplementedError(f"NKISHIM tensor-parallel collective {op!r}: add only")
        if [sorted(g) for g in groups] != [list(range(self.world))]: raise NotImplementedError(f"NKISHIM collective groups {groups}: one group of all ranks")
        self.n += 1
        return self.base.reduce_host(np.ascontiguousarray(a, np.float32).reshape(-1), dtypes.float32).reshape(a.shape)
      self._prev = list(nkishim.COLLECTIVE)   # one hook per process: the newest group's (tests open a second group)
      nkishim.COLLECTIVE[:] = [peer]
      self._hook = peer
  @property
  def rank(self) -> int: return self.base.rank
  @property
  def world(self) -> int: return self.base.world
  def usable(self, mode: str) -> bool:
    """Whether this rank's device can run mode at all (cc: NEURON or NKISHIM; nki.simulate has no collectives. shm: the
    persistent NEURON path's runtime worker)."""
    if mode == "cc": return self.t in ("NEURON", "NKISHIM")
    if mode == "shm": return self.t == "NEURON" and self._alloc() is not None
    return True
  def _alloc(self) -> Any:
    from fusezap.device import Device
    from fusezap.device.neuron import NeuronAllocator
    a = getattr(Device[self.device], "allocator", None)
    return a if isinstance(a, NeuronAllocator) else None
  def spec(self, n: int) -> Any:
    from fusezap.codegen.nki_kernels import AllReduceSpec, ar_kind
    return AllReduceSpec(int(n), self.world, ar_kind())
  def call(self, n: int, dt: DType, fuse: bool = False) -> Any:
    """The item dist.all_reduce records: an AllReduceSpec launch (mode cc), or None (the AllReduceCall of shm / host).
    fuse (I75, a prefill under FZ_NEURON_TP_PREFILL=fused): the collective, phbm form (LayerSpec part "tp"'s), marked
    fusable, whatever the separate mode is, when the collective passed its launch self-test (prefill_cc)."""
    if dt != dtypes.float32 or self.world == 1: return None
    from fusezap.device.neuron import NeuronCall
    if fuse and self.prefill_cc:
      from fusezap.codegen.nki_kernels import AllReduceSpec
      return NeuronCall(AllReduceSpec(int(n), self.world, "phbm", fuse=True))
    if self.mode_for(n, dt) != "cc": return None
    return NeuronCall(self.spec(n))
  def mode_for(self, n: int, dt: DType) -> str:
    """I75: the mode of one separate all-reduce of n elements. The setup's mode, except in shm mode for a large float32
    message when the collective passed its self-test (cc_ok): from FZ_NEURON_TP_AR_BIG bytes on (default 256 KiB), and
    always above the shm slot (FZ_NEURON_TP_SHM_SLOT, 8 MiB, which ShmAR refuses), the cc launch (inf2: shm 1.67 ms at 1 MiB
    against cc's flat ~0.35 ms; shm wins below ~200 KiB). Without cc a message above the slot goes through the host. Every
    rank decides from the same n, so the ranks' collectives pair up."""
    m = self.mode
    if m != "shm" or dt != dtypes.float32: return m
    nb, big = int(n) * 4, ar_big()
    if self.cc_ok and ((big and nb >= big) or nb > SHM_SLOT): return "cc"
    return "host" if nb > SHM_SLOT else m
  def mix(self) -> dict:
    """I75: the all-reduce plan in effect, as a record: the separate all-reduces' mode (and from which size cc takes over),
    where the decode layers' and the prefill's collectives run (inside their kernels, or the separate mode around the
    halves / members)."""
    sep = self.mode
    if sep == "shm": sep += f", cc from {ar_big()} bytes" if self.cc_ok and ar_big() else (f", cc above {SHM_SLOT} bytes" if self.cc_ok else f", host above {SHM_SLOT} bytes")
    return {"separate": sep, "decode_layers": "cc in the layer kernels" if self.layer_cc else f"halves around {self.mode} all-reduces",
            "prefill": f"cc in the fused prefill kernels ({prefill_fuse_name()} layers per NEFF)" if self.prefill_cc else f"members + {self.mode} all-reduces"}
  def all_reduce(self, dev: Any, out: Any, inp: Any, n: int, dt: DType) -> None:
    m = self.mode_for(n, dt)
    if m == "cc" and dt == dtypes.float32 and self.world > 1:
      from fusezap.ir import ScheduleItem
      from fusezap.device.neuron import NeuronCall
      dev.run(ScheduleItem((NeuronCall(self.spec(n)),), (out, inp), {}, "custom", 1))
      return
    if m == "shm" and dt == dtypes.float32 and (al := self._alloc()) is not None:
      al.call({"op": "ar", "in": inp.handle.id, "out": out.handle.id, "n": n * 4})
      return
    self.n += 1
    self.base.all_reduce(dev, out, inp, n, dt)
  def count(self) -> int:
    """Collectives this rank has launched so far: the runtime worker's queued / sent collective execs and "ar" requests
    (NeuronAllocator.cc_count), plus those run here. Every rank's count is the same between two mirrored calls."""
    al = self._alloc() if self.t == "NEURON" else None
    return self.n + (al.cc_count if al is not None else 0)
  def flush(self) -> None:
    """Send this rank's queued launches to its runtime worker. N8 run 1 hang: a worker rank never reads results, so its
    last launches of a step (collectives among them) stayed queued in NeuronAllocator.pending while rank 0's read sent its
    own: rank 0's runtime waited in the collective for a peer that never launched (CCOM "Timeout waiting for RX")."""
    if (al := self._alloc()) is not None: al.flush()
  def reduce_host(self, a: np.ndarray, dt: DType) -> np.ndarray: return self.base.reduce_host(a, dt)
  def close(self, timeout: float | None = None) -> None:
    if (h := getattr(self, "_hook", None)) is not None:
      from fusezap.device import nkishim
      if nkishim.COLLECTIVE[:1] == [h]: nkishim.COLLECTIVE[:] = self._prev   # the previous group's hook again
    from fusezap.device import neuron as _nd
    _nd.TP_WORLD[0] = 1   # TP=1 defaults again (neuron.fuse_layers)
    self.base.close(timeout)

# ---------------------------------------------------------------- the mode, chosen at launch (every rank the same)

def _comm(g: Any) -> NeuronComm:
  c = getattr(g.comm, "base", g.comm)
  assert isinstance(c, NeuronComm), f"not a NEURON group: {g.comm!r}"
  return c

def set_mode(g: Any, mode: str, layer_cc: bool = False, prefill_cc: bool = False, cc_ok: bool | None = None) -> None:
  c = _comm(g)
  c.mode, c.layer_cc, c.prefill_cc = mode, bool(layer_cc), bool(prefill_cc)
  c.cc_ok = bool(layer_cc or prefill_cc or mode == "cc") if cc_ok is None else bool(cc_ok)

def ar_big() -> int:
  """FZ_NEURON_TP_AR_BIG: bytes from which a separate all-reduce in shm mode takes the cc launch instead when cc passed its
  self-test (default 262144 = 256 KiB: a prefill / scoring row block of 7B is 14 KiB per row; 0 = never, except above the
  shm slot). I75: the default mix measured shm at 14 KiB only, and shm grows ~1.6 ms per MiB (I72 run 1)."""
  v = os.environ.get("FZ_NEURON_TP_AR_BIG", "").strip()
  return int(v) if v else 256 * 1024

def plan_mix(cost: dict, layer: str | None = None, prefill: str | None = None) -> dict:
  """I75: the group's all-reduce plan from the modes that loaded and passed their self-test on every rank (cost: mode -> ms
  per all-reduce inside a step, None for host; empty: nothing passed, host). The separate all-reduces take the cheapest
  measured mode (shm on inf2: 0.05 ms at one 7B row vs a cc launch's 0.38 ms); the decode layers (FZ_NEURON_TP_LAYER=fused)
  and the prefill (FZ_NEURON_TP_PREFILL=fused) put their collectives inside their kernels whenever cc passed, whatever a
  separate cc launch costs, and large separate messages in shm mode take cc launches (NeuronComm.mode_for: from
  FZ_NEURON_TP_AR_BIG bytes, and above the shm slot). So the default on inf2 is a mix: shm for small separate all-reduces
  (the vocab head's gather), cc launches for large ones (a members prefill, the logprobs scorer's trunk), cc inside the
  decode layer kernels and inside the prefill's fused kernels. -> {separate, layer_cc, prefill_cc, cc_ok, cost}."""
  cost = dict(cost) or {"host": None}
  mode = min(cost, key=lambda m: float("inf") if cost[m] is None else cost[m])
  layer, prefill = layer or tp_layer(), prefill or tp_prefill()
  cc = "cc" in cost
  return {"separate": mode, "layer_cc": layer == "fused" and cc, "prefill_cc": prefill == "fused" and cc, "cc_ok": cc, "cost": cost}

class TPDesync(RuntimeError):
  """A rank's collective count differs from rank 0's between two mirrored calls: its next collective would pair with the
  wrong peer launch (or wait forever in the runtime). Raised on the worker before it runs the call; rank 0 stops."""

def install(g: Any) -> None:
  """N8 guards on a NEURON group (dist.launch / worker_main): every top-level mirrored call ends by sending the rank's
  queued launches (g.end_op: the hang fix), and rank 0's collective count travels with each message (g.stamp), which a
  worker checks against its own before running it (g.check: a desync fails fast, naming the call, instead of hanging in
  a collective). FZ_NEURON_TP_CHECK=0 turns the check off."""
  c = _comm(g)
  g.end_op = c.flush
  from fusezap.device import neuron as _nd
  _nd.TP_WORLD[0] = max(int(g.world), 1)   # FZ_NEURON_FUSE / _FUSE_MAX defaults under TP (neuron.fuse_layers)
  if os.environ.get("FZ_NEURON_TP_CHECK", "1") in ("", "0"): return
  if g.rank == 0: g.stamp = c.count
  else:
    def check(stamp: Any, op: str) -> None:
      if stamp is None: return
      if (mine := c.count()) != stamp:
        raise TPDesync(f"tensor-parallel desync before '{op}': rank 0 has launched {stamp} collectives, rank {g.rank} {mine}")
    g.check = check

def _data(rank: int, n: int) -> np.ndarray: return (np.random.default_rng(4321 + rank).standard_normal(n) * (rank + 1)).astype(np.float32)

def prepare(g: Any, mode: str, n: int) -> Any:
  """Every rank (op "call"): whether mode can run here, and for cc the collective NEFF compiled and loaded (load only:
  a rank whose load fails must not leave its peer inside a collective execute). True or an error string."""
  c = _comm(g)
  try:
    if not c.usable(mode): return f"rank {g.rank}: {mode} not available on {c.device}"
    if mode == "cc" and c.t == "NEURON":
      from fusezap.device import Device
      from fusezap.device.neuron import NeuronNeffProgram
      dev = Device[c.device]
      prg = dev.spec_program(c.spec(n))
      if isinstance(prg, NeuronNeffProgram): prg.prepare(dev.allocator, ((n, "<f4"), (n, "<f4")))
    return True
  except Exception as e: return f"rank {g.rank}: {type(e).__name__}: {str(e)[:2000]}"

def selftest(g: Any, mode: str, n: int, iters: int = 17) -> dict:
  """Every rank (op "call"): `iters` all-reduces of rank-seeded data in mode through dist.all_reduce (as a model step
  records it), against the float32 rank-order sum; then the cost of one all-reduce inside a step (a chain of `iters` minus a
  chain of one, over iters - 1: the device copies, the exchange and the launch, without the read's round trip).
  {ok, err, ms, mode} or {ok False, error}."""
  from fusezap.tensor import Tensor
  from fusezap.serve import dist
  c = _comm(g)
  old, c.mode = c.mode, mode
  try:
    want = _data(0, n).copy()
    for r in range(1, g.world): want += _data(r, n)
    x = Tensor(_data(g.rank, n), device=g.device).realize()
    y = dist.all_reduce(x, g).numpy()
    err = float(np.abs(y - want).max())
    def chain(k: int) -> float:   # k all-reduces in a row (as in a step: queued together), then one read
      t0 = time.perf_counter()
      z = x
      for _ in range(k): z = dist.all_reduce(z, g)
      z.numpy()
      return time.perf_counter() - t0
    chain(1); chain(iters)
    one = min(chain(1) for _ in range(3))
    many = min(chain(iters) for _ in range(3))
    ms = 1e3 * max(many - one, 0.0) / (iters - 1)   # per all-reduce, the read's round trip taken out
    return {"ok": bool(err <= 1e-5 * max(1.0, float(np.abs(want).max()))), "err": err, "ms": ms, "mode": mode}
  except Exception as e: return {"ok": False, "error": f"rank {g.rank}: {type(e).__name__}: {str(e)[:2000]}", "mode": mode}
  finally: c.mode = old

def setup(g: Any, timeout: float) -> str:
  """Rank 0 after the group's init barrier: pick the all-reduce mode and set it on every rank; returns the line `fusezap
  serve` prints. FZ_NEURON_TP_AR=cc|shm|host: that mode (its load and self-test must pass). auto (default): every mode
  whose load and self-test pass on every rank, then the one with the lowest measured cost per all-reduce inside a step
  (N8 run 1: the collective cost 0.36 ms per execute at any size, the shm exchange 0.04 ms at 14 KiB including the device
  copies; "the first that passes" had picked the collective). host is the fallback when nothing else passes."""
  want = default_mode(g.device)
  order = [want] if want != "auto" else ["shm", "cc"]
  fused = tp_layer() == "fused"   # the layer kernels hold the collectives: cc if it works, whatever a separate launch costs
  n, notes, cost = 3584, [], {}
  for mode in order:
    if mode == "host": cost["host"] = None; continue
    g.post("call", "fusezap.serve.neuron_tp:prepare", mode, n)
    oks = [prepare(g, mode, n)] + g.barrier("prepare", timeout)
    if bad := [o for o in oks if o is not True]:
      notes.append(f"{mode}: {bad[0]}")
      if want != "auto": raise RuntimeError(f"FZ_NEURON_TP_AR={mode}: {bad[0]}")
      continue
    g.post("call", "fusezap.serve.neuron_tp:selftest", mode, n)
    res = [selftest(g, mode, n)] + g.barrier("selftest", timeout)
    if bad := [r for r in res if not r.get("ok")]:
      notes.append(f"{mode}: {bad[0]}")
      if want != "auto": raise RuntimeError(f"FZ_NEURON_TP_AR={mode} self-test failed: {bad[0]}")
      continue
    cost[mode] = max(r.get("ms", 0.0) for r in res)
  # the decode layers' collectives (FZ_NEURON_TP_LAYER=fused, the TP default since the I72 follow-up run) and the prefill's
  # (FZ_NEURON_TP_PREFILL=fused, I75): inside the kernels whenever the NKI collective loaded and passed its self-test on every
  # rank; the separate all-reduces (the vocab head, a members prefill) keep the cheapest measured mode (shm on inf2).
  # No cc: the layers run as halves, the prefill as the members, both around the separate mode (plan_mix).
  plan = plan_mix(cost)
  mode, layer_cc, prefill_cc, cc_ok = plan["separate"], plan["layer_cc"], plan["prefill_cc"], plan["cc_ok"]
  if fused and not layer_cc: notes.append("fused layers need the NKI collective (cc): the layers run as halves")
  if tp_prefill() == "fused" and not prefill_cc: notes.append("the fused prefill needs the NKI collective (cc): the prefill runs as the members")
  g.post("call", "fusezap.serve.neuron_tp:set_mode", mode, layer_cc, prefill_cc, cc_ok)
  set_mode(g, mode, layer_cc, prefill_cc, cc_ok)
  g.barrier("set_mode", timeout)
  from fusezap.codegen.nki_kernels import ar_kind
  what = {"cc": f"NKI collective all-reduce (AllReduceSpec {ar_kind()}, nrt_load_collectives)",
          "shm": "all-reduce through shared memory in the runtime workers", "host": "host all-reduce (dist.HostComm)"}[mode]
  meas = ", ".join(f"{m} {c:.3f} ms" for m, c in cost.items() if c is not None)
  if layer_cc: what += "; decode layers: one kernel each with its two NKI collectives inside (FZ_NEURON_TP_LAYER=fused)"
  elif fused or tp_layer() == "halves": what += "; decode layers: LayerSpec halves around the all-reduces"
  if prefill_cc: what += f"; prefill: the collectives inside its fused kernels, {prefill_fuse_name()} layers per NEFF (FZ_NEURON_TP_PREFILL=fused)"
  else: what += f"; prefill: the members around the {mode} all-reduces"
  return what + (f"; measured per all-reduce of {n * 4 // 1024} KiB in a step: {meas}" if meas else "") + (f" (not usable: {'; '.join(notes)})" if notes else "")

# ---------------------------------------------------------------- N8 I72 follow-up: the all-reduces inside the layer kernel

TP_LAYERS = ("halves", "members", "fused")

def tp_layer() -> str:
  """FZ_NEURON_TP_LAYER: how a shard runs a decoder layer. fused (default since the I72 follow-up inf2 run: 7B TP=2 served
  38.4 ms/token c1, 93.9 tok/s c4 vs halves + shm 58.3 / 62.4, AWS NxDI 40.6 / 81.7): one LayerSpec(part="tp") kernel per
  layer with both all-reduces inside it as nki.collectives, when the collective passed its launch self-test (setup), else
  the halves; halves: LayerSpec's attn and mlp halves around two all-reduce launches (run 2's path); members: the members."""
  m = (os.environ.get("FZ_NEURON_TP_LAYER", "") or "fused").strip().lower()
  if m not in TP_LAYERS: raise ValueError(f"FZ_NEURON_TP_LAYER={m!r}: one of {', '.join(TP_LAYERS)}")
  return m

def fused_layer(g: Any) -> int:
  """The world for LayerSpec(part="tp") when this group runs fused layers (FZ_NEURON_TP_LAYER=fused, mode cc, float32
  activations), else 0 (the halves). Why: in the halves path each all-reduce ends a launch, and on the persistent path every
  launch is a synchronous nrt execute (N8 TP run: 57 launches per token, executes 50.7 of the 58.5 ms, the halves 1.59 ms
  per layer vs 1.04 ms device for the whole layer); inside one kernel the device runs attn | collective | mlp | collective
  back to back and FZ_NEURON_FUSE can put many layers (and the head) in one NEFF per rank and token, NxD Inference's shape."""
  if g is None or getattr(g, "world", 1) < 2 or tp_layer() != "fused": return 0
  c = getattr(g.comm, "base", g.comm)
  return int(g.world) if isinstance(c, NeuronComm) and c.layer_cc and c.world > 1 else 0

# ---------------------------------------------------------------- I75: the prefill's all-reduces inside its fused kernels

TP_PREFILLS = ("fused", "members")
PREFILL_FUSE = "all"   # I75 inf2 run: one NEFF per rank and chunk (7B TP=2 c1 TTFT 87 ms vs 95 at 4 layers per NEFF)

def tp_prefill() -> str:
  """FZ_NEURON_TP_PREFILL: how a shard runs a prefill forward (rows beyond LayerSpec, so its layers are the member kernels).
  fused (default): the all-reduces are collectives inside the fused member kernels (AllReduceSpec(fuse=True)), grouped
  FZ_NEURON_TP_PREFILL_FUSE layers per NEFF, when the collective passed its launch self-test (setup), else the members;
  members: the member kernels around separate all-reduces in the separate mode (shm on inf2): the path before I75."""
  m = (os.environ.get("FZ_NEURON_TP_PREFILL", "") or "fused").strip().lower()
  if m not in TP_PREFILLS: raise ValueError(f"FZ_NEURON_TP_PREFILL={m!r}: one of {', '.join(TP_PREFILLS)}")
  return m

def prefill_fuse() -> int:
  """FZ_NEURON_TP_PREFILL_FUSE: decoder layers per fused prefill kernel (default `all` since the I75 inf2 run; `all` or 0:
  every layer and, with FZ_NEURON_HEAD_FUSE, the head: one NEFF per rank and chunk, ~204 s to compile at 7B, 0.85 GiB of
  planned intermediates per pool prefix, since fz.jit's plan cannot share memory inside one kernel). A number bounds the
  NEFF a prefill compiles (at 128 rows a 7B shard layer's members are ~11k instructions). A group of more than
  neuron.FUSE_FALLBACK (4) layers whose NEFF fails to compile runs as groups of 4 (FusedCall.subs, logged)."""
  v = (os.environ.get("FZ_NEURON_TP_PREFILL_FUSE", "") or str(PREFILL_FUSE)).strip().lower()
  from fusezap.device.neuron import FUSE_ALL
  if v in ("all", "0"): return FUSE_ALL
  if not v.isdigit(): raise ValueError(f"FZ_NEURON_TP_PREFILL_FUSE={v!r}: a layer count >= 1, or all")
  return int(v)

def prefill_fuse_name() -> str:
  from fusezap.device.neuron import FUSE_ALL
  k = prefill_fuse()
  return "all" if k >= FUSE_ALL else str(k)

def fused_prefill(g: Any) -> int:
  """Layers per prefill NEFF when this group runs a prefill with its collectives inside the fused kernels
  (FZ_NEURON_TP_PREFILL=fused and the collective passed its self-test: NeuronComm.prefill_cc), else 0 (the members with
  separate all-reduces)."""
  if g is None or getattr(g, "world", 1) < 2 or tp_prefill() != "fused": return 0
  c = getattr(g.comm, "base", g.comm)
  return prefill_fuse() if isinstance(c, NeuronComm) and c.prefill_cc and c.world > 1 else 0

def mix(g: Any) -> dict:
  """The group's all-reduce plan on this rank (NeuronComm.mix); every rank reports the same."""
  return _comm(g).mix()

# ---------------------------------------------------------------- vocab-parallel lm_head (FZ_NEURON_TP_HEAD=vocab)

def vocab_head() -> bool:
  """FZ_NEURON_TP_HEAD=vocab: each rank keeps V/W rows of lm_head and the logits are all-gathered (a zero-padded sum
  all-reduce: exact), so a core reads half of lm_head per step; default `replicated` (every rank the whole lm_head, as on CUDA)."""
  return (os.environ.get("FZ_NEURON_TP_HEAD", "") or "replicated").strip().lower() == "vocab"

def gather_cols(y: Any, g: Any, V: int) -> Any:
  """y (..., V/W) this rank's logits columns [rank V/W, (rank+1) V/W) -> (..., V) on every rank: zero-padded to V, then
  summed over the ranks (each column has exactly one nonzero term, so the sum is the gather, bit for bit)."""
  from fusezap.serve.dist import all_reduce
  Vl = int(y.shape[-1])
  assert Vl * g.world == V, f"vocab {V} over {g.world} ranks: {Vl} per rank"
  pad = tuple([None] * (len(y.shape) - 1) + [(g.rank * Vl, (g.world - 1 - g.rank) * Vl)])
  return all_reduce(y.pad(pad).contiguous(), g)
