"""NEURON persistent runtime worker (N4b): runs inside .venv-nrt (numpy + nrtpy 2.35), never in fusezap's own env (the nrtpy
and nki wheels conflict, docs/NKI_BACKEND.md §4 exception 5). Started by fusezap.device.neuron as
`<.venv-nrt python> nrt_worker.py`, one per fusezap process, and kept for the process's lifetime. It is the only process that
opens the NeuronCore. The .venv-nki worker only compiles.

Why it exists: run 1 (inf2.xlarge, 2026-10-26) measured ~1.31 s per nki.jit call at any size, because nki 0.6 recompiles and
reloads the NEFF on every standalone call. Here each NEFF is loaded once (`load`, ~9 ms for a small NEFF in run 1) and executed
any number of times on device tensors that live in this process (`alloc` by id, `write`/`read` only at the host boundary).

nrtpy API used (read from the nrtpy 2.35.30 wheel; run 1 confirmed load, tensor alloc, write and read on inf2):
  NrtpyModel.load_from_neff(path) -> .model_ref, .input_tensors_info / .output_tensors_info ({name: info with .shape .dtype
  .size}), .alias_info ({output name: input name}, from the NEFF's must_alias metadata);
  NrtpyTensor.from_numpy(a, name=) (device alloc + write), .write_from_numpy(a), .numpy(), .tensor_ref;
  get_nrtpy_singleton().slice_from_tensor(ref, name=) (an aliasing view of the same device memory, which nrtpy's own model
  call builds for each must-alias output) and .execute(model_ref, inputs={name: ref}, outputs={name: ref}).
Run 1's NRT_INVALID "Failed to execute model" came from leaving out the must-alias input `data0.must_alias_input`. Every
declared input is bound here, and each must-alias output is a slice of its input's tensor.

Protocol: the same pickle-5 frames as nkisim_worker.py. Every request may carry "new": {id: nbytes} (allocate these zeroed
device buffers first) and "free": [id] (release these). Ids are never reused. Requests:
  {"op": "version"}                                  -> {"ok", "nrtpy", "numpy"}
  {"op": "load", "key", "path"}                      -> {"ok", "inputs", "outputs", "alias", "seconds"}  (cached by key)
  {"op": "write", "id", "data"}                      -> {"ok"}                       host -> device (whole buffer)
  {"op": "read", "id", "n"}                          -> {"ok", "data"}               device -> host (first n bytes)
  {"op": "exec", "key", "in": {name: id}, "out": {name: id}, "alias": {out: in}, "iters": 1}
                                                     -> {"ok", "seconds": [per execute]}
  {"op": "bench", "key", "in", "out", "alias", "iters", "mode"} -> {"ok", "stats"}  (nrtpy's own benchmark, device or host)
  {"op": "batch", "reqs": [exec requests]}           -> {"ok", "seconds", "n", "wall"}  (N5: one round trip for a jit replay;
                                                        seconds: each execute's wall time, wall: the whole batch's, N6;
                                                        N7: writes and a last read may be in reqs: "data" is the read's)
  {"op": "stats"}                                    -> {"ok", "buffers", "bytes", "models", "execs", "loads"}
  {"op": "ar", "in", "out", "n"}                     -> {"ok", "seconds": []}  (N8 tensor parallelism, FZ_NEURON_TP_AR=shm:
                                                        the first n bytes of buffer `in` (float32) summed over the ranks
                                                        through ShmAR, written to `out`; queued in batches like exec)
Every reply also carries "t": the worker's seconds on the request (N7 profiling: the rest of the round trip is the pipe).
N8: a "load" may carry "cc": {"rank", "world"}: the NEFF holds collectives (nki.collectives), so it is loaded with
nrt_load_collectives (nrtpy load_from_neff(cc_enabled=True, rank_id, world_size); the runtime bootstraps the ranks through
NEURON_RT_ROOT_COMM_ID, which fusezap.serve.neuron_tp sets for every rank's worker).
N8 (I71): a request's new buffers (a batch's: all of them) are allocated before any of it runs; if the runtime refuses one
(NRT_RESOURCE) the ones made are freed and the reply is {"ok": False, "oom": True, "err"}: nothing ran. I96: except a batch
whose launches carry frees, which goes in request order (frees, allocations, run) so its footprint is the host's, not the sum
of every buffer it allocates; a refusal there names the launch and how many ran. "err" always ends with the worker's live bytes.
"""
import mmap, os, pickle, struct, sys, time, traceback

def _send(f, obj) -> None:
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

class ShmAR:
  """N8 (FZ_NEURON_TP_AR=shm): the ranks' runtime workers all-reduce through one shared-memory file (fusezap.serve.neuron_tp
  creates it zeroed; FZ_NEURON_TP_SHM, FZ_NEURON_CC_RANK / _WORLD, FZ_NEURON_TP_SLOT). Layout: a 64-byte header of one uint64
  sequence number per rank, then per rank two message slots (by the call's parity). Call k: write the message to slot
  (rank, k % 2), publish seq[rank] = k, wait until every rank's seq >= k, sum the slots in rank order in float32 (the same
  bits on every rank). A rank overwrites slot (rank, k % 2) again at call k + 2, which it enters only after every rank
  published k + 1, i.e. finished reading call k: two slots need no acknowledgements."""
  HDR = 64
  @staticmethod
  def size(world: int, slot: int) -> int: return ShmAR.HDR + world * 2 * slot
  def __init__(self, path: str, rank: int, world: int, slot: int, timeout: float = 600.0):
    import numpy as np
    assert 1 <= world <= ShmAR.HDR // 8 and 0 <= rank < world
    self.rank, self.world, self.slot, self.timeout, self.k = rank, world, slot, timeout, 0
    fd = os.open(path, os.O_RDWR)
    try:
      if os.fstat(fd).st_size < ShmAR.size(world, slot): raise ValueError(f"{path}: {os.fstat(fd).st_size} bytes, need {ShmAR.size(world, slot)}")
      self.mm = mmap.mmap(fd, ShmAR.size(world, slot))
    finally: os.close(fd)
    self.seq = np.frombuffer(self.mm, np.uint64, count=world, offset=0)
    self.data = np.frombuffer(self.mm, np.uint8, offset=ShmAR.HDR)
  def _slot(self, r: int, par: int, nb: int):
    o = (r * 2 + par) * self.slot
    return self.data[o:o + nb]
  def reduce(self, a):
    """a: this rank's float32 message -> the float32 sum over the ranks (a new array)."""
    import numpy as np
    a = np.ascontiguousarray(a, np.float32).reshape(-1)
    nb = a.nbytes
    if nb > self.slot: raise ValueError(f"shm all-reduce of {nb} bytes > the {self.slot}-byte slot (FZ_NEURON_TP_SHM_SLOT)")
    self.k += 1
    k, par = self.k, self.k & 1
    self._slot(self.rank, par, nb)[:] = a.view(np.uint8)
    self.seq[self.rank] = k
    end, spins = time.monotonic() + self.timeout, 0
    for r in range(self.world):
      while int(self.seq[r]) < k:
        spins += 1
        if spins & 1023 == 0:
          if time.monotonic() > end: raise TimeoutError(f"shm all-reduce {k}: rank {r} did not post in {self.timeout:.0f} s")
          if spins > 200_000: time.sleep(0)   # a long wait (the peer is in a kernel or compiling): yield the CPU
    acc = self._slot(0, par, nb).view(np.float32).copy()
    for r in range(1, self.world): acc += self._slot(r, par, nb).view(np.float32)
    return acc

_SHM: list = []
def _shm_ar() -> ShmAR:
  if not _SHM:
    e = os.environ
    _SHM.append(ShmAR(e["FZ_NEURON_TP_SHM"], int(e.get("FZ_NEURON_CC_RANK", "0")), int(e.get("FZ_NEURON_CC_WORLD", "1")),
                      int(e.get("FZ_NEURON_TP_SLOT", "") or 8 << 20), float(e.get("FZ_TP_TIMEOUT", "") or 600.0)))
  return _SHM[0]

class State:
  def __init__(self):
    self.bufs: dict = {}      # id -> NrtpyTensor (uint8, nbytes)
    self.models: dict = {}    # key -> (NrtpyModel, info)
    self.refs: dict = {}      # (key, in, out) -> (input refs, output refs, keep-alive slices, ids)
    self.execs = self.loads = 0

S = State()

def _info(d) -> dict:
  out = {}
  for k, v in dict(d).items():
    out[str(k)] = {"shape": [int(x) for x in getattr(v, "shape", ())], "dtype": str(getattr(v, "dtype", "")), "size": int(getattr(v, "size", 0))}
  return out

def _alloc(i: int, n: int) -> None:
  import numpy as np
  from nrtpy import NrtpyTensor
  if i not in S.bufs: S.bufs[i] = NrtpyTensor.from_numpy(np.zeros(max(int(n), 1), np.uint8), name=f"fz{i}")

def _free(ids) -> None:
  dead = set(ids)
  for k in [k for k, v in S.refs.items() if dead & v[3]]: del S.refs[k]   # slices first: they view the tensors
  for i in ids: S.bufs.pop(i, None)

def _ref(i: int, want: int, name: str, keep: list):
  """The device tensor of buffer i for a NEFF tensor of `want` bytes: the tensor itself, or a leading slice of a larger one."""
  from nrtpy import get_nrtpy_singleton
  t = S.bufs[i]
  have = int(t.shape[0]) if t.shape else 0
  if want and have < want: raise ValueError(f"buffer {i} has {have} bytes, the NEFF tensor {name!r} needs {want}")
  if not want or have == want: return t.tensor_ref
  r = get_nrtpy_singleton().slice_from_tensor(t.tensor_ref, offset=0, size=want, name=name)
  keep.append(r)
  return r

def _bind(key: str, ins: dict, outs: dict, alias: dict):
  """(input refs, output refs) for one binding of a loaded model, built once and cached: every declared input is bound,
  and a must-alias output is a slice of its input's tensor (nrtpy_model._make_alias_tensor does the same per call)."""
  from nrtpy import get_nrtpy_singleton
  ck = (key, tuple(sorted(ins.items())), tuple(sorted(outs.items())), tuple(sorted(alias.items())))
  if (hit := S.refs.get(ck)) is not None: return hit
  _, info = S.models[key]
  missing = [n for n in info["inputs"] if n not in ins]
  if missing: raise KeyError(f"NEFF inputs not bound: {missing} (declared {list(info['inputs'])})")
  keep: list = []
  iref = {n: _ref(i, info["inputs"][n]["size"], n, keep) for n, i in ins.items()}
  oref = {}
  for n in info["outputs"]:
    if n in alias:
      oref[n] = get_nrtpy_singleton().slice_from_tensor(iref[alias[n]], name=n)
      keep.append(oref[n])
    elif n in outs: oref[n] = _ref(outs[n], info["outputs"][n]["size"], n, keep)
    else: raise KeyError(f"NEFF output {n!r} not bound")
  hit = S.refs[ck] = (iref, oref, keep, set(ins.values()) | set(outs.values()))
  return hit

class OOM(Exception):
  """N8: the runtime refused a request's device allocations (NRT_RESOURCE); the ones it made were released again."""

def _alloc_all(reqs: list) -> None:
  """Allocate every new buffer the requests carry before any of them runs (N8, I71: run 1's 7B scoring pass failed at launch
  3 of a batch of 4, after launches 0..2 had run). On NRT_RESOURCE the buffers allocated here are freed again and OOM is
  raised, so a refused batch leaves the device as it was."""
  made = []
  try:
    for r in reqs:
      for i, n in (r.get("new") or {}).items():
        if i not in S.bufs: _alloc(i, n); made.append(i)
  except Exception as e:
    if "NRT_RESOURCE" not in str(e) and "allocate" not in str(e).lower(): raise
    for i in made: S.bufs.pop(i, None)
    raise OOM(f"{type(e).__name__}: {e}; the worker holds {live_bytes() / 2**30:.3f} GiB in {len(S.bufs)} buffers") from e

def live_bytes() -> int:
  """Device bytes of the buffers this worker holds (what the host's NeuronAllocator.held should equal, cache included)."""
  return int(sum(int(t.shape[0]) for t in S.bufs.values()))

def handle(req: dict) -> dict:
  import numpy as np
  op = req.get("op")
  # A batch whose launches carry frees is allocated and freed in request order (host order: a request's frees, then its
  # allocations); allocating every "new" up front kept all the buffers the host had since freed live at once (I96 W7c: a
  # 512^2 klein forward, one batch, 14.7 GB of tensors on a 7.4 GiB host account). A batch without frees (the same
  # footprint either way) keeps the all-or-nothing allocation: a refused batch ran nothing.
  seq = op == "batch" and any(r.get("free") for r in req.get("reqs") or [])
  if req.get("free"): _free(req["free"])
  if not seq: _alloc_all([req] + list(req.get("reqs") or []) if op == "batch" else [req])
  if op == "version":
    from importlib.metadata import version
    try: v = version("nrtpy")
    except Exception: v = "?"
    return {"ok": True, "nrtpy": v, "numpy": np.__version__}
  if op == "load":
    from nrtpy import NrtpyModel
    if req["key"] not in S.models:
      st = time.perf_counter()
      cc = req.get("cc")   # N8: a NEFF with collectives, loaded for this rank of the group
      m = NrtpyModel.load_from_neff(req["path"], **({"cc_enabled": True, "rank_id": int(cc["rank"]), "world_size": int(cc["world"])} if cc else {}))
      info = {"inputs": _info(m.input_tensors_info), "outputs": _info(m.output_tensors_info),
              "alias": {str(k): str(v) for k, v in dict(getattr(m, "alias_info", {}) or {}).items()}, "seconds": time.perf_counter() - st}
      S.models[req["key"]] = (m, info)
      S.loads += 1
    return {"ok": True, **S.models[req["key"]][1]}
  if op == "write":
    a = np.frombuffer(req["data"], np.uint8)
    t = S.bufs[req["id"]]
    if a.nbytes != t.shape[0]: a = np.concatenate([a, t.numpy()[a.nbytes:]]) if a.nbytes < t.shape[0] else a[:t.shape[0]]
    t.write_from_numpy(a)
    return {"ok": True}
  if op == "read":
    a = S.bufs[req["id"]].numpy().reshape(-1).view(np.uint8)
    return {"ok": True, "data": a[:req.get("n", a.nbytes)]}
  if op == "exec":
    from nrtpy import get_nrtpy_singleton
    m, _ = S.models[req["key"]]
    iref, oref, _, _ = _bind(req["key"], req["in"], req["out"], req.get("alias", {}))
    rt, ts = get_nrtpy_singleton(), []
    for _ in range(int(req.get("iters", 1))):
      st = time.perf_counter()
      rt.execute(m.model_ref, iref, oref)
      ts.append(time.perf_counter() - st)
    S.execs += len(ts)
    return {"ok": True, "seconds": ts}
  if op == "bench":
    from nrtpy import NrtpyTensor
    m, info = S.models[req["key"]]
    ins, outs, tmp = dict(req["in"]), dict(req["out"]), []
    for d, sec in ((ins, "inputs"), (outs, "outputs")):   # N6: a buffer freed since (a step's output): zeros of the NEFF tensor's size
      for n, i in d.items():
        if i not in S.bufs:
          j = -(10 ** 9 + len(S.refs) * 1000 + len(tmp))
          _alloc(j, info[sec][n]["size"]); d[n] = j; tmp.append(j)
    try:
      iref, oref, _, _ = _bind(req["key"], ins, outs, req.get("alias", {}))
      wrap = lambda d: {n: NrtpyTensor(r, (1,), np.uint8, name=n) for n, r in d.items()}
      st = m.benchmark(inputs=wrap(iref), outputs=wrap(oref), warmup_iter=10, benchmark_iter=int(req.get("iters", 100)), mode=req.get("mode", "device"))
    finally: _free(tmp)
    stats = {a: getattr(st, a) for a in ("mean_ms", "min_ms", "max_ms", "std_dev_ms", "iterations") if hasattr(st, a)}
    if getattr(st, "durations_ms", None): stats["median_ms"] = float(np.median(st.durations_ms))
    return {"ok": True, "stats": stats}
  if op == "batch":   # N5: queued launches in order, one round trip (each carries its own new/free)
    secs, t0, data = [], time.perf_counter(), None
    for k, r in enumerate(req["reqs"]):
      try: rep = handle(r)
      except OOM as e: raise OOM(f"launch {k} of a batch of {len(req['reqs'])} ({r.get('key')}; {k} ran): {e}") from e
      except Exception as e: raise RuntimeError(f"launch {k} of a batch of {len(req['reqs'])} ({r.get('key')}): {type(e).__name__}: {e}") from e
      if not rep.get("ok"): return {**rep, "err": f"launch {k} of a batch of {len(req['reqs'])}: {rep.get('err')}"}
      secs += rep.get("seconds", [])
      if "data" in rep: data = rep["data"]   # N7: queued writes and launches, then the step's read
    out = {"ok": True, "seconds": secs, "n": len(req["reqs"]), "wall": time.perf_counter() - t0}
    if data is not None: out["data"] = data
    return out
  if op == "ar":   # N8: the shm all-reduce of a device buffer (FZ_NEURON_TP_AR=shm)
    n, t = int(req["n"]), S.bufs[req["in"]]
    st = time.perf_counter()
    a = t.numpy().reshape(-1).view(np.uint8)[:n].view(np.float32)
    r = _shm_ar().reduce(a).view(np.uint8)
    o = S.bufs[req["out"]]
    have = int(o.shape[0])
    if r.nbytes < have: r = np.concatenate([r, o.numpy().reshape(-1).view(np.uint8)[r.nbytes:]])
    o.write_from_numpy(r[:have])
    return {"ok": True, "seconds": [], "ar_s": time.perf_counter() - st}
  if op == "stats":
    return {"ok": True, "buffers": len(S.bufs), "bytes": live_bytes(), "models": len(S.models),
            "execs": S.execs, "loads": S.loads}
  return {"ok": False, "err": f"unknown op {op!r}", "tb": ""}

def main() -> None:
  fin, fout = os.fdopen(os.dup(0), "rb"), os.fdopen(os.dup(1), "wb")
  os.dup2(2, 1); sys.stdout = sys.stderr   # libnrt / nrtpy logging must not corrupt the protocol
  while (req := _recv(fin)) is not None:
    st = time.perf_counter()
    try: rep = handle(req)
    except BaseException as e:
      if isinstance(e, KeyboardInterrupt): raise
      rep = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:4000]}", "tb": traceback.format_exc()[-6000:]}
      if isinstance(e, OOM): rep["oom"] = True   # N8: nothing of the request ran (the host raises NeuronOOM, a MemoryError)
    rep["t"] = time.perf_counter() - st   # N7: the worker's time on this request (the rest of a round trip is the pipe)
    _send(fout, rep)

if __name__ == "__main__": main()
