"""I72: the tensor-parallel all-reduce between the two NeuronCores of one Inferentia2, measured before any model runs.

Qwen2.5-7B at TP=2 needs two all-reduces per decoder layer (after o_proj and after down_proj): 56 per token of D = 3584
float32 per row (14 KiB at c1, 57 KiB at c4). AWS NxD Inference runs 7B at TP=2 in 40.7 ms per token on this instance
(run 1), so an all-reduce must cost well under ~0.3 ms to leave TP=2 worth it. This probes the mechanisms of
fusezap.serve.neuron_tp (docs/NKI_BACKEND.md §11) on the real runtime, each in its own processes, and writes the choice:

  compile  nki_kernels.AllReduceSpec (nki.collectives.all_reduce) in each form (phbm / hbm / sbuf) compiled to an inf2 NEFF
           by neuronx-cc (.venv-nki): which forms the compiler accepts for NeuronCore-v2
  cc       two runtime workers (nrt_worker.py, one NeuronCore each: NEURON_RT_VISIBLE_CORES=0 / 1), the NEFF loaded with
           nrt_load_collectives (rank 0 / 1 of 2, NEURON_RT_ROOT_COMM_ID), executed together: correctness (the sum of
           both ranks' data on both ranks) and the per-execute time at 1, 2, 4 rows of 3584 floats, then 1 / 4 / 16 MB
  shm      two runtime workers exchanging through shared memory (nrt_worker.ShmAR, the "ar" request): the same sizes
  host     one runtime worker: a read and a write round trip of the message (what FZ_NEURON_TP_AR=host pays per
           all-reduce at least, before the socket exchange between the fusezap processes)
  xcore    one nrtpy process with both cores visible: a NormSpec(add) NEFF executed on core 0 with one input tensor
           allocated on core 1 (whether the runtime binds a peer core's tensor; the "one process owns both cores" design)
  ref      a small ordinary kernel's execute time on core 0 (NormSpec 1 x 3584): the per-launch floor an all-reduce adds to

  .venv/bin/python scripts/nki_tp_probe.py --out results/x/tp_probe.json --choice results/x/tp_choice.env
Prints "tp_probe: <phase> ..." lines; --choice writes FZ_NEURON_TP_AR / FZ_NEURON_TP_AR_KIND for the job (cc with the
fastest correct form if any, else shm if correct, else host).
"""
from __future__ import annotations
import argparse, json, os, socket, subprocess, sys, tempfile, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

D = 3584
ROWS = (1, 2, 4)
BIG = (1 << 18, 1 << 20, 4 << 20)   # floats: 1, 4, 16 MiB

def log(*a) -> None: print("tp_probe:", *a, flush=True)

def free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0)); return s.getsockname()[1]

class W:
  """A runtime worker (nrt_worker.py under .venv-nrt) with its own environment; call() with a timeout (a collective
  whose peer never executes must not hang the probe)."""
  def __init__(self, env: dict, name: str):
    from fusezap.device.neuron import nrt_python
    from pathlib import Path
    import fusezap.device.neuron as N
    self.N, self.name = N, name
    e = {**os.environ, **env}
    if os.path.isdir("/opt/aws/neuron/lib") and "/opt/aws/neuron/lib" not in e.get("LD_LIBRARY_PATH", ""):
      e["LD_LIBRARY_PATH"] = "/opt/aws/neuron/lib" + (":" + e["LD_LIBRARY_PATH"] if e.get("LD_LIBRARY_PATH") else "")
    script = Path(N.__file__).with_name("nrt_worker.py")
    self.p = subprocess.Popen([str(nrt_python()), str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, env=e)
    self.lock = threading.Lock()
  def call(self, req: dict, timeout: float = 120.0) -> dict:
    box: list = []
    def run() -> None:
      try:
        with self.lock:
          self.N._send(self.p.stdin, req)
          box.append(self.N._recv(self.p.stdout))
      except Exception as e: box.append({"ok": False, "err": f"{type(e).__name__}: {e}"})
    th = threading.Thread(target=run, daemon=True)
    th.start(); th.join(timeout)
    if th.is_alive(): return {"ok": False, "err": f"TIMEOUT after {timeout:.0f} s ({self.name})", "timeout": True}
    rep = box[0] if box else {"ok": False, "err": "no reply"}
    if rep is None: rep = {"ok": False, "err": f"{self.name} exited ({self.p.poll()})"}
    return rep
  def close(self) -> None:
    try: self.p.kill()
    except Exception: pass

def both(ws: list, reqs: list, timeout: float = 120.0) -> list:
  """The same moment on every worker (a collective needs every rank inside it): one thread per worker."""
  out: list = [None] * len(ws)
  def go(i: int) -> None: out[i] = ws[i].call(reqs[i], timeout)
  ths = [threading.Thread(target=go, args=(i,), daemon=True) for i in range(len(ws))]
  for t in ths: t.start()
  for t in ths: t.join(timeout + 5)
  return [o or {"ok": False, "err": "TIMEOUT", "timeout": True} for o in out]

def tail(s: list) -> list: return list(s[1:]) if len(s) > 1 else list(s)   # the first execute dropped (setup) when there are more

def stats(s: list) -> dict:
  a = np.asarray(s, np.float64) * 1e3
  return {"n": len(a), "p50_ms": round(float(np.median(a)), 4), "mean_ms": round(float(a.mean()), 4), "min_ms": round(float(a.min()), 4)} if len(a) else {}

def data(rank: int, n: int) -> np.ndarray: return (np.random.default_rng(77 + rank).standard_normal(n) * (rank + 1)).astype(np.float32)

def neff(sp) -> tuple:
  from fusezap.device.neuron import cached_neff, neuron_target
  prog = sp.render()
  specs = [(sp.n, "<f4"), (sp.n, "<f4")] if hasattr(sp, "world") else None
  t0 = time.perf_counter()
  path, meta = cached_neff(prog, specs, neuron_target())
  return path, meta, time.perf_counter() - t0

def compile_phase(R: dict) -> list[str]:
  from fusezap.codegen.nki_kernels import AR_KINDS, AllReduceSpec
  ok = []
  for kind in AR_KINDS:
    try:
      path, meta, s = neff(AllReduceSpec(D, 2, kind))
      R["compile"][kind] = {"ok": True, "seconds": round(s, 1), "path": str(path)}
      ok.append(kind)
      log(f"compile {kind}: ok in {s:.1f} s")
    except Exception as e:
      R["compile"][kind] = {"ok": False, "err": str(e)[-1500:]}
      log(f"compile {kind}: FAILED {str(e)[-400:]}")
  return ok

def run_ar(ws: list, n: int, iters: int, loader, execer, R: dict, label: str) -> dict:
  """Allocate in/out on both workers, write rank data, load (loader(i) -> request or None), execute `iters` times
  together (execer(i) -> request), read back, check the sum. Returns a record."""
  rec: dict = {"n": n, "bytes": n * 4}
  base = 1000 + n % 997
  for i, w in enumerate(ws):
    r = w.call({"op": "write", "id": base, "data": data(i, n).view(np.uint8), "new": {base: n * 4, base + 1: n * 4}})
    if not r.get("ok"): rec["err"] = f"write: {r.get('err')}"; return rec
  lreq, info = [loader(i, base) for i in range(len(ws))], [None] * len(ws)
  if lreq[0] is not None:
    t0 = time.perf_counter()
    info = both(ws, lreq, 300)   # nrt_load_collectives may wait for every rank: loaded together
    rec["load_s"] = round(time.perf_counter() - t0, 2)
    if bad := [r for r in info if not r.get("ok")]: rec["err"] = f"load: {bad[0].get('err')}"[:1500]; return rec
  reps = both(ws, [execer(i, base, 3, info[i]) for i in range(len(ws))], 300)   # warm-up: the ranks' first calls meet here
  if not (bad := [r for r in reps if not r.get("ok")]): reps = both(ws, [execer(i, base, iters, info[i]) for i in range(len(ws))], 120 + iters)
  if bad := [r for r in reps if not r.get("ok")]: rec["err"] = f"exec: {bad[0].get('err')}"[:1500]; rec["hang"] = any(r.get("timeout") for r in reps); return rec
  secs = [r.get("seconds") or [] for r in reps]
  if any(secs):
    rec["exec"] = [stats(s[1:] if len(s) > 1 else s) for s in secs]          # per rank; the first execute dropped (setup)
    rec["first_ms"] = [round(1e3 * s[0], 3) if s else None for s in secs]
  if reps[0].get("wall") is not None: rec["batch_wall_ms_per"] = [round(1e3 * r["wall"] / max(r.get("n", 1), 1), 4) for r in reps]
  want = data(0, n) + data(1, n)
  outs = [np.frombuffer(w.call({"op": "read", "id": base + 1, "n": n * 4})["data"], np.float32) for w in ws]
  rec["max_err"] = float(max(np.abs(o - want).max() for o in outs))
  rec["same_bits"] = bool(all(np.array_equal(outs[0].view(np.int32), o.view(np.int32)) for o in outs))
  rec["ok"] = rec["max_err"] <= 1e-5 * float(np.abs(want).max()) and rec["same_bits"]
  for w in ws: w.call({"op": "stats", "free": [base, base + 1]})
  return rec

def cc_phase(R: dict, kinds: list[str]) -> str | None:
  from fusezap.codegen.nki_kernels import AllReduceSpec
  best = None
  for kind in kinds:
    root = f"127.0.0.1:{free_port()}"
    ws = [W({"NEURON_RT_VISIBLE_CORES": str(r), "NEURON_RT_ROOT_COMM_ID": root, "FZ_NEURON_CC_RANK": str(r), "FZ_NEURON_CC_WORLD": "2"}, f"cc rank {r}")
          for r in range(2)]
    res = R["cc"][kind] = []
    try:
      for n in [D * r for r in ROWS] + (list(BIG) if best is None else []):
        try: path, meta, _ = neff(AllReduceSpec(n, 2, kind))
        except Exception as e: res.append({"n": n, "err": f"compile: {str(e)[-800:]}"}); log(f"cc {kind} n={n}: compile failed"); break
        key = meta.get("key") or path.parent.name
        loader = lambda i, b, key=key, path=path: {"op": "load", "key": key, "path": str(path), "cc": {"rank": i, "world": 2}}
        def execer(i, b, it, info, key=key):   # the NEFF's own tensor names (neff_binding), argument 0 = out, 1 = in
          from fusezap.device.neuron import neff_binding
          ins, outs, al = neff_binding(info["inputs"], info["outputs"], info.get("alias", {}), 2)
          ids = (b + 1, b)
          return {"op": "exec", "key": key, "in": {nm: ids[k] for nm, k in ins.items()}, "out": {nm: ids[k] for nm, k in outs.items()},
                  "alias": al, "iters": it}
        rec = run_ar(ws, n, 50 if n <= 4 * D else 10, loader, execer, R, f"cc {kind}")
        res.append(rec)
        log(f"cc {kind} n={n} ({n * 4 >> 10} KiB): " + (f"ok={rec.get('ok')} exec p50 {[e.get('p50_ms') for e in rec.get('exec', [])]} ms, "
                                                      f"first {rec.get('first_ms')} ms, load {rec.get('load_s')} s, max_err {rec.get('max_err')}"
                                                      if "err" not in rec else f"FAILED {rec['err'][:500]}"))
        if "err" in rec or not rec.get("ok"): break
      if res and all(r.get("ok") for r in res) and best is None: best = kind
    finally:
      for w in ws: w.close()
  return best

def shm_phase(R: dict) -> bool:
  from fusezap.serve.neuron_tp import shm_create
  path = shm_create(2, 16 << 20)
  ws = [W({"NEURON_RT_VISIBLE_CORES": str(r), "FZ_NEURON_TP_SHM": path, "FZ_NEURON_CC_RANK": str(r), "FZ_NEURON_CC_WORLD": "2",
           "FZ_NEURON_TP_SLOT": str(16 << 20)}, f"shm rank {r}") for r in range(2)]
  ok = True
  try:
    for n in [D * r for r in ROWS] + list(BIG):
      it = 50 if n <= 4 * D else 10
      execer = lambda i, b, k, info, n=n: {"op": "batch", "reqs": [{"op": "ar", "in": b, "out": b + 1, "n": n * 4} for _ in range(k)]}
      rec = run_ar(ws, n, it, lambda i, b: None, execer, R, "shm")
      R["shm"].append(rec)
      log(f"shm n={n} ({n * 4 >> 10} KiB): " + (f"ok={rec.get('ok')} per all-reduce {rec.get('batch_wall_ms_per')} ms (batch wall / {it})"
                                              if "err" not in rec else f"FAILED {rec['err'][:500]}"))
      ok = ok and bool(rec.get("ok"))
  finally:
    for w in ws: w.close()
  return ok

def host_phase(R: dict) -> None:
  w = W({"NEURON_RT_VISIBLE_CORES": "0"}, "host")
  try:
    for n in [D * r for r in ROWS]:
      w.call({"op": "stats", "new": {5: n * 4}})
      rd, wr = [], []
      a = data(0, n).view(np.uint8)
      for _ in range(30):
        t0 = time.perf_counter(); w.call({"op": "read", "id": 5, "n": n * 4}); rd.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); w.call({"op": "write", "id": 5, "data": a}); wr.append(time.perf_counter() - t0)
      rec = {"n": n, "read": stats(rd[3:]), "write": stats(wr[3:])}
      R["host"].append(rec)
      log(f"host n={n}: read round trip p50 {rec['read'].get('p50_ms')} ms, write {rec['write'].get('p50_ms')} ms")
      w.call({"op": "stats", "free": [5]})
  finally: w.close()

def ref_phase(R: dict) -> None:
  from fusezap.codegen.nki_kernels import NormSpec
  sp = NormSpec(1, D, 1e-6, False)
  from fusezap.device.neuron import cached_neff, neuron_target
  path, meta = cached_neff(sp.render(), [(D, "<f4"), (D, "<f4"), (D, "<f4")], neuron_target())
  w = W({"NEURON_RT_VISIBLE_CORES": "0"}, "ref")
  try:
    w.call({"op": "stats", "new": {1: D * 4, 2: D * 4, 3: D * 4}})
    info = w.call({"op": "load", "key": "ref", "path": str(path)})
    from fusezap.device.neuron import neff_binding
    ins, outs, al = neff_binding(info["inputs"], info["outputs"], info.get("alias", {}), 3)
    r = w.call({"op": "exec", "key": "ref", "in": {k: 1 + v for k, v in ins.items()}, "out": {k: 1 + v for k, v in outs.items()}, "alias": al, "iters": 50})
    R["ref"] = {"kernel": sp.name, "exec": stats((r.get("seconds") or [])[1:]), "err": r.get("err")}
    log(f"ref {sp.name}: execute p50 {R['ref']['exec'].get('p50_ms')} ms")
  finally: w.close()

XCORE = r'''
import json, sys, time, numpy as np
from nrtpy import NrtpyModel, get_nrtpy_singleton
path, D = sys.argv[1], int(sys.argv[2])
rt = get_nrtpy_singleton()
out = {"visible": rt.get_visible_neuron_core_count() if hasattr(rt, "get_visible_neuron_core_count") else None}
def T(a, core):
  t = rt.allocate_tensor(size=a.nbytes, core_id=core, name=None)
  rt.tensor_write_from_pybuffer(t, np.ascontiguousarray(a))
  return t
try:
  m = NrtpyModel.load_from_neff(path, core_id=0)
  x, d, w = (np.random.default_rng(i).standard_normal(D).astype(np.float32) for i in range(3))
  bufs = {0: T(np.zeros(D, np.float32), 0), 1: T(np.zeros(D, np.float32), 0), 2: T(x, 0), 3: T(d, 1), 4: T(w, 0)}   # d on core 1
  ins, outs = {}, {}
  for n in m.input_tensors_info:
    k = int(str(n).split(".")[0][4:]); ins[str(n)] = bufs[k]
  for n in m.output_tensors_info:
    k = int(str(n)[4:])
    outs[str(n)] = rt.slice_from_tensor(ins[f"data{k}.must_alias_input"], name=str(n)) if f"data{k}.must_alias_input" in ins else bufs[k]
  ts = []
  for _ in range(20):
    t0 = time.perf_counter(); rt.execute(m.model_ref, ins, outs); ts.append(time.perf_counter() - t0)
  xo = np.empty(D, np.float32); rt.tensor_read_to_pybuffer(bufs[1], xo)
  out.update(ok=bool(np.abs(xo - (x + d)).max() <= 1e-6 * np.abs(x + d).max()), max_err=float(np.abs(xo - (x + d)).max()),
             exec_p50_ms=float(np.median(ts[1:]) * 1e3))
except Exception as e: out.update(ok=False, err=f"{type(e).__name__}: {str(e)[:1500]}")
print("XCORE " + json.dumps(out), flush=True)
'''

def xcore_phase(R: dict) -> None:
  from fusezap.codegen.nki_kernels import NormSpec
  from fusezap.device.neuron import cached_neff, neuron_target, nrt_python
  sp = NormSpec(1, D, 1e-6, True)
  path, _ = cached_neff(sp.render(), [(D, "<f4")] * 5, neuron_target())
  d = tempfile.mkdtemp(prefix="fz_xcore_")
  src = os.path.join(d, "xcore.py")
  with open(src, "w") as f: f.write(XCORE)
  env = {**os.environ, "NEURON_RT_VISIBLE_CORES": "0,1"}
  try:
    p = subprocess.run([str(nrt_python()), "-I", src, str(path), str(D)], capture_output=True, text=True, timeout=300, env=env)
    line = next((l for l in p.stdout.splitlines() if l.startswith("XCORE ")), None)
    R["xcore"] = json.loads(line[6:]) if line else {"ok": False, "err": (p.stderr or p.stdout)[-1500:]}
  except Exception as e: R["xcore"] = {"ok": False, "err": f"{type(e).__name__}: {e}"}
  log(f"xcore (core 0 kernel, input on core 1): {R['xcore']}")

# ---------------------------------------------------------------- N8 I72 follow-up: collectives inside one kernel
# Run 2 (7B TP=2 served 58.3 ms/token): rank 0's step = 50.7 ms of synchronous executes (57 launches: halves 0.557 + 1.031 ms
# per layer, head 6.2) + 6.0 ms of shm all-reduces in the worker + 1.8 ms host; the whole shard layer is 1.043 ms of device
# time in one kernel. So the cost is less the all-reduces themselves than the 2 launches per layer around them (no overlap of
# the attention half's serial tail with the MLP's weight stream, a launch's fixed cost each). Two questions before a model:
#   chain   is a collective's fixed cost (0.36-0.39 ms per AllReduceSpec launch vs a 0.11 ms plain kernel) per launch or per
#           collective? ARChainSpec: k all-reduces in one kernel (dep: serial, the layer's pattern; indep: k of the same data),
#           execute time at k = 1 .. 56 -> per launch a + per collective b (least squares)
#   layer   the shard's decoder layer with both all-reduces inside (LayerSpec part "tp") vs the halves around two shm "ar"
#           requests (the served path), on the same random 7B-shard inputs on both cores: execute time and bits (xo, dn, the
#           pool rows, the X / o scratch); the whole layer without collectives on each core (the floor), and FUSE layers of the
#           "tp" kernel as one FusedSpec NEFF (what FZ_NEURON_FUSE does in a model) vs as many single launches, bitwise

CHAIN_K = (1, 2, 4, 8, 16, 28, 56)
DRY = False
DRY_PROGS: dict = {}   # --dry: NEFF key -> NKIProgram (the shim runs the kernel source; no compiler, no device)

def neff_dry(prog) -> tuple:
  import hashlib
  from pathlib import Path
  key = "dry_" + hashlib.sha256(prog.src.encode()).hexdigest()[:24]
  DRY_PROGS[key] = prog
  return Path("/dry") / key / "kernel.neff", {"key": key}, 0.0

def compile_prog(prog, specs: list) -> tuple:
  """(path, meta, seconds) of a rendered program's NEFF at these argument specs ((count, numpy dtype str) each)."""
  if DRY: return neff_dry(prog)
  from fusezap.device.neuron import cached_neff, neuron_target
  t0 = time.perf_counter()
  path, meta = cached_neff(prog, specs, neuron_target())
  return path, meta, time.perf_counter() - t0

class _Peers:
  """--dry: the shim's collective and the "ar" request between ShimW threads (one per rank): the float32 rank-order sum."""
  def __init__(self, world: int):
    self.world, self.b, self.slots, self.local = world, threading.Barrier(world, timeout=120), [None] * world, threading.local()
  def reduce(self, a: np.ndarray) -> np.ndarray:
    self.slots[self.local.rank] = np.array(a, np.float32, copy=True).reshape(-1)
    self.b.wait()
    s = self.slots[0].copy()
    for x in self.slots[1:]: s += x
    self.b.wait()
    return s.reshape(np.shape(a))

class ShimW:
  """--dry stand-in for W: nrt_worker's requests (write / read / stats / load / exec / batch / ar) on host byte buffers, the
  kernels run by NKISHIM in the calling thread, collectives through _Peers. Checks the probe's own logic on the Mac."""
  peers: "_Peers | None" = None
  def __init__(self, rank: int, name: str):
    from fusezap.device import nkishim
    self.rank, self.name, self.bufs, self.models = rank, name, {}, {}
    if ShimW.peers is None:
      ShimW.peers = _Peers(2)
      nkishim.COLLECTIVE[:] = [lambda a, op, g: ShimW.peers.reduce(a)]
  def call(self, req: dict, timeout: float = 120.0) -> dict:
    ShimW.peers.local.rank = self.rank
    try: return self._handle(req)
    except Exception as e: return {"ok": False, "err": f"{type(e).__name__}: {e}"}
  def _handle(self, req: dict) -> dict:
    from fusezap.device import nkishim
    for i, n in (req.get("new") or {}).items(): self.bufs.setdefault(i, np.zeros(n, np.uint8))
    for i in req.get("free") or []: self.bufs.pop(i, None)
    op = req.get("op")
    if op == "stats": return {"ok": True}
    if op == "write":
      a = np.frombuffer(bytes(req["data"]), np.uint8); self.bufs[req["id"]][:a.size] = a; return {"ok": True}
    if op == "read": return {"ok": True, "data": self.bufs[req["id"]][:req.get("n")].copy()}
    if op == "load":
      p = DRY_PROGS[req["key"]]; self.models[req["key"]] = p
      return {"ok": True, "inputs": [f"data{i}" for i in range(p.n_bufs)], "outputs": [f"data{o}" for o in p.outs], "alias": {}}
    if op == "exec":
      p = self.models[req["key"]]
      ids = {**{int(k[4:]): v for k, v in req["in"].items()}, **{int(k[4:]): v for k, v in req["out"].items()}}
      from fusezap.dtype import dtypes, to_np
      arrs = [self.bufs[ids[i]].view(to_np(p.buf_dtypes.get(i, dtypes.float32))) for i in range(p.n_bufs)]
      ts = []
      for _ in range(int(req.get("iters", 1))):
        t0 = time.perf_counter(); nkishim.execute(p, arrs); ts.append(time.perf_counter() - t0)
      return {"ok": True, "seconds": ts}
    if op == "ar":
      n = int(req["n"]) // 4
      s = ShimW.peers.reduce(self.bufs[req["in"]][:4 * n].view(np.float32))
      self.bufs[req["out"]][:4 * n] = s.view(np.uint8); return {"ok": True, "seconds": []}
    if op == "batch":
      t0, secs = time.perf_counter(), []
      for r in req["reqs"]:
        rep = self._handle(r)
        if not rep.get("ok"): return rep
        secs += rep.get("seconds", [])
      return {"ok": True, "seconds": secs, "n": len(req["reqs"]), "wall": time.perf_counter() - t0}
    return {"ok": False, "err": f"dry: op {op!r}"}
  def close(self) -> None: pass

def cc_workers(tag: str, shm: str | None = None) -> list:
  """Two runtime workers, one per core, loaded for collectives (and the shm all-reduce when shm is given)."""
  if DRY: return [ShimW(r, f"{tag} rank {r}") for r in range(2)]
  root = f"127.0.0.1:{free_port()}"
  extra = lambda r: {"FZ_NEURON_TP_SHM": shm, "FZ_NEURON_TP_SLOT": str(16 << 20)} if shm else {}
  return [W({"NEURON_RT_VISIBLE_CORES": str(r), "NEURON_RT_ROOT_COMM_ID": root, "FZ_NEURON_CC_RANK": str(r), "FZ_NEURON_CC_WORLD": "2",
             **extra(r)}, f"{tag} rank {r}") for r in range(2)]

def binder(info: dict, nargs: int, ids: list, key: str, iters: int = 1) -> dict:
  """An exec request for a loaded NEFF (its tensor names, neff_binding) with argument i bound to buffer ids[i]."""
  from fusezap.device.neuron import neff_binding
  ins, outs, al = neff_binding(info["inputs"], info["outputs"], info.get("alias", {}), nargs)
  return {"op": "exec", "key": key, "in": {nm: ids[k] for nm, k in ins.items()}, "out": {nm: ids[k] for nm, k in outs.items()},
          "alias": al, "iters": iters}

def load_both(ws: list, prog, specs: list, cc: bool) -> tuple:
  """Compile prog, load it on both workers (collectives: rank i of 2, loaded together). (key, [info per rank], compile s)."""
  path, meta, s = compile_prog(prog, specs)
  key = meta.get("key") or path.parent.name
  reqs = [{"op": "load", "key": key, "path": str(path), **({"cc": {"rank": i, "world": 2}} if cc else {})} for i in range(len(ws))]
  info = both(ws, reqs, 600)
  if bad := [r for r in info if not r.get("ok")]: raise RuntimeError(f"load {prog.name}: {bad[0].get('err')}"[:1500])
  return key, info, s

def fit(ks: list, ms: list) -> dict:
  """Least squares ms = a + b * k: a per launch, b per collective."""
  if len(ks) < 2: return {}
  b, a = np.polyfit(np.asarray(ks, np.float64), np.asarray(ms, np.float64), 1)
  return {"per_launch_ms": round(float(a), 4), "per_collective_ms": round(float(b), 4)}

def chain_phase(R: dict, ks: tuple = CHAIN_K, sizes: tuple = (D, 4 * D), iters: int = 30) -> None:
  from fusezap.codegen.nki_kernels import ARChainSpec
  ws = cc_workers("chain")
  R["chain"] = {"runs": []}
  try:
    for form, ns, kk in (("dep", sizes, ks), ("indep", sizes[:1], tuple(k for k in ks if k <= 16))):
      for n in ns:
        pts = []
        base = 5000 + 10 * len(R["chain"]["runs"])
        x = [data(r, n) for r in range(2)]
        for i, w in enumerate(ws):
          r = w.call({"op": "write", "id": base, "data": x[i].view(np.uint8), "new": {base: n * 4, base + 1: n * 4}})
          if not r.get("ok"): raise RuntimeError(f"write: {r.get('err')}")
        for k in kk:
          sp = ARChainSpec(n, 2, k, form)
          rec = {"form": form, "n": n, "k": k}
          try:
            key, info, cs = load_both(ws, sp.render(), [(n, "<f4"), (n, "<f4")], True)
            reps = both(ws, [binder(info[i], 2, [base + 1, base], key, 3) for i in range(2)], 300)
            if not any(not r.get("ok") for r in reps):
              reps = both(ws, [binder(info[i], 2, [base + 1, base], key, iters) for i in range(2)], 300)
            if bad := [r for r in reps if not r.get("ok")]: raise RuntimeError(f"exec: {bad[0].get('err')}"[:1200])
            rec["exec"] = [stats(tail(r.get("seconds") or [])) for r in reps]
            want = (x[0] + x[1]) * np.float32(2.0 ** (k - 1) if form == "dep" else 1.0)
            outs = [np.frombuffer(bytes(w.call({"op": "read", "id": base + 1, "n": n * 4})["data"]), np.float32) for w in ws]
            rec["exact"] = bool(all(np.array_equal(o, want) for o in outs))
            rec["compile_s"] = round(cs, 1)
            pts.append((k, max(e.get("p50_ms") or 0 for e in rec["exec"])))
            log(f"chain {form} n={n} ({n * 4 >> 10} KiB) k={k}: exec p50 {[e.get('p50_ms') for e in rec['exec']]} ms, exact {rec['exact']}")
          except Exception as e:
            rec["err"] = str(e)[-1200:]
            log(f"chain {form} n={n} k={k}: FAILED {rec['err'][-400:]}")
            R["chain"]["runs"].append(rec)
            break
          R["chain"]["runs"].append(rec)
        f = fit([p[0] for p in pts], [p[1] for p in pts])
        R["chain"][f"fit_{form}_{n}"] = f
        if f: log(f"chain fit {form} n={n}: per launch {f['per_launch_ms']} ms + per collective {f['per_collective_ms']} ms "
                  f"(x56 per token: {round(56 * f['per_collective_ms'], 2)} ms)")
  finally:
    for w in ws: w.close()

SHARD7B = {"D": 3584, "Hq": 14, "Hkv": 2, "hd": 128, "I": 9472, "P": 1024}   # Qwen2.5-7B at TP=2: one core's shard
TINY = {"D": 256, "Hq": 14, "Hkv": 2, "hd": 128, "I": 192, "P": 512}         # --tiny (tests): the same head layout, small

def layer_specs(N: int, sh: dict, bf16: bool) -> dict:
  """The shard layer's kernels as the model runs them (LayerSpec with the env's layer switches, tiled, bf16 weights):
  whole (no collectives), attn / mlp (the halves), tp (both all-reduces inside)."""
  import dataclasses
  from fusezap.codegen.nki_kernels import LayerSpec, layer_variant
  from fusezap.codegen.nki_bf16 import LayerBF16, bf16_attn
  v = layer_variant()
  sp = LayerSpec(N, sh["D"], sh["Hq"], sh["Hkv"], sh["hd"], sh["I"], sh["P"], 1e-6, float(1 / np.sqrt(sh["hd"])), 0, True, True, False,
                 v["heads"], early=v["early"], tile=True, wcols=v["wcols"])
  wrap = (lambda s: LayerBF16(s, bf16_attn())) if bf16 else (lambda s: s)
  return {"whole": wrap(sp), "attn": wrap(dataclasses.replace(sp, part="attn")), "mlp": wrap(dataclasses.replace(sp, part="mlp")),
          "tp": wrap(dataclasses.replace(sp, part="tp", tpw=2))}

def layer_sizes(N: int, sh: dict, bf16: bool) -> dict:
  """Argument name -> (elements, numpy dtype str) for every LayerSpec argument (weights K-major; bf16 as uint16 bits)."""
  D, Hq, Hkv, hd, I, P = (sh[k] for k in ("D", "Hq", "Hkv", "hd", "I", "P"))
  Nq, Ho = (Hq + 2 * Hkv) * hd, Hq * hd
  wd = "<u2" if bf16 else "<f4"
  f = lambda n: (n, "<f4")
  return {"xo": f(N * D), "dn": f(N * D), "po": f(N * D), "o": f(N * D), "x": f(N * D), "d": f(N * D), "pool": f(P * 4 * Hkv * hd),
          "xs": f(N * (Nq + hd)), "os": f(N * Ho), "ln1": f(D), "ln2": f(D), "bqkv": f(Nq), "bo": f(D), "cos": f(N * hd), "sin": f(N * hd),
          "wt": f(N * P), "meta": (2 * N + P, "<i4"), "wqkv": (D * Nq, wd), "wo": (Ho * D, wd), "wgu": (D * 2 * I, wd), "wd": (I * D, wd)}

def layer_inputs(N: int, sh: dict, bf16: bool, rank: int) -> dict:
  """Random inputs of one rank: the residual (x, d), RoPE rows, the slot one-hot and key ranges shared by the ranks; the
  weights and the pool rows (this rank's kv heads) per rank. bf16 weights as bit patterns of ~N(0, 0.02)."""
  from fusezap.serve.weights import f32_to_bf16
  sz = layer_sizes(N, sh, bf16)
  P, hd = sh["P"], sh["hd"]
  g, gr = np.random.default_rng(900), np.random.default_rng(910 + rank)
  out = {"x": g.standard_normal(N * sh["D"]).astype(np.float32), "d": (g.standard_normal(N * sh["D"]) * 0.5).astype(np.float32),
         "cos": np.cos(g.uniform(0, 6.3, N * hd)).astype(np.float32), "sin": np.sin(g.uniform(0, 6.3, N * hd)).astype(np.float32)}
  slots = [P // 2 + t for t in range(N)]
  wt = np.zeros((N, P), np.float32); wt[np.arange(N), slots] = 1.0
  out["wt"] = wt.reshape(-1)
  out["meta"] = np.concatenate([np.zeros(N, np.int32), np.asarray(slots, np.int32), np.arange(P, dtype=np.int32)])
  for k in ("ln1", "ln2"): out[k] = (1.0 + 0.1 * gr.standard_normal(sh["D"])).astype(np.float32)
  for k in ("bqkv", "bo"): out[k] = (0.05 * gr.standard_normal(sz[k][0])).astype(np.float32)
  out["pool"] = gr.standard_normal(sz["pool"][0]).astype(np.float32)
  for k in ("wqkv", "wo", "wgu", "wd"):
    w = (gr.standard_normal(sz[k][0], dtype=np.float32) * np.float32(0.02))
    out[k] = f32_to_bf16(w) if bf16 else w
  return out

def layer_phase(R: dict, rows: tuple = (1, 4), fuse: int = 4, sh: dict = SHARD7B, bf16: bool = True, iters: int = 20) -> None:
  from fusezap.serve.neuron_tp import shm_create
  from fusezap.codegen.nki_kernels import FusedSpec
  shm = None if DRY else shm_create(2, 16 << 20)
  ws = cc_workers("layer", shm)
  R["layer"] = {"shape": sh, "bf16": bf16, "runs": []}
  nid = [20000]
  def new_ids(names) -> dict:
    out = {}
    for n in names: out[n] = nid[0]; nid[0] += 1
    return out
  try:
    for N in rows:
      rec: dict = {"rows": N}
      R["layer"]["runs"].append(rec)
      try:
        specs, sz = layer_specs(N, sh, bf16), layer_sizes(N, sh, bf16)
        inp = [layer_inputs(N, sh, bf16, r) for r in range(2)]
        # buffers: shared inputs (one copy per rank), per path the written ones; pool: a fresh copy per path (written)
        IN = new_ids(["x", "d", "ln1", "wqkv", "bqkv", "cos", "sin", "wt", "meta", "wo", "bo", "ln2", "wgu", "wd"])
        paths = {"whole": ["xo", "dn", "pool", "xs", "os"], "tp": ["xo", "dn", "pool", "xs", "os"],
                 "halves": ["x1", "po", "pool", "xs", "os", "o", "x2", "dnp", "dnr"]}
        OUT = {p: new_ids(ns) for p, ns in paths.items()}
        nb = lambda n: sz[{"x1": "xo", "x2": "xo", "dnp": "dn", "dnr": "dn"}.get(n, n)][0] * (2 if sz[{"x1": "xo", "x2": "xo", "dnp": "dn", "dnr": "dn"}.get(n, n)][1] == "<u2" else 4)
        for r, w in enumerate(ws):
          new = {i: nb(n) for n, i in IN.items()}
          for p, ids in OUT.items(): new.update({i: nb(n) for n, i in ids.items()})
          rep = w.call({"op": "stats", "new": new})
          if not rep.get("ok"): raise RuntimeError(f"alloc: {rep.get('err')}")
          for n, i in IN.items():
            rep = w.call({"op": "write", "id": i, "data": np.ascontiguousarray(inp[r][n]).view(np.uint8)}, 600)
            if not rep.get("ok"): raise RuntimeError(f"write {n}: {rep.get('err')}")
          for p, ids in OUT.items():
            w.call({"op": "write", "id": ids["pool"], "data": inp[r]["pool"].view(np.uint8)}, 600)
        def ids_for(spec, names: dict) -> list:   # argument i -> buffer id
          return [names.get(a, IN.get(a)) for a in spec.args()]
        def specs_for(spec, names: dict) -> list:
          inv = {v: k for k, v in {**IN, **names}.items()}
          return [(sz[_alias(inv[i])][0], sz[_alias(inv[i])][1]) for i in ids_for(spec, names)]
        def _alias(n: str) -> str: return {"x1": "xo", "x2": "xo", "dnp": "dn", "dnr": "dn"}.get(n, n)
        loaded = {}
        for nm, cc in (("whole", False), ("attn", False), ("mlp", False), ("tp", True)):
          names = {"whole": OUT["whole"], "tp": OUT["tp"],
                   "attn": {"xo": OUT["halves"]["x1"], "po": OUT["halves"]["po"], "pool": OUT["halves"]["pool"], "xs": OUT["halves"]["xs"], "os": OUT["halves"]["os"]},
                   "mlp": {"xo": OUT["halves"]["x2"], "dn": OUT["halves"]["dnp"], "x": OUT["halves"]["x1"], "o": OUT["halves"]["o"]}}[nm]
          prog = specs[nm].render()
          key, info, cs = load_both(ws, prog, specs_for(specs[nm], names), cc)
          loaded[nm] = (key, info, ids_for(specs[nm], names), prog.n_bufs)
          rec.setdefault("compile_s", {})[nm] = round(cs, 1)
          rec.setdefault("kernel", {})[nm] = prog.name
        ex = lambda nm, i, it: binder(loaded[nm][1][i], loaded[nm][3], loaded[nm][2], loaded[nm][0], it)
        H = OUT["halves"]
        nbytes = N * sh["D"] * 4
        step = lambda i: [ex("attn", i, 1), {"op": "ar", "in": H["po"], "out": H["o"], "n": nbytes}, ex("mlp", i, 1),
                          {"op": "ar", "in": H["dnp"], "out": H["dnr"], "n": nbytes}]
        both(ws, [{"op": "batch", "reqs": step(i)} for i in range(2)], 600)   # warm
        reps = both(ws, [{"op": "batch", "reqs": step(i) * iters} for i in range(2)], 600)
        if bad := [r for r in reps if not r.get("ok")]: raise RuntimeError(f"halves: {bad[0].get('err')}"[:1200])
        rec["halves_ms_per_layer"] = [round(1e3 * r["wall"] / iters, 4) for r in reps]
        secs = [np.asarray(r["seconds"]) for r in reps]
        rec["halves_attn_ms"] = [stats(list(s[0::2])).get("p50_ms") for s in secs]
        rec["halves_mlp_ms"] = [stats(list(s[1::2])).get("p50_ms") for s in secs]
        for nm in ("tp", "whole"):
          both(ws, [ex(nm, i, 3) for i in range(2)], 600)
          reps = both(ws, [ex(nm, i, iters) for i in range(2)], 600)
          if bad := [r for r in reps if not r.get("ok")]: raise RuntimeError(f"{nm}: {bad[0].get('err')}"[:1200])
          rec[f"{nm}_ms"] = [stats(tail(r.get("seconds") or [])).get("p50_ms") for r in reps]
        rd = lambda w, i, n: np.frombuffer(bytes(w.call({"op": "read", "id": i, "n": n})["data"]), np.uint8)
        T = OUT["tp"]
        pairs = {"xo": (T["xo"], H["x2"]), "dn": (T["dn"], H["dnr"]), "pool": (T["pool"], H["pool"]), "xs": (T["xs"], H["xs"]), "os": (T["os"], H["os"])}
        bits = {}
        for k, (a, b) in pairs.items():
          n = nb(k)
          bits[k] = all(np.array_equal(rd(w, a, n), rd(w, b, n)) for w in ws)
        bits["ranks_same_xo"] = np.array_equal(rd(ws[0], T["xo"], nb("xo")), rd(ws[1], T["xo"], nb("xo")))
        bits["xo_vs_whole_differs"] = not np.array_equal(rd(ws[0], T["xo"], nb("xo")), rd(ws[0], OUT["whole"]["xo"], nb("xo")))   # the sum matters
        rec["bitwise_tp_vs_halves"] = bits
        rec["ok"] = all(v for k, v in bits.items())
        log(f"layer rows {N}: whole {rec['whole_ms']} ms (no collectives) | halves + shm {rec['halves_ms_per_layer']} ms per layer "
            f"(attn {rec['halves_attn_ms']} + mlp {rec['halves_mlp_ms']} execute) | tp (2 collectives inside) {rec['tp_ms']} ms | "
            f"bits tp == halves {bits}")
        if fuse > 1:   # FUSE layers of the tp kernel in one NEFF (FusedSpec, as fuse_schedule builds it) vs FUSE single launches
          tp = specs["tp"]
          args = tp.args()
          wr = ["xo", "dn", "pool", "xs", "os"]
          per = [new_ids([f"{n}{j}" for n in wr]) for j in range(fuse)]
          for r, w in enumerate(ws):
            w.call({"op": "stats", "new": {i: nb(n[:-len(str(j))]) for j, ids in enumerate(per) for n, i in ids.items()}})
            for j, ids in enumerate(per): w.call({"op": "write", "id": ids[f"pool{j}"], "data": inp[r]["pool"].view(np.uint8)}, 600)
          def layer_ids(j: int) -> dict:
            m = {n: per[j][f"{n}{j}"] for n in wr}
            if j: m.update(x=per[j - 1][f"xo{j - 1}"], d=per[j - 1][f"dn{j - 1}"])
            return m
          order: list = []   # fused argument order: every written buffer first, then the inputs
          for j in range(fuse): order += [layer_ids(j)[n] for n in wr]
          rest = []
          for j in range(fuse):
            for i in ids_for(tp, layer_ids(j)):
              if i not in order and i not in rest: rest.append(i)
          allids = order + rest
          pos = {i: k for k, i in enumerate(allids)}
          prog1 = tp.render()
          parts = tuple((prog1.name, prog1.src, tuple(pos[i] for i in ids_for(tp, layer_ids(j)))) for j in range(fuse))
          dts = tuple(sorted({(pos[i], prog1.buf_dtypes[a]) for j in range(fuse) for a, i in zip(range(len(args)), ids_for(tp, layer_ids(j))) if a in prog1.buf_dtypes}))
          fs = FusedSpec(parts, len(allids), tuple(range(len(order))), dts)
          fprog = fs.render()
          inv = {}
          for j in range(fuse):
            for a, i in zip(args, ids_for(tp, layer_ids(j))): inv[i] = a
          fspecs = [sz[inv[i]] for i in allids]
          key, info, cs = load_both(ws, fprog, fspecs, True)
          rec["fused_compile_s"] = round(cs, 1)
          fx = lambda i, it: binder(info[i], fprog.n_bufs, allids, key, it)
          both(ws, [fx(i, 2) for i in range(2)], 900)
          reps = both(ws, [fx(i, iters) for i in range(2)], 900)
          if bad := [r for r in reps if not r.get("ok")]: raise RuntimeError(f"fused: {bad[0].get('err')}"[:1200])
          rec[f"fused{fuse}_ms_per_layer"] = [round((stats(tail(r.get("seconds") or [])).get("p50_ms") or 0) / fuse, 4) for r in reps]
          last = per[fuse - 1]
          fx_out = {n: [rd(w, last[f"{n}{fuse - 1}"], nb(n)) for w in ws] for n in ("xo", "dn")}
          # the same FUSE layers as single tp launches (one batch), into the same buffers: the fused kernel's bits
          seq = lambda i: [binder(loaded["tp"][1][i], loaded["tp"][3], ids_for(tp, layer_ids(j)), loaded["tp"][0], 1) for j in range(fuse)]
          both(ws, [{"op": "batch", "reqs": seq(i)} for i in range(2)], 900)
          reps = both(ws, [{"op": "batch", "reqs": seq(i) * iters} for i in range(2)], 900)
          rec[f"single{fuse}_ms_per_layer"] = [round(1e3 * r.get("wall", 0) / (iters * fuse), 4) for r in reps]
          rec["fused_bitwise_vs_single"] = all(np.array_equal(fx_out[n][r], rd(ws[r], last[f"{n}{fuse - 1}"], nb(n))) for n in ("xo", "dn") for r in range(2))
          rec["ok"] = rec["ok"] and rec["fused_bitwise_vs_single"]
          log(f"layer rows {N}: {fuse} tp layers fused in one NEFF {rec[f'fused{fuse}_ms_per_layer']} ms per layer vs {fuse} launches "
              f"{rec[f'single{fuse}_ms_per_layer']} ms per layer (batch wall); bitwise {rec['fused_bitwise_vs_single']}")
      except Exception as e:
        rec["err"] = f"{type(e).__name__}: {str(e)[-1500:]}"
        log(f"layer rows {N}: FAILED {rec['err'][-600:]}")
  finally:
    for w in ws: w.close()

def layer_prediction(R: dict) -> list[str]:
  """PREDICTIONS (labelled) of TPOT from the layer phase: 28 layers of the measured per-layer cost + the head (6.2 ms
  replicated, run 2) + host 1.8 ms (run 2)."""
  out = []
  for rec in (R.get("layer") or {}).get("runs", []):
    if "err" in rec: continue
    c = lambda k: max([v for v in rec.get(k) or [] if v is not None] or [None], key=lambda v: -1 if v is None else v)
    for k, what in (("halves_ms_per_layer", "halves + shm (served today)"), ("tp_ms", "tp, one launch per layer"),
                    (next((k for k in rec if k.startswith("fused") and k.endswith("_per_layer")), ""), "tp, layers fused in one NEFF")):
      if k and c(k) is not None: out.append(f"PREDICTION rows {rec['rows']}: {what}: 28 x {c(k):.3f} + head 6.2 + host 1.8 = {28 * c(k) + 8.0:.1f} ms/token")
  return out

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="results/nki_tp_probe.json")
  ap.add_argument("--choice", default="", help="write FZ_NEURON_TP_AR / FZ_NEURON_TP_AR_KIND here (sourceable)")
  ap.add_argument("--skip", default="", help="comma list of phases to skip (compile,cc,shm,host,ref,xcore)")
  ap.add_argument("--only", default="", help="N8 I72 follow-up: run only these of chain,layer (no AR choice) and exit")
  ap.add_argument("--rows", default="1,4", help="layer phase: decode rows")
  ap.add_argument("--fuse-layers", type=int, default=4, help="layer phase: tp layers fused in one NEFF (0/1: skip)")
  ap.add_argument("--chain-k", default=",".join(map(str, CHAIN_K)), help="chain phase: collectives per kernel")
  ap.add_argument("--iters", type=int, default=20)
  ap.add_argument("--f32", action="store_true", help="layer phase: float32 weights (default bf16, the 7B job's)")
  ap.add_argument("--tiny", action="store_true", help="layer phase: small shapes (tests)")
  ap.add_argument("--dry", action="store_true", help="NKISHIM in process instead of compiler + runtime (tests, on the Mac)")
  a = ap.parse_args()
  global DRY
  DRY = a.dry
  skip = set(a.skip.split(","))
  R: dict = {"compile": {}, "cc": {}, "shm": [], "host": [], "ref": {}, "xcore": {}, "D": D}
  t0 = time.perf_counter()
  def save() -> None:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f: json.dump(R, f, indent=1, default=str)
  if a.only:
    only = set(a.only.split(","))
    if "chain" in only:
      chain_phase(R, tuple(int(k) for k in a.chain_k.split(",")), iters=a.iters)
      save()
    if "layer" in only:
      layer_phase(R, tuple(int(r) for r in a.rows.split(",")), a.fuse_layers, TINY if a.tiny else SHARD7B, not a.f32, a.iters)
      for ln in layer_prediction(R): log(ln)
      save()
    log(f"done in {time.perf_counter() - t0:.0f} s")
    return
  kinds = compile_phase(R) if "compile" not in skip else []
  save()
  best = None
  if "ref" not in skip:
    try: ref_phase(R)
    except Exception as e: log(f"ref failed: {e}")
  if kinds and "cc" not in skip: best = cc_phase(R, kinds)
  save()
  shm_ok = False
  if "shm" not in skip: shm_ok = shm_phase(R)
  save()
  if "host" not in skip: host_phase(R)
  if "xcore" not in skip: xcore_phase(R)
  def at(recs: list, n: int) -> float | None:
    """The end-to-end cost of one all-reduce inside a step's batch: cc, the collective's execute time; shm, the batch's
    wall time per "ar" request, which includes the device -> host read, the exchange and the host -> device write."""
    r = next((x for x in recs if x.get("n") == n and x.get("ok")), None)
    if r is None: return None
    return max(e.get("p50_ms") or 0 for e in r["exec"]) if "exec" in r else max(r.get("batch_wall_ms_per") or [0])
  # N8 run 1: "cc if it works" picked the collective at 0.36 ms per all-reduce over shm at 0.04 ms. Pick by the measured
  # cost at the decode sizes (1 and 4 rows, the c1 and c4 steps), cheaper wins; host only if neither works.
  cand = {}
  if best: cand["cc"] = (best, at(R["cc"].get(best, []), D), at(R["cc"].get(best, []), 4 * D))
  if shm_ok: cand["shm"] = ("phbm", at(R["shm"], D), at(R["shm"], 4 * D))
  cand = {m: v for m, v in cand.items() if v[1] is not None and v[2] is not None}
  if cand: mode = min(cand, key=lambda m: cand[m][1] + cand[m][2])
  else: mode = "host"
  kind, m1, m4 = cand.get(mode, ("phbm", None, None))
  R["choice"] = {"mode": mode, "kind": kind, "ms_1row": m1, "ms_4rows": m4, "candidates": cand, "seconds": round(time.perf_counter() - t0, 1)}
  save()
  log(f"candidates (ms per all-reduce at 1 / 4 rows, device copies included): " + ", ".join(f"{m} {v[1]} / {v[2]}" for m, v in cand.items()))
  log(f"choice: FZ_NEURON_TP_AR={mode} FZ_NEURON_TP_AR_KIND={kind}; per all-reduce {m1} ms at 1 row, "
      f"{m4} ms at 4 rows (x56 per token: {round(56 * (m1 or 0), 2)} ms at c1)")
  if a.choice:
    with open(a.choice, "w") as f: f.write(f"export FZ_NEURON_TP_AR={mode} FZ_NEURON_TP_AR_KIND={kind}\n")

if __name__ == "__main__": main()
