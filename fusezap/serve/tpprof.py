"""Tensor-parallel step profile (I57 M3): where a TP decode step's time goes, per rank.

  FZ_TP_PROFILE=<dir> fusezap serve ... --tp 2      # every rank appends one JSON line per forward to <dir>/rank<r>.jsonl
  python -m fusezap.serve.tpprof <dir> [--since 5]  # the table below (forwards grouped by tag, kind and row count)

A line (one forward, rank r): seq (the forward's number, the same on every rank), kind (d decode, s lookahead decode, p
prefill, m mixed), N rows, tag (set_mode's), host monotonic times (shared by the processes of one machine): rank 0
t_post0 / t_post1 around the control send, t0 / t1 around the forward's input build + replay launch; a worker t_msg
(rank 0's post time carried in the message) and t_recv; GPU ms of the forward (CUDA events around the input build and
the replay, read when done: the profiler never waits) and the GPU idle ms since the previous forward ended.

Summary columns (medians): rank 0 post us (pickle + ring write) and launch us; per worker chan us (rank 0's post ->
its receive), disp us (receive -> its replay launched), lag us (its launch done - rank 0's: how late the worker's
graph starts); GPU ms and idle ms per rank; the forward period (rank 0's t0 to the next forward's t0).

kernel_breakdown(g, n): every rank replays its captured decode forward of n rows launch by launch with CUDA events
around each item (GEMMs by projection with TB/s, attention, norms, elementwise, all-reduces): GPU time per class. The
all-reduce items then include waiting for the other rank's host (launching ~1000 items one by one), so their cost comes
from bench_allreduce (graph replays) instead. set_mode(g, mode, env, tag): every rank switches the all-reduce path
("nccl", "car", "fuse", "skip": no communication, a lower bound) and the environment (e.g. GEMM rules), then drops its
engine's captured forwards so the next steps trace and capture them again (bench/tp_step.py's A/B).

I60 (M5): FZ_TP_AR_STAMP=1 stamps the GPU clock around every all-reduce inside the captured graphs (stamp_read /
stamp_stats: each all-reduce's intrinsic in-graph cost vs the rank-skew wait; see the section below); graph_ms(g, n)
each rank's GPU ms per replay of the decode graph.
"""
from __future__ import annotations
import collections, json, os, re, statistics, sys, time
from typing import Any

def from_env(eng: Any) -> "TPProf | None":
  d = os.environ.get("FZ_TP_PROFILE", "")
  if not d or d == "0" or getattr(eng, "tp", None) is None: return None
  if d == "1": d = os.path.join(os.path.expanduser("~"), ".cache", "fusezap", "tpprof")
  return TPProf(eng, d)

class TPProf:
  def __init__(self, eng: Any, d: str):
    os.makedirs(d, exist_ok=True)
    self.g, self.rank, self.tag = eng.tp, eng.tp.rank, os.environ.get("FZ_TP_PROFILE_TAG", "")
    self.f = open(os.path.join(d, f"rank{self.rank}.jsonl"), "a", buffering=1)   # line-buffered: ranks leave with os._exit
    self.cu = self.dev = None
    if eng.model.device.split(":")[0] == "CUDA":
      from fusezap.device import Device
      self.dev = Device[eng.model.device]
      self.cu = self.dev.cu
    self.pend: collections.deque = collections.deque()
    self.prev_end: Any = None
    self.e0: Any = None
    self.t0 = 0.0
    self._free: list = []
  def _ev(self) -> Any:
    if self.cu is None: return None
    self.dev.ctx()
    e = self._free.pop() if self._free else self.cu.cuEventCreate(0)[1]
    self.cu.cuEventRecord(e, self.dev.stream)
    return e
  def _done(self, e: Any) -> bool:
    q = self.cu.cuEventQuery(e)
    q = q[0] if isinstance(q, tuple) else q
    return int(getattr(q, "value", q)) == 0
  def _ms(self, a: Any, b: Any) -> float | None:
    err, ms = self.cu.cuEventElapsedTime(a, b)
    return round(float(ms), 4) if int(getattr(err, "value", err)) == 0 else None
  def begin(self) -> None:
    self.t0 = time.monotonic()
    self.e0 = self._ev()
  def end(self, seq: int, kind: str, N: int) -> None:
    rec = {"r": self.rank, "seq": seq, "kind": kind, "N": N, "tag": self.tag, "t0": round(self.t0, 6), "t1": round(time.monotonic(), 6)}
    g = self.g
    if self.rank == 0: rec["t_post0"], rec["t_post1"] = round(g.t_post[0], 6), round(g.t_post[1], 6)
    else: rec["t_msg"], rec["t_recv"] = round(g.t_msg, 6), round(g.t_recv, 6)
    if self.cu is None: return self._write(rec)
    self.pend.append((rec, self.e0, self._ev()))
    self.poll()
  def poll(self, wait: bool = False) -> None:
    while self.pend and (wait or self._done(self.pend[0][2])):
      rec, a, b = self.pend.popleft()
      if wait: self.cu.cuEventSynchronize(b)
      rec["gpu_ms"] = self._ms(a, b)
      rec["idle_ms"] = self._ms(self.prev_end, a) if self.prev_end is not None else None
      if self.prev_end is not None: self._free.append(self.prev_end)
      self._free.append(a)
      self.prev_end = b
      self._write(rec)
  def _write(self, rec: dict) -> None: self.f.write(json.dumps(rec) + "\n")
  def flush(self) -> None:
    if self.cu is not None: self.poll(wait=True)
    self.f.flush()

# ---------------------------------------------------------------- summary

def _med(xs: list) -> float | None:
  xs = [x for x in xs if x is not None]
  return round(statistics.median(xs), 3) if xs else None

def load(d: str) -> dict[int, list[dict]]:
  out: dict[int, list[dict]] = {}
  for fn in sorted(os.listdir(d)):
    if (m := re.fullmatch(r"rank(\d+)\.jsonl", fn)):
      with open(os.path.join(d, fn)) as f: out[int(m.group(1))] = [json.loads(l) for l in f if l.strip()]
  return out

def summarize(d: str, since: float = 0.0) -> list[dict]:
  """Rows per (tag, kind, N): medians over the forwards (those starting `since` s after the first, to skip warm-up)."""
  ranks = load(d)
  r0 = ranks.get(0, [])
  if not r0: return []
  t_first = min(x["t0"] for x in r0)
  r0 = [x for x in r0 if x["t0"] - t_first >= since]
  by = {r: {x["seq"]: x for x in xs} for r, xs in ranks.items() if r}
  r0s = sorted(r0, key=lambda x: x["t0"])
  period = {x["seq"]: (y["t0"] - x["t0"]) * 1e3 for x, y in zip(r0s, r0s[1:])}
  groups: dict[tuple, list[dict]] = collections.defaultdict(list)
  for x in r0: groups[(x.get("tag", ""), "d" if x["kind"] in ("d", "s") else x["kind"], x["N"])].append(x)
  rows = []
  for (tag, kind, N), xs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
    row: dict[str, Any] = {"tag": tag, "kind": kind, "N": N, "n": len(xs), "lookahead": round(sum(x["kind"] == "s" for x in xs) / len(xs), 3),
                           "post_us": _med([(x["t_post1"] - x["t_post0"]) * 1e6 for x in xs]),
                           "launch_us": _med([(x["t1"] - x["t0"]) * 1e6 for x in xs]),
                           "period_ms": _med([period.get(x["seq"]) for x in xs]),
                           "gpu_ms": {0: _med([x.get("gpu_ms") for x in xs])}, "idle_ms": {0: _med([x.get("idle_ms") for x in xs])}}
    for r, m in by.items():
      ws = [(x, m[x["seq"]]) for x in xs if x["seq"] in m]
      row.setdefault("chan_us", {})[r] = _med([(w["t_recv"] - w["t_msg"]) * 1e6 for _, w in ws])
      row.setdefault("disp_us", {})[r] = _med([(w["t1"] - w["t_recv"]) * 1e6 for _, w in ws])
      row.setdefault("lag_us", {})[r] = _med([(w["t1"] - x["t1"]) * 1e6 for x, w in ws])
      row["gpu_ms"][r] = _med([w.get("gpu_ms") for _, w in ws])
      row["idle_ms"][r] = _med([w.get("idle_ms") for _, w in ws])
    rows.append(row)
  return rows

def print_summary(rows: list[dict], out: Any = sys.stdout) -> None:
  fmt = lambda v: "-" if v is None else (" / ".join("-" if x is None else f"{x:g}" for x in v.values()) if isinstance(v, dict) else f"{v:g}")
  print("tag kind N n lookahead | r0 post_us launch_us period_ms | workers chan_us disp_us lag_us | gpu_ms (r0 / r1 ..) idle_ms", file=out)
  for r in rows:
    print(f"{r['tag'] or '-'} {r['kind']} {r['N']} {r['n']} {r['lookahead']} | {fmt(r['post_us'])} {fmt(r['launch_us'])} {fmt(r['period_ms'])} | "
          f"{fmt(r.get('chan_us'))} {fmt(r.get('disp_us'))} {fmt(r.get('lag_us'))} | {fmt(r['gpu_ms'])} {fmt(r['idle_ms'])}", file=out)

# ---------------------------------------------------------------- A/B and kernel breakdown (run on every rank)

def _engine(g: Any) -> Any:
  """The rank's LLMEngine: a worker's own, rank 0's as the caller registered it (g.engine)."""
  e = g.worker.eng if g.worker is not None else getattr(g, "engine", None)
  assert e is not None, "no engine on this rank (rank 0: set g.engine)"
  return e

MODES = {"nccl": ("base", False), "car": ("auto", False), "fuse": ("auto", True), "skip": ("skip", False)}

def set_mode(g: Any, mode: str, env: dict | None = None, tag: str = "", scope: str = "all", eng: Any = None) -> dict:
  """Every rank: the all-reduce path (MODES), env updates (None values unset), then the engine's forwards (scope "all",
  or "decode": the decode forwards only) are captured again on their next call. Call while the engine is idle (no
  pending lookahead)."""
  from fusezap.helpers import getenv
  eng = eng if eng is not None else _engine(g)
  g.comm.mode, g.comm.fuse = MODES[mode]
  for k, v in (env or {}).items():
    if v is None: os.environ.pop(k, None)
    else: os.environ[k] = str(v)
  getenv.cache_clear()
  for f in eng._by_key.values():
    if scope == "all" or f in eng._decodes.values(): f.jit.reset()
  eng._seen.clear()
  if (p := getattr(eng, "_tpprof", None)) is not None: p.flush(); p.tag = tag
  return {"rank": g.rank, "mode": mode, "car": g.comm.car is not None}

_GEMM_RE = re.compile(r"gemm|wgswap|mma|cublas", re.I)
_CLASS = [("allreduce", r"all_reduce|car"), ("kv_write", r"kv_write"), ("attention", r"attn|attention|paged|flash|combine|decode"),
          ("norm", r"rms|norm"), ("rope", r"rope"), ("act", r"silu"), ("argmax", r"argmax"), ("inputs", r"step_inputs|gather|embed")]

def _label(it: Any, cfg: Any) -> tuple[str, int]:
  """(class or gemm:<projection>, weight bytes for a GEMM) of a captured item."""
  if it.prg is not None: name = it.prg.name
  else:
    a = it.si.ast[0] if it.si is not None and it.si.ast else None
    name = getattr(a, "name", type(a).__name__) if a is not None else "?"
    if a is not None and hasattr(a, "N") and hasattr(a, "K") and hasattr(a, "M") and _GEMM_RE.search(name):
      N, K, hd = a.N, a.K, cfg.head_dim
      proj = ("qkv" if N == (cfg.n_heads + 2 * cfg.n_kv_heads) * hd else "o" if K == cfg.n_heads * hd and N == cfg.hidden_size else
              "gate_up" if N == 2 * cfg.intermediate_size else "down" if K == cfg.intermediate_size else
              "lm_head" if N == cfg.vocab_size else f"{N}x{K}")
      return f"gemm:{proj}", N * K * 2
  n = name.lower()
  return next((c for c, p in _CLASS if re.search(p, n)), "elementwise" if re.match(r"^[er]_", n) else f"other:{name[:40]}"), 0

def kernel_breakdown(g: Any, n: int, eng: Any = None, reps: int = 3) -> dict:
  """Every rank: GPU ms per item class of the captured decode forward of n rows (launch by launch, CUDA events per item;
  the inputs of its last replay, so it writes the same KV-cache slots with the same values). Engine idle; CUDA only."""
  from fusezap.ir import ScheduleItem
  from fusezap import jit as J
  from fusezap.shape.symbolic import sym_infer, Node
  eng = eng if eng is not None else _engine(g)
  f = eng._decode_fwd(n)
  c = f.jit.captured
  if c is None or c.gstage is None: return {"rank": g.rank, "error": f"decode forward {n} not captured as a graph yet"}
  dev = c.gdev
  cu = dev.cu
  pools = iter([t._realized_contiguous().realized for t in eng.cache.state()])   # the written inputs: the KV pools
  ins = [s if s is not None else next(pools) for s in c.gstage]                  # the rest: their staging copies
  res = lambda r: c._res(r, ins, c.gowned)
  evs: list = []
  def ev() -> Any:
    e = cu.cuEventCreate(0)[1]
    cu.cuEventRecord(e, dev.stream)
    return e
  tot: dict[str, float] = collections.defaultdict(float)
  gb: dict[str, float] = collections.defaultdict(float)
  cnt: dict[str, int] = collections.defaultdict(int)
  dev.synchronize()
  for rep in range(reps):
    evs = []
    for it in c.items:
      if it.const: continue
      bufs = [res(r) for r in it.slots]
      e0 = ev()
      if it.prg is not None:
        gs = it.gs if it.static else tuple(int(sym_infer(s, {})) for s in it.gs)
        it.prg(*[b.handle for b in bufs[:it.nb]], global_size=gs, local_size=it.ls, vals=())
      else:
        si = it.si
        J._run(it.dev, ScheduleItem(si.ast, tuple(bufs), {}, si.kind, si.n_outputs))
      evs.append((it, e0, ev()))
    dev.synchronize()
    if rep == 0: continue   # the first pass binds launchers
    for it, a, b in evs:
      lab, nb = _label(it, eng.model.cfg)
      ms = float(cu.cuEventElapsedTime(a, b)[1])
      tot[lab] += ms / (reps - 1); gb[lab] += nb / 1e9 / (reps - 1); cnt[lab] += 1
      cu.cuEventDestroy(a); cu.cuEventDestroy(b)
  out = {"rank": g.rank, "n": n, "items": sum(cnt.values()) // max(1, reps - 1), "total_ms": round(sum(tot.values()), 3),
         "classes": {k: round(v, 4) for k, v in sorted(tot.items(), key=lambda kv: -kv[1])},
         "gemm_TBps": {k: round(gb[k] / (tot[k] / 1e3) / 1e3, 3) for k in tot if k.startswith("gemm:") and tot[k] > 0},
         "count": {k: v // max(1, reps - 1) for k, v in cnt.items()}}
  return out

def graph_ms(g: Any, n: int, reps: int = 20, eng: Any = None) -> dict:
  """Every rank: GPU ms of one replay of the captured decode graph of n rows (reps back-to-back replays between two
  CUDA events, after one warm replay; every rank replays the same count, so the all-reduces pair up). Engine idle."""
  eng = eng if eng is not None else _engine(g)
  c = eng._decode_fwd(n).jit.captured
  if c is None or c.graph is None or getattr(c.gdev, "cu", None) is None: return {"rank": g.rank, "ms": None}
  dev, cu = c.gdev, c.gdev.cu
  dev.synchronize()
  c.graph.launch()
  dev.synchronize()
  e0, e1 = cu.cuEventCreate(0)[1], cu.cuEventCreate(0)[1]
  cu.cuEventRecord(e0, dev.stream)
  for _ in range(reps): c.graph.launch()
  cu.cuEventRecord(e1, dev.stream)
  dev.synchronize()
  ms = float(cu.cuEventElapsedTime(e0, e1)[1]) / reps
  cu.cuEventDestroy(e0); cu.cuEventDestroy(e1)
  return {"rank": g.rank, "ms": round(ms, 4)}

# ---------------------------------------------------------------- all-reduce stamps (I60 M5)
# FZ_TP_AR_STAMP=1 (read while a forward is traced): a 1-thread kernel writes the GPU clock (%globaltimer, ns) before
# and after every all-reduce (dist.all_reduce / all_reduce_add_norm) and at the start of every forward_tokens. The
# forward's first stamp ("begin") takes the next slot of a 16-forward ring (its time, its own id as the forward's tag,
# the forward's sequence number), the all-reduce stamps write into that slot at their id, so the last 16 forwards of a
# graph replay (or of a launch-by-launch one, FZ_GRAPH=0) are kept on the device without any host wait. stamp_read (every
# rank, engine idle) returns the slots of one decode forward; stamp_stats lines the ranks up: per all-reduce each rank's
# duration (post - pre: its own clock), the clock offset between the GPUs (the median of post_r - post_0: an all-reduce
# ends on every rank within ~1-2 us), so each all-reduce splits into the time the earliest rank waited for the latest
# (arrival skew: rank skew) and the latest rank's own duration (the all-reduce's intrinsic in-graph cost); the compute
# between consecutive all-reduces per rank; the forward period and each rank's graph start. Each stamp costs a launch
# (~1-2 us on the GPU: ~0.5 ms per 70B step): compare a stamped variant's segments with a stamped skip variant.

STAMP_R, STAMP_S = 16, 1 << 15   # ring slots (forwards), stamp ids (wrap; a 70B forward has 321)

def stamps_on() -> bool: return os.environ.get("FZ_TP_AR_STAMP", "") not in ("", "0")

_STAMP_SRC = r"""#define R %d
#define S %d
extern "C" __global__ void fz_ar_stamp(long long* buf, long long* meta, int* cnt, const int id, const int begin) {
  unsigned long long t;
  asm volatile("mov.u64 %%0, %%%%globaltimer;" : "=l"(t));
  if (threadIdx.x != 0) return;
  if (begin) { int c = cnt[0], s = c %% R; meta[s * 4] = (long long)t; meta[s * 4 + 1] = id; meta[s * 4 + 2] = c; cnt[0] = c + 1; }
  else { int s = (cnt[0] + R - 1) %% R; buf[(long long)s * S + id] = (long long)t; }
}""" % (STAMP_R, STAMP_S)

class _Ring:
  """A device's stamp ring: buf (R x S times), meta (R x 4: begin time, tag, sequence), cnt (forwards begun). CUDA:
  device buffers written by fz_ar_stamp; elsewhere host arrays written at launch (host clock: tests of the plumbing)."""
  def __init__(self, device: str):
    from fusezap.device import Buffer
    from fusezap.dtype import dtypes
    import numpy as np
    self.device, self.next = device, 0
    self.cuda = device.split(":")[0].upper() == "CUDA"
    z = lambda n, dt, npdt: Buffer(device, n, dt).ensure_allocated().copyin(memoryview(np.zeros(n, npdt)).cast("B"))
    if self.cuda: self.buf, self.meta, self.cnt = z(STAMP_R * STAMP_S, dtypes.int64, np.int64), z(STAMP_R * 4, dtypes.int64, np.int64), z(2, dtypes.int32, np.int32)
    else: self.hbuf, self.hmeta, self.hcnt = np.zeros(STAMP_R * STAMP_S, np.int64), np.zeros(STAMP_R * 4, np.int64), 0
  def read(self) -> tuple[Any, Any]:
    import numpy as np
    if not self.cuda: return self.hbuf.reshape(STAMP_R, STAMP_S).copy(), self.hmeta.reshape(STAMP_R, 4).copy()
    from fusezap.device import Device
    Device[self.device].synchronize()
    b, m = np.empty(STAMP_R * STAMP_S, np.int64), np.empty(STAMP_R * 4, np.int64)
    self.buf.copyout(memoryview(b).cast("B")); self.meta.copyout(memoryview(m).cast("B"))
    return b.reshape(STAMP_R, STAMP_S), m.reshape(STAMP_R, 4)

_RINGS: dict[str, _Ring] = {}
def ring(device: str) -> _Ring:
  if device not in _RINGS: _RINGS[device] = _Ring(device)
  return _RINGS[device]

class StampCall:
  """Payload of a kind="custom" item (bufs: a 1-element dummy output and the buffer the stamp follows, so the item is
  ordered after its producer and never constant-folded by fz.jit); launch writes the clock into the ring."""
  def __init__(self, rg: _Ring, sid: int, kind: str): self.ring, self.sid, self.kind = rg, sid, kind
  @property
  def kernel(self) -> "StampCall": return self
  @property
  def name(self) -> str: return "fz_ar_stamp"
  def launch(self, dev: Any, bufs: Any, var_vals: dict | None = None) -> None:
    r, b = self.ring, int(self.kind == "begin")
    if r.cuda:
      from fusezap.serve.cuda import CudaCall
      return CudaCall("fz_ar_stamp", _STAMP_SRC, (self.sid, b), (1, 1, 1), (32, 1, 1)).launch(dev, [r.buf, r.meta, r.cnt])
    t = time.perf_counter_ns()
    if b: s = r.hcnt % STAMP_R; r.hmeta[s * 4:s * 4 + 3] = (t, self.sid, r.hcnt); r.hcnt += 1
    else: r.hbuf[((r.hcnt + STAMP_R - 1) % STAMP_R) * STAMP_S + self.sid] = t
  def pretty(self, indent: int = 0) -> str: return f"{'  ' * indent}STAMP {self.kind} {self.sid}"

def stamp(src: Any, kind: str) -> None:
  """A stamp item after buffer src's producer (kind: begin / pre / post), run now and recorded by a fz.jit capture."""
  from fusezap.device import Buffer
  from fusezap.dtype import dtypes
  from fusezap.ir import ScheduleItem
  from fusezap import jit as J
  r = ring(src.device)
  sid, r.next = r.next, (r.next + 1) % STAMP_S
  J.run_item(ScheduleItem((StampCall(r, sid, kind),), (Buffer(src.device, 1, dtypes.int32), src), {}, "custom", 1))   # type: ignore[arg-type]

def stamp_read(g: Any, n: int, eng: Any = None) -> dict:
  """Every rank (engine idle): the ring slots holding the decode forward of n rows: per forward its sequence number,
  begin time and the pre / post time of each all-reduce (ns, this GPU's clock), in the forward's order."""
  eng = eng if eng is not None else _engine(g)
  c = eng._decode_fwd(n).jit.captured
  if c is None: return {"rank": g.rank, "error": f"decode forward {n} not captured"}
  sts = [it.si.ast[0] for it in c.items if it.si is not None and it.si.ast and isinstance(it.si.ast[0], StampCall)]
  beg = [s for s in sts if s.kind == "begin"]
  if not beg: return {"rank": g.rank, "error": "no stamps in the forward (FZ_TP_AR_STAMP=1 while it was traced?)"}
  pre, post = [s.sid for s in sts if s.kind == "pre"], [s.sid for s in sts if s.kind == "post"]
  buf, meta = beg[0].ring.read()
  fw = [{"seq": int(meta[s, 2]), "t0": int(meta[s, 0]), "pre": [int(buf[s, i]) for i in pre], "post": [int(buf[s, i]) for i in post]}
        for s in range(STAMP_R) if int(meta[s, 1]) == beg[0].sid and meta[s, 0] > 0]
  return {"rank": g.rank, "n": n, "ars": len(pre), "fwd": sorted(fw, key=lambda f: f["seq"])}

def stamp_stats(rs: list[dict]) -> dict:
  """Lines up stamp_read's per-rank results (see the section comment). Times in us per all-reduce and ms per forward
  (medians over the forwards every rank holds); 'late' counts which rank arrived last at each all-reduce."""
  import numpy as np
  if any("error" in r for r in rs): return {"error": next(r["error"] for r in rs if "error" in r)}
  by = [{f["seq"]: f for f in r["fwd"]} for r in rs]
  seqs = sorted(set.intersection(*[set(b) for b in by]))
  if not seqs: return {"error": "no forward held by every rank"}
  W = len(rs)
  P = np.array([[by[r][s]["pre"] for s in seqs] for r in range(W)], np.float64) / 1e3    # (W, F, J) us
  Q = np.array([[by[r][s]["post"] for s in seqs] for r in range(W)], np.float64) / 1e3
  T0 = np.array([[by[r][s]["t0"] for s in seqs] for r in range(W)], np.float64) / 1e3     # (W, F)
  if P.shape[2] == 0: return {"error": "no all-reduce stamps"}
  off = np.median((Q - Q[0:1]).reshape(W, -1), axis=1)                                    # GPU r's clock - GPU 0's
  Pa, T0a = P - off[:, None, None], T0 - off[:, None]
  dur = Q - P
  wait = Pa.max(0) - Pa.min(0)                                                            # (F, J) the earliest rank's wait
  intr = dur.min(0)                                                                       # the latest arriver's duration
  seg = np.concatenate([(P[:, :, :1] - T0[:, :, None]), P[:, :, 1:] - Q[:, :, :-1]], axis=2)   # compute before each all-reduce
  late = np.bincount(Pa.argmax(0).ravel(), minlength=W)
  nx = [k for k in range(len(seqs) - 1) if seqs[k + 1] == seqs[k] + 1]                    # consecutive forwards
  per = (T0[:, [k + 1 for k in nx]] - T0[:, nx]) if nx else None
  tail = (T0[:, [k + 1 for k in nx]] - Q[:, nx, -1]) if nx else None                     # last all-reduce -> next begin
  med = lambda x: round(float(np.median(x)), 3)
  J = P.shape[2]
  out = {"forwards": len(seqs), "ars": J, "clock_offset_us": [round(float(o), 2) for o in off],
         "period_ms": [round(med(per[r]) / 1e3, 3) for r in range(W)] if per is not None else None,
         "ar_ms": [round(med(dur[r].sum(1)) / 1e3, 3) for r in range(W)], "intrinsic_ms": round(med(intr.sum(1)) / 1e3, 3),
         "wait_ms": round(med(wait.sum(1)) / 1e3, 3), "compute_ms": [round(med(seg[r].sum(1)) / 1e3, 3) for r in range(W)],
         "compute_crit_ms": round(med(seg.max(0).sum(1)) / 1e3, 3), "tail_ms": [round(med(tail[r]) / 1e3, 3) for r in range(W)] if tail is not None else None,
         "intrinsic_us": med(intr), "intrinsic_us_p90": round(float(np.percentile(intr, 90)), 3), "wait_us": med(wait),
         "wait_us_p90": round(float(np.percentile(wait, 90)), 3), "dur_us": [med(dur[r]) for r in range(W)],
         "begin_skew_us": [med(T0a[r] - T0a[0]) for r in range(W)], "late": [int(x) for x in late],
         "by_parity": {k: {"intrinsic_us": med(intr[:, i::2]), "wait_us": med(wait[:, i::2]), "seg_us": [med(seg[r][:, i::2]) for r in range(W)],
                           "late": [int(x) for x in np.bincount(Pa[:, :, i::2].argmax(0).ravel(), minlength=W)]}
                       for i, k in ((0, "attn_o"), (1, "mlp_down")) if J > i}}
  return out

def main() -> None:
  import argparse
  ap = argparse.ArgumentParser(description="Summarize FZ_TP_PROFILE files")
  ap.add_argument("dir")
  ap.add_argument("--since", type=float, default=0.0, help="skip forwards that start within this many seconds of the first")
  ap.add_argument("--json", default=None)
  a = ap.parse_args()
  rows = summarize(a.dir, a.since)
  print_summary(rows)
  if a.json:
    with open(a.json, "w") as f: json.dump(rows, f, indent=1)

if __name__ == "__main__": main()
