"""NKI-N6: Qwen2.5-0.5B-Instruct served by LLMEngine on a NEURON tier, where the decode time goes, and the token check against
the float32 reference (docs/invarix/examples/refs/qwen2.5-0.5b-instruct-f32.json), without the HTTP layer:

  FZ_DEVICE=NEURON uv run python scripts/nki_n6_serve.py --out results/nki_n6 --variant all              # inf2 (scripts/nki_n6_inf2.sh)
  FZ_NEURON_FUSE=0 FZ_NEURON_CTX=0 ... scripts/nki_n6_serve.py --quick --variant n5                        # one before/after variant
  FZ_DEVICE=NKISHIM uv run python scripts/nki_n6_serve.py --quick --c1-tokens 8 --c4-tokens 6 --profile-steps 2   # Mac dry run

  launches   kernels per decode token: the non-constant items of each captured forward (one launch each; N6 fuses a layer
             into one), by kind, for every pool prefix the forward was captured at
  c1         one request alone: TTFT, per decode step wall time (TPOT median after the capture steps)
  c4         four requests together (one 4-row decode bucket): per step wall time and tokens/s
  profile    NEURON: the last --profile-steps decode steps of c1 again with FZ_NEURON_PROFILE: per kernel the launches per
             token, the runtime worker's execute time (nrt execute = device time + the runtime's own cost per call) and
             nrtpy's device-time benchmark of one launch; per token: step wall, worker batch wall, sum of executes, sum of
             device times, so step - batch = fusezap host side, batch - executes = worker side, executes - device = runtime;
             N7: the host side by piece (host_split): the engine step's phases (sched, prep incl. step metadata and the
             input uploads, replay = the jit call, post = token read-back + emit), the runtime-worker round trips per
             token, the pipe's share of them (wall in the round trips - the worker's own time) and the Python share of the
             step (step - wall in round trips)
  ttft       I75 (--ttft-runs k): k prompts alone, the time to each one's first token (c1 TTFT p50 / p99); --burst-runs b: b
             times 4 prompts added at once (the c4 burst a closed-loop client sends), each one's time to its first token
             (c4 TTFT p50 / p99: the prompts' prefills queue behind each other); --prefill-profile: one prompt's prefill step
             with FZ_NEURON_PROFILE (per kernel launches and execute ms, the step wall). I76 --ttft-lens 0,128: the same
             at each prompt length (0: the reference prompts as they are, 37-66 tokens, so three share a 128-row prefill
             forward; 128: each prompt repeated to exactly 128 tokens, Invarix short's prompts: one forward each)
  greedy     (not --quick) every prompt continued greedily, max_num_seqs at a time, vs the reference's greedy tokens
  top1       (not --quick) teacher-forced top-1 and |logprob error| on --top1-prompts prompts
The env switches in effect (FZ_NEURON_FUSE, FZ_NEURON_CTX, FZ_NEURON_KT, FZ_NEURON_AT_ROWS, FZ_NKI_FAST_EXP) are recorded.
Results: <out>/serve_<variant>.json and a summary on stdout.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusezap.device import Device
from fusezap.dtype import dtypes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWITCHES = ("FZ_NEURON_DTYPE", "FZ_NEURON_BF16_ATTN", "FZ_NEURON_FUSE", "FZ_NEURON_FUSE_MAX", "FZ_NEURON_CTX", "FZ_NEURON_CTX_MIN", "FZ_NEURON_KT", "FZ_NEURON_AT_ROWS",
            "FZ_NKI_FAST_EXP", "FZ_NEURON_KVW", "FZ_NEURON_NORM", "FZ_NEURON_BATCH", "FZ_NEURON_HOSTFAST", "FZ_NEURON_DEVICE_ARGMAX",
            "FZ_NEURON_QWRITE_MAX", "FZ_NEURON_HEAD", "FZ_NEURON_HEAD_W", "FZ_NEURON_HEAD_DMAS", "FZ_NEURON_LAYER", "FZ_NEURON_LAYER_HEADS",
            "FZ_NEURON_LAYER_ROPE", "FZ_NEURON_LOW_BLOCKS", "FZ_NEURON_LAYER_MM", "FZ_NEURON_LAYER_EARLY", "FZ_NEURON_LAYER_PF", "FZ_NEURON_HEAD_FUSE",
            "FZ_NEURON_TP_AR", "FZ_NEURON_TP_AR_KIND", "FZ_NEURON_TP_AR_FUSE", "FZ_NEURON_TP_HEAD",   # N8: tensor parallelism (--tp)
            "FZ_NEURON_LAYER_TILE", "FZ_NEURON_PLAN", "FZ_NEURON_HBM_GB", "FZ_NEURON_ALLOC_CACHE_MB", "FZ_LOGPROBS_JITS",
            "FZ_NEURON_BF16_WN", "FZ_NEURON_BF16_PF", "FZ_NEURON_BF16_PFQO", "FZ_NEURON_BF16_HEAD", "FZ_NEURON_BF16_HEAD_W",   # N8 (I70)
            "FZ_NEURON_TP_LAYER", "FZ_NEURON_TP_AR_BIG", "FZ_NEURON_TP_PREFILL", "FZ_NEURON_TP_PREFILL_FUSE", "FZ_NEURON_PREFILL_CTX", "FZ_PREFILL_FIRST",   # I73 / I75
            "FZ_NEURON_PREFILL_KERNEL", "FZ_NEURON_PFK_EXP", "FZ_NEURON_PFK_HEAD", "FZ_NEURON_PFK_BANKS", "FZ_NEURON_PFK_WN", "FZ_NEURON_PFK_EVICT",
            "FZ_NEURON_PREFILL_SIZES", "FZ_ADMIT_WAIT_MS", "FZ_NEURON_PREFILL_COST_MS", "FZ_PREFILL_FIT")   # I77
TP: list = []   # N8 --tp N: the tensor-parallel group (fusezap.serve.dist), rank 0 here
OUT: dict = {}

def log(*a) -> None: print(*a, flush=True)

def _kind(it) -> str:
  if it.kind != "custom" or it.si is None: return it.kind
  return it.si.ast[0].kernel.name.split("_")[1]

def launches(eng) -> dict:
  out = {}
  for nm, fs in (("decode", eng._decodes), ("prefill", eng._prefills)):
    for n, f in fs.items():
      for ctx, j in [(None, f.jit), *sorted(f.ctx_jits.items())]:
        if (c := j.captured) is None: continue
        items = [it for it in c.items if not it.const]
        out[f"{nm}_{n}" + (f"_ctx{ctx}" if ctx else "")] = {"launches": len(items), "constant": len(c.items) - len(items),
                                                            "by_kind": dict(Counter(_kind(it) for it in items))}
  return out

def alloc():
  return getattr(Device[Device.DEFAULT], "allocator", None)

def worker_calls() -> dict: return dict(getattr(alloc(), "calls", {}) or {})

def hbm(eng, phase: str) -> None:
  """N8 (I71): what holds the device memory after `phase` (neuron_kv.hbm_report), logged and kept in OUT["hbm"]."""
  from fusezap.serve.neuron_kv import hbm_line, hbm_report
  try:
    OUT.setdefault("hbm", {})[phase] = hbm_report(eng)
    log(f"[{OUT.get('variant')}] {hbm_line(eng, phase)}")
  except Exception as e: log(f"[{OUT.get('variant')}] hbm report failed: {type(e).__name__}: {e}")

class HostProf:
  """N7: LLMEngine.step's phase marks (the FZ_STEP_PROF hooks) summed over the profiled steps, in ms; meta: the step
  metadata (NeuronKVCache.step_meta*) inside prep."""
  def __init__(self, eng):
    self.sum: dict = {}
    self.steps, self._t = 0, 0.0
    c = eng.cache
    for nm in ("step_meta", "step_meta_rows"):
      f = getattr(c, nm)
      def timed(*a, _f=f, **k):
        st = time.perf_counter()
        try: return _f(*a, **k)
        finally: self.sum["meta"] = self.sum.get("meta", 0.0) + 1e3 * (time.perf_counter() - st)
      setattr(c, nm, timed)
    self.cache = c
  def uninstall(self) -> None:
    for nm in ("step_meta", "step_meta_rows"): self.cache.__dict__.pop(nm, None)
  def begin(self) -> None: self._t = time.perf_counter()
  def mark(self, name: str) -> None:
    t = time.perf_counter()
    key = name[2:] if name[:2] in ("d_", "s_", "p_") else name   # decode / lookahead / prefill phases summed by phase
    self.sum[key] = self.sum.get(key, 0.0) + 1e3 * (t - self._t)
    self._t = t
  def fwd(self, kind: str, which: int) -> None: pass
  def end(self, rows_d: int, rows_p: int, size_p: int) -> None:
    self.mark("post"); self.steps += 1

def run_alone(eng, ids, n_tokens: int, profile_steps: int = 0) -> tuple[dict, list[int]]:
  """One request: TTFT, the decode step times (the last profile_steps with the per-kernel profile on)."""
  from fusezap.serve.engine import SamplingParams
  rid = eng.add_request(ids, SamplingParams(max_tokens=n_tokens, ignore_eos=True))
  steps, psteps, toks, c0, t_first = [], [], [], worker_calls(), None
  al, t0, hp, pf0 = alloc(), time.perf_counter(), None, eng._prof
  while eng.has_unfinished():
    if profile_steps and al is not None and hasattr(al, "profile") and len(toks) == n_tokens - profile_steps:
      al.flush(); al.profile, al.prof, al.prof_wall, al.prof_ipc = True, {}, [0, 0.0], [0, 0.0, 0.0]
      eng._prof = hp = HostProf(eng)
    st = time.perf_counter()
    outs = eng.step()
    dt = time.perf_counter() - st
    for o in outs:
      if o.request_id == rid and o.new_token_ids:
        if t_first is None: t_first = time.perf_counter() - t0
        else: (psteps if getattr(al, "profile", False) else steps).append(dt)
        toks += o.new_token_ids
  eng.release(rid)
  if al is not None and hasattr(al, "profile"): al.flush(); al.profile = False
  if hp is not None: eng._prof = pf0; hp.uninstall()
  warm = steps[3:] if len(steps) > 6 else steps   # the first decode steps trace and capture the jitted forward
  r = {"ttft_s": round(t_first or 0, 3), "decode_steps": len(steps), "tpot_ms_median": round(1e3 * float(np.median(warm)), 2) if warm else None,
       "tpot_ms_p10": round(1e3 * float(np.percentile(warm, 10)), 2) if warm else None,
       "tpot_ms_p90": round(1e3 * float(np.percentile(warm, 90)), 2) if warm else None,
       "first_steps_s": [round(x, 3) for x in steps[:3]], "worker_calls": {k: v - c0.get(k, 0) for k, v in worker_calls().items()},
       "profile_steps_ms": [round(1e3 * x, 2) for x in psteps]}
  if hp is not None and hp.steps:
    r["host_phases_ms"] = {k: round(v / hp.steps, 3) for k, v in hp.sum.items()}
    if (ipc := getattr(al, "prof_ipc", None)) is not None: r["ipc"] = {"trips": ipc[0], "wall_ms": round(1e3 * ipc[1], 3), "worker_ms": round(1e3 * ipc[2], 3)}
  return r, toks

def run_together(eng, prompts: list, n_tokens: int) -> dict:
  """len(prompts) requests at once: the step times while all of them decode (one decode bucket), tokens/s."""
  from fusezap.serve.engine import SamplingParams
  rids = [eng.add_request(p, SamplingParams(max_tokens=n_tokens, ignore_eos=True)) for p in prompts]
  steps, t0, ntok = [], time.perf_counter(), 0
  while eng.has_unfinished():
    st = time.perf_counter()
    outs = eng.step()
    dt = time.perf_counter() - st
    got = sum(len(o.new_token_ids) for o in outs)
    ntok += got
    if got == len(prompts) and all(not eng.seqs[r].in_prompt for r in rids): steps.append(dt)
  wall = time.perf_counter() - t0
  for r in rids: eng.release(r)
  warm = steps[3:] if len(steps) > 6 else steps
  return {"requests": len(prompts), "decode_steps": len(steps), "tpot_ms_median": round(1e3 * float(np.median(warm)), 2) if warm else None,
          "tok_s_decode": round(len(prompts) / float(np.median(warm)), 1) if warm else None, "wall_s": round(wall, 2), "tokens": ntok}

def profile_report(c1: dict, eng) -> dict | None:
  """The per-kernel breakdown of the profiled decode steps (NEURON's persistent path only)."""
  al = alloc()
  if al is None or not hasattr(al, "prof") or not al.prof: return None
  from fusezap.device.neuron import neuron_profile
  rep = neuron_profile(al, iters=20, bench=not TP)   # N8: no device-time rerun under TP (a collective kernel alone would wait for its peer)
  n = max(len(c1.get("profile_steps_ms") or []), 1)
  ks = rep["kernels"]
  per_tok = lambda f: round(sum(f(v) for v in ks.values()) / n, 3)
  rep["per_token"] = {"steps": n, "step_ms": round(float(np.mean(c1["profile_steps_ms"])), 3) if c1.get("profile_steps_ms") else None,
                      "launches": round(sum(v["launches"] for v in ks.values()) / n, 1),
                      "worker_batch_ms": round(rep["worker_batch_ms"] / n, 3),
                      "execute_ms": per_tok(lambda v: v["exec_ms_total"]),
                      "device_ms": per_tok(lambda v: (v["device_ms"] or 0.0) * v["launches"])}
  pt = rep["per_token"]
  if pt["step_ms"] is not None:
    pt["host_ms"] = round(pt["step_ms"] - pt["worker_batch_ms"], 3)          # fusezap: replay, requests, pipe, scheduler
    pt["worker_ms"] = round(pt["worker_batch_ms"] - pt["execute_ms"], 3)     # the runtime worker around the executes
    pt["runtime_ms"] = round(pt["execute_ms"] - pt["device_ms"], 3)          # nrt execute beyond the device time
    hs = dict(c1.get("host_phases_ms") or {})
    if (ipc := c1.get("ipc")) is not None:   # N7: the host side by piece
      hs["trips"] = round(ipc["trips"] / n, 2)
      hs["ipc_wall_ms"] = round(ipc["wall_ms"] / n, 3)                         # all round trips, incl. the worker's time
      hs["pipe_ms"] = round((ipc["wall_ms"] - ipc["worker_ms"]) / n, 3)        # pickling, pipe, wake-ups
      hs["worker_other_ms"] = round(ipc["worker_ms"] / n - pt["worker_batch_ms"], 3)   # the worker outside batches (lone writes / reads)
      hs["python_ms"] = round(pt["step_ms"] - ipc["wall_ms"] / n, 3)           # fusezap's own Python per token
    pt["host_split"] = hs
  return rep

def _pct(xs: list, q: float) -> float | None: return round(1e3 * float(np.percentile(xs, q)), 2) if xs else None

def ttft(eng, prompts: list, runs: int, bursts: int) -> dict:
  """I75: c1 TTFT (each prompt alone, its first token) over `runs` prompts, and c4 TTFT over `bursts` bursts of 4 prompts
  added together (each one's time from the burst to its first token), in ms; the prompts' first tokens."""
  from fusezap.serve.engine import SamplingParams
  out: dict = {}
  c1, ids1 = [], []
  for k in range(runs):
    p = prompts[k % len(prompts)]
    rid, t0, first = eng.add_request(p, SamplingParams(max_tokens=2, ignore_eos=True)), time.perf_counter(), None
    while eng.has_unfinished():
      for o in eng.step():
        if o.request_id == rid and o.new_token_ids and first is None: first = time.perf_counter() - t0; ids1.append(o.new_token_ids[0])
    eng.release(rid)
    c1.append(first or 0.0)
  c4, ids4 = [], []
  for b in range(bursts):
    ps = [prompts[(4 * b + i) % len(prompts)] for i in range(4)]
    t0 = time.perf_counter()
    rids = [eng.add_request(p, SamplingParams(max_tokens=2, ignore_eos=True)) for p in ps]
    first: dict = {}
    while eng.has_unfinished():
      for o in eng.step():
        if o.new_token_ids and o.request_id not in first: first[o.request_id] = (time.perf_counter() - t0, o.new_token_ids[0])
    for r in rids: eng.release(r)
    c4 += [first[r][0] for r in rids]; ids4 += [first[r][1] for r in rids]
  out["c1_ms"], out["c4_ms"] = [round(1e3 * x, 2) for x in c1], [round(1e3 * x, 2) for x in c4]
  out.update(c1_p50=_pct(c1, 50), c1_p99=_pct(c1, 99), c4_p50=_pct(c4, 50), c4_p99=_pct(c4, 99), c1_ids=ids1, c4_ids=ids4)
  return out

def prefill_profile(eng, prompt: list) -> dict | None:
  """I75: one prompt's prefill step with FZ_NEURON_PROFILE on: per kernel launches and execute ms (no device rerun under
  TP), the step wall, the runtime-worker round trips."""
  from fusezap.serve.engine import SamplingParams
  al = alloc()
  if al is None or not hasattr(al, "profile"): return None
  from fusezap.device.neuron import neuron_profile
  rid = eng.add_request(prompt, SamplingParams(max_tokens=2, ignore_eos=True))
  al.flush(); al.profile, al.prof, al.prof_wall, al.prof_ipc = True, {}, [0, 0.0], [0, 0.0, 0.0]
  t0, trips0 = time.perf_counter(), al.trips
  eng.step()   # the prefill (a fresh request: its prompt in one chunk)
  wall = time.perf_counter() - t0
  al.flush(); al.profile = False
  rep = neuron_profile(al, iters=5, bench=not TP)
  rep["step_ms"], rep["trips"] = round(1e3 * wall, 3), al.trips - trips0
  rep["launches"] = sum(v["launches"] for v in rep["kernels"].values())
  while eng.has_unfinished(): eng.step()
  eng.release(rid)
  return rep

def prefill_bench(eng, reps: int) -> dict:
  """I77: every prefill forward at each pool prefix the engine prewarms (neuron_prewarm.plan), on its synthetic step (one
  sequence of min(rows, prefix) tokens whose blocks end at the prefix): `reps` timed replays (wall to the token read back:
  the device time plus ~4 ms of host and round trips, I75), and the step's first logits row (its argmax and a hash of its
  bits, so two runs of the same forward, e.g. the old and the new prefill path, can be compared bit for bit)."""
  import hashlib
  from fusezap.serve.neuron_prewarm import plan, synthetic_step
  dev = Device[eng.model.device]
  out: dict = {}
  for f, ctx in plan(eng):
    if not f.prefill: continue
    step = synthetic_step(eng, f, ctx)
    if step is None: continue
    ws = []
    for k in range(reps + 1):
      t0 = time.perf_counter()
      lg = f.run(*step)
      tok = lg.host_id(0)
      if k: ws.append(time.perf_counter() - t0)   # the first run may load a NEFF
    row = np.asarray(lg.row(0), np.float32)
    out[f"{f.N}@{ctx}"] = {"rows": f.N, "ctx": ctx, "tokens": int(min(f.N, ctx)), "wall_ms": [round(1e3 * w, 2) for w in ws],
                          "wall_ms_min": round(1e3 * min(ws), 2) if ws else None, "wall_ms_p50": _pct(ws, 50), "top": int(tok),
                          "logits_sha": hashlib.sha256(row.tobytes()).hexdigest()[:16]}
    if hasattr(dev, "synchronize"): dev.synchronize()
  return out

def chunk_check(eng, ref: dict, lens: list[int], n_tokens: int, reps: int = 2) -> dict:
  """I77 follow-up (--chunk-check): per prompt length L (the first reference prompt repeated to L tokens) the prompt alone
  through the engine under each prefill schedule it has: `auto` (its own: with several prefill sizes the cost sizing,
  Scheduler._fit_cost) and, with several sizes, `small` (every step the smallest forward) and `large` (the smallest forward
  holding all the step's rows: the schedule before cost sizing, e.g. 200 tokens in one 512-row forward). Per schedule: the
  forward sizes and rows of its prefill steps, the time to the first token per rep, a hash of the bits of the logits row
  of the prompt's last token (two runs or paths, e.g. v1's 128-row chunks and v2 exact's 512-row forward, must share it),
  its argmax, whether every rep gave the same bits, and the first n_tokens greedy tokens."""
  import hashlib
  from fusezap.serve.engine import SamplingParams
  p0 = list(ref["prompts"][0]["prompt_ids"])
  sizes, cfg = list(eng.prefill_sizes), eng.cfg
  saved = (cfg.prefill_cost, cfg.prefill_buckets)
  scheds = {"auto": saved}
  if len(sizes) > 1:
    scheds["small"] = ({n: (1.0 if n == sizes[0] else 1e6) for n in sizes}, tuple(sizes))
    scheds["large"] = (None, None)
  orig, out = eng._launch_prefill, {}
  try:
    for L in lens:
      ids = (p0 * (L // len(p0) + 1))[:L]
      for name, (cost, buckets) in scheds.items():
        cfg.prefill_cost, cfg.prefill_buckets = cost, buckets
        rec: dict = {"sizes": [], "rows": [], "ttft_ms": [], "shas": []}
        for _ in range(reps):
          seen: dict = {}
          def launch(so, seen=seen, rec=rec):
            done, lg, k, n = orig(so)
            if not rec["shas"]: rec["sizes"].append(n); rec["rows"].append(k)
            for s, l, i in done:
              if l is not None:
                row = np.asarray(l.row(i), np.float32)
                seen["sha"], seen["top"] = hashlib.sha256(row.tobytes()).hexdigest()[:16], int(np.argmax(row))
            return done, lg, k, n
          eng._launch_prefill = launch
          t0, first = time.perf_counter(), None
          rid = eng.add_request(ids, SamplingParams(max_tokens=n_tokens, ignore_eos=True))
          toks: list = []
          while eng.has_unfinished():
            for o in eng.step():
              if o.request_id == rid:
                if first is None and o.new_token_ids: first = time.perf_counter() - t0
                toks += o.new_token_ids
          eng.release(rid)
          eng._launch_prefill = orig
          rec["ttft_ms"].append(round(1e3 * first, 2) if first is not None else None)
          rec["shas"].append(seen.get("sha")); rec["top"], rec["ids"] = seen.get("top"), toks
        rec["logits_sha"] = rec["shas"][-1]
        rec["stable"] = len(set(rec["shas"])) == 1
        out.setdefault(str(L), {})[name] = rec
        log(f"chunk check {L} tokens [{name}]: forwards {rec['sizes']} (rows {rec['rows']}), TTFT {rec['ttft_ms']} ms, logits {rec['logits_sha']} "
            f"(stable {rec['stable']}), top {rec['top']}, ids {toks[:6]}")
  finally:
    eng._launch_prefill = orig
    cfg.prefill_cost, cfg.prefill_buckets = saved
  return out

def greedy(eng, ref: dict, n_prompts: int, n_tokens: int, max_seqs: int) -> dict:
  from fusezap.serve.engine import SamplingParams
  eos = set(ref.get("eos_token_ids") or [])
  ps, rows, t0 = ref["prompts"][:n_prompts], [], time.perf_counter()
  for b in range(0, len(ps), max_seqs):
    batch = ps[b:b + max_seqs]
    rids = {eng.add_request(q["prompt_ids"], SamplingParams(max_tokens=n_tokens)): q for q in batch}
    out = {r: [] for r in rids}
    while eng.has_unfinished():
      for o in eng.step(): out[o.request_id] += o.new_token_ids
    for r, q in rids.items():
      want, got = q["greedy_ids"][:n_tokens], out[r]
      n = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
      if n == min(len(got), len(want)) and len(got) < n_tokens and got and got[-1] in eos and len(want) == len(got): n = len(want)
      rows.append({"id": q["id"], "match": n, "of": len(want), "exact": n == len(want), "first_diff": None if n == len(want) else n,
                   "got": got[n:n + 3], "want": want[n:n + 3], "ids": got})   # ids: N7 compares host paths token by token
      eng.release(r)
    log(f"greedy batch {b // max_seqs}: " + ", ".join(f"{r['id']} {r['match']}/{r['of']}" for r in rows[-len(batch):]))
  return {"prompts": len(rows), "exact": sum(r["exact"] for r in rows), "match_tokens": sum(r["match"] for r in rows),
          "ref_tokens": sum(r["of"] for r in rows), "seconds": round(time.perf_counter() - t0, 1), "rows": rows}

def top1(eng, ref: dict, n_prompts: int) -> dict:
  from fusezap.serve.logprobs import scorer
  sc = scorer(eng)
  agree = total = 0
  err: list[float] = []
  t0 = time.perf_counter()
  for q in ref["prompts"][:n_prompts]:
    r = sc.score(list(q["prompt_ids"]) + list(q["greedy_ids"]), len(q["prompt_ids"]), 5)
    agree += sum(1 for i, top in enumerate(r["top"]) if top and top[0][0] == q["topk_ids"][i][0]); total += len(r["top"])
    err += [abs(x - y) for x, y in zip(r["logprobs"], q["greedy_logprobs"])]
  return {"agree": agree, "positions": total, "agreement": round(agree / total, 5) if total else None,
          "logprob_abs_err_mean": float(np.mean(err)) if err else None, "logprob_abs_err_max": float(np.max(err)) if err else None,
          "seconds": round(time.perf_counter() - t0, 1)}

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
  ap.add_argument("--reference", default=os.path.join(ROOT, "docs/invarix/examples/refs/qwen2.5-0.5b-instruct-f32.json"))
  ap.add_argument("--out", default="results/nki_n6")
  ap.add_argument("--variant", default="default")
  ap.add_argument("--quick", action="store_true", help="c1, c4 and the profile only (the before/after variants)")
  ap.add_argument("--prompts", type=int, default=24)
  ap.add_argument("--tokens", type=int, default=64)
  ap.add_argument("--top1-prompts", type=int, default=8)
  ap.add_argument("--c1-tokens", type=int, default=40)
  ap.add_argument("--c4-tokens", type=int, default=24)
  ap.add_argument("--profile-steps", type=int, default=8)
  ap.add_argument("--max-num-seqs", type=int, default=4)
  ap.add_argument("--num-blocks", type=int, default=256)
  ap.add_argument("--block-size", type=int, default=16)
  ap.add_argument("--max-model-len", type=int, default=1024)
  ap.add_argument("--max-prefill-tokens", type=int, default=128)
  ap.add_argument("--tp", type=int, default=1, help="N8: tensor parallel size (one NeuronCore per rank, fusezap.serve.neuron_tp)")
  ap.add_argument("--ttft-runs", type=int, default=0, help="I75: c1 TTFT over this many prompts alone")
  ap.add_argument("--burst-runs", type=int, default=0, help="I75: c4 TTFT over this many bursts of 4 prompts at once")
  ap.add_argument("--prefill-profile", action="store_true", help="I75: per-kernel profile of one prompt's prefill step")
  ap.add_argument("--ttft-lens", default="0", help="I76: comma list of prompt lengths for the TTFT runs (0: the reference prompts as they "
                                                   "are; N: each repeated to exactly N tokens, e.g. 128 = Invarix short)")
  ap.add_argument("--prefill-bench", type=int, default=0, help="I77: time every prefill forward at each pool prefix (this many replays) "
                  "and hash its first logits row (old vs new path, bit for bit)")
  ap.add_argument("--chunk-check", default="", help="I77: comma list of prompt lengths: each prompt alone under each prefill schedule "
                  "(auto / small / large forwards), the last prompt token's logits hash and the first --chunk-tokens tokens")
  ap.add_argument("--chunk-tokens", type=int, default=16)
  ap.add_argument("--no-prewarm", dest="prewarm", action="store_false",
                  help="skip the short c1 and c4 runs before measuring (their compiles and captures land in TTFT / the first steps); "
                       "the engine's own prewarm (NEURON, FZ_NEURON_PREWARM) still runs, so c1 is then the first request after start")
  a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True)
  dev = Device.DEFAULT
  ref = json.load(open(a.reference))
  OUT.update(device=dev, model=a.model, variant=a.variant, switches={k: os.environ.get(k) for k in SWITCHES if k in os.environ}, args=vars(a))
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine
  from fusezap.device import neuron
  t0 = time.perf_counter()
  g = None
  if a.tp > 1:   # N8: rank 1.. load their shards while this process (rank 0) loads its own; every forward is mirrored
    from fusezap.serve import dist
    g = dist.launch(a.tp, dev)
    TP.append(g)
    from fusezap.serve import neuron_tp as NT
    OUT["tp"] = {"world": a.tp, "all_reduce": g.ar_info, "mix": NT.mix(g)}
    log(f"[{a.variant}] tensor parallel {a.tp}: {g.ar_info}")
    g.post("load", {"model": a.model, "dtype": "float32", "max_len": a.max_model_len, "prefill_chunk": a.max_prefill_tokens})
  m = load_model(a.model, dtype=dtypes.float32, device=g.device if g else dev, max_len=a.max_model_len, prefill_chunk=a.max_prefill_tokens,
                 **({"tp": g} if g else {}))
  if g is not None: g.barrier("load", 3600)
  OUT["load_s"] = round(time.perf_counter() - t0, 1)
  OUT["dtype"] = getattr(m, "wdtype", m.dtype).name   # the weights' dtype (bf16 on NEURON keeps float32 activations: m.dtype)
  eng = LLMEngine(m, num_blocks=a.num_blocks, block_size=a.block_size, max_num_seqs=a.max_num_seqs, max_prefill_tokens=a.max_prefill_tokens,
                  max_model_len=a.max_model_len)
  if g is not None:
    g.post("engine", eng.tp_spec(), eng.tp_signature())
    g.barrier("engine")
  OUT["engine"] = {"decode_sizes": eng.decode_sizes, "prefill_sizes": eng.prefill_sizes, "pool_slots": eng.cache.num_blocks * eng.cache.bs,
                   "ctx_buckets": eng.cache.ctx_buckets() if hasattr(eng.cache, "ctx_buckets") else None}
  log(f"[{a.variant}] loaded {a.model} on {dev} in {OUT['load_s']} s; engine {OUT['engine']}; switches {OUT['switches']}")
  p0 = ref["prompts"][0]
  if dev == "NEURON":   # I75: the engine's own prewarm (every forward at every pool prefix), as `fusezap serve` runs it
    t0 = time.perf_counter()
    from fusezap.serve.neuron_prewarm import prewarm as engine_prewarm
    pw = engine_prewarm(eng, log)
    OUT["engine_prewarm"] = {k: pw.get(k) for k in ("seconds", "jobs", "forwards", "skipped")}
    OUT["engine_prewarm"]["neffs"] = {r: [{k: x[k] for k in ("kernel", "hit", "ok", "seconds")} for x in v] for r, v in pw.get("neffs", {}).items()}
    OUT["engine_prewarm"]["wall_s"] = round(time.perf_counter() - t0, 1)
    log(f"[{a.variant}] engine prewarm {OUT['engine_prewarm']['wall_s']} s: {json.dumps(OUT['engine_prewarm']['seconds'])}")
  if a.prewarm:   # compile and capture the c1 and c4 shapes first (a cold NEFF cache put them in run 1's TTFT and first steps)
    t0 = time.perf_counter()
    run_alone(eng, p0["prompt_ids"], 8)
    if a.c4_tokens and a.max_num_seqs >= 4: run_together(eng, [q["prompt_ids"] for q in ref["prompts"][1:5]], 8)
    OUT["prewarm_s"] = round(time.perf_counter() - t0, 1)
    log(f"[{a.variant}] prewarm {OUT['prewarm_s']} s")
  hbm(eng, "after prewarm")
  t0 = time.perf_counter()
  OUT["c1"], toks = run_alone(eng, p0["prompt_ids"], a.c1_tokens, a.profile_steps)
  g = p0["greedy_ids"]
  k = next((i for i, (x, y) in enumerate(zip(toks, g)) if x != y), min(len(toks), len(g)))
  OUT["c1"]["greedy_prefix"] = f"{k}/{min(len(toks), len(g))}"
  OUT["c1"]["ids"] = toks
  OUT["c1"]["seconds"] = round(time.perf_counter() - t0, 1)
  log(f"[{a.variant}] c1: {OUT['c1']}")
  OUT["launches"] = launches(eng)
  log(f"[{a.variant}] launches per forward: {json.dumps(OUT['launches'])}")
  try: OUT["profile"] = profile_report(OUT["c1"], eng)
  except Exception as e: OUT["profile"] = {"error": f"{type(e).__name__}: {e}"}
  if OUT["profile"]: log(f"[{a.variant}] profile per token: {json.dumps(OUT['profile'].get('per_token'))}")
  if a.c4_tokens and a.max_num_seqs >= 4:
    t0 = time.perf_counter()
    OUT["c4"] = run_together(eng, [q["prompt_ids"] for q in ref["prompts"][1:5]], a.c4_tokens)
    OUT["c4"]["seconds"] = round(time.perf_counter() - t0, 1)
    log(f"[{a.variant}] c4: {OUT['c4']}")
  if a.ttft_runs or a.burst_runs:
    for n in [int(x) for x in a.ttft_lens.split(",") if x.strip()]:
      t0 = time.perf_counter()
      ps = [q["prompt_ids"] for q in ref["prompts"]]
      if n: ps = [(list(p) * (n // len(p) + 1))[:n] for p in ps]
      tt = ttft(eng, ps, a.ttft_runs, a.burst_runs if a.max_num_seqs >= 4 else 0)
      tt["prompt_len"] = n or "reference"
      OUT.setdefault("ttft_by_len", {})[str(n)] = tt
      if "ttft" not in OUT: OUT["ttft"] = tt
      log(f"[{a.variant}] ttft, prompts {tt['prompt_len']} ({time.perf_counter() - t0:.1f} s): {json.dumps({k: v for k, v in tt.items() if not k.endswith('_ids')})}")
  if a.prefill_bench:
    try: OUT["prefill_bench"] = prefill_bench(eng, a.prefill_bench)
    except Exception as e: OUT["prefill_bench"] = {"error": f"{type(e).__name__}: {e}"}
    log(f"[{a.variant}] prefill bench: {json.dumps(OUT['prefill_bench'])}")
  if a.chunk_check:
    try: OUT["chunk_check"] = chunk_check(eng, ref, [int(x) for x in a.chunk_check.split(",") if x.strip()], a.chunk_tokens)
    except Exception as e:
      import traceback
      OUT["chunk_check"] = {"error": f"{type(e).__name__}: {e}"}
      log(traceback.format_exc()[-2000:])
  OUT["prefill_cost"] = {str(k): v for k, v in (eng.cfg.prefill_cost or {}).items()} or None
  if a.prefill_profile:
    try: OUT["prefill_profile"] = prefill_profile(eng, ref["prompts"][0]["prompt_ids"])
    except Exception as e: OUT["prefill_profile"] = {"error": f"{type(e).__name__}: {e}"}
  if not a.quick:
    OUT["greedy"] = greedy(eng, ref, a.prompts, a.tokens, a.max_num_seqs)
    hbm(eng, "after greedy")
    if a.top1_prompts:
      OUT["top1"] = top1(eng, ref, a.top1_prompts)
      hbm(eng, "after top-1 scoring")
  OUT["neff"] = dict(neuron.NEFF_STATS)
  OUT["worker_calls_total"] = worker_calls()
  with open(os.path.join(a.out, f"serve_{a.variant}.json"), "w") as f: json.dump(OUT, f, indent=1, default=str)
  summary()
  for grp in TP: grp.close()

def summary() -> None:
  v = OUT["variant"]
  log(f"\n==== NKI-N6 serve summary [{v}] ====")
  log(f"[{v}] device {OUT['device']}  switches {OUT['switches']}  load {OUT.get('load_s')} s  prewarm {OUT.get('prewarm_s')} s  engine {OUT.get('engine')}")
  if (tp := OUT.get("tp")): log(f"[{v}] tensor parallel {tp['world']}: {tp['all_reduce']}")
  dec = {k: x for k, x in (OUT.get("launches") or {}).items() if k.startswith("decode_1")}
  for k, x in dec.items(): log(f"[{v}] launches {k:<16} {x['launches']} per token ({x['by_kind']})")
  c = OUT.get("c1") or {}
  log(f"[{v}] c1 TTFT {c.get('ttft_s')} s  TPOT median {c.get('tpot_ms_median')} ms (p10 {c.get('tpot_ms_p10')}, p90 {c.get('tpot_ms_p90')})  "
      f"first steps {c.get('first_steps_s')} s  greedy prefix {c.get('greedy_prefix')}  worker {c.get('worker_calls')}")
  if (c4 := OUT.get("c4")): log(f"[{v}] c4 TPOT median {c4.get('tpot_ms_median')} ms, {c4.get('tok_s_decode')} tok/s decoding ({c4.get('decode_steps')} steps)")
  for tt in (OUT.get("ttft_by_len") or {}).values():
    log(f"[{v}] TTFT in process (prompts {tt.get('prompt_len', 'reference')}): c1 p50 {tt['c1_p50']} p99 {tt['c1_p99']} ms ({len(tt['c1_ms'])} prompts); "
        f"c4 burst p50 {tt['c4_p50']} p99 {tt['c4_p99']} ms ({len(tt['c4_ms'])} requests: {tt['c4_ms']})")
  for k, x in (OUT.get("prefill_bench") or {}).items():
    if isinstance(x, dict): log(f"[{v}] prefill forward {x['rows']} rows at pool prefix {x['ctx']} ({x['tokens']} tokens): wall min {x['wall_ms_min']} "
                                f"p50 {x['wall_ms_p50']} ms, top {x['top']}, logits {x['logits_sha']}")
  for L, d in (OUT.get("chunk_check") or {}).items():
    for nm, r in (d.items() if isinstance(d, dict) else ()):
      if isinstance(r, dict): log(f"[{v}] chunk check {L} tokens [{nm}]: forwards {r['sizes']}, TTFT {r['ttft_ms']} ms, logits {r['logits_sha']} (stable {r['stable']})")
  if OUT.get("prefill_cost"): log(f"[{v}] prefill cost sizing (ms per forward): {OUT['prefill_cost']}")
  if (pp := OUT.get("prefill_profile")):
    if "kernels" in pp:
      log(f"[{v}] prefill step {pp['step_ms']} ms: {pp['launches']} launches, executes {pp['exec_ms_total']} ms, worker batches {pp['worker_batch_ms']} ms, {pp['trips']} round trips")
      for nm, x in list(pp["kernels"].items())[:10]: log(f"[{v}]   prefill {nm:<44} {x['launches']:4d} launches  execute {x['exec_ms_mean']:8.3f} ms each, {x['exec_ms_total']:8.3f} total")
    else: log(f"[{v}] prefill profile: {pp}")
  pr = OUT.get("profile") or {}
  if (pt := pr.get("per_token")):
    log(f"[{v}] per token: step {pt.get('step_ms')} ms = host {pt.get('host_ms')} + worker {pt.get('worker_ms')} + runtime {pt.get('runtime_ms')} "
        f"+ device {pt.get('device_ms')} ms; {pt.get('launches')} launches, executes {pt.get('execute_ms')} ms")
    if (hs := pt.get("host_split")):
      log(f"[{v}] host split per token (ms): " + "  ".join(f"{k} {hs[k]}" for k in ("sched", "prep", "meta", "call", "post", "trips", "ipc_wall_ms",
                                                                                 "pipe_ms", "worker_other_ms", "python_ms") if k in hs))
    n = max(pt.get("steps") or 1, 1)
    for nm, x in list(pr.get("kernels", {}).items())[:12]:
      log(f"[{v}]   {nm:<44} {x['launches'] / n:6.1f}/tok  execute {x['exec_ms_mean']:8.3f} ms  device {x['device_ms']} ms")
  elif pr: log(f"[{v}] profile: {pr}")
  if (g := OUT.get("greedy")):
    log(f"[{v}] greedy {g['exact']}/{g['prompts']} prompts exact, {g['match_tokens']}/{g['ref_tokens']} tokens ({g['seconds']} s)")
    for r in g["rows"]:
      if not r["exact"]: log(f"[{v}]   {r['id']} first difference at {r['first_diff']}: got {r['got']} want {r['want']}")
  if (t := OUT.get("top1")): log(f"[{v}] top1 {t['agreement']} ({t['agree']}/{t['positions']}), |logprob err| mean {t['logprob_abs_err_mean']} max {t['logprob_abs_err_max']}")
  log(f"[{v}] neff {OUT.get('neff')}; runtime worker calls {OUT.get('worker_calls_total')}")

if __name__ == "__main__": main()
