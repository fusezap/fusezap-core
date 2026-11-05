"""I75: NEURON prewarm at engine start: every forward the engine can run (each decode bucket and prefill size, at each of
its pool prefixes: NeuronKVCache.ctx_buckets, the whole pool included) is captured, its NEFFs compiled and loaded before the
first request, so no request pays a compile (~204 s for one FUSE=all 7B TP=2 prefill NEFF) or a load (~2.6 s warm).

  A. capture: each (forward, pool prefix) runs twice on a synthetic step whose block table ends exactly at that prefix
     (free blocks: nothing is in use at start; the KV rows it writes are overwritten by a sequence's own prefill before
     they are read). The first call runs the items (compiling the small member NEFFs), the second captures and fuses.
  B. precompile: the fused kernels' NEFFs (the big ones) compile in parallel, FZ_NEURON_PREWARM_JOBS per rank
     (NEURONDevice.precompile; a group that fails to compile is marked and its 4-layer fallback groups compile instead).
  C. load: each forward replays once (its NEFFs load into the runtime worker; the fast replay request is built).

Under tensor parallelism rank 0 drives it: A and C are ordinary mirrored forwards (TPGroup.spmd: the workers run the same
step, end_op sends every rank's queued launches), B is one mirrored op "call" (precompile_op on every rank) followed by a
barrier, so no rank enters a collective while its peer compiles. Rank 0 alone decides what to prewarm (an HBM check per
prefix), and the workers just mirror what it sends. FZ_NEURON_PREWARM=0 skips it (forwards compile lazily, as before).
"""
from __future__ import annotations
import os, sys, time
from typing import Any, Callable
import numpy as np

PREWARM_RESERVE = 1 << 30   # HBM kept free after prewarm (the scorer's buckets, the allocator's cache, a capture's transients)

def enabled() -> bool: return os.environ.get("FZ_NEURON_PREWARM", "1") not in ("", "0")

def synthetic_step(eng: Any, f: Any, ctx: int) -> tuple | None:
  """Host arrays (ids, sv, pos, meta) of a step of forward f whose sequences' blocks end exactly at pool prefix ctx (so
  meta_ctx picks ctx), or None if f cannot run at ctx. Prefill: one sequence of min(N, ctx, max_len) tokens (the rest
  padding rows); decode: up to N sequences of one token each. Block ids are counted down from ctx's last block."""
  c = eng.cache
  bs, N = c.bs, f.N
  if ctx % bs or ctx > c.P: return None
  if f.prefill: lens = [min(N, ctx, c.max_blocks * bs)]
  else: lens = [1] * min(N, ctx // bs)
  if not lens or lens[0] < 1: return None
  top, tables, sid, pos, base = ctx // bs, {}, [], [], 1 << 40
  for j, n in enumerate(lens):
    nb = -(-n // bs)
    if top - nb < 0: return None
    tables[base + j] = list(range(top - nb, top))
    top -= nb
    sid += [base + j] * n
    pos += list(range(n))
  sid_a, pos_a = np.array(sid + [-1] * (N - len(sid)), np.int64), np.array(pos + [0] * (N - len(pos)), np.int64)
  sel = [i for i in range(len(sid)) if i + 1 == len(sid) or sid[i + 1] != sid[i]] if f.prefill else list(range(len(sid)))
  old = {k: c.tables.get(k) for k in tables}
  c.tables.update(tables)
  try: meta = c.step_meta_rows(f.N, f.R, sid_a[:len(sid)], pos_a[:len(sid)], sel[:f.R])
  finally:
    for k, v in old.items():
      if v is None: c.tables.pop(k, None)
      else: c.tables[k] = v
  if c.meta_ctx(meta, f.N, f.R, f.prefill) != ctx: return None
  return np.zeros(N, np.int64), np.full(N, -1, np.int64), pos_a, meta

def plan(eng: Any) -> list[tuple[Any, int]]:
  """(forward, pool prefix) pairs to prewarm: every decode bucket, then every prefill size, each at each of its prefixes.
  I77: a prefill size n above a smaller size m runs only steps of more than m rows, so its prefixes below m + 1 slots are
  unreachable (a 512-row forward never runs at a 128-slot prefix when a 128-row forward exists): not prewarmed."""
  out = []
  for fs, pf in ((eng._decodes, False), (eng._prefills, True)):
    prev = 0
    for n in sorted(fs):
      for ctx in eng.cache.ctx_buckets(pf):
        if not pf or ctx > prev: out.append((fs[n], ctx))
      if pf: prev = n
  return out

def precompile_op(g: Any, jobs: int) -> dict:
  """Every rank (op "call", and rank 0 itself): NEURONDevice.precompile on this rank's device; its records and seconds."""
  from fusezap.device import Device
  dev = Device[g.device] if g is not None else None
  t0 = time.perf_counter()
  recs = dev.precompile(jobs, getattr(g, "rank", 0), getattr(g, "world", 1)) if hasattr(dev, "precompile") else []
  return {"rank": getattr(g, "rank", 0), "seconds": round(time.perf_counter() - t0, 1), "neffs": recs,
          "load_mark": len(getattr(getattr(dev, "allocator", None), "load_log", []))}

def loads_op(g: Any, marks: dict) -> list:
  """Every rank: this rank's NEFF loads since its mark (marks: rank -> index into its allocator's load_log)."""
  from fusezap.device import Device
  return [tuple(x) for x in getattr(Device[g.device].allocator, "load_log", [])[int(marks.get(g.rank, 0)):]]

def _held(eng: Any) -> tuple[int, int]:
  from fusezap.device import Device
  al = Device[eng.model.device].allocator
  lim = al.limit() if hasattr(al, "limit") else int(getattr(al, "budget", 0))
  return int(getattr(al, "held", 0)) - int(getattr(al, "cached", 0)), lim

def prewarm(eng: Any, log: Callable[[str], None] | None = None) -> dict:
  """Capture, compile (in parallel) and load every forward at every pool prefix (module docstring). Returns the
  breakdown (also logged): per forward and prefix the capture and load seconds, per NEFF compiled / hit / failed and its
  seconds, each rank's NEFF loads, the totals, and what was left lazy (HBM)."""
  from fusezap.device import Device
  from fusezap.device.neuron import prewarm_jobs
  from fusezap.serve.neuron_kv import jit_device_bytes
  log = log or (lambda m: print(m, file=sys.stderr, flush=True))
  rep: dict = {"forwards": [], "skipped": [], "neffs": {}, "loads": {}}
  if getattr(eng, "neuron", False):   # I77: the prefill kernels' device self-test, before any forward is captured
    from fusezap.serve.neuron_prefill import gate
    if (line := gate(eng, log)): log(line); rep["prefill_kernels"] = line
  if not getattr(eng, "neuron", False) or not enabled(): return rep
  assert not eng.has_unfinished(), "prewarm runs before the first request"
  dev = Device[eng.model.device]
  tp, al = eng.tp, dev.allocator
  t_all = time.perf_counter()
  jobs = prewarm_jobs()
  # A. capture each (forward, prefix) not captured yet; rank 0 stops a forward's prefixes when the next one may not fit
  todo, done = [], []
  for f, ctx in plan(eng):
    j = f.jit_for(ctx)
    if j.captured is not None and j.cnt >= 2: done.append((f, ctx)); continue
    step = synthetic_step(eng, f, ctx)
    if step is None: rep["skipped"].append((f.key, ctx, "unreachable")); continue
    held, lim = _held(eng)
    est = max([jit_device_bytes(x) for x in [f.jit, *f.ctx_jits.values()]] + [0])
    if lim and held + int(1.25 * est) + PREWARM_RESERVE > lim:
      rep["skipped"].append((f.key, ctx, f"HBM: {held / 2**30:.2f} GiB held + ~{est / 2**30:.2f} of {lim / 2**30:.2f}"))
      log(f"fusezap: prewarm leaves {f.key} at pool prefix {ctx} lazy (HBM: {held / 2**30:.2f} GiB held, ~{est / 2**30:.2f} more, budget {lim / 2**30:.2f})")
      continue
    t0 = time.perf_counter()
    for _ in range(2): f.run(*step)
    if hasattr(dev, "synchronize"): dev.synchronize()
    todo.append((f, ctx, step, time.perf_counter() - t0))
  t_a = time.perf_counter() - t_all
  # B. the fused NEFFs, in parallel on every rank
  t0 = time.perf_counter()
  if tp is not None:
    fn = "fusezap.serve.neuron_prewarm:precompile_op"
    with tp.spmd("call", fn, jobs): mine = precompile_op(tp, jobs)
    others = tp.barrier("prewarm-compile", 4 * 3600)
  else: mine, others = precompile_op(type("G", (), {"device": eng.model.device, "rank": 0, "world": 1})(), jobs), []
  t_b = time.perf_counter() - t0
  marks = {r["rank"]: r.get("load_mark", 0) for r in [mine, *[o for o in others if isinstance(o, dict)]]}
  for r in [mine, *[o for o in others if isinstance(o, dict)]]:
    rep["neffs"][r["rank"]] = r["neffs"]
    for x in r["neffs"]:
      log(f"fusezap: prewarm rank {r['rank']}: NEFF {x['kernel'][:40]} {'cache hit' if x['hit'] else 'compiled' if x['ok'] else 'FAILED (fallback groups)' if not x.get('retry') else 'not precompiled'} "
          f"{x['seconds']} s")
  # C. one replay each: the NEFFs load, the fast replay request is built
  for f, ctx, step, ta in todo:
    t0 = time.perf_counter()
    f.run(*step)
    if hasattr(dev, "synchronize"): dev.synchronize()
    tc = time.perf_counter() - t0
    rep["forwards"].append({"forward": f"{'prefill' if f.prefill else 'decode'}_{f.N}", "ctx": ctx, "capture_s": round(ta, 2), "load_s": round(tc, 2)})
  for f, ctx in done: rep["forwards"].append({"forward": f"{'prefill' if f.prefill else 'decode'}_{f.N}", "ctx": ctx, "capture_s": 0.0, "load_s": 0.0, "before": True})
  rep["loads"][0] = [tuple(x) for x in getattr(al, "load_log", [])[marks.get(0, 0):]]
  if tp is not None:
    with tp.spmd("call", "fusezap.serve.neuron_prewarm:loads_op", marks): pass
    for r, o in enumerate(tp.barrier("prewarm-loads"), 1):
      if isinstance(o, list): rep["loads"][r] = o
  rep["seconds"] = {"capture": round(t_a, 1), "compile": round(t_b, 1), "load": round(sum(x["load_s"] for x in rep["forwards"]), 1),
                    "total": round(time.perf_counter() - t_all, 1)}
  rep["jobs"] = jobs
  for x in rep["forwards"]:
    log(f"fusezap: prewarm {x['forward']} at pool prefix {x['ctx']}: capture {x['capture_s']} s, first replay (NEFF loads) {x['load_s']} s")
  for r, ls in rep["loads"].items():
    if ls: log(f"fusezap: prewarm rank {r}: {len(ls)} NEFF loads, {sum(x[2] for x in ls):.2f} s, {sum(x[3] for x in ls) / 2**20:.1f} MiB "
               f"(largest {max(x[2] for x in ls):.2f} s: {max(ls, key=lambda x: x[2])[0][:40]})")
  nn = [x for v in rep["neffs"].values() for x in v]
  log(f"fusezap: prewarm {rep['seconds']['total']} s: capture {rep['seconds']['capture']} s, parallel compile {rep['seconds']['compile']} s "
      f"({sum(not x['hit'] and x['ok'] for x in nn)} compiled, {sum(x['hit'] for x in nn)} cache hits, {sum(not x['ok'] for x in nn)} failed; "
      f"{jobs} jobs per rank), loads {rep['seconds']['load']} s; {len(rep['forwards'])} forwards x prefixes, {len(rep['skipped'])} left lazy")
  return rep
