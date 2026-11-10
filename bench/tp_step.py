"""Tensor-parallel decode step A/B on CUDA (I57 M3): one model load, then every all-reduce path and GEMM rule measured
in the same processes, the per-rank kernel breakdown and the all-reduce latency at the step's sizes.

  python bench/tp_step.py --model <path> [--tp 2] [--buckets 1,8,32] [--modes fuse,car,nccl,skip] [--steps 48] [--ctx 128]
         [--env-ab 'S12:FZ_WG_SWAP_S=12;FILL2:FZ_WG_SWAP_FILL=2'] [--breakdown 1,32] [--vllm docs/.../report.json] [--json out] [--load-timeout 300]

Engine as `fusezap serve --tp N --max-model-len 8192 --num-blocks 8192 --max-num-seqs 32` builds it (bf16, lookahead).
For each mode (tpprof.MODES: fuse = custom all-reduce + fused add/RMSNorm (the default), car = custom all-reduce only,
nccl = NCCL only, skip = no communication at all: a lower bound, wrong tokens) and each --env-ab variant (applied on top
of fuse), every rank switches and drops its captured forwards (tpprof.set_mode); then per bucket b: b requests of ctx
random prompt tokens are prefilled, 6 decode steps capture the graphs, and --steps engine steps are timed (rank 0 wall
clock: TPOT = the time per step, every step one token per request, lookahead on). Also per mode: the time of a ctx-token
prompt's prefill step (TTFT without HTTP). Then, in the default mode: the kernel breakdown (tpprof.kernel_breakdown,
every rank, per --breakdown bucket) and the all-reduce latency per path at rows x hidden bf16 (dist.bench_allreduce,
graphs of 50 back-to-back all-reduces). Prints one line per measurement (TP_STEP / TP_KERNELS / TP_AR JSON) and a table.
A load that is not done on every rank within --load-timeout seconds exits 3 (TP_LOAD_TIMEOUT) instead of using the step's budget.

I60 (M5): --random PRESET (fusezap.serve.models.random_weights, full shapes, e.g. llama70b) builds every rank's slices
of a random model instead of loading --model (no download, ~40 s on 2x H200). --env-ab entries may set mode=<mode> (the
all-reduce path of that variant, default fuse). Per variant and bucket also each rank's GPU ms per decode graph replay
(tpprof.graph_ms: rank-order list) and, for a variant with FZ_TP_AR_STAMP=1, the stamps' split of the all-reduces
(tpprof.stamp_stats: intrinsic in-graph cost vs rank-skew wait vs compute; TP_STAMPS lines). --prefix names this
process's variants (one process per NCCL environment: NCCL reads NCCL_PROTO / NCCL_MAX_NCHANNELS / ... once per process).
--ar-chain both: the all-reduce latency also with a kernel between the all-reduces (chain False), as in a model step.
"""
from __future__ import annotations
import argparse, json, os, sys, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

def _all(g, fn: str, *args):
  """Run module function fn (module:name) as fn(g, *args) on every rank; returns [rank 0's, workers' ...]."""
  import importlib
  mod, name = fn.split(":")
  f = getattr(importlib.import_module(mod), name)
  with g.spmd("call", fn, *args): r0 = f(g, *args)
  return [r0] + g.barrier(name)

def measure(eng, b: int, ctx: int, steps: int, seed: int) -> dict:
  from fusezap.serve.engine import SamplingParams
  rng, V = np.random.default_rng(seed), eng.model.cfg.vocab_size
  sp = SamplingParams(max_tokens=steps + 64, ignore_eos=True)
  rids = [eng.add_request([int(t) for t in rng.integers(min(1000, V // 2), min(20000, V), ctx)], sp) for _ in range(b)]
  n0 = lambda: sum(len(eng.seqs[r].output) for r in rids)
  t = time.perf_counter()
  while min(len(eng.seqs[r].output) for r in rids) < 1: eng.step()
  t_pref = time.perf_counter() - t
  for _ in range(6): eng.step()   # trace, record, warm and capture the bucket's graph (after a mode switch)
  k0, t0 = n0(), time.perf_counter()
  for _ in range(steps): eng.step()
  dt, k1 = time.perf_counter() - t0, n0()
  for r in rids: eng.abort(r)
  while eng.has_unfinished() or eng._pend is not None or eng._pend_p is not None: eng.step()   # no launched step left behind
  for r in rids: eng.release(r)
  return {"b": b, "tpot_ms": round(dt / steps * 1e3, 3), "tok_s": round((k1 - k0) / dt, 1), "tokens": k1 - k0, "prefill_s": round(t_pref, 3),
          "lookahead": eng.stats.get("lookahead_steps")}

def ttft(eng, ctx: int, seed: int, reps: int = 3) -> float:
  from fusezap.serve.engine import SamplingParams
  rng, V = np.random.default_rng(seed), eng.model.cfg.vocab_size
  ts = []
  for i in range(reps + 1):
    rid = eng.add_request([int(t) for t in rng.integers(min(1000, V // 2), min(20000, V), ctx)], SamplingParams(max_tokens=1, ignore_eos=True))
    t0 = time.perf_counter()
    while eng.has_unfinished(): eng.step()
    ts.append(time.perf_counter() - t0)
    eng.release(rid)
  return round(float(np.median(ts[1:])) * 1e3, 2)

def print_tables(res: dict, buckets: list[int], vt: dict) -> None:
  """TPOT per variant and bucket (% of vLLM's TPOT at that concurrency, each rank's graph ms), then the stamp split."""
  names = list(dict.fromkeys(r["variant"] for r in res["steps"]))
  print("\nTPOT ms per step (rank 0 wall clock, decode-only steady state) (% of vLLM TP Invarix p50 TPOT) [graph ms per rank]")
  print(f"{'variant':24s} " + " ".join(f"{'b' + str(b):28s}" for b in buckets) + "  ttft_ms")
  for nm in names:
    row = {r["b"]: r for r in res["steps"] if r["variant"] == nm}
    def cell(b: int) -> str:
      if (r := row.get(b)) is None: return f"{'-':28s}"
      gm = "/".join("-" if x is None else f"{x:.2f}" for x in (r.get("graph_ms") or []))
      pct = f" ({100 * r['tpot_ms'] / v:3.0f}%)" if (v := vt.get(b)) else ""
      return f"{r['tpot_ms']:6.2f}{pct} [{gm}]".ljust(28)
    print(f"{nm:24s} " + " ".join(cell(b) for b in buckets) + f"  {res['ttft'].get(nm, '-')}")
  if vt: print(f"{'vLLM TP (Invarix p50)':24s} " + " ".join(f"{(vt.get(b) or float('nan')):<28.2f}" for b in buckets))
  st = [r for r in res["steps"] if r.get("stamps") and "error" not in r["stamps"]]
  if st:
    print("\nall-reduce stamps (medians per forward): period ms per rank | all-reduce ms per rank = intrinsic (latest arriver's) + "
          "wait (rank skew) | compute ms per rank, critical path, tail | per all-reduce us: intrinsic / wait (p90) | last to arrive per rank")
    for r in st:
      s = r["stamps"]
      par = "; ".join(f"{k}: intr {v['intrinsic_us']} wait {v['wait_us']} seg {v['seg_us']} late {v['late']}" for k, v in s.get("by_parity", {}).items())
      print(f"{r['variant']:24s} b{r['b']:<3d} period {s['period_ms']} | ar {s['ar_ms']} = {s['intrinsic_ms']} + {s['wait_ms']} | compute "
            f"{s['compute_ms']} crit {s['compute_crit_ms']} tail {s['tail_ms']} | {s['intrinsic_us']} / {s['wait_us']} ({s['wait_us_p90']}) us | "
            f"late {s['late']} begin skew {s['begin_skew_us']} us; {par}")
  for r in res["steps"]:
    if r.get("stamps") and "error" in r["stamps"]: print(f"{r['variant']} b{r['b']}: stamps: {r['stamps']['error']}")

def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--model", default=None); ap.add_argument("--tp", type=int, default=2)
  ap.add_argument("--random", default=None, help="random weights at this preset's full shapes (fusezap.serve.models.random_weights)")
  ap.add_argument("--layers", type=int, default=0, help="--random: the layer count (0: the preset's)")
  ap.add_argument("--prefix", default="", help="prefix of every variant name (e.g. the process's NCCL environment)")
  ap.add_argument("--ar-chain", default="1", choices=("1", "0", "both"), help="all-reduce latency: back to back (1), with a kernel between (0), both")
  ap.add_argument("--graph-reps", type=int, default=20, help="decode graph replays per rank timed per variant and bucket (0: off)")
  ap.add_argument("--buckets", default="1,8,32"); ap.add_argument("--modes", default="fuse,car,nccl,skip")
  ap.add_argument("--steps", type=int, default=48); ap.add_argument("--ctx", type=int, default=128)
  ap.add_argument("--max-model-len", type=int, default=8192); ap.add_argument("--num-blocks", type=int, default=8192)
  ap.add_argument("--env-ab", default="", help="NAME:K=V,K=V;NAME2:... variants: the fuse mode (or mode=<mode>) plus that environment")
  ap.add_argument("--breakdown", default="1,32"); ap.add_argument("--ar-rows", default="1,8,32,64,256,512")
  ap.add_argument("--vllm", default=None, help="a Invarix report.json of vLLM on the same model / GPUs (its TPOT per concurrency)")
  ap.add_argument("--json", default=None)
  ap.add_argument("--device", default="CUDA", help="CUDA; CPU / METAL only to check the script (host backend, modes nccl)")
  ap.add_argument("--dtype", default="bfloat16")
  ap.add_argument("--load-timeout", type=float, default=300.0, help="exit 3 if the weights are not loaded on every rank by then")
  a = ap.parse_args()
  assert a.model or a.random, "--model or --random"
  from fusezap.serve import dist, tpprof
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine
  from fusezap.dtype import dtypes
  t0 = time.perf_counter()
  loaded = threading.Event()
  def watchdog() -> None:   # fail fast on a starved load (the first M3 run: 15 min for nothing); the workers follow rank 0 out
    if not loaded.wait(a.load_timeout):
      print(f"TP_LOAD_TIMEOUT {json.dumps({'seconds': a.load_timeout})}: weights not loaded on every rank", flush=True)
      os._exit(3)
  threading.Thread(target=watchdog, daemon=True).start()
  g = dist.launch(a.tp, a.device)
  if a.random:
    from fusezap.serve.models import random_weights
    sh = random_weights.PRESETS[a.random]
    g.post("load", {"random": sh, "dtype": a.dtype, "max_len": a.max_model_len, "prefill_chunk": 128, "seed": 0, "layers": a.layers})
    m = random_weights.build(sh, g.device, getattr(dtypes, a.dtype), a.max_model_len, 128, tp=g, seed=0, layers=a.layers)
  else:
    g.post("load", {"model": a.model, "dtype": a.dtype, "max_len": a.max_model_len, "prefill_chunk": 128})
    m = load_model(a.model, dtype=getattr(dtypes, a.dtype), device=g.device, max_len=a.max_model_len, prefill_chunk=128, tp=g)
  g.barrier("load", a.load_timeout)
  loaded.set()
  print(f"TP_LOADED {json.dumps({'load_s': round(time.perf_counter() - t0, 1)})}", flush=True)
  eng = LLMEngine(m, num_blocks=a.num_blocks, block_size=16, max_num_seqs=32, max_model_len=a.max_model_len)
  g.post("engine", eng.tp_spec(), eng.tp_signature())
  g.barrier("engine", 1800)
  g.engine = eng   # tpprof's set_mode / kernel_breakdown on rank 0 (a worker reaches its own through g.worker)
  print(f"TP_INFO {json.dumps({'load_s': round(time.perf_counter() - t0, 1), 'ar': g.ar_info, 'decode_sizes': eng.decode_sizes, 'prefill_sizes': eng.prefill_sizes})}", flush=True)
  buckets = [int(x) for x in a.buckets.split(",") if x]
  modes = [md for md in a.modes.split(",") if md]
  envs = []
  for spec in filter(None, a.env_ab.split(";")):
    nm, kv = spec.split(":", 1)
    e = dict(p.split("=", 1) for p in kv.split(",") if p)
    envs.append((nm, e.pop("mode", "fuse"), e))
  # each env variant right after the plain variant of its mode: they change decode kernels only, so only the decode
  # forwards are captured again (the prefill forwards keep the mode's capture; TTFT is not measured again). Stamped
  # variants (FZ_TP_AR_STAMP) last: an instrument that fails on a new GPU then costs only itself (a failed mirrored
  # call ends the process; the JSON is written after every variant)
  st = lambda v: "FZ_TP_AR_STAMP" in v[2]
  variants: list[tuple[str, str, dict]] = []
  for md in modes: variants += [(md, md, {})] + [v for v in envs if v[1] == md and not st(v)]
  variants += [v for v in envs if v[1] not in modes and not st(v)] + [v for v in envs if st(v)]
  pre = a.prefix
  res: dict = {"info": {"ar": g.ar_info, "model": a.model, "random": a.random, "layers": a.layers or None, "prefix": pre,
                        "env": {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "FZ_TP_"))}},
               "steps": [], "ttft": {}, "kernels": [], "ar": []}
  env_keys = sorted({k for _, _, e in variants for k in e})
  def save() -> None:   # after every variant: a run cut short by its time limit keeps what it measured
    if a.json:
      with open(a.json, "w") as f: json.dump(res, f, indent=1)
  prev_mode, prev_env = None, {}
  for name, mode, env in variants:
    if mode in ("fuse", "car") and g.comm.car is None: print(f"TP_SKIP {name}: no custom all-reduce ({g.ar_info})", flush=True); continue
    full_env = {k: env.get(k) for k in env_keys}   # unset the other variants' keys
    scope = "decode" if (env or prev_env) and mode == prev_mode else "all"
    _all(g, "fusezap.serve.tpprof:set_mode", mode, full_env, pre + name, scope)
    prev_mode, prev_env = mode, env
    for b in buckets:
      r = {"variant": pre + name, "mode": mode, "env": env, **measure(eng, b, a.ctx, a.steps, 100 + b)}
      if tpprof.stamps_on():
        r["stamps"] = tpprof.stamp_stats(_all(g, "fusezap.serve.tpprof:stamp_read", b))
        print(f"TP_STAMPS {json.dumps({'variant': pre + name, 'b': b, **r['stamps']})}", flush=True)
      if a.graph_reps: r["graph_ms"] = [x.get("ms") for x in _all(g, "fusezap.serve.tpprof:graph_ms", b, a.graph_reps)]
      res["steps"].append(r)
      print(f"TP_STEP {json.dumps({k: v for k, v in r.items() if k != 'stamps'})}", flush=True)
    save()
    if env: continue
    res["ttft"][pre + name] = ttft(eng, a.ctx, 7)
    print(f"TP_TTFT {json.dumps({'variant': pre + name, 'ms': res['ttft'][pre + name], 'ctx': a.ctx})}", flush=True)
    if name == variants[0][0]:   # the default mode: kernel breakdown per bucket (graphs captured by measure)
      for b in [int(x) for x in a.breakdown.split(",") if x]:
        measure(eng, b, a.ctx, 4, 300 + b)
        ks = _all(g, "fusezap.serve.tpprof:kernel_breakdown", b)
        res["kernels"].append(ks)
        print(f"TP_KERNELS {json.dumps(ks)}", flush=True)
  _all(g, "fusezap.serve.tpprof:set_mode", "fuse" if g.comm.car is not None else "nccl", {k: None for k in env_keys}, "")
  D = m.cfg.hidden_size
  chains = {"1": (True,), "0": (False,), "both": (True, False)}[a.ar_chain]
  for rows in [int(x) for x in a.ar_rows.split(",") if x]:
    for md in (("auto", "base") if g.comm.car is not None else ("base",)):
      for ch in chains:
        r = _all(g, "fusezap.serve.dist:bench_allreduce", "bfloat16", rows * D, 50, md, ch)[0]
        res["ar"].append({"rows": rows, **r})
        print(f"TP_AR {json.dumps({'rows': rows, **r})}", flush=True)
  # ---- table
  vt = {}
  if a.vllm and os.path.exists(a.vllm):
    v = json.load(open(a.vllm))
    vt = {p.get("concurrency"): (p.get("tpot_ms") or {}).get("p50") for p in (v.get("performance") or {}).get("points", [])}
  res["vllm_tpot"] = vt
  print_tables(res, buckets, vt)
  for ks in res["kernels"]:
    for k in ks:
      if "classes" in k: print(f"kernels rank {k['rank']} b{k['n']}: total {k['total_ms']} ms over {k['items']} items; {k['classes']}; GEMM TB/s {k['gemm_TBps']}")
  for r in res["ar"]: print(f"all-reduce {r['rows']:4d} rows {r['path']:7s} {'chain' if r.get('chain') else 'interleaved':11s} {r['us']:8.2f} us  {r['GBps']:7.1f} GB/s")
  save()
  g.close()

if __name__ == "__main__": main()
