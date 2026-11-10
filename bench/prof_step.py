"""Engine step profile on CUDA (docs/status/SPEED.md): wall time of a steady decode step with and without CUDA graphs,
and the GPU time per kernel name of one decode forward (each recorded item timed alone with CUDA events).

  uv run python bench/prof_step.py [--model Qwen/Qwen2.5-7B-Instruct] [--seqs 1,32] [--steps 24] [--top 25] [--prefill] [--sim 32]
  uv run python bench/prof_step.py --env-ab FZ_BIAS_ROUND2 --seqs 1,32 --prefill-rows 2048   # graph GPU-time A/B of a flag

For each --seqs value: an LLMEngine like `fusezap serve` builds (bf16, 512 prompt tokens per sequence, max_prefill_tokens
256), all sequences prefilled first, then --steps decode steps are timed (engine.step(): host prep, the replay, the
logits read-back and sampling). FZ_GRAPH is toggled in-process; the per-kernel table uses the launch-by-launch path.
"""
from __future__ import annotations
import argparse, collections, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fusezap import Device
from fusezap.dtype import dtypes
from fusezap.helpers import getenv

REP = 20

def _graph(on: bool) -> None:
  os.environ["FZ_GRAPH"] = "1" if on else "0"
  getenv.cache_clear()

PREFILL = 256   # --max-prefill-tokens (the serve simulation; 0 = the engine's default)

MAX_LEN = 1024   # --max-model-len: sets the decode attention's partition count (128 keys each) like `fusezap serve`'s flag

def _engine(m, seqs: int, prefill: int = 256):
  from fusezap.serve.engine import LLMEngine
  return LLMEngine(m, num_blocks=max(1024, seqs * 48), block_size=16, max_num_seqs=seqs, max_prefill_tokens=prefill or None, max_model_len=MAX_LEN)

def decode_ms(m, seqs: int, steps: int, graph: bool) -> tuple[float, object]:
  from fusezap.serve.engine import SamplingParams
  _graph(graph)
  eng = _engine(m, seqs)
  rng = np.random.default_rng(0)
  # max_tokens well past the measured steps: the sequences prefilled first decode during the others' prefill steps, and
  # all of them must still be running when the steps (and kernel_table's) are measured at exactly `seqs` rows
  for _ in range(seqs): eng.add_request(rng.integers(1000, 30000, 512).tolist(), SamplingParams(max_tokens=min(240, steps + 200), ignore_eos=True))
  while any(s.in_prompt for s in eng.scheduler.running) or eng.scheduler.waiting: eng.step()
  for _ in range(4): eng.step()   # warm, capture, graph build
  dev = Device[m.device]
  dev.synchronize()
  ts = []
  for _ in range(steps):
    t = time.perf_counter()
    eng.step()
    ts.append(time.perf_counter() - t)
  fwd = eng._decode_fwd(seqs) if hasattr(eng, "_decode_fwd") else eng._decode   # the bucket these steps ran (speed run 4)
  if fwd.jit.captured is not None and (g := fwd.jit.captured.graph) is not None:
    eng.graph_ms = _graph_gpu_ms(dev, g, 30)
    print(f"  seqs={seqs}: the decode graph alone takes {eng.graph_ms:.3f} ms of GPU time")
  return float(np.median(ts)) * 1e3, eng

def _graph_gpu_ms(dev, g, n: int) -> float:
  """GPU time of one replay of a captured graph: n back-to-back launches between two events, after one warm launch."""
  cu = dev.cu
  e0, e1 = cu.cuEventCreate(0)[1], cu.cuEventCreate(0)[1]
  g.launch()
  dev.synchronize()
  cu.cuEventRecord(e0, 0)
  for _ in range(n): g.launch()
  cu.cuEventRecord(e1, 0)
  cu.cuEventSynchronize(e1)
  return float(cu.cuEventElapsedTime(e0, e1)[1]) / n

def kernel_table(eng, top: int, fwd=None) -> None:
  dev = Device[eng.model.device]
  fwd = fwd or eng._decode
  c = fwd.jit.captured
  items, seen, orig = c.items, [], c._launch
  c._launch = lambda ins, owned, vv: (seen.append((ins, owned)), orig(ins, owned, vv))   # one real step's buffers
  if fwd is eng._decode: eng.step()
  else: _prefill_once(eng)
  del c._launch
  ins, owned = seen[-1]
  cu = dev.cu
  e0, e1 = cu.cuEventCreate(0)[1], cu.cuEventCreate(0)[1]
  agg: dict[str, list] = collections.defaultdict(lambda: [0.0, 0])
  tot = 0.0
  try:
    for it in items:
      if it.done: continue
      c.items = [it]   # the item REP times in one graph, so the host launch cost does not count
      if (g := dev.capture_graph(lambda: [orig(ins, owned, {}) for _ in range(REP)])) is None: continue
      dev.synchronize()
      cu.cuEventRecord(e0, 0)
      g.launch()
      cu.cuEventRecord(e1, 0)
      cu.cuEventSynchronize(e1)
      ms = float(cu.cuEventElapsedTime(e0, e1)[1]) / REP
      name = it.prg.name if it.prg is not None else getattr(it.si.ast[0], "name", it.kind)
      agg[name][0] += ms; agg[name][1] += 1
      tot += ms
  finally: c.items = items
  print(f"  GPU time of one {'decode' if fwd is eng._decode else 'prefill'} forward, items timed alone: {tot:.2f} ms over {sum(n for _, n in agg.values())} launches")
  for name, (ms, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]:
    print(f"    {ms:8.3f} ms  {n:4d}x  {name[:90]}")

def _prefill_once(eng) -> None:
  """Full prefill steps of max_prefill_tokens rows: fresh 512-token prompts (max(1, rows / 512) of them, prefilled together)."""
  from fusezap.serve.engine import SamplingParams
  rng = np.random.default_rng(1)
  rids = [eng.add_request(rng.integers(1000, 30000, 512).tolist(), SamplingParams(max_tokens=1, ignore_eos=True)) for _ in range(max(1, PREFILL_ROWS // 512))]
  while any(eng.seqs[r].status.value != "finished" for r in rids): eng.step()
  for r in rids: eng.release(r)

def prefill_ms(m, steps: int, table: bool = True) -> object:
  """Wall time of a full prefill step (--prefill-rows prompt rows, nothing decoding), with graphs, then the prefill kernel table."""
  _graph(True)
  eng = _engine(m, 32, PREFILL_ROWS)
  dev = Device[m.device]
  for _ in range(3): _prefill_once(eng)   # warm, capture, graph build
  dev.synchronize()
  t = time.perf_counter()
  for _ in range(steps): _prefill_once(eng)
  ms = (time.perf_counter() - t) * 1e3 / (steps * max(1, 512 // PREFILL_ROWS))
  print(f"prefill: {ms:.2f} ms per {PREFILL_ROWS}-row prefill step with graphs ({PREFILL_ROWS / ms * 1e3:.0f} prompt tok/s)")
  if (g := eng._prefill.jit.captured.graph) is not None:
    eng.graph_ms = _graph_gpu_ms(dev, g, 8)
    print(f"  the prefill graph alone takes {eng.graph_ms:.3f} ms of GPU time")
  if not table: return eng
  _graph(False)
  eng0 = _engine(m, 32, PREFILL_ROWS)
  for _ in range(2): _prefill_once(eng0)   # the jit captures on its second call
  return eng0

def serve_sim(m, conc: int, n_req: int, prompt: int, out: int) -> None:
  """Closed loop like bench gate/c<conc>: conc requests in flight, each finished one replaced, n_req in all. Prints the
  wall time, output tok/s and the time per step kind (decode only, prefill only, both)."""
  from fusezap.serve.engine import SamplingParams
  _graph(True)
  eng = _engine(m, conc, PREFILL)
  from fusezap.serve.api import warmup
  warmup(eng)
  rng = np.random.default_rng(2)
  sp = SamplingParams(max_tokens=out, ignore_eos=True)
  for _ in range(conc): eng.add_request(rng.integers(1000, 30000, prompt).tolist(), SamplingParams(max_tokens=4, ignore_eos=True))
  while eng.has_unfinished(): eng.step()   # warm both graphs
  sent = done = 0
  for _ in range(conc): eng.add_request(rng.integers(1000, 30000, prompt).tolist(), sp); sent += 1
  kinds: dict[str, list] = collections.defaultdict(lambda: [0.0, 0])
  Device[m.device].synchronize()
  t0 = time.perf_counter()
  while eng.has_unfinished():
    d0, p0 = eng.stats["decode_steps"], eng.stats["prefill_steps"]
    t = time.perf_counter()
    outs = eng.step()
    k = ("D" if eng.stats["decode_steps"] > d0 else "") + ("P" if eng.stats["prefill_steps"] > p0 else "")
    kinds[k][0] += time.perf_counter() - t; kinds[k][1] += 1
    for o in outs:
      if o.finished:
        done += 1
        eng.release(o.request_id)
        if sent < n_req: eng.add_request(rng.integers(1000, 30000, prompt).tolist(), sp); sent += 1
  wall = time.perf_counter() - t0
  print(f"serve sim c{conc} (prefill sizes {eng.prefill_sizes}): {n_req} x {prompt}/{out}: {wall:.2f} s, {n_req * out / wall:.1f} output tok/s")
  for k, (s, n) in sorted(kinds.items()): print(f"  steps {k or '-':2s} {n:5d}  {s:7.2f} s  {s / max(n, 1) * 1e3:7.2f} ms/step")

def env_ab(m, spec: str, seqs: list[int], rounds: int, prefill: bool = True) -> None:
  """A/B of an env flag read when an engine is built or a forward is traced: spec NAME (values 0 and 1) or
  NAME=v1,v2,... Fresh engines per value, alternated over `rounds`, each timing the decode graph alone at every --seqs
  and (prefill) the --prefill-rows prefill graph alone (GPU ms)."""
  from fusezap.helpers import getenv
  name, _, vals = spec.partition("=")
  values = vals.split(",") if vals else ["0", "1"]
  res: dict[tuple, list[float]] = collections.defaultdict(list)
  old = os.environ.get(name)
  for r in range(rounds):
    for v in values:
      os.environ[name] = v
      getenv.cache_clear()
      for n in seqs:
        _, eng = decode_ms(m, n, 8, True)
        res[(v, f"decode seqs={n}")].append(getattr(eng, "graph_ms", float("nan")))
        del eng
      if prefill:
        eng = prefill_ms(m, 2, table=False)
        res[(v, f"prefill {PREFILL_ROWS} rows")].append(getattr(eng, "graph_ms", float("nan")))
        del eng
  if old is None: os.environ.pop(name, None)
  else: os.environ[name] = old
  getenv.cache_clear()
  print(f"env A/B {name} (GPU ms of the graph alone, {rounds} rounds alternated over {', '.join(values)}):")
  for what in dict.fromkeys(k for _, k in res):
    base = float(np.median(res[(values[0], what)]))
    cells = []
    for v in values:
      x = res[(v, what)]
      cells.append(f"{name}={v} {float(np.median(x)):8.3f} ms ({100 * (float(np.median(x)) / base - 1):+6.2f}%; {' '.join(f'{y:.3f}' for y in x)})")
    print(f"  {what:22s} " + "   ".join(cells))

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
  ap.add_argument("--seqs", default="1,32")
  ap.add_argument("--steps", type=int, default=24)
  ap.add_argument("--top", type=int, default=25)
  ap.add_argument("--prefill", action="store_true", help="also time a prefill step and list its kernels")
  ap.add_argument("--prefill-rows", type=int, default=256, help="the --prefill step's rows (512-token prompts prefilled together)")
  ap.add_argument("--sim", type=int, default=0, help="also run a closed-loop serve simulation at this concurrency (gate shape)")
  ap.add_argument("--max-prefill-tokens", type=int, default=256, help="the simulation's prefill budget (0: the engine default)")
  ap.add_argument("--max-model-len", type=int, default=1024, help="the engines' max_model_len (Invarix serves 2048, the bench 768)")
  ap.add_argument("--no-table", action="store_true", help="skip the per-kernel tables (step and graph times only)")
  ap.add_argument("--env-ab", default="", help="A/B an env flag read at trace time: NAME (0 vs 1) or NAME=v1,v2,...; graph GPU times only")
  ap.add_argument("--ab-no-prefill", action="store_true", help="--env-ab: decode graphs only")
  ap.add_argument("--ab-rounds", type=int, default=2)
  a = ap.parse_args()
  global MAX_LEN
  MAX_LEN = a.max_model_len
  global PREFILL, PREFILL_ROWS
  PREFILL, PREFILL_ROWS = a.max_prefill_tokens, a.prefill_rows
  from fusezap.serve.models import load_model
  m = load_model(a.model, dtype=dtypes.bfloat16, device="CUDA", max_len=max(1024, a.max_model_len), prefill_chunk=128)
  print(f"{a.model} on {Device[m.device].gpu_name}")
  seqs_l = [int(s) for s in a.seqs.split(",") if s]
  if a.env_ab:
    env_ab(m, a.env_ab, seqs_l, a.ab_rounds, not a.ab_no_prefill)
    return 0
  if a.sim: serve_sim(m, a.sim, 2 * a.sim, 512, 128)
  if a.prefill:
    eng0 = prefill_ms(m, 4, table=not a.no_table)
    if not a.no_table: kernel_table(eng0, a.top, eng0._prefill)
    del eng0
  for seqs in seqs_l:
    g_ms, eng = decode_ms(m, seqs, a.steps, True)
    built = eng._decode.jit.captured.graph is not None
    p_ms, eng0 = decode_ms(m, seqs, a.steps, False)
    print(f"seqs={seqs}: decode step p50 {g_ms:.2f} ms with graphs (built: {built}), {p_ms:.2f} ms launch by launch; "
          f"{len(eng0._decode.jit.captured.items)} items; host time per graph step about {g_ms - getattr(eng, 'graph_ms', g_ms):.2f} ms")
    if not a.no_table: kernel_table(eng0, a.top)
    del eng, eng0
  _graph(True)
  return 0

if __name__ == "__main__": sys.exit(main())
