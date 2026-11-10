"""I77: one prefill layer at Qwen2.5-7B's TP=2 shard shapes on ONE NeuronCore (no collectives), old path (the member kernels)
vs the prefill kernels (FZ_NEURON_PREFILL_KERNEL=v2): device time per fused layer NEFF, the bits, and the NEFFs for
neuron-profile (scripts/nki_n8_prefill_inf2.sh --pfk runs `neuron-profile capture` / `view` on them: a per-engine
timeline of one layer that holds no collective, so it runs alone).

  FZ_DEVICE=NEURON uv run python scripts/nki_pfk_bench.py --out results/pfk --configs v1:128:128,v2:128:128,v2:512:512
  FZ_DEVICE=NKISHIM uv run python scripts/nki_pfk_bench.py --hidden 256 --inter 384 --configs v1:64:128,v2:64:128   # Mac dry run

A random 3-layer bf16 model of the shard's shapes (D 3584, I 9472, 14 / 2 heads, head_dim 128, vocab 512), FZ_NEURON_FUSE=1
(one fused kernel per layer, the head apart): per config <path>:<rows>:<pool slots> (path v1 | v2 (ScalarE exp) | v2exact (the default exp)) a prefill of
`rows` tokens, captured, then --replays replays with the allocator's profile on: per fused kernel the runtime's execute
ms and nrtpy's device-time benchmark (device_ms); the middle layer's kernel (layer 1) is the per-layer number. Logits of
every config vs the v1 config of the same rows and slots (bitwise expected for v2exact; v2 within tolerance). Before a v2
config's capture the serving path's device self-test runs at its shapes (neuron_prefill.selftest, as LLMEngine's gate) and
its fallback knobs apply (I77 run 1: ScalarE PSUM evictions do not compile on NeuronCore-v2, VectorE ones do; the bench
captured with the env's knobs then and failed); each record carries the self-test and the knobs. Writes
<out>/pfk_bench.json (records, and profile_neffs: [{config, kernel, neff}] for the layer-1 kernels).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def log(*a) -> None: print(*a, flush=True)

def selftest(dev: str, m, rows: int, ctx: int, log=log) -> dict:
  """neuron_prefill's device self-test at this model's shapes (rows, pool prefix ctx), as LLMEngine's gate runs it before a
  capture, and its verdict applied (set_ok: the fallback knobs, e.g. VectorE evictions, for the kernels captured next).
  NEURON only (elsewhere it passes without running). {ok, level, knobs, why, seconds}."""
  from fusezap.serve import neuron_prefill as PF
  r = PF.selftest(type("G", (), {"device": dev, "rank": 0})(), m, (rows,), (ctx,), log)
  PF.set_ok(None, bool(r.get("ok")), r.get("why", ""), r.get("knobs") if r.get("ok") else None)
  return {k: (v[:600] if isinstance(v, str) else v) for k, v in r.items() if k in ("ok", "level", "knobs", "why", "seconds")}

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="")
  ap.add_argument("--configs", default="v1:128:128,v2exact:128:128,v2:128:128,v2exact:512:512,v2:512:512")
  ap.add_argument("--replays", type=int, default=3)
  ap.add_argument("--hidden", type=int, default=3584)
  ap.add_argument("--inter", type=int, default=9472)
  ap.add_argument("--heads", type=int, default=14)
  ap.add_argument("--kv", type=int, default=2)
  ap.add_argument("--hd", type=int, default=128)
  ap.add_argument("--budget-s", type=float, default=0, help="stop starting new configs after this many seconds (0: no limit)")
  a = ap.parse_args()
  os.environ.update({"FZ_NEURON_FUSE": "1", "FZ_NEURON_HEAD_FUSE": "0", "FZ_NEURON_DTYPE": "bf16", "FZ_NEURON_BF16_HEAD": "0", "FZ_NEURON_PLAN": "0"})
  from fusezap.device import Device
  from fusezap.device.neuron import FusedCall, neff_cache_dir, neuron_profile
  from fusezap.serve import neuron_kv as kv, neuron_prefill as PF
  from nki_n7_layer_bench import qwen_shaped
  dev = Device.DEFAULT
  assert kv.is_neuron(dev), f"FZ_DEVICE must be a NEURON tier, got {dev}"
  D = Device[dev]
  al = getattr(D, "allocator", None)
  timed = hasattr(al, "profile")
  out: dict = {"args": vars(a), "records": [], "profile_neffs": []}
  def save() -> None:
    if a.out:
      os.makedirs(a.out, exist_ok=True)
      with open(os.path.join(a.out, "pfk_bench.json"), "w") as f: json.dump(out, f, indent=1, default=str)
  t_all = time.perf_counter()
  models: dict = {}
  ref: dict = {}
  for cfg in a.configs.split(","):
    if a.budget_s and time.perf_counter() - t_all > a.budget_s: log(f"pfk_bench: budget reached, {cfg} and the rest not run"); break
    path, rows, ctx = cfg.split(":")
    rows, ctx = int(rows), int(ctx)
    mode, exp = ("v1", "exact") if path == "v1" else ("v2", "exact" if path.endswith("exact") else "act")
    rec: dict = {"config": cfg, "path": path, "rows": rows, "ctx": ctx}
    try:
      os.environ["FZ_NEURON_PFK_EXP"] = exp
      PF.set_mode(None, mode)
      if ctx not in models: models[ctx] = qwen_shaped(dev, 1, ctx, layers=3, hidden=a.hidden, inter=a.inter, heads=a.heads, kv=a.kv, hd=a.hd)
      m = models[ctx]
      if mode == "v2":   # the serving path's self-test and its fallback knobs (I77 run 1: ScalarE evictions fail on NeuronCore-v2)
        rec["selftest"] = st = selftest(dev, m, rows, ctx, log)
        if not st.get("ok"): raise RuntimeError(f"prefill kernels self-test failed at every fallback level: {st.get('why', '')[:600]}")
      rec["knobs"] = PF._knobs() if mode == "v2" else None
      m.prefill_chunk, m._neuron_steps = rows, None   # a fresh NeuronSteps: its own jits (this path) and a zeroed pool
      ids = [[(7 * i + 3) % 500 for i in range(rows)]]
      st = time.perf_counter()
      for _ in range(3): m.reset(); lg = m.prefill(ids)   # eager (member NEFFs compile), capture, first replay (the fused NEFFs)
      rec["first_calls_s"] = round(time.perf_counter() - st, 1)
      items = [it for it in m._neuron_steps.jits[(rows, 1)].captured.items if not it.const]
      names = [it.si.ast[0].spec.name if it.si is not None and getattr(it.si.ast[0], "spec", None) is not None else it.kind for it in items]
      fused = [it for it in items if it.si is not None and isinstance(it.si.ast[0], FusedCall)]
      rec["kernels"] = names
      rec["layer_kernel"] = mid = fused[1].si.ast[0].spec.name if len(fused) > 1 else None
      rec["members"] = [p[0] for p in fused[1].si.ast[0].spec.parts] if len(fused) > 1 else []
      if timed:
        al.flush(); al.profile, al.prof, al.prof_wall, al.prof_ipc, al.bindings = True, {}, [0, 0.0], [0, 0.0, 0.0], {}
        t0 = time.perf_counter()
        for _ in range(a.replays): m.reset(); lg = m.prefill(ids)
        rec["prefill_wall_ms"] = round(1e3 * (time.perf_counter() - t0) / a.replays, 3)
        al.flush(); al.profile = False
        rep = neuron_profile(al, iters=10, bench=True)
        rec["exec_ms"] = {nm: k["exec_ms_mean"] for nm, k in rep["kernels"].items()}
        rec["device_ms"] = {nm: k["device_ms"] for nm, k in rep["kernels"].items()}
        rec["layer_device_ms"] = rec["device_ms"].get(mid)
        rec["layer_exec_ms"] = rec["exec_ms"].get(mid)
        if rep.get("bench_errors"): rec["bench_errors"] = rep["bench_errors"]
        prg = D.spec_program(fused[1].si.ast[0].spec)
        for key, *_ in getattr(prg, "loaded", {}).values():
          out["profile_neffs"].append({"config": cfg, "kernel": mid, "neff": str(neff_cache_dir() / key / "kernel.neff")})
      lg = np.asarray(lg)
      if path == "v1": ref[(rows, ctx)] = lg
      if (rows, ctx) in ref:
        r = ref[(rows, ctx)]
        rec["logits_bitwise_vs_v1"] = bool(np.array_equal(lg.view(np.int32), r.view(np.int32)))
        rec["logits_maxdiff_vs_v1"] = float(np.abs(lg - r).max())
        rec["top1_vs_v1"] = bool((lg.argmax(-1) == r.argmax(-1)).all())
    except Exception as e:
      import traceback
      rec["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
      log(traceback.format_exc()[-3000:])
    finally:
      PF.set_mode(None, None)
      PF._STATE.update(ok=None, why="", knobs=None)   # the next config tests its own kernels
    out["records"].append(rec)
    save()
    log(f"pfk_bench {cfg}: layer {rec.get('layer_kernel')} device {rec.get('layer_device_ms')} ms, execute {rec.get('layer_exec_ms')} ms; "
        f"prefill wall {rec.get('prefill_wall_ms')} ms; bitwise vs v1 {rec.get('logits_bitwise_vs_v1')} (max diff {rec.get('logits_maxdiff_vs_v1')}); "
        f"first calls {rec.get('first_calls_s')} s" + (f"; ERROR {rec['error'][:300]}" if "error" in rec else ""))
  log("==== pfk_bench summary (one NeuronCore, shard shapes; ms per layer kernel)")
  for r in out["records"]:
    st = r.get("selftest") or {}
    log(f"  {r['config']:<16} device {r.get('layer_device_ms')}  execute {r.get('layer_exec_ms')}  bitwise vs v1 {r.get('logits_bitwise_vs_v1')}  "
        f"members {len(r.get('members') or [])}" + (f"  self-test level {st.get('level')} knobs {r.get('knobs')}" if st else "") +
        (f"  ERROR {r['error'][:200]}" if "error" in r else ""))
  save()

if __name__ == "__main__": main()
