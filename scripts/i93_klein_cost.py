"""I93: where one FLUX.2 [klein] transformer step's time goes on NEURON (inf2 / trn1), modelled per kernel.

  PYTHONPATH=. python scripts/i93_klein_cost.py --size 512 [--size 1024] [--json out.json] [--top 25]
  FZ_KLEIN_NEURON_FUSE=1 PYTHONPATH=. python scripts/i93_klein_cost.py --size 1024   # a variant: set its switches

One eager step is recorded exactly as the chip runs it (scripts/i96_dry_step.py: real shapes, unallocated weights, every
launch recorded, nothing computed). Each distinct kernel is then rendered and its instructions recorded on NKISHIM
(record-only, scripts/nki_layer_cost.py) and list-scheduled with that script's fitted device model (PARAMS: the N8 refit to
50 trn1 kernel measurements, Qwen decode kernels; bf16 matmul rates from the step 4 probes). Per launch, the host + runtime
time is added as a separate term (HOST_MS: N7 step 4's measured runtime + host + worker per launch of a fused decode).

The model was fitted on decode kernels (a few rows, weights streamed); klein's kernels are large-M GEMMs, wide elementwise
passes and 4608-key attention. So the absolute numbers are a model estimate, not a measurement, and the script prints the
ratio against the W7 measurements (8.6 s/step at 512^2, 28.3 s/step at 1024^2 on inf2) so the reader sees how far off it
is. The split by kernel class (matmul, attention, elementwise, reduce) and the per-kernel ranking are what this is for.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MEASURED_S = {512: 8.6, 1024: 28.3}   # W7, inf2.8xlarge, s/step after load (docs/diff/i96_w7/SUMMARY.txt)
HOST_MS = 0.3   # per launch, host + runtime worker + nrt (N7 step 4: 1.23 + 0.75 + 0.30 ms over 7 launches ~ 0.33 ms)

def kclass(name: str) -> str:
  if "attention" in name: return "attention"
  if "matmul" in name: return "matmul"
  if name.startswith("fz_klein"): return "klein-fused"
  if name.startswith("r_"): return "reduce"
  if name.startswith("E_"): return "elementwise"
  return "other"

def model_step(size: int, dt_name: str = "bfloat16", log=print) -> dict:
  import i96_dry_step as D
  import nki_layer_cost as C
  from fusezap.dtype import dtypes
  items, mem = D.dry_step(size, getattr(dtypes, dt_name))
  recs = D.programs(items)
  C.install()
  rows = []
  t0 = time.perf_counter()
  for r in recs:
    sizes = [c for c, _ in r["specs"]]
    tr = C.record(r["prog"], sizes)
    s = C.schedule(tr, C.PARAMS, crit=True)
    rows.append({"kernel": r["kernel"], "class": kclass(r["kernel"]), "launches": r["launches"], "instr": tr["n"],
                 "device_ms": 1e3 * s["s"], "path_eng": {k: round(1e3 * v, 3) for k, v in s.get("path_eng", {}).items()},
                 "arg_mib": round(r["arg_bytes"] / 2**20, 1)})
  n = sum(r["launches"] for r in rows)
  dev = sum(r["device_ms"] * r["launches"] for r in rows) / 1e3
  host = n * HOST_MS / 1e3
  by = defaultdict(float)
  for r in rows: by[r["class"]] += r["device_ms"] * r["launches"] / 1e3
  eng = defaultdict(float)
  for r in rows:
    for k, v in r["path_eng"].items(): eng[k] += v * r["launches"] / 1e3
  out = {"size": size, "launches": n, "distinct": len(rows), "device_s": dev, "host_s": host, "total_s": dev + host,
         "measured_s": MEASURED_S.get(size), "by_class_s": dict(by), "critical_path_by_engine_s": dict(eng),
         "memory": mem, "model_s": round(time.perf_counter() - t0, 1), "kernels": sorted(rows, key=lambda r: -r["device_ms"] * r["launches"])}
  return out

def show(o: dict, top: int, log=print) -> None:
  m = o["measured_s"]
  log(f"klein {o['size']}^2: {o['launches']} launches ({o['distinct']} distinct); model device {o['device_s']:.2f} s + host/runtime "
      f"{o['host_s']:.2f} s = {o['total_s']:.2f} s/step [model estimate]" + (f"; W7 measured {m} s (model/measured {o['total_s'] / m:.2f})" if m else ""))
  log(f"  peak HBM in use {o['memory']['peak_in_use_bytes'] / 2**30:.2f} GiB (weights {o['memory']['weights_bytes'] / 2**30:.2f})")
  log("  by class (device s): " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(o["by_class_s"].items(), key=lambda x: -x[1])))
  log("  critical path by engine (s): " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(o["critical_path_by_engine_s"].items(), key=lambda x: -x[1])))
  log(f"  top {top} kernels (launches x device ms = s; critical path ms by engine):")
  for r in o["kernels"][:top]:
    pe = " ".join(f"{k}:{v:.2f}" for k, v in sorted(r["path_eng"].items(), key=lambda x: -x[1])[:3])
    log(f"   {r['launches']:3d} x {r['device_ms']:9.3f} ms = {r['launches'] * r['device_ms'] / 1e3:6.2f} s  {r['kernel']:42s} {r['instr']:7d} instr  {pe}")

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--size", type=int, action="append")
  ap.add_argument("--dtype", default="bfloat16")
  ap.add_argument("--top", type=int, default=25)
  ap.add_argument("--json", default=None)
  a = ap.parse_args()
  res = []
  for s in a.size or [512]:
    o = model_step(s, a.dtype)
    show(o, a.top)
    res.append(o)
  if a.json:
    with open(a.json, "w") as f: json.dump(res, f, indent=1)

if __name__ == "__main__": main()
