"""Summarize an FZ_STEP_PROF file (fusezap/serve/stepprof.py), optionally per load window of bench/prof_window.py.

  python bench/step_prof_summary.py steps.jsonl [--windows windows.jsonl] [--json out.json]

Per window (or for the whole file): steps by kind (D decode only, P prefill only, DP both), and for each kind the median
of every host phase (sched, d_prep, d_call, s_prep, s_call, p_prep, p_call, post, total, in ms) and the GPU time of the
forwards that completed in the step (gpu_d, gpu_p) and the GPU idle before them (gap). Then the window's time split: GPU inside forwards, GPU idle between them (gaps), and the share of
wall time spent outside step() (the server loop).
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np

FIELDS = ("sched", "d_prep", "d_call", "s_prep", "s_call", "p_prep", "p_call", "post", "total", "gpu_d", "gpu_p", "gap")

def _derive(r: dict) -> dict:
  """GPU fields from the step's completed forwards (fusezap/serve/stepprof.py `fwd`: [kind, ms, idle ms before])."""
  if "fwd" in r:
    f = r["fwd"]
    r = {**r, "gpu_d": sum(x[1] for x in f if x[0] in "ds"), "gpu_p": sum(x[1] for x in f if x[0] == "p"),
         "gap": sum(x[2] for x in f if x[2] is not None and x[2] >= 0)}
  return r

def kind(r: dict) -> str: return ("D" if r.get("rows_d") else "") + ("P" if r.get("size_p") else "") or "-"

def summarize(steps: list[dict], wall_s: float | None = None) -> dict:
  steps = [_derive(r) for r in steps]
  out: dict = {"steps": len(steps), "kinds": {}}
  for k in ("D", "P", "DP"):
    rs = [r for r in steps if kind(r) == k]
    if not rs: continue
    d = {"n": len(rs), "rows_d_median": float(np.median([r["rows_d"] for r in rs])), "total_s": round(sum(r["total"] for r in rs) / 1e3, 4),
         "gpu_s": round(sum(r.get("gpu_d", 0) + r.get("gpu_p", 0) for r in rs) / 1e3, 4), "gap_s": round(sum(r.get("gap", 0) for r in rs) / 1e3, 4)}
    if k != "D": d["size_p"] = sorted({r["size_p"] for r in rs})
    for f in FIELDS:
      v = [r[f] for r in rs if f in r and r[f] is not None and r[f] >= 0]
      if v: d[f] = round(float(np.median(v)), 4)
    out["kinds"][k] = d
  if wall_s and steps:
    gpu = sum(r.get("gpu_d", 0) + r.get("gpu_p", 0) for r in steps) / 1e3
    gaps = sum(r.get("gap", 0) for r in steps) / 1e3
    in_step = sum(r["total"] for r in steps) / 1e3
    has_gpu = any("fwd" in r for r in steps)
    out.update({"wall_s": round(wall_s, 4), "gpu_in_forwards_pct": round(100 * gpu / wall_s, 2) if has_gpu else None,
                "gpu_gaps_pct": round(100 * gaps / wall_s, 2) if has_gpu else None, "host_in_step_pct": round(100 * in_step / wall_s, 2), "outside_step_pct": round(100 * (1 - in_step / wall_s), 2),
                "steps_per_s": round(len(steps) / wall_s, 2)})
  return out

def show(title: str, s: dict) -> None:
  gpu = f"GPU in forwards {s['gpu_in_forwards_pct']}%, GPU idle gaps {s['gpu_gaps_pct']}%; " if s.get("gpu_in_forwards_pct") is not None else "no GPU events (not CUDA); "
  print(f"== {title}: {s['steps']} steps" + (f", {s['wall_s']:.2f} s: {s['steps_per_s']} steps/s; {gpu}host inside step() {s['host_in_step_pct']}%, "
        f"outside (server loop) {s['outside_step_pct']}%" if "wall_s" in s else ""))
  for k, d in s["kinds"].items():
    print(f"  {k:2s} n={d['n']:5d} in step() {d['total_s']:.3f} s, GPU {d['gpu_s']:.3f} s, GPU idle {d['gap_s']:.3f} s; rows_d~{d['rows_d_median']:.0f}" +
          (f" size_p {d['size_p']}" if "size_p" in d else "") + "  " +
          "  ".join(f"{f} {d[f]:.3f}" for f in FIELDS if f in d))

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("steps")
  ap.add_argument("--windows", default="")
  ap.add_argument("--json", default="")
  a = ap.parse_args()
  steps = [json.loads(l) for l in open(a.steps) if l.strip()]
  res: dict = {}
  if a.windows:
    for w in (json.loads(l) for l in open(a.windows) if l.strip()):
      sel = [r for r in steps if w["t0"] <= r["t"] <= w["t1"]]
      s = summarize(sel, w["t1"] - w["t0"])
      s["window"] = {k: w.get(k) for k in ("mode", "conc", "tok_s", "itl_ms_p50", "itl_ms_p99", "round_s_median")}
      res[w["label"]] = s
      show(f"{w['label']} ({w['mode']}, conc {w['conc']}, {w.get('tok_s', '-')} tok/s)", s)
  else:
    res["all"] = summarize(steps)
    show("all steps", res["all"])
  if a.json:
    with open(a.json, "w") as f: json.dump(res, f, indent=1)
  return 0

if __name__ == "__main__": sys.exit(main())
