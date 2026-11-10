"""Where the time between closed-loop waves goes (speed run 4 round 12), from `fusezap serve`'s FZ_REQ_TRACE files and a
Invarix requests.csv (send_unix / done_unix).

  python bench/turnaround.py --api t.api.jsonl --engine t.engine.jsonl [--csv requests.csv --release candidate] [--min-wave 30]

A wave = requests whose engine `fin` times lie within --gap ms of each other; only boundaries between two waves of at least
--min-wave requests (the c=32 performance points) are shown. Per boundary, in ms after the wave's last engine fin:
  out        the engine sent the step's outputs down the pipe
  api_last   the API handler got the final output (max over the wave)      api_done   [DONE] written (max)
  cl_done    the client saw the last [DONE] (requests.csv, --csv)          cl_send    the client sent the next requests (first / last)
  recv       the API handler entered (next wave, first / last)             submit     sent to the engine (last)
  add        the engine got the add (first / last)                          sched      first scheduled (first / median / last)
Then the medians over all boundaries.
"""
from __future__ import annotations
import argparse, csv, json
import numpy as np

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--api", required=True); ap.add_argument("--engine", required=True); ap.add_argument("--csv")
  ap.add_argument("--release", default="candidate"); ap.add_argument("--phase", default="performance")
  ap.add_argument("--gap", type=float, default=30.0); ap.add_argument("--min-wave", type=int, default=30)
  a = ap.parse_args()
  api = {r["rid"]: r for r in map(json.loads, open(a.api)) if r.get("rid") is not None}
  eng = {r["rid"]: r for r in map(json.loads, open(a.engine))}
  recs = sorted(({**{"e_" + k: v for k, v in eng[r].items()}, **{"a_" + k: v for k, v in api.get(r, {}).items()}} for r in eng if "fin" in eng[r]),
                key=lambda d: d["e_fin"])
  W, cur = [], []
  for d in recs:
    if cur and d["e_fin"] - cur[-1]["e_fin"] > a.gap / 1e3: W.append(cur); cur = []
    cur.append(d)
  W += [cur] if cur else []
  cs = []
  if a.csv:
    cs = sorted((float(r["send_unix"]), float(r["done_unix"])) for r in csv.DictReader(open(a.csv))
                if r.get("send_unix") and r.get("done_unix") and r["phase"] == a.phase and r["release"] == a.release)
  print(f"{len(recs)} traced requests, {len(W)} waves, {sum(len(w) >= a.min_wave for w in W)} of >= {a.min_wave}")
  agg: dict = {}
  for k in range(len(W) - 1):
    w, w2 = W[k], W[k + 1]
    if len(w) < a.min_wave or len(w2) < a.min_wave: continue
    t0 = max(d["e_fin"] for d in w)
    def f(ds, key, fn=np.median):
      xs = [d[key] - t0 for d in ds if key in d]
      return float(fn(xs)) * 1e3 if xs else float("nan")
    first_add = min(d["e_add"] for d in w2)
    sends = [s for s, _ in cs if t0 - 0.05 < s < first_add + 0.05]
    dones = [e for _, e in cs if t0 - 0.01 < e < t0 + 0.3]
    v = {"out": f(w, "e_out", max), "api_last": f(w, "a_last", max), "api_done": f(w, "a_done", max),
         "cl_done": (max(dones) - t0) * 1e3 if dones else float("nan"),
         "cl_send0": (min(sends) - t0) * 1e3 if sends else float("nan"), "cl_send1": (max(sends) - t0) * 1e3 if sends else float("nan"),
         "recv0": f(w2, "a_recv", min), "recv1": f(w2, "a_recv", max), "submit1": f(w2, "a_submit", max),
         "add0": f(w2, "e_add", min), "add1": f(w2, "e_add", max), "sched0": f(w2, "e_sched", min), "sched_med": f(w2, "e_sched"), "sched1": f(w2, "e_sched", max)}
    for kk, x in v.items(): agg.setdefault(kk, []).append(x)
    print("  " + "  ".join(f"{kk} {x:6.1f}" for kk, x in v.items()) + f"  (next wave {len(w2)})")
  if agg: print("median over boundaries: " + "  ".join(f"{kk} {np.nanmedian(x):.1f}" for kk, x in agg.items()))

if __name__ == "__main__": main()
