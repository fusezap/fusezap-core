"""I69 follow-up: where a served request's time to first token goes, from Invarix's per-request rows (requests.csv: send_unix,
ttft_ms) joined with `fusezap serve`'s request traces (FZ_REQ_TRACE=<prefix>: <prefix>.api.jsonl, <prefix>.engine.jsonl).

  python scripts/nki_i69_ttft_breakdown.py <invarix perf dir or requests.csv> <trace prefix> [--conc 1] [--json out.json]

Per request (ms; all wall clocks of one machine):
  http_in   client send -> handler entered (the client's request, uvicorn's parser, starlette routing)
  parse     handler entered -> sent to the engine process (body JSON, checks, sampling params)
  pipe_in   -> the engine process received the add
  wait      -> the step that first scheduled it began (the loop finishing a step, the admit gate)
  step      -> that step ended with its first token (the prefill forward: host + device)
  pipe_out  -> the API handler took the first output (engine pipe, reader thread, event loop)
  http_out  -> the client parsed the first token (the response start when not sent earlier, SSE chunk, client)
and their medians; the sum equals the client's TTFT. Requests without a trace match are skipped (counted)."""
from __future__ import annotations
import argparse, csv, json, os, sys
import numpy as np

PARTS = ("http_in", "parse", "pipe_in", "wait", "step", "pipe_out", "http_out")

def load(perf: str, prefix: str, conc: int | None = 1) -> tuple[list[dict], int]:
  if perf.endswith(".json"):   # bench/serveq_sim.py --client --json: per point the TTFTs and send times
    d = json.load(open(perf)).get(f"c{conc or 1}", {})
    rows = [{"send_unix": su, "ttft_ms": t} for su, t in zip(d.get("send_unix", []), d.get("ttft", []))]
  else:
    rows_path = perf if perf.endswith(".csv") else os.path.join(perf, "requests.csv")
    rows = [r for r in csv.DictReader(open(rows_path)) if r.get("status") == "ok" and r.get("phase", "performance") == "performance"
            and (conc is None or int(r["concurrency"]) == conc)]
  api = [json.loads(l) for l in open(prefix + ".api.jsonl") if l.strip()] if os.path.exists(prefix + ".api.jsonl") else []
  eng = {}
  if os.path.exists(prefix + ".engine.jsonl"):
    for l in open(prefix + ".engine.jsonl"):
      if l.strip(): d = json.loads(l); eng[d["rid"]] = d
  out, miss = [], 0
  recvs = np.array([a["recv"] for a in api]) if api else np.zeros(0)
  for r in rows:
    s, ttft = float(r["send_unix"]), float(r["ttft_ms"])
    if not len(recvs): miss += 1; continue
    i = int(np.argmin(np.where(recvs >= s - 1e-4, recvs - s, np.inf)))
    a = api[i]
    e = eng.get(a.get("rid"))
    if not (0 <= a["recv"] - s < 0.05) or e is None or "first" not in a or "first_out" not in e or "sched" not in e: miss += 1; continue
    t_first = s + ttft / 1e3
    pts = (s, a["recv"], a["submit"], e["add"], e["sched"], e["first_out"], a["first"], t_first)
    d = {k: round(1e3 * (pts[j + 1] - pts[j]), 3) for j, k in enumerate(PARTS)}
    d["ttft"] = ttft
    out.append(d)
  return out, miss

def summarize(recs: list[dict]) -> dict:
  if not recs: return {}
  return {k: round(float(np.median([r[k] for r in recs])), 3) for k in (*PARTS, "ttft")}

def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("perf"); ap.add_argument("prefix")
  ap.add_argument("--conc", type=int, default=1)
  ap.add_argument("--label", default="")
  ap.add_argument("--json", default=None)
  a = ap.parse_args()
  try: recs, miss = load(a.perf, a.prefix, a.conc)
  except FileNotFoundError as e:
    print(f"{a.label or a.perf}: no data ({e})"); return 0
  m = summarize(recs)
  lab = a.label or os.path.basename(a.perf.rstrip("/"))
  if not m: print(f"{lab}: c{a.conc} no traced requests ({miss} unmatched)"); return 0
  host = round(m["ttft"] - m["step"], 3)
  print(f"{lab}: c{a.conc} TTFT medians (ms, n={len(recs)}, {miss} unmatched): " + "  ".join(f"{k} {m[k]}" for k in PARTS) +
        f"  | TTFT {m['ttft']} = step + {host} outside it")
  if a.json:
    with open(a.json, "w") as f: json.dump({"label": lab, "conc": a.conc, "n": len(recs), "unmatched": miss, "median": m, "requests": recs}, f, indent=1)
  return 0

if __name__ == "__main__": sys.exit(main())
