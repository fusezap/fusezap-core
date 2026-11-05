"""Summarize a FZ_SERVE_PROFILE file (fusezap/serve/serveprof.py): where `fusezap serve`'s time per token goes.

  python scripts/serve_prof_summary.py prof.jsonl [--label X] [--json out.json]

Periods (with --requests, Validate's requests.csv: one per concurrency and repeat; else busy periods, steps less than --idle
seconds apart) are reported one by one: wall, tokens/s, then per step kind (d decode only, p prefill only, dp both) the count, mean ms, the engine thread's
CPU ms, the ms inside runtime-worker round trips (dev) and the pool prefixes; the loop between steps (gap: adds, hand-off,
stats) and the hand-off alone (out); decode-only steps by rows; per kind how many steps' slots needed fit 512 / 768 / 1024. Then the HTTP side: per request the handler's lag behind
the engine (median of the requests' p50 and the worst p99) and its ms per output.
"""
from __future__ import annotations
import argparse, collections, json, statistics as stx

FITS = (512, 768, 1024)   # N7 step 3: how many steps of a kind fit each pool prefix (FZ_NEURON_CTX=mid adds 768)

def pct(xs: list[float], p: float) -> float | None:
  xs = sorted(x for x in xs if x is not None)
  return round(xs[min(len(xs) - 1, int(p * len(xs)))], 3) if xs else None

def mean(xs: list) -> float | None:
  xs = [x for x in xs if x is not None]
  return round(stx.fmean(xs), 3) if xs else None

def periods(steps: list[dict], idle: float) -> list[list[dict]]:
  out: list[list[dict]] = []
  for s in steps:
    if out and s["t"] - (out[-1][-1]["t"] + out[-1][-1]["ms"] / 1e3) < idle: out[-1].append(s)
    else: out.append([s])
  return out

def windows(paths: list[str]) -> list[tuple[str, float, float]]:
  """Validate's requests.csv: per (concurrency, repeat) the wall-clock window from the first send to the last done."""
  import csv
  w: dict = {}
  for p in paths:
    for r in csv.DictReader(open(p)):
      if not r.get("send_unix") or not r.get("done_unix"): continue
      k = f"c{r['concurrency']} r{r.get('repeat', 0)}"
      a, b = float(r["send_unix"]), float(r["done_unix"])
      lo, hi = w.get(k, (a, b))
      w[k] = (min(lo, a), max(hi, b))
  return sorted(((k, a, b) for k, (a, b) in w.items()), key=lambda x: x[1])

def summarize(recs: list[dict], idle: float = 0.5, wins: list[tuple[str, float, float]] | None = None) -> dict:
  steps = sorted((r for r in recs if r.get("k") == "step"), key=lambda r: r["t"])
  reqs = [r for r in recs if r.get("k") == "req"]
  res = {"periods": [], "requests": {}}
  if wins:   # a request's line is written when it ends: the window's requests end inside it
    groups = [(k, [s for s in steps if a <= s["t"] <= b], [r for r in reqs if a <= r["t"] <= b + 0.05]) for k, a, b in wins]
  else: groups = [(f"period {i}", ps, []) for i, ps in enumerate(periods(steps, idle))]
  for name, ps, rq in groups:
    if len(ps) < 4: continue
    wall = ps[-1]["t"] + ps[-1]["ms"] / 1e3 - ps[0]["t"]
    toks = sum(s["n"] for s in ps)   # outputs handed off ~ tokens (one per output and step)
    per = {"name": name, "steps": len(ps), "wall_s": round(wall, 3), "max_running": max(s["run"] for s in ps), "outputs": toks,
           "outputs_per_s": round(toks / wall, 1) if wall > 0 else None,
           "step_ms_sum": round(sum(s["ms"] for s in ps), 1), "gap_ms_sum": round(sum(s["gap"] or 0 for s in ps), 1), "kinds": {}}
    for kind in ("d", "p", "dp"):
      ks = [s for s in ps if s["kind"] == kind]
      if not ks: continue
      ctx = collections.Counter(tuple(s.get("ctx") or ()) for s in ks)
      per["kinds"][kind] = {"n": len(ks), "ms_mean": mean([s["ms"] for s in ks]), "ms_p50": pct([s["ms"] for s in ks], .5),
                            "ms_sum": round(sum(s["ms"] for s in ks), 1), "cpu_mean": mean([s["cpu"] for s in ks]),
                            "dev_mean": mean([s.get("dev") for s in ks]), "tok_mean": mean([s["tok"] for s in ks]),
                            "ctx": {"/".join(map(str, c)) or "-": n for c, n in ctx.most_common()},
                            "top_mean": mean([s.get("top") for s in ks]), "top_max": max((s.get("top") or 0) for s in ks),
                            "fits": {c: sum(1 for s in ks if s.get("top") and s["top"] <= c) for c in FITS}}
    by_rows: dict = collections.defaultdict(list)
    for s in ps:
      if s["kind"] == "d": by_rows[s["tok"]].append(s)
    per["decode_by_rows"] = {r: {"n": len(v), "ms_mean": mean([s["ms"] for s in v]), "dev_mean": mean([s.get("dev") for s in v]),
                                 "gap_mean": mean([s["gap"] for s in v])} for r, v in sorted(by_rows.items())}
    per["gap_ms"] = {"mean": mean([s["gap"] for s in ps]), "p50": pct([s["gap"] for s in ps], .5), "p99": pct([s["gap"] for s in ps], .99)}
    per["out_ms_mean"] = mean([s.get("out") for s in ps])
    per["http"] = http(rq)
    res["periods"].append(per)
  res["requests"] = http(reqs)
  return res

def http(reqs: list[dict]) -> dict:
  if not reqs: return {}
  return {"n": len(reqs), "lag_p50_median": pct([r["lag"][0] for r in reqs], .5), "lag_p99_max": max((r["lag"][1] or 0) for r in reqs),
          "lag_max": max((r["lag"][2] or 0) for r in reqs), "handler_ms_per_output": mean([r["hms"][0] for r in reqs]),
          "chunks_mean": mean([r["chunks"] for r in reqs]), "coalesced_mean": mean([r["coal"] for r in reqs])}

def http_line(q: dict) -> str:
  return (f"{q['n']} requests, handler lag behind the engine p50 {q['lag_p50_median']} ms (worst p99 {q['lag_p99_max']}, max {q['lag_max']}), "
          f"{q['handler_ms_per_output']} ms per output, {q['chunks_mean']} outputs per request, {q['coalesced_mean']} coalesced")

def report(res: dict, label: str = "") -> str:
  L = []
  for p in res["periods"]:
    L.append(f"serve_prof{' ' + label if label else ''} {p['name']}: {p['steps']} steps in {p['wall_s']} s, up to {p['max_running']} running, "
             f"{p['outputs']} outputs = {p['outputs_per_s']}/s; in steps {p['step_ms_sum']} ms, between steps {p['gap_ms_sum']} ms "
             f"(gap mean {p['gap_ms']['mean']} p50 {p['gap_ms']['p50']} p99 {p['gap_ms']['p99']} ms; hand-off {p['out_ms_mean']} ms)")
    for k, v in p["kinds"].items():
      L.append(f"  {k:<2} x{v['n']:<5} step {v['ms_mean']} ms (p50 {v['ms_p50']}, sum {v['ms_sum']}), cpu {v['cpu_mean']}, dev {v['dev_mean']}, "
               f"tokens {v['tok_mean']}, ctx {v['ctx']} (slots needed mean {v['top_mean']}, max {v['top_max']}; "
               f"fit {', '.join(f'<={c}: {n}' for c, n in v['fits'].items())})")
    if p["decode_by_rows"]:
      L.append("  decode-only by rows: " + "; ".join(f"{r} rows x{v['n']} {v['ms_mean']} ms (dev {v['dev_mean']}, gap {v['gap_mean']})"
                                                     for r, v in p["decode_by_rows"].items()))
    if p["http"]: L.append(f"  http: {http_line(p['http'])}")
  if (q := res.get("requests")): L.append(f"serve_prof{' ' + label if label else ''} http (whole file): {http_line(q)}")
  return "\n".join(L)

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("file")
  ap.add_argument("--label", default="")
  ap.add_argument("--idle", type=float, default=0.5, help="seconds without a step that end a busy period")
  ap.add_argument("--json", default=None)
  ap.add_argument("--requests", nargs="*", default=None, help="Validate requests.csv files: one period per concurrency and repeat")
  a = ap.parse_args()
  recs = []
  with open(a.file) as f:
    for line in f:
      try: recs.append(json.loads(line))
      except ValueError: pass
  res = summarize(recs, a.idle, windows(a.requests) if a.requests else None)
  print(report(res, a.label))
  if a.json: json.dump(res, open(a.json, "w"), indent=1)

if __name__ == "__main__": main()
