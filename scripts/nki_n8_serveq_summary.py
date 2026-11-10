"""I76: the served TTFT job's table, breakdown and bar (scripts/nki_n8_prefill_inf2.sh --serveq).

  python3 scripts/nki_n8_serveq_summary.py <out dir> [--old serveq_old] [--new serveq] [--json <file>]

Reads <out>/fz_tp2_bf16_7b_<label>_perf/{performance,requests}.csv (Invarix perf: profiles short = 128 / 128 tokens and
short40 = 40 / 128, c1 and c4) and the FZ_REQ_TRACE files <out>/trace_<label>.{api,engine}.jsonl, and prints
  - per label / profile / concurrency: TTFT p50 / p99, TPOT p50, tok/s (and the in-process TTFT from serve_serveq.json);
  - where a served request's TTFT goes, medians per burst position (the order its first token came in the c4 wave):
    client = Invarix's send to the handler (connection, HTTP parse), parse = handler to the engine pipe, pipe = to the
    engine process's add (it drains the pipe between steps: includes the rest of a step in flight), queue = add to the step that first scheduled it (earlier prompts' prefills), step = that step's
    start to the first output in the handler (the prefill forward + the output pipe), tail = to the client's first chunk;
  - the bar: PROMOTE when, against the old scheduler (FZ_ADMIT_WAIT_MS=0), short c4 TTFT p99 <= old x 1.03 + 5 ms, every
    TPOT p50 <= old + 1.0 ms and tok/s >= old x 0.97 (no regression where the gate cannot apply), short40 c4 TTFT p50 at
    most 0.8 x old or at the batched level (<= 120 ms; the gate's prediction ~98 ms), and every request ok; the accuracy bar is
    scripts/nki_bf16_compare.py's (printed after this)."""
from __future__ import annotations
import argparse, csv, json, os, sys
import numpy as np

def _rows(p: str) -> list[dict]:
  if not os.path.exists(p): return []
  with open(p) as f: return list(csv.DictReader(f))

def _jsonl(p: str) -> list[dict]:
  if not os.path.exists(p): return []
  out = []
  with open(p) as f:
    for ln in f:
      try: out.append(json.loads(ln))
      except ValueError: pass
  return out

def _f(x) -> float | None:
  try: return float(x)
  except (TypeError, ValueError): return None

def perf(out: str, lab: str) -> dict:
  """(profile, conc) -> performance.csv row of fz_tp2_bf16_7b_<lab>_perf."""
  d = os.path.join(out, f"fz_tp2_bf16_7b_{lab}_perf")
  return {(r["workload"], int(r["concurrency"])): r for r in _rows(os.path.join(d, "performance.csv"))}

def breakdown(out: str, lab: str, profile: str, conc: int) -> list[dict]:
  """Per burst position: median ms of each segment (see the module doc), from requests.csv and the request traces."""
  reqs = [r for r in _rows(os.path.join(out, f"fz_tp2_bf16_7b_{lab}_perf", "requests.csv"))
          if r.get("workload") == profile and r.get("concurrency") == str(conc) and r.get("status") == "ok" and _f(r.get("send_unix"))]
  api, eng = _jsonl(os.path.join(out, f"trace_{lab}.api.jsonl")), {e["rid"]: e for e in _jsonl(os.path.join(out, f"trace_{lab}.engine.jsonl"))}
  if not reqs or not api: return []
  used, rows = set(), []
  for r in sorted(reqs, key=lambda r: float(r["send_unix"])):
    s = float(r["send_unix"])
    cand = [(a["recv"] - s, i) for i, a in enumerate(api) if i not in used and 0 <= a["recv"] - s < 1.0]   # the first handler entry after the send
    if not cand: continue
    _, i = min(cand); used.add(i)
    a, e = api[i], eng.get(api[i].get("rid"), {})
    ttft = float(r["ttft_ms"])
    seg = {"ttft": ttft, "client": 1e3 * (a["recv"] - s), "parse": 1e3 * (a["submit"] - a["recv"])}
    if "add" in e: seg["pipe"] = 1e3 * (e["add"] - a["submit"])
    if "add" in e and "sched" in e: seg["queue"], seg["step"] = 1e3 * (e["sched"] - e["add"]), 1e3 * (a["first"] - e["sched"])
    seg["tail"] = ttft - 1e3 * (a["first"] - s)
    seg["first_unix"] = s + ttft / 1e3
    rows.append(seg)
  if not rows: return []
  # burst position: requests sent within 50 ms of each other form a wave; rank by first-token time inside it
  rows.sort(key=lambda x: x["first_unix"] - x["ttft"] / 1e3)
  waves, cur = [], [rows[0]]
  for x in rows[1:]:
    if x["first_unix"] - x["ttft"] / 1e3 - (cur[0]["first_unix"] - cur[0]["ttft"] / 1e3) < 0.05: cur.append(x)
    else: waves.append(cur); cur = [x]
  waves.append(cur)
  pos: dict[int, list[dict]] = {}
  for w in waves:
    for k, x in enumerate(sorted(w, key=lambda x: x["first_unix"])): pos.setdefault(k, []).append(x)
  keys = ("ttft", "client", "parse", "pipe", "queue", "step", "tail")
  return [{"pos": k + 1, "n": len(xs), **{q: round(float(np.median([x[q] for x in xs if q in x])), 2) if any(q in x for x in xs) else None for q in keys}}
          for k, xs in sorted(pos.items())]

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("out")
  ap.add_argument("--old", default="serveq_old")
  ap.add_argument("--new", default="serveq")
  ap.add_argument("--json", default="")
  a = ap.parse_args()
  P = {lab: perf(a.out, lab) for lab in (a.old, a.new)}
  res: dict = {"perf": {}, "breakdown": {}, "bar": {}}
  print(f"{'label':<14} {'profile':<8} {'conc':>4} {'TTFT p50':>9} {'TTFT p99':>9} {'TPOT p50':>9} {'tok/s':>7} {'errors':>6}")
  for lab, rows in P.items():
    for (prof, c), r in sorted(rows.items()):
      print(f"{lab:<14} {prof:<8} {c:>4} {r['ttft_p50_ms']:>9} {r['ttft_p99_ms']:>9} {r['tpot_p50_ms']:>9} {r['output_tok_s']:>7} {r['errors']:>6}")
      res["perf"][f"{lab}/{prof}/c{c}"] = {k: _f(r[k]) for k in ("ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "output_tok_s", "errors")}
  sj = os.path.join(a.out, "serve_serveq.json")
  if os.path.exists(sj):
    for n, t in (json.load(open(sj)).get("ttft_by_len") or {}).items():
      print(f"in process, prompts {t.get('prompt_len')}: c1 TTFT p50 {t.get('c1_p50')} p99 {t.get('c1_p99')}; c4 burst {t.get('c4_ms')} (p50 {t.get('c4_p50')}, p99 {t.get('c4_p99')})")
      res.setdefault("in_process", {})[n] = {k: t.get(k) for k in ("c1_p50", "c1_p99", "c4_p50", "c4_p99", "c4_ms")}
  print("--- served TTFT by burst position (medians, ms): client -> handler, parse, pipe -> engine add, queue (earlier prefills), step -> handler first output, tail -> client")
  for lab in (a.old, a.new):
    for prof in ("short", "short40"):
      for c in (1, 4):
        b = breakdown(a.out, lab, prof, c)
        if not b: continue
        res["breakdown"][f"{lab}/{prof}/c{c}"] = b
        for x in b:
          print(f"{lab:<14} {prof:<8} c{c} #{x['pos']} (n={x['n']}): TTFT {x['ttft']}  = client {x['client']} + parse {x['parse']} + pipe {x['pipe']} + queue {x['queue']} + step {x['step']} + tail {x['tail']}")
  o, n = P[a.old], P[a.new]
  checks = []
  def chk(name: str, ok: bool | None, detail: str) -> None:
    checks.append((name, ok, detail)); print(f"bar {name}: {'ok' if ok else 'MISSING' if ok is None else 'FAIL'} ({detail})")
  g = lambda d, prof, c, k: _f(d[(prof, c)][k]) if (prof, c) in d else None
  op99, np99 = g(o, "short", 4, "ttft_p99_ms"), g(n, "short", 4, "ttft_p99_ms")
  chk("short c4 TTFT p99", None if op99 is None or np99 is None else np99 <= op99 * 1.03 + 5, f"{op99} -> {np99} ms")
  for prof in ("short", "short40"):
    for c in (1, 4):
      ot, nt = g(o, prof, c, "tpot_p50_ms"), g(n, prof, c, "tpot_p50_ms")
      chk(f"{prof} c{c} TPOT p50", None if ot is None or nt is None else nt <= ot + 1.0, f"{ot} -> {nt} ms")
      ok_, nk = g(o, prof, c, "output_tok_s"), g(n, prof, c, "output_tok_s")
      chk(f"{prof} c{c} tok/s", None if ok_ is None or nk is None else nk >= 0.97 * ok_, f"{ok_} -> {nk}")
  o50, n50 = g(o, "short40", 4, "ttft_p50_ms"), g(n, "short40", 4, "ttft_p50_ms")
  # without the gate a wave's short prompts share a forward only when they happen to reach the engine before its step starts
  # (Mac sim runs: old p50 97-183 ms), so: 20% better, or at the batched level (one forward + the window: <= 120 ms)
  chk("short40 c4 TTFT p50", None if o50 is None or n50 is None else n50 <= max(0.8 * o50, 120.0), f"{o50} -> {n50} ms (gated prediction ~98)")
  errs = sum(int(_f(r["errors"]) or 0) for d in (o, n) for r in d.values())
  chk("errors", bool(o) and bool(n) and errs == 0, f"{errs}")
  res["bar"] = {nm: {"ok": ok, "detail": d} for nm, ok, d in checks}
  verdict = "PROMOTE" if all(ok for _, ok, _ in checks) else "DO NOT PROMOTE"
  res["verdict"] = verdict
  print(f"serveq perf bar: {verdict} (the accuracy bar follows)")
  if a.json:
    with open(a.json, "w") as f: json.dump(res, f, indent=1)

if __name__ == "__main__": main()
