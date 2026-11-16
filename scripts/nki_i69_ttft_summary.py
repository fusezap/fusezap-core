"""I69 follow-up: the --i69-ttft job's table, TTFT breakdown and choice (scripts/nki_n8_prefill_inf2.sh --i69-ttft).

  python3 scripts/nki_i69_ttft_summary.py <out dir> [--json <file>]

Reads
  <out>/ttft_variants.tsv                       label <TAB> kind (bitwise | act) <TAB> env words, in the order served (base first)
  <out>/fz_tp2_bf16_7b_ttft_<label>_{perf,acc}/ Invarix (short 128 / 128 at c1 and c4; accuracy auto where run)
  <out>/trace_<label>.{api,engine}.jsonl        FZ_REQ_TRACE of that server (scripts/nki_i69_ttft_breakdown.py)
  <out>/tokens_<label>.json                     greedy texts of 12 reference prompts x 64 tokens (scripts/nki_i69_tokens.py)
  <out>/server_tp2_bf16_7b_ttft_<label>.log     the server's log (a FZ_NEURON_EMBED_DEV that turned itself off says so)
The choice: among the bitwise variants (base included; base2, the base configuration served again last, only measures
drift) that served without errors, whose greedy texts equal base's, whose
c1 TPOT p50 is within base + 0.5 ms, that did not fall back (EMBED_DEV) and whose accuracy run (where one ran) is PROMOTE,
the lowest short c1 TTFT p50. The act variant (FZ_NEURON_PFK_EXP=act: ScalarE exp in the prefill, not the members' bits)
is reported beside it with its own accuracy decision, never chosen here.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NXDI_C1_TTFT = 50.6         # AWS NxD Inference TP=2 offline (I73 run, same instance type and workload; no HTTP in it)
STATUS_C1_TTFT = 55.4       # the public status page's FuseZap row (1 Nov, I78)
RUN2 = {"ttft_c1_p50": 52.153, "tpot_c1_p50": 38.855}   # 14 Nov I69 run 2, served (the current defaults), 8 requests a point
TPOT_SLACK = 0.5

def _load(p: str) -> dict:
  try:
    with open(p) as f: return json.load(f)
  except (OSError, ValueError): return {}

def _f(x) -> float | None:
  try: return float(x)
  except (TypeError, ValueError): return None

def variants(out: str) -> list[tuple[str, str, str]]:
  try:
    with open(os.path.join(out, "ttft_variants.tsv")) as f: return [tuple((ln.rstrip("\n").split("\t") + ["", ""])[:3]) for ln in f if ln.strip()]
  except OSError: return []

def perf(out: str, lab: str) -> dict:
  res: dict = {}
  p = os.path.join(out, f"fz_tp2_bf16_7b_ttft_{lab}_perf", "performance.csv")
  if os.path.exists(p):
    with open(p) as f:
      for r in csv.DictReader(f):
        res[int(r["concurrency"])] = {"ttft_p50": _f(r.get("ttft_p50_ms")), "ttft_p99": _f(r.get("ttft_p99_ms")), "tpot_p50": _f(r.get("tpot_p50_ms")),
                                      "tok_s": _f(r.get("output_tok_s")), "errors": int(_f(r.get("errors")) or 0)}
  return res

def accuracy(out: str, lab: str) -> tuple[str | None, dict]:
  rep = _load(os.path.join(out, f"fz_tp2_bf16_7b_ttft_{lab}_acc", "report.json"))
  if not rep: return None, {}
  acc = rep.get("accuracy") or {}
  keep = {k: acc.get(k) for k in ("method", "top1_agreement", "greedy_exact_match_mean", "logprob_abs_diff_mean") if k in acc}
  return (rep.get("decision") or {}).get("label") or rep.get("verdict"), keep

def fell_back(out: str, lab: str) -> str:
  try:
    with open(os.path.join(out, f"server_tp2_bf16_7b_ttft_{lab}.log"), errors="replace") as f:
      for ln in f:
        if "FZ_NEURON_EMBED_DEV off" in ln or ("self-test" in ln and "failed" in ln): return ln.strip()[:240]
  except OSError: pass
  return ""

def tokens_same(out: str, base: str, lab: str) -> tuple[bool | None, str]:
  from nki_i69_tokens import compare
  a, b = (os.path.join(out, f"tokens_{x}.json") for x in (base, lab))
  if not (os.path.exists(a) and os.path.exists(b)): return None, "no token file"
  return compare(a, b)

def breakdown(out: str, lab: str) -> dict:
  from nki_i69_ttft_breakdown import load, summarize
  try: recs, miss = load(os.path.join(out, f"fz_tp2_bf16_7b_ttft_{lab}_perf"), os.path.join(out, f"trace_{lab}"), 1)
  except (OSError, ValueError, KeyError): return {}
  m = summarize(recs)
  if m: m["n"], m["unmatched"] = len(recs), miss
  return m

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("out")
  ap.add_argument("--json", default="")
  a = ap.parse_args()
  vs = variants(a.out)
  if not vs: print("no variants recorded"); return
  S: dict = {"variants": {}}
  base = vs[0][0]
  P = {lab: perf(a.out, lab) for lab, _, _ in vs}
  print("--- served (fusezap serve --tp 2, Qwen2.5-7B bf16; Invarix short 128 in / 128 out; ms; FZ_REQ_TRACE on in every server)")
  print(f"  {'variant':<9} {'kind':<7} {'c1 TTFT p50':>11} {'p99':>7} {'c1 TPOT':>8} {'c4 TTFT p50':>11} {'p99':>7} {'c4 TPOT':>8} {'c4 tok/s':>8} {'err':>4}  tokens vs {base}; accuracy")
  for lab, kind, env in vs:
    p1, p4 = P[lab].get(1) or {}, P[lab].get(4) or {}
    same, why = tokens_same(a.out, base, lab) if lab != base else (True, "the reference")
    dec, acc = accuracy(a.out, lab)
    fb = fell_back(a.out, lab)
    S["variants"][lab] = {"kind": kind, "env": env, "c1": p1, "c4": p4, "tokens_same": same, "tokens": why, "decision": dec, "accuracy": acc,
                          "fell_back": fb, "breakdown_c1": breakdown(a.out, lab)}
    g = lambda d, k: "-" if d.get(k) is None else f"{d[k]:.2f}" if isinstance(d[k], float) else str(d[k])
    print(f"  {lab:<9} {kind:<7} {g(p1, 'ttft_p50'):>11} {g(p1, 'ttft_p99'):>7} {g(p1, 'tpot_p50'):>8} {g(p4, 'ttft_p50'):>11} {g(p4, 'ttft_p99'):>7} "
          f"{g(p4, 'tpot_p50'):>8} {g(p4, 'tok_s'):>8} {(p1.get('errors', 0) + p4.get('errors', 0)) if P[lab] else '-':>4}  {why}; "
          f"{dec or 'no accuracy run'}{' ' + json.dumps(acc) if acc else ''}")
    print(f"            env: {env or '(none)'}" + (f"\n            FELL BACK: {fb}" if fb else ""))
  print("--- where c1 TTFT goes (medians, ms; scripts/nki_i69_ttft_breakdown.py: client send -> handler -> engine -> step -> handler -> client)")
  for lab, _, _ in vs:
    m = S["variants"][lab]["breakdown_c1"]
    if not m: print(f"  {lab:<9} no traced requests"); continue
    print(f"  {lab:<9} http_in {m['http_in']:.2f}  parse {m['parse']:.2f}  pipe_in {m['pipe_in']:.2f}  wait {m['wait']:.2f}  step {m['step']:.2f}  "
          f"pipe_out {m['pipe_out']:.2f}  http_out {m['http_out']:.2f}  | TTFT {m['ttft']:.2f} (n {m['n']})")
  b1 = (P[base].get(1) or {})
  checks, best = {}, None
  for lab, kind, env in vs:
    if kind != "bitwise" or (lab != base and lab.startswith("base")): continue   # base2: the same configuration again (drift)
    v, p1 = S["variants"][lab], P[lab].get(1) or {}
    c = {"served": bool(P[lab]) and p1.get("ttft_p50") is not None,
         "no request errors": bool(P[lab]) and all(x["errors"] == 0 for x in P[lab].values()),
         f"greedy texts == {base}'s": v["tokens_same"] is True,
         f"c1 TPOT p50 <= {base} + {TPOT_SLACK} ms": (p1.get("tpot_p50") or 1e9) <= (b1.get("tpot_p50") or 0) + TPOT_SLACK,
         "did not fall back": not v["fell_back"],
         "accuracy PROMOTE (where run)": v["decision"] in (None, "PROMOTE")}
    checks[lab] = c
    if all(c.values()) and (best is None or p1["ttft_p50"] < (P[best].get(1) or {})["ttft_p50"]): best = lab
  S["checks"] = checks
  for lab, _, _ in vs[1:]:
    if lab.startswith("base") and (b2 := (P[lab].get(1) or {}).get("ttft_p50")) is not None and b1.get("ttft_p50") is not None:
      S["drift_c1_ttft_ms"] = round(b2 - b1["ttft_p50"], 3)
      print(f"  drift: {lab} (the base configuration again, last) c1 TTFT p50 {b2:.2f} vs {base} {b1['ttft_p50']:.2f} ({b2 - b1['ttft_p50']:+.2f} ms)")
  for lab, c in checks.items(): print(f"  checks {lab}: " + "; ".join(f"{k}: {v}" for k, v in c.items()))
  S["chosen"] = best
  if best is None:
    print("I69 first token: no bitwise variant passed its checks (base included): no recommendation")
  else:
    env = next(e for l, _, e in vs if l == best)
    t = (P[best].get(1) or {})["ttft_p50"]
    print(f"I69 first token: CHOSEN {best} ({env or 'the defaults'}): served short c1 TTFT p50 {t:.2f} ms, TPOT {(P[best].get(1) or {}).get('tpot_p50')} ms; "
          f"base {b1.get('ttft_p50')} (run 2: {RUN2['ttft_c1_p50']}), status page {STATUS_C1_TTFT}, AWS NxDI {NXDI_C1_TTFT} offline "
          f"(gap {t - NXDI_C1_TTFT:+.2f} ms: {'AHEAD' if t < NXDI_C1_TTFT else 'behind'})")
  for lab, kind, env in vs:
    if kind == "act":
      p1, v = P[lab].get(1) or {}, S["variants"][lab]
      if p1.get("ttft_p50") is not None:
        print(f"  not bitwise, reported only: {lab} c1 TTFT p50 {p1['ttft_p50']:.2f} ms (gap to NxDI {p1['ttft_p50'] - NXDI_C1_TTFT:+.2f}), "
              f"TPOT {p1.get('tpot_p50')}, accuracy {v['decision'] or 'not run'} {json.dumps(v['accuracy']) if v['accuracy'] else ''}")
  if a.json:
    with open(a.json, "w") as f: json.dump(S, f, indent=1, default=str)

if __name__ == "__main__": main()
