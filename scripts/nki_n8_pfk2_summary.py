"""I77 follow-up: the --pfk2 job's table, bits and recommendation (scripts/nki_n8_prefill_inf2.sh --pfk2).

  python3 scripts/nki_n8_pfk2_summary.py <out dir> [--json <file>]

Reads
  <out>/fz_tp2_bf16_7b_pfk2_{a,b}_perf/performance.csv   served (Invarix perf): profiles short 128 / 128, short40 40 / 128,
                                                          p200 200 / 56 at c1 and c4 (a = main's defaults: 128-row prefill;
                                                          b = FZ_NEURON_PREFILL_SIZES=128,512, sizes chosen by cost)
  <out>/fz_tp2_bf16_7b_pfk2_{a,b}_long_perf/...           p512 512 / 64 at c1
  <out>/bf16_compare_{a,b}.json                           scripts/nki_bf16_compare.py's verdicts per served config
  <out>/serve_pfk2_b.json, serve_pfk2_v1.json             in process (scripts/nki_n6_serve.py): prefill_bench, ttft_by_len,
                                                          chunk_check, prefill_cost, greedy, c1 / c4 decode
and prints
  - served c1 / c4 TTFT p50 / p99, TPOT p50, tok/s, errors per config and profile, against the I75 confirm (main 9241c80:
    short c1 TTFT p99 92.7 ms) and AWS NxD Inference TP=2 (offline: c1 TTFT p50 50.6 ms, TPOT c1 40.6 ms, 81.7 tok/s at c4);
  - in process: each prefill forward's measured wall per pool prefix (the cost table the scheduler should use: a suggested
    FZ_NEURON_PREFILL_COST_MS from the 128-row forward at the 256-slot prefix and the 512-row one at 512), the cost table in
    effect, TTFT by prompt length, and per chunk-check length the forwards each schedule ran, its TTFT and logits bits:
    small (128-row chunks) == large (one 512-row forward) == the members (v1) expected (v2 exact is bitwise);
  - the recommendation: make 128,512 (cost-chosen) the default when, against (a): short c1 TTFT p99 <= a + 3 ms and p200 c1
    TTFT p50 <= a + 5 ms (no regression for a lone short prompt: the cost choice ran 128-row forwards), a real gain (short c4
    TTFT p99 <= 0.95 a, or p512 c1 TTFT p50 <= 0.9 a), every TPOT p50 <= a + 1.0 ms and tok/s >= 0.97 a, no request errors,
    no bits that differ where checked, and the bf16 accuracy bar passes for (b).
"""
from __future__ import annotations
import argparse, csv, json, os

I75_C1_P99 = 92.7                                  # I75 confirm, main 9241c80: served short c1 TTFT p99 (ms)
NXDI = {"c1_ttft_p50": 50.6, "tpot_c1": 40.6, "c4_tok_s": 81.7}   # AWS NxD Inference TP=2 offline, I73 run (same instance)

def _load(p: str) -> dict:
  try:
    with open(p) as f: return json.load(f)
  except (OSError, ValueError): return {}

def _f(x) -> float | None:
  try: return float(x)
  except (TypeError, ValueError): return None

def perf(out: str, lab: str) -> dict:
  """(profile, conc) -> {ttft_p50, ttft_p99, tpot_p50, tok_s, errors} from the perf and long-perf dirs of label lab."""
  res: dict = {}
  for sub in ("perf", "long_perf"):
    p = os.path.join(out, f"fz_tp2_bf16_7b_{lab}_{sub}", "performance.csv")
    if not os.path.exists(p): continue
    with open(p) as f:
      for r in csv.DictReader(f):
        res[(r["workload"], int(r["concurrency"]))] = {"ttft_p50": _f(r.get("ttft_p50_ms")), "ttft_p99": _f(r.get("ttft_p99_ms")),
                                                       "tpot_p50": _f(r.get("tpot_p50_ms")), "tok_s": _f(r.get("output_tok_s")),
                                                       "errors": int(_f(r.get("errors")) or 0)}
  return res

def suggest_cost(pb: dict) -> str | None:
  """FZ_NEURON_PREFILL_COST_MS from the measured forwards: the 128-row one at the 256-slot prefix (a typical lone prompt's
  second chunk; 128 when absent) and the 512-row one at 512 (its smallest reachable prefix)."""
  def w(k: str) -> float | None: return _f((pb.get(k) or {}).get("wall_ms_min")) if isinstance(pb.get(k), dict) else None
  a, b = w("128@256") or w("128@128"), w("512@512")
  return f"128:{a:.0f},512:{b:.0f}" if a and b else None

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("out")
  ap.add_argument("--json", default="")
  a = ap.parse_args()
  P = {lab: perf(a.out, f"pfk2_{lab}") for lab in ("a", "b")}
  res: dict = {"served": {}, "in_process": {}, "bits": {}, "checks": {}}
  print("--- served (fusezap serve --tp 2; Invarix perf; a = main's defaults, 128-row prefill; b = 128,512 chosen by cost; ms)")
  print(f"{'config':<6} {'profile':<8} {'conc':>4} {'TTFT p50':>9} {'TTFT p99':>9} {'TPOT p50':>9} {'tok/s':>7} {'errors':>6}")
  fmt = lambda x: f"{x:9.1f}" if isinstance(x, (int, float)) else f"{'-':>9}"
  for lab in ("a", "b"):
    for (prof, c), r in sorted(P[lab].items()):
      tok = f"{r['tok_s']:7.1f}" if r["tok_s"] is not None else f"{'-':>7}"
      print(f"{lab:<6} {prof:<8} {c:>4} {fmt(r['ttft_p50'])} {fmt(r['ttft_p99'])} {fmt(r['tpot_p50'])} {tok} {r['errors']:>6}")
      res["served"][f"{lab}/{prof}/c{c}"] = r
  for lab in ("a", "b"):
    s1, s4 = P[lab].get(("short", 1)) or {}, P[lab].get(("short", 4)) or {}
    if s1:
      print(f"  ({lab}) short c1 TTFT p99 {s1.get('ttft_p99')} vs I75 confirm {I75_C1_P99}; c1 TTFT p50 {s1.get('ttft_p50')} vs NxDI {NXDI['c1_ttft_p50']}; "
            f"TPOT c1 {s1.get('tpot_p50')} vs NxDI {NXDI['tpot_c1']}; c4 {s4.get('tok_s')} tok/s vs NxDI {NXDI['c4_tok_s']}")
  B, V1 = _load(os.path.join(a.out, "serve_pfk2_b.json")), _load(os.path.join(a.out, "serve_pfk2_v1.json"))
  pb = B.get("prefill_bench") or {}
  if pb and "error" not in pb:
    print("--- in process (b): each prefill forward at each pool prefix, wall to the token (min of the replays, ms)")
    for k in sorted(pb, key=lambda k: tuple(int(x) for x in k.split("@"))):
      x = pb[k]
      print(f"  {k:<10} {x.get('wall_ms_min')} ms ({x.get('tokens')} tokens, {round(x['wall_ms_min'] * 128 / x['rows'], 1) if x.get('wall_ms_min') else '-'} ms per 128 rows)")
    res["in_process"]["prefill_bench"] = {k: x.get("wall_ms_min") for k, x in pb.items()}
  elif pb: print(f"--- prefill bench: {pb.get('error')}")
  sug = suggest_cost(pb) if pb else None
  res["in_process"]["cost_in_effect"], res["in_process"]["cost_suggested"] = B.get("prefill_cost"), sug
  print(f"  cost table in effect {B.get('prefill_cost')}; measured: FZ_NEURON_PREFILL_COST_MS={sug}")
  for ln, t in sorted((B.get("ttft_by_len") or {}).items(), key=lambda kv: int(kv[0])):
    print(f"  TTFT in process, {ln}-token prompts: c1 p50 {t.get('c1_p50')} / p99 {t.get('c1_p99')}, c4 burst p50 {t.get('c4_p50')} / p99 {t.get('c4_p99')}")
    res["in_process"].setdefault("ttft", {})[ln] = {k: t.get(k) for k in ("c1_p50", "c1_p99", "c4_p50", "c4_p99")}
  g = B.get("greedy") or {}
  if g: print(f"  greedy {g.get('exact')}/{g.get('prompts')} ({g.get('match_tokens')}/{g.get('ref_tokens')} tokens); "
              f"TPOT c1 {(B.get('c1') or {}).get('tpot_ms_median')} c4 {(B.get('c4') or {}).get('tpot_ms_median')} ms")
  cc, cv = B.get("chunk_check") or {}, V1.get("chunk_check") or {}
  diff = False
  if cc and "error" not in cc:
    print("--- chunk check: each prompt alone, the forwards per schedule, TTFT (ms per rep) and the last prompt token's logits bits")
    for ln in sorted(cc, key=int):
      row, shas = {}, {}
      for nm, r in cc[ln].items():
        shas[f"b/{nm}"] = r.get("logits_sha")
        print(f"  {ln:>4} tokens (b) {nm:<5}: forwards {r.get('sizes')}, TTFT {r.get('ttft_ms')}, logits {r.get('logits_sha')} (stable {r.get('stable')})")
        row[nm] = {"sizes": r.get("sizes"), "ttft_ms": r.get("ttft_ms"), "sha": r.get("logits_sha"), "stable": r.get("stable")}
      v = (cv.get(ln) or {}).get("auto") if isinstance(cv, dict) else None
      if v:
        shas["v1"] = v.get("logits_sha")
        print(f"  {ln:>4} tokens v1      : forwards {v.get('sizes')}, TTFT {v.get('ttft_ms')}, logits {v.get('logits_sha')}")
      same = len({x for x in shas.values() if x}) == 1
      diff |= not same
      res["bits"][ln] = {"schedules": row, "v1_sha": shas.get("v1"), "all_equal": same, "v1_checked": "v1" in shas}
      print(f"  {ln:>4} tokens: bits equal across {', '.join(shas)}: {same}")
  elif cc: print(f"--- chunk check: {cc.get('error')}")
  # the recommendation
  A_, B_ = P["a"], P["b"]
  g_ = lambda d, prof, c, k: (d.get((prof, c)) or {}).get(k)
  ch = res["checks"]
  def cmp(name: str, x, y, ok) -> None:
    ch[name] = {"a": x, "b": y, "ok": None if x is None or y is None else bool(ok(x, y))}
  cmp("short c1 TTFT p99 <= a + 3", g_(A_, "short", 1, "ttft_p99"), g_(B_, "short", 1, "ttft_p99"), lambda x, y: y <= x + 3)
  cmp("p200 c1 TTFT p50 <= a + 5", g_(A_, "p200", 1, "ttft_p50"), g_(B_, "p200", 1, "ttft_p50"), lambda x, y: y <= x + 5)
  gain_c4 = (g_(A_, "short", 4, "ttft_p99"), g_(B_, "short", 4, "ttft_p99"))
  gain_long = (g_(A_, "p512", 1, "ttft_p50"), g_(B_, "p512", 1, "ttft_p50"))
  gains = [y <= f * x for (x, y), f in ((gain_c4, 0.95), (gain_long, 0.9)) if x is not None and y is not None]
  ch["a gain: short c4 TTFT p99 <= 0.95 a or p512 c1 TTFT p50 <= 0.9 a"] = {"short_c4_p99": gain_c4, "p512_c1_p50": gain_long, "ok": any(gains) if gains else None}
  tp = [(k, A_[k]["tpot_p50"], B_[k]["tpot_p50"]) for k in A_ if k in B_ and A_[k]["tpot_p50"] is not None and B_[k]["tpot_p50"] is not None]
  ch["TPOT p50 <= a + 1.0 everywhere"] = {"ok": all(y <= x + 1.0 for _, x, y in tp) if tp else None, "worst": max((y - x for _, x, y in tp), default=None)}
  ts = [(k, A_[k]["tok_s"], B_[k]["tok_s"]) for k in A_ if k in B_ and A_[k]["tok_s"] and B_[k]["tok_s"]]
  ch["tok/s >= 0.97 a everywhere"] = {"ok": all(y >= 0.97 * x for _, x, y in ts) if ts else None}
  errs = sum(r["errors"] for d in (A_, B_) for r in d.values())
  ch["no request errors"] = {"ok": (errs == 0) if A_ and B_ else None, "errors": errs}
  ch["bits equal where checked"] = {"ok": (not diff) if res["bits"] else None}
  acc = (_load(os.path.join(a.out, "bf16_compare_b.json")).get("verdict") or {})
  ch["bf16 accuracy bar (b)"] = {"ok": all(acc.values()) if acc else None, "verdict": acc}
  print("--- recommendation checks: " + "; ".join(f"{k}: {v.get('ok')}" for k, v in ch.items()))
  ok = all(v.get("ok") for v in ch.values())
  missing = [k for k, v in ch.items() if v.get("ok") is None]
  res["recommend_128_512_default"] = ok
  print(f"I77 follow-up: FZ_NEURON_PREFILL_SIZES=128,512 (cost-chosen) as the default: {'RECOMMEND' if ok else 'DO NOT RECOMMEND'}"
        + (f" (not measured: {', '.join(missing)})" if missing else "")
        + (f"; set FZ_NEURON_PREFILL_COST_MS={sug} (measured) if it differs from the table in effect" if sug else ""))
  if a.json:
    with open(a.json, "w") as f: json.dump(res, f, indent=1, default=str)

if __name__ == "__main__": main()
