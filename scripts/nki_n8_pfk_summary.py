"""I77: the prefill kernels job's in-process table, bits and bar (scripts/nki_n8_prefill_inf2.sh --pfk).

  python3 scripts/nki_n8_pfk_summary.py <out dir> [--json <file>]

Reads <out>/serve_pfk_{old,new,exact}.json (scripts/nki_n6_serve.py: prefill_bench, ttft_by_len, c1 / c4 decode, greedy,
top-1, engine prewarm) and <out>/pfk_bench/pfk_bench.json (scripts/nki_pfk_bench.py: one NeuronCore, per layer NEFF), and
prints
  - per prefill forward and pool prefix: the wall to the token (min over the replays; device + ~4 ms host) old / new /
    exact, new's per-token cost, and whether the first logits row's bits equal the old path's (exact: must; new: ScalarE
    exp, need not);
  - TTFT in process by prompt length (c1 p50 / p99, c4 burst p50 / p99), the first tokens old vs new;
  - decode TPOT c1 / c4, greedy and top-1, HBM and the cold prewarm (compile) seconds;
  - the single-core layer bench and the neuron-profile summaries it wrote;
  - the bar: PROMOTE (perf) when new's 128-row forward at the 128-slot prefix is <= 59 ms (the 55 ms target + host), every
    128-row forward is faster than old's at its prefix, the 512-row forward costs less per token than the 128-row one at
    the 512-slot prefix, decode TPOT c1 / c4 within old + 0.5 ms, greedy >= 0.70 and >= old's minus one prompt (old's: I75
    confirm 23/24 when not run), the exact path's bits equal old's wherever both ran, and nothing failed.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

TARGET_MS = 59.0   # 55 ms device (the I77 target) + the ~4 ms of host and round trips a wall-to-token time carries (I75: 85.7 wall, 81.6 device)

def _load(p: str) -> dict:
  try:
    with open(p) as f: return json.load(f)
  except (OSError, ValueError): return {}

def _g(d: dict, *ks, default=None):
  for k in ks:
    if not isinstance(d, dict) or k not in d: return default
    d = d[k]
  return d

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("out")
  ap.add_argument("--json", default="")
  a = ap.parse_args()
  runs = {k: _load(os.path.join(a.out, f"serve_pfk_{k}.json")) for k in ("old", "new", "exact")}
  res: dict = {"forwards": {}, "ttft": {}, "checks": {}}
  print("--- prefill forwards in process (TP=2; wall to the token, min of 4 replays, ms; logits = the first row's bits vs old)")
  print(f"{'forward @ prefix':<18} {'old':>8} {'new':>8} {'exact':>8} {'new/old':>8} {'new ms/128 tok':>15} {'new bits':>9} {'exact bits':>11}")
  keys = sorted({k for r in runs.values() for k in (r.get("prefill_bench") or {}) if "@" in k}, key=lambda k: tuple(int(x) for x in k.split("@")))
  for k in keys:
    o, n, e = (_g(runs[x], "prefill_bench", k) or {} for x in ("old", "new", "exact"))
    rows = int(k.split("@")[0])
    row = {"old": o.get("wall_ms_min"), "new": n.get("wall_ms_min"), "exact": e.get("wall_ms_min"),
           "new_bits_eq_old": (n.get("logits_sha") == o.get("logits_sha")) if n and o else None,
           "exact_bits_eq_old": (e.get("logits_sha") == o.get("logits_sha")) if e and o else None,
           "new_top_eq_old": (n.get("top") == o.get("top")) if n and o else None}
    row["new_ms_per_128"] = round(row["new"] * 128 / rows, 2) if row["new"] else None
    res["forwards"][k] = row
    f = lambda x: f"{x:8.2f}" if isinstance(x, (int, float)) else f"{'-':>8}"
    ratio = f"{row['new'] / row['old']:8.3f}" if row["new"] and row["old"] else f"{'-':>8}"
    print(f"{k:<18} {f(row['old'])} {f(row['new'])} {f(row['exact'])} {ratio} {f(row['new_ms_per_128']):>15} {str(row['new_bits_eq_old']):>9} {str(row['exact_bits_eq_old']):>11}")
  print("--- TTFT in process by prompt length (ms: c1 p50 / p99, c4 burst p50 / p99); first tokens of the c1 prompts vs old")
  for ln in sorted({k for r in runs.values() for k in (r.get("ttft_by_len") or {})}, key=lambda x: int(x)):
    line = []
    for x in ("old", "new", "exact"):
      t = _g(runs[x], "ttft_by_len", ln)
      if t: line.append(f"{x} {t.get('c1_p50')} / {t.get('c1_p99')}, c4 {t.get('c4_p50')} / {t.get('c4_p99')}")
      res["ttft"].setdefault(ln, {})[x] = {k: t.get(k) for k in ("c1_p50", "c1_p99", "c4_p50", "c4_p99")} if t else None
    o, n = _g(runs["old"], "ttft_by_len", ln, "c1_ids"), _g(runs["new"], "ttft_by_len", ln, "c1_ids")
    same = (o == n) if o and n else None
    res["ttft"].setdefault(ln, {})["first_tokens_new_eq_old"] = same
    print(f"  {ln:>4} tokens: " + "; ".join(line) + f"; first tokens new == old: {same}")
  print("--- decode, accuracy, memory, compile")
  for x in ("old", "new", "exact"):
    r = runs[x]
    if not r: print(f"  {x}: not run"); continue
    g, t = r.get("greedy") or {}, r.get("top1") or {}
    hb = _g(r, "hbm", "after prewarm") or {}
    pw = _g(r, "engine_prewarm", "seconds") or {}
    nf = [y for v in (_g(r, "engine_prewarm", "neffs") or {}).values() for y in v]
    print(f"  {x:<6} TPOT c1 {_g(r, 'c1', 'tpot_ms_median')} ms, c4 {_g(r, 'c4', 'tpot_ms_median')} ms; greedy {g.get('exact')}/{g.get('prompts')} "
          f"({g.get('match_tokens')}/{g.get('ref_tokens')} tokens); top-1 {t.get('agreement')}; HBM after prewarm {round(hb.get('held', 0) / 2**30, 2) if hb else None} GiB "
          f"(forwards {json.dumps({k: round(v / 2**30, 2) for k, v in (hb.get('forwards') or {}).items()})}); prewarm {pw} "
          f"({sum(1 for y in nf if not y.get('ok'))} NEFFs failed); engine {r.get('engine')}")
  b = _load(os.path.join(a.out, "pfk_bench", "pfk_bench.json"))
  if b:
    print("--- one NeuronCore, a 3-layer model of the shard's shapes (scripts/nki_pfk_bench.py): the layer NEFF, ms")
    for r in b.get("records", []):
      print(f"  {r['config']:<16} device {r.get('layer_device_ms')}  execute {r.get('layer_exec_ms')}  members {len(r.get('members') or [])}  "
            f"bitwise vs v1 {r.get('logits_bitwise_vs_v1')} (max diff {r.get('logits_maxdiff_vs_v1')})" + (f"  ERROR {r['error'][:200]}" if r.get("error") else ""))
    for p in sorted(glob.glob(os.path.join(a.out, "pfk_bench", "*.summary.txt"))):
      print(f"  neuron-profile {os.path.basename(p)}: {sum(1 for _ in open(p))} lines (in the results)")
  # the bar
  F, O, N = res["forwards"], runs["old"], runs["new"]
  c = res["checks"]
  n128 = _g(F, "128@128", "new")
  c["128_rows_at_128_within_target"] = (n128 is not None and n128 <= TARGET_MS)
  c["every_128_row_forward_faster"] = bool([k for k in F if k.startswith("128@")]) and all(F[k]["new"] and F[k]["old"] and F[k]["new"] < F[k]["old"]
                                                                                           for k in F if k.startswith("128@") and F[k]["old"])
  p512, p128 = _g(F, "512@512", "new_ms_per_128"), _g(F, "128@512", "new")
  c["512_rows_cheaper_per_token"] = (p512 is not None and p128 is not None and p512 < p128)
  def tpot_ok(k: str) -> bool | None:
    o, n = _g(O, k, "tpot_ms_median"), _g(N, k, "tpot_ms_median")
    return None if o is None or n is None else n <= o + 0.5
  c["decode_c1_unchanged"], c["decode_c4_unchanged"] = tpot_ok("c1"), tpot_ok("c4")
  g = N.get("greedy") or {}
  og = O.get("greedy") or {"exact": 23, "prompts": 24}
  c["greedy_bar"] = (g.get("prompts") and g["exact"] / g["prompts"] >= 0.70 and g["exact"] >= og["exact"] * g["prompts"] / max(og["prompts"], 1) - 1) if g else None
  ex = [F[k]["exact_bits_eq_old"] for k in F if F[k]["exact_bits_eq_old"] is not None]
  c["exact_bits_equal_old"] = all(ex) if ex else None
  c["no_neff_failed"] = not any(not y.get("ok") for r in runs.values() for v in (_g(r, "engine_prewarm", "neffs") or {}).values() for y in v)
  must = ("128_rows_at_128_within_target", "every_128_row_forward_faster", "decode_c1_unchanged", "decode_c4_unchanged", "greedy_bar", "no_neff_failed")
  ok = all(c.get(k) for k in must) and c.get("exact_bits_equal_old") is not False
  res["promote_perf"] = ok
  print("--- the bar: " + ", ".join(f"{k} {v}" for k, v in c.items()))
  print(f"I77 prefill kernels perf bar: {'PROMOTE' if ok else 'DO NOT PROMOTE'}" + ("" if c.get("512_rows_cheaper_per_token") else " (512-row chunks: not cheaper per token, or not run: keep FZ_NEURON_PREFILL_SIZES off)"))
  if a.json:
    with open(a.json, "w") as f: json.dump(res, f, indent=1)

if __name__ == "__main__": main()
