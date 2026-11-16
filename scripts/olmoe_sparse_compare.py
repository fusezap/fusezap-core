#!/usr/bin/env python3
"""I105 A2: dense vs sparse MoE decode, from two scripts/nki_n6_serve.py results (serve_<dense>.json, serve_<sparse>.json):

  greedy      per prompt the engine's greedy tokens (dense vs sparse: identical prompts, first difference) and each vs the
              float32 reference (the run's own greedy match)
  decode top1 teacher-forced on the decode path (--decode-top1): each vs the reference's top-1, and sparse vs dense argmax
              at every position
  speed       c1 TPOT median, c4 TPOT median and tok/s, the ratio

  python3 scripts/olmoe_sparse_compare.py --out results/nki_olmoe_sparse --dense dense --sparse sparse --json ...
Prints a table; exit 0 always (the numbers are the record; the job decides nothing on them).
"""
from __future__ import annotations
import argparse, json, os

def load(out: str, v: str) -> dict:
  p = os.path.join(out, f"serve_{v}.json")
  return json.load(open(p)) if os.path.exists(p) else {}

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--dense", default="dense")
  ap.add_argument("--sparse", default="sparse")
  ap.add_argument("--json", default="")
  a = ap.parse_args()
  d, s = load(a.out, a.dense), load(a.out, a.sparse)
  res: dict = {"dense": a.dense, "sparse": a.sparse, "have": {"dense": bool(d), "sparse": bool(s)}}
  gd = {r["id"]: r for r in (d.get("greedy") or {}).get("rows", [])}
  gs = {r["id"]: r for r in (s.get("greedy") or {}).get("rows", [])}
  rows = []
  for k in gd.keys() & gs.keys():
    x, y = gd[k]["ids"], gs[k]["ids"]
    n = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    rows.append({"id": k, "same": x == y, "first_diff": None if x == y else n, "len": len(x), "dense_vs_ref": gd[k]["match"], "sparse_vs_ref": gs[k]["match"], "of": gd[k]["of"]})
  res["greedy"] = {"prompts": len(rows), "identical": sum(r["same"] for r in rows),
                   "prefix_tokens": sum((r["first_diff"] if r["first_diff"] is not None else r["len"]) for r in rows),
                   "tokens": sum(r["len"] for r in rows), "rows": sorted(rows, key=lambda r: str(r["id"])),
                   "dense_vs_ref": f"{(d.get('greedy') or {}).get('match_tokens')}/{(d.get('greedy') or {}).get('ref_tokens')}",
                   "sparse_vs_ref": f"{(s.get('greedy') or {}).get('match_tokens')}/{(s.get('greedy') or {}).get('ref_tokens')}"}
  td, ts = d.get("decode_top1") or {}, s.get("decode_top1") or {}
  ad = {r["id"]: r["argmax"] for r in td.get("rows", [])}
  ag = {r["id"]: r["argmax"] for r in ts.get("rows", [])}
  same = tot = 0
  for k in ad.keys() & ag.keys():
    for p, q in zip(ad[k], ag[k]): same += int(p == q); tot += 1
  res["decode_top1"] = {"dense_vs_ref": td.get("agreement"), "sparse_vs_ref": ts.get("agreement"), "sparse_vs_dense": round(same / tot, 5) if tot else None,
                        "positions": tot}
  sp = {}
  for nm, r in (("dense", d), ("sparse", s)):
    sp[nm] = {"c1_tpot_ms": (r.get("c1") or {}).get("tpot_ms_median"), "c4_tpot_ms": (r.get("c4") or {}).get("tpot_ms_median"),
              "c4_tok_s": (r.get("c4") or {}).get("tok_s_decode"), "prewarm_s": r.get("prewarm_s"), "decode_sizes": (r.get("engine") or {}).get("decode_sizes")}
  for k in ("c1_tpot_ms", "c4_tpot_ms"):
    x, y = sp["dense"][k], sp["sparse"][k]
    sp[f"{k}_speedup"] = round(x / y, 2) if x and y else None
  res["speed"] = sp
  g, t = res["greedy"], res["decode_top1"]
  print(f"greedy (engine, {g['prompts']} prompts): sparse == dense on {g['identical']} prompts, common prefix {g['prefix_tokens']}/{g['tokens']} tokens; "
        f"vs the float32 reference: dense {g['dense_vs_ref']}, sparse {g['sparse_vs_ref']}")
  for r in g["rows"]:
    if not r["same"]: print(f"  {r['id']}: first difference at {r['first_diff']} (dense {r['dense_vs_ref']}/{r['of']} vs ref, sparse {r['sparse_vs_ref']}/{r['of']})")
  print(f"decode top-1 (teacher-forced, decode path, {t['positions']} positions): dense vs ref {t['dense_vs_ref']}, sparse vs ref {t['sparse_vs_ref']}, sparse vs dense {t['sparse_vs_dense']}")
  print(f"TPOT median ms: c1 dense {sp['dense']['c1_tpot_ms']} -> sparse {sp['sparse']['c1_tpot_ms']} (x{sp['c1_tpot_ms_speedup']}); "
        f"c4 dense {sp['dense']['c4_tpot_ms']} -> sparse {sp['sparse']['c4_tpot_ms']} (x{sp['c4_tpot_ms_speedup']}); c4 tok/s {sp['dense']['c4_tok_s']} -> {sp['sparse']['c4_tok_s']}")
  if a.json:
    with open(a.json, "w") as f: json.dump(res, f, indent=1)

if __name__ == "__main__": main()
