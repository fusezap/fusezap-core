"""I72 follow-up: the TP=2 job's choice among its decode paths (scripts/nki_n8_ar_inf2.sh: <out>/ar_variants.tsv lines
"name<TAB>env words", the first the baseline; each variant's scripts/nki_n6_serve.py --quick result in serve_<name>.json).

  python3 scripts/nki_n8_ar_choose.py <out>   # stdout: the chosen variant's env words; stderr: the table it came from

Rule: a variant counts only if its c1 greedy tokens equal the baseline's (halves + shm, the served path: the fused-collective
layer is the same instructions and sums, so the same bits on NKISHIM); the lowest c1 TPOT wins (c4 TPOT breaks a tie
within 2%). Nothing ran, or the baseline did not run: the baseline's env words (the served path as of run 2).
"""
from __future__ import annotations
import json, os, sys

def load(out: str, name: str) -> dict:
  try:
    with open(os.path.join(out, f"serve_{name}.json")) as f: return json.load(f)
  except (OSError, ValueError): return {}

def main() -> None:
  out = sys.argv[1]
  rows = []
  try:
    with open(os.path.join(out, "ar_variants.tsv")) as f:
      for ln in f:
        if "\t" in ln: rows.append(tuple(ln.rstrip("\n").split("\t", 1)))
  except OSError: pass
  if not rows:
    print("no variants ran: the baseline (halves + shm)", file=sys.stderr); print("FZ_NEURON_TP_LAYER=halves"); return
  base_name, base_env = rows[0]
  base = load(out, base_name)
  ids0 = (base.get("c1") or {}).get("ids")
  best = None
  for name, env in rows:
    r = load(out, name)
    c1, c4 = (r.get("c1") or {}).get("tpot_ms_median"), (r.get("c4") or {}).get("tpot_ms_median")
    same = (r.get("c1") or {}).get("ids") == ids0 if r and ids0 is not None else None
    tps = round(4 * 1000 / c4, 1) if c4 else None
    print(f"{name:<28} c1 TPOT {c1} ms  c4 TPOT {c4} ms ({tps} tok/s)  c1 tokens == baseline: {same}  [{env}]", file=sys.stderr)
    if c1 is None or not same: continue
    key = (c1, c4 if c4 is not None else float("inf"))
    if best is None or key[0] < best[0][0] * 0.98 or (key[0] <= best[0][0] * 1.02 and key[1] < best[0][1]): best = (key, name, env)
  if best is None:
    print(f"chosen: {base_name} (no variant with the baseline's tokens)", file=sys.stderr); print(base_env); return
  print(f"chosen: {best[1]} (c1 {best[0][0]} ms)", file=sys.stderr)
  print(best[2])

if __name__ == "__main__": main()
