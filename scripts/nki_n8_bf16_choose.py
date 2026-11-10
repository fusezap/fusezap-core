"""N8 (I70): the automatic choices of scripts/nki_n8_bf16_trn1.sh from its bench results (stdout: env words for `env`, maybe
empty = the bf16 defaults; stderr: the table the choice came from, which the job appends to <out>/choice.log).

  python3 scripts/nki_n8_bf16_choose.py layer <out>    # fastest bf16 layer (v2se or v2se+w<cols>[p<k>][q]) bitwise == the bf16
                                                       # members (v1) in every record (logits, pool, tokens): its N8 switches
  python3 scripts/nki_n8_bf16_choose.py head <out>     # fastest bf16 head (b3 = MatmulBF16, b2:<W> = HeadBF16) whose logits are
                                                       # bitwise == b1-logits (MatmulBF16) and whose device argmax == np.argmax
  python3 scripts/nki_n8_bf16_choose.py same <out> <a> <b>   # serve_<a>.json vs serve_<b>.json: greedy rows and c1 tokens identical?
Rule: a variant's device ms is rows 1 + rows 4 at 512 slots; the fastest eligible variant wins only if it beats the
default (v2se / b3) by more than 1%.
"""
from __future__ import annotations
import json, os, re, sys

def log(*a) -> None: print(*a, file=sys.stderr)

def _load(out: str, f: str) -> list:
  try: return json.load(open(os.path.join(out, f)))["results"]
  except Exception as e: log(f"no {f} ({type(e).__name__})"); return []

def _pick(recs: list, exact, default: str, what: str, ok_variant) -> str:
  ms, bad = {}, set()
  for r in recs:
    v, g = r["variant"], r.get
    if not ok_variant(v): continue
    where = f"rows {r['rows']}" + (f" ctx {r['ctx']}" if "ctx" in r else "")
    if g("skipped") or g("device_ms") is None:
      bad.add(v); log(f"{what}_choice: {v} {where} skipped / untimed ({str(g('skipped'))[:160]})"); continue
    if not exact(r):
      bad.add(v); log(f"{what}_choice: {v} {where} NOT bitwise ({', '.join(f'{k} {r.get(k)}' for k in ('logits_bitwise_vs_v1', 'pool_bitwise_vs_v1', 'tokens_vs_v1', 'argmax_ok') if k in r)})")
    if r.get("ctx", 512) == 512: ms.setdefault(v, {})[r["rows"]] = g("device_ms")
  tot = {v: sum(d.values()) for v, d in ms.items() if len(d) == 2 and v not in bad}
  for v, t in sorted(tot.items(), key=lambda x: x[1]): log(f"{what}_choice: {v:<16} rows 1 + rows 4 at 512 slots = {t:.4f} ms")
  best = min(tot, key=tot.get) if tot else default
  if default in tot and best != default and tot[best] > 0.99 * tot[default]: best = default
  log(f"{what}_choice: chose {best}")
  return best

def layer(out: str) -> str:
  exact = lambda r: r.get("logits_bitwise_vs_v1") == 1.0 and r.get("pool_bitwise_vs_v1") == 1.0 and r.get("tokens_vs_v1") is True
  best = _pick(_load(out, "layer_bench.json"), exact, "v2se", "layer", lambda v: v == "v2se" or v.startswith("v2se+"))
  m = re.fullmatch(r"v2se\+w(\d+)(?:p(\d+))?(q?)", best)
  if not m: return ""
  return f"FZ_NEURON_BF16_WN={m.group(1)} FZ_NEURON_BF16_PF={m.group(2) or 0} FZ_NEURON_BF16_PFQO={1 if m.group(3) else 0}"

def head(out: str) -> str:
  exact = lambda r: r.get("logits_bitwise_vs_v1") == 1.0 and r.get("argmax_ok") is True
  best = _pick(_load(out, "head_bench.json"), exact, "b3", "head", lambda v: v == "b3" or v.startswith("b2"))
  if not best.startswith("b2"): return ""
  return f"FZ_NEURON_BF16_HEAD=1 FZ_NEURON_BF16_HEAD_W={best.split(':')[1] if ':' in best else 2048}"

def same(out: str, a: str, b: str) -> str:
  try: da, db = (json.load(open(os.path.join(out, f"serve_{v}.json"))) for v in (a, b))
  except Exception as e: return f"{a} vs {b}: missing ({type(e).__name__})"
  ga = {r["id"]: r.get("ids") for r in (da.get("greedy") or {}).get("rows", [])}
  gb = {r["id"]: r.get("ids") for r in (db.get("greedy") or {}).get("rows", [])}
  both = [k for k in ga if k in gb]
  eq = [k for k in both if ga[k] == gb[k]]
  c1 = None if da.get("c1", {}).get("ids") is None else da["c1"].get("ids") == db.get("c1", {}).get("ids")
  ok = len(eq) == len(both) and c1 is not False
  return (f"{a} vs {b}: greedy {len(eq)}/{len(both)} prompts token-identical, c1 tokens identical: {c1}"
          + ("  -> TOKEN-IDENTICAL" if ok else "  -> TOKENS DIFFER"))

if __name__ == "__main__":
  what, out = sys.argv[1], sys.argv[2]
  print({"layer": lambda: layer(out), "head": lambda: head(out), "same": lambda: same(out, sys.argv[3], sys.argv[4])}[what]())
