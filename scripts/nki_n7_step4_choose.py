"""NKI-N7 step 4: the automatic choices of scripts/nki_n7_step4.sh, from its bench results (stdout: env words for `env`;
stderr: the table the choice came from, which the job appends to <out>/choice.log).

  python3 scripts/nki_n7_step4_choose.py layer <out>       # fastest LayerSpec variant bitwise == v1 everywhere (else v1; no
                                                           # results at all: step 3's v2s)
  python3 scripts/nki_n7_step4_choose.py head <out>        # fastest head bitwise == v1's logits with a correct argmax (else v3;
                                                           # no results: step 3's v2:1024:1)
  python3 scripts/nki_n7_step4_choose.py candidate <out>   # the fastest layer + head regardless of bits, if it beats the chosen
                                                           # ones (empty when nothing non-bitwise is faster): for the 24-prompt check
  python3 scripts/nki_n7_step4_choose.py fuse <out> <base> <label>...   # the in-process run (serve_<label>.json) with the lowest
                                                           # c1 TPOT whose c1 tokens equal <base>'s (prints the label)
Rule: a variant's device ms is rows 1 + rows 4 at 512 slots; it is eligible when every bench record of it (all rows, all
pool sizes) is bitwise equal to v1 (layers: logits and KV pool, plus v1's tokens; heads: logits, and the device argmax ==
np.argmax). The fastest eligible one wins if it is faster than the reference path (v1 layers; v3 head), else the reference.
"""
from __future__ import annotations
import json, os, sys

LAYER_ENV = {"v1": {"FZ_NEURON_LAYER": "v1"}}
def _layer_env(v: str) -> str:
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  try:
    from fusezap.codegen.nki_kernels import LAYER_VARIANTS
    e = LAYER_VARIANTS.get(v, LAYER_ENV["v1"])
  except Exception: e = LAYER_ENV["v1"]   # the job's python3 may lack numpy: v1 then
  return " ".join(f"{k}={x}" for k, x in e.items())

def _head_env(v: str) -> str:
  base = v.removesuffix("-logits")
  if base.startswith(("v2", "v4")) and ":" in base:
    m, w, d = base.split(":")[:3]
    return f"FZ_NEURON_HEAD={m} FZ_NEURON_HEAD_W={w} FZ_NEURON_HEAD_DMAS={d}"
  return f"FZ_NEURON_HEAD={base}"

def log(*a) -> None: print(*a, file=sys.stderr)

def _rank(recs: list, exact, ref: str, what: str) -> tuple[dict, set]:
  """per variant: rows 1 + rows 4 device ms at 512 slots (layers) / any (heads); the set of ineligible variants."""
  ms, bad = {}, set()
  for r in recs:
    v, g = r["variant"], r.get
    if g("skipped") or g("device_ms") is None:
      bad.add(v); log(f"{what}_choice: {v} rows {r['rows']}{' ctx ' + str(r['ctx']) if 'ctx' in r else ''} skipped / untimed ({str(g('skipped'))[:120]})")
      continue
    if v != ref and not exact(r):
      bad.add(v); log(f"{what}_choice: {v} rows {r['rows']}{' ctx ' + str(r['ctx']) if 'ctx' in r else ''} NOT bitwise vs v1 ({_bits(r)})")
    if r.get("ctx", 512) == 512: ms.setdefault(v, {})[r["rows"]] = g("device_ms")
  tot = {v: sum(d.values()) for v, d in ms.items() if len(d) == 2}
  return tot, bad

def _bits(r: dict) -> str:
  return ", ".join(f"{k} {r.get(k)}" for k in ("logits_bitwise_vs_v1", "pool_bitwise_vs_v1", "tokens_vs_v1", "argmax_ok") if k in r)

def _load(out: str, f: str) -> list:
  try: return json.load(open(os.path.join(out, f)))["results"]
  except Exception as e: log(f"no {f} ({type(e).__name__})"); return []

def layer(out: str, any_bits: bool = False) -> tuple[str, dict]:
  exact = lambda r: r.get("logits_bitwise_vs_v1") == 1.0 and r.get("pool_bitwise_vs_v1") == 1.0 and r.get("tokens_vs_v1") is True
  recs = _load(out, "layer_bench.json")
  if not recs: log("layer_choice: no bench results: step 3's v2s (measured bitwise on trn1)"); return "v2s", {}
  tot, bad = _rank(recs, exact, "v1", "layer")
  ok = {v: t for v, t in tot.items() if any_bits or v not in bad}
  for v, t in sorted(tot.items(), key=lambda x: x[1]): log(f"layer_choice: {v:<6} rows 1 + rows 4 at 512 slots = {t:.4f} ms{'' if v not in bad else '  (not eligible)'}")
  best = min(ok, key=ok.get) if ok else "v1"
  if "v1" in tot and best != "v1" and ok[best] >= tot["v1"]: best = "v1"
  return best, tot

def head(out: str, any_bits: bool = False) -> tuple[str, dict]:
  exact = lambda r: r.get("logits_bitwise_vs_v1") == 1.0 and r.get("argmax_ok") is True
  recs = [r for r in _load(out, "head_bench.json") if not r["variant"].endswith("-logits")]
  if not recs: log("head_choice: no bench results: step 3's v2:1024:1 (measured bitwise on trn1)"); return "v2:1024:1", {}
  tot, bad = _rank(recs, exact, "v3", "head")
  ok = {v: t for v, t in tot.items() if any_bits or v not in bad}
  for v, t in sorted(tot.items(), key=lambda x: x[1]): log(f"head_choice: {v:<12} rows 1 + rows 4 = {t:.4f} ms{'' if v not in bad else '  (not eligible)'}")
  best = min(ok, key=ok.get) if ok else "v3"
  return best, tot

def fuse(out: str, base: str, labels: list[str]) -> str:
  def load(v):
    try: return json.load(open(os.path.join(out, f"serve_{v}.json")))
    except Exception: return None
  b = load(base)
  if b is None: log(f"fuse_choice: no serve_{base}.json: {base}"); return base
  best, bt = base, (b.get("c1") or {}).get("tpot_ms_median") or 1e9
  log(f"fuse_choice: {base:<12} c1 TPOT {bt} ms (the reference)")
  for v in labels:
    d = load(v)
    if d is None: log(f"fuse_choice: {v:<12} no result"); continue
    t, same = (d.get("c1") or {}).get("tpot_ms_median"), (d.get("c1") or {}).get("ids") == (b.get("c1") or {}).get("ids")
    nl = {k: x.get("launches") for k, x in (d.get("launches") or {}).items() if k.startswith("decode_1")}
    log(f"fuse_choice: {v:<12} c1 TPOT {t} ms, c1 tokens == {base}: {same}, launches {nl}")
    if same and t is not None and t < bt: best, bt = v, t
  log(f"fuse_choice: chose {best}")
  return best

def main() -> None:
  what, out = sys.argv[1], sys.argv[2]
  if what == "layer":
    v, _ = layer(out); log(f"layer_choice: chose {v}"); print(_layer_env(v))
  elif what == "head":
    v, _ = head(out); log(f"head_choice: chose {v}"); print(_head_env(v))
  elif what == "candidate":   # the fastest of everything, when it is not the bitwise choice and is faster
    (lb, lt), (hb, ht) = layer(out), head(out)
    (la, lta), (ha, hta) = layer(out, True), head(out, True)
    gain = (lt.get(lb, 0) - lta.get(la, 0)) * 12 + (ht.get(hb, 0) - hta.get(ha, 0)) / 2   # ~ms per token: 24 layers, one head (rows 1 + 4 halved)
    if (la, ha) != (lb, hb) and gain > 0.05:
      log(f"candidate: layer {la}, head {ha} (not bitwise; ~{gain:.2f} ms per token faster than layer {lb}, head {hb} by the benches)")
      print(_layer_env(la) + " " + _head_env(ha))
    else: log(f"candidate: none (the bitwise choice is the fastest, or within 0.05 ms per token)")
  elif what == "fuse": print(fuse(out, sys.argv[3], sys.argv[4:]))
  else: sys.exit(f"unknown choice {what!r}")

if __name__ == "__main__": main()
