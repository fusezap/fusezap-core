"""N8 (I71): the 7B job's layer choice from its layer bench (scripts/nki_n8_7b_inf2.sh, <out>/layer_bench.json).

  python3 scripts/nki_n8_7b_choose.py <out>    # stdout: env words for `env`; stderr: the table the choice came from

Rule: among the tiled variants (v2set*, FZ_NEURON_LAYER_TILE=1 with 512 / 1024 / 2048 weight columns) the fastest by device
ms per layer whose every record is bitwise equal to v1 (the bf16 members: logits and KV pool) with v1's tokens, if it is
faster than v1; else the members (FZ_NEURON_LAYER_TILE=0, I69's path). No results at all: the members too (a tiled NEFF that
does not compile in the bench must not be the serving path).
"""
from __future__ import annotations
import json, os, sys

WCOLS = {"v2set": 512, "v2set2": 1024, "v2set4": 2048, "v2et": 512}

def main() -> None:
  out = sys.argv[1]
  try:
    with open(os.path.join(out, "layer_bench.json")) as f: recs = json.load(f)["results"]
  except (OSError, ValueError, KeyError): recs = []
  ms: dict = {}
  bad: set = set()
  for r in recs:
    v = r.get("variant")
    if r.get("skipped") or r.get("device_ms") is None: bad.add(v); continue
    if v != "v1" and not (r.get("logits_bitwise_vs_v1") == 1.0 and r.get("pool_bitwise_vs_v1") == 1.0 and r.get("tokens_vs_v1")): bad.add(v)
    ms[v] = ms.get(v, 0.0) + float(r["device_ms"])
  for v, t in sorted(ms.items(), key=lambda x: x[1]): print(f"layer {v:<8} {t:8.3f} ms{'  (not eligible)' if v in bad else ''}", file=sys.stderr)
  ok = {v: t for v, t in ms.items() if v in WCOLS and v not in bad}
  ref = ms.get("v1")
  if ok and (ref is None or min(ok.values()) < ref):
    v = min(ok, key=ok.get)
    print(f"chosen: {v} ({ok[v]:.3f} ms vs v1 {ref})", file=sys.stderr)
    print(f"FZ_NEURON_LAYER=v2 FZ_NEURON_LAYER_HEADS={'each' if v == 'v2et' else 'stack'} FZ_NEURON_LAYER_EARLY=1 FZ_NEURON_LAYER_TILE=1 FZ_NEURON_LAYER_WCOLS={WCOLS[v]}")
  else:
    print(f"chosen: the members (no eligible tiled variant faster than v1 {ref})", file=sys.stderr)
    print("FZ_NEURON_LAYER_TILE=0")

if __name__ == "__main__": main()
