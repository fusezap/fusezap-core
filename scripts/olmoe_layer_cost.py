#!/usr/bin/env python3
"""I105 A2 run 2 follow-up: where an OLMoE decode layer's device time goes, from the N7/N8 cost model (scripts/nki_layer_cost.py:
the rendered kernel's instructions recorded on NKISHIM, a list schedule with PARAMS fitted to trn1 measurements; a model,
not a measurement). Builds a random checkpoint at OLMoE's real sizes with --layers layers and expert width --width (512 =
one rank's share under the sliced TP=2 layout, so the per-core kernels are the chip's minus the all-reduce), runs decode
steps of --rows rows on NKISHIM in bf16 (FZ_NEURON_FUSE=1: one fused member kernel per layer) dense and sparse, and prints
each fused kernel's modelled time per member ("# part k: <member>").

  .venv/bin/python scripts/olmoe_layer_cost.py --dir <scratch>/olmoe_l1 --rows 1 --json docs/status/i105/olmoe_layer_cost.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--dir", required=True)
  ap.add_argument("--layers", type=int, default=2)
  ap.add_argument("--width", type=int, default=512)
  ap.add_argument("--rows", type=int, default=1)
  ap.add_argument("--ctx", type=int, default=1024, help="pool slots (max_len): FZ_NEURON_CTX=0 scans all of them")
  ap.add_argument("--idma", default="scalar")
  ap.add_argument("--json", default="")
  a = ap.parse_args()
  from olmoe_nkishim_shapes import write_ckpt
  import nki_layer_cost as C
  from fusezap.device import nkishim as S
  from fusezap.dtype import dtypes
  from fusezap.serve.models import load_model
  if not os.path.exists(os.path.join(a.dir, "model.safetensors")): write_ckpt(a.dir, a.layers, Ie=a.width)
  os.environ.update(FZ_NEURON_DTYPE="bf16", FZ_NEURON_FUSE="1", FZ_NEURON_PLAN="0", FZ_NEURON_CTX="0", FZ_MOE_IDMA=a.idma)
  res: dict = {"layers": a.layers, "width": a.width, "rows": a.rows, "ctx": a.ctx, "params": "nki_layer_cost.PARAMS (trn1 fit)"}
  for mode in ("dense", "sparse"):
    if mode == "sparse": os.environ["FZ_MOE_SPARSE"] = "1"
    else: os.environ["FZ_MOE_SPARSE"] = "0"
    m = load_model(a.dir, dtype=dtypes.bfloat16, device="NKISHIM", max_len=a.ctx, prefill_chunk=16, max_batch=a.rows)
    m.reset()
    for t in range(3): m.decode([[5 + t]] * a.rows)
    items = [it for it in m._neuron_steps.jits[(a.rows, a.rows)].captured.items if not it.const]
    real = S.nisa.__dict__.copy()
    C.install()
    out = []
    try:
      for it in items:
        sp = getattr(it.si.ast[0], "spec", None) if it.si.ast else None
        if sp is None or sp.name == "fz_fence": continue
        prog = sp.render()
        tr = C.record(prog, [b.size for b in it.si.bufs[:prog.n_bufs]])
        r = C.schedule(tr, C.PARAMS, crit=True)
        out.append({"kernel": tr["name"][:60], "model_ms": round(r["s"] * 1e3, 4), "instructions": tr["n"],
                    "crit_us_by_section": {k: round(1e6 * v, 1) for k, v in r["path_sec"].items() if v > 0},
                    "busy_us_by_section": {k: {e: round(1e6 * x, 1) for e, x in b.items()} for k, b in r["busy"].items()}})
    finally:
      for nm in C.ISA: setattr(S.nisa, nm, real[nm])
      S.nl.ndarray = S.ndarray
    res[mode] = out
    print(f"== {mode}: {len(out)} kernels per decode step")
    for k in out:
      print(f"  {k['kernel']:<60} {k['model_ms']:8.3f} ms  {k['instructions']} instr")
      for s, v in sorted(k["crit_us_by_section"].items(), key=lambda x: -x[1])[:12]: print(f"     crit {v:9.1f} us  {s}")
    del m
  if a.json:
    with open(a.json, "w") as f: json.dump(res, f, indent=1)

if __name__ == "__main__": main()
