"""I77: the prefill layer's device time, old (the member kernels, I75's path) vs new (FZ_NEURON_PREFILL_KERNEL=v2,
fusezap/codegen/nki_prefill.py), by scripts/nki_layer_cost.py's list-schedule model on the kernels' NKISHIM instruction
traces, at Qwen2.5-7B's TP=2 shard shapes (D 3584, I 9472 per rank, 14 / 2 heads, head_dim 128, bf16 weights).

  uv run python scripts/nki_prefill_cost.py [--rows 128,512] [--ctx 128,256,512] [--json out.json]

What it does: a random 3-layer model of the shard's shapes runs one prefill of `rows` tokens at a pool of `ctx` slots on
NKISHIM (numpy; FZ_NEURON_FUSE=1: one fused kernel per layer, the head apart), once per path; the middle layer's fused
kernel is then recorded (instructions only) and scheduled with nki_layer_cost.PARAMS (fitted on trn1, Qwen2.5-0.5B). Per
path and shape: the layer's model ms, per member ("# part k: <kernel>") its share of the critical path and its engine
busy time, its DMA count and bytes, and the SBUF / PSUM peaks (nki_prefill.onchip_peak: live tiles per partition).
The all-reduces are not in it (a TP=1 model of the shard: no collectives); the head (lm_head of 152064 x 3584 for 4 rows)
is modelled separately: MatmulBF16 (old) vs PFLinear on half the vocabulary (new, FZ_NEURON_PFK_HEAD=split).

Calibration (MEASURED_INF2, I75 inf2 run, 128 rows): the old path's MLP half (norm, gate_up, silu*mul, down) 1.84 ms and
attention half (norm .. o_proj, 1024 slots) 1.60 ms per layer, separate launches. The model's old-path numbers against them
give a factor per half, applied to the new path's halves as well (printed as 'calibrated'): model, not measurement.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

SHARD = dict(hidden=3584, inter=9472, heads=14, kv=2, hd=128)
MEASURED_INF2 = {"mlp_half_128": 1.836, "attn_half_128_c1024": 1.599, "head_old": 6.456, "prefill_128_c128_all": 81.61}
HEAD = dict(V=152064, K=3584, R=4)

def log(*a) -> None: print(*a, flush=True)

def layer_item(mode: str, rows: int, ctx: int, shard: dict, exp: str = "exact"):
  """(fused program of the middle layer, its argument sizes, member names) of a prefill at rows x ctx on NKISHIM."""
  from fusezap.device import nkishim as S
  import nki_layer_cost as C
  from nki_n7_layer_bench import qwen_shaped
  from fusezap.serve import neuron_prefill as PF
  saved = {k: os.environ.get(k) for k in ("FZ_NEURON_FUSE", "FZ_NEURON_FUSE_MAX", "FZ_NEURON_HEAD_FUSE", "FZ_NEURON_DTYPE", "FZ_NEURON_PLAN", "FZ_NEURON_PFK_EXP", "FZ_NEURON_BF16_HEAD")}
  os.environ.update({"FZ_NEURON_FUSE": "1", "FZ_NEURON_FUSE_MAX": "0", "FZ_NEURON_HEAD_FUSE": "0", "FZ_NEURON_DTYPE": "bf16", "FZ_NEURON_PLAN": "0", "FZ_NEURON_BF16_HEAD": "0",
                     "FZ_NEURON_PFK_EXP": exp})
  PF.set_mode(None, mode)
  real = S.nisa.__dict__.copy()
  for nm in C.ISA: setattr(S.nisa, nm, getattr(S.isa, nm))   # the model runs for real
  try:
    m = qwen_shaped("NKISHIM", 1, ctx, layers=3, hidden=shard["hidden"], inter=shard["inter"], heads=shard["heads"], kv=shard["kv"], hd=shard["hd"])
    m.prefill_chunk = rows
    ids = [[(7 * i + 3) % 500 for i in range(rows)]]
    for _ in range(2): m.reset(); m.prefill(ids)   # the second call captures (and fuses)
    jf = m._neuron_steps.jits[(rows, 1)]
    fused = [it.si for it in jf.captured.items if not it.const and it.si is not None and hasattr(getattr(it.si.ast[0], "spec", None), "parts")]
  finally:
    for nm in C.ISA: setattr(S.nisa, nm, real[nm])
    PF.set_mode(None, None)
    for k, v in saved.items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = v
  si = fused[1]   # layer 1 (layer 0 also builds the step's one-hot slot matrix)
  sp = si.ast[0].spec
  return sp.render(), [b.size for b in si.bufs[:sp.n]], [p[0] for p in sp.parts]

def analyse(prog, sizes, members, P) -> dict:
  import nki_layer_cost as C
  tr = C.record(prog, sizes)
  r = C.schedule(tr, P, crit=True)
  dmas = [x for x in tr["trace"] if x[0] == "dma"]
  per = {}
  for sec in tr["sections"]:
    if not sec[0].isdigit() or ":" not in sec: continue
    i = int(sec.split(":")[0])
    nm = members[i] if i < len(members) else sec
    b = r["busy"].get(sec, {})
    per[f"{i}: {nm}"] = {"crit_us": round(1e6 * r["path_sec"].get(sec, 0.0), 1), "busy_us": {e: round(1e6 * v, 1) for e, v in sorted(b.items())},
                         "dmas": sum(1 for x in dmas if tr["sections"][x[9]] == sec)}
  return {"model_ms": round(1e3 * r["s"], 4), "instructions": tr["n"], "dmas": len(dmas),
          "dma_mb": round(sum(x[2] for x in dmas) / 1e6, 2), "members": per,
          "path_eng_us": {k: round(1e6 * v, 1) for k, v in sorted(r["path_eng"].items(), key=lambda x: -x[1])}}

def head_model(P) -> dict:
  import nki_layer_cost as C
  from fusezap.codegen.nki_bf16 import MatmulBF16
  from fusezap.codegen.nki_prefill import PFLinear
  V, K, R = HEAD["V"], HEAD["K"], HEAD["R"]
  old = MatmulBF16(R, V, K).render()
  new = PFLinear(R, V // 2, K, V, 0)
  return {"old_ms": round(1e3 * C.schedule(C.record(old, [R * V, R * K, K * V]), P)["s"], 3),
          "new_split_ms": round(1e3 * C.schedule(C.record(new.render(), new.sizes()), P)["s"], 3)}

def halves(res: dict) -> tuple[float, float]:
  """(attention half, MLP half) model ms of a layer: the members' critical-path shares split at the MLP's norm."""
  att = mlp = 0.0
  mem = list(res["members"].items())
  cut = next((i for i, (k, _) in enumerate(mem) if "pfmlp" in k or (i > 0 and "rmsnorm_add" in k)), len(mem))
  tot = sum(v["crit_us"] for _, v in mem) or 1.0
  for i, (k, v) in enumerate(mem): (att := att + v["crit_us"]) if i < cut else (mlp := mlp + v["crit_us"])
  return res["model_ms"] * att / tot, res["model_ms"] * mlp / tot

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--rows", default="128,512")
  ap.add_argument("--ctx", default="128,256,512")
  ap.add_argument("--paths", default="v1,v2exact,v2")
  ap.add_argument("--json", default="")
  a = ap.parse_args()
  os.environ.setdefault("FZ_DEVICE", "NKISHIM")
  import nki_layer_cost as C
  from fusezap.codegen.nki_prefill import onchip_peak
  C.install()
  P = dict(C.PARAMS)
  out: dict = {"params": P, "shard": SHARD, "measured_inf2": MEASURED_INF2, "layers": {}}
  t0 = time.perf_counter()
  for rows in [int(x) for x in a.rows.split(",")]:
    for ctx in [int(x) for x in a.ctx.split(",")]:
      if ctx < rows: continue
      for path in a.paths.split(","):
        mode, exp = ("v1", "exact") if path == "v1" else ("v2", "exact" if path.endswith("exact") else "act")
        prog, sizes, members = layer_item(mode, rows, ctx, SHARD, exp)
        res = analyse(prog, sizes, members, P)
        pk = onchip_peak(prog, sizes)
        res["sbuf_peak_kib"], res["psum_peak_kib"] = round(pk["sbuf"] / 1024, 1), round(pk["psum"] / 1024, 1)
        out["layers"][f"{path} r{rows} c{ctx}"] = res
        log(f"== {path} rows {rows} ctx {ctx}: model {res['model_ms']:.3f} ms per layer, {res['instructions']} instructions, {res['dmas']} DMAs "
            f"({res['dma_mb']} MB), SBUF peak {res['sbuf_peak_kib']} KiB, PSUM {res['psum_peak_kib']} KiB; critical path by engine (us): {res['path_eng_us']}")
        for k, v in res["members"].items(): log(f"     {k:<48} crit {v['crit_us']:8.1f} us  dmas {v['dmas']:5d}  busy {v['busy_us']}")
  out["head"] = head_model(P)
  log(f"== head (4 rows x 152064 x 3584 bf16): old MatmulBF16 {out['head']['old_ms']} ms, new PFLinear on half the vocabulary {out['head']['new_split_ms']} ms (model)")
  # calibration against the I75 inf2 measurements (old path, 128 rows) and the predicted 128-row prefill (28 layers + head)
  L = out["layers"]
  if "v1 r128 c512" in L or "v1 r128 c128" in L:
    o = L.get("v1 r128 c128") or L["v1 r128 c512"]
    oa, om = halves(o)
    f_mlp = MEASURED_INF2["mlp_half_128"] / om if om else 1.0
    o1k = L.get("v1 r128 c1024")
    f_att = MEASURED_INF2["attn_half_128_c1024"] / halves(o1k)[0] if o1k else f_mlp
    out["calibration"] = {"mlp": round(f_mlp, 3), "attn": round(f_att, 3)}
    log(f"calibration (inf2 old-path halves / model): MLP x{f_mlp:.2f}, attention x{f_att:.2f}" + ("" if o1k else " (attention: no 1024-slot run, the MLP factor)"))
    hf = MEASURED_INF2["head_old"] / out["head"]["old_ms"] if out["head"]["old_ms"] else 1.0
    for k, v in L.items():
      at, ml = halves(v)
      v["calibrated_ms"] = round(at * f_att + ml * f_mlp, 4)
      rows = int(re.search(r"r(\d+)", k).group(1))
      head = (out["head"]["new_split_ms"] if not k.startswith("v1") else out["head"]["old_ms"]) * hf
      v["prefill_28_layers_calibrated_ms"] = round(28 * v["calibrated_ms"] + head, 1)
      log(f"predict {k:<18} layer model {v['model_ms']:.3f} ms, calibrated {v['calibrated_ms']:.3f} ms (attn {at:.3f}, mlp {ml:.3f} model) -> "
          f"{rows}-row chunk ~{v['prefill_28_layers_calibrated_ms']} ms (28 layers + head, no all-reduces)")
  log(f"done in {time.perf_counter() - t0:.0f} s")
  if a.json:
    with open(a.json, "w") as f: json.dump(out, f, indent=1)

if __name__ == "__main__": main()
