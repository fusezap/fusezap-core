"""NKI-N7 step 3: one decoder layer alone on the device, per layer path, rows and pool prefix: device ms per launch (nrtpy's
device benchmark of the replayed launch) and the bits (every variant's logits, tokens and KV pool after the same steps vs
v1's). A 3-layer model with Qwen2.5-0.5B's shapes (random weights, seeded), FZ_NEURON_FUSE=1: the decode forward's
launches are [one-hot slots + layer 0, layer 1, layer 2, head]; layer 1 is the timed kernel (v1: the ten members fused,
fz_fused10_*; v2*: one LayerSpec, fz_layer*). Every (rows, ctx) gets its own model: `rows` sequences, a pool of exactly
`ctx` slots (the KV write and attention run over all of it, as at that pool prefix in serving).

  FZ_DEVICE=NEURON .venv/bin/python scripts/nki_n7_layer_bench.py --out results/nki_n7_step3 --rows 1,4 --ctx 512,768,1024 --variants v1,v2,v2s
  FZ_DEVICE=NKISHIM .venv/bin/python scripts/nki_n7_layer_bench.py --rows 1 --ctx 64 --hidden 64 --inter 96 --heads 4 --kv 2 --hd 16   # Mac dry run

Variants (nki_kernels.LAYER_VARIANTS): v1 (the members), v2 (LayerSpec: kv heads one at a time, RoPE by column slices),
v2s (kv heads stacked in one tile), v2a (RoPE by 3-D access patterns), v2sa (both); N7 step 4: e (early: input-only loads
first, qᵀ / oᵀ on chip, the first key block without the rescale; bitwise by construction), w (gate_up / down
weight-stationary; bitwise only if TensorE's float32 matmul is operand-symmetric): v2e, v2se, v2sw, v2swe. N8 (I70, --dtype
bf16): "<v>+w<cols>[p<blocks>][q]" streams the bf16 weights in <cols>-column tiles, prefetching <blocks> gate_up blocks during
attention (and with q every qkv / o_proj tile early): nki_bf16_n8.LayerBF16W, bitwise by construction (e.g. v2se+w2048p2q).
"bitwise_by_construction" in each record says which (nki_kernels.LAYER_BITWISE); the job's choice needs the measured bits. --variants takes "v2@512" to run a
variant at that ctx only. --ablate: at the first (rows, ctx) also each v1 member alone (a FusedSpec of that part only, on
the same buffers): the measured per-member breakdown. Writes <out>/layer_bench.json and one line per (rows, ctx, variant):
"layer_bench rows R ctx C <variant> <kernel> device X ms per layer ...".
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def log(*a) -> None: print(*a, flush=True)

def qwen_shaped(dev: str, rows: int, ctx: int, layers: int = 3, hidden: int = 896, inter: int = 4864, heads: int = 14, kv: int = 2,
                hd: int = 64, vocab: int = 512, seed: int = 0):
  """A random Qwen2-architecture model (q/k/v biases, no o_proj bias) of these sizes on `dev`: `rows` sequences of
  ctx // rows positions each (NeuronSteps' pool: exactly ctx slots)."""
  from fusezap.dtype import dtypes
  from fusezap.serve.models.llama import Llama, LlamaConfig
  c = {"model_type": "qwen2", "hidden_size": hidden, "intermediate_size": inter, "num_hidden_layers": layers, "num_attention_heads": heads,
       "num_key_value_heads": kv, "head_dim": hd, "vocab_size": vocab, "rms_norm_eps": 1e-6, "rope_theta": 1e6, "tie_word_embeddings": False,
       "hidden_act": "silu", "max_position_embeddings": 32768, "eos_token_id": vocab - 1}
  rng = np.random.default_rng(seed)
  r = lambda *s, sc=0.05: (rng.standard_normal(s) * sc).astype(np.float32)
  ws = {"model.embed_tokens.weight": r(vocab, hidden, sc=0.5), "model.norm.weight": 1 + r(hidden), "lm_head.weight": r(vocab, hidden)}
  for i in range(layers):
    p = f"model.layers.{i}."
    ws.update({p + "input_layernorm.weight": 1 + r(hidden), p + "post_attention_layernorm.weight": 1 + r(hidden),
               p + "self_attn.q_proj.weight": r(heads * hd, hidden, sc=0.1), p + "self_attn.k_proj.weight": r(kv * hd, hidden, sc=0.1),
               p + "self_attn.v_proj.weight": r(kv * hd, hidden), p + "self_attn.o_proj.weight": r(hidden, heads * hd),
               p + "mlp.gate_proj.weight": r(inter, hidden), p + "mlp.up_proj.weight": r(inter, hidden), p + "mlp.down_proj.weight": r(hidden, inter),
               p + "self_attn.q_proj.bias": r(heads * hd, sc=0.3), p + "self_attn.k_proj.bias": r(kv * hd, sc=0.3), p + "self_attn.v_proj.bias": r(kv * hd)})
  return Llama(LlamaConfig.from_hf(c), ws, dtypes.float32, dev, max_len=ctx // rows, prefill_chunk=16, max_batch=rows)

def set_variant(v: str) -> None:
  """A LAYER_VARIANTS name, optionally "+w<cols>[p<blocks>][q]" (N8, bf16 only: FZ_NEURON_BF16_WN / _PF / _PFQO=1)."""
  import re
  from fusezap.codegen.nki_kernels import LAYER_ENV, LAYER_VARIANTS
  from fusezap.codegen.nki_bf16_n8 import N8_ENV
  v, _, n8 = v.partition("+")
  for k in LAYER_ENV + tuple(e for e in N8_ENV if "HEAD" not in e): os.environ.pop(k, None)
  os.environ.update(LAYER_VARIANTS[v])
  os.environ["FZ_NEURON_BF16_WN"] = "512"   # N8 default is 4096 (nki_bf16_n8.WN_DEFAULT): a variant without +w is I68's layer
  if n8:
    m = re.fullmatch(r"w(\d+)(?:p(\d+))?(q?)", n8)
    assert m, f"variant suffix +{n8}: w<cols>[p<blocks>][q]"
    os.environ.update({"FZ_NEURON_BF16_WN": m.group(1), "FZ_NEURON_BF16_PF": m.group(2) or "0", "FZ_NEURON_BF16_PFQO": "1" if m.group(3) else "0"})

def steps(m, rows: int, n: int) -> list[np.ndarray]:
  """A fresh NeuronSteps (zeroed pool), then n greedy decode steps from position 0 (no prefill: only the decode forward's
  NEFFs are compiled; attention scans every slot of the pool whatever the positions): every step's logits."""
  m._neuron_steps, m.pos = None, 0
  out = [m.decode([[5 + 3 * b] for b in range(rows)])]
  for i in range(n - 1): out.append(m.decode([[int(np.argmax(out[-1][b]))] for b in range(rows)]))
  return out

def main() -> None:
  os.environ.setdefault("FZ_NEURON_PLAN", "0")   # N8: ablate() and the cost model read the captured items' buffers, which a planned capture drops
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="")
  ap.add_argument("--rows", default="1,4")
  ap.add_argument("--ctx", default="512,1024")
  ap.add_argument("--variants", default="v1,v2")
  ap.add_argument("--ablate", action="store_true")
  ap.add_argument("--replays", type=int, default=5)
  ap.add_argument("--hidden", type=int, default=896)
  ap.add_argument("--inter", type=int, default=4864)
  ap.add_argument("--heads", type=int, default=14)
  ap.add_argument("--kv", type=int, default=2)
  ap.add_argument("--hd", type=int, default=64)
  ap.add_argument("--dtype", default="f32", choices=["f32", "bf16", "bf16a"],
                  help="I68: bf16 = bf16 weights (FZ_NEURON_DTYPE=bf16; bits vs the bf16 members v1), bf16a = also FZ_NEURON_BF16_ATTN=1")
  args = ap.parse_args()
  os.environ["FZ_NEURON_FUSE"] = "1"
  os.environ["FZ_NEURON_HEAD_FUSE"] = "0"   # the head its own launch (N8 default: fused into the last layer's kernel)
  if args.dtype != "f32": os.environ["FZ_NEURON_DTYPE"] = "bf16"
  if args.dtype == "bf16a": os.environ["FZ_NEURON_BF16_ATTN"] = "1"
  from fusezap.codegen import nki_kernels as nk
  from fusezap.device import Device
  from fusezap.device.neuron import neuron_profile
  from fusezap.serve import neuron_kv as kv
  dev = Device.DEFAULT
  assert kv.is_neuron(dev), f"FZ_DEVICE must be a NEURON tier, got {dev}"
  D = Device[dev]
  al = getattr(D, "allocator", None)
  timed = hasattr(al, "profile")
  t0 = time.perf_counter()
  results: list = []
  def save() -> None:   # after every record: a phase timeout keeps what was measured
    if args.out:
      os.makedirs(args.out, exist_ok=True)
      with open(os.path.join(args.out, "layer_bench.json"), "w") as fh: json.dump({"args": vars(args), "results": results}, fh, indent=1)
  first = True
  for rows in (int(r) for r in args.rows.split(",")):
    for ctx in (int(c) for c in args.ctx.split(",")):
      m = qwen_shaped(dev, rows, ctx, hidden=args.hidden, inter=args.inter, heads=args.heads, kv=args.kv, hd=args.hd)
      ref = None
      for var in args.variants.split(","):
        var, _, only = var.partition("@")
        if only and int(only) != ctx: continue
        rec: dict = {"rows": rows, "ctx": ctx, "variant": var, "bitwise_by_construction": var.partition("+")[0] in nk.LAYER_BITWISE}
        try:
          set_variant(var)
          st = time.perf_counter()
          lg = steps(m, rows, 3)   # decode 1: eager, 2: capture (fused), 3: first replay (compiles the NEFFs)
          rec["first_calls_s"] = round(time.perf_counter() - st, 1)
          items = [it for it in m._neuron_steps.jits[(rows, rows)].captured.items if not it.const]
          names = [it.si.ast[0].spec.name for it in items]
          rec.update(launches=len(items), kernels=names, kernel=names[1])
          if timed:
            al.flush(); al.profile, al.prof, al.prof_wall, al.prof_ipc, al.bindings = True, {}, [0, 0.0], [0, 0.0, 0.0], {}
            for _ in range(args.replays): m.decode([[1 + b] for b in range(rows)])
            al.flush(); al.profile = False
            rep = neuron_profile(al, iters=20, bench=True)
            rec["device_ms_all"] = {nm: k["device_ms"] for nm, k in rep["kernels"].items()}
            rec["device_ms"] = rep["kernels"].get(names[1], {}).get("device_ms")
            if rep.get("bench_errors"): rec["bench_errors"] = rep["bench_errors"]
            if args.ablate and first and var == "v1":   # each member of layer 1's fused kernel alone, on the same buffers
              rec["ablation"] = ablate(D, al, items[1].si)
          lg = steps(m, rows, 6)   # the bits: a fresh pool, the same steps as every other variant
          pools = [p.numpy() for p in m._neuron_steps.cache.state()]
          toks = [np.argmax(x, -1).tolist() for x in lg]
          if ref is None and var == "v1": ref = (lg, toks, pools)
          if ref is not None:
            rec["logits_bitwise_vs_v1"] = round(float(np.mean([np.mean(a.view(np.int32) == b.view(np.int32)) for a, b in zip(lg, ref[0])])), 6)
            rec["tokens_vs_v1"] = toks == ref[1]
            rec["pool_bitwise_vs_v1"] = round(float(np.mean([np.mean(a.view(np.int32) == b.view(np.int32)) for a, b in zip(pools, ref[2])])), 6)
        except Exception as e:
          log(f"layer_bench rows {rows} ctx {ctx} {var:<5} SKIPPED: {type(e).__name__}: {str(e)[:300]}")
          results.append({**rec, "skipped": f"{type(e).__name__}: {str(e)[:300]}"}); save()
          continue
        results.append(rec); save()
        log(f"layer_bench rows {rows} ctx {ctx} {var:<5} {rec['kernel']:<32} device {rec.get('device_ms')} ms per layer  "
            f"logits bitwise vs v1 {rec.get('logits_bitwise_vs_v1')}  tokens vs v1 {rec.get('tokens_vs_v1')}  pool bitwise vs v1 "
            f"{rec.get('pool_bitwise_vs_v1')}  launches {rec['launches']}  first 3 calls {rec['first_calls_s']} s")
        for nm, d in (rec.get("ablation") or {}).items(): log(f"layer_bench   v1 member alone: {nm:<44} device {d} ms")
      first = False
      del m
  log(f"layer_bench done in {time.perf_counter() - t0:.0f} s")
  if args.out:
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "layer_bench.json"), "w") as fh: json.dump({"args": vars(args), "results": results}, fh, indent=1)

def ablate(D, al, si) -> dict:
  """Device ms of each member of a fused layer kernel alone (FusedSpec of that one part, the same arguments)."""
  from fusezap.codegen.nki_kernels import FusedSpec
  from fusezap.device.neuron import neuron_profile
  sp = si.ast[0].spec
  subs = {f"{k}: {p[0]}": FusedSpec((p,), sp.n, sp.outs, sp.dts) for k, p in enumerate(sp.parts)}
  al.flush(); al.profile, al.prof, al.prof_wall, al.prof_ipc, al.bindings = True, {}, [0, 0.0], [0, 0.0, 0.0], {}
  out = {}
  for nm, sub in subs.items():
    try: D.run_program(D.spec_program(sub), si.bufs)
    except Exception as e: out[nm] = f"{type(e).__name__}: {str(e)[:120]}"
  al.flush(); al.profile = False
  rep = neuron_profile(al, iters=20, bench=True)
  for nm, sub in subs.items(): out.setdefault(nm, rep["kernels"].get(sub.name, {}).get("device_ms"))
  return out

if __name__ == "__main__": main()
