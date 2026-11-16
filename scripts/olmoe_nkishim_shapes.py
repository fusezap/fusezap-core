#!/usr/bin/env python3
"""I105: OLMoE-1B-7B's real layer shapes on the NEURON path without hardware (NKISHIM): a random-weight checkpoint with the
published sizes (D 2048, 16 / 16 heads of 128, 64 experts of 1024, top 8, vocabulary 50,304) but --layers layers runs
LLMEngine on NKISHIM in float32 and bf16 (the kernels the chip will compile: the members of a decoder layer, the router, the
2048 x 131072 gate_up and 65536 x 2048 down matmuls, the head), and the float32 tokens are compared with CPU's. Prints the
HBM accounting (neuron_kv.hbm_report) per layer, to extrapolate to 16 layers.

  .venv/bin/python scripts/olmoe_nkishim_shapes.py --layers 2 --json results/i105/olmoe_nkishim_shapes.json

I105 A2, --sparse: only NKISHIM bf16, the dense model vs FZ_MOE_SPARSE=1 (the chosen experts' tiles by indirect DMA in the
decode steps, FZ_NKISHIM_MM=f64 so the matmul bits are shape-invariant like the PE array): tokens equal, and the number of
sparse kernel launches. --tp-width 512 shrinks the expert width to what one rank holds under the sliced TP=2 layout.
  .venv/bin/python scripts/olmoe_nkishim_shapes.py --sparse --layers 1 --tokens 4 --json docs/status/i105/olmoe_nkishim_sparse.json
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

REAL = {"architectures": ["OlmoeForCausalLM"], "model_type": "olmoe", "hidden_act": "silu", "hidden_size": 2048, "intermediate_size": 1024,
        "num_attention_heads": 16, "num_key_value_heads": 16, "num_experts": 64, "num_experts_per_tok": 8, "norm_topk_prob": False,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "max_position_embeddings": 4096, "vocab_size": 50304, "tie_word_embeddings": False,
        "eos_token_id": 50279, "clip_qkv": None}

def write_ckpt(path: str, layers: int, seed: int = 0, Ie: int = 1024) -> str:
  from test_tp import write_safetensors
  c = {**REAL, "num_hidden_layers": layers, "intermediate_size": Ie}
  rng = np.random.default_rng(seed)
  D, E, H, hd, V = 2048, 64, 16, 128, 50304
  r = lambda *s, sc=0.02: (rng.standard_normal(s, dtype=np.float32) * sc)
  ts = {"model.embed_tokens.weight": r(V, D, sc=1.0), "model.norm.weight": 1 + r(D, sc=0.1), "lm_head.weight": r(V, D)}
  for i in range(layers):
    p = f"model.layers.{i}."
    ts.update({p + "input_layernorm.weight": 1 + r(D, sc=0.1), p + "post_attention_layernorm.weight": 1 + r(D, sc=0.1),
               p + "self_attn.q_proj.weight": r(H * hd, D), p + "self_attn.k_proj.weight": r(H * hd, D), p + "self_attn.v_proj.weight": r(H * hd, D),
               p + "self_attn.o_proj.weight": r(D, H * hd), p + "self_attn.q_norm.weight": 1 + r(H * hd, sc=0.1),
               p + "self_attn.k_norm.weight": 1 + r(H * hd, sc=0.1), p + "mlp.gate.weight": r(E, D, sc=0.05)})
    for e in range(E):
      ts.update({p + f"mlp.experts.{e}.gate_proj.weight": r(Ie, D), p + f"mlp.experts.{e}.up_proj.weight": r(Ie, D),
                 p + f"mlp.experts.{e}.down_proj.weight": r(D, Ie)})
  os.makedirs(path, exist_ok=True)
  write_safetensors(os.path.join(path, "model.safetensors"), ts)
  with open(os.path.join(path, "config.json"), "w") as f: json.dump(c, f)
  return path

PROMPTS = [[1, 2, 3, 4, 5, 6, 7], [9, 8, 7], list(range(30, 60))]

def run(path: str, device: str, dtype, tokens: int) -> tuple[dict, dict]:
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine, SamplingParams
  from fusezap.serve.neuron_kv import hbm_report
  m = load_model(path, dtype=dtype, device=device, max_len=256, prefill_chunk=128)
  eng = LLMEngine(m, num_blocks=16, block_size=16, max_num_seqs=4, max_prefill_tokens=64, max_model_len=256, decode_beam=0)
  t0 = time.time()
  out = dict(enumerate(eng.generate(PROMPTS, SamplingParams(max_tokens=tokens, ignore_eos=True))))
  info = {"s": round(time.time() - t0, 1)}
  if device != "CPU":
    try: info["hbm"] = {k: v for k, v in hbm_report(eng).items() if isinstance(v, (int, float))}
    except Exception as e: info["hbm"] = f"{type(e).__name__}: {e}"
  return out, info

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--layers", type=int, default=2)
  ap.add_argument("--tokens", type=int, default=6)
  ap.add_argument("--dir", default="")
  ap.add_argument("--json", default="")
  ap.add_argument("--sparse", action="store_true", help="I105 A2: NKISHIM bf16 dense vs FZ_MOE_SPARSE=1 only")
  ap.add_argument("--tp-width", type=int, default=1024, help="the expert width (512: one rank's share under the sliced TP=2 layout)")
  a = ap.parse_args()
  from fusezap.dtype import dtypes
  d = a.dir or tempfile.mkdtemp(prefix="olmoe_shapes_")
  if not os.path.exists(os.path.join(d, "model.safetensors")): write_ckpt(d, a.layers, Ie=a.tp_width)
  res: dict = {"layers": a.layers, "tokens": a.tokens}
  if a.sparse:
    from fusezap.serve.models.llama import Llama
    os.environ.update(FZ_NEURON_DTYPE="bf16", FZ_NKISHIM_MM="f64")
    os.environ["FZ_MOE_SPARSE"] = "0"
    dn, res["dense_bf16"] = run(d, "NKISHIM", dtypes.bfloat16, a.tokens)
    calls: list[int] = []
    orig = Llama._moe_sparse
    Llama._moe_sparse = lambda self, h, l: calls.append(int(h.shape[0])) or orig(self, h, l)
    os.environ["FZ_MOE_SPARSE"] = "1"
    sp, res["sparse_bf16"] = run(d, "NKISHIM", dtypes.bfloat16, a.tokens)
    res["sparse_bf16"].update(tokens_equal_dense=sp == dn, sparse_traces_rows=calls, tokens=sp, expert_width=a.tp_width)
    print(json.dumps(res, indent=1, default=str))
    if a.json:
      with open(a.json, "w") as f: json.dump(res, f, indent=1, default=str)
    return
  cpu, res["cpu"] = run(d, "CPU", dtypes.float32, a.tokens)
  nk, res["nkishim_f32"] = run(d, "NKISHIM", dtypes.float32, a.tokens)
  res["nkishim_f32"]["tokens_equal_cpu"] = nk == cpu
  os.environ["FZ_NEURON_DTYPE"] = "bf16"
  nb, res["nkishim_bf16"] = run(d, "NKISHIM", dtypes.bfloat16, a.tokens)
  res["nkishim_bf16"]["tokens_equal_cpu"] = sum(x == y for r in cpu for x, y in zip(nb[r], cpu[r])), sum(len(t) for t in cpu.values())
  print(json.dumps(res, indent=1, default=str))
  if a.json:
    with open(a.json, "w") as f: json.dump(res, f, indent=1, default=str)

if __name__ == "__main__": main()
