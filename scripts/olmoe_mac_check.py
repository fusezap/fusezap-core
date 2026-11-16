#!/usr/bin/env python3
"""I105: allenai/OLMoE-1B-7B-0125-Instruct on the Mac (48 GB) against Hugging Face transformers in float32, without a
float32 load of the whole model (27.7 GB: it would swap next to anything else).

  template   FuseZap's chat rendering == HF apply_chat_template's ids; the stop ids
  trunc      the first --layers layers (+ final norm + lm_head) in float32, FuseZap (CPU and METAL) vs HF float32: greedy
             tokens for --tokens steps and the last-position logits
  full       the whole model, FuseZap on --device in --dtype (default METAL bfloat16), greedy --tokens per prompt; then HF
             with float32 compute over the bf16 checkpoint (each layer upcast to float32 only while it runs: an upcast of
             bf16 is exact, so the logits are a float32 model's) teacher-forced over prompt + FuseZap's tokens: per position
             whether HF's argmax is FuseZap's token (greedy agreement; float32 FuseZap should give 1.0) and top-1 agreement

  .venv/bin/python scripts/olmoe_mac_check.py --part template,trunc,full --json results/olmoe_mac.json
FuseZap and HF run in separate processes for `full` (--part full-fz then full-hf), so their memory never adds up.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
import numpy as np

MODEL, REV = "allenai/OLMoE-1B-7B-0125-Instruct", "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e"
CHATS = ["What is the capital of France? Answer in one sentence.", "Write a haiku about the sea.",
         "Explain what a mixture-of-experts language model is in two sentences.", "List three prime numbers greater than 50.",
         "Translate 'good morning' into Spanish and German.", "What is 17 times 23? Show the steps briefly.",
         "Give one tip for writing clear code.", "Name the largest planet in the solar system and one fact about it."]

def path() -> str:
  from huggingface_hub import snapshot_download
  return snapshot_download(MODEL, revision=REV, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])

def prompts(n: int) -> list[list[int]]:
  from fusezap.serve.tokenizer import Tokenizer
  tok = Tokenizer(path())
  return [tok.apply_chat_template([{"role": "user", "content": c}], add_generation_prompt=True) for c in CHATS[:n]]

def part_template() -> dict:
  import transformers as tr
  from fusezap.serve.tokenizer import Tokenizer
  tok, hft = Tokenizer(path()), tr.AutoTokenizer.from_pretrained(path())
  convs = [[{"role": "user", "content": c}] for c in CHATS[:4]] + \
          [[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "Name a color."}]]
  ok = []
  for m in convs:
    want = hft.apply_chat_template(m, add_generation_prompt=True, tokenize=True)
    want = list(want["input_ids"] if hasattr(want, "keys") else want)
    ok.append(tok.apply_chat_template(m, add_generation_prompt=True) == want)
  r = {"template_equal": f"{sum(ok)}/{len(ok)}", "stop_ids": sorted(tok.stop_token_ids), "bos": tok.bos_token_id,
       "decode_roundtrip": tok.decode(tok.encode("Hello, world! 123")) == "Hello, world! 123"}
  print("template:", r, flush=True)
  return r

def _hf_greedy(hfm, ids: list[int], n: int) -> tuple[list[int], np.ndarray]:
  import torch
  out, past, inp, first = [], None, torch.tensor([ids]), None
  with torch.no_grad():
    for _ in range(n):
      o = hfm(inp, past_key_values=past, use_cache=True)
      if first is None: first = o.logits[0, -1].float().numpy()
      past, t = o.past_key_values, int(o.logits[0, -1].argmax())
      out.append(t); inp = torch.tensor([[t]])
  return out, first

def part_trunc(layers: int, tokens: int, n: int) -> dict:
  import torch, transformers as tr
  from fusezap.dtype import dtypes
  from fusezap.serve.models.llama import Llama
  from fusezap.serve.sampling import Sampler
  p = path()
  ps = prompts(n)
  t0 = time.time()
  hfm = tr.AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32, num_hidden_layers=layers).eval()
  ref = [_hf_greedy(hfm, ids, tokens) for ids in ps]
  del hfm
  r: dict = {"layers": layers, "tokens": tokens, "prompts": n, "hf_s": round(time.time() - t0, 1)}
  for dev in ("CPU", "METAL"):
    t0 = time.time()
    m = Llama.from_pretrained(p, dtype=dtypes.float32, device=dev, max_len=256, prefill_chunk=128, load_layers=layers)
    same, dmax = [], 0.0
    for ids, (want, lg) in zip(ps, ref):
      got = list(m.generate(ids, tokens, Sampler(), stop=[]))
      same.append(next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), len(want)))
      m.reset(); dmax = max(dmax, float(np.abs(m.prefill([ids])[0] - lg).max()))
    r[dev] = {"greedy_prefix": same, "all_equal": all(s == tokens for s in same), "max_abs_dlogit_first": dmax, "s": round(time.time() - t0, 1)}
    print(f"trunc {layers} layers {dev}:", r[dev], flush=True)
    del m
  return r

def part_full_fz(device: str, dtype: str, tokens: int, n: int, out: str) -> dict:
  from fusezap.dtype import dtypes
  from fusezap.serve.models.llama import Llama
  t0 = time.time()
  m = Llama.from_pretrained(path(), dtype=getattr(dtypes, dtype), device=device, max_len=512, prefill_chunk=256)
  tl = time.time() - t0
  from fusezap.serve.tokenizer import Tokenizer
  tok = Tokenizer(path())
  rows = []
  t0 = time.time()
  for ids in prompts(n):
    # prompt one token at a time (decode steps): on METAL a bf16 prefill forward (library GEMM) allocated ~28 GB of
    # temporaries next to the 13.6 GB of weights (2x the weights; not investigated), which a 48 GB Mac cannot spare
    m.reset()
    lg = m.prefill([ids], chunk=1) if device == "METAL" and dtype != "float32" else m.prefill([ids])
    got = []
    for _ in range(tokens):
      got.append(int(np.argmax(lg[0])))
      lg = m.decode([[got[-1]]])
    rows.append({"prompt": ids, "tokens": got, "text": tok.decode(got)})
    print(f"full fz {device} {dtype}:", repr(rows[-1]["text"][:160]), flush=True)
  r = {"device": device, "dtype": dtype, "load_s": round(tl, 1), "gen_s": round(time.time() - t0, 1), "rows": rows}
  with open(out, "w") as f: json.dump(r, f)
  return r

def part_full_hf(fz: str) -> dict:
  """HF float32 compute, teacher-forced over each prompt + FuseZap's tokens (module docstring)."""
  import torch, transformers as tr
  r = json.load(open(fz))
  hfm = tr.AutoModelForCausalLM.from_pretrained(path(), dtype=torch.bfloat16).eval()
  for l in hfm.model.layers:   # hooks return None: inputs and outputs unchanged
    l.register_forward_pre_hook(lambda mod, a: (mod.float(), None)[1])
    l.register_forward_hook(lambda mod, a, o: (mod.bfloat16(), None)[1])
  hfm.model.embed_tokens.float(); hfm.model.norm.float(); hfm.lm_head.float(); hfm.model.rotary_emb.float()
  agree, top1, n_all, first_miss = [], 0, 0, []
  t0 = time.time()
  with torch.no_grad():
    for row in r["rows"]:
      ids, got = row["prompt"], row["tokens"]
      lg = hfm(torch.tensor([ids + got])).logits[0].float()
      am = lg[len(ids) - 1:len(ids) - 1 + len(got)].argmax(-1).tolist()
      hit = [a == b for a, b in zip(am, got)]
      agree.append(sum(hit)); n_all += len(hit)
      first_miss.append(next((i for i, h in enumerate(hit) if not h), len(hit)))
  res = {"fz": {k: r[k] for k in ("device", "dtype", "load_s", "gen_s")}, "positions": n_all,
         "hf_argmax_equals_fz_token": round(sum(agree) / max(n_all, 1), 5), "greedy_prefix_per_prompt": first_miss,
         "greedy_exact_prompts": f"{sum(1 for x, row in zip(first_miss, r['rows']) if x == len(row['tokens']))}/{len(first_miss)}",
         "hf_s": round(time.time() - t0, 1), "texts": [row["text"][:200] for row in r["rows"]]}
  print("full hf:", {k: v for k, v in res.items() if k != "texts"}, flush=True)
  return res

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--part", default="template,trunc,full")
  ap.add_argument("--layers", type=int, default=4)
  ap.add_argument("--tokens", type=int, default=32)
  ap.add_argument("--prompts", type=int, default=8)
  ap.add_argument("--device", default="METAL")
  ap.add_argument("--dtype", default="bfloat16")
  ap.add_argument("--fz-json", default="")
  ap.add_argument("--json", default="")
  a = ap.parse_args()
  res: dict = {"model": MODEL, "revision": REV}
  parts = a.part.split(",")
  if "template" in parts: res["template"] = part_template()
  if "trunc" in parts: res["trunc"] = part_trunc(a.layers, a.tokens, min(a.prompts, 4))
  if "full-fz" in parts: res["full_fz"] = {k: v for k, v in part_full_fz(a.device, a.dtype, a.tokens, a.prompts, a.fz_json).items() if k != "rows"}
  if "full-hf" in parts: res["full"] = part_full_hf(a.fz_json)
  if "full" in parts:   # two child processes: FuseZap, then HF
    fzj = (a.json or "olmoe_mac.json") + f".fz_{a.device}_{a.dtype}.json"
    base = [sys.executable, __file__, "--tokens", str(a.tokens), "--prompts", str(a.prompts), "--device", a.device, "--dtype", a.dtype, "--fz-json", fzj]
    subprocess.run(base + ["--part", "full-fz"], check=True)
    out = fzj + ".hf.json"
    subprocess.run(base + ["--part", "full-hf", "--json", out], check=True)
    res["full"] = json.load(open(out))["full"]
  if a.json:
    with open(a.json, "w") as f: json.dump(res, f, indent=1)

if __name__ == "__main__": main()
