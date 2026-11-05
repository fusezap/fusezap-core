"""I81 D3 gate G2 (docs/I82_DIFFUSION_SCOPE.md §4): the FLUX.2 [klein] 4B text encoder (Qwen3-4B, hidden states 9/18/27)
in FuseZap vs HF transformers float32 on torch CPU, the way diffusers 0.42's Flux2KleinPipeline calls it.

  uv run python scripts/diff_text_encoder_gate.py [--out results/diff/te_gate] [--device METAL] [--dtypes f32,bf16]

Each phase runs in its own subprocess, so only one 4B copy is in memory at a time (the f32 reference ~13 GB, FuseZap f32
~12 GB on METAL):
  ref-f32   transformers Qwen3ForCausalLM, float32, CPU, with the pipeline's chat template / padding / mask
  ref-bf16  the same in bfloat16 on CPU (the "reduced" class distance for the relative bf16 gate)
  fz-f32    fusezap Qwen3TextEncoder, float32, on --device
  fz-bf16   fusezap Qwen3TextEncoder, bfloat16, on --device
The references load 28 of the 36 layers (num_hidden_layers=28): hidden_states[27] is then the residual after layer 26,
not normed (HF norms only the last entry), and no later layer can change it. Token ids from the fusezap tokenizer path
must equal HF's.

Gates (per layer, over all B x 512 rows, pad rows included):
  G2 f32:  rel-L2(fz-f32, ref-f32) <= 1e-5
  G2 bf16: rel-L2(fz-bf16, ref-f32) <= 1.5 * rel-L2(ref-bf16, ref-f32)
Writes <out>/report.json and exits 1 if a gate fails.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
import numpy as np

REPO = "black-forest-labs/FLUX.2-klein-4B"
LAYERS = (9, 18, 27)
PROMPTS = ["A red fox standing in fresh snow at dawn, photorealistic.",
           "a bowl of ramen with a soft-boiled egg, top-down studio photo",
           "An astronaut riding a horse on Mars, oil painting, dramatic lighting, wide shot, highly detailed, "
           "with the earth rising over the horizon and dust kicked up by the hooves",
           "日本の古い寺院、桜の花びらが舞う春の朝"]

def _root() -> str:
  from fusezap.serve.models.text_encoder import resolve_subfolders
  return resolve_subfolders(REPO, ("text_encoder", "tokenizer"), allow_download=True)

def phase_ref(out: str, dt: str) -> None:
  import torch, transformers as tr
  root = _root()
  tok = tr.AutoTokenizer.from_pretrained(os.path.join(root, "tokenizer"))
  ids, masks = [], []
  for p in PROMPTS:   # diffusers 0.42 Flux2KleinPipeline._get_qwen3_prompt_embeds
    text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    ids.append(enc["input_ids"]); masks.append(enc["attention_mask"])
  ids, mask = torch.cat(ids), torch.cat(masks)
  cfg = tr.AutoConfig.from_pretrained(os.path.join(root, "text_encoder"))
  cfg.num_hidden_layers = max(LAYERS) + 1
  if hasattr(cfg, "layer_types") and cfg.layer_types: cfg.layer_types = cfg.layer_types[:cfg.num_hidden_layers]
  torch_dt = torch.float32 if dt == "f32" else torch.bfloat16
  t0 = time.time()
  m = tr.Qwen3ForCausalLM.from_pretrained(os.path.join(root, "text_encoder"), config=cfg, dtype=torch_dt).eval()
  t1 = time.time()
  hs = []
  with torch.no_grad():
    for b in range(len(PROMPTS)):   # one prompt per forward (bounded memory); rows are independent
      o = m(input_ids=ids[b:b + 1], attention_mask=mask[b:b + 1], output_hidden_states=True, use_cache=False)
      hs.append(torch.stack([o.hidden_states[k] for k in LAYERS], 1).float().numpy())
  t2 = time.time()
  np.savez(os.path.join(out, f"ref-{dt}.npz"), ids=ids.numpy(), mask=mask.numpy(), hs=np.concatenate(hs))
  json.dump({"load_s": t1 - t0, "encode_s": t2 - t1, "torch": torch.__version__, "transformers": tr.__version__},
            open(os.path.join(out, f"ref-{dt}.json"), "w"))

def phase_fz(out: str, dt: str, device: str) -> None:
  from fusezap.dtype import dtypes
  from fusezap.serve.models.text_encoder import Qwen3TextEncoder
  t0 = time.time()
  enc = Qwen3TextEncoder.from_pretrained(_root(), dtype=dtypes.float32 if dt == "f32" else dtypes.bfloat16, device=device)
  t1 = time.time()
  ids, mask = enc.tokenize(PROMPTS)
  hs, times = [], []
  for b in range(len(PROMPTS)):
    s = time.time()
    hs.append(enc.model.encode(ids[b:b + 1], mask[b:b + 1], LAYERS))
    times.append(time.time() - s)
  np.savez(os.path.join(out, f"fz-{dt}.npz"), ids=ids, mask=mask, hs=np.concatenate(hs))
  json.dump({"load_s": t1 - t0, "encode_s": times, "device": device}, open(os.path.join(out, f"fz-{dt}.json"), "w"))

def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
  a, b = a.astype(np.float64), b.astype(np.float64)
  return float(np.linalg.norm(a - b) / np.linalg.norm(b))

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="results/diff/te_gate")
  ap.add_argument("--device", default="METAL")
  ap.add_argument("--dtypes", default="f32,bf16")
  ap.add_argument("--phase", default=None, help=argparse.SUPPRESS)
  ap.add_argument("--reuse", action="store_true", help="skip phases whose .npz exists")
  a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True)
  if a.phase:
    kind, dt = a.phase.split("-")
    (phase_ref(a.out, dt) if kind == "ref" else phase_fz(a.out, dt, a.device))
    return
  dts = a.dtypes.split(",")
  phases = ["ref-f32"] + [f"ref-{d}" for d in dts if d != "f32"] + [f"fz-{d}" for d in dts]
  for ph in phases:
    if a.reuse and os.path.exists(os.path.join(a.out, f"{ph}.npz")): continue
    print(f"[{time.strftime('%H:%M:%S')}] {ph} ...", flush=True)
    subprocess.run([sys.executable, __file__, "--out", a.out, "--device", a.device, "--phase", ph], check=True)
  ref = np.load(os.path.join(a.out, "ref-f32.npz"))
  rep: dict = {"repo": REPO, "layers": LAYERS, "prompts": PROMPTS, "device": a.device, "gates": {}, "rel_l2": {}}
  ok = True
  for ph in phases[1:]:
    r = np.load(os.path.join(a.out, f"{ph}.npz"))
    if ph.startswith("fz"):
      same = bool((r["ids"] == ref["ids"]).all() and (r["mask"] == ref["mask"]).all())
      rep[f"{ph}.tokens_match_hf"] = same
      ok &= same
    rep["rel_l2"][ph] = [rel_l2(r["hs"][:, j], ref["hs"][:, j]) for j in range(len(LAYERS))]
    rep["rel_l2"][ph + ".max_abs"] = [float(np.abs(r["hs"][:, j] - ref["hs"][:, j]).max()) for j in range(len(LAYERS))]
  if "fz-f32" in rep["rel_l2"]:
    g = rep["gates"]["G2.f32"] = {"value": rep["rel_l2"]["fz-f32"], "threshold": 1e-5, "pass": max(rep["rel_l2"]["fz-f32"]) <= 1e-5}
    ok &= g["pass"]
  if "fz-bf16" in rep["rel_l2"] and "ref-bf16" in rep["rel_l2"]:
    thr = [1.5 * x for x in rep["rel_l2"]["ref-bf16"]]
    g = rep["gates"]["G2.bf16"] = {"value": rep["rel_l2"]["fz-bf16"], "threshold": thr,
                                   "pass": all(v <= t for v, t in zip(rep["rel_l2"]["fz-bf16"], thr))}
    ok &= g["pass"]
  for ph in phases:
    fn = os.path.join(a.out, f"{ph}.json")
    if os.path.exists(fn): rep[f"{ph}.timing"] = json.load(open(fn))
  rep["pass"] = bool(ok)
  json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
  print(json.dumps({k: rep[k] for k in ("gates", "rel_l2", "pass")}, indent=1))
  sys.exit(0 if ok else 1)

if __name__ == "__main__": main()
