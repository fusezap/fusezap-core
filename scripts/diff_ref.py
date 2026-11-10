"""I81 D1: diffusers reference tensors for FLUX.2 [klein] 4B, and the noise floors the G1-G6 gates are set from.

The reference is diffusers + transformers in float32 on the CPU (I82 §4). Each component runs alone, one process
stage at a time, and writes its outputs to disk, so no two big components share RAM (the 48 GB Mac rule):

  python scripts/diff_ref.py encode   [--run cpu_f32]            Qwen3 encoder -> prompt embeds (512, 7680) per prompt
  python scripts/diff_ref.py denoise  [--run cpu_f32] [--sizes 512,1024] [--prompts 0-7] [--seeds 0,1]
  python scripts/diff_ref.py decode   [--run cpu_f32] [...]       VAE decode of the final latents -> [-1,1] image + PNG
  python scripts/diff_ref.py pipeline [--size 512 --prompt 0 --seed 0]
                                       the stock Flux2KleinPipeline(prompt_embeds=, latents=) on the same inputs: checks that
                                       the staged loop here IS diffusers' pipeline (bit-identical image)
  python scripts/diff_ref.py calibrate                            noise floors -> docs/invarix/diff_gates.json

`--run` names a (device, dtype) pair: cpu_f32 (THE reference), mps_f32 (the f32 noise floor: same math, another
backend), mps_bf16 (diffusers' own bf16-vs-f32 distance, the "reduced" class). A non-reference run reads the reference
run's inputs so each component is measured in isolation:
  encode   own embeds (G2 floor)
  denoise  teacher-forced velocities on the reference x_t with the reference embeds (G3 floor); a free-running
           trajectory from the reference embeds (G4 floor); and an end-to-end trajectory from its own embeds
  decode   VAE(reference final latent) (G5 floor) and VAE(own end-to-end final latent) -> uint8 image (G6 floor)

Nondeterminism is removed as I82 §4 says: the initial noise is numpy (`noise(seed, size)`, also what `fusezap generate
--seed` uses), the sigmas are passed explicitly (np.linspace(1, 1/steps, steps), the pipeline default) with the
pipeline's own mu, and the padded token ids / attention mask come from the stock tokenizer + chat template.

Outputs (gitignored, results/diff/<run>/): embeds/p{i}.npz, denoise/{size}_p{i}_s{seed}.npz, decode/{size}_p{i}_s{seed}.npz
and .png. Model revision and package versions go into results/diff/<run>/meta.json.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
import numpy as np

REPO = "black-forest-labs/FLUX.2-klein-4B"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "diff")
GATES = os.path.join(ROOT, "docs", "invarix", "diff_gates.json")
STEPS = 4

PROMPTS = [
  "A red fox sitting in fresh snow at sunrise, soft golden light, shallow depth of field",
  "A bowl of ramen on a wooden table, steam rising, top-down photo",
  "An astronaut riding a horse on the surface of the moon, Earth in the sky",
  "A watercolor painting of a lighthouse on a rocky coast during a storm",
  "A close-up photo of a hummingbird drinking from a purple flower",
  "A cozy reading nook with a window seat, rain outside, warm lamp light",
  "A futuristic city skyline at night with flying cars and neon signs",
  "A hand-drawn map of a fantasy island with mountains, rivers and a castle",
]

def noise(seed: int, size: int | tuple[int, int]) -> np.ndarray:
  """The initial latent noise (1, 128, H/16, W/16) float32 for a size (H, W): numpy PCG64, so diffusers and fusezap get
  the same bits whatever their RNGs."""
  h, w = (size, size) if isinstance(size, int) else size
  return np.random.default_rng(seed).standard_normal((1, 128, h // 16, w // 16), dtype=np.float32)

def rel_l2(a, b) -> float:
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))

def max_abs(a, b) -> float: return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))

def to_uint8(img: np.ndarray) -> np.ndarray:
  """(3, H, W) in [-1, 1] -> (H, W, 3) uint8, as diffusers' image processor: (x/2+0.5).clamp(0,1)*255, round."""
  x = np.clip(np.asarray(img, np.float32) * 0.5 + 0.5, 0, 1)
  return np.round(x.transpose(1, 2, 0) * 255).astype(np.uint8)

def psnr(a: np.ndarray, b: np.ndarray) -> float:
  mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
  return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

def pix_within(a: np.ndarray, b: np.ndarray, lv: int = 1) -> float:
  return float(np.mean(np.abs(a.astype(np.int32) - b.astype(np.int32)) <= lv))

def write_png(path: str, rgb: np.ndarray) -> None:
  from PIL import Image
  Image.fromarray(rgb).save(path)

# ---------------------------------------------------------------- run helpers

def model_path() -> str:
  from huggingface_hub import snapshot_download
  return snapshot_download(REPO, local_files_only=True)

def run_spec(run: str):
  import torch
  dev, dt = run.split("_")
  return dev, {"f32": torch.float32, "bf16": torch.bfloat16}[dt]

def rdir(run: str, *sub: str) -> str:
  d = os.path.join(OUT, run, *sub)
  os.makedirs(d, exist_ok=True)
  return d

def write_meta(run: str, stage: str, extra: dict | None = None) -> None:
  import torch, diffusers, transformers
  fn = os.path.join(rdir(run), "meta.json")
  meta = json.load(open(fn)) if os.path.exists(fn) else {}
  meta.update({"repo": REPO, "revision": os.path.basename(model_path()), "torch": torch.__version__,
               "diffusers": diffusers.__version__, "transformers": transformers.__version__, "numpy": np.__version__})
  meta.setdefault("stages", {})[stage] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})}
  json.dump(meta, open(fn, "w"), indent=1)

def parse_list(s: str) -> list[int]:
  out = []
  for part in s.split(","):
    if "-" in part:
      a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
    elif part: out.append(int(part))
  return out

def cases(a) -> list[tuple[int, int, int]]:
  return [(sz, p, s) for sz in parse_list(a.sizes) for p in parse_list(a.prompts) for s in parse_list(a.seeds)]

def sync(dev: str) -> None:
  import torch
  if dev == "mps": torch.mps.synchronize()

# ---------------------------------------------------------------- stages

def stage_encode(a) -> None:
  import torch
  from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
  from diffusers import Flux2KleinPipeline
  dev, dt = run_spec(a.run)
  path = model_path()
  tok = Qwen2TokenizerFast.from_pretrained(path, subfolder="tokenizer")
  te = Qwen3ForCausalLM.from_pretrained(path, subfolder="text_encoder", torch_dtype=dt).to(dev).eval()
  d = rdir(a.run, "embeds")
  for i in parse_list(a.prompts):
    t0 = time.perf_counter()
    with torch.no_grad():
      e = Flux2KleinPipeline._get_qwen3_prompt_embeds(te, tok, PROMPTS[i], dtype=dt, device=dev)
    sync(dev)
    msgs = [{"role": "user", "content": PROMPTS[i]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, padding="max_length", truncation=True, max_length=512)
    np.savez(os.path.join(d, f"p{i}.npz"), embeds=e[0].float().cpu().numpy(), input_ids=np.array(enc["input_ids"], np.int32),
             attention_mask=np.array(enc["attention_mask"], np.int32), prompt=PROMPTS[i])
    print(f"encode {a.run} p{i}: {tuple(e.shape)} {time.perf_counter() - t0:.1f}s", flush=True)
  write_meta(a.run, "encode", {"prompts": parse_list(a.prompts)})

def load_embeds(run: str, i: int) -> np.ndarray: return np.load(os.path.join(OUT, run, "embeds", f"p{i}.npz"))["embeds"]

class Denoiser:
  """The klein pipeline's denoising loop, staged (I82 §4): the same calls Flux2KleinPipeline.__call__ makes."""
  def __init__(self, dev: str, dt):
    import torch
    from diffusers import Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
    path = model_path()
    self.dev, self.dt = dev, dt
    self.tr = Flux2Transformer2DModel.from_pretrained(path, subfolder="transformer", torch_dtype=dt).to(dev).eval()
    self.sched_cfg = path
    self.torch = torch
  def schedule(self, L: int, steps: int = STEPS):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(self.sched_cfg, subfolder="scheduler")
    sigmas = np.linspace(1.0, 1 / steps, steps)
    mu = compute_empirical_mu(image_seq_len=L, num_steps=steps)
    ts, _ = retrieve_timesteps(sch, steps, self.dev, sigmas=sigmas, mu=mu)
    sch.set_begin_index(0)
    return sch, ts, mu
  def velocity(self, x, embeds, t, txt_ids, img_ids):
    torch = self.torch
    timestep = t.expand(x.shape[0]).to(x.dtype)
    with torch.no_grad():
      return self.tr(hidden_states=x.to(self.tr.dtype), timestep=timestep / 1000, guidance=None, encoder_hidden_states=embeds,
                     txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0]
  def run(self, nz: np.ndarray, emb: np.ndarray, teacher_x: np.ndarray | None = None) -> dict:
    """Free-running from noise nz (1,128,h,w) with prompt embeds emb (512, 7680). teacher_x (steps+1, L, 128): also
    return the velocity at each reference x_t (teacher forcing)."""
    torch = self.torch
    from diffusers import Flux2KleinPipeline as P
    lat = torch.from_numpy(nz).to(self.dev, self.dt)
    img_ids = P._prepare_latent_ids(lat).to(self.dev)
    x = P._pack_latents(lat)
    e = torch.from_numpy(emb)[None].to(self.dev, self.dt)
    txt_ids = P._prepare_text_ids(e).to(self.dev)
    sch, ts, mu = self.schedule(x.shape[1])
    xs, vs, vtf = [x.float().cpu().numpy()[0]], [], []
    for k, t in enumerate(ts):
      v = self.velocity(x, e, t, txt_ids, img_ids)
      if teacher_x is not None:
        xt = torch.from_numpy(teacher_x[k])[None].to(self.dev, self.dt)
        vtf.append(self.velocity(xt, e, t, txt_ids, img_ids).float().cpu().numpy()[0])
      x = sch.step(v, t, x, return_dict=False)[0]
      vs.append(v.float().cpu().numpy()[0]); xs.append(x.float().cpu().numpy()[0])
    sync(self.dev)
    out = {"x": np.stack(xs), "v": np.stack(vs), "sigmas": sch.sigmas.cpu().numpy(), "timesteps": ts.float().cpu().numpy(),
           "mu": np.float64(mu), "img_ids": img_ids[0].cpu().numpy(), "txt_ids": txt_ids[0].cpu().numpy()}
    if teacher_x is not None: out["v_tf"] = np.stack(vtf)
    return out

def stage_denoise(a) -> None:
  dev, dt = run_spec(a.run)
  den = Denoiser(dev, dt)
  d = rdir(a.run, "denoise")
  ref = a.run == a.ref
  for size, p, s in cases(a):
    fn = os.path.join(d, f"{size}_p{p}_s{s}.npz")
    if os.path.exists(fn) and not a.force: continue
    t0 = time.perf_counter()
    nz = noise(s, size)
    if ref:
      r = den.run(nz, load_embeds(a.ref, p))
      np.savez(fn, noise=nz, **r)
    else:
      rr = np.load(os.path.join(OUT, a.ref, "denoise", f"{size}_p{p}_s{s}.npz"))
      r = den.run(nz, load_embeds(a.ref, p), teacher_x=rr["x"])
      e2e = den.run(nz, load_embeds(a.run, p)) if os.path.exists(os.path.join(OUT, a.run, "embeds", f"p{p}.npz")) else None
      np.savez(fn, noise=nz, **r, **({"x_e2e": e2e["x"]} if e2e else {}))
    print(f"denoise {a.run} {size} p{p} s{s}: {time.perf_counter() - t0:.1f}s", flush=True)
  write_meta(a.run, "denoise", {"cases": [list(c) for c in cases(a)]})

class Decoder:
  def __init__(self, dev: str, dt):
    import torch
    from diffusers import AutoencoderKLFlux2
    self.torch, self.dev, self.dt = torch, dev, dt
    self.vae = AutoencoderKLFlux2.from_pretrained(model_path(), subfolder="vae", torch_dtype=dt).to(dev).eval()
  def __call__(self, x_packed: np.ndarray, h: int, w: int) -> np.ndarray:
    """The pipeline's tail: unpack (L, 128) with the ids, BN denormalize, unpatchify, decode -> (3, H, W) float32."""
    torch = self.torch
    from diffusers import Flux2KleinPipeline as P
    x = torch.from_numpy(np.ascontiguousarray(x_packed))[None].to(self.dev, self.dt)
    ids = P._prepare_latent_ids(torch.zeros(1, 1, h, w))
    lat = P._unpack_latents_with_ids(x, ids.to(self.dev), h, w)
    bn = self.vae.bn
    mean = bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
    std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(lat.device, lat.dtype)
    lat = P._unpatchify_latents(lat * std + mean)
    with torch.no_grad(): img = self.vae.decode(lat, return_dict=False)[0]
    sync(self.dev)
    return img[0].float().cpu().numpy()

def stage_decode(a) -> None:
  dev, dt = run_spec(a.run)
  dec = Decoder(dev, dt)
  d = rdir(a.run, "decode")
  for size, p, s in cases(a):
    fn = os.path.join(d, f"{size}_p{p}_s{s}.npz")
    if os.path.exists(fn) and not a.force: continue
    t0 = time.perf_counter()
    h = w = size // 16
    rr = np.load(os.path.join(OUT, a.ref, "denoise", f"{size}_p{p}_s{s}.npz"))
    img_ref_lat = dec(rr["x"][-1], h, w)   # VAE of the reference final latent (G5)
    out = {"image": img_ref_lat}
    if a.run != a.ref:
      own = np.load(os.path.join(OUT, a.run, "denoise", f"{size}_p{p}_s{s}.npz"))
      if "x_e2e" in own: out["image_e2e"] = dec(own["x_e2e"][-1], h, w)
    np.savez(fn, **out)
    write_png(fn[:-4] + ".png", to_uint8(out.get("image_e2e", out["image"])))
    print(f"decode {a.run} {size} p{p} s{s}: {time.perf_counter() - t0:.1f}s", flush=True)
  write_meta(a.run, "decode", {"cases": [list(c) for c in cases(a)]})

def stage_pipeline(a) -> None:
  """Stock Flux2KleinPipeline (no text encoder: prompt_embeds injected) vs the staged reference image, bit for bit."""
  import torch
  from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
  from transformers import Qwen2TokenizerFast
  path = model_path()
  pipe = Flux2KleinPipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler"),
    vae=AutoencoderKLFlux2.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32), text_encoder=None,
    tokenizer=Qwen2TokenizerFast.from_pretrained(path, subfolder="tokenizer"),
    transformer=Flux2Transformer2DModel.from_pretrained(path, subfolder="transformer", torch_dtype=torch.float32), is_distilled=True)
  size, p, s = a.size, a.prompt, a.seed
  emb = torch.from_numpy(load_embeds(a.ref, p))[None]
  with torch.no_grad():
    img = pipe(prompt_embeds=emb, latents=torch.from_numpy(noise(s, size)), height=size, width=size, num_inference_steps=STEPS,
               output_type="np").images[0]
  pipe_u8 = np.round(img * 255).astype(np.uint8)
  ref = np.load(os.path.join(OUT, a.ref, "decode", f"{size}_p{p}_s{s}.npz"))["image"]
  staged = to_uint8(ref)
  ok = bool(np.array_equal(pipe_u8, staged))
  res = {"case": [size, p, s], "identical_uint8": ok, "max_level_diff": int(np.abs(pipe_u8.astype(int) - staged.astype(int)).max())}
  print(json.dumps(res), flush=True)
  write_png(os.path.join(rdir(a.ref, "decode"), f"{size}_p{p}_s{s}_stock_pipeline.png"), pipe_u8)
  write_meta(a.ref, "pipeline_check", res)
  if not ok: sys.exit(1)

# ---------------------------------------------------------------- calibration

def floors(run: str, ref: str = "cpu_f32") -> dict:
  """Distances of `run` from the reference, per gate (I82 §4 table), over every case both runs have."""
  out: dict = {"G2_rel_l2": [], "G3_rel_l2": [], "G4_rel_l2": [], "G5_max_abs": [], "G6_psnr": [], "G6_within1": [], "cases": []}
  ed = os.path.join(OUT, run, "embeds")
  for fn in sorted(os.listdir(ed)) if os.path.isdir(ed) else []:
    out["G2_rel_l2"].append(rel_l2(np.load(os.path.join(ed, fn))["embeds"], np.load(os.path.join(OUT, ref, "embeds", fn))["embeds"]))
  dd = os.path.join(OUT, run, "denoise")
  for fn in sorted(os.listdir(dd)) if os.path.isdir(dd) else []:
    o, r = np.load(os.path.join(dd, fn)), np.load(os.path.join(OUT, ref, "denoise", fn))
    out["cases"].append(fn[:-4])
    out["G3_rel_l2"].append([rel_l2(o["v_tf"][k], r["v"][k]) for k in range(len(r["v"]))])
    out["G4_rel_l2"].append(rel_l2(o["x"][-1], r["x"][-1]))
    dfn = os.path.join(OUT, run, "decode", fn)
    if os.path.exists(dfn):
      od, rd = np.load(dfn), np.load(os.path.join(OUT, ref, "decode", fn))
      out["G5_max_abs"].append(max_abs(od["image"], rd["image"]))
      if "image_e2e" in od:
        a, b = to_uint8(od["image_e2e"]), to_uint8(rd["image"])
        out["G6_psnr"].append(psnr(a, b)); out["G6_within1"].append(pix_within(a, b))
  return out

def _up(x: float, k: float = 10.0) -> float:
  """k * x rounded up to one significant digit (a threshold a margin above a measured floor)."""
  if x <= 0: return 0.0
  y = k * x
  e = 10 ** np.floor(np.log10(y))
  return float(np.ceil(y / e) * e)

def stage_calibrate(a) -> None:
  f32, bf16 = floors("mps_f32"), floors("mps_bf16")
  mx = lambda v: max((max(x) if isinstance(x, list) else x) for x in v) if v else None
  mn = lambda v: min(v) if v else None
  g = {
    "_doc": "I81 D1. Noise floors: diffusers on MPS vs diffusers f32 on CPU (the reference), same injected noise/sigmas. "
            "f32 thresholds: the I82 §4 proposal, loosened only where 10x the measured MPS-f32 floor (rounded up to one "
            "digit; G6: the floor PSNR - 3 dB) is looser than it. bf16 gates are relative: FuseZap-bf16 distance <= "
            "factor x diffusers-bf16 distance (G6: PSNR >= diffusers-bf16 PSNR - 1 dB). Written by scripts/diff_ref.py calibrate.",
    "date": time.strftime("%Y-%m-%d"),
    "floors": {"mps_f32": f32, "mps_bf16": bf16},
    "proposed_f32": {"G1_max_rel": 1e-4, "G2_rel_l2": 1e-5, "G3_rel_l2": 1e-4, "G4_rel_l2": 1e-3, "G5_max_abs": 1e-4,
                     "G6_within1": 0.999, "G6_psnr_db": 50.0},
  }
  thr = {"G1_max_rel": 1e-4}
  for k, prop in (("G2_rel_l2", 1e-5), ("G3_rel_l2", 1e-4), ("G4_rel_l2", 1e-3), ("G5_max_abs", 1e-4)):
    fl = mx(f32[k])
    thr[k] = max(prop, _up(fl)) if fl is not None else prop
  p = mn(f32["G6_psnr"]); w = mn(f32["G6_within1"])   # loosen the proposals only if the measured floor needs it
  thr["G6_psnr_db"] = min(50.0, float(np.floor(p - 3))) if p is not None and np.isfinite(p) else 50.0
  # G6 within +-1 stays at the proposal 0.999 (decided 2026-11-04: FuseZap METAL f32 measured 0.9999975 worst at 512^2);
  # the 10x-floor value is recorded alongside, not used
  thr["G6_within1"] = 0.999
  thr["G6_within1_calibrated"] = min(0.999, float(np.floor((1 - 10 * (1 - w)) * 1e4) / 1e4)) if w is not None else 0.999
  thr["_note_G6_within1"] = ("Kept at the I82 §4 proposal 0.999 (decided 2026-11-04), not the calibrated value (10x the MPS-f32 "
    "floor): FuseZap METAL f32 measured 0.9999975 worst case at 512^2 (16 cases), so the stricter bar holds with margin and "
    "catches regressions the loose one would not.")
  g["thresholds_f32"] = thr
  g["thresholds_bf16_relative"] = {"factor": 1.5, "G6_psnr_margin_db": 1.0,
    "diffusers_bf16": {k: mx(bf16[k]) for k in ("G2_rel_l2", "G3_rel_l2", "G4_rel_l2", "G5_max_abs")} |
                      {"G6_psnr_db": mn(bf16["G6_psnr"])}}
  os.makedirs(os.path.dirname(GATES), exist_ok=True)
  json.dump(g, open(GATES, "w"), indent=1)
  print(json.dumps({"thresholds_f32": thr, "bf16": g["thresholds_bf16_relative"]}, indent=1))

def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("stage", choices=["encode", "denoise", "decode", "pipeline", "calibrate"])
  ap.add_argument("--run", default="cpu_f32")
  ap.add_argument("--ref", default="cpu_f32")
  ap.add_argument("--sizes", default="512")
  ap.add_argument("--prompts", default="0-7")
  ap.add_argument("--seeds", default="0,1")
  ap.add_argument("--size", type=int, default=512)
  ap.add_argument("--prompt", type=int, default=0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--force", action="store_true")
  a = ap.parse_args(argv)
  import torch
  torch.manual_seed(0)
  {"encode": stage_encode, "denoise": stage_denoise, "decode": stage_decode, "pipeline": stage_pipeline,
   "calibrate": stage_calibrate}[a.stage](a)
  gc.collect()
  return 0

if __name__ == "__main__": sys.exit(main())
