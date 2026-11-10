"""`fusezap generate --model black-forest-labs/FLUX.2-klein-4B --prompt "..." [--seed 0] [--size 512] [--steps 4] [--out x.png]`
(also `python -m fusezap generate ...`): text to image with FLUX.2 [klein] (I81 D6).

Components load one at a time so peak memory is one component (the 48 GB Mac rule): the prompt encoder runs first (the
interim torch encoder in a child process, so its memory is really returned), then the transformer and the VAE load. --embeds FILE (.npy or an .npz with an `embeds` array, (512, 7680) or
(1, 512, 7680)) skips the encoder: that is how the correctness gates inject reference embeddings.

The encoder is a seam (fusezap.serve.models.flux2_klein: prompts -> (B, 512, 7680) embeddings). ENCODERS maps --encoder
names to loaders (path, device, dtype) -> encoder: "fusezap" (Qwen3TextEncoder, I81 D3, on --device in --encoder-dtype; the
"auto" choice) and "hf" (transformers' Qwen3 on the CPU, float32; the fallback when the fusezap one cannot load).
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from typing import Callable
import numpy as np

def _hf_encoder(model: str, device: str | None = None, dtype: str = "float32"):
  from fusezap.serve.models.flux2_klein import HFQwen3PromptEncoder
  return HFQwen3PromptEncoder(model)   # transformers on the CPU in float32 whatever --device / --dtype

def _fusezap_encoder(model: str, device: str | None = None, dtype: str = "float32"):
  from fusezap.dtype import dtypes
  from fusezap.serve.models.text_encoder import Qwen3TextEncoder
  return Qwen3TextEncoder.from_pretrained(model, getattr(dtypes, dtype), device)

ENCODERS: dict[str, Callable[..., Callable]] = {"fusezap": _fusezap_encoder, "hf": _hf_encoder}
AUTO_ORDER = ("fusezap", "hf")
CHILD_PROCESS = {"hf"}   # encoders run in a spawned child: torch keeps freed host memory, the OS gets it back at exit

def _encode(name: str, path: str, prompts: list[str], device: str | None = None, dtype: str = "float32") -> np.ndarray:
  enc = ENCODERS[name](path, device, dtype)
  out = np.asarray(enc(prompts), np.float32)
  if hasattr(enc, "release"): enc.release()
  del enc
  if device is not None:
    from fusezap.serve.models.flux2_vae import trim_cache
    gc.collect(); trim_cache(device)   # the encoder's device blocks go back before the transformer loads
  return out

def encode_prompts(name: str, path: str, prompts: list[str], child: bool, device: str | None = None, dtype: str = "float32") -> np.ndarray:
  if not child: return _encode(name, path, prompts, device, dtype)
  import multiprocessing as mp
  with mp.get_context("spawn").Pool(1) as pool: return pool.apply(_encode, (name, path, prompts, device, dtype))

def build_parser() -> argparse.ArgumentParser:
  ap = argparse.ArgumentParser(prog="fusezap generate", description="FuseZap: text to image (FLUX.2 [klein])")
  ap.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B", help="Hugging Face repo id or local diffusers dir")
  ap.add_argument("--prompt", default=None, help="the text prompt (or --embeds)")
  ap.add_argument("--embeds", default=None, help="precomputed prompt embeddings (.npy, or .npz with 'embeds'): skips the encoder")
  ap.add_argument("--seed", type=int, default=0, help="initial noise seed (numpy PCG64, the same noise the diffusers reference uses)")
  ap.add_argument("--size", default="512", help="N or WxH (multiples of 16), default 512")
  ap.add_argument("--steps", type=int, default=4, help="denoising steps (klein is distilled for 4)")
  ap.add_argument("--device", default=None, help="METAL, CPU, ... (default: the best available)")
  ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
  ap.add_argument("--encoder-dtype", default="float32", choices=["float32", "bfloat16"],
                  help="fusezap text encoder dtype (default float32: bf16 Qwen3 embeddings cost ~2 dB PSNR end to end)")
  ap.add_argument("--vae-dtype", default="float32", choices=["float32", "bfloat16"],
                  help="VAE dtype (default float32: the conv kernel is float32-only, a bfloat16 decode needs ~4x the memory)")
  ap.add_argument("--encoder", default="auto", help=f"prompt encoder: auto or one of {sorted(ENCODERS)}")
  ap.add_argument("--encoder-inproc", action="store_true", help="run the prompt encoder in this process (default for "
                  f"{sorted(CHILD_PROCESS)}: a child process, so its memory is returned before the transformer loads)")
  ap.add_argument("--out", default=None, help="output PNG (default: fusezap-<seed>.png)")
  ap.add_argument("--save-latents", default=None, help="also write the final packed latents (.npy)")
  return ap

def parse_size(s: str) -> tuple[int, int]:
  """'512' -> (512, 512); 'WxH' -> (H, W)."""
  if "x" in s.lower():
    w, h = s.lower().split("x")
    return int(h), int(w)
  return int(s), int(s)

def load_embeds(fn: str) -> np.ndarray:
  a = np.load(fn)
  e = a["embeds"] if isinstance(a, np.lib.npyio.NpzFile) else a
  e = np.asarray(e, np.float32)
  return e[None] if e.ndim == 2 else e

def pick_encoder(name: str) -> str:
  if name != "auto":
    if name not in ENCODERS: raise SystemExit(f"fusezap generate: unknown --encoder {name!r} (have {sorted(ENCODERS)})")
    return name
  return next(n for n in AUTO_ORDER if n in ENCODERS)

def main(argv: list[str] | None = None) -> int:
  a = build_parser().parse_args(argv)
  if (a.prompt is None) == (a.embeds is None): raise SystemExit("fusezap generate: give exactly one of --prompt or --embeds")
  from fusezap.dtype import dtypes
  from fusezap.device import Device
  from fusezap.serve.weights import resolve_model
  from fusezap.serve.models.flux2 import Flux2Transformer
  from fusezap.serve.models.flux2_vae import Flux2VAEDecoder, Flux2VAEConfig, to_uint8, trim_cache
  from fusezap.serve.models.flux2_klein import Flux2KleinPipeline, write_png
  H, W = parse_size(a.size)
  path = resolve_model(a.model)
  dev = Device.canonicalize(a.device)
  t0 = time.perf_counter()
  if a.embeds is not None: emb = load_embeds(a.embeds)
  else:
    name = pick_encoder(a.encoder)
    try: emb = encode_prompts(name, path, [a.prompt], name in CHILD_PROCESS and not a.encoder_inproc, dev, a.encoder_dtype)
    except (ImportError, OSError, NotImplementedError) as e:   # auto only: fall back to the next encoder (transformers)
      rest = AUTO_ORDER[AUTO_ORDER.index(name) + 1:] if a.encoder == "auto" and name in AUTO_ORDER else ()
      if not (rest := [n for n in rest if n in ENCODERS]): raise
      print(f"fusezap: the {name} encoder failed ({type(e).__name__}: {e}); falling back to {rest[0]}", flush=True)
      name = rest[0]
      emb = encode_prompts(name, path, [a.prompt], name in CHILD_PROCESS and not a.encoder_inproc, dev, a.encoder_dtype)
    print(f"fusezap: encoded the prompt ({name} encoder) in {time.perf_counter() - t0:.1f}s", flush=True)
  dt = getattr(dtypes, a.dtype)
  t1 = time.perf_counter()
  with open(os.path.join(path, "vae", "config.json")) as f: vcfg = Flux2VAEConfig.from_hf(json.load(f))
  pipe = Flux2KleinPipeline(Flux2Transformer.from_pretrained(path, dt, dev), None, vae_scale=2 ** (len(vcfg.block_out_channels) - 1))
  print(f"fusezap: loaded the {a.model} transformer on {dev} ({a.dtype}) in {time.perf_counter() - t1:.1f}s", flush=True)
  def cb(k: int, _x) -> None: print(f"fusezap: step {k + 1}/{a.steps} ({time.perf_counter() - t2:.1f}s)", flush=True)
  t2 = time.perf_counter()
  x, _, (h, w) = pipe.denoise_embeds(emb, size=(H, W), steps=a.steps, seed=a.seed, callback=cb)
  t3 = time.perf_counter()
  pipe.transformer = None   # one component at a time: the transformer's weights go before the VAE loads
  gc.collect(); trim_cache(dev)
  pipe.vae = Flux2VAEDecoder.from_pretrained(path, getattr(dtypes, a.vae_dtype), dev)
  t4 = time.perf_counter()
  img = pipe.decode_latents(x, h, w)
  t5 = time.perf_counter()
  fn = a.out or f"fusezap-{a.seed}.png"
  write_png(fn, to_uint8(img[0]))
  if a.save_latents: np.save(a.save_latents, x)
  print(f"fusezap: wrote {fn} ({W}x{H}, seed {a.seed}, {a.steps} steps): denoise {t3 - t2:.1f}s, decode {t5 - t4:.1f}s", flush=True)
  return 0

if __name__ == "__main__": sys.exit(main())
