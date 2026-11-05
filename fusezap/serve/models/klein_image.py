"""FLUX.2 [klein] as an ImageEngine DiffusionPipeline (fusezap/serve/image_engine.py), I81 D7 x D6.

  pipe = KleinImagePipeline("black-forest-labs/FLUX.2-klein-4B", device="METAL", dtype="bfloat16")
  fusezap serve-image flux2-klein --device METAL --dtype bfloat16 --warmup-sizes 512x512,1024x1024   (VAE in float32)

Glue only: the transformer, VAE decoder and the host-side math (noise, pack, schedule) are in
fusezap/serve/models/flux2.py, flux2_vae.py and flux2_klein.py (I81 D4-D6), imported when the pipeline is built; the text
encoder is Qwen3TextEncoder (fusezap/serve/models/text_encoder.py, D3). The step the engine captures is the transformer
forward of Flux2KleinPipeline.denoise with the timestep as a (B,) tensor input: t = ts / 1000, ts = sigma * 1000 in
float32 (timesteps()), and dt = sigma_next - sigma (the engine's Euler update), so one capture per (batch, size)
serves every step. Attention runs in float32 on METAL (flux2.sdpa). Tests: tests/test_klein_image.py (a tiny diffusers-layout
model vs the stock pipeline from the prompt); real weights: scripts/diff_klein_e2e_metal.py.
"""
from __future__ import annotations
import os
import numpy as np
from fusezap.dtype import dtypes
from fusezap.tensor import Tensor

REPO = "black-forest-labs/FLUX.2-klein-4B"

class KleinImagePipeline:
  name = "FLUX.2-klein-4B"
  default_size = (1024, 1024)
  default_steps = 4
  max_side = 2048
  jit = False   # ImageEngine runs the step and decode eagerly: a jit capture of the 4B forward holds all its intermediates

  def __init__(self, name_or_path: str = REPO, device: str | None = None, dtype: str = "bfloat16", encoder_dtype: str | None = None,
               vae_dtype: str | None = None, load_encoder: bool = True):
    """load_encoder=False: no text encoder on the device (requests must carry prompt_embeds; scripts/diff_klein_e2e.py's
    staged encode on a 24 GB GPU, where the float32 Qwen3 (~16 GB) and the bf16 transformer (~8 GB) do not fit together).
    encoder_dtype: default float32 (see below). vae_dtype: default_vae_dtype (float32 on METAL and on CUDA GPUs of 40 GB
    or more, bfloat16 under 40 GB; FZ_KLEIN_VAE_DTYPE overrides). On METAL the VAE is 84M parameters, and in float32
    every conv is the conv kernel (float32-only), so a 1024^2 decode peaks ~6 GB above the weights; in bfloat16 the convs fall back to
    im2col + GEMM (2.4-4.8 GB im2col buffers each) and the decode needs ~24 GB."""
    from fusezap.device import Device
    from fusezap.serve.models.text_encoder import Qwen3TextEncoder, resolve_subfolders
    from fusezap.serve.models.flux2 import Flux2Transformer
    from fusezap.serve.models.flux2_vae import Flux2VAEDecoder
    from fusezap.serve.models.flux2_klein import Flux2KleinPipeline
    self.device = Device.canonicalize(device)
    dt = getattr(dtypes, dtype)
    path = resolve_subfolders(name_or_path, ("text_encoder", "tokenizer", "transformer", "vae", "scheduler"))
    # the text encoder in float32 by default (encoder_dtype / FZ_KLEIN_TE_DTYPE): with bf16 Qwen3 embeddings the 1024^2
    # image is ~2 dB further from the f32 reference than diffusers bf16 (and ~9 dB worse than from f32 embeds); 6 GB more
    self.encoder = Qwen3TextEncoder.from_pretrained(path, getattr(dtypes, encoder_dtype or os.environ.get("FZ_KLEIN_TE_DTYPE", "float32")), self.device) \
                   if load_encoder else None
    self.vae_dtype, why = default_vae_dtype(self.device, vae_dtype)
    print(f"fusezap: FLUX.2 [klein] VAE in {self.vae_dtype} ({why})", flush=True)
    vdt = getattr(dtypes, self.vae_dtype)
    self.pipe = Flux2KleinPipeline(Flux2Transformer.from_pretrained(path, dt, self.device), Flux2VAEDecoder.from_pretrained(path, vdt, self.device),
                                   self.encoder.encode if self.encoder is not None else None)
    self.tr, self.vae = self.pipe.transformer, self.pipe.vae
    self._rope: dict = {}

  # ---- host side
  def check_size(self, w: int, h: int) -> None:
    f = self.pipe.vae_scale * 2
    if w % f or h % f or not (f <= w <= self.max_side and f <= h <= self.max_side):
      raise ValueError(f"size {w}x{h} is not supported: width and height must be multiples of {f} in [{f}, {self.max_side}]")
  def _grid(self, w: int, h: int) -> tuple[int, int]: return self.pipe.latent_grid(h, w)
  def encode(self, prompts: list[str]) -> np.ndarray:
    """The encoder's transient blocks (~16 GB at 512 tokens in bf16) go back to the OS right after, not into the device
    allocator's cache, where they would sit beside the transformer's and VAE's working sets (swap on a 48 GB Mac)."""
    import gc
    from fusezap.serve.models.flux2_vae import trim_cache
    if self.encoder is None: raise ValueError("this pipeline was built without a text encoder (load_encoder=False): pass prompt_embeds")
    out = self.encoder.encode(prompts)
    gc.collect(); trim_cache(self.device)
    return out
  def noise(self, seed: int, w: int, h: int) -> np.ndarray:
    from fusezap.serve.models.flux2_klein import noise, pack
    gh, gw = self._grid(w, h)
    return pack(noise(seed, self.tr.cfg.in_channels, gh, gw, 1))[0]
  def sigmas(self, steps: int, w: int, h: int) -> np.ndarray:
    from fusezap.serve.models.flux2_klein import schedule
    gh, gw = self._grid(w, h)
    sig, _, _ = schedule(gh * gw, steps)
    return np.asarray(sig, np.float32)
  def timesteps(self, steps: int, w: int, h: int) -> np.ndarray:
    from fusezap.serve.models.flux2_klein import schedule
    gh, gw = self._grid(w, h)
    _, ts, _ = schedule(gh * gw, steps)
    return (np.asarray(ts, np.float32)[:steps] / np.float32(1000)).astype(np.float32)
  # ---- traceable
  def velocity(self, x: Tensor, cond: Tensor, t: Tensor, w: int, h: int) -> Tensor:
    gh, gw = self._grid(w, h)
    key = (cond.shape[1], gh, gw)
    if key not in self._rope: self._rope[key] = self.tr.rope(cond.shape[1], (gh, gw))
    cos, sin = self._rope[key]
    return self.tr.forward(x, cond, t, cos, sin)
  def decode(self, x: Tensor, w: int, h: int) -> Tensor:
    gh, gw = self._grid(w, h)
    return self.vae.decode_packed(x, gh, gw)

SMALL_GPU_GB = 40

def device_total_gb(device: str) -> float | None:
  """A CUDA device's total memory in GiB, else None (METAL / CPU, or no answer)."""
  if str(device).split(":")[0].upper() != "CUDA": return None
  try:
    from fusezap.device import Device
    return Device[device].mem_total() / 2 ** 30
  except Exception: return None

def default_vae_dtype(device: str, explicit: str | None = None) -> tuple[str, str]:
  """(VAE dtype, why). An explicit argument or FZ_KLEIN_VAE_DTYPE wins; else bfloat16 on a CUDA GPU under 40 GB (an
  RTX 4090: the float32 VAE's 1024^2 decode materializes up to 9.7 GB of im2col next to the transformer and ran out
  of memory in serve-image and the e2e; I95), float32 everywhere else."""
  if explicit: return explicit, "requested"
  if (env := os.environ.get("FZ_KLEIN_VAE_DTYPE")): return env, "FZ_KLEIN_VAE_DTYPE"
  gb = device_total_gb(device)
  if gb is not None and gb < SMALL_GPU_GB: return "bfloat16", f"{gb:.0f} GB CUDA device, under {SMALL_GPU_GB} GB"
  return "float32", "default"

def _factory(device: str | None = None, dtype: str = "bfloat16") -> KleinImagePipeline:
  return KleinImagePipeline(os.environ.get("FZ_KLEIN_PATH", REPO), device, dtype)
