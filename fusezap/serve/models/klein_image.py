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

NEURON (I93): `fusezap serve-image flux2-klein --device NEURON --dtype bfloat16` runs the transformer on one NeuronCore
(bf16, the I96 W1 path and the I93 switches in fusezap/codegen/nki_klein.py) and the two parts NEURON cannot run on the host
CPU: the text encoder as transformers' Qwen3 in float32 (FuseZap's Qwen3 has no NEURON path: llama.py refuses q/k RMSNorm
there) and the VAE decoder as diffusers' AutoencoderKLFlux2 in float32 (TorchVAEDecoder; FuseZap's CPU VAE took > 18 min
for one 512^2 decode in I96 W7). FZ_KLEIN_ENCODER (fusezap | hf) and FZ_KLEIN_VAE (device | torch) override the placement
(defaults: fusezap / device off NEURON, hf / torch on it). One image per step on NEURON (max_batch_cap 1): another batch
size is another set of NEFFs. tests/test_klein_neuron_serve.py runs this on NKISHIM.
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
               vae_dtype: str | None = None, load_encoder: bool = True, encoder: str | None = None, vae: str | None = None):
    """load_encoder=False: no text encoder on the device (requests must carry prompt_embeds; scripts/diff_klein_e2e.py's
    staged encode on a 24 GB GPU, where the float32 Qwen3 (~16 GB) and the bf16 transformer (~8 GB) do not fit together).
    encoder_dtype: default float32 (see below). vae_dtype: default_vae_dtype (float32 on METAL and on CUDA GPUs of 40 GB
    or more, bfloat16 under 40 GB; FZ_KLEIN_VAE_DTYPE overrides). On METAL the VAE is 84M parameters, and in float32
    every conv is the conv kernel (float32-only), so a 1024^2 decode peaks ~6 GB above the weights; in bfloat16 the convs fall back to
    im2col + GEMM (2.4-4.8 GB im2col buffers each) and the decode needs ~24 GB."""
    from fusezap.device import Device
    from fusezap.serve.models.text_encoder import Qwen3TextEncoder, resolve_subfolders
    from fusezap.serve.models.flux2 import Flux2Transformer, _neuron
    from fusezap.serve.models.flux2_vae import Flux2VAEDecoder
    from fusezap.serve.models.flux2_klein import Flux2KleinPipeline
    self.device = Device.canonicalize(device)
    neuron = _neuron(self.device)
    dt = getattr(dtypes, dtype)
    path = resolve_subfolders(name_or_path, ("text_encoder", "tokenizer", "transformer", "vae", "scheduler"))
    self.encoder_kind = encoder or os.environ.get("FZ_KLEIN_ENCODER") or ("hf" if neuron else "fusezap")
    self.vae_kind = vae or os.environ.get("FZ_KLEIN_VAE") or ("torch" if neuron else "device")
    assert self.encoder_kind in ("fusezap", "hf") and self.vae_kind in ("device", "torch"), (self.encoder_kind, self.vae_kind)
    if neuron: self.max_batch_cap = int(os.environ.get("FZ_KLEIN_NEURON_BATCH", "1") or 1)
    # the text encoder in float32 by default (encoder_dtype / FZ_KLEIN_TE_DTYPE): with bf16 Qwen3 embeddings the 1024^2
    # image is ~2 dB further from the f32 reference than diffusers bf16 (and ~9 dB worse than from f32 embeds); 6 GB more
    if not load_encoder: self.encoder = None
    elif self.encoder_kind == "hf":   # transformers' Qwen3 on the host CPU, float32 (flux2_klein.HFQwen3PromptEncoder)
      from fusezap.serve.models.flux2_klein import HFQwen3PromptEncoder
      self.encoder = HFQwen3PromptEncoder(path)
    else: self.encoder = Qwen3TextEncoder.from_pretrained(path, getattr(dtypes, encoder_dtype or os.environ.get("FZ_KLEIN_TE_DTYPE", "float32")), self.device)
    if self.vae_kind == "torch":
      self.vae_dtype = "float32"
      vae_mod = TorchVAEDecoder(path)
      print("fusezap: FLUX.2 [klein] VAE in float32 on the host CPU (diffusers AutoencoderKLFlux2)", flush=True)
    else:
      self.vae_dtype, why = default_vae_dtype(self.device, vae_dtype)
      print(f"fusezap: FLUX.2 [klein] VAE in {self.vae_dtype} ({why})", flush=True)
      vae_mod = Flux2VAEDecoder.from_pretrained(path, getattr(dtypes, self.vae_dtype), self.device)
    enc = None if self.encoder is None else (self.encoder.encode if hasattr(self.encoder, "encode") else self.encoder)
    self.pipe = Flux2KleinPipeline(Flux2Transformer.from_pretrained(path, dt, self.device), vae_mod, enc)
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
    out = self.encoder.encode(prompts) if hasattr(self.encoder, "encode") else np.asarray(self.encoder(prompts), np.float32)
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

class TorchVAEDecoder:
  """The FLUX.2 VAE decoder as diffusers' AutoencoderKLFlux2 on the host CPU in float32, behind Flux2VAEDecoder's
  decode_packed: unpack (row-major ids), BN denormalize (running_mean / sqrt(running_var + eps)), unpatchify, decode,
  as Flux2KleinPipeline does. For devices whose own VAE is impractical (NEURON: no conv kernel, 9 GiB im2col buffers at
  1024^2). Needs torch + diffusers (the dev group). Takes and returns fusezap Tensors (the result on the NPY device)."""
  def __init__(self, path: str, threads: int | None = None):
    import torch
    from diffusers import AutoencoderKLFlux2
    from types import SimpleNamespace
    self.torch = torch
    if threads: torch.set_num_threads(int(threads))
    self.m = AutoencoderKLFlux2.from_pretrained(os.path.join(path, "vae") if os.path.isdir(os.path.join(path, "vae")) else path,
                                                torch_dtype=torch.float32).eval()
    c = self.m.config
    self.cfg = SimpleNamespace(block_out_channels=tuple(c.block_out_channels), patch_size=tuple(c.patch_size))
    self.device = "NPY"
    self.bn_mean = self.m.bn.running_mean.detach().float().reshape(1, -1, 1, 1)
    self.bn_std = torch.sqrt(self.m.bn.running_var.detach().float().reshape(1, -1, 1, 1) + float(c.batch_norm_eps))
  def decode_packed_np(self, x: np.ndarray, h: int, w: int) -> np.ndarray:
    torch = self.torch
    B, L, Cp = x.shape
    assert L == h * w, f"{L} tokens do not form a {h}x{w} grid"
    ph, pw = self.cfg.patch_size
    with torch.no_grad():
      lat = torch.from_numpy(np.ascontiguousarray(x, np.float32)).reshape(B, h, w, Cp).permute(0, 3, 1, 2)
      lat = lat * self.bn_std + self.bn_mean
      C = Cp // (ph * pw)
      lat = lat.reshape(B, C, ph, pw, h, w).permute(0, 1, 4, 2, 5, 3).reshape(B, C, h * ph, w * pw)
      return self.m.decode(lat, return_dict=False)[0].float().numpy()
  def decode_packed(self, x: Tensor, h: int, w: int) -> Tensor:
    return Tensor(self.decode_packed_np(x.float().numpy(), h, w), device=self.device)

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
