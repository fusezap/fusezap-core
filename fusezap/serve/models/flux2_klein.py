"""FLUX.2 [klein] text-to-image pipeline (diffusers `Flux2KleinPipeline`, distilled: no CFG), I81 D6.

  pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", device="METAL")
  out = pipe.generate_from_embeds(embeds, size=(512, 512), steps=4, seed=0)    # embeds: (1, 512, 7680) float32
  out = pipe.generate("a red fox", size=512, seed=0)                            # needs pipe.encode_prompt (below)
  write_png("fox.png", out.images[0])

The seam for the text encoder is `encode_prompt: Callable[[list[str]], np.ndarray]`, prompts -> (B, 512, 7680) float32
prompt embeddings (Qwen3 hidden states 9/18/27 of the chat-templated, 512-padded prompt, concatenated per token, as
Flux2KleinPipeline._get_qwen3_prompt_embeds). The pipeline itself only ever sees embeddings (`generate_from_embeds`).
FuseZap's own Qwen3 encoder (I81 D3) plugs in there; until then `HFQwen3PromptEncoder` (transformers on the CPU) is the
interim encoder for `fusezap generate`, and the gates inject the reference embeddings directly.

Host-side math repeats diffusers' float32 ops so the schedule is bit-identical (I82 §4): sigmas
np.linspace(1, 1/steps, steps) -> float32, the empirical mu for the token count, the exponential time shift; the step
timestep is sigma*1000 (float32) and the transformer gets that /1000, as the pipeline passes it. Flow-match Euler:
x += (sigma_next - sigma) * v in float32. The initial noise is numpy PCG64 `noise(seed, ...)`, the same bits
scripts/diff_ref.py feeds diffusers.
"""
from __future__ import annotations
import math, os, time, zlib, struct
from dataclasses import dataclass, field
from typing import Callable, Sequence
import numpy as np
from fusezap.tensor import Tensor
from fusezap.dtype import DType, dtypes
from fusezap.device import Device
from fusezap.serve.models.flux2 import Flux2Transformer, cut
from fusezap.serve.models.flux2_vae import Flux2VAEDecoder, to_uint8, trim_cache

MAX_TEXT = 512
TEXT_LAYERS = (9, 18, 27)

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
  """diffusers pipeline_flux2_klein.compute_empirical_mu."""
  a1, b1 = 8.73809524e-05, 1.89833333
  a2, b2 = 0.00016927, 0.45666666
  if image_seq_len > 4300: return float(a2 * image_seq_len + b2)
  m_200, m_10 = a2 * image_seq_len + b2, a1 * image_seq_len + b1
  a = (m_200 - m_10) / 190.0
  return float(a * num_steps + (m_200 - 200.0 * a))

def schedule(image_seq_len: int, steps: int) -> tuple[np.ndarray, np.ndarray, float]:
  """(sigmas (steps+1,) float32 ending in 0, timesteps (steps,) float32 = sigma*1000, mu): FlowMatchEulerDiscreteScheduler
  (klein config: use_dynamic_shifting, exponential time shift, no terminal stretch) set_timesteps(sigmas=, mu=)."""
  s = np.array(np.linspace(1.0, 1 / steps, steps)).astype(np.float32)
  mu = compute_empirical_mu(image_seq_len, steps)
  s = (math.exp(mu) / (math.exp(mu) + (1 / s - 1) ** 1.0)).astype(np.float32)
  ts = (s * np.float32(1000)).astype(np.float32)
  return np.concatenate([s, np.zeros(1, np.float32)]), ts, mu

def noise(seed: int, channels: int, h: int, w: int, batch: int = 1) -> np.ndarray:
  """Initial latent noise (batch, channels, h, w) float32, numpy PCG64 (scripts/diff_ref.noise for klein's 128 ch)."""
  return np.random.default_rng(seed).standard_normal((batch, channels, h, w), dtype=np.float32)

def pack(lat: np.ndarray) -> np.ndarray:
  """(B, C, h, w) -> (B, h*w, C), Flux2KleinPipeline._pack_latents."""
  B, C, h, w = lat.shape
  return np.ascontiguousarray(lat.reshape(B, C, h * w).transpose(0, 2, 1))

def write_png(path: str, rgb: np.ndarray) -> None:
  """(H, W, 3) uint8 -> an 8-bit RGB PNG (stdlib zlib; no imaging dependency)."""
  rgb = np.ascontiguousarray(rgb, np.uint8)
  H, W, C = rgb.shape
  assert C == 3, f"write_png wants (H, W, 3), got {rgb.shape}"
  raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(H))
  def chunk(t: bytes, d: bytes) -> bytes: return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
  with open(path, "wb") as f:
    f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))

@dataclass
class GenerateOutput:
  images: list[np.ndarray]                 # (H, W, 3) uint8 each
  image_f32: np.ndarray                    # (B, 3, H, W) decoder output in [-1, 1]
  latents: np.ndarray                      # (B, L, 128) final packed latents
  sigmas: np.ndarray
  timesteps: np.ndarray
  trajectory: list[np.ndarray] = field(default_factory=list)   # x_0..x_steps (B, L, C) when keep_trajectory
  velocities: list[np.ndarray] = field(default_factory=list)
  timings: dict = field(default_factory=dict)

class Flux2KleinPipeline:
  def __init__(self, transformer: Flux2Transformer | None, vae: Flux2VAEDecoder | None,
               encode_prompt: Callable[[list[str]], np.ndarray] | None = None, max_text: int = MAX_TEXT,
               vae_scale: int | None = None):
    """vae_scale: the VAE's spatial factor when `vae` is not loaded yet (default: from vae, else 8 as klein)."""
    self.transformer, self.vae, self.encode_prompt, self.max_text = transformer, vae, encode_prompt, max_text
    self._vae_scale = vae_scale

  @staticmethod
  def from_pretrained(name_or_path: str, dtype: DType = dtypes.float32, device: str | None = None,
                      encode_prompt: Callable[[list[str]], np.ndarray] | None = None) -> "Flux2KleinPipeline":
    return Flux2KleinPipeline(Flux2Transformer.from_pretrained(name_or_path, dtype, device),
                              Flux2VAEDecoder.from_pretrained(name_or_path, dtype, device), encode_prompt)

  @property
  def vae_scale(self) -> int:
    if self.vae is not None: return 2 ** (len(self.vae.cfg.block_out_channels) - 1)
    return self._vae_scale or 8
  def latent_grid(self, H: int, W: int) -> tuple[int, int]:
    """Token grid (h, w) for an image size: the pipeline's 2*(H // (vae_scale*2)) latent rows, halved by the 2x2 patch."""
    f = self.vae_scale * 2
    assert H % f == 0 and W % f == 0, f"image size {H}x{W} must be a multiple of {f}"
    return H // f, W // f

  def denoise(self, embeds: np.ndarray, x: np.ndarray, h: int, w: int, steps: int, keep: bool = False,
              callback: Callable[[int, np.ndarray], None] | None = None) -> tuple[np.ndarray, dict]:
    """The denoising loop: packed latents x (B, L, C) from noise, `steps` flow-match Euler steps."""
    tr, dev = self.transformer, self.transformer.device
    sig, ts, mu = schedule(x.shape[1], steps)
    cos, sin = tr.rope(embeds.shape[1], (h, w))
    e = Tensor(np.ascontiguousarray(embeds, np.float32), device=dev).realize()
    xt = Tensor(np.ascontiguousarray(x, np.float32), device=dev).realize()
    traj, vel, B = [x.astype(np.float32)] if keep else [], [], x.shape[0]
    for k in range(steps):
      t = Tensor(np.full((B,), ts[k] / np.float32(1000), np.float32), device=dev)
      v = cut(tr.forward(xt, e, t, cos, sin).float())
      xt = cut(xt + v * float(sig[k + 1] - sig[k]))             # scheduler.step: sample + dt * model_output, float32
      if keep: vel.append(v.numpy()); traj.append(xt.numpy())
      if callback is not None: callback(k, xt.numpy())
    return xt.numpy(), {"sigmas": sig, "timesteps": ts, "mu": mu, "trajectory": traj, "velocities": vel}

  def denoise_embeds(self, embeds: np.ndarray, size: int | tuple[int, int] = 512, steps: int = 4, seed: int = 0,
                     latents: np.ndarray | None = None, keep_trajectory: bool = False,
                     callback: Callable[[int, np.ndarray], None] | None = None) -> tuple[np.ndarray, dict, tuple[int, int]]:
    """The transformer half: embeds (B, T, joint_dim) -> final packed latents (B, h*w, C), the schedule/trajectory info
    and the token grid (h, w). latents: initial noise (B, C, h, w) instead of noise(seed)."""
    H, W = (size, size) if isinstance(size, int) else size
    h, w = self.latent_grid(H, W)
    B = embeds.shape[0]
    C = self.transformer.cfg.in_channels
    nz = noise(seed, C, h, w, B) if latents is None else np.asarray(latents, np.float32)
    assert nz.shape == (B, C, h, w), f"latents {nz.shape} != {(B, C, h, w)}"
    x, info = self.denoise(embeds, pack(nz), h, w, steps, keep_trajectory, callback)
    trim_cache(self.transformer.device)
    return x, info, (h, w)

  def decode_latents(self, x: np.ndarray, h: int, w: int) -> np.ndarray:
    """The VAE half: packed latents (B, h*w, C) -> (B, 3, H, W) float32 in [-1, 1]."""
    img = self.vae.decode_packed(Tensor(np.ascontiguousarray(x, np.float32), device=self.vae.device), h, w).float().numpy()
    trim_cache(self.vae.device)
    return img

  def generate_from_embeds(self, embeds: np.ndarray, size: int | tuple[int, int] = 512, steps: int = 4, seed: int = 0,
                           latents: np.ndarray | None = None, keep_trajectory: bool = False,
                           callback: Callable[[int, np.ndarray], None] | None = None) -> GenerateOutput:
    """embeds (B, T, joint_dim) float32 -> images (both components resident)."""
    t0 = time.perf_counter()
    x, info, (h, w) = self.denoise_embeds(embeds, size, steps, seed, latents, keep_trajectory, callback)
    t1 = time.perf_counter()
    img = self.decode_latents(x, h, w)
    t2 = time.perf_counter()
    return GenerateOutput([to_uint8(i) for i in img], img, x, info["sigmas"], info["timesteps"], info["trajectory"],
                          info["velocities"], {"denoise_s": t1 - t0, "decode_s": t2 - t1})

  def generate(self, prompt: str | Sequence[str], **kw) -> GenerateOutput:
    if self.encode_prompt is None: raise RuntimeError("no text encoder: pass encode_prompt=, or use generate_from_embeds")
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    return self.generate_from_embeds(np.asarray(self.encode_prompt(prompts), np.float32), **kw)

class HFQwen3PromptEncoder:
  """Interim prompt encoder (until FuseZap's Qwen3 encoder, I81 D3): transformers' Qwen3ForCausalLM on the CPU in float32,
  exactly Flux2KleinPipeline._get_qwen3_prompt_embeds (chat template with enable_thinking=False, pad to 512, hidden
  states 9/18/27 concatenated). Needs torch + transformers (the dev group). `release()` frees the model."""
  def __init__(self, name_or_path: str, device: str = "cpu", dtype: str = "float32"):
    import torch
    from transformers import Qwen3ForCausalLM, AutoTokenizer
    from fusezap.serve.weights import resolve_model
    path = resolve_model(name_or_path)
    self.torch, self.device = torch, device
    self.tok = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
    self.model = Qwen3ForCausalLM.from_pretrained(os.path.join(path, "text_encoder"), torch_dtype=getattr(torch, dtype)).to(device).eval()
  def __call__(self, prompts: list[str]) -> np.ndarray:
    torch = self.torch
    ids, masks = [], []
    for p in prompts:
      text = self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
      enc = self.tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_TEXT)
      ids.append(enc["input_ids"]); masks.append(enc["attention_mask"])
    with torch.no_grad():
      out = self.model(input_ids=torch.cat(ids).to(self.device), attention_mask=torch.cat(masks).to(self.device),
                       output_hidden_states=True, use_cache=False)
    hs = torch.stack([out.hidden_states[k] for k in TEXT_LAYERS], dim=1)       # (B, 3, T, D)
    B, n, T, D = hs.shape
    return hs.permute(0, 2, 1, 3).reshape(B, T, n * D).float().cpu().numpy()
  def release(self) -> None:
    self.model = None
    import gc; gc.collect()
