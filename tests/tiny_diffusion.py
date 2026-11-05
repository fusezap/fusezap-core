"""A tiny random flow-matching image pipeline for the ImageEngine / images-API tests (I81 D7). Not a real model: it has
the shapes and the call structure of FLUX.2 klein (latents as (N, C) patch tokens at 1/16 of the image side, text
conditioning (L, C_txt), a sigma timestep, joint non-causal attention over text + image tokens, a decode to
(3, H, W) in [-1, 1]) at sizes that run in milliseconds. Random weights from a numpy seed, built from Tensor ops only.

  TinyFlow(device="CPU")                     the step-level DiffusionPipeline (fusezap/serve/image_engine.py)
  TinyFlowMonolithic(device="CPU")           only generate() (the engine's fallback path)
  python -m fusezap serve-image tests.tiny_diffusion:TinyFlow  (a server for manual checks)
"""
from __future__ import annotations
import math
import numpy as np
from fusezap.tensor import Tensor

class TinyFlow:
  name = "tiny-flow"
  default_size = (64, 64)
  default_steps = 4
  L, CT, C, D, H = 8, 24, 16, 32, 2         # text tokens, text channels, latent channels, model width, heads
  P = 16                                   # pixels per latent token side (8x VAE x 2x2 patch)
  def __init__(self, device: str = "CPU", seed: int = 0, shift: float = 3.0, sizes: tuple = ((64, 64), (64, 32), (32, 32))):
    self.device, self.shift, self.sizes = device, shift, tuple(tuple(s) for s in sizes)
    rng = np.random.default_rng(seed)
    def W(*shape, s=None): return Tensor((rng.standard_normal(shape) * (s or 1 / math.sqrt(shape[-1]))).astype(np.float32), device=device).realize()
    self.emb = rng.standard_normal((256, self.CT)).astype(np.float32)      # byte embedding of the prompt (host)
    D, C = self.D, self.C
    self.w_t1, self.w_t2 = W(D, 32), W(D, D)
    self.w_in, self.w_c = W(D, C), W(D, self.CT)
    self.w_qkv, self.w_o = W(3 * D, D), W(D, D)
    self.w_m1, self.w_m2 = W(2 * D, D), W(D, 2 * D)
    self.w_out = W(C, D, s=0.05)
    self.w_dec = W(3 * self.P * self.P, C, s=0.3)
  # ---- host side
  def check_size(self, w: int, h: int) -> None:
    if (w, h) not in self.sizes: raise ValueError(f"size {w}x{h} is not supported (have {', '.join(f'{a}x{b}' for a, b in self.sizes)})")
  def tokens(self, w: int, h: int) -> int: return (w // self.P) * (h // self.P)
  def encode(self, prompts: list[str]) -> np.ndarray:
    out = np.zeros((len(prompts), self.L, self.CT), np.float32)
    for b, p in enumerate(prompts):
      ids = list(p.encode())[:self.L]
      out[b, :len(ids)] = self.emb[ids]
    return out
  def noise(self, seed: int, w: int, h: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((self.tokens(w, h), self.C)).astype(np.float32)
  def sigmas(self, steps: int, w: int, h: int) -> np.ndarray:
    s = np.linspace(1.0, 0.0, steps + 1)
    return (self.shift * s / (1 + (self.shift - 1) * s)).astype(np.float32)
  # ---- traceable
  def velocity(self, x: Tensor, cond: Tensor, t: Tensor, w: int, h: int) -> Tensor:
    B, N, D, H = x.shape[0], x.shape[1], self.D, self.H
    half = 16
    freqs = Tensor(np.exp(-math.log(10000) * np.arange(half, dtype=np.float32) / half), device=self.device)
    ang = (t * 1000).reshape(B, 1) * freqs.reshape(1, half)
    temb = ang.cos().cat(ang.sin(), axis=-1).linear(self.w_t1).silu().linear(self.w_t2)          # (B, D)
    hx = x.linear(self.w_in) + temb.reshape(B, 1, D)
    hc = cond.linear(self.w_c)
    j = hc.cat(hx, axis=1)                                                                         # (B, L + N, D)
    S = j.shape[1]
    qkv = j.layernorm().linear(self.w_qkv).reshape(B, S, 3, H, D // H)
    q, k, v = [qkv.shrink((None, None, (i, i + 1), None, None)).reshape(B, S, H, D // H).transpose(1, 2) for i in range(3)]
    a = q.scaled_dot_product_attention(k, v).transpose(1, 2).reshape(B, S, D)
    j = j + a.linear(self.w_o)
    j = j + j.layernorm().linear(self.w_m1).gelu().linear(self.w_m2)
    return j.shrink((None, (self.L, S), None)).linear(self.w_out)                                 # (B, N, C)
  def decode(self, x: Tensor, w: int, h: int) -> Tensor:
    B, P = x.shape[0], self.P
    gh, gw = h // P, w // P
    y = x.linear(self.w_dec).tanh().reshape(B, gh, gw, 3, P, P)                                    # unpatchify
    return y.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, h, w)
  # ---- the monolithic interface (also offered here)
  def generate(self, prompt: str | None = None, prompt_embeds: np.ndarray | None = None, seed: int = 0, size: tuple = (64, 64), steps: int = 4):
    from fusezap.serve.image_engine import reference_image, ImageParams
    img = reference_image(self, ImageParams(prompt=prompt, prompt_embeds=prompt_embeds, seed=seed, width=size[0], height=size[1], steps=steps))
    return (img.astype(np.float32) / 255 * 2 - 1).transpose(2, 0, 1)

class TinyFlowMonolithic:
  """Only generate() (and the metadata): the engine falls back to one request at a time."""
  name, default_size, default_steps = "tiny-flow-mono", (64, 64), 4
  def __init__(self, device: str = "CPU"):
    self.inner, self.device = TinyFlow(device), device
    self.calls = 0
  def check_size(self, w: int, h: int) -> None: self.inner.check_size(w, h)
  def generate(self, prompt=None, prompt_embeds=None, seed=0, size=(64, 64), steps=4):
    self.calls += 1
    return self.inner.generate(prompt, prompt_embeds, seed, size, steps)
