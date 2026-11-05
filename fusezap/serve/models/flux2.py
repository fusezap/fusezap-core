"""FLUX.2 transformer (diffusers `Flux2Transformer2DModel`), I81 D4. The denoiser of FLUX.2 [klein] 4B.

  tr = Flux2Transformer.from_pretrained("black-forest-labs/FLUX.2-klein-4B", dtype=dtypes.float32, device="METAL")
  cos, sin = tr.rope(txt_ids, img_ids)                       # host tables, once per (text length, latent grid)
  v = tr.forward(x, embeds, timestep, cos, sin)              # (B, L, 128) velocity; every input is a Tensor

An MMDiT: `num_layers` double-stream blocks (separate text/image weights, joint attention over [text, image]), then
`num_single_layers` single-stream "parallel" blocks (one fused qkv+MLP-in projection and one fused out projection) over
the concatenated sequence, then an adaLN-continuous output norm. Specific to FLUX.2 (vs FLUX.1):
  - shared modulation: one Flux2Modulation per stream type, computed once per step from the timestep embedding
    (double image, double text, single), not one per block;
  - SwiGLU feed-forward (silu(first half) * second half) and no biases anywhere;
  - per-head RMSNorm on q and k (eps 1e-6), 4-axis RoPE (T, H, W, L) with axes (32, 32, 32, 32), theta 2000, in the
    interleaved-pair layout (diffusers repeat_interleave_real=True: pairs (2i, 2i+1) rotate together);
  - klein has no guidance embedding (guidance_embeds=false).

Exactness choices (I82 §4, G1/G3): the op order follows diffusers' modules; RoPE tables come from the host in float64
like diffusers on CPU (Flux2PosEmbed) and are rounded to float32 once; the timestep stays a Tensor input
(`timestep` = sigma*1000/1000 as the pipeline passes it, scaled by 1000 inside like diffusers) so one captured step
serves every step and request (D7 jits `forward`). Weights keep diffusers' state-dict names (the name map is the
identity for the diffusers layout). `forward` realizes after each block so the lazy graph and its live intermediates
stay one block deep.
"""
from __future__ import annotations
import json, math, os
from dataclasses import dataclass
from typing import Any
import numpy as np
from fusezap.tensor import Tensor
from fusezap.dtype import DType, dtypes
from fusezap.device import Device
from fusezap.serve.weights import load_safetensors, resolve_model, to_dtype

@dataclass
class Flux2Config:
  in_channels: int = 128
  out_channels: int = 128
  num_layers: int = 5
  num_single_layers: int = 20
  num_heads: int = 24
  head_dim: int = 128
  joint_attention_dim: int = 7680
  timestep_channels: int = 256
  mlp_ratio: float = 3.0
  axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32)
  rope_theta: float = 2000.0
  eps: float = 1e-6
  guidance_embeds: bool = False
  patch_size: int = 1
  @property
  def dim(self) -> int: return self.num_heads * self.head_dim
  @property
  def mlp_hidden(self) -> int: return int(self.dim * self.mlp_ratio)
  @staticmethod
  def from_hf(c: dict) -> "Flux2Config":
    assert c.get("patch_size", 1) == 1, f"patch_size {c.get('patch_size')} is not supported"
    return Flux2Config(c.get("in_channels", 128), c.get("out_channels") or c.get("in_channels", 128), c.get("num_layers", 8),
      c.get("num_single_layers", 48), c.get("num_attention_heads", 48), c.get("attention_head_dim", 128),
      c.get("joint_attention_dim", 15360), c.get("timestep_guidance_channels", 256), float(c.get("mlp_ratio", 3.0)),
      tuple(c.get("axes_dims_rope", (32, 32, 32, 32))), float(c.get("rope_theta", 2000)), float(c.get("eps", 1e-6)),
      bool(c.get("guidance_embeds", True)), 1)

def rope_tables(ids: np.ndarray, axes_dims: tuple[int, ...], theta: float) -> tuple[np.ndarray, np.ndarray]:
  """(S, n_axes) position ids -> cos, sin (S, sum(axes_dims)) float32: diffusers Flux2PosEmbed on CPU (positions in
  float32, frequencies and cos/sin in float64, each value repeated for its pair, axes concatenated, then float32)."""
  pos = np.asarray(ids).astype(np.float32).astype(np.float64)
  cos, sin = [], []
  for i, d in enumerate(axes_dims):
    freqs = 1.0 / (np.float64(theta) ** (np.arange(0, d, 2, dtype=np.float64) / d))
    f = np.outer(pos[:, i], freqs)
    cos.append(np.repeat(np.cos(f), 2, axis=1)); sin.append(np.repeat(np.sin(f), 2, axis=1))
  return np.concatenate(cos, 1).astype(np.float32), np.concatenate(sin, 1).astype(np.float32)

def text_ids(T: int) -> np.ndarray:
  """Flux2KleinPipeline._prepare_text_ids: (T, 4) rows (0, 0, 0, l)."""
  ids = np.zeros((T, 4), np.int64); ids[:, 3] = np.arange(T)
  return ids

def latent_ids(h: int, w: int) -> np.ndarray:
  """Flux2KleinPipeline._prepare_latent_ids: (h*w, 4) rows (0, y, x, 0), row-major over (y, x)."""
  ids = np.zeros((h * w, 4), np.int64)
  ids[:, 1] = np.repeat(np.arange(h), w); ids[:, 2] = np.tile(np.arange(w), h)
  return ids

def sdpa(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
  """Non-causal scaled_dot_product_attention for the klein transformer and VAE. On METAL a half-precision q/k/v is cast
  to float32 around it (and the output back), so the float32 flash kernel serves it (I81 D8: faster than the half
  kernels at klein's 24 x 4608 x 128 and the only one for the VAE's 512-wide head; bf16 attention computed in float32
  is also closer to the reference). FZ_KLEIN_ATTN_F32=0 keeps the input dtype. On CUDA a half-precision q/k/v goes to
  the tensor-core flash kernel (fusezap.device.cuda_kernels.attention: f32 softmax and accumulation; head dims 64/128,
  the transformer's) instead of the composed sdpa, which would materialize every (L, S) score matrix in bf16;
  FZ_KLEIN_CUDA_FLASH=0 keeps the composed graph. float32 on CUDA stays composed (the scheduler hook's SIMT kernel)."""
  from fusezap.helpers import getenv
  dt = q.dtype
  dev = str(q.device).split(":")[0].upper()
  if dt != dtypes.float32 and dev == "METAL" and getenv("FZ_KLEIN_ATTN_F32", 1):
    return q.float().scaled_dot_product_attention(k.float(), v.float()).cast(dt)
  if dt in (dtypes.bfloat16, dtypes.float16) and dev == "CUDA" and q.shape[-1] == k.shape[-1] == v.shape[-1] in (64, 128) \
     and getenv("FZ_KLEIN_CUDA_FLASH", 1):
    from fusezap.device.cuda_kernels.attention import attention as flash
    return flash(q, k, v)
  return q.scaled_dot_product_attention(k, v)

def cut(t: Tensor) -> Tensor:
  """t realized, as a new graph root: a LazyBuffer over the same device Buffer with no sources. A realized LazyBuffer
  keeps its srcs, so without this a layer's output would pin every upstream intermediate buffer of the forward (25 GB
  at a 512^2 VAE decode). Under a jit capture the realize is recorded like any other."""
  from fusezap.lazy import LazyBuffer
  lb = t._realized_contiguous()
  return Tensor(LazyBuffer.from_buffer(lb.realized, t.shape))

def timestep_freqs(dim: int, max_period: float = 10000.0) -> np.ndarray:
  """diffusers get_timestep_embedding's exp(-ln(max_period) * arange(half) / half) (downscale_freq_shift=0), float32."""
  import math as _m
  half = dim // 2
  ex = (np.float32(-_m.log(max_period)) * np.arange(half, dtype=np.float32)) / np.float32(half)
  return np.exp(ex.astype(np.float32)).astype(np.float32)

class Flux2Transformer:
  def __init__(self, cfg: Flux2Config, weights: dict[str, np.ndarray], dtype: DType = dtypes.float32, device: str | None = None,
               realize_blocks: bool = True):
    self.cfg, self.dtype, self.device = cfg, dtype, Device.canonicalize(device)
    self.realize_blocks = realize_blocks
    want = set(self.weight_names())
    have = {k for k in weights if not k.startswith("time_guidance_embed.guidance_embedder.")}
    missing, extra = sorted(want - have), sorted(have - want)
    assert not missing and not extra, f"Flux2 weights: missing {missing[:6]}, unexpected {extra[:6]}"
    self.w: dict[str, Tensor] = {}
    for k in sorted(want):   # one tensor at a time: its host copy is dropped once the device holds it
      self.w[k] = Tensor(to_dtype(weights[k], dtype), device=self.device, dtype=dtype).realize()
    self._tfreq = Tensor(timestep_freqs(cfg.timestep_channels), device=self.device).realize()

  def weight_names(self) -> list[str]:
    c = self.cfg
    n = ["x_embedder.weight", "context_embedder.weight", "proj_out.weight", "norm_out.linear.weight",
         "time_guidance_embed.timestep_embedder.linear_1.weight", "time_guidance_embed.timestep_embedder.linear_2.weight",
         "double_stream_modulation_img.linear.weight", "double_stream_modulation_txt.linear.weight",
         "single_stream_modulation.linear.weight"]
    for i in range(c.num_layers):
      p = f"transformer_blocks.{i}."
      n += [p + s for s in ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "attn.to_out.0.weight",
                            "attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight", "attn.to_add_out.weight",
                            "attn.norm_q.weight", "attn.norm_k.weight", "attn.norm_added_q.weight", "attn.norm_added_k.weight",
                            "ff.linear_in.weight", "ff.linear_out.weight", "ff_context.linear_in.weight", "ff_context.linear_out.weight")]
    for i in range(c.num_single_layers):
      p = f"single_transformer_blocks.{i}.attn."
      n += [p + s for s in ("to_qkv_mlp_proj.weight", "to_out.weight", "norm_q.weight", "norm_k.weight")]
    return n

  @staticmethod
  def from_pretrained(name_or_path: str, dtype: DType = dtypes.float32, device: str | None = None, **kw) -> "Flux2Transformer":
    """A klein-style diffusers repo / snapshot (its transformer/ subfolder) or a transformer directory itself."""
    path = resolve_model(name_or_path)
    if os.path.isdir(os.path.join(path, "transformer")): path = os.path.join(path, "transformer")
    with open(os.path.join(path, "config.json")) as f: cfg = Flux2Config.from_hf(json.load(f))
    return Flux2Transformer(cfg, load_safetensors(path), dtype, device, **kw)

  # ---- host-side inputs
  def rope(self, txt: np.ndarray | int, img: np.ndarray | tuple[int, int]) -> tuple[Tensor, Tensor]:
    """cos, sin (T+L, head_dim) Tensors for text ids then image ids (an int T / an (h, w) grid: the pipeline's ids)."""
    t = text_ids(txt) if isinstance(txt, int) else np.asarray(txt)
    i = latent_ids(*img) if isinstance(img, tuple) else np.asarray(img)
    cos, sin = rope_tables(np.concatenate([t, i], 0), self.cfg.axes_dims_rope, self.cfg.rope_theta)
    return Tensor(cos, device=self.device).realize(), Tensor(sin, device=self.device).realize()

  # ---- building blocks (diffusers module order)
  def _lin(self, x: Tensor, name: str) -> Tensor: return x.linear(self.w[name])
  def _ln(self, x: Tensor) -> Tensor: return x.layernorm(-1, self.cfg.eps)
  def _rms(self, x: Tensor, name: str) -> Tensor:
    xf = x._upcast()
    y = xf * ((xf * xf).mean(-1, keepdim=True) + self.cfg.eps).rsqrt()
    return y.cast(x.dtype) * self.w[name]
  def _swiglu(self, h: Tensor) -> Tensor:
    a, b = h.chunk(2, -1)
    return a.silu() * b
  def _rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """diffusers apply_rotary_emb(use_real_unbind_dim=-1, sequence_dim=1): x (B, S, H, D), pairs (2i, 2i+1)."""
    B, S, H, D = x.shape
    xf = x._upcast()
    pr = xf.reshape(B, S, H, D // 2, 2)
    re, im = pr[..., 0], pr[..., 1]
    rot = Tensor.stack([-im, re], axis=-1).reshape(B, S, H, D)
    out = xf * cos.reshape(1, S, 1, D) + rot * sin.reshape(1, S, 1, D)
    return out.cast(x.dtype)
  def _heads(self, x: Tensor) -> Tensor:
    B, S, _ = x.shape
    return x.reshape(B, S, self.cfg.num_heads, self.cfg.head_dim)
  def _attn(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """(B, S, H, D) x3 -> (B, S, H*D): non-causal SDPA in the (B, H, S, D) layout torch's native backend uses."""
    B, S, H, D = q.shape
    o = sdpa(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3))
    return o.permute(0, 2, 1, 3).reshape(B, S, H * D)
  @staticmethod
  def _mod(m: Tensor, n: int) -> list[Tensor]:
    """Flux2Modulation.split: (B, 3n*D) -> 3n chunks (B, 1, D): (shift, scale, gate) per set."""
    return [c.unsqueeze(1) for c in m.chunk(3 * n, -1)]

  def _double(self, i: int, x: Tensor, c: Tensor, mi: list[Tensor], mt: list[Tensor], cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    p = f"transformer_blocks.{i}."
    sh, sc, g, sh2, sc2, g2 = mi
    csh, csc, cg, csh2, csc2, cg2 = mt
    nx = (1 + sc) * self._ln(x) + sh
    nc = (1 + csc) * self._ln(c) + csh
    q, k, v = (self._heads(self._lin(nx, p + f"attn.to_{n}.weight")) for n in "qkv")
    q, k = self._rms(q, p + "attn.norm_q.weight"), self._rms(k, p + "attn.norm_k.weight")
    eq, ek, ev = (self._heads(self._lin(nc, p + f"attn.add_{n}_proj.weight")) for n in "qkv")
    eq, ek = self._rms(eq, p + "attn.norm_added_q.weight"), self._rms(ek, p + "attn.norm_added_k.weight")
    q, k, v = eq.cat(q, axis=1), ek.cat(k, axis=1), ev.cat(v, axis=1)
    q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
    T = c.shape[1]
    o = self._attn(q, k, v)
    oc, ox = o[:, :T], o[:, T:]
    ox, oc = self._lin(ox, p + "attn.to_out.0.weight"), self._lin(oc, p + "attn.to_add_out.weight")
    x = x + g * ox
    nx = self._ln(x) * (1 + sc2) + sh2
    x = x + g2 * self._lin(self._swiglu(self._lin(nx, p + "ff.linear_in.weight")), p + "ff.linear_out.weight")
    c = c + cg * oc
    nc = self._ln(c) * (1 + csc2) + csh2
    c = c + cg2 * self._lin(self._swiglu(self._lin(nc, p + "ff_context.linear_in.weight")), p + "ff_context.linear_out.weight")
    return x, c

  def _single(self, i: int, h: Tensor, m: list[Tensor], cos: Tensor, sin: Tensor) -> Tensor:
    p, D, M = f"single_transformer_blocks.{i}.attn.", self.cfg.dim, self.cfg.mlp_hidden
    sh, sc, g = m
    n = (1 + sc) * self._ln(h) + sh
    pr = self._lin(n, p + "to_qkv_mlp_proj.weight")
    q, k, v, mlp = pr.split([D, D, D, 2 * M], -1)
    q, k, v = self._heads(q), self._heads(k), self._heads(v)
    q, k = self._rms(q, p + "norm_q.weight"), self._rms(k, p + "norm_k.weight")
    q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
    o = self._attn(q, k, v)
    out = self._lin(o.cat(self._swiglu(mlp), axis=-1), p + "to_out.weight")
    return h + g * out

  def temb(self, timestep: Tensor) -> Tensor:
    """Flux2TimestepGuidanceEmbeddings without guidance: sinusoid of timestep*1000 (cos first), linear, SiLU, linear."""
    t = timestep.cast(self.dtype) * 1000
    e = t.float().reshape(-1, 1) * self._tfreq.reshape(1, -1)
    e = e.cos().cat(e.sin(), axis=-1)                                     # flip_sin_to_cos=True
    e = e.cast(self.dtype)
    e = self._lin(e, "time_guidance_embed.timestep_embedder.linear_1.weight").silu()
    return self._lin(e, "time_guidance_embed.timestep_embedder.linear_2.weight")

  def _r(self, *ts: Tensor) -> list[Tensor]:
    return [cut(t) for t in ts] if self.realize_blocks else list(ts)

  def forward(self, x: Tensor, embeds: Tensor, timestep: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x (B, L, in_channels) packed latents, embeds (B, T, joint_attention_dim), timestep (B,) = sigma (the pipeline's
    t/1000), cos/sin (T+L, head_dim) from rope(). Returns the velocity (B, L, out_channels)."""
    with Tensor.inference(): return self._forward(x, embeds, timestep, cos, sin)

  def _forward(self, x: Tensor, embeds: Tensor, timestep: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    c = self.cfg
    x, embeds = x.cast(self.dtype), embeds.cast(self.dtype)
    temb = self.temb(timestep)
    st = temb.silu()
    mi = self._mod(self._lin(st, "double_stream_modulation_img.linear.weight"), 2)
    mt = self._mod(self._lin(st, "double_stream_modulation_txt.linear.weight"), 2)
    ms = self._mod(self._lin(st, "single_stream_modulation.linear.weight"), 1)
    mi, mt, ms, st = self._r(*mi), self._r(*mt), self._r(*ms), self._r(st)[0]
    h = self._lin(x, "x_embedder.weight")
    e = self._lin(embeds, "context_embedder.weight")
    for i in range(c.num_layers):
      h, e = self._r(*self._double(i, h, e, mi, mt, cos, sin))
    T = e.shape[1]
    h = e.cat(h, axis=1)
    for i in range(c.num_single_layers):
      h = self._r(self._single(i, h, ms, cos, sin))[0]
    h = h[:, T:]
    sc, sh = self._lin(st, "norm_out.linear.weight").chunk(2, -1)        # AdaLayerNormContinuous: scale first
    h = self._ln(h) * (1 + sc).unsqueeze(1) + sh.unsqueeze(1)
    return self._lin(h, "proj_out.weight")

  def __call__(self, x: np.ndarray, embeds: np.ndarray, timestep: np.ndarray | float, txt: np.ndarray | int | None = None,
               img: np.ndarray | tuple[int, int] | None = None) -> np.ndarray:
    """numpy convenience: x (B, L, C), embeds (B, T, Dj), timestep (B,) or float; ids default to the pipeline's
    (a square latent grid). Returns float32 numpy (B, L, out_channels)."""
    B, L, _ = x.shape
    T = embeds.shape[1]
    if img is None:
      s = int(round(math.sqrt(L))); assert s * s == L, "pass img ids or an (h, w) grid for a non-square latent"
      img = (s, s)
    cos, sin = self.rope(T if txt is None else txt, img)
    ts = np.broadcast_to(np.asarray(timestep, np.float32), (B,)).copy()
    dev = self.device
    out = self.forward(Tensor(np.ascontiguousarray(x, np.float32), device=dev), Tensor(np.ascontiguousarray(embeds, np.float32), device=dev),
                       Tensor(ts, device=dev), cos, sin)
    return out.float().numpy()
