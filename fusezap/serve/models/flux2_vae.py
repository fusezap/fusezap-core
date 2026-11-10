"""FLUX.2 VAE decoder (diffusers `AutoencoderKLFlux2`, decode path only), I81 D5.

  vae = Flux2VAEDecoder.from_pretrained("black-forest-labs/FLUX.2-klein-4B", device="METAL")
  img = vae.decode_packed(x, h, w)      # transformer output latents (B, h*w, 128) -> image (B, 3, 16h, 16w) in [-1, 1]
  img = vae.decode(z)                   # VAE latents (B, 32, 2h, 2w) -> image (B, 3, 16h, 16w)

decode_packed is the klein pipeline's tail: unpack the (row-major) tokens to (B, 128, h, w), undo the latent
BatchNorm with its running statistics (x * sqrt(var + eps) + mean), unpatchify 2x2 to (B, 32, 2h, 2w), then
post_quant_conv (1x1) and the Decoder: conv_in, mid block (resnet, single-head attention over all positions with a
GroupNorm in front and a residual, resnet), up blocks (layers_per_block+1 resnets each, a nearest-2x + 3x3 conv
upsampler on all but the last), GroupNorm + SiLU + conv_out. GroupNorm eps 1e-6, SiLU, 32 groups for klein.

Every layer is composed from the Tensor frontend (conv2d = im2col + matmul, group_norm, interpolate; I82 §2.2), so this
runs on any backend; D8 swaps in fast conv/attention kernels. Spatial tiling is not offered: the mid-block attention
is global, so tiled decodes are not exact (I82 §3). Weights keep diffusers' names (`decoder.*`, `post_quant_conv.*`,
`bn.running_mean/var`); encoder weights are ignored. `decode` realizes after each resnet/attention so at most one
layer's im2col buffer is live.
"""
from __future__ import annotations
import json, math, os
from dataclasses import dataclass
import numpy as np
from fusezap.tensor import Tensor
from fusezap.dtype import DType, dtypes
from fusezap.device import Device
from fusezap.serve.weights import load_safetensors, resolve_model, to_dtype
from fusezap.serve.models.flux2 import cut, sdpa

@dataclass
class Flux2VAEConfig:
  latent_channels: int = 32
  out_channels: int = 3
  block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
  layers_per_block: int = 2
  norm_num_groups: int = 32
  batch_norm_eps: float = 1e-4
  patch_size: tuple[int, int] = (2, 2)
  use_post_quant_conv: bool = True
  mid_block_add_attention: bool = True
  @staticmethod
  def from_hf(c: dict) -> "Flux2VAEConfig":
    assert c.get("act_fn", "silu") == "silu", f"act_fn {c.get('act_fn')} is not supported"
    assert all(t == "UpDecoderBlock2D" for t in c.get("up_block_types", ["UpDecoderBlock2D"])), "only UpDecoderBlock2D"
    boc = tuple(c.get("decoder_block_out_channels") or c.get("block_out_channels", (128, 256, 512, 512)))
    return Flux2VAEConfig(c.get("latent_channels", 32), c.get("out_channels", 3), boc, c.get("layers_per_block", 2),
                          c.get("norm_num_groups", 32), float(c.get("batch_norm_eps", 1e-4)), tuple(c.get("patch_size", (2, 2))),
                          bool(c.get("use_post_quant_conv", True)), bool(c.get("mid_block_add_attention", True)))

class Flux2VAEDecoder:
  EPS = 1e-6   # every GroupNorm in the decoder (resnet_eps and conv_norm_out)

  def __init__(self, cfg: Flux2VAEConfig, weights: dict[str, np.ndarray], dtype: DType = dtypes.float32, device: str | None = None,
               realize_layers: bool = True):
    self.cfg, self.dtype, self.device, self.realize_layers = cfg, dtype, Device.canonicalize(device), realize_layers
    want = set(self.weight_names())
    missing = sorted(want - set(weights))
    assert not missing, f"Flux2 VAE weights: missing {missing[:6]}"
    self.w = {k: Tensor(to_dtype(weights[k], dtype), device=self.device, dtype=dtype).realize() for k in sorted(want)
              if not k.startswith("bn.")}
    mean = to_dtype(weights["bn.running_mean"], dtypes.float32)
    var = to_dtype(weights["bn.running_var"], dtypes.float32)
    std = np.sqrt(var + np.float32(cfg.batch_norm_eps)).astype(np.float32)   # torch.sqrt(running_var + eps) in float32
    C = len(mean)
    self.bn_mean = Tensor(mean.reshape(1, C, 1, 1), device=self.device).realize()
    self.bn_std = Tensor(std.reshape(1, C, 1, 1), device=self.device).realize()

  # ---- weights
  def _resnet_names(self, p: str, cin: int, cout: int) -> list[str]:
    n = [p + s for s in ("norm1.weight", "norm1.bias", "conv1.weight", "conv1.bias", "norm2.weight", "norm2.bias", "conv2.weight", "conv2.bias")]
    return n + ([p + "conv_shortcut.weight", p + "conv_shortcut.bias"] if cin != cout else [])
  def weight_names(self) -> list[str]:
    c = self.cfg
    n = ["bn.running_mean", "bn.running_var", "decoder.conv_in.weight", "decoder.conv_in.bias", "decoder.conv_norm_out.weight",
         "decoder.conv_norm_out.bias", "decoder.conv_out.weight", "decoder.conv_out.bias"]
    if c.use_post_quant_conv: n += ["post_quant_conv.weight", "post_quant_conv.bias"]
    top = c.block_out_channels[-1]
    n += self._resnet_names("decoder.mid_block.resnets.0.", top, top) + self._resnet_names("decoder.mid_block.resnets.1.", top, top)
    if c.mid_block_add_attention:
      n += [f"decoder.mid_block.attentions.0.{s}.{t}" for s in ("group_norm", "to_q", "to_k", "to_v", "to_out.0") for t in ("weight", "bias")]
    for i, (cin, cout) in enumerate(self._up_channels()):
      for j in range(c.layers_per_block + 1):
        n += self._resnet_names(f"decoder.up_blocks.{i}.resnets.{j}.", cin if j == 0 else cout, cout)
      if i < len(c.block_out_channels) - 1: n += [f"decoder.up_blocks.{i}.upsamplers.0.conv.weight", f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"]
    return n
  def _up_channels(self) -> list[tuple[int, int]]:
    rev = list(reversed(self.cfg.block_out_channels))
    return [(rev[max(i - 1, 0)], rev[i]) for i in range(len(rev))]

  @staticmethod
  def from_pretrained(name_or_path: str, dtype: DType = dtypes.float32, device: str | None = None, **kw) -> "Flux2VAEDecoder":
    path = resolve_model(name_or_path)
    if os.path.isdir(os.path.join(path, "vae")): path = os.path.join(path, "vae")
    with open(os.path.join(path, "config.json")) as f: cfg = Flux2VAEConfig.from_hf(json.load(f))
    return Flux2VAEDecoder(cfg, load_safetensors(path), dtype, device, **kw)

  # ---- layers
  def _conv(self, x: Tensor, p: str, padding: int = 1) -> Tensor: return x.conv2d(self.w[p + ".weight"], self.w[p + ".bias"], padding=padding)
  def _gn(self, x: Tensor, p: str) -> Tensor: return x.group_norm(self.cfg.norm_num_groups, self.w[p + ".weight"], self.w[p + ".bias"], self.EPS)
  def _r(self, x: Tensor) -> Tensor: return cut(x) if self.realize_layers else x
  def _trim(self) -> None: trim_cache(self.device)
  def _resnet(self, x: Tensor, p: str) -> Tensor:
    h = self._conv(self._gn(x, p + "norm1").silu(), p + "conv1")
    h = self._conv(self._gn(h, p + "norm2").silu(), p + "conv2")
    if p + "conv_shortcut.weight" in self.w: x = self._conv(x, p + "conv_shortcut", 0)
    return self._r(x + h)
  def _attention(self, x: Tensor, p: str) -> Tensor:
    """diffusers Attention (AttnProcessor2_0), one head of dim C over the H*W positions, GroupNorm first, residual."""
    B, C, H, W = x.shape
    h = self._gn(x, p + "group_norm").reshape(B, C, H * W).permute(0, 2, 1)       # (B, HW, C)
    q, k, v = (h.linear(self.w[f"{p}to_{n}.weight"], self.w[f"{p}to_{n}.bias"]).unsqueeze(1) for n in "qkv")
    o = sdpa(q, k, v).squeeze(1)
    o = o.linear(self.w[p + "to_out.0.weight"], self.w[p + "to_out.0.bias"])
    return self._r(o.permute(0, 2, 1).reshape(B, C, H, W) + x)

  # ---- forward
  def decode(self, z: Tensor) -> Tensor:
    """VAE latents (B, latent_channels, H, W) -> image (B, out_channels, 8H, 8W) for klein's 4 levels, in [-1, 1]."""
    with Tensor.inference(): return self._decode(z)

  def _decode(self, z: Tensor) -> Tensor:
    c = self.cfg
    x = z.cast(self.dtype)
    if c.use_post_quant_conv: x = self._conv(x, "post_quant_conv", 0)
    x = self._r(self._conv(x, "decoder.conv_in"))
    x = self._resnet(x, "decoder.mid_block.resnets.0.")
    if c.mid_block_add_attention: x = self._attention(x, "decoder.mid_block.attentions.0.")
    x = self._resnet(x, "decoder.mid_block.resnets.1.")
    for i in range(len(c.block_out_channels)):
      for j in range(c.layers_per_block + 1): x = self._resnet(x, f"decoder.up_blocks.{i}.resnets.{j}.")
      if i < len(c.block_out_channels) - 1:
        x = self._r(self._conv(x.interpolate(scale_factor=2), f"decoder.up_blocks.{i}.upsamplers.0.conv"))
      if self.realize_layers: self._trim()   # each level's im2col sizes differ: drop the cached blocks of the last one
    x = self._gn(x, "decoder.conv_norm_out").silu()
    return self._conv(x, "decoder.conv_out")

  def denormalize(self, x: Tensor, h: int, w: int) -> Tensor:
    """Packed transformer latents (B, h*w, C*ph*pw) -> VAE latents (B, C, ph*h, pw*w): unpack (row-major ids), BN
    denormalize, unpatchify (Flux2KleinPipeline._unpack_latents_with_ids / bn / _unpatchify_latents)."""
    B, L, Cp = x.shape
    assert L == h * w, f"{L} tokens do not form a {h}x{w} grid"
    ph, pw = self.cfg.patch_size
    lat = x.float().reshape(B, h, w, Cp).permute(0, 3, 1, 2)
    lat = lat * self.bn_std + self.bn_mean
    C = Cp // (ph * pw)
    return lat.reshape(B, C, ph, pw, h, w).permute(0, 1, 4, 2, 5, 3).reshape(B, C, h * ph, w * pw)

  def decode_packed(self, x: Tensor, h: int, w: int) -> Tensor: return self.decode(self.denormalize(x, h, w))

def trim_cache(device: str) -> None:
  """Return the device allocator's cached free blocks (LRUAllocator.free_cache). Metal's cache is unbounded and a 512^2
  decode frees a different set of block sizes at every level."""
  try: dev = Device[device]
  except Exception: return
  if not hasattr(dev.allocator, "free_cache"): return
  dev.synchronize()   # METAL releases a freed block only once no in-flight kernel can read it
  dev.allocator.free_cache()

def to_uint8(img: np.ndarray) -> np.ndarray:
  """(3, H, W) float image in [-1, 1] -> (H, W, 3) uint8 like diffusers' image processor ((x/2+0.5).clamp(0,1), *255,
  round half to even as numpy/torch round)."""
  x = np.clip(np.asarray(img, np.float32) * np.float32(0.5) + np.float32(0.5), 0, 1)
  return np.round(x.transpose(1, 2, 0) * np.float32(255)).astype(np.uint8)
