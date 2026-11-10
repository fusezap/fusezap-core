"""I81 D5: the FLUX.2 VAE decoder vs diffusers' AutoencoderKLFlux2 on the I82 §1 tiny config (random weights,
torch.manual_seed(0)). Gate G1: max rel err <= 1e-4 (f32), on every tier the `device` fixture selects."""
from __future__ import annotations
import numpy as np
import pytest
import torch

diffusers = pytest.importorskip("diffusers")
from fusezap import Tensor
from fusezap.dtype import dtypes
from fusezap.serve.models.flux2_vae import Flux2VAEConfig, Flux2VAEDecoder, to_uint8

def tiny_vae(levels: int = 2, **over):
  torch.manual_seed(0)
  boc = (8, 16, 16)[:levels]
  m = diffusers.AutoencoderKLFlux2(block_out_channels=boc, latent_channels=4, norm_num_groups=4, layers_per_block=1,
                                   down_block_types=("DownEncoderBlock2D",) * levels, up_block_types=("UpDecoderBlock2D",) * levels, **over).eval()
  with torch.no_grad():
    for p in m.parameters(): p.add_(torch.randn_like(p) * 0.05)
    m.bn.running_mean.copy_(torch.randn_like(m.bn.running_mean) * 0.3)
    m.bn.running_var.copy_(torch.rand_like(m.bn.running_var) + 0.5)
  return m

def max_rel(a, b) -> float: return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))
def sd(m) -> dict: return {k: v.detach().numpy() for k, v in m.state_dict().items()}

@pytest.mark.parametrize("levels,shape", [(2, (1, 4, 6, 6)), (3, (2, 4, 4, 6))])
def test_vae_decode_tiny_g1(device, levels, shape):
  m = tiny_vae(levels)
  fz = Flux2VAEDecoder(Flux2VAEConfig.from_hf(dict(m.config)), sd(m))
  z = np.random.default_rng(5).standard_normal(shape, dtype=np.float32)
  with torch.no_grad(): ref = m.decode(torch.tensor(z), return_dict=False)[0].numpy()
  out = fz.decode(Tensor(z)).numpy()
  assert out.shape == ref.shape
  assert max_rel(out, ref) <= 1e-4, max_rel(out, ref)

def test_vae_decode_packed_matches_pipeline_tail(device):
  from diffusers import Flux2KleinPipeline as P
  m = tiny_vae(2)
  fz = Flux2VAEDecoder(Flux2VAEConfig.from_hf(dict(m.config)), sd(m))
  h, w = 3, 4
  x = np.random.default_rng(6).standard_normal((1, h * w, 16), dtype=np.float32)
  with torch.no_grad():
    lat = P._unpack_latents_with_ids(torch.tensor(x), P._prepare_latent_ids(torch.zeros(1, 1, h, w)), h, w)
    mean = m.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(m.bn.running_var.view(1, -1, 1, 1) + m.config.batch_norm_eps)
    z = P._unpatchify_latents(lat * std + mean)
    ref = m.decode(z, return_dict=False)[0].numpy()
  np.testing.assert_allclose(fz.denormalize(Tensor(x), h, w).numpy(), z.numpy(), rtol=1e-6, atol=1e-6)
  out = fz.decode_packed(Tensor(x), h, w).numpy()
  assert max_rel(out, ref) <= 1e-4, max_rel(out, ref)

def test_to_uint8_matches_image_processor():
  from diffusers.image_processor import VaeImageProcessor
  img = np.random.default_rng(1).uniform(-1.2, 1.2, (1, 3, 5, 7)).astype(np.float32)
  ref = np.asarray(VaeImageProcessor().postprocess(torch.tensor(img), output_type="pil")[0])
  np.testing.assert_array_equal(to_uint8(img[0]), ref)

def test_vae_missing_weight():
  m = tiny_vae(2)
  w = sd(m); w.pop("decoder.conv_out.bias")
  with pytest.raises(AssertionError): Flux2VAEDecoder(Flux2VAEConfig.from_hf(dict(m.config)), w)
