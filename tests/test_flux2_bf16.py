"""I81: bfloat16 weights for the FLUX.2 transformer and VAE on the tiny configs, gated relative to diffusers' own bf16
(the I82 §4 bf16 class): FuseZap-bf16's distance from the float32 reference may be at most 1.5x diffusers-bf16's, plus
the SPEC §10 bf16 tolerance as a floor for tiny outputs."""
from __future__ import annotations
import numpy as np
import pytest
import torch

diffusers = pytest.importorskip("diffusers")
from fusezap import Tensor
from fusezap.dtype import dtypes
from fusezap.serve.models.flux2 import Flux2Config, Flux2Transformer
from fusezap.serve.models.flux2_vae import Flux2VAEConfig, Flux2VAEDecoder
from test_flux2 import tiny_transformer, ref_forward
from test_flux2_vae import tiny_vae

def rel_l2(a, b): return float(np.linalg.norm((a - b).astype(np.float64)) / np.linalg.norm(b.astype(np.float64)))
def sd_bf16(m): return {k: v.detach().bfloat16().float().numpy() for k, v in m.state_dict().items()}   # same rounded weights

def test_flux2_tiny_bf16_relative(device):
  m = tiny_transformer()
  rng = np.random.default_rng(4)
  x, emb, t = rng.standard_normal((1, 16, 16), dtype=np.float32), rng.standard_normal((1, 8, 96), dtype=np.float32), np.array([0.6], np.float32)
  ref = ref_forward(m, x, emb, t, 4, 4)
  mb = tiny_transformer().bfloat16()
  with torch.no_grad():
    from diffusers import Flux2KleinPipeline as P
    d_bf16 = mb(hidden_states=torch.tensor(x).bfloat16(), encoder_hidden_states=torch.tensor(emb).bfloat16(), timestep=torch.tensor(t).bfloat16(),
                img_ids=P._prepare_latent_ids(torch.zeros(1, 1, 4, 4)), txt_ids=P._prepare_text_ids(torch.tensor(emb)), guidance=None,
                return_dict=False)[0].float().numpy()
  fz = Flux2Transformer(Flux2Config.from_hf(dict(m.config)), sd_bf16(m), dtypes.bfloat16)(x, emb, t, img=(4, 4))
  assert rel_l2(fz, ref) <= max(1.5 * rel_l2(d_bf16, ref), 1e-2), (rel_l2(fz, ref), rel_l2(d_bf16, ref))

def test_vae_tiny_bf16_relative(device):
  m = tiny_vae(2)
  z = np.random.default_rng(5).standard_normal((1, 4, 6, 6), dtype=np.float32)
  with torch.no_grad():
    ref = m.decode(torch.tensor(z), return_dict=False)[0].numpy()
    d_bf16 = tiny_vae(2).bfloat16().decode(torch.tensor(z).bfloat16(), return_dict=False)[0].float().numpy()
  fz = Flux2VAEDecoder(Flux2VAEConfig.from_hf(dict(m.config)), sd_bf16(m), dtypes.bfloat16).decode(Tensor(z)).float().numpy()
  assert rel_l2(fz, ref) <= max(1.5 * rel_l2(d_bf16, ref), 1e-2), (rel_l2(fz, ref), rel_l2(d_bf16, ref))
