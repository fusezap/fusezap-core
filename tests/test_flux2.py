"""I81 D4: the FLUX.2 transformer vs diffusers' Flux2Transformer2DModel on the I82 §1 tiny config (random weights,
torch.manual_seed(0), no downloads). Gate G1: max rel err <= 1e-4 (f32), on every tier the `device` fixture selects."""
from __future__ import annotations
import numpy as np
import pytest
import torch

diffusers = pytest.importorskip("diffusers")
from fusezap.dtype import dtypes
from fusezap.serve.models.flux2 import Flux2Config, Flux2Transformer, rope_tables, latent_ids, text_ids

TINY = dict(num_layers=1, num_single_layers=1, num_attention_heads=2, attention_head_dim=16, in_channels=16,
            joint_attention_dim=96, axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False)

def tiny_transformer(**over):
  torch.manual_seed(0)
  m = diffusers.Flux2Transformer2DModel(**(TINY | over)).eval()
  with torch.no_grad():   # norm weights are ones at init: perturb every parameter so a dropped weight shows
    for p in m.parameters(): p.add_(torch.randn_like(p) * 0.05)
  return m

def max_rel(a, b) -> float: return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))

def ref_forward(m, x, emb, t, h, w):
  from diffusers import Flux2KleinPipeline as P
  lat = torch.zeros(1, 1, h, w)
  with torch.no_grad():
    return m(hidden_states=torch.tensor(x), encoder_hidden_states=torch.tensor(emb), timestep=torch.tensor(t),
             img_ids=P._prepare_latent_ids(lat), txt_ids=P._prepare_text_ids(torch.tensor(emb)), guidance=None, return_dict=False)[0].numpy()

def test_ids_and_rope_match_diffusers():
  from diffusers import Flux2KleinPipeline as P
  from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed
  np.testing.assert_array_equal(latent_ids(3, 5), P._prepare_latent_ids(torch.zeros(1, 1, 3, 5))[0].numpy())
  np.testing.assert_array_equal(text_ids(7), P._prepare_text_ids(torch.zeros(1, 7, 2))[0].numpy())
  ids = np.concatenate([text_ids(512), latent_ids(32, 32)])
  c, s = rope_tables(ids, (32, 32, 32, 32), 2000)
  rc, rs = Flux2PosEmbed(2000, [32, 32, 32, 32])(torch.tensor(ids))
  np.testing.assert_array_equal(c, rc.numpy()); np.testing.assert_array_equal(s, rs.numpy())

@pytest.mark.parametrize("over,hw,T,B", [({}, (4, 4), 8, 1), (dict(num_layers=2, num_single_layers=3), (3, 5), 6, 2)])
def test_flux2_tiny_g1(device, over, hw, T, B):
  m = tiny_transformer(**over)
  cfg = Flux2Config.from_hf(dict(m.config))
  fz = Flux2Transformer(cfg, {k: v.detach().numpy() for k, v in m.state_dict().items()}, dtypes.float32)
  rng = np.random.default_rng(3)
  h, w = hw
  x = rng.standard_normal((B, h * w, 16), dtype=np.float32)
  emb = rng.standard_normal((B, T, 96), dtype=np.float32)
  for t in ([0.75] * B, [0.25 + 0.5 * i for i in range(B)]):
    t = np.array(t, np.float32)
    ref = ref_forward(m, x, emb, t, h, w)
    out = fz(x, emb, t, img=(h, w))
    assert out.shape == ref.shape
    assert max_rel(out, ref) <= 1e-4, max_rel(out, ref)

def test_flux2_weight_names_strict():
  m = tiny_transformer()
  sd = {k: v.detach().numpy() for k, v in m.state_dict().items()}
  cfg = Flux2Config.from_hf(dict(m.config))
  with pytest.raises(AssertionError): Flux2Transformer(cfg, {k: v for k, v in sd.items() if k != "proj_out.weight"})
  with pytest.raises(AssertionError): Flux2Transformer(cfg, sd | {"extra.weight": np.zeros(1, np.float32)})
