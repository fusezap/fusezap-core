"""I81 D6: the klein pipeline end to end on the I82 §1 tiny configs vs the stock diffusers Flux2KleinPipeline, with the
prompt embeddings injected (the encoder is gated on its own, D3) and the same numpy noise. G1 on the trajectory
(every step's velocity and latent), G4 (final latent) and G6 (uint8 image) on every tier the `device` fixture selects."""
from __future__ import annotations
import os, zlib
import numpy as np
import pytest
import torch

diffusers = pytest.importorskip("diffusers")
from fusezap.dtype import dtypes
from fusezap.serve.models.flux2 import Flux2Config, Flux2Transformer
from fusezap.serve.models.flux2_vae import Flux2VAEConfig, Flux2VAEDecoder
from fusezap.serve.models.flux2_klein import Flux2KleinPipeline, schedule, compute_empirical_mu, noise, pack, write_png
from test_flux2 import tiny_transformer
from test_flux2_vae import tiny_vae

KLEIN_SCHED = dict(base_image_seq_len=256, base_shift=0.5, invert_sigmas=False, max_image_seq_len=4096, max_shift=1.15,
                   num_train_timesteps=1000, shift=3.0, shift_terminal=None, stochastic_sampling=False, time_shift_type="exponential",
                   use_beta_sigmas=False, use_dynamic_shifting=True, use_exponential_sigmas=False, use_karras_sigmas=False)

def sd(m): return {k: v.detach().numpy() for k, v in m.state_dict().items()}
def rel_l2(a, b): return float(np.linalg.norm((a - b).astype(np.float64)) / np.linalg.norm(b.astype(np.float64)))
def max_rel(a, b): return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))

@pytest.mark.parametrize("L,steps", [(64, 4), (1024, 4), (4096, 4), (4608, 2), (256, 1), (1024, 8)])
def test_schedule_matches_diffusers(L, steps):
  from diffusers import FlowMatchEulerDiscreteScheduler
  from diffusers.pipelines.flux2.pipeline_flux2_klein import retrieve_timesteps, compute_empirical_mu as ref_mu
  assert compute_empirical_mu(L, steps) == ref_mu(L, steps)
  sch = FlowMatchEulerDiscreteScheduler(**KLEIN_SCHED)
  ts, _ = retrieve_timesteps(sch, steps, "cpu", sigmas=np.linspace(1.0, 1 / steps, steps), mu=ref_mu(L, steps))
  sig, mts, _ = schedule(L, steps)
  np.testing.assert_array_equal(sig, sch.sigmas.numpy()); np.testing.assert_array_equal(mts, ts.numpy())
  np.testing.assert_array_equal(mts / np.float32(1000), (ts / 1000).numpy())

def test_pipeline_tiny_e2e(device):
  from diffusers import Flux2KleinPipeline as RefPipe, FlowMatchEulerDiscreteScheduler
  tr, vae = tiny_transformer(), tiny_vae(2)
  ref = RefPipe(scheduler=FlowMatchEulerDiscreteScheduler(**KLEIN_SCHED), vae=vae, text_encoder=None, tokenizer=None,
                transformer=tr, is_distilled=True)
  H = W = 32                                  # vae_scale 2 -> a 8x8 token grid
  emb = np.random.default_rng(9).standard_normal((1, 8, 96), dtype=np.float32)
  nz = noise(7, 16, 8, 8)
  xs, vs = [], []
  def cb(pipe, i, t, kw):
    xs.append(kw["latents"].numpy().copy()); return {}
  orig = tr.forward
  def spy(*a, **k):
    out = orig(*a, **k); vs.append(out[0].detach().numpy().copy()); return out
  tr.forward = spy
  with torch.no_grad():
    img = ref(prompt_embeds=torch.tensor(emb), latents=torch.tensor(nz), height=H, width=W, num_inference_steps=4,
              output_type="np", callback_on_step_end=cb).images[0]
  tr.forward = orig
  pipe = Flux2KleinPipeline(Flux2Transformer(Flux2Config.from_hf(dict(tr.config)), sd(tr)),
                            Flux2VAEDecoder(Flux2VAEConfig.from_hf(dict(vae.config)), sd(vae)))
  out = pipe.generate_from_embeds(emb, size=(H, W), steps=4, seed=7, keep_trajectory=True)
  np.testing.assert_array_equal(out.trajectory[0], pack(nz))
  for k in range(4):
    assert max_rel(out.velocities[k], vs[k]) <= 1e-4, (k, max_rel(out.velocities[k], vs[k]))
    assert max_rel(out.trajectory[k + 1], xs[k]) <= 1e-4, (k, max_rel(out.trajectory[k + 1], xs[k]))
  assert rel_l2(out.latents, xs[-1]) <= 1e-3                                # G4
  ref_u8 = np.round(img * 255).astype(np.uint8)
  diff = np.abs(out.images[0].astype(int) - ref_u8.astype(int))
  assert (diff <= 1).mean() >= 0.999, diff.max()                            # G6
  assert out.images[0].shape == (H, W, 3)

def test_generate_cli_tiny(tmp_path, monkeypatch):
  """`fusezap generate --model <diffusers dir> --embeds ...` on a saved tiny model writes the same PNG the library makes."""
  from PIL import Image
  from fusezap.__main__ import main
  from fusezap.device import Device
  monkeypatch.setattr(Device, "DEFAULT", "CPU" if Device.available("CPU") else "NPY", raising=False)
  tr, vae = tiny_transformer(), tiny_vae(2)
  tr.save_pretrained(str(tmp_path / "m" / "transformer")); vae.save_pretrained(str(tmp_path / "m" / "vae"))
  emb = np.random.default_rng(2).standard_normal((8, 96), dtype=np.float32)
  np.savez(str(tmp_path / "e.npz"), embeds=emb)
  out = str(tmp_path / "o.png")
  assert main(["generate", "--model", str(tmp_path / "m"), "--embeds", str(tmp_path / "e.npz"), "--seed", "3", "--size", "48x32",
               "--steps", "2", "--out", out]) == 0
  png = np.asarray(Image.open(out).convert("RGB"))
  assert png.shape == (32, 48, 3)
  pipe = Flux2KleinPipeline(Flux2Transformer.from_pretrained(str(tmp_path / "m")), Flux2VAEDecoder.from_pretrained(str(tmp_path / "m")))
  np.testing.assert_array_equal(png, pipe.generate_from_embeds(emb[None], size=(32, 48), steps=2, seed=3).images[0])
  with pytest.raises(SystemExit): main(["generate", "--model", str(tmp_path / "m")])   # neither --prompt nor --embeds

def test_write_png_roundtrip(tmp_path):
  from PIL import Image
  rgb = np.random.default_rng(0).integers(0, 256, (5, 7, 3), dtype=np.uint8)
  fn = str(tmp_path / "x.png")
  write_png(fn, rgb)
  np.testing.assert_array_equal(np.asarray(Image.open(fn).convert("RGB")), rgb)

def test_latent_grid_errors():
  pipe = Flux2KleinPipeline(None, None)
  assert pipe.latent_grid(512, 768) == (32, 48)
  with pytest.raises(AssertionError): pipe.latent_grid(500, 512)
