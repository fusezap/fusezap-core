"""I81 D2: conv2d, group_norm, interpolate(nearest), nn.Conv2d / nn.GroupNorm against torch, forward and backward, on
every tier the `device` fixture selects (interp, NPY, CPU, METAL by default; NKISHIM with FZ_TEST_TIERS)."""
from __future__ import annotations
import numpy as np
import pytest
import torch
import torch.nn.functional as TF
import fusezap.nn as nn
from fusezap import Tensor, dtypes, manual_seed

R = np.random.default_rng(82)
def rnd(*shape, s=1.0): return (R.standard_normal(shape) * s).astype(np.float32)
def close(a, b, tol=1e-4):   # SPEC §10 reduction/matmul tolerance (rtol 1e-4, atol 1e-5)
  np.testing.assert_allclose(a.numpy() if isinstance(a, Tensor) else a, b.detach().numpy(), rtol=tol, atol=tol * 0.1)

CONV = [  # (N, C, H, W), (O, kh, kw), stride, padding, dilation
  ((1, 3, 7, 6), (4, 3, 3), 1, 1, 1),
  ((2, 4, 8, 8), (5, 3, 3), 2, 1, 1),     # VAE-encoder style downsample, batch 2
  ((1, 6, 5, 5), (3, 1, 1), 1, 0, 1),     # 1x1 (post_quant_conv)
  ((1, 2, 9, 7), (3, 3, 2), (2, 1), (1, 0), 1),
  ((1, 2, 8, 8), (2, 3, 3), 1, "same", 2),
  ((1, 2, 6, 6), (2, 3, 3), 1, "valid", 1),
  ((1, 3, 6, 6), (2, 2, 2), 2, 0, 1),     # patchify-style 2x2 s2
]

@pytest.mark.parametrize("i", range(len(CONV)))
def test_conv2d_forward_backward(device, i):
  xs, (O, kh, kw), s, p, d = CONV[i]
  x, w, b = rnd(*xs), rnd(O, xs[1], kh, kw, s=0.3), rnd(O)
  tx, tw, tb = (torch.tensor(a, requires_grad=True) for a in (x, w, b))
  ref = TF.conv2d(tx, tw, tb, stride=s, padding=p, dilation=d)
  fx, fw, fb = (Tensor(a, requires_grad=True) for a in (x, w, b))
  out = fx.conv2d(fw, fb, stride=s, padding=p, dilation=d)
  assert out.shape == tuple(ref.shape)
  close(out, ref)
  g = rnd(*ref.shape)
  ref.backward(torch.tensor(g)); out.backward(Tensor(g))
  close(fx.grad, tx.grad); close(fw.grad, tw.grad); close(fb.grad, tb.grad)

def test_conv2d_no_bias_and_errors(device):
  x, w = rnd(1, 2, 5, 5), rnd(3, 2, 3, 3)
  close(Tensor(x).conv2d(Tensor(w), padding=1), TF.conv2d(torch.tensor(x), torch.tensor(w), padding=1))
  with pytest.raises(NotImplementedError): Tensor(x).conv2d(Tensor(rnd(2, 1, 3, 3)), groups=2)
  with pytest.raises(AssertionError): Tensor(x).conv2d(Tensor(rnd(3, 4, 3, 3)))

@pytest.mark.parametrize("shape,G", [((2, 8, 5, 4), 4), ((1, 6, 3, 3), 1), ((1, 6, 3, 3), 6), ((2, 4, 7), 2)])
def test_group_norm_forward_backward(device, shape, G):
  x, w, b = rnd(*shape, s=3.0) + 1.5, rnd(shape[1]), rnd(shape[1])
  tx, tw, tb = (torch.tensor(a, requires_grad=True) for a in (x, w, b))
  ref = TF.group_norm(tx, G, tw, tb, eps=1e-6)
  fx, fw, fb = (Tensor(a, requires_grad=True) for a in (x, w, b))
  out = fx.group_norm(G, fw, fb, eps=1e-6)
  close(out, ref)
  g = rnd(*shape)
  ref.backward(torch.tensor(g)); out.backward(Tensor(g))
  close(fx.grad, tx.grad); close(fw.grad, tw.grad); close(fb.grad, tb.grad)
  close(Tensor(x).group_norm(G), TF.group_norm(torch.tensor(x), G))   # no affine, default eps

def test_group_norm_long_groups(device):
  """VAE-sized groups (here 2^16 elements): the two-level statistics stay near float64 (one f32 reduction over 2^20
  elements was 2.7e-4 off on METAL)."""
  x = rnd(1, 8, 128, 256, s=3.0) + 2.0
  out = Tensor(x).group_norm(2, eps=1e-6).numpy().astype(np.float64)
  ref = TF.group_norm(torch.tensor(x).double(), 2, eps=1e-6).numpy()
  assert np.linalg.norm(out - ref) / np.linalg.norm(ref) < 2e-6

def test_sum_over_reshaped_product(device):
  """A SUM whose source is a reshape of a MUL (group_norm's chunked sum of squares) once crashed the custom-kernel
  matcher (schedule._mm_spec indexed the reduce axis into the MUL's own shape)."""
  a, b = rnd(4, 8 * 16), rnd(4, 8 * 16)
  ta, tb = torch.tensor(a), torch.tensor(b)
  close((Tensor(a) * Tensor(b)).reshape(4, 8, 16).sum(-1).sum(-1, keepdim=True), (ta * tb).reshape(4, 8, 16).sum(-1).sum(-1, keepdim=True))
  d = Tensor(a) - Tensor(a).mean(-1, keepdim=True)
  close((d * d).reshape(4, 2, 64).sum(-1).sum(-1), ((ta - ta.mean(-1, keepdim=True)) ** 2).reshape(4, 2, 64).sum(-1).sum(-1))

def test_group_norm_bf16(device):
  x = rnd(1, 8, 4, 4, s=4.0)
  y = Tensor(x).cast(dtypes.bfloat16).group_norm(2)
  assert y.dtype == dtypes.bfloat16
  ref = TF.group_norm(torch.tensor(x).bfloat16().float(), 2).numpy()
  np.testing.assert_allclose(y.float().numpy(), ref, rtol=2e-2, atol=2e-2)

INTERP = [((1, 2, 3, 4), dict(scale_factor=2)), ((2, 1, 4, 4), dict(size=(8, 12))), ((1, 2, 8, 6), dict(size=(4, 3))),
          ((1, 1, 5, 7), dict(size=(7, 4))), ((1, 2, 5), dict(scale_factor=3)), ((1, 1, 3, 3), dict(scale_factor=(2, 1)))]

@pytest.mark.parametrize("i", range(len(INTERP)))
def test_interpolate_nearest(device, i):
  shape, kw = INTERP[i]
  x = rnd(*shape)
  tx = torch.tensor(x, requires_grad=True)
  ref = TF.interpolate(tx, mode="nearest", **{k: (tuple(float(f) for f in v) if k == "scale_factor" and isinstance(v, tuple) else v) for k, v in kw.items()})
  fx = Tensor(x, requires_grad=True)
  out = fx.interpolate(mode="nearest", **kw)
  np.testing.assert_array_equal(out.numpy(), ref.detach().numpy())   # nearest is a pure copy: bit-exact
  g = rnd(*ref.shape)
  ref.backward(torch.tensor(g)); out.backward(Tensor(g))
  close(fx.grad, tx.grad)

def test_interpolate_errors(device):
  with pytest.raises(NotImplementedError): Tensor(rnd(1, 1, 2, 2)).interpolate(scale_factor=2, mode="bilinear")

def test_nn_conv2d_groupnorm_modules(device):
  manual_seed(0)
  conv, gn = nn.Conv2d(4, 6, 3, padding=1), nn.GroupNorm(2, 6, eps=1e-6)
  assert conv.weight.shape == (6, 4, 3, 3) and conv.bias.shape == (6,)
  k = 1 / np.sqrt(4 * 9)
  assert float(np.abs(conv.weight.numpy()).max()) <= k + 1e-6 and float(np.abs(conv.bias.numpy()).max()) <= k + 1e-6
  assert (gn.weight.numpy() == 1).all() and (gn.bias.numpy() == 0).all()
  tconv, tgn = torch.nn.Conv2d(4, 6, 3, padding=1), torch.nn.GroupNorm(2, 6, eps=1e-6)
  sd = {f"conv.{k}": v.detach().numpy() for k, v in tconv.state_dict().items()}
  sd.update({f"gn.{k}": (v.detach().numpy() + 0.5) for k, v in tgn.state_dict().items()})
  class M:
    def __init__(self): self.conv, self.gn = conv, gn
  m = M()
  assert sorted(nn.get_state_dict(m)) == sorted(sd)   # torch state-dict names
  nn.load_state_dict(m, sd)
  with torch.no_grad():
    tgn.weight += 0.5; tgn.bias += 0.5
  x = rnd(1, 4, 6, 5)
  close(m.gn(m.conv(Tensor(x))).silu(), TF.silu(tgn(tconv(torch.tensor(x)))))
  assert nn.Conv2d(2, 3, 1, bias=False).bias is None and nn.GroupNorm(1, 2, affine=False).weight is None

def test_vae_resnet_half(device):
  """GN -> SiLU -> conv3x3 -> nearest 2x -> conv3x3, the decoder's up path, as one graph."""
  x, g, b = rnd(1, 8, 6, 6), rnd(8), rnd(8)
  w1, c1, w2, c2 = rnd(8, 8, 3, 3, s=0.2), rnd(8), rnd(4, 8, 3, 3, s=0.2), rnd(4)
  T = lambda a: torch.tensor(a)
  ref = TF.conv2d(TF.interpolate(TF.conv2d(TF.silu(TF.group_norm(T(x), 4, T(g), T(b), eps=1e-6)), T(w1), T(c1), padding=1),
                                 scale_factor=2.0, mode="nearest"), T(w2), T(c2), padding=1)
  F_ = lambda a: Tensor(a)
  out = F_(x).group_norm(4, F_(g), F_(b), eps=1e-6).silu().conv2d(F_(w1), F_(c1), padding=1).interpolate(scale_factor=2).conv2d(F_(w2), F_(c2), padding=1)
  close(out, ref)
