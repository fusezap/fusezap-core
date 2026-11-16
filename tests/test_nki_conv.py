"""I91 (I81 D9): NKI kernels for the FLUX.2 VAE / long attention, on the NKISHIM tier (FZ_TEST_TIERS=NKISIM: the real nki 0.6
simulator, scripts/nki_i91_pod.sh):
  * the implicit-GEMM conv (fusezap/codegen/nki_conv.py) behind the scheduler's "conv" match: every CONV_CASES geometry
    (strides, padding, dilation, odd shapes) and VAE-like channel counts (> 128: several channel blocks; Wo > 512: row
    segments; R > 1 whole rows per tile) against the composed conv (im2col + matmul on the same tier) and torch, to float32
    rounding (tap-major vs channel-major summation); the per-row DMA form (FZ_NEURON_CONV_ROWDMA) gives the same bits;
  * flash attention at the klein transformer's length (S = 4608, D = 128) and the VAE mid-block's (S = 16384, D = Dv = 512),
    as the scheduler's "attention" kernel against the composed softmax(QKᵀ)V on the same tier and a float64 reference
    (query rows cut to 128 / 256 so the shim stays quick; the key length, the D chunks and Dv = 512 are the real ones).
"""
from __future__ import annotations
import os
import numpy as np
import pytest
from fusezap import helpers
from fusezap.device import Device
from fusezap.tensor import Tensor
from fusezap.codegen.nki import NKIUnsupported
from fusezap.codegen.nki_conv import render_conv

TIER = next((t for t in os.getenv("FZ_TEST_TIERS", "").split(",") if t.strip() == "NKISIM"), "NKISHIM").strip()
R = np.random.default_rng(91)

def rnd(*shape, s: float = 1.0) -> np.ndarray: return (R.standard_normal(shape) * s).astype(np.float32)

def names(t: Tensor) -> list[str]:
  from fusezap.schedule import create_schedule
  return [si.ast[0].kernel.name if si.kind == "custom" else si.kind for si in create_schedule([t.lazydata]) if si.kind != "copy"]

@pytest.fixture
def conv_on(monkeypatch):
  """The tier's device with "conv" offered (FZ_NEURON_CONV=1 at device creation; the cached device gets the entry here)."""
  from fusezap.schedule import CONV_PARAMS, CustomKernel
  helpers.getenv.cache_clear()
  dev = Device[TIER]
  monkeypatch.setitem(dev.custom_kernels, "conv", CustomKernel("fz_conv", {TIER: "nki_conv.render_conv"}, ("C", "A", "X") + CONV_PARAMS, 3))
  yield dev
  helpers.getenv.cache_clear()

def _both(fn, monkeypatch, *arrs):
  outs, ks = [], None
  for flag in ("1", "0"):
    monkeypatch.setenv("FZ_CUSTOM", flag); helpers.getenv.cache_clear()
    if flag == "1": ks = names(fn(*[Tensor(a, device=TIER) for a in arrs]))
    outs.append(fn(*[Tensor(a, device=TIER) for a in arrs]).numpy())
  monkeypatch.delenv("FZ_CUSTOM"); helpers.getenv.cache_clear()
  return outs[0], outs[1], ks

def conv2d_composed(x: Tensor, weight: Tensor, stride=1, padding=0, dilation=1) -> Tensor:
  from test_metal_flash import conv2d_composed as f
  return f(x, weight, stride, padding, dilation)

CONV_CASES = {   # (C, H, W, O, kh, kw, stride, padding, dilation): test_metal_flash's geometries + VAE-like channel counts
  "3x3_s1_p1": (4, 9, 11, 5, 3, 3, 1, 1, 1),
  "3x3_s2_p1": (3, 10, 9, 6, 3, 3, 2, 1, 1),
  "3x3_s1_p0": (2, 7, 7, 3, 3, 3, 1, 0, 1),
  "5x5_p2": (3, 12, 10, 4, 5, 5, 1, 2, 1),
  "1x3_p01": (5, 6, 8, 7, 1, 3, 1, (0, 1), 1),
  "dilated": (4, 13, 13, 9, 3, 3, 1, 2, 2),
  "wide_tiles": (40, 33, 47, 70, 3, 3, 1, 1, 1),
  "rgb_out": (37, 20, 24, 3, 3, 3, 1, 1, 1),
  "vae_c256_3x3": (256, 6, 8, 160, 3, 3, 1, 1, 1),       # two channel blocks, M > 128 (two output blocks)
  "vae_c160_3x3": (160, 18, 20, 130, 3, 3, 1, 1, 1),    # a ragged channel block, R = 25 rows per tile
  "row_segments": (6, 3, 1030, 4, 3, 3, 1, 1, 1),        # Wo > 512: 512-column segments of a row, a ragged last one
}

@pytest.mark.parametrize("case", list(CONV_CASES))
def test_nki_conv_matches_composed(case, conv_on, monkeypatch):
  C, H, W, O, kh, kw, s, p, d = CONV_CASES[case]
  x, w = rnd(1, C, H, W), rnd(O, C, kh, kw, s=0.2)
  fn = lambda x, w: conv2d_composed(x.contiguous(), w, s, p, d)
  c, comp, ks = _both(fn, monkeypatch, x, w)
  assert "fz_conv" in ks and "fz_matmul" not in ks, ks
  np.testing.assert_allclose(c, comp, rtol=1e-4, atol=1e-5)
  torch = pytest.importorskip("torch")
  ref = torch.nn.functional.conv2d(torch.tensor(x).double(), torch.tensor(w).double(), stride=s, padding=p, dilation=d).numpy()
  np.testing.assert_allclose(c, ref, rtol=1e-4, atol=1e-4)

def test_1x1_conv_stays_a_matmul(conv_on, monkeypatch):
  """A 1x1 unpadded conv composes to a plain matmul (no shifted views): the scheduler keeps fz_matmul, no im2col to avoid."""
  x, w = rnd(1, 256, 12, 16), rnd(160, 256, 1, 1, s=0.2)
  c, comp, ks = _both(lambda x, w: conv2d_composed(x.contiguous(), w), monkeypatch, x, w)
  assert "fz_matmul" in ks and "fz_conv" not in ks, ks
  np.testing.assert_allclose(c, comp, rtol=1e-4, atol=1e-5)

@pytest.mark.parametrize("case", ["3x3_s1_p1", "3x3_s2_p1", "dilated", "vae_c160_3x3", "row_segments"])
def test_nki_conv_rowdma_same_bits(case, conv_on, monkeypatch):
  """FZ_NEURON_CONV_ROWDMA=1 (one DMA per valid output row instead of one 3-D access pattern per tap): the same bits."""
  C, H, W, O, kh, kw, s, p, d = CONV_CASES[case]
  x, w = rnd(1, C, H, W), rnd(O, C, kh, kw, s=0.2)
  fn = lambda x, w: conv2d_composed(x.contiguous(), w, s, p, d)
  outs = []
  for rd in ("0", "1"):
    monkeypatch.setenv("FZ_NEURON_CONV_ROWDMA", rd)
    outs.append(_both(fn, monkeypatch, x, w)[0])
  assert np.array_equal(outs[0].view(np.int32), outs[1].view(np.int32))

def test_nki_conv_source_and_refusals():
  """One DMA per tap and channel block for the X window (3-D on both sides); Wᵀ kept when it fits; bad params refused."""
  from fusezap.schedule import CONV_PARAMS
  C, H, W, O, kh, kw = 256, 64, 64, 128, 3, 3
  p = dict(M=O, N=H * W, K=C * 9, sam=C * 9, sak=1, oa=0, so=H * W, sp=1, C=C, H=H, W=W, Wo=W, kh=kh, kw=kw, sh=1, sw=1, dh=1, dw=1, ph=1, pw=1)
  prog = render_conv(tuple(p[k] for k in CONV_PARAMS))
  assert prog.name.startswith("fz_conv_128_256x64x64_3x3") and prog.n_bufs == 3
  assert prog.src.count("nisa.nc_matmul(") == 1 and "xt.ap(pattern=" in prog.src and "wt = nl.ndarray((128, 18 * mt)" in prog.src
  assert "_r" not in prog.name and "for r in range(ha, hb)" in render_conv(tuple(p[k] for k in CONV_PARAMS), rowdma=True).src
  with pytest.raises(NKIUnsupported): render_conv(tuple({**p, "K": 7}[k] for k in CONV_PARAMS))
  with pytest.raises(NKIUnsupported): render_conv(tuple({**p, "N": H * W + 1}[k] for k in CONV_PARAMS))

def test_neuron_conv_switch(monkeypatch):
  """FZ_NEURON_CONV decides whether a NEURON tier offers "conv" (read when the device is created)."""
  from fusezap.device.neuron import NEURONDevice
  dev = Device[TIER]
  for v, want in (("0", False), ("1", True)):
    monkeypatch.setenv("FZ_NEURON_CONV", v)
    d = type(dev).__new__(type(dev))
    NEURONDevice.__init__(d, dev.name, dev.renderer) if type(dev) is NEURONDevice else type(dev).__init__(d, dev.name)
    assert ("conv" in d.custom_kernels) == want

# ---------------------------------------------------------------- long attention (klein S = 4608 D = 128, VAE S = 16384 D = Dv = 512)

ATTN_CASES = {"klein_4608_d128": (2, 128, 4608, 128, 128), "vae_16384_d512": (1, 256, 16384, 512, 512)}   # heads, L, S, D, Dv

@pytest.mark.parametrize("case", list(ATTN_CASES))
def test_long_attention_kernel(case, monkeypatch):
  """softmax(q kᵀ / sqrt(D)) v through the scheduler's "attention" kernel on the tier vs the composed op on the same tier and
  a float64 reference (relative max error <= 1e-5: the exact exp, float32 PSUM over D chunks and 128-key chunks)."""
  Hh, L, S, D, Dv = ATTN_CASES[case]
  q, k, v = rnd(1, Hh, L, D), rnd(1, Hh, S, D), rnd(1, Hh, S, Dv)
  fn = lambda q, k, v: q.scaled_dot_product_attention(k, v)
  c, comp, ks = _both(fn, monkeypatch, q, k, v)
  assert "fz_attention" in ks, ks
  qd, kd, vd = (a.astype(np.float64) for a in (q, k, v))
  s = qd @ kd.transpose(0, 1, 3, 2) / np.sqrt(D)
  e = np.exp(s - s.max(-1, keepdims=True))
  ref = (e / e.sum(-1, keepdims=True)) @ vd
  rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
  assert rel(c, ref) <= 1e-5 and rel(comp, ref) <= 1e-5, (rel(c, ref), rel(comp, ref))
  np.testing.assert_allclose(c, comp, rtol=1e-4, atol=1e-5)
