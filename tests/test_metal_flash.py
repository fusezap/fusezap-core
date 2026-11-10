"""I81 D8: METAL flash attention for long sequences / wide heads (codegen/metal.py render_attention_flash) and the
implicit-GEMM conv kernel (render_conv) behind the scheduler's custom-kernel hook (schedule._match_conv), each against
the composed reference (FZ_CUSTOM=0) and torch. Timings: scripts/diff_metal_bench.py."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from fusezap import Tensor, helpers
from fusezap.device import Device

METAL = Device.available("METAL")
needs_metal = pytest.mark.skipif(not METAL, reason="needs Metal")
R = np.random.default_rng(11)

@pytest.fixture(autouse=True)
def _env(monkeypatch):
  helpers.getenv.cache_clear()
  for k in ("FZ_INTERP", "FZ_CUSTOM", "FZ_BEAM", "FZ_OPT", "FZ_METAL_FLASH", "FZ_METAL_FLASH_CFG", "FZ_METAL_CONV", "FZ_METAL_CONV_CFG"):
    monkeypatch.delenv(k, raising=False)
  yield
  helpers.getenv.cache_clear()

def rnd(*shape, s: float = 1.0) -> np.ndarray: return (R.standard_normal(shape) * s).astype(np.float32)
def names(t: Tensor) -> list[str]:
  from fusezap.schedule import create_schedule
  return [si.ast[0].kernel.name if si.kind == "custom" else si.kind for si in create_schedule([t.lazydata]) if si.kind != "copy"]
def rel(a: np.ndarray, b: np.ndarray) -> float: return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))

def custom_and_composed(fn, dev: str, monkeypatch, *arrs):
  """(custom output, composed output, kernel names with custom kernels on) of fn on fresh tensors."""
  outs = []
  for flag in ("1", "0"):
    monkeypatch.setenv("FZ_CUSTOM", flag); helpers.getenv.cache_clear()
    if flag == "1": ks = names(fn(*[Tensor(a, device=dev) for a in arrs]))
    outs.append(fn(*[Tensor(a, device=dev) for a in arrs]).numpy())
  monkeypatch.delenv("FZ_CUSTOM"); helpers.getenv.cache_clear()
  return outs[0], outs[1], ks

# ---------------------------------------------------------------- flash attention

def _sdpa(q, k, v): return q.scaled_dot_product_attention(k, v)

@needs_metal
@pytest.mark.parametrize("shape", [
  (2, 3, 100, 100, 16, 16),     # L, S not multiples of 8 / 32
  (1, 2, 77, 45, 24, 40),       # D != Dv, odd sizes (C = 1)
  (1, 2, 300, 520, 128, 128),   # the transformer's head dim
  (1, 1, 257, 257, 512, 512),   # the VAE mid-block's (one head, 512)
  (1, 1, 64, 33, 1024, 1024),   # wider than any earlier kernel handled in registers
])
def test_flash_hook_matches_composed(shape, monkeypatch):
  from fusezap.codegen.metal import flash_config
  B, H, L, S, D, Dv = shape
  q, k, v = rnd(B, H, L, D), rnd(B, H, S, D), rnd(B, H, S, Dv)
  c, p, ks = custom_and_composed(_sdpa, "METAL", monkeypatch, q, k, v)
  assert ks == ["fz_attention"], ks
  assert flash_config(D, Dv) is not None
  np.testing.assert_allclose(c, p, rtol=1e-4, atol=1e-5)
  ref = torch.nn.functional.scaled_dot_product_attention(*map(torch.tensor, (q, k, v))).numpy()
  np.testing.assert_allclose(c, ref, rtol=1e-4, atol=1e-5)

@needs_metal
def test_flash_strided_inputs_and_causal(monkeypatch):
  """Non-unit head-dim strides (transposed q / k / v views) take the staged fragment path; causal through the direct entry."""
  from fusezap.device import metal as M
  q, k, v = rnd(1, 2, 64, 40), rnd(1, 2, 64, 70), rnd(1, 2, 64, 70)
  fn = lambda q, k, v: q.transpose(-1, -2).scaled_dot_product_attention(k.transpose(-1, -2), v.transpose(-1, -2))
  c, p, ks = custom_and_composed(fn, "METAL", monkeypatch, q, k, v)
  assert ks == ["fz_attention"]
  np.testing.assert_allclose(c, p, rtol=1e-4, atol=1e-5)
  q, k, v = rnd(2, 90, 64), rnd(2, 90, 64), rnd(2, 90, 64)
  got = M.attention(Tensor(q, device="METAL"), Tensor(k, device="METAL"), Tensor(v, device="METAL"), causal=True).numpy()
  ref = torch.nn.functional.scaled_dot_product_attention(*map(torch.tensor, (q, k, v)), is_causal=True).numpy()
  np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)

@needs_metal
def test_flash_bf16_wide_heads_direct():
  """bf16 at head dim 512 (no threadgroup-tile kernel fits): the direct entry runs the f32 flash kernel on f32 copies."""
  from fusezap.device import metal as M
  from fusezap.dtype import dtypes
  q, k, v = rnd(1, 200, 512), rnd(1, 200, 512), rnd(1, 200, 512)
  T = lambda a: Tensor(a, device="METAL").cast(dtypes.bfloat16)
  got = M.attention(T(q), T(k), T(v))
  assert got.dtype == dtypes.bfloat16
  tq = [torch.tensor(a).bfloat16().float() for a in (q, k, v)]
  ref = torch.nn.functional.scaled_dot_product_attention(*tq).numpy()
  np.testing.assert_allclose(got.float().numpy(), ref, rtol=1e-2, atol=1e-2)

@needs_metal
@pytest.mark.parametrize("shape", [(24, 4608, 128), (1, 16384, 512)])
def test_flash_target_lengths(shape, monkeypatch):
  """The klein shapes at 1024^2: joint attention (24 heads x 4608 tokens x 128) and the VAE mid-block (16384 x 512)
  through the hook (the composed graph would materialize 2 GB / 1 GB of scores); vs torch on CPU in float64."""
  H, L, D = shape
  q, k, v = rnd(1, H, L, D, s=0.5), rnd(1, H, L, D, s=0.5), rnd(1, H, L, D)
  ks = names(_sdpa(*[Tensor(a, device="METAL") for a in (q, k, v)]))
  assert ks == ["fz_attention"], ks
  got = _sdpa(*[Tensor(a, device="METAL") for a in (q, k, v)]).numpy()
  hs = range(0, H, max(1, H // 3))   # a few heads against float64 (the reference is the slow part)
  for h in hs:
    ref = torch.nn.functional.scaled_dot_product_attention(*[torch.tensor(a[:, h:h + 1]).double() for a in (q, k, v)]).numpy()
    assert rel(got[:, h:h + 1], ref) < 1e-5, (h, rel(got[:, h:h + 1], ref))

# ---------------------------------------------------------------- conv

def conv2d_composed(x: Tensor, weight: Tensor, stride=1, padding=0, dilation=1) -> Tensor:
  """Tensor.conv2d composition (I81 D2), one sample: im2col as kh*kw shifted SHRINK views
  of one PAD, stacked to (C*kh*kw, Ho*Wo), weight (O, C*kh*kw) @ col."""
  _, C, H, W = x.shape
  O, _, kh, kw = weight.shape
  pair = lambda v: (v, v) if isinstance(v, int) else tuple(v)
  (sh, sw), (dh, dw), (ph, pw) = pair(stride), pair(dilation), pair(padding)
  Ho, Wo = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1, (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
  xp = x.pad(((0, 0), (0, 0), (ph, ph), (pw, pw)))
  cols = []
  for i in range(kh):
    for j in range(kw):
      v = xp.shrink(((0, 1), (0, C), (i * dh, i * dh + sh * (Ho - 1) + 1), (j * dw, j * dw + sw * (Wo - 1) + 1)))
      if sh > 1 or sw > 1: v = v[:, :, ::sh, ::sw]
      cols.append(v.reshape(C, Ho * Wo))
  col = Tensor.stack(cols, axis=1).reshape(C * kh * kw, Ho * Wo)
  return (weight.reshape(O, C * kh * kw) @ col).reshape(1, O, Ho, Wo)

CONV_CASES = {   # (C, H, W, O, kh, kw, stride, padding, dilation)
  "3x3_s1_p1": (4, 9, 11, 5, 3, 3, 1, 1, 1),
  "3x3_s2_p1": (3, 10, 9, 6, 3, 3, 2, 1, 1),
  "3x3_s1_p0": (2, 7, 7, 3, 3, 3, 1, 0, 1),
  "5x5_p2": (3, 12, 10, 4, 5, 5, 1, 2, 1),
  "1x3_p01": (5, 6, 8, 7, 1, 3, 1, (0, 1), 1),
  "dilated": (4, 13, 13, 9, 3, 3, 1, 2, 2),
  "wide_tiles": (40, 33, 47, 70, 3, 3, 1, 1, 1),   # M > 64, N not a multiple of the 128-pixel tile, C > 32
  "rgb_out": (37, 20, 24, 3, 3, 3, 1, 1, 1),        # M <= 8: the small-M tile
}

def _conv_devices() -> list[str]: return ["NPY"] + (["METAL"] if METAL else [])

@pytest.mark.parametrize("dev", _conv_devices())
@pytest.mark.parametrize("case", list(CONV_CASES))
def test_conv_kernel_matches_composed(dev, case, monkeypatch):
  C, H, W, O, kh, kw, s, p, d = CONV_CASES[case]
  x, w = rnd(1, C, H, W), rnd(O, C, kh, kw, s=0.2)
  fn = lambda x, w: conv2d_composed(x.contiguous(), w, s, p, d)
  c, comp, ks = custom_and_composed(fn, dev, monkeypatch, x, w)
  assert "fz_conv" in ks and "fz_matmul" not in ks, ks
  np.testing.assert_allclose(c, comp, rtol=1e-4, atol=1e-5)
  ref = torch.nn.functional.conv2d(torch.tensor(x), torch.tensor(w), stride=s, padding=p, dilation=d).numpy()
  np.testing.assert_allclose(c, ref, rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize("dev", _conv_devices())
def test_conv_lazy_input_and_fallbacks(dev, monkeypatch):
  """A computed (unrealized) input is X itself (materialized once, never im2col'd); inputs that are views of a larger
  buffer (a batch slice, an upsample's expand) are not provably im2col of a contiguous X and stay composed, correctly."""
  x, w = rnd(1, 6, 10, 10), rnd(8, 6, 3, 3, s=0.2)
  fn = lambda x, w: conv2d_composed((x * 0.5).silu(), w, 1, 1)
  c, comp, ks = custom_and_composed(fn, dev, monkeypatch, x, w)
  assert "fz_conv" in ks, ks
  np.testing.assert_allclose(c, comp, rtol=1e-4, atol=1e-5)
  xb = rnd(2, 6, 10, 10)
  fn = lambda x, w: conv2d_composed(x[1:2], w, 1, 1)                                   # a batch slice: a view
  c, comp, ks = custom_and_composed(fn, dev, monkeypatch, xb, w)
  assert "fz_conv" not in ks
  np.testing.assert_allclose(c, comp, rtol=1e-4, atol=1e-5)
  up = lambda x: x.reshape(1, 6, 10, 1, 10, 1).expand(1, 6, 10, 2, 10, 2).reshape(1, 6, 20, 20)
  fn = lambda x, w: conv2d_composed(up(x), w, 1, 1)                                     # nearest 2x upsample, then conv
  c, comp, ks = custom_and_composed(fn, dev, monkeypatch, x, w)
  assert "fz_conv" not in ks
  np.testing.assert_allclose(c, comp, rtol=1e-4, atol=1e-5)
  fn = lambda x, w: conv2d_composed(up(x).contiguous(), w, 1, 1)                        # ... made contiguous: matched
  c, comp, ks = custom_and_composed(fn, dev, monkeypatch, x, w)
  assert "fz_conv" in ks
  np.testing.assert_allclose(c, comp, rtol=1e-4, atol=1e-5)

@pytest.mark.parametrize("dev", _conv_devices())
def test_conv_lookalikes_stay_composed(dev, monkeypatch):
  """Stacks of views that are not one input's im2col: taps in the wrong order, two different inputs."""
  x, x2, w = rnd(1, 3, 6, 6), rnd(1, 3, 6, 6), rnd(4, 3, 3, 3, s=0.2)
  def shuffled(x, w):
    xp = x.contiguous().pad(((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = [xp.shrink(((0, 1), (0, 3), (i, i + 6), (j, j + 6))).reshape(3, 36) for i in range(3) for j in range(3)]
    cols[0], cols[1] = cols[1], cols[0]
    return w.reshape(4, 27) @ Tensor.stack(cols, axis=1).reshape(27, 36)
  c, comp, ks = custom_and_composed(shuffled, dev, monkeypatch, x, w)
  assert "fz_conv" not in ks
  np.testing.assert_allclose(c, comp, rtol=1e-4, atol=1e-5)
  def two(x, x2, w):
    xp, xq = x.contiguous().pad(((0, 0), (0, 0), (1, 1), (1, 1))), x2.contiguous().pad(((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = [(xp if (i + j) % 2 else xq).shrink(((0, 1), (0, 3), (i, i + 6), (j, j + 6))).reshape(3, 36) for i in range(3) for j in range(3)]
    return w.reshape(4, 27) @ Tensor.stack(cols, axis=1).reshape(27, 36)
  c, comp, ks = custom_and_composed(two, dev, monkeypatch, x, x2, w)
  assert "fz_conv" not in ks
  np.testing.assert_allclose(c, comp, rtol=1e-4, atol=1e-5)

@pytest.mark.parametrize("dev", _conv_devices())
def test_tensor_conv2d_hits_the_kernel(dev, monkeypatch):
  """Integration with D2: Tensor.conv2d's composition is what _match_conv recognizes (VAE shapes: 3x3 s1 p1), also on
  view inputs (conv2d makes its input contiguous): the VAE upsampler's interpolate (an expand), and a batch of 2 (each
  sample a slice). Every conv is the kernel and no im2col GEMM is left."""
  F = torch.nn.functional
  x, x2, w, b = rnd(1, 8, 12, 12), rnd(2, 8, 12, 12), rnd(16, 8, 3, 3, s=0.2), rnd(16)
  cases = [(lambda x, w, b: (x * 1.0).conv2d(w, b, padding=1), x, lambda t: t),                       # a computed input
           (lambda x, w, b: x.conv2d(w, b, padding=1), x, lambda t: t),                               # a realized input
           (lambda x, w, b: x.interpolate(scale_factor=2).conv2d(w, b, padding=1), x,                 # nearest 2x upsample
            lambda t: F.interpolate(t, scale_factor=2)),
           (lambda x, w, b: x.conv2d(w, b, padding=1), x2, lambda t: t)]                              # per-sample slices
  for fn, xin, pre in cases:
    c, comp, ks = custom_and_composed(fn, dev, monkeypatch, xin, w, b)
    assert "fz_conv" in ks and "fz_matmul" not in ks, ks
    np.testing.assert_allclose(c, comp, rtol=1e-4, atol=1e-5)
    ref = F.conv2d(pre(torch.tensor(xin)), torch.tensor(w), torch.tensor(b), padding=1).numpy()
    np.testing.assert_allclose(c, ref, rtol=1e-4, atol=1e-4)
