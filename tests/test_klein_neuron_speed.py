"""I93: the klein speed switches on a NEURON tier (NKISHIM), each checked against the path it replaces.

FZ_KLEIN_NEURON_ROPE=1 (fusezap/codegen/nki_klein.RopeSpec): bitwise the composed RoPE + attention layout, kernel by kernel
and through the whole tiny transformer (bf16 and float32).
"""
import math, os
import numpy as np
import pytest
from fusezap.tensor import Tensor
from fusezap.dtype import dtypes
from fusezap.serve.models.flux2 import Flux2Transformer, rope_tables, text_ids, latent_ids
from test_flux2_neuron import tiny_weights, inputs

DEV = "NKISHIM"

@pytest.fixture
def env(monkeypatch):
  def set_(**kw):
    for k, v in kw.items(): monkeypatch.setenv(k, str(v))
  return set_

def _bits(a: np.ndarray) -> np.ndarray: return np.ascontiguousarray(a).view(np.uint32 if a.dtype == np.float32 else np.uint16)

@pytest.mark.parametrize("dt,parts,H", [(dtypes.bfloat16, (8, 16), 2), (dtypes.bfloat16, (136,), 3), (dtypes.float32, (8, 150), 4),
                                        (dtypes.bfloat16, (130, 20), 5)])
def test_rope_spec_bitwise(env, dt, parts, H):
  """RopeSpec == the composed path (cat, _rope, permute, sdpa's .float()) bit for bit, rows straddling tiles and inputs."""
  D, B = 128, 1
  S = sum(parts)
  rng = np.random.default_rng(3)
  tr = Flux2Transformer.__new__(Flux2Transformer)
  tr.device, tr.dtype = DEV, dt
  xs = [Tensor((3 * rng.standard_normal((B, s, H, D))).astype(np.float32), device=DEV).cast(dt).realize() for s in parts]
  ids = np.concatenate([text_ids(parts[0]), latent_ids(1, S - parts[0])]) if len(parts) == 2 else latent_ids(1, S)
  c, s_ = rope_tables(ids, (32, 32, 32, 32), 2000.0)
  cos, sin = Tensor(c, device=DEV).realize(), Tensor(s_, device=DEV).realize()
  x = xs[0] if len(xs) == 1 else xs[0].cat(xs[1], axis=1)
  ref = tr._rope(x, cos, sin).permute(0, 2, 1, 3)
  ref = (ref.float() if dt != dtypes.float32 else ref).numpy()
  got = tr._rope_heads(xs, cos, sin).numpy()
  assert got.shape == ref.shape == (B, H, S, D) and got.dtype == np.float32
  assert np.array_equal(_bits(got), _bits(ref)), np.abs(got - ref).max()

@pytest.mark.parametrize("dt", [dtypes.bfloat16, dtypes.float32])
def test_klein_rope_switch_bitwise_tiny(env, dt):
  """The whole tiny transformer with FZ_KLEIN_NEURON_ROPE=1 == without, bit for bit; and fewer launches."""
  from fusezap.device import Device
  c, W = tiny_weights()
  x, e, T, hw = inputs(T=8, g=4)
  dev = Device[DEV]
  def run(flag):
    env(FZ_KLEIN_NEURON_ROPE=flag)
    tr = Flux2Transformer(c, W, dt, DEV)
    n, orig = [0], dev.run
    def count(si):
      n[0] += 1
      return orig(si)
    dev.run = count
    try: out = tr(x, e, 0.7, T, hw)
    finally: dev.run = orig
    return out, n[0]
  ref, n0 = run(0)
  got, n1 = run(1)
  assert np.array_equal(_bits(got), _bits(ref)), np.abs(got - ref).max()
  assert n1 < n0, (n1, n0)

@pytest.mark.parametrize("M,N,K,G", [(300, 640, 256, 4), (512, 512, 384, 2), (1000, 520, 200, 3), (256, 128, 1100, 4)])
def test_matmul_bf16g_bitwise(env, M, N, K, G):
  """MatmulBF16G == MatmulBF16 bit for bit (ragged row blocks, a partial group, ragged K and N)."""
  from fusezap.codegen.nki_bf16 import MatmulBF16
  from fusezap.codegen.nki_klein import MatmulBF16G
  from fusezap.device import nkishim as S
  from fusezap.serve.weights import f32_to_bf16
  rng = np.random.default_rng(5)
  a = rng.standard_normal((M, K)).astype(np.float32)
  w = f32_to_bf16(rng.standard_normal((K, N)).astype(np.float32))
  outs = []
  for sp in (MatmulBF16(M, N, K), MatmulBF16G(M, N, K, G)):
    c = np.zeros(M * N, np.float32)
    S.execute(sp.render(), [c, a.reshape(-1).copy(), w.reshape(-1).copy()])
    outs.append(c)
  assert np.array_equal(outs[0].view(np.uint32), outs[1].view(np.uint32))

def test_klein_mmg_switch_bitwise_tiny(env):
  """The tiny transformer at 256+ rows with FZ_KLEIN_NEURON_MMG=4 == without, bit for bit (bf16, NKISHIM)."""
  from fusezap.device import Device
  c, W = tiny_weights()
  x, e, T, hw = inputs(T=16, g=16)   # 256 image rows, 272 joint rows
  dev = Device[DEV]
  def run(g):
    env(FZ_KLEIN_NEURON_MMG=g)
    names, orig = [], dev.run
    def rec(si):
      names.append(getattr(getattr(si.ast[0], "spec", None), "name", "") if si.ast else "")
      return orig(si)
    dev.run = rec
    try: out = Flux2Transformer(c, W, dtypes.bfloat16, DEV)(x, e, 0.7, T, hw)
    finally: dev.run = orig
    return out, names
  ref, n0 = run(0)
  got, n1 = run(4)
  assert any(n.endswith("_g4") or "_bf16_g" in n for n in n1) and not any("_bf16_g" in n for n in n0)
  assert np.array_equal(_bits(got), _bits(ref)), np.abs(got - ref).max()
