"""Composite conformance checks a decoder model needs that tests/test_ops.py does not cover at the Tensor level:
RMSNorm and RoPE as llama.py builds them, llama.py's GQA causal attention graph, and on-device greedy / temperature
sampling. Every device tier runs them (conftest `device`), so a new backend's port is checked against them. References are float64 numpy over the inputs as the device sees them (bf16/f16-rounded)."""
from __future__ import annotations
import math
from types import SimpleNamespace
import numpy as np
import pytest
from fusezap import Tensor, dtypes
from fusezap.serve.models.llama import Llama

DTS = ["float32", "float16", "bfloat16"]
FZ = {"float32": dtypes.float32, "float16": dtypes.float16, "bfloat16": dtypes.bfloat16}

def _round(a: np.ndarray, dt: str) -> np.ndarray:
  """float32 host values as the device holds them in dt (float32 again)."""
  a = np.ascontiguousarray(a, np.float32)
  if dt == "float16": return a.astype(np.float16).astype(np.float32)
  if dt == "bfloat16":
    from conftest import bf16_to_f32, f32_to_bf16
    return bf16_to_f32(f32_to_bf16(a))
  return a

def _dev(a: np.ndarray, dt: str) -> Tensor:
  t = Tensor(np.ascontiguousarray(a, np.float32))
  return t if dt == "float32" else t.cast(FZ[dt])

def _host(t: Tensor) -> np.ndarray: return t.float().numpy() if t.dtype != dtypes.float32 else t.numpy()

def _rng(seed: int = 0): return np.random.default_rng(seed)

@pytest.mark.parametrize("dt", DTS)
@pytest.mark.parametrize("shape", [(4, 64), (2, 3, 128), (1, 1, 896)])
def test_rmsnorm(device, assert_close, shape, dt):
  """Llama._rmsnorm: x * rsqrt(mean(x^2) + eps) in float32, cast back, times the weight."""
  r = _rng()
  x, w = _round(r.standard_normal(shape) * 3, dt), _round(1 + 0.1 * r.standard_normal(shape[-1:]), dt)
  m = SimpleNamespace(cfg=SimpleNamespace(rms_eps=1e-6))
  out = _host(Llama._rmsnorm(m, _dev(x, dt), _dev(w, dt)))
  x64 = x.astype(np.float64)
  ref = w * (x64 / np.sqrt((x64 * x64).mean(-1, keepdims=True) + 1e-6))
  assert_close(out, ref.astype(np.float32), dt, reduce=True)

@pytest.mark.parametrize("dt", DTS)
@pytest.mark.parametrize("B,H,T,hd,start", [(1, 2, 5, 64, 0), (2, 4, 3, 128, 1000), (1, 1, 1, 64, 4095)])
def test_rope(device, assert_close, B, H, T, hd, start, dt):
  """Llama._rope (rotate-half form) with llama.py's host cos/sin tables at positions [start, start + T)."""
  from fusezap.serve.models.llama import LlamaConfig, rope_tables
  c = LlamaConfig(hd * H, 4 * hd, 1, H, H, hd, 32, 1e-6, 1e6, None, False, start + T, [0], 0, "llama")
  cos, sin = rope_tables(c, start + T)
  cos, sin = _round(cos[start:start + T], dt).reshape(1, 1, T, hd // 2), _round(sin[start:start + T], dt).reshape(1, 1, T, hd // 2)
  x = _round(_rng(1).standard_normal((B, H, T, hd)), dt)
  out = _host(Llama._rope(None, _dev(x, dt), _dev(cos, dt), _dev(sin, dt)))
  x1, x2 = x[..., :hd // 2].astype(np.float64), x[..., hd // 2:].astype(np.float64)
  ref = np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
  assert_close(out, ref.astype(np.float32), dt, reduce=True)

@pytest.mark.parametrize("dt", DTS)
@pytest.mark.parametrize("B,Hq,Hkv,T,S,hd", [(1, 4, 2, 5, 5, 64), (2, 8, 2, 3, 11, 64), (1, 14, 2, 1, 33, 64)])
def test_gqa_attention(device, assert_close, B, Hq, Hkv, T, S, hd, dt):
  """llama.py's composed attention graph: grouped-query heads, position mask (key j visible to query row l iff
  j <= S - T + l), scaled scores, softmax, times V. T < S is a chunked prefill / decode against a filled cache."""
  r, G = _rng(2), Hq // Hkv
  q, k, v = (_round(r.standard_normal(s) * 0.5, dt) for s in ((B, Hq, T, hd), (B, Hkv, S, hd), (B, Hkv, S, hd)))
  pos = np.arange(S, dtype=np.int32)
  qpos = Tensor(pos[S - T:].reshape(T, 1))
  scale = 1.0 / math.sqrt(hd)
  Q, K, V = _dev(q, dt), _dev(k, dt), _dev(v, dt)
  mask = Tensor(pos).reshape(1, S) <= qpos
  s = (Q.reshape(B, Hkv, G, T, hd) @ K.reshape(B, Hkv, 1, S, hd).transpose(-1, -2)) * scale
  o = mask.where(s, float("-inf")).softmax(-1) @ V.reshape(B, Hkv, 1, S, hd)
  out = _host(o.reshape(B, Hq, T, hd))
  q64, k64, v64 = (a.astype(np.float64) for a in (q, k, v))
  sc = np.einsum("bhgtd,bhsd->bhgts", q64.reshape(B, Hkv, G, T, hd), k64) * scale
  sc = np.where(pos[None, :] <= pos[S - T:, None], sc, -np.inf)
  p = np.exp(sc - sc.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
  ref = np.einsum("bhgts,bhsd->bhgtd", p, v64).reshape(B, Hq, T, hd)
  assert_close(out, ref.astype(np.float32), dt, reduce=True)

@pytest.mark.parametrize("dt", DTS)
@pytest.mark.parametrize("rows,vocab,temp", [(1, 1000, 1.0), (4, 32000, 0.7), (2, 151936, 1.3)])
def test_sampling(device, assert_close, rows, vocab, temp, dt):
  """On-device greedy (argmax of each row's logits: first index of the maximum) and the temperature distribution a
  sampler draws from, vs fusezap.serve.sampling.softmax on the host."""
  from fusezap.serve.sampling import softmax
  lg = _round(_rng(3).standard_normal((rows, vocab)) * 4, dt)
  t = _dev(lg, dt)
  np.testing.assert_array_equal(t.argmax(-1).numpy(), lg.argmax(-1).astype(np.int32))
  probs = _host((t.float() / temp).softmax(-1))
  assert_close(probs, softmax(lg.astype(np.float64) / temp).astype(np.float32), "float32", reduce=True)
