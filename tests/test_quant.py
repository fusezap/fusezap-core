"""P4-QUANT: weight-only int8 / int4 / fp8 quantization, AWQ / GPTQ import (fusezap/serve/quant.py) and the perplexity
bench's pieces.

Layer tests run on FZ_DEVICE (default METAL when available, else CPU). Model tests need Qwen2.5-0.5B-Instruct in the HF
cache; the HF comparison also needs torch + transformers (reference only).
"""
from __future__ import annotations
import importlib.util, os
import numpy as np
import pytest
from fusezap.tensor import Tensor
from fusezap.dtype import dtypes
from fusezap import helpers
from fusezap.serve.quant import (QuantLinear, QuantWeight, quantize_weight, dequantize_weight, pack_int4, unpack_int4,
                                 quantize_model, weight_bytes, import_checkpoint, LINEARS, fp8_e4m3_encode, fp8_e4m3_decode,
                                 gptq_weight, awq_weight, checkpoint_format)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODES = ["int8", "int4", "int4-sym", "fp8"]

def _device() -> str:
  from fusezap.device import Device
  return os.environ.get("FZ_DEVICE") or ("METAL" if Device.available("METAL") else "CPU")

def _model_path() -> str | None:
  try:
    from fusezap.serve.weights import resolve_model
    return resolve_model(MODEL, allow_download=False)
  except Exception: return None

needs_model = pytest.mark.skipif(_model_path() is None, reason=f"{MODEL} not in the HF cache")

def _bench():
  spec = importlib.util.spec_from_file_location("bench_ppl", os.path.join(os.path.dirname(__file__), "..", "bench", "bench_ppl.py"))
  mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore[union-attr]
  return mod

@pytest.fixture
def custom(monkeypatch):
  """custom(on): toggle the scheduler's library-kernel matching (FZ_CUSTOM, read through the cached getenv)."""
  def set_(on: bool):
    monkeypatch.setenv("FZ_CUSTOM", "1" if on else "0"); helpers.getenv.cache_clear()
  yield set_
  monkeypatch.delenv("FZ_CUSTOM", raising=False); helpers.getenv.cache_clear()

def _w(O: int, I: int, seed: int = 0) -> np.ndarray:
  r = np.random.default_rng(seed)
  w = r.standard_normal((O, I)).astype(np.float32) * 0.05
  w[0, :7] *= 20   # outliers
  w[1] = 0         # an all-zero row (scale must not be 0)
  return w

# ---------------------------------------------------------------- host quantization

def test_int8_per_channel_roundtrip():
  w = _w(64, 200)
  qw = quantize_weight(w, "int8")
  assert qw.q.dtype == np.int8 and qw.scale.shape == (64, 1) and qw.zero is None
  assert np.abs(qw.q.astype(np.int32)).max() <= 127
  np.testing.assert_allclose(qw.scale[:, 0], np.where(np.abs(w).max(1) == 0, 1, np.abs(w).max(1) / 127), rtol=1e-6)
  err = np.abs(dequantize_weight(qw) - w)
  assert (err <= qw.scale * 0.5 + 1e-7).all()
  assert (dequantize_weight(qw)[1] == 0).all()

def test_int4_packing_layout_roundtrip():
  r = np.random.default_rng(1)
  q = r.integers(0, 16, (5, 256)).astype(np.uint8)
  p = pack_int4(q)
  assert p.shape == (5, 128) and p.dtype == np.uint8
  assert p[0, 0] == q[0, 0] | (q[0, 64] << 4) and p[0, 64] == q[0, 128] | (q[0, 192] << 4)  # per-group half split
  np.testing.assert_array_equal(unpack_int4(p), q)

@pytest.mark.parametrize("mode", ["int4", "int4-sym"])
@pytest.mark.parametrize("I", [256, 200, 130])
def test_int4_group_quant_error_bound(mode, I):
  w = _w(48, I, 2)
  qw = quantize_weight(w, mode)
  G = -(-I // 128)
  assert qw.q.shape == (48, G * 64) and qw.scale.shape == (48, G)
  assert (qw.zero is None) == (mode == "int4-sym")
  if qw.zero is not None: assert ((qw.zero >= 0) & (qw.zero <= 15) & (qw.zero == np.rint(qw.zero))).all()
  dq = dequantize_weight(qw)
  assert dq.shape == w.shape
  s = np.repeat(qw.scale, 128, axis=1)[:, :I]
  assert (np.abs(dq - w) <= s * 0.5 * (1 + 1e-5) + 1e-7).all()   # round to nearest, nothing clipped
  assert (dq[1] == 0).all()

def test_modes_and_errors(tmp_path):
  with pytest.raises(ValueError): quantize_weight(_w(4, 128), "int3")
  assert quantize_weight(_w(4, 128), "int4-asym").zero is not None   # alias of int4
  assert quantize_weight(_w(4, 128), "fp8-e4m3").mode == "fp8"
  with pytest.raises(ValueError, match="not an AWQ/GPTQ"): checkpoint_format({})
  with pytest.raises(NotImplementedError, match="8 bits"): checkpoint_format({"quantization_config": {"quant_method": "gptq", "bits": 8}})
  with pytest.raises(NotImplementedError, match="gemm"): checkpoint_format({"quantization_config": {"quant_method": "awq", "bits": 4, "version": "gemv"}})
  assert checkpoint_format({"quantization_config": {"quant_method": "awq", "bits": 4, "version": "GEMM"}})[0] == "awq"
  (tmp_path / "config.json").write_text('{"model_type": "qwen2"}')
  with pytest.raises(ValueError, match="not an AWQ/GPTQ"): import_checkpoint(str(tmp_path))

# ---------------------------------------------------------------- fp8 (e4m3, per-tensor scale)

def test_fp8_codec():
  """e4m3fn: every finite code round-trips, max 448, min subnormal 2^-9, ties to even, saturation; and bit-identical to
  torch.float8_e4m3fn when torch is installed (reference only)."""
  codes = np.array([c for c in range(256) if c & 0x7F != 0x7F], np.uint8)
  v = fp8_e4m3_decode(codes)
  np.testing.assert_array_equal(fp8_e4m3_encode(v), codes)
  assert v.max() == 448 and fp8_e4m3_decode(np.uint8(1)) == 2.0 ** -9 and fp8_e4m3_decode(np.uint8(8)) == 2.0 ** -6
  assert np.isnan(fp8_e4m3_decode(np.array([0x7F, 0xFF], np.uint8))).all()
  assert fp8_e4m3_encode(np.array([1e6, -1e6, 1.0625, 1.1875, 2.0 ** -10, 3 * 2.0 ** -10]).astype(np.float32)).tolist() == \
    [0x7E, 0xFE, 0x38, 0x3A, 0x00, 0x02]   # saturate; 1+1/16 -> 1 (even), 1+3/16 -> 1+1/4 (even); 2^-10 -> 0 (even); 1.5*2^-9 -> 2*2^-9 (even)
  x = np.random.default_rng(0).standard_normal(50000).astype(np.float32) * 50
  pos = fp8_e4m3_decode(np.arange(127, dtype=np.uint8))
  x = np.concatenate([x, (pos[1:] + pos[:-1]) / 2, -(pos[1:] + pos[:-1]) / 2]).astype(np.float32)   # every midpoint
  try: import torch
  except ImportError: return
  ref = torch.tensor(np.clip(x, -448, 448)).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
  np.testing.assert_array_equal(fp8_e4m3_encode(x), ref)

def test_fp8_quant_per_tensor():
  w = _w(64, 200)
  qw = quantize_weight(w, "fp8")
  assert qw.q.dtype == np.uint8 and qw.q.shape == w.shape and qw.scale.shape == (1, 1) and qw.zero is None
  assert np.isclose(qw.scale[0, 0], np.abs(w).max() / 448, rtol=1e-6)
  assert qw.q.max() & 0x7F <= 0x7E and (qw.q & 0x7F).max() == 0x7E    # the max |w| maps to 448, nothing is NaN
  dq = dequantize_weight(qw)
  normal = np.abs(w) >= qw.scale[0, 0] * 2.0 ** -6
  assert (np.abs(dq - w)[normal] <= np.abs(w)[normal] * 2.0 ** -4 * (1 + 1e-5)).all()   # half an ulp of a 3-bit mantissa
  assert (np.abs(dq - w)[~normal] <= qw.scale[0, 0] * 2.0 ** -10 * (1 + 1e-5)).all()     # half a subnormal step
  assert (dq[1] == 0).all()
  assert quantize_weight(np.zeros((4, 8), np.float32), "fp8").scale[0, 0] == 1

# ---------------------------------------------------------------- device layer vs dequantized reference matmul

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dt", [dtypes.float32, dtypes.float16], ids=["f32", "f16"])
@pytest.mark.parametrize("shape", [(300, 896), (72, 200)], ids=["896", "pad200"])
@pytest.mark.parametrize("use_custom", [False, True], ids=["fused", "lib"])
def test_quant_linear_matches_dequant_reference(mode, dt, shape, use_custom, custom):
  """x.linear(QuantLinear) == x @ dequantize_weight(...).T + b (numpy), batched input, with bias. fused: composed
  kernels (decode path, FZ_CUSTOM=0); lib: the library GEMM where it matches (prefill path)."""
  custom(use_custom)
  O, I = shape
  npdt = np.float16 if dt == dtypes.float16 else np.float32
  w, r = _w(O, I, 3), np.random.default_rng(4)
  qw = quantize_weight(w, mode)
  ql = QuantLinear(qw, dt, _device())
  ref_w = dequantize_weight(qw, dt)
  np.testing.assert_array_equal(ql.numpy().astype(np.float32), ref_w)   # device dequantization is exact
  x = r.standard_normal((2, 5, I)).astype(npdt)
  b = r.standard_normal(O).astype(npdt)
  y = Tensor(x, device=_device(), dtype=dt).linear(ql, Tensor(b, device=_device(), dtype=dt)).numpy()
  ref = x.astype(np.float32) @ ref_w.T + b.astype(np.float32)
  assert y.dtype == npdt and y.shape == (2, 5, O)
  tol = 2e-3 if dt == dtypes.float16 else 1e-5
  np.testing.assert_allclose(y.astype(np.float32), ref, rtol=tol, atol=tol * np.abs(ref).max())
  np.testing.assert_allclose(ql(Tensor(x, device=_device(), dtype=dt)).numpy().astype(np.float32), x.astype(np.float32) @ ref_w.T,
                             rtol=tol, atol=tol * np.abs(ref).max())

@pytest.mark.parametrize("mode", MODES)
def test_dequant_fused_into_gemv(mode, custom):
  """Decode path (FZ_CUSTOM=0): the GEMV is one kernel that reads the int codes directly; no dequantized weight buffer."""
  from fusezap.schedule import create_schedule
  custom(False)
  ql = QuantLinear.from_float(_w(256, 384), mode, dtypes.float16, _device())
  x = Tensor(np.ones((1, 1, 384), np.float16), device=_device(), dtype=dtypes.float16).realize()
  sched = [si for si in create_schedule([x.linear(ql).lazydata]) if si.kind != "copy"]
  assert len(sched) == 1, [si.kind for si in sched]
  code_dt = dtypes.int8 if mode == "int8" else dtypes.uint8   # fp8: e4m3 bits in uint8
  assert any(b.dtype == code_dt and b.size == ql.q.numel() for b in sched[0].bufs[1:])
  assert all(b.dtype == code_dt or b.size < 256 * 384 // 2 for b in sched[0].bufs)   # no full-size float weight

def test_quant_linear_host_and_bytes():
  w = _w(128, 256)
  for mode, nbytes in (("int8", 128 * 256 + 128 * 4), ("int4", 128 * 128 + 2 * 128 * 2 * 4), ("int4-sym", 128 * 128 + 128 * 2 * 4),
                       ("fp8", 128 * 256 + 4)):
    qw = quantize_weight(w, mode)
    ql = QuantLinear(qw, dtypes.float32, _device())
    assert ql.nbytes == nbytes and ql.ndim == 2 and ql.shape == (128, 256)
    h = ql.host()
    np.testing.assert_array_equal(h.q, qw.q)
    np.testing.assert_array_equal(dequantize_weight(h), dequantize_weight(qw))

# ---------------------------------------------------------------- whole model

@pytest.fixture(scope="module")
def models():
  """(f32 reference model, {mode: quantized f32 model})."""
  from fusezap.serve.models import load_model
  kw = dict(dtype=dtypes.float32, device=_device(), max_len=64, prefill_chunk=32)
  base = load_model(MODEL, **kw)
  return base, {m: quantize_model(load_model(MODEL, **kw), m) for m in ("int8", "int4", "fp8")}

def _ids() -> list[int]: return list(np.random.default_rng(5).integers(100, 20000, 40))

def _dequant_twin(qm):
  """A copy of the quantized model's weights as plain Tensors, dequantized in numpy (the reference)."""
  from fusezap.serve.models import load_model
  ref = load_model(MODEL, dtype=dtypes.float32, device=_device(), max_len=64, prefill_chunk=32)
  for lq, lr in zip(qm.layers, ref.layers):
    for n in LINEARS: setattr(lr, n, Tensor(dequantize_weight(getattr(lq, n).host()), device=_device()).realize())
  return ref

@needs_model
@pytest.mark.parametrize("mode", ["int8", "int4", "fp8"])
def test_quantized_model_matches_dequantized_reference(models, mode):
  """Prefill + jitted decode logits of the quantized model == a model holding the numpy-dequantized weights."""
  qm = models[1][mode]
  assert all(isinstance(getattr(l, n), QuantLinear) for l in qm.layers for n in LINEARS)
  ref = _dequant_twin(qm)
  ids = _ids()
  for m in (qm, ref): m.reset()
  np.testing.assert_allclose(qm.prefill([ids]), ref.prefill([ids]), atol=2e-3, rtol=1e-4)
  for t in (11, 12, 13, 14):   # 3+ steps: the decode jit captures, then replays
    np.testing.assert_allclose(qm.decode([[t]]), ref.decode([[t]]), atol=2e-3, rtol=1e-4)
  assert qm._decode_jit.captured is not None
  del ref

@needs_model
def test_quantized_model_close_to_full_precision(models):
  """Teacher-forced top-1 agreement with the unquantized f32 model on its own greedy continuation of an English prompt
  (measured: int8 94%, int4 83%, fp8 100%; asserted >= 90% / 70% / 90%), plus the weight footprint."""
  base, q = models
  from fusezap.serve.tokenizer import Tokenizer
  prompt = Tokenizer.from_pretrained(_model_path()).encode("The history of the Roman Empire spans many centuries. It began when")
  full = prompt + list(base.generate(prompt, 64 - len(prompt), stop=[]))
  def logits(m):
    x = Tensor(m.embed(np.array([full])), device=m.device, dtype=m.dtype)
    return m.forward(x, m.cache.bind(m.cache.state()), 0, len(full), last_only=False).float().numpy()[0]
  n = len(prompt) - 1   # positions predicting the greedy continuation
  R = logits(base).argmax(-1)[n:]
  agree = {k: float((logits(m).argmax(-1)[n:] == R).mean()) for k, m in q.items()}
  assert agree["int8"] >= 0.90 and agree["int4"] >= 0.70 and agree["fp8"] >= 0.90, agree
  b, b8, b4 = weight_bytes(base), weight_bytes(q["int8"]), weight_bytes(q["int4"])
  assert b8["linear"] < 0.27 * b["linear"] and b4["linear"] < 0.15 * b["linear"]   # f32 base: 4 -> ~1 -> ~0.53 bytes/weight
  assert b8["lm_head"] == b["lm_head"]
  assert weight_bytes(q["fp8"])["linear"] < 0.26 * b["linear"]                                # 1 byte/weight
  with pytest.raises(ValueError): quantize_model(q["int8"], "int8")

# ---------------------------------------------------------------- perplexity bench pieces

def test_snappy_and_windows():
  bp = _bench()
  # literal "abcd", then a copy of length 8 at offset 4 (overlapping), then literal "xy"
  stream = bytes([14, 3 << 2]) + b"abcd" + bytes([((8 - 4) << 2) | 1, 4]) + bytes([1 << 2]) + b"xy"
  assert bp.snappy_decompress(stream) == b"abcdabcdabcdxy"
  ws = bp.windows(4096, 1024, 512)
  scored = [(b + f, b + 1024) for b, f in ws]       # target indices within ids[1:]
  assert scored[0] == (0, 1024) and sum(e - s for s, e in scored) == 4096
  cover = np.zeros(4096, int)
  for s, e in scored: cover[s:e] += 1
  assert (cover == 1).all()                          # every prediction scored exactly once
  assert all(f == 0 or f >= 512 for _, f in ws)      # later windows: >= ctx - stride tokens of context

def _wikitext_cached() -> bool:
  import glob
  return bool(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet")))

@pytest.mark.skipif(not _wikitext_cached(), reason="wikitext-2-raw-v1 not in the HF cache")
def test_parquet_reader_wikitext():
  bp = _bench()
  import glob
  fn = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet"))[0]
  rows = bp.read_parquet_strings(fn)
  assert len(rows) == 4358 and rows[0] == "" and rows[1] == " = Robert Boulter = \n"
  assert rows[3].startswith(" Robert Boulter is an English film , television and theatre actor .")

@needs_model
@pytest.mark.skipif(not _wikitext_cached(), reason="wikitext-2-raw-v1 not in the HF cache")
def test_nll_eval_matches_hf(models):
  """The bench's on-device NLL (forward over all positions, log-softmax, target pick) == HF torch f32 cross-entropy."""
  torch = pytest.importorskip("torch")
  tr = pytest.importorskip("transformers")
  bp = _bench()
  from fusezap.serve.tokenizer import Tokenizer
  ids = np.asarray(Tokenizer.from_pretrained(_model_path()).encode(bp.wikitext2_test()[:4000]), np.int64)[:65]
  mine = bp.NLLEval(models[0], 64)(ids)
  hfm = tr.AutoModelForCausalLM.from_pretrained(_model_path(), dtype=torch.float32).eval()
  with torch.no_grad(): lg = hfm(torch.tensor(ids[None, :-1])).logits[0]
  ref = torch.nn.functional.cross_entropy(lg, torch.tensor(ids[1:]), reduction="none").numpy()
  np.testing.assert_allclose(mine, ref, atol=2e-3, rtol=1e-4)

# ---------------------------------------------------------------- AWQ / GPTQ import
# Reference packers and dequantizers written from AutoGPTQ (qlinear_cuda_old.pack / forward) and AutoAWQ (WQLinear_GEMM
# .from_linear) / vLLM's AWQ reverse order, deliberately loop-style and independent of quant.py's vectorized unpack.

def ref_gptq_pack(q: np.ndarray, z: np.ndarray, s: np.ndarray, v2: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """q (in, out) codes, z (G, out) zero points, s (G, out) -> (qweight int32 (in/8, out), qzeros int32 (G, out/8), scales f16)."""
  I, O = q.shape
  qweight = np.zeros((I // 8, O), np.uint32)
  for row in range(I // 8):
    for j in range(8): qweight[row] |= q[8 * row + j].astype(np.uint32) << (4 * j)
  zs = (z if v2 else z - 1).astype(np.uint32) & 0xF
  qzeros = np.zeros((z.shape[0], O // 8), np.uint32)
  for col in range(O // 8):
    for j in range(8): qzeros[:, col] |= zs[:, 8 * col + j] << (4 * j)
  return qweight.view(np.int32), qzeros.view(np.int32), s.astype(np.float16)

def ref_gptq_dequant(qweight, qzeros, scales, g_idx, v2: bool = False) -> np.ndarray:
  """(out, in) float32: scales[g_idx] * (weight - (zeros + 1)[g_idx])."""
  qw, qz = qweight.view(np.uint32), qzeros.view(np.uint32)
  w = np.stack([(qw >> (4 * j)) & 15 for j in range(8)], 1).reshape(-1, qw.shape[1]).astype(np.float32)
  z = np.stack([(qz >> (4 * j)) & 15 for j in range(8)], 2).reshape(qz.shape[0], -1).astype(np.float32) + (0 if v2 else 1)
  return (scales.astype(np.float32)[g_idx] * (w - z[g_idx])).T

AWQ_PACK_ORDER, AWQ_REVERSE = [0, 2, 4, 6, 1, 3, 5, 7], [0, 4, 1, 5, 2, 6, 3, 7]

def ref_awq_pack(q: np.ndarray, z: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """q (in, out), z / s (G, out) -> (qweight int32 (in, out/8), qzeros int32 (G, out/8), scales f16 (G, out))."""
  def pack(a):
    out = np.zeros((a.shape[0], a.shape[1] // 8), np.uint32)
    for col in range(a.shape[1] // 8):
      for i in range(8): out[:, col] |= a[:, col * 8 + AWQ_PACK_ORDER[i]].astype(np.uint32) << (i * 4)
    return out.view(np.int32)
  return pack(q), pack(z), s.astype(np.float16)

def ref_awq_dequant(qweight, qzeros, scales, group: int) -> np.ndarray:
  def unpack(a):
    a = a.view(np.uint32)
    n = np.stack([(a >> (4 * i)) & 15 for i in range(8)], -1)          # nibble order
    return n[..., AWQ_REVERSE].reshape(a.shape[0], -1).astype(np.float32)
  w, z = unpack(qweight), unpack(qzeros)
  return ((w - np.repeat(z, group, 0)) * np.repeat(scales.astype(np.float32), group, 0)).T

def _rand_codes(I: int, O: int, G: int, seed: int, zlo: int = 0):
  r = np.random.default_rng(seed)
  return (r.integers(0, 16, (I, O)).astype(np.uint8), r.integers(zlo, 16, (G, O)).astype(np.float32),
          (r.random((G, O)) * 0.02 + 1e-3).astype(np.float32))

@pytest.mark.parametrize("v2", [False, True], ids=["v1", "v2"])
def test_gptq_unpack_matches_reference(v2):
  I, O, g = 384, 72, 128
  q, z, s = _rand_codes(I, O, I // g, 6, zlo=1)   # v1 stores z - 1, so z >= 1
  qweight, qzeros, scales = ref_gptq_pack(q, z, s, v2)
  g_idx = np.arange(I) // g
  qw = gptq_weight(qweight, qzeros, scales, g_idx, g, v2)
  assert qw.mode == "int4" and qw.g_idx is None and qw.shape == (O, I)   # trivial g_idx is dropped
  np.testing.assert_array_equal(unpack_int4(qw.q), q.T)
  np.testing.assert_array_equal(dequantize_weight(qw), ref_gptq_dequant(qweight, qzeros, scales, g_idx, v2))
  # symmetric checkpoint (all zero points 8): becomes int4-sym, same values
  qweight, qzeros, scales = ref_gptq_pack(q, np.full_like(z, 8), s, v2)
  qs = gptq_weight(qweight, qzeros, scales, None, g, v2)
  assert qs.mode == "int4-sym" and qs.zero is None
  np.testing.assert_array_equal(dequantize_weight(qs), ref_gptq_dequant(qweight, qzeros, scales, g_idx, v2))
  assert gptq_weight(qweight, qzeros, scales, None, g, v2, sym=False).mode == "int4"

def test_gptq_act_order_unpack():
  I, O, g = 256, 48, 128
  q, z, s = _rand_codes(I, O, I // g, 7, zlo=1)
  g_idx = np.random.default_rng(8).permutation(np.arange(I) // g).astype(np.int32)   # desc_act: groups scattered
  qw = gptq_weight(*ref_gptq_pack(q, z, s), g_idx, g)
  assert qw.g_idx is not None
  np.testing.assert_array_equal(dequantize_weight(qw), ref_gptq_dequant(*ref_gptq_pack(q, z, s), g_idx))

def test_awq_unpack_matches_reference():
  I, O, g = 384, 64, 128
  q, z, s = _rand_codes(I, O, I // g, 9)
  qweight, qzeros, scales = ref_awq_pack(q, z, s)
  qw = awq_weight(qweight, qzeros, scales, g)
  assert qw.mode == "int4" and qw.shape == (O, I)
  np.testing.assert_array_equal(unpack_int4(qw.q), q.T)
  np.testing.assert_array_equal(qw.zero, z.T)
  np.testing.assert_array_equal(dequantize_weight(qw), ref_awq_dequant(qweight, qzeros, scales, g))

@pytest.mark.parametrize("dt", [dtypes.float32, dtypes.float16], ids=["f32", "f16"])
@pytest.mark.parametrize("use_custom", [False, True], ids=["fused", "lib"])
def test_act_order_quant_linear(dt, use_custom, custom):
  """GPTQ act-order (scattered g_idx) on the device: exact dequantization and the matmul vs numpy."""
  custom(use_custom)
  I, O, g = 256, 40, 128
  q, z, s = _rand_codes(I, O, I // g, 10, zlo=1)
  g_idx = np.random.default_rng(11).permutation(np.arange(I) // g).astype(np.int32)
  qw = gptq_weight(*ref_gptq_pack(q, z, s), g_idx, g)
  ql = QuantLinear(qw, dt, _device())
  ref_w = dequantize_weight(qw, dt)
  np.testing.assert_array_equal(ql.numpy().astype(np.float32), ref_w)
  assert ql.host().g_idx is not None and ql.nbytes == O * I // 2 + 2 * O * 2 * dt.itemsize + I * 4
  npdt = np.float16 if dt == dtypes.float16 else np.float32
  x = np.random.default_rng(12).standard_normal((3, I)).astype(npdt)
  y = Tensor(x, device=_device(), dtype=dt).linear(ql).numpy().astype(np.float32)
  ref = x.astype(np.float32) @ ref_w.T
  tol = 2e-3 if dt == dtypes.float16 else 1e-5
  np.testing.assert_allclose(y, ref, rtol=tol, atol=tol * np.abs(ref).max())

# ---- whole-model import from synthetic checkpoints (a 2-layer, 2048-token slice of Qwen2.5-0.5B)

def _save_st(fn: str, tensors: dict[str, np.ndarray]) -> None:
  """Minimal safetensors writer (BF16 for fusezap's uint16 bf16 views, else the numpy dtype)."""
  import json, struct
  names = {np.dtype(np.float32): "F32", np.dtype(np.float16): "F16", np.dtype(np.int32): "I32", np.dtype(np.uint16): "BF16"}
  hdr, off, blobs = {}, 0, []
  for k, a in tensors.items():
    b = np.ascontiguousarray(a).tobytes()
    hdr[k] = {"dtype": "BF16" if getattr(a, "st_dtype", "") == "BF16" else names[np.asarray(a).dtype], "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
    off += len(b); blobs.append(b)
  h = json.dumps(hdr).encode()
  with open(fn, "wb") as f: f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))

PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

@pytest.fixture(scope="module")
def tiny_ckpts(tmp_path_factory):
  """{name: dir}: 'base' (bf16 slice), 'gptq' (sym, from int4-sym RTN), 'awq' (asym, from int4 RTN), 'gptq-act' (act-order)."""
  import json
  from fusezap.serve.weights import load_safetensors, bf16_to_f32
  src = _model_path()
  st, cfg = load_safetensors(src), json.load(open(os.path.join(src, "config.json")))
  cfg.update(num_hidden_layers=2, vocab_size=2048)
  base = {k: v for k, v in st.items() if not k.startswith("model.layers.") or int(k.split(".")[2]) < 2}
  base["model.embed_tokens.weight"] = st["model.embed_tokens.weight"][:2048]
  root = tmp_path_factory.mktemp("ckpt")
  def write(name: str, tensors: dict, qc: dict | None = None) -> str:
    d = root / name; d.mkdir()
    _save_st(str(d / "model.safetensors"), tensors)
    (d / "config.json").write_text(json.dumps({**cfg, **({"quantization_config": qc} if qc else {})}))
    return str(d)
  out = {"base": write("base", base)}
  for name, mode in (("gptq", "int4-sym"), ("awq", "int4"), ("gptq-act", "int4")):
    t = dict(base)
    for i in range(2):
      for p in PROJ:
        k = f"model.layers.{i}.{p}"
        w = bf16_to_f32(t.pop(k + ".weight"))
        if name == "gptq-act":   # quantize columns in a permuted order: groups scatter over the input (desc_act)
          perm = np.random.default_rng(i).permutation(w.shape[1])
          qw = quantize_weight(w[:, perm], mode)
          g_idx = np.empty(w.shape[1], np.int32); g_idx[perm] = np.arange(w.shape[1]) // 128
          codes = np.empty(w.shape, np.uint8); codes[:, perm] = unpack_int4(qw.q)
        else:
          qw = quantize_weight(w, mode)
          g_idx, codes = np.arange(w.shape[1], dtype=np.int32) // 128, unpack_int4(qw.q)
        z = np.full_like(qw.scale, 8) if qw.zero is None else qw.zero
        if name == "awq": qweight, qzeros, scales = ref_awq_pack(codes.T, z.T, qw.scale.T)
        else:
          qweight, qzeros, scales = ref_gptq_pack(codes.T, z.T, qw.scale.T)
          t[k + ".g_idx"] = g_idx
        t[k + ".qweight"], t[k + ".qzeros"], t[k + ".scales"] = qweight, qzeros, scales
    qc = {"quant_method": "awq", "bits": 4, "group_size": 128, "version": "gemm", "zero_point": True} if name == "awq" else \
         {"quant_method": "gptq", "bits": 4, "group_size": 128, "desc_act": name == "gptq-act", "sym": mode == "int4-sym"}
    out[name] = write(name, t, qc)
  return out

KW = dict(dtype=dtypes.float16, max_len=64, prefill_chunk=32)

def _logits(m, ids):
  m.reset()
  return [m.prefill([ids])] + [m.decode([[t]]) for t in (11, 12, 13, 14)]

@needs_model
@pytest.mark.parametrize("fmt,mode", [("gptq", "int4-sym"), ("awq", "int4")])
def test_import_checkpoint_matches_quantize_model(tiny_ckpts, fmt, mode):
  """A GPTQ (sym) / AWQ (asym) checkpoint holding fusezap's own RTN codes imports to the same weights and the same
  prefill + jitted decode logits as quantize_model on the full-precision slice (f16 model: scales round alike)."""
  from fusezap.serve.models import load_model
  m = import_checkpoint(tiny_ckpts[fmt], device=_device(), **KW)
  ref = quantize_model(load_model(tiny_ckpts["base"], device=_device(), **KW), mode)
  assert m.quant == f"{fmt}-int4" and m.cfg.n_layers == 2
  for lm, lr in zip(m.layers, ref.layers):
    for n in LINEARS:
      a, b = getattr(lm, n), getattr(lr, n)
      assert isinstance(a, QuantLinear) and a.mode == mode and a.shape == b.shape
      np.testing.assert_array_equal(a.q.numpy(), b.q.numpy())
      np.testing.assert_array_equal(a.scale.numpy(), b.scale.numpy())
  ids = list(np.random.default_rng(13).integers(0, 2048, 20))
  for x, y in zip(_logits(m, ids), _logits(ref, ids)): np.testing.assert_allclose(x, y, atol=1e-3, rtol=1e-3)
  assert m._decode_jit.captured is not None

@needs_model
def test_import_gptq_act_order_model(tiny_ckpts):
  """desc_act checkpoint: logits match a model holding the reference-dequantized weights as plain Tensors."""
  from fusezap.serve.models import load_model
  from fusezap.serve.weights import load_safetensors
  m = import_checkpoint(tiny_ckpts["gptq-act"], device=_device(), **KW)
  assert all(getattr(l, n).g_idx is not None for l in m.layers for n in LINEARS)
  st = load_safetensors(tiny_ckpts["gptq-act"])
  ref = load_model(tiny_ckpts["base"], device=_device(), **KW)
  for i, l in enumerate(ref.layers):
    deq = {p: ref_gptq_dequant(*(st[f"model.layers.{i}.{p}.{x}"] for x in ("qweight", "qzeros", "scales", "g_idx"))) for p in PROJ}
    np.testing.assert_allclose(dequantize_weight(m.layers[i].wqkv.host(), dtypes.float16),
                               np.concatenate([deq[p] for p in PROJ[:3]]), rtol=2e-3, atol=1e-6)   # f16 compute vs f32 reference
    for n, ps in (("wqkv", PROJ[:3]), ("wo", PROJ[3:4]), ("wgu", PROJ[4:6]), ("wd", PROJ[6:])):
      setattr(l, n, Tensor(np.concatenate([deq[p] for p in ps]).astype(np.float16), device=_device(), dtype=dtypes.float16).realize())
  ids = list(np.random.default_rng(14).integers(0, 2048, 20))
  for x, y in zip(_logits(m, ids), _logits(ref, ids)): np.testing.assert_allclose(x, y, atol=5e-2, rtol=2e-2)
