"""Weight-only quantization (SPEC §11.3): int8 per output channel, int4 in groups of 128 (two per byte, symmetric or
asymmetric), FP8 e4m3 with one scale per tensor, and AWQ / GPTQ int4 checkpoint import.

  m = load_model("Qwen/Qwen2.5-0.5B-Instruct", dtype=dtypes.float16)
  quantize_model(m, "int8")            # or "int4" (asymmetric, the default int4), "int4-sym", "fp8"
  m.prefill([ids]); m.decode([[t]])    # unchanged API; the jitted steps are re-captured
  m = import_checkpoint("some/Qwen2.5-GPTQ-Int4-dir")   # AWQ (gemm) / GPTQ 4-bit safetensors checkpoint -> Llama

`QuantLinear` is a drop-in for a (out, in) weight Tensor: `x.linear(qw, b)` works because `Tensor.linear` only calls
`weight.ndim` and `weight.transpose(-1, -2)`, and the transpose returns the *lazy* dequantize expression. The
scheduler fuses that expression into the matmul's reduce, so a composed GEMV (decode, FZ_CUSTOM=0) is one kernel that
loads the int8 / packed uint8 codes and the scales, dequantizes in registers and accumulates in f32 (dequantize in
kernel; the full-precision weight never exists in memory). Where the Metal library GEMM matches (prefill, f32
operands after the matmul's casts), the scheduler materializes the dequantized weight into a scratch buffer first and
then runs the GEMM (dequantize per call).

Formats (codes are stored on the device; scales and zeros in the model's compute dtype):
  int8      q int8 (out, in), w = q * s[o]; s = max|w[o, :]| / 127, q = round(w / s) in [-127, 127]
  int4-sym  q uint8 (out, in_p/2) packed, w = (q - 8) * s[o, g]; s = max|w_g| / 7.5 (GPTQ's sym grid: 2*max/15),
            q = clip(round(w / s) + 8, 0, 15)
  int4      (asymmetric) w = (q - z[o, g]) * s[o, g]; s = (max_g - min_g) / 15, z = clip(round(-min_g / s), 0, 15),
            q = clip(round(w / s) + z, 0, 15)
  fp8       q uint8 (out, in) = float8_e4m3fn bits (1 sign, 4 exponent bits with bias 7, 3 mantissa bits; no inf, max 448,
            round to nearest even, saturating), w = e4m3(q) * s; s = max|w| / 448, one scale for the whole tensor
            (the fused q/k/v and gate/up projections each count as one tensor)
Group g covers input columns [128 g, 128 g + 128). Packing: byte j of group g holds column 128 g + j in the low nibble and
column 128 g + 64 + j in the high nibble. (This half-split layout keeps the nibble choice constant over runs of 64
columns; with the interleaved layout, even/odd column in low/high nibble, the fused GEMV was 2-3x slower.) `in` is zero-padded to a multiple of the group size (in_p); the pad columns are cut off after
dequantization. An imported GPTQ act-order checkpoint (desc_act) keeps its `g_idx`: column i then uses group g_idx[i]
instead of i // 128 (gathered on the device per call, so it is slower than a checkpoint without act-order).

FP8 on the device: fusezap has no float8 dtype yet (SPEC §4 reserves float8_e4m3; CAST has no scale operand), so the
codes are uint8 and the dequantize is integer ops plus a bitcast: the 7 magnitude bits shifted left by 7 are an f16
whose value is exactly e4m3 / 256 (normals and subnormals alike: both formats put the exponent field right above the
mantissa), so w = sign * bitcast_f16((q & 0x7F) << 7) * (256 s). The device stores 256 s in the compute dtype. This
is weight-only FP8 (activations stay f16/f32); the FP8 GEMM with fp8 activations is P3-GEMM's (H100).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from fusezap.tensor import Tensor
from fusezap.dtype import DType, dtypes, to_np
from fusezap.helpers import prod

MODES = ("int8", "int4", "int4-sym", "int4-asym", "fp8", "fp8-e4m3")
GROUP = 128
FP8_MAX = 448.0

def _canon(mode: str) -> str:
  mode = {"int4-asym": "int4", "q8": "int8", "q4": "int4", "fp8-e4m3": "fp8", "e4m3": "fp8"}.get(mode, mode)
  if mode not in ("int8", "int4", "int4-sym", "fp8"): raise ValueError(f"unknown quant mode {mode!r} (have {MODES})")
  return mode

# ---------------------------------------------------------------- float8 e4m3 (fn) codec

def fp8_e4m3_decode(c: np.ndarray) -> np.ndarray:
  """uint8 float8_e4m3fn bits -> float32 (0x7F / 0xFF are NaN)."""
  c = np.asarray(c, np.uint8).astype(np.int32)
  e, m = (c >> 3) & 15, (c & 7).astype(np.float64)
  v = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * np.exp2(e - 7.0))
  v = np.where((c & 0x7F) == 0x7F, np.nan, v)
  return np.where(c >= 128, -v, v).astype(np.float32)

def fp8_e4m3_encode(x: np.ndarray) -> np.ndarray:
  """float -> uint8 float8_e4m3fn bits: round to nearest, ties to even, saturating at +-448 (no NaN out for finite x).
  The code of a magnitude is 8 (e + 7) + m for normals (e >= -6, the mantissa rounded with np.rint, whose carry into
  the next exponent is the next code) and round(a * 2^9) for subnormals, so ties to even mantissa = ties to even code."""
  x = np.asarray(x, np.float64)
  assert np.isfinite(x).all(), "fp8 encode: non-finite input"
  a = np.minimum(np.abs(x), FP8_MAX)
  mant, ex = np.frexp(a)                               # a = (2 mant) 2^(ex - 1), 2 mant in [1, 2)
  code = np.where(a >= 2.0 ** -6, (ex + 6) * 8 + np.rint((2 * mant - 1) * 8), np.rint(a * 2.0 ** 9)).astype(np.uint8)
  return np.where(np.signbit(x), code | 0x80, code).astype(np.uint8)   # -0 -> 0x80, like torch

# ---------------------------------------------------------------- host (numpy) quantization

@dataclass
class QuantWeight:
  """Host arrays of one quantized (out, in) weight. q: int8 (out, in), uint8 (out, in_p // 2) (int4) or uint8 (out, in)
  (fp8 e4m3 bits); scale (out, 1), (out, G) or (1, 1) (fp8) float32; zero (out, G) float32 (asymmetric int4) or None.
  g_idx (in,) int32: group of each input column (imported GPTQ act-order), None = column // group."""
  mode: str
  shape: tuple[int, int]
  q: np.ndarray
  scale: np.ndarray
  zero: np.ndarray | None = None
  group: int = GROUP
  g_idx: np.ndarray | None = None

def pack_int4(q: np.ndarray, group: int = GROUP) -> np.ndarray:
  """(out, in) codes in [0, 15], `in` a multiple of `group` -> (out, in/2) uint8. Within each group, byte j holds
  element j in the low nibble and element j + group/2 in the high nibble."""
  O, I = q.shape
  g = q.astype(np.uint8).reshape(O, I // group, 2, group // 2)
  return (g[:, :, 0] | (g[:, :, 1] << 4)).reshape(O, I // 2).astype(np.uint8)

def unpack_int4(p: np.ndarray, group: int = GROUP) -> np.ndarray:
  O, h = p.shape[0], group // 2
  g = p.reshape(O, -1, 1, h)
  return np.concatenate([g & 0xF, g >> 4], axis=2).reshape(O, -1)

def _grid4(wg: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
  """int4 codes, scales and zeros of grouped weights (O, G, group): round to nearest on the group's full range."""
  if mode == "int4-sym":
    s = np.abs(wg).max(2) / 7.5
    s = np.where(s == 0, 1.0, s).astype(np.float32)
    return np.clip(np.rint(wg / s[..., None]) + 8, 0, 15), s, None
  lo, hi = np.minimum(wg.min(2), 0), np.maximum(wg.max(2), 0)   # the grid always contains 0 (exact zero padding)
  s = (hi - lo) / 15.0
  s = np.where(s == 0, 1.0, s).astype(np.float32)
  z = np.clip(np.rint(-lo / s), 0, 15).astype(np.float32)
  return np.clip(np.rint(wg / s[..., None]) + z[..., None], 0, 15), s, z

def quantize_weight(w: np.ndarray, mode: str = "int8", group: int = GROUP) -> QuantWeight:
  """Quantize a float (out, in) weight on the host (math in float32), round to nearest (RTN). A data-free MSE clipping
  search per group was tried and raised int4 perplexity (outliers matter), so it is not used."""
  mode = _canon(mode)
  w = np.asarray(w, np.float32)
  assert w.ndim == 2, f"weight must be 2-D (out, in), got {w.shape}"
  O, I = w.shape
  if mode == "int8":
    s = np.abs(w).max(1, keepdims=True) / 127.0
    s = np.where(s == 0, 1.0, s).astype(np.float32)
    return QuantWeight(mode, (O, I), np.clip(np.rint(w / s), -127, 127).astype(np.int8), s, None, I)
  if mode == "fp8":
    s = np.float32(np.abs(w).max() / FP8_MAX) if w.size else np.float32(0)
    s = np.float32(1.0) if s == 0 else s
    return QuantWeight(mode, (O, I), fp8_e4m3_encode(w / s), np.full((1, 1), s, np.float32), None, I)
  Ip = -(-I // group) * group
  wg = np.pad(w, ((0, 0), (0, Ip - I))).reshape(O, Ip // group, group)
  q, s, z = _grid4(wg, mode)
  return QuantWeight(mode, (O, I), pack_int4(q.reshape(O, Ip), group), s, z, group)

def dequantize_weight(qw: QuantWeight, dtype: DType = dtypes.float32) -> np.ndarray:
  """Reference dequantization in numpy with the same op order and rounding as QuantLinear.dequantize: codes and scales
  in the compute dtype (float16 for an f16 model, else float32), then cast to `dtype` (float32 host array for bf16)."""
  cdt = np.float16 if dtype == dtypes.float16 else np.float32
  O, I = qw.shape
  if qw.mode == "int8": return (qw.q.astype(cdt) * qw.scale.astype(cdt)).astype(np.float32)
  if qw.mode == "fp8":
    mag = ((qw.q & 0x7F).astype(np.uint16) << 7).view(np.float16)
    return (np.where(qw.q < 128, mag, -mag).astype(cdt) * (qw.scale * 256).astype(cdt)).astype(np.float32)
  if qw.g_idx is not None:   # act-order: per-column group lookup
    q = unpack_int4(qw.q, qw.group)[:, :I].astype(cdt)
    q = q - (cdt(8) if qw.zero is None else qw.zero.astype(cdt)[:, qw.g_idx])
    return (q * qw.scale.astype(cdt)[:, qw.g_idx]).astype(np.float32)
  q = unpack_int4(qw.q, qw.group).astype(cdt).reshape(O, -1, qw.group)
  q = q - (cdt(8) if qw.zero is None else qw.zero.astype(cdt)[..., None])
  return (q * qw.scale.astype(cdt)[..., None]).reshape(O, -1)[:, :I].astype(np.float32)

def concat_weights(qws: list[QuantWeight]) -> QuantWeight:
  """Stack quantized weights along `out` (fused q/k/v, gate/up). int4 parts need the same mode, group and g_idx."""
  a = qws[0]
  if len(qws) == 1: return a
  assert a.mode != "fp8", "fp8 has one scale per tensor: quantize the fused weight instead"
  for b in qws[1:]:
    assert (b.mode, b.group, b.shape[1]) == (a.mode, a.group, a.shape[1]), "cannot fuse weights of different formats"
    if not ((a.g_idx is None and b.g_idx is None) or (a.g_idx is not None and b.g_idx is not None and np.array_equal(a.g_idx, b.g_idx))):
      raise NotImplementedError("fusing act-order weights with different g_idx is not supported")
  cat = lambda xs: None if xs[0] is None else np.concatenate(xs, 0)
  return QuantWeight(a.mode, (sum(b.shape[0] for b in qws), a.shape[1]), cat([b.q for b in qws]), cat([b.scale for b in qws]),
                     cat([b.zero for b in qws]), a.group, a.g_idx)

# ---------------------------------------------------------------- device layer

class QuantLinear:
  """A quantized (out, in) linear weight that stands in for a weight Tensor (see the module docstring).
  `x.linear(qw, bias)` or `qw(x, bias)` computes x @ dequant(qw).T + bias in the model dtype."""
  def __init__(self, qw: QuantWeight, dtype: DType = dtypes.float32, device: str | None = None):
    self.mode, self.shape, self.group, self.dtype = qw.mode, qw.shape, qw.group, dtype
    self.cdt = dtypes.float16 if dtype == dtypes.float16 else dtypes.float32   # dequantize math dtype
    def dev(a: np.ndarray, dt: DType) -> Tensor: return Tensor(np.ascontiguousarray(a.astype(to_np(dt))), device=device, dtype=dt).realize()
    self.q = dev(qw.q, dtypes.int8 if qw.mode == "int8" else dtypes.uint8)
    self.scale = dev(qw.scale * 256 if qw.mode == "fp8" else qw.scale, self.cdt)   # fp8: 256 s (see module doc)
    self.zero = None if qw.zero is None else dev(qw.zero, self.cdt)
    self.g_idx = None if qw.g_idx is None else dev(qw.g_idx, dtypes.int32)
    self.device = self.q.device
    self._native(None if qw.mode != "fp8" else Tensor(np.asarray(qw.scale, np.float32).reshape(1), device=self.device).realize())
  def _native(self, ws: Tensor | None) -> None:
    """CUDA sm_89+: fp8 runs W8A8 on the e4m3 tensor-core GEMM (ws = the weight's f32 scale on the device)."""
    from fusezap.serve import cuda as cu
    self.ws = ws if self.mode == "fp8" and cu.fp8_ok(self.device, self.shape[1]) else None
  @staticmethod
  def from_device_fp8(q: Tensor, s: Tensor, dtype: DType) -> "QuantLinear":
    """An fp8 QuantLinear from device codes q (out, in) uint8 and scale s (1,) f32 (fusezap.serve.cuda.quantize_fp8_weight)."""
    self = QuantLinear.__new__(QuantLinear)
    self.mode, self.shape, self.group, self.dtype = "fp8", tuple(q.shape), GROUP, dtype
    self.cdt = dtypes.float16 if dtype == dtypes.float16 else dtypes.float32
    self.q, self.zero, self.g_idx, self.device = q, None, None, q.device
    self.scale = (s * 256).cast(self.cdt).reshape(1, 1).contiguous().realize()
    self._native(s)
    return self
  @staticmethod
  def from_float(w: np.ndarray, mode: str = "int8", dtype: DType = dtypes.float32, device: str | None = None, group: int = GROUP) -> "QuantLinear":
    return QuantLinear(quantize_weight(w, mode, group), dtype, device)
  ndim = 2
  def host(self) -> QuantWeight:
    """The stored codes, scales and zeros copied back to the host (float32 scales/zeros)."""
    f = lambda t: None if t is None else t.numpy().astype(np.float32)
    s = f(self.scale) / 256 if self.mode == "fp8" else f(self.scale)
    return QuantWeight(self.mode, self.shape, self.q.numpy(), s, f(self.zero), self.group,  # type: ignore[arg-type]
                       None if self.g_idx is None else self.g_idx.numpy())
  @property
  def nbytes(self) -> int: return sum(t.numel() * t.dtype.itemsize for t in (self.q, self.scale, self.zero, self.g_idx) if t is not None)
  def dequantize(self) -> Tensor:
    """Lazy (out, in) weight in the model dtype. Nothing is realized: consumers fuse it into their kernels."""
    O, I = self.shape
    if self.mode == "int8": w = self.q.cast(self.cdt) * self.scale
    elif self.mode == "fp8":
      u = self.q.cast(dtypes.int16)
      mag = ((u % 128) * 128).bitcast(dtypes.float16)      # e4m3 magnitude / 256, exact
      w = (u < 128).where(mag, -mag).cast(self.cdt) * self.scale
    elif self.g_idx is not None:                          # act-order: gather the group of every column
      p = self.q.reshape(O, -1, 1, self.group // 2)
      q = (p % 16).cat(p // 16, axis=2).reshape(O, -1).shrink((None, (0, I))).cast(self.cdt)
      q = q - (8 if self.zero is None else self.zero[:, self.g_idx])
      w = q * self.scale[:, self.g_idx]
    else:
      p = self.q
      G = p.shape[1] * 2 // self.group
      p = p.reshape(O, G, 1, self.group // 2)
      q = (p % 16).cat(p // 16, axis=2).reshape(O, G, self.group).cast(self.cdt)
      q = q - (8 if self.zero is None else self.zero.reshape(O, G, 1))
      w = (q * self.scale.reshape(O, G, 1)).reshape(O, G * self.group)
      if G * self.group != I: w = w.shrink((None, (0, I)))
    return w.cast(self.dtype) if w.dtype != self.dtype else w
  def transpose(self, ax0: int = 1, ax1: int = 0) -> Tensor: return self.dequantize().transpose(ax0, ax1)
  @property
  def T(self) -> Tensor: return self.dequantize().T
  def numpy(self) -> np.ndarray: return self.dequantize().numpy()
  def __call__(self, x: Tensor, bias: Tensor | None = None) -> Tensor: return self.apply(x, bias)
  def apply(self, x: Tensor, bias: Tensor | None = None) -> Tensor:
    """x @ dequant(self)^T + bias with the best kernel for the device (Llama.linear calls this):
      CUDA fp8 (sm_89+)      W8A8: per-row e4m3 activations, the e4m3 tensor-core GEMM (fusezap/serve/cuda.py)
      CUDA f16/bf16, >= 64 rows (prefill)   dequantize into a scratch weight, then the 16-bit tensor-core GEMM (inside a
                             fz.jit capture the dequantize reads only constants, so it runs once and is kept)
      otherwise              Tensor.linear: the dequantize expression fuses into the composed GEMV (decode, METAL)"""
    from fusezap.serve import cuda as cu
    if self.ws is not None: return cu.fp8_linear(x, self.q, self.ws, bias, self.dtype)
    rows = prod(x.shape[:-1])
    if isinstance(rows, int) and rows >= 64 and cu.gemm16_ok(self.device, self.dtype, self.shape[1]) and x.dtype == self.dtype:
      return cu.linear(x, self.dequantize().contiguous(), bias)
    return x.linear(self, bias)  # type: ignore[arg-type]
  def __repr__(self) -> str: return f"QuantLinear({self.mode}, {self.shape}, {self.dtype})"

# ---------------------------------------------------------------- model rewrite

LINEARS = ("wqkv", "wo", "wgu", "wd")

def _host(t: Any) -> np.ndarray:
  from fusezap.serve.weights import bf16_to_f32
  a = t.numpy()
  return bf16_to_f32(a) if t.dtype == dtypes.bfloat16 else a.astype(np.float32)

def quantize_model(model: Any, mode: str = "int8", group: int = GROUP, head: bool = False) -> Any:
  """Replace the Linear weights of a fusezap.serve Llama in place (q/k/v, o, gate/up and down projections of every
  layer; the lm_head too if head=True) with QuantLinear, and drop the captured jit steps so they re-trace. Norms,
  biases and the host embedding table stay in full precision. Returns the model."""
  mode = _canon(mode)
  from fusezap.serve import cuda as cu
  def q(w: Tensor) -> QuantLinear:
    if mode == "fp8" and cu.fp8_ok(model.device, w.shape[1]):   # CUDA: quantize on the device (no host round trip)
      return QuantLinear.from_device_fp8(*cu.quantize_fp8_weight(w), model.dtype)
    return QuantLinear(quantize_weight(_host(w), mode, group), model.dtype, model.device)
  for l in model.layers:
    for n in LINEARS:
      w = getattr(l, n)
      if isinstance(w, QuantLinear): raise ValueError("model is already quantized")
      setattr(l, n, q(w))
  if head: model.lm_head = q(model.lm_head)
  model._decode_jit.reset()
  model._prefill_jits.clear()
  model.quant = mode
  return model

def weight_bytes(model: Any) -> dict[str, int]:
  """Device bytes of the model's weights: linear layers, lm_head, everything else (norms, biases, RoPE tables), and
  the host embedding table (an mmap of the checkpoint, read row by row)."""
  def nb(t: Any) -> int: return 0 if t is None else (t.nbytes if isinstance(t, QuantLinear) else t.numel() * t.dtype.itemsize)
  lin = sum(nb(getattr(l, n)) for l in model.layers for n in LINEARS)
  other = sum(nb(getattr(l, n)) for l in model.layers for n in ("bqkv", "bo", "bgu", "bd", "ln1", "ln2")) + nb(model.norm) + nb(model.cos) + nb(model.sin)
  head = nb(model.lm_head)
  return {"linear": lin, "lm_head": head, "other": other, "device_total": lin + head + other, "host_embed": int(model.embed_host.nbytes)}

# ---------------------------------------------------------------- AWQ / GPTQ checkpoint import

AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)   # AutoAWQ GEMM packing: nibble i of an int32 holds output column 8k + AWQ_ORDER[i]

def _nibbles(a: np.ndarray) -> np.ndarray:
  """int32 (..., n) -> uint8 (..., n, 8): nibble i = bits [4i, 4i + 4)."""
  return ((np.asarray(a).view(np.uint32)[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).astype(np.uint8)

def _int4_weight(q: np.ndarray, z: np.ndarray, s: np.ndarray, group: int, g_idx: np.ndarray | None, sym: bool) -> QuantWeight:
  """codes q (out, in) in [0, 15], zero points z and scales s (out, G) -> QuantWeight. sym: drop z (all 8)."""
  O, I = q.shape
  if g_idx is not None:
    g_idx = np.asarray(g_idx, np.int64)
    if np.array_equal(g_idx, np.arange(I) // group): g_idx = None
    else: assert g_idx.shape == (I,) and 0 <= g_idx.min() and g_idx.max() < s.shape[1], "bad g_idx"
  Ip = -(-I // group) * group
  if g_idx is None: assert s.shape[1] == Ip // group, f"scales {s.shape} do not match in={I}, group={group}"
  qp = pack_int4(np.pad(q, ((0, 0), (0, Ip - I))), group)
  zero = None if sym else np.ascontiguousarray(z, np.float32)
  return QuantWeight("int4-sym" if sym else "int4", (O, I), qp, np.ascontiguousarray(s, np.float32), zero, group,
                     None if g_idx is None else g_idx.astype(np.int32))

def gptq_weight(qweight: np.ndarray, qzeros: np.ndarray, scales: np.ndarray, g_idx: np.ndarray | None = None,
                group: int = GROUP, v2: bool = False, sym: bool | None = None) -> QuantWeight:
  """AutoGPTQ 4-bit layout -> QuantWeight. qweight int32 (in/8, out), codes packed along `in` (nibble j of row r = input
  8r + j); qzeros int32 (G, out/8), packed along `out`; scales (G, out); g_idx (in,) the group of each input. w = (q - z) s
  where z = qzeros + 1 for the original format (v1: zeros were stored minus one; the +1 is not masked, as in the GPTQ
  kernels) and z = qzeros for gptq_v2. sym=None: symmetric iff every zero point is 8."""
  I8, O = qweight.shape
  q = _nibbles(qweight).transpose(0, 2, 1).reshape(I8 * 8, O).T
  z = _nibbles(qzeros).reshape(qzeros.shape[0], -1)[:, :O].T.astype(np.float32) + (0 if v2 else 1)
  if sym is None: sym = bool((z == 8).all())
  return _int4_weight(np.ascontiguousarray(q), z, np.asarray(scales, np.float32).T, group, g_idx, sym and bool((z == 8).all()))

def awq_weight(qweight: np.ndarray, qzeros: np.ndarray, scales: np.ndarray, group: int = GROUP) -> QuantWeight:
  """AutoAWQ GEMM 4-bit layout -> QuantWeight. qweight int32 (in, out/8) and qzeros int32 (G, out/8), both packed along
  `out` in AWQ_ORDER; scales (G, out). w = (q - z) s."""
  def unpack(a: np.ndarray) -> np.ndarray:
    n = _nibbles(a)
    out = np.empty_like(n)
    out[..., list(AWQ_ORDER)] = n
    return out.reshape(a.shape[0], -1)
  q, z = unpack(qweight).T, unpack(qzeros).T.astype(np.float32)
  return _int4_weight(np.ascontiguousarray(q), z, np.asarray(scales, np.float32).T, group, None, False)

def checkpoint_format(cfg: dict, fmt: str | None = None) -> tuple[str, dict]:
  """The quantization method of an HF config ("gptq" | "awq") and its quantization_config, validated."""
  qc = dict(cfg.get("quantization_config") or {})
  fmt = (fmt or qc.get("quant_method") or "").lower()
  if fmt not in ("gptq", "awq"): raise ValueError(f"not an AWQ/GPTQ checkpoint (quant_method {fmt!r})")
  bits = qc.get("bits", qc.get("w_bit", 4))
  if bits != 4: raise NotImplementedError(f"{fmt} with {bits} bits: only 4-bit checkpoints are imported")
  if fmt == "awq":
    if str(qc.get("version", "gemm")).lower() != "gemm": raise NotImplementedError(f"AWQ version {qc.get('version')!r}: only 'gemm' is imported")
    if not qc.get("zero_point", True): raise NotImplementedError("AWQ without zero points is not supported")
  return fmt, qc

def import_checkpoint(path: str, fmt: str | None = None, dtype: DType = dtypes.float16, device: str | None = None, **kw) -> Any:
  """Load a 4-bit AWQ (GEMM) or GPTQ safetensors checkpoint (local dir or HF id, config.json with quantization_config)
  as a fusezap.serve Llama whose linears are QuantLinear. fmt overrides quant_method. kw go to Llama (max_len, ...).
  Linears the checkpoint keeps in full precision (`<name>.weight`) load as plain weights."""
  import json, os
  from fusezap.serve.weights import resolve_model, load_config, load_safetensors
  from fusezap.serve.models.llama import Llama, LlamaConfig
  path = resolve_model(path)
  cfg = load_config(path)
  fmt, qc = checkpoint_format(cfg, fmt)
  if cfg.get("model_type", "llama") not in ("llama", "qwen2", "mistral"): raise NotImplementedError(f"model_type {cfg.get('model_type')!r}")
  st = load_safetensors(path)
  lc = LlamaConfig.from_hf(cfg)
  gen = os.path.join(path, "generation_config.json")
  if os.path.exists(gen):
    with open(gen) as f: eos = json.load(f).get("eos_token_id")
    lc.eos_token_ids = sorted({*lc.eos_token_ids, *([eos] if isinstance(eos, int) else (eos or []))})
  gs = int(qc.get("group_size", qc.get("q_group_size", GROUP)))
  v2 = str(qc.get("checkpoint_format", "gptq")).lower() == "gptq_v2"
  def read(name: str) -> QuantWeight:
    group = gs if gs > 0 else (st[name + ".qweight"].shape[0] * (8 if fmt == "gptq" else 1))   # -1: one group per row
    if fmt == "awq": return awq_weight(st[name + ".qweight"], st[name + ".qzeros"], st[name + ".scales"], group)
    return gptq_weight(st[name + ".qweight"], st[name + ".qzeros"], st[name + ".scales"], st.get(name + ".g_idx"), group, v2,
                       None if qc.get("sym") is None else bool(qc["sym"]))
  weights: dict[str, np.ndarray] = dict(st)
  fused: dict[tuple[int, str], list[str]] = {}
  for i in range(lc.n_layers):
    a, m = f"model.layers.{i}.self_attn.", f"model.layers.{i}.mlp."
    for attr, names in (("wqkv", [a + "q_proj", a + "k_proj", a + "v_proj"]), ("wo", [a + "o_proj"]),
                        ("wgu", [m + "gate_proj", m + "up_proj"]), ("wd", [m + "down_proj"])):
      quant = [n + ".qweight" in st for n in names]
      if not any(quant): continue
      if not all(quant): raise NotImplementedError(f"{names}: fusing quantized and full-precision projections is not supported")
      for n in names: weights[n + ".weight"] = np.zeros((1, 1), np.float32)   # placeholder: replaced below, never used
      fused[(i, attr)] = names
  model = Llama(lc, weights, dtype, device, **kw)
  for (i, attr), names in fused.items(): setattr(model.layers[i], attr, QuantLinear(concat_weights([read(n) for n in names]), model.dtype, model.device))
  if "lm_head.qweight" in st: model.lm_head = QuantLinear(read("lm_head"), model.dtype, model.device)
  model.quant = f"{fmt}-int4"
  return model
