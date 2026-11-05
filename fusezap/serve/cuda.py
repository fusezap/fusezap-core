"""CUDA paths of fusezap.serve (P4-CUDA-SERVE): library GEMMs and FP8 for the model, launched as fz.jit-recordable items.

  linear(x, w, b)                 x (..., K) @ w (N, K)^T (+ b) for f16/bf16 on the tensor-core GEMM (mma.sync / sm_90a wgmma,
                                  fusezap.device.cuda_kernels), output written in the input dtype (one rounding of the f32 sum;
                                  with b, one rounding of the f32 sum + b, like cuBLAS's bias epilogue)
  fp8_linear(x, wq, ws, b, dt)    W8A8 FP8: x is quantized per row (token) to e4m3 on the device (scale = row max |x| / 448),
                                  then the e4m3 tensor-core GEMM with f32 accumulation; y = (xq @ wq^T) * sx[row] * ws (+ b)
  quantize_fp8_weight(w)          per-tensor e4m3 codes (uint8, the layout of fusezap.serve.quant) + a (1,) f32 device scale
  rmsnorm(x, w, eps, d)           fused residual add + RMSNorm, one block per row
  silu_mul(gu)                    SwiGLU's silu(gate) * up from the fused gate/up output, one bandwidth-bound pass
  rope_qkv(y, b, cos, sin, ...)   the qkv projection's bias add + RoPE on q and k + the copy of v, one pass (y: the GEMM's f32
                                  accumulator, rounded once after the bias add)
  argmax_rows(x)                  (R, V) f32 logits -> (R,) int32 greedy tokens on the device (np.argmax's tie rule)
  CudaCall                        a kind="custom" payload for a CUDA C++ kernel (NVRTC), params may be symbolic Nodes

Every launch goes through `fusezap.jit.run_item`, so a jitted forward records the GEMMs and kernels and replays them with
the replay's buffers. Rows are independent everywhere (per-row activation scales, fixed GEMM configs per shape), so a
token's result never depends on which other tokens share the step: batched == alone, as on METAL.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence
from fusezap.dtype import DType, dtypes
from fusezap.device import Device, Buffer
from fusezap.helpers import prod
from fusezap.shape.symbolic import Node, sym_infer

HALVES = (dtypes.float16, dtypes.bfloat16)
_KIND = {dtypes.float16: "f16", dtypes.bfloat16: "bf16"}
_OUT = {dtypes.float32: "f32", dtypes.float16: "f16", dtypes.bfloat16: "bf16"}

def is_cuda(device: str) -> bool: return device.split(":")[0].upper() == "CUDA"

def bias_round2() -> bool:
  """FZ_BIAS_ROUND2=1, for speed A/B measurements only (docs/status/SPEED.md run 4): the bias path before 6c02e9d, which
  rounded the GEMM output to the model dtype and then added the bias, round(round(xW) + b) (a bf16 GEMM output instead
  of an f32 one). Less accurate (Qwen's qkv agreed with HF bf16 on 69-72% of elements, not 99.95%); never a default.
  Read when a forward is traced, so it costs nothing per step."""
  import os
  return os.environ.get("FZ_BIAS_ROUND2", "") == "1"
def cc(device: str) -> tuple[int, int]: return tuple(getattr(Device[device], "cc", (0, 0)))  # type: ignore[return-value]
def gemm16_ok(device: str, dt: DType, K: int) -> bool:
  """The 16-bit tensor-core GEMM takes x (M, K) @ w (N, K)^T: f16/bf16, sm_80+, 16-byte rows (K % 8 == 0)."""
  return is_cuda(device) and dt in HALVES and K % 8 == 0 and cc(device) >= (8, 0)
def fp8_ok(device: str, K: int) -> bool:
  """The e4m3 GEMM: sm_89+ (Ada, Hopper) and K % 16 == 0."""
  return is_cuda(device) and K % 16 == 0 and cc(device) >= (8, 9)

def _run(call: Any, bufs: Sequence[Buffer], n_out: int = 1, var_vals: dict | None = None) -> None:
  from fusezap import jit as J
  from fusezap.ir import ScheduleItem
  J.run_item(ScheduleItem((call,), tuple(bufs), var_vals or {}, "custom", n_out))   # type: ignore[arg-type]

def _buf(t: Any) -> Buffer: return t._realized_contiguous().realized
def _wrap(buf: Buffer, shape: tuple[int, ...]) -> Any:
  from fusezap.tensor import Tensor
  from fusezap.lazy import LazyBuffer
  return Tensor(LazyBuffer.from_buffer(buf, shape))

# ---------------------------------------------------------------- raw CUDA C++ kernels

@dataclass(frozen=True, eq=False)
class CudaCall:
  """Payload (`ast[0]`) of a kind="custom" item: CUDA C++ source `src` defining `name`, int params (Nodes are evaluated
  with the item's var_vals at launch, so a jit replay passes new values), grid in blocks and block in threads."""
  name: str
  src: str
  params: tuple[Any, ...]
  global_size: tuple[int, int, int]
  local_size: tuple[int, int, int]
  @property
  def kernel(self) -> "CudaCall": return self
  def launch(self, dev: Any, bufs: Sequence[Buffer], var_vals: dict | None = None) -> float | None:
    if bufs[0].size == 0: return None
    vals = tuple(int(sym_infer(x, var_vals or {})) if isinstance(x, Node) else int(x) for x in self.params)
    progs = dev.__dict__.setdefault("_fz_serve_cu", {})
    if (prg := progs.get((self.name, self.src))) is None:
      progs[(self.name, self.src)] = prg = dev.runtime(self.name, dev.compiler.compile_cached(self.src))
    return prg(*[b.ensure_allocated().handle for b in bufs], global_size=self.global_size, local_size=self.local_size, vals=vals)
  def pretty(self, indent: int = 0) -> str: return f"{'  ' * indent}CUDA {self.name} {self.params}"

def run_cuda(call: CudaCall, bufs: Sequence[Buffer], n_out: int = 1) -> None:
  from fusezap import jit as J
  vv = {v: x for v, x in J.current_var_vals().items() if any(isinstance(p, Node) and v in p.vars() for p in call.params)}
  _run(call, bufs, n_out, vv)

# ---------------------------------------------------------------- GEMM items

@dataclass(eq=False)
class GemmCall:
  """C (M, N) = alpha * [sb] * A (M, K) @ B (N, K)^T with fusezap.device.cuda_kernels.gemm (kind f16/bf16/e4m3, out
  f32/f16/bf16; sb: optional device f32 scalar, bufs[3]). Launchers are bound once per buffer set (replays reuse them)."""
  M: int
  N: int
  K: int
  kind: str
  out: str
  alpha: float = 1.0
  impl: str | None = None   # None: the dispatch's pick; "wgswap": the decode kernel (a mixed forward's decode rows)
  a_off: int = 0            # element offsets into A and C (rows of a larger buffer)
  c_off: int = 0
  _bound: dict = field(default_factory=dict, repr=False)
  @property
  def kernel(self) -> "GemmCall": return self
  @property
  def name(self) -> str: return f"fz_serve_gemm_{self.kind}_{self.out}"
  def launch(self, dev: Any, bufs: Sequence[Buffer], var_vals: dict | None = None) -> float | None:
    from fusezap.device.cuda_kernels import gemm
    from fusezap.helpers import getenv
    if bufs[0].size == 0: return None
    hs = tuple(b.ensure_allocated().handle for b in bufs)
    if self.impl is None and self.alpha == 1.0 and len(hs) == 3 and not self.a_off and not self.c_off:
      from fusezap.device.cuda_kernels import cublaslt
      if (pl := cublaslt.plan_for(bufs[0].device, self.M, self.N, self.K, self.kind, self.out)) is not None:
        pl.launch(dev, hs[0], hs[1], hs[2], self.M)   # a pinned, self-checked cuBLASLt algo (prefill sizes; enable())
        return None
    if (f := self._bound.get(hs)) is None:
      if len(self._bound) > 64: self._bound.clear()
      eo, ea = (4 if self.out == "f32" else 1 if self.out == "e4m3" else 2), (1 if self.kind == "e4m3" else 2)
      self._bound[hs] = f = gemm(dev, hs[0] + self.c_off * eo, hs[1] + self.a_off * ea, hs[2], self.M, self.N, self.K, kind=self.kind,
                                 out=self.out, alpha=self.alpha, sb=hs[3] if len(hs) > 3 else 0, impl=self.impl, bind=True)
    return f(wait=getenv("FZ_DEBUG", 0) >= 1)
  def pretty(self, indent: int = 0) -> str: return f"{'  ' * indent}CUDA {self.name} M={self.M} N={self.N} K={self.K}"

def _rows(x: Any) -> tuple[tuple[int, ...], int, Any]:
  lead = tuple(x.shape[:-1])
  return lead, prod(lead), (x.reshape(prod(lead), x.shape[-1]) if x.ndim != 2 else x)

def redo_split_k(device: str, dt: DType, M: int, N: int, K: int) -> bool:
  """A mixed forward's decode rows must be summed like the decode forwards sum them: the prefill tile's K order equals
  the decode swap-AB kernel's only without split-K (bench/korder_check.py), so a linear whose decode config splits K
  (7B o_proj / down_proj, SK 2) runs again on the decode rows with the decode kernel (redo_rows in linear)."""
  from fusezap.device.cuda_kernels import wg_swap_ok, wg_swap_default, ONE_K_ORDER
  dev = Device[device]
  return not ONE_K_ORDER[0] and wg_swap_ok(dev, _KIND[dt], False, M) and wg_swap_default(dev, _KIND[dt], M, N, K).SK > 1

def linear(x: Any, w: Any, b: Any = None, out_dtype: DType | None = None, redo_rows: tuple[int, int] | None = None) -> Any:
  """x (..., K) @ w (N, K)^T (+ b) on the 16-bit tensor-core GEMM; x and w f16/bf16 of one dtype; output in out_dtype
  (default that dtype; float32 keeps the f32 accumulator unrounded, e.g. for logits). With a bias the GEMM writes its
  f32 accumulator and one elementwise kernel adds b in f32 and rounds once to out_dtype: round(xW + b), as cuBLAS's bias
  epilogue (HF transformers, vLLM) computes it, not round(round(xW) + b).
  redo_rows (r0, n): rows r0 .. r0+n-1 are a mixed forward's decode rows; if the decode kernel splits K for this shape
  they are computed again with it (redo_split_k), so they carry the decode forwards' bits."""
  assert x.dtype == w.dtype and x.dtype in HALVES and w.ndim == 2 and x.shape[-1] == w.shape[-1], f"linear {x.shape} {x.dtype}, {w.shape} {w.dtype}"
  (N, K), (lead, M, x2), odt = w.shape, _rows(x), out_dtype or x.dtype
  gdt = dtypes.float32 if b is not None and not bias_round2() else odt
  xb, wb = _buf(x2), _buf(w)
  out = Buffer(xb.device, M * N, gdt)
  if M and N: _run(GemmCall(M, N, K, _KIND[x.dtype], _OUT[gdt]), (out, xb, wb))
  if redo_rows and M and N and redo_split_k(xb.device, x.dtype, redo_rows[1], N, K):
    r0, n = redo_rows
    _run(GemmCall(n, N, K, _KIND[x.dtype], _OUT[gdt], impl="wgswap", a_off=r0 * K, c_off=r0 * N), (out, xb, wb))
  y = _wrap(out, (M, N)).reshape(lead + (N,))
  if b is None: return y
  if gdt != dtypes.float32: return y + b   # FZ_BIAS_ROUND2: the old round(round(xW) + b)
  y = y + b.cast(dtypes.float32)
  return y if odt == dtypes.float32 else y.cast(odt)

# ---------------------------------------------------------------- RMSNorm (+ residual add)

NORM_THREADS = 256

def render_rmsnorm(name: str, dt: DType, K: int, eps: float, res: bool) -> str:
  """One 256-thread block per row of K (K % 8 == 0) f16/bf16 elements, 16-byte loads, the row kept in registers.
  res: XO = round(x + d), the residual stream, and the norm reads it. H = round(w * round(xf * r)) with
  r = 1 / sqrt(sum(xf * xf) * f32(1/K) + f32(eps)): the ops and roundings of Llama._rmsnorm's composed kernels
  (fusezap.codegen.cuda), except for the order of the f32 sum."""
  import numpy as np
  from fusezap.codegen.cuda import BF16_HELPERS, HALF_HELPERS
  hlp, ld, st, rd = (BF16_HELPERS, "fz_bf2f", "fz_f2bf", "fz_bfround") if dt == dtypes.bfloat16 else (HALF_HELPERS, "fz_h2f", "fz_f2h", "fz_hround")
  V, NT = K // 8, NORM_THREADS
  NV = -(-V // NT)
  def f(v: float) -> str: return f"{float(np.float32(v))!r}f"
  US = "unsigned short* __restrict__"
  args = [f"{US} H"] + ([f"{US} XO"] if res else []) + [f"const {US} X"] + ([f"const {US} D"] if res else []) + [f"const {US} W"]
  add = (f"""      const unsigned int* pb = (const unsigned int*)&b;
      unsigned int* pr = (unsigned int*)&r;
      lo = {rd}(lo + {ld}((unsigned short)(pb[k] & 0xffffu))); hi = {rd}(hi + {ld}((unsigned short)(pb[k] >> 16)));
      pr[k] = (unsigned int){st}(lo) | ((unsigned int){st}(hi) << 16);""" if res else "")
  return hlp + f"""
extern "C" __global__ void __launch_bounds__({NT}) {name}({', '.join(args)}) {{
  __shared__ float red[{NT // 32}];
  const int t = threadIdx.x;
  const long long o = (long long)blockIdx.x * {K};
  const uint4* x4 = (const uint4*)(X + o);{f'''
  const uint4* d4 = (const uint4*)(D + o);
  uint4* xo4 = (uint4*)(XO + o);''' if res else ''}
  const uint4* w4 = (const uint4*)W;
  uint4 wv[{NV}];   // the weight slices load before the reduction, so their latency hides behind it
  #pragma unroll
  for (int j = 0; j < {NV}; j++) if (t + j * {NT} < {V}) wv[j] = w4[t + j * {NT}];
  float v[{NV}][8];
  float ss = 0.f;
  #pragma unroll
  for (int j = 0; j < {NV}; j++) {{
    const int i = t + j * {NT};
    if (i < {V}) {{
      const uint4 a = x4[i];{f'''
      const uint4 b = d4[i];
      uint4 r;''' if res else ''}
      const unsigned int* pa = (const unsigned int*)&a;
      #pragma unroll
      for (int k = 0; k < 4; k++) {{
        float lo = {ld}((unsigned short)(pa[k] & 0xffffu)), hi = {ld}((unsigned short)(pa[k] >> 16));
{add}
        v[j][2 * k] = lo; v[j][2 * k + 1] = hi;
        ss += lo * lo; ss += hi * hi;
      }}{f'''
      xo4[i] = r;''' if res else ''}
    }}
  }}
  for (int s = 16; s > 0; s >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, s);
  if ((t & 31) == 0) red[t >> 5] = ss;
  __syncthreads();
  if (t < 32) {{
    ss = t < {NT // 32} ? red[t] : 0.f;
    for (int s = 16; s > 0; s >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, s);
    if (t == 0) red[0] = ss;
  }}
  __syncthreads();
  const float rs = 1.0f / sqrtf(red[0] * {f(1.0 / K)} + {f(eps)});
  uint4* h4 = (uint4*)(H + o);
  #pragma unroll
  for (int j = 0; j < {NV}; j++) {{
    const int i = t + j * {NT};
    if (i < {V}) {{
      const uint4 a = wv[j];
      uint4 r;
      const unsigned int* pa = (const unsigned int*)&a;
      unsigned int* pr = (unsigned int*)&r;
      #pragma unroll
      for (int k = 0; k < 4; k++) {{
        const float lo = {ld}((unsigned short)(pa[k] & 0xffffu)) * {rd}(v[j][2 * k] * rs);
        const float hi = {ld}((unsigned short)(pa[k] >> 16)) * {rd}(v[j][2 * k + 1] * rs);
        pr[k] = (unsigned int){st}(lo) | ((unsigned int){st}(hi) << 16);
      }}
      h4[i] = r;
    }}
  }}
}}"""

def rmsnorm_ok(x: Any) -> bool:
  return is_cuda(x.device) and x.dtype in HALVES and all(isinstance(s, int) for s in x.shape) and x.shape[-1] % 8 == 0
def rmsnorm(x: Any, w: Any, eps: float, d: Any = None) -> tuple[Any, Any]:
  """x (..., K) f16/bf16 on CUDA, w (K,), d (optional, like x) -> (x + d, w * rmsnorm(x + d)) in one kernel (render_rmsnorm);
  with d None, (x, w * rmsnorm(x)). Rows are independent."""
  assert rmsnorm_ok(x) and w.dtype == x.dtype and w.shape == x.shape[-1:] and (d is None or (d.shape == x.shape and d.dtype == x.dtype))
  (lead, M, x2), K = _rows(x), x.shape[-1]
  xb, wb = _buf(x2), _buf(w)
  h = Buffer(xb.device, M * K, x.dtype)
  nm = f"fz_rmsnorm{'_res' if d is not None else ''}_{_OUT[x.dtype]}_{K}"
  call = CudaCall(nm, render_rmsnorm(nm, x.dtype, K, eps, d is not None), (), (M, 1, 1), (NORM_THREADS, 1, 1))
  if d is None:
    if M: run_cuda(call, (h, xb, wb))
    return x, _wrap(h, (M, K)).reshape(lead + (K,))
  xo = Buffer(xb.device, M * K, x.dtype)
  if M: run_cuda(call, (h, xo, xb, _buf(_rows(d)[2]), wb), 2)
  return _wrap(xo, (M, K)).reshape(lead + (K,)), _wrap(h, (M, K)).reshape(lead + (K,))

# ---------------------------------------------------------------- SwiGLU and RoPE (elementwise, 16-byte vectors)

EW_THREADS = 256
FUSE_ROWS = 128   # Llama.forward_tokens uses silu_mul / rope_qkv from this many rows (prefill forwards) up

def _hlp(dt: DType) -> tuple[str, str, str, str]:
  from fusezap.codegen.cuda import BF16_HELPERS, HALF_HELPERS
  return (BF16_HELPERS, "fz_bf2f", "fz_f2bf", "fz_bfround") if dt == dtypes.bfloat16 else (HALF_HELPERS, "fz_h2f", "fz_f2h", "fz_hround")
def _f32lit(v: float) -> str:
  import numpy as np
  return f"{float(np.float32(v))!r}f"
_UNPACK = """static __device__ __forceinline__ void fz_unpack8(const uint4 a, float* v) {{
  const unsigned int* p = (const unsigned int*)&a;
  #pragma unroll
  for (int k = 0; k < 4; k++) {{ v[2 * k] = {ld}((unsigned short)(p[k] & 0xffffu)); v[2 * k + 1] = {ld}((unsigned short)(p[k] >> 16)); }}
}}
static __device__ __forceinline__ uint4 fz_pack8(const float* v) {{
  uint4 r;
  unsigned int* p = (unsigned int*)&r;
  #pragma unroll
  for (int k = 0; k < 4; k++) p[k] = (unsigned int){st}(v[2 * k]) | ((unsigned int){st}(v[2 * k + 1]) << 16);
  return r;
}}"""

def render_silu_mul(name: str, dt: DType, I: int) -> str:
  """O (M, I) = silu(G) * U for GU (M, 2I) = [G | U] f16/bf16 (I % 8 == 0), one thread per 8 outputs. The ops and roundings
  of the composed `g.silu() * u` (Tensor.silu, fusezap.codegen.cuda): s = round(g * (1 / (1 + exp2(g * -log2 e)))) in f32,
  then round(s * u), so the result is bit-identical to it."""
  from fusezap.helpers import LOG2E
  hlp, ld, st, rd = _hlp(dt)
  V = I // 8
  return hlp + "\n" + _UNPACK.format(ld=ld, st=st) + f"""
extern "C" __global__ void __launch_bounds__({EW_THREADS}) {name}(unsigned short* __restrict__ O, const unsigned short* __restrict__ GU, const int n) {{
  const int i = blockIdx.x * {EW_THREADS} + threadIdx.x;
  if (i >= n) return;
  const int r = i / {V}, c = i - r * {V};
  const uint4* g4 = (const uint4*)(GU + (long long)r * {2 * I});
  float g[8], u[8];
  fz_unpack8(g4[c], g);
  fz_unpack8(g4[c + {V}], u);
  #pragma unroll
  for (int k = 0; k < 8; k++) g[k] = {rd}({rd}(g[k] * (1.0f/(1.0f + exp2f(g[k] * {_f32lit(-LOG2E)})))) * u[k]);
  ((uint4*)O)[i] = fz_pack8(g);
}}"""

def silu_mul_ok(x: Any) -> bool: return rmsnorm_ok(x) and x.shape[-1] % 16 == 0
def silu_mul(gu: Any) -> Any:
  """gu (..., 2I) f16/bf16 on CUDA, the fused gate/up projection -> (..., I) silu(gate) * up (render_silu_mul). Rows are
  independent."""
  assert silu_mul_ok(gu), f"silu_mul {gu.shape} {gu.dtype}"
  (lead, M, x2), I = _rows(gu), gu.shape[-1] // 2
  xb = _buf(x2)
  out = Buffer(xb.device, M * I, gu.dtype)
  nm, n = f"fz_silu_mul_{_OUT[gu.dtype]}_{I}", M * I // 8
  if M: run_cuda(CudaCall(nm, render_silu_mul(nm, gu.dtype, I), (n,), (-(-n // EW_THREADS), 1, 1), (EW_THREADS, 1, 1)), (out, xb))
  return _wrap(out, (M, I)).reshape(lead + (I,))

def render_rope_qkv(name: str, dt: DType, H: int, Hr: int, hd: int, bias: bool, yf32: bool = False, kvw: tuple[int, int] | None = None) -> str:
  """O (N, H, hd) from Y (N, H*hd) (+ B (H*hd,)), CS / SN (N, hd/2) and B of dtype dt, f16/bf16 (hd % 16 == 0): heads
  below Hr are rotated (RoPE, rotate-half), the rest copied. One thread per 8 pairs (j, j + hd/2).
  yf32: Y is the qkv GEMM's float32 accumulator and x = round(y + b) is its single rounding (cuBLAS's bias epilogue,
  as HF transformers / vLLM compute it); otherwise Y is in dt and x = round(y + b) adds the bias to the rounded GEMM
  output (the composed `y + b`). Then round(round(x1 * c) + round(-x2 * s)) and round(round(x2 * c) + round(x1 * s)),
  the ops and roundings of the composed `(x1 * cos + (-x2) * sin).cat(x2 * cos + x1 * sin)` (Llama.forward_tokens), so
  the result is bit-identical to the composed path on the same x.
  kvw = (Hq, Hkv) (I60): the k heads [Hq, Hq + Hkv) and v heads [Hq + Hkv, H) also go into the paged KV pool P
  (2, NB, Hkv, bs, hd) at the row's slot M[moff + r] (slot < 0: a padding row, not stored): kv_cache.cu_kv_write's
  stores of the same values, one launch fewer per layer."""
  hlp, ld, st, rd = _hlp(dt)
  h2, P = hd // 2, hd // 16
  US = "const unsigned short* __restrict__"
  if bias: add = f"""
  float b[8];
  fz_unpack8(b4[c], b);
  for (int k = 0; k < 8; k++) x1[k] = {rd}(x1[k] + b[k]);
  fz_unpack8(b4[c + {h2 // 8}], b);
  for (int k = 0; k < 8; k++) x2[k] = {rd}(x2[k] + b[k]);"""
  elif yf32: add = f"""
  for (int k = 0; k < 8; k++) {{ x1[k] = {rd}(x1[k]); x2[k] = {rd}(x2[k]); }}"""
  else: add = ""
  if yf32: load = f"""
  const float4* y4 = (const float4*)Y + (long long)r * {H * hd // 4};
  float x1[8], x2[8];
  fz_load8f(y4 + 2 * c, x1);
  fz_load8f(y4 + 2 * (c + {h2 // 8}), x2);"""
  else: load = f"""
  const uint4* y4 = (const uint4*)Y + (long long)r * {H * hd // 8};
  float x1[8], x2[8];
  fz_unpack8(y4[c], x1);
  fz_unpack8(y4[c + {h2 // 8}], x2);"""
  YT = "const float* __restrict__" if yf32 else US
  ldf = """
static __device__ __forceinline__ void fz_load8f(const float4* p, float* v) {
  const float4 a = p[0], b = p[1];
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}""" if yf32 else ""
  if kvw is not None:
    Hq, Hkv = kvw
    assert Hq + 2 * Hkv == H and Hr == Hq + Hkv, f"kvw {kvw} with H {H}, Hr {Hr}"
    pargs, margs, xparams = ", unsigned short* __restrict__ P", ", const int* __restrict__ M", ", const int moff, const int bs, const int NB"
    kv_store = f"""
  if (hh >= {Hq}) {{
    const int slot = M[moff + r];
    if (slot >= 0) {{
      const int kv = hh >= {Hq + Hkv} ? 1 : 0, kh = hh - {Hq} - kv * {Hkv};
      const long long dst = ((((long long)kv * NB + slot / bs) * {Hkv} + kh) * bs + slot % bs) * {hd // 8};
      ((uint4*)P)[dst + j] = lo;
      ((uint4*)P)[dst + j + {h2 // 8}] = hi;
    }}
  }}"""
  else: pargs, margs, xparams, kv_store = "", "", "", ""
  return hlp + "\n" + _UNPACK.format(ld=ld, st=st) + ldf + f"""
extern "C" __global__ void __launch_bounds__({EW_THREADS}) {name}(unsigned short* __restrict__ O{pargs}, {YT} Y, {US} CS, {US} SN{f', {US} B' if bias else ''}{margs}, const int n{xparams}) {{
  const int i = blockIdx.x * {EW_THREADS} + threadIdx.x;
  if (i >= n) return;
  const int r = i / {H * P}, hh = (i / {P}) % {H}, j = i % {P};
  const long long o = ((long long)r * {H} + hh) * {hd // 8};
  const int c = hh * {hd // 8} + j;{f'''
  const uint4* b4 = (const uint4*)B;''' if bias else ''}{load}{add}
  uint4 lo;
  if (hh < {Hr}) {{
    float cs[8], sn[8], a[8];
    fz_unpack8(((const uint4*)CS)[(long long)r * {h2 // 8} + j], cs);
    fz_unpack8(((const uint4*)SN)[(long long)r * {h2 // 8} + j], sn);
    #pragma unroll
    for (int k = 0; k < 8; k++) {{
      a[k] = {rd}({rd}(x1[k] * cs[k]) + {rd}(-x2[k] * sn[k]));
      x2[k] = {rd}({rd}(x2[k] * cs[k]) + {rd}(x1[k] * sn[k]));
    }}
    lo = fz_pack8(a);
  }} else lo = fz_pack8(x1);
  const uint4 hi = fz_pack8(x2);
  ((uint4*)O)[o + j] = lo;
  ((uint4*)O)[o + j + {h2 // 8}] = hi;{kv_store}
}}"""

def rope_qkv_ok(y: Any, hd: int, cos: Any) -> bool:
  """y: the qkv GEMM output, in cos's dtype (f16/bf16) or its float32 accumulator."""
  return is_cuda(y.device) and cos.dtype in HALVES and y.dtype in (cos.dtype, dtypes.float32) and all(isinstance(s, int) for s in y.shape) \
    and y.ndim == 2 and y.shape[-1] % 8 == 0 and hd % 16 == 0
def fuse_rows() -> int:
  """Llama.forward_tokens runs silu_mul / rope_qkv from this many rows up (FZ_FUSE_ROWS, default FUSE_ROWS; read when a
  forward is traced). Both paths give the same bits, so this only moves time (I60 A/B: FZ_FUSE_ROWS=1 at decode)."""
  import os
  return int(os.environ.get("FZ_FUSE_ROWS", "") or FUSE_ROWS)
def rope_kvw_on() -> bool:
  """FZ_ROPE_KVW (default 1; read when a forward is traced): the paged forwards' RoPE kernel also writes the step's k / v
  rows into the KV pool (rope_qkv kvw=), so attend() skips its kv_write launch, at every row count (decode rows too)."""
  import os
  return os.environ.get("FZ_ROPE_KVW", "1") != "0"
def rope_qkv(y: Any, b: Any, cos: Any, sin: Any, H: int, Hr: int, hd: int, kvw: Any = None) -> Any:
  """y (N, H*hd) on CUDA, the qkv projection without its bias: its float32 GEMM accumulator (one rounding of y + b, as
  cuBLAS's bias epilogue) or already rounded to cos's dtype (round(round(y) + b)); b (H*hd,) or None, cos / sin (N, hd/2)
  f16/bf16 -> (N, H, hd) in cos's dtype: round(y + b) with heads [0, Hr) rotated by RoPE (render_rope_qkv). Rows are
  independent. kvw: a kv_cache.KVDest (the paged pool of this layer, the step metadata, its slot offset): the k / v heads
  are written into the pool too (the pool is the item's second output, as kv_write's first)."""
  assert rope_qkv_ok(y, hd, cos) and y.shape[1] == H * hd and cos.shape == sin.shape == (y.shape[0], hd // 2), f"rope_qkv {y.shape} {cos.shape}"
  dt, yf32 = cos.dtype, y.dtype == dtypes.float32
  assert (b is None or (b.shape == (H * hd,) and b.dtype == dt)) and sin.dtype == dt
  N, yb = y.shape[0], _buf(y)
  out = Buffer(yb.device, N * H * hd, dt)
  kv = None if kvw is None else (H - 2 * kvw.Hkv, kvw.Hkv)
  nm, n = f"fz_rope_qkv{'_b' if b is not None else ''}{'_y32' if yf32 else ''}{'_kvw' if kv else ''}_{_OUT[dt]}_{H}_{Hr}_{hd}", N * H * hd // 16
  ins = (yb, _buf(cos), _buf(sin)) + ((_buf(b),) if b is not None else ())
  bufs, params, n_out = ((out,) + ins, (n,), 1) if kvw is None else ((out, kvw.pool) + ins + (kvw.meta,), (n, kvw.moff, kvw.bs, kvw.NB), 2)
  assert kvw is None or (kvw.pool.dtype == dt and kvw.hd == hd)
  if N: run_cuda(CudaCall(nm, render_rope_qkv(nm, dt, H, Hr, hd, b is not None, yf32, kv), params, (-(-n // EW_THREADS), 1, 1), (EW_THREADS, 1, 1)), bufs, n_out)
  return _wrap(out, (N, H, hd))

# ---------------------------------------------------------------- greedy sampling (row argmax)

ARGMAX_THREADS = 1024

def render_argmax(name: str, V: int) -> str:
  """One 1024-thread block per row of V f32 logits -> the row's argmax as int32, numpy's rule: the first index of the
  maximum, and the first NaN if the row has one (NaN wins over every number)."""
  NT = ARGMAX_THREADS
  return f"""static __device__ __forceinline__ bool fz_better(float a, int ia, float b, int ib) {{
  const bool na = a != a, nb = b != b;
  if (na || nb) return na && (!nb || ia < ib);
  return a > b || (a == b && ia < ib);
}}
extern "C" __global__ void __launch_bounds__({NT}) {name}(int* __restrict__ O, const float* __restrict__ X) {{
  __shared__ float rv[{NT // 32}];
  __shared__ int ri[{NT // 32}];
  const int t = threadIdx.x;
  const float* x = X + (long long)blockIdx.x * {V};
  float bv = -__int_as_float(0x7f800000);
  int bi = 0x7fffffff;
  for (int i = t; i < {V}; i += {NT}) {{
    const float v = x[i];
    if (fz_better(v, i, bv, bi)) {{ bv = v; bi = i; }}
  }}
  for (int s = 16; s > 0; s >>= 1) {{
    const float ov = __shfl_xor_sync(0xffffffffu, bv, s);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, s);
    if (fz_better(ov, oi, bv, bi)) {{ bv = ov; bi = oi; }}
  }}
  if ((t & 31) == 0) {{ rv[t >> 5] = bv; ri[t >> 5] = bi; }}
  __syncthreads();
  if (t < 32) {{
    bv = rv[t]; bi = ri[t];
    for (int s = 16; s > 0; s >>= 1) {{
      const float ov = __shfl_xor_sync(0xffffffffu, bv, s);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, s);
      if (fz_better(ov, oi, bv, bi)) {{ bv = ov; bi = oi; }}
    }}
    if (t == 0) O[blockIdx.x] = bi < {V} ? bi : 0;
  }}
}}"""

def argmax_rows(x: Any) -> Any:
  """x (R, V) float32 on CUDA -> (R,) int32 row argmax (render_argmax: np.argmax's ties and NaN rule), so greedy
  decoding reads back R ints instead of R * V logits."""
  assert is_cuda(x.device) and x.dtype == dtypes.float32 and x.ndim == 2 and all(isinstance(s, int) for s in x.shape), f"argmax_rows {x.shape} {x.dtype}"
  (R, V), xb = x.shape, _buf(x)
  out = Buffer(xb.device, R, dtypes.int32)
  nm = f"fz_argmax_{V}"
  if R and V: run_cuda(CudaCall(nm, render_argmax(nm, V), (), (R, 1, 1), (ARGMAX_THREADS, 1, 1)), (out, xb))
  return _wrap(out, (R,))

# ---------------------------------------------------------------- FP8 (e4m3)

_LD = {dtypes.float32: ("float", "x[i]", ""),
       dtypes.float16: ("unsigned short", "fz_h2f(x[i])", 'static __device__ __forceinline__ float fz_h2f(unsigned short h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }\n'),
       dtypes.bfloat16: ("unsigned short", "__uint_as_float(((unsigned int)x[i]) << 16)", "")}

def render_rowquant(name: str, dt: DType) -> str:
  """One 256-thread block per row: s = max |x| / 448 (1 for an all-zero row), q = e4m3(x / s) (round to nearest even,
  saturating: cvt.rn.satfinite.e4m3x2). Q (M, K) uint8 e4m3 bits, S (M,) f32."""
  T, ld, hlp = _LD[dt]
  return hlp + f"""extern "C" __global__ void __launch_bounds__(256) {name}(unsigned char* __restrict__ Q, float* __restrict__ S, const {T}* __restrict__ X, const int K) {{
  __shared__ float red[8];
  const int t = threadIdx.x;
  const {T}* x = X + (long long)blockIdx.x * K;
  unsigned char* q = Q + (long long)blockIdx.x * K;
  float m = 0.f;
  for (int i = t; i < K; i += 256) m = fmaxf(m, fabsf({ld}));
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((t & 31) == 0) red[t >> 5] = m;
  __syncthreads();
  if (t < 32) {{
    m = t < 8 ? red[t] : 0.f;
    for (int o = 4; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (t == 0) red[0] = m;
  }}
  __syncthreads();
  const float s = red[0] > 0.f ? red[0] / 448.f : 1.f;
  if (t == 0) S[blockIdx.x] = s;
  for (int i = t; i < K; i += 256) {{
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.f), "f"({ld} / s));
    q[i] = (unsigned char)(r & 0xffu);
  }}
}}"""

def quantize_rows_fp8(x: Any) -> tuple[Any, Any]:
  """x (M, K) f32/f16/bf16 CUDA tensor -> (q (M, K) uint8 e4m3 bits, s (M, 1) f32) with x[r] ~= e4m3(q[r]) * s[r]."""
  M, K = x.shape
  xb = _buf(x)
  q, s = Buffer(xb.device, M * K, dtypes.uint8), Buffer(xb.device, M, dtypes.float32)
  nm = f"fz_fp8_rowquant_{_OUT[x.dtype]}"
  if M: run_cuda(CudaCall(nm, render_rowquant(nm, x.dtype), (K,), (M, 1, 1), (256, 1, 1)), (q, s, xb), 2)
  return _wrap(q, (M, K)), _wrap(s, (M, 1))

def fp8_linear(x: Any, wq: Any, ws: Any, b: Any = None, out_dtype: DType | None = None) -> Any:
  """W8A8 FP8 linear: x (..., K) float, wq (N, K) uint8 e4m3 bits, ws a (1,) f32 device tensor (w ~= e4m3(wq) * ws).
  x is quantized per row on the device, the e4m3 GEMM accumulates in f32 and applies ws, then each row is scaled by its
  activation scale and cast to out_dtype (default x.dtype); + b."""
  (N, K), (lead, M, x2) = wq.shape, _rows(x)
  assert wq.dtype == dtypes.uint8 and x.shape[-1] == K and fp8_ok(x.device, K), f"fp8_linear {x.shape}, {wq.shape} {wq.dtype} on {x.device}"
  xq, sx = quantize_rows_fp8(x2)
  qb, wb, sb = _buf(xq), _buf(wq), _buf(ws)
  out = Buffer(qb.device, M * N, dtypes.float32)
  if M and N: _run(GemmCall(M, N, K, "e4m3", "f32"), (out, qb, wb, sb))
  y = (_wrap(out, (M, N)) * sx).cast(out_dtype or x.dtype).reshape(lead + (N,))
  return y + b if b is not None else y

def quantize_fp8_weight(w: Any) -> tuple[Any, Any]:
  """Per-tensor e4m3 of a (N, K) CUDA weight on the device (fusezap.device.cuda_kernels.quantize_fp8: s = max |w| / 448,
  round to nearest even, saturating) -> (q uint8 (N, K), s (1,) f32). Eager (load time), not jit-recorded."""
  from fusezap.device.cuda_kernels import quantize_fp8
  q, s = quantize_fp8(w)
  return q.realize(), s.realize()

# ---------------------------------------------------------------- decode step inputs on the device (speed run 4)

def render_step_inputs(name: str, D: int, H2: int) -> str:
  """One launch builds a decode step's jit inputs from one packed int32 upload PK = [tok (N) | src (N) | pos (N) | meta]:
  X (N, D) = E[token] (token = IDS[src] when src >= 0, else tok: the previous decode's device argmax or a host token),
  CS / SN (N, H2) = the RoPE table rows at pos, MO = meta. 16-bit elements, 16 bytes per thread; rows are copies, so the
  inputs are bit-identical to the host-built ones."""
  dv, hv = D // 8, H2 // 8
  return f"""
extern "C" __global__ void __launch_bounds__({EW_THREADS}) {name}(uint4* __restrict__ X, uint4* __restrict__ CS, uint4* __restrict__ SN,
    int* __restrict__ MO, const uint4* __restrict__ E, const uint4* __restrict__ CT, const uint4* __restrict__ ST, const int* __restrict__ PK,
    const int* __restrict__ IDS, const int N, const int nmeta) {{
  long long i = (long long)blockIdx.x * {EW_THREADS} + threadIdx.x;
  if (i < (long long)N * {dv}) {{
    const int r = (int)(i / {dv}), c = (int)(i % {dv});
    const int s = PK[N + r];
    const int tok = s >= 0 ? IDS[s] : PK[r];
    X[i] = E[(long long)tok * {dv} + c];
    return;
  }}
  i -= (long long)N * {dv};
  if (i < (long long)N * {hv}) {{
    const int r = (int)(i / {hv}), c = (int)(i % {hv});
    const long long p = PK[2 * N + r];
    CS[i] = CT[p * {hv} + c];
    SN[i] = ST[p * {hv} + c];
    return;
  }}
  i -= (long long)N * {hv};
  if (i < nmeta) MO[i] = PK[3 * N + i];
}}"""

class DeviceSteps:
  """The decode step's inputs built on the device (LLMEngine on CUDA with FZ_LOOKAHEAD): the embedding and RoPE tables
  live on the device, each step uploads one packed int32 array through a pinned host ring with cuMemcpyHtoDAsync (no
  synchronous upload, so the host never waits for the GPU here), and the step's argmax ids come back through
  cuMemcpyDtoHAsync into pinned memory plus an event. That lets the engine enqueue the next decode before it reads the
  previous one's tokens."""
  RING = 4
  def __init__(self, device: str, embed: Any, cos: Any, sin: Any, dtype: DType):
    import numpy as np
    from fusezap.tensor import Tensor
    dev = Device[device]
    self.device, self.dev, self.cu, self.dtype = device, dev, dev.cu, dtype
    self.E = Tensor(embed, device=device, dtype=dtype).realize()
    self.CT, self.ST = Tensor(cos, device=device, dtype=dtype).realize(), Tensor(sin, device=device, dtype=dtype).realize()
    self.D, self.H2 = int(self.E.shape[1]), int(self.CT.shape[1])
    self._host: list[tuple[int, int]] = []   # pinned slots (address, bytes)
    self._k = 0
    self._np = np
  @staticmethod
  def ok(D: int, hd: int, dtype: DType) -> bool: return dtype in HALVES and D % 8 == 0 and (hd // 2) % 8 == 0
  def _pinned(self, nbytes: int) -> int:
    """The next pinned host slot of at least nbytes (a ring: a slot is rewritten RING uploads later, by then its copy ran)."""
    from fusezap.device.cuda import _check
    if not self._host or self._host[0][1] < nbytes:
      self._host = [(int(_check(self.cu.cuMemHostAlloc(max(nbytes, 1 << 16), 0), "cuMemHostAlloc")), max(nbytes, 1 << 16)) for _ in range(self.RING)]
    self._k = (self._k + 1) % self.RING
    return self._host[self._k][0]
  def inputs(self, tok: Any, src: Any, pos: Any, meta: Any, ids: Buffer | None) -> tuple[Any, Any, Any, Any]:
    """(x, cos, sin, meta) device tensors for N rows: tok / src / pos int arrays of length N, meta the step metadata;
    ids: the device int32 argmaxes that src indexes (None: every src is -1)."""
    import ctypes
    from fusezap.device.cuda import _check
    np = self._np
    N = len(tok)
    pk = np.concatenate([np.asarray(tok, np.int32), np.asarray(src, np.int32), np.asarray(pos, np.int32), np.asarray(meta, np.int32).reshape(-1)])
    h = self._pinned(pk.nbytes)
    ctypes.memmove(h, pk.ctypes.data, pk.nbytes)
    dev = self.dev
    PK = Buffer(self.device, len(pk), dtypes.int32).ensure_allocated()
    dev.ctx()
    _check(self.cu.cuMemcpyHtoDAsync(PK.handle, h, pk.nbytes, dev.stream), "cuMemcpyHtoDAsync")
    X = Buffer(self.device, N * self.D, self.dtype)
    CS, SN = Buffer(self.device, N * self.H2, self.dtype), Buffer(self.device, N * self.H2, self.dtype)
    nmeta = len(pk) - 3 * N
    MO = Buffer(self.device, nmeta, dtypes.int32)
    n = N * self.D // 8 + N * self.H2 // 8 + nmeta
    nm = f"fz_step_inputs_{self.D}_{self.H2}"
    run_cuda(CudaCall(nm, render_step_inputs(nm, self.D, self.H2), (N, nmeta), (-(-n // EW_THREADS), 1, 1), (EW_THREADS, 1, 1)),
             (X, CS, SN, MO, _buf(self.E), _buf(self.CT), _buf(self.ST), PK, ids if ids is not None else PK), 4)
    return _wrap(X, (N, self.D)), _wrap(CS, (N, self.H2)), _wrap(SN, (N, self.H2)), _wrap(MO, (nmeta,))
  def readback(self, ids: Buffer) -> "IdsReadback": return IdsReadback(self, ids)

class IdsReadback:
  """The (R,) int32 device argmaxes of a launched step, copied into pinned memory asynchronously; get() waits for that
  copy only (an event right after it), not for anything enqueued later."""
  def __init__(self, ds: DeviceSteps, ids: Buffer):
    from fusezap.device.cuda import _check
    self.ds, self.n = ds, ids.size
    self.h = ds._pinned(4 * self.n)
    dev, cu = ds.dev, ds.cu
    dev.ctx()
    _check(cu.cuMemcpyDtoHAsync(self.h, ids.handle, 4 * self.n, dev.stream), "cuMemcpyDtoHAsync")
    self.ev = _check(cu.cuEventCreate(2), "cuEventCreate")   # CU_EVENT_DISABLE_TIMING
    _check(cu.cuEventRecord(self.ev, dev.stream), "cuEventRecord")
    self._v = None
  def get(self) -> Any:
    if self._v is None:
      import ctypes
      from fusezap.device.cuda import _check
      np = self.ds._np
      _check(self.ds.cu.cuEventSynchronize(self.ev), "cuEventSynchronize (a decode step failed)")
      self._v = np.frombuffer((ctypes.c_int32 * self.n).from_address(self.h), np.int32).copy()
      self.ds.cu.cuEventDestroy(self.ev)
    return self._v
