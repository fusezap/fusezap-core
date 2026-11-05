"""Metal Shading Language renderer (SPEC §7.3). Implemented by WP5 by subclassing CStyleRenderer.

  kernel void name(device float* data0 [[buffer(0)]], ..., constant int& var0 [[buffer(n)]],
                   uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]])

Metal binds at most 31 buffers (indices 0..30) per kernel, so the argument layout depends on the counts
(metal_layout, shared with fusezap.device.metal): "plain" is the signature above while bufs + vars fit in 31 slots;
"packed" keeps the buffers at 0..n-1 and packs every var into one `constant int* fz_vars [[buffer(n)]]`; "argbuf"
(more than 30 buffers) passes the buffer GPU addresses in one `constant struct&` argument buffer at buffer(0) (Metal 3,
argument buffers tier 2) and the vars at buffer(1).

SPECIAL gidxN -> gid.{x,y,z}, lidxN -> lid.{x,y,z}. float16 is `half`, bfloat16 is native `bfloat` (MSL 3.1) in
registers and memory. Transcendentals on half/bfloat go through float. Compile with fast math off (fusezap.device.metal).
"""
from __future__ import annotations
import math, struct
from typing import Any, Callable
from fusezap.helpers import getenv
from fusezap.dtype import DType, dtypes
from fusezap.ops import UnaryOps
from fusezap.codegen.uops import UOp, PtrDType
from fusezap.codegen.cstyle import CStyleRenderer, _base_code_for_op

_F32_MAX = 3.4028234663852886e38

def _f32(x: float) -> float:
  """Round a python float to f32 (overflow -> inf), the way a host f32 tensor would hold it."""
  if math.isnan(x) or math.isinf(x): return x
  try: return struct.unpack("<f", struct.pack("<f", x))[0]
  except OverflowError: return math.copysign(math.inf, x)

def _bf16(x: float) -> float:
  """Round a python float to bfloat16 (round-to-nearest-even via f32; above the f32 range -> inf)."""
  if math.isnan(x) or math.isinf(x): return x
  if abs(x) > _F32_MAX: return math.copysign(math.inf, x)
  u = struct.unpack("<I", struct.pack("<f", x))[0]
  return struct.unpack("<f", struct.pack("<I", (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000))[0]

MAX_BUFFER_SLOTS = 31

def metal_layout(nbufs: int, nvars: int) -> str:
  """How a kernel with nbufs buffers and nvars int vars binds its arguments (see module docstring)."""
  if nbufs + nvars <= MAX_BUFFER_SLOTS: return "plain"
  return "packed" if nbufs < MAX_BUFFER_SLOTS else "argbuf"

def _metal_code_for_op() -> dict[Any, Callable[..., str]]:
  d = _base_code_for_op()
  def fm(fn: str) -> Callable[..., str]:
    # metal has precise float overloads; half/bfloat math is done in float and rounded back once
    return lambda a, dt: f"{fn}({a})" if dt == dtypes.float32 else f"(({MetalRenderer.type_map[dt]}){fn}((float)({a})))"
  d.update({UnaryOps.EXP2: fm("exp2"), UnaryOps.LOG2: fm("log2"), UnaryOps.SIN: fm("sin"), UnaryOps.SQRT: fm("sqrt"),
            UnaryOps.RECIP: lambda a, dt: f"(({MetalRenderer.type_map[dt]})1/({a}))"})
  return d

class MetalRenderer(CStyleRenderer):
  device = "METAL"
  has_local = True
  global_max = (2**31 - 1, 65535, 65535)
  kernel_prefix = "kernel void"
  buffer_prefix = "device "
  arg_int_prefix = "constant int&"
  type_map = {dtypes.bool: "bool", dtypes.uint8: "uchar", dtypes.int8: "char", dtypes.int16: "short", dtypes.int32: "int",
              dtypes.int64: "long", dtypes.float16: "half", dtypes.bfloat16: "bfloat", dtypes.float32: "float"}
  code_for_op = _metal_code_for_op()

  def prelude(self, uops: list[UOp]) -> str: return "#include <metal_stdlib>\nusing namespace metal;"
  def render_special(self, uop: UOp) -> str:
    axis, name, _ = uop.arg
    return f"(int){'lid' if name.startswith('l') else 'gid'}.{'xyz'[axis]}"
  def float_literal(self, x: float) -> str: return super().float_literal(_f32(x))
  def render_const(self, val: Any, dt: DType) -> str:
    if dt == dtypes.bfloat16: return self.render_cast(dt, self.float_literal(_bf16(float(val))))
    if dt == dtypes.int64: return f"((long){int(val)})"   # MSL has no `long long`; avoid the ll suffix
    return super().render_const(val, dt)
  def render_cast(self, dt: DType, x: str) -> str:
    src = getattr(self._cur, "src", ())
    sdt = src[0].dtype if len(src) == 1 and not isinstance(src[0].dtype, PtrDType) else None
    # bfloat <-> non-float conversions go through float (MSL defines bfloat conversions from float/half)
    if dt == dtypes.bfloat16 and sdt is not None and not sdt.is_float: return f"((bfloat)(float)({x}))"
    if sdt == dtypes.bfloat16 and not dt.is_float:
      return f"((bool)((float)({x})))" if dt == dtypes.bool else f"(({self.tname(dt)})(float)({x}))"
    return f"(({self.tname(dt)})({x}))"
  def render_bitcast(self, dt: DType, x: str) -> str:
    # MSL refuses as_type to/from bool; bool is one byte holding 0/1, so a value cast is the same bits (nonzero -> true)
    src = getattr(self._cur, "src", ())
    sdt = src[0].dtype if len(src) == 1 else None
    if dt == dtypes.bool: return f"(({x}) != 0)"
    if sdt == dtypes.bool: return f"(({self.tname(dt)})({x}))"
    return f"as_type<{self.tname(dt)}>({x})"
  def render_kernel(self, name: str, body_lines: list[str], bufs: list[tuple[str, PtrDType, bool]], vars: list[tuple[str, Any]]) -> str:
    layout, pre, head = metal_layout(len(bufs), len(vars)), self.prelude(self._uops), []
    if layout == "argbuf":
      pre += f"\nstruct fz_args_{name} {{ " + " ".join(f"device {self.tname(dt.base)}* {n};" for n, dt, _ in bufs) + " };"
      args = [f"constant fz_args_{name}& fz_args [[buffer(0)]]"]
      head += [f"  device {self.tname(dt.base)}* {n} = fz_args.{n};" for n, dt, _ in bufs]
    else: args = [f"device {self.tname(dt.base)}* {n} [[buffer({i})]]" for i, (n, dt, _) in enumerate(bufs)]
    if layout == "plain": args += [f"{self.arg_int_prefix} {n} [[buffer({len(bufs) + i})]]" for i, (n, _) in enumerate(vars)]
    elif vars:
      args.append(f"constant int* fz_vars [[buffer({1 if layout == 'argbuf' else len(bufs)})]]")
      head += [f"  int {n} = fz_vars[{i}];" for i, (n, _) in enumerate(vars)]
    args += ["uint3 gid [[threadgroup_position_in_grid]]", "uint3 lid [[thread_position_in_threadgroup]]"]
    return pre + "\n" + f"{self.kernel_prefix} {name}({', '.join(args)}) {{\n" + "\n".join(head + body_lines) + "\n}\n"

# ---------------------------------------------------------------- custom kernels (SPEC §11.1, P2-MTL)
# Library kernels for the P2-CUSTOM hook, rendered as MSL source. Every param is an int `constant int&` after the buffers
# (the "plain" metal_layout, which MetalProgram uses for <= 31 slots). Accumulation is float32 for every input dtype.

MATMUL_PARAMS = ("Bt", "M", "N", "K", "sab", "sam", "sak", "sbb", "sbk", "sbn", "scb", "scm", "scn", "oa", "ob")
ATTN_PARAMS = ("Bt", "L", "S", "D", "Dv", "sqb", "sql", "sqd", "skb", "sks", "skd", "svb", "svs", "svd", "sob", "sol", "sod",
               "oq", "ok", "ov", "scale_bits", "log2e_bits")
MATMUL_TILE = (4, 2, 4, 4)   # simdgroups along M, along N; 8x8 fragments per simdgroup along M, along N (128x64 block, 256 threads)
MATMUL_GROUP = 8             # block rows per band of the threadgroup walk
_MSL_T = {dtypes.float32: "float", dtypes.float16: "half", dtypes.bfloat16: "bfloat"}

def _msl_args(bufs: list[str], params: tuple[str, ...]) -> str:
  a = bufs + [f"constant int& {p} [[buffer({len(bufs) + i})]]" for i, p in enumerate(params)]
  return ", ".join(a + ["uint3 gid [[threadgroup_position_in_grid]]", "uint tid [[thread_index_in_threadgroup]]",
                        "uint sg [[simdgroup_index_in_threadgroup]]", "uint lane [[thread_index_in_simdgroup]]"])

def matmul_launch(M: int, N: int, Bt: int, tile: tuple[int, int, int, int] = MATMUL_TILE) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
  """(global_size, local_size) in threadgroups / threads for render_matmul: every (batch, block row, block column) on x."""
  WM, WN, FM, FN = tile
  return ((-(-N // (WN * FN * 8))) * (-(-M // (WM * FM * 8))) * Bt, 1, 1), (WM * WN * 32, 1, 1)

def render_matmul(name: str = "fz_matmul", dt: DType = dtypes.float32, out_dt: DType = dtypes.float32,
                  tile: tuple[int, int, int, int] = MATMUL_TILE, group: int = MATMUL_GROUP) -> str:
  """C[b, m, n] = sum_k A[b, m, k] * B[b, k, n] with simdgroup 8x8 matrices (inputs `dt`, float accumulators, output
  `out_dt`). Params MATMUL_PARAMS (strides in elements, offsets on A/B). A simdgroup whose 8*FM x 8*FN tile lies inside
  the output and whose operands have a unit stride on one axis loads fragments straight from device memory (transposed
  loads for the other orientation); edge tiles, the K tail and general strides stage zero-padded tiles in threadgroup
  memory. Full tiles with a unit-stride output row are stored directly, the rest element by element. Threadgroups walk
  the blocks column-major within bands of `group` block rows, so the tiles in flight share A rows and B columns in cache."""
  WM, WN, FM, FN = tile
  T, OT, SM, SN, NS = _MSL_T[dt], _MSL_T[out_dt], FM * 8, FN * 8, WM * WN
  args = _msl_args([f"device {OT}* C [[buffer(0)]]", f"device const {T}* A [[buffer(1)]]", f"device const {T}* B [[buffer(2)]]"], MATMUL_PARAMS)
  IJ = [(i, j) for i in range(FM) for j in range(FN)]
  # unrolled in python: an array of simdgroup matrices indexed in a loop can end up in stack memory
  zero = "".join(f"  simdgroup_float8x8 c{i}_{j} = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n" for i, j in IJ)
  decl = f"    simdgroup_matrix<{T}, 8, 8> " + ", ".join([f"a{i}" for i in range(FM)] + [f"b{j}" for j in range(FN)]) + ";\n"
  mma = "".join(f"    simdgroup_multiply_accumulate(c{i}_{j}, a{i}, b{j}, c{i}_{j});\n" for i, j in IJ)
  ld_dev = "".join(f"    simdgroup_load(a{i}, a + (long)(m0 + {8 * i})*sam + (long)k*sak, ea, ulong2(0, 0), aT);\n" for i in range(FM)) + \
           "".join(f"    simdgroup_load(b{j}, b + (long)k*sbk + (long)(n0 + {8 * j})*sbn, eb, ulong2(0, 0), bT);\n" for j in range(FN))
  ld_tg = "".join(f"    simdgroup_load(a{i}, sA + {64 * i}, 8);\n" for i in range(FM)) + "".join(f"    simdgroup_load(b{j}, sB + {8 * j}, {SN});\n" for j in range(FN))
  st_dev = "".join(f"    simdgroup_store(c{i}_{j}, (device float*)c + (long)(m0 + {8 * i})*scm + n0 + {8 * j}, (ulong)scm);\n" for i, j in IJ)
  st_tg = "".join(f"""    simdgroup_store(c{i}_{j}, sC, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = lane; e < 64; e += 32) {{ int m = m0 + {8 * i} + e / 8, n = n0 + {8 * j} + e % 8; if (m < M && n < N) c[(long)m*scm + (long)n*scn] = ({OT})sC[e]; }}
    simdgroup_barrier(mem_flags::mem_threadgroup);
""" for i, j in IJ)
  direct_store = "scn == 1" if OT == "float" else "false"
  return f"""#include <metal_stdlib>
using namespace metal;
kernel void {name}({args}) {{
  threadgroup {T} tA[{NS}][{SM * 8}], tB[{NS}][{8 * SN}];
  threadgroup float tC[{NS}][64];
  const int nbn = (N + {WN * SN - 1}) / {WN * SN}, nbm = (M + {WM * SM - 1}) / {WM * SM}, bz = gid.x / (nbn * nbm), id = gid.x % (nbn * nbm);
  const int band = id / ({group} * nbn), rows = min({group}, nbm - band * {group}), bm = band * {group} + (id % ({group} * nbn)) % rows, bn = (id % ({group} * nbn)) / rows;
  const int m0 = bm*{WM * SM} + (sg/{WN})*{SM}, n0 = bn*{WN * SN} + (sg%{WN})*{SN};
  device const {T}* a = A + oa + (long)bz*sab;
  device const {T}* b = B + ob + (long)bz*sbb;
  device {OT}* c = C + (long)bz*scb;
  threadgroup {T}* sA = tA[sg]; threadgroup {T}* sB = tB[sg];
{zero}  const bool aN = sak == 1, aT = !aN && sam == 1, bN = sbn == 1, bT = !bN && sbk == 1;
  const bool inside = (aN || aT) && (bN || bT) && m0 + {SM} <= M && n0 + {SN} <= N;
  const ulong ea = aN ? sam : sak, eb = bN ? sbk : sbn;
  int k = 0;
  if (inside) for (; k + 8 <= K; k += 8) {{
{decl}{ld_dev}{mma}  }}
  for (; k < K; k += 8) {{
    for (int e = lane; e < {SM * 8}; e += 32) {{ int r = m0 + e / 8, q = k + e % 8; sA[e] = (r < M && q < K) ? a[(long)r*sam + (long)q*sak] : ({T})0; }}
    for (int e = lane; e < {8 * SN}; e += 32) {{ int q = k + e / {SN}, n = n0 + e % {SN}; sB[e] = (n < N && q < K) ? b[(long)q*sbk + (long)n*sbn] : ({T})0; }}
    simdgroup_barrier(mem_flags::mem_threadgroup);
{decl}{ld_tg}{mma}    simdgroup_barrier(mem_flags::mem_threadgroup);
  }}
  if (m0 + {SM} <= M && n0 + {SN} <= N && {direct_store}) {{
{st_dev}  }} else {{
    threadgroup float* sC = tC[sg];
{st_tg}  }}
}}
"""

ATTN_ROWS, ATTN_SG = 8, 4   # query rows per simdgroup, simdgroups per threadgroup (32 query rows per threadgroup)
_TG_MEM = 32 * 1024 - 1024

def attention_block(D: int, Dv: int) -> int | None:
  """Keys per block for render_attention (largest of 32/16/8 whose float tiles fit threadgroup memory), None if none fits."""
  DP, DVP, R = -(-D // 8) * 8, -(-Dv // 8) * 8, ATTN_ROWS * ATTN_SG
  for BK in (32, 16, 8):
    if 4 * (R * DP + BK * DP + BK * DVP + ATTN_SG * ATTN_ROWS * BK + ATTN_SG * 64 * 2) <= _TG_MEM: return BK
  return None

def attention_launch(L: int, Bt: int, D: int, Dv: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
  if attention_block(D, Dv) is None: return (-(-(L * Bt) // 64), 1, 1), (64, 1, 1)
  return (-(-L // (ATTN_ROWS * ATTN_SG)) * Bt, 1, 1), (ATTN_SG * 32, 1, 1)

def render_attention(name: str, D: int, Dv: int, dt: DType = dtypes.float32, out_dt: DType = dtypes.float32, causal: bool = False) -> str:
  """Flash attention: O[b, l] = softmax_s(scale * Q[b, l] . K[b, :, s], masked) @ V[b, s] with the online softmax
  (running max and sum per row, rescaling the accumulator per key block); K is indexed (b, d, s) like the composed
  q @ k^T. causal masks s > l (torch's top-left aligned tril). Params ATTN_PARAMS; scale and log2(e) are f32 bit patterns.
  Each simdgroup owns 8 query rows; per block of BK keys the threadgroup stages K and V in threadgroup memory (zero
  padded, any strides), scores and P @ V use simdgroup 8x8 float matrices, and the per-row softmax runs on 4 lanes per row.
  Head dims too large for threadgroup memory use a one-thread-per-row loop instead (render_attention_rows)."""
  if (BK := attention_block(D, Dv)) is None: return render_attention_rows(name, Dv, dt, out_dt, causal)
  T, OT, DP, DVP, R, NS = _MSL_T[dt], _MSL_T[out_dt], -(-D // 8) * 8, -(-Dv // 8) * 8, ATTN_ROWS * ATTN_SG, ATTN_SG
  CPL = BK // 4   # score columns per lane
  args = _msl_args([f"device {OT}* O [[buffer(0)]]", f"device const {T}* Q [[buffer(1)]]", f"device const {T}* Kb [[buffer(2)]]",
                    f"device const {T}* V [[buffer(3)]]"], ATTN_PARAMS)
  mask = "s >= S" + (" || s > l" if causal else "")
  kend = "min(S, l0 + " + str(R) + ")" if causal else "S"
  return f"""#include <metal_stdlib>
using namespace metal;
kernel void {name}({args}) {{
  threadgroup float tQ[{R * DP}], tK[{BK * DP}], tV[{BK * DVP}], tS[{NS}][{ATTN_ROWS * BK}], tD[{NS}][64], tO[{NS}][64];
  const int nbl = (L + {R - 1}) / {R}, bz = gid.x / nbl, l0 = (gid.x % nbl) * {R}, r0 = l0 + {ATTN_ROWS}*(int)sg;
  const float scale = as_type<float>(scale_bits), log2e = as_type<float>(log2e_bits);
  device const {T}* q = Q + oq + (long)bz*sqb;
  device const {T}* kp = Kb + ok + (long)bz*skb;
  device const {T}* v = V + ov + (long)bz*svb;
  device {OT}* o = O + (long)bz*sob;
  for (int e = tid; e < {R * DP}; e += {NS * 32}) {{ int r = l0 + e / {DP}, d = e % {DP}; tQ[e] = (r < L && d < D) ? (float)q[(long)r*sql + (long)d*sqd] : 0.0f; }}
  threadgroup float* sS = tS[sg]; threadgroup float* sD = tD[sg];
  for (int e = lane; e < 64; e += 32) sD[e] = 0.0f;
  simdgroup_float8x8 acc[{DVP // 8}];
  for (int j = 0; j < {DVP // 8}; j++) acc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  const int row = lane / 4, part = lane % 4, l = r0 + row;
  float m = -INFINITY, den = 0.0f;
  const int kend = {kend};
  for (int s0 = 0; s0 < kend; s0 += {BK}) {{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = tid; e < {BK * DP}; e += {NS * 32}) {{ int s = s0 + e / {DP}, d = e % {DP}; tK[e] = (s < S && d < D) ? (float)kp[(long)d*skd + (long)s*sks] : 0.0f; }}
    for (int e = tid; e < {BK * DVP}; e += {NS * 32}) {{ int s = s0 + e / {DVP}, d = e % {DVP}; tV[e] = (s < S && d < Dv) ? (float)v[(long)s*svs + (long)d*svd] : 0.0f; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = 0; j < {BK // 8}; j++) {{
      simdgroup_float8x8 st = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), fq, fk;
      for (int d = 0; d < {DP // 8}; d++) {{
        simdgroup_load(fq, tQ + {ATTN_ROWS}*sg*{DP} + 8*d, {DP});
        simdgroup_load(fk, tK + 8*j*{DP} + 8*d, {DP}, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(st, fq, fk, st);
      }}
      simdgroup_store(st, sS + 8*j, {BK});
    }}
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float t[{CPL}], bm = -INFINITY;
    for (int c = 0; c < {CPL}; c++) {{
      int s = s0 + part*{CPL} + c;
      float x = sS[row*{BK} + part*{CPL} + c] * scale;
      t[c] = ({mask}) ? -INFINITY : x;
      bm = (t[c] == t[c]) ? (t[c] > bm ? t[c] : bm) : t[c];
    }}
    float o1 = simd_shuffle_xor(bm, 1); bm = (o1 == o1) ? (o1 > bm ? o1 : bm) : o1;
    float o2 = simd_shuffle_xor(bm, 2); bm = (o2 == o2) ? (o2 > bm ? o2 : bm) : o2;
    const float mn = (bm == bm) ? (bm > m ? bm : m) : bm;
    const float alpha = (mn == -INFINITY) ? 1.0f : exp2((m - mn) * log2e);
    float ps = 0.0f;
    for (int c = 0; c < {CPL}; c++) {{
      float p = (mn == -INFINITY) ? 0.0f : exp2((t[c] - mn) * log2e);
      sS[row*{BK} + part*{CPL} + c] = p; ps += p;
    }}
    ps += simd_shuffle_xor(ps, 1); ps += simd_shuffle_xor(ps, 2);
    den = den * alpha + ps; m = mn;
    if (part == 0) sD[row*9] = alpha;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 fd, fp, fv;
    simdgroup_load(fd, sD, 8);
    for (int e = 0; e < {DVP // 8}; e++) simdgroup_multiply(acc[e], fd, acc[e]);
    for (int j = 0; j < {BK // 8}; j++) {{
      simdgroup_load(fp, sS + 8*j, {BK});
      for (int e = 0; e < {DVP // 8}; e++) {{ simdgroup_load(fv, tV + 8*j*{DVP} + 8*e, {DVP}); simdgroup_multiply_accumulate(acc[e], fp, fv, acc[e]); }}
    }}
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }}
  if (part == 0) sD[row*9] = S == 0 ? 0.0f : 1.0f / den;   // no keys: P @ V sums nothing, like the composed path
  simdgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 fd;
  simdgroup_load(fd, sD, 8);
  threadgroup float* sO = tO[sg];
  for (int e = 0; e < {DVP // 8}; e++) {{
    simdgroup_multiply(acc[e], fd, acc[e]);
    simdgroup_store(acc[e], sO, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int x = lane; x < 64; x += 32) {{ int r = r0 + x / 8, d = 8*e + x % 8; if (r < L && d < Dv) o[(long)r*sol + (long)d*sod] = ({OT})sO[x]; }}
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }}
}}
"""

# ---------------------------------------------------------------- I81 D8: flash attention for long sequences / wide heads

def flash_config(D: int, Dv: int) -> tuple[int, int, int] | None:
  """(G, C, BK) for render_attention_flash, or None (head dims not multiples of 8, or too wide): G groups of 8 query
  rows per threadgroup, C simdgroups per group splitting the head dims (each holds D/C of Q and Dv/C of the accumulator
  in registers: <= 128 + 128 floats per 8 rows), BK keys per block. FZ_METAL_FLASH_CFG=G,C,BK overrides (benchmarks)."""
  if D % 8 or Dv % 8 or D <= 0 or Dv <= 0: return None
  if (o := getenv("FZ_METAL_FLASH_CFG", "")):
    G, C, BK = (int(x) for x in str(o).split(","))
    return (G, C, BK) if D % (8 * C) == 0 and Dv % (8 * C) == 0 else None
  # tuned on an M5 Max shared with other jobs (I81 D8, scripts/diff_metal_bench.py; GPU-time medians): 24 x 4608 x 128
  # best at (2, 2, 64), 1 x 16384 x 512 at (2, 8, 32) (all within ~10% there)
  C = 2 if D % 16 == 0 and Dv % 16 == 0 else 1
  while (D // C > 64 or Dv // C > 64) and C < 8 and D % (16 * C) == 0 and Dv % (16 * C) == 0: C *= 2
  if D // C > 128 or Dv // C > 128: return None
  return (2, C, 64) if C <= 2 else (2, C, 32)

def attention_flash_launch(L: int, Bt: int, G: int, C: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
  return (-(-L // (8 * G)) * Bt, 1, 1), (G * C * 32, 1, 1)

def render_attention_flash(name: str, D: int, Dv: int, causal: bool = False, cfg: tuple[int, int, int] | None = None) -> str:
  """Flash attention for float32, any L / S, head dims multiples of 8 up to 128 * C (I81 D8: FLUX.2 joint attention at
  4608 tokens x 128, the VAE mid-block at 16384 tokens x 512). Same math and params (ATTN_PARAMS) as render_attention.
  A threadgroup has G groups of 8 query rows; group g's C simdgroups each hold the Q rows' slice of D / C head dims as
  simdgroup fragments in registers and accumulate O's slice of Dv / C value dims. Per block of BK keys: each simdgroup
  computes its partial scores (8 rows x BK keys over its D slice), the group's first simdgroup sums the C partials (in
  slice order), applies scale and masks and runs the online softmax (4 lanes per row; running max m and sum den), and
  every simdgroup rescales its accumulator by alpha (a diagonal 8x8 matrix) and adds P @ V for its value slice. K, V
  and Q fragments load straight from device memory when the slice has unit stride along the head dim and the 8 rows /
  keys are in range, else through a zero-padded 8x8 staging tile. Output: 1 / den on the diagonal, stored directly or
  through the staging tile at edges."""
  G, C, BK = cfg or flash_config(D, Dv)  # type: ignore[misc]
  DC, DVC, NS, CPL = D // C, Dv // C, G * C, BK // 4
  assert DC % 8 == 0 and DVC % 8 == 0 and BK % 8 == 0
  args = _msl_args(["device float* O [[buffer(0)]]", "device const float* Q [[buffer(1)]]", "device const float* Kb [[buffer(2)]]",
                    "device const float* V [[buffer(3)]]"], ATTN_PARAMS)
  R = 8 * G
  mask = "s >= S" + (" || s > l" if causal else "")
  kend = f"min(S, l0 + {R})" if causal else "S"
  # 8x8 fragment from device memory at (row r, col c) of a (rows x cols) operand with strides (sr, sc); `full`: in range
  def frag(dst: str, base: str, r: str, c: str, sr: str, sc: str, rmax: str, cmax: str, transpose: bool) -> str:
    tr = "true" if transpose else "false"
    return f"""    if ({sc} == 1 && {r} + 8 <= {rmax} && {c} + 8 <= {cmax}) simdgroup_load({dst}, {base} + (long)({r})*{sr} + ({c}), {sr}, ulong2(0, 0), {tr});
    else {{
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (int ei = lane; ei < 64; ei += 32) {{ int rr = {r} + ei / 8, cc = {c} + ei % 8; sX[ei] = (rr < {rmax} && cc < {cmax}) ? {base}[(long)rr*{sr} + (long)cc*{sc}] : 0.0f; }}
      simdgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_load({dst}, sX, 8, ulong2(0, 0), {tr});
    }}
"""
  qload = "".join(frag(f"qf[{d}]", "q", "r0", f"{d * 8} + c * {DC}", "sql", "sqd", "L", "D", False) for d in range(DC // 8))
  return f"""#include <metal_stdlib>
using namespace metal;
kernel void {name}({args}) {{
  threadgroup float tS[{NS}][{8 * BK}], tP[{G}][{8 * BK}], tA[{G}][64], tX[{NS}][64];
  const int nbl = (L + {R - 1}) / {R}, bz = gid.x / nbl, l0 = (gid.x % nbl) * {R};
  const int g = sg / {C}, c = sg % {C}, r0 = l0 + 8 * g;
  const float scale = as_type<float>(scale_bits), log2e = as_type<float>(log2e_bits);
  device const float* q = Q + oq + (long)bz*sqb;
  device const float* kp = Kb + ok + (long)bz*skb;
  device const float* v = V + ov + (long)bz*svb;
  device float* o = O + (long)bz*sob;
  threadgroup float* sX = tX[sg]; threadgroup float* sS = tS[sg]; threadgroup float* sP = tP[g]; threadgroup float* sA = tA[g];
  for (int e = lane; e < 64; e += 32) if (c == 0) sA[e] = 0.0f;
  simdgroup_float8x8 qf[{DC // 8}], acc[{DVC // 8}];
{qload}  for (int e = 0; e < {DVC // 8}; e++) acc[e] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  const int row = lane / 4, part = lane % 4, l = r0 + row;
  float m = -INFINITY, den = 0.0f;
  const int kend = {kend};
  for (int s0 = 0; s0 < kend; s0 += {BK}) {{
    for (int j = 0; j < {BK // 8}; j++) {{
      simdgroup_float8x8 st = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), fk;
      const int sj = s0 + 8 * j;
      for (int d = 0; d < {DC // 8}; d++) {{
        const int dd = c * {DC} + 8 * d;
{frag("fk", "kp", "sj", "dd", "sks", "skd", "S", "D", True).replace(chr(10) + "    ", chr(10) + "        ").replace("    if (", "        if (", 1)}        simdgroup_multiply_accumulate(st, qf[d], fk, st);
      }}
      simdgroup_store(st, sS + 8 * j, {BK});
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c == 0) {{
      float t[{CPL}], bm = -INFINITY;
      for (int k = 0; k < {CPL}; k++) {{
        const int col = part * {CPL} + k, s = s0 + col;
        float x = tS[sg][row * {BK} + col];
        for (int cc = 1; cc < {C}; cc++) x += tS[sg + cc][row * {BK} + col];
        x *= scale;
        t[k] = ({mask}) ? -INFINITY : x;
        bm = (t[k] == t[k]) ? (t[k] > bm ? t[k] : bm) : t[k];
      }}
      float o1 = simd_shuffle_xor(bm, 1); bm = (o1 == o1) ? (o1 > bm ? o1 : bm) : o1;
      float o2 = simd_shuffle_xor(bm, 2); bm = (o2 == o2) ? (o2 > bm ? o2 : bm) : o2;
      const float mn = (bm == bm) ? (bm > m ? bm : m) : bm;
      const float alpha = (mn == -INFINITY) ? 1.0f : exp2((m - mn) * log2e);
      float ps = 0.0f;
      for (int k = 0; k < {CPL}; k++) {{
        const float p = (mn == -INFINITY) ? 0.0f : exp2((t[k] - mn) * log2e);
        sP[row * {BK} + part * {CPL} + k] = p; ps += p;
      }}
      ps += simd_shuffle_xor(ps, 1); ps += simd_shuffle_xor(ps, 2);
      den = den * alpha + ps; m = mn;
      if (part == 0) sA[row * 9] = alpha;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 fd, fp, fv;
    simdgroup_load(fd, sA, 8);
    for (int e = 0; e < {DVC // 8}; e++) simdgroup_multiply(acc[e], fd, acc[e]);
    for (int j = 0; j < {BK // 8}; j++) {{
      const int sj = s0 + 8 * j;
      simdgroup_load(fp, sP + 8 * j, {BK});
      for (int e = 0; e < {DVC // 8}; e++) {{
        const int dv = c * {DVC} + 8 * e;
{frag("fv", "v", "sj", "dv", "svs", "svd", "S", "Dv", False).replace(chr(10) + "    ", chr(10) + "        ").replace("    if (", "        if (", 1)}        simdgroup_multiply_accumulate(acc[e], fp, fv, acc[e]);
      }}
    }}
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (c == 0 && part == 0) sA[row * 9] = S == 0 ? 0.0f : 1.0f / den;   // no keys: P @ V sums nothing, like the composed path
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 fd;
  simdgroup_load(fd, sA, 8);
  for (int e = 0; e < {DVC // 8}; e++) {{
    simdgroup_multiply(acc[e], fd, acc[e]);
    const int dv = c * {DVC} + 8 * e;
    if (sod == 1 && r0 + 8 <= L) simdgroup_store(acc[e], o + (long)r0*sol + dv, sol);
    else {{
      simdgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_store(acc[e], sX, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (int x = lane; x < 64; x += 32) {{ int r = r0 + x / 8, d = dv + x % 8; if (r < L && d < Dv) o[(long)r*sol + (long)d*sod] = sX[x]; }}
    }}
  }}
}}
"""

# ---------------------------------------------------------------- I81 D8: implicit-GEMM conv2d

CONV_TILE = (2, 4, 4, 4)   # simdgroups along M (output channels), along N (pixels); 8x8 fragments per simdgroup (64x128 block)
CONV_KS = 32               # input channels staged per step

def conv_launch(M: int, N: int, tile: tuple[int, int, int, int] | None = None) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
  WM, WN, FM, FN = tile or _conv_cfg(M)[0]
  return ((-(-N // (WN * FN * 8))) * (-(-M // (WM * FM * 8))), 1, 1), (WM * WN * 32, 1, 1)

def _conv_cfg(M: int = 64) -> tuple[tuple[int, int, int, int], int]:
  """(tile, KS) for M output channels: (1, 8, 1, 2) for M <= 8 (an RGB conv_out: 3x less wasted work), else CONV_TILE.
  FZ_METAL_CONV_CFG=WM,WN,FM,FN,KS overrides (benchmarks)."""
  if (o := getenv("FZ_METAL_CONV_CFG", "")):
    v = [int(x) for x in str(o).split(",")]
    return (v[0], v[1], v[2], v[3]), v[4]
  return ((1, 8, 1, 2), CONV_KS) if M <= 8 else (CONV_TILE, CONV_KS)

def render_conv(name: str = "fz_conv", tile: tuple[int, int, int, int] | None = None, KS: int | None = None, M: int = 64) -> str:
  """out[o*so + p*sp] = sum_k w[oa + o*sam + k*sak] * col[k, p] (float32) for the scheduler's "conv" match
  (fusezap/schedule.py _match_conv): col[k, p] = X[c, ho*sh + i*dh - ph, wo*sw + j*dw - pw] (0 outside), k = c*T + i*kw
  + j, p = ho*Wo + wo, X a contiguous (C, H, W) buffer. The im2col matrix is never written. The reduction runs tap by
  tap (t = 0..T-1), and within a tap over the input channels in steps of KS: the threadgroup gathers the (KS x BN) slice
  of col for its pixels (one input position per thread and tap, so a row is one address add) and the (BM x KS) slice
  of the weight into threadgroup memory (zero padded at the edges), then each simdgroup runs 8x8 simdgroup MMAs on its
  FM x FN fragments. The summation order (tap-major) differs from the composed matmul's (channel-major): float32
  results agree to rounding, not bit for bit. Params CONV_PARAMS (fusezap/schedule.py)."""
  from fusezap.schedule import CONV_PARAMS
  ctile, cks = _conv_cfg(M)
  WM, WN, FM, FN = tile or ctile
  KS = KS or cks
  SM, SN, NT = FM * 8, FN * 8, WM * WN * 32
  BM, BN = WM * SM, WN * SN
  assert NT % BN == 0 and KS % 8 == 0 and (BM * KS) % NT == 0
  args = _msl_args(["device float* out [[buffer(0)]]", "device const float* A [[buffer(1)]]", "device const float* X [[buffer(2)]]"], CONV_PARAMS)
  IJ = [(i, j) for i in range(FM) for j in range(FN)]
  zero = "".join(f"  simdgroup_float8x8 c{i}_{j} = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n" for i, j in IJ)
  decl = "        simdgroup_float8x8 " + ", ".join([f"a{i}" for i in range(FM)] + [f"b{j}" for j in range(FN)]) + ";\n"
  lds = "".join(f"        simdgroup_load(a{i}, tA + (wm * {SM} + {8 * i}) * {KS} + kk, {KS});\n" for i in range(FM)) + \
        "".join(f"        simdgroup_load(b{j}, tB + kk * {BN} + wn * {SN} + {8 * j}, {BN});\n" for j in range(FN))
  mma = "".join(f"        simdgroup_multiply_accumulate(c{i}_{j}, a{i}, b{j}, c{i}_{j});\n" for i, j in IJ)
  st_dev = "".join(f"    simdgroup_store(c{i}_{j}, out + (long)(m0 + {8 * i})*so + n0 + {8 * j}, (ulong)so);\n" for i, j in IJ)
  st_tg = "".join(f"""    simdgroup_store(c{i}_{j}, sC, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = lane; e < 64; e += 32) {{ int m = m0 + {8 * i} + e / 8, n = n0 + {8 * j} + e % 8; if (m < M && n < N) out[(long)m*so + (long)n*sp] = sC[e]; }}
    simdgroup_barrier(mem_flags::mem_threadgroup);
""" for i, j in IJ)
  return f"""#include <metal_stdlib>
using namespace metal;
kernel void {name}({args}) {{
  threadgroup float tA[{BM * KS}], tB[{KS * BN}], tC[{WM * WN}][64];
  const int nbn = (N + {BN - 1}) / {BN}, bm = gid.x / nbn, bn = gid.x % nbn;
  const int wm = sg / {WN}, wn = sg % {WN};
  const int m0 = bm * {BM} + wm * {SM}, n0 = bn * {BN} + wn * {SN};
  const int T = kh * kw;
  const long HW = (long)H * W;
  // this thread's pixel column of the col tile (fixed for the whole reduction)
  const int bcol = tid % {BN}, brow = tid / {BN}, p = bn * {BN} + bcol;
  const int ho = p / Wo, wo = p % Wo, hb = ho * sh - ph, wb = wo * sw - pw;
  const bool pin = p < N;
  // this thread's weight-tile elements: rows ar + {NT // KS}*u, column ac (fixed)
  const int ac = tid % {KS}, ar = tid / {KS};
{zero}  // software pipeline: the next step's operand values are loaded into registers while the current step multiplies
  float ra[{BM * KS // NT}], rb[{KS // (NT // BN)}];
  const int nc = (C + {KS - 1}) / {KS}, nsteps = T * nc;
  #define FZ_LOAD(step) {{ const int t = (step) / nc, c0 = ((step) - t * nc) * {KS}; \
    const int i = t / kw, j = t - i * kw, ih = hb + i * dh, iw = wb + j * dw; \
    const bool xin = pin && ih >= 0 && ih < H && iw >= 0 && iw < W; \
    device const float* xp = X + (long)ih * W + iw; \
    const int c = c0 + ac; const bool cin = c < C; \
    device const float* ap = A + oa + (long)(c * T + t) * sak; \
    for (int u = 0; u < {BM * KS // NT}; u++) {{ const int m = bm * {BM} + ar + u * {NT // KS}; ra[u] = (cin && m < M) ? ap[(long)m * sam] : 0.0f; }} \
    for (int u = 0; u < {KS // (NT // BN)}; u++) {{ const int cc = c0 + brow + u * {NT // BN}; rb[u] = (xin && cc < C) ? xp[cc * HW] : 0.0f; }} }}
  FZ_LOAD(0)
  for (int step = 0; step < nsteps; step++) {{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int u = 0; u < {BM * KS // NT}; u++) tA[(ar + u * {NT // KS}) * {KS} + ac] = ra[u];
    for (int u = 0; u < {KS // (NT // BN)}; u++) tB[(brow + u * {NT // BN}) * {BN} + bcol] = rb[u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (step + 1 < nsteps) FZ_LOAD(step + 1)
    for (int kk = 0; kk < {KS}; kk += 8) {{
{decl}{lds}{mma}    }}
  }}
  #undef FZ_LOAD
  if (m0 + {SM} <= M && n0 + {SN} <= N && sp == 1) {{
{st_dev}  }} else {{
    threadgroup float* sC = tC[sg];
{st_tg}  }}
}}
"""

ATTN_ROWS_CHUNK = 64   # float accumulator registers per thread in render_attention_rows

def render_attention_rows(name: str, Dv: int, dt: DType = dtypes.float32, out_dt: DType = dtypes.float32, causal: bool = False) -> str:
  """Fallback attention for any head dims: one thread per (batch, query row), online softmax. O is accumulated in float
  registers ATTN_ROWS_CHUNK value columns at a time (the key loop reruns per chunk, recomputing the identical scores) and
  written in the output dtype once at the end, so f16/bf16 outputs keep float32 accumulation."""
  T, OT, C = _MSL_T[dt], _MSL_T[out_dt], min(max(Dv, 1), ATTN_ROWS_CHUNK)
  args = _msl_args([f"device {OT}* O [[buffer(0)]]", f"device const {T}* Q [[buffer(1)]]", f"device const {T}* Kb [[buffer(2)]]",
                    f"device const {T}* V [[buffer(3)]]"], ATTN_PARAMS)
  return f"""#include <metal_stdlib>
using namespace metal;
kernel void {name}({args}) {{
  const int g = gid.x * 64 + tid;
  if (g >= Bt * L) return;
  const int bz = g / L, l = g % L;
  const float scale = as_type<float>(scale_bits), log2e = as_type<float>(log2e_bits);
  device const {T}* q = Q + oq + (long)bz*sqb + (long)l*sql;
  device const {T}* kp = Kb + ok + (long)bz*skb;
  device const {T}* v = V + ov + (long)bz*svb;
  device {OT}* o = O + (long)bz*sob + (long)l*sol;
  const int kend = {"min(S, l + 1)" if causal else "S"};
  for (int d0 = 0; d0 < Dv; d0 += {C}) {{
    const int dn = min({C}, Dv - d0);
    float acc[{C}], m = -INFINITY, den = 0.0f;
    for (int j = 0; j < {C}; j++) acc[j] = 0.0f;
    for (int s = 0; s < kend; s++) {{
      float t = 0.0f;
      for (int d = 0; d < D; d++) t = fma((float)q[(long)d*sqd], (float)kp[(long)d*skd + (long)s*sks], t);
      t *= scale;
      const float mn = (t == t) ? (t > m ? t : m) : t;
      const float alpha = (mn == -INFINITY) ? 1.0f : exp2((m - mn) * log2e), p = (mn == -INFINITY) ? 0.0f : exp2((t - mn) * log2e);
      den = den * alpha + p; m = mn;
      device const {T}* vs = v + (long)s*svs + (long)d0*svd;
      for (int j = 0; j < {C}; j++) if (j < dn) acc[j] = fma(p, (float)vs[(long)j*svd], acc[j] * alpha);
    }}
    const float inv = S > 0 ? 1.0f / den : 0.0f;   // no keys: P @ V sums nothing, like the composed path
    for (int j = 0; j < {C}; j++) if (j < dn) o[(long)(d0 + j)*sod] = ({OT})(acc[j] * inv);
  }}
}}
"""
