"""CUDA flash attention (SPEC §11.2, P3-ATTN): tensor-core prefill, a general SIMT kernel and paged decode.

Kernels (CUDA C++ for NVRTC, header-free like fusezap/codegen/cuda.py; 16-bit floats are their bit patterns in memory):
  render_flash(name, dt, D, causal, NW, BN)   prefill on tensor cores (sm_80+): float16/bfloat16, head dim 64 or 128, GQA,
      causal or full. A CTA owns BM = 16*NW query rows of one (batch, q head); each warp owns 16 rows and keeps its Q
      fragments, the running max/sum and its O tile (f32) in registers. Key blocks of BN rows of K and V stream through
      shared memory (cp.async, double buffered, 16-byte chunks XOR-swizzled by row so ldmatrix is conflict free).
      S = Q K^T and O += P V use mma.sync m16n8k16 with f32 accumulators; the online softmax (exp2 with the scale folded
      into log2e, like FlashAttention-2) runs on the accumulator fragments, and P is rounded to the input type for P V.
      Causal: key s is visible to query row l when s <= l + coff (coff = 0 is torch's top-left is_causal; coff = S - L
      aligns the last query with the last key, for a prefill chunk after cached context). Key blocks past the CTA's last
      visible key are skipped and the heaviest CTAs launch first.
  render_simt(name, dt, D, Dv, W, causal, paged)   the general kernel: any float dtype (f32 math), any head dims and
      strides. One warp per query row (dense) or per q head of a GQA group (paged); W warps share each 32-key chunk of
      K/V staged as float in shared memory; each lane scores one key of the chunk, the warp updates the running max/sum,
      and lane i accumulates output columns i, i+32, ... It is the scheduler-hook kernel (float32 sdpa) and the fallback.
  render_mdec(...)   the tensor-core GQA paged decode (default for f16/bf16 decode rows): a warp per (token, kv head,
      128-key partition), the MMA rows are the kv head's q heads; the split decode's combine merges partitions.
  render_decode(...)   the one-pass paged decode (f32, and FZ_PAGED_MMA=0): a CTA per (token, kv head, 64-key partition),
      all G q heads of the kv head from one K/V read, scores and P V in one pass; a combine kernel merges partitions.
  paged decode (dense=False above): reads K/V straight from fusezap/serve/kv_cache.py's pool layout (2, NB, Hkv, bs, hd)
      through the step's block tables: the same buffers, params and metadata as kv_cache.msl_paged_attn (`ATTN_PARAMS`).

Python entry points (everything launches through `fusezap.jit.run_item`, so a fz.jit capture records and replays them):
  attention(q, k, v, causal=False, scale=None, coff=0)  Tensors (..., Hq, L, D), (..., Hkv, S, D), (..., Hkv, S, Dv), Hq a
      multiple of Hkv -> (..., Hq, L, Dv) in the input dtype. == scaled_dot_product_attention(is_causal, enable_gqa).
  paged_attention(x, pool, meta, N, Hkv, G, hd, bs, NB, MB, xs, mrow, mpos, mbt, scale)  drop-in for
      kv_cache.paged_attention on CUDA buffers.
  install(dev)  offers "attention" to the P2-CUSTOM scheduler hook on a CUDADevice (the float32 sdpa graph becomes one
      kind="custom" item) and routes those items to the SIMT kernel; other custom items fall through to dev.run.
"""
from __future__ import annotations
import ctypes, math, struct
from dataclasses import dataclass
from string import Template
from typing import Any, Sequence
from fusezap.dtype import DType, dtypes
from fusezap.helpers import getenv, prod
from fusezap.codegen.cuda import CUDA_PRELUDE, HALF_HELPERS, BF16_HELPERS

_HALVES = (dtypes.float16, dtypes.bfloat16)
_FLOATS = (dtypes.float32,) + _HALVES
LOG2E = math.log2(math.e)

def _bits(x: float) -> int: return struct.unpack("<i", struct.pack("<f", x))[0]
def _tag(dt: DType) -> str: return {dtypes.float32: "f32", dtypes.float16: "f16", dtypes.bfloat16: "bf16"}[dt]

# ---------------------------------------------------------------- tensor-core prefill

FLASH_PARAMS = ("B", "Hq", "G", "L", "S", "sqb", "sqh", "sql", "skb", "skh", "sks", "svb", "svh", "svs", "sob", "soh", "sol",
                "coff", "scale_bits")

_PTX = Template(r"""typedef unsigned short u16;
typedef unsigned int u32;
static __device__ __forceinline__ u32 fz_smem(const void* p) { u32 r; asm("{ .reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a; }" : "=r"(r) : "l"(p)); return r; }
static __device__ __forceinline__ void fz_cp16(u32 dst, const void* src, int n) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(n) : "memory"); }
static __device__ __forceinline__ void fz_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
static __device__ __forceinline__ void fz_wait0() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
static __device__ __forceinline__ void fz_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
static __device__ __forceinline__ void fz_ldm4(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory"); }
static __device__ __forceinline__ void fz_ldm4t(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory"); }
static __device__ __forceinline__ void fz_mma(float* c, const u32* a, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.${TY}.${TY}.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1)); }
static __device__ __forceinline__ u32 fz_pack(float lo, float hi) { u32 r; asm("cvt.rn.${TY}x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi)); return r; }
static __device__ __forceinline__ float fz_ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
""")

_FLASH_STEP = r"""    const int st = jb & 1, n0 = jb * ${BN};
    float s[${BN} / 8][4];
#pragma unroll
    for (int i = 0; i < ${BN} / 8; i++) { s[i][0] = 0.f; s[i][1] = 0.f; s[i][2] = 0.f; s[i][3] = 0.f; }
#pragma unroll
    for (int ks = 0; ks < ${D} / 16; ks += 2) {
#pragma unroll
      for (int nt = 0; nt < ${BN} / 8; nt++) {
        const int r = nt * 8 + (lane & 7), c = ks * 2 + (lane >> 3);
        u32 b0, b1, b2, b3;
        fz_ldm4(sK + (u32)((st * ${BN} + r) * ${D} + ((c ^ (r & 7)) << 3)) * 2u, b0, b1, b2, b3);
        fz_mma(s[nt], qa[ks], b0, b1);
        fz_mma(s[nt], qa[ks + 1], b2, b3);
      }
    }
${MASK}    float x0 = mx0, x1 = mx1;
#pragma unroll
    for (int nt = 0; nt < ${BN} / 8; nt++) { x0 = fmaxf(x0, fmaxf(s[nt][0], s[nt][1])); x1 = fmaxf(x1, fmaxf(s[nt][2], s[nt][3])); }
    x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1)); x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
    x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1)); x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
    const float c0 = x0 == -INFINITY ? 0.f : x0 * sl, c1 = x1 == -INFINITY ? 0.f : x1 * sl;   // a row with no visible key yet
    const float a0 = fz_ex2(mx0 * sl - c0), a1 = fz_ex2(mx1 * sl - c1);
    mx0 = x0; mx1 = x1;
    float p0 = 0.f, p1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < ${BN} / 8; nt++) {
      s[nt][0] = fz_ex2(fmaf(s[nt][0], sl, -c0)); s[nt][1] = fz_ex2(fmaf(s[nt][1], sl, -c0));
      s[nt][2] = fz_ex2(fmaf(s[nt][2], sl, -c1)); s[nt][3] = fz_ex2(fmaf(s[nt][3], sl, -c1));
      p0 += s[nt][0] + s[nt][1]; p1 += s[nt][2] + s[nt][3];
    }
    l0 = l0 * a0 + p0; l1 = l1 * a1 + p1;
#pragma unroll
    for (int i = 0; i < ${D} / 8; i++) { o[i][0] *= a0; o[i][1] *= a0; o[i][2] *= a1; o[i][3] *= a1; }
${PIPE_V}
#pragma unroll
    for (int kk = 0; kk < ${BN} / 16; kk++) {
      const u32 pa[4] = {fz_pack(s[2 * kk][0], s[2 * kk][1]), fz_pack(s[2 * kk][2], s[2 * kk][3]),
                         fz_pack(s[2 * kk + 1][0], s[2 * kk + 1][1]), fz_pack(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
      for (int dt = 0; dt < ${D} / 8; dt += 2) {
        const int r = kk * 16 + (lane & 7) + (lane & 8), c = dt + (lane >> 4);
        u32 b0, b1, b2, b3;
        fz_ldm4t(sV + (u32)((st * ${BN} + r) * ${D} + ((c ^ (r & 7)) << 3)) * 2u, b0, b1, b2, b3);
        fz_mma(o[dt], pa, b0, b1);
        fz_mma(o[dt + 1], pa, b2, b3);
      }
    }
  }
"""   # key block jb of the online softmax (S, ${MASK}, max/sum, P V) and the end of the jb loop

_FLASH_LSUM = r"""  l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
"""

_DENSE_MASK = r"""    if (n0 + ${BN} > S || (${CAUSAL} && n0 + ${BN} - 1 > m0 + coff)) {
#pragma unroll
      for (int nt = 0; nt < ${BN} / 8; nt++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int key = n0 + nt * 8 + 2 * t4 + (e & 1), row = e < 2 ? r0 : r1;
          if (key >= S || (${CAUSAL} && key > row + coff)) s[nt][e] = -INFINITY;
        }
    }
"""

_FLASH = Template(r"""extern "C" __global__ void __launch_bounds__(${NT}) ${NAME}(u16* __restrict__ O, const u16* __restrict__ Q, const u16* __restrict__ K,
    const u16* __restrict__ V, const int B, const int Hq, const int G, const int L, const int S, const int sqb, const int sqh, const int sql,
    const int skb, const int skh, const int sks, const int svb, const int svh, const int svs, const int sob, const int soh, const int sol,
    const int coff, const int scale_bits) {
  extern __shared__ __align__(16) unsigned char fz_sm[];
  const u32 sK = fz_smem(fz_sm), sV = sK + 2u * ${BN} * ${D} * 2u;     // [2 stages][BN rows][D], 16-byte chunks swizzled by row
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int nmb = (L + ${BM} - 1) / ${BM};
  int bid = blockIdx.x;
  const int mb = ${CAUSAL} ? nmb - 1 - bid % nmb : bid % nmb;           // causal: the heaviest query blocks first
  bid /= nmb;
  const int h = bid % Hq, b = bid / Hq, kh = h / G, m0 = mb * ${BM};
  const u16* q = Q + (long long)b * sqb + (long long)h * sqh;
  const u16* k = K + (long long)b * skb + (long long)kh * skh;
  const u16* v = V + (long long)b * svb + (long long)kh * svh;
  const float sl = __int_as_float(scale_bits) * 1.4426950408889634f;
  const int r0 = m0 + w * 16 + g, r1 = r0 + 8;
  u32 qa[${D} / 16][4];
#pragma unroll
  for (int ks = 0; ks < ${D} / 16; ks++) {
    const int c = ks * 16 + 2 * t4;
    qa[ks][0] = r0 < L ? *(const u32*)(q + (long long)r0 * sql + c) : 0u;
    qa[ks][1] = r1 < L ? *(const u32*)(q + (long long)r1 * sql + c) : 0u;
    qa[ks][2] = r0 < L ? *(const u32*)(q + (long long)r0 * sql + c + 8) : 0u;
    qa[ks][3] = r1 < L ? *(const u32*)(q + (long long)r1 * sql + c + 8) : 0u;
  }
  float o[${D} / 8][4];
#pragma unroll
  for (int i = 0; i < ${D} / 8; i++) { o[i][0] = 0.f; o[i][1] = 0.f; o[i][2] = 0.f; o[i][3] = 0.f; }
  float mx0 = -INFINITY, mx1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int kend = ${CAUSAL} ? min(S, m0 + ${BM} + coff) : S;            // keys [0, kend) are visible to some row of the CTA
  const int nb = kend > 0 ? (kend + ${BN} - 1) / ${BN} : 0;
${PIPE_HEAD}
""" + _FLASH_STEP + _FLASH_LSUM + r"""  u16* out = O + (long long)b * sob + (long long)h * soh;
#pragma unroll
  for (int dt = 0; dt < ${D} / 8; dt++) {
    const int c = dt * 8 + 2 * t4;
    if (r0 < L) *(u32*)(out + (long long)r0 * sol + c) = fz_pack(o[dt][0] * i0, o[dt][1] * i0);
    if (r1 < L) *(u32*)(out + (long long)r1 * sol + c) = fz_pack(o[dt][2] * i1, o[dt][3] * i1);
  }
}
""")

def flash_config(D: int) -> tuple[int, int]:
  """(NW warps, BN keys per block) for the tensor-core prefill; FZ_FLASH_NW / FZ_FLASH_BN override (tuning)."""
  nw, bn = (4, 64) if D == 128 else (4, 64)
  return getenv("FZ_FLASH_NW", nw), getenv("FZ_FLASH_BN", bn)

def flash_smem(D: int, BN: int) -> int: return 2 * 2 * BN * D * 2   # 2 stages x (K, V) x BN x D x 2 bytes

def _dense_src(what: str) -> str:
  g, st = ("k", "sks") if what == "K" else ("v", "svs")
  return f"{g} + (long long)(ok ? key : 0) * {st} + c * 8"

def _paged_src(what: str) -> str:   # key `key` of the CTA's sequence through its block table bt; V is the pool's second half
  return f"(ok ? P + {'vo + ' if what == 'V' else ''}((long long)(bt[key / bs] * Hkv + kh) * bs + key % bs) * ${{D}} : P) + c * 8"

def _pipe(split: bool, lim: str = "S", src: Any = _dense_src) -> tuple[str, str]:
  """The K/V double-buffer pipeline. split=False: K and V of block jb+1 are one cp.async group, waited for together at
  the top of the iteration. split=True: separate groups, so the S = Q K^T of block jb overlaps the V load (two barriers).
  Keys >= lim load as zeros; src(K|V) is the global address of 16-byte chunk c of key `key` (valid when ok)."""
  def load(what: str, jb: str) -> str:
    sm = "sK" if what == "K" else "sV"
    return f"""#pragma unroll
      for (int it = 0; it < ${{CH}}; it++) {{
        const int i = tid + it * ${{NT}}, r = i / (${{D}} / 8), c = i % (${{D}} / 8), key = ({jb}) * ${{BN}} + r, ok = key < {lim};
        const u32 off = (u32)((((({jb}) & 1) * ${{BN}}) + r) * ${{D}} + ((c ^ (r & 7)) << 3)) * 2u;
        fz_cp16({sm} + off, {src(what)}, ok ? 16 : 0);
      }}"""
  if not split:
    head = f"""  if (nb > 0) {{
{load("K", "0")}
{load("V", "0")}
  }}
  fz_commit();
  for (int jb = 0; jb < nb; jb++) {{
    fz_wait0();
    __syncthreads();                                                    // block jb landed; block jb-1's stage is free
    if (jb + 1 < nb) {{
{load("K", "jb + 1")}
{load("V", "jb + 1")}
    }}
    fz_commit();"""
    return head, ""
  head = f"""  if (nb > 0) {{
{load("K", "0")}
  }}
  fz_commit();
  if (nb > 0) {{
{load("V", "0")}
  }}
  fz_commit();
  for (int jb = 0; jb < nb; jb++) {{
    fz_wait1();                                                         // K of block jb landed (V of jb may be in flight)
    __syncthreads();                                                    // block jb-1's K stage is free
    if (jb + 1 < nb) {{
{load("K", "jb + 1")}
    }}
    fz_commit();"""
  v = f"""    fz_wait1();                                                         // V of block jb landed (K of jb+1 may be in flight)
    __syncthreads();                                                    // block jb-1's V stage is free
    if (jb + 1 < nb) {{
{load("V", "jb + 1")}
    }}
    fz_commit();"""
  return head, v

def render_flash(name: str, dt: DType, D: int, causal: bool, NW: int = 4, BN: int = 64, split: bool | None = None) -> str:
  assert dt in _HALVES and D in (64, 128) and BN % 16 == 0 and (BN * D // 8) % (32 * NW) == 0, (dt, D, NW, BN)
  if split is None: split = bool(getenv("FZ_FLASH_SPLITKV", 0))
  sub = dict(NAME=name, TY="bf16" if dt == dtypes.bfloat16 else "f16", D=D, BN=BN, BM=16 * NW, NT=32 * NW,
             CH=BN * D // 8 // (32 * NW), CAUSAL=int(causal))
  head, v = _pipe(split)
  src = _FLASH.safe_substitute(PIPE_HEAD=head, PIPE_V=v, MASK=_DENSE_MASK)
  return CUDA_PRELUDE + "\n" + _PTX.substitute(sub) + Template(src).substitute(sub)

# ---------------------------------------------------------------- tensor-core paged prefill

_PAGED_MASK = r"""    if (n0 + ${BN} - 1 > min(lim0, lim1)) {
#pragma unroll
      for (int nt = 0; nt < ${BN} / 8; nt++)
#pragma unroll
        for (int e = 0; e < 4; e++)
          if (n0 + nt * 8 + 2 * t4 + (e & 1) > (e < 2 ? lim0 : lim1)) s[nt][e] = -INFINITY;
    }
"""

_PFLASH = Template(r"""extern "C" __global__ void __launch_bounds__(${NT}) ${NAME}(u16* __restrict__ O, const u16* __restrict__ X, const u16* __restrict__ P,
    const int* __restrict__ M, const int N, const int Hkv, const int bs, const int NB, const int MB, const int xs, const int mrow,
    const int mpos, const int mbt, const int scale_bits) {
  extern __shared__ __align__(16) unsigned char fz_sm[];
  __shared__ int srow[${BM}], spos[${BM}];
  const u32 sK = fz_smem(fz_sm), sV = sK + 2u * ${BN} * ${D} * 2u;     // [2 stages][BN rows][D], 16-byte chunks swizzled by row
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3, Hq = Hkv * ${G};
  const int h = blockIdx.x % Hq, kh = h / ${G}, m0 = ((N + ${BM} - 1) / ${BM} - 1 - blockIdx.x / Hq) * ${BM};   // later rows first
  for (int i = tid; i < ${BM}; i += ${NT}) { const int t = m0 + i; srow[i] = t < N ? M[mrow + t] : -1; spos[i] = t < N ? M[mpos + t] : -1; }
  __syncthreads();
  const float sl = __int_as_float(scale_bits) * 1.4426950408889634f;
  const int u0 = w * 16 + g, u1 = u0 + 8, r0 = m0 + u0, r1 = m0 + u1, q0 = srow[u0], q1 = srow[u1];
  const u16* x0p = X + (long long)r0 * xs + h * ${D};
  const u16* x1p = X + (long long)r1 * xs + h * ${D};
  u32 qa[${D} / 16][4];
#pragma unroll
  for (int ks = 0; ks < ${D} / 16; ks++) {
    const int c = ks * 16 + 2 * t4;
    qa[ks][0] = q0 >= 0 ? *(const u32*)(x0p + c) : 0u;
    qa[ks][1] = q1 >= 0 ? *(const u32*)(x1p + c) : 0u;
    qa[ks][2] = q0 >= 0 ? *(const u32*)(x0p + c + 8) : 0u;
    qa[ks][3] = q1 >= 0 ? *(const u32*)(x1p + c + 8) : 0u;
  }
  float o[${D} / 8][4];
#pragma unroll
  for (int i = 0; i < ${D} / 8; i++) { o[i][0] = 0.f; o[i][1] = 0.f; o[i][2] = 0.f; o[i][3] = 0.f; }
  float mx0 = -INFINITY, mx1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const long long vo = (long long)NB * Hkv * bs * ${D};
  // one pass per sequence among the CTA's rows (ascending block-table row); rows of other sequences see no keys in it,
  // which leaves their running max/sum/output bit-identical, so a row's result does not depend on its neighbours
  for (int cur = -1;;) {
    int seq = 0x7fffffff, kend = 0;
    for (int i = 0; i < ${BM}; i++) if (srow[i] > cur && srow[i] < seq) seq = srow[i];
    if (seq == 0x7fffffff) break;
    for (int i = 0; i < ${BM}; i++) if (srow[i] == seq) kend = max(kend, spos[i] + 1);
    cur = seq;
    const int* bt = M + mbt + seq * MB;
    const int lim0 = q0 == seq ? spos[u0] : -1, lim1 = q1 == seq ? spos[u1] : -1;   // key j visible to the row iff j <= lim
    const int nb = (kend + ${BN} - 1) / ${BN};
    __syncthreads();                                                    // the previous pass's last stage is free
${PIPE_HEAD}
""" + _FLASH_STEP + r"""  }
""" + _FLASH_LSUM + r"""  const int ld = Hq * ${D};
#pragma unroll
  for (int dt = 0; dt < ${D} / 8; dt++) {
    const int c = h * ${D} + dt * 8 + 2 * t4;
    if (r0 < N) *(u32*)(O + (long long)r0 * ld + c) = fz_pack(o[dt][0] * i0, o[dt][1] * i0);
    if (r1 < N) *(u32*)(O + (long long)r1 * ld + c) = fz_pack(o[dt][2] * i1, o[dt][3] * i1);
  }
}
""")

def render_paged_flash(name: str, dt: DType, hd: int, G: int, NW: int = 4, BN: int = 64) -> str:
  """Paged prefill on tensor cores: render_flash's key-block loop over kv_cache's pool (2, NB, Hkv, bs, hd) through the
  step's block tables. A CTA owns BM = 16*NW consecutive token rows and one q head; each key block is loaded once and
  shared by those rows (a prompt chunk's tokens are consecutive rows of one sequence). Token t sees keys 0..pos_t of
  its own sequence; padding rows (block-table row < 0) give zeros."""
  assert dt in _HALVES and hd in (64, 128) and BN % 16 == 0 and BN % 8 == 0 and (BN * hd // 8) % (32 * NW) == 0, (dt, hd, NW, BN)
  sub = dict(NAME=name, TY="bf16" if dt == dtypes.bfloat16 else "f16", D=hd, BN=BN, BM=16 * NW, NT=32 * NW, G=G,
             CH=BN * hd // 8 // (32 * NW))
  head, v = _pipe(False, "kend", _paged_src)
  src = _PFLASH.safe_substitute(PIPE_HEAD=head, PIPE_V=v, MASK=_PAGED_MASK)
  return CUDA_PRELUDE + "\n" + _PTX.substitute(sub) + Template(src).substitute(sub)

# ---------------------------------------------------------------- SIMT kernel (dense prefill / hook, and paged decode)

DENSE_PARAMS = ("Bt", "Hq", "G", "L", "S", "sqb", "sqh", "sql", "sqd", "skb", "skh", "sks", "skd", "svb", "svh", "svs", "svd",
                "sob", "soh", "sol", "sod", "oq", "ok", "ov", "coff", "scale_bits", "log2e_bits")
PAGED_PARAMS = ("N", "Hkv", "bs", "NB", "MB", "xs", "mrow", "mpos", "mbt", "scale_bits")   # == kv_cache.ATTN_PARAMS

_SIMT = Template(r"""static __device__ __forceinline__ float fz_wmax(float x) {
  for (int o = 16; o > 0; o >>= 1) { const float y = __shfl_xor_sync(0xffffffffu, x, o); x = y > x ? y : x; }
  return x;
}
static __device__ __forceinline__ float fz_wsum(float x) { for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o); return x; }
extern "C" __global__ void __launch_bounds__(${NT}) ${NAME}(${ARGS}) {
  extern __shared__ float fz_sm[];
  float* qs = fz_sm;                      // [W][D] this block's query rows (float)
  float* ks = qs + ${W} * ${D};           // [32][D+1] the key chunk (padded rows: lane j reads row j)
  float* vs = ks + 32 * (${D} + 1);       // [32][Dv] the value chunk
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
${SETUP}
  for (int d = lane; d < ${D}; d += 32) qs[w * ${D} + d] = live ? ${LDQ} : 0.f;
  float m = -INFINITY, den = 0.f, acc[${DC}];
#pragma unroll
  for (int c = 0; c < ${DC}; c++) acc[c] = 0.f;
  for (int j0 = 0; j0 < kend; j0 += 32) {
    __syncthreads();                      // the previous chunk is consumed (and qs is written, first time)
    for (int i = tid; i < 32 * ${D}; i += ${NT}) {
      const int r = i / ${D}, d = i - r * ${D}, j = j0 + r;
      ks[r * (${D} + 1) + d] = j < kend ? ${LDK} : 0.f;
    }
    for (int i = tid; i < 32 * ${DV}; i += ${NT}) {
      const int r = i / ${DV}, d = i - r * ${DV}, j = j0 + r;
      vs[r * ${DV} + d] = j < kend ? ${LDV} : 0.f;
    }
    __syncthreads();
    const int j = j0 + lane;
    float s = -INFINITY;
    if (j < lim) {
      const float* qq = qs + w * ${D};
      const float* kr = ks + lane * (${D} + 1);
      float dot = 0.0f;
      for (int d = 0; d < ${D}; d++) dot = dot + qq[d] * kr[d];
      s = dot * scale;
    }
    const float cm = fz_wmax(s), mn = m > cm ? m : cm;
    if (mn == -INFINITY) continue;        // nothing visible to this row yet (warp-uniform)
    const float p = j < lim ? exp2f((s - mn) * log2e) : 0.0f, alpha = exp2f((m - mn) * log2e);
    den = den * alpha + fz_wsum(p);
#pragma unroll
    for (int c = 0; c < ${DC}; c++) acc[c] *= alpha;
    const int nk = min(32, lim - j0);
    for (int kk = 0; kk < nk; kk++) {
      const float pk = __shfl_sync(0xffffffffu, p, kk);
      const float* vr = vs + kk * ${DV};
#pragma unroll
      for (int c = 0; c < ${DC}; c++) { const int d = lane + 32 * c; if (d < ${DV}) acc[c] += pk * vr[d]; }
    }
    m = mn;
  }
#pragma unroll
  for (int c = 0; c < ${DC}; c++) { const int d = lane + 32 * c; if (d < ${DV} && ${STORE_OK}) O[ooff + (long long)d * ${SOD}] = ${ST}; }
}
""")

_DENSE_SETUP = Template(r"""  const int nrb = (L + ${W} - 1) / ${W};
  int bid = blockIdx.x;
  const int rb = bid % nrb; bid /= nrb;
  const int h = bid % Hq, b = bid / Hq, kh = h / G, l = rb * ${W} + w;
  const bool live = l < L;
  const long long qoff = oq + (long long)b * sqb + (long long)h * sqh + (long long)l * sql;
  const long long kb = ok + (long long)b * skb + (long long)kh * skh, vb = ov + (long long)b * svb + (long long)kh * svh;
  const long long ooff = (long long)b * sob + (long long)h * soh + (long long)l * sol;
  const float scale = __int_as_float(scale_bits), log2e = __int_as_float(log2e_bits);
  const int lim = !live ? 0 : ${CAUSAL} ? min(S, l + coff + 1) : S;
  const int kend = ${CAUSAL} ? min(S, min(L - 1, rb * ${W} + ${W} - 1) + coff + 1) : S;""")

_PAGED_SETUP = Template(r"""  const int t = blockIdx.x / Hkv, kh = blockIdx.x % Hkv, h = kh * ${W} + w;
  const int row = M[mrow + t];
  const bool live = row >= 0;
  const int lim = live ? M[mpos + t] + 1 : 0, kend = lim;
  const int* bt = M + mbt + (live ? row : 0) * MB;
  const long long qoff = (long long)t * xs + (long long)h * ${D}, vo = (long long)NB * Hkv * bs * ${D};
  const long long ooff = ((long long)t * Hkv * ${W} + h) * ${D};
  const float scale = __int_as_float(scale_bits), log2e = 1.4426950408889634f;
  if (!live) { for (int d = lane; d < ${D}; d += 32) O[ooff + d] = ${ZERO}; return; }   // padding row (block-uniform)""")

def simt_smem(D: int, Dv: int, W: int) -> int: return (W * D + 32 * (D + 1) + 32 * Dv) * 4

def render_simt(name: str, dt: DType, D: int, Dv: int, W: int = 4, causal: bool = False, paged: bool = False) -> str:
  assert dt in _FLOATS and D > 0 and Dv > 0 and 1 <= W <= 32
  T = "float" if dt == dtypes.float32 else "unsigned short"
  ld = {dtypes.float32: "", dtypes.float16: "fz_h2f", dtypes.bfloat16: "fz_bf2f"}[dt]
  st = {dtypes.float32: "", dtypes.float16: "fz_f2h", dtypes.bfloat16: "fz_f2bf"}[dt]
  zero = "0.0f" if dt == dtypes.float32 else "(unsigned short)0"
  if paged:
    assert D == Dv
    bufs = [f"{T}* __restrict__ O", f"const {T}* __restrict__ Q", f"const {T}* __restrict__ P", "const int* __restrict__ M"]
    setup = _PAGED_SETUP.substitute(W=W, D=D, ZERO=zero)
    koff = f"((long long)(bt[j / bs] * Hkv + kh) * bs + j % bs) * {D} + d"
    LDQ, LDK, LDV, SOD, ok = f"{ld}(Q[qoff + d])", f"{ld}(P[{koff}])", f"{ld}(P[vo + {koff}])", "1", "true"
    params = PAGED_PARAMS
  else:
    bufs = [f"{T}* __restrict__ O", f"const {T}* __restrict__ Q", f"const {T}* __restrict__ K", f"const {T}* __restrict__ V"]
    setup = _DENSE_SETUP.substitute(W=W, CAUSAL=int(causal))
    LDQ, LDK, LDV = f"{ld}(Q[qoff + (long long)d * sqd])", f"{ld}(K[kb + (long long)j * sks + (long long)d * skd])", \
                    f"{ld}(V[vb + (long long)j * svs + (long long)d * svd])"
    SOD, ok, params = "sod", "live", DENSE_PARAMS
  args = ", ".join(bufs + [f"const int {p}" for p in params])
  helpers = {dtypes.float32: "", dtypes.float16: HALF_HELPERS, dtypes.bfloat16: BF16_HELPERS}[dt]
  return "\n".join([CUDA_PRELUDE, helpers, _SIMT.substitute(NAME=name, ARGS=args, SETUP=setup, NT=32 * W, W=W, D=D, DV=Dv,
                    DC=-(-Dv // 32), LDQ=LDQ, LDK=LDK, LDV=LDV, SOD=SOD, STORE_OK=ok, ST=f"{st}(acc[c] / den)")])

# ---------------------------------------------------------------- split-K paged decode

PART = 256                              # keys per chunk; CTA p of a token takes chunks p, p + NP, p + 2NP, ...
SPLIT_ROWS = 1 << 14                    # cap on N * Hq * NP scratch rows (f32 acc = rows * hd * 4 bytes, ml = rows * 8)
SPLIT_PARAMS = PAGED_PARAMS + ("NP",)
COMBINE_PARAMS = ("N", "Hq", "mrow", "mpos", "NP")

_LD8 = {
  dtypes.float32: """static __device__ __forceinline__ void fz_ld8(const float* p, float* v) {
  const fz_u4 a = *(const fz_u4*)p, b = *(const fz_u4*)(p + 4);
  v[0] = __uint_as_float(a.x); v[1] = __uint_as_float(a.y); v[2] = __uint_as_float(a.z); v[3] = __uint_as_float(a.w);
  v[4] = __uint_as_float(b.x); v[5] = __uint_as_float(b.y); v[6] = __uint_as_float(b.z); v[7] = __uint_as_float(b.w);
}""",
  dtypes.float16: """static __device__ __forceinline__ void fz_ld8(const unsigned short* p, float* v) {
  const fz_u4 a = *(const fz_u4*)p;
  const unsigned u[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
  for (int i = 0; i < 4; i++) { v[2 * i] = fz_h2f((unsigned short)(u[i] & 0xffffu)); v[2 * i + 1] = fz_h2f((unsigned short)(u[i] >> 16)); }
}""",
  dtypes.bfloat16: """static __device__ __forceinline__ void fz_ld8(const unsigned short* p, float* v) {
  const fz_u4 a = *(const fz_u4*)p;
  const unsigned u[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
  for (int i = 0; i < 4; i++) { v[2 * i] = __uint_as_float(u[i] << 16); v[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u); }
}"""}

_SPLIT = Template(r"""struct __align__(16) fz_u4 { unsigned x, y, z, w; };
${LD8}
extern "C" __global__ void __launch_bounds__(128) ${NAME}(float* __restrict__ Acc, float* __restrict__ ML, const ${T}* __restrict__ X,
    const ${T}* __restrict__ P, const int* __restrict__ M, const int N, const int Hkv, const int bs, const int NB, const int MB, const int xs,
    const int mrow, const int mpos, const int mbt, const int scale_bits, const int NP) {
  extern __shared__ float fz_sm[];
  float* ss = fz_sm;                         // [G][PART] scores, then probabilities
  float* red = ss + ${G} * ${PART};          // [RPI][G * HD] per-group partial outputs
  float* mr = red + ${RPI} * ${G} * ${HD};    // [G] running max, [G] running sum, [G] rescale of acc for this chunk
  float* lr = mr + ${G};
  float* al = lr + ${G};
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, sub = tid % ${LPR}, grp = tid / ${LPR};
  int bid = blockIdx.x;
  const int p = bid % NP; bid /= NP;
  const int kh = bid % Hkv, t = bid / Hkv;
  const int row = M[mrow + t];
  if (row < 0) return;                       // padding token: the combine kernel writes zeros
  const int pos = M[mpos + t];
  if (p * ${PART} > pos) return;             // partition past the sequence: never read
  const int* bt = M + mbt + row * MB;
  const float sl = __int_as_float(scale_bits) * 1.4426950408889634f;
  float q[${G}][8];
  {
    const ${T}* qp = X + (long long)t * xs + (long long)kh * ${G} * ${HD} + sub * 8;
#pragma unroll
    for (int g = 0; g < ${G}; g++) {
      fz_ld8(qp + g * ${HD}, q[g]);
#pragma unroll
      for (int e = 0; e < 8; e++) q[g][e] *= sl;
    }
  }
  const long long vo = (long long)NB * Hkv * bs * ${HD};
  if (tid < ${G}) { mr[tid] = -INFINITY; lr[tid] = 0.f; }
  // chunks p, p + NP, ...: online softmax across this CTA's chunks (one chunk when NP covers max_seq_len)
  for (${LOOP}) {
  const int j0 = c * ${PART}, jn = min(${PART}, pos + 1 - j0);
  // scores (log2 domain): LPR lanes per key, each a 16-byte slice of the row; U keys in flight per lane
  for (int k0 = 0; k0 < jn; k0 += ${RPI} * ${U}) {
    float x[${U}][8];
#pragma unroll
    for (int u = 0; u < ${U}; u++) {
      const int k = k0 + u * ${RPI} + grp;
      if (k < jn) { const int j = j0 + k; fz_ld8(P + ((long long)(bt[j / bs] * Hkv + kh) * bs + j % bs) * ${HD} + sub * 8, x[u]); }
      else {
#pragma unroll
        for (int e = 0; e < 8; e++) x[u][e] = 0.f;
      }
    }
#pragma unroll
    for (int u = 0; u < ${U}; u++) {
      const int k = k0 + u * ${RPI} + grp;
      float d[${G}];
#pragma unroll
      for (int g = 0; g < ${G}; g++) {
        float a = 0.f;
#pragma unroll
        for (int e = 0; e < 8; e++) a = fmaf(q[g][e], x[u][e], a);
        d[g] = a;
      }
#pragma unroll
      for (int o = ${LPR} / 2; o > 0; o >>= 1)
#pragma unroll
        for (int g = 0; g < ${G}; g++) d[g] += __shfl_xor_sync(0xffffffffu, d[g], o);
      if (sub == 0 && k < jn) {
#pragma unroll
        for (int g = 0; g < ${G}; g++) ss[g * ${PART} + k] = d[g];
      }
    }
  }
  __syncthreads();
  // softmax over the chunk, merged into the running max/sum: one warp per head
  for (int g = w; g < ${G}; g += 4) {
    float m = mr[g];
    for (int k = lane; k < jn; k += 32) m = fmaxf(m, ss[g * ${PART} + k]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float l = 0.f;
    for (int k = lane; k < jn; k += 32) { const float e = exp2f(ss[g * ${PART} + k] - m); ss[g * ${PART} + k] = e; l += e; }
    for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
    if (lane == 0) { const float a = exp2f(mr[g] - m); al[g] = a; lr[g] = lr[g] * a + l; mr[g] = m; }
  }
  __syncthreads();
  // P V: group grp takes keys grp, grp + RPI, ...; lane sub owns columns sub*8 .. sub*8+7
  float acc[${G}][8];
#pragma unroll
  for (int g = 0; g < ${G}; g++)
#pragma unroll
    for (int e = 0; e < 8; e++) acc[g][e] = 0.f;
  for (int k0 = 0; k0 < jn; k0 += ${RPI} * ${U}) {
    float x[${U}][8];
#pragma unroll
    for (int u = 0; u < ${U}; u++) {
      const int k = k0 + u * ${RPI} + grp;
      if (k < jn) { const int j = j0 + k; fz_ld8(P + vo + ((long long)(bt[j / bs] * Hkv + kh) * bs + j % bs) * ${HD} + sub * 8, x[u]); }
    }
#pragma unroll
    for (int u = 0; u < ${U}; u++) {
      const int k = k0 + u * ${RPI} + grp;
      if (k < jn) {
#pragma unroll
        for (int g = 0; g < ${G}; g++) {
          const float pk = ss[g * ${PART} + k];
#pragma unroll
          for (int e = 0; e < 8; e++) acc[g][e] = fmaf(pk, x[u][e], acc[g][e]);
        }
      }
    }
  }
  // this thread's running output lives in its own red slots (registers stay free for the score phase)
#pragma unroll
  for (int g = 0; g < ${G}; g++) {
    float* r = red + grp * ${G} * ${HD} + g * ${HD} + sub * 8;
    const float a = al[g];
#pragma unroll
    for (int e = 0; e < 8; e++) r[e] = c == p ? acc[g][e] : fmaf(r[e], a, acc[g][e]);
  }
  __syncthreads();                           // ss / al are rewritten by the next chunk
  }
  if (tid < ${G}) { float* ml = ML + ((((long long)t * Hkv + kh) * ${G} + tid) * NP + p) * 2; ml[0] = mr[tid]; ml[1] = lr[tid]; }
  for (int i = tid; i < ${G} * ${HD}; i += 128) {
    float a = 0.f;
#pragma unroll
    for (int r = 0; r < ${RPI}; r++) a += red[r * ${G} * ${HD} + i];
    const int g = i / ${HD}, d = i % ${HD};
    Acc[((((long long)t * Hkv + kh) * ${G} + g) * NP + p) * ${HD} + d] = a;
  }
}
""")

_COMBINE = Template(r"""extern "C" __global__ void __launch_bounds__(${HD}) ${NAME}(${T}* __restrict__ O, const float* __restrict__ Acc, const float* __restrict__ ML,
    const int* __restrict__ M, const int N, const int Hq, const int mrow, const int mpos, const int NP) {
  const int t = blockIdx.x / Hq, h = blockIdx.x % Hq, d = threadIdx.x;
  const long long th = (long long)t * Hq + h;
  if (M[mrow + t] < 0) { O[th * ${HD} + d] = ${ZERO}; return; }
  const int np = min(NP, (M[mpos + t] + ${PART}) / ${PART});
  const float* ml = ML + th * NP * 2;
  float mx = -INFINITY;
  for (int p = 0; p < np; p++) mx = fmaxf(mx, ml[2 * p]);
  float num = 0.f, den = 0.f;
  for (int p = 0; p < np; p++) {
    const float wt = exp2f(ml[2 * p] - mx);
    den += wt * ml[2 * p + 1];
    num += wt * Acc[(th * NP + p) * ${HD} + d];
  }
  O[th * ${HD} + d] = ${ST};
}
""")

# ---------------------------------------------------------------- one-pass paged decode (speed run 4)

DECODE_PART = 64                        # keys per partition (FZ_PAGED_DECODE_PART): fixed, so a token's sum order never
DECODE_ROWS = 1 << 16                   # depends on its batch; cap on np_n * Hq * NP scratch rows (FZ_PAGED_DECODE_ROWS)

_DECODE = Template(r"""struct __align__(16) fz_u4 { unsigned x, y, z, w; };
${LD8}
extern "C" __global__ void __launch_bounds__(128) ${NAME}(float* __restrict__ Acc, float* __restrict__ ML, const ${T}* __restrict__ X,
    const ${T}* __restrict__ P, const int* __restrict__ M, const int N, const int Hkv, const int bs, const int NB, const int MB, const int xs,
    const int mrow, const int mpos, const int mbt, const int scale_bits, const int NP) {
  __shared__ float sacc[${RPI}][${G}][${HD}];  // each key group's running output, merged at the end
  __shared__ float smx[${RPI}][${G}], ssm[${RPI}][${G}];
  const int tid = threadIdx.x, sub = tid % ${LPR}, grp = tid / ${LPR};
  int bid = blockIdx.x;
  const int p = bid % NP; bid /= NP;
  const int kh = bid % Hkv, t = bid / Hkv;
  const int row = M[mrow + t];
  if (row < 0) return;                       // padding token: the combine kernel writes zeros
  const int pos = M[mpos + t];
  if (p * ${PART} > pos) return;             // partition past the sequence: never read
  const int* bt = M + mbt + row * MB;
  const float sl = __int_as_float(scale_bits) * 1.4426950408889634f;
  float q[${G}][8], acc[${G}][8], m[${G}], l[${G}];
  {
    const ${T}* qp = X + (long long)t * xs + (long long)kh * ${G} * ${HD} + sub * 8;
#pragma unroll
    for (int g = 0; g < ${G}; g++) {
      fz_ld8(qp + g * ${HD}, q[g]);
#pragma unroll
      for (int e = 0; e < 8; e++) { q[g][e] *= sl; acc[g][e] = 0.f; }
      m[g] = -INFINITY; l[g] = 0.f;
    }
  }
  const long long vo = (long long)NB * Hkv * bs * ${HD};
  // one pass: LPR lanes per key (a 16-byte slice each), RPI key groups per CTA, U keys per group in flight; K and V of a
  // key load together and every group keeps an online softmax (log2 domain) for each q head of the kv head
  for (${LOOP}) {
    const int j0 = c * ${PART}, jn = min(${PART}, pos + 1 - j0);
    for (int k0 = 0; k0 < jn; k0 += ${RPI} * ${U}) {
      float kx[${U}][8], vx[${U}][8];
#pragma unroll
      for (int u = 0; u < ${U}; u++) {
        const int k = k0 + u * ${RPI} + grp;
        if (k < jn) {
          const int j = j0 + k;
          const long long off = ((long long)(bt[j / bs] * Hkv + kh) * bs + j % bs) * ${HD} + sub * 8;
          fz_ld8(P + off, kx[u]);
          fz_ld8(P + vo + off, vx[u]);
        } else {
#pragma unroll
          for (int e = 0; e < 8; e++) { kx[u][e] = 0.f; vx[u][e] = 0.f; }
        }
      }
#pragma unroll
      for (int u = 0; u < ${U}; u++) {
        const int k = k0 + u * ${RPI} + grp;
        float d[${G}];
#pragma unroll
        for (int g = 0; g < ${G}; g++) {
          float a = 0.f;
#pragma unroll
          for (int e = 0; e < 8; e++) a = fmaf(q[g][e], kx[u][e], a);
          d[g] = a;
        }
#pragma unroll
        for (int o = ${LPR} / 2; o > 0; o >>= 1)
#pragma unroll
          for (int g = 0; g < ${G}; g++) d[g] += __shfl_xor_sync(0xffffffffu, d[g], o);
        if (k < jn) {
#pragma unroll
          for (int g = 0; g < ${G}; g++) {
            const float mn = fmaxf(m[g], d[g]), a = exp2f(m[g] - mn), w = exp2f(d[g] - mn);
            l[g] = fmaf(l[g], a, w);
#pragma unroll
            for (int e = 0; e < 8; e++) acc[g][e] = fmaf(w, vx[u][e], acc[g][e] * a);
            m[g] = mn;
          }
        }
      }
    }
  }
#pragma unroll
  for (int g = 0; g < ${G}; g++) {
    if (sub == 0) { smx[grp][g] = m[g]; ssm[grp][g] = l[g]; }
#pragma unroll
    for (int e = 0; e < 8; e++) sacc[grp][g][sub * 8 + e] = acc[g][e];
  }
  __syncthreads();
  // merge the key groups in group order; a group that saw no key has max -inf, so weight 0
  for (int i = tid; i < ${G} * ${HD}; i += 128) {
    const int g = i / ${HD}, dd = i % ${HD};
    float mx = -INFINITY;
#pragma unroll
    for (int r = 0; r < ${RPI}; r++) mx = fmaxf(mx, smx[r][g]);
    float num = 0.f, den = 0.f;
#pragma unroll
    for (int r = 0; r < ${RPI}; r++) {
      const float w = smx[r][g] == -INFINITY ? 0.f : exp2f(smx[r][g] - mx);
      num = fmaf(w, sacc[r][g][dd], num);
      den = fmaf(w, ssm[r][g], den);
    }
    const long long th = ((long long)t * Hkv + kh) * ${G} + g;
    Acc[(th * NP + p) * ${HD} + dd] = num;
    if (dd == 0) { ML[(th * NP + p) * 2] = mx; ML[(th * NP + p) * 2 + 1] = den; }
  }
}
""")

def decode_ok(hd: int, G: int, xs: int) -> bool:
  """The one-pass decode applies where the split kernel does and its static shared memory (each key group's G x hd
  partial output: 4 KB per q head) fits 48 KB, so G <= 11 (FZ_PAGED_DECODE=0 keeps the split kernel for decode)."""
  return split_ok(hd, G, xs) and (128 // (hd // 8)) * G * hd * 4 + 64 * G <= 48 * 1024 and bool(getenv("FZ_PAGED_DECODE", 1))

def decode_part() -> int: return max(8, getenv("FZ_PAGED_DECODE_PART", DECODE_PART))

def render_decode(name: str, dt: DType, hd: int, G: int, part: int, loop: bool = True, U: int = 2) -> tuple[str, str]:
  """(partition kernel, combine kernel) of the one-pass paged decode: a CTA per (token, kv head, partition of `part`
  keys) serves the kv head's G q heads from one read of each K/V row, scores and P V in one pass (an online softmax per
  key group, the groups merged in order at the end); the combine kernel is the split decode's, with this partition
  size. loop=False: NP covers max_seq_len, one chunk per CTA."""
  T = "float" if dt == dtypes.float32 else "unsigned short"
  helpers = {dtypes.float32: "", dtypes.float16: HALF_HELPERS, dtypes.bfloat16: BF16_HELPERS}[dt]
  st = {dtypes.float32: "", dtypes.float16: "fz_f2h", dtypes.bfloat16: "fz_f2bf"}[dt]
  lpr = hd // 8
  src = _DECODE.substitute(NAME=name, T=T, LD8=_LD8[dt], G=G, HD=hd, PART=part, LPR=lpr, RPI=128 // lpr, U=U,
                           LOOP=f"int c = p; c * {part} <= pos; c += NP" if loop else "int c = p, once = 1; once; once = 0")
  comb = _COMBINE.substitute(NAME=name + "_combine", T=T, HD=hd, PART=part, ST=f"{st}(num / den)",
                             ZERO="0.0f" if dt == dtypes.float32 else "(unsigned short)0")
  return "\n".join([CUDA_PRELUDE, helpers, src]), "\n".join([CUDA_PRELUDE, helpers, comb])

def decode_np(N: int, Hq: int, MB: int, bs: int, part: int, rows: int | None = None) -> int:
  rows = getenv("FZ_PAGED_DECODE_ROWS", DECODE_ROWS) if rows is None else rows
  return max(1, min(-(-MB * bs // part), rows // max(1, N * Hq)))

def decode_calls(dt: DType, N: int, Hkv: int, G: int, hd: int, bs: int, NB: int, MB: int, xs: int, mrow: int, mpos: int, mbt: int,
                 scale: float, np_n: int = 0, part: int | None = None, rows: int | None = None, U: int | None = None) -> tuple["AttnCall", "AttnCall"]:
  """The one-pass decode launches. NP partitions of `part` keys (default decode_part()) cover max_seq_len, capped so the
  f32 scratch would hold at most `rows` np_n * Hq * NP rows; np_n (default N) is the largest row count of the forwards
  that must agree bitwise (the decode buckets), so NP, and with it a token's sum order, never depends on the batch."""
  assert np_n == 0 or np_n >= N, f"np_n {np_n} < N {N}"
  part = part or decode_part()
  U = U or max(1, getenv("FZ_PAGED_DECODE_U", 2))   # keys in flight per lane group (registers vs latency hiding)
  NP = decode_np(np_n or N, Hkv * G, MB, bs, part, rows)
  loop = NP < -(-MB * bs // part)
  name = f"fz_paged_decode_{_tag(dt)}_d{hd}_g{G}_p{part}{'' if U == 2 else f'_u{U}'}{'_loop' if loop else ''}"
  src, comb = render_decode(name, dt, hd, G, part, loop, U)
  return (AttnCall(name, src, 0, (N, Hkv, bs, NB, MB, xs, mrow, mpos, mbt, _bits(scale), NP), (N * Hkv * NP, 1, 1), (128, 1, 1)),
          AttnCall(name + "_combine", comb, 0, (N, Hkv * G, mrow, mpos, NP), (N * Hkv * G, 1, 1), (hd, 1, 1)))


# ---------------------------------------------------------------- tensor-core GQA paged decode (speed run 4, A/B 3)

_MDEC = Template(r"""extern "C" __global__ void __launch_bounds__(32) ${NAME}(float* __restrict__ Acc, float* __restrict__ ML, const u16* __restrict__ X,
    const u16* __restrict__ P, const int* __restrict__ M, const int N, const int Hkv, const int bs, const int NB, const int MB, const int xs,
    const int mrow, const int mpos, const int mbt, const int scale_bits, const int NP, const int X0) {
  extern __shared__ __align__(16) unsigned char fz_sm[];
  const u32 sK = fz_smem(fz_sm), sV = sK + 2u * ${BN} * ${D} * 2u;     // [2 stages][BN rows][D], 16-byte chunks swizzled by row
  const int tid = threadIdx.x, lane = tid, g = lane >> 2, t4 = lane & 3;
  int bid = blockIdx.x;
  const int p = bid % NP; bid /= NP;
  const int kh = bid % Hkv, t = bid / Hkv;
  const int row = M[mrow + t];
  if (row < 0) return;                                                  // padding token: the combine kernel writes zeros
  const int pos = M[mpos + t];
  if (p * ${PART} > pos) return;                                        // partition past the sequence
  const int* bt = M + mbt + row * MB;
  const float sl = __int_as_float(scale_bits) * 1.4426950408889634f;
  // the MMA's 16 rows are the kv head's G q heads (rows >= G are zero and never written)
  const u16* xq = X + (long long)(t + X0) * xs + (long long)kh * ${G} * ${D};
  u32 qa[${D} / 16][4];
#pragma unroll
  for (int ks = 0; ks < ${D} / 16; ks++) {
    const int c = ks * 16 + 2 * t4;
    qa[ks][0] = g < ${G} ? *(const u32*)(xq + g * ${D} + c) : 0u;
    qa[ks][1] = g + 8 < ${G} ? *(const u32*)(xq + (g + 8) * ${D} + c) : 0u;
    qa[ks][2] = g < ${G} ? *(const u32*)(xq + g * ${D} + c + 8) : 0u;
    qa[ks][3] = g + 8 < ${G} ? *(const u32*)(xq + (g + 8) * ${D} + c + 8) : 0u;
  }
  float o[${D} / 8][4];
#pragma unroll
  for (int i = 0; i < ${D} / 8; i++) { o[i][0] = 0.f; o[i][1] = 0.f; o[i][2] = 0.f; o[i][3] = 0.f; }
  float mx0 = -INFINITY, mx1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const long long vo = (long long)NB * Hkv * bs * ${D};
  const int lim0 = pos, lim1 = pos, kend = pos + 1;                     // key j visible iff j <= pos (all rows: one token)
  // key blocks of this CTA: partitions p, p + NP, ... (each PART keys), BN keys per block
  for (int c0 = p; c0 * ${PART} <= pos; c0 += NP) {
    const int jb0 = c0 * (${PART} / ${BN}), nb = min((c0 + 1) * (${PART} / ${BN}), (kend + ${BN} - 1) / ${BN});
    __syncthreads();
${PIPE_HEAD}
""" + _FLASH_STEP + r"""  }
""" + _FLASH_LSUM + r"""  const long long th0 = ((long long)t * Hkv + kh) * ${G};
  if (g < ${G}) {
    float* a = Acc + ((th0 + g) * NP + p) * ${D};
#pragma unroll
    for (int dt = 0; dt < ${D} / 8; dt++) { const int c = dt * 8 + 2 * t4; a[c] = o[dt][0]; a[c + 1] = o[dt][1]; }
    if (t4 == 0) { ML[((th0 + g) * NP + p) * 2] = mx0 * sl; ML[((th0 + g) * NP + p) * 2 + 1] = l0; }
  }
  if (g + 8 < ${G}) {
    float* a = Acc + ((th0 + g + 8) * NP + p) * ${D};
#pragma unroll
    for (int dt = 0; dt < ${D} / 8; dt++) { const int c = dt * 8 + 2 * t4; a[c] = o[dt][2]; a[c + 1] = o[dt][3]; }
    if (t4 == 0) { ML[((th0 + g + 8) * NP + p) * 2] = mx1 * sl; ML[((th0 + g + 8) * NP + p) * 2 + 1] = l1; }
  }
}
""")

MDEC_PART = 128   # keys per partition of the tensor-core decode (FZ_PAGED_MMA_PART); fixed per engine, never per batch

def mdec_ok(dev: Any, dt: DType, hd: int, G: int, xs: int) -> bool:
  """The tensor-core GQA decode applies (off by default: FZ_PAGED_MMA=1; A/B 3 measured it slower than the one-pass kernel): f16/bf16, hd 64 / 128, G <= 16 q heads per kv head
  (the MMA's 16 rows), 4-byte aligned q rows, sm_80+."""
  return bool(getenv("FZ_PAGED_MMA", 0)) and dt in _HALVES and hd in (64, 128) and 1 <= G <= 16 and xs % 2 == 0 and _cc(dev) >= (8, 0)

def render_mdec(name: str, dt: DType, hd: int, G: int, part: int, BN: int = 64) -> tuple[str, str]:
  """(partition kernel, combine kernel) of the tensor-core paged decode: one warp per (token, kv head, partition); the
  MMA's rows are the kv head's G q heads, so each K/V block (cp.async double buffer, swizzled ldmatrix, the flash
  kernels' online softmax with P rounded to the input type) is read once for all of them. Each partition writes its
  unnormalized output and (max in the log2 domain, sum) in the split decode's layout; that combine kernel merges them."""
  assert dt in _HALVES and hd in (64, 128) and part % BN == 0 and (BN * hd // 8) % 32 == 0, (dt, hd, part, BN)
  sub = dict(NAME=name, TY="bf16" if dt == dtypes.bfloat16 else "f16", D=hd, BN=BN, NT=32, G=G, PART=part, CH=BN * hd // 8 // 32)
  head, v = _pipe(False, "kend", _paged_src)
  head = head.replace("if (nb > 0)", "if (jb0 < nb)").replace("for (int jb = 0; jb < nb; jb++)", "for (int jb = jb0; jb < nb; jb++)")
  head = head.replace("(0) * ${BN}", "(jb0) * ${BN}").replace("((0) & 1)", "((jb0) & 1)")
  src = _MDEC.safe_substitute(PIPE_HEAD=head, PIPE_V=v, MASK=_PAGED_MASK)
  part_src = CUDA_PRELUDE + "\n" + _PTX.substitute(sub) + Template(src).substitute(sub)
  st = {dtypes.float16: "fz_f2h", dtypes.bfloat16: "fz_f2bf"}[dt]
  helpers = {dtypes.float16: HALF_HELPERS, dtypes.bfloat16: BF16_HELPERS}[dt]
  ctpl = _COMBINE.template.replace("const int NP) {", "const int NP, const int O0) {").replace(
    "O[th * ${HD} + d]", "O[(th + (long long)O0 * Hq) * ${HD} + d]")
  assert ctpl.count("O0") == 3, "the combine template changed"
  comb = Template(ctpl).substitute(NAME=name + "_combine", T="unsigned short", HD=hd, PART=part, ST=f"{st}(num / den)", ZERO="(unsigned short)0")
  return part_src, "\n".join([CUDA_PRELUDE, helpers, comb])

def mdec_calls(dt: DType, N: int, Hkv: int, G: int, hd: int, bs: int, NB: int, MB: int, xs: int, mrow: int, mpos: int, mbt: int,
               scale: float, np_n: int = 0, part: int | None = None, rows: int | None = None, row0: int = 0) -> tuple["AttnCall", "AttnCall"]:
  """The tensor-core decode launches; NP as decode_calls (fixed by np_n, never by the step's rows). row0: the decode
  rows are rows row0 .. row0 + N - 1 of the step (x, the metadata and the output; a mixed forward's decode region)."""
  assert np_n == 0 or np_n >= N, f"np_n {np_n} < N {N}"
  part = part or max(64, getenv("FZ_PAGED_MMA_PART", MDEC_PART) // 64 * 64)
  NP = decode_np(np_n or N, Hkv * G, MB, bs, part, rows)
  name = f"fz_paged_mdec_{_tag(dt)}_d{hd}_g{G}_p{part}"
  src, comb = render_mdec(name, dt, hd, G, part)
  return (AttnCall(name, src, flash_smem(hd, 64), (N, Hkv, bs, NB, MB, xs, mrow + row0, mpos + row0, mbt, _bits(scale), NP, row0),
                   (N * Hkv * NP, 1, 1), (32, 1, 1)),
          AttnCall(name + "_combine", comb, 0, (N, Hkv * G, mrow + row0, mpos + row0, NP, row0), (N * Hkv * G, 1, 1), (hd, 1, 1)))


# ---------------------------------------------------------------- 4-warp tensor-core GQA paged decode (speed run 4, round 14)

_TCD = Template(r"""extern "C" __global__ void __launch_bounds__(128) ${NAME}(float* __restrict__ Acc, float* __restrict__ ML, const u16* __restrict__ X,
    const u16* __restrict__ P, const int* __restrict__ M, const int N, const int Hkv, const int bs, const int NB, const int MB, const int xs,
    const int mrow, const int mpos, const int mbt, const int scale_bits, const int NP, const int X0) {
  // one CTA (4 warps) per (token, kv head, 128-key partition); warp w owns keys w*32 .. w*32+31 of the partition. The
  // MMA rows are the kv head's G q heads (rows >= G are zero, never written), so each K / V row is read once for all of
  // them. Partition-major block order: the partitions past every token's context (most of the grid at a large
  // max_seq_len) are the last blocks dispatched and exit at once.
  extern __shared__ __align__(16) unsigned char fz_sm[];
  u16* sK = (u16*)fz_sm;                                          // [128 keys][${D} + 8] (16-byte row padding: no bank conflicts)
  u16* sV = sK + 128 * (${D} + 8);
  long long* soff = (long long*)(sV + 128 * (${D} + 8));          // each key's row in the pool (the K half)
  float* red_m = (float*)(soff + 128);                            // [4 warps][16 rows]
  float* red_l = red_m + 64;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t4 = lane & 3;
  const int TK = N * Hkv;
  // work items (partition, token, kv head) in partition-major order, item = p * TK + tk, CTA c takes c, c + gridDim.x,
  // ...; the loop ends at the first partition past every token's position, so a large max_seq_len costs no CTAs (the
  // graph's grid is fixed at capture: FZ_PAGED_TC_CTAS). An item's arithmetic does not depend on the CTA that runs it.
  __shared__ int s_maxpos;
  if (tid == 0) {
    int mp = -1;
    for (int i = 0; i < N; i++) if (M[mrow + i] >= 0) mp = max(mp, M[mpos + i]);
    s_maxpos = mp;
  }
  __syncthreads();
  const int maxpos = s_maxpos;
  for (int item = blockIdx.x; item < NP * TK; item += gridDim.x) {
  const int p = item / TK, tk = item % TK;
  if (p * 128 > maxpos) break;                                    // this and every later item: past every position
  const int kh = tk % Hkv, t = tk / Hkv;
  const int row = M[mrow + t];
  if (row < 0) continue;                                          // padding token: the combine kernel writes zeros
  const int pos = M[mpos + t];
  if (p * 128 > pos) continue;                                    // partition past the sequence: never read
  __syncthreads();                                                // the previous item is done with the shared tiles
  const int j0 = p * 128, jn = min(128, pos + 1 - j0);           // keys j0 .. j0 + jn - 1 are visible
  const int* bt = M + mbt + row * MB;
  const float sl = __int_as_float(scale_bits) * 1.4426950408889634f;
  // this warp's 32 key rows: offsets, then K and V tiles by cp.async (two groups: S needs only K)
  const int kw = w * 32;
  { const int r = kw + lane; if (r < jn) { const int j = j0 + r; soff[r] = ((long long)(bt[j / bs] * Hkv + kh) * bs + j % bs) * ${D}; } }
  __syncwarp();
  const long long vo = (long long)NB * Hkv * bs * ${D};
#pragma unroll
  for (int it = 0; it < 32 * ${D} / 8 / 32; it++) {              // 16-byte chunks of the warp's 32 rows
    const int i = lane + it * 32, r = kw + i / (${D} / 8), c = i % (${D} / 8);
    const bool ok = r < jn;
    fz_cp16(fz_smem(sK + r * (${D} + 8) + c * 8), P + (ok ? soff[r] : 0) + c * 8, ok ? 16 : 0);
  }
  fz_commit();
#pragma unroll
  for (int it = 0; it < 32 * ${D} / 8 / 32; it++) {
    const int i = lane + it * 32, r = kw + i / (${D} / 8), c = i % (${D} / 8);
    const bool ok = r < jn;
    fz_cp16(fz_smem(sV + r * (${D} + 8) + c * 8), P + vo + (ok ? soff[r] : 0) + c * 8, ok ? 16 : 0);
  }
  fz_commit();
  // q fragments (A operand, rows = q heads), while the tiles land
  const u16* xq = X + (long long)(t + X0) * xs + (long long)kh * ${G} * ${D};
  u32 qa[${D} / 16][4];
#pragma unroll
  for (int ks = 0; ks < ${D} / 16; ks++) {
    const int c = ks * 16 + 2 * t4;
    qa[ks][0] = g < ${G} ? *(const u32*)(xq + g * ${D} + c) : 0u;
    qa[ks][1] = g + 8 < ${G} ? *(const u32*)(xq + (g + 8) * ${D} + c) : 0u;
    qa[ks][2] = g < ${G} ? *(const u32*)(xq + g * ${D} + c + 8) : 0u;
    qa[ks][3] = g + 8 < ${G} ? *(const u32*)(xq + (g + 8) * ${D} + c + 8) : 0u;
  }
  fz_wait1();
  __syncwarp();
  // S = Q K^T for the warp's 4 tiles of 8 keys
  float s[4][4];
#pragma unroll
  for (int nt = 0; nt < 4; nt++) {
    s[nt][0] = 0.f; s[nt][1] = 0.f; s[nt][2] = 0.f; s[nt][3] = 0.f;
    const u16* kr = sK + (kw + nt * 8 + g) * (${D} + 8) + 2 * t4;
#pragma unroll
    for (int ks = 0; ks < ${D} / 16; ks++) fz_mma(s[nt], qa[ks], *(const u32*)(kr + ks * 16), *(const u32*)(kr + ks * 16 + 8));
  }
  // scale into the log2 domain, mask the keys past the position, the row max over the CTA's 128 keys (4 warps)
  float m0 = -INFINITY, m1 = -INFINITY;
#pragma unroll
  for (int nt = 0; nt < 4; nt++) {
#pragma unroll
    for (int e = 0; e < 2; e++) {
      const bool ok = kw + nt * 8 + 2 * t4 + e < jn;
      s[nt][e] = ok ? s[nt][e] * sl : -INFINITY;
      s[nt][2 + e] = ok ? s[nt][2 + e] * sl : -INFINITY;
      m0 = fmaxf(m0, s[nt][e]); m1 = fmaxf(m1, s[nt][2 + e]);
    }
  }
  m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1)); m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
  m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
  if (t4 == 0) { red_m[w * 16 + g] = m0; red_m[w * 16 + g + 8] = m1; }
  __syncthreads();
  m0 = fmaxf(fmaxf(red_m[g], red_m[16 + g]), fmaxf(red_m[32 + g], red_m[48 + g]));            // key 0 is always visible: finite
  m1 = fmaxf(fmaxf(red_m[8 + g], red_m[24 + g]), fmaxf(red_m[40 + g], red_m[56 + g]));
  // P = 2^(s - m) (masked keys give 0), the row sums in a fixed order, P rounded to the input type for the PV MMA
  float l0 = 0.f, l1 = 0.f;
#pragma unroll
  for (int nt = 0; nt < 4; nt++) {
    s[nt][0] = exp2f(s[nt][0] - m0); s[nt][1] = exp2f(s[nt][1] - m0);
    s[nt][2] = exp2f(s[nt][2] - m1); s[nt][3] = exp2f(s[nt][3] - m1);
    l0 += s[nt][0] + s[nt][1]; l1 += s[nt][2] + s[nt][3];
  }
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  if (t4 == 0) { red_l[w * 16 + g] = l0; red_l[w * 16 + g + 8] = l1; }
  u32 pa[2][4];
#pragma unroll
  for (int j = 0; j < 2; j++) {
    pa[j][0] = fz_pack(s[2 * j][0], s[2 * j][1]); pa[j][1] = fz_pack(s[2 * j][2], s[2 * j][3]);
    pa[j][2] = fz_pack(s[2 * j + 1][0], s[2 * j + 1][1]); pa[j][3] = fz_pack(s[2 * j + 1][2], s[2 * j + 1][3]);
  }
  fz_wait0();
  __syncwarp();
  // O_w = P V over the warp's 32 keys: 16 tiles of 8 columns
  float o[${D} / 8][4];
#pragma unroll
  for (int nt = 0; nt < ${D} / 8; nt++) {
    o[nt][0] = 0.f; o[nt][1] = 0.f; o[nt][2] = 0.f; o[nt][3] = 0.f;
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const u16* vr = sV + (kw + j * 16 + 2 * t4) * (${D} + 8) + nt * 8 + g;
      const u32 b0 = (u32)vr[0] | ((u32)vr[${D} + 8] << 16);
      const u32 b1 = (u32)vr[8 * (${D} + 8)] | ((u32)vr[9 * (${D} + 8)] << 16);
      fz_mma(o[nt], pa[j], b0, b1);
    }
  }
  // the 4 warps' partial outputs, summed in warp order (the K tile's space, no longer read)
  __syncthreads();
  float* red = (float*)sK;                                        // [4 warps][16 rows][${D}]
#pragma unroll
  for (int nt = 0; nt < ${D} / 8; nt++) {
    const int c = nt * 8 + 2 * t4;
    red[(w * 16 + g) * ${D} + c] = o[nt][0]; red[(w * 16 + g) * ${D} + c + 1] = o[nt][1];
    red[(w * 16 + g + 8) * ${D} + c] = o[nt][2]; red[(w * 16 + g + 8) * ${D} + c + 1] = o[nt][3];
  }
  __syncthreads();
  const long long th0 = ((long long)t * Hkv + kh) * ${G};
  for (int i = tid; i < ${G} * ${D}; i += 128) {
    const int r = i / ${D}, d = i % ${D};
    Acc[((th0 + r) * NP + p) * ${D} + d] = (red[r * ${D} + d] + red[(16 + r) * ${D} + d]) + (red[(32 + r) * ${D} + d] + red[(48 + r) * ${D} + d]);
  }
  if (tid < ${G}) {
    ML[((th0 + tid) * NP + p) * 2] = fmaxf(fmaxf(red_m[tid], red_m[16 + tid]), fmaxf(red_m[32 + tid], red_m[48 + tid]));
    ML[((th0 + tid) * NP + p) * 2 + 1] = (red_l[tid] + red_l[16 + tid]) + (red_l[32 + tid] + red_l[48 + tid]);
  }
  }
}
""")

TCD_PART = 128
TCD_CTAS = 396   # FZ_PAGED_TC_CTAS: the fixed grid (132 SMs x 3 resident CTAs of ~71 KB on an H100); 0: one CTA per item

def tcd_smem(hd: int) -> int: return 2 * 128 * (hd + 8) * 2 + 128 * 8 + 2 * 64 * 4

def tcd_ok(dev: Any, dt: DType, hd: int, G: int, xs: int) -> bool:
  """The 4-warp tensor-core decode applies (default since round 15; FZ_PAGED_TC=0 falls back to kpt): f16/bf16, hd 128,
  G <= 16 q heads per kv head (the MMA's 16 rows), 4-byte aligned q rows, sm_80+ (mma.sync + cp.async)."""
  return bool(getenv("FZ_PAGED_TC", 1)) and dt in _HALVES and hd == 128 and 1 <= G <= 16 and xs % 2 == 0 and _cc(dev) >= (8, 0)

def render_tcd(name: str, dt: DType, hd: int, G: int) -> tuple[str, str]:
  sub = dict(NAME=name, TY="bf16" if dt == dtypes.bfloat16 else "f16", D=hd, G=G)
  part_src = CUDA_PRELUDE + "\n" + _PTX.substitute(sub) + _TCD.substitute(sub)
  st = {dtypes.float16: "fz_f2h", dtypes.bfloat16: "fz_f2bf"}[dt]
  helpers = {dtypes.float16: HALF_HELPERS, dtypes.bfloat16: BF16_HELPERS}[dt]
  comb = Template(_combine_o0()).substitute(NAME=name + "_combine", T="unsigned short", HD=hd, PART=TCD_PART, ST=f"{st}(num / den)",
                                            ZERO="(unsigned short)0")
  return part_src, "\n".join([CUDA_PRELUDE, helpers, comb])

def tcd_calls(dt: DType, N: int, Hkv: int, G: int, hd: int, bs: int, NB: int, MB: int, xs: int, mrow: int, mpos: int, mbt: int,
              scale: float, np_n: int = 0, row0: int = 0) -> tuple["AttnCall", "AttnCall"]:
  """NP = the 128-key partitions of max_seq_len (as kpt: never depends on the batch); one CTA per (token, kv head,
  partition), blocks in partition-major order; the combine merges a token's own partitions in order. A token's result
  depends only on its own context (its partitions, each summed in a fixed order), so batched == alone."""
  assert np_n == 0 or np_n >= N
  NP = -(-MB * bs // TCD_PART)
  ctas = getenv("FZ_PAGED_TC_CTAS", TCD_CTAS)
  grid = N * Hkv * NP if ctas <= 0 else min(N * Hkv * NP, ctas)
  name = f"fz_paged_tcd_{_tag(dt)}_d{hd}_g{G}"
  src, comb = render_tcd(name, dt, hd, G)
  return (AttnCall(name, src, tcd_smem(hd), (N, Hkv, bs, NB, MB, xs, mrow + row0, mpos + row0, mbt, _bits(scale), NP, row0), (grid, 1, 1), (128, 1, 1)),
          AttnCall(name + "_combine", comb, 0, (N, Hkv * G, mrow + row0, mpos + row0, NP, row0), (N * Hkv * G, 1, 1), (hd, 1, 1)))


# ---------------------------------------------------------------- key-per-thread paged decode (speed run 4, round 4)

_UP8 = {
  dtypes.float16: """static __device__ __forceinline__ void fz_up8(const fz_u4 a, float* v) {
  const unsigned u[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
  for (int i = 0; i < 4; i++) { v[2 * i] = fz_h2f((unsigned short)(u[i] & 0xffffu)); v[2 * i + 1] = fz_h2f((unsigned short)(u[i] >> 16)); }
}""",
  dtypes.bfloat16: """static __device__ __forceinline__ void fz_up8(const fz_u4 a, float* v) {
  const unsigned u[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
  for (int i = 0; i < 4; i++) { v[2 * i] = __uint_as_float(u[i] << 16); v[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u); }
}"""}

_KPT = Template(r"""struct __align__(16) fz_u4 { unsigned x, y, z, w; };
typedef unsigned int u32;
static __device__ __forceinline__ u32 fz_smem(const void* p) { u32 r; asm("{ .reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a; }" : "=r"(r) : "l"(p)); return r; }
${LD8}
${UP8}
extern "C" __global__ void __launch_bounds__(128) ${NAME}(float* __restrict__ Acc, float* __restrict__ ML, const unsigned short* __restrict__ X,
    const unsigned short* __restrict__ P, const int* __restrict__ M, const int N, const int Hkv, const int bs, const int NB, const int MB,
    const int xs, const int mrow, const int mpos, const int mbt, const int scale_bits, const int NP, const int X0, const int NPL) {
  // NPL CTAs per (token, kv head); CTA c computes the token's 128-key partitions c, c + NPL, ... up to its position, one
  // at a time with the same code (a partition's result does not depend on NPL or on which CTA ran it); thread r scores
  // key r of the partition for the kv head's G q heads
  // (X0: the first of these rows in X, for a mixed forward's decode rows; the metadata offsets come shifted already)
  __shared__ __align__(16) float sq[${G}][${HD}];                 // q * scale * log2(e)
  __shared__ __align__(16) unsigned short sk[128][${HD} + 8];     // the K tile (rows padded by 16 bytes); then the group partials
  __shared__ __align__(16) float sp[${G}][128];                   // probabilities
  __shared__ float smx[${G}][4], ssm[${G}][4];
  __shared__ long long soff[128];                                 // each key's row in the pool (the K half)
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  int bid = blockIdx.x;
  const int p0 = bid % NPL; bid /= NPL;
  const int kh = bid % Hkv, t = bid / Hkv;
  const int row = M[mrow + t];
  if (row < 0) return;                                            // padding token: the combine kernel writes zeros
  const int pos = M[mpos + t];
  for (int p = p0; p * 128 <= pos; p += NPL) {                    // partitions past the sequence: never run
  __syncthreads();                                                // the previous partition is done with the shared tiles
  const int j0 = p * 128, jn = min(128, pos + 1 - j0);
  const int* bt = M + mbt + row * MB;
  const float sl = __int_as_float(scale_bits) * 1.4426950408889634f;
  for (int i = tid; i < ${G} * ${HD} / 8; i += 128) {
    float v[8];
    const int g = i / (${HD} / 8), c = i % (${HD} / 8);
    fz_ld8(X + (long long)(t + X0) * xs + (long long)(kh * ${G} + g) * ${HD} + c * 8, v);
#pragma unroll
    for (int e = 0; e < 8; e++) sq[g][c * 8 + e] = v[e] * sl;
  }
  if (tid < jn) { const int j = j0 + tid; soff[tid] = ((long long)(bt[j / bs] * Hkv + kh) * bs + j % bs) * ${HD}; }
  __syncthreads();
  const long long vo = (long long)NB * Hkv * bs * ${HD};
  // K tile -> shared memory, coalesced (16-byte chunks of consecutive rows); V slices of this thread's keys -> registers,
  // all in flight together before any score is computed
#pragma unroll
  for (int it = 0; it < ${HD} / 8; it++) {                       // cp.async: 16 copies in flight per thread, no registers
    const int i = tid + it * 128, r = i / (${HD} / 8), c = i % (${HD} / 8);
    const u32 dst = fz_smem(&sk[r][c * 8]);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(P + (r < jn ? soff[r] : 0) + c * 8), "r"(r < jn ? 16 : 0) : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  const int cg = tid % (${HD} / 8), kg = tid / (${HD} / 8);       // P V: column group (8 columns) x key group
  fz_u4 vr[${KPG}];                                               // keys kg, kg + KG, ... of the partition
${VLOAD_EARLY}
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  // scores of key tid
  float s[${G}];
#pragma unroll
  for (int g = 0; g < ${G}; g++) s[g] = 0.f;
  if (tid < jn) {
#pragma unroll 4
    for (int c = 0; c < ${HD} / 8; c++) {
      float kv[8];
      fz_up8(*(const fz_u4*)&sk[tid][c * 8], kv);
#pragma unroll
      for (int g = 0; g < ${G}; g++) {
        const float4 q0 = *(const float4*)&sq[g][c * 8], q1 = *(const float4*)&sq[g][c * 8 + 4];
        s[g] = fmaf(q0.x, kv[0], s[g]); s[g] = fmaf(q0.y, kv[1], s[g]); s[g] = fmaf(q0.z, kv[2], s[g]); s[g] = fmaf(q0.w, kv[3], s[g]);
        s[g] = fmaf(q1.x, kv[4], s[g]); s[g] = fmaf(q1.y, kv[5], s[g]); s[g] = fmaf(q1.z, kv[6], s[g]); s[g] = fmaf(q1.w, kv[7], s[g]);
      }
    }
  } else {
#pragma unroll
    for (int g = 0; g < ${G}; g++) s[g] = -INFINITY;
  }
  // softmax over the partition's keys, per q head: block max and sum in a fixed order
#pragma unroll
  for (int g = 0; g < ${G}; g++) {
    float m = s[g];
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) smx[g][w] = m;
  }
  __syncthreads();
  float mx[${G}];
#pragma unroll
  for (int g = 0; g < ${G}; g++) {
    mx[g] = fmaxf(fmaxf(smx[g][0], smx[g][1]), fmaxf(smx[g][2], smx[g][3]));
    const float e = tid < jn ? exp2f(s[g] - mx[g]) : 0.f;
    sp[g][tid] = e;
    float l = e;
    for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
    if (lane == 0) ssm[g][w] = l;
  }
  __syncthreads();
${VLOAD_LATE}
  // P V for this thread's keys and 8 columns
  float acc[${G}][8];
#pragma unroll
  for (int g = 0; g < ${G}; g++)
#pragma unroll
    for (int e = 0; e < 8; e++) acc[g][e] = 0.f;
#pragma unroll
  for (int i = 0; i < ${KPG}; i++) {
    const int k = kg + ${KG} * i;
    float v[8];
    fz_up8(vr[i], v);
#pragma unroll
    for (int g = 0; g < ${G}; g++) {
      const float pk = sp[g][k];
#pragma unroll
      for (int e = 0; e < 8; e++) acc[g][e] = fmaf(pk, v[e], acc[g][e]);
    }
  }
  // the key groups' partials (in the K tile's space, no longer read), summed in group order
  float* red = (float*)&sk[0][0];
#pragma unroll
  for (int g = 0; g < ${G}; g++)
#pragma unroll
    for (int e = 0; e < 8; e++) red[(kg * ${G} + g) * ${HD} + cg * 8 + e] = acc[g][e];
  __syncthreads();
  for (int i = tid; i < ${G} * ${HD}; i += 128) {
    const int g = i / ${HD}, d = i % ${HD};
    float a = 0.f;
#pragma unroll
    for (int r = 0; r < ${KG}; r++) a += red[(r * ${G} + g) * ${HD} + d];
    const long long th = ((long long)t * Hkv + kh) * ${G} + g;
    Acc[(th * NP + p) * ${HD} + d] = a;
    if (d == 0) {
      ML[(th * NP + p) * 2] = fmaxf(fmaxf(smx[g][0], smx[g][1]), fmaxf(smx[g][2], smx[g][3]));
      ML[(th * NP + p) * 2 + 1] = (ssm[g][0] + ssm[g][1]) + (ssm[g][2] + ssm[g][3]);
    }
  }
  }
}
""")

KPT_PART = 128
KPT_CTAS = 0     # FZ_PAGED_KPT_CTAS: target CTAs per decode attention launch; 0 (default): one CTA per partition. Round 12:
                 # 512 (NPL 4 at 32 rows) made a CTA run 2 of a ~600-key context's 5 partitions in series: bench c32 -2.4 to -3.9%

def kpt_ok(dt: DType, hd: int, G: int, xs: int, rows: int, tc: bool = False) -> bool:
  """The key-per-thread decode applies (default since round 4; FZ_PAGED_KPT=0 reverts to the one-pass kernel): f16/bf16,
  hd 128, G <= 8, 16-byte q rows, and one
  partition per CTA within the scratch cap (rows = np_n * Hq * partitions of max_seq_len)."""
  return (tc or bool(getenv("FZ_PAGED_KPT", 1))) and dt in _HALVES and hd == 128 and 1 <= G <= (16 if tc else 8) and xs % (2 if tc else 8) == 0 and \
    rows <= getenv("FZ_PAGED_DECODE_ROWS", DECODE_ROWS) * 4

_KPT_VLOAD = """#pragma unroll
  for (int i = 0; i < ${KPG}; i++) {
    const int k = kg + ${KG} * i;
    if (k < jn) vr[i] = *(const fz_u4*)(P + vo + soff[k] + cg * 8);
    else { vr[i].x = 0u; vr[i].y = 0u; vr[i].z = 0u; vr[i].w = 0u; }
  }"""

def render_kpt(name: str, dt: DType, hd: int, G: int, vpre: bool = True) -> tuple[str, str]:
  """vpre: the V slices load before the scores (in flight with the K tile; 64 more registers) or after the softmax."""
  helpers = {dtypes.float16: HALF_HELPERS, dtypes.bfloat16: BF16_HELPERS}[dt]
  st = {dtypes.float16: "fz_f2h", dtypes.bfloat16: "fz_f2bf"}[dt]
  KG = 128 // (hd // 8)
  vl = Template(_KPT_VLOAD).substitute(KG=KG, KPG=128 // KG)
  src = _KPT.substitute(NAME=name, LD8=_LD8[dt], UP8=_UP8[dt], G=G, HD=hd, KG=KG, KPG=128 // KG,
                        VLOAD_EARLY=vl if vpre else "", VLOAD_LATE="" if vpre else vl)
  comb = Template(_combine_o0()).substitute(NAME=name + "_combine", T="unsigned short", HD=hd, PART=KPT_PART, ST=f"{st}(num / den)",
                                            ZERO="(unsigned short)0")
  return "\n".join([CUDA_PRELUDE, helpers, src]), "\n".join([CUDA_PRELUDE, helpers, comb])

def _combine_o0() -> str:
  """The split decode's combine with an output row offset O0 (a mixed forward's decode rows start at row O0 of O)."""
  t = _COMBINE.template.replace("const int NP) {", "const int NP, const int O0) {").replace("O[th * ${HD} + d]", "O[(th + (long long)O0 * Hq) * ${HD} + d]")
  assert t.count("O0") == 3, "the combine template changed"
  return t

def kpt_calls(dt: DType, N: int, Hkv: int, G: int, hd: int, bs: int, NB: int, MB: int, xs: int, mrow: int, mpos: int, mbt: int,
              scale: float, np_n: int = 0, vpre: bool | None = None, row0: int = 0) -> tuple["AttnCall", "AttnCall"]:
  """NP = the 128-key partitions of max_seq_len (fixed by max_seq_len, so it never depends on the batch); the combine kernel
  merges a token's own partitions in order. NPL CTAs per (token, kv head) loop over the partitions (CTA c: c, c + NPL, ...
  up to the token's position). Round 12: one CTA per partition (NPL = NP) launched 16 partitions per token at Invarix's
  max_model_len 2048 for contexts <= 640, so 70% of a 32-row step's 2048 CTAs only read the metadata and exited; NPL is
  min(NP, ceil(FZ_PAGED_KPT_CTAS / (N * Hkv))) when FZ_PAGED_KPT_CTAS > 0; the default 0 keeps NPL = NP (round 12: 512, i.e.
  NPL 4 at 32 rows, put two of a ~600-key context's partitions on one CTA in series and lost 2.4-3.9% at bench c32,
  more than the exited CTAs cost; a graph-captured grid cannot follow the batch's longest context). A partition's
  arithmetic and the combine order do not depend on NPL, so the bits do not either."""
  assert np_n == 0 or np_n >= N
  NP = -(-MB * bs // KPT_PART)
  ctas = getenv("FZ_PAGED_KPT_CTAS", KPT_CTAS)
  NPL = NP if ctas <= 0 else max(1, min(NP, -(-ctas // max(1, N * Hkv))))
  vpre = bool(getenv("FZ_PAGED_KPT_VPRE", 1)) if vpre is None else vpre
  name = f"fz_paged_kpt_{_tag(dt)}_d{hd}_g{G}{'' if vpre else '_vlate'}"
  src, comb = render_kpt(name, dt, hd, G, vpre)
  return (AttnCall(name, src, 0, (N, Hkv, bs, NB, MB, xs, mrow + row0, mpos + row0, mbt, _bits(scale), NP, row0, NPL), (N * Hkv * NPL, 1, 1), (128, 1, 1)),
          AttnCall(name + "_combine", comb, 0, (N, Hkv * G, mrow + row0, mpos + row0, NP, row0), (N * Hkv * G, 1, 1), (hd, 1, 1)))

def call_np(a: "AttnCall") -> int:
  """The partition count NP of a kpt / mdec partition call (params[10] in both: N, Hkv, bs, NB, MB, xs, mrow, mpos, mbt,
  scale, NP, ...), which sizes the f32 Acc / ML scratch. Round 12: reading it as params[-2] broke when kpt gained a
  trailing NPL parameter (params[-2] became X0 = 0: a zero-sized scratch, inf / NaN outputs)."""
  return int(a.params[10])

def split_ok(hd: int, G: int, xs: int) -> bool:
  """The split-K decode applies: 8-element (16/32-byte) slices that tile a warp, 16-byte aligned q rows, G <= 16."""
  return hd % 8 == 0 and 32 % (hd // 8) == 0 and xs % 8 == 0 and 1 <= G <= 16 and bool(getenv("FZ_PAGED_SPLIT", 1))

def split_smem(hd: int, G: int) -> int: return (G * PART + (128 // (hd // 8)) * G * hd + 3 * G) * 4

def split_np(N: int, Hq: int, MB: int, bs: int, rows: int | None = None) -> int:
  """Partitions per token: enough to give every PART-key chunk of max_seq_len (MB * bs) its own CTA, capped so the f32
  scratch holds at most `rows` (FZ_PAGED_SPLIT_ROWS, default SPLIT_ROWS) N * Hq * NP rows. Under fz.jit every call's scratch
  is its own persistent buffer, so the cap bounds the memory per layer whatever max_seq_len and the step's token count
  (max_prefill_tokens for prefill chunks) are; past the cap a CTA loops over chunks p, p + NP, ..."""
  rows = getenv("FZ_PAGED_SPLIT_ROWS", SPLIT_ROWS) if rows is None else rows
  return max(1, min(-(-MB * bs // PART), rows // max(1, N * Hq)))

def render_split(name: str, dt: DType, hd: int, G: int, loop: bool = True) -> tuple[str, str]:
  """(partition kernel, combine kernel) sources for the split-K paged decode. loop=False: each CTA takes exactly one
  chunk (NP covers max_seq_len), so the chunk loop and the running max/sum fold away at compile time."""
  T = "float" if dt == dtypes.float32 else "unsigned short"
  helpers = {dtypes.float32: "", dtypes.float16: HALF_HELPERS, dtypes.bfloat16: BF16_HELPERS}[dt]
  st = {dtypes.float32: "", dtypes.float16: "fz_f2h", dtypes.bfloat16: "fz_f2bf"}[dt]
  lpr = hd // 8
  part = _SPLIT.substitute(NAME=name, T=T, LD8=_LD8[dt], G=G, HD=hd, PART=PART, LPR=lpr, RPI=128 // lpr, U=4,
                            LOOP="int c = p; c * " + str(PART) + " <= pos; c += NP" if loop else "int c = p, once = 1; once; once = 0")
  comb = _COMBINE.substitute(NAME=name + "_combine", T=T, HD=hd, PART=PART, ST=f"{st}(num / den)",
                             ZERO="0.0f" if dt == dtypes.float32 else "(unsigned short)0")
  return "\n".join([CUDA_PRELUDE, helpers, part]), "\n".join([CUDA_PRELUDE, helpers, comb])

# ---------------------------------------------------------------- launching

def _program(dev: Any, name: str, src: str, smem: int) -> Any:
  """Compiled program for (name, src) on dev, cached on the device instance; dynamic shared memory of `smem` bytes."""
  progs = dev.__dict__.setdefault("_fz_attn_progs", {})
  if (prg := progs.get((name, src))) is None:
    if getenv("FZ_DEBUG", 0) >= 4: print(src)
    progs[(name, src)] = prg = _Prog(dev, name, dev.compiler.compile_cached(src), smem)
  return prg

def max_smem(dev: Any) -> int:
  """Opt-in dynamic shared memory per block of this CUDA device (bytes)."""
  if (v := dev.__dict__.get("_fz_max_smem")) is None:
    from fusezap.device.cuda import _check
    A = dev.cu.CUdevice_attribute
    dev._fz_max_smem = v = int(_check(dev.cu.cuDeviceGetAttribute(A.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev.device)))
  return v

def _prog_cls() -> type:
  from fusezap.device.cuda import CUDAProgram, _check, debug_sync
  class Prog(CUDAProgram):
    """CUDAProgram with dynamic shared memory (opted in above 48 KB) and prepared launches for tight loops."""
    def __init__(self, dev: Any, name: str, lib: bytes, smem: int):
      super().__init__(dev, name, lib)
      self.smem = smem
      if smem > (mx := max_smem(dev)): raise RuntimeError(f"{name} needs {smem} bytes of shared memory, the device allows {mx}")
      if smem > 48 * 1024:
        A = dev.cu.CUfunction_attribute
        _check(dev.cu.cuFuncSetAttribute(self.fxn, A.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem), f"cuFuncSetAttribute({name}, smem={smem})")
    def prepare(self, bufs: Sequence[int], vals: Sequence[int], global_size: tuple[int, ...], local_size: tuple[int, ...]) -> Any:
      """A zero-argument launcher with the kernel params packed once (returns None; the launch is asynchronous)."""
      gs, ls = (tuple(int(x) for x in global_size) + (1, 1, 1))[:3], (tuple(int(x) for x in local_size) + (1, 1, 1))[:3]
      if ls[0] * ls[1] * ls[2] > self.max_threads: raise RuntimeError(f"local size {ls} exceeds {self.max_threads} threads for {self.name}")
      ptrs = (ctypes.c_uint64 * max(len(bufs), 1))(*[int(b) for b in bufs])
      ints = (ctypes.c_int32 * max(len(vals), 1))(*[int(v) for v in vals])
      pa, ia = ctypes.addressof(ptrs), ctypes.addressof(ints)
      params = (ctypes.c_void_p * max(len(bufs) + len(vals), 1))(*[pa + 8 * i for i in range(len(bufs))], *[ia + 4 * i for i in range(len(vals))])
      cu, dev, fxn, smem, keep, pp = self.dev.cu, self.dev, self.fxn, self.smem, (ptrs, ints, params), ctypes.addressof(params)
      def launch() -> None:
        _ = keep
        if 0 in gs: return
        dev.ctx()
        _check(cu.cuLaunchKernel(fxn, *gs, *ls, smem, dev.stream, pp, 0), f"cuLaunchKernel({self.name}, grid={gs}, block={ls}, smem={smem})")
        debug_sync(dev, self.name, gs, ls)
      return launch
    def __call__(self, *bufs: Any, global_size: tuple[int, ...] = (1, 1, 1), local_size: tuple[int, ...] | None = None,
                 vals: tuple[int, ...] = (), wait: bool = False) -> float | None:
      run = self.prepare(bufs, vals, global_size, local_size or (1,))
      if 0 in global_size: return 0.0 if wait else None
      cu, dev = self.dev.cu, self.dev
      if wait: _check(cu.cuEventRecord(dev.ev_start, 0), "cuEventRecord")
      run()
      if not wait: return None
      _check(cu.cuEventRecord(dev.ev_end, 0), "cuEventRecord")
      _check(cu.cuEventSynchronize(dev.ev_end), f"kernel {self.name}")
      return float(_check(cu.cuEventElapsedTime(dev.ev_start, dev.ev_end), "cuEventElapsedTime")) * 1e-3
  return Prog

class _ProgFactory:
  cls: type | None = None
  def __call__(self, *a: Any) -> Any:
    if _ProgFactory.cls is None: _ProgFactory.cls = _prog_cls()
    return _ProgFactory.cls(*a)
_Prog = _ProgFactory()

@dataclass(frozen=True, eq=False)
class AttnCall:
  """Payload (`ast[0]`) of a kind="custom" ScheduleItem for one of these kernels. `launch` is the fusezap.jit hook for
  external calls, so a fz.jit capture records it and a replay launches it with that replay's buffers."""
  name: str
  src: str
  smem: int
  params: tuple[int, ...]
  global_size: tuple[int, int, int]
  local_size: tuple[int, int, int]
  @property
  def kernel(self) -> "AttnCall": return self
  def program(self, dev: Any) -> Any: return _program(dev, self.name, self.src, self.smem)
  def launch(self, dev: Any, bufs: Sequence[Any], var_vals: dict | None = None, wait: bool | None = None) -> float | None:
    if bufs[0].size == 0: return None
    w = getenv("FZ_DEBUG", 0) >= 1 if wait is None else wait
    return self.program(dev)(*[b.ensure_allocated().handle for b in bufs], global_size=self.global_size, local_size=self.local_size,
                             vals=self.params, wait=w)
  def prepare(self, dev: Any, bufs: Sequence[Any]) -> Any:
    """Zero-argument async launcher (benchmarks)."""
    return self.program(dev).prepare([b.ensure_allocated().handle for b in bufs], self.params, self.global_size, self.local_size)
  def pretty(self, indent: int = 0) -> str: return f"{'  ' * indent}CUDA {self.name} {self.params}"

def _run(call: AttnCall, bufs: Sequence[Any], n_out: int = 1) -> None:
  from fusezap import jit as J
  from fusezap.ir import ScheduleItem
  J.run_item(ScheduleItem((call,), tuple(bufs), {}, "custom", n_out))   # type: ignore[arg-type]

def _cc(dev: Any) -> tuple[int, int]: return tuple(getattr(dev, "cc", (0, 0)))  # type: ignore[return-value]

def use_flash(dev: Any, dt: DType, D: int, Dv: int, scale: float) -> bool:
  """The tensor-core prefill applies: f16/bf16, D == Dv in (64, 128), sm_80+, a positive scale (FZ_FLASH=0 disables)."""
  return bool(getenv("FZ_FLASH", 1)) and dt in _HALVES and D == Dv and D in (64, 128) and _cc(dev) >= (8, 0) and scale > 0

def dense_call(dev: Any, dt: DType, Bt: int, Hq: int, G: int, L: int, S: int, D: int, Dv: int, q_st: tuple[int, int, int, int],
               k_st: tuple[int, int, int, int], v_st: tuple[int, int, int, int], o_st: tuple[int, int, int, int], causal: bool,
               scale: float, coff: int = 0, offs: tuple[int, int, int] = (0, 0, 0), flash: bool | None = None,
               cfg: tuple | None = None) -> AttnCall:
  """The prefill / hook launch. Strides per tensor are (batch, head, row, column) in elements; q head h reads kv head h // G.
  The tensor-core kernel needs unit column strides, batch/head/row strides that are multiples of 8 (16-byte rows) and
  zero element offsets; cfg = (NW, BN[, split K/V pipeline]) overrides its tile (benchmark sweeps)."""
  if flash is None:
    flash = use_flash(dev, dt, D, Dv, scale) and all(s[3] == 1 for s in (q_st, k_st, v_st, o_st)) and offs == (0, 0, 0) and \
            all(x % 8 == 0 for s in (q_st, k_st, v_st, o_st) for x in s[:3])
  if flash:
    NW, BN, split = (tuple(cfg) + (None,))[:3] if cfg else flash_config(D) + (None,)
    if split is None: split = bool(getenv("FZ_FLASH_SPLITKV", 0))
    name = f"fz_flash_{_tag(dt)}_d{D}_{16 * NW}x{BN}{'s' if split else ''}{'_causal' if causal else ''}"
    params = (Bt, Hq, G, L, S, *q_st[:3], *k_st[:3], *v_st[:3], *o_st[:3], coff, _bits(scale))
    return AttnCall(name, render_flash(name, dt, D, causal, NW, BN, split), flash_smem(D, BN), params,
                    (-(-L // (16 * NW)) * Hq * Bt, 1, 1), (32 * NW, 1, 1))
  W = 4
  name = f"fz_attn_simt_{_tag(dt)}_d{D}_{Dv}_w{W}{'_causal' if causal else ''}"
  params = (Bt, Hq, G, L, S, *q_st, *k_st, *v_st, *o_st, *offs, coff, _bits(scale), _bits(LOG2E))
  return AttnCall(name, render_simt(name, dt, D, Dv, W, causal), simt_smem(D, Dv, W), params, (-(-L // W) * Hq * Bt, 1, 1), (32 * W, 1, 1))

def paged_call(dt: DType, N: int, Hkv: int, G: int, hd: int, bs: int, NB: int, MB: int, xs: int, mrow: int, mpos: int, mbt: int,
               scale: float) -> AttnCall:
  name = f"fz_paged_attn_{_tag(dt)}_d{hd}_g{G}"
  return AttnCall(name, render_simt(name, dt, hd, hd, G, paged=True), simt_smem(hd, hd, G),
                  (N, Hkv, bs, NB, MB, xs, mrow, mpos, mbt, _bits(scale)), (N * Hkv, 1, 1), (32 * G, 1, 1))

def use_paged_flash(dev: Any, dt: DType, hd: int, xs: int) -> bool:
  """The tensor-core paged prefill applies (FZ_PAGED_FLASH=0 disables): f16/bf16, hd 64 or 128, sm_80+, even row stride."""
  return bool(getenv("FZ_PAGED_FLASH", 1)) and dt in _HALVES and hd in (64, 128) and xs % 2 == 0 and _cc(dev) >= (8, 0)

def paged_flash_call(dt: DType, N: int, Hkv: int, G: int, hd: int, bs: int, NB: int, MB: int, xs: int, mrow: int, mpos: int,
                     mbt: int, scale: float, NW: int | None = None, BN: int = 64) -> AttnCall:
  NW = getenv("FZ_PAGED_FLASH_NW", 4) if NW is None else NW
  name = f"fz_paged_flash_{_tag(dt)}_d{hd}_g{G}_{16 * NW}x{BN}"
  return AttnCall(name, render_paged_flash(name, dt, hd, G, NW, BN), flash_smem(hd, BN), (N, Hkv, bs, NB, MB, xs, mrow, mpos, mbt, _bits(scale)),
                  (-(-N // (16 * NW)) * Hkv * G, 1, 1), (32 * NW, 1, 1))

def split_calls(dt: DType, N: int, Hkv: int, G: int, hd: int, bs: int, NB: int, MB: int, xs: int, mrow: int, mpos: int, mbt: int,
                scale: float, rows: int | None = None, np_n: int = 0) -> tuple[AttnCall, AttnCall]:
  """The split-K paged decode: one CTA per (token, kv head, partition) writes its partial max/sum and unnormalized
  output, the combine kernel merges a token's partitions. Partition p takes the PART-key chunks p, p + NP, ... (split_np
  sizes NP for np_n rows, default N: a token's sum order depends on NP, so forwards of several row counts that must
  agree bitwise pass their largest); partitions past a token's position exit at once."""
  assert np_n == 0 or np_n >= N, f"np_n {np_n} < N {N}: the scratch cap would not hold"
  NP = split_np(np_n or N, Hkv * G, MB, bs, rows)
  loop = NP < -(-MB * bs // PART)
  name = f"fz_paged_split_{_tag(dt)}_d{hd}_g{G}{'_loop' if loop else ''}"
  part, comb = render_split(name, dt, hd, G, loop)
  return (AttnCall(name, part, split_smem(hd, G), (N, Hkv, bs, NB, MB, xs, mrow, mpos, mbt, _bits(scale), NP), (N * Hkv * NP, 1, 1), (128, 1, 1)),
          AttnCall(name + "_combine", comb, 0, (N, Hkv * G, mrow, mpos, NP), (N * Hkv * G, 1, 1), (hd, 1, 1)))

# ---------------------------------------------------------------- Tensor / Buffer entry points

def attention(q: Any, k: Any, v: Any, causal: bool = False, scale: float | None = None, coff: int = 0) -> Any:
  """Flash attention on CUDA == scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale, enable_gqa=True):
  q (..., Hq, L, D), k (..., Hkv, S, D), v (..., Hkv, S, Dv) with Hq % Hkv == 0 (q head h uses kv head h // (Hq/Hkv));
  2-d inputs are one head. Leading dims broadcast. One float dtype; f32 softmax and accumulation; output in that dtype.
  causal: key s visible to row l iff s <= l + coff. Inference only (no autograd)."""
  from fusezap.device import Buffer, Device
  from fusezap.lazy import LazyBuffer
  from fusezap.tensor import Tensor
  assert q.dtype == k.dtype == v.dtype and q.dtype in _FLOATS, f"attention needs one float dtype, got {q.dtype}, {k.dtype}, {v.dtype}"
  squeeze = q.ndim == 2
  if squeeze: q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
  assert q.ndim >= 3 and k.ndim >= 3 and v.ndim >= 3, f"attention shapes {q.shape}, {k.shape}, {v.shape}"
  (Hq, L, D), (Hkv, S, Dk), (Hv, Sv, Dv) = q.shape[-3:], k.shape[-3:], v.shape[-3:]
  assert D == Dk and S == Sv and Hkv == Hv and Hq % Hkv == 0, f"attention shapes {q.shape}, {k.shape}, {v.shape}"
  nb = max(t.ndim - 3 for t in (q, k, v))
  q, k, v = [t.reshape((1,) * (nb - (t.ndim - 3)) + tuple(t.shape)) for t in (q, k, v)]
  bs = tuple(max(t.shape[i] for t in (q, k, v)) for i in range(nb))
  Bt = prod(bs)
  def base(t: Any) -> Any:
    if prod(t.shape[:nb]) != 1 and tuple(t.shape[:nb]) != bs: t = t.expand(bs + tuple(t.shape[nb:]))
    assert t.device.split(":")[0].upper() == "CUDA", f"CUDA attention needs CUDA tensors, got {t.device}"
    return t._realized_contiguous().realized
  Q, K, V = base(q), base(k), base(v)
  O = Buffer(q.device, Bt * Hq * L * Dv, q.dtype)
  sc = 1.0 / math.sqrt(D) if scale is None else float(scale)
  st = lambda buf, H, R, C: (0 if buf.size == H * R * C else H * R * C, R * C, C, 1)
  call = dense_call(Device[q.device], q.dtype, Bt, Hq, Hq // Hkv, L, S, D, Dv, st(Q, Hq, L, D), st(K, Hkv, S, D), st(V, Hkv, S, Dv),
                    (Hq * L * Dv, L * Dv, Dv, 1), causal, sc, coff)
  if Bt and Hq and L and Dv:
    if S: _run(call, (O, Q, K, V))
    else: O.copyin(memoryview(bytearray(O.nbytes)))   # no keys: zeros, like the Metal kernel
  out = Tensor(LazyBuffer.from_buffer(O, bs + (Hq, L, Dv)))
  return out.reshape(L, Dv) if squeeze and not bs else out

def paged_attention(x: Any, pool: Any, meta: Any, N: int, Hkv: int, G: int, hd: int, bs: int, NB: int, MB: int, xs: int, mrow: int,
                    mpos: int, mbt: int, scale: float, prefill: bool = False, np_n: int = 0, mix: int = 0, mix_exact: bool = True) -> Any:
  """Drop-in for fusezap.serve.kv_cache.paged_attention on CUDA Buffers: token t (block-table row meta[mrow+t], position
  meta[mpos+t]; row < 0 is padding and gives zeros) attends to positions 0..pos of its sequence, read from the pool
  (2, NB, Hkv, bs, hd) through block table meta[mbt + row*MB:]. x holds the step's q (row t, heads 0..Hkv*G-1, row stride xs).
  prefill: the rows are prompt chunks (runs of consecutive positions), so the tensor-core paged prefill shares each K/V
  block across a run's rows (or, where it does not apply, the split-K kernel); otherwise (decode) the tensor-core GQA
  decode (f16/bf16, hd 64 / 128; FZ_PAGED_MMA=0 disables it), else the one-pass decode kernel, splits each row's keys
  into fixed-size partitions, as many as np_n rows allow (default N; see decode_calls). FZ_PAGED_DECODE=0 (with
  FZ_PAGED_MMA=0) runs decode on the split-K kernel.
  mix (a mixed forward, LLMEngine): rows 0 .. mix-1 are prompt chunks (paged prefill kernel), rows mix .. N-1 decode rows,
  which take exactly the decode forwards' kernel and partitions (np_n), so their output is bit-identical to a decode step.
  mix_exact=False (throughput mode): the decode rows may use any kernel; where no decode kernel takes a row offset (e.g.
  hd 64 without the mma kernel) every row runs the paged prefill kernel (a decode row is a one-token chunk)."""
  from fusezap.device import Buffer, Device
  out = Buffer(x.device, N * Hkv * G * hd, x.dtype)
  if not (N and Hkv): return out
  if mix and not mix_exact and not (kpt_ok(x.dtype, hd, G, xs, np_n * Hkv * G * -(-MB * bs // KPT_PART)) or mdec_ok(Device[x.device], x.dtype, hd, G, xs)):
    _run(paged_flash_call(x.dtype, N, Hkv, G, hd, bs, NB, MB, xs, mrow, mpos, mbt, scale), (out, x, pool, meta))
    return out
  if mix:
    nd = N - mix
    kpt = kpt_ok(x.dtype, hd, G, xs, np_n * Hkv * G * -(-MB * bs // KPT_PART))
    assert use_paged_flash(Device[x.device], x.dtype, hd, xs) and (kpt or mdec_ok(Device[x.device], x.dtype, hd, G, xs)) and np_n and 0 < mix < N
    _run(paged_flash_call(x.dtype, mix, Hkv, G, hd, bs, NB, MB, xs, mrow, mpos, mbt, scale), (out, x, pool, meta))   # rows < mix
    # rows >= mix: exactly the decode forwards' kernel (the same choice, partitions and np_n), at a row offset
    if tcd_ok(Device[x.device], x.dtype, hd, G, xs) and kpt: a, c = tcd_calls(x.dtype, nd, Hkv, G, hd, bs, NB, MB, xs, mrow, mpos, mbt, scale, np_n=np_n, row0=mix)
    elif kpt: a, c = kpt_calls(x.dtype, nd, Hkv, G, hd, bs, NB, MB, xs, mrow, mpos, mbt, scale, np_n=np_n, row0=mix)
    else: a, c = mdec_calls(x.dtype, nd, Hkv, G, hd, bs, NB, MB, xs, mrow, mpos, mbt, scale, np_n=np_n, row0=mix)
    NP = call_np(a)
    acc, ml = Buffer(x.device, nd * Hkv * G * NP * hd, dtypes.float32), Buffer(x.device, nd * Hkv * G * NP * 2, dtypes.float32)
    _run(a, (acc, ml, x, pool, meta), 2)
    _run(c, (out, acc, ml, meta))   # writes rows mix .. N-1 of out; the paged prefill wrote rows 0 .. mix-1
    return out
  if prefill and use_paged_flash(Device[x.device], x.dtype, hd, xs):
    _run(paged_flash_call(x.dtype, N, Hkv, G, hd, bs, NB, MB, xs, mrow, mpos, mbt, scale), (out, x, pool, meta))
  elif not prefill and tcd_ok(Device[x.device], x.dtype, hd, G, xs) and kpt_ok(x.dtype, hd, G, xs, (np_n or N) * Hkv * G * -(-MB * bs // TCD_PART), tc=True):
    a, c = tcd_calls(x.dtype, N, Hkv, G, hd, bs, NB, MB, xs, mrow, mpos, mbt, scale, np_n=np_n)
    NP = call_np(a)
    acc, ml = Buffer(x.device, N * Hkv * G * NP * hd, dtypes.float32), Buffer(x.device, N * Hkv * G * NP * 2, dtypes.float32)
    _run(a, (acc, ml, x, pool, meta), 2)
    _run(c, (out, acc, ml, meta))
  elif not prefill and kpt_ok(x.dtype, hd, G, xs, (np_n or N) * Hkv * G * -(-MB * bs // KPT_PART)):
    a, c = kpt_calls(x.dtype, N, Hkv, G, hd, bs, NB, MB, xs, mrow, mpos, mbt, scale, np_n=np_n)
    NP = call_np(a)
    acc, ml = Buffer(x.device, N * Hkv * G * NP * hd, dtypes.float32), Buffer(x.device, N * Hkv * G * NP * 2, dtypes.float32)
    _run(a, (acc, ml, x, pool, meta), 2)
    _run(c, (out, acc, ml, meta))
  elif not prefill and mdec_ok(Device[x.device], x.dtype, hd, G, xs):
    a, c = mdec_calls(x.dtype, N, Hkv, G, hd, bs, NB, MB, xs, mrow, mpos, mbt, scale, np_n=np_n)
    NP = call_np(a)
    acc, ml = Buffer(x.device, N * Hkv * G * NP * hd, dtypes.float32), Buffer(x.device, N * Hkv * G * NP * 2, dtypes.float32)
    _run(a, (acc, ml, x, pool, meta), 2)
    _run(c, (out, acc, ml, meta))
  elif split_ok(hd, G, xs):
    if not prefill and decode_ok(hd, G, xs): a, c = decode_calls(x.dtype, N, Hkv, G, hd, bs, NB, MB, xs, mrow, mpos, mbt, scale, np_n=np_n)
    else: a, c = split_calls(x.dtype, N, Hkv, G, hd, bs, NB, MB, xs, mrow, mpos, mbt, scale, np_n=np_n)
    NP = a.params[-1]
    acc, ml = Buffer(x.device, N * Hkv * G * NP * hd, dtypes.float32), Buffer(x.device, N * Hkv * G * NP * 2, dtypes.float32)
    _run(a, (acc, ml, x, pool, meta), 2)
    _run(c, (out, acc, ml, meta))
  else: _run(paged_call(x.dtype, N, Hkv, G, hd, bs, NB, MB, xs, mrow, mpos, mbt, scale), (out, x, pool, meta))
  return out

# ---------------------------------------------------------------- the scheduler hook (P2-CUSTOM)

HOOK_NAMES = ("fz_attention", "fz_attention_causal")

def hook_call(dev: Any, name: str, params: Sequence[int], dt: DType) -> AttnCall:
  """The SIMT launch for a scheduler-matched sdpa item (fusezap.schedule._match_attention params)."""
  Bt, L, S, D, Dv, sqb, sql, sqd, skb, sks, skd, svb, svs, svd, sob, sol, sod, oq, ok, ov, scale, log2e = params
  W, causal = 4, name.endswith("causal")
  nm = f"fz_attn_simt_{_tag(dt)}_d{D}_{Dv}_w{W}{'_causal' if causal else ''}"
  vals = (Bt, 1, 1, L, S, sqb, 0, sql, sqd, skb, 0, sks, skd, svb, 0, svs, svd, sob, 0, sol, sod, oq, ok, ov, 0, scale, log2e)
  return AttnCall(nm, render_simt(nm, dt, D, Dv, W, causal), simt_smem(D, Dv, W), vals, (-(-L // W) * Bt, 1, 1), (32 * W, 1, 1))

def hook_fits(dev: Any, params: Sequence[int], W: int = 4) -> bool:
  """The SIMT kernel's shared memory for a scheduler match (fusezap.schedule._match_attention params: D, Dv at 3, 4)
  fits the device's opt-in limit: the VAE's head dim 512 needs 139392 bytes, over an RTX 4090's 101376 (sm_89), so
  there the sdpa stays composed (its GEMMs still take the matmul hook)."""
  if dev is None: return True
  try: return simt_smem(int(params[3]), int(params[4]), W) <= max_smem(dev)
  except Exception: return True   # no device to ask (tests): the launch-time check still stands

def custom_kernels(dev: Any = None) -> dict[str, Any]:
  from fusezap.schedule import ATTENTION, CustomKernel
  return {"attention": CustomKernel("fz_attention", {"CUDA": render_simt}, ATTENTION.signature, 4, accept=lambda p: hook_fits(dev, p))}

def install(dev: Any) -> None:
  """Offer "attention" to the scheduler hook on this CUDA device and run those items with the SIMT kernel. Idempotent;
  other kind="custom" items (P3-GEMM's, run_custom's) go to the device's previous `run`."""
  dev.custom_kernels = {**(getattr(dev, "custom_kernels", None) or {}), **custom_kernels(dev)}
  if getattr(dev.run, "_fz_attn", False): return
  orig = dev.run
  def run(si: Any) -> float | None:
    if si.kind == "custom" and si.ast and getattr(c := si.ast[0], "kernel", None) is not None and getattr(c.kernel, "name", None) in HOOK_NAMES \
       and not hasattr(c, "launch"):
      return hook_call(dev, c.kernel.name, c.params, si.bufs[1].dtype).launch(dev, si.bufs)
    return orig(si)
  run._fz_attn = True  # type: ignore[attr-defined]
  dev.run = run
