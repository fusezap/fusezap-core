"""float32 GEMM on tensor cores via 3xTF32 (sm_80+): the kernel behind the scheduler's "matmul" hook.

The hook (fusezap.schedule._mm_spec) hands over float32 operands with arbitrary element strides and offsets:
params (Bt, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, oa, ob). Each value x splits into big = tf32(x) and
small = tf32(x - big); a_s*b_b + a_b*b_s + a_b*b_b with mma.sync m16n8k8 keeps about fp32 accuracy (the dropped a_s*b_s
term is ~2^-22 relative). The MMA accumulator truncates, which biases long sums, so each 16-wide K slice accumulates from 0
and is then added into the f32 accumulators (round to nearest): the result matches the composed f32 matmul at the f32
matmul tolerance, on tensor cores. Non-finite values: small = 0 and the cross terms see them as 0 (no inf * 0 = nan
that fp32 would not produce); a_b * b_b carries inf/nan like fp32.
Tiles: BM x BN x 16, 256 threads (warps 2 x 4), register-staged global loads into double-buffered padded shared memory.
The load mapping is picked per operand at render time ("k": K has unit stride, threads walk K; "r": the row axis
(M for A, N for B) has unit stride; anything else uses "k"), so loads coalesce for every common layout.
"""
from __future__ import annotations
from fusezap.device.cuda_kernels._rt import COMMON

BM, BN, BK, T = 128, 128, 16, 256
WM, WN = 2, 4
PARAMS = ("Bt", "M", "N", "K", "sab", "sam", "sak", "sbb", "sbk", "sbn", "scb", "scm", "scn", "oa", "ob")

def layout(sr: int, sk: int) -> str: return "r" if sr == 1 and sk != 1 else "k"

def name_of(la: str, lb: str) -> str: return f"fz_tf32x3_{la}{lb}"

def render(la: str, lb: str) -> str:
  WTM, WTN = BM // WM, BN // WN
  MT, NT = WTM // 16, WTN // 8
  P = BK + 4
  def ld(op: str, lay: str, R: int, rows: str, sr: str, sk: str) -> str:
    n = R * BK // T
    idx = "int kk = e % 16, rr = e / 16;" if lay == "k" else f"int rr = e % {R}, kk = e / {R};"
    return f"""  #pragma unroll
  for (int p = 0; p < {n}; p++) {{
    int e = tid + {T} * p; {idx}
    int gr = {op}0 + rr, gk = k0 + kk;
    r{op}[p] = (gr < {rows} && gk < K) ? {op.upper()}b[(long long)gr * {sr} + (long long)gk * {sk}] : 0.f;
  }}"""
  def st(op: str, lay: str, R: int) -> str:
    idx = "int kk = e % 16, rr = e / 16;" if lay == "k" else f"int rr = e % {R}, kk = e / {R};"
    return f"""  #pragma unroll
  for (int p = 0; p < {R * BK // T}; p++) {{ int e = tid + {T} * p; {idx} s{op.upper()}[buf][rr][kk] = r{op}[p]; }}"""
  return COMMON + f"""
// x -> big = tf32(x), small = tf32(x - big), bigx = big where finite else 0 (the operand of the cross terms, so an inf
// meets the other side's small part as 0 * small, not inf * 0 = nan; a_big * b_big still carries inf and nan)
static __device__ __forceinline__ void split(float x, u32& big, u32& small, u32& bigx) {{
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(big) : "f"(x));
  const float b = __uint_as_float(big);
  const bool fin = fabsf(b) < __int_as_float(0x7f800000);
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(small) : "f"(fin ? x - b : 0.f));
  bigx = fin ? big : 0u;
}}
#define MMA(c, a0, a1, a2, a3, b0, b1) asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};" \\
  : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
extern "C" __global__ void __launch_bounds__({T}) {name_of(la, lb)}(float* __restrict__ C, const float* __restrict__ A, const float* __restrict__ B,
    const int Bt, const int M, const int N, const int K, const int sab, const int sam, const int sak, const int sbb, const int sbk,
    const int sbn, const int scb, const int scm, const int scn, const int oa, const int ob) {{
  __shared__ __align__(16) float sA[2][{BM}][{P}];
  __shared__ __align__(16) float sB[2][{BN}][{P}];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int tiles_m = (M + {BM - 1}) / {BM}, tiles_n = (N + {BN - 1}) / {BN};
  const int pid = blockIdx.x, npg = 8 * tiles_n, gq = pid / npg, fm = gq * 8, gsz = min(tiles_m - fm, 8);
  const int tm = fm + (pid % npg) % gsz, tn = (pid % npg) / gsz;
  if (tm >= tiles_m) return;
  const int a0 = tm * {BM}, b0 = tn * {BN}, bz = blockIdx.y;
  const float* Ab = A + oa + (long long)bz * sab;
  const float* Bb = B + ob + (long long)bz * sbb;
  float* Cb = C + (long long)bz * scb;
  const int wm0 = (warp / {WN}) * {WTM}, wn0 = (warp % {WN}) * {WTN};
  const int nk = (K + {BK - 1}) / {BK};
  float acc[{MT}][{NT}][4];
  #pragma unroll
  for (int i = 0; i < {MT}; i++)
    #pragma unroll
    for (int j = 0; j < {NT}; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  float ra[{BM * BK // T}], rb[{BN * BK // T}];
  int k0 = 0, buf = 0;
{ld("a", la, BM, "M", "sam", "sak")}
{ld("b", lb, BN, "N", "sbn", "sbk")}
{st("a", la, BM)}
{st("b", lb, BN)}
  __syncthreads();
  for (int kt = 0; kt < nk; kt++) {{
    const bool more = kt + 1 < nk;
    if (more) {{
      k0 = (kt + 1) * {BK};
{ld("a", la, BM, "M", "sam", "sak")}
{ld("b", lb, BN, "N", "sbn", "sbk")}
    }}
    float tmp[{MT}][{NT}][4];   // this K slice, added into acc with round-to-nearest (the MMA accumulator truncates)
    #pragma unroll
    for (int i = 0; i < {MT}; i++)
      #pragma unroll
      for (int j = 0; j < {NT}; j++) tmp[i][j][0] = tmp[i][j][1] = tmp[i][j][2] = tmp[i][j][3] = 0.f;
    #pragma unroll
    for (int ks = 0; ks < {BK}; ks += 8) {{
      u32 bb[{NT}][2], bs[{NT}][2], bx[{NT}][2];
      #pragma unroll
      for (int j = 0; j < {NT}; j++) {{
        const int n = wn0 + j * 8 + g;
        split(sB[buf][n][ks + t], bb[j][0], bs[j][0], bx[j][0]); split(sB[buf][n][ks + t + 4], bb[j][1], bs[j][1], bx[j][1]);
      }}
      #pragma unroll
      for (int i = 0; i < {MT}; i++) {{
        u32 ab[4], as_[4], ax[4];
        const int r = wm0 + i * 16 + g;
        split(sA[buf][r][ks + t], ab[0], as_[0], ax[0]); split(sA[buf][r + 8][ks + t], ab[1], as_[1], ax[1]);
        split(sA[buf][r][ks + t + 4], ab[2], as_[2], ax[2]); split(sA[buf][r + 8][ks + t + 4], ab[3], as_[3], ax[3]);
        #pragma unroll
        for (int j = 0; j < {NT}; j++) {{
          MMA(tmp[i][j], as_[0], as_[1], as_[2], as_[3], bx[j][0], bx[j][1]);
          MMA(tmp[i][j], ax[0], ax[1], ax[2], ax[3], bs[j][0], bs[j][1]);
          MMA(tmp[i][j], ab[0], ab[1], ab[2], ab[3], bb[j][0], bb[j][1]);
        }}
      }}
    }}
    #pragma unroll
    for (int i = 0; i < {MT}; i++)
      #pragma unroll
      for (int j = 0; j < {NT}; j++)
        #pragma unroll
        for (int h = 0; h < 4; h++) acc[i][j][h] += tmp[i][j][h];
    if (more) {{
      buf ^= 1;
{st("a", la, BM)}
{st("b", lb, BN)}
    }}
    __syncthreads();
  }}
  #pragma unroll
  for (int i = 0; i < {MT}; i++)
    #pragma unroll
    for (int j = 0; j < {NT}; j++)
      #pragma unroll
      for (int h = 0; h < 4; h++) {{
        const int r = a0 + wm0 + i * 16 + g + 8 * (h >> 1), c = b0 + wn0 + j * 8 + 2 * t + (h & 1);
        if (r < M && c < N) Cb[(long long)r * scm + (long long)c * scn] = acc[i][j][h];
      }}
}}
"""

def launch_dims(M: int, N: int, Bt: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
  return (-(-M // BM) * -(-N // BN), Bt, 1), (T, 1, 1)
