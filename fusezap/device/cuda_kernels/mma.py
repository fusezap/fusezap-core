"""mma.sync tensor-core GEMM (sm_80+; fp8 e4m3 on sm_89+): C[b] = alpha * A[b] @ op(B[b]), f32 accumulate.

  A (M, K) row-major (K-major), B either (N, K) row-major (K-major, "nt": the nn.Linear weight layout) or (K, N) row-major
  ("nn", 16-bit only: ldmatrix.trans has no 8-bit form). Inputs are f16, bf16 or e4m3 (uint8 bits); C is f32/f16/bf16.
  Block tile BM x BN, K step BKB bytes (BKB / esize elements), S-stage cp.async pipeline into padded shared memory
  (row pitch + 16 bytes: conflict-free ldmatrix without a swizzle), warps WM x WN, ldmatrix.x4 fragments.
  Out-of-range rows and K chunks are zero-filled by cp.async (src-size 0), so any M, N work; K * esize (and for "nn"
  N * esize) must be a multiple of 16 and the bases 16-byte aligned. Stores are guarded; pairs when ldc/scb are even.
Everything is in bytes, so one generator serves 16-bit (m16n8k16) and fp8 (m16n8k32): both consume 32 bytes of K per mma
and their A/B fragments have the same byte layout. fp8 MMAs accumulate with reduced precision (~14 bits kept), so for fp8
each stage sums into a fresh accumulator set that is then added into the f32 accumulators (promotion every BKB elements).
"""
from __future__ import annotations
from dataclasses import dataclass
from fusezap.device.cuda_kernels._rt import COMMON

ESIZE = {"f16": 2, "bf16": 2, "e4m3": 1}
OUT_C = {"f32": "float", "f16": "u16", "bf16": "u16"}

@dataclass(frozen=True)
class MmaCfg:
  BM: int = 128
  BN: int = 128
  WM: int = 2
  WN: int = 2
  BKB: int = 64    # K bytes per stage
  S: int = 4       # pipeline stages
  SK: int = 1      # K splits (split-K: grid z; the last CTA of a tile sums the f32 partials in split order)
  @property
  def threads(self) -> int: return 32 * self.WM * self.WN

def smem_bytes(cfg: MmaCfg, kind: str, nn: bool) -> int:
  e = ESIZE[kind]
  b = (cfg.BKB // e) * (cfg.BN * e + 16) if nn else cfg.BN * (cfg.BKB + 16)
  return cfg.S * (cfg.BM * (cfg.BKB + 16) + b)

def valid(cfg: MmaCfg, kind: str, nn: bool) -> bool:
  WTM, WTN = cfg.BM // cfg.WM, cfg.BN // cfg.WN
  if cfg.BM % (16 * cfg.WM) or cfg.BN % (16 * cfg.WN) or cfg.BKB % 32 or (nn and kind == "e4m3"): return False
  ch_a, ch_b = cfg.BM * cfg.BKB // 16, (cfg.BN * cfg.BKB // 16)
  acc = WTM * WTN // 32 * (2 if kind == "e4m3" else 1)   # fp8 keeps a second (per-stage) accumulator set
  return ch_a % cfg.threads == 0 and ch_b % cfg.threads == 0 and acc <= 128 and cfg.S >= 2 and 1 <= cfg.SK <= 64

def name_of(kind: str, out: str, nn: bool, cfg: MmaCfg) -> str:
  return f"fz_mma_{kind}_{out}_{'nn' if nn else 'nt'}_{cfg.BM}x{cfg.BN}_w{cfg.WM}x{cfg.WN}_k{cfg.BKB}_s{cfg.S}" + (f"_sk{cfg.SK}" if cfg.SK > 1 else "")

def render(kind: str, out: str, nn: bool, cfg: MmaCfg) -> str:
  assert valid(cfg, kind, nn), (kind, nn, cfg)
  e, T = ESIZE[kind], cfg.threads
  BM, BN, BKB, S = cfg.BM, cfg.BN, cfg.BKB, cfg.S
  WTM, WTN = BM // cfg.WM, BN // cfg.WN
  MT, NT = WTM // 16, WTN // 8
  AP = BKB + 16                                  # A row pitch (bytes)
  BP = BN * e + 16 if nn else BKB + 16           # B row pitch: k rows (nn) or n rows (nt)
  BKE = BKB // e
  a_stage, b_stage = BM * AP, (BKE * BP if nn else BN * BP)
  mma = (f"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32" if kind == "e4m3" else
         f"mma.sync.aligned.m16n8k16.row.col.f32.{kind}.{kind}.f32")
  st = {"f32": "", "f16": "fz_f2h", "bf16": "fz_f2bf"}[out]
  pack = {"f16": "fz_pack_h", "bf16": "fz_pack_bf"}.get(out)
  name = name_of(kind, out, nn, cfg)
  # fp8 MMA accumulates with reduced precision: each stage (BKB elements of K) sums from 0 in tmp, then adds into acc in f32
  promote = kind == "e4m3"
  tgt = "tmp" if promote else "acc"
  tmp_init = f"""    float tmp[{MT}][{NT}][4];
    #pragma unroll
    for (int i = 0; i < {MT}; i++)
      #pragma unroll
      for (int j = 0; j < {NT}; j++) tmp[i][j][0] = tmp[i][j][1] = tmp[i][j][2] = tmp[i][j][3] = 0.f;""" if promote else ""
  tmp_add = f"""    #pragma unroll
    for (int i = 0; i < {MT}; i++)
      #pragma unroll
      for (int j = 0; j < {NT}; j++)
        #pragma unroll
        for (int h = 0; h < 4; h++) acc[i][j][h] += tmp[i][j][h];""" if promote else ""
  # B tile loads: (rows, 16-byte chunks per row, row bound, byte bound, global row pitch in bytes)
  if nn: bload = f"""  for (int c = tid; c < {BKE * BN * e // 16}; c += {T}) {{
    int r = c / {BN * e // 16}, cb = (c % {BN * e // 16}) * 16, gk = k0 + r; long long gnb = (long long)n0 * {e} + cb;
    bool ok = gk < K && gnb < (long long)N * {e};
    const u8* src = ok ? Bb + (long long)gk * ldb * {e} + gnb : Bb;
    cp16(sB + r * {BP} + cb, src, ok);
  }}"""
  else: bload = f"""  for (int c = tid; c < {BN * BKB // 16}; c += {T}) {{
    int r = c / {BKB // 16}, cb = (c % {BKB // 16}) * 16, gn = n0 + r; long long gkb = (long long)k0 * {e} + cb;
    bool ok = gn < N && gkb < (long long)K * {e};
    const u8* src = ok ? Bb + (long long)gn * ldb * {e} + gkb : Bb;
    cp16(sB + r * {BP} + cb, src, ok);
  }}"""
  if nn: bfrag = f"""        int kr = kk * 16 + (lane & 7) + 8 * ((lane >> 3) & 1), nc = wn0 + j * 16 + 8 * (lane >> 4);
        ldsm4t(fb[2 * j][0], fb[2 * j][1], fb[2 * j + 1][0], fb[2 * j + 1][1], sB + kr * {BP} + nc * 2);"""
  else: bfrag = f"""        int nr = wn0 + j * 16 + (lane & 7) + 8 * (lane >> 4), kc = kk * 32 + 16 * ((lane >> 3) & 1);
        ldsm4(fb[2 * j][0], fb[2 * j][1], fb[2 * j + 1][0], fb[2 * j + 1][1], sB + nr * {BP} + kc);"""
  ostore = (f"""          if (vec && c + 1 < N) *(float2*)(Cb + (long long)r * ldc + c) = make_float2(v0, v1);
          else {{ if (c < N) Cb[(long long)r * ldc + c] = v0; if (c + 1 < N) Cb[(long long)r * ldc + c + 1] = v1; }}""" if out == "f32" else
            f"""          if (vec && c + 1 < N) *(u32*)(Cb + (long long)r * ldc + c) = {pack}(v0, v1);
          else {{ if (c < N) Cb[(long long)r * ldc + c] = {st}(v0); if (c + 1 < N) Cb[(long long)r * ldc + c + 1] = {st}(v1); }}""")
  SK = cfg.SK
  sk_params = ", float* __restrict__ W, int* __restrict__ sem" if SK > 1 else ""
  # split-K: every CTA writes its f32 partial (valid rows only) to its slot of W, the last of a tile's SK CTAs to arrive
  # (sem counter, reset by it for the next launch) sums the SK slots in split order (deterministic) and runs the epilogue
  frag = f"""  #pragma unroll
  for (int i = 0; i < {MT}; i++)
    #pragma unroll
    for (int j = 0; j < {NT}; j++)
      #pragma unroll
      for (int h = 0; h < 2; h++)
        if (m0 + wm0 + i * 16 + (lane >> 2) + 8 * h < M) {{
          float* p = WS + (((i * {NT} + j) * 2 + h) * {T} + tid) * 2;
          BODY
        }}"""
  put = frag.replace("BODY", "*(float2*)p = make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);")
  get = frag.replace("float* p", "const float* p").replace("BODY", 'float x, y; asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(x), "=f"(y) : "l"(p)); '
                                                           "acc[i][j][2 * h] += x; acc[i][j][2 * h + 1] += y;")
  fixup = f"""  {{
    __shared__ int fz_last;
    const long long ntl = (long long)tiles_m * tiles_n, slot = {BM * BN}LL;
    {{ float* WS = W + ((bz * {SK} + (long long)blockIdx.z) * ntl + pid) * slot;
{put}
    }}
    __threadfence();
    __syncthreads();
    if (tid == 0) {{ int* c = sem + bz * ntl + pid; fz_last = atomicAdd(c, 1) == {SK - 1}; if (fz_last) *c = 0; }}
    __syncthreads();
    if (!fz_last) return;
    __threadfence();
    #pragma unroll
    for (int i = 0; i < {MT}; i++)
      #pragma unroll
      for (int j = 0; j < {NT}; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
    for (int s = 0; s < {SK}; s++) {{ const float* WS = W + ((bz * {SK} + (long long)s) * ntl + pid) * slot;
{get}
    }}
  }}""" if SK > 1 else ""
  return COMMON + f"""
static __device__ __forceinline__ void cp16(u8* dst, const u8* src, bool ok) {{
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(fz_smem(dst)), "l"(src), "r"(ok ? 16 : 0) : "memory");
}}
static __device__ __forceinline__ void ldsm4(u32& a, u32& b, u32& c, u32& d, const u8* p) {{
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0,%1,%2,%3}}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(fz_smem(p)));
}}
static __device__ __forceinline__ void ldsm4t(u32& a, u32& b, u32& c, u32& d, const u8* p) {{
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%0,%1,%2,%3}}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(fz_smem(p)));
}}
extern "C" __global__ void __launch_bounds__({T}) {name}({OUT_C[out]}* __restrict__ C, const u8* __restrict__ A, const u8* __restrict__ B,
    const int M, const int N, const int K, const int lda, const int ldb, const int ldc, const long long sab, const long long sbb,
    const long long scb, const float alpha, const float* __restrict__ sa, const float* __restrict__ sb{sk_params}) {{
  extern __shared__ __align__(128) u8 smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int tiles_m = (M + {BM - 1}) / {BM}, tiles_n = (N + {BN - 1}) / {BN};
  const int pid = blockIdx.x, npg = 8 * tiles_n, g = pid / npg, fm = g * 8, gsz = min(tiles_m - fm, 8);
  const int tm = fm + (pid % npg) % gsz, tn = (pid % npg) / gsz;
  if (tm >= tiles_m) return;
  const int m0 = tm * {BM}, n0 = tn * {BN}, bz = blockIdx.y;
  const u8* Ab = A + bz * sab * {e};
  const u8* Bb = B + bz * sbb * {e};
  {OUT_C[out]}* Cb = C + bz * scb;
  const int wm0 = (warp / {cfg.WN}) * {WTM}, wn0 = (warp % {cfg.WN}) * {WTN};
  const int nk_all = (K * {e} + {BKB - 1}) / {BKB};
  const int per = (nk_all + {SK - 1}) / {SK}, kb = {"blockIdx.z * per" if SK > 1 else "0"}, nk = max(0, min(nk_all - kb, per));
  float acc[{MT}][{NT}][4];
  #pragma unroll
  for (int i = 0; i < {MT}; i++)
    #pragma unroll
    for (int j = 0; j < {NT}; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  auto load = [&](int kt) {{
    u8* sA = smem + (kt % {S}) * {a_stage};
    u8* sB = smem + {S * a_stage} + (kt % {S}) * {b_stage};
    const int k0 = (kb + kt) * {BKE};
    for (int c = tid; c < {BM * BKB // 16}; c += {T}) {{
      int r = c / {BKB // 16}, cb = (c % {BKB // 16}) * 16, gm = m0 + r; long long gkb = (long long)k0 * {e} + cb;
      bool ok = gm < M && gkb < (long long)K * {e};
      const u8* src = ok ? Ab + (long long)gm * lda * {e} + gkb : Ab;
      cp16(sA + r * {AP} + cb, src, ok);
    }}
  {bload}
  }};
  #pragma unroll
  for (int s = 0; s < {S - 1}; s++) {{ if (s < nk) load(s); asm volatile("cp.async.commit_group;"); }}
  for (int kt = 0; kt < nk; kt++) {{
    asm volatile("cp.async.wait_group {S - 2};" ::: "memory");
    __syncthreads();
    if (kt + {S - 1} < nk) load(kt + {S - 1});
    asm volatile("cp.async.commit_group;");
    const u8* sA = smem + (kt % {S}) * {a_stage};
    const u8* sB = smem + {S * a_stage} + (kt % {S}) * {b_stage};
{tmp_init}
    #pragma unroll
    for (int kk = 0; kk < {BKB // 32}; kk++) {{
      u32 fa[{MT}][4], fb[{NT}][2];
      #pragma unroll
      for (int i = 0; i < {MT}; i++)
        ldsm4(fa[i][0], fa[i][1], fa[i][2], fa[i][3], sA + (wm0 + i * 16 + (lane & 15)) * {AP} + kk * 32 + (lane >> 4) * 16);
      #pragma unroll
      for (int j = 0; j < {NT // 2}; j++) {{
{bfrag}
      }}
      #pragma unroll
      for (int i = 0; i < {MT}; i++)
        #pragma unroll
        for (int j = 0; j < {NT}; j++)
          asm volatile("{mma} {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"
            : "+f"({tgt}[i][j][0]), "+f"({tgt}[i][j][1]), "+f"({tgt}[i][j][2]), "+f"({tgt}[i][j][3])
            : "r"(fa[i][0]), "r"(fa[i][1]), "r"(fa[i][2]), "r"(fa[i][3]), "r"(fb[j][0]), "r"(fb[j][1]));
    }}
{tmp_add}
  }}
  asm volatile("cp.async.wait_group 0;" ::: "memory");
{fixup}
  const bool vec = ((ldc | scb) & 1) == 0;
  const float scale = alpha * (sa ? sa[0] : 1.f) * (sb ? sb[0] : 1.f);
  #pragma unroll
  for (int i = 0; i < {MT}; i++)
    #pragma unroll
    for (int j = 0; j < {NT}; j++)
      #pragma unroll
      for (int h = 0; h < 2; h++) {{
        const int r = m0 + wm0 + i * 16 + (lane >> 2) + 8 * h, c = n0 + wn0 + j * 8 + 2 * (lane & 3);
        if (r < M) {{
          const float v0 = scale * acc[i][j][2 * h], v1 = scale * acc[i][j][2 * h + 1];
{ostore}
        }}
      }}
}}
"""

def launch_dims(cfg: MmaCfg, M: int, N: int, Bt: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
  return (-(-M // cfg.BM) * -(-N // cfg.BN), Bt, cfg.SK), (cfg.threads, 1, 1)

def splitk_ws(cfg: MmaCfg, M: int, N: int, Bt: int) -> tuple[int, int]:
  """(f32 workspace elements, int32 counters) a split-K launch needs; (0, 0) for SK == 1."""
  if cfg.SK == 1: return 0, 0
  ntl = -(-M // cfg.BM) * -(-N // cfg.BN) * Bt
  return ntl * cfg.SK * cfg.BM * cfg.BN, ntl
