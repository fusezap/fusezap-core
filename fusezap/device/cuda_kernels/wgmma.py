"""Hopper (sm_90a) GEMM: TMA loads into 128B-swizzled shared memory, mbarrier pipeline, warp-specialized warpgroups,
wgmma.mma_async with f32 accumulators. C[b] = alpha * A[b] @ op(B[b]).

  A (M, K) row-major (K-major). B (N, K) row-major ("nt", K-major) or (K, N) row-major ("nn", MN-major: the descriptor's
  transpose bit; 16-bit only, wgmma has no transposed fp8 operand). Inputs f16 / bf16 / e4m3 (uint8 bits), C f32/f16/bf16.
  One block = 1 producer warpgroup (one thread issues the TMA loads of each stage) + NC consumer warpgroups, each owning
  64 rows x BN columns (m64nBNk16, or k32 for fp8). A stage is 128 bytes of K: BM x 128 B of A, BN x 128 B of B.
  full[s] completes when a stage's bytes land (expect_tx); empty[s] when every consumer has finished reading it.
  TMA zero-fills out-of-range rows/columns, so any M, N and K work as long as the row pitch is a multiple of 16 bytes.
  With NC = 2 the producer gives registers to the consumers (setmaxnreg), which hold 64 x BN / 128 accumulators each.
  fp8 accumulates with reduced precision in the tensor cores, so each stage's 128 K elements go into a second accumulator
  set that is added into the f32 one (needs BN <= 128: twice the registers).
"""
from __future__ import annotations
from dataclasses import dataclass
from fusezap.device.cuda_kernels._rt import COMMON

ESIZE = {"f16": 2, "bf16": 2, "e4m3": 1}
OUT_C = {"f32": "float", "f16": "u16", "bf16": "u16"}

@dataclass(frozen=True)
class WgCfg:
  BN: int = 256
  NC: int = 2      # consumer warpgroups (BM = 64 * NC)
  S: int = 4       # pipeline stages
  regs: bool = True  # setmaxnreg producer/consumer split
  wait1: bool = False  # keep one wgmma batch in flight; a stage is released one iteration later
  @property
  def BM(self) -> int: return 64 * self.NC
  @property
  def threads(self) -> int: return 128 * (self.NC + 1)

def valid(cfg: WgCfg, kind: str) -> bool: return kind != "e4m3" or (cfg.BN <= 128 and not cfg.wait1)

def smem_bytes(cfg: WgCfg) -> int: return cfg.S * (cfg.BM + cfg.BN) * 128 + 1024 + 16 * cfg.S

def name_of(kind: str, out: str, nn: bool, cfg: WgCfg) -> str:
  return f"fz_wgmma_{kind}_{out}_{'nn' if nn else 'nt'}_{cfg.BM}x{cfg.BN}_s{cfg.S}{'_r' if cfg.regs else ''}{'_w1' if cfg.wait1 else ''}"

# TMA descriptor struct, mbarrier, TMA load and smem matrix descriptor helpers
HELPERS = r"""struct __align__(64) TMap { u64 v[16]; };
static __device__ __forceinline__ void mbar_init(u64* b, u32 n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(fz_smem(b)), "r"(n) : "memory"); }
static __device__ __forceinline__ void mbar_expect(u64* b, u32 bytes) {
  asm volatile("{ .reg .b64 st; mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1; }" :: "r"(fz_smem(b)), "r"(bytes) : "memory");
}
static __device__ __forceinline__ void mbar_arrive(u64* b) {
  asm volatile("{ .reg .b64 st; mbarrier.arrive.shared::cta.b64 st, [%0]; }" :: "r"(fz_smem(b)) : "memory");
}
static __device__ __forceinline__ void mbar_wait(u64* b, u32 parity) {
  u32 ok = 0, a = fz_smem(b);
  do asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(a), "r"(parity) : "memory");
  while (!ok);
}
static __device__ __forceinline__ void tma3(u8* dst, const TMap* tm, u64* bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
    :: "r"(fz_smem(dst)), "l"((u64)tm), "r"(c0), "r"(c1), "r"(c2), "r"(fz_smem(bar)) : "memory");
}
static __device__ __forceinline__ u64 desc(u32 a, u32 lbo, u32 sbo) {
  return (u64)((a & 0x3FFFF) >> 4) | ((u64)(lbo >> 4) << 16) | ((u64)(sbo >> 4) << 32) | (1ull << 62);
}
"""

def _wgmma(kind: str, BN: int, nn: bool) -> str:
  n = BN // 2
  regs = ",".join(f"%{i}" for i in range(n))
  if kind == "e4m3": ins, tail = f"wgmma.mma_async.sync.aligned.m64n{BN}k32.f32.e4m3.e4m3", "p, 1, 1"
  else: ins, tail = f"wgmma.mma_async.sync.aligned.m64n{BN}k16.f32.{kind}.{kind}", f"p, 1, 1, 0, {1 if nn else 0}"
  outs = ", ".join(f'"+f"(d[{i}])' for i in range(n))
  return f"""static __device__ __forceinline__ void wgmma(float* d, u64 da, u64 db, int sd) {{
  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{n + 2}, 0;\\n{ins} {{{regs}}}, %{n}, %{n + 1}, {tail};\\n}}\\n"
    : {outs} : "l"(da), "l"(db), "r"(sd));
}}"""

def render(kind: str, out: str, nn: bool, cfg: WgCfg) -> str:
  assert not (nn and kind == "e4m3"), "wgmma has no MN-major fp8 operand"
  assert valid(cfg, kind), f"{cfg} is not valid for {kind}"
  e, BM, BN, S, NC = ESIZE[kind], cfg.BM, cfg.BN, cfg.S, cfg.NC
  BKE = 128 // e                     # K elements per stage (128 bytes)
  ND = BN // 2                       # accumulators per consumer thread
  st = {"f32": "", "f16": "fz_f2h", "bf16": "fz_f2bf"}[out]
  pack = {"f16": "fz_pack_h", "bf16": "fz_pack_bf"}.get(out)
  name = name_of(kind, out, nn, cfg)
  if nn:
    bload = f"""        #pragma unroll
        for (int c = 0; c < {BN // 64}; c++) tma3(sB + s * {BN * 128} + c * {BKE * 128}, &tmB, full + s, n0 + 64 * c, kt * {BKE}, bb);"""
    bdesc = f"desc(fz_smem(sB + s * {BN * 128} + kk * 2048), {BKE * 128}, 1024)"
  else:
    bload = f"        tma3(sB + s * {BN * 128}, &tmB, full + s, kt * {BKE}, n0, bb);"
    bdesc = f"desc(fz_smem(sB + s * {BN * 128} + kk * 32), 16, 1024)"
  ostore = (f"""          if (vec && c + 1 < N) *(float2*)(Cb + (long long)r * ldc + c) = make_float2(v0, v1);
          else {{ if (c < N) Cb[(long long)r * ldc + c] = v0; if (c + 1 < N) Cb[(long long)r * ldc + c + 1] = v1; }}""" if out == "f32" else
            f"""          if (vec && c + 1 < N) *(u32*)(Cb + (long long)r * ldc + c) = {pack}(v0, v1);
          else {{ if (c < N) Cb[(long long)r * ldc + c] = {st}(v0); if (c + 1 < N) Cb[(long long)r * ldc + c + 1] = {st}(v1); }}""")
  # fp8 wgmma accumulates with reduced precision: each stage (128 K elements) starts a fresh set (scale-d 0 on its first
  # k step) that is added into d in f32
  promote = kind == "e4m3"
  tgt, sd, tmpdecl = ("dt", "kk > 0", f", dt[{ND}] = {{}}") if promote else ("d", "1", "")
  add = f"""      #pragma unroll
      for (int i = 0; i < {ND}; i++) d[i] += dt[i];""" if promote else ""
  dec = "  asm volatile(\"setmaxnreg.dec.sync.aligned.u32 40;\");" if cfg.regs else ""
  inc = "  asm volatile(\"setmaxnreg.inc.sync.aligned.u32 232;\");" if cfg.regs else ""
  return COMMON + HELPERS + f"""{_wgmma(kind, BN, nn)}
extern "C" __global__ void __launch_bounds__({cfg.threads}, 1) {name}({OUT_C[out]}* __restrict__ C, const __grid_constant__ TMap tmA,
    const __grid_constant__ TMap tmB, const int M, const int N, const int K, const int ldc, const long long scb, const float alpha,
    const int bta, const int btb, const float* __restrict__ sa, const float* __restrict__ sb) {{
  extern __shared__ __align__(1024) u8 smem_raw[];
  u8* smem = smem_raw + ((1024 - (fz_smem(smem_raw) & 1023)) & 1023);
  u8* sA = smem;
  u8* sB = smem + {S * BM * 128};
  u64* full = (u64*)(sB + {S * BN * 128});
  u64* empty = full + {S};
  const int tiles_m = (M + {BM - 1}) / {BM}, tiles_n = (N + {BN - 1}) / {BN};
  const int pid = blockIdx.x, npg = 8 * tiles_n, gq = pid / npg, fm = gq * 8, gsz = min(tiles_m - fm, 8);
  const int tm = fm + (pid % npg) % gsz, tn = (pid % npg) / gsz;
  const int m0 = tm * {BM}, n0 = tn * {BN}, bz = blockIdx.y, ba = bta > 1 ? bz : 0, bb = btb > 1 ? bz : 0;
  const int wg = threadIdx.x / 128, t = threadIdx.x % 128;
  const int nk = (K + {BKE - 1}) / {BKE};
  if (threadIdx.x == 0) {{
    #pragma unroll
    for (int s = 0; s < {S}; s++) {{ mbar_init(full + s, 1); mbar_init(empty + s, {NC}); }}
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }}
  __syncthreads();
  if (wg == 0) {{
{dec}
    if (t == 0) {{
      for (int kt = 0; kt < nk; kt++) {{
        const int s = kt % {S};
        if (kt >= {S}) mbar_wait(empty + s, ((kt / {S}) - 1) & 1);
        mbar_expect(full + s, {(BM + BN) * 128});
        tma3(sA + s * {BM * 128}, &tmA, full + s, kt * {BKE}, m0, ba);
{bload}
      }}
    }}
  }} else {{
{inc}
    const int cw = wg - 1;
    float d[{ND}]{tmpdecl};
    #pragma unroll
    for (int i = 0; i < {ND}; i++) d[i] = 0.f;
    for (int kt = 0; kt < nk; kt++) {{
      const int s = kt % {S};
      mbar_wait(full + s, (kt / {S}) & 1);
      #pragma unroll
      for (int i = 0; i < {ND}; i++) asm volatile("" : "+f"(d[i]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      #pragma unroll
      for (int kk = 0; kk < 4; kk++) wgmma({tgt}, desc(fz_smem(sA + s * {BM * 128} + cw * {64 * 128} + kk * 32), 16, 1024), {bdesc}, {sd});
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned {1 if cfg.wait1 else 0};" ::: "memory");
      #pragma unroll
      for (int i = 0; i < {ND}; i++) asm volatile("" : "+f"({tgt}[i]) :: "memory");
{add}
      {"if (t == 0 && kt > 0) mbar_arrive(empty + (kt - 1) % " + str(S) + ");" if cfg.wait1 else "if (t == 0) mbar_arrive(empty + s);"}
    }}
    {"asm volatile(\"wgmma.wait_group.sync.aligned 0;\" ::: \"memory\");" if cfg.wait1 else ""}
    #pragma unroll
    for (int i = 0; i < {ND}; i++) asm volatile("" : "+f"(d[i]) :: "memory");
    {OUT_C[out]}* Cb = C + bz * scb;
    const bool vec = ((ldc | scb) & 1) == 0;
    const float scale = alpha * (sa ? sa[0] : 1.f) * (sb ? sb[0] : 1.f);
    const int w = t / 32, lane = t % 32;
    #pragma unroll
    for (int i = 0; i < {BN // 8}; i++)
      #pragma unroll
      for (int h = 0; h < 2; h++) {{
        const int r = m0 + cw * 64 + 16 * w + (lane >> 2) + 8 * h, c = n0 + 8 * i + 2 * (lane & 3);
        if (r < M) {{
          const float v0 = scale * d[4 * i + 2 * h], v1 = scale * d[4 * i + 2 * h + 1];
{ostore}
        }}
      }}
  }}
}}
"""

def launch_dims(cfg: WgCfg, M: int, N: int, Bt: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
  return (-(-M // cfg.BM) * -(-N // cfg.BN), Bt, 1), (cfg.threads, 1, 1)

# ---------------------------------------------------------------- swap-AB decode form (small M, "nt", 16-bit)

@dataclass(frozen=True)
class SwapCfg:
  """C = A @ B^T with the operands swapped: the weight B (N, K) is the wgmma A operand (64 rows per consumer warpgroup) and
  the activation A (M, K) the wgmma B operand (BX rows = the instruction's n, 8..64), so a 32-row decode batch runs m64n32k16
  instead of padding to 64 rows; C is stored transposed. Split-K over grid z as in mma.py."""
  NC: int = 1      # consumer warpgroups (BW = 64 * NC weight rows per CTA)
  BX: int = 32     # activation rows per CTA (multiple of 8)
  S: int = 8       # pipeline stages of 128 K bytes
  SK: int = 1      # K splits (the last CTA of a tile sums the f32 partials in split order)
  @property
  def BW(self) -> int: return 64 * self.NC
  @property
  def threads(self) -> int: return 128 * (self.NC + 1)

def swap_valid(cfg: SwapCfg, kind: str) -> bool:
  return kind in ("f16", "bf16") and cfg.BX % 8 == 0 and 8 <= cfg.BX <= 64 and cfg.S >= 2 and 1 <= cfg.SK <= 64
def swap_smem_bytes(cfg: SwapCfg) -> int: return cfg.S * (cfg.BW + cfg.BX) * 128 + 1024 + 16 * cfg.S
def swap_name(kind: str, out: str, cfg: SwapCfg) -> str:
  return f"fz_wgswap_{kind}_{out}_{cfg.BW}x{cfg.BX}_s{cfg.S}" + (f"_sk{cfg.SK}" if cfg.SK > 1 else "")

def render_swap(kind: str, out: str, cfg: SwapCfg) -> str:
  """C[b] (M, N) = alpha * A[b] (M, K) @ B[b] (N, K)^T; tmW maps B (box 64 x BW), tmX maps A (box 64 x BX)."""
  assert swap_valid(cfg, kind), (kind, cfg)
  BW, BX, S, NC, SK = cfg.BW, cfg.BX, cfg.S, cfg.NC, cfg.SK
  BKE, ND = 64, BX // 2
  st = {"f32": "", "f16": "fz_f2h", "bf16": "fz_f2bf"}[out]
  sk_params = ", float* __restrict__ Wk, int* __restrict__ sem" if SK > 1 else ""
  # split-K: each consumer thread writes its ND partials (float4s) to its CTA's slot; the last of a tile's SK CTAs to arrive
  # (sem counter, reset by it for the next launch) sums the SK slots in split order (deterministic) and runs the epilogue
  slot = lambda z: f"(const float4*)(Wk + ((bz * {SK} + (long long){z}) * ntl + pid) * {BW * BX}LL) + (cw * 128 + t) * {ND // 4}"
  fixup = f"""  {{
    __shared__ int fz_last;
    const long long ntl = (long long)gridDim.x;
    if (wg > 0) {{
      float4* p = (float4*){slot("blockIdx.z")};
      #pragma unroll
      for (int i = 0; i < {ND // 4}; i++) p[i] = make_float4(d[4 * i], d[4 * i + 1], d[4 * i + 2], d[4 * i + 3]);
    }}
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {{ int* c = sem + bz * ntl + pid; fz_last = atomicAdd(c, 1) == {SK - 1}; if (fz_last) *c = 0; }}
    __syncthreads();
    if (!fz_last) return;
    __threadfence();
    if (wg > 0) {{
      #pragma unroll
      for (int i = 0; i < {ND}; i++) d[i] = 0.f;
      for (int z = 0; z < {SK}; z++) {{
        const float4* p = {slot("z")};
        #pragma unroll
        for (int i = 0; i < {ND // 4}; i++) {{
          float x0, x1, x2, x3;
          asm volatile("ld.global.cg.v4.f32 {{%0,%1,%2,%3}}, [%4];" : "=f"(x0), "=f"(x1), "=f"(x2), "=f"(x3) : "l"(p + i));
          d[4 * i] += x0; d[4 * i + 1] += x1; d[4 * i + 2] += x2; d[4 * i + 3] += x3;
        }}
      }}
    }}
  }}""" if SK > 1 else ""
  return COMMON + HELPERS + f"""{_wgmma(kind, BX, False)}
extern "C" __global__ void __launch_bounds__({cfg.threads}) {swap_name(kind, out, cfg)}({OUT_C[out]}* __restrict__ C, const __grid_constant__ TMap tmW,
    const __grid_constant__ TMap tmX, const int M, const int N, const int K, const int ldc, const long long scb, const float alpha,
    const int btx, const int btw, const float* __restrict__ sa, const float* __restrict__ sb{sk_params}) {{
  extern __shared__ __align__(1024) u8 smem_raw[];
  u8* smem = smem_raw + ((1024 - (fz_smem(smem_raw) & 1023)) & 1023);
  u8* sW = smem;
  u8* sX = smem + {S * BW * 128};
  u64* full = (u64*)(sX + {S * BX * 128});
  u64* empty = full + {S};
  const int tiles_x = (M + {BX - 1}) / {BX};
  const int pid = blockIdx.x, n0 = (pid / tiles_x) * {BW}, m0 = (pid % tiles_x) * {BX};
  const int bz = blockIdx.y, bx = btx > 1 ? bz : 0, bw = btw > 1 ? bz : 0;
  const int wg = threadIdx.x / 128, t = threadIdx.x % 128, cw = wg - 1;
  const int nk_all = (K + {BKE - 1}) / {BKE}, per = (nk_all + {SK - 1}) / {SK}, kb = {"blockIdx.z * per" if SK > 1 else "0"};
  const int nk = max(0, min(nk_all - kb, per));
  if (threadIdx.x == 0) {{
    #pragma unroll
    for (int s = 0; s < {S}; s++) {{ mbar_init(full + s, 1); mbar_init(empty + s, {NC}); }}
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }}
  __syncthreads();
  float d[{ND}];
  #pragma unroll
  for (int i = 0; i < {ND}; i++) d[i] = 0.f;
  if (wg == 0) {{
    if (t == 0) {{
      for (int kt = 0; kt < nk; kt++) {{
        const int s = kt % {S};
        if (kt >= {S}) mbar_wait(empty + s, ((kt / {S}) - 1) & 1);
        mbar_expect(full + s, {(BW + BX) * 128});
        tma3(sW + s * {BW * 128}, &tmW, full + s, (kb + kt) * {BKE}, n0, bw);
        tma3(sX + s * {BX * 128}, &tmX, full + s, (kb + kt) * {BKE}, m0, bx);
      }}
    }}
  }} else {{
    for (int kt = 0; kt < nk; kt++) {{
      const int s = kt % {S};
      mbar_wait(full + s, (kt / {S}) & 1);
      #pragma unroll
      for (int i = 0; i < {ND}; i++) asm volatile("" : "+f"(d[i]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      #pragma unroll
      for (int kk = 0; kk < 4; kk++)
        wgmma(d, desc(fz_smem(sW + s * {BW * 128} + cw * {64 * 128} + kk * 32), 16, 1024), desc(fz_smem(sX + s * {BX * 128} + kk * 32), 16, 1024), 1);
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      #pragma unroll
      for (int i = 0; i < {ND}; i++) asm volatile("" : "+f"(d[i]) :: "memory");
      if (t == 0) mbar_arrive(empty + s);
    }}
  }}
{fixup}
  if (wg == 0) return;
  {OUT_C[out]}* Cb = C + bz * scb;
  const float scale = alpha * (sa ? sa[0] : 1.f) * (sb ? sb[0] : 1.f);
  const int w = t / 32, lane = t % 32;
  #pragma unroll
  for (int i = 0; i < {BX // 8}; i++)
    #pragma unroll
    for (int h = 0; h < 2; h++) {{
      const int r = n0 + cw * 64 + 16 * w + (lane >> 2) + 8 * h, c = m0 + 8 * i + 2 * (lane & 3);
      if (r < N) {{
        if (c < M) Cb[(long long)c * ldc + r] = {st}(scale * d[4 * i + 2 * h]);
        if (c + 1 < M) Cb[(long long)(c + 1) * ldc + r] = {st}(scale * d[4 * i + 2 * h + 1]);
      }}
    }}
}}
"""

def swap_launch_dims(cfg: SwapCfg, M: int, N: int, Bt: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
  return (-(-M // cfg.BX) * -(-N // cfg.BW), Bt, cfg.SK), (cfg.threads, 1, 1)

def swap_splitk_ws(cfg: SwapCfg, M: int, N: int, Bt: int) -> tuple[int, int]:
  """(f32 workspace elements, int32 counters) a split-K launch needs; (0, 0) for SK == 1."""
  if cfg.SK == 1: return 0, 0
  ntl = -(-M // cfg.BX) * -(-N // cfg.BW) * Bt
  return ntl * cfg.SK * cfg.BW * cfg.BX, ntl
