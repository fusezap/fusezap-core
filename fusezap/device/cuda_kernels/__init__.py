"""CUDA library GEMMs (P3-GEMM, SPEC §11.2): tensor-core matmul plugged into the custom-kernel hook, plus direct entry points.

Kernels (sources rendered per configuration, compiled by NVRTC and cached per device):
  tf32.py   float32 3xTF32 mma.sync (sm_80+), any strides: the scheduler's "matmul" hook (float32 SUM(MUL)) and f32 matmul()
  mma.py    f16/bf16 mma.sync m16n8k16 and fp8 e4m3 m16n8k32 (sm_89+), cp.async pipeline + ldmatrix (sm_80+)
  wgmma.py  sm_90a: TMA + mbarrier pipeline + warp-specialized wgmma (f16/bf16, fp8 e4m3)
  fp8.py    per-tensor e4m3 quantization on the device (sm_89+)
Dispatch (gemm): M <= FZ_GEMM_SMALL_M (default 64; decode) -> on sm_90 16-bit "nt" with TMA-compatible strides the swap-AB wgmma
(weights as the 64-row operand, wg_swap_default), else mma with a short tile; both split K to fill every SM (mma_small_default);
else sm_90 with TMA-compatible strides -> wgmma (16-bit tile by a wave cost model, wg_pick), else sm_80+ -> mma (fp8 needs sm_89+). Tile configs have
per-arch defaults; FZ_BEAM>=1 times the candidate configs for a (gpu, kernel, dtype, layout, M, N, K) once and caches the
winner in ~/.cache/fusezap/cuda_gemm_tune.json. Shapes a 16-bit kernel cannot take (row pitch not a multiple of 16 bytes)
run as float32 3xTF32 on cast copies.

Hook: install(dev) (called by CUDADevice() unless FZ_CUDA_LIB=0) offers {"matmul": HOOK} on a CUDADevice (sm_80+) to
fusezap.schedule's matcher and routes those
kind="custom" items to run_hook (schedule.run_custom would launch one thread). FZ_CUSTOM=0 still disables matching.
f16/bf16 `a @ b` (composed by Tensor.matmul as SUM(MUL) over f32 casts) is matched through the casts by _match16, which
install() puts in front of schedule._match_custom, and runs as HOOK16 = the 16-bit mma.sync / wgmma kernel with f32 output
(the f32 -> 16-bit cast of the result stays a separate kernel). FZ_MMA16=0 at install time keeps 16-bit inputs on 3xTF32.
Batches above 65535 (the gridDim.y limit; every kernel here indexes the batch with blockIdx.y) launch in chunks.
Direct entry points (inference only, no autograd; inputs realized as contiguous CUDA buffers):
  matmul(a, b)                      (..., M, K) @ (..., K, N), batch dims broadcast; f32 / f16 / bf16, output in that dtype
  linear(x, w)                      x (..., K) @ w (N, K)^T (the K-major weight layout every kernel reads natively)
  quantize_fp8(x) -> (q, s)         q uint8 e4m3 bits, s a (1,) float32 device tensor with x ~= e4m3(q) * s
  fp8_linear(xq, wq, sx, sw, dt)    (e4m3(xq) * sx) @ (e4m3(wq) * sw)^T in f32, output dt; sx/sw floats or (1,) f32 tensors
"""
from __future__ import annotations
import functools, json, os, threading
from typing import Any, Callable, TYPE_CHECKING
from fusezap.dtype import DType, dtypes
from fusezap.helpers import CACHE_DIR, getenv, prod
from fusezap.device.cuda_kernels import mma, wgmma, tf32, fp8, gemm_table
from fusezap.device.cuda_kernels._rt import Kernel, tensor_map, TMA_U8, TMA_U16
from fusezap.device.cuda import _check
if TYPE_CHECKING:
  from fusezap.tensor import Tensor
  from fusezap.ir import ScheduleItem
  from fusezap.lazy import LazyBuffer

KIND = {dtypes.float16: "f16", dtypes.bfloat16: "bf16"}
OUT = {dtypes.float32: "f32", dtypes.float16: "f16", dtypes.bfloat16: "bf16"}
ESIZE = {"f16": 2, "bf16": 2, "e4m3": 1}

# ---------------------------------------------------------------- kernel cache / configs

def kernel(dev: Any, name: str, render: Callable[[], str], arch: str | None = None, smem: int = 0) -> Kernel:
  if (ks := dev.__dict__.get("_fz_gemm")) is None: dev._fz_gemm = ks = {}
  if (k := ks.get(name)) is None:
    src = render()
    if getenv("FZ_DEBUG", 0) >= 4: print(src)
    ks[name] = k = Kernel(dev, name, src, arch, smem)
  return k

def smem_limit(dev: Any) -> int:
  if (v := dev.__dict__.get("_fz_smem")) is None:
    cu = dev.cu
    dev._fz_smem = v = int(cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev.device)[1])
  return v

def mma_candidates(dev: Any, kind: str, nn: bool) -> list[mma.MmaCfg]:
  cs = [mma.MmaCfg(128, 128, 2, 2, 64, 4), mma.MmaCfg(128, 128, 2, 2, 64, 3), mma.MmaCfg(128, 256, 2, 4, 64, 3),
        mma.MmaCfg(256, 128, 4, 2, 64, 3), mma.MmaCfg(128, 128, 2, 4, 128, 3), mma.MmaCfg(128, 128, 2, 4, 64, 4),
        mma.MmaCfg(128, 128, 4, 2, 64, 4), mma.MmaCfg(64, 128, 2, 2, 64, 4), mma.MmaCfg(64, 64, 2, 2, 128, 4)]
  return [c for c in cs if mma.valid(c, kind, nn) and mma.smem_bytes(c, kind, nn) <= smem_limit(dev)]

def mma_default(dev: Any, kind: str, nn: bool) -> mma.MmaCfg:
  big = smem_limit(dev) >= 160 * 1024   # A100 / H100 class shared memory
  prefs = ([mma.MmaCfg(128, 128, 2, 4, 128, 3), mma.MmaCfg(128, 128, 2, 4, 64, 4)] if kind == "e4m3" else
           [mma.MmaCfg(128, 256, 2, 4, 64, 4 if big else 3), mma.MmaCfg(128, 128, 2, 2, 64, 3)])
  for c in prefs + mma_candidates(dev, kind, nn):
    if mma.valid(c, kind, nn) and mma.smem_bytes(c, kind, nn) <= smem_limit(dev): return c
  raise RuntimeError(f"no mma config fits {smem_limit(dev)} bytes of shared memory")

# Decode shapes (M <= FZ_GEMM_SMALL_M, default 64) are weight-bandwidth bound. The 128-row tiles above launch ceil(N / BN)
# CTAs, 14 to 18 for a 7B model's N = 3584 / 4608, so most SMs would idle. SPEED-2 (bench/gemm_small.py, docs/status/SPEED.md):
# on an H100 the fast decode tile is BM 16 / 32 / 64 (by M) x BN 64 with 4 warps across N and 256-byte K stages (3 deep),
# and split-K over grid z until about two CTAs per SM stream weights: 2.3-2.9 TB/s against 0.3-1.75 TB/s for the
# earlier "widest BN that fills the SMs" rule, whose 1-warp BN 16 tiles were starved on o_proj / down_proj.
# The split count depends only on (N, K), never on M, so a row's result does not depend on the other rows of a batch.
SMALL_M_BN = ((128, 4), (64, 2), (32, 2), (16, 1))
SMALL_TILES = ((64, 4, 256, 3), (64, 4, 128, 4), (64, 2, 128, 4))   # (BN, WN, BKB, S), preferred first
def small_m(M: int) -> bool: return 0 < M <= getenv("FZ_GEMM_SMALL_M", 64)
def sm_count(dev: Any) -> int:
  if (v := dev.__dict__.get("_fz_sms")) is None:
    cu = dev.cu
    dev._fz_sms = v = int(cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev.device)[1])
  return v
def _small_bm(M: int) -> tuple[int, int]:
  BM = 16 if M <= 16 else 32 if M <= 32 else 64
  return BM, 1 if BM <= 32 else 2
def split_k(dev: Any, kind: str, N: int, K: int, BN: int, BKB: int, Bt: int = 1) -> int:
  """K splits for a small-M tile: enough that the CTAs reach two per SM, at least 4 K stages per split, at most 16."""
  ntl, nk = -(-N // BN) * Bt, -(-K * ESIZE[kind] // BKB)
  return max(1, min(-(-2 * sm_count(dev) // ntl), nk // 4, 16))
def mma_small_candidates(dev: Any, kind: str, nn: bool, M: int, N: int = 0, K: int = 0, Bt: int = 1) -> list[mma.MmaCfg]:
  BM, WM = _small_bm(M)
  cs = [mma.MmaCfg(BM, BN, WM, WN, BKB, S) for BN, WN in SMALL_M_BN for BKB, S in ((128, 6), (128, 4), (64, 6))]
  if K: cs = [mma.MmaCfg(BM, BN, WM, WN, BKB, S, sk) for BN, WN, BKB, S in SMALL_TILES
              for sk in dict.fromkeys((split_k(dev, kind, N, K, BN, BKB, Bt), 1))] + cs
  return [c for c in cs if mma.valid(c, kind, nn) and mma.smem_bytes(c, kind, nn) <= smem_limit(dev)]
def mma_small_sweep(dev: Any, kind: str, nn: bool, M: int) -> list[mma.MmaCfg]:
  """The wide config space bench/gemm_small.py times (tile widths, K stage depths, K splits)."""
  BM, WM = _small_bm(M)
  cs = [mma.MmaCfg(BM, BN, WM, WN, BKB, S, SK) for BN, WN in ((128, 4), (64, 2), (64, 4), (32, 2), (16, 1))
        for BKB, S in ((128, 6), (128, 4), (256, 4), (256, 3), (64, 8)) for SK in (1, 2, 3, 4, 6, 8, 12, 16)]
  return [c for c in cs if mma.valid(c, kind, nn) and mma.smem_bytes(c, kind, nn) <= smem_limit(dev)]
def mma_small_default(dev: Any, kind: str, nn: bool, M: int, N: int, Bt: int = 1, K: int = 0) -> mma.MmaCfg:
  """The first SMALL_TILES tile whose BM 64 form fits (so BN, BKB and the split never change with M), split per split_k
  (K = 0: unknown, no split)."""
  BM, WM = _small_bm(M)
  ok = lambda c: mma.valid(c, kind, nn) and mma.smem_bytes(c, kind, nn) <= smem_limit(dev)
  for BN, WN, BKB, S in SMALL_TILES:
    c = mma.MmaCfg(BM, BN, WM, WN, BKB, S, split_k(dev, kind, N, K, BN, BKB, Bt) if K else 1)
    if ok(c) and ok(mma.MmaCfg(64, BN, 2, WN, BKB, S)): return c
  return mma_default(dev, kind, nn)

# SPEED-R3: on sm_90 the 16-bit "nt" decode linears run the swap-AB wgmma kernel (wgmma.SwapCfg): weights stream through TMA
# as the 64-row wgmma operand, the M <= 64 activation rows are the instruction's n (BX = M rounded up to 8), split-K over
# grid z. NC and the split depend only on (N, K) and the device; BX (n) and S (pipeline depth) follow M but never change the
# order of a row's sum, so a row's result does not depend on the other rows of the batch (test_wg_swap_rows_independent).
# FZ_WG_SWAP=0 keeps mma.sync for these shapes.
SWAP_S = 8
def wg_swap_ok(dev: Any, kind: str, nn: bool, M: int) -> bool:
  return small_m(M) and kind in ("f16", "bf16") and not nn and dev.cc == (9, 0) and bool(getenv("FZ_WG_SWAP", 1))
ONE_K_ORDER = [False]   # set by LLMEngine when it mixes decode rows into prefill forwards (set_one_k_order)
def set_one_k_order(on: bool) -> None:
  """Decode GEMMs without split-K (SK = 1): then the swap-AB kernel sums every output in the same K order as the prefill
  wgmma tiles, bit for bit (bench/korder_check.py, H100, the 7B shapes: 100%; SK = 2 on o / down: 0.2-1.4% of f32 sums),
  so a decode row computed inside a prefill forward equals the decode forward's (speed run 4, plan item 7 option B)."""
  ONE_K_ORDER[0] = bool(on)
def wg_swap_split(dev: Any, kind: str, N: int, K: int, NC: int, Bt: int = 1) -> int:
  """K splits for the swap kernel: the most that keep tiles x splits within one CTA per SM, at least 4 K stages per split,
  at most 16. (H100 sweeps: two CTAs per SM, as mma.sync wants, cost 10-20% at M = 32 on qkv / o / down, where the f32
  partials grow with the rows; rounding the split up past one wave (7B qkv SK 2, o / down SK 3) cost 3.9 / 1.9 / 4.6 us at
  M = 32 and 2.0 / 0.8 / 2.5 us at M = 1 against SK 1 / 2 / 2, and 32B qkv / o / down gained at SK 1 too.)"""
  if ONE_K_ORDER[0]: return 1
  ntl, nk = -(-N // (64 * NC)) * Bt, -(-K * ESIZE[kind] // 128)
  # FZ_WG_SWAP_FILL=f (I57 M3 A/B, default 1): up to f CTAs per SM (a 70B rank's qkv, N = 5120, is 80 tiles: one CTA on
  # 80 of 132 SMs at f = 1). Depends on (N, K) only, like the default, so a row's sum never depends on the batch.
  return max(1, min(getenv("FZ_WG_SWAP_FILL", 1) * sm_count(dev) // ntl, nk // 4, 16))
def wg_swap_default(dev: Any, kind: str, M: int, N: int, K: int, Bt: int = 1, table: bool = True) -> wgmma.SwapCfg:
  """The per-shape table's config when it has this (device, N, K) (gemm_table.lookup: tuned at engine start or shipped;
  its NC / SK fixed per shape, S per BX), else the rule: BX = M rounded up to 8; SWAP_S stages, fewer if two CTAs per SM
  would not fit (the depth never changes a sum). Wide
  shapes of two waves or more (gate_up, lm_head) take 4 stages at BX <= 8 and 6 above, so more CTAs share an SM (speed run
  4 sweep, H100: 7B gate_up 91.2 -> 89.1 us at M = 1 and 98.5 -> 93.1 us at M = 32; lm_head 363.8 -> 359.7 us at M = 32;
  the narrower qkv / o / down keep 8). table=False: the rule alone (the autotune's baseline)."""
  c = _wg_swap_rule(dev, kind, M, N, K, Bt)
  if table and not ONE_K_ORDER[0] and (t := gemm_table.lookup(dev, kind, N, K, Bt)) is not None:
    NC, SK, Sb = t
    S = Sb.get(c.BX) or max(2, min(c.S, (smem_limit(dev) - 2048) // ((64 * NC + c.BX) * 128)))
    tc = wgmma.SwapCfg(NC, c.BX, S, SK)
    if wgmma.swap_valid(tc, kind) and wgmma.swap_smem_bytes(tc) <= smem_limit(dev): return tc
  return c
def _wg_swap_rule(dev: Any, kind: str, M: int, N: int, K: int, Bt: int = 1) -> wgmma.SwapCfg:
  BX = min(64, -(-max(M, 1) // 8) * 8)
  S = max(2, min(SWAP_S, (smem_limit(dev) // 2 - 2048) // ((64 + BX) * 128)))
  wide = -(-N // 64) * Bt >= 2 * sm_count(dev)
  if wide and getenv("FZ_WG_SWAP_WIDE_S", 1): S = min(S, 4 if BX <= 8 else 6)
  # FZ_WG_SWAP_S (I57 M3 A/B): the narrower shapes (one wave or so: qkv / o / down) at this depth, as far as one CTA per SM
  # fits in shared memory: more bytes in flight per SM for the H200's higher bandwidth (the depth never changes a sum)
  if not wide and (fs := getenv("FZ_WG_SWAP_S", 0)) > 0: S = max(2, min(fs, (smem_limit(dev) - 2048) // ((64 + BX) * 128)))
  return wgmma.SwapCfg(1, BX, S, wg_swap_split(dev, kind, N, K, 1, Bt))
def wg_swap_candidates(dev: Any, kind: str, M: int, N: int, K: int, Bt: int = 1) -> list[wgmma.SwapCfg]:
  """The default plus other depths and split counts (bench/gemm_small.py --swap sweeps these; FZ_BEAM picks among them)."""
  BX = min(64, -(-max(M, 1) // 8) * 8)
  cs = list(dict.fromkeys([wg_swap_default(dev, kind, M, N, K, Bt), wg_swap_default(dev, kind, M, N, K, Bt, table=False)])) + [
        wgmma.SwapCfg(NC, BX, S, SK) for NC in (1, 2) for S in (4, 6, 8, 12)
        for SK in dict.fromkeys((1, 2, 3, 4, 6, 8, 12, 16, wg_swap_split(dev, kind, N, K, NC, Bt)))]
  return [c for c in dict.fromkeys(cs) if wgmma.swap_valid(c, kind) and wgmma.swap_smem_bytes(c) <= smem_limit(dev)]

# SPEED-5 / SPEED-R3-step3 (bench/gemm_prefill.py and a per-tile sweep, H100, the 0.5B / 3B / 7B / 32B linears at M = 256-2048):
# every tile gives bit-identical sums (each element accumulates the same k16 steps in the same order), so the tile only
# changes time. WG_TILES holds the best (stages, wait1) of each BM x BN; WG_EFF is each tile's relative throughput when
# every SM is busy (fitted to that sweep; 128 x 256 = 1). Largest tile first: it wins ties.
WG_TILES = (wgmma.WgCfg(256, 2, 4, True), wgmma.WgCfg(192, 2, 4, True), wgmma.WgCfg(256, 1, 5, True, True), wgmma.WgCfg(128, 2, 5, True),
            wgmma.WgCfg(192, 1, 6, True, True), wgmma.WgCfg(128, 1, 8, True, True), wgmma.WgCfg(64, 2, 5, True), wgmma.WgCfg(64, 1, 8, True, True))
WG_EFF = (1.0, 0.93, 0.85, 0.88, 0.75, 0.75, 0.5, 0.6)
WG_CANDIDATES = [wgmma.WgCfg(256, 2, 4, True), wgmma.WgCfg(256, 2, 3, True), wgmma.WgCfg(128, 2, 6, False),
                 wgmma.WgCfg(128, 2, 5, True), wgmma.WgCfg(128, 1, 8, False), wgmma.WgCfg(256, 1, 6, False),
                 wgmma.WgCfg(256, 2, 4, True, True), wgmma.WgCfg(128, 2, 6, False, True), wgmma.WgCfg(192, 2, 4, True, True)]
WG_CANDIDATES += [c for c in WG_TILES if c not in WG_CANDIDATES]
def wg_candidates(kind: str, dev: Any = None) -> list[wgmma.WgCfg]:
  return [c for c in WG_CANDIDATES if wgmma.valid(c, kind) and (dev is None or wgmma.smem_bytes(c) <= smem_limit(dev))]
def wg_default(kind: str) -> wgmma.WgCfg: return wgmma.WgCfg(128, 2, 6, False) if kind == "e4m3" else wgmma.WgCfg(256, 2, 4, True)
WG_TAIL = 0.6   # a partly filled last wave costs WG_TAIL + (1 - WG_TAIL) x its filled fraction of a full wave
def wg_cost(dev: Any, c: wgmma.WgCfg, eff: float, M: int, N: int, Bt: int = 1) -> float:
  """Modelled time of an (M, N) GEMM with tile c, in units of a 128 x 256 tile's time per K: waves x tile area / eff (one CTA
  per SM; a partial last wave counts WG_TAIL + the rest by its fill)."""
  full, part = divmod(-(-M // c.BM) * -(-N // c.BN) * Bt, sm_count(dev))
  return (full + (WG_TAIL + (1 - WG_TAIL) * part / sm_count(dev) if part else 0)) * c.BM * c.BN / eff
def wg_pick(dev: Any, kind: str, M: int, N: int, Bt: int = 1) -> wgmma.WgCfg:
  """The 16-bit wgmma tile for (M, N) up to the prefill sizes (M <= 2048): the WG_TILES entry with the lowest wg_cost. One
  128 x 256 tile for every shape left most SMs idle on the narrow linears (M = 256: 28-36 CTAs for N = 3584 / 4608) and
  quantized badly at M = 512-2048 (7B qkv at M = 2048: 288 CTAs = 2.2 waves; 128 x 192 is 15% faster). Over the sweep the
  pick is within 0.7% of the best tile per shape (128 x 256 everywhere: 11% slower; the M <= 512 waves x area rule: 5%).
  Larger M, fp8 and devices whose shared memory is too small keep wg_default."""
  if kind not in ("f16", "bf16") or M > getenv("FZ_WG_PICK_MAX", 2048): return wg_default(kind)   # FZ_WG_PICK_MAX: the model above 2048 too
  ok = [(c, e) for c, e in zip(WG_TILES, WG_EFF) if wgmma.smem_bytes(c) <= smem_limit(dev)]
  return min(ok, key=lambda ce: wg_cost(dev, *ce, M, N, Bt))[0] if ok else wg_default(kind)

_tune_lock = threading.Lock()
_TUNE_FILE = CACHE_DIR / "cuda_gemm_tune.json"
def _tune_db() -> dict:
  try: return json.loads(_TUNE_FILE.read_text())
  except Exception: return {}
def _tune_save(key: str, val: list) -> None:
  with _tune_lock:
    db = _tune_db()
    db[key] = val
    _TUNE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _TUNE_FILE.with_name(f"{_TUNE_FILE.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(db, indent=0, sort_keys=True))
    os.replace(tmp, _TUNE_FILE)

# ---------------------------------------------------------------- raw launcher (device addresses)

MAXB = 65535   # CUDA's gridDim.y (and .z) limit; every GEMM launch here puts the batch index on blockIdx.y

def gemm(dev: Any, C: int, A: int, B: int, M: int, N: int, K: int, *, kind: str, out: str, nn: bool = False, Bt: int = 1,
         lda: int | None = None, ldb: int | None = None, ldc: int | None = None, sab: int = 0, sbb: int = 0, scb: int = 0,
         alpha: float = 1.0, sa: int = 0, sb: int = 0, impl: str | None = None, cfg: Any = None, wait: bool = False,
         bind: bool = False) -> Any:
  """C[b] = alpha * sa * sb * A[b] @ op(B[b]) on raw device addresses. A (M, K) pitch lda; B (N, K) pitch ldb, or (K, N) when
  nn; C (M, N) pitch ldc; batch strides in elements (0 = broadcast). kind f16/bf16/e4m3, out f32/f16/bf16. sa/sb: optional
  device addresses of f32 scalars. impl "wgmma" / "mma" or None (pick); cfg a WgCfg / MmaCfg or None (default / tuned).
  Returns seconds when wait, else None; bind=True returns a launcher f(wait=False) with the parameters built once."""
  e = ESIZE[kind]
  lda, ldb, ldc = lda or K, ldb or (N if nn else K), ldc or N
  if Bt > MAXB:   # the kernels index the batch with blockIdx.y (gridDim.y <= 65535): launch it in chunks
    eo = 4 if out == "f32" else 2
    fs = [gemm(dev, C + b0 * scb * eo, A + b0 * sab * e, B + b0 * sbb * e, M, N, K, kind=kind, out=out, nn=nn, Bt=min(MAXB, Bt - b0),
               lda=lda, ldb=ldb, ldc=ldc, sab=sab, sbb=sbb, scb=scb, alpha=alpha, sa=sa, sb=sb, impl=impl, cfg=cfg, wait=wait, bind=bind)
          for b0 in range(0, Bt, MAXB)]
    if not bind: return sum(f or 0.0 for f in fs) if wait else None
    def launch(wait: bool = False) -> float | None:
      ts = [f(wait=wait) for f in fs]
      return sum(t or 0.0 for t in ts) if wait else None
    return launch
  if impl is None:
    mma_fits = K * e % 16 == 0 and (not nn or N * e % 16 == 0)   # mma's extra row-alignment needs over TMA's
    tma_ok = wgmma_ok(dev, A, B, kind, nn, lda, ldb, sab, sbb)
    impl = ("wgswap" if tma_ok and wg_swap_ok(dev, kind, nn, M) else "mma" if (small_m(M) and mma_fits) or not tma_ok else "wgmma")
  if impl == "mma":
    assert dev.cc >= ((8, 9) if kind == "e4m3" else (8, 0)), f"{kind} mma.sync needs sm_{'89' if kind == 'e4m3' else '80'}+, have {dev.cc}"
    assert all(x * e % 16 == 0 for x in (lda, ldb, sab, sbb)) and A % 16 == 0 and B % 16 == 0 and K * e % 16 == 0 and (not nn or N * e % 16 == 0), \
      "mma GEMM needs 16-byte aligned rows (K * esize % 16 == 0)"
  if cfg is None: cfg = _pick(dev, impl, kind, out, nn, M, N, K, Bt, lambda c: gemm(dev, C, A, B, M, N, K, kind=kind, out=out, nn=nn, Bt=Bt,
                              lda=lda, ldb=ldb, ldc=ldc, sab=sab, sbb=sbb, scb=scb, alpha=alpha, sa=sa, sb=sb, impl=impl, cfg=c, wait=True))
  pad = lambda nbytes: -(-nbytes // 16) * 16   # an unbatched operand's (unused) batch stride must still be 16-byte aligned
  if impl == "wgswap":
    assert not nn and kind in ("f16", "bf16"), "the swap-AB wgmma takes 16-bit nt operands"
    k = kernel(dev, wgmma.swap_name(kind, out, cfg), lambda: wgmma.render_swap(kind, out, cfg), "sm_90a", wgmma.swap_smem_bytes(cfg))
    bta, btb = (Bt if sab else 1), (Bt if sbb else 1)
    tw = _tmap(dev, B, TMA_U16, e, (K, N, btb), (ldb * e, sbb * e or pad(N * ldb * e)), (64, cfg.BW, 1))
    tx = _tmap(dev, A, TMA_U16, e, (K, M, bta), (lda * e, sab * e or pad(M * lda * e)), (64, cfg.BX, 1))
    gs, bs = wgmma.swap_launch_dims(cfg, M, N, Bt)
    ws = [("p", w) for w in splitk_scratch(dev, *wgmma.swap_splitk_ws(cfg, M, N, Bt))] if cfg.SK > 1 else []
    return (k.bind if bind else functools.partial(k, wait=wait))(gs, bs, [("p", C), ("b", tw), ("b", tx), ("i", M), ("i", N), ("i", K), ("i", ldc), ("p", scb),
                      ("f", alpha), ("i", bta), ("i", btb), ("p", sa), ("p", sb)] + ws)
  if impl == "wgmma":
    name = wgmma.name_of(kind, out, nn, cfg)
    k = kernel(dev, name, lambda: wgmma.render(kind, out, nn, cfg), "sm_90a", wgmma.smem_bytes(cfg))
    bta, btb = (Bt if sab else 1), (Bt if sbb else 1)
    tc = TMA_U8 if e == 1 else TMA_U16
    ta = _tmap(dev, A, tc, e, (K, M, bta), (lda * e, sab * e or pad(M * lda * e)), (128 // e, cfg.BM, 1))
    tb = (_tmap(dev, B, tc, e, (N, K, btb), (ldb * e, sbb * e or pad(K * ldb * e)), (64, 128 // e, 1)) if nn else
          _tmap(dev, B, tc, e, (K, N, btb), (ldb * e, sbb * e or pad(N * ldb * e)), (128 // e, cfg.BN, 1)))
    gs, bs = wgmma.launch_dims(cfg, M, N, Bt)
    return (k.bind if bind else functools.partial(k, wait=wait))(gs, bs, [("p", C), ("b", ta), ("b", tb), ("i", M), ("i", N), ("i", K), ("i", ldc), ("p", scb), ("f", alpha),
                      ("i", bta), ("i", btb), ("p", sa), ("p", sb)])
  name = mma.name_of(kind, out, nn, cfg)
  k = kernel(dev, name, lambda: mma.render(kind, out, nn, cfg), None, mma.smem_bytes(cfg, kind, nn))
  gs, bs = mma.launch_dims(cfg, M, N, Bt)
  ws = [("p", w) for w in splitk_scratch(dev, *mma.splitk_ws(cfg, M, N, Bt))] if cfg.SK > 1 else []
  return (k.bind if bind else functools.partial(k, wait=wait))(gs, bs, [("p", C), ("p", A), ("p", B), ("i", M), ("i", N), ("i", K), ("i", lda), ("i", ldb), ("i", ldc),
                    ("p", sab), ("p", sbb), ("p", scb), ("f", alpha), ("p", sa), ("p", sb)] + ws)

def splitk_scratch(dev: Any, nf: int, nsem: int) -> tuple[int, int]:
  """Device-wide split-K scratch: (f32 partials, int32 tile counters, zeroed) with at least nf / nsem elements. Launches
  run in stream order and each kernel leaves its counters at 0, so every split-K GEMM shares it. It grows by doubling;
  outgrown blocks stay allocated because bound launchers and captured graphs keep their addresses."""
  cu = dev.cu
  cur = dev.__dict__.setdefault("_fz_splitk", [0, 0, 0, 0, []])   # W, nW, sem, nsem, every block ever allocated
  if cur[1] < nf or cur[3] < nsem:
    dev.ctx()
    if cur[1] < nf:
      cur[1] = max(nf, 2 * cur[1], 1 << 20)
      cur[0] = int(_check(cu.cuMemAlloc(4 * cur[1]), "cuMemAlloc(split-K)")); cur[4].append(cur[0])
    if cur[3] < nsem:
      cur[3] = max(nsem, 2 * cur[3], 1 << 14)
      cur[2] = int(_check(cu.cuMemAlloc(4 * cur[3]), "cuMemAlloc(split-K)")); cur[4].append(cur[2])
      _check(cu.cuMemsetD32(cur[2], 0, cur[3]), "cuMemsetD32(split-K)")
  return cur[0], cur[2]

def wgmma_ok(dev: Any, A: int, B: int, kind: str, nn: bool, lda: int, ldb: int, sab: int, sbb: int) -> bool:
  e = ESIZE[kind]
  return (dev.cc == (9, 0) and not getenv("FZ_NO_WGMMA") and not (nn and kind == "e4m3") and A % 16 == 0 and B % 16 == 0
          and all(x * e % 16 == 0 for x in (lda, ldb, sab, sbb)))

def _tmap(dev: Any, addr: int, code: int, e: int, dims: tuple, strides: tuple, box: tuple) -> bytes:
  if (cache := dev.__dict__.get("_fz_tmaps")) is None: dev._fz_tmaps = cache = {}
  key = (addr, code, dims, strides, box)
  if (tm := cache.get(key)) is None:
    if len(cache) > 512: cache.clear()
    cache[key] = tm = tensor_map(addr, code, e, dims, strides, box)
  return tm

def _pick(dev: Any, impl: str, kind: str, out: str, nn: bool, M: int, N: int, K: int, Bt: int, run: Callable[[Any], float | None]) -> Any:
  small = impl == "mma" and small_m(M)
  default = (wg_pick(dev, kind, M, N, Bt) if impl == "wgmma" else wg_swap_default(dev, kind, M, N, K, Bt) if impl == "wgswap" else
             mma_small_default(dev, kind, nn, M, N, Bt, K) if small else mma_default(dev, kind, nn))
  if not getenv("FZ_BEAM", 0): return default
  key = f"{dev.gpu_name}|{impl}|{kind}|{out}|{'nn' if nn else 'nt'}|{M}|{N}|{K}"
  memo = dev.__dict__.setdefault("_fz_tuned", {})
  if key in memo: return memo[key]
  if (hit := _tune_db().get(key)) is not None:
    memo[key] = cfg = (wgmma.WgCfg(*hit) if impl == "wgmma" else wgmma.SwapCfg(*hit) if impl == "wgswap" else mma.MmaCfg(*hit))
    return cfg
  cands = wg_candidates(kind, dev) if impl == "wgmma" else wg_swap_candidates(dev, kind, M, N, K, Bt) if impl == "wgswap" else mma_small_candidates(dev, kind, nn, M, N, K, Bt) + mma_candidates(dev, kind, nn) if small else mma_candidates(dev, kind, nn)
  best, best_t = default, float("inf")
  for c in cands:
    try:
      run(c)
      t = min(run(c) or float("inf") for _ in range(3))
    except Exception as ex:
      if getenv("FZ_DEBUG", 0) >= 1: print(f"cuda gemm candidate {c} failed: {ex}")
      continue
    if getenv("FZ_DEBUG", 0) >= 2: print(f"cuda gemm {key} {c}: {t * 1e6:.1f} us")
    if t < best_t: best, best_t = c, t
  memo[key] = best
  _tune_save(key, list(best.__dict__.values()))
  return best

# ---------------------------------------------------------------- f32 3xTF32 (hook + f32 matmul)

def tf32_gemm(dev: Any, C: int, A: int, B: int, params: tuple[int, ...], wait: bool = False) -> float | None:
  """params as the scheduler's matmul hook: (Bt, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, oa, ob) in elements."""
  Bt, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, oa, ob = params
  assert all(-2**31 <= int(p) < 2**31 for p in params), f"tf32 gemm params overflow int32: {params}"
  la, lb = tf32.layout(sam, sak), tf32.layout(sbn, sbk)
  k = kernel(dev, tf32.name_of(la, lb), lambda: tf32.render(la, lb))
  t = 0.0
  for b0 in range(0, Bt, MAXB):
    nb = min(MAXB, Bt - b0)
    gs, bs = tf32.launch_dims(M, N, nb)
    r = k(gs, bs, [("p", C + 4 * b0 * scb), ("p", A + 4 * (oa + b0 * sab)), ("p", B + 4 * (ob + b0 * sbb))] +
          [("i", v) for v in (nb, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, 0, 0)], wait)
    t += r or 0.0
  return t if wait else None

def run_hook(dev: Any, si: "ScheduleItem") -> float | None:
  C, A, B = si.bufs
  assert C.dtype == A.dtype == B.dtype == dtypes.float32, f"CUDA matmul hook is float32, got {C.dtype} {A.dtype} {B.dtype}"
  if C.size == 0: return None
  return tf32_gemm(dev, *[b.ensure_allocated().handle for b in si.bufs], si.ast[0].params, wait=getenv("FZ_DEBUG", 0) >= 1)

# ---------------------------------------------------------------- 16-bit hook: f16/bf16 `a @ b` on the mma.sync / wgmma kernels
# Tensor.matmul composes a 16-bit a @ b as SUM(MUL(CAST(a, f32), CAST(b, f32))) and casts the sum back, and the scheduler's
# matcher only takes float32, so on its own it would materialize f32 copies of both operands for the 3xTF32 kernel. When the
# matched operands are CASTs of f16/bf16 tensors, _match16 swaps in the 16-bit sources (no cast kernels; same values, since a
# 16-bit product is exact in f32) and the HOOK16 item runs gemm(kind=f16/bf16, out=f32). The f32 -> 16-bit output cast stays.

def _src16(s: "LazyBuffer") -> "LazyBuffer | None":
  """If s reads CAST(h -> float32) of a 16-bit h (unrealized), the view of h's base that s reads through the cast, else None."""
  from fusezap.ops import UnaryOps, LoadOps
  from fusezap.schedule import compose
  c = s.base
  if c.realized is not None or c.op is not UnaryOps.CAST or c.dtype != dtypes.float32: return None
  h = c.srcs[0]
  if h.dtype not in KIND or (h.base.realized is None and h.base.op is LoadOps.CONST): return None
  return h._view(compose(h.st, s.st))

def _gemm_params(F: tuple[int, ...], k: int, vx: tuple, vy: tuple) -> tuple[int, ...] | None:
  """The scheduler's matmul params (Bt, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, oa, ob) for operand views
  (strides, offset) over a MUL shape F reduced on axis k (the grouping of schedule._mm_spec)."""
  from fusezap.schedule import _canon, _row_major, _collapse
  (sx, ox), (sy, oy) = vx, vy
  R = tuple(1 if i == k else s for i, s in enumerate(F))
  sc = _canon(R, _row_major(R))
  grp: dict[str, list[int]] = {"b": [], "m": [], "n": []}
  for i, s in enumerate(F):
    if i != k and s != 1: grp["m" if sy[i] == 0 and sx[i] != 0 else "n" if sx[i] == 0 and sy[i] != 0 else "b"].append(i)
  out: list[int] = []
  for g, ts in (("b", (sx, sy, sc)), ("m", (sx, sc)), ("n", (sy, sc))):
    cs = [_collapse(F, grp[g], t) for t in ts]
    if any(c is None for c in cs): return None
    out.append(cs[0][0])
    grp[g] = [c[1] for c in cs]  # type: ignore[index]
  if out[1] == 1 and out[2] == 1: return None
  (sab, sbb, scb), (sam, scm), (sbn, scn) = grp["b"], grp["m"], grp["n"]
  return (*out, F[k], sab, sam, sx[k], sbb, sy[k], sbn, scb, scm, scn, ox, oy)

def _mm16(p: tuple[int, ...], e: int = 2) -> dict | None:
  """gemm() arguments for matmul params p on e-byte operands, or None when the 16-bit kernels cannot read that layout:
  A K-contiguous, B K- or N-contiguous, C N-contiguous, 16-byte aligned pitches / batch strides / offsets, K % (16/e) == 0."""
  Bt, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, oa, ob = (int(x) for x in p)
  if min(Bt, M, N, K) <= 0 or min(sab, sam, sbb, sbk, sbn, scb, scm, scn, oa, ob) < 0 or sak != 1 or (N > 1 and scn != 1): return None
  if sbk == 1: nn, ldb = False, (sbn if N > 1 else K)
  elif sbn == 1 and N > 1: nn, ldb = True, sbk
  else: return None
  lda, ldc = (sam if M > 1 else K), (scm if M > 1 else N)
  if not all(x * e % 16 == 0 for x in (K, lda, ldb, sab, sbb, oa, ob)) or (nn and N * e % 16) or lda <= 0 or ldb <= 0 or ldc <= 0: return None
  if max(M, N, K, lda, ldb, ldc) >= 2**31: return None
  return dict(Bt=Bt, M=M, N=N, K=K, nn=nn, lda=lda, ldb=ldb, ldc=ldc, sab=sab if Bt > 1 else 0, sbb=sbb if Bt > 1 else 0,
              scb=scb if Bt > 1 else M * N, oa=oa, ob=ob)

def _match16(b: "LazyBuffer") -> Any:
  from fusezap.schedule import _Match, _device_kernels, _sview
  m = _MATCH0(b)
  if m is None or m.kernel is not HOOK or _device_kernels(b.device).get("matmul16") is not HOOK16: return m
  hs = [_src16(s) for s in m.inputs]
  if hs[0] is None or hs[1] is None or hs[0].dtype != hs[1].dtype: return m
  if (vx := _sview(hs[0])) is None or (vy := _sview(hs[1])) is None: return m
  if (p := _gemm_params(tuple(b.srcs[0].shape), b.arg[0], vx, vy)) is None or _mm16(p, 2) is None: return m
  return _Match(HOOK16, (hs[0], hs[1]), p)
_MATCH0: Any = None

def run_hook16(dev: Any, si: "ScheduleItem") -> float | None:
  C, A, B = si.bufs
  assert C.dtype == dtypes.float32 and A.dtype == B.dtype and A.dtype in KIND, f"CUDA 16-bit matmul hook: {C.dtype} {A.dtype} {B.dtype}"
  if C.size == 0: return None
  g = _mm16(si.ast[0].params, 2)
  assert g is not None, f"16-bit matmul hook params the kernels cannot take: {si.ast[0].params}"
  hc, ha, hb = [x.ensure_allocated().handle for x in si.bufs]
  return gemm(dev, hc, ha + 2 * g["oa"], hb + 2 * g["ob"], g["M"], g["N"], g["K"], kind=KIND[A.dtype], out="f32", nn=g["nn"], Bt=g["Bt"],
              lda=g["lda"], ldb=g["ldb"], ldc=g["ldc"], sab=g["sab"], sbb=g["sbb"], scb=g["scb"], wait=getenv("FZ_DEBUG", 0) >= 1)

def _hook_kernel() -> Any:
  from fusezap.schedule import CustomKernel
  return CustomKernel("fz_matmul_tf32x3", {"CUDA": "fusezap.device.cuda_kernels.tf32 (rendered per operand layout)"}, ("C", "A", "B") + tf32.PARAMS, 3)
def _hook16_kernel() -> Any:
  from fusezap.schedule import CustomKernel
  return CustomKernel("fz_matmul_mma16", {"CUDA": "fusezap.device.cuda_kernels.mma / wgmma (f16/bf16 in, f32 out)"},
                      ("C", "A", "B") + tf32.PARAMS, 3, (dtypes.float16, dtypes.bfloat16))
HOOK: Any = None
HOOK16: Any = None

def install(dev: Any) -> bool:
  """Offer the tensor-core matmul to the scheduler on CUDADevice `dev` (needs sm_80+). Returns whether it was installed.
  float32 a @ b runs 3xTF32; f16/bf16 a @ b runs the 16-bit mma.sync / wgmma kernels (see _match16; FZ_MMA16=0 disables)."""
  global HOOK, HOOK16, _MATCH0
  if tuple(dev.cc) < (8, 0): return False
  if HOOK is None: HOOK, HOOK16 = _hook_kernel(), _hook16_kernel()
  ks = {**(getattr(dev, "custom_kernels", None) or {}), "matmul": HOOK}
  if getenv("FZ_MMA16", 1): ks["matmul16"] = HOOK16
  else: ks.pop("matmul16", None)
  dev.custom_kernels = ks
  from fusezap import schedule
  if not getattr(schedule._match_custom, "_fz_gemm16", False):   # stopgap until the matcher looks through 16-bit casts itself
    _MATCH0 = schedule._match_custom
    _match16._fz_gemm16 = True  # type: ignore[attr-defined]
    schedule._match_custom = _match16
  if getattr(dev.run, "_fz_gemm", False): return True
  orig = dev.run
  def run(si: "ScheduleItem") -> float | None:
    if si.kind == "custom" and si.ast and (k := getattr(si.ast[0], "kernel", None)) is not None:
      if k is HOOK: return run_hook(dev, si)
      if k is HOOK16: return run_hook16(dev, si)
    return orig(si)
  run._fz_gemm = True  # type: ignore[attr-defined]
  dev.run = run
  return True

# ---------------------------------------------------------------- direct entry points (Tensors in, Tensor out)

def _dev(name: str) -> Any:
  from fusezap.device import Device
  assert name.split(":")[0].upper() == "CUDA", f"CUDA kernels need CUDA tensors, got {name}"
  return Device[name]

def _base(t: "Tensor") -> Any: return t._realized_contiguous().realized

def _batch(ts: list["Tensor"], tail: list[int]) -> tuple[list["Tensor"], tuple[int, ...]]:
  """Broadcast the leading (batch) dims of ts (each keeps its last tail[i] dims); size-1 batches stay unexpanded."""
  nb = max(t.ndim - n for t, n in zip(ts, tail))
  ts = [t.reshape((1,) * (nb - (t.ndim - n)) + tuple(t.shape)) for t, n in zip(ts, tail)]
  bs = tuple(max(t.shape[i] for t in ts) for i in range(nb))
  out = []
  for t in ts:
    if prod(t.shape[:nb]) != 1 and tuple(t.shape[:nb]) != bs: t = t.expand(bs + tuple(t.shape[nb:]))
    out.append(t)
  return out, bs

def _wrap(buf: Any, shape: tuple[int, ...]) -> "Tensor":
  from fusezap.lazy import LazyBuffer
  from fusezap.tensor import Tensor
  return Tensor(LazyBuffer.from_buffer(buf, shape))

def _new(dev: str, n: int, dt: DType) -> Any:
  from fusezap.device import Buffer
  return Buffer(dev, n, dt).ensure_allocated()

def _gemm_tensors(a: "Tensor", b: "Tensor", nn: bool, kind: str | None, out_dt: DType, alpha: float = 1.0, sa: Any = None, sb: Any = None) -> "Tensor":
  """a (..., M, K) with b (..., K, N) if nn else (..., N, K); kind None = float32 3xTF32."""
  (a, b), bs = _batch([a, b], [2, 2])
  M, K = a.shape[-2], a.shape[-1]
  N = b.shape[-1] if nn else b.shape[-2]
  Bt, dev = prod(bs), _dev(a.device)
  A, B = _base(a), _base(b)
  C = _new(a.device, Bt * M * N, out_dt)
  sab, sbb = (0 if A.size == M * K else M * K), (0 if B.size == K * N else K * N)
  if Bt and M and N:
    if K == 0: dev.cu.cuMemsetD8(C.handle, 0, C.nbytes)
    elif kind is None:
      params = (Bt, M, N, K, sab, K, 1, sbb, *((N, 1) if nn else (1, K)), M * N, N, 1, 0, 0)
      tf32_gemm(dev, C.handle, A.handle, B.handle, params)
    else: gemm(dev, C.handle, A.handle, B.handle, M, N, K, kind=kind, out=OUT[out_dt], nn=nn, Bt=Bt, sab=sab, sbb=sbb, scb=M * N,
               alpha=alpha, sa=sa or 0, sb=sb or 0)
  return _wrap(C, bs + (M, N))

def _fits16(dev: Any, K: int, N: int, nn: bool) -> bool: return tuple(dev.cc) >= (8, 0) and K % 8 == 0 and (not nn or N % 8 == 0)

def matmul(a: "Tensor", b: "Tensor") -> "Tensor":
  """a @ b for (..., M, K) x (..., K, N) with batch broadcast; one dtype of float32/float16/bfloat16; f32 accumulation,
  output in that dtype (like Tensor.matmul)."""
  assert a.ndim >= 2 and b.ndim >= 2 and a.shape[-1] == b.shape[-2], f"matmul shapes {a.shape} @ {b.shape}"
  assert a.dtype == b.dtype and a.dtype in OUT, f"matmul needs one float dtype, got {a.dtype}, {b.dtype}"
  if a.dtype in KIND and _fits16(_dev(a.device), a.shape[-1], b.shape[-1], True): return _gemm_tensors(a, b, True, KIND[a.dtype], a.dtype)
  r = _gemm_tensors(a.cast(dtypes.float32), b.cast(dtypes.float32), True, None, dtypes.float32)
  return r if a.dtype == dtypes.float32 else r.cast(a.dtype).realize()

def linear(x: "Tensor", w: "Tensor") -> "Tensor":
  """x (..., K) @ w(N, K)^T (nn.Linear without bias), same dtype rules as matmul."""
  assert w.ndim == 2 and x.shape[-1] == w.shape[-1], f"linear shapes {x.shape}, {w.shape}"
  assert x.dtype == w.dtype and x.dtype in OUT, f"linear needs one float dtype, got {x.dtype}, {w.dtype}"
  lead = tuple(x.shape[:-1])
  x2 = x.reshape(prod(lead), x.shape[-1]) if x.ndim != 2 else x
  if x.dtype in KIND and _fits16(_dev(x.device), x.shape[-1], w.shape[0], False): r = _gemm_tensors(x2, w, False, KIND[x.dtype], x.dtype)
  else:
    r = _gemm_tensors(x2.cast(dtypes.float32), w.cast(dtypes.float32), False, None, dtypes.float32)
    if x.dtype != dtypes.float32: r = r.cast(x.dtype).realize()
  return r.reshape(lead + (w.shape[0],))

def quantize_fp8(x: "Tensor") -> tuple["Tensor", "Tensor"]:
  """Per-tensor e4m3: (q uint8 bits shaped like x, scale (1,) float32) with x ~= e4m3(q) * scale (sm_89+)."""
  dev = _dev(x.device)
  assert tuple(dev.cc) >= (8, 9), f"fp8 needs sm_89+, have {dev.cc}"
  assert x.dtype in OUT, f"quantize_fp8 takes float32/float16/bfloat16, got {x.dtype}"
  dt, n = OUT[x.dtype], prod(x.shape)
  X = _base(x)
  Q, S, AM = _new(x.device, n, dtypes.uint8), _new(x.device, 1, dtypes.float32), _new(x.device, 1, dtypes.int32)
  dev.ctx()
  dev.cu.cuMemsetD32(AM.handle, 0, 1)
  k = kernel(dev, f"fz_absmax_{dt}", lambda: fp8.render(dt))
  kq = kernel(dev, f"fz_fp8_quant_{dt}", lambda: fp8.render(dt))
  blocks = max(1, min(-(-n // 256), 4096))
  k((blocks,), (256,), [("p", AM.handle), ("p", X.handle), ("p", n)])
  kq((blocks,), (256,), [("p", Q.handle), ("p", S.handle), ("p", X.handle), ("p", AM.handle), ("p", n)])
  return _wrap(Q, tuple(x.shape)), _wrap(S, (1,))

def fp8_linear(xq: "Tensor", wq: "Tensor", sx: "float | Tensor" = 1.0, sw: "float | Tensor" = 1.0, out_dtype: DType = dtypes.bfloat16) -> "Tensor":
  """(e4m3(xq) * sx) @ (e4m3(wq) * sw)^T: xq (..., K) and wq (N, K) uint8 e4m3 bits; per-tensor scales as floats or (1,)
  float32 CUDA tensors (no host sync); f32 accumulation; output out_dtype. Needs K % 16 == 0 and sm_89+."""
  assert xq.dtype == wq.dtype == dtypes.uint8 and wq.ndim == 2 and xq.shape[-1] == wq.shape[-1], f"fp8_linear {xq.shape} {xq.dtype}, {wq.shape} {wq.dtype}"
  K, dev = xq.shape[-1], _dev(xq.device)
  assert K % 16 == 0 and tuple(dev.cc) >= (8, 9), f"fp8_linear needs K % 16 == 0 (K={K}) and sm_89+ (have {dev.cc})"
  alpha, ptrs = 1.0, []
  for s in (sx, sw):
    if isinstance(s, (int, float)): alpha *= float(s); ptrs.append(0)
    else:
      assert s.dtype == dtypes.float32 and prod(s.shape) == 1, f"fp8 scale must be a float or a (1,) float32 tensor, got {s.shape} {s.dtype}"
      ptrs.append(_base(s).handle)
  lead = tuple(xq.shape[:-1])
  x2 = xq.reshape(prod(lead), K) if xq.ndim != 2 else xq
  r = _gemm_tensors(x2, wq, False, "e4m3", out_dtype, alpha, ptrs[0], ptrs[1])
  return r.reshape(lead + (wq.shape[0],))
