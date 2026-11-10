"""One-shot all-reduce over CUDA IPC + peer access (NVLink) for tensor parallelism (I57 M3), as vLLM's custom_all_reduce.

  car = CustomAR(dev, rank, world)          # every rank: a 2 x slot data buffer + a signal area, cuMemAlloc'd
  h = car.handle()                          # 64-byte cuIpcGetMemHandle, exchanged over the control channel
  car.open([h0, h1, ...])                   # every rank maps the others' buffers (cuIpcOpenMemHandle, lazy peer access)
  car.all_reduce(dev, out, inp, n, dtype)   # one kernel on dev.stream: capturable in CUDA graphs (fixed addresses)
  car.all_reduce_add_norm(...)              # the same sum fused with the residual add + RMSNorm that follows it

Small all-reduces (a decode step: 1-64 rows of hidden_size 16-bit values, 16 KB - 1 MB) are latency-bound: NCCL's took
~10-28 us each on 2x H200 (160 per 70B decode step). One kernel per call instead: G blocks (FZ_TP_AR_BLOCKS, default 32,
the same on every rank and in both kernels) of 512 threads (the fused form: 256); block b
  1. copies its share of the input into this rank's data slot (flag & 1: double-buffered), so the inputs need no
     registration and any buffer works (graph replays included);
  2. signals every peer's block b (st.release.sys of a per-block counter into the peer's signal area) and waits for
     theirs (ld.acquire.sys; counters only grow, compared wrap-safe);
  3. reads the same share from every rank's slot (ld.global.cg: L2, never a stale L1 line) and sums in float32 in rank
     order, then rounds once: every rank gets the same bits, equal to NCCL's for two ranks and to the host backend's.
The counter of block b lives in this rank's signal area and advances once per call (every call launches all G blocks),
so the slot parity is the same for every block of a call. A rank reuses a slot two calls later, after its peers passed
the barrier of the call in between, so their previous kernel (the reader) has finished: no end barrier. The fused
form (all_reduce_add_norm) runs one row per block (grid-stride) with render_rmsnorm's exact per-row arithmetic (256
threads, the same element-to-thread map and reduction tree), so its x + d and the norm are bit-identical to the
all-reduce followed by fusezap.serve.cuda.rmsnorm.

Messages above the slot (FZ_TP_AR_MAX_MB, default 8 MB per slot), not a multiple of 16 bytes, or without peer access
between every pair of devices (no NVLink / P2P: RTX 4090s) go to NCCL. Every rank must take the same path for the same
call: the choice depends only on the size, the dtype and a group-wide flag (enabled only if every rank set up).
FZ_TP_AR_SPIN_S > 0 (tests): a barrier that waits that long gives up and sets an error word (checked by check()) instead
of spinning forever.
"""
from __future__ import annotations
import ctypes, os
from typing import Any
from fusezap.dtype import DType, dtypes

MAX_BLOCKS, MAX_W = 64, 8
T = 256   # threads per block (render_rmsnorm's NORM_THREADS: the fused form must match its reduction)
SIG_WORDS = MAX_BLOCKS * MAX_W + MAX_BLOCKS + 32   # start flags [block][src rank] | counters [block] | error word
CNT, ERR = MAX_BLOCKS * MAX_W, MAX_BLOCKS * MAX_W + MAX_BLOCKS
SIG_BYTES = 4096
assert SIG_WORDS * 4 <= SIG_BYTES

def _env_f(name: str, default: float) -> float:
  try: return float(os.environ.get(name, "") or default)
  except ValueError: return default

def blocks() -> int: return max(1, min(MAX_BLOCKS, int(_env_f("FZ_TP_AR_BLOCKS", 32))))
def slot_bytes() -> int: return max(1 << 16, int(_env_f("FZ_TP_AR_MAX_MB", 8) * (1 << 20)) // 16 * 16)
def enabled_env() -> bool:
  """FZ_TP_CUSTOM_AR (default 1); NCCL_P2P_DISABLE=1 (the 4090 runs) turns it off as well."""
  return os.environ.get("FZ_TP_CUSTOM_AR", "1") not in ("0", "") and os.environ.get("NCCL_P2P_DISABLE", "") not in ("1",)

ESZ = {dtypes.float32: 4, dtypes.float16: 2, dtypes.bfloat16: 2}
def takes(nbytes: int, dt: DType, slot: int) -> bool:
  """The custom kernel handles this message (every rank decides the same from the same arguments)."""
  return dt in ESZ and 0 < nbytes <= slot and nbytes % 16 == 0

# ---------------------------------------------------------------- kernel sources

_COMMON = r"""typedef unsigned int u32; typedef unsigned long long u64; typedef unsigned short u16;
struct FzPeers { u64 data[8]; u64 sig[8]; };
static __device__ __forceinline__ void fz_st_rel(u32* p, u32 v) { asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
static __device__ __forceinline__ u32 fz_ld_acq(const u32* p) { u32 v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
static __device__ __forceinline__ uint4 fz_ld_cg(const uint4* p) {
  uint4 v; asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }
static __device__ __forceinline__ u64 fz_gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
"""

def _barrier(W: int, rank: int, spin_ns: int) -> str:
  """Block b's barrier with block b of every other rank (thread r < W signals and waits for rank r)."""
  spin = (f"""      if (t0 == 0ull) t0 = fz_gtime();
      else if (fz_gtime() - t0 > {spin_ns}ull) {{ atomicExch(sig + {ERR}, 1u); break; }}""" if spin_ns > 0 else "")
  return f"""static __device__ __forceinline__ void fz_barrier(const FzPeers& P, u32* sig, const u32 flag) {{
  __syncthreads();   // this block's slot writes happen before the release below (cumulative)
  if (threadIdx.x < {W} && threadIdx.x != {rank}) {{
    fz_st_rel((u32*)P.sig[threadIdx.x] + blockIdx.x * {MAX_W} + {rank}, flag);
    const u32* mine = sig + blockIdx.x * {MAX_W} + threadIdx.x;
    u64 t0 = 0ull; (void)t0;
    while ((int)(fz_ld_acq(mine) - flag) < 0) {{
{spin}
    }}
  }}
  __syncthreads();
}}
"""

def _cvt(dt: DType) -> tuple[str, int, str, str]:
  """(helpers, elements per 16 bytes, unpack(word, k) -> float expr template, pack template)."""
  from fusezap.codegen.cuda import BF16_HELPERS, HALF_HELPERS
  if dt == dtypes.bfloat16: return BF16_HELPERS, 8, "fz_bf2f", "fz_f2bf"
  if dt == dtypes.float16: return HALF_HELPERS, 8, "fz_h2f", "fz_f2h"
  return "", 4, "", ""

def _sum_vars(dt: DType, W: int, names: list[str], out: str) -> str:
  """out (uint4) = the rank-order float32 sum of the W packs names[0..W-1] (16 bytes each), rounded once to dt."""
  _, _, ld, st = _cvt(dt)
  if dt == dtypes.float32:
    return "".join(f"    {{ float a = __uint_as_float({names[0]}.{c});" + "".join(f" a += __uint_as_float({names[r]}.{c});" for r in range(1, W))
                   + f" {out}.{c} = __float_as_uint(a); }}\n" for c in "xyzw")
  comp = []
  for c in "xyzw":
    lo = f"{ld}((u16)({names[0]}.{c} & 0xffffu))" + "".join(f" + {ld}((u16)({names[r]}.{c} & 0xffffu))" for r in range(1, W))
    hi = f"{ld}((u16)({names[0]}.{c} >> 16))" + "".join(f" + {ld}((u16)({names[r]}.{c} >> 16))" for r in range(1, W))
    comp.append(f"    {{ const float lo = {lo}; const float hi = {hi}; {out}.{c} = (u32){st}(lo) | ((u32){st}(hi) << 16); }}")
  return "\n".join(comp) + "\n"

def _loads(W: int, rank: int, dst: str, own: str, off: str) -> str:
  """dst[r] = rank r's pack: this rank's from `own`, the others' from their slot at uint4 offset `off` (L2 only)."""
  return "".join(f"{dst}[{r}] = {own}; " if r == rank else f"{dst}[{r}] = fz_ld_cg((const uint4*)(P.data[{r}] + sl) + {off}); " for r in range(W))

TA = 512   # threads per block of the plain all-reduce (only the fused form must keep render_rmsnorm's 256)

def ar_name(dt: DType, W: int, rank: int, G: int, spin_ns: int) -> str:
  return f"fz_car_{dt.name}_w{W}r{rank}_g{G}" + (f"_spin{spin_ns}" if spin_ns else "")

def render_ar(dt: DType, W: int, rank: int, G: int, slot: int, spin_ns: int = 0) -> str:
  """OUT (n16 x 16 bytes) = sum over ranks of IN: copy into this rank's slot, barrier, sum every rank's slot. Phase 2
  issues U packs' loads (from every rank) before it sums any, so each thread keeps U x W loads in flight."""
  hlp = _cvt(dt)[0]
  nm = ar_name(dt, W, rank, G, spin_ns)
  U, S = max(1, 8 // W), G * TA
  return _COMMON + hlp + "\n" + _barrier(W, rank, spin_ns) + f"""
extern "C" __global__ void __launch_bounds__({TA}) {nm}(uint4* __restrict__ OUT, const uint4* __restrict__ IN, u32* __restrict__ sig,
    const FzPeers P, const int n16) {{
  const u32 flag = sig[{CNT} + blockIdx.x] + 1u;
  const u64 sl = (u64)(flag & 1u) * {slot}ull;
  uint4* mine = (uint4*)(P.data[{rank}] + sl);
  const int tid = blockIdx.x * {TA} + threadIdx.x;
  #pragma unroll 4
  for (int i = tid; i < n16; i += {S}) mine[i] = IN[i];
  fz_barrier(P, sig, flag);
  for (int i0 = tid; i0 < n16; i0 += {U * S}) {{
    uint4 L[{U}][{W}];
    #pragma unroll
    for (int u = 0; u < {U}; u++) {{
      const int i = i0 + u * {S};
      if (i < n16) {{ {_loads(W, rank, "L[u]", "IN[i]", "i")}}}
    }}
    #pragma unroll
    for (int u = 0; u < {U}; u++) {{
      const int i = i0 + u * {S};
      if (i < n16) {{
        uint4 r;
{_sum_vars(dt, W, [f"L[u][{r}]" for r in range(W)], "r")}        OUT[i] = r;
      }}
    }}
  }}
  if (threadIdx.x == 0) sig[{CNT} + blockIdx.x] = flag;
}}
"""

def norm_name(dt: DType, W: int, rank: int, G: int, K: int, spin_ns: int) -> str:
  return f"fz_carnorm_{dt.name}_w{W}r{rank}_g{G}_k{K}" + (f"_spin{spin_ns}" if spin_ns else "")

def render_ar_norm(dt: DType, W: int, rank: int, G: int, slot: int, K: int, eps: float, spin_ns: int = 0) -> str:
  """M rows of K (bf16 / f16): D = the all-reduce of P (rank-order f32 sum, rounded), XO = round(X + D), H = w * norm(XO):
  render_rmsnorm(res=True)'s per-row code with D taken from the all-reduce (one row per block, grid-stride)."""
  import numpy as np
  assert dt in (dtypes.bfloat16, dtypes.float16) and K % 8 == 0
  hlp, _, ld, st = _cvt(dt)
  rd = "fz_bfround" if dt == dtypes.bfloat16 else "fz_hround"
  V = K // 8
  NV = -(-V // T)
  f = lambda v: f"{float(np.float32(v))!r}f"
  nm = norm_name(dt, W, rank, G, K, spin_ns)
  s = _sum_vars(dt, W, [f"pk[j][{r}]" for r in range(W)], "b")
  return _COMMON + hlp + "\n" + _barrier(W, rank, spin_ns) + f"""
extern "C" __global__ void __launch_bounds__({T}) {nm}(u16* __restrict__ H, u16* __restrict__ XO, const u16* __restrict__ PIN,
    const u16* __restrict__ X, const u16* __restrict__ Wt, u32* __restrict__ sig, const FzPeers P, const int M) {{
  __shared__ float red[{T // 32}];
  const int t = threadIdx.x;
  const u32 flag = sig[{CNT} + blockIdx.x] + 1u;
  const u64 sl = (u64)(flag & 1u) * {slot}ull;
  uint4* mine = (uint4*)(P.data[{rank}] + sl);
  for (int row = blockIdx.x; row < M; row += {G}) {{
    const long long o16 = (long long)row * {V};
    const uint4* p4 = (const uint4*)PIN + o16;
    #pragma unroll
    for (int j = 0; j < {NV}; j++) if (t + j * {T} < {V}) mine[o16 + t + j * {T}] = p4[t + j * {T}];
  }}
  fz_barrier(P, sig, flag);
  const uint4* w4 = (const uint4*)Wt;
  for (int row = blockIdx.x; row < M; row += {G}) {{
    const long long o = (long long)row * {K}, o16 = (long long)row * {V};
    const uint4* x4 = (const uint4*)(X + o);
    const uint4* p4 = (const uint4*)PIN + o16;
    uint4* xo4 = (uint4*)(XO + o);
    uint4 wv[{NV}], xa[{NV}], pk[{NV}][{W}];   // every load of the row issued before the first sum
    #pragma unroll
    for (int j = 0; j < {NV}; j++) {{
      const int i = t + j * {T};
      if (i < {V}) {{ wv[j] = w4[i]; xa[j] = x4[i]; {_loads(W, rank, "pk[j]", "p4[i]", "o16 + i")}}}
    }}
    float v[{NV}][8];
    float ss = 0.f;
    #pragma unroll
    for (int j = 0; j < {NV}; j++) {{
      const int i = t + j * {T};
      if (i < {V}) {{
        const uint4 a = xa[j];
        uint4 b;
{s}        uint4 r;
        const u32* pa = (const u32*)&a;
        const u32* pb = (const u32*)&b;
        u32* pr = (u32*)&r;
        #pragma unroll
        for (int k = 0; k < 4; k++) {{
          float lo = {ld}((u16)(pa[k] & 0xffffu)), hi = {ld}((u16)(pa[k] >> 16));
          lo = {rd}(lo + {ld}((u16)(pb[k] & 0xffffu))); hi = {rd}(hi + {ld}((u16)(pb[k] >> 16)));
          pr[k] = (u32){st}(lo) | ((u32){st}(hi) << 16);
          v[j][2 * k] = lo; v[j][2 * k + 1] = hi;
          ss += lo * lo; ss += hi * hi;
        }}
        xo4[i] = r;
      }}
    }}
    for (int s = 16; s > 0; s >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, s);
    if ((t & 31) == 0) red[t >> 5] = ss;
    __syncthreads();
    if (t < 32) {{
      ss = t < {T // 32} ? red[t] : 0.f;
      for (int s = 16; s > 0; s >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, s);
      if (t == 0) red[0] = ss;
    }}
    __syncthreads();
    const float rs = 1.0f / sqrtf(red[0] * {f(1.0 / K)} + {f(eps)});
    __syncthreads();   // red[0] is read by every thread before the next row's reduction writes it
    uint4* h4 = (uint4*)(H + o);
    #pragma unroll
    for (int j = 0; j < {NV}; j++) {{
      const int i = t + j * {T};
      if (i < {V}) {{
        const uint4 a = wv[j];
        uint4 r;
        const u32* pa = (const u32*)&a;
        u32* pr = (u32*)&r;
        #pragma unroll
        for (int k = 0; k < 4; k++) {{
          const float lo = {ld}((u16)(pa[k] & 0xffffu)) * {rd}(v[j][2 * k] * rs);
          const float hi = {ld}((u16)(pa[k] >> 16)) * {rd}(v[j][2 * k + 1] * rs);
          pr[k] = (u32){st}(lo) | ((u32){st}(hi) << 16);
        }}
        h4[i] = r;
      }}
    }}
  }}
  if (threadIdx.x == 0) sig[{CNT} + blockIdx.x] = flag;
}}
"""

# ---------------------------------------------------------------- driver glue (libcuda through ctypes: IPC, peer access)

# The handle is 64 opaque bytes, NULs included: c_ubyte, not c_char. A c_char array field reads back as bytes cut at the
# first NUL, which broke the first M3 run: rank 0 got a few-byte prefix of each peer's handle, padded with whatever
# followed it in memory, and cuIpcOpenMemHandle returned CUDA_ERROR_INVALID_VALUE on every rank.
IPC_BYTES = 64
class _IpcHandle(ctypes.Structure): _fields_ = [("reserved", ctypes.c_ubyte * IPC_BYTES)]
def handle_bytes(h: _IpcHandle) -> bytes: return ctypes.string_at(ctypes.addressof(h), IPC_BYTES)
def handle_from(hb: Any) -> _IpcHandle:
  if not isinstance(hb, (bytes, bytearray)) or len(hb) != IPC_BYTES:
    raise ValueError(f"CUDA IPC handle: {IPC_BYTES} bytes expected, got {type(hb).__name__} {hb!r:.80}")
  h = _IpcHandle()
  ctypes.memmove(ctypes.addressof(h), bytes(hb), IPC_BYTES)
  return h
_LIB: list = []
def _cuda() -> Any:
  if not _LIB: _LIB.append(ctypes.CDLL("libcuda.so.1"))
  return _LIB[0]
def _ck(ret: int, what: str) -> None:
  if ret != 0: raise RuntimeError(f"CUDA error {ret} in {what}")

def can_access_peer(a: int, b: int) -> bool:
  """Device ordinals a and b can access each other's memory (P2P over NVLink or PCIe). cuInit first (idempotent): in a
  fresh process (the M3 script's standalone check) cuDeviceGet otherwise fails with CUDA_ERROR_NOT_INITIALIZED (3)."""
  lib, v, w = _cuda(), ctypes.c_int(0), ctypes.c_int(0)
  _ck(lib.cuInit(0), "cuInit")
  da, db = ctypes.c_int(), ctypes.c_int()
  _ck(lib.cuDeviceGet(ctypes.byref(da), a), "cuDeviceGet"); _ck(lib.cuDeviceGet(ctypes.byref(db), b), "cuDeviceGet")
  _ck(lib.cuDeviceCanAccessPeer(ctypes.byref(v), da, db), "cuDeviceCanAccessPeer")
  _ck(lib.cuDeviceCanAccessPeer(ctypes.byref(w), db, da), "cuDeviceCanAccessPeer")
  return bool(v.value and w.value)

def ordinal(device: str) -> int: return int(device.split(":")[1]) if ":" in device else 0

class CustomAR:
  """This rank's side of the custom all-reduce (see the module docstring). ok: set up (open() succeeded)."""
  def __init__(self, dev: Any, rank: int, world: int):
    assert 2 <= world <= MAX_W, f"custom all-reduce: world {world} (2..{MAX_W})"
    self.dev, self.rank, self.world = dev, rank, world
    self.G, self.slot, self.spin_ns = blocks(), slot_bytes(), int(_env_f("FZ_TP_AR_SPIN_S", 0) * 1e9)
    self.ok, self.opened = False, []
    cu = dev.cu
    dev.ctx()
    from fusezap.device.cuda import _check
    self.nbytes = SIG_BYTES + 2 * self.slot
    self.base = int(_check(cu.cuMemAlloc(self.nbytes), "cuMemAlloc(custom all-reduce)"))
    _check(cu.cuMemsetD32(self.base, 0, SIG_BYTES // 4), "cuMemsetD32(custom all-reduce)")
    dev.synchronize()
    self.sig, self.data = self.base, self.base + SIG_BYTES
    self.peers: list[int] = []
    self._k: dict = {}
  def handle(self) -> bytes:
    self.dev.ctx()
    h = _IpcHandle()
    _ck(_cuda().cuIpcGetMemHandle(ctypes.byref(h), ctypes.c_uint64(self.base)), "cuIpcGetMemHandle")
    return handle_bytes(h)
  def open(self, handles: list[bytes]) -> None:
    """Map every other rank's buffer (handles in rank order; this rank's own entry is ignored)."""
    self.dev.ctx()
    lib = _cuda()
    fn = getattr(lib, "cuIpcOpenMemHandle_v2", None) or lib.cuIpcOpenMemHandle
    if isinstance(fn, ctypes._CFuncPtr):   # the handle goes by value (a 64-byte struct)
      fn.argtypes, fn.restype = [ctypes.POINTER(ctypes.c_uint64), _IpcHandle, ctypes.c_uint], ctypes.c_int
    assert len(handles) == self.world, f"custom all-reduce: {len(handles)} handles for {self.world} ranks"
    bases = []
    for r, hb in enumerate(handles):
      if r == self.rank: bases.append(self.base); continue
      h, p = handle_from(hb), ctypes.c_uint64(0)
      _ck(fn(ctypes.byref(p), h, ctypes.c_uint(1)), f"cuIpcOpenMemHandle(rank {r})")   # CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
      bases.append(int(p.value)); self.opened.append(int(p.value))
    self.peers = bases
    self.ok = True
  def _pk(self) -> bytes:
    data = [b + SIG_BYTES for b in self.peers] + [0] * (MAX_W - self.world)
    sig = list(self.peers) + [0] * (MAX_W - self.world)
    return (ctypes.c_uint64 * 16)(*data, *sig)
  def takes(self, nbytes: int, dt: DType) -> bool: return self.ok and takes(nbytes, dt, self.slot)
  def norm_takes(self, M: int, K: int, dt: DType) -> bool:
    return self.ok and dt in (dtypes.bfloat16, dtypes.float16) and K % 8 == 0 and 0 < M * K * 2 <= self.slot
  def _kernel(self, name: str, render: Any) -> Any:
    if (k := self._k.get(name)) is None:
      from fusezap.device.cuda_kernels._rt import Kernel
      self._k[name] = k = Kernel(self.dev, name, render())
    return k
  def all_reduce(self, dev: Any, out: Any, inp: Any, n: int, dt: DType) -> None:
    nb = n * ESZ[dt]
    assert self.takes(nb, dt) and out.handle % 16 == 0 and inp.handle % 16 == 0, "custom all-reduce: size or alignment"
    nm = ar_name(dt, self.world, self.rank, self.G, self.spin_ns)
    k = self._kernel(nm, lambda: render_ar(dt, self.world, self.rank, self.G, self.slot, self.spin_ns))
    k((self.G, 1, 1), (TA, 1, 1), [("p", out.handle), ("p", inp.handle), ("p", self.sig), ("b", bytes(self._pk())), ("i", nb // 16)])
  def all_reduce_add_norm(self, dev: Any, h: Any, xo: Any, p: Any, x: Any, w: Any, M: int, K: int, eps: float, dt: DType) -> None:
    assert self.norm_takes(M, K, dt)
    nm = norm_name(dt, self.world, self.rank, self.G, K, self.spin_ns)
    k = self._kernel(nm, lambda: render_ar_norm(dt, self.world, self.rank, self.G, self.slot, K, eps, self.spin_ns))
    k((self.G, 1, 1), (T, 1, 1), [("p", h.handle), ("p", xo.handle), ("p", p.handle), ("p", x.handle), ("p", w.handle), ("p", self.sig),
                                   ("b", bytes(self._pk())), ("i", M)])
  def check(self) -> None:
    """Raise if a barrier gave up (FZ_TP_AR_SPIN_S): a peer never reached the same call."""
    import numpy as np
    from fusezap.device.cuda import _check
    self.dev.synchronize()
    v = np.zeros(1, np.uint32)
    _check(self.dev.cu.cuMemcpyDtoH(v.ctypes.data, self.sig + 4 * ERR, 4), "cuMemcpyDtoH")
    if v[0]: raise RuntimeError(f"custom all-reduce: rank {self.rank}: a barrier timed out (FZ_TP_AR_SPIN_S)")
  def close(self) -> None:
    """Unmap the peers' buffers (this rank's own is freed with the context at exit)."""
    lib = _cuda()
    for p in self.opened:
      try: lib.cuIpcCloseMemHandle(ctypes.c_uint64(p))
      except Exception: pass
    self.opened, self.ok = [], False
