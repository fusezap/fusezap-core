"""The custom all-reduce (fusezap/serve/custom_ar.py, I57 M3) and its routing (dist.TPComm).

  1. the CUDA sources parse as C++ (clang++ -fsyntax-only with stand-ins for the CUDA built-ins; skipped without clang++):
     every dtype, world size, rank, a spin limit, the fused add + RMSNorm form at several row widths
  2. the routing: which messages the custom kernel takes (size, alignment, dtype, slot), every rank deciding alike;
     TPComm sends the rest to the base collective, mode "base" / "skip", the fused form's row limit
  3. a numpy model of the one-shot algorithm (double-buffered slots, per-block counters, rank-order f32 sums) over many
     calls of different kinds: no slot is overwritten while a peer can still read it, and the sums equal NCCL's
  4. CUDA, two GPUs with peer access (scripts/tp_m3_70b.sh): dist.selftest_custom on both ranks (bits vs NCCL, eager and
     in CUDA graphs, the fused form vs all-reduce + rmsnorm), latency vs NCCL at decode sizes
"""
from __future__ import annotations
import json, os, re, shutil, subprocess
import numpy as np
import pytest
from fusezap.dtype import dtypes
from fusezap.serve import custom_ar as CA
from fusezap.serve import dist

PRE = r"""
#define __global__
#define __device__
#define __forceinline__ inline
#define __launch_bounds__(x)
#define __shared__ static
#define __restrict__
#define FZ_ASM(...) 0
struct fz_dim3 { unsigned x, y, z; }; extern fz_dim3 threadIdx, blockIdx, blockDim, gridDim;
struct uint4 { unsigned x, y, z, w; }; struct float4 { float x, y, z, w; };
void __syncthreads(); void __threadfence_system();
unsigned atomicExch(unsigned*, unsigned);
float __shfl_xor_sync(unsigned, float, int);
float __uint_as_float(unsigned); unsigned __float_as_uint(float);
float sqrtf(float);
"""

def _sources() -> list[str]:
  out = [CA.render_ar(dt, W, r, G, 1 << 23, sp) for dt in (dtypes.bfloat16, dtypes.float16, dtypes.float32) for W in (2, 4)
         for r in (0, W - 1) for G in (1, 32) for sp in (0, 10 ** 9)]
  out += [CA.render_ar_norm(dt, W, r, 32, 1 << 23, K, 1e-5, sp) for dt in (dtypes.bfloat16, dtypes.float16) for W in (2, 8)
          for r in (0, 1) for K in (8192, 5120, 896) for sp in (0, 5)]
  return out

@pytest.mark.skipif(shutil.which("clang++") is None, reason="needs clang++")
def test_sources_parse():
  """The kernels' CUDA C++ parses (inline PTX replaced by a no-op; NVRTC itself only runs on a CUDA machine)."""
  srcs = _sources()
  for s in srcs:
    s2 = re.sub(r"\basm\s+volatile\s*\(|\basm\s*\(", "FZ_ASM(", s)
    p = subprocess.run(["clang++", "-fsyntax-only", "-x", "c++", "-std=c++17", "-Wall", "-Wno-unused-function", "-Wno-uninitialized",
                        "-Wno-unused-variable", "-Wno-unused-value", "-"], input=PRE + s2, text=True, capture_output=True)
    assert p.returncode == 0, p.stderr[:4000] + "\n--- source ---\n" + s
  names = {re.search(r'extern "C" __global__ void __launch_bounds__\(\d+\) (\w+)\(', s).group(1) for s in srcs}
  assert len(names) == len(srcs), "kernel names must tell every variant apart (the module cache is by name)"

def test_sources_shape():
  """Rank-order sums (own value at its rank's position), the barrier signals every other rank, a fixed block count."""
  s = CA.render_ar(dtypes.bfloat16, 2, 1, 32, 1 << 20)
  assert "L[u][0] = fz_ld_cg((const uint4*)(P.data[0] + sl) + i); L[u][1] = IN[i];" in s
  assert "fz_bf2f((u16)(L[u][0].x & 0xffffu)) + fz_bf2f((u16)(L[u][1].x & 0xffffu))" in s   # rank 0's term first, on every rank
  assert "threadIdx.x < 2 && threadIdx.x != 1" in s and f"i += {32 * CA.TA}" in s and "uint4 L[4][2];" in s   # G * TA stride, 4 packs in flight
  n = CA.render_ar_norm(dtypes.bfloat16, 2, 0, 32, 1 << 20, 8192, 1e-5)
  assert "row += 32" in n and "pk[j][0] = p4[i]; pk[j][1] = fz_ld_cg((const uint4*)(P.data[1] + sl) + o16 + i);" in n
  assert "fz_bfround(lo + fz_bf2f" in n and "__launch_bounds__(256)" in n
  # the fused norm's per-row arithmetic is render_rmsnorm's (same constants, reduction and rounding)
  from fusezap.serve.cuda import render_rmsnorm
  r = render_rmsnorm("x", dtypes.bfloat16, 8192, 1e-5, True)
  for frag in ("1.0f / sqrtf(red[0] * 0.0001220703125f + 9.999999747378752e-06f)", "for (int s = 16; s > 0; s >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, s);",
               "ss += lo * lo; ss += hi * hi;"):
    assert frag in r and frag in n, frag

def test_takes_and_routing(monkeypatch):
  bf, f32 = dtypes.bfloat16, dtypes.float32
  slot = 8 << 20
  assert CA.takes(16, bf, slot) and CA.takes(slot, bf, slot) and not CA.takes(slot + 16, bf, slot)
  assert not CA.takes(8 * 2 + 2, bf, slot) and not CA.takes(0, bf, slot) and not CA.takes(64, dtypes.int32, slot) and CA.takes(64, f32, slot)
  monkeypatch.setenv("FZ_TP_AR_MAX_MB", "1")
  assert CA.slot_bytes() == 1 << 20
  monkeypatch.setenv("FZ_TP_AR_BLOCKS", "999")
  assert CA.blocks() == CA.MAX_BLOCKS
  monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
  assert not CA.enabled_env()
  monkeypatch.delenv("NCCL_P2P_DISABLE")
  monkeypatch.setenv("FZ_TP_CUSTOM_AR", "0")
  assert not CA.enabled_env()

class _Fake:
  name, rank, world = "fake", 0, 2
  def __init__(self): self.calls = []
  def all_reduce(self, dev, out, inp, n, dt): self.calls.append(("base", n))
  def close(self, timeout=None): self.calls.append(("close",))

class _FakeCar:
  slot = 1 << 20
  def __init__(self): self.ok, self.calls = True, []
  def takes(self, nb, dt): return CA.takes(nb, dt, self.slot)
  def norm_takes(self, M, K, dt): return dt in (dtypes.bfloat16, dtypes.float16) and M * K * 2 <= self.slot
  def all_reduce(self, dev, out, inp, n, dt): self.calls.append(("car", n))
  def close(self): self.calls.append(("close",))

class _Dev:
  def __init__(self): self.copies = []
  def copy_async(self, dst, src, n): self.copies.append(n)

class _B:
  handle = 0

def test_tpcomm_dispatch(monkeypatch):
  monkeypatch.delenv("FZ_TP_AR_MODE", raising=False)
  c = dist.TPComm(_Fake())
  bf = dtypes.bfloat16
  c.all_reduce(None, _B(), _B(), 8192, bf)             # no custom all-reduce: the base collective
  c.car = car = _FakeCar()
  c.all_reduce(None, _B(), _B(), 8192, bf)             # 16 KB: custom
  c.all_reduce(None, _B(), _B(), 1 << 20, bf)          # 2 MB > the slot: base
  c.all_reduce(None, _B(), _B(), 8192, dtypes.int32)   # not a float type: base
  assert c.base.calls == [("base", 8192), ("base", 1 << 20), ("base", 8192)] and car.calls == [("car", 8192)]
  assert c.fuse_ok(64, 8192, bf) and not c.fuse_ok(65, 8192, bf) and not c.fuse_ok(64, 8192, dtypes.float32)
  monkeypatch.setenv("FZ_TP_AR_FUSE_ROWS", "8")
  assert not c.fuse_ok(9, 8192, bf) and c.fuse_ok(8, 8192, bf)
  c.mode = "base"
  c.all_reduce(None, _B(), _B(), 8192, bf)
  assert c.base.calls[-1] == ("base", 8192) and not c.fuse_ok(1, 8192, bf)
  c.mode, d = "skip", _Dev()
  c.all_reduce(d, _B(), _B(), 8192, bf)
  assert d.copies == [16384]
  c.mode, c.fuse = "auto", False
  assert not c.fuse_ok(1, 8192, bf)
  c.close()
  assert car.calls[-1] == ("close",) and c.base.calls[-1] == ("close",)
  monkeypatch.setenv("FZ_TP_AR_MODE", "base")
  assert dist.TPComm(_Fake()).mode == "base"

def test_fuse_ok_needs_tp_and_rows():
  class G: world, comm = 2, None
  class T:
    def __init__(self, shape, dt=dtypes.bfloat16): self.shape, self.dtype = shape, dt
  assert not dist.fuse_ok(None, T((1, 8192)))
  g = G()
  g.comm = dist.TPComm(_Fake())
  assert not dist.fuse_ok(g, T((1, 8192)))   # no custom all-reduce
  g.comm.car = _FakeCar()
  assert dist.fuse_ok(g, T((1, 8192))) and not dist.fuse_ok(g, T((1, 1, 8192))) and not dist.fuse_ok(g, T((128, 8192)))

# ---------------------------------------------------------------- 3. the algorithm, modelled

class _Sim:
  """W ranks' slots, signal flags and counters as the kernels use them, with the GPU-side ordering the stream gives:
  a rank's kernel n + 1 starts only after its kernel n ended; within a call, block b of a rank writes its share, signals,
  waits for block b of every peer, then reads. A scheduler interleaves the ranks' blocks at random (consistent with the
  barrier), and every read checks that the slot still holds the value written for this call."""
  def __init__(self, W: int, G: int, n: int, rng):
    self.W, self.G, self.n, self.rng = W, G, n, rng
    self.slot = np.zeros((W, 2, n), np.float64)          # data slots (two per rank)
    self.tag = np.full((W, 2, n), -1, np.int64)          # which call wrote each element
    self.start = np.zeros((W, G, W), np.int64)           # start[dst][block][src]
    self.cnt = np.zeros((W, G), np.int64)
  def run(self, calls: list[list[np.ndarray]], shares: list) -> list[list[np.ndarray]]:
    """calls[c][r]: rank r's input of call c; shares[c](b): the element indices block b handles in call c. Returns each
    rank's outputs. A rank's G blocks of its current call run concurrently (any interleaving); its next call starts
    when every block of this one is done; the ranks advance independently."""
    W, G = self.W, self.G
    call = [0] * W
    ph = [["write"] * G for _ in range(W)]   # per rank, per block: write -> wait -> read -> done
    outs = [[None] * len(calls) for _ in range(W)]
    acc = [[np.zeros(self.n) for _ in calls] for _ in range(W)]
    while any(c < len(calls) for c in call):
      r, b = int(self.rng.integers(W)), int(self.rng.integers(G))
      ci = call[r]
      if ci >= len(calls) or ph[r][b] == "done": continue
      flag = self.cnt[r, b] + 1
      sl, idx = flag & 1, shares[ci](b)
      if ph[r][b] == "write":
        self.slot[r, sl, idx] = calls[ci][r][idx]; self.tag[r, sl, idx] = ci
        for p in range(W):
          if p != r: self.start[p, b, r] = flag
        ph[r][b] = "wait"
      elif ph[r][b] == "wait":
        if all(self.start[r, b, p] >= flag for p in range(W) if p != r): ph[r][b] = "read"
      else:
        tot = np.zeros(len(idx))
        for p in range(W):
          assert (self.tag[p, sl, idx] == ci).all(), f"rank {r} call {ci} block {b}: rank {p}'s slot was overwritten"
          tot += self.slot[p, sl, idx]
        acc[r][ci][idx] = tot
        self.cnt[r, b] = flag
        ph[r][b] = "done"
        if all(x == "done" for x in ph[r]):   # the kernel ends: the next call's blocks may start
          outs[r][ci] = acc[r][ci]
          call[r], ph[r] = ci + 1, ["write"] * G
    return outs

@pytest.mark.parametrize("W,G", [(2, 1), (2, 4), (4, 3), (8, 2)])
def test_double_buffered_one_shot_model(W, G):
  """60 calls of three kinds (whole-message grid-stride, rows per block, a single row) on W ranks with blocks of each
  rank's kernel run in order but the ranks interleaved at random: every read sees the value of its own call (a slot is
  never rewritten while a peer may read it) and every rank's sum is the plain sum."""
  rng = np.random.default_rng(W * 10 + G)
  n = 48
  kinds = [lambda b: np.arange(b, n, G), lambda b: np.arange(b * 4, n, G * 4) if b * 4 < n else np.arange(0),
           lambda b: np.arange(n) if b == 0 else np.arange(0)]
  shares, calls = [], []
  for c in range(60):
    shares.append(kinds[int(rng.integers(3))])
    calls.append([rng.standard_normal(n) for _ in range(W)])
  outs = _Sim(W, G, n, rng).run(calls, shares)
  for c in range(60):
    want = np.zeros(n)
    covered = np.zeros(n, bool)
    for b in range(G): covered[shares[c](b)] = True
    for r in range(W): want += calls[c][r]
    for r in range(W): np.testing.assert_allclose(outs[r][c][covered], want[covered])

class _FakeCuda:
  """libcuda stand-in: cuIpcGetMemHandle writes a handle with NULs in it (as real ones have); cuIpcOpenMemHandle
  succeeds only for exactly those 64 bytes, as the driver does (the first M3 run: CUDA_ERROR_INVALID_VALUE)."""
  def __init__(self): self.inits, self.handles, self.opened = 0, {}, []
  def cuInit(self, flags): self.inits += 1; return 0
  def cuIpcGetMemHandle(self, ph, ptr):
    import ctypes
    raw = bytes([7, 0, 0, 1]) + int(ptr.value).to_bytes(8, "little") + bytes(52)   # zeros after byte 1
    self.handles[raw] = int(ptr.value) + 0x1000
    ctypes.memmove(ctypes.addressof(ph._obj), raw, 64)
    return 0
  def cuIpcOpenMemHandle_v2(self, pp, h, flags):
    raw = CA.handle_bytes(h)
    if raw not in self.handles: return 1
    pp._obj.value = self.handles[raw]; self.opened.append(raw)
    return 0
  def cuDeviceGet(self, p, o): p._obj.value = o; return 0
  def cuDeviceCanAccessPeer(self, p, a, b): p._obj.value = 1; return 0

def test_ipc_handle_keeps_all_64_bytes(monkeypatch):
  """The IPC handle crosses the control channel as all 64 bytes, NULs included, and opens on the peer (the bug: a
  c_char field cut it at the first NUL); can_access_peer calls cuInit first (a fresh process failed in cuDeviceGet)."""
  import ctypes, pickle
  fake = _FakeCuda()
  monkeypatch.setattr(CA, "_cuda", lambda: fake)
  h = CA._IpcHandle()
  assert fake.cuIpcGetMemHandle(ctypes.byref(h), ctypes.c_uint64(0xabc000)) == 0
  hb = pickle.loads(pickle.dumps(CA.handle_bytes(h)))
  assert len(hb) == 64 and hb[1:3] == b"\0\0" and CA.handle_bytes(CA.handle_from(hb)) == hb
  class Dev:
    def ctx(self): pass
  cars = []
  for r in range(2):
    c = CA.CustomAR.__new__(CA.CustomAR)
    c.dev, c.rank, c.world, c.base, c.opened, c.ok = Dev(), r, 2, 0x100000 * (r + 1), [], False
    cars.append(c)
  hs = [c.handle() for c in cars]
  assert all(len(x) == 64 for x in hs)
  for c in cars: c.open(hs)
  assert cars[0].peers == [0x100000, 0x200000 + 0x1000] and cars[1].peers == [0x100000 + 0x1000, 0x200000] and all(c.ok for c in cars)
  with pytest.raises(ValueError, match="64 bytes"): CA.handle_from(hs[0][:2])
  assert CA.can_access_peer(0, 1) and fake.inits >= 1

def test_wrap_safe_counter_compare():
  """The barrier compares counters as (int)(seen - flag) < 0: right across the 2^32 wrap (a 70B server runs ~8000
  all-reduces a second: the u32 counters wrap after ~6 days)."""
  def waiting(seen: int, flag: int) -> bool: return ((seen - flag) & 0xFFFFFFFF) >= 1 << 31
  assert waiting(5, 6) and not waiting(6, 6) and not waiting(7, 6)
  assert waiting(0xFFFFFFFF, 0) and not waiting(0, 0xFFFFFFFF) and not waiting(1, 0) and waiting(0xFFFFFFFE, 0xFFFFFFFF)

# ---------------------------------------------------------------- 4. CUDA, two GPUs with peer access

def _cuda2_p2p() -> bool:
  if not (os.environ.get("FZ_DEVICE") or "").upper().startswith("CUDA"): return False
  try:
    from fusezap.device import Device
    return Device.available("CUDA:1") and CA.can_access_peer(0, 1) and CA.enabled_env()
  except Exception: return False
needs_p2p = pytest.mark.skipif(not _cuda2_p2p(), reason="needs FZ_DEVICE=CUDA, two GPUs with peer access")

@pytest.fixture(scope="module")
def group():
  g = dist.launch(2, "CUDA", on_dead=lambda *a: None)
  yield g
  g.close()

@needs_p2p
def test_custom_selftest(group):
  """Both ranks: the custom kernel's bits equal NCCL's (eager and in CUDA graphs), every rank's equal, the fused
  add + RMSNorm equals all-reduce + rmsnorm; at 256 / 512 rows (4 / 8 MB) the custom path still takes the message."""
  assert group.comm.car is not None, group.ar_info
  with group.spmd("call", "fusezap.serve.dist:selftest_custom"):
    r0 = dist.selftest_custom(group)
  (r1,) = group.barrier("selftest_custom")
  print("CAR_SELFTEST", json.dumps({"rank0": r0, "rank1": r1}))
  assert r0["ok"] and r1["ok"], (r0, r1)
  assert all(r["custom"] for r in r0["rows"]) and all("fused_eq" in r for r in r0["rows"] if r["rows"] <= 64)

@needs_p2p
def test_custom_latency_vs_nccl(group):
  """Time per all-reduce at decode and prefill sizes (bf16 rows x 8192), custom vs NCCL, inside replayed CUDA graphs."""
  res = []
  for rows in (1, 8, 32, 64, 256, 512):
    for mode in ("auto", "base"):
      with group.spmd("call", "fusezap.serve.dist:bench_allreduce", "bfloat16", rows * 8192, 50, mode, True):
        r = dist.bench_allreduce(group, "bfloat16", rows * 8192, 50, mode, True)
      group.barrier("bench")
      res.append({"rows": rows, **r})
      print("CAR_BENCH", json.dumps({"rows": rows, **r}))
  by = {(r["rows"], r["path"]): r["us"] for r in res}
  # M3: 8.78 vs 8.47 us at one row (back to back, the custom path never ran in serving since the script read this as a
  # failure); a latency check, not a correctness one: only far slower fails
  assert by[(1, "custom")] < 1.5 * by[(1, "nccl")], by
