"""The four 7B prefill linears at M = 2048 / 4096 / 8192: FuseZap's wgmma dispatch vs torch.matmul vs cuBLASLt, and
whether one PINNED cuBLASLt algo (no split-K) gives bit-identical rows across M (speed run 4 c32-analysis.md, fix 1 and
diagnostic 2). Two parts, each in its own process (FuseZap's driver-API context and torch's are kept apart):

  python bench/gemm_vs_cublas.py --part ours   [--M 2048,4096,8192] [--json out.json]   # FZ_WG_PICK_MAX applies
  python bench/gemm_vs_cublas.py --part cublas [--M 2048,4096,8192] [--json out.json]   # needs torch with CUDA

cublas part, per shape: torch.matmul (cuBLAS's own pick per M); cuBLASLt's heuristic top algo per M (its tile, split-K
and stages printed); and the M=2048 heuristic algo with split-K forced to 1, pinned and reused at every M
(cublasLtMatmulAlgoCheck first). Bit identity: X is one (8192, K) matrix; the first 2048 output rows of each M must be
the same bits (bf16 out, f32 accumulate). Timings rotate over weight copies larger than L2, CUDA events, median of 5.
"""
from __future__ import annotations
import argparse, ctypes, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

H, I, QKV = 3584, 18944, 4608
SHAPES = [("qkv", QKV, H), ("o", H, H), ("gate_up", 2 * I, H), ("down", H, I)]
L2 = 128 << 20

def part_ours(Ms: list[int]) -> list[dict]:
  from fusezap import Device
  import fusezap.device.cuda_kernels as ck
  from bench.gemm_prefill import bench
  dev = Device["CUDA"]
  print(f"ours: {dev.gpu_name}, FZ_WG_PICK_MAX={os.environ.get('FZ_WG_PICK_MAX', '(default)')}")
  out = []
  for M in Ms:
    for nm, N, K in SHAPES:
      d = ck.wg_pick(dev, "bf16", M, N)
      res = bench(dev, M, N, K, [d])
      t = res[0][0] if res else float("nan")
      tf = 2 * M * N * K / t / 1e6 if res else float("nan")
      print(f"  M={M:5d} {nm:8s} N={N:5d} K={K:5d}: {t:8.1f} us {tf:5.0f} TFLOP/s  {d}")
      out.append({"part": "ours", "M": M, "shape": nm, "N": N, "K": K, "us": t, "tflops": tf, "cfg": str(d)})
  return out

# ---------------------------------------------------------------- cuBLASLt through ctypes
COMPUTE_32F, R_32F, R_16BF = 68, 0, 14
DESC_TRANSA, DESC_TRANSB, PREF_MAX_WS = 3, 4, 1
CFG_ID, CFG_TILE, CFG_SPLITK, CFG_RED, CFG_SWZ, CFG_CUSTOM, CFG_STAGES, CFG_INNER, CFG_CLUSTER = range(9)

class Heur(ctypes.Structure):
  _fields_ = [("algo", ctypes.c_uint64 * 8), ("workspaceSize", ctypes.c_size_t), ("state", ctypes.c_int),
              ("wavesCount", ctypes.c_float), ("reserved", ctypes.c_int * 4)]

def _lt():
  import torch
  cands = []
  try:
    import nvidia.cublas as nc
    cands += glob.glob(os.path.join(list(nc.__path__)[0], "lib", "libcublasLt.so*"))
  except Exception: pass
  cands += glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcublasLt*.so*")) + ["libcublasLt.so.12", "libcublasLt.so"]
  for c in cands:
    try: return ctypes.CDLL(c), c
    except OSError: continue
  raise OSError("libcublasLt not found")

class Lt:
  def __init__(self):
    self.lib, self.path = _lt()
    self.h = ctypes.c_void_p()
    self._ok(self.lib.cublasLtCreate(ctypes.byref(self.h)), "create")
    import torch
    self.ws_size = 64 << 20
    self.ws = torch.empty(self.ws_size, dtype=torch.uint8, device="cuda")
  @staticmethod
  def _ok(st, what):
    if st != 0: raise RuntimeError(f"cuBLASLt {what}: status {st}")
  def descs(self, M: int, N: int, K: int):
    """C^T (N x M, col-major, ld N) = op_T(W as K x N col-major) @ (X^T as K x M col-major): row-major C = X @ W^T."""
    L = self.lib
    d = ctypes.c_void_p(); self._ok(L.cublasLtMatmulDescCreate(ctypes.byref(d), COMPUTE_32F, R_32F), "desc")
    t, n = ctypes.c_int32(1), ctypes.c_int32(0)
    self._ok(L.cublasLtMatmulDescSetAttribute(d, DESC_TRANSA, ctypes.byref(t), 4), "transa")
    self._ok(L.cublasLtMatmulDescSetAttribute(d, DESC_TRANSB, ctypes.byref(n), 4), "transb")
    def lay(rows, cols, ld):
      l = ctypes.c_void_p(); self._ok(L.cublasLtMatrixLayoutCreate(ctypes.byref(l), R_16BF, ctypes.c_uint64(rows), ctypes.c_uint64(cols), ctypes.c_int64(ld)), "layout")
      return l
    return d, lay(K, N, K), lay(K, M, K), lay(N, M, N)
  def heuristic(self, M, N, K, k=8) -> list[Heur]:
    L = self.lib
    d, a, b, c = self.descs(M, N, K)
    p = ctypes.c_void_p(); self._ok(L.cublasLtMatmulPreferenceCreate(ctypes.byref(p)), "pref")
    ws = ctypes.c_uint64(self.ws_size)
    self._ok(L.cublasLtMatmulPreferenceSetAttribute(p, PREF_MAX_WS, ctypes.byref(ws), 8), "pref ws")
    res, cnt = (Heur * k)(), ctypes.c_int(0)
    self._ok(L.cublasLtMatmulAlgoGetHeuristic(self.h, d, a, b, c, c, p, k, res, ctypes.byref(cnt)), "heuristic")
    return [res[i] for i in range(cnt.value) if res[i].state == 0]
  def cfg(self, algo, attr) -> int:
    v, w = ctypes.c_uint32(0), ctypes.c_size_t(0)
    self.lib.cublasLtMatmulAlgoConfigGetAttribute(ctypes.byref(algo), attr, ctypes.byref(v), 4, ctypes.byref(w))
    return int(v.value)
  def set_cfg(self, algo, attr, val: int) -> None:
    v = ctypes.c_uint32(val)
    self._ok(self.lib.cublasLtMatmulAlgoConfigSetAttribute(ctypes.byref(algo), attr, ctypes.byref(v), 4), "set cfg")
  def describe(self, algo) -> str:
    return (f"id {self.cfg(algo, CFG_ID)} tile {self.cfg(algo, CFG_TILE)} splitK {self.cfg(algo, CFG_SPLITK)} red {self.cfg(algo, CFG_RED)} "
            f"stages {self.cfg(algo, CFG_STAGES)} inner {self.cfg(algo, CFG_INNER)} cluster {self.cfg(algo, CFG_CLUSTER)} swz {self.cfg(algo, CFG_SWZ)}")
  def check(self, algo, M, N, K) -> bool:
    d, a, b, c = self.descs(M, N, K)
    r = Heur()
    return self.lib.cublasLtMatmulAlgoCheck(self.h, d, a, b, c, c, ctypes.byref(algo), ctypes.byref(r)) == 0 and r.workspaceSize <= self.ws_size
  def matmul(self, algo, x, w, y, M, N, K) -> None:
    import torch
    d, a, b, c = self.descs(M, N, K)
    al, be = ctypes.c_float(1.0), ctypes.c_float(0.0)
    st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    self._ok(self.lib.cublasLtMatmul(self.h, d, ctypes.byref(al), ctypes.c_void_p(w.data_ptr()), a, ctypes.c_void_p(x.data_ptr()), b,
                                     ctypes.byref(be), ctypes.c_void_p(y.data_ptr()), c, ctypes.c_void_p(y.data_ptr()), c,
                                     ctypes.byref(algo), ctypes.c_void_p(self.ws.data_ptr()), ctypes.c_size_t(self.ws_size), st), "matmul")

def _time(fn_per_w, ws, reps=5) -> float:
  import torch
  for f in fn_per_w: f()
  torch.cuda.synchronize()
  ts = []
  for _ in range(reps):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(4):
      for f in fn_per_w: f()
    e1.record(); torch.cuda.synchronize()
    ts.append(e0.elapsed_time(e1) * 1e3 / (4 * len(fn_per_w)))
  return float(np.median(ts))

def part_cublas(Ms: list[int]) -> list[dict]:
  import torch
  lt = Lt()
  print(f"cublas: {torch.cuda.get_device_name()}, torch {torch.__version__}, cuBLASLt {lt.path}")
  g = torch.Generator(device="cuda").manual_seed(0)
  out = []
  Mx = max(Ms)
  for nm, N, K in SHAPES:
    ncp = max(2, -(-L2 // (N * K * 2)))
    ws = [(torch.randn(N, K, device="cuda", generator=g) * 0.02).to(torch.bfloat16) for _ in range(ncp)]
    X = torch.randn(Mx, K, device="cuda", generator=g).to(torch.bfloat16)
    h0 = lt.heuristic(min(Ms), N, K)
    pinned = None
    if h0:   # the smallest M's top algo, split-K forced off, reused at every M
      pinned = (ctypes.c_uint64 * 8).from_buffer_copy(h0[0].algo)
      lt.set_cfg(pinned, CFG_SPLITK, 1)
    rows = {}
    for M in Ms:
      x = X[:M]
      ys = [torch.empty(M, N, device="cuda", dtype=torch.bfloat16) for _ in ws]
      tf = lambda t: 2 * M * N * K / t / 1e6
      t_torch = _time([lambda w=w, y=y: torch.matmul(x, w.t(), out=y) for w, y in zip(ws, ys)], ws)
      yt = torch.matmul(x, ws[0].t()).view(torch.int16)[:min(Ms)].clone()
      rec = {"part": "cublas", "M": M, "shape": nm, "N": N, "K": K, "torch_us": t_torch, "torch_tflops": tf(t_torch)}
      hs = lt.heuristic(M, N, K)
      if hs:
        a = (ctypes.c_uint64 * 8).from_buffer_copy(hs[0].algo)
        t_h = _time([lambda w=w, y=y: lt.matmul(a, x, w, y, M, N, K) for w, y in zip(ws, ys)], ws)
        lt.matmul(a, x, ws[0], ys[0], M, N, K); torch.cuda.synchronize()
        rec.update(heur_us=t_h, heur_tflops=tf(t_h), heur_algo=lt.describe(a))
        rows.setdefault("heur", []).append(ys[0].view(torch.int16)[:min(Ms)].clone())
      rows.setdefault("torch", []).append(yt)
      if pinned is not None and lt.check(pinned, M, N, K):
        t_p = _time([lambda w=w, y=y: lt.matmul(pinned, x, w, y, M, N, K) for w, y in zip(ws, ys)], ws)
        lt.matmul(pinned, x, ws[0], ys[0], M, N, K); torch.cuda.synchronize()
        rec.update(pinned_us=t_p, pinned_tflops=tf(t_p), pinned_algo=lt.describe(pinned))
        rows.setdefault("pinned", []).append(ys[0].view(torch.int16)[:min(Ms)].clone())
      elif pinned is not None: rec["pinned_us"] = None; rec["pinned_algo"] = "not supported at this M"
      print(f"  M={M:5d} {nm:8s}: torch {t_torch:8.1f} us {tf(t_torch):4.0f} TF | heuristic {rec.get('heur_us', float('nan')):8.1f} us "
            f"{rec.get('heur_tflops', 0):4.0f} TF [{rec.get('heur_algo', '-')}] | pinned {rec.get('pinned_us') or float('nan'):8.1f} us "
            f"{rec.get('pinned_tflops', 0):4.0f} TF [{rec.get('pinned_algo', '-')}]")
      out.append(rec)
    for k, rs in rows.items():
      same = all(bool(torch.equal(rs[0], r)) for r in rs[1:]) if len(rs) == len(Ms) else None
      print(f"  {nm:8s} {k:7s}: first {min(Ms)} rows bit-identical across M {Ms}: {same}")
      out.append({"part": "cublas", "shape": nm, "identity": k, "same_across_M": same})
  return out

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--part", choices=["ours", "cublas"], required=True)
  ap.add_argument("--M", default="2048,4096,8192")
  ap.add_argument("--json", default=None)
  a = ap.parse_args()
  Ms = sorted(int(m) for m in a.M.split(","))
  res = part_ours(Ms) if a.part == "ours" else part_cublas(Ms)
  if a.json:
    with open(a.json, "w") as f: json.dump(res, f, indent=1)

if __name__ == "__main__": main()
