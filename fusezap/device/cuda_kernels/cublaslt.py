"""cuBLASLt for the prefill GEMMs (speed run 4 round 12; docs/status/speed-run4/c32-analysis.md, round 11 revision fix 2a).

  enable(device, shapes, Ms, log)  -> {(N, K, kind, out): Plan or None}  at engine start: one pinned algo per shape
  plan_for(device, M, N, K, kind, out) -> Plan | None                    what GemmCall asks per GEMM
  Plan.launch(dev, C, A, B, M)                                           C (M, N) = A (M, K) @ B (N, K)^T on dev.stream

One algorithm per (N, K, kind, out) is chosen once (cuBLASLt's heuristic top pick at the largest registered M, split-K
forced to 1) and pinned for every registered M (the engine's prefill sizes). Before a plan is used it must pass a
self-check on the device: random A, B; every registered M runs the same rows in a different order and position, and
every row must come out bit-identical to the largest M's (so a token's prefill result does not depend on the bucket
or on the other rows: batched == alone). A shape that fails the check, an M outside the registered set (mixed
forwards, the echo-logprob scorer, decode rows), or no libcublasLt (FZ_CUBLASLT=0 also) keeps FuseZap's own wgmma
kernel, so the fallback is always the previous build's arithmetic. Bits differ from our kernel's (another K order):
switching to cuBLASLt shifts near-tie tokens once, as the kpt switch did; the library version is part of the result
(logged once, `version`).

cuBLASLt runs on the CUDA runtime, which uses the device's primary context (the one fusezap.device.cuda retains), and
launches on the stream the jit is recording (a stream capture records it into the forward's CUDA graph; a fixed algo
with a preallocated workspace allocates nothing, which capture requires).
"""
from __future__ import annotations
import ctypes, glob, os
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from fusezap.helpers import getenv

COMPUTE_32F, R_32F, R_16F, R_16BF = 68, 0, 2, 14
DESC_TRANSA, DESC_TRANSB, PREF_MAX_WS = 3, 4, 1
CFG_ID, CFG_TILE, CFG_SPLITK, CFG_RED = 0, 1, 2, 3
WS_BYTES = 32 << 20
_DT = {"bf16": R_16BF, "f16": R_16F, "f32": R_32F}

class _Heur(ctypes.Structure):
  _fields_ = [("algo", ctypes.c_uint64 * 8), ("workspaceSize", ctypes.c_size_t), ("state", ctypes.c_int),
              ("wavesCount", ctypes.c_float), ("reserved", ctypes.c_int * 4)]

_LIB: list = []       # [CDLL or None, path, version]
_PLANS: dict = {}     # (device, group, N, K, kind, out) -> Plan | None
_MS: dict = {}        # (device, group) -> set of registered M

def _load() -> Any:
  if _LIB: return _LIB[0]
  cands = [os.environ["FZ_CUBLASLT_LIB"]] if os.environ.get("FZ_CUBLASLT_LIB") else []
  try:
    import nvidia.cublas as nc   # the pip wheel torch / vLLM use
    for d in nc.__path__: cands += sorted(glob.glob(os.path.join(d, "lib", "libcublasLt.so*")))
  except Exception: pass
  cands += sorted(glob.glob("/usr/local/cuda/lib64/libcublasLt.so*")) + ["libcublasLt.so.12", "libcublasLt.so.13", "libcublasLt.so"]
  for c in cands:
    try:
      lib = ctypes.CDLL(c)
      lib.cublasLtGetVersion.restype = ctypes.c_size_t
      _LIB[:] = [lib, c, int(lib.cublasLtGetVersion())]
      return lib
    except (OSError, AttributeError): continue
  _LIB[:] = [None, None, None]
  return None

def version() -> int | None: return _LIB[2] if _LIB else None
def available() -> bool: return bool(getenv("FZ_CUBLASLT", 1)) and _load() is not None

def _ok(st: int, what: str) -> None:
  if st != 0: raise RuntimeError(f"cuBLASLt {what}: status {st}")

@dataclass
class Plan:
  N: int
  K: int
  kind: str
  out: str
  algo: Any          # (c_uint64 * 8), pinned
  desc: Any          # per M: (matmul desc, A, B, C layouts)
  ws: Any            # workspace Buffer
  h: Any             # cublasLt handle
  Ms: frozenset = frozenset()   # the M values the self-check covered
  algos: dict = field(default_factory=dict)   # W15 FZ_CUBLASLT_PER_M: M -> its own algo, self-checked bit-identical to the pinned one
  def algo_for(self, M: int) -> Any: return self.algos.get(M, self.algo)
  def descs(self, M: int):
    if (d := self.desc.get(M)) is None: self.desc[M] = d = _descs(_LIB[0], M, self.N, self.K, self.kind, self.out)
    return d
  def launch(self, dev: Any, C: int, A: int, B: int, M: int) -> None:
    lib = _LIB[0]
    d, la, lb, lc = self.descs(M)
    al, be = ctypes.c_float(1.0), ctypes.c_float(0.0)
    dev.ctx()
    _ok(lib.cublasLtMatmul(self.h, d, ctypes.byref(al), ctypes.c_void_p(B), la, ctypes.c_void_p(A), lb, ctypes.byref(be),
                           ctypes.c_void_p(C), lc, ctypes.c_void_p(C), lc, ctypes.byref(self.algo_for(M)), ctypes.c_void_p(self.ws.handle),
                           ctypes.c_size_t(WS_BYTES), ctypes.c_void_p(int(dev.stream))), "matmul")

def _descs(lib: Any, M: int, N: int, K: int, kind: str, out: str):
  """Row-major C (M, N) = A (M, K) @ W (N, K)^T as column-major C^T (N x M, ld N) = op_T(W: K x N, ld K) @ (A^T: K x M, ld K)."""
  d = ctypes.c_void_p(); _ok(lib.cublasLtMatmulDescCreate(ctypes.byref(d), COMPUTE_32F, R_32F), "desc")
  t, n = ctypes.c_int32(1), ctypes.c_int32(0)
  _ok(lib.cublasLtMatmulDescSetAttribute(d, DESC_TRANSA, ctypes.byref(t), 4), "transa")
  _ok(lib.cublasLtMatmulDescSetAttribute(d, DESC_TRANSB, ctypes.byref(n), 4), "transb")
  def lay(dt, rows, cols, ld):
    l = ctypes.c_void_p(); _ok(lib.cublasLtMatrixLayoutCreate(ctypes.byref(l), dt, ctypes.c_uint64(rows), ctypes.c_uint64(cols), ctypes.c_int64(ld)), "layout")
    return l
  return d, lay(_DT[kind], K, N, K), lay(_DT[kind], K, M, K), lay(_DT[out], N, M, N)

def _handle(dev: Any) -> Any:
  h = getattr(dev, "_cublaslt", None)
  if h is None:
    dev.ctx()
    h = ctypes.c_void_p(); _ok(_LIB[0].cublasLtCreate(ctypes.byref(h)), "create")
    dev._cublaslt = h
  return h

def _make(device: str, dev: Any, N: int, K: int, kind: str, out: str, Ms: list[int]) -> tuple["Plan | None", str]:
  """Pin an algo for the shape and self-check it over Ms; (plan or None, why)."""
  from fusezap.device import Buffer
  from fusezap.dtype import dtypes
  lib, h = _LIB[0], _handle(dev)
  Mx = max(Ms)
  d, la, lb, lc = _descs(lib, Mx, N, K, kind, out)
  p = ctypes.c_void_p(); _ok(lib.cublasLtMatmulPreferenceCreate(ctypes.byref(p)), "pref")
  wsz = ctypes.c_uint64(WS_BYTES)
  _ok(lib.cublasLtMatmulPreferenceSetAttribute(p, PREF_MAX_WS, ctypes.byref(wsz), 8), "pref ws")
  res, cnt = (_Heur * 8)(), ctypes.c_int(0)
  _ok(lib.cublasLtMatmulAlgoGetHeuristic(h, d, la, lb, lc, lc, p, 8, res, ctypes.byref(cnt)), "heuristic")
  heur = [res[i] for i in range(cnt.value) if res[i].state == 0]
  if not heur: return None, "no heuristic algo"
  ws = Buffer(device, WS_BYTES, dtypes.uint8).ensure_allocated()
  for hr in heur:   # the first algo that, with split-K off, runs at every M and passes the bit check
    algo = (ctypes.c_uint64 * 8).from_buffer_copy(hr.algo)
    one = ctypes.c_uint32(1)
    if lib.cublasLtMatmulAlgoConfigSetAttribute(ctypes.byref(algo), CFG_SPLITK, ctypes.byref(one), 4) != 0: continue
    plan = Plan(N, K, kind, out, algo, {}, ws, h, frozenset(Ms))
    if all(_supported(lib, h, plan, M) for M in Ms):
      chk = _checker(device, dev, plan, Mx)
      if chk(Ms) is None:
        if per_m_on(): _per_m(plan, [M for M in Ms if M != Mx], chk)
        return plan, "ok"
  return None, "no algo passed the bit-identity check across M"

def per_m_on() -> bool: return bool(getenv("FZ_CUBLASLT_PER_M", 1))   # W15: default on (0 = one pinned algo for every M)

def _heuristic(M: int, N: int, K: int, kind: str, out: str, h: Any, n: int = 4) -> list:
  """cuBLASLt's top n algos at this M, split-K forced to 1 (the K order of the pinned algos)."""
  lib = _LIB[0]
  d, la, lb, lc = _descs(lib, M, N, K, kind, out)
  p = ctypes.c_void_p(); _ok(lib.cublasLtMatmulPreferenceCreate(ctypes.byref(p)), "pref")
  wsz = ctypes.c_uint64(WS_BYTES)
  _ok(lib.cublasLtMatmulPreferenceSetAttribute(p, PREF_MAX_WS, ctypes.byref(wsz), 8), "pref ws")
  res, cnt = (_Heur * 8)(), ctypes.c_int(0)
  _ok(lib.cublasLtMatmulAlgoGetHeuristic(h, d, la, lb, lc, lc, p, 8, res, ctypes.byref(cnt)), "heuristic")
  out_ = []
  for i in range(cnt.value):
    if res[i].state != 0: continue
    a = (ctypes.c_uint64 * 8).from_buffer_copy(res[i].algo)
    one = ctypes.c_uint32(1)
    if lib.cublasLtMatmulAlgoConfigSetAttribute(ctypes.byref(a), CFG_SPLITK, ctypes.byref(one), 4) == 0: out_.append(a)
  return out_[:n]

def _per_m(plan: Plan, Ms: list[int], chk: Any) -> None:
  """W15 FZ_CUBLASLT_PER_M (default on; 0 = off): each M below the largest gets cuBLASLt's own top algo at that M (split-K 1)
  when it runs there and its rows are bit-identical to the pinned algo's at the largest M (chk([M]): the same check
  as across M); else M keeps the pinned algo. Pod A (2x H200): heuristic algos per M gave the same bits as the M = 2048
  one on every 70B rank shape (gemm_vs_cublas "heur ... bit-identical across M: True"), at cuBLAS-per-M speed (qkv
  M = 128: 28.7 vs 101 us pinned)."""
  lib = _LIB[0]
  for M in sorted(Ms):
    for a in _heuristic(M, plan.N, plan.K, plan.kind, plan.out, plan.h):
      if bytes(a) == bytes(plan.algo): break   # the pinned one is already the pick here
      plan.algos[M] = a
      try: ok = _supported(lib, plan.h, plan, M) and chk([M]) is None
      except Exception: ok = False
      if ok: break
      del plan.algos[M]

def _supported(lib: Any, h: Any, plan: Plan, M: int) -> bool:
  d, la, lb, lc = plan.descs(M)
  r = _Heur()
  return lib.cublasLtMatmulAlgoCheck(h, d, la, lb, lc, lc, ctypes.byref(plan.algo_for(M)), ctypes.byref(r)) == 0 and r.workspaceSize <= WS_BYTES

def _selfcheck(device: str, dev: Any, plan: Plan, Ms: list[int]) -> str | None:
  """Every M: the rows of A (a permutation of the largest M's first M rows, so a row moves to another position and
  tile) must give the largest M's bits exactly. None: passed."""
  return _checker(device, dev, plan, max(Ms))(Ms)

def _checker(device: str, dev: Any, plan: Plan, Mx: int) -> Any:
  """The self-check's data built once (random A, W on the device, the reference rows of the largest M Mx at the
  algo for Mx); the returned check(Ms) runs each M at plan.algo_for(M) -> None (passed) or why."""
  from fusezap.device import Buffer
  from fusezap.dtype import dtypes
  N, K = plan.N, plan.K
  rng = np.random.default_rng(1234)
  bf = lambda a: (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16) if plan.kind == "bf16" else np.asarray(a, np.float16).view(np.uint16)
  A = bf(rng.standard_normal((Mx, K)).astype(np.float32))
  W = bf((rng.standard_normal((N, K)) * 0.05).astype(np.float32))
  eo = 4 if plan.out == "f32" else 2
  wb = Buffer(device, N * K * 2, dtypes.uint8).ensure_allocated(); wb.copyin(memoryview(np.ascontiguousarray(W)).cast("B"))
  def run(rows: np.ndarray) -> np.ndarray:
    M = len(rows)
    ab = Buffer(device, M * K * 2, dtypes.uint8).ensure_allocated(); ab.copyin(memoryview(np.ascontiguousarray(A[rows])).cast("B"))
    cb = Buffer(device, M * N * eo, dtypes.uint8).ensure_allocated()
    plan.launch(dev, cb.handle, ab.handle, wb.handle, M)
    dev.synchronize()
    out = np.empty(M * N * eo, np.uint8); cb.copyout(memoryview(out))
    return out.reshape(M, N * eo)
  ref = run(np.arange(Mx))
  def check(Ms: list[int]) -> str | None:
    for M in Ms:
      rows = rng.permutation(Mx)[:M]
      if not np.array_equal(run(rows), ref[rows]): return f"M={M} differs"
    return None
  return check

GROUPS = ("prefill", "decode")

def enable(device: str, shapes: list[tuple[int, int, str, str]], Ms: list[int], log: Any = None, group: str = "prefill") -> dict:
  """Register the sizes Ms of a group ("prefill": the prefill buckets; "decode": the decode buckets 1 .. max_num_seqs)
  and build (or reuse) a pinned, self-checked plan per (N, K, kind, out) covering every M registered in that group. A
  group has its own algo (picked at its largest M), so a decode plan never changes the prefill bits or vice versa."""
  if not available(): return {}
  from fusezap.device import Device
  assert group in GROUPS, group
  dev, gk = Device[device], (device, group)
  _MS.setdefault(gk, set()).update(Ms)
  out = {}
  for N, K, kind, out_ in shapes:
    k = (device, group, N, K, kind, out_)
    if k not in _PLANS or (_PLANS[k] is not None and not _MS[gk] <= _PLANS[k].Ms):
      try: plan, why = _make(device, dev, N, K, kind, out_, sorted(_MS[gk]))
      except Exception as e: plan, why = None, f"error: {e}"
      _PLANS[k] = plan
      if log: log(f"fusezap: cuBLASLt {group} {N}x{K} {kind}->{out_}: {'pinned algo, bit-identical over M ' + str(sorted(_MS[gk])) + (f'; own algo (same bits) at M {sorted(plan.algos)}' if plan.algos else '') if plan else 'not used (' + why + ')'} (cuBLASLt {version()})")
    out[(N, K, kind, out_)] = _PLANS[k]
  return out

def extend(device: str, Ms: list[int], log: Any = None, group: str = "prefill") -> bool:
  """W15 (FZ_PREFILL_MIN=128): add the sizes Ms to every pinned plan of the group without changing its algo. Each plan
  must run at the new sizes and pass the self-check over its sizes plus Ms (the same algo, so the bits at the old sizes
  are unchanged); if any plan fails, nothing changes and False is returned (the caller then drops those sizes, so a
  prompt's bits never depend on which bucket ran it). No plans (no library, FZ_CUBLASLT=0): True, FuseZap's kernels
  run every M."""
  if not available(): return True
  from fusezap.device import Device
  new = set(Ms)
  plans = [p for k, p in _PLANS.items() if k[0] == device and k[1] == group and p is not None]
  for p in plans:
    add = sorted(new - p.Ms)
    if not add: continue
    why = None
    try:
      if not all(_supported(_LIB[0], p.h, p, M) for M in add): why = f"the pinned algo does not run at M {add}"
      else:
        chk = _checker(device, Device[device], p, max(p.Ms | new))
        why = chk(sorted(p.Ms | new))
        if why is None and per_m_on(): _per_m(p, [M for M in add if M != max(p.Ms | new)], chk)
    except Exception as e: why = f"error: {e}"
    if why is not None:
      if log: log(f"fusezap: cuBLASLt {group} {p.N}x{p.K} {p.kind}->{p.out}: not extended to M {add} ({why})")
      return False
  for p in plans: p.Ms = frozenset(p.Ms | new)
  _MS.setdefault((device, group), set()).update(new)
  if log and plans: log(f"fusezap: cuBLASLt {group}: {len(plans)} pinned plans extended to M {sorted(new)}, bit-identical over M {sorted(plans[0].Ms)}"
                        + "".join(f"; {p.N}x{p.K} own algo (same bits) at M {sorted(p.algos)}" for p in plans if p.algos))
  return True

def keep_rule(t_lt: dict, t_fz: dict) -> bool:
  """W15 FZ_CUBLASLT_CHECK: keep a shape's pinned plan only if its time summed over the checked sizes (the smallest and
  the largest prefill size) is at most FuseZap's kernel's. Pod A (2x H200, 15 Nov): the algos pinned at M = 2048 ran the
  70B TP rank's GEMMs 1.6-4x slower than FuseZap's kernel at M = 128 / 256 (Qwen gate_up 529 vs 130 us at 128) and
  gained at most ~15% at 2048."""
  return sum(t_lt.values()) <= KEEP_MARGIN * sum(t_fz.values())

KEEP_MARGIN = 1.03   # near a tie keep the plan: pod A2's Llama gate_up was within 0.4% and its two ranks decided differently

def check_speed(device: str, Ms: list[int], log: Any = None, group: str = "prefill") -> dict:
  """FZ_CUBLASLT_CHECK=1 (default off): time every pinned plan of the group and FuseZap's own kernel (the dispatch's
  pick) at the smallest and largest of Ms, random weights streamed past L2, and drop the plan where keep_rule says so:
  that shape then runs FuseZap's kernel at every M (its sums do not depend on M, so batched == alone still holds; bits
  differ from the plan's, as before cuBLASLt). Call before any forward is captured. Returns {(N, K, kind, out): (t_lt,
  t_fz, kept)}. A timing near a tie can decide differently on another start; the log line records each decision."""
  if not available() or not Ms: return {}
  import numpy as np
  from fusezap.device import Device, Buffer
  from fusezap.dtype import dtypes
  from fusezap.device.cuda_kernels import gemm
  from fusezap.device.cuda_kernels.gemm_table import _time_us
  dev, sizes, res = Device[device], sorted({min(Ms), max(Ms)}), {}
  for k, p in list(_PLANS.items()):
    if p is None or k[0] != device or k[1] != group: continue
    N, K, kind, out = p.N, p.K, p.kind, p.out
    eo, nw = (4 if out == "f32" else 2), max(1, min(4, -(-(256 << 20) // (N * K * 2))))
    rng = np.random.default_rng(0)
    ws = [Buffer(device, N * K * 2, dtypes.uint8).ensure_allocated() for _ in range(nw)]
    for w in ws: w.copyin(memoryview(rng.integers(0x3c00, 0x3c80, N * K, dtype=np.uint16)).cast("B"))
    A = Buffer(device, max(sizes) * K * 2, dtypes.uint8).ensure_allocated()
    A.copyin(memoryview(rng.integers(0x3c00, 0x3c80, max(sizes) * K, dtype=np.uint16)).cast("B"))
    C = Buffer(device, max(sizes) * N * eo, dtypes.uint8).ensure_allocated()
    t_lt, t_fz = {}, {}
    try:
      for M in sizes:
        reps = max(2, int(2000 / max(1.0, 2 * M * N * K / 4e8)))   # ~2 ms per trial at ~400 TFLOP/s
        t_lt[M] = _time_us(dev, [lambda w=w, M=M: p.launch(dev, C.handle, A.handle, w.handle, M) for w in ws], reps)
        t_fz[M] = _time_us(dev, [gemm(dev, C.handle, A.handle, w.handle, M, N, K, kind=kind, out=out, bind=True) for w in ws], reps)
    except Exception as e:
      if log: log(f"fusezap: cuBLASLt check {N}x{K}: timing failed ({type(e).__name__}: {e}); plan kept")
      continue
    kept = keep_rule(t_lt, t_fz)
    if not kept: _PLANS[k] = None
    res[(N, K, kind, out)] = (t_lt, t_fz, kept)
    if log: log(f"fusezap: cuBLASLt check {group} {N}x{K} {kind}->{out}: pinned {', '.join(f'M={m} {t:.1f}' for m, t in t_lt.items())} us vs "
                f"FuseZap {', '.join(f'M={m} {t:.1f}' for m, t in t_fz.items())} us: {'kept' if kept else 'dropped (FuseZap kernel at every M)'}")
  return res

def plan_for(device: str, M: int, N: int, K: int, kind: str, out: str) -> Plan | None:
  """The pinned plan for this GEMM, or None (FuseZap's kernel): only for a registered shape and a self-checked M."""
  if not _PLANS or not getenv("FZ_CUBLASLT", 1): return None
  for g in GROUPS:
    if g == "decode" and not os.environ.get("FZ_CUBLASLT_DECODE"): continue   # plans of an earlier engine in this process
    p = _PLANS.get((device, g, N, K, kind, out))
    if p is not None and M in p.Ms: return p
  return None
