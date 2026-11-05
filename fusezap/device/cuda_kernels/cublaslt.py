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
from dataclasses import dataclass
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
  def descs(self, M: int):
    if (d := self.desc.get(M)) is None: self.desc[M] = d = _descs(_LIB[0], M, self.N, self.K, self.kind, self.out)
    return d
  def launch(self, dev: Any, C: int, A: int, B: int, M: int) -> None:
    lib = _LIB[0]
    d, la, lb, lc = self.descs(M)
    al, be = ctypes.c_float(1.0), ctypes.c_float(0.0)
    dev.ctx()
    _ok(lib.cublasLtMatmul(self.h, d, ctypes.byref(al), ctypes.c_void_p(B), la, ctypes.c_void_p(A), lb, ctypes.byref(be),
                           ctypes.c_void_p(C), lc, ctypes.c_void_p(C), lc, ctypes.byref(self.algo), ctypes.c_void_p(self.ws.handle),
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
    if all(_supported(lib, h, plan, M) for M in Ms) and _selfcheck(device, dev, plan, Ms) is None: return plan, "ok"
  return None, "no algo passed the bit-identity check across M"

def _supported(lib: Any, h: Any, plan: Plan, M: int) -> bool:
  d, la, lb, lc = plan.descs(M)
  r = _Heur()
  return lib.cublasLtMatmulAlgoCheck(h, d, la, lb, lc, lc, ctypes.byref(plan.algo), ctypes.byref(r)) == 0 and r.workspaceSize <= WS_BYTES

def _selfcheck(device: str, dev: Any, plan: Plan, Ms: list[int]) -> str | None:
  """Every M: the rows of A (a permutation of the largest M's first M rows, so a row moves to another position and
  tile) must give the largest M's bits exactly. None: passed."""
  from fusezap.device import Buffer
  from fusezap.dtype import dtypes
  N, K, Mx = plan.N, plan.K, max(Ms)
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
  for M in Ms:
    rows = rng.permutation(Mx)[:M]
    if not np.array_equal(run(rows), ref[rows]): return f"M={M} differs"
  return None

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
      if log: log(f"fusezap: cuBLASLt {group} {N}x{K} {kind}->{out_}: {'pinned algo, bit-identical over M ' + str(sorted(_MS[gk])) if plan else 'not used (' + why + ')'} (cuBLASLt {version()})")
    out[(N, K, kind, out_)] = _PLANS[k]
  return out

def plan_for(device: str, M: int, N: int, K: int, kind: str, out: str) -> Plan | None:
  """The pinned plan for this GEMM, or None (FuseZap's kernel): only for a registered shape and a self-checked M."""
  if not _PLANS or not getenv("FZ_CUBLASLT", 1): return None
  for g in GROUPS:
    if g == "decode" and not os.environ.get("FZ_CUBLASLT_DECODE"): continue   # plans of an earlier engine in this process
    p = _PLANS.get((device, g, N, K, kind, out))
    if p is not None and M in p.Ms: return p
  return None
