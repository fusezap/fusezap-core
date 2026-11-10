"""Per-shape decode GEMM configs for the sm_90 swap-AB kernel (I57 M3b): a table keyed by (GPU, SMs, kind, N, K, batch),
filled by a timed autotune at engine start (once per device type and model shapes, cached on disk) or shipped below.

  e = lookup(dev, "bf16", N, K)                    # (NC, SK, {BX: S}) or None: wg_swap_default's rule decides
  autotune(dev, "bf16", shapes, Ms, log)           # LLMEngine start (CUDA sm_90, FZ_GEMM_AUTOTUNE=1): times candidates
  pick(times, bxs)                                 # the choice from {(BX, NC, S, SK): us} (pure; tests)

Why: the first M3 run's GEMM sweep (one TP=2 rank of Llama-3.3-70B on H200) timed one forward's linears at 63.9 ms at
M = 32 with the swap-AB defaults against 37.8 ms with the best config per shape (31.0 vs 19.9 ms at M = 1); the
defaults' rule (one CTA per SM, 8 stages, fewer on wide shapes) was fitted on an H100's 7B shapes, and several 70B
shapes ran at 0.5-1.3 TB/s of the H200's 4.8 (qkv at M = 32: 145 us with SK 1 vs 34 us with SK 3; down: SK 1 vs 2-3).

Determinism. The K split (SK) and the consumer warpgroups (NC) are chosen once per (N, K), never per M: SK changes the
order of a row's f32 sum and NC is kept with it as wg_swap_default always kept it; only the pipeline depth S (which never
changes a sum) varies with BX = M rounded up to 8. So a row's result still does not depend on the other rows of a batch
(test_wg_swap_rows_independent), the decode buckets agree, and the mixed forwards' redo_split_k sees the same SK. An
entry, once on disk, is used by every later run (first writer wins across processes: the TP ranks of a first start
tune at once and both end up with the entry written first), so a model's tokens do not change between starts. A config
other than the default changes the numerics of the split shapes the way any SK change does (another f32 summation
order, then one rounding): the accuracy gates (Invarix top-1 vs the float32 reference) are the check, as for the
heuristic's SK. FZ_GEMM_TABLE=0 ignores the table (the heuristic alone: the previous build's configs); FZ_ONE_K=1
(SK 1 everywhere) bypasses it too.

Order of precedence: the disk table (tuned on this machine type; FZ_CACHE_DIR/cuda_gemm_table.json), then SHIPPED, then
wg_swap_default's rule. The autotune runs only for shapes without a disk entry, N x K >= FZ_GEMM_AUTOTUNE_MIN (4M
elements: tiny test models keep the rule), within FZ_GEMM_AUTOTUNE_S seconds (240; a shape not finished is not written
and keeps the shipped entry / the rule this run). Two stages per shape: SK over {1, 2, 3, 4, 6, the rule's} at NC 1,
S in {4, the rule's}, at every decode BX; then NC in {1, 2} x S in {4, 6, 8, 12} at the chosen SK. Each stage's kernels
are compiled first in a thread pool (FZ_GEMM_AUTOTUNE_JOBS, 16) into the disk kernel cache; each candidate then runs on
the model's own weights (several layers in rotation, so the bytes exceed L2 as in a forward), timed with CUDA events
(best of 3).
"""
from __future__ import annotations
import contextlib, json, math, os, pathlib, time
from typing import Any, Callable
from fusezap.helpers import CACHE_DIR

VERSION = 1
MAX_BX = 64

# Shipped entries: key -> (NC, SK, {BX: S}). From the first M3 run's sweep (logs/gemm_sweep.log, 2026-10-26, bench/gemm_small.py
# --swap --quick, H200, NC 1 only) at M = 1 and 32 (the M = 8 block of that sweep timed identical kernels 2-5x slower
# than at M = 1: disturbed, not used); SK chosen to minimize the sum over both BX of the best time at that SK:
#   qkv 5120 x 8192: SK 3 (24.7 + 33.7 us vs the rule's SK 1: 33.5 + 145.0)   o 8192 x 4096: SK 1, S 8 (= the rule)
#   gate_up 28672 x 8192: SK 1, S 4 (122.3 + 279.9 vs 122.3 + 363.9)        down 8192 x 14336: SK 2 (75.9 + 169.7 vs 201.7 + 246.9)
#   lm_head 128256 x 8192: SK 1 (S 6 / 4: 592.4 + 885.9 vs 908.9 + 1207.1)
# One forward of a rank's linears at M = 1 / 32: the rule 31.0 / 63.9 ms, this table ~20.0 / ~41.7 ms (the sweep's own
# times; its absolute values exceed M2's whole 28.9 ms TPOT, so the sweep ran slow: the autotune's log is the measurement).
# Qwen2.5-72B's rank shares qkv (5120 x 8192) and o (8192 x 4096); its gate_up / down / lm_head are tuned at start.
SHIPPED: dict[str, tuple[int, int, dict[int, int]]] = {
  "NVIDIA H200|132|bf16|5120|8192|1": (1, 3, {8: 4, 32: 6}),
  "NVIDIA H200|132|bf16|8192|4096|1": (1, 1, {8: 8, 32: 8}),
  "NVIDIA H200|132|bf16|28672|8192|1": (1, 1, {8: 4, 32: 4}),
  "NVIDIA H200|132|bf16|8192|14336|1": (1, 2, {8: 6, 32: 8}),
  "NVIDIA H200|132|bf16|128256|8192|1": (1, 1, {8: 6, 32: 4}),
}

def enabled() -> bool: return os.environ.get("FZ_GEMM_TABLE", "1") != "0"
def bx_of(M: int) -> int: return min(MAX_BX, -(-max(M, 1) // 8) * 8)
def path() -> pathlib.Path: return pathlib.Path(os.environ.get("FZ_GEMM_TABLE_FILE") or CACHE_DIR / "cuda_gemm_table.json")

def device_key(dev: Any) -> str | None:
  """'<GPU name>|<SMs>' or None (a stand-in device without a name: tests of the rule)."""
  name = getattr(dev, "gpu_name", None)
  if not isinstance(name, str) or not name: return None
  from fusezap.device.cuda_kernels import sm_count
  return f"{name}|{sm_count(dev)}"
def key(dk: str, kind: str, N: int, K: int, Bt: int = 1) -> str: return f"{dk}|{kind}|{N}|{K}|{Bt}"

_DISK: list = []   # [path, {key: entry}] loaded once per process (put() keeps it current)
def _load() -> dict:
  p = path()
  if not _DISK or _DISK[0] != p:
    try:
      d = json.loads(p.read_text())
      ents = d.get("entries", {}) if d.get("version") == VERSION else {}
    except Exception: ents = {}
    _DISK[:] = [p, ents]
  return _DISK[1]
def reset() -> None: _DISK.clear()

def _as_tuple(e: Any) -> tuple[int, int, dict[int, int]] | None:
  if isinstance(e, tuple): return e
  try: return int(e["NC"]), int(e["SK"]), {int(b): int(s) for b, s in e["S"].items()}
  except Exception: return None

def lookup(dev: Any, kind: str, N: int, K: int, Bt: int = 1) -> tuple[int, int, dict[int, int]] | None:
  """(NC, SK, {BX: S}) for this shape: the disk table's entry, else SHIPPED's, else None. S may be missing for a BX
  (wg_swap_default then takes its rule's depth for that BX with the entry's NC / SK)."""
  if not enabled() or (dk := device_key(dev)) is None: return None
  k = key(dk, kind, N, K, Bt)
  e = _load().get(k)
  return _as_tuple(e) if e is not None else SHIPPED.get(k)

def tuned(k: str) -> bool: return k in _load()

def put(k: str, entry: dict) -> dict:
  """Store entry under k unless another process stored one first (then that one is returned and kept): the file is
  read, merged and replaced under an exclusive lock (fcntl), atomically (tmp + rename)."""
  p = path()
  p.parent.mkdir(parents=True, exist_ok=True)
  with open(p.with_suffix(".lock"), "a+") as lf:
    with contextlib.suppress(Exception):
      import fcntl
      fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
    try:
      d = json.loads(p.read_text())
      ents = d.get("entries", {}) if d.get("version") == VERSION else {}
    except Exception: ents = {}
    if k not in ents:
      ents[k] = entry
      tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
      tmp.write_text(json.dumps({"version": VERSION, "entries": ents}, indent=0, sort_keys=True))
      os.replace(tmp, p)
  _DISK[:] = [p, ents]
  return ents[k]

# ---------------------------------------------------------------- choice (pure)

INF = float("inf")
def _shared(times: dict, bxs: list[int], part: Callable[[tuple], int]) -> int | None:
  """The value of a shared parameter (part of a (BX, NC, S, SK) key) with the lowest sum over bxs of each BX's best time
  among its keys; only values timed at every BX count; ties go to the smaller value."""
  by: dict[int, dict[int, float]] = {}
  for kk, t in times.items():
    d = by.setdefault(part(kk), {})
    d[kk[0]] = min(d.get(kk[0], INF), t)
  tot = {v: sum(d[b] for b in bxs) for v, d in by.items() if all(b in d for b in bxs)}
  return min(tot, key=lambda v: (tot[v], v)) if tot else None

def pick(times: dict[tuple[int, int, int, int], float], bxs: list[int]) -> tuple[int, int, dict[int, int], dict[int, float]] | None:
  """times {(BX, NC, S, SK): us} -> (NC, SK, {BX: S}, {BX: us}): SK first (the sum over bxs of the best time at that SK),
  then NC at that SK, then each BX's fastest S (ties: the smaller S). None if no SK was timed at every BX."""
  times = {k: t for k, t in times.items() if math.isfinite(t)}
  if (sk := _shared(times, bxs, lambda k: k[3])) is None: return None
  t2 = {k: t for k, t in times.items() if k[3] == sk}
  if (nc := _shared(t2, bxs, lambda k: k[1])) is None: return None
  S, us = {}, {}
  for b in bxs:
    t, s = min((t, k[2]) for k, t in t2.items() if k[0] == b and k[1] == nc)
    S[b], us[b] = s, t
  return nc, sk, S, us

# ---------------------------------------------------------------- autotune (CUDA)

def _time_us(dev: Any, fs: list, reps: int, trials: int = 3) -> float:
  cu, best = dev.cu, INF
  for f in fs: f()
  dev.synchronize()
  e0, e1 = dev.ev_start, dev.ev_end
  for _ in range(trials):
    cu.cuEventRecord(e0, dev.stream)
    for _ in range(reps):
      for f in fs: f()
    cu.cuEventRecord(e1, dev.stream); cu.cuEventSynchronize(e1)
    best = min(best, float(cu.cuEventElapsedTime(e0, e1)[1]) * 1e3 / (reps * len(fs)))
  return best

def _precompile(dev: Any, kind: str, out: str, cfgs: list) -> None:
  """NVRTC for the candidates in a thread pool (FZ_GEMM_AUTOTUNE_JOBS, default 16), into the disk kernel cache that
  Kernel() then reads: the compiles, not the timings, are most of a cold start's autotune."""
  from concurrent.futures import ThreadPoolExecutor
  from fusezap.device.cuda_kernels import wgmma
  from fusezap.device.cuda_kernels._rt import compiler
  from fusezap.device.cuda_kernels import smem_limit
  c, lim = compiler(dev, "sm_90a"), smem_limit(dev)
  def one(cfg: Any) -> None:
    with contextlib.suppress(Exception): c.compile_cached(wgmma.render_swap(kind, out, cfg))
  cfgs = [x for x in dict.fromkeys(cfgs) if wgmma.swap_valid(x, kind) and wgmma.swap_smem_bytes(x) <= lim]
  jobs = int(os.environ.get("FZ_GEMM_AUTOTUNE_JOBS", "") or 16)
  with ThreadPoolExecutor(max(1, min(jobs, os.cpu_count() or 1))) as ex: list(ex.map(one, cfgs))

def tune_shape(dev: Any, kind: str, out: str, N: int, K: int, ws: list[int], bxs: list[int], deadline: float,
               rule: Callable[[int], Any]) -> tuple[dict, dict] | None:
  """Time the candidates of one (N, K) at every BX on the weights ws (device addresses of (N, K) matrices); returns
  (entry, {(BX, NC, S, SK): us}) or None when the deadline passed first. rule(BX): wg_swap_default's config at M = BX."""
  from fusezap.device.cuda_kernels import gemm, smem_limit, ESIZE, _new
  from fusezap.device.cuda_kernels import wgmma
  from fusezap.dtype import dtypes
  import numpy as np
  smem, nk = smem_limit(dev), -(-K * ESIZE[kind] // 128)
  mx = max(bxs)
  X = _new(dev.name, mx * K, dtypes.bfloat16 if kind == "bf16" else dtypes.float16)
  rng = np.random.default_rng(0)
  X.copyin(memoryview(rng.integers(0x3c00, 0x3c80, mx * K, dtype=np.uint16)).cast("B"))   # small positive values, as the sweep
  C = _new(dev.name, mx * N, dtypes.float32 if out == "f32" else dtypes.bfloat16)
  nw = max(1, min(len(ws), -(-(256 << 20) // (N * K * ESIZE[kind]))))   # enough layers' weights to stream past L2
  est = N * K * ESIZE[kind] / 3e6                                      # us per launch at 3 TB/s
  reps = max(1, math.ceil(1500 / (est * nw)))
  times: dict = {}
  def run(bx: int, nc: int, s: int, sk: int) -> None:
    k = (bx, nc, s, sk)
    if k in times: return
    if time.monotonic() > deadline: raise TimeoutError
    cfg = wgmma.SwapCfg(nc, bx, s, sk)
    if not wgmma.swap_valid(cfg, kind) or wgmma.swap_smem_bytes(cfg) > smem: return
    try:
      fs = [gemm(dev, C.handle, X.handle, w, bx, N, K, kind=kind, out=out, impl="wgswap", cfg=cfg, bind=True) for w in ws[:nw]]
      times[k] = _time_us(dev, fs, reps)
    except TimeoutError: raise
    except Exception: times[k] = INF
  rules = {b: rule(b) for b in bxs}
  sks = sorted({1, 2, 3, 4, 6, *(c.SK for c in rules.values())} & set(range(1, max(1, min(16, nk // 4)) + 1)) | {1})
  stage_a = [(b, rules[b].NC, rules[b].S, rules[b].SK) for b in bxs] + [(b, 1, s, sk) for b in bxs for sk in sks for s in sorted({4, rules[b].S})]
  try:
    _precompile(dev, kind, out, [wgmma.SwapCfg(nc, b, s, sk) for b, nc, s, sk in stage_a])
    for k in stage_a: run(*k)
    first = pick({k: t for k, t in times.items() if k[1] == 1}, bxs)
    if first is None: return None
    sk = first[1]
    stage_b = [(b, nc, s, sk) for b in bxs for nc in (1, 2) for s in (4, 6, 8, 12)]
    _precompile(dev, kind, out, [wgmma.SwapCfg(nc, b, s, k_) for b, nc, s, k_ in stage_b if (b, nc, s, k_) not in times])
    for k in stage_b: run(*k)
  except TimeoutError: return None
  ch = pick({k: t for k, t in times.items() if k[3] == sk}, bxs)
  if ch is None: return None
  nc, sk, S, us = ch
  dflt = {b: times.get((b, rules[b].NC, rules[b].S, rules[b].SK), INF) for b in bxs}
  entry = {"NC": nc, "SK": sk, "S": {str(b): s for b, s in S.items()}, "us": {str(b): round(t, 2) for b, t in us.items()},
           "rule_us": {str(b): round(t, 2) for b, t in dflt.items()}, "src": "autotune", "when": time.strftime("%Y-%m-%d %H:%M:%S")}
  return entry, times

def autotune(dev: Any, kind: str, shapes: list[tuple[str, int, int, str, list[int]]], Ms: list[int],
             log: Callable[[str], None] | None = None, budget_s: float | None = None) -> list[dict]:
  """shapes: (name, N, K, out, weight addresses); Ms: the decode row counts (their BX set is tuned). Shapes with a disk
  entry are skipped; the rest are tuned within budget_s (FZ_GEMM_AUTOTUNE_S, default 240) and stored (put). Returns one
  report per shape: name, N, K, status (cached / tuned / skipped / timeout), entry, seconds."""
  from fusezap.device.cuda_kernels import wg_swap_default
  from fusezap.helpers import getenv
  log = log or (lambda m: None)
  dk = device_key(dev)
  if dk is None or not enabled(): return []
  bxs = sorted({bx_of(M) for M in Ms if 0 < M <= MAX_BX})
  if not bxs: return []
  budget = float(os.environ.get("FZ_GEMM_AUTOTUNE_S", "") or 240) if budget_s is None else budget_s
  t_all, deadline, rep = time.monotonic(), time.monotonic() + budget, []
  mn = getenv("FZ_GEMM_AUTOTUNE_MIN", 1 << 22)
  for name, N, K, out, ws in shapes:
    k, t0 = key(dk, kind, N, K), time.monotonic()
    r = {"name": name, "N": N, "K": K, "status": "", "entry": None, "s": 0.0}
    if tuned(k): r.update(status="cached", entry=_load()[k])
    elif N * K < mn or not ws: r["status"] = "skipped"
    else:
      res = tune_shape(dev, kind, out, N, K, ws, bxs, deadline, lambda b: wg_swap_default(dev, kind, b, N, K, table=False))
      if res is None: r["status"] = "timeout"
      else: r.update(status="tuned", entry=put(k, res[0]))
    r["s"] = round(time.monotonic() - t0, 1)
    e = r["entry"]
    if e is not None:
      us, ru = sum(e.get("us", {}).values()), sum(v for v in e.get("rule_us", {}).values() if math.isfinite(v))
      log(f"fusezap: gemm table {name} {N}x{K}: {r['status']}: NC {e['NC']} SK {e['SK']} S {e['S']}"
          + (f": {us:.1f} us over BX {bxs} vs the rule's {ru:.1f} us" if us and ru else "") + f" ({r['s']} s)")
    else: log(f"fusezap: gemm table {name} {N}x{K}: {r['status']} ({r['s']} s)")
    rep.append(r)
  log(f"fusezap: gemm autotune {time.monotonic() - t_all:.1f} s for {len(shapes)} shapes (BX {bxs}; table {path()})")
  return rep
