"""N7 step 3/4: a device-time model of the serving kernels (a decoder layer: v1 = the ten members fused into one NEFF, v2* =
nki_kernels.LayerSpec; the head: final norm + lm_head + argmax), from the instructions the rendered kernel issues on NKISHIM
(Qwen2.5-0.5B's shapes; every nisa call recorded with its operands' shapes, the tiles it reads and writes and its source
line, so it belongs to a member kernel ("# part k" of a fused kernel) or a LayerSpec section ("# k."); record-only, nothing
is computed).

  uv run python scripts/nki_layer_cost.py [--rows 1,4] [--ctx 512,1024] [--variants v1,v2s,v2se,v2sw,v2swe] [--head] [--json out.json]
  uv run python scripts/nki_layer_cost.py --fit [--json fit.json]    # refit PARAMS to the trn1 measurements (MEASURED)
  uv run python scripts/nki_layer_cost.py --rows 1,4 --ctx 512 --variants v1,v1+bf16,v2,v2+bf16,v2s+bf16,v2se+bf16,v2s+bf16a   # I68: bf16

A "+bf16" variant has bf16 weights (FZ_NEURON_DTYPE=bf16: nki_bf16.LayerBF16, v1 = the bf16 members; "+bf16a" also
FZ_NEURON_BF16_ATTN=1): a matmul with a 2-byte moving operand costs BF16["col"] / BF16["stat"] of the float32 TensorE time
per moving / stationary column (the step 4 probes), and its DMAs move 2-byte elements (the traced HBM operands' dtype).

The model: a list schedule over the data dependencies (RAW / WAW on each tile and HBM buffer), in program order per engine
(TensorE "pe", VectorE "dve", ScalarE "act", GpSimd "pool") and DMA queue. Costs (PARAMS):
  TensorE    te_fix + te_col * moving columns + te_stat * stationary columns (a transpose of (P, F): F stationary, P moving)
  VectorE &c ve_fix + ve_el * free elements per partition (the widest operand)
  DMA        on its queue (round robin over `queues`; DMAs into one tile stay on that tile's queue and wait for each other,
             the tile-level dependency), dma_q of queue time, then dma_lat + bytes / hbm on one shared HBM link whose free
             time the transfers fill as they become ready (they interleave; runs < 512 B move whole 64 B lines); a DMA into a
             fresh tile waits for the last reader of the tile `depth` allocations back of the same name (the compiler's buffer
             rotation; fewer when a tile is big: floor(sbuf / bytes per partition))
  sync       per dependency between two engines; kfix per launch (kernel start and end)
N7 step 3 calibrated a different model on v1 alone (x0.72) and predicted v2 at 0.218 ms per layer; trn1 measured 0.323
(v2s 0.318). Step 4 fits PARAMS to all 43 trn1 measurements (the v1 members alone, v1 / v2 / v2s / v2a layers at rows 1, 4 and
512 / 768 / 1024 slots, a 4-layer fused NEFF, the head variants of steps 2 / 2b) and reports per kernel the critical path
(which section and engine the time is spent in). Weight-stationary (w) variants depend on te_stat, which only the step 4
probes (scripts/nki_n7_probe.py) measure directly: their predictions are printed for te_stat = the fit and two bounds.
"""
from __future__ import annotations
import argparse, json, math, os, re, sys, time
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusezap.device import nkishim as S

# N8 (I70): refitted (--fit --iters 1500, 2026-10-28) on the 50 trn1 measurements below, now including the 7 I68 bf16 layers
# (step 4's fit predated them and put the bf16 layers at ~0.67x of trn1), with one new parameter, dma_row (seconds per
# contiguous run of a DMA on the HBM link: a descriptor cost); rms log error 0.080 (step 4's PARAMS on the same 50: 0.165).
# The fit leaves dma_row at its lower bound and lowers hbm to 241 GB/s and te_col to 1.45 ns: the float32 layers are now
# DMA-bound with VectorE second (step 4's fit: TensorE-bound), the bf16 layers within 0.89-0.97x of trn1. The run length
# (1 KiB bf16 / 2 KiB float32 / 28 KiB head runs) does not separate in these measurements (every layer uses the same runs),
# so dma_row is not identified: what the model credits to wider DMAs (N8's FZ_NEURON_BF16_WN) is fewer DMAs per queue
# (dma_q 1.1 us each) and less tile rotation, not a per-run cost; the trn1 bench decides. Not reproduced: the head splits
# (v2:1024:4 0.88x, v2:2048:1 0.85x) and the heads at W <= 1024 (1.03-1.11x).
# seconds, bytes/s, bytes
PARAMS = {"te_col": 1.45e-09, "te_stat": 3.7e-11, "te_fix": 3.33e-08, "ve_el": 7.43e-10, "ve_fix": 1.39e-07, "dma_lat": 5e-07,
          "dma_q": 1.11e-06, "hbm": 2.41e+11, "sync": 3e-08, "kfix": 9.22e-06, "depth": 15, "sbuf": 1.4e+05, "dma_row": 1.14e-11}
# step 4's fit (43 measurements, no bf16 rows, no dma_row): te_col 2.02e-09, te_stat 6.64e-11, te_fix 4.06e-08, ve_el 9.81e-10,
# ve_fix 5.98e-08, dma_lat 5e-07, dma_q 5.21e-07, hbm 3.93e+11, sync 2.04e-08, kfix 1.15e-05, depth 16, sbuf 160 KiB
FIXED = {"queues": 16}
FIT_KEYS = ("te_col", "te_stat", "te_fix", "ve_el", "ve_fix", "dma_lat", "dma_q", "hbm", "sync", "kfix", "depth", "sbuf", "dma_row")
BOUNDS = {"te_col": (1.0e-9, 3e-9), "te_stat": (1e-11, 4e-9), "te_fix": (1e-8, 3e-7), "ve_el": (2e-10, 2e-9), "ve_fix": (3e-8, 5e-7),
          "dma_lat": (5e-7, 5e-6), "dma_q": (5e-8, 3e-6), "hbm": (2e11, 8.2e11), "sync": (2e-8, 1e-6), "kfix": (2e-6, 3e-5), "depth": (2, 16),
          "sbuf": (48 * 1024, 192 * 1024), "dma_row": (1e-11, 3e-8)}

# trn1 device ms per launch (nrtpy device benchmark): docs/status/nki-n7-step3-trn1 (layer_bench, fz_fuse4), -step2(b)-trn1 (head_bench)
MEASURED = {
  "v1 r1 c512": 0.396335, "v1 r1 c768": 0.435603, "v1 r1 c1024": 0.469207, "v1 r4 c512": 0.4577105, "v1 r4 c768": 0.500424, "v1 r4 c1024": 0.5343435,
  "v2 r1 c512": 0.3230865, "v2 r1 c768": 0.35636, "v2 r1 c1024": 0.361241, "v2 r4 c512": 0.3399855, "v2 r4 c768": 0.371417, "v2 r4 c1024": 0.3808745,
  "v2s r1 c512": 0.31768, "v2s r4 c512": 0.3289945, "v2a r1 c512": 0.3302045, "v2a r4 c512": 0.3377445,
  "member 0: fz_rmsnorm_add_1_896": 0.022712, "member 1: fz_matmul_1_1_1152_896": 0.031584, "member 2: E_19_2_32": 0.020088,
  "member 3: fz_kv_write_512_1_256": 0.0173965, "member 4: fz_attention_keyrange_2_7_512_64_64": 0.0853465, "member 5: fz_matmul_1_1_896_896": 0.0289385,
  "member 6: fz_rmsnorm_add_1_896": 0.0227385, "member 7: fz_matmul_1_1_9728_896": 0.1483605, "member 8: E_4864": 0.018065, "member 9: fz_matmul_1_1_896_4864": 0.101917,
  "fused4 v2s r1 c512": 1.245169,
  "head v1-logits r1": 2.157, "head v1-logits r4": 2.2179, "head v3 r1": 2.1708, "head v3 r4": 2.256,
  "head v2:1024:4 r1": 2.5927, "head v2:1024:4 r4": 2.626, "head v2:1024:4-logits r1": 2.5789, "head v2:1024:4-logits r4": 2.5884,
  "head v2:512:1 r1": 2.0802, "head v2:512:1 r4": 2.2091, "head v2:1024:1 r1": 2.0644, "head v2:1024:1 r4": 2.1291,
  "head v2:2048:1 r1": 2.7124, "head v2:2048:1 r4": 2.7625, "head v2:4096:1 r1": 3.8452, "head v2:4096:1 r4": 3.9297,
  # I68 bf16 weights on trn1 (docs/status bf16 job, nki_n7_layer_bench --dtype bf16 / bf16a): added after PARAMS was fitted
  "v1+bf16 r1 c512": 0.345, "v1+bf16 r4 c512": 0.412, "v2+bf16 r1 c512": 0.246, "v2+bf16 r4 c512": 0.265,
  "v2s+bf16 r1 c512": 0.2395, "v2s+bf16 r4 c512": 0.2484, "v2s+bf16a r1 c512": 0.2432,
}

# I68 bf16: a matmul whose moving operand is 2 bytes (nki_bf16: both operands bf16) costs this fraction of the float32
# TensorE time per moving / stationary column: the step 4 probes on trn1 (scripts/nki_n7_probe.py, ns per 1k weights):
# moving 3.288 bf16 vs 13.1 float32, stationary 2.275 vs 12.3. Fixed (not fitted). With step 4's PARAMS the bf16 layers came
# out at ~0.67x of trn1; N8's refit (PARAMS above, the bf16 rows included) puts them at 0.89-0.97x.
BF16 = {"col": 3.288 / 13.1, "stat": 2.275 / 12.3}

QWEN = dict(D=896, I=4864, Hq=14, Hkv=2, hd=64, V=151936)
# N8 (I71): --model 7b puts Qwen2.5-7B's shapes here (D 3584, I 18944, 28 / 4 heads, head_dim 128, vocab 152064). The model was
# fitted on Qwen2.5-0.5B kernels on trn1; at 7B its only measurement is I69 run 1's member path on inf2 (MEASURED_7B).
QWEN_7B = dict(D=3584, I=18944, Hq=28, Hkv=4, hd=128, V=152064)
# inf2.8xlarge, I69 run 1: bf16 members, FUSE=1, 4 decode rows, 1024 slots:
# 97.056 ms device per token over 29 launches (layer 0 + one-hot, 27 middle layers, the head)
MEASURED_7B = {"token v1+bf16 r4 c1024": 97.056}
REC: list = []
ALLOC: dict = {}
ENG = {"nc_matmul": "pe", "nc_transpose": "pe", "dma_copy": "dma", "activation": "act"}
ISA = ("dma_copy", "memset", "tensor_copy", "tensor_tensor", "tensor_scalar", "activation", "reciprocal", "tensor_reduce",
       "tensor_copy_predicated", "nc_matmul", "nc_transpose", "iota", "affine_select")

def install() -> None:
  """Record every nisa call (and every on-chip allocation's line) instead of executing it."""
  for nm in ISA:
    def g(*a, _nm=nm, **k):
      REC.append((_nm, a, k, sys._getframe(1).f_lineno))
    setattr(S.nisa, nm, g)
  orig = S.ndarray
  def nd(shape, dtype, buffer=S.SBUF, name="", address=None):
    t = orig(shape, dtype, buffer, name, address)
    if buffer in S.ON_CHIP: ALLOC[id(t.storage)] = (int(np.prod(t.shape[1:])) * dtype.itemsize, sys._getframe(1).f_lineno)
    return t
  S.nl.ndarray = nd

def _ts(a: tuple, k: dict) -> list:
  return [v for v in list(a) + list(k.values()) if isinstance(v, S.NkiTensor)]

def record(prog, sizes: list[int]) -> dict:
  """The kernel's instruction trace (record-only on NKISHIM) as compact tuples, with its source for sections and names."""
  REC.clear(); ALLOC.clear()
  from fusezap.dtype import to_np
  arrays = [np.zeros(max(1, s), to_np(prog.buf_dtypes.get(i)) if i in prog.buf_dtypes else np.uint8) for i, s in enumerate(sizes)]
  S.execute(prog, arrays)
  lines = prog.src.split("\n")
  secs = []
  for i, ln in enumerate(lines, 1):
    if m := re.match(r"\s*# part (\d+): (\S+)", ln): secs.append((i, f"{m.group(1)}: {m.group(2)}"))
    elif m := re.match(r"\s*# (\d+)\. (.*)", ln): secs.append((i, f"{m.group(1)}. {m.group(2)[:40]}"))
  sec_of = lambda ln: next((nm for st, nm in reversed(secs) if ln >= st), "setup")
  sid: dict = {}
  ids = lambda t: sid.setdefault(id(t.storage), len(sid))
  name_of = lambda ln: re.sub(r"[0-9_]+$", "", (re.match(r"\s*(\w+)\s*=", lines[ln - 1]) or re.match(r"(\w*)", "")).group(1) or "?")
  out, sec_names = [], []
  for nm, a, k, ln in REC:
    dst = k.get("dst", a[0] if a else None)
    ts = _ts(a, k)
    w = [dst] if isinstance(dst, S.NkiTensor) else []
    r = [t for t in ts if t is not dst] + ([dst] if k.get("accumulate") and dst is not None else [])
    eng = "pool" if k.get("engine") == S.engine.gpsimd else "act" if k.get("engine") == S.engine.scalar else ENG.get(nm, "dve")   # I77: engine=scalar
    c1 = c2 = 0.0
    pp, grp, fresh = 0, -1, False
    if nm == "nc_matmul":
      c1, c2 = k["moving"].shape[1], k["stationary"].shape[1]
      if k["moving"].dtype.itemsize == 2: c1, c2 = c1 * BF16["col"], c2 * BF16["stat"]   # I68: bf16 operands (the probes' rates)
    elif nm == "nc_transpose": c1, c2 = k["data"].shape[0], k["data"].shape[1]
    elif nm == "dma_copy":
      src = k["src"]
      hbm = src if src.buffer not in S.ON_CHIP else dst
      run = hbm.shape[-1] * hbm.dtype.itemsize if hbm.strides[-1] == 1 else hbm.dtype.itemsize
      b = hbm.size * hbm.dtype.itemsize
      c1 = b if run >= 512 else b * max(1.0, 64 / max(run, 1))
      c2 = b / max(run, 1)   # N8: the contiguous runs (one descriptor each), dma_row apiece on the link
      if dst.buffer in S.ON_CHIP and (al := ALLOC.get(id(dst.storage))) is not None:
        pp, grp = al[0], hash(name_of(al[1]))
    else: c1 = max((int(np.prod(t.shape[1:])) if t.ndim > 1 else 1) for t in ts) if ts else 1
    sec = sec_of(ln)
    if sec not in sec_names: sec_names.append(sec)
    out.append((eng, nm == "nc_transpose", float(c1), float(c2), tuple(ids(t) for t in r), tuple(ids(t) for t in w),
                ids(w[0]) if (nm == "dma_copy" and w and w[0].buffer in S.ON_CHIP) else -1, pp, grp, sec_names.index(sec), ln))
  return {"name": prog.name, "trace": out, "sections": sec_names, "n": len(out)}

def schedule(tr: dict, P: dict, crit: bool = False) -> dict:
  """The list schedule of one kernel trace under PARAMS P: device seconds (incl. kfix), per section the end and engine
  busy time, and (crit) the critical path's time per section and per engine."""
  Q, SB = FIXED["queues"], P["sbuf"]
  te_col, te_stat, te_fix, ve_el, ve_fix = P["te_col"], P["te_stat"], P["te_fix"], P["ve_el"], P["ve_fix"]
  lat, dq, hbm, sync, depth = P["dma_lat"], P["dma_q"], P["hbm"], P["sync"], max(1, int(P["depth"]))
  drow = P.get("dma_row", 0.0)
  ready: dict = {}          # storage -> (end of its last write, engine, instr)
  last_read: dict = {}      # storage -> (end of its latest read so far)
  eng_free: dict = defaultdict(float); eng_last: dict = {}
  q_free = [0.0] * Q; q_last = [-1] * Q
  tile_q: dict = {}
  busy_s: list = []; busy_e: list = []; busy_o: list = []   # the HBM link's busy intervals, sorted by start (non-overlapping)
  def link_fit(t: float, d: float, i: int) -> tuple[float, float, int]:
    """Book d seconds of the link from t on, in its free gaps (transfers interleave on the link; whoever is ready first
    goes first, not program order): (first start, end, the transfer it first waited behind or -1)."""
    import bisect
    j = bisect.bisect_right(busy_s, t) - 1
    owner = -1
    if j >= 0 and busy_e[j] > t: t, owner = busy_e[j], busy_o[j]
    j = bisect.bisect_left(busy_s, t)
    start, rem = None, d
    while rem > 1e-15:
      gap = (busy_s[j] if j < len(busy_s) else math.inf) - t
      if gap > 1e-15:
        seg = min(rem, gap)
        if start is None: start = t
        if j > 0 and busy_e[j - 1] >= t - 1e-15: busy_e[j - 1] = t + seg; busy_o[j - 1] = i   # extend the interval before
        else: busy_s.insert(j, t); busy_e.insert(j, t + seg); busy_o.insert(j, i); j += 1
        t += seg; rem -= seg
        if j < len(busy_s) and busy_s[j] <= t + 1e-15:   # the gap is full: merge with the next interval
          busy_e[j - 1] = busy_e[j]; busy_o[j - 1] = busy_o[j]; del busy_s[j], busy_e[j], busy_o[j]
          if rem > 1e-15: t = busy_e[j - 1]
        continue
      if j < len(busy_s): t, owner = max(t, busy_e[j]), busy_o[j] if owner < 0 else owner; j += 1
    return (start if start is not None else t), t, owner
  groups: dict = defaultdict(list)
  written: set = set()
  qi = 0
  tr_ = tr["trace"]
  n = len(tr_)
  ends = [0.0] * n; starts = [0.0] * n; why = [-1] * n
  for i, (eng, istr, c1, c2, rd, wr, dtile, pp, grp, sec, ln) in enumerate(tr_):
    t0, w = 0.0, -1
    for s in rd + wr:
      if (r := ready.get(s)) is not None:
        t = r[0] + (sync if r[1] != eng else 0.0)
        if t > t0: t0, w = t, r[2]
    if eng == "dma":
      if dtile >= 0 and dtile not in written:
        d = max(1, min(depth, int(SB // max(pp, 1))))
        lst = groups[grp]
        if len(lst) >= d and (fr := last_read.get(lst[-d])) is not None and fr[0] > t0: t0, w = fr[0], fr[1]
        lst.append(dtile)
      q = tile_q.get(dtile) if dtile >= 0 else None
      if q is None: q = qi % Q; qi += 1
      if dtile >= 0: tile_q[dtile] = q
      st = max(t0, q_free[q])
      if q_free[q] > t0: w = q_last[q]
      xs0 = st + lat / 2
      xs, xe, owner = link_fit(xs0, c1 / hbm + c2 * drow, i)   # the HBM link's free time from xs0 on (not program order)
      if xs > xs0 and owner >= 0: w = owner
      end = xe + lat / 2
      q_free[q] = st + dq; q_last[q] = i
      st = max(st, xs - lat / 2)
    else:
      if eng == "pe": busy = te_fix + c1 * te_col + c2 * te_stat
      else: busy = ve_fix + c1 * ve_el
      st = max(t0, eng_free[eng])
      if eng_free[eng] > t0: w = eng_last.get(eng, -1)
      end = st + busy
      eng_free[eng] = end; eng_last[eng] = i
    starts[i], ends[i], why[i] = st, end, w
    for s in rd:
      if (lr := last_read.get(s)) is None or end > lr[0]: last_read[s] = (end, i)
    for s in wr: ready[s] = (end, eng, i); written.add(s)
  total = (max(ends) if ends else 0.0) + P["kfix"]
  out = {"s": total}
  if crit:
    secs = tr["sections"]
    sec_end, busy = defaultdict(float), defaultdict(lambda: defaultdict(float))
    for i, x in enumerate(tr_):
      sec_end[secs[x[9]]] = max(sec_end[secs[x[9]]], ends[i])
      busy[secs[x[9]]][x[0]] += ends[i] - starts[i]
    path_sec, path_eng = defaultdict(float), defaultdict(float)
    i = int(np.argmax(ends)) if ends else -1
    t = ends[i] if i >= 0 else 0.0
    while i >= 0:   # walk back: the instruction's own time, then the wait before it (charged to its section, "wait")
      sec, eng = secs[tr_[i][9]], tr_[i][0]
      s = min(starts[i], t)
      path_sec[sec] += t - s; path_eng[eng] += t - s
      t, j = s, why[i]
      if j >= 0:
        e = min(ends[j], t)
        path_sec[sec] += t - e; path_eng["wait"] += t - e; t = e
      i = j
    path_sec["setup"] += t
    out.update(sec_end={k: v for k, v in sec_end.items()}, busy={k: dict(v) for k, v in busy.items()}, path_sec=dict(path_sec), path_eng=dict(path_eng))
  return out

# ---------------------------------------------------------------- the kernels

def layer_sizes(sp) -> list[int]:
  while not hasattr(sp, "N"): sp = sp.base   # LayerBF16 (I68) / LayerBF16W (N8): the LayerSpec's arguments (the same element counts)
  N, D, Hq, Hkv, hd, I, P = sp.N, sp.D, sp.Hq, sp.Hkv, sp.hd, sp.I, sp.P
  Nq, Ho, C2 = (Hq + 2 * Hkv) * hd, Hq * hd, 4 * Hkv * hd // 2
  sz = {"xo": N * D, "dn": N * D, "pool": P * C2, "xs": N * (Nq + hd), "os": N * Ho, "x": N * D, "d": N * D, "ln1": D, "wqkv": D * Nq,
        "bqkv": Nq, "cos": N * hd, "sin": N * hd, "wt": N * P, "meta": sp.base + 2 * N + P, "wo": Ho * D, "bo": D, "ln2": D,
        "wgu": D * 2 * I, "wd": I * D}
  return [sz[a] for a in sp.args()]

def layer_spec(rows: int, ctx: int, variant: str, base: int = 64):
  """LayerSpec of a LAYER_VARIANTS name; "<v>+bf16" / "<v>+bf16a": nki_bf16.LayerBF16 of it (I68); "<v>+bf16+w<cols>[p<k>][q]":
  nki_bf16_n8.LayerBF16W of that (N8: FZ_NEURON_BF16_WN=<cols>, _PF=<k>, _PFQO=1; e.g. v2se+bf16+w2048, v2se+bf16+w512p2)."""
  from fusezap.codegen import nki_kernels as nk
  variant, _, dt = variant.partition("+")
  dt, _, n8 = dt.partition("+")
  e = nk.LAYER_VARIANTS[variant]
  q = QWEN
  sp = nk.LayerSpec(rows, q["D"], q["Hq"], q["Hkv"], q["hd"], q["I"], ctx, 1e-6, 0.125, base, True, True, False, e["FZ_NEURON_LAYER_HEADS"],
                      e["FZ_NEURON_LAYER_ROPE"], e["FZ_NEURON_LAYER_MM"], e["FZ_NEURON_LAYER_EARLY"] == "1", int(e["FZ_NEURON_LAYER_PF"]),
                      tile=e.get("FZ_NEURON_LAYER_TILE") == "1", wcols=int(e.get("FZ_NEURON_LAYER_WCOLS") or nk.LAYER_WCOLS))
  if not dt: return sp
  from fusezap.codegen.nki_bf16 import LayerBF16
  assert dt in ("bf16", "bf16a"), f"variant suffix +{dt}: +bf16 or +bf16a"
  sp = LayerBF16(sp, dt == "bf16a")
  if not n8: return sp
  from fusezap.codegen.nki_bf16_n8 import LayerBF16W
  m = re.fullmatch(r"w(\d+)(?:p(\d+))?(q?)", n8)
  assert m, f"variant suffix +{n8}: w<cols>[p<blocks>][q]"
  return LayerBF16W(sp, int(m.group(1)), int(m.group(2) or 0), m.group(3) == "q")

def fused(parts: list) -> tuple:
  """parts: [(NKIProgram, [fused argument index per member argument])] -> (fused NKIProgram, the fused arguments' sizes)."""
  from fusezap.codegen.nki_kernels import FusedSpec
  n = 1 + max(j for _, am in parts for j in am)
  outs = sorted({am[o] for p, am in parts for o in p.outs})
  dts = {}
  for p, am in parts:
    for i, dt in p.buf_dtypes.items(): dts.setdefault(am[i], dt)
  sp = FusedSpec(tuple((p.name, p.src, tuple(am)) for p, am in parts), n, tuple(outs), tuple(sorted(dts.items())))
  return sp.render()

def fused_layers(rows: int, ctx: int, variant: str, k: int) -> tuple:
  """k LayerSpecs as one FusedSpec (FZ_NEURON_FUSE=k: layer i's xo / dn are layer i+1's x / d; cos, sin, wt, meta shared)."""
  sp = layer_spec(rows, ctx, variant)
  prog, names, sz = sp.render(), sp.args(), dict(zip(sp.args(), layer_sizes(sp)))
  shared, idx, sizes, parts = {}, {}, [], []
  def arg(key, size):
    if key not in idx: idx[key] = len(sizes); sizes.append(size)
    return idx[key]
  for li in range(k):
    am = []
    for a in names:
      if a in ("cos", "sin", "wt", "meta"): key = a
      elif a == "x": key = ("xo", li - 1) if li else "x_in"
      elif a == "d": key = ("dn", li - 1) if li else "d_in"
      else: key = (a, li)
      am.append(arg(key, sz[a]))
    parts.append((prog, am))
  return fused(parts), sizes

def head_prog(rows: int, variant: str, V: int | None = None, K: int = 896) -> tuple:
  """The serving head as one fused kernel: NormSpec (add) + lm_head (v1 / v3: render_matmul on K-major; v2 / v4 W:d:
  HeadSpec on head_tiles) + ArgmaxSpec (not with -logits)."""
  from fusezap.codegen import nki_kernels as nk
  V = V or QWEN["V"]
  base, logits = variant.removesuffix("-logits"), variant.endswith("-logits")
  norm = nk.NormSpec(rows, K, 1e-6, True).render()
  if base.startswith(("v2", "v4")):
    W, d = (int(x) for x in base.split(":")[1:3]) if ":" in base else (nk.HEAD_W, nk.HEAD_DMAS)
    mm = nk.HeadSpec(rows, V, K, W, d, base.startswith("v4")).render()
  elif base.startswith("b2"):   # N8: HeadBF16 (bf16 head tiles, W columns per block; "b2:2048")
    from fusezap.codegen.nki_bf16_n8 import HeadBF16
    mm = HeadBF16(rows, V, K, int(base.split(":")[1]) if ":" in base else 2048).render()
  elif base.startswith("b3"):   # I68: MatmulBF16 (the bf16 lm_head without N8)
    from fusezap.codegen.nki_bf16 import MatmulBF16
    mm = MatmulBF16(rows, V, K).render()
  else: mm = nk.render_matmul((1, rows, V, K, 0, K, 1, 0, V, 1, 0, V, 1, 0, 0))
  parts = [(norm, [0, 1, 4, 5, 6]), (mm, [2, 0, 7])] + ([] if logits else [(nk.ArgmaxSpec(rows, V).render(), [3, 2])])
  sizes = [rows * K, rows * K, rows * V, rows, rows * K, rows * K, K, V * K]
  return fused(parts), sizes

def v1_layer(rows: int, ctx: int, dt: str = ""):
  """v1's fused10 layer item from nki_n7_layer_bench's 3-layer Qwen-shaped model on NKISHIM (the members are renderer
  kernels too, so the model builds them): (fused program, sizes, [(member name, program)]). dt "bf16" / "bf16a": the bf16
  members (FZ_NEURON_DTYPE=bf16, "bf16a" also FZ_NEURON_BF16_ATTN=1)."""
  from fusezap.codegen.nki_kernels import FusedSpec
  from nki_n7_layer_bench import qwen_shaped, set_variant, steps
  saved = {k: os.environ.get(k) for k in ("FZ_NEURON_FUSE", "FZ_NEURON_FUSE_MAX", "FZ_NEURON_HEAD_FUSE", "FZ_NEURON_LAYER", "FZ_NEURON_DTYPE",
                                          "FZ_NEURON_BF16_ATTN", "FZ_NEURON_PLAN")}
  os.environ.update({"FZ_NEURON_PLAN": "0", "FZ_NEURON_FUSE": "1", "FZ_NEURON_FUSE_MAX": "0", "FZ_NEURON_HEAD_FUSE": "0", "FZ_NEURON_DTYPE": "bf16" if dt else "",
                     "FZ_NEURON_BF16_ATTN": "1" if dt == "bf16a" else "0"})
  set_variant("v1")
  real = S.nisa.__dict__.copy()
  for nm in ISA: setattr(S.nisa, nm, getattr(S.isa, nm))   # the model runs for real
  try:
    q = QWEN
    m = qwen_shaped("NKISHIM", rows, ctx, hidden=q["D"], inter=q["I"], heads=q["Hq"], kv=q["Hkv"], hd=q["hd"])
    steps(m, rows, 3)
    si = [it for it in m._neuron_steps.jits[(rows, rows)].captured.items if not it.const][1].si
  finally:
    for nm in ISA: setattr(S.nisa, nm, real[nm])
    for k, v in saved.items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = v
  sp = si.ast[0].spec
  sizes = [b.size for b in si.bufs[:sp.n]]
  members = [(f"member {k}: {p[0]}", FusedSpec((p,), sp.n, sp.outs, sp.dts).render()) for k, p in enumerate(sp.parts)]
  return sp.render(), sizes, members

def traces(fit: bool, rows=(1, 4), ctxs=(512, 1024), variants=("v1", "v2s"), head: bool = False) -> dict:
  """label -> trace: every measured kernel (fit) or the requested ones."""
  out: dict = {}
  want = set(MEASURED) if fit else set()
  for r in rows:
    for c in ctxs:
      for v in variants: want.add(f"{v} r{r} c{c}")
  t0 = time.perf_counter()
  v1_cfg = sorted({(m.group(1) or "", int(m.group(2)), int(m.group(3))) for k in want if (m := re.match(r"v1(\+bf16a?)? r(\d+) c(\d+)$", k))})
  for dt, r, c in v1_cfg:
    prog, sizes, members = v1_layer(r, c, dt[1:])
    out[f"v1{dt} r{r} c{c}"] = record(prog, sizes)
    if (dt, r, c) == ("", 1, 512) and fit:
      for nm, p in members: out[nm] = record(p, sizes)
  for k in sorted(want):
    if k in out or k.startswith(("v1 ", "v1+", "member", "head", "fused")): continue
    v, r, c = re.match(r"(\S+) r(\d+) c(\d+)$", k).groups()
    sp = layer_spec(int(r), int(c), v)
    out[k] = record(sp.render(), layer_sizes(sp))
  if fit or head:
    for k in [k for k in MEASURED if k.startswith("head")] + ([] if fit else [f"head {v} r{r}" for v in ("v4:1024:1", "v4:512:1", "b3", "b2:1024", "b2:2048", "b2:4096") for r in rows]):
      v, r = re.match(r"head (\S+) r(\d+)$", k).groups()
      if k not in out: out[k] = record(*head_prog(int(r), v))
  if fit: out["fused4 v2s r1 c512"] = record(*fused_layers(1, 512, "v2s", 4))
  print(f"recorded {len(out)} kernels in {time.perf_counter() - t0:.1f} s", file=sys.stderr)
  return out

# ---------------------------------------------------------------- fit

def loss(P: dict, trs: dict) -> float:
  e = [math.log(schedule(trs[k], P)["s"] * 1e3 / ms) for k, ms in MEASURED.items() if k in trs]
  return float(np.mean(np.square(e)))

def fit(trs: dict, P0: dict, iters: int = 900) -> dict:
  """Nelder-Mead on log(PARAMS[FIT_KEYS]) minimizing the mean squared log error over MEASURED."""
  lo, hi = (np.log([BOUNDS[k][j] for k in FIT_KEYS]) for j in (0, 1))   # physical ranges (an unbounded fit drifts to zero latencies)
  x0 = np.clip(np.log([P0[k] for k in FIT_KEYS]), lo, hi)
  f = lambda x: loss({**P0, **{k: float(np.exp(v)) for k, v in zip(FIT_KEYS, np.clip(x, lo, hi))}}, trs) + float(np.sum(np.square(x - np.clip(x, lo, hi))))
  n = len(x0)
  sim = [x0] + [x0 + np.eye(n)[i] * 0.4 for i in range(n)]
  val = [f(x) for x in sim]
  for it in range(iters):
    o = np.argsort(val); sim = [sim[i] for i in o]; val = [val[i] for i in o]
    c = np.mean(sim[:-1], axis=0)
    xr = c + (c - sim[-1]); fr = f(xr)
    if fr < val[0]:
      xe = c + 2 * (c - sim[-1]); fe = f(xe)
      sim[-1], val[-1] = (xe, fe) if fe < fr else (xr, fr)
    elif fr < val[-2]: sim[-1], val[-1] = xr, fr
    else:
      xc = c + 0.5 * (sim[-1] - c); fc = f(xc)
      if fc < val[-1]: sim[-1], val[-1] = xc, fc
      else:
        sim = [sim[0]] + [sim[0] + 0.5 * (s - sim[0]) for s in sim[1:]]; val = [val[0]] + [f(s) for s in sim[1:]]
    if it % 100 == 0: print(f"fit iter {it}: rms log error {math.sqrt(min(val)):.4f}", file=sys.stderr)
  best = np.clip(sim[int(np.argmin(val))], lo, hi)
  return {**P0, **{k: float(np.exp(v)) for k, v in zip(FIT_KEYS, best)}}

# ---------------------------------------------------------------- report

def report(trs: dict, P: dict, label: str) -> dict:
  r = schedule(trs[label], P, crit=True)
  ms = r["s"] * 1e3
  print(f"== {label} ({trs[label]['name']}): model {ms:.4f} ms" + (f", measured {MEASURED[label]:.4f}" if label in MEASURED else "") +
        f"; {trs[label]['n']} instructions")
  print(f"   {'section':<46} {'end us':>8} {'crit us':>8}  busy us by engine")
  for sec in trs[label]["sections"]:
    b = r["busy"].get(sec, {})
    print(f"   {sec:<46} {1e6 * r['sec_end'].get(sec, 0):8.1f} {1e6 * r['path_sec'].get(sec, 0):8.1f}  " +
          " ".join(f"{e} {1e6 * v:.1f}" for e, v in sorted(b.items())))
  print("   critical path by engine (us): " + ", ".join(f"{e} {1e6 * v:.1f}" for e, v in sorted(r["path_eng"].items(), key=lambda x: -x[1])) +
        f"; setup + kfix {1e6 * (r['path_sec'].get('setup', 0) + P['kfix']):.1f}")
  return {"model_ms": round(ms, 4), "measured_ms": MEASURED.get(label), "path_sec_us": {k: round(1e6 * v, 1) for k, v in r["path_sec"].items()},
          "path_eng_us": {k: round(1e6 * v, 1) for k, v in r["path_eng"].items()}, "instructions": trs[label]["n"]}

def head_7b(rows: int) -> tuple:
  """I69's bf16 head at 7B as one fused kernel: NormSpec (add) + MatmulBF16 lm_head (4 x 152064 x 3584, K-major bf16) + ArgmaxSpec."""
  from fusezap.codegen import nki_kernels as nk
  from fusezap.codegen.nki_bf16 import MatmulBF16
  V, K = QWEN["V"], QWEN["D"]
  parts = [(nk.NormSpec(rows, K, 1e-6, True).render(), [0, 1, 4, 5, 6]), (MatmulBF16(rows, V, K).render(), [2, 0, 7]), (nk.ArgmaxSpec(rows, V).render(), [3, 2])]
  return fused(parts), [rows * K, rows * K, rows * V, rows, rows * K, rows * K, K, V * K]

def predict_7b(trs: dict, P: dict, rows, ctxs, vs, res: dict) -> None:
  """N8: per token at Qwen2.5-7B's 28 layers, model (not measured): 28 x layer + the head; the member path's against I69 run 1."""
  for r in rows:
    k = f"head v1+bf16 r{r}"
    if k not in trs: trs[k] = record(*head_7b(r))
    hd_ms = schedule(trs[k], P)["s"] * 1e3
    for c in ctxs:
      line = []
      for v in vs:
        if (lk := f"{v} r{r} c{c}") not in trs: continue
        lm = schedule(trs[lk], P)["s"] * 1e3
        tok = 28 * lm + hd_ms
        res["kernels"].setdefault(lk, {})["token_7b_model_ms"] = round(tok, 2)
        line.append(f"{v} {lm:.3f} ms/layer -> {tok:.1f} ms/token")
      if line:
        meas = MEASURED_7B.get(f"token v1+bf16 r{r} c{c}")
        print(f"predict 7b rows {r} ctx {c} (head {hd_ms:.2f} ms): " + "; ".join(line) +
              (f"; measured (I69 run 1, v1+bf16 members, inf2) {meas} ms device per token" if meas else ""))

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--rows", default="1,4")
  ap.add_argument("--ctx", default="512,1024")
  ap.add_argument("--variants", default="v1,v2s,v2se,v2sw,v2swe")
  ap.add_argument("--head", action="store_true", help="also the head variants (v3, v2:W:1, v4:W:1)")
  ap.add_argument("--fit", action="store_true", help="refit PARAMS to MEASURED (prints the new PARAMS)")
  ap.add_argument("--iters", type=int, default=900)
  ap.add_argument("--queues", type=int, default=FIXED["queues"], help="DMA queues the schedule spreads DMAs over (FIXED)")
  ap.add_argument("--json", default="")
  ap.add_argument("--model", default="0.5b", choices=("0.5b", "7b"), help="7b: Qwen2.5-7B's shapes (N8; v1 then builds a 3-layer 7B-shaped model)")
  args = ap.parse_args()
  if args.model == "7b": QWEN.update(QWEN_7B)
  os.environ.setdefault("FZ_DEVICE", "NKISHIM")
  FIXED["queues"] = args.queues
  install()
  rows, ctxs, vs = tuple(int(x) for x in args.rows.split(",")), tuple(int(x) for x in args.ctx.split(",")), tuple(args.variants.split(","))
  trs = traces(args.fit, rows, ctxs, vs, args.head)
  P = dict(PARAMS)
  if args.fit:
    print(f"start: rms log error {math.sqrt(loss(P, trs)):.4f} (PARAMS)")
    P = fit(trs, P, args.iters)
    print("PARAMS = {" + ", ".join(f'"{k}": {v:.3g}' for k, v in P.items()) + "}")
  res: dict = {"params": P, "fixed": FIXED, "kernels": {}}
  if any(k in trs for k in MEASURED):
    errs = []
    print(f"{'measured kernel':<46} {'trn1 ms':>9} {'model ms':>9} {'model/trn1':>10}")
    for k, ms in MEASURED.items():
      if k not in trs: continue
      m = schedule(trs[k], P)["s"] * 1e3
      errs.append(math.log(m / ms))
      print(f"{k:<46} {ms:9.4f} {m:9.4f} {m / ms:10.3f}")
      res["kernels"][k] = {"measured_ms": ms, "model_ms": round(m, 4)}
    print(f"rms log error {math.sqrt(np.mean(np.square(errs))):.4f} over {len(errs)} measurements")
  for k in sorted(trs, key=lambda k: (not k.startswith("v"), k)):
    if k.startswith(("v", "fused")) or (args.head and k.startswith("head") and ("v4" in k or k.startswith("head b") or k in ("head v3 r1", "head v2:1024:1 r1"))):
      res["kernels"].setdefault(k, {}).update(report(trs, P, k))
  # predictions (labelled): weight-stationary forms under three te_stat values; per token at 24 layers
  print("--- predictions (model, not measured)")
  for k in sorted(trs):
    if "w" not in k.split(" ")[0][2:] and "v4" not in k: continue
    sens = {nm: schedule(trs[k], {**P, "te_stat": v})["s"] * 1e3 for nm, v in (("fit", P["te_stat"]), ("te_stat = te_col / 4", P["te_col"] / 4),
                                                                                  ("te_stat = te_col", P["te_col"]))}
    print(f"predict {k:<28} " + ", ".join(f"{nm}: {v:.4f} ms" for nm, v in sens.items()))
    res["kernels"].setdefault(k, {})["te_stat_sensitivity_ms"] = {nm: round(v, 4) for nm, v in sens.items()}
  if args.model == "7b": predict_7b(trs, P, rows, ctxs, vs, res)
  for r in rows:
    line = []
    for c in ctxs:
      if (a := res["kernels"].get(f"v1 r{r} c{c}", {}).get("model_ms")) is None: continue
      line.append(f"ctx {c}: v1 {a:.3f}" + "".join(f", {v} {res['kernels'][k]['model_ms']:.3f} ({24 * (res['kernels'][k]['model_ms'] - a):+.2f} ms/token)"
                                                for v in vs if v != "v1" and (k := f"{v} r{r} c{c}") in res["kernels"]))
    if line: print(f"predict rows {r}: " + "; ".join(line))
  if args.json:
    with open(args.json, "w") as f: json.dump(res, f, indent=1)

if __name__ == "__main__": main()
