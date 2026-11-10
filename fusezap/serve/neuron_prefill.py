"""I77: the prefill kernels (fusezap/codegen/nki_prefill.py) in the serving forward on NEURON, their switches, the device
self-test that gates them, and the bigger prefill chunks.

FZ_NEURON_PREFILL_KERNEL=v2 (default; v1: the member kernels, I75's path): a prefill forward's decoder layer (StepLayout.prefill,
bf16 K-major weights, no gate / up / down biases) runs as PFQKV, KVWriteSpec, PFAttn, PFLinear (o_proj) and PFMLP with the
same all-reduces as the members (dist.all_reduce, fused into the prefill NEFF under FZ_NEURON_TP_PREFILL=fused), so a
layer is 5 kernels + 2 collectives instead of ~12 + 2; with FZ_NEURON_PFK_EXP=exact the forward's values are the members'
bit for bit (the default exact), with act (ScalarE exp in attention and silu) within the bf16 bar.
FZ_NEURON_PFK_HEAD=split (default with v2 under TP; `wide`: no split; `off`: the members' lm_head): the prefill's lm_head
as PFLinear on this rank's half of the vocabulary columns of its (replicated) K-major lm_head, the halves gathered by one
all-reduce of the zero-padded logits (exact): each rank streams half of the 1.09 GB head per prefill (I75: 6.5 ms).
FZ_NEURON_PFK_EXP (exact default: bitwise; act faster), FZ_NEURON_PFK_EVICT, FZ_NEURON_PFK_BANKS / FZ_NEURON_PFK_WN (PSUM banks
per weight block, the widest weight DMA): nki_prefill's docstring.
FZ_NEURON_PREFILL_SIZES=128,512 (I77; the default on NEURON since I78, 128: one size): the prefill forwards' row counts (ascending; the largest becomes
max_prefill_tokens), so a burst of prompts or a long prompt runs as one 512-row forward whose weights are read once
(with v2: PFLinear / PFMLP feed every 128-row block from one weight tile) instead of four 128-row forwards. Which size a
step runs is chosen by estimated cost of the oldest pending prompt (FZ_NEURON_PREFILL_COST_MS, prefill_cost_ms;
Scheduler._fit_cost): a 200-token prompt runs as two 128-row forwards (~106 ms) rather than one 512-row one (~195 ms
exact), and separate short prompts are never bundled into a 512-row forward (I78: that delayed each); FZ_PREFILL_FIT=0: always the
largest that the step's rows need. Only with v2 on
(LLMEngine keeps the smallest size otherwise: the members would re-stream the weights per 128 rows); a forward of more than
128 rows groups FZ_NEURON_PFK_FUSE_BIG (7) layers per NEFF (big_fuse).

The self-test (selftest, NEURON only): before the first capture, each kernel at the model's real shapes is compiled
(neuronx-cc) and run once on random inputs against its float64 reference (nki_prefill.ref_*); a failure (compile or
values) turns v2 off on every rank (`set_ok`), logged, and the forward runs the members. Under tensor parallelism
neuron_tp.setup runs it on every rank and keeps v2 only if all pass (the ranks must run the same kernels)."""
from __future__ import annotations
import os, sys, time
from typing import Any
import numpy as np

MODES = ("v1", "v2")
_STATE: dict = {"ok": None, "why": "", "override": None, "knobs": None}
# the self-test's fallbacks, most capable first: a level whose kernels fail to compile or to match their reference on the
# device is replaced by the next (ScalarE PSUM evictions -> VectorE; then the exact exp too), and the ranks take the most
# conservative level any of them needed
FALLBACKS = ({}, {"evict": "vector"}, {"evict": "vector", "exp": "exact"})

def kernel_mode() -> str:
  """FZ_NEURON_PREFILL_KERNEL (v2 default | v1), or the mode set_mode put (mirrored on every TP rank)."""
  m = _STATE["override"] or (os.environ.get("FZ_NEURON_PREFILL_KERNEL", "") or "v2").strip().lower()
  if m not in MODES: raise ValueError(f"FZ_NEURON_PREFILL_KERNEL={m!r}: one of {', '.join(MODES)}")
  return m

def enabled() -> bool:
  """v2 asked for and not refused by the self-test."""
  return kernel_mode() == "v2" and _STATE["ok"] is not False

def set_mode(g: Any, mode: str | None) -> None:
  """Every rank (op "call", and rank 0 itself): use `mode` for the forwards captured from now on (None: the env again)."""
  _STATE["override"] = mode

def set_ok(g: Any, ok: bool, why: str = "", knobs: dict | None = None) -> None:
  """Every rank: the self-test's verdict (False: v2 off, the members run) and the fallback knobs it settled on."""
  _STATE["ok"], _STATE["why"], _STATE["knobs"] = bool(ok), why, dict(knobs) if knobs else None

def head_mode(tp: bool) -> str:
  """FZ_NEURON_PFK_HEAD: split (default under TP), wide (default without: there the bf16 head is HeadBF16 already, whose
  values MatmulBF16's are), off."""
  v = (os.environ.get("FZ_NEURON_PFK_HEAD", "") or ("split" if tp else "off")).strip().lower()
  if v not in ("split", "wide", "off"): raise ValueError(f"FZ_NEURON_PFK_HEAD={v!r}: split, wide or off")
  return v

DEFAULT_SIZES = "128,512"   # I78: the real chip's default (NKISHIM / NKISIM: one size unless set)

def prefill_sizes(max_prefill_tokens: int, device: str = "") -> list[int] | None:
  """FZ_NEURON_PREFILL_SIZES (NEURON): the prefill forwards' row counts, or None (one size: max_prefill_tokens). Unset:
  DEFAULT_SIZES on the real chip (device NEURON), None elsewhere; FZ_NEURON_PREFILL_SIZES=128: 128 rows only (I77)."""
  v = os.environ.get("FZ_NEURON_PREFILL_SIZES", "").strip()
  if not v and device.split(":")[0].upper() == "NEURON": v = DEFAULT_SIZES
  if not v: return None
  s = sorted({int(x) for x in v.split(",") if x.strip()})
  if not s or s[0] <= 0 or any(x % 16 for x in s): raise ValueError(f"FZ_NEURON_PREFILL_SIZES={v!r}: positive multiples of 16")
  return s

# I77 run 1 (inf2.8xlarge, Qwen2.5-7B bf16 TP=2, in process, wall to the token): one prefill forward's ms by its rows, per
# exp mode. act: 42.6 ms at 128 rows (128-slot prefix), 130 ms for a 512-token prompt in one 512-row forward. exact: 49.5 /
# 53.7 / ~58 ms at 128 rows (128 / 256 / 512 slots); 512 rows not measured (scripts/nki_prefill_cost.py's model: ~168 ms
# device at 512 slots, + ~4 ms host)
# I78 (the --pfk2 job, served, exact): 51-69 ms at 128 rows and 174-241 ms at 512 by prefix; 53 / 195 typical
COST_MS = {"act": {128: 43.0, 512: 130.0}, "exact": {128: 53.0, 512: 195.0}}

def _parse_cost(v: str) -> dict[int, float]:
  out = {}
  for part in v.split(","):
    if not part.strip(): continue
    n, _, ms = part.partition(":")
    try: n_, ms_ = int(n), float(ms)
    except ValueError: raise ValueError(f"FZ_NEURON_PREFILL_COST_MS={v!r}: rows:ms pairs, e.g. 128:50,512:170") from None
    if n_ <= 0 or ms_ <= 0: raise ValueError(f"FZ_NEURON_PREFILL_COST_MS={v!r}: positive rows and ms")
    out[n_] = ms_
  return out

def prefill_cost_ms(sizes: list[int], exp: str | None = None) -> dict[int, float]:
  """The estimated time (ms) of one prefill forward of each size, for the scheduler's cost sizing (Scheduler._fit_cost):
  FZ_NEURON_PREFILL_COST_MS (e.g. 128:50,512:170) over the I77 run 1 defaults of the exp mode in effect (COST_MS; exp: the
  self-test's knobs, else FZ_NEURON_PFK_EXP). A size in neither is interpolated linearly in its rows between the nearest
  known sizes (extrapolated from the two nearest outside them; proportionally from a single one)."""
  exp = exp or _knobs().get("exp", "exact")
  known = {**COST_MS.get(exp, COST_MS["exact"]), **_parse_cost(os.environ.get("FZ_NEURON_PREFILL_COST_MS", ""))}
  pts = sorted(known.items())
  def at(n: int) -> float:
    if n in known: return known[n]
    if len(pts) == 1: return pts[0][1] * n / pts[0][0]
    lo = [p for p in pts if p[0] < n]
    hi = [p for p in pts if p[0] > n]
    a, b = (lo[-1], hi[0]) if lo and hi else (pts[0], pts[1]) if not lo else (pts[-2], pts[-1])
    return max(1e-3, a[1] + (b[1] - a[1]) * (n - a[0]) / (b[0] - a[0]))
  return {int(n): round(at(int(n)), 3) for n in sizes}

def big_fuse(N: int, pk: int) -> int:
  """Layers per prefill NEFF for an N-row forward on the prefill kernels: pk (FZ_NEURON_TP_PREFILL_FUSE) up to 128 rows (FZ_NEURON_PFK_BIG_ROWS);
  above, FZ_NEURON_PFK_FUSE_BIG (default 7: four NEFF launches per 28-layer chunk, ~0.3 ms each) when pk is larger, so a
  512-row NEFF is ~1/4 of the instructions of a whole-chunk one (neuronx-cc time and memory) and fz.jit's plan can share
  the groups' intermediates (~0.4 instead of ~1.5 GB per pool prefix at 7B). 0 or `all`: pk."""
  big_rows = int(os.environ.get("FZ_NEURON_PFK_BIG_ROWS", "") or 128)   # (tests: a smaller threshold)
  if N <= big_rows or not pk: return pk
  v = (os.environ.get("FZ_NEURON_PFK_FUSE_BIG", "") or "7").strip().lower()
  if v in ("all", "0"): return pk
  return min(pk, int(v))

def _knobs() -> dict:
  """The kernels' knobs: the env (nki_prefill.pfk_*), then whatever fallback the self-test settled on."""
  from fusezap.codegen.nki_prefill import pfk_banks, pfk_evict, pfk_exp, pfk_wn
  return {"banks": pfk_banks(), "wn": pfk_wn(), "exp": pfk_exp(), "evict": pfk_evict(), **(_STATE.get("knobs") or {})}

def layer_ok(model: Any, l: Any, N: int, x: Any) -> bool:
  """The prefill kernels cover this layer: bf16 K-major weights, no gate / up / down biases, float32 rows, head_dim <= 128
  dividing 512 (RoPE per 512-column tile)."""
  from fusezap.dtype import dtypes
  c = model.cfg
  return (getattr(model, "nbf16", False) and l.wqkv.dtype == dtypes.bfloat16 and l.bgu is None and l.bd is None and
          x.dtype == dtypes.float32 and isinstance(N, int) and N > 0 and c.head_dim <= 128 and 512 % c.head_dim == 0 and
          c.n_heads % c.n_kv_heads == 0)

def specs(N: int, D: int, Hq: int, Hkv: int, hd: int, I: int, P: int, eps: float, scale: float, base: int, add: bool, qkv_bias: bool,
          o_bias: bool, kn: dict | None = None) -> dict:
  """The layer's kernel specs (nki_prefill) for N rows at pool prefix P."""
  from fusezap.codegen import nki_prefill as K
  from fusezap.codegen.nki_kernels import KVWriteSpec
  kn = kn or _knobs()
  Wx = (Hq + 2 * Hkv + 1) * hd
  return {"qkv": K.PFQKV(N, D, Hq, Hkv, hd, float(eps), add, qkv_bias, kn["banks"], kn["wn"]),
          "kvw": KVWriteSpec(P, N, 4 * Hkv * hd // 2, Wx, Hq * hd),
          "attn": K.PFAttn(N, Hq, Hkv, hd, P, float(scale), base, kn["exp"]),
          "o": K.PFLinear(N, D, Hq * hd, 0, 0, kn["banks"], kn["wn"]),
          "mlp": K.PFMLP(N, D, I, float(eps), o_bias, kn["exp"], kn["banks"], kn["wn"], evict=kn.get("evict", "act"))}

def layer(cache: Any, i: int, x: Any, d: Any, l: Any, cos: Any, sin: Any, n_heads: int, scale: float, eps: float,
          all_reduce: Any = None) -> tuple[Any, Any]:
  """Decoder layer i of a prefill forward on the prefill kernels (module docstring): (x2, d2) as Llama.forward_tokens's
  member path leaves them: x2 the residual after attention, d2 down's product (all-reduced under TP), the next layer's
  residual term. all_reduce: the shard's sum over the ranks (dist.all_reduce with fuse), None without TP."""
  from fusezap.device import Buffer
  from fusezap.device.neuron import NeuronCall
  from fusezap.dtype import dtypes
  from fusezap.ir import ScheduleItem
  from fusezap import jit as J
  from fusezap.serve.neuron_kv import _buf, _wrap
  Lo, N, D = cache.layout, int(x.shape[0]), int(x.shape[-1])
  Hq, Hkv, hd, P = n_heads, cache.n_kv, cache.hd, cache.ctx or cache.P
  I = int(l.wd.shape[0])
  COS, SIN = cache._tables(cos, sin)
  sp = specs(N, D, Hq, Hkv, hd, I, P, eps, scale, Lo.size, d is not None, l.bqkv is not None, l.bo is not None)
  if "w" not in cache._step: cache._step["w"] = cache._onehot(cache._ints(Lo.slot, N), P).contiguous().realize()   # (N, P) one-hot slots
  dev, f32 = cache.device, dtypes.float32
  def run(spec: Any, bufs: dict, n_out: int) -> None:
    J.run_item(ScheduleItem((NeuronCall(spec),), tuple(bufs[a] for a in spec.args()), {}, "custom", n_out))
  Wx = (Hq + 2 * Hkv + 1) * hd
  xs, os_, po = Buffer(dev, N * Wx, f32), Buffer(dev, N * Hq * hd, f32), Buffer(dev, N * D, f32)
  b = {"xs": xs, "x": _buf(x), "ln1": _buf(l.ln1), "wqkv": _buf(l.wqkv), "cos": _buf(COS), "sin": _buf(SIN)}
  if d is not None: b.update(xo=Buffer(dev, N * D, f32), d=_buf(d))
  if l.bqkv is not None: b["bqkv"] = _buf(l.bqkv)
  run(sp["qkv"], b, len(sp["qkv"].outs()))
  x1 = _wrap(b["xo"], (N, D)) if d is not None else x
  pool = _buf(cache.pools[i])
  J.run_item(ScheduleItem((NeuronCall(sp["kvw"]),), (pool, _buf(cache._step["w"]), xs), {}, "custom", 1))
  run(sp["attn"], {"os": os_, "xs": xs, "pool": pool, "meta": _buf(cache._inputs[0])}, 1)
  run(sp["o"], {"c": po, "a": os_, "w": _buf(l.wo)}, 1)
  o = _wrap(po, (N, D))
  if all_reduce is not None: o = all_reduce(o)
  m = {"xo": Buffer(dev, N * D, f32), "dn": Buffer(dev, N * D, f32), "x": _buf(x1), "o": _buf(o), "ln2": _buf(l.ln2), "wgu": _buf(l.wgu), "wd": _buf(l.wd)}
  if l.bo is not None: m["bo"] = _buf(l.bo)
  run(sp["mlp"], m, 2)
  dn = _wrap(m["dn"], (N, D))
  return _wrap(m["xo"], (N, D)), (all_reduce(dn) if all_reduce is not None else dn)

def head(model: Any, h: Any, all_reduce: Any = None) -> Any:
  """The prefill's lm_head on h (R, K) float32 -> (R, V) float32 logits (model.lm_head: K-major bf16 (K, V)). Under TP with
  FZ_NEURON_PFK_HEAD=split this rank computes columns [rank V/W, (rank+1) V/W) into a zero-padded (R, V) buffer (PFLinear
  pad) and all_reduce sums the ranks' (exact: one nonzero term per column); else the whole head as one PFLinear (MatmulBF16's
  bits, wide tiles)."""
  from fusezap.codegen.nki_prefill import PFLinear, pfk_banks, pfk_wn
  from fusezap.device import Buffer
  from fusezap.device.neuron import NeuronCall
  from fusezap.dtype import dtypes
  from fusezap.ir import ScheduleItem
  from fusezap import jit as J
  from fusezap.serve.neuron_kv import _buf, _wrap
  K, V = int(model.lm_head.shape[0]), int(model.lm_head.shape[1])
  R = int(np.prod(h.shape[:-1]))
  tp = model.tp
  split = tp is not None and all_reduce is not None and head_mode(True) == "split" and V % tp.world == 0
  Vl, off = (V // tp.world, tp.rank * (V // tp.world)) if split else (V, 0)
  out = Buffer(model.device if isinstance(model.device, str) else h.device, R * V, dtypes.float32)
  sp = PFLinear(R, Vl, K, V, off, pfk_banks(), pfk_wn(), pad=split)   # split: this rank's columns, zeros elsewhere
  J.run_item(ScheduleItem((NeuronCall(sp),), (out, _buf(h), _buf(model.lm_head)), {}, "custom", 1))
  y = _wrap(out, (*h.shape[:-1], V))
  return all_reduce(y) if split else y

# ---------------------------------------------------------------- the device self-test

def _rand(rng, shape, sc=1.0): return (rng.standard_normal(shape) * sc).astype(np.float32)

def _precompile(dev: Any, cases: list) -> None:
  """The NEFFs of (spec, host arrays) cases, compiled in parallel threads, each with its own nki worker (as the I75 prewarm's
  NEURONDevice.precompile): cached_neff at the argument sizes / dtypes run_neff will bind. A failure is left to the launch."""
  import threading
  from fusezap.device import neuron as NR
  if not isinstance(getattr(dev, "allocator", None), NR.NeuronAllocator): return
  target, versions = NR.neuron_target(), NR._nki_versions()
  def one(spec: Any, arrs: list) -> None:
    w = None
    try:
      NR._TLS.worker = w = NR.compile_worker()
      prog = spec.render()
      NR.cached_neff(prog, [(int(a.size), np.dtype(a.dtype).str) for a in arrs], target, versions)
    except Exception: pass
    finally:
      NR._TLS.worker = None
      if w is not None:
        try: w.close()
        except Exception: pass
  ts = [threading.Thread(target=one, args=c, daemon=True) for c in cases]
  for t in ts: t.start()
  for t in ts: t.join()

def selftest(g: Any, model: Any = None, rows: tuple[int, ...] = (), ctxs: tuple[int, ...] = (), log: Any = None, force: bool = False) -> dict:
  """Every rank (op "call" under TP, else directly): compile and run each prefill kernel once at this model's shapes
  (rows: the prefill sizes, ctxs: the prefill pool prefixes; the smallest of each by default) on random inputs, against
  nki_prefill's float64 references (relative max error <= 2e-2 of each output's scale with exact exp: bf16 operands; the
  references round like the kernels, so errors are float32 accumulation order). NEURON only (NKISHIM / NKISIM: the tests
  check the bits; force: run there too, tests): {ok, why, kernels: {name: seconds}}. Never raises."""
  from fusezap.device import Device, Buffer
  from fusezap.dtype import dtypes, to_np
  from fusezap.serve.weights import f32_to_bf16
  from fusezap.codegen import nki_prefill as K
  t_all = time.perf_counter()
  dev_name = getattr(g, "device", None) or (model.device if model is not None else Device.DEFAULT)
  model = model if model is not None else getattr(getattr(g, "worker", None), "model", None)
  if (dev_name.split(":")[0].upper() != "NEURON" and not force) or model is None: return {"ok": True, "why": "not a NEURON device: no self-test", "kernels": {}}
  out: dict = {"ok": False, "why": "", "kernels": {}, "level": None, "tried": []}
  for level, fb in enumerate(FALLBACKS):
    why = _selftest_once(dev_name, model, rows, ctxs, {**_knobs(), **fb}, out["kernels"])
    out["tried"].append({"level": level, "knobs": fb, "why": why})
    if not why:
      out.update(ok=True, level=level, knobs=fb, why="" if level == 0 else f"fallback level {level} {fb}: level 0 failed: {out['tried'][0]['why'][:600]}")
      break
  if not out["ok"]: out["why"] = out["tried"][0]["why"]
  out["seconds"] = round(time.perf_counter() - t_all, 1)
  if log is not None: log(f"fusezap: prefill kernels (FZ_NEURON_PREFILL_KERNEL=v2) self-test on rank {getattr(g, 'rank', 0)}: "
                          f"{('passed' + (' (' + out['why'][:300] + ')' if out['why'] else '')) if out['ok'] else 'FAILED: ' + out['why']} ({out['seconds']} s)")
  return out

def _selftest_once(dev_name: str, model: Any, rows: tuple, ctxs: tuple, kn: dict, times: dict) -> str:
  """One self-test pass with these knobs: '' if every kernel compiled and matched its reference, else why not. Every prefill
  size N (rows; 128 if none) at the smallest prefix that holds it; all the NEFFs compile in parallel first."""
  from fusezap.device import Device, Buffer
  from fusezap.dtype import dtypes, to_np
  from fusezap.serve.weights import f32_to_bf16
  from fusezap.codegen import nki_prefill as K
  try:
    dev = Device[dev_name]
    c = model.cfg
    D, Hq, Hkv, hd, I = c.hidden_size, c.n_heads, c.n_kv_heads, c.head_dim, c.intermediate_size
    Nq, meta_base = (Hq + 2 * Hkv) * hd, 8
    rng = np.random.default_rng(77)
    def bufs(arrs: list) -> list:
      bs = []
      for a in arrs:
        b = Buffer(dev_name, a.size, dtypes.bfloat16 if a.dtype == np.uint16 else dtypes.int32 if a.dtype == np.int32 else dtypes.float32).ensure_allocated()
        b.copyin(memoryview(np.ascontiguousarray(a)).cast("B"))
        bs.append(b)
      return bs
    def get(b: Any, shape) -> np.ndarray:
      a = np.empty(b.size, to_np(b.dtype)); b.copyout(memoryview(a).cast("B")); return a.reshape(shape)
    def check(name: str, got: np.ndarray, want: np.ndarray, tol: float = 2e-2) -> None:
      err = float(np.max(np.abs(got.astype(np.float64) - want)) / max(1e-6, float(np.max(np.abs(want)))))
      if not np.isfinite(err) or err > tol: raise AssertionError(f"{name}: relative max error {err:.3g} > {tol}")
    # shared weights (the largest shapes), per size its activations
    w = f32_to_bf16(_rand(rng, (D, Nq), 0.05)); bq = _rand(rng, Nq, 0.1); ln1, ln2 = 1 + _rand(rng, D, 0.1), 1 + _rand(rng, D, 0.1)
    wo = f32_to_bf16(_rand(rng, (Hq * hd, D), 0.05))
    wgu, wd = f32_to_bf16(_rand(rng, (D, 2 * I), 0.05)), f32_to_bf16(_rand(rng, (I, D), 0.05))
    cases, checks = [], []
    for N in sorted(set(rows or (128,))):
      P = min([x for x in ctxs if x >= N] or [max(ctxs or (N,))])
      sp = specs(N, D, Hq, Hkv, hd, I, P, float(c.rms_eps), 1.0 / np.sqrt(hd), meta_base, True, True, False, kn)
      tol = 2e-2 if sp["attn"].exp == "exact" else 5e-2
      q, at, o, m = sp["qkv"], sp["attn"], sp["o"], sp["mlp"]
      x, d = _rand(rng, (N, D)), _rand(rng, (N, D))
      ang = rng.uniform(0, 3, (N, hd // 2)).astype(np.float32)
      cs, sn = np.concatenate([np.cos(ang)] * 2, -1), np.concatenate([-np.sin(ang), np.sin(ang)], -1)
      aq = {"xo": np.zeros(N * D, np.float32), "xs": np.zeros(N * (Nq + hd), np.float32), "x": x, "d": d, "ln1": ln1, "wqkv": w, "bqkv": bq, "cos": cs, "sin": sn}
      Xa = _rand(rng, (N, Nq + hd)); pool = _rand(rng, (P, 4 * Hkv * hd // 2))   # attention: one sequence over the first P slots
      meta = np.zeros(meta_base + 2 * N + P, np.int32)
      meta[meta_base:meta_base + N], meta[meta_base + N:meta_base + 2 * N] = 0, np.arange(N) % P
      meta[meta_base + 2 * N:] = np.arange(P)
      A = _rand(rng, (N, Hq * hd))
      xm, om = _rand(rng, (N, D)), _rand(rng, (N, D))
      cs_ = [(q, [aq[k] for k in q.args()]), (at, [np.zeros(N * Hq * hd, np.float32), Xa, pool, meta]),
             (o, [np.zeros(N * D, np.float32), A, wo]), (m, [np.zeros(N * D, np.float32), np.zeros(N * D, np.float32), xm, om, ln2, wgu, wd])]
      cases += cs_
      def chk(N=N, q=q, at=at, o=o, m=m, cs_=cs_, x=x, d=d, cs=cs, sn=sn, Xa=Xa, pool=pool, meta=meta, A=A, xm=xm, om=om, tol=tol) -> None:
        bs = launch(*cs_[0])
        xo_w, X_w = K.ref_qkv(q, x, d, ln1, w, bq, cs, sn)
        check(q.name + " xo", get(bs[0], (N, D)), xo_w); check(q.name + " X", get(bs[1], (N, Nq + hd)), X_w, tol)
        bs = launch(*cs_[1])
        check(at.name, get(bs[0], (N, Hq * hd)), K.ref_attn(at, Xa, pool.reshape(-1), meta), tol)
        bs = launch(*cs_[2])
        check(o.name, get(bs[0], (N, D)), K.ref_linear(A, wo))
        bs = launch(*cs_[3])
        xo_w, dn_w = K.ref_mlp(m, xm, om, None, ln2, wgu, wd)
        check(m.name + " xo", get(bs[0], (N, D)), xo_w); check(m.name + " dn", get(bs[1], (N, D)), dn_w, tol)
      checks.append(chk)
    def launch(spec: Any, arrs: list) -> list:
      t0 = time.perf_counter()
      bs = bufs(arrs)
      prg = dev.spec_program(spec)
      dev.run_program(prg, tuple(bs))
      if hasattr(dev, "synchronize"): dev.synchronize()
      times[spec.name] = round(time.perf_counter() - t0, 2)
      return bs
    _precompile(dev, cases)   # every NEFF compiles in parallel (each is its own neuronx-cc run)
    for chk in checks: chk()
  except Exception as e: return f"{type(e).__name__}: {str(e)[:1500]}"
  return ""

def gate(eng: Any, log: Any = None, rows: tuple | None = None, ctxs: tuple | None = None) -> str:
  """Before any forward is captured (LLMEngine on NEURON calls it before fixing its prefill sizes; the engine prewarm again,
  a no-op once decided): run the self-test when v2 is asked for, on every rank under TP (rank 0 drives: the workers each
  test their own core through a mirrored call, and v2 stays on only if every rank passes; set_ok mirrors the verdict and the
  fallback knobs). A tensor-parallel worker never drives it (rank 0's verdict reaches it). eng: anything with .model, .tp
  and (unless rows / ctxs are given) .prefill_sizes and .cache.ctx_buckets. Returns a line for the log ('' if nothing ran)."""
  if kernel_mode() != "v2" or _STATE["ok"] is not None: return ""
  tp = getattr(eng, "tp", None)
  if tp is not None and getattr(tp, "rank", 0) != 0: return ""
  log = log or (lambda m: print(m, file=sys.stderr, flush=True))
  rows = tuple(rows if rows is not None else eng.prefill_sizes)
  ctxs = tuple(ctxs if ctxs is not None else (eng.cache.ctx_buckets(True) if hasattr(eng.cache, "ctx_buckets") else ()))
  if os.environ.get("FZ_NEURON_PFK_SELFTEST", "1") in ("", "0"):
    if tp is not None:
      with tp.spmd("call", "fusezap.serve.neuron_prefill:set_ok", True, "self-test skipped"): set_ok(tp, True, "self-test skipped")
    else: set_ok(None, True, "self-test skipped (FZ_NEURON_PFK_SELFTEST=0)")
    return "fusezap: prefill kernels v2 (self-test skipped)"
  if tp is not None:
    with tp.spmd("call", "fusezap.serve.neuron_prefill:selftest", None, rows, ctxs): mine = selftest(tp, eng.model, rows, ctxs, log)
    res = [mine] + [r for r in tp.barrier("pfk-selftest", 3600) if isinstance(r, dict)]
  else: res = [selftest(type("G", (), {"device": eng.model.device, "rank": 0})(), eng.model, rows, ctxs, log)]
  bad = [r for r in res if not r.get("ok")]
  ok, why = not bad, (bad[0].get("why", "") if bad else "")
  level = max((int(r.get("level") or 0) for r in res), default=0) if ok else 0
  knobs = FALLBACKS[level] if ok else None
  if tp is not None:
    with tp.spmd("call", "fusezap.serve.neuron_prefill:set_ok", ok, why, knobs): set_ok(tp, ok, why, knobs)
  else: set_ok(None, ok, why, knobs)
  return f"fusezap: prefill kernels (FZ_NEURON_PREFILL_KERNEL=v2) " + \
         (f"on, knobs {_knobs()}" + (f" (self-test fallback level {level}: {next((r['why'] for r in res if r.get('why')), '')[:400]})" if level else "")
          if ok else "OFF (the members run): " + why)
