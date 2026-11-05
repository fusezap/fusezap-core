"""Per-kernel GPU time of one decode step on ONE GPU, random weights at a TP rank's exact shapes (I60, I57 M4).

  python bench/decode_kernels.py [--preset llama70b-tp2] [--buckets 1,8,32] [--ctx 512] [--variants 'old:FZ_ROPE_KVW=0;new:']
         [--ar none|copy] [--share-layers auto|0|1] [--json out.json]
  python bench/decode_kernels.py --config path/config.json --tp 2 ...   # any Llama-style checkpoint's rank shapes
  python bench/decode_kernels.py --dry                                   # a tiny model on the default device (checks the script)

The model is fusezap.serve's Llama with tp=None and the local (per-rank) config: hidden 8192, 32 q / 4 kv heads, head dim
128, intermediate 14336, 80 layers, vocab 128256 (llama70b-tp2), bf16 random weights (one random block tiled; values do not
change the time of these kernels), so it needs no download and one GPU. Row-parallel all-reduces are absent (--ar none)
or emulated by a device-to-device copy at each of them (--ar copy, FZ_KPROF_AR=copy per variant: the kernel count and
order of a TP rank's step; on 2x H200 NCCL measured 8.5 us per decode all-reduce, add 160 x that for the real step).
--share-layers: every layer reuses layer 0's weight buffers (for GPUs with less memory than ~70 GB of weights plus the KV
pool; auto = when they do not fit; a layer's 0.87 GB still leaves L2 cold between its uses).

The engine is LLMEngine as `fusezap serve --max-model-len 8192 --num-blocks 8192 --max-num-seqs 32` builds it (paged KV,
decode buckets, CUDA graphs, lookahead). Per variant (env changes, then every forward is captured again) and per bucket b:
b requests of ctx random tokens are prefilled, the decode graph captured, then
  - tpot_ms: engine.step() wall time per decode step (lookahead on: GPU-bound when the GPU is the bottleneck)
  - graph_ms: GPU time of one replay of the captured decode graph (events around 20 back-to-back replays)
  - per item: the graph's items launched one by one in graph order behind a 50 ms spin kernel (so the host is always
    ahead and no item waits for its launch), a CUDA event between consecutive items: each item's GPU time in its real
    cache context (the previous kernel's traffic in L2, weights cold). sum_ms vs graph_ms estimates the inter-kernel gaps.
Items are grouped by class (gemm per projection, attention partial / combine, norms, rope, kv_write, act, argmax, inputs,
all-reduce; composed kernels named by the GEMM before them), each with its count per step, ms, % of the step, achieved
TB/s and the ideal: its minimum bytes (weights + activations, the KV rows read by attention) at --peak-tbps (4.8 = H200),
and an expectation: max(bytes / (--eff x peak), count x --launch-us) (a vLLM-like kernel per class; launch-bound kernels
cost about a launch). Prints TABLE / KPROF JSON lines; --json writes everything.
"""
from __future__ import annotations
import argparse, collections, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from fusezap.serve.models.random_weights import PRESETS  # noqa: E402  (the local per-rank shapes: the "-tp2" presets)

def shapes_from_config(path: str, tp: int) -> dict:
  from fusezap.serve.models.llama import LlamaConfig
  from fusezap.serve.dist import local_cfg
  c = json.load(open(path))
  lc = local_cfg(LlamaConfig.from_hf(c), tp) if tp > 1 else LlamaConfig.from_hf(c)
  return dict(hidden_size=lc.hidden_size, intermediate_size=lc.intermediate_size, n_layers=lc.n_layers, n_heads=lc.n_heads,
              n_kv_heads=lc.n_kv_heads, vocab_size=lc.vocab_size, bias=c.get("model_type") == "qwen2", rope_theta=lc.rope_theta)

# ---------------------------------------------------------------- the random model

def build_model(sh: dict, device: str, dtype, max_len: int, share: bool, seed: int = 0):
  """Llama with tp=None at the local shapes `sh`, random weights (fusezap.serve.models.random_weights)."""
  from fusezap.serve.models import random_weights
  return random_weights.build(sh, device, dtype, max_len, 512, share, None, seed)

def patch_ar() -> None:
  """FZ_KPROF_AR=copy (read when a forward is traced): a device-to-device copy where a TP rank all-reduces (after o_proj
  and down_proj), so the step has a TP rank's items; the copy is the 'skip' mode of dist.TPComm."""
  from fusezap.serve.models.llama import Llama
  from fusezap.serve import dist
  orig = Llama._rowpar
  class _Copy:
    name = "copy"
    def all_reduce(self, dev, out, inp, n, dt):
      if hasattr(dev, "copy_async"): return dev.copy_async(out.handle, inp.handle, n * dt.itemsize)
      host = memoryview(np.empty(n * dt.itemsize, np.uint8))   # --dry off CUDA: through the host
      dev.allocator.copyout(host, inp.handle); dev.allocator.copyin(out.handle, host)
  class _G:   # what dist.all_reduce needs of a TPGroup
    comm = _Copy()
  def rowpar(self, x, w, b, redo=None, fuse=False):
    if self.tp is not None or os.environ.get("FZ_KPROF_AR", "") != "copy": return orig(self, x, w, b, redo, fuse)
    y = dist.all_reduce(self.linear(x, w, None, redo), _G())
    return y if b is None else y + b
  Llama._rowpar = rowpar

# ---------------------------------------------------------------- the engine and its steps

def measure(eng, b: int, ctx: int, steps: int, seed: int, during=None) -> dict:
  """b requests of ctx random tokens: prefill, capture, `steps` timed decode steps; then during() (the breakdown) while
  the requests still decode, then abort and release them."""
  from fusezap.serve.engine import SamplingParams
  rng, V = np.random.default_rng(seed), eng.model.cfg.vocab_size
  sp = SamplingParams(max_tokens=steps + 64, ignore_eos=True)
  rids = [eng.add_request([int(t) for t in rng.integers(min(1000, V // 2), min(20000, V - 2), ctx)], sp) for _ in range(b)]
  n0 = lambda: sum(len(eng.seqs[r].output) for r in rids)
  t = time.perf_counter()
  while min(len(eng.seqs[r].output) for r in rids) < 1: eng.step()
  t_pref = time.perf_counter() - t
  for _ in range(6): eng.step()   # trace, record, warm and capture the bucket's graph
  k0, t0 = n0(), time.perf_counter()
  for _ in range(steps): eng.step()
  dt, k1 = time.perf_counter() - t0, n0()
  extra = during() if during is not None else {}
  for r in rids: eng.abort(r)
  while eng.has_unfinished() or eng._pend is not None or eng._pend_p is not None: eng.step()
  for r in rids: eng.release(r)
  return {"b": b, "tpot_ms": round(dt / steps * 1e3, 3), "tok_s": round((k1 - k0) / dt, 1), "prefill_s": round(t_pref, 2), **extra}

class Timer:
  """GPU intervals between consecutive marks: CUDA events (one per mark), elsewhere synchronize + host clock."""
  def __init__(self, dev):
    self.dev, self.cu = dev, getattr(dev, "cu", None) if dev.name.split(":")[0].upper() == "CUDA" else None
    self.evs: list = []
    self.spin = None
  def spin_ahead(self, us: int = 50000) -> None:
    """A kernel that busy-waits `us` microseconds, so every launch after it is queued before the GPU reaches it."""
    if self.cu is None: return
    from fusezap.serve.cuda import CudaCall
    from fusezap.device import Buffer
    from fusezap.dtype import dtypes
    src = """extern "C" __global__ void fz_kprof_spin(int* o, const int us) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < (unsigned long long)us * 1000ull);
  if (threadIdx.x == 0 && o[0] == 0x7fffffff) o[0] = 0;
}"""
    if self.spin is None: self.spin = Buffer(self.dev.name, 1, dtypes.int32).ensure_allocated()
    CudaCall("fz_kprof_spin", src, (us,), (1, 1, 1), (32, 1, 1)).launch(self.dev, [self.spin])
  def marks(self, n: int) -> None:
    if self.cu is not None:
      while len(self.evs) < n: self.evs.append(self.cu.cuEventCreate(0)[1])
    self.k, self.host = 0, []
  def mark(self) -> None:
    if self.cu is not None: self.cu.cuEventRecord(self.evs[self.k], self.dev.stream)
    else: self.dev.synchronize(); self.host.append(time.perf_counter())
    self.k += 1
  def intervals(self) -> list[float]:
    """ms between consecutive marks (after the device is done)."""
    self.dev.synchronize()
    if self.cu is None: return [(b - a) * 1e3 for a, b in zip(self.host, self.host[1:])]
    return [float(self.cu.cuEventElapsedTime(self.evs[i], self.evs[i + 1])[1]) for i in range(self.k - 1)]

def graph_ms(dev, c, n: int = 20) -> float | None:
  """GPU ms of one replay of the captured graph (n back-to-back launches between two events)."""
  if c is None or c.graph is None: return None
  t = Timer(dev)
  c.graph.launch()
  dev.synchronize()
  t.marks(2); t.mark()
  for _ in range(n): c.graph.launch()
  t.mark()
  return t.intervals()[0] / n

def item_name(it) -> str:
  if it.prg is not None: return it.prg.name
  a = it.si.ast[0] if it.si is not None and it.si.ast else None
  return getattr(a, "name", type(a).__name__) if a is not None else "?"

def classify(items, cfg) -> list[tuple[str, str]]:
  """(class, label) per item: GEMMs by projection, the serve kernels by name, composed kernels by the GEMM before them."""
  hd, D, I, V = cfg.head_dim, cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
  out, prev = [], ""
  for it in items:
    nm, a = item_name(it), (it.si.ast[0] if it.prg is None and it.si is not None and it.si.ast else None)
    n = nm.lower()
    if a is not None and all(hasattr(a, k) for k in ("M", "N", "K")) and "gemm" in n:
      N, K = a.N, a.K
      prev = ("qkv" if N == (cfg.n_heads + 2 * cfg.n_kv_heads) * hd else "o" if K == cfg.n_heads * hd and N == D else
              "gate_up" if N == 2 * I else "down" if K == I else "lm_head" if N == V else f"{N}x{K}")
      out.append((f"gemm:{prev}", f"{nm} M={a.M} N={N} K={K}")); continue
    if "all_reduce" in n: c = "allreduce"
    elif "rope_qkv" in n: c = "rope+kv_write" if "_kvw" in n else "rope"
    elif "kv_write" in n: c = "kv_write"
    elif "combine" in n: c = "attn:combine"
    elif any(k in n for k in ("paged", "attn", "tcd", "kpt", "mdec", "flash")): c = "attn:partial"
    elif "rmsnorm_res" in n: c = "norm:add+rms"
    elif "rmsnorm" in n: c = "norm:rms"
    elif "silu" in n: c = "act"
    elif "argmax" in n: c = "argmax"
    elif any(k in n for k in ("step_inputs", "rows", "gather", "embed")): c = "inputs"
    elif it.prg is not None:   # a composed kernel (fusezap codegen): what it is follows from the GEMM before it
      c = {"qkv": "rope(composed)", "gate_up": "act(composed)", "lm_head": "logits(composed)"}.get(prev, "elementwise")
    else: c = f"other:{nm[:32]}"
    out.append((c, nm))
  return out

def ideal_bytes(cfg, N: int, ctx: int, bias: bool) -> dict[str, float]:
  """Minimum bytes per decode step and class (2-byte weights / activations): weights + activations for the GEMMs, the
  N tokens' KV rows (ctx keys each) + q + output for attention, inputs + outputs for the rest."""
  E, D, I, V, L = 2, cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size, cfg.n_layers
  Hq, Hkv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
  Q = (Hq + 2 * Hkv) * hd
  per = {"gemm:qkv": Q * D * E + N * D * E + N * Q * (4 if bias else E), "gemm:o": D * Hq * hd * E + N * Hq * hd * E + N * D * E,
         "gemm:gate_up": 2 * I * D * E + N * D * E + N * 2 * I * E, "gemm:down": D * I * E + N * I * E + N * D * E,
         "attn": N * ctx * Hkv * hd * E * 2 + 2 * N * Hq * hd * E, "kv_write": 2 * N * 2 * Hkv * hd * E,
         "rope": 2 * N * Q * E + N * hd * E, "norm": 2 * (4 * N * D * E + D * E), "act": N * 3 * I * E, "allreduce": 2 * 2 * N * D * E}
  out = {k: v * L for k, v in per.items()}
  out["rope+kv_write"] = out["rope"] + out["kv_write"]
  out.update({"gemm:lm_head": V * D * E + N * D * E + N * V * 4, "argmax": N * V * 4, "inputs": N * D * E})
  return out

def _group(c: str) -> str:
  """The ideal-bytes key of a class."""
  if c.startswith("attn"): return "attn"
  if c.startswith("norm"): return "norm"
  if c in ("rope(composed)",): return "rope"
  if c in ("act(composed)",): return "act"
  return c

def breakdown(eng, fwd, ctx: int, reps: int, peak: float, eff: float, launch_us: float, bias: bool) -> dict:
  """Per-item GPU times of fwd's captured decode replay (see the module docstring), grouped by class."""
  from fusezap.ir import ScheduleItem
  from fusezap import jit as J
  from fusezap.shape.symbolic import sym_infer
  from fusezap.device import Device
  c = fwd.jit.captured
  assert c is not None, "decode forward not captured"
  dev = Device[eng.model.device]
  if c.gstage is not None:
    pools = iter([t._realized_contiguous().realized for t in eng.cache.state()])   # the written inputs: the KV pools
    ins, owned = [s if s is not None else next(pools) for s in c.gstage], c.gowned
  else:   # launch-by-launch replays (no graph): the persistent buffers and the last inputs are not kept, so rerun a step
    seen = []
    orig = c._launch
    c._launch = lambda i, o, vv: (seen.append((i, o)), orig(i, o, vv))
    try: eng.step()
    finally: del c._launch
    if not seen: return {"error": "no replay seen"}
    ins, owned = seen[-1]
  res = lambda r: c._res(r, ins, owned)
  items = [it for it in c.items if not it.const]
  cls = classify(items, eng.model.cfg)
  t = Timer(dev)
  acc = np.zeros(len(items))
  for rep in range(reps + 1):
    t.marks(len(items) + 1)
    t.spin_ahead()
    t.mark()
    for it in items:
      bufs = [res(r) for r in it.slots]
      if it.prg is not None:
        gs = it.gs if it.static else tuple(int(sym_infer(s, {})) for s in it.gs)
        it.prg(*[b.handle for b in bufs[:it.nb]], global_size=gs, local_size=it.ls, vals=())
      else: J._run(it.dev, ScheduleItem(it.si.ast, tuple(bufs), {}, it.si.kind, it.si.n_outputs))
      t.mark()
    iv = t.intervals()
    if rep: acc += np.asarray(iv)   # rep 0 binds launchers and warms
  ms = acc / reps
  N = fwd.N
  ib = ideal_bytes(eng.model.cfg, N, ctx, bias)
  agg: dict = collections.defaultdict(lambda: [0, 0.0])
  for (k, _), x in zip(cls, ms): agg[k][0] += 1; agg[k][1] += float(x)
  tot = float(ms.sum())
  groups: dict = collections.defaultdict(list)
  for k in agg: groups[_group(k)].append(k)
  rows = []
  for k, (n, x) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    g = _group(k)
    share = x / max(1e-9, sum(agg[j][1] for j in groups[g]))   # a group's bytes split by its classes' time (attn partial / combine)
    nb = ib.get(g, 0.0) * share
    ideal = nb / (peak * 1e9)
    expect = max(nb / (eff * peak * 1e9), n * launch_us * 1e-3)
    rows.append({"class": k, "count": n, "ms": round(x, 4), "pct": round(100 * x / max(tot, 1e-9), 2), "GB": round(nb / 1e9, 4),
                 "TBps": round(nb / 1e12 / (x / 1e3), 3) if x > 0 and nb else None, "ideal_ms": round(ideal, 4), "expect_ms": round(expect, 4),
                 "excess_ms": round(x - expect, 4)})
  # one decoder layer in launch order, from the middle of the step
  qk = [i for i, (k, _) in enumerate(cls) if k == "gemm:qkv"]
  layer = []
  if len(qk) >= 2:
    mid = len(qk) // 2
    lo, hi = qk[mid], qk[mid + 1] if mid + 1 < len(qk) else len(items)
    lo = max(0, lo - 1)   # the norm before the qkv GEMM
    layer = [{"i": i, "class": cls[i][0], "name": cls[i][1][:80], "us": round(float(ms[i]) * 1e3, 2)} for i in range(lo, hi)]
  top: dict = collections.defaultdict(lambda: [0, 0.0])
  for (k, nm), x in zip(cls, ms): top[nm][0] += 1; top[nm][1] += float(x)
  return {"N": N, "items": len(items), "sum_ms": round(tot, 3), "classes": rows, "layer": layer,
          "top": [{"name": nm[:90], "count": n, "ms": round(x, 4)} for nm, (n, x) in sorted(top.items(), key=lambda kv: -kv[1][1])[:20]]}

def print_table(tag: str, r: dict) -> None:
  bd = r["breakdown"]
  g = r.get("graph_ms")
  print(f"\n== {tag} b={r['b']} (bucket {bd['N']} rows, ctx {r['ctx']}): TPOT {r['tpot_ms']} ms (engine wall), graph {g} ms, "
        f"items {bd['items']}, sum of items {bd['sum_ms']} ms" + (f", graph - sum {round(g - bd['sum_ms'], 3)} ms (gaps)" if g else ""))
  print(f"{'class':18s} {'count':>5s} {'ms':>8s} {'%':>6s} {'TB/s':>6s} {'ideal_ms':>8s} {'expect':>8s} {'excess':>8s}")
  for c in bd["classes"]:
    tb = f"{c['TBps']:.2f}" if c["TBps"] else "-"
    print(f"{c['class']:18s} {c['count']:5d} {c['ms']:8.3f} {c['pct']:6.2f} {tb:>6s} {c['ideal_ms']:8.3f} {c['expect_ms']:8.3f} {c['excess_ms']:8.3f}")
  te = sum(c["expect_ms"] for c in bd["classes"])
  print(f"{'total':18s} {bd['items']:5d} {bd['sum_ms']:8.3f} {100:6.2f} {'':6s} {sum(c['ideal_ms'] for c in bd['classes']):8.3f} {te:8.3f} {bd['sum_ms'] - te:8.3f}")
  if bd["layer"]:
    print("one layer in launch order (us): " + ", ".join(f"{x['class']} {x['us']}" for x in bd["layer"]))

def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--preset", default="llama70b-tp2", choices=sorted(PRESETS)); ap.add_argument("--config", default=None)
  ap.add_argument("--tp", type=int, default=2); ap.add_argument("--layers", type=int, default=0)
  ap.add_argument("--buckets", default="1,8,32"); ap.add_argument("--ctx", type=int, default=512); ap.add_argument("--steps", type=int, default=24)
  ap.add_argument("--reps", type=int, default=3); ap.add_argument("--max-model-len", type=int, default=8192)
  ap.add_argument("--num-blocks", type=int, default=8192); ap.add_argument("--max-num-seqs", type=int, default=32)
  ap.add_argument("--max-prefill-tokens", type=int, default=1024, help="prefill forward rows (0: the engine's default); only the prompts use them")
  ap.add_argument("--variants", default=":", help="NAME:K=V,K=V;NAME2:... (env per variant, every forward captured again); default one variant")
  ap.add_argument("--ar", default="none", choices=("none", "copy"), help="default FZ_KPROF_AR for variants that do not set it")
  ap.add_argument("--share-layers", default="auto"); ap.add_argument("--peak-tbps", type=float, default=4.8)
  ap.add_argument("--eff", type=float, default=0.9); ap.add_argument("--launch-us", type=float, default=2.5)
  ap.add_argument("--device", default=None); ap.add_argument("--dtype", default="bfloat16")
  ap.add_argument("--json", default=None); ap.add_argument("--dry", action="store_true", help="tiny model, short run (checks the script anywhere)")
  a = ap.parse_args()
  from fusezap.device import Device
  from fusezap.dtype import dtypes
  from fusezap.helpers import getenv
  from fusezap.serve.engine import LLMEngine
  if a.dry: a.preset, a.buckets, a.ctx, a.steps, a.reps, a.max_model_len, a.num_blocks, a.max_num_seqs, a.max_prefill_tokens = "tiny", "1,2", 40, 4, 1, 256, 128, 4, 0
  sh = shapes_from_config(a.config, a.tp) if a.config else dict(PRESETS[a.preset])
  if a.layers: sh["n_layers"] = a.layers
  dev = Device[Device.canonicalize(a.device)]
  dt = getattr(dtypes, a.dtype)
  D, I, V, L = sh["hidden_size"], sh["intermediate_size"], sh["vocab_size"], sh["n_layers"]
  Q = (sh["n_heads"] + 2 * sh["n_kv_heads"]) * 128
  wbytes = L * (Q * D + D * sh["n_heads"] * 128 + 3 * I * D) * 2 + 2 * V * D * 2
  kvbytes = L * a.num_blocks * 16 * 2 * sh["n_kv_heads"] * 128 * 2
  free = dev.mem_free() if hasattr(dev, "mem_free") else 1 << 60
  share = (wbytes + kvbytes + (12 << 30) > free) if a.share_layers == "auto" else a.share_layers == "1"
  info = {"shapes": sh, "device": dev.name, "weights_GB": round(wbytes / 1e9, 1), "kv_GB": round(kvbytes / 1e9, 1),
          "free_GB": round(free / 1e9, 1), "share_layers": share, "ctx": a.ctx}
  print(f"KPROF_INFO {json.dumps(info)}", flush=True)
  patch_ar()
  t0 = time.perf_counter()
  m = build_model(sh, dev.name, dt, a.max_model_len, share)
  print(f"KPROF_BUILT {json.dumps({'s': round(time.perf_counter() - t0, 1)})}", flush=True)
  eng = LLMEngine(m, num_blocks=a.num_blocks, block_size=16, max_num_seqs=a.max_num_seqs, max_model_len=a.max_model_len,
                  max_prefill_tokens=a.max_prefill_tokens or None)
  print(f"KPROF_ENGINE {json.dumps({'s': round(time.perf_counter() - t0, 1), 'decode_sizes': eng.decode_sizes, 'prefill_sizes': eng.prefill_sizes})}", flush=True)
  variants = []
  for spec in filter(None, a.variants.split(";")):
    nm, kv = spec.split(":", 1) if ":" in spec else (spec, "")
    variants.append((nm or "default", dict(p.split("=", 1) for p in kv.split(",") if p)))
  keys = sorted({k for _, e in variants for k in e} | {"FZ_KPROF_AR"})
  base_env = {k: os.environ.get(k) for k in keys}
  out = {"info": info, "runs": []}
  for nm, env in variants:
    full = {**base_env, "FZ_KPROF_AR": base_env.get("FZ_KPROF_AR") or a.ar, **env}
    for k, v in full.items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = v
    getenv.cache_clear()
    for f in eng._by_key.values(): f.jit.reset()
    for b in [int(x) for x in a.buckets.split(",") if x]:
      def during(b=b) -> dict:
        fwd = eng._decode_fwd(b)
        g = graph_ms(dev, fwd.jit.captured)
        return {"graph_ms": None if g is None else round(g, 3),
                "breakdown": breakdown(eng, fwd, a.ctx + a.steps, a.reps, a.peak_tbps, a.eff, a.launch_us, bool(sh.get("bias")))}
      r = {"variant": nm, "env": env, "ctx": a.ctx, **measure(eng, b, a.ctx, a.steps, 100 + b, during)}
      out["runs"].append(r)
      print(f"KPROF {json.dumps({k: v for k, v in r.items() if k != 'breakdown'} | {'sum_ms': r['breakdown'].get('sum_ms'), 'classes': {c['class']: c['ms'] for c in r['breakdown'].get('classes', [])}})}", flush=True)
      print_table(nm, r)
  # A/B summary: per bucket, each variant's graph / sum ms and the class deltas against the first variant
  if len(variants) > 1:
    print("\n== A/B (graph ms per bucket; per-class ms deltas vs the first variant)")
    first = {r["b"]: r for r in out["runs"] if r["variant"] == variants[0][0]}
    for r in out["runs"]:
      f = first.get(r["b"])
      if f is None or r is f: continue
      fc = {c["class"]: c["ms"] for c in f["breakdown"]["classes"]}
      rc = {c["class"]: c["ms"] for c in r["breakdown"]["classes"]}
      d = {k: round(rc.get(k, 0) - fc.get(k, 0), 3) for k in sorted(set(fc) | set(rc)) if abs(rc.get(k, 0) - fc.get(k, 0)) >= 0.005}
      print(f"b={r['b']:<3d} {r['variant']:>12s} vs {variants[0][0]}: graph {f['graph_ms']} -> {r['graph_ms']} ms, TPOT {f['tpot_ms']} -> {r['tpot_ms']} ms; {d}")
  if a.json:
    with open(a.json, "w") as fh: json.dump(out, fh, indent=1)

if __name__ == "__main__": main()
