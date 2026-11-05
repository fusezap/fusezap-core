"""Kernel-level view of a profiled serving window, for FuseZap and vLLM alike (docs/status/SPEED.md "Run 4").

  python bench/kernel_prof.py fz-c32.sqlite:FuseZap vllm-c32.sqlite:vLLM [--layers 28] [--top 30] [--json out.json]

Inputs: an `nsys export --type sqlite` file of an `nsys start` / `nsys stop` window (traced with
--cuda-graph-trace=node, so kernels inside CUDA graphs are listed one by one), or a torch-profiler / kineto chrome trace
(.json or .json.gz, e.g. vLLM's /start_profile) as a fallback. Every kernel in the file counts, so the file should hold
one steady window (bench/prof_window.py makes those).

Forwards in the window = launches of the most frequent attention kernel / --layers (one attention launch per layer per
forward on both engines; checked against the norm count). For each input it prints:
  - the window's wall span, GPU busy time (union of kernel intervals) and idle time, per forward
  - kernel time per forward by class (gemm, attention, norm, rope, act, kv_write, sampling, reduce, elementwise, other)
  - the top kernels by (name, grid, block): a GEMM's grid tells its shape apart, so each linear is its own row
  - one decoder layer in launch order from the middle of the window, and the forward's tail (final norm, lm_head,
    sampling, the gap to the next forward)
  - memcpys by kind and the CUDA API calls per forward (graph launches vs kernel launches, copies, syncs; nsys only)
With several inputs, a side-by-side table of the class times per forward comes first.
"""
from __future__ import annotations
import argparse, collections, gzip, json, os, re, sqlite3, sys
from typing import Any

CLASSES = [
  ("kv_write", r"reshape_and_cache|kv_write|concat_and_cache|cache_kernel"),
  ("attention", r"flash|attn|attention|fmha|paged|combine|mha|decode_kernel"),
  ("gemm", r"gemm|wgmma|wgswap|fz_mma|matmul|cutlass|cublas|xmma|nvjet|gemv|sm90_|sm80_|splitk|tf32|ampere_|hopper"),
  ("norm", r"rms|norm|rsqrt"),
  ("rope", r"rope|rotary"),
  ("act", r"silu|act_and_mul|swiglu|gelu"),
  ("sampling", r"argmax|sampl|topk|top_k|softmax|sort|gumbel|exponential|penalt|random"),
  ("reduce", r"^r_|triton_red|reduce"),
  ("elementwise", r"^e_|triton_poi|elementwise|vectorized|unrolled|copy|fill|cast|bf2f|f2bf|bfround|index|cat"),
]
_CRE = [(c, re.compile(p)) for c, p in CLASSES]

def klass(name: str) -> str:
  n = name.lower()
  return next((c for c, r in _CRE if r.search(n)), "other")

def short(name: str, n: int = 70) -> str:
  s = re.sub(r"\s+", " ", name)
  if s.startswith("void "): s = s[5:]
  return s if len(s) <= n else s[:n - 3] + "..."

# ---------------------------------------------------------------- loading

def load_sqlite(path: str) -> dict:
  db = sqlite3.connect(path)
  tabs = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
  strs = dict(db.execute("SELECT id, value FROM StringIds")) if "StringIds" in tabs else {}
  K = "CUPTI_ACTIVITY_KIND_KERNEL"
  kern = []
  if K in tabs:
    cols = {r[1] for r in db.execute(f"PRAGMA table_info({K})")}
    nm = "demangledName" if "demangledName" in cols else "shortName"
    g = ", graphNodeId" if "graphNodeId" in cols else ", NULL"
    sid = ", streamId" if "streamId" in cols else ", 0"
    for st, en, n, gx, gy, gz, bx, by, bz, gn, sm in db.execute(f"SELECT start, end, {nm}, gridX, gridY, gridZ, blockX, blockY, blockZ{g}{sid} FROM {K}"):
      kern.append((st, en, strs.get(n, str(n)), (gx, gy, gz), (bx, by, bz), gn is not None, sm))
  mem = []
  if "CUPTI_ACTIVITY_KIND_MEMCPY" in tabs:
    kinds = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}
    for st, en, ck, nb in db.execute("SELECT start, end, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
      mem.append((st, en, kinds.get(ck, str(ck)), nb))
  api = []
  if "CUPTI_ACTIVITY_KIND_RUNTIME" in tabs:
    for st, en, n in db.execute("SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
      api.append((st, en, strs.get(n, str(n))))
  return {"kernels": kern, "memcpy": mem, "api": api, "source": "nsys"}

def load_kineto(path: str) -> dict:
  op = gzip.open if path.endswith(".gz") else open
  with op(path, "rt") as f: tr = json.load(f)
  evs = tr["traceEvents"] if isinstance(tr, dict) else tr
  kern, mem, api = [], [], []
  for e in evs:
    if e.get("ph") != "X": continue
    c, st = e.get("cat", ""), int(float(e.get("ts", 0)) * 1e3)
    en = st + int(float(e.get("dur", 0)) * 1e3)
    a = e.get("args") or {}
    if c == "kernel":
      kern.append((st, en, e.get("name", "?"), tuple(a.get("grid", (0, 0, 0))), tuple(a.get("block", (0, 0, 0))), a.get("graph id") is not None,
                   a.get("stream", 0)))
    elif c in ("gpu_memcpy", "gpu_memset"): mem.append((st, en, e.get("name", c), a.get("bytes", 0)))
    elif c in ("cuda_runtime", "cuda_driver"): api.append((st, en, e.get("name", "?")))
  return {"kernels": kern, "memcpy": mem, "api": api, "source": "kineto"}

def load(path: str) -> dict:
  return load_sqlite(path) if path.endswith((".sqlite", ".db")) else load_kineto(path)

# ---------------------------------------------------------------- analysis

def clip_outliers(ks: list[tuple]) -> tuple[list[tuple], int]:
  """Kernel records whose end lies past the next kernel's start on the same stream (impossible: a stream runs its kernels
  in order) end at that start. An nsys window of FuseZap at c32 had its kernels ending ~25 ms late (CUPTI end stamps of
  graph nodes were wrong there; the starts and the in-order layer view were right). The last kernel of a stream that is
  longer than 5x the median of its (name, grid) and 0.2 ms more ends at start + that median. Records are (start, end,
  name, grid, block, from_graph[, stream])."""
  import statistics
  by: dict[tuple, list[int]] = collections.defaultdict(list)
  for k in ks: by[(k[2], k[3])].append(k[1] - k[0])
  med = {key: statistics.median(v) for key, v in by.items()}
  streams: dict[Any, list[int]] = collections.defaultdict(list)
  for i, k in enumerate(ks): streams[k[6] if len(k) > 6 else 0].append(i)
  out, n = list(ks), 0
  for idx in streams.values():
    for a, b in zip(idx, idx[1:] + [None]):
      k = out[a]
      if b is not None: end = min(k[1], max(ks[b][0], k[0]))
      else:
        m = med[(k[2], k[3])]
        end = k[0] + int(m) if k[1] - k[0] > max(5 * m, m + 200_000) else k[1]
      if end < k[1] - 1000: n += 1   # more than 1 us of overlap
      out[a] = (k[0], end) + tuple(k[2:])
  return out, n

def analyze(d: dict, layers: int, top: int) -> dict:
  ks, clipped = clip_outliers(sorted(d["kernels"]))
  if not ks: return {"error": "no kernels in the file"}
  t0, t1 = ks[0][0], max(k[1] for k in ks)
  busy, cur_s, cur_e = 0, ks[0][0], ks[0][1]
  for st, en, *_ in ks[1:]:
    if st > cur_e: busy += cur_e - cur_s; cur_s, cur_e = st, en
    else: cur_e = max(cur_e, en)
  busy += cur_e - cur_s
  cls = [klass(k[2]) for k in ks]
  att = collections.Counter(k[2] for k, c in zip(ks, cls) if c == "attention")
  norms = sum(1 for c in cls if c == "norm")
  att_name, att_n = att.most_common(1)[0] if att else ("", 0)
  fwd = round(att_n / layers) if att_n else round(norms / (2 * layers + 1))
  fwd = max(fwd, 1)
  per = lambda ns: ns / 1e6 / fwd   # ns in the window -> ms per forward
  by_class: dict[str, float] = collections.defaultdict(float)
  for k, c in zip(ks, cls): by_class[c] += k[1] - k[0]
  rows: dict[tuple, list] = {}
  for k, c in zip(ks, cls):
    key = (k[2], k[3], k[4])
    r = rows.setdefault(key, [0, 0, c])
    r[0] += 1; r[1] += k[1] - k[0]
  top_rows = sorted(rows.items(), key=lambda kv: -kv[1][1])[:top]
  res = {
    "source": d["source"], "kernels": len(ks), "clipped_kernels": clipped, "forwards": fwd, "attention_kernel": short(att_name, 90), "attention_launches": att_n,
    "norm_launches": norms, "norms_per_forward": round(norms / fwd, 2), "kernels_per_forward": round(len(ks) / fwd, 1),
    "graph_kernels_pct": round(100 * sum(1 for k in ks if k[5]) / len(ks), 1),
    "window_ms": round((t1 - t0) / 1e6, 3), "step_ms": round(per(t1 - t0), 4), "gpu_busy_ms": round(per(busy), 4),
    "gpu_idle_ms": round(per(t1 - t0 - busy), 4), "kernel_sum_ms": round(per(sum(k[1] - k[0] for k in ks)), 4),
    "class_ms": {c: round(per(v), 4) for c, v in sorted(by_class.items(), key=lambda kv: -kv[1])},
    "top": [{"name": short(n, 90), "grid": list(g), "block": list(b), "class": c, "per_fwd": round(cnt / fwd, 2),
             "mean_us": round(t / cnt / 1e3, 2), "ms_per_fwd": round(per(t), 4)} for (n, g, b), (cnt, t, c) in top_rows],
  }
  idx = [i for i, k in enumerate(ks) if k[2] == att_name]
  if len(idx) >= 3:
    mid = len(idx) // 2
    if mid % layers == layers - 1: mid -= 1   # stay inside one forward
    a, b = idx[mid], idx[mid + 1]
    res["layer"] = [{"name": short(k[2], 60), "class": klass(k[2]), "grid": list(k[3]), "us": round((k[1] - k[0]) / 1e3, 2),
                     "gap_before_us": round((k[0] - ks[i - 1][1]) / 1e3, 2) if i else 0.0} for i, k in ((j, ks[j]) for j in range(a, b))]
    tails = [j for j in range(len(idx) - 1) if j % layers == layers - 1]
    if tails:
      j = tails[len(tails) // 2]
      a, b = idx[j], idx[j + 1]
      res["tail"] = [{"name": short(k[2], 60), "class": klass(k[2]), "grid": list(k[3]), "us": round((k[1] - k[0]) / 1e3, 2),
                      "gap_before_us": round((k[0] - ks[i - 1][1]) / 1e3, 2) if i else 0.0} for i, k in ((q, ks[q]) for q in range(a, b))]
  mem: dict[str, list] = collections.defaultdict(lambda: [0, 0, 0])
  for st, en, kd, nb in d["memcpy"]:
    m = mem[kd]; m[0] += 1; m[1] += en - st; m[2] += nb or 0
  res["memcpy"] = {kd: {"per_fwd": round(n / fwd, 2), "ms_per_fwd": round(per(t), 4), "kb_per_fwd": round(nb / 1024 / fwd, 1)} for kd, (n, t, nb) in mem.items()}
  api: dict[str, list] = collections.defaultdict(lambda: [0, 0])
  for st, en, n in d["api"]:
    a_ = api[n]; a_[0] += 1; a_[1] += en - st
  res["api"] = [{"name": n, "per_fwd": round(c / fwd, 2), "mean_us": round(t / c / 1e3, 2), "ms_per_fwd": round(per(t), 4)}
                for n, (c, t) in sorted(api.items(), key=lambda kv: -kv[1][1])[:15]]
  return res

def show(label: str, r: dict) -> None:
  if "error" in r: print(f"== {label}: {r['error']}"); return
  if r.get("clipped_kernels"): print(f"   ({r['clipped_kernels']} kernel records with implausible end stamps clipped to the next kernel's start)")
  print(f"== {label} ({r['source']}): {r['forwards']} forwards in {r['window_ms']:.1f} ms -> {r['step_ms']:.3f} ms per forward; GPU busy "
        f"{r['gpu_busy_ms']:.3f}, idle {r['gpu_idle_ms']:.3f} ms per forward; {r['kernels_per_forward']} kernels per forward "
        f"({r['graph_kernels_pct']}% from graphs); norms per forward {r['norms_per_forward']} (attention kernel: {r['attention_kernel']})")
  print("  class ms per forward: " + ", ".join(f"{c} {v:.3f}" for c, v in r["class_ms"].items()))
  print("  top kernels (name, grid, block):")
  for t in r["top"]:
    print(f"    {t['ms_per_fwd']:8.4f} ms/fwd  {t['per_fwd']:6.2f}x  {t['mean_us']:8.2f} us  {t['class']:10s} {t['name'][:70]} grid {tuple(t['grid'])} block {tuple(t['block'])}")
  for key, title in (("layer", "one decoder layer, launch order (from an attention launch to the next)"), ("tail", "forward tail (last attention to the next forward's first)")):
    if key in r:
      print(f"  {title}: {sum(x['us'] for x in r[key]):.1f} us of kernels, {sum(x['gap_before_us'] for x in r[key][1:]):.1f} us of gaps")
      for x in r[key]: print(f"    {x['us']:9.2f} us  (+{x['gap_before_us']:7.2f} gap)  {x['class']:10s} {x['name']} grid {tuple(x['grid'])}")
  if r["memcpy"]: print("  memcpy per forward: " + ", ".join(f"{k} {v['per_fwd']}x {v['ms_per_fwd']:.4f} ms {v['kb_per_fwd']} KiB" for k, v in r["memcpy"].items()))
  if r["api"]:
    print("  CUDA API per forward (host):")
    for a in r["api"]: print(f"    {a['ms_per_fwd']:8.4f} ms/fwd  {a['per_fwd']:7.2f}x  {a['mean_us']:8.2f} us  {a['name']}")

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("inputs", nargs="+", help="file[:label]")
  ap.add_argument("--layers", type=int, default=28)
  ap.add_argument("--top", type=int, default=30)
  ap.add_argument("--json", default="")
  a = ap.parse_args()
  res = {}
  for spec in a.inputs:
    path, _, label = spec.partition(":") if not os.path.exists(spec) else (spec, "", "")
    label = label or os.path.basename(path)
    try: res[label] = analyze(load(path), a.layers, a.top)
    except Exception as e: res[label] = {"error": f"{type(e).__name__}: {e}"}
  ok = {k: v for k, v in res.items() if "error" not in v}
  if len(ok) > 1:
    cls = list(dict.fromkeys(c for v in ok.values() for c in v["class_ms"]))
    print("ms per forward      " + "".join(f"{k[:14]:>15s}" for k in ok))
    for f in ("step_ms", "gpu_busy_ms", "gpu_idle_ms"): print(f"  {f:17s}" + "".join(f"{v[f]:15.4f}" for v in ok.values()))
    for c in cls: print(f"  {c:17s}" + "".join(f"{v['class_ms'].get(c, 0.0):15.4f}" for v in ok.values()))
    print()
  for k, v in res.items(): show(k, v)
  if a.json:
    with open(a.json, "w") as f: json.dump(res, f, indent=1)
  return 0 if ok else 1

if __name__ == "__main__": sys.exit(main())
