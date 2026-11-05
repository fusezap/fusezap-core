"""NKI-N7 step 2: the serving head alone (final RMSNorm with the residual add, lm_head 151936 x 896 float32, device argmax), as
the engine fuses it into one NEFF, per head variant and batch rows: device time per launch (nrtpy's device benchmark of the
replayed launch) and the bits (every variant's logits and tokens vs v1's, the tokens vs np.argmax of its own logits).

  FZ_DEVICE=NEURON uv run python scripts/nki_n7_head_bench.py --out results/nki_n7_step2 --rows 1,4 --variants v1,v3,v1-logits
  FZ_DEVICE=NKISHIM uv run python scripts/nki_n7_head_bench.py --rows 1 --vocab 4096   # Mac dry run (bits only, no timing)

Variants: v1 (render_matmul on kmajor(lm_head), the composed argmax: N7 step 1), v3 (render_matmul + ArgmaxSpec), v2
(HeadSpec at FZ_NEURON_HEAD_W / FZ_NEURON_HEAD_DMAS, ArgmaxSpec), v2:<W>:<dmas> (HeadSpec with that block width and DMAs per
block), v4 / v4:<W>:<dmas> (N7 step 4: HeadSpec weight-stationary, bitwise vs v1 only if TensorE is operand-symmetric); a "-logits" suffix
drops the argmax (the FZ_NEURON_DEVICE_ARGMAX=0 head). Random weights (seeded), Qwen2.5-0.5B's shapes by default.
N8 (I70), bf16 lm_head (the weights rounded to bf16): b1 (MatmulBF16 on the K-major bf16 weight; with -logits the bits
reference of the bf16 variants), b3 (MatmulBF16 + ArgmaxSpec: the I68 bf16 head), b2:<W> (nki_bf16_n8.HeadBF16 on
head_tiles_bf16 with W columns per block + ArgmaxSpec: FZ_NEURON_BF16_HEAD=1, FZ_NEURON_BF16_HEAD_W=W).
Writes <out>/head_bench.json and one line per (rows, variant): "head_bench rows R <variant> <kernel> device X ms ...".
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def log(*a) -> None: print(*a, flush=True)

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="")
  ap.add_argument("--rows", default="1,4")
  ap.add_argument("--variants", default="v1,v2")
  ap.add_argument("--vocab", type=int, default=151936)
  ap.add_argument("--hidden", type=int, default=896)
  ap.add_argument("--replays", type=int, default=5)
  args = ap.parse_args()
  os.environ.setdefault("FZ_NEURON_FUSE", "1")
  from fusezap import Tensor
  from fusezap.codegen import nki_kernels as nk
  from fusezap.device import Device
  from fusezap.device.neuron import kmajor, neuron_profile
  from fusezap.jit import Jit
  from fusezap.serve import neuron_kv as kv
  dev = Device.DEFAULT
  assert kv.is_neuron(dev), f"FZ_DEVICE must be a NEURON tier, got {dev}"
  al = getattr(Device[dev], "allocator", None)
  timed = hasattr(al, "profile")
  V, K = args.vocab, args.hidden
  rng = np.random.default_rng(0)
  t0 = time.perf_counter()
  w = (rng.standard_normal((V, K)) * 0.05).astype(np.float32)
  nw = Tensor((1 + 0.1 * rng.standard_normal(K)).astype(np.float32), device=dev).realize()
  weights: dict = {}
  def weight(var: str):
    if var.startswith(("b1", "b3")) or var.startswith("b2"):   # N8: bf16 weights (the same random values rounded)
      from fusezap.codegen.nki_bf16_n8 import head_tiles_bf16
      from fusezap.dtype import dtypes
      from fusezap.serve.weights import f32_to_bf16
      key = ("bk",) if not var.startswith("b2") else ("bt", int(var.split(":")[1]) if ":" in var else 2048)
      if key not in weights:
        wb = f32_to_bf16(w)
        a = np.ascontiguousarray(wb.T) if key[0] == "bk" else head_tiles_bf16(wb, key[1])
        weights[key] = Tensor(a, device=dev, dtype=dtypes.bfloat16).realize()
      return weights[key]
    if not var.startswith(("v2", "v4")): key = "k"
    else: key = int(var.split(":")[1]) if ":" in var else nk.HEAD_W
    if key not in weights:
      weights[key] = kmajor(Tensor(w, device=dev).realize()) if key == "k" else Tensor(nk.head_tiles(w, key), device=dev).realize()
    return weights[key]
  results: list = []
  def save() -> None:   # after every record: a phase timeout keeps what was measured
    if args.out:
      os.makedirs(args.out, exist_ok=True)
      with open(os.path.join(args.out, "head_bench.json"), "w") as fh: json.dump({"vocab": V, "hidden": K, "results": results}, fh, indent=1)
  for M in (int(r) for r in args.rows.split(",")):
    x = (rng.standard_normal((M, K)) * 2).astype(np.float32)
    d = rng.standard_normal((M, K)).astype(np.float32)
    ref = None
    for var in args.variants.split(","):
      base, greedy = var.removesuffix("-logits"), not var.endswith("-logits")
      v2 = base.startswith(("v2", "v4"))   # HeadSpec on head_tiles' layout (v4: weight-stationary)
      W, D = (int(base.split(":")[1]), int(base.split(":")[2])) if v2 and ":" in base else (None, None)
      bf = base.startswith("b")   # N8: bf16 heads
      os.environ["FZ_NEURON_HEAD"] = "v3" if bf else base.split(":")[0]
      try:
        ws = base.startswith("v4")
        if v2: nk.HeadSpec(M, V, K, W or nk.HEAD_W, D or nk.HEAD_DMAS, ws).render()   # NKIUnsupported: SBUF limits (HEAD_SBUF)
        wt = weight(base)
        def head(xi: Tensor, di: Tensor):
          h = kv.rmsnorm(xi, nw, 1e-6, di)[1]
          if bf: lg = kv.head_matmul(h, wt, V, int(base.split(":")[1]) if ":" in base else 2048) if base.startswith("b2") else kv.matmul_bf16(h, wt)
          else: lg = kv.head_matmul(h, wt, V, W, D, ws) if v2 else h @ wt
          if greedy and nk.head_argmax(): lg = kv._wrap(kv._buf(lg), tuple(lg.shape))   # as LLMEngine: no copy of a view out
          return (lg, kv.argmax_rows(lg)) if greedy else lg
        f = Jit(head)
        xs = [Tensor(x, device=dev).realize(), Tensor(d, device=dev).realize()]
        st = time.perf_counter()
        for _ in range(3): out = f(*xs)   # eager, capture (fused), first replay: compiles the NEFF
        first_s = time.perf_counter() - st
        rec = {"rows": M, "variant": var, "first_calls_s": round(first_s, 1),
               "launches": sum(not it.const for it in f.captured.items) if f.captured else None}
        if timed:
          al.flush(); al.profile, al.prof, al.prof_wall, al.prof_ipc, al.bindings = True, {}, [0, 0.0], [0, 0.0, 0.0], {}
          for _ in range(args.replays): out = f(*xs)
          al.flush(); al.profile = False
          rep = neuron_profile(al, iters=20, bench=True)
          rec["kernels"] = {nm: {"launches": k["launches"], "exec_ms_mean": k["exec_ms_mean"], "device_ms": k["device_ms"]} for nm, k in rep["kernels"].items()}
          rec["device_ms"] = round(sum((k["device_ms"] or 0.0) * k["launches"] for k in rep["kernels"].values()) / args.replays, 4)
          if rep.get("bench_errors"): rec["bench_errors"] = rep["bench_errors"]
        lg, idx = (out[0].numpy(), out[1].numpy()) if greedy else (out.numpy(), None)
        rec["argmax_ok"] = None if idx is None else bool(idx.tolist() == np.argmax(lg, -1).tolist())
        if ref is None and base in ("v1", "b1"): ref = (lg, idx)
        if ref is not None:
          rec["logits_bitwise_vs_v1"] = round(float(np.mean(lg.view(np.int32) == ref[0].view(np.int32))), 6)
          if idx is not None and ref[1] is not None: rec["tokens_vs_v1"] = bool(idx.tolist() == ref[1].tolist())
      except Exception as e:
        log(f"head_bench rows {M} {var:<14} SKIPPED: {type(e).__name__}: {str(e)[:300]}")
        results.append({"rows": M, "variant": var, "skipped": f"{type(e).__name__}: {str(e)[:300]}"}); save()
        continue
      results.append(rec); save()
      ks = " ".join(f"{nm} {k['launches'] / args.replays:.0f}/call device {k['device_ms']} ms (execute {k['exec_ms_mean']} ms)" for nm, k in (rec.get("kernels") or {}).items())
      log(f"head_bench rows {M} {var:<14} device {rec.get('device_ms')} ms per call  [{ks}]  launches {rec['launches']}  "
          f"argmax==np.argmax {rec['argmax_ok']}  logits bitwise vs v1 {rec.get('logits_bitwise_vs_v1')}  tokens vs v1 {rec.get('tokens_vs_v1')}  "
          f"first 3 calls {rec['first_calls_s']} s")
  log(f"head_bench done in {time.perf_counter() - t0:.0f} s")
  if args.out:
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "head_bench.json"), "w") as fh: json.dump({"vocab": V, "hidden": K, "results": results}, fh, indent=1)

if __name__ == "__main__": main()
