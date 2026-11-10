"""N7 step 2: an engine-time estimate of the serving head's kernels (final norm, lm_head, argmax), v1 vs v2, from the
instructions they issue on NKISHIM (the real rendered sources, run with the shim; every nisa call recorded with its shapes).

  uv run python scripts/nki_head_cost.py [--rows 1,4] [--vocab 151936] [--hidden 896]

Model (NeuronCore-v2, trn1; assumptions, not measurements): TensorE float32 at 1/4 of the bf16 rate, 0.725 G moving
columns/s (95 TF bf16 per core = 128 x 128 x 2 x 2.9 G), + 64 cycles per instruction; VectorE/ScalarE/GpSimd one float32
element per lane per cycle at 1.4 GHz + 64 cycles per instruction (a PSUM read the same); DMA 410 GB/s per core for runs
of >= 512 B (shorter runs move whole 64 B lines), ~2 us fixed per DMA, 16 queues. Per kernel the engines overlap
perfectly (time = the busiest engine: the best a schedule can do) and kernels run one after the other (each reads the last
one's output). v1 calibrates it: N7 step 1 measured 2.16 ms (norm + lm_head) and 3.45 ms (+ argmax) per launch at 1 row.
"""
from __future__ import annotations
import argparse, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusezap.device import nkishim as S

TE_COLS, CLK, OVH, HBM, DMA_FIX, QUEUES = 0.725e9, 1.4e9, 64, 410e9, 2e-6, 16
REC: list = []

def _wrap(nm, f):
  def g(*a, **k):
    REC.append((nm, a, k))
    return f(*a, **k)
  return g

def install() -> None:
  for nm in ("dma_copy", "memset", "tensor_copy", "tensor_tensor", "tensor_scalar", "activation", "reciprocal", "tensor_reduce",
             "tensor_copy_predicated", "nc_matmul", "nc_transpose", "iota", "affine_select"):
    setattr(S.nisa, nm, _wrap(nm, getattr(S.isa, nm)))

def cost(rec: list) -> dict:
  eng = defaultdict(float)
  n = defaultdict(int)
  dma_bytes = dma_n = 0
  for nm, a, k in rec:
    dst = k.get("dst")
    if nm == "dma_copy":
      src = k["src"]
      hbm = src if src.buffer not in S.ON_CHIP else dst
      run = hbm.shape[-1] * hbm.dtype.itemsize if hbm.strides[-1] == 1 else hbm.dtype.itemsize
      b = hbm.size * hbm.dtype.itemsize
      dma_bytes += b if run >= 512 else b * max(1.0, 64 / run)
      dma_n += 1
      continue
    if nm in ("nc_matmul", "nc_transpose"):
      cols = k["moving"].shape[1] if nm == "nc_matmul" else k["data"].shape[0]
      eng["tensor"] += cols / TE_COLS + OVH / CLK
      n["tensor"] += 1
      continue
    t = dst if dst is not None else a[0]
    free = int(np.prod(t.shape[1:])) if t.ndim > 1 else 1
    e = "gpsimd" if k.get("engine") == S.engine.gpsimd else "vector"
    eng[e] += free / CLK + OVH / CLK
    n[e] += 1
  eng["dma"] = max(dma_bytes / HBM, dma_n * DMA_FIX / QUEUES)
  n["dma"] = dma_n
  return {"ms": 1e3 * max(eng.values()), "engines_ms": {e: round(1e3 * v, 3) for e, v in eng.items()}, "instr": dict(n)}

def measure(fn) -> dict:
  REC.clear(); fn(); return cost(list(REC))

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--rows", default="1,4")
  ap.add_argument("--vocab", type=int, default=151936)
  ap.add_argument("--hidden", type=int, default=896)
  args = ap.parse_args()
  os.environ.setdefault("FZ_DEVICE", "NKISHIM")
  install()
  from fusezap import Tensor
  from fusezap.codegen import nki_kernels as nk
  from fusezap.device.neuron import kmajor
  from fusezap.serve import neuron_kv as kv
  V, K, dev = args.vocab, args.hidden, "NKISHIM"
  rng = np.random.default_rng(0)
  w = (rng.standard_normal((V, K)) * 0.05).astype(np.float32)
  nw = Tensor((1 + 0.1 * rng.standard_normal(K)).astype(np.float32), device=dev).realize()
  wk = kmajor(Tensor(w, device=dev).realize())
  wt = Tensor(nk.head_tiles(w), device=dev).realize()
  for M in (int(r) for r in args.rows.split(",")):
    x = Tensor(rng.standard_normal((M, K)).astype(np.float32), device=dev).realize()
    d = Tensor(rng.standard_normal((M, K)).astype(np.float32), device=dev).realize()
    res = {}
    for head in ("v1", "v2", "v3"):
      os.environ["FZ_NEURON_HEAD"] = head
      h = {}
      norm = measure(lambda: h.setdefault("h", kv.rmsnorm(x, nw, 1e-6, d)[1].realize()))
      lg = {}
      mm = measure(lambda: lg.setdefault("lg", (kv.head_matmul(h["h"], wt, V) if head == "v2" else h["h"] @ wk).realize()))
      am = measure(lambda: kv.argmax_rows(lg["lg"]).realize())
      res[head] = (norm, mm, am, lg["lg"].numpy())
      tot = norm["ms"] + mm["ms"]
      print(f"rows {M} head {head}: norm {norm['ms']:.3f} + lm_head {mm['ms']:.3f} = {tot:.3f} ms; argmax {am['ms']:.3f} -> with argmax {tot + am['ms']:.3f} ms")
      for nm, c in (("lm_head", mm), ("argmax", am)): print(f"    {nm:8} engines {c['engines_ms']}  instructions {c['instr']}")
    for hd in ("v2", "v3"): print(f"rows {M}: {hd} logits bitwise == v1: {np.array_equal(res['v1'][3].view(np.int32), res[hd][3].view(np.int32))}")

if __name__ == "__main__": main()
