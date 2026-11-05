"""Paged decode attention on CUDA, kernel by kernel (docs/status/SPEED.md "Run 4"; a tool, not the harness).

  uv run python bench/decode_attn.py [--parts 32,64,128] [--max-len 2048] [--cases 1x600,8x600,32x600,32x1024,1x2000]

Qwen2.5-7B's decode attention (28 q heads, 4 kv heads, hd 128, bf16, block size 16): for each case (sequences x context)
one new token per sequence, with the pool sized for --max-len like the served engine (so the grid has the engine's
partition count), timed for the one-pass kernel at each --parts partition size and for the split kernel, as a burst of
back-to-back launches (partition + combine) between CUDA events, best of 5. Each one-pass output is checked against the
split kernel's (bf16 tolerance). Prints us per launch pair and the K/V read bandwidth; one layer's time x 28 is the
forward's attention.
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--parts", default="64,128,kpt,kptlate,tcd", help="one-pass partition sizes (optionally u<keys in flight>), "
                  "mma<part> for the 1-warp tensor-core decode, kpt for the key-per-thread one, tcd for the 4-warp tensor-core one")
  ap.add_argument("--max-len", type=int, default=2048)
  ap.add_argument("--cases", default="1x600,4x600,8x600,16x600,32x600,32x1024,32x1800,1x2000")
  a = ap.parse_args()
  from fusezap.device import Device, Buffer
  from fusezap.dtype import dtypes
  from fusezap.tensor import Tensor
  from fusezap.device.cuda_kernels import attention as A
  from fusezap.serve.kv_cache import PagedKVCache, StepLayout
  from fusezap.serve.weights import to_dtype
  dev = Device["CUDA"]
  cu = dev.cu
  HQ, HKV, D, bs = 28, 4, 128, 16
  G = HQ // HKV
  e0, e1 = cu.cuEventCreate(0)[1], cu.cuEventCreate(0)[1]
  def timeit(run, n: int = 200) -> float:
    run(); dev.synchronize()
    best = 1e9
    for _ in range(5):
      cu.cuEventRecord(e0, 0)
      for _ in range(n): run()
      cu.cuEventRecord(e1, 0)
      cu.cuEventSynchronize(e1)
      best = min(best, float(cu.cuEventElapsedTime(e0, e1)[1]) * 1e3 / n)
    return best
  rng = np.random.default_rng(0)
  print(f"{dev.gpu_name}: decode attention, Hq={HQ} Hkv={HKV} hd={D} bf16, pool for max_len {a.max_len}; us per layer (partition + combine)")
  for case in a.cases.split(","):
    B, ctx = (int(x) for x in case.split("x"))
    nblk = B * -(-a.max_len // bs)
    cache = PagedKVCache(1, HKV, D, nblk + 1, bs, dtypes.bfloat16, "CUDA", max_seq_len=a.max_len)
    for s in range(B): cache.ensure(s, ctx)
    pool = to_dtype(rng.standard_normal((2, nblk + 1, HKV, bs, D)).astype(np.float32), dtypes.bfloat16)
    pb = Tensor(pool.reshape(-1), device="CUDA", dtype=dtypes.bfloat16)._realized_contiguous().realized
    meta = cache.step_meta(B, B, [(s, ctx - 1) for s in range(B)], [])
    L = StepLayout(B, B, cache.max_blocks)
    mb = Tensor(meta, device="CUDA", dtype=dtypes.int32)._realized_contiguous().realized
    q = to_dtype(rng.standard_normal((B, HQ * D)).astype(np.float32), dtypes.bfloat16)
    xb = Tensor(q.reshape(-1), device="CUDA", dtype=dtypes.bfloat16)._realized_contiguous().realized
    args_ = (dtypes.bfloat16, B, HKV, G, D, bs, nblk + 1, L.MB, HQ * D, L.row, L.pos, L.bt, 1 / math.sqrt(D))
    kv_bytes = 2 * B * HKV * ctx * D * 2
    res, ref = [], None
    for kind in ["split"] + [f"p{p}" for p in a.parts.split(",")]:   # "pmma128": the tensor-core decode
      if kind == "split": pa, pc = A.split_calls(*args_)
      elif kind.startswith("pmma"): pa, pc = A.mdec_calls(*args_, part=int(kind[4:]))
      elif kind.startswith("pkpt"): pa, pc = A.kpt_calls(*args_, vpre=kind != "pkptlate")
      elif kind == "ptcd": pa, pc = A.tcd_calls(*args_)
      else:
        p_, _, u_ = kind[1:].partition("u")
        pa, pc = A.decode_calls(*args_, part=int(p_), U=int(u_) if u_ else 2)
      NP = A.call_np(pa) if kind.startswith(("pmma", "pkpt", "ptcd")) else pa.params[-1]
      acc, ml = Buffer("CUDA", B * HQ * NP * D, dtypes.float32), Buffer("CUDA", B * HQ * NP * 2, dtypes.float32)
      out = Buffer("CUDA", B * HQ * D, dtypes.bfloat16)
      r1, r2 = pa.prepare(dev, (acc, ml, xb, pb, mb)), pc.prepare(dev, (out, acc, ml, mb))
      def run(): r1(); r2()
      us = timeit(run)
      host = np.empty(B * HQ * D, np.uint16)
      out.copyout(memoryview(host).cast("B"))
      val = (host.astype(np.uint32) << 16).view(np.float32)
      if ref is None: ref, err = val, 0.0
      else: err = float(np.max(np.abs(val - ref) / (np.abs(ref) + 1)))
      res.append(f"{kind} {us:7.1f} us ({kv_bytes / us / 1e3:5.0f} GB/s, NP {NP}{', ERR %.1e' % err if err > 2e-2 else ''})")
    print(f"  {B:3d} x {ctx:5d}: " + " | ".join(res), flush=True)
  return 0

if __name__ == "__main__": sys.exit(main())
