"""Can a decode row ride in a prefill GEMM and keep its bits? (docs/status/SPEED.md "Run 4", plan item 7, option B)

  uv run python bench/korder_check.py

For Qwen2.5-7B's linears (qkv, o, gate_up, down) on sm_90: X (256, K) bf16 through the prefill GEMM (the wgmma tile the
dispatch picks at M = 256), and the same rows in blocks of 32 through the decode swap-AB kernel, with its default split
(SK as served) and with SK = 1. Prints, per shape and output dtype (f32 accumulator, bf16), the fraction of elements
that are bit-identical. Option B needs 100% for the SK = 1 swap at every shape.
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

def main() -> int:
  from fusezap import Device
  from fusezap.device import Buffer
  from fusezap.dtype import dtypes
  import fusezap.device.cuda_kernels as ck
  from fusezap.device.cuda_kernels import wgmma
  dev = Device["CUDA"]
  if tuple(dev.cc) != (9, 0): print("needs sm_90"); return 0
  rng = np.random.default_rng(0)
  M, B = 256, 32
  shapes = {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}
  def bf16(a): return (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
  ok_all = True
  for name, (N, K) in shapes.items():
    x = Buffer("CUDA", M * K, dtypes.bfloat16).ensure_allocated(); x.copyin(memoryview(bf16(rng.standard_normal((M, K)) * 0.5)).cast("B"))
    w = Buffer("CUDA", N * K, dtypes.bfloat16).ensure_allocated(); w.copyin(memoryview(bf16(rng.standard_normal((N, K)) * 0.02)).cast("B"))
    for out, odt, npdt in (("f32", dtypes.float32, np.float32), ("bf16", dtypes.bfloat16, np.uint16)):
      es = odt.itemsize
      def run(impl, cfg_fn):
        c = Buffer("CUDA", M * N, odt).ensure_allocated()
        if impl == "pre": ck.gemm(dev, c.handle, x.handle, w.handle, M, N, K, kind="bf16", out=out)
        else:
          for r0 in range(0, M, B):
            ck.gemm(dev, c.handle + r0 * N * es, x.handle + r0 * K * 2, w.handle, B, N, K, kind="bf16", out=out, impl="wgswap", cfg=cfg_fn())
        dev.synchronize()
        h = np.empty(M * N, npdt); c.copyout(memoryview(h).cast("B"))
        return h
      pre = run("pre", None)
      d = ck.wg_swap_default(dev, "bf16", B, N, K)
      res = []
      for label, cf in (("swap default (SK %d)" % d.SK, lambda: d), ("swap SK 1", lambda: wgmma.SwapCfg(d.NC, d.BX, d.S, 1))):
        got = run("swap", cf)
        frac = float(np.mean(got == pre))
        res.append(f"{label}: {100 * frac:.3f}% bit-identical")
        if "SK 1" in label: ok_all &= frac == 1.0
      print(f"{name:8s} N={N:6d} K={K:6d} out {out:4s}: " + "; ".join(res), flush=True)
  print("option B (one K order): " + ("POSSIBLE: the SK = 1 swap kernel matches the prefill tile bit for bit" if ok_all else "NOT as is: the swap kernel and the prefill tile round differently"))
  return 0

if __name__ == "__main__": sys.exit(main())
