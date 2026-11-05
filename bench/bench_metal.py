"""P2-MTL benchmark: the Metal simdgroup-matrix matmul (and flash attention) vs torch MPS.

  uv run python bench/bench_metal.py [--sizes 1024 2048 4096] [--rounds 5] [--quick]

Both sides are timed the same way: wall time of a ~50 ms burst of back-to-back launches followed by one synchronize, best
burst of several interleaved rounds with an idle gap before each (the GPU may be shared, and sustained load lowers its
clock). The fusezap kernel is launched through
METALDevice.run_custom_call, the path a kind="custom" ScheduleItem takes. Each result is checked against torch's.

Gate (exit code 1 if missed): float32 matmul at 4096^2 reaches >= 60% of torch MPS throughput (SPEC §11.1).
float16/bfloat16 matmul and flash attention are printed for information (on M5-class GPUs MPS runs half precision
GEMM on the GPU neural accelerators, which simdgroup matrices do not use).
"""
from __future__ import annotations
import argparse, math, sys, time
import numpy as np

GATE_SIZE, GATE_RATIO = 4096, 0.60

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
  ap.add_argument("--rounds", type=int, default=6)
  ap.add_argument("--gap", type=float, default=0.15, help="idle seconds before each timed burst")
  ap.add_argument("--quick", action="store_true", help="float32 matmul only")
  args = ap.parse_args()
  import torch
  from fusezap import Tensor, dtypes
  from fusezap.device import Device, Buffer
  from fusezap.device.metal import attention
  if not Device.available("METAL") or not torch.backends.mps.is_available():
    print("needs Metal and torch MPS"); return 1
  d = Device["METAL"]
  TDT = {dtypes.float32: torch.float32, dtypes.float16: torch.float16, dtypes.bfloat16: torch.bfloat16}

  def burst(fn, n: int, sync) -> float:
    st = time.perf_counter()
    for _ in range(n): fn()
    sync()
    return (time.perf_counter() - st) / n

  def race(ours, theirs, flops: float) -> tuple[float, float]:
    """Best TFLOPS of each side over interleaved rounds (after a warmup burst). Bursts are ~50 ms with an idle gap
    before each: sustained load lowers the GPU clock, which would penalize whichever side runs later."""
    t_us, t_th = burst(ours, 2, d.synchronize), burst(theirs, 2, torch.mps.synchronize)
    n = max(3, min(200, int(0.05 / max(t_us, t_th, 1e-6))))
    for _ in range(args.rounds):
      time.sleep(args.gap)
      t_us = min(t_us, burst(ours, n, d.synchronize))
      time.sleep(args.gap)
      t_th = min(t_th, burst(theirs, n, torch.mps.synchronize))
    return flops / t_us / 1e12, flops / t_th / 1e12

  ok, gate_ratio = True, None
  print(f"{'case':<28}{'fusezap TFLOPS':>16}{'MPS TFLOPS':>12}{'ratio':>8}{'max rel err':>13}")
  print("-" * 77)
  dts = [dtypes.float32] if args.quick else [dtypes.float32, dtypes.float16, dtypes.bfloat16]
  for dt in dts:
    for n in args.sizes:
      ta, tb = torch.randn(n, n, device="mps", dtype=TDT[dt]), torch.randn(n, n, device="mps", dtype=TDT[dt])
      A, B, C = (Buffer("METAL", n * n, dt) for _ in range(3))
      A.copyin(memoryview(ta.cpu().view(torch.int16 if dt != dtypes.float32 else torch.float32).numpy()).cast("B"))
      B.copyin(memoryview(tb.cpu().view(torch.int16 if dt != dtypes.float32 else torch.float32).numpy()).cast("B"))
      params = (1, n, n, n, 0, n, 1, 0, n, 1, 0, n, 1, 0, 0)
      ours = lambda: d.run_custom_call("fz_matmul", params, (C, A, B), wait=False)
      theirs = lambda: ta @ tb
      us, th = race(ours, theirs, 2 * n ** 3)
      d.synchronize()
      ref = (ta @ tb).float().cpu().numpy()
      raw = C.numpy()
      got = raw if dt == dtypes.float32 else (raw.view(np.float16).astype(np.float32) if dt == dtypes.float16 else
                                               (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32))
      err = float(np.abs(got.reshape(n, n) - ref).max() / max(np.abs(ref).max(), 1e-30))
      tol = 1e-4 if dt == dtypes.float32 else 1e-2
      ratio = us / th
      tag = ""
      if dt == dtypes.float32 and n == GATE_SIZE:
        gate_ratio, tag = ratio, ("  GATE ok" if ratio >= GATE_RATIO else "  GATE MISSED")
        ok &= ratio >= GATE_RATIO
      if err > tol: ok, tag = False, tag + "  WRONG"
      print(f"{'matmul ' + dt.name + f' {n}^2':<28}{us:>16.2f}{th:>12.2f}{ratio:>8.2f}{err:>13.1e}{tag}", flush=True)
  if not args.quick:
    for dt in (dtypes.float32, dtypes.float16):
      for causal in (False, True):
        Bh, L, D = 8, 2048, 64
        tq, tk, tv = (torch.randn(1, Bh, L, D, device="mps", dtype=TDT[dt]) for _ in range(3))
        q, k, v = (Tensor(x.float().cpu().numpy(), device="METAL").cast(dt).realize() for x in (tq, tk, tv))
        out = attention(q, k, v, causal=causal)
        O = out.lazydata.realized
        Q, K, V = (x._realized_contiguous().realized for x in (q, k, v))
        params = (Bh, L, L, D, D, L * D, D, 1, L * D, D, 1, L * D, D, 1, L * D, D, 1, 0, 0, 0,
                  int(np.float32(1 / math.sqrt(D)).view(np.int32)), int(np.float32(math.log2(math.e)).view(np.int32)))
        name = "fz_attention_causal" if causal else "fz_attention"
        ours = lambda: d.run_custom_call(name, params, (O, Q, K, V), wait=False)
        theirs = lambda: torch.nn.functional.scaled_dot_product_attention(tq, tk, tv, is_causal=causal)
        flops = 4 * Bh * L * L * D * (0.5 if causal else 1)
        us, th = race(ours, theirs, flops)
        d.synchronize()
        ref = theirs().float().cpu().numpy().reshape(Bh, L, D)
        got = out.cast(dtypes.float32).numpy().reshape(Bh, L, D)
        err = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
        tag = "  WRONG" if err > (1e-3 if dt == dtypes.float32 else 2e-2) else ""
        ok &= not tag
        print(f"{'attn ' + dt.name + (' causal' if causal else '') + f' {Bh}x{L}x{D}':<28}{us:>16.2f}{th:>12.2f}{us / th:>8.2f}{err:>13.1e}{tag}", flush=True)
  print("-" * 77)
  if gate_ratio is not None: print(f"gate: float32 matmul {GATE_SIZE}^2 at {gate_ratio:.2f}x torch MPS (need >= {GATE_RATIO:.2f}): {'PASS' if gate_ratio >= GATE_RATIO else 'FAIL'}")
  else: print(f"gate size {GATE_SIZE} not benchmarked")
  return 0 if ok and gate_ratio is not None else 1

if __name__ == "__main__":
  sys.exit(main())
