"""Prefill-shape GEMM sweep on CUDA sm_90 (docs/status/SPEED.md step 5): the bf16 linears of a model at M = the prefill
rows, timed per wgmma tile config, plus whether each config's output is bit-identical to the dispatch default's.

  uv run python bench/gemm_prefill.py [--M 256] [--hidden 3584 --inter 18944 --qkv 4608] [--all] [--top 8]

Launches rotate over enough weight copies to exceed L2 (as bench/gemm_small.py), in one CUDA graph per config.
"""
from __future__ import annotations
import argparse, itertools, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from fusezap import Device
from fusezap.device import Buffer
from fusezap.dtype import dtypes
import fusezap.device.cuda_kernels as ck
from fusezap.device.cuda_kernels import wgmma

def sweep_cfgs(dev) -> list:
  cs = [wgmma.WgCfg(BN, NC, S, r, w) for BN, NC, S, r, w in itertools.product((64, 128, 192, 256), (1, 2), (3, 4, 5, 6, 8), (True, False), (False, True))]
  return [c for c in cs if wgmma.valid(c, "bf16") and wgmma.smem_bytes(c) <= ck.smem_limit(dev)]

def bench(dev, M: int, N: int, K: int, cfgs: list, l2: int = 128 << 20) -> list[tuple[float, object, bool]]:
  cu = dev.cu
  ncp = max(2, -(-l2 // (N * K * 2)))
  rng = np.random.default_rng(0)
  ws = [Buffer("CUDA", N * K, dtypes.bfloat16).ensure_allocated() for _ in range(ncp)]
  for w in ws: w.copyin(memoryview(rng.integers(0x3c00, 0x3c80, N * K, dtype=np.uint16) ^ rng.integers(0, 2, N * K, dtype=np.uint16) << 15).cast("B"))
  x = Buffer("CUDA", M * K, dtypes.bfloat16).ensure_allocated()
  x.copyin(memoryview(rng.integers(0x3b00, 0x3c80, M * K, dtype=np.uint16) ^ rng.integers(0, 2, M * K, dtype=np.uint16) << 15).cast("B"))
  c = Buffer("CUDA", M * N, dtypes.float32).ensure_allocated()
  e0, e1 = cu.cuEventCreate(0)[1], cu.cuEventCreate(0)[1]
  ref, res = None, []
  for cfg in cfgs:
    try:
      fs = [ck.gemm(dev, c.handle, x.handle, w.handle, M, N, K, kind="bf16", out="f32", cfg=cfg, impl="wgmma", bind=True) for w in ws]
      fs[0](); dev.synchronize()
      out = c.numpy().copy()
      if ref is None: ref = out
      same = bool(np.array_equal(out.view(np.uint32), ref.view(np.uint32)))
      reps = max(2, 16 // ncp)
      g = dev.capture_graph(lambda: [f() for _ in range(reps) for f in fs])
      assert g is not None, "capture failed"
      g.launch(); dev.synchronize()
      cu.cuEventRecord(e0, 0)
      for _ in range(3): g.launch()
      cu.cuEventRecord(e1, 0); cu.cuEventSynchronize(e1)
      res.append((float(cu.cuEventElapsedTime(e0, e1)[1]) * 1e3 / (3 * reps * ncp), cfg, same))
    except Exception as ex: print(f"    {cfg}: failed {str(ex)[:120]}")
  return res

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--M", default="256")
  ap.add_argument("--hidden", type=int, default=3584); ap.add_argument("--inter", type=int, default=18944)
  ap.add_argument("--qkv", type=int, default=4608)
  ap.add_argument("--all", action="store_true"); ap.add_argument("--top", type=int, default=8)
  a = ap.parse_args()
  dev = Device["CUDA"]
  print(f"{dev.gpu_name}, {ck.sm_count(dev)} SMs, smem/block {ck.smem_limit(dev)}")
  H = a.hidden
  shapes = [("qkv", a.qkv, H), ("o", H, H), ("gate_up", 2 * a.inter, H), ("down", H, a.inter)]
  for M in [int(m) for m in a.M.split(",")]:
    tot_d, tot_b, tot_r = 0.0, 0.0, 0.0
    for nm, N, K in shapes:
      d = ck.wg_pick(dev, "bf16", M, N) if hasattr(ck, "wg_pick") else ck.wg_default("bf16")
      cands = list(dict.fromkeys([ck.wg_default("bf16"), d] + sweep_cfgs(dev)))
      res = bench(dev, M, N, K, cands)
      tf = 2 * M * N * K / 1e12
      t0 = next((t for t, c, _ in res if c == ck.wg_default("bf16")), float("nan"))
      dt = next((t for t, c, _ in res if c == d), float("nan"))
      res.sort(key=lambda r: r[0])
      tot_r += 28 * t0; tot_d += 28 * dt; tot_b += 28 * res[0][0]
      print(f"M={M} {nm} N={N} K={K}: old default {t0:.1f} us ({tf / t0 * 1e6:.0f} TFLOP/s), dispatch {dt:.1f} us {d}")
      for t, c, same in (res if a.all else res[:a.top]): print(f"    {t:8.1f} us {tf / t * 1e6:5.0f} TFLOP/s {'same' if same else 'DIFF'}  {c}")
      print(f"    bit-identical to the old default: {sum(s for _, _, s in res)}/{len(res)}")
    print(f"M={M}: one forward of linears = {tot_r / 1e3:.2f} ms old default, {tot_d / 1e3:.2f} ms dispatch, {tot_b / 1e3:.2f} ms best per shape")

if __name__ == "__main__": main()
