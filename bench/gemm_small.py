"""Decode-shape GEMM sweep on CUDA (docs/status/SPEED.md): the bf16 small-M linears of a model, timed per tile config.

  uv run python bench/gemm_small.py [--M 1,32] [--hidden 3584 --inter 18944 --qkv 4608 --vocab 152064] [--all] [--top 6] [--swap]
  # one tensor-parallel rank of Llama-3.3-70B at TP=2 (I57 M3): o_proj K = 4096, the intermediate halved, 80 layers
  uv run python bench/gemm_small.py --swap --quick --M 1,8,32 --hidden 8192 --inter 14336 --qkv 5120 --o-k 4096 --vocab 128256 --layers 80

Each shape (M, N, K) runs as a CUDA graph of launches that rotate over enough weight copies to exceed L2, so the time is
the HBM-streaming time a decode step sees. Prints us per launch, the weight bandwidth in TB/s, the mma.sync default
(ck.mma_small_default) and the fastest configs; --all prints every candidate. --swap (sm_90) times the swap-AB wgmma configs
(ck.wg_swap_candidates) against the same mma.sync default, and reports the swap rule (ck.wg_swap_default(table=False)) and
the per-shape table's config (ck.gemm_table: tuned at engine start or shipped; = the rule without an entry) too.
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from fusezap import Device
from fusezap.device import Buffer
from fusezap.dtype import dtypes
import fusezap.device.cuda_kernels as ck
from fusezap.device.cuda_kernels import mma

def bench(dev, M: int, N: int, K: int, out: str, cfgs: list, l2: int = 128 << 20) -> list[tuple[float, object]]:
  """cfgs: MmaCfg (mma.sync) or wgmma.SwapCfg (swap-AB wgmma) entries."""
  cu = dev.cu
  ncp = max(2, -(-l2 // (N * K * 2)))
  ws = [Buffer("CUDA", N * K, dtypes.bfloat16).ensure_allocated() for _ in range(ncp)]
  rng = np.random.default_rng(0)
  for w in ws: w.copyin(memoryview(rng.integers(0x3c00, 0x3c80, N * K, dtype=np.uint16)).cast("B"))
  x = Buffer("CUDA", M * K, dtypes.bfloat16).ensure_allocated()
  x.copyin(memoryview(rng.integers(0x3c00, 0x3c80, M * K, dtype=np.uint16)).cast("B"))   # defined inputs (was uninitialized memory)
  c = Buffer("CUDA", M * N, dtypes.float32 if out == "f32" else dtypes.bfloat16).ensure_allocated()
  res = []
  e0, e1 = cu.cuEventCreate(0)[1], cu.cuEventCreate(0)[1]
  for cfg in cfgs:
    try:
      fs = [ck.gemm(dev, c.handle, x.handle, w.handle, M, N, K, kind="bf16", out=out, cfg=cfg,
                     impl="mma" if isinstance(cfg, mma.MmaCfg) else "wgswap", bind=True) for w in ws]
      reps = max(4, 64 // ncp)
      for f in fs: f()
      g = dev.capture_graph(lambda: [f() for _ in range(reps) for f in fs])
      assert g is not None, "capture failed"
      g.launch(); dev.synchronize()
      cu.cuEventRecord(e0, 0)
      for _ in range(3): g.launch()
      cu.cuEventRecord(e1, 0); cu.cuEventSynchronize(e1)
      res.append((float(cu.cuEventElapsedTime(e0, e1)[1]) * 1e3 / (3 * reps * ncp), cfg))
    except Exception as ex: print(f"    {cfg}: failed {str(ex)[:120]}")
  return res

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--M", default="1,32")
  ap.add_argument("--hidden", type=int, default=3584); ap.add_argument("--inter", type=int, default=18944)
  ap.add_argument("--qkv", type=int, default=4608); ap.add_argument("--vocab", type=int, default=152064)
  ap.add_argument("--all", action="store_true"); ap.add_argument("--top", type=int, default=6); ap.add_argument("--swap", action="store_true")
  ap.add_argument("--o-k", type=int, default=0, help="o_proj's K (default hidden; a TP rank holds hidden / tp)")
  ap.add_argument("--layers", type=int, default=28, help="layers per forward (the per-forward totals)")
  ap.add_argument("--quick", action="store_true", help="--swap: NC 1, stages 4 / 6 / 8 / 12, splits 1-4 and the default only")
  a = ap.parse_args()
  dev = Device["CUDA"]
  print(f"{dev.gpu_name}, {ck.sm_count(dev)} SMs, smem/block {ck.smem_limit(dev)}")
  H = a.hidden
  shapes = [("qkv", a.qkv, H, "bf16"), ("o", H, a.o_k or H, "bf16"), ("gate_up", 2 * a.inter, H, "bf16"), ("down", H, a.inter, "bf16"),
            ("lm_head", a.vocab, H, "f32")]
  for M in [int(m) for m in a.M.split(",")]:
    tot_d, tot_b, tot_s, tot_t = 0.0, 0.0, 0.0, 0.0
    for nm, N, K, out in shapes:
      d = ck.mma_small_default(dev, "bf16", False, M, N, 1, K)
      sd = ck.wg_swap_default(dev, "bf16", M, N, K, table=False) if a.swap else None   # the rule
      tb = ck.wg_swap_default(dev, "bf16", M, N, K) if a.swap else None                # the per-shape table's (= sd without an entry)
      sw = ck.wg_swap_candidates(dev, "bf16", M, N, K) if a.swap else []
      if a.quick: sw = [c for c in sw if c.NC == 1 and c.S in (4, 6, 8, 12) and c.SK <= 4]
      cands = list(dict.fromkeys([d] + ([sd, tb] + sw if a.swap else ck.mma_small_sweep(dev, "bf16", False, M))))
      res = bench(dev, M, N, K, out, cands)
      gb = N * K * 2 / 1e9
      dt = next((t for t, c in res if c == d), float("nan"))
      res.sort(key=lambda r: r[0])
      w = 1 if nm == "lm_head" else a.layers
      tot_d += w * dt; tot_b += w * res[0][0]
      print(f"M={M} {nm} N={N} K={K}: default {dt:.1f} us ({gb / dt * 1e3:.2f} TB/s) {d}")
      if sd is not None:
        st = next((t for t, c in res if c == sd), float("nan"))
        tt = next((t for t, c in res if c == tb), float("nan"))
        tot_s += w * st; tot_t += w * tt
        print(f"    swap default {st:.1f} us ({gb / st * 1e3:.2f} TB/s) {sd}")
        print(f"    table {tt:.1f} us ({gb / tt * 1e3:.2f} TB/s) {tb}")
      for t, c in (res if a.all else res[:a.top]): print(f"    {t:8.1f} us {gb / t * 1e3:5.2f} TB/s  {c}")
    print(f"M={M}: one forward of linears = {tot_d / 1e3:.2f} ms with the mma defaults, {tot_b / 1e3:.2f} ms with the best per shape"
          + (f", {tot_s / 1e3:.2f} ms with the swap defaults, {tot_t / 1e3:.2f} ms with the table" if a.swap else ""))

if __name__ == "__main__": main()
