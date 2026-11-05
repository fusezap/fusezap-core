"""Op micro-benchmarks: fusezap vs torch CPU vs torch MPS (elementwise, reduce, matmul 256..4096, softmax). Prints a table.

  uv run python bench/bench_ops.py [--quick] [--device CPU|METAL] [--only matmul,softmax]
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch

def _sync(dev: str):
  if dev == "mps": torch.mps.synchronize()

def time_torch(fn, dev: str, iters: int = 10, warmup: int = 3) -> float:
  for _ in range(warmup): fn()
  _sync(dev); t = time.perf_counter()
  for _ in range(iters): fn()
  _sync(dev); return (time.perf_counter() - t) / iters

def time_fz(fn, dev: str, iters: int = 10, warmup: int = 3, sync=None) -> float:
  """realize only dispatches (Program wait=False), so synchronize the fusezap device before reading the clock."""
  if sync is None:
    from fusezap import Device
    sync = Device[dev].synchronize
  for _ in range(warmup): fn().realize()
  sync(); t = time.perf_counter()
  for _ in range(iters): fn().realize()
  sync(); return (time.perf_counter() - t) / iters

def cases(quick: bool):
  """(group, name, shape-spec, fusezap fn(inputs), torch fn(inputs), flops, bytes)."""
  ew = [1 << 20] if quick else [1 << 16, 1 << 20, 1 << 24]
  out = []
  for n in ew:
    out.append(("elementwise", f"a*b+c n={n}", [(n,)] * 3, lambda a, b, c: a * b + c, lambda a, b, c: a * b + c, 2 * n, 16 * n))
    out.append(("elementwise", f"gelu n={n}", [(n,)], lambda a: a.gelu(), lambda a: torch.nn.functional.gelu(a, approximate="tanh"), 10 * n, 8 * n))
  for r, c in ([(1024, 1024)] if quick else [(256, 256), (1024, 1024), (4096, 4096)]):
    out.append(("reduce", f"sum(-1) {r}x{c}", [(r, c)], lambda a: a.sum(axis=-1), lambda a: a.sum(-1), r * c, 4 * r * c))
    out.append(("reduce", f"sum(0) {r}x{c}", [(r, c)], lambda a: a.sum(axis=0), lambda a: a.sum(0), r * c, 4 * r * c))
    out.append(("softmax", f"softmax {r}x{c}", [(r, c)], lambda a: a.softmax(), lambda a: a.softmax(-1), 5 * r * c, 8 * r * c))
  for n in ([256, 1024] if quick else [256, 512, 1024, 2048, 4096]):
    out.append(("matmul", f"{n}x{n}x{n}", [(n, n), (n, n)], lambda a, b: a @ b, lambda a, b: a @ b, 2 * n ** 3, 12 * n * n))
  return out

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--quick", action="store_true"); ap.add_argument("--device", default=None); ap.add_argument("--only", default="")
  ap.add_argument("--iters", type=int, default=10)
  args = ap.parse_args()
  from fusezap import Tensor, Device
  dev = args.device or ("METAL" if Device.available("METAL") else "CPU")
  tdevs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
  only = {s for s in args.only.split(",") if s}
  rng = np.random.default_rng(0)
  hdr = f"{'op':<12}{'case':<26}{'fz['+dev+'] ms':>14}" + "".join(f"{'torch '+d+' ms':>16}" for d in tdevs) + f"{'GFLOP/s fz':>12}{'GB/s fz':>10}"
  print(hdr); print("-" * len(hdr))
  for group, name, shapes, ffz, ftc, flops, nbytes in cases(args.quick):
    if only and group not in only: continue
    arrs = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    row = f"{group:<12}{name:<26}"
    try:
      ins = [Tensor(a, device=dev).realize() for a in arrs]
      t = time_fz(lambda: ffz(*ins), dev, args.iters); row += f"{t * 1e3:>14.3f}"
      gf, gb = flops / t / 1e9, nbytes / t / 1e9
    except Exception as e:  # unimplemented until WP2 lands
      row += f"{'ERR':>14}"; gf = gb = float("nan"); print(f"  [fusezap] {type(e).__name__}: {str(e)[:80]}")
    for d in tdevs:
      tins = [torch.tensor(a, device=d) for a in arrs]
      row += f"{time_torch(lambda: ftc(*tins), d, args.iters) * 1e3:>16.3f}"
    print(row + f"{gf:>12.1f}{gb:>10.1f}")

if __name__ == "__main__": main()
