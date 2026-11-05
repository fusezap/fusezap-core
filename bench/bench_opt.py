"""P2-OPT benchmark: unoptimized vs hand-coded (FZ_OPT=1) vs FZ_BEAM kernels, timed on the device (GPU time per launch).

  FZ_DEVICE=METAL uv run python bench/bench_opt.py [--beam 4] [--reps 10]

Gate (exit code 1 if missed): the BEAM winner is >= 3x faster than the unoptimized kernel on every case marked
"gate" (matmul, and reduces with few outputs, where the unoptimized one-thread-per-output loop starves the GPU).
Cases marked "info" are printed for honesty: e.g. sum(-1) over 4096x4096 already runs near memory bandwidth unoptimized.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--beam", type=int, default=int(os.getenv("FZ_BEAM", "0") or 0) or 4)
  ap.add_argument("--reps", type=int, default=10)
  ap.add_argument("--rounds", type=int, default=3)
  ap.add_argument("--quick", action="store_true")
  args = ap.parse_args()
  os.environ.pop("FZ_BEAM", None)   # this script drives the search itself; realize() must stay unoptimized
  os.environ["FZ_CUSTOM"] = "0"     # benchmark the generated kernels, not the library matmul (P2-MTL)
  from fusezap import Tensor, Device
  from fusezap.device import Buffer
  from fusezap.realize import run_schedule
  from fusezap.schedule import create_schedule
  from fusezap.codegen.lower import Opts, heuristic_opts, beam_search, time_opts, lower
  dev = os.getenv("FZ_DEVICE", "METAL" if Device.available("METAL") else "CPU").upper()
  d = Device[dev]
  rng = np.random.default_rng(0)
  r = lambda *s: rng.standard_normal(s).astype(np.float32)
  cases = [("gate", "matmul 1024^3", lambda: (r(1024, 1024), r(1024, 1024)), lambda a, b: a @ b, lambda a, b: a @ b, lambda n: 2 * 1024 ** 3)]
  if not args.quick: cases.append(("gate", "matmul 2048^3", lambda: (r(2048, 2048), r(2048, 2048)), lambda a, b: a @ b, lambda a, b: a @ b, lambda n: 2 * 2048 ** 3))
  cases += [
    ("gate", "sum(-1) 512x32768", lambda: (r(512, 32768),), lambda a: a.sum(axis=-1), lambda a: a.sum(-1), None),
    ("gate", "sum() 2^24", lambda: (r(1 << 24),), lambda a: a.sum(), lambda a: a.sum(), None),
    ("info", "sum(-1) 4096x4096", lambda: (r(4096, 4096),), lambda a: a.sum(axis=-1), lambda a: a.sum(-1), None),
    ("info", "sum(0) 4096x4096", lambda: (r(4096, 4096),), lambda a: a.sum(axis=0), lambda a: a.sum(0), None),
  ]
  print(f"device {dev}, beam width {args.beam}, reps {args.reps} (best GPU time per launch)")
  hdr = f"{'':5}{'case':<20}{'unopt ms':>10}{'hand ms':>10}{'beam ms':>10}{'speedup':>9}  {'beam opts'}"
  print(hdr); print("-" * 110)
  ok = True
  for kind, name, mk, ffz, fnp, flops in cases:
    arrs = mk()
    t = ffz(*[Tensor(a, device=dev) for a in arrs])
    sched = create_schedule([t.lazydata]); run_schedule(sched[:-1])
    si = sched[-1]
    for b in si.bufs: b.ensure_allocated()
    out = Buffer(dev, si.bufs[0].size, si.bufs[0].dtype)
    bufs = [out] + list(si.bufs[1:])
    def tm(o):
      return time_opts(d, si.ast, o, bufs, si.var_vals, reps=args.reps)
    hand = heuristic_opts(si.ast, d.renderer)
    st = time.perf_counter()
    best = beam_search(d, si.ast, list(si.bufs), si.var_vals, args.beam, si.n_outputs)
    bt = time.perf_counter() - st
    # the GPU may be shared and its clocks vary: interleave the three kernels for a few rounds and keep each one's best
    t0 = th = tb = float("inf")
    for _ in range(args.rounds):
      t0, th, tb = min(t0, tm(Opts())), min(th, tm(hand)), min(tb, tm(best))
    # error vs a float64 reference, relative to the same op on |inputs| (the conditioning of a sum / dot product)
    ref = np.asarray(fnp(*[a.astype(np.float64) for a in arrs]))
    scale = np.maximum(np.asarray(fnp(*[np.abs(a).astype(np.float64) for a in arrs])), 1e-30)
    got = out.numpy().reshape(ref.shape)
    err = float((np.abs(got - ref) / scale).max())
    good = err < 1e-5
    sp = t0 / tb
    if kind == "gate" and (sp < 3 or not good): ok = False
    extra = f"  {flops(0) / tb / 1e9:.0f} GFLOP/s" if flops else f"  {arrs[0].nbytes / tb / 1e9:.0f} GB/s"
    print(f"{kind:5}{name:<20}{t0*1e3:10.3f}{th*1e3:10.3f}{tb*1e3:10.3f}{sp:8.1f}x  {best}{extra} (search {bt:.1f}s, err {err:.1e}{'' if good else ' WRONG'})")
  print("GATE", "PASS" if ok else "FAIL", "(beam >= 3x unoptimized on every gate case, outputs match numpy)")
  return 0 if ok else 1

if __name__ == "__main__": sys.exit(main())
