"""NKI-N7 step 4: microbenchmarks on the NEURON device (nki_kernels.ProbeSpec), each kernel's device time by nrtpy's device
benchmark, two sizes per probe so the launch's fixed cost cancels: cost per instruction = (t(n2) - t(n1)) / (n2 - n1).

  FZ_DEVICE=NEURON .venv/bin/python scripts/nki_n7_probe.py --out results/nki_n7_step4
  FZ_DEVICE=NKISHIM uv run python scripts/nki_n7_probe.py --quick     # Mac dry run: every probe renders and runs on the shim

TensorE (float32 unless noted; m = GEMV rows):
  mv m      nc_matmul with (128, m) stationary and (128, 512) moving tiles (render_matmul's / HeadSpec's / LayerSpec's form)
  ws m      nc_matmul with (128, 128) stationary slices and (128, m) moving (the weight-stationary form: HeadSpec ws, LayerSpec mm=ws)
  mv16/ws16 the same in bfloat16 (the bf16 / fp32 rate ratio)
  tr m      nc_transpose of (m, 128) tiles (LayerSpec's hᵀ, gᵀ / uᵀ, kᵀ ... transposes)
  -> us per instruction, and ns per 1024 weight elements (mv: 128 x 512 per instruction, ws: 128 x 128): which form a
     float32 GEMV should use
DMA (no compute but one column read per tile): f floats per partition per block, n blocks
  blk       one DMA of (128, f) per block into one tile (HeadSpec W:1)
  grp4      four DMAs of 32 partitions into one tile (HeadSpec W:4: step 2 measured it slower)
  col4      four DMAs of (128, f/4) into four tiles (split by columns, separate tiles)
  run2k     f/512 DMAs of (128, 512), 2 KiB runs at row stride f (render_matmul's K-major weight tiles)
  -> us per block and GB/s
Writes <out>/probe.json and one line per probe: "probe <name> ...".
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def log(*a) -> None: print(*a, flush=True)

def probes(quick: bool) -> list[tuple[str, tuple]]:
  """(label, (spec at n1, spec at n2)) per probe."""
  from fusezap.codegen.nki_kernels import ProbeSpec as P
  te = (8, 32) if quick else (64, 512)
  ws = (16, 64) if quick else (256, 1024)
  dn, f = ((2, 4), 1024) if quick else ((4, 20), 7168)
  out = []
  for m in (1, 4):
    out.append((f"mv m={m}", tuple(P("mv", n, m) for n in te)))
    out.append((f"ws m={m}", tuple(P("ws", n, m) for n in ws)))
  out.append(("mv16 m=1", tuple(P("mv", n, 1, dt="bfloat16") for n in te)))
  out.append(("ws16 m=1", tuple(P("ws", n, 1, dt="bfloat16") for n in ws)))
  for m in (1, 64): out.append((f"tr m={m}", tuple(P("tr", n, m) for n in te)))
  for mode in ("blk", "grp4", "col4", "run2k"): out.append((f"dma {mode} f={f}", tuple(P("dma", n, f=f, mode=mode) for n in dn)))
  out.append((f"dma blk f={f // 4}", tuple(P("dma", 4 * n, f=f // 4, mode="blk") for n in dn)))
  return out

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="")
  ap.add_argument("--quick", action="store_true", help="small sizes (the Mac dry run)")
  ap.add_argument("--iters", type=int, default=20)
  args = ap.parse_args()
  from fusezap.device import Buffer, Device
  from fusezap.device.neuron import neuron_profile
  from fusezap.dtype import dtypes
  from fusezap.serve import neuron_kv as kv
  dev = Device.DEFAULT
  assert kv.is_neuron(dev), f"FZ_DEVICE must be a NEURON tier, got {dev}"
  D = Device[dev]
  al = getattr(D, "allocator", None)
  timed = hasattr(al, "profile")
  ps = probes(args.quick)
  big = max(sp.in_size() for _, pair in ps for sp in pair)
  outn = max(sp.out_size() for _, pair in ps for sp in pair)
  rng = np.random.default_rng(0)
  src = Buffer(dev, big, dtypes.float32).copyin(memoryview((rng.standard_normal(big) * 0.1).astype(np.float32)).cast("B"))
  dst = Buffer(dev, outn, dtypes.float32).ensure_allocated()
  t0 = time.perf_counter()
  errors: dict = {}
  res = []
  for label, (a, b) in ps:   # one probe at a time, results written as they come (a phase timeout keeps the earlier ones)
    if timed: al.flush(); al.profile, al.prof, al.prof_wall, al.prof_ipc, al.bindings = True, {}, [0, 0.0], [0, 0.0, 0.0], {}
    for sp in (a, b):
      try: D.run_program(D.spec_program(sp), (dst, src))
      except Exception as e:
        errors[sp.name] = f"{type(e).__name__}: {str(e)[:300]}"
        log(f"probe {sp.name} SKIPPED: {errors[sp.name]}")
    ms: dict = {}
    if timed:
      al.flush(); al.profile = False
      rep = neuron_profile(al, iters=args.iters, bench=True)
      ms = {nm: k["device_ms"] for nm, k in rep["kernels"].items()}
      for nm, e in (rep.get("bench_errors") or {}).items(): errors.setdefault(nm, e)
    r = {"probe": label, "kernels": [a.name, b.name], "n": [a.n, b.n], "device_ms": [ms.get(a.name), ms.get(b.name)]}
    if None not in r["device_ms"]:
      per = (r["device_ms"][1] - r["device_ms"][0]) * 1e3 / (b.n - a.n)   # us per instruction / block
      r["us_per"] = round(per, 4)
      r["fixed_us"] = round(r["device_ms"][0] * 1e3 - per * a.n, 2)
      if a.kind in ("mv", "ws"):
        el = TKTN if a.kind == "mv" else 128 * 128
        r["ns_per_1k_weights"] = round(per * 1e3 / (el / 1024), 3)
        if a.kind == "mv" and per > 0: r["g_moving_cols_per_s"] = round(512 / per / 1e3, 3)
      if a.kind == "dma": r["gb_per_s"] = round(128 * a.f * 4 / (per * 1e-6) / 1e9, 1) if per > 0 else None
    res.append(r)
    extra = "  ".join(f"{k} {r[k]}" for k in ("us_per", "fixed_us", "ns_per_1k_weights", "g_moving_cols_per_s", "gb_per_s") if k in r)
    log(f"probe {label:<18} device ms {r['device_ms']} at n {r['n']}  {extra}")
    if args.out:
      os.makedirs(args.out, exist_ok=True)
      with open(os.path.join(args.out, "probe.json"), "w") as fh: json.dump({"results": res, "errors": errors}, fh, indent=1)
  log(f"probe done in {time.perf_counter() - t0:.0f} s; {len(errors)} errors")

TKTN = 128 * 512

if __name__ == "__main__": main()
