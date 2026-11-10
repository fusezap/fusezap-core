"""I96 W3 + W4: one eager klein transformer step on a NEURON tier, dry: every kernel recorded, nothing computed.

  PYTHONPATH=. python scripts/i96_dry_step.py --size 1024 --json out.json                  # Mac: launches, NEFF list, HBM
  PYTHONPATH=. python scripts/i96_dry_step.py --size 1024 --compile --target inf2 --jobs 8  # Linux + .venv-nki --cc: NEFFs

The step is exactly what ImageEngine / Flux2KleinPipeline.denoise runs per denoising step on NEURON (eager, realize per
block, the W1 routing: bf16 K-major weights through neuron_kv.matmul_bf16, float32 flash attention). The weights are
unallocated device buffers of the real shapes (5 double + 20 single blocks, 3.88 B parameters) and the device's `run`
is replaced by a recorder, so nothing is computed and no data is copied. The allocator still allocates every buffer the
eager step would hold (np.empty: virtual pages, never touched), so its peak is the step's device memory.

Outputs:
  - launches per step, distinct NEFFs (kernel + argument sizes, the persistent path's compile key without versions);
  - W3: peak bytes in use on the device (weights + the step's live intermediates, + the free list separately) against
    the 15.00 GiB one runtime worker can allocate on an inf2 core (I69 7B run 1);
  - with --compile (W4): each distinct kernel compiled to a NEFF for --target with neuronx-cc through the persistent
    path's own cache (neuron.cached_neff, FZ_NEFF_CACHE or ~/.cache/fusezap/neff), --jobs at a time, with seconds,
    NEFF bytes and failures per kernel. The cache directory is what the chip job (scripts/nki_klein_inf2.sh) unpacks.
The tier is NKISHIM (any host). Its rendered sources are the NEURON tier's (same renderer and kernel templates), so the
keys match the chip's for the same nki / neuronx-cc versions and target; the chip job reports its cache hits.
"""
from __future__ import annotations
import argparse, json, math, os, sys, threading, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusezap.tensor import Tensor
from fusezap.dtype import dtypes, to_np
from fusezap.device import Buffer, Device
from fusezap.lazy import LazyBuffer

DEV = "NKISHIM"
GiB = 1 << 30

def empty(shape, dt=dtypes.float32) -> Tensor:
  return Tensor(LazyBuffer.from_buffer(Buffer(DEV, math.prod(shape), dt), tuple(shape)))

def transformer(dt):
  """The real klein config (transformer/config.json of FLUX.2-klein-4B) over unallocated weights, as __init__ lays them out."""
  from fusezap.serve.models.flux2 import Flux2Config, Flux2Transformer
  c = Flux2Config(num_layers=5, num_single_layers=20, num_heads=24, head_dim=128, joint_attention_dim=7680, guidance_embeds=False)
  D, M, hd = c.dim, c.mlp_hidden, c.head_dim
  tr = Flux2Transformer.__new__(Flux2Transformer)
  tr.cfg, tr.dtype, tr.device, tr.realize_blocks = c, dt, DEV, True
  tr.kmajor = dt == dtypes.bfloat16   # Flux2Transformer.__init__ on a NEURON tier (FZ_KLEIN_NEURON_BF16MM=1)
  def shape(n: str) -> tuple:
    if n.endswith(("norm_q.weight", "norm_k.weight", "norm_added_q.weight", "norm_added_k.weight")): return (hd,)
    if "to_qkv_mlp_proj" in n: return (3 * D + 2 * M, D)
    if n.endswith("attn.to_out.weight"): return (D, D + M)
    if "linear_in" in n: return (2 * M, D)
    if "linear_out" in n: return (D, M)
    if n.startswith("double_stream_modulation"): return (6 * D, D)
    if n.startswith("single_stream_modulation"): return (3 * D, D)
    return {"x_embedder.weight": (D, 128), "context_embedder.weight": (D, 7680), "proj_out.weight": (128, D),
            "norm_out.linear.weight": (2 * D, D), "time_guidance_embed.timestep_embedder.linear_1.weight": (D, 256)}.get(n, (D, D))
  tr.w = {}
  for n in tr.weight_names():
    s = shape(n)
    tr.w[n] = empty(s[::-1] if tr.kmajor and len(s) == 2 else s, dt)
  tr._tfreq = empty((128,))
  return tr

def dry_step(size: int, dt) -> tuple[list, dict]:
  """Run one eager step with Device[DEV].run recording instead of executing. Returns (items, memory stats)."""
  dev = Device[DEV]
  al = dev.allocator
  al.free_cache()
  items: list = []
  peak = {"in_use": 0, "held": 0}
  orig_alloc = al.alloc
  def alloc(nbytes: int):
    h = orig_alloc(nbytes)
    peak["in_use"] = max(peak["in_use"], al.held - al.cached); peak["held"] = max(peak["held"], al.held)
    return h
  orig_run = dev.run
  al.alloc = alloc
  dev.run = lambda si: items.append(record(si))   # the record only: a kept ScheduleItem would keep its buffers alive
  try:
    tr = transformer(dt)
    T, g = 512, size // 16
    for w in tr.w.values(): w.lazydata.realized.ensure_allocated()
    tr._tfreq.lazydata.realized.ensure_allocated()
    weights = al.held - al.cached
    cos, sin = empty((T + g * g, tr.cfg.head_dim)), empty((T + g * g, tr.cfg.head_dim))
    x, e, t = empty((1, g * g, 128)), empty((1, T, 7680)), empty((1,))
    t0 = time.perf_counter()
    v = tr.forward(x, e, t, cos, sin)
    v = v.float()._realized_contiguous()
    wall = time.perf_counter() - t0
  finally:
    dev.run, al.alloc = orig_run, orig_alloc
  mem = {"weights_bytes": weights, "peak_in_use_bytes": peak["in_use"], "peak_held_bytes": peak["held"],
         "step_transient_peak_bytes": peak["in_use"] - weights, "record_s": round(wall, 1)}
  return items, mem

def record(si) -> tuple | None:
  """(NKIProgram, argument specs) of one launch, as NEURONDevice.run_neff would bind it; None for a copy."""
  dev = Device[DEV]
  if si.kind == "copy": return None
  call = si.ast[0]
  if si.kind == "custom":
    if (sp := getattr(call, "spec", None)) is not None: prog = sp.render()
    else: prog = dev.custom_program(call.kernel.name, call.params).prog
  else: prog = dev.renderer.render(si.ast, si.var_vals)
  n = prog.n_bufs
  specs = [((b.size, to_np(b.dtype).str) if b.size else (1, to_np(b.dtype).str)) for b in si.bufs[:n]]
  specs += [(1, np.dtype(np.uint8).str)] * (n - len(specs))
  return (prog, specs)

def programs(items: list) -> list[dict]:
  """One record per distinct (kernel source, argument sizes): what the persistent path compiles once."""
  out: dict = {}
  for it in items:
    if it is None: continue
    prog, specs = it
    k = (prog.name, prog.src, tuple(specs))
    if k not in out:
      out[k] = {"kernel": prog.name, "prog": prog, "specs": specs, "launches": 0, "src_lines": prog.src.count("\n"),
                "arg_bytes": int(sum(c * np.dtype(d).itemsize for c, d in specs))}
    out[k]["launches"] += 1
  return list(out.values())

def compile_all(recs: list[dict], target: str, jobs: int, log, save=None) -> None:
  from fusezap.device import neuron as N
  versions = N._nki_versions()
  q, lock = sorted(recs, key=lambda r: r["src_lines"]), threading.Lock()   # small first: failures of the big ones show last
  def loop() -> None:
    w = N.compile_worker() if jobs > 1 else None
    if w is not None: N._TLS.worker = w
    try:
      while True:
        with lock:
          if not q: return
          r = q.pop(0)
        key = N.neff_key(r["prog"], r["specs"], target, versions)
        r["key"], r["hit"] = key, (N.neff_cache_dir() / key / "kernel.neff").exists()
        t0 = time.perf_counter()
        try:
          path, meta = N.cached_neff(r["prog"], r["specs"], target, versions)
          r.update(ok=True, seconds=round(time.perf_counter() - t0, 1), neff_bytes=path.stat().st_size)
        except Exception as e:
          r.update(ok=False, seconds=round(time.perf_counter() - t0, 1), error=f"{type(e).__name__}: {e}"[:400])
        log(f"NEFF {r['kernel']} {'hit' if r.get('hit') else 'compiled' if r['ok'] else 'FAILED'} {r['seconds']} s "
            f"({r['src_lines']} source lines, {r.get('neff_bytes', 0) / 2**20:.1f} MiB){'' if r['ok'] else ' ' + r['error'][:200]}")
        if save is not None:
          with lock: save()   # after every kernel: a phase that times out still leaves its per-kernel record
    finally:
      N._TLS.worker = None
      if w is not None:
        try: w.close()
        except Exception: pass
  ts = [threading.Thread(target=loop) for _ in range(max(1, jobs))]
  for t in ts: t.start()
  for t in ts: t.join()

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--size", type=int, default=1024)
  ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
  ap.add_argument("--compile", action="store_true", help="compile every distinct kernel to a NEFF (needs .venv-nki with neuronx-cc)")
  ap.add_argument("--target", default=os.environ.get("FZ_NEURON_TARGET", "inf2"))
  ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 4))
  ap.add_argument("--json", default=None)
  a = ap.parse_args()
  def log(*m) -> None: print(*m, flush=True)
  items, mem = dry_step(a.size, getattr(dtypes, a.dtype))
  def save() -> None:
    if a.json:
      with open(a.json, "w") as f:
        json.dump({"size": a.size, "dtype": a.dtype, "launches": len(kern), "distinct": len(recs), "memory": mem, "target": a.target if a.compile else None,
                   "kernels": [{k: v for k, v in r.items() if k != "prog"} for r in recs]}, f, indent=1)
  recs = programs(items)
  kern = [it for it in items if it is not None]
  log(f"klein step {a.size}^2 {a.dtype}: {len(kern)} launches, {len(recs)} distinct NEFFs, {len(items) - len(kern)} copies "
      f"(recorded in {mem['record_s']} s)")
  log(f"device memory (W3): weights {mem['weights_bytes'] / GiB:.2f} GiB, peak in use {mem['peak_in_use_bytes'] / GiB:.2f} GiB "
      f"(step transient {mem['step_transient_peak_bytes'] / GiB:.2f} GiB), peak held incl. free list {mem['peak_held_bytes'] / GiB:.2f} GiB; "
      f"one inf2 core allocates 15.00 GiB (FZ_NEURON_HBM_GB default 15.25)")
  big = sorted(recs, key=lambda r: -r["src_lines"])[:5]
  log("largest kernels (source lines): " + ", ".join(f"{r['kernel']} {r['src_lines']}" for r in big))
  if a.compile:
    t0 = time.perf_counter()
    compile_all(recs, a.target, a.jobs, log, save)
    ok = [r for r in recs if r.get("ok")]
    log(f"compile {a.target}: {len(ok)}/{len(recs)} ok, {sum(1 for r in recs if r.get('hit'))} cache hits, "
        f"{time.perf_counter() - t0:.0f} s wall at {a.jobs} jobs, {sum(r.get('seconds', 0) for r in recs):.0f} s summed, "
        f"NEFFs {sum(r.get('neff_bytes', 0) for r in recs) / 2**20:.0f} MiB")
  save()

if __name__ == "__main__": main()
