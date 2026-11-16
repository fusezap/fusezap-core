"""I93: FLUX.2 [klein] transformer s/step on a NeuronCore, main vs the I93 speed variants, in one process.

  python scripts/i93_klein_speed.py --model <klein dir> --refs .scratch/i96/refs --sizes 512,1024 \
      --variants "main=;rope=FZ_KLEIN_NEURON_ROPE=1;ropemmg=FZ_KLEIN_NEURON_ROPE=1,FZ_KLEIN_NEURON_MMG=4" --profile main,ropemmg --json out.json

The transformer loads once (bf16, the I96 W1 path). Per variant (its env switches set in this process before its steps;
the I93 switches are read per call) and size: one warm step (loads, and if needed compiles, the variant's NEFFs), then the
4 teacher-forced steps of prompt 0 (the reference x_k, as i96_klein_chip.py's G3): s/step (median and each), kernel
launches per step, peak HBM in use during the variant (the allocator's peak, reset per variant), G3 per step vs the float32
reference. The first variant is the base: a variant listed in --bitwise must give the base's velocities bit for bit (the
I93 RopeSpec / MatmulBF16G are exact by construction; on the chip this checks it). --profile <variants>: one more step of
those with the runtime's per-kernel execute times (FZ_NEURON_PROFILE accounting, no device bench), summed by kernel name.
FZ_NKI_FAST_EXP is read at import, so a variant set with it runs in its own process (the job's second call).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i96_klein_chip import Launches, bars, device_stats, rel_l2, log

SWITCHES = ("FZ_KLEIN_NEURON_ROPE", "FZ_KLEIN_NEURON_MMG")

def parse_variants(s: str) -> list[tuple[str, dict]]:
  out = []
  for part in [p for p in s.split(";") if p.strip()]:
    name, _, kv = part.partition("=")
    env = dict(x.split("=", 1) for x in kv.split(",") if x.strip())
    out.append((name.strip(), env))
  return out

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--refs", default=".scratch/i96/refs")
  ap.add_argument("--sizes", default="512,1024")
  ap.add_argument("--prompt", type=int, default=0)
  ap.add_argument("--device", default="NEURON")
  ap.add_argument("--variants", default="main=")
  ap.add_argument("--bitwise", default="", help="variants whose velocities must equal the first variant's bit for bit")
  ap.add_argument("--profile", default="", help="variants to profile per kernel (one extra step each)")
  ap.add_argument("--steps", type=int, default=4, help="teacher-forced steps timed per variant and size (<= 4)")
  ap.add_argument("--deadline", type=float, default=0)
  ap.add_argument("--json", default=None)
  a = ap.parse_args()
  from fusezap.tensor import Tensor
  from fusezap.dtype import dtypes
  from fusezap.device import Device
  from fusezap.serve.models.flux2 import Flux2Transformer, cut
  B = bars("bfloat16")
  variants = parse_variants(a.variants)
  bitwise = {v for v in a.bitwise.split(",") if v}
  prof = {v for v in a.profile.split(",") if v}
  rep: dict = {"device": a.device, "fast_exp": os.environ.get("FZ_NKI_FAST_EXP", "0"), "bars": B, "runs": [], "bitwise_ok": True}
  def save() -> None:
    if a.json:
      with open(a.json, "w") as f: json.dump(rep, f, indent=1)
  def late() -> bool: return bool(a.deadline) and time.time() > a.deadline
  for k in SWITCHES: os.environ.pop(k, None)
  t0 = time.perf_counter()
  tr = Flux2Transformer.from_pretrained(a.model, dtypes.bfloat16, a.device)
  rep["load_s"] = round(time.perf_counter() - t0, 1)
  log(f"transformer bf16 on {a.device}: loaded in {rep['load_s']} s (FZ_NKI_FAST_EXP={rep['fast_exp']}), {device_stats(a.device)}")
  L = Launches(a.device)
  al = Device[a.device].allocator
  base_v: dict = {}
  for size in [int(s) for s in a.sizes.split(",")]:
    r = np.load(os.path.join(a.refs, "denoise", f"{size}_p{a.prompt}_s0.npz"))
    emb = np.load(os.path.join(a.refs, "embeds", f"p{a.prompt}.npz"))["embeds"][None].astype(np.float32)
    ts = r["timesteps"].astype(np.float32)
    g = size // 16
    cos, sin = tr.rope(emb.shape[1], (g, g))
    e = Tensor(emb, device=a.device).realize()
    def step(k: int) -> tuple[np.ndarray, float, int]:
      x = Tensor(np.ascontiguousarray(r["x"][k][None], np.float32), device=a.device).realize()
      t = Tensor(np.full((1,), ts[k] / np.float32(1000), np.float32), device=a.device)
      n0, s0 = L.n, time.perf_counter()
      v = cut(tr.forward(x, e, t, cos, sin).float()).numpy()[0]
      return v, time.perf_counter() - s0, L.n - n0
    for name, env in variants:
      if late(): log(f"deadline: skipping {name} at {size}^2"); rep.setdefault("skipped", []).append(f"{name}_{size}"); continue
      for k in SWITCHES: os.environ.pop(k, None)
      os.environ.update(env)
      run = {"variant": name, "env": env, "size": size, "step_s": [], "launches": [], "G3": []}
      try:
        if hasattr(al, "peak"): al.peak = al.held
        _, w, n = step(0)   # warm: the variant's NEFFs load (or compile)
        run["warm_s"], run["warm_launches"] = round(w, 2), n
        vs = []
        for k in range(min(a.steps, len(ts))):
          v, s, n = step(k)
          vs.append(v); run["step_s"].append(round(s, 3)); run["launches"].append(n); run["G3"].append(rel_l2(v, r["v"][k]))
        run["s_per_step"] = float(np.median(run["step_s"]))
        run["device"] = device_stats(a.device)
        if size not in base_v: base_v[size] = (name, vs)
        elif name in bitwise:
          bn, bv = base_v[size]
          same = [bool(np.array_equal(x.view(np.uint32), y.view(np.uint32))) for x, y in zip(vs, bv)]
          run["bitwise_vs"] = {"base": bn, "equal_steps": same, "max_abs": [float(np.abs(x - y).max()) for x, y in zip(vs, bv)]}
          rep["bitwise_ok"] &= all(same)
        if name in prof and hasattr(al, "profile"):
          from fusezap.device.neuron import neuron_profile
          al.flush(); al.profile, al.prof, al.prof_wall = True, {}, [0, 0.0]
          _, s, _ = step(1)
          p = neuron_profile(al, bench=False)
          al.profile = False
          agg: dict = {}
          for kn, d in p["kernels"].items():
            base = kn.split("(")[0]
            ent = agg.setdefault(base, [0, 0.0]); ent[0] += d["launches"]; ent[1] += d["exec_ms_total"]
          top = sorted(agg.items(), key=lambda kv: -kv[1][1])
          run["profile"] = {"step_s": round(s, 3), "exec_ms_total": p["exec_ms_total"], "worker_batch_ms": p["worker_batch_ms"],
                            "by_kernel": [{"kernel": k, "launches": n, "exec_ms": round(ms, 2)} for k, (n, ms) in top]}
          log(f"  profile {name} {size}^2: step {s:.2f} s, runtime exec {p['exec_ms_total'] / 1e3:.2f} s; top: " +
              ", ".join(f"{k} {n}x {ms / 1e3:.2f} s" for k, (n, ms) in top[:8]))
        bw = run.get("bitwise_vs")
        log(f"{name} {size}^2: {run['s_per_step']:.2f} s/step (steps {run['step_s']}, warm {run['warm_s']} s), "
            f"{run['launches'][0]} launches, peak {run['device'].get('peak_gib')} GiB, G3 worst {max(run['G3']):.3e} (bar {B['G3']:.3g})"
            + (f", bitwise vs {bw['base']}: {bw['equal_steps']}" if bw else ""))
      except Exception as ex:   # a variant whose kernel does not compile or run: recorded, the others go on
        import traceback
        run["error"] = f"{type(ex).__name__}: {str(ex)[:600]}"
        log(f"{name} {size}^2: FAILED {run['error']}\n{traceback.format_exc()[-1500:]}")
      rep["runs"].append(run)
      save()
  for k in SWITCHES: os.environ.pop(k, None)
  ok = [r for r in rep["runs"] if "error" not in r and r["G3"]]
  rep["pass_g3"] = bool(ok) and all(max(r["G3"]) <= B["G3"] for r in ok)
  log(f"I93 klein speed on {a.device}: {len(ok)}/{len(rep['runs'])} runs ok, G3 {'PASS' if rep['pass_g3'] else 'FAIL'}, "
      f"bitwise {'OK' if rep['bitwise_ok'] else 'MISMATCH'}")
  save()

if __name__ == "__main__": main()
