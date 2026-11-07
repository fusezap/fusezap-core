"""I96 W7/W8: FLUX.2 [klein] 4B's transformer on a NeuronCore, gated against diffusers' float32 CPU reference.

  python scripts/i96_klein_chip.py --model <klein dir> --refs .scratch/i96/refs --sizes 512,1024 --prompts 0,1 --json out.json

The transformer runs on --device (NEURON on inf2 / trn1; NKISHIM or CPU for a dry run of this script) in --dtype
(bfloat16: the W1 path, bf16 K-major weights through neuron_kv.matmul_bf16, float32 flash attention), eager, one step at a
time, exactly as Flux2KleinPipeline.denoise. The prompt embeddings and the noise are the reference's (scripts/diff_ref.py
output: refs/embeds/p<i>.npz, refs/denoise/<size>_p<i>_s0.npz), so no text encoder runs here.
  G3 teacher-forced: each step k fed the reference latent x_k; our velocity vs the reference v_k (rel-L2)
  G4 free-running: our own 4 steps from the reference noise; final latent vs the reference x_4 (rel-L2)
Bars (docs/validate/diff_gates.json): float32 thresholds_f32; bf16 relative, factor x diffusers-bf16's worst distance
(G3 1.5 x 0.101, G4 1.5 x 0.206). Also per step: seconds, kernel launches (first and warm), NEFF compiles / cache hits, HBM.
--vae-host decodes the free-run latent with FuseZap's VAE on the host CPU (float32) and reports G6 PSNR vs the reference
image (bar: diffusers-bf16 PSNR - 1 dB).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def log(*a) -> None: print(*a, flush=True)
def rel_l2(a, b) -> float: return float(np.linalg.norm((np.asarray(a, np.float64) - b).ravel()) / (np.linalg.norm(np.asarray(b, np.float64).ravel()) + 1e-30))

def bars(dtype: str) -> dict:
  g = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "validate", "diff_gates.json")))
  if dtype == "float32":
    t = g["thresholds_f32"]
    return {"G3": t["G3_rel_l2"], "G4": t["G4_rel_l2"], "G6_psnr_db": t["G6_psnr_db"]}
  r = g["thresholds_bf16_relative"]
  d = r["diffusers_bf16"]
  return {"G3": r["factor"] * d["G3_rel_l2"], "G4": r["factor"] * d["G4_rel_l2"], "G6_psnr_db": d["G6_psnr_db"] - r["G6_psnr_margin_db"]}

def device_stats(dev: str) -> dict:
  from fusezap.device import Device
  out = {}
  try:
    al = Device[dev].allocator
    out = {"held_gib": round(getattr(al, "held", 0) / 2**30, 3), "cached_gib": round(getattr(al, "cached", 0) / 2**30, 3)}
    if hasattr(al, "peak"): out["peak_gib"] = round(al.peak / 2**30, 3)
    if hasattr(al, "neffs"): out |= {"neffs_loaded": al.neffs, "neff_mib": round(al.neff_bytes / 2**20, 1)}
  except Exception as e: out["error"] = str(e)[:200]
  try:
    from fusezap.device.neuron import NEFF_STATS
    out |= {"neff_compiles": NEFF_STATS["compiles"], "neff_hits": NEFF_STATS["hits"], "neff_compile_s": round(NEFF_STATS["compile_s"], 1)}
  except Exception: pass
  return out

class Launches:
  """Counts device kernel launches (NEURONDevice.run / any device's run) between start() and stop()."""
  def __init__(self, dev: str):
    from fusezap.device import Device
    self.d = Device[dev]; self.n = 0; self.orig = self.d.run
    def run(si, *a, **k):
      self.n += 1
      return self.orig(si, *a, **k)
    self.d.run = run

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True, help="a klein diffusers dir (transformer/, vae/) or the HF repo id")
  ap.add_argument("--refs", default=".scratch/i96/refs")
  ap.add_argument("--sizes", default="512,1024")
  ap.add_argument("--prompts", default="0,1")
  ap.add_argument("--device", default="NEURON")
  ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
  ap.add_argument("--no-g4", action="store_true")
  ap.add_argument("--vae-host", action="store_true", help="decode the free-run latents on the host CPU (G6)")
  ap.add_argument("--deadline", type=float, default=0, help="unix time: skip what would start after it")
  ap.add_argument("--json", default=None)
  a = ap.parse_args()
  from fusezap.tensor import Tensor
  from fusezap.dtype import dtypes
  from fusezap.serve.models.flux2 import Flux2Transformer, cut
  B = bars(a.dtype)
  rep: dict = {"device": a.device, "dtype": a.dtype, "bars": B, "cases": [], "pass": True}
  def save() -> None:
    if a.json:
      with open(a.json, "w") as f: json.dump(rep, f, indent=1)
  def late() -> bool: return bool(a.deadline) and time.time() > a.deadline
  t0 = time.perf_counter()
  tr = Flux2Transformer.from_pretrained(a.model, getattr(dtypes, a.dtype), a.device)
  rep["load_s"] = round(time.perf_counter() - t0, 1)
  rep["after_load"] = device_stats(a.device)
  log(f"transformer {a.dtype} on {a.device}: loaded in {rep['load_s']} s, kmajor={getattr(tr, 'kmajor', False)}, {rep['after_load']}")
  L, vae = Launches(a.device), None
  for size in [int(s) for s in a.sizes.split(",")]:
    for p in [int(s) for s in a.prompts.split(",")]:
      if late(): log(f"deadline: skipping {size}^2 p{p}"); rep["skipped"] = rep.get("skipped", []) + [f"{size}_p{p}"]; continue
      r = np.load(os.path.join(a.refs, "denoise", f"{size}_p{p}_s0.npz"))
      emb = np.load(os.path.join(a.refs, "embeds", f"p{p}.npz"))["embeds"][None].astype(np.float32)
      sig, ts = r["sigmas"].astype(np.float32), r["timesteps"].astype(np.float32)
      g = size // 16
      cos, sin = tr.rope(emb.shape[1], (g, g))
      e = Tensor(emb, device=a.device).realize()
      case = {"size": size, "prompt": p, "G3": [], "step_s": [], "launches": []}
      for k in range(len(ts)):   # G3: teacher-forced on the reference x_k
        x = Tensor(np.ascontiguousarray(r["x"][k][None], np.float32), device=a.device).realize()
        t = Tensor(np.full((1,), ts[k] / np.float32(1000), np.float32), device=a.device)
        n0, s0 = L.n, time.perf_counter()
        v = cut(tr.forward(x, e, t, cos, sin).float()).numpy()[0]
        case["step_s"].append(round(time.perf_counter() - s0, 2)); case["launches"].append(L.n - n0)
        case["G3"].append(rel_l2(v, r["v"][k]))
        log(f"{size}^2 p{p} step {k}: G3 rel-L2 {case['G3'][-1]:.3e} (bar {B['G3']:.3g}), {case['step_s'][-1]} s, {case['launches'][-1]} launches, {device_stats(a.device)}")
        save()
      if not a.no_g4 and not late():   # G4: our own trajectory from the reference noise
        x = Tensor(np.ascontiguousarray(r["x"][0][None], np.float32), device=a.device).realize()
        s0 = time.perf_counter()
        for k in range(len(ts)):
          t = Tensor(np.full((1,), ts[k] / np.float32(1000), np.float32), device=a.device)
          v = cut(tr.forward(x, e, t, cos, sin).float())
          x = cut(x + v * float(sig[k + 1] - sig[k]))
        xf = x.numpy()[0]
        case["G4"], case["denoise_s"] = rel_l2(xf, r["x"][-1]), round(time.perf_counter() - s0, 1)
        log(f"{size}^2 p{p}: G4 final latent rel-L2 {case['G4']:.3e} (bar {B['G4']:.3g}), 4 steps {case['denoise_s']} s")
        np.save(os.path.join(os.path.dirname(a.json) if a.json else ".", f"latent_{size}_p{p}.npy"), xf)
        if a.vae_host and not late():
          from fusezap.serve.models.flux2_vae import Flux2VAEDecoder, to_uint8
          from fusezap.serve.models.flux2_klein import write_png
          if vae is None: vae = Flux2VAEDecoder.from_pretrained(a.model, dtypes.float32, "CPU")
          s0 = time.perf_counter()
          img = vae.decode_packed(Tensor(xf[None], device="CPU"), g, g).numpy()[0]
          u8 = to_uint8(img)
          ref = to_uint8(np.load(os.path.join(a.refs, "decode", f"{size}_p{p}_s0.npz"))["image"]) if \
                os.path.exists(os.path.join(a.refs, "decode", f"{size}_p{p}_s0.npz")) else None
          case["vae_host_s"] = round(time.perf_counter() - s0, 1)
          if ref is not None:
            mse = float(np.mean((u8.astype(np.float64) - ref) ** 2))
            case["G6_psnr_db"] = 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
          write_png(os.path.join(os.path.dirname(a.json) if a.json else ".", f"klein_{a.device.lower()}_{size}_p{p}.png"), u8)
          log(f"{size}^2 p{p}: VAE on the host CPU {case['vae_host_s']} s, G6 PSNR {case.get('G6_psnr_db', float('nan')):.2f} dB (bar {B['G6_psnr_db']:.1f})")
      case["pass"] = max(case["G3"]) <= B["G3"] and case.get("G4", 0) <= B["G4"] and case.get("G6_psnr_db", 1e9) >= B["G6_psnr_db"]
      rep["pass"] &= case["pass"]
      rep["cases"].append(case)
      save()
  rep["end"] = device_stats(a.device)
  log(f"I96 klein on {a.device} {a.dtype}: {'PASS' if rep['pass'] and rep['cases'] else 'FAIL'} "
      f"({len(rep['cases'])} cases; worst G3 {max([max(c['G3']) for c in rep['cases']] or [float('nan')]):.3e})")
  save()

if __name__ == "__main__": main()
