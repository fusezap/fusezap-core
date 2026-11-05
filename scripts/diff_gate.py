"""I81 D4-D6: FuseZap's FLUX.2 [klein] components against the diffusers float32 CPU reference (scripts/diff_ref.py), with
real weights, one component in memory at a time.

  python scripts/diff_gate.py transformer [--device METAL] [--sizes 512] [--prompts 0-7] [--seeds 0,1]
      G3 teacher-forced: each step's velocity on the reference x_t and reference embeds, rel-L2 vs the reference
      G4 free-running: our own 4-step trajectory from the same noise and embeds, final latent rel-L2
  python scripts/diff_gate.py vae [...]
      G5 VAE decode of the reference final latent, max abs on [-1, 1] pixels
      G6 end to end: VAE decode of OUR final latent (from the G4 run) -> uint8, vs the reference uint8 image:
         fraction of pixel values within +-1 level, and PSNR
  python scripts/diff_gate.py encoder [...]
      G2 FuseZap's Qwen3TextEncoder (I81 D3) on --device: our prompt embeddings vs the reference ones, rel-L2 per prompt;
         writes them for --embeds own
  python scripts/diff_gate.py report        every gate's worst case vs its threshold (docs/validate/diff_gates.json)

--embeds ref (default): the reference prompt embeddings (transformers' Qwen3 in float32) are injected, so the gates
measure the transformer, the scheduler loop and the VAE. --embeds own: the encoder stage's embeddings, i.e. the whole
FuseZap pipeline from the prompt text (G3 is then teacher-forced on the reference x_t with our embeddings). Results go to
results/diff/fz_<device>_<dtype>[_te]/ (_te: --embeds own; gates.json, our latents, PNGs next to the reference PNGs).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diff_ref import OUT, GATES, REPO, STEPS, cases, parse_list, rel_l2, max_abs, psnr, pix_within, to_uint8 as ref_to_uint8

def fz_dir(a, *sub: str) -> str:
  d = os.path.join(OUT, f"fz_{a.device.lower()}_{'f32' if a.dtype == 'float32' else 'bf16'}{'_te' if a.embeds == 'own' else ''}", *sub)
  os.makedirs(d, exist_ok=True)
  return d

def save_results(a, key: str, res: dict) -> None:
  fn = os.path.join(fz_dir(a), "gates.json")
  g = json.load(open(fn)) if os.path.exists(fn) else {}
  g.setdefault(key, {}).update(res)
  json.dump(g, open(fn, "w"), indent=1)

def load_case_embeds(a, p: int) -> np.ndarray:
  src = fz_dir(a, "embeds") if a.embeds == "own" else os.path.join(OUT, a.ref, "embeds")
  return np.load(os.path.join(src, f"p{p}.npz"))["embeds"][None]

def gate_encoder(a) -> None:
  import gc
  from fusezap.dtype import dtypes
  from fusezap.serve.models.text_encoder import Qwen3TextEncoder
  from fusezap.serve.models.flux2_vae import trim_cache
  from diff_ref import PROMPTS
  t0 = time.perf_counter()
  enc = Qwen3TextEncoder.from_pretrained(REPO, getattr(dtypes, a.dtype), a.device)
  print(f"loaded text encoder on {a.device} in {time.perf_counter() - t0:.1f}s", flush=True)
  d = fz_dir(a, "embeds")
  for p in sorted({p for _, p, _ in cases(a)}):
    t1 = time.perf_counter()
    e = enc([PROMPTS[p]])[0]
    t2 = time.perf_counter()
    ref = np.load(os.path.join(OUT, a.ref, "embeds", f"p{p}.npz"))["embeds"]
    np.savez(os.path.join(d, f"p{p}.npz"), embeds=e)
    D = e.shape[-1] // 3
    res = {"G2_rel_l2": rel_l2(e, ref), "G2_rel_l2_per_layer": [rel_l2(e[:, i * D:(i + 1) * D], ref[:, i * D:(i + 1) * D]) for i in range(3)],
           "encode_s": t2 - t1}
    save_results(a, f"p{p}", res)
    print(f"p{p}: G2 {res['G2_rel_l2']:.2e} (layers {' '.join(f'{x:.2e}' for x in res['G2_rel_l2_per_layer'])}, {t2 - t1:.1f}s)", flush=True)
  enc.release(); del enc; gc.collect(); trim_cache(a.device)

def gate_transformer(a) -> None:
  from fusezap.dtype import dtypes
  from fusezap.tensor import Tensor
  from fusezap.serve.models.flux2 import Flux2Transformer
  from fusezap.serve.models.flux2_klein import Flux2KleinPipeline, schedule, pack
  t0 = time.perf_counter()
  tr = Flux2Transformer.from_pretrained(REPO, getattr(dtypes, a.dtype), a.device)
  print(f"loaded transformer on {a.device} in {time.perf_counter() - t0:.1f}s", flush=True)
  pipe = Flux2KleinPipeline(tr, None)
  d = fz_dir(a, "denoise")
  for size, p, s in cases(a):
    name = f"{size}_p{p}_s{s}"
    r = np.load(os.path.join(OUT, a.ref, "denoise", name + ".npz"))
    emb = load_case_embeds(a, p)
    h = w = size // 16
    t1 = time.perf_counter()
    sig, ts, mu = schedule(h * w, STEPS)
    assert np.array_equal(sig, r["sigmas"]) and np.array_equal(ts, r["timesteps"]), "schedule differs from diffusers"
    cos, sin = tr.rope(emb.shape[1], (h, w))
    e = Tensor(emb, device=tr.device).realize()
    g3 = []
    for k in range(STEPS):   # teacher forcing: the reference x_k in, our velocity out
      t = Tensor(np.full((1,), ts[k] / np.float32(1000), np.float32), device=tr.device)
      v = tr.forward(Tensor(r["x"][k][None], device=tr.device), e, t, cos, sin).float().numpy()[0]
      g3.append(rel_l2(v, r["v"][k]))
    t2 = time.perf_counter()
    x, info = pipe.denoise(emb, pack(r["noise"]), h, w, STEPS, keep=True)
    t3 = time.perf_counter()
    g4 = rel_l2(x[0], r["x"][-1])
    np.savez(os.path.join(d, name + ".npz"), x=np.stack([t[0] for t in info["trajectory"]]), v=np.stack([v[0] for v in info["velocities"]]))
    res = {"G3_rel_l2": g3, "G4_rel_l2": g4, "G4_traj_rel_l2": [rel_l2(info["trajectory"][k][0], r["x"][k]) for k in range(STEPS + 1)],
           "teacher_s": t2 - t1, "denoise_s": t3 - t2}
    save_results(a, name, res)
    print(f"{name}: G3 {' '.join(f'{x:.2e}' for x in g3)}  G4 {g4:.2e}  (teacher {t2 - t1:.1f}s, denoise {t3 - t2:.1f}s)", flush=True)

def gate_vae(a) -> None:
  from fusezap.dtype import dtypes
  from fusezap.tensor import Tensor
  from fusezap.serve.models.flux2_vae import Flux2VAEDecoder, to_uint8
  from fusezap.serve.models.flux2_klein import write_png
  t0 = time.perf_counter()
  vae = Flux2VAEDecoder.from_pretrained(REPO, getattr(dtypes, a.dtype), a.device)
  print(f"loaded VAE on {a.device} in {time.perf_counter() - t0:.1f}s", flush=True)
  d = fz_dir(a, "decode")
  for size, p, s in cases(a):
    name = f"{size}_p{p}_s{s}"
    r = np.load(os.path.join(OUT, a.ref, "denoise", name + ".npz"))
    ref_img = np.load(os.path.join(OUT, a.ref, "decode", name + ".npz"))["image"]
    h = w = size // 16
    t1 = time.perf_counter()
    img = vae.decode_packed(Tensor(r["x"][-1][None], device=vae.device), h, w).float().numpy()[0]
    t2 = time.perf_counter()
    res = {"G5_max_abs": max_abs(img, ref_img), "G5_rel_l2": rel_l2(img, ref_img), "decode_s": t2 - t1}
    own = os.path.join(fz_dir(a, "denoise"), name + ".npz")
    if os.path.exists(own):
      e2e = vae.decode_packed(Tensor(np.load(own)["x"][-1][None], device=vae.device), h, w).float().numpy()[0]
      u8, ref_u8 = to_uint8(e2e), ref_to_uint8(ref_img)
      assert np.array_equal(ref_u8, to_uint8(ref_img)), "uint8 conversion differs from the reference's"
      res |= {"G6_within1": pix_within(u8, ref_u8), "G6_psnr": psnr(u8, ref_u8), "G6_max_level": int(np.abs(u8.astype(int) - ref_u8).max())}
      write_png(os.path.join(d, name + ".png"), u8)
      np.save(os.path.join(d, name + "_image.npy"), e2e)
    save_results(a, name, res)
    print(f"{name}: G5 {res['G5_max_abs']:.2e}" + (f"  G6 within1 {res['G6_within1']:.5f} PSNR {res['G6_psnr']:.1f} dB" if "G6_psnr" in res else "")
          + f"  ({t2 - t1:.1f}s)", flush=True)

def report(a) -> None:
  thr = json.load(open(GATES))["thresholds_f32"]
  g = json.load(open(os.path.join(fz_dir(a), "gates.json")))
  vals = lambda k: [x for r in g.values() if k in r for x in (r[k] if isinstance(r[k], list) else [r[k]])]
  rows = [("G2", "G2_rel_l2", max, "<=", thr["G2_rel_l2"]), ("G3", "G3_rel_l2", max, "<=", thr["G3_rel_l2"]), ("G4", "G4_rel_l2", max, "<=", thr["G4_rel_l2"]),
          ("G5", "G5_max_abs", max, "<=", thr["G5_max_abs"]), ("G6 within +-1", "G6_within1", min, ">=", thr["G6_within1"]),
          ("G6 PSNR dB", "G6_psnr", min, ">=", thr["G6_psnr_db"])]
  out = {}
  for name, k, agg, op, t in rows:
    v = vals(k)
    if not v: continue
    worst = agg(v)
    ok = worst <= t if op == "<=" else worst >= t
    out[name] = {"worst": worst, "threshold": t, "pass": bool(ok), "n": len(v)}
    print(f"{name:14s} worst {worst:.7g} {op} {t:.7g}  {'PASS' if ok else 'FAIL'}  ({len(v)} values)")
  save_results(a, "_report", out)
  gj = json.load(open(GATES))   # the committed record: worst case per gate for this backend and size set
  key = f"fz_{a.device.lower()}_{'f32' if a.dtype == 'float32' else 'bf16'}{'_te' if a.embeds == 'own' else ''}_{a.sizes.replace(',', '+')}"
  gj.setdefault("measured", {})[key] = {"date": time.strftime("%Y-%m-%d"), "embeds": a.embeds,
                                       "cases": len([k for k in g if not k.startswith("_") and "_p" in k]), **out}
  json.dump(gj, open(GATES, "w"), indent=1)

def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("stage", choices=["encoder", "transformer", "vae", "report"])
  ap.add_argument("--device", default="METAL")
  ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
  ap.add_argument("--ref", default="cpu_f32")
  ap.add_argument("--sizes", default="512")
  ap.add_argument("--prompts", default="0-7")
  ap.add_argument("--seeds", default="0,1")
  ap.add_argument("--embeds", default="ref", choices=["ref", "own"], help="reference prompt embeddings, or the encoder stage's")
  a = ap.parse_args(argv)
  {"encoder": gate_encoder, "transformer": gate_transformer, "vae": gate_vae, "report": report}[a.stage](a)
  return 0

if __name__ == "__main__": sys.exit(main())
