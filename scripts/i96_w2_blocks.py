"""I96 W2: klein's first double block and first single block at real shapes and real weights, on NKISHIM vs CPU float32.

  PYTHONPATH=. python scripts/i96_w2_blocks.py --size 512 --dtype both [--ref results/diff/cpu_f32] [--json out.json]

NKISHIM is the numpy model of nki 0.6: every NEURON kernel the step would launch (the generic renderer's kernels, the
TensorE matmul / MatmulBF16 / flash attention) runs instruction by instruction on the host, so this checks the NKI
lowering of the real block shapes (S = T + L = 512 + 1024 at 512^2, 512 + 4096 at 1024^2) with nothing but the Mac.
Inputs: diffusers' float32 CPU reference latent x_0 and prompt embeddings (scripts/diff_ref.py output) of prompt 0.
The model is the real transformer config cut to 1 double + 1 single block (block 0 of each), so the weights are the
checkpoint's. Stages compared (rel-L2 against CPU float32): the modulations, the embedders, double block 0 (image and
text streams), single block 0 on [text, image] after it.
Bars: float32 rel-L2 <= 1e-5 (the device reorders float32 sums); bf16: NKISHIM-bf16 vs CPU-f32 rel-L2 <= 1.5 x CPU-bf16
vs CPU-f32 (I82 §4's relative bar; CPU bf16 here is FuseZap's own CPU path, the reference for "reduced" precision).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from fusezap.tensor import Tensor
from fusezap.dtype import dtypes

def build(path: str, dt, dev: str):
  from fusezap.serve.models.flux2 import Flux2Config, Flux2Transformer
  from fusezap.serve.weights import load_safetensors
  with open(os.path.join(path, "config.json")) as f: cfg = Flux2Config.from_hf(json.load(f))
  cfg.num_layers, cfg.num_single_layers = 1, 1
  w = load_safetensors(path)
  want = set(Flux2Transformer.weight_names(type("C", (), {"cfg": cfg})()))
  return Flux2Transformer(cfg, {k: v for k, v in w.items() if k in want}, dt, dev)

def stages(tr, x: np.ndarray, emb: np.ndarray, sigma: float, h: int, w: int) -> dict[str, np.ndarray]:
  from fusezap.serve.models.flux2 import cut
  dev, out = tr.device, {}
  T = emb.shape[1]
  cos, sin = tr.rope(T, (h, w))
  with Tensor.inference():
    xt, et = Tensor(x, device=dev).cast(tr.dtype), Tensor(emb, device=dev).cast(tr.dtype)
    st = tr.temb(Tensor(np.array([sigma], np.float32), device=dev)).silu()
    mi = [cut(m) for m in tr._mod(tr._lin(st, "double_stream_modulation_img.linear.weight"), 2)]
    mt = [cut(m) for m in tr._mod(tr._lin(st, "double_stream_modulation_txt.linear.weight"), 2)]
    ms = [cut(m) for m in tr._mod(tr._lin(st, "single_stream_modulation.linear.weight"), 1)]
    out["modulation_img"] = np.concatenate([m.float().numpy().ravel() for m in mi])
    hh, ee = cut(tr._lin(xt, "x_embedder.weight")), cut(tr._lin(et, "context_embedder.weight"))
    out["x_embedder"], out["context_embedder"] = hh.float().numpy(), ee.float().numpy()
    t0 = time.perf_counter()
    hh, ee = (cut(t) for t in tr._double(0, hh, ee, mi, mt, cos, sin))
    out["double0_img"], out["double0_txt"] = hh.float().numpy(), ee.float().numpy()
    out["_double_s"] = np.array(time.perf_counter() - t0)
    t0 = time.perf_counter()
    s = cut(tr._single(0, ee.cat(hh, axis=1), ms, cos, sin))
    out["single0"] = s.float().numpy()
    out["_single_s"] = np.array(time.perf_counter() - t0)
  return out

def rel(a, b) -> float: return float(np.linalg.norm((a - b).ravel()) / (np.linalg.norm(b.ravel()) + 1e-30))

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--size", type=int, default=512)
  ap.add_argument("--dtype", choices=("float32", "bfloat16", "both"), default="both")
  ap.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
  ap.add_argument("--ref", default=os.environ.get("FZ_DIFF_REF", "results/diff/cpu_f32"))
  ap.add_argument("--json", default=None)
  a = ap.parse_args()
  from fusezap.serve.weights import resolve_model
  path = resolve_model(a.model)
  if os.path.isdir(os.path.join(path, "transformer")): path = os.path.join(path, "transformer")
  den = np.load(os.path.join(a.ref, "denoise", f"{a.size}_p0_s0.npz"))
  emb = np.load(os.path.join(a.ref, "embeds", "p0.npz"))["embeds"][None].astype(np.float32)
  x, sigma = den["x"][0][None].astype(np.float32), float(den["timesteps"][0]) / 1000.0
  g = a.size // 16
  ref = stages(build(path, dtypes.float32, "CPU"), x, emb, sigma, g, g)
  dts = ("float32", "bfloat16") if a.dtype == "both" else (a.dtype,)
  report: dict = {"size": a.size, "tokens": emb.shape[1] + g * g, "cpu_f32_s": {k: float(ref[k]) for k in ("_double_s", "_single_s")}}
  ok = True
  for d in dts:
    dt = getattr(dtypes, d)
    cpu = stages(build(path, dt, "CPU"), x, emb, sigma, g, g) if d != "float32" else None
    t0 = time.perf_counter()
    got = stages(build(path, dt, "NKISHIM"), x, emb, sigma, g, g)
    r = {"nkishim_s": round(time.perf_counter() - t0, 1), "double_s": float(got["_double_s"]), "single_s": float(got["_single_s"]), "stages": {}}
    for k in ref:
      if k.startswith("_"): continue
      e = rel(got[k], ref[k])
      row = {"nkishim_vs_cpu_f32": e}
      if cpu is not None:
        c = rel(cpu[k], ref[k]); row["cpu_bf16_vs_cpu_f32"] = c
        row["pass"] = e <= 1.5 * c + 1e-7
      else: row["pass"] = e <= 1e-5
      ok &= row["pass"]
      r["stages"][k] = row
      print(f"{a.size}^2 {d:9s} {k:18s} " + " ".join(f"{kk}={vv:.3e}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in row.items()), flush=True)
    print(f"{a.size}^2 {d}: NKISHIM double block {r['double_s']:.1f} s, single block {r['single_s']:.1f} s", flush=True)
    report[d] = r
  report["pass"] = ok
  print("PASS" if ok else "FAIL")
  if a.json:
    with open(a.json, "w") as f: json.dump(report, f, indent=1)

if __name__ == "__main__": main()
