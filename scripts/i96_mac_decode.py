#!/usr/bin/env python3
# I96: G6 for the NEURON runs. Decodes the latents a chip job saved (results/nki_klein/latent_<size>_p<i>.npy) with FuseZap's
# float32 VAE on Apple METAL and scores PSNR vs the diffusers float32 reference image in .scratch/i96/refs/decode.
#   python scripts/i96_mac_decode.py <fetched job results dir> [out dir, default docs/diff/i96_w7] [chip, default inf2]
#   -> <out>/{g6_mac_decode.json, klein_<chip>_*.png}, for every latent_<size>_p<i>.npy the job saved
import glob, json, os, re, sys, time, numpy as np
from fusezap.tensor import Tensor
from fusezap.dtype import dtypes
from fusezap.serve.models.flux2_vae import Flux2VAEDecoder, to_uint8
from fusezap.serve.models.flux2_klein import write_png
R, REFS = sys.argv[1], ".scratch/i96/refs"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/diff/i96_w7"
CHIP = sys.argv[3] if len(sys.argv) > 3 else "inf2"
os.makedirs(OUT, exist_ok=True)
vae = Flux2VAEDecoder.from_pretrained("black-forest-labs/FLUX.2-klein-4B", dtypes.float32, "METAL")
res = {"decoder": "FuseZap VAE, float32, Apple METAL (M5 Max), on latents produced by the transformer on " + CHIP, "bar_G6_psnr_db": 17.0, "cases": []}
cases = sorted((int(m[1]), int(m[2])) for f in glob.glob(f"{R}/nki_klein/latent_*_p*.npy") if (m := re.search(r"latent_(\d+)_p(\d+)\.npy$", f)))
for size, p in cases:
  xf = np.load(f"{R}/nki_klein/latent_{size}_p{p}.npy"); g = size // 16
  s0 = time.perf_counter(); u8 = to_uint8(vae.decode_packed(Tensor(xf[None], device="METAL"), g, g).numpy()[0])
  ref = to_uint8(np.load(f"{REFS}/decode/{size}_p{p}_s0.npz")["image"])
  mse = float(np.mean((u8.astype(np.float64) - ref) ** 2)); psnr = 99.0 if mse == 0 else float(10 * np.log10(255.0**2 / mse))
  write_png(f"{OUT}/klein_{CHIP}_{size}_p{p}.png", u8)
  res["cases"].append({"size": size, "prompt": p, "G6_psnr_db": round(psnr, 2), "pass": psnr >= 17.0, "decode_s": round(time.perf_counter() - s0, 1)})
  print(res["cases"][-1], flush=True)
json.dump(res, open(f"{OUT}/g6_mac_decode.json", "w"), indent=1)
