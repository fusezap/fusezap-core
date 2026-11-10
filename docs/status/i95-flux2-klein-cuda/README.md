# I95: FLUX.2 [klein] 4B on NVIDIA CUDA, 1024x1024 end to end (5 Nov 2026 UTC)

Machines: RunPod, one GPU each (H100 80GB HBM3, H200, RTX 4090), driver 595.91.07, torch 2.14.1+cu130, diffusers 0.42.1.
Model: black-forest-labs/FLUX.2-klein-4B at the reference's revision, 4 steps, 1024x1024, 8 prompts, seed 0. Reference:
the same pipeline in float32 on CPU (scripts/diff_ref.py, made on the Mac, shipped to the pods). Script:
scripts/diff_klein_cuda_job.sh -> scripts/diff_klein_e2e.py, commit 83b2cf3. Raw numbers: `<gpu>_report.json` and
`<gpu>_serve_smoke.json` in this folder (the tables below are computed from them).

### H100: NVIDIA H100 80GB HBM3

FuseZap: text encoder float32 (resident), transformer bf16, VAE float32.

| | FuseZap bf16 (CUDA) | diffusers bf16 (torch CUDA) |
|---|---|---|
| Mean PSNR against the float32 reference | 32.7 dB (29.3 to 35.6) | 24.7 dB |
| Mean SSIM | 0.984 | 0.921 |
| Seconds per image (mean, warm) | 3.15 s | 0.48 s |
| Encode / denoise (4 steps) / decode | 0.33 / 1.04 / 1.77 s | - |
| Peak device memory (allocator) | 35.6 GB | 19.6 GB (torch reserved) |
| Peak device memory (nvidia-smi) | 36.1 GB | 20.3 GB |
| Gate G6 (FuseZap >= diffusers - 1 dB) | 8/8 prompts; closer than diffusers on 8/8 | |
| FuseZap from the reference embeds (transformer + VAE only) | 33.2 dB, 2.83 s | |

Served: `fusezap serve-image flux2-klein --device CUDA --dtype bfloat16`, two 1024x1024 requests from the `openai` client: 3.80 s and 3.64 s, PSNR 35.56 dB against the reference for prompt 0 (the end-to-end image's: 35.56 dB).

### H200: NVIDIA H200 (141 GB)

FuseZap: text encoder float32 (resident), transformer bf16, VAE float32.

| | FuseZap bf16 (CUDA) | diffusers bf16 (torch CUDA) |
|---|---|---|
| Mean PSNR against the float32 reference | 32.7 dB (29.3 to 35.6) | 24.6 dB |
| Mean SSIM | 0.984 | 0.921 |
| Seconds per image (mean, warm) | 3.63 s | 0.45 s |
| Encode / denoise (4 steps) / decode | 0.47 / 1.34 / 1.81 s | - |
| Peak device memory (allocator) | 35.6 GB | 19.6 GB (torch reserved) |
| Peak device memory (nvidia-smi) | 36.1 GB | 20.3 GB |
| Gate G6 (FuseZap >= diffusers - 1 dB) | 8/8 prompts; closer than diffusers on 8/8 | |
| FuseZap from the reference embeds (transformer + VAE only) | 33.2 dB, 3.17 s | |

Served: `fusezap serve-image flux2-klein --device CUDA --dtype bfloat16`, two 1024x1024 requests from the `openai` client: 4.18 s and 4.05 s, PSNR 35.56 dB against the reference for prompt 0 (the end-to-end image's: 35.56 dB).

### RTX 4090: NVIDIA GeForce RTX 4090 (24 GB)

FuseZap: text encoder float32 (staged: loaded alone, encodes, released), transformer bf16, VAE bfloat16.

| | FuseZap bf16 (CUDA) | diffusers bf16 (torch CUDA) |
|---|---|---|
| Mean PSNR against the float32 reference | 34.0 dB (29.8 to 38.7) | 24.5 dB |
| Mean SSIM | 0.986 | 0.914 |
| Seconds per image (mean, warm) | 3.07 s | 1.36 s |
| Encode / denoise (4 steps) / decode | 0.35 / 1.80 / 0.91 s | - |
| Peak device memory (allocator) | 17.1 GB | 19.6 GB (torch reserved) |
| Peak device memory (nvidia-smi) | 17.5 GB | 20.0 GB |
| Gate G6 (FuseZap >= diffusers - 1 dB) | 8/8 prompts; closer than diffusers on 8/8 | |

Served: the serve-image smoke ran out of device memory (cuMemAlloc of 1 GB) with the bf16 text encoder, the transformer and the float32 VAE resident together; serve-image now picks the bf16 VAE under 40 GB (not re-run yet).

## Notes

- FuseZap's images are closer to the float32 reference than diffusers bf16 on every prompt on all three GPUs (gate G6
  8/8 each), by about 8 dB of mean PSNR.
- FuseZap is slower: 6.5x diffusers on the H100 (3.15 vs 0.48 s/image), 8.0x on the H200,
  2.3x on the RTX 4090. About half of FuseZap's time on the H100 is the float32 VAE decode (the convs are
  im2col + 3xTF32 GEMM on CUDA: there is no CUDA conv kernel yet). Speed work is I94.
- The H200 is slower than the H100 for FuseZap (3.63 vs 3.15 s/image; denoise 1.34 vs
  1.04 s) with the same kernels and the same images; not investigated
  (one pod each, so it may be the host rather than the chip). diffusers is about the same on both.
- FuseZap's images on the H100 and the H200 are identical, pixel for pixel, for every prompt (diffusers' differ by up
  to 205 levels between the two).
- RTX 4090: the float32 VAE ran out of memory at 1024^2 (a 4.8 GB im2col beside the transformer), so its numbers are
  with FZ_KLEIN_VAE_DTYPE=bfloat16; that is now the default on CUDA GPUs under 40 GB (KleinImagePipeline logs the choice).
  The text encoder was staged (float32, loaded alone, released before the transformer loads).
- First run, two bugs, both fixed in 83b2cf3:
  - H100/H200: CUDA_ERROR_ILLEGAL_ADDRESS at 1024^2: the VAE's 2.4e9-element im2col was indexed in int32; index math
    now switches to int64 where a value can leave int32 (codegen/lower.py).
  - RTX 4090: the float32 attention kernel for the VAE's head dim 512 needed 139 KB of shared memory (the card allows
    99 KB); the scheduler now checks the device limit and keeps that attention composed.
