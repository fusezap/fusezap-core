# FLUX.2 [klein] 4B on Apple Metal, 1024x1024 end to end (4 Nov 2026)

Machine: Apple M5 Max, 48 GB. Model: black-forest-labs/FLUX.2-klein-4B, 4 steps, 1024x1024, bf16 transformer,
float32 Qwen3 text encoder and VAE (the defaults). 8 prompts, seed 0. Reference: the same pipeline in float32 on
CPU (scripts/diff_ref.py). Script: scripts/diff_klein_e2e_metal.py. Raw numbers: report.json in this folder.

| | FuseZap bf16 (Metal) | diffusers bf16 (MPS) |
|---|---|---|
| Mean PSNR against the float32 reference | 33.9 dB (29.4 to 39.3) | 25.5 dB |
| Mean SSIM | 0.986 | 0.929 |
| Seconds per image (mean) | 53.1 s | 4.36 s |
| Peak memory | 25.5 GB | 25.2 GB |

FuseZap's images are closer to the float32 reference on all 8 prompts (gate G6 passes on every prompt) and it is
about 12x slower (53.1 / 4.36 = 12.2). Speed work is under way.

Served: `fusezap serve-image flux2-klein --device METAL --dtype bfloat16` answers `/v1/images/generations` from the
official `openai` client; the served PNG is bit-identical to the end-to-end image for prompt 0.
