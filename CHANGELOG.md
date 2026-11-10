# Changelog

All notable changes to FuseZap Core and FuseZap Invarix are recorded here. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) and versions follow [Semantic Versioning](https://semver.org/)
in its 0.y.z form (see docs/releases/README.md for what each part of the version promises before 1.0).

Nothing has been released yet. `scripts/release.py cut X.Y.Z` moves the Unreleased section below into a dated
`## [X.Y.Z] - YYYY-MM-DD` section. Each pull request adds its "Release note:" line here.

## [Unreleased]

### Added

- **AWS Inferentia2 and Trainium1 backend (NEURON).** FuseZap Core compiles its kernels to NKI and runs them on
  NeuronCores through a persistent runtime path. `fusezap serve` serves Qwen2.5-0.5B (fp32, one core) and
  Qwen2.5-7B (bf16, one core or tensor parallel over two cores) with paged KV cache and continuous batching, fused
  per-layer kernels, on-device all-reduce, prefill kernels sized by measured cost, and prewarm of every compiled
  forward before the first request. On inf2.8xlarge, Qwen2.5-7B bf16 TP=2 reaches TPOT 38.8 ms and 99.3 tok/s at 4
  users with Invarix PROMOTE (AWS NxD Inference on the same instance type: 40.6 ms, 81.7 tok/s).
- **Tensor parallelism in Core** (`fusezap serve --tensor-parallel-size N` / `--tp N`): one process per device,
  NCCL or a custom all-reduce over NVLink inside CUDA graphs, a fused all-reduce + residual + RMSNorm, bounded
  shutdown, and weight loading that streams each rank's shards with bounded host memory. Llama-3.3-70B bf16 on two
  H200s runs level with vLLM 0.32 TP=2 (TPOT 20.1 / 20.6 / 22.0 ms at 1 / 8 / 32 users vs 19.5 / 20.7 / 21.2).
- **NVIDIA CUDA backend**: NVRTC-compiled kernels, tensor-core GEMMs (including sm_90 WGMMA and a swap-AB decode GEMM
  with a per-shape tuning table), flash attention for prefill and paged decode, CUDA-graph replay of decode steps,
  bf16 / int8 / fp8 (W8A8) serving. Qwen2.5-7B bf16 on one H100 serves at TPOT 5.6 ms (vLLM 0.32: 5.9 ms) with
  top-1 agreement 0.996 against a float32 reference.
- **Apple Metal backend and the OpenAI-compatible server**: `fusezap serve` with streaming chat and completions, the
  official `openai` client end to end, paged KV cache, continuous batching, int8 / int4 / fp8 weights and AWQ / GPTQ
  import, and Prometheus `/metrics` with vLLM's metric names.
- **Image generation (preview): FLUX.2 [klein] 4B on Apple Metal and NVIDIA CUDA.** `fusezap generate` makes an image from a prompt
  and `fusezap serve-image flux2-klein` serves it through an OpenAI-compatible `/v1/images/generations` endpoint (the
  official `openai` client works end to end). The Qwen3 text encoder and the VAE run in float32 by default and a
  1024x1024 image fits a 48 GB Mac (peak 25.5 GB). At 1024x1024 in bf16, FuseZap's images are closer to the float32
  reference than diffusers' bf16 MPS images (mean PSNR 33.9 dB vs 25.5 dB over 8 prompts), but about 12x slower
  (53 s vs 4.4 s per image); speed is the next piece of work.
  On NVIDIA (H100, H200, RTX 4090; CUDA, same 8 prompts) FuseZap's images are again closer to the reference on every
  prompt (mean PSNR 32.7-34.0 dB vs diffusers bf16 24.5-24.7 dB), and slower: 3.15 s vs 0.48 s per image on an H100
  (6.5x), 3.07 s vs 1.36 s on an RTX 4090, where the VAE runs in bf16 (the default on CUDA GPUs
  under 40 GB). Details: docs/status/i95-flux2-klein-cuda/.
- **Structured outputs and tools**: grammar-constrained decoding (`response_format` JSON schema / JSON object, strict
  named and required tool calls; optional `llguidance`), and tool calls for Hermes, Mistral and Llama 3.x
  (`llama3_json`) chat formats.
- **FuseZap Invarix** (`fusezap invarix`): qualifies a serving release against a float32 reference and a baseline
  release, with accuracy (teacher-forced top-1, greedy exact match), functional, structured (101 tool-call and 50 JSON
  tasks), performance (short / mixed / long-context profiles), stability and reliability checks; stored baselines; a
  PROMOTE / BLOCK / REVIEW decision; HTML, JSON and JUnit reports. Report schema `fusezap.invarix/v0`.
- **Compiler core**: lazy tensors with a scheduler, symbolic shapes, kernel fusion, a BEAM search over kernel
  options (`FZ_BEAM`), jitted kernel replay (`fz.jit`), custom-kernel hooks, and a C backend for CPUs.
- `fusezap.__version__`, this changelog, the release policy and tooling (`docs/releases/`, `scripts/release.py`,
  `scripts/release_matrix.py`), and SECURITY.md.

### Changed

- `fusezap serve` answers a request that sets an OpenAI or vLLM field it does not implement (presence / frequency
  penalties, logit_bias, best_of, suffix, min_p and others) to a non-neutral value with a 400 `unsupported_parameter`
  error naming the field, instead of ignoring it; neutral values (0, null, {}) are still accepted.
  `parallel_tool_calls: false` now keeps at most one tool call, and `top_k: -1` means no top-k limit, as in vLLM.
- The Core version moves from 0.0.1 to 0.1.0.dev0 (a pre-release; nothing is published yet). The version now has
  one source, `fusezap/_version.py`, read by the package metadata.

### Fixed

- Generated kernels index buffers of more than 2^31 elements with 64-bit math (they overflowed int32: an illegal
  memory access on CUDA in the FLUX.2 VAE at 1024x1024). Kernels whose indexes fit are unchanged.
- The CUDA attention kernel is only chosen when its shared memory fits the GPU (the VAE's head dim 512 did not fit
  an RTX 4090); otherwise attention stays composed.

### Known issues

- Invarix's tool-call gate blocks Llama-3.3-70B because the model sends numeric arguments as strings; vLLM blocks on
  the same gate.
- Qwen2.5-72B on two H200s runs at about 70% of vLLM TP=2; the all-reduce fix that brought Llama-3.3-70B level with
  vLLM has not been measured on it yet.
- Qwen2.5-32B on one H100 has one measurement (bf16 at 78% / 59% of vLLM at 1 / 32 users) and no Invarix verdict
  yet: Preview in the support matrix.
