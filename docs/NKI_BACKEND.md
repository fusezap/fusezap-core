# NKI backend design: AWS Inferentia2 / Trainium1 (device `NEURON`)

This is the backend's working log. N0-N9 are its milestones (section 8); I68-I78 label the work items described
below. The hardware job scripts it names (`nki_*_inf2.sh`, `nki_*_trn1.sh`) and the raw run logs are kept in the
private development repository; the results they produced are quoted here.

Status (2026-10-21): N0 to N3 done (2026-10-26: N4 and N5 on hardware, N6 on the Mac; see the N4, N5 and N6 paragraphs below). The renderer (`fusezap/codegen/nki.py`), the NKISHIM numpy shim, the NEURON device (`fusezap/device/neuron.py`), the NKISIM tier (the real nki 0.6.0 simulator in a `.venv-nki` worker, plus a NEFF compile path) and the TensorE matmul and flash-attention custom kernels (`fusezap/codegen/nki_kernels.py`, N3) are in, and NKI-final closed the N3 review (a bool mask with flipped rows now stays composed instead of failing in the renderer). Everything runs in the nki 0.6.0 simulator and compiles for inf2; nothing has run on hardware (N4). The **[unverified]** tags left below are hardware-only questions (numerics on a NeuronCore, the nrtpy allocator, launch overhead) that no simulator can answer; the run's logs (not published) lists what N4 must check first. Target: inf2.xlarge (NeuronCore-v2), with trn1.2xlarge running the same code.

**N4 run 1 (2026-10-26, inf2.xlarge, SDK 2.33 DLAMI, ~9 min).**
- Verified on hardware: the first NEURON kernels (E_128, E_262144 adds) ran on the NeuronCore and were correct. nki 0.6 returns the outputs (out_mode "returned"). The bare worker round trip is 356 µs. NEFF load into nrtpy (2.35.30) takes ~9 ms for a small NEFF; the first load in a process (6.1 s) includes runtime init. Host↔device copies through nrtpy: write 1 MiB 101 µs, 16 MiB 1.23 ms (13.7 GB/s); read 1 MiB 161 µs, 16 MiB 11.2 ms (1.5 GB/s). The N3 gate's first 6 cases (0.5B matmul, causal/pos attention, the llama.py graph) ran on the chip within tolerance before the stop.
- The nki.jit path costs **~1.31 s per call at any size** (first call 14.3 s). Cause, from nki 0.6.0's `framework/compiled.py`: the standalone (numpy) path compiles with `enable_cache=False` and loads the NEFF on every call; the 1.3 s is the compile time.
- Harness bugs, all fixed (N4b): (1) the worker set `NKI_ARTIFACTS_DIR` for one NEFF and never unset it, so every later compile hit NCC_IDRV077 "artifacts already exist". Now each compile gets a fresh unique dir, and the env var is set only for one call. (2) The nrtpy probe left out the must-alias input `data0.must_alias_input`, so execution failed with NRT_INVALID(2). Every declared input is now bound. (3) `Device.DEFAULT` raised StopIteration on a host with no METAL/CUDA/CPU (the DLAMI has gcc, not clang), which broke every test fixture. It now falls back to NPY, and setup installs clang.
- **N4b, the persistent path (Mac-tested with fakes, [unverified on hardware]):** `FZ_NEURON_RT=nrt` (the default). Each kernel is compiled once per (source, argument sizes, compiler versions) in `.venv-nki`, compile only (an nki `StandaloneKernel` whose executor only records; no runtime starts there). The NEFF is disk-cached under `~/.cache/fusezap/neff/<key>`, written once by an atomic rename. It is loaded once into a long-lived `.venv-nrt` worker (`fusezap/device/nrt_worker.py`) and launched by buffer id on device-resident tensors (`NeuronAllocator`). Each launch is one pipe round trip with no data and no load; copies happen only at copy items. `FZ_NEURON_RT=jit` keeps the nki.jit path as the fallback. The rerun (`scripts/nki_n4_inf2.sh`) measures loaded-NEFF launch latency first.

**N4 run 2 (2026-10-26, inf2.xlarge, 22 min).** The persistent path works on the chip: a loaded NEFF launches in 163-172 µs (fusezap -> worker -> execute; execute alone 85-94 µs, nrtpy device time 8-19 µs), the worker round trip is 29 µs, copies 4 KiB ~115 µs, 1 MiB 0.8 ms in / 1.8 ms out. Custom tests 256/256; op suite 1383 passed / 3 failed (exp2, tanh float32) / 48 skipped / 9 xfailed. Numerics: reciprocal bit-exact, VectorE add/mul/compare and the int32 paths exact, ScalarE exp ~1e-5 relative, sqrt 3.5e-6, and **ScalarE log wrong far from 1** (log2 of inputs in [1e-30, 1e30], in practice all above 1e27, was off by up to 2e19 relative with 0 % bitwise agreement; not denormals, zeros or negatives, and the renderer's ln * log2(e) is exact in the simulator: the activation table has a limited domain and, like ScalarE sin, no range reduction). The N3 gate crashed at its first case (it called a NEFF program on numpy arrays).

**N5 (2026-10-26, Mac only; hardware run `nki_n5_inf2.sh` pending).**
- **Exact transcendentals.** EXP2, exp (`Tensor.exp` = EXP2(x * log2 e), recognized and computed from x by Cody-Waite), LOG2 and the attention kernel's exp(s - m) run on VectorE (`nki.exact_exp` / `exact_log2`): range reduction by exact float/int steps (rint by the float -> int32 copy, exponent and significand by int32 bit masks), a degree-9 polynomial for 2^f / e^r and an atanh-series for log2 of the significand in [sqrt(2)/2, sqrt(2)), scaling by two exact powers of two built from exponent bits, then the special values (0, -0, ±inf, NaN, negatives, subnormals) by predicated copies. SQRT keeps ScalarE plus one Newton step with the exact `reciprocal`. NKISHIM: within 1 ulp of the correctly rounded value over the whole float32 range (log2 2 ulp), and only instructions run 2 found exact on hardware. `FZ_NKI_FAST_EXP=1` restores the ScalarE instructions (the hardware run measures both, with the error by input binade).
- **N3 gate** runs on device Buffers through `NEURONDevice.run_program` on every tier (full-size weights on NEURON).
- **Serving** (`fusezap serve <model> --device NEURON`, `fusezap/serve/neuron_kv.py`). LLMEngine keeps its scheduler and paged block tables; the device side needs no indirect DMA: the pool is (slots, 2 * Hkv * hd) per layer; the KV write is one NKI kernel per layer (`fz_kv_write`: the step's one-hot (tokens x slots) matrix times the rotated [k | v | 1] rows on TensorE, exact, then a predicated copy into the pool, which the kernel reads and writes as its must-alias argument); attention is `render_attention` with mask "keyrange" (per slot a key id = block-table row * stride + position from the step metadata, per token [lo, hi]; rows = (token, q head of the group) with two strides; no visible key gives zeros) over every slot; RMSNorm with the residual add is one kernel (`fz_rmsnorm_add`, the composed kernels' instructions in their order, bit-identical); RoPE is x * [cos, cos] + x.flip(half) * [-sin, sin] (Llama._rope's products, no `cat`) in one kernel writing [q | k | v | 1]; weights are relaid K-major at load; greedy tokens are a device argmax. Every kernel has static shapes, so a forward (each decode bucket 1, 2, 4, ..., max_num_seqs, each prefill size) compiles to a fixed NEFF set. `Llama.prefill/decode/generate` go through the same ragged forward on NEURON (`NeuronSteps`: positions as data instead of a symbolic start, which would compile per position). The scheduler now materializes reads that would need several views on single-view devices (`NEURONDevice.single_view`), so any graph lowers. Launches are queued in `NeuronAllocator` and sent as one `batch` request at the next copy (`FZ_NEURON_BATCH=0`: one request each).
- **Launches per decode token (Qwen2.5-0.5B, 24 layers):** the first working version took 17 per layer on the tiny test checkpoint (RoPE through `cat` split into 4 kernels plus copies, the composed RMSNorm's 2 kernels re-reading the residual chain, a one-hot matmul + WHERE for the KV write), about 417 per token at 24 layers; measured on Qwen: 344 after the RoPE fix, **245** now = 10 per layer (norm, qkv, rope, KV write, attention, o_proj, norm, gate_up, silu*mul, down) + 5 per step; prefill 248. At run 2's ~170 µs per launch that is ~42 ms of launch overhead per token, less with batching (~85 µs of it was nrtpy's own execute call).
- **Mac results (NKISHIM):** Qwen2.5-0.5B greedy tokens equal the float32 reference (2 prompts batched), teacher-forced top-1 64/64 with |logprob error| 5.7e-6 mean; LLMEngine and Llama.generate on a tiny checkpoint equal CPU's tokens, also under preemption; the same engine on the persistent path with the fake nrtpy.

**N5 run 1 (2026-10-26, inf2.xlarge, 32 min).** Qwen2.5-0.5B served on the chip token-exact: greedy 24/24 prompts and 1441/1441 tokens equal the float32 reference, teacher-forced top-1 1.0, |logprob error| mean 4.2e-6. TPOT 161 ms at c1: 245 launches per token, ~0.66 ms each in the decode loop (a warm single launch was 0.17 ms in N4). The exact VectorE exp/exp2/log2 matched the simulator bit for bit; ops3 142/142, N3 gate 8/8, op suite 1246 passed / 0 failed. Two failures: `sqrt_wide` on the exact path (69 % bitwise equal, normwise error 4.7: ScalarE sqrt, like its log, is wrong far from 1, and one Newton step from there cannot recover), and the HTTP check (`fusezap serve` had no uvicorn: setup synced no extras).

**N6 (2026-10-26, Mac; hardware run `nki_n6_inf2.sh` pending).** Launches and device work per token, each change behind an env switch so one hardware run shows its gain (`scripts/nki_n6_serve.py --quick` per variant):
- **Fusion (`FZ_NEURON_FUSE=k`, default 1; exception 9).** fz.jit hands every capture to `NEURONDevice.fuse_schedule`, which turns each run of items between two fences into one kind="custom" item: a `FusedSpec` kernel whose body is the members' bodies in order (each `data<i>` renamed to the fused argument, every other name suffixed per member, by `ast`), written buffers first, every written buffer returned (must-alias outputs). `Llama.forward_tokens` records a fence before each decoder layer on NEURON (after realizing the residual term, so a layer is always [norm, qkv, RoPE, KV write, attention, o_proj, norm, gate_up, silu*mul, down]) and one before the head. So layers 1..23 are one fused kernel, compiled once per (bucket, pool prefix) and launched per layer on that layer's buffers; layer 0 (no residual add, plus the step's one-hot slot matrix) and the head (final norm, lm_head, device argmax) are one each: **25 launches per decode token instead of 245** (k = 4: 7). The members run the same instructions in the same order on the same buffers, so on NKISHIM the fused replay equals the unfused one bit for bit (prefill and decode, tests/test_neuron_serve.py). A fused kernel that does not compile or load falls back to its members, logged (`FusedCall.members`). **[unverified on hardware]**: that neuronx-cc orders a member's DMA reads of an HBM argument after an earlier member's DMA writes to it (the kernels phase compares fused and unfused logits on the chip; the run falls back to `FZ_NEURON_FUSE=0` for the HTTP phase if the default is not token-exact).
- **Pool-prefix buckets (`FZ_NEURON_CTX`, default on).** N5's KV write and attention touched all P pool slots every step (4096 in run 1). A step now runs at the smallest prefix (powers of two from `FZ_NEURON_CTX_MIN` = 512, then P) that holds every block in its block tables (`NeuronKVCache.meta_ctx`; the allocator hands out low block ids first): one jitted forward per prefix (`_Forward.jit_for`). c1 with a short prompt: 512 slots, 8x less attention and KV-write work than run 1.
- **Operands by rows (`FZ_NEURON_KT`, `FZ_NEURON_AT_ROWS`).** Attention's kᵀ (head dim on the partitions) was a strided DMA of single 4-byte elements: 12.6 M per decode token at 4096 slots (N4's 128x1024 Aᵀ load ran ~0.7 ns per element). It is now 128 key rows per DMA (one contiguous run per partition) and an `nc_transpose` on TensorE; a matmul's Aᵀ the same from 16 rows up (prefill). Exact in the simulator; the kernels phase prints the share of equal bits on the chip.
- **Range-reduced sqrt.** x = m * 2^(2k) with m in [1, 4) from the exponent and significand bits (subnormals scaled by 2^24), ScalarE sqrt(m), one Newton step, times 2^k built from exponent bits; 0, -0, inf, NaN and negatives by predicated copies. <= 1 ulp over the float32 range on NKISHIM, also with a ScalarE sqrt modelled as wrong outside [1, 4).
- **Profiling (`FZ_NEURON_PROFILE`, `neuron.neuron_profile`).** Per kernel name: launches, the runtime worker's execute time per launch, nrtpy's device-time benchmark of the last binding; the worker's batch wall time. Per token: step = host (fusezap: replay, requests, pipe) + worker (around the executes) + runtime (nrt execute beyond device time) + device.
- **HTTP.** Setup syncs `--extra serve --extra invarix`; the job runs `fusezap serve` + `fusezap invarix` (accuracy, performance short c1 and c4) and vLLM-Neuron from the DLAMI's `aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0` venv (TP=2 bf16, then TP=1 float32) on the same workload (`nki_n6_compare.py`).

**N6 run 1 (2026-10-27, inf2.xlarge, 42 min).** Token-exact on the chip (24/24 prompts, 1441/1441 tokens, top-1 1.0). TPOT at c1: 165 ms (N5 kernels) -> 72 (pool prefixes) -> 61 (operands by rows) -> **26.2 (FUSE=1, 25 launches)** -> 22.3 ms (FUSE=4, 7 launches); c4 133 tok/s; per token at FUSE=1: device 13.4 + runtime 5.9 + host 6.3 ms. Fused vs unfused logits bitwise equal on the chip (risk 9 closed for inf2). HTTP (`fusezap serve` + invarix): accuracy top-1 1.0, c1 TPOT 27.1 ms; the HTTP c4 number (70.6 ms) included an in-run compile. No baseline: the DLAMI's vLLM 0.29 + vllm-neuron 0.29 refuses inf2 in worker init (`get_efa_bdf_mapping`: "Unsupported instance family: inf2. Supported: trn2, trn3pd, trn3pds"), at TP=2 and TP=1.

**N6b (2026-10-27, Mac; hardware jobs `nki_n6b_inf2_baseline.sh` and `nki_n6b_trn1.sh` pending).** A baseline from AWS's own inf2/trn1 library, and Trainium1.
- **Which AWS library (Neuron docs, read 2026-10-27).** SDK 2.33 (19 Oct 2026) ships vLLM Neuron 0.29.0.1.2.0 on TorchNeuron (native PyTorch 2.13), documented for Trn2/Trn3 only [13]. For Inf2/Trn1, "Torch/XLA support for inference ... is in maintenance mode" since 2.31, on PyTorch 2.9 [14]. NxD Inference is in maintenance; 0.10.18399 (SDK 2.31) is the last release, "no new releases are planned" [15]; its end of support in 2.34 is for Trn2/Trn3 [16]. transformers-neuronx ended in 2.26 [17]; optimum-neuron is not in the Neuron docs or pip repo. So AWS's supported serving path on inf2/trn1 today is **NxD Inference 0.10 through its vLLM integration, vllm-neuron 0.5.3** (vLLM 0.16.0, "Supported AWS instances: Inf2, Trn1/Trn1n, Trn2", models incl. Qwen 2.5 [18]), an OpenAI server, so `fusezap invarix` measures it exactly as it measures FuseZap. Pins = AWS's own vLLM 0.16.0 DLC [19]: neuronx-cc 2.26.6360.0, torch-neuronx 2.9.0.2.15.32035 (SDK 2.33's 2.9 build), neuronx-distributed 0.19.28492, neuronx-distributed-inference 0.10.18399, nki 0.5.0, vllm-neuron `0.5.3` from GitHub (vllm 0.16.0 and torch 2.9.1 from PyPI), in `~/venv-nxdi`; the AMI's driver and runtime 2.35.30 stay (`N6B_NXDI_CC` swaps the compiler pin). Flags from its README: local snapshot path (known issue: Qwen's tied embeddings break shard-on-load from a hub id), `--no-enable-prefix-caching --num-gpu-blocks-override 4` (= max_num_seqs; the plugin sets block_size = max_model_len), on-device sampling (the default; no logprobs, so accuracy by greedy text; `NEURON_ON_DEVICE_SAMPLING_DISABLED=1` for a logprobs run); fallback the prefix-caching flags (`--block-size 32 --num-gpu-blocks-override 128`), then NxD Inference offline (`nki_n6b_nxdi.py`: `NeuronQwen2ForCausalLM` + `HuggingFaceGenerationAdapter`, per-token stamps by a streamer, batch-1 and batch-4 models).
- **Measured the same way.** TP=2 bf16 (AWS's default shape on inf2.xlarge) and TP=1 float32 (FuseZap's precision and core count), `fusezap invarix` performance short (128 in / 128 out, c1 and c4) after an unmeasured pass of the same workload (fixes the HTTP c4 pollution; also in `nki_n6_inf2.sh`; `nki_n6_serve.py` now prewarms its c1 and c4 shapes), accuracy vs the float32 reference. `nki_n6b_summary.py`: one table (TPOT/TTFT c1, tok/s c1 and c4, TPOT c4, accuracy, dtype, TP) and FuseZap's HTTP default as a share of each AWS row.
- **Trainium1.** Same DLAMI (its docs show trn1.2xlarge `neuron-ls`). The NEFF target now follows the instance type (`neuron.neuron_target()`: DMI product name trn1* -> `trn1`, else `inf2`; `FZ_NEURON_TARGET` wins; the target is in the NEFF key). The trn1 job runs numerics, numerics again with inf2 NEFFs (do they run unchanged?), the kernels tests, the full N6 serve checks with FUSE=1 and 4, HTTP invarix, the DLAMI vLLM 0.29 (expected to refuse trn1) and the same AWS baseline.
- **Not known until the run:** whether NxD Inference 0.10 compiles Qwen2.5-0.5B (14 query / 2 KV heads, hidden 896) at TP=2 on NeuronCore-v2 and in float32; how long its compile takes on 4 vCPUs; whether the 2.26 compiler's NEFFs and SDK 2.31-era user space run on runtime 2.35.30 (the DLC pairs them with runtime 2.33.10).

**N6b inf2 runs 1 and 2 (2026-10-27).** FuseZap, prewarmed: in process FUSE=1 25.9 ms / 132 tok/s at c4, FUSE=4 22.1 ms / 164 tok/s; over HTTP FUSE=4 23.9 ms / 141 tok/s, FUSE=1 28.2 ms / 115 tok/s. Run 1's HTTP accuracy phase timed out: its in-process run was `--quick` (no top-1), so invarix's first echo request compiled the scoring NEFFs cold (N6's top-1 took 437 s, mostly compile); fz_fuse1 now runs in full first. AWS: run 1 failed at import (`libneuronpjrt-path` not on PATH: the venv was not activated; fixed). Run 2 served (TP=2 bf16 ready in 520 s, TP=1 fp32 in 320 s) but answered **gibberish after the first token** (greedy exact 7.4 % / 11.9 %), so its 11.7 / 24.9 ms are void. Cause, from the log and vllm-project/vllm-neuron#57 (closed, maintenance-mode plugin): the default on-device sampling (`OnDeviceSamplingConfig(dynamic=True)`) uses a top-k whose indices are wrong in every batch row but the first for reduction sizes including **151936, Qwen2.5's vocabulary** (75968 per rank at TP=2) once batch > 1 (aws-neuron/aws-neuron-sdk#1400); max_num_seqs 4 makes the batch 4 even at c1. Not the tied embeddings: the log builds and loads `lm_head.weight`, and the first token (prefill) is right. AWS's recommended workaround on Inf2/Trn1 (#57): `--additional-config '{"override_neuron_config": {"on_device_sampling_config": null}}'` (sampling on the host, logprobs available); not `NEURON_ON_DEVICE_SAMPLING_DISABLED=1`, which fails at startup with `TypeError: 'module' object is not callable` (vllm-neuron#29: `from vllm.v1.sample import sampler as Sampler`), which is what killed run 2's logprobs run. The jobs now serve with that override, add on-device greedy argmax (`{"dynamic": false}`) as a second configuration if time is left, and put a **correctness gate** in front of every performance measurement (`scripts/nki_n6b_gate.py`: 3 reference prompts greedily, serial and concurrent, vs the f32 reference's first 16 tokens; fail = "OUTPUT INCORRECT", no speed measured); `nki_n6b_summary.py` marks such rows and never compares them (run 2's rows are marked from their accuracy reports).

**N7 step 1: host time per decode token (2026-10-27, Mac; hardware job `nki_n7_step1.sh` pending).** N6b trn1 at FUSE=4: step 21.38 ms = device 12.81 + runtime 1.88 + worker 0.26 + **host 6.43**. Measured on the Mac with the real `nrt_worker.py` subprocess over a no-op nrtpy (Qwen2.5-0.5B, the same 7 fused launches, so host time only): 4.27 ms, of which ~3.4 ms was `FusedSpec.name` (repr + sha256 of every member's source, on every launch; `NEURONDevice.run` reads it for the fence check) and the frozen dataclass hash re-walking the sources per program-cache lookup; both are now computed once per spec (host 0.83 ms). The rest: six runtime-worker round trips per step (4 input writes, the launch batch, the token read; 0.28 ms of pipe), each step input scheduled as a copy item, and `run()`'s per-launch work over ~60 arguments of a fused layer. `FZ_NEURON_HOSTFAST=1` (default; 0 = N6b's host path): writes up to `FZ_NEURON_QWRITE_MAX` (1 MiB) are queued with the launches and the read takes the queue along (**one round trip per step**), `LLMEngine` writes its inputs straight into device buffers (`neuron_kv.upload`), and fz.jit replays launch through `NeffReplay` (exec request built once, re-bound only where a buffer changed): host 0.85 -> **0.39 ms** (python 0.27, pipe 0.13). The greedy token was already a device argmax (R int32 back, not the 600 KB logits row); `FZ_NEURON_DEVICE_ARGMAX=0` reads the logits for comparison. `scripts/nki_n6_serve.py` now splits the host part (`host_split`: sched, prep, meta, call, post, round trips, pipe, python).

**N7 step 2: the head NEFF (2026-10-27, Mac; hardware job `nki_n7_step2.sh` pending).** Step 1 on trn1 (FUSE=4): device 12.80 ms per token, the head (final norm + lm_head 151936 x 896 float32 + device argmax) 3.45 ms per launch, 2.16 ms without the argmax. Where it went: lm_head through `render_matmul` is 2079 weight DMAs of 256 KiB (2 KiB runs at row stride V) feeding 2079 float32 `nc_matmul` (TensorE at 1/4 rate: ~1.5 ms of moving columns, the floor next to ~1.33 ms of HBM), and the composed argmax kept the R rows on R partitions, so its ~2100 VectorE instructions each walked 512 values of one row per lane (~1.3 ms, ~0.6 us each). `FZ_NEURON_HEAD=v2` (default; v1 = step 1): `HeadSpec` streams the lm_head relaid once at load (`head_tiles`: per block of `FZ_NEURON_HEAD_W`=1024 columns one contiguous run per partition holding all K chunks; `FZ_NEURON_HEAD_DMAS`=4 DMAs per block, 752 DMAs in all, one store per block) with the same `nc_matmul` sequence (same stationary Aᵀ chunks, 512-column moving tiles, PSUM accumulation over K in order), so the logits are bit-identical; `ArgmaxSpec` puts the vocabulary on the 128 partitions (1187 values per lane): per partition max and first index, two TensorE transposes, the same on one partition (14 VectorE instructions per row; ties go to the lowest index, across partition runs too). Cost model (`scripts/nki_head_cost.py`, engines overlapping perfectly): v1 2.0 ms, v2 1.57 ms at 1 and 4 rows (TensorE-bound); `scripts/nki_n7_head_bench.py` measures the fused head alone per variant on the device. **trn1 head_bench** (device ms per launch, 1 / 4 rows; every variant's logits bitwise == v1, tokens == v1): v1 3.451 / 3.537, v2 2.593 / 2.626, v1-logits 2.157 / 2.218, v2-logits 2.579 / 2.588: ArgmaxSpec ~0.01-0.04 ms instead of ~1.3, but HeadSpec's matmul is slower than render_matmul's (the cost model's perfect overlap was wrong for it). **Step 2b:** `FZ_NEURON_HEAD=v3` (now the default; v1 and v2 stay selectable) = render_matmul on `kmajor(lm_head)` (no head_tiles copy, no extra 545 MB) + ArgmaxSpec; LLMEngine hands ArgmaxSpec the logits buffer itself, so no copy of a view is added. HeadSpec by block width and DMAs per block (rows 1): 512:4 2.648, 1024:4 2.593, 2048:4 2.861, **1024:1 2.071**, 1024:8 3.848 ms: one DMA per block is best and beats render_matmul (2.16), so `FZ_NEURON_HEAD_DMAS` now defaults to 1 and a block's weight tile is capped at 128 KiB per partition (`HEAD_SBUF`). Hardware job `nki_n7_step2b.sh` measures v3 against v2 at W 512 / 1024 / 2048 / 4096 (one DMA each) and serves with whichever is fastest at rows 1 + 4 with bitwise-v1 logits.

**N7 serve: what `fusezap serve` adds at 4 users (2026-10-27, Mac; hardware job `nki_n7_serve.sh` pending).** Step 2b on trn1 (head v2:1024:1): in process c1 TPOT 14.62 ms, c4 16.86 ms (237 tok/s decoding); over HTTP (Invarix short, 128 in / 128 out, closed loop) c1 16.32 ms, c4 19.69 ms / 194 tok/s, TTFT p99 229 ms (AWS vLLM + vllm-neuron: c4 16.4 ms, 238 tok/s). `FZ_SERVE_PROFILE=<file.jsonl>` (`fusezap/serve/serveprof.py`, summary `scripts/serve_prof_summary.py --requests <invarix requests.csv>`) writes one line per engine step (wall, engine CPU, runtime-worker round trips, time between steps, kind, tokens, pool prefix and slots needed) and one per request (the handler's lag behind the engine, ms per output). On the Mac over the fake runtime (device time 0 or ~14.3 ms per step, Invarix short c1/c4): between steps ~0.05 ms, the handler ~0.1 ms per output and off the critical path, so the HTTP layer is not the gap. The gap is the device work the served steps ask for: (1) **pool prefix**: the block allocator was LIFO, so after Invarix's warm-up and c4 points a lone request held blocks up to ~1000 slots and c1 decoded at the 1024-slot prefix (in process: 512). `BlockAllocator(lowest_first=True)` on NEURON (`FZ_NEURON_LOW_BLOCKS=1`, default; 0 = LIFO) keeps the blocks in use packed at the bottom: c1 at 512 again. Served c4 needs up to 4 x 256 = 1024 slots (mean ~800), so it runs at 1024 whatever the allocator; the in-process c4 (4 x 24 tokens) ran at 512, so it is not the served c4's floor (the job measures that floor in process with `FZ_NEURON_CTX_MIN=1024`). (2) **prefill interleaving**: a 128-token prefill (~35 ms on trn1) runs between the others' decodes at every arrival of the closed loop; `FZ_PREFILL_FIRST` now defaults to 4 on NEURON (was 0), so requests that arrive together are prefilled before the decoders' next step instead of one prompt per decode + prefill step (a scheduler sim with P = 35 ms and D = 18.5 ms: TPOT 19.05 -> 18.91 ms, 202 -> 206 tok/s, TTFT p99 196 -> 140 ms; the same sim with D = 18.5 reproduces the measured c4 within ~4%). (3) `--engine-process auto` now runs the engine in its own process on NEURON too (on the Mac at device time 0: 0.18 ms less per c4 step, handler lag 0.36 -> 0.04 ms). Tokens are unchanged by all three (block ids and step order do not change a token; tests/test_neuron_serve.py).

**N7 step 3: the decoder-layer NEFFs (2026-10-27, Mac; hardware job `nki_n7_step3.sh` pending).** N7 serve on trn1 (FUSE=4, head v2:1024:1), per token at the 512-slot prefix, 1 row: step 14.79 ms = host 0.87 + worker 0.51 + runtime 2.0 + device 11.42 ms, of which the 24 layers 9.4 ms (1.56 ms per 4-layer NEFF, ~0.39 ms per layer against a ~0.15 ms weight-read floor: ~60 MB at ~410 GB/s) and the head 2.07 ms; at 1024 slots device 13.19 ms (+0.074 ms per layer per 512 slots); served c4 (Invarix short) ran every decode step at 1024 slots, 198.6 tok/s. **Where a layer's time goes** (`scripts/nki_layer_cost.py`: the rendered kernel run on NKISHIM with every instruction recorded and attributed to its member, engine-busy time at nki_head_cost's rates, a list schedule over the data dependencies with a 4-deep weight-tile rotation; calibrated x0.72 on v1's measured 0.39 ms; v1 = the ten members fused, 1 row, 512 slots, model us before calibration): norm 9, qkv matmul 38 (31 DMAs, 7 of them element-strided Aᵀ loads), bias + RoPE (renderer kernel, 20 DMAs of 128-byte runs through HBM) 31, KV write 21, attention 58 (224 VectorE instructions: per kv head and 512-key block the exact exp's 44 instructions on (7 rows, 512 keys) tiles, 7 of 128 lanes busy; 24 DMAs of 256-byte K/V rows), o_proj 31, norm 9, gate_up 187 (35 MB, 133 matmuls), silu*mul 7, down 146 (38 element-strided Aᵀ loads before its first matmul) = 538; engine busy: TensorE 178 us (float32 moving columns at 1/4 rate), DMA 151, VectorE 77; every member's output goes to HBM and the next one's input comes back (a DMA round trip each, 9 per layer). At 1024 slots attention 109 and the KV write 42. **`FZ_NEURON_LAYER=v2` (new default; v1 = the members): `nki_kernels.LayerSpec`, one kernel per decoder layer** (`NeuronKVCache.layer`, from `Llama.forward_tokens` on NEURON when the step fits: a kv head's Hq/Hkv * rows attention rows on the 128 partitions, so decode; prefill buckets keep the members) with the members' arithmetic: the same `nc_matmul` sequences (stationary Aᵀ chunks, 512-column K-major weight tiles, PSUM accumulation over K in order), NormSpec's norm, the renderer kernels' elementwise chains (`nki.exact_exp`, the cat's +0.0 padding adds), KVWriteSpec's one-hot matmuls and predicated copy, render_attention's per-head matmuls and online softmax. New are only exact moves: hᵀ and the MLP activation by TensorE transposes (finite values), the pool rows kept in SBUF after the KV write (attention reads K and V from them), silu(gate) * up computed on (128, ·) tiles with the features on the partitions (also the down projection's stationary Aᵀ, no strided loads), and attention's exact exp on the transposed key block (keys on the partitions: 128 x (keys/128 * rows) tiles instead of (rows, 512)), which is also P·V's stationary; the row sums come from the block transposed back (the same values in the same layout for `tensor_reduce`). X and the attention output still pass through HBM once (the qᵀ and o_proj Aᵀ loads). Bitwise on NKISHIM against the members for every output (residual, down projection, every pool row, X, attention output; 1 to 5 rows, prefixes 300 to 768 of 1024, D and I not multiples of 128, ragged key chunks, first layer, one kv head) and through the model (prefill and decode logits, engine tokens = CPU's). Variants for the hardware run (`LAYER_VARIANTS`): v2 (default: kv heads one at a time, the key mask shared; RoPE by per-head column slices: only plain slices and 2-D DMA patterns), v2s (`FZ_NEURON_LAYER_HEADS=stack`: both kv heads' rows in one tile at 32-partition offsets, the softmax instructions once; partition-shifting PSUM -> SBUF copies), v2a (`FZ_NEURON_LAYER_ROPE=ap`: RoPE by 3-D access patterns on SBUF tiles). Cost model per layer (calibrated ms; 1 row / 4 rows): 512 slots v1 0.390 / 0.501, v2 0.218 / 0.252; 768 v1 0.419 / 0.531, v2 0.238 / 0.273; 1024 v1 0.441 / 0.553, v2 0.250 / 0.287 (v2s and v2a within ~2% of v2); v2 is TensorE-bound (float32 moving columns: 184 us of the 302 modelled). That is ~4.1 ms per token at c1 and ~6 ms per c4 step if the hardware follows the model. `scripts/nki_n7_layer_bench.py` measures one layer alone per variant, rows 1 / 4 and 512 / 768 / 1024 slots on the device (device ms per launch; logits, tokens and the KV pool after the same steps bitwise vs v1; `--ablate`: each v1 member alone, the measured breakdown); `nki_n7_step3.sh` serves with the fastest variant whose bits equal v1's everywhere (`layer_choice`). **4 users:** with lowest-first blocks, served c4 (128-token prompts, 128-token outputs, 4 requests in step) needs <= 768 slots in 128 of its 254 decode steps per repeat (mean 798, max 1024; `scripts/serve_prof_summary.py` now prints how many steps of a kind fit 512 / 768 / 1024). `FZ_NEURON_CTX=mid` adds the 3/2 prefixes (768, 1536, ...), `FZ_NEURON_CTX=512,768,1024` sets them outright; off by default (the 768 prefix's last key block is 256 keys wide where 1024's is 512, so attention's row sums are not guaranteed to round the same on the device; tokens equal CPU's on NKISHIM). With v2 the per-slot cost is smaller (model: +0.032 ms per layer per 512 slots at 1 row, v1 +0.051; measured v1 +0.074), so 768 saves ~0.35 ms on the c4 steps that fit (~0.17 ms per c4 step on average). A placement that packs concurrent sequences tighter cannot do much more: the lowest-first allocator already keeps the slots in use at the bottom, and the c4 steps' prefix follows the sum of the four sequences' lengths. Per-row key ranges (scanning only each sequence's own blocks) need runtime-dependent loop bounds or gathers, which these kernels do not use (exception 2).

**N7 step 4: the 1-user gap to AWS NxDI (2026-10-28, Mac; hardware job `nki_n7_step4.sh` pending).** Step 3 on trn1 (v2s layers, head v2:1024:1, FUSE=4): c1 11.83 ms per token = device 9.55 (5 x fused4 1.245 + fused5 1.257 + head 2.065) + runtime 1.23 + host 0.75 + worker 0.30, 7 launches; AWS NxDI in process 9.6 ms. The device time alone is NxDI's whole token; the gap is the 2.3 ms around it, plus whatever the device can still lose. **The cost model, refitted** (`scripts/nki_layer_cost.py --fit`): step 3's model was calibrated on v1 alone (x0.72) and predicted v2 at 0.218 ms; uncalibrated it said 0.306 (measured 0.318). It now records each kernel's instructions once (record-only on NKISHIM), schedules them with tile-level dependencies, in order per engine and DMA queue, DMA transfers sharing one HBM link, a buffer-rotation depth limited by SBUF per tile, and fits 12 bounded parameters to 43 trn1 kernels (the v1 members alone, v1 / v2 / v2s / v2a at rows 1, 4 and 512 / 768 / 1024 slots, the 4-layer fused NEFF, 16 head variants of steps 2 / 2b): rms error 8%, every layer and member within 10%. Fitted: a float32 matmul's moving column 2.0 ns (step 3 assumed 1.4: that is the missing ~0.1 ms per layer, ~116 K moving columns per layer), ~40 ns per TensorE instruction, VectorE ~1 ns per element per lane + 60 ns, DMA ~0.5 us + 393 GB/s HBM to SBUF, 11.5 us per launch. Where v2s's 0.32 ms goes (model 0.338, critical path, 1 row, 512 slots): gate_up with silu*mul 0.153 ms, down 0.073, attention 0.033, o_proj 0.025, qkv 0.019, norm 0.008, RoPE 0.007, KV write 0.009; by engine TensorE 0.25 (GEMV in the hᵀ-stationary form: 512 moving columns per instruction), VectorE 0.06, DMA 0.01. **The head**: render_matmul (2079 DMAs of 2 KiB runs, 2.17 ms) and HeadSpec 1024:1 (149 DMAs of 28 KiB runs, 2.06 ms) issue the same 2079 matmuls and take the same time, so the head is TensorE-bound at ~1 us per 512-column float32 matmul, not DMA-bound; 545 MB at the fitted 393 GB/s is ~1.4 ms, hidden under it. Wider blocks (2048, 4096: 56 / 112 KiB per partition) cannot double-buffer in SBUF, so DMA and TensorE serialize (4096: 3.85 ms = both); four DMAs into one tile (1024:4) are ordered by the tile's dependency, and the model does not reproduce how much that costs on the chip (0.87x) — `scripts/nki_n7_probe.py` measures the DMA patterns directly (one DMA per tile, four by partition group into one tile, four by columns into four tiles, 2 KiB runs), with the float32 matmul in both operand forms, transposes, and bf16 for reference. **Changes, each behind its own switch** (the job's choice takes only variants bitwise equal to v1 in every bench case): `FZ_NEURON_LAYER_EARLY=1` (LayerSpec `early`, variants v2e / v2se: every input-only load and the masks of every key block first, qᵀ from X and o_proj's Aᵀ from the attention output by TensorE transposes instead of HBM round trips, attention rows g-major (q head, token) so a q head is one transpose, the first key block without the online-softmax rescale since lr and acc start at +0.0; bitwise by construction; the model: -6 us per layer); `FZ_NEURON_LAYER_MM=ws` (`mm`, v2sw / v2swe: gate_up and down weight-stationary, 128-column weight slices as the stationary and hᵀ / the activation as the moving N columns, so gᵀ / uᵀ come out of PSUM in silu*mul's layout; the same products and PSUM accumulation over the same K chunks; bitwise only if TensorE's float32 matmul is operand-symmetric) and `FZ_NEURON_LAYER_PF=48` (v2swep: with ws the MLP is DMA-bound by the model, so its first 48 gate weight tiles, 12.6 MB, load during attention); `FZ_NEURON_HEAD=v4` (HeadSpec ws on head_tiles' layout, logitsᵀ back to rows by transposes); `FZ_NEURON_HEAD_FUSE=1` (the head joins the last layers' fused kernel; with `FZ_NEURON_FUSE=24` a decode forward is one launch, `FZ_NEURON_FUSE_MAX=44` keeps the prefill NEFFs at 4.4 layers of members). The ws forms hinge on what loading a 128 x 128 float32 stationary costs: the fit puts it at ~0.07 ns per column (from the transposes, whose data is the stationary), which would make the MLP and the head DMA-bound (model: v2swe 0.233 ms per layer, v2swep 0.225, head v4 1.41 ms); if it costs what a moving column costs, both are slower (0.36 / 2.91). Predictions (model, not measured), per token at c1: FUSE=24 + head fuse ~-1.0 ms runtime and ~-0.4 host + worker (6 launches fewer), early ~-0.15 ms, ws if it is cheap and bitwise -2.3 ms (layers) -0.6 (head). The NKISHIM tests check early and ws bit for bit against the members with `FZ_NKISHIM_MM=f64` (an order-independent matmul: BLAS's float32 kernels round a moved row or a swapped operand differently, TensorE's PE array does not move rows); the head fuse and FUSE=24 bit for bit in the default shim.

**N8 base: step 4's choice is the default (2026-10-28).** trn1 step 4 chose, bitwise vs v1 everywhere and token-identical: layer v2se (0.314 ms per layer at 1 row / 512 slots), head v2:1024:1 (2.067 ms), `FZ_NEURON_FUSE=24 FZ_NEURON_FUSE_MAX=44 FZ_NEURON_HEAD_FUSE=1` (one launch per decode token; c1 10.89 ms in process, served 11.01 ms / c4 303.8 tok/s). These are now the code defaults: `FZ_NEURON_LAYER=v2`, `FZ_NEURON_LAYER_HEADS=stack` (a step too wide for stacked heads, > 9 rows on Qwen2.5-0.5B, uses one tile per kv head instead of the members), `ROPE=slice`, `EARLY=1`, `MM=mv`, `PF=0`; `FZ_NEURON_HEAD=v2` (`HEAD_W=1024`, `HEAD_DMAS=1`); `FZ_NEURON_FUSE=24`, `FZ_NEURON_FUSE_MAX=44` (0 = no cap), `FZ_NEURON_HEAD_FUSE=1`. The weight-stationary forms (`MM=ws`, `PF>0`, `HEAD=v4`) stay selectable and off: on trn1 they were NOT bitwise, and v2swep produced wrong tokens (0/24 prompts; device kernel max |diff| 142): a bug, not float drift. bf16 (`FZ_NEURON_DTYPE=bf16`) with these defaults: LayerBF16 of the early layer is bitwise vs the bf16 members (NKISHIM), the lm_head stays MatmulBF16 (HeadSpec is float32-only) inside the fused forward, `layer_fits` counts early's live key-block masks, and a bf16 layer with `MM=ws` runs as the members (LayerBF16 has no weight-stationary form and refuses it). The step 3 / step 4 / N7 serve / bf16 / 7B jobs pin the defaults they were written against. `scripts/nki_layer_cost.py` takes `<variant>+bf16` / `+bf16a` on step 4's fitted model (bf16 matmuls at the step 4 probe rates, 2-byte DMAs) with the trn1 bf16 layers in `MEASURED`: the model puts them at ~0.67x of trn1 (not refitted).

**N8 bf16 (I70): the bf16 gap to AWS NxD Inference (2026-10-28, Mac; hardware job `nki_n8_bf16_trn1.sh` pending).** I68 on trn1 (Qwen2.5-0.5B, 1 core, in process): NxDI 0.10 bf16 TP=1 offline 5.11 ms per token c1, 447.8 tok/s c4, greedy 10/24 prompts vs the float32 reference; FuseZap bf16 (FUSE=4, v2s, MatmulBF16 head) 9.49 ms, 395.3 tok/s, 22/24. FuseZap's profile: device 7.22 ms (5 x 4 layers 0.950 + 4 layers 0.947 + head 1.521: ~0.237 ms per layer, MatmulBF16 head + argmax 1.52 ms) + runtime 1.24 + host 0.74 + worker 0.30, 7 launches. **What NxDI does** (source read from the neuronx-distributed-inference 0.10.18399 wheel, the version the trn1 job ran with neuronx-distributed 0.19.28492; paths under `neuronx_distributed_inference/`; the job's own NeuronConfig is `nki_n6b_nxdi.py` build(): tp 1, batch 1 / 4, max_context_length 256, seq_len 512, torch_dtype bfloat16, enable_bucketing, padding right, OnDeviceSamplingConfig(top_k=1), everything else default):
- *Graph and launches.* Qwen2 is the generic decoder (`models/qwen2/modeling_qwen2.py:104-153`: input RMSNorm, NeuronAttentionBase, residual add, post-attention RMSNorm, `NeuronLlamaMLP`, residual add). Decode is one traced `token_generation_model` per sequence bucket (`models/model_base.py:3143`, called once per token at `:3723` from the HF adapter's loop, `utils/hf_adapter.py:184-199`): all 24 layers, the final norm (`model_base.py:1427`), lm_head + `.float()` (`:987-988`), on-device sampling (`:1150-1190`; top_k 1 is a plain argmax, `modules/generation/sampling.py:372-384, 459-464`) and the KV cache update (`model_base.py:1407-1416`, written back by aliasing) in one NEFF: one launch per token, as FuseZap's FUSE=24 + head fuse. `async_mode` is off (`models/config.py:183`).
- *Kernels: none.* Every NKI kernel switch defaults off (`models/config.py:152` fused_qkv, `:394` flash_decoding, `:420-453` attn_kernel / qkv_kernel / mlp_kernel / attn_tkg_nki / attn_tkg_builtin / attn_block_tkg, `:416` k_cache_transposed, `:395-414` kv_cache_tiling only with speculation), and Qwen2 sets none (`modeling_qwen2.py:57-80`); the MLP takes its `_native_mlp` branch (`models/llama/modeling_llama.py:707-737`). So the whole graph is XLA HLO compiled by neuronx-cc with Qwen2's flags (`modeling_qwen2.py:237-242`): `--auto-cast=none --enable-mixed-precision-accumulation --model-type transformer -O1`, `--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 --vectorize-strided-dma`; the log also shows a weight-layout pass ("Updating the hlo module with optimized layout": the compiler picks each weight's HBM layout). The TP=1 warning "Overriding attention sharding strategy to GQA.CONVERT_TO_MHA" is benign: with 2 kv heads and TP 1 the head counts are unchanged (`modules/attention/gqa.py:62-75, 78-100, 783`), so K/V stay 2 heads, broadcast by `repeat_kv` in attention.
- *Precision: everything bf16 between ops.* torch_dtype bf16 casts every parameter and the checkpoint (`models/model_wrapper.py:1469-1476`, `models/application_base.py:645-681`, cast_type "config" `config.py:118`): embedding, q/k/v/o, gate/up/down, lm_head, norm weights. The residual stream and every activation are bf16 (the residual adds `modeling_qwen2.py:144, 149` in bf16). RMSNorm computes in float32 and rounds its output to bf16 (`modules/custom_calls.py:21-34`, AwsNeuronRmsNorm). RoPE's cos/sin are computed in float32 and cast to bf16 (`modules/attention/utils.py:331-343`), the rotation in bf16 (`:240-249`). Attention decode (`modules/attention/attention_base.py:1383-1462`, `compute_for_token_gen`): Q·K_priorᵀ in bf16 (scores rounded to bf16 by the matmul's output dtype, the mask applied in bf16), cast to float32 (`:1428`, `:1437`), softmax in float32 (`manual_softmax`, `utils.py:252-270`), probabilities rounded to bf16 (`:1456`), P·V in bf16, prior + active outputs added in bf16. The KV cache is bf16 (`modules/kvcache/kv_cache_manager.py:141-161`; attention_dtype None, `config.py:126`), BHSD layout (`modules/kvcache/utils.py:519-549`), and the new token's K/V take part in attention from registers ("active") while the cache write happens after the layers. Logits are bf16 from lm_head, then `.float()` before the argmax: near ties are decided on bf16-rounded logits. Matmul accumulation is float32 (`--enable-mixed-precision-accumulation`).
- *Attention span: the bucket.* With bucketing on, token generation has buckets 128 / 256 / 512 for max_length 512 (`modules/autobucketing.py:8-19, 226-279`), each its own NEFF (the log: "Generating 3 hlos for key: token_generation_model"); a step runs the smallest bucket above its position (first_fit, `models/model_wrapper.py:841-930`) and reads only that many cache rows (`model_wrapper.py:1564-1583` sets n_positions per bucket; `model_base.py:1316-1331` get_cache(seq_len = n_positions)). The c1 workload (128 prompt + 128 output tokens) decodes in the 256 bucket; FuseZap's pool prefix is 512 (FZ_NEURON_CTX_MIN default).
- *Where its 5.1 ms goes (estimate, not measured: NxDI was not profiled).* Weights read per token: 716 MB of layers + 272 MB lm_head in bf16 = 988 MB, so >= 193 GB/s average over the whole 5.11 ms including its host loop; FuseZap's bf16 layer streams its 30 MB in 0.237 ms (~127 GB/s) and its MatmulBF16 head 272 MB in ~1.5 ms (~180 GB/s). The difference is weight streaming and the attention span, not launches (both 1 per token) and not numerics: NxDI's extra bf16 rounding (residual, activations, KV, logits) is what costs it the greedy bar (10/24), and in FuseZap's LayerSpec the activations never leave SBUF, so storing them in bf16 would save nothing.

**What N8 builds (`fusezap/codegen/nki_bf16_n8.py`, each switch default off, each bitwise equal to the bf16 path by construction).** `FZ_NEURON_BF16_WN=1024|2048|4096`: LayerBF16's four weight matmuls load one (128, WN) bf16 tile per K chunk (2-8 KiB runs per partition instead of 1 KiB; gate_up 35 DMAs at 2048 instead of 133) and feed its 512-column slices to the same nc_matmul, each into its own PSUM tile, so every PSUM chain (one 512-column output block over the K chunks in order) is the same instructions on the same values; a transform of the rendered LayerBF16 source (`wide_weights`, NKIUnsupported when a block does not match), so it follows render_layer. `FZ_NEURON_BF16_PF=<k>`: the first k gate_up blocks (every K chunk of WN columns) load at the start of attention, when the DMA queues are idle; `FZ_NEURON_BF16_PFQO=1`: every qkv tile at the top of the kernel and every o_proj tile at the start of attention; the prefetch is clamped to `nki_bf16.LAYER_SBUF` (Qwen2.5-0.5B at 512 slots: 62 KiB of room, 28 KiB per gate_up block at WN 2048, 28 KiB for PFQO). `FZ_NEURON_BF16_HEAD=1` (+ `FZ_NEURON_BF16_HEAD_W`, default 2048): the lm_head as `HeadBF16`, HeadSpec's kernel with nki_bf16's edits on bf16 head tiles (`head_tiles_bf16`, relaid once at load; the K-major bf16 lm_head is then not kept): MatmulBF16's bits with 75 DMAs of 28 KiB runs instead of 2079 of 1 KiB. Hooks: `NeuronKVCache.layer` (`layer_bf16w` on the LayerBF16 spec), `neuron_kv.head_matmul` (bf16 tiles -> HeadBF16), `Llama.__init__` (the bf16 head tiles). The 256-slot bucket is the existing `FZ_NEURON_CTX_MIN=256` (not bitwise vs 512: the last key block is narrower). Not built: bf16 residual / activations (accuracy, no bytes saved), a bf16 KV pool (attention reads ~0.5 MB per layer at 512 slots and its cost is VectorE instructions: bf16a was 0.243 vs 0.239 ms), weight-stationary bf16 (TensorE is ~10 us of a bf16 layer's critical path). The float32 weight-stationary bug (v2sw* / head v4) is not fixed here; the likely cause, for the record: `_mm_ws` interleaves up to 512 / M accumulation chains inside one PSUM tile (one bank), each started with accumulate=False while its neighbours are mid-chain (odd_shapes v2swe down max |diff| 4.92, qwen v2swep 142; the residual, computed before the MLP, is bitwise): N8's wide tiles keep one chain per PSUM tile (a full bank at M <= 128 rows), several banks active at once.

**Defaults after the trn1 N8 run.** The trn1 run found the 4096-column tiles (layer 0.178 ms at 1 row, 0.185 ms at 4) and HeadBF16 at W 2048 (1.04 ms vs MatmulBF16's 1.52) bitwise vs the bf16 members on device and token-identical (24/24 prompts) to the bf16 baseline, so in bf16 `FZ_NEURON_BF16_WN` defaults to 4096 and `FZ_NEURON_BF16_HEAD` to 1 (`_HEAD_W` 2048); prefetch stays off. The wide default applies only to a whole untiled layer whose SBUF estimate plus the wider tiles fits (else 2048, 1024, off); tiled layers (I71, own `FZ_NEURON_LAYER_WCOLS`) and TP halves (I72) are unchanged, and HeadBF16 is off by default under TP (`FZ_NEURON_BF16_HEAD=1` turns it on). `FZ_NEURON_BF16_WN=512 FZ_NEURON_BF16_HEAD=0` gives I68's kernels. The 256-slot bucket (`FZ_NEURON_CTX_MIN=256`) was token-identical 24/24 too but stays opt-in. float32 defaults are unchanged.

**NKISHIM** (`tests/test_neuron_bf16_n8.py`, 46 tests): the wide / prefetching layer is bit-identical to the bf16 members for every LAYER_CASES case (Qwen 1 row / 512, 4 rows / 768 of 1024, odd shapes, two key blocks, one kv head) at WN 1024 / 2048 / 4096, with PF and PFQO (FZ_NKISHIM_MM=f64 as the other early-layer tests), and to LayerBF16 in the default shim too; HeadBF16 = MatmulBF16 bit for bit (1 / 4 / 130 rows, K tails, W 512-4096); through the model, prefill and decode logits equal the default bf16 path's bit for bit (FUSE=1 / 24, head fused, defaults) and the engine's tokens equal CPU float32's. **Cost model** (`scripts/nki_layer_cost.py`, refitted on 50 trn1 measurements incl. the 7 bf16 layers, a per-run DMA cost added: rms log error 0.080; the bf16 layers at 0.89-0.97x of trn1; the run length is not identified by these measurements, so the model credits wider DMAs only with fewer DMAs per queue and less tile rotation). **Predictions (model and arithmetic, not measurements), per layer at 1 row / 512 slots (model / 0.92 = its bias on the bf16 layers):** v2se+bf16 0.207 model (trn1 v2s+bf16 0.2395), w1024 0.194, w2048 0.180, w2048p1q 0.179, w2048p2 and w4096 0.167, at 256 slots each ~0.0125 less; head: the model puts MatmulBF16 and HeadBF16 both at ~1.16 ms (pure bandwidth at its fitted 241 GB/s), trn1 measured MatmulBF16 at 1.52 ms (1 KiB runs: 0.77x of the model), HeadSpec float32 streams 545 MB at 264 GB/s, so HeadBF16 ~1.0-1.15 ms. Per token c1 in process: the bf16 default with FUSE=24 + head fuse ~8.7 ms (I68's 9.49 at FUSE=4 minus step 4's 0.76 ms launch saving); + HeadBF16 ~8.2; + w2048p2 ~6.8; + CTX_MIN=256 ~6.5 (NxDI 5.11: the rest is FuseZap's ~1.4 ms of host + worker + runtime per token at one launch); c4 ~7.0-7.3 ms per step (~550 tok/s vs NxDI 448). **Compile risks:** the wide layer keeps 2-8 PSUM banks accumulating at once (WN 4096: all 8 of trn1's; a PSUM allocation failure is a compile error the bench records as SKIPPED); the prefetched tiles are live across attention (clamped to the SBUF estimate; v2swep's 96 KiB at the top of a float32 layer is the pattern that failed); HeadBF16 at W 4096 is 56 KiB per partition per block (bf16), no double buffering at 2 blocks.

**[unverified]** marks claims not confirmed against the docs this session. Sources are cited as [n].

## 1. Facts the design depends on
- **Memory and engines.**
  - SBUF: 24 MiB (128 partitions × 192 KiB). PSUM: 2 MiB (8 banks of 512 fp32 per partition). HBM: 32 GiB at 820 GB/s.
  - TensorE is a 128×128 systolic array: 92 TFLOPS in bf16/fp16, 23 TFLOPS in fp32, and its output is always fp32 into PSUM.
  - VectorE and ScalarE each have 128 lanes and compute in fp32.
  - A matmul needs K on the partition axis of both operands. The stationary tile is at most 128×128, and the moving tile is at most 128×512 [2].
- **Toolchain.**
  - `@nki.jit` called on numpy arrays compiles to `kernel.neff` under `NKI_ARTIFACTS_DIR`, and `NEURON_PLATFORM_TARGET_OVERRIDE=inf2` selects the target [3,6]. **Verified (NKI-N2, no device):** this needs `neuronx-cc` on `PATH`. `neuronx-cc==2.27.5334` is the release that installs next to `nki==0.6.0`; an unpinned `neuronx-cc==2.*` pulls 2.28 and upgrades nki to 0.7.0. The NEFF is written, then the call raises `ImportError: nki.runtime is not available` because there is no libnrt. Compiling one of our kernels takes 1.0–2.5 s on a 2-vCPU pod (median 1.36 s), and NEFFs are 11–114 KB. `trn1` compiles the same.
  - The CPU simulator runs a kernel with numpy I/O and **needs no device** [7]. **Verified (C-N1 review, real nki 0.6.0 wheel):** in 0.6.0 the entry point is `nki.simulate(kernel)(*arrays)`, not `nki.simulate_kernel`. **Verified (NKI-N2):** it writes outputs into the argument arrays in place and returns them. bf16 arguments must be `ml_dtypes.bfloat16` arrays; a uint16 array is converted by value, not reinterpreted.
  - **The simulator accepts more than the compiler (verified, NKI-N2, `scripts/nki_op_matrix.py`, inf2, neuronx-cc 2.27).**
    - `divide`, `fmod` and `mod` are rejected on every engine, in both `tensor_tensor` and `tensor_scalar`. The simulator runs all three.
    - GpSimd (Pool, "Gen2") compiles only int32 add/subtract/multiply (and bitwise ops). maximum, minimum and the compares are rejected there ("Gen2 Pool engine only supports add, subtract, multiply, and pow"), and so are float ops.
    - With no engine named, nki 0.6 sends int32 add/subtract/multiply to GpSimd (exact, wrapping). Everything else runs on VectorE in float32, so int32 max and compares are exact only below 2**24.
    - Compute instructions must start on a partition quadrant: a `memset` on `t[1:4, :]` is rejected. DMAs may start on any partition.
    - An HBM access pattern with a negative partition step is rejected; negative free-axis steps compile.
    - `tensor_scalar(multiply)` on int32 data fails MLIR verification.
    - The renderer avoids all of these, and NKISHIM refuses them (`assert`), so a kernel that passes the shim also compiles.
  - **TensorE, PSUM and masks (verified, NKI-N3, `scripts/nki_n3_probe.py`; each case simulated and compiled for inf2):**
    - `nc_matmul(dst, stationary, moving, accumulate=)` writes `stationary.T @ moving` into a float32 PSUM tile. `accumulate=False` overwrites, `True` adds. With `None` (the default) the simulator decides per element from a "written" bit; our kernels always pass `i > 0`. The simulator computes it as a float32 `np.matmul`.
    - Moving free ≤ 512 (one 2 KiB PSUM bank of float32), stationary free ≤ 128, K ≤ 128 on the partitions. 64 unrolled 128×128×512 matmuls with fresh tiles in each iteration compile, so the compiler reuses SBUF/PSUM across unrolled iterations.
    - `tensor_scalar`, `tensor_copy` and `tensor_tensor` read PSUM operands directly.
    - `nc_transpose(dst=PSUM, data=SBUF)` uses TensorE (an identity matmul). In the simulator the transpose is exact. On NeuronCore-v2 the docs say it is not bit-accurate for NaN/Inf **[unverified on hardware]**.
    - `iota(dst, pattern, offset, channel_multiplier)` writes `offset + p*channel_multiplier + Σ j_i*step_i` and can write straight into float32. `affine_select(dst, pattern, channel_multiplier, on_true_tile, on_false_value, cmp_op, offset)` keeps on_true where `cmp(affine, 0)`. With `-inf` as on_false_value it compiles for inf2.
    - `dma_transpose` takes 2-byte dtypes only (0.6 docstring), so the float32 kernels load transposed operands with a strided DMA access pattern instead.
    - The 0.6 simulator peaks at about **7× its HBM inputs**: a 224 MiB weight gave 1.8 GiB RSS. On a 4 GB pod, the largest weight it can run is about 400 MiB.
  - `nrtpy` (beta) runs a NEFF without PyTorch: `NrtpyModel.load_from_neff`, `NrtpyTensor.from_numpy`, `model(inputs=…)`, `.numpy()` [3].
- **Platform.** The `nki` (0.6.0, Neuron 2.32), `neuronx-cc` and `nrtpy` wheels are `linux_x86_64` only (checked against the pip index [5]), so nothing runs on the Mac. The `nki` and `nrtpy` wheels can't be installed in the same environment [3].

## 2. Mapping our ops to tile programs
The renderer lowers the kernel AST (`LazyOp`) to readable NKI Python.

| Kernel | Tile program |
|---|---|
| Elementwise | Flatten the output to `(P, F)`: outer dims go to P (128 per tile) and the inner contiguous run to F (512–2048, tunable). The body is an `nl.affine_range` loop of load → compute → store. Ops: ADD/MUL/MAX/CMP* → `nisa.tensor_tensor` (VectorE; int32 add/sub/mul on GpSimd). int32 MAX/CMPLT split each value into exact 16-bit halves, and CMPEQ tests `x - y == 0`. EXP2 → ScalarE `activation(exp, scale=ln2)`. LOG2/SIN/SQRT/RECIP → ScalarE or `nisa.reciprocal`. SIN range reduction uses `k = rint(x/2π)` via a float→int32 copy. WHERE → predicated copy. CAST → `tensor_copy`; float→int truncation is round-then-step-back (no fmod). BITCAST → an SBUF dtype view (**verified**: simulates and compiles). **IDIV/MOD → not supported** (verified, NKI-N2: neuronx-cc 2.27 has no divide/fmod/mod for inf2 on any engine; the 0.6 simulator's GpSimd divide would work, but it does not compile). They report as `NKIUnsupported`. |
| Reduce, axis on the free dim (RMSNorm, softmax rows) | Rows go on partitions. `nisa.tensor_reduce` runs along F, and wide rows accumulate into fp32 in SBUF. The prologue and epilogue fuse into the same loop. |
| Reduce, axis on the partition dim | SUM is `onesᵀ·X` on TensorE (accumulating in PSUM). MAX is `nc_transpose`, then a free-axis reduce. The renderer prefers a layout that puts the reduced axis on F. |
| Matmul (custom hook, **done N3**) | `render_matmul`: `nc_matmul` with stationary `Aᵀ[K≤128, M≤128]` and moving `B[K≤128, N≤512]`, accumulating ⌈K/128⌉ steps in one PSUM bank. PSUM is then evicted to SBUF and stored. A row block's Aᵀ chunks stay in SBUF across the N tiles while they take ≤ 64 KiB per partition. `neuron.kmajor(w)` relays a weight K-major once at load time, so each moving tile is one contiguous run per partition. A torch-layout (N, K) weight and the activation's Aᵀ load use a strided DMA access pattern, because `dma_transpose` is 2-byte only and these are float32. A negative partition step (a flipped operand) becomes one DMA per row. |
| Attention (custom hook, **done N3**) | `render_attention`: flash tiles. `qᵀ` (stationary, D on the partitions) times kᵀ gives S (≤ 128 rows × ≤ 512 keys) in PSUM, scaled on the way to SBUF and masked. VectorE computes the row max. ScalarE computes `exp(s−m)` with the new max as a (P, 1) bias, and VectorE computes the row sum. The running sum and output are rescaled by `exp(m_old−m_new)`. P is transposed on TensorE in 128-key chunks for P·V, which accumulates in PSUM. The running max starts at −FLT_MAX, so a fully masked key block contributes zeros. Masks: `affine_select` for causal (key blocks past a row tile's last visible key are skipped), `iota` key indices against an int32 position tensor for decode, and a materialized bool mask for the scheduler's masked pattern. GQA: the q heads of a kv head are folded into the rows, so they share K/V reads. |

**Views.**
- Affine single views (permute, expand, shrink, flip) become DMA access patterns, at no cost.
- PAD masks become `iota`, compare, select.
- Views that need div/mod or several views are materialized by a DMA gather kernel.

## 3. Static shapes and buckets
NKI has no runtime integer arguments, so every `Var` becomes part of the compile key.
- Each `var_val` is rounded up to a power-of-two bucket ≥ 16, matching `llama.py`'s prefill buckets. The tail is masked.
- Decode uses serve's existing `full_read=True` path: it reads the fixed-size cache and masks by an int32 position *tensor*. Decode therefore has no symbolic dims and needs one NEFF per batch bucket (1/4/8/16/32).
- NEFFs are disk-cached, keyed by nki version, target and source.
- `fusezap neuron warmup <model>` precompiles every bucket. Compiling needs no device, so warmup can run on a cheap Linux CPU pod and the NEFFs are then rsynced to the inf2 instance.

## 4. The 4-part interface, and the exceptions to it
| Part | NEURON implementation |
|---|---|
| Renderer | `fusezap/codegen/nki.py: NKIRenderer` |
| Compiler | `NKICompiler.compile(src) -> NEFF bytes`. Runs as a subprocess in `.venv-nki`, and `cache_key` holds the nki version and target. N2 has `neuron.compile_neff(prog, arrays, target)`: the worker calls `nki.jit` with the argument shapes and returns `kernel.neff`. nki traces with concrete shapes, so the compiler needs the buffer sizes, not only the source. |
| Runtime | `NeuronNeffProgram` (N4b): `NrtpyModel.load_from_neff` once, in the `.venv-nrt` runtime worker. Buffers are bound by tensor name: `data<i>` or `data<i>.must_alias_input` maps to argument i, and a must-alias output runs on a `slice_from_tensor` of its input (`neuron.neff_binding`; the bindings are built once per buffer set). `NeuronHwProgram` (nki.jit, `FZ_NEURON_RT=jit`) is the fallback. |
| Device | `fusezap/device/neuron.py`: `NEURONDevice` (N4) and its tiers `NKISHIMDevice` (N1: the compiler returns the source, `NKIShimProgram` runs it through the shim) and `NKISIMDevice` (N2). Programs are cached by (AST, bound `var_vals`, tile limits). Custom items (N3) are rendered by `fusezap/codegen/nki_kernels.py` and cached by (kernel, params). |
| Allocator | `NeuronAllocator(LRUAllocator)` over `NrtpyTensor`, with synchronous copyin/copyout through numpy. |

**Exceptions to the abstraction.** We publish these as-is.
1. **UOps are bypassed.** UOps describe per-thread scalar SSA. NKI works in 128-partition tiles, so only `ShapeTracker` is shared, and only for affine views.
2. **No runtime vals.** Symbolic dims are compiled in, and `global_size`/`local_size` are ignored. `run_compiled`'s cache key must include the bucketed `var_vals`.
3. **The matmul/attention match is mandatory.** As composed ops, matmul is a MUL plus a SUM and never reaches TensorE. A missed match is a performance bug, not a fallback. **N3:** NEURON also offers an `"attention_mask"` pattern to the scheduler: `WHERE(bool mask, scores, -inf)` before the softmax, with the mask as a 4th input. That covers is_causal, a bool attn_mask and llama.py's position-mask GQA graph (prefill and decode). Only NEURON lists it, so every other device keeps those graphs as before. A float additive mask (`qk + attn_mask`) still splits into two `fz_matmul`s. Missing matches: float32 only (bf16 operands are matched only as float32 casts, N5). The hook needs float32 Q/K/V with single-view strides. A mask whose query rows run backwards (a negative row stride, e.g. `mask.flip(0)`) is not matched and stays composed, because the renderer loads mask rows with an ascending DMA pattern (NKI-final review fix; a negative key stride is matched).
4. **The allocator may need types.** NEFF I/O is shaped and typed. If nrtpy can't alias raw bytes, allocations are keyed by `(size, dtype)`. **Read from source (NKI-N3, no device):** the nki 0.6.0 wheel bundles an `nrtpy` package (`SpikeModel`, `SpikeTensor` over `_nrtpy.so`, which needs libnrt). A device tensor is allocated by byte size only (`allocate_tensor(size=array.nbytes)`). Shape and dtype are Python-side metadata on the `SpikeTensor` wrapper. `slice_from_tensor` makes aliasing views, which the model uses for must-alias outputs. So untyped byte buffers with typed wrappers look possible, and the `(size, dtype)` key is probably not needed. This is **[unverified on hardware]** (N4). **N4b:** `NeuronAllocator` allocates untyped uint8 device tensors by id. `_alloc` only names the buffer; the worker allocates it, zeroed, with the next request, and frees ride along the same way. A buffer larger than a NEFF tensor is bound through a leading `slice_from_tensor`. The standalone nrtpy 2.35.30 wheel has the same API as the bundled one (`NrtpyModel`/`NrtpyTensor`, `alias_info` parsed from the NEFF's `must_alias` metadata). Run 1 confirmed that allocation, write and read work on inf2. Whether execute accepts uint8 tensors for float32 NEFF tensors is still **[unverified]**: the rerun's first launch answers it.
5. **Two Python environments**, because of the wheel conflict. The compiler is a subprocess, not an import.
6. **BEAM search space.** Our `Opts` (upcast/local/group_reduce) mean nothing here. NEURON gets its own knobs: F width, the partition axis, VectorE vs ScalarE, and buffer depth. Every candidate costs a compile, so BEAM runs only in warmup, at width ≤ 4. N6 knobs (env, measured per variant on hardware, not searched yet): layers per fused kernel (`FZ_NEURON_FUSE`), the smallest pool prefix (`FZ_NEURON_CTX_MIN`), operands by rows + TensorE transpose (`FZ_NEURON_KT`, `FZ_NEURON_AT_ROWS`); N7: the head (`FZ_NEURON_HEAD`, `FZ_NEURON_HEAD_W`, `FZ_NEURON_HEAD_DMAS`), the layer path (`FZ_NEURON_LAYER`, `FZ_NEURON_LAYER_HEADS`, `FZ_NEURON_LAYER_ROPE`), extra pool prefixes (`FZ_NEURON_CTX=mid`).
7. **Dtypes.** There's no int64, bool is uint8, and fp32 matmul runs at ¼ of the bf16 rate. If IDIV/MOD has no engine op, the conformance report shows it failing.
8. **Serve's Metal kernels** (paged KV write, row gather, decode attention in `kv_cache.py`) have NEURON versions without indirect DMA (N5, `fusezap/serve/neuron_kv.py`): a one-hot TensorE scatter for the write, a one-hot matmul for the row gather, and a "keyrange" mask over every pool slot for attention. Their cost grows with the pool size (`--num-blocks`).
9. **Launch granularity.** One schedule item becomes one NEFF execution (~170 µs each on inf2, N4 run 2). N5: launches are batched into one runtime-worker request per jit replay, and the serving path uses three hand-written NKI kernels (KV write, RMSNorm + residual, attention) where the composed graph took several items; SPEC §8 still has no slot for fusing scheduler items into one kernel. N6: the NEURON device rewrites a captured fz.jit forward itself (`fuse_schedule`, called by `jit._device_fuse` for a capture whose items all run on one device that offers it): items between two model-recorded fences become one fused NKI kernel, one launch per decoder layer. The model change is the fence (`neuron_kv.fence`, NEURON only); the fused kernel is the members' rendered kernels concatenated, so no new lowering.

## 5. Vendor kernels through the custom-kernel hook
`NEURONDevice.custom_kernels = {"matmul": CustomKernel("fz_matmul", …), "attention": CustomKernel("fz_attention", …), "attention_mask": CustomKernel("fz_attention_mask", …)}`, keyed by the tier name (NEURON / NKISHIM / NKISIM). **Done in N3.**
- Params are baked into the source, and `NEURONDevice.run` handles `kind="custom"` itself, as METAL does. An unknown custom item raises `NKIUnsupported`.
- The kernels follow the structure of the nki-samples tutorial matmul and fused attention (MIT-0) [8], written for the nki 0.6 API. No sample code is copied. RMSNorm stays a composed row reduce.
- Direct entry points: `neuron.attention(q, k, v, causal=, coff=, pos=)` is flash attention with GQA (causal via affine_select; decode via a position tensor), and `neuron.kmajor(w)`. Wiring them into llama.py/serve is N5.
- `nkilib` (Apache-2.0: attention, MLP, QKV, RMSNorm) [9] only claims Trn2/Trn3 support [10], so on inf2 we use a kernel from it only if it passes our suite there.
- Every vendor kernel must match its composed equivalent in `tests/test_custom.py`.

## 6. Test plan
| Tier | Where | Gate |
|---|---|---|
| `NKISHIM` | Mac, every commit | Golden-source tests, plus a **numpy shim** of the `nl`/`nisa` subset we emit. The shim executes the generated kernels, so `test_ops.py` runs locally: `FZ_TEST_TIERS=NKISHIM uv run pytest tests/test_ops.py`. A kernel the renderer refuses reports as **xfailed with its `NKIUnsupported` reason** (1386 passed, 48 skipped, 9 xfailed). Since N2 the shim matches the real simulator bit for bit on the op suite, and it refuses what neuronx-cc rejects for inf2. |
| `NKISIM` | RunPod CPU pod (x86_64, Python 3.12, `nki==0.6.0`) | The device runs `nki.simulate` in a `.venv-nki` worker process (`scripts/nki_env.sh`). Gate (NKI-N2): `test_ops.py` gives **1386 passed, 48 skipped, 9 xfailed, the same per test as NKISHIM**. `FZ_NKISIM_SHIM_CHECK` compared each of 1924 kernel runs bit for bit with the shim and found 0 differences. `FZ_NKISIM_NEFF` compiled all **1233 distinct op-suite kernels to inf2 NEFFs with 0 failures**. The test_custom cases that apply before N3 (composed matmul/attention) are in `tests/test_neuron.py`. |
| `NKISIM` custom kernels (N3) | the same pod | **Gate met (NKI-N3).** `tests/test_custom.py` passes on NKISIM (33 of 33, the same as NPY/CPU/NKISHIM), and so does `tests/test_neuron.py`. The op suite on NKISIM is unchanged at 1386 passed, 48 skipped, 9 xfailed, and its matmul/linear/sdpa cases now run 92 `fz_matmul`, 5 `fz_attention` and 5 `fz_attention_mask` in the simulator. The shim check shows 0 bitwise differences. 1272 of 1272 distinct kernels compile for inf2, 58 of them custom. `scripts/nki_n3_gate.py` covers 42 matmul and attention cases at the Qwen2.5-0.5B/7B shapes (GQA, causal prefill to 640, chunked prefill, decode to S=1024, the llama.py graph), all within 2e-6 (matmul) and 1e-5 (attention) relative to float64. The largest seen are 7.0e-7 and 8.3e-7. Every kernel compiles for inf2. |
| `NEURON` | inf2.xlarge, Neuron DLAMI (Ubuntu 24.04; SDK 2.33 on 2026-10-20) via the EC2 runner; job `scripts/nki_n4_inf2.sh` (run 1: 2026-10-26; N4b rerun pending). The tier runs on the persistent path (`NeuronNeffProgram` in the `.venv-nrt` runtime worker, NEFFs compiled once in `.venv-nki`). `FZ_NEURON_RT=jit` selects nki.jit (`NeuronHwProgram`; run 1: correct, 1.31 s per call). The Mac checks the path with a fake nrtpy that runs kernels through NKISHIM (`tests/test_neuron_nrt.py`) | The op suite. Qwen2.5-0.5B-Instruct **token-identical to HF for 64 tokens in f32**, and bf16 top-1 ≥ 99%. The serve batch tests. |
| `NEURON` (7B) | inf2.8xlarge | Qwen2.5-7B in bf16. Throughput against vLLM-Neuron 0.24 / NxDI on the same instance [11]. |

- The tiers are added to `conftest.TIERS`, are selected with `FZ_TEST_TIERS`, and each gets a column in the P5-KIT conformance report.
- **remote.py change.** `pod_create_body` hard-codes `computeType: "GPU"`. Add `--cpu <flavor>` and `--vcpus N`, which set `computeType: "CPU"`, `cpuFlavorIds` (`cpu3c…cpu5m`) and `vcpuCount` [12].
- Also add a CPU price-hint table for the cost guard, a CPU image with `/start.sh`, and mocked tests. About half a day.
- **Done (C-N0, NKI-N0).** The image is `runpod/base:1.4.1-ubuntu2404`, **verified** by real launches (2026-10-19 and 2026-10-21): x86_64, Python 3.12.3, system pip. Real price: $0.06/h for cpu3c and $0.07/h for cpu5c, both at 2 vCPU. On such a pod, `pip download --no-deps nki==0.6.0 --extra-index-url https://pip.repos.neuron.amazonaws.com` fetches `nki-0.6.0+31049202112.g85070674-cp312-cp312-linux_x86_64.whl` (17.5 MB). Some cpu3c hosts never start the container, so use `--boot-timeout 300`.

## 7. Measuring time to second model
- **Start:** the commit where Qwen2.5-0.5B passes the NEURON gate.
- **Stop:** Qwen2.5-7B passes on inf2.8xlarge. Optionally also Mistral-7B, as a second architecture.
- **Record:**
  - wall-clock time
  - agent output tokens
  - instance-hours and dollars
  - NEFF count and compile time
  - **lines changed, split into backend (`device/neuron.py`, `codegen/nki.py`) and model/serve**
- **Claim:** zero backend lines. Every backend line it needs is added to §4.
- **Contrast:** NxDI/vLLM-Neuron ship per-architecture model code.

## 8. Milestones
| # | Work | Effort |
|---|---|---|
| N0 | remote.py CPU pods (**done**: C-N0, NKI-N0) | 0.5 d |
| N1 | Renderer (elementwise and reduce) + shim + NEURON device skeleton and `NKISHIM` tier (**done**: C-N1, NKI-N1) | 1 wk |
| N2 | `NKISIM`, op suite in the simulator, NEFF compile path (**done**: NKI-N2) | 1 wk |
| N3 | matmul and attention kernels (sim) (**done**: NKI-N3; review fix in NKI-final) | 1 wk |
| N4 | `NEURON` device, op suite on hardware; **measure per-NEFF launch overhead first** | 1 wk |
| N5 | Serve: buckets, KV/gather kernels, 0.5B token-exact (**Mac part done**; hardware: `nki_n5_inf2.sh`) | 1–2 wk |
| N6 | NEURON knobs (fusion, pool prefixes, operands by rows), bench against vLLM-Neuron (**Mac part done**; hardware: `nki_n6_inf2.sh`) | 1 wk |
| N7 | 7B on inf2.8xlarge, then a structurally different second model (e.g. Llama-3.1-8B or a small MoE). **Record the effort for each model separately:** backend lines changed, new exceptions, agent-hours and wall time. Publish both numbers. | 3–5 d |
| N8 | Conformance report, exceptions list, PORTING.md | 2 d |
| N9 | Failure behaviour on hardware: request cancel, device-memory exhaustion, kernel/NEFF load failure, runtime restart after a Neuron device reset, overload backpressure (p50/p95/p99 TTFT/TPOT), plus a 24 h sustained-load run. Goes into the qualification report. | 3–4 d |
| | **Total** | **8–10 wk** |

Hardware: inf2.xlarge, inf2.8xlarge and trn1.2xlarge. N0–N3 need no Inferentia/Trainium instance.

## 9. Open risks
1. **Inf2 is becoming second-class.** The current ISA and Library docs say "Trn2, Trn3" [1,10], and only the about page still lists inf2/trn1 [4]. SDK 2.33's vLLM is Trn2/Trn3 only; inf2/trn1 inference is on the PyTorch/XLA stack in maintenance (N6b) [13-15]. *Mitigation:* pin SDK 2.32 / nki 0.6.0 and keep trn1 as the fallback.
2. **API churn.** `nki` went from 0.1 to 0.6 in about a year, and the legacy `neuronxcc.nki` API differs from it. *Mitigation:* keep all NKI spelling in one renderer table.
3. **Per-NEFF launch overhead is unknown.** Decode runs hundreds of kernels per token, so a large overhead makes exception 9 the critical path. *Mitigation:* measure it first in N4. Run 1 measured only the nki.jit path (1.31 s, a recompile and reload per call). The loaded-NEFF launch on the persistent path is the rerun's first measurement. Its floor is one pipe round trip (~0.36 ms in run 1) plus nrt execute, so decode at hundreds of launches per token still needs exception 9 (fused kernels) or launches batched in the worker (N5).
4. **f32 token-exactness.** ScalarE transcendentals and the reduction order may break it **[unverified]**. *Fallback:* the bf16 top-1 gate, reported honestly.
5. **Compile time** grows with buckets × kernels. *Mitigation:* the cache, plus precompiling on a CPU pod.
6. **Simulator fidelity.** Out-of-bounds indexes are only checked at runtime [3].
7. **inf2.xlarge has 16 GiB of host RAM**, so the 7B work moves to inf2.8xlarge.
8. **The Inf quota defaults to 0.**
9. **Fused kernels on hardware (N6).** A fused layer reads, in a later member, HBM buffers an earlier member wrote; nki 0.6 has no explicit barrier, so this relies on neuronx-cc's dependency tracking through DRAM tensors **[unverified]**. A fused NEFF is also ~10x a member's instruction count (compile time per (bucket, prefix), the NEFF cache keeps it). *Mitigation:* fused vs unfused logits on the chip in the kernels phase, the token check of the default, a compile/load failure falls back to the members, and `FZ_NEURON_FUSE=0` restores N5's launches.

## 10. bf16 (I68) and Qwen2.5-7B on inf2.8xlarge (I69)

**I68: bf16 (2026-10-27, Mac; hardware job `nki_bf16_trn1.sh` pending).**
- **Switch.** `fusezap serve --device NEURON --dtype bfloat16`, `Llama(..., dtype=bfloat16)` on a NEURON tier, or `FZ_NEURON_DTYPE=bf16` with any dtype. Other devices are unchanged.
- **What is bf16 (`fusezap/codegen/nki_bf16.py`).** The linear weights (qkv, o_proj, gate_up, down, lm_head) are bf16 in HBM, relaid K-major on the host at load, so no float32 copy ever exists on the device. lm_head is kept K-major only, with no torch-layout copy (7B: 1.09 GB saved). Every weight matmul runs on TensorE with bf16 operands: the weight's moving tiles come straight from HBM, and the activation's stationary Aᵀ chunks are rounded to bf16 (one `tensor_copy` per chunk block, round to nearest even). Products are exact and accumulate in float32 PSUM, the same contract as the CUDA bf16 GEMM. The residual stream, RMSNorm, RoPE, softmax with the exact exp, the KV pool, logits and argmax stay float32 (`model.dtype` stays float32), so a NEURON bf16 forward rounds only the weight-matmul operands. That is fewer rounding points than the CUDA bf16 path, whose residual stream and activations are bf16 between kernels.
- **Attention.** By default attention stays float32 on a float32 pool: it reads ~0.5-2 MB per layer at a 512-slot prefix, against 30-470 MB of weights. `FZ_NEURON_BF16_ATTN=1` also rounds q, k (after RoPE), p and v for q·kᵀ and p·v. The pool stays float32, so the KV write stays an exact one-hot scatter.
- **How the kernels are made.** The bf16 kernels are the float32 kernels' rendered sources with three mechanical edits (`bf16_weight_matmuls`):
  1. A weight's moving tile becomes bfloat16.
  2. A bf16 copy of the stationary tile is made before the loop that consumes it.
  3. The weight arguments' dtypes become bfloat16.

  So `MatmulBF16` (`render_matmul`) and `LayerBF16` (`render_layer`) follow every change to the float32 renderers, and an edit that no longer finds its pattern raises `NKIUnsupported`. Fused kernels take their members' sources and dtypes, so FUSE=k needs nothing. `Llama.linear` sends a bf16 weight to `neuron_kv.matmul_bf16`. `NeuronKVCache.layer` picks `LayerBF16` when the layer's weights are bf16.
- **Mac results (NKISHIM, `tests/test_neuron_bf16.py`, 32 tests).**
  - `MatmulBF16` is within float32-accumulation error of numpy's bf16-emulated reference (bf16(A) @ W in float64) in every layout `render_matmul` takes.
  - `LayerBF16` is bit-identical to the bf16 members for every LayerSpec case and variant: the same Aᵀ values, rounded the same way.
  - `FZ_NEURON_BF16_ATTN` is within tolerance of a float64 reference on bf16-rounded q, k, v.
  - Through the model, v1 and v2 give the same logits bit for bit, fused or not. Engine tokens equal CPU float32's on the tiny checkpoint, and on a small model with Qwen2.5-7B's head layout (28 / 4 heads, head_dim 128).
  - **Qwen2.5-0.5B bf16 vs the float32 reference: teacher-forced top-1 0.99784 (461/462 positions, 8 prompts), |logprob error| mean 0.0036 (max 0.045), greedy 8/8 prompts exact over 32 tokens (238/238).**
- **The bf16 bar** (`nki_bf16_compare.py`; the bars the repo already uses for bf16, FuseZap Invarix):
  - teacher-forced top-1 ≥ 0.98 (`accuracy.top1_agreement`; CUDA bf16 Llama-3.3-70B TP=2: 0.9993);
  - greedy exact ≥ 0.70 (`TEXT_MATCH_DEFAULTS`, Qwen2.5 0.5B / 7B bf16);
  - greedy not more than 5 points below the AWS bf16 baseline.

  The float32 bar (greedy exact ~1.0) does not apply to bf16.
- **Cost model** (`scripts/nki_layer_cost.py --variants v2,v2+bf16`; TensorE at 4x for a 2-byte moving operand): per layer at 1 row / 512 slots, v2 0.218 ms becomes v2+bf16 0.182 ms; at 4 rows, 0.252 becomes 0.215 ms. Engine busy time halves (DMA 148 → 76 µs, TensorE 184 → 64 µs), but the schedule is latency-bound. **Prediction, not a measurement:** step 3's measured 0.32 ms per layer drops to ~0.27 ms. The lm_head (545 MB → 272 MB) drops from 2.06 ms to ~1.1 ms. c1 TPOT ~10 ms in process on trn1, against 11.8 ms in float32.
- **Hardware risks [unverified]:**
  - nc_matmul with a bf16 stationary from a `tensor_copy` cast, and with a bf16 moving tile DMAed from HBM: standard nki usage, but not yet compiled here.
  - Casting the whole stationary tile reads its unwritten partitions (K tail chunks). This is harmless on hardware, and NKISHIM allows it.
  - neuronx-cc's SBUF allocation with the extra bf16 tiles in LayerSpec.

**I69: Qwen2.5-7B-Instruct bf16 on inf2.8xlarge (2026-10-27, Mac; hardware job `nki_7b_inf2.sh` pending).**
- **One NeuronCore or TP=2?** One core is enough for capacity. TP=2 is needed for speed.
  - The Inferentia2 device has 32 GiB of HBM and two NeuronCore-v2 [22]; `neuron-ls` on inf2.xlarge and trn1.2xlarge shows "32 GB" for the one device with 2 cores (docs/status env logs).
  - AWS: "Each NeuronCore has a dedicated high bandwidth memory (HBM) of 16 GB storing tensor parallel modules" [21]. AWS's sizing rule is weights + KV < 16 GB × TP degree.
  - FuseZap's device weights for 7B in bf16 are 14.14 GB: the linear layers 13.05 GB plus lm_head 1.09 GB. The embedding table stays on the host, and there is no torch-layout lm_head copy. Add a 117 MB float32 KV pool (1024 slots) and the NEFFs: one core's 16 GiB (17.2 GB) holds it with ~2.5 GB to spare.
  - Decode reads every weight per token, and one core's HBM delivers ~410 GB/s (N7 serve's per-layer floor). That is a ≥ 34 ms/token floor on one core against ~17 ms on two, which is why NxD Inference runs 7B at TP=2 on inf2.
  - **[unverified]** whether the runtime gives one core's allocations the full 16 GiB. `scripts/nki_hbm_probe.py` (the job's first device phase) measures it. If core 0 alone holds < 15 GiB, the job skips the FuseZap phases: then TP=2 is required for capacity too.
- **This pass:** 7B on one core through the I68 bf16 path, with no new backend code. One 7B-specific guard: LayerSpec keeps a step's rows whole in SBUF, ~400 KiB per partition at D 3584 / I 18944 against 192 KiB. `nki_bf16.layer_fits` sends such a layer to the members, which tile every operand (`FZ_NEURON_LAYER_SBUF`, default 176 KiB; Qwen2.5-0.5B needs ≤ 165 KiB even at the full 4096-slot pool, so its path is unchanged).
- **The job.** A tiled 7B LayerSpec (gate_up / down in column blocks) is the next speed step. The job runs FUSE=1, one decode size (`FZ_DECODE_BUCKETS=0`) and one pool prefix (`FZ_NEURON_CTX=0`) to bound the NEFF count, and compares against AWS NxD Inference offline at TP=2 bf16. The reference is the repo's `qwen2.5-7b-instruct-f32.json` (HF float32 compute over the bf16 checkpoint, the one the CUDA 7B runs used). No CPU reference is built on the instance.
- **TP=2 design (not implemented in this pass).**
  - Mirror Core TP (`fusezap/serve/dist.py`). `Llama(tp=TPGroup)` already slices weights (`tp_slice`) and sets the local config (`local_cfg`): 14 of 28 query heads and 2 of 4 kv heads per rank, column-parallel qkv / gate_up, row-parallel o_proj / down with an all-reduce after each, and `TPGroup.spmd` mirroring prefill and decode.
  - NEURON side: one runtime worker per core (`NEURON_RT_VISIBLE_CORES=0` / `1`), one fusezap process per rank as on CUDA. `Llama.kmajor` is off under TP today (`self.tp is None`), so the K-major relayout must move to after `tp_slice`.
  - The all-reduce is the open part, and nki 0.6 has no collective instruction. The candidates:
    - (a) Through the host: read both partials (N × D float32, 14 KB per row), add, write both back. That is 56 round trips per token at ~0.3-0.4 ms each, ~20 ms: it cancels the gain.
    - (b) nrt collectives: aws-neuronx-collectives 2.35 is installed, but nrtpy exposes no collective call, and a NEFF needs neuronx-cc's collective ops, which NKI kernels do not emit.
    - (c) Cross-core tensor binding: one runtime process owns both cores and launches a NEFF on core 1 with a tensor allocated on core 0 (the same device HBM). Then the all-reduce is an add kernel on each core over the peer's partial, plus a sync, and the two cores' executes must overlap (nrtpy's execute is synchronous: one thread per core).

    (c) is the only candidate that keeps one round trip per step. It needs a hardware probe first: `nrt_tensor_allocate` on nc 0 bound into an nc 1 execute.
- **Interface misfits so far (the N7 record for this model).**
  - Backend lines (`device/neuron.py`, `codegen/nki.py`): 0.
  - Model/serve lines: `models/llama.py` +25, `serve/neuron_kv.py` +18. bf16 itself (`nki_bf16.py` and its tests) is I68 precision work, not per-model.
  - New exceptions: none for 1-core 7B. TP=2 would add exception 10: one NEFF per core and no collective in the NKI ISA, so the all-reduce is host-driven or cross-core binding, outside the 4-part interface.
  - Agent wall time for I68 + I69 in this pass: ~3 h (Mac only).

**I71 (N8): Qwen2.5-7B on one NeuronCore, HBM headroom and a tiled decoder layer (2026-10-28, Mac; hardware job `nki_n8_7b_inf2.sh` pending).** I69 run 1: 110 ms/token c1, 34-36 tok/s c4 on the members (29 launches, 97 ms device), and NRT_RESOURCE at 15.96 GiB of HBM (in-process top-1 scoring crashed; 12 served accuracy requests returned 500, Invarix accuracy BLOCK).
- **Where the ~2.6 GiB beyond the weights went.** The run's own table: tensors 15.83 GiB = weights 13.17 + KV pool 0.11 + **~2.55 GiB of activations**; NEFF code 0.11 GiB and DMA rings 0.02 GiB (47 NEFFs) were not the problem, nor duplicated weights (no torch-layout copies, embedding on the host). Two causes, both host-side policy:
  1. On NEURON fz.jit did not plan (`plan=cuda` only): every captured forward kept each intermediate of every layer allocated for good. `scripts/nki_hbm_account.py` on NKISHIM with 7B's layer shapes (1 and 2 layers, bf16, extrapolated): prefill_128 **45 MiB per layer = 1.24 GiB at 28 layers**, the scorer's 512-row bucket (a 300-token accuracy request) **205 MiB per layer = 5.6 GiB**; decode_4 0.3 MiB per layer.
  2. NeuronAllocator's free list was unbounded (LRUAllocator with no cap): the eager and capture passes' buffers (the same ~1.2 GiB for prefill) stayed on the device after the capture made its own. And the budget never triggered `free_cache`: allocations ride lazily to the worker, so the runtime's refusal surfaced mid-batch (launch 3 of 4 had run launches 0-2) instead of as a MemoryError.
  Also: the scorer's trunk (`Llama.forward`) recorded no layer fences, so on NEURON `fuse_schedule` fused up to 44 items across layers into one NEFF, inside which nothing can share memory.
- **Fixes (all host-side; one runtime-worker change).** `FZ_NEURON_PLAN=1` (default): the engine's decode / prefill forwards, `NeuronSteps` and the scorer plan their intermediates (`nrt_worker._ref` already binds a NEFF tensor to a leading slice of a larger arena). Per-layer fences in `Llama.forward` on NEURON (same items, same bits). `FZ_NEURON_ALLOC_CACHE_MB=256` caps the free list. `FZ_NEURON_HBM_GB=15.25` (core 0 took 15.85 GiB of tensors next to 0.13 GiB of NEFF code and rings): an allocation past it hands the free list back and retries, then raises `NeuronOOM` (a MemoryError) before anything reaches the worker. The worker allocates every new buffer of a request (a batch: all of them) before running any of it; on NRT_RESOURCE it frees those and replies `oom` (host: `NeuronOOM`; nothing ran). `FZ_LOGPROBS_JITS=2` on NEURON: at most two scorer buckets captured (LRU). The scorer already released its captures and retried once on MemoryError; the API now answers a MemoryError with **503 engine_out_of_memory** (retryable) instead of 500, and after a step's MemoryError drops the scorer's captures and the free list. Accounting: `neuron_kv.hbm_report` / `hbm_line` (weights + constants, KV pool, each captured forward, scorer, free list, NEFFs loaded and their bytes, the worker's own count), printed by `fusezap serve` after warm-up and by `nki_n6_serve.py` after prewarm / greedy / scoring; `FZ_NEURON_HBM_LOG=1` adds one line per NEFF load and per GiB crossed.
- **Mac measurement (NKISHIM, 7B layer shapes, `nki_hbm_account.py`, extrapolated to 28 layers; prediction for the device):** captured forwards + scorer after warm-up (decode 4 rows + prefill 128) **1.23 GiB unplanned -> 0.19 GiB planned**; after scoring one 300-token sequence (bucket 512) **6.9 GiB -> 0.94 GiB**. A planning fix in `fz.jit._plan` was needed for that: buffers first written by the same item now take arenas largest first (smaller ones used to grab the 19 MB gate_up arena, so a fused 7B layer + head added a new one per layer: 0.83 GiB planned before the fix). Predicted device total for 7B after Invarix's accuracy requests: 13.17 weights + 0.11 KV + ~0.2-0.9 activations + <= 0.25 free list ~ 14.5 GiB of the 15.25 budget; a much longer scoring request (a 1024- or 2048-row bucket) would exceed it, release the scorer's captures, retry once, then answer 503.
- **The tiled layer (`LayerSpec.tile`, `FZ_NEURON_LAYER_TILE=auto|1|0`, default auto = tile a layer whose untiled rows do not fit SBUF).** Same instructions per element as LayerSpec (so the members' bits), restructured so nothing row-wide but D stays in SBUF: four (N, D) row tiles reused (`ra`: d, then o_proj's output, then down's; `rh`: both norm outputs; `w1`: ln1 then ln2; x2 computed into x1); qkv evicted per 512-column slice with its bias (held in X's columns until the add) and RoPE straight into X (no (N, Nq) row, no (N, QK) xa / fl); gate and up per column block, each slice transposed and silu'd into aT, the down projection's Aᵀ (no (N, 2I) row: 148 KiB at 7B). Weight DMAs of `FZ_NEURON_LAYER_WCOLS` columns (default 2048: 4 KiB contiguous runs per partition in bf16, 512 KiB per DMA, four 512-column matmuls per DMA into four PSUM banks) instead of 512; per output column the same stationary chunks and accumulation, so the same bits whatever the width. SBUF estimate (`layer_sbuf_bytes`): 7B at 4 rows 400 KiB untiled; tiled 134 / 167 KiB at 512 / 1024 slots (512 columns), 143 / 176 KiB (2048); past 1024 slots the members. `nki_bf16`'s edits needed two generalizations (several DMAs into one tag's tiles; `ps<tag>_<j>` PSUM tiles and slices of a wide weight tile), so LayerBF16 of the tiled layer is still a mechanical edit.
- **Simulator results (NKISHIM, FZ_NKISHIM_MM=f64):** the tiled layer is bitwise equal to the members for all 5 LAYER_CASES (rows 1-5, prefixes 300-768 of 1024, D / I not multiples of 128, no residual term, o_proj bias), heads each / stacked, 512 / 1024 / 2048 columns, float32 and bf16 (40 cases); at Qwen2.5-7B's full shapes (4 rows, 512 slots, bf16) tiled == untiled LayerBF16 on every output (512 and 2048 columns); through a model with 7B's head layout (28 / 4 heads, hd 128) prefill and decode logits equal the members' bit for bit at FUSE=1 and 24. Non-tiled kernels render byte-identical to the base branch.
- **Cost model (`scripts/nki_layer_cost.py --model 7b`; predictions, not measurements).** Per layer, bf16, 4 rows: tiled 1.36 ms (512 slots) / 1.43 ms (1024), DMA-bound on the HBM link at the fitted 393 GB/s (466 MB of weights per layer: floor 1.19 ms); the members 1.63 / 1.79 ms; the wide DMAs change little in the model (1.42 vs 1.44). Per token with the bf16 head (2.86 ms): tiled 40.9 / 43.0 ms, members 48.4 / 53.1 ms. **The model is not calibrated at 7B:** run 1 measured 97.1 ms device per token on the members where it says 53.1 (0.55x; its bf16 numbers ran ~0.67x of trn1 on Qwen2.5-0.5B too). Scaled by run 1's 1.83x: tiled ~2.6 ms per layer, **c1 ~75-85 ms at FUSE=1, ~70-80 ms at FUSE=4** (run 1: 110); whether longer DMA runs recover the gap (trn1 step 4's probe: one 7 KiB-run block per DMA 276 GB/s) is what the job's layer bench measures (v2set4 vs v2set).
- **The job (`nki_n8_7b_inf2.sh`, inf2.8xlarge, ≤ 106 min):** HBM probe; kernels (the tiled layer on NEURON vs the members, share of equal bits); layer bench at 7B shapes (v1, v2set4, v2set: device ms, bits vs v1); the choice (`nki_n8_7b_choose.py`: the fastest tiled variant bitwise == v1, else the members); FuseZap 7B in process FUSE=1 (greedy 12 x 32, top-1 on 6 prompts: run 1's OOM point; HBM lines) and FUSE=4 (--quick, same c1 tokens required); served with the faster (gate, Invarix perf + accuracy; HTTP 500 count, must be 0); NxDI TP=2 and, if 25 min are left, TP=1.
- **Compile risks [unverified]:** neuronx-cc's SBUF allocation of the tiled layer at 2048 columns and 1024 slots (estimate 176 of 192 KiB; 512 columns: 167); a tiled 7B layer is ~10K instructions (FUSE=4: ~40K, like Qwen2.5-0.5B at FUSE=24); four PSUM banks of matmul output live at once plus the transposes'; a one-iteration `for` loop per column block (the bf16 edits key on it).
- **Effort record (N7 log, I71):** backend lines (`device/neuron.py`, `device/nrt_worker.py`, `codegen/nki*.py`): device/neuron.py +53 -1, device/nrt_worker.py +21 -1, codegen/nki_kernels.py +169 -43, codegen/nki_bf16.py +13 -6, codegen/nki.py 0 (plus jit.py +3 -1, the arena order); model/serve: models/llama.py +6, serve/neuron_kv.py +51, serve/logprobs.py +13, serve/engine.py +3, serve/api.py +38. New exceptions: none (the tiled layer is LayerSpec with a flag; HBM handling is allocator / runtime-worker policy inside exception 4's untyped buffers). Agent wall time: ~1 h (Mac only).

## 11. Tensor parallelism TP=2 on one Inferentia2 (I72)

**Goal.** Qwen2.5-7B-Instruct bf16 across the two NeuronCores of inf2.8xlarge, AWS NxD Inference's shape (run 1, same
instance: AWS TP=2 40.7 ms/token c1, 82 tok/s c4; FuseZap one core 110 ms). Code: `fusezap/serve/neuron_tp.py`; job
`nki_n8_7b_tp_inf2.sh`; probe `scripts/nki_tp_probe.py`; tests `tests/test_neuron_tp.py`.

**The all-reduce: what exists (read from the wheels and the docs, 2026-10-28).** Two per layer, 56 per token, 3584 float32
per row.
- **NKI has collectives.** §10 said "nki 0.6 has no collective instruction": wrong. They are not in `nki.isa` but in
  `nki.collectives`, in the nki 0.5.0, 0.6.0 and 0.7.0 wheels [23]: `all_reduce(srcs, dsts, replica_group, op)`, all_gather,
  reduce_scatter, all_to_all(_v), collective_permute(_implicit(_reduce)), `ReplicaGroup([[0, 1]])`; op add / max / min;
  srcs and dsts all HBM or all SBUF (not PSUM); several tensors per call only on HBM. 0.7's docstring adds: at LNC 1 the
  sources must be `private_hbm`. The tracer (`_backends/mlir_tracer/isa.py`) emits a CollectiveCompute op and marks the
  kernel `has_collectives`; the baremetal pipeline keeps `sync_before_global_cc` for them (`compiler/_baremetal_driver.py`).
  The CPU simulator raises `NotImplementedError("Collective operations are not supported in the simulator backend")`
  (`_backends/simulator/__init__.py`), so NKISIM cannot run them. No check in the Python layer refuses NeuronCore-v2;
  whether neuronx-cc 2.27 compiles one for inf2 is **[unverified]** (the probe's first phase).
- **The runtime loads collective NEFFs.** nrtpy 2.35.30 [24]: `NrtpyModel.load_from_neff(path, name, core_id=0,
  cc_enabled=False, rank_id=0, world_size=1)` calls `Nrtpy.load_model(path, core_id, cc_enabled, rank_id, world_size)`, and
  `_nrtpy.so` imports `nrt_load_collectives` next to `nrt_load`. `allocate_tensor(size, core_id, name)` takes a core too. The
  disassembly shows `load_model`, `execute` and `allocate_tensor` run under nanobind's `gil_scoped_release`
  (call_guard), as do the tensor reads / writes: one Python process could drive both cores from two threads. The
  public `nrt.h` page documents `nrt_cc_comm_t` / `nrt_cc_op_type_t` but not `nrt_load_collectives` [25]. The ranks
  bootstrap through `NEURON_RT_ROOT_COMM_ID` (aws-neuronx-collectives release notes [26]); `NEURON_RT_EXEC_TIMEOUT` (default
  30 s [27]) bounds an execute, including one waiting for its peer.
- **NxD Inference** gets its collectives from XLA's all-reduce HLO compiled by neuronx-cc into its NEFFs (the same
  CollectiveCompute), one model loaded per rank with the collectives path. We do not need XLA for that: NKI emits the op.
- **Host paths.** Through the fusezap processes (`dist.HostComm`): each all-reduce flushes the step's batch (a read round
  trip, a socket exchange, a write): ~0.3-0.5 ms each, ~20-28 ms/token, which cancels TP's gain. Inside the runtime
  workers through shared memory: no fusezap process involved, the batch stays one round trip, the cost is a device read
  and write of 14 KiB plus a spin wait.

**Choice.** (1) `cc`: an NKI collective kernel (`nki_kernels.AllReduceSpec`, `fz_all_reduce_<kind>_w2_<n>`) launched like
any kernel, loaded per rank with `cc_enabled / rank_id / world_size`; it stays in the step's batch and the cores exchange
the partials themselves. Forms (`FZ_NEURON_TP_AR_KIND`): `phbm` (default: DMA to a private-HBM scratch, all-reduce
private to private, DMA back), `hbm` (straight on the arguments), `sbuf` (SBUF tiles). (2) Fallback `shm`: the runtime
workers' shared-memory all-reduce (`nrt_worker.ShmAR`, an "ar" request queued with the launches). (3) `host`: HostComm.
`FZ_NEURON_TP_AR=auto` (default) loads, then self-tests, cc then shm then host at launch on every rank and keeps the first that
passes; the job sets the probe's choice explicitly. `FZ_NEURON_TP_AR_FUSE=1` lets `fuse_schedule` put the collective inside
the fused layer kernel (one NEFF per layer with the collective; off by default). Not chosen: one process owning both cores
(possible per the nrtpy facts above, `xcore` in the probe checks a cross-core tensor binding) — it needs a two-device
fusezap process; the multi-process design reuses `dist.py` whole.

**Structure (mirrors Core TP).** One fusezap process per rank (`dist.launch(2, "NEURON")`), each with its own
runtime worker on its own core (`NEURON_RT_VISIBLE_CORES=r`; `rank_envs` also sets `NEURON_RT_ROOT_COMM_ID`, the rank /
world for collective loads, the shm file, `NEURON_RT_EXEC_TIMEOUT=600`). `Llama(tp=g)`: Megatron slices (`tp_slice`), bf16
K-major per shard (the relayout now runs after the slice), the KV pool with the rank's kv heads (7B: 14 q / 2 kv heads,
I 9472 per core), rank 0 mirrors every forward. A shard's layer runs as its member kernels fused between the two
all-reduces; LayerSpec (one kernel per layer) stays single-device, since both all-reduces sit inside a layer. The tiled
per-shard layer kernel (I71) plugs in as two halves around them (attention ... o_proj partial | add, norm, MLP ...
down partial). lm_head: replicated (default, as CUDA), or `FZ_NEURON_TP_HEAD=vocab`: V/2 rows per core and the logits
all-gathered by a zero-padded sum all-reduce (exact); the device argmax then runs on the full logits on both ranks.
`cached_neff` now serializes a key's compile across processes (a lock file), so the ranks leave a cold first step together.

**Mac results (NKISHIM, two real processes; the shim's `nki.collectives.all_reduce` calls the group's host sum).**
`tests/test_neuron_tp.py` 24 passed: AllReduceSpec in all three forms runs on the shim; ShmAR over two processes gives the
rank-order float32 sum with identical bits on both ranks over 42 mixed-size calls; the worker's "ar" request and a
collective NEFF load (cc kwargs) against the fake nrtpy; the allocator queues "ar" with the launches (one round trip).
TP=2 vs TP=1: a tiny Qwen2-style model's logits within 1e-4 and identical greedy tokens (prefill, decode, generate, and
LLMEngine over five prompts), each layer's two all-reduces recorded as AllReduceSpec launches; bf16 at 7B's head layout
(28 / 4 heads, head_dim 128 -> 14 / 2 per rank): logits within 2% and TP=1 bf16's engine tokens; host mode = cc mode bit for
bit; the vocab head within 1e-6 (BLAS width effect on the shim) with the same top-1. Qwen2.5-0.5B bf16 TP=2 in process
(scripts/nki_n6_serve.py --tp 2): greedy 6/6 vs the f32 reference, 97 launches per decode token (48 all-reduces + 49 fused).

**Predictions (not measurements).** Member kernels per shard: ~55-60 ms/token c1 (half of 110) + 56 all-reduces (cc: ~20-60
µs each, +1-3 ms; shm ~0.1 ms, +5-6 ms; host +17-28 ms). Still slower than AWS's 40.7 ms until the tiled per-shard layer
kernel lands (per-core weight-read floor ~17 ms/token at ~410 GB/s).

**Hardware risks [unverified].** neuronx-cc compiling `nki.collectives.all_reduce` for inf2 at all (and which form); the
runtime loading it per rank in separate processes (bootstrap through NEURON_RT_ROOT_COMM_ID); the collective's latency;
a rank that fails mid-collective leaves its peer in nrt execute until NEURON_RT_EXEC_TIMEOUT; fused kernels around
must-alias buffers as before.

**inf2.8xlarge run 1 (2026-10-28, stopped at 0.85 h).**
- Probe: AllReduceSpec `phbm` compiles (7.4 s) and is exact, 0.36 ms per execute at every size up to 1 MB (0.69 ms at
  16 MB); `hbm` is refused ("Collective instruction cannot read IO tensors"), `sbuf` too ("SB CC is only supported on
  trn2+"). shm: 0.04 ms at 14 KiB, 0.07 at 56 KiB, 1.7 ms at 1 MB, with the device read and write included (each "ar"
  request reads its device buffer and writes the sum back inside the batch). A host read round trip alone is 0.21 ms.
  xcore: NRT_INVALID (a NEFF on core 0 cannot take a tensor allocated on core 1). Reference small kernel: 0.11 ms.
- The chooser took "cc if it works" (20.5 ms of all-reduce per token at c1) over shm (~2.2 ms). Fixed: the probe and the
  launch's `auto` measure every working mode's cost per all-reduce inside a step and take the cheaper (shm here).
- **The TP=2 phase hung in its first step (root cause).** NeuronAllocator queues launches and sends them at the next
  load, large write or read. Rank 0 reads every step's result; a worker rank never reads. After a step's last NEFF load,
  the worker's remaining launches (the last layers' collectives among them) stayed in its queue, while rank 0's read sent
  its own and its runtime waited in the collective for a peer launch that never came (CCOM "Timeout waiting for RX",
  240 -> 1920 s). The Mac tests missed it: NKISHIM runs every launch at once. **Fix:** every top-level mirrored call ends
  with a flush on every rank (`TPGroup.end_op` = `NeuronComm.flush`). **Guards:** rank 0's collective count (queued
  collective launches and "ar" requests, `NeuronAllocator.cc_count`) travels with every message, and a worker whose
  count differs raises `TPDesync` before running the call; `NEURON_RT_EXEC_TIMEOUT` 120 s per rank; the job's TP phases
  have their own caps and a watchdog that kills them on a CCOM timeout, a desync, a shm timeout or 15 min without output.
  `tests/test_neuron_tp.py` reproduces the hang with two real runtime workers (fake nrtpy, the shm all-reduce): without
  the flush rank 0 times out, with it it gets the sum.

**The per-shard layer kernel (merge of I71's tiled layer).** `LayerSpec.part`: `attn` runs steps 1-6 (x1 = x + d, norm,
qkv, RoPE, KV write, attention, o_proj) and writes the residual x1 and o_proj's product (no bias); `mlp` reads x1 and the
all-reduced product, adds the bias and the residual, norm 2, gate_up, silu * up, down, and writes x2 and down's product.
The same instructions as the whole layer cut at o_proj's output (the whole layer's source is byte-identical to I71's: 288
variants checked), tiled or not, bf16 by the same mechanical edits (each half edits its own two weights). A shard runs
attn | all-reduce | mlp | all-reduce per layer (`NeuronKVCache.layer(all_reduce=)`, `FZ_NEURON_TP_LAYER=halves`, default;
`members` keeps the member kernels). NKISHIM (FZ_NKISHIM_MM=f64): TP=2 halves == TP=2 members bit for bit (float32 untiled
and tiled, bf16 at 7B's head layout tiled) and TP=1's greedy tokens; Qwen2.5-0.5B bf16 TP=2 in process: 97 launches per
decode token (48 halves + 48 all-reduces + head), greedy 6/6. I71's HBM policy applies per rank: each rank's allocator has
its own 15.25 GiB budget on its own core's HBM, and the worker's all-or-nothing batch allocation covers "ar" requests like
launches. Under TP an out-of-memory on one rank is fatal for the group (the peer would wait in an all-reduce): the scorer's
release-and-retry does not apply. The job benches the tiled layer at a shard's shapes (v1 / v2set4 / v2set), then A/Bs
halves vs members at TP=2 and keeps the faster with the members' tokens.

**The all-reduce on the device (I72 follow-up).** Run 2: TP=2 halves + shm served 58.3 ms/token c1, 62.4 tok/s c4 (AWS NxDI TP=2: 40.7 ms, 81.5 tok/s). Rank 0's
per-token profile: step 58.5 ms = **50.7 ms of nrt executes** (57 launches; the attn half 0.557 ms + the mlp half 1.031 ms per
layer, head 6.2) + **6.0 ms** in the worker's 56 shm all-reduces (0.107 ms each in the step, vs 0.046 alone) + 1.8 ms host.
The whole shard layer as one kernel is 1.043 ms of device time (layer bench). nrtpy's `execute` is synchronous (its wheel
imports `nrt_execute` only [24]), so each launch is device time + ~0.1 ms of runtime, and cutting the layer at its all-reduces
costs ~0.45 ms per layer (two launches; the MLP's weight stream cannot start under the attention half's serial tail): ~13 ms
per token, twice the exchanges themselves.
- **Options weighed.** (a) The collective inside the layer kernel: chosen (below). (b) Keep shm but stop draining: the
  mlp half reads the attn half's sum, and layer i+1 reads layer i's, so a launch can never be posted ahead of the exchange it
  depends on; double-buffering hides nothing on the critical path: **not pursued**. (c) Runtime peer exchange: `nisa.sendrecv`
  and `nisa.core_barrier` (device-side send/receive between the two cores sharing an HBM stack) are NeuronCore-v3+ with LNC=2
  only (nki 0.6 `isa/__init__.pyi` [23]); inf2 is v2. The async runtime API (`nrt_async.h`, SDK >= 2.29 [28]) has
  `nrta_execute_schedule` (non-blocking execute), `nrta_tensor_copy` ("between two tensors allocated on the same Logical Neuron
  Core") and `nrta_cc_prepare` / `nrta_cc_schedule` (host-scheduled collectives on an NRTA_XU_COLLECTIVES unit, "work-in-progress
  and subject to change"); ordering between the compute and collectives units is not documented, nrtpy 2.35 binds none of
  it (ctypes against libnrt would be needed), and a host-scheduled collective has the same per-collective fixed cost as the
  kernel's. `nrt_async_sendrecv_*` is a network API between servers' logical cores (peer IP) [29]; `nrt_tensor_copy` [25] is
  same-process; xcore (a core-1 tensor in a core-0 NEFF) failed NRT_INVALID in run 1. The implicit async mode
  (NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS) is gone since 2.31 [30]. Worth a later look: `nrta_execute_schedule` to post a
  step's launches without waiting (the ~0.1 ms per launch), independent of the all-reduce.
- **Built: `FZ_NEURON_TP_LAYER=fused`** (default off at first; the TP default since the inf2 run, below). `LayerSpec(part="tp", tpw=W)`: the whole layer's source plus two
  collectives in AllReduceSpec's phbm form (the one run 1 compiled and ran exactly on inf2): o_proj's product goes from its
  SBUF tile to a private-HBM scratch, `nki.collectives.all_reduce` into a second one, back into the tile before the bias
  and residual add; down's product the same way, the sum DMA'd straight to dn. The same instructions and sums as the halves
  (for two ranks a + b = b + a, so shm's rank-order sum gives the same bits). `neuron_kv.layer(tp=)` runs it when the group's
  mode is cc (`neuron_tp.fused_layer`; `setup` keeps cc when it passes, whatever a separate launch costs); bf16 by the same
  edits. It has no all-reduce launch, so `fuse_schedule` groups `FZ_NEURON_FUSE` layers (and with FZ_NEURON_HEAD_FUSE the
  head) into one NEFF: FUSE=28 + head is one launch per rank and token, NxD Inference's shape. Prefill (rows beyond
  LayerSpec) still runs the members with AllReduceSpec launches.
- **Probe** (`scripts/nki_tp_probe.py --only chain,layer`): `ARChainSpec` runs k = 1..56 collectives in one kernel (dep: serial,
  the layer's DMA-in / all_reduce / DMA-out pattern; indep: k of the same data) and fits execute = a per launch + b per
  collective, the open question: the run-1 numbers (0.39 ms at 14 KiB, 0.33-0.43 to 1 MiB, 0.40 at 4 MiB, 0.70 at 16 MiB)
  only say the cost is flat in size below ~4 MiB (~54 GB/s above), not whether the ~0.25 ms beyond a plain launch is per launch
  (the runtime's collective setup) or per instruction (each CollectiveCompute is preceded by a CoreBarrier, nki 0.6
  `compiler/_baremetal_driver.py` pass `sync_before_global_cc` [23], plus the cross-core handshake). Then the 7B shard layer
  (bf16, tiled, 2048 columns, 1 / 4 rows, 1024 slots) on random inputs: whole (no collectives), halves + two shm "ar", tp; bits
  tp vs halves; 4 tp layers in one FusedSpec NEFF vs 4 launches. `--dry` runs all of it on NKISHIM in process.
- **Mac results (NKISHIM).** tests/test_neuron_tp.py: TP=2 fused == halves bit for bit (float32 untiled FUSE=1, float32 tiled
  FUSE=24, bf16 at 7B's head layout tiled FUSE=24), a decode step one FusedSpec launch per rank with FUSE=24, no all-reduce
  launch left, TP=1's greedy tokens; ARChainSpec exact (2^(k-1) x sum); the probe's dry run: tp == halves + shm on every
  output, fused == single launches.
- **Predictions (not measurements).** c1 = 28 x (1.04 + 2b + ~0.1 launch at FUSE=1) + head 6.2 + host 1.8: b = 0.05 ms -> ~41 ms
  (FUSE=1), ~38 ms (FUSE=28 + head); b = 0.25 ms -> ~52 / ~49 ms. Either beats 58.3; parity with NxDI's 40.7 needs b <~ 0.1.
- **Hardware risks [unverified].** neuronx-cc compiling a 7B layer with collectives (private-HBM scratch inside a large
  kernel), and 28 of them in one NEFF (compile time, instruction memory); the CoreBarrier before each collective stalling
  every engine (the MLP's first weight DMAs wait instead of streaming during the exchange); nrt_load_collectives of a large
  fused NEFF per rank; FusedSpec's renamed private-HBM tensors per layer (56 x 2 x 56 KiB); a slow collective inside a
  fused NEFF stretching NEURON_RT_EXEC_TIMEOUT (120 s: fine). Job: `nki_n8_ar_inf2.sh` (probe, chain + layer probe,
  --quick A/B of halves+shm / fused FUSE=1 / 4 / 28+head, `nki_n8_ar_choose.py` keeps the fastest with the baseline's c1 tokens,
  then 24-prompt greedy + top-1, served Invarix, AWS NxDI TP=2).
- **inf2.8xlarge run.** Chain fit: 0.324 ms per launch + 0.0085 ms per
  chained collective (the fixed cost is per launch). Fused == halves bitwise; FUSE=1 / 4 / 28 token-identical; chosen fused,
  FUSE=28 + head fused: in process 38.8 ms/token c1, 103.9 tok/s c4, greedy 23/24, top-1 0.9957; served 38.37 ms, 93.9 tok/s,
  top-1 0.9987, greedy 0.967, PROMOTE. AWS NxDI TP=2 on the same instance: 40.6 ms, 81.7 tok/s.
- **Defaults since then (TP on NEURON only; TP=1 unchanged).** FZ_NEURON_TP_LAYER defaults to `fused`; `setup` (FZ_NEURON_TP_AR
  `auto`) self-tests shm and cc on every rank, keeps the cheaper one for the separate all-reduces (prefill's member path, the
  vocab head: shm on inf2) and puts the decode layers' all-reduces inside the layer kernels whenever cc loaded and passed
  (`NeuronComm.layer_cc`); cc failing falls back on its own to the halves around the separate mode. Under TP
  (`neuron.TP_WORLD`, set by `neuron_tp.install`, reset at close) FZ_NEURON_FUSE defaults to every layer and FZ_NEURON_FUSE_MAX
  to no cap (FZ_NEURON_HEAD_FUSE already defaults on): one NEFF per rank and decode step. Explicit FZ_NEURON_TP_LAYER /
  _TP_AR / _FUSE / _FUSE_MAX / _HEAD_FUSE still win (FZ_NEURON_TP_AR=shm or host: the halves). The I72 job
  (nki_n8_7b_tp_inf2.sh) pins FZ_NEURON_TP_LAYER=halves, FUSE=1, HEAD_FUSE=0, so run 2 stays reproducible.
  Untested on hardware: the default mixes shm (separate all-reduces) with cc (inside the layers) in one step; the run used cc
  for both.

**Time to first token (I75).** The I73 run served TTFT p99 575 ms at c4 (c1 146 ms; NxDI offline
c1 50.6 ms). Where it goes, read from the code and the run's request log:
- **Scheduling.** NEURON's prefill forward has 128 rows (`max_prefill_tokens`, one size), so a step prefills one 128-token
  prompt. A closed-loop c4 burst (four prompts within 1.5 ms) runs A alone, then B, C, D as three prefill-first steps
  (`FZ_PREFILL_FIRST=4`, decoders waiting): the served TTFTs were 147 / 290 / 432 / 576 ms, i.e. exactly 1-4 prefills. So the
  p99 is the burst's total prefill work queued FCFS, not a long prefill blocking others: any order or batching pays the sum
  (a 512-row forward costs ~4x a 128-row one here: attention and the elementwise kernels are linear in rows, render_matmul
  streams the weights once per 128-row block), and interleaving decode steps would only delay it. No scheduler change; the
  lever is the cost of one prefill.
- **One 128-row prefill, before (members).** Per rank: 4L+1 = 113 launches (2L+1 fused member groups split by 2L = 56
  separate all-reduces; with cc each is a NEFF launch of ~0.35-0.4 ms, ~20 ms; in the default mix each is an shm "ar" of
  1.8 MB inside the step, ~1.2 ms, ~65 ms: I72 run 2 TTFT 187 ms vs 143 with cc), 2 runtime-worker round trips (the
  1.8 MB embedding write is above FZ_NEURON_QWRITE_MAX, then one batch with the token read). Device time by
  `nki_layer_cost.py`'s trn1 fit at the shard's shapes (128 rows, 1024 slots): 3.3 ms per layer: attention 1.40 (exact-exp
  softmax over every pool slot: linear in the prefix), gate_up 0.62 and down 0.32 (weight-stream bound), silu*mul 0.42,
  RoPE 0.22, qkv + o_proj 0.17, norms + KV write 0.16; x28 = 93 ms (the run: ~120 ms besides the all-reduces).
- **Built.** `FZ_NEURON_TP_PREFILL=fused` (default; `members`: the old path): a prefill forward's member layers record
  their all-reduces as `AllReduceSpec(fuse=True)` (phbm, the form inside the decode layer kernels; same source and NEFF as a
  separate cc launch), and its fences carry `FZ_NEURON_TP_PREFILL_FUSE` (default 4 layers per NEFF; `all`: the chunk and the
  head in one NEFF per rank), so `fuse_schedule` puts the members and their collectives in one kernel per group: 7 launches
  per chunk, no separate all-reduce. The same instructions as the members with cc launches: bit for bit (NKISHIM, float32
  and bf16 at 7B's head layout, groups of 1 / 3 / all). Needs cc to pass the launch self-test (`NeuronComm.prefill_cc`), else
  the members around the separate mode (the launch line says so). `FZ_NEURON_PREFILL_CTX` (default: FZ_NEURON_CTX's value):
  the prefill forwards' pool prefixes on their own, so decode can stay at the whole pool (`FZ_NEURON_CTX=0`, one decode
  NEFF) while a prompt's attention scans 512 of 1024 slots (bitwise: 512-key blocks either way); `128,256,512` also smaller
  ones (narrower last key block: bits not guaranteed). `FZ_NEURON_TP_AR_BIG` (default 256 KiB; 0 = off): in shm mode a
  separate float32 all-reduce from that size on takes the cc launch when cc passed (shm grows ~1.6 ms per MiB, cc is flat
  ~0.35 ms), and above the 8 MiB shm slot always (cc, else the host): the I73 default mix sent a members prefill's 1.8 MB
  and the logprobs scorer's 7-15 MB trunk messages through shm, and a 1024-row scoring bucket (14.7 MB) would have been
  refused by ShmAR. `neuron_tp.plan_mix` states the all-reduce plan setup picks (inf2: shm for small separate all-reduces,
  cc launches for large ones, cc in the decode layer kernels and in the prefill's fused kernels), `NeuronComm.mix()` /
  the serve script's `tp.mix` record it. Not done: a dedicated prefill layer kernel (LayerSpec covers 128 / G attention
  rows: 18 tokens at 7B's G = 7; the members already tile 128-row blocks, and the prefill's cost is device work, not the
  members' HBM round trips); NKISIM cannot run collectives, so the checks are on NKISHIM.
- **Job** `nki_n8_prefill_inf2.sh` (~1.3 h): the default mix vs cc+cc with the old prefill (greedy vs the
  reference, decode, TTFT, a prefill kernel profile), the fused prefill (+ prefill prefix 512 / smaller / one NEFF per
  chunk), `nki_n8_prefill_choose.py` (lowest in-process c4 burst TTFT p99 with the baseline's tokens), the chosen path's
  greedy 24 x 64 + top-1, served old vs chosen (Invarix short c1,c4, PROMOTE bar); NxDI's numbers reused from I73.
  Predictions (model): c1 TTFT ~120 ms fused (from ~190 in the shm mix, 146 with cc), ~100 ms with the 512-slot prefix,
  c4 p99 ~4x that.
- **Run 2 and the follow-up (defaults, prewarm).** Run 2 picked `FZ_NEURON_TP_PREFILL_FUSE=all` + `FZ_NEURON_PREFILL_CTX=128,256,512`
  (in process c1 TTFT 87 ms, c4 burst p99 176 ms, greedy 23/24, top-1 0.9957; served c1 p99 93, c4 p99 370 ms); both are
  now the TP defaults (an unset `FZ_NEURON_PREFILL_CTX` under TP = 128,256,512 below the pool, then the pool; explicit env
  wins, `""` follows FZ_NEURON_CTX; TP=1 unchanged). A FUSE=all group whose NEFF fails to compile runs as groups of 4 layers
  (`FusedCall.subs`, `neuron.FUSE_FALLBACK`; before: its ~340 members one by one). Run 2's gap: warm-up captured only the
  128-slot prefill, so the first prompt at another prefix compiled (~197 s cold) or loaded (2.6 s warm) in its TTFT.
  `fusezap/serve/neuron_prewarm.py` (run by `fusezap serve`'s warm-up and `nki_n6_serve.py`; `FZ_NEURON_PREWARM=0` skips)
  now captures every decode bucket and prefill size at every pool prefix (the pool included) on a synthetic step whose
  block table ends at that prefix, compiles the fused NEFFs in parallel (`NEURONDevice.precompile`,
  `FZ_NEURON_PREWARM_JOBS` threads per rank, each its own nki worker, ranks starting at different keys under the per-key
  lock), then replays each once (loads); under TP the steps are mirrored forwards and the compile is one op "call" plus a
  barrier. HBM: each FUSE=all prefill prefix keeps ~0.85 GiB of planned intermediates (7B TP=2: four prefixes ~3.4 GiB,
  ~10.8 of the 15.25 GiB budget after prewarm, estimate); rank 0 skips a prefix that would leave less than 1 GiB, and loaded
  NEFF bytes past `FZ_NEURON_NEFF_RESERVE_MB` (256) come off the tensor budget. Estimates (not measured): cold start
  ~5-8 min of prewarm (the four ~204 s FUSE=all compiles run together), warm ~15-30 s. `scripts/nki_ttft_probe.py` checks a
  served engine's first request per prefix against its steady TTFT; `nki_n8_prefill_inf2.sh --confirm` runs the defaults
  cold, then served.

**Served TTFT at c4 (I76).** I75 confirm served Invarix short (128 in / 128 out) at c4 with TTFT
p50 234.6 / p99 375.5 ms against an in-process c4 burst p99 of 176 ms, which looked like ~200 ms of serving overhead. It is
not: the in-process burst used the reference prompts (37-66 tokens), three of which share one 128-row prefill forward
([88, 88, 88, 176] ms), while Invarix's prompts are exactly 128 token ids (`prompt_format: tokens`), one forward each. The
served requests show it: per c4 wave the first tokens land 89-90, 93-97 and 92-93 ms apart, i.e. one prefill per prompt at
the pool prefix its blocks reach (lowest-first blocks: 128, 256, 512, 512 slots; I75 in process: 87 ms at 128, ~95 at 512).
- **Model** (`serveq_sim.py`): the real LLMEngine and Scheduler (and with `--http` the real `fusezap serve` app,
  ProcEngine pipe and Invarix's closed-loop client; `--serve` for `fusezap invarix` itself) over a tiny CPU checkpoint, every
  forward a fixed latency from the I75 chip run (128-row prefill 87 / 89.5 / 94 / 103 ms at prefix 128 / 256 / 512 / 1024,
  decode 38.5 ms), prefill-first 4, lowest-first blocks. Virtual time: c1 TTFT 90.6, c4 TTFTs [90.6, 180.1, 274.1, 368.1],
  p50 224.6 / p99 367.6, TPOT p50 39.6, 97.4 tok/s; real time over HTTP on the Mac: c1 90.8, c4 p50 230.0 / p99 371.0,
  TPOT 40.03, 96.4 tok/s (chip: 90.6, 234.6 / 375.5, 39.75, 96.9). The request trace (FZ_REQ_TRACE) puts FuseZap's own
  serving path at ~1 ms per request (handler 0.14, pipe 0.2, engine add to step 0.04, output back ~0.6) plus ~2.5 ms of
  HTTP / client. **c4 TTFT p99 is the sum of the burst's four prefill forwards (364.5 ms of device time at their prefixes)
  + ~4-10 ms.** No order or batching lowers it: a 512-row forward streams the weights once per 128-row block
  (render_matmul), so it costs ~4 x 128 rows, and interleaving decode steps only adds to it (sim: FZ_PREFILL_FIRST 2 / 1 / 0
  give p99 415 / 415 / 487 ms). The levers are the prefill forward itself (87 ms vs NxDI's 50.6 ms c1 TTFT; prefix cost
  +7 ms at 512 slots) or weight reuse across row blocks in a wider prefill, both kernel work.
- **Built: the burst gate** (`scheduler.AdmitGate`, `FZ_ADMIT_WAIT_MS`, default 8 on NEURON, 0 elsewhere;
  `FZ_ADMIT_RECENT_MS` 250). A closed-loop wave's k replacements reach the engine a few ms apart; before, the first one's
  prefill started alone and the rest waited a whole forward, even when they would have fit in it. The serving loops
  (engine process and AsyncEngine) now wait up to 8 ms after the first waiting prompt while replacements are owed (k
  sequences finished in the last 250 ms, fewer than k requests added since), nothing decodes and the waiting prompts leave
  room in the 128-row forward and in max_num_seqs. Invarix short's 128-token prompts fill a forward alone, so they never
  wait: short c1 / c4 are unchanged (sim and HTTP sim identical). 40-token prompts at c4 (HTTP sim): TTFT p50 183 -> 98 ms,
  p99 185 -> 187, TPOT p50 38.98 -> 39.65 (the first three now wait for the fourth prompt's prefill), tok/s unchanged; c1
  unchanged. Through `fusezap invarix` itself against `serveq_sim.py --serve` (two runs each): short c4 p99 372 / 372 -> 379 /
  377 ms (noise), short40 c4 p50 135 / 97 -> 98 / 95 ms: without the gate a wave's short prompts share a forward only when
  they happen to reach the engine before its step starts; with it they always do. Scheduling only: the tokens are the engine's (tests/test_admit_gate.py).
- `scripts/nki_n6_serve.py --ttft-lens 0,128`: the in-process TTFT runs at the reference prompts and at 128-token prompts.
  `nki_n8_prefill_inf2.sh --serveq` (~30 min): in process (cold compile) TTFT at both lengths, then served
  old (FZ_ADMIT_WAIT_MS=0) vs new with Invarix profiles short and short40 (40 / 128) at c1 / c4 and request traces, the
  new one with accuracy (auto); `scripts/nki_n8_serveq_summary.py`: the table, the TTFT breakdown by burst position, the
  perf bar. Predictions: in process 128-token c4 burst ~[87, 177, 271, 365] ms; served short unchanged (~91 / ~230 / ~371,
  TPOT ~39.8-40.0); short40 c4 p50 ~-45%.

**Prefill kernels (I77; not chip-verified yet).** A 7B TP=2 128-row prefill chunk was one fused NEFF of
the 28 layers' member kernels, 81.6 ms device at the 128-slot prefix (I75), against NxDI's 50.6 ms c1 TTFT, and a
512-row chunk would cost ~4x (render_matmul streams the weights per 128-row block).
- **Where the time went** (measured: I75's separate-launch run at 1024 slots, MLP half 1.84 ms and attention half 1.60 ms
  per layer, head 6.5 ms; model: `scripts/nki_prefill_cost.py`, nki_layer_cost's trn1-fitted schedule on the members'
  NKISHIM traces at the shard's shapes, calibrated per half to those measurements, which then gives the old chunk 83.6 ms
  at 128 slots (measured 81.6) and 89.6 at 512 (~90)). Per layer at 128 rows / 128 slots (model, critical path): gate_up
  615 us (its 135.8 MB in 1,101 DMAs of (128, 512) bf16 tiles, 1 KiB runs), the renderer's silu*mul 410 us (VectorE: the
  exact exp's ~45 instructions per element on 128 x 9,472), attention 622 us (1,114 DMAs: q, lo, hi and the output one
  token at a time; the softmax exp on VectorE), down 312, RoPE 213 (238 DMAs), qkv 83, o_proj 64, norms 110, KV write 6.
  The weight floor: 7.1 GiB per rank at the decode NEFF's ~207 GB/s is ~37 ms per chunk.
- **Built** (`fusezap/codegen/nki_prefill.py`, `fusezap/serve/neuron_prefill.py`, FZ_NEURON_PREFILL_KERNEL=v2; default v1
  until a chip run): a layer as PFQKV (norm + qkv + bias + RoPE in the eviction), KVWriteSpec, PFAttn (rows = one q head's
  tokens: q by one DMA + a transpose, the pool prefix's rows and kᵀ in SBUF once, each key block's mask once per row block,
  ~34 DMAs per layer instead of ~1,070), PFLinear (o_proj) and PFMLP (norm, gate_up, silu*mul on the rows, aᵀ by transposes
  straight into down's bf16 stationary tiles; no (N, 2I) / (N, I) HBM rows), the same all-reduces between them; weight
  DMAs of up to 4096 columns (8 KiB runs; PSUM-bounded: 4096 / 2048 / 1024 columns at 128 / 256 / 512 rows), each weight
  tile feeding every 128-row block of the chunk; the prefill's lm_head split over the ranks (each streams half the
  vocabulary of its replicated K-major head, one zero-padded all-reduce gathers the logits: exact). FZ_NEURON_PFK_EXP=act
  (default) puts attention's and silu's exp on ScalarE (not bitwise, ~1e-5 relative; within the bf16 bar), exact keeps
  nki.exact_exp (the members' bits: tests/test_nki_prefill.py checks v2 exact == the members bit for bit on NKISHIM, one
  device and TP=2, with FZ_NKISHIM_MM=f64). FZ_NEURON_PREFILL_SIZES=128,512 (+ --max-prefill-tokens 512): a second prefill
  forward of 512 rows (prewarm skips its prefixes only a 128-row step reaches), grouped 7 layers per NEFF
  (FZ_NEURON_PFK_FUSE_BIG). A device self-test (`neuron_prefill.gate`, in the engine prewarm, every TP rank) compiles and
  runs the four kernels at the model's shapes against float64 references before anything is captured, falling back to
  VectorE evictions, then the exact exp, then the members.
- **Model, not measured** (same calibration): 128-row chunk 42 / 43 / 47 ms device at 128 / 512 / 1024 slots with act (exact
  54 / 60 / 72); 512-row chunk 94 / 106 ms at 512 / 1024 slots (~24 ms per 128 tokens; exact 168 / 214: VectorE-bound by
  the exact exp). HBM: a 128-row forward's planned intermediates ~0.37 GB per prefix (members 0.85), a 512-row one ~0.4 GB
  in 7-layer groups. NEFFs: ~5.1k instructions per layer at 128 rows (act 3.3k; members 8.7k), ~14k at 512 rows (act).
- `nki_n8_prefill_inf2.sh --pfk` (~66 min): v2 in process (prefill forward bench at each prefix, TTFT at
  40 / 128 / 512 tokens, greedy, decode), the old path, a single-core layer bench + `neuron-profile` of the layer NEFFs
  (`scripts/nki_pfk_bench.py`), served v2 with and without the I76 gate (Invarix short / short40, accuracy), v2 exact
  (its logits hashes must equal the old path's); `scripts/nki_n8_pfk_summary.py` states the bar.
- Run 1 (30 Oct): 128-row forward 85.6 -> 42.6 ms (act) / 49.5 (exact,
  bitwise); a 512-token prompt in one 512-row forward 130 ms (act); served short c1 TTFT p99 50 ms. Two gaps: a 200-token
  served prompt took 167 ms (it ran one 512-row forward, not two 128-row ones), and the single-core bench / neuron-profile
  phase failed (no self-test fallback in the bench; SDK 2.33 removed neuron-profile for neuron-explorer).
- Follow-up: with several FZ_NEURON_PREFILL_SIZES the scheduler covers the pending prompt tokens by the cheapest sequence
  of forwards (`Scheduler._fit_cost`; per-size ms from `neuron_prefill.prefill_cost_ms`: exact 128:50 / 512:172 (I78: 53 / 195), act
  128:43 / 512:130, FZ_NEURON_PREFILL_COST_MS overrides, FZ_PREFILL_FIT=0 turns it off): 200 tokens -> 2 x 128 rows, 400+
  -> 512. `scripts/nki_pfk_bench.py` runs the serving path's self-test and applies its fallback knobs.
  `nki_n8_prefill_inf2.sh --pfk2` (~50 min, time budget 60 min): main's defaults vs 128,512 by cost, served
  (Invarix short / short40 / p200 at c1, c4, p512 at c1, accuracy), the in-process chunk check (the last prompt token's
  logits bits: 128-row chunks == one 512-row forward == the members), the fixed bench + neuron-explorer;
  `scripts/nki_n8_pfk2_summary.py` recommends whether 128,512 becomes the default.
- I78 (1 Nov, `--pfk2`; 0.83 h): served, main's defaults (exact,
  128 rows) c1 short TTFT p50 / p99 55.1 / 56.3 ms (AWS NxDI p50 50.6), TPOT 38.5 ms, c4 99.7 tok/s; 128,512 by cost: p512
  c1 TTFT 229 -> 201 ms, p200 c4 p99 501 -> 465, but c4 short p50 145 -> 204 (four 128-token prompts bundled into one
  512-row forward wait ~195 ms each). Logits bits equal across 128 / 512 / v1 at 200, 400, 512 tokens; Invarix PROMOTE.
  Measured: 128 rows 51-69 ms, 512 rows 174-241 by prefix. So: `_fit_cost` now sizes by the oldest pending prompt
  alone (a burst of short prompts runs 128 rows a step; a long prompt's 512-row forward still takes others in its spare
  rows), the exact table is 128:53 / 512:195, and FZ_NEURON_PREFILL_SIZES defaults to 128,512 on NEURON (the
  largest, 512, is max_prefill_tokens unless given; FZ_NEURON_PREFILL_SIZES=128 for one size; NKISHIM / NKISIM: one
  size unless set). Not yet served on the chip with the per-prompt rule.

**N7 effort record (this model, TP).**
- Backend lines (`device/neuron.py`, `codegen/nki.py`): neuron.py +21 / -4 (collective load flag, "ar" queued like a launch,
  the all-reduce kept out of fusion unless asked, the cross-process compile lock); nki.py 0. Other device files: the
  runtime worker +72 / -2 (ShmAR, "ar", cc load), the shim +23 / -2 (`nki.collectives`). Kernels (`nki_kernels.py`) +68;
  serve / model: `neuron_tp.py` +258 (new), `dist.py` +15 / -5, `llama.py` +23 / -7.
- **New exception 10: a collective kernel is bound to a process group.** The all-reduce is a NEFF whose load depends on
  the process's rank and world (nrt_load_collectives), not only on its source and buffer sizes, and it pairs up only if
  every rank launches it in the same order: one runtime worker per NeuronCore, one fusezap process per rank, the SPMD
  mirroring of `dist.py`. The 4-part interface has no notion of a group: the rank comes from the environment
  (FZ_NEURON_CC_RANK / _WORLD), and the shim needs a process-level hook (`nkishim.COLLECTIVE`) for the peer. The fallback
  all-reduce (shm) is a runtime-worker request, not a kernel.
- Agent wall time: ~1.5 h (Mac only).

## 12. The second model on Inferentia2: Llama-3.1-8B-Instruct

**Mac prep (2026-11-04, from main 27e3f04; hardware job `nki_llama8b_inf2.sh` pending).**
Llama-3.1-8B differs from Qwen2.5-7B in: no q/k/v biases; llama3 RoPE frequency scaling (factor 8, original length 8192,
theta 5e5); 32 / 8 heads (group 4; 16 / 4 per TP=2 rank) vs 28 / 4; D 4096, I 14336, 32 layers vs 3584 / 18944 / 28; a
128,256-token vocabulary (lm_head 1.05 GB bf16) vs 152,064 (1.09 GB); untied lm_head (both); three EOS ids (128001
`<|end_of_text|>`, 128008 `<|eom_id|>`, 128009 `<|eot_id|>`); its own chat template (jinja, a date preamble).
- **Already handled before this pass (0 lines):** the config parser (`rope_scaling` and transformers 5's `rope_parameters`),
  llama3 `rope_inv_freq`, the RoPE tables built on the host (so every NEURON kernel takes scaled RoPE unchanged), optional
  q/k/v / o_proj biases everywhere (LayerSpec, PFQKV, PFMLP, the members), eos ids from config.json + generation_config.json
  into the engine's stop set, the jinja chat template, K-major bf16 weights, the head's vocabulary on 128 partitions.
  The real model runs on METAL bf16 from the HF cache (coherent text, stops on 128009).
- **Two NEURON gaps found on the Mac, both shape limits, not model code** (NKISHIM and the on-chip estimators):
  1. The TP=2 shard's tiled decode layer at the 1024-slot pool needs 180,608 B of SBUF per partition at
     FZ_NEURON_LAYER_WCOLS=2048 (budget 180,224; Qwen's shard 133 KiB: Llama keeps 4 kv heads per rank in the pool rows, Qwen 2): it
     would have run the members (slower: ~1.2x per layer by I71's cost model). `NeuronKVCache.layer` now steps the width down (1024, 512: same
     bits) before falling back (+2 lines).
  2. The 128-row prefill MLP (PFMLP) put down's 4096 columns in all 8 PSUM banks while the last silu block's transposes were
     live (onchip_peak 16,896 B > 16 KiB). Down now takes 7 banks when one row block would take all 8 (+9 / -2 lines;
     Qwen2.5-7B's kernels unchanged: 3584 columns never reach 8 banks).
  Also found, not fixed (not on the served path): on one NeuronCore (TP=1) the 8B layer fits SBUF neither untiled nor tiled
  (32 query heads x 4 rows, 8 kv heads in the pool rows) and HeadBF16 rejects K 4096 (32 K chunks x 2048 columns > 128 KiB);
  8 kv heads never stack (8 x 32 rows > 128 partitions: heads "each"). TP=2 is the 8B's layout, as for Qwen2.5-7B.
- **HBM per rank (TP=2, bf16; budget FZ_NEURON_HBM_GB 15.25 GiB):** layers 6.50 GiB + lm_head (replicated) 0.98 + KV pool
  0.13 (1024 slots float32) + planned forwards 0.20 after warm-up / 1.04 after a 300-token scoring request
  (`scripts/nki_hbm_account.py --shapes l8b-tp2`, NKISHIM, extrapolated from 1 and 2 layers) + free list <= 0.25 = **~8.9 GiB
  (~9.3 with 512-row prefill forwards)**; Qwen2.5-7B's weights are 7.09 GiB per rank against Llama's 7.48. The embedding table stays on the host (1.05 GB bf16).
- **Tests (Mac):** `tests/test_llama3.py` (18 + the job script check): a random-weight checkpoint from the published config
  matches HF float32 greedy on CPU for 32 tokens (full vocabulary; and with all RoPE bands scaled); llama3 tables vs HF to
  32k positions; the engine stops on each EOS id; the real tokenizer's template == HF's ids; NKISHIM at the 8B head layout
  without biases (engine tokens == CPU float32; bf16 layer kernel == bf16 members; LayerSpec == members bitwise, tiled /
  stacked / each). Parametrized onto the existing NEURON tests: prefill kernels (float64 references, one device and TP=2
  bitwise vs the members, on-chip budget at the shard shapes), TP=2 bf16 tokens and the fused layer halves, the
  128,256 / 64,128 argmax and bf16 head. NKISIM (the real nki simulator) needs Linux x86_64: not run here.
- **The job** (`nki_llama8b_inf2.sh`, ~1.1 h wall): setup (16 GB download through the runner's
  --hf-token), the tokenizer / RoPE checks on the real files, the float32 reference on the instance's CPU (the Mac's 48 GB
  swapped on the float32 8B), FuseZap in process at TP=2 with a cold NEFF cache (the compile time a new model pays; greedy
  24 x 64 vs the reference token for token, top-1 on 8), then `fusezap serve --tp 2` with Invarix (gate, short / short40
  at c1 and c4, accuracy). Same NEURON switches as the Qwen2.5-7B runs, no Llama-specific flag.
- **Llama-3.1-8B record (Mac part):** `fusezap/` +11 / -2 lines in 2 files (`serve/neuron_kv.py` +2, `codegen/nki_prefill.py` +9 / -2),
  both shape fixes that apply to any model with these sizes; 0 lines in `device/` (backend), `codegen/nki.py`, `models/`,
  the tokenizer, engine or API. New special cases: none keyed on the model (no `model_type` branch, no Llama flag in the
  job). Agent wall time ~1.4 h (Mac, no hardware; ~0.4 h of it test runs); the hardware time comes from the job.

## Sources
1. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html
2. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium_inferentia2_arch.html
3. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrtpy/index.html ; …/nrtpy/tutorial-debug-neff-nrtpy.html
4. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/index.html
5. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html ; https://pip.repos.neuron.amazonaws.com/nki/
6. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.jit.html
7. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.simulate_kernel.html
8. https://github.com/aws-neuron/nki-samples
9. https://github.com/aws-neuron/nki-library
10. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/index.html ; …/nki/api/nki.isa.html
11. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/vllm/index.html
13. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/vllm-neuron.html ; https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/multiframework-dlami.html
14. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/announcements/neuron2.x/announce-maintenance-pytorch-xla-inference.html
15. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/announcements/neuron2.x/announce-nxdi-maintenance-support-2-31.html
16. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/announcements/neuron2.x/announce-eos-nxdi-nxd-core-inference.html
17. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/announcements/neuron2.x/announce-intent-eos-tnx.html
18. https://github.com/vllm-project/vllm-neuron/blob/0.5.3/README.md ; https://awsdocs-neuron.readthedocs-hosted.com/en/latest/inference-legacy/nxd-inference/developer_guides/vllm-user-guide-v1.html
19. https://github.com/aws-neuron/deep-learning-containers/blob/main/vllm/inference/0.16.0/Dockerfile.neuronx
20. https://github.com/vllm-project/vllm-neuron/issues/57 ; https://github.com/vllm-project/vllm-neuron/issues/29
12. https://docs.runpod.io/api-reference/pods/POST/pods
21. https://aws.amazon.com/blogs/machine-learning/achieve-high-performance-with-lowest-cost-for-generative-ai-inference-using-aws-inferentia2-and-aws-trainium-on-amazon-sagemaker/ ("Each NeuronCore has a dedicated high bandwidth memory (HBM) of 16 GB"; read 2026-10-27)
22. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/inferentia2.html ("32GiB of high-bandwidth device memory (HBM)", two NeuronCore-v2 per chip; no per-core figure)
23. https://pip.repos.neuron.amazonaws.com/nki/ (nki-0.5.0, 0.6.0, 0.7.0 cp312 wheels: nki/collectives/_ops.py, __init__.pyi; nki/_backends/simulator/__init__.py; nki/compiler/_baremetal_driver.py; read 2026-10-28)
24. https://pip.repos.neuron.amazonaws.com/nrtpy/ (nrtpy-2.35.30 cp312 wheel: nrtpy/nrtpy_model.py load_from_neff, _nrtpy.cpython-312-x86_64-linux-gnu.so symbols and disassembly; read 2026-10-28)
25. https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/_sources/neuron-runtime/api/nrt.rst
26. https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/release-notes/runtime/aws-neuronx-collectives/index.html (NEURON_RT_ROOT_COMM_ID validation)
27. https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.0/_sources/neuron-runtime/nrt-configurable-parameters.rst (NEURON_RT_EXEC_TIMEOUT, default 30 s; NEURON_RT_VISIBLE_CORES)
28. https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/_sources/neuron-runtime/api/nrt_async.rst (nrta_execute_schedule, nrta_tensor_copy, nrta_cc_prepare / nrta_cc_schedule "work-in-progress", nrta_xu_t with NRTA_XU_COLLECTIVES; read 2026-10-28)
29. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/api/nrt_async_sendrecv.html ("Network communication between logical neuron cores", peer IP; early release; read 2026-10-28)
30. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/neuron-runtime/guides/how-to-migrate-async-apis.rst and …/about-neuron/announcements/neuron2.x/announce-no-support-implicit-async-mode.html (implicit async removed in 2.31; read 2026-10-28)
