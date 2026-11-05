# FuseZap vs vLLM on an H100: serving benchmark (P4-BENCH)

## Latest: speed run 3 final measurement (2026-10-22)

**Verdict: the SPEC §0 throughput gate PASSES, and the 80% stretch goal is met at both gate points on this pod.** The
comparison is VALID, and `--compare` printed no warnings:

| gate point | FuseZap out tok/s (3 reps) | vLLM 0.32.0 out tok/s (3 reps) | FuseZap / vLLM | needed | verdict |
|---|---:|---:|---:|---:|---|
| `gate/c1` (batch 1) | **135.4** (135.4 / 135.4 / 135.1) | 164.6 (164.6 / 165.1 / 164.6) | **82.2%** | ≥ 50% | PASS (stretch met) |
| `gate/c32` (batch 32) | **2620.2** (2622.9 / 2620.2 / 2606.5) | 3252.6 (3263.8 / 3252.6 / 3250.4) | **80.6%** | ≥ 50% | PASS (stretch met) |

Both engines ran on one pod, back to back, with vLLM first. No request failed or stopped short on
either engine. The build was clean (`dirty_paths` empty). The concurrency sweep on the same pod ran the gate points a
second time and gave 81.9% / **77.0%** (FuseZap 134.9 / 2568.2 tok/s, vLLM 164.7 / 3334.3). So c32 lands on both sides
of 80% on one pod: read the stretch result at c32 as "about 80%", not as a clear pass.

**Method: unchanged.** Speed run 3 did not touch docs/BENCH_METHOD.md, bench_vs_vllm.py or workloads.py.
The job ran the same commands as run 2's final measurement (gate pair, `--compare`, fp8 pair, concurrency sweep with
`--max-model-len 2816` and no rate points, then the CUDA test files). FuseZap's servers used the engine defaults:
`--max-num-seqs 1` for c1, and 32 rows with `--num-blocks 1600 --block-size 16` for c32. The engine runs in its own
process on CUDA by default since 0e62f94, so the harness did not pass `--engine-process`. vLLM ran at its defaults with
`VLLM_USE_FLASHINFER_SAMPLER=0`. No BEAM and no tuning flags were used.

**Setup.**
- Hardware: NVIDIA H100 80GB HBM3 (SXM), driver 595.91.07 (CUDA 13.2), RunPod secure cloud at $3.99/h.
- FuseZap: commit `c7b891b` (origin/main HEAD at the time of the run), clean tree.
- vLLM: 0.32.0 in its own venv.
- Server start-up: vLLM was ready in 117 s. FuseZap's servers were ready in 14 s (1 row) and 11 s (32 rows).

**What changed in the engine since the previous Latest (a761b7e)** (each was measured as its own A/B):

| commit | step | same-pod A/B vs HEAD (gate/c32) |
|---|---|---|
| e59c84d | swap-AB wgmma decode GEMM on sm_90: the weights are the 64-row wgmma operand, the batch rows are n, with split-K | 1672.5 → 1954.6 tok/s (+16.9%) |
| 0e62f94 | `fusezap serve` runs the engine in its own spawned process on CUDA, so HTTP and the engine no longer share one GIL | 2182.9 → 2485.0 (+13.8%) |
| 43eda37 | sm_90 GEMM config rules refit from one H100 sweep: the swap-AB split stays within one wave, and wg_pick uses a wave-cost model | 2382.5 → 2417.5 (+1.5%) |
| 79cc1e2 | SiLU*mul and qkv bias + RoPE as single bit-identical CUDA kernels in the prefill forwards | 2482.2 → 2524.1 (+1.7%) |
| c7b891b | review fix: when the engine process dies, `/health` and new requests return 503 and the server exits with code 1 | speed unchanged |

**Latency (pooled over all 3 repetitions; ms):**

| point | engine | TTFT p50 / p90 / p99 | TPOT p50 / p99 | ITL p50 / p99 | E2E p50 / p90 / p99 |
|---|---|---|---|---|---|
| gate/c1 | FuseZap | 24.3 / 25.1 / 26.1 | 7.2 / 7.3 | 7.2 / 7.8 | 942 / 945 / 950 |
| gate/c1 | vLLM | 30.5 / 33.3 / 40.5 | 5.8 / 5.9 | 5.8 / 6.8 | 773 / 775 / 781 |
| gate/c32 | FuseZap | 242.0 / 465.1 / 529.3 | 10.1 / 11.6 | 8.3 / 60.5 | 1527 / 1685 / 1848 |
| gate/c32 | vLLM | 353.6 / 432.2 / 445.2 | 7.1 / 8.9 | 6.3 / 15.0 | 1241 / 1270 / 1275 |

Compared with run 2: TPOT p50 at c32 went from 15.7 to 10.1 ms (vLLM 7.1), and TTFT p99 at c32 from 0.66 to 0.53 s
(vLLM 0.45 s). FuseZap's TTFT median is now below vLLM's at both points. The ITL tail is still 60 ms at c32 against
vLLM's 15 ms.

**Power and cost (not part of the gate):** GPU 0 board power over the measured repetitions, at $3.99/h for the whole pod:

| point | power W (FuseZap / vLLM) | J per output token (FuseZap / vLLM) | output tokens per $ (FuseZap / vLLM) |
|---|---|---|---|
| gate/c1 | 391 / 443 | 2.89 / 2.69 | 122,176 / 148,553 |
| gate/c32 | 536 / 557 | 0.205 / 0.171 | 2,364,048 / 2,934,692 |

**Concurrency sweep (same pod, both engines, `--max-model-len 2816`, no rate points).** `--compare`: VALID, `GATE: PASS`.

| point | FuseZap tok/s | vLLM tok/s | FuseZap % | run 2 % | TTFT p50 ms (FZ / vLLM) | TPOT p50 ms (FZ / vLLM) | J/token (FZ / vLLM) |
|---|---:|---:|---:|---:|---|---|---|
| short/c1 | 142.1 | 165.4 | 86% | 73% | 16.9 / 28.3 | 6.96 / 5.86 | 2.80 / 2.64 |
| short/c8 | 991.2 | 1204.6 | 82% | 50% | 62.7 / 74.9 | 7.60 / 6.03 | 0.425 / 0.364 |
| short/c32 | 3474.1 | 4302.4 | 81% | 45% | 116.9 / 152.0 | 8.32 / 6.20 | 0.139 / 0.116 |
| gate/c1 | 134.9 | 164.7 | 82% | 70% | 23.9 / 28.9 | 7.25 / 5.86 | 2.88 / 2.71 |
| gate/c8 | 877.5 | 1131.1 | 78% | 52% | 99.2 / 119.1 | 8.35 / 6.10 | 0.522 / 0.420 |
| gate/c32 | 2568.2 | 3334.3 | 77% | 53% | 280.2 / 274.4 | 9.90 / 7.35 | 0.209 / 0.172 |
| long/c1 | 131.3 | 162.8 | 81% | 70% | 77.1 / 60.3 | 7.29 / 5.89 | 3.03 / 2.80 |
| long/c8 | 769.2 | 1024.1 | 75% | 52% | 332.6 / 316.2 | 9.02 / 6.45 | 0.641 / 0.507 |
| long/c32 | 1682.6 | 2378.5 | 71% | 50% | 423.1 / 1000.0 | 16.76 / 9.52 | 0.332 / 0.251 |
| sharegpt/c1 | 139.3 | 170.2 | 82% | 71% | 17.4 / 29.0 | 7.17 / 5.76 | 2.85 / 2.66 |
| sharegpt/c8 | 718.4 | 952.5 | 75% | 49% | 41.9 / 78.2 | 8.07 / 5.98 | 0.606 / 0.478 |
| sharegpt/c32 | 2321.0 | 3129.5 | 74% | 44% | 83.1 / 117.8 | 9.57 / 6.64 | 0.207 / 0.164 |

Every sweep point is now 71% to 86% of vLLM (run 2: 44% to 73%). The decode-heavy c32 points gained the most
(short/c32 45% → 81%, sharegpt/c32 44% → 74%), because their gap was the decode GEMM and the per-token server path.

**FP8 (W8A8, both engines `--quant fp8`; outside the bf16 gate):**

| point | FuseZap tok/s | vLLM tok/s | ratio | previous Latest |
|---|---:|---:|---:|---:|
| gate/c1 | 155.8 (TPOT 6.25 ms) | 238.3 (TPOT 4.01 ms) | 65.4% | 59.1% |
| gate/c32 | 2508.1 (TPOT 10.46 ms) | 4264.8 (TPOT 5.40 ms) | 58.8% | 47.2% |

**32B (Qwen2.5-32B-Instruct, one H100, measured earlier in run 3 at 2745c5c / 9170ff2; not rerun here).** bf16:
gate/c1 31.4 vs 40.1 tok/s = 78.3%, gate/c32 476.1 vs 805.3 = 59.1%. fp8: 68.0% / 52.0%. Each pair ran on one pod,
`--compare` VALID; bf16 top-1 vs HF f32 is 0.9941. These numbers predate steps 1-4 of run 3 (the swap-AB GEMM and the
engine process), so they understate today's build. The "32B" section below has the details.

**Honest gaps.**
- **The c32 stretch result is borderline.** 80.6% in the gate pair and 77.0% in the sweep's rerun of the same point, on
  the same pod, later in the same job. FuseZap's own repetitions are tight (2606.5 to 2622.9); the difference is
  mostly vLLM (3252.6 vs 3334.3). H100 pods also differ by about 5%. The 50% gate itself has a wide margin now.
- **FP8 is slower than bf16 at c32.** FuseZap fp8 gives 2508.1 tok/s at c32, against 2620.2 for bf16 on the same pod.
  Run 3's decode work (the swap-AB wgmma kernel) covers the 16-bit GEMMs only, so the fp8 path did not get it. At c1,
  fp8 is still 1.15× bf16. FuseZap fp8 is at 58.8% of vLLM fp8 at c32.
- **Decode is still the gap.** TPOT p50 at c32 is 10.1 ms against vLLM's 7.1 ms; at c1 it is 7.2 vs 5.8 ms.
- **ITL tail.** ITL p99 at c32 is 60 ms against vLLM's 15 ms. A decode step that shares the GPU with a 2048-token
  prefill step waits for it, and FuseZap does not yet split big prefills into mixed steps. long/c32 has the slowest
  TPOT (16.8 vs 9.5 ms) for the same reason.
- **TTFT p99 at c32** is 0.53 s against vLLM's 0.45 s, although FuseZap's p50 is lower (242 vs 354 ms).
- **Not rerun this time:** the request-rate sweep, the 30-minute stability run, Llama-3.1-8B (it needs an `HF_TOKEN`) and
  32B on this build. The engine-process mode (0e62f94) has not had a 30-minute stability run yet.
- **Tokens per $** uses the whole pod at RunPod's on-demand price ($3.99/h). It is not a hosted-service price.
- Raw JSON, markdown and server logs are under `~/.cache/fusezap/remote/results/20261022-033241-ab-tmp-final-sh/` on
  the machine that ran the job. They are not committed.

## History: speed run 2 final measurement (2026-10-20)

**Verdict: the SPEC §0 throughput gate PASSES on this pod.** The comparison is VALID. FuseZap is at least 50% of
vLLM at batch 1 and at batch 32:

| gate point | FuseZap out tok/s (3 reps) | vLLM 0.32.0 out tok/s (3 reps) | FuseZap / vLLM | needed | verdict |
|---|---:|---:|---:|---:|---|
| `gate/c1` (batch 1) | **114.8** (114.2 / 114.8 / 117.1) | 164.8 (164.7 / 164.9 / 164.8) | **69.6%** | ≥ 50% | PASS |
| `gate/c32` (batch 32) | **1750.3** (1768.6 / 1750.3 / 1726.6) | 3314.1 (3326.1 / 3294.9 / 3314.1) | **52.8%** | ≥ 50% | PASS |

`bench_vs_vllm.py --compare` printed `GATE: PASS` (exit 0). Both engines ran on one pod, back to
back, vLLM first. Neither engine had a failed or short request. The compare printed one warning: the build had a
docs-only dirty path (a `.DS_Store` file, which rsync does not copy). The margin at batch 32 is small: 2.8 points, about
93 tok/s. The concurrency sweep on the same pod ran the gate points again and gave the same c32 ratio, 52.8%
(1734.1 vs 3287.3 tok/s), with c1 at 70.5%. The stretch goal (80%) is not met at either point.

**Method: unchanged.** docs/BENCH_METHOD.md, bench_vs_vllm.py and workloads.py were not touched in speed run
2. Gate workload: 512 in / 128 out, greedy, `ignore_eos`, 3 repetitions plus a warm-up wave, unique prompts. FuseZap's
batch-1 point ran on a `--max-num-seqs 1` server and its batch-32 point on a 32-row server (`--num-blocks 1600
--block-size 16`). vLLM ran at its defaults with `VLLM_USE_FLASHINFER_SAMPLER=0`. No BEAM and no tuning flags were
used. FuseZap's prefill budget is the engine default (2048 tokens per step on this H100), not a bench flag.

**Setup.**
- Hardware: NVIDIA H100 80GB HBM3 (SXM), driver 595.91.07 (CUDA 13.2), Xeon Platinum 8468 host, RunPod secure cloud at
  $3.99/h.
- FuseZap: commit `a761b7e`, with the fusezap/ tree unmodified.
- vLLM: 0.32.0 in its own venv.
- Server start-up: vLLM was ready in 96 s. FuseZap's servers were ready in 19 s (1 row) and 15 s (32 rows), load plus
  warm-up, all four prefill sizes included.

**What changed in the engine since the previous Latest (861ffb3)** (each was measured as its own A/B):

| commit | step |
|---|---|
| bc1144f | SPEED-R2-1: CUDA prefill budget of 2048 tokens per step, with prefill forwards of 256 / 512 / 1024 / 2048 rows; each step uses the smallest one that holds its prompt rows. Same-pod A/B: gate/c32 1539.8 → 1872.7 tok/s, c1 unchanged |
| c235e28 | SPEED-R2-2: review fixes. Planned CUDA prefill jits share intermediates (`Jit(plan=True)`), the default budget follows free memory (24 GB cards get 1024, the H100 keeps 2048), and all prefill sizes share one split-K partition count, so logits do not depend on the batch. Speed unchanged in its A/B |

**Latency (pooled over all 3 repetitions; ms):**

| point | engine | TTFT p50 / p90 / p99 | TPOT p50 / p99 | ITL p50 / p99 | E2E p50 / p90 / p99 |
|---|---|---|---|---|---|
| gate/c1 | FuseZap | 27.5 / 29.2 / 49.1 | 8.48 / 8.92 | 8.45 / 9.50 | 1103 / 1128 / 1161 |
| gate/c1 | vLLM | 22.7 / 24.2 / 27.0 | 5.92 / 5.93 | 5.92 / 6.79 | 774 / 776 / 778 |
| gate/c32 | FuseZap | 280.5 / 533.7 / 655.9 | 15.70 / 17.19 | 13.81 / 70.29 | 2282 / 2549 / 2641 |
| gate/c32 | vLLM | 304.9 / 404.8 / 437.5 | 6.84 / 8.73 | 6.38 / 12.58 | 1222 / 1247 / 1292 |

The c32 TTFT tail fell from 1.48 s to 0.66 s at p99 (vLLM 0.44 s), because a wave of 32 prompts now goes through in
about 8 prefill steps of 2048 tokens instead of 64 steps of 256. The cost is in the median and in ITL: TTFT p50 rose
from 115 to 281 ms, and ITL p99 is 70 ms, since a decode step that shares the GPU with a 2048-token prefill step waits
for it. TPOT p50 at c32 went from 18.4 to 15.7 ms (vLLM 6.8).

**Power and cost (not part of the gate):** GPU 0 board power over the measured repetitions, at $3.99/h for the whole pod:

| point | power W (FuseZap / vLLM) | J per output token (FuseZap / vLLM) | output tokens per $ (FuseZap / vLLM) |
|---|---|---|---|
| gate/c1 | 441 / 435 | 3.82 / 2.64 | 103,545 / 148,685 |
| gate/c32 | 470 / 557 | 0.269 / 0.168 | 1,579,216 / 2,990,164 |

At c32 FuseZap's energy per token fell from 0.365 to 0.269 J. It is still 1.6× vLLM's (1.4× at c1).

**Concurrency sweep (same pod, both engines, `--max-model-len 2816` for both, no rate points).** `--compare`: VALID,
`GATE: PASS`.

| point | FuseZap tok/s | vLLM tok/s | FuseZap % | TTFT p50 ms (FZ / vLLM) | TPOT p50 ms (FZ / vLLM) | E2E p50 ms (FZ / vLLM) | J/token (FZ / vLLM) |
|---|---:|---:|---:|---|---|---|---|
| short/c1 | 121.1 | 165.7 | 73% | 16.3 / 17.3 | 8.18 / 5.93 | 1,055 / 771 | 3.69 / 2.61 |
| short/c8 | 625.3 | 1246.3 | 50% | 52.6 / 45.6 | 12.38 / 6.11 | 1,648 / 823 | 0.657 / 0.360 |
| short/c32 | 1990.3 | 4376.7 | 45% | 143.7 / 138.6 | 14.83 / 6.20 | 2,031 / 928 | 0.197 / 0.115 |
| gate/c1 | 115.4 | 163.7 | 70% | 29.2 / 24.4 | 8.35 / 5.94 | 1,090 / 779 | 3.83 / 2.68 |
| gate/c8 | 586.0 | 1134.7 | 52% | 109.0 / 112.5 | 12.72 / 6.18 | 1,736 / 897 | 0.746 / 0.411 |
| gate/c32 | 1734.1 | 3287.3 | 53% | 221.5 / 305.7 | 16.60 / 7.24 | 2,312 / 1,226 | 0.270 / 0.169 |
| long/c1 | 113.3 | 161.5 | 70% | 85.0 / 58.2 | 8.48 / 5.96 | 2,250 / 1,577 | 3.96 / 2.78 |
| long/c8 | 517.8 | 1003.9 | 52% | 224.3 / 322.1 | 14.41 / 6.68 | 3,903 / 2,027 | 0.882 / 0.503 |
| long/c32 | 1187.2 | 2361.8 | 50% | 225.6 / 904.7 | 24.77 / 9.74 | 6,710 / 3,451 | 0.408 / 0.246 |
| sharegpt/c1 | 117.4 | 166.4 | 71% | 19.3 / 23.0 | 8.42 / 5.93 | 1,757 / 1,276 | 3.77 / 2.66 |
| sharegpt/c8 | 459.0 | 939.0 | 49% | 78.2 / 49.5 | 12.94 / 6.15 | 2,168 / 1,072 | 0.915 / 0.480 |
| sharegpt/c32 | 1356.5 | 3095.9 | 44% | 103.8 / 90.1 | 16.30 / 6.73 | 3,798 / 1,653 | 0.297 / 0.163 |

Compared with the previous sweep, the prefill-heavy points gained most (long/c32 41% → 50%, gate/c32 46% → 53%). The
decode-heavy c32 points did not: short/c32 went 50% → 45% and sharegpt/c32 45% → 44%. Their prompts are short, so the
larger prefill budget barely helps them, and they are within the ~5% pod-to-pod spread of the old numbers.

**FP8 (W8A8, both engines `--quant fp8`; outside the bf16 gate):**

| point | FuseZap tok/s | vLLM tok/s | ratio | previous Latest |
|---|---:|---:|---:|---:|
| gate/c1 | 142.8 (TPOT 6.89 ms) | 241.4 (TPOT 3.99 ms) | 59.1% | 61.4% |
| gate/c32 | 2024.1 (TPOT 13.83 ms) | 4284.2 (TPOT 5.40 ms) | 47.2% | 38.0% |

**Honest gaps.**
- **The c32 pass is narrow.** 52.8% on this pod, with FuseZap's own repetitions spread over 2.4% (1726.6 to 1768.6) and
  H100 pods differing by about 5%. Run 2's step A/Bs measured 52.4% to 56.6% on other pods, so the verdict has held on
  every pod since bc1144f, but a slow pod could land near 50%.
- **Decode is still the gap.** TPOT p50 at c32 is 15.7 ms against vLLM's 6.8 ms. The M = 32 decode GEMMs (BN 128 tiles
  at BM 32), the M = 256 gate_up GEMM, and fusing SiLU*mul and RoPE into neighbouring kernels are the open items.
  short/c32 and sharegpt/c32, which are mostly decode, are below 50% (45% / 44%), as is sharegpt/c8 (49%).
- **Latency trade-off.** TTFT p99 at c32 fell to 0.66 s (vLLM 0.44 s), but TTFT p50 rose to 281 ms and ITL p99 is
  70 ms against vLLM's 12.6 ms. FuseZap does not yet split a big prefill step into mixed steps that keep decode rows
  moving.
- **Batch 1 passes at 70%.** The gap there is 8.5 vs 5.9 ms per token. This pod's c1 is about 4% below the previous
  Latest pod's (114.8 vs 119.2); run 2's A/Bs showed no c1 change from the engine.
- **24 GB cards** get a 1024-token default budget, not 2048, so this gain is smaller there (not measured against vLLM).
- **Not rerun this time:** the request-rate sweep, the 30-minute stability run and Llama-3.1-8B (it needs an
  `HF_TOKEN`). The P4-BENCH results below are the latest for those. vLLM again ran with
  `VLLM_USE_FLASHINFER_SAMPLER=0` (the image's nvcc 12.4).
- **Tokens per $** uses the whole pod at RunPod's on-demand price ($3.99/h). It is not a hosted-service price.
- Raw JSON, markdown and server logs are under
  `~/.cache/fusezap/remote/results/20261020-164846-END-date-s-78-60-t-r-END-date-s-/` on the machine that ran the
  job. They are not committed.

## 32B: Qwen2.5-32B-Instruct on one H100 (2026-10-21)

The existing Qwen path loads Qwen2.5-32B-Instruct and serves it on one H100 80GB, in bf16 and fp8.
Both engines ran on the same pod for each pair. The method is BENCH_METHOD,
unchanged: the gate workload (512 in / 128 out), 3 reps plus a warm-up wave, `--max-model-len 768`. vLLM is 0.32.0.
`--compare` was VALID for both pairs. c32 fits in memory for both engines (vLLM: 31,792 KV tokens; FuseZap: 1600
blocks), so no concurrency was lowered.

| point | FuseZap tok/s | vLLM tok/s | FuseZap / vLLM | TPOT p50 ms (FZ / vLLM) | TTFT p99 ms (FZ / vLLM) |
|---|---:|---:|---:|---|---|
| bf16 gate/c1 | **31.4** | 40.1 | **78.3%** | 31.4 / 24.6 | 78 / 68 |
| bf16 gate/c32 | **476.1** | 805.3 | **59.1%** | 60.2 / 31.4 | 3,533 / 1,679 |
| fp8 gate/c1 | 47.4 | 69.8 | 68.0% | 20.6 / 14.0 | |
| fp8 gate/c32 | 633.2 | 1216.9 | 52.0% | 46.4 / 20.0 | |

- Pods: bf16 on one pod (build 2745c5c), fp8 on another (build 9170ff2, which has the fp8 prefill-budget
  fix; on main after rebase as d2b2634 / 7f75cfe, with the serve code identical). Both are H100 80GB HBM3 SXM at
  $3.99/h. The bf16 and fp8 rows are on different pods.
- **Accuracy**: top-1 vs HF float32 on the P4-CUDA-SERVE 8-prompt × 128-token method. bf16 is **0.9941**, equal to
  HF's own bf16 (7B: 0.9932). fp8 W8A8 is 0.9746. At 32B the f32 reference uses f32 compute over bf16-stored linear
  weights. That is exactly HF f32's arithmetic and is checked identical on 0.5B.
- **Memory**: the weights take 59.6 GiB. The largest KV pool that served is 45,328 tokens (11.1 GiB). One 32,768-token
  sequence fits, or 22 sequences at 2048 tokens.
- **Gaps**: 32B bf16 gets the minimum 256-row prefill budget, because capture keeps every intermediate alive. That
  gives the c32 TTFT p99 of 3.5 s vs vLLM's 1.7 s. Decode TPOT is about 2× vLLM's in both precisions.
- Two general fixes (not special cases): the model's unused contiguous KV cache now allocates lazily (8.6 GB at a 32k
  context), and the fp8 prefill budget counts W8A8 activations. Without the second, `serve --quant fp8` ran out of
  memory at warm-up on 32B.

## History: SPEED run final measurement (2026-10-20, 861ffb3)

**Verdict: the SPEC §0 throughput gate still FAILS, at batch 32 only.** The comparison is VALID. FuseZap passes at
batch 1 with room to spare and is 6 points short at batch 32:

| gate point | FuseZap out tok/s (3 reps) | vLLM 0.32.0 out tok/s (3 reps) | FuseZap / vLLM | needed | verdict |
|---|---:|---:|---:|---:|---|
| `gate/c1` (batch 1) | **119.2** (119.1 / 119.6 / 119.2) | 165.1 (165.3 / 165.1 / 164.9) | **72.2%** | ≥ 50% | PASS |
| `gate/c32` (batch 32) | **1414.3** (1417.3 / 1414.3 / 1410.0) | 3213.9 (3192.4 / 3241.2 / 3213.9) | **44.0%** | ≥ 50% | FAIL |

`bench_vs_vllm.py --compare` printed `GATE: FAIL` (exit 1), not INVALID. Both engines ran on one pod,
back to back, vLLM first. Neither engine had a failed or short request. The compare printed one
warning: the build had a docs-only dirty path (a `.DS_Store` file, which rsync does not copy). Batch 32 needs about
1607 tok/s on this pod, 13.6% more than FuseZap measured. Since P4-BENCH, the ratios have risen from 2.7% / 1.4% to
72.2% / 44.0%, and from B2-INT's 18.5% / 12.9%.

**Method: unchanged.** docs/BENCH_METHOD.md, bench_vs_vllm.py and workloads.py are as committed. The only
harness change in the SPEED run was the a0aef87 bug fix, which makes `--compare` refuse cross-pod gate points and dirty
builds. It does not change how anything is measured. Gate workload: 512 in / 128 out, greedy, `ignore_eos`, 3
repetitions plus a warm-up wave, unique prompts. FuseZap's batch-1 point ran on a `--max-num-seqs 1` server and its
batch-32 point on a 32-row server (`--num-blocks 1600 --block-size 16`). vLLM ran at its defaults with
`VLLM_USE_FLASHINFER_SAMPLER=0`. No BEAM and no tuning flags were used.

**Setup.**
- Hardware: NVIDIA H100 80GB HBM3 (SXM), driver 595.91.07 (CUDA 13.2), persistence on, Xeon Platinum 8470 host, RunPod
  secure cloud at $3.99/h.
- FuseZap: commit `861ffb3`, with the fusezap/ tree unmodified.
- vLLM: 0.32.0 in its own venv.
- Server start-up: vLLM was ready in 83 s. FuseZap's servers were ready in 26 s (1 row) and 18 s (32 rows), load plus
  warm-up.

**What changed in the engine since B2-INT** (the speed run, one change per step, each
measured on its own):

| commit | step |
|---|---|
| 280aa24 | SPEED-1: fixed-shape jitted decode and prefill forwards replay as one CUDA graph (staged inputs, in-place KV pools) |
| ce0c4b0 | SPEED-2: split-K small-M decode GEMM tiles (BN 64, 4 warps, deterministic last-CTA reduction); the decode linears went from 16.7 to 5.7 ms at M=1 |
| c45150e | SPEED-3: fused residual add + RMSNorm kernel. The residual stream was never realized, so every norm re-added all earlier layers' terms |
| 012d6a1 | SPEED-4: greedy argmax on the device; a step reads back R ints instead of R × 152,064 f32 logits |
| 093a3b0 | SPEED-5: wgmma prefill tiles chosen by wave count at M ≤ 512 (bit-identical sums) |
| f02201c | SPEED-6: tensor-core paged prefill attention |
| a0aef87 | review fix (harness): `--compare` is INVALID for cross-pod gate points or a dirty build |

**Latency (pooled over all 3 repetitions; ms):**

| point | engine | TTFT p50 / p90 / p99 | TPOT p50 / p99 | ITL p50 / p99 | E2E p50 / p90 / p99 |
|---|---|---|---|---|---|
| gate/c1 | FuseZap | 25.2 / 28.4 / 50.8 | 8.21 / 8.26 | 8.21 / 8.48 | 1069 / 1074 / 1096 |
| gate/c1 | vLLM | 28.4 / 30.0 / 32.1 | 5.85 / 5.87 | 5.86 / 6.22 | 771 / 773 / 776 |
| gate/c32 | FuseZap | 115.1 / 1202.3 / 1480.9 | 18.38 / 19.04 | 14.91 / 30.87 | 2493 / 3611 / 3898 |
| gate/c32 | vLLM | 338.2 / 443.6 / 463.5 | 7.21 / 8.89 | 6.43 / 11.71 | 1259 / 1289 / 1320 |

**Power and cost (not part of the gate):** GPU 0 board power over the measured repetitions, at $3.99/h for the whole pod:

| point | power W (FuseZap / vLLM) | J per output token (FuseZap / vLLM) | output tokens per $ (FuseZap / vLLM) |
|---|---|---|---|
| gate/c1 | 494 / 480 | 4.14 / 2.91 | 107,586 / 148,988 |
| gate/c32 | 516 / 591 | 0.365 / 0.184 | 1,276,086 / 2,899,754 |

The H100 now draws about 500 W under FuseZap. In P4-BENCH it drew 130 W, close to idle. The GPU is busy, but each
token still costs 1.4× (c1) and 2.0× (c32) vLLM's energy.

**Concurrency sweep (same pod, both engines, `--max-model-len 2816` for both, no rate points).** `--compare`: VALID,
`GATE: FAIL`.

| point | FuseZap tok/s | vLLM tok/s | FuseZap % | TTFT p50 ms (FZ / vLLM) | TPOT p50 ms (FZ / vLLM) | E2E p50 ms (FZ / vLLM) | J/token (FZ / vLLM) |
|---|---:|---:|---:|---|---|---|---|
| short/c1 | 124.2 | 166.1 | 75% | 17.1 / 26.5 | 7.97 / 5.85 | 1,031 / 769 | 4.04 / 2.86 |
| short/c8 | 659.3 | 1204.8 | 55% | 42.8 / 73.7 | 11.65 / 6.04 | 1,535 / 840 | 0.716 / 0.397 |
| short/c32 | 2119.5 | 4238.3 | 50% | 52.7 / 163.9 | 13.82 / 6.25 | 1,825 / 962 | 0.228 / 0.127 |
| gate/c1 | 117.9 | 165.1 | 71% | 27.4 / 28.3 | 8.28 / 5.86 | 1,080 / 772 | 4.19 / 2.95 |
| gate/c8 | 569.3 | 1111.5 | 51% | 68.5 / 129.6 | 12.88 / 6.12 | 1,715 / 911 | 0.881 / 0.453 |
| gate/c32 | 1486.5 | 3208.5 | 46% | 79.2 / 306.5 | 17.66 / 7.21 | 2,391 / 1,264 | 0.362 / 0.185 |
| long/c1 | 114.7 | 162.2 | 71% | 99.0 / 60.8 | 8.32 / 5.90 | 2,222 / 1,566 | 4.38 / 3.05 |
| long/c8 | 484.6 | 995.2 | 49% | 230.8 / 372.1 | 14.43 / 6.61 | 3,942 / 2,040 | 1.10 / 0.542 |
| long/c32 | 954.6 | 2308.5 | 41% | 290.2 / 1007.5 | 25.27 / 9.79 | 6,983 / 3,513 | 0.579 / 0.265 |
| sharegpt/c1 | 121.4 | 169.5 | 72% | 18.6 / 29.1 | 8.19 / 5.79 | 1,713 / 1,258 | 4.11 / 2.90 |
| sharegpt/c8 | 478.6 | 954.6 | 50% | 43.5 / 70.3 | 12.41 / 5.99 | 2,043 / 1,089 | 1.01 / 0.523 |
| sharegpt/c32 | 1416.0 | 3121.4 | 45% | 111.3 / 120.4 | 16.01 / 6.69 | 3,834 / 1,648 | 0.337 / 0.176 |

The sweep's own gate points are a second run of the gate on the same pod, with a longer context. They give 71.4% /
46.3%: FuseZap's c32 came out 5% higher than in the gate job above (1486.5 vs 1414.3 tok/s), while vLLM's barely moved
(3208.5 vs 3213.9). The official numbers are the gate job's. So two runs of FuseZap's c32 on one pod differed
by 5%, and both fail.

**FP8 (W8A8, both engines `--quant fp8`; outside the bf16 gate):**

| point | FuseZap tok/s | vLLM tok/s | ratio | FuseZap bf16 → fp8 | P4-BENCH FuseZap fp8 |
|---|---:|---:|---:|---:|---:|
| gate/c1 | 146.9 (TPOT 6.62 ms) | 239.1 (TPOT 4.00 ms) | 61.4% | 1.23× | 3.0 |
| gate/c32 | 1612.3 (TPOT 16.07 ms) | 4245.3 (TPOT 5.60 ms) | 38.0% | 1.14× | 22.5 |

FuseZap's fp8 is now faster than its bf16, which it was not in P4-BENCH. vLLM's fp8 gains more (1.45× / 1.32×), so
the fp8 ratio is lower than the bf16 one.

**Honest gaps.**
- **Batch 32 is the gap.** FuseZap's c32 TPOT is 18.4 ms against vLLM's 7.2 ms. At concurrency 8 FuseZap still runs a
  32-row step, padding rows included, which is why c8 sits at about 50%. The SPEED-6 profile puts 8.0 ms of the 10.3 ms
  decode graph in the M = 32 decode GEMMs. The remaining candidates are BN 128 tiles at BM 32, then fusing SiLU*mul,
  RoPE and the bf16 output casts into neighbouring kernels, and the M = 256 gate_up GEMM in the interleaved prefill.
- **TTFT tail at c32.** FuseZap's TTFT p50 is lower than vLLM's (115 vs 338 ms), but its p90 / p99 are 1.2 / 1.5 s
  against 0.44 / 0.46 s. A request that arrives behind a wave of prefills waits for several 256-token prefill steps.
- **Batch 1 passes, at 72% of vLLM.** The remaining gap there is 8.2 vs 5.9 ms per token.
- **One pod.** The gate numbers come from a single pod. Earlier same-build runs gave 69.0% / 41.9% on another pod,
  and H100 pods differ by about 5%. The verdict does not depend on the pod: c1 passes and c32
  fails on every pod measured so far.
- **Not rerun this time:** the request-rate sweep, the 30-minute stability run and Llama-3.1-8B (it needs an
  `HF_TOKEN`). The P4-BENCH results below are the latest for those. vLLM again ran with
  `VLLM_USE_FLASHINFER_SAMPLER=0` (the image's nvcc 12.4).
- **Tokens per $** uses the whole pod at RunPod's on-demand price ($3.99/h). It is not a hosted-service price.
- Raw JSON, markdown and server logs are under `~/.cache/fusezap/remote/results/FINAL-h100/` on the machine that ran
  the job. They are not committed.

## History: P4-BENCH package run (2026-10-19)

**Verdict: the SPEC §0 throughput gate FAILS.** The comparison is VALID: same pod, same GPU, same requests, every
gate point measured, no failed or short requests. On Qwen2.5-7B-Instruct in bf16 (512 tokens in, 128 out), FuseZap
reaches:

| gate point | FuseZap out tok/s | vLLM 0.32.0 out tok/s | FuseZap / vLLM | needed |
|---|---:|---:|---:|---:|
| `gate/c1` (batch 1) | 4.2 | 158.1 | **2.7%** | ≥ 50% |
| `gate/c32` (batch 32) | 42.9 | 3156.8 | **1.4%** | ≥ 50% |

`bench_vs_vllm.py --compare` printed `GATE: FAIL` (exit code 1), not INVALID.

The best tuning I found (BEAM width 2) raises the gate points to 6.9 tok/s (4.4%) at batch 1 and 58.5 tok/s (1.9%) at
batch 32. Neither gets near 50%. The cause is not a benchmark or tuning matter. FuseZap's CUDA serving path has two kernel-level bottlenecks,
measured kernel by kernel below:

1. Every composed reduction (RMSNorm) runs single-threaded per output row on CUDA.
2. The decode GEMMs at batch 1 to 32 use prefill-sized tensor-core tiles, so only a handful of SMs work.

Both are in code that P4-BENCH does not own. This report measures them and states what would have to change.
vLLM is the reference we are trying to reach a share of. On this hardware it is not a target to beat.

## Method

The method is docs/BENCH_METHOD.md, unchanged, run with bench_vs_vllm.py and workloads.py.

- **Requests.** Seeded prompts, sent as token ids to `/v1/completions`. Each request has exactly the target input
  length. Generation is greedy (`temperature 0`) with `ignore_eos`, so both engines generate exactly `max_tokens`.
  Every point and every repetition gets its own prompts, so vLLM's prefix cache never gets a hit.
- **Repetitions.** 3 repetitions per point plus one warm-up wave. Throughput is the median over the repetitions.
  Latency percentiles are pooled over all requests.
- **Servers.** FuseZap's concurrency-1 points run on a `--max-num-seqs 1` server and every other point on a 32-row
  server. vLLM runs at its defaults.
- **Same machine.** The gate pair ran on one pod, one engine after the other.
- **Power.** GPU 0 is sampled with `nvidia-smi` every 500 ms during the measured repetitions.
- **Tokens per dollar.** These use the pods' actual price, **$3.99/h** (RunPod on-demand H100 SXM, secure cloud),
  for the whole pod.

## Versions and hardware

| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (SXM5), 700 W power limit, driver 595.91.07 (CUDA 13.2). Persistence mode was on, and the SM clock maxes out at 1980 MHz |
| Hosts | RunPod secure cloud, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`. The gate pod had a 2× Xeon Platinum 8480+ (224 threads) and 2 TB RAM. The other pods had a Xeon 8470, a Xeon 8570 or an AMD EPYC 9554 (listed per run below) |
| Model | Qwen/Qwen2.5-7B-Instruct, bf16 weights and activations. `--max-model-len` is the longest request rounded up to 256: 768 for the gate, 2816 for vLLM's sweep, 768 to 2816 for FuseZap's sweep parts (see the gaps) |
| vLLM | **0.32.0** (PyPI, 2026-10-18), in its own venv. Started as `vllm serve Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --max-model-len 768 --seed 0`, otherwise at defaults: CUDA graphs, chunked prefill, prefix caching. Its log reported 105k tokens of KV cache. Runs with `VLLM_USE_FLASHINFER_SAMPLER=0`, because the image's nvcc 12.4 cannot JIT FlashInfer 0.7's sampler. Greedy sampling is unaffected |
| FuseZap | commit `6d908068caf8`. The fusezap/ tree is unmodified; only the bench harness had uncommitted changes, and it does not run in the server. Started as `python -m fusezap serve Qwen/Qwen2.5-7B-Instruct --device CUDA --dtype bfloat16 --max-model-len 768 --max-num-seqs {1,32} --num-blocks {1024,1600} --block-size 16`. The decode beam is 0 by default on CUDA (hand-coded opts) |
| Harness | bench_vs_vllm.py as committed with this report (see docs/BENCH_METHOD.md). Python 3.12, httpx streaming client on the same pod |

## The gate (same pod, back to back)

Pod: H100 SXM on a Xeon 8480+ host. vLLM ran first, then FuseZap. Server start-up: vLLM was ready in 83 s. FuseZap
was ready in 35 s for the 1-row server and 22 s for the 32-row server, load plus warm-up.

| point | engine | out tok/s | req/s | TTFT p50 / p99 ms | TPOT p50 ms | ITL p50 / p99 ms | E2E p50 / p99 ms | GPU W (mean) | J / out token | out tokens per $ |
|---|---|---:|---:|---|---:|---|---|---:|---:|---:|
| gate/c1 | vLLM | 158.1 | 1.24 | 30.5 / 35.2 | 6.11 | 6.11 / 7.03 | 806 / 810 | 428 | 2.71 | 142,649 |
| gate/c1 | FuseZap | 4.2 | 0.03 | 1559.6 / 1566.0 | 227.72 | 227.61 / 229.47 | 30,479 / 30,524 | 130 | 30.98 | 3,789 |
| gate/c32 | vLLM | 3156.8 | 24.66 | 322.9 / 452.5 | 7.58 | 6.69 / 11.98 | 1,279 / 1,327 | 545 | 0.173 | 2,848,219 |
| gate/c32 | FuseZap | 42.9 | 0.34 | 3214.9 / 67,435.6 | 659.02 | 288.41 / 1074.07 | 86,844 / 151,330 | 131 | 3.06 | 38,712 |

FuseZap's TTFT p99 at c32 is 67 s. In a closed loop, a request that arrives while 31 others are decoding waits for
prefill steps, which FuseZap runs between decode steps, 256 tokens at a time.

Both engines: 0 failed requests and 0 short outputs at both points. At about 130 W, the H100 is close to idle
while FuseZap runs: its GPU time goes to kernels that keep one thread or a few SMs busy (see *Why*).

## Tuning rounds (the 3-round cap)

| round | GPU | change | gate/c1 | gate/c32 | server start-up |
|---|---|---|---:|---:|---|
| 0 | H100 | defaults (decode beam 0, wgmma GEMMs) | 4.2 tok/s (2.7%) | 42.9 tok/s (1.4%) | 35 s / 22 s |
| 1 | RTX 4090 | `decode_beam=2` (BEAM on the composed decode kernels, plus GEMM tile tuning) | – | – | did not finish within 14 min per config (NVRTC compiles every candidate); no number |
| 1 | RTX 4090 | `--max-prefill-tokens 1024` | – | – | the prefill scratch ran out of memory on 24 GB; 256 stays the default |
| 2 | H100 | `FZ_NO_WGMMA=1` (mma.sync GEMMs, including the 64×64 tiles) | 3.7 tok/s (2.3%) | not reached in the time box | 26 s |
| 2 | H100 | `--beam 2` | **6.9 tok/s (4.4%)**, TPOT 132.8 ms | the 32-row server's BEAM search did not finish in 40 min | 569 s |
| 3 | H100 | `--beam 2`, 32-row server only, 70-minute start-up allowance | (6.9, round 2) | **58.5 tok/s (1.9%)**, TPOT 499 ms | 3169 s (53 min of BEAM search) |
| – | H100 | `--quant fp8` (W8A8; outside the bf16 gate) | 3.0 tok/s (vLLM fp8: 235.4, 1.3%) | 22.5 tok/s (vLLM fp8: 4354.5, 0.5%) | 44 s / 28 s / 34 s |

BEAM helps the batch-1 decode (TPOT 228 → 133 ms). On CUDA its only actions for a reduction are unroll and upcast:
it cannot spread a reduction across threads (see *Why*), and no tuning knob reaches the prefill kernels: the engine compiles prefill
with `FZ_BEAM=0`. The FP8 path is slower than bf16 at batch 1, because each linear adds a row-quantize kernel and
a scale/cast kernel (a known gap).

## The full sweep

vLLM's full sweep ran on one pod (Xeon 8470 host). FuseZap's sweep is too slow for one pod's 2-hour limit, so it
ran on seven pods, each one H100 80GB HBM3 SXM from the same RunPod pool. The files were merged with
`--merge`; the two gate points come from the gate pod above. So, unlike the gate, **sweep rows compare different
physical H100s of the same model.** vLLM's own gate numbers on two pods differ by 4% (158.1 vs 164.9 tok/s at
c1), which is far below the gaps here. `--compare` on the merged files: VALID, `GATE: FAIL`.

| point | FuseZap tok/s | vLLM tok/s | FuseZap % | TTFT p50 ms (FZ / vLLM) | TPOT p50 ms (FZ / vLLM) | E2E p50 ms (FZ / vLLM) | GPU W (FZ / vLLM) | J/token (FZ / vLLM) | tokens per $ (FZ / vLLM) |
|---|---:|---:|---:|---|---|---|---|---|---|
| short/c1 | 4.3 | 166.2 | 2.6% | 782 / 21 | 229.2 / 5.88 | 29,891 / 769 | 128 / 432 | 29.8 / 2.60 | 3,863 / 149,918 |
| short/c8 | 25.4 | 1217.5 | 2.1% | 2,139 / 53 | 298.7 / 6.08 | 40,061 / 826 | 131 / 437 | 5.19 / 0.357 | 22,957 / 1,098,472 |
| short/c32 | 78.9 | 4324.2 | 1.8% | 2,165 / 143 | 376.7 / 6.31 | 49,906 / 935 | 132 / 495 | 1.67 / 0.115 | 71,186 / 3,901,489 |
| gate/c1 | 4.2 | 164.9 | 2.5% | 1,560 / 26 | 227.7 / 5.88 | 30,479 / 772 | 130 / 440 | 31.0 / 2.67 | 3,789 / 148,751 |
| gate/c8 | 19.9 | 1145.2 | 1.7% | 3,216 / 108 | 366.6 / 6.15 | 49,406 / 888 | 132 / 473 | 6.62 / 0.414 | 17,987 / 1,033,219 |
| gate/c32 | 42.9 | 3291.5 | 1.3% | 3,215 / 309 | 659.0 / 7.27 | 86,844 / 1,234 | 131 / 556 | 3.06 / 0.170 | 38,712 / 2,969,762 |
| long/c1 | 3.9 | 162.2 | 2.4% | 6,337 / 62 | 229.8 / 5.91 | 64,936 / 1,570 | 130 / 451 | 33.0 / 2.78 | 3,557 / 146,323 |
| long/c8 | 15.6 | 1003.4 | 1.6% | 9,623 / 362 | 450.5 / 6.57 | 123,304 / 2,027 | 173 / 506 | 11.1 / 0.504 | 14,089 / 905,342 |
| long/c32 | 26.3 | 2333.8 | 1.1% | 9,767 / 990 | 1046.0 / 9.84 | 275,946 / 3,464 | 177 / 576 | 6.74 / 0.248 | 23,749 / 2,105,688 |
| sharegpt/c1 | 4.3 | 166.7 | 2.6% | 783 / 30 | 229.5 / 5.88 | 49,273 / 1,278 | 128 / 441 | 30.0 / 2.64 | 3,869 / 150,412 |
| sharegpt/c8 | 18.5 | 939.7 | 2.0% | 2,420 / 59 | 309.6 / 6.12 | 55,517 / 1,089 | 131 / 452 | 6.96 / 0.479 | 16,682 / 847,831 |
| sharegpt/c32 | 54.6 | 3070.1 | 1.8% | 4,600 / 118 | 415.5 / 6.84 | 107,048 / 1,674 | 132 / 502 | 2.42 / 0.164 | 49,275 / 2,769,996 |
| sharegpt/r1 | 57.9 | 214.5 | 27% | 2,420 / 27 | 448.2 / 5.85 | 87,076 / 1,113 | 127 / 352 | 2.23 / 1.68 | 52,210 / 193,498 |
| sharegpt/r2 | 59.3 | 461.0 | 13% | 2,421 / 26 | 440.3 / 5.88 | 79,786 / 1,097 | 127 / 437 | 2.14 / 0.969 | 53,475 / 415,977 |
| sharegpt/r4 | 59.2 | 819.6 | 7.2% | 2,426 / 27 | 438.6 / 5.98 | 83,810 / 1,114 | 136 / 458 | 2.31 / 0.552 | 53,442 / 739,490 |
| sharegpt/r8 | 60.4 | 1623.1 | 3.7% | 2,424 / 29 | 426.3 / 6.16 | 88,171 / 1,256 | 137 / 485 | 2.22 / 0.292 | 54,517 / 1,464,418 |
| sharegpt/r16 | 59.2 | 2943.0 | 2.0% | 2,426 / 33 | 439.8 / 6.68 | 84,797 / 1,235 | 132 / 509 | 2.30 / 0.180 | 53,416 / 2,655,379 |
| sharegpt/rinf | 60.5 | 3597.5 | 1.7% | 2,429 / 39 | 435.9 / 6.82 | 89,512 / 1,429 | 132 / 522 | 2.17 / 0.145 | 54,596 / 3,245,887 |

How to read the sweep:

- **FuseZap's throughput depends on the server's row count, not on the load.** Its 32-row decode step costs the
  same however many rows are live, because it runs every padding row. Prefill steps of 256 tokens are interleaved
  with decode, which is why TPOT grows with prompt length and concurrency. The rate sweep shows this: FuseZap is saturated at 1 request/s
  already, so every rate gives about 59 tok/s (with at most 32 requests in flight).
- **The 27% at r1 does not mean FuseZap is close.** At 1 request/s vLLM is idle most of the time: it serves the
  offered load (214 tok/s) with 26 ms TTFT. FuseZap cannot keep up with the offered load and queues, so its E2E
  latency is 80× longer.
- **vLLM's closed-loop throughput** scales from about 165 tok/s at batch 1 to 3,300 to 4,300 tok/s at 32, with
  TPOT staying between 6 and 10 ms. Its TPOT p50 at r1 is 5.85 ms, which is about 2.5 TB/s of weight traffic for a
  15.2 GB model, close to the H100's bandwidth.

Pods (all H100 80GB HBM3, driver 595.91.07, $3.99/h):
- gate: Xeon 8480+
- vLLM sweep, stability and fp8: Xeon 8470
- FuseZap c8/c32 points: two Xeon 8470 pods
- FuseZap c1 points: two Xeon 8470 pods
- FuseZap rates: Xeon 8470 / 8470 / 8570
- FuseZap stability and fp8: Xeon 8480+
- tuning: EPYC 9554

## Stability (30 minutes, `gate` workload, closed loop at concurrency 32)

| | vLLM 0.32.0 | FuseZap |
|---|---|---|
| requests (completed / failed / short) | 48,589 / 0 / 0 | 670 / 0 / 0 |
| output tok/s over the run | 3,453.6 | 46.6 |
| output tokens per $ at $3.99/h | 3,116,000 | 42,100 |
| per-minute output tok/s (min to max, full windows after the ramp) | 3,411 to 3,501 | 41 to 62 |
| drift, first vs last 7 windows: output tok/s | +0.9% | +3.3% |
| drift: TPOT p50 | −0.8% (7.58 → 7.53 ms) | 0.0% (658 → 660 ms) |
| drift: TTFT p50 | −1.3% | −67%: the first windows hold the initial 32-request prefill queue, then it settles |
| drift: E2E p99 | +2.7% | −9.5% |
| server RSS growth (host, process tree) | +1.9 MiB (3.66 GB) | +47.8 MiB (22.1 GB) |
| GPU memory growth | 0 MiB (75.8 GB used, vLLM's pre-allocated pool) | 0 MiB (19.8 GB used) |
| GPU power mean / max, temperature max | 584 / 616 W, 60 °C | 140 / 147 W, 39 °C |

Neither engine had an error, latency creep or a GPU memory leak in 30 minutes.

FuseZap's numbers come in steps:
- Its per-window throughput moves between 41 and 62 tok/s because a whole wave of 32 requests finishes within a
  window or two (each E2E is about 87 s).
- The ramp it excludes is 120 s. The drift compares windows after the first median E2E.
- The +48 MiB of host RSS over 30 minutes is small next to the 22 GB resident set, most of which is the
  memory-mapped safetensors. Python allocator growth is the likely cause, and it needs a longer run to call it
  either way.

The final window of each run is partial (requests draining after the 30 minutes) and is excluded.

## Why FuseZap is 30 to 70 times slower here

I profiled one RTX 4090 pod per kernel: every recorded item timed with a device sync around it, over the engine's
captured decode and prefill jits for Qwen2.5-7B bf16. The H100 TPOT (228 ms at batch 1) is the same as the 4090's
(222 ms), which already shows the GPU's bandwidth and FLOPs are not the limit.

Per batch-1 decode step on the RTX 4090 (213 ms in all, 367 kernel launches):

| kernel | per step | what it is |
|---|---:|---|
| `r_3584` (global size 1), 57 per step | **≈146 ms (69%)** | the RMSNorm mean-of-squares over d_model = 3584, run by **one thread**, 2.56 ms per call |
| `fz_serve_gemm_bf16_bf16 M=1 N=3584 K=18944` (down_proj) ×28 | 32.8 ms | 1.17 ms per call. 136 MB of weight at 1 TB/s would take 0.14 ms: only 14 CTAs of 128×256 tiles cover N = 3584 |
| gate/up `M=1 N=37888 K=3584` ×28 | 19.7 ms | 0.70 ms per call (39% of bandwidth) |
| qkv and o_proj `M=1` ×28 each | 13.2 ms | 0.24 ms per call: 18 and 14 CTAs |
| lm_head `M=1 N=152064` (f32 out) | 2.3 ms | |
| attention (paged split-K), KV writes, elementwise | ≈4 ms | already fast |

Prefill: `r_256_3584` (the RMSNorm of 256 rows) is launched as a single workgroup and takes 9.5 ms per call. That is
about 540 ms of every 640 ms prefill step of 256 tokens, or 2.5 ms per prompt token. vLLM's TTFT for a whole
512-token prompt is 26 ms.

The root causes are in `fusezap/codegen/lower.py`:
- The group-for-reduce heuristic (`heuristic_opts`), the BEAM `group_reduce` action and `check_opts` are all
  METAL-only (simdgroup reductions).
- On CUDA, a reduction therefore gets one thread per output. BEAM can only unroll it, which is why `--beam 2` helps
  the batch-1 decode (228 → 133 ms) and nothing else.

A rough upper bound, from the profile: a warp/block reduction on CUDA would remove about 69% of the decode step.
Split-K or GEMV kernels for M ≤ 32 would bring the GEMMs to bandwidth. Batch 1 would then be bounded by about 15
GB of weights at 3 TB/s, plus about 367 Python-side launches (the jit replays item by item, without CUDA graphs).
That is roughly 8 to 10 ms per token, or 100 to 125 tok/s, which is 60 to 75% of vLLM. That is a projection, not a
measurement.

## Gaps and caveats

- **Only the gate pair is same-pod.** The sweep, stability and fp8 rows put each engine on a different pod. All are
  H100 80GB HBM3 SXM at $3.99/h, and the CPUs vary, as listed above. With a gap of 30 to 70 times, this does not
  change any conclusion. A same-pod sweep would need one pod for about 7 hours.
- **Tuned FuseZap** (`--beam 2`): the best numbers are 6.9 tok/s at c1 (4.4%) and 58.5 tok/s at c32 (1.9%). They ran on two
  different pods, and the c32 server needed 53 minutes of BEAM search before it was ready. `--merge` + `--compare`
  against the gate pod's vLLM file: VALID, `GATE: FAIL`. The tuned build does not change the verdict either.
- **FP8.** FuseZap's fp8 (W8A8) is slower than its bf16: 3.0 / 22.5 tok/s at c1 / c32 (two pods). vLLM's fp8 (dynamic per-tensor W8A8, `--quantization fp8`) reaches 235.4 / 4354.5
  tok/s, 1.49× / 1.38× its bf16. FP8 is outside the bf16 gate.
- **The context length differs in the split sweep.** Each FuseZap sub-run sized `--max-model-len` from its own
  longest request: 768 to 2816 tokens. vLLM's single sweep used 2816. The merged compare prints this as a warning.
  The setting only sizes the KV pool and the attention bound. FuseZap's step time is set by its fixed-shape norms and
  GEMMs, so a shorter context can only flatter FuseZap here, if anything. A rerun should pin `--max-model-len 2816`
  on every part, as docs/BENCH_METHOD.md now says.
- **Llama-3.1-8B was not run.** It is gated on Hugging Face and needs the owner's licence acceptance and an `HF_TOKEN`.
- **vLLM's FlashInfer sampler was off** (`VLLM_USE_FLASHINFER_SAMPLER=0`), a workaround for the image's nvcc 12.4.
  With greedy decoding vLLM samples by argmax either way. The effect on its throughput should be within noise, but
  I did not measure it on a CUDA 12.8+ image.
- **Commit.** The gate pod and the vLLM sweep pod ran before the harness could read the commit on a pod (rsync drops
  `.git`), so their files say `commit: null`. They ran the same tree as every later pod: `6d908068caf8` plus the
  uncommitted harness, synced by the same `remote.sh`. The local copies used for the merge were stamped with it.
- **Tokens per dollar** is the whole pod at RunPod's on-demand price at the time ($3.99/h). It is not a
  hosted-service price.
- **Power** is nvidia-smi board power, sampled every 500 ms, over each point's measured repetitions only. FuseZap's
  ~130 W is close to the H100's idle power (~70 W after boost). Its joules per token are dominated by run time, not
  by load.
- The raw JSON, markdown and server logs are not committed (the run's rule). They are under
  `~/.cache/fusezap/remote/results/*P4-BENCH*` on the machine that ran the jobs.

## Reproduce

Gate (one pod, about 40 minutes; `M='--model Qwen/Qwen2.5-7B-Instruct'`, setup as in docs/BENCH_METHOD.md):

    export PATH=$HOME/vllm-env/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 FZ_BENCH_GIT_COMMIT=<sha>
    G="$M --workloads gate --concurrency 1,32 --rates none --out-dir results/bench/gate"
    .venv/bin/python bench_vs_vllm.py --engine vllm $G --startup-timeout 900 --time-limit 900
    .venv/bin/python bench_vs_vllm.py --engine fusezap --device CUDA $G --startup-timeout 900 --time-limit 1800
    .venv/bin/python bench_vs_vllm.py --compare results/bench/gate/*-fusezap.json results/bench/gate/*-vllm.json --usd-per-hour 3.99

Sweep subsets per pod:
- `--workloads short,sharegpt --concurrency 8,32`
- `--workloads gate --concurrency 8`
- `--workloads long --concurrency 8,32`
- `--workloads short,sharegpt --concurrency 1`
- `--workloads long --concurrency 1`
- `--workloads '' --rates 1,2` (likewise `4,8` and `16,inf`)
- stability: `--stability-minutes 30`

Then on the Mac, `--merge fz-full.json <files>` and `--compare fz-full.json <vllm full>.json --usd-per-hour 3.99`.

## Integration rerun (B2-INT, 2026-10-20)

After P4-BENCH's two root causes were fixed in the engine (group-for-reduce on CUDA in `fusezap/codegen/lower.py`;
short decode GEMM tiles for M ≤ 64 in `fusezap/device/cuda_kernels`), the gate pair was rerun on one H100 80GB HBM3
SXM pod (driver 595.91.07, vLLM 0.32.0, $3.99/h). Same commands as above, 3 reps, no BEAM. The FuseZap build is 1a68611
plus the B2-INT changes. `--compare`: VALID, `GATE: FAIL`.

| point | FuseZap | vLLM | ratio | P4-BENCH run |
|---|---:|---:|---:|---:|
| gate/c1 | 30.3 tok/s (TPOT p50 32.6 ms, 221 W) | 164.1 tok/s (TPOT 5.9 ms, 437 W) | 18.5% | 2.7% |
| gate/c32 | 414.3 tok/s (TPOT p50 64.2 ms, 246 W) | 3213.3 tok/s (TPOT 6.9 ms, 552 W) | 12.9% | 1.4% |

Tokens per $ at gate/c32: FuseZap 374 k, vLLM 2.90 M. On an RTX 4090 the same build runs at 40.3 / 347.7 tok/s.
That the 4090 beats the H100 at batch 1 means decode is now launch-bound on the host, not GPU-bound.
