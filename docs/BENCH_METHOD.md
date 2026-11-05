# Benchmark method: FuseZap vs vLLM

This page explains how we compare FuseZap's server with vLLM, so anyone can check the numbers. The harness
(`bench_vs_vllm.py`, with its request sets in `workloads.py`) is not in this repository: it shares its load generator
with FuseZap Validate, which is licensed separately. The method below (workloads, gate, repetitions, how runs are
compared and merged) is complete without it, and the commands show exactly how the published numbers were run.

## What the gate is

SPEC §0, item 4: on the same GPU, with a 7B model in bf16 and 512 input / 128 output tokens, FuseZap's output-token
throughput must be **at least 50% of vLLM's at batch 1 and at batch 32**. The stretch goal is 80%. In the harness, the
`gate` workload (512 / 128) at concurrency 1 and 32 measures exactly this, and `--compare` prints PASS or FAIL for
it. Every other number in a report is context. None of it is part of the gate.

We are not trying to show that FuseZap beats vLLM on an H100. vLLM is the most-tuned open-source server on NVIDIA
hardware. The claim we want to support is that FuseZap reaches a credible share of vLLM there, while running on hardware
vLLM does not target.

## Setup

| | |
|---|---|
| GPU | one NVIDIA H100 80GB (SXM or PCIe; the report records which, from `nvidia-smi`). Both engines run on the same machine, one after the other, never at the same time |
| Models | **Qwen/Qwen2.5-7B-Instruct** (Apache-2.0, ungated; this is the gate model). **meta-llama/Llama-3.1-8B-Instruct** is the second architecture check. It is gated on Hugging Face: the owner must accept the Llama 3.1 licence on the model page and put an `HF_TOKEN` on the instance before it can be downloaded |
| Precision | bf16 weights and activations for both engines (`--dtype bfloat16`, the default on CUDA). Quantized runs (`--quant fp8` for both engines; int8/int4 on FuseZap only) are reported separately and never count toward the bf16 gate |
| vLLM | a pinned release (0.32.0 for the H100 report, `docs/bench-h100.md`), installed in its own virtualenv on the instance. The version goes into every result file. The harness never imports vLLM. It runs `vllm serve` as a separate process |
| FuseZap | the commit under test (`git rev-parse HEAD`, plus a dirty flag, in every result file), run as `python -m fusezap serve` |
| Engine settings | the same model, dtype, context length (`--max-model-len`: the longest request rounded up to 256 tokens) and seed. Everything else is left at each engine's defaults, so vLLM keeps CUDA graphs, chunked prefill and automatic prefix caching (it gets no hits: no prompt is ever sent twice, see below). FuseZap gets two harness settings, both described under "Fairness": `--max-num-seqs`, and a KV-cache pool sized so that many sequences fit without preemption, which is what vLLM does automatically with `--gpu-memory-utilization` |

## Workloads

All prompts are generated from fixed seeds, so nothing is downloaded and both engines get byte-identical requests in
the same order. The seed of each request set is derived from the point key (for example `gate/c32`), the repetition
and `--seed`, so every point, every repetition and every warm-up wave has its own prompts: no prompt is sent twice in
a run. (An earlier version seeded by workload and repetition only; because a seeded set of n requests starts with the
same requests as a smaller one, `gate/c32` re-sent all of `gate/c1`'s prompts and the rate points re-sent each
other's, which vLLM's prefix cache would have served as cache hits.)

| Workload | Input / output tokens | Why |
|---|---|---|
| `short` | 128 / 128 | a short chat turn; mostly decode-bound |
| `gate` | 512 / 128 | the SPEC §0 gate shape |
| `long` | 2048 / 256 | a long prompt; mostly prefill-bound, which stresses attention and chunked prefill |
| `sharegpt` | mixed | a ShareGPT-like mix. Input and output lengths are drawn from log-normal distributions fitted to the published ShareGPT-V3 statistics (both medians about 180 to 190 tokens, with long tails), clipped to 16-2048 input and 16-512 output tokens |

The prompts are seeded English-like text. In the default `--prompt-format tokens`, the harness tokenizes that text
with the model's own tokenizer, trims it to exactly the target input length, and sends the token ids to
`/v1/completions`. Both engines therefore prefill exactly the same tokens, and no chat template is involved.
`--prompt-format chat` sends the text to `/v1/chat/completions` instead, for an end-to-end check that includes
templating.

Every request is greedy (`temperature: 0`), streamed, and has the same `max_tokens` on both engines, with
`ignore_eos: true` so generation does not stop at an end-of-sequence token. That way both engines generate the same
number of tokens. If an engine ignores the flag, its early-stopping requests appear in the `short` column. Throughput
is always computed from the tokens actually generated, and `--compare` prints a warning when this happens.

## Load patterns

- **Closed-loop concurrency.** Every workload runs at concurrency 1, 8 and 32: that many clients, and each one sends
  its next request as soon as its previous one finishes. Each point sends max(16, 2 × concurrency) requests per
  repetition. Concurrency 1 is the batch-1 gate and 32 is the batch-32 gate.
- **Open-loop request rate.** The `sharegpt` mix also runs at 1, 2, 4, 8 and 16 requests/s, and at "inf" (everything
  sent at once). Arrival times follow a seeded Poisson process, and requests are sent on schedule whether or not
  earlier ones have finished, up to `--max-concurrency` in flight (default 32; a request that arrives while 32 are
  in flight waits in the client). Each rate point has 128 requests. This shows where each engine's latency starts to
  climb.

## Fairness

These are harness choices, applied the same way to both engines unless stated, and recorded in the result JSON:

- **Unique prompts.** See Workloads: no prompt is repeated across points, repetitions or warm-ups, so vLLM's
  automatic prefix cache (on by default) never gets a hit. FuseZap has no prefix cache.
- **FuseZap's batch-1 server.** FuseZap's decode step always runs `--max-num-seqs` rows; the rows no sequence uses
  are padding. A server started with `--max-num-seqs 32` therefore does 32 rows of work per token even at batch 1,
  while vLLM's CUDA graphs run at the real batch size. So when the harness launches FuseZap, it runs every
  concurrency-1 point (including `gate/c1`) on a server started with `--max-num-seqs 1`, then starts a second server
  with `--max-num-seqs max(32, largest concurrency)` for every other point. `meta.servers` lists both commands and
  which points ran on each. `--single-server` restores one server for everything, which handicaps FuseZap at batch 1.
  Points at concurrency 8 run on the 32-row server, so FuseZap still pays for 24 padding rows there; that is a known
  handicap and outside the gate.
- **Batch cap in the rate sweep.** FuseZap's second server batches at most 32 sequences; vLLM's default
  `max_num_seqs` is 256 or more. The client therefore keeps at most `--max-concurrency` (default 32) requests in
  flight during the rate sweep, so neither engine batches more than 32 sequences. With `--max-concurrency 0` the
  client sends every request on schedule. vLLM can then batch up to its own limit, and FuseZap queues anything past 32.
  Report such a run as "each engine at its own batch limit", not as a like-for-like comparison.

## Warm-up and repetitions

- The harness waits until `/health` and `/v1/models` both answer. Model load time and FuseZap's first-run kernel
  tuning are not measured; the server's startup time is recorded but not compared.
- Before each point, one warm-up wave runs at that point's concurrency: at least 4 requests, each capped at 16 output
  tokens, with seeds that the measured requests never use. This is where vLLM captures its CUDA graph for that batch
  size and FuseZap compiles and captures its kernels for that shape. Warm-up requests are not measured.
- Each point is repeated 3 times (`--reps`), and each repetition uses fresh prompts, so a prefix cache cannot
  benefit from a prompt it saw in an earlier repetition or at an earlier point. Throughput is the **median over repetitions**. Latency
  percentiles are pooled over every completed request from all repetitions. The per-repetition throughputs are kept
  in the JSON, so you can see the spread.

## Metrics

| Metric | Definition |
|---|---|
| TTFT | time from sending the request to the first streamed chunk containing text |
| ITL | inter-token latency: the gaps between consecutive streamed text chunks |
| TPOT | time per output token after the first: (E2E - TTFT) / (output tokens - 1), per request |
| E2E | time from sending the request to the end of the stream |
| output tok/s | output tokens summed over completed requests, divided by the repetition's wall time (first send to last completion) |
| req/s | completed requests divided by the same wall time |

Each per-request metric is reported as the mean, p50, p90, p95 and p99. Token counts come from the server's
`usage.completion_tokens` (`stream_options.include_usage`). If the server does not send it, the harness counts the
streamed chunks instead.

## Power, cost and stability

These numbers go into the report next to the throughput, but they are never part of the gate.

- **GPU power.** While each point's measured repetitions run (not the warm-up), the harness samples GPU 0 with
  `nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu,clocks.sm -lms 500`
  (`--gpu-sample-ms`; 0 turns it off). It records the mean and the maximum of each field per point, plus **joules per
  output token**: the mean board power × the measured wall time ÷ the output tokens. This is board power for the
  whole GPU, idle time included, so at concurrency 1 it is mostly the cost of keeping the GPU busy for one stream.
  Without nvidia-smi, or on a GPU with no power sensor (`[N/A]`), the fields are empty.
- **Tokens per dollar.** `--usd-per-hour` (or `FZ_BENCH_USD_PER_HOUR`) is the pod's actual price, which `remote.sh`
  prints as `created at $X/h`. The harness reports output tokens per dollar = output tok/s × 3600 ÷ $/h. `--compare
  --usd-per-hour X` recomputes it for both files, which matters when the price is only known after the job has run.
  Both engines run on the same pod, so the ratio of tokens per dollar is the ratio of throughputs. The absolute
  number is what a customer would pay for on that GPU.
- **Stability run.** `--stability-minutes M` replaces the points with one closed-loop run at
  `--stability-concurrency` (default 32) on `--stability-workload` (default `gate`) for M minutes. Every request
  has its own seed, which no point uses, so no prompt is ever sent twice. The run is cut into
  `--stability-window`-second windows (default 60), assigned by completion time. For each window the harness
  records completed and failed requests, output tok/s, TTFT, TPOT and E2E percentiles, GPU power and memory, and the
  server's resident memory (from `/proc`, the process and its children). The **drift** line compares the first and
  the last quarter of the full windows: the change in output tok/s, TPOT p50, TTFT p50 and E2E p99, the error total,
  and the growth of server RSS and GPU memory. The last window is left out because it is partial. A stable server
  shows changes of a few percent, no errors and flat memory.

## What is compared, and what is not

**Compared:** serving throughput and latency through the same HTTP API, with the same requests, model, precision and
GPU. The client is identical for both engines, so its overhead is the same on both sides. The gate counts only
`gate/c1` and `gate/c32`.

**Not compared:**

- Output quality. Correctness is checked separately: P4-MODEL requires FuseZap to be token-identical to HF
  `transformers`, and P4-QUANT documents the perplexity deltas.
- Startup or compile time.
- Multi-GPU or tensor-parallel serving (not in v1).
- Speculative decoding.
- Prefix-cache hit rates. The prompts are deliberately unique across the whole run.
- Engine-specific tuning flags beyond the defaults. If a report uses any, they are listed in `server_cmd` in the
  JSON.

A `--quick` run (the Mac dry run) checks that the harness works. It is never a gate measurement, and `--compare`
says so when either file comes from one.

## Outputs

Each run writes `bench/results/<UTC timestamp>-<engine>.json`, `.md` and `.server.log` (the server's own output;
FuseZap runs with a batch-1 server write `.server-seqs1.log` and `.server-seqs32.log`).
The JSON stores everything needed to audit the run:

- the engine and its version, the model, dtype and quantization, and the GPU name and driver
- the FuseZap git commit (with a dirty flag), the exact server commands (secrets redacted), which points ran on which
  server, and the sampling settings
- whether the run is complete, and which points were not measured if not
- the seed, repetitions and warm-up
- for every point: per-repetition throughput, latency percentiles, error counts and short-output counts

Comparing two runs:

    uv run python bench_vs_vllm.py --compare bench/results/<ts>-fusezap.json bench/results/<ts>-vllm.json

This prints FuseZap as a percentage of vLLM at every point both files share. It then prints the gate lines and one of:

- `GATE: PASS` (exit code 0) or `GATE: FAIL` (1).
- `GATE: MISSING POINTS` (2): `gate/c1` or `gate/c32` is missing from either file.
- `GATE: INVALID` (3): the two runs are not comparable, and no verdict is given. This happens when the model, dtype,
  quantization, GPU, prompt format, seed or scale differ; when either file is a `--quick` run or an incomplete run;
  or when `gate/c1` or `gate/c32` has failed requests or requests that stopped before `max_tokens` on either side.
  The reasons are listed under "Not comparable". Both servers honour `ignore_eos` (FuseZap's API passes it to the
  engine since P4-API). If an engine ever drops it, a gate point where the model emits an end-of-sequence token
  shows up here as INVALID, which is better than a misleading tok/s.
  It is also INVALID when a gate point ran on different machines (`meta.host`, the pod's hostname; for a merged file,
  the host of the part that ran that point) or a host is not recorded, and when the FuseZap build had uncommitted
  changes outside `docs/` (`meta.git.dirty_paths`). These two checks were added after the SPEED review: before them,
  `--compare` accepted a FuseZap run divided by a vLLM run from another pod, and H100 pods differ by about 5% (vLLM
  gave 164.1 / 3213.3 tok/s on one pod and 158.1 / 3156.8 on another), which is as large as the gains being judged.

Failed or short requests outside the gate points, a different machine outside the gate, and a docs-only dirty tree
are printed as warnings only.

Each run's JSON and markdown are rewritten after every point. If a run stops early (an error, `--time-limit`,
`SIGTERM`), the files keep every finished point, and `meta.complete` is false with the missing points in
`meta.skipped`. The run exits with code 4 after `--time-limit` and 130 after `SIGINT` or `SIGTERM`; it stops the
server in both cases.

## Reproducing on an H100

Our cloud runner (`remote.sh`, not in this repository) starts a RunPod pod, rsyncs the repo to `/workspace/fusezap`,
runs one command over SSH, copies the pod's `results/` directory back to the machine that started it, and always
terminates the pod. Write results to `results/bench` on the pod so they come back with the job.

One rule shapes these commands. `remote.sh job` copies `results/` back only after the command returns. If the job's
`--max-hours` wall clock runs out first, the pod is terminated and nothing comes back. So every command below keeps
itself inside the job's limit. Each bench call has a `--time-limit` (start no new point after that many seconds), a
bounded `--startup-timeout` (per server; `--time-limit` is also checked before each server starts, so FuseZap's
second server is never started late), and an outer `timeout` as a hard backstop. The outer timeouts, plus the
install, stay well under `--max-hours`. Steps are joined with `;` rather than `&&`, so a step that fails or
times out does not skip the steps after it, and whatever was measured still comes back. The bench calls run
`.venv/bin/python` directly so that `timeout`'s SIGTERM reaches the harness, which then stops the server and keeps
the finished points.

**Job 1: the gate first** (about 1.5 to 2 hours; `--max-hours 3`). This is only the `gate` workload at concurrency
1 and 32, so the gate result comes back even if FuseZap's first-run kernel tuning on a 7B model is slow. FuseZap
starts two servers here (batch 1 and batch 32), and each gets up to 40 minutes to tune and start:

    remote.sh job --gpu h100 --max-hours 3 -- "
      uv sync --extra serve &&
      uv venv ~/vllm-env && VIRTUAL_ENV=~/vllm-env uv pip install vllm==0.32.0 &&
      export PATH=\$HOME/vllm-env/bin:\$PATH VLLM_USE_FLASHINFER_SAMPLER=0 FZ_BENCH_GIT_COMMIT=<sha> &&
      G='--model Qwen/Qwen2.5-7B-Instruct --workloads gate --concurrency 1,32 --rates none --out-dir results/bench/gate';
      timeout -k 60 30m  .venv/bin/python bench_vs_vllm.py --engine vllm \$G --startup-timeout 900 --time-limit 1500;
      timeout -k 60 100m .venv/bin/python bench_vs_vllm.py --engine fusezap --device CUDA \$G --startup-timeout 2400 --time-limit 3600;
      .venv/bin/python bench_vs_vllm.py --compare results/bench/gate/*-fusezap.json results/bench/gate/*-vllm.json | tee results/bench/gate/compare.md"

The model download happens inside vLLM's first startup (hence its 15-minute `--startup-timeout`); FuseZap reads the
same Hugging Face cache. Worst case: about 15 minutes of install, plus 30 and 100 minutes for the two bench calls, which is
2 hours 25 minutes inside a 3-hour limit.

**Job 2: the full sweep** (`--max-hours 5`). Run it only after job 1 came back clean. FuseZap's tuned kernels are
cached on the pod only, so this job tunes again. Give it the same 40-minute startup allowance:

    remote.sh job --gpu h100 --max-hours 5 -- "
      uv sync --extra serve &&
      uv venv ~/vllm-env && VIRTUAL_ENV=~/vllm-env uv pip install vllm==0.32.0 &&
      export PATH=\$HOME/vllm-env/bin:\$PATH VLLM_USE_FLASHINFER_SAMPLER=0 FZ_BENCH_GIT_COMMIT=<sha> &&
      F='--model Qwen/Qwen2.5-7B-Instruct --out-dir results/bench/full';
      timeout -k 60 80m  .venv/bin/python bench_vs_vllm.py --engine vllm \$F --startup-timeout 900 --time-limit 4200;
      timeout -k 60 170m .venv/bin/python bench_vs_vllm.py --engine fusezap --device CUDA \$F --startup-timeout 2400 --time-limit 7200;
      .venv/bin/python bench_vs_vllm.py --compare results/bench/full/*-fusezap.json results/bench/full/*-vllm.json | tee results/bench/full/compare.md"

Worst case: about 15 minutes, plus 80 and 170 minutes, which is 4 hours 25 minutes inside a 5-hour limit. If a time limit cuts a
sweep short, the result is marked incomplete, and `--compare` reports INVALID for it. Rerun the missing points with
`--workloads`, `--concurrency` and `--rates`.

Notes on these commands:

- `uv sync --extra serve` installs FuseZap's serve extra and the default dev group. The dev group includes `httpx`,
  which the bench client uses.
- vLLM goes into its own virtualenv. The harness only ever runs `vllm serve` as a subprocess, so it just needs `vllm`
  on `PATH`.
- For Llama-3.1-8B-Instruct, the owner must first accept the licence on Hugging Face and pass `HF_TOKEN` to the pod.
  Then run the same pair of commands with `--model meta-llama/Llama-3.1-8B-Instruct`.
- Do not put secrets on the server command line if you can avoid it. If you must, the harness redacts the values of
  secret-looking flags (`--api-key X`, `--hf-token=X`, `HF_TOKEN=X`) in the printed command and in `meta.server_cmd`,
  and replaces those values with `***` in the `.server.log` (servers echo their arguments). The client's own
  `--api-key` is never written to the result files.

- `VLLM_USE_FLASHINFER_SAMPLER=0`: remote.sh's default image (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel`) ships
  nvcc 12.4. vLLM 0.32.0's FlashInfer sampler JIT-compiles at startup with an nvcc flag that 12.4 does not know
  (`--compress-mode=size`), and the engine then fails to start. With the variable set, vLLM uses its PyTorch
  sampler. Requests are greedy, so this does not change any output. vLLM records the variable in `meta.env`.
- `FZ_BENCH_GIT_COMMIT=<sha>` (the output of `git rev-parse HEAD` on the machine that syncs; add
  `FZ_BENCH_GIT_DIRTY_PATHS=a,b` for uncommitted files): rsync does not copy `.git`, so the harness cannot read the
  commit on the pod itself. Without the variable, `meta.git.commit` is null.
- **Splitting a sweep across pods.** A slow engine may not fit the whole sweep into one pod's wall clock. Run subsets
  (`--workloads`, `--concurrency`, `--rates`) on several pods, then merge each engine's files on the Mac:
  `bench_vs_vllm.py --merge OUT.json IN1.json IN2.json ...`. The merge refuses files that differ in the engine,
  version, commit, model, dtype, quant, GPU, prompt format, seed, scale, reps or warm-up, and points that appear in
  more than one file. It lists every source under `meta.merged_from`. Pass the same `--max-model-len` to every part, and to the other
  engine, because each run otherwise sizes it from its own longest request. The merge and `--compare` warn when it
  differs. docs/bench-h100.md was made this way: each pod
  ran at most 2 hours.

Do not commit the per-run `results/` files. The report (`docs/bench-h100.md`) carries every number, and the raw
JSON stays on the machine that ran the jobs.

A full run takes roughly 30-60 minutes per engine per model, on top of the model download and the vLLM install: 4
workloads × 3 concurrencies plus 6 rate points, each repeated 3 times. To spend less GPU time, cut the sweep, for
example with `--workloads gate --concurrency 1,32 --rates ''` for the gate alone, as job 1 above does. Do not cut
the repetitions.

The Mac dry run uses the cached 0.5B model and takes a few minutes:

    uv run python bench_vs_vllm.py --engine fusezap --model Qwen/Qwen2.5-0.5B-Instruct --device METAL --quick

To run the harness against a server that is already up, pass `--base-url http://host:port`. To change how the server
starts, use `--server-arg=--flag` (repeatable) or `--server-cmd "... {port} {model}"`.
