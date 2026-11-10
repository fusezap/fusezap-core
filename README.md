# FuseZap Core

FuseZap Core is a small, compile-first inference engine for large language models. Models are built from lazy
tensors; the engine fuses and schedules the graph into kernels and compiles them for the device it runs on (Metal,
CUDA, NKI for AWS Neuron, or C for CPUs). On top of the engine, `fusezap serve` is an OpenAI-compatible HTTP server
with paged KV cache and continuous batching.

FuseZap is built by Techyodl. Website: https://fusezap.ai

> **Status: pre-release.** Nothing has been published to PyPI yet; the first release will be 0.1.0 (the source carries
> `0.1.0.dev0`). CLI flags, `FZ_*` environment variables and the Python API can still change before 1.0. See
> [CHANGELOG.md](CHANGELOG.md) and the release policy in [docs/releases/README.md](docs/releases/README.md).

## What it does

- **Models**: decoder-only text models with the Llama architecture, loaded from Hugging Face safetensors checkpoints:
  `model_type` `llama`, `qwen2` and `mistral` (for example Llama 3.x, Qwen2.5, Mistral). No vision-language, audio or
  mixture-of-experts models, and no sliding-window attention yet.
- **Image generation** (Preview): one text-to-image diffusion model, FLUX.2 [klein] 4B, served separately with
  `fusezap serve-image flux2-klein`. It is checked for correctness against a float32 reference image; it is not fast
  yet (see the matrix notes).
- **Backends** (`--device`):
  - `CUDA`: NVIDIA GPUs. NVRTC-compiled kernels, tensor-core GEMMs, flash attention, CUDA-graph decode, bf16 / int8 /
    fp8, tensor parallelism (`--tp N`).
  - `METAL`: Apple Silicon GPUs.
  - `CPU`: a C backend compiled with the system `clang`.
  - `NEURON`: AWS Inferentia2 and Trainium1 (NeuronCore-v2) through NKI kernels and the Neuron runtime. Needs the AWS
    Neuron SDK on an inf2 or trn1 instance.
- **Serving**: `/v1/chat/completions` and `/v1/completions` with streaming, tool calls (Hermes, Mistral and Llama 3.x
  formats), structured outputs (`response_format` JSON schema, with the optional `llguidance`), weight-only int8 /
  int4 / fp8 quantization and AWQ / GPTQ import, and Prometheus `/metrics`.

### Support matrix

Measured on real hardware. The source, with every number and its evidence, is
[docs/releases/matrix/sources.yaml](docs/releases/matrix/sources.yaml) (`python scripts/release_matrix.py
--no-evidence-check` renders it). Every cell is **Preview** until the first public release: it ran on that hardware in a development build, and the
source lists the gate it passed. **Supported** starts with the first release. A dash means "not run", not "does not work".

| Model | AWS Inferentia2 | AWS Trainium1 | NVIDIA H100 | NVIDIA H200 | NVIDIA RTX 4090 | Apple Silicon (Metal) | CPU |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | Preview | Preview | — | — | — | Preview | Preview |
| Qwen2.5-7B-Instruct | Preview | — | Preview | — | Preview | — | — |
| Qwen2.5-32B-Instruct | — | — | Preview | — | — | — | — |
| Qwen2.5-72B-Instruct | — | — | — | Preview | — | — | — |
| Llama-3.3-70B-Instruct | — | — | — | Preview | — | — | — |
| FLUX.2-klein-4B (image) | Preview | Preview | Preview | Preview | Preview | Preview | — |

Mistral checkpoints load through the same Llama code path and are covered by unit tests, but have not been measured
on hardware yet. Known gaps are in [CHANGELOG.md](CHANGELOG.md) under "Known issues".

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). The Metal backend needs macOS on Apple Silicon; the CPU
backend needs `clang`; CUDA needs an NVIDIA driver (kernels are compiled at run time with NVRTC).

```sh
git clone https://github.com/fusezap/fusezap-core && cd fusezap-core
uv sync --extra serve                  # add --extra metal on Apple Silicon
uv run fusezap serve Qwen/Qwen2.5-0.5B-Instruct --device METAL --port 8000    # or --device CUDA / CPU
```

Then, from another terminal, with the official OpenAI client:

```sh
uv run python examples/openai_client.py --base-url http://127.0.0.1:8000/v1
```

`uv run fusezap serve --help` lists every option (dtype, quantization, KV cache size, tensor parallelism). The first
start compiles and tunes the decode kernels; they are cached in `~/.cache/fusezap` (or `FZ_CACHE_DIR`), so later
starts are faster.

The engine is also a small tensor library:

```python
import fusezap as fz
x = fz.Tensor.rand(64, 64)
print((x @ x.T).relu().sum().numpy())
```

`python examples/gpt.py` trains a tiny character-level GPT with it.

## Running the tests

```sh
uv sync --extra serve
uv run pytest -n auto                  # the default suite on every device tier available here
FZ_SLOW=1 uv run pytest -n auto        # also the slow server and end-to-end tests
```

The default tiers are `interp`, `NPY`, `CPU` and (on a Mac) `METAL`; `FZ_TEST_TIERS=interp,CPU` restricts them, and the
hardware tiers `CUDA` and `NEURON` run only when `FZ_TEST_TIERS` names them. Tests whose hardware, optional package or
model checkpoint (in the local Hugging Face cache) is missing skip themselves. CI runs the `interp`, `NPY` and `CPU`
tiers on Linux.

## Documentation

- [docs/SPEC.md](docs/SPEC.md): architecture and design
- [docs/BENCH_METHOD.md](docs/BENCH_METHOD.md): how performance is measured against vLLM
- [docs/NKI_BACKEND.md](docs/NKI_BACKEND.md): the AWS Inferentia2 / Trainium1 backend
- [docs/quant.md](docs/quant.md): quantization
- [docs/status/](docs/status/): engineering logs and the measurement runs behind the numbers above

## FuseZap Invarix

FuseZap Invarix (`fusezap invarix`), the release-qualification tool that some docs and the support matrix refer to
("Invarix PROMOTE"), is a separate, commercially licensed product and is not part of this repository. Without it,
`fusezap invarix` prints a short "not installed" message; everything else here works without it.

The release-history stores that Invarix can index its reports into are open source and live here, in
[fusezap/invarix_stores/](fusezap/invarix_stores/): SQLite built in; DuckDB, PostgreSQL and MySQL as the extras
`invarix-duckdb`, `invarix-postgres` and `invarix-mysql`. To add another database, see
[WRITING_A_STORE.md](fusezap/invarix_stores/WRITING_A_STORE.md).

## Contributing, security and licence

- Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).
- Report vulnerabilities privately to **security@fusezap.ai** ([SECURITY.md](SECURITY.md)), never in a public issue.
- FuseZap Core is licensed under the [Apache License 2.0](LICENSE); see also [NOTICE](NOTICE). "FuseZap" and the
  FuseZap logo are trademarks of Techyodl and are not licensed under Apache-2.0.
