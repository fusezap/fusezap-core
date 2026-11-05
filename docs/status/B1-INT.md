# B1-INT: Run B integration, Mac half (Phase 4 on METAL)

**Status:** done (integration round 1)

**Gate:** `FZ_SLOW=1 uv run pytest -n 6` (all tiers: interp, NPY, CPU, METAL; includes every serve test and the CLI
end-to-end test) plus `uv run fusezap serve Qwen/Qwen2.5-0.5B-Instruct --device METAL --port <free>` answering a
streaming request from the official openai client.
**Result:** passed. `5739 passed, 144 skipped in 588.68s (0:09:48)` (the 144 skips are test_ops' "axis out of range"
cases). The server, started by hand through the console script, answered `examples/openai_client.py` (models, chat,
stream, completion); `tests/test_cli_e2e.py` does the same for both `fusezap serve` and `python -m fusezap serve`
(2 passed, 8 s).

## What changed
- `pyproject.toml` / `uv.lock`: `[project.scripts] fusezap = "fusezap.serve.cli:main"`; `jinja2>=3.1` added to the
  `serve` extra (needed for HF-exact chat templates other than Qwen2.5's); pytest `addopts = "-q"` removed, so a gate's
  `-q` no longer becomes `-qq` and hides the summary line.
- `fusezap/jit.py`: public hooks for out-of-tree kernels: `run_item(si)` runs a ScheduleItem and records it into an
  active capture; a `kind="custom"` item whose `ast[0]` has `launch(dev, bufs, var_vals)` is launched through it, at capture
  and at replay. `current_var_vals()` returns the bound symbolic values.
- `fusezap/serve/kv_cache.py`: `ServeCall.launch`; `run_call` and `lib_linear` use `jit.run_item`. The `dev.run`
  monkeypatch (`_install`) and all reads of `jit._state` are gone.
- `tests/conftest.py`: `slow` marker semantics: skipped unless `FZ_SLOW=1` or `-m slow`. Marked slow: the LLMEngine
  tests in `tests/test_scheduler.py` (6), the uvicorn server tests in `tests/test_api_server.py` (13) and
  `tests/test_cli_e2e.py` (2). On a cold kernel cache those take ~9 min of BEAM tuning; warm, under a minute.
- `tests/test_cli_e2e.py` (new): console script and `python -m fusezap` in a subprocess, official openai client,
  streaming chat equals the non-stream answer.
- `tests/test_metal_kernels.py`: `test_attention_direct` bf16 kernel-vs-composed bound 1e-2 -> 2e-2. It flaked once
  under xdist (2 of 24576 elements off by 0.0146): the module RNG makes inputs depend on test order, and the composed
  bf16 path rounds its own intermediates (~2 ulp apart near |x|=1). The comparison to the f32 reference keeps 1e-2.
- `docs/SPEC.md` §11.3: console script, engine/tokenizer modules, FP8 as built (uint8 e4m3 codes, weight-only; native
  dtypes + CAST scale + FP8 GEMM deferred to P3-GEMM on H100), the jit hooks, and the slow marker.
- The project status log: Run B section (packages, commits, gates, numbers, open gaps, waiting-for-H100 list).

## Measured (Qwen2.5-0.5B-Instruct, METAL, M5 Max, decode BEAM 4)
- bench_decode, batch 1: f32 137.4 tok/s (p50 7.28 ms), f16 149.2 tok/s (p50 6.67 ms); prefill 512: f32 4821, f16 5323 tok/s.
- bench_batch, f32: aggregate 51.4 / 384.1 / 1140.8 tok/s at batch 1 / 8 / 32 (decode-only 51.8 / 409.0 / 1426.7);
  request 0 at batch 32 == batch 1.
- Perplexity (from P4-QUANT's bench_ppl, not rerun): f16 13.2014, int8 13.1729 (-0.22%), fp8 13.2692 (+0.51%),
  int4 asym 15.7149 (+19.0%), int4 sym 17.6320 (+33.6%).
These are faster than the package gate runs (which shared the GPU with other agents).

## Known gaps / not resolved here
- One command buffer per launch still dominates decode wall time; no in-place scatter store in the IR.
- No native float8 dtypes yet (deferred to the H100 half with the FP8 GEMM).
- Act-order GPTQ uses a one-hot gather (needs a core gather op); `import_checkpoint` placeholder-weight workaround
  (needs a `Llama.from_weights` hook); no calibrated AWQ/GPTQ checkpoint cached to test against.
