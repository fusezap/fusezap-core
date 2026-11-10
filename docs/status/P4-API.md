# P4-API: OpenAI-compatible server (SPEC §11.3)

**Status:** done (Mac half, METAL; no CUDA code)

**Gate:** `uv run pytest tests/test_api_server.py -q`
**Result:** passed, 10 passed in ~12 s (warm kernel cache). Run with `-o addopts=""` to see pytest's summary line.

## Built
- `fusezap/serve/api.py`: starlette app (`create_app(engine, tokenizer, model_name, api_key=None)`).
  - Routes: `GET /health`, `GET /v1/models`, `GET /v1/models/{id}`, `POST /v1/chat/completions`, `POST /v1/completions`.
  - Stream (SSE, `data: [DONE]`, `stream_options.include_usage`) and non-stream; usage counts; `max_tokens` /
    `max_completion_tokens`; `stop` (up to 4 strings, with hold-back across token boundaries, aborts the engine request);
    `temperature` (default 1.0 as OpenAI), `top_p`, `top_k`, `seed`, `repetition_penalty`, `n`, list prompts.
  - OpenAI-format errors (400 / 401 / 404 / 500); context-length check; optional bearer `--api-key`.
  - `AsyncEngine`: the LLMEngine lives on one worker thread. Handlers enqueue add/abort commands; the thread drains them between
    `step()` calls so concurrent HTTP requests join the running batch (continuous batching) and posts each step's outputs to
    per-request asyncio queues. Client disconnect aborts the request and frees its blocks.
- `fusezap/serve/cli.py`, `fusezap/__main__.py`: `fusezap serve <model> --device --dtype --quant --port` plus `--host`,
  `--served-model-name`, `--api-key`, `--max-model-len`, `--max-num-seqs`, `--max-prefill-tokens`, `--num-blocks`,
  `--block-size`, `--beam`, `--no-warmup`. Runs as `uv run python -m fusezap serve ...`. Verified by hand with
  `examples/openai_client.py`.
- `tests/test_api_server.py`: real uvicorn on a free port, Qwen2.5-0.5B-Instruct on METAL, official `openai` client:
  chat non-stream == LLMEngine greedy text and usage; streaming deltas incremental, finish_reason, usage chunk; SSE wire format;
  completions (text, stream, token-id prompt); models/health; max_tokens, stop (stream and not); seeded sampling; error format;
  8 concurrent streaming requests each equal to alone-greedy, with decode steps well under serial (batched).

## Speed
Serving adds no engine work: throughput is the engine's (P4-SERVE: ~480 tok/s aggregate at 32 concurrent on M5 Max). Not
separately benchmarked over HTTP.

## Known gaps
- Chat `tools` are only rendered into the prompt; tool calls in the output are not parsed. No logprobs, no presence/frequency penalties.
- Prompts longer than the context are rejected, not truncated. Text prompts are encoded without special tokens.
- One engine per process; requests wait in the engine's queue (no HTTP-level backpressure).

## Review fixes
- Token-id prompts are range-checked against the vocab (400, param=prompt) before submit.
- top_p=0 is served as greedy; repetition_penalty<=0 is a 400 invalid_request_error.
- usage.prompt_tokens counts each prompt once regardless of n.
- Non-stream requests abort their generations when the client disconnects.
Regression tests for each in tests/test_api_server.py.
