# I14: tool calling and structured output (JSON) in `fusezap serve`

**Status:** done on CPU / METAL / NEURON (simulated); proven earlier on CUDA by I61 / I62.

**Gate:** `uv run pytest tests/test_serve_structured.py tests/test_serve_openai_ext.py tests/test_serve_llama_tools.py -q -k "not real"`
(and `test_serve_constrain.py`, private: it uses Invarix's checker)
**Result:** 10 passed, 3 skipped (slow) in ~11 s; the other three files 37 passed, 1 skipped. Slow:
`FZ_SLOW=1 uv run pytest tests/test_serve_structured.py -k "real or cpu_mask" -s`: 3 passed in ~31 s (Qwen2.5-0.5B on
METAL, the tiny-model overhead on CPU).

## What the API does (I61, I62, and this item)
- `tools` / `tool_choice` (none, auto, required, a named function), `parallel_tool_calls`, with the format taken from
  the model's tokenizer and chat template: Hermes `<tool_call>` (Qwen2.5), Llama 3.1 / 3.2 / 3.3 JSON
  (`{"name": ..., "parameters": ...}`, optional `<|python_tag|>`), Mistral `[TOOL_CALLS]`. Calls come back as
  OpenAI `tool_calls` (arguments a JSON string), finish reason `tool_calls`; follow-up turns with assistant
  `tool_calls` and `tool` results render through the template.
- `response_format` `json_object` / `json_schema`, and tools with `"strict": true` under a named / required
  `tool_choice`: grammar-constrained decoding (llguidance, optional `serve` extra; a 400 without it or for a schema
  keyword it cannot enforce, never silently unconstrained). The output always parses and matches the schema.
- **New here: live Hermes streams.** Qwen2.5 tool streams were sent only once the whole reply was generated. Now
  content streams as it is generated up to `<tool_call>` (a partial tag and trailing whitespace held back), and the
  calls are held to the end of the choice and sent as tool_calls deltas (id, type and name, then the arguments), so
  the reassembled stream equals the non-streamed reply, including malformed blocks (content) and length cuts. Llama
  3.x streams already worked this way; Mistral stays buffered (its `[TOOL_CALLS]` is a special token the streamed
  text does not carry).

## Backends: the mask is on the host
The grammar state lives in the engine process and is applied to the step's logits row on the host
(`_Logits.token`: `sampler(cons.mask(row))`), exactly where temperature sampling already reads the row. Nothing in
the compiled forwards changes, so it is backend-agnostic:
- CUDA reads that one row (one vocab-wide copy); Metal reads it from shared memory; CPU / NEURON take the step's host
  copy of the logits.
- NEURON's fixed-shape decode is untouched: the same NEFFs (layers and head fused, device argmax) run; the
  constrained row ignores its device argmax and uses the logits the forward already returns. Unconstrained rows in
  the same step keep their device argmax. What a constrained request costs there: the logits read back for that
  step (R x V float32, e.g. 4.9 MB at 8 rows x 152k) and no lookahead / prefill pipelining while it runs (as for any
  sampled request). Not measured on a chip (no launches for this item).
- Checked: `test_constrained_on_nkishim_matches_cpu` (NKISHIM, N8 defaults) and
  `test_constrained_on_the_persistent_neuron_path` (the NEFF path with the fake runtime): a batch of two greedy
  unconstrained rows and two schema-constrained rows (greedy and seeded sampling) gives CPU's tokens exactly, and the
  constrained outputs are schema-valid.

## Overhead on CPU (host mask)
The mask is CPU work whatever the device, so it is measured on its own and in serving (15 Nov, Apple M5 Max, busy
with other jobs):
- llguidance mask for the Qwen2.5 vocabulary (151,936 ids) plus the -inf fill, a person JSON schema, 28 tokens:
  **p50 0.090 ms, max 0.62 ms per token**.
- CPU backend, tiny random model with the full Qwen2.5 vocabulary (the forward is small, so this is the worst case
  for the relative cost), one request: TPOT 5.41 ms unconstrained vs 5.49 ms with a JSON schema: **+0.08 ms/token
  (+1.5%)**.
- Qwen2.5-0.5B float32 on METAL over HTTP: TPOT 36.52 ms vs 36.63 ms with json_object: **+0.12 ms/token (+0.3%)**.
- Qwen2.5-0.5B float32 end to end on the CPU backend was not measured: on the busy Mac each request took more than
  the client's 600 s timeout (48 min for the run before it was moved to METAL), constrained or not.

## Tests added (tests/test_serve_structured.py, public: no Invarix import)
- `hermes_split` unit cases; stream = non-streamed reply for content, one call, text then two calls, a malformed
  block, a `<tool` look-alike in text; a length cut mid-call; named `tool_choice` with n=2.
- The official `openai` client (3.33) against the HTTP app with a fake engine and the Qwen2.5 tokenizer: tool calls,
  streamed tool calls accumulated as the client exposes them, the follow-up turn, 400 for an unknown named tool,
  `response_format` json_object / json_schema (plain and streamed), 400 for `uniqueItems`.
- NEURON: the two tests above.
- Slow, Qwen2.5-0.5B-Instruct on METAL (CPU when there is no METAL) with the `openai` client: json_schema (strict,
  plain and streamed) valid, valid when sampled (3 seeds; the schema bounds its strings: unbounded, the 0.5B model
  loops inside a string until max_tokens, valid JSON so far but cut), strict named and required tool calls valid against the tool schema,
  auto tool calls streamed = non-streamed; the overhead numbers above, and the CPU tiny-model one.
- The tiny tokenizer helper moved here from test_serve_constrain.py (which imports it back).

## Limitations
- `tool_choice: "auto"` is not constrained (as in vLLM): an unconstrained model can write invalid arguments, which
  are returned as content. Use strict tools with required / named for guaranteed calls.
- Calls are streamed per choice end, not argument-by-argument as they are generated.
- Mistral tool streams are still buffered.
- No chip run of a constrained request on NEURON in this item; the CUDA proof is I62's (Llama-3.3-70B and
  Qwen2.5-72B TP=2, json_schema / json_object 50/50).
