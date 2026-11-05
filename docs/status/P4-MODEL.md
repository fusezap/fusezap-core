# P4-MODEL: Llama-architecture model, safetensors weights, tokenizer, sampling (SPEC §11.3)

**Status:** done (Mac half; nothing CUDA-specific here)

**Gate:** `uv run pytest tests/test_serve_model.py -q && uv run python bench/bench_decode.py`
**Result:** passed. `34 passed in 15.30s` on METAL (FZ_DEVICE unset: METAL is the default), and bench_decode exits 0.

```
model Qwen/Qwen2.5-0.5B-Instruct, device METAL, batch 1, max_len 1024, decode BEAM 4
dtype       prefill  128 tok/s  prefill  512 tok/s   decode tok/s   ms/token  p50 ms  kernels/step
float32                   2613                3109           83.7      11.95    9.44           266
float16                   2875                4796          104.7       9.55    9.62           266
```
(gate run after the review fixes; the first gate run gave f32 100.3 / f16 104.6 decode tok/s. The f32 mean is pulled
up by a few slow steps: p50 is 9.44 ms)
Apple M5 Max, with the GPU shared with a busy Chrome GPU process. Decode is the mean over 128 greedy steps after a
128-token prompt, including the logits read-back and host argmax. Across runs the f32 decode rate ranged 65-100 tok/s
and f16 93-110 tok/s; see "Speed" below.

## Built
- `fusezap/serve/models/llama.py`
  - `LlamaConfig.from_hf(config.json)`: hidden/intermediate sizes, layers, heads, KV heads, `head_dim`, `rms_norm_eps`,
    `rope_theta`, rope scaling (`linear`, `llama3`; it also reads transformers 5's `rope_parameters`), tied embeddings,
    EOS ids, and an active sliding window (rejected when `max_len` exceeds it).
  - `Llama(cfg, weights, dtype, device, max_len, max_batch, prefill_chunk, beam)`. `Llama.from_pretrained(id_or_dir)`
    also merges the EOS ids from `generation_config.json`.
    - RMSNorm in float32 (as in HF). RoPE uses rotate-half with f32 cos/sin tables computed like HF. GQA folds the
      query heads onto their KV head, so K/V are never repeated. SwiGLU.
    - Qwen2 q/k/v biases (and o/mlp biases) are used when the checkpoint has them.
    - Fused projections: q/k/v are one matmul, gate/up are one matmul.
    - Token embeddings are gathered on the host from the mmap'd table (no one-hot matmul). The tied `lm_head` is on
      the device.
  - `forward(x, cache, start, T, last_only, last, hiddens)` is the traced step. `start` and `T` can be ints or Vars.
    `hiddens` collects HF's `hidden_states`.
  - `prefill(ids, start=None, chunk=None)`
    - Splits the prompt into chunks, pads each to a power-of-two bucket (16..prefill_chunk) and runs that bucket's
      `fz.jit` step, with a symbolic `start` and a symbolic index for the logits row.
    - Shapes are concrete per bucket, so the linear layers can use the Metal library GEMM (f32), and continued
      prefill (multi-turn, `start > 0`) works.
    - Padding rows write cache rows past the prompt. Decode rewrites those rows before anything reads them.
  - `decode(ids, start=None)`: one `fz.jit` step with a symbolic `start_pos`. After two calls it replays: the tests
    check that the traced fn is not run and nothing is lowered.
  - `generate(ids, n, sampler, stop)` is the batch-1 generator.
  - Kernel settings while a step compiles (`_opt_env`):
    - Decode uses `FZ_CUSTOM=0` plus `FZ_BEAM=beam` (default: the `FZ_BEAM` env var), or the hand-coded opts
      (`FZ_OPT=1`) when beam is 0. At M=1 the simdgroup GEMM uses 14 of its 128x64 tiles for N=896. It measured
      67 ms/token, against 10 ms with BEAM-tuned composed GEMV.
    - Prefill uses `FZ_CUSTOM=1` and the hand-coded opts.
- `fusezap/serve/weights.py`
  - A numpy safetensors reader: the header is parsed and each tensor is a zero-copy view of one read-only `np.memmap`.
    BF16 tensors come back as uint16 bits tagged `st_dtype="BF16"`, and the tag survives slicing and fancy indexing.
  - Sharded checkpoints via `model.safetensors.index.json`, or every `*.safetensors` in the directory.
  - `to_dtype` converts bf16 to f32/f16/bf16 host formats. `f32_to_bf16` rounds to nearest even.
  - `resolve_model`: a local directory, or an HF repo id through the HF cache (`local_files_only` with allow_patterns,
    because a plain snapshot lookup fails on this cache, which is missing README/LICENSE). It downloads only with
    `FZ_HF_DOWNLOAD=1` or `allow_download=True`. It fetches metadata first (top level only), then only the shards
    `model.safetensors.index.json` lists (or every top-level `*.safetensors` without an index). `consolidated*` and
    subdirectories such as `original/` are never fetched, so Mistral repos download ~14.5 GB, not ~29 GB.
- `fusezap/serve/tokenizer.py`
  - `Tokenizer` wraps `tokenizers.Tokenizer` (tokenizer.json) and reads EOS, BOS and PAD from tokenizer_config.json.
    `stop_token_ids` merges in generation_config's EOS list.
  - `apply_chat_template` / `render_chat`: the ChatML template is rendered in Python.
    - It covers Qwen2.5 completely: default system prompt (read from the template), tools block, assistant
      `tool_calls`, grouped `tool` responses, late system messages and the generation prompt.
    - The Python renderer is used only for Qwen2.5's template (detected by its tool markers). Every other template,
      including other ChatML ones such as SmolLM2's, renders through jinja2 in HF's apply_chat_template environment:
      ImmutableSandboxedEnvironment, trim/lstrip blocks, `loopcontrols`, a `{% generation %}` tag, HF's `tojson`,
      `raise_exception` and `strftime_now` (so Llama 3.2/3.3 get today's date), and all special tokens as variables.
    - The template is read from `chat_template.jinja` when the model ships one (as HF does), else tokenizer_config.json.
      special_tokens_map.json is merged in.
    - Without jinja2: other ChatML templates use a plain ChatML renderer (the default system prompt is parsed from the
      template, including literals with real newlines) that refuses tools; plain Llama 3.0 templates use a header
      renderer.
    - Tested against HF apply_chat_template (with and without tools) for SmolLM2-135M, Llama 3.0/3.1/3.2 and
      Mistral v0.3 tokenizers when they are cached, plus cache-free unit tests.
  - `IncrementalDecoder` emits text deltas for streaming and never splits a UTF-8 character.
- `fusezap/serve/sampling.py`
  - `Sampler(temperature, top_k, top_p, repetition_penalty, seed)` applies, in order: repetition penalty, temperature,
    top-k, top-p, softmax, then a seeded `numpy.random.Generator` draw.
  - Greedy when temperature == 0 or top_k == 1, with the first index winning ties.
  - `probs()` exposes the distribution. The order and semantics match HF's processors and warpers (tested against
    them).
- `fusezap/serve/models/__init__.py`: `load_model(id, dtype, device, **kw)`, plus the `model_type` registry
  (llama, qwen2, mistral).
- `tests/test_serve_model.py` (26 tests; the model tests run on FZ_DEVICE, default METAL)
  - **Gate:** 64 greedy tokens of a fixed chat prompt ("Write a short poem about the sea.") are token-identical to HF
    transformers (float32, torch CPU, plain greedy loop with KV cache).
  - **Gate:** per-layer logits close to HF. For each of the 25 hidden states, the logit lens (final norm + lm_head)
    is checked at atol 1e-3, the final logits at atol 1e-3, and the hidden states at rtol 1e-4, atol 1e-3 + 2e-5·max|h|.
    Measured: final logits max error 5e-4.
  - Jitted prefill and decode logits teacher-forced vs HF (atol 1e-3). Measured about 5e-5.
  - Chunked prefill (chunks of 16, several buckets) and a second-turn prefill at start > 0 match HF.
  - Decode replay does no tracing and no lowering.
  - Greedy with repetition_penalty=1.1 matches HF's processor end to end (24 tokens).
  - f16 top-1 agrees with HF f32 at >= 95% of teacher-forced positions.
  - Config parsing and RoPE (default and llama3) match HF.
  - Safetensors reader: shards and index, f16, read-only mmap. bf16 conversion and rounding vs torch. Model tensors
    vs `safetensors.torch`.
  - Chat template vs `apply_chat_template` on 4 conversations x tools on/off x generation prompt on/off, as text and
    as ids. Encode/decode round trip and streaming decoder vs the HF tokenizer.
  - Sampling: greedy and ties, temperature distribution (empirical, seeded), seeded reproducibility and reseed,
    top-k and top-p kept sets vs HF warpers, the full warper chain distribution vs HF, sampled ids stay in the kept
    set, repetition-penalty signs.
- `bench/bench_decode.py`: prefill (128 and 512 tokens) and decode tok/s at batch 1 for f32 and f16 on METAL.

## KV-cache interface (for P4-KV)
The model reads and writes the cache only through this interface, so a paged cache replaces `ContiguousKVCache`
without any change to the model:

```python
class KVCache:
  def state(self) -> list[Tensor]      # every tensor the step reads or rewrites; passed into the jitted step,
                                       # and the new versions come back as its outputs
  def inputs(self) -> list[Tensor]     # extra per-step inputs (e.g. block tables, slot mappings); [] here
  def bind(self, state, inputs=None) -> "KVCache"   # a view over these tensors that the traced step mutates lazily
  def load(self, state) -> None        # adopt the step's output state
  def update(self, layer, k, v, start, T) -> tuple[Tensor, Tensor]
    # k, v: (B, Hkv, T, hd), already rotated, for positions [start, start+T); start and T are ints or Vars.
    # Returns K, V: (B, Hkv, Sk, hd) with Sk >= start+T, where row j holds position j for j < start+T.
    # Rows at or past start+T must be finite; the model masks keys whose position is after the query's position.
```
- The jitted steps call `self.cache.bind(state, inputs)` and return `cache.state()`. `Llama.prefill` and `decode` call
  `cache.load(outputs)`.
- A paged cache would keep block pools in `state()`, put block tables in `inputs()` (fixed-size, padded, so the jit
  input shapes stay fixed), scatter k/v into blocks in `update`, and gather the sequence's blocks into a
  position-ordered (B, Hkv, Sk, hd) view, with Sk = max blocks per sequence x block size (concrete).
- `ContiguousKVCache` keeps one (2, B, Hkv, L, hd) tensor per layer, with L = max_len + prefill_chunk of slack for
  padded prefill buckets.
  - Decode writes with `where(arange == start, k broadcast, old)`. Prefill writes with a one-hot (L, T) matmul
    scatter, which is exact.
  - `full_read=True` (the default) returns all L rows. All reduce lengths are then concrete, so BEAM can parallelize
    them. With the symbolic `start+1` read, attention's serial loops were 45% of decode GPU time at position 400:
    9.1 -> 5.3 ms/token f32 GPU time, and 512-token prefill went from 1.1k to 3.0k tok/s.
- The model owns the causal mask (key position <= query position) and RoPE. The cache sees already-rotated k.
- Batching: `max_batch` sequences share positions. Ragged batches belong to P4-SCHED with a paged cache, which then
  needs a per-sequence `start`. Make the start a tensor input (positions per row) instead of one Var, and build the
  mask from it.

## Speed (METAL, Apple M5 Max)
- Decode f32 is 266 kernels/step (11 per layer + head). Each kernel's GPU time measured alone (wait after each)
  sums to 5.3-6.0 ms/token at positions 30-400. The weight GEMVs reach ~570 GB/s on the lm_head.
- The wall-clock step is ~9-10 ms in short runs, and 13-17 ms after about 1.5 s of sustained decode.
  - The jump comes with time, not position: it shows at a fixed position too.
  - Allocation does not grow. Disabling GC or adding an autorelease pool does not help.
  - Per-step GPU busy time (summed from the command buffers' GPU timestamps) rises from about 7 to about 14 ms. So
    the kernels run slower on the GPU itself (clock or power state, and the GPU was shared with Chrome), not in Python.
- Host cost per launch is about 32 us in `MetalProgram.__call__` (pyobjc: a new command buffer and encoder per
  kernel), so 266 launches take about 8.5 ms. Decode is therefore launch-bound even when the GPU is fast.
- Prefill (jit replay of a padded bucket): f32 2.7k tok/s (128) and 3.0k tok/s (512); f16 2.9k and 3.9k tok/s.
- The first run on a cold `~/.cache/fusezap` pays a BEAM search of a few minutes per dtype for the decode kernels.
  `--beam 0` skips it (hand opts: measured 22 ms/token, 45 tok/s, f32).

## Known gaps
- Batch > 1 is supported only as equal-length sequences sharing positions (`max_batch`). Continuous batching is P4-SCHED.
- No sliding-window attention. Mistral v0.1-style windows are rejected when `max_len` exceeds the window. Mistral
  v0.3 and Qwen2.5 do not enable it.
- f16 runs as composed kernels. The P2-CUSTOM hook only matches f32 matmuls, so half prefill does not use the native
  half simdgroup GEMM.
- bf16 works through the same code path (weights are kept in bf16 bits) but is not in the gate or the bench.
- The contiguous cache rewrites all L rows every decode step (about 0.1 ms/token at L=1536 for this model). A paged
  cache with an in-place scatter would avoid that, but needs an in-place store op.
- Sampling and argmax run on the host (logits read back each step, 600 KB).

## Interface issues (for the integrator)
0. jinja2 is not declared in the `serve` extra. It is installed here only transitively (torch/transformers, dev).
   Every non-Qwen2.5 chat template needs it to match HF exactly. Proposed: add `jinja2>=3.1` to `serve`.
1. `fusezap.helpers.getenv` is `lru_cache`d, so `FZ_CUSTOM` (read through it in `schedule._device_kernels`) cannot be
   toggled at run time. `lower.run_compiled` reads `FZ_BEAM`/`FZ_OPT` uncached. The model clears `getenv`'s cache
   around its compile calls. Proposed: a per-call or context option for custom-kernel matching and opts, instead of
   process env vars.
2. Metal per-kernel launch costs ~32 us of host time (a new command buffer and encoder per `MetalProgram.__call__`),
   which is more than the ~20 us average GPU time of a decode kernel. Encoding the jit replay's launches into a few
   command buffers, with fewer pyobjc calls per dispatch, would move decode from launch-bound to GPU-bound. A quick
   experiment with 16 dispatches per command buffer gave ~12 vs ~15 ms/token in sustained decode.
3. P2-CUSTOM's matmul hook prefers the simdgroup GEMM even at M=1, where it is 6x slower than a tuned composed GEMV.
   A size heuristic in `_mm_spec` (skip when M*Bt is small) would remove the need for the model's `FZ_CUSTOM=0`
   toggle. It also only matches f32, so f16 prefill cannot reach the native half GEMM.
4. `fusezap.jit._programs()` keys cached programs by (AST, renderer, device), and the most recent one wins. If the
   same AST was compiled under different opts (e.g. eager hand opts, then BEAM), replay uses whichever came last.
   That is correct but not necessarily the tuned program.
5. There is no in-place / scatter store, so a KV-cache update rewrites the whole cache tensor. A paged cache
   (P4-KV) will want `Tensor.assign` into a view, or a scatter op (a SPEC change).
