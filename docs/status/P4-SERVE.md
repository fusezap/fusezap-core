# P4-SERVE: paged KV cache + continuous batching (SPEC §11.3)

**Status:** done (Mac half, METAL; no CUDA code)

**Gate:** `uv run pytest tests/test_kv_cache.py tests/test_scheduler.py tests/test_serve_model.py -q && uv run python bench/bench_batch.py`
**Result:** passed. 60 passed, exit 0 (15 kv_cache + 11 scheduler + 34 serve_model). pyproject's `addopts = "-q"`
plus the gate's `-q` hides pytest's summary line, so the count comes from the progress dots. bench_batch exits 0 and
request 0 at batch 32 produced the same tokens as at batch 1. The rerun after the review fix measured 20.5 / 146.7 /
457.6 aggregate tok/s at batch 1 / 8 / 32.

**Review fix (high):** the prefill forward picked its logits rows only when `max_prefill_tokens != max_num_seqs`
(`select=self.R != self.N`). When the two settings were equal, finishing prompts got the logits of token rows 0, 1,
... instead of their last prompt tokens. Prefill now always selects (`select=self.prefill or self.R != self.N`). The
regression test `test_prefill_selects_rows_when_prefill_tokens_eq_num_seqs` (an engine with 8 / 8 and prompts of 5,
2, 7 and 1 tokens, compared to running each alone) fails without the fix and passes with it.

```
model Qwen/Qwen2.5-0.5B-Instruct, device METAL, float32, prompt 128, gen 128, decode BEAM 4
batch   agg tok/s  decode tok/s  p50 step ms  steps  seconds
    1        21.8          21.7        47.53    128     5.88
    8       154.8         163.7        48.30    131     6.62
   32       478.9         590.5        51.18    143     8.55
```
These are the best of two runs. In the gate run itself the GPU was shared and slower: 16.3 / 126.3 / 440.6 aggregate tok/s.
"agg" is generated tokens divided by wall time, with prefill included. "decode" counts only the steps that ran no
prefill. Measured on an Apple M5 Max.

## Built
- `fusezap/serve/kv_cache.py`
  - `BlockAllocator`: a LIFO free list that catches double frees.
  - `PagedKVCache`
    - Per layer it keeps one `(2, num_blocks, n_kv, block_size, hd)` pool, plus per-sequence block tables.
    - Bookkeeping: `ensure`, `can_ensure`, `blocks_needed`, `slot`, `block_table`, `free` and `reset`.
  - It implements the P4-MODEL KVCache interface (`state/inputs/bind/load/update`). `update()` scatters by block table
    and gathers back into a position-ordered view, so `Llama.prefill/decode` run on it unchanged. Batch rows are mapped
    to sequences with `set_rows`.
  - The ragged path:
    - `step_meta(N, R, tokens, sel)` packs one int32 metadata tensor (`StepLayout`). Per token it holds the block-table
      row, position and slot, followed by the logits-row selection and the block tables.
    - `attend(layer, qkv, Hq, scale)` writes k/v into their slots in place and runs paged attention.
    - `select(x)` is an exact row gather.
  - MSL kernels:
    - `fz_kv_write`, `fz_kv_write_rows` and `fz_kv_gather`.
    - `fz_rows`, the row gather.
    - `fz_paged_attn_{hd}_{G}`: one threadgroup per (token, kv head), one simdgroup per query head. Each lane scores
      one key of a 32-key chunk, the softmax is online, and accumulation is in float. Every token attends to keys
      0..pos of its own sequence, so the same kernel serves prefill and decode.
  - The kernels launch as kind="custom" ScheduleItems with a `ServeCall` payload. When a `fz.jit` capture is recording,
    they are appended to the recording, so a replay launches them in order. Params can be symbolic and are evaluated
    from the item's var_vals.
  - Other devices get numpy reference kernels, tested against METAL.
  - `lib_linear`: x @ w^T through P2-MTL's simdgroup GEMM, recorded the same way.
- `fusezap/serve/models/llama.py`: added `Llama.forward_tokens(x, cache, cos, sin, select, wide_linear)`. It is the
  ragged step: N tokens from any number of sequences, each at its own position, with RoPE rows supplied per token.
  Nothing else changed.
- `fusezap/serve/scheduler.py`: `Scheduler`, `SchedulerConfig`, `Sequence`, `SchedulerOutput`. The policy is
  first come, first served:
  1. Decode first: one token per running sequence past its prompt.
  2. Then chunked prefill of running prompts.
  3. Then admission of waiting requests in arrival order, under the token budget, `max_num_seqs` and the free blocks.
     The head of the queue is never skipped.
  - When blocks run out, the newest running sequence that is not yet scheduled in this step is preempted, and its
    output is recomputed later. The oldest sequence is never preempted. `add` rejects a request that cannot fit the
    pool on its own, so every request finishes in a bounded number of steps.
- `fusezap/serve/engine.py`
  - `LLMEngine(model, num_blocks, block_size, max_num_seqs, max_prefill_tokens, max_model_len, decode_beam)` with
    `add_request(prompt_ids, SamplingParams)`, `step() -> [RequestOutput]` (streaming deltas), `abort`, `generate` and
    `release`.
  - `SamplingParams` sets max_tokens, temperature, top_k, top_p, repetition_penalty, seed, stop_token_ids and
    ignore_eos. Sampling uses P4-MODEL's `Sampler`, with one per request.
  - A step runs up to two fixed-shape `fz.jit` forwards on the shared cache:
    - decode: `max_num_seqs` rows, BEAM-tuned composed kernels, with gate/up and lm_head on the simdgroup GEMM;
    - prefill: `max_prefill_tokens` rows on the library GEMM, returning the logits of the chunks that finish a prompt.
  - Neither forward has symbolic shapes. After capture they replay with no tracing.
- **Why outputs are exact.** Every kernel computes each row independently, and each forward has one fixed shape. On top
  of that, a given position is always computed by the same kind of forward:
  - prompt positions always by prefill, in any chunking;
  - generated positions always by decode;
  - after a preemption, generated tokens are recomputed one decode row per step, teacher forced, without resampling.
  So a request's tokens are bit-identical whether it runs alone, batched, chunked or preempted. The tests check this.
- `tests/test_kv_cache.py` (15 tests)
  - Allocator, block tables and the metadata layout.
  - **Paged attention equals contiguous attention:** fusezap's composed masked sdpa on contiguous K/V, in f32 and f16,
    over 4 head configs (GQA 14/2 hd 64, MHA, hd 40, hd 128). Each case has ragged decode, prompt, continued-chunk and
    one-token sequences with interleaved blocks. The test also checks slot contents, that padding rows are inert, and
    that free blocks stay untouched.
  - Paged attention vs torch.
  - The numpy reference vs METAL.
  - jit replay of `attend` with new metadata.
  - Llama.prefill/decode with a PagedKVCache (`update()`) vs ContiguousKVCache: same logits and the same 10 greedy
    tokens.
- `tests/test_scheduler.py` (11 tests)
  - Simulated scheduling with invariant checks after every step: disjoint blocks, budgets, no double scheduling.
  - Chunked prefill and admission under the budget, steps that mix prefill and decode, first come first served without
    skipping the queue head, late arrivals, the capacity check, and abort freeing blocks.
  - Preemption with no starvation: the oldest request is never preempted, and request i finishes within twice the
    serial work of requests 0..i.
  - **Gate, 32 concurrent greedy requests** (prompts of 1-300 tokens, 8-39 output tokens): each is token-identical to
    running alone. Streamed deltas concatenate to the final tokens. Alone also equals P4-MODEL's contiguous
    `Llama.generate` for 3 requests.
  - **Gate, small pool** (24 blocks = 384 slots for ~4k tokens of demand): preemption happens and all 32 requests
    complete with exactly their alone tokens, within the starvation bound.
  - Abort and stop tokens mid-run.
  - Seeded temperature/top-k/top-p/repetition-penalty sampling: batched equals alone.
- `bench/bench_batch.py`: aggregate tok/s at batch 1, 8 and 32. It exits non-zero on a token mismatch.

## Speed notes
- Decode step at 32 rows, GPU time per step measured with a wait after each kernel: about 30 ms. That breaks down as
  library GEMM (gate/up and lm_head) 18.8 ms, down_proj 4.8, qkv 3.1, paged attention 2.1, and the rest under 1 ms each.
  Wall time is about 48-60 ms per step: roughly 340 launches, each with its own command buffer (P4-MODEL issue 2).
- Before routing gate/up and lm_head to the GEMM, BEAM-tuned composed kernels at M=32 took 67 ms for lm_head alone,
  and 115 ms of GPU time per step. With hand opts (BEAM 0), a 32-row step took 727 ms.
- A cold `~/.cache/fusezap` pays a BEAM search for the 32-row decode kernels. It took about 20 minutes here, once.
- Batch 1 pays for the full 32-row decode shape: about 20 tok/s, against ~90 tok/s for P4-MODEL's batch-1 contiguous
  path. A smaller decode bucket would be faster but would break the bit-exact batch invariance, because different
  shapes get different BEAM kernels. It is not done.

## Known gaps
- Decode and prefill are two forwards within one step, not one fused ragged forward. This is deliberate: it keeps
  outputs exact (see above). It costs two sets of launches when both run.
- Preemption recomputes only; there is no swapping to host memory. Recomputed generated tokens take one decode step
  each.
- No prefix caching or block sharing. The allocator has no refcounts.
- The paged attention kernel is simple: serial over 32-key chunks, q in registers, and no K/V sharing through
  threadgroup memory across the G query heads. It is about 2 ms per step at 32 rows x ~200 positions.
- Only METAL has fast kernels. CPU/NPY use the numpy references.

## Interface issues (for the integrator)
1. There is no registry for out-of-tree custom kernels.
   - `kv_cache._install` wraps `dev.run` on the device instance, the same stopgap as `schedule.install_custom_kernels`,
     to route `ServeCall` items.
   - `run_call` appends items to `fusezap.jit._state["rec"]` (private) so that captures record them.
   - Proposed: a public `jit.record(si)` hook (or `realize.run_items`), plus `CustomKernel` with a launch hook that
     `CompiledDevice.run` dispatches.
2. There is no in-place / scatter store in the IR. The KV write is a custom kernel that writes into an input buffer
   (the pool) in place, which the jit tolerates because the pool is a jit input.
3. P4-MODEL issue 2 (one command buffer per launch) now dominates: wall time is ~1.6-2x the GPU time per decode step.
4. The composed matmul at M=32 is slow even after BEAM: a 32x151936x896 lm_head took 67 ms. The engine routes the wide
   projections to the library GEMM by hand (`lib_linear`). A size heuristic in the scheduler's matmul hook would do
   this generally.
