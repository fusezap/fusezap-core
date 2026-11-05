# P2-OPT: kernel optimizer (Opts + FZ_BEAM)

**Status:** done

**Gate:** `uv run pytest tests/test_opts.py && FZ_DEVICE=METAL uv run python bench/bench_opt.py`
**Result:** passed. `37 passed in 921.25s` (slow because the GPU was shared with other suites running at the same time), then `GATE PASS`:

```
     case                  unopt ms   hand ms   beam ms  speedup  beam opts
gate matmul 1024^3            1.324     0.319     0.319     4.1x  upcast (0,8),(2,4) unroll (1,8) local (2,16),(0,4)  6722 GFLOP/s
gate matmul 2048^3           13.358     4.051     3.883     3.4x  same                                              4425 GFLOP/s
gate sum(-1) 512x32768        0.903     0.114     0.106     8.5x  group 32, unroll 32, local 4, upcast 2            634 GB/s
gate sum() 2^24             294.938    10.085     5.781    51.0x  group 32, unroll 32                               12 GB/s
info sum(-1) 4096x4096        0.194     0.105     0.102     1.9x  (unoptimized is already near bandwidth)
info sum(0) 4096x4096         0.126     0.126     0.106     1.2x  (coalesced already)
```
The machine is shared and timings are noisy. Across runs the unoptimized matmul ranged 1.3-2.9 ms at 1024 and 13-29 ms
at 2048, and the speedup stayed at or above 3.0x. The "info" cases are wide-output reduces, where the unoptimized kernel already
coalesces or saturates bandwidth. 3x is not reachable there.

Extra checks (not part of the gate), run with this lower.py on HEAD:
- `FZ_TEST_TIERS=CPU,METAL pytest tests/test_ops.py tests/test_integration.py tests/test_nn.py`: all passed (default path unchanged).
- The same suites plus `tests/test_gpt_parity.py` with `FZ_OPT=1` (hand-coded opts on every kernel): all passed.
- `tests/test_codegen.py tests/test_metal.py`: passed.

## Built
- `fusezap/codegen/lower.py`
  - `Opts(upcast, unroll, local, group_reduce)`. Axes index the kernel's full shape F. An empty `Opts()` gives exactly the MVP lowering, auto-local split included.
    - **upcast (output axis, n):** each thread computes n adjacent outputs. There is one accumulator per upcast combo, and loads are CSE'd across the combos, so a matmul's A/B loads are shared in registers.
    - **unroll (reduce axis, n):** the reduce body runs n times per loop iteration. Accumulator updates are emitted without CSE, so a value that does not depend on the reduce index is still added n times.
    - **local (output axis, n):** an explicit threadgroup dim (`lidx0..2` in the order given). GPU only. It turns off the automatic gidx0 local split.
    - **group_reduce=32 (METAL):** the 32 lanes of one SIMD group split the innermost reduce axis with coalesced indexing (`r*(u*32) + k*32 + lane`). Partials are combined with `simd_sum`/`simd_max`. For float MAX, NaN is counted separately so it still propagates. Lane 0 stores the result. It takes `lidx0`, and local opts follow it.
  - `check_opts` raises `OptError` for: illegal axes, non-divisors, symbolic axes, repeats, amounts < 2, local dims on CPU, group_reduce other than 32 on METAL, more than 3 local dims, threadgroups over 1024 threads, upcast products over 64, unroll products over 32, and dtypes other than f32/f16/bf16/i32 for group_reduce.
  - `opt_actions` / `apply_action`: the BEAM move set is upcast {2,3,4,8}, unroll {2,4,8}, local {2..32} and group_reduce. Amounts multiply into an existing entry for the same axis.
  - `heuristic_opts` (hand-coded, `FZ_OPT=1`, also a BEAM seed):
    - group-for-reduce for wide reduces when the reduce axis is contiguous and there are at most 64K outputs, or when there are at most 1024 outputs;
    - for matmul-like kernels, the largest register tile that leaves at least 16K threads, plus unroll and locals 16x4.
  - `time_opts`: compile, launch about 2 ms of warm-up work, then take the best of `reps` timed launches (GPU time).
  - `beam_search(device, ast, bufs, var_vals, width)`, used by `FZ_BEAM=n`:
    - It starts from `Opts()` and the hand-coded opts, keeps the best `width` per round, and stops when a round brings no improvement (at most 8 rounds).
    - It then re-times the finalists in 3 interleaved rounds before it picks a winner, because timings on a shared GPU are noisy.
    - Outputs go to scratch buffers, so the real outputs are never touched.
    - The winner is cached as JSON at `CACHE_DIR/beam/<hash(version, device, renderer, width, canonical AST)>`. A cached entry is validated again on load: an invalid one triggers a new search. `FZ_NOCACHE` skips the read.
  - `run_compiled` reads `FZ_BEAM` and `FZ_OPT` from `os.environ` at call time, and its in-memory program cache key includes both. Defaults are unchanged: no env means unoptimized.
- `tests/test_opts.py`: every opt alone and in combination on matmul, sum/max/mean on the last and first axes, multi-axis and full reduces, NaN max, reduce + epilogue, softmax-style elementwise, broadcast (CSE hazard), padded/flipped masked views, f16/int32 sums, a 50-case random opt-sequence fuzz, validity rejections, symbolic axes, JSON round trip, heuristics, and beam search plus the disk cache. Each optimized kernel is compared against the unoptimized kernel on the same inputs and against numpy. All tests run on CPU (upcast/unroll) and METAL (all opts).
- `bench/bench_opt.py`: unoptimized vs hand-coded vs BEAM, GPU time, interleaved best-of-rounds, and error against a float64 reference.

## Interface issues
- The UOp spec (`codegen/uops.py`) has no cross-thread op: no threadgroup memory, barrier or SIMD reduction. group_reduce works around this inside lower.py: it emits `ALU` UOps whose arg is the string `"simd_sum"`/`"simd_max"`, and it renders with a copy of the Metal renderer whose `code_for_op` has those two entries. This is outside the UOp contract. A proper `UOps.SIMD_REDUCE` (or DEFINE_LOCAL + BARRIER) in the spec would remove the hack. It would also allow threadgroup-wide (more than 32 lanes) reductions, which a full reduction to a scalar needs to approach bandwidth: it is 51x faster than unoptimized but still only about 12 GB/s.
- There are no vector types in UOps (float4 loads), which caps matmul at about 4-6 TFLOP/s here.

## Known gaps
- The heuristics are opt-in (`FZ_OPT=1`), and BEAM is opt-in (`FZ_BEAM=n`). Unoptimized remains the default.
- Symbolic axes are never split. A reduce over a symbolic axis can only use group/unroll if the axis is static.
- The upcast layout is contiguous per thread only; there is no strided upcast option.
