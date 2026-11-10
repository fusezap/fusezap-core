# P2-MTL: Metal custom kernels (SPEC §11.1)

**Status:** done

**Gate:** `uv run pytest tests/test_metal_kernels.py && uv run python bench/bench_metal.py`
**Result:** passed. `108 passed in 1.87s`. bench_metal.py exits 0: `gate: float32 matmul 4096^2 at 0.85x torch MPS (need >= 0.60): PASS`.
Across runs, 4096 f32 has measured 0.79-0.88x since the tile tuning.

Also run, with `FZ_TEST_TIERS=METAL`: `test_ops`, `test_nn`, `test_gpt_parity`, `test_jit`, `test_custom`, `test_metal`,
`test_integration`, `test_opts`. All pass except 4 tests in P2-OPT's `tests/test_opts.py`: `test_matmul_opts`,
`test_random_opts_fuzz`, `test_heuristic_opts_valid_and_correct` and `test_beam_search_and_cache`. Their `kernel()`
helper asserts that `a @ b` schedules as a composed `kind="kernel"`, and on METAL it is now a custom item. They pass
with `FZ_CUSTOM=0` (see interface issues).

## Built
- `fusezap/codegen/metal.py`: MSL generators for the library kernels.
  - `render_matmul(name, dt, out_dt, tile, group)`: batched GEMM with `simdgroup_matrix` 8x8 fragments and float32
    accumulators. Inputs can be float32, float16 or bfloat16, and so can the output. The default tile is 4x2 simdgroups,
    each owning 4x4 fragments, so one threadgroup computes a 128x64 block with 256 threads. Threadgroups walk the
    blocks in bands of 8 block rows for cache reuse.
    - A simdgroup whose tile lies inside the output, with operands that have a unit stride on either axis, loads its
      fragments straight from device memory (transposed `simdgroup_load` for the other orientation).
    - Edge tiles, the K tail (K % 8) and general strides stage zero-padded tiles through threadgroup memory.
    - Full tiles with a unit-stride float output row use `simdgroup_store`. Everything else stores element by element
      with guards.
    - The code is unrolled in Python. With an indexed array of fragments, the 4x4 tile fell to under 1 TFLOPS, which
      points to the accumulators going to stack memory.
  - `render_attention(name, D, Dv, dt, out_dt, causal)`: flash attention with an online softmax (running max and sum
    per row, accumulator rescaled per key block).
    - Each simdgroup owns 8 query rows (32 per threadgroup). Per block of BK keys, the threadgroup stages K and V in
      threadgroup memory: zero padded, any strides, converted to float.
    - QK^T and P@V use `simdgroup_float8x8`. The softmax runs on 4 lanes per row with `simd_shuffle_xor`. Rows are
      rescaled with a diagonal-matrix multiply.
    - `causal` masks s > l (torch's top-left `tril`, the same as the composed `is_causal`) and skips key blocks past
      the threadgroup's last row.
    - BK is 32, 16 or 8, the largest that fits threadgroup memory (`attention_block`). Head dims that fit no block
      (e.g. D=160, Dv=136) fall back to `render_attention_rows`, one thread per row.
    - `render_attention_rows(name, Dv, ...)` keeps O in float32 registers (`ATTN_ROWS_CHUNK`=64 columns per pass)
      and writes the output dtype once at the end. An earlier version accumulated in the f16/bf16 output buffer, which
      gave a max error of 1.41 at bf16 S=2000; regression test `test_attention_rows_long_keys_float_accum`.
    - NaN propagates like the composed MAX. A key block that is masked for every query row it covers is a no-op
      (alpha=1, p=0).
  - `matmul_launch`, `attention_launch`: grids. Batch, block row and block column all ride on x, because y and z are
    limited to 65535.
- `fusezap/device/metal.py`
  - `METALDevice.custom_kernels`: `{"matmul": fz_matmul, "attention": fz_attention}`, offered to the P2-CUSTOM
    scheduler hook. They are on by default, and `FZ_CUSTOM=0` disables them. The "attention" source is the generator,
    because it depends on the head dims. It is rendered per call.
  - `METALDevice.run`: executes `kind="custom"` items itself through `run_custom_call(name, params, bufs)`. That call
    picks the source from the kernel name and the buffer dtypes, caches one program per (name, dtypes, head dims) on
    the device, and launches the right grid. `schedule.run_custom` would launch a single thread. Timing follows
    `run_compiled` (`wait` when FZ_DEBUG>=1). `fusezap/jit.py` replays custom items through `dev.run`, so replay works
    too (tested).
  - Direct entry points for what the scheduler does not match:
    - `matmul(a, b)`: float32, float16 or bfloat16; batch dims broadcast; output in the input dtype.
    - `attention(q, k, v, causal=False, scale=None)`: same dtypes and broadcasting, causal or not.
    - Both realize their inputs as contiguous METAL buffers and return a realized Tensor. They are inference only, with
      no autograd history.
- `tests/test_metal_kernels.py`: 108 tests. Each kernel is checked against the composed equivalent (the same
  expression with `FZ_CUSTOM=0`) and against torch, at the SPEC §10 tolerances.
  - Hook-path matmul (17 forms): edges, K tail, batched, broadcast, transposed A/B/both (direct transposed loads),
    `linear`, strided slices, vec/mat, K=1, chained with epilogues, a 1024x768x1000 matmul. `broadcast_a` is
    `(70,45)@(4,45,100)`, where the M axis is split by a batch axis; it checks the composed fallback.
  - Backward, `FZ_CUSTOM=0`, and jit replay.
  - Hook-path attention (7 shapes): L/S not multiples of the block, L != S, Dv != D, head dims not multiples of 8,
    D=128, and the per-row fallback. Also the explicit softmax form and large-score stability.
  - Direct matmul in f32/f16/bf16 over 8 shapes, plus non-contiguous inputs.
  - Direct attention: 7 shapes x {f32, f16, bf16} x {causal, full} vs composed sdpa(is_causal) and torch. Also an
    explicit scale with broadcast batch dims. Per-row fallback with long key sequences (S=500-3000) in f16/bf16,
    causal and full, with offset V, vs torch (8 cases).
- `bench/bench_metal.py`
  - Matmul at 1024, 2048 and 4096 in f32, f16 and bf16, plus flash attention (8 heads x 2048 x 64, f32/f16, causal
    or not), all vs torch MPS, with the ratio printed.
  - Both sides are timed the same way: about 50 ms bursts with an idle gap before each, best of interleaved rounds.
    The GPU is shared, and sustained load lowers its clock, which penalized whichever side ran second.
  - Every result is checked against torch's. The script exits non-zero if the f32 4096 ratio is below 0.60 or any
    result is wrong.

## Bench (Apple M5 Max, torch 2.14.1, GPU shared with other builders; numbers vary run to run)
```
case                          fusezap TFLOPS  MPS TFLOPS   ratio  max rel err
matmul float32 1024^2                   1.67        3.30    0.50      0.0e+00
matmul float32 2048^2                   8.78       11.99    0.73      0.0e+00
matmul float32 4096^2                   9.46       11.14    0.85      0.0e+00  GATE ok
matmul float16 1024^2                   7.39       21.56    0.34      3.8e-04
matmul float16 2048^2                   5.25       22.03    0.24      5.1e-04
matmul float16 4096^2                   8.32       47.09    0.18      7.0e-04
matmul bfloat16 1024^2                  4.97       30.79    0.16      1.4e-03
matmul bfloat16 2048^2                  4.52       22.54    0.20      2.1e-03
matmul bfloat16 4096^2                  7.44       41.33    0.18      2.9e-03
attn float32 8x2048x64                  1.87        7.90    0.24      4.4e-07
attn float32 causal 8x2048x64           1.99        9.43    0.21      1.0e-07
attn float16 8x2048x64                  2.22       41.90    0.05      5.0e-04
attn float16 causal 8x2048x64           1.96       21.06    0.09      3.2e-04
```
At 1024, both sides are dominated by per-launch overhead (0.2 ms kernels).

## Known gaps
- The scheduler hook only matches float32 and non-causal attention (P2-CUSTOM's `_mm_spec` / `_match_attention`).
  - A half/bf16 `a @ b` therefore still runs through the hook, but as cast kernels plus the float32 simdgroup matmul.
    It is correct, with extra memory traffic.
  - Causal sdpa runs as two hook matmuls plus composed softmax kernels.
  - The native half/bf16 and causal kernels are reachable through `fusezap.device.metal.matmul` / `attention`. See
    interface issues.
- On this M5 Max, MPS runs float16/bfloat16 GEMM at 45-60 TFLOPS, apparently on the GPU neural accelerators.
  `simdgroup_matrix` reaches about 12 TFLOPS, roughly 0.2x. The gate is float32, where the SPEC names no dtype for
  Metal; the half ratio is printed for information only.
- Flash attention runs at about 0.25x of MPS sdpa in f32 and 0.05-0.1x in f16. It stages K/V as float through
  threadgroup memory every block and reloads the Q fragments each block. Not tuned; no gate.
- The per-row fallback attention kernel (head dims > ~128 in f32) is slow. It accumulates O in float32 registers 64
  value columns at a time and reruns the key loop (recomputing the scores) per chunk, so Dv > 64 costs ceil(Dv/64)
  passes over K.
- The direct entry points have no autograd.

## Interface issues (for the integrator)
1. `tests/test_opts.py` (P2-OPT) and the matmul cases of `bench/bench_opt.py` assume `a @ b` on METAL is a composed kernel.
   Now that METAL offers custom kernels, they need `FZ_CUSTOM=0`. For example, add
   `monkeypatch.setenv("FZ_CUSTOM", "0")` in their fixture and `os.environ["FZ_CUSTOM"] = "0"` in bench_opt.
2. `schedule._mm_spec` / `_match_attention` (P2-CUSTOM) only match float32 and no mask. Two changes would route
   half/bf16 matmul and causal sdpa to the native kernels with no other change on the Metal side:
   - Match `SUM(MUL(CAST f32 (x16), CAST f32 (y16)))` with x16/y16 as the inputs.
   - Match the `tril.where(qk, -inf)` mask as a `causal` flag, using kernel name `fz_attention_causal` with the same
     params. `METALDevice.run_custom_call` already dispatches on the buffer dtypes and that name.
3. `schedule.run_custom` launches compiled custom kernels with `global_size=(1,)`, which is single threaded. METAL
   bypasses it by overriding `METALDevice.run`. A grid hook on `CustomKernel` (e.g. `launch(params) -> (gs, ls)`) would
   let `run_custom` serve GPU devices too.
4. `CustomKernel.src_per_device` is one source per device, but attention needs threadgroup-memory sizes per head dim.
   METAL stores the generator callable, which `run_custom` would misread as an NPY callable; it never sees it, because
   METAL's own `run` handles custom items.
