# bench

`bench_ops.py` times fusezap against torch CPU and torch MPS and prints a table (ms per call, plus fusezap GFLOP/s and GB/s).

    uv run python bench/bench_ops.py            # full: elementwise, reduce, softmax, matmul 256..4096
    uv run python bench/bench_ops.py --quick    # smaller sweep
    uv run python bench/bench_ops.py --only matmul --device CPU

Timing: 3 warmup calls, then the mean of `--iters` (default 10) calls, each realized, with the fusezap device (`Device[dev].synchronize()`) and torch MPS synchronized after warmup and before the end time is read.
The Phase 2 target is matmul >= 60% of torch MPS at 4096 (SPEC §11.1); compare the `fz` and `torch mps` columns on the 4096 row.
Until WP2 lands, fusezap cells print `ERR` and the torch columns still work.
