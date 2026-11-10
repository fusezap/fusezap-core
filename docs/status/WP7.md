# WP7 status: done

Gate: `uv run pytest tests/test_ops.py --collect-only -q && uv run python -c "import ast,sys; ast.parse(open('bench/bench_ops.py').read())"` passes.

Built:
- tests/test_ops.py: conformance suite for every public Tensor method vs torch (forward, plus backward where differentiable), parametrized over the `device` fixture; broadcasting, float32/float16/int32/bool, 0-d / size-1 / zero-size, non-contiguous (permute, strided slice). Public API only.
- tests/conftest.py: extended (marker + `rng` fixture); existing fixtures unchanged.
- bench/bench_ops.py, bench/README.md.

Known gaps: tests cannot pass until WP2 lands (expected); only collection verified. Some assertions (e.g. `.T` on ndim>2, bitcast, cumsum of int, int matmul) encode assumptions where SPEC is silent.

## Review fixes (round 2)
- test_reduce reference: `_ref_reduce` uses amax/amin for tuple/None axes (regression: test_reduce_reference_builds).
- test_matmul_noncontig second check made shape-valid (regression: test_matmul_noncontig_shapes).
- test_reshape_view torch reference uses `reshape(tuple(new))` (regression: test_reshape_ref_scalar_shape).
- bench time_fz(fn, dev, ...) now synchronizes the fusezap device after warmup and before the end time (tests/test_bench_ops.py).
- bfloat16 coverage added (cast/roundtrip, binary, unary, reduce, matmul, softmax); conftest gains bf16_to_f32/f32_to_bf16 and assert_close decodes uint16 bf16 bits when dtype="bfloat16".
