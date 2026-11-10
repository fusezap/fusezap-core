# P2-INT: Run A Phase 2 integration

**Status:** done

**Gate:** `uv run pytest -n 6` (all tiers: interp, NPY, CPU, METAL) plus `bench/bench_jit.py`, `bench/bench_metal.py`,
`FZ_DEVICE=METAL bench/bench_opt.py`.
**Result:** passed. `5597 passed, 144 skipped in 729.37s`. All three benches exit 0.

## Starting point
The first full run after the four Phase 2 packages merged: `8 failed, 5587 passed, 144 skipped`.

## Changes
- `fusezap/__init__.py`: exports `Var`, `Jit`, `bind`, `bound` and `fz.jit`. A plain `from fusezap.jit import jit` would
  shadow the submodule on the package, so `import fusezap.jit as J` returned the function (this broke
  `test_jit.py::test_replay_without_cached_programs` on every tier). Instead the `fusezap.jit` module gets a callable
  ModuleType subclass: `@fz.jit` decorates (returns a `Jit`) and `fusezap.jit` stays the module.
- `tests/test_opts.py`: the `kernel()` helper schedules with `FZ_CUSTOM=0`. P2-MTL made METAL match `a @ b` to a custom
  item, so 4 METAL tests in test_opts asserted on a custom item instead of the composed kernel they optimize.
- `bench/bench_opt.py`: sets `FZ_CUSTOM=0` for the same reason (it benchmarks generated kernels, not the library GEMM).
- `tests/test_custom.py::test_device_without_custom_kernels`: skips devices whose `custom_kernels` is non-empty
  (METAL now defines it as a property, which is not in the instance `__dict__`).
- `tests/test_api.py` (new): top-level exports and `@fz.jit` with a bound symbolic var.

## Known gaps
Open performance gaps: Metal f16/bf16 matmul and attention vs MPS; single-scalar sum bandwidth.
