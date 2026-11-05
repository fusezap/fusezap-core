# A-INT: Run A Phase 1 integration (round 1)

**Status:** done

**Gate:** `uv run pytest` (all tiers: interp, NPY, CPU, METAL; includes tests/test_gpt_parity.py on every tier)
**Result:** passed. `5342 passed, 144 skipped in 488.39s` (serial). Also passes under `FZ_NOCACHE=1 -n 8`.

## Starting point
All eight packages (WP1-WP7, WP4a/4b) were merged on main. The full suite already passed before any
integration change (5325 passed, 144 skipped; the skips are intentional "axis out of range" parametrizations in
test_ops.py). GPT parity passes on interp, NPY, CPU and METAL (20 AdamW steps, |loss diff| < 1e-4).

## Seams fixed
- **Disk compile cache ignored compiler flags** (`fusezap/device/__init__.py`, `fusezap/device/cpu.py`): the cache was
  keyed only by source, so changing flags or `FZ_CC` reused stale `.so` files. Added `Compiler.cache_key()`
  (default `""`); `ClangCompiler.cache_key()` returns compiler path + flags.
- **Signed int overflow UB on CPU** (WP2 interface issue): WP2's signed `min/argmin` and `-x` rely on two's-complement
  wrap. CPU now compiles with `-fwrapv`. Metal already wraps.
- **CPU bitcast to bool** (WP5 interface issue): `__builtin_bit_cast(_Bool, byte)` gives an invalid bool for bytes
  other than 0/1. `CRenderer.render_bitcast` now emits `((x) != 0)`, the same as Metal.

## Interface issues resolved in docs/SPEC.md
- §4: bf16 `Tensor.numpy()` returns uint16 bits (confirmed as the contract; WP7 tests are right). Small-int
  `sum/cumsum` accumulate in `default_int`. Integer wrap semantics; runtime out-of-range float->int is
  implementation defined.
- §5.1: Node divisor may have `vmin == 0`; index math uses `d + (d<1)`; LtNode may appear in a divisor.
- §7.3: Metal plain/packed/argbuf layouts via `metal_layout()`; `BITCAST` to bool normalizes.
- §8: `-fwrapv` and the cache key.

## Tests
`tests/test_cstyle_cpu.py::test_compile_cached_and_timing` now clears `FZ_NOCACHE`, because it asserts a cache hit (it failed whenever the suite ran with FZ_NOCACHE=1).

`tests/test_integration.py` (all tiers): a symbolic dim with vmin 0 (KV-cache style shrink, permute, reshape, reduce)
for p in {0,1,5,8}, int32 wrap at the limits (+1, neg, min, argmin), bitcast byte to bool, bf16 numpy bits, and the
CPU cache key. The bool test fails on CPU without the renderer fix.

## Known gaps
- `fz.jit` (SPEC §9, `fusezap/jit.py`) does not exist yet; nothing in Phase 1 needs it, so no placeholder was exported.
- Runtime float->int casts of out-of-range values remain platform-defined in generated C/Metal.
- The full suite takes about 8.5 minutes serially; `-n 8` (pytest-xdist) is available.
