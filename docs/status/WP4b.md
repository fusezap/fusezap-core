# WP4b — Lowering (AST -> UOps) + compiled run path

**Status:** done

**Gate:** `FZ_TEST_TIERS=CPU uv run pytest tests/test_ops.py tests/test_codegen.py tests/test_skeleton.py` -> `1441 passed, 48 skipped in 328.36s`
(the skips are the op suite's own tier/dtype skips). I also ran `FZ_TEST_TIERS=METAL uv run pytest tests/test_ops.py` and it passed.

## Built
- `fusezap/codegen/lower.py`
  - `lower(ast, renderer, opts=None) -> ProgramSpec`. `Opts` is accepted and ignored, as the MVP allows.
    - F is the kernel (reduce input) shape and R is the STORE shape. Output axes (`R[i] != 1`) are grouped onto at most 3 `SPECIAL` dims. `gidx0` is the innermost group. Outer groups are merged first, and a group above `renderer.global_max[axis]` is merged inward. SPECIALs are emitted outermost first, so CPU loops nest outer to inner.
    - A renderer with `has_local=True` (Metal) splits `gidx0` into `gidx0*L + lidx0`, where L is the largest divisor of the size that is <= 256. `global_size` then counts threadgroups and `local_size=(L,1,1)`, so no bounds guard is needed.
    - Each reduce axis (`R[i]==1!=F[i]`) gets its own nested `RANGE`. A `DEFIFZ_ACC` is declared before the loops and updated by `PHI` inside them.
    - SUM and MAX of float16/bfloat16 accumulate in float32 and cast back, like the interpreter. MAX inits to -inf, the int min, or False. Float MAX propagates NaN (`WHERE(x==x, MAX(x,acc), x)`), like `np.max`/`torch.amax`.
    - Index math: every axis is a symbolic `Var`. LOAD/CONST/STORE index and validity come only from `ShapeTracker.expr_idxs`, and the resulting Node trees (Mul/Div/Mod/Sum/Lt/And) are lowered to int32 ALU UOps. These are CSE'd per loop scope; each RANGE opens a scope, because C locals do not outlive their block.
    - Masked LOADs become `LOAD(buf, idx, valid, 0)`. Masked CONSTs become `WHERE(valid, c, 0)`. A symbolic CONST value is lowered from its Node and CAST. Epilogue leaves are read with the reduced axes pinned to 0.
    - Symbolic dims become `DEFIFZ_VAR`s, sorted by name. `ProgramSpec.vars` gives that order, and `global_size` may hold Nodes.
    - Value `IDIV`/`MOD` with a non-constant divisor are guarded: a zero divisor gives 0 (interpreter semantics) instead of trapping/UB.
    - Kernel names are `E_<dims>` or `r_<out dims>_<reduce dims>`.
  - `run_compiled(device, si)`
    - It caches `(ProgramSpec, Program)` by `(ast, type(renderer), device.name)`. The device name is included because programs are device objects.
    - It compiles with `device.compiler.compile_cached` and builds `device.runtime(name, lib)`.
    - It launches with the buffers' handles, the `global_size` evaluated with `si.var_vals`, the var values in `spec.vars` order, and `wait=FZ_DEBUG>=1`, so the measured time is reported.
    - An empty output (a 0 global dim) is not launched.
    - `FZ_DEBUG>=4` prints the source, and `>=5` also prints the UOps.
- Review fixes:
  - The `run_compiled` cache key now uses a canonical AST (`_ast_key`): ConstBuffer values are keyed by `repr`, so `0.0`/`-0.0` no longer share a program, and NaN consts hit the cache instead of adding one entry per call.
  - CAST-of-CONST folding to int32 now follows the interpreter: int sources wrap to int32 range, and float sources fold through `np.trunc(...).astype(int32)`, so out-of-range values and NaN/inf match it.
  - `store()` lowers the index before it emits the `IF`, so a CSE'd index ALU is never declared inside the guarded block. Multi-root ASTs with masked STOREs that share a view now compile.
- `tests/test_codegen.py` (18 tests, including one regression test per review fix):
  - UOp structure: SPECIAL order and sizes, rank-5 collapse to 3 dims, reduce loop layout, guarded masked load, the GPU local split and global_max merge, symbolic DEFIFZ_VAR.
  - Full CPU pipeline against torch: matmul/softmax/epilogue, padded/flipped/strided views with reduces, multi-axis reduces, NaN max, int div/mod by zero, f16 sum accumulating in f32.
  - Runtime: a symbolic size with one compile for several values, FZ_DEBUG=4 source printing, cache reuse.

## Known gaps / notes
- No optimizations: no upcast/unroll/group-reduce, and the prologue is not hoisted out of the reduce loop. Loop-invariant loads are recomputed inside the loop.
- Index math is int32. Buffers above 2^31 elements are not supported.
- `INT_MIN // -1` and runtime (non-constant) float->int casts out of range are C UB (not guarded). Constant folds match the interpreter.
- Most of the CPU gate time is `ctypes.CDLL` (about 0.3 s per new kernel). `CPUProgram` (WP4a) writes every library to a fresh temp file, and macOS scans each new dylib. A content-addressed stable path in the cache dir would make the first load of a cached kernel cheap.
