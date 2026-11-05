# WP4a — C-style renderer + CPU device

**Status:** done

**Gate:** `uv run pytest tests/test_cstyle_cpu.py tests/test_skeleton.py` → `50 passed`

## Built
- `fusezap/codegen/cstyle.py`
  - `Renderer` base (`device`, `has_local`, `global_max`, `supports_dtype`, `render`).
  - `CStyleRenderer`: a single driver `render(name, uops)` that walks the linear UOp list and renders every UOp in the
    `uops.py` spec. Each value gets an SSA local (`alu3`, `val0`, `acc1`, `ridx0`, `cast2`); CONSTs are inlined.
    Overridable table: `type_map`, `kernel_prefix`, `buffer_prefix`, `buffer_suffix`, `arg_int_prefix`, `code_for_op`
    (`lambda *srcs, dtype`), `render_special`, `render_kernel(name, body_lines, bufs, vars)`, `render_cast`,
    `render_bitcast`, `render_const`. Extra optional hooks: `buf_type_map`, `prelude`, `render_alu`, `render_load`,
    `render_store`, `var_name`, `render_size`, `float_literal`, plus `self._cur` (the UOp being rendered).
    With `has_local=True` (GPU) a SPECIAL renders as `int gidx0 = <render_special(u)>;`. With `has_local=False` it opens a
    `for` loop that closes at the end of the kernel.
  - `CRenderer`: `void name(float* restrict data0, ..., const int n)`. Math is `exp2f/log2f/sinf/sqrtf`, and RECIP is `1.0f/x`.
    float16 is `_Float16`. bfloat16 is a `float` in registers and `unsigned short` in memory: loads/stores convert, and every
    bf16 ALU result and cast is rounded to bf16 (round-to-nearest-even, NaN kept), matching torch's per-op rounding.
    Bitcasts use `__builtin_bit_cast`.
- `fusezap/device/cpu.py`: `CPUAllocator` (an LRUAllocator over ctypes byte arrays), `ClangCompiler`
  (`clang -O2 -shared -fPIC -ffp-contract=off -march=native`, source on stdin, error includes the source, `FZ_CC` overrides
  the compiler), `CPUProgram` (bytes → temp .so → `ctypes.CDLL`; ignores global/local size; `wait=True` returns seconds),
  `CPUDevice`.
- `tests/test_cstyle_cpu.py` (hand-built UOp programs compared with torch/numpy): elementwise add/mul/where, 2-D SPECIAL,
  SUM/MAX reductions (DEFIFZ_ACC/RANGE/PHI), a guarded LOAD that would otherwise read out of bounds, IF/ENDIF, unary math,
  inf/nan constants, casts (f→i truncation, i→f, →bool, bool→f, f→f16, bitcast), int32 IDIV/MOD with negative operands,
  f16 storage, bf16 storage plus an f32 reduction over bf16, f32→bf16 rounding, a symbolic DEFIFZ_VAR as a loop bound and
  a value (one compile run with three sizes), compile cache, a compile error, LRU reuse, and a GPU-style subclass of the
  renderer surface.

## Review fixes
- NEG renders as `(-(x))`, so NEG of a negative inlined literal no longer becomes C pre-decrement `--c`.
- `_bf16_round` maps finite values above the f32 range to ±inf instead of crashing in `struct.pack`. bf16 CONST/ACC 1e39 now
  renders `INFINITY`, which matches torch.
- `fz_f2bf` keeps the sign and payload bits of a NaN. It sets the quiet bit only when truncating would turn the NaN into
  inf. bf16 LOAD->STORE copies and BITCASTs in either direction now keep the exact bit patterns (for example 0x7f81 and 0xff81).
- Regression tests: `test_neg_of_negative_const`, `test_bf16_const_above_f32_range`, `test_bf16_nan_bits_preserved`.

## Known gaps / notes
- `CPUDevice.run` uses `lower.run_compiled` (WP4b), which is not tested here.
- `MAX` follows the spec's `a > b ? a : b`, so `MAX(nan, x)` returns `x` (torch.maximum propagates NaN).
- A symbolic SPECIAL size renders as the DEFIFZ_VAR's name when the size is that Var object. Otherwise it uses
  `Node.render(RenderOps)` from WP1, which is still a stub (untested).
- The UOp enum names in `uops.py` are `DEFIFZ_GLOBAL/DEFIFZ_VAR/DEFIFZ_ACC` (they look like a rename accident of DEFINE_*).
  This code uses them as they are.
- Single-threaded only. The disk cache key is the source text alone; the compiler flags are not part of it.
