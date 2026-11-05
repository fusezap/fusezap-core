# P2-CUSTOM: custom kernels (SPEC §11.1)

**Status:** done

**Gate:** `uv run pytest tests/test_custom.py tests/test_ops.py -q`
**Result:** passed. `5013 passed, 144 skipped` after the review fixes (pyproject `addopts=-q` plus the gate's `-q` hides the summary line; counted from the progress output, rc=0) (all tiers: NPY, CPU, METAL and interp). The skips are test_ops' own
"axis out of range" skips.

Also run:
- The full `tests/test_ops.py`, `tests/test_nn.py` and `tests/test_gpt_parity.py` on CPU with the C reference kernels
  installed (`FZ_CUSTOM_CPU=1 PYTHONPATH=tests pytest -p test_custom`). No failures.
- `test_schedule`, `test_integration`, `test_tensor_*`, `test_nn` and `test_gpt_parity` on NPY, CPU and interp (NPY uses
  the custom kernels by default). No failures.

## Built
- `fusezap/schedule.py`
  - `CustomKernel(name, src_per_device, signature, nbufs, dtypes)`. `src_per_device` maps a device base name to C-like
    source (compiled devices) or to a python callable `fn(arrays, params)` (NPY).
  - `CustomCall(kernel, params)` is the payload of a `ScheduleItem(kind="custom")`: `ast=(CustomCall,)`,
    `bufs=(out, *inputs)`, `var_vals={}`. It lives in `ast` because ScheduleItem has no field for it. That also keeps
    the payload when `fusezap/jit.py` rebuilds items for replay.
  - Matching happens in `_Graph.srcs`. A matched reduce's srcs become the kernel's inputs, so the toposort and planner see
    `out <- inputs` and nothing in between. The planner forces the custom output and every non-leaf input to be
    materialized. Consumers (epilogues) read the custom output as an ordinary buffer.
  - `"matmul"`: `SUM(MUL(x, y))` over one axis. Each non-reduced axis falls into a batch, M or N group, and each group
    must collapse to one stride per operand and output. The operands must be float32 single unmasked int views with no
    CONST operand. M*N must be > 1, so a plain fused dot such as `(x*x).sum(1)` stays composed. Params are
    `(Bt, M, N, K, strides..., offsets)`.
  - `"attention"`: the exact graph that `Tensor.softmax` / `scaled_dot_product_attention` build, i.e.
    `softmax(q @ k^T [* c], -1) @ v` with no mask or dropout. It becomes one kernel with inputs (q, k, v). The scale and
    log2(e) are passed as f32 bit patterns in int params.
  - Reference kernels `MATMUL` and `ATTENTION` (`REFERENCE_KERNELS`):
    - **CPU:** plain C. It repeats the composed op order: SUM accumulates k = 0..K-1 from 0.0f, MAX keeps NaN, and it
      uses `exp2f` and `1.0f/x`. The CPU output is **bit-identical** to the composed kernels. The matmul uses k-outer
      and n-inner loops with an accumulator row.
    - **NPY:** numpy, with the same numerics as the composed NPY path (product, then np.sum over K).
  - `run_custom(device, si)` executes an item: NPY calls the callable, and a compiled device does
    `compiler.compile_cached` + `runtime`.
  - `install_custom_kernels(device)` offers the kernels on a device instance and routes `kind="custom"` to `run_custom`.
  - `FZ_CUSTOM=0` disables matching. A device without `custom_kernels`, or without source for itself, falls back to
    composed ops.
- `fusezap/device/npy.py`: `NPYDevice.custom_kernels` (the reference kernels, on by default), and `run` dispatches
  `kind="custom"`.
- `tests/test_custom.py`:
  - Custom vs composed (`FZ_CUSTOM=1` vs `0`) and vs torch on NPY and CPU (bit-equal on CPU). It covers 13 matmul forms:
    batched, broadcast, transposed, strided, vec/mat, K=1, chains with epilogues, 65x130x33. It also covers attention
    for 5 shapes plus the explicit softmax form.
  - Fallbacks: dot-sum, padded operand, int, half, const operand, causal, mask, softmax on another axis, shared scores.
  - Backward through the custom kernels vs torch, the schedule structure, `FZ_CUSTOM=0`, and a device without kernels.
  - A subprocess run of the op suite's matmul/linear/sdpa/nn tests on CPU with the kernels installed, through this
    file's pytest plugin hook (`FZ_CUSTOM_CPU=1 PYTHONPATH=tests pytest -p test_custom ...`).

## Review fixes
- `_match_attention`: a CONST operand counts as the softmax scale (or the log2(e) factor) only when its view has no mask.
  Before this, a padded const such as `full(...,0.5).pad(...)` was matched as a uniform scale even though it reads 0
  outside its mask. Regression test: `test_attention_masked_const_scale_falls_back`.
- `run_custom`: compiled programs are now cached on the device instance (`device._fz_progs`) and keyed on
  `(kernel name, source)`. The old key was a module-level `(id(device), name)`, which ran a stale binary after a kernel
  with the same name was replaced and could also hit a reused `id`. Regression test: `test_replaced_kernel_recompiles`.

## Interface issues (for the integrator)
1. `fusezap/ir.py` has no `CustomKernel` and `ScheduleItem` has no field for one. The payload rides in `ast`
   (`(CustomCall,)`). Suggestion: add `ScheduleItem.custom: CustomCall | None`, or bless `ast=(CustomCall,)` in SPEC §6.3.
2. `CompiledDevice.run` -> `run_compiled` asserts `kind == "kernel"`, and `CPUDevice` has no `custom_kernels`. So the C
   reference kernel is not used on CPU by default; tests turn it on with `install_custom_kernels`. To enable it: in
   `CompiledDevice.run`, `if si.kind == "custom": from fusezap.schedule import run_custom; return run_custom(self, si)`,
   and in `CPUDevice`, `custom_kernels = fusezap.schedule.REFERENCE_KERNELS`.
3. `fusezap/jit.py` (in progress elsewhere) replays non-compiled items through `dev.run`, which works for custom items
   because the payload is in `ast`. Precompiling custom items for replay is not done.

## Known gaps
- Only float32. Half/bf16 matmuls match only after their f32 cast prologue is materialized. Int matmuls stay composed.
- A matmul whose M (or N) axes are split by an axis of the other group (e.g. `(2,1,5,4) @ (3,4,6)`) is not one GEMM and
  stays composed. Symbolic shapes stay composed.
- Attention with a mask, causal attention or dropout falls back to two custom matmuls plus composed softmax kernels.
  The attention kernel needs O(S) scratch per row (malloc).
- No Metal source yet (the simdgroup matmul and flash attention are a separate package). The reference C kernels are
  single-threaded like the rest of the CPU backend. Performance target "custom == composed" means numerically equal,
  which holds bit-for-bit on CPU.
