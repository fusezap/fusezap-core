# P2-JIT Kernel replay: done

Gate: `uv run pytest tests/test_jit.py && uv run python bench/bench_jit.py`: **passed**
(`39 passed`; the bench output on METAL is below; review-fix rerun: 16.4x on METAL). I also ran it on a clean `origin/main` worktree with only the P2-JIT
files added. That run passed on NPY and CPU (`20 passed`, CPU bench 64.5x); METAL was not available in that fresh venv.

```
device METAL, GPT L=2 d=64 h=4, 32 decode steps, symbolic pos
  without jit:   10.182 ms/step total,    9.973 ms/step overhead
  with jit:       0.828 ms/step total,    0.558 ms/step overhead
  overhead reduction: 17.9x  (target >= 5x); outputs match: yes
device CPU (--device CPU):  overhead 9.36 ms -> 0.105 ms per step (89x)
```

## Built
- `fusezap/jit.py`
  - `jit(fn)` returns a `Jit`.
    - Call 1 runs `fn` eagerly.
    - Call 2 runs it again and records every ScheduleItem that executes.
    - Calls 3 and later replay the recorded items with the new input buffers and var values. They do no graph
      building, scheduling or lowering. `Jit.reset()` drops the capture.
  - Inputs:
    - Tensor args, positional or keyword and also nested in list/tuple/dict, are the inputs. They are realized (as
      contiguous bases) before the call.
    - Symbolic values come from `bind(var, val)` args (`fn` receives the Var), from `var_vals={var: val}`, or both.
    - Replay checks input shapes, dtypes, devices and the set of vars against the capture.
  - Buffers:
    - Buffers written by recorded items are "owned".
      - A jit output gets a fresh buffer on every call, so earlier outputs stay valid and KV caches can ping-pong.
        A symbolic output shape is allocated at this call's size and returned with a concrete shape.
      - An intermediate is one persistent buffer, sized at the size expression's `vmax`.
    - All other buffers, such as weights and host staging buffers, are constants.
    - A recorded item that has no var values, reads only constants and writes only persistent intermediates runs once
      and is then skipped. In the GPT decode step these are the arange host copies.
  - Programs: compiled kernels launch the cached `Program` directly, with the bufs' handles, the evaluated symbolic
    global size and the var values. Copies replay as a host round trip. Kernels with no cached program, and other kinds
    (e.g. `custom`), fall back to `Device.run(ScheduleItem(...))` with the substituted buffers.
  - Copies replay as a host round trip of the current logical size (the symbolic size at this call's var values), so a
    `.to(dev)` of a symbolically sized tensor works even though persistent intermediates are allocated at `vmax`. The
    contiguous source the scheduler builds for a COPY of a view gets its symbolic size from the COPY's destination.
  - Nesting: a jitted fn called while another jit is capturing runs its `fn` inline, so the outer capture records its
    kernels. Vars bound inside it that the outer jit does not receive are baked at their captured value.
  - A capture in which a symbolic buffer is empty (create_schedule skips its kernels) is dropped; the next call
    captures again.
  - `bound(var_vals)` is a context manager that lets eager code realize graphs with symbolic shapes. It is the non-jit
    baseline, and the jit uses it internally.
- `bench/bench_jit.py`
  - `decode_step(m, tok, caches, pos)` is a KV-cache decode step for `examples/gpt.py`'s GPT:
    - The cache is fixed at `Tmax` and updated with `WHERE(arange == pos)`.
    - Attention reads `cache[:, :, :pos+1]`, with `pos` a `Var`.
    - The positional embedding row is a symbolic shrink.
  - The bench times the step with and without jit and asserts that the outputs are equal (f32 tolerance) and that
    overhead drops by at least 5x.
  - Overhead = host time of the step call minus the kernel time inside it. CPU kernels run synchronously and are
    subtracted; METAL only enqueues, so nothing is subtracted.
- `tests/test_jit.py` (NPY, CPU, METAL)
  - Elementwise and matmul replay vs torch. `fn` runs exactly twice, and replay calls `create_schedule` zero times.
  - Fresh and nested outputs, view and kwarg inputs, symbolic input lengths and symbolic output shapes.
  - `bound`, mismatch errors, and state threaded through args.
  - The fallback path when no program is cached.
  - GPT decode: jit replay == eager symbolic step == full forward on the prefix.
  - Regressions (review fixes): symbolic cross-device copies (view source, persistent intermediate), nested jit
    (incl. pass-through and baked vars), capture at a var value that makes a buffer empty.

## Known gaps / caveats
- Same contract as tinygrad's TinyJit:
  - Tensors reached through closures are constants. Rebinding a weight after capture is not seen by the replay.
  - Python side effects of `fn`, and `.numpy()`/`.item()` values read inside it, are baked into the capture.
- `bound`: the lazy graph refers to the Var, not to a value. Realize what a step built before leaving its `bound`
  block, or a later realize uses the later value.
- Per-kernel launch cost on METAL (one command buffer per kernel, ~17 us each) now dominates the replayed step.
  Batching launches into one command buffer belongs to the Metal device.
- With `FZ_INTERP=1`, `bound` with var values is unsupported (asserts). The interpreter takes no var_vals.
- Not yet exported as `fz.jit`; see the interface issues below.

## Interface issues (for the integrator)
1. Export `jit` (and `bind`, `bound`) from `fusezap/__init__.py`: `from fusezap.jit import jit, bind, bound`.
2. `fusezap.realize.realize(lbs)` cannot pass `var_vals` to `create_schedule`, so symbolic shapes cannot be realized
   through the Tensor API. `fusezap/jit.py` works around this by swapping `fusezap.realize.realize` inside `bound(...)`.
   Proposed: `realize(lbs, var_vals=None)` plus a capture hook (e.g. a module-level list that `run_schedule` appends
   to). This is the same hook the jit needs to record.
3. There is no public way to get the compiled program for an AST. The jit reads `fusezap.codegen.lower._cache`,
   matching on the first three key fields (ast key, renderer type, device name). P2-OPT added `bw` and `hand` to that
   key. Proposed: `get_program(device, ast) -> (ProgramSpec, Program)` in lower.py, which `run_compiled` would also use.
4. `Tensor.numpy()` asserts a concrete shape. A tensor with a symbolic shape cannot be read even inside `bound`.
