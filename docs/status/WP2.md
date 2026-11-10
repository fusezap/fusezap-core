# WP2 Frontend — status: done

Gate: `FZ_TEST_TIERS=interp uv run pytest tests/test_ops.py tests/test_skeleton.py` — passed (`1423 passed, 48 skipped`;
the skips are test_reduce's own "axis out of range" skips).
Own tests: `FZ_TEST_TIERS=interp uv run pytest tests/test_tensor_lazy.py tests/test_tensor_frontend.py tests/test_tensor_review2.py` — 49 passed.
(NPY/CPU/METAL tiers of the own tests fail with NotImplementedError from `create_schedule`: WP3 has not landed yet.)

## Review round 2 fixes (regressions in tests/test_tensor_review2.py)
- A list/ndarray bool index raises NotImplementedError (it was cast to 0/1 int positions). Non-integer index arrays raise IndexError.
- Signed-int `min`/`argmin` use `~x = -1 - x` (order-reversing, no overflow at the dtype min) instead of `-x`.
- `pow` (general path) is an `autograd.Pow` Function with torch's backward rules: grad_b = 0 where e == 0, else e*b**(e-1);
  grad_e = 0 where b == 0 and e >= 0, else r*ln(b). No NaN at base 0.
- `exp` and `log` upcast f16/bf16 to f32 (the MUL by log2(e)/ln2 was rounded to half before EXP2/after LOG2).
- `clip`/`clamp` with python-scalar bounds is where-based: x on a bound gets zero gradient (torch), NaN passes through.
  Tensor bounds still use maximum/minimum (tie splits).
- `cumsum` accumulates bool/int8/uint8/int16 in default_int, like `sum()` (torch gives int64; we give int32).
- `mean`/`var`/`std` (so layernorm) divide by a runtime count when the reduced size is symbolic.


## Built
- `fusezap/lazy.py`: `LazyBuffer` is a base (op/srcs/arg/realized) or a view (`base`, `st`). A view forwards op/srcs/arg/realized
  to its base. A movement op that yields the base's identity view returns the base itself. CONST is a `()`-shaped base,
  reshaped and expanded (the arg is normalized to the dtype's python type). COPY takes a host ndarray (already in host dtype,
  bf16 = uint16 bits) or `src=(lb,)`. `from_buffer` wraps a realized Buffer (op EMPTY). `e()` asserts exact shapes, devices,
  dtypes (CMPLT/CMPEQ give bool, WHERE needs a bool condition, IDIV/MOD are ints only, float-only unary math, BITCAST keeps the
  itemsize). `r()` keeps the reduced axes as size 1 and has `arg` = sorted axes. `contiguous()` is a no-op only on a realized
  base. `copy_to_device` of an unrealized const builds a const on the target.
- `fusezap/interpret.py`: numpy evaluator. It toposorts the unrealized bases iteratively (deep graphs are fine), realizes every
  non-CONST base into a `Buffer` on its device, and never materializes CONSTs. Views are evaluated through
  `ShapeTracker.expr_node` on an arange, substituted with numpy (cached per ShapeTracker). Invalid (masked) lanes read 0.
  C semantics: IDIV/MOD truncate, MAX is `a > b ? a : b`, and float->int casts truncate. f16/bf16 math is done in f32 and
  rounded once per op. bf16 lives in buffers as uint16 bits (RNE, NaN kept). `realize(lbs, var_vals)` accepts values for
  symbolic Vars.
- `fusezap/autograd.py`: `Function` / `apply` with `needs_input_grad` and `requires_grad` inference (True / None / False),
  and `differentiable=False` for Less/Eq/IDiv/Mod. It has every required Function plus Exp2, Log2, IDiv and Mod. On a tie,
  binary Max splits the gradient in half (like torch.maximum). MaxReduce spreads the gradient evenly over the maxima.
- `fusezap/tensor.py`: the full stub API.
  - Numpy broadcasting. Weak python scalars: a float scalar on an int/bool tensor gives default_float, and an int scalar on a
    bool tensor gives default_int. Promotion follows `promote`.
  - getitem supports int, slice (any step, negative too), None, Ellipsis, and one int Tensor index in torch placement,
    including negative indices. It is built from shrink/flip/pad/reshape plus a one-hot where+sum gather.
  - gather, cat, stack, split, chunk, repeat.
  - matmul works by broadcast-mul plus sum, including 1-D operands and batch broadcasting.
  - f16/bf16 reductions, matmul, softmax, log_softmax, logsumexp, var, layernorm, tanh, sigmoid, gelu and silu accumulate in
    f32 and round once.
  - Stable softmax and logsumexp (logsumexp guards an inf max).
  - cumsum is a masked expand+sum. pow has integer fast paths and correct signs for a negative base.
  - cross_entropy (ignore_index, label_smoothing, reductions), sdpa (causal, bool or float mask), triu/tril, one_hot, dropout.
  - Philox RNG on the host (`manual_seed`). `arange` is a host COPY (exact `start + i*step`).
  - `backward()` walks the graph iteratively, accumulates gradients of intermediates locally, and writes `.grad` only on
    leaves with `requires_grad=True`. `no_grad` / `Tensor.inference()` / the `Tensor.no_grad` flag are supported.
  - Both `axis=` and `dim=` keywords are accepted.
  - `assign` replaces `lazydata` after a cast and shape/device checks.

## Known gaps
- `assign` rebinds `lazydata`; it does not write into the old Buffer. Views taken of the old tensor keep the old value.
  True in-place stores need a scheduler-level store, which jit (Phase 2) may want.
- Only one Tensor index per `__getitem__`. Boolean-mask indexing is not supported.
- Symbolic shapes work in lazy and interp (`realize(lbs, var_vals)`), but `Tensor` has no var binding API yet, and
  `Tensor.numpy()` needs concrete shapes.
- `arange`, `rand*`, `eye` (via arange) and tensor-index one-hots use host COPYs of index data. This is correct, but these
  ops are not generated on device.
- cumsum costs O(n^2) work (masked expand+sum). Tensor-index gathers cost O(n·|idx|).
- Signed-int `min` relies on two's-complement wrap of NEG at the dtype min (numpy wraps; C codegen should too).
- `x <= y` / `x >= y` are `lt | eq`, so they are NaN-correct. `sign(nan)` is 0, where torch gives nan.
- The interpreter runs every op on the host. On the METAL device, buffers are copied in and out through the allocator.
