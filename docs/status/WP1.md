# WP1 Shapes — status: done

Gate: `uv run pytest tests/test_symbolic.py tests/test_view.py tests/test_skeleton.py` — passed (59 tests).
Also stress-run offline with 5000 random movement sequences, 2000 symbolic sequences and 5000 random expressions: all pass.

## Built
- `fusezap/shape/symbolic.py`: `Var`, `NumNode`, `MulNode`, `DivNode`, `ModNode`, `LtNode`, `SumNode`, `AndNode`.
  Constant results fold to plain `int`. Simplifier: constant folding, like-term combining (`a*c1 + a*c2`),
  `(x*c)//c`, factoring multiples out of `//` and `%`, gcd reduction, `%` dropped by bounds, `(x//m)*m + x%m -> x`,
  normalized `<` (`x*c < b -> x < ceil(b/c)`). `//`/`%` are floor semantics; for int divisors the numerator is
  normalized to `vmin >= 0`, so C truncating `/`,`%` render identically (property-tested via a C-semantics render).
  `substitute` works with numpy arrays as values; `subs`/`sym_subs` for partial substitution; `smax`, `smin`, `ands`.
- `fusezap/shape/view.py`: `View.create` canonicalization (size-1 strides 0, full masks -> None, empty masks ->
  canonical all-`(0,0)`, single-element mask dims folded into the offset), all six movement ops with masks,
  mask-aware `View.reshape` (groups dims by product, merges strides, splits flat mask ranges into boxes),
  `ShapeTracker` with view merging (structural + symbolic affine composition), `expr_idxs`, `expr_node`,
  `real_size`, `unit_stride_axes`, `vars`, `subs`.
- Tests: `tests/test_symbolic.py`, `tests/test_view.py` (hypothesis vs numpy indexing / `as_strided` on an
  arange buffer, including pad/shrink/flip/expand and symbolic substitution).

## Review fixes
- Symbolic dims that may be 0 (vmin 0, e.g. a KV-cache length or `n-1` from a shrink) no longer crash
  `expr_idxs`/`expr_node`. `floordiv`/`mod` accept a Node divisor with vmin 0 (bounds computed as if it were >= 1,
  `substitute` divides by 1 in place of 0); an int 0 divisor still asserts. `_unflatten` divides by
  `max(s, 1)` (`s + (s<1)` for symbolic s, also int 0 dims), so rendered code never divides by 0 at runtime.
  Regression tests: three repros from the review plus a hypothesis fuzz over `Var('p', 0, 5)` that checks every
  p in 0..5 (including 0) against numpy, and `test_divisor_may_be_zero` in test_symbolic.py.

## Known gaps
- Reshapes that split/merge a symbolic dim with other dims only merge when the view is contiguous or the
  symbolic dim maps 1:1; otherwise a second view is pushed (correct, just not merged). View merging via
  symbolic composition is skipped when any view is symbolic.
- `AndNode` renders as `&&`; there is no AND in BinaryOps, so lowering must emit it as MUL of 0/1 (or nested WHERE).
- Index divisors for dims that may be 0 render as `(p+(p<1))`; fine, but slightly heavier code.
- Symbolic (Node) divisors are not normalized; their numerators are assumed nonnegative (true for index math).
