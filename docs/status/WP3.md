# WP3 Scheduler — status: done

Gate: `FZ_TEST_TIERS=NPY uv run pytest tests/test_ops.py tests/test_schedule.py tests/test_skeleton.py` — passed
(`1453 passed, 48 skipped`; the skips are test_reduce's own "axis out of range" skips).
Also on NPY: `tests/test_tensor_lazy.py tests/test_tensor_frontend.py tests/test_tensor_review2.py` all pass. An MLP
forward+backward (6 layers, gelu/layernorm/softmax/log, 45 kernels) gives bit-identical loss and grads on NPY and interp.

## Built
- `fusezap/schedule.py` `create_schedule(outs, var_vals=None)`:
  - Iterative toposort of the unrealized bases. Planning picks the bases that become kernel outputs:
    - Forced: outputs, CONTIGUOUS, COPY, the inputs of a COPY, and EMPTY (an EMPTY gets a Buffer only, no item).
    - Reverse pass (consumers first) tracks, for each base, the kernel root it fuses into, whether a reduce sits above
      it, and whether the path to the root has a non-trivial view. A reduce is materialized if it is reached from more
      than one kernel, has a reduce above it, is reached through anything other than a contiguous reshape, or is a
      second reduce fusing into the same root.
    - A second pass prefers the §6.2 rule "consumed by more than one kernel **and** contains a reduce". It materializes
      the epilogue base (e.g. `e = x.sum(1)+1` used by two kernels) instead of the reduce, so the epilogue is computed
      once.
    - Pad safety: a fused elementwise tree read through a masked view (pad) is kept fused only if it maps all-zero
      inputs to 0. That is checked by evaluating the tree on zeros with the interpreter's numerics. `exp(x).pad` is
      materialized, `(x*2).pad` fuses.
  - Kernel AST per §6.3: one STORE root, `bufs[0]` is the output, and inputs follow in order of first use (one buffer
    loaded through several views shares one idx). Leaf sts compose the view chain with `compose(inner, outer)`. The
    view chain is the views concatenated, then simplified. F = the reduce input shape; every LOAD/CONST has shape F.
    Epilogue leaves are the leaf view, then reshape to R, then expand to F, so they have stride 0 on the reduced axes.
    The STORE st is contiguous over R. The AST is a DAG: nodes are deduped by (base, st), and the one REDUCE node is
    shared.
  - COPY items have `kind="copy"`, `ast=None` and `bufs=(dst, src)`. A COPY of a non-contiguous view first gets a
    synthesized CONTIGUOUS kernel. A host COPY's src is a staging `Buffer("NPY")` that wraps the host array.
  - Zero-size outputs get no item (the Buffer allocates on first use). Output Buffers are created unallocated and
    assigned to `.realized` once the whole schedule is built. `ScheduleItem.var_vals` holds the Vars used by the AST.
  - `FZ_DEBUG>=2` prints one line per item (kind, output, inputs, ops). `>=3` also prints `LazyOp.pretty`.
- `fusezap/realize.py` `run_schedule`: allocates all bufs, runs copy items itself (a host round trip through the two
  allocators), and dispatches kernels to `Device[bufs[0].device].run(si)`. With `FZ_DEBUG>=1` it prints one line per
  item with its time: the device's returned time, else wall time after `synchronize()`.
- `fusezap/device/npy.py`: `NPYAllocator` (an LRUAllocator whose handles are flat `np.uint8` arrays) and `NPYDevice.run`.
  It evaluates the AST with numpy. Leaf indices come from `ShapeTracker.expr_idxs` with one Var per axis, substituted
  with broadcast aranges and cached per (st, grid). Masked lanes read 0, for LOAD and CONST alike. Prologue nodes run
  over F, and the reduce plus epilogue run over R, with the epilogue leaves' reduced axes pinned to 0. Elementwise and
  reduce numerics reuse `fusezap.interpret`, so NPY == interp isolates scheduler and AST bugs. The STORE goes through
  the STORE st's index and valid mask.
- `tests/test_schedule.py` (28 tests):
  - Kernel counts: an elementwise chain, a chain through views, prologue+reduce+epilogue, a keepdim reshape epilogue,
    a reduce through an expand, softmax = 3, a multi-consumer reduce (2 outputs = 3 kernels, 1 kernel using it twice
    = 1), the epilogue materialized for two consumer kernels, one reduce per kernel (siblings, nested, max+sum),
    CONTIGUOUS, pad safety (including a padded CONST).
  - AST conventions (full-shape leaves, reduced STORE st, epilogue stride 0, outputs first, deferred allocation), COPY
    items, topological order, symbolic shapes (`Var` shrink, two values), FZ_DEBUG output.
  - 11 NPY-vs-interp bit-equality cases (chain, reduce, softmax, matmul, layernorm, views/pad/flip, cat+cumsum,
    min/argmax/mean/std, int idiv/mod, half, where).

## Known gaps
- `create_schedule` has an extra optional `var_vals` kwarg. There is no Tensor-level Var binding yet, so `realize()`
  passes none, and symbolic buffer sizes then assert.
- Pure elementwise bases with several consumer kernels are recomputed in each kernel (per the MVP rules). Within one
  kernel, nodes are deduped by (base, st).
- Copies go through a host round trip in `run_schedule` (fine for NPY/CPU; Metal could do better later).
- `create_schedule` raises the recursion limit to 20000, because `LazyOp.walk` and AST building recurse.

## Review fix 1 (offset-0 leading shrink)
- A leading slice of a reduce (`x.sum(1)[:2]`, `x.max(0)[:3]`, `x.sum(1,keepdim=True)[:2]+1`) has a contiguous st, so
  it was treated as a trivial reshape and fused. The STORE then covered all of R but the output was smaller (IndexError
  on NPY, OOB write on CPU/METAL). The reverse pass now counts a view as trivial only if it is contiguous **and**
  `lb.size == b.size`, as §6.2 requires.
- The same root cause hit plain elementwise fusion (`x.exp()[:2]*1`): `compose()`'s contiguous-outer shortcut reshaped
  inner to the smaller outer shape. That shortcut now requires equal sizes. Otherwise it uses the general view chain,
  where the outer indexes a prefix of S.
- Regression tests: `test_reduce_leading_shrink`, `test_elementwise_leading_shrink`. A random NPY-vs-interp fuzz
  (sum/max/shrink/pad/exp chains, 1500 seeds) gives 0 failures.
