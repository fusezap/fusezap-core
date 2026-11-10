# WP6 nn / GPT — status: done

Gate: `FZ_TEST_TIERS=interp uv run pytest tests/test_nn.py tests/test_gpt_parity.py tests/test_skeleton.py` — 37 passed.

## Built
- `fusezap/nn/__init__.py`: Linear, Embedding (one-hot matmul), LayerNorm, RMSNorm, Dropout, get_state_dict (dotted torch-style names),
  get_parameters (requires_grad tensors, deduped), load_state_dict (strict; accepts Tensor or ndarray).
- `fusezap/optim.py`: SGD (momentum, weight_decay, nesterov), AdamW; both match torch (tests). step() assigns then realizes params and state.
- `examples/gpt.py`: char-level GPT with tiny built-in text; `python examples/gpt.py` trains it.
- `tests/test_gpt_parity.py`: L=2,d=64,h=4,T=32, 20 AdamW steps, loss within 1e-4 of torch (gelu is the tanh form), parametrized over `device`.

## Known gaps
- Only the interp tier was run (NPY/CPU/METAL need the scheduler/codegen). AdamW's per-step bias-correction constants are baked into the
  graph as consts, so compiled tiers will recompile each step; switch to scalar tensors if that matters.
- Linear init uses U(-1/sqrt(in), 1/sqrt(in)) (same bound as torch), not bit-identical RNG.

## Review fixes (round 2)
- AdamW: per-parameter step counts (only incremented when grad present).
- RMSNorm: upcasts fp16/bf16 before squaring, returns input dtype.
- get_state_dict: tied tensors keep every name (get_parameters dedups); cycle-safe walk (path-based object stack).
- nn.scaled_dot_product_attention added.
- Optimizer._realize: one realize() call for params + state (SGD momentum buffers no longer realized separately).
- Regression tests added in tests/test_nn.py. Gate: 43 passed.
