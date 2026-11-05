# FuseZap — Architecture & Interface Spec (v0.2, inference-first v1)

FuseZap is a Techyodl product. The Python package is `fusezap` (`import fusezap as fz`); env vars use the `FZ_` prefix.

Status: **approved direction; Phase 1 contracts are frozen in code.** This document plus the interface
stubs under `fusezap/` are the contract that builders implement against. If code and this doc disagree,
fix one of them in the same change — never silently diverge.

## 0. Product

The shared thread is *being fast on hardware other than NVIDIA*.

| Layer | What it is | License |
|---|---|---|
| `fusezap` engine | compiler + runtime, backends for CPU, Metal, CUDA, AMD (HIP) and AWS Inferentia2/Trainium1 (NEURON) | Apache-2.0 |
| `fusezap.serve` | OpenAI-compatible inference server: paged KV cache, continuous batching, quantization | Apache-2.0 |
| Backend kit | template backend, conformance suite, "port your chip" guide | Apache-2.0 |

**v1 definition of done** (all must hold; measured by `bench/` scripts and committed reports):
1. The Phase 1 gate (§10) passes on CPU and METAL.
2. The op conformance suite passes on CPU, METAL, CUDA (H100) and HIP (MI300X).
3. `fusezap serve <model>` serves Qwen2.5-0.5B/1.5B/7B-Instruct (Apache-2.0, ungated). Mistral-7B-v0.3 is an
   optional second architecture check (it may need an HF login). Greedy output is token-identical to HF `transformers` for 64 tokens in f32;
   in bf16, the top-1 token matches at ≥ 99% of positions. The bf16 gate is measured on **Qwen2.5-7B-Instruct**
   (`tests/test_serve_cuda.py::test_7b_bf16_top1_vs_hf_spec`): HF f32 greedily continues 8 chat prompts by 128 tokens and
   FuseZap bf16 is teacher-forced on the same 1024 positions. At 0.5B bf16 itself falls short (HF's own bf16 scores
   0.979 against HF f32), so the 0.5B bf16 test is a regression check (within 2 points of HF bf16), not this gate.
4. Throughput (7B, bf16, 512 in / 128 out tokens) is at least the **gate**: ≥ 50% of vLLM on the same GPU
   at batch 1 and 32. **Stretch:** ≥ 80%. On MI300X it is compared with vLLM-ROCm.
5. int8 and int4 weight-only quantization, plus FP8 on H100/MI300X, all with documented accuracy deltas.
6. `pip install fusezap` works from a built wheel. Docs, benchmark report and vendor one-pager are in `docs/`.

Honest positioning: the cost-per-token advantage comes mainly from running well on cheaper hardware
(AMD and others), not from beating vLLM on H100. The gate is set so that v1 is credible, not dominant.

## 1. Goals and non-goals

**Phase 1 goal:** a small, readable, compile-first deep learning engine that trains a small GPT
end to end on Apple Silicon (CPU and Metal), with numerics matching PyTorch. Later phases (§12)
build on it without changing the layering.

Design principles (the lessons from PyTorch / JAX we are acting on):

| # | Principle | Concretely |
|---|---|---|
| P1 | **Small primitive core** | ~28 primitive ops (§3). Everything else (matmul, softmax, conv, attention) is composed in the frontend. Backends implement only primitives. |
| P2 | **Lazy + compiled by default, eager feel** | Tensor ops build a lazy graph; realization schedules fused kernels. No separate "compile mode", so no graph breaks. |
| P3 | **Views are free** | Movement ops (reshape/permute/expand/pad/shrink/flip) never copy; they compose `ShapeTracker`s and are folded into kernel index math. |
| P4 | **Symbolic shapes from day one** | A dim may be an `int` or a symbolic `Var` with bounds. Kernels take symbolic dims as runtime args, so one compile serves every value. |
| P5 | **Inspectable everything** | `FZ_DEBUG` levels print the schedule, kernel AST, and generated source. Generated kernels are human-readable. |
| P6 | **Backend-neutral IR** | One kernel AST (§6) and one loop IR (§7) for all devices; a backend = renderer + compiler + runtime + allocator. |
| P7 | **Sharding in the type, later in the runtime** | `LazyBuffer` carries an optional `Sharding`; it propagates but only single-device execution exists in the MVP. |

**Non-goals for Phase 1:** CUDA/HIP backends, auto-tuning, FP8, library kernels and serving. These
arrive in Phases 2–5 (§12).
**Non-goals for v1:** multi-node anything, large-scale training features (FSDP, fault tolerance), MoE
models (stretch only, if contingency remains), Windows, and a PyTorch import layer.

## 2. Layering

```
 user code
    │
 fusezap.Tensor  (fusezap/tensor.py)            frontend API + autograd (fusezap/autograd.py)
    │  builds
 LazyBuffer graph (fusezap/lazy.py)        primitive ops on lazy buffers, views via ShapeTracker
    │  realize()  (fusezap/realize.py)
    ├── FZ_INTERP=1 → fusezap/interpret.py  reference: evaluate the graph op by op with numpy
    └── default     → fusezap/schedule.py   split the graph into kernels → list[ScheduleItem]
                         │
                     Device.run(si)    (fusezap/device/)
                         ├── NPY     executes the kernel AST directly with numpy (reference)
                         ├── CPU     lower → C source → clang → ctypes
                         └── METAL   lower → MSL source → runtime compile → Metal
                 lower: fusezap/codegen/lower.py (AST → UOps), renderers: fusezap/codegen/{cstyle,metal}.py
```

Three tiers of execution give every layer an oracle: **interp** (graph-level, trusted
reference) ≡ **NPY** (tests the scheduler and AST) ≡ **CPU/METAL** (tests codegen). Every
op test runs on all available tiers and compares against PyTorch.

## 3. Primitive ops (`fusezap/ops.py`)

| Class | Ops | Notes |
|---|---|---|
| `UnaryOps` | `NEG, EXP2, LOG2, SIN, SQRT, RECIP, CAST, BITCAST` | `CAST`/`BITCAST` take `arg=DType`. `exp(x)` is `EXP2(x*log2(e))` in the frontend. |
| `BinaryOps` | `ADD, MUL, MAX, CMPLT, CMPEQ, IDIV, MOD` | Comparisons return `bool`. `IDIV`/`MOD` use C semantics (truncate toward zero), ints only. `MAX(a,b) = a > b ? a : b`. |
| `TernaryOps` | `WHERE` | `WHERE(cond: bool, a, b)`. |
| `ReduceOps` | `SUM, MAX` | `arg=axes: tuple[int,...]`; reduced axes **stay as size 1** (keepdim). The frontend squeezes. |
| `MovementOps` | `RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP` | View-only. `PAD` fills with 0; padding with another value is `WHERE` in the frontend. |
| `LoadOps` | `CONST, EMPTY, COPY, CONTIGUOUS` | `CONST` is a scalar broadcast through a view. `EMPTY` is an unfilled allocation. `COPY` moves between devices or from host. `CONTIGUOUS` forces materialization. |

Total: 28. **Adding a primitive op requires a spec change.** Prefer composing in the frontend.

Elementwise ops (Unary/Binary/Ternary) require all sources to have the same shape and
dtype (except `WHERE`'s condition and the comparisons' bool output). The frontend handles
broadcasting and dtype promotion, and the lazy layer asserts them.

## 4. DTypes (`fusezap/dtype.py`)

`bool, int8, uint8, int16, int32, int64, float16, bfloat16, float32`.

- Defaults: `dtypes.default_float = float32`, `dtypes.default_int = int32` (Metal lacks
  fast int64; the frontend uses int32 for indices).
- Promotion (`fusezap.dtype.promote`): `bool < uint8 < int8 < int16 < int32 < int64 < float16 < bfloat16 < float32`,
  except that `float16` with `bfloat16` gives `float32`, and int with float gives the float type. Python
  scalars adopt the tensor's dtype ("weak" scalars, like JAX).
- `bfloat16` on the host is stored as `uint16` bits, and `Tensor.numpy()` of a bf16 tensor returns that
  `uint16` bit array (decode with `bits.astype(uint32) << 16` viewed as f32). The CPU backend computes bf16 by upcasting
  to f32 in the generated C. Metal uses native `bfloat`.
- Accumulation: `sum`/`cumsum` of `bool/int8/uint8/int16` accumulate in `default_int` (int32, not torch's int64).
- Integer arithmetic wraps (two's complement) on every tier; the CPU backend compiles with `-fwrapv`. The frontend
  relies on this (e.g. signed `min` is `~max(~x)`). Out-of-range float->int casts are implementation defined at
  runtime (constant folds follow numpy).
- Reserved, not implemented: `float8_e4m3`, `float8_e5m2`.

## 5. Shapes and views (`fusezap/shape/`)

### 5.1 Symbolic (`fusezap/shape/symbolic.py`)
- `sint = int | Node`. `Var(name, min, max)` is a `Node`. Nodes support `+ - * // %` with ints and
  other nodes, comparisons `<` (returning a Node), and `.vmin/.vmax` bounds.
- Simplification must at least: fold constants, combine `a*c1 + a*c2`, simplify `(x*c)//c`,
  and drop `% n` when `0 <= x < n` by bounds.
- `x // d` and `x % d` with a Node divisor `d` may have `d.vmin == 0` (a dim that may be empty). Bounds are computed
  as if `d >= 1`; index math divides by `d + (d<1)`, so a `LtNode` can appear inside a divisor and code that walks
  trees must not assume `d.vmin > 0`. An int divisor must still be `> 0`.
- `Node.render(ctx: Renderer-agnostic callback) -> str` renders to C-like source.
  `Node.substitute(dict[Var,int]) -> int`.
- **MVP restriction:** symbolic dims are allowed in shapes and in `RESHAPE`/`EXPAND`/`SHRINK`
  args. `PAD` and `PERMUTE` with symbolic args just need to not crash.

### 5.2 View and ShapeTracker (`fusezap/shape/view.py`)
- `View(shape, strides, offset, mask, contiguous)` is frozen and hashable, and is built with
  `View.create(shape, strides=None, offset=0, mask=None)`. Strides for size-1 dims are canonicalized to 0.
- `ShapeTracker(views: tuple[View, ...])`. Movement methods return a new ShapeTracker and
  merge into the last view when possible. `reshape` pushes a new View only when merging fails.
- `ShapeTracker.expr_idxs(idxs: list[Node]) -> (idx: Node, valid: Node)` gives the linear buffer index
  and validity (from masks/padding) for a logical index. This is the single source of
  truth for index math; codegen calls nothing else.
- `ShapeTracker.expr_node(gidx: Node)` does the same, given one flattened logical index.
- Properties: `shape`, `size` (elements in the logical view), `real_size()` (minimum buffer
  elements backing it), `contiguous`.

## 6. Lazy graph and kernel AST

### 6.1 LazyBuffer (`fusezap/lazy.py`)
- A `LazyBuffer` is either a **base** (`op`, `srcs`, `arg`, and possibly `realized: Buffer`)
  or a **view** of a base (`base`, `st: ShapeTracker`). Movement ops return views and never add nodes.
- Constructors and methods are listed in the stub. Every method asserts its shape and dtype preconditions.
- Hash-consing is not required in the MVP; common-subexpression elimination is a later optimization.

### 6.2 Scheduler (`fusezap/schedule.py`)
`create_schedule(outs: list[LazyBuffer]) -> list[ScheduleItem]`, topologically ordered.

Fusion rules (MVP):
1. Chains of elementwise ops fuse into their consumer.
2. **A kernel holds at most one reduce**, and its elementwise prologue and epilogue fuse into it.
3. A base is materialized (becomes a kernel output) if it is: an output; `CONTIGUOUS`;
   a reduce whose consumer reads it through a non-trivial view (anything besides
   reshapes that keep the reduce contiguous); consumed by more than one kernel **and** contains a
   reduce; or the input of a `COPY`.
4. `CONST` and `EMPTY` never become kernels (`EMPTY` is just an allocation).
5. `COPY` becomes its own ScheduleItem, with `ast = None` and `kind = "copy"`.

### 6.3 Kernel AST (`fusezap/ir.py`)
A kernel is `tuple[LazyOp, ...]` of `STORE` roots (the MVP emits exactly one).
`LazyOp(op, src: tuple[LazyOp, ...], arg)`, where `op` is a `BufferOps`, `UnaryOps`, `BinaryOps`,
`TernaryOps` or `ReduceOps` value.

- `BufferOps.LOAD` takes `arg=MemBuffer(idx, dtype, st)`; `idx` indexes into `ScheduleItem.bufs`.
- `BufferOps.CONST` takes `arg=ConstBuffer(val, dtype, st)`.
- `BufferOps.STORE` takes `arg=MemBuffer(idx, dtype, st)` and `src=(value,)`. Outputs are `bufs[0..n_out)`.
- Shape convention: every LOAD/CONST `st` in a kernel has the **full shape** (the reduce's input
  shape). The STORE `st` has the reduced axes set to 1. All shapes in one kernel have the same rank.

`ScheduleItem(ast, bufs: tuple[Buffer, ...], var_vals: dict[Var, int], kind="kernel"|"copy")`.

## 7. Codegen

### 7.1 Loop IR — UOps (`fusezap/codegen/uops.py`)
`UOp(op: UOps, dtype: DType|None, src: tuple[UOp, ...], arg)`. A kernel is a `list[UOp]` in
execution order (linearized, SSA). The ops and their semantics are defined in the stub
docstrings and are the **only** contract between lowering and renderers.

### 7.2 Lowering (`fusezap/codegen/lower.py`)
`lower(ast, renderer: Renderer) -> ProgramSpec`. MVP strategy:
- Global dimensions are the output (STORE) shape, collapsed where possible and mapped onto
  up to 3 hardware dims (`SPECIAL gidx0..2`). On CPU these become loops.
- Reduce axes become a `RANGE` loop with an accumulator (`DEFIFZ_ACC` + `PHI`) inside each output element.
- Index math comes from `ShapeTracker.expr_idxs`. Masks become guarded loads (`LOAD` with a valid
  operand and an alternative value).
- Optimization hooks (`Opts`: upcast, unroll, local/group-for-reduce) exist in the
  signature, but the MVP may ignore them. As built (Phase 2): group-for-reduce (`group_reduce=32`) is one SIMD group
  per output on METAL (`simd_sum`) and one warp on CUDA (`__shfl_xor_sync`); `lower.GROUP_REDUCE_DEVICES` lists them.

### 7.3 Renderers
- `CStyleRenderer` (`fusezap/codegen/cstyle.py`) is the shared base that turns UOps into C-like source. Subclasses override
  a small table: type names, kernel prefix, buffer qualifiers, `SPECIAL` rendering, and math function names.
- `CRenderer` (CPU) renders `void name(float* restrict data0, ..., int var0)`. `SPECIAL` dims become
  `for` loops. One thread in the MVP.
- `MetalRenderer` (`fusezap/codegen/metal.py`) renders a `kernel void name(device float* data0 [[buffer(0)]], ...,
  constant int& var0 [[buffer(n)]], uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]])`.
  That plain layout holds while buffers + vars <= 31 (Metal slots 0..30). `metal_layout(nbufs, nvars)` picks the
  layout for both renderer and runtime: *packed* (<= 30 buffers; all vars in one `constant int* fz_vars [[buffer(nbufs)]]`)
  or *argbuf* (> 30 buffers; pointers in an argument-buffer struct at `buffer(0)`, vars at `buffer(1)`; needs Metal 3).
- `BITCAST` to `bool` renders as `((x) != 0)` on every C-style backend (a byte other than 0/1 is true).

## 8. Devices (`fusezap/device/`)

```
Device["CPU"] / Device["METAL"] / Device["NPY"]  -> device object (registry, lazily created)
device.allocator : Allocator    alloc(nbytes) / free(h, nbytes) / copyin(h, memoryview) / copyout(memoryview, h)
device.run(si)   : executes one ScheduleItem
CompiledDevice(device) adds: renderer, compiler (src -> bytes, disk-cached), runtime (name, lib) -> Program
Program.__call__(*bufs, global_size, local_size, vals=()) -> float|None   (seconds when wait=True)
```

- `Buffer(device, size, dtype)` is allocated lazily (`ensure_allocated()`), with `copyin(memoryview)`,
  `copyout(memoryview)` and `numpy() -> np.ndarray`.
- The allocator keeps a size-bucketed free-list cache (`LRUAllocator`). Buffers are freed when the Python
  object is collected.
- CPU: `clang -O2 -shared -fPIC -ffp-contract=off -fwrapv -march=native`. No `-ffast-math`, to keep numerics
  matching PyTorch. Loaded via `ctypes.CDLL`. The disk cache lives at `~/.cache/fusezap/`, keyed by
  `Compiler.cache_key()` (compiler path + flags) plus the source.
- METAL: `pyobjc-framework-Metal`. Source is compiled at runtime with `newLibraryWithSource:options:error:`
  (there is no offline `metal` compiler on this machine). Buffers use `MTLResourceStorageModeShared`.
- Device selection: `FZ_DEVICE` env var, default the first available of `METAL`, `CUDA` (§11.2), `CPU`.

## 9. Frontend (`fusezap/tensor.py`, `fusezap/autograd.py`, `fusezap/nn/`, `fusezap/optim.py`)

- The `Tensor` API surface is fixed by the stub in `fusezap/tensor.py`. Signatures and semantics mirror
  PyTorch where names coincide (`sum(axis, keepdim)`, `matmul`, `softmax(axis)`, ...).
  Here `axis` corresponds to PyTorch's `dim`; both keyword names are accepted.
- Autograd: reverse mode over `Function` nodes (`fusezap/autograd.py`). Each differentiable primitive is
  a `Function` with `forward(*lazybuffers)` and `backward(grad_lazybuffer)`. `Tensor.backward()`
  topologically sorts and accumulates `.grad`. The backward pass builds lazy graph too, so the
  forward and backward passes are scheduled and fused together.
- `Tensor.no_grad` is a context manager. `requires_grad=None` means "inferred" (like tinygrad).
- RNG: a Philox stream on the host (numpy) produces a buffer via `COPY`. `fusezap.manual_seed(n)`.
- `fusezap.nn`: `Linear, Embedding, LayerNorm, RMSNorm, Dropout`, `scaled_dot_product_attention`,
  `get_parameters(obj)` (recursive attribute walk, like tinygrad), and
  `state_dict / load_state_dict`. `fusezap.optim`: `SGD`, `AdamW` (PyTorch-equivalent defaults).

## 10. Testing contract (`tests/`)

- `pytest` with fixtures from `tests/conftest.py`: `device` (parametrized over the available tiers:
  `interp`, `NPY`, `CPU`, `METAL`) and the `assert_close(ne_tensor, torch_tensor, dtype)` helper with
  the tolerance table below.
- Every Tensor method has a forward test and, if differentiable, a backward test against PyTorch.
- Movement and symbolic layers have `hypothesis` property tests: random op sequences on `ShapeTracker`
  equal numpy's `as_strided`/indexing.
- Tolerances: f32 `rtol=1e-5, atol=1e-6` (elementwise), `rtol=1e-4, atol=1e-5` (reductions/matmul);
  f16 and bf16 `rtol=1e-2, atol=1e-2`.
- End-to-end: `tests/test_gpt_parity.py` builds GPT (L=2, d=64, h=4, T=32, char vocab) with identical
  weights in both frameworks, runs 20 AdamW steps, and requires every loss to match within `1e-4` (f32, CPU and METAL).

## 11. v1 components beyond Phase 1

Each component plugs into an extension point that already exists, so Phase 1 code is not rewritten.

### 11.1 Performance foundation (Phase 2)
- **Kernel replay (`fusezap/jit.py`, `@fz.jit`):** the first two calls capture the schedule. Later calls replay
  the compiled programs with new buffers and new symbolic values, skipping scheduling and Python
  graph building. This is essential for decode, where per-token Python overhead would dominate.
- **Kernel optimizer:** the `Opts` in `lower.py` gets implemented (upcast, unroll, local/workgroup tiling,
  group-for-reduce), plus `FZ_BEAM=n`, which times candidate Opts on the device and caches the winner per
  (AST, device) in `~/.cache/fusezap/`.
- **Custom kernels:** `ScheduleItem.kind = "custom"` with `CustomKernel(name, src_per_device, signature)`.
  The scheduler pattern-matches matmul and attention subgraphs into it when the device offers one
  (`device.custom_kernels`). Each library kernel must match its composed-op equivalent in the op suite.
- **Metal:** a simdgroup-matrix matmul and a flash-attention kernel. Gate: matmul ≥ 60% of MPS (torch `mps`)
  at 4096².

### 11.2 CUDA backend (Phase 3) — `fusezap/device/cuda.py`, `fusezap/codegen/cuda.py`
- Driver API plus NVRTC through NVIDIA's `cuda-bindings` (`cuda.bindings.driver` / `.nvrtc`; torch's Linux CUDA wheels
  bring it, or `pip install cuda-bindings`); TMA descriptors (`cuTensorMapEncodeTiled`) go through libcuda via
  `ctypes`. No PyTorch or CuPy dependency at runtime. Without cuda-bindings, a driver or a GPU, `Device.available("CUDA")`
  is False. Each `CUDA:<n>` uses its device's primary context, made current whenever another one is.
- `CUDARenderer(CStyleRenderer)`, with custom kernels for tensor-core matmul (mma.sync / WGMMA on
  H100), flash attention (prefill and decode), and FP8 GEMM.
- As built (Run B): `fusezap/device/cuda_kernels/` holds the library kernels. `CUDADevice()` installs them into the
  P2-CUSTOM hook (`custom_kernels`: `matmul` = 3xTF32 for float32, `matmul16` = f16/bf16 inputs read through their f32
  casts by `cuda_kernels._match16`, `attention` = the SIMT float32 kernel); `FZ_CUDA_LIB=0` at construction skips that,
  `FZ_CUSTOM=0` still disables matching, `FZ_MMA16=0` keeps 16-bit inputs on 3xTF32. GEMM dispatch: M ≤ `FZ_GEMM_SMALL_M`
  (default 64, decode) runs, on sm_90 for 16-bit "nt" operands, the swap-AB WGMMA (weights as the 64-row operand, the
  rows as n; `FZ_WG_SWAP=0` turns it off), else mma.sync with a short tile, both split-K to fill every SM; otherwise WGMMA + TMA on sm_90 (`FZ_NO_WGMMA=1`
  turns it off), mma.sync on sm_80+. fp8 e4m3 tensor cores need sm_89+; FP8 activations are uint8 e4m3 codes with one
  f32 scale per row (`fusezap.serve.cuda`), weights one scale per tensor. Serving on CUDA (`fusezap.serve.cuda`) calls the
  GEMMs and attention directly through `jit.run_item`; the lm_head writes float32 logits.
- Gate: the op suite on CUDA; matmul ≥ 60% of cuBLAS (torch) at 4096² in bf16. Tests: `FZ_TEST_TIERS=CUDA` adds the CUDA
  tier (opt-in, `tests/conftest.py`).

### 11.3 Inference server (Phase 4) — `fusezap/serve/`
| Module | Responsibility |
|---|---|
| `models/llama.py` | Llama-architecture decoder (RMSNorm, RoPE, GQA, SwiGLU) parametrized by HF `config.json`; covers Qwen2.5, Mistral and Llama |
| `weights.py` | safetensors loading (mmap), dtype conversion, sharded checkpoints, HF Hub download |
| `kv_cache.py` | paged KV cache: fixed-size blocks, block tables, symbolic sequence lengths |
| `scheduler.py` | continuous batching: admits, preempts and recomputes requests; prefill/decode mix under a token budget |
| `sampling.py` | greedy, temperature, top-k, top-p, repetition penalty, seeded |
| `quant.py` | int8 and int4 weight-only (group size 128, AWQ/GPTQ checkpoint import), FP8 (e4m3) with per-tensor scales |
| `api.py` | OpenAI-compatible `/v1/chat/completions`, `/v1/completions` and `/v1/models`, SSE streaming, `/health`, Prometheus `/metrics`, via `starlette` + `uvicorn` |
| `metrics.py` | `/metrics` in the Prometheus text format with vLLM's names (`vllm:num_requests_running` / `_waiting`, `vllm:kv_cache_usage_perc`, `vllm:num_preemptions_total`, `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`, each labelled `model_name`; `process_resident_memory_bytes`), updated per engine step and on every abort |
| `cli.py` | `fusezap serve <hf-model-id> --device --dtype --quant --port` (console script `fusezap`, also `python -m fusezap`) |
| `engine.py` | `LLMEngine`: jitted prefill/decode forwards over the paged cache, driven by the scheduler (used by `api.py`) |
| `tokenizer.py`, `models/__init__.py` | HF `tokenizers` wrapper with HF-exact chat templates (jinja2, part of the `serve` extra); `load_model(id_or_path)` |
| `tools.py` | `tools` / `tool_choice` (none, auto, required, named): Hermes and Mistral formats rendered and parsed into `tool_calls` |
| `constrain.py` | grammar-constrained decoding (llguidance, optional): `response_format` `json_schema` / `json_object`, and named / required tool calls whose tools set `"strict": true` (envelope + arguments matching the parameter schema); `call_constraint(CallSpec \| tools.CallConstraint, tok, prefix_ids)` builds a tool call's grammar from a format-independent description (prefix, args_key, close, call_open, separator, schemas, strict, parallel), so a tool-call format needs no grammar code |

Constrained decoding. `pip install 'fusezap[serve]'` includes llguidance (MIT, no Python dependencies, ~3 MB
abi3 wheels for macOS arm64 / x86_64 and Linux x86_64 / aarch64); on a server without it (`pip install llguidance` adds
it), `response_format` json_object / json_schema and strict tool calls get HTTP 400 naming the package, and nothing
else changes. xgrammar (vLLM's default) was not used because its wheel requires torch, transformers and triton. The API
builds a `constrain.Constraint` (plain data: the grammar, the forced-prefix token ids, the tokenizer.json path) into
`SamplingParams.constraint`; the engine gives the request a matcher (`Sequence.cons`) and at every token of that request
reads its logits row alone to the host (CUDA: one vocab-wide copy; Metal: the shared buffer), sets the tokens the
grammar forbids to -inf, samples or argmaxes as usual (greedy and sampled alike), advances the matcher, and finishes the
request with "stop" once the grammar is complete. JSON is compact (", " and ": ", no free whitespace); a schema
keyword the compiler does not implement (e.g. uniqueItems) is a 400, not ignored. Unconstrained rows in the same steps
keep their device argmax, and their tokens are unchanged. Lookahead and the prefill pipeline, which feed or read device
argmaxes, pause while a constrained request runs. With tensor parallelism only rank 0 samples, so the mask is applied
there and the workers' forwards are unchanged (they get plain token ids). Host cost per constrained token at Qwen's
152k vocabulary: ~0.1 ms (mask ~0.02 ms median, applying it ~0.06 ms), plus the one-row read. tool_choice auto is
never constrained. `response_format` is accepted on `/v1/completions` too.

FP8, Mac half (Run B, as built): weight-only e4m3 with one scale per tensor. Codes are stored as `uint8` e4m3 bits and
decoded in the GEMV with integer ops plus an int16->f16 bitcast; activations stay f16/f32. The native
`float8_e4m3`/`float8_e5m2` dtypes (§4) with a scale operand on `CAST`, and the FP8 GEMM with fp8 activations, are
still reserved for the H100 half (P3-GEMM); once they exist the dequantize becomes `q.bitcast(float8_e4m3).cast(cdt) * s`.

Out-of-tree kernels (serve's paged attention, KV writes, row gathers) use the public `fusezap.jit` hooks:
`jit.run_item(si)` runs a ScheduleItem now and records it into an active capture; a `kind="custom"` item whose `ast[0]`
has `launch(dev, bufs, var_vals)` is launched through that method at capture and replay. `jit.current_var_vals()`
returns the bound symbolic values.

Tests: model-free serve tests and the Qwen2.5-0.5B tests (skipped when the model is not cached) run in the default
suite; tests marked `@pytest.mark.slow` (the CLI end-to-end server processes) are skipped unless `FZ_SLOW=1` or `-m slow`.

### 11.4 AMD backend + backend kit (Phase 5)
- A HIP device and renderer (planned): HIP runtime plus hiprtc via `ctypes`, with MFMA matmul and
  attention custom kernels for MI300X (gfx942). Gates: the op suite, serving correctness, and ≥ 50% of vLLM-ROCm.
- A template backend and a porting guide (planned): a documented minimal backend (allocator, compiler, runtime,
  renderer table), and a vendor-facing conformance suite with a pass/fail report generator.

### 11.5 Release (Phase 6)
Wheel build, documentation site sources, a benchmark report with reproducible numbers, README, and a vendor
one-pager.

### 11.6 Still reserved (not in v1)
Multi-device and sharding (`LazyBuffer.sharding` + `ScheduleItem.kind="collective"`), MoE, and the
PyTorch import layer.
Tensor parallelism for serving does not use either yet: it runs one process per device (fusezap/serve/dist.py),
each with an ordinary single-device graph over its weight shard, and the all-reduce is a kind="custom" item
(`dist.AllReduceCall`, launched through `fz.jit.run_item`), so jit replay and CUDA-graph capture need no change.

## 12. Phase 1 work packages (parallel builders)

Later phases follow the same ownership rules.

| WP | Owns (write access) | Depends on (interface only) | Done when |
|---|---|---|---|
| **WP1 Shapes** | `fusezap/shape/*`, `tests/test_symbolic.py`, `tests/test_view.py` | — | the stub API is implemented; hypothesis tests pass vs numpy `as_strided` |
| **WP2 Frontend** | `fusezap/tensor.py`, `fusezap/autograd.py`, `fusezap/lazy.py`, `fusezap/interpret.py`, `tests/test_tensor_*.py` | WP1 | the full Tensor API runs on `FZ_INTERP=1`, with fwd and bwd matching torch |
| **WP3 Scheduler** | `fusezap/schedule.py`, `run_schedule` in `fusezap/realize.py`, `fusezap/device/npy.py`, `tests/test_schedule.py` | WP1, WP2's lazy API | the op suite passes on NPY; kernel counts match the fusion rules in tests |
| **WP4 CPU codegen** | `fusezap/codegen/lower.py`, `fusezap/codegen/cstyle.py`, `fusezap/device/cpu.py`, `tests/test_codegen.py` | WP1, WP3's AST | the op suite passes on CPU |
| **WP5 Metal** | `fusezap/codegen/metal.py`, `fusezap/device/metal.py`, `tests/test_metal.py` | the UOp spec, CStyleRenderer | hand-built UOp programs run; the op suite passes on METAL after integration |
| **WP6 nn / GPT** | `fusezap/nn/*`, `fusezap/optim.py`, `examples/gpt.py`, `tests/test_nn.py`, `tests/test_gpt_parity.py` | the Tensor API | GPT parity passes on interp, then on all tiers |
| **WP7 Harness** | `tests/conftest.py`, `tests/test_ops.py`, `bench/*` | the Tensor API | the op conformance suite covers every Tensor method; `bench/bench_ops.py` compares against torch CPU/MPS |

Rules for builders: write only inside your own files; if a shared interface (`fusezap/ops.py`, `fusezap/dtype.py`,
`fusezap/helpers.py`, `fusezap/ir.py`, `fusezap/codegen/uops.py`, `fusezap/device/__init__.py`, the stubs' signatures, or this spec) is wrong, report it instead of editing it. The
integrator owns the shared files.
