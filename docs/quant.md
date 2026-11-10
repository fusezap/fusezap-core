# Weight-only quantization (P4-QUANT)

`fusezap/serve/quant.py` quantizes the Linear weights of a `fusezap.serve` Llama-architecture model in place:

```python
from fusezap.dtype import dtypes
from fusezap.serve.models import load_model
from fusezap.serve.quant import quantize_model, weight_bytes
m = load_model("Qwen/Qwen2.5-0.5B-Instruct", dtype=dtypes.float16)
quantize_model(m, "int8")      # "int8" | "int4" (asymmetric, g128) | "int4-sym" (g128) | "fp8" (e4m3, per-tensor)
m.prefill([ids]); m.decode([[tok]])

from fusezap.serve.quant import import_checkpoint
m = import_checkpoint("path/or/hf-id-of-a-4bit-GPTQ-or-AWQ-checkpoint")   # -> Llama with QuantLinear layers
```

## Formats

| mode | codes | scales / zeros | bits per weight (Qwen2.5-0.5B linears) |
|---|---|---|---|
| `int8` | int8, symmetric, round to nearest, [-127, 127] | one scale per output channel (max abs / 127) | 8.0 + 16 / in |
| `int4` (= `int4-asym`) | 4-bit codes in [0, 15], two per byte | scale and zero point per group of 128 inputs: s = (max - min) / 15, z = round(-min / s) | 4.25 |
| `int4-sym` | 4-bit codes, w = (q - 8) s | scale per group of 128: s = max abs / 7.5 (GPTQ's symmetric grid) | 4.125 |
| `fp8` (= `fp8-e4m3`) | float8_e4m3fn bits in uint8 (sign, 4-bit exponent with bias 7, 3-bit mantissa, max 448, no inf), round to nearest even, saturating | one scale per tensor: s = max abs / 448 | 8.0 |

- Scales and zeros are stored in the model's compute dtype (f16 for an f16 model, f32 otherwise).
- fp8 "per tensor" means per stored tensor: the fused q/k/v and gate/up projections each get one scale. The host
  encoder is bit-identical to `torch.float8_e4m3fn` (tested on 50k random values and every rounding midpoint).
- int4 packing: within each group of 128 input columns, byte j holds column j in the low nibble and column j + 64 in
  the high nibble. With the interleaved layout (even/odd columns), the fused GEMV ran 2-3x slower.
- Inputs that are not a multiple of 128 are zero-padded to whole groups. The pad columns are cut off after dequantization.
- The q/k/v, o, gate/up and down projections of every layer are quantized. The lm_head stays in the model dtype
  unless `head=True`, and so do the norms, biases and the host embedding table (it is tied to the lm_head in Qwen2.5-0.5B).

## FP8 on the device

fusezap has no float8 dtype yet: SPEC §4 reserves `float8_e4m3`/`float8_e5m2`, and CAST has no scale operand.
So the codes live in a uint8 buffer and the dequantize is a few integer ops plus a bitcast. The 7 magnitude bits,
shifted left by 7, form an f16 whose value is exactly e4m3 / 256. This holds for normals and subnormals alike, because
both formats put the exponent field right above the mantissa. So w = sign * bitcast_f16((q & 0x7F) << 7) * (256 s),
and the device stores 256 s in the compute dtype. Like int8, this fuses into the decode GEMV. It is weight-only FP8:
activations stay f16/f32. The FP8 GEMM with fp8 activations (H100) is P3-GEMM's and needs the real dtypes.

## AWQ / GPTQ checkpoint import

`import_checkpoint(path, fmt=None, dtype=float16, device=None, **llama_kw)` reads a safetensors checkpoint whose
`config.json` has a `quantization_config` (`quant_method` "gptq" or "awq", 4 bits). It converts every
`<proj>.qweight/qzeros/scales[/g_idx]` into the int4 format above, fuses q/k/v and gate/up, and builds the Llama with
`QuantLinear` layers. Full-precision tensors (embeddings, norms, biases, an unquantized lm_head) load as usual.

| format | qweight | qzeros | w |
|---|---|---|---|
| GPTQ (AutoGPTQ) | int32 (in/8, out), packed along `in` | int32 (G, out/8), packed along `out` | (q - (qzeros + 1)) s (v1). With `checkpoint_format: gptq_v2`, (q - qzeros) s |
| AWQ (AutoAWQ GEMM) | int32 (in, out/8), packed along `out` in the order 0,2,4,6,1,3,5,7 | same packing | (q - qzeros) s |

- A GPTQ checkpoint whose zero points are all 8 (`sym: true`) becomes `int4-sym`. Everything else becomes `int4`
  with zero points.
- `group_size: -1` (one group per row) is supported.
- GPTQ act-order (`desc_act: true`, a scattered `g_idx`) keeps `g_idx` on the device. Column i then uses group
  `g_idx[i]`. The scale and zero lookup is a one-hot gather over the G groups on every call, so this is slower
  than the contiguous layout. A trivial `g_idx` (i // group) is dropped.
- Not supported (they raise): bits other than 4 (for example GPTQ-Int8), AWQ `gemv`/other versions, AWQ without
  zero points, and fused projections where only some parts are quantized.
- Tests: `tests/test_quant.py` packs random codes with reference packers written from AutoGPTQ's `pack` and
  AutoAWQ's `from_linear`, then checks the import against those libraries' dequantization formulas (bit-exact).
  End to end, it writes 2-layer GPTQ (sym), AWQ (asym) and GPTQ act-order checkpoints of Qwen2.5-0.5B in safetensors.
  The GPTQ and AWQ imports give the same codes, scales and prefill + jitted decode logits as `quantize_model` on the
  same weights. The act-order import matches a model that holds the reference-dequantized weights.
- No real GPTQ/AWQ checkpoint is cached on this machine, and the task rules forbid downloading other models, so no
  calibrated (AWQ/GPTQ-quantized) perplexity is measured yet. Run `bench/bench_ppl.py` on such a checkpoint when one
  is available.

## How it runs

`QuantLinear` stands in for the weight Tensor. `Tensor.linear` only calls `weight.ndim` and `weight.transpose(-1, -2)`,
so `llama.py` is unchanged, and the transpose returns the lazy dequantize expression.

- **Decode (dequantize in the kernel):** decode compiles its matmuls as composed kernels (FZ_CUSTOM=0). The scheduler
  fuses the dequantize expression into the GEMV's reduce. The kernel loads the int8 or packed uint8 codes and the
  scales, dequantizes in registers and accumulates in f32. The full-precision weight never exists in memory.
  `tests/test_quant.py::test_dequant_fused_into_gemv` checks this: one kernel, with no float buffer of weight size.
- **Prefill (dequantize per call):** where the Metal library GEMM matches (FZ_CUSTOM=1, f32 operands after the
  matmul's casts), the scheduler first materializes the dequantized weight into a scratch buffer, then runs the GEMM.
- Isolated GEMV GPU time (BEAM 4, M5 Max, f16 activations, best of 30):

  | shape (out x in) | f16 | int8 | int4 |
  |---|---|---|---|
  | 9728 x 896 (gate/up) | 26.0 us | 20.1 us | 10.2 us |
  | 896 x 4864 (down) | 13.5 us | 12.0 us | 7.8 us |

- Whole decode step, Qwen2.5-0.5B, f16 activations, batch 1, BEAM 4, position ~130. The GPU time is summed from the
  per-kernel GPU timestamps of a replayed step. The lm_head is not quantized and takes about 0.6 ms of every step.

  | weights | GPU ms / step (best run) | wall decode tok/s (best of 3 runs) |
  |---|---|---|
  | f16 | 4.8 | 60 |
  | int8 | 3.5 | 105 |
  | int4 | 3.7 | 89 |

  The GPU was shared with a busy Chrome GPU process during these runs. The same configuration varied by 3-10x between
  runs (one int4 run summed to 40 ms of GPU time, an f16 run to 13 ms), so treat these figures as best-case.
  Batch-1 decode is launch-bound anyway: about 266 kernels per step at roughly 32 us of host time each (see
  docs/status/P4-MODEL.md). So the end-to-end rate tracks weight bytes much less than the kernel times do.
- A later wall-clock run with fp8 (BEAM 4, 64 decode tokens after a 128-token prompt, best of 3) gave f16 91,
  int8 97, fp8 94 and int4 99 tok/s. The fp8 decode (integer ops plus a bitcast) costs about the same as int8.

## Accuracy: WikiText-2 perplexity

<!-- ppl-table -->
Protocol: Qwen/Qwen2.5-0.5B-Instruct, WikiText-2 raw test, first 32768 tokens (+1 target), ctx 1024, stride 512, 32768 scored tokens, METAL, activations float16. Generated by `bench/bench_ppl.py` on 2026-10-16.

| weights | perplexity | Δ vs f16 | linear weights | device weights (total) | eval time |
|---|---|---|---|---|---|
| float16 | 13.2014 | — | 682.5 MiB | 942.4 MiB | 24 s |
| int8 (per-channel) | 13.1729 | -0.22% | 341.8 MiB | 601.7 MiB | 21 s |
| int4 asym (g128) | 15.7149 | +19.04% | 181.3 MiB | 441.2 MiB | 22 s |
| int4 sym (g128) | 17.6320 | +33.56% | 176.0 MiB | 435.9 MiB | 21 s |
| fp8 e4m3 (per-tensor) | 13.2692 | +0.51% | 341.3 MiB | 601.2 MiB | 22 s |
<!-- /ppl-table -->

- The gate is int8 within +1% of f16, and `bench/bench_ppl.py` exits nonzero otherwise. The int4 and fp8 deltas are
  measured and documented, not gated.
- fp8 e4m3 with a single scale per tensor costs +0.5%. That is a little more than per-channel int8 (-0.2%), at the
  same size. e4m3 has only 3 mantissa bits, but it is a float format, so one scale covers each tensor's range.
- "device weights" counts every weight on the device: the linears, the f16 lm_head (260 MiB, tied to the embeddings),
  the norms, the biases and the RoPE tables. The host embedding table (an mmap of the bf16 checkpoint, 260 MiB) is
  not included.
- Both int4 variants are plain round to nearest, with no calibration. A 0.5B model is known to be sensitive to 4-bit
  RTN, and its int4 delta is large. I tried a data-free per-group MSE clipping search (shrinking the range by up to
  38%), and it made perplexity worse (16.33 -> 16.74 on the first 4k tokens), so it is not used. Activation-aware
  methods (AWQ / GPTQ) are the way to close the gap, and so is a larger model, which is less sensitive. Such
  checkpoints load with `import_checkpoint` (above).

## CUDA: FP8 W8A8 on the e4m3 tensor cores, and Qwen2.5-7B perplexity (P4-CUDA-SERVE)

On CUDA sm_89+ (Ada, Hopper), `fp8` is no longer weight-only. It runs W8A8 on the FP8 GEMM from P3-GEMM:

- **Weights.** Each linear is quantized on the device to per-tensor e4m3 (`fusezap/serve/cuda.py:quantize_fp8_weight`).
  The codes are the same uint8 e4m3 bits as the host encoder, with s = max|w| / 448. The fused q/k/v and gate/up
  projections each count as one tensor. On a random tensor the device codes and scale are bit-identical to `quantize_weight`
  (tested).
- **Activations.** Each GEMM input row (token) is quantized on the device with its own scale, max|x| / 448, rounding to
  nearest even and saturating. Per-row scales keep every token independent of the others in the step, so batched output
  equals alone output for fp8 too (tested).
- **GEMM.** The e4m3 tensor-core GEMM (wgmma on H100, mma.sync m16n8k32 on Ada) accumulates in f32 and applies the weight
  scale through a device pointer. Each row is then scaled by its activation scale and cast to the model dtype.
- **Requirements.** K % 16 == 0. Older GPUs and METAL keep the weight-only uint8 + bitcast path described above.

int8 / int4 on CUDA:

- **Decode** (32 rows) keeps the composed GEMV, with the dequantize fused into the GEMV.
- **Prefill** (>= 64 rows) dequantizes into a scratch bf16 weight, then runs the 16-bit tensor-core GEMM. Inside a jitted
  step that dequantize reads only constants, so fz.jit runs it once and keeps the result. While a prefill jit exists, an
  int8 model therefore also holds a bf16 copy of its linears. This is a known gap: there is no fused int8 GEMM yet.

<!-- ppl-cuda-table -->
Protocol: Qwen/Qwen2.5-7B-Instruct, the same WikiText-2 protocol as above (raw test, first 32768 tokens, ctx 1024, stride 512, 32768 scored tokens), CUDA on an NVIDIA H100 80GB HBM3, activations bfloat16. Generated by `python -m fusezap.serve.ppl --device CUDA --modes bf16,int8,fp8` on 2026-10-20 (rerun after the lm_head started writing float32 logits on CUDA; before: bf16 6.6231, int8 +0.08%, fp8 +0.89%).

| weights | perplexity | Δ vs bf16 | linear weights | device weights (total) | eval time |
|---|---|---|---|---|---|
| bfloat16 | 6.6216 | — | 12.15 GiB | 13.17 GiB | 64 s |
| int8 (per-channel, weight-only) | 6.6276 | +0.09% | 6.08 GiB | 7.10 GiB | 55 s |
| fp8 e4m3 W8A8 (per-tensor weights, per-token activations) | 6.6811 | +0.90% | 6.08 GiB | 7.09 GiB | 98 s |
<!-- /ppl-cuda-table -->

- fp8 W8A8 costs +0.90% perplexity at 7B. That is more than weight-only fp8 cost at 0.5B on METAL (+0.51%), because the
  activations are now quantized too. int8 weight-only stays within +0.1%.
- The fp8 eval is slower than bf16 here. Each linear adds a row-quantize kernel and a separate scale/cast kernel, and the
  FP8 GEMM reaches only 0.5-0.8x of cuBLAS at these shapes (P3-GEMM). Fusing the quantize into the previous op, and the
  scale into the GEMM epilogue, is open work.
- Teacher-forced top-1 agreement with the f32 HF reference, on the H100 (tests/test_serve_cuda.py):
  - **7B bf16 (SPEC §0 item 3 gate, >= 99%):** fusezap 0.9932 over 1024 positions (8 chat prompts x 128 greedy tokens;
    per prompt 1.000-0.977); HF transformers' own bf16 0.9912 (`test_7b_bf16_top1_vs_hf_spec`, FZ_7B=1).
  - 0.5B bf16: fusezap 0.969 over 97 positions (HF transformers' own bf16: 0.979). bf16 itself is below 99% at 0.5B.
  - 0.5B fp8 W8A8: 0.897 (0.887 before the float32 logits).

## Not done

- **Native FP8 dtypes.** On CUDA sm_89+, fp8 now runs W8A8 on the FP8 GEMM (see the CUDA section). The codes are still
  uint8 tensors fed to custom kernels. The reserved `float8_e4m3`/`float8_e5m2` dtypes and a scale operand on CAST
  (SPEC §4, §11.3) belong to the integrator. METAL fp8 stays weight-only.
- **Calibrated int4 accuracy.** The importer is tested on synthetic checkpoints only (see above). The int4 rows in
  the table are data-free RTN.
- There is no custom Metal int4/int8 GEMM. The library GEMM hook only matches f32 matmuls (schedule.py is not this
  package's), so quantized prefill dequantizes per call.
