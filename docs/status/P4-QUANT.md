# P4-QUANT: weight-only int8 / int4 / fp8 quantization, AWQ/GPTQ import + WikiText-2 perplexity (SPEC §11.3)

**Status:** done (Mac half). FP8 e4m3 with per-tensor scales (weight-only) and AWQ/GPTQ 4-bit checkpoint import were
added after review. The native float8 dtypes and the FP8 GEMM remain for the integrator and the H100 half.

**Gate:** `uv run pytest tests/test_quant.py -q && uv run python bench/bench_ppl.py`
**Result:** passed (rc=0). `66 passed in 26.34s` (METAL). The bench prints `gate: int8 ppl 13.1729 vs f16 13.2014
(-0.216%, limit +1%): PASS`.

## Review fixes
1. **FP8 (spec gap).** `quantize_weight(w, "fp8")` and `quantize_model(m, "fp8")` now work (aliases `fp8-e4m3`, `e4m3`).
   - e4m3fn codes are stored in uint8 with one scale per tensor (max |w| / 448).
   - The host codec `fp8_e4m3_encode`/`fp8_e4m3_decode` rounds to nearest even and saturates.
   - On the device, the dequantize is the bitcast trick in docs/quant.md, and it fuses into the decode GEMV.
   - WikiText-2: 13.2692 (+0.51% vs f16) at int8 size.
   - Regression tests:
     - `test_fp8_codec`: bit-identical to `torch.float8_e4m3fn` on 50k values and every midpoint, ties, saturation.
     - `test_fp8_quant_per_tensor`: error bounds.
     - fp8 was added to the device-layer, fused-GEMV, bytes and whole-model tests (logits vs the dequantized twin,
       and 100% teacher-forced top-1 agreement).
2. **AWQ/GPTQ import (spec gap).** `import_checkpoint(path, fmt=None, dtype, device, **kw)` returns a Llama.
   - `gptq_weight` reads qweight/qzeros/scales/g_idx: v1 (+1) and gptq_v2 zeros, sym detection, act-order `g_idx`,
     and `group_size` -1.
   - `awq_weight` reads the AutoAWQ GEMM packing.
   - `checkpoint_format` validates the config, and `concat_weights` fuses q/k/v and gate/up.
   - Regression tests:
     - Reference packers and dequant formulas from AutoGPTQ/AutoAWQ, checked bit-exact (v1, v2, sym, act-order).
     - Act-order `QuantLinear` on METAL (fused and library paths).
     - End to end, synthetic 2-layer Qwen2.5 GPTQ/AWQ/act-order safetensors checkpoints. GPTQ and AWQ import to
       the same codes, scales and logits as `quantize_model`, and act-order matches the dequantized reference model.
     - Error cases: 8 bits, AWQ gemv, a config that is not quantized.

| weights | WikiText-2 ppl | Δ vs f16 | linear weights | device weights |
|---|---|---|---|---|
| float16 | 13.2014 | — | 682.5 MiB | 942.4 MiB |
| int8 (per-channel) | 13.1729 | -0.22% | 341.8 MiB | 601.7 MiB |
| int4 asym (g128) | 15.7149 | +19.04% | 181.3 MiB | 441.2 MiB |
| int4 sym (g128) | 17.6320 | +33.56% | 176.0 MiB | 435.9 MiB |
| fp8 e4m3 (per-tensor) | 13.2692 | +0.51% | 341.3 MiB | 601.2 MiB |

Protocol: Qwen2.5-0.5B-Instruct, WikiText-2 raw test split joined with "\n\n", first 32768 next-token predictions,
ctx 1024, stride 512 (HF's strided recipe), METAL, f16 activations. Full table and notes: `docs/quant.md`.

## Built
- `fusezap/serve/quant.py`
  - `quantize_weight(w, mode, group=128) -> QuantWeight` (numpy, RTN) and `dequantize_weight` (the numpy reference).
    - `int8`: symmetric, one scale per output channel.
    - `int4` (= `int4-asym`): scale and zero point per group of 128.
    - `int4-sym`: GPTQ's symmetric grid, (q - 8) * s.
    - All-zero rows and groups get scale 1. Inputs are zero-padded to whole groups.
  - `pack_int4` / `unpack_int4`: two codes per byte, half-split per group (byte j = column j low, column j+64 high).
    This is 2-3x faster in the fused GEMV than the interleaved layout.
  - `QuantLinear`: a drop-in for a (out, in) weight Tensor.
    - `Tensor.linear` only uses `.ndim` and `.transpose(-1, -2)`, which returns the lazy dequantize expression, so
      `llama.py` is untouched.
    - In decode (FZ_CUSTOM=0) the dequantize fuses into the GEMV: one kernel reads the codes and scales and
      dequantizes in registers.
    - In prefill (FZ_CUSTOM=1, f32 operands) the scheduler materializes the dequantized weight, then runs the
      library GEMM (dequantize per call).
    - Also provides `host()`, `nbytes`, `numpy()` and `__call__`.
  - `quantize_model(model, mode, group=128, head=False)`: swaps wqkv/wo/wgu/wd of every layer (and the lm_head if
    `head=True`) in place and resets the model's decode and prefill jits.
  - `weight_bytes(model)`: linear, lm_head, other and device total, plus the host embedding table.
  - `"fp8"`: e4m3 weight-only with per-tensor scales (see Review fixes). `import_checkpoint`: AWQ/GPTQ 4-bit.
- `bench/bench_ppl.py`
  - A built-in parquet reader (thrift compact protocol, snappy, RLE/dictionary). Neither `datasets` nor `pyarrow` is
    installed. It reads all 4358 test rows.
  - `NLLEval`: one `fz.jit` step per window. It runs the forward over all positions with a no-op KV cache, then
    log-softmax and the target pick on the device, so only (ctx,) floats come back.
  - Runs f16, int8, int4 and int4-sym. It prints the table, rewrites the table section of `docs/quant.md`, and exits 1
    if int8 is more than +1% over f16.
- `tests/test_quant.py` (66 tests; the original 43 are listed below, the new ones under Review fixes)
  - Host quantization:
    - int8 per-channel round trip and error bound (<= s/2).
    - int4 packing layout and round trip.
    - int4 sym and asym error bounds at I = 256, 200 and 130 (padding).
    - Mode aliases and errors (fp8, unknown mode, AWQ/GPTQ import).
  - `QuantLinear` on METAL vs the numpy dequantized reference matmul:
    - Covers 3 modes x f32/f16 x 2 shapes x the fused and library-GEMM paths, with a batched input and a bias.
    - Device dequantization is bit-exact to numpy.
  - Fusion check: the decode-path GEMV is one kernel that reads the int8/uint8 codes, with no full-size float buffer.
  - Whole model (Qwen2.5-0.5B, f32):
    - Prefill and 4 jitted decode steps (the jit captures, then replays) of the int8 and int4 models match a model
      holding the numpy-dequantized weights (atol 2e-3).
    - Teacher-forced top-1 agreement with the unquantized model is >= 90% for int8 (measured 94%) and >= 70% for int4
      (measured 83%).
    - The weight footprint is checked, and re-quantizing raises.
  - Bench pieces: snappy (with an overlapping copy), the window protocol (every prediction scored exactly once, and
    later windows have >= ctx-stride context), the parquet reader on the cached WikiText test file, and the bench's
    on-device NLL vs HF torch f32 cross-entropy on a 64-token window (atol 2e-3).
- `docs/quant.md`: formats, how quantized layers run, the perplexity table and the gaps.

## Speed (METAL, M5 Max, GPU shared with Chrome; very noisy)
- Isolated GEMV, best GPU time (BEAM 4, f16 activations):

  | shape | f16 | int8 | int4 |
  |---|---|---|---|
  | 9728x896 | 26.0 us | 20.1 us | 10.2 us |
  | 896x4864 | 13.5 us | 12.0 us | 7.8 us |

- Decode step, summed GPU time (best run): f16 4.8 ms, int8 3.5 ms, int4 3.7 ms. The lm_head stays f16 (about 0.6 ms).
- Wall decode, best of 3 runs: f16 60, int8 105, int4 89 tok/s. A later run with fp8 (same GPU contention caveat)
  gave f16 91, int8 97, fp8 94 and int4 99 tok/s. Repeated runs of the same configuration varied by 3-10x.
  Batch-1 decode is launch-bound (about 266 launches at ~32 us of host time each).
- Perplexity eval throughput is about 400-1450 scored tok/s per mode (each window runs a 1024-token forward).

## Known gaps
- The int4 RTN delta is large on this 0.5B model (+19% asym). A data-free MSE clipping search made it worse
  (16.33 -> 16.74 on the first 4k tokens) and was removed. AWQ/GPTQ (activation-aware) would be the fix.
- AWQ/GPTQ import is verified on synthetic checkpoints only. No real calibrated checkpoint is cached, and
  downloading other models is not allowed here, so calibrated int4 perplexity is not measured. Unsupported, and it
  raises: bits other than 4, AWQ gemv, and partly-quantized fused projections. The act-order scale/zero lookup is a
  one-hot gather over the groups on every call (slower; it was not isolated in a timing).
- FP8 is weight-only and decoded with integer ops plus a bitcast. The native float8 dtypes, the scaled CAST and the
  FP8 GEMM (activations in fp8) are for the integrator and P3-GEMM (H100).
- There is no custom Metal quantized GEMM. Quantized prefill dequantizes per call into a scratch buffer, because the
  library-GEMM hook only matches f32 matmuls (schedule.py is not this package's file).
- The lm_head/embeddings are not quantized by default (`head=True` quantizes the lm_head; it is not in the bench).
- bf16 models: the dequantize math runs in f32 and is then cast. Neither tested nor benched.

## Interface issues (for the integrator)
1. `QuantLinear` relies on `Tensor.linear` touching only `weight.ndim` and `weight.transpose(-1, -2)`. If `linear`
   changes (e.g. `weight.T`, a dtype check), keep that duck-typing working, or add a sanctioned "lazy weight" protocol.
2. `quantize_model` resets `model._decode_jit` and `model._prefill_jits` (Llama internals). An engine that keeps its
   own jits (P4-SCHED `engine.py`) must be built after quantization, or offer a reset hook.
3. A `"qmatmul"` custom-kernel pattern in `schedule._match_custom` (int8/uint8 codes x scales) would let prefill use
   a Metal dequantize-in-GEMM kernel instead of materializing the dequantized weight. SPEC §11.1 patterns are
   integrator-owned.
4. FP8: fusezap still has no `float8_e4m3`/`float8_e5m2` dtypes and no scale operand on CAST (SPEC §4/§11.3,
   integrator-owned dtype.py/ops.py). quant.py works around this with uint8 codes, integer ops and an int16->f16
   bitcast. Once the dtypes exist, `QuantLinear.dequantize` for fp8 should become `q.bitcast(float8_e4m3).cast(cdt) * s`.
5. Reading parquet needed a hand-written reader. If more datasets are used later, consider `pyarrow` as a dev
   dependency.
