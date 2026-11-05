
---

# Round 11 revision (data: the round 11 run and its vllm-c32.sqlite trace)

## What changes from the round 10 analysis

1. **The "prefill GEMMs are 20-25% slower" claim is withdrawn.** In the same-pod microbench (r11/gemm), the four
   prefill GEMMs together take 1350 us (ours) vs 1252 us (torch) per layer at M=2048, which is +7.8%. At M=4096 the
   gap is about +2%. Our down GEMM is faster than cuBLAS. My vLLM figure came from profile1 forwards at M=1024, where
   cuBLAS runs at 770-840 TF. That does not carry over to our sizes.

2. **The vLLM nsys trace contains no real prefill and cannot answer the prefill question.**
   - The traced rep re-sent the warm-up run's prompts. Both runs used key gate/c32, seed 0, rep 0, against the same
     ns_vllm server (vllm_warm at 04:41:07, vllm_c32 at 04:41:27).
   - In the sqlite, the only prompt-sized forwards are two of 480 tokens. That is 32 requests x about 15 uncached tail
     tokens per wave. A real wave would prefill 32 x 512 = 16384 tokens.
   - The other forwards are 32-row decodes (7112 launches of the 32-row norm, i.e. 254 forwards) plus a few small
     ramp forwards of 1-48 tokens.
   - The run reports 4201 tok/s, TTFT p50 141 ms and a 1.95 s rep, against 3174 tok/s and 2.58 s for the warm run.
     That only fits about 97% prefix-cache hits.
   - Harness hazard: two bench invocations with the same seed against one vLLM server hand vLLM prefix-cache hits. The
     normal bench and Validate are safe, because each rep has its own seed and vLLM is restarted fresh.

3. **What the trace does measure cleanly:** vLLM's 32-row decode step and its idle time between waves.
   - Decode windows 100-880 ms and 1100-1800 ms: **6.446 / 6.481 ms per forward (6.23 ms busy + 0.21 ms idle)**,
     median interval 6.442 ms, measured under nsys.
   - Between waves: the last wave-1 decode is followed by **44.6 ms of full GPU idle** before the first wave-2 kernel.

4. **FuseZap under Validate's config (r11/prof/steps_validate_c32.jsonl), wave 2:**
   - 69.4 ms of GPU idle before the first prefill.
   - Prefill GPU 414.8 ms for 16384 rows: 25.3 us/token, no padding left, no decode forwards in the prefill phase.
   - 127 decode-only steps at a **6.716 ms median interval** (GPU 6.69 ms).
   - About 8 ms of other gaps.
   - Total 1350 ms, matching Validate's 3036 tok/s (2.70 s per rep).
   - **The decode step is not at parity in Validate's config: 6.72 vs 6.44 ms.** In the bench it was 6.59-6.64 ms.

## Revised per-wave decomposition (Validate @32, r11: FuseZap 1349 ms vs vLLM 1245 ms per wave; ratio 0.923)

| component | FuseZap | vLLM | Δ (ms) | basis |
|---|---:|---:|---:|---|
| 127 decode steps | 127 x 6.72 = 853 | 127 x 6.44 = 818 | **+35-39** | measured on both sides (vLLM under nsys) |
| idle before the first prefill of the wave | 69.4 | 44.6 | **+25** | measured (vLLM from the cached-prompt trace; same client) |
| prefill of 16384 prompt tokens (+ ramp) | 415 + 8 | ≈ 382 | **≈ +41** | vLLM side by subtraction: 1245 - 818 - 45 |
| **total** | **1349** | **1245** | **104** | |

The prefill row has the least support, because vLLM's prefill is still not measured directly. It is consistent with
two independent estimates. GEMMs account for +1.3 us/token (98 us/layer x 28 / 2048 rows, i.e. about 21 ms/wave). Our
non-GEMM prefill work (profile1: 278 us/layer at 2048 rows vs about 180 us for vLLM, with attention 85 vs about 45,
SiLU 98 vs about 70, a separate RoPE 37) accounts for about 1.4 us/token, or about 23 ms/wave. Together that is about
44 ms, close to the 41 derived.

Two further observations:

- In-burst slowdown: the first two 2048-row prefill steps of a wave take 47.5-48.4 ms, the later ones 51.7-52.2 ms.
  That looks like a clock or power drop under sustained prefill (FuseZap draws 610 W vs vLLM's 558 W in r7). If true,
  part of the prefill deficit is power, not instruction count.
- Prefill-first + no-padding worked as designed: 0 padding rows, no decode forwards in the prefill phase. It gained
  only +0.5-1.3% because, with the head-start correction, the decode forwards were never worth more than about 1-2%.
  My round 10 prediction of +1.5-3.5% was too high.

**So vLLM's ~8% advantage splits roughly into thirds:**

- decode steps, 0.28 ms/step slower: about 37%
- prefill per token, mostly non-GEMM work, plus about 8% slower GEMMs at M=2048: about 40%
- server turnaround between waves: about 24%

There is no single fundamental architectural flaw. Batch-invariant scheduling costs very little, and prefill-first
already removed what it did cost.

## Revised ranked fixes (predicted gain at Validate @32; ratio 0.923 → about 1.0 if all three land)

| # | fix | gain | effort | batched == alone |
|---|---|---|---|---|
| 1 | **Decode step 6.72 → 6.44 ms.** First check whether the cost scales with max_model_len. Validate runs at 2048 vs 768 in the bench, where FuseZap's step was 6.59-6.64. fz_paged_kpt launches one CTA per 128-key partition of *max_seq_len* (16 partitions at 2048 for a context of at most 640), plus a combine over all of them. If that is the cause: launch only the partitions up to the batch's longest context, and combine each token over its own partitions in a fixed order. Then close the decode GEMM gap at M=32 against vLLM (r11: gate_up 91.2, down 49.2+1.9, qkv 14.9, o 11.5 us/layer). This needs a current FuseZap kernel table. | **+2.5-3%** | low-medium | **Kept**: a token's partitions and sum order depend only on its own context length |
| 2 | **Prefill per token 25.3 → about 23 us.** (a) Per-shape cuBLASLt where it wins: qkv 88 vs 104 us, o 76 vs 82, gate_up 694 vs 794 at M=2048; keep our down. That is about -120 us/layer, -3.4 ms per 2048-row step, about -27 ms/wave. r11 shows the cuBLASLt heuristic is bit-identical across M=2048/4096/8192; also verify the smaller buckets (256-1024) and pin the cuBLAS version. (b) Fuse SiLU*mul into the gate_up epilogue and bias+RoPE into the qkv epilogue, and bring prefill attention to FA3 class: about -100-150 us/layer, -20-30 ms/wave. | **+2% (a), +1.5-2.5% (b)** | (a) low-medium, (b) medium-high | **Kept** if the K order does not depend on M (already true for (a)). (b) changes bits once (one less bf16 rounding), still batch-invariant. |
| 3 | **Turnaround between waves, 69 → about 45 ms.** Diagnose first. Candidates: the finish path (last token → usage chunk → [DONE] across 32 streams that all finish in the same step, on one API event loop), Starlette and JSON parse of 512-int prompts, the pipe to the engine process. | **+1.5-2%** | low-medium once located | **Kept** (host only) |

Lower priority: the in-burst clock drop (check SM clocks / power during a wave; any fix is in the GEMMs anyway),
throughput-mode mixing (the guarantee is lost; little gain now that prefill-first is in), a larger prefill budget for
Validate (about 0.5%).

## Diagnostics for the next run

1. **Redo vLLM c32 under nsys with fresh prompts.** Use `--seed 1` or any seed the warm-up did not use, and keep the
   default prefix caching. Capture one whole rep and check that the forward token counts add up to 32768 prompt tokens.
   Report per-forward tokens, GPU busy/idle, and per-shape GEMM us at vLLM's actual chunk M (about 8192), plus
   attention / act / norm per prefill forward. This gives vLLM's real prefill GPU time per wave.
2. **FuseZap decode graph at 32 rows, context about 600**, with max_model_len 768 / 2048 / 4096, plus nsys of one
   Validate-config decode step: per-kernel table against r11's vLLM table.
3. **nvidia-smi `clocks.sm` / `power.draw` at 100 ms** during a FuseZap and a vLLM wave (gpu.csv already has 5 s
   samples), to confirm or reject the in-burst throttle.
4. **Turnaround timestamps.** In the API process: request received, add sent to the engine, final chunk written. In
   the engine: add received, first scheduled. Add the client send/[DONE] times to Validate's requests.csv. Run both
   engines at c32 so the 69 vs 45 ms can be attributed.
5. **cuBLASLt bit identity at M = 256 / 512 / 1024** (the smaller prefill buckets) for the per-shape choice in fix 2a.

---

# Round 12 revision (fresh-seed vLLM nsys, FuseZap turnaround traces, Validate with and without cuBLASLt)

## 1. vLLM at c32, measured (r12 vllm-c32.sqlite; bench rep 2.584 s, 3171 tok/s, same as its warm run, so no cache hits)

- **The trace holds the whole rep.** 260 forwards and 40904 rows: 32768 prompt tokens plus about 8100 decode rows.
- **Each wave is 3 prefill forwards plus 127 decode forwards.**
  - Wave 1: 2048 / 8192 / 6167 tokens. Wave 2: 512 / 8192 / 7697. That is 16384 prompt tokens per wave, with about 20
    decode rows riding inside.
  - Prefill GPU busy: **326.2 ms (wave 1) / 334.9 ms (wave 2) per wave**.
  - An 8192-token forward is 159.3-163.2 ms busy, **19.5-19.9 us/token**. Of that, GEMM is 139.6-143.4 ms (17.0-17.5
    us/token, about 750-766 TF). gate_up uses nvjet 192x192 coopB: 2893-2977 us at M=8192, about 750-770 TF.
    down + o average 855-878 us per launch, qkv 368-377 us.
  - Non-GEMM work in that forward: attention 5.3 ms, SiLU 8.3, norms 4.2, other 1.9. Total **2.4 us/token**.
- **Decode, 32 rows:** median 6.289 ms busy, **6.503 ms wall** per forward (under nsys). Validate's client sees
  6.31 ms/token on the last-prefilled request.
- **Idle per wave:**
  - About 0.21 ms per decode step, so about 27 ms over 127 steps.
  - Between waves: **84.0 ms of full GPU idle**, then a lone 512-token forward with 37 ms of wall time, then two gaps of
    about 6 ms. The 8192-token forward starts 133 ms after wave 1 ends.
  - r11's 44.6 ms came from the cached-prompt run and is wrong.
- **The two waves:** wave 1 spans 1258 ms, wave 2 1195 ms, with 96 ms from the end of wave 1 to the next prefill. That
  gives 1291 ms per wave, matching 2.584 s for two waves.

## 2. FuseZap's ~69 ms gap between waves is the client, not the server (trace/validate.api.jsonl + engine.jsonl + requests.csv send/done)

Times are relative to the engine finishing the last request of wave 1, over 5 reps:

| segment | ms |
|---|---|
| engine → API, last token written | +0.6-2.7 (one rep +13) |
| API sends [DONE] | +1.1-3.0 |
| **client sees the last [DONE]** | **+52-65** |
| client logs its first wave-2 send | +3-7 (early finishers) |
| **API receives the first wave-2 request** | **+62-69** |
| API parse → submit | 0.4-2.3 |
| engine add | +0.5-1.3 after submit |
| engine schedule | +0.1 |

Over the whole rep, client send → API receive is a median of about 24 ms. At the wave boundary it is about 60 ms,
because the client's single asyncio loop is still draining 32 finished streams and building 32 bodies of 512 ids
(prep about 1.0 ms each). The server side adds at most about 4 ms.

vLLM's gap is the same client cost or more: 84-96 ms of GPU idle in the bench trace, and its wave-2 sends spread over
43-59 ms vs FuseZap's 55-70 ms. **This is not a FuseZap deficit.** It is about 5% of every wave for both engines, so it
pulls the ratio slightly toward 1: with an instant client the ratio would be about 0.916 instead of 0.922.

## 3. Per-wave decomposition, Validate r12 (cuBLASLt on), client view, mean over 5 reps x 2 waves

| | FuseZap | vLLM | Δ |
|---|---:|---:|---:|
| wave span (first send → last [DONE]) | 1374.7 | 1265.2 | **+109.5** |
| prefill phase (first send → last request's first token) | 467.3 | 461.2 | +6 |
| decode tail of the last-prefilled request, per token (client itl_mean) | 7.05 (API side 7.02) | 6.31 | +0.73 x 127 = **+93** |
| rest (end-of-wave client backlog, spread) | | | about +10 |

- **The decode tail is about 85% of the gap.** FuseZap's API-side tail (7.02 ms/token) equals its client-side tail,
  so this is not a client artifact.
- **The r12 decode regression accounts for about half of it.** Bench new vs base: c32 TPOT 8.33 vs 7.91 (+5%), tok/s
  0.976. Removing it puts FuseZap at about 6.67 ms/token (r11 measured 6.72), and the remaining gap to vLLM is
  0.36-0.41 ms x 127, about 46-52 ms/wave.
- **The prefill phase is a wash at the wave level, even though vLLM's prefill GPU time is lower.** FuseZap's prefill
  is about 400 ms of GPU for 16384 tokens (r11 measured 415; cuBLASLt took about 1.3% of a wave off it) vs vLLM's 330.
  vLLM gives that back in its wave ramp. It runs the first arrivals in a small forward (512 or 2048 tokens), then
  needs 2-3 forwards, idling 37+12 ms. FuseZap's prefill-first steps run back to back the moment the requests land.
- **Clocks are not a FuseZap-specific problem.** In gpu_clock_validate.csv, both engines run prefill at a median
  1665 MHz (minimum 1485-1545) and decode at 1965-1980 MHz. Microbenchmarks at 1980 MHz overstate in-situ GEMM speed
  equally for both engines.

## 4. Revised ranked fixes (Validate @32 is 0.922 now)

My last two predictions overshot: prefill-first + no-padding was +0.5-1.3% against +1.5-3.5% predicted, and cuBLASLt
was +1.3% against +2%. The numbers below are the measured or upper-bound figures, with the basis stated.

| # | fix | predicted @32 | basis / confidence | batched == alone |
|---|---|---|---|---|
| 1 | **Revert or fix the r12 kpt decode regression** (FZ_PAGED_KPT_CTAS / the CTA loop). | **+2.5-3.5%** (to about 0.95) | measured: bench base/new c32 0.976, TPOT +5%; high | kept |
| 2 | **Close the 32-row decode step: about 6.67 → 6.31 ms.** First get an FuseZap per-kernel decode table under Validate's config and compare it with r12's vLLM table: gate_up 93.1, down 49.1+1.9 (split-K), qkv+bias 14.9, o 11.5, attention about 22+2.5, norm/act/kv about 11 us per layer, 6.29 ms busy. Test the max_model_len partition hypothesis (2048 in Validate vs 768 in the bench) with the decode-graph sweep. | **+2.5-3.5%** if fully closed; +1-2% realistic | vLLM measured; FuseZap side needs its kernel table; medium | kept (M-independent sums; partitions per token) |
| 3 | **Prefill non-GEMM and in-situ overhead.** FuseZap is about 24.5 us/token at 2048 rows vs vLLM's 19.5 at 8192. GEMM explains about 0.5-1.5 us of that after cuBLASLt; the rest is non-GEMM and unexplained in-situ time, about 5 us/token. In order: an nsys of one Validate-config FuseZap prefill step (per kernel, same clocks); SiLU*mul in the gate_up epilogue (about 98 us/layer, about 20 ms/wave); bias+RoPE in the qkv epilogue (about 8 ms); FA3-class prefill attention (1.16 → about 0.65 us/token, about 8 ms); then any staging copies or gaps the nsys shows. | **+1.5-3%** for the first items, up to about 6% if FuseZap reached vLLM's 19.5 us/token | vLLM side measured; FuseZap breakdown is from profile1 plus subtraction; medium | kept (fusions change bits once, still batch-invariant) |
| - | Client turnaround | 0 for FuseZap's ratio | measured: client-bound for both | - |

All three together: about 0.92 → 0.99-1.02. Items 1-2 are the decode work and carry most of the expected gain; item 3
is real GPU time but has a lower confidence on its size.

## 5. Diagnostics for the next run

1. FuseZap nsys of a Validate-config c32 window (max_model_len 2048, 2048-row prefill) after the regression fix: the
   per-kernel decode table and one prefill step, compared with r12's vLLM forwards. bench/kernel_prof.py already does
   this.
2. Decode graph at 32 rows, context about 600, with max_model_len 768 / 2048 / 4096 (partition hypothesis).
3. Optional: a multi-process or pre-built-body client mode in Validate, to show the 5% client share explicitly. It
   does not change the ranking.
