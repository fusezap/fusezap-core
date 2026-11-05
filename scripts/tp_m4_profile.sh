#!/usr/bin/env bash
# I60 (I57 M4, speed): where a 70B TP=2 rank's decode step spends its time, kernel by kernel, on ONE H200 (~22 min;
# hard cap 0.5 h). No model download: random bf16 weights at a Llama-3.3-70B TP=2 rank's exact shapes (hidden 8192,
# 32 q / 4 kv heads, intermediate 14336, 80 layers, vocab 128256), fusezap.serve's engine as `fusezap serve
# --max-model-len 8192 --num-blocks 8192 --max-num-seqs 32` builds it, the all-reduces replaced by nothing or by a copy.
#   FZ_BENCH_GIT_COMMIT=<sha> bash scripts/tp_m4_profile.sh      (image: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04)
# (an H100 also works: the script then shares one layer's weights across the 80 layers, --share-layers auto; set
# PEAK=3.35 for its bandwidth.) M3's numbers to explain: decode TPOT 28.4 ms (c1) .. 31.6 ms (c32) vs vLLM TP=2 19.5-21.5;
# the step is GPU-bound (graph 28.47 ms per rank); the linears alone sum to ~17.9 ms + lm_head 0.49 (autotune times),
# NCCL 160 x 8.5 us = 1.4 ms: ~9 ms unaccounted.
# Steps (progress in results/logs/m4.log; every step bounded by the FZ_PROOF_HOURS deadline, default 0.45):
#   0. uv sync (~2-3 min)
#   1. CUDA tests of this branch's kernels (~2 min): RoPE + KV-cache write in one kernel (rope_qkv kvw= vs rope_qkv +
#      kv_write, eager and in graphs), rope_qkv / silu_mul vs the composed kernels, and with Qwen2.5-0.5B (1 GB download)
#      the engine's greedy tokens with FZ_ROPE_KVW=0 / 1 and FZ_FUSE_ROWS=1 (must be identical)
#   2. bench/decode_kernels.py at the 70B TP=2 rank shapes, buckets 1 / 8 / 32, ctx 512 (~8-10 min): per variant the
#      engine TPOT, the decode graph's GPU ms, and each kernel's GPU time in graph order (class table: count per step, ms,
#      % of step, achieved TB/s, ideal ms at PEAK TB/s, expected ms for a vLLM-like kernel, excess). Variants (env):
#        old    FZ_ROPE_KVW=0              M3's decode path (composed RoPE, separate kv_write)
#        kvw    (defaults)                 this branch: RoPE + KV write in one kernel (one launch per layer fewer)
#        rows1  FZ_FUSE_ROWS=1             + the fused SiLU*mul / RoPE kernels at decode rows (bit-identical)
#        old_ar FZ_ROPE_KVW=0,FZ_KPROF_AR=copy   M3's path + a device copy at each of the 160 all-reduce points (a TP rank's
#                                   items; M3's 28.47 ms graph minus this one is what the NCCL kernels really cost in the graph)
#        kpt    FZ_PAGED_TC=0              attention on the key-per-thread decode kernel instead of the tensor-core one
#   3. the long-context point (~3 min): ctx 2048 at buckets 1 / 32, defaults (attention's share as the context grows)
#   4. FZ_CUDA_CARVEOUT=100 (~3 min, a fresh process): every kernel asks for the max-shared L1 split, so the graph never
#      reconfigures it between the GEMMs (~200 KB of shared memory) and the small kernels; compare with step 2's kvw
# VARIANTS / BUCKETS / CTX override step 2; SKIP_TESTS=1 skips step 1; SKIP_LONG=1 step 3; SKIP_CARVEOUT=1 step 4; PRESET (llama70b-tp2;
# qwen72b-tp2 for Qwen2.5-72B's rank) the shapes.
set -u
cd "$(dirname "$0")/.."
R=results; mkdir -p $R/logs
LOG=$R/logs/m4.log
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $LOG; }
DEADLINE=$(( $(date +%s) + $(python3 -c "print(int(3600 * float('${FZ_PROOF_HOURS:-0.45}')))") ))
T_START=$(date +%s)
T() { local left=$(( DEADLINE - $(date +%s) - 60 )); [ $left -lt 30 ] && left=30; [ $1 -lt $left ] && echo $1 || echo $left; }
nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.memory --format=csv 2>&1 | tee -a $LOG
(nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw --format=csv,noheader -l 5 > $R/logs/gpu.csv 2>&1 &)
log "commit ${FZ_BENCH_GIT_COMMIT:-?}; uv sync"
timeout $(T 900) uv sync --extra serve --group dev > $R/logs/sync.log 2>&1 || log "uv sync failed: $(tail -2 $R/logs/sync.log)"
PY=.venv/bin/python
timeout 120 uv pip install --python $PY pytest-timeout >> $R/logs/sync.log 2>&1 || true
PT_TO=$($PY -c "import pytest_timeout" 2>/dev/null && echo "--timeout=300 --timeout-method=thread")
export FZ_HF_DOWNLOAD=1 HF_HUB_ENABLE_HF_TRANSFER=0
PRESET=${PRESET:-llama70b-tp2}
PEAK=${PEAK:-4.8}

# ---- 1. CUDA tests of the new kernels and the token identity
if [ "${SKIP_TESTS:-0}" != 1 ]; then
  log "1. CUDA tests (rope_qkv kvw, rope / silu vs composed, engine tokens FZ_ROPE_KVW=0/1, FZ_FUSE_ROWS=1)"
  timeout $(T 300) $PY -c "from fusezap.serve.weights import resolve_model; print(resolve_model('Qwen/Qwen2.5-0.5B-Instruct', allow_download=True))" >> $LOG 2>&1
  t0=$(date +%s)
  FZ_DEVICE=CUDA FZ_SLOW=1 timeout $(T 480) $PY -m pytest -v -rs -o faulthandler_timeout=200 $PT_TO tests/test_serve_cuda.py \
    -k "kvw or rope_qkv or silu_mul or jit_cuda_graph_replay or rope_kvw" > $R/logs/tests.log 2>&1
  log "tests exit $? in $(( $(date +%s) - t0 )) s"
  grep -E "(PASSED|FAILED|ERROR|SKIPPED)" $R/logs/tests.log | cut -c1-200 | tee -a $LOG
  tail -1 $R/logs/tests.log | tee -a $LOG
fi

# ---- 2. the per-kernel breakdown at 1 / 8 / 32 rows, before / after and the A/B switches
VARIANTS=${VARIANTS:-"old:FZ_ROPE_KVW=0;kvw:;rows1:FZ_FUSE_ROWS=1;old_ar:FZ_ROPE_KVW=0,FZ_KPROF_AR=copy;kpt:FZ_PAGED_TC=0"}
log "2. decode_kernels $PRESET buckets ${BUCKETS:-1,8,32} ctx ${CTX:-512}: $VARIANTS"
t0=$(date +%s)
timeout $(T 900) $PY bench/decode_kernels.py --preset $PRESET --buckets ${BUCKETS:-1,8,32} --ctx ${CTX:-512} --steps 32 --reps 3 \
  --peak-tbps $PEAK --variants "$VARIANTS" --json $R/kprof-$PRESET.json > $R/logs/kprof.log 2>&1
log "decode_kernels exit $? in $(( $(date +%s) - t0 )) s"
grep -vE "^KPROF " $R/logs/kprof.log | grep -vE "^\s*$" | cut -c1-400 | tee -a $LOG

# ---- 3. a longer context (attention's share)
if [ "${SKIP_LONG:-0}" != 1 ]; then
  log "3. decode_kernels $PRESET ctx 2048, buckets 1 / 32, defaults"
  t0=$(date +%s)
  timeout $(T 420) $PY bench/decode_kernels.py --preset $PRESET --buckets 1,32 --ctx 2048 --steps 24 --reps 2 --peak-tbps $PEAK \
    --variants "kvw2k:" --json $R/kprof-$PRESET-ctx2048.json > $R/logs/kprof2k.log 2>&1
  log "decode_kernels (ctx 2048) exit $? in $(( $(date +%s) - t0 )) s"
  grep -vE "^KPROF " $R/logs/kprof2k.log | grep -vE "^\s*$" | cut -c1-400 | tee -a $LOG
fi

# ---- 4. the L1 / shared-memory carveout fixed at max shared for every kernel (a fresh process: it applies per kernel load)
if [ "${SKIP_CARVEOUT:-0}" != 1 ]; then
  log "4. decode_kernels $PRESET with FZ_CUDA_CARVEOUT=100, buckets 1 / 32, defaults (vs step 2's kvw)"
  t0=$(date +%s)
  FZ_CUDA_CARVEOUT=100 timeout $(T 420) $PY bench/decode_kernels.py --preset $PRESET --buckets 1,32 --ctx ${CTX:-512} --steps 24 --reps 2 \
    --peak-tbps $PEAK --variants "kvw_c100:" --json $R/kprof-$PRESET-carveout.json > $R/logs/kprof_c100.log 2>&1
  log "decode_kernels (carveout 100) exit $? in $(( $(date +%s) - t0 )) s"
  grep -E "^== |^total" $R/logs/kprof_c100.log | cut -c1-300 | tee -a $LOG
fi
grep -iE "Traceback|Error" $R/logs/kprof*.log | head -6 | tee -a $LOG
log "GPU clocks over the run (sm / mem MHz, power W; first and last samples): $(head -1 $R/logs/gpu.csv | cut -d, -f4-6) .. $(tail -1 $R/logs/gpu.csv | cut -d, -f4-6)"
pkill -f "nvidia-smi --query-gpu=timestamp" 2>/dev/null
log "done in $(( ($(date +%s) - T_START) / 60 )) min"
