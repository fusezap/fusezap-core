#!/usr/bin/env bash
# I95: FLUX.2 [klein] 4B image generation on FuseZap's CUDA backend vs diffusers bf16 on the same GPU, Gate G6 against the
# float32 CPU reference (scripts/diff_ref.py). One GPU per machine; run from the repo root on a machine with an NVIDIA
# GPU, AFTER staging the references (scripts/diff_stage_refs.sh <dir with the diff_ref.py cpu_f32 output>; ~134 MB):
#   FZ_JOB_HOURS=1.75 bash scripts/diff_klein_cuda_job.sh h100
# Everything lands in results/diff/cuda/<gpu tag>/; progress in its logs/job.log.
# Steps (each bounded by the FZ_JOB_HOURS deadline; a failed step is logged and the rest still run):
#   0. GPU facts, host check (exit 3 HOST_CHECK_FAIL on a bad host: retry on another pod; HOST_CHECK=0 runs anyway)
#   1. uv sync --extra serve --group dev (torch + diffusers + openai; one retry), torch.cuda check
#   2. kernel probe: tests/test_klein_image.py with FZ_DEVICE=CUDA (the real FuseZap klein modules on a tiny model vs
#      diffusers, ~1 min) -> logs/probe.log; informational, the e2e smoke below is what skips FuseZap
#   3. the 4B weights at the reference's revision (meta.json), HF token from ~/.cache/huggingface/token (never printed)
#   4. scripts/diff_klein_e2e.py --device CUDA: df-bf16 (baseline first, so it is kept whatever happens next), fz-bf16
#      (256^2 smoke first: a missing kernel / compile error stops the FuseZap phases and is saved in fz-bf16.error.json),
#      fz-bf16-ref; on a GPU under 40 GB the float32 encoder is staged (encode all prompts, release, then load the
#      transformer + VAE) and the VAE is bf16 (KleinImagePipeline's default there). If fz-bf16 still runs out of device
#      memory, it is retried once with FZ_KLEIN_VAE_DTYPE=bfloat16 (vae_bf16/).
#   5. serve-image smoke: fusezap serve-image flux2-klein (bf16, :18100) + the openai client, 2 x 1024^2 images
#      (under 40 GB: bf16 text encoder, where the f32 one does not fit beside the transformer, and the bf16 VAE) -> serve_smoke.json
#   6. summary.txt (report.json's summary + phase status)
# Overrides: FZ_JOB_HOURS (default 1.75; keep >= 15 min under --max-hours), PROMPTS (default 0-7), SIZE (1024),
# REF_DIR (.stage/diff_ref), SKIP_SERVE=1, SKIP_PROBE=1, HOST_CHECK=0, MIN_MEM_GB (30).
set -u
cd "$(dirname "$0")/.."
TAG=${1:-${FZ_GPU_TAG:-gpu}}
OUT=results/diff/cuda/$TAG; mkdir -p $OUT/logs
LOG=$OUT/logs/job.log
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $LOG; }
DEADLINE=$(( $(date +%s) + $(python3 -c "print(int(3600 * float('${FZ_JOB_HOURS:-1.75}')))") ))
T() { local left=$(( DEADLINE - $(date +%s) - 60 )); [ $left -lt 30 ] && left=30; [ $1 -lt $left ] && echo $1 || echo $left; }
left() { echo $(( DEADLINE - $(date +%s) )); }
PROMPTS=${PROMPTS:-0-7}; SIZE=${SIZE:-1024}; REF_DIR=${REF_DIR:-.stage/diff_ref}
REPO=black-forest-labs/FLUX.2-klein-4B
log "I95 klein CUDA job on $TAG, commit $(cat .stage/commit 2>/dev/null || echo ?), deadline in $(( $(left) / 60 )) min"

# ---- 0. GPU facts, host check
nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm,power.limit --format=csv 2>&1 | tee -a $LOG
GPU_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | head -1 | awk '{printf "%d", $1/1024}')
log "GPU memory ${GPU_GB:-?} GiB"
python3 scripts/host_check.py --gb ${HC_GB:-2} --min-mem-gb ${MIN_MEM_GB:-30} --min-read-mbs ${MIN_READ_MBS:-200} 2>&1 | tee -a $LOG
if [ "${PIPESTATUS[0]}" = 3 ] && [ "${HOST_CHECK:-1}" != 0 ]; then log "HOST_CHECK_FAIL: stopping (retry on another pod; HOST_CHECK=0 runs anyway)"; exit 3; fi
(nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu --format=csv,noheader -l 5 > $OUT/logs/gpu_timeline.csv 2>&1 &)
SRV_PID=
cleanup() { [ -n "$SRV_PID" ] && { kill -TERM -- -$SRV_PID 2>/dev/null; sleep 3; kill -KILL -- -$SRV_PID 2>/dev/null; }; pkill -f "nvidia-smi --query-gpu=timestamp" 2>/dev/null; }
trap cleanup EXIT
[ -f $REF_DIR/cpu_f32/meta.json ] || { log "no staged references in $REF_DIR/cpu_f32 (run scripts/diff_stage_refs.sh on the laptop before the job); stopping"; exit 2; }
REV=$(python3 -c "import json; print(json.load(open('$REF_DIR/cpu_f32/meta.json'))['revision'])")
log "references: $(ls $REF_DIR/cpu_f32/decode | wc -l) images, $(ls $REF_DIR/cpu_f32/embeds | wc -l) embeds, model revision $REV"

# ---- 1. environment
t0=$(date +%s)
log "uv sync"
timeout $(T 900) uv sync --extra serve --group dev > $OUT/logs/sync.log 2>&1 || timeout $(T 600) uv sync --extra serve --group dev >> $OUT/logs/sync.log 2>&1 \
  || { log "uv sync failed after $(( $(date +%s) - t0 )) s: $(tail -2 $OUT/logs/sync.log | tr '\n' ' '); stopping"; exit 1; }
log "uv sync done in $(( $(date +%s) - t0 )) s"
PY=.venv/bin/python
$PY -c "import torch, diffusers, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-', 'diffusers', diffusers.__version__, 'transformers', transformers.__version__)" 2>&1 | tail -1 | tee -a $LOG
$PY -c "from fusezap.device import Device; print('fusezap CUDA available:', Device.available('CUDA'))" 2>&1 | tail -1 | tee -a $LOG

# ---- 2. kernel probe (tiny model, real modules)
if [ "${SKIP_PROBE:-0}" != 1 ]; then
  log "probe: tests/test_klein_image.py on CUDA"
  FZ_DEVICE=CUDA timeout $(T 600) $PY -m pytest -q -x tests/test_klein_image.py > $OUT/logs/probe.log 2>&1; log "probe exit $?: $(tail -1 $OUT/logs/probe.log)"
fi

# ---- 3. weights (the reference's revision; the token is read by huggingface_hub from ~/.cache/huggingface/token)
t0=$(date +%s); log "download $REPO@$REV"
MP=$(timeout $(T 1500) $PY - "$REPO" "$REV" 2>> $OUT/logs/download.log <<'EOF' | tail -1
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2], allow_patterns=["model_index.json", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*", "scheduler/*"]))
EOF
)
[ -d "$MP/transformer" ] || { log "model download failed: $(tail -3 $OUT/logs/download.log | tr '\n' ' '); stopping"; exit 1; }
log "model at $MP ($(du -sh -L $MP | cut -f1)) in $(( $(date +%s) - t0 )) s"

# ---- 4. end to end (keeps ~15 min for the serve smoke and the fetch)
SERVE_RESERVE=$([ "${SKIP_SERVE:-0}" = 1 ] && echo 120 || echo 900)
E2E_DEADLINE=$(( DEADLINE - SERVE_RESERVE ))
log "e2e: prompts $PROMPTS at ${SIZE}^2, $(( (E2E_DEADLINE - $(date +%s)) / 60 )) min budget"
e2e() { timeout $(( E2E_DEADLINE - $(date +%s) + 60 )) $PY scripts/diff_klein_e2e.py --device CUDA --model "$MP" --ref-root $REF_DIR --ref cpu_f32 \
          --size $SIZE --prompts $PROMPTS --deadline $E2E_DEADLINE "$@"; }
e2e --out $OUT > $OUT/logs/e2e.log 2>&1; RC=$?
log "e2e exit $RC"; grep -E "warm|FAILED|: ok|failed|skipped|killed|out of time" $OUT/logs/e2e.log | tail -40 | tee -a $LOG
if [ -f $OUT/fz-bf16.error.json ] && grep -qiE "out of memory|CUDA_ERROR_OUT_OF_MEMORY|cuMemAlloc" $OUT/fz-bf16.error.json && [ $(( E2E_DEADLINE - $(date +%s) )) -gt 900 ]; then
  log "fz-bf16 ran out of device memory: retrying with FZ_KLEIN_VAE_DTYPE=bfloat16 in $OUT/vae_bf16"
  mkdir -p $OUT/vae_bf16 && cp $OUT/df-bf16* $OUT/vae_bf16/ 2>/dev/null
  FZ_KLEIN_VAE_DTYPE=bfloat16 e2e --out $OUT/vae_bf16 --reuse --phases df-bf16,fz-bf16 > $OUT/logs/e2e_vae_bf16.log 2>&1
  log "e2e (VAE bf16) exit $?"; grep -E "warm|FAILED|failed" $OUT/logs/e2e_vae_bf16.log | tail -12 | tee -a $LOG
fi

# ---- 5. serve-image smoke through the openai client
if [ "${SKIP_SERVE:-0}" != 1 ] && [ $(left) -gt 300 ]; then
  SRV_ENV=""; [ "${GPU_GB:-80}" -lt 40 ] && SRV_ENV="FZ_KLEIN_TE_DTYPE=bfloat16"
  log "serve-image smoke ($SRV_ENV)"
  env FZ_KLEIN_PATH="$MP" $SRV_ENV setsid $PY -m fusezap serve-image flux2-klein --device CUDA --dtype bfloat16 --port 18100 --max-batch 1 \
      --batch-sizes 1 --warmup-sizes ${SIZE}x${SIZE} --log-level warning > $OUT/logs/serve.log 2>&1 &
  SRV_PID=$!
  ok=0; for i in $(seq 1 $(T 900)); do curl -sf http://127.0.0.1:18100/health > /dev/null 2>&1 && { ok=1; break; }; kill -0 $SRV_PID 2>/dev/null || break; sleep 1; done
  if [ $ok = 1 ]; then
    log "server ready after ${i} s"
    timeout $(T 600) $PY - $OUT $SIZE $REF_DIR "$SRV_ENV" >> $LOG 2>&1 <<'EOF'
import base64, io, json, os, sys, time
import numpy as np
from PIL import Image
from openai import OpenAI
sys.path.insert(0, "scripts")
from diff_ref import PROMPTS
from diff_klein_e2e import psnr
out, size, ref_dir, env = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
c = OpenAI(base_url="http://127.0.0.1:18100/v1", api_key="x", timeout=900)
ref = np.asarray(Image.open(os.path.join(ref_dir, "cpu_f32", "decode", f"{size}_p0_s0.png")).convert("RGB"))
rows = []
for k in range(2):
  t0 = time.time()
  r = c.images.generate(prompt=PROMPTS[0], size=f"{size}x{size}", response_format="b64_json", extra_body={"seed": 0, "num_inference_steps": 4})
  dt = time.time() - t0
  img = np.asarray(Image.open(io.BytesIO(base64.b64decode(r.data[0].b64_json))).convert("RGB"))
  Image.fromarray(img).save(os.path.join(out, f"serve_p0_s0_{k}.png"))
  rows.append({"s": dt, "seed": r.data[0].seed, "size": r.size, "psnr_vs_ref": psnr(img, ref)})
  print(f"serve smoke {k}: {dt:.2f} s, {r.size}, PSNR vs ref {rows[-1]['psnr_vs_ref']:.2f} dB", flush=True)
json.dump({"requests": rows, "env": env, "ok": all(r["size"] == f"{size}x{size}" for r in rows)}, open(os.path.join(out, "serve_smoke.json"), "w"), indent=1)
EOF
    log "serve smoke exit $?"
  else
    log "server not ready: $(tail -5 $OUT/logs/serve.log | tr '\n' ' ')"
  fi
  kill -TERM -- -$SRV_PID 2>/dev/null; for i in $(seq 1 30); do kill -0 $SRV_PID 2>/dev/null || break; sleep 1; done; kill -KILL -- -$SRV_PID 2>/dev/null; SRV_PID=
fi

# ---- 6. summary
$PY - $OUT > $OUT/summary.txt 2>&1 <<'EOF'
import json, os, sys
out = sys.argv[1]
for sub in ("", "vae_bf16"):
  fn = os.path.join(out, sub, "report.json")
  if not os.path.exists(fn): continue
  r = json.load(open(fn))
  print(f"== {sub or 'main'}: pass {r['pass']}, fz encode {r.get('fz_encode')}, phases {r.get('phase_status')}")
  for k, v in r.get("summary", {}).items(): print(" ", k, json.dumps(v))
  for row in r["rows"]: print("  p%d s%d" % (row["prompt"], row["seed"]), {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items() if k not in ("prompt", "seed")})
  for k, v in (r.get("errors") or {}).items(): print("  ERROR", k, v)
fn = os.path.join(out, "serve_smoke.json")
if os.path.exists(fn): print("== serve smoke", open(fn).read())
EOF
cat $OUT/summary.txt | tee -a $LOG
log "done ($(( $(left) / 60 )) min left)"
exit $RC
