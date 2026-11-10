#!/usr/bin/env bash
# I95: copy what scripts/diff_klein_e2e.py reads from a scripts/diff_ref.py float32 CPU run into .stage/diff_ref/cpu_f32/
# (gitignored, but not excluded from scripts/remote.sh's rsync, unlike results/), so a pod job gets the exact CPU
# references (GPU float32 torch is not bit-identical to CPU float32, so they are shipped, not regenerated):
#   meta.json, decode/<size>_p<i>_s<seed>.png, embeds/p<i>.npz   (8 prompts at 1024^2: ~134 MB; the .npz latents are not needed)
#   scripts/diff_stage_refs.sh [SRC (default results/diff/cpu_f32)] [SIZE (1024)]
set -eu
cd "$(dirname "$0")/.."
SRC=${1:-results/diff/cpu_f32}; SIZE=${2:-1024}; DST=.stage/diff_ref/cpu_f32
[ -f "$SRC/meta.json" ] || { echo "no $SRC/meta.json"; exit 1; }
rm -rf "$DST"; mkdir -p "$DST/decode" "$DST/embeds"
cp "$SRC/meta.json" "$DST/"
cp "$SRC"/decode/${SIZE}_p*_s*.png "$DST/decode/"
cp "$SRC"/embeds/p*.npz "$DST/embeds/"
git rev-parse --short HEAD > .stage/commit 2>/dev/null || true
echo "staged $(ls "$DST/decode" | wc -l | tr -d ' ') images + $(ls "$DST/embeds" | wc -l | tr -d ' ') embeds ($(du -sh "$DST" | cut -f1)) in $DST"
