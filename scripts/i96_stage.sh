#!/usr/bin/env bash
# I96: stage what the klein NEURON jobs read into .scratch/i96/ (gitignored, but rsynced to the instance / pod: the runners
# exclude /results/, not .scratch/). Run on the Mac before scripts/aws_remote.sh job ... nki_klein_inf2.sh.
#   scripts/i96_stage.sh [ref dir, default results/diff/cpu_f32] [prompts, default "0 1"]
# refs/: diffusers' float32 CPU reference (scripts/diff_ref.py): embeds/p<i>.npz, denoise/{512,1024}_p<i>_s0.npz,
# decode/{512,1024}_p<i>_s0.npz, meta.json (~90 MB for 2 prompts). neff_<target>.tar: a NEFF cache from the W4 pod job
# (scripts/nki_klein_neff_pod.sh), copied here from the fetched results if present.
set -euo pipefail
cd "$(dirname "$0")/.."
REF=${1:-results/diff/cpu_f32}
PROMPTS=${2:-"0 1"}
D=.scratch/i96/refs
mkdir -p "$D/embeds" "$D/denoise" "$D/decode"
cp "$REF/meta.json" "$D/"
for p in $PROMPTS; do
  cp "$REF/embeds/p$p.npz" "$D/embeds/"
  for s in 512 1024; do
    cp "$REF/denoise/${s}_p${p}_s0.npz" "$D/denoise/"
    [ -f "$REF/decode/${s}_p${p}_s0.npz" ] && cp "$REF/decode/${s}_p${p}_s0.npz" "$D/decode/"
  done
done
for t in inf2 trn1; do
  f=$(ls -t "$HOME"/.cache/fusezap/remote/results/*/nki_klein_neff/neff_$t.tar 2>/dev/null | head -1 || true)
  [ -n "$f" ] && cp "$f" .scratch/i96/ && echo "NEFF cache for $t: $f"
done
du -sh .scratch/i96/* | sed 's/^/staged: /'
