#!/usr/bin/env bash
# NKISIM env (docs/NKI_BACKEND.md §4 exception 5): nki 0.6.0 in its own venv, never in the main package's env.
# Linux x86_64 + Python 3.12 only (the wheel is cp312 linux_x86_64), e.g. a small cloud CPU instance:
#   scripts/nki_env.sh && FZ_TEST_TIERS=NKISIM uv run pytest tests/test_ops.py
# Creates .venv-nki next to the repo (FZ_NKI_VENV overrides); the NKISIM device finds its python there.
# `--cc` also installs neuronx-cc 2.x (2.27, the release that pairs with nki 0.6.0; 2 GB) for the compile-to-NEFF path (no device needed).
# `--nrt` also creates .venv-nrt (FZ_NRT_VENV overrides): numpy + nrtpy 2.35.30, the runtime of the persistent NEURON path
# (fusezap/device/nrt_worker.py; its own venv because the nrtpy and nki wheels conflict). Needs libnrt (an inf2/trn1 DLAMI) to run.
set -euo pipefail
cd "$(dirname "$0")/.."
CC=0; NRT=0
for a in "$@"; do case "$a" in --cc) CC=1 ;; --nrt) NRT=1 ;; *) echo "nki_env.sh: unknown flag $a" >&2; exit 2 ;; esac; done
V="${FZ_NKI_VENV:-.venv-nki}"
IDX=(--extra-index-url https://pip.repos.neuron.amazonaws.com)
if [ ! -x "$V/bin/python" ] || ! "$V/bin/python" -c "import nki" 2>/dev/null; then
  if command -v uv >/dev/null; then
    uv venv -q --clear -p 3.12 "$V"
    uv pip install -q -p "$V/bin/python" "nki==0.6.0" "${IDX[@]}" --index-strategy unsafe-best-match
  else
    python3 -m venv --clear "$V"
    "$V/bin/pip" install -q "nki==0.6.0" "${IDX[@]}"
  fi
fi
if [ "$CC" = 1 ] && ! "$V/bin/python" -c "import neuronxcc" 2>/dev/null; then
  uv pip install -q -p "$V/bin/python" "nki==0.6.0" "neuronx-cc==2.27.*" "${IDX[@]}" --index-strategy unsafe-best-match
fi
"$V/bin/python" -c "import nki, sys; print('nki', getattr(nki, '__version__', '?'), 'python', sys.version.split()[0], 'at', sys.executable)"
if [ "$NRT" = 1 ]; then
  N="${FZ_NRT_VENV:-.venv-nrt}"
  if [ ! -x "$N/bin/python" ] || ! "$N/bin/python" -c "import nrtpy" 2>/dev/null; then
    uv venv -q --clear -p 3.12 "$N"
    uv pip install -q -p "$N/bin/python" numpy "nrtpy==2.35.30" "${IDX[@]}" --index-strategy unsafe-best-match
  fi
  "$N/bin/python" -c "from importlib.metadata import version as v; import sys; print('nrtpy', v('nrtpy'), 'numpy', v('numpy'), 'at', sys.executable)"
fi
