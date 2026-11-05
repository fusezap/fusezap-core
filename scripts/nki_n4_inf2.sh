#!/usr/bin/env bash
# NKI-N4 rerun ("N4b") on one inf2.xlarge (docs/NKI_BACKEND.md §8 N4 and "N4 run 1"; run-1 logs: docs/status/nki-n4-run1/).
# Run from the repo root on the instance (about 75 min expected, 2 h hard limit):
#   bash scripts/nki_n4_inf2.sh
# Runs on the Neuron multi-framework DLAMI (Ubuntu 24.04, Python 3.12, libnrt in /opt/aws/neuron). Every phase has its own
# time limit inside FZ_N4_BUDGET_MIN (default 105 min, so the runner's 2 h limit keeps a margin for boot and fetch); a phase that
# fails or runs out of time is recorded and the next one runs. Everything lands in results/nki_n4/, with SUMMARY.txt last.
# The NEURON tier runs on the persistent path (FZ_NEURON_RT=nrt, the default): NEFFs compiled once (compile only, in .venv-nki,
# disk-cached under ~/.cache/fusezap/neff), loaded once into the .venv-nrt runtime worker, launched on device-resident buffers.
# Phases run one at a time: only one process may hold a NeuronCore (pytest -n 2 pins one core per xdist worker).
#   1 env       neuron-ls, Neuron packages, the AMI (results/aws_instance.json); the DLAMI venv listing is in run 1's env.log
#   2 setup     uv sync, clang (the CPU test tier; run 1 had only gcc), scripts/nki_env.sh --cc --nrt (.venv-nki: nki 0.6.0 +
#               neuronx-cc 2.27; .venv-nrt: nrtpy 2.35.30), the NEURON tier up on the persistent path
#   3 launch    (a) loaded-NEFF launch latency first: probe --only nrt,neff (tiny + 1 MiB add: first launch, warm launch,
#               execute alone, nrtpy device/host benchmark, a 200-launch dependent chain, copy-in/out 4 KiB..16 MiB)
#   4 nrtpy     the raw runtime on the probe's NEFFs (scripts/nki_n4_nrtpy.py, now passing data0.must_alias_input)
#   5 jit       the nki.jit fallback for comparison (FZ_NEURON_RT=jit, 5 warm calls; run 1: ~1.31 s per call)
#   6 numerics  (b) probe --only numerics: ScalarE transcendentals, fp32 sums, TensorE matmul/attention, int32, NaN/Inf vs sim + f64
#   7 custom    (c) tests/test_custom.py + tests/test_neuron.py + tests/test_neuron_nrt.py on NEURON (persistent path)
#   8 gate      (d) scripts/nki_n3_gate.py --quick on NEURON (Qwen-shaped matmul/attention cases vs float64)
#   9 ops       (e) tests/test_ops.py on NEURON with the time that is left, 2 xdist workers (one NeuronCore each);
#               NKISIM recorded 1386 passed / 48 skipped / 9 xfailed
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:/opt/aws/neuron/bin:$PATH"
export LD_LIBRARY_PATH="/opt/aws/neuron/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OUT=results/nki_n4
mkdir -p "$OUT"
T0=$(date +%s)
BUDGET=$(( ${FZ_N4_BUDGET_MIN:-105} * 60 ))
STATUS="$OUT/phases.tsv"
: > "$STATUS"

left() { echo $(( T0 + BUDGET - $(date +%s) )); }
# phase <name> <max seconds> <command...>: runs with a time limit (the smaller of max and what is left), logs to $OUT/<name>.log
phase() {
  local name=$1 cap=$2; shift 2
  local l; l=$(left)
  local lim=$(( cap < l ? cap : l ))
  if [ "$lim" -lt 60 ]; then
    printf '%s\tSKIPPED (out of time)\t0\n' "$name" >> "$STATUS"; echo "== $name: skipped, out of time"; return 0
  fi
  echo "== $name (limit ${lim}s) $(date -u +%H:%M:%S)"
  local st; st=$(date +%s)
  if declare -F "$1" >/dev/null; then set -- bash -c "$(declare -f "$1"); $1"; fi   # a function of this script
  timeout --kill-after=30 "$lim" "$@" 2>&1 | tee "$OUT/$name.log"
  local rc=${PIPESTATUS[0]}
  local dt=$(( $(date +%s) - st ))
  local what="exit $rc"; [ "$rc" = 124 ] && what="TIMEOUT after ${lim}s"
  printf '%s\t%s\t%s\n' "$name" "$what" "$dt" >> "$STATUS"
  echo "== $name: $what (${dt}s)"
  pkill -f "fusezap/device/(nrt|nkisim)_worker.py" 2>/dev/null || true   # a timed-out phase must not leave a worker holding a core
  return 0
}

env_info() {
  echo "date: $(date -u +%FT%TZ)"; uname -a; nproc; free -g; df -h / | tail -1
  echo "--- instance / AMI (written by the runner)"; cat results/aws_instance.json 2>/dev/null || echo "(no results/aws_instance.json)"
  echo "--- /dev/neuron*"; ls -l /dev/neuron* 2>&1
  echo "--- neuron-ls"; neuron-ls 2>&1 || echo "neuron-ls failed"
  echo "--- Neuron packages"; dpkg -l 2>/dev/null | awk '/aws-neuronx/ {print $2, $3}'
  echo "--- compilers"; command -v cc gcc clang 2>&1
}

setup() {
  set -e
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v clang >/dev/null || (sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq clang)
  uv sync
  scripts/nki_env.sh --cc --nrt
  .venv-nki/bin/python -c "from importlib.metadata import version as v; print('venv-nki: nki', v('nki'), 'neuronx-cc', v('neuronx-cc'))"
  .venv-nrt/bin/python -c "from importlib.metadata import version as v; print('venv-nrt: nrtpy', v('nrtpy'), 'numpy', v('numpy'))"
  uv run python -c "from fusezap.device import Device; d = Device['NEURON']; print('NEURON tier up:', type(d).__name__, d.compiler.cache_key(), d.runtime.__name__, type(d.allocator).__name__)"
}

PY="uv run python"
phase env 120 env_info
phase setup 1800 setup
phase launch 900 env FZ_DEVICE=NEURON $PY scripts/nki_n4_probe.py --out "$OUT" --only nrt,neff --iters 300 --json probe_nrt.json
phase nrtpy 300 .venv-nrt/bin/python scripts/nki_n4_nrtpy.py --neff-dir "$OUT/neff" --out "$OUT/nrtpy.json" --iters 1000
phase jit 300 env FZ_DEVICE=NEURON FZ_NEURON_RT=jit $PY scripts/nki_n4_probe.py --out "$OUT" --only launch --iters 5 --json probe_jit.json
phase numerics 1200 env FZ_DEVICE=NEURON $PY scripts/nki_n4_probe.py --out "$OUT" --only numerics --json probe_numerics.json
phase custom 1500 env FZ_TEST_TIERS=NEURON FZ_NEURON_XDIST_CORES=2 uv run pytest tests/test_custom.py tests/test_neuron.py tests/test_neuron_nrt.py -n 2 -v -rfE -p no:cacheprovider
phase gate 1200 env FZ_DEVICE=NEURON $PY scripts/nki_n3_gate.py --quick --json "$OUT/n3_gate.json"
phase ops 3600 env FZ_TEST_TIERS=NEURON FZ_NEURON_XDIST_CORES=2 uv run pytest tests/test_ops.py -n 2 -v -rfE -p no:cacheprovider

count() { grep -cE "\] $1 " "$2" 2>/dev/null; }   # pytest -v under xdist: "[gw0] [ 12%] PASSED tests/..."
summary() {
  echo "================ NKI-N4b on $(jq -r '.instance_type + " " + .id + " (" + .ami_name + ")"' results/aws_instance.json 2>/dev/null || echo inf2) ================"
  echo "wall $(( ($(date +%s) - T0) / 60 )) min of ${FZ_N4_BUDGET_MIN:-105}"
  echo "--- phases (name, result, seconds)"; cat "$STATUS"
  echo "--- versions"
  grep -E "^aws-neuronx-(runtime-lib|dkms|tools|collectives)" "$OUT/env.log" 2>/dev/null
  grep -E "venv-nki:|venv-nrt:|NEURON tier up" "$OUT/setup.log" 2>/dev/null
  for p in launch jit numerics; do
    echo "--- probe: $p"; sed -n '/==== NKI-N4 probe summary/,$p' "$OUT/$p.log" 2>/dev/null | tail -n +2 | cut -c1-260
  done
  echo "--- raw runtime (nrtpy)"; sed -n '/==== nrtpy summary/,$p' "$OUT/nrtpy.log" 2>/dev/null | cut -c1-260
  for t in custom ops; do
    f="$OUT/$t.log"; [ -f "$f" ] || continue
    echo "--- $t tests on NEURON: passed $(count PASSED "$f"), failed $(count FAILED "$f"), errors $(count ERROR "$f"), skipped $(count SKIPPED "$f"), xfailed $(count XFAIL "$f")"
    grep -E "^(FAILED|ERROR) " "$f" | head -40
    tail -1 "$f"
  done
  if [ -f "$OUT/n3_gate.json" ]; then
    echo "--- N3 gate on NEURON"
    python3 -c "import json; r=json.load(open('$OUT/n3_gate.json')); r=r.get('rows', r) if isinstance(r, dict) else r; ok=[x for x in r if x.get('ok')]; print(f'{len(ok)} of {len(r)} cases within tolerance'); [print('  FAIL', {k: x.get(k) for k in ('case','name','model','rel','tol','err','error') if k in x}) for x in r if not x.get('ok')][:20]" 2>&1
  else tail -5 "$OUT/gate.log" 2>/dev/null; fi
  echo "--- NEFF cache: $(ls ~/.cache/fusezap/neff 2>/dev/null | wc -l) NEFFs"
}
summary 2>&1 | tee "$OUT/SUMMARY.txt"
exit 0
