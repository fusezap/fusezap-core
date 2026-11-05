"""scripts/nki_n5_inf2.sh's plumbing, run on the Mac with every external command stubbed (no uv, no device, no AWS): each of
the ten phases runs and is recorded, function phases see the script's settings, and SUMMARY.txt is written."""
from __future__ import annotations
import os, shutil, subprocess, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUBS = {
  "uv": 'echo "STUB uv $*"; case "$*" in *nki_n5_serve.py*) echo "==== NKI-N5 serve summary ===="; echo "launches  decode_8 245 per forward";; esac',
  "timeout": 'shift; shift; exec "$@"',   # GNU timeout --kill-after=30 LIMIT cmd...
  "curl": "exit 0", "neuron-ls": "echo neuron-ls stub", "pkill": "exit 0", "jq": "exit 1", "dpkg": "exit 0",
  "nproc": "echo 4", "free": "echo free",
}

@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None, reason="needs bash")
def test_n5_script_runs_every_phase(tmp_path):
  (tmp_path / "scripts").mkdir()
  shutil.copy(os.path.join(ROOT, "scripts", "nki_n5_inf2.sh"), tmp_path / "scripts")
  (tmp_path / "scripts" / "nki_env.sh").write_text("#!/bin/bash\necho nki_env stub\n")
  for v in (".venv-nki", ".venv-nrt"):
    (tmp_path / v / "bin").mkdir(parents=True)
    (tmp_path / v / "bin" / "python").write_text("#!/bin/bash\necho venv stub\n")
  stub = tmp_path / "stub"
  stub.mkdir()
  for name, body in STUBS.items(): (stub / name).write_text(f"#!/bin/bash\n{body}\n")
  for p in [*stub.iterdir(), tmp_path / "scripts" / "nki_env.sh", *tmp_path.glob(".venv-*/bin/python")]: p.chmod(0o755)
  env = {"PATH": f"{stub}:/usr/bin:/bin", "HOME": str(tmp_path), "FZ_N5_BUDGET_MIN": "30"}
  p = subprocess.run(["bash", str(tmp_path / "scripts" / "nki_n5_inf2.sh")], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
  assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
  out = tmp_path / "results" / "nki_n5"
  phases = [line.split("\t")[0] for line in (out / "phases.tsv").read_text().splitlines()]
  assert phases == ["env", "setup", "numerics", "ops3", "gate", "kernels", "serve", "tokens", "validate", "ops"], phases
  assert "FZ_NKI_FAST_EXP=1" in (out / "numerics.log").read_text()          # the function phase saw OUT and ran both probes
  val = (out / "validate.log").read_text()
  assert "fusezap serve Qwen/Qwen2.5-0.5B-Instruct --device NEURON --max-num-seqs 8 --num-blocks 256" in val and "--checks accuracy" in val
  assert "launches  decode_8 245" in (out / "SUMMARY.txt").read_text()
