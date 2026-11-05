"""I68 / I69 job scripts (scripts/nki_bf16_trn1.sh, scripts/nki_7b_inf2.sh) run on the Mac with every external command
stubbed (no uv, no device, no AWS): every phase runs or is skipped on purpose, function phases see the script's settings, and
SUMMARY.txt is written. Also scripts/nki_bf16_compare.py on hand-made result files."""
from __future__ import annotations
import json, os, shutil, subprocess, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUBS = {
  "uv": 'echo "STUB uv $*"', "git": 'echo "STUB git $*"', "sudo": "exit 0", "setsid": 'exec "$@"',
  "timeout": 'while [ "${1#-}" != "$1" ]; do shift; done; shift; exec "$@"',   # GNU timeout [--kill-after=30] LIMIT cmd...
  "curl": "exit 0", "neuron-ls": "echo neuron-ls stub", "pkill": "exit 0", "jq": "exit 1", "dpkg": "exit 0",
  "nproc": "echo 4", "free": "echo free", "clang": "exit 0",
}
PY_STUB = '#!/bin/bash\necho "STUB py FZ_NEURON_DTYPE=${FZ_NEURON_DTYPE:-} NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-} $*"\ncase "$*" in *resolve_model*) echo /tmp/model;; esac\n'

def _dry_run(tmp_path, script: str, env_extra: dict, extra: tuple = (), pre=None) -> tuple[subprocess.CompletedProcess, str]:
  (tmp_path / "scripts").mkdir()
  for f in (script, "nki_n6b_lib.sh", "nki_n6b_summary.py", *extra): shutil.copy(os.path.join(ROOT, "scripts", f), tmp_path / "scripts")
  if pre is not None: pre(tmp_path)
  (tmp_path / "scripts" / "nki_env.sh").write_text("#!/bin/bash\necho nki_env stub\n")
  for v in (".venv", ".venv-nki", ".venv-nrt"):
    (tmp_path / v / "bin").mkdir(parents=True)
    (tmp_path / v / "bin" / "python").write_text(PY_STUB)
  stub = tmp_path / "stub"
  stub.mkdir()
  for name, body in STUBS.items(): (stub / name).write_text(f"#!/bin/bash\n{body}\n")
  for p in [*stub.iterdir(), tmp_path / "scripts" / "nki_env.sh", *tmp_path.glob(".venv*/bin/python")]: p.chmod(0o755)
  py3 = os.path.dirname(shutil.which("python3") or "/usr/bin/python3")
  env = {"PATH": f"{stub}:{py3}:/usr/bin:/bin", "HOME": str(tmp_path), **env_extra}
  p = subprocess.run(["bash", str(tmp_path / "scripts" / script)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
  return p, p.stdout + p.stderr

@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None or shutil.which("python3") is None, reason="needs bash, python3")
def test_bf16_trn1_script_runs_every_phase(tmp_path):
  p, log = _dry_run(tmp_path, "nki_bf16_trn1.sh", {"FZ_BF16_BUDGET_MIN": "76"})
  assert p.returncode == 0, log[-4000:]
  out = tmp_path / "results" / "nki_bf16_trn1"
  phases = [ln.split("\t")[0] for ln in (out / "phases.tsv").read_text().splitlines()]
  for want in ("env", "setup", "kernels", "layer_bench", "layer_bench_attn", "fz_bf16", "fz_f32", "fzhttp_bf16", "setup_nxdi",
               "awslib_tp1_bf16_argmax", "awslib_tp1_bf16", "fz_bf16a"):
    assert want in phases, (want, phases)
  assert "FZ_NEURON_DTYPE=bf16" in (out / "fz_bf16.log").read_text()
  http = (out / "fzhttp_bf16.log").read_text()
  assert "--dtype bfloat16" in http and "--checks accuracy" in http and "--target-format bfloat16" in http
  assert "--dtype bf16" in (out / "layer_bench.log").read_text()
  assert "serving layer path (bf16): FZ_NEURON_LAYER=v1" in log   # no layer_bench.json in a dry run: the members
  assert (out / "SUMMARY.txt").exists()

@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None or shutil.which("python3") is None, reason="needs bash, python3")
def test_7b_inf2_script_runs_every_phase(tmp_path):
  p, log = _dry_run(tmp_path, "nki_7b_inf2.sh", {"FZ_7B_BUDGET_MIN": "106"})
  assert p.returncode == 0, log[-4000:]
  out = tmp_path / "results" / "nki_7b_inf2"
  tsv = (out / "phases.tsv").read_text()
  phases = [ln.split("\t")[0] for ln in tsv.splitlines()]
  for want in ("env", "setup", "hbm_probe", "kernels", "setup_nxdi", "aws7b_tp2"): assert want in phases, (want, phases)
  probe = (out / "hbm_probe.log").read_text()
  assert probe.count("scripts/nki_hbm_probe.py --json") == 2
  assert "core 0 alone cannot hold 15 GiB" in tsv and "fz7b" not in phases   # the stubbed probe wrote no result: FuseZap skipped
  assert (out / "SUMMARY.txt").exists()

@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None or shutil.which("python3") is None, reason="needs bash, python3")
def test_n8_bf16_trn1_script_runs_every_phase(tmp_path):
  """N8 (I70) job: every phase runs or is skipped on purpose; with no bench results the choice is the bf16 defaults."""
  p, log = _dry_run(tmp_path, "nki_n8_bf16_trn1.sh", {"FZ_N8_BUDGET_MIN": "76"}, ("nki_n8_bf16_choose.py",))
  assert p.returncode == 0, log[-4000:]
  out = tmp_path / "results" / "nki_n8_bf16_trn1"
  phases = [ln.split("\t")[0] for ln in (out / "phases.tsv").read_text().splitlines()]
  for want in ("env", "setup", "kernels", "layer_bench", "layer_bench_256", "head_bench", "fz_base", "fz_n8", "fz_n8_256", "fzhttp_n8",
               "setup_nxdi", "awslib_tp1_bf16", "nxdi_info", "fz_f32"):
    assert want in phases, (want, phases)
  assert "fz_head" not in phases and "fz_layer" not in phases   # no N8 choice in a dry run: nothing to split
  assert "N8 layer switches: none" in log and "N8 head switches: none" in log
  lb = (out / "layer_bench.log").read_text()
  assert "--dtype bf16" in lb and "v2se+w2048p2" in lb and "b2:2048" in (out / "head_bench.log").read_text()
  assert "--variant n8_256" in (out / "fz_n8_256.log").read_text()
  assert "tests/test_neuron_bf16_n8.py" in (out / "kernels.log").read_text()
  http = (out / "fzhttp_n8.log").read_text()
  assert "--dtype bfloat16" in http and "--checks accuracy" in http
  assert (out / "SUMMARY.txt").exists()

def test_n8_choose_picks_the_fastest_bitwise(tmp_path):
  """scripts/nki_n8_bf16_choose.py: a faster variant that is not bitwise (or not timed in every row) is never chosen."""
  def rec(v, rows, ms, bits=1.0, ctx=512): return {"variant": v, "rows": rows, "ctx": ctx, "device_ms": ms, "logits_bitwise_vs_v1": bits, "pool_bitwise_vs_v1": bits, "tokens_vs_v1": bits == 1.0}
  lay = [rec("v1", 1, 0.34), rec("v1", 4, 0.41), rec("v2se", 1, 0.24), rec("v2se", 4, 0.25), rec("v2se+w2048", 1, 0.20), rec("v2se+w2048", 4, 0.21),
         rec("v2se+w4096", 1, 0.15, 0.5), rec("v2se+w4096", 4, 0.16), rec("v2se+w2048p1q", 1, 0.19), rec("v2se+w2048p1q", 4, 0.20),
         rec("v2se+w2048p1q", 1, 0.19, 0.9, ctx=256)]
  head = [{"variant": "b1-logits", "rows": 1, "device_ms": 1.5, "logits_bitwise_vs_v1": 1.0, "argmax_ok": None},
          {"variant": "b3", "rows": 1, "device_ms": 1.52, "logits_bitwise_vs_v1": 1.0, "argmax_ok": True},
          {"variant": "b3", "rows": 4, "device_ms": 1.55, "logits_bitwise_vs_v1": 1.0, "argmax_ok": True},
          {"variant": "b2:2048", "rows": 1, "device_ms": 1.05, "logits_bitwise_vs_v1": 1.0, "argmax_ok": True},
          {"variant": "b2:2048", "rows": 4, "device_ms": 1.08, "logits_bitwise_vs_v1": 1.0, "argmax_ok": True},
          {"variant": "b2:4096", "rows": 1, "device_ms": 0.9, "logits_bitwise_vs_v1": 1.0, "argmax_ok": False},
          {"variant": "b2:4096", "rows": 4, "device_ms": 0.9, "logits_bitwise_vs_v1": 1.0, "argmax_ok": True}]
  json.dump({"results": lay}, open(tmp_path / "layer_bench.json", "w"))
  json.dump({"results": head}, open(tmp_path / "head_bench.json", "w"))
  run = lambda *a: subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "nki_n8_bf16_choose.py"), *a], capture_output=True, text=True, timeout=60)
  assert run("layer", str(tmp_path)).stdout.strip() == "FZ_NEURON_BF16_WN=2048 FZ_NEURON_BF16_PF=0 FZ_NEURON_BF16_PFQO=0"   # w2048p1q: not bitwise at 256
  assert run("head", str(tmp_path)).stdout.strip() == "FZ_NEURON_BF16_HEAD=1 FZ_NEURON_BF16_HEAD_W=2048"
  for v, ids in (("a", [1, 2]), ("b", [1, 2]), ("c", [1, 3])):
    json.dump({"greedy": {"rows": [{"id": "p0", "ids": ids}]}, "c1": {"ids": ids}}, open(tmp_path / f"serve_{v}.json", "w"))
  assert "TOKEN-IDENTICAL" in run("same", str(tmp_path), "a", "b").stdout and "TOKENS DIFFER" in run("same", str(tmp_path), "a", "c").stdout

def test_bf16_compare_reads_the_results(tmp_path):
  ids = lambda k: [1, 2, 3, 4] if k else [1, 2, 9, 9]
  fz = {"greedy": {"prompts": 2, "exact": 1, "match_tokens": 6, "ref_tokens": 8, "rows": [{"id": "p0", "ids": ids(1)}, {"id": "p1", "ids": ids(0)}]},
        "top1": {"agreement": 0.995, "logprob_abs_err_mean": 0.004}}
  aws = {"greedy": {"prompts": 2, "exact": 1, "match_tokens": 6, "ref_tokens": 8, "rows": [{"id": "p0", "ids": ids(1)}, {"id": "p1", "ids": ids(1)}]}}
  (tmp_path / "serve_bf16.json").write_text(json.dumps(fz))
  (tmp_path / "nxdi_tp1_bf16.json").write_text(json.dumps(aws))
  r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "nki_bf16_compare.py"), "--out", str(tmp_path), "--aws", "tp1_bf16",
                      "--json", str(tmp_path / "c.json")], capture_output=True, text=True, timeout=60)
  assert r.returncode == 0, r.stderr
  c = json.load(open(tmp_path / "c.json"))
  assert c["bars"]["top1_agreement"] == 0.98 and c["bars"]["greedy_exact_match"] == 0.70
  assert c["fz_bf16_vs_aws_bf16_nxdi"] == {"prompts": 2, "identical": 1, "prefix_tokens_mean": 3.0}
  assert c["verdict"] == {"top-1": True, "greedy": False, "greedy vs AWS bf16 (>= -0.05)": True}
  assert "bf16 bar: top-1 PASS, greedy FAIL" in r.stdout

def test_summary_reports_the_weight_dtype(tmp_path):
  """scripts/nki_n6b_summary.py: FuseZap's rows name the weights' dtype (bf16 on NEURON computes in float32, so the model's
  activation dtype is float32): from serve_<v>.json's "dtype" (nki_n6_serve.py), else FZ_NEURON_DTYPE in its switches;
  an HTTP label "bf16_7b" is bfloat16 on one core."""
  import importlib.util
  spec = importlib.util.spec_from_file_location("sm", os.path.join(ROOT, "scripts", "nki_n6b_summary.py"))
  sm = importlib.util.module_from_spec(spec); spec.loader.exec_module(sm)
  assert sm.shape("bf16_7b") == ("bfloat16", "1 core") and sm.shape("fuse4") == ("float32", "1 core") and sm.shape("tp2_bf16") == ("bfloat16", "TP=2")
  json.dump({"dtype": "bfloat16"}, open(tmp_path / "serve_bf16_7b.json", "w"))
  json.dump({"switches": {"FZ_NEURON_DTYPE": "bf16"}}, open(tmp_path / "serve_bf16.json", "w"))
  json.dump({"switches": {}}, open(tmp_path / "serve_f32.json", "w"))
  got = {r["system"]: r["dtype"] for r in sm.rows(str(tmp_path)) if r["source"] == "in process"}
  assert got == {"FuseZap bf16": "bfloat16", "FuseZap bf16_7b": "bfloat16", "FuseZap f32": "float32"}

@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None or shutil.which("python3") is None, reason="needs bash, python3")
def test_n8_7b_inf2_script_runs_every_phase(tmp_path):
  """N8 (I71) job: the HBM probe passes (a result file is in place), so every FuseZap phase runs: kernels (the tiled layer on
  NEURON), the 7B-shaped layer bench, the layer choice (no bench results in a dry run: the members), in process FUSE=1 and 4,
  served, then NxDI TP=2 and TP=1."""
  def pre(tp):
    d = tp / "results" / "nki_n8_7b_inf2"
    d.mkdir(parents=True)
    (d / "hbm_core0.json").write_text(json.dumps({"fits_7b_bf16_1core": True}))
  p, log = _dry_run(tmp_path, "nki_n8_7b_inf2.sh", {"FZ_7B_BUDGET_MIN": "106"}, ("nki_n8_7b_choose.py",), pre)
  assert p.returncode == 0, log[-4000:]
  out = tmp_path / "results" / "nki_n8_7b_inf2"
  tsv = (out / "phases.tsv").read_text()
  phases = [ln.split("\t")[0] for ln in tsv.splitlines()]
  for want in ("env", "setup", "hbm_probe", "kernels", "layer_bench", "fz7b", "fz7b_fuse4", "fzhttp7b", "setup_nxdi", "aws7b_tp2", "aws7b_tp1"):
    assert want in phases, (want, phases)
  assert "FZ_NEURON_LAYER_TILE=0" in tsv   # no layer bench results: the members
  assert "--hidden 3584 --inter 18944 --heads 28 --kv 4 --hd 128 --dtype bf16" in (out / "layer_bench.log").read_text()
  fz = (out / "fz7b.log").read_text()
  assert "FZ_NEURON_DTYPE=bf16" in fz and "NEURON_RT_VISIBLE_CORES=0" in fz and "--top1-prompts 6" in fz
  assert "test_tiled_layer_matches_the_members" in (out / "kernels.log").read_text()
  assert all("NXDI venv not usable" in (out / f"aws7b_tp{t}.log").read_text() for t in (1, 2))   # no NxDI venv in a dry run
  assert "HTTP 500 responses" in (out / "fzhttp7b.log").read_text()
  assert (out / "SUMMARY.txt").exists()

def test_n8_7b_choose(tmp_path):
  """The 7B layer choice: the fastest tiled variant bitwise == v1 in every record, if faster than v1; else the members."""
  import subprocess
  def run(results):
    (tmp_path / "layer_bench.json").write_text(json.dumps({"results": results}))
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "nki_n8_7b_choose.py"), str(tmp_path)], capture_output=True, text=True)
    return p.stdout.strip()
  bit = {"logits_bitwise_vs_v1": 1.0, "pool_bitwise_vs_v1": 1.0, "tokens_vs_v1": True}
  assert run([{"variant": "v1", "device_ms": 3.4}, {"variant": "v2set4", "device_ms": 2.0, **bit}, {"variant": "v2set", "device_ms": 2.2, **bit}]).endswith(
    "FZ_NEURON_LAYER_TILE=1 FZ_NEURON_LAYER_WCOLS=2048")
  assert run([{"variant": "v1", "device_ms": 3.4}, {"variant": "v2set4", "device_ms": 2.0, **bit, "pool_bitwise_vs_v1": 0.9},
              {"variant": "v2set", "device_ms": 2.2, **bit}]).endswith("FZ_NEURON_LAYER_WCOLS=512")
  assert run([{"variant": "v1", "device_ms": 3.4}, {"variant": "v2set4", "skipped": "NKISimError: neuronx-cc failed"}]) == "FZ_NEURON_LAYER_TILE=0"
  assert run([]) == "FZ_NEURON_LAYER_TILE=0"
