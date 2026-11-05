"""scripts/release_matrix.py: loader, status/evidence rules, markdown / JSON / HTML output (no network)."""
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("release_matrix", ROOT / "scripts" / "release_matrix.py")
RM = importlib.util.module_from_spec(spec)
sys.modules["release_matrix"] = RM
spec.loader.exec_module(RM)

BASE = {
  "schema": "fusezap.matrix/v1", "updated": "2026-11-02",
  "chips": [{"id": "nvidia-h100", "name": "NVIDIA H100"}, {"id": "aws-inf2", "name": "AWS Inferentia2"}],
  "models": [{"id": "qwen2.5-7b", "name": "Qwen2.5-7B-Instruct"}, {"id": "llama-3.3-70b", "name": "Llama-3.3-70B-Instruct"}],
  "cells": [],
}


def cell(**kw):
  c = {"chip": "nvidia-h100", "model": "qwen2.5-7b", "status": "Supported", "verified": "2026-10-26",
       "gate": "Validate accuracy + performance", "validate": "PROMOTE", "evidence": ["ev.md"]}
  c.update(kw)
  return {k: v for k, v in c.items() if v is not None}


def mx(*cells, **top):
  d = {**BASE, **top, "cells": list(cells)}
  return RM.parse(yaml.safe_load(yaml.safe_dump(d)))


@pytest.fixture
def repo(tmp_path, monkeypatch):
  for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
               "GIT_COMMITTER_EMAIL": "t@example.invalid", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}.items():
    monkeypatch.setenv(k, v)
  (tmp_path / "ev.md").write_text("x")
  (tmp_path / "docs/status/run1").mkdir(parents=True)
  (tmp_path / "docs/status/run1/README.md").write_text("results")
  (tmp_path / "docs/keep.md").write_text("stays")
  for a in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "init"]):
    subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
  sha = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
  # results that only live in a commit (as on a results branch): delete them from the working tree
  subprocess.run(["git", "-C", str(tmp_path), "rm", "-rq", "docs/status/run1"], check=True)
  subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "results moved to a branch"], check=True)
  return tmp_path, sha


# ---------------------------------------------------------------- the real source file


def test_real_sources_validate_and_render():
  m = RM.load(RM.DEFAULT_SOURCE, ROOT)
  pinned = {e.split(":")[0] for c in m.cells for e in c.evidence if re.match(r"^[0-9a-f]{7,40}(:|$)", e)}
  missing = [s for s in pinned if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{s}^{{commit}}"]).returncode]
  if missing: pytest.skip(f"results commits not in this clone: {missing}")
  assert RM.validate(m, ROOT) == []
  # the cells the task names are present with the measured numbers
  inf2 = m.cell("aws-inf2", "qwen2.5-7b")
  assert inf2.status == "Preview" and inf2.validate == "PROMOTE" and inf2.verified == "2026-11-01"
  assert {x.name: x.value for x in inf2.metrics} == {"TTFT p50 c1 (short)": 55.4, "TPOT p50 c1": 38.8, "output tok/s c4 (short)": 99.3}
  assert m.cell("nvidia-h100", "qwen2.5-7b").status == "Preview"    # every cell is Preview until the first public release
  assert m.cell("nvidia-h200", "qwen2.5-72b").status == "Preview"
  assert all(c.status != "Supported" for c in m.cells)
  assert m.cell("nvidia-h100", "flux.2-klein-4b").status == "Preview"
  assert m.cell("nvidia-h200", "llama-3.3-70b").status == "Preview"   # accuracy + performance only; full suite not clean
  assert m.cell("apple-metal", "qwen2.5-0.5b") and m.cell("cpu", "qwen2.5-0.5b").status == "Preview"
  for fmt in RM.RENDER: assert RM.RENDER[fmt](m)


# ---------------------------------------------------------------- structure


@pytest.mark.parametrize("bad, msg", [
  ({"schema": "x"}, "schema must be"),
  ({"updated": "2 Nov"}, "not an ISO date"),
  ({"chips": [{"id": "H100", "name": "x"}]}, "lowercase id"),
  ({"chips": [{"id": "a", "name": "x"}, {"id": "a", "name": "y"}]}, "duplicate chips id"),
])
def test_parse_rejects_bad_top_level(bad, msg):
  with pytest.raises(RM.MatrixError, match=msg): mx(**bad)


@pytest.mark.parametrize("kw, msg", [
  ({"status": "GA"}, "status must be one of"),
  ({"chip": "tpu"}, "chip 'tpu' is not in chips"),
  ({"model": "gpt"}, "model 'gpt' is not in models"),
  ({"validate": "PASS"}, "validate must be one of"),
  ({"verified": "yesterday"}, "not an ISO date"),
  ({"colour": "green"}, "unknown keys"),
  ({"metrics": [{"name": "TPOT"}]}, "needs name and value"),
])
def test_parse_rejects_bad_cells(kw, msg):
  with pytest.raises(RM.MatrixError, match=msg): mx(cell(**kw))


def test_parse_rejects_duplicate_cell():
  with pytest.raises(RM.MatrixError, match="duplicate cell"): mx(cell(), cell(status="Preview"))


def test_yaml_error_is_a_matrix_error(tmp_path):
  with pytest.raises(RM.MatrixError, match="does not parse"): RM.load("m.yaml", tmp_path, text="cells: [\n")


# ---------------------------------------------------------------- status and evidence rules


def test_evidence_forms(repo):
  root, sha = repo
  assert RM.resolve_evidence("ev.md", root) is None
  assert RM.resolve_evidence("docs", root) is None                              # a directory is fine
  assert RM.resolve_evidence(sha, root) is None and RM.resolve_evidence(sha[:7], root) is None
  assert RM.resolve_evidence(f"{sha}:docs/status/run1/README.md", root) is None  # only in the commit
  assert "does not exist in the working tree" in RM.resolve_evidence("docs/status/run1/README.md", root)
  assert "does not exist in commit" in RM.resolve_evidence(f"{sha}:nope.md", root)
  assert "not in the local repo" in RM.resolve_evidence("0" * 40, root)
  assert "not in the local repo" in RM.resolve_evidence("deadbeef:ev.md", root)
  assert "no URLs" in RM.resolve_evidence("https://example.com/x", root)
  assert "inside the repo" in RM.resolve_evidence("../etc/passwd", root)
  assert "does not exist" in RM.resolve_evidence("main:ev.md", root)          # branch names are not accepted
  assert RM.resolve_evidence(RM.PRIVATE_EVIDENCE, root) is None               # unpublished evidence, said so
  assert RM.resolve_evidence("private: maybe", root) is not None              # only the one exact form


def test_status_rules(repo):
  root, sha = repo
  ok = mx(cell(), cell(chip="aws-inf2", status="Preview", gate=None, validate=None, evidence=[f"{sha}:docs/status/run1/README.md"]),
          cell(model="llama-3.3-70b", status="Planned", verified=None, gate=None, validate=None, evidence=None))
  assert RM.validate(ok, root) == []
  cases = [
    (cell(gate=None), "Supported needs `gate`"),
    (cell(evidence=None), "needs at least one evidence entry"),
    (cell(verified=None), "needs a verified date"),
    (cell(validate="BLOCK"), "use Preview"),
    (cell(evidence=["missing.md"]), "missing.md does not exist"),
    (cell(status="Preview", gate=None, evidence=["https://x"]), "no URLs"),
    (cell(status="Planned", verified=None, evidence=None, gate=None), "no Validate verdict or metrics"),
  ]
  for c, msg in cases:
    errs = RM.validate(mx(c), root)
    assert any(msg in e for e in errs), (msg, errs)


# ---------------------------------------------------------------- rendering


def sample():
  return mx(cell(config="bf16, 1 GPU", build="323463f", tested_with={"vllm": "0.32.0"},
                 metrics=[{"name": "TPOT p50 c1", "value": 5.57, "unit": "ms", "baseline": "vLLM 0.32.0: 5.85 ms"}],
                 evidence=["ev.md", "6e8ce58535b5dd49ba56a015aee452550cda51c0:docs/status/x/README.md"], notes="One pod | 3 repeats."),
            cell(chip="aws-inf2", status="Preview", gate=None, validate=None, notes="<b>not</b> gated"),
            cell(chip="aws-inf2", model="llama-3.3-70b", status="Planned", verified=None, gate=None, validate=None, evidence=None))


def test_markdown():
  md = RM.to_markdown(sample())
  assert "| Model | NVIDIA H100 | AWS Inferentia2 |" in md
  assert "| Qwen2.5-7B-Instruct | **Supported** | Preview |" in md
  assert "| Llama-3.3-70B-Instruct | — | Planned |" in md
  assert "TPOT p50 c1 5.57 ms (vLLM 0.32.0: 5.85 ms)" in md
  assert "`6e8ce58535b5:docs/status/x/README.md`" in md                       # shortened for display
  assert "One pod \\| 3 repeats." in md                                       # pipes escaped
  lines = [l for l in md.splitlines() if l.startswith("| NVIDIA") or l.startswith("| AWS")]
  assert lines[0].startswith("| NVIDIA H100 |")                               # Supported rows first


def test_json():
  j = json.loads(RM.RENDER["json"](sample()))
  assert j["schema"] == "fusezap.matrix/v1" and j["statuses"] == ["Supported", "Preview", "Experimental", "Planned"]
  assert [c["status"] for c in j["cells"]] == ["Supported", "Preview", "Planned"]
  c = j["cells"][0]
  assert c["evidence"][1] == "6e8ce58535b5dd49ba56a015aee452550cda51c0:docs/status/x/README.md"  # full ref kept
  assert c["metrics"] == [{"name": "TPOT p50 c1", "value": 5.57, "unit": "ms", "target": None, "baseline": "vLLM 0.32.0: 5.85 ms"}]
  assert j["cells"][2]["verified"] is None and j["cells"][2]["evidence"] == []


def test_html_is_self_contained_and_escaped():
  h = RM.to_html(sample())
  assert h.startswith('<section class="fz-matrix"') and h.rstrip().endswith("</section>")
  assert "<script" not in h and not re.search(r"""(src|href)\s*=|@import|url\(""", h)
  assert "&lt;b&gt;not&lt;/b&gt; gated" in h and "<b>not</b>" not in h
  assert "prefers-color-scheme:dark" in h and "overflow-x:auto" in h
  assert 'class="fzm-s fzm-Supported"' in h


def test_cli(repo, tmp_path, capsys):
  root, _ = repo
  src = root / "m.yaml"
  src.write_text(yaml.safe_dump({**BASE, "cells": [cell()]}))
  assert RM.main(["m.yaml", "--repo", str(root), "--check"]) == 0
  assert "1 cells (1 Supported" in capsys.readouterr().out
  out = tmp_path / "m.json"
  assert RM.main(["m.yaml", "--repo", str(root), "--format", "json", "--out", str(out)]) == 0
  assert json.loads(out.read_text())["cells"][0]["chip"] == "nvidia-h100"
  src.write_text(yaml.safe_dump({**BASE, "cells": [cell(evidence=["gone.md"])]}))
  assert RM.main(["m.yaml", "--repo", str(root), "--check"]) == 1
  assert "gone.md does not exist" in capsys.readouterr().err
  assert RM.main(["m.yaml", "--repo", str(root), "--no-evidence-check"]) == 0
