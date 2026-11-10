"""scripts/release.py: check / draft / cut in a throwaway git repo (no network, no remote, nothing pushed)."""
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
  spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


RM = _load("release_matrix")
REL = _load("release")

PYPROJECT = """[project]
name = "fusezap"
dynamic = ["version"]

[tool.hatch.version]
path = "fusezap/_version.py"
"""

SOURCES = """schema: fusezap.matrix/v1
updated: 2026-11-02
chips:
  - {id: nvidia-h100, name: NVIDIA H100}
  - {id: apple-metal, name: Apple Silicon (Metal)}
models:
  - {id: qwen2.5-7b, name: Qwen2.5-7B-Instruct}
cells:
  - chip: nvidia-h100
    model: qwen2.5-7b
    status: Supported
    verified: 2026-10-26
    gate: Invarix accuracy + performance
    invarix: PROMOTE
    metrics:
      - {name: TPOT p50 c1, value: 5.57, unit: ms, baseline: "vLLM 0.32.0: 5.85 ms"}
    evidence: [docs/status/h100.md]
  - chip: apple-metal
    model: qwen2.5-7b
    status: Planned
"""

CHANGELOG = """# Changelog

## [Unreleased]

### Added

- Serving on the H100.
"""


def git(repo, *args, check=True):
  r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
  if check: assert r.returncode == 0, r.stderr
  return r.stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
  for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
               "GIT_COMMITTER_EMAIL": "t@example.invalid", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}.items():
    monkeypatch.setenv(k, v)
  r = tmp_path / "repo"
  (r / "fusezap").mkdir(parents=True)
  (r / "docs/releases/matrix").mkdir(parents=True)
  (r / "docs/status").mkdir(parents=True)
  (r / "pyproject.toml").write_text(PYPROJECT)
  (r / "fusezap/_version.py").write_text('__version__ = "0.1.0.dev0"\n')
  (r / "CHANGELOG.md").write_text(CHANGELOG)
  shutil.copy(ROOT / "docs/releases/TEMPLATE.md", r / "docs/releases/TEMPLATE.md")
  shutil.copy(ROOT / "docs/releases/index.yaml", r / "docs/releases/index.yaml")   # the real, empty index
  (r / "docs/releases/matrix/sources.yaml").write_text(SOURCES)
  (r / "docs/status/h100.md").write_text("evidence\n")
  git(r, "init", "-q", "-b", "main")
  git(r, "add", "-A")
  git(r, "commit", "-q", "-m", "init")
  return r


def run(repo, *argv):
  return REL.main(["--repo", str(repo), *argv])


def fill_note(repo, version, breaking="None."):
  """draft the note, then fill every section the way a person would."""
  assert run(repo, "draft", version, "--date", "2026-12-01") == 0
  p = repo / f"docs/releases/v{version}.md"
  text = re.sub(r"<!--.*?-->", "", p.read_text(), flags=re.S)
  text = re.sub(r"(?m)^(## (?!Support matrix|Breaking changes).+)$", r"\1\n\nNone.", text)
  text = text.replace("## Breaking changes\n", f"## Breaking changes\n\n{breaking}\n")
  p.write_text(text)
  git(repo, "add", "-A")
  git(repo, "commit", "-q", "-m", f"note {version}")
  return p


def status(repo): return git(repo, "status", "--porcelain")


# ---------------------------------------------------------------- the real repo


def test_real_repo_version_has_one_source():
  repo = REL.Repo(ROOT)
  srcs = REL.version_sources(repo)
  assert set(srcs) == {"fusezap/_version.py"}          # pyproject is dynamic: no second copy to drift
  ns = {}
  exec((ROOT / "fusezap/_version.py").read_text(), ns)
  assert srcs["fusezap/_version.py"] == ns["__version__"] == "0.1.0.dev0"
  assert "from fusezap._version import __version__" in (ROOT / "fusezap/__init__.py").read_text()


def test_real_repo_template_and_index():
  repo = REL.Repo(ROOT)
  hs = REL.template_headings(repo)
  assert hs == ["Highlights", "New chips and models", "Support matrix", "Fixes", "Performance", "Breaking changes",
                "Deprecations", "Security", "Known issues", "Upgrade notes", "What's next", "Checksums"]
  _, releases = REL.load_index(repo)
  assert releases == []                                # nothing released: no fake entries
  cl = (ROOT / "CHANGELOG.md").read_text()
  assert "## [Unreleased]" in cl and not re.search(r"(?m)^## \[\d", cl)


def test_versions():
  V = REL.Version.parse
  assert str(V("v0.3.0rc1")) == "0.3.0rc1" and V("0.3.0rc1").tag == "v0.3.0rc1" and V("0.3.2").branch == "release/0.3"
  assert V("0.3.0rc1").key < V("0.3.0rc2").key < V("0.3.0").key < V("0.3.1").key < V("0.4.0").key
  for bad in ("0.3", "0.3.0.dev0", "01.2.3", "0.3.0-rc1", "0.3.0rc0"):
    with pytest.raises(REL.ReleaseError): V(bad)


# ---------------------------------------------------------------- check


def test_check_fresh_repo_lists_every_missing_piece(repo, capsys):
  assert run(repo, "check", "v0.1.0") == 1
  err = capsys.readouterr().err
  for want in ("fusezap/_version.py says 0.1.0.dev0, expected 0.1.0", "docs/releases/v0.1.0.md does not exist",
               "no entry for 0.1.0", "no '## [0.1.0] - YYYY-MM-DD' section", "matrix/v0.1.0.yaml does not exist",
               "tag v0.1.0 does not exist"):
    assert want in err, want


def test_static_pyproject_version_must_agree(repo):
  (repo / "pyproject.toml").write_text('[project]\nname = "fusezap"\nversion = "0.0.9"\n')
  errs = REL.check(REL.Repo(repo), "0.1.0", pre_tag=True)
  assert any("pyproject.toml says 0.0.9" in e for e in errs)


# ---------------------------------------------------------------- cut


def test_cut_dry_run_writes_nothing(repo, capsys):
  fill_note(repo, "0.1.0")
  head = git(repo, "rev-parse", "HEAD")
  assert run(repo, "cut", "0.1.0", "--dry-run", "--date", "2026-12-01") == 0
  out = capsys.readouterr().out
  assert "would write fusezap/_version.py" in out and "would annotated tag v0.1.0" in out and "dry run" in out
  assert status(repo) == "" and git(repo, "rev-parse", "HEAD") == head
  assert git(repo, "tag") == "" and git(repo, "branch", "--list", "release/*") == ""


def test_cut_minor_end_to_end(repo, capsys):
  fill_note(repo, "0.1.0")
  assert run(repo, "cut", "0.1.0", "--date", "2026-12-01") == 0
  out = capsys.readouterr().out
  # pushes are printed, never run (the repo has no remote at all)
  assert "Nothing was pushed" in out and "git push origin release/0.1" in out and "git push origin v0.1.0" in out
  assert git(repo, "remote") == ""
  assert git(repo, "cat-file", "-t", "v0.1.0") == "tag"                       # annotated
  assert git(repo, "rev-parse", "release/0.1") == git(repo, "rev-parse", "v0.1.0^{commit}")
  assert git(repo, "log", "-1", "--format=%s", "v0.1.0") == "Release 0.1.0"
  # main moved on to the next dev version
  assert git(repo, "log", "-1", "--format=%s") == "Start 0.2.0.dev0"
  assert '"0.2.0.dev0"' in (repo / "fusezap/_version.py").read_text()
  assert status(repo) == ""
  cl = git(repo, "show", "v0.1.0:CHANGELOG.md")
  assert "## [Unreleased]\n\n## [0.1.0] - 2026-12-01\n\n### Added\n\n- Serving on the H100." in cl
  idx = git(repo, "show", "v0.1.0:docs/releases/index.yaml")
  assert idx.startswith("# FuseZap release index")                            # the header comments survive
  _, rel = REL.load_index(REL.Repo(repo))
  assert rel == [{"version": "0.1.0", "date": "2026-12-01", "line": "0.1", "line_date": "2026-12-01", "support": "current",
                  "invarix_maintenance_needed": "2026-12-01", "products": {"core": "0.1.0", "invarix": "0.1.0"},
                  "notes": "docs/releases/v0.1.0.md", "matrix": "docs/releases/matrix/v0.1.0.yaml"}]
  assert git(repo, "show", "v0.1.0:docs/releases/matrix/v0.1.0.yaml") == SOURCES.strip()
  # the full check passes on the tag, and fails on main (the tag is not HEAD, the version moved on)
  assert run(repo, "check", "v0.1.0") == 1
  git(repo, "checkout", "-q", "v0.1.0")
  capsys.readouterr()
  assert run(repo, "check", "v0.1.0") == 0
  assert "v0.1.0 OK" in capsys.readouterr().out


def test_cut_refuses_dirty_tree(repo, capsys):
  fill_note(repo, "0.1.0")
  (repo / "stray.txt").write_text("x")
  for argv in (("cut", "0.1.0"), ("cut", "0.1.0", "--dry-run")):
    assert run(repo, *argv) == 1
    assert "not clean" in capsys.readouterr().err
  assert git(repo, "tag") == ""


def test_cut_refuses_without_note_or_changes(repo, capsys):
  assert run(repo, "cut", "0.1.0") == 1
  assert "release.py draft 0.1.0" in capsys.readouterr().err
  fill_note(repo, "0.1.0")
  (repo / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n\n")
  git(repo, "commit", "-qam", "empty changelog")
  assert run(repo, "cut", "0.1.0") == 1
  assert "Unreleased section is empty" in capsys.readouterr().err
  assert status(repo) == "" and git(repo, "tag") == ""


def test_cut_refuses_when_check_would_fail(repo, capsys):
  fill_note(repo, "0.1.0")
  (repo / "docs/status/h100.md").unlink()                                     # the Supported cell's evidence
  git(repo, "commit", "-qam", "lose evidence")
  head = git(repo, "rev-parse", "HEAD")
  assert run(repo, "cut", "0.1.0") == 1
  err = capsys.readouterr().err
  assert "docs/status/h100.md does not exist" in err and "nothing was changed" in err
  assert status(repo) == "" and git(repo, "rev-parse", "HEAD") == head and git(repo, "tag") == ""


def test_patch_and_next_minor(repo, capsys):
  fill_note(repo, "0.1.0")
  assert run(repo, "cut", "0.1.0", "--date", "2026-12-01") == 0
  # a patch must be cut on release/0.1
  assert run(repo, "cut", "0.1.1") == 1
  assert "cut 0.1.1 on that branch" in capsys.readouterr().err
  git(repo, "checkout", "-q", "release/0.1")
  (repo / "CHANGELOG.md").write_text((repo / "CHANGELOG.md").read_text().replace(
    "## [Unreleased]\n", "## [Unreleased]\n\n### Fixed\n\n- A hang on shutdown.\n", 1))
  git(repo, "commit", "-qam", "fix")
  fill_note(repo, "0.1.1", breaking="The --foo flag is gone.")
  assert run(repo, "cut", "0.1.1", "--date", "2026-12-09") == 1                 # nothing removed in a patch
  assert "a patch release has no breaking changes" in capsys.readouterr().err
  p = repo / "docs/releases/v0.1.1.md"
  p.write_text(p.read_text().replace("The --foo flag is gone.", "None."))
  git(repo, "commit", "-qam", "no breaking")
  assert run(repo, "cut", "0.1.1", "--date", "2026-12-09") == 0
  out = capsys.readouterr().out
  assert "git push origin release/0.1" in out and "git push origin v0.1.1" in out and "bring the [0.1.1]" in out
  _, rel = REL.load_index(REL.Repo(repo))
  e = rel[-1]
  assert (e["version"], e["date"], e["line_date"], e["invarix_maintenance_needed"]) == ("0.1.1", "2026-12-09", "2026-12-01", "2026-12-01")
  assert e["products"] == {"core": "0.1.1", "invarix": "0.1.0"}                # Invarix did not move
  assert '"0.1.1"' in (repo / "fusezap/_version.py").read_text()              # no dev bump on a release branch
  assert run(repo, "check", "v0.1.1") == 0
  # the next minor from main marks 0.1 as N-1
  git(repo, "checkout", "-q", "main")
  git(repo, "checkout", "-q", "release/0.1", "--", "docs/releases/index.yaml")  # bring the patch entry to main
  (repo / "CHANGELOG.md").write_text((repo / "CHANGELOG.md").read_text().replace(
    "## [Unreleased]\n", "## [Unreleased]\n\n### Added\n\n- Trainium.\n", 1))
  git(repo, "commit", "-qam", "next")
  fill_note(repo, "0.2.0")
  assert run(repo, "cut", "0.2.0", "--date", "2027-01-05") == 0
  _, rel = REL.load_index(REL.Repo(repo))
  assert [(r["version"], r["support"]) for r in rel] == [("0.1.0", "maintained"), ("0.1.1", "maintained"), ("0.2.0", "current")]


def test_rc_starts_the_line(repo, capsys):
  fill_note(repo, "0.1.0rc1")
  assert run(repo, "cut", "0.1.0rc1", "--date", "2026-11-24") == 0
  assert git(repo, "rev-parse", "release/0.1") == git(repo, "rev-parse", "v0.1.0rc1^{commit}")
  assert run(repo, "cut", "0.1.0") == 1                                        # final goes on release/0.1
  assert "on that branch" in capsys.readouterr().err


# ---------------------------------------------------------------- check failures on a finished release


def test_check_catches_broken_releases(repo, capsys):
  fill_note(repo, "0.1.0")
  assert run(repo, "cut", "0.1.0", "--date", "2026-12-01") == 0
  git(repo, "checkout", "-q", "v0.1.0")
  r = REL.Repo(repo)
  note = (repo / "docs/releases/v0.1.0.md").read_text()

  def errs_with(path, text):
    r.overlay = {path: text}
    try: return REL.check(r, "v0.1.0")
    finally: r.overlay = {}

  assert REL.check(r, "v0.1.0") == []
  assert any("missing section '## Checksums'" in e for e in errs_with("docs/releases/v0.1.0.md", note.split("## Checksums")[0]))
  assert any("'Breaking changes' is empty" in e for e in
             errs_with("docs/releases/v0.1.0.md", note.replace("## Breaking changes\n\nNone.", "## Breaking changes\n\n<!-- todo -->")))
  assert any("unfilled placeholders {version}" in e for e in errs_with("docs/releases/v0.1.0.md", note + "\n{version}\n"))
  cl = (repo / "CHANGELOG.md").read_text()
  assert any("no '## [0.1.0]" in e for e in errs_with("CHANGELOG.md", cl.replace("## [0.1.0] - 2026-12-01", "## 0.1.0")))
  assert any("dated 2026-12-02" in e for e in errs_with("CHANGELOG.md", cl.replace("2026-12-01", "2026-12-02")))
  idx = (repo / "docs/releases/index.yaml").read_text()
  assert any("no entry for 0.1.0" in e for e in errs_with("docs/releases/index.yaml", idx.replace("version: 0.1.0", "version: 0.0.9")))
  assert any("invarix_maintenance_needed must equal line_date" in e for e in
             errs_with("docs/releases/index.yaml", idx.replace("invarix_maintenance_needed: '2026-12-01'", "invarix_maintenance_needed: '2027-01-01'")
                       .replace("invarix_maintenance_needed: 2026-12-01", "invarix_maintenance_needed: 2027-01-01")))
  mx = (repo / "docs/releases/matrix/v0.1.0.yaml").read_text()
  assert any("does not parse" in e for e in errs_with("docs/releases/matrix/v0.1.0.yaml", mx + "\n  - [\n"))
  assert any("needs `gate`" in e for e in errs_with("docs/releases/matrix/v0.1.0.yaml", mx.replace("    gate: Invarix accuracy + performance\n", "")))
  # tag problems: a lightweight tag, a tag that is not HEAD
  git(repo, "checkout", "-q", "main")
  assert any("does not point at HEAD" in e for e in REL.check(r, "v0.1.0"))
  git(repo, "tag", "-d", "v0.1.0")
  git(repo, "tag", "v0.1.0", "release/0.1")
  git(repo, "checkout", "-q", "v0.1.0")
  assert any("lightweight" in e for e in REL.check(r, "v0.1.0"))
