#!/usr/bin/env python3
"""FuseZap release tooling: check a release, cut one locally, draft its note. It never pushes.

  python scripts/release.py check v0.2.0            # everything a tag needs (CI runs this on the tag)
  python scripts/release.py check v0.2.0 --pre-tag  # the same without the tag checks (before cutting, in PRs)
  python scripts/release.py draft 0.2.0             # docs/releases/v0.2.0.md from TEMPLATE.md + the live matrix
  python scripts/release.py cut 0.2.0 --dry-run     # show what cut would change, write nothing
  python scripts/release.py cut 0.2.0               # bump, move Unreleased, freeze matrix, index entry, commit,
                                                    # release/0.2 branch (first cut on a line), annotated tag;
                                                    # prints the push commands instead of running them

`check vX.Y.Z` fails unless:
  - pyproject.toml, fusezap/_version.py (fusezap.__version__) and the tag agree on X.Y.Z (pyproject reads the version
    from _version.py through [tool.hatch.version]; a static `version` in pyproject must equal it),
  - docs/releases/vX.Y.Z.md exists, has every `## ` heading of docs/releases/TEMPLATE.md, no unfilled {placeholder},
    and "Breaking changes" is non-empty ("None." when nothing breaks; a patch release must say None),
  - CHANGELOG.md has a non-empty `## [X.Y.Z] - YYYY-MM-DD` section whose date is the index entry's date,
  - docs/releases/index.yaml has a well-formed entry for X.Y.Z,
  - docs/releases/matrix/vX.Y.Z.yaml parses and every cell follows the status rules, every Supported cell with
    evidence that resolves to a repo file or a commit (scripts/release_matrix.py, the same loader),
  - (without --pre-tag) the annotated tag vX.Y.Z exists, points at HEAD, and its _version.py says X.Y.Z.

Policy: docs/releases/README.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _load_matrix_module():
  spec = importlib.util.spec_from_file_location("release_matrix", HERE / "release_matrix.py")
  mod = importlib.util.module_from_spec(spec)
  sys.modules.setdefault("release_matrix", mod)
  spec.loader.exec_module(mod)
  return mod


RM = _load_matrix_module()

VERSION_FILE = "fusezap/_version.py"
CHANGELOG = "CHANGELOG.md"
INDEX = "docs/releases/index.yaml"
TEMPLATE = "docs/releases/TEMPLATE.md"
SOURCES = "docs/releases/matrix/sources.yaml"
INDEX_SCHEMA = "fusezap.releases/v1"
SUPPORT = ("current", "maintained", "security", "eol")
VALIDATE_SECURITY_MONTHS = 6
INDEX_FIELDS = ("version", "date", "line", "line_date", "support", "validate_maintenance_needed", "products", "notes", "matrix")

_REL = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:rc([1-9]\d*))?$")
_VERSION_LINE = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']\s*$', re.M)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_PLACEHOLDER = re.compile(r"\{(version|date|line|line_date|validate_version)\}")


class ReleaseError(Exception):
  pass


# ---------------------------------------------------------------- versions


@dataclass(frozen=True)
class Version:
  major: int
  minor: int
  patch: int
  rc: int | None = None

  @classmethod
  def parse(cls, s: str) -> "Version":
    s = s[1:] if s.startswith("v") else s
    m = _REL.match(s)
    if not m: raise ReleaseError(f"not a release version (X.Y.Z or X.Y.ZrcN): {s!r}")
    return cls(int(m[1]), int(m[2]), int(m[3]), int(m[4]) if m[4] else None)

  def __str__(self): return f"{self.major}.{self.minor}.{self.patch}" + (f"rc{self.rc}" if self.rc else "")
  @property
  def tag(self) -> str: return f"v{self}"
  @property
  def line(self) -> str: return f"{self.major}.{self.minor}"
  @property
  def branch(self) -> str: return f"release/{self.line}"
  @property
  def key(self): return (self.major, self.minor, self.patch, 0 if self.rc else 1, self.rc or 0)
  @property
  def note(self) -> str: return f"docs/releases/{self.tag}.md"
  @property
  def matrix(self) -> str: return f"docs/releases/matrix/{self.tag}.yaml"


# ---------------------------------------------------------------- repo access (working tree + an overlay for dry runs)


class Repo:
  def __init__(self, root: str | Path = REPO):
    self.root = Path(root).resolve()
    self.overlay: dict[str, str] = {}

  def read(self, rel: str) -> str | None:
    if rel in self.overlay: return self.overlay[rel]
    p = self.root / rel
    return p.read_text() if p.is_file() else None

  def exists(self, rel: str) -> bool: return rel in self.overlay or (self.root / rel).exists()

  def git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, text=True)
    if check and r.returncode: raise ReleaseError(f"git {' '.join(args)} failed: {r.stderr.strip() or r.stdout.strip()}")
    return r

  def git_ok(self, *args: str) -> bool: return self.git(*args, check=False).returncode == 0
  def out(self, *args: str) -> str: return self.git(*args).stdout.strip()


def version_sources(repo: Repo) -> dict[str, str]:
  """{file: version} for every place the Core version is written."""
  pp = repo.read("pyproject.toml")
  if pp is None: raise ReleaseError("pyproject.toml not found")
  proj = tomllib.loads(pp)
  project = proj.get("project", {})
  path = proj.get("tool", {}).get("hatch", {}).get("version", {}).get("path", VERSION_FILE)
  out = {}
  if "version" in project: out["pyproject.toml"] = str(project["version"])
  elif "version" not in project.get("dynamic", []):
    raise ReleaseError("pyproject.toml has neither a static version nor dynamic = [\"version\"]")
  elif path != VERSION_FILE:
    raise ReleaseError(f"pyproject.toml [tool.hatch.version] path is {path!r}, expected {VERSION_FILE!r}")
  vf = repo.read(VERSION_FILE)
  m = _VERSION_LINE.search(vf or "")
  if not m: raise ReleaseError(f"{VERSION_FILE}: no __version__ = \"...\" line")
  out[VERSION_FILE] = m.group(1)
  return out


def set_version(repo: Repo, version: str) -> dict[str, str]:
  """New contents of the version files (not written)."""
  out = {VERSION_FILE: _VERSION_LINE.sub(f'__version__ = "{version}"', repo.read(VERSION_FILE), count=1)}
  pp = repo.read("pyproject.toml")
  if "version" in tomllib.loads(pp).get("project", {}):
    out["pyproject.toml"] = re.sub(r'(?m)^version\s*=\s*"[^"]*"', f'version = "{version}"', pp, count=1)
  return out


# ---------------------------------------------------------------- notes, changelog, index


def template_headings(repo: Repo) -> list[str]:
  t = repo.read(TEMPLATE)
  if t is None: raise ReleaseError(f"{TEMPLATE} not found")
  hs = re.findall(r"(?m)^## (.+?)\s*$", _COMMENT.sub("", t))
  if not hs: raise ReleaseError(f"{TEMPLATE} has no '## ' headings")
  return hs


def note_sections(text: str) -> dict[str, str]:
  """{heading: body with HTML comments removed and stripped}."""
  text = _COMMENT.sub("", text)
  parts = re.split(r"(?m)^## (.+?)\s*$", text)
  return {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def changelog_section(text: str, version: str) -> tuple[str, str] | None:
  """(date, body) of `## [version] - date`, or None."""
  m = re.search(rf"(?m)^## \[{re.escape(version)}\] - (\S+)\s*$", text)
  if not m: return None
  rest = text[m.end():]
  nxt = re.search(r"(?m)^## \[", rest)
  return m.group(1), (rest[:nxt.start()] if nxt else rest).strip()


def move_unreleased(text: str, version: str, date: str) -> str:
  m = re.search(r"(?m)^## \[Unreleased\]\s*$", text)
  if not m: raise ReleaseError(f"{CHANGELOG} has no '## [Unreleased]' section")
  rest = text[m.end():]
  nxt = re.search(r"(?m)^## \[", rest)
  body = (rest[:nxt.start()] if nxt else rest).strip()
  if not _COMMENT.sub("", body).strip(): raise ReleaseError(f"{CHANGELOG}: the Unreleased section is empty; nothing to release")
  tail = rest[nxt.start():] if nxt else ""
  return text[:m.start()] + f"## [Unreleased]\n\n## [{version}] - {date}\n\n{body}\n" + ("\n" + tail if tail else "")


def load_index(repo: Repo) -> tuple[str, list[dict]]:
  """(header comments, releases)."""
  text = repo.read(INDEX)
  if text is None: raise ReleaseError(f"{INDEX} not found")
  try: data = yaml.safe_load(text)
  except yaml.YAMLError as e: raise ReleaseError(f"{INDEX}: YAML does not parse: {e}") from e
  if not isinstance(data, dict) or data.get("schema") != INDEX_SCHEMA:
    raise ReleaseError(f"{INDEX}: schema must be {INDEX_SCHEMA!r}")
  rel = data.get("releases") or []
  if not isinstance(rel, list): raise ReleaseError(f"{INDEX}: releases must be a list")
  header = text[:text.index("schema:")] if "schema:" in text else ""
  return header, [{k: (v.isoformat() if isinstance(v, dt.date) else v) for k, v in r.items()} for r in rel]


def dump_index(header: str, releases: list[dict]) -> str:
  body = yaml.safe_dump({"schema": INDEX_SCHEMA, "releases": releases}, sort_keys=False, default_flow_style=None, width=120)
  return header + body


def _add_months(d: dt.date, months: int) -> dt.date:
  y, m = divmod(d.month - 1 + months, 12)
  return d.replace(year=d.year + y, month=m + 1, day=min(d.day, 28))


def index_entry_errors(e: dict, v: Version, releases: list[dict]) -> list[str]:
  w = f"{INDEX}: {v}"
  errs = [f"{w}: missing field {f!r}" for f in INDEX_FIELDS if f not in e]
  if errs: return errs
  for f in ("date", "line_date", "validate_maintenance_needed"):
    try: dt.date.fromisoformat(str(e[f]))
    except ValueError: errs.append(f"{w}: {f} is not an ISO date: {e[f]!r}")
  if str(e["line"]) != v.line: errs.append(f"{w}: line is {e['line']!r}, expected {v.line!r}")
  if e["support"] not in SUPPORT: errs.append(f"{w}: support must be one of {SUPPORT}")
  if str(e["validate_maintenance_needed"]) != str(e["line_date"]):
    errs.append(f"{w}: validate_maintenance_needed must equal line_date (licences compare against the line date)")
  if v.patch == 0 and not v.rc and str(e["line_date"]) != str(e["date"]):
    errs.append(f"{w}: an X.Y.0 release's line_date is its own date")
  if v.patch > 0:
    base = next((r for r in releases if str(r.get("version")) == f"{v.line}.0"), None)
    if base is None: errs.append(f"{w}: patch release without a {v.line}.0 entry")
    elif str(base.get("line_date")) != str(e["line_date"]): errs.append(f"{w}: line_date must be {v.line}.0's ({base.get('line_date')})")
  if not isinstance(e["products"], dict) or set(e["products"]) != {"core", "validate"}:
    errs.append(f"{w}: products must be {{core: ..., validate: ...}}")
  elif str(e["products"]["core"]) != str(v): errs.append(f"{w}: products.core is {e['products']['core']}, expected {v}")
  if e["notes"] != v.note: errs.append(f"{w}: notes must be {v.note}")
  if e["matrix"] != v.matrix: errs.append(f"{w}: matrix must be {v.matrix}")
  return errs


# ---------------------------------------------------------------- check


def check(repo: Repo, version: str, pre_tag: bool = False) -> list[str]:
  v = Version.parse(version)
  errs: list[str] = []

  # versions
  try:
    srcs = version_sources(repo)
    for f, x in srcs.items():
      if x != str(v): errs.append(f"{f} says {x}, expected {v}")
  except ReleaseError as e: errs.append(str(e))

  # release note
  note = repo.read(v.note)
  if note is None: errs.append(f"{v.note} does not exist (scripts/release.py draft {v})")
  else:
    try:
      secs = note_sections(note)
      for h in template_headings(repo):
        if h not in secs: errs.append(f"{v.note}: missing section '## {h}'")
      bc = secs.get("Breaking changes")
      if bc is not None and not bc: errs.append(f"{v.note}: 'Breaking changes' is empty (write 'None.' if nothing breaks)")
      if bc and v.patch > 0 and not re.match(r"(?i)none\b", bc):
        errs.append(f"{v.note}: a patch release has no breaking changes; 'Breaking changes' must say None")
      ph = sorted(set(_PLACEHOLDER.findall(_COMMENT.sub("", note))))
      if ph: errs.append(f"{v.note}: unfilled placeholders {', '.join('{' + p + '}' for p in ph)}")
    except ReleaseError as e: errs.append(str(e))

  # index
  entry = None
  try:
    _, releases = load_index(repo)
    found = [r for r in releases if str(r.get("version")) == str(v)]
    if not found: errs.append(f"{INDEX}: no entry for {v}")
    elif len(found) > 1: errs.append(f"{INDEX}: {len(found)} entries for {v}")
    else:
      entry = found[0]
      errs += index_entry_errors(entry, v, releases)
  except ReleaseError as e: errs.append(str(e))

  # changelog
  cl = repo.read(CHANGELOG)
  if cl is None: errs.append(f"{CHANGELOG} not found")
  else:
    sec = changelog_section(cl, str(v))
    if sec is None: errs.append(f"{CHANGELOG}: no '## [{v}] - YYYY-MM-DD' section")
    else:
      date, body = sec
      try: dt.date.fromisoformat(date)
      except ValueError: errs.append(f"{CHANGELOG}: [{v}] date {date!r} is not YYYY-MM-DD")
      if not body: errs.append(f"{CHANGELOG}: [{v}] section is empty")
      if entry and "date" in entry and str(entry["date"]) != date:
        errs.append(f"{CHANGELOG}: [{v}] is dated {date}, {INDEX} says {entry['date']}")

  # matrix
  mtext = repo.read(v.matrix)
  if mtext is None: errs.append(f"{v.matrix} does not exist")
  else:
    try:
      mx = RM.load(v.matrix, repo.root, text=mtext)
      errs += RM.validate(mx, repo.root)
    except RM.MatrixError as e: errs.append(str(e))

  # tag
  if not pre_tag:
    ref = f"refs/tags/{v.tag}"
    if not repo.git_ok("rev-parse", "-q", "--verify", ref): errs.append(f"tag {v.tag} does not exist")
    else:
      if repo.out("cat-file", "-t", ref) != "tag": errs.append(f"tag {v.tag} is lightweight; release tags are annotated")
      if repo.out("rev-parse", f"{ref}^{{commit}}") != repo.out("rev-parse", "HEAD"):
        errs.append(f"tag {v.tag} does not point at HEAD (check out the tag to check it)")
      r = repo.git("show", f"{v.tag}:{VERSION_FILE}", check=False)
      m = _VERSION_LINE.search(r.stdout or "")
      if not m or m.group(1) != str(v): errs.append(f"tag {v.tag}: {VERSION_FILE} says {m.group(1) if m else '?'}, expected {v}")
  return errs


# ---------------------------------------------------------------- draft


def draft_text(repo: Repo, v: Version, date: str, line_date: str) -> str:
  t = repo.read(TEMPLATE)
  matrix_md = RM.to_markdown(RM.load(SOURCES, repo.root, text=repo.read(SOURCES)))
  t = (t.replace("{version}", str(v)).replace("{validate_version}", str(v)).replace("{line_date}", line_date)
       .replace("{date}", date).replace("{line}", v.line))
  return re.sub(r"(?m)^(## Support matrix\s*\n)", lambda m: m.group(1) + "\n" + matrix_md, t, count=1)


# ---------------------------------------------------------------- cut


def _today() -> str: return dt.date.today().isoformat()


def plan_cut(repo: Repo, v: Version, date: str, validate_version: str | None = None) -> tuple[dict[str, str], dict]:
  """(new file contents, facts) for cutting v. Raises ReleaseError on any refusal."""
  if repo.out("status", "--porcelain"): raise ReleaseError("the working tree is not clean; commit or remove the changes first")
  if repo.git_ok("rev-parse", "-q", "--verify", f"refs/tags/{v.tag}"): raise ReleaseError(f"tag {v.tag} already exists")
  branch = repo.out("rev-parse", "--abbrev-ref", "HEAD")
  if branch == "HEAD": raise ReleaseError("HEAD is detached; check out a branch")
  line_exists = repo.git_ok("rev-parse", "-q", "--verify", f"refs/heads/{v.branch}")
  if line_exists and branch != v.branch:
    raise ReleaseError(f"{v.branch} exists: cut {v} on that branch (currently on {branch})")
  if not line_exists:
    if v.patch > 0: raise ReleaseError(f"{v} is a patch but {v.branch} does not exist; patches are cut on the release branch")
    if branch.startswith("release/"): raise ReleaseError(f"the first cut of line {v.line} starts from main, not {branch}")
  header, releases = load_index(repo)
  for r in releases:
    try: rv = Version.parse(str(r.get("version")))
    except ReleaseError: continue
    if rv == v: raise ReleaseError(f"{INDEX} already has {v}")
    if rv.line == v.line and rv.key > v.key: raise ReleaseError(f"{v} is older than {rv} on the same line")
  if not repo.exists(v.note): raise ReleaseError(f"{v.note} does not exist: run `scripts/release.py draft {v}` and fill it in first")
  if repo.read(SOURCES) is None and not repo.exists(v.matrix): raise ReleaseError(f"{SOURCES} not found")

  files = set_version(repo, str(v))
  files[CHANGELOG] = move_unreleased(repo.read(CHANGELOG) or "", str(v), date)
  if not repo.exists(v.matrix): files[v.matrix] = repo.read(SOURCES)

  if v.patch > 0:
    base = next((r for r in releases if str(r.get("version")) == f"{v.line}.0"), None)
    if base is None: raise ReleaseError(f"{INDEX} has no {v.line}.0 entry for patch {v}")
    line_date = str(base["line_date"])
  else:
    line_date = date
  if validate_version is None:   # a minor ships both products at X.Y.0; a Core patch keeps the line's Validate version
    prev = [r for r in releases if str(r.get("line")) == v.line]
    validate_version = str(prev[-1]["products"]["validate"]) if v.patch > 0 and prev else str(v)
  validate_v = str(Version.parse(validate_version))
  if Version.parse(validate_v).line != v.line: raise ReleaseError(f"Validate {validate_v} is not on line {v.line}")
  entry = {"version": str(v), "date": date, "line": v.line, "line_date": line_date, "support": "current",
           "validate_maintenance_needed": line_date, "products": {"core": str(v), "validate": validate_v},
           "notes": v.note, "matrix": v.matrix}
  if v.patch == 0 and not v.rc:   # a new minor: the previous line becomes N-1, older lines security-only or EOL
    lines = sorted({str(r["line"]) for r in releases if str(r.get("line")) != v.line}, key=lambda s: tuple(map(int, s.split("."))))
    today = dt.date.fromisoformat(date)
    for r in releases:
      ln = str(r.get("line"))
      if ln == v.line: continue
      if lines and ln == lines[-1]: r["support"] = "maintained"
      else:
        until = _add_months(dt.date.fromisoformat(str(r["line_date"])), VALIDATE_SECURITY_MONTHS)
        r["support"] = "security" if today <= until else "eol"
  releases.append(entry)
  files[INDEX] = dump_index(header, releases)
  facts = {"branch": branch, "new_line": not line_exists, "line_date": line_date,
           "dev_bump": f"{v.major}.{v.minor + 1}.0.dev0" if not line_exists else None}
  return files, facts


def cut(repo: Repo, version: str, dry_run: bool = False, date: str | None = None, remote: str = "origin",
        validate_version: str | None = None) -> int:
  v = Version.parse(version)
  date = date or _today()
  files, facts = plan_cut(repo, v, date, validate_version)
  repo.overlay.update(files)
  errs = check(repo, str(v), pre_tag=True)
  if errs:
    repo.overlay.clear()
    for e in errs: print(f"release: {e}", file=sys.stderr)
    raise ReleaseError(f"{v} would not pass `check`; nothing was changed")
  print(f"release: cut {v} on {facts['branch']} (line {v.line}, line date {facts['line_date']}, build date {date})")
  for f in sorted(files): print(f"  {'would write' if dry_run else 'write'} {f}")
  steps = [f"commit 'Release {v}'", f"annotated tag {v.tag}"]
  if facts["new_line"]: steps.insert(1, f"branch {v.branch} at the release commit")
  if facts["dev_bump"]: steps.append(f"on {facts['branch']}: commit 'Start {facts['dev_bump']}' (version {facts['dev_bump']})")
  for s in steps: print(f"  {'would ' if dry_run else ''}{s}")
  if dry_run:
    repo.overlay.clear()
    print("release: dry run; nothing written")
    return 0
  repo.overlay.clear()
  for f, text in files.items():
    p = repo.root / f
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
  repo.git("add", "--", *files)
  repo.git("commit", "-q", "-m", f"Release {v}")
  if facts["new_line"]: repo.git("branch", v.branch)
  repo.git("tag", "-a", v.tag, "-m", f"FuseZap {v}")
  errs = check(repo, str(v))
  if errs:
    for e in errs: print(f"release: {e}", file=sys.stderr)
    raise ReleaseError(f"{v.tag} was made but fails check; undo with: git tag -d {v.tag} && git reset --hard HEAD~1"
                       + (f" && git branch -D {v.branch}" if facts["new_line"] else ""))
  if facts["dev_bump"]:
    for f, text in set_version(repo, facts["dev_bump"]).items(): (repo.root / f).write_text(text)
    repo.git("commit", "-q", "-am", f"Start {facts['dev_bump']}")
  push = [f"git push {remote} {v.branch}", f"git push {remote} {v.tag}"]
  if facts["branch"] != v.branch: push.insert(0, f"git push {remote} {facts['branch']}")
  print(f"release: {v.tag} made locally and passes check. Nothing was pushed. After approval, run:")
  for p in push: print(f"  {p}")
  if v.patch > 0: print(f"release: then bring the [{v}] CHANGELOG section and the index entry to main (cherry-pick or a PR)")
  return 0


# ---------------------------------------------------------------- CLI


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description="FuseZap release tooling (never pushes)")
  ap.add_argument("--repo", default=str(REPO), help="repository root (default: this checkout)")
  sub = ap.add_subparsers(dest="cmd", required=True)
  c = sub.add_parser("check", help="check that a release vX.Y.Z is complete")
  c.add_argument("version")
  c.add_argument("--pre-tag", action="store_true", help="skip the tag checks (before cut, in PRs)")
  d = sub.add_parser("draft", help="write docs/releases/vX.Y.Z.md from the template and the live matrix")
  d.add_argument("version")
  d.add_argument("--date", help="release date (default today)")
  k = sub.add_parser("cut", help="cut a release locally: bump, changelog, matrix, index, commit, branch, tag")
  k.add_argument("version")
  k.add_argument("--dry-run", action="store_true", help="print what would change; write nothing")
  k.add_argument("--date", help="release date (default today)")
  k.add_argument("--remote", default="origin", help="remote named in the printed push commands")
  k.add_argument("--validate-version", help="Validate version on this release (default: X.Y.Z for a minor, "
                 "the line's current Validate version for a Core patch)")
  a = ap.parse_args(argv)
  repo = Repo(a.repo)
  try:
    if a.cmd == "check":
      errs = check(repo, a.version, pre_tag=a.pre_tag)
      for e in errs: print(f"release: {e}", file=sys.stderr)
      v = Version.parse(a.version)
      print(f"release: {v.tag} {'FAILS' if errs else 'OK'}" + (f" ({len(errs)} problems)" if errs else "") + (" (pre-tag)" if a.pre_tag else ""))
      return 1 if errs else 0
    if a.cmd == "draft":
      v = Version.parse(a.version)
      if repo.exists(v.note): raise ReleaseError(f"{v.note} already exists")
      date = a.date or _today()
      line_date = date
      if v.patch > 0:
        base = next((r for r in load_index(repo)[1] if str(r.get("version")) == f"{v.line}.0"), None)
        line_date = str(base["line_date"]) if base else date
      p = repo.root / v.note
      p.write_text(draft_text(repo, v, date, line_date))
      print(f"release: wrote {v.note}; fill in every section, then `scripts/release.py cut {v} --dry-run`")
      return 0
    if a.cmd == "cut": return cut(repo, a.version, dry_run=a.dry_run, date=a.date, remote=a.remote,
                                  validate_version=a.validate_version)
  except (ReleaseError, RM.MatrixError) as e:
    print(f"release: {e}", file=sys.stderr)
    return 1
  return 2


if __name__ == "__main__":
  sys.exit(main())
