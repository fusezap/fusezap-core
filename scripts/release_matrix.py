#!/usr/bin/env python3
"""FuseZap support matrix: chips x models x status, built from evidence in the repo.

The source is a hand-maintained YAML file (docs/releases/matrix/sources.yaml; a release freezes a copy as
docs/releases/matrix/vX.Y.Z.yaml). This script loads and validates it and renders it as:

  md    markdown for release notes (summary grid + one row per cell with metrics and evidence)
  json  for the website's /matrix page (schema fusezap.matrix/v1, cells sorted, evidence kept as written)
  html  a self-contained HTML fragment (one <section>, scoped inline <style>, no scripts, no external resources)

  python scripts/release_matrix.py                      # validate, then print markdown
  python scripts/release_matrix.py --format json --out matrix.json
  python scripts/release_matrix.py --check              # validate only (exit 1 on any problem)

scripts/release.py check uses the same loader (load / validate) on the release's matrix file.

Status rules (docs/releases/README.md "Support matrix"):
  Supported     (from the first public release) passed a gate on real hardware (Validate PROMOTE or a package gate), with evidence that resolves
  Preview       ran on real hardware, but no gate pass yet (or only once, or with a caveat in `notes`)
  Experimental  simulator or behind a flag only
  Planned       not run; no date, no promise

Evidence entries (each must resolve, checked with git, no network):
  docs/status/x/README.md                   a file or directory in the working tree
  <sha>                                     a commit (7-40 hex chars) present in the local object store
  <sha>:docs/status/x/README.md             a file inside that commit (results kept on a results branch)
  private: available on request             evidence that exists but is not published (raw logs, Validate reports);
                                            the public export writes this for every entry the public tree lacks
Branch names are not accepted: a branch moves, a commit does not.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SCHEMA = "fusezap.matrix/v1"
STATUSES = ("Supported", "Preview", "Experimental", "Planned")
VALIDATE_VERDICTS = ("PROMOTE", "REVIEW", "BLOCK")
REPO = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path("docs/releases/matrix/sources.yaml")

_SHA = re.compile(r"^[0-9a-f]{7,40}$")
_SHA_PATH = re.compile(r"^([0-9a-f]{7,40}):(.+)$")
PRIVATE_EVIDENCE = "private: available on request"   # the one accepted form of unpublished evidence
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class MatrixError(ValueError):
  pass


@dataclass
class Metric:
  name: str
  value: float | str
  unit: str = ""
  target: str = ""       # e.g. "<= 40.6 ms (AWS NxD Inference TP=2)": what the number is held against
  baseline: str = ""     # e.g. "vLLM 0.32.0: 5.85 ms": the same-pod comparison, if any

  def text(self) -> str:
    v = f"{self.value:g}" if isinstance(self.value, (int, float)) else str(self.value)
    s = f"{self.name} {v}{(' ' + self.unit) if self.unit else ''}"
    extra = [x for x in (self.baseline, f"target {self.target}" if self.target else "") if x]
    return s + (f" ({'; '.join(extra)})" if extra else "")


@dataclass
class Cell:
  chip: str
  model: str
  status: str
  config: str = ""
  verified: str = ""            # ISO date the evidence was produced on hardware
  gate: str = ""                # what was gated (required for Supported)
  validate: str = ""            # Validate decision, if Validate ran: PROMOTE / REVIEW / BLOCK
  build: str = ""               # engine commit the evidence measured
  metrics: list[Metric] = field(default_factory=list)
  tested_with: dict = field(default_factory=dict)
  evidence: list[str] = field(default_factory=list)
  notes: str = ""

  @property
  def key(self) -> tuple[str, str]:
    return (self.chip, self.model)


@dataclass
class Matrix:
  schema: str
  updated: str
  chips: list[dict]
  models: list[dict]
  cells: list[Cell]
  path: str = ""

  def chip(self, cid: str) -> dict: return next(c for c in self.chips if c["id"] == cid)
  def model(self, mid: str) -> dict: return next(m for m in self.models if m["id"] == mid)
  def cell(self, chip: str, model: str) -> Cell | None: return next((c for c in self.cells if c.key == (chip, model)), None)


# ---------------------------------------------------------------- loading


def _date(v, where: str) -> str:
  if isinstance(v, dt.date): return v.isoformat()
  if isinstance(v, str):
    try: return dt.date.fromisoformat(v).isoformat()
    except ValueError: pass
  raise MatrixError(f"{where}: not an ISO date (YYYY-MM-DD): {v!r}")


def parse(data, path: str = "<matrix>") -> Matrix:
  """Structure checks only (schema, ids, enums, dates). `validate` adds the evidence and status rules."""
  if not isinstance(data, dict): raise MatrixError(f"{path}: top level must be a mapping")
  if data.get("schema") != SCHEMA: raise MatrixError(f"{path}: schema must be {SCHEMA!r}, got {data.get('schema')!r}")
  updated = _date(data.get("updated"), f"{path}: updated")
  chips, models = data.get("chips") or [], data.get("models") or []
  for kind, items in (("chips", chips), ("models", models)):
    if not isinstance(items, list): raise MatrixError(f"{path}: {kind} must be a list")
    seen = set()
    for i, it in enumerate(items):
      if not isinstance(it, dict) or not _ID.match(str(it.get("id", ""))) or not it.get("name"):
        raise MatrixError(f"{path}: {kind}[{i}] needs a lowercase id and a name: {it!r}")
      if it["id"] in seen: raise MatrixError(f"{path}: duplicate {kind} id {it['id']!r}")
      seen.add(it["id"])
  chip_ids, model_ids = {c["id"] for c in chips}, {m["id"] for m in models}
  cells, seen = [], set()
  for i, raw in enumerate(data.get("cells") or []):
    where = f"{path}: cells[{i}]"
    if not isinstance(raw, dict): raise MatrixError(f"{where}: must be a mapping")
    unknown = set(raw) - set(Cell.__dataclass_fields__)
    if unknown: raise MatrixError(f"{where}: unknown keys {sorted(unknown)}")
    if raw.get("chip") not in chip_ids: raise MatrixError(f"{where}: chip {raw.get('chip')!r} is not in chips")
    if raw.get("model") not in model_ids: raise MatrixError(f"{where}: model {raw.get('model')!r} is not in models")
    if raw.get("status") not in STATUSES: raise MatrixError(f"{where}: status must be one of {STATUSES}, got {raw.get('status')!r}")
    where = f"{path}: cell {raw['chip']} x {raw['model']}"
    if (raw["chip"], raw["model"]) in seen: raise MatrixError(f"{where}: duplicate cell (one cell per chip x model)")
    seen.add((raw["chip"], raw["model"]))
    v = raw.get("validate") or ""
    if v and v not in VALIDATE_VERDICTS: raise MatrixError(f"{where}: validate must be one of {VALIDATE_VERDICTS}, got {v!r}")
    metrics = []
    for j, m in enumerate(raw.get("metrics") or []):
      if not isinstance(m, dict) or "name" not in m or "value" not in m:
        raise MatrixError(f"{where}: metrics[{j}] needs name and value")
      extra = set(m) - set(Metric.__dataclass_fields__)
      if extra: raise MatrixError(f"{where}: metrics[{j}] unknown keys {sorted(extra)}")
      metrics.append(Metric(**{k: (str(x) if k != "value" else x) for k, x in m.items()}))
    ev = raw.get("evidence") or []
    if isinstance(ev, str): ev = [ev]
    tw = raw.get("tested_with") or {}
    if not isinstance(tw, dict): raise MatrixError(f"{where}: tested_with must be a mapping")
    cells.append(Cell(chip=raw["chip"], model=raw["model"], status=raw["status"], config=str(raw.get("config") or ""),
                      verified=_date(raw["verified"], f"{where}: verified") if raw.get("verified") else "",
                      gate=str(raw.get("gate") or ""), validate=v, build=str(raw.get("build") or ""), metrics=metrics,
                      tested_with={str(k): str(x) for k, x in tw.items()}, evidence=[str(e) for e in ev],
                      notes=str(raw.get("notes") or "").strip()))
  return Matrix(schema=SCHEMA, updated=updated, chips=chips, models=models, cells=cells, path=path)


def load(path: str | Path, repo: str | Path = REPO, text: str | None = None) -> Matrix:
  """Load a matrix file (relative paths are under `repo`). `text` overrides the file's content (release.py dry runs)."""
  p = Path(path)
  if text is None: text = (p if p.is_absolute() else Path(repo) / p).read_text()
  try: data = yaml.safe_load(text)
  except yaml.YAMLError as e: raise MatrixError(f"{path}: YAML does not parse: {e}") from e
  return parse(data, str(path))


# ---------------------------------------------------------------- validation


def _git_ok(repo: Path, *args: str) -> bool:
  return subprocess.run(["git", "-C", str(repo), *args], capture_output=True).returncode == 0


def resolve_evidence(ref: str, repo: str | Path = REPO) -> str | None:
  """None if `ref` resolves (working-tree path, commit, or commit:path), else the reason it does not."""
  repo = Path(repo)
  if ref == PRIVATE_EVIDENCE: return None
  if not ref or ref.startswith(("http://", "https://")): return "must be a repo path, a commit or commit:path (no URLs)"
  m = _SHA_PATH.match(ref)
  if m:
    if not _git_ok(repo, "cat-file", "-e", f"{m.group(1)}^{{commit}}"): return f"commit {m.group(1)} is not in the local repo"
    return None if _git_ok(repo, "cat-file", "-e", ref) else f"{m.group(2)} does not exist in commit {m.group(1)}"
  if _SHA.match(ref) and not (repo / ref).exists():
    return None if _git_ok(repo, "cat-file", "-e", f"{ref}^{{commit}}") else f"commit {ref} is not in the local repo"
  p = Path(ref)
  if p.is_absolute() or ".." in p.parts: return "repo paths must be relative and stay inside the repo"
  return None if (repo / p).exists() else f"{ref} does not exist in the working tree"


def validate(mx: Matrix, repo: str | Path = REPO) -> list[str]:
  """Every problem as one line; empty list = valid."""
  errs = []
  for c in mx.cells:
    where = f"{mx.path}: {c.chip} x {c.model} ({c.status})"
    if c.status != "Planned":
      if not c.verified: errs.append(f"{where}: needs a verified date")
      if not c.evidence: errs.append(f"{where}: needs at least one evidence entry")
    if c.status == "Supported":
      if not c.gate: errs.append(f"{where}: Supported needs `gate` (what passed on real hardware)")
      if c.validate and c.validate != "PROMOTE": errs.append(f"{where}: Supported with Validate {c.validate}: use Preview")
    if c.status == "Planned" and (c.validate or c.metrics):
      errs.append(f"{where}: a Planned cell has no Validate verdict or metrics")
    for e in c.evidence:
      why = resolve_evidence(e, repo)
      if why: errs.append(f"{where}: evidence {e!r}: {why}")
  return errs


# ---------------------------------------------------------------- rendering


def _sorted_cells(mx: Matrix) -> list[Cell]:
  ci = {c["id"]: i for i, c in enumerate(mx.chips)}
  mi = {m["id"]: i for i, m in enumerate(mx.models)}
  return sorted(mx.cells, key=lambda c: (STATUSES.index(c.status), ci[c.chip], mi[c.model]))


def _used(mx: Matrix):
  chips = [c for c in mx.chips if any(x.chip == c["id"] for x in mx.cells)]
  models = [m for m in mx.models if any(x.model == m["id"] for x in mx.cells)]
  return chips, models


def _md(s: str) -> str: return s.replace("|", "\\|").replace("\n", " ")


def short_ref(ref: str) -> str:
  """A commit-pinned evidence ref with its sha cut to 12 characters, for display (JSON keeps the full ref)."""
  m = _SHA_PATH.match(ref)
  if m: return f"{m.group(1)[:12]}:{m.group(2)}"
  return ref[:12] if _SHA.match(ref) else ref


def to_markdown(mx: Matrix) -> str:
  chips, models = _used(mx)
  out = [f"Support matrix as of {mx.updated}. Status: **Supported** = passed a gate on real hardware; **Preview** = ran "
         "on real hardware, not gated yet; **Experimental** = simulator or flag only; **Planned** = not run, no date.", ""]
  out.append("| Model | " + " | ".join(_md(c["name"]) for c in chips) + " |")
  out.append("|---|" + "---|" * len(chips))
  for m in models:
    row = []
    for c in chips:
      cell = mx.cell(c["id"], m["id"])
      row.append("—" if cell is None else (f"**{cell.status}**" if cell.status == "Supported" else cell.status))
    out.append(f"| {_md(m['name'])} | " + " | ".join(row) + " |")
  out += ["", "| Chip | Model | Config | Status | Verified | Validate | Key metrics | Tested with | Evidence |",
          "|---|---|---|---|---|---|---|---|---|"]
  for c in _sorted_cells(mx):
    ev = "<br>".join(f"`{_md(short_ref(e))}`" for e in c.evidence) or "—"
    met = "<br>".join(_md(m.text()) for m in c.metrics) or "—"
    tw = ", ".join(f"{k} {v}" for k, v in c.tested_with.items()) or "—"
    out.append(f"| {_md(mx.chip(c.chip)['name'])} | {_md(mx.model(c.model)['name'])} | {_md(c.config) or '—'} | {c.status} | "
               f"{c.verified or '—'} | {c.validate or '—'} | {met} | {_md(tw)} | {ev} |")
  notes = [(c, c.notes) for c in _sorted_cells(mx) if c.notes]
  if notes:
    out += ["", "Notes:"]
    out += [f"- {mx.chip(c.chip)['name']} x {mx.model(c.model)['name']}: {_md(n)}" for c, n in notes]
  return "\n".join(out) + "\n"


def to_json(mx: Matrix) -> dict:
  def cell(c: Cell) -> dict:
    return {"chip": c.chip, "model": c.model, "status": c.status, "config": c.config, "verified": c.verified or None,
            "gate": c.gate or None, "validate": c.validate or None, "build": c.build or None,
            "metrics": [{"name": m.name, "value": m.value, "unit": m.unit or None, "target": m.target or None,
                         "baseline": m.baseline or None} for m in c.metrics],
            "tested_with": c.tested_with, "evidence": c.evidence, "notes": c.notes or None}
  return {"schema": SCHEMA, "updated": mx.updated, "statuses": list(STATUSES), "chips": mx.chips, "models": mx.models,
          "cells": [cell(c) for c in _sorted_cells(mx)]}


_CSS = """
.fz-matrix{--fzm-fg:#1a1a1a;--fzm-muted:#5c5c5c;--fzm-line:#d9d9d9;--fzm-bg:#ffffff;--fzm-head:#f4f4f4;
 --fzm-sup:#0b6b3a;--fzm-pre:#7a5200;--fzm-exp:#4a3d8f;--fzm-pla:#5c5c5c;
 color:var(--fzm-fg);background:var(--fzm-bg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){.fz-matrix{--fzm-fg:#ececec;--fzm-muted:#a8a8a8;--fzm-line:#3a3a3a;--fzm-bg:#161616;
 --fzm-head:#222222;--fzm-sup:#5fd394;--fzm-pre:#f0b955;--fzm-exp:#a99cf0;--fzm-pla:#a8a8a8}}
.fz-matrix .fzm-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.fz-matrix table{border-collapse:collapse;width:100%;margin:0 0 1.25em}
.fz-matrix th,.fz-matrix td{border:1px solid var(--fzm-line);padding:6px 8px;text-align:left;vertical-align:top}
.fz-matrix th{background:var(--fzm-head);font-weight:600}
.fz-matrix .fzm-s{font-weight:600;white-space:nowrap}
.fz-matrix .fzm-Supported{color:var(--fzm-sup)}.fz-matrix .fzm-Preview{color:var(--fzm-pre)}
.fz-matrix .fzm-Experimental{color:var(--fzm-exp)}.fz-matrix .fzm-Planned{color:var(--fzm-pla)}
.fz-matrix .fzm-muted,.fz-matrix caption{color:var(--fzm-muted)}
.fz-matrix caption{caption-side:top;text-align:left;padding:0 0 6px}
.fz-matrix code{font-size:12px;word-break:break-all}
"""


def to_html(mx: Matrix) -> str:
  e = html.escape
  chips, models = _used(mx)
  h = [f'<section class="fz-matrix" data-schema="{SCHEMA}" data-updated="{e(mx.updated)}">', f"<style>{_CSS.strip()}</style>"]
  h.append('<div class="fzm-scroll"><table><caption>Support matrix as of ' + e(mx.updated) + "</caption><thead><tr><th>Model</th>"
           + "".join(f"<th>{e(c['name'])}</th>" for c in chips) + "</tr></thead><tbody>")
  for m in models:
    h.append(f"<tr><th>{e(m['name'])}</th>")
    for c in chips:
      cell = mx.cell(c["id"], m["id"])
      h.append('<td class="fzm-muted">—</td>' if cell is None else
               f'<td><span class="fzm-s fzm-{cell.status}">{cell.status}</span>'
               + (f'<br><span class="fzm-muted">{e(cell.config)}</span>' if cell.config else "") + "</td>")
    h.append("</tr>")
  h.append("</tbody></table></div>")
  h.append('<div class="fzm-scroll"><table><thead><tr><th>Chip</th><th>Model</th><th>Status</th><th>Verified</th>'
           "<th>Validate</th><th>Key metrics</th><th>Evidence</th></tr></thead><tbody>")
  for c in _sorted_cells(mx):
    met = "<br>".join(e(m.text()) for m in c.metrics) or "—"
    ev = "<br>".join(f'<code title="{e(x)}">{e(short_ref(x))}</code>' for x in c.evidence) or "—"
    note = f'<br><span class="fzm-muted">{e(c.notes)}</span>' if c.notes else ""
    h.append(f"<tr><td>{e(mx.chip(c.chip)['name'])}</td><td>{e(mx.model(c.model)['name'])}"
             + (f'<br><span class="fzm-muted">{e(c.config)}</span>' if c.config else "")
             + f'</td><td><span class="fzm-s fzm-{c.status}">{c.status}</span></td><td>{e(c.verified or "—")}</td>'
             f"<td>{e(c.validate or '—')}</td><td>{met}{note}</td><td>{ev}</td></tr>")
  h.append("</tbody></table></div></section>")
  return "\n".join(h) + "\n"


RENDER = {"md": to_markdown, "json": lambda mx: json.dumps(to_json(mx), indent=1, ensure_ascii=False) + "\n", "html": to_html}


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("source", nargs="?", default=str(DEFAULT_SOURCE), help=f"matrix YAML (default {DEFAULT_SOURCE})")
  ap.add_argument("--repo", default=str(REPO), help="repo root that relative paths and evidence resolve against")
  ap.add_argument("--format", choices=sorted(RENDER), default="md")
  ap.add_argument("--out", help="write here instead of stdout")
  ap.add_argument("--check", action="store_true", help="validate only")
  ap.add_argument("--no-evidence-check", action="store_true", help="skip the evidence/status rules (render a draft)")
  a = ap.parse_args(argv)
  try: mx = load(a.source, a.repo)
  except (MatrixError, OSError) as e:
    print(f"release_matrix: {e}", file=sys.stderr); return 1
  errs = [] if a.no_evidence_check else validate(mx, a.repo)
  for x in errs: print(f"release_matrix: {x}", file=sys.stderr)
  if errs: return 1
  if a.check:
    n = {s: sum(c.status == s for c in mx.cells) for s in STATUSES}
    print(f"release_matrix: {a.source} OK: {len(mx.cells)} cells ({', '.join(f'{v} {k}' for k, v in n.items())})")
    return 0
  text = RENDER[a.format](mx)
  if a.out: Path(a.out).write_text(text)
  else: sys.stdout.write(text)
  return 0


if __name__ == "__main__":
  sys.exit(main())
