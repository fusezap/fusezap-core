# FuseZap releases: policy

This is the release policy for FuseZap Core (the engine, Apache-2.0) and FuseZap Invarix (the release
qualification tool). Nothing has been released yet; the
first release will be 0.1.0.

Files here:

| File | What |
|---|---|
| `README.md` | this policy |
| `TEMPLATE.md` | the release-note template; every `## ` heading in it is required in each note |
| `vX.Y.Z.md` | one release note per version (`scripts/release.py draft X.Y.Z` starts one) |
| `index.yaml` | every release: version, build date, line date, support status, Invarix maintenance needed |
| `matrix/sources.yaml` | the live support matrix (chips x models x status, with evidence) |
| `matrix/vX.Y.Z.yaml` | the matrix frozen at each release by `cut` |
| `matrix/README.md` | the matrix schema |

Elsewhere: `CHANGELOG.md` (Keep a Changelog), `SECURITY.md`, `scripts/release.py`, `scripts/release_matrix.py`, and the
"Release note: … / none" line in `.github/pull_request_template.md`. Published copies: GitHub Releases (which give an
Atom feed) and fusezap.ai/releases with RSS.

## One shared minor train

"FuseZap 0.3" means Core 0.3.x plus Invarix 0.3.x, tested together and released on the same day. Before 1.0 a minor
ships monthly, on the first Tuesday of the month. Patches (0.3.1, 0.3.2) ship when needed, and independently per
product: a Core 0.3.2 can ship while Invarix stays at 0.3.1. The index records both versions of every release
(`products`).

## Versioning

- [SemVer](https://semver.org/), staying in 0.y.z until 1.0. The minor (y) is the train; the patch (z) is a fix on it.
- **Nothing is removed in a patch.** Breaking changes and removals only land in a minor, after the deprecation stages
  below. `scripts/release.py check` fails a patch whose note lists breaking changes.
- Pre-releases are `0.3.0rc1`, `0.3.0rc2`, ... (PEP 440, which is also the git tag after a `v`: `v0.3.0rc1`). Between
  releases `main` carries the next minor's dev version, e.g. `0.4.0.dev0`.
- **1.0** comes when the server API (the HTTP API of `fusezap serve`) and the backend porting API are frozen.
- The version has one source, `fusezap/_version.py`. `pyproject.toml` reads it (`[tool.hatch.version]`) and
  `fusezap.__version__` imports it. Invarix keeps its own version (`fusezap.invarix.__version__`) until it moves to
  its own package; both are recorded per release in `index.yaml`.

## Support: the latest minor and the one before (N-1)

- The latest minor (`current`) and the previous one (`maintained`, N-1) get fixes.
- Invarix lines get security fixes for **6 months** after the line first shipped (`security`), then `eol`.
- `cut` updates the `support` field of older entries when a new minor ships.

## What qualifies for a patch

A patch release carries only:

1. wrong output (a token, logprob or report value that is incorrect),
2. a crash or a hang,
3. a security issue,
4. a speed regression of more than 5% on a Supported chip (the support matrix) against the previous release on the
   same line, measured as docs/BENCH_METHOD.md describes,
5. new model support that needs no API change.

Anything else waits for the next minor.

## Deprecation: three stages

1. **Warning**: the old behaviour still works and prints a deprecation warning naming the replacement and the
   release it moves to stage 2.
2. **Off by default**: the old behaviour needs an explicit opt-in.
3. **Removed**.

Removal is at least 2 minors after stage 1, and never in a patch. This covers `FZ_*` environment variables, CLI flags,
the HTTP API, the Invarix report schema (`fusezap.invarix/vN`) and Invarix's exit codes. Every stage change is listed
under "Deprecations" in the release note.

## Release notes

`docs/releases/vX.Y.Z.md`, from `TEMPLATE.md`: Highlights; New chips and models; Support matrix (chips x models x
status, each with an evidence link, plus "tested with" versions); Fixes; Performance (each number with its baseline,
hardware, commit and a command to reproduce it); Breaking changes ("None." when nothing breaks); Deprecations;
Security; Known issues (with workaround and target release); Upgrade notes; What's next; Checksums. Customer-facing
language, no customer names, no prices.

### What's next, without overpromising

Three horizons, no dates:

- **Next release**: in progress, with a link to the branch.
- **Exploring**: being looked at, no commitment.
- **Not planned**: asked for, and we are not doing it.

Each note also says what happened to the previous note's "Next" items: shipped, slipped (and why), or dropped.
Per-chip commitments go in the partner contract, not the public note.

## Support matrix

Each cell is a chip x model with a status:

| Status | Means |
|---|---|
| **Supported** | starts with the first public release: a cell in a released version that passed a gate on real hardware (Invarix PROMOTE, or a package gate), with evidence that resolves to a repo file or a commit. No cell is Supported yet |
| **Preview** | ran on real hardware in a development build. Every cell is Preview until the first public release; the gate it passed, if any, and any caveat are stated in the cell |
| **Experimental** | simulator only, or behind a flag |
| **Planned** | not run; no date is promised |

The source is `matrix/sources.yaml`, edited by hand when evidence lands (schema in `matrix/README.md`).
`scripts/release_matrix.py` renders it as markdown (release notes), JSON (the website's /matrix page) and an HTML
fragment. `cut` freezes it as `matrix/vX.Y.Z.yaml`; `check` validates the frozen copy with the same loader.

## Line date vs build date (Invarix maintenance)

Every release has two dates in `index.yaml`:

- `date`, the **build date**: the day this version was cut.
- `line_date`, the **line date**: the day X.Y.0 first shipped. Every patch on the line repeats it.

Invarix maintenance is checked against the **line date**, never the build date (`invarix_maintenance_needed` =
`line_date`). A Invarix customer entitled to line 0.3 therefore receives every 0.3.x patch, including a security fix
built after their maintenance ended; renewed maintenance is needed only to move to a line that first shipped after it
ended.

## How to cut a release

1. On `main`: fill `CHANGELOG.md`'s Unreleased section, update `matrix/sources.yaml`, run
   `scripts/release.py draft X.Y.0`, fill every section of the note, and commit.
2. `scripts/release.py cut X.Y.0 --dry-run`, then `scripts/release.py cut X.Y.0`. It refuses on a dirty tree. It
   sets the version, moves Unreleased into `## [X.Y.0] - <date>`, freezes the matrix, adds the index entry, commits
   "Release X.Y.0", creates `release/X.Y` (first cut on a line, including an rc1), makes the annotated tag `vX.Y.0`
   locally, checks it, and moves `main` to `X.(Y+1).0.dev0`.
3. It prints the push commands. **Pushing stays manual and approved**: nothing is pushed by the tool.
4. Patches: commit the fixes on `release/X.Y` (cherry-picked from `main`), draft and fill `vX.Y.Z.md`, then
   `cut X.Y.Z` on that branch. Bring the CHANGELOG section and index entry back to `main`.
5. CI (when it exists) runs `scripts/release.py check vX.Y.Z` on the tag.
