# FuseZap {version}

<!--
Release note template. Copy to docs/releases/v{version}.md (or run `scripts/release.py draft {version}`).
Every `## ` heading below is required: `scripts/release.py check` fails a tag whose note is missing one, and fails
when "Breaking changes" is empty. HTML comments do not count as content. Write "None." where a section has nothing.
Policy: docs/releases/README.md. Customer-facing language; no customer names, no prices.
-->

Released {date}. Line {line} (first shipped {line_date}). Core {version}, Invarix {invarix_version}.

## Highlights

<!-- Three to five bullets a user would care about. Link the matrix row or the benchmark for every claim. -->

## New chips and models

<!-- New cells in the support matrix since the previous release, with their status. "None." if none. -->

## Support matrix

<!-- Paste `python scripts/release_matrix.py docs/releases/matrix/v{version}.yaml` here (cut freezes the matrix as
that file). Every Supported cell links its evidence. Add the "tested with" versions (driver, SDK, vLLM baseline). -->

## Fixes

<!-- User-visible fixes: what was wrong, who was affected, which versions. -->

## Performance

<!-- Each number with its baseline, hardware, commit and a command to reproduce it, e.g.
- Qwen2.5-7B bf16, 1x H100: TPOT p50 c1 5.57 ms vs vLLM 0.32.0 5.85 ms, commit 323463f,
  the command that produced it (docs/BENCH_METHOD.md). -->

## Breaking changes

<!-- Required. "None." when nothing breaks. Never in a patch release (see the policy). -->

## Deprecations

<!-- Stage 1 (warning), 2 (off by default) or 3 (removed) for each FZ_* variable, CLI flag, HTTP API field, report
schema field or Invarix exit code, with the release it moves to the next stage. "None." if none. -->

## Security

<!-- Fixed advisories (ID, severity, affected versions). "None." if none. Report issues as SECURITY.md says. -->

## Known issues

<!-- Each with a workaround and the target release. -->

## Upgrade notes

<!-- What to do when moving from the previous minor: env changes, cache resets, report-schema changes. -->

## What's next

<!-- Three horizons, no dates (policy "What's next"):
- Next release (in progress, link the branch)
- Exploring (no commitment)
- Not planned
Then what happened to the previous note's "Next" items: shipped, slipped (and why) or dropped. -->

## Checksums

<!-- sha256 of every published artifact (wheel, sdist), e.g. `shasum -a 256 dist/*`. -->
