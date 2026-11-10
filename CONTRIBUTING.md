# Contributing to FuseZap Core

Thanks for your interest. FuseZap Core is pre-release, so interfaces still move; please open an issue before a large
change so we can agree on the approach first.

## Pull requests

1. Fork, branch from `main`, and keep each pull request to one change.
2. Add or update tests. Run the suite before you push:
   ```sh
   uv sync --extra serve
   uv run pytest -n auto
   ```
   Hardware you do not have (CUDA, METAL, NEURON) is skipped; say in the pull request which tiers and hardware you ran.
   Performance claims need the hardware, the baseline and the command that reproduces them (docs/BENCH_METHOD.md).
3. Fill in the pull request template, including the one-line "Release note:" (or "none"), which goes into
   CHANGELOG.md.
4. CI must pass, and a code owner reviews every change before it is merged. How pull requests are decided, who
   reviews them, and how long you can expect to wait (a first response within 5 working days) are in
   [GOVERNANCE.md](GOVERNANCE.md).

## Sign your commits (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/) (DCO) instead of a CLA. By adding a
`Signed-off-by` line to each commit you certify that you wrote the change, or otherwise have the right to submit it
under the Apache License 2.0:

```sh
git commit -s -m "Fix the thing"
```

This adds `Signed-off-by: Your Name <you@example.com>`, which must match the commit author. Pull requests with
unsigned commits cannot be merged.

## Reporting bugs and security issues

Bugs: open a GitHub issue with the version (`python -c "import fusezap; print(fusezap.__version__)"`), hardware,
backend, the command or request, and what you expected. Security issues: never in a public issue; see
[SECURITY.md](SECURITY.md).

## Conduct

Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md).

By contributing you agree that your contribution is licensed under the [Apache License 2.0](LICENSE).
