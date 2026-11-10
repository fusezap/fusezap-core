# Security policy

## Reporting a vulnerability

Please report security issues privately. Do not open a public issue, pull request or discussion.

- Email **security@fusezap.ai**.
- Or use GitHub's private vulnerability reporting ("Report a vulnerability" on the repository's Security tab) once
  the repository is public.

Include what you can of:

- the affected product (FuseZap Core or FuseZap Invarix) and version (`python -c "import fusezap; print(fusezap.__version__)"`),
- the hardware and backend (CUDA, METAL, NEURON, CPU) and the command or request that triggers it,
- the impact you see (for example: remote code execution through a model file, a request that crashes or hangs
  `fusezap serve`, a secret written to a log or report),
- a minimal reproduction, and whether the issue is already public.

We will acknowledge your report within 3 business days, keep you updated at least weekly while we work on it, and
credit you in the advisory unless you ask us not to. Please give us up to 90 days to ship a fix before disclosing.

## Supported versions

FuseZap is before 1.0 and ships on one monthly release train (docs/releases/README.md).

| Version | Security fixes |
|---|---|
| Latest minor (0.N) | yes |
| Previous minor (0.N-1) | yes |
| Older Invarix lines | security fixes for 6 months after the line first shipped |
| Anything else | no; please upgrade |

Nothing has been released yet, so no version is listed as supported today.

Security fixes ship as patch releases (0.N.x) and are listed in the release note's "Security" section and in the
GitHub security advisory. A security fix never removes or changes a public interface in a patch beyond what the fix
requires.

## Scope

In scope: the `fusezap` package (the engine, `fusezap serve` and its HTTP API, model and checkpoint loading) and the
scripts in this repository that a user is told to run. FuseZap Invarix is licensed separately and is not in this
repository; report its security issues to the same address.

Out of scope: vulnerabilities in third-party dependencies with no FuseZap-specific impact (report them upstream),
and denial of service that needs more requests than the configured limits allow.
