# How FuseZap Core is run

FuseZap Core is maintained by FuseZap (Techyodl). This page says who decides what, how a pull request is accepted or
declined, and how long you can expect to wait.

## Roles

| Role | Who | What they do |
|---|---|---|
| **Maintainer** | Listed in [MAINTAINERS.md](MAINTAINERS.md) | Final say on scope and design, merges, releases, repository settings |
| **Reviewer** | Listed in [MAINTAINERS.md](MAINTAINERS.md) and [.github/CODEOWNERS](.github/CODEOWNERS) for their areas | Review and approve pull requests in the areas they own |
| **Contributor** | Anyone | Issues, discussion, pull requests (signed off, see [CONTRIBUTING.md](CONTRIBUTING.md)) |

Reviewers are added by a maintainer after a record of good reviews or contributions in an area. Maintainers are added by
the existing maintainers.

## Response time

We aim to give every new issue and pull request a first response within **5 working days**: a label, a question, a
review, or a decision. If you have heard nothing after that, a polite ping on the thread is welcome.

## How a pull request is decided

1. **Checks.** CI (`test-cpu`, `build`) and the sign-off check (`DCO`) must pass. For a first-time contributor, a
   maintainer approves the CI run first.
2. **Triage.** A maintainer labels the pull request (`area/*`, `type/*`) and checks it is in scope. A large change
   without an agreed issue is labelled `status/needs-issue` and paused until the approach is agreed.
3. **Review.** GitHub requests the code owner of each area touched. A pull request needs at least one approving review
   from a code owner, every review thread resolved, and the branch up to date with `main`. A new push dismisses earlier
   approvals.
4. **Landing.** FuseZap Core is developed in an internal repository and published here. An approved pull request is
   tested there on the hardware we have (including GPUs and AWS Inferentia2/Trainium1 where it matters), then
   squash-merged here, so it shows as merged under your name, and labelled `status/landed`. Your authorship and sign-off
   are kept.
5. **Declined.** When we decline a pull request we say why, in one of these terms, and what would change the answer:
   - out of scope for FuseZap Core (see the roadmap on [fusezap.ai](https://fusezap.ai/status.html));
   - a large change that needs an agreed issue first;
   - no tests, or tests that do not cover the change;
   - a performance claim without the hardware, the baseline and the command that reproduces it
     ([docs/BENCH_METHOD.md](docs/BENCH_METHOD.md));
   - commits without a `Signed-off-by` line;
   - a change we can't maintain (for example, a dependency or backend we can't test).

Disagreements are settled on the thread; a maintainer makes the final call and records the reason.

## Security

Never in a public issue or pull request. See [SECURITY.md](SECURITY.md).
