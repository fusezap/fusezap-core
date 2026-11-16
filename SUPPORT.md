# Getting help with FuseZap

FuseZap Core is open source (Apache-2.0) and before 1.0. Support is community support through GitHub, plus the
fixes the release policy promises for supported versions.

## Where to ask

| What | Where |
|---|---|
| A bug, wrong output, a crash or hang, a speed regression | a GitHub issue (the bug, performance and backend forms) |
| A security issue | privately, as [SECURITY.md](SECURITY.md) says: never in a public issue |
| Anything else (questions, evaluations, partnerships) | hello@fusezap.ai |

A first response to an issue comes within 5 working days ([GOVERNANCE.md](GOVERNANCE.md)).

## What to include

- the version: `python -c "import fusezap; print(fusezap.__version__)"`,
- the hardware and backend (CUDA, METAL, NEURON, CPU), and for NEURON the Neuron SDK version,
- the model, its dtype and the command (or the HTTP request, for `fusezap serve`) that shows the problem,
- what you expected and what you got; for wrong output, the prompt and the tokens or text.

## Which versions get fixes

The latest minor release and the one before it (N-1). Fixes ship as patch releases (0.N.x) when they qualify: wrong
output, a crash or a hang, a security issue, a speed regression of more than 5% on a Supported chip, or new model
support that needs no API change. Everything else lands in the next minor. Which models run on which chips, and how
well tested each pair is, is the support matrix in each release note. The full policy is
[docs/releases/README.md](docs/releases/README.md).
