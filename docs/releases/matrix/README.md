# Support matrix schema (fusezap.matrix/v1)

`sources.yaml` is the live matrix, edited by hand when hardware evidence lands. `scripts/release.py cut X.Y.Z` copies
it to `vX.Y.Z.yaml`, which is then frozen. `scripts/release_matrix.py` loads, validates and renders either file:

```
python scripts/release_matrix.py --check                       # validate sources.yaml
python scripts/release_matrix.py                               # markdown (release notes)
python scripts/release_matrix.py --format json --out m.json    # the website's /matrix page
python scripts/release_matrix.py --format html --out m.html    # a self-contained HTML fragment
python scripts/release_matrix.py docs/releases/matrix/v0.1.0.yaml   # a frozen release matrix
```

## Top level

| Key | Type | Notes |
|---|---|---|
| `schema` | string | `fusezap.matrix/v1` |
| `updated` | date | when the file was last reviewed |
| `chips` | list | `{id, name, vendor, backend, hardware}`; `id` lowercase `[a-z0-9._-]`, unique |
| `models` | list | `{id, name, hf}`; `id` lowercase, unique |
| `cells` | list | one per chip x model (duplicates are an error) |

Column order in the rendered tables follows the order of `chips` and `models`; chips and models without a cell are
left out.

## Cell

| Key | Required | Notes |
|---|---|---|
| `chip`, `model` | yes | ids from `chips` / `models` |
| `status` | yes | `Supported`, `Preview`, `Experimental` or `Planned` (rules in docs/releases/README.md) |
| `config` | no | dtype, parallelism, instance and serve flags, e.g. `bf16, TP=2 (2x H200), fusezap serve --tp 2` |
| `verified` | yes, unless Planned | ISO date the evidence was produced on hardware |
| `gate` | yes for Supported | what passed on real hardware |
| `invarix` | no | Invarix decision if Invarix ran: `PROMOTE`, `REVIEW` or `BLOCK`. A Supported cell cannot carry REVIEW or BLOCK |
| `build` | no | the engine commit that was measured |
| `metrics` | no | list of `{name, value, unit, target, baseline}`; `target` = what it is held against, `baseline` = the same-pod comparison. A Planned cell has none |
| `tested_with` | no | mapping, e.g. `{vllm: 0.32.0, neuron_sdk: 2.33.0, driver: 595.91.07}` |
| `evidence` | yes, unless Planned | list; each entry must resolve (below) |
| `notes` | no | caveats, in customer-facing language |

## Evidence entries

Checked with git only (no network):

- `docs/status/<run>/README.md`: a file or directory in the working tree (relative, inside the repo).
- `<40-hex commit>`: a commit in the local object store (7 to 40 hex characters).
- `<commit>:docs/status/<run>/README.md`: a file inside a commit, for results kept on a results branch.
- `private: available on request`: evidence that exists but is not published (raw run logs, FuseZap Invarix
  reports). The public source tree is built with this form for every entry it does not contain.

Branch names and URLs are refused: a branch moves, and a URL cannot be checked offline. A commit pinned this way must
stay reachable (keep its branch, or merge the results), or `check` fails on a fresh clone.

## JSON output

`{schema, updated, statuses, chips, models, cells}`. Cells are sorted by status (Supported first), then chip and model
order. Each cell has every key above; absent values are `null`, `metrics` and `evidence` are lists, evidence refs are
kept in full. The markdown and HTML outputs shorten commit refs to 12 characters for display.
