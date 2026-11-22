# prototype_testbed — the sponsor prototype arm

Runs the **sponsor prototype's individual-criteria assessment** N times and writes
prediction matrices in the shape `n8n_testbed.metrics` already scores.

The unit of observation is the **(requirement × criterion) decision cell**: for each
requirement, does it violate each INCOSE criterion, one at a time. 100 runs turns
each cell into a flag frequency, which is what the KPI engine aggregates.

## Why this exists alongside `n8n_testbed.run_matrix --configs baseline`

That arm calls itself "the prototype as shipped". It is not, on two independent axes.

**1. It rebuilds the prompt instead of calling the prototype.**
`shared/evaluator.py` is already a verbatim port of the prototype's `ai_analyzer.py`.
`n8n_testbed/inject.py` reconstructs the prompt around it, and the reconstruction
drops exactly the mechanisms that make the prototype's per-criterion behaviour
distinctive:

| | prototype | `inject.py` |
|---|---|---|
| Decision output | dense — 7 × `satisfied: true/false` | sparse — `{"violated": [...]}` (`OUTPUT_MODE="minimal"`) |
| Span mandate | `affected_text` + replacement **required** | absent |
| Temperature | 0.1 | 0.7 — *"raised from 0.1 so repeated runs show real variance"* |
| max_tokens | 1200 | 500 |
| Context handling | no leniency clause | adds *"if the context or ConOps supplies the need… do not mark violated"* |
| Prompt shape | single user message | split system (cached prefix) + user |
| Model | `claude-sonnet-4-5` | `claude-sonnet-4-6` |

Sparse elicitation at 0.7 with no span requirement is structurally **n8n's**
process, not the prototype's.

**2. It feeds re-authored requirements.**
`run_matrix` reads `data/specs/Project Specifications.xlsx`. That workbook does not
contain the prototype's input; it contains a cleaned rewrite of it. For PM, **all 13
requirements differ**:

| | `.txt` (prototype input) | `Project Specifications.xlsx` |
|---|---|---|
| IDs | `1` … `13` | `FR.1` … `RR.1` |
| FR.1 | `…to a fixed destination point.` + all 4 Notes inline | `…at (X, Y, Z) = (0 mm, 0 mm, 101.6 mm) to a…`, Notes folded/dropped |
| PR.1 | `execute FR1 accurately with +/- 2.5mm [0.1<?>] accuracy` | `execute FR.1 with an accuracy of ±2.5 mm (±0.1 inches)` |
| ER.1 | `atmosphere comparable to that of Earth. Assume 21 degrees centigrade` | `Earth-like atmosphere at approximately 21°C` |
| RR.1 | `shall not use more than 1.5 Watt-hours` + `Note:` | `shall not consume more than 1.5 watt-hours of energy, assuming…` |

Those edits land directly on the evidence for the criteria being scored — A5
(the dropped Notes), A4 (repaired mojibake and units), A10 (`system`→`System`,
the `accurately with…accuracy` redundancy), and A9/A4.3 (the `.txt` says
"execute **FR1**" but no requirement is labelled FR1, a dangling reference the
rewrite resolves away).

Rule-injection configs (M1/M2/ICAI…) still belong in `n8n_testbed.inject` — modifying
the criteria block is its job. The **baseline** must not go through it.

## Layout

| Path | Role |
|---|---|
| `txt_input.py` | Ingests the `.txt` pair exactly as the prototype does; joins parser IDs to GT columns |
| `vendored_parser.py` | Copy of the prototype's `requirements_parser.py`, used when no checkout is present |
| `diagnostics.py` | The recall losses the matrix cannot show |
| `prototype_replica.py` | The runner |

## Fidelity — reproduced, including the defects

- one call **per requirement**, `ThreadPoolExecutor(min(10, n))` — no set context
- temperature **0.1**, `max_tokens` **1200**, model **`claude-sonnet-4-5`**
- dense output: seven `satisfied` verdicts
- the span mandate
- a criterion the model omits is backfilled `satisfied: True`
- `_error_result` marks **all seven** satisfied

The last two are how the prototype converts a failure into seven clean cells. They
are kept — but **counted**, so a failed run is visible instead of scoring as
"found no issues". Retries apply to transport errors only; a requirement that still
fails is recorded as the prototype would record it, and counted.

## Diagnostics

| Counter | Detects |
|---|---|
| `not_evaluated` | criterion the model skipped, backfilled as a pass |
| `error_result` | prototype's failure path fired → 7 phantom passes (counted per criterion) |
| `span_missing` | violated but no `affected_text`, despite the prompt requiring it |
| `span_not_found` | `affected_text` is not a substring — a hallucinated span |
| `placeholder_fix` | violation survived with `[No suggestion provided — review manually]` |

`span_missing` + `span_not_found`, broken out per criterion, are the **measurement**
of the span mandate: the claim is that requiring a verbatim substring suppresses the
criteria whose defect is an *absence* (A2, A5, A9) relative to those whose defect is
a token that is *present* (A4, A6, A10). These counters test that instead of assuming it.

## Running

```bash
# what the prototype would receive — no API calls
python -m prototype_testbed.txt_input --dataset PM --diff-spec

# 1 run, to confirm the path is live before committing to 100
python -m prototype_testbed.prototype_replica --runs 1 --dataset PM --config protosmoke

# the real thing
python -m prototype_testbed.prototype_replica --runs 100 --dataset PM

# scored by the existing KPI engine, unchanged
python -m n8n_testbed.metrics --datasets PM --configs prototype n8nbatch
```

Matrices land in `n8n_testbed/runs/` (where `metrics.py` looks) as
`PM_prototype_matrix.xlsx` + `PM_prototype_raw.json`. The raw dump pins the input
files, the parser used, the ID join, the model, the temperature and the sponsor
checkout's git SHA — "the prototype" is a moving target.

`runs/` is versioned. Re-running with an existing `--config` overwrites a matrix a
published figure may depend on; use a new tag.

## Cost

One call per requirement per run — 13 × 100 = **1,300 calls** for PM, against the
n8n arm's 100 whole-set calls. Dense output at `max_tokens=1200` versus the
minimal arm's 500, and `evaluator.py`'s single-message prompt cannot use the
system-prefix caching `inject.py` relies on.

`max_tokens=1200` is also tight: seven explanations plus seven spans plus
`suggested_full_text` can truncate on a heavily-violating requirement →
`JSONDecodeError` → `_error_result` → all seven recorded as passes. That biases
against exactly the worst requirements. `error_result` catches it; watch that counter.
