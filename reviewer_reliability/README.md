# Reviewer reliability — component (c)

Scores the **SME reviewers** against ground truth, rather than the model. Produces
`docs/figures/PM_reviewer_rates_adjudicated.png`.

## What is being measured

Each SME saw only the violations the AI flagged, and said accept / modify / reject on
each. That makes the reviewer a binary classifier over the flagged set:

```
TP  endorsed & GT violation      FP  endorsed & GT clean
FN  rejected & GT violation      TN  rejected & GT clean
```

The universe is the AI's flags, not all 91 (requirement × criterion) cells — a
reviewer is never asked about a defect the AI missed. These rates therefore describe
**reviewing behaviour, not independent detection**, and the chart caption says so.

## The `modify` problem

A `modify` is neither an endorsement nor a rejection on its face, and the policy you
pick moves the numbers materially. Four are implemented:

| Policy | Rule |
|---|---|
| `adjudicated` | **The published one.** All 19 `modify` decisions labelled by hand, case by case, against the criterion, the AI's suggestion, the SME's final text and their note. The table is `reviewer_rates.ADJUDICATED`; the full reasoning for each is rendered by `adjudication_doc` |
| `split` | Mechanical: did the reviewer carry the AI's *added tokens* into their final text? Agrees with the hand labels on 9 of 19 — and only 2 of SME 6's 10, where editing-instruction notation (`*remove preceding "execute"*`) and reworded-but-equivalent fixes both defeat token matching |
| `endorse` / `reject` | Count every modify one way. Useful only as bounds |

```bash
python -m reviewer_reliability.figures reviewer_rates    # adjudicated
python -m reviewer_reliability.reviewer_rates --compare  # all four side by side
```

## Naming the reviewers

The boxplot shows the spread and deliberately hides who is who. `scorecard.py` is
its counterpart — same data, same `adjudicated` policy, one row per reviewer, with
TPR / TNR / FPR / FNR / Accuracy / Precision shaded direction-aware (a low FPR
shades like a high TPR, so the eye reads good/bad rather than big/small).

Two things it makes explicit that the boxplot leaves implicit:

* **`n` is not a fixed panel.** It is however many violations the AI flagged in
  that session — 39 to 42, and every one is scored (raw record count == the app's
  own `total_violations` == the number scored, in all five sessions; nothing is
  filtered). The spread is AI non-determinism across five independent flagging
  runs, pairwise Jaccard 0.78–0.93. So each small figure in a cell rescores that
  reviewer on the **32-cell common core** flagged in every session, where everyone
  judged identical items. It moves TPR/TNR by at most 0.05 and changes no ranking.
* **The four rates are two numbers.** FNR = 1 − TPR and FPR = 1 − TNR exactly,
  each pair sharing a denominator. Only Accuracy and Precision add information,
  which is why they sit in their own block.

## Everything here

| Command | Output |
|---|---|
| `python -m reviewer_reliability.figures` | All four artefacts below |
| `python -m reviewer_reliability.figures reviewer_rates` | `docs/figures/PM_reviewer_rates_adjudicated.png` |
| `python -m reviewer_reliability.figures scorecard` | `docs/figures/PM_reviewer_scorecard.png` + `runs/PM_reviewer_scorecard.csv` |
| `python -m reviewer_reliability.adjudication_doc` | `runs/PM_modify_adjudication.txt` + `.docx` — the case for every `modify` verdict |
| `python -m reviewer_reliability.figures reliability` | `outputs/reviewer_reliability/reviewer_reliability_PM.pdf` |
| `python -m reviewer_reliability.figures consistency` | `outputs/reviewer_reliability/ai_flagging_consistency_PM.pdf` — how stably the AI flags the same cell across sessions |

Reads the six Point Mass sessions in `data/sme_feedback/` and `data/ground_truth/Point_Mass_v5_GroundTruth.xlsx`.
No API calls.

## Provenance caveat on SME 6

`d566d756` (SME 6) was added after the original five. Its ten `modify` decisions
were adjudicated **with each item's ground-truth status visible**, and came out
perfectly aligned with it — all five GT-violation modifies endorsed, all five
GT-clean modifies rejected. That is also what anchoring would produce. The
original nine show no such alignment (five of their seven GT-clean modifies were
endorsed, i.e. scored as false positives). Treat SME 6's TNR and TPR as
provisional until that block is re-labelled blind.

Also note SME 6 is a behaviourally distinct reviewer: 28% accept against 67–100%
for the others, 48% reject, and 62% of decisions carry a written note — ten of
which are complaints about the prototype itself rather than the requirements. If
you are measuring reviewing behaviour, check whether that is the same task.
