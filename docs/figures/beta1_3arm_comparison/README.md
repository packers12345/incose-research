# Beta 1 subrule arms — per-criterion comparison

Three arms, PM dataset, 100 runs each, scored against `PM_gt.xlsx` (identical to
`data/ground_truth/Point_Mass_v5_GroundTruth.xlsx` — verified cell by cell).

| Colour | Figure label | Source workbook | Testbed config |
|---|---|---|---|
| Red | Beta 1 | `GR_matrix_subrules_Sonnet 4.6.xlsx` | `n8nsub46` |
| Blue | Beta 1 + SME subrules | `Beta1_ManuallyCraftedSubrules_SMESubrules.xlsx` | `beta1msme` |
| Gold | Beta 1 + deconflicted SME subrules | `SMEPrioritySubrulesV2.xlsx` | `smepriority` |

The gold arm is the one built from `n8n_prompt_GR_subrules_icaiv2_smepriority.txt`
— the GR subrules prompt with A5.1 and A10.2 deleted, resolving the two places
where a codified subrule conflicted with an SME-derived rule.

## Figures

- `accuracy_by_criterion.png` — (TP+TN) / all 13 requirements
- `precision_by_criterion.png` — TP / (TP+FP)
- `f1_by_criterion.png` — harmonic mean of precision and recall

One line per criterion, one dot per arm at that arm's median across runs. Arms
that tie are drawn as concentric rings. Spread is not shown — see the pooled
boxplots in `../` for that.

## Reading them

Row labels carry each criterion's ground-truth split, and it matters:

- **A2 and A3 have 1 and 2 ground-truth positives.** Their rates move in steps
  of 0.5 or 1.0. Treat those rows as ladder artefacts, not measurements.
- **Accuracy is inflated by true negatives on the sparse criteria** — an arm
  that never flags A2 still scores ~0.92 there.
- **Precision alone flatters the blue arm.** It hits 1.00 on A9 and A10 and is
  `n/a` on A2 (never flagged it in any run), while its A9 F1 is 0.20. High
  precision here means "rarely flags", not "flags well".
- **A9 carries 9 of the 30 ground-truth positives**, so it drives the pooled
  numbers more than any other row. It is also the widest spread on these
  charts: accuracy 0.38 / 0.54 / 0.62.
- **A3 is 0.00 F1 in all three arms.** No configuration detects it.

## Provenance caveat

`n8nsub46` ran on `claude-sonnet-4-6` at temperature 0.7. The model and
temperature behind the other two workbooks are not recorded here. If they
differ, part of any gap is a model effect rather than a subrule effect — the
confound already flagged in `docs/ARMS.md`.

`beta1msme` is the first 100 of 107 filled sheets (`Execution_237`–`336`);
`smepriority` is the first 100 of 132 filled (`Execution_583`–`682`). Empty
sheets were excluded before scoring, following the ARMS.md precedent.

Note: `beta1msme` is NOT the same run as the older `beta1sme` config — zero
sheets in common, checked directly. Figures built from `beta1sme` are a
different comparison.

## Regenerating

```
python -m n8n_testbed.plots.rates_by_criterion \
    --configs n8nsub46 beta1msme smepriority \
    --labels "Beta 1" "Beta 1 + SME subrules" "Beta 1 + deconflicted SME subrules"
```

That writes long filenames into `outputs/n8n_testbed/plots/by_criterion/`; the
copies here are renamed to short names.

## Summary table

Medians across 100 runs. B1 = Beta 1, +SME = Beta 1 + SME subrules,
+DC = Beta 1 + deconflicted SME subrules. Also in `summary_table.csv`.

| Criterion | GT +/- | Acc B1 | Acc +SME | Acc +DC | Prec B1 | Prec +SME | Prec +DC | F1 B1 | F1 +SME | F1 +DC |
|---|---|---|---|---|---|---|---|---|---|---|
| A2 Necessary | 1+ / 12- | 0.92 | 0.92 | 0.92 | 0.50 | n/a | 0.50 | 0.67 | 0.00 | 0.67 |
| A3 Appropriate | 2+ / 11- | 0.77 | 0.85 | 0.85 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| A4 Unambiguous | 4+ / 9- | 0.62 | 0.54 | 0.62 | 0.43 | 0.29 | 0.43 | 0.44 | 0.33 | **0.55** |
| A5 Complete | 2+ / 11- | 0.85 | 0.85 | **0.92** | 0.50 | 0.50 | 0.67 | 0.57 | 0.50 | **0.67** |
| A6 Singular | 2+ / 11- | 0.77 | 0.69 | 0.77 | 0.25 | 0.00 | 0.25 | 0.33 | 0.00 | 0.33 |
| A9 Correct | 9+ / 4- | 0.54 | **0.38** | **0.62** | 0.89 | 1.00 | 1.00 | 0.57 | **0.20** | **0.62** |
| A10 Conforming | 10+ / 3- | 0.77 | **0.92** | 0.77 | 0.84 | 1.00 | 0.82 | 0.82 | **0.95** | 0.86 |
| **POOLED** | 30+ / 61- | **0.76** | 0.73 | 0.75 | 0.64 | 0.64 | 0.62 | **0.62** | 0.50 | **0.62** |

`n/a` = the arm never flagged that criterion in any run, so precision is
undefined. POOLED is micro-averaged over all 91 cells, not the mean of the
rows above it — criteria are weighted by how many cells they carry, so A9 and
A10 (19 of the 30 positives) dominate it.

Reading the pooled row against the per-criterion rows is the point of this
table. +DC wins or ties Beta 1 on every per-criterion accuracy row and leads
F1 on three, yet lands at pooled parity (F1 0.62 both, accuracy 0.75 vs 0.76),
because its gains sit on the criteria carrying the fewest positives. +SME is
the opposite: it takes A10 by +0.13 F1 and loses everywhere else, including
-0.37 on A9, which is why its pooled F1 falls to 0.50.
