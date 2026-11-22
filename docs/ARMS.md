# The four PM arms — what actually differs

All four evaluate the same 13 Point Mass requirements against the same ground
truth (`data/ground_truth/Point_Mass_v5_GroundTruth.xlsx`, A2–A10 × 13 = 91
cells, 30 violations), 100 runs each, and are scored by the same KPI engine.
They differ on exactly two things: **whether the prompt carries INCOSE
sub-rules**, and **whether the 5 SME-induced rules are injected**.

| Arm | Sub-rules | SME rules | Matrix |
|---|---|---|---|
| **Baseline_NoSubrules** | — | — | `PM_beta1base_matrix.xlsx` |
| **Baseline_SMESubrules** | — | 5 | `PM_beta1sme_matrix.xlsx` |
| **Beta1** | 17 | — | `PM_n8nsub46_matrix.xlsx` |
| **Beta1 w SME Subrules** | 17 | 5 | `PM_aigenv2sme_matrix.xlsx` |

That is a clean 2×2: rubric depth × SME injection.

---

## 1. Baseline_NoSubrules

**Prompt:** `n8n_prompt_individual_only.txt` (from the `AI4RE benchmark with
loop version 2 GR` workflow)

Criterion descriptions only. No sub-rules anywhere. Its entire A10 is:

> A10-Conforming: The requirement statement conforms to an approved standard
> pattern and style guide or standard for writing and managing requirements.

**This criterion is unactionable as written** — it asks the model to check
conformance to a style guide that is never supplied. Verified: the prompt never
contains `shall be able to`, `no more than`, `no less than`, `positive
statement`, `imperative verb`, or `typo`.

The consequence dominates every number this arm produces: **106 of 117 runs flag
A10 zero or one time**, against 10 ground-truth positives. A10 TPR = 0.089.

> The label "Manually Crafted Subrules" on the source workbook is wrong. There
> are no sub-rules in it. Treat this arm as *no rubric detail*, not as *manually
> authored rubric detail*.

## 2. Baseline_SMESubrules

**Prompt:** `n8n_prompt_individual_icaiv2.txt`

Arm 1 plus the 5 SME rules from `rules_group_icai_v2_studio.json`, injected
under their criterion using the repo convention
(`inject.py:97`, `- [Expert-reviewed guidance from prior reviews] …`):

- **A10** ×4 — quantified limits over qualitative language; active voice over
  `shall be able to`; quantified limits over vague negatives; flag modal weakness
- **A5** ×1 — do not replace concise language with longer phrasings

Nothing else changes. Because arm 1's A10 was empty, these four rules are the
*only* actionable A10 guidance in the prompt — which is why A10 TPR jumps
0.089 → 0.874.

## 3. Beta1

**Prompt:** the `AI4RE GR_subrules` workflow's system message — **17 sub-rules**
(A2.1–A2.2, A3.1–A3.3, A4.1–A4.3, A5.1, A6.1, A9.1–A9.3, A10.1–A10.4).

A10.3 explicitly bans `shall be able to`, `no more than`, `no less than`. A10
detection is therefore near ceiling *before* any SME rule is added: TPR 0.822.

> **Attribution note.** This arm is **not** `n8n_prompt_incose_app_rubric.txt`.
> `GR_matrix_subrules_Sonnet 4.6.xlsx` is dated 2026-08-27 21:42; that prompt
> file was written 2026-08-29 11:28, two days later. The two rubrics also differ:
> see *Rubric wording* below.

## 4. Beta1 w SME Subrules

**Prompt:** arm 3's rubric plus the same 5 SME rules.

> **Needs confirmation.** Two files match this description and both predate the
> run (`AIGenSubruleswSMESubrulesV2.xlsx`, 2026-08-30 14:17):
> `n8n_prompt_GR_subrules_icaiv2.txt` (n8n wording) and
> `n8n_prompt_incose_app_rubric_icaiv2.txt` (INCOSE-app wording). They are not
> equivalent — see below. Which one was run determines whether this arm is a
> matched pair with Beta1.

---

## Rubric wording: two variants of the same 17 sub-rules

Both rubrics carry the same 17 sub-rule IDs. **8 are identical, 9 differ**, and
three of those differences change meaning:

| | n8n wording (Beta1) | INCOSE-app wording |
|---|---|---|
| **A4.3** | "…only one interpretation. **It is acceptable if the terms are defined in a note, or elsewhere in the requirement document such as a glossary, nomenclature, or appendix.**" | "…only one interpretation." |
| **A9.2** | "…ConOps, higher-level requirement, **or the provided source document. Any imposed characteristic can be justified by supporting engineering or problem formulation work…**" | "…ConOps, or higher-level requirement." |
| **A9.3** | "…**or any other provided source document.**" | "…or higher-level requirement." |

The INCOSE-app variant is **stricter on A4** (no glossary allowance) and
**narrower on A9** (fewer valid trace targets). Any A4/A9 difference between arms
using different variants is definitional, not model behaviour.

The other six differences are cosmetic (quote style, "and not" vs "not").

---

## Results

100 runs each; medians shown, means in `outputs/fourarm_summary.csv`.

| Arm | TPR | TNR | FPR | FNR | Acc | Prec | F1 |
|---|---|---|---|---|---|---|---|
| Baseline_NoSubrules | 0.20 | 0.77 | 0.23 | 0.80 | 0.59 | 0.32 | 0.26 |
| Baseline_SMESubrules | 0.40 | **0.89** | **0.11** | 0.60 | 0.73 | 0.64 | 0.50 |
| **Beta1** | 0.60 | 0.84 | 0.16 | 0.40 | **0.76** | 0.64 | **0.62** |
| Beta1 w SME Subrules | **0.63** | 0.75 | 0.25 | **0.37** | 0.71 | 0.56 | 0.60 |

**Figures:** `docs/figures/PM_fourarm_named.png`,
`PM_beta1base_vs_beta1sme_overall_box.png`,
`PM_n8nsub46_vs_aigenv2sme_overall_box.png`

### What the 2×2 shows

**Sub-rules are worth more than SME rules.** Adding the 17 sub-rules
(arm 1 → 3) moves F1 0.26 → 0.62. Adding the 5 SME rules to the bare prompt
(arm 1 → 2) moves it 0.26 → 0.50. Adding them on top of the sub-rules
(arm 3 → 4) moves it 0.62 → 0.60 — no gain (p = 0.56).

**The SME rules invert direction depending on the base.** On the bare prompt
they make the model *more conservative* (FPR 0.23 → 0.11). On the sub-ruled
prompt they make it *more aggressive* (0.16 → 0.25). Same five rules. This is
the interaction effect, and it is the most substantive finding in the set.

**Why:** 2 of the 5 rules restate A10.3 (`shall be able to`) and add nothing
where A10.3 already exists. The other 2 are about measurable limits — A4.1
material filed under A10 (see `icai_v3/classify.py`) — and they raise A4 recall
(0.555 → 0.723) while nearly doubling A4 false positives (2.99 → 4.92/run).

**Neither arm detects A3.** TPR 0.010 (Beta1) and 0.035 (Beta1 w SME) against 2
ground-truth positives. No rubric or rule set in play addresses it.

---

## Caveats

- **Baseline_NoSubrules had 10 empty `Execution_` sheets** in its source
  workbook. Empty sheets score as "found no issues"; they were excluded, and the
  first 100 non-empty were used.
- **Model is not held constant.** Beta1 ran on `claude-sonnet-4-6`; the
  `AI4RE GR_subrules_GPT 5.5` workflow variant runs `openai/gpt-5.5`. Confirm
  which produced arm 4 before treating its comparison against Beta1 as a
  rule-injection effect.
- **A2/A3/A5/A6 have only 1–2 ground-truth positives each**, so their
  per-criterion rates are coarse. Use counts
  (`docs/figures/by_criterion/`), not rates, for those.
