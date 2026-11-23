# Thesis evaluation runs (Point Mass, 100 runs per configuration)

The four 100-run workbooks behind the Chapter 3 results (Tables 1-3 and 10-13).
Each tab is one n8n execution in matrix form: criteria A1-A11 by requirement, `x` = flagged.
Scored against `../ground_truth/Point_Mass_v5_GroundTruth.xlsx`.

| Thesis configuration | File | Testbed config | Raw n8n export | Executions used |
|---|---|---|---|---|
| Baseline | `PM_beta1base_matrix.xlsx` | `beta1base` | `Beta 1_Manually Crafted Subrules_Benchmark.xlsx` (name is misleading: no sub-rules) | first 100 non-empty tabs (Execution_19 onward), renumbered Execution_01-100 |
| Beta 1 | `PM_n8nsub46_matrix.xlsx` | `n8nsub46` | `GR_matrix_subrules_Sonnet 4.6.xlsx` | Execution_785-887 |
| Baseline + CL (rules 2, 3, 5) | `PM_defspluscl_matrix.xlsx` | `defspluscl` | `DefsPlusCLSubrules.xlsx` | Execution_145-244 |
| Beta 1 + CL (rules 2, 3, 5) | `PM_cltop3_matrix.xlsx` | `cltop3` | `CLPlusTop3SMESubrules.xlsx` | Execution_1-100 |

These are byte-identical copies of the files in `n8n_testbed/runs/`, which the scoring
scripts read (`n8n_testbed/nb104.py`, `mwu_tests.py`, `criterion_compare.py`).

Notes
- Baseline Execution_19 (tab Execution_01 here) came from the earlier two-part prompt
  (set-level evaluation included); excluding it leaves every reported median and test unchanged.
- Prompt provenance and differences between configurations: see `../../docs/ARMS.md`.
