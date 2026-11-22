"""Mann-Whitney U tests on the AI4RE notebook's own outputs (104 cells).

Reads each configuration's confusion_detail.xlsx from "Notebook Results (104 cells)":
  * Summary      -> one value per run for TPR, TNR, FPR, FNR, Accuracy, Precision, F1
  * F1_by_issue  -> one F1 per run for each criterion A2-A10 (A1 has no ground-truth
                    violations, so it is not tested)

Per comparison: two-sided scipy.stats.mannwhitneyu (default settings), Holm correction
within each family of seven tests, rank-biserial r = 2*U_cmp/(n_ref*n_cmp) - 1, i.e.
positive r = the compared configuration scored higher. U_ref = n_ref*n_cmp - U_cmp is
the U reported in the Results Overview tables.

    python -m n8n_testbed.mwu_notebook_outputs
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

RESULTS = Path(__file__).resolve().parents[2] / "Notebook Results (104 cells)"
CONFIGS = {"Baseline": "1 Baseline (definitions only)",
           "Beta 1": "2 Beta 1 (17 manually crafted subrules)",
           "Human-AI team": "3 Human-AI team (6 SMEs)",
           "Beta 1 + CL": "4 Beta 1 + CL (rules 2,3,5)",
           "Baseline + CL": "5 Baseline + CL (rules 2,3,5)"}
COMPARISONS = [("Baseline", "Beta 1"), ("Beta 1", "Human-AI team"),
               ("Baseline", "Baseline + CL"), ("Beta 1", "Beta 1 + CL")]
METRICS = ["TPR", "TNR", "FPR", "FNR", "Accuracy", "Precision", "F1"]
CRITERIA = ["A2", "A3", "A4", "A5", "A6", "A9", "A10"]


def holm(p):
    p = np.asarray(p, float)
    order, adj, running = np.argsort(p), np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def test(ref, cmp):
    a, b = np.asarray(ref, float), np.asarray(cmp, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    u_cmp, p = mannwhitneyu(b, a, alternative="two-sided")
    n = len(a) * len(b)
    return {"n_ref": len(a), "n_cmp": len(b), "med_ref": np.median(a), "med_cmp": np.median(b),
            "U_cmp": u_cmp, "U_ref": n - u_cmp, "p": p, "r": 2 * u_cmp / n - 1}


def main() -> None:
    summ = {k: pd.read_excel(RESULTS / v / "confusion_detail.xlsx", sheet_name="Summary") for k, v in CONFIGS.items()}
    f1 = {k: pd.read_excel(RESULTS / v / "confusion_detail.xlsx", sheet_name="F1_by_issue", index_col=0)
          for k, v in CONFIGS.items()}
    rows = []
    for ref, cmp in COMPARISONS:
        fams = [[{"comparison": f"{cmp} vs {ref}", "family": "overall", "measure": m,
                  **test(summ[ref][m], summ[cmp][m])} for m in METRICS],
                [{"comparison": f"{cmp} vs {ref}", "family": "per-criterion F1", "measure": c,
                  **test(f1[ref].loc[c], f1[cmp].loc[c])} for c in CRITERIA]]
        for fam in fams:
            for row, padj in zip(fam, holm([x["p"] for x in fam])):
                row["p_holm"] = padj
                rows.append(row)
    out = RESULTS / "MWU tests (from notebook outputs, 104 cells).xlsx"
    pd.DataFrame(rows).to_excel(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
