"""Mann-Whitney U tests for RQ1 and RQ3, on the same per-run values the figures use.

  * OVERALL      - the seven pooled metrics per run (all 91 cells of a run), i.e.
                   the values behind the overall box plots (compare_boxplot).
  * PER CRITERION - F1 per criterion per run (13 cells), i.e. the values whose
                   medians are the dots in rates_by_criterion.

Two-sided Mann-Whitney U (scipy, tie-corrected normal approximation; per-criterion
F1 takes few distinct values, so ties are heavy). Effect size is the rank-biserial
correlation r = 1 - 2U/(n1 n2) with U for the first arm, so r > 0 means the second
arm tends to score higher. Holm correction within each family (overall: 7 metrics,
per criterion: 7 criteria) and per comparison.

Runs where a metric is undefined (precision with no flags) are dropped for that
metric, and the n used is reported.

    python -m n8n_testbed.mwu_tests
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from n8n_testbed import metrics
from shared.paths import output_dir

import os
# AI4RE_CELLS=104: overall metrics on the notebook's 104 cells (A1 included).
# Per-criterion F1 (A2-A10) is identical either way.
CELLS = os.getenv("AI4RE_CELLS", "91").strip()

CRITERIA = ["A2", "A3", "A4", "A5", "A6", "A9", "A10"]
OVERALL = [("recall", "TPR"), ("specificity", "TNR"), ("fpr", "FPR"), ("fnr", "FNR"),
           ("accuracy", "Accuracy"), ("precision", "Precision"), ("f1", "F1")]
ARMS = {"Baseline": "beta1base", "Beta 1": "n8nsub46", "CL + top-3": "cltop3",
        "Human-AI team": "smeteam", "DefsPlusCL": "defspluscl"}
COMPARISONS = [("RQ1", "Baseline", "Beta 1"),
               ("RQ2", "Beta 1", "Human-AI team"),
               ("RQ2", "Baseline", "Human-AI team"),
               ("RQ3", "Beta 1", "CL + top-3"),
               ("RQ3", "Baseline", "CL + top-3"),
               ("RQ3", "Baseline", "DefsPlusCL")]


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


N_PERM = 20000


def _perm_p(a, b, rng) -> float:
    """Two-sided permutation p for the difference in MEDIANS (exact-in-spirit for
    small n: no large-sample assumption). Used for RQ2, where one arm has 6."""
    obs = abs(np.median(b) - np.median(a))
    pool, n = np.concatenate([a, b]), len(a)
    hits = 0
    for _ in range(N_PERM):
        rng.shuffle(pool)
        hits += abs(np.median(pool[n:]) - np.median(pool[:n])) >= obs - 1e-12
    return (hits + 1) / (N_PERM + 1)


def _test(a, b, rng=None) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    out = {"n1": len(a), "n2": len(b), "median1": np.median(a), "median2": np.median(b),
           "U": u, "p": p, "r_rb": 1 - 2 * u / (len(a) * len(b))}
    if rng is not None:
        out["p_perm_median"] = _perm_p(a, b, rng)
    return out


def _magnitude(r: float) -> str:
    r = abs(r)
    return "large" if r >= 0.5 else "medium" if r >= 0.3 else "small" if r >= 0.1 else "negligible"


def run() -> pd.DataFrame:
    scored = {arm: metrics.score("PM", cfg) for arm, cfg in ARMS.items()}
    if CELLS == "104":
        from n8n_testbed import nb104
        nb = {arm: nb104.score(cfg) for arm, cfg in ARMS.items()}
    rows = []
    rng = np.random.default_rng(0)
    for rq, a, b in COMPARISONS:
        small = min(len(scored[a].execution.unique()), len(scored[b].execution.unique())) < 20
        g = rng if small else None
        da, db = scored[a], scored[b]
        fam = []
        for key, lab in OVERALL:
            if CELLS == "104":      # notebook scoring, A1 included (n8n_testbed.nb104)
                va, vb = nb[a][lab].astype(float), nb[b][lab].astype(float)
            else:
                va = da[da.rule == "POOLED"][key].astype(float)
                vb = db[db.rule == "POOLED"][key].astype(float)
            fam.append({"RQ": rq, "comparison": f"{b} vs {a}", "family": "overall",
                        "measure": lab, **_test(va, vb, g)})
        crit = []
        for c in CRITERIA:
            va = da[da.rule == c]["f1"].astype(float)
            vb = db[db.rule == c]["f1"].astype(float)
            crit.append({"RQ": rq, "comparison": f"{b} vs {a}", "family": "per-criterion F1",
                         "measure": c, **_test(va, vb, g)})
        for group in (fam, crit):
            for row, padj in zip(group, holm([g["p"] for g in group])):
                row["p_holm"] = padj
                row["effect"] = _magnitude(row["r_rb"])
            rows.extend(group)
    return pd.DataFrame(rows)


def main() -> None:
    df = run()
    out = output_dir("criterion_compare") / f"PM_mwu_RQ1_RQ2_RQ3_{CELLS}cells.xlsx"
    df.round(4).to_excel(out, index=False)
    pd.set_option("display.width", 200)
    for (comp, fam), g in df.groupby(["comparison", "family"], sort=False):
        print(f"\n{comp}  -  {fam}")
        print(g.assign(med=g.median1.map("{:.2f}".format) + " -> " + g.median2.map("{:.2f}".format),
                       p=g.p.map("{:.1e}".format), p_holm=g.p_holm.map("{:.1e}".format),
                       r=g.r_rb.map("{:+.2f}".format))
              [["measure", "n1", "n2", "med", "U", "p", "p_holm", "r", "effect"]
               + (["p_perm_median"] if g.get("p_perm_median") is not None and g["p_perm_median"].notna().any() else [])]
              .to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
