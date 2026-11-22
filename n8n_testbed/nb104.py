"""Overall scoring on 104 cells, following AI4RE_Data_processing.ipynb (cell 1).

The notebook is the reference for every overall number:

  * ISSUE_IDS_TO_KEEP = A1, A2, A3, A4, A5, A6, A9, A10  ->  8 criteria x 13 = 104 cells
    (A1 is blank everywhere in the ground truth, so an A1 flag is a false positive)
  * summary columns FR, PR, ER, RR, ALL dropped
  * a cell is flagged only if it reads "x"
  * a criterion or column missing from a run is scored as "not flagged"
    (the SME sheets have no A1 row, so the team's A1 cells are not flagged)
  * metrics per run; F1 = 2TP / (2TP + FP + FN); NaN when a denominator is 0

Model runs come from the cleaned runs/PM_<config>_matrix.xlsx files (first 100
non-empty runs). Per-criterion F1 is unchanged by A1 and is not recomputed here.

    python -m n8n_testbed.nb104
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import openpyxl
import pandas as pd

from shared.paths import TESTBED_RUNS_DIR as RUNS_DIR

GT_PATH = RUNS_DIR.parents[1] / "data" / "ground_truth" / "Point_Mass_v5_GroundTruth.xlsx"
ISSUE_IDS = ["A1", "A2", "A3", "A4", "A5", "A6", "A9", "A10"]
SUMMARY_COLS = {"FR", "PR", "ER", "RR", "ALL"}
METRICS = ["TPR", "TNR", "FPR", "FNR", "Accuracy", "Precision", "F1"]


def _read(ws) -> Dict[str, Dict[str, bool]]:
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return {}
    cols = {i: str(h).strip() for i, h in enumerate(rows[0])
            if i >= 2 and h is not None and str(h).strip() not in SUMMARY_COLS
            and not str(h).startswith("Unnamed")}
    out = {}
    for r in rows[1:]:
        if r and r[0] is not None and str(r[0]).strip() in ISSUE_IDS:
            out[str(r[0]).strip()] = {cols[i]: str(r[i] if r[i] is not None else "").strip().lower() == "x"
                                      for i in cols}
    return out


def _div(n, d):
    return n / d if d else np.nan


def metrics_from_counts(tp, tn, fp, fn) -> Dict[str, float]:
    return {"TPR": _div(tp, tp + fn), "TNR": _div(tn, tn + fp), "FPR": _div(fp, fp + tn),
            "FNR": _div(fn, tp + fn), "Accuracy": _div(tp + tn, tp + tn + fp + fn),
            "Precision": _div(tp, tp + fp), "F1": _div(2 * tp, 2 * tp + fp + fn)}


def load_gt() -> Dict[str, Dict[str, bool]]:
    wb = openpyxl.load_workbook(GT_PATH, data_only=True)
    return _read(wb[wb.sheetnames[0]])


def score(config: str) -> pd.DataFrame:
    """One row per run: counts and the seven overall metrics on 104 cells."""
    gt = load_gt()
    wb = openpyxl.load_workbook(RUNS_DIR / f"PM_{config}_matrix.xlsx", data_only=True)
    rows = []
    for sheet in wb.sheetnames:
        pred = _read(wb[sheet])
        tp = tn = fp = fn = 0
        for c in ISSUE_IDS:
            for req, g in gt[c].items():
                p = pred.get(c, {}).get(req, False)
                tp += p and g; tn += (not p) and (not g)
                fp += p and (not g); fn += (not p) and g
        rows.append({"run": sheet, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                     "cells": tp + tn + fp + fn, **metrics_from_counts(tp, tn, fp, fn)})
    return pd.DataFrame(rows)


ARMS = {"Baseline": "beta1base", "Beta 1": "n8nsub46", "Human-AI team": "smeteam",
        "Baseline + CL": "defspluscl", "Beta 1 + CL": "cltop3"}


def main() -> None:
    tab = []
    for arm, cfg in ARMS.items():
        df = score(cfg)
        tab.append({"Configuration": arm, "runs": len(df), "cells/run": int(df.cells.iloc[0]),
                    **{m: round(float(df[m].median()), 2) for m in METRICS}})
    print(pd.DataFrame(tab).to_string(index=False))


if __name__ == "__main__":
    main()
