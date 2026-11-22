"""One criterion x requirement matrix per SME, in the model-matrix shape.

Why this needs three states, not two
------------------------------------
Every model matrix in this repo is two-state: "x" means flagged, blank means
not flagged, and blank is scored as a negative. That works because the model
renders a verdict on all 91 cells.

An SME does not. The review UI shows them only the violations the AI flagged
and asks Accept / Modify / Reject. On PM each reviewer saw 39-42 of the 91
cells - 43-46%. Across all six the union is 49; **42 cells were never shown to
anyone**, and 8 of the 30 ground-truth positives are among them.

So a two-state SME matrix would read "clean" on 42 cells the reviewer was never
asked about, and scoring it like a model matrix would credit them with ~42 true
negatives they never earned. Hence:

    x   endorsed  - the SME agreed this cell is a violation
    o   rejected  - the SME was shown this flag and declined it
    .   not shown - the AI never flagged it, so the SME had no opportunity

Only x and o are scoreable. "." is missing data, not a negative. Every sheet
carries its own coverage line so the denominator travels with the matrix.

`modify` follows the same policy switch as reviewer_rates (default: the
published `adjudicated` labels).

    python -m reviewer_reliability.sme_matrix
    python -m reviewer_reliability.sme_matrix --modify split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill

from shared.paths import FEEDBACK_DIR, REVIEWER_RUNS_DIR
from .reviewer_rates import (ADJUDICATED, CRITERIA, LABEL, REQ_TO_PM,
                             REVIEWERS, AFFILIATION, load_gt, modify_polarity)

ENDORSED, REJECTED, NOT_SHOWN = "x", "o", ""

_HDR = PatternFill("solid", fgColor="EFEFEF")
_ENDORSE = PatternFill("solid", fgColor="FCE4E4")   # pale red
_REJECT = PatternFill("solid", fgColor="E4EEFC")    # pale blue


def sme_decisions(rid: str, modify: str = "adjudicated") -> Dict:
    """{(criterion, req_id): 'x'|'o'} for one reviewer - only cells they saw."""
    gt = load_gt()
    data = json.loads((FEEDBACK_DIR / f"feedback_{rid}.json").read_text(encoding="utf-8"))
    out: Dict = {}
    for rq in data["requirement_feedback"]:
        req = REQ_TO_PM.get(int(rq["req_id"]))
        orig = rq.get("original_text", "")
        for v in rq.get("violation_feedback", []):
            crit = v.get("rule_id")
            if req is None or crit not in gt:
                continue
            action = v.get("user_action", "")
            if action == "modify":
                if modify == "endorse":
                    endorsed = True
                elif modify == "reject":
                    endorsed = False
                elif modify == "adjudicated":
                    lab = ADJUDICATED.get((rid, v.get("violation_id")))
                    if lab is None:
                        raise KeyError(
                            f"no human label for {rid} {v.get('violation_id')}; "
                            "add it to ADJUDICATED or use --modify split")
                    endorsed = lab == "endorse"
                else:
                    pol, _ = modify_polarity(v.get("ai_suggestion", ""),
                                             v.get("user_text", ""), orig)
                    endorsed = pol == "endorse"
            else:
                endorsed = action == "accept"
            out[(crit, req)] = ENDORSED if endorsed else REJECTED
    return out


def _write_sheet(ws, title: str, grid: Dict, req_ids, note: str) -> None:
    ws.append(["Issue ID", "Issue Description"] + list(req_ids))
    for c in ws[1]:
        c.fill = _HDR
        c.font = Font(bold=True)
        c.alignment = Alignment(horizontal="center")
    for crit in CRITERIA:
        row = [crit, title]
        for r in req_ids:
            row.append(grid.get((crit, r), NOT_SHOWN))
        ws.append(row)
    for r in ws.iter_rows(min_row=2):
        for cell in r[2:]:
            if cell.value == ENDORSED:
                cell.fill = _ENDORSE
            elif cell.value == REJECTED:
                cell.fill = _REJECT
            cell.alignment = Alignment(horizontal="center")
    ws.append([])
    ws.append([note])
    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 26


def build(modify: str = "adjudicated") -> Path:
    gt = load_gt()
    req_ids = list(REQ_TO_PM.values())
    total = len(CRITERIA) * len(req_ids)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # Ground truth first, so every SME sheet can be read against it.
    gt_grid = {(c, r): (ENDORSED if b else REJECTED)
               for c in gt for r, b in gt[c].items()}
    npos = sum(1 for v in gt_grid.values() if v == ENDORSED)
    _write_sheet(wb.create_sheet("GroundTruth"), "ground truth", gt_grid, req_ids,
                 f"Ground truth: {npos} violations of {total} cells. "
                 f"x = violation, o = clean.")

    summary = [("SME", "Affiliation", "Shown", "Coverage",
                "Endorsed", "Rejected", "TP", "FP", "FN", "TN")]
    for rid in REVIEWERS:
        d = sme_decisions(rid, modify)
        shown = len(d)
        tp = fp = fn = tn = 0
        for (crit, req), mark in d.items():
            truth = gt[crit].get(req, False)
            endorsed = mark == ENDORSED
            if endorsed and truth:
                tp += 1
            elif endorsed and not truth:
                fp += 1
            elif not endorsed and truth:
                fn += 1
            else:
                tn += 1
        note = (f"{LABEL[rid]} ({AFFILIATION[rid]}) saw {shown} of {total} cells "
                f"({shown / total:.0%}). x = endorsed the AI's flag, "
                f"o = rejected it, blank = NEVER SHOWN (the AI did not flag it, "
                f"so the reviewer had no opportunity - this is missing data, "
                f"not a negative). Scoreable cells: {shown}.")
        _write_sheet(wb.create_sheet(LABEL[rid].replace(" ", "_")),
                     LABEL[rid], d, req_ids, note)
        summary.append((LABEL[rid], AFFILIATION[rid], shown,
                        f"{shown / total:.0%}",
                        sum(1 for v in d.values() if v == ENDORSED),
                        sum(1 for v in d.values() if v == REJECTED),
                        tp, fp, fn, tn))

    ws = wb.create_sheet("Coverage", 1)
    for row in summary:
        ws.append(list(row))
    for c in ws[1]:
        c.fill = _HDR
        c.font = Font(bold=True)
    ws.append([])
    ws.append([f"Grid is {len(CRITERIA)} criteria x {len(req_ids)} requirements "
               f"= {total} cells. A reviewer is only ever shown the cells the AI "
               f"flagged, so 'Shown' is the real denominator for that reviewer - "
               f"NOT {total}."])
    ws.append([f"modify policy: {modify}"])
    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 12

    REVIEWER_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = REVIEWER_RUNS_DIR / f"PM_sme_matrices_{modify}.xlsx"
    wb.save(path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-SME decision matrices")
    ap.add_argument("--modify", default="adjudicated",
                    choices=["adjudicated", "split", "endorse", "reject"])
    args = ap.parse_args()
    path = build(args.modify)

    gt = load_gt()
    total = len(CRITERIA) * len(REQ_TO_PM)
    print(f"wrote {path}")
    print(f"  sheets: GroundTruth, Coverage, {', '.join(LABEL[r] for r in REVIEWERS)}")
    print(f"  grid  : {len(CRITERIA)} x {len(REQ_TO_PM)} = {total} cells")
    seen = [len(sme_decisions(r, args.modify)) for r in REVIEWERS]
    print(f"  shown : {min(seen)}-{max(seen)} cells per SME "
          f"({min(seen)/total:.0%}-{max(seen)/total:.0%})")


if __name__ == "__main__":
    main()
