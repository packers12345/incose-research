"""Per-criterion comparison of the five PM arms, scored one consistent way.

The arms
--------
    Baseline       criterion definitions only, no sub-rules   (PM_beta1base)
    Beta1          17 INCOSE sub-rules                         (PM_n8nsub46)
    DefsPlusCL     definitions + CL-induced sub-rules          (DefsPlusCLSubrules.xlsx)
    CLPlusTop3     CL sub-rules + top-3 SME sub-rules          (PM_cltop3)
    Human-AI team  prototype flags, filtered by an SME         (6 adjudicated SME sessions)

plus one reference that is not an arm: "Prototype (shown to SMEs)", the flags the
AI put in front of each reviewer. Team minus Prototype isolates what the reviewers
changed, since the team's AI front end was the prototype, not any arm above.

What this fixes relative to AI4RE_Data_processing.ipynb
-------------------------------------------------------
1. Empty tabs. The notebook scores the first N tabs, and an empty tab scores as
   "found no issues". Here empty sheets are skipped and the first 100 NON-empty
   runs are used (the Baseline source workbook has 10 empty tabs in its first 100).
2. A1 is not scored. Ground-truth A1 is blank everywhere, the SME sheets have no
   A1 row, and the testbed / ARMS.md universe is A2-A10 x 13 = 91 cells.
3. Small criteria. A2/A3/A5/A6 have 1-2 positives, so per-run rates are 0/0.5/1
   and precision is undefined whenever nothing is flagged. The headline numbers
   here are POOLED: TP/FP/FN/TN summed over runs, then the rate, with a bootstrap
   95% CI that resamples runs (sessions for the team). Per-run means/medians in the
   notebook's style are kept on their own sheet.

The Human-AI team is scored as a SYSTEM over all 91 cells: an SME-endorsed flag
(x) is a positive; a rejected flag (o) and a cell never shown (blank) are both
negatives - that is what the team would hand back. This differs from
reviewer_reliability, which scores reviewers on shown cells only; that is a
reviewer-skill question, not an arm comparison.

    python -m n8n_testbed.criterion_compare
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import openpyxl
import pandas as pd

from shared.paths import TESTBED_RUNS_DIR as RUNS_DIR, output_dir
from reviewer_reliability.reviewer_rates import REVIEWERS, LABEL, REQ_TO_PM
from reviewer_reliability.sme_matrix import sme_decisions, ENDORSED

ROOT = RUNS_DIR.parents[2]          # the folder holding the n8n export workbooks
GT_PATH = RUNS_DIR.parents[1] / "data" / "ground_truth" / "Point_Mass_v5_GroundTruth.xlsx"

CRITERIA = ["A2", "A3", "A4", "A5", "A6", "A9", "A10"]
CRITERION_NAME = {"A2": "Necessary", "A3": "Appropriate", "A4": "Unambiguous",
                  "A5": "Complete", "A6": "Singular", "A9": "Correct",
                  "A10": "Conforming"}
REQ_COL = re.compile(r"^[A-Z]{2}\.\d+$")
N_RUNS = 100
N_BOOT = 2000
SME6 = "d566d756"                   # adjudicated with GT visible - see README

MODEL_ARMS = {
    "Baseline":   RUNS_DIR / "PM_beta1base_matrix.xlsx",
    "Beta1":      RUNS_DIR / "PM_n8nsub46_matrix.xlsx",
    "DefsPlusCL": ROOT / "DefsPlusCLSubrules.xlsx",
    "CLPlusTop3": RUNS_DIR / "PM_cltop3_matrix.xlsx",
}
TEAM = "Human-AI team"
PROTO = "Prototype (shown to SMEs)"
ARM_ORDER = list(MODEL_ARMS) + [TEAM, PROTO]

Grid = Dict[str, Dict[str, bool]]   # {criterion: {req: flagged}}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _read_sheet(ws) -> Grid | None:
    """A matrix sheet as {criterion: {req: flagged}}, or None if it is empty."""
    rows = list(ws.iter_rows(values_only=True))
    if len(rows) < 3:
        return None
    # Requirement columns only: the n8n exports and the v5 GT also carry set-level
    # summary columns (FR, PR, ER, RR, ALL), which would score as 5 extra cells.
    cols = {i: str(h).strip() for i, h in enumerate(rows[0])
            if i >= 2 and h and REQ_COL.match(str(h).strip())}
    grid: Grid = {}
    for r in rows[1:]:
        if r and r[0] and str(r[0]).strip() in CRITERIA:
            grid[str(r[0]).strip()] = {cols[i]: str(r[i] or "").strip().lower() == "x"
                                       for i in cols}
    return grid or None


def load_runs(path: Path, n: int = N_RUNS) -> List[tuple]:
    """First `n` non-empty runs, in workbook tab order: [(sheet, grid), ...]."""
    wb = openpyxl.load_workbook(path, data_only=True)
    runs = []
    for name in wb.sheetnames:
        g = _read_sheet(wb[name])
        if g is not None:
            runs.append((name, g))
        if len(runs) == n:
            break
    return runs


def load_gt() -> Grid:
    wb = openpyxl.load_workbook(GT_PATH, data_only=True)
    gt = _read_sheet(wb[wb.sheetnames[0]])
    reqs = set(REQ_TO_PM.values())
    for c in CRITERIA:
        if set(gt[c]) != reqs:
            raise ValueError(f"GT {c} columns {sorted(gt[c])} != the 13 PM requirements")
    return gt


def team_runs(modify: str = "adjudicated", drop: tuple = ()) -> tuple:
    """(team, prototype) run lists from the SME feedback, one run per session.

    Built from the same `sme_decisions` that writes PM_sme_matrices_*.xlsx, so the
    modify policy can be varied for the sensitivity sheet.
    """
    reqs = list(REQ_TO_PM.values())
    team, proto = [], []
    for rid in REVIEWERS:
        if rid in drop:
            continue
        d = sme_decisions(rid, modify)
        t = {c: {r: d.get((c, r)) == ENDORSED for r in reqs} for c in CRITERIA}
        p = {c: {r: (c, r) in d for r in reqs} for c in CRITERIA}
        team.append((LABEL[rid], t))
        proto.append((LABEL[rid], p))
    return team, proto


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def counts(runs: List[tuple], gt: Grid) -> np.ndarray:
    """Array [run, criterion, (tp, fp, fn, tn)]."""
    out = np.zeros((len(runs), len(CRITERIA), 4), dtype=int)
    for i, (_, g) in enumerate(runs):
        for j, c in enumerate(CRITERIA):
            for req, truth in gt[c].items():
                p = g.get(c, {}).get(req, False)
                out[i, j] += (p and truth, p and not truth,
                              (not p) and truth, (not p) and (not truth))
    return out


def _div(n, d):
    n, d = np.asarray(n, float), np.asarray(d, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d == 0, np.nan, n / d)


def rates(tp, fp, fn, tn) -> Dict[str, np.ndarray]:
    """Notebook definitions; F1 = 2TP/(2TP+FP+FN)."""
    return {"TPR": _div(tp, tp + fn), "FPR": _div(fp, fp + tn),
            "Precision": _div(tp, tp + fp), "F1": _div(2 * tp, 2 * tp + fp + fn),
            "Accuracy": _div(tp + tn, tp + fp + fn + tn), "TNR": _div(tn, tn + fp)}


METRICS = ["TPR", "FPR", "Precision", "F1", "Accuracy"]


def _with_overall(c: np.ndarray) -> np.ndarray:
    """Append an 'Overall' pseudo-criterion: all seven criteria pooled."""
    return np.concatenate([c, c.sum(axis=1, keepdims=True)], axis=1)


def pooled(c: np.ndarray) -> Dict[str, np.ndarray]:
    """Pooled rate per criterion (+Overall): sum counts over runs, then divide."""
    s = _with_overall(c).sum(axis=0)
    return rates(*s.T)


def boot_samples(c: np.ndarray, rng, n_boot: int = N_BOOT) -> Dict[str, np.ndarray]:
    """Pooled rates over `n_boot` resamples of runs: {metric: [boot, criterion]}."""
    c = _with_overall(c)
    idx = rng.integers(0, len(c), size=(n_boot, len(c)))
    s = c[idx].sum(axis=1)                       # [boot, crit, 4]
    return rates(s[..., 0], s[..., 1], s[..., 2], s[..., 3])


def ci(samples: np.ndarray):
    return np.nanpercentile(samples, 2.5, axis=0), np.nanpercentile(samples, 97.5, axis=0)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

LABELS = CRITERIA + ["Overall"]


def _fmt(v, lo, hi):
    if np.isnan(v):
        return "n/a"
    return f"{v:.2f} ({lo:.2f}-{hi:.2f})" if not np.isnan(lo) else f"{v:.2f}"


def build(n_boot: int = N_BOOT, seed: int = 0) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    gt = load_gt()
    gt_pos = {c: sum(gt[c].values()) for c in CRITERIA}
    gt_pos["Overall"] = sum(gt_pos.values())

    runs = {arm: load_runs(p) for arm, p in MODEL_ARMS.items()}
    runs[TEAM], runs[PROTO] = team_runs()
    C = {arm: counts(r, gt) for arm, r in runs.items()}
    P = {arm: pooled(c) for arm, c in C.items()}
    B = {arm: boot_samples(c, rng, n_boot) for arm, c in C.items()}

    # -- long pooled table ------------------------------------------------- #
    long_rows = []
    for arm in ARM_ORDER:
        c = _with_overall(C[arm])
        n = len(c)
        for j, lab in enumerate(LABELS):
            tp, fp, fn, tn = c[:, j].sum(axis=0)
            row = {"Arm": arm, "Criterion": lab,
                   "Name": CRITERION_NAME.get(lab, "all seven"),
                   "GT positives": gt_pos[lab],
                   "Runs": n, "Flags/run": round((tp + fp) / n, 2),
                   "TP/run": round(tp / n, 2), "FP/run": round(fp / n, 2),
                   "FN/run": round(fn / n, 2), "TP": tp, "FP": fp, "FN": fn, "TN": tn}
            for m in METRICS:
                lo, hi = ci(B[arm][m])
                row[m] = round(float(P[arm][m][j]), 3) if not np.isnan(P[arm][m][j]) else None
                row[f"{m} CI low"] = round(float(lo[j]), 3) if not np.isnan(lo[j]) else None
                row[f"{m} CI high"] = round(float(hi[j]), 3) if not np.isnan(hi[j]) else None
            long_rows.append(row)
    long = pd.DataFrame(long_rows)

    # -- wide, one sheet per metric --------------------------------------- #
    wide = {}
    for m in METRICS:
        tbl = {"Criterion": [f"{l} {CRITERION_NAME.get(l, '')}".strip() for l in LABELS],
               "GT positives": [gt_pos[l] for l in LABELS]}
        for arm in ARM_ORDER:
            lo, hi = ci(B[arm][m])
            tbl[arm] = [_fmt(P[arm][m][j], lo[j], hi[j]) for j in range(len(LABELS))]
        wide[m] = pd.DataFrame(tbl)

    # -- deltas vs Baseline (independent resampling) ----------------------- #
    drows = []
    for arm in ARM_ORDER[1:]:
        if arm == PROTO:
            continue
        for m in ("TPR", "FPR", "Precision", "F1"):
            d = P[arm][m] - P["Baseline"][m]
            lo, hi = ci(B[arm][m] - B["Baseline"][m])
            for j, lab in enumerate(LABELS):
                sig = (not np.isnan(lo[j])) and (lo[j] > 0 or hi[j] < 0)
                drows.append({"Arm": arm, "Metric": m, "Criterion": lab,
                              "Baseline": _r(P["Baseline"][m][j]), "Arm value": _r(P[arm][m][j]),
                              "Delta": _r(d[j]), "CI low": _r(lo[j]), "CI high": _r(hi[j]),
                              "CI excludes 0": "yes" if sig else ""})
    deltas = pd.DataFrame(drows)

    # -- team vs its own prototype flags (paired by session) -------------- #
    ct, cp = _with_overall(C[TEAM]), _with_overall(C[PROTO])
    idx = rng.integers(0, len(ct), size=(n_boot, len(ct)))
    bt = rates(*np.moveaxis(ct[idx].sum(axis=1), -1, 0))
    bp = rates(*np.moveaxis(cp[idx].sum(axis=1), -1, 0))
    prow = []
    for m in ("TPR", "FPR", "Precision", "F1"):
        lo, hi = ci(bt[m] - bp[m])
        for j, lab in enumerate(LABELS):
            tp_t, fp_t = ct[:, j, 0].sum(), ct[:, j, 1].sum()
            tp_p, fp_p = cp[:, j, 0].sum(), cp[:, j, 1].sum()
            prow.append({"Metric": m, "Criterion": lab,
                         "Prototype": _r(P[PROTO][m][j]), "Team": _r(P[TEAM][m][j]),
                         "Delta": _r(P[TEAM][m][j] - P[PROTO][m][j]),
                         "CI low": _r(lo[j]), "CI high": _r(hi[j]),
                         "True flags SMEs removed": tp_p - tp_t,
                         "False flags SMEs removed": fp_p - fp_t})
    team_vs_proto = pd.DataFrame(prow)

    # -- team sensitivity -------------------------------------------------- #
    srows = []
    for label, kw in [("adjudicated (published)", {}),
                      ("modify = all endorse", {"modify": "endorse"}),
                      ("modify = all reject", {"modify": "reject"}),
                      ("adjudicated, without SME 6", {"drop": (SME6,)})]:
        t, _ = team_runs(**kw)
        pr = pooled(counts(t, gt))
        for j, lab in enumerate(LABELS):
            srows.append({"Variant": label, "Sessions": len(t), "Criterion": lab,
                          **{m: _r(pr[m][j]) for m in ("TPR", "FPR", "Precision", "F1")}})
    sens = pd.DataFrame(srows)

    # -- notebook-style per-run summary + per-run counts ------------------- #
    prr, cnt = [], []
    for arm in ARM_ORDER:
        c = _with_overall(C[arm])
        r = rates(c[..., 0], c[..., 1], c[..., 2], c[..., 3])   # [run, crit]
        for j, lab in enumerate(LABELS):
            row = {"Arm": arm, "Criterion": lab}
            for m in METRICS:
                v = r[m][:, j]
                row[f"{m} mean"] = _r(np.nanmean(v)) if np.isfinite(v).any() else None
                row[f"{m} median"] = _r(np.nanmedian(v)) if np.isfinite(v).any() else None
                row[f"{m} runs defined"] = int(np.isfinite(v).sum())
            prr.append(row)
        for i, (sheet, _) in enumerate(runs[arm]):
            for j, lab in enumerate(CRITERIA):
                tp, fp, fn, tn = C[arm][i, j]
                cnt.append({"Arm": arm, "Run": sheet, "Criterion": lab,
                            "TP": tp, "FP": fp, "FN": fn, "TN": tn})

    return {"pooled_long": long, "wide": wide, "deltas": deltas,
            "team_vs_proto": team_vs_proto, "sensitivity": sens,
            "per_run_summary": pd.DataFrame(prr), "per_run_counts": pd.DataFrame(cnt),
            "runs": {a: len(r) for a, r in runs.items()},
            "sources": {**{a: str(p) for a, p in MODEL_ARMS.items()},
                        TEAM: "data/sme_feedback (adjudicated modify labels)",
                        PROTO: "data/sme_feedback (every flag shown, x or o)"},
            "first_last": {a: (r[0][0], r[-1][0]) for a, r in runs.items()}}


def _r(v):
    v = float(v)
    return None if np.isnan(v) else round(v, 3)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

README = """Per-criterion comparison of the PM arms - how to read this workbook

Universe: Point Mass, 7 criteria (A2-A10) x 13 requirements = 91 cells; ground truth
Point_Mass_v5_GroundTruth.xlsx, 30 violations. A1 is not scored.

Headline numbers are POOLED: TP/FP/FN/TN summed over every run of an arm, then
TPR = TP/(TP+FN), FPR = FP/(FP+TN), Precision = TP/(TP+FP), F1 = 2TP/(2TP+FP+FN).
Brackets are bootstrap 95% CIs (2000 resamples of runs; of sessions for the team).
Pooling is used because A2/A3/A5/A6 have only 1-2 true violations, so a single run's
recall can only be 0, 0.5 or 1 and precision is undefined when nothing is flagged.
Notebook-style per-run means/medians are on the PerRun_summary sheet.

Model arms: first 100 NON-empty runs of each workbook, in tab order. Empty tabs are
skipped (the notebook scores them as 'found nothing').

Human-AI team: 6 SME sessions, modify decisions coded with the hand-adjudicated
labels. Scored as a system over all 91 cells: endorsed flag = positive; rejected flag
and never-shown cell = negative. 'Prototype (shown to SMEs)' is every flag the AI put
in front of the reviewer, endorsed or not - the team's own AI. It is a reference,
not an arm. Team minus Prototype = what the reviewers changed; reviewers can remove
flags but never add them, so team TPR can never exceed prototype TPR.

Cautions
- Team n = 6 sessions vs 100 runs: its CIs are wide; plot as points, not boxes.
- SME 6 (d566d756) was adjudicated with ground truth visible; see Team_sensitivity.
- SME 1 endorsed every flag, so its team matrix equals its prototype matrix.
- The team's AI front end is the prototype app, not any of the four model arms.
"""


def write_workbook(res: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readme = pd.DataFrame({"": README.splitlines()})
    src = pd.DataFrame([{"Arm": a, "Runs scored": res["runs"][a],
                         "First run": res["first_last"][a][0],
                         "Last run": res["first_last"][a][1],
                         "Source": res["sources"][a]} for a in ARM_ORDER])
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False, header=False)
        src.to_excel(xw, sheet_name="Sources", index=False)
        for m, df in res["wide"].items():
            df.to_excel(xw, sheet_name=m, index=False)
        res["pooled_long"].to_excel(xw, sheet_name="Pooled_by_criterion", index=False)
        res["deltas"].to_excel(xw, sheet_name="Delta_vs_Baseline", index=False)
        res["team_vs_proto"].to_excel(xw, sheet_name="Team_vs_Prototype", index=False)
        res["sensitivity"].to_excel(xw, sheet_name="Team_sensitivity", index=False)
        res["per_run_summary"].to_excel(xw, sheet_name="PerRun_summary", index=False)
        res["per_run_counts"].to_excel(xw, sheet_name="PerRun_counts", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2" if ws.title not in ("README",) else None
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col[:60])
                ws.column_dimensions[col[0].column_letter].width = min(max(9, width + 2), 60 if ws.title == "README" else 28)


MODEL_COLOR, TEAM_COLOR = "#2a78d6", "#eb6834"     # validated pair (dataviz palette slots 1-2)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot(res: Dict, path: Path) -> None:
    """Small multiples: metric rows x criterion columns; arms as labelled rows.

    Arm identity is carried by the row label, so colour only separates the model
    arms (blue) from the team (orange); the prototype reference is a hollow marker.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    long = res["pooled_long"]
    metrics = [("TPR", "Recall (TPR)  ↑"), ("FPR", "False-positive rate  ↓"),
               ("F1", "F1  ↑")]
    arms = ARM_ORDER[::-1]                       # first arm at the top
    fig, axes = plt.subplots(len(metrics), len(LABELS), figsize=(17, 8.6),
                             sharey=True, sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    for r, (m, mlabel) in enumerate(metrics):
        for c, lab in enumerate(LABELS):
            ax = axes[r, c]
            ax.set_facecolor("#fcfcfb")
            sub = long[long.Criterion == lab].set_index("Arm")
            for y, arm in enumerate(arms):
                v, lo, hi = sub.loc[arm, [m, f"{m} CI low", f"{m} CI high"]]
                if pd.isna(v):
                    continue
                col = TEAM_COLOR if arm in (TEAM, PROTO) else MODEL_COLOR
                if not pd.isna(lo):
                    ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round",
                            alpha=0.55, zorder=2)
                ax.plot(v, y, "o", ms=6.5, zorder=3,
                        mfc="#fcfcfb" if arm == PROTO else col, mec=col, mew=1.6)
            if lab == "Overall":
                ax.axvspan(-0.05, 1.05, color="#f0efec", zorder=0)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.7, len(arms) - 0.3)
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(["0", ".5", "1"])
            ax.grid(axis="x", color=GRID, lw=0.8)
            ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                gp = int(sub["GT positives"].iloc[0])
                name = CRITERION_NAME.get(lab, "all 7 criteria")
                ax.set_title(f"{lab}\n{name} · {gp} GT+", fontsize=9.5, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(mlabel, fontsize=10, color=INK, labelpad=8)
    axes[0, 0].set_yticks(range(len(arms)))
    axes[0, 0].set_yticklabels([a.replace(" (shown to SMEs)", "\n(shown to SMEs)")
                                for a in arms], fontsize=8.5, color=INK)
    fig.suptitle("PM per-criterion performance by arm — pooled over runs, 95% bootstrap CI",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.005,
             "Model arms (blue): first 100 non-empty runs each. Human-AI team (orange): 6 SME "
             "sessions, adjudicated modify labels, scored over all 91 cells. Hollow = the "
             "prototype's own flags in those sessions (reference). A2/A3/A5/A6 have only "
             "1–2 GT positives — read their rates as counts.",
             fontsize=8, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.025, 1, 0.95), h_pad=1.2, w_pad=1.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path,
                    default=output_dir("criterion_compare") / "PM_criterion_compare.xlsx")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    a = ap.parse_args()

    res = build(a.boot)
    write_workbook(res, a.out)
    plot(res, a.out.with_suffix(".png"))
    print(f"wrote {a.out}\n      {a.out.with_suffix('.png')}")
    print("runs scored:", res["runs"])
    for m in ("TPR", "FPR", "F1"):
        print(f"\n{m} (pooled, 95% CI)")
        print(res["wide"][m].to_string(index=False))


if __name__ == "__main__":
    main()
