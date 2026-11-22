"""Per-reviewer scorecard: TPR / TNR / FPR / FNR / Accuracy / Precision.

Companion to `reviewer_rates.py`. That module asks how the rates spread ACROSS
reviewers and answers with a boxplot, which deliberately hides who is who; this
one asks what each reviewer actually scored. Same data, same `adjudicated`
policy for `modify`, same universe — only the question differs.

Every cell carries two numbers:

    large   the reviewer's OWN flag set (n = 39..42) — the published figure.
    small   the COMMON CORE flagged in every session.

The second one exists because `n` is not a design choice and should not be read
as one. It is however many violations the AI happened to flag in that session:
raw record count == the app's own `total_violations` == the number scored, for
every session. Nothing is filtered — the `crit not in gt` guard in
reviewer_rates would drop A1/A11 flags, which have no scorable GT row, but the
prototype that produced this feedback strips A1 and A11 from its criteria list
before prompting (`ai_analyzer._load_criteria`), so it cannot emit one. The
guard stays for data from any other flagger. The 39..42 spread is run-to-run
variation across sessions (pairwise Jaccard 0.78..0.93) at the prototype's
temperature 0.1, one call per requirement.

That leaves each reviewer scored on a slightly different question set, so the
rows are not strictly a common panel. The common core is where they are: the
cells every session flagged, judged by everyone. It moves TPR/TNR by at most
0.05 and changes no ranking, which is the point of printing it.

Shading is direction-aware — a low FPR is shaded like a high TPR — so the eye
reads "good/bad", never "big/small".

PROVISIONAL ROWS
----------------
A session whose `modify` decisions have not been hand-adjudicated yet cannot be
given a point estimate: `reviewer_rates.ADJUDICATED` has no labels for it, and
guessing them here would launder a mechanical rule into a published number. Such
a session is passed via `extra=` and rendered as the RANGE between the two
bounding policies (every modify endorsed / every modify rejected), on a neutral
hatched fill, excluded from the median. Replace it with a real row by adding the
labels to ADJUDICATED and moving the session into reviewer_rates.REVIEWERS.

    python -m reviewer_reliability.scorecard              # the five settled reviewers
    python -m reviewer_reliability.scorecard --with-provisional
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from shared.paths import FEEDBACK_DIR, output_dir, REVIEWER_RUNS_DIR

from . import reviewer_rates as rr

OUTPUT_DIR = output_dir("reviewer_reliability")
PLOT_DIR = OUTPUT_DIR / "plots" / "notebook_style"

# Sessions present in data/sme_feedback that reviewer_rates does not score,
# because their `modify` decisions have no entry in ADJUDICATED yet.
PROVISIONAL = []          # d566d756 adjudicated and promoted into reviewer_rates.REVIEWERS

INK = "#1a1a1a"
MUTED = "#6b7280"
RULE = "#d1d5db"
POOR = "#B22222"
MID = "#f4f1ec"
GOOD = "#2f7d5f"
PROV_FILL = "#eeecea"

CMAP = LinearSegmentedColormap.from_list("goodness", [POOR, MID, GOOD])

METRICS = [
    ("tpr",  "TPR",       True),
    ("tnr",  "TNR",       True),
    ("fpr",  "FPR",       False),
    ("fnr",  "FNR",       False),
    ("acc",  "Accuracy",  True),
    ("prec", "Precision", True),
]
KEYS = [m[0] for m in METRICS]


def _decisions(rid: str, gt, force: str | None = None) -> Dict[Tuple[str, str], bool]:
    """(requirement, criterion) -> did this reviewer endorse the flag?

    `force` overrides how `modify` resolves: None uses the hand adjudication in
    reviewer_rates.ADJUDICATED (and raises if a label is missing, by design);
    "endorse"/"reject" forces every modify one way, for bounding an unadjudicated
    session.
    """
    data = json.loads((FEEDBACK_DIR / f"feedback_{rid}.json").read_text(encoding="utf-8"))
    out: Dict[Tuple[str, str], bool] = {}
    for req in data["requirement_feedback"]:
        pm = rr.REQ_TO_PM.get(int(req["req_id"]))
        for v in req.get("violation_feedback", []):
            crit = v.get("rule_id")
            if pm is None or crit not in gt:
                continue
            action = v.get("user_action", "")
            if action == "modify":
                if force is None:
                    endorsed = rr.ADJUDICATED[(rid, v["violation_id"])] == "endorse"
                else:
                    endorsed = force == "endorse"
            else:
                endorsed = action == "accept"
            out[(pm, crit)] = endorsed
    return out


def _rates(decisions, cells, gt) -> Dict[str, float]:
    tp = tn = fp = fn = 0
    for key in cells:
        endorsed, truth = decisions[key], gt[key[1]].get(key[0], False)
        if endorsed and truth:
            tp += 1
        elif endorsed:
            fp += 1
        elif truth:
            fn += 1
        else:
            tn += 1
    n = tp + tn + fp + fn
    nan = float("nan")
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "tpr": tp / (tp + fn) if tp + fn else nan,
        "tnr": tn / (tn + fp) if tn + fp else nan,
        "fpr": fp / (fp + tn) if fp + tn else nan,
        "fnr": fn / (fn + tp) if fn + tp else nan,
        "acc": (tp + tn) / n if n else nan,
        "prec": tp / (tp + fp) if tp + fp else nan,
    }


def build(extra: Sequence[str] = ()) -> Tuple[List[Dict], int]:
    """One row per reviewer, plus the size of the common core.

    Settled reviewers carry point estimates; sessions named in `extra` carry
    (low, high) bounds and are marked provisional.
    """
    gt = rr.load_gt()
    settled = {rid: _decisions(rid, gt) for rid in rr.REVIEWERS}
    bounds = {rid: (_decisions(rid, gt, "endorse"), _decisions(rid, gt, "reject"))
              for rid in extra}

    keysets = [set(d) for d in settled.values()] + [set(d[0]) for d in bounds.values()]
    core = set.intersection(*keysets)

    labels = dict(rr.LABEL)
    for i, rid in enumerate(extra, start=len(rr.REVIEWERS) + 1):
        labels[rid] = f"SME {i}"

    rows: List[Dict] = []
    for rid in list(rr.REVIEWERS) + list(extra):
        row = {"reviewer": labels[rid], "session": rid,
               "provisional": rid in bounds}
        if rid in settled:
            own, cor = _rates(settled[rid], settled[rid], gt), _rates(settled[rid], core, gt)
            row["n"], row["n_core"] = own["n"], cor["n"]
            for k in ("tp", "tn", "fp", "fn"):
                row[k] = own[k]
            for k in KEYS:
                row[k], row[f"{k}_core"] = own[k], cor[k]
        else:
            lo_d, hi_d = bounds[rid]
            own = [_rates(d, d, gt) for d in (lo_d, hi_d)]
            cor = [_rates(d, core, gt) for d in (lo_d, hi_d)]
            row["n"], row["n_core"] = own[0]["n"], cor[0]["n"]
            for k in ("tp", "tn", "fp", "fn"):
                row[k] = (own[0][k], own[1][k])
            for k in KEYS:
                row[k] = (min(o[k] for o in own), max(o[k] for o in own))
                row[f"{k}_core"] = (min(c[k] for c in cor), max(c[k] for c in cor))
        rows.append(row)
    return rows, len(core)


def _text_on(rgb) -> str:
    r, g, b = rgb[:3]
    return "#ffffff" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else INK


def plot(rows: List[Dict], n_core: int) -> Path:
    has_prov = any(r["provisional"] for r in rows)
    n_rows = len(rows)
    w_label, w_n = 1.30, 0.72
    w_cell = 1.16 if has_prov else 0.92
    h_row, h_head = 0.60, 0.72

    total_w = w_label + w_n + w_cell * len(METRICS)
    total_h = h_head + h_row * (n_rows + 1)

    fig, ax = plt.subplots(figsize=(total_w * 1.18, total_h * 1.16 + 1.25))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")
    ax.invert_yaxis()

    xs = [w_label + w_n + i * w_cell for i in range(len(METRICS))]

    ax.text(w_label + w_n / 2, h_head * 0.52, "n", ha="center", va="center",
            fontsize=11, color=MUTED, style="italic")
    for (key, head, higher), x in zip(METRICS, xs):
        ax.text(x + w_cell / 2, h_head * 0.40, head, ha="center", va="center",
                fontsize=11.5, color=INK, fontweight="bold")
        ax.text(x + w_cell / 2, h_head * 0.76,
                "higher is better" if higher else "lower is better",
                ha="center", va="center", fontsize=6.4, color=MUTED)
    ax.plot([0, total_w], [h_head, h_head], color=INK, lw=1.1)

    x_split = xs[4] - 0.035
    ax.plot([x_split, x_split], [h_head * 0.16, total_h - h_row * 0.06],
            color=RULE, lw=1.0)

    settled = [r for r in rows if not r["provisional"]]

    for i, row in enumerate(rows):
        y = h_head + i * h_row
        prov = row["provisional"]

        ax.text(0.06, y + h_row / 2, row["reviewer"], ha="left", va="center",
                fontsize=12, color=MUTED if prov else INK,
                style="italic" if prov else "normal")
        ax.text(w_label + w_n / 2, y + h_row * 0.40, f"{int(row['n'])}",
                ha="center", va="center", fontsize=11, color=INK)
        ax.text(w_label + w_n / 2, y + h_row * 0.755, f"{int(row['n_core'])}",
                ha="center", va="center", fontsize=7.6, color=MUTED)

        for (key, _h, higher), x in zip(METRICS, xs):
            v, vc = row[key], row[f"{key}_core"]
            rect = Rectangle((x + 0.035, y + 0.045), w_cell - 0.07, h_row - 0.09,
                             edgecolor="white", lw=1.2)
            if prov:
                # No point estimate exists, so no shading is honest here.
                rect.set(facecolor=PROV_FILL, hatch="///", edgecolor="#ffffff")
                ax.add_patch(rect)
                ax.text(x + w_cell / 2, y + h_row * 0.40,
                        f"{v[0]:.2f}–{v[1]:.2f}", ha="center", va="center",
                        fontsize=10, color=INK)
                ax.text(x + w_cell / 2, y + h_row * 0.775,
                        f"{vc[0]:.2f}–{vc[1]:.2f}", ha="center", va="center",
                        fontsize=7, color=MUTED)
            else:
                fill = CMAP(v if higher else 1.0 - v)
                rect.set(facecolor=fill)
                ax.add_patch(rect)
                fg = _text_on(fill)
                ax.text(x + w_cell / 2, y + h_row * 0.40, f"{v:.2f}",
                        ha="center", va="center", fontsize=12, color=fg)
                ax.text(x + w_cell / 2, y + h_row * 0.765, f"{vc:.2f}",
                        ha="center", va="center", fontsize=7.6, color=fg, alpha=0.72)

    y = h_head + n_rows * h_row
    ax.plot([0, total_w], [y, y], color=RULE, lw=1.0)
    med_label = ("median  " + f"(SME 1–{len(settled)})") if has_prov else "median"
    ax.text(0.06, y + h_row / 2, med_label, ha="left", va="center",
            fontsize=10.5 if has_prov else 11, color=MUTED, style="italic")
    for key, x in zip(KEYS, xs):
        m = float(np.median([r[key] for r in settled]))
        mc = float(np.median([r[f"{key}_core"] for r in settled]))
        ax.text(x + w_cell / 2, y + h_row * 0.42, f"{m:.2f}",
                ha="center", va="center", fontsize=11.5, color=MUTED)
        ax.text(x + w_cell / 2, y + h_row * 0.78, f"{mc:.2f}",
                ha="center", va="center", fontsize=7.6, color=MUTED, alpha=0.8)

    ax.set_title("SME reviewer agreement with ground truth, by reviewer",
                 fontweight="bold", fontsize=14.5, color=INK, pad=14)

    caption = (
        f"Large figure: the reviewer's own flag set (n above).  Small figure: the "
        f"{n_core} cells flagged in every session, where everyone judged identical "
        f"items.\n"
        "n is not a fixed panel — it is however many violations the AI flagged in "
        "that session, all of which are scored. Universe is those flags, not all 91\n"
        "requirement x criterion cells, so these are reviewing rates, not independent "
        "detection. 'modify' resolved by hand adjudication. Shading is direction-aware.\n"
        "The four rates carry two independent numbers: FNR = 1 - TPR and FPR = 1 - TNR "
        "exactly, each pair sharing a denominator. Only Accuracy and Precision add "
        "information."
    )
    if has_prov:
        names = ", ".join(r["reviewer"] for r in rows if r["provisional"])
        caption += (
            f"\nHatched: {names} has no hand adjudication for its `modify` decisions "
            "yet, so no point estimate exists. The range spans every modify endorsed "
            "to every modify\nrejected. Excluded from the median until those labels "
            "are supplied."
        )

    ax.text(0, total_h + 0.30, caption, transform=ax.transData, ha="left", va="top",
            fontsize=8.2, color=MUTED, linespacing=1.65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    name = "PM_reviewer_scorecard_provisional" if has_prov else "PM_reviewer_scorecard"
    out = PLOT_DIR / f"{name}.png"
    fig.subplots_adjust(left=0.015, right=0.985, top=0.88, bottom=0.20)
    plt.savefig(out, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.32)
    plt.close(fig)
    return out


def _fmt(v, count: bool = False) -> str:
    """Rate columns hold (min, max); count columns hold (endorse, reject)."""
    if not isinstance(v, tuple):
        return str(int(v)) if count else f"{v:.3f}"
    return f"{int(v[0])}|{int(v[1])}" if count else f"{v[0]:.3f}-{v[1]:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--with-provisional", action="store_true",
                    help=f"include unadjudicated sessions ({', '.join(PROVISIONAL)}) "
                         "as bounded ranges")
    a = ap.parse_args()

    extra = PROVISIONAL if a.with_provisional else ()
    rows, n_core = build(extra)

    counts = ["n", "n_core", "tp", "tn", "fp", "fn"]
    cols = ["reviewer", "session"] + counts + KEYS
    table = pd.DataFrame([
        {c: (r[c] if c in ("reviewer", "session") else _fmt(r[c], c in counts))
         for c in cols}
        for r in rows])
    print(table.to_string(index=False))
    print(f"\ncommon core: {n_core} cells flagged in every session")
    if extra:
        print("provisional: " + ", ".join(r["reviewer"] for r in rows if r["provisional"])
              + "  — rates shown as min-max, counts as endorse|reject, across the two "
                "bounding policies")

    REVIEWER_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_provisional" if extra else ""
    csv = REVIEWER_RUNS_DIR / f"PM_reviewer_scorecard{suffix}.csv"
    table.to_csv(csv, index=False)
    print(f"\nwrote {csv}")
    print(f"wrote {plot(rows, n_core)}")


if __name__ == "__main__":
    main()
