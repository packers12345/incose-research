"""Sponsor vs external experts on one axis — TPR, TNR, FPR, FNR, eight boxes.

`group_rates` draws each affiliation in its own figure, which makes the two
groups read as separate results. This puts them on shared metric rows so the
gap between them is the thing you see, in the same two-series language as
`n8n_testbed.plots.compare_boxplot` (paired offset boxes, median labelled per
box, identity never carried by colour alone — the table prints underneath).

Covers the full `reviewer_rates.METRICS` set in its canonical order, so the
rows line up with the single-group figures from `group_rates`.

READ THE BOX WITH CARE — the warning in `group_rates` applies here and is
worth more, not less, when the groups sit side by side and invite comparison.
Each box spans THREE reviewers. Quartiles on three points are interpolated
between min, median and max; they describe no distribution. The rings are the
data. A visible gap between two three-point boxes is not evidence of a
difference between sponsor and external reviewers as populations.

    python -m reviewer_reliability.group_compare
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from shared.paths import REVIEWER_RUNS_DIR

from . import reviewer_rates as rr

PLOT_DIR = rr.OUTPUT_DIR / "plots" / "notebook_style"

# reviewer_rates.METRICS stays at the four rates so the single-group figures are
# unchanged; this chart adds the three summary metrics computed from the same
# confusion matrix.
METRICS = ["tpr", "tnr", "fpr", "fnr", "accuracy", "precision", "f1"]
LABELS = ["TPR", "TNR", "FPR", "FNR", "Accuracy", "Precision", "F1"]

# Slots 0-2 of the validated categorical palette in compare_boxplot: worst
# adjacent CVD separation dE 14.2 (deutan), normal-vision floor 20.1.
SERIES = ["#B22222", "#1F6FEB", "#B8860B"]
INK = "#1a1a1a"
MUTED = "#6b7280"

# "Expert" in the ask; "External" is the label carried in reviewer_rates.
# AFFILIATION and the by-group CSV, so the data keeps its name and the figure
# spells out what the group is.
# Display labels only — the affiliation keys in reviewer_rates.AFFILIATION stay
# as they are, so the data and the figure never disagree about who is in which
# group. "NNSA" reads as "Sponsor" on the chart.
GROUP_LABEL = {"NNSA": "Sponsor", "External": "External experts",
               "All": "All reviewers"}

# The aggregate is not a third group — it is the same six reviewers pooled, so
# its rings are the union of the other two series and its median is the median
# of all six, NOT the midpoint between the group medians.
ALL = "All"


def _style(bp, color: str) -> None:
    for box in bp["boxes"]:
        box.set(color=color, facecolor="none", linewidth=1.3)
    for med in bp["medians"]:
        med.set(color=color, linewidth=2.0)
    for part in ("whiskers", "caps"):
        for art in bp[part]:
            art.set(color=color, linewidth=1.0)
    for fl in bp["fliers"]:
        fl.set(marker="o", markersize=3.5, markerfacecolor="none",
               markeredgecolor=color, alpha=0.55)


def generate(df, out_name: str | None = None) -> Path:
    groups = list(rr.GROUPS) + [ALL]
    series = [{m: (df if g == ALL else df[df["group"] == g])[m].dropna().values
               for m in METRICS}
              for g in groups]
    sizes = [len(df) if g == ALL else int((df["group"] == g).sum()) for g in groups]

    n_series = len(series)
    fig, ax = plt.subplots(figsize=(9.8, 12.6))
    # ROW pushes the four metric groups apart; span is how much of that row the
    # three series occupy, and the boxes are drawn thin inside it. Keeping the
    # boxes well short of the row pitch is what stops adjacent metrics reading
    # as one block of stripes.
    ROW = 1.30
    span = 0.62
    offsets = [-span / 2 + i * span / (n_series - 1) for i in range(n_series)]
    width = span / n_series * 0.52

    for si, (data, color) in enumerate(zip(series, SERIES)):
        positions = [(i + 1) * ROW + offsets[si] for i in range(len(METRICS))]
        bp = ax.boxplot([data[m] for m in METRICS], positions=positions,
                        widths=width, vert=False, patch_artist=True,
                        showfliers=True, manage_ticks=False)
        _style(bp, color)
        # One ring per reviewer, jittered off the box spine. At n=3 these ARE
        # the result; the box behind them is scaffolding.
        for pos, m in zip(positions, METRICS):
            vals = data[m]
            jitter = np.linspace(-width * 0.30, width * 0.30, len(vals))
            ax.scatter(vals, pos + jitter, s=38, facecolors="none",
                       edgecolors=color, linewidths=1.2, zorder=3,
                       clip_on=False)
            # Median values live in a column past the axis, each on its own
            # series row. Printed above the box they landed on the neighbouring
            # series' box whenever two medians were close.
            ax.text(1.07, pos, f"{np.median(vals):.2f}", va="center", ha="left",
                    color=color, fontsize=10, fontweight="bold", clip_on=False)

    ax.text(1.07, ROW - 0.62, "median", va="center", ha="left", fontsize=9,
            color=MUTED, clip_on=False)
    ax.set_yticks([(i + 1) * ROW for i in range(len(METRICS))])
    ax.set_yticklabels(LABELS, fontsize=13, color=INK)
    ax.set_ylim(ROW - 0.75, len(METRICS) * ROW + 0.75)
    ax.invert_yaxis()
    # Rates of exactly 0.00 and 1.00 are real values here, and a reviewer sits
    # on each. With xlim flush to the data those rings are sliced in half by the
    # spine and the whisker caps disappear; pad the view without moving a tick.
    ax.set_xlim(-0.035, 1.035)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", labelsize=12, colors=INK)
    ax.set_xlabel("Rate", fontsize=13, color=INK)
    ax.set_ylabel("Metric", fontsize=13, color=INK)
    ax.xaxis.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d1d5db")

    ax.set_title("PM: reviewer rates by affiliation, with all six pooled",
                 fontweight="bold", fontsize=14, color=INK, pad=34)
    ax.legend(handles=[Line2D([0], [0], color=c, lw=2.4,
                              label=f"{GROUP_LABEL[g]}  (n={n})")
                       for c, g, n in zip(SERIES, groups, sizes)],
              loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3,
              frameon=False, fontsize=12, labelcolor=INK,
              handlelength=1.6, columnspacing=2.4)
    fig.text(0.5, -0.02,
             "Each ring is one reviewer. The group boxes span three reviewers "
             "each and carry no distributional weight; the gold box is the same "
             "six pooled,\nso its rings are the other two series combined and "
             "its median is the median of all six, not the midpoint between the "
             "group medians.\nHigher is better for TPR/TNR/Accuracy/Precision/F1; lower for "
             "FPR/FNR.",
             ha="center", fontsize=9.5, color=MUTED)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / (out_name or "PM_reviewer_rates_NNSA_vs_External.png")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Identity must never be colour-alone: the same numbers, as text.
    print(f"{'metric':<8}" + "".join(f"{GROUP_LABEL[g][:20]:>22}" for g in groups)
          + f"{'Spons-Ext':>10}")
    print("-" * (8 + 22 * len(groups) + 10))
    for m, lab in zip(METRICS, LABELS):
        meds = [float(np.median(s[m])) for s in series]
        cells = "".join(
            f"{f'{np.median(s[m]):.2f} (mean {np.mean(s[m]):.2f})':>22}"
            for s in series)
        print(f"{lab:<8}{cells}{meds[0] - meds[1]:>+10.2f}")
    return out


def main() -> None:
    from shared.paths import DOCS_FIGURE_DIR
    import shutil

    df = rr.score("adjudicated", verbose=False)
    df["group"] = df["session"].map(rr.AFFILIATION)
    missing = df[df["group"].isna()]
    if len(missing):
        raise KeyError(f"no affiliation for {list(missing['session'])}; "
                       "add it to reviewer_rates.AFFILIATION")

    for g in rr.GROUPS:
        sub = df[df["group"] == g]
        print(f"--- {GROUP_LABEL[g]}  (n={len(sub)}) ---")
        print(sub[["reviewer", "session", "n", "tp", "tn", "fp", "fn"] + METRICS]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()

    out = generate(df)
    DOCS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out, DOCS_FIGURE_DIR / out.name)
    print(f"\nwrote {out}")
    print(f"wrote {DOCS_FIGURE_DIR / out.name}")


if __name__ == "__main__":
    main()
