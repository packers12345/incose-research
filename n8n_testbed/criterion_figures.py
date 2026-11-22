"""Side-by-side per-criterion F1 figures for chosen PM arms.

Each figure puts A2..A10 on the x-axis with one box per arm, side by side:

  * BOX     - the spread of per-run F1 (one value per run / SME session). This is
              the notebook's view: how much a single run varies.
  * DIAMOND - POOLED F1 (TP/FP/FN summed over every run, then F1), with its
              bootstrap 95% CI as the black bar. This is the headline estimate.

The two answer different questions, so both are drawn. They can disagree: on a
criterion with 1-2 ground-truth positives a single run's F1 can only take a few
values, so the box collapses onto 0 or 1 while the pooled value sits between.

    python -m n8n_testbed.criterion_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from shared.paths import output_dir
from .criterion_compare import (build, CRITERIA, CRITERION_NAME, TEAM,
                                write_workbook)

# (fill, darker edge) - the edge carries contrast for the two light hues.
ARM_STYLE = {
    # StoryDeck palette (compare_boxplot.SERIES): Baseline red, Beta 1 blue,
    # the third arm in each comparison gold.
    "Baseline":   ("#B22222", "#8a1a1a"),
    "Beta1":      ("#1F6FEB", "#1550b0"),
    "CLPlusTop3": ("#B8860B", "#86620a"),
    TEAM:         ("#B8860B", "#86620a"),
}
ARM_LABEL = {"Baseline": "Baseline (definitions only)", "Beta1": "Beta 1 (17 sub-rules)",
             "CLPlusTop3": "CL + top-3 SME sub-rules", TEAM: "Human-AI team (6 sessions)"}

FIGURES = [
    ("F1_baseline_vs_beta1", ["Baseline", "Beta1"]),
    ("F1_beta1_vs_team", ["Beta1", TEAM]),
    ("F1_cltop3_vs_beta1_vs_baseline", ["Baseline", "Beta1", "CLPlusTop3"]),
]

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def per_run_f1(counts: pd.DataFrame) -> pd.DataFrame:
    c = counts.copy()
    denom = 2 * c.TP + c.FP + c.FN
    c["F1"] = np.where(denom == 0, np.nan, 2 * c.TP / denom.replace(0, np.nan))
    return c


def draw(res: dict, arms: list, path: Path) -> None:
    runs = per_run_f1(res["per_run_counts"])
    pooled = res["pooled_long"].set_index(["Arm", "Criterion"])
    gtpos = pooled.xs(arms[0])["GT positives"]

    n = len(arms)
    width = 0.78 / n
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)

    for i, crit in enumerate(CRITERIA):
        for k, arm in enumerate(arms):
            x = i + (k - (n - 1) / 2) * width
            fill, edge = ARM_STYLE[arm]
            vals = runs[(runs.Arm == arm) & (runs.Criterion == crit)].F1.dropna().values
            bp = ax.boxplot([vals], positions=[x], widths=width * 0.82, patch_artist=True,
                            showfliers=arm != TEAM, manage_ticks=False,
                            flierprops=dict(marker="o", ms=3, mfc=fill, mec="none", alpha=0.6),
                            medianprops=dict(color=edge, lw=2),
                            whiskerprops=dict(color=edge, lw=1.2),
                            capprops=dict(color=edge, lw=1.2),
                            boxprops=dict(facecolor=fill + "59", edgecolor=edge, lw=1.4))
            if arm == TEAM:      # 6 sessions: show every point, not just a box
                ax.scatter(x + rng.uniform(-width * 0.2, width * 0.2, len(vals)), vals,
                           s=22, color=fill, edgecolor=SURFACE, lw=0.8, zorder=4)
            v = pooled.loc[(arm, crit), "F1"]
            lo, hi = pooled.loc[(arm, crit), ["F1 CI low", "F1 CI high"]]
            if not pd.isna(v):
                ax.plot([x, x], [lo, hi], color=INK, lw=1.6, zorder=5,
                        solid_capstyle="round")
                ax.plot(x, v, "D", ms=6.5, mfc="white", mec=INK, mew=1.4, zorder=6)
        if i:
            ax.axvline(i - 0.5, color=GRID, lw=0.8, zorder=0)

    ax.set_xticks(range(len(CRITERIA)))
    ax.set_xticklabels([f"{c}\n{CRITERION_NAME[c]}\n{int(gtpos[c])} GT+" for c in CRITERIA],
                       fontsize=10, color=INK)
    ax.set_xlim(-0.5, len(CRITERIA) - 0.5)
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel("F1", fontsize=11, color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    for s in ax.spines.values():
        s.set_visible(False)

    title = " vs ".join(ARM_LABEL[a].split(" (")[0] for a in arms)
    ax.set_title(f"Per-criterion F1 — {title}", loc="left", fontsize=13,
                 fontweight="bold", color=INK, pad=50)
    ax.text(0, 1.115, "Box = spread of per-run F1 (whiskers 1.5×IQR).  Diamond = pooled F1, "
            "bar = 95% bootstrap CI.", transform=ax.transAxes, fontsize=9.5, color=MUTED)
    handles = [Patch(facecolor=ARM_STYLE[a][0] + "59", edgecolor=ARM_STYLE[a][1], lw=1.4,
                     label=ARM_LABEL[a]) for a in arms]
    handles.append(Line2D([0], [0], marker="D", color=INK, mfc="white", mew=1.4, lw=1.6,
                          label="Pooled F1 ± 95% CI"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(handles),
              frameon=False, fontsize=9, handlelength=1.6, columnspacing=1.4,
              labelcolor=INK)
    runs_note = ", ".join(f"{a}: {res['runs'][a]} {'sessions' if a == TEAM else 'runs'}"
                          for a in arms)
    fig.text(0.01, 0.012, f"Point Mass, v5 ground truth, 13 requirements per criterion. {runs_note}.\n"
             "A2/A3/A5/A6 have only 1–2 GT positives, so a single run's F1 can take only a few "
             "values; trust the pooled diamond there.", fontsize=8.3, color=MUTED, ha="left",
             linespacing=1.4)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    out = output_dir("criterion_compare")
    res = build()
    write_workbook(res, out / "PM_criterion_compare.xlsx")
    for name, arms in FIGURES:
        draw(res, arms, out / f"PM_{name}.png")
        print(f"wrote {out / f'PM_{name}.png'}")


if __name__ == "__main__":
    main()
