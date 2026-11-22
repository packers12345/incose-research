"""The reviewer-rate boxplot, split by reviewer affiliation.

Same data, same `adjudicated` policy and same visual language as
`reviewer_rates.plot` — one box per metric, one ring per reviewer — but drawn
once per group in `reviewer_rates.AFFILIATION`, so NNSA reviewers and external
experts can be read separately rather than pooled.

    python -m reviewer_reliability.group_rates

Writes docs/figures/PM_reviewer_rates_NNSA.png and ..._External.png, plus a
per-group summary CSV.

READ THE BOX WITH CARE. Each group holds three reviewers. A three-point box is
min / median / max with quartiles interpolated between them; it carries no
distributional information worth the name. The rings are the data — the box is
scaffolding. Group medians are reported because they are asked for, not because
three reviewers per arm supports inference about the populations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

from shared.paths import REVIEWER_RUNS_DIR

from . import reviewer_rates as rr

PLOT_DIR = rr.OUTPUT_DIR / "plots" / "notebook_style"

SERIES = rr.SERIES
INK = rr.INK
MUTED = rr.MUTED


def plot_group(df: pd.DataFrame, group: str, modify: str = "adjudicated") -> Path:
    """One boxplot for one affiliation group. Mirrors reviewer_rates.plot."""
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    data = [df[m].dropna().values for m in rr.METRICS]
    bp = ax.boxplot(data, tick_labels=[rr.MLABEL[m] for m in rr.METRICS], vert=False,
                    patch_artist=True, widths=0.45, showfliers=False)
    for box in bp["boxes"]:
        box.set(color=SERIES, facecolor="none", linewidth=1.3)
    for med in bp["medians"]:
        med.set(color=SERIES, linewidth=2.2)
    for part in ("whiskers", "caps"):
        for art in bp[part]:
            art.set(color=SERIES, linewidth=1.0)

    # With n=3 the rings are the whole story; label each so the reader can tell
    # which reviewer is which without cross-referencing the table.
    rng = np.random.default_rng(0)
    for i, m in enumerate(rr.METRICS, start=1):
        vals = df[m].values
        jitter = rng.uniform(-0.13, 0.13, len(vals))
        ax.scatter(vals, i + jitter, s=42, facecolors="none",
                   edgecolors=INK, linewidths=1.1, zorder=3)
        ax.text(float(np.nanmedian(vals)), i - 0.34,
                f"{np.nanmedian(vals):.2f}", ha="center", va="center",
                color=SERIES, fontsize=12)

    ax.set_xlim(-0.035, 1.035)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(labelsize=12, colors=INK)
    ax.set_xlabel("Rate", fontsize=13, color=INK)
    ax.set_ylabel("Metric", fontsize=13, color=INK)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d1d5db")

    who = ", ".join(df["reviewer"])
    ax.set_title(f"{group} reviewers — agreement with ground truth  "
                 f"(n={len(df)} reviewers)",
                 fontweight="bold", fontsize=14, color=INK, pad=12)
    ax.text(0.5, -0.155,
            f"{who}.  Each ring is one reviewer; 'modify' policy: {modify}.\n"
            f"With three reviewers the box is min/median/max — read the rings, not "
            f"the quartiles. Universe is the AI-flagged\nviolations each reviewer "
            f"saw, not all requirement x criterion cells.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color=MUTED, linespacing=1.5)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / f"PM_reviewer_rates_{group}.png"
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.28)
    plt.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
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

    cols = ["reviewer", "session", "group", "n", "tp", "tn", "fp", "fn"] + rr.METRICS
    REVIEWER_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(REVIEWER_RUNS_DIR / "PM_reviewer_rates_by_group.csv", index=False)

    DOCS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for group in rr.GROUPS:
        sub = df[df["group"] == group].reset_index(drop=True)
        print(f"--- {group}  (n={len(sub)}) ---")
        print(sub[["reviewer", "session", "n", "tp", "tn", "fp", "fn"]
                  + rr.METRICS].to_string(index=False,
                                          float_format=lambda v: f"{v:.3f}"))
        print("    median: " + "  ".join(
            f"{rr.MLABEL[m]}={sub[m].median():.2f}" for m in rr.METRICS))
        produced = plot_group(sub, group)
        shutil.copyfile(produced, DOCS_FIGURE_DIR / produced.name)
        print(f"    wrote {DOCS_FIGURE_DIR / produced.name}\n")

    print("group medians side by side:")
    print(f"  {'metric':<8}" + "".join(f"{g:>12}" for g in rr.GROUPS) + f"{'delta':>10}")
    for m in rr.METRICS:
        meds = [df[df['group'] == g][m].median() for g in rr.GROUPS]
        print(f"  {rr.MLABEL[m]:<8}" + "".join(f"{v:>12.2f}" for v in meds)
              + f"{meds[0] - meds[1]:>+10.2f}")
    print(f"\nwrote {REVIEWER_RUNS_DIR / 'PM_reviewer_rates_by_group.csv'}")


if __name__ == "__main__":
    main()
