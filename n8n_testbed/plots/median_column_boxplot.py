"""Grouped boxplot with the medians in a right-hand column — the StoryDeck style.

Same data and palette as compare_boxplot (pooled metrics per run, slots
red / blue / gold in fixed order), but each box's median is printed in a
"median" column at the right edge instead of above the box, so three series
per metric row stay readable.

    python -m n8n_testbed.plots.median_column_boxplot \
        --configs beta1base n8nsub46 cltop3 \
        --labels "Baseline AI" "Beta 1" "CL + top-3 SME sub-rules" \
        --title "PM: Baseline AI, Beta 1 and CL + top-3 SME sub-rules"
"""

from __future__ import annotations

import argparse
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from .compare_boxplot import METRICS, LABELS, SERIES, INK, MUTED, PLOT_DIR, _pooled, _style


def generate(dataset: str, configs: List[str], labels: List[str], title: str,
             out_name: str):
    series = [_pooled(dataset, c) for c in configs]
    n = len(series)
    fig, ax = plt.subplots(figsize=(8.5, 10))
    span = 0.62
    offsets = [-span / 2 + i * span / (n - 1) for i in range(n)] if n > 1 else [0.0]
    width = span / n * 0.62

    for si, (data, color) in enumerate(zip(series, SERIES)):
        pos = [i + 1 + offsets[si] for i in range(len(METRICS))]
        bp = ax.boxplot([data[m] for m in METRICS], positions=pos, widths=width,
                        vert=False, patch_artist=True, showfliers=True, manage_ticks=False)
        _style(bp, color)
        for p, m in zip(pos, METRICS):
            ax.text(1.07, p, f"{np.median(data[m]):.2f}", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=11, fontweight="bold", color=color)
    ax.text(1.07, 0.55, "median", transform=ax.get_yaxis_transform(), va="center",
            ha="left", fontsize=9.5, color=MUTED)

    ax.set_yticks(range(1, len(METRICS) + 1))
    ax.set_yticklabels(LABELS, fontsize=13, color=INK)
    ax.set_ylim(0.35, len(METRICS) + 0.6)
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03)
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
    ax.set_title(title, fontweight="bold", fontsize=14, color=INK, pad=30)
    ax.legend(handles=[Line2D([0], [0], color=c, lw=2.4, label=l)
                       for c, l in zip(SERIES, labels)],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=n, frameon=False,
              fontsize=12, labelcolor=INK, handlelength=1.8, columnspacing=2.0)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / out_name
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PM")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"{a.dataset}_{'_vs_'.join(a.configs)}_median_col.png"
    print("wrote", generate(a.dataset, a.configs, a.labels, a.title, out))


if __name__ == "__main__":
    main()
