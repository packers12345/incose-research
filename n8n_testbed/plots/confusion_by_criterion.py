"""Per-criterion confusion-count boxplots: one figure each for TP, FP, FN, TN.

The pooled charts show where an arm sits overall; they cannot show WHICH
criterion moved. These do: one row per criterion, one box per arm, counts (not
rates) across runs.

Counts rather than rates on purpose. A criterion's rate denominator is tiny and
uneven here - A2 has 1 ground-truth positive and A10 has 10, so an A2 TPR is
one of {0, 1} while A10's is nearly continuous. Rates make those look
comparable; counts do not pretend they are, and the ground-truth totals are
printed on the axis labels so the ceiling for each row is visible.

Only the seven scored criteria appear (A2-A6, A9, A10). A1 and A11 have no
ground-truth row - the GT workbook carries them but with zero positives, and
the prototype strips them from its prompt entirely - so there is nothing to
score against and a box for them would be an artefact.

    python -m n8n_testbed.plots.confusion_by_criterion \
        --configs n8nsub46 aigenv2sme \
        --labels "GR subrules (Sonnet 4.6)" "AIGen V2 + SME subrules"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from n8n_testbed import metrics
from shared.datasets import CRITERIA, CRITERIA_NAMES
from shared.paths import output_dir

PLOT_DIR = output_dir("n8n_testbed") / "plots" / "by_criterion"

# Same validated categorical palette the pooled comparison charts use.
SERIES = ["#B22222", "#1F6FEB", "#B8860B", "#7B3294"]
INK = "#1a1a1a"
MUTED = "#6b7280"

COUNTS = [("tp", "True positives", "caught a real violation"),
          ("fp", "False positives", "flagged a clean cell"),
          ("fn", "False negatives", "missed a real violation"),
          ("tn", "True negatives", "correctly left clean")]


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


def _per_criterion(config: str) -> Dict[str, Dict[str, np.ndarray]]:
    """{criterion: {tp/fp/fn/tn: array over runs}}"""
    df = metrics.score("PM", config, criteria=CRITERIA)
    out = {}
    for c in CRITERIA:
        g = df[df["rule"] == c]
        out[c] = {k: g[k].values.astype(float) for k, *_ in COUNTS}
    return out


def generate(configs: List[str], labels: List[str], dataset: str = "PM",
             figsize=(9, 7)) -> List[Path]:
    if len(configs) > len(SERIES):
        raise ValueError(f"palette has {len(SERIES)} validated slots")
    series = [_per_criterion(c) for c in configs]
    n_runs = [len(series[i][CRITERIA[0]]["tp"]) for i in range(len(configs))]

    gt = metrics._gt_map(dataset)
    npos = {c: sum(gt[c].values()) for c in CRITERIA}

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    n_series = len(series)
    span = 0.52
    offsets = ([0.0] if n_series == 1
               else [-span / 2 + i * span / (n_series - 1) for i in range(n_series)])
    width = span / n_series * 0.78

    for key, title, gloss in COUNTS:
        fig, ax = plt.subplots(figsize=figsize)
        # ceiling for this panel: positives for TP/FN, negatives for FP/TN
        ceiling = {c: (npos[c] if key in ("tp", "fn") else 13 - npos[c])
                   for c in CRITERIA}

        for si, (data, color) in enumerate(zip(series, SERIES)):
            positions = [i + 1 + offsets[si] for i in range(len(CRITERIA))]
            bp = ax.boxplot([data[c][key] for c in CRITERIA], positions=positions,
                            widths=width, vert=False, patch_artist=True,
                            showfliers=True, manage_ticks=False)
            _style(bp, color)
            for i, c in enumerate(CRITERIA):
                m = float(np.median(data[c][key]))
                ax.text(m, i + 1 + offsets[si] + (0.20 if si % 2 == 0 else -0.30),
                        f"{m:.1f}", va="bottom" if si % 2 == 0 else "top",
                        ha="center", fontsize=8.5, color=color)

        # the reachable maximum per row, so a low box reads as "missed" not "few exist"
        for i, c in enumerate(CRITERIA):
            ax.plot([ceiling[c]], [i + 1], marker="|", markersize=13,
                    color=MUTED, alpha=0.75, linestyle="none", zorder=0)

        ax.set_yticks(range(1, len(CRITERIA) + 1))
        ax.set_yticklabels([f"{c} {CRITERIA_NAMES[c]}\n({npos[c]}+ / {13 - npos[c]}-)"
                            for c in CRITERIA], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(f"{title} per run  (of 13 requirements)", fontsize=11)
        ax.set_xlim(left=-0.3)
        ax.grid(axis="x", color="#d7d7d7", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.tick_params(colors=INK)

        ax.set_title(f"{dataset}: {title} by criterion  -  {gloss}",
                     fontsize=12.5, fontweight="bold", color=INK, pad=26)
        handles = [Line2D([0], [0], color=SERIES[i], lw=2.4,
                          label=f"{labels[i]}  (n={n_runs[i]})")
                   for i in range(n_series)]
        handles.append(Line2D([0], [0], color=MUTED, marker="|", linestyle="none",
                              markersize=11, label="reachable maximum"))
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  ncol=min(3, n_series + 1), frameon=False, fontsize=9.5)

        fig.text(0.5, 0.005,
                 "Box = IQR across runs, whiskers 1.5xIQR. Row label shows ground-truth "
                 "positives/negatives for that criterion.",
                 ha="center", fontsize=8.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.02, 1, 1))

        stem = "_vs_".join(configs)
        path = PLOT_DIR / f"{dataset}_{stem}_{key}_by_criterion.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
        print(f"  wrote {path.name}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-criterion TP/FP/FN/TN boxplots")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--dataset", default="PM")
    args = ap.parse_args()
    if len(args.labels) != len(args.configs):
        raise SystemExit("--labels must match --configs")
    generate(args.configs, args.labels, args.dataset)


if __name__ == "__main__":
    main()
