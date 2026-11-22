"""Per-criterion accuracy, precision and F1 — one figure each, two arms.

`confusion_by_criterion` deliberately plots COUNTS, because most per-criterion
rate denominators here are tiny (A2 has 1 ground-truth positive, A3 has 2) and
a rate built on them invites comparisons the data cannot support. This module
plots rates anyway, because they were asked for, and it is worth being precise
about which half of that warning each one inherits:

  * ACCURACY is denominated on all 13 requirements for every criterion. Its
    denominator is uniform across rows, so the rows are legitimately
    comparable. It is also dominated by true negatives on the sparse criteria —
    a criterion with 1 positive scores ~0.92 by never flagging anything, so
    read it against the ground-truth counts on each row label, not as a
    quality score.

  * PRECISION is denominated on TP+FP, which is whatever the arm chose to flag
    that run. It is undefined when the arm flags nothing (metrics._metrics
    returns None, and those runs drop out), so a precision dot can summarise
    fewer runs than the arm actually has — the per-dot n is printed in the
    table this writes to stdout.

  * F1 inherits precision's denominator problem and adds recall's: on a
    criterion with 1-2 ground-truth positives, recall moves in steps of 1.0 or
    0.5, so F1 is a coarse ladder rather than a continuous score. It is the
    least trustworthy of the three at this level.

One figure per rate rather than shared panels, so each can be dropped into a
deck on its own — the same reason `confusion_by_criterion` writes four files.

Drawn as a dumbbell — median dot per arm, joined by a connector — rather than
paired boxes. At this level the two arms' spreads sit on top of each other and
the denominators are small, so a box invites a spread reading the rows cannot
support; the pooled chart is where spread belongs. Equal medians are drawn as a
split marker (left half series 0, right half series 1) so an unchanged row
still reads as two arms rather than one missing.

    python -m n8n_testbed.plots.rates_by_criterion \
        --configs n8nsub46 smepriority \
        --labels "SME subrules (Sonnet 4.6)" "SME-priority (A5.1/A10.2 removed)"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from n8n_testbed import metrics
from shared.datasets import CRITERIA, CRITERIA_NAMES
from shared.paths import output_dir

PLOT_DIR = output_dir("n8n_testbed") / "plots" / "by_criterion"

# Same validated categorical palette the other comparison charts use.
SERIES = ["#B22222", "#1F6FEB", "#B8860B", "#7B3294"]
INK = "#1a1a1a"
MUTED = "#6b7280"

RATES = [
    ("accuracy", "Accuracy", "(TP+TN) / all 13 requirements",
     "Inflated by true negatives on the sparse criteria: an arm that never "
     "flags a 1-positive criterion still scores ~0.92."),
    ("precision", "Precision", "TP / (TP+FP)",
     "Undefined when the arm flagged nothing that run, so those runs drop "
     "out and the per-dot n below can be under the arm's full run count."),
    ("f1", "F1", "harmonic mean of precision and recall",
     "On a criterion with 1-2 ground-truth positives recall moves in steps "
     "of 0.5 or 1.0, so F1 here is a coarse ladder, not a continuous score."),
]


def _series(config: str, dataset: str) -> Dict[str, Dict[str, np.ndarray]]:
    df = metrics.score(dataset, config, criteria=CRITERIA)
    out = {}
    for c in CRITERIA:
        sub = df[df["rule"] == c]
        # dropna, not fillna: a run where the criterion was never flagged has
        # no precision, and substituting 0 would invent a failure the arm did
        # not commit.
        out[c] = {k: sub[k].dropna().values for k, _, _, _ in RATES}
    return out


def _one(key: str, title: str, denom: str, caveat: str, dataset: str,
         configs: List[str], labels: List[str], series, npos) -> Path:
    n_series = len(series)
    fig, ax = plt.subplots(figsize=(10.0, 7.0))

    # One line per criterion, one dot per arm, however many arms. Equal medians
    # are common here, so a tie is drawn as CONCENTRIC RINGS - the arms in a tie
    # group get descending marker sizes at the same point, so every arm stays
    # visible without moving a dot off its true value or off the shared line.
    ring = [13.0, 9.0, 5.5, 3.0]

    for i, c in enumerate(CRITERIA):
        pos = i + 1
        meds = [float(np.median(s[c][key])) if len(s[c][key]) else np.nan
                for s in series]
        good = [m for m in meds if not np.isnan(m)]
        if len(good) > 1 and max(good) - min(good) > 1e-9:
            ax.plot([min(good), max(good)], [pos, pos], color="#9ca3af",
                    linewidth=1.6, zorder=1, solid_capstyle="round")

        # Group arms that land on the same median, keeping series order.
        groups: Dict[float, List[int]] = {}
        for si, m in enumerate(meds):
            if np.isnan(m):
                continue
            key_m = round(m, 2)
            groups.setdefault(key_m, []).append(si)

        for value, members in groups.items():
            for depth, si in enumerate(members):
                ax.plot(meds[si], pos, marker="o",
                        markersize=ring[min(depth, len(ring) - 1)],
                        markerfacecolor="none" if depth < len(members) - 1
                        else SERIES[si],
                        markeredgecolor=SERIES[si], markeredgewidth=2.0,
                        zorder=3 + depth, linestyle="none")
            # One label per distinct value. Tied arms share it, so it is inked
            # neutral; a lone arm takes its own colour.
            colour = INK if len(members) > 1 else SERIES[members[0]]
            if value <= min(groups) and value < 0.06:
                ax.text(value, pos - 0.24, f"{value:.2f}", va="bottom",
                        ha="left", color=colour, fontsize=10)
            elif value <= min(groups):
                ax.text(value - 0.020, pos, f"{value:.2f}", va="center",
                        ha="right", color=colour, fontsize=10)
            elif value >= max(groups):
                ax.text(value + 0.020, pos, f"{value:.2f}", va="center",
                        ha="left", color=colour, fontsize=10)
            else:
                ax.text(value, pos - 0.24, f"{value:.2f}", va="bottom",
                        ha="center", color=colour, fontsize=10)

    ax.set_yticks(range(1, len(CRITERIA) + 1))
    ax.set_yticklabels([f"{c} {CRITERIA_NAMES[c]}\n({npos[c]}+ / {13 - npos[c]}-)"
                        for c in CRITERIA], fontsize=9.5, color=INK)
    ax.set_ylim(0.4, len(CRITERIA) + 0.6)
    ax.invert_yaxis()
    # Pad both ends so a dot at 0.00 or 1.00 is drawn whole, not clipped by
    # the axis edge (same idea as the notebook's X_PADDING).
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", labelsize=12, colors=INK)
    ax.set_xlabel("Rate", fontsize=13, color=INK)
    ax.set_ylabel("Criterion", fontsize=13, color=INK)
    ax.xaxis.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d1d5db")

    n_runs = [len(s[CRITERIA[0]]["accuracy"]) for s in series]
    spread = (f"{n_runs[0]} runs per config" if len(set(n_runs)) == 1
              else f"{min(n_runs)}-{max(n_runs)} runs per config")
    ax.set_title(f"{dataset}: {title} by criterion  -  {denom}\n{spread}",
                 fontweight="bold", fontsize=13.5, color=INK, pad=34)
    ax.legend(handles=[Line2D([0], [0], color=c, lw=2.4, label=f"{l}  (n={n})")
                       for c, l, n in zip(SERIES, labels, n_runs)],
              loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=min(4, n_series), frameon=False, fontsize=11.5,
              labelcolor=INK, handlelength=1.6, columnspacing=2.4)
    fig.text(0.5, -0.02,
             "Each dot is that arm's MEDIAN across runs; the connector spans "
             "the arms. Arms that tie are drawn as concentric rings.\n"
             "Row label shows ground-truth positives/negatives. " + caveat,
             ha="center", fontsize=9.5, color=MUTED)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / f"{dataset}_{'_vs_'.join(configs)}_{key}_by_criterion.png"
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def generate(dataset: str, configs: List[str], labels: List[str]) -> List[Path]:
    series = [_series(c, dataset) for c in configs]
    if len(series) > len(SERIES):
        raise ValueError(f"palette has {len(SERIES)} validated slots, "
                         f"got {len(configs)} configs")
    gt = metrics._gt_map(dataset)
    npos = {c: sum(gt[c].values()) for c in CRITERIA}

    written = []
    for key, title, denom, caveat in RATES:
        written.append(_one(key, title, denom, caveat, dataset, configs,
                            labels, series, npos))
        # Identity must never be colour-alone.
        print(f"\n{title}")
        print(f"{'criterion':<12}" + "".join(f"{l[:20]:>22}" for l in labels))
        print("-" * (12 + 22 * len(labels)))
        for c in CRITERIA:
            cells = ""
            for s in series:
                v = s[c][key]
                cells += (f"{f'{np.median(v):.2f} (n={len(v)})':>22}"
                          if len(v) else f"{'n/a':>22}")
            print(f"{c + ' (' + str(npos[c]) + '+)':<12}{cells}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--dataset", default="PM")
    a = ap.parse_args()
    if len(a.configs) != len(a.labels):
        ap.error("--configs and --labels must have the same length")
    print()
    for p in generate(a.dataset, a.configs, a.labels):
        print(f"  wrote {p.name}")


if __name__ == "__main__":
    main()
