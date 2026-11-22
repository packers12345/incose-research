"""One diagnostic figure per pairing of PM arms (A -> B), three panels:

  (a) DIFFERENCE  - pooled F1 of B minus A per criterion, with a bootstrap 95% CI.
                    A filled marker means the CI excludes zero.
  (b) TRADE-OFF   - precision vs recall per criterion, an arrow from A to B, over
                    faint iso-F1 curves. Shows whether F1 moved by catching more
                    violations, by raising fewer false flags, or by trading one
                    for the other.
  (c) WHERE       - criterion x requirement grid. Each number is how much more
                    (or less) often B flags that cell than A, in percentage points.
                    Colour says whether that change moves toward ground truth:
                    blue = B closer, red = A closer. Outlined cells are real
                    violations.

Scoring is the same as criterion_compare: pooled counts, v5 ground truth,
first 100 non-empty runs, team scored over all 91 cells.

    python -m n8n_testbed.criterion_pairings
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np

from shared.paths import output_dir
from reviewer_reliability.reviewer_rates import REQ_TO_PM
from .criterion_compare import (CRITERIA, CRITERION_NAME, MODEL_ARMS, TEAM,
                                load_gt, load_runs, team_runs, counts,
                                pooled, boot_samples, ci)
from .criterion_figures import ARM_STYLE, ARM_LABEL

PAIRINGS = [
    ("Baseline", "Beta1"),
    ("Beta1", TEAM),
    ("Beta1", "CLPlusTop3"),
    ("Baseline", "CLPlusTop3"),
]
SHORT = {"Baseline": "Baseline", "Beta1": "Beta 1", "CLPlusTop3": "CLPlusTop3",
         TEAM: "Human-AI team"}

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list(
    "better_worse", ["#b8312f", "#e88a87", "#f0efec", "#86b6ef", "#184f95"])
LABELS = CRITERIA + ["Overall"]
REQS = list(REQ_TO_PM.values())
N_BOOT = 2000


def load_all():
    runs = {arm: load_runs(p) for arm, p in MODEL_ARMS.items()}
    runs[TEAM], _ = team_runs()
    return runs


def flag_rate(runs) -> np.ndarray:
    """[criterion, requirement] share of runs that flagged the cell."""
    m = np.zeros((len(CRITERIA), len(REQS)))
    for _, g in runs:
        for i, c in enumerate(CRITERIA):
            for j, r in enumerate(REQS):
                m[i, j] += g.get(c, {}).get(r, False)
    return m / len(runs)


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #

def panel_delta(ax, a, b, P, B):
    d = P[b]["F1"] - P[a]["F1"]
    lo, hi = ci(B[b]["F1"] - B[a]["F1"])
    fill, edge = ARM_STYLE[b]
    ys = np.arange(len(LABELS))[::-1]
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    for k, y in enumerate(ys):
        if np.isnan(d[k]):
            continue
        sig = lo[k] > 0 or hi[k] < 0
        ax.plot([lo[k], hi[k]], [y, y], color=edge, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(d[k], y, "o", ms=8, mfc=fill if sig else SURFACE, mec=edge, mew=1.8, zorder=3)
        ax.annotate(f"{d[k]:+.2f}", (hi[k], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    ax.axhspan(-0.5, 0.5, color="#f0efec", zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{l} {CRITERION_NAME.get(l, '')}".strip() for l in LABELS],
                       fontsize=9, color=INK)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.6, len(LABELS) - 0.4)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"← {SHORT[a]} better      F1 difference ({SHORT[b]} − {SHORT[a]})      "
                  f"{SHORT[b]} better →", fontsize=9, color=INK)
    ax.set_title("(a) Is there a real difference?  filled = 95% CI excludes 0",
                 loc="left", fontsize=10.5, color=INK, fontweight="bold")


def panel_tradeoff(ax, a, b, P):
    # iso-F1 curves: precision = F*r / (2r - F)
    r = np.linspace(0.01, 1, 300)
    for F in (0.2, 0.4, 0.6, 0.8):
        p = F * r / (2 * r - F)
        ok = (2 * r - F > 0) & (p <= 1.0) & (p >= 0)
        ax.plot(r[ok], p[ok], color=GRID, lw=1, zorder=0)
        idx = np.where(ok)[0][-1]
        ax.text(r[idx], p[idx] - 0.03, f"F1={F}", fontsize=7.5, color=MUTED, ha="right")
    fa, ea = ARM_STYLE[a]
    fb, eb = ARM_STYLE[b]
    skipped = []
    for k, crit in enumerate(CRITERIA):
        ra, pa = P[a]["TPR"][k], P[a]["Precision"][k]
        rb, pb = P[b]["TPR"][k], P[b]["Precision"][k]
        if np.isnan(pa) or np.isnan(pb):
            skipped.append(crit)
            continue
        ax.annotate("", xy=(rb, pb), xytext=(ra, pa),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3,
                                    shrinkA=5, shrinkB=6, mutation_scale=11), zorder=2)
        ax.plot(ra, pa, "o", ms=7, mfc=SURFACE, mec=ea, mew=1.8, zorder=3)
        ax.plot(rb, pb, "o", ms=8, mfc=fb, mec=eb, mew=1.2, zorder=4)
        ax.annotate(crit, (rb, pb), xytext=(6, 5), textcoords="offset points",
                    fontsize=9, color=INK, fontweight="bold", zorder=5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xlabel("Recall (share of real violations caught)", fontsize=9, color=INK)
    ax.set_ylabel("Precision (share of flags that are real)", fontsize=9, color=INK)
    ax.set_title(f"(b) How it moved  ○ {SHORT[a]} → ● {SHORT[b]}", loc="left",
                 fontsize=10.5, color=INK, fontweight="bold")
    if skipped:
        ax.text(0.0, -0.16, f"Not plotted (one arm never flags it, so precision is undefined): "
                f"{', '.join(skipped)}", transform=ax.transAxes, fontsize=8, color=MUTED)


def panel_where(ax, a, b, R, gt, fig):
    diff = R[b] - R[a]
    truth = np.array([[gt[c][r] for r in REQS] for c in CRITERIA])
    better = np.where(truth, diff, -diff)        # >0: B is closer to ground truth
    im = ax.imshow(better, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    for i in range(len(CRITERIA)):
        for j in range(len(REQS)):
            v = diff[i, j] * 100
            txt = "0" if abs(v) < 0.5 else f"{v:+.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.6,
                    color="white" if abs(better[i, j]) > 0.6 else INK)
            if truth[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       ec=INK, lw=1.6, zorder=3))
    ax.set_xticks(range(len(REQS)))
    ax.set_xticklabels(REQS, fontsize=8.5, color=INK)
    ax.set_yticks(range(len(CRITERIA)))
    ax.set_yticklabels([f"{c} {CRITERION_NAME[c]}" for c in CRITERIA], fontsize=8.5, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(REQS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CRITERIA)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=1.5)
    ax.tick_params(which="minor", length=0)
    ax.set_title(f"(c) Where: change in how often each cell is flagged, {SHORT[b]} − {SHORT[a]} "
                 "(percentage points).  Outlined = real violation.",
                 loc="left", fontsize=10.5, color=INK, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels([f"{SHORT[a]}\ncloser to GT", "no change", f"{SHORT[b]}\ncloser to GT"],
                          fontsize=8, color=INK)
    cb.outline.set_visible(False)


# --------------------------------------------------------------------------- #

def draw(a, b, runs, gt, P, B, R, path: Path):
    fig = plt.figure(figsize=(14, 11.2))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 0.9], width_ratios=[1.05, 1],
                          hspace=0.32, wspace=0.28, left=0.1, right=0.92, top=0.9, bottom=0.07)
    ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        for s in ax.spines.values():
            s.set_visible(False)
    panel_delta(ax1, a, b, P, B)
    panel_tradeoff(ax2, a, b, P)
    panel_where(ax3, a, b, R, gt, fig)

    fig.text(0.02, 0.965, f"{ARM_LABEL[b]}  vs  {ARM_LABEL[a]}", fontsize=15,
             fontweight="bold", color=INK)
    n = lambda arm: f"{len(runs[arm])} {'sessions' if arm == TEAM else 'runs'}"
    fig.text(0.02, 0.94, f"Point Mass, v5 ground truth (30 violations in 91 cells).  "
             f"{SHORT[a]}: {n(a)}.  {SHORT[b]}: {n(b)}.  Pooled counts; bootstrap 95% CIs "
             f"over runs ({N_BOOT} resamples).", fontsize=9.5, color=MUTED)
    if TEAM in (a, b):
        fig.text(0.02, 0.012, "Human-AI team: a cell counts as flagged when the SME endorsed the "
                 "AI's flag (adjudicated modify labels); rejected and never-shown cells count as "
                 "not flagged. With 6 sessions its flag rates move in steps of 17 points.",
                 fontsize=8.3, color=MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(0)
    gt = load_gt()
    runs = load_all()
    C = {arm: counts(r, gt) for arm, r in runs.items()}
    P = {arm: pooled(c) for arm, c in C.items()}
    B = {arm: boot_samples(c, rng, N_BOOT) for arm, c in C.items()}
    R = {arm: flag_rate(r) for arm, r in runs.items()}
    out = output_dir("criterion_compare")
    for a, b in PAIRINGS:
        name = f"PM_pair_{b}_vs_{a}".replace("Human-AI team", "Team").replace(" ", "")
        path = out / f"{name}.png"
        draw(a, b, runs, gt, P, B, R, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
