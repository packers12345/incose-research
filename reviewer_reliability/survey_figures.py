"""Figures for the baseline / prototype-v1 surveys.

At n=6 and n=4 a boxplot's quartiles are an artefact of four numbers, so every
panel plots the individual responses as rings on top of the box and says so in
the caption - the same convention the reviewer-rate figures use. Read the rings.

    python -m reviewer_reliability.survey_figures
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from shared.paths import output_dir, DOCS_FIGURE_DIR
from .survey_analysis import (BASELINE_FROM, BASELINE_GLOB, INFERRED_PAIRS,
                              PROTOTYPE_FROM, PROTOTYPE_GLOB, _label,
                              _section, _sections, load)

PLOT_DIR = output_dir("reviewer_reliability") / "plots" / "survey"

BASE_C = "#B22222"
PROTO_C = "#1F6FEB"
INK = "#1a1a1a"
MUTED = "#6b7280"
LIKERT_TICKS = {1: "1\nstrongly\ndisagree", 2: "2\ndisagree", 3: "3\nneutral",
                4: "4\nagree", 5: "5\nstrongly\nagree"}


def _style(bp, color):
    for b in bp["boxes"]:
        b.set(color=color, facecolor="none", linewidth=1.3)
    for m in bp["medians"]:
        m.set(color=color, linewidth=2.2)
    for part in ("whiskers", "caps"):
        for a in bp[part]:
            a.set(color=color, linewidth=1.0)
    for f in bp["fliers"]:
        f.set(marker="", markersize=0)      # individuals are drawn explicitly


def _panel(rows: List[str], data: Dict[str, List[float]], color: str,
           title: str, sub: str, path: Path, n: int, figsize=(10, 5.6)):
    fig, ax = plt.subplots(figsize=figsize)
    series = [data[r] for r in rows]
    bp = ax.boxplot(series, vert=False, patch_artist=True, widths=0.5,
                    showfliers=False, manage_ticks=False,
                    positions=range(1, len(rows) + 1))
    _style(bp, color)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(series, start=1):
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.plot(vals, [i + j for j in jitter], linestyle="none", marker="o",
                markersize=7, markerfacecolor="none", markeredgecolor="#333333",
                markeredgewidth=1.1, alpha=0.85, zorder=3)
        m = float(np.median(vals))
        ax.text(m, i - 0.36, f"{m:.2f}", ha="center", va="bottom",
                color=color, fontsize=10.5)

    ax.set_yticks(range(1, len(rows) + 1))
    ax.set_yticklabels(rows, fontsize=10.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0.6, 5.4)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels([LIKERT_TICKS[i] for i in (1, 2, 3, 4, 5)], fontsize=9)
    ax.axvline(3, color="#d1d5db", linewidth=1.0, zorder=0)
    ax.xaxis.grid(True, color="#eceff3", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d1d5db")
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=14)
    ax.set_xlabel("Response", fontsize=11, color=INK)

    ax.legend(handles=[Line2D([0], [0], marker="o", linestyle="none",
                              markerfacecolor="none", markeredgecolor="#333333",
                              markersize=7, label=f"one response (n={n})")],
              loc="lower right", frameon=False, fontsize=9.5)
    fig.text(0.5, 0.005, sub, ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def sections_panel(df, color, title, path, note=""):
    secs = _sections(df)
    data = {}
    for sec, keys in secs.items():
        vals = []
        for _, r in df.iterrows():
            got = [r[k] for k in keys if k in r and pd.notna(r[k])]
            if got:
                vals.append(sum(got) / len(got))
        data[sec] = vals
    _panel(list(secs), data, color, title,
           "Box = spread across respondents; rings are individual respondents "
           "(their mean over that section's items). " + note,
           path, len(df), figsize=(10, 1.1 * len(secs) + 2.6))


def items_panel(df, color, title, path):
    rows, data = [], {}
    for key, h in df.attrs["headers"].items():
        if key not in df.columns:
            continue
        text = re.sub(r"^\s*[A-E]\.\s*[^.]*\.\s*\d+\.\s*", "", str(h))
        lab = f"{key}  {text[:56]}"
        vals = [r[key] for _, r in df.iterrows() if pd.notna(r.get(key))]
        if vals:
            rows.append(lab)
            data[lab] = vals
    _panel(rows, data, color, title,
           "Box = spread across respondents; rings are individual respondents. "
           "At n=4 the quartiles are an artefact - read the rings.",
           path, len(df), figsize=(12, 0.52 * len(rows) + 2.8))


def paired_shift(base, proto, path):
    pairs = []
    for _, b in base.iterrows():
        who = _label(b)
        m = proto[proto.apply(lambda r: _label(r) == who, axis=1)]
        if m.empty:
            continue
        p = m.iloc[0]
        e = [b[k] for k in ("E14", "E15", "E16") if k in b and pd.notna(b[k])]
        d = [p[k] for k in ("D12", "D13", "D14") if k in p and pd.notna(p[k])]
        if e and d:
            pairs.append((who, sum(e) / len(e), sum(d) / len(d)))
    if not pairs:
        return

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for who, e, d in pairs:
        ax.plot([0, 1], [e, d], marker="o", markersize=9, linewidth=2.0,
                color=BASE_C if d < e else "#2E7D32", alpha=0.85,
                markerfacecolor="white", markeredgewidth=2.0)
        ax.text(-0.045, e, f"{e:.2f}", ha="right", va="center", fontsize=10, color=INK)
        ax.text(1.045, d, f"{d:.2f}  {who}", ha="left", va="center", fontsize=10, color=INK)

    ax.set_xlim(-0.55, 1.95)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before use\nbaseline E\nopenness to adoption",
                        "After use\nprototype D\nadoption intent"], fontsize=10.5)
    ax.set_ylim(0.8, 5.4)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1 strongly disagree", "2 disagree", "3 neutral",
                        "4 agree", "5 strongly agree"], fontsize=9.5)
    ax.axhline(3, color="#d1d5db", linewidth=1.0, zorder=0)
    ax.yaxis.grid(True, color="#eceff3", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d1d5db")
    ax.set_title("Willingness to adopt, before and after using the prototype",
                 fontsize=13, fontweight="bold", color=INK, pad=14)
    fig.text(0.5, 0.005,
             f"All {len(pairs)} paired respondents declined. Two identities are "
             "inferred from same-day timing, not recorded IDs (marked *).",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def main() -> None:
    base = load(BASELINE_GLOB, BASELINE_FROM)
    proto = load(PROTOTYPE_GLOB, PROTOTYPE_FROM)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    sections_panel(base, BASE_C,
                   f"Baseline survey: reviewer disposition (n={len(base)})",
                   PLOT_DIR / "survey_baseline_sections.png",
                   note="Section D is REVERSE-CODED: high = more concerned.")
    sections_panel(proto, PROTO_C,
                   f"Prototype v1: verdict on the tool (n={len(proto)})",
                   PLOT_DIR / "survey_prototype_sections.png")
    items_panel(proto, PROTO_C,
                f"Prototype v1: every item (n={len(proto)})",
                PLOT_DIR / "survey_prototype_items.png")
    paired_shift(base, proto, PLOT_DIR / "survey_paired_shift.png")

    DOCS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for p in PLOT_DIR.glob("survey_*.png"):
        (DOCS_FIGURE_DIR / p.name).write_bytes(p.read_bytes())
    print(f"\ncopied to {DOCS_FIGURE_DIR}")


if __name__ == "__main__":
    main()
