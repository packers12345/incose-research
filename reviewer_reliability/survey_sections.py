"""Pre/post survey section scores: one table + one box plot per survey.

Each respondent's SECTION score is the mean of that section's Likert items
(1 = strongly disagree ... 5 = strongly agree). The table reports the median and
range of those scores across respondents; the box plots show the same scores
with every respondent drawn as a dot (at n = 6 the box is four numbers, so read
the dots).

Items are assigned to sections by ITEM NUMBER, so both export formats work: the
early exports prefix headers with "A. Requirements Engineering Experience.",
the later "(1-8)" / "(1-9)" exports do not.

Perceived risks and concerns is reported RAW: high = more concerned.

    python -m reviewer_reliability.survey_sections
"""

from __future__ import annotations

import glob
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd

from shared.paths import output_dir
from .survey_analysis import SEARCH_DIR, _score, BASELINE_FROM, PROTOTYPE_FROM

OUT_DIR = output_dir("reviewer_reliability") / "survey_sections"

# (section label in the table, short label on the plot, item numbers)
PRE_SECTIONS = [
    ("Requirements engineering experience", "Requirements experience", range(1, 4)),
    ("Familiarity with AI tools", "Familiarity with AI tools", range(4, 7)),
    ("Attitudes toward AI in requirements engineering", "Attitudes toward AI", range(7, 11)),
    ("Perceived risks and concerns", "Concerns about AI", range(11, 14)),
    ("Openness to adoption", "Openness to adoption", range(14, 17)),
]
POST_SECTIONS = [
    ("Perceived usefulness", "Usefulness", range(1, 5)),
    ("Trust in AI verdicts", "Trust in verdicts", range(5, 9)),
    ("Ease of use", "Ease of use", range(9, 12)),
    ("Adoption intent", "Adoption intent", range(12, 15)),
]
PRE_GLOB = "INCOSE*Baseline Survey(1-8).xlsx"
POST_GLOB = "Prototype*Survey(1-9).xlsx"

# Style of the deck figures: tinted box, darker edge and median, black dots.
STYLE = {"pre": dict(fill="#F6E7E4", edge="#A63A32", whisker="#D4A19B"),
         "post": dict(fill="#E4EDF8", edge="#2D5E9E", whisker="#93AFD2")}
INK, MUTED, GRID = "#1a1a1a", "#6b7280", "#e5e7eb"


def _item_no(header) -> int | None:
    """Item number from either header form: '...Tools.4. I have used' or '4. I have used'."""
    m = re.search(r"(?:^|\.)\s*(\d{1,2})\.\s", str(header))
    return int(m.group(1)) if m else None


def load(pattern: str, cutoff: datetime) -> Tuple[pd.DataFrame, str]:
    paths = sorted(glob.glob(str(SEARCH_DIR / pattern)))
    if not paths:
        raise FileNotFoundError(f"no workbook matching {pattern} in {SEARCH_DIR}")
    ws = openpyxl.load_workbook(paths[0], data_only=True)["Sheet1"]
    hdr = [c.value for c in ws[1]]
    cols = {i: _item_no(h) for i, h in enumerate(hdr) if i >= 6 and _item_no(h)}
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] is None or r[2] is None or r[2] < cutoff:
            continue
        rec = {"completed": r[2]}
        for i, n in cols.items():
            s = _score(r[i])
            if s is not None:
                rec[n] = s
        rows.append(rec)
    return pd.DataFrame(rows), Path(paths[0]).name


def section_scores(df: pd.DataFrame, sections) -> pd.DataFrame:
    return pd.DataFrame({lab: df[[i for i in items if i in df.columns]].mean(axis=1)
                         for lab, _, items in sections})


def table(pre: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for survey, scores in (("Pre-study", pre), ("Post-study", post)):
        for k, col in enumerate(scores.columns):
            v = scores[col].dropna()
            rows.append({"Survey": survey if k == 0 else "", "Section": col,
                         "Median": f"{np.median(v):.2f}",
                         "Range": f"{v.min():.2f}–{v.max():.2f}", "n": len(v)})
    return pd.DataFrame(rows)


def boxplot(scores: pd.DataFrame, sections, which: str, path: Path) -> None:
    st = STYLE[which]
    labels = [short for _, short, _ in sections]
    data = [scores[lab].dropna().values for lab, _, _ in sections]
    n = len(data)
    fig, ax = plt.subplots(figsize=(9.0, 1.0 + 0.95 * n))
    pos = np.arange(n, 0, -1)
    ax.boxplot(data, positions=pos, vert=False, widths=0.34, patch_artist=True,
               whis=(0, 100), showfliers=False, manage_ticks=False,
               boxprops=dict(facecolor=st["fill"], edgecolor=st["edge"], linewidth=1.4),
               medianprops=dict(color=st["edge"], linewidth=2.6),
               whiskerprops=dict(color=st["whisker"], linewidth=1.4),
               capprops=dict(linewidth=0))
    for p, v in zip(pos, data):
        # Stack respondents who share a score so every dot stays visible.
        vals, counts = np.unique(np.round(v, 4), return_counts=True)
        for val, c in zip(vals, counts):
            offs = (np.arange(c) - (c - 1) / 2) * 0.09
            ax.scatter([val] * c, p + offs, s=34, color="#2b2b2b", zorder=4)
        ax.text(1.03, p, f"{np.median(v):.2f}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=14, fontweight="bold", color=st["edge"])
    ax.text(1.03, n + 0.62, "median", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=10.5, color=MUTED)

    ax.set_yticks(pos)
    ax.set_yticklabels(labels, fontsize=13, color=INK)
    ax.set_ylim(0.45, n + 0.75)
    ax.set_xlim(0.85, 5.15)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.tick_params(axis="x", labelsize=12, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.9)
    ax.axvline(3, color="#cfd3da", linewidth=1, linestyle=(0, (3, 3)), zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cfd3da")
    ax.set_xlabel("1 = strongly disagree        3 = neutral        5 = strongly agree",
                  fontsize=11, color=MUTED, labelpad=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_docx(tbl: pd.DataFrame, n_pre: int, n_post: int, path: Path) -> None:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    def border(cell, edge, sz=8):
        tcPr = cell._tc.get_or_add_tcPr()
        b = tcPr.find(qn("w:tcBorders"))
        if b is None:
            b = OxmlElement("w:tcBorders"); tcPr.append(b)
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:color"), "000000")
        b.append(e)

    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
    cap = doc.add_paragraph()
    cap.add_run("Table X. ").bold = True
    cap.add_run("Survey section scores. Each respondent's section score is the mean of that "
                "section's items (1 = strongly disagree, 5 = strongly agree; pre-study "
                f"n = {n_pre}, post-study n = {n_post}). Perceived risks and concerns is not "
                "reversed: higher means more concerned.")
    t = doc.add_table(rows=1, cols=4)
    for c, h in zip(t.rows[0].cells, ["Survey", "Section", "Median", "Range"]):
        c.text = ""; c.paragraphs[0].add_run(h).bold = True
        border(c, "top", 12); border(c, "bottom", 6)
    for _, r in tbl.iterrows():
        cells = t.add_row().cells
        for c, v in zip(cells, [r.Survey, r.Section, r.Median, r.Range]):
            c.text = str(v)
    for c in t.rows[-1].cells:
        border(c, "bottom", 12)
    from docx.shared import Inches
    t.autofit = False
    widths = [Inches(1.2), Inches(3.6), Inches(1.0), Inches(1.2)]
    for row in t.rows:
        for c, w in zip(row.cells, widths):
            c.width = w
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(0)
    doc.save(path)


def main() -> None:
    pre_df, pre_src = load(PRE_GLOB, BASELINE_FROM)
    post_df, post_src = load(POST_GLOB, PROTOTYPE_FROM)
    pre = section_scores(pre_df, PRE_SECTIONS)
    post = section_scores(post_df, POST_SECTIONS)
    tbl = table(pre, post)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    boxplot(pre, PRE_SECTIONS, "pre", OUT_DIR / "survey_pre_study_sections_box.png")
    boxplot(post, POST_SECTIONS, "post", OUT_DIR / "survey_post_study_sections_box.png")
    tbl.to_excel(OUT_DIR / "survey_section_table.xlsx", index=False)
    write_docx(tbl, len(pre), len(post), OUT_DIR / "survey_section_table.docx")

    print(f"pre : {pre_src}  n={len(pre)}")
    print(f"post: {post_src}  n={len(post)}")
    print(tbl.to_string(index=False))
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
