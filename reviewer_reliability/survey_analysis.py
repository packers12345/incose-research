"""Baseline and Prototype-v1 survey analysis (responses from 2026-06-09 / 06-10 on).

Two different instruments, not one instrument administered twice
----------------------------------------------------------------
The baseline survey measures the respondent - experience, AI familiarity,
attitudes, concerns, openness. The prototype survey measures the tool -
usefulness, trust, ease of use, adoption intent. They share no items, so there
is no before/after delta to compute. What can be done is (a) describe each, and
(b) ask whether a respondent's prior disposition tracks their verdict on the
tool, for the respondents who took both.

Who took both
-------------
Only `logan_fegenbush_lanl` is matchable by reviewer ID. Three further
respondents entered "n/a", so identity has to come from timing:

    2026-07-24  baseline 10:44 -> prototype 12:35   (1h51m apart)
    2026-08-01  baseline 13:04 -> prototype 13:40   (36m apart)

Same-day, same-session ordering with no other respondents that day. Treated as
the same person and flagged as INFERRED everywhere it matters, because it is an
inference and not a recorded fact. `SNL Reviewer-1` and `Sara` have baseline
only; `Keith Rivenbark` has prototype only.

Statistics
----------
n = 6 baseline, 4 prototype, 3 paired. No significance testing is performed and
none should be: at these counts a t-test reports noise with a p-value attached.
Everything below is descriptive, and individual responses are shown rather than
hidden behind a mean.

Reverse-coded items: baseline D11-D13 measure CONCERN, so a high score is a
negative disposition, not a positive one. They are scored separately and never
folded into a single "attitude" total.

    python -m reviewer_reliability.survey_analysis
"""

from __future__ import annotations

import glob
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import openpyxl
import pandas as pd

from shared.paths import REVIEWER_RUNS_DIR, output_dir

SEARCH_DIR = Path(__file__).resolve().parents[2]
BASELINE_GLOB = "INCOSE*Baseline Survey.xlsx"
PROTOTYPE_GLOB = "Prototype*Survey.xlsx"

BASELINE_FROM = datetime(2026, 6, 9)
PROTOTYPE_FROM = datetime(2026, 6, 10)

# Both instruments mix "5 = Strongly Agree" and bare "Strongly agree" forms,
# and case is inconsistent across rows.
LIKERT = {
    "strongly disagree": 1, "disagree": 2, "neutral": 3,
    "agree": 4, "strongly agree": 5,
}

# Baseline D11-D13 measure concern; high = more worried. Kept out of any
# aggregate that treats high as good.
BASELINE_REVERSE = {"D11", "D12", "D13"}

# Timing-inferred identity for respondents who entered "n/a".
INFERRED_PAIRS = {"2026-07-24", "2026-08-01"}


def _score(v) -> Optional[int]:
    if v is None:
        return None
    t = str(v).strip().lower()
    m = re.match(r"^([1-5])\s*=", t)
    if m:
        return int(m.group(1))
    t = re.sub(r"^[1-5]\s*=\s*", "", t)
    return LIKERT.get(t)


def _item_key(header: str) -> str:
    """'C. Attitudes ....9. I believe ...' -> 'C9'"""
    m = re.match(r"\s*([A-E])\.\s*[^.]*\.\s*(\d+)\.", str(header))
    return f"{m.group(1)}{m.group(2)}" if m else str(header)[:12]


def _section(header: str) -> str:
    m = re.match(r"\s*([A-E])\.\s*([^.]+)\.", str(header))
    return f"{m.group(1)}. {m.group(2).strip()}" if m else "other"


def load(pattern: str, cutoff: datetime) -> pd.DataFrame:
    path = glob.glob(str(SEARCH_DIR / pattern))
    if not path:
        raise FileNotFoundError(f"no workbook matching {pattern} in {SEARCH_DIR}")
    ws = openpyxl.load_workbook(path[0], data_only=True)["Sheet1"]
    hdr = [c.value for c in ws[1]]

    rows: List[Dict] = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] is None or r[2] is None or r[2] < cutoff:
            continue
        rec = {"id": r[0], "completed": r[2], "date": f"{r[2]:%Y-%m-%d}",
               "reviewer": str(r[5]).strip() if r[5] else "n/a"}
        for i, h in enumerate(hdr):
            if i < 6 or h is None:
                continue
            s = _score(r[i])
            if s is not None:
                rec[_item_key(h)] = s
            elif isinstance(r[i], str) and len(r[i]) > 25:
                rec.setdefault("_free", []).append((str(h)[:45], r[i]))
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.attrs["headers"] = {_item_key(h): h for h in hdr[6:] if h}
    df.attrs["source"] = Path(path[0]).name
    return df


def _sections(df: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for key, h in df.attrs["headers"].items():
        if key in df.columns:
            out.setdefault(_section(h), []).append(key)
    return out


def _label(row) -> str:
    who = row["reviewer"]
    if who == "n/a":
        star = "*" if row["date"] in INFERRED_PAIRS else ""
        return f"n/a {row['date']}{star}"
    return who


def report() -> str:
    base = load(BASELINE_GLOB, BASELINE_FROM)
    proto = load(PROTOTYPE_GLOB, PROTOTYPE_FROM)
    L: List[str] = []

    def head(t):
        L.extend(["", "=" * 78, t, "=" * 78])

    head("SURVEY ANALYSIS - baseline and prototype v1")
    L += [f"Baseline  : {base.attrs['source']}  n={len(base)}  (from {BASELINE_FROM:%Y-%m-%d})",
          f"Prototype : {proto.attrs['source']}  n={len(proto)}  (from {PROTOTYPE_FROM:%Y-%m-%d})",
          "",
          "Pre-cutoff rows are test submissions (reviewer IDs '3' and 'TT_test1')",
          "and are excluded. n is small: everything here is descriptive, and no",
          "significance test is run because none would mean anything at n=4-6."]

    # ---- respondents -------------------------------------------------------
    head("RESPONDENTS")
    b_ids = {_label(r): r["date"] for _, r in base.iterrows()}
    p_ids = {_label(r): r["date"] for _, r in proto.iterrows()}
    L += [f"  baseline  : {', '.join(b_ids)}",
          f"  prototype : {', '.join(p_ids)}",
          "",
          "  * = identity inferred from same-day timing, not a recorded ID."]
    both = [k for k in b_ids if k in p_ids]
    L += [f"  took both : {', '.join(both) if both else '(none)'}   -> {len(both)} paired"]

    # ---- per-section means -------------------------------------------------
    for df, title, reverse in ((base, "BASELINE", BASELINE_REVERSE),
                               (proto, "PROTOTYPE v1", set())):
        head(f"{title} - section means (1 = strongly disagree, 5 = strongly agree)")
        secs = _sections(df)
        w = max(len(s) for s in secs) + 2
        L.append("  " + "section".ljust(w) + "".join(
            f"{_label(r):>16}" for _, r in df.iterrows()) + "      mean")
        for sec, keys in secs.items():
            vals = []
            line = "  " + sec.ljust(w)
            for _, r in df.iterrows():
                got = [r[k] for k in keys if k in r and pd.notna(r[k])]
                m = sum(got) / len(got) if got else None
                vals += got
                line += f"{m:>16.2f}" if m is not None else f"{'-':>16}"
            line += f"{sum(vals)/len(vals):>10.2f}" if vals else ""
            if any(k in reverse for k in keys):
                line += "   (concern: high = more worried)"
            L.append(line)

    # ---- prototype item detail --------------------------------------------
    head("PROTOTYPE v1 - every item")
    L.append("  " + "item".ljust(58) + "".join(f"{_label(r):>16}" for _, r in proto.iterrows()))
    for key, h in proto.attrs["headers"].items():
        if key not in proto.columns:
            continue
        text = re.sub(r"^\s*[A-E]\.\s*[^.]*\.\s*\d+\.\s*", "", str(h))
        line = f"  {key:<5}{text[:52]:<53}"
        for _, r in proto.iterrows():
            line += f"{r[key]:>16.0f}" if pd.notna(r.get(key)) else f"{'-':>16}"
        L.append(line)

    # ---- paired disposition vs verdict ------------------------------------
    head("PAIRED: prior disposition vs verdict on the tool")
    L += ["  Baseline E (openness to adoption) against prototype D (adoption intent).",
          "  Same person, before and after using it.", ""]
    L.append("  " + "respondent".ljust(22) + "baseline E".rjust(12)
             + "prototype D".rjust(14) + "   shift")
    for who in both:
        b = base[base.apply(lambda r: _label(r) == who, axis=1)].iloc[0]
        p = proto[proto.apply(lambda r: _label(r) == who, axis=1)].iloc[0]
        e = [b[k] for k in ("E14", "E15", "E16") if k in b and pd.notna(b[k])]
        d = [p[k] for k in ("D12", "D13", "D14") if k in p and pd.notna(p[k])]
        if not e or not d:
            continue
        em, dm = sum(e) / len(e), sum(d) / len(d)
        L.append(f"  {who:<22}{em:>12.2f}{dm:>14.2f}   {dm - em:+.2f}")

    # ---- free text ---------------------------------------------------------
    head("PROTOTYPE v1 - free text")
    for _, r in proto.iterrows():
        if "_free" not in r or not isinstance(r["_free"], list):
            continue
        L.append(f"\n  --- {_label(r)} ---")
        for q, a in r["_free"]:
            L.append(f"    {q}")
            for chunk in [a[i:i + 88] for i in range(0, len(a), 88)]:
                L.append(f"        {chunk}")
    return "\n".join(L)


def main() -> None:
    text = report()
    print(text)
    REVIEWER_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = REVIEWER_RUNS_DIR / "survey_analysis.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
