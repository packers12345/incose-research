"""The human-adjudication record for every `modify` decision, as .txt and .docx.

`reviewer_rates.ADJUDICATED` holds the verdicts and nothing else. This renders
the evidence behind each one — criterion, AI suggestion, reviewer's final text,
their note, the verdict, and what a simple word-counting cross-check would say —
so a reader can audit the labelling against the source rather than take it on
trust. Each decision shows the decision with a short reason, the three texts,
what the SME did with the AI's phrase, and the SME's note.

    python -m reviewer_reliability.adjudication_doc

Run it as a module, not as a loose script: a second copy of this package exists
in the repo's sibling `Requirements-Assistant-main/` tree, and a bare
`python path/to/script.py` resolves the import there instead of here, silently
rendering a stale reviewer set.
"""

from __future__ import annotations

import difflib
import json
import re
import textwrap
from pathlib import Path
from typing import Dict, List

from shared.paths import FEEDBACK_DIR, REVIEWER_RUNS_DIR

from . import reviewer_rates as rr

OUT_TXT = REVIEWER_RUNS_DIR / "PM_modify_adjudication.txt"
OUT_DOCX = REVIEWER_RUNS_DIR / "PM_modify_adjudication.docx"

W = 78

HEAD = [
    ("Figure",   "docs/figures/PM_reviewer_rates_adjudicated.png"),
    ("Scorecard", "docs/figures/PM_reviewer_scorecard.png"),
    ("Code",     "reviewer_reliability/reviewer_rates.py -> ADJUDICATED"),
    ("Command",  "python -m reviewer_reliability.adjudication_doc"),
    ("Ground truth", "data/ground_truth/Point_Mass_v5_GroundTruth.xlsx ('x' = real issue)"),
]

METHOD = (
    "A `modify` is neither an accept nor a reject, but scoring the reviewer as a "
    "binary classifier needs one. All {n} modifies were labelled by hand against the "
    "criterion the AI invoked, the AI's suggested text, the SME's final text and the "
    "SME's note. The test: does the edit accept the premise of the flag as "
    "categorised? Adopting the AI's words for a different reason than the AI gave is "
    "not an endorsement."
)

AUTOCHECK = (
    "Every hand label in the tables below is shown next to "
    "the answer a simple automatic rule would have given, purely as a sanity check on "
    "the hand labelling. The automatic rule is: look at the words the AI ADDED to the "
    "requirement, then see how many of them the reviewer kept in their final text. "
    "Kept most of them -> guess ENDORSE. Dropped most -> guess REJECT. It counts words, "
    "so it cannot tell why a reviewer kept a word, and it is fooled when two people "
    "reach the same wording for different reasons, or when a reviewer rewrites the same "
    "fix in their own words. It is NOT used in any published number - the hand labels "
    "are. It agrees with them on {agree} of {n}; the disagreements are listed so you "
    "can see where word-counting and reading diverge."
)

BALANCE = (
    "{clean} of the {n} sit on cells the ground truth calls clean, so the policy "
    "mostly moves items between FP and TN, shifting TNR/FPR more than TPR/FNR. The "
    "{viol} on GT-violation cells move TP against FN and so bear directly on TPR."
)

CAVEAT = (
    "PROVENANCE CAVEAT. The nine decisions for SME 2, SME 3 and SME 5 were "
    "adjudicated without reference to ground truth. The ten for SME 6 were "
    "adjudicated with each item's GT status visible, and came out perfectly aligned "
    "with it - all five GT-violation modifies endorsed, all five GT-clean modifies "
    "rejected. That is also the pattern anchoring would produce. The earlier nine "
    "show no such alignment: five of their seven GT-clean modifies were endorsed, "
    "producing false positives. SME 6's TNR and TPR should be treated as provisional "
    "until the block is re-labelled blind."
)


READING = (
    "Each decision shows: the decision and a short reason; the original requirement, "
    "the AI's proposed phrase and the SME's final text; what the SME did with the "
    "AI's phrase; and the SME's note. Reasons for the {rec} decisions marked in the "
    "ADJUDICATED comment block in reviewer_rates.py were written when the labels were "
    "set; the other {recon} were added later to explain the existing labels."
)

# Decisions whose reasoning was written down when the label was set (the ADJUDICATED
# comment block in reviewer_rates.py). Used only for the count in READING.
RECORDED = {("6e1e38fc", "10-A2"), ("ffd92981", "12-A5"), ("d566d756", "3-A2"),
            ("d566d756", "4-A2"), ("d566d756", "4-A5"), ("d566d756", "8-A2"),
            ("d566d756", "10-A2")}

# One short reason per decision - under eight words each.
REASON = {
    ("aee204f7", "9-A4"): "Used AI's ambiguity fix word for word.",
    ("aee204f7", "10-A2"): "Adopted AI rewrite; note raises separate issue.",
    ("6e1e38fc", "1-A10"): "Kept the 'shall move' fix.",
    ("6e1e38fc", "2-A10"): "Kept the 'shall move' fix.",
    ("6e1e38fc", "10-A2"): "Fixed ambiguity (A4), not necessity (A2).",
    ("6e1e38fc", "12-A5"): "Kept AI's leak-rate limit.",
    ("ffd92981", "6-A2"): "Kept traceability clause; fixed one word.",
    ("ffd92981", "9-A5"): "Kept AI's start and end points.",
    ("ffd92981", "12-A5"): "Deleted the limit A5 asked for.",
    ("d566d756", "3-A2"): "Style edit for concision, not necessity.",
    ("d566d756", "4-A2"): "Style edit for concision, not necessity.",
    ("d566d756", "4-A5"): "Did not adopt AI's ending condition.",
    ("d566d756", "5-A10"): "Removed 'no more than' in own words.",
    ("d566d756", "6-A10"): "Kept AI's 'minimum of' fix.",
    ("d566d756", "7-A10"): "Replaced 'no more than'; note agrees.",
    ("d566d756", "8-A2"): "Style edit ('be able to'), not necessity.",
    ("d566d756", "8-A10"): "Removed 'no less than' in own words.",
    ("d566d756", "10-A2"): "Fixed ambiguity (A4), not necessity (A2).",
    ("d566d756", "11-A4"): "Used AI's measurable limit word for word.",
}
assert all(len(v.split()) < 8 for v in REASON.values()), "a reason is 8+ words"

_INSTR = re.compile(r"\*([^*]+)\*")


def _instructions(t: str) -> List[str]:
    return [" ".join(m.split()) for m in _INSTR.findall(t or "")]


def _plain(t: str) -> str:
    # Drop typed *instructions* and the UI's highlight markers before comparing.
    return _INSTR.sub(" ", t or "").replace("\u27ea", "").replace("\u27eb", "")


def _norm(w: str) -> str:
    return re.sub(r"[^\w\[\]%]", "", w.lower())


def _contains(hay: List[str], needle: List[str]) -> bool:
    n = len(needle)
    return n > 0 and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def compare(ai: str, sme: str) -> Dict:
    """What the SME did with the AI's phrase."""
    a, b = _plain(ai).split(), _plain(sme).split()
    na = [x for x in (_norm(w) for w in a) if x]
    nb = [x for x in (_norm(w) for w in b) if x]
    out = {"instructions": _instructions(sme), "removed": [], "added": []}
    if na == nb:
        out["kind"] = "Used it word for word."
        return out
    if _contains(nb, na):
        out["kind"] = "Kept it unchanged, inside a longer rewrite."
        return out
    wa, wb = [_norm(w) for w in a], [_norm(w) for w in b]
    sm = difflib.SequenceMatcher(a=wa, b=wb, autojunk=False)
    kept = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            kept += sum(1 for w in wa[i1:i2] if w)
        if tag in ("replace", "delete"):
            chunk = " ".join(a[i1:i2]).strip()
            if _norm(chunk):
                out["removed"].append(_clip(chunk, 140))
        if tag in ("replace", "insert"):
            chunk = " ".join(b[j1:j2]).strip()
            if _norm(chunk):
                out["added"].append(_clip(chunk, 140))
    share = kept / max(1, len(na))
    if share >= 0.7:
        out["kind"] = f"Kept most of it ({share:.0%} of its words)."
    elif share >= 0.3:
        out["kind"] = f"Kept part of it ({share:.0%} of its words)."
    else:
        out["kind"] = f"Wrote their own ({share:.0%} of its words kept)."
    out["removed"], out["added"] = out["removed"][:6], out["added"][:6]
    return out


def _txt_detail(r: Dict) -> List[str]:
    c = r["cmp"]
    L = [f"  My decision: {r['adj'].upper()} - {r['reason']}", "",
         "  Original requirement", _wrap(r["orig"], "    "),
         "  AI's proposed phrase", _wrap(r["ai"], "    "),
         "  SME's final text", _wrap(r["user"], "    "),
         "", "  What the SME did with the AI's phrase", _wrap(c["kind"], "    ")]
    for x in c["removed"]:
        L.append(_wrap(f"- dropped: \"{x}\"", "    "))
    for x in c["added"]:
        L.append(_wrap(f"- added: \"{x}\"", "    "))
    for x in c["instructions"]:
        L.append(_wrap(f"- typed instruction: {x}", "    "))
    L += ["", "  SME note", _wrap(r["notes"] or "No note written.", "    ")]
    return L


def _docx_detail(doc, r: Dict) -> None:
    from docx.shared import Inches, Pt, RGBColor

    GREY = RGBColor(0x6B, 0x72, 0x80)
    GREEN = RGBColor(0x1E, 0x7B, 0x3A)
    RED = RGBColor(0xB2, 0x22, 0x22)

    def para(indent=0.2, after=2, before=0):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(indent)
        p.paragraph_format.space_after = Pt(after)
        p.paragraph_format.space_before = Pt(before)
        return p

    def add(p, text, size=10, bold=False, italic=False, mono=False, colour=None):
        run = p.add_run(text)
        run.font.size = Pt(size)
        run.bold, run.italic = bold, italic
        if mono:
            run.font.name = "Consolas"
            run.font.size = Pt(size - 0.5)
        if colour is not None:
            run.font.color.rgb = colour

    c = r["cmp"]
    p = para(after=6)
    add(p, "My decision:  ", size=11, bold=True)
    add(p, r["adj"].upper(), size=11, bold=True,
        colour=GREEN if r["adj"] == "endorse" else RED)
    add(p, f"  \u2014  {r['reason']}", size=11)

    for label, val in (("Original requirement", r["orig"]),
                       ("AI's proposed phrase", r["ai"]),
                       ("SME's final text", r["user"])):
        p = para(after=0)
        add(p, label, bold=True)
        p = para(indent=0.4, after=4)
        add(p, val, mono=True)

    p = para(after=0, before=2)
    add(p, "What the SME did with the AI's phrase:  ", bold=True)
    add(p, c["kind"])
    for lead, items in (("dropped", c["removed"]), ("added", c["added"])):
        for x in items:
            b = para(indent=0.4, after=0)
            add(b, f"- {lead}:  ", colour=GREY)
            add(b, x, mono=True)
    for x in c["instructions"]:
        b = para(indent=0.4, after=0)
        add(b, "- typed instruction:  ", colour=GREY)
        add(b, x, mono=True)

    p = para(after=10, before=4)
    add(p, "SME note:  ", bold=True)
    if r["notes"]:
        add(p, r["notes"])
    else:
        add(p, "No note written.", italic=True, colour=GREY)


def _cell(endorsed: bool, truth: bool) -> str:
    return ("TP" if truth else "FP") if endorsed else ("FN" if truth else "TN")


def _clip(t: str, n: int = 190) -> str:
    t = " ".join((t or "").split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " ..."


def _wrap(t: str, ind: str = "  ") -> str:
    return textwrap.fill(" ".join(t.split()), W, initial_indent=ind,
                         subsequent_indent=ind)


def collect() -> List[Dict]:
    gt = rr.load_gt()
    recs: List[Dict] = []
    for rid in rr.REVIEWERS:
        data = json.loads((FEEDBACK_DIR / f"feedback_{rid}.json").read_text(encoding="utf-8"))
        for rq in data["requirement_feedback"]:
            req = rr.REQ_TO_PM.get(int(rq["req_id"]))
            for v in rq.get("violation_feedback", []):
                if v.get("user_action") != "modify":
                    continue
                vid, crit = v["violation_id"], v["rule_id"]
                if req is None or crit not in gt:
                    continue
                adj = rr.ADJUDICATED[(rid, vid)]
                if (rid, vid) not in REASON:
                    raise KeyError(f"no reason for {rid} {vid}; add it to REASON")
                orig = " ".join(rq.get("original_text", "").split())
                ai = " ".join(v.get("ai_suggestion", "").split())
                user = " ".join(v.get("user_text", "").split())
                spl, spl_why = rr.modify_polarity(v.get("ai_suggestion", ""),
                                                  v.get("user_text", ""),
                                                  rq.get("original_text", ""))
                truth = bool(gt[crit].get(req, False))
                recs.append(dict(
                    rid=rid, label=rr.LABEL[rid], vid=vid, req=req, crit=crit,
                    cname=v.get("criterion_name", ""), truth=truth,
                    orig=orig, ai=ai, user=user,
                    notes=" ".join((v.get("notes") or "").split()),
                    adj=adj, spl=spl, spl_why=spl_why,
                    cell=_cell(adj == "endorse", truth),
                    reason=REASON[(rid, vid)], cmp=compare(ai, user)))
    return recs


def render(recs: List[Dict]):
    n = len(recs)
    agree = sum(1 for r in recs if r["adj"] == r["spl"])
    clean = sum(1 for r in recs if not r["truth"])
    n_end = sum(1 for r in recs if r["adj"] == "endorse")
    method = METHOD.format(n=n, agree=agree)
    autocheck = AUTOCHECK.format(n=n, agree=agree)
    balance = BALANCE.format(clean=clean, n=n, viol=n - clean)
    n_rec = sum(1 for r in recs if (r["rid"], r["vid"]) in RECORDED)
    reading = READING.format(rec=n_rec, recon=n - n_rec, n=n)

    by_rev = {}
    for r in recs:
        by_rev[r["label"]] = by_rev.get(r["label"], 0) + 1
    tally = (f"{n} modifies: {n_end} endorse, {n - n_end} reject. "
             f"The auto-check agrees on {agree} of {n}. "
             + ", ".join(f"{k} = {v}" for k, v in sorted(by_rev.items()))
             + "; the remaining reviewers logged none.")

    cols = ["#", "reviewer", "session", "id", "req", "crit", "GT", "verdict",
            "cell", "auto-check"]
    rows = [[str(i), r["label"], r["rid"], r["vid"], r["req"], r["crit"],
             "violation" if r["truth"] else "clean", r["adj"], r["cell"], r["spl"]]
            for i, r in enumerate(recs, 1)]

    # ---------------------------------------------------------------- text
    L = ["=" * W,
         "ADJUDICATED `modify` DECISIONS - SME reviewer reliability (Point Mass v5)",
         "=" * W, ""]
    L += [f"{k + ':':<14}{v}" for k, v in HEAD]
    L += ["", "METHOD", _wrap(method), "",
          "WHAT 'AUTO-CHECK' MEANS", _wrap(autocheck), "",
          "TALLY", _wrap(tally), "",
          _wrap(balance), "", _wrap(CAVEAT), "",
          "HOW TO READ EACH DECISION", _wrap(reading), ""]

    widths = [max(len(c), *(len(r[i]) for r in rows)) + 2 for i, c in enumerate(cols)]
    L += ["-" * W,
          "".join(c.ljust(w) for c, w in zip(cols, widths)).rstrip(),
          "-" * W]
    L += ["".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
    L += ["", "  GT         = does v5 ground truth mark this requirement x criterion",
          "               cell as a real issue?",
          "  verdict    = the HAND label. This is what the published rates use.",
          "  cell       = resulting confusion-matrix cell (TP / FP / FN / TN).",
          "  auto-check = what the word-counting rule would have guessed. Not used",
          "               in any published number; shown only as a cross-check.", ""]
    L += [f"WHERE THE HAND LABEL AND THE AUTO-CHECK DISAGREE  ({n - agree} of {n})",
          "  The hand label wins in every case; these are listed so the divergence is",
          "  visible rather than hidden."]
    for r in recs:
        if r["adj"] != r["spl"]:
            L.append(f"  {r['label']:<6} {r['vid']:<7} hand={r['adj']:<8} "
                     f"auto-check={r['spl']:<8} ({r['spl_why']})")
    L += ["", "=" * W, f"THE {n} DECISIONS", "=" * W]

    for i, r in enumerate(recs, 1):
        L += ["",
              f"[{i}] {r['label']}  {r['vid']}  {r['req']} x {r['crit']} ({r['cname']})",
              "-" * W, ""] + _txt_detail(r)
    L += ["", "=" * W, ""]

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(L), encoding="utf-8")

    # ---------------------------------------------------------------- docx
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.9)
        s.top_margin = s.bottom_margin = Inches(0.8)

    doc.add_heading("Adjudicated `modify` decisions", level=0)
    doc.add_paragraph("SME reviewer reliability - Point Mass v5").runs[0].italic = True

    t = doc.add_table(rows=0, cols=2)
    t.style = "Table Grid"
    for k, v in HEAD:
        c = t.add_row().cells
        c[0].text = k
        c[0].paragraphs[0].runs[0].bold = True
        c[1].text = v
        for run in c[1].paragraphs[0].runs:
            run.font.name = "Consolas"
            run.font.size = Pt(9)

    doc.add_heading("Method", level=1)
    doc.add_paragraph(method)
    doc.add_heading("What 'auto-check' means", level=2)
    doc.add_paragraph(autocheck)
    doc.add_heading("Tally", level=1)
    doc.add_paragraph(tally)
    doc.add_paragraph(balance)
    cav = doc.add_paragraph(CAVEAT)
    for run in cav.runs:
        run.italic = True
        run.font.color.rgb = RGBColor(0xB2, 0x22, 0x22)
    doc.add_heading("How to read each decision", level=1)
    doc.add_paragraph(reading)

    doc.add_heading("Summary", level=1)
    tb = doc.add_table(rows=1, cols=len(cols))
    tb.style = "Light Grid Accent 1"
    for i, c in enumerate(cols):
        tb.rows[0].cells[i].text = c
    for row, r in zip(rows, recs):
        cells = tb.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = val
            run = cells[i].paragraphs[0].runs[0]
            run.font.size = Pt(9)
            if i in (7, 9) and val == "reject":
                run.bold = True
        if r["spl"] != r["adj"]:
            for c in cells:
                c.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xB2, 0x22, 0x22)
    note = doc.add_paragraph()
    run = note.add_run("GT = does v5 ground truth mark this requirement x criterion "
                       "cell as a real issue.  cell = resulting confusion-matrix cell. "
                       " auto-check = what a word-counting rule would have guessed, "
                       "shown only as a cross-check and used in no published number.  "
                       "Rows in red are where the hand label and the auto-check diverge.")
    run.font.size = Pt(9)
    run.italic = True

    doc.add_heading(f"The {n} decisions", level=1)
    for i, r in enumerate(recs, 1):
        doc.add_heading(f"{i}. {r['label']}  {r['vid']}  -  {r['req']} x {r['crit']} "
                        f"({r['cname']})", level=2)
        _docx_detail(doc, r)

    doc.save(OUT_DOCX)


def main() -> None:
    recs = collect()
    render(recs)
    n_end = sum(1 for r in recs if r["adj"] == "endorse")
    agree = sum(1 for r in recs if r["adj"] == r["spl"])
    print(f"{len(recs)} modifies across {len(rr.REVIEWERS)} reviewers: "
          f"{n_end} endorse, {len(recs) - n_end} reject; auto-check agrees on {agree}")
    print(f"wrote {OUT_TXT}")
    print(f"wrote {OUT_DOCX}")


if __name__ == "__main__":
    main()
