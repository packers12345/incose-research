"""Per-criterion diagnostics — the recall losses the scored matrix cannot show.

The n8n arm tracks `a1_suppressed_true_flags` and `a11_diverted_reqs`, because
those are the ways n8n loses a true flag before it reaches the matrix. The
prototype has no A1 and no A11; it loses flags four other ways, and each one
lands in the matrix as an ordinary "not violated" cell:

  not_evaluated    the model returned no entry for a criterion, so
                   analyze_requirement backfilled it `satisfied: True` with the
                   explanation "Not evaluated." A skipped criterion and a passed
                   criterion are the same cell.
  error_result     the call raised or the JSON failed to parse, so
                   _error_result marked ALL SEVEN criteria satisfied. One
                   failure injects seven false negatives. Counted per
                   requirement AND per criterion so its weight is visible.
  span_missing     the model called a criterion violated but returned no
                   affected_text, despite the prompt saying it is REQUIRED.
  span_not_found   affected_text is not a substring of the requirement — a
                   hallucinated span. The UI's indexOf() lookup silently
                   degrades to "no change" when this happens.

span_missing and span_not_found are the measurement of the span mandate. The
claim is that requiring a verbatim substring suppresses the criteria whose
defect is an ABSENCE (A2 Necessary, A5 Complete, A9 Correct) relative to those
whose defect is a token that is present (A4, A6, A10). Broken out per criterion,
these counters test that claim instead of assuming it.

`placeholder_fix` counts the evaluator's own "[No suggestion provided — review
manually]" fallback: a violation that survived with no repair attached.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

COUNTERS = ("not_evaluated", "error_result", "span_missing",
            "span_not_found", "placeholder_fix")

_PLACEHOLDER = "[No suggestion provided"
_NOT_EVALUATED = "Not evaluated."


def blank() -> Dict:
    return {
        "totals": {k: 0 for k in COUNTERS},
        "per_criterion": defaultdict(lambda: {k: 0 for k in COUNTERS}),
        "error_requirements": [],
    }


def observe(diag: Dict, result: Dict) -> None:
    """Fold one analyze_requirement() result into the running diagnostics."""
    req_id = result.get("req_id", "?")
    text = result.get("original_text", "") or ""
    errored = bool(result.get("error"))

    if errored:
        diag["error_requirements"].append({"req_id": req_id,
                                           "error": result["error"]})

    for ev in result.get("criteria_evaluations", []):
        cid = ev.get("criterion_id", "?")
        per = diag["per_criterion"][cid]

        if errored:
            _bump(diag, per, "error_result")
            continue

        if ev.get("satisfied", True):
            # A backfilled criterion is indistinguishable from a real pass in
            # the matrix; the explanation string is the only tell.
            if ev.get("explanation", "") == _NOT_EVALUATED:
                _bump(diag, per, "not_evaluated")
            continue

        span = ev.get("affected_text")
        if not span:
            _bump(diag, per, "span_missing")
        elif span not in text:
            _bump(diag, per, "span_not_found")

        if str(ev.get("suggested_replacement") or "").startswith(_PLACEHOLDER):
            _bump(diag, per, "placeholder_fix")


def _bump(diag: Dict, per: Dict, key: str) -> None:
    diag["totals"][key] += 1
    per[key] += 1


def merge(into: Dict, other: Dict) -> None:
    for k in COUNTERS:
        into["totals"][k] += other["totals"][k]
    for cid, counts in other["per_criterion"].items():
        for k in COUNTERS:
            into["per_criterion"][cid][k] += counts[k]
    into["error_requirements"].extend(other["error_requirements"])


def to_plain(diag: Dict) -> Dict:
    """defaultdict -> dict, for json.dumps."""
    return {
        "totals": dict(diag["totals"]),
        "per_criterion": {cid: dict(v) for cid, v in diag["per_criterion"].items()},
        "error_requirements": diag["error_requirements"],
    }


def summary_line(diag: Dict) -> str:
    t = diag["totals"]
    parts = [f"{k}={t[k]}" for k in COUNTERS if t[k]]
    return ", ".join(parts) if parts else "clean"


def report(diag: Dict, criteria: List[str]) -> str:
    """Per-criterion table — the span-mandate evidence."""
    rows = ["", "  per-criterion diagnostics",
            f"  {'crit':<6}" + "".join(f"{k:>16}" for k in COUNTERS)]
    for cid in criteria:
        c = diag["per_criterion"].get(cid, {k: 0 for k in COUNTERS})
        rows.append(f"  {cid:<6}" + "".join(f"{c[k]:>16}" for k in COUNTERS))
    return "\n".join(rows)
