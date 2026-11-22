"""ICAI v3 — file each induced rule under the criterion its CONTENT governs.

The defect this fixes
---------------------
In v2 a rule's criterion is never decided. It is inherited:

    corpus.py:133      criterion_id = viol["rule_id"]      # the criterion the AI FLAGGED
            |
    icai_v2.py:136     {"principle": p, "criterion": case.record.criterion_id}

The generator is asked only for `{"principles": [...]}`; the criterion is
stapled on afterwards from the source record. So the field answers *"what was
the AI's flag about?"* (provenance) while `inject.load_sme_rules` reads it as
*"what does this rule govern?"* (semantics). Those agree only while the induced
principle stays on the topic the AI originally flagged.

It often does not. The SME sees a flag already filed under a criterion — the
review UI offers Accept / Reject / Modify and no way to say "this is actually
A4" — so when their correction shifts topic, the label cannot follow. Observed
on the studio set: two rules about measurable limits, which are A4.1 material
("any qualitative descriptor has a measurable or quantifiable definition"),
carry A10 because the originating flag was A10.2 (negative statement).

Clustering compounds it. k-means groups by embedding similarity of the
principle text, which is criterion-blind, and the survivor inherits ONE
arbitrary member's criterion:

    icai_v2.py:186     pick = dict(rng.choice(members))
    icai_v2.py:316     max(g, key=lambda m: (m["net"], m["accuracy"]))

and in the Method-A path the tie-break is not even arbitrary:

    icai_simple.py:171 "criterion": sorted(b["criteria"])[0]

`sorted(['A2','A4','A10','A9'])` is `['A10','A2','A4','A9']` — "A10" sorts
first lexicographically, so every mixed cluster there resolves to A10. A
systematic bias, not noise.

What v3 adds
------------
One classification pass over finished rules. For each rule the model is shown
the full rubric (descriptions + sub-rules) and asked which criterion the rule's
CONTENT governs, and whether it is a detection rule at all — the studio set's
A5 entry ("Do not replace concise language with longer phrasings") is guidance
about REWRITES, induced from SMEs rejecting the AI's replacement text, and
belongs under no detection criterion.

Provenance is preserved, never overwritten: `source_criterion` keeps the v2
value so the two can be compared and the divergence rate measured.

Output stays a drop-in rule set — `inject.load_sme_rules` keys off
`r["criterion"]`, which v3 sets to the content criterion.

    python -m icai_v3.classify --rules icai_v2_studio --tag icaiv3_studio
    python -m icai_v3.classify --rules icai_v2_studio --dry-run   # no API calls
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from shared.llm import ensure_env
ensure_env()

from shared import llm                                    # noqa: E402
from shared.paths import RULE_SETS_DIR, RUBRIC_DIR        # noqa: E402

# A rule may legitimately govern no detection criterion.
NOT_A_DETECTION_RULE = "NONE"


def _rubric_block() -> str:
    """Descriptions + sub-rules, so the classifier judges against the real text."""
    data = json.loads((RUBRIC_DIR / "incose_rules.json").read_text(encoding="utf-8"))
    lines = []
    for c in data["individual_criteria"]:
        lines.append(f"{c['criterion_id']} - {c['name']}: {c['description']}")
        for sr in c.get("sub_rules", []):
            lines.append(f"    {sr}")
    return "\n".join(lines)


def _valid_criteria() -> List[str]:
    data = json.loads((RUBRIC_DIR / "incose_rules.json").read_text(encoding="utf-8"))
    return [c["criterion_id"] for c in data["individual_criteria"]]


def _prompt(rules: List[Dict]) -> str:
    listing = "\n".join(
        f'{i}. "{r["rule_text"]}"' for i, r in enumerate(rules))
    return f"""You are classifying expert-derived rules against the INCOSE requirement-quality criteria.

Each rule below was induced from expert reviewer feedback. It currently carries the criterion of the ORIGINAL AI FLAG the expert was responding to, which is not necessarily the criterion the rule itself governs. Your job is to decide, from the rule's own content, which criterion it governs.

The criteria and their sub-rules:

{_rubric_block()}

Rules to classify:
{listing}

For each rule decide:

1. `criterion` - the criterion whose description or sub-rules the rule actually tests. Judge the rule's CONTENT, not what it might have been filed under. If a rule tests measurability of a qualitative descriptor, that is A4 (see A4.1), even if it also mentions phrasing.
2. `sub_rule` - the specific sub-rule id it most closely governs (e.g. "A4.1"), or "" if it governs the criterion generally.
3. `scope` - apply this test: does the rule name a condition an evaluator could FIND IN THE REQUIREMENT TEXT?
   - "detection" if yes. Many of these rules are phrased as a preference ("Prefer X over Y"), but if Y is something you could spot in a requirement, the rule is a detection rule: seeing Y is the trigger to flag. Phrasing as a preference does NOT make it rewrite guidance.
   - "rewrite" ONLY if the rule constrains the wording of the SUGGESTED REPLACEMENT and names no condition in the requirement itself - i.e. you could not use it to decide whether any given requirement has a defect.
4. `confidence` - 0.0 to 1.0.
5. `rationale` - one sentence.

A rule whose scope is "rewrite" governs no detection criterion: set its `criterion` to "{NOT_A_DETECTION_RULE}".

Valid criterion values: {', '.join(_valid_criteria())}, {NOT_A_DETECTION_RULE}.

Return ONLY valid JSON, no markdown:
{{"classifications": [{{"index": 0, "criterion": "A4", "sub_rule": "A4.1", "scope": "detection", "confidence": 0.9, "rationale": "..."}}]}}"""


def classify(rules: List[Dict], dry_run: bool = False,
             model: str = None) -> List[Dict]:
    """Return rules with content_criterion / source_criterion resolved."""
    if not rules:
        return []

    if dry_run:
        out = [{"index": i, "criterion": r.get("criterion", ""), "sub_rule": "",
                "scope": "detection", "confidence": 0.0,
                "rationale": "[dry-run] provenance criterion kept"}
               for i, r in enumerate(rules)]
    else:
        data = llm.extract_json(
            llm.call_llm(_prompt(rules), max_tokens=3000, model=model))
        out = data.get("classifications", [])

    valid = set(_valid_criteria()) | {NOT_A_DETECTION_RULE}
    by_index = {int(c["index"]): c for c in out if "index" in c}

    classified = []
    for i, r in enumerate(rules):
        source = r.get("criterion", "")
        c = by_index.get(i)
        if c is None or c.get("criterion") not in valid:
            # No usable classification: keep provenance rather than invent one,
            # and say so, so a silent fallback is never mistaken for a decision.
            content, sub, scope, conf = source, "", "unclassified", 0.0
            why = "classifier returned nothing usable; provenance retained"
        else:
            content = c["criterion"]
            sub = c.get("sub_rule", "") or ""
            scope = c.get("scope", "detection")
            conf = float(c.get("confidence", 0.0))
            why = c.get("rationale", "")

        new = dict(r)
        new["criterion"] = content          # what inject.load_sme_rules reads
        new["sub_rule_id"] = sub or r.get("sub_rule_id", "")
        extra = dict(r.get("extra", {}))
        extra.update({
            "source_criterion": source,     # provenance, never overwritten
            "content_criterion": content,
            "classification_scope": scope,
            "classification_confidence": conf,
            "classification_rationale": why,
            "reclassified": content != source,
        })
        new["extra"] = extra
        classified.append(new)
    return classified


def purity_report(rules: List[Dict]) -> Dict:
    """How far provenance and content disagree - the size of the v2 defect."""
    moved = [r for r in rules if r["extra"].get("reclassified")]
    rewrite = [r for r in rules
               if r["extra"].get("classification_scope") == "rewrite"]
    return {
        "total": len(rules),
        "reclassified": len(moved),
        "rewrite_scope": len(rewrite),
        "source_distribution": dict(Counter(
            r["extra"].get("source_criterion", "") for r in rules)),
        "content_distribution": dict(Counter(r["criterion"] for r in rules)),
        "moves": [{"rule": r["rule_text"][:70],
                   "from": r["extra"]["source_criterion"],
                   "to": r["criterion"],
                   "scope": r["extra"].get("classification_scope"),
                   "why": r["extra"].get("classification_rationale", "")[:90]}
                  for r in moved],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ICAI v3 - re-file induced rules under the criterion their content governs")
    ap.add_argument("--rules", required=True,
                    help="tag of an existing rule set (rules_group_<tag>.json)")
    ap.add_argument("--tag", default=None,
                    help="output tag (default: <rules>_v3)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="no API calls; keeps provenance so the wiring can be checked")
    ap.add_argument("--drop-rewrite", action="store_true",
                    help="drop rules classified as rewrite guidance. Off by default: "
                         "dropping is destructive and a scope misjudgement silently "
                         "shrinks the rule set, so the report names them and you decide.")
    args = ap.parse_args()

    src = RULE_SETS_DIR / f"rules_group_{args.rules}.json"
    if not src.exists():
        raise SystemExit(f"no rule set at {src}")
    rules = json.loads(src.read_text(encoding="utf-8"))

    classified = classify(rules, dry_run=args.dry_run, model=args.model)
    report = purity_report(classified)

    kept = classified
    if args.drop_rewrite:
        kept = [r for r in classified
                if r["extra"].get("classification_scope") != "rewrite"]

    tag = args.tag or f"{args.rules}_v3"
    out = RULE_SETS_DIR / f"rules_group_{tag}.json"
    out.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"read  {src.name}: {report['total']} rules")
    print(f"  reclassified : {report['reclassified']}")
    print(f"  rewrite scope: {report['rewrite_scope']}"
          f"{' (dropped)' if args.drop_rewrite else ' (kept - pass --drop-rewrite to remove)'}")
    print(f"  source  dist : {report['source_distribution']}")
    print(f"  content dist : {report['content_distribution']}")
    if report["moves"]:
        print("\n  moves:")
        for m in report["moves"]:
            print(f"    [{m['from']:>3} -> {m['to']:>4}] {m['rule']}")
            print(f"          {m['why']}")
    print(f"\nwrote {out.name}  ({len(kept)} rules)")
    print(f"inject with:  --configs {tag}")


if __name__ == "__main__":
    main()
