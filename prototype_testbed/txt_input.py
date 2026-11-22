"""Ingest requirements the way the SPONSOR PROTOTYPE does — from .txt.

Why this module exists
----------------------
`shared/spec.py` reads `data/specs/Project Specifications.xlsx`. That workbook
does not contain the requirements the prototype is given; it contains a
re-authored, cleaned variant of them. Feeding it to the evaluator measures a
different object. Compare PM:

    .txt   1  "The System shall be able to move a point mass from an origin
              point to a fixed destination point. Note 1: ... Note 4: ..."
    xlsx   FR.1  "The System shall be able to move a point mass from an origin
              point at (X, Y, Z) = (0 mm, 0 mm, 101.6 mm) to a ..."

    .txt   3  "The system shall execute FR1 accurately with +/- 2.5mm [0.1?]
              accuracy in every Cartesian direction."
    xlsx   PR.1  "The System shall execute FR.1 with an accuracy of +/-2.5 mm
              (+/-0.1 inches) in each Cartesian direction."

The rewrite folds the Notes into the sentence, repairs the mojibake, normalises
`system`->`System`, removes the `accurately with ... accuracy` redundancy, and
turns the dangling `FR1` reference into a `FR.1` that actually resolves. Those
are edits to A4, A5, A9 and A10 evidence — the very cells being scored.

So this module reproduces the prototype's ingestion byte for byte:

  * read the file as bytes, decode utf-8 with errors='replace'  (main.py does
    exactly this, which is where the U+FFFD in `[0.1?]` comes from)
  * parse with the prototype's own `requirements_parser.parse_requirements`
  * validate with its `validate_requirements`
  * the context file becomes ONE flat string — no Project Context / ConOps
    split, because the prototype has no such split

No cleaning, no mojibake repair, no normalisation. The prototype sees the
artefacts, so the replica must too.

ID mapping
----------
The prototype's parser yields positional IDs (`1`..`13` for PM) because the
.txt is a numbered list. Ground truth is keyed `FR.1`..`RR.1`. The two are the
same requirements in the same order, so they are joined BY POSITION, and the
join is asserted (count must match) and written into the raw dump so a reader
can audit it. A mapping file may be supplied to override.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from shared.paths import SAMPLES_DIR, REPO_ROOT

# The prototype's parser is the authority on how a .txt becomes requirements.
# Import it from a checkout if one is present, else use the vendored copy.
_SPONSOR_BACKEND = Path(
    REPO_ROOT.parent / "Sponsor Version of Prototype"
    / "Requirements-Assistant-main" / "backend")


def _load_parser():
    if _SPONSOR_BACKEND.is_dir():
        sys.path.insert(0, str(_SPONSOR_BACKEND))
        try:
            from requirements_parser import parse_requirements, validate_requirements
            return parse_requirements, validate_requirements, "sponsor-checkout"
        except ImportError:
            pass
    from .vendored_parser import parse_requirements, validate_requirements
    return parse_requirements, validate_requirements, "vendored"


# dataset -> (requirements .txt, context .txt)
TXT_INPUTS = {
    "PM": (SAMPLES_DIR / "PM_requirements.txt", SAMPLES_DIR / "PM_context.txt"),
}


@dataclass
class TxtInput:
    dataset: str
    context: str                      # the context file, verbatim, as one string
    parsed: List[Dict]                # [{id, text}] straight from the parser
    id_map: Dict[str, str]            # parser id -> ground-truth req_id
    parser_source: str = ""
    files: Dict[str, str] = field(default_factory=dict)

    def texts_by_gt_id(self) -> Dict[str, str]:
        return {self.id_map[r["id"]]: r["text"] for r in self.parsed}


def _read_as_prototype(path: Path) -> str:
    """main.py: `content = await file.read()` then `.decode('utf-8', errors='replace')`."""
    return path.read_bytes().decode("utf-8", errors="replace")


def load_txt_input(dataset: str, gt_req_ids: List[str],
                   mapping_path: Path | None = None) -> TxtInput:
    req_path, ctx_path = TXT_INPUTS[dataset]
    parse_requirements, validate_requirements, source = _load_parser()

    raw = _read_as_prototype(req_path)
    parsed = parse_requirements(raw)

    verdict = validate_requirements(parsed)
    if not verdict.get("valid"):
        # The prototype returns HTTP 400 here and analyses nothing. A run that
        # cannot start is a result, not an exception to paper over.
        raise ValueError(f"{dataset}: prototype would reject this file — "
                         f"{verdict.get('error')}")

    if mapping_path and mapping_path.exists():
        id_map = json.loads(mapping_path.read_text(encoding="utf-8"))
        missing = [r["id"] for r in parsed if r["id"] not in id_map]
        if missing:
            raise ValueError(f"{dataset}: mapping file omits parser ids {missing}")
    else:
        if len(parsed) != len(gt_req_ids):
            raise ValueError(
                f"{dataset}: positional join needs equal counts — parser produced "
                f"{len(parsed)} requirements, ground truth has {len(gt_req_ids)} "
                f"({gt_req_ids}). Supply --id-map to join explicitly.")
        id_map = {r["id"]: gt for r, gt in zip(parsed, gt_req_ids)}

    return TxtInput(
        dataset=dataset,
        context=_read_as_prototype(ctx_path),
        parsed=parsed,
        id_map=id_map,
        parser_source=source,
        files={"requirements": str(req_path), "context": str(ctx_path)},
    )


def _main() -> None:
    """Show what the prototype would actually receive, and the GT join."""
    import argparse
    from shared import datasets

    ap = argparse.ArgumentParser(description="Preview .txt ingestion (no API calls)")
    ap.add_argument("--dataset", default="PM")
    ap.add_argument("--diff-spec", action="store_true",
                    help="show where the spec workbook text differs from the .txt")
    args = ap.parse_args()

    ds = datasets.load_dataset(args.dataset)
    ti = load_txt_input(args.dataset, ds.req_ids)

    print(f"parser: {ti.parser_source}")
    print(f"requirements: {ti.files['requirements']}")
    print(f"context: {len(ti.context)} chars, flat (no ConOps split)\n")
    for r in ti.parsed:
        print(f"  {r['id']:>3} -> {ti.id_map[r['id']]:<6} {r['text'][:88]}")

    if args.diff_spec:
        from shared import spec
        sp = spec.load_spec(args.dataset)
        print("\n--- .txt vs Project Specifications.xlsx ---")
        for r in ti.parsed:
            gt_id = ti.id_map[r["id"]]
            xl = sp.requirements.get(gt_id, "")
            if " ".join(r["text"].split()) != " ".join(xl.split()):
                print(f"\n{gt_id}:")
                print(f"  txt  : {r['text'][:150]}")
                print(f"  xlsx : {xl[:150]}")


if __name__ == "__main__":
    _main()
