"""Replicate the SPONSOR PROTOTYPE's individual-criteria assessment, N times.

The unit of observation is the (requirement x criterion) decision cell: for each
requirement, does it violate each INCOSE criterion, one at a time. That is what
the prototype produces and what the ground-truth matrix scores, so it is what
this arm reproduces - 100 runs of it, written as Execution_01..Execution_NN
sheets that `n8n_testbed.metrics` reads with no changes.

What makes this the prototype and not a paraphrase
--------------------------------------------------
Two things, and the existing `n8n_testbed.run_matrix --configs baseline` arm
gets both wrong while describing itself as "the prototype as shipped":

1. THE CALL. This module calls `shared.evaluator.analyze_requirement` - the
   verbatim port of the prototype's `ai_analyzer.py`. It does not rebuild the
   prompt. `n8n_testbed/inject.py` rebuilds it, and the rebuild runs
   OUTPUT_MODE="minimal" (a sparse violated-id list, no per-criterion satisfied
   flag, no affected_text, no suggested_replacement), TEMPERATURE=0.7 "raised
   from 0.1 so repeated runs show real variance", max_tokens 500, a split
   cached system prompt, and an extra leniency clause the prototype does not
   have. Sparse elicitation at 0.7 with no span requirement is structurally
   n8n's process, not the prototype's - it removes the three mechanisms that
   make the prototype's per-criterion behaviour distinctive.

2. THE INPUT. This module reads the .txt files through the prototype's own
   parser (see txt_input.py). `run_matrix` reads
   `data/specs/Project Specifications.xlsx`, which holds a re-authored variant:
   for PM all 13 requirements differ, the Notes are folded in or dropped, the
   mojibake is repaired, `system` is normalised to `System`, and the dangling
   `FR1` reference becomes a resolvable `FR.1`. Those edits land on A4, A5, A9
   and A10 evidence - the cells being scored.

Rule-injection configs (M1/M2/ICAI...) still belong in `n8n_testbed.inject`,
which has to modify the criteria block to do its job. The BASELINE must not go
through it.

Faithfully reproduced, including the parts that are defects
-----------------------------------------------------------
  * one call per requirement, ThreadPoolExecutor(min(10, n)) - no set context
  * temperature 0.1, max_tokens 1200, model claude-sonnet-4-5
  * dense output: seven `satisfied` verdicts, one per criterion
  * the span mandate (affected_text + suggested_replacement REQUIRED)
  * a criterion the model omits is backfilled `satisfied: True`
  * _error_result marks all seven satisfied

The last two are how the prototype turns a failure into seven clean cells. They
are kept, but counted (see diagnostics.py) so they are visible in the run rather
than silently scored as "found no issues". Retries are applied to transport
errors only, before the prototype's own error path is accepted; a requirement
that still fails is recorded as the prototype would record it, and counted.

    python -m prototype_testbed.prototype_replica --runs 100 --dataset PM
    python -m n8n_testbed.metrics --datasets PM --configs prototype
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import openpyxl

from shared.llm import ensure_env
ensure_env()

from shared import datasets                                    # noqa: E402
from shared.datasets import CRITERIA, CRITERIA_NAMES           # noqa: E402
from shared.paths import TESTBED_RUNS_DIR as RUNS_DIR, REPO_ROOT, output_dir  # noqa: E402

from . import diagnostics                                      # noqa: E402
from .txt_input import load_txt_input                          # noqa: E402

RUNS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = output_dir("prototype_testbed")

# The prototype pins this in ai_analyzer._call_ai. shared.evaluator.resolve_model
# reads ANTHROPIC_MODEL, so --model is applied by setting that env var rather
# than by editing the ported evaluator.
PROTOTYPE_MODEL = "claude-sonnet-4-5"

# Transport-level retries only. The prototype has none, but a 429 recorded as
# seven satisfied criteria is measurement noise, not prototype behaviour.
MAX_RETRIES = 4


def _sponsor_sha() -> str:
    """Pin what was replicated - 'the prototype' is a moving target."""
    candidates = [
        REPO_ROOT.parent / "Sponsor Version of Prototype" / "Requirements-Assistant-main",
        REPO_ROOT,
    ]
    for candidate in candidates:
        try:
            out = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10)
            if out.returncode == 0:
                return out.stdout.strip()[:12]
        except Exception:  # noqa: BLE001
            continue
    return "unknown"


def _analyze_one(req_id: str, text: str, context: str) -> Dict:
    """One requirement -> the prototype's full result, retrying transport errors.

    `analyze_requirement` catches everything internally and returns
    _error_result, so a retry loop cannot see the exception. Instead the result
    is inspected: an `error` key means the prototype's failure path fired, and
    that is worth one more attempt before it is accepted as data.
    """
    from shared.evaluator import analyze_requirement

    result = None
    for attempt in range(MAX_RETRIES):
        result = analyze_requirement({"id": req_id, "text": text}, context)
        if not result.get("error"):
            return result
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 * (attempt + 1))
    return result


def _run_once(reqs: List[Tuple[str, str]], context: str,
              gt_of: Dict[str, str]) -> Tuple[Dict[str, Dict], Dict]:
    """One full pass. Returns {gt_req_id: result} and this pass's diagnostics."""
    results: Dict[str, Dict] = {}
    diag = diagnostics.blank()

    # The prototype's own fan-out width, on the prototype's own requirement list.
    with ThreadPoolExecutor(max_workers=min(10, len(reqs))) as pool:
        futs = {pool.submit(_analyze_one, rid, txt, context): rid
                for rid, txt in reqs}
        for fut in as_completed(futs):
            parser_id = futs[fut]
            result = fut.result()
            diagnostics.observe(diag, result)
            results[gt_of[parser_id]] = result
    return results, diag


def _violated(result: Dict) -> Dict[str, bool]:
    """The per-criterion decision cell: violated == not satisfied."""
    return {ev["criterion_id"]: not ev.get("satisfied", True)
            for ev in result.get("criteria_evaluations", [])}


def _matrix_sheet(ws, req_ids: List[str], results: Dict[str, Dict]) -> None:
    """Same shape n8n_testbed writes, so metrics.py needs no changes."""
    ws.append(["Rule ID", "Rule Description"] + req_ids)
    for cid in CRITERIA:
        row = [cid, CRITERIA_NAMES[cid]]
        for rid in req_ids:
            row.append("x" if _violated(results.get(rid, {})).get(cid) else None)
        ws.append(row)


def _gt_workbook(ds) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "GT"
    ws.append(["Rule ID", "Rule Description"] + ds.req_ids)
    for cid in CRITERIA:
        ws.append([cid, CRITERIA_NAMES[cid]] +
                  ["x" if ds.gt[cid][rid] else None for rid in ds.req_ids])
    wb.save(RUNS_DIR / f"{ds.name}_gt.xlsx")


def run(dataset: str, n_runs: int, config: str = "prototype",
        model: str = PROTOTYPE_MODEL) -> Path:
    ds = datasets.load_dataset(dataset)
    ti = load_txt_input(dataset, ds.req_ids)

    os.environ["ANTHROPIC_MODEL"] = model

    # Prompt order is the .txt's own order - what the prototype's parser emits.
    # ds.req_ids only orders the matrix columns.
    reqs = [(r["id"], r["text"]) for r in ti.parsed]

    _gt_workbook(ds)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    totals = diagnostics.blank()
    executions = []

    print(f"{dataset}/{config}: {len(reqs)} requirements from "
          f"{Path(ti.files['requirements']).name} via {ti.parser_source} parser, "
          f"{n_runs} runs on {model}")

    for run_idx in range(1, n_runs + 1):
        t0 = time.time()
        results, diag = _run_once(reqs, ti.context, ti.id_map)

        ws = wb.create_sheet(f"Execution_{run_idx:02d}")
        _matrix_sheet(ws, ds.req_ids, results)
        diagnostics.merge(totals, diag)

        executions.append({
            "run": run_idx,
            "evaluations": {rid: r.get("criteria_evaluations", [])
                            for rid, r in results.items()},
            "diagnostics": diagnostics.to_plain(diag),
        })

        n_flags = sum(sum(_violated(r).values()) for r in results.values())
        print(f"  run {run_idx:02d}/{n_runs}: {n_flags} flags, "
              f"{diagnostics.summary_line(diag)}, {time.time() - t0:.0f}s")

    path = RUNS_DIR / f"{dataset}_{config}_matrix.xlsx"
    wb.save(path)
    (RUNS_DIR / f"{dataset}_{config}_raw.json").write_text(
        json.dumps({
            "dataset": dataset,
            "config": config,
            "arm": "sponsor-prototype-replica",
            "input": {
                "mode": "txt",
                "files": ti.files,
                "parser": ti.parser_source,
                "id_map": ti.id_map,
                "context_chars": len(ti.context),
            },
            "evaluator": {
                "module": "shared.evaluator (verbatim ai_analyzer.py port)",
                "model": model,
                "temperature": 0.1,
                "max_tokens": 1200,
                "output": "dense 7-verdict + span mandate",
                "sponsor_sha": _sponsor_sha(),
            },
            "runs": n_runs,
            "req_order": [r for r, _ in reqs],
            "totals": diagnostics.to_plain(totals),
            "executions": executions,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  totals over {n_runs} runs: {diagnostics.summary_line(totals)}")
    print(diagnostics.report(totals, CRITERIA))
    print(f"\n  -> {path.name}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sponsor-prototype replica - per-criterion matrices from .txt input")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--dataset", default="PM")
    ap.add_argument("--config", default="prototype",
                    help="output tag; runs/<dataset>_<config>_matrix.xlsx")
    ap.add_argument("--model", default=PROTOTYPE_MODEL)
    args = ap.parse_args()
    run(args.dataset, args.runs, args.config, args.model)


if __name__ == "__main__":
    main()
