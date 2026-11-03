"""The HumanEval audit: which tasks can be graded honestly (DESIGN.md §8.2).

    python -m hallu_harness audit-humaneval [--out DIR] [--workers 8]

A task is kept only if all of these hold:

1. EvalPlus's reference solution passes every HumanEval+ test (else the
   published checker or its expected results are broken — HumanEval/32's
   checker unpacks a float with `*`, for example);
2. every docstring example agrees with EvalPlus's reference solution (else
   the model would get contradictory signals);
3. the original HumanEval reference solution passes every HumanEval+ test
   at the grading limits;
4. EvalPlus's reference solution passes the original HumanEval `check()`.

(3) and (4) together say the two datasets agree on what the function does
and on how fast it must be. An original reference that fails (3) only by
timing out is the HumanEval-canonical solution being graded too slow, for a
speed the prompt never asks for, so the task is excluded too. Its detail
then says `time_limit_only`, and whether the original reference returns the
same results given 60 s per input (`agrees_given_generous_limits`), so such
a task can be reinstated deliberately.

Step 1 also measures the reference solution's CPU time on every HumanEval+
input; those times set each input's limit (`floor_s` / `reference_multiplier`) and are
written next to the exclusion list, so every machine grades with the same
limits. Docstring lines that look like examples but could not be parsed are
listed for manual review: the model reads them, the harness cannot check them.
"""
from __future__ import annotations

import concurrent.futures
import gzip
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from ...exec_lab_sandbox import run_program

from .benchmark import (
    EXCLUSIONS_PATH,
    REFERENCE_TIMES_PATH,
    Evaluation,
    HumanEvalEvaluator,
    hidden_report_from_records,
    load_sources,
    task_from_rows,
)

TIMING_LIMIT_S = 60.0
TIMING_TOTAL_S = 3600.0


def audit_task(he_row: dict, hp_row: dict, evaluation: Evaluation) -> dict:
    task = task_from_rows(he_row, hp_row)
    tid, entry = task.task_id, task.payload["entry_point"]
    evaluator = HumanEvalEvaluator(task, evaluation, None)
    count = task.payload["check"].input_count
    reference = hp_row["prompt"] + hp_row["canonical_solution"]
    original = he_row["prompt"] + he_row["canonical_solution"]
    reasons: list[str] = []
    detail: dict = {}

    # 1. the EvalPlus reference on HumanEval+, timed
    records, result = evaluator.run_hidden_records(reference, [TIMING_LIMIT_S] * count, TIMING_TOTAL_S)
    report = hidden_report_from_records(records, count, result, [TIMING_LIMIT_S] * count, no_visible=False)
    times = [0.0] * count
    for record in records:
        if "i" in record:
            times[record["i"]] = float(record.get("t", 0.0))
    if not report.accepted:
        reasons.append("reference_fails_humanevalplus")
        detail["reference_on_humanevalplus"] = report.private()

    # 2. docstring examples vs the EvalPlus reference
    visible = evaluator.run_visible(reference) if task.payload["examples"] else None
    if visible is not None and not visible.all_passed:
        reasons.append("docstring_conflicts_with_reference")
        detail["conflicting_examples"] = [r.private() for r in visible.results if not r.passed]

    # 3. the original HumanEval reference on HumanEval+ (limits from the reference times)
    if report.accepted:
        tl = evaluation.hidden_time_limit
        limits = [max(tl.floor_s, tl.reference_multiplier * t) for t in times]
        records3, result3 = evaluator.run_hidden_records(original, limits, evaluation.hidden_total_timeout_s)
        original_report = hidden_report_from_records(records3, count, result3, limits, no_visible=False)
        if not original_report.accepted:
            reasons.append("original_reference_fails_humanevalplus")
            detail["original_reference_on_humanevalplus"] = original_report.private()
            if {v for v, _ in original_report.breakdown} == {"TIME_LIMIT_EXCEEDED"}:
                # Say whether it is only slow, or also wrong where it finishes.
                records4, result4 = evaluator.run_hidden_records(original, [TIMING_LIMIT_S] * count, TIMING_TOTAL_S)
                generous = hidden_report_from_records(records4, count, result4, [TIMING_LIMIT_S] * count, no_visible=False)
                finished_wrong = {v for v, _ in generous.breakdown} - {"TIME_LIMIT_EXCEEDED"}
                detail["time_limit_only"] = {
                    "agrees_given_generous_limits": generous.accepted,
                    "wrong_where_it_finishes": bool(finished_wrong),
                    "at_generous_limits": generous.private(),
                }

    # 4. the EvalPlus reference on the original HumanEval check()
    program = f"{reference}\n\n{he_row['test']}\n\ncheck({entry})\n"
    outcome = run_program(program, "", timeout=120, memory_mb=evaluation.memory_mb, wall_timeout=120)
    if outcome.exit_code != 0 or outcome.timed_out:
        reasons.append("reference_fails_original_tests")
        detail["reference_on_original"] = {"exit_code": outcome.exit_code, "timed_out": outcome.timed_out,
                                           "stderr_tail": outcome.stderr.strip()[-800:]}

    meta = task.metadata
    return {
        "task_id": tid,
        "reasons": reasons,
        "detail": detail,
        "times": times if report.accepted else None,
        "visible_test_count": meta["visible_test_count"],
        "forms": sorted({e["form"] for e in meta["visible_examples"]}),
        "unparsed": meta["unparsed_docstring_examples"],
        "hidden_test_count": count,
        "reference_seconds_total": round(sum(times), 4),
    }


def main(out: str | None = None, workers: int = 8) -> int:
    he_rows, hp_rows, info = load_sources()
    evaluation = Evaluation()
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(audit_task, a, b, evaluation): a["task_id"] for a, b in zip(he_rows, hp_rows)}
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            results.append(row)
            flag = ", ".join(row["reasons"]) or "ok"
            print(f"[{n:3}/{len(futures)}] {row['task_id']:14} {flag}", flush=True)
    results.sort(key=lambda r: int(r["task_id"].split("/")[1]))

    excluded = {r["task_id"]: {"reasons": r["reasons"], "detail": r["detail"]} for r in results if r["reasons"]}
    kept = [r for r in results if not r["reasons"]]
    out_dir = Path(out) if out else EXCLUSIONS_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "description": "HumanEval tasks excluded because the docstring, the original tests and HumanEval+'s tests do not all agree (DESIGN.md §8.2). Written by `python -m hallu_harness audit-humaneval`.",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": info,
        "checks": [
            "reference_fails_humanevalplus",
            "docstring_conflicts_with_reference",
            "original_reference_fails_humanevalplus",
            "reference_fails_original_tests",
        ],
        "summary": {"tasks": len(results), "kept": len(kept), "excluded": len(excluded),
                    "kept_with_visible_examples": sum(1 for r in kept if r["visible_test_count"]),
                    "kept_visible_examples": sum(r["visible_test_count"] for r in kept)},
        "excluded": excluded,
        "unparsed_docstring_examples_for_review": {r["task_id"]: r["unparsed"] for r in results if r["unparsed"]},
        "excluded_only_for_time": sorted((r["task_id"] for r in results if r["reasons"] == ["original_reference_fails_humanevalplus"]
                                          and "time_limit_only" in r["detail"]), key=lambda t: int(t.split("/")[1])),
    }
    (out_dir / EXCLUSIONS_PATH.name).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    times_doc = {
        "description": "Per-input CPU time (seconds) of EvalPlus's reference solution on every HumanEval+ input, measured by the audit; sets each input's time limit.",
        "measured_on": {"host": platform.node(), "workers": workers, "cpu_count": os.cpu_count(), "per_input_limit_while_timing_s": TIMING_LIMIT_S},
        "sources": info,
        "tasks": {r["task_id"]: r["times"] for r in results if r["times"] is not None},
    }
    with gzip.open(out_dir / REFERENCE_TIMES_PATH.name, "wt", encoding="utf-8") as handle:
        json.dump(times_doc, handle)
    print(json.dumps(doc["summary"], indent=2))
    listing = "; ".join(k + " (" + ", ".join(v["reasons"]) + ")" for k, v in excluded.items())
    print(f"excluded: {listing}")
    return 0
