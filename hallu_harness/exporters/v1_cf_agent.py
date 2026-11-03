"""Schema v2 -> cf_agent's v1 records, so `analysis/traj_lib.py` and the
existing notebooks run unchanged on new runs (DESIGN.md §11.3).

    python -m hallu_harness export --run trajectories/<run_id> --format v1 --out analysis/runs_v1

One v1 *attempt* per v2 step of kind `action` (planning and answer-from-
memory calls are not attempts, as in cf_agent's smolagents loop):

    iteration        the step number among action steps
    raw_reply        the model's raw output (v1 removed the executed code block;
                     with tool calls the program is inside JSON, so the reply is
                     kept whole)
    code             the "code" argument of the step's first grade_code call
    sentinel         DONE / SORRY for the text protocol, else None
    code_block_count the number of tool calls the step made (cf_agent's own
                     smolagents mapping)
    truncation       "thinking" / "code" when the reply hit the length cap
    samples, hidden  from the grade_code call's private grading detail
    feedback         what the model was told next: the observation(s), or the
                     framework's own feedback when it added something

`convert(document)` does one trajectory in memory; traj_lib uses it to read
v2 runs directly.
"""
from __future__ import annotations

import json
from pathlib import Path

from cf_agent.agent.parsing import sentinel as code_sentinel
from cf_agent.agent.parsing import truncation_reason


def _sample_block(private: dict | None) -> dict:
    visible = (private or {}).get("visible") or []
    results = [
        {
            "index": v["index"],
            "passed": v["passed"],
            "input": v.get("input"),
            "expected_output": v.get("expected_output"),
            "actual_output": v.get("actual_output"),
            "error": v.get("error"),
        }
        for v in visible
        if v.get("index", 0) > 0
    ]
    passed = sum(1 for r in results if r["passed"])
    return {"passed": passed, "total": len(results), "all_passed": bool(results) and passed == len(results), "results": results}


def _hidden_block(private: dict | None) -> dict | None:
    hidden = (private or {}).get("hidden")
    if not hidden:
        return None
    return {
        "verdict": hidden["verdict"],
        "passed": hidden["passed"],
        "total": hidden["total"],
        "note": hidden.get("note", ""),
        "time_limit": hidden.get("time_limit"),
        "max_seconds": hidden.get("max_seconds"),
        "failure_breakdown": hidden.get("failure_breakdown", {}),
        "failed_test_numbers": hidden.get("failed_test_numbers", []),
    }


def _problem(document: dict) -> dict:
    meta = dict(document["dataset"]["metadata"])
    meta.setdefault("id", document["dataset"]["task_id"])
    meta.setdefault("title", meta.get("entry_point", meta["id"]))
    meta.setdefault("rating", None)
    if "examples" not in meta:
        meta["examples"] = [{"input": e["call"], "output": e["expected"]} for e in meta.get("visible_examples", [])]
    return meta


def convert(document: dict) -> dict:
    framework = document["framework"]["name"]
    attempts = []
    for step in document["steps"]:
        if step["kind"] != "action":
            continue
        mc = step["model_call"]
        raw = mc["raw_output"]
        calls = step["tool_calls"]
        grade = next((c for c in calls if c["name"] == "grade_code"), None)
        code = None
        if grade is not None and isinstance(grade.get("arguments"), dict):
            code = grade["arguments"].get("code")
        sentinel = None
        if framework == "text" and step["parse"]["status"] == "ok" and not calls:
            # DONE / SORRY: the step parsed but called nothing. Since
            # PROMPT_REVISION 5 it is a single-word reply; before, a block
            # holding print("DONE").
            from cf_agent.agent.parsing import extract_python_blocks

            from ..frameworks.text.adapter import reply_word

            blocks = extract_python_blocks(raw)
            sentinel = code_sentinel(blocks[-1]) if blocks else reply_word(raw)
            code = blocks[-1] if blocks else None
        private = grade["result"]["private"] if grade is not None else None
        observations = [c["result"]["observation"] for c in calls if c["result"]["observation"] is not None]
        if framework == "text":
            # the text loop sends its own message (a note + the observation, or format feedback)
            feedback = step.get("framework_feedback") or ("\n\n".join(observations) if observations else None)
        else:
            # smolagents wraps observations in "Observation:\n"; traj_lib classifies the tool's own text
            feedback = "\n\n".join(observations) if observations else step.get("framework_feedback")
        attempts.append({
            "iteration": len(attempts) + 1,
            "step": step["step"],
            "raw_reply": raw,
            "raw_reply_chars": len(raw),
            "code": code,
            "code_sha256": None,
            "sentinel": sentinel,
            "code_block_count": len(calls),
            "truncation": (truncation_reason(raw) or "code") if mc["hit_length_cap"] else None,
            "seconds": step["seconds"],
            "samples": _sample_block(private),
            "hidden": _hidden_block(private),
            "feedback": feedback,
            "parse_status": step["parse"]["status"],
        })
    outcome = document["outcome"] or {}
    grade = outcome.get("grade") or {}
    return {
        "schema_version": 1,
        "converted_from": {"schema_version": document["schema_version"], "trajectory_id": document["trajectory_id"], "run_id": document["run_id"]},
        "run_id": document["run_id"],
        "model": document["model"],
        "config": document["harness"].get("config"),
        "problem": _problem(document),
        "prompts": {"system": document["system_prompt"], "initial_user": document["user_query"]},
        "outcome": {
            "problem_id": document["dataset"]["task_id"],
            "rating": _problem(document).get("rating"),
            "framework": framework,
            "solved": outcome.get("solved"),
            "stop_reason": outcome.get("stop_reason"),
            "attempts": len(attempts),
            "hidden_verdict": grade.get("verdict"),
            "hidden_passed": grade.get("passed"),
            "hidden_total": grade.get("total"),
            "seconds": outcome.get("seconds"),
            "final_code": outcome.get("final_solution"),
        },
        "attempts": attempts,
    }


def export_run(run_dir: str | Path, out: str | Path) -> Path:
    run_dir, out = Path(run_dir), Path(out)
    target = out / run_dir.name
    (target / "problems").mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((run_dir / "tasks").glob("*.json")):
        if path.name.endswith(".partial.json"):
            continue
        v1 = convert(json.loads(path.read_text()))
        name = path.name.replace(".json", "") + ".json"
        (target / "problems" / name).write_text(json.dumps(v1, indent=2, ensure_ascii=False))
        rows.append({**v1["outcome"], "file": name})
    (target / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (target / "run.json").write_text(json.dumps({"converted_from": str(run_dir), "problems": rows}, indent=2))
    return target
