"""Schema v2 -> the AgentHallu "history" format (the shape
`smolagent_AgentHallu/trajectory_convert.py` produced), for tools and
labelling pipelines built around that benchmark.

    {"model_id", "agent_type", "question", "question_source", "question_domain",
     "history": [{"step", "content", "tool_calls": [{"name", "arguments"}], "tool_responses": [...]}],
     "agent_answer"}

`content` is the model's raw output; a step whose reply could not be parsed,
or whose tool call failed, carries "[Error] ..." with what the framework
reported, as AgentHallu's own exporter does. Everything AgentHallu has no
field for (exact inputs, tokens, private grading) is dropped — keep the v2
records as the source of truth.
"""
from __future__ import annotations

import json
from pathlib import Path


def convert(document: dict) -> dict:
    history = []
    for step in document["steps"]:
        entry: dict = {"step": step["step"]}
        content = [step["model_call"]["raw_output"].strip()]
        failed = step["parse"]["status"] in ("failed",) or any(c["status"] not in ("executed",) for c in step["tool_calls"])
        if failed and step.get("framework_feedback"):
            content.append(f"[Error] {step['framework_feedback']}")
        entry["content"] = "\n".join(c for c in content if c)
        if step["tool_calls"]:
            entry["tool_calls"] = [{"name": c["name"], "arguments": c["arguments"]} for c in step["tool_calls"]]
            entry["tool_responses"] = [
                c["result"]["observation"] if c["result"]["observation"] is not None else f"[Error] {c['result']['error']}"
                for c in step["tool_calls"]
            ]
        history.append(entry)
    return {
        "model_id": document["model"].get("id"),
        "agent_type": f"{document['framework']['name']}-{document['framework'].get('agent_class', '')}".rstrip("-"),
        "question": document["user_query"],
        "question_source": document["dataset"]["name"],
        "question_domain": "code",
        "history": history,
        "agent_answer": (document["outcome"] or {}).get("final_answer") or "",
    }


def export_run(run_dir: str | Path, out: str | Path) -> Path:
    run_dir, out = Path(run_dir), Path(out)
    target = out / run_dir.name
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted((run_dir / "tasks").glob("*.json")):
        if path.name.endswith(".partial.json"):
            continue
        (target / path.name).write_text(json.dumps(convert(json.loads(path.read_text())), indent=2, ensure_ascii=False))
    return target
