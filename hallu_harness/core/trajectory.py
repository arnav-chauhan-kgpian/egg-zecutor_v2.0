"""Writing trajectories to disk (schema v2, DESIGN.md §11).

    <output_dir>/<run_id>/
      run.json                          manifest: resolved config, environment, running totals
      index.jsonl                       one line per finished trajectory (append-only)
      tasks/<slug>__s<k>.json           complete trajectory, written once, atomically
      tasks/<slug>__s<k>.tokens.npz     per-step token ids, log-probs, entropy, top-k
      tasks/<slug>__s<k>.partial.json   exists only while running, or after a hard kill
      lens/                             logit-lens readouts (`run.lens`, or `python -m hallu_harness lens`)

Every write goes through a temporary file and `os.replace`, so a reader (or
`sync.sh`) never sees a half-written file and a crash mid-write cannot damage
one already written. The partial checkpoint is rewritten after every step:
a host-RAM OOM kills the process without running any `finally`, and the
checkpoint is then the only record of the steps that completed.

Resume skips a trajectory only if it was completed under the same
*experiment key* — the hash of the resolved config minus the settings that
cannot change what a model is shown (verbosity, output directory, resume),
plus the grading data, the prompt revision and the resolved model and
framework versions (`experiment_key`). Runs of different frameworks or
models can share an output directory without skipping each other's work.
"""
from __future__ import annotations

import io
import json
import os
import re
import socket
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import ResolvedConfig
from .protocol import PROMPT_REVISION
from .provenance import environment
from .recorder import StepRecorder
from .types import Outcome, Step, Task

SCHEMA_VERSION = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


def run_labels(cfg: ResolvedConfig) -> tuple[str, str, str]:
    """The framework, dataset and model, as they name runs and logs."""
    return cfg.framework, slug(cfg.dataset_preset), slug(cfg.model.id.split("/")[-1])


def new_run_id(cfg: ResolvedConfig, started: datetime | None = None) -> str:
    """`<UTC start time>-<host>-<framework>-<dataset>-<model>`: the run's
    directory under `output_dir`."""
    stamp = (started or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    host = slug(socket.gethostname().split(".")[0])
    return "-".join([stamp, host, *run_labels(cfg)])


# Selection fields only choose which tasks run; none changes what one
# trajectory contains (its id names the task and sample). Leaving them out of
# the key lets a run be widened — more tasks, more samples — and resumed
# without redoing finished trajectories.
_SELECTION_ONLY = {"ids", "contest", "min_rating", "max_rating", "tags", "eligible_only", "limit", "shuffle", "seed", "excluded_tasks"}


# The facts of a backend's `info` that can change what a model generates even
# under an identical config: the weights and tokenizer actually resolved (a
# preset with `revision: null` follows the hub), the chat template, numerics,
# the decoding settings in effect, and the library versions.
KEY_MODEL_FIELDS = ("revision", "tokenizer_revision", "tokenizer", "chat_template_sha256", "dtype", "quantization",
                    "context_window", "sampling_effective", "versions")


def software_identity(model_info: dict, framework_info: dict) -> dict:
    """The part of the experiment key that comes from the loaded model and
    the installed framework rather than from the config."""
    return {
        "framework_version": framework_info.get("version"),
        "model": {k: model_info[k] for k in KEY_MODEL_FIELDS if k in model_info},
    }


def experiment_key(cfg: ResolvedConfig, grading_inputs: dict | None = None, software: dict | None = None) -> str:
    """What makes two trajectories of the same task and sample comparable:
    the resolved config without display, output and selection settings, plus
    hashes of any data files grading depends on (`Benchmark.grading_inputs`),
    the revision of the text the model is shown (`PROMPT_REVISION`), and the
    resolved model and framework (`software_identity`), so a resumed run
    never mixes trajectories made with different weights, chat templates or
    library versions."""
    body = cfg.to_dict()
    body.pop("source", None)
    body.pop("sets", None)
    run = dict(body.get("run", {}))
    for key in ("verbosity", "verbose_input", "output_dir", "resume", "log_file", "log_dir", "samples_per_task", "lens"):
        run.pop(key, None)
    body["run"] = run
    dataset = dict(body.get("dataset", {}))
    dataset["selection"] = {k: v for k, v in (dataset.get("selection") or {}).items() if k not in _SELECTION_ONLY}
    body["dataset"] = dataset
    body["grading_inputs"] = grading_inputs or {}
    body["prompt_revision"] = PROMPT_REVISION
    body["software"] = software or {}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, document: dict) -> None:
    _atomic_write_bytes(path, json.dumps(document, indent=2, ensure_ascii=False).encode("utf-8"))


# ---------------------------------------------------------------- document


def step_tokens(step: Step) -> dict[str, np.ndarray]:
    """The arrays of one step for the token sidecar (empty if not recorded)."""
    g = step.generation
    prefix = f"step_{step.number}"
    out: dict[str, np.ndarray] = {}
    if g.input_token_ids is not None:
        out[f"{prefix}/input_ids"] = np.asarray(g.input_token_ids, dtype=np.int32)
    if g.output_token_ids is not None:
        out[f"{prefix}/output_ids"] = np.asarray(g.output_token_ids, dtype=np.int32)
    if g.output_logprobs is not None:
        out[f"{prefix}/logprobs"] = np.asarray(g.output_logprobs, dtype=np.float32)
    if g.output_entropy is not None:
        out[f"{prefix}/entropy"] = np.asarray(g.output_entropy, dtype=np.float32)
    if g.top_logprobs:
        out[f"{prefix}/top_ids"] = np.asarray([[i for i, _ in row] for row in g.top_logprobs], dtype=np.int32)
        out[f"{prefix}/top_logprobs"] = np.asarray([[v for _, v in row] for row in g.top_logprobs], dtype=np.float32)
    return out


def step_dict(step: Step, tokens_file: str | None) -> dict:
    g = step.generation
    has_tokens = g.input_token_ids is not None or g.output_token_ids is not None
    return {
        "step": step.number,
        "kind": step.kind,
        "model_call": {
            "exact_input": g.exact_input,
            "exact_input_kind": g.exact_input_kind,
            "raw_output": g.raw_output,
            "input_tokens": g.input_tokens,
            "output_tokens": g.output_tokens,
            "hit_length_cap": g.hit_length_cap,
            "finish_reason": g.finish_reason,
            "context_window": g.context_window,
            "reply_budget": g.reply_budget,
            "content_decode": g.content_decode,
            "seed": step.seed,
            "sampling": g.sampling,
            "seconds": round(g.seconds, 3),
            "tokens": {"file": tokens_file, "step_key": f"step_{step.number}"} if (has_tokens and tokens_file) else None,
            "segments": g.segments,
        },
        "parse": step.parse.to_dict(),
        "tool_calls": [record.to_dict() for record in step.tool_calls],
        "framework_feedback": step.framework_feedback,
        "seconds": round(step.seconds, 3),
    }


def build_document(
    *,
    trajectory_id: str,
    run_id: str,
    status: str,
    cfg: ResolvedConfig,
    model_info: dict,
    framework_info: dict,
    tool_specs: list[dict],
    task: Task,
    sample_index: int,
    seed: int,
    recorder: StepRecorder,
    system_prompt: str | None,
    user_query: str | None,
    outcome: Outcome | None,
    environment_info: dict,
    tokens_file: str | None,
    benchmark_info: dict,
    experiment_key: str,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": trajectory_id,
        "run_id": run_id,
        "status": status,
        "recorded_at": _now(),
        "user_query": user_query,
        "system_prompt": system_prompt,
        "dataset": {
            "name": cfg.dataset.benchmark,
            "preset": cfg.dataset_preset,
            "split": cfg.dataset.split,
            "task_id": task.task_id,
            "metadata": task.metadata,
            "config": cfg.dataset.model_dump(),
            "benchmark": benchmark_info,
        },
        "model": {**model_info, "preset": cfg.model_preset},
        "framework": framework_info,
        "tools": tool_specs,
        "sampling": {"sample_index": sample_index, "seed": seed},
        "harness": {
            **environment_info,
            "config_sha256": cfg.sha256,
            "experiment_key": experiment_key,
            "prompt_revision": PROMPT_REVISION,
            "config": cfg.to_dict(),
        },
        "steps": [step_dict(step, tokens_file) for step in recorder.steps],
        "outcome": outcome.to_dict() if outcome is not None else None,
    }


# ---------------------------------------------------------------- writer


class TrajectoryWriter:
    """Writes one run's records."""

    def __init__(self, root: Path, run_id: str, cfg: ResolvedConfig, model_info: dict, framework_info: dict, key: str):
        self.root = root
        self.run_id = run_id
        self.cfg = cfg
        self.environment = environment()
        self.summaries: list[dict] = []
        (root / "tasks").mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "started_at": _now(),
            "finished_at": None,
            "config": cfg.to_dict(),
            "config_sha256": cfg.sha256,
            "experiment_key": key,
            "prompt_revision": PROMPT_REVISION,  # in the key too; here so a run's wording can be read back
            "model": model_info,
            "framework": framework_info,
            "environment": self.environment,
            "totals": {},
            "trajectories": [],
        }
        self._write_manifest()

    @classmethod
    def start(cls, cfg: ResolvedConfig, model_info: dict, framework_info: dict, root: str | Path | None = None,
              key: str | None = None, run_id: str | None = None) -> "TrajectoryWriter":
        run_id = run_id or new_run_id(cfg)
        base = Path(root or cfg.run.output_dir)
        return cls(base / run_id, run_id, cfg, model_info, framework_info, key or experiment_key(cfg))

    # -- paths ------------------------------------------------------------

    def paths(self, task_slug: str, sample_index: int) -> dict[str, Path]:
        stem = f"{task_slug}__s{sample_index}"
        tasks = self.root / "tasks"
        return {
            "json": tasks / f"{stem}.json",
            "tokens": tasks / f"{stem}.tokens.npz",
            "partial": tasks / f"{stem}.partial.json",
            "partial_tokens": tasks / f"{stem}.partial.tokens.npz",
        }

    # -- writing ----------------------------------------------------------

    def write_tokens(self, path: Path, recorder: StepRecorder) -> bool:
        arrays: dict[str, np.ndarray] = {}
        for step in recorder.steps:
            arrays.update(step_tokens(step))
        if not arrays:
            return False
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        _atomic_write_bytes(path, buffer.getvalue())
        return True

    def checkpoint(self, paths: dict[str, Path], document: dict, recorder: StepRecorder) -> None:
        self.write_tokens(paths["partial_tokens"], recorder)
        _atomic_write_json(paths["partial"], document)

    def record(self, paths: dict[str, Path], document: dict, recorder: StepRecorder, summary: dict) -> Path:
        has_tokens = self.write_tokens(paths["tokens"], recorder)
        if not has_tokens:
            for step in document["steps"]:
                step["model_call"]["tokens"] = None
        _atomic_write_json(paths["json"], document)
        for key in ("partial", "partial_tokens"):
            try:
                paths[key].unlink()
            except FileNotFoundError:
                pass
        row = {**summary, "file": paths["json"].name, "experiment_key": self.manifest["experiment_key"]}
        with (self.root / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.summaries.append(row)
        self.manifest["trajectories"] = self.summaries
        self.manifest["totals"] = self._totals()
        self._write_manifest()
        return paths["json"]

    def set_manifest_field(self, key: str, value: Any) -> None:
        self.manifest[key] = value
        self._write_manifest()

    def finish(self) -> Path:
        self.manifest["finished_at"] = _now()
        self.manifest["totals"] = self._totals()
        self._write_manifest()
        return self.root

    def _totals(self) -> dict:
        done = len(self.summaries)
        solved = sum(1 for s in self.summaries if s.get("solved"))
        stops: dict[str, int] = {}
        for s in self.summaries:
            stops[s.get("stop_reason", "?")] = stops.get(s.get("stop_reason", "?"), 0) + 1
        return {
            "trajectories": done,
            "solved": solved,
            "solve_rate": round(solved / done, 4) if done else None,
            "crashed": sum(1 for s in self.summaries if s.get("status") == "crashed"),
            "stop_reasons": stops,
            "seconds": round(sum(s.get("seconds") or 0 for s in self.summaries), 1),
        }

    def _write_manifest(self) -> None:
        _atomic_write_json(self.root / "run.json", self.manifest)


def completed_trajectories(output_dir: str | Path, key: str) -> set[str]:
    """Trajectory ids completed (status `complete`) under experiment `key`,
    across every run in `output_dir`. Read from the append-only indexes, which
    stay valid for a run killed mid-sweep."""
    done: set[str] = set()
    for index in Path(output_dir).glob("*/index.jsonl"):
        for line in index.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # a torn last line from a hard kill
                continue
            if row.get("experiment_key") == key and row.get("status") == "complete":
                done.add(row["trajectory_id"])
    return done


def load_tokens(trajectory_json: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """The token sidecar of a trajectory, as {"step_N": {"input_ids": ..., ...}}."""
    path = Path(trajectory_json)
    sidecar = path.with_name(path.name.replace(".json", ".tokens.npz"))
    out: dict[str, dict[str, np.ndarray]] = {}
    with np.load(sidecar) as data:
        for key in data.files:
            step, field = key.split("/", 1)
            out.setdefault(step, {})[field] = data[key]
    return out
