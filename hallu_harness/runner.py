"""Running a resolved config: every task x sample, one trajectory each.

The model is loaded once; everything else is built fresh per trajectory — the
task environment, the tools bound to it, the recorder and the framework's
agent — so nothing leaks between tasks. After every step a partial
checkpoint is written; after the run the final solution is graded and the
complete record is written atomically. With `run.lens` on, each step's
logit-lens readouts are computed as soon as its model call returns and
written to `<run>/lens/` (`interp/lens.py`, `OnlineLens`).

Statuses in `index.jsonl`:
  complete  the trajectory ended for a reason the harness understands
            (final answer, budget, context exhausted, ...); never re-run
  error     it ended because a model call or the framework failed (a CUDA
            OOM, a provider outage); re-run on resume
  crashed   the harness itself failed: setting up the task (fetching its
            test data), inside a tool, or grading the final solution;
            re-run on resume

A harness failure is never shown to the model. Inside a tool it is raised as
`HarnessAbort`, which the frameworks do not catch (`core/errors.py`).
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any, TextIO

from .core import registry
from .core.config import ResolvedConfig
from .core.console import Console
from .core.errors import ConfigError, HarnessAbort
from .core.recorder import StepRecorder
from .core.seeds import derive_seed
from .core.trajectory import (
    TrajectoryWriter,
    build_document,
    completed_trajectories,
    experiment_key,
    slug,
    software_identity,
)
from .core.types import FrameworkResult, Outcome


def build_backend(cfg: ResolvedConfig):
    return registry.get("backend", cfg.model.backend)(cfg.model, cfg.run.recording)


def _first_message(generation, role: str) -> str | None:
    for message in generation.messages or []:
        if message.get("role") == role:
            return message.get("content")
    return None


class Runner:
    def __init__(self, cfg: ResolvedConfig, backend: Any = None, stream: TextIO | None = None, output_dir: str | Path | None = None,
                 run_id: str | None = None):
        self.cfg = cfg
        self.stream = stream or sys.stdout
        self.console = Console(cfg.run.verbosity, cfg.run.verbose_input, self.stream)
        self.backend = backend if backend is not None else build_backend(cfg)
        self.benchmark = registry.get("benchmark", cfg.dataset.benchmark)(cfg.dataset)
        self.adapter_cls = registry.get("framework", cfg.framework)
        self.framework_info = self.adapter_cls.framework_info(cfg.params)
        self.tool_classes = [registry.get("tool", name) for name in cfg.tool_names]
        self.output_dir = Path(output_dir or cfg.run.output_dir)
        self.experiment_key = experiment_key(cfg, self.benchmark.grading_inputs,
                                             software_identity(self.backend.info, self.framework_info))
        self.writer = TrajectoryWriter.start(cfg, self.backend.info, self.framework_info, root=self.output_dir,
                                             key=self.experiment_key, run_id=run_id)
        self.lens = self._start_lens()

    def _start_lens(self):
        """The online logit lens (`run.lens`), or None. It needs the model in
        this process, so other backends run without it, and say so."""
        if not self.cfg.run.lens.enabled:
            return None
        info = self.backend.info
        model = getattr(self.backend, "model", None)
        if info.get("backend") != "hf_local" or model is None:
            reason = f"the {info.get('backend')} backend runs no model in this process"
            self.writer.set_manifest_field("lens", {"online": False, "reason": reason})
            self.console.note(f"[lens] off: {reason}")
            return None
        from .interp.lens import LensStructureError, OnlineLens

        try:
            lens = OnlineLens(model, self.writer.root / "lens", {k: info.get(k) for k in ("id", "revision", "dtype", "quantization")},
                              self.cfg.run.lens, note=self.console.note)
        except LensStructureError as exc:
            raise ConfigError(f"run.lens: {exc}") from None
        self.writer.set_manifest_field("lens", {"online": True, "dir": "lens"})
        return lens

    # ------------------------------------------------------------------

    def run(self) -> Path:
        cfg = self.cfg
        done = completed_trajectories(self.output_dir, self.experiment_key) if cfg.run.resume else set()
        self.console.note(f"[run] {self.writer.run_id} -> {self.writer.root}")
        try:
            for task in self.benchmark.load():
                for sample in range(cfg.run.samples_per_task):
                    trajectory_id = f"{self.benchmark.name}/{slug(task.task_id)}/s{sample}"
                    if trajectory_id in done:
                        self.console.note(f"[skip] {trajectory_id} already complete under this experiment key")
                        continue
                    self.run_one(task, sample, trajectory_id)
        finally:
            self.writer.finish()
            if self.lens is not None:
                self.lens.close()
        return self.writer.root

    def run_one(self, task, sample: int, trajectory_id: str) -> dict:
        cfg = self.cfg
        seed = derive_seed(cfg.run.seed, self.benchmark.name, task.task_id, sample)
        paths = self.writer.paths(slug(task.task_id), sample)
        tool_params = cfg.dataset.tools.params
        started = time.perf_counter()
        # Setting up fetches the task's test data. If that fails, the
        # trajectory is recorded as crashed before any model call, and the
        # sweep goes on.
        env, tools, tool_specs, setup_error = None, [], [], None
        try:
            env = self.benchmark.make_env(task, self.adapter_cls.protocol, tool_params)
            tools = [cls(env, **tool_params.get(cls.name, {})) for cls in self.tool_classes]
            tool_specs = [t.spec() for t in tools]
        except Exception:  # noqa: BLE001 - recorded as a crashed trajectory
            env, setup_error = None, traceback.format_exc()

        recorder = StepRecorder(seed)

        def document(status: str, outcome: Outcome | None, result: FrameworkResult | None) -> dict:
            first = recorder.steps[0].generation if recorder.steps else None
            system_prompt = _first_message(first, "system") if first else None
            user_query = _first_message(first, "user") if first else None
            if system_prompt is None and result is not None:
                system_prompt = result.system_prompt
            if user_query is None and result is not None:
                user_query = result.user_query
            return build_document(
                trajectory_id=trajectory_id,
                run_id=self.writer.run_id,
                status=status,
                cfg=cfg,
                model_info=self.backend.info,
                framework_info={**self.framework_info, "result": (result.native if result else None)},
                tool_specs=tool_specs,
                task=task,
                sample_index=sample,
                seed=seed,
                recorder=recorder,
                system_prompt=system_prompt,
                user_query=user_query,
                outcome=outcome,
                environment_info=self.writer.environment,
                tokens_file=paths["tokens"].name,
                benchmark_info=self.benchmark.info,
                experiment_key=self.experiment_key,
            )

        def checkpoint(event: str, rec: StepRecorder, payload: Any) -> None:
            if event == "step_end":
                self.writer.checkpoint(paths, document("partial", None, None), rec)

        recorder.listeners = [self.console, checkpoint]
        if self.lens is not None:
            self.lens.start(paths["json"].name[: -len(".json")], {"trajectory_id": trajectory_id, "file": paths["json"].name})
            recorder.listeners.append(self.lens)
        adapter = self.adapter_cls(cfg.params, self.backend, recorder, self.benchmark, cfg.params["prompt_mode"])
        self.console.start_trajectory(
            f"{cfg.framework} | {self.benchmark.name} {task.task_id} | sample {sample}", cfg.params["max_model_calls"]
        )

        status = "complete"
        if env is None:
            status = "crashed"
            result = self._crash_result(setup_error)
        else:
            try:
                result = adapter.run(task, tools, env)
                # A failed model call ends the trajectory in every framework;
                # model_errors also catches one a framework swallowed.
                if result.stop_reason == "error" or recorder.model_errors:
                    status = "error"
            except (Exception, HarnessAbort):  # noqa: BLE001 - a harness failure must not lose the steps recorded so far
                status = "crashed"
                recorder.finish()
                result = self._crash_result(traceback.format_exc())

        grade = {"final_solution": None, "solution_source": "none", "grade": None, "solved": False}
        if env is not None:
            try:
                grade = self.benchmark.final_grade(env, result)
            except Exception:  # noqa: BLE001 - recorded, and the trajectory is re-run on resume
                status = "crashed"
                grade = {**grade, "grading_error": traceback.format_exc()}

        outcome = Outcome(
            final_answer=result.final_answer,
            final_solution=grade["final_solution"],
            solution_source=grade["solution_source"],
            grade=grade["grade"] if "grading_error" not in grade else {"error": grade["grading_error"]},
            solved=bool(grade["solved"]),
            stop_reason=result.stop_reason,
            model_calls=len(recorder.steps),
            tool_calls=recorder.tool_call_count,
            parse_failures=recorder.parse_failure_count,
            seconds=time.perf_counter() - started,
            error=result.error,
            context=recorder.context_exhausted.to_dict() if recorder.context_exhausted else None,
            model_errors=list(recorder.model_errors),
        )
        doc = document(status, outcome, result)
        summary = {
            "trajectory_id": trajectory_id,
            "task_id": task.task_id,
            "sample": sample,
            "status": status,
            "solved": outcome.solved,
            "stop_reason": outcome.stop_reason,
            "solution_source": outcome.solution_source,
            "model_calls": outcome.model_calls,
            "tool_calls": outcome.tool_calls,
            "parse_failures": outcome.parse_failures,
            "length_cap_steps": sum(1 for s in recorder.steps if s.generation.hit_length_cap),
            "seconds": round(outcome.seconds, 3),
            "framework": cfg.framework,
            "dataset": cfg.dataset_preset,
            "model": cfg.model.id,
            "error": (result.error or "")[:500] or None,
        }
        if self.lens is not None:
            self.lens.finish()  # before the record, so a trajectory indexed as done has its lens file
        self.writer.record(paths, doc, recorder, summary)
        self.console.end_trajectory(summary)
        if hasattr(self.backend, "clear_cache"):
            self.backend.clear_cache()
        return summary

    def _crash_result(self, error: str | None) -> FrameworkResult:
        return FrameworkResult(
            final_answer=None,
            stop_reason="error",
            system_prompt=None,
            user_query=None,
            framework_version=str(self.framework_info.get("version")),
            error=error,
        )
