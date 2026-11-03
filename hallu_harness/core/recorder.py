"""The step recorder: the one place a trajectory is assembled.

Bridges and adapters report what they observe as events; the recorder turns
them into `Step`s and notifies listeners (the console and the checkpoint
writer). Both listeners read the very objects the trajectory is built from,
so what is printed and what is stored cannot disagree.

A step is opened by a model call — the engine is the only component that
sees the exact input and output — and everything until the next model call
hangs off it: the parse result, the tool calls in the order the framework
made them, their results, and whatever the framework itself fed back.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Protocol

from .errors import ContextExhausted
from .seeds import derive_seed
from .types import Generation, ParseRecord, Step, ToolCallRecord


class Listener(Protocol):
    def __call__(self, event: str, recorder: "StepRecorder", payload: Any) -> None: ...


class StepRecorder:
    """Collects one trajectory's steps."""

    def __init__(self, trajectory_seed: int, listeners: list[Callable] | None = None):
        self.trajectory_seed = trajectory_seed
        self.listeners: list[Callable] = list(listeners or [])
        self.steps: list[Step] = []
        self.next_kind: str = "action"
        self.context_exhausted: ContextExhausted | None = None
        self.model_errors: list[str] = []
        self._calls = 0

    # -- seeds ------------------------------------------------------------

    def seed_for_next_call(self) -> int:
        """The seed for model call number len(steps)+1 of this trajectory."""
        return derive_seed(self.trajectory_seed, "call", self._calls + 1)

    # -- events -----------------------------------------------------------

    def _emit(self, event: str, payload: Any = None) -> None:
        for listener in self.listeners:
            listener(event, self, payload)

    @property
    def current(self) -> Step | None:
        return self.steps[-1] if self.steps else None

    def on_model_call(self, generation: Generation, seed: int, started: float, kind: str | None = None) -> Step:
        """A model call completed: closes the previous step, opens a new one."""
        if self.current is not None and not self.current.ended:
            self.end_step()
        self._calls += 1
        step = Step(
            number=len(self.steps) + 1,
            kind=kind or self.next_kind,
            generation=generation,
            seed=seed,
            started=started,
        )
        self.next_kind = "action"
        self.steps.append(step)
        self._emit("model_call", step)
        return step

    def on_context_exhausted(self, exc: ContextExhausted) -> None:
        """The engine refused a call because the conversation no longer fits.
        Kept here as well as raised, because some frameworks swallow model
        errors (smolagents' answer-from-memory call does)."""
        self.context_exhausted = exc
        self._emit("context_exhausted", exc)

    def on_model_error(self, message: str) -> None:
        self.model_errors.append(message)
        self._emit("model_error", message)

    def on_parse(self, status: str, error: str | None = None, attempted: bool | None = None) -> None:
        step = self._require_step("parse")
        step.parse = ParseRecord(status=status, error=error, attempted_tool_call=attempted)
        self._emit("parse", step)

    def on_tool_call(
        self,
        name: str,
        arguments: Any,
        raw_arguments: str | None = None,
        call_id: str | None = None,
    ) -> ToolCallRecord:
        """A tool call was parsed. Recorded *before* it runs, so a call that
        raises, names an unknown tool or fails validation is still kept."""
        step = self._require_step("tool_call")
        record = ToolCallRecord(
            index=len(step.tool_calls) + 1,
            name=name,
            arguments=arguments,
            raw_arguments=raw_arguments,
            call_id=call_id,
        )
        step.tool_calls.append(record)
        self._emit("tool_call", record)
        return record

    def on_tool_result(
        self,
        record: ToolCallRecord,
        observation: str | None,
        status: str = "executed",
        private: dict | None = None,
        error: str | None = None,
        seconds: float = 0.0,
    ) -> None:
        record.status = status
        record.observation = observation
        record.private = private or {}
        record.error = error
        record.seconds = seconds
        self._emit("tool_result", record)

    def on_framework_feedback(self, text: str | None) -> None:
        step = self._require_step("framework_feedback")
        step.framework_feedback = text
        self._emit("framework_feedback", step)

    def end_step(self) -> None:
        step = self.current
        if step is None or step.ended:
            return
        step.ended = True
        step.seconds = time.perf_counter() - step.started
        self._emit("step_end", step)

    def finish(self) -> None:
        if self.current is not None and not self.current.ended:
            self.end_step()

    # -- queries ----------------------------------------------------------

    def _require_step(self, event: str) -> Step:
        if self.current is None:
            raise RuntimeError(f"{event} reported before any model call")
        return self.current

    def find_pending(self, name: str | None = None) -> ToolCallRecord | None:
        """The earliest call of the current step still awaiting its result."""
        if self.current is None:
            return None
        for record in self.current.tool_calls:
            if record.status == "pending" and (name is None or record.name == name):
                return record
        return None

    @property
    def tool_call_count(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps)

    @property
    def parse_failure_count(self) -> int:
        return sum(1 for s in self.steps if s.parse.status == "failed")
