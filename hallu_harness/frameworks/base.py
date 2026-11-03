"""What every framework adapter provides.

The reply phrases each framework uses live in `core/protocol.py`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.protocol import Protocol, TEXT_PROTOCOL, TOOL_PROTOCOL_QWEN, TOOL_PROTOCOL_SMOLAGENTS  # noqa: F401
from ..core.recorder import StepRecorder
from ..core.types import FrameworkResult, Task
from ..models.base import ModelBackend


class FrameworkAdapter(ABC):
    """Runs one trajectory under one framework.

    Constructed fresh for every trajectory (no state leaks between tasks).
    `run` must never let an exception escape for an expected condition:
    context exhaustion, a failed model call or a framework error all end the
    run with a `FrameworkResult` whose `stop_reason` says what happened.
    """

    name: str = ""
    protocol: Protocol
    content_decode: str = "skip_special_tokens"

    def __init__(self, params: dict, backend: ModelBackend, recorder: StepRecorder, benchmark: Any, prompt_mode: str):
        self.params = params
        self.backend = backend
        self.recorder = recorder
        self.benchmark = benchmark
        self.prompt_mode = prompt_mode

    @property
    def max_model_calls(self) -> int:
        return int(self.params["max_model_calls"])

    @abstractmethod
    def run(self, task: Task, tools: list, env: Any) -> FrameworkResult: ...

    @classmethod
    @abstractmethod
    def framework_info(cls, params: dict) -> dict:
        """Name, version, agent class and resolved params, for the trajectory."""

    # -- shared by the bridges ---------------------------------------------

    def generate(self, messages: list[dict], tools: list[dict] | None = None, stop: list[str] | None = None, kind: str | None = None):
        """One model call through the shared engine, recorded.

        Context exhaustion is recorded on the recorder *and* re-raised: some
        frameworks swallow model errors, and the recorder is then the only
        place the stop reason survives.
        """
        import time

        from ..core.errors import ContextExhausted

        seed = self.recorder.seed_for_next_call()
        started = time.perf_counter()
        try:
            generation = self.backend.generate(
                messages, tools=tools, stop=stop, seed=seed, content_decode=self.content_decode
            )
        except ContextExhausted as exc:
            self.recorder.on_context_exhausted(exc)
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised for the framework to handle
            self.recorder.on_model_error(f"{type(exc).__name__}: {exc}")
            raise
        self.recorder.on_model_call(generation, seed=seed, started=started, kind=kind)
        return generation
