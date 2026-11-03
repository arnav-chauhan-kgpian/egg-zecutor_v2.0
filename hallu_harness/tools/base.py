"""The framework-agnostic tool interface (DESIGN.md §7).

A tool is defined once — name, description, JSON-schema parameters, `call()` —
and each framework's bridge wraps it, invoking it through `run()`. Its
description is model-facing text, so it comes from the benchmark and is
identical in every framework.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.errors import HarnessAbort
from ..core.types import ToolResult


class ToolArgumentError(TypeError):
    """The model called a tool with arguments it does not take. This is the
    model's mistake, so the framework reports it to the model as it reports
    any tool error."""


class Tool(ABC):
    name: str = ""
    allowed_params: set[str] = set()  # keys a dataset may set under tools.params.<name>

    def __init__(self, env: Any, **params: Any):
        self.env = env
        self.params = params

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON schema with exactly `type`, `properties`, `required` (the shape
        Qwen-Agent's validator accepts and smolagents' converter produces)."""

    @abstractmethod
    def call(self, **arguments: Any) -> ToolResult: ...

    def run(self, **arguments: Any) -> ToolResult:
        """`call`, as every framework bridge invokes it.

        A `ToolArgumentError` is the model's mistake and is raised for the
        framework to report to the model. Any other exception is a failure
        of the harness and is raised as `HarnessAbort`, which ends the
        trajectory before the model can see it.
        """
        try:
            return self.call(**arguments)
        except ToolArgumentError:
            raise
        except Exception as exc:
            raise HarnessAbort(f"tool {self.name!r} failed inside the harness: {type(exc).__name__}: {exc}") from exc

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}
