"""`submit` — an explicit final submission, for frameworks without a
final-answer tool when an experiment wants one (DESIGN.md §7.2).

The submitted code is graded after the run (solution source `submitted`);
nothing is run now and the model is told only that it was received. A run
that enables this tool must list `submitted` in the dataset's
`solution_policy` (checked in `validate_resolved`), so the description below
is true.
"""
from __future__ import annotations

from ..core.types import ToolResult
from .base import Tool, ToolArgumentError


class SubmitTool(Tool):
    name = "submit"
    allowed_params: set[str] = set()

    @property
    def description(self) -> str:
        return (
            f"Records your final {self.env.texts.code_noun} as your submission. Nothing is run, no result is "
            "returned, and the task does not end; the submission is graded after the task ends."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"code": {"type": "string", "description": self.env.texts.code_argument_description()}},
            "required": ["code"],
        }

    def call(self, code: str = "", **unexpected) -> ToolResult:
        if unexpected:
            raise ToolArgumentError(f"submit got unexpected argument(s): {', '.join(sorted(unexpected))}")
        self.env.submitted_code = code.strip("\n") if isinstance(code, str) and code.strip() else None
        return ToolResult(observation="Submitted.", private={"received": self.env.submitted_code is not None})
