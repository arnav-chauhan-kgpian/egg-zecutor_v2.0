"""`run_visible_tests` — the visible tests only (an ablation; DESIGN.md §7.2).

The model learns what the visible tests show and nothing about the hidden
ones; the final solution is still graded on the hidden tests after the run.
"""
from __future__ import annotations

from ..core.types import ToolResult
from .base import Tool, ToolArgumentError


class RunVisibleTestsTool(Tool):
    name = "run_visible_tests"
    allowed_params: set[str] = {"max_output_tokens"}

    @property
    def description(self) -> str:
        texts = self.env.texts
        noun = texts.code_noun
        return (
            f"Runs a complete {noun} on the visible tests shown in the task (the {texts.visible_tests}s) and "
            "returns, for each, whether it passed and, for a failure, the expected and the actual result. It never "
            f"runs the hidden tests, and a {noun} run only with this tool is never taken as your solution."
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
            raise ToolArgumentError(f"run_visible_tests got unexpected argument(s): {', '.join(sorted(unexpected))}")
        env = self.env
        if not isinstance(code, str) or not code.strip():
            return ToolResult(observation=env.texts.empty_code(env.protocol), private={"ran": False})
        report = env.evaluator.run_visible(code.strip("\n"))
        return ToolResult(
            observation=env.texts.visible_feedback(report, env.protocol),
            private={"ran": True, "visible": [r.private() for r in report.results], "visible_all_passed": report.all_passed},
        )
