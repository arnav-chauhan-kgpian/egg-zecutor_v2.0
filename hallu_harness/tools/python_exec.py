"""`python_exec` — run any Python in the sandbox and see what it printed
(DESIGN.md §7.2, Phase 4).

A scratch execution tool: the model can test a hypothesis without submitting
anything for grading. Output is returned in full; only a runaway is cut, at
the sandbox's capture limit, and the model is told so.
"""
from __future__ import annotations

from ..exec_lab_sandbox import run_program

from ..core.types import ToolResult
from .base import Tool, ToolArgumentError

DEFAULT_TIMEOUT_S = 10.0


class PythonExecTool(Tool):
    name = "python_exec"
    allowed_params: set[str] = {"timeout_s", "memory_mb"}

    @property
    def description(self) -> str:
        timeout = float(self.params.get("timeout_s", DEFAULT_TIMEOUT_S))
        return (
            "Runs a Python 3 program in an isolated sandbox with the given text on standard input, for at most "
            f"{timeout:g} seconds, and returns its exit code, its standard output and any standard error. Use it "
            "to try things out: it grades nothing, and a program run only with this tool is never taken as your "
            "solution."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The Python 3 program to run."},
                "stdin": {"type": "string", "description": "Text passed to the program on standard input (may be empty)."},
            },
            "required": ["code"],
        }

    def call(self, code: str = "", stdin: str = "", **unexpected) -> ToolResult:
        if unexpected:
            raise ToolArgumentError(f"python_exec got unexpected argument(s): {', '.join(sorted(unexpected))}")
        timeout = float(self.params.get("timeout_s", DEFAULT_TIMEOUT_S))
        result = run_program(code or "", stdin or "", timeout=timeout, memory_mb=self.params.get("memory_mb"), wall_timeout=timeout)
        parts = [f"exit code: {result.exit_code}" + (" (timed out)" if result.timed_out else "")]
        parts.append("stdout:\n" + (result.stdout if result.stdout else "(empty)"))
        if result.output_limit_hit:
            parts.append("(output truncated: the program printed more than the sandbox captures)")
        if result.stderr.strip() and not result.timed_out:
            parts.append("stderr:\n" + result.stderr)
        return ToolResult(
            observation="\n".join(parts),
            private={"exit_code": result.exit_code, "timed_out": result.timed_out, "seconds": round(result.seconds, 3)},
        )
