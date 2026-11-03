"""ExecLab-backed code execution sandbox.

Replaces ``cf_agent.sandbox.process`` and related modules for trajectory
generation in hallu_harness. Every call is synchronous from the caller's
perspective: it POSTs to the ExecLab engine, polls until the run reaches
COMPLETED or FAILED, and returns result objects shaped like the
``cf_agent.sandbox`` equivalents so that all existing callers work unchanged.

Environment variables:
    EXEC_LAB_API_URL   Base URL of the ExecLab API (default: http://localhost:4000)
    EXEC_LAB_API_TOKEN | JWT bearer token for the API (optional)
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

API_URL = os.environ.get("EXEC_LAB_API_URL", "http://localhost:4000")
API_TOKEN = os.environ.get("EXEC_LAB_API_TOKEN", "")
PYTHON_LANGUAGE_ID = 71  # Judge0 id for Python 3
DEFAULT_POLL_INTERVAL = 0.5  # seconds between poll attempts


# ---------------------------------------------------------------------------
# run_program — replaces cf_agent.sandbox.process.run_program
# ---------------------------------------------------------------------------

class _RunResult:
    """Shape-compatible with cf_agent.sandbox.process.run_program result.

    Attributes:
        exit_code: Process exit code (-1 if unreachable / wall timeout).
        stdout: Program standard output.
        stderr: Program standard error.
        timed_out: True if the run hit its CPU or wall time limit.
        seconds: Measured CPU time in seconds.
        output_limit_hit: True if output exceeded the sandbox cap.
        error_message: Human-readable failure reason, if any.
    """

    def __init__(
        self,
        exit_code: int,
        stdout: str,
        stderr: str,
        timed_out: bool,
        seconds: float,
        output_limit_hit: bool,
        error_message: str | None = None,
    ) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out
        self.seconds = seconds
        self.output_limit_hit = output_limit_hit
        self.error_message = error_message


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    return headers


def run_program(
    code: str,
    stdin: str = "",
    timeout: float = 30.0,
    memory_mb: int | None = None,
    wall_timeout: float = 60.0,
) -> _RunResult:
    """Execute code via the ExecLab engine.

    Mirrors ``cf_agent.sandbox.process.run_program`` so existing callers need
    no changes beyond the import site.

    Args:
        code: Python 3 source to execute.
        stdin: Text fed to the program on standard input.
        timeout: CPU time limit in seconds (per input / per run).
        memory_mb: Memory limit in megabytes, or None for the engine default.
        wall_timeout: Hard cap on how long this function may block polling.

    Returns:
        A ``_RunResult`` with exit_code, stdout, stderr, timed_out, seconds,
        output_limit_hit, and error_message.
    """
    body: dict[str, Any] = {
        "code": code,
        "languageId": PYTHON_LANGUAGE_ID,
        "stdin": stdin,
        "timeLimit": int(timeout),
    }
    if memory_mb is not None:
        body["memoryLimit"] = memory_mb

    deadline = time.time() + wall_timeout
    with httpx.Client(timeout=30.0) as client:
        # 1. Queue the run (async; returns 202 immediately).
        resp = client.post(
            f"{API_URL}/api/v1/executions/run",
            json=body,
            headers=_headers(),
        )
        resp.raise_for_status()
        execution_id = resp.json()["execution"]["id"]

        # 2. Poll until terminal state or wall timeout.
        while time.time() < deadline:
            resp = client.get(
                f"{API_URL}/api/v1/executions/{execution_id}",
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()["execution"]
            status = data["status"]

            if status in ("COMPLETED", "FAILED"):
                timed_out = (
                    status == "FAILED"
                    or data.get("judgeStatus") == "Time Limit Exceeded"
                )
                return _RunResult(
                    exit_code=data.get("exitCode") or -1,
                    stdout=data.get("stdout") or "",
                    stderr=data.get("stderr") or "",
                    timed_out=timed_out,
                    seconds=(data.get("timeMs") or 0) / 1000.0,
                    output_limit_hit=False,
                    error_message=data.get("errorMessage"),
                )

            time.sleep(DEFAULT_POLL_INTERVAL)

        # 3. Wall timeout exceeded before the run settled.
        return _RunResult(
            exit_code=-1,
            stdout="",
            stderr="",
            timed_out=True,
            seconds=wall_timeout,
            output_limit_hit=False,
            error_message="wall timeout exceeded",
        )


# ---------------------------------------------------------------------------
# run_against_examples — replaces cf_agent.sandbox.samples.run_against_examples
# ---------------------------------------------------------------------------

class _Example:
    __slots__ = ("input", "output")

    def __init__(self, inp: str, out: str) -> None:
        self.input = inp
        self.output = out


class _Result:
    __slots__ = ("example", "actual_output", "passed", "error", "stopped_for_extra_output")

    def __init__(
        self,
        example: _Example,
        actual_output: str,
        passed: bool,
        error: str | None,
        stopped_for_extra_output: bool,
    ) -> None:
        self.example = example
        self.actual_output = actual_output
        self.passed = passed
        self.error = error
        self.stopped_for_extra_output = stopped_for_extra_output


def run_against_examples(
    code: str,
    examples: list[dict[str, str]],
    timeout: float = 30.0,
    memory_mb: int | None = None,
) -> list[_Result]:
    """Run code against a list of example tests via ExecLab.

    Mirrors ``cf_agent.sandbox.samples.run_against_examples`` so the Codeforces
    benchmark can drop in this adapter without changing its call sites.

    Each example is executed in its own ExecLab run with the example's input
    as stdin. Codeforces solutions typically read from stdin and write to
    stdout directly, so the user's code is passed verbatim as the `code`
    parameter.
    """
    results: list[_Result] = []
    for example in examples:
        ex_in = example.get("input", "")
        ex_out = example.get("output", "")

        result = run_program(
            code,
            stdin=ex_in,
            timeout=timeout,
            memory_mb=memory_mb,
            wall_timeout=timeout + 30.0,
        )

        passed = (
            result.exit_code == 0
            and not result.timed_out
            and result.stdout.strip() == ex_out.strip()
        )
        results.append(
            _Result(
                example=_Example(ex_in, ex_out),
                actual_output=result.stdout,
                passed=passed,
                error=result.stderr or result.error_message,
                stopped_for_extra_output=result.output_limit_hit,
            )
        )

    return results


def default_timeout(problem: Any, multiplier: float) -> float:
    """Return default timeout for a problem based on its time limit.

    Mirrors ``cf_agent.dataset.problem.default_timeout``.
    """
    base = getattr(problem, "enforced_time_limit", None)
    if callable(base):
        base = base()
    else:
        base = getattr(problem, "time_limit", 1.0)
    return base * multiplier


# ---------------------------------------------------------------------------
# compare functions — replaces cf_agent.sandbox.compare
# ---------------------------------------------------------------------------

def split_tokens(text: str) -> list[str]:
    """Simple whitespace tokenization."""
    return text.split()


def token_limits(text: str) -> tuple[int, int]:
    """Return (max_tokens, max_chars) for displaying a value's repr.

    Mirrors cf_agent.sandbox.compare.token_limits: roughly one token per four
    characters, with a floor so small values still get a reasonable display.
    """
    max_chars = max(2000, len(text) * 2 + 100)
    max_tokens = max(500, max_chars // 4)
    return max_tokens, max_chars


def display_output(
    text: str,
    max_tokens: int | None = None,
    max_chars: int | None = None,
) -> tuple[str, bool]:
    """Truncate output to limits, return (text, was_truncated).

    Mirrors cf_agent.sandbox.compare.display_output.
    """
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars], True
    if max_tokens is not None:
        tokens = split_tokens(text)
        if len(tokens) > max_tokens:
            return " ".join(tokens[:max_tokens]), True
    return text, False


def cap_long_tokens(text: str, max_chars: int) -> str:
    """Cap text at max_chars characters, preserving word boundaries where possible.

    Mirrors cf_agent.sandbox.compare.cap_long_tokens.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Aliases that mirror cf_agent.sandbox.process names used by callers.
# ---------------------------------------------------------------------------

def wall_limit(seconds: float) -> float:
    """Wall-clock backstop for a per-input CPU time limit.

    ExecLab enforces its own wall-clock limit; this returns the input unchanged
    so callers can keep their existing call sites.
    """
    return seconds


WALL_TIME_FACTOR = 1.0
WALL_TIME_SLACK_S = 0.0
