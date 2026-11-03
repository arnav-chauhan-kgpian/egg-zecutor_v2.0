"""The data that flows through a run.

Everything that ends up in a trajectory is one of these dataclasses, built by
the component that observed it: the engine builds `Generation`s, tools build
`ToolResult`s, the recorder assembles `Step`s. Nothing is reconstructed after
the fact from logs or framework internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------- model


@dataclass
class Generation:
    """One model call, exactly as it happened."""

    exact_input: str  # the templated string the weights saw (or the API request JSON)
    exact_input_kind: str  # "rendered_template" | "api_request" | "scripted"
    raw_output: str  # decoded WITH special tokens, BEFORE any stop-sequence trim — this is stored
    content: str  # what is handed back to the framework (not stored; recoverable from the next input)
    input_tokens: int | None
    output_tokens: int | None
    hit_length_cap: bool  # stopped because the reply budget ran out
    finish_reason: str  # "eos" | "length" | "stop_sequence"
    seconds: float
    sampling: dict  # the effective decoding settings, including the seed
    context_window: int | None = None
    reply_budget: int | None = None
    content_decode: str = "skip_special_tokens"  # how `content` was decoded
    # Token-level record (DESIGN.md §11.4). None when the backend cannot provide it.
    input_token_ids: list[int] | None = None
    output_token_ids: list[int] | None = None
    output_logprobs: list[float] | None = None
    output_entropy: list[float] | None = None
    top_logprobs: list[list[tuple[int, float]]] | None = None
    segments: list[dict] | None = None
    messages: list[dict] | None = None  # the chat messages rendered into exact_input (not stored)


# --------------------------------------------------------------------- tools


@dataclass
class ToolResult:
    """What a tool returns.

    `observation` is the exact text handed back to the framework; `private`
    holds facts the harness learned while running the tool and did not tell
    the model (grading detail). Hidden-test *content* is never put in
    `private` — counts, verdicts and test numbers only.
    """

    observation: str
    private: dict = field(default_factory=dict)
    status: str = "ok"  # "ok" | "error"


# --------------------------------------------------------------------- tasks


@dataclass
class Task:
    """One dataset instance."""

    task_id: str
    query: str  # the problem as the model reads it (rendered statement / prompt)
    metadata: dict  # recorded verbatim in the trajectory
    payload: Any = None  # dataset-private object (problem, tests handle); never serialized


# --------------------------------------------------------------------- records


@dataclass
class ToolCallRecord:
    """One tool call as the framework parsed it, and what came of it."""

    index: int
    name: str
    arguments: Any
    raw_arguments: str | None = None
    call_id: str | None = None
    status: str = "pending"  # executed | raised | unknown_tool | invalid_arguments | pending
    observation: str | None = None
    private: dict = field(default_factory=dict)
    error: str | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
            "raw_arguments": self.raw_arguments,
            "status": self.status,
            "result": {
                "observation": self.observation,
                "private": self.private,
                "error": self.error,
                "seconds": round(self.seconds, 3),
            },
        }


@dataclass
class ParseRecord:
    status: str = "ok"  # ok | no_tool_call | failed | recovered
    error: str | None = None
    attempted_tool_call: bool | None = None

    def to_dict(self) -> dict:
        return {"status": self.status, "error": self.error, "attempted_tool_call": self.attempted_tool_call}


@dataclass
class Step:
    """One model call and everything that hangs off it."""

    number: int
    kind: str  # action | planning | final_from_memory
    generation: Generation
    seed: int
    started: float
    parse: ParseRecord = field(default_factory=ParseRecord)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    framework_feedback: str | None = None
    seconds: float = 0.0
    ended: bool = False


@dataclass
class FrameworkResult:
    """What an adapter reports when its run ends."""

    final_answer: str | None
    # final_answer: the model finished or gave up as its protocol says;
    # no_tool_call: Qwen-Agent ended on a reply whose call did not parse, or that was cut off;
    # no_final_answer: smolagents ended without a final_answer call;
    # budget | context_exhausted | error
    stop_reason: str
    system_prompt: str | None
    user_query: str | None
    framework_version: str
    error: str | None = None
    native: dict = field(default_factory=dict)  # framework-specific facts worth keeping


@dataclass
class Outcome:
    """The graded result of one trajectory."""

    final_answer: str | None
    final_solution: str | None
    solution_source: str  # accepted | final_answer_code | last_submitted | none
    grade: dict | None
    solved: bool
    stop_reason: str
    model_calls: int
    tool_calls: int
    parse_failures: int
    seconds: float
    error: str | None = None
    context: dict | None = None
    model_errors: list[str] = field(default_factory=list)  # failed model calls, including any a framework swallowed

    def to_dict(self) -> dict:
        return {
            "final_answer": self.final_answer,
            "final_solution": self.final_solution,
            "solution_source": self.solution_source,
            "grade": self.grade,
            "solved": self.solved,
            "stop_reason": self.stop_reason,
            "context": self.context,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "parse_failures": self.parse_failures,
            "seconds": round(self.seconds, 3),
            "error": self.error,
            "model_errors": self.model_errors,
        }
