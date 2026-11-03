"""Shared by the QA variants: the argument-writing section, and the trace of
tool calls handed to the benchmark after a run."""
from __future__ import annotations


def require_qa_benchmark(benchmark, framework: str) -> None:
    if not getattr(benchmark, "qa", False):
        raise RuntimeError(f"{framework} is for question-answering benchmarks; {benchmark.name!r} is not one")


def arguments_section(tool_names: list[str]) -> str:
    """How to write a tool call's arguments. No worked example: a small model
    copies one (the coding prompt's escaped "\\n" example is suspected of
    Coder-32B's double-escaped newlines)."""
    text = (
        "# Writing arguments\n"
        "\n"
        'The "arguments" of a tool call are a JSON object. A string in it is a JSON string: write a line break in '
        'it as \\n, a double quote as \\" and a backslash as \\\\.'
    )
    if "python" in tool_names:
        text += ('\n\nThe "code" argument of python holds your complete program as such a string, without ``` fences. '
                 "Each call runs only that program: nothing is kept between calls.")
    return text


def trace_of(recorder) -> list[dict]:
    """Every tool call of the run, in order, with what the model was shown."""
    out = []
    for step in recorder.steps:
        for call in step.tool_calls:
            out.append({"step": step.number, "tool": call.name, "arguments": call.arguments,
                        "observation": call.observation, "status": call.status})
    return out
