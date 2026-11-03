"""Printing a run as it happens (DESIGN.md §12).

    quiet    one line per trajectory
    normal   + per step: parsed tool calls, observations, parse errors
    verbose  + the exact input and the exact output of every model call

Plain `print`, never a pretty-printer: wrapping, markup interpretation and
elision would each defeat the only purpose of this output, which is to be an
exact record. The console is a recorder listener, so it prints the very
objects the trajectory is written from.

With `verbose_input: delta` (the default) the full input is printed once per
trajectory, and afterwards only what was appended since the previous call.
When the new input does not extend the previous one — the framework or the
chat template rewrote history — the whole input is printed under a notice.
The trajectory always stores every input in full.
"""
from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from .types import Step, ToolCallRecord

RULE = "=" * 100
SECTION_RULE = "-" * 100


class Console:
    def __init__(self, verbosity: str = "normal", verbose_input: str = "delta", stream: TextIO | None = None):
        self.verbosity = verbosity
        self.verbose_input = verbose_input
        self.stream = stream or sys.stdout
        self.header = ""
        self.max_calls: int | None = None
        self._previous_input: str | None = None

    # -- primitives -------------------------------------------------------

    def _print(self, text: str = "") -> None:
        print(text, file=self.stream, flush=True)

    def banner(self, title: str) -> None:
        self._print(f"\n{RULE}\n{title}\n{RULE}")

    def section(self, title: str, body: str) -> None:
        self._print(f"\n{SECTION_RULE}\n{title}\n{SECTION_RULE}")
        self._print(body)

    @property
    def normal(self) -> bool:
        return self.verbosity in ("normal", "verbose")

    @property
    def verbose(self) -> bool:
        return self.verbosity == "verbose"

    # -- trajectory boundaries ---------------------------------------------

    def start_trajectory(self, header: str, max_calls: int | None) -> None:
        self.header = header
        self.max_calls = max_calls
        self._previous_input = None
        if self.normal:
            self.banner(f"[{header}] start")

    def end_trajectory(self, summary: dict) -> None:
        line = (
            f"[{self.header}] {'SOLVED' if summary.get('solved') else 'not solved'} — "
            f"{summary.get('model_calls')} model call(s), {summary.get('tool_calls')} tool call(s), "
            f"stop: {summary.get('stop_reason')}, {summary.get('seconds', 0):.1f}s"
        )
        if summary.get("status") != "complete":
            line += f" [{summary.get('status')}: {summary.get('error')}]"
        self._print(line if not self.normal else f"\n{line}")

    # -- recorder listener ------------------------------------------------

    def __call__(self, event: str, recorder: Any, payload: Any) -> None:
        handler = getattr(self, f"_on_{event}", None)
        if handler is not None:
            handler(payload)

    def _on_model_call(self, step: Step) -> None:
        if not self.normal:
            return
        g = step.generation
        of = f"/{self.max_calls}" if (self.max_calls and step.kind == "action") else ""
        self.banner(f"[{self.header}]  STEP {step.number}{of}  ({step.kind})")
        if self.verbose:
            self._print_input(step)
            tokens = f"{g.output_tokens} tokens, " if g.output_tokens is not None else ""
            self.section(
                f"2.  EXACT OUTPUT OF MODEL  (verbatim, {tokens}finish={g.finish_reason}, "
                f"length cap: {'YES' if g.hit_length_cap else 'no'}, {g.seconds:.1f}s)",
                g.raw_output or "(empty reply)",
            )
        else:
            cap = "  ** HIT LENGTH CAP **" if g.hit_length_cap else ""
            self._print(
                f"model: {g.input_tokens} input tokens -> {g.output_tokens} output tokens, "
                f"finish={g.finish_reason}, {g.seconds:.1f}s{cap}"
            )

    def _print_input(self, step: Step) -> None:
        g = step.generation
        text = g.exact_input
        tokens = f", {g.input_tokens} tokens" if g.input_tokens is not None else ""
        kind = g.exact_input_kind
        prev = self._previous_input
        self._previous_input = text
        if self.verbose_input == "full" or prev is None:
            self.section(f"1.  EXACT INPUT TO MODEL  ({kind}{tokens})", text)
        elif text.startswith(prev):
            self.section(
                f"1.  EXACT INPUT TO MODEL — ONLY WHAT WAS APPENDED SINCE THE PREVIOUS CALL "
                f"({kind}{tokens} in total)",
                text[len(prev):] or "(nothing appended)",
            )
        else:
            self.section(
                "1.  EXACT INPUT TO MODEL — !! CONTEXT REWRITTEN: this input does not extend the previous "
                f"one, so it is printed in full ({kind}{tokens})",
                text,
            )

    def _on_parse(self, step: Step) -> None:
        if not self.normal:
            return
        p = step.parse
        attempted = "" if p.attempted_tool_call is None else f", attempted tool call: {p.attempted_tool_call}"
        line = f"parse: {p.status}{attempted}"
        if p.error:
            line += f"\n{p.error}"
        self.section("PARSE STATUS", line)

    def _on_tool_call(self, record: ToolCallRecord) -> None:
        if not self.normal:
            return
        args = record.arguments
        body = [f"tool: {record.name}" + (f"   (call id {record.call_id})" if record.call_id else "")]
        if isinstance(args, dict) and isinstance(args.get("code"), str):
            others = {k: v for k, v in args.items() if k != "code"}
            if others:
                body.append("arguments: " + json.dumps(others, ensure_ascii=False, default=str))
            body += ['"code" argument, as it will run:', "-" * 60, args["code"], "-" * 60]
        else:
            body.append("arguments: " + json.dumps(args, ensure_ascii=False, default=str, indent=2))
        self.section(f"3.  PARSED TOOL CALL #{record.index}", "\n".join(body))

    def _on_tool_result(self, record: ToolCallRecord) -> None:
        if not self.normal:
            return
        self.section(
            f"4.  TOOL OUTPUT #{record.index}  (status {record.status}; the observation, byte-exact)",
            record.observation if record.observation is not None else "(none)",
        )
        if self.verbose and record.private:
            self.section(
                f"4b. PRIVATE GRADING DETAIL #{record.index}  (never shown to the model)",
                json.dumps(record.private, indent=2, ensure_ascii=False, default=str),
            )

    def _on_framework_feedback(self, step: Step) -> None:
        """Printed when it says something the tool outputs above did not —
        an error the framework reports, a parse failure. When it only wraps
        the observations ("Observation:\n..."), it is shown in verbose mode
        alone, since the next step's input carries it anyway."""
        if not self.normal or not step.framework_feedback:
            return
        routine = step.parse.status == "ok" and all(r.status == "executed" for r in step.tool_calls)
        if routine and not self.verbose:
            return
        self.section("FRAMEWORK FEEDBACK  (what the framework itself tells the model next)", step.framework_feedback)

    def _on_context_exhausted(self, exc: Any) -> None:
        if self.normal:
            self.banner(f"[{self.header}] CONTEXT EXHAUSTED — {exc}")

    def _on_model_error(self, message: str) -> None:
        if self.normal:
            self.banner(f"[{self.header}] MODEL CALL FAILED — {message}")

    def note(self, text: str) -> None:
        if self.normal:
            self._print(text)
