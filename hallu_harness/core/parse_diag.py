"""Framework-independent parse diagnostics (DESIGN.md R6).

Each framework parses replies its own way and reports failure differently:
smolagents raises, Qwen-Agent silently treats a malformed call as a final
answer, the text protocol finds no code block. To make tool-call parse
failures (category 5) comparable across frameworks, every reply is also
checked here for whether it *attempted* a tool call, whatever the framework
concluded.
"""
from __future__ import annotations

import json
import re

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_MARKERS = (
    re.compile(r"<tool_call>"),
    re.compile(r"(^|\n)\s*Action\s*:", re.IGNORECASE),
    re.compile(r"\{\s*\"name\"\s*:"),
    re.compile(r"\"arguments\"\s*:"),
)
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)(</tool_call>|\Z)", re.DOTALL)


def visible_text(text: str) -> str:
    """The reply minus complete think blocks; an unterminated think block
    means everything after `<think>` is reasoning, so it is dropped too."""
    text = _THINK.sub("", text or "")
    opened = text.lower().find("<think>")
    return text[:opened] if opened >= 0 else text


def attempted_tool_call(text: str) -> bool:
    visible = visible_text(text)
    return any(marker.search(visible) for marker in _MARKERS)


def tagged_calls_parse(text: str, loads=json.loads) -> list[bool]:
    """For each `<tool_call>` block in the visible reply: is it closed, and
    does its body parse with `loads`? Qwen-Agent parses with `json5` and falls
    back to a lenient extractor when that fails; the adapter passes
    `json5.loads`, and calls that needed the fallback are recorded as
    `recovered`."""
    out = []
    for match in _TOOL_CALL_BLOCK.finditer(visible_text(text)):
        try:
            loads(match.group(1).strip())
            out.append(match.group(2) == "</tool_call>")
        except Exception:  # noqa: BLE001 - any parser's failure means "did not parse"
            out.append(False)
    return out
