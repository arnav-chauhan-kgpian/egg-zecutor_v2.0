"""Reading the model's answer out of a finished run, as the QA protocol told
it (`core/qa_protocol.py`).

* smolagents: the "answer" argument of final_answer. smolagents hands back
  the value, which the adapter turns into a string, so an answer given as a
  JSON list (`["a", "b"]`) arrives as Python's text for a list; it is
  rejoined as "a, b", the comma-separated list the instructions ask for.
* Qwen-Agent: the reply that ended the task, think blocks removed;
  everything after its last "FINAL ANSWER:" (any letter case). Markdown
  emphasis or code marks around the answer (`**42**`, `` `42` ``) are
  stripped. A reply without the marker gives no answer.
* SORRY (alone, ignoring quotes and a final period) is giving up.

The extraction never changes what counts as correct beyond that: the answer
string goes to GAIA's own scorer as is.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from ...core.qa_protocol import QA_QWEN, QA_SMOLAGENTS
from ..base import is_sorry

MARKER = re.compile(r"FINAL\s+ANSWER\s*:", re.IGNORECASE)
THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
WRAPPING = " \t\r\n*`"


@dataclass
class Extraction:
    answer: str | None  # what goes to the scorer; None if the run gave none
    found: bool  # the protocol's answer form was present
    abstained: bool  # SORRY
    method: str  # how it was read: final_answer | final_answer_list | marker | no_marker | none

    def to_dict(self) -> dict:
        return {"answer": self.answer, "found": self.found, "abstained": self.abstained, "method": self.method}


def _rejoin_list(text: str) -> str | None:
    """"a, b" for Python's text of a list or tuple of plain values; else None."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "[(":
        return None
    try:
        value = ast.literal_eval(stripped)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, (str, int, float)) for v in value):
        return ", ".join(str(v) for v in value)
    return None


def extract(final_answer: str | None, protocol_name: str) -> Extraction:
    if final_answer is None:
        return Extraction(None, False, False, "none")
    if protocol_name == QA_SMOLAGENTS.name:
        text = final_answer.strip()
        if is_sorry(text):
            return Extraction(None, True, True, "final_answer")
        joined = _rejoin_list(text)
        if joined is not None:
            return Extraction(joined, True, False, "final_answer_list")
        return Extraction(text, True, False, "final_answer")
    if protocol_name == QA_QWEN.name:
        visible = THINK.sub("", final_answer)
        if re.search(r"<think>", visible, re.IGNORECASE):  # an unclosed think block: the reply never left it
            return Extraction(None, False, False, "no_marker")
        markers = list(MARKER.finditer(visible))
        if not markers:
            if is_sorry(visible):
                return Extraction(None, True, True, "no_marker")
            return Extraction(None, False, False, "no_marker")
        answer = visible[markers[-1].end():].strip(WRAPPING)
        if is_sorry(answer):
            return Extraction(None, True, True, "marker")
        return Extraction(answer or None, True, False, "marker")
    raise ValueError(f"no answer rule for protocol {protocol_name!r}")
