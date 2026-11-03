"""Grounding labels for a finished GAIA trajectory, computed after the run
and never shown to the model.

* `reference_in_observations` / `answer_in_observations`: the steps whose
  tool observation contains the reference answer, or the model's answer.
  Numbers match as numbers (any number in the text with `$ % ,` removed
  that equals it); strings match as whole words, ignoring case, with any
  run of whitespace or punctuation between words; a list matches when
  every item does, in one observation. A string under three characters, or
  a number of fewer than three digits, is too common to mean anything, so
  `informative` is False and the step lists are left empty.
* `unseen_urls`: URLs the model fetched (`visit_webpage`, `find_in_page`)
  that appear neither in the question nor in any earlier observation: a URL
  the model was never shown.

Together they separate an answer the evidence contained and the model
ignored, an answer found in no evidence, and a correct answer reached
without any. They are heuristics for selecting cases, not verdicts.
"""
from __future__ import annotations

import re

from ...tools.documents import normalize_url
from .scorer import is_float, normalize_number_str, split_string

_NUMBER = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")
_URL = re.compile(r"https?://[^\s<>\"')\]]+")
URL_TOOLS = {"visit_webpage": "url", "find_in_page": "source"}


def _digits(text: str) -> int:
    return sum(c.isdigit() for c in text)


def _matcher(item: str):
    """A predicate over text for one answer item, and whether it is informative."""
    item = item.strip()
    if not item:
        return (lambda text: False), False
    if is_float(item.replace(",", "").replace("$", "").replace("%", "")):
        target = normalize_number_str(item)

        def number_in(text: str) -> bool:
            return any(normalize_number_str(m.group(0)) == target for m in _NUMBER.finditer(text))

        return number_in, _digits(item) >= 3
    words = re.findall(r"\w+", item)
    if not words:
        return (lambda text: item.lower() in text.lower()), len(item) >= 3
    pattern = re.compile(r"(?<!\w)" + r"[\W_]+".join(re.escape(w) for w in words) + r"(?!\w)", re.IGNORECASE)
    return (lambda text: pattern.search(text) is not None), len("".join(words)) >= 3


def occurrence_steps(answer: str | None, trace: list[dict]) -> tuple[list[int], bool]:
    """Steps whose observation contains `answer` (every item of a list), and
    whether the answer is informative enough for that to mean anything."""
    if not answer:
        return [], False
    one_number = re.fullmatch(r"\s*[-+]?\$?[\d,]*\.?\d+%?\s*", answer) is not None  # "1,000" is a number, not a list
    items = [answer] if one_number else [x for x in split_string(answer) if x.strip()]
    matchers = [_matcher(x) for x in items]
    informative = all(ok for _, ok in matchers)
    if not informative:
        return [], False
    steps = []
    for entry in trace:
        text = entry.get("observation") or ""
        if text and all(match(text) for match, _ in matchers):
            steps.append(entry["step"])
    return sorted(set(steps)), True


def unseen_urls(question: str, trace: list[dict]) -> list[dict]:
    seen = {normalize_url(u) for u in _URL.findall(question)}
    out = []
    for entry in trace:
        key = URL_TOOLS.get(entry.get("tool"))
        arguments = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        url = arguments.get(key) if key else None
        if isinstance(url, str) and url.lower().startswith(("http://", "https://")):
            if normalize_url(url) not in seen:
                out.append({"step": entry["step"], "tool": entry["tool"], "url": url})
        for found in _URL.findall(entry.get("observation") or ""):
            seen.add(normalize_url(found))
    return out


def compute(question: str, reference: str, answer: str | None, trace: list[dict]) -> dict:
    # smolagents records final_answer as a tool call whose "observation" is the
    # answer itself; it is not evidence the model was shown.
    trace = [entry for entry in trace if entry.get("tool") != "final_answer"]
    ref_steps, ref_informative = occurrence_steps(reference, trace)
    ans_steps, ans_informative = occurrence_steps(answer, trace)
    tools: dict[str, int] = {}
    for entry in trace:
        tools[entry["tool"]] = tools.get(entry["tool"], 0) + 1
    return {
        "reference_in_observations": ref_steps,
        "reference_informative": ref_informative,
        "answer_in_observations": ans_steps,
        "answer_informative": ans_informative,
        "unseen_urls": unseen_urls(question, trace),
        "tool_calls": tools,
    }
