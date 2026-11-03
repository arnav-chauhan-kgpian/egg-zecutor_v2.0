"""GAIA's official answer comparison: `question_scorer` from `scorer.py` in
the Hugging Face space `gaia-benchmark/leaderboard` (read 2026-10-15 and
2026-10-29), ported verbatim except that its `print` calls and its warning
are removed. Do not "improve" it: scores must match the leaderboard's.

* reference is a number: the answer with `$`, `%` and `,` removed must
  parse to the same float (an answer that does not parse becomes inf);
* reference contains `,` or `;`: both are split on either character, must
  have the same number of items, and are compared item by item, as numbers
  or as strings with all whitespace removed and lower-cased (punctuation
  kept);
* otherwise both are compared as strings with whitespace and ASCII
  punctuation removed, lower-cased.
"""
from __future__ import annotations

import re
import string


def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def split_string(s: str, char_list: list[str] = [",", ";"]) -> list[str]:  # noqa: B006 - upstream's signature
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def normalize_str(input_str, remove_punct=True) -> str:
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def question_scorer(model_answer: str, ground_truth: str) -> bool:
    if model_answer is None:
        model_answer = "None"

    if is_float(ground_truth):
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)

    elif any(char in ground_truth for char in [",", ";"]):
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                comparisons.append(normalize_str(ma_elem, remove_punct=False) == normalize_str(gt_elem, remove_punct=False))
        return all(comparisons)

    else:
        return normalize_str(model_answer) == normalize_str(ground_truth)


def answer_type(ground_truth: str) -> str:
    """Which branch of `question_scorer` a reference takes."""
    if is_float(ground_truth):
        return "number"
    if any(char in ground_truth for char in [",", ";"]):
        return "list"
    return "string"
