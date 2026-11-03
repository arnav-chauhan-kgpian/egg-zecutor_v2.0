"""HumanEval, graded with HumanEval+'s tests (DESIGN.md §2.3, §8.2).

Tasks come from `openai/openai_humaneval`; their hidden tests from EvalPlus's
`evalplus/humanevalplus`, joined on task id (same 164 ids, same order, same
entry points — checked at load). Visible tests are the docstring's examples
(`doctest_extract`). Only tasks where the docstring, the original tests and
HumanEval+'s tests all agree are used: `excluded_tasks.json`, written by
`python -m hallu_harness audit-humaneval`, lists the rest with reasons.

Per-input time limits follow EvalPlus: a multiple of the EvalPlus reference
solution's measured CPU time on that input, with a floor. The reference times
are measured once by the audit and shipped with the exclusion list, so every
machine grades with the same limits. All hidden tests run in one process,
stopped after `hidden_total_timeout_s` of real time; the instructions say so.

A failed example shows both values exactly as `repr()` prints them; the
returned one is cut by characters, not by Codeforces' token rule
(`shown_value`, and `texts.py` for why).
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...core.config import resolve_repo_path
from ...core.errors import ConfigError
from ...core.types import Task
from ..base import Benchmark, Evaluator, HiddenReport, VisibleReport, VisibleResult
from ..codeforces.benchmark import cap_long_tokens
from ..exec_lab_sandbox import run_program
from .doctest_extract import extract
from .runner_program import REPR_CAP_CHARS, build_program, decompose_check, parse_output
from .texts import SHOWN_VALUE_SLACK_CHARS, HumanEvalTexts

HERE = Path(__file__).parent


def token_limits(text: str) -> tuple[int, int]:
    """Return (max_tokens, max_chars) for displaying a value's repr.

    Mirrors cf_agent.sandbox.compare.token_limits: roughly one token per four
    characters, with a floor so small values still get a reasonable display.
    """
    max_chars = max(2000, len(text) * 2 + 100)
    max_tokens = max(500, max_chars // 4)
    return max_tokens, max_chars


def wall_limit(seconds: float) -> float:
    """Wall-clock backstop for a per-input CPU time limit.

    ExecLab enforces its own wall-clock limit; this returns the input unchanged
    so callers can keep their existing call sites.
    """
    return seconds

HERE = Path(__file__).parent
EXCLUSIONS_PATH = HERE / "excluded_tasks.json"
REFERENCE_TIMES_PATH = HERE / "reference_times.json.gz"
HUMANEVAL = "openai/openai_humaneval"
HUMANEVAL_PLUS = "evalplus/humanevalplus"
FAILURE_ORDER = ("RUNTIME_ERROR", "MEMORY_LIMIT_EXCEEDED", "TIME_LIMIT_EXCEEDED", "WRONG_ANSWER")


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[str] | None = None
    limit: int | None = Field(None, gt=0)
    shuffle: bool = False
    seed: int = 0
    excluded_tasks: str | None = str(EXCLUSIONS_PATH)
    humaneval_revision: str | None = None
    humanevalplus_revision: str | None = None


class HiddenTimeLimit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    floor_s: float = Field(1.0, gt=0)
    reference_multiplier: float = Field(4.0, gt=0)


class Evaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    visible_tests: Literal["docstring_examples"] = "docstring_examples"
    hidden_tests: bool = True
    hidden_time_limit: HiddenTimeLimit = Field(default_factory=HiddenTimeLimit)
    # The hidden tests run one after another in one process, stopped after
    # this many seconds of real time; the task instructions say so.
    hidden_total_timeout_s: float = Field(600.0, gt=0)
    visible_time_limit_s: float = Field(5.0, gt=0)
    load_time_limit_s: float = Field(10.0, gt=0)
    memory_mb: float = Field(4096.0, gt=0)
    reference_times: str | None = str(REFERENCE_TIMES_PATH)
    solution_policy: list[Literal["accepted", "final_answer_code", "last_submitted", "submitted"]] = Field(
        default_factory=lambda: ["accepted", "final_answer_code", "last_submitted"]
    )


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")


@lru_cache(maxsize=4)
def load_sources(humaneval_revision: str | None = None, humanevalplus_revision: str | None = None) -> tuple[list[dict], list[dict], dict]:
    """Both datasets, joined and checked. Returns (humaneval, humanevalplus, info)."""
    from datasets import load_dataset

    he = load_dataset(HUMANEVAL, split="test", revision=humaneval_revision)
    hp = load_dataset(HUMANEVAL_PLUS, split="test", revision=humanevalplus_revision)
    he_rows, hp_rows = list(he), list(hp)
    if [r["task_id"] for r in he_rows] != [r["task_id"] for r in hp_rows]:
        raise ConfigError("HumanEval and HumanEval+ task ids do not line up")
    for a, b in zip(he_rows, hp_rows):
        if a["entry_point"] != b["entry_point"]:
            raise ConfigError(f"{a['task_id']}: entry points differ between HumanEval and HumanEval+")
    info = {
        "humaneval": HUMANEVAL,
        "humaneval_fingerprint": getattr(he, "_fingerprint", None),
        "humaneval_revision": humaneval_revision,
        "humanevalplus": HUMANEVAL_PLUS,
        "humanevalplus_fingerprint": getattr(hp, "_fingerprint", None),
        "humanevalplus_revision": humanevalplus_revision,
    }
    return he_rows, hp_rows, info


def load_reference_times(path: str | Path | None) -> dict[str, list[float]]:
    if not path:
        return {}
    return _load_reference_times(str(resolve_repo_path(path)))


@lru_cache(maxsize=4)
def _load_reference_times(path: str) -> dict[str, list[float]]:
    if not Path(path).exists():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)["tasks"]


def _helper_functions(prompt: str, entry_point: str) -> list[str]:
    try:
        return [n.name for n in ast.parse(prompt).body if isinstance(n, ast.FunctionDef) and n.name != entry_point]
    except SyntaxError:
        return []


def _original_test_count(test: str) -> int:
    try:
        return sum(isinstance(n, ast.Assert) for n in ast.walk(ast.parse(test)))
    except SyntaxError:
        return 0


def task_from_rows(he_row: dict, hp_row: dict) -> Task:
    prompt, entry = he_row["prompt"], he_row["entry_point"]
    examples, unparsed = extract(prompt, entry)
    check = decompose_check(hp_row["test"])
    return Task(
        task_id=he_row["task_id"],
        query=prompt,
        metadata={
            "task_id": he_row["task_id"],
            "entry_point": entry,
            "prompt": prompt,
            "visible_test_count": len(examples),
            "visible_examples": [{"call": e.call, "expected": e.expected, "form": e.form} for e in examples],
            "unparsed_docstring_examples": [{"text": u.text, "reason": u.reason} for u in unparsed],
            "hidden_test_count": check.input_count,
            "hidden_test_source": HUMANEVAL_PLUS,
            "original_test_count": _original_test_count(he_row["test"]),
            "prompt_helper_functions": _helper_functions(prompt, entry),
            "canonical_solution": f"{HUMANEVAL_PLUS}:{he_row['task_id']} (by reference, not copied)",
        },
        payload={
            "prompt": prompt,
            "entry_point": entry,
            "examples": examples,
            "check": check,
            "humaneval_test": he_row["test"],
            "humanevalplus_canonical": hp_row["canonical_solution"],
            "humaneval_canonical": he_row["canonical_solution"],
            "humanevalplus_prompt": hp_row["prompt"],
        },
    )


def value_cut(setting) -> int | None:
    """`max_output_tokens` for HumanEval: "expected" (the default) cuts a
    returned value SHOWN_VALUE_SLACK_CHARS characters past the expected
    repr's length; None cuts it only at the sandbox's REPR_CAP_CHARS. The
    slack, or None."""
    if setting == "expected":
        return SHOWN_VALUE_SLACK_CHARS
    if setting is None:
        return None
    raise ConfigError(
        f"tools.params.grade_code.max_output_tokens (humaneval) must be 'expected' or null, not {setting!r}: "
        "a returned value is cut by characters, relative to the expected value"
    )


def expected_repr(expected: str) -> str:
    """The expected value as repr() prints it, so both values in a report are
    spelled alike: a docstring may write "a" for 'a', or [1,2] for [1, 2]."""
    try:
        return repr(ast.literal_eval(expected))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return expected  # extract() keeps only literals; this is a safety net


def shown_value(actual: str, expected: str, slack: int | None) -> tuple[str, bool]:
    """A returned value's repr as the model is shown it, and whether it was cut.

    Verbatim, because the spaces inside a string decide equality. The cut is
    `slack` characters past the expected repr's length: the first character
    at which the two differ lies within len(expected) + 1, so the difference
    is always shown, with up to `slack` - 1 characters after it.
    """
    if slack is None or len(actual) <= len(expected) + slack:
        return actual, False
    return actual[: len(expected) + slack], True


class HumanEvalEvaluator(Evaluator):
    def __init__(self, task: Task, evaluation: Evaluation, reference_times: list[float] | None, max_output_tokens="expected"):
        self.task = task
        self.p = task.payload
        self.evaluation = evaluation
        self.reference_times = reference_times
        self.max_output_tokens = max_output_tokens

    @property
    def visible_test_count(self) -> int:
        return len(self.p["examples"])

    # -- running ------------------------------------------------------------

    def _run(self, payload: dict, timeout: float) -> tuple[list[dict], object]:
        """One sandboxed process, stopped after `timeout` seconds of real
        (and of CPU) time. The tests' own time limits are enforced inside it."""
        program, nonce = build_program(payload)
        result = run_program(program, "", timeout=timeout, memory_mb=self.evaluation.memory_mb, wall_timeout=timeout)
        return parse_output(result.stdout, nonce), result

    def _base_payload(self, code: str, mode: str) -> dict:
        return {
            "mode": mode,
            "prompt": self.p["prompt"],
            "solution": code,
            "entry_point": self.p["entry_point"],
            "load_limit": self.evaluation.load_time_limit_s,
        }

    def load_problem(self, code: str) -> str | None:
        payload = {**self._base_payload(code, "visible"), "examples": [], "limits": [], "repr_cap": 0}
        records, result = self._run(payload, wall_limit(self.evaluation.load_time_limit_s) + 30)
        for record in records:
            if "load_error" in record:
                return record["load_error"]
            if record.get("loaded"):
                return None
        return _process_failure(result)

    def run_visible(self, code: str) -> VisibleReport:
        examples = self.p["examples"]
        limit = self.evaluation.visible_time_limit_s
        payload = {
            **self._base_payload(code, "visible"),
            "examples": [{"i": n, "call": e.call, "expected": e.expected} for n, e in enumerate(examples, 1)],
            "limits": [limit] * len(examples),
            "repr_cap": REPR_CAP_CHARS,
        }
        # Long enough that every example reaches its own limits first.
        records, result = self._run(payload, wall_limit(self.evaluation.load_time_limit_s) + wall_limit(limit) * len(examples) + 30)
        load_error = next((r["load_error"] for r in records if "load_error" in r), None)
        if load_error is not None:
            return VisibleReport([VisibleResult(index=0, passed=False, input_text="", expected_text="", actual_text="", error=load_error)])
        by_index = {r["i"]: r for r in records if "i" in r}
        slack = value_cut(self.max_output_tokens)
        shown = []
        for n, example in enumerate(examples, 1):
            record = by_index.get(n)
            _, max_chars = token_limits(example.expected)
            expected = expected_repr(example.expected)
            if record is None:
                shown.append(VisibleResult(n, False, example.call, expected, "", error=_process_failure(result)))
                continue
            if record.get("err") is not None:
                shown.append(VisibleResult(n, False, example.call, expected, "", error=cap_long_tokens(record["err"], max_chars)))
                continue
            actual, cut = shown_value(record.get("repr", ""), expected, slack)
            shown.append(VisibleResult(n, bool(record.get("ok")), example.call, expected, actual,
                                       truncated=cut or bool(record.get("repr_cut"))))
        return VisibleReport(shown)

    def limits(self) -> list[float]:
        count = self.p["check"].input_count
        if self.reference_times is None or len(self.reference_times) != count:
            raise ConfigError(
                f"{self.task.task_id}: no reference times for its {count} hidden inputs; run "
                "`python -m hallu_harness audit-humaneval` first"
            )
        tl = self.evaluation.hidden_time_limit
        return [max(tl.floor_s, tl.reference_multiplier * t) for t in self.reference_times]

    def run_hidden_records(self, code: str, limits: list[float], total_timeout: float) -> tuple[list[dict], object]:
        """Raw per-input records (verdict and seconds); the audit uses these
        to time the reference solution."""
        check = self.p["check"]
        payload = {
            **self._base_payload(code, "hidden"),
            "check_prelude": check.prelude,
            "loop_body": check.loop_body,
            "limits": limits,
        }
        return self._run(payload, total_timeout)

    def run_hidden(self, code: str) -> HiddenReport:
        check = self.p["check"]
        limits = self.limits()
        records, result = self.run_hidden_records(code, limits, self.evaluation.hidden_total_timeout_s)
        return hidden_report_from_records(records, check.input_count, result, limits, no_visible=self.visible_test_count == 0)


def _process_failure(result) -> str:
    if result.timed_out:
        return "the process ran out of time"
    if getattr(result, "output_limit_hit", False):
        return "the process printed more output than the sandbox captures"
    return f"the process ended unexpectedly (exit code {result.exit_code})" + (f": {result.stderr.strip()[-500:]}" if result.stderr.strip() else "")


def hidden_report_from_records(records: list[dict], count: int, result, limits, no_visible: bool) -> HiddenReport:
    load_error = next((r["load_error"] for r in records if "load_error" in r), None)
    verdicts: dict[int, str] = {r["i"]: r["v"] for r in records if "i" in r and "v" in r}
    seconds = [r.get("t", 0.0) for r in records if "i" in r]
    if load_error is not None:
        verdicts = {}
        missing = "RUNTIME_ERROR"
    elif result.timed_out:
        missing = "TIME_LIMIT_EXCEEDED"
    else:
        missing = "RUNTIME_ERROR"
    per_test = [verdicts.get(i, missing) for i in range(count)]
    failed = [i + 1 for i, v in enumerate(per_test) if v != "ACCEPTED"]
    counts: dict[str, int] = {}
    for v in per_test:
        if v != "ACCEPTED":
            counts[v] = counts.get(v, 0) + 1
    breakdown = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    overall = next((v for v in FAILURE_ORDER if v in counts), "ACCEPTED")
    note = ""
    if load_error is not None:
        note = "the solution failed to load"
    elif len(verdicts) < count:
        note = f"{count - len(verdicts)} test(s) did not report a result ({_process_failure(result)})"
    return HiddenReport(
        verdict=overall,
        passed=count - len(failed),
        total=count,
        breakdown=breakdown,
        failed_numbers=failed,
        max_seconds=max(seconds, default=0.0),
        time_limit={"per_input_min": min(limits, default=None), "per_input_max": max(limits, default=None)},
        note=note,
        no_visible_tests=no_visible,
    )


class HumanEvalBenchmark(Benchmark):
    name = "humaneval"
    Selection = Selection
    Evaluation = Evaluation
    Prompt = Prompt

    @classmethod
    def validate_config(cls, cfg) -> None:
        super().validate_config(cfg)
        value_cut(cfg.tools.params.get("grade_code", {}).get("max_output_tokens", "expected"))
        times = cls.Evaluation(**cfg.evaluation).reference_times
        if not times or not resolve_repo_path(times).exists():
            # They set every hidden test's time limit, and the final solution
            # is graded on the hidden tests even with hidden_tests off.
            raise ConfigError(
                f"dataset evaluation (humaneval): reference_times file {times!r} not found; run "
                "`python -m hallu_harness audit-humaneval`"
            )

    def make_texts(self):
        ev = self.evaluation
        slack = value_cut(self.cfg.tools.params.get("grade_code", {}).get("max_output_tokens", "expected"))
        return HumanEvalTexts(ev.visible_time_limit_s, ev.hidden_time_limit.floor_s, ev.hidden_time_limit.reference_multiplier, slack,
                              ev.hidden_total_timeout_s)

    def _sources(self):
        return load_sources(self.selection.humaneval_revision, self.selection.humanevalplus_revision)

    def exclusions(self) -> dict[str, dict]:
        path = self.selection.excluded_tasks
        if not path:
            return {}
        resolved = resolve_repo_path(path)
        if not resolved.exists():
            raise ConfigError(f"excluded_tasks file not found: {path}; run `python -m hallu_harness audit-humaneval`")
        return json.loads(resolved.read_text()).get("excluded", {})

    @property
    def grading_inputs(self) -> dict:
        times = self.evaluation.reference_times
        resolved = resolve_repo_path(times) if times else None
        return {"reference_times_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest() if resolved and resolved.exists() else None}

    @property
    def info(self) -> dict:
        _, _, info = self._sources()
        path = self.selection.excluded_tasks
        digest = hashlib.sha256(resolve_repo_path(path).read_bytes()).hexdigest() if path and resolve_repo_path(path).exists() else None
        times = self.evaluation.reference_times
        times_digest = hashlib.sha256(resolve_repo_path(times).read_bytes()).hexdigest() if times and resolve_repo_path(times).exists() else None
        return {**info, "excluded_tasks_file": path, "excluded_tasks_sha256": digest,
                "excluded_task_ids": sorted(self.exclusions()), "reference_times_sha256": times_digest}

    def load(self) -> Iterator[Task]:
        he_rows, hp_rows, _ = self._sources()
        excluded = self.exclusions()
        wanted = set(self.selection.ids) if self.selection.ids else None
        rows = [(a, b) for a, b in zip(he_rows, hp_rows) if a["task_id"] not in excluded and (wanted is None or a["task_id"] in wanted)]
        if wanted:  # asked for by id: say why any will not run
            known = {a["task_id"] for a in he_rows}
            for tid in self.selection.ids:
                if tid not in known:
                    raise ConfigError(f"unknown HumanEval task id {tid!r} (ids look like 'HumanEval/0')")
                if tid in excluded:
                    print(f"[skip] {tid}: listed in excluded_tasks ({', '.join(excluded[tid].get('reasons', [])) or 'see the file'})", file=sys.stderr)
        if wanted:
            order = {tid: n for n, tid in enumerate(self.selection.ids)}
            rows.sort(key=lambda ab: order[ab[0]["task_id"]])
        if self.selection.shuffle:
            random.Random(self.selection.seed).shuffle(rows)
        if self.selection.limit is not None:
            rows = rows[: self.selection.limit]
        for a, b in rows:
            yield task_from_rows(a, b)

    def make_evaluator(self, task: Task, grade_params: dict) -> Evaluator:
        value_cut(grade_params.get("max_output_tokens", "expected"))
        times = load_reference_times(self.evaluation.reference_times).get(task.task_id)
        evaluator = HumanEvalEvaluator(task, self.evaluation, times, grade_params.get("max_output_tokens", "expected"))
        evaluator.limits()  # no reference times for this task: fail at setup, before any model call
        return evaluator
