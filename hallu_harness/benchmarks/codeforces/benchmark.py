"""Codeforces (`open-r1/codeforces`), as a hallu_harness benchmark.

A thin wrapper over `cf_agent`: problems come from `cf_agent.dataset`, sample
runs from `cf_agent.sandbox.samples`, hidden grading from
`cf_agent.judge.judge_solution` (what `check_hidden_tests` runs) — so
`cf_agent` and every framework here grade identically, with the corrected
token comparison (DESIGN.md §8.1.1) and the early stop for over-printing
programs (§8.1.2).

Only fixed-answer problems are used: the eligibility filter excludes every
problem that ships a checker, and `excluded_tasks.json` lists the problems
that accept several answers without one (§8.1).

The hidden tests are fetched once, when the trajectory is set up, not during
grading. A failure to fetch them is the harness's, not the model's: it is
raised before the first model call and the trajectory is recorded as
crashed. `check_hidden_tests` is not used because it reports every such
failure as UNVERIFIABLE, which the model would then be shown as a gap in the
test data. For an eligible problem nothing else can make grading
UNVERIFIABLE; for a problem run with `eligible_only: false`, a checker, a
non-stdio problem or missing test data is a fact about the problem, and is
still reported as UNVERIFIABLE.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from cf_agent.dataset import catalog
from cf_agent.dataset.problem import DEFAULT_TIME_LIMIT_MULTIPLIER, Problem
from cf_agent.dataset.testdata import FullTests, load_full_tests
from cf_agent.judge.grader import judge_solution
from cf_agent.judge.report import JudgeReport
from cf_agent.judge.verdicts import Verdict
from ..exec_lab_sandbox import (
    cap_long_tokens,
    default_timeout,
    display_output,
    run_against_examples,
    split_tokens,
    token_limits,
)

from ...core.config import resolve_repo_path
from ...core.errors import ConfigError, HarnessFailure
from ...core.types import Task
from ..base import Benchmark, Evaluator, HiddenReport, VisibleReport, VisibleResult
from .texts import CodeforcesTexts

EXCLUSIONS_PATH = Path(__file__).with_name("excluded_tasks.json")


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[str] | None = None
    contest: str | None = None
    min_rating: int | None = None
    max_rating: int | None = None
    tags: list[str] | None = None
    eligible_only: bool = True
    limit: int | None = Field(None, gt=0)
    shuffle: bool = False
    seed: int = 0
    excluded_tasks: str | None = str(EXCLUSIONS_PATH)


class Evaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hidden_tests: bool = True
    time_limit_multiplier: float = Field(DEFAULT_TIME_LIMIT_MULTIPLIER, gt=0)
    comparison: Literal["tokens_yesno_ci"] = "tokens_yesno_ci"
    solution_policy: list[Literal["accepted", "final_answer_code", "last_submitted", "submitted"]] = Field(
        default_factory=lambda: ["accepted", "final_answer_code", "last_submitted"]
    )


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_tags: Literal[False] = False  # tags are dropped in cf_agent.dataset; see its README


def problem_ineligibility(problem: Problem) -> str | None:
    """`catalog.eligibility_problem`, decided from a loaded `Problem`."""
    if problem.has_special_judge:
        return "special judge (ships a checker)"
    if not problem.examples:
        return "no sample tests"
    if not problem.generated_test_count:
        return "no generated hidden tests"
    if problem.input_mode != "stdio":
        return f"input_mode is {problem.input_mode!r}"
    return None


def problem_metadata(problem: Problem, split: str) -> dict:
    return {
        "id": problem.id,
        "split": split,
        "title": problem.title,
        "contest_name": problem.contest_name,
        "rating": problem.rating,
        "time_limit": problem.time_limit,
        "enforced_time_limit": problem.enforced_time_limit(),
        "memory_limit": problem.memory_limit,
        "generated_test_count": problem.generated_test_count,
        "has_special_judge": problem.has_special_judge,
        "input_mode": problem.input_mode,
        "visible_test_count": len(problem.examples),
        "examples": [{"input": e.input, "output": e.output} for e in problem.examples],
        "description": problem.description,
        "input_format": problem.input_format,
        "output_format": problem.output_format,
        "note": problem.note,
        "rendered": problem.render(),
    }


def display_limit(setting, expected_tokens: int) -> int | None:
    """`max_output_tokens`: "expected" (default) = the expected output's token
    count; an integer = that many; None = no cap (DESIGN.md §7.5)."""
    if setting == "expected":
        return expected_tokens
    if setting is None:
        return None
    if isinstance(setting, int) and setting >= 0:
        return setting
    raise ValueError(f"max_output_tokens must be 'expected', a non-negative integer or null, not {setting!r}")


def ungradable(tests: FullTests) -> str | None:
    """Why a fetched test set cannot be graded by exact match — the checks of
    `cf_agent.judge.grader._refuse_to_grade` that concern the data."""
    if tests.has_special_judge:
        return "the test data ships a checker"
    if tests.input_mode != "stdio":
        return f"the test data's input_mode is {tests.input_mode!r}"
    if not tests.cases:
        return "the test data holds no test cases"
    return None


class CodeforcesEvaluator(Evaluator):
    """`tests` is the problem's full test set; None, with `unavailable`
    saying why, only for an ineligible problem that has none."""

    def __init__(self, problem: Problem, split: str, time_limit_multiplier: float, max_output_tokens="expected",
                 tests: FullTests | None = None, unavailable: str | None = None):
        self.problem = problem
        self.split = split
        self.multiplier = time_limit_multiplier
        self.max_output_tokens = max_output_tokens
        self.tests = tests
        self.unavailable = unavailable

    @property
    def visible_test_count(self) -> int:
        return len(self.problem.examples)

    def run_visible(self, code: str) -> VisibleReport:
        problem = self.problem
        results = run_against_examples(
            code, problem.examples, default_timeout(problem, self.multiplier), problem.memory_limit
        )
        return visible_report(results, self.max_output_tokens)

    def run_hidden(self, code: str) -> HiddenReport:
        if self.tests is None:
            report = hidden_report(JudgeReport(self.problem.id, Verdict.UNVERIFIABLE, note=self.unavailable or ""))
        else:
            report = hidden_report(judge_solution(code, self.problem, self.tests, self.multiplier))
        report.no_visible_tests = not self.problem.examples  # only possible with eligible_only: false
        return report


def visible_report(results, max_output_tokens="expected") -> VisibleReport:
    """cf_agent sample results -> what the model is shown (DESIGN.md §7.5)."""
    shown: list[VisibleResult] = []
    for index, result in enumerate(results, 1):
        max_tokens, max_chars = token_limits(result.example.output)
        expected, _ = display_output(result.example.output)
        actual, truncated = display_output(result.actual_output, display_limit(max_output_tokens, max_tokens), max_chars)
        error = cap_long_tokens(result.error, max_chars) if result.error else None
        # Stopped at the first token too many, or finished before the stop
        # could act: either way the output has more tokens than the expected.
        extra = result.stopped_for_extra_output or len(split_tokens(result.actual_output)) > max_tokens
        shown.append(
            VisibleResult(
                index=index,
                passed=result.passed,
                input_text=result.example.input,
                expected_text=expected,
                actual_text=actual,
                truncated=truncated or result.stopped_for_extra_output,
                error=error,
                stopped_for_extra_output=result.stopped_for_extra_output,
                extra_output=extra,
            )
        )
    return VisibleReport(shown)


def hidden_report(report) -> HiddenReport:
    """cf_agent JudgeReport -> counts, verdicts and test numbers only."""
    return HiddenReport(
        verdict=report.verdict.value,
        passed=report.passed,
        total=report.total,
        breakdown=[(verdict.value, count) for verdict, count in report.failure_breakdown],
        failed_numbers=[o.number for o in report.failures],
        stopped_numbers=[o.number for o in report.outcomes if o.stopped_for_extra_output],
        max_seconds=report.max_seconds,
        time_limit=report.time_limit,
        note=report.note,
    )


def cap_long_tokens(text: str, max_chars: int) -> str:
    """Error text keeps its whitespace (tracebacks are indented); only a
    runaway token is cut (DESIGN.md §7.5)."""
    out, run = [], []
    for ch in text:
        if ch in " \t\n\r\v\f":
            out.append("".join(run[:max_chars]))
            run = []
            out.append(ch)
        else:
            run.append(ch)
    out.append("".join(run[:max_chars]))
    return "".join(out)


class CodeforcesBenchmark(Benchmark):
    name = "codeforces"
    Selection = Selection
    Evaluation = Evaluation
    Prompt = Prompt

    def __init__(self, cfg):
        super().__init__(cfg)
        # The last problem's test set: a task's samples run one after another,
        # and one problem's tests can be large.
        self._tests: tuple[str, FullTests | None, str | None] | None = None

    def make_texts(self):
        return CodeforcesTexts(self.evaluation.time_limit_multiplier)

    def hidden_tests(self, problem: Problem) -> tuple[FullTests | None, str | None]:
        """(test set, None), or (None, why) for an ineligible problem without
        one. Raises HarnessFailure when an eligible problem's test set is
        missing or cannot be graded; any other failure to fetch it (network,
        cache) propagates as it is."""
        if self._tests is not None and self._tests[0] == problem.id:
            return self._tests[1], self._tests[2]
        eligible = problem_ineligibility(problem) is None
        try:
            tests, unavailable = load_full_tests(problem.id, split=self.split), None
        except LookupError as exc:
            if eligible:
                raise HarnessFailure(f"{problem.id}: no hidden tests found: {exc}") from exc
            tests, unavailable = None, str(exc)
        if tests is not None and eligible:
            reason = ungradable(tests)
            if reason is not None:
                raise HarnessFailure(f"{problem.id} is eligible by its dataset row, but {reason}")
        self._tests = (problem.id, tests, unavailable)
        return tests, unavailable

    @property
    def split(self) -> str:
        return self.cfg.split or catalog.DEFAULT_SPLIT

    def exclusions(self) -> dict[str, dict]:
        path = self.selection.excluded_tasks
        if not path:
            return {}
        try:
            doc = json.loads(resolve_repo_path(path).read_text())
        except FileNotFoundError:
            raise ConfigError(f"excluded_tasks file not found: {path}") from None
        return doc.get("excluded", {}).get(self.split, {})

    @property
    def info(self) -> dict:
        path = self.selection.excluded_tasks
        digest = hashlib.sha256(resolve_repo_path(path).read_bytes()).hexdigest() if path else None
        return {
            "dataset": catalog.DATASET_NAME,
            "config": catalog.DATASET_CONFIG,
            "split": self.split,
            "excluded_tasks_file": path,
            "excluded_tasks_sha256": digest,
            "excluded_task_ids": sorted(self.exclusions()),
            "comparison": "open-r1 normalize_output: tokens, whitespace-insensitive, YES/NO case-insensitive",
        }

    def _candidates(self) -> list[Problem]:
        s = self.selection
        if s.ids:
            return [catalog.get_problem_by_id(pid, split=self.split) for pid in s.ids]
        if s.contest:
            return catalog.get_problems_by_contest(s.contest, splits=(self.split,), eligible_only=False)
        return list(catalog.iter_problems(self.split, s.min_rating, s.max_rating, s.tags, eligible_only=False))

    def load(self) -> Iterator[Task]:
        s = self.selection
        excluded = self.exclusions()
        problems = []
        for problem in self._candidates():
            reason = problem_ineligibility(problem) if s.eligible_only else None
            if reason is None and problem.id in excluded:
                reason = "listed in excluded_tasks"
            if reason is not None:
                if s.ids:  # asked for by id: say why it will not run
                    print(f"[skip] {problem.id}: {reason}", file=sys.stderr)
                continue
            problems.append(problem)
        if s.shuffle:
            random.Random(s.seed).shuffle(problems)
        if s.limit is not None:
            problems = problems[: s.limit]
        for problem in problems:
            yield Task(
                task_id=problem.id,
                query=problem.render(),
                metadata=problem_metadata(problem, self.split),
                payload={"problem": problem},
            )

    def make_evaluator(self, task: Task, grade_params: dict) -> Evaluator:
        display_limit(grade_params.get("max_output_tokens", "expected"), 0)  # validate early
        problem = task.payload["problem"]
        # Fetched even with hidden_tests off: the final solution is graded on them.
        tests, unavailable = self.hidden_tests(problem)
        return CodeforcesEvaluator(
            problem,
            self.split,
            self.evaluation.time_limit_multiplier,
            grade_params.get("max_output_tokens", "expected"),
            tests=tests,
            unavailable=unavailable,
        )
