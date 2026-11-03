"""What a dataset plug-in provides (DESIGN.md §5.3, §8).

A benchmark knows its tasks, how to grade a solution against visible and
hidden tests, and every sentence the model is shown about the task. It never
knows which framework is running: the framework's reply phrases arrive as a
`Protocol`.
"""
from __future__ import annotations

import ast
import hashlib
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

from pydantic import BaseModel, ValidationError

from ..core.config import DatasetConfig
from ..core.errors import ConfigError
from ..core.types import FrameworkResult, Task
from ..core.protocol import Protocol

# ----------------------------------------------------------------- reports


@dataclass
class VisibleResult:
    """One visible test, as shown to the model."""

    index: int
    passed: bool
    input_text: str  # what was run: a sample's stdin, or a call expression
    expected_text: str
    actual_text: str  # already cut to the display rule (DESIGN.md §7.5)
    truncated: bool = False  # the "(output truncated)" note applies
    error: str | None = None
    stopped_for_extra_output: bool = False
    extra_output: bool = False  # printed more tokens than the expected output has (stopped or not)

    def private(self) -> dict:
        return {
            "index": self.index,
            "passed": self.passed,
            "input": self.input_text,
            "expected_output": self.expected_text,
            "actual_output": self.actual_text,
            "truncated": self.truncated,
            "error": self.error,
            "stopped_for_extra_output": self.stopped_for_extra_output,
            "extra_output": self.extra_output,
        }


@dataclass
class VisibleReport:
    results: list[VisibleResult]

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)


CONCLUSIVE_FAILURES = ("WRONG_ANSWER", "RUNTIME_ERROR", "TIME_LIMIT_EXCEEDED", "MEMORY_LIMIT_EXCEEDED")


@dataclass
class HiddenReport:
    """The hidden-test result. Counts, verdicts and test numbers only: hidden
    test content never enters a report that can reach the model or a file."""

    verdict: str  # ACCEPTED | WRONG_ANSWER | RUNTIME_ERROR | TIME_LIMIT_EXCEEDED | MEMORY_LIMIT_EXCEEDED | UNVERIFIABLE | NO_SOLUTION
    passed: int = 0
    total: int = 0
    breakdown: list[tuple[str, int]] = field(default_factory=list)  # most common first
    failed_numbers: list[int] = field(default_factory=list)
    stopped_numbers: list[int] = field(default_factory=list)  # stopped for extra output
    max_seconds: float = 0.0
    time_limit: Any = None
    note: str = ""
    no_visible_tests: bool = False  # the task has no visible test, so none ran before these

    @property
    def accepted(self) -> bool:
        return self.verdict == "ACCEPTED"

    @property
    def conclusive(self) -> bool:
        return self.verdict not in ("UNVERIFIABLE", "NO_SOLUTION")

    @property
    def failed(self) -> int:
        return len(self.failed_numbers)

    def private(self) -> dict:
        return {
            "verdict": self.verdict,
            "passed": self.passed,
            "total": self.total,
            "failure_breakdown": dict(self.breakdown),
            "failed_test_numbers": self.failed_numbers,
            "stopped_for_extra_output_test_numbers": self.stopped_numbers,
            "max_seconds": round(self.max_seconds, 4),
            "time_limit": self.time_limit,
            "note": self.note,
            "no_visible_tests": self.no_visible_tests,
        }


# ----------------------------------------------------------------- environment


@dataclass
class GradeCall:
    number: int
    code: str
    ran: bool
    visible: VisibleReport | None = None
    hidden: HiddenReport | None = None
    repeated_from: int | None = None


class Evaluator(ABC):
    @property
    @abstractmethod
    def visible_test_count(self) -> int: ...

    @abstractmethod
    def run_visible(self, code: str) -> VisibleReport: ...

    @abstractmethod
    def run_hidden(self, code: str) -> HiddenReport: ...

    def load_problem(self, code: str) -> str | None:
        """Why the code cannot even be loaded (a syntax error, a missing
        definition), for tasks with no visible tests to reveal it; None if it
        loads. Not a test: reporting it is not a hidden-test disclosure."""
        return None


DEFAULT_SOLUTION_POLICY = ("accepted", "final_answer_code", "last_submitted")


class TaskEnv:
    """Per-trajectory state shared by the tools: the evaluator, the verified
    solution, every grading call, and a cache of hidden results by program.

    It also carries the run settings the task instructions describe (hidden
    tests on or off, the review turn, the solution policy), so the
    instructions are rendered from the same values the run obeys."""

    def __init__(self, task: Task, evaluator: Evaluator, texts: "Texts", protocol: Protocol, hidden_enabled: bool,
                 review_turn: bool, solution_policy: tuple[str, ...] = DEFAULT_SOLUTION_POLICY):
        self.task = task
        self.evaluator = evaluator
        self.texts = texts
        self.protocol = protocol
        self.hidden_enabled = hidden_enabled
        self.review_turn = review_turn
        self.solution_policy = tuple(solution_policy)
        self.calls: list[GradeCall] = []
        self.accepted_code: str | None = None
        self.accepted_report: HiddenReport | None = None
        self.review_offered = False
        self.submitted_code: str | None = None
        self._hidden_cache: dict[str, HiddenReport] = {}

    def hidden(self, code: str) -> HiddenReport:
        key = hashlib.sha256(code.encode("utf-8")).hexdigest()
        if key not in self._hidden_cache:
            self._hidden_cache[key] = self.evaluator.run_hidden(code)
        return self._hidden_cache[key]

    @property
    def last_submitted_code(self) -> str | None:
        for call in reversed(self.calls):
            if call.ran:
                return call.code
        return None


# ----------------------------------------------------------------- texts

# The interpreter every program runs under: the sandbox runs `sys.executable`.
PYTHON_VERSION = f"Python {sys.version_info.major}.{sys.version_info.minor}"


def ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 4th, ..., 11th, 12th, 13th, 21st, ..."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


class Texts(ABC):
    """Every sentence a benchmark shows the model, parameterized by protocol.

    `task_instructions` is rendered from the trajectory's `TaskEnv` and its
    reply budget, so every statement it makes about grading, the review turn,
    the budget and the final solution holds for the run it is shown in.
    `resubmit=False` drops the closing "send a corrected program" sentence
    from failure feedback; it is used once a verified solution exists, when
    the model is told to finish instead.
    """

    code_noun = "program"
    # How the solution kept by the `accepted` policy is described:
    # "the first {code_noun} that {accepted_phrase}".
    accepted_phrase = "passed every hidden test"
    # The visible tests, as the instructions name them: "every {visible_tests}".
    visible_tests = "visible test"

    @abstractmethod
    def task_instructions(self, env: TaskEnv, max_replies: int) -> str: ...

    @abstractmethod
    def user_message(self, task: Task, protocol: Protocol) -> str: ...

    @abstractmethod
    def grade_code_description(self, hidden_tests: bool) -> str: ...

    @abstractmethod
    def code_argument_description(self) -> str: ...

    @abstractmethod
    def visible_feedback(self, report: VisibleReport, protocol: Protocol, resubmit: bool = True) -> str: ...

    @abstractmethod
    def hidden_failed(self, report: HiddenReport, protocol: Protocol, resubmit: bool = True) -> str: ...

    @abstractmethod
    def hidden_passed(self, report: HiddenReport, protocol: Protocol, offer_review: bool) -> str: ...

    @abstractmethod
    def verified_again(self, report: HiddenReport, protocol: Protocol) -> str: ...

    @abstractmethod
    def unverifiable(self, report: HiddenReport, protocol: Protocol) -> str: ...

    @abstractmethod
    def visible_passed_no_hidden(self, protocol: Protocol, ran_visible: bool = True) -> str:
        """With hidden tests off: the visible tests passed, or (`ran_visible`
        False) the task has none and the code loaded."""

    @abstractmethod
    def already_verified(self, protocol: Protocol) -> str: ...

    @abstractmethod
    def empty_code(self, protocol: Protocol) -> str: ...

    @abstractmethod
    def repeat_warning(self, previous: int, protocol: Protocol) -> str:
        """`previous`: the earlier identical program's position among the
        programs graded so far (1 = the first program that was run)."""

    @abstractmethod
    def revision_safety_note(self, protocol: Protocol) -> str: ...

    # -- shared sections of the task instructions -----------------------------

    def solution_rule(self, env: TaskEnv) -> str:
        """Which program is graded as the solution: `solution_policy`, in order,
        as `Benchmark.final_grade` applies it."""
        noun, protocol = self.code_noun, env.protocol
        phrases = {
            "accepted": f"the first {noun} that {self.accepted_phrase}" if env.hidden_enabled else "",
            "final_answer_code": protocol.fill("final_answer_source", noun),
            "last_submitted": f"the last {noun} you sent to be graded",
            "submitted": f"the {noun} you last passed to the submit tool",
        }
        items = [phrases[name] for name in env.solution_policy if phrases.get(name)]
        if not items:
            return "No program is graded as your solution."
        rule = "Your solution is " + "; if there is none, ".join(items) + "."
        if env.hidden_enabled:
            return rule + " A solution not yet graded on the hidden tests is graded on them after the task ends."
        return rule + " It is graded on the hidden tests after the task ends; you are not told the result."

    def ending_section(self, env: TaskEnv, max_replies: int) -> str:
        """The "Ending the task" section, identical in structure for every
        benchmark: the budget, when and how to finish, giving up, and which
        program becomes the solution."""
        noun, p = self.code_noun, env.protocol
        if not env.hidden_enabled:
            when = f"Once a grading result tells you that your {noun} passed every {self.visible_tests}, you may finish."
        elif env.review_turn:
            when = (
                f"When a grading result tells you that a {noun} {self.accepted_phrase}, you may either finish at once "
                f"or first revise the {noun} once and have the revision graded. Whatever the revision's result, the "
                f"{noun} that already passed remains your solution. After that one revision, finish."
            )
        else:
            when = f"When a grading result tells you that a {noun} {self.accepted_phrase}, finish."
        lines = [
            "# Ending the task",
            "",
            f"- You may send at most {max_replies} replies. Once they are used up, no further {noun} is graded, "
            "and your solution is chosen as described below.",
            f"- {when}",
            f"- To finish the task (to submit), {p.fill('finish_rule', noun)}. Do not finish before a grading result "
            "tells you that you may.",
            "- If you have tried several substantially different approaches and cannot solve the task, "
            f"{p.fill('give_up', noun)} instead of sending {noun}s you expect to fail.",
            f"- {self.solution_rule(env)}",
        ]
        return "\n".join(lines)


# ----------------------------------------------------------------- code extraction

_FENCE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def is_sorry(answer: str | None) -> bool:
    return isinstance(answer, str) and answer.strip().strip('."\'` ').upper() == "SORRY"


def extract_code(answer: str | None) -> str | None:
    """The program in a final answer: the last fenced block outside any think
    block; else the whole answer if it is itself valid, non-empty Python."""
    if not isinstance(answer, str) or is_sorry(answer):
        return None
    visible = _THINK.sub("", answer)
    if re.search(r"<think>", visible, re.IGNORECASE):
        return None
    blocks = [b.strip("\n") for b in _FENCE.findall(visible) if b.strip()]
    if blocks:
        return blocks[-1]
    candidate = visible.strip("\n")
    if not candidate.strip():
        return None
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return None
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Expr, ast.Assign, ast.For, ast.While, ast.If, ast.Import, ast.ImportFrom)) for node in tree.body):
        return None
    # A bare name or a sentence that happens to parse ("done") is not a program.
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, (ast.Name, ast.Constant)):
        return None
    return candidate


# ----------------------------------------------------------------- benchmark


class Benchmark(ABC):
    name: str = ""
    Selection: type[BaseModel]
    Evaluation: type[BaseModel]
    Prompt: type[BaseModel]

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.selection = self.Selection(**cfg.selection)
        self.evaluation = self.Evaluation(**cfg.evaluation)
        self.prompt = self.Prompt(**cfg.prompt)
        self.texts = self.make_texts()

    @classmethod
    def validate_config(cls, cfg: DatasetConfig) -> None:
        for section, model in (("selection", cls.Selection), ("evaluation", cls.Evaluation), ("prompt", cls.Prompt)):
            try:
                model(**getattr(cfg, section))
            except ValidationError as exc:
                errors = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
                raise ConfigError(f"dataset {section} ({cls.name}): {errors}") from None
        evaluation = cls.Evaluation(**cfg.evaluation)
        policy = list(getattr(evaluation, "solution_policy", DEFAULT_SOLUTION_POLICY))
        if getattr(evaluation, "hidden_tests", True) and policy[:1] != ["accepted"]:
            raise ConfigError(
                f"dataset evaluation ({cls.name}): with hidden_tests on, solution_policy must start with 'accepted': "
                "the instructions and the grading feedback tell the model that the first program to pass every "
                "hidden test is kept as its solution"
            )

    @abstractmethod
    def make_texts(self) -> Texts: ...

    @abstractmethod
    def load(self) -> Iterator[Task]:
        """Tasks in a deterministic order for a given config."""

    @abstractmethod
    def make_evaluator(self, task: Task, grade_params: dict) -> Evaluator:
        """`grade_params` are the dataset's `tools.params.grade_code` (e.g.
        `max_output_tokens`, the display rule of DESIGN.md §7.5)."""

    @property
    def info(self) -> dict:
        """Dataset provenance for the trajectory (revisions, exclusion lists)."""
        return {}

    @property
    def grading_inputs(self) -> dict:
        """Hashes of data files that change how a program is graded (e.g.
        HumanEval's reference timings, which set the time limits). They are
        part of the experiment key, so a resumed run never mixes
        trajectories graded under different files."""
        return {}

    def make_env(self, task: Task, protocol: Protocol, tool_params: dict) -> TaskEnv:
        grade_params = tool_params.get("grade_code", {})
        return TaskEnv(
            task=task,
            evaluator=self.make_evaluator(task, grade_params),
            texts=self.texts,
            protocol=protocol,
            hidden_enabled=bool(getattr(self.evaluation, "hidden_tests", True)),
            review_turn=bool(grade_params.get("review_turn", True)),
            solution_policy=tuple(getattr(self.evaluation, "solution_policy", DEFAULT_SOLUTION_POLICY)),
        )

    # -- the final grade ----------------------------------------------------

    def final_grade(self, env: TaskEnv, result: FrameworkResult) -> dict:
        """Applies the dataset's solution policy (DESIGN.md §7.4).

        The solution is chosen in policy order — the program verified on the
        hidden tests, then code in the final answer, then the last program
        submitted for grading — and graded on the hidden tests if it has not
        been already. What the model claimed is kept separately.
        """
        policy = env.solution_policy
        candidates = {
            "accepted": env.accepted_code,
            "final_answer_code": extract_code(result.final_answer),
            "last_submitted": env.last_submitted_code,
            "submitted": env.submitted_code,
        }
        source, solution = "none", None
        for name in policy:
            if candidates.get(name):
                source, solution = name, candidates[name]
                break

        if source == "accepted":
            report = env.accepted_report
        elif solution is not None and self.grades_final_solution:
            report = env.hidden(solution)
        else:
            report = None
        return {
            "final_solution": solution,
            "solution_source": source,
            "grade": report.private() if report is not None else None,
            "solved": bool(report is not None and report.accepted),
        }

    grades_final_solution = True
