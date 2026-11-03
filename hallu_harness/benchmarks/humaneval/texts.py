"""Every sentence the model is shown about a HumanEval task.

The same structure and policy as Codeforces (`benchmarks/codeforces/texts.py`)
— visible results in full under the display rule, hidden results as counts
and failure types only, one review turn, the framework's reply phrases from
its `Protocol` — with "function" for "program" and the docstring's examples
as the visible tests. The task instructions describe how the code is loaded
(`runner_program.py`) and are rendered from the evaluation's time limits.

One display rule differs from Codeforces: a failed example's values are shown
exactly as `repr()` prints them, never with whitespace collapsed, because the
spaces inside a string decide equality. A returned value is cut by
characters, `SHOWN_VALUE_SLACK_CHARS` past the expected repr's length. The
first character at which the two reprs differ always lies within the
expected repr's length + 1, so the difference is always on screen; the cut
only stops a runaway value from flooding the context.
"""
from __future__ import annotations

from cf_agent.sandbox.process import WALL_TIME_FACTOR, WALL_TIME_SLACK_S

from ...core.protocol import Protocol
from ..base import PYTHON_VERSION, HiddenReport, TaskEnv, Texts, VisibleReport, ordinal
from ..codeforces.texts import TRUNCATION_NOTE, indent
from .runner_program import REPR_CAP_CHARS

SHOWN_VALUE_SLACK_CHARS = 50

VERDICT_HINTS = {
    "WRONG_ANSWER": "a logic error: your function returns the wrong value on some inputs.",
    "RUNTIME_ERROR": "your function raises an exception on some input — check your assumptions about the inputs it can receive (consider all edge cases).",
    "TIME_LIMIT_EXCEEDED": "your function is too slow on some inputs — reconsider its complexity.",
    "MEMORY_LIMIT_EXCEEDED": "your function uses too much memory on some input — reconsider what you are storing.",
}
DEFAULT_HINT = "a mismatch between your function's behavior and the problem specification."


class HumanEvalTexts(Texts):
    code_noun = "function"
    visible_tests = "docstring example"

    def __init__(self, visible_time_limit_s: float = 5.0, hidden_floor_s: float = 1.0, hidden_reference_multiplier: float = 4.0,
                 value_slack_chars: int | None = SHOWN_VALUE_SLACK_CHARS, hidden_total_timeout_s: float = 600.0):
        self.visible_time_limit_s = visible_time_limit_s
        self.hidden_floor_s = hidden_floor_s
        self.hidden_reference_multiplier = hidden_reference_multiplier
        self.value_slack_chars = value_slack_chars  # None: cut only at the sandbox's REPR_CAP_CHARS
        self.hidden_total_timeout_s = hidden_total_timeout_s

    def task_instructions(self, env: TaskEnv, max_replies: int) -> str:
        lines = [
            "You are an expert Python programmer. Your task is to implement one Python function, given its "
            f"signature and docstring, in {PYTHON_VERSION}.",
            "",
            "Your replies are processed automatically; no person reads them while the task is running. Do not ask "
            "questions or wait for confirmation.",
            "",
            "# How a function is graded",
            "",
            "Your code is executed after the code given in the task (its imports, any helper functions, and the "
            "function's signature and docstring), in the same namespace, so it may use the names defined there. Code "
            'under `if __name__ == "__main__":` is not executed. If your code cannot be executed, or does not define '
            "the function, you are shown the error and nothing else is run. Otherwise:",
            "",
            "1. The function is called on each example in its docstring that the grader can read: a doctest (a "
            "line `>>> call` followed by the expected value), a call with literal arguments followed by `==`, "
            "`->`, `=>` or a similar marker and a literal expected value, or an `Input:` line with a literal "
            "followed by an `Output:` line with a literal. Other examples, such as one whose expected value is "
            "written as arithmetic, are not run. Every example that can be read is run, even after one fails.",
            "2. If any example fails, grading stops there. For each failed example you are shown the call, the "
            "expected result, and the value your function returned or the error it raised.",
        ]
        if env.hidden_enabled:
            lines += [
                "3. If every example passes, or the docstring has no example that can be run, the function is called "
                "on the task's hidden tests. If any hidden test fails, you are told how many hidden tests there are, "
                "how many failed, and how many failed with each verdict: WRONG_ANSWER, RUNTIME_ERROR, "
                "TIME_LIMIT_EXCEEDED or MEMORY_LIMIT_EXCEEDED. You are never shown a hidden test's arguments, its "
                "expected result, your function's result on it, or which hidden tests failed.",
                "4. If every hidden test passes, you are told so.",
            ]
        else:
            lines.append("3. Hidden tests are not run while the task is running. If every example passes, you are told so.")
        lines += [
            "",
            "A test passes if the value your function returns equals the expected value (==); floating-point values "
            "are compared with a small tolerance. A wrong value fails the test with WRONG_ANSWER; an exception raised "
            "by your function fails it with RUNTIME_ERROR, or with MEMORY_LIMIT_EXCEEDED if it is a MemoryError; "
            "exceeding the time limit fails it with TIME_LIMIT_EXCEEDED. In the report of a failed example, the "
            "expected value and the value your function returned are both shown exactly as repr() prints them. "
            f"{self._cut_rule()}",
            "",
            "# Limits",
            "",
            f"- Each docstring example has a time limit of {self.visible_time_limit_s:g} seconds of CPU time.",
        ]
        if env.hidden_enabled:
            lines += [
                f"- Each hidden test has its own time limit: {self.hidden_reference_multiplier:g} times the CPU "
                f"time a reference solution takes on that test, and at least {self.hidden_floor_s:g} "
                f"second{'' if self.hidden_floor_s == 1 else 's'}. The hidden tests include edge cases and inputs "
                "larger than the examples.",
                "- The hidden tests run one after another in a single process, which is stopped after "
                f"{self.hidden_total_timeout_s:g} seconds of real time; every hidden test that has not finished by "
                "then fails with TIME_LIMIT_EXCEEDED.",
            ]
        lines.append(
            "- A test that uses more CPU time than its limit is stopped and fails with TIME_LIMIT_EXCEEDED, as is "
            f"a test still running after {WALL_TIME_FACTOR:g} times its time limit plus {WALL_TIME_SLACK_S:g} "
            f"second{'' if WALL_TIME_SLACK_S == 1 else 's'} of real time (for example, because the function waits)."
        )
        lines += [
            "",
            "# Rules",
            "",
            "- Always send the complete function definition, together with any helper functions and imports it "
            "uses: never a fragment, a diff, or only the changed lines.",
            "- Keep the function's name and signature exactly as given.",
            "- Return the result. Do not read from standard input; anything your code prints is ignored.",
            "- Your code must not use the network or open files.",
            "",
            self.ending_section(env, max_replies),
        ]
        return "\n".join(lines)

    def user_message(self, task, protocol: Protocol) -> str:
        prompt = task.payload["prompt"].rstrip()
        entry = task.payload["entry_point"]
        return (
            f"Complete the function `{entry}` below.\n\n```python\n{prompt}\n```\n\n"
            f"{protocol.fill('initial', self.code_noun)}"
        )

    def _cut_rule(self) -> str:
        if self.value_slack_chars is None:
            limit = f"if its repr() is longer than {REPR_CAP_CHARS} characters"
        else:
            limit = f"if its repr() is more than {self.value_slack_chars} characters longer than the expected value's"
        return f'A returned value is cut short {limit}; the note "(output truncated)" marks a cut.'

    def grade_code_description(self, hidden_tests: bool) -> str:
        runs = (
            "The function is called on the examples from its docstring and, only if every example passes, on the "
            "task's hidden tests."
            if hidden_tests
            else "The function is called on the examples from its docstring."
        )
        return f"Grades a complete Python 3 implementation of the requested function and returns the result as text. {runs}"

    def code_argument_description(self) -> str:
        # No backslash: tool listings show this through repr() or JSON,
        # which would double it. The encoding is in the system prompt.
        return "The complete Python 3 function definition, with any helper functions and imports it uses."

    # ------------------------------------------------------------ feedback

    def load_error(self, error: str, protocol: Protocol, resubmit: bool = True) -> str:
        text = "Your code could not be loaded, so nothing was tested.\n  error:\n" + indent(error)
        if not resubmit:
            return text
        return text + f"\n\nFix it, then {protocol.fill('resubmit_after_visible', self.code_noun)}."

    def visible_feedback(self, report: VisibleReport, protocol: Protocol, resubmit: bool = True) -> str:
        results = report.results
        if len(results) == 1 and results[0].index == 0:  # a load error, not an example
            return self.load_error(results[0].error or "", protocol, resubmit)
        failed = [r for r in results if not r.passed]
        lines = ["I ran your function on the examples from the docstring.", ""]
        for result in results:
            if result.passed:
                lines.append(f"Example {result.index}: PASS")
                continue
            lines.append(f"Example {result.index}: FAIL")
            lines.append("  call:")
            lines.append(indent(result.input_text))
            lines.append("  expected:")
            lines.append(indent(result.expected_text))
            if result.error is None:
                lines.append("  your function returned:")
                returned = indent(result.actual_text)
                if result.truncated:
                    returned += "\n    " + TRUNCATION_NOTE
                lines.append(returned)
            else:
                lines.append("  error:")
                lines.append(indent(result.error))
            lines.append("")
        if not failed:
            lines.append(f"All {len(results)} examples passed.")
        elif not resubmit:
            lines.append(f"{len(failed)} of {len(results)} examples failed.")
        else:
            lines.append(
                f"{len(failed)} of {len(results)} examples failed. Work out why from the differences above, then "
                f"{protocol.fill('resubmit_after_visible', self.code_noun)}."
            )
        return "\n".join(lines)

    def hidden_failed(self, report: HiddenReport, protocol: Protocol, resubmit: bool = True) -> str:
        breakdown = "\n".join(f"  - {verdict}: {count}" for verdict, count in report.breakdown)
        hints = "\n".join(f"  - {verdict}: {VERDICT_HINTS.get(verdict, DEFAULT_HINT)}" for verdict, _ in report.breakdown)
        if report.no_visible_tests:
            lead = "The docstring gives no examples I can run, so I checked your function against"
            examples = ("", "", " you have not justified")
        else:
            lead = "Your function passed every example from the docstring, so I checked it against"
            examples = (
                " Passing the examples does not mean your function is correct; it only means it is correct on those "
                "inputs.",
                " (not just the examples)",
                " that only happens to work for the examples",
            )
        return (
            f"{lead} {report.total} hidden tests you cannot see. It failed {report.failed} of them, broken down "
            f"by how they failed:\n{breakdown}\n\n"
            "I will not show you the failing inputs, the expected results, or what your function returned — "
            f"reasoning your way to the bug without that crutch is the point.{examples[0]} Think critically about "
            "your approach from first principles: re-read the specification, check whether your function "
            f"satisfies every requirement in it{examples[1]}, look for edge cases (empty inputs, single elements, "
            f"negative numbers, boundaries, ties), and question any step{examples[2]}."
            f"\n\nWhat each failure type above suggests:\n{hints}"
            + (f"\n\n{protocol.fill('resubmit_after_hidden', self.code_noun)}" if resubmit else "")
        )

    def hidden_passed(self, report: HiddenReport, protocol: Protocol, offer_review: bool) -> str:
        head = (
            f"Congrats — your function passes all hidden test cases. It passed every one of {report.total} "
            "hidden tests.\n\n"
        )
        if not offer_review:
            return head + protocol.fill("finish_now", self.code_noun)
        return head + (
            "Nothing further is required of you. You may now do either of two things:\n\n"
            f"  1. {protocol.fill('submit_option', self.code_noun)}\n"
            "  2. Review your function once more first — re-read it for anything you would still change (a fragile "
            "assumption, an unhandled edge case, an unnecessarily slow step) and "
            f"{protocol.fill('revise_option', self.code_noun)}\n\n"
            "This offer is made once. If you revise, the version already verified is kept regardless, so a "
            "revision cannot cost you a correct solution — but do not change working code merely because you "
            "were asked whether you wanted to."
        )

    def verified_again(self, report: HiddenReport, protocol: Protocol) -> str:
        return (
            f"Your revised function also passes all {report.total} hidden tests. You have now used your one review "
            "turn, so there is nothing further to check. " + protocol.fill("finish_now", self.code_noun)
        )

    def unverifiable(self, report: HiddenReport, protocol: Protocol) -> str:
        return (
            "Your function passed every example. I then tried to grade it against this task's hidden tests and "
            f"could not: {report.note}. That says nothing about whether your function is correct — it is a gap in "
            "the test data, not a verdict on your solution, and there is nothing here for you to fix.\n\n"
            "Re-read your function once for anything you would still change, then finish. "
            + protocol.fill("finish_now", self.code_noun)
        )

    def visible_passed_no_hidden(self, protocol: Protocol, ran_visible: bool = True) -> str:
        if not ran_visible:
            return (
                "Your code loaded and defines the function, but it was not run on any test: the docstring gives no "
                "examples I can run, and hidden-test grading is switched off for this run. If you are satisfied "
                f"with the function, {protocol.fill('finish_rule', self.code_noun)}."
            )
        return (
            "Your function passed every example. Hidden-test grading is switched off for this run, so the examples "
            "are the only evidence available. If you are satisfied with the function, "
            f"{protocol.fill('finish_rule', self.code_noun)}."
        )

    def already_verified(self, protocol: Protocol) -> str:
        return (
            "That is byte-for-byte the function that already passed every hidden test, so I did not run it again — "
            "resending it unchanged is how you say you stand by it, and you do. "
            + protocol.fill("finish_now", self.code_noun)
        )

    def empty_code(self, protocol: Protocol) -> str:
        return (
            'You called grade_code with an empty "code" argument, so there was nothing to run. The "code" argument '
            "must be a JSON string holding your COMPLETE Python 3 function, with every line break written as \\n. "
            "Call grade_code again with the whole function in it."
        )

    def repeat_warning(self, previous: int, protocol: Protocol) -> str:
        return (
            f"NOTE: that function is byte-for-byte identical to the {ordinal(previous)} function you had graded, "
            "and it failed again. Do not send it again — the fix has to change the logic, not just the wording of "
            "your explanation."
        )

    def revision_safety_note(self, protocol: Protocol) -> str:
        return (
            "NOTE: the function that already passed every hidden test is still kept as this run's answer, so this "
            "revision has cost you nothing. " + protocol.fill("finish_now", self.code_noun)
        )
