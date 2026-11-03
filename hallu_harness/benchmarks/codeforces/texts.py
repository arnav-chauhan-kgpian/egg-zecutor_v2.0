"""Every sentence the model is shown about a Codeforces problem.

The feedback wording is `cf_agent`'s (`agent/prompts.py` and
`smolagent/prompts.py`), with two deliberate changes (DESIGN.md §7.5):

* nothing is capped at 25 lines any more. Input and expected output are
  shown in full; the program's output is shown up to as many tokens as the
  expected output has, with excess whitespace removed. A program that
  printed more tokens gets a note saying so (`extra_output_note`): the shown
  part can equal the expected output, and "(output truncated)" alone left
  the model to find a difference it could not see. "(output truncated)"
  marks the other cut, of a runaway token;
* the sentences that say how to reply come from the framework's `Protocol`,
  so the tool-calling frameworks and the text framework share every other
  word. With the text protocol they reproduce `cf_agent`'s text-loop
  wording; with a tool protocol, `cf_agent`'s retargeted smolagents wording
  (checked in tests/test_codeforces_texts.py).

Further changes keep every sentence true of the run (checked in
tests/test_codeforces_texts.py, "deliberate differences"):

* the hidden tests are the problem's full test set, which usually includes
  the samples, so they are neither "additional" nor "unseen"; a pass is
  reported as passing "the N hidden tests";
* a problem usually has several samples, so the feedback speaks of "the
  sample tests" and "those inputs", not "that one input";
* the repeat warning names the earlier program by its position among the
  programs graded ("the 2nd program you had graded"), which every framework
  counts the same way, and says only that it failed again;
* once a verified solution exists, failure feedback leaves out the "send a
  corrected program" sentence (`resubmit=False`), because the model is told
  to finish instead;
* a problem without samples (only with `eligible_only: false`) is not
  described as having passed them.

The task instructions (`task_instructions`) are new: the harmonized system
instructions every framework receives (DESIGN.md §9.3). They are rendered
from the trajectory's settings (hidden tests, review turn, solution policy,
reply budget, time-limit multiplier), so each statement in them is true of
the run they are shown in.
"""
from __future__ import annotations

from cf_agent.dataset.problem import DEFAULT_TIME_LIMIT_MULTIPLIER, Problem
from cf_agent.sandbox.process import WALL_TIME_FACTOR, WALL_TIME_SLACK_S

from ..base import PYTHON_VERSION, HiddenReport, TaskEnv, Texts, VisibleReport, ordinal
from ...core.protocol import Protocol

TRUNCATION_NOTE = "(output truncated)"


def extra_output_note(cut: bool) -> str:
    """Shown instead of TRUNCATION_NOTE when the program printed more tokens
    than the expected output has. The extra tokens are not shown, so the
    output shown can equal the expected one; without this note the model is
    told to work out a difference it cannot see."""
    return "(your program printed more tokens than the expected output has, which fails the test" + (
        "; the rest is not shown)" if cut else ")")

VERDICT_HINTS = {
    "WRONG_ANSWER": "a logic error: the algorithm itself computes the wrong thing on some inputs.",
    "RUNTIME_ERROR": "your program crashes on some input — check assumptions about input shape (e.g. empty lines, list bounds).",
    "TIME_LIMIT_EXCEEDED": "your algorithm is too slow for the largest inputs the constraints allow — reconsider its complexity.",
    "MEMORY_LIMIT_EXCEEDED": "your program uses too much memory on some input — reconsider what you are storing.",
}
DEFAULT_HINT = "a mismatch between your program's behavior and the problem statement."


def indent(text: str, prefix: str = "    ") -> str:
    """Quotes a block of text, in full; `(empty)` for nothing."""
    text = text.strip("\n")
    if not text.strip():
        return prefix + "(empty)"
    return "\n".join(prefix + line for line in text.splitlines())


class CodeforcesTexts(Texts):
    code_noun = "program"
    visible_tests = "sample test"

    def __init__(self, time_limit_multiplier: float = DEFAULT_TIME_LIMIT_MULTIPLIER):
        self.time_limit_multiplier = time_limit_multiplier

    # ------------------------------------------------------------ the task

    def task_instructions(self, env: TaskEnv, max_replies: int) -> str:
        lines = [
            "You are an expert competitive programmer. Your task is to solve one Codeforces problem by writing a "
            f"complete {PYTHON_VERSION} program that reads its input from standard input and writes its answer to "
            "standard output.",
            "",
            "Your replies are processed automatically; no person reads them while the task is running. Do not ask "
            "questions or wait for confirmation.",
            "",
            "# How a program is graded",
            "",
            "1. The program is run on each sample test printed in the problem statement. Every sample is run, even "
            "after one fails.",
            "2. If any sample fails, grading stops there. For each failed sample you are shown its input, the "
            "expected output, the output your program printed, and any error message.",
        ]
        if env.hidden_enabled:
            lines += [
                "3. If every sample passes, the program is run on the problem's hidden tests: its full test set, "
                "which usually also contains the samples. If any hidden test fails, you are told how many hidden "
                "tests there are, how many failed, and how many failed with each verdict: WRONG_ANSWER, "
                "RUNTIME_ERROR, TIME_LIMIT_EXCEEDED or MEMORY_LIMIT_EXCEEDED. You are never shown a hidden test's "
                "input, its expected output, your program's output on it, or which hidden tests failed.",
                "4. If every hidden test passes, you are told so.",
            ]
        else:
            lines.append("3. Hidden tests are not run while the task is running. If every sample passes, you are told so.")
        lines += [
            "",
            "An output is correct if it consists of the same tokens as the expected output, in the same order. A "
            "token is a maximal run of non-whitespace characters, so the amount and kind of whitespace, line breaks "
            "included, does not matter; the words YES and NO match regardless of letter case. A program that prints "
            "more tokens than the expected output contains is stopped at that point and fails the test. In the "
            "report of a failed sample, both outputs are shown with runs of whitespace within a line collapsed and "
            "blank lines omitted. If your program printed more tokens than the expected output has, a note says so, "
            "and the extra tokens may not be shown. Otherwise a long output from your program may be cut short; the "
            'note "(output truncated)" marks such a cut.',
            "",
            "# Limits",
            "",
            "- Each test runs under the time limit and the memory limit given in the problem's header and again "
            "after the statement. Where Codeforces states a time limit, the one given is "
            f"{self.time_limit_multiplier:g} times it, to allow for the speed of Python.",
            "- The time limit is on CPU time. A program that uses more is stopped and fails the test with "
            "TIME_LIMIT_EXCEEDED. A test that is still running after "
            f"{WALL_TIME_FACTOR:g} times its time limit plus {WALL_TIME_SLACK_S:g} "
            f"second{'' if WALL_TIME_SLACK_S == 1 else 's'} of real time (for "
            "example, because the program waits) is stopped and fails the same way.",
            "- The tests can be far larger than the samples. Your algorithm must meet the time limit on the largest "
            "inputs the constraints allow.",
            "",
            "# Rules",
            "",
            "- Always send the complete program: never a fragment, a diff, or a single changed function.",
            "- Print nothing except the required output: no prompts and no debugging output.",
            "- The program must not use the network or open files.",
            "",
            self.ending_section(env, max_replies),
        ]
        return "\n".join(lines)

    def user_message(self, task, protocol: Protocol) -> str:
        problem: Problem = task.payload["problem"]
        limit = problem.enforced_time_limit(self.time_limit_multiplier)  # what the grader enforces
        memory = f" and memory limit ({problem.memory_limit:g}MB)" if problem.memory_limit else ""
        return (
            f"{problem.render(time_limit=limit)}\n\n---\n\nSolve the problem above.\n\n"
            f"Your program must run comfortably within the time limit "
            f"({limit:g}s){memory} given the constraints above, including when the "
            f"input holds many test cases.\n\n{protocol.fill('initial', self.code_noun)}"
        )

    def grade_code_description(self, hidden_tests: bool) -> str:
        runs = (
            "The program is run on the sample tests from the problem statement and, only if every sample passes, "
            "on the problem's hidden tests."
            if hidden_tests
            else "The program is run on the sample tests from the problem statement."
        )
        return (
            f"Grades a complete Python 3 program and returns the result as text. {runs} The program must read its "
            "input from standard input and write its answer to standard output."
        )

    def code_argument_description(self) -> str:
        # No backslash: tool listings show this through repr() or JSON,
        # which would double it. The encoding is in the system prompt.
        return "The complete Python 3 program."

    # ------------------------------------------------------------ feedback

    def visible_feedback(self, report: VisibleReport, protocol: Protocol, resubmit: bool = True) -> str:
        results = report.results
        failed = [r for r in results if not r.passed]
        lines = ["I ran your program against the sample test(s).", ""]
        for result in results:
            if result.passed:
                lines.append(f"Sample {result.index}: PASS")
                continue
            lines.append(f"Sample {result.index}: FAIL")
            lines.append("  input:")
            lines.append(indent(result.input_text))
            lines.append("  expected output:")
            lines.append(indent(result.expected_text))
            lines.append("  your program printed:")
            printed = indent(result.actual_text)
            if result.extra_output:
                printed += "\n    " + extra_output_note(result.truncated)
            elif result.truncated:
                printed += "\n    " + TRUNCATION_NOTE
            lines.append(printed)
            if result.error:
                lines.append("  error:")
                lines.append(indent(result.error))
            lines.append("")
        if not failed:
            lines.append(f"All {len(results)} sample tests passed.")
        elif not resubmit:
            lines.append(f"{len(failed)} of {len(results)} sample tests failed.")
        else:
            lines.append(
                f"{len(failed)} of {len(results)} sample tests failed. Work out why from the "
                f"differences above, then {protocol.fill('resubmit_after_visible', self.code_noun)}."
            )
        return "\n".join(lines)

    def hidden_failed(self, report: HiddenReport, protocol: Protocol, resubmit: bool = True) -> str:
        breakdown_lines = "\n".join(f"  - {verdict}: {count}" for verdict, count in report.breakdown)
        hint_lines = "\n".join(
            f"  - {verdict}: {VERDICT_HINTS.get(verdict, DEFAULT_HINT)}" for verdict, _ in report.breakdown
        )
        closing = f"\n\n{protocol.fill('resubmit_after_hidden', self.code_noun)}" if resubmit else ""
        if report.no_visible_tests:
            lead = "This problem has no sample tests, so I ran your program on"
            samples = ("", "", "")
        else:
            lead = "Your program passed every sample test, so I ran it on"
            samples = (
                " Passing the sample tests does not mean your algorithm is correct; it only means it is correct on "
                "those inputs.",
                " (not just the samples)",
                " that only happens to work for the samples",
            )
        return (
            f"{lead} the problem's hidden tests: {report.total} tests, which usually include the samples. It "
            f"failed {report.failed} of them, broken down by how they failed:\n{breakdown_lines}\n\n"
            "I will not show you the failing input, expected output, or your program's actual "
            f"output — reasoning your way to the bug without that crutch is the point.{samples[0]} Think "
            "critically about your approach from first principles: re-derive it, check whether it actually "
            f"satisfies every constraint in the statement{samples[1]}, look for edge cases (empty input, all-equal "
            f"elements, minimum/maximum bounds), and question any step{samples[2] or ' you have not justified'}."
            f"\n\nWhat each failure type above suggests:\n{hint_lines}"
            f"{closing}"
        )

    def hidden_passed(self, report: HiddenReport, protocol: Protocol, offer_review: bool) -> str:
        head = (
            f"Congrats — your code passes all hidden test cases. It passed every one of the "
            f"{report.total} hidden tests.\n\n"
        )
        if not offer_review:
            return head + protocol.fill("finish_now", self.code_noun)
        return head + (
            "Nothing further is required of you. You may now do either of two things:\n\n"
            f"  1. {protocol.fill('submit_option', self.code_noun)}\n"
            "  2. Review your program once more first — re-read it for anything you would "
            "still change (a fragile assumption, an unhandled edge case, an unnecessarily "
            f"slow step) and {protocol.fill('revise_option', self.code_noun)}\n\n"
            "This offer is made once. If you revise, the version already verified is kept "
            "regardless, so a revision cannot cost you a correct solution — but do not "
            "change working code merely because you were asked whether you wanted to."
        )

    def verified_again(self, report: HiddenReport, protocol: Protocol) -> str:
        return (
            f"Your revised program also passes all {report.total} hidden tests. You "
            "have now used your one review turn, so there is nothing further to check. "
            + protocol.fill("finish_now", self.code_noun)
        )

    def unverifiable(self, report: HiddenReport, protocol: Protocol) -> str:
        lead = "This problem has no sample tests." if report.no_visible_tests else "Your program passed every sample test."
        return (
            f"{lead} I then tried to grade it against this problem's "
            f"hidden tests and could not: {report.note}. That says nothing about whether your program "
            "is correct — it is a gap in the test data, not a verdict on your solution, and there is "
            "nothing here for you to fix.\n\n"
            "Re-read your program once for anything you would still change, then finish. "
            + protocol.fill("finish_now", self.code_noun)
        )

    def visible_passed_no_hidden(self, protocol: Protocol, ran_visible: bool = True) -> str:
        if not ran_visible:
            return (
                "Your program was not run on any test: this problem has no sample tests, and hidden-test grading "
                "is switched off for this run. If you are satisfied with the program, "
                f"{protocol.fill('finish_rule', self.code_noun)}."
            )
        return (
            "Your program passed every sample test. Hidden-test grading is switched off for this run, "
            "so the sample result is the only evidence available. If you are satisfied with the program, "
            f"{protocol.fill('finish_rule', self.code_noun)}."
        )

    def already_verified(self, protocol: Protocol) -> str:
        return (
            "That is byte-for-byte the program that already passed every sample and every hidden test, "
            "so I did not run it again — resending it unchanged is how you say you stand by it, and you "
            "do. " + protocol.fill("finish_now", self.code_noun)
        )

    def empty_code(self, protocol: Protocol) -> str:
        return (
            'You called grade_code with an empty "code" argument, so there was no program to run. '
            'The "code" argument must be a JSON string holding your COMPLETE Python 3 program, with '
            "every line break written as \\n. Call grade_code again with the whole program in it."
        )

    def repeat_warning(self, previous: int, protocol: Protocol) -> str:
        return (
            f"NOTE: that program is byte-for-byte identical to the {ordinal(previous)} program you had graded, "
            "and it failed again. Do not send it again — the fix has to change the logic, not just the wording "
            "of your explanation."
        )

    def revision_safety_note(self, protocol: Protocol) -> str:
        return (
            "NOTE: the program that already passed every hidden test is still kept as this run's answer, "
            "so this revision has cost you nothing. " + protocol.fill("finish_now", self.code_noun)
        )
