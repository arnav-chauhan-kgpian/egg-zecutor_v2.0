"""`grade_code` — run a solution against the visible tests, then the hidden ones.

The dataset-agnostic form of `cf_agent/smolagent/tool.py`. What one call does,
in order:

1. **Empty code** is refused without running anything.
2. **Code byte-identical to the verified solution** is accepted without
   re-running the hidden set: resending it unchanged is the review turn
   saying "I stand by it", and re-running hundreds of processes to learn what
   is already known is pure cost.
3. **Every visible test runs**, even after one fails: "1 and 3 pass, 2 fails"
   teaches more than "the first failure was 2". If any fails, that is the
   whole answer and the hidden tests are never touched.
4. **Only then the hidden tests.** The model is told the total, how many
   failed, and the breakdown by type — never an input, an expected output,
   its own output, or which test.
5. **A pass is offered one review turn**, once per trajectory (unless the
   dataset sets `review_turn: false`).
6. **A repeated program** is named as one, because models otherwise resubmit
   an identical program with a reworded explanation and treat that as
   progress.

Once a solution passes the hidden tests it is the trajectory's answer
whatever the model does next (`env.accepted_code`): a later program that
also passes does not replace it, and a failed revision costs nothing. The
model is told so — a failed revision's feedback ends with the safety note
and the instruction to finish, not with a request for another attempt.

Every sentence comes from the benchmark's `Texts`, with the framework's reply
phrases from its `Protocol`. `private` carries what the model is not told:
per-visible-test detail and the hidden verdict, counts and test numbers.
"""
from __future__ import annotations

import time

from ..benchmarks.base import GradeCall, VisibleReport, VisibleResult
from ..core.types import ToolResult
from .base import Tool, ToolArgumentError


class GradeCodeTool(Tool):
    name = "grade_code"
    allowed_params: set[str] = {"max_output_tokens", "review_turn"}

    @property
    def description(self) -> str:
        return self.env.texts.grade_code_description(self.env.hidden_enabled)

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"code": {"type": "string", "description": self.env.texts.code_argument_description()}},
            "required": ["code"],
        }

    def call(self, code: str = "", **unexpected) -> ToolResult:
        if unexpected:
            raise ToolArgumentError(f"grade_code got unexpected argument(s): {', '.join(sorted(unexpected))}")
        env, texts, protocol = self.env, self.env.texts, self.env.protocol
        started = time.perf_counter()
        number = len(env.calls) + 1

        if not isinstance(code, str) or not code.strip():
            env.calls.append(GradeCall(number, code if isinstance(code, str) else "", ran=False))
            return self._result(texts.empty_code(protocol), number, ran=False, started=started, reason="empty code")

        code = code.strip("\n")
        # The earlier identical program, by call number (recorded) and by its
        # position among the programs graded so far (what the model is told:
        # calls that were refused or never reached the tool are not counted).
        graded = [c for c in env.calls if c.ran]
        position = next((k for k, c in enumerate(graded, 1) if c.code == code), None)
        repeated_from = graded[position - 1].number if position is not None else None

        if env.accepted_code is not None and code == env.accepted_code:
            env.calls.append(GradeCall(number, code, ran=False, repeated_from=repeated_from))
            return self._result(
                texts.already_verified(protocol), number, ran=False, started=started,
                reason="identical to the verified solution; not re-run", repeated_from=repeated_from,
            )

        call = GradeCall(number, code, ran=True, repeated_from=repeated_from)
        env.calls.append(call)
        # Once a solution is verified, a failed revision is told to finish,
        # not to try again (the safety note, added in `_annotate`).
        resubmit = env.accepted_code is None

        if env.evaluator.visible_test_count > 0:
            call.visible = env.evaluator.run_visible(code)
            if not call.visible.all_passed:
                body = texts.visible_feedback(call.visible, protocol, resubmit=resubmit)
                return self._result(self._annotate(body, position), number, call=call, started=started)
        else:
            problem = env.evaluator.load_problem(code)
            if problem is not None:
                # No visible tests to reveal a syntax error or a missing
                # definition; report it as what it is, not as hidden failures.
                call.visible = VisibleReport([VisibleResult(index=0, passed=False, input_text="", expected_text="", actual_text="", error=problem)])
                body = texts.visible_feedback(call.visible, protocol, resubmit=resubmit)
                return self._result(self._annotate(body, position), number, call=call, started=started)

        if not env.hidden_enabled:
            ran_visible = env.evaluator.visible_test_count > 0
            return self._result(texts.visible_passed_no_hidden(protocol, ran_visible), number, call=call, started=started)

        call.hidden = env.hidden(code)
        report = call.hidden
        if report.accepted:
            if env.accepted_code is None:  # the first verified program stays the answer
                env.accepted_code, env.accepted_report = code, report
            if env.review_offered or not env.review_turn:
                body = texts.verified_again(report, protocol) if env.review_offered else texts.hidden_passed(report, protocol, offer_review=False)
            else:
                env.review_offered = True
                body = texts.hidden_passed(report, protocol, offer_review=True)
            return self._result(body, number, call=call, started=started)

        if not report.conclusive:
            return self._result(texts.unverifiable(report, protocol), number, call=call, started=started)

        body = texts.hidden_failed(report, protocol, resubmit=resubmit)
        return self._result(self._annotate(body, position), number, call=call, started=started)

    # -- helpers ------------------------------------------------------------

    def _annotate(self, body: str, repeated_position: int | None) -> str:
        """A failure's feedback, plus the notes that apply: the repeat warning,
        and — when a verified solution already exists, the review-turn case —
        the safety note. The trajectory is already solved, the model cannot
        see that, and telling it is the difference between one calm
        confirmation and several panicked rewrites."""
        env = self.env
        if repeated_position is not None:
            body = f"{body}\n\n{env.texts.repeat_warning(repeated_position, env.protocol)}"
        if env.accepted_code is not None:
            body = f"{body}\n\n{env.texts.revision_safety_note(env.protocol)}"
        return body

    def _result(self, observation: str, number: int, *, started: float, call: GradeCall | None = None,
                ran: bool = True, reason: str | None = None, repeated_from: int | None = None) -> ToolResult:
        env = self.env
        private = {
            "call_number": number,
            "ran": call is not None and call.ran if call is not None else ran,
            "reason_not_run": reason,
            "repeated_from_call": call.repeated_from if call is not None else repeated_from,
            "visible": [r.private() for r in call.visible.results] if (call is not None and call.visible) else None,
            "visible_all_passed": call.visible.all_passed if (call is not None and call.visible) else None,
            "hidden": call.hidden.private() if (call is not None and call.hidden) else None,
            "accepted": bool(call is not None and call.hidden is not None and call.hidden.accepted),
            "trajectory_has_verified_solution": env.accepted_code is not None,
            "seconds": round(time.perf_counter() - started, 3),
        }
        return ToolResult(observation=observation, private=private)
