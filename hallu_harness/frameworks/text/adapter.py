"""The text protocol: no tools, one ```python block per reply (DESIGN.md §9.4).

`cf_agent/agent/loop.py` as a framework adapter. The model's only action is
writing a fenced block; the adapter takes the last one outside any think
block (cf_agent's parsing rules), records it as a call to the dataset's code
tool, runs it through that same tool — so grading, feedback and the
trajectory schema are exactly those of the tool-calling frameworks — and
sends the observation back as the next user message.

* no block -> cf_agent's format feedback; a reply cut off while thinking or
  mid-block -> cf_agent's truncation feedback (parse status `failed`);
* several blocks -> the last one runs, and cf_agent's "only the last block
  ran" note is put in front of the result;
* those three messages (`feedback.py`) use the dataset's noun for the code,
  and offer finishing once a grading result allows it;
* a reply that is the single word DONE ends the run on the last program
  sent; SORRY gives up; otherwise the budget or the context rule ends it.
  (cf_agent used a block holding `print("DONE")`. Models appended it to
  their programs, which then printed DONE and failed, and sent it unfenced,
  where it was rejected as a reply without code.)

One deliberate difference from cf_agent's loop: the run no longer ends by
itself when a revision passes or a verified program is resent. As in the
tool-calling frameworks, the model is told to finish and must do so — the
same ending rule for every framework.
"""
from __future__ import annotations

import re
import time

from cf_agent.agent.parsing import _CODE_FENCE_RE, extract_python_blocks, strip_thinking, truncation_reason

from ...core.errors import ContextExhausted, HarnessAbort
from ...core.protocol import TEXT_PROTOCOL
from ...core.types import FrameworkResult
from ..base import FrameworkAdapter
from . import feedback as form


def protocol_section(code_noun: str) -> str:
    """How to reply — cf_agent's text SYSTEM_PROMPT, the protocol half, with
    what the adapter does about each kind of reply."""
    return (
        "# How to reply\n"
        "\n"
        f"A reply that sends a {code_noun} must have exactly this form:\n"
        "\n"
        "Thought: a few sentences on your approach or, after a grading result, on what the result tells you and "
        "what you will change.\n"
        "\n"
        "```python\n"
        f"<your complete Python 3 {code_noun}>\n"
        "```\n"
        "\n"
        f"- The reply must contain exactly one ```python block, holding the complete {code_noun}. The {code_noun} "
        "in it is graded, and the grading result is sent to you as the next message.\n"
        "- If a reply contains more than one ```python block, only the last one is run; every earlier block is "
        "ignored.\n"
        "- A reply consisting of the single word DONE finishes the task, and one consisting of the single word "
        'SORRY gives up, as described under "Ending the task".\n'
        "- Any other reply without a ```python block runs nothing, but it still counts toward your limit."
    )


def reply_word(output: str) -> str | None:
    """DONE or SORRY if the reply, outside any think block, is that single
    word (any letter case, ignoring surrounding quotes, backticks, asterisks
    and a final period); else None."""
    visible = strip_thinking(output).strip().strip("'\"`*.").strip().upper()
    return visible if visible in ("DONE", "SORRY") else None


class TextAdapter(FrameworkAdapter):
    name = "text"
    protocol = TEXT_PROTOCOL

    def __init__(self, params, backend, recorder, benchmark, prompt_mode):
        super().__init__(params, backend, recorder, benchmark, prompt_mode)
        self.content_decode = "skip_special_tokens" if params["content_special_tokens"] == "skip" else "keep_special_tokens"

    @classmethod
    def framework_info(cls, params: dict) -> dict:
        import hallu_harness

        return {
            "name": "text",
            "version": f"hallu_harness {hallu_harness.__version__} (cf_agent text protocol)",
            "agent_class": "TextAdapter",
            "params": params,
            "prompt_mode": params["prompt_mode"],
            "code_tool": params["code_tool"],
            "content_decode": "skip_special_tokens" if params["content_special_tokens"] == "skip" else "keep_special_tokens",
            "budget": {"max_model_calls": params["max_model_calls"], "native": {"max_iterations": params["max_model_calls"]}},
        }

    def run(self, task, tools, env) -> FrameworkResult:
        texts = self.benchmark.texts
        tool = next((t for t in tools if t.name == self.params["code_tool"]), None)
        system = texts.task_instructions(env, self.max_model_calls) + "\n\n" + protocol_section(texts.code_noun)
        user = texts.user_message(task, self.protocol)
        if tool is None:
            return FrameworkResult(None, "error", system, user, self.framework_info(self.params)["version"],
                                   error=f"code tool {self.params['code_tool']!r} is not among this run's tools")

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        final_answer, stop, error = None, None, None

        for _ in range(self.max_model_calls):
            try:
                generation = self.generate(messages)
            except ContextExhausted:
                stop = "context_exhausted"
                break
            except Exception as exc:  # noqa: BLE001 - recorded; ends the trajectory
                stop, error = "error", f"{type(exc).__name__}: {exc}"
                break

            output = generation.content.strip()
            messages.append({"role": "assistant", "content": output})
            word = reply_word(output)
            if word is not None:  # DONE finishes, SORRY gives up
                self.recorder.on_parse("ok", None, False)
                final_answer, stop = word, "final_answer"
                self.recorder.end_step()
                break
            blocks = extract_python_blocks(output)

            if not blocks:
                # Any reply the length cap ended without a block was cut off
                # before its code: inside <think>, inside an open block, or in
                # plain prose. The model is told so, never just "no block".
                cut = truncation_reason(output) or ("thinking" if generation.hit_length_cap else None)
                visible = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL | re.IGNORECASE)
                attempted = "```" in visible or cut is not None
                if cut:
                    feedback = form.truncated(env, cut)
                elif _CODE_FENCE_RE.search(strip_thinking(output)):  # a block, but only whitespace in it
                    feedback = form.empty_code_block(env)
                else:
                    feedback = form.no_code_block(env)
                self.recorder.on_parse(
                    "failed" if attempted else "no_tool_call",
                    f"reply cut off mid-{cut}" if cut else "no ```python block in the reply",
                    attempted,
                )
                self.recorder.on_framework_feedback(feedback)
                self.recorder.end_step()
                messages.append({"role": "user", "content": feedback})
                continue

            code = blocks[-1]
            self.recorder.on_parse("ok", None, True)
            record = self.recorder.on_tool_call(tool.name, {"code": code}, raw_arguments=code)
            started = time.perf_counter()
            try:
                result = tool.run(code=code)
            except (Exception, HarnessAbort) as exc:
                # Not the model's doing (its only argument is the code): the
                # trajectory ends here, and the model is told nothing.
                self.recorder.on_tool_result(record, observation=None, status="raised",
                                             error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - started)
                raise
            self.recorder.on_tool_result(record, observation=result.observation, status="executed", private=result.private,
                                         seconds=time.perf_counter() - started)
            note = form.multiple_blocks_note(env, len(blocks))
            feedback = f"{note}\n\n{result.observation}" if note else result.observation
            if note:
                self.recorder.on_framework_feedback(feedback)
            self.recorder.end_step()
            messages.append({"role": "user", "content": feedback})
        else:
            stop = "budget"

        self.recorder.finish()
        return FrameworkResult(
            final_answer=final_answer,
            stop_reason=stop,
            system_prompt=system,
            user_query=user,
            framework_version=self.framework_info(self.params)["version"],
            error=error,
        )
