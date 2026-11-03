"""How a framework's model replies — the only text that differs between frameworks.

A *protocol* is how a framework's model acts: by tool calls (smolagents,
Qwen-Agent) or by fenced code blocks (the text framework). The task
instructions and all feedback come from the benchmark and are identical for
every framework; only the handful of phrases that say *how to reply* differ,
and those are defined here. That replaces `cf_agent`'s `retarget()`
string-swapping with parameters.

Each phrase is a format string with `{code}`, the benchmark's noun for what
the model writes ("program" for Codeforces, "function" for HumanEval).
"""
from __future__ import annotations

from dataclasses import dataclass

# Revision of the model-facing text (task instructions, reply sections, tool
# descriptions, feedback). It is part of the experiment key, so a resumed run
# never mixes trajectories recorded under different wordings. Bump it
# whenever any sentence the model is shown changes.
PROMPT_REVISION = 9


@dataclass(frozen=True)
class Protocol:
    name: str
    initial: str  # last sentence of the task message
    resubmit_after_visible: str  # "...then {resubmit_after_visible}."
    resubmit_after_hidden: str  # a sentence
    submit_option: str  # "1. Submit, by ..."
    revise_option: str  # "2. Review ... and {revise_option}"
    finish_now: str  # a sentence telling the model to finish on the verified solution
    give_up: str  # a clause: how to say "I cannot solve this"
    finish_rule: str  # a clause for the task instructions: how to finish
    final_answer_source: str  # where a final answer's code is read from; "" if a final answer holds none

    def fill(self, field: str, code: str) -> str:
        return getattr(self, field).format(code=code)


TOOL_PROTOCOL_SMOLAGENTS = Protocol(
    name="tool:smolagents",
    initial='Call grade_code with the complete {code} as its "code" argument.',
    resubmit_after_visible='call grade_code again with a corrected COMPLETE {code} as its "code" argument',
    resubmit_after_hidden='Call grade_code again with your complete corrected {code} as its "code" argument.',
    submit_option='Submit, by calling final_answer with your verified {code} as its "answer" argument',
    revise_option="call grade_code once more with your complete revised {code}. It will be run and graded again.",
    finish_now='Finish now by calling final_answer with that {code} as its "answer" argument.',
    give_up="call final_answer with the single word SORRY",
    finish_rule='call final_answer with your final {code} as its "answer" argument',
    final_answer_source="the {code} in your final_answer call",
)

TOOL_PROTOCOL_QWEN = Protocol(
    name="tool:qwen_agent",
    initial='Call grade_code with the complete {code} as its "code" argument.',
    resubmit_after_visible='call grade_code again with a corrected COMPLETE {code} as its "code" argument',
    resubmit_after_hidden='Call grade_code again with your complete corrected {code} as its "code" argument.',
    submit_option="Submit, by replying without calling any tool, with your verified {code} in a single ```python block",
    revise_option="call grade_code once more with your complete revised {code}. It will be run and graded again.",
    finish_now="Finish now by replying without calling any tool, with that {code} in a single ```python block.",
    give_up="reply with the single word SORRY and no tool call",
    finish_rule="reply without calling any tool, with your final {code} in a single ```python block",
    final_answer_source="the {code} in the reply that ends the task",
)

# Finishing and giving up are single-word replies, not code: a block holding
# print("DONE") was appended to programs (which then printed DONE and
# failed), and sent unfenced, where it was rejected as a reply without code.
TEXT_PROTOCOL = Protocol(
    name="text",
    initial="Reply with your Thought and a single ```python block containing the complete {code}.",
    resubmit_after_visible="reply with a corrected COMPLETE {code} in a single ```python block",
    resubmit_after_hidden="Reply with your complete corrected {code} in a single ```python block.",
    submit_option="Submit, by replying with the single word DONE and nothing else",
    revise_option="reply with your complete revised {code} in a single ```python block. It will be run and graded again.",
    finish_now="Finish now by replying with the single word DONE and nothing else.",
    give_up="reply with the single word SORRY and nothing else",
    finish_rule="reply with the single word DONE and nothing else",
    final_answer_source="",  # DONE carries no program
)
