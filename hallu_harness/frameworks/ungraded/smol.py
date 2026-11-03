"""`smolagents_ungraded`: the smolagents adapter, with a harmonized system
prompt that names no grading tool.

The prompt is built from the coding prompt's own pieces (`smol/prompts.py`),
so the reply format, how an action is read, the planning and budget notes
stay exactly what the parser and params do. Three things differ:

* the thought is "after a tool result", not "after a grading result";
* the JSON-encoding section is about final_answer's "answer" argument, not
  grade_code's "code" argument;
* the form of a reply shows a tool call with placeholders, and the reply that
  ends the task with final_answer.

Each replaced sentence must occur exactly once in the coding text, so an
upstream rewording fails loudly instead of leaving `grade_code` in the prompt.
"""
from __future__ import annotations

from ...core.ungraded_protocol import UNGRADED_PROMPT_REVISION, UNGRADED_SMOLAGENTS
from ..smol import adapter as smol_adapter
from ..smol.adapter import SmolagentsAdapter
from ..smol.params import SmolParams
from ..smol.prompts import JSON_ENCODING_SECTION, _raw, action_blob, reply_section
from .patching import require_ungraded_benchmark, swap_globals

GRADING_THOUGHT = (
    "Thought: a few sentences on your approach or, after a grading result, on what the result tells you and what you "
    "will change."
)
TOOL_THOUGHT = (
    "Thought: a few sentences on your approach or, after a tool result, on what the result tells you and what you "
    "will do next."
)
ENCODING_SUBJECT = 'The "code" argument is a JSON string, not a code block. In it:\n'
ENCODING_RULE = (
    "grade_code runs exactly the code in this string, so the string must hold the complete code on every call: "
    "a fragment, a diff, or a single revised function fails as an incomplete program."
)


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one occurrence of {old[:60]!r} in the coding prompt; found {text.count(old)}")
    return text.replace(old, new)


def encoding_section(code_noun: str) -> str:
    text = replace_once(JSON_ENCODING_SECTION, ENCODING_SUBJECT,
                        'The "answer" argument of final_answer is a JSON string, not a code block. In it:\n')
    return replace_once(text, ENCODING_RULE,
                        f"final_answer takes the text of this string, exactly, as your final {code_noun}, so the string "
                        f"must hold the complete {code_noun}.")


def ungraded_system_prompt(texts, env, params: dict) -> str:
    """Drop-in for `smol.prompts.harmonized_system_prompt`."""
    noun = texts.code_noun
    head = texts.task_instructions(env, params["max_model_calls"]) + "\n\n# Tools\n\nYou can call these tools, and no others:"
    reply = replace_once(
        reply_section(params["tool_call_parser"], params["answer_from_memory_on_budget"], params["planning_interval"],
                      params["chat_template_tools"]),
        GRADING_THOUGHT, TOOL_THOUGHT,
    )
    call_form = action_blob("<tool name>", {"<argument name>": "<argument value>"})
    end_form = action_blob("final_answer", {"answer": f"<your complete {noun}, encoded as described above>"})
    tail = (
        f"\n\n{reply}\n\n"
        f"{encoding_section(noun)}\n\n"
        "# The form of a reply\n\n"
        "A reply that calls a tool has this form, with each part in angle brackets replaced:\n\n"
        "Thought: <your thought>\n\n"
        f"Action:\n{call_form}\n\n"
        f"The reply that ends the task with your final {noun} has this form:\n\n"
        "Thought: <your thought>\n\n"
        f"Action:\n{end_form}"
    )
    tools = "\n{%- for tool in tools.values() %}\n- {{ tool.to_tool_calling_prompt() }}\n{%- endfor %}"
    return _raw(head) + tools + _raw(tail)


class SmolagentsUngradedAdapter(SmolagentsAdapter):
    name = "smolagents_ungraded"
    protocol = UNGRADED_SMOLAGENTS
    Params = SmolParams

    @classmethod
    def framework_info(cls, params: dict) -> dict:
        info = super().framework_info(params)
        return {**info, "name": cls.name, "base_framework": info["name"], "ungraded_prompt_revision": UNGRADED_PROMPT_REVISION}

    def run(self, task, tools, env):
        require_ungraded_benchmark(self.benchmark, self.name)
        with swap_globals(smol_adapter, harmonized_system_prompt=ungraded_system_prompt):
            return super().run(task, tools, env)
