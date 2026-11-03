"""`smolagents_qa`: the smolagents adapter with a question-answering system
prompt, final-answer tool and cut-off message.

The prompt reuses the coding prompt's own pieces (`smol/prompts.py`), so the
reply format, how an action is read, and the planning and budget notes stay
exactly what the parser and params do. What differs:

* the thought is "after a tool result", not "after a grading result";
* the argument section is about JSON strings in general, and python's
  "code" argument when the run has that tool;
* the form of a reply shows a tool call with placeholders, and the reply
  that ends the task with final_answer;
* final_answer is described as taking the answer (the coding one asks for
  "your complete final program");
* a reply cut off by the length cap is told to keep its thought short and
  its code compact, not "no comments, no docstrings".
"""
from __future__ import annotations

from ...core.qa_protocol import QA_PROMPT_REVISION, QA_SMOLAGENTS
from ..smol import adapter as smol_adapter
from ..smol.adapter import SmolagentsAdapter
from ..smol.params import SmolParams
from ..smol.prompts import _raw, action_blob, reply_section
from ..ungraded.patching import swap_globals
from ..ungraded.smol import GRADING_THOUGHT, TOOL_THOUGHT, replace_once
from .common import arguments_section, require_qa_benchmark, trace_of

QA_TRUNCATED_REPLY_ERROR = (
    "your reply was cut off: it ran into the limit on how much you may write in one turn, so it ended before its "
    "action was complete, and nothing was run. Keep the thought to a few sentences and any code compact, so that the "
    "whole action fits in one reply."
)


def qa_system_prompt(texts, env, params: dict) -> str:
    """Drop-in for `smol.prompts.harmonized_system_prompt`."""
    head = texts.task_instructions(env, params["max_model_calls"]) + "\n\n# Tools\n\nYou can call these tools, and no others:"
    reply = replace_once(
        reply_section(params["tool_call_parser"], params["answer_from_memory_on_budget"], params["planning_interval"],
                      params["chat_template_tools"]),
        GRADING_THOUGHT, TOOL_THOUGHT,
    )
    call_form = action_blob("<tool name>", {"<argument name>": "<argument value>"})
    end_form = action_blob("final_answer", {"answer": "<your answer>"})
    tail = (
        f"\n\n{reply}\n\n"
        f"{arguments_section(texts.tool_names)}\n\n"
        "# The form of a reply\n\n"
        "A reply that calls a tool has this form, with each part in angle brackets replaced:\n\n"
        "Thought: <your thought>\n\n"
        f"Action:\n{call_form}\n\n"
        "The reply that ends the task has this form:\n\n"
        "Thought: <your thought>\n\n"
        f"Action:\n{end_form}"
    )
    tools = "\n{%- for tool in tools.values() %}\n- {{ tool.to_tool_calling_prompt() }}\n{%- endfor %}"
    return _raw(head) + tools + _raw(tail)


def qa_final_answer_tool(code_noun: str):
    """Drop-in for `smol.prompts.final_answer_tool`. The argument keeps type
    `any`, as the coding tool does: a number sent as a JSON number is still an
    answer, not a type error the model must retry."""
    from smolagents.default_tools import FinalAnswerTool

    class AnswerFinalAnswerTool(FinalAnswerTool):
        description = ("Ends the task with your answer. Call it with your answer, written as described under \"Your "
                       "answer\", or with the single word SORRY if you cannot find the answer.")
        inputs = {"answer": {"type": "any", "description": "Your answer, or the single word SORRY."}}

    return AnswerFinalAnswerTool()


class SmolagentsQAAdapter(SmolagentsAdapter):
    name = "smolagents_qa"
    protocol = QA_SMOLAGENTS
    Params = SmolParams

    @classmethod
    def framework_info(cls, params: dict) -> dict:
        info = super().framework_info(params)
        return {**info, "name": cls.name, "base_framework": info["name"], "qa_prompt_revision": QA_PROMPT_REVISION}

    def run(self, task, tools, env):
        require_qa_benchmark(self.benchmark, self.name)
        try:
            with swap_globals(smol_adapter, harmonized_system_prompt=qa_system_prompt, final_answer_tool=qa_final_answer_tool,
                              TRUNCATED_REPLY_ERROR=QA_TRUNCATED_REPLY_ERROR):
                return super().run(task, tools, env)
        finally:
            env.trace = trace_of(self.recorder)
