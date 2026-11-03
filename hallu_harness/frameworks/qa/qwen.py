"""`qwen_agent_qa`: the Qwen-Agent adapter with question-answering reply
sections.

The coding adapter composes its system message as instructions, then
`REPLY_SECTION`, `JSON_ENCODING_SECTION` and `FORM_SECTION` (the first and
last formatted with the benchmark's noun), and Qwen-Agent appends its own
tool block. FnCallAgent ends the run on the first reply without a tool call,
so that reply carries the answer, after "FINAL ANSWER:" (`answers.py`).
"""
from __future__ import annotations

from ...core.qa_protocol import QA_PROMPT_REVISION, QA_QWEN
from ..qwenagent import adapter as qwen_adapter
from ..qwenagent.adapter import QwenAgentAdapter
from ..qwenagent.params import QwenAgentParams
from ..ungraded.patching import swap_globals
from .common import arguments_section, require_qa_benchmark, trace_of

REPLY_SECTION = (
    "# How to reply\n"
    "\n"
    "A reply that calls a tool consists of one thought followed by one tool call:\n"
    "\n"
    "Thought: a few sentences on your approach or, after a tool result, on what the result tells you and what you "
    "will do next.\n"
    "\n"
    "Then one call to one of the tools listed below.\n"
    "\n"
    "- Make exactly one tool call per reply, even though the tool section below permits several.\n"
    "- A reply that contains no tool call ends the task at once. Every reply before the one with your answer must "
    'therefore call a tool, and the reply without a tool call must end with your answer, as "Ending the task" says.'
)
FORM_SECTION = (
    "# The form of a reply\n"
    "\n"
    "A reply that calls a tool has this form, with each part in angle brackets replaced:\n"
    "\n"
    "Thought: <your thought>\n"
    "<tool_call>\n"
    '{{"name": "<tool name>", "arguments": {{"<argument name>": "<argument value>"}}}}\n'
    "</tool_call>\n"
    "\n"
    "The reply that ends the task has this form:\n"
    "\n"
    "Thought: <your thought>\n"
    "FINAL ANSWER: <your answer>"
)


class QwenAgentQAAdapter(QwenAgentAdapter):
    name = "qwen_agent_qa"
    protocol = QA_QWEN
    Params = QwenAgentParams

    @classmethod
    def framework_info(cls, params: dict) -> dict:
        info = super().framework_info(params)
        return {**info, "name": cls.name, "base_framework": info["name"], "qa_prompt_revision": QA_PROMPT_REVISION}

    def run(self, task, tools, env):
        require_qa_benchmark(self.benchmark, self.name)
        if not tools:
            raise RuntimeError(f"{self.name} needs tools: its prompt describes tool calls")
        sections = {"REPLY_SECTION": REPLY_SECTION, "JSON_ENCODING_SECTION": arguments_section(self.benchmark.texts.tool_names),
                    "FORM_SECTION": FORM_SECTION}
        try:
            with swap_globals(qwen_adapter, **sections):
                return super().run(task, tools, env)
        finally:
            env.trace = trace_of(self.recorder)
