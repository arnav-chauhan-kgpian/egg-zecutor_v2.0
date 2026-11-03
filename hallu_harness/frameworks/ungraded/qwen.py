"""`qwen_agent_ungraded`: the Qwen-Agent adapter, with harmonized reply
sections that name no grading tool.

The coding adapter composes its system message as instructions, then
`REPLY_SECTION`, `JSON_ENCODING_SECTION` and `FORM_SECTION` (the first and
last formatted with the code noun), and Qwen-Agent appends its own tool
block. Here those three are replaced for the run, rendered for whether the
run has tools at all: with none (a single-turn baseline), Qwen-Agent adds no
tool block, and the first reply is the answer.
"""
from __future__ import annotations

from ...core.ungraded_protocol import UNGRADED_PROMPT_REVISION, UNGRADED_QWEN
from ..qwenagent import adapter as qwen_adapter
from ..qwenagent.adapter import QwenAgentAdapter
from ..qwenagent.params import QwenAgentParams
from .patching import require_ungraded_benchmark, swap_globals

REPLY_WITH_TOOLS = (
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
    "- A reply that contains no tool call ends the task at once, and its text is your final answer. Every reply "
    "before your final answer must therefore call a tool."
)
ARGUMENTS_SECTION = (
    "# Tool arguments\n"
    "\n"
    'A tool call\'s arguments are a JSON object. Inside a JSON string, write each double quote as \\" and each '
    "backslash as \\\\."
)
FORM_WITH_TOOLS = (
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
    "```python\n"
    "<your complete {code}>\n"
    "```"
)
REPLY_WITHOUT_TOOLS = (
    "# How to reply\n"
    "\n"
    "You have no tools. Your first reply ends the task, and its text is your final answer."
)
FORM_WITHOUT_TOOLS = (
    "# The form of a reply\n"
    "\n"
    "Your reply has this form, with each part in angle brackets replaced:\n"
    "\n"
    "Thought: <your thought>\n"
    "```python\n"
    "<your complete {code}>\n"
    "```"
)


def sections(has_tools: bool, code_noun: str) -> dict[str, str]:
    """The adapter's three module globals, as this run should render them."""
    if has_tools:
        return {"REPLY_SECTION": REPLY_WITH_TOOLS, "JSON_ENCODING_SECTION": ARGUMENTS_SECTION, "FORM_SECTION": FORM_WITH_TOOLS}
    program = (f"# Your {code_noun}\n\nGive the complete {code_noun} in a single ```python block; text outside the "
               f"block is not part of the {code_noun}.")
    return {"REPLY_SECTION": REPLY_WITHOUT_TOOLS, "JSON_ENCODING_SECTION": program, "FORM_SECTION": FORM_WITHOUT_TOOLS}


class QwenAgentUngradedAdapter(QwenAgentAdapter):
    name = "qwen_agent_ungraded"
    protocol = UNGRADED_QWEN
    Params = QwenAgentParams

    @classmethod
    def framework_info(cls, params: dict) -> dict:
        info = super().framework_info(params)
        return {**info, "name": cls.name, "base_framework": info["name"], "ungraded_prompt_revision": UNGRADED_PROMPT_REVISION}

    def run(self, task, tools, env):
        require_ungraded_benchmark(self.benchmark, self.name)
        with swap_globals(qwen_adapter, **sections(bool(tools), self.benchmark.texts.code_noun)):
            return super().run(task, tools, env)
