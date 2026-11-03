"""smolagents' system prompt in harmonized mode (DESIGN.md §9.3).

The benchmark's task instructions — identical in every framework — followed
by the parts only smolagents needs: its tool list (rendered by smolagents
itself, from each tool's `to_tool_calling_prompt()`), its reply format
(Thought, then an Action JSON object), how to encode code inside a JSON
string, and the form of a reply, generated with `json.dumps`. The form holds
a placeholder, not a worked program: a small model sends a worked example's
program as its own answer to an unrelated task.

What the prompt says about extraction is true of the parser in use: the
stock parser reads the text from the first "{" to the last "}" of the reply
(so a `<tool_call>`-wrapped object is read too, and two calls are one
unparseable blob); the tagged parser reads the object inside the first
`<tool_call>` pair, else does what the stock one does. With
`chat_template_tools` (stock smolagents), the chat template appends its own
tool listing, which invites several calls per reply in `<tool_call>` tags;
the prompt then says which rules win. Without it (the default) there is one
tool listing and one reply format, and the prompt does not mention tags.
What it says about the budget follows the params: the from-memory call after
the step budget, and planning steps, which smolagents does not count toward
`max_steps`.

Literal text is wrapped in Jinja `{% raw %}` blocks: smolagents renders the
template with Jinja, and code or JSON could otherwise be read as template
syntax.
"""
from __future__ import annotations

import importlib.resources
import json

import yaml

def json_encoding_section(final_answer_noun: str | None = None) -> str:
    """How to write code inside a JSON string. With smolagents, a program
    given to final_answer is a JSON string too (`final_answer_noun`)."""
    if final_answer_noun is None:
        subject = 'The "code" argument is a JSON string, not a code block. In it:\n'
    else:
        subject = (
            f'The "code" argument, and a {final_answer_noun} given as the "answer" argument of final_answer, are JSON '
            "strings, not code blocks. In them:\n"
        )
    return JSON_ENCODING_SECTION.replace('The "code" argument is a JSON string, not a code block. In it:\n', subject)


JSON_ENCODING_SECTION = (
    "# Encoding code in a JSON string\n"
    "\n"
    'The "code" argument is a JSON string, not a code block. In it:\n'
    "- write each line break as \\n, so that the code occupies a single line of JSON;\n"
    '- write each double quote as \\" and each backslash as \\\\; for example, the Python string literal "\\n" '
    'is written \\"\\\\n\\";\n'
    "- indent with four spaces, not tabs;\n"
    "- do not enclose the code in ``` fences.\n"
    "\n"
    "grade_code runs exactly the code in this string, so the string must hold the complete code on every call: "
    "a fragment, a diff, or a single revised function fails as an incomplete program."
)


def _raw(text: str) -> str:
    assert "{% endraw %}" not in text
    return "{% raw %}" + text + "{% endraw %}"


def action_blob(name: str, arguments: dict) -> str:
    return json.dumps({"name": name, "arguments": arguments}, indent=2)


def reply_section(tool_call_parser: str, answer_from_memory: bool, planning_interval: int | None,
                  chat_template_tools: bool = False) -> str:
    """How to reply: the format, how the action is read, and the budget
    details that are particular to smolagents."""
    if tool_call_parser == "tagged":
        extraction = (
            "If your reply contains <tool_call></tool_call> tags, the JSON object inside the first pair of tags is "
            'read as your action. Otherwise your action is read as the text from the first "{" to the last "}" in '
            "your reply, which must be one valid JSON object; then write no brace anywhere else in the reply, the "
            "thought included (braces inside a JSON string value are part of that value and are allowed). Make "
            "exactly one tool call per reply: if several are tagged, only the first is run."
        )
    else:
        extraction = (
            'Your action is read as the text from the first "{" to the last "}" in your reply, which must be one '
            "valid JSON object. Write no brace anywhere else in the reply, the thought included; braces inside a "
            "JSON string value are part of that value and are allowed. "
            + ("The object may be wrapped in <tool_call></tool_call> tags. " if chat_template_tools else "")
            + "Make exactly one tool call per reply: a reply with more than one cannot be read."
        )
    lines = [
        "# How to reply",
        "",
        "Every reply consists of one thought followed by one action:",
        "",
        "Thought: a few sentences on your approach or, after a grading result, on what the result tells you and "
        "what you will change.",
        "",
        'Action: a single JSON object with exactly two keys: "name", the name of the tool to call, and '
        '"arguments", an object holding that tool\'s arguments. Write nothing after the object.',
        "",
        extraction,
        "",
        "A reply whose action cannot be read runs nothing; you are shown the error, and the reply still counts "
        "toward your limit."
        + (" If the tools are also listed elsewhere in this message in another format, the rules given here apply "
           "regardless." if chat_template_tools else ""),
    ]
    if planning_interval is not None:
        every = "step" if planning_interval == 1 else f"{planning_interval} steps"
        lines += [
            "",
            f"At the first step and then every {every}, you are also asked to write a plan; a planning reply runs "
            "nothing and does not count toward your limit.",
        ]
    if answer_from_memory:
        lines += [
            "",
            "If your replies are used up before you finish, you are then asked once, in a separate request, for "
            "your final answer.",
        ]
    return "\n".join(lines)


def harmonized_system_prompt(texts, env, params: dict) -> str:
    """A Jinja template for smolagents' `prompt_templates['system_prompt']`."""
    noun = texts.code_noun
    form = action_blob("grade_code", {"code": f"<your complete {noun}, encoded as described above>"})
    head = texts.task_instructions(env, params["max_model_calls"]) + "\n\n# Tools\n\nYou can call these tools, and no others:"
    reply = reply_section(params["tool_call_parser"], params["answer_from_memory_on_budget"], params["planning_interval"],
                          params["chat_template_tools"])
    tail = (
        f"\n\n{reply}\n\n"
        f"{json_encoding_section(noun)}\n\n"
        "# The form of a reply\n\n"
        f"A reply that has a {noun} graded has this form, with each part in angle brackets replaced:\n\n"
        "Thought: <your thought>\n\n"
        f"Action:\n{form}"
    )
    tools = (
        "\n{%- for tool in tools.values() %}\n- {{ tool.to_tool_calling_prompt() }}\n{%- endfor %}"
    )
    return _raw(head) + tools + _raw(tail)


def final_answer_tool(code_noun: str):
    """smolagents' FinalAnswerTool, described as this task uses it. Stock, it
    "Provides a final answer to the given problem", whose argument is "The
    final answer to the problem": for a programming problem that reads as the
    expected output, and a 0.5B model answered a Codeforces problem with its
    first sample's output. The argument keeps its type (`any`), so smolagents
    validates a call exactly as before."""
    from smolagents.default_tools import FinalAnswerTool

    class TaskFinalAnswerTool(FinalAnswerTool):
        description = f"Ends the task. Call it with your final {code_noun}, or with the single word SORRY to give up."
        inputs = {"answer": {"type": "any", "description": f"Your complete final {code_noun}, or the single word SORRY."}}

    return TaskFinalAnswerTool()


def prompt_templates(system_prompt: str | None) -> dict:
    """smolagents' stock template dict, with only the system prompt replaced
    (or nothing replaced, in stock mode). The planning and final-answer
    templates stay stock."""
    templates = yaml.safe_load(
        importlib.resources.files("smolagents.prompts").joinpath("toolcalling_agent.yaml").read_text()
    )
    if system_prompt is not None:
        templates["system_prompt"] = system_prompt
    return templates
