"""Every sentence the model is shown about a GAIA question.

The task message is the question itself, plus a line naming the attached
file when there is one. The instructions say what the run does and no more:

* the answer-format rules are GAIA's own (from the system prompt in the
  GAIA paper, Mialon et al. 2023), which is what its scorer expects;
* the answer is compared with a reference after the task, and the result is
  not shown;
* the web is today's, while the question was written in 2023
  (`question_date_note`), which is true of every question and the reason
  some pages no longer say what the question assumes;
* what each tool can do with the attached file is said only for the tools
  the run has.

How to finish, how to give up, and where the answer is read from are the
framework's phrases (`core/qa_protocol.py`). The framework's reply
sections (`frameworks/qa/`) follow these instructions in the system prompt.

Changing any sentence here means bumping `QA_PROMPT_REVISION`.
"""
from __future__ import annotations

from ...core.types import Task

WEB_TOOLS = ("web_search", "visit_webpage", "find_in_page")


class GaiaTexts:
    code_noun = "answer"  # read by the framework adapters

    def __init__(self, tool_names: list[str], question_date_note: bool = True):
        self.tool_names = list(tool_names)
        self.question_date_note = question_date_note

    def attachment_lines(self, name: str) -> list[str]:
        lines = [f"- The question comes with a file, {name}."]
        if "read_file" in self.tool_names:
            lines.append("- read_file shows the file as text" + (", and find_in_page searches it." if "find_in_page" in self.tool_names else "."))
        if "python" in self.tool_names:
            lines.append(f"- A program run with python finds the file in its working directory as {name}.")
        return lines

    def task_instructions(self, env, max_replies: int) -> str:
        p = env.protocol
        lines = [
            "You are a general AI assistant. Your task is to answer the question in the task message. Use the tools "
            "listed below to find the information you need.",
            "",
            "Your replies are processed automatically; no person reads them while the task is running. Do not ask "
            "questions or wait for confirmation.",
            "",
            "# Your answer",
            "",
            "- After the task ends, your answer is compared with a reference answer. You are not told the result.",
            "- Your answer should be a number OR as few words as possible OR a comma separated list of numbers and/or "
            "strings.",
            "- If you are asked for a number, do not write it with commas as thousands separators, and do not use units "
            "such as $ or a percent sign unless the question asks for them.",
            "- If you are asked for a string, do not use articles or abbreviations (e.g. for cities), and write digits "
            "in plain text unless the question asks otherwise.",
            "- If you are asked for a comma separated list, apply the rules above to each element, depending on whether "
            "it is a number or a string.",
            "- Give the answer alone, without an explanation.",
        ]
        if any(name in self.tool_names for name in WEB_TOOLS):
            lines += ["", "# The web", "", "- The web tools reach the web as it is today."]
            if self.question_date_note:
                lines.append("- The question was written in 2023. Pages may have changed since then; a question about a "
                             "date or a version refers to that date or version, not to today's page.")
        attachment = getattr(env, "attachment", None)
        if attachment is not None:
            lines += ["", "# The attached file", "", *self.attachment_lines(attachment.name)]
        lines += [
            "",
            "# Ending the task",
            "",
            f"- You may send at most {max_replies} replies. If they are used up before you finish, the task ends "
            "without an answer.",
            f"- To finish the task, {p.finish_rule}.",
            f"- If you cannot find the answer, {p.give_up}.",
            f"- Your answer is {p.answer_source}.",
        ]
        return "\n".join(lines)

    def user_message(self, task: Task, protocol) -> str:
        name = task.metadata.get("file_name")
        return task.query + (f"\n\nAttached file: {name}" if name else "")
