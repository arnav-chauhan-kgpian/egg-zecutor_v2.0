"""Reply phrases for question-answering tasks (GAIA): the model finds an
answer with its tools and ends the task with that answer, which is compared
with a reference answer after the task.

The coding protocols (`core/protocol.py`) are about programs and grading;
none of their phrases fits here, so this is its own small set: how to
finish, how to give up, and where the answer is read from, per framework.

`QA_PROMPT_REVISION` is the revision of every model-facing text on this
path: the GAIA instructions, the QA framework variants' reply sections and
final-answer tool, and the descriptions of `visit_webpage`, `find_in_page`,
`read_file` and `python`. The benchmark puts it in the experiment key
(through `Benchmark.grading_inputs`); bump it whenever any of those texts
change.
"""
from __future__ import annotations

from dataclasses import dataclass

QA_PROMPT_REVISION = 4  # 2: web_search names its source, find_in_page keeps line breaks; 3: web_search's Serper fallback;
                        # 4: visit_webpage's aria rules, python's Biopython and no-install sentence


@dataclass(frozen=True)
class QAProtocol:
    name: str
    finish_rule: str  # a clause: "To finish the task, <finish_rule>."
    give_up: str  # a clause: "If you cannot find the answer, <give_up>."
    answer_source: str  # a noun phrase: "Your answer is <answer_source>."


QA_SMOLAGENTS = QAProtocol(
    name="qa:smolagents",
    finish_rule='call final_answer with your answer as its "answer" argument',
    give_up='call final_answer with the single word SORRY as its "answer" argument',
    answer_source='the "answer" argument of your final_answer call',
)

QA_QWEN = QAProtocol(
    name="qa:qwen_agent",
    finish_rule='reply without calling any tool, and end that reply with a line of the form "FINAL ANSWER: <your answer>"',
    give_up='reply without calling any tool, and end that reply with the line "FINAL ANSWER: SORRY"',
    answer_source='everything after the last "FINAL ANSWER:" in the reply that ends the task, so write nothing after '
                  "your answer",
)

QA_PROTOCOLS = (QA_SMOLAGENTS.name, QA_QWEN.name)
