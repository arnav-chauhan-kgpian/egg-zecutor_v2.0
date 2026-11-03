"""Reply phrases for code tasks that have no grading tool (HALoGEN's
code-imports subset): the model looks things up with its tools and ends the
task with its final program, which is evaluated after the task.

How a framework finishes and gives up is the same as on the coding
benchmarks, so those phrases are the coding protocols' own. The phrases
about grading and resubmitting are blanked, so that no text that mentions
`grade_code` can reach a model through them.

`UNGRADED_PROMPT_REVISION` is the revision of every model-facing text on this
path: the HALoGEN task instructions, the ungraded framework variants' reply
sections, and the web and GitHub tool descriptions. `core/protocol.py`'s
`PROMPT_REVISION` covers the coding texts and is unchanged by this path. The
benchmark puts this revision in the experiment key (through
`Benchmark.grading_inputs`); bump it whenever any of those texts change.
"""
from __future__ import annotations

from dataclasses import replace

from .protocol import TOOL_PROTOCOL_QWEN, TOOL_PROTOCOL_SMOLAGENTS, Protocol

UNGRADED_PROMPT_REVISION = 2  # 2: github_repository_search cuts long descriptions

_GRADING_FIELDS = ("initial", "resubmit_after_visible", "resubmit_after_hidden", "submit_option", "revise_option", "finish_now")

UNGRADED_SMOLAGENTS: Protocol = replace(TOOL_PROTOCOL_SMOLAGENTS, name="tool:smolagents_ungraded",
                                        **{f: "" for f in _GRADING_FIELDS})
UNGRADED_QWEN: Protocol = replace(TOOL_PROTOCOL_QWEN, name="tool:qwen_agent_ungraded", **{f: "" for f in _GRADING_FIELDS})
UNGRADED_PROTOCOLS = (UNGRADED_SMOLAGENTS.name, UNGRADED_QWEN.name)
