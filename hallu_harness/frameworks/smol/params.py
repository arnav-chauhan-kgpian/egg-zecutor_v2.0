"""Framework parameters for smolagents (validated before any model loads)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SmolParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_class: Literal["ToolCallingAgent"] = "ToolCallingAgent"
    max_model_calls: int = Field(6, gt=0)  # -> max_steps
    # smolagents' stock behaviour (True): once the step budget is spent, one
    # more model call composes an answer "from memory", recorded as step kind
    # final_from_memory. Off by default: that call sees the last grading
    # result, and code in its answer is graded as the final answer, so it is
    # one more informed attempt than Qwen-Agent and the text protocol get
    # for the same max_model_calls.
    answer_from_memory_on_budget: bool = False
    # Which turn of the history carries smolagents' record of each past call,
    # "Calling tools:\n[{'id': ..., 'function': {...}}]". "assistant" is
    # stock smolagents: it is appended to the model's own reply, so the model
    # sees itself as having written it and starts writing it too. "user": it
    # opens the next user turn, just before "Observation:".
    calling_tools_turn: Literal["user", "assistant"] = "user"
    # Whether the tool schemas also go to the chat template, as stock
    # TransformersModel sends them. Qwen's template then appends a second
    # tools section that contradicts smolagents' own reply rules (several
    # calls per reply, in <tool_call> tags). Off: one listing, one format.
    chat_template_tools: bool = False
    planning_interval: int | None = Field(None, gt=0)
    prompt_mode: Literal["harmonized", "stock"] = "harmonized"
    tool_call_parser: Literal["stock", "tagged"] = "stock"  # decided: stock (DESIGN.md §18)
    # How the reply handed back to smolagents is decoded. "skip" is what the
    # stock TransformersModel does (skip_special_tokens=True).
    content_special_tokens: Literal["skip", "keep"] = "skip"
