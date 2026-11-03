"""Framework parameters for Qwen-Agent (validated before any model loads)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QwenAgentParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_class: Literal["FnCallAgent"] = "FnCallAgent"
    max_model_calls: int = Field(6, gt=0)  # -> qwen_agent's MAX_LLM_CALL_PER_RUN
    fncall_prompt_type: Literal["nous", "qwen"] = "nous"
    parallel_function_calls: bool = False
    thought_in_content: bool = False
    # Qwen-Agent silently truncates the conversation past this many tokens,
    # counted approximately with its own tokenizer. It must stay off: the
    # engine ends the trajectory instead (DESIGN.md §6).
    max_input_tokens: int = 0
    prompt_mode: Literal["harmonized", "stock"] = "harmonized"
    # How the reply handed back to Qwen-Agent is decoded; "skip" is what a
    # vLLM OpenAI-compatible server does by default.
    content_special_tokens: Literal["skip", "keep"] = "skip"

    @field_validator("max_input_tokens")
    @classmethod
    def _no_truncation(cls, value: int) -> int:
        if value != 0:
            raise ValueError(
                "must be 0: any other value lets Qwen-Agent silently truncate the conversation "
                "(the engine ends the trajectory with context_exhausted instead)"
            )
        return value
