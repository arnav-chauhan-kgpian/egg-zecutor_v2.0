"""Framework parameters for the text protocol (validated before any model loads)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TextParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_model_calls: int = Field(6, gt=0)
    code_tool: str = "grade_code"  # the tool ```python blocks are routed to
    prompt_mode: Literal["harmonized", "stock"] = "harmonized"  # identical for this framework (no stock prompt exists)
    # cf_agent's text loop decoded with special tokens and removed only the
    # turn-ending ones; that is kept for comparability with earlier runs.
    content_special_tokens: Literal["skip", "keep"] = "keep"
