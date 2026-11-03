"""Exceptions with a meaning the harness acts on."""
from __future__ import annotations


class HarnessError(Exception):
    """Base class for errors raised by the harness itself."""


class ConfigError(HarnessError):
    """A configuration that cannot be run as written. Raised before any model
    is loaded, so a typo costs seconds rather than a GPU-hour."""


class ContextExhausted(HarnessError):
    """The next prompt plus the reply reserve no longer fits the window.

    Raised by the model engine *instead of* generating (DESIGN.md §6): no
    component is allowed to shorten the conversation, so the trajectory ends
    here with `stop_reason: context_exhausted`.
    """

    def __init__(self, prompt_tokens: int, window: int, reply_reserve: int):
        self.prompt_tokens = prompt_tokens
        self.window = window
        self.reply_reserve = reply_reserve
        super().__init__(
            f"context exhausted: prompt of {prompt_tokens} tokens + reply reserve of "
            f"{reply_reserve} exceeds the {window}-token window"
        )

    def to_dict(self) -> dict:
        return {
            "prompt_tokens_at_stop": self.prompt_tokens,
            "window": self.window,
            "reply_reserve": self.reply_reserve,
        }


class ReplayMismatch(HarnessError):
    """An offline replay did not reproduce the recorded log-probs."""


class HarnessFailure(HarnessError):
    """The harness could not do its own part of a trajectory: fetch the test
    data, run the grader, or a tool failed for a reason that is not the
    model's. Never reported to the model; the trajectory is recorded as
    `crashed` and re-run on resume."""


class HarnessAbort(BaseException):
    """Carries a harness failure out of a framework's agent loop.

    smolagents and Qwen-Agent catch every `Exception` a tool raises, show it
    to the model as an error message and go on. A failure of the harness is
    not the model's doing and must not enter its context, so it travels as a
    `BaseException`, which neither framework catches. The runner catches it
    and records the trajectory as `crashed`.
    """
