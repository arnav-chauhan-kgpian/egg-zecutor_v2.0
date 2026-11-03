"""The interface every framework bridge calls, and nothing else does.

One backend instance is created per run and shared by every trajectory; the
frameworks never load weights or choose decoding settings themselves
(DESIGN.md §6). They may pass only two things: stop sequences (part of their
protocol) and nothing else — the seed comes from the recorder, and the
decoding of `content` is the adapter's declared policy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.config import ModelConfig, RecordingSettings
from ..core.types import Generation


class ModelBackend(ABC):
    """A model the harness can call."""

    def __init__(self, cfg: ModelConfig, recording: RecordingSettings):
        self.cfg = cfg
        self.recording = recording

    @property
    @abstractmethod
    def info(self) -> dict:
        """Model metadata for the trajectory (DESIGN.md §6)."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
        seed: int,
        content_decode: str = "skip_special_tokens",
    ) -> Generation:
        """One model call.

        `messages` are plain `{"role", "content"}` dicts with string content;
        `tools` are JSON-schema function specs passed to the chat template (the
        smolagents path) or None (Qwen-Agent writes its own tool block into the
        system message). Raises `ContextExhausted` instead of generating when
        the prompt plus the reply reserve does not fit the window.

        `content_decode` is how `Generation.content` — the text handed back to
        the framework — is decoded: "skip_special_tokens" (what stock
        smolagents and a vLLM server do) or "keep_special_tokens" (cf_agent's
        text loop, which drops only the turn-ending tokens). `raw_output` is
        always decoded with special tokens and before any trimming.
        """

    def close(self) -> None:
        """Releases GPU memory; the backend is unusable afterwards."""

    @property
    def tokenizer(self) -> Any:
        return None


def resolve_sampling(generation, checkpoint) -> dict:
    """The decoding settings in effect: explicit settings win; anything left
    unset is the checkpoint's own `generation_config.json` value, and failing
    that transformers' default (top_k=50, not 0) — so an un-flagged run is the
    model's native behaviour, whichever backend serves it. `temperature: 0`
    means greedy. `checkpoint` is a transformers `GenerationConfig`."""
    from transformers import GenerationConfig

    hf_defaults = GenerationConfig._get_default_generation_params()

    def pick(name):
        explicit = getattr(generation, name)
        if explicit is not None:
            return explicit
        value = getattr(checkpoint, name, None)
        return hf_defaults[name] if value is None else value

    do_sample = getattr(checkpoint, "do_sample", None)
    do_sample = bool(hf_defaults["do_sample"] if do_sample is None else do_sample)
    if generation.temperature is not None:
        do_sample = generation.temperature > 0
    return {
        "do_sample": do_sample,
        "temperature": float(pick("temperature")) if do_sample else None,
        "top_p": float(pick("top_p")) if do_sample else None,
        "top_k": int(pick("top_k") or 0) if do_sample else None,
        "repetition_penalty": float(pick("repetition_penalty")),
    }


def cut_at_stop_sequences(text: str, stop: list[str] | None) -> str:
    """smolagents' `remove_content_after_stop_sequences`: cut at the earliest
    occurrence of any stop sequence, excluding it."""
    if not stop:
        return text
    for sequence in stop:
        text = text.split(sequence)[0]
    return text
