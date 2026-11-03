"""Replays fixed replies — for exercising everything except the weights.

With `scripted.tokenizer` set, the real chat template is rendered and real
tokens are counted, so the context rule, token ids and segment spans behave
exactly as with `hf_local`; only the reply text is scripted (and log-probs
are absent). Without a tokenizer, `exact_input` is the messages as JSON.

A reply is either a string or a dict:

    {"text": "...", "finish_reason": "length"}   # simulate hitting the cap
    {"raise": "context"}                         # simulate ContextExhausted
    {"raise": "error", "message": "..."}         # simulate a failed call

The last reply repeats if the run asks for more, so a script cannot hang.
"""
from __future__ import annotations

import json

from ..core.config import ModelConfig, RecordingSettings
from ..core.errors import ContextExhausted
from ..core.types import Generation
from .base import ModelBackend, cut_at_stop_sequences
from .tokens import input_segments, output_segments


class ScriptedBackend(ModelBackend):
    def __init__(self, cfg: ModelConfig, recording: RecordingSettings, replies: list | None = None):
        super().__init__(cfg, recording)
        self.replies = list(replies if replies is not None else cfg.scripted.replies)
        if not self.replies:
            raise ValueError("a scripted backend needs at least one reply")
        self.index = 0
        self.calls: list[list[dict]] = []  # messages seen per call, for tests
        self._tokenizer = None
        if cfg.scripted.tokenizer:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(cfg.scripted.tokenizer)
        self.window = cfg.scripted.context_window
        self.max_new_tokens = cfg.generation.max_new_tokens
        reserve = cfg.generation.reply_reserve
        self.reply_reserve = self.max_new_tokens if reserve == "full" else int(reserve)

    @property
    def info(self) -> dict:
        return {
            "id": self.cfg.id,
            "backend": "scripted",
            "tokenizer": self.cfg.scripted.tokenizer,
            "context_window": self.window,
            "max_new_tokens": self.max_new_tokens,
            "reply_reserve": self.reply_reserve,
            "replies": len(self.replies),
        }

    @property
    def tokenizer(self):
        return self._tokenizer

    def _next(self):
        reply = self.replies[min(self.index, len(self.replies) - 1)]
        self.index += 1
        return reply

    def generate(self, messages, *, tools=None, stop=None, seed, content_decode="skip_special_tokens") -> Generation:
        self.calls.append(messages)
        reply = self._next()
        if isinstance(reply, str):
            reply = {"text": reply}

        tok = self._tokenizer
        if tok is not None:
            text = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            exact_input, kind, prompt_tokens = tok.decode(ids, skip_special_tokens=False), "rendered_template", len(ids)
        else:
            text, ids = None, None
            exact_input = json.dumps({"messages": messages, "tools": tools}, indent=2, ensure_ascii=False)
            kind, prompt_tokens = "scripted", None

        if reply.get("raise") == "context" or (
            prompt_tokens is not None and prompt_tokens + self.reply_reserve > self.window
        ):
            raise ContextExhausted(prompt_tokens or self.window, self.window, self.reply_reserve)
        if reply.get("raise") == "error":
            raise RuntimeError(reply.get("message", "scripted model failure"))

        raw = reply["text"]
        finish = reply.get("finish_reason", "eos")
        out_ids = tok(raw, add_special_tokens=False)["input_ids"] if tok is not None else None
        content = cut_at_stop_sequences(raw, stop)
        if stop and finish == "eos" and any(s in raw for s in stop):
            finish = "stop_sequence"

        segments = None
        if tok is not None and self.recording.segments:
            segments = (input_segments(tok, messages, tools, {}, text, ids) or []) + (
                output_segments(tok, out_ids) or []
            )
        return Generation(
            exact_input=exact_input,
            exact_input_kind=kind,
            raw_output=raw,
            content=content,
            input_tokens=prompt_tokens,
            output_tokens=len(out_ids) if out_ids is not None else None,
            hit_length_cap=finish == "length",
            finish_reason=finish,
            seconds=0.0,
            sampling={"scripted": True, "seed": seed, "stop": list(stop or [])},
            context_window=self.window,
            reply_budget=self.max_new_tokens,
            content_decode=content_decode,
            input_token_ids=list(ids) if (ids is not None and self.recording.token_ids) else None,
            output_token_ids=out_ids if (out_ids is not None and self.recording.token_ids) else None,
            segments=segments,
            messages=messages,
        )
