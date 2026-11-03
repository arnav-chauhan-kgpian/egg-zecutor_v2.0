"""A hosted OpenAI-compatible chat endpoint (OpenRouter by default).

The provider applies the chat template server-side and never shows it, so
`exact_input` is the request JSON exactly as sent (`exact_input_kind:
api_request`) and no token ids are recorded (`tokens: null`); such
trajectories cannot be replayed for activations (DESIGN.md §11.4).

The context rule still holds (§6): the prompt is rendered and counted with
the model's own tokenizer (`api.tokenizer`) before sending, and a provider's
"context length exceeded" error is mapped to `ContextExhausted`. Provider-side
prompt compression is switched off (OpenRouter's `transforms: []`).

Retries follow `cf_agent`'s `ApiModel`: a raised transport error, or an HTTP
200 whose body reports `finish_reason == "error"`, is retried with
exponential backoff; when every try fails the call raises rather than hand
back a fragment as though the model had said it.
"""
from __future__ import annotations

import json
import os
import time

from ..core.config import ModelConfig, RecordingSettings
from ..core.errors import ConfigError, ContextExhausted
from ..core.types import Generation
from .base import ModelBackend, cut_at_stop_sequences

_CONTEXT_ERRORS = ("context length", "maximum context", "context_length_exceeded", "too many tokens", "prompt is too long")


class OpenAICompatBackend(ModelBackend):
    RETRY_BACKOFF_SECONDS = 2.0

    def __init__(self, cfg: ModelConfig, recording: RecordingSettings):
        super().__init__(cfg, recording)
        from openai import OpenAI

        api = cfg.api
        key = os.environ.get(api.api_key_env)
        if not key:
            raise ConfigError(f"no API key: set the environment variable {api.api_key_env}")
        self.client = OpenAI(base_url=api.base_url, api_key=key, timeout=api.timeout_s)
        self.max_new_tokens = cfg.generation.max_new_tokens
        reserve = cfg.generation.reply_reserve
        self.reply_reserve = self.max_new_tokens if reserve == "full" else int(reserve)
        self.window = api.context_window
        self._tokenizer = None
        if api.tokenizer:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(api.tokenizer)
        if self.window is None and self._tokenizer is not None:
            self.window = getattr(self._tokenizer, "model_max_length", None)
        g = cfg.generation
        self.sampling = {k: v for k, v in {
            "temperature": g.temperature, "top_p": g.top_p,
        }.items() if v is not None}
        self.extra_body = {}
        if g.top_k is not None:
            self.extra_body["top_k"] = g.top_k
        if g.repetition_penalty is not None:
            self.extra_body["repetition_penalty"] = g.repetition_penalty
        if api.disable_provider_transforms:
            self.extra_body["transforms"] = []

    @property
    def info(self) -> dict:
        return {
            "id": self.cfg.id,
            "backend": "openai_compat",
            "base_url": self.cfg.api.base_url,
            "context_window": self.window,
            "max_new_tokens": self.max_new_tokens,
            "reply_reserve": self.reply_reserve,
            "sampling_explicit": {**self.sampling, **{k: v for k, v in self.extra_body.items() if k != "transforms"}},
            "provider_transforms_disabled": self.cfg.api.disable_provider_transforms,
            "token_counting_tokenizer": self.cfg.api.tokenizer,
            "note": "the provider applies the chat template; exact_input is the request JSON and no token ids are recorded",
        }

    @property
    def tokenizer(self):
        return self._tokenizer

    def _count_prompt(self, messages, tools) -> int | None:
        if self._tokenizer is None:
            return None
        text = self._tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate(self, messages, *, tools=None, stop=None, seed, content_decode="skip_special_tokens") -> Generation:
        prompt_tokens = self._count_prompt(messages, tools)
        if prompt_tokens is not None and self.window and prompt_tokens + self.reply_reserve > self.window:
            raise ContextExhausted(prompt_tokens, self.window, self.reply_reserve)
        request = {
            "model": self.cfg.id,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "seed": seed,
            **self.sampling,
        }
        if tools:
            request["tools"] = tools
        if stop:
            request["stop"] = list(stop)
        if self.extra_body:
            request["extra_body"] = self.extra_body
        exact_input = json.dumps(request, indent=2, ensure_ascii=False)

        last_error: Exception | None = None
        retries = max(1, self.cfg.api.max_retries)
        for attempt in range(1, retries + 1):
            started = time.perf_counter()
            try:
                completion = self.client.chat.completions.create(**request)
            except Exception as exc:  # noqa: BLE001 - transport failures are retried
                if any(marker in str(exc).lower() for marker in _CONTEXT_ERRORS):
                    raise ContextExhausted(prompt_tokens or -1, self.window or -1, self.reply_reserve) from exc
                last_error = exc
                self._backoff(attempt, retries)
                continue
            choice = completion.choices[0]
            finish = getattr(choice, "finish_reason", None)
            if finish == "error":
                last_error = RuntimeError("provider returned finish_reason='error' inside an HTTP 200")
                self._backoff(attempt, retries)
                continue
            seconds = time.perf_counter() - started
            message = choice.message
            raw = message.content or ""
            if not raw and getattr(message, "tool_calls", None):
                # A native tool call carries no text; record the object rather than an empty reply.
                raw = json.dumps([tc.model_dump() for tc in message.tool_calls], ensure_ascii=False)
            usage = getattr(completion, "usage", None)
            mapped = {"stop": "eos", "length": "length", "tool_calls": "eos"}.get(finish or "", finish or "eos")
            if mapped == "eos" and stop and any(s in raw for s in stop):
                mapped = "stop_sequence"
            return Generation(
                exact_input=exact_input,
                exact_input_kind="api_request",
                raw_output=raw,
                content=cut_at_stop_sequences(raw, stop),
                input_tokens=getattr(usage, "prompt_tokens", None) if usage else prompt_tokens,
                output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                hit_length_cap=finish == "length",
                finish_reason=mapped,
                seconds=seconds,
                sampling={**self.sampling, **self.extra_body, "seed": seed, "stop": list(stop or []),
                          "provider_finish_reason": finish},
                context_window=self.window,
                reply_budget=self.max_new_tokens,
                content_decode="provider",
                messages=messages,
            )
        raise RuntimeError(f"{self.cfg.id}: all {retries} attempts failed ({last_error})") from last_error

    def _backoff(self, attempt: int, retries: int) -> None:
        if attempt < retries:
            time.sleep(self.RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
