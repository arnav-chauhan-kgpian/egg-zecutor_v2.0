"""A vLLM server, driven through its OpenAI-compatible *completions* endpoint
(DESIGN.md §6, Phase 4).

The chat template is rendered locally with the model's own tokenizer and the
prompt is sent as token ids, so `exact_input` is exactly what the server
conditions on — the guarantee `hf_local` gives, at vLLM's throughput. The
reply comes back with `skip_special_tokens: false` and the stop string kept
(`include_stop_str_in_output`), so `raw_output` is untrimmed; token ids are
read from the log-prob entries (`return_tokens_as_token_ids`).

Requirements on the server side, recorded in `info`:

* vLLM's OpenAI-compatible server, serving the same weights revision;
* its default log-prob mode, which reports log-probs of the raw model
  distribution (`--logprobs-mode raw_logprobs` in recent versions) — the
  same quantity `hf_local` records, so replays can verify either;
* no entropy is available (only the top-k), so `output_entropy` is None.

Every decoding setting is sent explicitly, resolved exactly as `hf_local`
resolves it (`resolve_sampling`: the config, else the checkpoint's
`generation_config.json`), so the server's own defaults never decide one.

The context rule is enforced here exactly as in `hf_local`.

Not exercised against a live vLLM server on the machine this was written on
(vLLM is not installed there); tests/test_backends.py checks it against a
mock server that speaks the same protocol.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

from ..core.config import ModelConfig, RecordingSettings
from ..core.errors import ConfigError, ContextExhausted
from ..core.types import Generation
from .base import ModelBackend, cut_at_stop_sequences, resolve_sampling
from .tokens import input_segments, output_segments


def _post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _get(url: str, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _token_id(token: str) -> int | None:
    return int(token.split(":", 1)[1]) if token.startswith("token_id:") else None


class VLLMBackend(ModelBackend):
    def __init__(self, cfg: ModelConfig, recording: RecordingSettings):
        super().__init__(cfg, recording)
        from transformers import AutoTokenizer, GenerationConfig

        v = cfg.vllm
        self.base_url = v.base_url.rstrip("/")
        self.headers = {}
        if v.api_key_env:
            key = os.environ.get(v.api_key_env)
            if not key:
                raise ConfigError(f"no API key: set the environment variable {v.api_key_env}")
            self.headers["Authorization"] = f"Bearer {key}"
        self.timeout = v.timeout_s
        self._tokenizer = AutoTokenizer.from_pretrained(v.tokenizer or cfg.id, revision=cfg.revision)

        served = _get(f"{self.base_url}/models", self.headers, 30).get("data", [])
        entry = next((m for m in served if m.get("id") == cfg.id), None)
        if entry is None:
            raise ConfigError(f"the vLLM server at {self.base_url} does not serve {cfg.id!r} (serves {[m.get('id') for m in served]})")
        server_window = entry.get("max_model_len")
        window = server_window or getattr(self._tokenizer, "model_max_length", None)
        if cfg.generation.max_context:
            window = min(window, cfg.generation.max_context) if window else cfg.generation.max_context
        if not window:
            raise ConfigError("cannot determine the context window; set generation.max_context")
        self.window = int(window)
        self.max_new_tokens = cfg.generation.max_new_tokens
        reserve = cfg.generation.reply_reserve
        self.reply_reserve = self.max_new_tokens if reserve == "full" else int(reserve)
        self.template_kwargs = dict(cfg.chat_template.kwargs)
        if cfg.chat_template.enable_thinking is not None:
            self.template_kwargs["enable_thinking"] = cfg.chat_template.enable_thinking
        g = cfg.generation
        try:
            checkpoint = GenerationConfig.from_pretrained(v.tokenizer or cfg.id, revision=cfg.revision)
        except OSError as exc:
            if any(getattr(g, name) is None for name in ("temperature", "top_p", "top_k", "repetition_penalty")):
                raise ConfigError(
                    f"cannot read the generation_config.json of {cfg.id} ({exc}); set temperature, top_p, top_k and "
                    "repetition_penalty explicitly, or the server's defaults would decide them"
                ) from exc
            checkpoint = GenerationConfig()
        self.sampling_effective = resolve_sampling(g, checkpoint)
        s = self.sampling_effective
        # vLLM's spelling: temperature 0 is greedy, top_k -1 is off.
        self.sampling = {
            "temperature": s["temperature"] if s["do_sample"] else 0.0,
            "top_p": s["top_p"] if s["do_sample"] else 1.0,
            "top_k": (s["top_k"] or -1) if s["do_sample"] else -1,
            "repetition_penalty": s["repetition_penalty"],
        }
        self.server_window = server_window

    @property
    def info(self) -> dict:
        return {
            "id": self.cfg.id,
            "requested_revision": self.cfg.revision,
            "backend": "vllm",
            "base_url": self.base_url,
            "server_max_model_len": self.server_window,
            "context_window": self.window,
            "max_new_tokens": self.max_new_tokens,
            "reply_reserve": self.reply_reserve,
            "sampling_effective": self.sampling_effective,
            "sampling_request": self.sampling,
            "chat_template_kwargs": self.template_kwargs,
            "logprobs": "as reported by the server (raw distribution under vLLM's default logprobs mode)",
        }

    @property
    def tokenizer(self):
        return self._tokenizer

    def generate(self, messages, *, tools=None, stop=None, seed, content_decode="skip_special_tokens") -> Generation:
        text = self._tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False, **self.template_kwargs)
        ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) + self.reply_reserve > self.window:
            raise ContextExhausted(len(ids), self.window, self.reply_reserve)
        budget = min(self.max_new_tokens, self.window - len(ids))
        k = max(1, self.recording.top_logprobs)
        body = {
            "model": self.cfg.id,
            "prompt": ids,
            "max_tokens": budget,
            "seed": seed,
            "logprobs": k,
            "return_tokens_as_token_ids": True,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "include_stop_str_in_output": True,
            **self.sampling,
        }
        if stop:
            body["stop"] = list(stop)
        started = time.perf_counter()
        response = _post(f"{self.base_url}/completions", body, self.headers, self.timeout)
        seconds = time.perf_counter() - started
        choice = response["choices"][0]
        raw = choice.get("text") or ""
        lp = choice.get("logprobs") or {}
        tokens = lp.get("tokens") or []
        out_ids = [_token_id(t) for t in tokens]
        if any(t is None for t in out_ids):
            out_ids = None  # the server did not return ids; never guess them by re-tokenizing
        logprobs = [float(x) for x in (lp.get("token_logprobs") or [])] or None
        top = None
        if self.recording.top_logprobs and lp.get("top_logprobs"):
            top = []
            for row in lp["top_logprobs"]:
                pairs = sorted(((_token_id(t), float(v)) for t, v in (row or {}).items()), key=lambda p: -p[1])
                top.append([(i, v) for i, v in pairs if i is not None][: self.recording.top_logprobs])

        finish = choice.get("finish_reason")
        if finish == "length":
            mapped = "length"
        elif stop and any(s in raw for s in stop):
            mapped = "stop_sequence"
        else:
            mapped = "eos"
        if content_decode == "keep_special_tokens":
            content = raw
            for token in {self._tokenizer.eos_token, self._tokenizer.pad_token, "<|im_end|>", "<|endoftext|>"}:
                if token:
                    content = content.replace(token, "")
        else:
            content = self._tokenizer.decode(out_ids, skip_special_tokens=True) if out_ids is not None else raw
        content = cut_at_stop_sequences(content, stop)

        segments = None
        if self.recording.segments:
            ins = input_segments(self._tokenizer, messages, tools, self.template_kwargs, text, ids)
            outs = output_segments(self._tokenizer, out_ids) if out_ids is not None else None
            segments = (ins or []) + (outs or []) if (ins is not None or outs is not None) else None
        return Generation(
            exact_input=self._tokenizer.decode(ids, skip_special_tokens=False),
            exact_input_kind="rendered_template",
            raw_output=raw,
            content=content,
            input_tokens=len(ids),
            output_tokens=len(out_ids) if out_ids is not None else None,
            hit_length_cap=mapped == "length",
            finish_reason=mapped,
            seconds=seconds,
            sampling={**self.sampling_effective, "seed": seed, "stop": list(stop or []), "server_finish_reason": finish},
            context_window=self.window,
            reply_budget=budget,
            content_decode=content_decode,
            input_token_ids=list(ids) if self.recording.token_ids else None,
            output_token_ids=out_ids if self.recording.token_ids else None,
            output_logprobs=logprobs if self.recording.logprobs else None,
            output_entropy=None,
            top_logprobs=top,
            segments=segments,
            messages=messages,
        )
