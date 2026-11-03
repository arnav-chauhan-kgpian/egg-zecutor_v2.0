"""The local engine: a Hugging Face model on this machine's GPUs.

Ported from `cf_agent`'s `LocalModel` / `LocalToolCallingModel`, with the
same commitments — the exact prompt, the exact reply, the same loading rules
— plus three things the research needs (DESIGN.md §6, §11.4):

* **The context rule.** If the prompt plus the reply reserve does not fit
  the window, `ContextExhausted` is raised instead of generating. Nothing is
  ever truncated, and nothing is generated past the trained positions.
* **Token-level recording.** Exact input and output ids; per generated token
  the log-prob of the chosen token, the entropy and the top-k, all from the
  *raw* logits; and segment spans.
* **A decoding chain we can vouch for.** transformers appends a caller's
  logits processors *after* its own repetition penalty but *before*
  temperature/top-k/top-p, and fills any `None` field of a passed generation
  config from the checkpoint's `generation_config.json`. So the engine sets
  every sampling field to an explicit neutral value and builds the chain
  itself, in transformers' own order, with the recorder first:

      [LogprobRecorder, RepetitionPenalty, Temperature, TopK, TopP]

  At load time it asks transformers which processors it would add on its own
  and refuses to run unless the answer is "none".
"""
from __future__ import annotations

import gc
import hashlib
import os
import re
import time
from typing import Any

from ..core.config import ModelConfig, RecordingSettings
from ..core.errors import ConfigError, ContextExhausted
from ..core.types import Generation
from .base import ModelBackend, cut_at_stop_sequences, resolve_sampling
from .tokens import input_segments, output_segments

FALLBACK_CONTEXT_WINDOW = 32768
DEFAULT_CACHE_DIR = os.environ.get("CF_AGENT_MODEL_CACHE", "/tmp/cf_agent_models")

_SNAPSHOT = re.compile(r"/snapshots/([0-9a-f]{40})/")


def _resolve_dtype(dtype: str) -> str:
    """`auto` -> bfloat16 on compute capability >= 8 (Ampere+), float16
    below (a P100 has no bfloat16), float32 without a GPU."""
    if dtype != "auto":
        return dtype
    import torch

    if not torch.cuda.is_available():
        return "float32"
    major, _ = torch.cuda.get_device_capability(0)
    return "bfloat16" if major >= 8 else "float16"


def _keep_gqa_off_math_sdpa() -> bool:
    """Below compute capability 8 (a V100 or P100) PyTorch has no flash SDPA
    kernel, and the memory-efficient one refuses `enable_gqa`, which
    transformers passes for grouped-query models whenever there is no
    attention mask. Those calls fall to the math kernel, which materialises a
    float32 [heads, N, N] matrix per layer: 18.6 GiB for Qwen3-14B at 11k
    tokens. Making transformers repeat K/V itself lets the memory-efficient
    kernel run instead. Returns whether the patch was applied."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] >= 8:
        return False
    import transformers.integrations.sdpa_attention as sdpa_attention

    sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False  # looked up at call time by sdpa_attention_forward
    return True


def execution_device(module):
    """The device `module` runs on. Under an accelerate device map that
    offloads to CPU or disk, its weights sit on the `meta` device and are
    brought to the hook's execution device for each call, so a weight's
    device is not where inputs belong."""
    import torch

    hooks = [getattr(module, "_hf_hook", None)]
    while hooks:
        hook = hooks.pop()
        if hook is None:
            continue
        if getattr(hook, "execution_device", None) is not None:
            return torch.device(hook.execution_device)
        hooks.extend(getattr(hook, "hooks", None) or [])  # SequentialHook
    return next(module.parameters()).device


def _snapshot_hash(model_id: str, filename: str, revision: str | None, cache_dir: str | None) -> str | None:
    """The commit a cached file was resolved from, read from its snapshot path."""
    try:
        from transformers.utils import cached_file

        path = cached_file(model_id, filename, revision=revision, cache_dir=cache_dir)
    except Exception:  # noqa: BLE001 - provenance is best-effort; the id is still recorded
        return None
    if path is None:
        return None
    match = _SNAPSHOT.search(str(path))
    return match.group(1) if match else None


class LogprobRecorder:
    """A logits processor that records and never modifies.

    Placed first in the chain, it sees the raw logits for each position.
    The chosen token is not known until the next call (it is then the last
    input id), so the previous position's log-softmax row is kept — one
    vocabulary-sized vector, never the whole sequence's logits.
    """

    def __init__(self, top_k: int, want_logprobs: bool):
        self.top_k = top_k
        self.want_logprobs = want_logprobs
        self.prev = None
        self.logprobs: list[float] = []
        self.entropy: list[float] = []
        self.top: list[list[tuple[int, float]]] = []

    def __call__(self, input_ids, scores):
        import torch

        if self.prev is not None and self.want_logprobs:
            self.logprobs.append(float(self.prev[int(input_ids[0, -1])]))
        logsoftmax = torch.log_softmax(scores[0].float(), dim=-1)
        if self.want_logprobs:
            self.entropy.append(float(torch.special.entr(logsoftmax.exp()).sum()))
        if self.top_k:
            values, indices = torch.topk(logsoftmax, self.top_k)
            self.top.append([(int(i), float(v)) for i, v in zip(indices.tolist(), values.tolist())])
        self.prev = logsoftmax
        return scores

    def finish(self, last_token: int | None) -> None:
        if self.prev is not None and self.want_logprobs and last_token is not None:
            self.logprobs.append(float(self.prev[last_token]))
        self.prev = None


class HFLocalBackend(ModelBackend):
    def __init__(self, cfg: ModelConfig, recording: RecordingSettings):
        super().__init__(cfg, recording)
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        # cuDNN SDPA failed to build plans for Qwen2.5-Coder-32B ("No valid execution plans built").
        torch.backends.cuda.enable_cudnn_sdp(False)
        gqa_kv_repeated = _keep_gqa_off_math_sdpa()

        load = cfg.load
        self.cache_dir = load.cache_dir or DEFAULT_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        common = {"revision": cfg.revision, "cache_dir": self.cache_dir, "trust_remote_code": load.trust_remote_code}

        hf_config = AutoConfig.from_pretrained(cfg.id, **common)
        prequant = getattr(hf_config, "quantization_config", None)
        dtype = _resolve_dtype(load.dtype)
        model_kwargs: dict[str, Any] = {"device_map": load.device_map, **common}
        if prequant is not None:
            method = prequant.get("quant_method") if isinstance(prequant, dict) else getattr(prequant, "quant_method", "?")
            quantization = f"prequantized:{method}"
            if dtype == "bfloat16" and str(method).lower() == "awq":
                dtype = "float16"  # AWQ kernels have no bf16 path
        elif load.quantization == "nf4":
            from transformers import BitsAndBytesConfig

            quantization = "nf4"
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, dtype)
            )
        else:
            quantization = "none"
        model_kwargs["dtype"] = dtype
        if load.max_memory:
            model_kwargs["max_memory"] = {(int(k) if str(k).isdigit() else k): v for k, v in load.max_memory.items()}

        print(f"[engine] loading {cfg.id} (dtype {dtype}, quantization {quantization}, device_map {load.device_map})", flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.id, **common)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.id, **model_kwargs)
        self.model.eval()
        self._torch = torch

        text_config = hf_config.get_text_config() if hasattr(hf_config, "get_text_config") else hf_config
        trained = getattr(text_config, "max_position_embeddings", None)
        self.trained_window = trained
        window = trained or FALLBACK_CONTEXT_WINDOW
        if cfg.generation.max_context:
            if trained and cfg.generation.max_context > trained:
                raise ConfigError(
                    f"max_context {cfg.generation.max_context} exceeds the model's trained window {trained}"
                )
            window = min(window, cfg.generation.max_context)
        self.window = window
        self.max_new_tokens = cfg.generation.max_new_tokens
        reserve = cfg.generation.reply_reserve
        self.reply_reserve = self.max_new_tokens if reserve == "full" else int(reserve)
        if self.reply_reserve >= self.window:
            raise ConfigError(f"reply reserve {self.reply_reserve} leaves no room in a {self.window}-token window")

        self.template_kwargs = dict(cfg.chat_template.kwargs)
        if cfg.chat_template.enable_thinking is not None:
            self.template_kwargs["enable_thinking"] = cfg.chat_template.enable_thinking

        self.eos_ids = self._eos_ids()
        self.sampling = self._effective_sampling()
        self._verify_no_hidden_processors()

        self._info = {
            "id": cfg.id,
            "revision": _snapshot_hash(cfg.id, "config.json", cfg.revision, self.cache_dir) or cfg.revision,
            "tokenizer_revision": _snapshot_hash(cfg.id, "tokenizer_config.json", cfg.revision, self.cache_dir),
            "requested_revision": cfg.revision,
            "backend": "hf_local",
            "dtype": dtype,
            "quantization": quantization,
            "device_map": load.device_map,
            "trained_context_window": trained,
            "context_window": self.window,
            "max_new_tokens": self.max_new_tokens,
            "reply_reserve": self.reply_reserve,
            "generation": cfg.generation.model_dump(),
            "sampling_effective": self.sampling,
            "generation_config_defaults": self._generation_defaults(),
            "chat_template_sha256": hashlib.sha256((self._tokenizer.chat_template or "").encode()).hexdigest(),
            "chat_template_kwargs": self.template_kwargs,
            "eos_token_ids": sorted(self.eos_ids),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "attention": {"cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(), "gqa_kv_repeated": gqa_kv_repeated},
        }

    # ------------------------------------------------------------ setup

    @property
    def info(self) -> dict:
        return self._info

    @property
    def tokenizer(self):
        return self._tokenizer

    def _eos_ids(self) -> set[int]:
        eos = self.model.generation_config.eos_token_id
        ids = set(eos if isinstance(eos, list) else ([eos] if eos is not None else []))
        if self._tokenizer.eos_token_id is not None:
            ids.add(self._tokenizer.eos_token_id)
        return ids

    def _generation_defaults(self) -> dict:
        keep = ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "eos_token_id",
                "pad_token_id", "bos_token_id")
        conf = self.model.generation_config.to_dict()
        return {k: conf.get(k) for k in keep if k in conf}

    def _effective_sampling(self) -> dict:
        """`resolve_sampling` against the loaded checkpoint (cf_agent's
        policy: an un-flagged run is the model's native behaviour)."""
        return resolve_sampling(self.cfg.generation, self.model.generation_config)

    def _neutral_generation_config(self, max_new_tokens: int):
        """A generation config in which transformers itself adds no logits
        processor: every sampling field explicitly neutral (a `None` would be
        refilled from the checkpoint), every other processor field off."""
        from transformers import GenerationConfig

        conf = GenerationConfig(**self.model.generation_config.to_dict())
        conf.max_new_tokens = max_new_tokens
        conf.max_length = None
        conf.do_sample = self.sampling["do_sample"]
        conf.temperature = 1.0
        conf.top_p = 1.0
        conf.top_k = 0 if self.sampling["do_sample"] else 50
        conf.repetition_penalty = 1.0
        conf.num_beams = 1
        conf.num_return_sequences = 1
        conf.pad_token_id = conf.pad_token_id if conf.pad_token_id is not None else self._tokenizer.pad_token_id
        if conf.pad_token_id is None:
            conf.pad_token_id = sorted(self.eos_ids)[0] if self.eos_ids else None
        conf.return_dict_in_generate = False
        conf.output_scores = False
        conf.output_logits = False
        return conf

    def _verify_no_hidden_processors(self) -> None:
        from transformers import LogitsProcessorList

        conf = self._neutral_generation_config(16)
        try:
            prepared, _ = self.model._prepare_generation_config(conf)
            processors = self.model._get_logits_processor(
                generation_config=prepared,
                input_ids_seq_length=8,
                encoder_input_ids=None,
                prefix_allowed_tokens_fn=None,
                logits_processor=LogitsProcessorList(),
                device=str(self._input_device()),
                model_kwargs={},
            )
        except TypeError as exc:  # a transformers version with a different private API
            raise ConfigError(
                f"cannot verify transformers' default logits processors ({exc}); pin transformers to the "
                "version recorded in hallu_harness/requirements.txt"
            ) from exc
        if len(processors) != 0:
            names = ", ".join(type(p).__name__ for p in processors)
            raise ConfigError(
                "the checkpoint's generation config enables logits processors the engine does not "
                f"control ({names}); refusing to run rather than record log-probs that are not raw"
            )

    def _input_device(self):
        return execution_device(self.model.get_input_embeddings())

    def _processors(self, recorder: LogprobRecorder | None):
        from transformers import LogitsProcessorList
        from transformers.generation.logits_process import (
            RepetitionPenaltyLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        s = self.sampling
        chain = []
        if recorder is not None:
            chain.append(recorder)
        if s["repetition_penalty"] != 1.0:
            chain.append(RepetitionPenaltyLogitsProcessor(penalty=s["repetition_penalty"]))
        if s["do_sample"]:
            if s["temperature"] != 1.0:
                chain.append(TemperatureLogitsWarper(s["temperature"]))
            if s["top_k"]:
                chain.append(TopKLogitsWarper(top_k=s["top_k"], min_tokens_to_keep=1))
            if s["top_p"] < 1.0:
                chain.append(TopPLogitsWarper(top_p=s["top_p"], min_tokens_to_keep=1))
        return LogitsProcessorList(chain)

    # ------------------------------------------------------------ rendering

    def render(self, messages: list[dict], tools: list[dict] | None) -> tuple[str, list[int]]:
        """The exact prompt and its ids: the chat template applied with the
        generation prompt, then tokenized without adding special tokens —
        which is what `apply_chat_template(tokenize=True)` does."""
        text = self._tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False, **self.template_kwargs
        )
        ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        return text, ids

    def check_context(self, prompt_tokens: int) -> int:
        """Returns the reply budget, or raises ContextExhausted."""
        if prompt_tokens + self.reply_reserve > self.window:
            raise ContextExhausted(prompt_tokens, self.window, self.reply_reserve)
        return min(self.max_new_tokens, self.window - prompt_tokens)

    # ------------------------------------------------------------ generation

    def generate(self, messages, *, tools=None, stop=None, seed, content_decode="skip_special_tokens") -> Generation:
        torch = self._torch
        text, ids = self.render(messages, tools)
        budget = self.check_context(len(ids))

        rec = self.recording
        want_lp = rec.logprobs
        recorder = LogprobRecorder(rec.top_logprobs, want_lp) if (want_lp or rec.top_logprobs) else None
        conf = self._neutral_generation_config(budget)
        if stop:
            conf.stop_strings = list(stop)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = self._input_device()
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        started = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                generation_config=conf,
                logits_processor=self._processors(recorder),
                tokenizer=self._tokenizer if stop else None,
            )
        seconds = time.perf_counter() - started
        generated = out[0, len(ids):].tolist()
        if recorder is not None:
            recorder.finish(generated[-1] if generated else None)

        raw_output = self._tokenizer.decode(generated, skip_special_tokens=False)
        if generated and generated[-1] in self.eos_ids:
            finish = "eos"
        elif stop and any(s in raw_output for s in stop):
            finish = "stop_sequence"
        elif len(generated) >= budget:
            finish = "length"
        else:
            finish = "stop_sequence"

        if content_decode == "keep_special_tokens":
            content = raw_output
            for token in {self._tokenizer.eos_token, self._tokenizer.pad_token, "<|im_end|>", "<|endoftext|>"}:
                if token:
                    content = content.replace(token, "")
        else:
            content = self._tokenizer.decode(generated, skip_special_tokens=True)
        content = cut_at_stop_sequences(content, stop)

        logprobs = recorder.logprobs if (recorder is not None and want_lp) else None
        if logprobs is not None and len(logprobs) != len(generated):
            raise RuntimeError(f"recorded {len(logprobs)} log-probs for {len(generated)} generated tokens")

        segments = None
        if rec.segments:
            ins = input_segments(self._tokenizer, messages, tools, self.template_kwargs, text, ids)
            outs = output_segments(self._tokenizer, generated)
            segments = (ins or []) + (outs or []) if (ins is not None or outs is not None) else None

        return Generation(
            exact_input=self._tokenizer.decode(ids, skip_special_tokens=False),
            exact_input_kind="rendered_template",
            raw_output=raw_output,
            content=content,
            input_tokens=len(ids),
            output_tokens=len(generated),
            hit_length_cap=finish == "length",
            finish_reason=finish,
            seconds=seconds,
            sampling={**self.sampling, "seed": seed, "stop": list(stop or [])},
            context_window=self.window,
            reply_budget=budget,
            content_decode=content_decode,
            input_token_ids=list(ids) if rec.token_ids else None,
            output_token_ids=generated if rec.token_ids else None,
            output_logprobs=logprobs,
            output_entropy=(recorder.entropy if (recorder is not None and want_lp) else None),
            top_logprobs=(recorder.top if (recorder is not None and rec.top_logprobs) else None),
            segments=segments,
            messages=messages,
        )

    def close(self) -> None:
        del self.model
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def clear_cache(self) -> None:
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
