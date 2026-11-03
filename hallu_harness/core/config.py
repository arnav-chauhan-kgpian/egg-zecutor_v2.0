"""Configuration: one file per framework, composed from shared presets.

A framework file names a model preset (`configs/models/<name>.yaml`) and a
dataset preset (`configs/datasets/<name>.yaml`) and may override keys of
either. Resolution order, later wins (DESIGN.md §10):

1. the framework file;
2. the model preset, with `model_overrides` deep-merged into it;
3. the dataset preset, with `dataset_overrides` deep-merged into it;
4. `--set a.b.c=value` (value parsed as YAML; `--set model=<name>` and
   `--set dataset=<name>` swap a preset).

Every section is validated with `extra="forbid"`, so a misspelled key is an
error before any model is loaded, never a silently different run. The fully
resolved config is hashed and stored with every trajectory.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from . import registry
from .errors import ConfigError

# ----------------------------------------------------------------- model


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoadConfig(_Strict):
    device_map: str = "auto"
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"
    quantization: Literal["none", "nf4"] = "none"  # pre-quantized checkpoints are detected from config.json
    max_memory: dict[str, str] | None = None  # e.g. {"0": "40GiB", "1": "40GiB"}
    cache_dir: str | None = None
    trust_remote_code: bool = False


class GenerationSettings(_Strict):
    max_new_tokens: int = Field(8192, gt=0)
    max_context: int | None = Field(None, gt=0)  # None = the model's trained window
    reply_reserve: Literal["full"] | int = "full"  # full = max_new_tokens (DESIGN.md §6)
    # None = not passed: the checkpoint's generation_config.json decides.
    temperature: float | None = Field(None, ge=0)  # 0 = greedy
    top_p: float | None = Field(None, gt=0, le=1)
    top_k: int | None = Field(None, ge=0)
    repetition_penalty: float | None = Field(None, gt=0)

    @field_validator("reply_reserve")
    @classmethod
    def _reserve_positive(cls, value):
        if isinstance(value, int) and value <= 0:
            raise ValueError("reply_reserve must be 'full' or a positive integer")
        return value


class ChatTemplateSettings(_Strict):
    enable_thinking: bool | None = None  # None = not passed to the template
    kwargs: dict[str, Any] = Field(default_factory=dict)  # any other template variables


class ApiSettings(_Strict):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"  # the NAME of an env var; never a literal key
    timeout_s: float = 600.0
    max_retries: int = 3
    disable_provider_transforms: bool = True  # e.g. OpenRouter middle-out compression
    tokenizer: str | None = None  # HF tokenizer used to count prompt tokens locally
    context_window: int | None = None


class VLLMSettings(_Strict):
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str | None = None
    timeout_s: float = 1800.0
    tokenizer: str | None = None  # defaults to the model id


class ScriptedSettings(_Strict):
    replies: list[str] = Field(default_factory=list)
    tokenizer: str | None = None  # render real chat templates and count real tokens
    context_window: int = 32768


class ModelConfig(_Strict):
    id: str
    revision: str | None = None
    backend: Literal["hf_local", "openai_compat", "vllm", "scripted"] = "hf_local"
    load: LoadConfig = Field(default_factory=LoadConfig)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    chat_template: ChatTemplateSettings = Field(default_factory=ChatTemplateSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    vllm: VLLMSettings = Field(default_factory=VLLMSettings)
    scripted: ScriptedSettings = Field(default_factory=ScriptedSettings)


# ----------------------------------------------------------------- dataset


class ToolsSettings(_Strict):
    required: list[str]
    optional: list[str] = Field(default_factory=list)
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)


class DatasetConfig(_Strict):
    benchmark: str
    split: str | None = None
    selection: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    tools: ToolsSettings
    prompt: dict[str, Any] = Field(default_factory=dict)


# ----------------------------------------------------------------- run


class RecordingSettings(_Strict):
    token_ids: bool = True
    logprobs: bool = True
    top_logprobs: int = Field(5, ge=0)
    segments: bool = True


class LensSettings(_Strict):
    """Logit-lens readouts computed during the run, right after each model
    call, by the model the run has loaded (DESIGN.md §11.5). They change
    nothing the model is shown or generates, so they are not part of the
    experiment key."""

    enabled: bool = False
    top_k: int = Field(5, ge=1)
    chunk: int = Field(128, gt=0)  # positions decoded at once (memory: 4 x chunk x vocab floats)
    # The replay check (§11.4, `interp.extract.logprob_check`): a step is
    # rejected if more than max(max_fraction of its tokens, 2) differ from the
    # recorded log-probs by more than 0.1 nats, or any token by more than
    # `tolerance`.
    tolerance: float = Field(2.0, gt=0)
    max_fraction: float = Field(0.05, ge=0, le=1)
    # A step's readouts are not computed while the lens directory's disk has
    # less free space than this, so lens files never fill the disk the
    # trajectories are written to.
    min_free_gb: float = Field(5.0, ge=0)


class RunConfig(_Strict):
    samples_per_task: int = Field(1, ge=1)
    seed: int = 0
    verbosity: Literal["quiet", "normal", "verbose"] = "normal"
    verbose_input: Literal["full", "delta"] = "delta"
    recording: RecordingSettings = Field(default_factory=RecordingSettings)
    lens: LensSettings = Field(default_factory=LensSettings)
    output_dir: str = "trajectories"
    resume: bool = True
    # Where `run` copies everything it prints (core/runlog.py): one file,
    # appended to, or a directory that gets `<run_id>.log` per run.
    log_file: str | None = None
    log_dir: str | None = None

    @model_validator(mode="after")
    def _one_log_target(self):
        if self.log_file and self.log_dir:
            raise ValueError("set log_file or log_dir, not both")
        return self


# ----------------------------------------------------------------- resolved


FRAMEWORK_PARAMS = {
    "smolagents": "hallu_harness.frameworks.smol.params:SmolParams",
    "qwen_agent": "hallu_harness.frameworks.qwenagent.params:QwenAgentParams",
    "text": "hallu_harness.frameworks.text.params:TextParams",
}

_TOP_KEYS = {"framework", "model", "model_overrides", "dataset", "dataset_overrides", "tools_extra", "params", "run"}


class ResolvedConfig(BaseModel):
    """The configuration a run actually uses, after every merge and default."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    framework: str
    model_preset: str
    dataset_preset: str
    model: ModelConfig
    dataset: DatasetConfig
    tools_extra: list[str]
    params: dict[str, Any]
    run: RunConfig
    source: str | None = None
    sets: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return json.loads(self.model_dump_json())

    @property
    def sha256(self) -> str:
        body = self.to_dict()
        body.pop("source", None)  # where the file lived does not change the run
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()

    @property
    def tool_names(self) -> list[str]:
        """dataset.tools.required, then the framework's extras, de-duplicated in order."""
        out: list[str] = []
        for name in [*self.dataset.tools.required, *self.tools_extra]:
            if name not in out:
                out.append(name)
        return out


# ----------------------------------------------------------------- helpers

_ENV = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def _interpolate(value: Any) -> Any:
    """`${env:NAME}` / `${env:NAME:default}` in string values; `~` expanded in
    values that then look like paths."""
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    if not isinstance(value, str):
        return value

    def repl(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ConfigError(f"environment variable {name} is not set and has no default")

    out = _ENV.sub(repl, value)
    return os.path.expanduser(out) if out.startswith("~") else out


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    """A path from a config: absolute, or relative to the working directory,
    or else relative to the repository root (so presets work from anywhere)."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return REPO_ROOT / candidate


def deep_merge(base: dict, override: dict) -> dict:
    """`override` merged into a copy of `base`; dicts merge, everything else replaces."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def _preset_path(ref: str, kind: str, config_path: Path, configs_root: Path) -> Path:
    candidate = Path(ref)
    if ref.endswith((".yaml", ".yml")):
        if candidate.is_absolute():
            return candidate
        for base in (Path.cwd(), config_path.parent):
            if (base / candidate).exists():
                return base / candidate
        return config_path.parent / candidate
    return configs_root / f"{kind}s" / f"{ref}.yaml"


def _parse_set(item: str) -> tuple[list[str], Any]:
    if "=" not in item:
        raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
    key, raw = item.split("=", 1)
    key = key.strip()
    if not key or any(not part for part in key.split(".")):
        raise ConfigError(f"--set: bad key {key!r}")
    try:
        value = yaml.safe_load(raw) if raw.strip() else ""
    except yaml.YAMLError as exc:
        raise ConfigError(f"--set {key}: value is not valid YAML: {exc}") from None
    return key.split("."), value


def _set_path(tree: dict, path: list[str], value: Any) -> None:
    node = tree
    for part in path[:-1]:
        if part not in node or node[part] is None:
            node[part] = {}
        if not isinstance(node[part], dict):
            raise ConfigError(f"--set {'.'.join(path)}: {part!r} is not a mapping")
        node = node[part]
    node[path[-1]] = value


def _format_validation(where: str, exc: ValidationError) -> str:
    lines = [f"invalid {where}:"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        lines.append(f"  {loc}: {err['msg']}" if loc else f"  {err['msg']}")  # no loc: a whole-section check
    return "\n".join(lines)


def framework_params_model(framework: str):
    """The pydantic model that validates `params` for a framework."""
    if framework in FRAMEWORK_PARAMS:
        module_name, attr = FRAMEWORK_PARAMS[framework].split(":")
        return getattr(importlib.import_module(module_name), attr)
    adapter = registry.get("framework", framework)
    params = getattr(adapter, "Params", None)
    if params is None:
        raise ConfigError(f"framework {framework!r} declares no Params model")
    return params


# ----------------------------------------------------------------- loading


def load_config(path: str | Path, sets: list[str] | None = None, configs_root: str | Path | None = None) -> ResolvedConfig:
    """Reads, composes and validates a framework config. See the module doc."""
    path = Path(path).resolve()
    raw = _read_yaml(path)
    unknown = set(raw) - _TOP_KEYS
    if unknown:
        raise ConfigError(f"{path}: unknown top-level keys: {', '.join(sorted(unknown))}")
    if "framework" not in raw:
        raise ConfigError(f"{path}: missing 'framework'")
    for key in ("model", "dataset"):
        if not isinstance(raw.get(key), str):
            raise ConfigError(f"{path}: '{key}' must name a preset (or a path to one)")

    sets = list(sets or [])
    post: list[tuple[list[str], Any]] = []
    for item in sets:
        keys, value = _parse_set(item)
        if keys in (["model"], ["dataset"], ["framework"]):
            if not isinstance(value, str):
                raise ConfigError(f"--set {keys[0]} expects a preset name")
            raw[keys[0]] = value
        else:
            post.append((keys, value))

    root = Path(configs_root).resolve() if configs_root else path.parent.parent
    model_path = _preset_path(raw["model"], "model", path, root)
    dataset_path = _preset_path(raw["dataset"], "dataset", path, root)
    model = _interpolate(deep_merge(_read_yaml(model_path), raw.get("model_overrides") or {}))
    dataset = _interpolate(deep_merge(_read_yaml(dataset_path), raw.get("dataset_overrides") or {}))

    tree: dict[str, Any] = {
        "framework": raw["framework"],
        "model": model,
        "dataset": dataset,
        "tools_extra": list(raw.get("tools_extra") or []),
        "params": copy.deepcopy(raw.get("params") or {}),
        "run": copy.deepcopy(raw.get("run") or {}),
    }
    for keys, value in post:
        if keys[0] not in tree:
            raise ConfigError(f"--set {'.'.join(keys)}: unknown section {keys[0]!r}")
        if len(keys) == 1:
            tree[keys[0]] = value
        else:
            _set_path(tree, keys, value)

    try:
        model_cfg = ModelConfig(**tree["model"])
    except ValidationError as exc:
        raise ConfigError(_format_validation(f"model config ({model_path.name})", exc)) from None
    try:
        dataset_cfg = DatasetConfig(**tree["dataset"])
    except ValidationError as exc:
        raise ConfigError(_format_validation(f"dataset config ({dataset_path.name})", exc)) from None
    try:
        run_cfg = RunConfig(**tree["run"])
    except ValidationError as exc:
        raise ConfigError(_format_validation("run config", exc)) from None

    framework = tree["framework"]
    if framework not in registry.names("framework"):
        raise ConfigError(f"unknown framework {framework!r}; known: {', '.join(registry.names('framework'))}")
    params_model = framework_params_model(framework)
    try:
        params = params_model(**tree["params"]).model_dump()
    except ValidationError as exc:
        raise ConfigError(_format_validation(f"params for framework {framework!r}", exc)) from None

    resolved = ResolvedConfig(
        framework=framework,
        model_preset=raw["model"],
        dataset_preset=raw["dataset"],
        model=model_cfg,
        dataset=dataset_cfg,
        tools_extra=tree["tools_extra"],
        params=params,
        run=run_cfg,
        source=str(path),
        sets=sets,
    )
    validate_resolved(resolved)
    return resolved


def validate_resolved(cfg: ResolvedConfig) -> None:
    """Cross-section checks that no single section can make on its own."""
    benchmark_cls = registry.get("benchmark", cfg.dataset.benchmark)
    benchmark_cls.validate_config(cfg.dataset)

    tools = cfg.dataset.tools
    for name in [*tools.required, *tools.optional]:
        if name not in registry.names("tool"):
            raise ConfigError(f"dataset lists unknown tool {name!r}; known: {', '.join(registry.names('tool'))}")
    for name in cfg.tools_extra:
        if name not in tools.optional:
            raise ConfigError(
                f"tools_extra names {name!r}, which the dataset does not list under tools.optional"
            )
    # The model is told how its solution is chosen (the solution policy) and
    # that the submit tool's submission is graded; both must be true.
    policy = list(getattr(benchmark_cls.Evaluation(**cfg.dataset.evaluation), "solution_policy", []))
    if "submit" in cfg.tool_names and "submitted" not in policy:
        raise ConfigError(
            "the submit tool is in use but dataset.evaluation.solution_policy does not list 'submitted'; "
            "the tool tells the model that its submission is graded"
        )
    if "submitted" in policy and "submit" not in cfg.tool_names:
        raise ConfigError(
            "dataset.evaluation.solution_policy lists 'submitted' but the submit tool is not in use; "
            "the model would be told about a tool it does not have"
        )
    for name, params in tools.params.items():
        if name not in cfg.tool_names:
            raise ConfigError(f"tools.params configures {name!r}, which this run does not use")
        allowed = getattr(registry.get("tool", name), "allowed_params", set())
        unknown = set(params) - set(allowed)
        if unknown:
            raise ConfigError(
                f"tools.params.{name}: unknown key(s) {', '.join(sorted(unknown))}; allowed: {', '.join(sorted(allowed)) or 'none'}"
            )

    if cfg.run.lens.enabled and not cfg.run.recording.token_ids:
        raise ConfigError("run.lens needs run.recording.token_ids: the readouts are computed from the recorded ids")

    if cfg.model.backend == "hf_local" and cfg.model.generation.reply_reserve != "full":
        reserve = cfg.model.generation.reply_reserve
        if reserve > cfg.model.generation.max_new_tokens:
            raise ConfigError("reply_reserve cannot exceed max_new_tokens")


# ----------------------------------------------------------------- diff


def _flatten(tree: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(tree, dict):
        out: dict[str, Any] = {}
        for key, value in tree.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: tree}


def diff_configs(a: ResolvedConfig, b: ResolvedConfig) -> dict[str, list[tuple[str, Any, Any]]]:
    """Differences between two resolved configs, grouped by section.

    For a fair framework comparison only `framework` and `params` (and the
    framework's own tool extras) should differ; anything under `model`,
    `dataset` or `run` is a confound.
    """
    da, db = a.to_dict(), b.to_dict()
    out: dict[str, list[tuple[str, Any, Any]]] = {}
    for section in ("framework", "model", "dataset", "tools_extra", "params", "run"):
        fa, fb = _flatten(da.get(section), section), _flatten(db.get(section), section)
        diffs = [(key, fa.get(key), fb.get(key)) for key in sorted(set(fa) | set(fb)) if fa.get(key) != fb.get(key)]
        if diffs:
            out[section] = diffs
    return out
