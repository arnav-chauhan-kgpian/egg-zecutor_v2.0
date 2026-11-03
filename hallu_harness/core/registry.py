"""Name -> implementation, for the four kinds of plug-in.

Built-ins are registered by import path, so naming a framework in a config
imports that framework and nothing else: a machine without `qwen_agent`
installed can still run smolagents, and vice versa. Out-of-tree plug-ins can
register a class directly with the decorators.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable

from .errors import ConfigError

_BUILTINS: dict[str, dict[str, str]] = {
    "framework": {
        "smolagents": "hallu_harness.frameworks.smol.adapter:SmolagentsAdapter",
        "qwen_agent": "hallu_harness.frameworks.qwenagent.adapter:QwenAgentAdapter",
        "text": "hallu_harness.frameworks.text.adapter:TextAdapter",
    },
    "benchmark": {
        "codeforces": "hallu_harness.benchmarks.codeforces.benchmark:CodeforcesBenchmark",
        "humaneval": "hallu_harness.benchmarks.humaneval.benchmark:HumanEvalBenchmark",
    },
    "tool": {
        "grade_code": "hallu_harness.tools.grade_code:GradeCodeTool",
        "run_visible_tests": "hallu_harness.tools.run_visible:RunVisibleTestsTool",
        "submit": "hallu_harness.tools.submit:SubmitTool",
        "python_exec": "hallu_harness.tools.python_exec:PythonExecTool",
    },
    "backend": {
        "hf_local": "hallu_harness.models.hf_local:HFLocalBackend",
        "scripted": "hallu_harness.models.scripted:ScriptedBackend",
        "openai_compat": "hallu_harness.models.openai_compat:OpenAICompatBackend",
        "vllm": "hallu_harness.models.vllm_backend:VLLMBackend",
    },
}

_REGISTERED: dict[str, dict[str, Any]] = {kind: {} for kind in _BUILTINS}


def _register(kind: str, name: str) -> Callable[[Any], Any]:
    def decorator(obj: Any) -> Any:
        if name in _BUILTINS[kind] or name in _REGISTERED[kind]:
            raise ConfigError(f"a {kind} named {name!r} is already registered")
        _REGISTERED[kind][name] = obj
        return obj

    return decorator


def register_framework(name: str):
    return _register("framework", name)


def register_benchmark(name: str):
    return _register("benchmark", name)


def register_tool(name: str):
    return _register("tool", name)


def register_backend(name: str):
    return _register("backend", name)


def names(kind: str) -> list[str]:
    return sorted(set(_BUILTINS[kind]) | set(_REGISTERED[kind]))


def get(kind: str, name: str) -> Any:
    if name in _REGISTERED[kind]:
        return _REGISTERED[kind][name]
    if name not in _BUILTINS[kind]:
        raise ConfigError(f"unknown {kind} {name!r}; known: {', '.join(names(kind))}")
    module_name, attr = _BUILTINS[kind][name].split(":")
    return getattr(importlib.import_module(module_name), attr)
