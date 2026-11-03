"""Replacing a module's globals for the duration of a run.

The coding adapters read their prompt pieces as module globals at call time
(`harmonized_system_prompt` in `smol/adapter.py`; `REPLY_SECTION`,
`FORM_SECTION` and `JSON_ENCODING_SECTION` in `qwenagent/adapter.py`), so
setting those names while `run` executes changes what the model is shown and
nothing else. The repo already sets Qwen-Agent's `MAX_LLM_CALL_PER_RUN` this
way (`_budget()` in `qwenagent/adapter.py`).

* A name the module no longer defines is an error, so a rename upstream
  fails loudly instead of silently showing the coding prompt.
* The originals are restored in `finally`, whatever happens in the run.
* Trajectories run one at a time; a second swap of the same module while one
  is active is an error rather than a race.
"""
from __future__ import annotations

import contextlib
from types import ModuleType

_ACTIVE: set[str] = set()


@contextlib.contextmanager
def swap_globals(module: ModuleType, **values):
    missing = sorted(name for name in values if not hasattr(module, name))
    if missing:
        raise RuntimeError(f"{module.__name__} no longer defines {', '.join(missing)}: the ungraded variant cannot "
                           "replace its coding prompt")
    if module.__name__ in _ACTIVE:
        raise RuntimeError(f"{module.__name__}'s globals are already swapped by another run")
    saved = {name: getattr(module, name) for name in values}
    _ACTIVE.add(module.__name__)
    try:
        for name, value in values.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)
        _ACTIVE.discard(module.__name__)


def require_ungraded_benchmark(benchmark, framework: str) -> None:
    if not getattr(benchmark, "ungraded", False):
        raise RuntimeError(f"{framework} runs code tasks without a grading tool; benchmark {benchmark.name!r} has one, so "
                           "run it under the coding framework instead")
