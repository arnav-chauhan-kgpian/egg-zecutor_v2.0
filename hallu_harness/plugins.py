"""Plug-ins added beside the built-ins, registered with the registry's
decorators. `cli.py` imports this module; tests that build a `Runner`
directly import it themselves.

* benchmark `halogen_code_imports` (benchmarks/halogen_code_imports/);
* frameworks `smolagents_ungraded` and `qwen_agent_ungraded`
  (frameworks/ungraded/): the coding adapters with prompts that name no
  grading tool;
* tools `web_search`, `fetch_webpage`, `github_repository_search`,
  `github_code_search`, `github_get_file`.

Nothing here imports smolagents, Qwen-Agent, ddgs or requests: those are
imported when a run uses them, as the built-ins do.

`num_examples` is added to the selection keys the experiment key leaves out
(`core/trajectory.py`, `_SELECTION_ONLY`): like `limit`, it only chooses which
tasks run, so a run widened from 100 to 200 examples resumes the 100 it has.
"""
from __future__ import annotations

from .benchmarks.halogen_code_imports.benchmark import HalogenCodeImportsBenchmark
from .core import registry, trajectory
from .core.errors import ConfigError
from .frameworks.ungraded.qwen import QwenAgentUngradedAdapter
from .frameworks.ungraded.smol import SmolagentsUngradedAdapter
from .tools.fetch_webpage import FetchWebpageTool
from .tools.github_tools import GithubCodeSearchTool, GithubGetFileTool, GithubRepositorySearchTool
from .tools.web_search import WebSearchTool

registry.register_benchmark(HalogenCodeImportsBenchmark.name)(HalogenCodeImportsBenchmark)
for _adapter in (SmolagentsUngradedAdapter, QwenAgentUngradedAdapter):
    registry.register_framework(_adapter.name)(_adapter)
for _tool in (WebSearchTool, FetchWebpageTool, GithubRepositorySearchTool, GithubCodeSearchTool, GithubGetFileTool):
    registry.register_tool(_tool.name)(_tool)

trajectory._SELECTION_ONLY.add("num_examples")

UNGRADED_FRAMEWORKS = {SmolagentsUngradedAdapter.name: "smolagents", QwenAgentUngradedAdapter.name: "qwen_agent"}


def check_config(cfg) -> list[str]:
    """Checks a resolved config cannot make on its own, run by `run` and
    `check` before anything loads. Raises `ConfigError`; returns warnings.

    * A benchmark without a grading tool runs only under an ungraded
      framework, and an ungraded framework only on such a benchmark: the
      coding prompts name grade_code, and the ungraded ones do not.
    * A tool that needs something from the environment (a GitHub token) says
      so now, not on its first call in the middle of a sweep.
    """
    ungraded_benchmark = getattr(registry.get("benchmark", cfg.dataset.benchmark), "ungraded", False)
    if ungraded_benchmark and cfg.framework not in UNGRADED_FRAMEWORKS:
        hint = next((f" (--set framework={v})" for v, base in UNGRADED_FRAMEWORKS.items() if base == cfg.framework), "")
        raise ConfigError(
            f"benchmark {cfg.dataset.benchmark!r} has no grading tool, but framework {cfg.framework!r} is built around one "
            f"(its prompts send code to grade_code); use one of {', '.join(sorted(UNGRADED_FRAMEWORKS))}{hint}"
        )
    if cfg.framework in UNGRADED_FRAMEWORKS and not ungraded_benchmark:
        raise ConfigError(
            f"framework {cfg.framework!r} is for benchmarks without a grading tool; run {cfg.dataset.benchmark!r} under "
            f"{UNGRADED_FRAMEWORKS[cfg.framework]!r}"
        )
    warnings = []
    for name in cfg.tool_names:
        check = getattr(registry.get("tool", name), "check_environment", None)
        if check is not None:
            warning = check(cfg.dataset.tools.params.get(name, {}))
            if warning:
                warnings.append(warning)
    return warnings
