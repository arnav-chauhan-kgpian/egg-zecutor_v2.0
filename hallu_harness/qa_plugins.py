"""Question-answering plug-ins added beside the built-ins and `plugins.py`,
registered with the registry's decorators. `cli.py` imports this module;
tests that build a `Runner` directly import it themselves.

* benchmark `gaia` (benchmarks/gaia/);
* frameworks `smolagents_qa` and `qwen_agent_qa` (frameworks/qa/): the
  coding adapters with prompts for answering a question;
* tools `gaia_web_search` (shown to the model as `web_search`),
  `visit_webpage`, `find_in_page`, `read_file` and `python`.

Nothing here imports smolagents, Qwen-Agent, ddgs, openpyxl or requests:
those are imported when a run uses them, as the built-ins do.

`levels`, `attachments` and `solvable_only` are added to the selection keys
the experiment key leaves out (`core/trajectory.py`, `_SELECTION_ONLY`):
like `limit`, they only choose which tasks run, so a run widened from level
1 to levels 1-2 resumes the level-1 trajectories it has.
"""
from __future__ import annotations

from .benchmarks.gaia.benchmark import GaiaBenchmark
from .core import registry, trajectory
from .core.errors import ConfigError
from .frameworks.qa.qwen import QwenAgentQAAdapter
from .frameworks.qa.smol import SmolagentsQAAdapter
from .tools.gaia_search import REGISTRY_NAME as GAIA_SEARCH, GaiaWebSearchTool
from .tools.page_tools import FindInPageTool, ReadFileTool, VisitWebpageTool
from .tools.python_run import PythonTool

registry.register_benchmark(GaiaBenchmark.name)(GaiaBenchmark)
for _adapter in (SmolagentsQAAdapter, QwenAgentQAAdapter):
    registry.register_framework(_adapter.name)(_adapter)
registry.register_tool(GAIA_SEARCH)(GaiaWebSearchTool)
for _tool in (VisitWebpageTool, FindInPageTool, ReadFileTool, PythonTool):
    registry.register_tool(_tool.name)(_tool)

trajectory._SELECTION_ONLY.update({"levels", "attachments", "solvable_only"})

QA_FRAMEWORKS = {SmolagentsQAAdapter.name: "smolagents", QwenAgentQAAdapter.name: "qwen_agent"}


def check_config(cfg) -> None:
    """Checks a resolved config cannot make on its own, run by `run` and
    `check` before anything loads. Raises `ConfigError`.

    * A question-answering benchmark runs only under a QA framework, and a QA
      framework only on such a benchmark: the coding prompts send code to
      grade_code, and the QA prompts ask for an answer.
    * GAIA task ids in the selection exist (this reads the dataset's metadata,
      downloading it on first use).
    * smolagents' extra "answer from memory" call after the budget is off:
      the instructions say the task ends without an answer when the replies
      are used up.
    """
    qa_benchmark = getattr(registry.get("benchmark", cfg.dataset.benchmark), "qa", False)
    if qa_benchmark and cfg.framework not in QA_FRAMEWORKS:
        hint = next((f" (--set framework={v})" for v, base in QA_FRAMEWORKS.items() if base == cfg.framework), "")
        raise ConfigError(
            f"benchmark {cfg.dataset.benchmark!r} asks questions to answer, but framework {cfg.framework!r} is built around "
            f"grading code; use one of {', '.join(sorted(QA_FRAMEWORKS))}{hint}"
        )
    if cfg.framework in QA_FRAMEWORKS and not qa_benchmark:
        raise ConfigError(f"framework {cfg.framework!r} is for question-answering benchmarks; run {cfg.dataset.benchmark!r} "
                          f"under {QA_FRAMEWORKS[cfg.framework]!r}")
    if cfg.dataset.benchmark == GaiaBenchmark.name and cfg.dataset.selection.get("ids"):
        # Unknown ids fail here, before a run directory is created or a model
        # loads (`load` would raise them only once the run has started).
        GaiaBenchmark(cfg.dataset).rows()
    if cfg.framework == SmolagentsQAAdapter.name and cfg.params.get("answer_from_memory_on_budget"):
        raise ConfigError("smolagents_qa: answer_from_memory_on_budget must be false; the instructions say the task ends "
                          "without an answer once the replies are used up")
