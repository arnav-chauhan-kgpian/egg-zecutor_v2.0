"""HALoGEN's code-packages subset: write a program for a Stack Overflow
question; afterwards, check that every module it imports exists.

HALoGEN (Ravichander et al., ACL 2025) gives each model one turn per prompt
and checks the names its replies import against PyPI. Here the model works as
an agent, with whatever tools the run gives it (web search, page fetching,
GitHub search and file reading in the preset; none for a single-turn
baseline), and ends with a final program.

* **Tasks.** The 1085 prompts in file order, with ids `code-0000` ...
  `code-1084` (the row index). `selection.num_examples = x` takes x of them
  evenly spaced: rows `k * total // x` for k < x, so x = 100 is every 10.85th
  row and every row of x is also a row of 2x. It is applied after `ids`.
* **No grading tool.** Nothing is run, so the model is never shown a result;
  only the `*_ungraded` framework variants can run this benchmark, because the
  coding adapters' prompts name `grade_code` (`plugins.check_config`, and
  `make_env` below for runs that bypass the CLI).
* **The grade** (`final_grade`) is the import check of the final program,
  per module (`imports.py`), plus HALoGEN's own unit extraction over the
  whole final reply for comparison, and the names after any `pip install`.
  `solved` means a program was given and every module it imports resolved:
  HALoGEN's per-response utility of 1.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field

from ...core.config import DatasetConfig, resolve_repo_path
from ...core.errors import ConfigError
from ...core.protocol import Protocol
from ...core.types import FrameworkResult, Task
from ...core.ungraded_protocol import UNGRADED_PROMPT_REVISION, UNGRADED_PROTOCOLS
from ..base import Benchmark, extract_code, is_sorry
from .data import IMPORT_MAP_PATH, PYPI_NAMES_PATH, load_import_map, load_prompts, load_pypi_names, prompts_info
from .imports import Resolver, extract_imports, halogen_units, pip_install_names
from .texts import HalogenTexts

# Tools that run or grade code. The instructions say the program is not run
# during the task, which is true only without them.
CODE_TOOLS = {"grade_code", "run_visible_tests", "python_exec", "submit"}


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[str] | None = None
    num_examples: int | None = Field(None, gt=0)  # None = every task


class Evaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pypi_names: str = str(PYPI_NAMES_PATH)
    import_map: str = str(IMPORT_MAP_PATH)


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    html_unescape: bool = True


def task_id(index: int) -> str:
    return f"code-{index:04d}"


def strided(indices: list[int], count: int | None) -> list[int]:
    """`count` of `indices`, evenly spaced: positions k * len // count."""
    if count is None or count >= len(indices):
        return list(indices)
    return [indices[k * len(indices) // count] for k in range(count)]


@dataclass
class HalogenEnv:
    """What the tools and the instructions see of a trajectory. The web and
    GitHub tools keep their own state and do not use it."""

    task: Task
    texts: HalogenTexts
    protocol: Protocol


class HalogenCodeImportsBenchmark(Benchmark):
    name = "halogen_code_imports"
    ungraded = True  # no grading tool: `plugins.check_config` pairs it with the *_ungraded frameworks
    Selection = Selection
    Evaluation = Evaluation
    Prompt = Prompt

    @classmethod
    def validate_config(cls, cfg: DatasetConfig) -> None:
        super().validate_config(cfg)
        listed = set(cfg.tools.required) | set(cfg.tools.optional)
        if listed & CODE_TOOLS:
            raise ConfigError(
                f"dataset tools ({cls.name}): {', '.join(sorted(listed & CODE_TOOLS))} run or grade code, but the task "
                "instructions tell the model that its program is not run during the task"
            )
        evaluation = cls.Evaluation(**cfg.evaluation)
        load_pypi_names(evaluation.pypi_names)
        load_import_map(evaluation.import_map)

    def make_texts(self) -> HalogenTexts:
        return HalogenTexts(html_unescape=self.prompt.html_unescape)

    def selected_indices(self) -> list[int]:
        prompts = load_prompts()
        if self.selection.ids:
            known = {task_id(i): i for i in range(len(prompts))}
            unknown = [tid for tid in self.selection.ids if tid not in known]
            if unknown:
                raise ConfigError(f"unknown {self.name} task id(s) {', '.join(unknown)} (ids look like 'code-0042')")
            indices = [known[tid] for tid in self.selection.ids]
        else:
            indices = list(range(len(prompts)))
        return strided(indices, self.selection.num_examples)

    def load(self) -> Iterator[Task]:
        prompts = load_prompts()
        for index in self.selected_indices():
            yield Task(
                task_id=task_id(index),
                query=self.texts.render_prompt(prompts[index]),
                metadata={"row": index, "prompt": prompts[index], "source": "HALoGEN code (prompts_code.csv)"},
            )

    def make_evaluator(self, task: Task, grade_params: dict):
        raise NotImplementedError("halogen_code_imports has no grading tool and no evaluator")

    def make_env(self, task: Task, protocol: Protocol, tool_params: dict) -> HalogenEnv:
        if protocol.name not in UNGRADED_PROTOCOLS:
            raise ConfigError(
                f"{self.name} runs only under the *_ungraded frameworks (protocols {', '.join(UNGRADED_PROTOCOLS)}); "
                f"this run uses {protocol.name}, whose prompts name a grading tool"
            )
        return HalogenEnv(task=task, texts=self.texts, protocol=protocol)

    # -- data provenance ------------------------------------------------------

    def _resolver(self) -> tuple[Resolver, str]:
        header, names = load_pypi_names(self.evaluation.pypi_names)
        return Resolver(names, load_import_map(self.evaluation.import_map)["map"]), header

    @property
    def grading_inputs(self) -> dict:
        return {
            "pypi_names_sha256": hashlib.sha256(resolve_repo_path(self.evaluation.pypi_names).read_bytes()).hexdigest(),
            "import_map_sha256": hashlib.sha256(resolve_repo_path(self.evaluation.import_map).read_bytes()).hexdigest(),
            # Not grading, but the only per-benchmark entry of the experiment
            # key: the revision of this path's model-facing text.
            "ungraded_prompt_revision": UNGRADED_PROMPT_REVISION,
        }

    @property
    def info(self) -> dict:
        header, _ = load_pypi_names(self.evaluation.pypi_names)
        return {
            "prompts": prompts_info(),
            "total_tasks": len(load_prompts()),
            "num_examples": self.selection.num_examples,
            "pypi_names": {"file": self.evaluation.pypi_names, "header": header},
            "import_map": {"file": self.evaluation.import_map, "sources": load_import_map(self.evaluation.import_map)["sources"]},
            **self.grading_inputs,
        }

    # -- the final grade ----------------------------------------------------

    def final_grade(self, env: HalogenEnv, result: FrameworkResult) -> dict:
        resolver, _ = self._resolver()
        answer = result.final_answer
        program = extract_code(answer)
        records, parsed = extract_imports(program) if program else ([], False)
        modules = list(dict.fromkeys(r.module for r in records if r.level == 0 and r.module))
        resolved = {m: resolver.module(m) for m in modules}
        unresolved = [m for m in modules if resolved[m]["resolution"] == "unresolved"]

        units = halogen_units(answer or "")
        halogen_missing = [u for u in units if not (resolver.module(u)["stdlib"] or resolver.on_pypi(u))]
        pip_names = pip_install_names(answer or "")
        grade = {
            "checker": "imports",
            "abstained": is_sorry(answer),
            "program_found": program is not None,
            "program_parses": parsed if program is not None else None,
            "imports": [r.to_dict() for r in records],
            "modules": [resolved[m] for m in modules],
            "unresolved_modules": unresolved,
            "unresolved_fraction": round(len(unresolved) / len(modules), 4) if modules else 0.0,
            # HALoGEN's rule, offline: its regexes over the whole reply; a unit
            # counts as missing unless the PyPI snapshot lists the name (HALoGEN:
            # PyPI's JSON API answers 200) or it is in the standard library
            # (HALoGEN: docs.python.org has a page for it). The snapshot also
            # lists names without releases (cv2, yaml), which HALoGEN's live
            # lookups reported missing, so this undercounts HALoGEN's rate.
            "halogen": {
                "units": units,
                "not_found": halogen_missing,
                "not_found_rate": round(len(halogen_missing) / len(units), 4) if units else 0.0,
            },
            "pip_install": [{"name": n, "on_pypi": resolver.on_pypi(n)} for n in pip_names],
        }
        return {
            "final_solution": program,
            "solution_source": "final_answer_code" if program is not None else "none",
            "grade": grade,
            "solved": program is not None and not unresolved,
        }
