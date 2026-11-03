"""GAIA (Mialon et al., 2023): general-assistant questions with a short,
unambiguous answer, answered with web, file and Python tools.

* **Tasks.** The validation split of `gaia-benchmark/GAIA` at a pinned
  commit (`data.py`), in file order; the test split's answers are not
  public. By default only questions the text tools can serve are run
  (`capabilities.py`: 114 of 165 on 2026-10-29); `selection` narrows them by
  id, level and attachment, and `limit` takes the first n.
* **No grading tool.** Nothing is graded during the task. Only the QA
  framework variants (`smolagents_qa`, `qwen_agent_qa`) run it: the coding
  adapters' prompts name `grade_code` (`qa_plugins.check_config`, and
  `make_env` for runs that bypass the CLI).
* **The grade** is GAIA's own scorer on the answer the protocol defines
  (`answers.py`, `scorer.py`), plus grounding labels (`labels.py`). The
  reference answer is never written into the trajectory, only its SHA-256:
  GAIA's terms forbid resharing the answers, and trajectories get exported.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...core.config import DatasetConfig
from ...core.errors import ConfigError
from ...core.qa_protocol import QA_PROMPT_REVISION, QA_PROTOCOLS, QAProtocol
from ...core.types import FrameworkResult, Task
from ...tools.documents import Attachment
from ..base import Benchmark
from . import answers, capabilities, labels
from .data import REPO, REVISION, SPLIT, load_rows
from .scorer import answer_type, question_scorer
from .texts import GaiaTexts

# Tools that run or grade code against tests: the instructions say nothing is
# graded during the task. (`python` is the QA sandbox and is fine.)
CODE_TOOLS = {"grade_code", "run_visible_tests", "python_exec", "submit"}


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[str] | None = None
    levels: list[Literal[1, 2, 3]] | None = None
    attachments: Literal["any", "with", "without"] = "any"
    solvable_only: bool = True  # only questions the text tools can serve (capabilities.py)
    limit: int | None = Field(None, gt=0)
    revision: str = REVISION  # in the experiment key: a different snapshot is a different experiment


class Evaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scorer: Literal["gaia_official"] = "gaia_official"


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_date_note: bool = True  # "The question was written in 2023. Pages may have changed ..."


@dataclass
class GaiaEnv:
    """What the tools and the instructions see of a trajectory."""

    task: Task
    texts: GaiaTexts
    protocol: QAProtocol
    attachment: Attachment | None
    tool_params: dict  # every tool's params, so find_in_page pages a document as the reading tools do
    documents: object = None  # the trajectory's DocumentStore, created by the first tool that reads
    trace: list[dict] = field(default_factory=list)  # every tool call, filled in by the QA adapters for the labels


class GaiaBenchmark(Benchmark):
    name = "gaia"
    qa = True  # answered, not graded: `qa_plugins.check_config` pairs it with the QA frameworks
    Selection = Selection
    Evaluation = Evaluation
    Prompt = Prompt

    @classmethod
    def validate_config(cls, cfg: DatasetConfig) -> None:
        super().validate_config(cfg)
        if (cfg.split or SPLIT) != SPLIT:
            raise ConfigError(f"dataset split ({cls.name}): only {SPLIT!r} has public answers; got {cfg.split!r}")
        listed = set(cfg.tools.required) | set(cfg.tools.optional)
        if listed & CODE_TOOLS:
            raise ConfigError(f"dataset tools ({cls.name}): {', '.join(sorted(listed & CODE_TOOLS))} grade code, but the "
                              "instructions tell the model that nothing is graded during the task")
        if cfg.tools.optional:
            raise ConfigError(f"dataset tools ({cls.name}): list every tool under tools.required; the instructions name "
                              "what the run's tools can do, so the tool list must be fixed by the dataset")
        if "web_search" in listed and "gaia_web_search" in listed:
            raise ConfigError(f"dataset tools ({cls.name}): web_search and gaia_web_search both show the model a tool "
                              "named web_search; use gaia_web_search")

    def make_texts(self) -> GaiaTexts:
        return GaiaTexts(tool_names=[self.model_facing(n) for n in self.cfg.tools.required],
                         question_date_note=self.prompt.question_date_note)

    @staticmethod
    def model_facing(registry_name: str) -> str:
        from ...core import registry

        return getattr(registry.get("tool", registry_name), "name", registry_name)

    # -- tasks ----------------------------------------------------------------

    def rows(self) -> list[dict]:
        rows = list(load_rows(self.selection.revision, SPLIT))
        by_id = {r["task_id"]: r for r in rows}
        if self.selection.ids:
            unknown = [i for i in self.selection.ids if i not in by_id]
            if unknown:
                raise ConfigError(f"unknown {self.name} task id(s): {', '.join(unknown)}")
            rows = [by_id[i] for i in self.selection.ids]
        out = []
        for row in rows:
            if self.selection.levels and row["level"] not in self.selection.levels:
                continue
            has_file = bool(row["file_name"])
            if (self.selection.attachments == "with" and not has_file) or (self.selection.attachments == "without" and has_file):
                continue
            if self.selection.solvable_only:
                verdict = capabilities.assess(row)
                if not verdict["solvable"]:
                    if self.selection.ids:  # asked for by id: say why it is not run
                        print(f"[skip] {self.name} {row['task_id']}: {'; '.join(verdict['reasons'])}", file=sys.stderr)
                    continue
            out.append(row)
        return out[: self.selection.limit] if self.selection.limit else out

    def load(self) -> Iterator[Task]:
        for row in self.rows():
            verdict = capabilities.assess(row)
            annotator = row["annotator"]
            yield Task(
                task_id=row["task_id"],
                query=row["question"].strip(),
                metadata={
                    "level": row["level"],
                    "file_name": row["file_name"] or None,
                    "needs": verdict["needs"],
                    "solvable_by_rules": verdict["solvable"],
                    "annotator_steps": annotator.get("Number of steps"),
                    "annotator_tools": annotator.get("Tools"),
                    "answer_type": answer_type(row["reference"]),
                    "source": f"{REPO}@{self.selection.revision} {SPLIT}",
                },
                payload=row,
            )

    def make_evaluator(self, task: Task, grade_params: dict):
        raise NotImplementedError("gaia has no grading tool and no evaluator")

    def make_env(self, task: Task, protocol, tool_params: dict) -> GaiaEnv:
        if getattr(protocol, "name", None) not in QA_PROTOCOLS:
            raise ConfigError(f"{self.name} runs only under the QA frameworks (protocols {', '.join(QA_PROTOCOLS)}); this run "
                              f"uses {getattr(protocol, 'name', protocol)!r}, whose prompts name a grading tool")
        row = task.payload
        attachment = Attachment(row["file_name"], row["file_path"]) if row["file_name"] else None
        if attachment is not None and not attachment.path.exists():
            raise FileNotFoundError(f"attachment {attachment.path} is missing from the GAIA snapshot")
        return GaiaEnv(task=task, texts=self.texts, protocol=protocol, attachment=attachment, tool_params=dict(tool_params))

    # -- provenance -------------------------------------------------------------

    @property
    def grading_inputs(self) -> dict:
        return {
            "gaia_revision": self.selection.revision,
            "capability_rules_version": capabilities.RULES_VERSION,
            "capability_overrides_sha256": hashlib.sha256(capabilities.OVERRIDES_PATH.read_bytes()).hexdigest(),
            # Not grading, but the only per-benchmark entry of the experiment
            # key: the revision of this path's model-facing text.
            "qa_prompt_revision": QA_PROMPT_REVISION,
        }

    @property
    def info(self) -> dict:
        return {"repo": REPO, "revision": self.selection.revision, "split": SPLIT, **self.grading_inputs}

    # -- the final grade --------------------------------------------------------

    def final_grade(self, env: GaiaEnv, result: FrameworkResult) -> dict:
        row = env.task.payload
        reference = row["reference"]
        extraction = answers.extract(result.final_answer, env.protocol.name)
        correct = bool(extraction.answer is not None and not extraction.abstained and question_scorer(extraction.answer, reference))
        grade = {
            "scorer": "gaia_official",
            "answer_type": answer_type(reference),
            "model_answer": extraction.answer,
            "answer_found": extraction.found,
            "abstained": extraction.abstained,
            "extraction": extraction.method,
            "correct": correct,
            "reference_sha256": hashlib.sha256(reference.encode("utf-8")).hexdigest(),
            "labels": labels.compute(row["question"], reference, extraction.answer, env.trace),
        }
        return {
            "final_solution": extraction.answer,
            "solution_source": "final_answer" if extraction.answer is not None else "none",
            "grade": grade,
            "solved": correct,
        }
