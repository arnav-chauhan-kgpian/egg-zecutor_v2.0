"""GAIA's validation split from the gated `gaia-benchmark/GAIA` dataset, at a
pinned commit.

Only validation has public answers (the test split's are "?"), so it is the
only split this benchmark serves. The dataset is gated: the machine needs a
Hugging Face login (`hf auth login`) for an account that accepted GAIA's
terms. The terms forbid resharing the questions and answers in a crawlable
form, so nothing here writes them to the repository; the snapshot stays in
the Hugging Face cache.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

from ...core.errors import ConfigError

REPO = "gaia-benchmark/GAIA"
REVISION = "682dd723ee1e1697e00360edccf2366dc8418dd9"  # 2026-10-29; the October 2025 parquet layout
SPLIT = "validation"


@lru_cache(maxsize=4)
def snapshot_dir(revision: str = REVISION, split: str = SPLIT) -> Path:
    """The split's directory (metadata.parquet and attachments), downloaded
    into the Hugging Face cache on first use."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        root = snapshot_download(REPO, repo_type="dataset", revision=revision, allow_patterns=[f"2023/{split}/*"])
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        raise ConfigError(
            f"cannot read the gated dataset {REPO}: log in with `hf auth login` as an account that accepted GAIA's "
            f"terms on https://huggingface.co/datasets/{REPO} ({type(exc).__name__})"
        ) from None
    path = Path(root) / "2023" / split
    if not (path / "metadata.parquet").exists():
        raise ConfigError(f"{REPO}@{revision} has no 2023/{split}/metadata.parquet")
    return path


def _annotator(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


@lru_cache(maxsize=4)
def load_rows(revision: str = REVISION, split: str = SPLIT) -> tuple[dict, ...]:
    """One dict per question, in file order: task_id, question, level (int),
    reference, file_name ("" if none), file_path (absolute, or None),
    annotator (the Annotator Metadata dict)."""
    import pandas as pd

    directory = snapshot_dir(revision, split)
    frame = pd.read_parquet(directory / "metadata.parquet")
    rows = []
    for record in frame.to_dict("records"):
        name = record.get("file_name") or ""
        rows.append({
            "task_id": str(record["task_id"]),
            "question": str(record["Question"]),
            "level": int(record["Level"]),
            "reference": str(record["Final answer"]),
            "file_name": name,
            "file_path": (directory / name) if name else None,
            "annotator": _annotator(record.get("Annotator Metadata")),
        })
    return tuple(rows)
