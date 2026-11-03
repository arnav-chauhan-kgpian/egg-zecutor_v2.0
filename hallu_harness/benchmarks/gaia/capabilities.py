"""Which GAIA questions the text tools can serve.

The tools read text only: web pages, PDFs, spreadsheets and text files, plus
a Python sandbox (`page_tools.py`, `python_run.py`). A question is
*solvable* here when nothing it needs is outside that:

* its attachment, if any, is a type `read_file` shows (`FILE_SUFFIXES`:
  .xlsx, .pdf, and text files); images, audio, Word, PowerPoint and zip
  files are not;
* the annotator's tool list names no image, audio or video capability
  (image recognition, OCR, colour recognition, speech-to-text, video, ...);
* the question links no YouTube video.

These rules read the annotators' free text, so `capability_overrides.json`
records hand-checked exceptions with the reason for each (task ids only; no
question text, which GAIA's terms keep out of the repository). Bump
`RULES_VERSION` when a rule changes: it is part of the experiment key.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from ...tools.documents import FILE_SUFFIXES

RULES_VERSION = 1
OVERRIDES_PATH = Path(__file__).with_name("capability_overrides.json")

MODALITIES = {
    "image": re.compile(r"image recognition|image processing|image search|\bocr\b|colou?r recognition|computer vision|"
                        r"gif parsing|extract text from images|rubik", re.IGNORECASE),
    "audio": re.compile(r"speech[- ]to[- ]text|audio", re.IGNORECASE),
    "video": re.compile(r"video|youtube", re.IGNORECASE),
}
ATTACHMENT_MODALITY = {".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image", ".mp3": "audio", ".m4a": "audio",
                       ".wav": "audio", ".mov": "video", ".mp4": "video"}
YOUTUBE = re.compile(r"youtube\.com|youtu\.be", re.IGNORECASE)


@lru_cache(maxsize=1)
def load_overrides(path: str = str(OVERRIDES_PATH)) -> dict:
    data = json.loads(Path(path).read_text())
    return data.get("overrides", {})


def assess(row: dict, overrides: dict | None = None) -> dict:
    """{"solvable": bool, "needs": [...], "reasons": [...], "override": reason or None}."""
    needs, reasons = set(), []
    tools_text = str(row.get("annotator", {}).get("Tools", ""))
    for modality, pattern in MODALITIES.items():
        if pattern.search(tools_text):
            needs.add(modality)
            reasons.append(f"the annotator used {modality} tools")
    suffix = Path(row["file_name"]).suffix.lower() if row.get("file_name") else ""
    if suffix:
        if suffix in ATTACHMENT_MODALITY:
            needs.add(ATTACHMENT_MODALITY[suffix])
        if suffix not in FILE_SUFFIXES:
            reasons.append(f"its attachment is a {suffix} file, which read_file cannot show")
    if YOUTUBE.search(row["question"]):
        needs.add("video")
        reasons.append("it links a YouTube video")
    solvable = not reasons
    override = (overrides if overrides is not None else load_overrides()).get(row["task_id"])
    if override is not None:
        solvable = bool(override["solvable"])
        reasons = [override["reason"]] if not solvable else []
    return {"solvable": solvable, "needs": sorted(needs), "reasons": reasons,
            "attachment_suffix": suffix or None, "override": override["reason"] if override else None}
