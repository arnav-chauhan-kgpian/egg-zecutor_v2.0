"""`python -m hallu_harness gaia-data`: download the pinned validation split
and say which questions the text tools can serve, and why the others are
left out. Prints counts and task ids only, never question text."""
from __future__ import annotations

import collections

from . import capabilities
from .data import REPO, REVISION, SPLIT, load_rows, snapshot_dir


def main(revision: str | None = None) -> int:
    revision = revision or REVISION
    rows = load_rows(revision, SPLIT)
    print(f"{REPO}@{revision} {SPLIT}: {len(rows)} questions in {snapshot_dir(revision, SPLIT)}")
    solvable = collections.Counter()
    reasons = collections.Counter()
    for row in rows:
        verdict = capabilities.assess(row)
        if verdict["solvable"]:
            solvable[row["level"]] += 1
        else:
            for reason in verdict["reasons"]:
                reasons[reason if verdict["override"] is None else "hand-checked: " + reason[:60]] += 1
    total = sum(solvable.values())
    print(f"served by the text tools: {total} (level 1: {solvable[1]}, level 2: {solvable[2]}, level 3: {solvable[3]})")
    print(f"left out: {len(rows) - total}")
    for reason, count in reasons.most_common():
        print(f"  {count:3d}  {reason}")
    return 0
