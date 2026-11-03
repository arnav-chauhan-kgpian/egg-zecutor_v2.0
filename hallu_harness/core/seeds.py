"""Deterministic seeds.

Every trajectory gets a seed derived from `(run seed, task id, sample index)`
and every model call within it one derived from `(trajectory seed, call
index)`. The same inputs give the same seeds on any machine and in any task
order, so a resumed or re-sharded sweep samples exactly what an
uninterrupted one would have.
"""
from __future__ import annotations

import hashlib


def derive_seed(*parts: object) -> int:
    """A 31-bit seed from any sequence of printable parts."""
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
