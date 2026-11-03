"""Token-level bookkeeping: byte offsets of tokens, and segment spans.

Segments label token ranges by role (DESIGN.md §11.4) so that activations,
log-probs or anything else indexed by token can later be selected by meaning:
"the feedback tokens", "the last token before the reply", "the tool call".

Input segments come from the chat messages: each message's tokens, then the
generation prompt. They are found by rendering the conversation one message
at a time and checking that each rendering is a prefix of the full prompt —
true for append-only templates. When a template is not prefix-stable (e.g.
one that rewrites earlier turns), ChatML turn markers in the token ids are
used instead; when neither works, segments are recorded as `null` rather than
guessed.

Output segments come from the generated text: `think`, `tool_call`, `code`
and `text`, located on the exact bytes of the generated tokens so that a
token boundary falling inside a multi-byte character cannot shift them.
"""
from __future__ import annotations

import bisect
import re
from functools import lru_cache


@lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    """GPT-2's byte-to-unicode table, inverted (byte-level BPE)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_bytes(tokenizer, ids: list[int]) -> list[bytes] | None:
    """The exact bytes each token contributes to `decode(ids)`.

    Works for byte-level BPE tokenizers (Qwen, GPT-2 family, SmolLM2): normal
    tokens map through the byte decoder, added tokens contribute their literal
    content. Returns None for any other tokenizer, or if the concatenation
    does not reproduce `decode(ids, skip_special_tokens=False)` exactly.
    """
    if not ids:
        return []
    decoder = _byte_decoder()
    added = {tid: tok.content for tid, tok in getattr(tokenizer, "added_tokens_decoder", {}).items()}
    pieces: list[bytes] = []
    for tid, tok in zip(ids, tokenizer.convert_ids_to_tokens(ids)):
        if tid in added:
            pieces.append(added[tid].encode("utf-8"))
            continue
        if tok is None:
            return None
        try:
            pieces.append(bytes(decoder[ch] for ch in tok))
        except KeyError:
            return None  # not byte-level BPE
    joined = b"".join(pieces).decode("utf-8", errors="replace")
    if joined != tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return None
    return pieces


def _offsets(pieces: list[bytes]) -> list[int]:
    """Start byte offset of each token, plus the total length at the end."""
    out = [0]
    for piece in pieces:
        out.append(out[-1] + len(piece))
    return out


def _token_at(offsets: list[int], byte_pos: int) -> int:
    """Index of the token whose byte range contains `byte_pos` (clamped)."""
    index = bisect.bisect_right(offsets, byte_pos) - 1
    return max(0, min(index, len(offsets) - 2))


# ---------------------------------------------------------------- input side


def _kind_for(role: str, user_seen: int) -> str:
    if role == "system":
        return "system"
    if role == "user":
        return "task" if user_seen == 0 else "observation"
    if role in ("tool", "function", "tool-response", "ipython"):
        return "observation"
    if role == "assistant":
        return "assistant"
    return role


def input_segments(tokenizer, messages: list[dict], tools, template_kwargs: dict, full_text: str, ids: list[int]) -> list[dict] | None:
    """Token spans of each message and of the generation prompt."""
    pieces = token_bytes(tokenizer, ids)
    if pieces is None:
        return None
    offsets = _offsets(pieces)
    full_bytes = full_text.encode("utf-8")

    boundaries: list[int] = []  # byte offset where each message ends
    try:
        for i in range(1, len(messages) + 1):
            prefix = tokenizer.apply_chat_template(
                messages[:i], tools=tools, add_generation_prompt=False, tokenize=False, **template_kwargs
            )
            prefix_bytes = prefix.encode("utf-8")
            if not full_bytes.startswith(prefix_bytes):
                raise ValueError("template is not prefix-stable")
            boundaries.append(len(prefix_bytes))
    except Exception:  # noqa: BLE001 - any failure falls back to turn markers
        return _chatml_segments(tokenizer, messages, ids)

    spans: list[dict] = []
    start_byte = 0
    user_seen = 0
    for message, end_byte in zip(messages, boundaries):
        if end_byte not in offsets:
            return _chatml_segments(tokenizer, messages, ids)  # boundary inside a token
        kind = _kind_for(message.get("role", ""), user_seen)
        if message.get("role") == "user":
            user_seen += 1
        spans.append({"kind": kind, "where": "input", "start": offsets.index(start_byte), "end": offsets.index(end_byte)})
        start_byte = end_byte
    spans.append({"kind": "generation_prompt", "where": "input", "start": offsets.index(start_byte), "end": len(ids)})
    return [s for s in spans if s["end"] > s["start"] or s["kind"] == "generation_prompt"]


def _chatml_segments(tokenizer, messages: list[dict], ids: list[int]) -> list[dict] | None:
    """Fallback: split on `<|im_start|>` tokens (ChatML templates only)."""
    start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if start_id is None or start_id == getattr(tokenizer, "unk_token_id", None):
        return None
    starts = [i for i, t in enumerate(ids) if t == start_id]
    if not starts:
        return None
    if starts[0] != 0:
        starts[0] = 0  # anything before the first turn belongs to it
    roles = []
    for begin in starts:
        roles.append(tokenizer.decode(ids[begin + 1 : begin + 4], skip_special_tokens=False).split("\n")[0].strip())
    spans: list[dict] = []
    user_seen = 0
    for n, begin in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(ids)
        role = roles[n]
        last = n == len(starts) - 1
        if last and role == "assistant":
            kind = "generation_prompt"
        else:
            kind = _kind_for(role, user_seen)
            if role == "user":
                user_seen += 1
        spans.append({"kind": kind, "where": "input", "start": begin, "end": end})
    return spans


# ---------------------------------------------------------------- output side

_THINK = re.compile(rb"<think>.*?(</think>|\Z)", re.DOTALL)
_TOOL_CALL = re.compile(rb"<tool_call>.*?(</tool_call>|\Z)", re.DOTALL)
_ACTION = re.compile(rb"Action:\s*(\{.*\})", re.DOTALL)
_FENCE = re.compile(rb"```[^\n`]*\n.*?(```|\Z)", re.DOTALL)


def output_segments(tokenizer, ids: list[int]) -> list[dict] | None:
    """Spans of `think`, `tool_call`, `code` and `text` in a generated reply.

    Priority, for overlapping matches: think > tool_call > code. Everything
    not covered is `text`, so the spans tile the output exactly.
    """
    pieces = token_bytes(tokenizer, ids)
    if pieces is None:
        return None
    if not ids:
        return []
    offsets = _offsets(pieces)
    data = b"".join(pieces)

    claimed: list[tuple[int, int, str]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e, _ in claimed)

    for m in _THINK.finditer(data):
        claimed.append((m.start(), m.end(), "think"))
    for m in _TOOL_CALL.finditer(data):
        if free(m.start(), m.end()):
            claimed.append((m.start(), m.end(), "tool_call"))
    if not any(k == "tool_call" for _, _, k in claimed):
        m = _ACTION.search(data)
        if m and free(m.start(1), m.end(1)):
            claimed.append((m.start(1), m.end(1), "tool_call"))
    for m in _FENCE.finditer(data):
        if free(m.start(), m.end()):
            claimed.append((m.start(), m.end(), "code"))
    claimed.sort()

    # Map byte ranges to token ranges; a token belongs to the span containing its first byte.
    spans: list[dict] = []
    token_kind = ["text"] * len(ids)
    for start, end, kind in claimed:
        first = _token_at(offsets, start)
        last = _token_at(offsets, max(start, end - 1))
        for t in range(first, last + 1):
            token_kind[t] = kind
    run_start = 0
    for t in range(1, len(ids) + 1):
        if t == len(ids) or token_kind[t] != token_kind[run_start]:
            spans.append({"kind": token_kind[run_start], "where": "output", "start": run_start, "end": t})
            run_start = t
    return spans
