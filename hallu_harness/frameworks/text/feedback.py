"""What the text protocol tells the model about the form of a reply.

cf_agent's text-loop messages (`cf_agent/agent/prompts.py`: no code block,
a reply cut off while thinking or inside the block, several blocks), with two
changes so that every sentence is true of the run:

* the dataset's noun for what the model writes ("program" for Codeforces,
  "function" for HumanEval) instead of "program" everywhere;
* the closing request follows the state of the task. Until a grading result
  allows finishing, the model is asked for its complete code; once one does
  (a verified solution, a pass with hidden tests off, an unverifiable
  result), finishing is offered as well, as the task instructions say.
"""
from __future__ import annotations

from ...benchmarks.base import TaskEnv


def may_finish(env: TaskEnv) -> bool:
    """Whether a grading result has told the model that it may finish
    ("Ending the task" in the task instructions)."""
    if env.accepted_code is not None:
        return True
    for call in env.calls:
        if not call.ran:
            continue
        if call.hidden is not None and not call.hidden.conclusive:
            return True  # unverifiable: told to finish
        if not env.hidden_enabled and (call.visible is None or call.visible.all_passed):
            return True  # passed every visible test (or loaded, with none to run)
    return False


def _closing(env: TaskEnv, lead: str) -> str:
    noun = env.texts.code_noun
    request = f"{lead} your complete {noun} in a single ```python block"
    if may_finish(env):
        return f"{request}, or finish: {env.protocol.fill('finish_rule', noun)}."
    return f"{request}."


def no_code_block(env: TaskEnv) -> str:
    return "Your reply contained no ```python code block, so nothing was run. " + _closing(env, "Reply with")


def empty_code_block(env: TaskEnv) -> str:
    return "Your reply's ```python block was empty, so nothing was run. " + _closing(env, "Reply with")


def truncated(env: TaskEnv, where: str) -> str:
    """A reply that held no code because the length cap cut it off: `where`
    is "thinking" or "code" (`cf_agent.agent.parsing.truncation_reason`)."""
    noun = env.texts.code_noun
    if where == "code":
        return (
            "Your reply was cut off in the middle of your ```python block: it ran into the limit on how much you "
            f"may write in one turn, so the block was never closed and the {noun} is incomplete. Nothing was run — "
            f"a half-written {noun} is not worth running.\n\n"
            "Shorten what comes before the code (a few sentences of thought is plenty) and keep the code itself "
            "compact: no comments, no docstrings, no defensive extras, no worked examples. Short variable names "
            "are fine.\n\n" + _closing(env, "Reply now with")
        )
    return (
        "Your reply was cut off while you were still reasoning: it ran into the limit on how much you may write "
        f"in one turn, so it ended mid-thought and contained no {noun} at all. Nothing was run.\n\n"
        "This is not a formatting mistake — you did not get far enough to make one. Reasoning at the same length "
        "again will hit the same limit at the same point. Keep your thinking to a few sentences: settle on one "
        f"approach without exploring alternatives in writing, then spend the rest of the turn on the {noun} "
        f"itself. The {noun} is the part that matters, and it comes last, so it is the part that is lost.\n\n"
        + _closing(env, "Reply now with a brief thought and")
    )


def multiple_blocks_note(env: TaskEnv, block_count: int) -> str | None:
    """Put before the grading result when a reply held several blocks."""
    if block_count <= 1:
        return None
    ignored = block_count - 1
    earlier = "the earlier block was" if ignored == 1 else f"the {ignored} earlier blocks were"
    return (
        f"NOTE: your reply contained {block_count} ```python blocks. I executed ONLY THE LAST one — {earlier} "
        f"ignored and never ran, so nothing below says anything about {'it' if ignored == 1 else 'them'}. Send "
        f"exactly one ```python block per reply, containing the whole {env.texts.code_noun}."
    )
