"""Qwen-Agent (FnCallAgent) on the shared engine (DESIGN.md §9.2).

* **LLM bridge.** A `BaseFnCallModel` whose `_chat_no_stream` receives the
  messages Qwen-Agent has already prepared — its own `nous` tool block in the
  system message, earlier calls and results flattened to `<tool_call>` /
  `<tool_response>` text — converts them to plain dicts and calls the shared
  engine without tool schemas (the tool text is already in the system
  message). Qwen-Agent's own parser then runs on the reply, unmodified.
* **Tool bridge.** Each harness tool becomes a `BaseTool`, validating its
  arguments with Qwen-Agent's own `_verify_json_format_args`. Invalid
  arguments get Qwen-Agent's error message without its traceback, whose
  frames would be the harness's code. A failure of the harness inside a tool
  passes through Qwen-Agent as `HarnessAbort` instead of becoming an error
  message for the model.
* **Recording.** The agent's `_call_llm` and `_call_tool` are wrapped on the
  instance: every parsed call is recorded — including a call to an unknown
  tool, which never reaches a tool object and to which Qwen-Agent answers
  "Tool X does not exists." — and each model call gets a parse status from
  the framework-independent diagnostics, using `json5` as Qwen-Agent does.
* **Budget.** `MAX_LLM_CALL_PER_RUN` is a module constant read at import;
  it is set to `max_model_calls` for the duration of a run and restored.
* **Stop reasons.** A reply without a tool call ends the run. It is
  `final_answer` when the reply did not attempt a call and was not cut off
  (the protocol's finish and give-up), and `no_tool_call` when it attempted
  a call Qwen-Agent did not parse or hit the length cap.
* **No truncation.** `max_input_tokens` is 0 (validated in params), so
  Qwen-Agent never shortens the conversation; the engine ends the
  trajectory instead.
* **No side effects.** FnCallAgent's RAG memory is unused without files and
  is not created; Qwen-Agent's workspace is pointed at a cache directory, so
  nothing is written into the repository.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "QWEN_AGENT_DEFAULT_WORKSPACE", str(Path.home() / ".cache" / "hallu_harness" / "qwen_agent_workspace")
)

from ...core import parse_diag  # noqa: E402
from ...core.errors import HarnessAbort  # noqa: E402
from ...core.protocol import TOOL_PROTOCOL_QWEN  # noqa: E402
from ...core.types import FrameworkResult, ToolResult  # noqa: E402
from ..base import FrameworkAdapter  # noqa: E402
from ..smol.prompts import JSON_ENCODING_SECTION  # noqa: E402

# Qwen-Agent's own tool block (`nous`) follows this in the system message. It
# invites several calls per reply, and Qwen-Agent runs every one; FnCallAgent
# also ends the run on the first reply without a call. Both are stated here,
# since a model told neither would end a task by thinking out loud. The reply
# form mirrors smolagents': a Thought, then the action, here a <tool_call>
# block in Qwen's own format, shown in full under "The form of a reply" as
# smolagents shows its own. With the reasoning merely allowed ("You may
# write...", until PROMPT_REVISION 8) or required without a shown form
# (revision 8), Qwen3 with thinking off sent bare calls, so Qwen-Agent was the
# only framework whose replies carried no reasoning.
REPLY_SECTION = (
    "# How to reply\n"
    "\n"
    "A reply that has a {code} graded consists of one thought followed by one tool call:\n"
    "\n"
    "Thought: a few sentences on your approach or, after a grading result, on what the result tells you and "
    "what you will change.\n"
    "\n"
    'Then one call to grade_code, with the complete {code} as its "code" argument.\n'
    "\n"
    "- Make exactly one tool call per reply, even though the tool section below permits several.\n"
    "- A reply that contains no tool call ends the task at once, and its text is your final answer. Until a "
    "grading result tells you that you may finish, every reply must therefore call a tool."
)
FORM_SECTION = (
    "# The form of a reply\n"
    "\n"
    "A reply that has a {code} graded has this form, with each part in angle brackets replaced:\n"
    "\n"
    "Thought: <your thought>\n"
    "<tool_call>\n"
    '{{"name": "grade_code", "arguments": {{"code": "<your complete {code}, encoded as described above>"}}}}\n'
    "</tool_call>"
)


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        text = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
        if text is not None:
            parts.append(text)
    return "".join(parts)


@contextlib.contextmanager
def _budget(n: int):
    from qwen_agent.agents import fncall_agent

    previous = fncall_agent.MAX_LLM_CALL_PER_RUN
    fncall_agent.MAX_LLM_CALL_PER_RUN = n
    try:
        yield
    finally:
        fncall_agent.MAX_LLM_CALL_PER_RUN = previous


def _make_llm(adapter: "QwenAgentAdapter"):
    from qwen_agent.llm.function_calling import BaseFnCallModel
    from qwen_agent.llm.schema import ASSISTANT, Message

    class BridgeLLM(BaseFnCallModel):
        def _chat_no_stream(self, messages, generate_cfg):
            plain = [{"role": m.role, "content": _content_text(m.content)} for m in messages]
            stop = generate_cfg.get("stop") or None
            generation = adapter.generate(plain, tools=None, stop=stop)
            return [Message(role=ASSISTANT, content=generation.content)]

        def _chat_stream(self, messages, delta_stream, generate_cfg):
            yield self._chat_no_stream(messages, generate_cfg)

    p = adapter.params
    return BridgeLLM(
        {
            "model": adapter.backend.info.get("id", "model"),
            "model_type": "hallu_bridge",
            "generate_cfg": {
                "fncall_prompt_type": p["fncall_prompt_type"],
                "parallel_function_calls": p["parallel_function_calls"],
                "thought_in_content": p["thought_in_content"],
                "max_input_tokens": p["max_input_tokens"],
                "max_retries": 0,
            },
        }
    )


def _make_tool(tool, adapter: "QwenAgentAdapter"):
    from qwen_agent.tools import BaseTool

    class BridgedTool(BaseTool):
        name = tool.name
        description = tool.description
        parameters = tool.parameters

        def call(self, params, **kwargs):
            try:
                args = self._verify_json_format_args(params)
                result: ToolResult = tool.run(**args)
            except Exception as exc:
                # The model's arguments (bad JSON, schema, unknown keys). The
                # text is Qwen-Agent's own error message without the traceback
                # it appends: those frames are the harness's (its file paths
                # and source lines), nothing the model wrote. A HarnessAbort is
                # not caught here or by Qwen-Agent, and ends the run.
                adapter._tool_status = ("invalid_arguments", f"{type(exc).__name__}: {exc}")
                return f"An error occurred when calling tool `{self.name}`:\n{type(exc).__name__}: {exc}"
            adapter._tool_result = result
            return result.observation

    return BridgedTool()


class QwenAgentAdapter(FrameworkAdapter):
    name = "qwen_agent"
    protocol = TOOL_PROTOCOL_QWEN

    def __init__(self, params, backend, recorder, benchmark, prompt_mode):
        super().__init__(params, backend, recorder, benchmark, prompt_mode)
        self.content_decode = "skip_special_tokens" if params["content_special_tokens"] == "skip" else "keep_special_tokens"
        self._tool_status: tuple[str, str] | None = None
        self._tool_result: ToolResult | None = None
        self._last_output_had_calls = False

    @classmethod
    def framework_info(cls, params: dict) -> dict:
        import qwen_agent

        return {
            "name": "qwen_agent",
            "version": qwen_agent.__version__,
            "agent_class": params["agent_class"],
            "params": params,
            "prompt_mode": params["prompt_mode"],
            "fncall_prompt_type": params["fncall_prompt_type"],
            "content_decode": "skip_special_tokens" if params["content_special_tokens"] == "skip" else "keep_special_tokens",
            "max_input_tokens": params["max_input_tokens"],
            "rag_memory": False,
            "budget": {"max_model_calls": params["max_model_calls"], "native": {"MAX_LLM_CALL_PER_RUN": params["max_model_calls"]}},
        }

    def run(self, task, tools, env) -> FrameworkResult:
        import logging

        import json5
        import qwen_agent
        from qwen_agent.agents import FnCallAgent
        from qwen_agent.utils.utils import json_loads as qwen_json_loads  # what BaseTool's argument check parses with

        # Qwen-Agent logs every tool error with its traceback at WARNING. The
        # error is recorded in the trajectory, and the console prints it.
        logging.getLogger("qwen_agent_logger").setLevel(logging.ERROR)

        texts = self.benchmark.texts
        instructions = texts.task_instructions(env, self.max_model_calls)
        if self.prompt_mode == "harmonized":
            reply = REPLY_SECTION.format(code=texts.code_noun)
            form = FORM_SECTION.format(code=texts.code_noun)
            system_message = f"{instructions}\n\n{reply}\n\n{JSON_ENCODING_SECTION}\n\n{form}"
            user_text = texts.user_message(task, self.protocol)
        else:
            from qwen_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE

            system_message = DEFAULT_SYSTEM_MESSAGE
            user_text = instructions + "\n\n" + texts.user_message(task, self.protocol)

        class NoMemoryFnCallAgent(FnCallAgent):
            def __init__(self, *args, **kwargs):
                self.mem = None  # RAG memory: unused without files; not created
                super().__init__(*args, **kwargs)

        agent = NoMemoryFnCallAgent(
            function_list=[_make_tool(t, self) for t in tools],
            llm=_make_llm(self),
            system_message=system_message,
        )

        original_call_llm = agent._call_llm
        original_call_tool = agent._call_tool

        def arguments_parse(arguments) -> bool:
            if not isinstance(arguments, str):
                return True
            try:
                qwen_json_loads(arguments)
                return True
            except Exception:  # noqa: BLE001 - any failure means the validator will reject it
                return False

        def call_llm(messages, functions=None, stream=True, extra_generate_cfg=None):
            outputs = list(original_call_llm(messages=messages, functions=functions, stream=stream, extra_generate_cfg=extra_generate_cfg))
            final = outputs[-1] if outputs else []
            calls = [m for m in final if m.function_call]
            self._last_output_had_calls = bool(calls)
            step = self.recorder.current
            if step is not None and not step.ended:
                raw = step.generation.raw_output
                attempted = parse_diag.attempted_tool_call(raw)
                parsed = parse_diag.tagged_calls_parse(raw, loads=json5.loads)
                # The same reply gets the same label in every framework: a
                # call whose JSON is broken is `failed` here as in smolagents,
                # even though Qwen-Agent's lenient extractor still makes a
                # call of it (which its argument validator then rejects).
                if calls and (not parsed or all(parsed)):
                    status, error = "ok", None
                elif calls and len(calls) == len(parsed) and all(arguments_parse(m.function_call.arguments) for m in calls):
                    status = "recovered"
                    error = "a <tool_call> block was not valid JSON; Qwen-Agent's lenient extractor recovered a call with valid arguments"
                elif calls:
                    status = "failed"
                    error = ("a <tool_call> block was not valid JSON, or was not closed; the call Qwen-Agent's lenient "
                             "extractor made of it has arguments that do not parse")
                else:
                    status = "failed" if attempted else "no_tool_call"
                    error = "the reply attempted a tool call that Qwen-Agent did not parse; Qwen-Agent treats it as a final answer" if attempted else None
                self.recorder.on_parse(status, error, attempted)
            return iter(outputs)

        def call_tool(tool_name, tool_args="{}", **kwargs):
            raw = tool_args if isinstance(tool_args, str) else json.dumps(tool_args, ensure_ascii=False)
            try:
                arguments = json5.loads(tool_args) if isinstance(tool_args, str) else tool_args
            except Exception:  # noqa: BLE001 - keep the raw string when it does not parse
                arguments = tool_args
            record = self.recorder.on_tool_call(tool_name, arguments, raw_arguments=raw)
            self._tool_status, self._tool_result = None, None
            started = time.perf_counter()
            try:
                observation = original_call_tool(tool_name, tool_args, **kwargs)
            except HarnessAbort as exc:
                self.recorder.on_tool_result(record, observation=None, status="raised", error=str(exc),
                                             seconds=time.perf_counter() - started)
                raise
            seconds = time.perf_counter() - started
            if tool_name not in agent.function_map:
                status, error = "unknown_tool", None
            elif self._tool_status is not None:
                status, error = self._tool_status
            else:
                status, error = "executed", None
            text = observation if isinstance(observation, str) else json.dumps(observation, ensure_ascii=False, default=str)
            self.recorder.on_tool_result(
                record,
                observation=text,
                status=status,
                private=self._tool_result.private if self._tool_result is not None else {},
                error=error,
                seconds=seconds,
            )
            return observation

        agent._call_llm = call_llm
        agent._call_tool = call_tool

        response: list = []
        error: BaseException | None = None
        with _budget(self.max_model_calls):
            try:
                for response in agent.run([{"role": "user", "content": user_text}]):
                    pass
            except Exception as exc:  # noqa: BLE001 - every failure ends the trajectory with a reason
                error = exc

        final_answer = None
        if self.recorder.context_exhausted is not None:
            stop = "context_exhausted"
        elif error is not None:
            stop = "error"
        elif self._last_output_had_calls:
            stop = "budget"  # FnCallAgent stops calling the model once the budget is spent
        else:
            # FnCallAgent ends on the first reply without a parsed call. That is
            # this protocol's way to finish (and to give up), so it is
            # `final_answer`, as in the other frameworks — unless the reply
            # attempted a call that did not parse, or was cut off by the length
            # cap: then the run ended without the model deciding to finish.
            last = self.recorder.current
            deliberate = last is not None and last.parse.status == "no_tool_call" and not last.generation.hit_length_cap
            stop = "final_answer" if deliberate else "no_tool_call"
            texts_out = [_content_text(m.get("content") if isinstance(m, dict) else m.content)
                         for m in response if (m.get("role") if isinstance(m, dict) else m.role) == "assistant"
                         and not (m.get("function_call") if isinstance(m, dict) else m.function_call)]
            final_answer = texts_out[-1] if texts_out else None

        self.recorder.finish()
        return FrameworkResult(
            final_answer=final_answer,
            stop_reason=stop,
            system_prompt=system_message or None,
            user_query=user_text,
            framework_version=qwen_agent.__version__,
            error=None if error is None else f"{type(error).__name__}: {error}",
            native={"budget_exhausted_with_pending_calls": stop == "budget"},
        )
