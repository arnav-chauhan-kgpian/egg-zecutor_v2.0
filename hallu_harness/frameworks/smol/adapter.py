"""smolagents (ToolCallingAgent) on the shared engine (DESIGN.md §9.1).

* **Model bridge.** A `smolagents.Model` whose `generate` prepares messages
  and tool schemas exactly as stock `TransformersModel` does — its own
  `_prepare_completion_kwargs` with `tool_choice=None`, text-flattened
  messages, tool-role conversion — and hands them to the shared engine. One
  deliberate difference, set by `calling_tools_turn`: smolagents' "Calling
  tools:" record of each past call goes in the user turn (`user`, the
  default), not appended to the model's own reply (`assistant`, stock). And
  the tool schemas reach the chat template only with `chat_template_tools`
  (stock): otherwise Qwen's template adds a second, contradicting tools
  section. The
  reply goes back decoded as stock does (`skip_special_tokens`), cut at
  smolagents' stop sequences. Parsing is smolagents' own (`stock`) or
  cf_agent's `<tool_call>`-aware variant (`tagged`).
* **Tool bridge.** Each harness tool becomes a `smolagents.Tool`. Tools run
  one at a time (`max_tool_threads=1`): grade_code is stateful, and parallel
  calls would race on it. A failure of the harness inside a tool passes
  through smolagents as `HarnessAbort` instead of becoming an error message
  for the model.
* **Recording from events.** smolagents yields every `ToolCall` *before*
  executing it, so each call is recorded even if it then fails validation,
  names an unknown tool or raises — cases where smolagents itself leaves
  `ActionStep.tool_calls` unset. Each `ActionStep` closes the recorder's
  step with the parse status and the exact error text smolagents feeds back.
* **Kinds.** The extra model call smolagents makes after the step budget
  (`provide_final_answer`) is recorded as `final_from_memory`; planning
  calls as `planning`.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

from ...core import parse_diag
from ...core.errors import ContextExhausted, HarnessAbort
from ...core.protocol import TOOL_PROTOCOL_SMOLAGENTS
from ...core.types import FrameworkResult, ToolResult
from ..base import FrameworkAdapter
from .prompts import final_answer_tool, harmonized_system_prompt, prompt_templates

TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TRUNCATED_REPLY_ERROR = (
    "your reply was cut off: it ran into the limit on how much you may write in one turn, so it ended before its "
    "action was complete, and nothing was run. Keep the thought to a few sentences and the code compact (no "
    "comments, no docstrings), so that the whole action fits in one reply."
)
TRUNCATED_THINKING_ERROR = (
    "your reply was cut off while you were still inside your <think> block, so it contained no "
    "action at all and nothing ran. This is not a formatting mistake. Keep your reasoning to a "
    "few sentences and spend the turn on the action itself."
)


def parse_json_blob(json_blob: str) -> tuple[dict, str]:
    """`smolagents.utils.parse_json_blob` (1.26) with one fix. A decoding
    error's position is in the extracted JSON, and smolagents used it to index
    the whole reply: the error quoted the wrong part of the reply ("decoding
    failed on that specific part"), its line and column did not match the
    text it showed, and its several-calls check looked in the wrong place.
    Here the position, the quoted part and the text shown are all the JSON's,
    and an error in the first four characters is quoted too (a negative slice
    start quoted nothing). Every message is otherwise smolagents' own."""
    try:
        first_accolade_index = json_blob.find("{")
        last_accolade_index = [a.start() for a in list(re.finditer("}", json_blob))][-1]
        json_str = json_blob[first_accolade_index : last_accolade_index + 1]
        json_data = json.loads(json_str, strict=False)
        return json_data, json_blob[:first_accolade_index]
    except IndexError:
        raise ValueError("The model output does not contain any JSON blob.")
    except json.JSONDecodeError as e:
        place = e.pos
        if json_str[place - 1 : place + 2] == "},\n":
            raise ValueError(
                "JSON is invalid: you probably tried to provide multiple tool calls in one action. PROVIDE ONLY ONE TOOL CALL."
            )
        raise ValueError(
            f"The JSON blob you used is invalid due to the following error: {e}.\n"
            f"JSON blob was: {json_str}, decoding failed on that specific part of the blob:\n"
            f"'{json_str[max(place - 4, 0) : place + 5]}'."  # smolagents' negative start quoted '' near the start
        )


def tool_call_from_text(text: str, tool_name_key: str, tool_arguments_key: str):
    """`smolagents.models.get_tool_call_from_text` (1.26), on the fixed
    `parse_json_blob`."""
    import uuid

    from smolagents.models import ChatMessageToolCall, ChatMessageToolCallFunction, parse_json_if_needed

    tool_call_dictionary, _ = parse_json_blob(text)
    try:
        tool_name = tool_call_dictionary[tool_name_key]
    except Exception as e:
        raise ValueError(
            f"Tool call needs to have a key '{tool_name_key}'. Got keys: {list(tool_call_dictionary.keys())} instead"
        ) from e
    tool_arguments = tool_call_dictionary.get(tool_arguments_key, None)
    if isinstance(tool_arguments, str):
        tool_arguments = parse_json_if_needed(tool_arguments)
    return ChatMessageToolCall(
        id=str(uuid.uuid4()),
        type="function",
        function=ChatMessageToolCallFunction(name=tool_name, arguments=tool_arguments),
    )


def _plain_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        role = m["role"]
        role = role.value if hasattr(role, "value") else str(role)
        content = m.get("content")
        if isinstance(content, list):  # not expected with flattening, but be exact if it happens
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        out.append({"role": role, "content": content if content is not None else ""})
    return out


def _make_model_bridge(adapter: "SmolagentsAdapter"):
    import smolagents
    from smolagents.models import ChatMessage, MessageRole, TokenUsage, tool_role_conversions

    from cf_agent.agent.parsing import strip_thinking

    # Stock: tool-call -> assistant, tool-response -> user. Consecutive
    # messages of one role are then merged, so the "Calling tools:" message
    # joins the turn on its side.
    role_conversions = {**tool_role_conversions, MessageRole.TOOL_CALL: MessageRole(adapter.params["calling_tools_turn"])}

    class SmolModelBridge(smolagents.Model):
        def __init__(self):
            super().__init__(flatten_messages_as_text=True, model_id=adapter.backend.info.get("id", "model"))

        def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
            if response_format is not None:
                raise ValueError("structured outputs are not supported by the harness engine")
            completion = self._prepare_completion_kwargs(
                messages=messages,
                stop_sequences=stop_sequences,
                tools_to_call_from=tools_to_call_from,
                tool_choice=None,
                custom_role_conversions=role_conversions,
                **kwargs,
            )
            plain = _plain_messages(completion.pop("messages"))
            stop = completion.pop("stop", None)
            tools = completion.pop("tools", None)
            if not adapter.params["chat_template_tools"]:
                tools = None  # the system prompt already lists the tools, with smolagents' own reply rules
            generation = adapter.generate(plain, tools=tools, stop=stop, kind=adapter.pending_kind)
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=generation.content,
                raw={"finish_reason": generation.finish_reason},
                token_usage=TokenUsage(
                    input_tokens=generation.input_tokens or 0, output_tokens=generation.output_tokens or 0
                ),
            )

        def parse_tool_calls(self, message):
            if not message.tool_calls and message.content:
                source = message.content
                if adapter.params["tool_call_parser"] == "tagged":
                    visible = strip_thinking(message.content)
                    if not visible.strip():
                        raise ValueError(TRUNCATED_THINKING_ERROR)
                    match = TOOL_CALL_TAG_RE.search(visible)
                    source = match.group(1) if match else visible
                try:
                    message.tool_calls = [tool_call_from_text(source, self.tool_name_key, self.tool_arguments_key)]
                except ValueError:
                    # Cut off by the length cap: the parse error is a symptom,
                    # and the model is told the cause, as the text protocol does.
                    if (message.raw or {}).get("finish_reason") == "length":
                        raise ValueError(TRUNCATED_REPLY_ERROR) from None
                    raise
            return super().parse_tool_calls(message)

    return SmolModelBridge()


def _make_tool_bridge(tool, adapter: "SmolagentsAdapter"):
    import smolagents

    spec = tool.parameters
    required = set(spec.get("required", []))
    inputs = {}
    for key, schema in spec["properties"].items():
        entry = {"type": schema.get("type", "string"), "description": schema.get("description", "")}
        if key not in required:
            entry["nullable"] = True
        inputs[key] = entry
    first_input = next(iter(spec["properties"]))

    class BridgedTool(smolagents.Tool):
        name = tool.name
        description = tool.description
        output_type = "string"
        skip_forward_signature_validation = True

        def forward(self, *args, **kwargs):
            if args and not kwargs and len(args) == 1:
                kwargs = {first_input: args[0]}  # smolagents passes a bare string argument positionally
            record = adapter.claim_pending(tool.name)
            started = time.perf_counter()
            try:
                # An argument error goes back to smolagents, which reports it to
                # the model; a HarnessAbort passes through smolagents and ends the run.
                result: ToolResult = tool.run(**kwargs)
            except (Exception, HarnessAbort) as exc:
                adapter.store_tool_outcome(record, None, f"{type(exc).__name__}: {exc}", time.perf_counter() - started)
                raise
            adapter.store_tool_outcome(record, result, None, time.perf_counter() - started)
            return result.observation

    BridgedTool.inputs = inputs
    return BridgedTool()


class SmolagentsAdapter(FrameworkAdapter):
    name = "smolagents"
    protocol = TOOL_PROTOCOL_SMOLAGENTS

    def __init__(self, params, backend, recorder, benchmark, prompt_mode):
        super().__init__(params, backend, recorder, benchmark, prompt_mode)
        self.content_decode = "skip_special_tokens" if params["content_special_tokens"] == "skip" else "keep_special_tokens"
        self.pending_kind: str | None = None
        self._lock = threading.Lock()
        self._claimed: set[int] = set()
        self._outcomes: dict[int, tuple[ToolResult | None, str | None, float]] = {}
        self._steps_closed = 0

    @classmethod
    def framework_info(cls, params: dict) -> dict:
        import smolagents

        return {
            "name": "smolagents",
            "version": smolagents.__version__,
            "agent_class": params["agent_class"],
            "params": params,
            "prompt_mode": params["prompt_mode"],
            "tool_call_parser": params["tool_call_parser"],
            "chat_template_tools": params["chat_template_tools"],
            "calling_tools_turn": params["calling_tools_turn"],
            "content_decode": "skip_special_tokens" if params["content_special_tokens"] == "skip" else "keep_special_tokens",
            "tool_threads": 1,
            "stop_sequences": ["Observation:", "Calling tools:"],
            "budget": {
                "max_model_calls": params["max_model_calls"],
                "native": {"max_steps": params["max_model_calls"], "answer_from_memory_on_budget": params["answer_from_memory_on_budget"]},
            },
        }

    # -- tool-call bookkeeping (called from the tool bridge, possibly in a worker thread)

    def claim_pending(self, name: str):
        with self._lock:
            step = self.recorder.current
            for record in step.tool_calls if step else []:
                if record.status == "pending" and record.name == name and id(record) not in self._claimed:
                    self._claimed.add(id(record))
                    return record
        return None

    def store_tool_outcome(self, record, result, error, seconds) -> None:
        if record is None:
            return
        with self._lock:
            self._outcomes[id(record)] = (result, error, seconds)

    # -- the run --------------------------------------------------------

    def run(self, task, tools, env) -> FrameworkResult:
        import smolagents
        from smolagents import ToolCallingAgent
        from smolagents.agents import ToolOutput
        from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep, ToolCall
        from smolagents.models import ChatMessage, MessageRole
        from smolagents.monitoring import LogLevel
        from smolagents.utils import AgentMaxStepsError

        texts = self.benchmark.texts
        if self.prompt_mode == "harmonized":
            templates = prompt_templates(harmonized_system_prompt(texts, env, self.params))
            task_text = texts.user_message(task, self.protocol)
        else:
            templates = prompt_templates(None)
            task_text = texts.task_instructions(env, self.max_model_calls) + "\n\n" + texts.user_message(task, self.protocol)

        bridged = [_make_tool_bridge(t, self) for t in tools]
        if self.prompt_mode == "harmonized":
            bridged.append(final_answer_tool(texts.code_noun))  # stock mode keeps smolagents' own
        agent = ToolCallingAgent(
            tools=bridged,
            model=_make_model_bridge(self),
            prompt_templates=templates,
            max_steps=self.max_model_calls,
            planning_interval=self.params["planning_interval"],
            verbosity_level=LogLevel.OFF,
            max_tool_threads=1,
        )

        original_final = agent.provide_final_answer

        def provide_final_answer(task_arg):
            if not self.params["answer_from_memory_on_budget"]:
                return ChatMessage(role=MessageRole.ASSISTANT, content=None)
            self.pending_kind = "final_from_memory"
            try:
                return original_final(task_arg)
            finally:
                self.pending_kind = None

        agent.provide_final_answer = provide_final_answer

        original_plan = agent._generate_planning_step

        def planning(task_arg, is_first_step, step):
            self.pending_kind = "planning"
            try:
                yield from original_plan(task_arg, is_first_step=is_first_step, step=step)
            finally:
                self.pending_kind = None

        agent._generate_planning_step = planning

        final_answer: Any = None
        error: BaseException | None = None
        seen: set[int] = set()
        try:
            for event in agent.run(task_text, stream=True, max_steps=self.max_model_calls, reset=True):
                if isinstance(event, ToolCall):
                    self.recorder.on_tool_call(event.name, event.arguments, call_id=event.id)
                elif isinstance(event, ToolOutput):
                    self._on_tool_output(event)
                elif isinstance(event, ActionStep):
                    if id(event) not in seen:
                        seen.add(id(event))
                        self._close_step(event)
                elif isinstance(event, PlanningStep):
                    self._close_planning(event)
                elif isinstance(event, FinalAnswerStep):
                    final_answer = event.output
        except Exception as exc:  # noqa: BLE001 - every failure ends the trajectory with a reason
            error = exc

        steps = list(agent.memory.steps)
        budget_hit = any(isinstance(getattr(s, "error", None), AgentMaxStepsError) for s in steps)
        final_tool_used = any(getattr(s, "is_final_answer", False) for s in steps)

        current = self.recorder.current
        if current is not None and not current.ended and current.kind == "final_from_memory":
            raw = current.generation.raw_output
            self.recorder.on_parse("no_tool_call", None, parse_diag.attempted_tool_call(raw))

        native: dict[str, Any] = {"budget_hit": budget_hit, "final_answer_tool_called": final_tool_used}
        if self.recorder.context_exhausted is not None:
            stop = "context_exhausted"
            if budget_hit:
                final_answer = None  # smolagents' text is an error string, not an answer
        elif error is not None:
            stop = "error"
        elif self.recorder.model_errors:
            # smolagents' answer-from-memory call catches the model's failure
            # and returns the error text as if it were the answer.
            stop, final_answer = "error", None
        elif budget_hit:
            stop = "budget"
            native["final_answer_from_memory"] = self.params["answer_from_memory_on_budget"]
        elif final_tool_used:
            stop = "final_answer"
        else:
            stop = "no_final_answer"

        self.recorder.finish()
        return FrameworkResult(
            final_answer=None if final_answer is None else str(final_answer),
            stop_reason=stop,
            system_prompt=agent.system_prompt,
            user_query=f"New task:\n{task_text}",
            framework_version=smolagents.__version__,
            error=self._describe(error) if error is not None else (self.recorder.model_errors[-1] if stop == "error" else None),
            native=native,
        )

    # -- event handlers ---------------------------------------------------

    @staticmethod
    def _describe(exc: BaseException) -> str:
        cause = exc.__cause__ if exc.__cause__ is not None else exc
        return f"{type(exc).__name__}: {exc}" if cause is exc else f"{type(exc).__name__} <- {type(cause).__name__}: {cause}"

    def _on_tool_output(self, event) -> None:
        record = next((r for r in self.recorder.current.tool_calls if r.call_id == event.id), None) if self.recorder.current else None
        if record is None:
            return
        with self._lock:
            result, error, seconds = self._outcomes.pop(id(record), (None, None, 0.0))
        self.recorder.on_tool_result(
            record,
            observation=event.observation,
            status="executed",
            private=result.private if result is not None else {},
            error=error,
            seconds=seconds,
        )

    def _new_model_call_since_last_close(self) -> bool:
        return len(self.recorder.steps) > self._steps_closed and self.recorder.current is not None and not self.recorder.current.ended

    def _close_planning(self, event) -> None:
        if not self._new_model_call_since_last_close():
            return
        self.recorder.on_parse("no_tool_call", None, None)
        self.recorder.end_step()
        self._steps_closed = len(self.recorder.steps)

    def _close_step(self, event) -> None:
        from smolagents.memory import MessageRole
        from smolagents.utils import AgentParsingError, AgentToolCallError, AgentToolExecutionError

        if not self._new_model_call_since_last_close():
            return  # the model call for this ActionStep never happened (it raised)
        step = self.recorder.current
        raw = step.generation.raw_output
        attempted = parse_diag.attempted_tool_call(raw)
        err = event.error

        if isinstance(err, AgentParsingError):
            self.recorder.on_parse("failed" if attempted else "no_tool_call", str(err), attempted)
        else:
            self.recorder.on_parse("ok" if step.tool_calls else "no_tool_call", None, attempted)

        for record in step.tool_calls:
            if record.status != "pending":
                continue
            with self._lock:
                result, tool_error, seconds = self._outcomes.pop(id(record), (None, None, 0.0))
            if isinstance(err, AgentToolCallError):
                status = "invalid_arguments"
            elif isinstance(err, AgentToolExecutionError) and str(err).startswith("Unknown tool"):
                status = "unknown_tool"
            elif result is not None:
                status = "executed_not_returned"  # ran, but a sibling call's failure stopped its output
            else:
                status = "raised"
            self.recorder.on_tool_result(
                record,
                observation=None,
                status=status,
                private=result.private if result is not None else {},
                error=tool_error or (str(err) if err is not None else None),
                seconds=seconds,
            )

        # What the framework puts in the next user turn, joined as smolagents
        # merges one role's consecutive messages.
        user_side = {MessageRole.TOOL_RESPONSE}
        if self.params["calling_tools_turn"] == "user":
            user_side.add(MessageRole.TOOL_CALL)
        feedback = []
        for message in event.to_messages():
            if message.role in user_side:
                for part in message.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        feedback.append(part["text"])
        if feedback:
            self.recorder.on_framework_feedback("\n".join(feedback))
        self.recorder.end_step()
        self._steps_closed = len(self.recorder.steps)
