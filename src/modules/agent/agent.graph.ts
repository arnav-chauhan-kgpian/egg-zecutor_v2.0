import type { AIMessage, ToolCall } from '@langchain/core/messages';
import { HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import {
  END,
  START,
  StateGraph,
  interrupt,
  type LangGraphRunnableConfig,
} from '@langchain/langgraph';
import { ChatGroq } from '@langchain/groq';
import { ChatOpenAI } from '@langchain/openai';
import { checkpointer } from '../../lib/checkpointer';
import { getToolsForSession } from './tools';
import { ApprovalStatus, AgentState, type AgentStateType, type AgentStateUpdate } from './agent.state';
import { pruneMessagesForModel } from './utils/contextWindow';

export const SENSITIVE_SHELL_TOOL = 'shell_executor';
export const SANDBOX_TOOL = 'execute_code_sandbox';
/** Sandbox runs at or under this many seconds run unreviewed. */
export const SANDBOX_AUTO_APPROVE_SECONDS = 300;

export function agentModelName(): string {
  if (process.env.GROQ_API_KEY) {
    return process.env.AGENT_MODEL ?? 'openai/gpt-oss-120b';
  }
  return process.env.AGENT_MODEL ?? 'gpt-4o-mini';
}

/**
 * Chat model for the reasoner. Groq (OpenAI-compatible, tool-calling capable)
 * when GROQ_API_KEY is set, otherwise OpenAI. The default model is chosen per
 * provider since model ids are not portable across them.
 */
export function createAgentModel(): ChatGroq | ChatOpenAI {
  if (process.env.GROQ_API_KEY) {
    return new ChatGroq({ model: agentModelName(), temperature: 0 });
  }
  return new ChatOpenAI({ model: agentModelName(), temperature: 0 });
}

const SYSTEM_PROMPT = `You are the ExecLab research agent. Work step by step.

1. Formulate a short plan before acting, and state it briefly.
2. Prefer sandboxed execution (execute_code_sandbox, default Python) over guessing: run computations, verify results, and iterate.
3. Treat shell_executor as privileged: only for one-shot shell commands you cannot do in Python.
4. Programs hand files back with ::artifact:<name>:<mime>:<base64>:: markers; after a run, check the reported artifact metadata and use read_artifact to evaluate file outputs (plots, JSON, CSV) before summarising.
5. Use calculator_engine for exact arithmetic instead of mental math.
6. Use vector_rag_search to recall prior findings and save_memory_artifact to persist durable facts.
7. Keep final answers concise and cite measured outputs (exit codes, timings, artifact names) rather than asserting from memory.`;

function toolCallsOf(message: AIMessage | undefined): ToolCall[] {
  if (!message || !Array.isArray(message.tool_calls)) return [];
  return message.tool_calls;
}

function sandboxTimeLimitSeconds(call: ToolCall): number | null {
  const args = call.args as Record<string, unknown> | string | undefined;
  if (!args || typeof args === 'string') return null;
  const value = args.timeLimit;
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** True when a tool call needs human sign-off before running. */
export function callNeedsApproval(call: ToolCall): boolean {
  if (call.name === SENSITIVE_SHELL_TOOL) return true;
  if (call.name === SANDBOX_TOOL) {
    const limit = sandboxTimeLimitSeconds(call);
    // Unknown limit = fail open to the engine default (120s); only long explicit runs gate.
    return limit !== null && limit > SANDBOX_AUTO_APPROVE_SECONDS;
  }
  return false;
}

/** a. Reasoner — prune, bind session tools, invoke the model. */
async function reasonerNode(state: AgentStateType): Promise<AgentStateUpdate> {
  const tools = getToolsForSession(state.userId, state.agentSessionId, state.enabledTools);
  const model = createAgentModel().bindTools(tools);
  const response = await model.invoke([
    new SystemMessage(SYSTEM_PROMPT),
    ...pruneMessagesForModel(state.messages),
  ]);
  return { messages: [response] };
}

export interface ApprovalResume {
  approved?: boolean;
}

function parseApprovalResume(resume: unknown): boolean {
  if (typeof resume === 'boolean') return resume;
  if (typeof resume === 'string') {
    if (/approv/i.test(resume)) return true;
    if (/reject|deny|cancel/i.test(resume)) return false;
    return false;
  }
  if (resume && typeof resume === 'object' && 'approved' in resume) {
    return (resume as ApprovalResume).approved === true;
  }
  // Fail closed: an unrecognised resume value must not authorise execution.
  return false;
}

/**
 * b. Human approval — interrupt boundary.
 *
 * Pre-approved sessions (approvalStatus already APPROVED, e.g. via a trusted
 * resume path) skip the interrupt. Otherwise pauses with the pending calls
 * attached; resume with `{ approved: boolean }` (or "APPROVED"/"REJECTED").
 * Never wraps interrupt() in try/catch — it propagates via GraphInterrupt.
 */
async function humanApprovalNode(state: AgentStateType): Promise<AgentStateUpdate> {
  if (state.approvalStatus === ApprovalStatus.APPROVED) {
    return { approvalRequired: false };
  }
  const last = state.messages[state.messages.length - 1] as AIMessage | undefined;
  const pending = toolCallsOf(last)
    .filter(callNeedsApproval)
    .map((call) => ({ name: call.name, args: call.args }));
  const resume = (await interrupt({
    type: 'approval_request',
    message: 'Sensitive tool call(s) need human sign-off.',
    toolCalls: pending,
  })) as unknown;
  const approved = parseApprovalResume(resume);
  if (approved) {
    return { approvalRequired: false, approvalStatus: ApprovalStatus.APPROVED };
  }
  return {
    approvalRequired: false,
    approvalStatus: ApprovalStatus.REJECTED,
    messages: [
      new HumanMessage(
        'The pending sensitive action was REJECTED by the human reviewer. ' +
          'Do not retry it; explain what was blocked and propose a safe alternative.',
      ),
    ],
  };
}

/**
 * c. Tools — session-bound async execution wrapper (ToolNode equivalent).
 * Tools are rebuilt per invocation so they always close over the current
 * user/session; each call runs independently and failures become ToolMessages.
 */
async function toolsNode(
  state: AgentStateType,
  config: LangGraphRunnableConfig,
): Promise<AgentStateUpdate> {
  const last = state.messages[state.messages.length - 1] as AIMessage | undefined;
  const calls = toolCallsOf(last);
  const tools = getToolsForSession(state.userId, state.agentSessionId, state.enabledTools);
  const byName = new Map(tools.map((tool) => [tool.name, tool]));
  const outputs = await Promise.all(
    calls.map(async (call) => {
      const id = call.id ?? '';
      const bound = byName.get(call.name);
      if (!bound) {
        return new ToolMessage(`Error: unknown tool "${call.name}".`, id, call.name);
      }
      try {
        const output = await bound.invoke(call.args, config);
        const content = typeof output === 'string' ? output : JSON.stringify(output);
        return new ToolMessage(content, id, call.name);
      } catch (error) {
        const reason = error instanceof Error ? error.message : String(error);
        return new ToolMessage(`Error: tool ${call.name} failed: ${reason}`, id, call.name);
      }
    }),
  );
  return { messages: outputs };
}

/**
 * reasoner -> END | human_approval | tools.
 * Pre-approved sessions flow straight to tools; gated calls pause at approval.
 */
export function shouldContinue(state: AgentStateType): string {
  const last = state.messages[state.messages.length - 1] as AIMessage | undefined;
  const calls = toolCallsOf(last);
  if (calls.length === 0) return END;
  if (state.approvalStatus === ApprovalStatus.APPROVED) return 'tools';
  return calls.some(callNeedsApproval) ? 'human_approval' : 'tools';
}

/** human_approval -> tools | END. Rejection ends the turn (see node note). */
export function shouldResume(state: AgentStateType): string {
  return state.approvalStatus === ApprovalStatus.APPROVED ? 'tools' : END;
}

export function buildAgentGraph() {
  return new StateGraph(AgentState)
    .addNode('reasoner', reasonerNode)
    .addNode('human_approval', humanApprovalNode)
    .addNode('tools', toolsNode)
    .addEdge(START, 'reasoner')
    .addConditionalEdges('reasoner', shouldContinue, {
      tools: 'tools',
      human_approval: 'human_approval',
      [END]: END,
    })
    .addConditionalEdges('human_approval', shouldResume, {
      tools: 'tools',
      [END]: END,
    })
    .addEdge('tools', 'reasoner')
    .compile({ checkpointer });
}

/** Compiled singleton — shares the isolated PostgresSaver checkpoint pool. */
export const agentGraph = buildAgentGraph();

export type AgentGraph = typeof agentGraph;
