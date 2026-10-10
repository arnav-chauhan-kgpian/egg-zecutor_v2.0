import type { BaseMessage } from '@langchain/core/messages';
import { HumanMessage } from '@langchain/core/messages';
import { Command } from '@langchain/langgraph';
import { prisma } from '../../lib/prisma';
import { ApiError } from '../../utils/ApiError';
import { agentGraph } from './agent.graph';
import type { ApprovalStatus } from './agent.state';
import { AGENT_TOOL_NAMES, isKnownAgentTool } from './tools';

export type AgentRunStatus = 'ACTIVE' | 'SUSPENDED' | 'COMPLETED';

export interface StreamChunkHandler {
  (chunk: unknown): void | Promise<void>;
}

export interface RunAgentInput {
  userId: string;
  agentSessionId?: string;
  inputPrompt: string;
  /** User's tool allowlist (ChatGPT-style toggles). Omitted/empty = all tools. */
  tools?: string[];
  streamHandler?: StreamChunkHandler;
}

export interface ResumeAgentInput {
  userId: string;
  agentSessionId: string;
  approvalDecision: 'APPROVED' | 'REJECTED';
  streamHandler?: StreamChunkHandler;
}

export interface AgentRunResult {
  agentSessionId: string;
  status: AgentRunStatus;
  /** Present when the graph paused at the approval boundary. */
  interrupt?: unknown;
  messages: BaseMessage[];
}

function threadConfig(agentSessionId: string) {
  return { configurable: { thread_id: agentSessionId } };
}

function chunkHasInterrupt(chunk: unknown): unknown | null {
  if (chunk && typeof chunk === 'object' && '__interrupt__' in chunk) {
    return (chunk as Record<string, unknown>).__interrupt__;
  }
  return null;
}

/**
 * Ensures the AgentSession row exists and is owned by the caller.
 * Revives terminal sessions (COMPLETED/SUSPENDED) back to ACTIVE on a new turn
 * so the same thread_id keeps its checkpoint history.
 */
export async function ensureAgentSession(userId: string, agentSessionId?: string) {
  if (agentSessionId) {
    const existing = await prisma.agentSession.findUnique({ where: { id: agentSessionId } });
    if (!existing) {
      return prisma.agentSession.create({
        data: { id: agentSessionId, userId, status: 'ACTIVE' },
      });
    }
    if (existing.userId !== userId) throw new ApiError(404, 'Agent session not found');
    if (existing.status !== 'ACTIVE') {
      return prisma.agentSession.update({
        where: { id: agentSessionId },
        data: { status: 'ACTIVE' },
      });
    }
    return existing;
  }
  return prisma.agentSession.create({ data: { userId, status: 'ACTIVE' } });
}

/**
 * Validates the user's tool allowlist against the registry. Unknown names are
 * a client bug (the picker only offers known tools), so fail fast rather than
 * silently running with fewer tools than the user asked for.
 */
function sanitizeEnabledTools(tools: string[] | undefined): string[] {
  if (!tools || tools.length === 0) return [];
  const unknown = [...new Set(tools)].filter((name) => !isKnownAgentTool(name));
  if (unknown.length > 0) {
    throw new ApiError(
      400,
      `Unknown agent tools: ${unknown.join(', ')}. Available: ${AGENT_TOOL_NAMES.join(', ')}`,
    );
  }
  return [...new Set(tools)];
}

async function finalMessages(agentSessionId: string): Promise<BaseMessage[]> {
  const snapshot = await agentGraph.getState(threadConfig(agentSessionId));
  const values = snapshot.values as { messages?: BaseMessage[] } | undefined;
  return values?.messages ?? [];
}

async function drainStream(
  input: Parameters<typeof agentGraph.stream>[0],
  agentSessionId: string,
  streamHandler?: StreamChunkHandler,
): Promise<{ interrupt: unknown | null }> {
  let interruptPayload: unknown | null = null;
  const stream = await agentGraph.stream(input, {
    ...threadConfig(agentSessionId),
    streamMode: 'updates',
  });
  for await (const chunk of stream) {
    const hit = chunkHasInterrupt(chunk);
    if (hit !== null) interruptPayload = hit;
    await streamHandler?.(chunk);
  }
  return { interrupt: interruptPayload };
}

/**
 * Runs one agent turn: ensures the session, invokes the graph on the session
 * thread, and persists ACTIVE -> COMPLETED (or SUSPENDED when paused at the
 * approval interrupt).
 */
export async function runAgentGraph(input: RunAgentInput): Promise<AgentRunResult> {
  const session = await ensureAgentSession(input.userId, input.agentSessionId);
  const graphInput = {
    messages: [new HumanMessage(input.inputPrompt)],
    userId: input.userId,
    agentSessionId: session.id,
    approvalRequired: false,
    approvalStatus: null as ApprovalStatus,
    enabledTools: sanitizeEnabledTools(input.tools),
  };
  const { interrupt: interruptPayload } = await drainStream(
    graphInput,
    session.id,
    input.streamHandler,
  );
  if (interruptPayload !== null) {
    await prisma.agentSession.update({
      where: { id: session.id },
      data: { status: 'SUSPENDED' },
    });
    return {
      agentSessionId: session.id,
      status: 'SUSPENDED',
      interrupt: interruptPayload,
      messages: await finalMessages(session.id),
    };
  }
  await prisma.agentSession.update({
    where: { id: session.id },
    data: { status: 'COMPLETED' },
  });
  return { agentSessionId: session.id, status: 'COMPLETED', messages: await finalMessages(session.id) };
}

/**
 * Resumes an interrupted turn after human review. APPROVED continues to tool
 * execution; REJECTED records the decision and ends the turn.
 */
export async function resumeAgentGraph(input: ResumeAgentInput): Promise<AgentRunResult> {
  const session = await prisma.agentSession.findUnique({ where: { id: input.agentSessionId } });
  if (!session || session.userId !== input.userId) {
    throw new ApiError(404, 'Agent session not found');
  }
  const approved = input.approvalDecision === 'APPROVED';
  // Resume-only command: no goto, just delivers the human decision to the
  // paused interrupt(). Cast is target-typing only — stream() branches on the
  // runtime COMMAND_SYMBOL, and this command carries no state update.
  const resumeCommand = new Command({ resume: { approved } }) as Parameters<
    typeof agentGraph.stream
  >[0];
  const { interrupt: interruptPayload } = await drainStream(
    resumeCommand,
    session.id,
    input.streamHandler,
  );
  if (interruptPayload !== null) {
    await prisma.agentSession.update({
      where: { id: session.id },
      data: { status: 'SUSPENDED' },
    });
    return {
      agentSessionId: session.id,
      status: 'SUSPENDED',
      interrupt: interruptPayload,
      messages: await finalMessages(session.id),
    };
  }
  await prisma.agentSession.update({
    where: { id: session.id },
    data: { status: 'COMPLETED' },
  });
  return { agentSessionId: session.id, status: 'COMPLETED', messages: await finalMessages(session.id) };
}

export function getAgentGraph() {
  return agentGraph;
}
