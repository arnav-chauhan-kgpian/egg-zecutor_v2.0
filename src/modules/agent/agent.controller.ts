import type { Request, Response } from 'express';
import { z } from 'zod';
import { prisma } from '../../lib/prisma';
import { ApiError } from '../../utils/ApiError';
import { asyncHandler } from '../../utils/asyncHandler';
import { agentGraph } from './agent.graph';
import {
  ensureAgentSession,
  resumeAgentGraph,
  runAgentGraph,
} from './agent.service';
import { describeAgentTools } from './tools';

// --- validation schemas (used by the router's validateBody/validateQuery) ---

export const chatSchema = z.object({
  agentSessionId: z.string().uuid().optional(),
  prompt: z.string().min(1, 'prompt cannot be empty').max(50_000, 'prompt is too large'),
  /** Tool allowlist (user toggles). Omitted/empty = all tools enabled. */
  tools: z.array(z.string().min(1).max(64)).max(32).optional(),
});

export const resumeSchema = z.object({
  agentSessionId: z.string().uuid(),
  approved: z.boolean(),
  reason: z.string().max(2000).optional(),
});

export const sessionListSchema = z.object({
  page: z.coerce.number().int().positive().default(1),
  pageSize: z.coerce.number().int().positive().max(100).default(20),
});

// --- SSE plumbing ------------------------------------------------------------

function setSseHeaders(res: Response): void {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();
}

function writeSse(res: Response, event: string, data: unknown): void {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

interface ToolCallLike {
  name: string;
  args: unknown;
}

interface MessageLike {
  getType?: () => string;
  content?: unknown;
  name?: string;
  tool_calls?: ToolCallLike[];
  tool_call_id?: string;
}

function asMessages(update: unknown): MessageLike[] {
  if (update && typeof update === 'object' && 'messages' in update) {
    const messages = (update as { messages?: unknown }).messages;
    if (Array.isArray(messages)) return messages as MessageLike[];
  }
  return [];
}

function messageText(content: unknown): string {
  return typeof content === 'string' ? content : JSON.stringify(content ?? '');
}

/**
 * Translates `streamMode: 'updates'` chunks into the public SSE event set.
 * AI messages without tool calls become message_delta; tool requests become
 * tool_start; ToolMessages become tool_end; `__interrupt__` becomes interrupt.
 */
function emitChunkAsEvents(chunk: unknown, emit: (event: string, data: unknown) => void): void {
  if (!chunk || typeof chunk !== 'object') return;
  const record = chunk as Record<string, unknown>;
  if ('__interrupt__' in record) {
    emit('interrupt', { approvalRequired: true, pendingToolCalls: record.__interrupt__ });
    return;
  }
  for (const update of Object.values(record)) {
    for (const message of asMessages(update)) {
      const type = message.getType?.();
      if (type === 'ai') {
        const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
        if (calls.length > 0) {
          for (const call of calls) {
            emit('tool_start', { tool: call.name, input: call.args });
          }
        } else {
          emit('message_delta', messageText(message.content));
        }
      } else if (type === 'tool') {
        emit('tool_end', { tool: message.name, output: messageText(message.content) });
      }
    }
  }
}

/**
 * Runs a graph turn over SSE. Ownership/session checks happen BEFORE headers;
 * everything after headers is delivered as events (an `error` event, never a
 * status change). On client disconnect the emitter goes inert — the graph run
 * continues to a persisted terminal state the client can re-read via history.
 */
async function streamTurn(
  req: Request,
  res: Response,
  sessionId: string,
  run: (streamHandler: (chunk: unknown) => Promise<void>) => Promise<{ status: string }>,
): Promise<void> {
  setSseHeaders(res);
  let closed = false;
  const onClose = () => {
    closed = true;
  };
  req.on('close', onClose);
  const emit = (event: string, data: unknown) => {
    if (!closed) writeSse(res, event, data);
  };
  try {
    writeSse(res, 'session', { agentSessionId: sessionId });
    const result = await run(async (chunk) => {
      emitChunkAsEvents(chunk, emit);
    });
    emit('complete', { status: result.status, agentSessionId: sessionId });
  } catch (error) {
    emit('error', { message: error instanceof Error ? error.message : String(error) });
  } finally {
    req.off('close', onClose);
    if (!closed) res.end();
  }
}

// --- handlers ----------------------------------------------------------------

export const chatHandler = async (req: Request, res: Response): Promise<void> => {
  const body = req.body as z.infer<typeof chatSchema>;
  // Ownership checked pre-headers so a foreign id still yields a JSON 404.
  const session = await ensureAgentSession(req.user!.id, body.agentSessionId);
  await streamTurn(req, res, session.id, async (streamHandler) => {
    const result = await runAgentGraph({
      userId: req.user!.id,
      agentSessionId: session.id,
      inputPrompt: body.prompt,
      tools: body.tools,
      streamHandler,
    });
    return { status: result.status };
  });
};

export const resumeHandler = async (req: Request, res: Response): Promise<void> => {
  const body = req.body as z.infer<typeof resumeSchema>;
  // Pre-header ownership check (resumeAgentGraph re-checks; this keeps the
  // 404 in JSON rather than an SSE error event).
  const session = await prisma.agentSession.findUnique({ where: { id: body.agentSessionId } });
  if (!session || session.userId !== req.user!.id) {
    throw new ApiError(404, 'Agent session not found');
  }
  await streamTurn(req, res, session.id, async (streamHandler) => {
    const result = await resumeAgentGraph({
      userId: req.user!.id,
      agentSessionId: session.id,
      approvalDecision: body.approved ? 'APPROVED' : 'REJECTED',
      streamHandler,
    });
    return { status: result.status };
  });
};

/** Catalog for the user-facing tool picker. No secrets, same for everyone. */
export const listToolsHandler = (_req: Request, res: Response): void => {
  res.json({ tools: describeAgentTools() });
};

export const listSessionsHandler = asyncHandler(async (req, res) => {
  const { page, pageSize } = sessionListSchema.parse(req.query);
  const where = { userId: req.user!.id };
  const [sessions, total] = await Promise.all([
    prisma.agentSession.findMany({
      where,
      orderBy: { updatedAt: 'desc' },
      skip: (page - 1) * pageSize,
      take: pageSize,
      include: { _count: { select: { executions: true } } },
    }),
    prisma.agentSession.count({ where }),
  ]);
  res.json({
    sessions: sessions.map((session) => ({
      ...session,
      executionCount: session._count.executions,
      _count: undefined,
    })),
    pagination: { page, pageSize, total },
  });
});

function serializeMessage(message: {
  id?: string;
  name?: string;
  getType: () => string;
  content: unknown;
  tool_calls?: ToolCallLike[];
  tool_call_id?: string;
}) {
  const base = {
    id: message.id,
    type: message.getType(),
    name: message.name,
    content: messageText(message.content),
  };
  if (base.type === 'ai' && message.tool_calls) {
    return { ...base, tool_calls: message.tool_calls };
  }
  if (base.type === 'tool' && message.tool_call_id) {
    return { ...base, tool_call_id: message.tool_call_id };
  }
  return base;
}

export const getSessionHandler = asyncHandler(async (req, res) => {
  const session = await prisma.agentSession.findFirst({
    where: { id: req.params.id!, userId: req.user!.id },
  });
  if (!session) throw new ApiError(404, 'Agent session not found');

  const snapshot = await agentGraph.getState({ configurable: { thread_id: session.id } });
  const values = snapshot.values as { messages?: Parameters<typeof serializeMessage>[0][] } | undefined;
  const executions = await prisma.execution.findMany({
    where: { agentSessionId: session.id, userId: req.user!.id },
    orderBy: { createdAt: 'desc' },
    select: {
      id: true,
      name: true,
      languageId: true,
      status: true,
      judgeStatus: true,
      exitCode: true,
      timeMs: true,
      createdAt: true,
      completedAt: true,
      artifacts: {
        select: { id: true, name: true, mimeType: true, sizeBytes: true, createdAt: true },
        orderBy: { createdAt: 'asc' },
      },
    },
  });

  res.json({
    session,
    messages: (values?.messages ?? []).map(serializeMessage),
    executions,
  });
});
