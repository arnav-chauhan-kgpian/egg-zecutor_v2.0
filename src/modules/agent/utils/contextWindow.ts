import type { AIMessage, BaseMessage, ToolMessage } from '@langchain/core/messages';

/** Rough chars-per-token for English + tool JSON. */
export const CHARS_PER_TOKEN = 4;
/** Default budget: ~100k tokens. */
export const DEFAULT_MAX_CHARS = 400_000;
/** Recent history always kept for the prompt (~10 turns). */
export const DEFAULT_KEEP_RECENT_MESSAGES = 20;
/** Absolute floor — never prune down to fewer than this many tail messages. */
const MIN_TAIL_MESSAGES = 4;

/** Approximate serialized size of one message (content + tool calls). */
export function messageChars(message: BaseMessage): number {
  const content = message.content;
  const contentChars =
    typeof content === 'string'
      ? content.length
      : JSON.stringify(content ?? '').length;
  let extra = 0;
  const asAi = message as AIMessage;
  if (Array.isArray(asAi.tool_calls) && asAi.tool_calls.length > 0) {
    extra = JSON.stringify(asAi.tool_calls).length;
  }
  const asTool = message as ToolMessage;
  if (typeof asTool.tool_call_id === 'string') extra += asTool.tool_call_id.length;
  return contentChars + extra + (message.name?.length ?? 0) + 64;
}

export function totalChars(messages: BaseMessage[]): number {
  return messages.reduce((sum, message) => sum + messageChars(message), 0);
}

export function approxTokens(messages: BaseMessage[]): number {
  return Math.ceil(totalChars(messages) / CHARS_PER_TOKEN);
}

export interface PruneOptions {
  maxChars?: number;
  keepRecentMessages?: number;
}

function isSystemMessage(message: BaseMessage): boolean {
  return message.getType() === 'system';
}

function isToolMessage(message: BaseMessage): boolean {
  return message.getType() === 'tool';
}

/**
 * Non-destructive windowing for the model prompt.
 *
 * Keeps leading system messages + the most recent N messages verbatim. If the
 * window is still over budget, oldest ToolMessages outside the protected tail
 * are dropped first (tool outputs are the bulkiest and most replaceable),
 * then oldest non-system messages. Never mutates state — the checkpointer
 * keeps full history; only the prompt is trimmed.
 */
export function pruneMessagesForModel(
  messages: BaseMessage[],
  options: PruneOptions = {},
): BaseMessage[] {
  const maxChars = options.maxChars ?? DEFAULT_MAX_CHARS;
  const keepRecent = options.keepRecentMessages ?? DEFAULT_KEEP_RECENT_MESSAGES;
  if (messages.length === 0 || totalChars(messages) <= maxChars) return messages;

  const systems = messages.filter(isSystemMessage);
  const rest = messages.filter((message) => !isSystemMessage(message));
  let window = rest.slice(-keepRecent);
  if (totalChars([...systems, ...window]) <= maxChars) return [...systems, ...window];

  // Phase 2: drop oldest tool outputs outside the protected tail.
  const tailCount = Math.min(MIN_TAIL_MESSAGES, window.length);
  const head = window.slice(0, window.length - tailCount);
  const tail = window.slice(window.length - tailCount);
  const keptHead = head.filter((message) => !isToolMessage(message));
  window = [...keptHead, ...tail];
  if (totalChars([...systems, ...window]) <= maxChars) return [...systems, ...window];

  // Phase 3: drop oldest non-system messages until under budget or at floor.
  while (
    window.length > MIN_TAIL_MESSAGES &&
    totalChars([...systems, ...window]) > maxChars
  ) {
    window = window.slice(1);
  }
  return [...systems, ...window];
}
