import { getExecution } from '../../executions/execution.service';
import { ExecutionStatus } from '../../../lib/enums';

/**
 * Session context bound by `getToolsForSession()`. Tools created via the bare
 * `allAgentTools` registry carry no context — user-scoped tools return a
 * plain-text error asking the caller to bind a session instead of throwing.
 */
export interface ToolContext {
  userId?: string;
  agentSessionId?: string;
}

/** Max wall-clock time a tool waits for an execution to settle. */
export const POLL_TIMEOUT_MS = 60_000;

/** stdout/stderr chars returned to the LLM per field (context-window guard). */
export const MAX_OUTPUT_CHARS = 8000;

const TERMINAL: Set<string> = new Set([ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]);

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Polls an execution until it reaches COMPLETED/FAILED or the timeout elapses.
 * Linear-ish backoff (500ms -> 5s cap). Never throws for a timeout — the
 * caller reports the in-flight status so the LLM can check back later.
 */
export async function pollExecutionToCompletion(
  executionId: string,
  userId: string,
  timeoutMs: number = POLL_TIMEOUT_MS,
) {
  let last = await getExecution(executionId, userId);
  const started = Date.now();
  let delay = 500;
  while (
    !TERMINAL.has(last.status) &&
    Date.now() - started < timeoutMs
  ) {
    await sleep(delay);
    last = await getExecution(executionId, userId);
    delay = Math.min(5000, Math.floor(delay * 1.5));
  }
  return last;
}

export function isTerminalStatus(status: string): boolean {
  return TERMINAL.has(status);
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB'];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[unit]}`;
}

export function truncate(text: string | null | undefined, max: number): string | null {
  if (text == null) return null;
  if (text.length <= max) return text;
  return `${text.slice(0, max)}\n…[truncated ${text.length - max} chars]`;
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Base64 has no `:` in its alphabet, so a marker line is unambiguous.
 * Mirrors the engine regex in src/services/execution/artifacts.ts.
 */
const MARKER = /^::artifact:([^:]+):(?:([^:]+):)?([A-Za-z0-9+/=]*)::$/;

export interface StrippedArtifact {
  name: string;
  mimeType: string | null;
  sizeBytes: number;
}

function base64SizeBytes(b64: string): number {
  const padding = b64.endsWith('==') ? 2 : b64.endsWith('=') ? 1 : 0;
  return Math.max(0, Math.floor((b64.length * 3) / 4) - padding);
}

/**
 * Defensive strip for stdout that still carries `::artifact::` markers
 * (e.g. a backend echoing raw output). Returns the cleaned text plus
 * one-line summaries to substitute for the huge base64 blobs.
 */
export function stripArtifactMarkers(stdout: string): {
  clean: string;
  found: StrippedArtifact[];
} {
  const kept: string[] = [];
  const found: StrippedArtifact[] = [];
  for (const line of stdout.split('\n')) {
    const match = line.trim().match(MARKER);
    if (!match) {
      kept.push(line);
      continue;
    }
    const [, name, mimeType, b64] = match;
    const sizeBytes = base64SizeBytes(b64 ?? '');
    found.push({ name, mimeType: mimeType ?? null, sizeBytes });
    kept.push(
      `[Artifact created: ${name} (${formatBytes(sizeBytes)}${mimeType ? `, ${mimeType}` : ''})]`,
    );
  }
  return { clean: kept.join('\n'), found };
}
