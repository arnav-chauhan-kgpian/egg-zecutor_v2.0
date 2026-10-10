import axios, { AxiosError } from 'axios';
import type {
  AgentSessionDetail,
  AgentSessionSummary,
  AgentToolInfo,
  AuthResponse,
  EngineInfo,
  Execution,
  ExecutionSummary,
  LanguageInfo,
  Pagination,
} from './types';

/**
 * Browser-facing API base URL. Inlined at build time by Next, so it must be a
 * URL the user's browser can reach (not a Docker service name).
 */
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:4000';

export const TOKEN_STORAGE_KEY = 'hrc.token';
export const USER_STORAGE_KEY = 'hrc.user';

export const api = axios.create({
  baseURL: `${API_URL}/api/v1`,
  headers: { 'Content-Type': 'application/json' },
  // Execution is asynchronous now — every call here returns promptly, so a
  // short timeout is correct. Long runs are observed by polling.
  timeout: 30_000,
  // Datasets go up as base64 in the JSON body.
  maxBodyLength: 64 * 1024 * 1024,
  maxContentLength: 64 * 1024 * 1024,
});

api.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const token = window.localStorage.getItem(TOKEN_STORAGE_KEY);
    if (token) config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

/** Unwraps the API's `{ error: { message, details } }` envelope into a message. */
export function apiErrorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const payload = error.response?.data as
      | { error?: { message?: string; details?: Array<{ field?: string; message?: string }> } }
      | undefined;

    const detail = payload?.error?.details
      ?.map((item) => (item.field ? `${item.field}: ${item.message}` : item.message))
      .filter(Boolean)
      .join('; ');

    if (payload?.error?.message) {
      return detail ? `${payload.error.message} (${detail})` : payload.error.message;
    }
    if (error.code === 'ECONNABORTED') return 'Request timed out.';
    if (error.code === 'ERR_NETWORK') return `Cannot reach the API at ${API_URL}.`;
    return error.message;
  }
  return error instanceof Error ? error.message : 'Something went wrong';
}

// --- engine -------------------------------------------------------------------

export async function fetchLanguages(): Promise<LanguageInfo[]> {
  const { data } = await api.get<{ languages: LanguageInfo[] }>('/executions/languages');
  return data.languages;
}

export async function fetchEngine(): Promise<EngineInfo> {
  const { data } = await api.get<{ engine: EngineInfo }>('/executions/engine');
  return data.engine;
}

// --- executions ---------------------------------------------------------------

export interface CreateExecutionPayload {
  code: string;
  languageId: number;
  name?: string;
  stdin?: string;
  /** Base64 zip, unpacked into the working directory before the run. */
  additionalFiles?: string;
  timeLimit?: number;
  memoryLimit?: number;
}

/** Queues a run. Resolves as soon as the engine accepts it (HTTP 202). */
export async function createExecution(payload: CreateExecutionPayload): Promise<Execution> {
  const { data } = await api.post<{ execution: Execution }>('/executions', payload);
  return data.execution;
}

export async function fetchExecution(id: string): Promise<Execution> {
  const { data } = await api.get<{ execution: Execution }>(`/executions/${id}`);
  return data.execution;
}

export async function fetchExecutions(
  page = 1,
  pageSize = 20,
): Promise<{ executions: ExecutionSummary[]; pagination: Pagination }> {
  const { data } = await api.get(`/executions?page=${page}&pageSize=${pageSize}`);
  return data;
}

export async function deleteExecution(id: string): Promise<void> {
  await api.delete(`/executions/${id}`);
}

/**
 * Artifact download URL.
 *
 * The route requires a bearer token, so this is fetched as a blob rather than
 * being handed straight to an <a href> or <img src>.
 */
export function artifactUrl(executionId: string, artifactId: string): string {
  return `${API_URL}/api/v1/executions/${executionId}/artifacts/${artifactId}`;
}

export async function fetchArtifactBlob(executionId: string, artifactId: string): Promise<Blob> {
  const { data } = await api.get(`/executions/${executionId}/artifacts/${artifactId}`, {
    responseType: 'blob',
  });
  return data as Blob;
}

// --- auth ---------------------------------------------------------------------

export async function login(identifier: string, password: string): Promise<AuthResponse> {
  const { data } = await api.post<AuthResponse>('/auth/login', { identifier, password });
  return data;
}

export async function register(
  email: string,
  username: string,
  password: string,
): Promise<AuthResponse> {
  const { data } = await api.post<AuthResponse>('/auth/register', { email, username, password });
  return data;
}

// --- agent --------------------------------------------------------------------

export async function fetchAgentSessions(
  page = 1,
  pageSize = 20,
): Promise<{ sessions: AgentSessionSummary[]; pagination: Pagination }> {
  const { data } = await api.get(`/agent/sessions?page=${page}&pageSize=${pageSize}`);
  return data;
}

export async function fetchAgentSessionDetail(sessionId: string): Promise<AgentSessionDetail> {
  const { data } = await api.get<AgentSessionDetail>(`/agent/sessions/${sessionId}`);
  return data;
}

/** Tool catalog backing the user-facing picker. Same for every user. */
export async function fetchAgentTools(): Promise<AgentToolInfo[]> {
  const { data } = await api.get<{ tools: AgentToolInfo[] }>('/agent/tools');
  return data.tools;
}

export type AgentEvent =
  | { type: 'session'; data: { agentSessionId: string } }
  | { type: 'message_delta'; data: string }
  | { type: 'tool_start'; data: { tool: string; input: unknown } }
  | { type: 'tool_end'; data: { tool?: string; output: string } }
  | { type: 'interrupt'; data: { approvalRequired: boolean; pendingToolCalls: unknown[] } }
  | { type: 'complete'; data: { status: string; agentSessionId: string } }
  | { type: 'error'; data: { message: string } };

export interface AgentStreamResult {
  agentSessionId: string | null;
  status: string | null;
}

function parseSseData(raw: string): unknown {
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

/**
 * Low-level SSE POST. Axios can't stream response bodies progressively, so this
 * uses fetch + a chunked line parser over the ReadableStream. Each blank-line
 * terminated block emits one AgentEvent; unknown event names pass through as
 * best-effort message_delta/error fallbacks are unnecessary — the server only
 * sends the documented set.
 */
export async function streamAgentRequest(
  path: '/agent/chat' | '/agent/resume',
  body: Record<string, unknown>,
  onEvent?: (event: AgentEvent) => void,
  onError?: (error: Error) => void,
  signal?: AbortSignal,
): Promise<AgentStreamResult> {
  const token = typeof window !== 'undefined' ? window.localStorage.getItem(TOKEN_STORAGE_KEY) : null;
  if (!token) throw new Error('Sign in to use the agent.');

  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/v1${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify(body),
      signal,
    });
  } catch (error) {
    const err = error instanceof Error ? error : new Error('Agent request failed.');
    onError?.(err);
    throw err;
  }

  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => '');
    let message = `Agent request failed (HTTP ${res.status}).`;
    try {
      const payload = JSON.parse(text) as { error?: { message?: string } };
      if (payload?.error?.message) message = payload.error.message;
    } catch {
      if (text) message = text.slice(0, 500);
    }
    const err = new Error(message);
    onError?.(err);
    throw err;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let eventName = '';
  let dataLines: string[] = [];
  let agentSessionId: string | null = null;
  let status: string | null = null;

  const dispatch = () => {
    const raw = dataLines.join('\n');
    dataLines = [];
    const name = eventName || 'message';
    eventName = '';
    if (!raw && name === 'message') return;
    const data = parseSseData(raw);
    const record = data as Record<string, unknown>;
    if (name === 'session' && record && typeof record.agentSessionId === 'string') {
      agentSessionId = record.agentSessionId;
    }
    if (name === 'complete' && record && typeof record.status === 'string') {
      status = record.status;
      if (typeof record.agentSessionId === 'string') agentSessionId = record.agentSessionId;
    }
    onEvent?.({ type: name, data } as AgentEvent);
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        const trimmed = line.endsWith('\r') ? line.slice(0, -1) : line;
        if (trimmed === '') {
          dispatch();
        } else if (trimmed.startsWith('event:')) {
          eventName = trimmed.slice(6).trim();
        } else if (trimmed.startsWith('data:')) {
          dataLines.push(trimmed.slice(5).trimStart());
        }
        // SSE comments (`: ...`) and other fields are ignored.
      }
    }
    if (dataLines.length > 0 || eventName) dispatch();
  } catch (error) {
    if (signal?.aborted) return { agentSessionId, status };
    const err = error instanceof Error ? error : new Error('Agent stream failed.');
    onError?.(err);
    throw err;
  } finally {
    reader.releaseLock();
  }
  return { agentSessionId, status };
}

/** Streams one chat turn; resolves with the session id and terminal status. */
export function streamAgentChat(
  prompt: string,
  agentSessionId?: string,
  enabledTools?: string[],
  onEvent?: (event: AgentEvent) => void,
  onError?: (error: Error) => void,
  signal?: AbortSignal,
): Promise<AgentStreamResult> {
  return streamAgentRequest(
    '/agent/chat',
    {
      prompt,
      ...(agentSessionId ? { agentSessionId } : {}),
      ...(enabledTools && enabledTools.length > 0 ? { tools: enabledTools } : {}),
    },
    onEvent,
    onError,
    signal,
  );
}

/**
 * Sends a HITL decision and streams the resumed turn. Returns the same
 * terminal payload as streamAgentChat so approve/reject plug into one pipeline.
 */
export function sendAgentResume(
  sessionId: string,
  approved: boolean,
  reason?: string,
  onEvent?: (event: AgentEvent) => void,
  onError?: (error: Error) => void,
  signal?: AbortSignal,
): Promise<AgentStreamResult> {
  return streamAgentRequest(
    '/agent/resume',
    reason !== undefined ? { agentSessionId: sessionId, approved, reason } : { agentSessionId: sessionId, approved },
    onEvent,
    onError,
    signal,
  );
}
