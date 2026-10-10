'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  apiErrorMessage,
  fetchAgentSessionDetail,
  fetchAgentSessions,
  fetchAgentTools,
  sendAgentResume,
  streamAgentChat,
  type AgentEvent,
} from '@/lib/api';
import type { AgentChatMessage, AgentSessionSummary, AgentToolInfo } from '@/lib/types';
import { useAuth } from '../AuthProvider';
import { ApprovalModal } from './ApprovalModal';
import { AgentTraceTimeline, type TraceStep } from './AgentTraceTimeline';
import { ToolPicker } from './ToolPicker';

const TOOL_SELECTION_KEY = 'hrc.agent-tools';

/** Catalog fallback if GET /agent/tools is unreachable; names must match the backend registry. */
const FALLBACK_TOOLS: AgentToolInfo[] = [
  { name: 'execute_code_sandbox', description: 'Run code in a sandbox and return output plus artifacts.' },
  { name: 'shell_executor', description: 'Run a bash shell command (needs approval).' },
  { name: 'read_artifact', description: 'Read a file produced by a previous run.' },
  { name: 'vector_rag_search', description: 'Recall related long-term memories.' },
  { name: 'save_memory_artifact', description: 'Persist a fact or finding for later recall.' },
  { name: 'web_search', description: 'Search the live web for current information.' },
  { name: 'calculator_engine', description: 'Exact arithmetic instead of mental math.' },
  { name: 'workspace_list_files', description: 'List this session’s runs and file artifacts.' },
];

interface ArtifactBadge {
  name: string;
  executionId?: string;
}

interface ChatItem {
  id: string;
  role: 'user' | 'assistant' | 'tool' | 'error';
  text: string;
  toolName?: string;
  toolInput?: unknown;
  pending?: boolean;
  executionId?: string;
  artifacts?: ArtifactBadge[];
}

function historyToItems(messages: AgentChatMessage[]): ChatItem[] {
  const callNames = new Map<string, string>();
  for (const message of messages) {
    if (message.type === 'ai' && message.tool_calls) {
      for (const call of message.tool_calls) {
        if (message.id && call.name) callNames.set(`${message.id}:${call.name}`, call.name);
      }
    }
  }
  const items: ChatItem[] = [];
  messages.forEach((message, index) => {
    const id = message.id ?? `hist-${index}`;
    if (message.type === 'human') {
      items.push({ id, role: 'user', text: message.content });
    } else if (message.type === 'ai') {
      if (message.content) items.push({ id, role: 'assistant', text: message.content });
    } else if (message.type === 'tool') {
      items.push({
        id,
        role: 'tool',
        text: message.content,
        toolName: message.name ?? 'tool',
        pending: false,
      });
    }
  });
  void callNames;
  return items;
}

/** Pulls execution/artifacts references out of a tool output for badges. */
function extractExecutionArtifacts(output: string): { executionId?: string; artifacts: ArtifactBadge[] } {
  try {
    const parsed = JSON.parse(output) as {
      executionId?: unknown;
      artifacts?: Array<{ name?: unknown }>;
    };
    if (parsed && typeof parsed === 'object') {
      const executionId = typeof parsed.executionId === 'string' ? parsed.executionId : undefined;
      const artifacts = Array.isArray(parsed.artifacts)
        ? parsed.artifacts
            .filter((a) => a && typeof a.name === 'string')
            .map((a) => ({ name: a.name as string, executionId }))
        : [];
      if (executionId || artifacts.length > 0) return { executionId, artifacts };
    }
  } catch {
    // Not JSON — fall through to marker-line scanning.
  }
  const badges: ArtifactBadge[] = [];
  const marker = /\[[Aa]rtifact created:\s*([^\]()]+?)(?:\s*\(.*?\))?\]/g;
  let match: RegExpExecArray | null;
  while ((match = marker.exec(output)) !== null) {
    badges.push({ name: match[1].trim() });
  }
  return { artifacts: badges };
}

function shortId(id: string): string {
  return id.length > 8 ? `${id.slice(0, 8)}…` : id;
}

function ToolCard({ item }: { item: ChatItem }) {
  const [open, setOpen] = useState(false);
  const preview = item.text.length > 1200 ? `${item.text.slice(0, 1200)}\n…[truncated]` : item.text;
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 dark:border-slate-700 dark:bg-slate-800/60">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2 px-2 py-1.5 text-left"
      >
        <span className={`h-2 w-2 shrink-0 rounded-full ${item.pending ? 'animate-pulse bg-sky-500' : 'bg-emerald-500'}`} />
        <span className="truncate font-mono text-xs font-semibold text-slate-700 dark:text-slate-200">
          {item.toolName ?? 'tool'}
        </span>
        <span className="ml-auto text-[11px] text-slate-400">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <div className="space-y-1 border-t border-slate-200 px-2 py-1.5 dark:border-slate-700">
          {item.toolInput !== undefined && (
            <pre className="max-h-32 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] text-slate-500 dark:text-slate-400">
              {JSON.stringify(item.toolInput, null, 2).slice(0, 2000)}
            </pre>
          )}
          <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] text-slate-700 dark:text-slate-200">
            {item.pending ? 'Running…' : preview}
          </pre>
        </div>
      )}
    </div>
  );
}

export function AgentChatPane({ onAgentExecution }: { onAgentExecution: (executionId: string) => void }) {
  const { user } = useAuth();

  const [sessions, setSessions] = useState<AgentSessionSummary[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionStatus, setSessionStatus] = useState<string | null>(null);
  const [items, setItems] = useState<ChatItem[]>([]);
  const [trace, setTrace] = useState<TraceStep[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [approvalBusy, setApprovalBusy] = useState(false);
  const [showApproval, setShowApproval] = useState(false);
  const [pendingCalls, setPendingCalls] = useState<unknown[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [catalog, setCatalog] = useState<AgentToolInfo[]>(FALLBACK_TOOLS);
  /** Null = all tools enabled (default). Persisted per browser. */
  const [enabledTools, setEnabledTools] = useState<string[] | null>(null);

  const idRef = useRef(0);
  const streamItemIdRef = useRef<string | null>(null);
  const stepRef = useRef(0);
  const lastLoadedExecutionRef = useRef<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const nextId = useCallback(() => {
    idRef.current += 1;
    return `local-${idRef.current}`;
  }, []);

  const pushStep = useCallback((label: string, detail?: string) => {
    stepRef.current += 1;
    const id = `step-${stepRef.current}`;
    setTrace((prev) => [
      ...prev.map((step) => (step.state === 'active' ? { ...step, state: 'done' as const } : step)),
      { id, label, detail, state: 'active' as const },
    ]);
    return id;
  }, []);

  const finishSteps = useCallback((finalLabel = 'Response ready') => {
    setTrace((prev) => [
      ...prev.map((step) => ({ ...step, state: 'done' as const })),
      { id: `step-done-${Date.now()}`, label: finalLabel, state: 'done' as const },
    ]);
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      const result = await fetchAgentSessions(1, 20);
      setSessions(result.sessions);
    } catch {
      // Session list is a nicety; chat works without it.
    }
  }, []);

  useEffect(() => {
    void refreshSessions();
    // Tool catalog + persisted selection. Unknown persisted names (registry
    // changed since) are dropped; an empty remainder means "all".
    fetchAgentTools()
      .then((tools) => {
        if (tools.length > 0) setCatalog(tools);
      })
      .catch(() => {
        // FALLBACK_TOOLS stays.
      });
    try {
      const stored = window.localStorage.getItem(TOOL_SELECTION_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as unknown;
        if (Array.isArray(parsed) && parsed.every((name) => typeof name === 'string')) {
          setEnabledTools(parsed.length > 0 ? parsed : null);
        }
      }
    } catch {
      window.localStorage.removeItem(TOOL_SELECTION_KEY);
    }
  }, [refreshSessions]);

  useEffect(() => {
    const node = scrollRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [items]);

  const loadSession = useCallback(
    async (id: string) => {
      setSessionId(id);
      setShowApproval(false);
      setPendingCalls([]);
      try {
        const detail = await fetchAgentSessionDetail(id);
        setSessionStatus(detail.session.status);
        setItems(historyToItems(detail.messages));
        if (detail.session.status === 'SUSPENDED') setShowApproval(true);
      } catch (err) {
        setError(apiErrorMessage(err));
      }
    },
    [],
  );

  const syncExecutions = useCallback(
    async (id: string) => {
      try {
        const detail = await fetchAgentSessionDetail(id);
        setSessionStatus(detail.session.status);
        const latest = detail.executions[0];
        if (latest && latest.id !== lastLoadedExecutionRef.current) {
          lastLoadedExecutionRef.current = latest.id;
          onAgentExecution(latest.id);
        }
      } catch {
        // OutputConsole sync is best-effort.
      }
    },
    [onAgentExecution],
  );

  const handleEvent = useCallback(
    (event: AgentEvent) => {
      switch (event.type) {
        case 'session': {
          if (event.data.agentSessionId) setSessionId(event.data.agentSessionId);
          break;
        }
        case 'message_delta': {
          const chunk = event.data;
          const targetId = streamItemIdRef.current;
          if (targetId) {
            setItems((prev) =>
              prev.map((item) => (item.id === targetId ? { ...item, text: item.text + chunk } : item)),
            );
          } else {
            const id = `a-${Date.now()}-${idRef.current}`;
            streamItemIdRef.current = id;
            setItems((prev) => [...prev, { id, role: 'assistant', text: chunk }]);
          }
          break;
        }
        case 'tool_start': {
          streamItemIdRef.current = null;
          pushStep(`Executing ${event.data.tool}`, JSON.stringify(event.data.input).slice(0, 120));
          setItems((prev) => [
            ...prev,
            {
              id: `t-${Date.now()}-${prev.length}`,
              role: 'tool',
              text: '',
              toolName: event.data.tool,
              toolInput: event.data.input,
              pending: true,
            },
          ]);
          break;
        }
        case 'tool_end': {
          const { artifacts, executionId } = extractExecutionArtifacts(event.data.output);
          for (const badge of artifacts) {
            pushStep(`Artifact generated: ${badge.name}`, badge.executionId ? shortId(badge.executionId) : undefined);
          }
          setItems((prev) => {
            const index = [...prev].map((item) => item.toolName).lastIndexOf(event.data.tool);
            if (index === -1) {
              return [
                ...prev,
                {
                  id: `t-${Date.now()}-${prev.length}`,
                  role: 'tool',
                  text: event.data.output,
                  toolName: event.data.tool,
                  pending: false,
                  executionId,
                  artifacts,
                },
              ];
            }
            const next = [...prev];
            next[index] = { ...next[index], text: event.data.output, pending: false, executionId, artifacts };
            return next;
          });
          break;
        }
        case 'interrupt': {
          streamItemIdRef.current = null;
          setSessionStatus('SUSPENDED');
          setPendingCalls(event.data.pendingToolCalls ?? []);
          setShowApproval(true);
          pushStep('Paused for approval', `${(event.data.pendingToolCalls ?? []).length} call(s)`);
          break;
        }
        case 'complete': {
          streamItemIdRef.current = null;
          setSessionStatus(event.data.status);
          setShowApproval(false);
          finishSteps(event.data.status === 'SUSPENDED' ? 'Paused — awaiting approval' : 'Response ready');
          if (event.data.agentSessionId) {
            setSessionId(event.data.agentSessionId);
            void syncExecutions(event.data.agentSessionId);
          }
          void refreshSessions();
          break;
        }
        case 'error': {
          streamItemIdRef.current = null;
          setItems((prev) => [...prev, { id: `e-${Date.now()}`, role: 'error', text: event.data.message }]);
          break;
        }
      }
    },
    [finishSteps, pushStep, refreshSessions, syncExecutions],
  );

  const send = useCallback(
    async (prompt: string) => {
      const text = prompt.trim();
      if (!user) {
        setError('Sign in to use the agent.');
        return;
      }
      if (!text || busy) return;
      setError(null);
      setBusy(true);
      setTrace([]);
      streamItemIdRef.current = null;
      setItems((prev) => [...prev, { id: nextId(), role: 'user', text }]);
      pushStep('Thinking…');
      // Subset-only payload: null or the full catalog means "all tools".
      const known = new Set(catalog.map((tool) => tool.name));
      const subset = enabledTools?.filter((name) => known.has(name));
      const toolsParam = subset && subset.length < catalog.length ? subset : undefined;
      try {
        const result = await streamAgentChat(text, sessionId ?? undefined, toolsParam, handleEvent);
        if (result.agentSessionId) setSessionId(result.agentSessionId);
      } catch (err) {
        setItems((prev) => [...prev, { id: nextId(), role: 'error', text: apiErrorMessage(err) }]);
      } finally {
        setBusy(false);
      }
    },
    [busy, catalog, enabledTools, handleEvent, nextId, pushStep, sessionId, user],
  );

  const changeTools = useCallback(
    (next: string[] | null) => {
      setEnabledTools(next);
      try {
        if (next === null) window.localStorage.removeItem(TOOL_SELECTION_KEY);
        else window.localStorage.setItem(TOOL_SELECTION_KEY, JSON.stringify(next));
      } catch {
        // Persistence is a nicety.
      }
    },
    [],
  );

  const decide = useCallback(
    async (approved: boolean, reason?: string) => {
      if (!sessionId || approvalBusy) return;
      setApprovalBusy(true);
      setError(null);
      try {
        await sendAgentResume(sessionId, approved, reason, handleEvent);
      } catch (err) {
        setItems((prev) => [...prev, { id: nextId(), role: 'error', text: apiErrorMessage(err) }]);
      } finally {
        setApprovalBusy(false);
      }
    },
    [approvalBusy, handleEvent, nextId, sessionId],
  );

  return (
    <div className="relative flex h-full min-h-0 flex-col bg-white dark:bg-slate-900">
      <div className="flex h-11 shrink-0 items-center gap-2 border-b border-slate-200 px-3 dark:border-slate-800">
        <select
          value={sessionId ?? ''}
          onChange={(event) => {
            const id = event.target.value;
            if (id) void loadSession(id);
          }}
          className="max-w-56 truncate rounded-md border border-slate-300 bg-white px-2 py-1 font-mono text-xs text-slate-700 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200"
        >
          <option value="">Select session…</option>
          {sessions.map((session) => (
            <option key={session.id} value={session.id}>
              {shortId(session.id)} · {session.status}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={() => {
            setSessionId(null);
            setSessionStatus(null);
            setItems([]);
            setTrace([]);
            setShowApproval(false);
            setPendingCalls([]);
            lastLoadedExecutionRef.current = null;
          }}
          className="rounded-md border border-slate-300 px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
        >
          + New Session
        </button>
        <ToolPicker tools={catalog} enabled={enabledTools} disabled={busy} onChange={changeTools} />
        <span className="ml-auto flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
          {sessionStatus && (
            <span className="rounded-full bg-slate-200 px-2 py-0.5 font-mono text-[11px] dark:bg-slate-800">
              {sessionStatus}
            </span>
          )}
          {busy ? (
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 animate-pulse rounded-full bg-sky-500" /> reasoning…
            </span>
          ) : (
            <span>idle</span>
          )}
        </span>
      </div>

      <div className="shrink-0 border-b border-slate-200 dark:border-slate-800">
        <AgentTraceTimeline steps={trace} />
      </div>

      <div ref={scrollRef} className="min-h-0 flex-1 space-y-2 overflow-auto p-3">
        {items.length === 0 && !busy && (
          <p className="text-sm text-slate-400 dark:text-slate-500">
            Describe a research task — the agent plans, runs sandboxed code, and reports back with
            artifacts.
          </p>
        )}
        {items.map((item) => {
          if (item.role === 'user') {
            return (
              <div key={item.id} className="flex justify-end">
                <p className="max-w-[85%] whitespace-pre-wrap rounded-lg bg-sky-600 px-3 py-1.5 text-sm text-white">
                  {item.text}
                </p>
              </div>
            );
          }
          if (item.role === 'error') {
            return (
              <p key={item.id} className="rounded-md bg-red-50 px-3 py-1.5 font-mono text-xs text-red-600 dark:bg-red-950 dark:text-red-300">
                {item.text}
              </p>
            );
          }
          if (item.role === 'tool') {
            return (
              <div key={item.id} className="space-y-1">
                <ToolCard item={item} />
                {(item.artifacts ?? []).length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {(item.artifacts ?? []).map((badge, index) => (
                      <button
                        key={`${badge.name}-${index}`}
                        type="button"
                        disabled={!badge.executionId}
                        onClick={() => badge.executionId && onAgentExecution(badge.executionId)}
                        className="rounded-full bg-emerald-100 px-2 py-0.5 font-mono text-[11px] text-emerald-800 hover:bg-emerald-200 disabled:cursor-default disabled:opacity-70 dark:bg-emerald-950 dark:text-emerald-200"
                      >
                        📎 {badge.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          }
          return (
            <div key={item.id} className="prose prose-sm max-w-none dark:prose-invert">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{item.text || '…'}</ReactMarkdown>
            </div>
          );
        })}
        {error && <p className="font-mono text-xs text-red-600 dark:text-red-400">{error}</p>}
      </div>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          void send(input);
          setInput('');
        }}
        className="flex shrink-0 gap-2 border-t border-slate-200 p-2 dark:border-slate-800"
      >
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              const text = input;
              setInput('');
              void send(text);
            }
          }}
          rows={2}
          placeholder={user ? 'Ask the agent…' : 'Sign in to chat with the agent…'}
          className="min-h-10 flex-1 resize-none rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm text-slate-800 placeholder:text-slate-400 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="rounded-md bg-emerald-600 px-4 text-sm font-semibold text-white transition hover:bg-emerald-500 disabled:opacity-50"
        >
          {busy ? '…' : 'Send'}
        </button>
      </form>

      <ApprovalModal
        open={showApproval}
        pendingToolCalls={pendingCalls}
        busy={approvalBusy}
        onApprove={(reason) => void decide(true, reason)}
        onReject={(reason) => void decide(false, reason)}
      />
    </div>
  );
}
