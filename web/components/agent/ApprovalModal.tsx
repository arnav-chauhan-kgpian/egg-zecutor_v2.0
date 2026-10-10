'use client';

import { useState } from 'react';

function formatArgs(input: unknown): string {
  if (input === undefined || input === null) return '—';
  const text = typeof input === 'string' ? input : JSON.stringify(input, null, 2);
  return text.length > 2000 ? `${text.slice(0, 2000)}\n…[truncated]` : text;
}

function ToolCallCard({ call, index }: { call: unknown; index: number }) {
  const record = (call ?? {}) as { name?: unknown; args?: unknown; tool?: unknown; input?: unknown };
  const name = String(record.name ?? record.tool ?? `tool ${index + 1}`);
  const args = 'args' in record ? record.args : record.input;
  const sensitive = /shell_executor|execute_code_sandbox/i.test(name);
  return (
    <div className="rounded-md border border-amber-200 bg-amber-50 p-2 dark:border-amber-900 dark:bg-amber-950/40">
      <p className="font-mono text-xs font-semibold text-amber-800 dark:text-amber-200">
        {index + 1}. <span className="rounded bg-amber-200/70 px-1 dark:bg-amber-900">{name}</span>
        {sensitive && <span className="ml-1 text-[10px] uppercase tracking-wide">sensitive</span>}
      </p>
      <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] text-amber-900 dark:text-amber-100">
        {formatArgs(args)}
      </pre>
    </div>
  );
}

interface ApprovalModalProps {
  open: boolean;
  pendingToolCalls: unknown[];
  busy: boolean;
  onApprove: (reason?: string) => void;
  onReject: (reason?: string) => void;
}

/**
 * Human-in-the-loop gate. Rendered over the chat pane whenever the backend
 * pauses at the approval interrupt (or the session sits SUSPENDED with pending
 * calls). Approve resumes tool execution; reject ends the turn.
 */
export function ApprovalModal({ open, pendingToolCalls, busy, onApprove, onReject }: ApprovalModalProps) {
  const [reason, setReason] = useState('');

  if (!open) return null;

  return (
    <div className="absolute inset-0 z-10 flex items-center justify-center bg-slate-950/50 p-4">
      <div className="w-full max-w-md rounded-lg border border-amber-300 bg-white shadow-xl dark:border-amber-800 dark:bg-slate-900">
        <div className="border-b border-amber-200 bg-amber-100 px-4 py-2 dark:border-amber-900 dark:bg-amber-950">
          <p className="text-sm font-semibold text-amber-900 dark:text-amber-100">
            ⚠ Agent requires permission to run a sensitive action
          </p>
        </div>
        <div className="max-h-64 space-y-2 overflow-auto px-4 py-3">
          {pendingToolCalls.length === 0 ? (
            <p className="text-xs text-slate-500 dark:text-slate-400">
              The agent paused for review, but no pending calls were attached. Approving resumes
              the run; rejecting ends it.
            </p>
          ) : (
            pendingToolCalls.map((call, index) => (
              <ToolCallCard key={index} call={call} index={index} />
            ))
          )}
          <input
            value={reason}
            onChange={(event) => setReason(event.target.value)}
            placeholder="Optional note for the agent…"
            className="w-full rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-800 placeholder:text-slate-400 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200"
          />
        </div>
        <div className="flex justify-end gap-2 border-t border-slate-200 px-4 py-3 dark:border-slate-800">
          <button
            type="button"
            disabled={busy}
            onClick={() => onReject(reason || undefined)}
            className="rounded-md bg-rose-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-rose-500 disabled:opacity-50"
          >
            {busy ? 'Working…' : 'Reject Action'}
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => onApprove(reason || undefined)}
            className="rounded-md bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-emerald-500 disabled:opacity-50"
          >
            {busy ? 'Working…' : 'Approve & Continue'}
          </button>
        </div>
      </div>
    </div>
  );
}
