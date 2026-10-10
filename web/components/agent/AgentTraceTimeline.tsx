'use client';

export interface TraceStep {
  id: string;
  label: string;
  detail?: string;
  state: 'done' | 'active' | 'idle';
}

function StepDot({ state }: { state: TraceStep['state'] }) {
  if (state === 'done') {
    return <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-emerald-500" />;
  }
  if (state === 'active') {
    return <span className="mt-1 h-2 w-2 shrink-0 animate-pulse rounded-full bg-sky-500" />;
  }
  return <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-slate-300 dark:bg-slate-700" />;
}

/**
 * Step-by-step run progress: Thinking… → tool executions → artifacts →
 * Response ready. Purely presentational; the parent derives steps from the
 * live SSE event flow.
 */
export function AgentTraceTimeline({ steps }: { steps: TraceStep[] }) {
  if (steps.length === 0) {
    return (
      <p className="px-3 py-2 text-xs text-slate-400 dark:text-slate-500">
        Ask the agent to run code, recall memory, or search the web.
      </p>
    );
  }
  return (
    <ol className="space-y-1 px-3 py-2">
      {steps.map((step) => (
        <li key={step.id} className="flex items-start gap-2">
          <StepDot state={step.state} />
          <div className="min-w-0">
            <p className="truncate text-xs font-medium text-slate-700 dark:text-slate-200">
              {step.label}
            </p>
            {step.detail && (
              <p className="truncate font-mono text-[11px] text-slate-400 dark:text-slate-500">
                {step.detail}
              </p>
            )}
          </div>
        </li>
      ))}
    </ol>
  );
}
