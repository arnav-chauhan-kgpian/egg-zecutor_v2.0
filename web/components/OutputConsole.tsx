'use client';

import { useEffect, useRef, useState } from 'react';
import { formatDuration, formatMemory } from '@/lib/languages';
import type { Execution } from '@/lib/types';
import { ArtifactList } from './ArtifactList';

type Tab = 'stdout' | 'stderr' | 'compile' | 'artifacts';

function StatusPill({ execution }: { execution: Execution }) {
  const { status, judgeStatus, exitCode } = execution;

  const tone =
    status === 'FAILED'
      ? 'bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300'
      : status === 'COMPLETED'
        ? exitCode === 0
          ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300'
          : 'bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300'
        : 'bg-sky-100 text-sky-700 dark:bg-sky-950 dark:text-sky-300';

  const label =
    status === 'PENDING' || status === 'PROCESSING' ? status.toLowerCase() : (judgeStatus ?? status);

  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${tone}`}>
      {label}
      {status === 'PROCESSING' && <span className="ml-1 animate-pulse">●</span>}
    </span>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span className="whitespace-nowrap text-xs text-slate-500 dark:text-slate-400">
      {label} <span className="font-mono text-slate-700 dark:text-slate-200">{value}</span>
    </span>
  );
}

/**
 * Monospace log pane.
 *
 * Auto-scrolls while a run is in flight — but only if the user hasn't scrolled
 * up to read something, otherwise following long logs is impossible.
 */
function LogPane({ text, live, empty }: { text: string | null; live: boolean; empty: string }) {
  const ref = useRef<HTMLPreElement>(null);
  const pinned = useRef(true);

  useEffect(() => {
    const node = ref.current;
    if (!node || !live || !pinned.current) return;
    node.scrollTop = node.scrollHeight;
  }, [text, live]);

  if (!text) {
    return (
      <div className="flex h-full items-center justify-center p-4 text-sm text-slate-400 dark:text-slate-500">
        {empty}
      </div>
    );
  }

  return (
    <pre
      ref={ref}
      onScroll={(event) => {
        const node = event.currentTarget;
        pinned.current = node.scrollHeight - node.scrollTop - node.clientHeight < 40;
      }}
      className="h-full overflow-auto whitespace-pre-wrap break-words p-3 font-mono text-xs leading-relaxed text-slate-800 dark:text-slate-200"
    >
      {text}
    </pre>
  );
}

interface OutputConsoleProps {
  execution: Execution | null;
  error: string | null;
  busy: boolean;
}

export function OutputConsole({ execution, error, busy }: OutputConsoleProps) {
  const [tab, setTab] = useState<Tab>('stdout');

  // Surface whichever stream actually has something when a run finishes, so a
  // crash isn't hidden behind an empty stdout tab.
  useEffect(() => {
    if (!execution || execution.status !== 'COMPLETED') return;
    if (execution.compileOutput) setTab('compile');
    else if (!execution.stdout && execution.stderr) setTab('stderr');
  }, [execution?.id, execution?.status]);

  const artifactCount = execution?.artifacts.length ?? 0;
  const live = execution?.status === 'PROCESSING' || execution?.status === 'PENDING';

  const tabs: Array<{ id: Tab; label: string; badge?: number }> = [
    { id: 'stdout', label: 'stdout' },
    { id: 'stderr', label: 'stderr' },
    { id: 'compile', label: 'compile' },
    { id: 'artifacts', label: 'artifacts', badge: artifactCount },
  ];

  return (
    <div className="flex h-full min-h-0 flex-col bg-white dark:bg-slate-900">
      <div className="flex h-11 shrink-0 items-center justify-between gap-3 border-b border-slate-200 px-3 dark:border-slate-800">
        <div className="flex items-center gap-1">
          {tabs.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setTab(item.id)}
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition ${
                tab === item.id
                  ? 'bg-slate-200 text-slate-900 dark:bg-slate-800 dark:text-slate-100'
                  : 'text-slate-500 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800'
              }`}
            >
              {item.label}
              {item.badge ? (
                <span className="ml-1 rounded bg-emerald-600 px-1 text-[10px] font-bold text-white">
                  {item.badge}
                </span>
              ) : null}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-3 overflow-hidden">
          {execution && (
            <>
              <Stat label="time" value={formatDuration(execution.timeMs)} />
              <Stat label="mem" value={formatMemory(execution.memoryKb)} />
              <Stat label="exit" value={execution.exitCode?.toString() ?? '—'} />
              <StatusPill execution={execution} />
            </>
          )}
        </div>
      </div>

      <div className="min-h-0 flex-1">
        {error ? (
          <div className="p-3 font-mono text-xs text-red-600 dark:text-red-400">{error}</div>
        ) : !execution ? (
          <div className="flex h-full items-center justify-center p-4 text-sm text-slate-400 dark:text-slate-500">
            {busy ? 'Queuing…' : 'Run a script to see output here.'}
          </div>
        ) : execution.status === 'FAILED' ? (
          <div className="space-y-2 p-3">
            <p className="font-mono text-xs text-red-600 dark:text-red-400">
              {execution.errorMessage ?? 'The engine could not produce a result.'}
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              This is an engine failure, not a failure of your script.
            </p>
          </div>
        ) : tab === 'artifacts' ? (
          <ArtifactList execution={execution} />
        ) : tab === 'compile' ? (
          <LogPane text={execution.compileOutput} live={false} empty="No compiler output." />
        ) : tab === 'stderr' ? (
          <LogPane text={execution.stderr} live={live} empty="Nothing on stderr." />
        ) : (
          <LogPane
            text={execution.stdout}
            live={live}
            empty={live ? 'Running…' : 'Nothing on stdout.'}
          />
        )}
      </div>
    </div>
  );
}
