'use client';

import { useEffect, useRef, useState } from 'react';
import type { AgentToolInfo } from '@/lib/types';

/** execute_code_sandbox -> Execute Code Sandbox */
export function prettyToolName(name: string): string {
  return name
    .split('_')
    .map((word) => (word ? word[0].toUpperCase() + word.slice(1) : word))
    .join(' ');
}

interface ToolPickerProps {
  tools: AgentToolInfo[];
  /** Null = everything enabled (the default). */
  enabled: string[] | null;
  disabled?: boolean;
  onChange: (enabled: string[] | null) => void;
}

/**
 * ChatGPT-style tool toggles: which tools this turn may use. The popover
 * stays open for multi-toggling; selection persists via the parent
 * (localStorage). Null (all on) is the default and is sent as "no filter".
 */
export function ToolPicker({ tools, enabled, disabled, onChange }: ToolPickerProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (event: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [open ]);

  const allOn = enabled === null;
  const enabledCount = allOn ? tools.length : enabled.length;

  const toggle = (name: string) => {
    const current = new Set(allOn ? tools.map((tool) => tool.name) : enabled);
    if (current.has(name)) current.delete(name);
    else current.add(name);
    onChange(current.size >= tools.length ? null : [...current]);
  };

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((value) => !value)}
        title="Choose which tools the agent may use this turn"
        className="flex items-center gap-1 rounded-md border border-slate-300 px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
      >
        🛠 Tools
        <span className="rounded-full bg-slate-200 px-1.5 font-mono text-[11px] dark:bg-slate-800">
          {enabledCount}/{tools.length}
        </span>
      </button>
      {open && (
        <div className="absolute left-0 top-full z-20 mt-1 max-h-80 w-72 overflow-auto rounded-lg border border-slate-200 bg-white p-1 shadow-xl dark:border-slate-700 dark:bg-slate-900">
          <div className="flex items-center justify-between px-2 py-1">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
              Agent tools
            </span>
            <button
              type="button"
              onClick={() => onChange(null)}
              className="text-[11px] font-medium text-emerald-600 hover:underline dark:text-emerald-400"
            >
              Enable all
            </button>
          </div>
          {tools.map((tool) => {
            const on = allOn || enabled.includes(tool.name);
            return (
              <button
                key={tool.name}
                type="button"
                onClick={() => toggle(tool.name)}
                className="flex w-full items-start gap-2 rounded-md px-2 py-1.5 text-left hover:bg-slate-100 dark:hover:bg-slate-800"
              >
                <span
                  className={`mt-0.5 flex h-4 w-7 shrink-0 items-center rounded-full p-0.5 transition ${
                    on ? 'justify-end bg-emerald-500' : 'justify-start bg-slate-300 dark:bg-slate-700'
                  }`}
                >
                  <span className="h-3 w-3 rounded-full bg-white" />
                </span>
                <span className="min-w-0">
                  <span className="block truncate font-mono text-xs font-semibold text-slate-700 dark:text-slate-200">
                    {prettyToolName(tool.name)}
                  </span>
                  <span className="block truncate text-[11px] text-slate-400" title={tool.description}>
                    {tool.description}
                  </span>
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
