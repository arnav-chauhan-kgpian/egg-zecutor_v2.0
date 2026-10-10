'use client';

import dynamic from 'next/dynamic';
import { editorLanguage } from '@/lib/languages';
import type { LanguageInfo } from '@/lib/types';
import { useTheme } from './ThemeProvider';

// Monaco touches `window`/`navigator`, so it must not render on the server.
const MonacoEditor = dynamic(() => import('@monaco-editor/react'), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center bg-slate-100 text-sm text-slate-500 dark:bg-slate-950 dark:text-slate-400">
      Loading editor…
    </div>
  ),
});

interface CodeEditorPaneProps {
  code: string;
  languageId: number;
  languages: LanguageInfo[];
  running: boolean;
  onCodeChange: (code: string) => void;
  onLanguageChange: (languageId: number) => void;
  onReset: () => void;
  onRun: () => void;
}

export function CodeEditorPane({
  code,
  languageId,
  languages,
  running,
  onCodeChange,
  onLanguageChange,
  onReset,
  onRun,
}: CodeEditorPaneProps) {
  const { theme } = useTheme();

  return (
    <div className="flex h-full min-h-0 flex-col bg-white dark:bg-slate-900">
      <div className="flex h-11 shrink-0 items-center justify-between gap-2 border-b border-slate-200 px-3 dark:border-slate-800">
        <div className="flex min-w-0 items-center gap-2">
          <label htmlFor="language" className="sr-only">
            Language
          </label>
          <select
            id="language"
            value={languageId}
            onChange={(event) => onLanguageChange(Number(event.target.value))}
            className="max-w-56 truncate rounded-md border border-slate-300 bg-white px-2 py-1 text-sm font-medium text-slate-800 focus:border-slate-500 focus:outline-none dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
          >
            {languages.map((language) => (
              <option key={language.id} value={language.id}>
                {language.name}
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={onReset}
            title="Reset to the starter script"
            className="rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-600 transition hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
          >
            Reset
          </button>
        </div>

        <button
          type="button"
          onClick={onRun}
          disabled={running}
          className="rounded-md bg-emerald-600 px-4 py-1.5 text-sm font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {running ? 'Running…' : 'Run'}
        </button>
      </div>

      <div className="min-h-0 flex-1">
        <MonacoEditor
          height="100%"
          language={editorLanguage(languageId).monaco}
          theme={theme === 'dark' ? 'vs-dark' : 'vs'}
          value={code}
          onChange={(value) => onCodeChange(value ?? '')}
          options={{
            fontSize: 13,
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            automaticLayout: true,
            tabSize: 4,
            renderWhitespace: 'selection',
            smoothScrolling: true,
            padding: { top: 12, bottom: 12 },
          }}
        />
      </div>
    </div>
  );
}
