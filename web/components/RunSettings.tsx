'use client';

import { useRef, useState } from 'react';
import { formatBytes } from '@/lib/languages';

export interface RunConfig {
  stdin: string;
  timeLimit: number | null;
  memoryLimitMb: number | null;
  /** Base64 zip, unpacked into the working directory before the run. */
  additionalFiles: string | null;
  additionalFilesName: string | null;
}

export const DEFAULT_RUN_CONFIG: RunConfig = {
  stdin: '',
  timeLimit: null,
  memoryLimitMb: null,
  additionalFiles: null,
  additionalFilesName: null,
};

function NumberField({
  label,
  hint,
  value,
  placeholder,
  onChange,
}: {
  label: string;
  hint: string;
  value: number | null;
  placeholder: string;
  onChange: (value: number | null) => void;
}) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-slate-600 dark:text-slate-300">{label}</span>
      <input
        type="number"
        min={1}
        value={value ?? ''}
        placeholder={placeholder}
        onChange={(event) => {
          const raw = event.target.value.trim();
          onChange(raw === '' ? null : Number(raw));
        }}
        className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2 py-1 font-mono text-sm text-slate-800 focus:border-slate-500 focus:outline-none dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
      />
      <span className="mt-0.5 block text-[11px] text-slate-400 dark:text-slate-500">{hint}</span>
    </label>
  );
}

export function RunSettings({
  config,
  onChange,
}: {
  config: RunConfig;
  onChange: (config: RunConfig) => void;
}) {
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const set = <K extends keyof RunConfig>(key: K, value: RunConfig[K]) =>
    onChange({ ...config, [key]: value });

  /** Reads a .zip into base64 for Judge0's `additional_files`. */
  function onPickZip(file: File | undefined) {
    if (!file) return;
    setError(null);

    if (!/\.zip$/i.test(file.name)) {
      setError('additional_files must be a .zip archive.');
      return;
    }

    const reader = new FileReader();
    reader.onerror = () => setError('Could not read that file.');
    reader.onload = () => {
      // readAsDataURL gives "data:<mime>;base64,<payload>" — keep the payload.
      const result = String(reader.result ?? '');
      const base64 = result.slice(result.indexOf(',') + 1);
      onChange({ ...config, additionalFiles: base64, additionalFilesName: file.name });
    };
    reader.readAsDataURL(file);
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-4 overflow-auto p-3">
      <div>
        <label
          htmlFor="stdin"
          className="text-xs font-medium text-slate-600 dark:text-slate-300"
        >
          stdin
        </label>
        <textarea
          id="stdin"
          value={config.stdin}
          onChange={(event) => set('stdin', event.target.value)}
          rows={6}
          spellCheck={false}
          placeholder="Piped to the process on stdin."
          className="mt-1 w-full resize-y rounded-md border border-slate-300 bg-white p-2 font-mono text-xs text-slate-800 focus:border-slate-500 focus:outline-none dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
        />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <NumberField
          label="CPU time limit"
          hint="seconds — blank uses the engine default"
          value={config.timeLimit}
          placeholder="default"
          onChange={(value) => set('timeLimit', value)}
        />
        <NumberField
          label="Memory limit"
          hint="MB — blank uses the engine default"
          value={config.memoryLimitMb}
          placeholder="default"
          onChange={(value) => set('memoryLimitMb', value)}
        />
      </div>

      <div>
        <span className="text-xs font-medium text-slate-600 dark:text-slate-300">
          additional_files
        </span>
        <p className="mt-0.5 text-[11px] text-slate-400 dark:text-slate-500">
          A .zip unpacked into the working directory before the run — datasets, fixtures,
          multi-file projects.
        </p>

        <div className="mt-2 flex items-center gap-2">
          <input
            ref={fileInput}
            type="file"
            accept=".zip,application/zip"
            onChange={(event) => onPickZip(event.target.files?.[0])}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInput.current?.click()}
            className="rounded-md border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-700 transition hover:bg-slate-100 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800"
          >
            Choose .zip
          </button>

          {config.additionalFilesName && (
            <>
              <span className="min-w-0 truncate font-mono text-xs text-slate-600 dark:text-slate-300">
                {config.additionalFilesName}
                {config.additionalFiles && (
                  <span className="ml-1 text-slate-400">
                    ({formatBytes(Math.floor((config.additionalFiles.length * 3) / 4))})
                  </span>
                )}
              </span>
              <button
                type="button"
                onClick={() => {
                  if (fileInput.current) fileInput.current.value = '';
                  onChange({ ...config, additionalFiles: null, additionalFilesName: null });
                }}
                className="text-xs text-red-600 hover:underline dark:text-red-400"
              >
                clear
              </button>
            </>
          )}
        </div>

        {error && <p className="mt-1 text-xs text-red-600 dark:text-red-400">{error}</p>}
      </div>
    </div>
  );
}
