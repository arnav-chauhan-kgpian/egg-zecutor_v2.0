/**
 * Backend-neutral execution contract.
 *
 * Both backends (Judge0 and the local Docker runner) speak these types, so the
 * service layer, the callback route and the frontend are identical regardless
 * of which one is configured.
 */

/** Everything needed to run one script once. */
export interface ExecutionSpec {
  code: string;
  languageId: number;
  stdin?: string | null;
  /** Base64-encoded zip unpacked into the working directory before the run. */
  additionalFiles?: string | null;
  /** Seconds of CPU time. Null means "use the engine default". */
  cpuTimeLimit?: number | null;
  /** Kilobytes. Null means "use the engine default". */
  memoryLimit?: number | null;
}

/**
 * The result of a finished run.
 *
 * `status` describes the ENGINE, not the program: a script that exits 1 or
 * times out is still COMPLETED, because the engine did its job. FAILED means
 * no verdict could be produced at all.
 */
export interface ExecutionOutcome {
  status: 'COMPLETED' | 'FAILED';
  /** Human-readable verdict, e.g. "Accepted", "Time Limit Exceeded". */
  judgeStatus: string;
  stdout: string | null;
  stderr: string | null;
  compileOutput: string | null;
  exitCode: number | null;
  timeMs: number | null;
  memoryKb: number | null;
  errorMessage?: string | null;
}

/**
 * What a backend returns when a run is started.
 *
 * `deferred` — the backend accepted the work and will report later, either by
 *   calling the webhook or by being polled with `token`.
 * `settled` — the backend ran it inline and the result is already here.
 */
export type StartResult =
  | { kind: 'deferred'; token: string }
  | { kind: 'settled'; outcome: ExecutionOutcome };

export interface ExecutionBackend {
  readonly kind: 'judge0' | 'docker' | 'native';
  /** True when results arrive via the webhook rather than polling. */
  readonly usesCallback: boolean;
  start(spec: ExecutionSpec): Promise<StartResult>;
  /** Fetch a deferred result by token. Undefined on backends that settle inline. */
  poll?(token: string): Promise<ExecutionOutcome | null>;
  health(): Promise<boolean>;
}

export function failedOutcome(reason: string): ExecutionOutcome {
  return {
    status: 'FAILED',
    judgeStatus: 'Internal Error',
    stdout: null,
    stderr: null,
    compileOutput: null,
    exitCode: null,
    timeMs: null,
    memoryKb: null,
    errorMessage: reason.slice(0, 4000),
  };
}
