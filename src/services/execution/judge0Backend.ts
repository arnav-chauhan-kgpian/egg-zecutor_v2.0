/**
 * Judge0 backend — the primary execution engine.
 *
 * Submissions are always created with `wait=false`, so the API never blocks on
 * a long research workload. Results come back two ways:
 *
 *   1. Webhook — when JUDGE0_CALLBACK_URL is set, Judge0 PUTs the finished
 *      submission to it. This is the fast path.
 *   2. Polling — a background poller reconciles anything the webhook missed
 *      (Judge0 retries a callback only a handful of times, and a container
 *      restart can lose one entirely).
 *
 * Everything is base64-encoded on the wire so arbitrary bytes in source,
 * stdin and program output survive the round trip.
 */
import { env } from '../../config/env';
import { ApiError } from '../../utils/ApiError';
import type { ExecutionBackend, ExecutionOutcome, ExecutionSpec, StartResult } from './types';

// Judge0 status ids — https://ce.judge0.com/#statuses-and-languages
export const Judge0Status = {
  IN_QUEUE: 1,
  PROCESSING: 2,
  ACCEPTED: 3,
  WRONG_ANSWER: 4,
  TIME_LIMIT_EXCEEDED: 5,
  COMPILATION_ERROR: 6,
  RUNTIME_ERROR_SIGSEGV: 7,
  RUNTIME_ERROR_SIGXFSZ: 8,
  RUNTIME_ERROR_SIGFPE: 9,
  RUNTIME_ERROR_SIGABRT: 10,
  RUNTIME_ERROR_NZEC: 11,
  RUNTIME_ERROR_OTHER: 12,
  INTERNAL_ERROR: 13,
  EXEC_FORMAT_ERROR: 14,
} as const;

/** A submission is still in flight while queued or running. */
export function isTerminal(statusId: number): boolean {
  return statusId > Judge0Status.PROCESSING;
}

export interface Judge0Submission {
  token?: string;
  status?: { id: number; description: string };
  stdout?: string | null;
  stderr?: string | null;
  compile_output?: string | null;
  message?: string | null;
  exit_code?: number | null;
  time?: string | null;
  memory?: number | null;
}

const encode = (value: string) => Buffer.from(value, 'utf8').toString('base64');
const decode = (value: string | null | undefined) =>
  value == null ? null : Buffer.from(value, 'base64').toString('utf8');

function headers(): Record<string, string> {
  const result: Record<string, string> = { 'Content-Type': 'application/json' };
  if (env.JUDGE0_API_KEY) {
    // RapidAPI style; harmless on self-hosted instances that ignore them.
    result['X-RapidAPI-Key'] = env.JUDGE0_API_KEY;
    if (env.JUDGE0_API_HOST) result['X-RapidAPI-Host'] = env.JUDGE0_API_HOST;
  }
  if (env.JUDGE0_AUTH_TOKEN) result['X-Auth-Token'] = env.JUDGE0_AUTH_TOKEN;
  return result;
}

async function judge0Fetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const baseUrl = env.JUDGE0_API_URL!.replace(/\/+$/, '');

  let response: Response;
  try {
    response = await fetch(`${baseUrl}${path}`, {
      ...init,
      headers: { ...headers(), ...(init.headers as Record<string, string> | undefined) },
      signal: AbortSignal.timeout(env.JUDGE0_REQUEST_TIMEOUT_MS),
    });
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    throw new ApiError(502, `Judge0 is unreachable: ${reason}`);
  }

  if (!response.ok) {
    const body = await response.text().catch(() => '');
    throw new ApiError(502, `Judge0 responded ${response.status}: ${body.slice(0, 500)}`);
  }

  return (await response.json()) as T;
}

/**
 * Maps a finished Judge0 submission onto the neutral outcome shape.
 *
 * Exported because the webhook route receives exactly this payload and needs
 * the same mapping without re-fetching.
 */
export function toOutcome(submission: Judge0Submission): ExecutionOutcome {
  const statusId = submission.status?.id ?? Judge0Status.INTERNAL_ERROR;
  const description = submission.status?.description ?? 'Internal Error';

  // Judge0 reports CPU seconds as a decimal string.
  const seconds = submission.time != null ? Number.parseFloat(submission.time) : NaN;

  return {
    // An internal error means Judge0 itself could not run the code — that is
    // an engine failure. Everything else (non-zero exit, TLE, compile error)
    // is a legitimate result of a successful run.
    status: statusId === Judge0Status.INTERNAL_ERROR ? 'FAILED' : 'COMPLETED',
    judgeStatus: description,
    stdout: decode(submission.stdout),
    stderr: decode(submission.stderr),
    compileOutput: decode(submission.compile_output),
    exitCode: submission.exit_code ?? null,
    timeMs: Number.isFinite(seconds) ? Math.round(seconds * 1000) : null,
    memoryKb: submission.memory ?? null,
    errorMessage:
      statusId === Judge0Status.INTERNAL_ERROR ? decode(submission.message) ?? description : null,
  };
}

/** Fields Judge0 should send back on the callback and on a poll. */
const RESULT_FIELDS =
  'token,status,stdout,stderr,compile_output,message,exit_code,time,memory';

/**
 * The callback URL handed to Judge0, with the shared secret attached.
 *
 * The callback route is unauthenticated by necessity — Judge0 has no user
 * session — so it is guarded by JUDGE0_CALLBACK_SECRET. Judge0 only ever GETs
 * back the URL it was given and cannot be told to send a custom header, so the
 * secret has to travel in the query string or it never arrives at all.
 *
 * Deriving it here rather than expecting JUDGE0_CALLBACK_URL to already carry
 * `?secret=` removes a silent failure mode: with the secret configured but
 * absent from the URL, every webhook is rejected 401 and the engine quietly
 * degrades to reconciler polling — which still produces results, so nothing
 * looks broken while the fast path is dead.
 *
 * A secret already present in the configured URL is left alone.
 */
function callbackUrl(): string | undefined {
  const base = env.JUDGE0_CALLBACK_URL;
  if (!base) return undefined;
  if (!env.JUDGE0_CALLBACK_SECRET) return base;

  try {
    const url = new URL(base);
    if (!url.searchParams.has('secret')) {
      url.searchParams.set('secret', env.JUDGE0_CALLBACK_SECRET);
    }
    return url.toString();
  } catch {
    // Not parseable as an absolute URL — hand it over untouched rather than
    // dropping the webhook entirely.
    return base;
  }
}

export const judge0Backend: ExecutionBackend = {
  kind: 'judge0',

  get usesCallback() {
    return Boolean(env.JUDGE0_CALLBACK_URL);
  },

  async start(spec: ExecutionSpec): Promise<StartResult> {
    const body: Record<string, unknown> = {
      language_id: spec.languageId,
      source_code: encode(spec.code),
      stdin: spec.stdin ? encode(spec.stdin) : undefined,
      // Judge0 unzips this into the working directory — datasets, fixtures,
      // multi-file projects.
      additional_files: spec.additionalFiles || undefined,
      cpu_time_limit: spec.cpuTimeLimit ?? env.JUDGE0_CPU_TIME_LIMIT,
      // Judge0 kills the run at wall_time_limit regardless of CPU time, so a
      // research job that blocks on I/O needs headroom above the CPU limit.
      wall_time_limit: Math.min(
        (spec.cpuTimeLimit ?? env.JUDGE0_CPU_TIME_LIMIT) + env.JUDGE0_WALL_TIME_MARGIN,
        env.JUDGE0_WALL_TIME_MAX,
      ),
      memory_limit: spec.memoryLimit ?? env.JUDGE0_MEMORY_LIMIT,
      callback_url: callbackUrl(),
    };

    for (const key of Object.keys(body)) {
      if (body[key] === undefined) delete body[key];
    }

    const created = await judge0Fetch<Judge0Submission>(
      '/submissions?base64_encoded=true&wait=false',
      { method: 'POST', body: JSON.stringify(body) },
    );

    if (!created.token) {
      throw new ApiError(502, 'Judge0 accepted the submission but returned no token');
    }

    return { kind: 'deferred', token: created.token };
  },

  /** Returns null while the submission is still queued or running. */
  async poll(token: string): Promise<ExecutionOutcome | null> {
    const submission = await judge0Fetch<Judge0Submission>(
      `/submissions/${encodeURIComponent(token)}?base64_encoded=true&fields=${RESULT_FIELDS}`,
    );

    const statusId = submission.status?.id ?? Judge0Status.IN_QUEUE;
    if (!isTerminal(statusId)) return null;

    return toOutcome(submission);
  },

  async health(): Promise<boolean> {
    try {
      await judge0Fetch<unknown>('/about');
      return true;
    } catch {
      return false;
    }
  },
};
