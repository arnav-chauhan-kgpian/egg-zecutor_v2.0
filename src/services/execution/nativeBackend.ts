/**
 * Native backend — runs code as an ordinary child process of the API.
 *
 * This exists for one reason: to make the engine work with **no Docker and no
 * Judge0**. `npm install && npm run dev` and nothing else. Whatever
 * interpreters and compilers are already on PATH are the language set.
 *
 * ============================ SECURITY ============================
 * There is NO SANDBOX HERE. Submitted code runs with the full privileges of
 * the API process: it can read and write your files, open sockets, and spawn
 * processes. The limits below (wall-clock timeout, captured-output cap) exist
 * to stop *accidents*, not attackers — nothing here contains hostile code.
 *
 * That is an acceptable trade only when the person writing the code and the
 * person running the server are the same person. The server therefore refuses
 * to listen on a non-loopback address while this backend is active unless the
 * operator explicitly overrides it (see EXECUTOR_NATIVE_ALLOW_REMOTE).
 *
 * For anything multi-user or internet-facing, use the Docker or Judge0 backend.
 * ==================================================================
 *
 * Compared with the sandboxed backends this one cannot:
 *   - enforce a memory limit    (no cgroup, so memoryKb is reported as null)
 *   - report CPU time           (timeMs is wall-clock, which is what we have)
 *   - unpack `additionalFiles`  (no unzip is guaranteed to exist; rejected
 *                                explicitly rather than silently ignored)
 */
import { spawn, spawnSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { env } from '../../config/env';
import { NATIVE_LANGUAGES, type NativeLanguageSpec } from './languages';
import {
  failedOutcome,
  type ExecutionBackend,
  type ExecutionSpec,
  type StartResult,
} from './types';

const isWindows = process.platform === 'win32';
/** Reported as the exit code when compilation fails, matching dockerBackend. */
const COMPILE_FAILED = 97;

/**
 * First candidate that exists on PATH, or null.
 *
 * Cached because resolution shells out, and a busy engine would otherwise probe
 * the filesystem on every single run.
 */
const resolved = new Map<string, string | null>();

function resolveTool(candidates: string[]): string | null {
  const key = candidates.join('|');
  const cached = resolved.get(key);
  if (cached !== undefined) return cached;

  let found: string | null = null;
  for (const candidate of candidates) {
    // Actually run the tool rather than just locating it. Windows ships an
    // App Execution Alias at WindowsApps/python3.exe that exists on PATH even
    // when Python is not installed — invoking it opens the Microsoft Store and
    // exits non-zero. A `where`/`command -v` hit would pick that stub and every
    // Python run would fail bizarrely, so require a working --version instead.
    const probe = spawnSync(candidate, ['--version'], {
      stdio: 'ignore',
      timeout: 5_000,
      shell: false,
    });

    if (probe.status === 0) {
      found = candidate;
      break;
    }
  }

  resolved.set(key, found);
  return found;
}

/** Which of the native languages this machine can actually run, by id. */
export function availableNativeLanguages(): number[] {
  return Object.entries(NATIVE_LANGUAGES)
    .filter(([, spec]) => toolFor(spec) !== null)
    .map(([id]) => Number(id));
}

function toolFor(spec: NativeLanguageSpec): string | null {
  const candidates = spec.compiler?.candidates ?? spec.interpreter?.candidates ?? [];
  return resolveTool(candidates);
}

interface RawRun {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  timedOut: boolean;
  spawnError: string | null;
}

function execute(
  command: string,
  args: string[],
  options: { cwd: string; timeoutMs: number; stdin?: string | null },
): Promise<RawRun> {
  return new Promise((resolve) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      stdio: ['pipe', 'pipe', 'pipe'],
      // Detached on Unix puts the child in its own process group, so a timeout
      // can kill the whole tree rather than just the parent — a compiler that
      // forks, or a script that spawns helpers, would otherwise survive.
      detached: !isWindows,
      env: { ...process.env, NODE_OPTIONS: '' },
    });

    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];
    let stdoutBytes = 0;
    let stderrBytes = 0;
    let timedOut = false;
    let spawnError: string | null = null;

    // Cap captured output so a runaway print loop cannot exhaust the API's heap.
    child.stdout.on('data', (chunk: Buffer) => {
      if (stdoutBytes >= env.EXECUTOR_MAX_OUTPUT_BYTES) return;
      stdoutChunks.push(chunk);
      stdoutBytes += chunk.length;
    });
    child.stderr.on('data', (chunk: Buffer) => {
      if (stderrBytes >= env.EXECUTOR_MAX_OUTPUT_BYTES) return;
      stderrChunks.push(chunk);
      stderrBytes += chunk.length;
    });

    const timer = setTimeout(() => {
      timedOut = true;
      if (child.pid === undefined) return;

      if (isWindows) {
        // Windows has no process groups to signal; taskkill /T walks the tree.
        spawnSync('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' });
      } else {
        // Negative pid targets the whole process group created by `detached`.
        try {
          process.kill(-child.pid, 'SIGKILL');
        } catch {
          try {
            child.kill('SIGKILL');
          } catch {
            /* Already gone. */
          }
        }
      }
    }, options.timeoutMs);

    child.on('error', (error) => {
      spawnError = error.message;
    });

    child.on('close', (code) => {
      clearTimeout(timer);
      resolve({
        stdout: Buffer.concat(stdoutChunks).toString('utf8'),
        stderr: Buffer.concat(stderrChunks).toString('utf8'),
        exitCode: code,
        timedOut,
        spawnError,
      });
    });

    child.stdin.on('error', () => {
      /* The program may exit without reading stdin; EPIPE here is expected. */
    });
    child.stdin.end(options.stdin ?? '');
  });
}

export const nativeBackend: ExecutionBackend = {
  kind: 'native',
  usesCallback: false,

  async start(spec: ExecutionSpec): Promise<StartResult> {
    const language = NATIVE_LANGUAGES[spec.languageId];
    if (!language) {
      return {
        kind: 'settled',
        outcome: failedOutcome(
          `language ${spec.languageId} is not supported by the native backend ` +
            `(supported: ${Object.keys(NATIVE_LANGUAGES).join(', ')}). ` +
            'Use EXECUTOR=docker or EXECUTOR=judge0 for the full language set.',
        ),
      };
    }

    const tool = toolFor(language);
    if (!tool) {
      const candidates = (language.compiler ?? language.interpreter)!.candidates;
      return {
        kind: 'settled',
        outcome: failedOutcome(
          `none of ${candidates.join(', ')} is on PATH, so this machine cannot run ` +
            `language ${spec.languageId} natively. Install one, or use EXECUTOR=docker.`,
        ),
      };
    }

    // No guaranteed unzip, and silently dropping a dataset the program then
    // fails to open is worse than refusing the run.
    if (spec.additionalFiles) {
      return {
        kind: 'settled',
        outcome: failedOutcome(
          'additionalFiles is not supported by the native backend. ' +
            'Use EXECUTOR=docker or EXECUTOR=judge0 to upload a dataset zip.',
        ),
      };
    }

    const timeLimit = spec.cpuTimeLimit ?? env.JUDGE0_CPU_TIME_LIMIT;
    const workdir = await mkdtemp(path.join(os.tmpdir(), 'eggzec-'));

    try {
      const source = path.join(workdir, language.filename);
      await writeFile(source, spec.code, 'utf8');

      let command = tool;
      let args: string[];

      if (language.compiler) {
        const binary = path.join(workdir, isWindows ? `${randomUUID().slice(0, 8)}.exe` : 'a.out');

        const build = await execute(tool, language.compiler.args(source, binary), {
          cwd: workdir,
          timeoutMs: env.EXECUTOR_COMPILE_TIMEOUT_S * 1000,
        });

        if (build.spawnError) {
          return { kind: 'settled', outcome: failedOutcome(build.spawnError) };
        }
        if (build.timedOut) {
          return {
            kind: 'settled',
            outcome: failedOutcome(`compilation exceeded ${env.EXECUTOR_COMPILE_TIMEOUT_S}s`),
          };
        }
        if (build.exitCode !== 0) {
          return {
            kind: 'settled',
            outcome: {
              status: 'COMPLETED',
              judgeStatus: 'Compilation Error',
              stdout: null,
              stderr: null,
              compileOutput: `${build.stdout}${build.stderr}`.trim() || 'compilation failed',
              exitCode: COMPILE_FAILED,
              timeMs: null,
              memoryKb: null,
              errorMessage: null,
            },
          };
        }

        command = binary;
        args = [];
      } else {
        args = language.interpreter!.args(source);
      }

      const startedAt = Date.now();
      const raw = await execute(command, args, {
        cwd: workdir,
        timeoutMs: timeLimit * 1000,
        stdin: spec.stdin,
      });
      const elapsedMs = Date.now() - startedAt;

      if (raw.spawnError) {
        return { kind: 'settled', outcome: failedOutcome(raw.spawnError) };
      }

      const base = {
        status: 'COMPLETED' as const,
        stdout: raw.stdout || null,
        stderr: raw.stderr || null,
        compileOutput: null,
        // Wall-clock, not CPU time: without a cgroup there is nothing better to
        // measure. Peak memory is simply unavailable, so it stays null rather
        // than being guessed at.
        timeMs: elapsedMs,
        memoryKb: null,
        errorMessage: null,
      };

      if (raw.timedOut) {
        return {
          kind: 'settled',
          outcome: {
            ...base,
            judgeStatus: 'Time Limit Exceeded',
            exitCode: raw.exitCode,
            timeMs: Math.round(timeLimit * 1000),
            errorMessage: `exceeded ${timeLimit}s`,
          },
        };
      }

      return {
        kind: 'settled',
        outcome: {
          ...base,
          judgeStatus: raw.exitCode === 0 ? 'Accepted' : `Runtime Error (exit ${raw.exitCode})`,
          exitCode: raw.exitCode,
        },
      };
    } finally {
      await rm(workdir, { recursive: true, force: true }).catch(() => {
        /* Best effort; a leftover temp dir is not worth failing a run over. */
      });
    }
  },

  /** Healthy when at least one toolchain is present — otherwise nothing runs. */
  async health(): Promise<boolean> {
    return availableNativeLanguages().length > 0;
  },
};
