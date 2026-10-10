/**
 * Local Docker backend — runs code in throwaway containers via the mounted
 * host socket.
 *
 * This exists because Judge0 1.13.x is hard-wired to cgroup v1 and cannot run
 * on a cgroup v2 host, and 1.13.1 (April 2024) is the newest image published.
 * It implements the same contract as the Judge0 backend so the service layer,
 * the API and the UI do not care which one is configured. It settles inline
 * rather than deferring, so there is no webhook and no token.
 *
 * Isolation per run: no network, dropped capabilities, no privilege
 * escalation, read-only root filesystem with a size-capped exec tmpfs, memory
 * and swap capped to the same value, CPU quota, PID cap, unprivileged uid.
 *
 * SECURITY: needs /var/run/docker.sock, which is root-equivalent on the host.
 * Fine for local development; front it with a socket proxy or move execution
 * to a disposable VM before exposing this publicly.
 */
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { env } from '../../config/env';
import { DOCKER_LANGUAGES, type DockerLanguageSpec } from './languages';
import {
  failedOutcome,
  type ExecutionBackend,
  type ExecutionOutcome,
  type ExecutionSpec,
  type StartResult,
} from './types';

/** Emitted on stderr by the in-container wrapper, stripped before reporting. */
const METRICS_PREFIX = '__JX__';
/** 128 + SIGKILL — what `timeout -s KILL` and the OOM killer both produce. */
const SIGKILL_EXIT = 137;
/** Reported as the exit code when compilation fails. */
const COMPILE_FAILED = 97;

function imageFor(spec: DockerLanguageSpec): string {
  switch (spec.imageEnv) {
    case 'python':
      return env.EXECUTOR_IMAGE_PYTHON;
    case 'node':
      return env.EXECUTOR_IMAGE_NODE;
    case 'cpp':
      return env.EXECUTOR_IMAGE_CPP;
    case 'bash':
      return env.EXECUTOR_IMAGE_BASH;
  }
}

/** Reads cumulative CPU microseconds for this container's cgroup. */
const CPU_USEC = "awk '/usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || echo 0";
/** Non-zero once the kernel OOM-kills anything in this container's cgroup. */
const OOM_COUNT = "awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null || echo 0";

/**
 * Unpacks $FILES_B64 (a base64 zip) into the working directory.
 *
 * Tries three extractors because no single one is present in all runner
 * images: busybox `unzip` covers alpine, python's zipfile module covers the
 * python image, and GNU `unzip` covers debian-based ones.
 */
const UNPACK_FILES = [
  'if [ -n "$FILES_B64" ]; then',
  '  echo "$FILES_B64" | base64 -d > /tmp/_extra.zip 2>/dev/null || { echo "additional_files: not valid base64" >&2; exit 1; }',
  '  if command -v unzip >/dev/null 2>&1; then unzip -o -q /tmp/_extra.zip -d /tmp;',
  '  elif command -v python3 >/dev/null 2>&1; then python3 -m zipfile -e /tmp/_extra.zip /tmp;',
  '  elif command -v busybox >/dev/null 2>&1; then busybox unzip -o -q /tmp/_extra.zip -d /tmp;',
  '  else echo "additional_files: no unzip available in this image" >&2; exit 1; fi',
  '  rm -f /tmp/_extra.zip',
  'fi',
].join('\n');

const decodeSource = (filename: string) => `echo "$SRC_B64" | base64 -d > /tmp/${filename}`;

function runScript(spec: DockerLanguageSpec, timeLimitSeconds: number, compiled: boolean): string {
  return [
    compiled ? '' : decodeSource(spec.filename),
    UNPACK_FILES,
    `CPU0=$(${CPU_USEC})`,
    `timeout -s KILL ${timeLimitSeconds} ${spec.run}`,
    'RC=$?',
    `CPU1=$(${CPU_USEC})`,
    'PEAK=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo 0)',
    `OOM=$(${OOM_COUNT})`,
    `echo "${METRICS_PREFIX} $RC $CPU0 $CPU1 $PEAK $OOM" >&2`,
    'exit 0',
  ]
    .filter(Boolean)
    .join('\n');
}

function compileScript(spec: DockerLanguageSpec): string {
  return [
    decodeSource(spec.filename),
    UNPACK_FILES,
    spec.compile!,
    'RC=$?',
    `OOM=$(${OOM_COUNT})`,
    `echo "${METRICS_PREFIX} $RC 0 0 0 $OOM" >&2`,
    'exit 0',
  ].join('\n');
}

interface RawRun {
  stdout: string;
  stderr: string;
  programExit: number | null;
  timeMs: number | null;
  memoryKb: number | null;
  oomKilled: boolean;
  killedByWatchdog: boolean;
  dockerError: string | null;
}

function parseMetrics(stderr: string) {
  const lines = stderr.split('\n');

  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i]!;
    if (!line.startsWith(`${METRICS_PREFIX} `)) continue;

    const [, rc, cpu0, cpu1, peak, oom] = line.split(' ');
    lines.splice(i, 1);

    const asNumber = (value: string | undefined) => (/^\d+$/.test(value ?? '') ? Number(value) : null);
    const beforeUsec = asNumber(cpu0);
    const afterUsec = asNumber(cpu1);
    const peakBytes = asNumber(peak);

    return {
      cleaned: lines.join('\n'),
      programExit: Number.isFinite(Number(rc)) ? Number(rc) : null,
      timeMs:
        beforeUsec !== null && afterUsec !== null && afterUsec >= beforeUsec
          ? Math.round((afterUsec - beforeUsec) / 1000)
          : null,
      memoryKb: peakBytes !== null && peakBytes > 0 ? Math.round(peakBytes / 1024) : null,
      oomKilled: (asNumber(oom) ?? 0) > 0,
    };
  }

  return { cleaned: stderr, programExit: null, timeMs: null, memoryKb: null, oomKilled: false };
}

interface ContainerOptions {
  image: string;
  script: string;
  srcB64: string;
  filesB64: string;
  memoryMb: number;
  cpus: number;
  watchdogMs: number;
  stdin?: string | null;
  volume?: { name: string; readOnly: boolean };
  asRoot?: boolean;
}

function dockerArgs(name: string, options: ContainerOptions): string[] {
  const args = [
    'run',
    '--rm',
    '--interactive',
    '--name', name,
    '--network', 'none',
    '--memory', `${options.memoryMb}m`,
    // Equal to --memory: no swap, so exceeding the cap OOM-kills immediately
    // instead of thrashing and reporting a bogus timeout.
    '--memory-swap', `${options.memoryMb}m`,
    '--cpus', String(options.cpus),
    '--pids-limit', String(env.EXECUTOR_PIDS_LIMIT),
    '--read-only',
    '--tmpfs', `/tmp:exec,mode=1777,size=${env.EXECUTOR_TMPFS_MB}m`,
    '--cap-drop', 'ALL',
    '--security-opt', 'no-new-privileges',
    '--workdir', '/tmp',
    '--env', `SRC_B64=${options.srcB64}`,
    '--env', `FILES_B64=${options.filesB64}`,
  ];

  if (options.volume) {
    args.push('--volume', `${options.volume.name}:/build${options.volume.readOnly ? ':ro' : ''}`);
  }
  // A fresh named volume is owned by root, so the compile step runs as uid 0 to
  // write into it. Still network-less, capability-less and no-new-privs; test
  // runs then execute the artifact as nobody with the volume read-only.
  if (!options.asRoot) args.push('--user', '65534:65534');

  args.push(options.image, 'sh', '-c', options.script);
  return args;
}

function runContainer(options: ContainerOptions): Promise<RawRun> {
  const name = `hrc-exec-${randomUUID().slice(0, 12)}`;

  return new Promise<RawRun>((resolve) => {
    const child = spawn('docker', dockerArgs(name, options), { stdio: ['pipe', 'pipe', 'pipe'] });

    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];
    let stdoutBytes = 0;
    let stderrBytes = 0;
    let killedByWatchdog = false;
    let spawnError: string | null = null;

    // Cap captured output so a runaway print loop can't exhaust the API's heap
    // before the container's own limits stop it.
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

    // Backstop: the in-container `timeout` should fire first. This only trips
    // if the container itself wedges (image pull stall, daemon hiccup).
    const watchdog = setTimeout(() => {
      killedByWatchdog = true;
      spawn('docker', ['kill', name], { stdio: 'ignore' });
    }, options.watchdogMs);

    child.on('error', (error) => {
      spawnError = error.message;
    });

    child.on('close', () => {
      clearTimeout(watchdog);
      const parsed = parseMetrics(Buffer.concat(stderrChunks).toString('utf8'));

      resolve({
        stdout: Buffer.concat(stdoutChunks).toString('utf8'),
        stderr: parsed.cleaned,
        programExit: parsed.programExit,
        timeMs: parsed.timeMs,
        memoryKb: parsed.memoryKb,
        oomKilled: parsed.oomKilled,
        killedByWatchdog,
        // The wrapper always exits 0, so a missing metrics line means the
        // container never got far enough to run anything.
        dockerError:
          spawnError ??
          (parsed.programExit === null && !killedByWatchdog
            ? parsed.cleaned.trim() || 'container did not start'
            : null),
      });
    });

    child.stdin.on('error', () => {
      /* Program may exit without reading stdin; EPIPE here is expected. */
    });
    child.stdin.end(options.stdin ?? '');
  });
}

function docker(args: string[]): Promise<{ code: number; output: string }> {
  return new Promise((resolve) => {
    const child = spawn('docker', args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let output = '';
    child.stdout.on('data', (chunk: Buffer) => (output += chunk.toString()));
    child.stderr.on('data', (chunk: Buffer) => (output += chunk.toString()));
    child.on('error', (error) => resolve({ code: -1, output: error.message }));
    child.on('close', (code) => resolve({ code: code ?? -1, output }));
  });
}

function toOutcome(raw: RawRun, timeLimitSeconds: number, memoryLimitKb: number): ExecutionOutcome {
  if (raw.dockerError) return failedOutcome(raw.dockerError);

  const base = {
    status: 'COMPLETED' as const,
    stdout: raw.stdout || null,
    stderr: raw.stderr || null,
    compileOutput: null,
    timeMs: raw.timeMs,
    memoryKb: raw.memoryKb,
    errorMessage: null,
  };

  // Checked before the SIGKILL branch: an OOM kill also surfaces as 137, and
  // reporting a timeout for a program that blew the memory cap sends people
  // optimising entirely the wrong thing.
  if (raw.oomKilled) {
    return {
      ...base,
      judgeStatus: 'Memory Limit Exceeded',
      exitCode: raw.programExit,
      errorMessage: `exceeded ${Math.round(memoryLimitKb / 1024)}MB`,
    };
  }

  if (raw.killedByWatchdog || raw.programExit === SIGKILL_EXIT) {
    return {
      ...base,
      judgeStatus: 'Time Limit Exceeded',
      exitCode: raw.programExit,
      timeMs: Math.round(timeLimitSeconds * 1000),
      errorMessage: `exceeded ${timeLimitSeconds}s`,
    };
  }

  return {
    ...base,
    judgeStatus: raw.programExit === 0 ? 'Accepted' : `Runtime Error (exit ${raw.programExit})`,
    exitCode: raw.programExit,
  };
}

export const dockerBackend: ExecutionBackend = {
  kind: 'docker',
  usesCallback: false,

  async start(spec: ExecutionSpec): Promise<StartResult> {
    const language = DOCKER_LANGUAGES[spec.languageId];
    if (!language) {
      return {
        kind: 'settled',
        outcome: failedOutcome(
          `language ${spec.languageId} is not supported by the local Docker backend ` +
            `(supported: ${Object.keys(DOCKER_LANGUAGES).join(', ')}). ` +
            'Point EXECUTOR at judge0 for the full language set.',
        ),
      };
    }

    const image = imageFor(language);
    const srcB64 = Buffer.from(spec.code, 'utf8').toString('base64');
    const filesB64 = spec.additionalFiles ?? '';
    const timeLimit = spec.cpuTimeLimit ?? env.JUDGE0_CPU_TIME_LIMIT;
    const memoryLimitKb = spec.memoryLimit ?? env.JUDGE0_MEMORY_LIMIT;
    const memoryMb = Math.max(16, Math.round(memoryLimitKb / 1024));

    let volume: string | undefined;

    try {
      if (language.compile) {
        volume = `hrc-build-${randomUUID().slice(0, 12)}`;
        const created = await docker(['volume', 'create', volume]);
        if (created.code !== 0) {
          volume = undefined;
          return {
            kind: 'settled',
            outcome: failedOutcome(`could not create build volume: ${created.output.trim()}`),
          };
        }

        const build = await runContainer({
          image,
          script: compileScript(language),
          srcB64,
          filesB64,
          // Compilation gets its own budget: g++ on <bits/stdc++.h> needs far
          // more memory than a run is allowed, and sharing one limit makes
          // every C++ build die with `cc1plus: Killed`.
          memoryMb: env.EXECUTOR_COMPILE_MEMORY_MB,
          cpus: env.EXECUTOR_COMPILE_CPUS,
          watchdogMs: env.EXECUTOR_COMPILE_TIMEOUT_S * 1000 + env.EXECUTOR_STARTUP_GRACE_MS,
          volume: { name: volume, readOnly: false },
          asRoot: true,
        });

        if (build.dockerError) {
          return { kind: 'settled', outcome: failedOutcome(build.dockerError) };
        }
        if (build.programExit !== 0) {
          const hint = build.oomKilled
            ? `\ncompiler ran out of memory (limit ${env.EXECUTOR_COMPILE_MEMORY_MB}MB) — raise EXECUTOR_COMPILE_MEMORY_MB`
            : '';
          return {
            kind: 'settled',
            outcome: {
              status: 'COMPLETED',
              judgeStatus: 'Compilation Error',
              stdout: null,
              stderr: null,
              compileOutput: `${build.stderr}${hint}`.trim() || 'compilation failed',
              exitCode: COMPILE_FAILED,
              timeMs: null,
              memoryKb: null,
              errorMessage: null,
            },
          };
        }
      }

      const raw = await runContainer({
        image,
        script: runScript(language, timeLimit, Boolean(language.compile)),
        srcB64,
        filesB64,
        memoryMb,
        cpus: env.EXECUTOR_CPUS,
        watchdogMs: timeLimit * 1000 + env.EXECUTOR_STARTUP_GRACE_MS,
        stdin: spec.stdin,
        ...(volume ? { volume: { name: volume, readOnly: true } } : {}),
      });

      return { kind: 'settled', outcome: toOutcome(raw, timeLimit, memoryLimitKb) };
    } finally {
      if (volume) await docker(['volume', 'rm', '-f', volume]);
    }
  },

  async health(): Promise<boolean> {
    const { code } = await docker(['version', '--format', '{{.Server.Version}}']);
    return code === 0;
  },
};
