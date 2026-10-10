import { Prisma } from '@prisma/client';
import { ExecutionStatus } from '../../lib/enums';
import { env } from '../../config/env';
import { prisma } from '../../lib/prisma';
import {
  backend,
  extractArtifacts,
  languageName,
  type ExecutionOutcome,
  type ExecutionSpec,
} from '../../services/execution';
import { ApiError } from '../../utils/ApiError';

/**
 * Fields returned to clients. `code` and `additionalFiles` are omitted from
 * list responses — a dataset zip has no business in a listing payload.
 */
const SUMMARY_SELECT = {
  id: true,
  name: true,
  languageId: true,
  status: true,
  judgeStatus: true,
  exitCode: true,
  timeMs: true,
  memoryKb: true,
  createdAt: true,
  completedAt: true,
} satisfies Prisma.ExecutionSelect;

const DETAIL_SELECT = {
  ...SUMMARY_SELECT,
  code: true,
  stdin: true,
  timeLimit: true,
  memoryLimit: true,
  stdout: true,
  stderr: true,
  compileOutput: true,
  errorMessage: true,
  updatedAt: true,
  startedAt: true,
  artifacts: {
    select: { id: true, name: true, mimeType: true, sizeBytes: true, createdAt: true },
    orderBy: { createdAt: 'asc' },
  },
} satisfies Prisma.ExecutionSelect;

export interface CreateExecutionInput {
  userId: string;
  code: string;
  languageId: number;
  name?: string;
  stdin?: string;
  additionalFiles?: string;
  timeLimit?: number;
  memoryLimit?: number;
  /** Owning agent session (LangGraph thread_id). Plain runs omit it. */
  agentSessionId?: string;
}

function withLanguageName<T extends { languageId: number }>(execution: T) {
  return { ...execution, languageName: languageName(execution.languageId) };
}

/**
 * Persists the run, then hands it to the backend.
 *
 * Returns as soon as the record exists — the caller gets 202 and an id to poll.
 * Nothing about the backend's latency is on the request path, which is the
 * whole point of the async model.
 */
/**
 * Rejects a run when the machine (or the user) already has enough in flight.
 *
 * Rate limiting alone is not enough: a handful of 15-minute jobs will pin the
 * host long after the request rate looks calm. This bounds concurrent work,
 * which is what actually costs CPU and memory.
 */
async function assertCapacity(userId: string): Promise<void> {
  const inFlight = { status: { in: [ExecutionStatus.PENDING, ExecutionStatus.PROCESSING] } };

  const [mine, total] = await Promise.all([
    prisma.execution.count({ where: { userId, ...inFlight } }),
    prisma.execution.count({ where: inFlight }),
  ]);

  if (mine >= env.MAX_CONCURRENT_RUNS_PER_USER) {
    throw new ApiError(
      429,
      `You already have ${mine} run(s) in flight (limit ${env.MAX_CONCURRENT_RUNS_PER_USER}). ` +
        'Wait for one to finish.',
    );
  }
  if (total >= env.MAX_CONCURRENT_RUNS_TOTAL) {
    throw new ApiError(
      503,
      `The engine is at capacity (${total} runs in flight). Try again shortly.`,
    );
  }
}

export async function createExecution(input: CreateExecutionInput) {
  await assertCapacity(input.userId);

  const execution = await prisma.execution.create({
    data: {
      userId: input.userId,
      code: input.code,
      languageId: input.languageId,
      name: input.name ?? null,
      stdin: input.stdin ?? null,
      additionalFiles: input.additionalFiles ?? null,
      timeLimit: input.timeLimit ?? null,
      memoryLimit: input.memoryLimit ?? null,
      agentSessionId: input.agentSessionId ?? null,
      status: ExecutionStatus.PENDING,
    },
    select: DETAIL_SELECT,
  });

  // Deliberately not awaited: dispatch owns the rest of the lifecycle and
  // writes its own terminal state. Failures are captured inside.
  void dispatch(execution.id, {
    code: input.code,
    languageId: input.languageId,
    stdin: input.stdin ?? null,
    additionalFiles: input.additionalFiles ?? null,
    cpuTimeLimit: input.timeLimit ?? null,
    memoryLimit: input.memoryLimit ?? null,
  });

  return withLanguageName(execution);
}

async function dispatch(executionId: string, spec: ExecutionSpec): Promise<void> {
  try {
    await prisma.execution.update({
      where: { id: executionId },
      data: { status: ExecutionStatus.PROCESSING, startedAt: new Date() },
    });

    const started = await backend.start(spec);

    if (started.kind === 'deferred') {
      // Judge0 has the work. The webhook (or the reconciler) finishes it.
      await prisma.execution.update({
        where: { id: executionId },
        data: { judge0Token: started.token },
      });
      return;
    }

    await finalize(executionId, started.outcome);
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    await finalize(executionId, {
      status: 'FAILED',
      judgeStatus: 'Internal Error',
      stdout: null,
      stderr: null,
      compileOutput: null,
      exitCode: null,
      timeMs: null,
      memoryKb: null,
      errorMessage: reason.slice(0, 4000),
    }).catch(() => {
      /* The record may have been deleted mid-run; nothing useful to do. */
    });
  }
}

/**
 * Writes a terminal result, extracting any artifacts the program emitted.
 *
 * Idempotent by design: a webhook and the reconciler can race for the same
 * execution, and whichever loses must not clobber or duplicate. The update is
 * conditioned on the row still being non-terminal.
 */
export async function finalize(executionId: string, outcome: ExecutionOutcome): Promise<boolean> {
  const parsed = extractArtifacts(outcome.stdout, {
    maxBytes: env.ARTIFACT_MAX_TOTAL_BYTES,
    maxCount: env.ARTIFACT_MAX_COUNT,
  });

  // Parser warnings belong with the program's own diagnostics.
  const stderr = parsed.warnings.length
    ? [outcome.stderr, ...parsed.warnings.map((warning) => `[artifacts] ${warning}`)]
        .filter(Boolean)
        .join('\n')
    : outcome.stderr;

  return prisma.$transaction(async (tx) => {
    // Only claim rows that have not already reached a terminal state.
    const claimed = await tx.execution.updateMany({
      where: {
        id: executionId,
        status: { in: [ExecutionStatus.PENDING, ExecutionStatus.PROCESSING] },
      },
      data: {
        status: outcome.status === 'FAILED' ? ExecutionStatus.FAILED : ExecutionStatus.COMPLETED,
        judgeStatus: outcome.judgeStatus,
        stdout: parsed.stdout,
        stderr,
        compileOutput: outcome.compileOutput,
        exitCode: outcome.exitCode,
        timeMs: outcome.timeMs,
        memoryKb: outcome.memoryKb,
        errorMessage: outcome.errorMessage ?? null,
        completedAt: new Date(),
      },
    });

    if (claimed.count === 0) return false;

    if (parsed.artifacts.length) {
      await tx.artifact.createMany({
        data: parsed.artifacts.map((artifact) => ({ executionId, ...artifact })),
      });
    }

    return true;
  });
}

export async function getExecution(id: string, userId: string) {
  const execution = await prisma.execution.findFirst({
    where: { id, userId },
    select: DETAIL_SELECT,
  });
  if (!execution) throw new ApiError(404, 'Execution not found');
  return withLanguageName(execution);
}

export async function listExecutions(userId: string, page: number, pageSize: number) {
  const [executions, total] = await Promise.all([
    prisma.execution.findMany({
      where: { userId },
      select: SUMMARY_SELECT,
      orderBy: { createdAt: 'desc' },
      skip: (page - 1) * pageSize,
      take: pageSize,
    }),
    prisma.execution.count({ where: { userId } }),
  ]);

  return {
    executions: executions.map(withLanguageName),
    pagination: { page, pageSize, total },
  };
}

export async function getArtifact(executionId: string, artifactId: string, userId: string) {
  const artifact = await prisma.artifact.findFirst({
    // Scoped through the execution so one user cannot read another's output.
    where: { id: artifactId, executionId, execution: { userId } },
  });
  if (!artifact) throw new ApiError(404, 'Artifact not found');
  return artifact;
}

export async function deleteExecution(id: string, userId: string) {
  const deleted = await prisma.execution.deleteMany({ where: { id, userId } });
  if (deleted.count === 0) throw new ApiError(404, 'Execution not found');
}

/** Looks an execution up by the Judge0 token carried on a callback. */
export async function findByToken(token: string) {
  return prisma.execution.findUnique({ where: { judge0Token: token }, select: { id: true } });
}
