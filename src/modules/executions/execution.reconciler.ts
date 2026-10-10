/**
 * Reconciler for deferred (Judge0) executions.
 *
 * The webhook is the fast path, but it is not a guarantee: Judge0 retries a
 * callback only a few times, a container restart can drop one, and a network
 * blip loses it entirely. Without this sweep a lost callback strands a run in
 * PROCESSING forever and the UI spins.
 *
 * Every tick it polls Judge0 for PROCESSING rows that hold a token, and gives
 * up on anything older than JUDGE0_MAX_WAIT_MS. `finalize` is idempotent, so
 * racing the webhook is harmless — whichever arrives second is a no-op.
 *
 * Only runs when the active backend defers; the Docker backend settles inline.
 */
import { ExecutionStatus } from '../../lib/enums';
import { env } from '../../config/env';
import { prisma } from '../../lib/prisma';
import { backend } from '../../services/execution';
import { finalize } from './execution.service';

let timer: NodeJS.Timeout | null = null;

/**
 * Fails runs that this process was executing when it died.
 *
 * The Docker backend runs inline, so its work lives in this process and has no
 * token to re-poll. After a restart those rows would sit in PROCESSING forever
 * and the UI would spin. Anything still PROCESSING without a token at boot is
 * unrecoverable by definition, so say so instead of hanging.
 *
 * Token-bearing (Judge0) rows are deliberately left alone — the judge is still
 * working on them and the sweep below will collect the result.
 */
export async function recoverOrphanedRuns(): Promise<number> {
  const orphaned = await prisma.execution.updateMany({
    where: {
      status: { in: [ExecutionStatus.PENDING, ExecutionStatus.PROCESSING] },
      judge0Token: null,
    },
    data: {
      status: ExecutionStatus.FAILED,
      judgeStatus: 'Internal Error',
      errorMessage: 'the API restarted while this run was in flight; resubmit to try again',
      completedAt: new Date(),
    },
  });

  if (orphaned.count > 0) {
    console.log(`[reconciler] failed ${orphaned.count} run(s) orphaned by a restart`);
  }
  return orphaned.count;
}

async function sweep(): Promise<void> {
  const cutoff = new Date(Date.now() - env.JUDGE0_MAX_WAIT_MS);

  const pending = await prisma.execution.findMany({
    where: { status: ExecutionStatus.PROCESSING, judge0Token: { not: null } },
    select: { id: true, judge0Token: true, startedAt: true, createdAt: true },
    take: 50,
  });

  for (const execution of pending) {
    const startedAt = execution.startedAt ?? execution.createdAt;

    if (startedAt < cutoff) {
      await finalize(execution.id, {
        status: 'FAILED',
        judgeStatus: 'Internal Error',
        stdout: null,
        stderr: null,
        compileOutput: null,
        exitCode: null,
        timeMs: null,
        memoryKb: null,
        errorMessage:
          `no result after ${Math.round(env.JUDGE0_MAX_WAIT_MS / 1000)}s — ` +
          'the Judge0 callback never arrived and polling found no verdict',
      });
      continue;
    }

    try {
      const outcome = await backend.poll!(execution.judge0Token!);
      if (outcome) await finalize(execution.id, outcome);
    } catch {
      // Transient — Judge0 may be restarting. Try again next tick; the cutoff
      // above is what eventually gives up.
    }
  }
}

export function startReconciler(): void {
  if (!backend.poll) return;
  if (timer) return;

  timer = setInterval(() => {
    void sweep().catch((error) => {
      console.error('[reconciler] sweep failed:', error instanceof Error ? error.message : error);
    });
  }, env.JUDGE0_POLL_INTERVAL_MS);

  // Do not hold the process open on shutdown.
  timer.unref();
}

export function stopReconciler(): void {
  if (timer) clearInterval(timer);
  timer = null;
}
