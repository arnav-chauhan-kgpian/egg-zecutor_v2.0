import { createApp } from './app';
import { env } from './config/env';
import { prisma } from './lib/prisma';
import { closeCheckpointer, setupCheckpointer } from './lib/checkpointer';
import {
  recoverOrphanedRuns,
  startReconciler,
  stopReconciler,
} from './modules/executions/execution.reconciler';
import { engineInfo } from './services/execution';

const app = createApp();

const server = app.listen(env.PORT, env.HOST, () => {
  console.log(`API listening on http://${env.HOST}:${env.PORT} [${env.NODE_ENV}]`);
  console.log(
    `Execution engine: ${engineInfo.kind}` +
      (engineInfo.endpoint ? ` (${engineInfo.endpoint})` : '') +
      (engineInfo.usesCallback ? ` — webhook ${engineInfo.callbackUrl}` : ' — inline'),
  );

  // Loud on purpose. Everything else about this backend looks and behaves like
  // the sandboxed ones, so the one thing that differs has to be impossible to
  // miss in the logs.
  if (!engineInfo.sandboxed) {
    console.warn(
      '\n  ⚠  NO SANDBOX — submitted code runs as this process, with your\n' +
        '     files, network and privileges. Fine when you are the only one\n' +
        '     writing the code; never expose this port to anyone else.\n' +
        '     Install Docker and restart for an isolated backend.\n',
    );
  }
  // Inline runs cannot survive a restart; clear them so the UI stops spinning.
  void recoverOrphanedRuns().catch((error) =>
    console.error('[reconciler] recovery failed:', error instanceof Error ? error.message : error),
  );
  // LangGraph checkpoint tables (PostgresSaver) on the isolated pool.
  void setupCheckpointer().catch((error) =>
    console.error('[checkpointer] setup failed:', error instanceof Error ? error.message : error),
  );
  // No-op unless the backend defers results (Judge0).
  startReconciler();
});

async function shutdown(signal: string) {
  console.log(`\n${signal} received, shutting down...`);
  stopReconciler();
  server.close(async () => {
    await closeCheckpointer().catch((error) =>
      console.error('[checkpointer] close failed:', error instanceof Error ? error.message : error),
    );
    await prisma.$disconnect();
    process.exit(0);
  });

  // Don't hang forever if connections refuse to drain.
  setTimeout(() => process.exit(1), 10_000).unref();
}

process.on('SIGINT', () => void shutdown('SIGINT'));
process.on('SIGTERM', () => void shutdown('SIGTERM'));
