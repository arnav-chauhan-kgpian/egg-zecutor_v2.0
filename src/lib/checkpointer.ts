import pg from 'pg';
import { PostgresSaver } from '@langchain/langgraph-checkpoint-postgres';
import { env } from '../config/env';

/**
 * Isolated connection pool for LangGraph checkpointing.
 *
 * Deliberately SEPARATE from the Prisma pool: checkpoint reads/writes happen
 * on every graph step (potentially several per agent turn), so sharing
 * Prisma's pool risks connection starvation under concurrent sessions.
 * Tune with CHECKPOINTER_POOL_MAX (default 5).
 */
function poolMax(): number {
  const raw = Number(process.env.CHECKPOINTER_POOL_MAX ?? 5);
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 5;
}

export const checkpointerPool = new pg.Pool({
  connectionString: env.DATABASE_URL,
  max: poolMax(),
  idleTimeoutMillis: 30_000,
  connectionTimeoutMillis: 10_000,
  application_name: 'egg-zecutor-checkpointer',
});

checkpointerPool.on('error', (error) => {
  console.error('[checkpointer] pool error:', error instanceof Error ? error.message : error);
});

/**
 * Shared PostgresSaver instance. Use the AgentSession id verbatim as the
 * LangGraph `thread_id`:
 *
 *   const config = { configurable: { thread_id: agentSessionId } };
 *   await graph.invoke(input, config);
 */
export const checkpointer = new PostgresSaver(checkpointerPool);

let setupPromise: Promise<PostgresSaver> | null = null;

/**
 * Create checkpoint tables/migrations if needed. Idempotent and safe under
 * concurrent calls — the first call wins, the rest await the same promise.
 * Call once on API boot (see src/server.ts) before any graph invocation.
 */
export function setupCheckpointer(): Promise<PostgresSaver> {
  if (!setupPromise) {
    setupPromise = checkpointer
      .setup()
      .then(() => {
        console.log('[checkpointer] PostgresSaver ready');
        return checkpointer;
      })
      .catch((error) => {
        // Allow retry on next boot/call instead of caching a rejection forever.
        setupPromise = null;
        throw error;
      });
  }
  return setupPromise;
}

/**
 * Drain the isolated pool. Call on graceful shutdown (after Prisma disconnect).
 */
export async function closeCheckpointer(): Promise<void> {
  setupPromise = null;
  await checkpointerPool.end();
}
