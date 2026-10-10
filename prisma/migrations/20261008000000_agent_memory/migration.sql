-- Stateful LangGraph agent orchestrator: sessions + vector memory.
--
-- * AgentSession.id is used verbatim as the LangGraph thread_id, so
--   PostgresSaver checkpoint rows line up 1:1 with this record.
-- * Execution.agentSessionId is nullable: standalone runs have no session.
-- * MemoryArtifact.embedding is vector(1536) (e.g. text-embedding-3-small)
--   with an HNSW cosine index for similarity recall.
-- * The LangGraph checkpoint tables themselves are created at runtime by
--   PostgresSaver.setup() (see src/lib/checkpointer.ts), not here.

-- pgvector must exist before any vector column or vector_* index is used.
CREATE EXTENSION IF NOT EXISTS vector;

-- CreateTable
CREATE TABLE "agent_sessions" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'ACTIVE',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "agent_sessions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "memory_artifacts" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "embedding" vector(1536),
    "metadata" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "memory_artifacts_pkey" PRIMARY KEY ("id")
);

-- AlterTable: link executions to their owning agent session (nullable).
ALTER TABLE "executions" ADD COLUMN "agentSessionId" TEXT;

-- CreateIndex
CREATE INDEX "agent_sessions_userId_idx" ON "agent_sessions"("userId");

-- CreateIndex
CREATE INDEX "agent_sessions_status_idx" ON "agent_sessions"("status");

-- CreateIndex
CREATE INDEX "memory_artifacts_userId_idx" ON "memory_artifacts"("userId");

-- CreateIndex
CREATE INDEX "executions_agentSessionId_idx" ON "executions"("agentSessionId");

-- HNSW cosine-similarity index for vector recall.
-- NULL embeddings are skipped automatically; m=16 / ef_construction=64 are
-- the standard recall-vs-build-time tradeoff for 1536-dim OpenAI embeddings.
CREATE INDEX "memory_artifacts_embedding_hnsw_idx"
    ON "memory_artifacts"
    USING hnsw ("embedding" vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- AddForeignKey
ALTER TABLE "agent_sessions" ADD CONSTRAINT "agent_sessions_userId_fkey"
    FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "memory_artifacts" ADD CONSTRAINT "memory_artifacts_userId_fkey"
    FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "executions" ADD CONSTRAINT "executions_agentSessionId_fkey"
    FOREIGN KEY ("agentSessionId") REFERENCES "agent_sessions"("id") ON DELETE SET NULL ON UPDATE CASCADE;
