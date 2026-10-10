import { tool } from '@langchain/core/tools';
import { Prisma } from '@prisma/client';
import { z } from 'zod';
import { prisma } from '../../../lib/prisma';
import { embedText, toVectorLiteral } from './embeddings';
import { errorMessage, truncate, type ToolContext } from './common';

export const ragInputSchema = z.object({
  query: z.string().min(1).max(8000),
  limit: z.number().int().min(1).max(20).default(4),
});

export type RagInput = z.infer<typeof ragInputSchema>;

interface RagRow {
  id: string;
  content: string;
  metadata: unknown;
  createdAt: Date;
  distance: number;
}

/**
 * Semantic recall over the user's long-term MemoryArtifact store.
 * Cosine distance via pgvector `<=>` (lower = closer). Rows without an
 * embedding are skipped. Without OPENAI_API_KEY the offline fallback
 * embedding is used and results are weak — the response flags this.
 */
export function createRagTool(ctx: ToolContext = {}) {
  return tool(
    async (input: RagInput): Promise<string> => {
      try {
        if (!ctx.userId) {
          return 'Error: vector_rag_search needs a bound session (userId). Use getToolsForSession(userId, agentSessionId).';
        }
        const { vector, provider } = await embedText(input.query);
        const literal = toVectorLiteral(vector);
        const rows = await prisma.$queryRaw<RagRow[]>(Prisma.sql`
          SELECT "id", "content", "metadata", "createdAt",
                 "embedding" <=> ${literal}::vector AS "distance"
          FROM "memory_artifacts"
          WHERE "userId" = ${ctx.userId} AND "embedding" IS NOT NULL
          ORDER BY "distance" ASC
          LIMIT ${input.limit}
        `);
        return JSON.stringify(
          {
            query: input.query,
            embeddingProvider: provider,
            ...(provider === 'fallback'
              ? { note: 'Offline fallback embedding in use — ranking is approximate. Set OPENAI_API_KEY for semantic recall.' }
              : {}),
            matches: rows.map((row) => ({
              id: row.id,
              distance: row.distance,
              createdAt: row.createdAt,
              metadata: row.metadata,
              snippet: truncate(row.content, 1000),
            })),
          },
          null,
          2,
        );
      } catch (error) {
        return `Error: vector_rag_search failed: ${errorMessage(error)}`;
      }
    },
    {
      name: 'vector_rag_search',
      description:
        'Search long-term memory for snippets semantically similar to the query. Returns top matches with ids, similarity distance and metadata. Use to recall prior findings before acting.',
      schema: ragInputSchema,
    },
  );
}

/** Unbound instance for the static registry; bind via getToolsForSession for real recall. */
export const vectorRagSearchTool = createRagTool();
