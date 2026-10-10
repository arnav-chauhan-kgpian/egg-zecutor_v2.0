import { tool } from '@langchain/core/tools';
import { Prisma } from '@prisma/client';
import { z } from 'zod';
import { prisma } from '../../../lib/prisma';
import { embedText, toVectorLiteral } from './embeddings';
import { errorMessage, type ToolContext } from './common';

export const saveMemoryInputSchema = z.object({
  content: z.string().min(1).max(50_000),
  metadata: z.record(z.unknown()).optional(),
});

export type SaveMemoryInput = z.infer<typeof saveMemoryInputSchema>;

/**
 * Persist a memory chunk for later vector recall. The text is embedded
 * (OpenAI when configured, offline fallback otherwise) and the vector is
 * written via raw SQL because Prisma maps `vector(1536)` as Unsupported.
 * Always succeeds at storing the text — a failed embedding write leaves the
 * row with a NULL embedding rather than losing the memory.
 */
export function createSaveMemoryTool(ctx: ToolContext = {}) {
  return tool(
    async (input: SaveMemoryInput): Promise<string> => {
      try {
        if (!ctx.userId) {
          return 'Error: save_memory_artifact needs a bound session (userId). Use getToolsForSession(userId, agentSessionId).';
        }
        const created = await prisma.memoryArtifact.create({
          data: {
            userId: ctx.userId,
            content: input.content,
            metadata: (input.metadata ?? undefined) as Prisma.InputJsonValue | undefined,
          },
          select: { id: true, createdAt: true },
        });
        let embedded = false;
        let provider: string | null = null;
        try {
          const { vector, provider: used } = await embedText(input.content);
          provider = used;
          await prisma.$executeRaw(Prisma.sql`
            UPDATE "memory_artifacts"
            SET "embedding" = ${toVectorLiteral(vector)}::vector
            WHERE "id" = ${created.id}
          `);
          embedded = true;
        } catch {
          // Text is already persisted; a NULL embedding just excludes the row from recall.
          embedded = false;
        }
        return JSON.stringify(
          {
            ok: true,
            id: created.id,
            createdAt: created.createdAt,
            embedded,
            embeddingProvider: provider,
            ...(!embedded ? { note: 'Text saved; embedding unavailable, row excluded from recall until re-embedded.' } : {}),
          },
          null,
          2,
        );
      } catch (error) {
        return `Error: save_memory_artifact failed: ${errorMessage(error)}`;
      }
    },
    {
      name: 'save_memory_artifact',
      description:
        'Save a durable memory (fact, finding, preference, summary) for later recall via vector_rag_search. Returns the memory ID.',
      schema: saveMemoryInputSchema,
    },
  );
}

/** Unbound instance for the static registry; bind via getToolsForSession to save. */
export const saveMemoryArtifactTool = createSaveMemoryTool();
