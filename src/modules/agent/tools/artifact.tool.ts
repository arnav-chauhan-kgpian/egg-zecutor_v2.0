import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { prisma } from '../../../lib/prisma';
import { errorMessage, formatBytes, truncate, type ToolContext } from './common';

export const readArtifactInputSchema = z.object({
  artifactId: z.string().uuid(),
  encoding: z.enum(['utf8', 'base64']).default('utf8'),
});

export type ReadArtifactInput = z.infer<typeof readArtifactInputSchema>;

/** Max chars returned for utf8 reads — full dumps would blow the context window. */
const MAX_TEXT_CHARS = 12_000;
/** Max chars returned for base64 reads. */
const MAX_B64_CHARS = 100_000;

const TEXTISH = /^(text\/|application\/(json|xml|javascript|x-www-form-urlencoded)|.*\+(json|xml))|json|csv|markdown|html/i;

/**
 * Read a file previously produced by a run. Text-ish payloads decode to UTF-8;
 * anything else (or explicit base64 encoding) returns base64, truncated.
 * When bound to a session, reads are scoped to the caller's own executions.
 */
export function createReadArtifactTool(ctx: ToolContext = {}) {
  return tool(
    async (input: ReadArtifactInput): Promise<string> => {
      try {
        const artifact = ctx.userId
          ? await prisma.artifact.findFirst({
              where: { id: input.artifactId, execution: { userId: ctx.userId } },
            })
          : await prisma.artifact.findUnique({ where: { id: input.artifactId } });
        if (!artifact) return `Error: artifact not found: ${input.artifactId}`;
        if (input.encoding === 'base64') {
          const truncated = artifact.content.length > MAX_B64_CHARS;
          return JSON.stringify(
            {
              id: artifact.id,
              name: artifact.name,
              mimeType: artifact.mimeType,
              sizeBytes: artifact.sizeBytes,
              size: formatBytes(artifact.sizeBytes),
              encoding: 'base64',
              truncated,
              content: truncated ? `${artifact.content.slice(0, MAX_B64_CHARS)}…[truncated]` : artifact.content,
            },
            null,
            2,
          );
        }
        let text: string;
        try {
          text = Buffer.from(artifact.content, 'base64').toString('utf8');
        } catch {
          return `Error: artifact ${artifact.name} is not valid base64 and cannot be decoded as text. Retry with encoding "base64".`;
        }
        if (!TEXTISH.test(artifact.mimeType) && /[\u0000-\u0008\u000e-\u001f]/.test(text.slice(0, 2000))) {
          return JSON.stringify(
            {
              id: artifact.id,
              name: artifact.name,
              mimeType: artifact.mimeType,
              sizeBytes: artifact.sizeBytes,
              warning: 'Binary-looking content; UTF-8 may be garbled. Retry with encoding "base64".',
              content: truncate(text, MAX_TEXT_CHARS),
            },
            null,
            2,
          );
        }
        return JSON.stringify(
          {
            id: artifact.id,
            name: artifact.name,
            mimeType: artifact.mimeType,
            sizeBytes: artifact.sizeBytes,
            size: formatBytes(artifact.sizeBytes),
            encoding: 'utf8',
            truncated: text.length > MAX_TEXT_CHARS,
            content: truncate(text, MAX_TEXT_CHARS),
          },
          null,
          2,
        );
      } catch (error) {
        return `Error: read_artifact failed: ${errorMessage(error)}`;
      }
    },
    {
      name: 'read_artifact',
      description:
        'Read a file artifact produced by a previous run by its artifact ID. Returns text content (utf8) or base64 data. Use after execute_code_sandbox reports artifacts.',
      schema: readArtifactInputSchema,
    },
  );
}

/** Unbound instance for the static registry; ownership scoping applies when bound. */
export const readArtifactTool = createReadArtifactTool();
