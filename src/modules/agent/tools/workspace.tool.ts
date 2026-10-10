import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { prisma } from '../../../lib/prisma';
import { errorMessage, formatBytes, type ToolContext } from './common';

export const workspaceInputSchema = z.object({
  agentSessionId: z.string().uuid(),
});

export type WorkspaceInput = z.infer<typeof workspaceInputSchema>;

/**
 * Lists the session's workspace: recent executions plus the artifacts they
 * produced. Scoped to the bound user when available (defence in depth —
 * session ids are unguessable UUIDs, but a bound tool should never leak
 * across users even if handed a foreign session id).
 */
export function createWorkspaceTool(ctx: ToolContext = {}) {
  return tool(
    async (input: WorkspaceInput): Promise<string> => {
      try {
        const executions = await prisma.execution.findMany({
          where: {
            agentSessionId: input.agentSessionId,
            ...(ctx.userId ? { userId: ctx.userId } : {}),
          },
          select: {
            id: true,
            name: true,
            languageId: true,
            status: true,
            judgeStatus: true,
            exitCode: true,
            timeMs: true,
            createdAt: true,
            completedAt: true,
            artifacts: {
              select: { id: true, name: true, mimeType: true, sizeBytes: true, createdAt: true },
              orderBy: { createdAt: 'asc' },
            },
          },
          orderBy: { createdAt: 'desc' },
          take: 20,
        });
        const totalArtifacts = executions.reduce((sum, e) => sum + e.artifacts.length, 0);
        const tree = executions.map((e) => ({
          executionId: e.id,
          name: e.name,
          languageId: e.languageId,
          status: e.status,
          judgeStatus: e.judgeStatus,
          exitCode: e.exitCode,
          timeMs: e.timeMs,
          createdAt: e.createdAt,
          completedAt: e.completedAt,
          artifacts: e.artifacts.map((a) => ({
            artifactId: a.id,
            name: a.name,
            mimeType: a.mimeType,
            sizeBytes: a.sizeBytes,
            size: formatBytes(a.sizeBytes),
            createdAt: a.createdAt,
          })),
        }));
        return JSON.stringify(
          {
            agentSessionId: input.agentSessionId,
            executions: tree.length,
            totalArtifacts,
            recent: tree,
            ...(tree.length === 0
              ? { note: 'No runs recorded for this session yet. Use execute_code_sandbox to run code first.' }
              : {}),
          },
          null,
          2,
        );
      } catch (error) {
        return `Error: workspace_list_files failed: ${errorMessage(error)}`;
      }
    },
    {
      name: 'workspace_list_files',
      description:
        'List recent runs and their file artifacts for an agent session (file/artifact tree summary). Use to see what the session has produced before reading a specific artifact.',
      schema: workspaceInputSchema,
    },
  );
}

/** Session id travels in the input; user scoping applies when bound. */
export const workspaceListFilesTool = createWorkspaceTool();
