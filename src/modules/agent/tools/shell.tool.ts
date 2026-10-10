import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { createExecution } from '../../executions/execution.service';
import {
  MAX_OUTPUT_CHARS,
  errorMessage,
  isTerminalStatus,
  pollExecutionToCompletion,
  stripArtifactMarkers,
  truncate,
  type ToolContext,
} from './common';

export const shellInputSchema = z.object({
  command: z.string().min(1).max(100_000),
});

export type ShellInput = z.infer<typeof shellInputSchema>;

const BASH_LANGUAGE_ID = 46;

/**
 * Run a shell command inside the sandbox (bash, languageId 46).
 * Shares the sandbox polling/stripping behaviour with execute_code_sandbox.
 */
export function createShellTool(ctx: ToolContext = {}) {
  return tool(
    async (input: ShellInput): Promise<string> => {
      try {
        if (!ctx.userId) {
          return 'Error: shell_executor needs a bound session (userId). Use getToolsForSession(userId, agentSessionId).';
        }
        const created = await createExecution({
          userId: ctx.userId,
          code: input.command,
          languageId: BASH_LANGUAGE_ID,
          agentSessionId: ctx.agentSessionId,
          name: ctx.agentSessionId ? `agent-shell:${ctx.agentSessionId.slice(0, 8)}` : 'agent-shell',
        });
        const execution = await pollExecutionToCompletion(created.id, ctx.userId);
        const stripped = stripArtifactMarkers(execution.stdout ?? '');
        return JSON.stringify(
          {
            executionId: execution.id,
            status: execution.status,
            timedOut: !isTerminalStatus(execution.status),
            exitCode: execution.exitCode,
            stdout: truncate(stripped.clean, MAX_OUTPUT_CHARS),
            stderr: truncate(execution.stderr, MAX_OUTPUT_CHARS),
            timeMs: execution.timeMs,
            errorMessage: execution.errorMessage,
          },
          null,
          2,
        );
      } catch (error) {
        return `Error: shell_executor failed: ${errorMessage(error)}`;
      }
    },
    {
      name: 'shell_executor',
      description:
        'Run a bash shell command in the sandbox and return exit code, stdout and stderr. Prefer execute_code_sandbox for scripts; use this for one-shot shell commands.',
      schema: shellInputSchema,
    },
  );
}

/** Unbound instance for the static registry; bind via getToolsForSession for real runs. */
export const shellExecutorTool = createShellTool();
