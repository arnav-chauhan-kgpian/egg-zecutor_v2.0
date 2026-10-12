import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { createExecution } from '../../executions/execution.service';
import {
  MAX_OUTPUT_CHARS,
  POLL_TIMEOUT_MS,
  errorMessage,
  isTerminalStatus,
  pollExecutionToCompletion,
  stripArtifactMarkers,
  truncate,
  type ToolContext,
} from './common';

export const sandboxInputSchema = z.object({
  code: z.string().min(1).max(1_000_000).describe('Source code to execute.'),
  languageId: z
    .number()
    .int()
    .positive()
    .default(71)
    .describe('Judge0 language id (71 = Python 3, 63 = Node.js, 54 = C++, 50 = C, 46 = bash).'),
  stdin: z.string().max(1_000_000).optional().describe('Optional text fed to the program on stdin.'),
  timeLimit: z
    .number()
    .positive()
    .optional()
    .describe(
      'Maximum CPU time in whole SECONDS (NOT milliseconds), ideally 5-300. Values outside 1..900 are clamped into range. Omit for the engine default (~120s); short scripts need no timeLimit at all.',
    ),
});

export type SandboxInput = z.infer<typeof sandboxInputSchema>;

/**
 * Run code in the ExecLab sandbox and wait for the verdict.
 *
 * Large `::artifact::` base64 blobs are never returned verbatim: the engine
 * already lifts them into Artifact rows, and any residual markers are
 * replaced with `[Artifact created: name (size)]` summaries. Artifacts come
 * back as metadata — use the read_artifact tool to inspect one.
 */
export function createSandboxTool(ctx: ToolContext = {}) {
  return tool(
    async (input: SandboxInput): Promise<string> => {
      try {
        if (!ctx.userId) {
          return 'Error: execute_code_sandbox needs a bound session (userId). Use getToolsForSession(userId, agentSessionId).';
        }
        // Clamp rather than reject: weaker models guess milliseconds or huge
        // values, and a turn-killing schema error is worse than a sane limit.
        const timeLimit =
          input.timeLimit === undefined
            ? undefined
            : Math.min(900, Math.max(1, Math.floor(input.timeLimit)));
        const created = await createExecution({
          userId: ctx.userId,
          code: input.code,
          languageId: input.languageId,
          stdin: input.stdin,
          timeLimit,
          agentSessionId: ctx.agentSessionId,
          name: ctx.agentSessionId ? `agent:${ctx.agentSessionId.slice(0, 8)}` : 'agent:adhoc',
        });
        const execution = await pollExecutionToCompletion(created.id, ctx.userId);
        const stripped = stripArtifactMarkers(execution.stdout ?? '');
        const timedOut = !isTerminalStatus(execution.status);
        return JSON.stringify(
          {
            executionId: execution.id,
            status: execution.status,
            timedOut,
            ...(timedOut ? { pollAfterMs: POLL_TIMEOUT_MS } : {}),
            judgeStatus: execution.judgeStatus,
            exitCode: execution.exitCode,
            stdout: truncate(stripped.clean, MAX_OUTPUT_CHARS),
            stderr: truncate(execution.stderr, MAX_OUTPUT_CHARS),
            compileOutput: truncate(execution.compileOutput, MAX_OUTPUT_CHARS),
            timeMs: execution.timeMs,
            memoryKb: execution.memoryKb,
            errorMessage: execution.errorMessage,
            artifacts: execution.artifacts,
          },
          null,
          2,
        );
      } catch (error) {
        return `Error: execute_code_sandbox failed: ${errorMessage(error)}`;
      }
    },
    {
      name: 'execute_code_sandbox',
      description:
        'Execute code (default Python 3, languageId 71) in a sandbox and return exit code, stdout, stderr, timing and artifact metadata. Use for running computations, data analysis and generating files/plots.',
      schema: sandboxInputSchema,
    },
  );
}

/** Unbound instance for the static registry; bind via getToolsForSession for real runs. */
export const executeCodeSandboxTool = createSandboxTool();
