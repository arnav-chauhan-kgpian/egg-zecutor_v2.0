import type { DynamicStructuredTool } from '@langchain/core/tools';
import { createReadArtifactTool, readArtifactTool } from './artifact.tool';
import { calculatorEngineTool, createCalculatorTool } from './calculator.tool';
import type { ToolContext } from './common';
import { createSaveMemoryTool, saveMemoryArtifactTool } from './memory.tool';
import { createRagTool, vectorRagSearchTool } from './rag.tool';
import { createSandboxTool, executeCodeSandboxTool } from './sandbox.tool';
import { createShellTool, shellExecutorTool } from './shell.tool';
import { createWebSearchTool, webSearchTool } from './webSearch.tool';
import { createWorkspaceTool, workspaceListFilesTool } from './workspace.tool';

export type { ToolContext };
export {
  calculatorEngineTool,
  createCalculatorTool,
  createReadArtifactTool,
  createRagTool,
  createSandboxTool,
  createSaveMemoryTool,
  createShellTool,
  createWebSearchTool,
  createWorkspaceTool,
  executeCodeSandboxTool,
  readArtifactTool,
  saveMemoryArtifactTool,
  shellExecutorTool,
  vectorRagSearchTool,
  webSearchTool,
  workspaceListFilesTool,
};

/**
 * Static registry of all 8 agent tools with no session bound.
 * User-scoped tools (sandbox, shell, rag, memory) return a plain-text
 * "needs a bound session" message until bound — stateless tools
 * (artifact read, web search, calculator, workspace listing) work as-is.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyTool = DynamicStructuredTool<any, any, any, any, any, string>;

export const allAgentTools: AnyTool[] = [
  executeCodeSandboxTool as unknown as AnyTool,
  shellExecutorTool as unknown as AnyTool,
  readArtifactTool as unknown as AnyTool,
  vectorRagSearchTool as unknown as AnyTool,
  saveMemoryArtifactTool as unknown as AnyTool,
  webSearchTool as unknown as AnyTool,
  calculatorEngineTool as unknown as AnyTool,
  workspaceListFilesTool as unknown as AnyTool,
];

/**
 * Build session-bound tools for one agent turn/session. The returned
 * instances close over `userId`/`agentSessionId`, so the LLM never handles
 * ownership identifiers — user-scoped reads/writes are pre-scoped, and new
 * executions are labelled with the session.
 *
 * `enabledTools` is the user's ChatGPT-style allowlist: only named tools are
 * bound. Empty/omitted = all tools (the default).
 */
export function getToolsForSession(
  userId: string,
  agentSessionId: string,
  enabledTools?: string[],
): AnyTool[] {
  const ctx: ToolContext = { userId, agentSessionId };
  const all = [
    createSandboxTool(ctx) as unknown as AnyTool,
    createShellTool(ctx) as unknown as AnyTool,
    createReadArtifactTool(ctx) as unknown as AnyTool,
    createRagTool(ctx) as unknown as AnyTool,
    createSaveMemoryTool(ctx) as unknown as AnyTool,
    createWebSearchTool(ctx) as unknown as AnyTool,
    createCalculatorTool(ctx) as unknown as AnyTool,
    createWorkspaceTool(ctx) as unknown as AnyTool,
  ];
  if (!enabledTools || enabledTools.length === 0) return all;
  const wanted = new Set(enabledTools);
  return all.filter((tool) => wanted.has(tool.name));
}

/** Names the model may be offered (allowlist vocabulary for user selection). */
export const AGENT_TOOL_NAMES: string[] = allAgentTools.map((tool) => tool.name);

export function isKnownAgentTool(name: string): boolean {
  return AGENT_TOOL_NAMES.includes(name);
}

/** Catalog for the user-facing tool picker (GET /agent/tools). */
export function describeAgentTools(): Array<{ name: string; description: string }> {
  return allAgentTools.map((tool) => ({ name: tool.name, description: tool.description }));
}
