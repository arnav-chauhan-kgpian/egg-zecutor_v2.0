import { Annotation, messagesStateReducer } from '@langchain/langgraph';
import type { BaseMessage } from '@langchain/core/messages';

/**
 * Human-in-the-loop decision carried in graph state. `null` = no decision yet.
 */
export const ApprovalStatus = {
  APPROVED: 'APPROVED',
  REJECTED: 'REJECTED',
  PENDING: 'PENDING',
} as const;

export type ApprovalStatus = (typeof ApprovalStatus)[keyof typeof ApprovalStatus] | null;

/** Overwrite-with-latest reducer for scalar channels. */
function latest<T>(left: T, right?: T): T {
  return right === undefined ? left : right;
}

/**
 * Agent graph state. `agentSessionId` is used verbatim as the LangGraph
 * `thread_id`, so checkpoints line up 1:1 with the AgentSession row.
 */
export const AgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  userId: Annotation<string>({
    reducer: latest<string>,
    default: () => '',
  }),
  agentSessionId: Annotation<string>({
    reducer: latest<string>,
    default: () => '',
  }),
  approvalRequired: Annotation<boolean>({
    reducer: latest<boolean>,
    default: () => false,
  }),
  approvalStatus: Annotation<ApprovalStatus>({
    reducer: latest<ApprovalStatus>,
    default: () => null,
  }),
  /**
   * Tool allowlist for this turn, chosen by the user (ChatGPT-style toggles).
   * Empty = all tools enabled (also the pre-feature default, so old
   * checkpoints and clients without the field keep working).
   */
  enabledTools: Annotation<string[]>({
    reducer: latest<string[]>,
    default: () => [],
  }),
});

export type AgentStateType = typeof AgentState.State;
export type AgentStateUpdate = typeof AgentState.Update;
