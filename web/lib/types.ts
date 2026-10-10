export type ExecutionStatus = 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

export interface LanguageInfo {
  id: number;
  name: string;
  ext: string;
}

export interface ArtifactSummary {
  id: string;
  name: string;
  mimeType: string;
  sizeBytes: number;
  createdAt: string;
}

export interface ExecutionSummary {
  id: string;
  name: string | null;
  languageId: number;
  languageName: string;
  status: ExecutionStatus;
  judgeStatus: string | null;
  exitCode: number | null;
  timeMs: number | null;
  memoryKb: number | null;
  createdAt: string;
  completedAt: string | null;
}

export interface Execution extends ExecutionSummary {
  code: string;
  stdin: string | null;
  timeLimit: number | null;
  memoryLimit: number | null;
  stdout: string | null;
  stderr: string | null;
  compileOutput: string | null;
  errorMessage: string | null;
  updatedAt: string;
  startedAt: string | null;
  artifacts: ArtifactSummary[];
}

export interface Pagination {
  page: number;
  pageSize: number;
  total: number;
}

export interface EngineInfo {
  kind: 'judge0' | 'docker';
  usesCallback: boolean;
  endpoint: string | null;
  callbackUrl: string | null;
  healthy: boolean;
}

export interface AuthUser {
  id: string;
  email: string;
  username: string;
  role: 'USER' | 'ADMIN';
}

export interface AuthResponse {
  user: AuthUser;
  token: string;
}

/** A run is finished once the engine stops working on it. */
export function isTerminal(status: ExecutionStatus): boolean {
  return status === 'COMPLETED' || status === 'FAILED';
}

// --- agent ------------------------------------------------------------------

export type AgentSessionStatus = 'ACTIVE' | 'SUSPENDED' | 'COMPLETED';

export interface AgentSessionSummary {
  id: string;
  userId?: string;
  status: AgentSessionStatus;
  createdAt: string;
  updatedAt: string;
  executionCount?: number;
}

export interface AgentToolCallSummary {
  name: string;
  args: unknown;
}

/** One entry of the GET /agent/tools catalog backing the picker UI. */
export interface AgentToolInfo {
  name: string;
  description: string;
}

export interface AgentChatMessage {
  id?: string;
  type: 'human' | 'ai' | 'tool' | 'system';
  name?: string;
  content: string;
  tool_calls?: AgentToolCallSummary[];
  tool_call_id?: string;
}

export interface AgentSessionExecution {
  id: string;
  name: string | null;
  languageId: number;
  status: ExecutionStatus;
  judgeStatus: string | null;
  exitCode: number | null;
  timeMs: number | null;
  createdAt: string;
  completedAt: string | null;
  artifacts: ArtifactSummary[];
}

export interface AgentSessionDetail {
  session: AgentSessionSummary;
  messages: AgentChatMessage[];
  executions: AgentSessionExecution[];
}
