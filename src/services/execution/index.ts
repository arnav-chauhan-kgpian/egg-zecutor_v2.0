import { env, executorKind } from '../../config/env';
import { dockerBackend } from './dockerBackend';
import { judge0Backend } from './judge0Backend';
import { nativeBackend } from './nativeBackend';
import type { ExecutionBackend } from './types';

/**
 * The configured backend.
 *
 * Judge0 is the intended engine. The Docker backend is the fallback for hosts
 * where Judge0 physically cannot run (cgroup v2). The native backend is the
 * fallback for hosts with no container runtime at all — no sandbox, but no
 * setup either. All three implement the identical contract, so nothing above
 * this line changes.
 */
const BACKENDS: Record<typeof executorKind, ExecutionBackend> = {
  judge0: judge0Backend,
  docker: dockerBackend,
  native: nativeBackend,
};

export const backend: ExecutionBackend = BACKENDS[executorKind];

const ENDPOINTS: Record<typeof executorKind, string | null> = {
  judge0: env.JUDGE0_API_URL ?? null,
  docker: 'local-docker',
  native: 'local-process',
};

export const engineInfo = {
  kind: backend.kind,
  usesCallback: backend.usesCallback,
  endpoint: ENDPOINTS[executorKind],
  callbackUrl: env.JUDGE0_CALLBACK_URL || null,
  /** True when submitted code runs without isolation — surfaced in the UI. */
  sandboxed: executorKind !== 'native',
};

export * from './types';
export * from './languages';
export * from './artifacts';
export { toOutcome as judge0ToOutcome, isTerminal as judge0IsTerminal } from './judge0Backend';
export { availableNativeLanguages } from './nativeBackend';
export type { Judge0Submission } from './judge0Backend';
