import { Router } from 'express';
import rateLimit, { ipKeyGenerator } from 'express-rate-limit';
import { z } from 'zod';
import { env } from '../../config/env';
import { requireAuth } from '../../middleware/auth';
import { validateBody } from '../../middleware/validate';
import {
  backend,
  engineInfo,
  judge0ToOutcome,
  judge0IsTerminal,
  LANGUAGES,
  type Judge0Submission,
} from '../../services/execution';
import { ApiError } from '../../utils/ApiError';
import { asyncHandler } from '../../utils/asyncHandler';
import {
  createExecution,
  deleteExecution,
  finalize,
  findByToken,
  getArtifact,
  getExecution,
  listExecutions,
} from './execution.service';

export const executionRouter = Router();

// --- validation --------------------------------------------------------------

const BASE64 = /^[A-Za-z0-9+/]*={0,2}$/;

const createSchema = z.object({
  code: z.string().min(1, 'code cannot be empty').max(1_000_000, 'code is too large'),
  // Numeric so any language the configured Judge0 instance supports works
  // without an API change.
  languageId: z.coerce.number().int().positive(),
  name: z.string().max(200).optional(),
  stdin: z.string().max(1_000_000).optional(),
  additionalFiles: z
    .string()
    .max(env.ADDITIONAL_FILES_MAX_BASE64_BYTES, 'additionalFiles exceeds the configured limit')
    .refine((value) => BASE64.test(value), 'additionalFiles must be base64')
    .optional(),
  timeLimit: z.coerce.number().positive().max(env.JUDGE0_CPU_TIME_LIMIT_MAX).optional(),
  memoryLimit: z.coerce.number().int().positive().max(env.JUDGE0_MEMORY_LIMIT_MAX).optional(),
});

const listSchema = z.object({
  page: z.coerce.number().int().positive().default(1),
  pageSize: z.coerce.number().int().positive().max(100).default(20),
});

// --- engine metadata ----------------------------------------------------------
// Unauthenticated: the playground needs the language list to render its picker
// before a user signs in, and none of this is sensitive.

executionRouter.get('/languages', (_req, res) => {
  res.json({ languages: LANGUAGES });
});

executionRouter.get(
  '/engine',
  asyncHandler(async (_req, res) => {
    res.json({ engine: { ...engineInfo, healthy: await backend.health() } });
  }),
);

// --- Judge0 webhook -----------------------------------------------------------
// Declared before `/:id` so the literal path wins.
//
// Judge0 sends the finished submission to `callback_url`. It uses PUT by
// default, so both verbs are accepted. This route is NOT behind requireAuth —
// Judge0 has no user session — so it is guarded by a shared secret instead.

function assertCallbackAuthorised(req: {
  header: (name: string) => string | undefined;
  query: Record<string, unknown>;
}): void {
  if (!env.JUDGE0_CALLBACK_SECRET) return;

  const provided = req.header('x-callback-secret') ?? String(req.query.secret ?? '');
  if (provided !== env.JUDGE0_CALLBACK_SECRET) {
    throw new ApiError(401, 'Invalid callback secret');
  }
}

const handleCallback = asyncHandler(async (req, res) => {
  assertCallbackAuthorised(req);

  const submission = req.body as Judge0Submission;
  const token = submission?.token;

  if (!token) throw new ApiError(400, 'Callback payload has no token');

  const execution = await findByToken(token);
  // 200 rather than 404: an unknown token means the row was deleted or belongs
  // to another deployment, and Judge0 should not keep retrying either way.
  if (!execution) {
    res.status(200).json({ ok: true, ignored: 'unknown token' });
    return;
  }

  const statusId = submission.status?.id ?? 0;
  if (!judge0IsTerminal(statusId)) {
    // Judge0 occasionally calls back on an intermediate state; wait for the
    // real one rather than writing a half-finished result.
    res.status(200).json({ ok: true, ignored: 'non-terminal status' });
    return;
  }

  // Judge0 serialises the callback with SubmissionSerializer.default_fields —
  // token, time, memory, stdout, stderr, compile_output, message, status.
  // `exit_code` is NOT among them, so the payload on its own records every
  // webhook-delivered run with a null exit code, while the reconciler (which
  // asks for the field explicitly) records the real one. Same run, different
  // answer depending on which path happened to win.
  //
  // The key is absent rather than null when Judge0 omitted it, so re-reading
  // the submission is confined to that case: a genuine null exit code (a
  // compile error never ran) is left alone, and the webhook stays the fast
  // path for everything it does report.
  let outcome = judge0ToOutcome(submission);

  if (!('exit_code' in submission) && backend.poll) {
    const authoritative = await backend.poll(token).catch((error: unknown) => {
      console.warn(
        `[callback] ${token}: re-read failed —`,
        error instanceof Error ? error.message : error,
      );
      return null;
    });

    // Judge0 fires this callback from an `ensure` block, and the write it is
    // reporting is not reliably visible to a concurrent reader yet — the
    // re-read comes back "Processing" often enough to matter. Committing the
    // partial payload here would pin a terminal row with a null exit code that
    // nothing ever revisits, so hand the run to the reconciler instead: it
    // sweeps every JUDGE0_POLL_INTERVAL_MS and polls with the full field list.
    // Costs a couple of seconds in the racy case and is always complete.
    if (!authoritative) {
      res.status(200).json({ ok: true, deferred: 'reconciler' });
      return;
    }

    outcome = authoritative;
  }

  const applied = await finalize(execution.id, outcome);
  res.status(200).json({ ok: true, applied });
});

executionRouter.post('/callback', handleCallback);
executionRouter.put('/callback', handleCallback);

// --- executions ---------------------------------------------------------------

executionRouter.use(requireAuth);

/**
 * Per-user, not per-IP: a shared demo is often several people behind one NAT,
 * and keying on IP would have them throttling each other. Every request here
 * is past requireAuth, so req.user is always set.
 */
const runLimiter = rateLimit({
  windowMs: env.RATE_LIMIT_WINDOW_MS,
  limit: env.RATE_LIMIT_MAX_RUNS,
  standardHeaders: 'draft-7',
  legacyHeaders: false,
  // req.user is always set (this router is behind requireAuth). The IP
  // fallback goes through ipKeyGenerator so an IPv6 client cannot sidestep the
  // limit by varying the low bits of its /64.
  keyGenerator: (req, res) => req.user?.id ?? ipKeyGenerator(req.ip ?? '', 56) ?? 'anonymous',
  message: {
    error: { message: 'Too many runs queued. Wait a moment and try again.' },
  },
});

/**
 * Queues a run. Responds 202 with the record in PENDING/PROCESSING — poll
 * GET /:id (or wait for the UI's poller) for the result.
 *
 * Exposed at both `/executions` (REST-shaped) and `/executions/run` (verb-shaped,
 * and the path the deployment docs quote). Same handler, same semantics — the
 * alias exists so neither convention is wrong.
 */
executionRouter.post(
  ['/', '/run'],
  runLimiter,
  validateBody(createSchema),
  asyncHandler(async (req, res) => {
    const body = req.body as z.infer<typeof createSchema>;
    const execution = await createExecution({ userId: req.user!.id, ...body });
    res.status(202).json({ execution });
  }),
);

executionRouter.get(
  '/',
  asyncHandler(async (req, res) => {
    const { page, pageSize } = listSchema.parse(req.query);
    res.json(await listExecutions(req.user!.id, page, pageSize));
  }),
);

executionRouter.get(
  '/:id',
  asyncHandler(async (req, res) => {
    res.json({ execution: await getExecution(req.params.id!, req.user!.id) });
  }),
);

executionRouter.delete(
  '/:id',
  asyncHandler(async (req, res) => {
    await deleteExecution(req.params.id!, req.user!.id);
    res.status(204).end();
  }),
);

/** Streams an artifact back with its declared type, as a download. */
executionRouter.get(
  '/:id/artifacts/:artifactId',
  asyncHandler(async (req, res) => {
    const artifact = await getArtifact(req.params.id!, req.params.artifactId!, req.user!.id);
    const buffer = Buffer.from(artifact.content, 'base64');

    res.setHeader('Content-Type', artifact.mimeType);
    res.setHeader('Content-Length', String(buffer.length));
    // The name is sanitised to a basename at parse time, so it is safe here.
    res.setHeader('Content-Disposition', `attachment; filename="${artifact.name}"`);
    res.send(buffer);
  }),
);
