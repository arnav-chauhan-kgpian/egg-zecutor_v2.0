import { Router } from 'express';
import rateLimit from 'express-rate-limit';
import { env } from '../../config/env';
import { requireAuth } from '../../middleware/auth';
import { validateBody } from '../../middleware/validate';
import { loginHandler, meHandler, registerHandler } from './auth.controller';
import { loginSchema, registerSchema } from './auth.schemas';

export const authRouter = Router();

/**
 * Per-IP, unlike the run limiter: these routes are pre-auth, so there is no
 * user to key on. Bounds credential stuffing and stops a demo audience from
 * accidentally filling the users table.
 */
const authLimiter = rateLimit({
  windowMs: env.RATE_LIMIT_WINDOW_MS,
  limit: env.RATE_LIMIT_MAX_AUTH,
  standardHeaders: 'draft-7',
  legacyHeaders: false,
  // Only failed attempts count, so a working demo login never locks anyone out.
  skipSuccessfulRequests: true,
  message: { error: { message: 'Too many attempts. Wait a moment and try again.' } },
});

authRouter.post('/register', authLimiter, validateBody(registerSchema), registerHandler);
authRouter.post('/login', authLimiter, validateBody(loginSchema), loginHandler);

// Protected: demonstrates the auth middleware.
authRouter.get('/me', requireAuth, meHandler);
