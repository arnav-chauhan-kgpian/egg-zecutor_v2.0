import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import { corsOrigins, env } from './config/env';
import { errorHandler, notFoundHandler } from './middleware/errorHandler';
import { apiRouter } from './routes';
import { engineInfo } from './services/execution';

export function createApp() {
  const app = express();

  app.use(helmet());
  app.use(cors({ origin: corsOrigins }));
  // Generous: `additionalFiles` carries a base64 dataset zip, and a research
  // script can legitimately be large. Bounded further by the route schema.
  app.use(express.json({ limit: env.REQUEST_BODY_LIMIT }));
  app.use(express.urlencoded({ extended: true, limit: env.REQUEST_BODY_LIMIT }));

  if (env.NODE_ENV !== 'test') {
    app.use(morgan(env.NODE_ENV === 'development' ? 'dev' : 'combined'));
  }

  app.get('/health', (_req, res) => {
    res.json({ status: 'ok', uptime: process.uptime(), engine: engineInfo });
  });

  // Versioned from the start: the execution contract is the product surface
  // here, and the webhook URL handed to Judge0 has to stay stable.
  app.use('/api/v1', apiRouter);
  // Unversioned alias so existing clients (and the auth flow) keep working.
  app.use('/api', apiRouter);

  app.use(notFoundHandler);
  app.use(errorHandler);

  return app;
}
