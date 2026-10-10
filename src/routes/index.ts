import { Router } from 'express';
import { authRouter } from '../modules/auth/auth.routes';
import { agentRouter } from '../modules/agent/agent.router';
import { executionRouter } from '../modules/executions/execution.routes';

export const apiRouter = Router();

apiRouter.use('/auth', authRouter);
apiRouter.use('/executions', executionRouter);
apiRouter.use('/agent', agentRouter);
