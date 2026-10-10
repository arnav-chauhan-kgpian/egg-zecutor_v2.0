import { Router } from 'express';
import { requireAuth } from '../../middleware/auth';
import { validateBody, validateQuery } from '../../middleware/validate';
import {
  chatHandler,
  chatSchema,
  getSessionHandler,
  listSessionsHandler,
  listToolsHandler,
  resumeHandler,
  resumeSchema,
  sessionListSchema,
} from './agent.controller';

export const agentRouter = Router();

// All agent endpoints require a signed-in owner; every handler scopes by req.user.id.
agentRouter.use(requireAuth);

agentRouter.post('/chat', validateBody(chatSchema), chatHandler);
agentRouter.post('/resume', validateBody(resumeSchema), resumeHandler);
agentRouter.get('/tools', listToolsHandler);
agentRouter.get('/sessions', validateQuery(sessionListSchema), listSessionsHandler);
agentRouter.get('/sessions/:id', getSessionHandler);
