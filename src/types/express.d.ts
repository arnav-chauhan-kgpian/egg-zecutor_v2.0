import type { Role } from '../lib/enums';

declare global {
  namespace Express {
    interface Request {
      user?: {
        id: string;
        email: string;
        username: string;
        role: Role;
      };
    }
  }
}

export {};
