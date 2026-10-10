import bcrypt from 'bcryptjs';
import type { User } from '@prisma/client';
import { env } from '../../config/env';
import { toRole } from '../../lib/enums';
import { prisma } from '../../lib/prisma';
import { ApiError } from '../../utils/ApiError';
import { signAccessToken } from '../../utils/jwt';
import type { LoginInput, RegisterInput } from './auth.schemas';

export type PublicUser = Omit<User, 'passwordHash'>;

export function toPublicUser(user: User): PublicUser {
  const { passwordHash: _passwordHash, ...rest } = user;
  return rest;
}

function issueToken(user: User) {
  return signAccessToken({
    sub: user.id,
    email: user.email,
    username: user.username,
    // The column is TEXT rather than a DB enum (so one schema serves both
    // PostgreSQL and SQLite), so narrow before it becomes a signed claim.
    role: toRole(user.role),
  });
}

export async function register(input: RegisterInput) {
  const existing = await prisma.user.findFirst({
    where: { OR: [{ email: input.email }, { username: input.username }] },
    select: { email: true, username: true },
  });

  if (existing) {
    throw ApiError.conflict(
      existing.email === input.email ? 'Email is already registered' : 'Username is already taken',
    );
  }

  const passwordHash = await bcrypt.hash(input.password, env.BCRYPT_SALT_ROUNDS);

  const user = await prisma.user.create({
    data: {
      email: input.email,
      username: input.username,
      passwordHash,
    },
  });

  return { user: toPublicUser(user), token: issueToken(user) };
}

export async function login(input: LoginInput) {
  const identifier = input.identifier.toLowerCase();

  const user = await prisma.user.findFirst({
    where: { OR: [{ email: identifier }, { username: input.identifier }] },
  });

  // Compare against a dummy hash when the user is missing so response timing
  // does not reveal whether the account exists.
  const passwordMatches = user
    ? await bcrypt.compare(input.password, user.passwordHash)
    : await bcrypt.compare(input.password, DUMMY_HASH);

  if (!user || !passwordMatches) {
    throw ApiError.unauthorized('Invalid credentials');
  }

  return { user: toPublicUser(user), token: issueToken(user) };
}

export async function getUserById(id: string): Promise<PublicUser> {
  const user = await prisma.user.findUnique({ where: { id } });
  if (!user) {
    throw ApiError.notFound('User not found');
  }
  return toPublicUser(user);
}

// bcrypt hash of a random string; only used to equalise timing on login.
const DUMMY_HASH = '$2a$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy';
