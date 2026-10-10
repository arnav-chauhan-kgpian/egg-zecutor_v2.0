/**
 * Domain enums, defined here rather than in the Prisma schema.
 *
 * SQLite has no native enum type, and Prisma refuses to generate an `enum`
 * block for it. Supporting both PostgreSQL (the deployed stack) and SQLite (the
 * zero-setup local default, so `npm run dev` needs no database server) from a
 * single schema therefore means the columns are `String`.
 *
 * The trade is that the *database* no longer rejects an unknown value; the
 * union types below restore that at compile time, which is where these values
 * are actually written from. Every call site keeps its original shape —
 * `ExecutionStatus.PENDING`, `Role.ADMIN` — so this reads identically to the
 * generated enums it replaces.
 */

export const ExecutionStatus = {
  PENDING: 'PENDING',
  PROCESSING: 'PROCESSING',
  COMPLETED: 'COMPLETED',
  FAILED: 'FAILED',
} as const;

export type ExecutionStatus = (typeof ExecutionStatus)[keyof typeof ExecutionStatus];

export const Role = {
  USER: 'USER',
  ADMIN: 'ADMIN',
} as const;

export type Role = (typeof Role)[keyof typeof Role];

/**
 * Narrows a role loaded from the database.
 *
 * The column is TEXT, so unlike a native enum the database will happily return
 * a value this build has never heard of — a hand-edited row, or a rollback
 * after a new role shipped. Fail closed: anything unrecognised becomes the
 * least privileged role rather than being trusted into a signed token.
 */
export function toRole(value: string): Role {
  return value === Role.ADMIN ? Role.ADMIN : Role.USER;
}
