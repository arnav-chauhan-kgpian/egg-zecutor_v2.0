-- Move the Role and ExecutionStatus enums out of the database and into the
-- application (src/lib/enums.ts).
--
-- WHY: SQLite has no enum type and Prisma will not generate an `enum` block for
-- it, but `provider` cannot be read from an env var — so supporting PostgreSQL
-- (the deployed stack) and SQLite (the zero-setup local default, which is what
-- lets `npm run dev` work with no database server and no Docker) from one
-- schema requires these columns to be TEXT.
--
-- Non-destructive: the stored values are unchanged, only their column type is.
-- The enum labels and the union types that replace them are identical, so
-- existing rows remain valid.

-- The default is enum-typed, so it has to be dropped before the cast and
-- restored as text afterwards.
ALTER TABLE "users" ALTER COLUMN "role" DROP DEFAULT;
ALTER TABLE "users" ALTER COLUMN "role" SET DATA TYPE TEXT USING "role"::TEXT;
ALTER TABLE "users" ALTER COLUMN "role" SET DEFAULT 'USER';

ALTER TABLE "executions" ALTER COLUMN "status" DROP DEFAULT;
ALTER TABLE "executions" ALTER COLUMN "status" SET DATA TYPE TEXT USING "status"::TEXT;
ALTER TABLE "executions" ALTER COLUMN "status" SET DEFAULT 'PENDING';

-- Nothing references the types now that both columns are TEXT.
DROP TYPE "Role";
DROP TYPE "ExecutionStatus";
