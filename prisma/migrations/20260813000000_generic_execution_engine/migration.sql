-- Refactor: LeetCode-style grading platform -> generic execution engine.
--
-- DESTRUCTIVE. Drops problems, test_cases and submissions along with their
-- data. There is no back-fill: the old rows model problem/verdict pairs that
-- have no meaning in the new schema (no expected output, no hidden cases).
-- `users` is preserved, so accounts and logins survive.

-- CreateEnum
CREATE TYPE "ExecutionStatus" AS ENUM ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED');

-- DropForeignKey
ALTER TABLE "submissions" DROP CONSTRAINT "submissions_problemId_fkey";

-- DropForeignKey
ALTER TABLE "submissions" DROP CONSTRAINT "submissions_userId_fkey";

-- DropForeignKey
ALTER TABLE "test_cases" DROP CONSTRAINT "test_cases_problemId_fkey";

-- DropTable
DROP TABLE "problems";

-- DropTable
DROP TABLE "submissions";

-- DropTable
DROP TABLE "test_cases";

-- DropEnum
DROP TYPE "Difficulty";

-- DropEnum
DROP TYPE "SubmissionStatus";

-- CreateTable
CREATE TABLE "executions" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "code" TEXT NOT NULL,
    "languageId" INTEGER NOT NULL,
    "name" TEXT,
    "status" "ExecutionStatus" NOT NULL DEFAULT 'PENDING',
    "stdin" TEXT,
    "additionalFiles" TEXT,
    "timeLimit" DOUBLE PRECISION,
    "memoryLimit" INTEGER,
    "stdout" TEXT,
    "stderr" TEXT,
    "compileOutput" TEXT,
    "judgeStatus" TEXT,
    "exitCode" INTEGER,
    "timeMs" INTEGER,
    "memoryKb" INTEGER,
    "errorMessage" TEXT,
    "judge0Token" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "startedAt" TIMESTAMP(3),
    "completedAt" TIMESTAMP(3),

    CONSTRAINT "executions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "artifacts" (
    "id" TEXT NOT NULL,
    "executionId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "mimeType" TEXT NOT NULL DEFAULT 'application/octet-stream',
    "content" TEXT NOT NULL,
    "sizeBytes" INTEGER NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "artifacts_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "executions_judge0Token_key" ON "executions"("judge0Token");

-- CreateIndex
CREATE INDEX "executions_userId_createdAt_idx" ON "executions"("userId", "createdAt");

-- CreateIndex
CREATE INDEX "executions_status_idx" ON "executions"("status");

-- CreateIndex
CREATE INDEX "artifacts_executionId_idx" ON "artifacts"("executionId");

-- AddForeignKey
ALTER TABLE "executions" ADD CONSTRAINT "executions_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "artifacts" ADD CONSTRAINT "artifacts_executionId_fkey" FOREIGN KEY ("executionId") REFERENCES "executions"("id") ON DELETE CASCADE ON UPDATE CASCADE;
