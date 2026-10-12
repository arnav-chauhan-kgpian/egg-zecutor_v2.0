# ExecLab

[![CI](https://github.com/arnav-chauhan-kgpian/egg-zecutor_v2.0/actions/workflows/ci.yml/badge.svg)](https://github.com/arnav-chauhan-kgpian/egg-zecutor_v2.0/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-strict-3178c6.svg)](./tsconfig.json)

A generic code execution engine for AI research workloads, powered by Judge0. Node.js + Express +
TypeScript API with Prisma on PostgreSQL and JWT auth, plus a Next.js execution playground in
[`web/`](./web/README.md).

> **What's new in ExecLab:**
> LangGraph orchestrator (Groq default, OpenAI fallback) with **8 user-toggleable
> tools** (sandbox, shell, calculator, web search, vector memory, artifacts, workspace),
> **human-in-the-loop approvals** for sensitive actions, SSE-streamed chat
> (`/api/v1/agent/*`), pgvector-backed long-term memory, and an agent assistant
> pane in the playground. See `src/modules/agent/` and `web/components/agent/`.

Run one script, under limits you choose, and get back everything it produced: stdout, stderr,
compiler output, exit code, CPU time, peak memory, and any files it emitted.

```
.        API (Express, port 4000)
└─ web/  Execution Playground (Next.js App Router, port 3000)
```

> **Just want to run it?** `npm install && npm run dev`, then open
> <http://localhost:3000/playground>. That one command generates secrets, starts PostgreSQL,
> migrates, seeds, and runs both the API and the playground — see
> [Quick start](#quick-start-local-development). **Node 20+ is the only hard requirement** — with
> Docker you get PostgreSQL and a sandboxed executor, without it SQLite and an unsandboxed one.
> For the fully containerised stack instead, see [DEMO.md](./DEMO.md).

> **Refactored from a LeetCode-style grading platform.** There are no problems, no expected output
> and no hidden test cases. `Problem`, `TestCase` and `Submission` were dropped in
> `20260813000000_generic_execution_engine`; `Execution` and `Artifact` replace them.

## Stack

| Concern    | Choice                                     |
| ---------- | ------------------------------------------ |
| Runtime    | Node.js 20+ (developed on 24)              |
| Framework  | Express 4                                  |
| Language   | TypeScript 5 (strict), `tsx` for dev       |
| Database   | PostgreSQL 16 via Prisma 6                 |
| Auth       | JWT (`jsonwebtoken`) + bcrypt (`bcryptjs`) |
| Validation | Zod                                        |
| Execution  | Judge0 CE (async + webhook), local Docker fallback |
| Hardening  | helmet, cors, morgan                       |

## Execution model

Runs are **asynchronous**. Nothing about a workload's duration sits on the request path.

```
POST /api/v1/executions            -> 202 { execution: { id, status: PENDING } }
   |
   |  Judge0: POST /submissions?wait=false&callback_url=...   -> token
   |  Docker: runs inline in a throwaway container
   v
PUT/POST /api/v1/executions/callback   <- Judge0 pushes the finished submission
   |  (reconciler polls as a fallback if the callback is lost)
   v
GET /api/v1/executions/:id         -> COMPLETED | FAILED + logs + artifacts
```

`status` describes the **engine**, not the program. A script that exits 1, times out or fails to
compile is `COMPLETED` — the engine did its job. `FAILED` means no verdict could be produced.

### Endpoints

| Method              | Path                                        | Auth   | Purpose                          |
| ------------------- | ------------------------------------------- | ------ | -------------------------------- |
| `GET`               | `/api/v1/executions/languages`              | no     | Judge0 language ids              |
| `GET`               | `/api/v1/executions/engine`                 | no     | active backend + health          |
| `POST`              | `/api/v1/executions`                        | bearer | queue a run → **202**            |
| `GET`               | `/api/v1/executions`                        | bearer | list runs                        |
| `GET`               | `/api/v1/executions/:id`                    | bearer | run detail + artifacts           |
| `DELETE`            | `/api/v1/executions/:id`                    | bearer | delete a run                     |
| `GET`               | `/api/v1/executions/:id/artifacts/:aid`     | bearer | download an artifact             |
| `POST`/`PUT`        | `/api/v1/executions/callback`               | secret | Judge0 webhook                   |

Request body for a run — only `code` and `languageId` are required:

```jsonc
{
  "code": "print('hi')",
  "languageId": 71,            // Judge0 language id
  "name": "sweep run 3",
  "stdin": "...",
  "additionalFiles": "<base64 zip>",  // unpacked into the working directory
  "timeLimit": 300,            // seconds of CPU, <= JUDGE0_CPU_TIME_LIMIT_MAX
  "memoryLimit": 2048000       // KB,           <= JUDGE0_MEMORY_LIMIT_MAX
}
```

### Artifacts

A run has no filesystem you can reach after it exits, so scripts hand files back through stdout:

```
::artifact:<name>:<mimeType>:<base64>::
::artifact:<name>:<base64>::            # mime inferred from the extension
```

One marker per line. Matching lines are lifted out, stored as `Artifact` rows and **stripped from
the stdout you see**, so logs stay readable. Bounded by `ARTIFACT_MAX_COUNT` and
`ARTIFACT_MAX_TOTAL_BYTES`; anything dropped is reported on stderr as `[artifacts] ...`.

```python
import base64, json
payload = base64.b64encode(json.dumps({"ok": True}).encode()).decode()
print(f"::artifact:result.json:application/json:{payload}::")
```

The playground previews images and text inline and offers everything as a download.

### Limits

Defaults are research-scale (120 s CPU, 1 GB), not competitive-programming-scale.

Judge0 enforces its **own** ceilings and returns `422` if a request exceeds them, so
`deploy/judge0.conf` must stay `>=` the API's `JUDGE0_*_LIMIT_MAX`. Both are raised together; if you
raise one, raise the other.

> **Deploying?** Jump to [Production deployment](#production-deployment) — one command brings up
> the API, frontend, PostgreSQL and Judge0.

## Quick start (local development)

Two commands from a fresh clone. **Node 20+ is the only hard requirement.**

```bash
npm install
npm run dev
```

Then open **<http://localhost:3000/playground>** and sign in as `coder@example.com` /
`Password123!`.

`npm run dev` adapts to what the machine has:

| | Database | Code execution | Sandbox |
| --- | --- | --- | --- |
| **Docker present** | PostgreSQL in a container | throwaway container per run | yes |
| **No Docker** | SQLite file (`prisma/dev.db`) | child process of the API | **no** |

Both modes are fully functional — same API, same playground, same artifacts. Without Docker you
get whatever interpreters are already on your PATH (Python, Node, and a C/C++ compiler if you have
one), and **no isolation**: submitted code runs with your privileges. That is fine when you are the
only person writing the code, which is the point of the mode; it is not safe to expose. The API
binds to `127.0.0.1` and refuses to start on a public interface while it is active.

Install Docker and re-run to switch to the sandboxed backend — nothing else changes.

`npm run dev` is idempotent and does everything the first run needs:

1. detects whether a Docker daemon is reachable, and picks the mode above
2. generates `.env` with fresh random secrets, if missing
3. starts PostgreSQL and waits for it to be healthy (Docker mode only)
4. runs `prisma generate`, applies the schema, seeds the two demo accounts
5. installs the playground's dependencies, if missing
6. runs the API and the playground together, prefixed `[api]` / `[web]`, until `Ctrl+C`

On later runs it skips whatever is already done, so it is also just "start the app".

Judge0 is the third backend — the one this engine is really built around — but it needs a cgroup v1
host. See [Execution backends](#execution-backends).

### Running the pieces separately

`npm run dev` is a convenience, not a requirement. Nothing depends on it:

```bash
npm run dev:api      # API only  — tsx watch, port 4000
npm run dev:web      # playground only — next dev, port 3000
```

Both read the same `.env`. If you would rather wire the database up by hand, point `DATABASE_URL`
at any reachable PostgreSQL and run `npm run prisma:deploy && npm run seed`.
`prisma/migrations/0_init/migration.sql` is checked in, so `prisma migrate deploy` works against a
fresh database without generating a migration first.

### Scripts

| Script                   | Purpose                                       |
| ------------------------ | --------------------------------------------- |
| `npm run dev`            | **Everything**: Postgres, migrations, seed, API + playground |
| `npm run dev:api`        | API only, watch mode (port 4000)              |
| `npm run dev:web`        | Playground only (port 3000)                   |
| `npm run e2e`            | End-to-end suite against a running stack      |
| `npm run build`          | Compile TypeScript to `dist/`                 |
| `npm start`              | Run the compiled server                       |
| `npm run typecheck`      | Type-check without emitting                   |
| `npm run prisma:migrate` | `prisma migrate dev`                          |
| `npm run prisma:deploy`  | `prisma migrate deploy` (production)          |
| `npm run prisma:studio`  | Browse data in Prisma Studio                  |
| `npm run prisma:reset`   | Drop, re-migrate and re-seed                  |
| `npm run seed`           | Run the seed script                           |

## Environment

Validated at boot in `src/config/env.ts` — the process exits with a readable message if anything
is missing or malformed.

| Variable             | Required | Default       | Notes                              |
| -------------------- | -------- | ------------- | ---------------------------------- |
| `DATABASE_URL`       | yes      | —             | PostgreSQL connection string       |
| `JWT_SECRET`         | yes      | —             | Must be ≥ 32 characters            |
| `PORT`               | no       | `4000`        |                                    |
| `NODE_ENV`           | no       | `development` |                                    |
| `JWT_EXPIRES_IN`     | no       | `7d`          |                                    |
| `BCRYPT_SALT_ROUNDS` | no       | `10`          |                                    |
| `CORS_ORIGIN`        | no       | `*`           | Comma-separated list, or `*`       |
| `EXECUTOR`               | no | `auto`    | `auto` \| `docker` \| `judge0`     |
| `JUDGE0_API_URL`         | no | *(unset)* | e.g. `http://localhost:2358`. Unset ⇒ `auto` picks `docker` |
| `JUDGE0_API_KEY`         | no | —         | RapidAPI key, if hosted there      |
| `JUDGE0_API_HOST`        | no | —         | Sent as `X-RapidAPI-Host`          |
| `JUDGE0_AUTH_TOKEN`      | no | —         | Sent as `X-Auth-Token` (self-hosted) |
| `JUDGE0_CPU_TIME_LIMIT`  | no | `120`     | Default seconds of CPU time per run |
| `JUDGE0_MEMORY_LIMIT`    | no | `1024000` | Default KB per run                 |
| `JUDGE0_WALL_TIME_MARGIN`| no | `60`      | Wall-clock headroom above the CPU limit |
| `JUDGE0_CALLBACK_URL`    | no | *(unset)* | Webhook Judge0 posts to. Empty ⇒ polling only |
| `JUDGE0_CALLBACK_SECRET` | no | —         | Guards the callback route; appended to the URL automatically |
| `JUDGE0_POLL_INTERVAL_MS`| no | `2000`    | Reconciler sweep interval          |
| `JUDGE0_MAX_WAIT_MS`     | no | `1900000` | Declare a stuck run `FAILED` after this |
| `ARTIFACT_MAX_TOTAL_BYTES`| no| `8388608` | Total decoded artifact bytes kept per run |
| `ARTIFACT_MAX_COUNT`     | no | `20`      | Artifacts kept per run             |

Per-run `timeLimit`/`memoryLimit` override the defaults, bounded by `JUDGE0_CPU_TIME_LIMIT_MAX`
(`900`) and `JUDGE0_MEMORY_LIMIT_MAX` (`4096000`). Abuse controls (`RATE_LIMIT_*`,
`MAX_CONCURRENT_RUNS_*`) and the `EXECUTOR_*` tuning knobs are documented in `.env.example`.

## Data model

```
User 1──n Execution 1──n Artifact
```

- **User** — `id`, `email` (unique), `username` (unique), `passwordHash`, `role` (`USER`/`ADMIN`), `createdAt`
- **Execution** — one run of one script. `code`, `languageId` (Judge0's numeric id), `name?`,
  `stdin?`, `additionalFiles?` (base64 zip), `timeLimit?`/`memoryLimit?`, plus the result:
  `stdout`, `stderr`, `compileOutput`, `judgeStatus`, `exitCode`, `timeMs`, `memoryKb`,
  `errorMessage`, `judge0Token?` (unique), and `createdAt`/`updatedAt`/`startedAt`/`completedAt`
- **Artifact** — a file the run produced. `name` (sanitised to a basename), `mimeType`, `content`
  (base64), `sizeBytes`

`ExecutionStatus` is `PENDING` → `PROCESSING` → `COMPLETED` | `FAILED`. `COMPLETED` describes the
run reaching a conclusion — a non-zero exit, a compile error and a timeout all qualify. `FAILED`
means the engine could not produce a result at all.

Tables are snake_case via `@@map`; deleting a user cascades to their executions, and deleting an
execution cascades to its artifacts.

## API

Base path `/api`. All responses are JSON; errors use `{ "error": { "message", "details?" } }`.

### Auth

| Method | Route                | Auth   | Body                              |
| ------ | -------------------- | ------ | --------------------------------- |
| `POST` | `/api/auth/register` | public | `{ email, username, password }`   |
| `POST` | `/api/auth/login`    | public | `{ identifier, password }`        |
| `GET`  | `/api/auth/me`       | Bearer | —                                 |

`identifier` accepts either the email or the username. Register and login both return
`{ user, token }`; `passwordHash` is never serialized.

Rules: password 8–72 chars, username 3–30 chars matching `^[a-zA-Z0-9_]+$`, email normalized to
lowercase.

### Engine metadata

Public, so the playground can render its language picker before anyone signs in.

| Method | Route                          | Auth   | Notes                                     |
| ------ | ------------------------------ | ------ | ----------------------------------------- |
| `GET`  | `/api/v1/executions/languages` | public | Judge0 language ids with display names    |
| `GET`  | `/api/v1/executions/engine`    | public | Active backend, endpoint, health          |

### Executions

All routes below require a Bearer token, except the callback (see [Webhook](#webhook)).

| Method   | Route                                        | Notes                                            |
| -------- | -------------------------------------------- | ------------------------------------------------ |
| `POST`   | `/api/v1/executions` <br> `/api/v1/executions/run` | Queues a run → **202** with the record. Aliases of one handler |
| `GET`    | `/api/v1/executions`                         | The caller's runs, `?page=1&pageSize=20`         |
| `GET`    | `/api/v1/executions/:id`                     | Full detail incl. stdout/stderr/artifacts        |
| `DELETE` | `/api/v1/executions/:id`                     | **204**; cascades to artifacts                   |
| `GET`    | `/api/v1/executions/:id/artifacts/:artifactId` | Streams the file with its declared MIME type   |

Create body — only `code` and `languageId` are required:

```jsonc
{
  "code": "print('hi')",
  "languageId": 71,            // Judge0 id; any id the backend supports
  "name": "optional label",
  "stdin": "fed to the program",
  "additionalFiles": "<base64 zip>",  // unpacked into the working directory
  "timeLimit": 120,            // seconds of CPU time
  "memoryLimit": 1024000       // kilobytes
}
```

Execution is **asynchronous**: the call returns as soon as the row exists, in `PENDING` or
`PROCESSING`. Poll `GET /:id` until `status` is `COMPLETED` or `FAILED`.

`COMPLETED` describes the *run*, not the program's success — a non-zero exit, a timeout and a
compile error are all completed runs. `FAILED` means the engine itself could not produce a result.

Runs are scoped to their owner: another user's execution or artifact returns `404`, not `403`.

`GET /health` returns `{ status, uptime, engine: { kind, endpoint, usesCallback } }` and needs no
auth.

Everything is also mounted unversioned at `/api/...` as a compatibility alias.

## Code execution (Judge0)

Two interchangeable backends implement the same contract, so the service layer, the API and the UI
do not know which one is configured. `EXECUTOR` selects it:

| `EXECUTOR` | Backend                                                                    |
| ---------- | -------------------------------------------------------------------------- |
| `judge0`   | Judge0 — the intended engine. Requires `JUDGE0_API_URL` and a cgroup v1 host |
| `docker`   | Local throwaway containers via the mounted host socket                       |
| `auto`     | `judge0` when `JUDGE0_API_URL` is set, otherwise `docker` (default)          |

**Judge0 backend.** Submissions are always created with `wait=false`, so no research workload sits
on an HTTP request. Everything crosses the wire base64-encoded so arbitrary bytes in source, stdin
and output survive intact. Results arrive two ways:

1. **Webhook** — when `JUDGE0_CALLBACK_URL` is set, Judge0 `PUT`s the finished submission to it.
   The fast path.
2. **Polling** — a reconciler sweeps `PROCESSING` rows holding a token, catching anything the
   webhook missed. `JUDGE0_MAX_WAIT_MS` is the point at which a run is declared `FAILED`.

`finalize` is idempotent and conditioned on the row still being non-terminal, so the webhook and
the reconciler can race harmlessly — whichever arrives second is a no-op.

**Docker backend.** Runs each submission in a throwaway container: no network, all capabilities
dropped, `no-new-privileges`, read-only root with a size-capped exec tmpfs, memory and swap capped
to the same value (so exceeding the cap OOM-kills instead of thrashing into a bogus timeout), CPU
quota, PID cap, unprivileged uid. Compiled languages build into a scratch volume first, under a
separate and larger memory budget — `g++` on `<bits/stdc++.h>` needs far more than a run is
allowed. It settles inline, so there is no token and no webhook.

It covers a deliberate subset of languages (`src/services/execution/languages.ts`): Python 3 (71),
Python 2 (70), Node.js (63), C (50), C++ (54). Anything else fails with an explicit message rather
than a silently wrong verdict. The point is to keep the engine usable on a cgroup v2 host, not to
reimplement Judge0's 60 languages.

**Language ids are Judge0's**, passed through as an integer, so any language the configured Judge0
instance supports works without an API change. `GET /api/v1/executions/languages` returns the
labelled set.

### Webhook

`POST` and `PUT /api/v1/executions/callback` receive finished Judge0 submissions. Judge0 has no
user session, so the route sits outside `requireAuth` and is guarded by `JUDGE0_CALLBACK_SECRET`
instead.

Judge0 can only fetch back the URL it was given — it cannot be told to send a custom header — so
the secret travels in the query string. The backend appends it to `JUDGE0_CALLBACK_URL`
automatically; you do not put it there yourself, and a secret already present is left alone. The
route accepts either `?secret=` or an `x-callback-secret` header.

An unknown token answers `200` rather than `404`: it means the row was deleted or belongs to
another deployment, and Judge0 should stop retrying either way. Non-terminal statuses are ignored
rather than written as half-finished results.

### Running the bundled Judge0

The Judge0 services ship in this repo's compose file behind an opt-in `judge0` profile, so a plain
`docker compose up -d` does **not** start them. To use them, set `JUDGE0_API_URL` in `.env.docker`
and bring the profile up:

```bash
# .env.docker
JUDGE0_API_URL=http://judge0-server:2358
EXECUTOR=judge0

docker compose --env-file .env.docker --profile judge0 up -d
```

Read [Judge0 requires cgroup v1](#judge0-requires-cgroup-v1) first — on a cgroup v2 host every
submission comes back `Internal Error`.

For a hosted instance instead, set `JUDGE0_API_URL` plus `JUDGE0_API_KEY` and `JUDGE0_API_HOST`
(RapidAPI), or `JUDGE0_AUTH_TOKEN` (self-hosted with `JUDGE0_AUTHN_TOKEN` set).

### Example

```bash
curl -X POST http://localhost:4000/api/v1/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"me@example.com","username":"me","password":"Password123!"}'

TOKEN=$(curl -s -X POST http://localhost:4000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"identifier":"me@example.com","password":"Password123!"}' | jq -r .token)

# Queue a run — returns 202 immediately.
ID=$(curl -s -X POST http://localhost:4000/api/v1/executions/run \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"code":"print(\"hello\")","languageId":71}' | jq -r .execution.id)

# Poll until it settles.
curl -s "http://localhost:4000/api/v1/executions/$ID" \
  -H "Authorization: Bearer $TOKEN" | jq '.execution | {status, judgeStatus, stdout, timeMs}'
```

`scripts/e2e.sh` does exactly this for Python, C++ and Node.js and asserts on the results — see
[Testing](#testing).

## Seed data

`npm run seed` is idempotent (upserts by unique key). There is nothing content-like to seed any
more — no problems, no test cases — so it only guarantees two accounts exist, which makes the
playground usable straight after `migrate deploy`:

| Email                 | Username | Password       | Role  |
| --------------------- | -------- | -------------- | ----- |
| `admin@example.com`   | `admin`  | `Admin123!`    | ADMIN |
| `coder@example.com`   | `coder`  | `Password123!` | USER  |

## Testing

`scripts/e2e.sh` drives the public HTTP surface exactly as a client would — authenticate, queue
runs through `POST /executions/run`, poll to a terminal state — then asserts on stdout, exit code,
measured timing, artifact extraction and download, and confirms the rows landed in Postgres.

```bash
docker compose --env-file .env.docker up -d
bash scripts/e2e.sh                      # or: bash scripts/e2e.sh http://host:4000
```

It covers Python (with stdin, stderr and an emitted artifact), C++ (exercising the compile step)
and Node.js, and exits non-zero on the first failed assertion so CI can gate on it.

## Project layout

```
prisma/
  schema.prisma              # data model
  seed.ts                    # seed script
  migrations/0_init/         # checked-in initial migration
prisma.config.ts             # schema path + seed command (replaces package.json#prisma)
src/
  app.ts                     # express app assembly
  server.ts                  # listen + graceful shutdown
  config/env.ts              # zod-validated environment
  lib/prisma.ts              # shared PrismaClient
  middleware/
    auth.ts                  # requireAuth / optionalAuth / requireRole / requireAdmin
    errorHandler.ts          # ApiError, Zod and Prisma error mapping + 404
    validate.ts              # validateBody / validateQuery
  services/execution/
    index.ts                 # picks the backend, re-exports the contract
    types.ts                 # ExecutionSpec / ExecutionOutcome / ExecutionBackend
    judge0Backend.ts         # Judge0 client: submit, poll, status mapping, callback URL
    dockerBackend.ts         # throwaway-container backend (cgroup v2 hosts)
    languages.ts             # Judge0 language ids + the Docker-runnable subset
    artifacts.ts             # ::artifact:…:: stdout marker parser
  modules/
    auth/                    # schemas, service, controller, routes
    executions/              # routes, service, reconciler
  routes/index.ts            # mounts module routers under /api/v1 (and /api)
  utils/                     # ApiError, asyncHandler, jwt helpers
  types/express.d.ts         # req.user augmentation
web/                         # Next.js frontend (own package.json + README)
```

## Production deployment

```bash
cp .env.docker.example .env.docker   # fill in the secrets it lists
./deploy.sh
```

`deploy.sh` renders the Judge0 config, builds the images, starts PostgreSQL, applies migrations,
seeds the accounts, brings up Judge0, then starts the API and frontend.

| Flag             | Effect                                                        |
| ---------------- | ------------------------------------------------------------- |
| *(none)*         | Build if needed, migrate, seed, start everything              |
| `--build`        | Force a no-cache rebuild of the `api` and `web` images        |
| `--no-seed`      | Apply migrations but skip seeding                             |
| `--skip-judge0`  | App stack only; the API falls back to mock grading            |
| `--down`         | Stop the stack (volumes preserved)                            |
| `--logs`         | Follow logs for every service                                 |

It refuses to run with an unset/short `JWT_SECRET`, a missing `POSTGRES_PASSWORD`, a `DATABASE_URL`
still containing `CHANGE_ME`, or (unless `--skip-judge0`) missing Judge0 datastore passwords, and
warns when `CORS_ORIGIN` is still `*`.

### Services

| Service          | Image / build            | Exposed                  | Profile   |
| ---------------- | ------------------------ | ------------------------ | --------- |
| `web`            | `web/Dockerfile`         | `:3000`                  | default   |
| `api`            | `Dockerfile` (`runtime`) | `:4000`                  | default   |
| `migrate`        | `Dockerfile` (`migrator`)| one-shot, exits          | default   |
| `postgres`       | `postgres:16-alpine`     | `127.0.0.1:5435`         | default   |
| `judge0-server`  | `judge0/judge0:1.13.1`   | `127.0.0.1:2358`         | `judge0`  |
| `judge0-workers` | `judge0/judge0:1.13.1`   | internal                 | `judge0`  |
| `judge0-db`      | `postgres:16-alpine`     | internal                 | `judge0`  |
| `judge0-redis`   | `redis:7-alpine`         | internal                 | `judge0`  |

Judge0 gets its own PostgreSQL and Redis, matching its standard deployment — the app database and
the judge's queue stay in separate failure domains. Databases and Judge0 are bound to loopback;
only `web` and `api` are meant to be public (put a TLS-terminating reverse proxy in front).

#### Execution backends

`EXECUTOR` selects which backend runs the code:

| Value    | Behaviour                                                                       |
| -------- | ------------------------------------------------------------------------------- |
| `docker` | One throwaway container per run. Works on cgroup v2. Subset of languages.        |
| `judge0` | Use a Judge0 server. Needs `JUDGE0_API_URL` *and* a cgroup v1 host.              |
| `native` | Child process of the API. **No sandbox**, no Docker. Uses toolchains on PATH.    |
| `auto`   | Default — `judge0` if `JUDGE0_API_URL` is set, else `docker` if its daemon answers, else `native`. |

`native` exists so the engine works with no infrastructure at all, which is what makes
`npm install && npm run dev` sufficient on a machine without Docker. It is the only backend with no
isolation: submitted code runs with the API process's privileges, and the wall-clock timeout and
output cap bound accidents, not attackers. It also cannot enforce a memory limit (no cgroup, so
`memoryKb` is `null`), reports wall-clock rather than CPU time, and rejects `additionalFiles`
instead of silently ignoring a dataset it cannot unpack. The API refuses to bind to a non-loopback
`HOST` while it is active unless `EXECUTOR_NATIVE_ALLOW_REMOTE=true` is set explicitly.

`EXECUTOR=docker` is the default in `.env.docker` because Judge0 cannot run here (below). It mounts
`/var/run/docker.sock` into `api` and spawns siblings against the host daemon. Each run gets:

```
--network none          --cap-drop ALL       --security-opt no-new-privileges
--memory/--memory-swap  --cpus               --pids-limit
--read-only + tmpfs     --user 65534:65534   timeout -s KILL
```

CPU time comes from the cgroup (`cpu.stat`), peak memory from `memory.peak`, and an OOM kill is
reported as *memory limit exceeded* rather than a timeout. Compiled languages are built **once per
submission** into a scratch volume with a separate, larger budget
(`EXECUTOR_COMPILE_MEMORY_MB`) — g++ on `<bits/stdc++.h>` needs far more memory than a submission is
allowed at run time, and sharing one limit makes every C++ build die with `cc1plus: Killed`.

The time limit is **wall-clock**, so keep `EXECUTOR_CONCURRENCY * EXECUTOR_CPUS` at or below the
cores you can spare — oversubscribing makes correct solutions time out under load.

> **Security.** The docker socket is root-equivalent on the host. This is fine for local
> development. Before exposing the platform to untrusted users, put a socket proxy in front of it or
> move execution to a disposable VM.

#### Why Judge0 is opt-in

Judge0 1.13.x requires **cgroup v1**, and on a cgroup v2 host — Docker Desktop / WSL2 and every
modern Linux distro — `isolate` cannot create its jails:

```
Failed to create control group /sys/fs/cgroup/memory/box-0/: No such file or directory
```

The stack would come up healthy and then fail *every* submission with `Internal Error`. Judge0
1.13.1 (April 2024) is the newest image published, and it is still cgroup v1 only, so there is
nothing to upgrade to — hence the local Docker executor above.

To run the real judge on a cgroup v1 host, set `JUDGE0_API_URL=http://judge0-server:2358` and
`EXECUTOR=judge0` in `.env.docker`, then enable the profile:

```bash
docker compose --env-file .env.docker --profile judge0 up -d
```

Check which backend is live at any time:

```bash
curl http://localhost:4000/health
# {"status":"ok","uptime":18.4,
#  "engine":{"kind":"docker","usesCallback":false,"endpoint":"local-docker","callbackUrl":"…"}}
```

`api` starts only after `postgres` is healthy *and* `migrate` has exited successfully
(`service_completed_successfully`), so it never serves traffic against an unmigrated schema.

### Images

Both Dockerfiles are multi-stage and run as the non-root `node` user with a `HEALTHCHECK`.

- **API** — `deps` → `build` (tsc + `prisma generate`) → `runtime`, which takes `node_modules` from a
  separate `prod-deps` stage and copies only the generated Prisma client across from `build`. The
  Prisma CLI and `tsx` stay in the `migrator` stage, out of the runtime image.
- **Web** — `deps` → `build` → `runtime` running Next's `output: 'standalone'` server (~21 MB of
  traced files instead of the full `node_modules`).

### Two configuration gotchas

1. **`NEXT_PUBLIC_API_URL` is baked in at build time**, so it must be the URL the *browser* uses
   (e.g. `https://api.example.com`), and changing it needs `./deploy.sh --build`.
2. **Server-side rendering uses `API_INTERNAL_URL`** (`http://api:4000`) because the browser-facing
   URL isn't resolvable inside the network. Both are set in `.env.docker`.

### Running real Judge0 on Windows (WSL2)

Docker Desktop's WSL2 backend mounts cgroup **v2**, so Judge0 cannot execute anything there (see
below). A second WSL distro configured for cgroup v1 can, and `scripts/` contains everything
needed.

The obstacle is that WSL2's init mounts `/sys/fs/cgroup` as cgroup2 **before** PID 1 starts, and
systemd — which is what would normally honour `systemd.unified_cgroup_hierarchy=0` — inherits that
mount rather than replacing it. Setting the kernel command line alone therefore does nothing. What
does work is disabling systemd so nothing claims the controllers at boot, then remounting them as
v1 before dockerd starts. A controller can only live in one hierarchy at a time, so the ordering is
the whole trick.

In an Ubuntu WSL distro (`wsl --install -d Ubuntu` if you don't have one):

```bash
# 1. Docker Engine, natively in the distro — not Docker Desktop integration.
curl -fsSL https://get.docker.com | sh

# 2. cgroup v1 at boot, then dockerd.
sudo install -m 0755 scripts/wsl-cgroup-v1.sh /usr/local/sbin/
sudo install -m 0755 scripts/wsl-boot.sh      /usr/local/sbin/
sudo cp scripts/wsl.conf          /etc/wsl.conf        # sets systemd=false + the boot hook
sudo cp scripts/docker-daemon.json /etc/docker/daemon.json
```

Then from Windows, `wsl --shutdown`, reopen the distro, and confirm:

```bash
stat -fc %T /sys/fs/cgroup     # tmpfs, with memory/ cpuacct/ cpuset/ beneath it
docker info | grep -i cgroup   # Cgroup Version: 1
```

Now bring the stack up **inside that distro**, with `.env.judge0` (identical to `.env.docker` but
`EXECUTOR=judge0` and `JUDGE0_API_URL=http://judge0-server:2358`):

```bash
docker compose --env-file .env.judge0 --profile judge0 up -d
ENV_FILE=.env.judge0 bash scripts/e2e.sh
```

Everything runs on one Docker network, so the webhook reaches `http://api:4000/...` exactly as
designed. Stop the Docker Desktop stack first or the two will fight over ports 3000/4000.

Two things worth knowing about that setup:

- **`docker-daemon.json` pins `userland-proxy-path`.** WSL runs the `[boot]` command with a minimal
  `PATH`, and without it dockerd exits with `invalid userland-proxy-path`.
- **`wsl-boot.sh` starts dockerd under `setsid`.** WSL tears down the boot command's process group
  once it returns, so a plain `dockerd &` is reaped seconds after it finishes initialising — the log
  reads "Daemon has completed initialization" and then the socket is simply gone.

To revert entirely: restore `systemd=true` in `/etc/wsl.conf`, remove the `command=` line, and
`wsl --shutdown`.

### Judge0 requires cgroup v1

Judge0 v1.13.x uses `isolate`, which needs **cgroup v1**. On a host running cgroup v2 (most modern
distros) submissions fail with internal errors. Add to the kernel cmdline and reboot:

```
systemd.unified_cgroup_hierarchy=0 systemd.legacy_systemd_cgroup_controller
```

`deploy.sh` detects cgroup v2 and warns. The workers also run `privileged: true`, which `isolate`
needs to build its sandboxes — so run this stack on a host you control, not a shared runner.

Tune `JUDGE0_WORKER_COUNT` to at most the host's core count; each worker grades one submission at a
time.

## Notes and known gaps

- **Artifacts are stored inline** as base64 in the row. Research outputs are small and this avoids
  needing object storage to read a result back, but it puts a hard ceiling
  (`ARTIFACT_MAX_TOTAL_BYTES`) on what a run can return. Anything larger needs S3-style storage and
  a signed URL.
- **Artifacts travel through stdout**, so a program that prints a line looking like the marker has
  it lifted out as a file. The marker is deliberately unusual, but it is not escaped.
- **The Docker backend needs `/var/run/docker.sock`**, which is root-equivalent on the host. Fine
  for local development; front it with a socket proxy, or move execution to a disposable VM, before
  exposing this to untrusted users.
- **Judge0 cannot run on a cgroup v2 host** — see below. That is why the Docker backend exists.
- Output is capped at `EXECUTOR_MAX_OUTPUT_BYTES` and silently truncated past it; the record does
  not say it was truncated.
- Tokens are stateless with no refresh or revocation; logout is a client-side token discard.
- Login compares against a dummy bcrypt hash when no user matches, so response timing does not
  reveal whether an account exists.
- `scripts/e2e.sh` covers the engine end-to-end, but there are no unit tests — the artifact parser
  and the Judge0 status mapping are pure functions and deserve them.

## Deployment verification

What was actually checked, on a Windows 11 / Docker Desktop (WSL2, cgroup v2) host.

**Verified end-to-end**

- All seven services build and start: `api`, `web`, `postgres`, `judge0-server`, `judge0-workers`,
  `judge0-db`, `judge0-redis`.
- `migrate` runs `prisma migrate deploy && prisma db seed` inside a container; the API waits on it.
- `scripts/e2e.sh` green on the Docker backend — Python, C++ and Node.js all reach `COMPLETED` with
  exit code 0, correct stdout, measured timing, and artifacts extracted, stored and downloadable
  with the right MIME type. Rows confirmed in Postgres.
- The Judge0 backend's full submit path: source submitted with `wait=false`, token persisted, row
  moved to `PROCESSING`, and Judge0's finished submission delivered back to
  `PUT /api/v1/executions/callback?secret=…` → `200 {"ok":true,"applied":true}`.
- Callback handling against a synthetic Judge0 payload: base64 stdout decoded, `time` seconds →
  `timeMs`, memory mapped, marker lines stripped from stdout, and two artifacts extracted — one
  with a declared MIME type, one inferred from its extension. A callback without the secret is
  rejected `401`.
- Judge0 1.13.1 runs fine against `postgres:16-alpine`; its migrations apply cleanly.

**Known not to work on this host**

- **Judge0 cannot execute code here.** `isolate` wants the cgroup v1 layout
  (`/sys/fs/cgroup/memory/box-0/`) and WSL2 provides `cgroup2fs`, so every submission returns
  `Internal Error` — `No such file or directory @ rb_sysopen - /box/script.py`. The integration is
  correct and proven up to the sandbox; the sandbox itself needs a cgroup v1 host. This is exactly
  why `EXECUTOR=docker` is the default here.

### Troubleshooting: `judge0-server` crashlooping on `password authentication failed`

The `judge0-db-data` volume outlives `.env.docker`. Postgres only runs its initialisation — the
step that creates `POSTGRES_USER` — on an **empty** data directory, so a volume first created under
different credentials keeps the old ones forever and Judge0 cannot log in.

Either recreate the volume (it holds only Judge0's own submission queue, nothing you authored):

```bash
docker compose --profile judge0 down
docker volume rm hackerrank-clone_judge0-db-data
```

or repair it in place, which preserves the existing schema:

```bash
PW=$(grep -E '^POSTGRES_PASSWORD=' deploy/judge0.conf | cut -d= -f2)
docker compose --env-file .env.docker --profile judge0 exec -T judge0-db \
  psql -U postgres -c "CREATE ROLE judge0 LOGIN SUPERUSER PASSWORD '$PW';"
docker compose --env-file .env.docker --profile judge0 exec -T judge0-db \
  psql -U postgres -c "ALTER DATABASE judge0 OWNER TO judge0;"
docker compose --env-file .env.docker --profile judge0 restart judge0-server judge0-workers
```

## Contributing

Issues and pull requests are welcome — see [CONTRIBUTING.md](./CONTRIBUTING.md) for setup, the
checks CI runs, and the conventions this codebase follows.

Security issues should **not** go in the public tracker. See [SECURITY.md](./SECURITY.md); this
project executes untrusted code by design, and that file explains what the sandbox does and does
not protect.

## License

[MIT](./LICENSE) © Arnav Chauhan

Judge0 is a separate project, licensed under the GPLv3, and is used here as an unmodified upstream
container image rather than being vendored or linked into this codebase.

### Troubleshooting: a second checkout cannot reach its database

`docker-compose.yml` sets `name: hackerrank-clone`, so **every clone on the machine shares one
Compose project** — and therefore one Postgres volume — unless told otherwise. Postgres applies
`POSTGRES_PASSWORD` only when it initialises an *empty* data directory, so a second clone generates
its own secrets, inherits the first clone's volume, and can never authenticate against it:

```
Error: P1000: Authentication failed against database server
```

`setup` now writes a per-directory `COMPOSE_PROJECT_NAME` into `.env` to prevent this. Checkouts
created before that was added do not have it; add it by hand:

```bash
echo "COMPOSE_PROJECT_NAME=$(basename "$PWD")" >> .env
```

Two further consequences worth knowing:

- **Only one checkout can run at a time.** They all publish ports 3000, 4000 and 5435. Stop the
  other one with `docker compose down` from its directory, or set a different `POSTGRES_PORT`.
- **A failed start can leave a container without its port bindings.** If the first `up` fails
  because a port was taken, Compose keeps the container; the next `up` finds it healthy and reuses
  it as-is, so it reports success while nothing listens on the host and Prisma fails with `P1001`.
  `npm run dev` now detects this and recreates the container, but by hand it is:

  ```bash
  docker compose up -d --force-recreate postgres
  ```
