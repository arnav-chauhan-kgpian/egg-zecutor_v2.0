# Contributing

Thanks for taking a look. Issues and pull requests are welcome.

## Getting set up

Node 20+ is the only hard prerequisite:

```bash
npm install
npm run dev
```

That brings up the API and the playground together. With Docker running it uses PostgreSQL and the
sandboxed Docker executor; without Docker it falls back to SQLite and the **unsandboxed** native
executor, which is fine for local work but never for anything exposed — see
[SECURITY.md](./SECURITY.md).

Note that a change touching the execution path should be checked against a **sandboxed** backend,
since the native one has no isolation and different limits (no memory cap, wall-clock rather than
CPU time, no `additionalFiles`).

The pieces run independently too:

```bash
npm run dev:api     # tsx watch, port 4000
npm run dev:web     # next dev, port 3000
```

For the fully containerised stack, see [DEMO.md](./DEMO.md).

## Before you open a PR

Both packages must typecheck and build. CI runs exactly this:

```bash
npx prisma generate && npm run typecheck && npm run build
cd web && npm run build
```

If you touched the execution path, run the end-to-end suite too. It drives the real HTTP surface —
auth, queue, poll to terminal, artifact download, and the Postgres rows:

```bash
docker compose --env-file .env.docker up -d
bash scripts/e2e.sh
```

It exits non-zero on the first failed assertion, and prints a pass/fail line per assertion.

## Working on Judge0 specifically

Judge0 1.13.x requires a **cgroup v1** host. On cgroup v2 (Docker Desktop, WSL2, and every modern
Linux distro) `isolate` cannot build its jails and every submission returns `Internal Error`. This
is not a bug in this project — it is why the Docker backend exists.

The README has a full runbook for getting a cgroup v1 host under WSL2, including the boot-ordering
trick that makes it work. Once you have one:

```bash
docker compose --env-file .env.judge0 --profile judge0 up -d
ENV_FILE=.env.judge0 bash scripts/e2e.sh
```

## Conventions

- **TypeScript strict**, no `any` where a real type will do.
- **Comments explain *why*, not *what*.** The existing code leans heavily on this — a comment that
  restates the line below it will get flagged in review. Comments that capture a non-obvious
  constraint, a race, or a decision you had to reason about are exactly what's wanted.
- **Never commit secrets.** `.env`, `.env.docker`, `.env.judge0`, and `deploy/judge0.conf` hold
  generated credentials and are gitignored. Edit the `*.example` / `*.template` files instead.
- Keep migrations additive where you can, and describe destructive ones in the migration file
  itself — `20260813000000_generic_execution_engine` is the model to follow.

## Reporting security issues

Please don't use the public issue tracker. See [SECURITY.md](./SECURITY.md).
