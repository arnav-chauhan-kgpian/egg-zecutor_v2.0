# Security Policy

This project **executes untrusted code by design**. That makes its threat model unusual, and worth
reading before you deploy it anywhere other than your own machine.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem. Use GitHub's
[private vulnerability reporting](https://github.com/arnav-chauhan-kgpian/egg-zecutor/security/advisories/new)
instead. Include a description, reproduction steps, and the impact you think it has.

This is a personal project with no SLA. Expect a first response within a week or so.

## What the sandbox does and does not protect

Isolation is provided by whichever backend is configured, not by this codebase:

| Backend  | Isolation                                                                  |
| -------- | -------------------------------------------------------------------------- |
| `judge0` | [`isolate`](https://github.com/ioi/isolate) — cgroup + namespace sandboxing |
| `docker` | One throwaway container per run, with cpu/memory/pids limits                |
| `native` | **None.** Code runs as a child process of the API                          |

### `EXECUTOR=native` has no sandbox

This backend exists so the project runs on a machine with no container runtime — `npm install &&
npm run dev` and nothing else. The trade is explicit: submitted code executes with the full
privileges of the API process and can read and write your files, open sockets, and spawn
processes. The wall-clock timeout and captured-output cap stop *accidents*, not attackers.

It is a reasonable default only when the person writing the code and the person running the server
are the same person. Two guards enforce that:

- the API refuses to start when `HOST` is not loopback while this backend is active, unless
  `EXECUTOR_NATIVE_ALLOW_REMOTE=true` is set deliberately
- a warning is printed on every startup

`auto` only selects it when no Judge0 is configured *and* no Docker daemon answers, so enabling it
by accident on a host that has a sandbox available is not possible. For anything multi-user or
internet-facing, use `docker` or `judge0`.

The API enforces per-run CPU time, wall time, memory, and output size, plus per-user and global
concurrency caps. Those bound *resource* abuse. They are not a substitute for the sandbox.

## Deployment cautions

These are the things most likely to bite you:

- **`judge0-workers` runs `privileged: true`.** `isolate` needs kernel privileges to build its
  jails. Run this stack on a host you control — not a shared CI runner.
- **`EXECUTOR=docker` mounts the host Docker socket into the API container.** Access to that socket
  is equivalent to root on the host. It is fine for local development; front it with a socket proxy
  or move execution to a disposable VM before exposing it to untrusted users.
- **Bind addresses default to loopback.** `BIND_ADDRESS` defaults to `127.0.0.1` deliberately. Set
  it to `0.0.0.0` only if you have thought about the two points above.
- **Judge0 runs with `ENABLE_NETWORK=false` by default.** Turning it on gives executed code network
  access from inside your network.
- **Set `JUDGE0_CALLBACK_SECRET`.** The callback route mutates execution records from an
  unauthenticated request. Without the shared secret, anyone who can reach the API can forge
  results. The engine attaches it to the callback URL automatically.

## Secrets

`setup.sh` / `setup.ps1` generate real credentials into `.env`, `.env.docker`, `.env.judge0`, and
`deploy/judge0.conf`. All four are gitignored and must never be committed. Only the `*.example` and
`*.template` files, which contain placeholders, belong in the repository.

If you believe a secret was committed, rotate it — removing the file does not remove it from git
history.
