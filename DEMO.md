# Demo guide

A generic code execution engine. Write a script, run it under limits you choose, get back its
logs, stats and any files it produced.

---

## Easiest path: run it from source

```bash
git clone https://github.com/arnav-chauhan-kgpian/egg-zecutor.git
cd egg-zecutor
npm install
npm run dev
```

Open **<http://localhost:3000/playground>** and sign in as `coder@example.com` / `Password123!`.

`npm run dev` generates its own secrets, starts PostgreSQL, migrates, seeds, and runs both the API
and the playground. Requires Node 20+ and a running Docker.

---

## Before you hand this over

If you share this as a **zip or folder copy** rather than a git clone, four generated files contain
live secrets and must not travel with it:

```
.env
.env.docker
.env.judge0
deploy/judge0.conf
```

Delete those before zipping; the recipient regenerates their own on first run. They are all listed
in `.gitignore`, so cloning or pushing excludes them automatically.

---

## Fully containerised alternative

Runs everything in Docker, including the API and frontend — no Node on the host.

**Requirements:** Docker Desktop (running). Nothing else — no Node, no Python, no database.

```bash
cd hackerank-clone

./setup.sh          # macOS / Linux / Git Bash
.\setup.ps1         # Windows PowerShell

docker compose up -d
```

`setup` is safe to re-run; it leaves existing config alone. Pass `--force` / `-Force` to regenerate
with new secrets.

`setup` generates secrets into `.env.docker` (gitignored, so it is never shared) and pre-pulls the
runner images. First `up` builds the images and takes a few minutes; after that it is seconds.

Open **http://localhost:3000**.

| Account             | Password       | Role  |
| ------------------- | -------------- | ----- |
| `coder@example.com` | `Password123!` | user  |
| `admin@example.com` | `Admin123!`    | admin |

Or register your own from the header.

**Stop:** `docker compose down` — data is preserved. `docker compose down -v` wipes the database too.

---

## What to show

### 1. Just run something

The editor opens with a starter script. Press **Run**. You get stdout, CPU time, peak memory and
the exit code. Nothing blocks — the run is queued and the UI polls.

### 2. Failure modes are real, not simulated

Try each of these and watch the verdict change:

```python
raise SystemExit(7)        # Runtime Error (exit 7)
while True: pass           # Time Limit Exceeded
x = bytearray(2_000_000_000)   # Memory Limit Exceeded
```

```cpp
int main(){ this is not valid }   // Compilation Error, with real g++ output
```

Note the distinction: a script that crashes is still `COMPLETED` — the engine did its job. `FAILED`
means the *engine* broke, not your code.

### 3. Artifacts — the interesting part

A run has no filesystem you can reach after it exits, so scripts hand files back through stdout:

```python
import base64, json

data = {"experiment": "sweep-3", "accuracy": 0.91}
payload = base64.b64encode(json.dumps(data).encode()).decode()
print(f"::artifact:result.json:application/json:{payload}::")

png = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
print(f"::artifact:pixel.png:image/png:{base64.b64encode(png).decode()}::")
```

Open the **artifacts** tab: both files are there, the image renders inline, and both download. The
marker lines are stripped from stdout so logs stay clean.

### 4. Input files

Zip a dataset, attach it under **additional_files**, and read it as a normal relative path — it is
unpacked into the working directory before the run:

```python
import csv, os
print(sorted(os.listdir(".")))
print(list(csv.DictReader(open("dataset.csv"))))
```

### 5. Limits are per-run

Set **CPU time limit** and **Memory limit** in the settings pane. Defaults are 120 s and 1 GB —
research-scale, not competitive-programming-scale.

---

## API

Everything the UI does is a plain REST call. Versioned at `/api/v1`.

```bash
TOKEN=$(curl -s localhost:4000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"identifier":"coder@example.com","password":"Password123!"}' | jq -r .token)

# Queue a run -> 202 with an id
ID=$(curl -s localhost:4000/api/v1/executions \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"code":"print(6*7)","languageId":71}' | jq -r .execution.id)

# Poll it
curl -s localhost:4000/api/v1/executions/$ID -H "Authorization: Bearer $TOKEN" | jq
```

`GET /api/v1/executions/languages` lists the language ids. `GET /health` shows which backend is
live. Full endpoint table in [README.md](./README.md#endpoints).

---

## Security — read before exposing this

**Submitted code is sandboxed.** Every run gets its own container with no network, all capabilities
dropped, no privilege escalation, a read-only root filesystem, and memory/CPU/PID caps. A demo user's
script cannot reach the host or the Docker socket.

**The API container is not.** It mounts `/var/run/docker.sock` to spawn those containers, and socket
access is root-equivalent on the host. So the trust boundary is: *you trust this application's code*,
not *you trust the people using it*.

Practically:

- **Ports bind to `127.0.0.1` by default.** Only your machine can reach it. Each demo-er runs their
  own copy.
- **Do not set `BIND_ADDRESS=0.0.0.0` on a machine you care about**, and do not put this on the
  public internet as-is. If you need shared hosting, run it on a disposable VM, or switch to the
  Judge0 backend (`EXECUTOR=judge0`), which uses `isolate` and needs no socket.
- Abuse controls are on: 20 runs/min per user, 3 concurrent runs per user, 12 total.

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `POSTGRES_PASSWORD is required` | You skipped `setup.sh` / `setup.ps1`. |
| `bind: An attempt was made to access a socket…` (Windows) | Hyper-V reserved the port. `netsh int ipv4 show excludedportrange protocol=tcp`, then set a free `API_PORT`/`WEB_PORT`/`POSTGRES_PORT` in `.env.docker`. |
| First run takes ~30 s | Cold image pull. `setup` pre-pulls; `docker pull python:3.11-alpine` manually if you skipped it. |
| Run stuck on `processing` | `docker compose logs api`. A restart mid-run now fails the run automatically rather than hanging. |
| `language N is not supported` | The local Docker backend runs Python, JS, C and C++. The full ~60-language set needs `EXECUTOR=judge0`. |
| Everything returns `Internal Error` under `EXECUTOR=judge0` | Judge0 1.13.x requires cgroup v1; on Docker Desktop/WSL2 and modern Linux it cannot sandbox. Use the default backend. See README. |

Reset completely:

```bash
docker compose down -v
docker compose up -d
```
