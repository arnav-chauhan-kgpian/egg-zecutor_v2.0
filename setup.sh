#!/usr/bin/env bash
#
# One-time bootstrap. Generates the config that is deliberately NOT committed:
#
#   .env.docker         real secrets, gitignored
#   deploy/judge0.conf  rendered from the template, gitignored
#
# Safe to re-run: existing files are left alone unless --force is passed.
#
#   ./setup.sh            create anything missing
#   ./setup.sh --force    regenerate from scratch (new secrets)
#
# Windows: use setup.ps1 instead, or run this from Git Bash.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

FORCE=false
[[ "${1:-}" == "--force" ]] && FORCE=true

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m  %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH"
docker info >/dev/null 2>&1 || die "Cannot talk to the Docker daemon — is Docker Desktop running?"

# openssl ships with git-bash and every mainstream distro; fall back to Docker
# so this works even on a machine that only has Docker.
secret() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex "$1"
  else
    docker run --rm alpine sh -c "head -c $1 /dev/urandom | od -An -tx1 | tr -d ' \n'"
  fi
}

# ---------------------------------------------------------------- .env.docker
if [[ -f .env.docker && "$FORCE" == false ]]; then
  log ".env.docker already exists — leaving it alone (--force to regenerate)"
else
  [[ -f .env.docker.example ]] || die "Missing .env.docker.example"
  log "Generating .env.docker with fresh secrets"

  POSTGRES_PASSWORD="$(secret 24)"
  JWT_SECRET="$(secret 48)"
  JUDGE0_POSTGRES_PASSWORD="$(secret 24)"
  JUDGE0_REDIS_PASSWORD="$(secret 24)"
  JUDGE0_CALLBACK_SECRET="$(secret 24)"

  # Start from the example so new keys added later are picked up automatically,
  # then substitute the values that must be unique per install.
  cp .env.docker.example .env.docker

  # `|` as the delimiter: hex secrets never contain it, unlike `/`.
  sed -i.bak \
    -e "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=${POSTGRES_PASSWORD}|" \
    -e "s|^JWT_SECRET=.*|JWT_SECRET=${JWT_SECRET}|" \
    -e "s|^JUDGE0_POSTGRES_PASSWORD=.*|JUDGE0_POSTGRES_PASSWORD=${JUDGE0_POSTGRES_PASSWORD}|" \
    -e "s|^JUDGE0_REDIS_PASSWORD=.*|JUDGE0_REDIS_PASSWORD=${JUDGE0_REDIS_PASSWORD}|" \
    -e "s|^JUDGE0_CALLBACK_SECRET=.*|JUDGE0_CALLBACK_SECRET=${JUDGE0_CALLBACK_SECRET}|" \
    -e "s|^DATABASE_URL=.*|DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/hackerrank_clone?schema=public|" \
    .env.docker
  rm -f .env.docker.bak

  # Isolate this checkout's containers and volumes.
  #
  # docker-compose.yml hard-codes `name: hackerrank-clone`, so without this every
  # clone on the machine shares one Compose project — and therefore one Postgres
  # volume. Postgres only applies POSTGRES_PASSWORD when it initialises an empty
  # data directory, so the second clone generates fresh secrets, inherits the
  # first clone's volume, and can never authenticate against it (P1000).
  #
  # COMPOSE_PROJECT_NAME takes precedence over the `name:` key, and Compose reads
  # it straight out of this file. Project names allow [a-z0-9_-] only.
  COMPOSE_PROJECT_NAME="$(basename "$PWD" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_-' '-' | sed 's/^[^a-z0-9]*//; s/-*$//')"
  COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-hackerrank-clone}"

  {
    echo ""
    echo "# Scopes this checkout's containers and volumes so a second clone on the"
    echo "# same machine does not inherit this one's database."
    echo "COMPOSE_PROJECT_NAME=${COMPOSE_PROJECT_NAME}"
  } >> .env.docker

  log "Wrote .env.docker (project: ${COMPOSE_PROJECT_NAME})"
fi

# ----------------------------------------------------------------------- .env
# Compose reads `.env` by DEFAULT — not `.env.docker`. Without this file a
# plain `docker compose up -d` fails on "POSTGRES_PASSWORD is required", and
# passing --env-file every time is a footgun nobody remembers. So `.env` is
# generated from the same secrets, with DATABASE_URL pointed at the published
# host port instead of the service name, which is what the Prisma CLI needs
# when you run it outside a container.
if [[ -f .env && "$FORCE" == false ]]; then
  log ".env already exists — leaving it alone"
else
  log "Generating .env for plain \`docker compose up -d\` + host-side Prisma"

  PG_PW="$(grep -E '^POSTGRES_PASSWORD=' .env.docker | cut -d= -f2-)"
  PG_PORT="$(grep -E '^POSTGRES_PORT=' .env.docker | cut -d= -f2- || true)"
  PG_PORT="${PG_PORT:-5435}"

  {
    grep -vE '^(DATABASE_URL|POSTGRES_PORT)=' .env.docker
    echo ""
    echo "# Host-side port published by the postgres service."
    echo "POSTGRES_PORT=${PG_PORT}"
    echo ""
    echo "# Host-side connection string for the Prisma CLI (npx prisma studio,"
    echo "# migrate, etc). Containers do NOT use this — docker-compose.yml"
    echo "# derives their DATABASE_URL from POSTGRES_* and the service name."
    echo "DATABASE_URL=\"postgresql://postgres:${PG_PW}@localhost:${PG_PORT}/hackerrank_clone?schema=public\""
  } > .env

  log "Wrote .env"
fi

# ---------------------------------------------------------------- .env.judge0
# Selects the real Judge0 backend instead of the local Docker one. Identical to
# .env.docker down to the secrets — only EXECUTOR and JUDGE0_API_URL differ.
# Generated here so the Judge0 runbook in the README does not begin with a file
# the user has to hand-assemble.
#
# Judge0 1.13.x requires a cgroup v1 host. On cgroup v2 (Docker Desktop, WSL2,
# most modern distros) every submission returns "Internal Error" — read the
# README section before reaching for this file.
if [[ -f .env.judge0 && "$FORCE" == false ]]; then
  log ".env.judge0 already exists — leaving it alone"
else
  log "Generating .env.judge0 for the real Judge0 backend"

  sed \
    -e 's|^EXECUTOR=.*|EXECUTOR=judge0|' \
    -e 's|^JUDGE0_API_URL=.*|JUDGE0_API_URL=http://judge0-server:2358|' \
    .env.docker > .env.judge0

  log "Wrote .env.judge0"
fi

# ----------------------------------------------------------- deploy/judge0.conf
# Only consumed by the optional `judge0` profile, but rendering it now means
# enabling that profile later never fails on a missing mount.
if [[ -f deploy/judge0.conf && "$FORCE" == false ]]; then
  log "deploy/judge0.conf already exists — leaving it alone"
elif [[ -f deploy/judge0.conf.template ]]; then
  log "Rendering deploy/judge0.conf"

  # shellcheck disable=SC1091
  set -a; source .env.docker; set +a

  sed \
    -e "s|__JUDGE0_WORKER_COUNT__|${JUDGE0_WORKER_COUNT:-2}|g" \
    -e "s|__JUDGE0_MAX_BATCH_SIZE__|${JUDGE0_MAX_BATCH_SIZE:-20}|g" \
    -e "s|__JUDGE0_POSTGRES_DB__|${JUDGE0_POSTGRES_DB:-judge0}|g" \
    -e "s|__JUDGE0_POSTGRES_USER__|${JUDGE0_POSTGRES_USER:-judge0}|g" \
    -e "s|__JUDGE0_POSTGRES_PASSWORD__|${JUDGE0_POSTGRES_PASSWORD:-}|g" \
    -e "s|__JUDGE0_REDIS_PASSWORD__|${JUDGE0_REDIS_PASSWORD:-}|g" \
    -e "s|__JUDGE0_AUTHN_TOKEN__|${JUDGE0_AUTHN_TOKEN:-}|g" \
    -e "s|__JUDGE0_AUTHZ_TOKEN__|${JUDGE0_AUTHZ_TOKEN:-}|g" \
    deploy/judge0.conf.template > deploy/judge0.conf
  chmod 600 deploy/judge0.conf
fi

# ------------------------------------------------------------- runner images
# Pre-pulling keeps the first demo run from stalling behind a cold image pull.
if [[ "${SKIP_PULL:-}" != "1" ]]; then
  log "Pre-pulling language runner images (skip with SKIP_PULL=1)"
  for image in python:3.11-alpine node:20-alpine gcc:13; do
    printf '    %s\n' "$image"
    docker pull -q "$image" >/dev/null 2>&1 || warn "could not pull $image — first run will be slow"
  done
fi

echo
log "Ready. Start the stack with:"
printf '\n    docker compose up -d\n\n'
printf '    Playground  http://localhost:3000\n'
printf '    API health  http://localhost:4000/health\n\n'
printf '    Demo login  coder@example.com / Password123!\n\n'
