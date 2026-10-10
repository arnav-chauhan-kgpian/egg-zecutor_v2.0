#!/usr/bin/env bash
#
# Deploys the platform: builds images, renders the Judge0 config, brings up
# PostgreSQL, applies Prisma migrations, seeds the initial problems, then starts
# the API, frontend and Judge0.
#
#   ./deploy.sh                 build if needed, migrate, seed, start everything
#   ./deploy.sh --build         force a rebuild of the API and web images
#   ./deploy.sh --no-seed       migrate but skip seeding
#   ./deploy.sh --skip-judge0   run only the app stack (API uses mock grading)
#   ./deploy.sh --down          stop the stack (volumes are kept)
#   ./deploy.sh --logs          follow logs for all services
#
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

ENV_FILE=".env.docker"
JUDGE0_TEMPLATE="deploy/judge0.conf.template"
JUDGE0_CONF="deploy/judge0.conf"

FORCE_BUILD=false
RUN_SEED=true
SKIP_JUDGE0=false
ACTION="up"

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m  %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------- args --
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)       FORCE_BUILD=true ;;
    --no-seed)     RUN_SEED=false ;;
    --skip-judge0) SKIP_JUDGE0=true ;;
    --down)        ACTION="down" ;;
    --logs)        ACTION="logs" ;;
    -h|--help)     sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)             die "Unknown option: $1 (try --help)" ;;
  esac
  shift
done

# ------------------------------------------------------------------ toolchain --
command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH"

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  die "Neither 'docker compose' nor 'docker-compose' is available"
fi

docker info >/dev/null 2>&1 || die "Cannot talk to the Docker daemon — is it running?"

# ----------------------------------------------------------------- env file ---
if [[ ! -f "$ENV_FILE" ]]; then
  [[ -f "${ENV_FILE}.example" ]] || die "Missing $ENV_FILE and ${ENV_FILE}.example"
  cp "${ENV_FILE}.example" "$ENV_FILE"
  die "Created $ENV_FILE from the example. Fill in the required secrets, then re-run."
fi

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

# The judge0 services sit behind the `judge0` Compose profile, so the profile
# has to be active for this script to start, stop or tail them. Activating it
# unconditionally is safe: profiles only auto-start profiled services when no
# service names are given, and every `up` below names its services explicitly.
# Keeping it on for --down/--logs means judge0 containers are never orphaned.
compose() { "${COMPOSE[@]}" --env-file "$ENV_FILE" --profile judge0 "$@"; }

# --------------------------------------------------------- down / logs paths --
if [[ "$ACTION" == "down" ]]; then
  log "Stopping the stack (volumes preserved)"
  compose down
  exit 0
fi

if [[ "$ACTION" == "logs" ]]; then
  compose logs -f --tail=100
  exit 0
fi

# --------------------------------------------------------------- validation ---
require() {
  local name="$1" value="${!1:-}"
  [[ -n "$value" && "$value" != "CHANGE_ME" ]] || die "$name must be set in $ENV_FILE"
}

require POSTGRES_PASSWORD
require DATABASE_URL
require JWT_SECRET

(( ${#JWT_SECRET} >= 32 )) || die "JWT_SECRET must be at least 32 characters (got ${#JWT_SECRET})"

if [[ "$DATABASE_URL" == *CHANGE_ME* ]]; then
  die "DATABASE_URL in $ENV_FILE still contains CHANGE_ME — set the real password"
fi

if [[ "$DATABASE_URL" != *"@postgres:"* ]]; then
  warn "DATABASE_URL host is not 'postgres'; inside Compose the database is reachable at that service name"
fi

if [[ "${CORS_ORIGIN:-*}" == "*" ]]; then
  warn "CORS_ORIGIN is '*' — restrict it to your frontend origin before exposing this publicly"
fi

if [[ "$SKIP_JUDGE0" == false ]]; then
  require JUDGE0_POSTGRES_PASSWORD
  require JUDGE0_REDIS_PASSWORD
fi

# ------------------------------------------------------- judge0 config file ---
render_judge0_conf() {
  log "Rendering $JUDGE0_CONF"
  [[ -f "$JUDGE0_TEMPLATE" ]] || die "Missing $JUDGE0_TEMPLATE"

  local tmp
  tmp="$(mktemp)"
  # Substitute placeholders rather than using envsubst, which isn't installed
  # everywhere; sed keeps this dependency-free.
  sed \
    -e "s|__JUDGE0_POSTGRES_DB__|${JUDGE0_POSTGRES_DB:-judge0}|g" \
    -e "s|__JUDGE0_POSTGRES_USER__|${JUDGE0_POSTGRES_USER:-judge0}|g" \
    -e "s|__JUDGE0_POSTGRES_PASSWORD__|${JUDGE0_POSTGRES_PASSWORD:-}|g" \
    -e "s|__JUDGE0_REDIS_PASSWORD__|${JUDGE0_REDIS_PASSWORD:-}|g" \
    -e "s|__JUDGE0_AUTHN_TOKEN__|${JUDGE0_AUTHN_TOKEN:-}|g" \
    -e "s|__JUDGE0_AUTHZ_TOKEN__|${JUDGE0_AUTHZ_TOKEN:-}|g" \
    -e "s|__JUDGE0_MAX_BATCH_SIZE__|${JUDGE0_MAX_BATCH_SIZE:-20}|g" \
    -e "s|__JUDGE0_WORKER_COUNT__|${JUDGE0_WORKER_COUNT:-2}|g" \
    "$JUDGE0_TEMPLATE" > "$tmp"

  mv "$tmp" "$JUDGE0_CONF"
  chmod 600 "$JUDGE0_CONF"
}

render_judge0_conf

# --------------------------------------------------------------- cgroup check --
# Ask the Docker daemon, not the machine running this script: on Windows/macOS
# the daemon lives in a VM, so testing /sys/fs/cgroup here always came back
# "v1" and the warning never fired on exactly the hosts that needed it.
if [[ "$SKIP_JUDGE0" == false ]]; then
  cgroup_version="$(docker info --format '{{.CgroupVersion}}' 2>/dev/null || true)"
  if [[ "$cgroup_version" == "2" ]]; then
    warn "Docker is running with cgroup v2; Judge0 ${JUDGE0_VERSION:-1.13.1} requires cgroup v1."
    warn "  isolate cannot create its jails and EVERY submission returns Internal Error."
    warn "  Re-run with --skip-judge0 to use mock grading, or move to a cgroup v1 host."
    warn "  See the 'Judge0 is opt-in' section of the README."
  fi
fi

# ---------------------------------------------------------------------- build --
BUILD_ARGS=()
[[ "$FORCE_BUILD" == true ]] && BUILD_ARGS+=(--no-cache)

log "Building api and web images"
compose build "${BUILD_ARGS[@]}" api web

# ------------------------------------------------------------------- database --
log "Starting PostgreSQL"
compose up -d postgres

log "Waiting for PostgreSQL to accept connections"
for attempt in {1..60}; do
  if compose exec -T postgres pg_isready \
      -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-hackerrank_clone}" >/dev/null 2>&1; then
    log "PostgreSQL is ready"
    break
  fi
  (( attempt == 60 )) && die "PostgreSQL did not become ready in time (see: ./deploy.sh --logs)"
  sleep 2
done

# ------------------------------------------------------ migrations + seeding ---
# The `migrate` service runs `prisma migrate deploy && prisma db seed`.
if [[ "$RUN_SEED" == true ]]; then
  log "Applying migrations and seeding problems"
  compose run --rm migrate
else
  log "Applying migrations (skipping seed)"
  compose run --rm migrate sh -c 'npx prisma migrate deploy'
fi

# ----------------------------------------------------------------- start apps --
if [[ "$SKIP_JUDGE0" == true ]]; then
  # Blank the endpoint so the API uses mock grading instead of trying to reach a
  # judge0-server that isn't running (which would 502 on every submission).
  export JUDGE0_API_URL=""
  warn "Skipping Judge0 — the API runs in MOCK mode: submitted code is NOT executed"
  compose up -d --no-deps postgres api web
else
  log "Starting Judge0 (server, workers, Postgres, Redis)"
  compose up -d judge0-db judge0-redis judge0-server judge0-workers

  log "Waiting for Judge0 to answer /about"
  judge0_ready=false
  for attempt in {1..60}; do
    if compose exec -T judge0-server curl -fsS http://localhost:2358/about >/dev/null 2>&1; then
      judge0_ready=true
      break
    fi
    sleep 3
  done
  if [[ "$judge0_ready" == true ]]; then
    log "Judge0 is ready"
  else
    warn "Judge0 did not report ready. The stack will still start; the API returns 502 on"
    warn "submissions until it recovers. Check: compose logs judge0-server judge0-workers"
  fi

  log "Starting API and frontend"
  compose up -d api web
fi

# ---------------------------------------------------------------------- report --
echo
log "Service status"
compose ps

echo
log "Deployed"
printf '    Frontend  http://localhost:%s\n' "${WEB_PORT:-3000}"
printf '    API       http://localhost:%s\n' "${API_PORT:-4000}"
printf '    Health    http://localhost:%s/health\n' "${API_PORT:-4000}"
if [[ "$SKIP_JUDGE0" == false ]]; then
  printf '    Judge0    http://127.0.0.1:%s/about\n' "${JUDGE0_PORT:-2358}"
fi
echo
printf '    Logs      ./deploy.sh --logs\n'
printf '    Stop      ./deploy.sh --down\n'
