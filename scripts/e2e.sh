#!/usr/bin/env bash
# End-to-end verification of the execution engine.
#
# Drives the public HTTP surface exactly as a client would: authenticate, queue
# runs through POST /executions/run, poll to a terminal state, then assert on
# the stored record — stdout, exit code, artifacts — and confirm the row landed
# in Postgres. Afterwards it exercises the stateful agent loop: SSE chat that
# must pause at the human-approval gate (SUSPENDED), resume with APPROVED, and
# run to COMPLETED.
#
#   ./scripts/e2e.sh [API_BASE]
#
# ENV_FILE selects which compose config the Postgres assertions talk to
# (.env.docker for the local Docker backend, .env.judge0 for real Judge0):
#
#   ENV_FILE=.env.judge0 ./scripts/e2e.sh
#
# Exits non-zero on the first failed assertion so CI can gate on it.
set -uo pipefail

API="${1:-http://localhost:4000}"
ENV_FILE="${ENV_FILE:-.env.docker}"
EMAIL="e2e@test.local"
USERNAME="e2etest"
PASSWORD="E2eTest!2026"

PASS=0
FAIL=0
SKIP=0

ok()   { echo "  PASS  $1"; PASS=$((PASS + 1)); }
bad()  { echo "  FAIL  $1"; echo "        expected: $2"; echo "        actual:   $3"; FAIL=$((FAIL + 1)); }
skip() { echo "  SKIP  $1"; SKIP=$((SKIP + 1)); }

# jq is not guaranteed on a Windows/git-bash host, so field extraction goes
# through node, which the repo already depends on.
jf() { node -e '
  let raw = "";
  process.stdin.on("data", (c) => (raw += c));
  process.stdin.on("end", () => {
    let value;
    try { value = JSON.parse(raw); } catch { process.stdout.write(""); return; }
    for (const key of process.argv[1].split(".")) {
      if (value == null) break;
      value = value[key];
    }
    process.stdout.write(value == null ? "" : String(value));
  });' "$1"; }

echo "=== engine ==="
curl -s -m 10 "$API/health" | node -e '
  let r=""; process.stdin.on("data",c=>r+=c); process.stdin.on("end",()=>{
    const e=JSON.parse(r).engine;
    console.log(`  backend=${e.kind} endpoint=${e.endpoint} webhook=${e.usesCallback}`);
  });'

echo "=== auth ==="
curl -s -m 20 -X POST "$API/api/v1/auth/register" -H 'Content-Type: application/json' \
  -d "{\"email\":\"$EMAIL\",\"username\":\"$USERNAME\",\"password\":\"$PASSWORD\"}" >/dev/null

TOKEN=$(curl -s -m 20 -X POST "$API/api/v1/auth/login" -H 'Content-Type: application/json' \
  -d "{\"identifier\":\"$EMAIL\",\"password\":\"$PASSWORD\"}" | jf token)

if [ -z "$TOKEN" ]; then echo "  FATAL: could not obtain a token"; exit 1; fi
ok "authenticated as $USERNAME"

# run <label> <languageId> <source-file> <stdin> <expected-stdout-substring> <expected-artifacts>
run_case() {
  local label="$1" lang="$2" srcfile="$3" stdin="$4" want="$5" wantArtifacts="$6"
  echo "=== $label ==="

  # Build the JSON body in node so the source survives quoting intact.
  local body
  body=$(SRC_FILE="$srcfile" LANG_ID="$lang" STDIN="$stdin" NAME="$label" node -e '
    const fs = require("fs");
    process.stdout.write(JSON.stringify({
      code: fs.readFileSync(process.env.SRC_FILE, "utf8"),
      languageId: Number(process.env.LANG_ID),
      name: process.env.NAME,
      stdin: process.env.STDIN || undefined,
    }));')

  local created id
  created=$(curl -s -m 60 -X POST "$API/api/v1/executions/run" \
    -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d "$body")
  id=$(echo "$created" | jf execution.id)

  if [ -z "$id" ]; then bad "$label queued" "an execution id" "$created"; return; fi
  ok "$label queued id=$id"

  local status detail
  for _ in $(seq 1 120); do
    detail=$(curl -s -m 20 "$API/api/v1/executions/$id" -H "Authorization: Bearer $TOKEN")
    status=$(echo "$detail" | jf execution.status)
    case "$status" in COMPLETED|FAILED) break ;; esac
    sleep 1
  done

  if [ "$status" != "COMPLETED" ]; then
    bad "$label reached COMPLETED" "COMPLETED" "$status / $(echo "$detail" | jf execution.errorMessage)"
    echo "        stderr: $(echo "$detail" | jf execution.stderr)"
    echo "        compile: $(echo "$detail" | jf execution.compileOutput)"
    return
  fi
  ok "$label completed (judge=$(echo "$detail" | jf execution.judgeStatus))"

  local exitCode stdout artifacts
  exitCode=$(echo "$detail" | jf execution.exitCode)
  stdout=$(echo "$detail" | jf execution.stdout)
  artifacts=$(echo "$detail" | node -e '
    let r=""; process.stdin.on("data",c=>r+=c);
    process.stdin.on("end",()=>process.stdout.write(String(JSON.parse(r).execution.artifacts.length)));')

  [ "$exitCode" = "0" ] && ok "$label exit code 0" || bad "$label exit code" "0" "$exitCode"

  case "$stdout" in
    *"$want"*) ok "$label stdout contains \"$want\"" ;;
    *)         bad "$label stdout" "contains \"$want\"" "$stdout" ;;
  esac

  [ "$artifacts" = "$wantArtifacts" ] \
    && ok "$label produced $artifacts artifact(s)" \
    || bad "$label artifact count" "$wantArtifacts" "$artifacts"

  # Timing/memory should be measured, not null.
  local timeMs
  timeMs=$(echo "$detail" | jf execution.timeMs)
  [ -n "$timeMs" ] && ok "$label reported timeMs=$timeMs" || bad "$label timing" "a number" "null"

  # Download the first artifact and confirm the bytes come back intact.
  if [ "$wantArtifacts" != "0" ]; then
    local aid mime
    aid=$(echo "$detail" | node -e '
      let r=""; process.stdin.on("data",c=>r+=c);
      process.stdin.on("end",()=>process.stdout.write(JSON.parse(r).execution.artifacts[0].id));')
    mime=$(curl -s -m 20 -o /tmp/e2e-artifact.bin -w '%{content_type}' \
      "$API/api/v1/executions/$id/artifacts/$aid" -H "Authorization: Bearer $TOKEN")
    local size
    size=$(wc -c < /tmp/e2e-artifact.bin | tr -d ' ')
    [ "$size" -gt 0 ] \
      && ok "$label artifact downloaded ($size bytes, $mime)" \
      || bad "$label artifact download" ">0 bytes" "$size bytes"
  fi

  LAST_ID="$id"
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# --- Python: stdin, stderr, and an emitted artifact --------------------------
cat > "$TMP/case.py" <<'PY'
import base64, json, sys

name = sys.stdin.readline().strip() or "world"
print(f"hello {name} from python")
print("this line goes to stderr", file=sys.stderr)

payload = json.dumps({"engine": "python", "ok": True}).encode()
print(f"::artifact:result.json:application/json:{base64.b64encode(payload).decode()}::")
print("done")
PY
run_case "python" 71 "$TMP/case.py" "researcher" "hello researcher from python" 1

# --- C++: exercises the compile step ----------------------------------------
cat > "$TMP/case.cpp" <<'CPP'
#include <iostream>
int main() {
    long long total = 0;
    for (int i = 1; i <= 1000; ++i) total += i;
    std::cout << "sum=" << total << " from c++" << std::endl;
    return 0;
}
CPP
run_case "cpp" 54 "$TMP/case.cpp" "" "sum=500500 from c++" 0

# --- Node.js ------------------------------------------------------------------
cat > "$TMP/case.js" <<'JS'
const values = [1, 2, 3, 4, 5];
const total = values.reduce((a, b) => a + b, 0);
console.log(`sum=${total} from node ${process.version.split(".")[0]}`);
JS
run_case "node" 63 "$TMP/case.js" "" "sum=15 from node" 0

# --- SSE helpers --------------------------------------------------------------
# Raw SSE responses are saved to files (curl -N) and queried with node,
# following the same no-jq convention as jf() above.

# sse_has <file> <event-name> -> "1" or "0"
sse_has() {
  SSE_FILE="$1" SSE_EVENT="$2" node -e '
    const fs = require("fs");
    const raw = fs.readFileSync(process.env.SSE_FILE, "utf8");
    const found = raw.split(/\r?\n\r?\n/).some((block) => {
      const m = block.match(/^event:\s*(\S+)/m);
      return m && m[1] === process.env.SSE_EVENT;
    });
    process.stdout.write(found ? "1" : "0");'
}

# sse_data <file> <event-name> <dotted-path> -> data field of first match
# (No bare `return`: node -e evaluates as a script, where it is illegal.)
sse_data() {
  SSE_FILE="$1" SSE_EVENT="$2" SSE_PATH="$3" node -e '
    const fs = require("fs");
    const raw = fs.readFileSync(process.env.SSE_FILE, "utf8");
    const blocks = raw.split(/\r?\n\r?\n/);
    let out = "";
    for (const block of blocks) {
      const m = block.match(/^event:\s*(\S+)/m);
      if (!m || m[1] !== process.env.SSE_EVENT) continue;
      const payload = block.split(/\r?\n/)
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trimStart()).join("\n");
      let value;
      try { value = JSON.parse(payload); } catch { value = payload; }
      for (const key of process.env.SSE_PATH.split(".")) {
        if (value == null || typeof value !== "object") { value = undefined; break; }
        value = value[key];
      }
      out = value == null ? "" : String(value);
      break;
    }
    process.stdout.write(out);'
}

# sse_tool_started <file> <tool-name> -> "1" if a tool_start named it fired
sse_tool_started() {
  SSE_FILE="$1" SSE_TOOL="$2" node -e '
    const fs = require("fs");
    const raw = fs.readFileSync(process.env.SSE_FILE, "utf8");
    const found = raw.split(/\r?\n\r?\n/).some((block) => {
      if (!/^event:\s*tool_start/m.test(block)) return false;
      const payload = block.split(/\r?\n/)
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trimStart()).join("\n");
      try { return JSON.parse(payload).tool === process.env.SSE_TOOL; }
      catch { return false; }
    });
    process.stdout.write(found ? "1" : "0");'
}

# wait_session_status <id> <want> <tries> -> "1" once observed, else "0"
wait_session_status() {
  local id="$1" want="$2" tries="$3" got=""
  for _ in $(seq 1 "$tries"); do
    got=$(curl -s -m 20 "$API/api/v1/agent/sessions/$id" -H "Authorization: Bearer $TOKEN" | jf session.status)
    [ "$got" = "$want" ] && { echo 1; return; }
    sleep 2
  done
  echo 0
}

# --- agent (SSE chat + HITL resume) -------------------------------------------
echo "=== agent (SSE + HITL) ==="

# Negative auth check: no token must be rejected, never streamed.
UNAUTH_CODE=$(curl -s -m 20 -o /dev/null -w '%{http_code}' -X POST "$API/api/v1/agent/chat" \
  -H 'Content-Type: application/json' -d '{"prompt":"hi"}')
[ "$UNAUTH_CODE" = "401" ] \
  && ok "agent chat without token rejected (401)" \
  || bad "agent chat auth" "401" "$UNAUTH_CODE"

# The prompt forces the privileged tool so the approval gate must trip.
AGENT_PROMPT="Run the shell command \`echo hello-from-agent\` using the shell_executor tool. Use shell_executor for this turn; do not call any other tool first."
AGENT_BODY=$(PROMPT="$AGENT_PROMPT" node -e 'process.stdout.write(JSON.stringify({ prompt: process.env.PROMPT }))')

curl -s -N -m 300 -X POST "$API/api/v1/agent/chat" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' -d "$AGENT_BODY" -o "$TMP/agent-chat.sse"

# The stream must at least open with a session event; otherwise the body is an
# HTTP-level error page, not SSE, and nothing downstream can work.
if [ "$(sse_has "$TMP/agent-chat.sse" session)" != "1" ]; then
  bad "agent chat stream" "a session event" "$(head -c 300 "$TMP/agent-chat.sse")"
  AGENT_SKIP=1
fi

# Preflight: without a usable server-side LLM the reasoner cannot issue tool
# calls, so the approval flow is unreachable. Skip (don't fail) on key/model/
# connectivity errors; anything else is a real failure.
AGENT_SKIP=""
CHAT_ERR=$(sse_data "$TMP/agent-chat.sse" error message)
if [ -n "$CHAT_ERR" ]; then
  CHAT_ERR_LC=$(printf '%s' "$CHAT_ERR" | tr '[:upper:]' '[:lower:]')
  case "$CHAT_ERR_LC" in
    *api*key*|*openai*|*unauthorized*|*401*|*403*|*quota*|*billing*|*insufficient*|*model*not*found*|*enotfound*|*eai_again*|*econnrefused*|*fetch*failed*|*timeout*)
      skip "agent HITL flow (no usable LLM from the API: $CHAT_ERR)"
      AGENT_SKIP=1 ;;
    *) bad "agent chat stream" "no error event" "$CHAT_ERR"; AGENT_SKIP=1 ;;
  esac
fi

if [ -z "$AGENT_SKIP" ]; then
  [ "$(sse_has "$TMP/agent-chat.sse" interrupt)" = "1" ] \
    && ok "agent chat emitted an interrupt event" \
    || bad "agent chat interrupt" "one interrupt event" "none (see $TMP/agent-chat.sse)"

  [ "$(sse_tool_started "$TMP/agent-chat.sse" shell_executor)" = "1" ] \
    && ok "agent requested shell_executor (privileged tool)" \
    || bad "agent tool_start" "shell_executor requested" "not observed"

  AGENT_SESSION=$(sse_data "$TMP/agent-chat.sse" session agentSessionId)
  [ -z "$AGENT_SESSION" ] && AGENT_SESSION=$(sse_data "$TMP/agent-chat.sse" complete agentSessionId)
  if [ -z "$AGENT_SESSION" ]; then
    bad "agent session id" "a session uuid from the stream" "none"
  else
    ok "agent session $AGENT_SESSION"

    [ "$(wait_session_status "$AGENT_SESSION" SUSPENDED 30)" = "1" ] \
      && ok "agent session reached SUSPENDED" \
      || bad "agent session status" "SUSPENDED" "$(curl -s -m 20 "$API/api/v1/agent/sessions/$AGENT_SESSION" -H "Authorization: Bearer $TOKEN" | jf session.status)"

    RESUME_BODY=$(AGENT_SESSION="$AGENT_SESSION" node -e 'process.stdout.write(JSON.stringify({
      agentSessionId: process.env.AGENT_SESSION, approved: true, reason: "e2e approval" }))')
    curl -s -N -m 300 -X POST "$API/api/v1/agent/resume" \
      -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
      -H 'Accept: text/event-stream' -d "$RESUME_BODY" -o "$TMP/agent-resume.sse"

    RESUME_ERR=$(sse_data "$TMP/agent-resume.sse" error message)
    [ -z "$RESUME_ERR" ] \
      && ok "agent resume streamed without errors" \
      || bad "agent resume stream" "no error event" "$RESUME_ERR"

    [ "$(sse_data "$TMP/agent-resume.sse" complete status)" = "COMPLETED" ] \
      && ok "agent resume completed the turn" \
      || bad "agent resume complete" "COMPLETED" "$(sse_data "$TMP/agent-resume.sse" complete status)"

    [ "$(wait_session_status "$AGENT_SESSION" COMPLETED 60)" = "1" ] \
      && ok "agent session reached COMPLETED" \
      || bad "agent session status" "COMPLETED" "$(curl -s -m 20 "$API/api/v1/agent/sessions/$AGENT_SESSION" -H "Authorization: Bearer $TOKEN" | jf session.status)"

    curl -s -m 20 "$API/api/v1/agent/sessions?page=1&pageSize=20" \
      -H "Authorization: Bearer $TOKEN" | grep -q "$AGENT_SESSION" \
      && ok "agent session listed in GET /agent/sessions" \
      || bad "agent session listing" "session present" "absent"
  fi
fi

# --- agent tool allowlist (user-selected tools) --------------------------------
if [ -z "${AGENT_SKIP:-}" ]; then
  echo "=== agent tools ==="
  TOOLS_COUNT=$(curl -s -m 20 "$API/api/v1/agent/tools" -H "Authorization: Bearer $TOKEN" \
    | node -e 'let r="";process.stdin.on("data",c=>r+=c);process.stdin.on("end",()=>{try{process.stdout.write(String(JSON.parse(r).tools.length))}catch{}})')
  [ "$TOOLS_COUNT" = "8" ] \
    && ok "agent tool catalog lists 8 tools" \
    || bad "agent tool catalog" "8 tools" "$TOOLS_COUNT"

  # Calculator-only turn: must complete with no interrupt and no other tool.
  ALLOW_BODY=$(node -e 'process.stdout.write(JSON.stringify({
    prompt: "What is 12 times 11? Use calculator_engine only.",
    tools: ["calculator_engine"] }))')
  curl -s -N -m 300 -X POST "$API/api/v1/agent/chat" \
    -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
    -H 'Accept: text/event-stream' -d "$ALLOW_BODY" -o "$TMP/agent-allow.sse"

  [ "$(sse_tool_started "$TMP/agent-allow.sse" calculator_engine)" = "1" ] \
    && ok "allowlisted turn used calculator_engine" \
    || bad "allowlisted turn" "calculator_engine used" "not observed"
  [ "$(sse_tool_started "$TMP/agent-allow.sse" shell_executor)" = "0" ] \
    && ok "allowlisted turn never touched shell_executor" \
    || bad "allowlist enforcement" "no shell_executor" "observed"
  [ "$(sse_has "$TMP/agent-allow.sse" interrupt)" = "0" ] \
    && ok "allowlisted turn needed no approval" \
    || bad "allowlisted turn interrupt" "none" "observed"
  [ "$(sse_data "$TMP/agent-allow.sse" complete status)" = "COMPLETED" ] \
    && ok "allowlisted turn completed" \
    || bad "allowlisted turn complete" "COMPLETED" "$(sse_data "$TMP/agent-allow.sse" complete status)"

  # Unknown names fail fast with a 400-style error event, not silent filtering.
  curl -s -N -m 60 -X POST "$API/api/v1/agent/chat" \
    -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
    -H 'Accept: text/event-stream' -d '{"prompt":"hi","tools":["nope_not_a_tool"]}' \
    -o "$TMP/agent-badtool.sse"
  BADTOOL_ERR=$(sse_data "$TMP/agent-badtool.sse" error message)
  case "$BADTOOL_ERR" in
    *"Unknown agent tools"*) ok "unknown tool names rejected" ;;
    *) bad "unknown tool names" "Unknown agent tools error" "${BADTOOL_ERR:-none}" ;;
  esac
fi

# --- persistence --------------------------------------------------------------
echo "=== postgres ==="
ROWS=$(docker compose --env-file "$ENV_FILE" exec -T postgres \
  psql -U postgres -d hackerrank_clone -tAc \
  "SELECT count(*) FROM executions WHERE status='COMPLETED';" 2>/dev/null | tr -d '\r ')
[ -n "$ROWS" ] && [ "$ROWS" -ge 3 ] \
  && ok "postgres holds $ROWS completed execution(s)" \
  || bad "postgres execution rows" ">=3" "$ROWS"

ARTS=$(docker compose --env-file "$ENV_FILE" exec -T postgres \
  psql -U postgres -d hackerrank_clone -tAc "SELECT count(*) FROM artifacts;" 2>/dev/null | tr -d '\r ')
[ -n "$ARTS" ] && [ "$ARTS" -ge 1 ] \
  && ok "postgres holds $ARTS artifact(s)" \
  || bad "postgres artifact rows" ">=1" "$ARTS"

AGENT_ROWS=$(docker compose --env-file "$ENV_FILE" exec -T postgres \
  psql -U postgres -d hackerrank_clone -tAc "SELECT count(*) FROM agent_sessions WHERE status='COMPLETED';" 2>/dev/null | tr -d '\r ')
if [ -z "${AGENT_SKIP:-}" ]; then
  [ -n "$AGENT_ROWS" ] && [ "$AGENT_ROWS" -ge 1 ] \
    && ok "postgres holds $AGENT_ROWS completed agent session(s)" \
    || bad "postgres agent session rows" ">=1 completed" "$AGENT_ROWS"
else
  echo "  SKIP  postgres agent session rows (agent flow skipped)"
fi

echo
echo "=== summary ==="
echo "  passed: $PASS"
echo "  failed: $FAIL"
echo "  skipped: $SKIP"
[ "$FAIL" -eq 0 ] || exit 1
echo "  ALL GREEN"
