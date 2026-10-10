<#
.SYNOPSIS
    One-time bootstrap for Windows.

.DESCRIPTION
    Generates the config that is deliberately NOT committed:

      .env.docker         real secrets, gitignored
      deploy/judge0.conf  rendered from the template, gitignored

    Safe to re-run: existing files are left alone unless -Force is passed.

.EXAMPLE
    .\setup.ps1
    .\setup.ps1 -Force      # regenerate from scratch (new secrets)
#>
param(
    [switch]$Force,
    [switch]$SkipPull
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Info { param($m) Write-Host "==> $m" -ForegroundColor Cyan }
function Warn { param($m) Write-Host "!!  $m" -ForegroundColor Yellow }
function Die  { param($m) Write-Host "xx  $m" -ForegroundColor Red; exit 1 }

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Die 'docker is not installed or not on PATH'
}
docker info *>$null
if ($LASTEXITCODE -ne 0) { Die 'Cannot talk to the Docker daemon - is Docker Desktop running?' }

# Cryptographically secure; Get-Random is not.
function New-Secret {
    param([int]$Bytes)
    $buffer = [byte[]]::new($Bytes)

    # RandomNumberGenerator::Fill is .NET Core only, so it does not exist in
    # Windows PowerShell 5.1 - the shell that ships with Windows, and therefore
    # the one most people run this with. Create() is present on both .NET
    # Framework and .NET Core, so it works everywhere.
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($buffer) } finally { $rng.Dispose() }

    -join ($buffer | ForEach-Object { $_.ToString('x2') })
}

# ---------------------------------------------------------------- .env.docker
if ((Test-Path .env.docker) -and -not $Force) {
    Info '.env.docker already exists - leaving it alone (-Force to regenerate)'
} else {
    if (-not (Test-Path .env.docker.example)) { Die 'Missing .env.docker.example' }
    Info 'Generating .env.docker with fresh secrets'

    $pgPassword     = New-Secret 24
    $jwtSecret      = New-Secret 48
    $judgePgPass    = New-Secret 24
    $judgeRedisPass = New-Secret 24
    $callbackSecret = New-Secret 24

    # Start from the example so keys added later are picked up automatically,
    # then replace the values that must be unique per install.
    $lines = Get-Content .env.docker.example

    $replacements = @{
        '^POSTGRES_PASSWORD='         = "POSTGRES_PASSWORD=$pgPassword"
        '^JWT_SECRET='                = "JWT_SECRET=$jwtSecret"
        '^JUDGE0_POSTGRES_PASSWORD='  = "JUDGE0_POSTGRES_PASSWORD=$judgePgPass"
        '^JUDGE0_REDIS_PASSWORD='     = "JUDGE0_REDIS_PASSWORD=$judgeRedisPass"
        '^JUDGE0_CALLBACK_SECRET='    = "JUDGE0_CALLBACK_SECRET=$callbackSecret"
        '^DATABASE_URL='              = "DATABASE_URL=postgresql://postgres:$pgPassword@postgres:5432/hackerrank_clone?schema=public"
    }

    $out = foreach ($line in $lines) {
        $replaced = $line
        foreach ($pattern in $replacements.Keys) {
            if ($line -match $pattern) { $replaced = $replacements[$pattern]; break }
        }
        $replaced
    }

    # LF endings: this file is read inside Linux containers.
    [IO.File]::WriteAllText(
        (Join-Path $PSScriptRoot '.env.docker'),
        ($out -join "`n") + "`n",
        (New-Object System.Text.UTF8Encoding $false)
    )
    # Isolate this checkout's containers and volumes.
    #
    # docker-compose.yml hard-codes `name: hackerrank-clone`, so without this
    # every clone on the machine shares one Compose project - and therefore one
    # Postgres volume. Postgres only applies POSTGRES_PASSWORD when it
    # initialises an empty data directory, so the second clone generates fresh
    # secrets, inherits the first clone's volume, and can never authenticate
    # against it (P1000).
    #
    # COMPOSE_PROJECT_NAME takes precedence over the `name:` key, and Compose
    # reads it straight out of this file. Project names allow [a-z0-9_-] only.
    $projectName = (Split-Path -Leaf $PSScriptRoot).ToLower() -replace '[^a-z0-9_-]', '-'
    $projectName = $projectName.Trim('-')
    if (-not $projectName) { $projectName = 'hackerrank-clone' }

    Add-Content -Path (Join-Path $PSScriptRoot '.env.docker') -Value @(
        ''
        "# Scopes this checkout's containers and volumes so a second clone on the"
        '# same machine does not inherit this one''s database.'
        "COMPOSE_PROJECT_NAME=$projectName"
    )

    Info "Wrote .env.docker (project: $projectName)"
}

# ----------------------------------------------------------------------- .env
# Compose reads `.env` by DEFAULT - not `.env.docker`. Without this file a
# plain `docker compose up -d` fails on "POSTGRES_PASSWORD is required", and
# passing --env-file every time is a footgun nobody remembers. So `.env` is
# generated from the same secrets, with DATABASE_URL pointed at the published
# host port instead of the service name, which is what the Prisma CLI needs
# when you run it outside a container.
if ((Test-Path .env) -and -not $Force) {
    Info '.env already exists - leaving it alone'
} else {
    Info 'Generating .env for plain `docker compose up -d` + host-side Prisma'

    $dockerEnv = Get-Content .env.docker
    $pgPw   = ($dockerEnv | Where-Object { $_ -match '^POSTGRES_PASSWORD=' }) -replace '^POSTGRES_PASSWORD=', ''
    $pgPort = ($dockerEnv | Where-Object { $_ -match '^POSTGRES_PORT=' })     -replace '^POSTGRES_PORT=', ''
    if (-not $pgPort) { $pgPort = '5435' }

    $kept = $dockerEnv | Where-Object { $_ -notmatch '^(DATABASE_URL|POSTGRES_PORT)=' }
    $extra = @(
        ''
        '# Host-side port published by the postgres service.'
        "POSTGRES_PORT=$pgPort"
        ''
        '# Host-side connection string for the Prisma CLI (npx prisma studio,'
        '# migrate, etc). Containers do NOT use this - docker-compose.yml'
        '# derives their DATABASE_URL from POSTGRES_* and the service name.'
        "DATABASE_URL=`"postgresql://postgres:$pgPw@localhost:$pgPort/hackerrank_clone?schema=public`""
    )

    [IO.File]::WriteAllText(
        (Join-Path $PSScriptRoot '.env'),
        (($kept + $extra) -join "`n") + "`n",
        (New-Object System.Text.UTF8Encoding $false)
    )
    Info 'Wrote .env'
}

# ---------------------------------------------------------------- .env.judge0
# Selects the real Judge0 backend instead of the local Docker one. Identical to
# .env.docker down to the secrets - only EXECUTOR and JUDGE0_API_URL differ.
# Generated here so the Judge0 runbook in the README does not begin with a file
# the user has to hand-assemble.
#
# Judge0 1.13.x requires a cgroup v1 host. On cgroup v2 (Docker Desktop, WSL2,
# most modern distros) every submission returns "Internal Error" - read the
# README section before reaching for this file.
if ((Test-Path .env.judge0) -and -not $Force) {
    Info '.env.judge0 already exists - leaving it alone'
} else {
    Info 'Generating .env.judge0 for the real Judge0 backend'

    $judge0Env = Get-Content .env.docker | ForEach-Object {
        $_ -replace '^EXECUTOR=.*', 'EXECUTOR=judge0' `
           -replace '^JUDGE0_API_URL=.*', 'JUDGE0_API_URL=http://judge0-server:2358'
    }

    [IO.File]::WriteAllText(
        (Join-Path $PSScriptRoot '.env.judge0'),
        ($judge0Env -join "`n") + "`n",
        (New-Object System.Text.UTF8Encoding $false)
    )
    Info 'Wrote .env.judge0'
}

# ----------------------------------------------------------- deploy/judge0.conf
# Only consumed by the optional `judge0` profile, but rendering it now means
# enabling that profile later never fails on a missing mount.
if ((Test-Path deploy/judge0.conf) -and -not $Force) {
    Info 'deploy/judge0.conf already exists - leaving it alone'
} elseif (Test-Path deploy/judge0.conf.template) {
    Info 'Rendering deploy/judge0.conf'

    $envMap = @{}
    foreach ($line in Get-Content .env.docker) {
        if ($line -match '^\s*([A-Z0-9_]+)\s*=\s*(.*)$') { $envMap[$Matches[1]] = $Matches[2] }
    }
    function EnvOr { param($k, $d) if ($envMap.ContainsKey($k) -and $envMap[$k]) { $envMap[$k] } else { $d } }

    $conf = Get-Content deploy/judge0.conf.template -Raw
    $conf = $conf -replace '__JUDGE0_WORKER_COUNT__',      (EnvOr 'JUDGE0_WORKER_COUNT' '2')
    $conf = $conf -replace '__JUDGE0_MAX_BATCH_SIZE__',    (EnvOr 'JUDGE0_MAX_BATCH_SIZE' '20')
    $conf = $conf -replace '__JUDGE0_POSTGRES_DB__',       (EnvOr 'JUDGE0_POSTGRES_DB' 'judge0')
    $conf = $conf -replace '__JUDGE0_POSTGRES_USER__',     (EnvOr 'JUDGE0_POSTGRES_USER' 'judge0')
    $conf = $conf -replace '__JUDGE0_POSTGRES_PASSWORD__', (EnvOr 'JUDGE0_POSTGRES_PASSWORD' '')
    $conf = $conf -replace '__JUDGE0_REDIS_PASSWORD__',    (EnvOr 'JUDGE0_REDIS_PASSWORD' '')
    $conf = $conf -replace '__JUDGE0_AUTHN_TOKEN__',       (EnvOr 'JUDGE0_AUTHN_TOKEN' '')
    $conf = $conf -replace '__JUDGE0_AUTHZ_TOKEN__',       (EnvOr 'JUDGE0_AUTHZ_TOKEN' '')

    # LF endings: judge0.conf is sourced by a shell inside a Linux container,
    # and a trailing \r turns every value into garbage.
    [IO.File]::WriteAllText(
        (Join-Path $PSScriptRoot 'deploy/judge0.conf'),
        ($conf -replace "`r`n", "`n"),
        (New-Object System.Text.UTF8Encoding $false)
    )
}

# ------------------------------------------------------------- runner images
if (-not $SkipPull) {
    Info 'Pre-pulling language runner images (skip with -SkipPull)'
    foreach ($image in @('python:3.11-alpine', 'node:20-alpine', 'gcc:13')) {
        Write-Host "    $image"
        docker pull -q $image *>$null
        if ($LASTEXITCODE -ne 0) { Warn "could not pull $image - first run will be slow" }
    }
}

Write-Host ''
Info 'Ready. Start the stack with:'
Write-Host ''
Write-Host '    docker compose up -d'
Write-Host ''
Write-Host '    Playground  http://localhost:3000'
Write-Host '    API health  http://localhost:4000/health'
Write-Host ''
Write-Host '    Demo login  coder@example.com / Password123!'
Write-Host ''
