# syntax=docker/dockerfile:1.7

###############################################################################
# Next.js frontend — multi-stage build producing a standalone server image.
#
# NOTE: NEXT_PUBLIC_API_URL is inlined into the client bundle at BUILD time, so
# it is a build arg, not a runtime env var. It must be a URL the end user's
# browser can reach. Server-side fetches use API_INTERNAL_URL at runtime.
###############################################################################

ARG NODE_VERSION=22-alpine

# --- base --------------------------------------------------------------------
FROM node:${NODE_VERSION} AS base
RUN apk add --no-cache libc6-compat
WORKDIR /app
ENV NPM_CONFIG_UPDATE_NOTIFIER=false CI=true

# --- deps --------------------------------------------------------------------
FROM base AS deps
COPY package.json package-lock.json ./
RUN npm ci

# --- build -------------------------------------------------------------------
FROM deps AS build
ARG NEXT_PUBLIC_API_URL=http://localhost:4000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
ENV NEXT_TELEMETRY_DISABLED=1
COPY . .
RUN npm run build

# --- runtime -----------------------------------------------------------------
FROM base AS runtime
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0 \
    NEXT_TELEMETRY_DISABLED=1

# `output: 'standalone'` traces exactly the files the server needs.
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static

USER node

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||3000)+'/problems').then(r=>process.exit(r.status<500?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server.js"]
