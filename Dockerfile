# syntax=docker/dockerfile:1.7

###############################################################################
# Express API — multi-stage build
#
# Targets:
#   runtime   (default) slim production image, prod deps only
#   migrator            full deps + Prisma CLI, for `migrate deploy` / `db seed`
###############################################################################

ARG NODE_VERSION=22-alpine

# --- base --------------------------------------------------------------------
FROM node:${NODE_VERSION} AS base
# Prisma's query engine needs OpenSSL; libc6-compat covers glibc-ish binaries.
RUN apk add --no-cache openssl libc6-compat
WORKDIR /app
# Every stage runs `prisma generate` explicitly (or doesn't need the client), so
# suppress @prisma/client's postinstall — it runs before the schema is copied.
ENV NPM_CONFIG_UPDATE_NOTIFIER=false \
    CI=true \
    PRISMA_SKIP_POSTINSTALL_GENERATE=true

# --- deps: every dependency, used for building and for the migrator ----------
FROM base AS deps
COPY package.json package-lock.json ./
RUN npm ci

# --- build: compile TypeScript and generate the Prisma client ----------------
FROM deps AS build
COPY prisma.config.ts tsconfig.json ./
COPY prisma ./prisma
COPY src ./src
RUN npx prisma generate && npm run build

# --- prod-deps: production dependencies only ---------------------------------
FROM base AS prod-deps
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# --- migrator: runs migrations and the seed, then exits -----------------------
# Keeps the Prisma CLI and tsx (dev deps) out of the runtime image.
FROM deps AS migrator
COPY prisma.config.ts tsconfig.json ./
COPY prisma ./prisma
RUN npx prisma generate
ENV NODE_ENV=production
CMD ["sh", "-c", "npx prisma migrate deploy && npx prisma db seed"]

# --- runtime -----------------------------------------------------------------
FROM base AS runtime
ENV NODE_ENV=production PORT=4000

# The docker CLI is used by the local execution backend (EXECUTOR=docker) to
# spawn one throwaway container per test case against the mounted host socket.
# Client only — no daemon runs in this image.
RUN apk add --no-cache docker-cli

COPY --from=prod-deps /app/node_modules ./node_modules
# The generated client lives outside the dependency tree, so bring it across
# from the build stage rather than re-running `prisma generate` here.
COPY --from=build /app/node_modules/.prisma ./node_modules/.prisma
COPY --from=build /app/node_modules/@prisma/client ./node_modules/@prisma/client
COPY --from=build /app/dist ./dist
COPY package.json ./
COPY prisma ./prisma

# Run unprivileged; the node image already ships a `node` user.
RUN chown -R node:node /app
USER node

EXPOSE 4000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||4000)+'/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "dist/server.js"]
