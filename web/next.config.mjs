import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Emits .next/standalone with a minimal server + traced node_modules, which
  // is what the production Docker image runs.
  output: 'standalone',
  // The API project at the repo root has its own lockfile; pin the workspace
  // root here so Turbopack doesn't guess.
  turbopack: { root: here },
};

export default nextConfig;
