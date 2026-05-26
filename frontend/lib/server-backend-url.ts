/**
 * Server-only backend base URL for Route Handlers and RSC.
 * Do not import this file from client components (`"use client"`).
 */

import { getPublicBackendBase } from "./config";

function stripTrailingSlash(url: string): string {
  return url.replace(/\/$/, "");
}

/**
 * Base URL for server-side proxy routes (`app/api/search`, `app/api/parse`).
 *
 * - `BACKEND_URL` — Docker Compose / server (e.g. `http://backend:8000`)
 * - `NEXT_PUBLIC_BACKEND_URL` — fallback for local dev
 * - `http://backend:8000` — Compose service default when env is unset in container
 */
export function getServerBackendBase(): string {
  const fromEnv = process.env.BACKEND_URL?.trim();
  if (fromEnv) {
    return stripTrailingSlash(fromEnv);
  }

  const publicUrl = process.env.NEXT_PUBLIC_BACKEND_URL?.trim();
  if (publicUrl) {
    return stripTrailingSlash(publicUrl);
  }

  return "http://backend:8000";
}

/** SSR health check: prefer internal URL, fall back to public base. */
export function getHealthCheckBackendBase(): string {
  if (process.env.BACKEND_URL?.trim()) {
    return getServerBackendBase();
  }
  return getPublicBackendBase();
}
