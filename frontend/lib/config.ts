/**
 * Browser-safe frontend config. No Node.js built-ins — safe for client components.
 *
 * Client code calls same-origin `/api/*` routes; this URL is only for optional
 * direct health checks from the browser (see `lib/api-server.ts` for SSR).
 */

function stripTrailingSlash(url: string): string {
  return url.replace(/\/$/, "");
}

/** Public API base (env `NEXT_PUBLIC_BACKEND_URL`). */
export function getPublicBackendBase(): string {
  const fromEnv = process.env.NEXT_PUBLIC_BACKEND_URL?.trim();
  if (fromEnv) {
    return stripTrailingSlash(fromEnv);
  }
  return "http://localhost:8000";
}
