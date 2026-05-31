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

/**
 * Pipeline JSON inspector (Parse / pipeline / Listings columns).
 * Hidden in production builds unless explicitly forced on for staging.
 */
export function isDevInspectorEnabled(): boolean {
  const explicit = process.env.NEXT_PUBLIC_DEV_INSPECTOR?.trim().toLowerCase();
  if (explicit === "true") {
    return true;
  }
  if (explicit === "false") {
    return false;
  }
  return process.env.NODE_ENV !== "production";
}
