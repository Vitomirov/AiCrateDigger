/**
 * Server-only headers for Route Handlers proxying to the FastAPI backend.
 */

function resolveClientIp(request: Request): string | null {
  const forwarded = request.headers.get("x-forwarded-for");
  if (forwarded) {
    const first = forwarded.split(",")[0]?.trim();
    if (first) {
      return first;
    }
  }
  return request.headers.get("x-real-ip");
}

/** Headers for `/search` and `/parse` BFF proxies (secret + client IP). */
export function buildBackendProxyHeaders(request: Request): Record<string, string> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };

  const secret = process.env.INTERNAL_API_SECRET?.trim();
  if (secret) {
    headers["X-Internal-Api-Secret"] = secret;
  }

  const clientIp = resolveClientIp(request);
  if (clientIp) {
    headers["X-Forwarded-For"] = clientIp;
  }

  return headers;
}
