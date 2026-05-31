/**
 * Server-only checks for production BFF routes.
 * Do not import from client components.
 */

export function assertProductionBffReady(): void {
  if (process.env.NODE_ENV !== "production") {
    return;
  }

  const secret = process.env.INTERNAL_API_SECRET?.trim();
  if (!secret) {
    throw new Error(
      "INTERNAL_API_SECRET must be set for production Next.js route handlers (BFF proxy).",
    );
  }
}
