/**
 * Server-only API helpers (RSC / Route Handlers). Not for client components.
 */

import type { HealthResponse } from "./api-types";
import { getHealthCheckBackendBase } from "./server-backend-url";

export async function fetchHealth(): Promise<HealthResponse> {
  const base = getHealthCheckBackendBase();
  const response = await fetch(`${base}/health`, {
    method: "GET",
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Health-check failed with status ${response.status}`);
  }

  return (await response.json()) as HealthResponse;
}
