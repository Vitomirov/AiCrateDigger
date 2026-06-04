/**
 * Browser-safe API client. Uses same-origin `/api/*` proxies only — no Node built-ins.
 */

import type { SearchResponseDto } from "./api-types";

export type {
  HealthResponse,
  ListingResultDto,
  ParsedQueryDto,
  SearchEmptyReason,
  SearchResponseDto,
} from "./api-types";

export class RateLimitError extends Error {
  readonly status = 429;

  constructor(message: string) {
    super(message);
    this.name = "RateLimitError";
  }
}

function extractErrorDetail(data: unknown, fallback: string): string {
  if (typeof data !== "object" || data === null || !("detail" in data)) {
    return fallback;
  }
  const detail = (data as { detail: unknown }).detail;
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail
      .map((row) =>
        typeof row === "object" && row !== null && "msg" in row
          ? String((row as { msg: unknown }).msg)
          : JSON.stringify(row),
      )
      .join("; ");
  }
  if (detail != null) {
    return JSON.stringify(detail);
  }
  return fallback;
}

export async function postSearch(query: string): Promise<SearchResponseDto> {
  const response = await fetch("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
    cache: "no-store",
  });

  const text = await response.text();
  let data: unknown;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error("Search returned non-JSON response");
  }

  if (!response.ok) {
    const detailMsg = extractErrorDetail(data, `Search failed (${response.status})`);
    if (response.status === 429) {
      throw new RateLimitError(detailMsg);
    }
    throw new Error(detailMsg);
  }

  return data as SearchResponseDto;
}
