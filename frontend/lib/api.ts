export type HealthResponse = {
  status: string;
  service: string;
};

export type ListingResultDto = {
  url: string;
  title: string;
  score: number;
  price: string | null;
  location: string | null;
  availability: string;
  seller_type: string;
  domain: string | null;
};

export type SearchResponseDto = {
  results: ListingResultDto[];
  debug?: Record<string, unknown> | null;
};

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${backendUrl}/health`, {
    method: "GET",
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Health-check failed with status ${response.status}`);
  }

  return (await response.json()) as HealthResponse;
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
    let detailMsg = `Search failed (${response.status})`;
    if (typeof data === "object" && data !== null && "detail" in data) {
      const detail = (data as { detail: unknown }).detail;
      if (typeof detail === "string") {
        detailMsg = detail;
      } else if (Array.isArray(detail)) {
        detailMsg = detail
          .map((row) =>
            typeof row === "object" && row !== null && "msg" in row
              ? String((row as { msg: unknown }).msg)
              : JSON.stringify(row),
          )
          .join("; ");
      } else if (detail != null) {
        detailMsg = JSON.stringify(detail);
      }
    }
    throw new Error(detailMsg);
  }

  return data as SearchResponseDto;
}
