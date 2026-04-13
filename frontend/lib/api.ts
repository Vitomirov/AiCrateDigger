export type HealthResponse = {
  status: string;
  service: string;
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
