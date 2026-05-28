import { NextResponse } from "next/server";

import { getServerBackendBase } from "../../../lib/server-backend-url";

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

export async function POST(request: Request): Promise<NextResponse> {
  const backendBase = getServerBackendBase();

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body" }, { status: 400 });
  }

  const clientIp = resolveClientIp(request);
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (clientIp) {
    headers["X-Forwarded-For"] = clientIp;
  }

  try {
    const res = await fetch(`${backendBase}/search`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      cache: "no-store",
    });

    const text = await res.text();
    const contentType = res.headers.get("content-type") ?? "application/json";

    return new NextResponse(text, {
      status: res.status,
      headers: { "Content-Type": contentType },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "fetch failed";
    console.error("search_proxy_failed", { backendBase, message });
    return NextResponse.json(
      { detail: `Could not reach search backend at ${backendBase}.` },
      { status: 502 },
    );
  }
}
