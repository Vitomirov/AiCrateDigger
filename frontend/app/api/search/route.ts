import { NextResponse } from "next/server";

import { buildBackendProxyHeaders } from "../../../lib/backend-proxy-headers";
import { assertProductionBffReady } from "../../../lib/production-guard";
import { getServerBackendBase } from "../../../lib/server-backend-url";

export async function POST(request: Request): Promise<NextResponse> {
  assertProductionBffReady();
  const backendBase = getServerBackendBase();

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body" }, { status: 400 });
  }

  try {
    const res = await fetch(`${backendBase}/search`, {
      method: "POST",
      headers: buildBackendProxyHeaders(request),
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
