import { NextResponse } from "next/server";

import { getServerBackendBase } from "../../../lib/server-backend-url";

export async function POST(request: Request): Promise<NextResponse> {
  const backendBase = getServerBackendBase();

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body" }, { status: 400 });
  }

  try {
    const res = await fetch(`${backendBase}/parse`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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
    console.error("parse_proxy_failed", { backendBase, message });
    return NextResponse.json(
      { detail: `Could not reach parse backend at ${backendBase}.` },
      { status: 502 },
    );
  }
}
