import { NextResponse } from "next/server";

const DEFAULT_BACKEND_API_BASE_URL = "http://127.0.0.1:4001/api";

function getBackendApiBaseUrl(): string {
  return (
    process.env.CHEMVERIFY_API_BASE_URL?.replace(/\/$/, "") ??
    DEFAULT_BACKEND_API_BASE_URL
  );
}

export async function proxyToBackend(path: string, init: RequestInit = {}) {
  try {
    const response = await fetch(`${getBackendApiBaseUrl()}${path}`, {
      ...init,
      cache: "no-store",
      headers: {
        ...(init.headers ?? {}),
      },
    });

    const body = await response.arrayBuffer();
    const contentType = response.headers.get("content-type");
    const cacheControl = response.headers.get("cache-control");
    const contentDisposition = response.headers.get("content-disposition");

    return new NextResponse(body, {
      status: response.status,
      headers: {
        ...(contentType ? { "content-type": contentType } : {}),
        ...(cacheControl ? { "cache-control": cacheControl } : {}),
        ...(contentDisposition ? { "content-disposition": contentDisposition } : {}),
      },
    });
  } catch (error) {
    return NextResponse.json(
      {
        detail: error instanceof Error ? error.message : "Backend proxy request failed.",
      },
      { status: 502 },
    );
  }
}
