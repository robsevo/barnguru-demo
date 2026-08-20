import { NextRequest, NextResponse } from "next/server";

// Vercel proxy timeout. The default 10s on the Hobby tier is tight for a cold
// serverless start plus a first model-parquet read; 60s gives that room.
export const maxDuration = 60;

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function proxy(req: NextRequest, pathParts: string[]): Promise<NextResponse> {
  const path = pathParts.join("/");
  const { search } = new URL(req.url);
  const targetUrl = `${API_URL}/${path}${search}`;

  const headers = new Headers(req.headers);
  headers.delete("host");
  // Inject x-forwarded-proto so the backend's stream-proxy can correctly construct
  // segment URLs. Next.js adds x-forwarded-host but not x-forwarded-proto, causing
  // the backend (running on http://localhost:8000) to default to https and produce
  // https://localhost:3000/api/stream-proxy?url=... which fails with ERR_SSL_PROTOCOL_ERROR.
  if (!headers.has("x-forwarded-proto")) {
    headers.set("x-forwarded-proto", req.url.startsWith("https://") ? "https" : "http");
  }

  const upstream = await fetch(targetUrl, {
    method: req.method,
    headers,
    body: req.method !== "GET" && req.method !== "HEAD" ? req.body : undefined,
    // @ts-expect-error — Node 18+ fetch supports duplex
    duplex: "half",
  });

  // Stream the body through. Buffering whole segments via arrayBuffer() pushes
  // HLS fragment fetches past hls.js's timeout on slow ticks, which the player
  // raises as a fatal network error — that used to drop us to the Brightcove
  // iframe mid-clip. JSON endpoints stream fine too; body size doesn't change.
  const resHeaders = new Headers();
  const passthroughHeaders = [
    "content-type",
    "content-length",
    "content-range",
    "accept-ranges",
    "cache-control",
    "etag",
    "last-modified",
  ];
  for (const h of passthroughHeaders) {
    const v = upstream.headers.get(h);
    if (v) resHeaders.set(h, v);
  }
  if (!resHeaders.has("content-type")) {
    resHeaders.set("content-type", "application/json");
  }
  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: resHeaders,
  });
}

export async function GET(req: NextRequest, { params }: { params: Promise<{ backend: string[] }> }) {
  const { backend } = await params;
  return proxy(req, backend);
}

export async function POST(req: NextRequest, { params }: { params: Promise<{ backend: string[] }> }) {
  const { backend } = await params;
  return proxy(req, backend);
}

export async function PUT(req: NextRequest, { params }: { params: Promise<{ backend: string[] }> }) {
  const { backend } = await params;
  return proxy(req, backend);
}

export async function DELETE(req: NextRequest, { params }: { params: Promise<{ backend: string[] }> }) {
  const { backend } = await params;
  return proxy(req, backend);
}
