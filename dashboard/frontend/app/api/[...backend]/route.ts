import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function proxy(req: NextRequest, pathParts: string[]): Promise<NextResponse> {
  const path = pathParts.join("/");
  const { search } = new URL(req.url);
  const targetUrl = `${API_URL}/${path}${search}`;

  const headers = new Headers(req.headers);
  headers.set("ngrok-skip-browser-warning", "skip");
  headers.delete("host");

  const upstream = await fetch(targetUrl, {
    method: req.method,
    headers,
    body: req.method !== "GET" && req.method !== "HEAD" ? req.body : undefined,
    // @ts-expect-error — Node 18+ fetch supports duplex
    duplex: "half",
  });

  const body = await upstream.arrayBuffer();
  return new NextResponse(body, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("Content-Type") ?? "application/json",
    },
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
