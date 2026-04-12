import { NextRequest, NextResponse } from "next/server";
import { verifyToken } from "@/lib/auth";

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Auth routes and static assets — always allow
  if (
    pathname === "/login" ||
    pathname.startsWith("/api/auth/") ||
    pathname.startsWith("/_next/") ||
    pathname.startsWith("/favicon")
  ) {
    const headers = new Headers(request.headers);
    headers.set("ngrok-skip-browser-warning", "true");
    return NextResponse.next({ request: { headers } });
  }

  const token   = request.cookies.get("grtzky_session")?.value;
  const payload = token ? await verifyToken(token) : null;

  if (!payload) {
    // API calls → 401, page requests → redirect to login
    if (pathname.startsWith("/api/")) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    return NextResponse.redirect(new URL("/login", request.url));
  }

  // Rob-only routes: /whiz and /api/brain
  if (pathname.startsWith("/whiz") || pathname.startsWith("/api/brain")) {
    if (payload.username !== "rob") {
      if (pathname.startsWith("/api/")) {
        return NextResponse.json({ error: "Forbidden" }, { status: 403 });
      }
      return NextResponse.redirect(new URL("/", request.url));
    }
  }

  const headers = new Headers(request.headers);
  headers.set("ngrok-skip-browser-warning", "true");
  return NextResponse.next({ request: { headers } });
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico|favicon|apple-touch-icon|icon-|manifest.json|gretz.jpg|logo).*)"],
};
