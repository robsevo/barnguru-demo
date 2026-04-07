import { NextRequest, NextResponse } from "next/server";

// Derive the expected cookie value from AUTH_USERS using Web Crypto (Edge-compatible).
// Same logic as the login route — no separate AUTH_COOKIE_VALUE needed.
async function expectedCookie(): Promise<string> {
  const raw  = process.env.AUTH_USERS ?? "";
  const data = new TextEncoder().encode("gretzky:" + raw);
  const hash = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(hash))
    .map(b => b.toString(16).padStart(2, "0"))
    .join("");
}

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

  const session  = request.cookies.get("grtzky_session")?.value;
  const expected = await expectedCookie();

  if (session !== expected) {
    // API calls → 401, page requests → redirect to login
    if (pathname.startsWith("/api/")) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    return NextResponse.redirect(new URL("/login", request.url));
  }

  // Rob-only routes: /whiz and /api/brain
  if (pathname.startsWith("/whiz") || pathname.startsWith("/api/brain")) {
    const user = request.cookies.get("gretzky_user")?.value;
    if (user !== "rob") {
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
