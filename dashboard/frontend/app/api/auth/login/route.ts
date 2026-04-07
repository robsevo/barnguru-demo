import { NextRequest, NextResponse } from "next/server";
import { createHash } from "crypto";

interface User {
  username:    string;
  password:    string;
  secret_word: string;
}

function getUsers(): User[] {
  const raw = process.env.AUTH_USERS;
  if (raw) {
    try {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) return parsed as User[];
    } catch {}
  }
  return [];
}

// Derive a stable cookie token from AUTH_USERS so you don't need AUTH_COOKIE_VALUE.
function getCookieVal(): string {
  const raw = process.env.AUTH_USERS ?? "";
  return createHash("sha256").update("gretzky:" + raw).digest("hex");
}

export async function POST(req: NextRequest) {
  const { username, password, secret_word } = await req.json().catch(() => ({}));

  const users = getUsers();
  const match = users.find(
    u => u.username === username && u.password === password && u.secret_word === secret_word
  );

  if (!match) {
    return NextResponse.json({ error: "Invalid credentials" }, { status: 401 });
  }

  const res = NextResponse.json({ ok: true });
  res.cookies.set("grtzky_session", getCookieVal(), {
    httpOnly: true,
    secure:   process.env.NODE_ENV === "production",
    sameSite: "lax",
    path:     "/",
    maxAge:   60 * 60 * 24 * 7,
  });
  res.cookies.set("gretzky_user", match.username, {
    httpOnly: false,
    secure:   process.env.NODE_ENV === "production",
    sameSite: "lax",
    path:     "/",
    maxAge:   60 * 60 * 24 * 7,
  });
  return res;
}
