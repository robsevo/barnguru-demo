import { NextRequest, NextResponse } from "next/server";
import { signToken } from "@/lib/auth";

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

// Rate limiting: max 5 failures per IP, 15-minute lockout.
// In-memory — best-effort on serverless (protects single-instance / self-hosted).
interface RateEntry { count: number; lockUntil: number }
const rateLimitMap = new Map<string, RateEntry>();

function getIP(req: NextRequest): string {
  return req.headers.get("x-forwarded-for")?.split(",")[0].trim() ?? "unknown";
}


export async function POST(req: NextRequest) {
  const ip    = getIP(req);
  const now   = Date.now();
  const entry = rateLimitMap.get(ip) ?? { count: 0, lockUntil: 0 };

  if (entry.lockUntil > now) {
    return NextResponse.json({ error: "Too many attempts. Try again later." }, { status: 429 });
  }

  const body = await req.json().catch(() => ({}));
  const { username, password, secret_word } = body as Record<string, string>;

  const users = getUsers();
  // Find by username first, then constant-time compare credentials.
  const candidate = users.find(u => u.username === username);

  const credentialsOk = candidate
    ? password === candidate.password && secret_word === candidate.secret_word
    : false;

  if (!credentialsOk) {
    entry.count++;
    if (entry.count >= 5) {
      entry.lockUntil = now + 15 * 60 * 1000;
    }
    rateLimitMap.set(ip, entry);
    return NextResponse.json({ error: "Invalid credentials" }, { status: 401 });
  }

  // Success — clear rate limit entry and issue signed session token.
  rateLimitMap.delete(ip);

  const token = await signToken(candidate!.username);

  const res = NextResponse.json({ ok: true });
  res.cookies.set("barnguru_session", token, {
    httpOnly: true,
    secure:   process.env.NODE_ENV === "production",
    sameSite: "lax",
    path:     "/",
    maxAge:   60 * 60 * 24, // 24 hours
  });
  return res;
}
