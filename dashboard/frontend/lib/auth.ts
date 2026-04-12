// Shared auth token helpers — Web Crypto only (works in Edge middleware + Node API routes).
// Token format: base64(JSON{username,exp}) + "." + base64(HMAC-SHA256)

const TOKEN_TTL_SECONDS = 60 * 60 * 24; // 24 hours

function getSecret(): string {
  const s = process.env.AUTH_SESSION_SECRET;
  if (!s) throw new Error("AUTH_SESSION_SECRET env var not set");
  return s;
}

async function hmacKey(secret: string, usage: "sign" | "verify"): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    [usage],
  );
}

export async function signToken(username: string): Promise<string> {
  const payload = btoa(
    JSON.stringify({ username, exp: Math.floor(Date.now() / 1000) + TOKEN_TTL_SECONDS }),
  );
  const key = await hmacKey(getSecret(), "sign");
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(payload));
  const sigB64 = btoa(String.fromCharCode(...new Uint8Array(sig)));
  return payload + "." + sigB64;
}

export async function verifyToken(token: string): Promise<{ username: string } | null> {
  const dot = token.lastIndexOf(".");
  if (dot === -1) return null;
  const payload = token.slice(0, dot);
  const sig = token.slice(dot + 1);
  try {
    const key = await hmacKey(getSecret(), "verify");
    const sigBytes = Uint8Array.from(atob(sig), c => c.charCodeAt(0));
    const valid = await crypto.subtle.verify(
      "HMAC",
      key,
      sigBytes,
      new TextEncoder().encode(payload),
    );
    if (!valid) return null;
    const data = JSON.parse(atob(payload));
    if (typeof data.exp !== "number" || data.exp < Math.floor(Date.now() / 1000)) return null;
    if (typeof data.username !== "string") return null;
    return { username: data.username };
  } catch {
    return null;
  }
}
