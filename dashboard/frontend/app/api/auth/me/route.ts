import { NextRequest, NextResponse } from "next/server";
import { verifyToken } from "@/lib/auth";

export async function GET(req: NextRequest) {
  const token = req.cookies.get("barnguru_session")?.value ?? null;
  if (!token) return NextResponse.json({ username: null });
  const payload = await verifyToken(token);
  return NextResponse.json({ username: payload?.username ?? null });
}
