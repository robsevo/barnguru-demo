import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const username = req.cookies.get("gretzky_user")?.value ?? null;
  return NextResponse.json({ username });
}
