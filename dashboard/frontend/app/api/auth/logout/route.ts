import { NextResponse } from "next/server";

export async function POST() {
  const res = NextResponse.json({ ok: true });
  res.cookies.set("grtzky_session", "", { maxAge: 0, path: "/" });
  res.cookies.set("gretzky_user", "", { maxAge: 0, path: "/" });
  return res;
}
