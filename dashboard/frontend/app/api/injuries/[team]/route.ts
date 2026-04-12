import { NextRequest, NextResponse } from "next/server";

const NHL_REMAP: Record<string, string> = {
  UTH: "UTA", LA: "LAK", NJ: "NJD", SJ: "SJS", TB: "TBL",
};

// Derive next-game play probability from status_raw
function playProbability(statusRaw: string): number | null {
  const s = (statusRaw ?? "").toLowerCase();
  if (s.includes("injured reserve") || s.includes("ir-") || s.includes("ir ")) return 0;
  if (s === "out") return 0;
  if (s.includes("day-to-day") || s.includes("dtd")) return 50;
  if (s.includes("probable")) return 75;
  if (s.includes("questionable")) return 40;
  if (s.includes("doubtful")) return 15;
  return null;
}

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ team: string }> }
) {
  const { team: rawTeam } = await params;
  const team = NHL_REMAP[rawTeam.toUpperCase()] ?? rawTeam.toUpperCase();

  const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36";
  try {
    const res = await fetch("http://localhost:8000/phase1/injuries", {
      headers: { "User-Agent": UA },
      cache: "no-store",
    });
    if (!res.ok) return NextResponse.json({ team, injuries: [], error: `API ${res.status}` });

    const data = await res.json();
    const raw = (data.injuries ?? []) as Record<string, unknown>[];

    const injuries = raw
      .filter(i => String(i.team_code ?? "").toUpperCase() === team)
      .map(i => ({
        player_name:      String(i.player_name ?? ""),
        position:         String(i.position ?? ""),
        status:           String(i.status ?? ""),
        status_raw:       String(i.status_raw ?? ""),
        injury_type:      String(i.injury_type ?? ""),
        injury_detail:    i.injury_detail ? String(i.injury_detail) : null,
        return_estimate:  i.return_date_estimate ? String(i.return_date_estimate) : null,
        reported_at:      i.reported_at ? String(i.reported_at) : null,
        play_probability: playProbability(String(i.status_raw ?? i.status ?? "")),
      }));

    // Sort: Out first (0%), then DTD (50%), then others
    injuries.sort((a, b) => (a.play_probability ?? 99) - (b.play_probability ?? 99));

    return NextResponse.json({ team, injuries, as_of: data.date ?? null });
  } catch (err) {
    return NextResponse.json({ team, injuries: [], error: String(err).slice(0, 80) });
  }
}
