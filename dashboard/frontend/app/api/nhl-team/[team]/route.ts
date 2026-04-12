import { NextRequest, NextResponse } from "next/server";

const NHL_REMAP: Record<string, string> = {
  UTH: "UTA", LA: "LAK", NJ: "NJD", SJ: "SJS", TB: "TBL",
};

function fmtToi(secs: unknown): string {
  try {
    const s = Math.round(Number(secs) || 0);
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  } catch {
    return String(secs ?? "");
  }
}

function nameStr(obj: unknown): string {
  if (obj && typeof obj === "object" && "default" in obj)
    return String((obj as Record<string, unknown>).default ?? "");
  return String(obj ?? "");
}

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ team: string }> }
) {
  const { team: rawTeam } = await params;
  const team = NHL_REMAP[rawTeam.toUpperCase()] ?? rawTeam.toUpperCase();

  const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";
  const opts = { headers: { "User-Agent": UA }, redirect: "follow" as const, cache: "no-store" as const };

  // Fetch stats, roster (for jersey numbers), and injuries in parallel
  const [statsRes, rosterRes, injuriesRes] = await Promise.allSettled([
    fetch(`https://api-web.nhle.com/v1/club-stats/${team}/now`, opts),
    fetch(`https://api-web.nhle.com/v1/roster/${team}/current`, opts),
    fetch(`http://localhost:8000/phase1/injuries`, opts),
  ]);

  if (statsRes.status === "rejected" || (statsRes.status === "fulfilled" && !statsRes.value.ok)) {
    return NextResponse.json({ team, skaters: [], goalies: [], error: "NHL API unavailable" });
  }

  const statsData = await statsRes.value.json();

  // Build jersey number + status map from roster
  const jerseyMap = new Map<number, number>(); // playerId → sweaterNumber
  if (rosterRes.status === "fulfilled" && rosterRes.value.ok) {
    const roster = await rosterRes.value.json();
    for (const group of ["forwards", "defensemen", "goalies"]) {
      for (const p of (roster[group] ?? []) as Record<string, unknown>[]) {
        if (p.id && p.sweaterNumber) jerseyMap.set(Number(p.id), Number(p.sweaterNumber));
      }
    }
  }

  // Build injury map from live API: playerName → {status, injury_type}
  const injuryMap = new Map<string, { status: string; detail: string }>();
  if (injuriesRes.status === "fulfilled" && injuriesRes.value.ok) {
    const injData = await injuriesRes.value.json();
    for (const inj of (injData.injuries ?? []) as Record<string, unknown>[]) {
      if (inj.team_code === team && inj.player_name) {
        const key = String(inj.player_name).toLowerCase();
        injuryMap.set(key, {
          status: String(inj.status ?? ""),
          detail: String(inj.injury_type ?? inj.injury_detail ?? ""),
        });
      }
    }
  }

  const skaters = ((statsData.skaters ?? []) as Record<string, unknown>[]).map((p) => {
    const ppg = Number(p.powerPlayGoals ?? 0);
    const ppa = Number(p.powerPlayAssists ?? 0);
    const id = Number(p.playerId ?? 0);
    const firstName = nameStr(p.firstName);
    const lastName = nameStr(p.lastName);
    const fullName = `${firstName} ${lastName}`.toLowerCase();
    const inj = injuryMap.get(fullName);
    return {
      player_id:    id || null,
      headshot:     p.headshot ?? "",
      first_name:   firstName,
      last_name:    lastName,
      position:     p.positionCode ?? "",
      jersey:       jerseyMap.get(id) ?? null,
      gp:           p.gamesPlayed ?? 0,
      goals:        p.goals ?? 0,
      assists:      p.assists ?? 0,
      points:       p.points ?? 0,
      plus_minus:   p.plusMinus ?? 0,
      pim:          p.penaltyMinutes ?? 0,
      pp_points:    ppg + ppa,
      gwg:          p.gameWinningGoals ?? 0,
      shots:        p.shots ?? 0,
      shooting_pct: Math.round(Number(p.shootingPctg ?? 0) * 1000) / 10,
      avg_toi:      fmtToi(p.avgTimeOnIcePerGame),
      faceoff_pct:  Math.round(Number(p.faceoffWinPctg ?? 0) * 1000) / 10,
      injury_status: inj?.status ?? null,
      injury_detail: inj?.detail ?? null,
    };
  });

  skaters.sort((a, b) => (b.points as number) - (a.points as number) || (b.goals as number) - (a.goals as number));

  // NHL API field name varies — try all known variants
  const goalieRaw = (
    statsData.goalies ?? statsData.goaltenders ?? statsData.goalieStats ?? []
  ) as Record<string, unknown>[];

  const goalies = goalieRaw.map((g) => {
    const id = Number(g.playerId ?? 0);
    const firstName = nameStr(g.firstName);
    const lastName = nameStr(g.lastName);
    const fullName = `${firstName} ${lastName}`.toLowerCase();
    const inj = injuryMap.get(fullName);
    const gaa = g.goalsAgainstAvg ?? g.goalsAgainstAverage ?? g.gaa ?? 0;
    const svp = g.savePctg ?? g.savePercentage ?? g.svPctg ?? 0;
    const otl = g.otLosses ?? g.overtimeLosses ?? g.ot ?? 0;
    return {
      player_id:     id || null,
      headshot:      g.headshot ?? "",
      first_name:    firstName,
      last_name:     lastName,
      jersey:        jerseyMap.get(id) ?? null,
      gp:            g.gamesPlayed ?? g.gp ?? 0,
      wins:          g.wins ?? g.w ?? 0,
      losses:        g.losses ?? g.l ?? 0,
      ot_losses:     Number(otl),
      gaa:           Math.round(Number(gaa) * 100) / 100,
      sv_pct:        Math.round(Number(svp) * 1000) / 1000,
      shutouts:      g.shutouts ?? g.so ?? 0,
      injury_status: inj?.status ?? null,
      injury_detail: inj?.detail ?? null,
    };
  });

  return NextResponse.json({ team, skaters, goalies, _raw_goalie_count: goalieRaw.length });
}
