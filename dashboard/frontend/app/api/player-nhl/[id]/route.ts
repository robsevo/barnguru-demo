import { NextRequest, NextResponse } from "next/server";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36";

  try {
    const res = await fetch(`https://api-web.nhle.com/v1/player/${id}/landing`, {
      headers: { "User-Agent": UA },
      cache: "no-store",
    });
    if (!res.ok) return NextResponse.json({ error: `NHL API ${res.status}` });

    const d = await res.json();

    // Height: NHL returns inches (e.g. 71) or cm object
    let height_cm: number | null = null;
    if (d.heightInInches) height_cm = Math.round(d.heightInInches * 2.54);
    else if (d.height) height_cm = Number(d.height) || null;

    // Weight: NHL returns lbs or kg
    let weight_kg: number | null = null;
    if (d.weightInPounds) weight_kg = Math.round(d.weightInPounds / 2.205);
    else if (d.weight) weight_kg = Number(d.weight) || null;

    // Birth date
    const birth_date: string | null = d.birthDate ?? null;

    // Birth city/province/country
    const birth_city: string | null = d.birthCity?.default ?? d.birthCity ?? null;
    const birth_province: string | null = d.birthStateProvince?.default ?? d.birthStateProvince ?? null;
    const birth_country: string | null = d.birthCountry ?? null;
    const birthplace = [birth_city, birth_province, birth_country].filter(Boolean).join(", ");

    // Shoots/catches
    const shoots_catches: string | null = d.shootsCatches ?? null;

    // Jersey number
    const jersey_number: number | null = d.sweaterNumber ?? null;

    // Draft
    const draft_year: number | null   = d.draftDetails?.year        ?? null;
    const draft_round: number | null  = d.draftDetails?.round       ?? null;
    const draft_pick: number | null   = d.draftDetails?.pickInRound ?? null;
    const draft_overall: number | null = d.draftDetails?.overallPick ?? null;
    const draft_team: string | null   = d.draftDetails?.teamAbbrev  ?? null;

    const current_team_abbrev: string | null = d.currentTeamAbbrev ?? null;

    return NextResponse.json({
      height_cm,
      weight_kg,
      birth_date,
      birthplace,
      shoots_catches,
      jersey_number,
      draft_year,
      draft_round,
      draft_pick,
      draft_overall,
      draft_team,
      current_team_abbrev,
    });
  } catch (err) {
    return NextResponse.json({ error: String(err).slice(0, 80) });
  }
}
