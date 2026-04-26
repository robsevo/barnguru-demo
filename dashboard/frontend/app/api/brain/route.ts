import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const OBSERVATIONS_PATH = path.join(
  process.cwd(),
  "..",
  "data",
  "whiz_brain",
  "observations.json"
);

export async function GET(req: NextRequest) {
  // Secondary guard — middleware already enforces, belt-and-suspenders
  const user = req.cookies.get("gretzky_user")?.value;
  if (user !== "rob") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  try {
    const raw  = await fs.readFile(OBSERVATIONS_PATH, "utf8");
    const data = JSON.parse(raw) as { observations: Observation[] };
    return NextResponse.json({
      observations: data.observations ?? [],
      fetched_at:   new Date().toISOString(),
    });
  } catch {
    return NextResponse.json({
      observations: [],
      fetched_at:   new Date().toISOString(),
      error:        "Observations file not found — no games watched yet.",
    });
  }
}

interface Observation {
  id:                  string;
  subject_type:        string;
  subject_name:        string;
  player_id:           string;
  team:                string;
  opponent:            string;
  game_id:             string;
  game_date:           string;
  observation:         string;
  category:            string;
  confidence:          number;
  sim_delta:           Record<string, number>;
  expires_after_games: number;
  active:              boolean;
  noted_at:            string;
  /** "manual" — Phase 16 auto-observer is removed for now. */
  source?:             string;
}
