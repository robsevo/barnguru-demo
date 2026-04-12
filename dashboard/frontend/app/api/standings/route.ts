import { NextResponse } from "next/server";

const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36";

interface NHLTeamStanding {
  teamAbbrev?:     { default?: string };
  teamCommonName?: { default?: string };
  divisionName?:   string;
  conferenceName?: string;
  gamesPlayed?:    number;
  wins?:           number;
  losses?:         number;
  otLosses?:       number;
  points?:         number;
  divisionSequence?:  number;
  wildcardSequence?:  number;
  conferenceSequence?: number;
  leagueSequence?:    number;
  pointPctg?:      number;
  streakCode?:     string;
  streakCount?:    number;
  goalFor?:        number;
  goalAgainst?:    number;
  goalDifferential?: number;
  homeWins?:       number;
  homeLosses?:     number;
  homeOtLosses?:   number;
  roadWins?:       number;
  roadLosses?:     number;
  roadOtLosses?:   number;
  l10Wins?:        number;
  l10Losses?:      number;
  l10OtLosses?:    number;
  shootoutWins?:   number;
  shootoutLosses?: number;
  clinchIndicator?: string;
  regulationWins?: number;
}

export async function GET() {
  try {
    const today = new Date().toLocaleDateString("en-CA"); // YYYY-MM-DD
    const res = await fetch(
      `https://api-web.nhle.com/v1/standings/${today}`,
      { headers: { "User-Agent": UA }, cache: "no-store" }
    );
    if (!res.ok) {
      return NextResponse.json({ standings: [], error: `NHL API ${res.status}` });
    }
    const raw = await res.json();

    const out = (raw.standings as NHLTeamStanding[] ?? [])
      .filter(t => t.teamAbbrev?.default)
      .map(t => ({
        team:           t.teamAbbrev?.default ?? "",
        team_name:      t.teamCommonName?.default ?? t.teamAbbrev?.default ?? "",
        division:       t.divisionName ?? "",
        conference:     t.conferenceName ?? "",
        gp:             t.gamesPlayed ?? 0,
        w:              t.wins ?? 0,
        l:              t.losses ?? 0,
        otl:            t.otLosses ?? 0,
        pts:            t.points ?? 0,
        div_rank:       t.divisionSequence ?? 0,
        wildcard_rank:  t.wildcardSequence ?? 0,
        conf_rank:      t.conferenceSequence ?? 0,
        league_rank:    t.leagueSequence ?? 0,
        pts_pct:        Math.round((t.pointPctg ?? 0) * 1000) / 1000,
        streak_code:    t.streakCode ?? "",
        streak_count:   t.streakCount ?? 0,
        gf:             t.goalFor ?? 0,
        ga:             t.goalAgainst ?? 0,
        goal_diff:      t.goalDifferential ?? 0,
        home_w:         t.homeWins ?? 0,
        home_l:         t.homeLosses ?? 0,
        home_otl:       t.homeOtLosses ?? 0,
        away_w:         t.roadWins ?? 0,
        away_l:         t.roadLosses ?? 0,
        away_otl:       t.roadOtLosses ?? 0,
        l10_w:          t.l10Wins ?? 0,
        l10_l:          t.l10Losses ?? 0,
        l10_otl:        t.l10OtLosses ?? 0,
        so_w:           t.shootoutWins ?? 0,
        so_l:           t.shootoutLosses ?? 0,
        clinch_indicator: t.clinchIndicator ?? "",
        regulation_wins: t.regulationWins ?? 0,
      }));

    return NextResponse.json({ standings: out });
  } catch (err) {
    return NextResponse.json({ standings: [], error: String(err).slice(0, 120) });
  }
}
