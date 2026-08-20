"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { HudPanel, HudBadge } from "@/components/hud";

/**
 * Team page.
 *
 * Replaced a 1,844-line version that pulled a roster from the NHL's public API,
 * injuries from a second service and cap hits from a third. All three describe
 * real players, and this build's league is generated — so every one of them
 * returned nothing, slowly, and the page rendered three empty panels to say so.
 *
 * One request to `/api/teams/{abbrev}`, which returns the club, its standings
 * row, its coach and both rosters.
 */

interface Skater {
  player_id: number; name: string; position: string; number: number;
  gp: number; goals: number; assists: number; points: number;
  plusMinus: number; toiPerGame: number;
}

interface Goalie {
  player_id: number; name: string; number: number; gp: number;
  wins: number; losses: number; shutouts: number;
  savePctg: number; goalsAgainstAverage: number;
}

interface TeamPayload {
  team: { abbrev: string; name: string; conference: string; division: string } | null;
  season: number;
  standing: {
    gp: number; w: number; l: number; otl: number; pts: number;
    gf: number; ga: number; diff: number;
    div_rank: number; conf_rank: number; league_rank: number;
  } | null;
  coach: { name: string; seasons: number } | null;
  skaters: Skater[];
  goalies: Goalie[];
}

function Stat({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="text-center">
      <div className="text-[9px] uppercase tracking-[0.16em] text-white/30">{label}</div>
      <div className={`mt-0.5 text-[16px] font-mono tabular-nums ${accent ? "text-[var(--brand-hex)]" : "text-white/90"}`}>
        {value}
      </div>
    </div>
  );
}

export default function TeamPage() {
  const { team: abbrev } = useParams<{ team: string }>();
  const [data, setData] = useState<TeamPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetch(`/api/teams/${abbrev}`)
      .then(r => (r.ok ? r.json() : Promise.reject(new Error(`team ${r.status}`))))
      .then(d => { if (alive) setData(d); })
      .catch(e => { if (alive) setError(String(e.message ?? e)); });
    return () => { alive = false; };
  }, [abbrev]);

  if (error) {
    return (
      <main className="mx-auto w-full max-w-5xl px-4 py-8">
        <HudPanel title="Not available">
          <p className="text-[12px] text-[var(--text-secondary)]">{error}</p>
          <Link href="/league/teams" className="mt-3 inline-block text-[12px] text-[var(--brand-hex)]">
            ← All teams
          </Link>
        </HudPanel>
      </main>
    );
  }
  if (!data) {
    return (
      <main className="mx-auto w-full max-w-5xl px-4 py-8">
        <p className="text-[12px] text-[var(--text-secondary)]">Loading…</p>
      </main>
    );
  }

  const { team, standing, coach, skaters, goalies } = data;
  const roster = [...skaters].sort((a, b) => b.points - a.points);

  return (
    <main className="mx-auto w-full max-w-5xl px-4 py-6 flex flex-col gap-4">
      <Link href="/league/teams" className="text-[11px] text-white/35 hover:text-white/70 transition-colors">
        ← All teams
      </Link>

      <HudPanel scanline>
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center justify-center w-16 h-16 rounded-xl border border-[var(--border-default)]
                          bg-white/[0.03] text-[var(--brand-hex)] text-[15px] font-black tracking-wider">
            {team?.abbrev}
          </div>
          <div className="min-w-0">
            <h1 className="text-[26px] font-black tracking-[0.05em] uppercase text-white leading-none">
              {team?.name}
            </h1>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <span className="text-[10px] uppercase tracking-[0.18em] text-white/30">
                {team?.conference} · {team?.division}
              </span>
              {coach && <HudBadge tone="neutral">{coach.name} · {coach.seasons}y</HudBadge>}
            </div>
          </div>
        </div>
      </HudPanel>

      {standing && (
        <HudPanel title="Record" subtitle={String(data.season)}>
          <div className="grid grid-cols-4 sm:grid-cols-8 gap-y-3">
            <Stat label="GP"  value={String(standing.gp)} />
            <Stat label="W"   value={String(standing.w)} />
            <Stat label="L"   value={String(standing.l)} />
            <Stat label="OTL" value={String(standing.otl)} />
            <Stat label="PTS" value={String(standing.pts)} accent />
            <Stat label="GF"  value={String(standing.gf)} />
            <Stat label="GA"  value={String(standing.ga)} />
            <Stat label="DIFF" value={standing.diff > 0 ? `+${standing.diff}` : String(standing.diff)} />
          </div>
          <div className="mt-4 pt-3 border-t border-white/[0.05] grid grid-cols-3 gap-3">
            <Stat label="Division"   value={`#${standing.div_rank}`} />
            <Stat label="Conference" value={`#${standing.conf_rank}`} />
            <Stat label="League"     value={`#${standing.league_rank}`} />
          </div>
        </HudPanel>
      )}

      <HudPanel title="Skaters" subtitle={`${roster.length} on the roster`} padded={false}>
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-[9px] uppercase tracking-[0.18em] text-white/30">
                <th className="text-left font-semibold px-3 py-2">Player</th>
                {["GP", "G", "A", "P", "+/-", "TOI"].map(h => (
                  <th key={h} className="text-right font-semibold px-2 py-2">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {roster.map(p => (
                <tr key={p.player_id} className="border-t border-white/[0.04] hover:bg-white/[0.02]">
                  <td className="px-3 py-1.5">
                    <span className="font-mono text-white/25 mr-2 tabular-nums">{p.number}</span>
                    <Link href={`/players/${p.player_id}`} className="text-white/85 hover:text-[var(--brand-hex)] transition-colors">
                      {p.name}
                    </Link>
                    <HudBadge className="ml-2" tone="neutral">{p.position}</HudBadge>
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-white/45">{p.gp}</td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-white/70">{p.goals}</td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-white/70">{p.assists}</td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-[var(--brand-hex)]">{p.points}</td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-white/70">
                    {p.plusMinus > 0 ? `+${p.plusMinus}` : p.plusMinus}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-white/70">{p.toiPerGame.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </HudPanel>

      <HudPanel title="Goaltending" padded={false}>
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-[9px] uppercase tracking-[0.18em] text-white/30">
                <th className="text-left font-semibold px-3 py-2">Goalie</th>
                {["GP", "W", "L", "SO", "SV%", "GAA"].map(h => (
                  <th key={h} className="text-right font-semibold px-2 py-2">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {goalies.map(g => (
                <tr key={g.player_id} className="border-t border-white/[0.04]">
                  <td className="px-3 py-1.5">
                    <span className="font-mono text-white/25 mr-2 tabular-nums">{g.number}</span>
                    <span className="text-white/85">{g.name}</span>
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-white/45">{g.gp}</td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-white/70">{g.wins}</td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-white/70">{g.losses}</td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-white/70">{g.shutouts}</td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-[var(--brand-hex)]">
                    {g.savePctg.toFixed(3)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-white/70">
                    {g.goalsAgainstAverage.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {goalies.length === 0 && (
            <p className="px-3 py-6 text-[12px] text-[var(--text-secondary)]">No goaltenders on this roster.</p>
          )}
        </div>
      </HudPanel>
    </main>
  );
}
