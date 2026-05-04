"use client";

import React, { useEffect, useState, useCallback, useRef, Suspense } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { TEAM_COLORS } from "@/utils/nhl";
import TeamLogoLink from "@/components/TeamLogoLink";

interface StandingRow {
  team: string;
  team_name: string;
  division: string;
  conference: string;
  gp: number;
  w: number;
  l: number;
  otl: number;
  pts: number;
  div_rank: number;
  wildcard_rank: number;
  conf_rank: number;
  league_rank: number;
  pts_pct: number;
  streak_code: string;
  streak_count: number;
  gf: number;
  ga: number;
  goal_diff: number;
  home_w: number;
  home_l: number;
  home_otl: number;
  away_w: number;
  away_l: number;
  away_otl: number;
  l10_w: number;
  l10_l: number;
  l10_otl: number;
  so_w: number;
  so_l: number;
  clinch_indicator: string;
  regulation_wins: number;
}

type SortKey = "pts" | "gp" | "w" | "l" | "otl" | "pts_pct" | "gf" | "ga" | "goal_diff" | "regulation_wins";
type TabFilter = "League" | "East" | "West" | "Atlantic" | "Metropolitan" | "Central" | "Pacific";
type ViewMode = "tree" | "hunt" | "table";

interface SeriesSeed {
  team: string;
  team_id: number;
  wins: number;
}
interface SeriesEntry {
  letter: string;
  round: number;
  label: string;
  neededToWin: number;
  topSeed:     SeriesSeed;
  bottomSeed:  SeriesSeed;
  winningTeamId?: number | null;
  losingTeamId?:  number | null;
}
interface BracketData {
  season: number;
  currentRound: number;
  rounds: { round: number; label: string; abbrev: string; series: SeriesEntry[] }[];
}

const DIVISIONS = ["Atlantic", "Metropolitan", "Central", "Pacific"];
const EAST_DIVS = ["Atlantic", "Metropolitan"];
const WEST_DIVS = ["Central", "Pacific"];

function playoffStatus(row: StandingRow): "division" | "wildcard" | "bubble" | "out" {
  if (row.div_rank <= 3) return "division";
  if (row.wildcard_rank > 0 && row.wildcard_rank <= 2) return "wildcard";
  if (row.wildcard_rank > 0 && row.wildcard_rank <= 4) return "bubble";
  return "out";
}

function TeamLogoImg({ abbrev, size = 20, smSize }: { abbrev: string; size?: number; smSize?: number; linkToTeam?: boolean }) {
  return <TeamLogoLink abbrev={abbrev} size={size} smSize={smSize} />;
}

function ClinchBadge({ indicator }: { indicator: string }) {
  if (!indicator) return null;
  // NHL clinch codes — each distinct tier gets its own badge. Earlier code
  // collapsed "p" → "z" which made two teams read as Presidents' Trophy
  // winners; there's only ever one (best in league), while "z" is the
  // separate best-in-conference tag that the other conference leader gets.
  //   x = clinched playoff berth
  //   y = clinched division
  //   z = clinched home-ice / best in conference
  //   p = Presidents' Trophy (best regular-season record in the NHL — unique)
  //   e = eliminated
  const colors: Record<string, string> = {
    x: "bg-[#4ade80]/20 text-[#4ade80] border-[#4ade80]/30",
    y: "bg-[#f472b6]/15 text-[#f472b6]/90 border-[#f472b6]/25",
    z: "bg-[#38bdf8]/20 text-[#38bdf8] border-[#38bdf8]/30",
    p: "bg-[#a78bfa]/20 text-[#a78bfa] border-[#a78bfa]/30",
    e: "bg-white/10 text-white/40 border-white/15",
  };
  return (
    <span className={`text-[8px] font-black uppercase px-1 py-0.5 rounded border ${colors[indicator] ?? "bg-white/10 text-white/40 border-white/10"}`}>
      {indicator}
    </span>
  );
}

function WcBadge({ rank }: { rank: number }) {
  if (rank < 1 || rank > 2) return null;
  return (
    <span className={`text-[8px] font-black uppercase px-1 py-0.5 rounded border ${
      rank === 1
        ? "bg-[#4ade80]/20 text-[#4ade80] border-[#4ade80]/30"
        : "bg-[#38bdf8]/15 text-[#38bdf8]/90 border-[#38bdf8]/25"
    }`}>
      WC{rank}
    </span>
  );
}

function PlayoffDot({ status }: { status: ReturnType<typeof playoffStatus> }) {
  if (status === "division") return <span className="w-1.5 h-1.5 rounded-full bg-[#4ade80] shadow-[0_0_4px_rgba(74,222,128,0.8)] shrink-0" />;
  if (status === "wildcard") return <span className="w-1.5 h-1.5 rounded-full bg-[#fbbf24] shadow-[0_0_4px_rgba(251,191,36,0.8)] shrink-0" />;
  return <span className="w-1.5 h-1.5 rounded-full border border-white/[0.18] shrink-0" title="Not in a playoff spot" />;
}

function SortHeader({ label, col, sortKey, sortDir, onSort, hideMobile }: {
  label: string; col: SortKey; sortKey: SortKey; sortDir: "asc" | "desc"; onSort: (c: SortKey) => void; hideMobile?: boolean;
}) {
  const active = sortKey === col;
  return (
    <th
      className={`px-2 py-2.5 text-right cursor-pointer select-none transition-colors whitespace-nowrap ${hideMobile ? "hidden sm:table-cell" : ""} ${active ? "text-white/70" : "text-white/28 hover:text-white/50"}`}
      onClick={() => onSort(col)}
    >
      <span className="text-[9px] font-black uppercase tracking-wider">{label}</span>
      {active && <span className="ml-0.5 text-[8px]">{sortDir === "desc" ? "↓" : "↑"}</span>}
    </th>
  );
}

// ── Playoff Race View ─────────────────────────────────────────────────────────

function PlayoffRaceConference({ name, rows }: { name: string; rows: StandingRow[] }) {
  const conf = name === "East" ? EAST_DIVS : WEST_DIVS;
  const confRows = rows
    .filter(r => conf.includes(r.division))
    .sort((a, b) => a.conf_rank - b.conf_rank);

  const divTeams = (div: string) => confRows
    .filter(r => r.division === div && r.div_rank <= 3)
    .sort((a, b) => a.div_rank - b.div_rank);

  const wcTeams  = confRows.filter(r => r.wildcard_rank > 0 && r.wildcard_rank <= 2).sort((a, b) => a.wildcard_rank - b.wildcard_rank);
  const huntRows = confRows.filter(r => r.conf_rank > 8 && r.conf_rank <= 13);
  const wc2pts   = wcTeams.find(r => r.wildcard_rank === 2)?.pts ?? 0;

  function badge(row: StandingRow) {
    if (row.div_rank === 1) return { label: `1st ${row.division.slice(0, 3).toUpperCase()}`, cls: "text-white/50 border-white/10 bg-white/[0.03]" };
    if (row.div_rank === 2) return { label: `2nd ${row.division.slice(0, 3).toUpperCase()}`, cls: "text-white/35 border-white/10 bg-white/[0.02]" };
    if (row.div_rank === 3) return { label: `3rd ${row.division.slice(0, 3).toUpperCase()}`, cls: "text-white/35 border-white/10 bg-white/[0.02]" };
    return null;
  }

  function TeamRow({ row, idx }: { row: StandingRow; idx: number }) {
    const color = TEAM_COLORS[row.team] ?? "#666";
    const b = badge(row);
    const streakColor = row.streak_code === "W" ? "text-[#4ade80]" : row.streak_code === "L" ? "text-[#ef4444]/70" : "text-white/35";
    return (
      <div className="flex items-center gap-2.5 px-3 py-2 border-b border-white/[0.04] hover:bg-white/[0.03] transition-colors">
        <span className="text-[11px] font-black text-white/20 w-4 shrink-0 tabular-nums">{idx + 1}</span>
        <div className="w-1 h-5 rounded-full shrink-0" style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}99, 0 0 2px ${color}` }} />
        <TeamLogoImg abbrev={row.team} size={30} smSize={38} linkToTeam />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="text-[14px] font-bold text-white/85">{row.team}</span>
            <span className="text-[11px] text-white/30 hidden sm:inline truncate">{row.team_name}</span>
            {row.clinch_indicator && <ClinchBadge indicator={row.clinch_indicator} />}
          </div>
          <div className="text-[10px] text-white/35 font-mono">{row.w}-{row.l}-{row.otl}</div>
        </div>
        {b && <span className={`hidden sm:inline text-[8px] font-black uppercase px-1.5 py-0.5 rounded border whitespace-nowrap ${b.cls}`}>{b.label}</span>}
        <span className={`text-[11px] font-bold font-mono hidden sm:inline ${streakColor}`}>{row.streak_code}{row.streak_count}</span>
        <div className="text-right">
          <div className="text-[9px] text-white/20 font-mono">{row.gp}gp</div>
          <div className="text-[9px] text-white/20">{82 - row.gp} left</div>
        </div>
        <span className="text-[15px] font-black text-white tabular-nums w-8 text-right">{row.pts}</span>
      </div>
    );
  }

  return (
    <div className="flex-1 min-w-0">
      <div className="flex items-center justify-between mb-3 px-1">
        <h2 className="text-[11px] font-black uppercase tracking-[0.25em] text-white/50">{name}ern</h2>
        <span className="text-[9px] text-white/20 font-semibold">PLAYOFF PICTURE</span>
      </div>

      <div className="rounded-xl border border-[#C9A84C]/[0.15] bg-gradient-to-b from-white/[0.015] via-white/[0.005] to-transparent overflow-hidden shadow-[0_8px_32px_rgba(0,0,0,0.7),0_2px_8px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(220,228,240,0.05),inset_0_-1px_0_rgba(0,0,0,0.35),0_0_0_1px_rgba(190,198,205,0.04)]">
        {/* Division teams — 3 per division, labelled */}
        {conf.map((div, dIdx) => {
          const teams = divTeams(div);
          return (
            <div key={div}>
              <div className={`px-3 py-1 ${dIdx > 0 ? "border-t border-white/[0.07]" : ""}`}>
                <span className="text-[8px] font-black uppercase tracking-[0.25em] text-white/30">{div}</span>
              </div>
              {teams.map((row, i) => <TeamRow key={row.team} row={row} idx={i} />)}
            </div>
          );
        })}

        {/* Main separator — after 6 division spots */}
        <div className="border-t-2 border-[#C9A84C]/25 mx-0" />

        {/* Wild Cards */}
        <div className="px-3 py-1.5 bg-[#fbbf24]/[0.04]">
          <span className="text-[8px] font-black uppercase tracking-[0.25em] text-[#fbbf24]/60">Wild Cards</span>
        </div>
        {wcTeams.map((row) => {
          const color = TEAM_COLORS[row.team] ?? "#666";
          const streakColor = row.streak_code === "W" ? "text-[#4ade80]" : row.streak_code === "L" ? "text-[#ef4444]/70" : "text-white/35";
          return (
            <div key={row.team} className="flex items-center gap-2.5 px-3 py-2 border-b border-white/[0.04] hover:bg-white/[0.03] transition-colors">
              <span className={`text-[8px] font-black uppercase px-1.5 py-0.5 rounded border ${row.wildcard_rank === 1 ? "text-[#C9A84C]/90 border-[#C9A84C]/30 bg-[#C9A84C]/10" : "text-[#38bdf8]/80 border-[#38bdf8]/25 bg-[#38bdf8]/8"}`}>WC{row.wildcard_rank}</span>
              <div className="w-1 h-5 rounded-full shrink-0" style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}99, 0 0 2px ${color}` }} />
              <TeamLogoImg abbrev={row.team} size={30} smSize={38} linkToTeam />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-[14px] font-bold text-white/85">{row.team}</span>
                  <span className="text-[11px] text-white/30 hidden sm:inline">{row.team_name}</span>
                </div>
                <div className="text-[10px] text-white/35 font-mono">{row.w}-{row.l}-{row.otl}</div>
              </div>
              <span className={`text-[11px] font-bold font-mono hidden sm:inline ${streakColor}`}>{row.streak_code}{row.streak_count}</span>
              <div className="text-right">
                <div className="text-[9px] text-white/20 font-mono">{row.gp}gp</div>
                <div className="text-[9px] text-white/20">{82 - row.gp} left</div>
              </div>
              <span className="text-[15px] font-black text-white tabular-nums w-8 text-right">{row.pts}</span>
            </div>
          );
        })}

        {/* In the Hunt section */}
        {huntRows.length > 0 && (
          <>
            <div className="px-3 py-1.5 bg-[#fbbf24]/[0.03] border-t border-[#fbbf24]/15">
              <span className="text-[8px] font-black uppercase tracking-[0.25em] text-[#fbbf24]/40">In the Hunt</span>
            </div>
            {huntRows.map(row => {
              const color = TEAM_COLORS[row.team] ?? "#666";
              const ptsBehind = wc2pts - row.pts;
              const gamesLeft = 82 - row.gp;
              return (
                <div key={row.team} className="flex items-center gap-2.5 px-3 py-2 border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors opacity-70">
                  <span className="text-[11px] font-black text-white/15 w-4 shrink-0">—</span>
                  <div className="w-1 h-5 rounded-full shrink-0" style={{ backgroundColor: color, opacity: 0.5, boxShadow: `0 0 4px ${color}80` }} />
                  <TeamLogoImg abbrev={row.team} size={30} smSize={38} linkToTeam />
                  <div className="flex-1 min-w-0">
                    <div className="text-[14px] font-bold text-white/60">{row.team}</div>
                    <div className="text-[10px] text-white/38 font-mono">{row.w}-{row.l}-{row.otl}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-[10px] font-bold text-[#fbbf24]/60">{ptsBehind > 0 ? `${ptsBehind} back` : "—"}</div>
                    <div className="text-[9px] text-white/20">{gamesLeft} left</div>
                  </div>
                  <span className="text-[13px] font-black text-white/40 tabular-nums w-8 text-right">{row.pts}</span>
                </div>
              );
            })}
          </>
        )}
      </div>
    </div>
  );
}

// ── Playoff Tree / Bracket ─────────────────────────────────────────────────────

function PlayoffBracket({ rows, bracket }: { rows: StandingRow[]; bracket: BracketData | null }) {

  // Mobile-only round pill state. Null = follow bracket.currentRound automatically; once the
  // user picks a round we honor that pick. The pill is hidden on desktop where the full
  // bracket is always visible side-by-side.
  const [mobileRound, setMobileRound] = useState<number | null>(null);
  const activeRound = mobileRound ?? bracket?.currentRound ?? 1;

  // Build a lookup keyed on unordered team-pair so any matchup can find its series record
  const seriesByPair = new Map<string, SeriesEntry>();
  (bracket?.rounds ?? []).forEach(round => {
    round.series.forEach(s => {
      const a = s.topSeed.team, b = s.bottomSeed.team;
      const key = [a, b].sort().join("|");
      seriesByPair.set(key, s);
    });
  });
  function lookupSeries(a?: string, b?: string): SeriesEntry | null {
    if (!a || !b) return null;
    return seriesByPair.get([a, b].sort().join("|")) ?? null;
  }
  function winsFor(team: string, s: SeriesEntry | null): number | null {
    if (!s) return null;
    if (s.topSeed.team    === team) return s.topSeed.wins;
    if (s.bottomSeed.team === team) return s.bottomSeed.wins;
    return null;
  }
  // Find the round-N series whose two teams are both drawn from the candidate pool.
  // Used to attach the Division Final / Conference Final card to its feeder matchups.
  function findSeriesInRound(round: number, candidates: (string | undefined)[]): SeriesEntry | null {
    const pool = new Set(candidates.filter(Boolean) as string[]);
    if (pool.size === 0) return null;
    const r = (bracket?.rounds ?? []).find(x => x.round === round);
    if (!r) return null;
    return r.series.find(s => pool.has(s.topSeed.team) && pool.has(s.bottomSeed.team)) ?? null;
  }
  function rowFor(team?: string): StandingRow | undefined {
    return team ? rows.find(r => r.team === team) : undefined;
  }

  function SeedBadge({ seed }: { seed: number | string }) {
    return (
      <span className={`text-[8px] font-black w-7 h-[18px] flex items-center justify-center rounded shrink-0 border tabular-nums ${
        seed === "WC1"
          ? "bg-[#C9A84C]/10 text-[#C9A84C]/75 border-[#C9A84C]/22"
          : seed === "WC2"
            ? "bg-[#38bdf8]/10 text-[#38bdf8]/75 border-[#38bdf8]/22"
          : seed === "D1"
            ? "bg-[#fbbf24]/10 text-[#fbbf24]/75 border-[#fbbf24]/20"
            : seed === "D2"
              ? "bg-[#C0C0C0]/10 text-[#C0C0C0]/75 border-[#C0C0C0]/20"
              : seed === "D3"
                ? "bg-[#CD7F32]/10 text-[#CD7F32]/75 border-[#CD7F32]/20"
                : "bg-white/[0.04] text-white/38 border-white/[0.08]"
      }`}>{seed}</span>
    );
  }

  function BracketTeamRow({ row, seed, divider, wins, neededToWin, eliminated, clinched }: {
    row: StandingRow | undefined;
    seed?: number | string;
    divider?: boolean;
    wins?: number | null;
    neededToWin?: number;
    eliminated?: boolean;
    clinched?: boolean;
  }) {
    if (!row) return (
      <div className={`flex items-center gap-2 px-3 py-3 min-h-[52px] ${divider ? "border-b border-[#C9A84C]/[0.15]" : ""}`}>
        {seed !== undefined && seed !== "" && <SeedBadge seed={seed} />}
        <div className="w-0.5 h-6 rounded-full bg-white/[0.07] shrink-0" />
        <span className="text-[11px] text-white/18 italic flex-1">TBD</span>
        <span className="text-[13px] font-black text-white/15">—</span>
      </div>
    );
    const color = TEAM_COLORS[row.team] ?? "#666";
    const hasSeries = typeof wins === "number";
    return (
      <div className={`flex items-center gap-2.5 px-3 py-2.5 min-h-[52px] ${divider ? "border-b border-[#C9A84C]/[0.15]" : ""} ${eliminated ? "opacity-50" : ""}`}>
        {seed !== undefined && seed !== "" && <SeedBadge seed={seed} />}
        <div className="w-0.5 h-7 rounded-full shrink-0" style={{ backgroundColor: color, boxShadow: `0 0 8px ${color}bb` }} />
        <TeamLogoImg abbrev={row.team} size={32} smSize={42} linkToTeam />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="text-[13px] font-bold text-white/90 leading-none">{row.team}</span>
            {clinched && <span className="text-[7px] font-black uppercase px-1 py-0.5 rounded border border-[#4ade80]/30 bg-[#4ade80]/10 text-[#4ade80]">Won</span>}
            {eliminated && <span className="text-[7px] font-black uppercase px-1 py-0.5 rounded border border-white/15 bg-white/[0.04] text-white/35">Out</span>}
          </div>
          <div className="text-[9px] text-white/30 font-mono mt-0.5">{row.w}-{row.l}-{row.otl}</div>
        </div>
        <div className="text-right shrink-0">
          {hasSeries ? (
            <>
              <div className={`text-[22px] font-black tabular-nums leading-none ${clinched ? "text-[#fbbf24]" : "text-white"}`}>{wins}</div>
              <div className="text-[7px] text-white/25 font-mono mt-0.5">of {neededToWin ?? 4}</div>
            </>
          ) : (
            <>
              <div className="text-[14px] font-black text-white tabular-nums">{row.pts}</div>
              <div className="text-[7px] text-white/15 font-mono">pts</div>
            </>
          )}
        </div>
      </div>
    );
  }

  function MatchupCard({ top, topSeed, bottom, bottomSeed }: {
    top: StandingRow | undefined; topSeed?: number | string;
    bottom: StandingRow | undefined; bottomSeed?: number | string;
  }) {
    const series      = lookupSeries(top?.team, bottom?.team);
    const topWins     = winsFor(top?.team ?? "", series);
    const bottomWins  = winsFor(bottom?.team ?? "", series);
    const needed      = series?.neededToWin;
    const topClinched = series?.winningTeamId != null && top?.team    === series.topSeed.team    && series.winningTeamId === series.topSeed.team_id;
    const botClinched = series?.winningTeamId != null && bottom?.team === series.bottomSeed.team && series.winningTeamId === series.bottomSeed.team_id;
    const topOut      = series?.losingTeamId  != null && top?.team    === series.topSeed.team    && series.losingTeamId  === series.topSeed.team_id;
    const botOut      = series?.losingTeamId  != null && bottom?.team === series.bottomSeed.team && series.losingTeamId  === series.bottomSeed.team_id;

    // Status line for middle divider — "2-1 FLA leads" or "TIED 2-2" or "FLA wins 4-2"
    let statusText: string | null = null;
    if (series && top && bottom && typeof topWins === "number" && typeof bottomWins === "number") {
      if (series.winningTeamId != null) {
        const winner = series.winningTeamId === series.topSeed.team_id ? top.team : bottom.team;
        statusText = `${winner} wins ${Math.max(topWins, bottomWins)}–${Math.min(topWins, bottomWins)}`;
      } else if (topWins === bottomWins) {
        statusText = topWins === 0 ? "Series tied" : `Tied ${topWins}–${bottomWins}`;
      } else {
        const leader = topWins > bottomWins ? top.team : bottom.team;
        statusText = `${leader} leads ${Math.max(topWins, bottomWins)}–${Math.min(topWins, bottomWins)}`;
      }
    }

    return (
      <div className="rounded-xl border border-[#C9A84C]/[0.18] bg-gradient-to-b from-white/[0.018] via-white/[0.006] to-transparent overflow-hidden shadow-[0_8px_28px_rgba(0,0,0,0.72),inset_0_1px_0_rgba(220,228,240,0.07)]">
        <BracketTeamRow row={top} seed={topSeed} divider wins={topWins} neededToWin={needed} clinched={topClinched} eliminated={topOut} />
        <div className="flex items-center gap-2 px-3 py-1">
          <div className="flex-1 h-px bg-white/[0.04]" />
          {statusText ? (
            <span className="text-[9px] font-black tracking-[0.18em] text-[#C9A84C]/70 uppercase whitespace-nowrap">{statusText}</span>
          ) : (
            <span className="text-[7px] font-black tracking-[0.3em] text-white/10 uppercase">vs</span>
          )}
          <div className="flex-1 h-px bg-white/[0.04]" />
        </div>
        <BracketTeamRow row={bottom} seed={bottomSeed} wins={bottomWins} neededToWin={needed} clinched={botClinched} eliminated={botOut} />
      </div>
    );
  }

  // Two matchup cards stacked with bracket connector lines feeding into the
  // Round 2 (Division Final) card. `mirror` flips the desktop layout so R2 sits
  // on the inner side of the page — used for the right-hand column of each
  // conference so both halves of the bracket point inward toward the Conf Final.
  // Mobile shows only the selected round; desktop always renders both alongside the arm.
  function BracketGroup({ top, bottom, label, divFinal, mirror }: {
    top: React.ReactNode;
    bottom: React.ReactNode;
    label: string;
    divFinal?: React.ReactNode;
    mirror?: boolean;
  }) {
    const showR1Mobile = activeRound === 1;
    const showR2Mobile = activeRound === 2 && !!divFinal;
    return (
      <div>
        <div className={`text-[8px] font-black uppercase tracking-[0.28em] text-white/22 mb-2 px-0.5 ${mirror ? "sm:text-right" : ""}`}>{label}</div>
        <div className={`flex flex-col sm:items-stretch ${mirror ? "sm:flex-row-reverse" : "sm:flex-row"}`}>
          {/* Round 1 cards */}
          <div className={`flex-1 flex-col gap-3 min-w-0 sm:flex ${showR1Mobile ? "flex" : "hidden"}`}>
            {top}
            {bottom}
          </div>
          {/* Bracket arm — desktop only. In mirror mode the vertical bar moves to
              the left edge so the arm visually flows from R1 (outer) to R2 (inner). */}
          <div className={`hidden sm:block w-5 shrink-0 relative self-stretch ${mirror ? "mr-2" : "ml-2"}`}>
            <div className="absolute left-0 right-0 h-px bg-white/[0.16]" style={{ top: "24%" }} />
            <div className="absolute left-0 right-0 h-px bg-white/[0.16]" style={{ top: "76%" }} />
            <div className={`absolute w-px bg-white/[0.16] ${mirror ? "left-0" : "right-0"}`} style={{ top: "24%", bottom: "24%" }} />
            {!divFinal && (
              <div
                className={`absolute w-2 h-2 rounded-full bg-white/[0.22] -translate-y-1/2 ${mirror ? "left-0 -ml-[3px]" : "right-0 -mr-[3px]"}`}
                style={{ top: "50%" }}
              />
            )}
          </div>
          {/* Round 2 (Division Final) slot */}
          {divFinal && (
            <div className={`min-w-0 sm:flex-1 sm:flex sm:items-center sm:mt-0 ${mirror ? "sm:mr-2" : "sm:ml-2"} ${showR2Mobile ? "flex items-center" : "hidden sm:flex"}`}>
              <div className="w-full">{divFinal}</div>
            </div>
          )}
        </div>
      </div>
    );
  }

  function ConferenceBracket({ name, flipped }: { name: "East" | "West"; flipped?: boolean }) {
    const divNames = name === "East" ? EAST_DIVS : WEST_DIVS;
    const confRows = rows.filter(r => divNames.includes(r.division));
    const inRows = confRows.filter(r => r.conf_rank <= 8);

    const seed1 = inRows.find(r => r.conf_rank === 1);
    const leaderDiv = seed1?.division ?? divNames[0];
    const otherDiv  = divNames.find(d => d !== leaderDiv) ?? divNames[1];

    const divTeams = (div: string) => inRows.filter(r => r.division === div).sort((a, b) => a.div_rank - b.div_rank);
    const leaderTeams = divTeams(leaderDiv);
    const otherTeams  = divTeams(otherDiv);
    const wc1 = inRows.find(r => r.wildcard_rank === 1);
    const wc2 = inRows.find(r => r.wildcard_rank === 2);

    // Round 2 (Division Final) — find the round-2 series whose teams come from this division's
    // Round 1 pool. Falls back to null until the NHL API publishes the matchup.
    const leaderDivFinal = findSeriesInRound(2, [leaderTeams[0]?.team, wc2?.team, leaderTeams[1]?.team, leaderTeams[2]?.team]);
    const otherDivFinal  = findSeriesInRound(2, [otherTeams[0]?.team,  wc1?.team, otherTeams[1]?.team,  otherTeams[2]?.team]);

    // Round 3 (Conference Final) — pulls from any team that could plausibly come out of either division.
    const confFinal = findSeriesInRound(3, [
      leaderTeams[0]?.team, wc2?.team, leaderTeams[1]?.team, leaderTeams[2]?.team,
      otherTeams[0]?.team,  wc1?.team, otherTeams[1]?.team,  otherTeams[2]?.team,
    ]);

    const roundLabel = activeRound >= 4 ? "Stanley Cup Final"
                     : activeRound === 3 ? "Conference Final"
                     : activeRound === 2 ? "Round 2"
                     : "Round 1";

    const header = (
      <div className="flex items-center gap-3 mb-5">
        <div className="w-0.5 h-4 rounded-full bg-white/25" />
        <h2 className="text-[13px] font-black uppercase tracking-[0.22em] text-white/55">{name}ern Conference</h2>
        <div className="flex-1 h-px bg-white/[0.06]" />
        <span className="hidden sm:inline-block text-[7px] font-black uppercase tracking-[0.22em] text-white/18 border border-white/[0.07] px-2 py-0.5 rounded-full">{roundLabel}</span>
      </div>
    );

    const tree = (
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        <BracketGroup
          label={leaderDiv}
          top={<MatchupCard top={leaderTeams[0]} topSeed={`D${leaderTeams[0]?.div_rank ?? 1}`} bottom={wc2} bottomSeed="WC2" />}
          bottom={<MatchupCard top={leaderTeams[1]} topSeed={`D${leaderTeams[1]?.div_rank ?? 2}`} bottom={leaderTeams[2]} bottomSeed={`D${leaderTeams[2]?.div_rank ?? 3}`} />}
          divFinal={leaderDivFinal && (
            <MatchupCard top={rowFor(leaderDivFinal.topSeed.team)} bottom={rowFor(leaderDivFinal.bottomSeed.team)} />
          )}
        />
        <BracketGroup
          label={otherDiv}
          mirror
          top={<MatchupCard top={otherTeams[0]} topSeed={`D${otherTeams[0]?.div_rank ?? 1}`} bottom={wc1} bottomSeed="WC1" />}
          bottom={<MatchupCard top={otherTeams[1]} topSeed={`D${otherTeams[1]?.div_rank ?? 2}`} bottom={otherTeams[2]} bottomSeed={`D${otherTeams[2]?.div_rank ?? 3}`} />}
          divFinal={otherDivFinal && (
            <MatchupCard top={rowFor(otherDivFinal.topSeed.team)} bottom={rowFor(otherDivFinal.bottomSeed.team)} />
          )}
        />
      </div>
    );

    const confFinalCard = (
      <div>
        {confFinal ? (
          <div className="max-w-xl mx-auto">
            <div className="text-[8px] font-black uppercase tracking-[0.28em] text-[#C9A84C]/55 mb-2 text-center">{name}ern Conference Final</div>
            <MatchupCard top={rowFor(confFinal.topSeed.team)} bottom={rowFor(confFinal.bottomSeed.team)} />
          </div>
        ) : (
          <div className="flex items-center gap-3">
            <div className="flex-1 h-px bg-white/[0.05]" />
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-[#C9A84C]/[0.15] bg-white/[0.015]">
              <span className="text-[7px] font-black uppercase tracking-[0.25em] text-white/18">Conf Final</span>
              <span className="text-white/10 text-[10px]">·</span>
              <span className="text-[8px] text-white/12 italic">Winner advances to SCF</span>
            </div>
            <div className="flex-1 h-px bg-white/[0.05]" />
          </div>
        )}
      </div>
    );

    // East renders header → tree → CF (CF sits below, flowing into the centre Cup).
    // West renders CF → header → tree (CF sits above the header, mirrored
    // across the centre Cup so the bracket is symmetric: East climbs down,
    // Cup, West climbs back up). Same content either way.
    return (
      <div>
        {flipped ? (
          <>
            <div className="mb-5">{confFinalCard}</div>
            {header}
            {tree}
          </>
        ) : (
          <>
            {header}
            {tree}
            <div className="mt-5">{confFinalCard}</div>
          </>
        )}
      </div>
    );
  }

  // Stanley Cup centerpiece — sits between East CF and West CF so the bracket
  // reads top-to-bottom: East R1 → East R2 → East CF → ⌘ Cup ⌘ → West CF → West R2 → West R1.
  function StanleyCupCenter() {
    return (
      <div className="flex items-center justify-center pt-1 pb-1">
        <div className="rounded-2xl border border-[#fbbf24]/18 bg-[#fbbf24]/[0.025] px-6 py-4 text-center shadow-[0_0_40px_rgba(251,191,36,0.06)]">
          <div className="flex justify-center mb-2">
            <svg width="60" height="82" viewBox="0 0 60 82" fill="none" xmlns="http://www.w3.org/2000/svg">
              <ellipse cx="30" cy="80" rx="18" ry="2" fill="#cbd5e1" opacity="0.18"/>
              <rect x="4"  y="74" width="52" height="5"   rx="1.5" fill="#e2e8f0" opacity="0.65"/>
              <rect x="8"  y="70" width="44" height="4.5" rx="1"   fill="#d1d5db" opacity="0.55"/>
              <rect x="4"  y="74" width="52" height="1.5" rx="0.5" fill="white"   opacity="0.18"/>
              <rect x="9"  y="62" width="42" height="8.5" rx="0.5" fill="#cbd5e1" opacity="0.52"/>
              <rect x="9"  y="53" width="42" height="8.5" rx="0.5" fill="#cbd5e1" opacity="0.47"/>
              <rect x="9"  y="44" width="42" height="8.5" rx="0.5" fill="#cbd5e1" opacity="0.42"/>
              <rect x="10" y="35" width="40" height="8.5" rx="0.5" fill="#cbd5e1" opacity="0.38"/>
              <rect x="11" y="26" width="38" height="8.5" rx="0.5" fill="#cbd5e1" opacity="0.33"/>
              <line x1="9"  y1="62" x2="51" y2="62" stroke="white" strokeWidth="0.8" opacity="0.30"/>
              <line x1="9"  y1="53" x2="51" y2="53" stroke="white" strokeWidth="0.8" opacity="0.26"/>
              <line x1="9"  y1="44" x2="51" y2="44" stroke="white" strokeWidth="0.8" opacity="0.22"/>
              <line x1="10" y1="35" x2="50" y2="35" stroke="white" strokeWidth="0.8" opacity="0.20"/>
              <rect x="9"  y="62" width="42" height="2.5" rx="0.3" fill="white" opacity="0.12"/>
              <rect x="9"  y="53" width="42" height="2.5" rx="0.3" fill="white" opacity="0.10"/>
              <rect x="9"  y="44" width="42" height="2.5" rx="0.3" fill="white" opacity="0.09"/>
              <rect x="10" y="35" width="40" height="2.5" rx="0.3" fill="white" opacity="0.08"/>
              <rect x="11" y="26" width="38" height="2.5" rx="0.3" fill="white" opacity="0.07"/>
              <rect x="22" y="18" width="16" height="9"   rx="1"   fill="#cbd5e1" opacity="0.55"/>
              <line x1="22" y1="22.5" x2="38" y2="22.5"  stroke="white" strokeWidth="0.6" opacity="0.22"/>
              <path d="M13 5 C11 5 9 18 30 18 C51 18 49 5 47 5 Z" fill="#cbd5e1" opacity="0.42"/>
              <path d="M18 7 C16 7 15 17 30 17 C45 17 44 7 42 7 Z" fill="white"   opacity="0.08"/>
              <rect x="11" y="2" width="38" height="5"   rx="2"   fill="#e2e8f0" opacity="0.78"/>
              <rect x="14" y="2.5" width="32" height="2" rx="0.8" fill="white"   opacity="0.22"/>
            </svg>
          </div>
          <div className="text-[7px] font-black uppercase tracking-[0.35em] text-[#fbbf24]/50 mb-1">Stanley Cup Final</div>
          <div className="text-[11px] font-bold text-white/25 italic">East Champion vs West Champion</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-10">
      {/* Mobile-only round pill — desktop renders all rounds side by side */}
      <div className="sm:hidden flex justify-center">
        <div className="flex rounded-lg border border-white/[0.10] overflow-hidden bg-white/[0.02]">
          {[1, 2].map(r => (
            <button
              key={r}
              onClick={() => setMobileRound(r)}
              className={`px-4 py-1.5 text-[10px] font-black uppercase tracking-wider transition-colors ${
                activeRound === r
                  ? "bg-[#C9A84C]/12 text-[#C9A84C]/85"
                  : "text-white/35 hover:text-white/55"
              }`}
            >
              Round {r}
            </button>
          ))}
        </div>
      </div>
      <ConferenceBracket name="East" />
      <StanleyCupCenter />
      <ConferenceBracket name="West" flipped />
    </div>
  );
}

// ── Table Rows ─────────────────────────────────────────────────────────────────

function StandingTableRow({ row, rank, showRank }: { row: StandingRow; rank?: number; showRank?: boolean }) {
  const status = playoffStatus(row);
  const color = TEAM_COLORS[row.team] ?? "#666";
  const diffSign = row.goal_diff > 0 ? "+" : "";
  const streakColor = row.streak_code === "W" ? "text-[#4ade80]" : row.streak_code === "L" ? "text-[#ef4444]/70" : "text-white/35";
  const showWc = row.div_rank > 3 && row.wildcard_rank >= 1 && row.wildcard_rank <= 2;

  return (
    <tr className="border-b border-white/[0.04] hover:bg-white/[0.02] transition-colors">
      {showRank && (
        <td className="px-2 py-2 text-[10px] font-mono text-white/30 text-right w-6 tabular-nums">{rank}</td>
      )}
      {/* Team */}
      <td className="px-2 py-1.5">
        <div className="flex items-center gap-1.5">
          <PlayoffDot status={status} />
          <div className="w-1 h-4 rounded-full shrink-0" style={{ backgroundColor: color, boxShadow: `0 0 5px ${color}80` }} />
          <TeamLogoImg abbrev={row.team} size={26} smSize={36} linkToTeam />
          <span className="text-[12px] font-bold text-white/85 whitespace-nowrap">{row.team}</span>
          <span className="text-[10px] text-white/30 hidden sm:inline truncate max-w-[90px]">{row.team_name}</span>
          {row.div_rank <= 3 && (
            <span className={`text-[8px] font-black px-1 py-0.5 rounded border whitespace-nowrap ${
              row.div_rank === 1
                ? "text-[#fbbf24]/75 border-[#fbbf24]/20 bg-[#fbbf24]/10"
                : row.div_rank === 2
                  ? "text-[#C0C0C0]/75 border-[#C0C0C0]/20 bg-[#C0C0C0]/10"
                  : "text-[#CD7F32]/75 border-[#CD7F32]/20 bg-[#CD7F32]/10"
            }`}>
              {row.div_rank === 1 ? "D1" : row.div_rank === 2 ? "D2" : "D3"}
            </span>
          )}
          {showWc && <WcBadge rank={row.wildcard_rank} />}
          {row.clinch_indicator && <ClinchBadge indicator={row.clinch_indicator} />}
        </div>
      </td>
      <td className="hidden sm:table-cell px-2 py-2 text-right text-[12px] font-mono tabular-nums text-white/45">{row.gp}</td>
      <td className="px-1.5 py-1.5 text-right text-[12px] font-mono tabular-nums font-bold text-white/80">{row.w}</td>
      <td className="px-1.5 py-1.5 text-right text-[12px] font-mono tabular-nums text-white/45">{row.l}</td>
      <td className="px-1.5 py-1.5 text-right text-[12px] font-mono tabular-nums text-white/45">{row.otl}</td>
      <td className="px-1.5 py-1.5 text-right text-[13px] font-black tabular-nums text-white">{row.pts}</td>
      <td className="px-1.5 py-1.5 text-right text-[12px] font-mono tabular-nums text-white/40">{82 - row.gp}</td>
      <td className="hidden sm:table-cell px-2 py-2 text-right text-[11px] font-mono tabular-nums text-white/35">{(row.pts_pct * 100).toFixed(1)}%</td>
      <td className="hidden sm:table-cell px-2 py-2 text-right text-[12px] font-mono tabular-nums text-white/40 whitespace-nowrap">{row.l10_w}-{row.l10_l}-{row.l10_otl}</td>
      <td className={`hidden sm:table-cell px-2 py-2 text-right text-[12px] font-mono tabular-nums font-bold whitespace-nowrap ${streakColor}`}>{row.streak_code}{row.streak_count}</td>
      <td className="hidden sm:table-cell px-2 py-2 text-right text-[11px] font-mono tabular-nums text-white/35 whitespace-nowrap">{row.home_w}-{row.home_l}-{row.home_otl}</td>
      <td className="hidden sm:table-cell px-2 py-2 text-right text-[11px] font-mono tabular-nums text-white/35 whitespace-nowrap">{row.away_w}-{row.away_l}-{row.away_otl}</td>
      <td className="hidden sm:table-cell px-2 py-2 text-right text-[12px] font-mono tabular-nums text-white/45">{row.gf}</td>
      <td className="hidden sm:table-cell px-2 py-2 text-right text-[12px] font-mono tabular-nums text-white/45">{row.ga}</td>
      <td className={`hidden sm:table-cell px-2 py-2 text-right text-[12px] font-mono tabular-nums font-semibold ${row.goal_diff > 0 ? "text-[#4ade80]/80" : row.goal_diff < 0 ? "text-[#ef4444]/70" : "text-white/38"}`}>
        {diffSign}{row.goal_diff}
      </td>
      <td className="hidden sm:table-cell px-2 py-2 text-right text-[12px] font-mono tabular-nums text-white/30">{row.regulation_wins}</td>
    </tr>
  );
}

function DivisionSection({ division, rows }: { division: string; rows: StandingRow[] }) {
  const divRows = rows.filter(r => r.division === division).sort((a, b) => a.div_rank - b.div_rank);
  if (divRows.length === 0) return null;

  return (
    <>
      <tr>
        <td colSpan={16} className="px-3 pt-3 pb-1">
          <span className="text-[9px] font-black uppercase tracking-[0.25em] text-white/38">{division}</span>
        </td>
      </tr>
      {divRows.map((row, idx) => {
        const isPlayoffCutoff = row.div_rank === 3 && divRows[idx + 1]?.div_rank === 4;
        return (
          <React.Fragment key={row.team}>
            <StandingTableRow row={row} rank={idx + 1} showRank />
            {isPlayoffCutoff && (
              <tr>
                <td colSpan={16} className="p-0"><div className="border-b border-[#C9A84C]/20 mx-3" /></td>
              </tr>
            )}
          </React.Fragment>
        );
      })}
    </>
  );
}

function LeagueSection({ rows }: { rows: StandingRow[] }) {
  const isInPlayoff = (r: StandingRow) => r.div_rank <= 3 || (r.wildcard_rank >= 1 && r.wildcard_rank <= 2);
  const inPlayoff  = [...rows].filter(r =>  isInPlayoff(r)).sort((a, b) => a.league_rank - b.league_rank);
  const outPlayoff = [...rows].filter(r => !isInPlayoff(r)).sort((a, b) => a.league_rank - b.league_rank);
  const allSorted  = [...inPlayoff, ...outPlayoff];
  const cutoff     = inPlayoff.length;

  return (
    <>
      {allSorted.map((row, idx) => (
        <React.Fragment key={row.team}>
          {idx === cutoff && (
            <tr>
              <td colSpan={16} className="p-0">
                <div className="relative border-b-2 border-[#C9A84C]/25 mx-3">
                  <span className="absolute right-0 -top-2.5 text-[7px] font-black uppercase tracking-[0.2em] text-[#C9A84C]/35 pr-1">Playoff line</span>
                </div>
              </td>
            </tr>
          )}
          <StandingTableRow row={row} rank={idx + 1} showRank />
        </React.Fragment>
      ))}
    </>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────────

function StandingsPageInner() {
  const searchParams = useSearchParams();
  const queryView    = searchParams?.get("view");
  const initialView: ViewMode =
    queryView === "table" || queryView === "hunt" ? queryView : "tree";

  const [standings, setStandings] = useState<StandingRow[]>([]);
  const [bracket,   setBracket]   = useState<BracketData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabFilter>("League");
  const [view, setView] = useState<ViewMode>(initialView);
  const [sortKey, setSortKey] = useState<SortKey>("pts");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const tabScrollRef = useRef<HTMLDivElement>(null);
  const [tabCanLeft, setTabCanLeft] = useState(false);
  const [tabCanRight, setTabCanRight] = useState(false);

  const fetchStandings = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const [sRes, bRes] = await Promise.all([
        fetch("/api/standings"),
        fetch("/api/playoff-bracket?season=20252026"),
      ]);
      const sData = await sRes.json();
      if (sData.error) throw new Error(sData.error);
      setStandings(sData.standings ?? []);
      if (bRes.ok) {
        const bData = await bRes.json();
        if (!bData.error) setBracket(bData as BracketData);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load standings");
    } finally { setLoading(false); }
  }, []);

  useEffect(() => { fetchStandings(); }, [fetchStandings]);

  useEffect(() => {
    const el = tabScrollRef.current;
    if (!el) return;
    const update = () => {
      setTabCanLeft(el.scrollLeft > 0);
      setTabCanRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
    };
    update();
    el.addEventListener("scroll", update, { passive: true });
    return () => el.removeEventListener("scroll", update);
  }, []);

  function handleSort(col: SortKey) {
    if (sortKey === col) setSortDir(d => d === "desc" ? "asc" : "desc");
    else { setSortKey(col); setSortDir("desc"); }
  }

  const TABS: TabFilter[] = ["League", "East", "West", "Atlantic", "Metropolitan", "Central", "Pacific"];
  const filteredRows =
    tab === "League" ? standings :
    tab === "East"   ? standings.filter(r => EAST_DIVS.includes(r.division)) :
    tab === "West"   ? standings.filter(r => WEST_DIVS.includes(r.division)) :
    standings.filter(r => r.division === tab);

  const TableHeader = ({ showRank }: { showRank?: boolean }) => (
    <thead>
      <tr className="border-b border-white/[0.07]">
        {showRank && <th className="px-2 py-2.5 w-6" />}
        <th className="px-2 py-2 text-left text-[9px] font-black uppercase tracking-wider text-white/28">Team</th>
        <SortHeader label="GP"   col="gp"              sortKey={sortKey} sortDir={sortDir} onSort={handleSort} hideMobile />
        <SortHeader label="W"    col="w"               sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
        <SortHeader label="L"    col="l"               sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
        <SortHeader label="OTL"  col="otl"             sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
        <SortHeader label="PTS"  col="pts"             sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
        <th className="px-1.5 py-2.5 text-right text-[9px] font-black uppercase tracking-wider text-white/28 whitespace-nowrap">REM</th>
        <th className="hidden sm:table-cell px-2 py-2.5 text-right text-[9px] font-black uppercase tracking-wider text-white/28">PTS%</th>
        <th className="hidden sm:table-cell px-2 py-2.5 text-right text-[9px] font-black uppercase tracking-wider text-white/28">L10</th>
        <th className="hidden sm:table-cell px-2 py-2.5 text-right text-[9px] font-black uppercase tracking-wider text-white/28">STRK</th>
        <th className="hidden sm:table-cell px-2 py-2.5 text-right text-[9px] font-black uppercase tracking-wider text-white/28">HOME</th>
        <th className="hidden sm:table-cell px-2 py-2.5 text-right text-[9px] font-black uppercase tracking-wider text-white/28">AWAY</th>
        <SortHeader label="GF"   col="gf"              sortKey={sortKey} sortDir={sortDir} onSort={handleSort} hideMobile />
        <SortHeader label="GA"   col="ga"              sortKey={sortKey} sortDir={sortDir} onSort={handleSort} hideMobile />
        <SortHeader label="DIFF" col="goal_diff"       sortKey={sortKey} sortDir={sortDir} onSort={handleSort} hideMobile />
        <SortHeader label="ROW"  col="regulation_wins" sortKey={sortKey} sortDir={sortDir} onSort={handleSort} hideMobile />
      </tr>
    </thead>
  );

  return (
    <main className="min-h-screen flex flex-col px-3 pt-6 pb-12 max-w-[1300px] mx-auto w-full overflow-x-hidden">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div>
          <Link href="/" className="text-[9px] font-black uppercase tracking-[0.3em] text-white/20 hover:text-white/40 transition-colors mb-1.5 block">
            ← GRTZKY
          </Link>
          <h1 className="text-[22px] sm:text-[34px] font-black tracking-[0.08em] uppercase text-white leading-none">Standings</h1>
          <p className="text-[10px] text-white/38 mt-1">2025–26 NHL Season</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex rounded-lg border border-white/[0.07] overflow-hidden">
            <button onClick={() => setView("tree")}  className={`px-2 py-1 text-[9px] font-bold uppercase tracking-wide transition-colors ${view === "tree"  ? "bg-[#C9A84C]/10 text-[#C9A84C]/80" : "text-white/30 hover:text-white/50"}`}>Bracket</button>
            <button onClick={() => setView("hunt")}  className={`px-2 py-1 text-[9px] font-bold uppercase tracking-wide transition-colors border-l border-white/[0.07] ${view === "hunt"  ? "bg-white/[0.08] text-white/80"    : "text-white/30 hover:text-white/50"}`}>Race</button>
            <button onClick={() => setView("table")} className={`px-2 py-1 text-[9px] font-bold uppercase tracking-wide transition-colors border-l border-white/[0.07] ${view === "table" ? "bg-white/[0.08] text-white/80"    : "text-white/30 hover:text-white/50"}`}>Table</button>
          </div>
          <button onClick={fetchStandings} className="text-[13px] font-bold text-white/38 hover:text-white/50 transition-colors px-1.5 py-1">↻</button>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-[#ef4444]/20 bg-[#ef4444]/[0.04] p-4 mb-5 text-[12px] text-[#ef4444]/70">{error}</div>
      )}

      {/* Legend */}
      <div className="flex flex-col gap-2 mb-4">
        {/* Row 1: dot indicators */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#4ade80] shadow-[0_0_4px_rgba(74,222,128,0.8)]" />
            <span className="text-[9px] text-white/38 uppercase tracking-wide">In</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#fbbf24] shadow-[0_0_4px_rgba(251,191,36,0.8)]" />
            <span className="text-[9px] text-white/38 uppercase tracking-wide">Wild Card</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full border border-white/[0.18]" />
            <span className="text-[9px] text-white/38 uppercase tracking-wide">Out</span>
          </div>
        </div>
        {/* Row 2: division seeds */}
        <div className="flex items-center gap-2.5">
          {[
            { badge: "D1", cls: "border-[#fbbf24]/20 bg-[#fbbf24]/10 text-[#fbbf24]/75", label: "Div 1st" },
            { badge: "D2", cls: "border-[#C0C0C0]/20 bg-[#C0C0C0]/10 text-[#C0C0C0]/75", label: "Div 2nd" },
            { badge: "D3", cls: "border-[#CD7F32]/20 bg-[#CD7F32]/10 text-[#CD7F32]/75", label: "Div 3rd" },
          ].map(({ badge, cls, label }) => (
            <div key={badge} className="flex items-center gap-1">
              <span className={`px-1 py-0.5 rounded border text-[8px] font-black ${cls}`}>{badge}</span>
              <span className="text-[9px] text-white/25">{label}</span>
            </div>
          ))}
        </div>
        {/* Row 3: wild card + clinch badges */}
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5">
          {[
            { badge: "WC1", cls: "border-[#C9A84C]/30 bg-[#C9A84C]/10 text-[#C9A84C]",     label: "1st wild card" },
            { badge: "WC2", cls: "border-[#38bdf8]/25 bg-[#38bdf8]/10 text-[#38bdf8]/80",  label: "2nd wild card" },
            { badge: "x",   cls: "border-[#4ade80]/30 bg-[#4ade80]/10 text-[#4ade80]",      label: "clinched" },
            { badge: "y",   cls: "border-[#f472b6]/25 bg-[#f472b6]/15 text-[#f472b6]/90",   label: "div winner" },
            { badge: "z",   cls: "border-[#38bdf8]/30 bg-[#38bdf8]/20 text-[#38bdf8]",      label: "conference" },
            { badge: "p",   cls: "border-[#a78bfa]/30 bg-[#a78bfa]/20 text-[#a78bfa]",      label: "Presidents'" },
            { badge: "E",   cls: "border-white/15 bg-white/[0.04] text-white/35",            label: "eliminated" },
          ].map(({ badge, cls, label }) => (
            <div key={badge} className="flex items-center gap-1">
              <span className={`px-1 py-0.5 rounded border text-[8px] font-black ${cls}`}>{badge}</span>
              <span className="text-[9px] text-white/25">{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Rotate hint — mobile only */}
      <p className="sm:hidden text-[10px] text-white/25 text-center mb-3 flex items-center justify-center gap-1.5">
        <span>↻</span> Rotate phone for more stats
      </p>

      {view === "hunt" ? (
        /* ── Playoff Race ── */
        <div className={`flex flex-col sm:flex-row gap-4 ${loading ? "opacity-50 pointer-events-none" : ""}`}>
          {loading ? (
            <>
              <div className="flex-1 h-96 rounded-xl border border-white/[0.10] bg-white/[0.02] animate-pulse" />
              <div className="flex-1 h-96 rounded-xl border border-white/[0.10] bg-white/[0.02] animate-pulse" />
            </>
          ) : (
            <>
              <PlayoffRaceConference name="East" rows={standings} />
              <div className="hidden sm:block w-px bg-white/[0.06]" />
              <PlayoffRaceConference name="West" rows={standings} />
            </>
          )}
        </div>
      ) : view === "tree" ? (
        /* ── Playoff Bracket ── */
        <div className={loading ? "opacity-50 pointer-events-none" : ""}>
          {loading ? (
            <div className="h-[500px] rounded-xl border border-white/[0.10] bg-white/[0.02] animate-pulse" />
          ) : (
            <PlayoffBracket rows={standings} bracket={bracket} />
          )}
        </div>
      ) : (
        /* ── Table View ── */
        <>
          <div className="relative flex items-center mb-3">
            {tabCanLeft && (
              <button onClick={() => tabScrollRef.current?.scrollBy({ left: -160, behavior: "smooth" })}
                className="absolute left-0 z-10 h-full px-1 flex items-center bg-gradient-to-r from-[#0a0b0d] to-transparent text-white/40 hover:text-white/80 transition-colors">
                ❮
              </button>
            )}
            <div ref={tabScrollRef} className="overflow-x-auto pb-1 flex-1" style={{ scrollbarWidth: "none" }}>
              <div className="flex gap-1 min-w-max">
                {TABS.map(t => (
                  <button key={t} onClick={() => setTab(t)}
                    className={`px-3 py-1.5 rounded-lg text-[10px] font-black uppercase tracking-wide whitespace-nowrap transition-all ${
                      tab === t ? "bg-white/[0.08] text-white/85 border border-white/[0.07]" : "text-white/30 hover:text-white/50 border border-transparent"
                    }`}
                  >{t}</button>
                ))}
              </div>
            </div>
            {tabCanRight && (
              <button onClick={() => tabScrollRef.current?.scrollBy({ left: 160, behavior: "smooth" })}
                className="absolute right-0 z-10 h-full px-1 flex items-center bg-gradient-to-l from-[#0a0b0d] to-transparent text-white/40 hover:text-white/80 transition-colors">
                ❯
              </button>
            )}
          </div>

          <div className="rounded-xl border border-[#C9A84C]/[0.15] bg-gradient-to-b from-white/[0.015] via-white/[0.005] to-transparent overflow-x-auto shadow-[0_8px_32px_rgba(0,0,0,0.7),0_2px_8px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(220,228,240,0.05),inset_0_-1px_0_rgba(0,0,0,0.35),0_0_0_1px_rgba(190,198,205,0.04)]">
            <table className="w-full border-collapse">
              <TableHeader showRank />
              <tbody>
                {loading ? (
                  Array.from({ length: 8 }).map((_, i) => (
                    <tr key={i} className="border-b border-white/[0.04]">
                      {Array.from({ length: 10 }).map((__, j) => (
                        <td key={j} className="px-2 py-2.5">
                          <div className="h-3 rounded bg-white/[0.06] animate-pulse" style={{ width: j === 0 ? 100 : 24 }} />
                        </td>
                      ))}
                    </tr>
                  ))
                ) : tab === "League" ? (
                  <LeagueSection rows={filteredRows} />
                ) : tab === "East" ? (
                  EAST_DIVS.map(div => <DivisionSection key={div} division={div} rows={filteredRows} />)
                ) : tab === "West" ? (
                  WEST_DIVS.map(div => <DivisionSection key={div} division={div} rows={filteredRows} />)
                ) : (
                  DIVISIONS.filter(d => d === tab).map(div => (
                    <DivisionSection key={div} division={div} rows={filteredRows} />
                  ))
                )}
              </tbody>
            </table>
          </div>

          {!loading && standings.length > 0 && (
            <p className="mt-3 text-[9px] text-white/15 text-center">ROW = Regulation + OT Wins</p>
          )}
        </>
      )}
    </main>
  );
}

export default function StandingsPage() {
  return (
    <Suspense fallback={null}>
      <StandingsPageInner />
    </Suspense>
  );
}
