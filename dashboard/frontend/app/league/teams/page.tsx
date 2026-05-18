"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { TEAM_COLORS, TEAM_SECONDARY, TEAM_FULL_NAMES, logoUrl } from "@/utils/nhl";
import { HudGrid } from "@/components/hud";

// ---------------------------------------------------------------------------
// Division structure (2025-26 NHL season)
// ---------------------------------------------------------------------------
const DIVISIONS: { name: string; conference: string; teams: string[] }[] = [
  {
    name: "Atlantic",
    conference: "Eastern",
    teams: ["BOS", "BUF", "DET", "FLA", "MTL", "OTT", "TBL", "TOR"],
  },
  {
    name: "Metropolitan",
    conference: "Eastern",
    teams: ["CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH"],
  },
  {
    name: "Central",
    conference: "Western",
    teams: ["UTA", "CHI", "COL", "DAL", "MIN", "NSH", "STL", "WPG"],
  },
  {
    name: "Pacific",
    conference: "Western",
    teams: ["ANA", "CGY", "EDM", "LAK", "SEA", "SJS", "VAN", "VGK"],
  },
];

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

interface TeamCardProps {
  abbrev: string;
  onSelect: (abbrev: string) => void;
  fatigue?: { mean_fi: number; last_game: string | null } | null;
}

function fiHex(v: number): string {
  if (v >= 0.18) return "#f87171";   // most-fatigued end of league range
  if (v >= 0.14) return "#fb923c";
  if (v >= 0.11) return "#fbbf24";
  return "#4ade80";
}

function TeamCard({ abbrev, onSelect, fatigue }: TeamCardProps) {
  const primary   = TEAM_COLORS[abbrev]   ?? "#888888";
  const secondary = TEAM_SECONDARY[abbrev] ?? "#333333";
  const fullName  = TEAM_FULL_NAMES[abbrev] ?? abbrev;
  const [r, g, b] = hexToRgb(primary);

  // Split city / nickname for two-line display
  const nameParts = fullName.split(" ");
  const nickname  = nameParts[nameParts.length - 1];
  const city      = nameParts.slice(0, -1).join(" ");

  const fiColor = fatigue ? fiHex(fatigue.mean_fi) : "#374151";
  const fiPct   = fatigue ? Math.min(100, Math.max(4, fatigue.mean_fi * 400)) : 0;

  return (
    <button
      onClick={() => onSelect(abbrev)}
      className="group relative flex flex-col items-center gap-3 rounded-2xl border p-4 sm:p-5 transition-all duration-200 hover:scale-[1.03] active:scale-[0.98] cursor-pointer w-full text-left"
      style={{
        background: `linear-gradient(145deg, rgba(${r},${g},${b},0.10) 0%, rgba(0,0,0,0.6) 100%)`,
        borderColor: `rgba(${r},${g},${b},0.22)`,
        boxShadow: `0 0 0 0 rgba(${r},${g},${b},0)`,
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLButtonElement).style.boxShadow = `0 4px 24px rgba(${r},${g},${b},0.25)`;
        (e.currentTarget as HTMLButtonElement).style.borderColor = `rgba(${r},${g},${b},0.45)`;
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLButtonElement).style.boxShadow = `0 0 0 0 rgba(${r},${g},${b},0)`;
        (e.currentTarget as HTMLButtonElement).style.borderColor = `rgba(${r},${g},${b},0.22)`;
      }}
    >
      {/* Color accent bar */}
      <div
        className="absolute top-0 left-0 right-0 h-[2px] rounded-t-2xl"
        style={{ background: `linear-gradient(90deg, ${secondary}00, ${primary}, ${secondary}00)` }}
      />

      {/* Logo */}
      <div
        className="rounded-xl p-2.5 sm:p-3 lg:p-4 flex items-center justify-center"
        style={{
          background: `radial-gradient(circle at 40% 35%, rgba(${r},${g},${b},0.18) 0%, rgba(0,0,0,0.5) 100%)`,
        }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={logoUrl(abbrev)}
          alt={abbrev}
          width={52}
          height={52}
          className="object-contain sm:w-[64px] sm:h-[64px] lg:w-[88px] lg:h-[88px]"
          style={{ filter: `drop-shadow(0 2px 8px rgba(${r},${g},${b},0.45))` }}
        />
      </div>

      {/* Name */}
      <div className="text-center min-w-0">
        <p className="text-[10px] font-medium text-white/40 leading-tight">{city}</p>
        <p className="text-[13px] font-black uppercase tracking-wide leading-tight" style={{ color: primary }}>
          {nickname}
        </p>
        <p className="text-[9px] font-semibold text-white/20 mt-0.5 tracking-widest">{abbrev}</p>
      </div>

      {/* Fatigue indicator — only renders once the API responds */}
      {fatigue && (
        <div className="w-full space-y-1">
          <div className="flex items-center justify-between gap-2 text-[9px] font-mono">
            <span className="text-white/30 uppercase tracking-[0.16em]">FI · 3wk</span>
            <span style={{ color: fiColor }}>{fatigue.mean_fi.toFixed(3)}</span>
          </div>
          <div className="h-1 rounded-full bg-white/[0.05] overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{ width: `${fiPct}%`, backgroundColor: fiColor, opacity: 0.85 }}
            />
          </div>
        </div>
      )}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------
interface TeamFatigueRow { team: string; mean_fi: number; max_fi: number; last_game: string | null; rows: number; }

export default function LeagueTeamsPage() {
  const router = useRouter();
  const [fatigue, setFatigue] = useState<Record<string, TeamFatigueRow> | null>(null);
  const [fatigueMeta, setFatigueMeta] = useState<{ window_start?: string; window_end?: string } | null>(null);

  useEffect(() => {
    fetch("/api/phase3/team-fatigue?window_days=21")
      .then((r) => r.json())
      .then((d) => {
        if (!d || d.status !== "ok") return;
        const map: Record<string, TeamFatigueRow> = {};
        for (const t of d.teams ?? []) map[t.team] = t;
        setFatigue(map);
        setFatigueMeta({ window_start: d.window_start, window_end: d.window_end });
      })
      .catch(() => {});
  }, []);

  function handleTeamClick(abbrev: string) {
    router.push(`/teams/${abbrev}`);
  }

  return (
    <main className="relative min-h-screen p-4 sm:p-6 max-w-5xl mx-auto w-full overflow-x-hidden">
      <HudGrid />

      {/* HUD dossier strip */}
      <div className="relative z-10 mb-3 flex items-center gap-2 flex-wrap">
        <span className="hud-mono text-[10px] uppercase tracking-[0.20em] text-[var(--brand-hex)]" aria-hidden>◢</span>
        <span className="hud-mono text-[10px] uppercase tracking-[0.20em] text-[var(--brand-hex)]">
          LEAGUE INDEX · 32 FRANCHISES
        </span>
        <span className="hud-mono text-[9px] uppercase tracking-[0.16em] text-[var(--text-secondary)]">· 2025-26</span>
        <span className="ml-auto flex items-center gap-1">
          <span className="hud-pulse-dot" style={{ background: "#4ade80" }} />
          <span className="hud-mono jarvis-flicker text-[9px] uppercase tracking-[0.18em] text-[#4ade80]">SYNCED</span>
        </span>
      </div>

      {/* Header */}
      <div className="mb-6 sm:mb-8 relative z-10">
        <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-white/20 mb-1">
          NHL · 2025–26
        </p>
        <h1 className="text-2xl sm:text-3xl font-black uppercase tracking-wider text-white/90">
          League
        </h1>
        <p className="mt-1 text-[11px] text-white/30">
          Select a team to view their roster, cap, and injuries.
        </p>
        {fatigueMeta?.window_end && (
          <p className="mt-2 text-[9px] font-mono text-white/30">
            Cards show <span className="text-white/55">Phase 3 Fatigue Index</span> — mean FI per team across the
            final 3 weeks of the regular season ({fatigueMeta.window_start} → {fatigueMeta.window_end}).
            <span className="ml-2 inline-flex items-center gap-1">
              <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: "#4ade80" }} /> rested
              <span className="ml-3 inline-block w-2 h-2 rounded-full" style={{ backgroundColor: "#fbbf24" }} /> avg
              <span className="ml-3 inline-block w-2 h-2 rounded-full" style={{ backgroundColor: "#fb923c" }} /> tired
              <span className="ml-3 inline-block w-2 h-2 rounded-full" style={{ backgroundColor: "#f87171" }} /> heavy
            </span>
          </p>
        )}
      </div>

      {/* Divisions */}
      <div className="space-y-8">
        {DIVISIONS.map((div) => (
          <section key={div.name}>
            {/* Division header */}
            <div className="flex items-center gap-3 mb-3">
              <div>
                <p className="text-[9px] font-semibold uppercase tracking-[0.22em] text-white/20">
                  {div.conference} Conference
                </p>
                <h2 className="text-[15px] sm:text-[17px] font-black uppercase tracking-wider text-white/80">
                  {div.name} Division
                </h2>
              </div>
              <div className="flex-1 h-px bg-white/[0.06]" />
            </div>

            {/* Team grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {div.teams.map((abbrev) => (
                <TeamCard
                  key={abbrev}
                  abbrev={abbrev}
                  onSelect={handleTeamClick}
                  fatigue={fatigue?.[abbrev] ?? null}
                />
              ))}
            </div>
          </section>
        ))}
      </div>
    </main>
  );
}
