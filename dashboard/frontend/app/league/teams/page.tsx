"use client";

import { useRouter } from "next/navigation";
import { TEAM_COLORS, TEAM_SECONDARY, TEAM_FULL_NAMES, logoUrl } from "@/utils/nhl";

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
}

function TeamCard({ abbrev, onSelect }: TeamCardProps) {
  const primary   = TEAM_COLORS[abbrev]   ?? "#888888";
  const secondary = TEAM_SECONDARY[abbrev] ?? "#333333";
  const fullName  = TEAM_FULL_NAMES[abbrev] ?? abbrev;
  const [r, g, b] = hexToRgb(primary);

  // Split city / nickname for two-line display
  const nameParts = fullName.split(" ");
  const nickname  = nameParts[nameParts.length - 1];
  const city      = nameParts.slice(0, -1).join(" ");

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
    </button>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------
export default function LeagueTeamsPage() {
  const router = useRouter();

  function handleTeamClick(abbrev: string) {
    router.push(`/teams/${abbrev}`);
  }

  return (
    <main className="min-h-screen p-4 sm:p-6 max-w-5xl mx-auto w-full overflow-x-hidden">
      {/* Header */}
      <div className="mb-6 sm:mb-8">
        <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-white/20 mb-1">
          NHL · 2025–26
        </p>
        <h1 className="text-2xl sm:text-3xl font-black uppercase tracking-wider text-white/90">
          League
        </h1>
        <p className="mt-1 text-[11px] text-white/30">
          Select a team to view their roster, cap, and injuries.
        </p>
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
                <TeamCard key={abbrev} abbrev={abbrev} onSelect={handleTeamClick} />
              ))}
            </div>
          </section>
        ))}
      </div>
    </main>
  );
}
