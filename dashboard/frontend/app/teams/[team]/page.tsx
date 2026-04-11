"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { TEAM_COLORS, TEAM_SECONDARY, TEAM_FULL_NAMES, logoUrl } from "@/utils/nhl";
import { useTheme } from "@/utils/themeContext";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TeamStanding {
  team: string;
  team_name: string;
  gp: number;
  w: number;
  l: number;
  otl: number;
  pts: number;
  div_rank: number;
  division: string;
  streak_code: string;
  streak_count: number;
}

interface Skater {
  player_id: number | null;
  headshot: string;
  first_name: string;
  last_name: string;
  position: string;
  jersey: number | null;
  gp: number;
  goals: number;
  assists: number;
  points: number;
  plus_minus: number;
  pim: number;
  pp_points: number;
  gwg: number;
  shots: number;
  shooting_pct: number;
  avg_toi: string;
  faceoff_pct: number;
  injury_status: string | null;
  injury_detail: string | null;
}

interface Goalie {
  player_id: number | null;
  headshot: string;
  first_name: string;
  last_name: string;
  jersey: number | null;
  gp: number;
  wins: number;
  losses: number;
  ot_losses: number;
  gaa: number;
  sv_pct: number;
  shutouts: number;
  injury_status: string | null;
  injury_detail: string | null;
}

interface CapPlayer {
  name: string;
  position: string;
  age: number | null;
  cap_hit: number | null;
  contract_type: string;
  expiry_year: string | number | null;
  years_remaining: string | number | null;
}

interface CapData {
  players: CapPlayer[];
  cap_ceiling: number;
  total_hit: number | null;
  cap_space: number | null;
  status: string;
}

interface InjuryEntry {
  player_name: string;
  position: string;
  status: string;
  status_raw: string;
  injury_type: string;
  injury_detail: string | null;
  return_estimate: string | null;
  reported_at: string | null;
  play_probability: number | null;
}

interface InjuryData {
  injuries: InjuryEntry[];
  as_of: string | null;
}

type SortKey = "points" | "goals" | "assists" | "plus_minus" | "pim" | "pp_points" | "gwg" | "shots" | "shooting_pct" | "avg_toi" | "gp";
type GoalieSortKey = "wins" | "losses" | "ot_losses" | "gaa" | "sv_pct" | "shutouts" | "gp";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Blends a hex color heavily toward the app's dark background (#0d0f13).
// Ensures even white/light team secondary colors produce a dark usable shade.
function darkBlend(hex: string, darkness = 0.82): string {
  const bg = [13, 15, 19]; // #0d0f13
  const clean = hex.replace("#", "");
  const r = parseInt(clean.slice(0, 2), 16) || 0;
  const g = parseInt(clean.slice(2, 4), 16) || 0;
  const b = parseInt(clean.slice(4, 6), 16) || 0;
  const rr = Math.round(r * (1 - darkness) + bg[0] * darkness);
  const gg = Math.round(g * (1 - darkness) + bg[1] * darkness);
  const bb = Math.round(b * (1 - darkness) + bg[2] * darkness);
  return `#${rr.toString(16).padStart(2, "0")}${gg.toString(16).padStart(2, "0")}${bb.toString(16).padStart(2, "0")}`;
}

// Relative luminance (0 = black, 1 = white)
function luminance(hex: string): number {
  const clean = hex.replace("#", "");
  const r = parseInt(clean.slice(0, 2), 16) / 255;
  const g = parseInt(clean.slice(2, 4), 16) / 255;
  const b = parseInt(clean.slice(4, 6), 16) / 255;
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

// Returns whichever of two hex colors is darker (lower luminance)
function darkerOf(a: string, b: string): string {
  return luminance(a) <= luminance(b) ? a : b;
}

function fmtMoney(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`;
  return `$${n}`;
}

function capPct(hit: number | null, ceiling: number): number {
  if (!hit || !ceiling) return 0;
  return Math.min(100, (hit / ceiling) * 100);
}

function Headshot({ src, firstName, lastName, size = 40, teamColor = "#ffffff", bgColor = "#1a1a2e" }: {
  src: string; firstName: string; lastName: string; size?: number; teamColor?: string; bgColor?: string;
}) {
  const [err, setErr] = useState(false);
  const ring = `0 0 0 2px ${teamColor}80, 0 0 10px ${teamColor}40`;
  if (src && !err) {
    return (
      <img
        src={src}
        alt={`${firstName} ${lastName}`}
        width={size}
        height={size}
        onError={() => setErr(true)}
        className="rounded-full object-cover shrink-0"
        style={{ width: size, height: size, objectPosition: "50% 8%", boxShadow: ring, backgroundColor: bgColor }}
      />
    );
  }
  return (
    <div
      className="rounded-full flex items-center justify-center shrink-0 font-bold"
      style={{ width: size, height: size, fontSize: size * 0.32, color: teamColor, backgroundColor: bgColor, boxShadow: ring }}
    >
      {firstName?.[0]}{lastName?.[0]}
    </div>
  );
}

// Returns the jersey number, or a league badge if player isn't on active NHL roster.
// We don't have per-player league data from the stats endpoint, so we use "M" (minors)
// as default. Junior ("J") and university ("U") are set if the caller passes a hint.
function JerseyCell({ jersey, league = "M" }: { jersey: number | null; league?: "M" | "J" | "U" }) {
  if (jersey !== null) {
    return <span className="text-[12px] font-mono text-white/40">{jersey}</span>;
  }
  const colors: Record<string, string> = { M: "#94a3b8", J: "#fbbf24", U: "#38bdf8" };
  const c = colors[league] ?? "#94a3b8";
  return (
    <span
      className="text-[9px] font-black px-1 py-0.5 rounded border"
      style={{ color: c, borderColor: `${c}50`, backgroundColor: `${c}14` }}
    >
      {league}
    </span>
  );
}

function SkeletonRows({ n = 12 }: { n?: number }) {
  return (
    <div className="space-y-1.5">
      {Array.from({ length: n }).map((_, i) => (
        <div key={i} className="h-10 rounded-xl bg-white/[0.03] animate-pulse" style={{ opacity: 1 - i * 0.06 }} />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function TeamPage() {
  const params  = useParams();
  const router  = useRouter();
  const rawTeam = typeof params.team === "string" ? params.team : Array.isArray(params.team) ? params.team[0] : "";
  const NHL_REMAP: Record<string, string> = { UTH: "UTA", LA: "LAK", NJ: "NJD", SJ: "SJS", TB: "TBL" };
  const team    = NHL_REMAP[rawTeam.toUpperCase()] ?? rawTeam.toUpperCase();

  const teamColor      = TEAM_COLORS[team]     ?? "#a78bfa";
  const secondaryColor = TEAM_SECONDARY[team] ?? "#1a1a2e";
  const darkSecondary  = darkBlend(secondaryColor); // always a dark usable shade for gradients
  // For table backgrounds: use whichever team color is darkest, blended very dark (0.92)
  const tableDarkBg    = darkBlend(darkerOf(teamColor, secondaryColor), 0.92);

  const { theme: activeTheme, setTeamTheme, clearTheme } = useTheme();
  const isThisTeamActive = activeTheme?.abbrev === team;

  const [tab,        setTab]        = useState<"stats" | "cap" | "injuries">("stats");
  const [standing,   setStanding]   = useState<TeamStanding | null>(null);
  const [skaters,    setSkaters]    = useState<Skater[] | null>(null);
  const [goalies,    setGoalies]    = useState<Goalie[] | null>(null);
  const [capData,      setCapData]      = useState<CapData | null>(null);
  const [injuryData,   setInjuryData]   = useState<InjuryData | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [capLoading,   setCapLoading]   = useState(false);
  const [injuryLoading, setInjuryLoading] = useState(false);
  const [sortKey,      setSortKey]      = useState<SortKey>("points");
  const [sortDir,      setSortDir]      = useState<1 | -1>(1);
  const [goalieSortKey, setGoalieSortKey] = useState<GoalieSortKey>("wins");
  const [goalieSortDir, setGoalieSortDir] = useState<1 | -1>(1);

  // Fetch standings + stats on mount
  useEffect(() => {
    setStatsLoading(true);
    setSkaters(null);
    setGoalies(null);

    fetch("/api/standings")
      .then(r => r.ok ? r.json() : null)
      .then(d => d && setStanding(d.standings?.find((t: TeamStanding) => t.team === team) ?? null))
      .catch(() => {});

    fetch(`/api/nhl-team/${team}`)
      .then(r => r.json())
      .then(d => {
        setSkaters(Array.isArray(d.skaters) ? d.skaters : []);
        setGoalies(Array.isArray(d.goalies) ? d.goalies : []);
        setStatsLoading(false);
      })
      .catch(() => { setSkaters([]); setGoalies([]); setStatsLoading(false); });
  }, [team]);

  // Lazy-load cap data when tab switches
  useEffect(() => {
    if (tab === "cap" && !capData) {
      setCapLoading(true);
      fetch(`/api/puckpedia/${team}`)
        .then(r => r.json())
        .then(d => { setCapData(d); setCapLoading(false); })
        .catch(() => { setCapData({ players: [], cap_ceiling: 88_000_000, total_hit: null, cap_space: null, status: "error" }); setCapLoading(false); });
    }
  }, [tab, team, capData]);

  // Lazy-load injury data when tab switches
  useEffect(() => {
    if (tab === "injuries" && !injuryData) {
      setInjuryLoading(true);
      fetch(`/api/injuries/${team}`)
        .then(r => r.json())
        .then(d => { setInjuryData(d); setInjuryLoading(false); })
        .catch(() => { setInjuryData({ injuries: [], as_of: null }); setInjuryLoading(false); });
    }
  }, [tab, team, injuryData]);

  const handleSort = useCallback((key: SortKey) => {
    if (key === sortKey) {
      setSortDir(d => d === 1 ? -1 : 1);
    } else {
      setSortKey(key);
      setSortDir(1);
    }
  }, [sortKey]);

  const handleGoalieSort = useCallback((key: GoalieSortKey) => {
    if (key === goalieSortKey) {
      setGoalieSortDir(d => d === 1 ? -1 : 1);
    } else {
      setGoalieSortKey(key);
      setGoalieSortDir(1);
    }
  }, [goalieSortKey]);

  // sortDir: 1 = descending (highest first), -1 = ascending (lowest first)
  const sortedSkaters = (skaters ?? []).slice().sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    if (typeof av === "string" && typeof bv === "string") return sortDir * bv.localeCompare(av);
    return sortDir * ((Number(bv) || 0) - (Number(av) || 0));
  });

  const sortedGoalies = (goalies ?? []).slice().sort((a, b) => {
    // GAA sorts ascending by default (lower is better) — flip direction
    const flip = goalieSortKey === "gaa" ? -1 : 1;
    return flip * goalieSortDir * ((Number(b[goalieSortKey]) || 0) - (Number(a[goalieSortKey]) || 0));
  });

  // Group cap players
  const capPlayers = capData?.players ?? [];
  const fwds  = capPlayers.filter(p => ["C","L","R","LW","RW","F","W"].some(pos => p.position.toUpperCase().includes(pos)));
  const defs  = capPlayers.filter(p => ["D","LD","RD"].some(pos => p.position.toUpperCase() === pos));
  const glies = capPlayers.filter(p => p.position.toUpperCase() === "G");
  [fwds, defs, glies].forEach(arr => arr.sort((a, b) => (b.cap_hit ?? 0) - (a.cap_hit ?? 0)));

  const sBg  = (k: SortKey)       => sortKey      === k ? "rgba(255,255,255,0.05)" : undefined;
  const gBg  = (k: GoalieSortKey) => goalieSortKey === k ? "rgba(255,255,255,0.05)" : undefined;

  const tableContainerStyle = {
    border: `1px solid ${teamColor}18`,
    background: `linear-gradient(160deg, ${tableDarkBg} 0%, #060708 70%)`,
  } as const;
  const theadRowStyle = {
    borderBottom: `1px solid ${teamColor}15`,
    backgroundColor: `${tableDarkBg}`,
  } as const;

  // [key, label, mobileVisible]
  const SKATER_COLS: [SortKey, string, boolean][] = [
    ["goals","G", true], ["assists","A", true], ["points","PTS", true],
    ["gp","GP", false], ["plus_minus","+/-", false], ["pim","PIM", false],
    ["pp_points","PPP", false], ["gwg","GWG", false],
    ["shots","S", false], ["shooting_pct","S%", false], ["avg_toi","TOI", false],
  ];

  return (
    <main className="min-h-screen p-4 sm:p-6 max-w-5xl mx-auto">

      {/* Back + Set Theme row */}
      <div className="mb-4 flex items-center justify-between">
        <button
          onClick={() => router.back()}
          className="flex items-center gap-1.5 text-[11px] text-white/30 hover:text-white/60 transition-colors"
        >
          ← Back
        </button>

        {isThisTeamActive ? (
          <button
            onClick={() => clearTheme()}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[9px] font-bold uppercase tracking-wider transition-all duration-150 border"
            style={{
              color: teamColor,
              borderColor: `${teamColor}40`,
              background: `${teamColor}10`,
            }}
          >
            <span>✓ Active</span>
            <span className="text-white/30 ml-1">· Reset</span>
          </button>
        ) : (
          <button
            onClick={() => setTeamTheme({ abbrev: team, primaryColor: teamColor, secondaryColor, logoUrl: logoUrl(team) })}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[9px] font-bold uppercase tracking-wider transition-all duration-150 border hover:opacity-90"
            style={{
              color: teamColor,
              borderColor: `${teamColor}35`,
              background: `${teamColor}0d`,
            }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={logoUrl(team)} alt={team} width={14} height={14} className="object-contain" />
            Set as Theme
          </button>
        )}
      </div>

      {/* Header card */}
      <div
        className="mb-3 rounded-2xl border border-white/[0.08] overflow-hidden"
        style={{ background: `linear-gradient(160deg, ${tableDarkBg} 0%, #060708 65%)` }}
      >
        <div className="flex items-center gap-3 sm:gap-5 px-4 sm:px-5 pt-4 sm:pt-5 pb-3 sm:pb-4">
          <div
            className="shrink-0 rounded-2xl p-2 sm:p-2.5 flex items-center justify-center"
            style={{
              background: `radial-gradient(circle at 40% 35%, ${darkBlend(secondaryColor, 0.50)} 0%, ${darkBlend(secondaryColor, 0.88)} 60%, #080a0c 100%)`,
              boxShadow: `0 4px 20px rgba(0,0,0,0.5), 0 1px 0 rgba(255,255,255,0.06) inset`,
            }}
          >
            <img
              src={logoUrl(team)}
              alt={team}
              width={80}
              height={80}
              className="object-contain w-[76px] h-[76px] sm:w-[84px] sm:h-[84px] -m-1"
              style={{ filter: `drop-shadow(0 2px 10px rgba(255,255,255,0.18)) drop-shadow(0 0 4px rgba(255,255,255,0.10))` }}
            />
          </div>
          <div className="w-px shrink-0 bg-white/[0.08]" style={{ height: 36 }} />
          <div className="flex-1 min-w-0">
            <h1 className="text-lg sm:text-2xl font-bold text-white tracking-tight leading-tight">
              {TEAM_FULL_NAMES[team] ?? standing?.team_name ?? team}
            </h1>
            {standing ? (
              <div className="mt-1 space-y-1">
                {/* Record line — always visible */}
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className="text-[11px] sm:text-sm text-white/45 font-mono">
                    {standing.w}–{standing.l}–{standing.otl}
                  </span>
                  <span className="text-white/20 text-[10px]">·</span>
                  <span className="text-[11px] sm:text-sm text-white/45 font-mono">
                    {standing.pts} pts
                  </span>
                  <span className="text-white/20 text-[10px]">·</span>
                  <span className="text-[11px] sm:text-sm text-white/45 font-mono">
                    {standing.gp} GP
                  </span>
                  {standing.streak_code && standing.streak_count ? (
                    <span
                      className="text-[9px] sm:text-[10px] font-bold px-1.5 py-0.5 rounded border"
                      style={standing.streak_code === "W"
                        ? { color: "#4ade80", borderColor: "rgba(74,222,128,0.35)", backgroundColor: "rgba(74,222,128,0.10)" }
                        : { color: "#f87171", borderColor: "rgba(248,113,113,0.35)", backgroundColor: "rgba(248,113,113,0.10)" }
                      }
                    >
                      {standing.streak_count}{standing.streak_code}
                    </span>
                  ) : null}
                </div>
                {/* Division — second line */}
                <p className="text-[10px] sm:text-[11px] text-white/30">
                  {["1st","2nd","3rd"][standing.div_rank - 1] ?? `${standing.div_rank}th`} in {standing.division}
                </p>
              </div>
            ) : (
              <div className="h-3.5 w-40 rounded bg-white/[0.06] animate-pulse mt-1.5" />
            )}
          </div>
        </div>

        {/* Tab bar */}
        <div className="flex border-t border-white/[0.07]">
          {(["stats", "injuries", "cap"] as const).map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className="flex-1 py-3 text-xs sm:text-sm font-semibold uppercase tracking-wide transition-all duration-150"
              style={tab === t
                ? { color: teamColor, borderBottom: `2px solid ${teamColor}`, backgroundColor: `${teamColor}08` }
                : { color: "rgba(255,255,255,0.30)", borderBottom: "2px solid transparent" }}
            >
              {t === "stats" ? "Roster" : t === "injuries" ? "Injuries" : "Cap"}
            </button>
          ))}
        </div>
      </div>

      {/* ── STATS TAB ── */}
      {tab === "stats" && (
        statsLoading ? <SkeletonRows n={14} /> :
        (!skaters || skaters.length === 0) ? (
          <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] px-6 py-10 text-center space-y-1">
            <p className="text-white/40 text-sm">No roster data available for {team}.</p>
            <p className="text-white/20 text-xs">Could not load stats from NHL API — try again</p>
            <button
              onClick={() => { setStatsLoading(true); setSkaters(null); setGoalies(null);
                fetch(`/api/nhl-team/${team}`).then(r => r.json()).then(d => { setSkaters(Array.isArray(d.skaters) ? d.skaters : []); setGoalies(Array.isArray(d.goalies) ? d.goalies : []); setStatsLoading(false); }).catch(() => { setSkaters([]); setGoalies([]); setStatsLoading(false); }); }}
              className="mt-2 px-4 py-1.5 rounded-lg border border-white/[0.12] text-xs text-white/50 hover:text-white/80 hover:border-white/[0.22] transition-all"
            >Retry</button>
          </div>
        ) : (
        <div className="space-y-4">

          {/* Rotate hint — mobile only */}
          <p className="sm:hidden text-[11px] text-white/60 text-center flex items-center justify-center gap-1.5">
            <span>↻</span> Rotate for more stats
          </p>

          {/* Skaters table */}
          <div>
            <p className="text-[11px] font-semibold text-white/35 uppercase tracking-widest mb-3 px-1">
              Skaters &nbsp;·&nbsp; {sortedSkaters.length} players
            </p>
            <div className="rounded-2xl overflow-hidden" style={tableContainerStyle}>
              <table className="w-full">
                <thead>
                  <tr style={theadRowStyle}>
                    <th className="px-3 py-3 text-left w-10">
                      <span className="text-[10px] font-semibold text-white/30 uppercase tracking-wider">#</span>
                    </th>
                    <th className="px-3 py-3 text-left">
                      <span className="text-[10px] font-semibold text-white/30 uppercase tracking-wider">Player</span>
                    </th>
                    <th className="px-2 py-3 text-center hidden sm:table-cell">
                      <span className="text-[10px] font-semibold text-white/30 uppercase tracking-wider">Pos</span>
                    </th>
                    {SKATER_COLS.map(([key, label, mobile]) => (
                      <th
                        key={key}
                        onClick={() => handleSort(key)}
                        className={`px-2 py-3 text-center cursor-pointer select-none${mobile ? "" : " hidden sm:table-cell"}`}
                        style={{ backgroundColor: sBg(key) }}
                      >
                        <span
                          className="text-[10px] font-semibold uppercase tracking-wider"
                          style={{ color: sortKey === key ? teamColor : "rgba(255,255,255,0.28)" }}
                        >
                          {label}{sortKey === key ? (sortDir === 1 ? " ↑" : " ↓") : ""}
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedSkaters.map((p, i) => {
                    const injColor = p.injury_status === "Out" ? "#f87171" : p.injury_status === "DTD" ? "#fbbf24" : p.injury_status ? "#fb923c" : null;
                    return (
                    <tr
                      key={p.player_id ?? i}
                      onClick={() => router.push(`/players/${encodeURIComponent(`${p.first_name} ${p.last_name}`)}`)}
                      className="border-b border-white/[0.04] last:border-0 hover:bg-white/[0.03] transition-colors cursor-pointer"
                      style={injColor ? { backgroundColor: `${injColor}06` } : {}}
                    >
                      <td className="px-3 py-2.5 text-center w-10"><JerseyCell jersey={p.jersey} /></td>
                      <td className="px-3 py-2.5">
                        <div className="flex items-center gap-3">
                          <Headshot src={p.headshot} firstName={p.first_name} lastName={p.last_name} teamColor={teamColor} bgColor={darkSecondary} size={44} />
                          <div className="flex flex-col min-w-0">
                            <div className="flex items-center gap-1.5 flex-wrap">
                              <span className="text-[13px] font-semibold text-white/85 truncate">
                                {p.first_name} {p.last_name}
                              </span>
                              {injColor && (
                                <span className="text-[8px] font-bold uppercase px-1.5 py-0.5 rounded border shrink-0"
                                  style={{ color: injColor, borderColor: `${injColor}40`, backgroundColor: `${injColor}14` }}>
                                  {p.injury_status}
                                </span>
                              )}
                            </div>
                            <span className="text-[11px] text-white/35 sm:hidden">{p.position}</span>
                          </div>
                        </div>
                      </td>
                      <td className="px-2 py-2.5 text-[12px] text-white/45 text-center hidden sm:table-cell">{p.position}</td>
                      <td className="px-2 py-2.5 text-[13px] tabular-nums text-center" style={{ color: sortKey === "goals" ? teamColor : "rgba(255,255,255,0.55)", backgroundColor: sBg("goals") }}>{p.goals}</td>
                      <td className="px-2 py-2.5 text-[13px] tabular-nums text-center" style={{ color: sortKey === "assists" ? teamColor : "rgba(255,255,255,0.55)", backgroundColor: sBg("assists") }}>{p.assists}</td>
                      <td className="px-2 py-2.5 text-[13px] tabular-nums font-bold text-center" style={{ color: sortKey === "points" ? teamColor : "rgba(255,255,255,0.90)", backgroundColor: sBg("points") }}>{p.points}</td>
                      <td className="px-2 py-2.5 text-[12px] tabular-nums text-center hidden sm:table-cell" style={{ color: sortKey === "gp" ? teamColor : "rgba(255,255,255,0.55)", backgroundColor: sBg("gp") }}>{p.gp}</td>
                      <td className="px-2 py-2.5 text-[12px] tabular-nums text-center hidden sm:table-cell"
                        style={{ color: p.plus_minus > 0 ? "#4ade80" : p.plus_minus < 0 ? "#f87171" : "rgba(255,255,255,0.40)", backgroundColor: sBg("plus_minus") }}>
                        {p.plus_minus > 0 ? `+${p.plus_minus}` : p.plus_minus}
                      </td>
                      <td className="px-2 py-2.5 text-[12px] tabular-nums text-center hidden sm:table-cell" style={{ color: sortKey === "pim" ? teamColor : "rgba(255,255,255,0.45)", backgroundColor: sBg("pim") }}>{p.pim}</td>
                      <td className="px-2 py-2.5 text-[12px] tabular-nums text-center hidden sm:table-cell" style={{ color: sortKey === "pp_points" ? teamColor : "rgba(255,255,255,0.45)", backgroundColor: sBg("pp_points") }}>{p.pp_points}</td>
                      <td className="px-2 py-2.5 text-[12px] tabular-nums text-center hidden sm:table-cell" style={{ color: sortKey === "gwg" ? teamColor : "rgba(255,255,255,0.45)", backgroundColor: sBg("gwg") }}>{p.gwg}</td>
                      <td className="px-2 py-2.5 text-[12px] tabular-nums text-center hidden sm:table-cell" style={{ color: sortKey === "shots" ? teamColor : "rgba(255,255,255,0.45)", backgroundColor: sBg("shots") }}>{p.shots}</td>
                      <td className="px-2 py-2.5 text-[12px] tabular-nums text-center hidden sm:table-cell" style={{ color: sortKey === "shooting_pct" ? teamColor : "rgba(255,255,255,0.45)", backgroundColor: sBg("shooting_pct") }}>{p.shooting_pct}%</td>
                      <td className="px-2 py-2.5 text-[12px] tabular-nums text-center hidden sm:table-cell" style={{ color: sortKey === "avg_toi" ? teamColor : "rgba(255,255,255,0.45)", backgroundColor: sBg("avg_toi") }}>{p.avg_toi}</td>
                    </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Goalies table — always render so user can see if data is missing */}
          <div>
            <p className="text-[11px] font-semibold text-white/35 uppercase tracking-widest mb-3 px-1">
              Goalies{goalies && goalies.length > 0 ? ` · ${goalies.length}` : ""}
            </p>
            <div className="rounded-2xl overflow-hidden" style={tableContainerStyle}>
              <table className="w-full">
                <thead>
                  <tr style={theadRowStyle}>
                    <th className="px-3 py-3 text-[10px] font-semibold text-white/30 uppercase tracking-wider text-center w-10">#</th>
                    <th className="px-3 py-3 text-[10px] font-semibold text-white/30 uppercase tracking-wider text-left">Player</th>
                    {([ ["gp","GP",false], ["wins","W",true], ["losses","L",true], ["ot_losses","OT",false], ["gaa","GAA",true], ["sv_pct","SV%",true], ["shutouts","SO",false] ] as [GoalieSortKey,string,boolean][]).map(([key, label, mobile]) => (
                      <th
                        key={key}
                        onClick={() => handleGoalieSort(key)}
                        className={`px-3 py-3 text-center cursor-pointer select-none${mobile ? "" : " hidden sm:table-cell"}`}
                        style={{ backgroundColor: gBg(key) }}
                      >
                        <span className="text-[10px] font-semibold uppercase tracking-wider"
                          style={{ color: goalieSortKey === key ? teamColor : "rgba(255,255,255,0.28)" }}>
                          {label}{goalieSortKey === key ? (goalieSortDir === 1 ? " ↑" : " ↓") : ""}
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(!goalies || goalies.length === 0) ? (
                    <tr>
                      <td colSpan={9} className="px-4 py-8 text-center text-[12px] text-white/30">
                        No goalie data returned from NHL API.
                      </td>
                    </tr>
                  ) : sortedGoalies.map((g, i) => {
                    const injColor = g.injury_status === "Out" ? "#f87171" : g.injury_status === "DTD" ? "#fbbf24" : g.injury_status ? "#fb923c" : null;
                    return (
                    <tr
                      key={g.player_id ?? i}
                      onClick={() => router.push(`/players/${encodeURIComponent(`${g.first_name} ${g.last_name}`)}`)}
                      className="border-b border-white/[0.04] last:border-0 hover:bg-white/[0.03] transition-colors cursor-pointer"
                      style={injColor ? { backgroundColor: `${injColor}06` } : {}}
                    >
                      <td className="px-3 py-2.5 text-center w-10"><JerseyCell jersey={g.jersey} /></td>
                      <td className="px-3 py-2.5">
                        <div className="flex items-center gap-3">
                          <Headshot src={g.headshot} firstName={g.first_name} lastName={g.last_name} teamColor={teamColor} bgColor={darkSecondary} size={44} />
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="text-[13px] font-semibold text-white/85">{g.first_name} {g.last_name}</span>
                            {injColor && (
                              <span className="text-[8px] font-bold uppercase px-1.5 py-0.5 rounded border shrink-0"
                                style={{ color: injColor, borderColor: `${injColor}40`, backgroundColor: `${injColor}14` }}>
                                {g.injury_status}
                              </span>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="px-3 py-2.5 text-[12px] tabular-nums text-center hidden sm:table-cell"
                        style={{ color: goalieSortKey === "gp" ? teamColor : "rgba(255,255,255,0.55)", backgroundColor: gBg("gp") }}>{g.gp}</td>
                      <td className="px-3 py-2.5 text-[13px] tabular-nums font-semibold text-center"
                        style={{ color: goalieSortKey === "wins" ? teamColor : "rgba(255,255,255,0.80)", backgroundColor: gBg("wins") }}>{g.wins}</td>
                      <td className="px-3 py-2.5 text-[13px] tabular-nums text-center"
                        style={{ color: goalieSortKey === "losses" ? teamColor : "rgba(255,255,255,0.55)", backgroundColor: gBg("losses") }}>{g.losses}</td>
                      <td className="px-3 py-2.5 text-[12px] tabular-nums text-center hidden sm:table-cell"
                        style={{ color: goalieSortKey === "ot_losses" ? teamColor : "rgba(255,255,255,0.45)", backgroundColor: gBg("ot_losses") }}>{g.ot_losses}</td>
                      <td className="px-3 py-2.5 text-[13px] tabular-nums font-semibold text-center"
                        style={{ color: goalieSortKey === "gaa" ? teamColor : g.gaa < 2.5 ? "#4ade80" : g.gaa < 3.0 ? "#fbbf24" : "#f87171", backgroundColor: gBg("gaa") }}>
                        {g.gaa.toFixed(2)}
                      </td>
                      <td className="px-3 py-2.5 text-[13px] tabular-nums font-bold text-center"
                        style={{ color: teamColor, backgroundColor: gBg("sv_pct") }}>
                        .{String(Math.round(g.sv_pct * 1000)).padStart(3, "0")}
                      </td>
                      <td className="px-3 py-2.5 text-[12px] tabular-nums text-center hidden sm:table-cell"
                        style={{ color: goalieSortKey === "shutouts" ? teamColor : "rgba(255,255,255,0.55)", backgroundColor: gBg("shutouts") }}>{g.shutouts}</td>
                    </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        )
      )}

      {/* ── CAP TAB ── */}
      {tab === "cap" && (
        capLoading ? <SkeletonRows n={14} /> :
        (!capData || (capData.players?.length ?? 0) === 0) ? (
          <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] px-6 py-14 text-center space-y-1">
            <p className="text-white/40 text-sm">Cap data unavailable.</p>
            <p className="text-white/20 text-xs mt-1">Could not parse PuckPedia data for this team.{capData?.status ? ` (${capData.status})` : ""}</p>
            <button
              onClick={() => { setCapData(null); setCapLoading(true);
                fetch(`/api/puckpedia/${team}`).then(r => r.json()).then(d => { setCapData(d); setCapLoading(false); }).catch(() => { setCapData({ players: [], cap_ceiling: 88_000_000, total_hit: null, cap_space: null, status: "error" }); setCapLoading(false); }); }}
              className="mt-2 px-4 py-1.5 rounded-lg border border-white/[0.12] text-xs text-white/50 hover:text-white/80 hover:border-white/[0.22] transition-all"
            >Retry</button>
          </div>
        ) : (
          <div className="space-y-8">

            {/* Summary row */}
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: "Total Cap Hit", value: fmtMoney(capData.total_hit) },
                { label: "Cap Space",     value: fmtMoney(capData.cap_space) },
                { label: "Cap Ceiling",   value: fmtMoney(capData.cap_ceiling) },
              ].map(card => (
                <div key={card.label} className="rounded-xl px-4 py-3" style={{ border: `1px solid ${teamColor}18`, background: `linear-gradient(160deg, ${tableDarkBg} 0%, #060708 70%)` }}>
                  <p className="text-[9px] text-white/30 uppercase tracking-wider mb-1">{card.label}</p>
                  <p className="text-lg font-bold" style={{ color: teamColor }}>{card.value}</p>
                </div>
              ))}
            </div>

            {/* Team cap bar */}
            {capData.total_hit && (
              <div>
                <div className="flex justify-between text-[9px] text-white/30 mb-1.5 px-0.5">
                  <span>$0</span>
                  <span className="font-semibold" style={{ color: teamColor }}>
                    {((capData.total_hit / capData.cap_ceiling) * 100).toFixed(1)}% of cap used
                  </span>
                  <span>$88M ceiling</span>
                </div>
                <div className="h-2.5 rounded-full bg-white/[0.07] overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{ width: `${(capData.total_hit / capData.cap_ceiling) * 100}%`, backgroundColor: teamColor }}
                  />
                </div>
              </div>
            )}

            {/* Player groups */}
            {[
              { label: "Forwards",  players: fwds },
              { label: "Defence",   players: defs },
              { label: "Goalies",   players: glies },
            ].filter(g => g.players.length > 0).map(group => (
              <div key={group.label}>
                <p className="text-[11px] font-semibold text-white/35 uppercase tracking-widest mb-3 px-1">
                  {group.label} &nbsp;·&nbsp; {group.players.length} players
                </p>
                <div className="rounded-2xl overflow-hidden" style={tableContainerStyle}>
                  <table className="w-full">
                    <thead>
                      <tr style={theadRowStyle}>
                        <th className="px-3 py-3 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-left">Player</th>
                        <th className="px-3 py-3 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-center hidden sm:table-cell">Pos</th>
                        <th className="px-3 py-3 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-center hidden sm:table-cell">Age</th>
                        <th className="px-3 py-3 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-center">Type</th>
                        <th className="px-3 py-3 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-center">Cap Hit</th>
                        <th className="px-3 py-3 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-center hidden sm:table-cell">Expiry</th>
                        <th className="px-3 py-3 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-center hidden sm:table-cell">% Cap</th>
                        <th className="w-24 hidden sm:table-cell"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {group.players.map((p, i) => {
                        const pct = capPct(p.cap_hit, capData.cap_ceiling);
                        const TYPE_COLORS: Record<string, string> = {
                          UFA: "#4ade80", RFA: "#38bdf8", ELC: "#fbbf24", "ENTRY LEVEL": "#fbbf24", "ENTRY-LEVEL": "#fbbf24",
                        };
                        const typeKey  = (p.contract_type || "").toUpperCase();
                        const typeColor = Object.entries(TYPE_COLORS).find(([k]) => typeKey.includes(k))?.[1] ?? "#94a3b8";
                        return (
                          <tr key={i} className="border-b border-white/[0.04] last:border-0 hover:bg-white/[0.03] transition-colors">
                            <td className="px-3 py-2.5">
                              <span className="text-[13px] font-semibold text-white/85">{p.name}</span>
                            </td>
                            <td className="px-3 py-2.5 text-[12px] text-white/45 text-center hidden sm:table-cell">{p.position || "—"}</td>
                            <td className="px-3 py-2.5 text-[12px] tabular-nums text-white/45 text-center hidden sm:table-cell">{p.age ?? "—"}</td>
                            <td className="px-3 py-2.5 text-center">
                              {p.contract_type ? (
                                <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded border whitespace-nowrap"
                                  style={{ color: typeColor, borderColor: `${typeColor}40`, backgroundColor: `${typeColor}12` }}>
                                  {p.contract_type}
                                </span>
                              ) : <span className="text-white/20">—</span>}
                            </td>
                            <td className="px-3 py-2.5 text-[13px] tabular-nums font-bold text-center"
                              style={{ color: teamColor }}>{fmtMoney(p.cap_hit)}</td>
                            <td className="px-3 py-2.5 text-[12px] tabular-nums text-white/45 text-center hidden sm:table-cell">{p.expiry_year ?? "—"}</td>
                            <td className="px-3 py-2.5 text-[12px] tabular-nums text-white/45 text-center hidden sm:table-cell">{pct.toFixed(1)}%</td>
                            <td className="px-3 py-2.5 w-24 hidden sm:table-cell">
                              <div className="h-1.5 rounded-full bg-white/[0.07] overflow-hidden">
                                <div
                                  className="h-full rounded-full"
                                  style={{ width: `${pct}%`, backgroundColor: teamColor, opacity: 0.7 }}
                                />
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        )
      )}

      {/* ── INJURIES TAB ── */}
      {tab === "injuries" && (
        injuryLoading ? <SkeletonRows n={6} /> :
        (!injuryData || injuryData.injuries.length === 0) ? (
          <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] px-6 py-14 text-center space-y-1">
            <p className="text-white/40 text-sm">No injuries reported for {team}.</p>
            {injuryData?.as_of && <p className="text-white/20 text-xs mt-1">As of {injuryData.as_of}</p>}
            <button
              onClick={() => { setInjuryData(null); setInjuryLoading(true);
                fetch(`/api/injuries/${team}`).then(r => r.json()).then(d => { setInjuryData(d); setInjuryLoading(false); }).catch(() => { setInjuryData({ injuries: [], as_of: null }); setInjuryLoading(false); }); }}
              className="mt-2 px-4 py-1.5 rounded-lg border border-white/[0.12] text-xs text-white/50 hover:text-white/80 hover:border-white/[0.22] transition-all"
            >Retry</button>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Header row */}
            <div className="flex items-center justify-between px-1">
              <p className="text-[11px] font-semibold text-white/35 uppercase tracking-widest">
                Injury Report &nbsp;·&nbsp; {injuryData.injuries.length} players
              </p>
              {injuryData.as_of && (
                <p className="text-[10px] text-white/25 font-mono">as of {injuryData.as_of}</p>
              )}
            </div>

            <div className="rounded-2xl overflow-hidden" style={tableContainerStyle}>
              <table className="w-full">
                <thead>
                  <tr style={theadRowStyle}>
                    <th className="px-3 py-3 text-[10px] font-semibold text-white/30 uppercase tracking-wider text-left">Player</th>
                    <th className="px-3 py-3 text-[10px] font-semibold text-white/30 uppercase tracking-wider text-center hidden sm:table-cell">Pos</th>
                    <th className="px-3 py-3 text-[10px] font-semibold text-white/30 uppercase tracking-wider text-center">Status</th>
                    <th className="px-3 py-3 text-[10px] font-semibold text-white/30 uppercase tracking-wider text-left">Injury</th>
                    <th className="px-3 py-3 text-[10px] font-semibold text-white/30 uppercase tracking-wider text-center hidden sm:table-cell">Return Est.</th>
                    <th className="px-3 py-3 text-[10px] font-semibold text-white/30 uppercase tracking-wider text-center">Next Game</th>
                  </tr>
                </thead>
                <tbody>
                  {injuryData.injuries.map((inj, i) => {
                    const statusColor = inj.status === "Out" ? "#f87171" : inj.status === "DTD" ? "#fbbf24" : "#fb923c";
                    const prob = inj.play_probability;
                    const probColor = prob === 0 ? "#f87171" : prob != null && prob >= 70 ? "#4ade80" : prob != null && prob >= 40 ? "#fbbf24" : "#fb923c";
                    const returnFmt = inj.return_estimate
                      ? new Date(inj.return_estimate).toLocaleDateString("en-US", { month: "short", day: "numeric" })
                      : "—";
                    return (
                      <tr key={i} className="border-b border-white/[0.04] last:border-0"
                        style={{ backgroundColor: `${statusColor}05` }}>
                        <td className="px-3 py-3">
                          <span className="text-[13px] font-semibold text-white/85">{inj.player_name}</span>
                        </td>
                        <td className="px-3 py-3 text-[12px] text-white/45 text-center hidden sm:table-cell">{inj.position}</td>
                        <td className="px-3 py-3 text-center">
                          <span className="text-[9px] font-bold uppercase px-1.5 py-0.5 rounded border whitespace-nowrap"
                            style={{ color: statusColor, borderColor: `${statusColor}40`, backgroundColor: `${statusColor}14` }}>
                            {inj.status_raw}
                          </span>
                        </td>
                        <td className="px-3 py-3">
                          <span className="text-[12px] text-white/60">{inj.injury_type}</span>
                          {inj.injury_detail && (
                            <span className="text-[10px] text-white/30 ml-1.5">· {inj.injury_detail}</span>
                          )}
                        </td>
                        <td className="px-3 py-3 text-[12px] tabular-nums text-white/50 text-center hidden sm:table-cell">{returnFmt}</td>
                        <td className="px-3 py-3 text-center">
                          {prob === null ? (
                            <span className="text-[11px] text-white/25">—</span>
                          ) : (
                            <div className="flex flex-col items-center gap-1">
                              <span className="text-[13px] font-bold tabular-nums" style={{ color: probColor }}>{prob}%</span>
                              <div className="w-12 h-1 rounded-full bg-white/[0.07] overflow-hidden">
                                <div className="h-full rounded-full" style={{ width: `${prob}%`, backgroundColor: probColor }} />
                              </div>
                            </div>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <p className="text-[9px] text-white/20 text-center px-2">
              Play probability derived from injury status. DTD = 50%, Probable = 75%, Out/IR = 0%.
            </p>
          </div>
        )
      )}

      <p className="mt-10 text-[9px] text-white/15 font-mono text-center">
        Stats from NHL API · Cap data from PuckPedia · For informational use only
      </p>
    </main>
  );
}
