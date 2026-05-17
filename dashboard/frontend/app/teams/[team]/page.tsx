"use client";

import React, { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { TEAM_COLORS, TEAM_SECONDARY, TEAM_FULL_NAMES, logoUrl, normalizePlayerName } from "@/utils/nhl";
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
  expiry_status: string;
  expiry_year: string | number | null;
  years_remaining: string | number | null;
  status: string | null;
}

interface DeadCapEntry {
  name: string;
  position: string;
  cap_hit: number | null;
  full_cap_hit: number | null;
  retained_pct: string | null;
  kind: "retained" | "buyout" | "buried";
  expiry_year: number | null;
  note: string | null;
}

interface DraftPick {
  year: string;
  round: number;
  team: string;
  is_traded_away: boolean;
  conditions: string[];
}

interface CapProjection {
  name: string;
  status: string;
  arb: string;
  proj_length: number;
  proj_cap_hit: number | null;
  proj_total: number | null;
  proj_pct: number | null;
}

interface Reserve {
  name: string;
  slug: string;
  position: string;
  born: string;
  drafted_by: string;
  draft_year: number;
  round: number;
  overall: number;
  must_sign_by: string;
}

interface RosterPlayer {
  name: string;
  slug: string;
  position: string;  // e.g. "LW, RW" or "C" or "LD"
  cap_hit: number | null;
  contract_type: string;
  expiry_status: string;
  status: string;
}

interface NonRosterPlayer {
  name: string;
  slug: string;
  position: string;
  cap_hit: number | null;
  minors_salary: number | null;
  contract_type: string;
  expiry_status: string;
  status: string;  // "Minor" | "Junior" | "Loan" | "NCAA" etc.
  terms: string;
  terms_details: string;
  draft_year: number;
  born: string;
}

interface CapData {
  players: CapPlayer[];
  dead_cap: DeadCapEntry[];
  dead_cap_total: number | null;
  cap_ceiling: number;
  total_hit: number | null;
  cap_space: number | null;
  projected_cap_space: number | null;
  playoff_cap: number | null;       // This is playoff cap HIT; space = ceiling - playoff_cap
  ltir: number | null;
  draft_picks: DraftPick[];
  projections: CapProjection[];
  reserves: Reserve[];
  nhl_roster: { forwards: RosterPlayer[]; defense: RosterPlayer[]; goalies: RosterPlayer[] };
  non_roster: { forwards: NonRosterPlayer[]; defense: NonRosterPlayer[]; goalies: NonRosterPlayer[] };
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
  const abs = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(0)}K`;
  return `${sign}$${abs}`;
}

function capPct(hit: number | null, ceiling: number): number {
  if (!hit || !ceiling) return 0;
  return Math.min(100, (hit / ceiling) * 100);
}

const POS_LABEL: Record<string, string> = {
  C: "C", L: "LW", R: "RW", D: "D", G: "G",
  LW: "LW", RW: "RW", LD: "LD", RD: "RD", F: "F",
};
function fmtPos(pos: string): string {
  return POS_LABEL[pos?.toUpperCase()] ?? pos ?? "—";
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

  const { theme: activeTheme, setTeamTheme, clearTheme, previewActive, setPreviewTheme } = useTheme();
  // Not "truly" active if we only have a preview
  const isThisTeamActive = activeTheme?.abbrev === team && !previewActive;

  // Auto-apply this team's theme as a preview while on this page.
  // Do NOT clear on unmount — ThemeInjector handles cleanup when the user
  // leaves the /teams + /players area, so navigating team→player keeps the theme.
  useEffect(() => {
    setPreviewTheme({
      abbrev: team,
      primaryColor: teamColor,
      secondaryColor,
      logoUrl: logoUrl(team),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [team]);

  const [tab,        setTab]        = useState<"stats" | "cap" | "injuries">("stats");
  const [capSubTab,  setCapSubTab]  = useState<"summary" | "depth_chart" | "draft_picks">("summary");
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

  // Phase 3 team fatigue snapshot (regular-season last 3 weeks).
  const [teamFI, setTeamFI] = useState<{
    mean_fi: number; max_fi: number; last_game: string | null; rows: number;
    window_start: string; window_end: string;
  } | null>(null);
  useEffect(() => {
    fetch("/api/phase3/team-fatigue?window_days=21")
      .then((r) => r.json())
      .then((d) => {
        if (!d || d.status !== "ok") return;
        const row = (d.teams ?? []).find((t: { team: string }) => t.team === team);
        if (!row) return;
        setTeamFI({
          mean_fi: row.mean_fi,
          max_fi:  row.max_fi,
          last_game: row.last_game,
          rows: row.rows,
          window_start: d.window_start,
          window_end:   d.window_end,
        });
      })
      .catch(() => {});
  }, [team]);

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
        .catch(() => { setCapData({ players: [], dead_cap: [], dead_cap_total: null, cap_ceiling: 95_500_000, total_hit: null, cap_space: null, projected_cap_space: null, playoff_cap: null, ltir: null, draft_picks: [], projections: [], reserves: [], nhl_roster: { forwards: [], defense: [], goalies: [] }, non_roster: { forwards: [], defense: [], goalies: [] }, status: "error" }); setCapLoading(false); });
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

  // Group cap players — split "LW, RW" → ["LW","RW"] and match whole tokens
  // to avoid "RD".includes("R") false-positive bleed between groups
  const capPlayers = capData?.players ?? [];
  const posToks = (p: { position: string }) =>
    new Set(p.position.toUpperCase().split(/[\s,]+/).map(s => s.trim()).filter(Boolean));
  const D_TOKS  = new Set(["D","LD","RD"]);
  const FW_TOKS = new Set(["C","LW","RW","F","W","L","R"]);
  const glies = capPlayers.filter(p => posToks(p).has("G"));
  const defs  = capPlayers.filter(p => {
    const toks = posToks(p);
    return !toks.has("G") && [...toks].every(t => D_TOKS.has(t));
  });
  const fwds  = capPlayers.filter(p => {
    const toks = posToks(p);
    return !toks.has("G") && !defs.includes(p) && [...toks].some(t => FW_TOKS.has(t));
  });
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
    <main className="min-h-screen p-4 sm:p-6 max-w-5xl mx-auto w-full overflow-x-hidden">

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
            <img src={logoUrl(team)} alt={team} width={24} height={24} className="object-contain" />
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
                  {teamFI && (() => {
                    const v = teamFI.mean_fi;
                    const c = v >= 0.18 ? "#f87171" : v >= 0.14 ? "#fb923c" : v >= 0.11 ? "#fbbf24" : "#4ade80";
                    return (
                      <span
                        className="text-[9px] sm:text-[10px] font-bold px-1.5 py-0.5 rounded border font-mono"
                        title={`Phase 3 mean Fatigue Index across ${teamFI.window_start} → ${teamFI.window_end} (${teamFI.rows.toLocaleString()} player-games)`}
                        style={{
                          color:           c,
                          borderColor:     `${c}55`,
                          backgroundColor: `${c}1a`,
                        }}
                      >
                        FI {v.toFixed(3)}
                      </span>
                    );
                  })()}
                </div>
                {/* Division — second line */}
                <p className="text-[10px] sm:text-[11px] text-white/30">
                  {["1st","2nd","3rd"][standing.div_rank - 1] ?? `${standing.div_rank}th`} in {standing.division}
                  {teamFI && (
                    <>
                      <span className="text-white/20 mx-1.5">·</span>
                      <span className="text-white/30">
                        FI {teamFI.window_start} → {teamFI.window_end}, max{" "}
                        <span className="text-white/55">{teamFI.max_fi.toFixed(3)}</span>
                      </span>
                    </>
                  )}
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
              <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr style={theadRowStyle}>
                    <th className="px-2 py-2.5 text-left w-8">
                      <span className="text-[9px] font-semibold text-white/30 uppercase tracking-wider">#</span>
                    </th>
                    <th className="px-2 py-2.5 text-left">
                      <span className="text-[9px] font-semibold text-white/30 uppercase tracking-wider">Player</span>
                    </th>
                    <th className="px-1.5 py-2.5 text-center hidden sm:table-cell">
                      <span className="text-[9px] font-semibold text-white/30 uppercase tracking-wider">Pos</span>
                    </th>
                    {SKATER_COLS.map(([key, label, mobile]) => (
                      <th
                        key={key}
                        onClick={() => handleSort(key)}
                        className={`px-1.5 py-2.5 text-center cursor-pointer select-none${mobile ? "" : " hidden sm:table-cell"}`}
                        style={{ backgroundColor: sBg(key) }}
                      >
                        <span
                          className="text-[9px] font-semibold uppercase tracking-wider"
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
                      onClick={() => router.push(`/players/${encodeURIComponent(normalizePlayerName(`${p.first_name} ${p.last_name}`))}`)}
                      className="border-b border-white/[0.04] last:border-0 hover:bg-white/[0.03] transition-colors cursor-pointer"
                      style={injColor ? { backgroundColor: `${injColor}06` } : {}}
                    >
                      <td className="px-2 py-2 text-center w-8"><JerseyCell jersey={p.jersey} /></td>
                      <td className="px-2 py-2">
                        <div className="flex items-center gap-2">
                          <Headshot src={p.headshot} firstName={p.first_name} lastName={p.last_name} teamColor={teamColor} bgColor={darkSecondary} size={40} />
                          <div className="flex flex-col min-w-0">
                            <div className="flex items-center gap-1 flex-wrap">
                              <span className="text-[11px] font-semibold text-white/85 truncate">
                                {p.first_name} {p.last_name}
                              </span>
                              {injColor && (
                                <span className="text-[7px] font-bold uppercase px-1 py-0.5 rounded border shrink-0"
                                  style={{ color: injColor, borderColor: `${injColor}40`, backgroundColor: `${injColor}14` }}>
                                  {p.injury_status}
                                </span>
                              )}
                            </div>
                            <span className="text-[9px] text-white/35 sm:hidden">{fmtPos(p.position)}</span>
                          </div>
                        </div>
                      </td>
                      <td className="px-1.5 py-2 text-[10px] text-white/45 text-center hidden sm:table-cell">{fmtPos(p.position)}</td>
                      <td className="px-1.5 py-2 text-[11px] tabular-nums text-center" style={{ color: sortKey === "goals" ? teamColor : "rgba(255,255,255,0.55)", backgroundColor: sBg("goals") }}>{p.goals}</td>
                      <td className="px-1.5 py-2 text-[11px] tabular-nums text-center" style={{ color: sortKey === "assists" ? teamColor : "rgba(255,255,255,0.55)", backgroundColor: sBg("assists") }}>{p.assists}</td>
                      <td className="px-1.5 py-2 text-[11px] tabular-nums font-bold text-center" style={{ color: sortKey === "points" ? teamColor : "rgba(255,255,255,0.90)", backgroundColor: sBg("points") }}>{p.points}</td>
                      <td className="px-1.5 py-2 text-[10px] tabular-nums text-center hidden sm:table-cell" style={{ color: sortKey === "gp" ? teamColor : "rgba(255,255,255,0.55)", backgroundColor: sBg("gp") }}>{p.gp}</td>
                      <td className="px-1.5 py-2 text-[10px] tabular-nums text-center hidden sm:table-cell"
                        style={{ color: p.plus_minus > 0 ? "#4ade80" : p.plus_minus < 0 ? "#f87171" : "rgba(255,255,255,0.40)", backgroundColor: sBg("plus_minus") }}>
                        {p.plus_minus > 0 ? `+${p.plus_minus}` : p.plus_minus}
                      </td>
                      <td className="px-1.5 py-2 text-[10px] tabular-nums text-center hidden sm:table-cell" style={{ color: sortKey === "pim" ? teamColor : "rgba(255,255,255,0.45)", backgroundColor: sBg("pim") }}>{p.pim}</td>
                      <td className="px-1.5 py-2 text-[10px] tabular-nums text-center hidden sm:table-cell" style={{ color: sortKey === "pp_points" ? teamColor : "rgba(255,255,255,0.45)", backgroundColor: sBg("pp_points") }}>{p.pp_points}</td>
                      <td className="px-1.5 py-2 text-[10px] tabular-nums text-center hidden sm:table-cell" style={{ color: sortKey === "gwg" ? teamColor : "rgba(255,255,255,0.45)", backgroundColor: sBg("gwg") }}>{p.gwg}</td>
                      <td className="px-1.5 py-2 text-[10px] tabular-nums text-center hidden sm:table-cell" style={{ color: sortKey === "shots" ? teamColor : "rgba(255,255,255,0.45)", backgroundColor: sBg("shots") }}>{p.shots}</td>
                      <td className="px-1.5 py-2 text-[10px] tabular-nums text-center hidden sm:table-cell" style={{ color: sortKey === "shooting_pct" ? teamColor : "rgba(255,255,255,0.45)", backgroundColor: sBg("shooting_pct") }}>{p.shooting_pct}%</td>
                      <td className="px-1.5 py-2 text-[10px] tabular-nums text-center hidden sm:table-cell" style={{ color: sortKey === "avg_toi" ? teamColor : "rgba(255,255,255,0.45)", backgroundColor: sBg("avg_toi") }}>{p.avg_toi}</td>
                    </tr>
                    );
                  })}
                </tbody>
              </table>
              </div>
            </div>
          </div>

          {/* Goalies table — always render so user can see if data is missing */}
          <div>
            <p className="text-[11px] font-semibold text-white/35 uppercase tracking-widest mb-3 px-1">
              Goalies{goalies && goalies.length > 0 ? ` · ${goalies.length}` : ""}
            </p>
            <div className="rounded-2xl overflow-hidden" style={tableContainerStyle}>
              <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr style={theadRowStyle}>
                    <th className="px-2 py-2.5 text-[9px] font-semibold text-white/30 uppercase tracking-wider text-center w-8">#</th>
                    <th className="px-2 py-2.5 text-[9px] font-semibold text-white/30 uppercase tracking-wider text-left">Player</th>
                    {([ ["gp","GP",false], ["wins","W",true], ["losses","L",false], ["ot_losses","OT",false], ["gaa","GAA",true], ["sv_pct","SV%",true], ["shutouts","SO",false] ] as [GoalieSortKey,string,boolean][]).map(([key, label, mobile]) => (
                      <th
                        key={key}
                        onClick={() => handleGoalieSort(key)}
                        className={`px-1.5 py-2.5 text-center cursor-pointer select-none${mobile ? "" : " hidden sm:table-cell"}`}
                        style={{ backgroundColor: gBg(key) }}
                      >
                        <span className="text-[9px] font-semibold uppercase tracking-wider"
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
                      <td colSpan={9} className="px-4 py-8 text-center text-[11px] text-white/30">
                        No goalie data returned from NHL API.
                      </td>
                    </tr>
                  ) : sortedGoalies.map((g, i) => {
                    const injColor = g.injury_status === "Out" ? "#f87171" : g.injury_status === "DTD" ? "#fbbf24" : g.injury_status ? "#fb923c" : null;
                    return (
                    <tr
                      key={g.player_id ?? i}
                      onClick={() => router.push(`/players/${encodeURIComponent(normalizePlayerName(`${g.first_name} ${g.last_name}`))}`)}
                      className="border-b border-white/[0.04] last:border-0 hover:bg-white/[0.03] transition-colors cursor-pointer"
                      style={injColor ? { backgroundColor: `${injColor}06` } : {}}
                    >
                      <td className="px-2 py-2 text-center w-8"><JerseyCell jersey={g.jersey} /></td>
                      <td className="px-2 py-2">
                        <div className="flex items-center gap-2">
                          <Headshot src={g.headshot} firstName={g.first_name} lastName={g.last_name} teamColor={teamColor} bgColor={darkSecondary} size={40} />
                          <div className="flex items-center gap-1 flex-wrap min-w-0">
                            <span className="text-[11px] font-semibold text-white/85 truncate">{g.first_name} {g.last_name}</span>
                            {injColor && (
                              <span className="text-[7px] font-bold uppercase px-1 py-0.5 rounded border shrink-0"
                                style={{ color: injColor, borderColor: `${injColor}40`, backgroundColor: `${injColor}14` }}>
                                {g.injury_status}
                              </span>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="px-1.5 py-2 text-[10px] tabular-nums text-center hidden sm:table-cell"
                        style={{ color: goalieSortKey === "gp" ? teamColor : "rgba(255,255,255,0.55)", backgroundColor: gBg("gp") }}>{g.gp}</td>
                      <td className="px-1.5 py-2 text-[11px] tabular-nums font-semibold text-center"
                        style={{ color: goalieSortKey === "wins" ? teamColor : "rgba(255,255,255,0.80)", backgroundColor: gBg("wins") }}>{g.wins}</td>
                      <td className="px-1.5 py-2 text-[11px] tabular-nums text-center hidden sm:table-cell"
                        style={{ color: goalieSortKey === "losses" ? teamColor : "rgba(255,255,255,0.55)", backgroundColor: gBg("losses") }}>{g.losses}</td>
                      <td className="px-1.5 py-2 text-[10px] tabular-nums text-center hidden sm:table-cell"
                        style={{ color: goalieSortKey === "ot_losses" ? teamColor : "rgba(255,255,255,0.45)", backgroundColor: gBg("ot_losses") }}>{g.ot_losses}</td>
                      <td className="px-1.5 py-2 text-[11px] tabular-nums font-semibold text-center"
                        style={{ color: goalieSortKey === "gaa" ? teamColor : g.gaa < 2.5 ? "#4ade80" : g.gaa < 3.0 ? "#fbbf24" : "#f87171", backgroundColor: gBg("gaa") }}>
                        {g.gaa.toFixed(2)}
                      </td>
                      <td className="px-1.5 py-2 text-[11px] tabular-nums font-bold text-center"
                        style={{ color: teamColor, backgroundColor: gBg("sv_pct") }}>
                        .{String(Math.round(g.sv_pct * 1000)).padStart(3, "0")}
                      </td>
                      <td className="px-1.5 py-2 text-[10px] tabular-nums text-center hidden sm:table-cell"
                        style={{ color: goalieSortKey === "shutouts" ? teamColor : "rgba(255,255,255,0.55)", backgroundColor: gBg("shutouts") }}>{g.shutouts}</td>
                    </tr>
                    );
                  })}
                </tbody>
              </table>
              </div>
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
            <p className="text-white/20 text-xs mt-1">Could not parse cap data for this team.{capData?.status ? ` (${capData.status})` : ""}</p>
            <button
              onClick={() => { setCapData(null); setCapLoading(true);
                fetch(`/api/puckpedia/${team}`).then(r => r.json()).then(d => { setCapData(d); setCapLoading(false); }).catch(() => { setCapData({ players: [], dead_cap: [], dead_cap_total: null, cap_ceiling: 95_500_000, total_hit: null, cap_space: null, projected_cap_space: null, playoff_cap: null, ltir: null, draft_picks: [], projections: [], reserves: [], nhl_roster: { forwards: [], defense: [], goalies: [] }, non_roster: { forwards: [], defense: [], goalies: [] }, status: "error" }); setCapLoading(false); }); }}
              className="mt-2 px-4 py-1.5 rounded-lg border border-white/[0.12] text-xs text-white/50 hover:text-white/80 hover:border-white/[0.22] transition-all"
            >Retry</button>
          </div>
        ) : (() => {
          const isOverCap = (capData.cap_space ?? 0) < 0;
          const capUsedPct = capData.total_hit ? Math.min(100, (capData.total_hit / capData.cap_ceiling) * 100) : 0;
          // Neutral cap accent — never team color (red = over cap, not team identity)
          const CAP_ACCENT = "#94a3b8";
          const CONTRACT_COLORS: Record<string, string> = {
            ELC:    "#fbbf24",
            "2-WAY":"#38bdf8",
            EXT:    "#c084fc",
            PTO:    "#fb923c",
            ATO:    "#fb923c",
            STD:    "#64748b",
          };
          const EXPIRY_COLORS: Record<string, string> = {
            UFA: "#4ade80",
            RFA: "#38bdf8",
          };
          return (
          <div className="space-y-6">

            {/* Summary cards — mirrors PuckPedia layout: Projected | Current | Playoff */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="rounded-xl px-4 py-3 border border-white/[0.07] bg-white/[0.02]">
                <p className="text-[9px] text-white/30 uppercase tracking-wider mb-1">Projected Cap Space</p>
                <p className="text-[17px] font-black tabular-nums"
                  style={{ color: (capData.projected_cap_space ?? 0) < 0 ? "#f87171" : "#4ade80" }}>
                  {fmtMoney(capData.projected_cap_space)}
                </p>
                <p className="text-[8px] text-white/18 leading-tight mt-1">Cap space projected at end of season</p>
              </div>
              <div className="rounded-xl px-4 py-3 border bg-white/[0.02]"
                style={{ borderColor: isOverCap ? "rgba(248,113,113,0.25)" : "rgba(74,222,128,0.20)" }}>
                <p className="text-[9px] text-white/30 uppercase tracking-wider mb-1">Current Cap Space</p>
                <p className="text-[17px] font-black tabular-nums"
                  style={{ color: isOverCap ? "#f87171" : "#4ade80" }}>
                  {fmtMoney(capData.cap_space)}
                </p>
                <p className="text-[8px] text-white/18 leading-tight mt-1">
                  {(capData.ltir ?? 0) > 0
                    ? "Cap hit addable without exceeding cap by more than LTIR pool"
                    : "Cap hit addable without exceeding the upper limit"}
                </p>
              </div>
              {capData.playoff_cap != null ? (() => {
                // playoff_cap from CapWages is the projected 20-man ROSTER CAP HIT, not cap space.
                // Playoff Cap Space = ceiling - playoff_cap
                const playoffSpace = capData.cap_ceiling - capData.playoff_cap;
                const playoffOver = playoffSpace < 0;
                return (
                  <div className="rounded-xl px-4 py-3 border bg-white/[0.02]"
                    style={{ borderColor: playoffOver ? "rgba(248,113,113,0.20)" : "rgba(255,255,255,0.07)" }}>
                    <p className="text-[9px] text-white/30 uppercase tracking-wider mb-1">Projected Playoff Cap Space</p>
                    <p className="text-[17px] font-black tabular-nums"
                      style={{ color: playoffOver ? "#f87171" : "rgba(255,255,255,0.70)" }}>
                      {fmtMoney(playoffSpace)}
                    </p>
                    <p className="text-[8px] text-white/18 leading-tight mt-1">20-man roster cap hit: {fmtMoney(capData.playoff_cap)} · LTIR excluded</p>
                  </div>
                );
              })() : <div />}
              <div className="rounded-xl px-4 py-3 border border-white/[0.07] bg-white/[0.02]">
                <p className="text-[9px] text-white/30 uppercase tracking-wider mb-1">Cap Ceiling</p>
                <p className="text-[17px] font-black tabular-nums text-white/55">{fmtMoney(capData.cap_ceiling)}</p>
                <p className="text-[8px] text-white/18 leading-tight mt-1">Total cap hit: {fmtMoney(capData.total_hit)}</p>
              </div>
            </div>

            {/* Playoff eligibility / LTIR card — only shown when team has LTIR */}
            {(capData.ltir ?? 0) > 0 && (
              <div className="rounded-2xl overflow-hidden border border-white/[0.07]"
                style={{ background: "linear-gradient(135deg, rgba(251,191,36,0.04) 0%, rgba(255,255,255,0.01) 60%, transparent 100%)" }}>
                {/* Header */}
                <div className="flex items-center gap-2 px-4 py-2.5 border-b border-white/[0.05]">
                  <svg width="10" height="10" viewBox="0 0 12 12" fill="none" className="shrink-0">
                    <polygon points="6,1 7.5,4.5 11,5 8.5,7.5 9,11 6,9.5 3,11 3.5,7.5 1,5 4.5,4.5" fill="#fbbf24" fillOpacity="0.7"/>
                  </svg>
                  <span className="text-[9px] font-black uppercase tracking-[0.22em] text-[#fbbf24]/60">Playoff Eligibility</span>
                  <span className="ml-2 text-[8px] text-white/18 font-mono">Per-game compliance required · LTIR not available · {fmtMoney(capData.cap_ceiling)} ceiling</span>
                </div>
                {/* Stats grid */}
                <div className="grid grid-cols-3 divide-x divide-white/[0.05]">
                  <div className="px-4 py-3 flex flex-col gap-0.5">
                    <span className="text-[8px] font-semibold uppercase tracking-[0.18em] text-white/25">Current Cap Space</span>
                    <span className="text-[18px] font-black tabular-nums leading-tight"
                      style={{ color: isOverCap ? "#f87171" : "#4ade80" }}>
                      {fmtMoney(capData.cap_space)}
                    </span>
                    <span className="text-[8px] text-white/20">
                      {isOverCap ? "Over ceiling — must comply per-game in playoffs" : "Cap hit addable without exceeding upper limit"}
                    </span>
                  </div>
                  {(capData.ltir ?? 0) > 0 ? (
                    <div className="px-4 py-3 flex flex-col gap-0.5">
                      <span className="text-[8px] font-semibold uppercase tracking-[0.18em] text-white/25">LTIR Relief</span>
                      <span className="text-[18px] font-black tabular-nums leading-tight text-[#fb923c]">
                        {fmtMoney(capData.ltir)}
                      </span>
                      <span className="text-[8px] text-white/20">SELTIR: full relief · Std: capped $3.82M</span>
                    </div>
                  ) : (
                    <div className="px-4 py-3 flex flex-col gap-0.5">
                      <span className="text-[8px] font-semibold uppercase tracking-[0.18em] text-white/25">LTIR Relief</span>
                      <span className="text-[18px] font-black tabular-nums leading-tight text-white/20">—</span>
                      <span className="text-[8px] text-white/18">No injured reserve relief</span>
                    </div>
                  )}
                  {capData.playoff_cap != null ? (
                    <div className="px-4 py-3 flex flex-col gap-0.5">
                      <span className="text-[8px] font-semibold uppercase tracking-[0.18em] text-white/25">Projected Playoff Cap Space</span>
                      <span className="text-[18px] font-black tabular-nums leading-tight text-white/50">
                        {fmtMoney(capData.playoff_cap)}
                      </span>
                      <span className="text-[8px] text-white/20">20-player lineup + dead cap · scratched/injured excluded</span>
                    </div>
                  ) : <div />}
                </div>
              </div>
            )}

            {/* Cap usage bar */}
            {capData.total_hit && (
              <div>
                <div className="flex justify-between text-[9px] text-white/30 mb-1.5 px-0.5">
                  <span>$0</span>
                  <span className={`font-semibold ${isOverCap ? "text-[#f87171]" : "text-white/50"}`}>
                    {capUsedPct.toFixed(1)}% of cap used
                    {isOverCap && " · OVER CAP"}
                  </span>
                  <span>{fmtMoney(capData.cap_ceiling)}</span>
                </div>
                <div className="h-2 rounded-full bg-white/[0.07] overflow-hidden">
                  <div className="h-full rounded-full transition-all duration-700"
                    style={{ width: `${capUsedPct}%`, background: isOverCap ? "#f87171" : "linear-gradient(90deg,#64748b,#94a3b8)" }} />
                </div>
              </div>
            )}

            {/* Cap section sub-tabs */}
            <div className="flex flex-wrap gap-2">
              {([
                { key: "summary",     label: "Summary"     },
                { key: "depth_chart", label: "Depth Chart" },
                { key: "draft_picks", label: "Draft Picks" },
              ] as const).map(({ key, label }) => (
                <button
                  key={key}
                  onClick={() => setCapSubTab(key)}
                  className="px-4 py-2 rounded-full text-[11px] font-black uppercase tracking-widest border transition-all duration-200 whitespace-nowrap"
                  style={capSubTab === key ? {
                    color: teamColor,
                    borderColor: `${teamColor}60`,
                    background: `linear-gradient(135deg, ${teamColor}20 0%, ${teamColor}0a 100%)`,
                    boxShadow: `0 0 16px ${teamColor}30, inset 0 1px 0 rgba(255,255,255,0.10)`,
                  } : {
                    color: "rgba(255,255,255,0.40)",
                    borderColor: "rgba(255,255,255,0.10)",
                    background: "rgba(255,255,255,0.03)",
                    boxShadow: "inset 0 1px 0 rgba(255,255,255,0.05)",
                  }}
                >
                  {label}
                </button>
              ))}
            </div>

            {capSubTab === "summary" && (<>

            {/* Player groups */}
            {[
              { label: "Forwards", players: fwds },
              { label: "Defence",  players: defs },
              { label: "Goalies",  players: glies },
            ].filter(g => g.players.length > 0).map(group => {
              const groupTotal = group.players.reduce((s, p) => s + (p.cap_hit ?? 0), 0);
              return (
              <div key={group.label}>
                <p className="text-[11px] font-semibold text-white/35 uppercase tracking-widest mb-2 px-1 flex items-center gap-2">
                  {group.label}
                  <span className="text-white/20">· {group.players.length}</span>
                  <span className="ml-auto text-[10px] font-black tabular-nums text-white/40">{fmtMoney(groupTotal)}</span>
                </p>
                <div className="rounded-2xl overflow-hidden" style={tableContainerStyle}>
                  <div className="overflow-x-auto">
                    <table className="w-full min-w-[320px]">
                      <thead>
                        <tr style={theadRowStyle}>
                          <th className="px-3 py-2.5 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-left">Player</th>
                          <th className="px-3 py-2.5 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-center hidden sm:table-cell">Pos</th>
                          <th className="px-3 py-2.5 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-center hidden sm:table-cell">Age</th>
                          {/* Mobile: Yrs | Desktop: Contract type */}
                          <th className="px-3 py-2.5 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-center sm:hidden">Yrs</th>
                          <th className="px-3 py-2.5 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-center hidden sm:table-cell">Contract</th>
                          <th className="px-3 py-2.5 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-center">Cap Hit</th>
                          <th className="px-3 py-2.5 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-center hidden sm:table-cell">Expiry</th>
                          <th className="px-3 py-2.5 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-center hidden sm:table-cell">% Cap</th>
                          <th className="w-20 hidden sm:table-cell" />
                        </tr>
                      </thead>
                      <tbody>
                        {group.players.map((p, i) => {
                          const pct = capPct(p.cap_hit, capData.cap_ceiling);
                          const ctColor = CONTRACT_COLORS[p.contract_type] ?? CAP_ACCENT;
                          const exKey = (p.expiry_status ?? "").toUpperCase();
                          const exColor = EXPIRY_COLORS[exKey] ?? "rgba(255,255,255,0.30)";
                          const isInjured = (p.status ?? "").toUpperCase().includes("IR") || (p.status ?? "").toUpperCase().includes("DTD");
                          const barColor = pct > 12 ? "#94a3b8" : pct > 8 ? "#fbbf24" : "#4ade80";
                          const yrsLeft = typeof p.years_remaining === "number" ? p.years_remaining : null;
                          return (
                            <tr key={i}
                              className="border-b border-white/[0.04] last:border-0 hover:bg-white/[0.035] transition-colors cursor-pointer"
                              onClick={() => router.push(`/players/${encodeURIComponent(normalizePlayerName(p.name))}`)}>
                              <td className="px-3 py-2.5">
                                <div className="flex items-center gap-2">
                                  <span className="text-[13px] font-semibold text-white/85">{p.name}</span>
                                  {isInjured && (
                                    <span className="text-[8px] font-bold px-1 py-0.5 rounded border border-[#f87171]/30 text-[#f87171]/70 bg-[#f87171]/05 leading-none">
                                      {p.status}
                                    </span>
                                  )}
                                </div>
                              </td>
                              <td className="px-3 py-2.5 text-[12px] text-white/70 text-center hidden sm:table-cell">{p.position || "—"}</td>
                              <td className="px-3 py-2.5 text-[12px] tabular-nums text-white/40 text-center hidden sm:table-cell">{p.age ?? "—"}</td>
                              {/* Mobile: years remaining */}
                              <td className="px-3 py-2.5 text-[12px] tabular-nums text-white/50 text-center sm:hidden">
                                {yrsLeft != null ? `${yrsLeft}yr` : "—"}
                              </td>
                              {/* Desktop: contract type + expiry status inline */}
                              <td className="px-3 py-2.5 text-center hidden sm:table-cell">
                                <div className="flex items-center justify-center gap-1.5">
                                  <span className="text-[9px] font-bold px-1.5 py-0.5 rounded border whitespace-nowrap"
                                    style={{ color: ctColor, borderColor: `${ctColor}35`, background: `${ctColor}10` }}>
                                    {p.contract_type || "—"}
                                  </span>
                                  {p.expiry_status && (
                                    <span className="text-[9px] font-semibold whitespace-nowrap" style={{ color: exColor }}>
                                      {p.expiry_status.split(" ")[0]}
                                    </span>
                                  )}
                                </div>
                              </td>
                              <td className="px-3 py-2.5 text-[13px] tabular-nums font-bold text-center text-white/90">
                                {fmtMoney(p.cap_hit)}
                              </td>
                              <td className="px-3 py-2.5 text-[12px] tabular-nums text-white/40 text-center hidden sm:table-cell">
                                {p.expiry_year ?? "—"}
                              </td>
                              <td className="px-3 py-2.5 text-[11px] tabular-nums text-white/35 text-center hidden sm:table-cell">
                                {pct.toFixed(1)}%
                              </td>
                              <td className="px-3 py-2.5 w-20 hidden sm:table-cell">
                                <div className="h-1.5 rounded-full bg-white/[0.07] overflow-hidden">
                                  <div className="h-full rounded-full transition-all duration-500"
                                    style={{ width: `${pct}%`, backgroundColor: barColor }} />
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
              );
            })}

            {/* Dead cap / buyouts / retained salary */}
            {(capData.dead_cap?.length ?? 0) > 0 && (
              <div>
                <p className="text-[11px] font-semibold text-white/35 uppercase tracking-widest mb-2 px-1">
                  Dead Cap <span className="text-white/20">· {fmtMoney(capData.dead_cap_total)}</span>
                </p>
                <div className="rounded-2xl overflow-hidden" style={tableContainerStyle}>
                  <div className="overflow-x-auto">
                    <table className="w-full min-w-[420px]">
                      <thead>
                        <tr style={theadRowStyle}>
                          <th className="px-3 py-2.5 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-left">Player</th>
                          <th className="px-3 py-2.5 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-center">Type</th>
                          <th className="px-3 py-2.5 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-center">Cap Charge</th>
                          <th className="px-3 py-2.5 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-center hidden sm:table-cell">Full AAV</th>
                          <th className="px-3 py-2.5 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-center hidden sm:table-cell">Until</th>
                          <th className="px-3 py-2.5 text-[10px] font-semibold text-white/28 uppercase tracking-wider text-left hidden md:table-cell">Detail</th>
                        </tr>
                      </thead>
                      <tbody>
                        {capData.dead_cap.map((d, i) => {
                          const KIND_LABEL: Record<string, string> = { retained: "RETAINED", buyout: "BUYOUT", buried: "BURIED" };
                          const KIND_COLOR: Record<string, string> = { retained: "#fb923c", buyout: "#f87171", buried: "#94a3b8" };
                          const color = KIND_COLOR[d.kind] ?? "#94a3b8";
                          return (
                            <tr key={i} className="border-b border-white/[0.04] last:border-0 hover:bg-white/[0.025] transition-colors">
                              <td className="px-3 py-2.5">
                                <div className="flex items-center gap-1.5">
                                  <span className="text-[13px] font-semibold text-white/70">{d.name}</span>
                                  {d.position && <span className="text-[9px] text-white/60">{d.position}</span>}
                                </div>
                              </td>
                              <td className="px-3 py-2.5 text-center">
                                <span className="text-[9px] font-bold px-1.5 py-0.5 rounded border whitespace-nowrap"
                                  style={{ color, borderColor: `${color}35`, background: `${color}10` }}>
                                  {KIND_LABEL[d.kind]}
                                  {d.retained_pct && ` · ${d.retained_pct}`}
                                </span>
                              </td>
                              <td className="px-3 py-2.5 text-[13px] tabular-nums font-bold text-center"
                                style={{ color }}>
                                {fmtMoney(d.cap_hit)}
                              </td>
                              <td className="px-3 py-2.5 text-[12px] tabular-nums text-white/30 text-center hidden sm:table-cell">
                                {d.kind === "retained" ? fmtMoney(d.full_cap_hit) : "—"}
                              </td>
                              <td className="px-3 py-2.5 text-[12px] tabular-nums text-white/35 text-center hidden sm:table-cell">
                                {d.expiry_year ?? "—"}
                              </td>
                              <td className="px-3 py-2.5 text-[10px] text-white/20 hidden md:table-cell max-w-[220px] truncate">
                                {d.note ?? "—"}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            </>)}

            {/* ── Cap Projections ─────────────────────────────────────────── */}
            {/* ── Draft Picks ──────────────────────────────────────────────── */}
            {capSubTab === "draft_picks" && (() => {
              const picks = capData.draft_picks ?? [];
              const owned = picks.filter(p => !p.is_traded_away);
              const traded = picks.filter(p => p.is_traded_away);
              const byYear = (arr: DraftPick[]) =>
                arr.reduce<Record<string, DraftPick[]>>((acc, p) => {
                  (acc[p.year] ??= []).push(p);
                  return acc;
                }, {});
              const ownedByYear  = byYear(owned);
              const tradedByYear = byYear(traded);
              const roundLabel = (r: number) => r === 1 ? "1st" : r === 2 ? "2nd" : r === 3 ? "3rd" : `${r}th`;
              const roundColor  = (r: number) => r === 1 ? "#fbbf24" : r === 2 ? "#94a3b8" : r === 3 ? "#fb923c" : "rgba(255,255,255,0.25)";

              return (
                <div className="space-y-5">
                  {picks.length === 0 ? (
                    <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] px-6 py-10 text-center">
                      <p className="text-white/25 text-xs">No draft pick data available.</p>
                    </div>
                  ) : (<>
                    {/* Owned picks */}
                    {Object.keys(ownedByYear).sort().map(year => (
                      <div key={year}>
                        <p className="text-[11px] font-semibold text-white/35 uppercase tracking-widest mb-2 px-1 flex items-center gap-2">
                          {year} Draft
                          <span className="text-white/20">· {ownedByYear[year].length} pick{ownedByYear[year].length !== 1 ? "s" : ""}</span>
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {ownedByYear[year].sort((a,b) => a.round - b.round).map((p, i) => (
                            <div key={i} className="rounded-xl px-3 py-2 border border-white/[0.08] bg-white/[0.02] flex flex-col items-center gap-1 min-w-[72px]">
                              <span className="text-[9px] font-black uppercase tracking-wider" style={{ color: roundColor(p.round) }}>{roundLabel(p.round)}</span>
                              <span className="text-[9px] text-white/30">Round {p.round}</span>
                              {p.conditions.length > 0 && (
                                <span className="text-[7px] text-white/20 text-center leading-tight">{p.conditions.join(" · ")}</span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}

                    {/* Traded away picks */}
                    {traded.length > 0 && (
                      <div>
                        <p className="text-[11px] font-semibold text-[#f87171]/40 uppercase tracking-widest mb-2 px-1 flex items-center gap-2">
                          Traded Away
                          <span className="text-[#f87171]/20">· {traded.length} pick{traded.length !== 1 ? "s" : ""}</span>
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {Object.keys(tradedByYear).sort().flatMap(year =>
                            tradedByYear[year].sort((a,b) => a.round - b.round).map((p, i) => (
                              <div key={`${year}-${i}`} className="rounded-xl px-3 py-2 border border-[#f87171]/10 bg-[#f87171]/[0.02] flex flex-col items-center gap-1 min-w-[72px] opacity-60">
                                <span className="text-[9px] font-black uppercase tracking-wider text-[#f87171]/60">{year} {roundLabel(p.round)}</span>
                                <span className="text-[8px] text-white/20 line-through">Round {p.round}</span>
                              </div>
                            ))
                          )}
                        </div>
                      </div>
                    )}
                  </>)}
                </div>
              );
            })()}

            {/* ── Depth Chart ──────────────────────────────────────────────── */}
            {capSubTab === "depth_chart" && (() => {
              const roster    = capData.nhl_roster ?? { forwards: [], defense: [], goalies: [] };
              const minors    = capData.non_roster  ?? { forwards: [], defense: [], goalies: [] };
              const prospects = capData.reserves    ?? [];

              // Split NHL forwards into LW / C / RW
              // Use the FIRST listed position as the primary slot — CapWages lists primary first
              // e.g. "LW, RW" → LW, "RD, LD" → RD. Prevents multi-pos players appearing twice.
              const primaryPos = (p: RosterPlayer) => p.position.split(",")[0].trim().toUpperCase();

              const lw = roster.forwards.filter(p => { const pp = primaryPos(p); return pp === "LW" || pp === "L"; });
              const c  = roster.forwards.filter(p => primaryPos(p) === "C");
              const rw = roster.forwards.filter(p => { const pp = primaryPos(p); return pp === "RW" || pp === "R"; });
              // Any forward whose primary pos doesn't fit LW/C/RW — put in the shortest column
              const fwdOther = roster.forwards.filter(p => !lw.includes(p) && !c.includes(p) && !rw.includes(p));
              fwdOther.forEach(p => {
                const shortest = [lw, c, rw].reduce((a, b) => a.length <= b.length ? a : b);
                shortest.push(p);
              });

              const ld = roster.defense.filter(p => { const pp = primaryPos(p); return pp === "LD" || pp === "L"; });
              const rd = roster.defense.filter(p => { const pp = primaryPos(p); return pp === "RD" || pp === "R" || pp === "D"; });
              // Untagged — split evenly
              const defOther = roster.defense.filter(p => !ld.includes(p) && !rd.includes(p));
              const defHalf = Math.ceil(defOther.length / 2);
              const ldFull = [...ld, ...defOther.slice(0, defHalf)];
              const rdFull = [...rd, ...defOther.slice(defHalf)];

              const normPos = (pos: string) => pos.split(",").map(s => {
                const t = s.trim().toUpperCase();
                return t === "L" ? "LW" : t === "R" ? "RW" : t;
              }).join(" · ");

              const isIR = (s: string) => /IR|INJURED|LTIR/i.test(s);
              const isOnLoan = (s: string) => /LOAN|MINOR|JUNIOR|NCAA/i.test(s);

              // Glass card for NHL roster — team-colored glow on hover
              // AHL logo — hides itself via CSS on error so no broken-image icon
              const ahlLogoStyle = (accent: string): React.CSSProperties => ({
                filter: `drop-shadow(0 0 4px ${accent}40)`,
              });
              const onAHLLogoError = (e: React.SyntheticEvent<HTMLImageElement>) => {
                (e.target as HTMLImageElement).style.display = "none";
              };

              function NHLCard({ p }: { p: RosterPlayer }) {
                const injured = isIR(p.status);
                return (
                  <div
                    onClick={() => router.push(`/players/${encodeURIComponent(normalizePlayerName(p.name))}`)}
                    className="group relative rounded-xl cursor-pointer transition-all duration-200 overflow-hidden"
                    style={{
                      background: `linear-gradient(135deg, ${teamColor}14 0%, rgba(255,255,255,0.04) 60%, transparent 100%)`,
                      border: `1px solid ${teamColor}28`,
                      boxShadow: `0 2px 12px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.08)`,
                    }}
                  >
                    {/* hover glow overlay */}
                    <div className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none"
                      style={{ background: `linear-gradient(135deg, ${teamColor}22 0%, transparent 70%)`, boxShadow: `inset 0 0 20px ${teamColor}18` }} />
                    <div className="relative px-3 py-2.5 flex flex-col gap-0.5">
                      <div className="flex items-center justify-between gap-1.5">
                        <span className="text-[13px] font-bold text-white/90 leading-tight truncate">
                          {p.name.split(" ").slice(-1)[0]}
                        </span>
                        {injured && (
                          <span className="text-[7px] font-black px-1 py-0.5 rounded border border-[#f87171]/40 text-[#f87171]/80 bg-[#f87171]/08 shrink-0 leading-none">
                            IR
                          </span>
                        )}
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-[9px] font-semibold text-white/70">{normPos(p.position)}</span>
                        {p.cap_hit && <span className="text-[9px] tabular-nums text-white/35">{fmtMoney(p.cap_hit)}</span>}
                      </div>
                    </div>
                  </div>
                );
              }

              // Formation column
              function FormationCol({ label, players }: { label: string; players: RosterPlayer[] }) {
                return (
                  <div className="flex flex-col gap-1.5 min-w-0">
                    <p className="text-[11px] sm:text-[13px] font-black uppercase tracking-[0.18em] text-center mb-1 text-white/80">{label}</p>
                    {players.map((p, i) => <NHLCard key={i} p={p} />)}
                  </div>
                );
              }

              // Minor-league / prospect card — muted glass, smaller type
              function SubCard({ p, accent, showCap = true }: { p: NonRosterPlayer | Reserve; accent: string; showCap?: boolean }) {
                const statusStr = (p as NonRosterPlayer).status ?? "";
                const capHit = (p as NonRosterPlayer).cap_hit;
                const signBy = (p as Reserve).must_sign_by;
                const sub = showCap && capHit ? fmtMoney(capHit) : signBy ? signBy : null;
                return (
                  <div
                    onClick={() => router.push(`/players/${encodeURIComponent(normalizePlayerName(p.name))}`)}
                    className="group relative rounded-xl cursor-pointer transition-all duration-200 overflow-hidden"
                    style={{
                      background: `linear-gradient(135deg, ${accent}0c 0%, rgba(255,255,255,0.025) 70%, transparent 100%)`,
                      border: `1px solid ${accent}20`,
                      boxShadow: `0 1px 8px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05)`,
                    }}
                  >
                    <div className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none"
                      style={{ background: `${accent}10` }} />
                    <div className="relative px-2.5 py-2 flex flex-col gap-0.5">
                      <span className="text-[12px] font-semibold text-white/75 truncate leading-tight">
                        {p.name.split(" ").slice(-1)[0]}
                      </span>
                      <div className="flex items-center justify-between gap-1">
                        <span className="text-[9px] text-white/65">
                          {normPos(p.position)}
                          {statusStr && !isIR(statusStr) ? <> · <span className="text-white/25">{statusStr}</span></> : null}
                        </span>
                        {sub && <span className="text-[8px] tabular-nums text-white/25 shrink-0">{sub}</span>}
                      </div>
                    </div>
                  </div>
                );
              }

              // NHL → AHL primary affiliate mapping
              // Logos served from assets.nhle.com/logos/ahl/svg/{code}_light.svg
              const NHL_AHL_AFFILIATE: Record<string, { name: string; code: string }> = {
                ANA: { name: "San Diego Gulls",                code: "SDG" },
                BOS: { name: "Providence Bruins",              code: "PRO" },
                BUF: { name: "Rochester Americans",            code: "ROC" },
                CAR: { name: "Chicago Wolves",                 code: "CHI" },
                CGY: { name: "Calgary Wranglers",              code: ""    }, // no logo on NHL CDN yet
                CHI: { name: "Rockford IceHogs",               code: "RCK" },
                COL: { name: "Colorado Eagles",                code: "COL" },
                CBJ: { name: "Cleveland Monsters",             code: "CLM" },
                DAL: { name: "Texas Stars",                    code: "TXS" },
                DET: { name: "Grand Rapids Griffins",          code: "GRG" },
                EDM: { name: "Bakersfield Condors",            code: "BAK" },
                FLA: { name: "Charlotte Checkers",             code: "CHA" },
                LAK: { name: "Ontario Reign",                  code: "ONT" },
                MIN: { name: "Iowa Wild",                      code: "IOW" },
                MTL: { name: "Laval Rocket",                   code: "LAV" },
                NSH: { name: "Milwaukee Admirals",             code: "MIL" },
                NJD: { name: "Utica Comets",                   code: "UTI" },
                NYI: { name: "Bridgeport Islanders",           code: "BID" },
                NYR: { name: "Hartford Wolf Pack",             code: "HAR" },
                OTT: { name: "Belleville Senators",            code: "BEL" },
                PHI: { name: "Lehigh Valley Phantoms",         code: "LVP" },
                PIT: { name: "Wilkes-Barre/Scranton Penguins", code: "WBS" },
                SEA: { name: "Coachella Valley Firebirds",     code: "CVF" },
                SJS: { name: "San Jose Barracuda",             code: "SJB" },
                STL: { name: "Springfield Thunderbirds",       code: "SPR" },
                TBL: { name: "Syracuse Crunch",                code: "SYR" },
                TOR: { name: "Toronto Marlies",                code: "TOM" },
                UTA: { name: "Tucson Roadrunners",             code: "TUC" },
                VAN: { name: "Abbotsford Canucks",             code: "ABC" },
                VGK: { name: "Henderson Silver Knights",       code: "HEN" },
                WSH: { name: "Hershey Bears",                  code: "HER" },
                WPG: { name: "Manitoba Moose",                 code: "MAN" },
              };

              // Three-column section for minors or prospects
              function ThreeColSection({
                sectionLabel, sectionAccent, badge, logoSrc, logoAlt, fwds, defs, gols,
              }: {
                sectionLabel: string; sectionAccent: string; badge?: string;
                logoSrc?: string; logoAlt?: string;
                fwds: (NonRosterPlayer | Reserve)[];
                defs: (NonRosterPlayer | Reserve)[];
                gols: (NonRosterPlayer | Reserve)[];
              }) {
                if (!fwds.length && !defs.length && !gols.length) return null;
                const total = fwds.length + defs.length + gols.length;
                return (
                  <div className="rounded-2xl overflow-hidden"
                    style={{
                      background: `linear-gradient(160deg, ${sectionAccent}0a 0%, rgba(255,255,255,0.012) 50%, transparent 100%)`,
                      border: `1px solid ${sectionAccent}18`,
                      boxShadow: `0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05)`,
                    }}>
                    {/* Section header */}
                    <div className="flex items-center gap-3 px-4 py-3 border-b"
                      style={{ borderColor: `${sectionAccent}18` }}>
                      {logoSrc && (
                        <img
                          src={logoSrc}
                          alt={logoAlt ?? ""}
                          onError={onAHLLogoError}
                          className="object-contain shrink-0 opacity-75 w-[30px] h-[30px] sm:w-[42px] sm:h-[42px]"
                          style={ahlLogoStyle(sectionAccent)}
                        />
                      )}
                      <div className="flex items-center gap-2 flex-1">
                        <span className="text-[11px] font-black uppercase tracking-[0.25em]"
                          style={{ color: `${sectionAccent}cc` }}>
                          {sectionLabel}
                        </span>
                        {badge && (
                          <span className="text-[7px] font-black uppercase tracking-widest px-1.5 py-0.5 rounded-full border"
                            style={{ color: `${sectionAccent}99`, borderColor: `${sectionAccent}30`, background: `${sectionAccent}10` }}>
                            {badge}
                          </span>
                        )}
                      </div>
                      <span className="text-[9px] tabular-nums" style={{ color: `${sectionAccent}50` }}>{total}</span>
                    </div>
                    {/* Three columns */}
                    <div className="grid grid-cols-3 divide-x p-3 gap-0"
                      style={{ borderColor: `${sectionAccent}10` }}>
                      {(["Forward", "Defense", "Goalie"] as const).map((colLabel, ci) => {
                        const colPlayers = ci === 0 ? fwds : ci === 1 ? defs : gols;
                        return (
                          <div key={colLabel} className="flex flex-col gap-1.5 px-2 first:pl-0 last:pr-0">
                            <p className="text-[8px] font-black uppercase tracking-[0.22em] text-center mb-1"
                              style={{ color: `${sectionAccent}50` }}>
                              {colLabel}
                            </p>
                            {colPlayers.length > 0
                              ? colPlayers.map((p, i) => <SubCard key={i} p={p} accent={sectionAccent} />)
                              : <div className="text-center text-[9px] text-white/10 py-2">—</div>
                            }
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              }

              // Split prospects by position
              const prospFwds = prospects.filter(p => /^(LW|RW|C|F|W)$/i.test(p.position.trim()) || /LW|RW/.test(p.position));
              const prospDefs = prospects.filter(p => /^(LD|RD|D)$/i.test(p.position.trim()));
              const prospGols = prospects.filter(p => /^G$/i.test(p.position.trim()));
              const prospOther = prospects.filter(p => !prospFwds.includes(p) && !prospDefs.includes(p) && !prospGols.includes(p));

              const hasNHL       = roster.forwards.length + roster.defense.length + roster.goalies.length > 0;
              const hasMinors    = minors.forwards.length + minors.defense.length + minors.goalies.length > 0;
              const hasProspects = prospects.length > 0;

              if (!hasNHL && !hasMinors && !hasProspects) {
                return (
                  <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] px-6 py-10 text-center">
                    <p className="text-white/25 text-xs">No depth chart data available.</p>
                  </div>
                );
              }

              // Prospects accent — muted neutral blue so they feel "further away" from NHL
              const minorsAccent    = "#94a3b8";  // cool slate — AHL professional, not quite NHL
              const prospectsAccent = "#64748b";  // darker slate — even further from the show

              return (
                <div className="space-y-5">

                  {/* ── NHL ROSTER — fully branded ─────────────────── */}
                  {hasNHL && (
                    <div className="rounded-2xl overflow-hidden"
                      style={{
                        background: `linear-gradient(160deg, ${teamColor}18 0%, ${darkSecondary} 55%, rgba(0,0,0,0.6) 100%)`,
                        border: `1px solid ${teamColor}35`,
                        boxShadow: `0 8px 40px rgba(0,0,0,0.5), 0 0 0 1px ${teamColor}10, inset 0 1px 0 rgba(255,255,255,0.08)`,
                      }}>

                      {/* Header with team logo */}
                      <div className="flex items-center gap-3 px-4 py-3 border-b"
                        style={{ borderColor: `${teamColor}25`, background: `${teamColor}0c` }}>
                        <img src={logoUrl(team)} alt={team} className="object-contain opacity-90 shrink-0 w-[36px] h-[36px] sm:w-[52px] sm:h-[52px]" />
                        <div className="flex flex-col">
                          <span className="text-[12px] sm:text-[17px] font-black uppercase tracking-[0.2em]"
                            style={{ color: teamColor }}>NHL Roster</span>
                          <span className="text-[8px] sm:text-[10px] text-white/25 uppercase tracking-wider">
                            {roster.forwards.length + roster.defense.length + roster.goalies.length} players
                          </span>
                        </div>
                      </div>

                      <div className="p-3 space-y-3">
                        {/* Forwards: LW | C | RW */}
                        <div className="grid grid-cols-3 gap-2">
                          <FormationCol label="Left Wing"  players={lw} />
                          <FormationCol label="Center"     players={c} />
                          <FormationCol label="Right Wing" players={rw} />
                        </div>
                        {/* Divider */}
                        <div className="h-px mx-1" style={{ background: `${teamColor}20` }} />
                        {/* Defence: LD | RD | Goalie */}
                        <div className="grid grid-cols-3 gap-2">
                          <FormationCol label="Left Defence"  players={ldFull} />
                          <FormationCol label="Right Defence" players={rdFull} />
                          <FormationCol label="Goalie"        players={roster.goalies} />
                        </div>
                      </div>
                    </div>
                  )}

                  {/* ── MINORS ─────────────────────────────────────── */}
                  {hasMinors && (() => {
                    const aff = NHL_AHL_AFFILIATE[team];
                    const ahlLogoSrc = aff?.code
                      ? `https://assets.nhle.com/logos/ahl/svg/${aff.code}_light.svg`
                      : undefined;
                    return (
                      <ThreeColSection
                        sectionLabel={aff ? aff.name : "Minors"}
                        badge="AHL · ECHL · LOAN"
                        sectionAccent={minorsAccent}
                        logoSrc={ahlLogoSrc}
                        logoAlt={aff?.name}
                        fwds={minors.forwards as NonRosterPlayer[]}
                        defs={minors.defense  as NonRosterPlayer[]}
                        gols={minors.goalies  as NonRosterPlayer[]}
                      />
                    );
                  })()}

                  {/* ── PROSPECTS ──────────────────────────────────── */}
                  {hasProspects && (
                    <ThreeColSection
                      sectionLabel="Prospects"
                      badge="UNSIGNED"
                      sectionAccent={prospectsAccent}
                      fwds={[...prospFwds, ...prospOther] as Reserve[]}
                      defs={prospDefs as Reserve[]}
                      gols={prospGols as Reserve[]}
                    />
                  )}
                </div>
              );
            })()}

            <p className="text-[9px] text-white/15 text-center pb-2">Cap data via CapWages · {new Date().getFullYear()} season</p>
          </div>
          );
        })()
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
              <div className="overflow-x-auto">
              <table className="w-full min-w-[360px]">
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
                        <td className="px-3 py-3 text-[12px] text-white/45 text-center hidden sm:table-cell">{fmtPos(inj.position)}</td>
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
