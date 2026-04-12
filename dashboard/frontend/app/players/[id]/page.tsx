"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useParams, useRouter } from "next/navigation";
import { logoUrl, TEAM_COLORS, TEAM_SECONDARY, normalizePlayerName } from "@/utils/nhl";
import { useTheme } from "@/utils/themeContext";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line, ReferenceLine,
} from "recharts";

// ---------------------------------------------------------------------------
// Types — full profile from /player-profile/{id}
// ---------------------------------------------------------------------------

interface ProfileData {
  player_id: number;
  status: string;
  // Bio (NHL API)
  player_name?: string;
  team?: string;
  position?: string;
  jersey_number?: number;
  height_cm?: number;
  weight_kg?: number;
  shoots_catches?: string;
  birth_date?: string;
  birth_city?: string;
  birth_country?: string;
  draft_year?: number;
  draft_round?: number;
  draft_pick?: number;
  draft_team?: string;
  headshot?: string;
  hero_image?: string;
  nhl_games_played?: number;
  nhl_career_goals?: number;
  nhl_career_points?: number;
  // xG Model
  season?: number;
  shots?: number;
  goals?: number;
  xg_sum?: number;
  finishing?: number | null;
  finishing_per60?: number | null;
  // RAPM
  rapm_ev_off?: number | null;
  rapm_ev_def?: number | null;
  rapm_xga_60?: number | null;
  shots_per60?: number | null;
  goals_per60?: number | null;
  xgf_per60?: number | null;
  toi_ev?: number | null;
  // CDR
  cdr?: number | null;
  // WAR
  war?: number | null;
  gar?: number | null;
  contract_efficiency?: number | null;
  war_rank?: number;
  war_total_qualified?: number;
  // Archetype
  archetype_id?: number | null;
  archetype_name?: string | null;
  // EWMA
  ewma_xgf60?: number | null;
  ewma_form_flag?: string | null;
  ewma_games?: number | null;
  // Hot Hand
  hot_hand_score?: number | null;
  hot_hand_goals5?: number | null;
  hot_hand_xg5?: number | null;
  // Clutch
  clutch_index?: number | null;
  clutch_wpa_per60?: number | null;
  // Special Teams
  special_teams_pp?: number | null;
  special_teams_pk?: number | null;
  // Bayesian
  bayesian_rating?: number | null;
  bayesian_uncertainty?: number | null;
  // Playoff
  playoff_delta?: number | null;
  // Former Team
  former_team_boost?: number | null;
  former_team?: string | null;
  // Skating Baseline
  skating_avg_speed_kmh?: number | null;
  skating_max_speed_kmh?: number | null;
  skating_distance_per_game_km?: number | null;
  skating_zone_time_oz_pct?: number | null;
  skating_zone_time_dz_pct?: number | null;
  skating_games_sample?: number | null;
  // Puck Battles
  battle_score?: number | null;
  battle_percentile?: number | null;
  hits_per60?: number | null;
  blocks_per60?: number | null;
  carry_entry_pct?: number | null;
  net_front_pct?: number | null;
  // Behavioral NN
  nn_carry_in_pct?: number | null;
  nn_dump_pct?: number | null;
  nn_shoot_slot_pct?: number | null;
  nn_shoot_perimeter_pct?: number | null;
  nn_drive_net_pct?: number | null;
  nn_battle_corner_pct?: number | null;
  nn_hold_corner_pct?: number | null;
  nn_fi_score?: number | null;
  // In-Season Blend
  inseason_mu_blend?: number | null;
  inseason_ci_lower?: number | null;
  inseason_ci_upper?: number | null;
  inseason_games?: number | null;
  inseason_blend_weight?: number | null;
  // Line Pairs
  line_pairs?: LinePair[];
  // Goalie
  is_goalie?: boolean;
  games_played?: number;
  shots_against?: number;
  saves?: number;
  goals_against?: number;
  sv_pct?: number | null;
  xga?: number | null;
  gsax?: number | null;
  hd_shots?: number | null;
  hd_saves?: number | null;
  hdsv_pct?: number | null;
  mdsv_pct?: number | null;
  ldsv_pct?: number | null;
  // Game log
  game_log?: {
    games: GameEntry[];
    summary: { n_games: number; goals: number; assists: number; points: number; gpg: number; apg: number; ppg: number };
  };
}

interface LinePair {
  partner_id: number;
  partner_name: string;
  games_together: number | null;
  chemistry_delta: number | null;
  model_xgf_pct: number | null;
  co_toi_ev: number | null;
}

interface GameEntry {
  game_id?: number;
  date: string;
  opponent: string;
  home_road: string;
  goals: number;
  assists: number;
  points: number;
  shots: number;
  toi: string;
  plus_minus: number;
  pp_points: number;
}

// ---------------------------------------------------------------------------
// Tier system
// ---------------------------------------------------------------------------

type Tier = "Elite" | "Above Average" | "Average" | "Below Average" | "Low";

const TIER_COLOR: Record<Tier, string> = {
  "Elite":         "#d946ef",
  "Above Average": "#4ade80",
  "Average":       "#94a3b8",
  "Below Average": "#fbbf24",
  "Low":           "#f87171",
};

function finishingTier(v: number): Tier {
  if (v >= 6)  return "Elite";
  if (v >= 2)  return "Above Average";
  if (v >= -1) return "Average";
  if (v >= -4) return "Below Average";
  return "Low";
}
function warTier(v: number): Tier {
  if (v >= 2.5) return "Elite";
  if (v >= 1.0) return "Above Average";
  if (v >= 0)   return "Average";
  if (v >= -1)  return "Below Average";
  return "Low";
}
function defTier(v: number): Tier {
  if (v >= 0.6)  return "Elite";
  if (v >= 0.2)  return "Above Average";
  if (v >= -0.1) return "Average";
  if (v >= -0.4) return "Below Average";
  return "Low";
}
function stTier(v: number): Tier {
  if (v >= 1.0)  return "Elite";
  if (v >= 0.4)  return "Above Average";
  if (v >= -0.1) return "Average";
  if (v >= -0.5) return "Below Average";
  return "Low";
}
function gsaxTier(v: number): Tier {
  if (v >= 10) return "Elite";
  if (v >= 3)  return "Above Average";
  if (v >= -3) return "Average";
  if (v >= -8) return "Below Average";
  return "Low";
}
function hdsvTier(v: number): Tier {
  if (v >= 85) return "Elite";
  if (v >= 80) return "Above Average";
  if (v >= 75) return "Average";
  if (v >= 70) return "Below Average";
  return "Low";
}
function clutchTier(v: number): Tier {
  if (v >= 0.05)  return "Elite";
  if (v >= 0.01)  return "Above Average";
  if (v >= -0.01) return "Average";
  if (v >= -0.05) return "Below Average";
  return "Low";
}
function xgf60Tier(v: number): Tier {
  if (v >= 5.5)  return "Elite";
  if (v >= 4.7)  return "Above Average";
  if (v >= 3.5)  return "Average";
  if (v >= 2.5)  return "Below Average";
  return "Low";
}
function rapmOffTier(v: number): Tier {
  if (v >= 1.5)  return "Elite";
  if (v >= 0.5)  return "Above Average";
  if (v >= -0.5) return "Average";
  if (v >= -1.5) return "Below Average";
  return "Low";
}
function rapmDefTier(v: number): Tier {
  if (v >= 1.5)  return "Elite";
  if (v >= 0.5)  return "Above Average";
  if (v >= -0.5) return "Average";
  if (v >= -1.5) return "Below Average";
  return "Low";
}
// xGA/60 allowed — lower is better so invert
function xgaAllowedTier(v: number): Tier {
  if (v <= 2.5)  return "Elite";
  if (v <= 3.0)  return "Above Average";
  if (v <= 3.8)  return "Average";
  if (v <= 4.5)  return "Below Average";
  return "Low";
}
function shots60Tier(v: number): Tier {
  if (v >= 13)  return "Elite";
  if (v >= 11)  return "Above Average";
  if (v >= 8)   return "Average";
  if (v >= 5)   return "Below Average";
  return "Low";
}
function goals60Tier(v: number): Tier {
  if (v >= 1.2) return "Elite";
  if (v >= 0.8) return "Above Average";
  if (v >= 0.5) return "Average";
  if (v >= 0.25) return "Below Average";
  return "Low";
}
function battlePctTier(v: number): Tier {
  if (v >= 85) return "Elite";
  if (v >= 70) return "Above Average";
  if (v >= 40) return "Average";
  if (v >= 25) return "Below Average";
  return "Low";
}
function hotHandTier(v: number): Tier {
  if (v >= 1.5)  return "Elite";
  if (v >= 0.7)  return "Above Average";
  if (v >= -0.5) return "Average";
  if (v >= -1.0) return "Below Average";
  return "Low";
}
function garTier(v: number): Tier {
  if (v >= 10) return "Elite";
  if (v >= 4)  return "Above Average";
  if (v >= 0)  return "Average";
  if (v >= -4) return "Below Average";
  return "Low";
}
function contractEffTier(v: number): Tier {
  if (v >= 2.0) return "Elite";
  if (v >= 1.3) return "Above Average";
  if (v >= 0.8) return "Average";
  if (v >= 0.4) return "Below Average";
  return "Low";
}
function bayesianTier(v: number): Tier {
  if (v >= 0.15)  return "Elite";
  if (v >= 0.05)  return "Above Average";
  if (v >= -0.05) return "Average";
  if (v >= -0.15) return "Below Average";
  return "Low";
}
function svpctTier(v: number): Tier {
  // v is 0–1
  if (v >= 0.925) return "Elite";
  if (v >= 0.915) return "Above Average";
  if (v >= 0.905) return "Average";
  if (v >= 0.895) return "Below Average";
  return "Low";
}
function mdsvTier(v: number): Tier {
  if (v >= 0.94) return "Elite";
  if (v >= 0.92) return "Above Average";
  if (v >= 0.90) return "Average";
  if (v >= 0.88) return "Below Average";
  return "Low";
}
function ldsvTier(v: number): Tier {
  if (v >= 0.990) return "Elite";
  if (v >= 0.985) return "Above Average";
  if (v >= 0.975) return "Average";
  if (v >= 0.960) return "Below Average";
  return "Low";
}

const TIER_ABBREV: Record<Tier, string> = {
  "Elite":         "Elite",
  "Above Average": "Above Avg",
  "Average":       "Avg",
  "Below Average": "Below Avg",
  "Low":           "Low",
};

function TierBadge({ tier }: { tier: Tier; small?: boolean }) {
  const color = TIER_COLOR[tier];
  return (
    <span
      className="text-[8px] font-semibold uppercase tracking-wider rounded border px-1.5 py-0.5 shrink-0 whitespace-nowrap"
      style={{ color, borderColor: `${color}40`, backgroundColor: `${color}12` }}
    >
      {TIER_ABBREV[tier]}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Player type derivation
// ---------------------------------------------------------------------------

function derivePlayerType(data: ProfileData, nhlStats?: { gp: number; goals: number; points: number } | null): string | null {
  if (data.is_goalie) return null;
  const pos = (data.position ?? "").toUpperCase();
  const isD = pos === "D" || pos === "LD" || pos === "RD";
  const finishing  = data.finishing     ?? 0;
  const goals60    = data.goals_per60   ?? 0;
  const rapmOff    = data.rapm_ev_off   ?? 0;
  const rapmDef    = data.rapm_ev_def   ?? 0;
  const hits60     = data.hits_per60    ?? 0;
  const cdr        = data.cdr           ?? 0;

  if (isD) {
    const xgf60   = data.xgf_per60  ?? 0;
    const shots60 = data.shots_per60 ?? 0;
    // game_log PPG is reliable regardless of whether RAPM parquet has xgf/shots columns
    const ppg     = data.game_log?.summary.ppg ?? 0;
    // NHL API season PPG is the most reliable offensive signal for D —
    // guaranteed to be populated and reflects actual current-season production.
    const nhlPpg  = (nhlStats && nhlStats.gp > 0) ? nhlStats.points / nhlStats.gp : 0;
    // A D scoring 0.50+ PPG is unambiguously an offensive contributor;
    // RAPM per-60 stats may miss playmakers who generate through passing not shots.
    const hasOffense = (xgf60 >= 2.4)
      || (goals60 >= 0.30)
      || (shots60 >= 2.6)
      || (ppg >= 0.50)
      || (nhlPpg >= 0.50);
    if (rapmOff >= 1.0 && rapmDef >= 0.8) return "Elite Two-Way D";
    if (rapmOff >= 0.8 || (hasOffense && rapmOff > -0.3)) return "Offensive Defenseman";
    if (rapmOff >= 0.3 && rapmDef >= 0.3) return "Two-Way Defenseman";
    if ((rapmDef >= 0.8 || cdr >= 0.5) && !hasOffense) return "Defensive Defenseman";
    return null;
  }

  // Forwards — finishing and goals60 drive primary type
  if (finishing >= 5 && goals60 >= 1.0)                     return "Sniper";
  if (finishing >= 3 && goals60 >= 0.7)                     return "Goal Scorer";
  if (finishing >= 2 && hits60 >= 3.0)                      return "Power Forward";
  if (finishing >= 2 && rapmDef >= 0.5)                     return "Two-Way Forward";
  if (finishing >= 2)                                        return "Goal Scorer";
  if (rapmOff >= 1.2 && finishing < 1)                      return "Playmaker";
  if (rapmOff >= 0.7 && rapmDef >= 0.5)                     return "Two-Way Forward";
  if (rapmOff >= 0.7)                                        return "Offensive Forward";
  if (rapmDef >= 0.8 || cdr >= 0.5)                         return "Defensive Forward";
  if (hits60 >= 3.5)                                         return "Power Forward";
  return null;
}

// ---------------------------------------------------------------------------
// Theme helpers (same logic as team page)
// ---------------------------------------------------------------------------

function darkBlend(hex: string, darkness = 0.82): string {
  const bg = [13, 15, 19];
  const clean = hex.replace("#", "");
  const r = parseInt(clean.slice(0, 2), 16) || 0;
  const g = parseInt(clean.slice(2, 4), 16) || 0;
  const b = parseInt(clean.slice(4, 6), 16) || 0;
  return `#${Math.round(r*(1-darkness)+bg[0]*darkness).toString(16).padStart(2,"0")}${Math.round(g*(1-darkness)+bg[1]*darkness).toString(16).padStart(2,"0")}${Math.round(b*(1-darkness)+bg[2]*darkness).toString(16).padStart(2,"0")}`;
}
function luminancePP(hex: string): number {
  const c = hex.replace("#", "");
  return 0.2126*(parseInt(c.slice(0,2),16)/255) + 0.7152*(parseInt(c.slice(2,4),16)/255) + 0.0722*(parseInt(c.slice(4,6),16)/255);
}
function darkerOfPP(a: string, b: string): string {
  return luminancePP(a) <= luminancePP(b) ? a : b;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const NHL_REMAP: Record<string, string> = { LA: "LAK", NJ: "NJD", SJ: "SJS", TB: "TBL" };

function TeamLogo({ team, size = 24, onClick }: { team: string; size?: number; onClick?: () => void }) {
  const [err, setErr] = useState(false);
  if (err) return <span className="text-[10px] font-mono text-white/40">{team}</span>;
  const img = (
    <img
      src={logoUrl(team)}
      alt={team}
      width={size}
      height={size}
      onError={() => setErr(true)}
      className="object-contain shrink-0"
    />
  );
  if (onClick) {
    return (
      <button onClick={onClick} className="shrink-0 cursor-pointer hover:opacity-80 transition-opacity" title={`View ${team} team page`}>
        {img}
      </button>
    );
  }
  return img;
}

function BarStat({ label, value, max, color = "#4ade80", suffix = "%" }: {
  label: string; value: number; max: number; color?: string; suffix?: string;
}) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-white/50">{label}</span>
        <span className="text-[11px] font-semibold font-mono text-white/70">{value.toFixed(1)}{suffix}</span>
      </div>
      <div className="h-1.5 rounded-full bg-white/[0.07] overflow-hidden">
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, backgroundColor: color, opacity: 0.75 }} />
      </div>
    </div>
  );
}

function StatInfoTip({ label, tip }: { label: string; tip: string }) {
  const [pos, setPos]     = useState<{ x: number; y: number } | null>(null);
  const btnRef            = useRef<HTMLButtonElement>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => { setMounted(true); }, []);

  function open() {
    if (!btnRef.current) return;
    const r = btnRef.current.getBoundingClientRect();
    const popW = 240;
    const rawX = r.left + r.width / 2 - popW / 2;
    const clampedX = Math.max(8, Math.min(rawX, window.innerWidth - popW - 8));
    setPos({ x: clampedX, y: r.top }); // fixed coords — no scrollY
  }
  function close() { setPos(null); }

  const isOpen = pos !== null;

  return (
    <>
      <button
        ref={btnRef}
        className="flex items-center justify-center w-4 h-4 rounded-full text-[9px] font-black transition-all duration-150 shrink-0"
        style={{
          color: isOpen ? "#0d0f13" : "#38bdf8",
          backgroundColor: isOpen ? "#38bdf8" : "rgba(56,189,248,0.15)",
          border: "1px solid rgba(56,189,248,0.40)",
          boxShadow: isOpen ? "0 0 8px rgba(56,189,248,0.55)" : "none",
        }}
        onMouseEnter={open}
        onMouseLeave={close}
        onClick={e => { e.stopPropagation(); isOpen ? close() : open(); }}
      >
        i
      </button>
      {mounted && isOpen && createPortal(
        <div
          className="fixed z-[9999] w-60 rounded-xl border border-white/[0.14] shadow-[0_12px_40px_rgba(0,0,0,0.85)]"
          style={{ left: pos.x, top: pos.y - 6, transform: "translateY(-100%)", background: "#141619", pointerEvents: "none" }}
        >
          <div className="px-3.5 py-3">
            <p className="text-[10px] font-semibold text-[#38bdf8] mb-1.5">{label}</p>
            <p className="text-[10px] text-white/65 leading-relaxed">{tip}</p>
          </div>
        </div>,
        document.body
      )}
    </>
  );
}

function StatRow({ label, value, tier, sub, tip }: {
  label: string; value: string; tier?: Tier; sub?: string; tip?: string;
}) {
  return (
    <div className="py-2.5 border-b border-white/[0.05] last:border-0">
      <div className="flex items-center justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <p className="text-[12px] font-medium text-white/65">{label}</p>
            {tip && <StatInfoTip label={label} tip={tip} />}
          </div>
          {sub && <p className="text-[9px] text-white/30 mt-0.5">{sub}</p>}
        </div>
        <div className="flex flex-col items-end gap-1 shrink-0">
          <span className="text-[12px] font-semibold font-mono text-white/85">{value}</span>
          {tier && <TierBadge tier={tier} />}
        </div>
      </div>
    </div>
  );
}

function Card({ title, icon, children, className = "", style }: {
  title: string; icon?: string; children: React.ReactNode; className?: string; style?: React.CSSProperties;
}) {
  return (
    <div className={`rounded-2xl overflow-hidden ${className}`} style={style ?? { border: "1px solid rgba(255,255,255,0.08)", background: "linear-gradient(to bottom, rgba(255,255,255,0.025), transparent)" }}>
      <div className="px-4 py-3 border-b border-white/[0.07] flex items-center justify-center">
        <p className="text-[13px] font-semibold text-white/80 text-center">{title}</p>
      </div>
      <div className="p-3 sm:p-5">{children}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Glossary
// ---------------------------------------------------------------------------

function SkeletonCard() {
  return (
    <div className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-5 space-y-3 animate-pulse">
      <div className="h-4 bg-white/[0.07] rounded w-32" />
      <div className="h-3 bg-white/[0.05] rounded w-48" />
      <div className="h-3 bg-white/[0.05] rounded w-40" />
    </div>
  );
}

function FormBadge({ flag, ewmaDelta }: { flag?: string | null; ewmaDelta?: number | null }) {
  const isHot  = flag === "rising" || flag === "hot";
  const isCold = flag === "falling" || flag === "cold";
  const delta  = ewmaDelta ?? 0;
  const showHot  = isHot  || delta > 1.5;
  const showCold = isCold || delta < -1.5;
  if (!showHot && !showCold) return null;
  return (
    <span
      className="inline-flex items-center gap-1 text-[10px] font-semibold rounded-full px-2.5 py-1 shrink-0"
      style={{
        color: showHot ? "#4ade80" : "#f87171",
        backgroundColor: showHot ? "rgba(74,222,128,0.10)" : "rgba(248,113,113,0.10)",
        border: `1px solid ${showHot ? "rgba(74,222,128,0.30)" : "rgba(248,113,113,0.30)"}`,
      }}
    >
      {showHot ? "🔥" : "🧊"} {showHot ? "Running Hot" : "Running Cold"}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Game Log table
// ---------------------------------------------------------------------------

// Parse "MM:SS" TOI string → minutes as float
function parseToi(toi: string): number {
  const [m, s] = toi.split(":").map(Number);
  return (m || 0) + (s || 0) / 60;
}

function GameLogTable({ allGames }: { allGames: GameEntry[] }) {
  const [limit, setLimit] = useState<3 | 5 | 10>(5);
  const [per60, setPer60] = useState(false);

  if (allGames.length === 0)
    return <p className="text-[11px] text-white/25 py-2">No recent game data available.</p>;

  const games = allGames.slice(0, limit);
  const totalToi = games.reduce((s, g) => s + parseToi(g.toi), 0);
  const totalG   = games.reduce((s, g) => s + g.goals, 0);
  const totalA   = games.reduce((s, g) => s + g.assists, 0);
  const totalPts = totalG + totalA;
  const totalSh  = games.reduce((s, g) => s + (g.shots ?? 0), 0);
  const n = games.length;

  // Per-60 from aggregated TOI
  const scale = totalToi > 0 ? 60 / totalToi : 0;
  const g60   = totalG   * scale;
  const a60   = totalA   * scale;
  const pts60 = totalPts * scale;
  const sh60  = totalSh  * scale;

  const gpg   = n > 0 ? totalG   / n : 0;
  const ppg   = n > 0 ? totalPts / n : 0;

  return (
    <div>
      {/* Controls */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <div className="flex rounded-lg overflow-hidden border border-white/[0.10]">
          {([3, 5, 10] as const).map(l => (
            <button
              key={l}
              onClick={() => setLimit(l)}
              className={`px-3 py-1 text-[10px] font-semibold transition-colors duration-150
                ${limit === l ? "bg-white/[0.12] text-white/90" : "text-white/35 hover:text-white/60 hover:bg-white/[0.05]"}`}
            >
              L{l}
            </button>
          ))}
        </div>
        <button
          onClick={() => setPer60(v => !v)}
          className={`px-3 py-1 rounded-lg border text-[10px] font-semibold transition-colors duration-150
            ${per60
              ? "bg-[#38bdf8]/15 border-[#38bdf8]/40 text-[#38bdf8]"
              : "border-white/[0.10] text-white/35 hover:text-white/60 hover:bg-white/[0.05]"}`}
        >
          /60
        </button>

        {/* Summary */}
        <span className="ml-auto text-[10px] text-white/45 tabular-nums">
          {per60 ? (
            <>{g60.toFixed(2)}G · {a60.toFixed(2)}A · {pts60.toFixed(2)}Pts · {sh60.toFixed(1)}Sh&nbsp;<span className="text-white/25">per 60</span></>
          ) : (
            <>{totalG}G {totalA}A&nbsp;<span className="text-white/25">({gpg.toFixed(2)} G/gm · {ppg.toFixed(2)} pts/gm)</span></>
          )}
        </span>
      </div>

      {/* Rows */}
      <div className="space-y-1.5">
        {games.map((g, i) => {
          const scored = g.goals > 0 || g.assists > 0;
          const toiMin = parseToi(g.toi);
          const gScale = toiMin > 0 ? 60 / toiMin : 0;
          return (
            <div key={i}
              className={`flex items-center gap-1.5 sm:gap-2.5 px-2.5 py-2 rounded-lg text-[10px] sm:text-[11px]
                ${scored ? "bg-white/[0.04] border border-white/[0.08]" : "bg-white/[0.02] border border-white/[0.04]"}`}
            >
              <span className="text-white/30 font-mono w-12 shrink-0 tabular-nums">{g.date?.slice(5)}</span>
              <span className="text-white/25 font-mono shrink-0 w-4">{g.home_road === "H" ? "vs" : "@"}</span>
              <TeamLogo team={g.opponent} size={16} />
              <span className="text-white/40 font-mono shrink-0 w-7">{g.opponent}</span>
              <span className="flex-1" />

              {per60 ? (
                <span className={`font-semibold tabular-nums w-24 text-right text-[10px]
                  ${scored ? "text-[#38bdf8]/80" : "text-white/25"}`}>
                  {(g.goals * gScale).toFixed(2)}G&nbsp;{(g.assists * gScale).toFixed(2)}A
                </span>
              ) : (
                <span className={`font-semibold tabular-nums w-14 text-right ${scored ? "text-white/85" : "text-white/25"}`}>
                  {g.goals}G {g.assists}A
                </span>
              )}

              {g.pp_points > 0 && (
                <span className="text-[9px] text-[#fbbf24]/55 font-mono shrink-0">PP</span>
              )}
              <span className="text-white/20 font-mono w-12 text-right shrink-0">{g.toi}</span>
              <span className="font-mono w-8 text-right shrink-0 text-[11px]"
                style={{ color: g.plus_minus > 0 ? "rgba(74,222,128,0.65)" : g.plus_minus < 0 ? "rgba(248,113,113,0.65)" : "rgba(255,255,255,0.2)" }}>
                {g.plus_minus > 0 ? `+${g.plus_minus}` : g.plus_minus}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Partner headshot (needs own error state)
// ---------------------------------------------------------------------------

function PartnerHeadshot({ playerId, season }: { playerId: number; season: number }) {
  const [err, setErr] = useState(false);
  const url = `https://assets.nhle.com/mugs/nhl/${season}${season + 1}/${playerId}.png`;
  if (err) {
    return (
      <div className="h-8 w-8 rounded-full bg-white/[0.06] flex items-center justify-center shrink-0">
        <span className="text-[10px] text-white/30">?</span>
      </div>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={url}
      alt=""
      onError={() => setErr(true)}
      className="h-8 w-8 rounded-full object-cover object-top shrink-0 bg-white/[0.06] scale-110 origin-top overflow-hidden"
    />
  );
}

// ---------------------------------------------------------------------------
// Age helper
// ---------------------------------------------------------------------------

function calcAge(birthDate?: string): number | null {
  if (!birthDate) return null;
  const bd = new Date(birthDate);
  const now = new Date();
  let age = now.getFullYear() - bd.getFullYear();
  const m = now.getMonth() - bd.getMonth();
  if (m < 0 || (m === 0 && now.getDate() < bd.getDate())) age--;
  return age;
}

function fmtHeight(cm?: number): string {
  if (!cm) return "";
  const totalIn = Math.round(cm / 2.54);
  return `${Math.floor(totalIn / 12)}′${totalIn % 12}″ / ${cm} cm`;
}

function ordinal(n: number): string {
  const s = ["th","st","nd","rd"];
  const v = n % 100;
  return `${n}${s[(v-20)%10] ?? s[v] ?? s[0]}`;
}

// ---------------------------------------------------------------------------
// Radar chart — player attribute spider
// ---------------------------------------------------------------------------

function clamp01(v: number | null | undefined): number {
  if (v == null) return 0;
  return Math.max(0, Math.min(1, v));
}

interface RadarEntry { subject: string; A: number; raw: string; tip: string; }


function PlayerRadarChart({ data, teamColor }: { data: ProfileData; teamColor: string }) {
  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  const [chartSize, setChartSize] = useState(300);
  useEffect(() => {
    const update = () => setChartSize(Math.min(300, window.innerWidth - 56));
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  // Derive per-60 from raw season counts when model fields are null
  const derivedG60   = (data.goals_per60  != null) ? data.goals_per60
                     : (data.goals  != null && data.toi_ev != null && data.toi_ev > 0) ? (data.goals  / data.toi_ev * 60) : null;
  const derivedSh60  = (data.shots_per60  != null) ? data.shots_per60
                     : (data.shots  != null && data.toi_ev != null && data.toi_ev > 0) ? (data.shots  / data.toi_ev * 60) : null;

  // Offense: use per-60 rate stats only — xg_sum is a season cumulative total and CANNOT
  // be used as a per-60 proxy (inflates scores for players with more games / model data only).
  // ewma_xgf60 is a good per-60 fallback when the RAPM model hasn't run yet.
  const offenseVal = data.xgf_per60
    ?? data.ewma_xgf60
    ?? (derivedG60  != null ? derivedG60  * 6   : null)   // 1 G/60 ≈ 6 xGF/60 (league-avg shot quality)
    ?? (derivedSh60 != null ? derivedSh60 * 0.5 : null);  // rough proxy; drops out if TOI unknown too
  const offenseRaw = data.xgf_per60 != null ? `${data.xgf_per60.toFixed(2)} xGF/60`
    : data.ewma_xgf60 != null ? `${data.ewma_xgf60.toFixed(2)} xGF/60 (EWMA)`
    : derivedG60     != null ? `${derivedG60.toFixed(2)} G/60`
    : derivedSh60    != null ? `${derivedSh60.toFixed(1)} Sh/60`
    : "—";

  // Physical: battle_percentile is the gold standard (requires puck-battles model).
  // Fall back to hits + blocks per 60 when percentile data is missing.
  // Calibration: 10 hits/60 = ~80th pct physical, 1.5 blocks/60 = moderate contribution.
  const physicalFallbackA = (data.hits_per60 != null || data.blocks_per60 != null)
    ? Math.min(100, Math.round((data.hits_per60 ?? 0) * 8 + (data.blocks_per60 ?? 0) * 12))
    : 0;
  const physicalA   = data.battle_percentile != null ? Math.round(clamp01(data.battle_percentile / 100) * 100) : physicalFallbackA;
  const physicalRaw = data.battle_percentile != null ? `${data.battle_percentile.toFixed(0)}th pct`
    : data.hits_per60 != null ? `${data.hits_per60.toFixed(1)} hits/60`
    : "—";

  const chartData: RadarEntry[] = [
    { subject: "Offense",  A: offenseVal != null ? Math.round(clamp01(offenseVal / 8) * 100) : 0, raw: offenseRaw,   tip: "Expected goals generated per 60 min at 5v5. League avg ~4.1." },
    { subject: "Finish",   A: data.finishing != null ? Math.round(clamp01((data.finishing + 10) / 20) * 100) : 0,   raw: data.finishing != null ? `${data.finishing > 0 ? "+" : ""}${data.finishing.toFixed(1)} vs xG` : "—",       tip: "Goals above what shot quality predicts — pure finishing skill." },
    { subject: "Defense",  A: (data.cdr ?? data.rapm_ev_def) != null ? Math.round(clamp01(((data.cdr ?? data.rapm_ev_def ?? 0) + 2) / 4) * 100) : 0, raw: data.cdr != null ? `${data.cdr > 0 ? "+" : ""}${data.cdr.toFixed(2)} CDR` : "—", tip: "Composite Defensive Rating — shot suppression + RAPM defensive." },
    { subject: "Physical", A: physicalA, raw: physicalRaw, tip: "Puck battle percentile — hits, blocks, and contested zone battles. Uses hits+blocks/60 when percentile model data is unavailable." },
    { subject: "Sp.Teams", A: (data.special_teams_pp != null || data.special_teams_pk != null) ? Math.round(clamp01(((data.special_teams_pp ?? 0) + (data.special_teams_pk ?? 0) + 2) / 4) * 100) : 0, raw: data.special_teams_pp != null ? `PP ${data.special_teams_pp > 0 ? "+" : ""}${data.special_teams_pp.toFixed(2)}` : "—", tip: "Combined power play + penalty kill impact rating." },
    { subject: "Form",     A: data.hot_hand_score != null ? Math.round(clamp01((data.hot_hand_score + 2) / 4) * 100) : 0, raw: data.hot_hand_score != null ? `${data.hot_hand_score.toFixed(2)} hot hand` : "—", tip: "Recent form score — statistical hot/cold streak over last 5 games." },
  ];

  if (chartData.every(d => d.A === 0)) return null;

  const active = activeIdx !== null ? chartData[activeIdx] : null;

  return (
    <>
      {/* Viewport-aware fixed dims — avoids ResponsiveContainer drift */}
      <div style={{ lineHeight: 0 }}>
        <RadarChart
          width={chartSize} height={chartSize}
          cx={chartSize / 2} cy={chartSize / 2}
          outerRadius={Math.round(chartSize * 0.328)}
          data={chartData}
          margin={{ top: Math.round(chartSize * 0.078), right: Math.round(chartSize * 0.1), bottom: Math.round(chartSize * 0.078), left: Math.round(chartSize * 0.1) }}
          onMouseLeave={() => setActiveIdx(null)}
        >
          <PolarGrid stroke="rgba(255,255,255,0.07)" />
          <PolarAngleAxis
            dataKey="subject"
            tick={({ x, y, payload, index }: { x: number; y: number; payload: { value: string }; index: number }) => {
              const isActive = activeIdx === index;
              return (
                <text
                  x={x} y={y}
                  textAnchor="middle" dominantBaseline="central"
                  fontSize={10}
                  fontWeight={isActive ? 900 : 600}
                  fill={isActive ? teamColor : "rgba(255,255,255,0.40)"}
                  style={{ cursor: "pointer" }}
                  onClick={() => setActiveIdx(activeIdx === index ? null : index)}
                  onMouseEnter={() => setActiveIdx(index)}
                >
                  {payload.value}
                </text>
              );
            }}
          />
          <Radar
            name="Player"
            dataKey="A"
            stroke={teamColor}
            fill={teamColor}
            fillOpacity={0.20}
            strokeWidth={1.5}
            dot={(props: { cx: number; cy: number; index: number }) => {
              const isActive = activeIdx === props.index;
              return (
                <circle
                  key={props.index}
                  cx={props.cx} cy={props.cy}
                  r={isActive ? 5 : 3}
                  fill={isActive ? teamColor : `${teamColor}99`}
                  stroke={isActive ? "rgba(255,255,255,0.6)" : "none"}
                  strokeWidth={1.5}
                  style={{ cursor: "pointer" }}
                  onMouseEnter={() => setActiveIdx(props.index)}
                  onClick={() => setActiveIdx(activeIdx === props.index ? null : props.index)}
                />
              );
            }}
          />
        </RadarChart>
      </div>
      {/* Active stat callout */}
      <div className="min-h-[2.75rem] flex flex-col items-center justify-center gap-1 pt-1">
        {active ? (
          <>
            <span className="text-[11px] font-black tracking-wide" style={{ color: teamColor }}>{active.subject}</span>
            <span className="text-[12px] font-semibold text-white/70">{active.raw}</span>
          </>
        ) : (
          <span className="text-[10px] text-white/20">hover an axis or dot</span>
        )}
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Goalie Radar Chart
// ---------------------------------------------------------------------------

function GoalieRadarChart({ data, teamColor, nhlSvPct, nhlGaa }: { data: ProfileData; teamColor: string; nhlSvPct?: number; nhlGaa?: number }) {
  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  const [chartSize, setChartSize] = useState(300);
  useEffect(() => {
    const update = () => setChartSize(Math.min(300, window.innerWidth - 56));
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  // sv_pct fields are stored as 0-1 decimals (e.g. 0.8995, 0.7413)
  // Normalization floors/ranges calibrated against real MoneyPuck league data
  const svFull = data.sv_pct ?? nhlSvPct;
  const chartData: RadarEntry[] = [
    {
      subject: "GSAx",
      // Range: -15 (replacement level) to +20 (elite season)
      A: data.gsax != null ? Math.round(clamp01((data.gsax + 15) / 35) * 100) : 0,
      raw: data.gsax != null ? `${data.gsax > 0 ? "+" : ""}${data.gsax.toFixed(1)} GSAx` : "—",
      tip: "Goals Saved Above Expected — extra saves vs an average goalie. +15 is an elite season.",
    },
    {
      subject: "HD Sv%",
      // League range: ~0.65 (poor) to 0.88 (elite). Avg ~0.80
      A: data.hdsv_pct != null ? Math.round(clamp01((data.hdsv_pct - 0.65) / 0.23) * 100) : 0,
      raw: data.hdsv_pct != null ? `${(data.hdsv_pct * 100).toFixed(1)}% HD` : "—",
      tip: "High-danger save % on close-range slot shots. League avg ~80%. Most predictive goalie metric.",
    },
    {
      subject: "Sv%",
      // League range: ~0.880 (poor) to 0.935 (elite)
      A: svFull != null ? Math.round(clamp01((svFull - 0.880) / 0.055) * 100) : 0,
      raw: svFull != null ? `.${Math.round(svFull * 1000)} Sv%` : "—",
      tip: "Overall save percentage across all shot types.",
    },
    {
      subject: "Mid-D",
      // League range: ~0.82 (poor) to 0.96 (elite). Avg ~0.87
      A: data.mdsv_pct != null ? Math.round(clamp01((data.mdsv_pct - 0.82) / 0.14) * 100) : 0,
      raw: data.mdsv_pct != null ? `${(data.mdsv_pct * 100).toFixed(1)}% mid-D` : "—",
      tip: "Mid-danger save % on shots from outside the slot. League avg ~87%.",
    },
    {
      subject: "Low-D",
      // League range: ~0.92 (poor) to 0.995 (elite). Avg ~0.965
      A: data.ldsv_pct != null ? Math.round(clamp01((data.ldsv_pct - 0.92) / 0.075) * 100) : 0,
      raw: data.ldsv_pct != null ? `${(data.ldsv_pct * 100).toFixed(1)}% low-D` : "—",
      tip: "Low-danger save % on perimeter shots. Should be 96%+ for NHL starters.",
    },
    {
      subject: "Durability",
      // games_played / 65 (full season). Starter = 50+ GP → top of range
      A: data.games_played != null ? Math.round(clamp01(data.games_played / 65) * 100) : 0,
      raw: data.games_played != null ? `${data.games_played} GP` : "—",
      tip: "Games played this season — proxy for starter role and durability.",
    },
  ];

  if (chartData.every(d => d.A === 0)) return null;

  const active = activeIdx !== null ? chartData[activeIdx] : null;

  return (
    <>
      <div style={{ lineHeight: 0 }}>
        <RadarChart
          width={chartSize} height={chartSize}
          cx={chartSize / 2} cy={chartSize / 2}
          outerRadius={Math.round(chartSize * 0.328)}
          data={chartData}
          margin={{ top: Math.round(chartSize * 0.078), right: Math.round(chartSize * 0.1), bottom: Math.round(chartSize * 0.078), left: Math.round(chartSize * 0.1) }}
          onMouseLeave={() => setActiveIdx(null)}
        >
          <PolarGrid stroke="rgba(255,255,255,0.07)" />
          <PolarAngleAxis
            dataKey="subject"
            tick={({ x, y, payload, index }: { x: number; y: number; payload: { value: string }; index: number }) => {
              const isActive = activeIdx === index;
              return (
                <text
                  x={x} y={y}
                  textAnchor="middle" dominantBaseline="central"
                  fontSize={10}
                  fontWeight={isActive ? 900 : 600}
                  fill={isActive ? teamColor : "rgba(255,255,255,0.40)"}
                  style={{ cursor: "pointer" }}
                  onClick={() => setActiveIdx(activeIdx === index ? null : index)}
                  onMouseEnter={() => setActiveIdx(index)}
                >
                  {payload.value}
                </text>
              );
            }}
          />
          <Radar
            name="Goalie"
            dataKey="A"
            stroke={teamColor}
            fill={teamColor}
            fillOpacity={0.20}
            strokeWidth={1.5}
            dot={(props: { cx: number; cy: number; index: number }) => {
              const isActive = activeIdx === props.index;
              return (
                <circle
                  key={props.index}
                  cx={props.cx} cy={props.cy}
                  r={isActive ? 5 : 3}
                  fill={isActive ? teamColor : `${teamColor}99`}
                  stroke={isActive ? "rgba(255,255,255,0.6)" : "none"}
                  strokeWidth={1.5}
                  style={{ cursor: "pointer" }}
                  onMouseEnter={() => setActiveIdx(props.index)}
                  onClick={() => setActiveIdx(activeIdx === props.index ? null : props.index)}
                />
              );
            }}
          />
        </RadarChart>
      </div>
      <div className="min-h-[2.75rem] flex flex-col items-center justify-center gap-1 pt-1">
        {active ? (
          <>
            <span className="text-[11px] font-black tracking-wide" style={{ color: teamColor }}>{active.subject}</span>
            <span className="text-[12px] font-semibold text-white/70">{active.raw}</span>
          </>
        ) : (
          <span className="text-[10px] text-white/20">hover an axis or dot</span>
        )}
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Game-by-game bar chart
// ---------------------------------------------------------------------------

function GameLogChart({ games, teamColor }: { games: GameEntry[]; teamColor: string }) {
  const recent = [...games].slice(0, 10).reverse();
  if (recent.length < 3) return null;

  const chartData = recent.map((g, i) => ({
    game: i + 1,
    date: g.date?.slice(5) ?? "",
    opp:  g.opponent ?? "",
    g:    g.goals,
    a:    g.assists,
    pts:  g.goals + g.assists,
    pm:   g.plus_minus ?? 0,
  }));

  const maxPts = Math.max(...chartData.map(d => d.pts), 1);

  return (
    <ResponsiveContainer width="100%" height={130}>
      <BarChart data={chartData} barGap={2} barSize={18}
        margin={{ top: 4, right: 4, left: -28, bottom: 0 }}>
        <XAxis
          dataKey="opp"
          tick={{ fill: "rgba(255,255,255,0.30)", fontSize: 8 }}
          axisLine={false} tickLine={false}
        />
        <YAxis
          domain={[0, maxPts + 1]}
          tick={{ fill: "rgba(255,255,255,0.25)", fontSize: 8 }}
          axisLine={false} tickLine={false}
          allowDecimals={false}
        />
        <Tooltip
          cursor={{ fill: "rgba(255,255,255,0.04)" }}
          contentStyle={{ background: "#0f1114", border: `1px solid ${teamColor}30`, borderRadius: 8, fontSize: 10 }}
          labelStyle={{ color: "rgba(255,255,255,0.5)", marginBottom: 2 }}
          formatter={(val: number, name: string) => [val, name === "g" ? "G" : name === "a" ? "A" : name]}
        />
        <Bar dataKey="a" stackId="pts" fill="rgba(255,255,255,0.20)" radius={[0,0,0,0]} />
        <Bar dataKey="g" stackId="pts" fill={teamColor} fillOpacity={0.75} radius={[3,3,0,0]}>
          {chartData.map((_, idx) => (
            <Cell key={idx} fill={teamColor} fillOpacity={chartData[idx].pts > 0 ? 0.80 : 0.25} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------------------
// EWMA trend line chart
// ---------------------------------------------------------------------------

function EwmaTrendChart({ xgf60, leagueAvg = 4.09, teamColor }: {
  xgf60: number | null; leagueAvg?: number; teamColor: string;
}) {
  if (xgf60 == null) return null;
  // Simulate 8-week trend converging to current value (visual only — real data would come from backend)
  const trend = Array.from({ length: 8 }, (_, i) => {
    const weight = i / 7;
    const noise  = (Math.sin(i * 2.3) * 0.4);
    return {
      week: `W${i + 1}`,
      xgf: parseFloat((leagueAvg + (xgf60 - leagueAvg) * weight + noise * (1 - weight)).toFixed(2)),
    };
  });
  trend[7].xgf = xgf60;

  const isHot  = xgf60 > leagueAvg + 0.5;
  const isCold = xgf60 < leagueAvg - 0.5;
  const lineColor = isHot ? "#4ade80" : isCold ? "#f87171" : teamColor;

  return (
    <ResponsiveContainer width="100%" height={100}>
      <LineChart data={trend} margin={{ top: 4, right: 8, left: -28, bottom: 0 }}>
        <XAxis dataKey="week" tick={{ fill: "rgba(255,255,255,0.25)", fontSize: 8 }} axisLine={false} tickLine={false} />
        <YAxis domain={["auto","auto"]} tick={{ fill: "rgba(255,255,255,0.20)", fontSize: 8 }} axisLine={false} tickLine={false} />
        <ReferenceLine y={leagueAvg} stroke="rgba(255,255,255,0.15)" strokeDasharray="3 3" />
        <Tooltip
          contentStyle={{ background: "#0f1114", border: `1px solid ${lineColor}30`, borderRadius: 8, fontSize: 10 }}
          labelStyle={{ color: "rgba(255,255,255,0.4)" }}
          formatter={(v: number) => [v.toFixed(2), "xGF/60"]}
        />
        <Line type="monotone" dataKey="xgf" stroke={lineColor} strokeWidth={2}
          dot={false} activeDot={{ r: 3, fill: lineColor }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------------------
// Shot / zone visualizations
// ---------------------------------------------------------------------------

interface ShotPoint { x: number; y: number; xg: number; goal: boolean; type: string; }

/** Half-rink SVG shot map — dots coloured by xG, goals as star markers */
function ShotMapViz({ shots, teamColor }: { shots: ShotPoint[]; teamColor: string }) {
  // NHL coords: x 0→100 (offensive half), y -42.5→42.5
  // SVG viewport: 400w × 340h, goal at x=89 mapped to SVG ~340px
  const W = 400; const H = 340;
  // map rink x [25,100] → SVG x [0,W], rink y [-42.5,42.5] → SVG y [0,H]
  const rx = (x: number) => ((x - 25) / 75) * W;
  const ry = (y: number) => ((y + 42.5) / 85) * H;

  const xgColor = (xg: number) => {
    if (xg >= 0.2) return "#ef4444";   // high danger — red
    if (xg >= 0.08) return "#f97316";  // medium — orange
    return "#3b82f6";                   // low — blue
  };

  // Deduplicate / limit to last 400 shots for performance
  const pts = shots.slice(-400);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-[360px] mx-auto" style={{ filter: "drop-shadow(0 2px 12px rgba(0,0,0,0.6))" }}>
      {/* Rink outline */}
      <rect x={0} y={0} width={W} height={H} rx={8} fill="#0a0d12" stroke="rgba(255,255,255,0.07)" strokeWidth={1} />
      {/* Centre ice line */}
      <line x1={0} y1={H/2} x2={W} y2={H/2} stroke="rgba(255,255,255,0.05)" strokeWidth={1} />
      {/* Blue line at rink x=25 → SVG x=0 edge, so we mark rink x=25 */}
      {/* Goal line at rink x=89 */}
      <line x1={rx(89)} y1={0} x2={rx(89)} y2={H} stroke="rgba(239,68,68,0.25)" strokeWidth={1.5} strokeDasharray="4 3" />
      {/* Crease arc */}
      <path d={`M ${rx(89)} ${ry(-3)} A ${(6/85)*H} ${(6/85)*H} 0 0 1 ${rx(89)} ${ry(3)}`}
        fill="none" stroke="rgba(239,68,68,0.3)" strokeWidth={1.5} />
      {/* Faceoff circles */}
      {[[-22, 20], [-22, -20], [0, 0]].map(([cx, cy], i) => (
        <circle key={i} cx={rx(cx + 25)} cy={ry(cy)} r={(15/85)*H}
          fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth={1} />
      ))}
      {/* Shot dots */}
      {pts.map((s, i) => {
        const sx = rx(s.x); const sy = ry(s.y);
        if (s.goal) {
          // Star marker for goals
          const r1 = 6; const r2 = 3; const arms = 5;
          const pts2 = Array.from({ length: arms * 2 }, (_, j) => {
            const angle = (j * Math.PI) / arms - Math.PI / 2;
            const r = j % 2 === 0 ? r1 : r2;
            return `${sx + r * Math.cos(angle)},${sy + r * Math.sin(angle)}`;
          }).join(" ");
          return <polygon key={i} points={pts2} fill={teamColor} opacity={0.9}
            stroke="rgba(255,255,255,0.4)" strokeWidth={0.5} />;
        }
        return <circle key={i} cx={sx} cy={sy} r={3.5}
          fill={xgColor(s.xg)} opacity={0.55} />;
      })}
      {/* Legend */}
      <circle cx={12} cy={H - 36} r={3.5} fill="#3b82f6" opacity={0.7} />
      <text x={20} y={H - 32} fontSize={8} fill="rgba(255,255,255,0.35)">Low xG</text>
      <circle cx={12} cy={H - 22} r={3.5} fill="#f97316" opacity={0.7} />
      <text x={20} y={H - 18} fontSize={8} fill="rgba(255,255,255,0.35)">Med xG</text>
      <circle cx={12} cy={H - 8} r={3.5} fill="#ef4444" opacity={0.7} />
      <text x={20} y={H - 4} fontSize={8} fill="rgba(255,255,255,0.35)">High xG</text>
      {/* Goal star legend */}
      <polygon points={`${W-50},${H-10} ${W-48},${H-15} ${W-46},${H-10} ${W-52},${H-13} ${W-44},${H-13}`}
        fill={teamColor} opacity={0.9} />
      <text x={W - 42} y={H - 5} fontSize={8} fill="rgba(255,255,255,0.35)">Goal</text>
    </svg>
  );
}

/** Offensive zone tendency map — 5-zone coloured cells */
function ZoneTendencyMap({ data, teamColor }: { data: ProfileData; teamColor: string }) {
  const slot    = data.nn_shoot_slot_pct    ?? 0;
  const perim   = data.nn_shoot_perimeter_pct ?? 0;
  const net     = data.nn_drive_net_pct     ?? 0;
  const corner  = data.nn_battle_corner_pct ?? 0;
  const hold    = data.nn_hold_corner_pct   ?? 0;

  const zones = [
    { label: "Net Front",  pct: net,    x: 155, y: 100, w: 60,  h: 60,  color: "#fbbf24" },
    { label: "Slot",       pct: slot,   x: 115, y: 60,  w: 140, h: 85,  color: teamColor  },
    { label: "Perimeter",  pct: perim,  x: 10,  y: 10,  w: 340, h: 200, color: "#94a3b8"  },
    { label: "L Corner",   pct: corner, x: 10,  y: 150, w: 90,  h: 70,  color: "#38bdf8"  },
    { label: "R Corner",   pct: hold,   x: 260, y: 150, w: 90,  h: 70,  color: "#38bdf8"  },
  ];

  return (
    <svg viewBox="0 0 360 230" className="w-full max-w-[340px] mx-auto">
      {/* Background */}
      <rect x={0} y={0} width={360} height={230} rx={8} fill="#0a0d12" stroke="rgba(255,255,255,0.07)" strokeWidth={1} />
      {/* Rink outline */}
      <rect x={10} y={10} width={340} height={200} rx={12} fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth={1.5} />
      {/* Goal crease */}
      <rect x={160} y={160} width={40} height={40} rx={4} fill="rgba(239,68,68,0.12)" stroke="rgba(239,68,68,0.3)" strokeWidth={1} />
      {/* Zone overlays — rendered back-to-front so smaller zones appear on top */}
      {[zones[2], zones[0], zones[3], zones[4], zones[1]].map((z, i) => (
        <g key={i}>
          <rect x={z.x} y={z.y} width={z.w} height={z.h} rx={4}
            fill={z.color} fillOpacity={Math.min(z.pct / 45, 1) * 0.35}
            stroke={z.color} strokeOpacity={Math.min(z.pct / 45, 1) * 0.6} strokeWidth={1} />
          <text x={z.x + z.w / 2} y={z.y + z.h / 2 - 4} textAnchor="middle" fontSize={7}
            fill="rgba(255,255,255,0.45)" fontWeight="bold" letterSpacing="1">
            {z.label.toUpperCase()}
          </text>
          <text x={z.x + z.w / 2} y={z.y + z.h / 2 + 9} textAnchor="middle" fontSize={10}
            fill="rgba(255,255,255,0.75)" fontWeight="bold">
            {z.pct.toFixed(0)}%
          </text>
        </g>
      ))}
      {/* Goal line */}
      <line x1={10} y1={170} x2={350} y2={170} stroke="rgba(239,68,68,0.2)" strokeWidth={1} strokeDasharray="3 2" />
    </svg>
  );
}

/** Full-rink skating zone time split */
function SkatingZoneMap({ data }: { data: ProfileData }) {
  const oz = data.skating_zone_time_oz_pct ?? 0;
  const dz = data.skating_zone_time_dz_pct ?? 0;
  const nz = Math.max(0, 100 - oz - dz);

  const zones = [
    { label: "Offensive Zone", pct: oz, color: "#4ade80", y: 10, h: 90 },
    { label: "Neutral Zone",   pct: nz, color: "#fbbf24", y: 105, h: 50 },
    { label: "Defensive Zone", pct: dz, color: "#f87171", y: 160, h: 90 },
  ];

  return (
    <svg viewBox="0 0 200 260" className="w-full max-w-[180px] mx-auto">
      <rect x={0} y={0} width={200} height={260} rx={8} fill="#0a0d12" stroke="rgba(255,255,255,0.07)" strokeWidth={1} />
      {/* Rink outline */}
      <rect x={20} y={10} width={160} height={240} rx={14}
        fill="none" stroke="rgba(255,255,255,0.10)" strokeWidth={1.5} />
      {/* Blue lines */}
      <line x1={20} y1={100} x2={180} y2={100} stroke="rgba(99,179,237,0.35)" strokeWidth={1.5} />
      <line x1={20} y1={160} x2={180} y2={160} stroke="rgba(99,179,237,0.35)" strokeWidth={1.5} />
      {/* Red line */}
      <line x1={20} y1={130} x2={180} y2={130} stroke="rgba(239,68,68,0.25)" strokeWidth={1.5} />
      {/* Zone fills */}
      {zones.map((z, i) => (
        <g key={i}>
          <rect x={20} y={z.y} width={160} height={z.h} rx={i === 0 ? 14 : i === 2 ? 14 : 0}
            fill={z.color} fillOpacity={Math.min(z.pct / 45, 1) * 0.22} />
          <text x={100} y={z.y + z.h / 2 - 5} textAnchor="middle" fontSize={7}
            fill="rgba(255,255,255,0.40)" fontWeight="bold" letterSpacing="1">
            {z.label.toUpperCase()}
          </text>
          <text x={100} y={z.y + z.h / 2 + 10} textAnchor="middle" fontSize={13}
            fill={z.color} fillOpacity={0.85} fontWeight="bold">
            {z.pct.toFixed(0)}%
          </text>
        </g>
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Main Profile Page
// ---------------------------------------------------------------------------

export default function PlayerProfilePage() {
  const params = useParams();
  const router = useRouter();
  const { theme, cortexPinned, setPreviewTheme } = useTheme();
  // The route param is a URL-encoded player name (e.g. "Nathan%20MacKinnon").
  // Normalize accents so old bookmarks / direct URLs with ý, é, etc. still resolve.
  const playerName = normalizePlayerName(decodeURIComponent(params.id as string));

  const [data, setData] = useState<ProfileData | null>(null);
  const [loading, setLoading] = useState(true);
  const [imgErr, setImgErr] = useState(false);
  const [shots, setShots] = useState<ShotPoint[]>([]);
  // Current-season stats fetched from nhl-team route
  const [nhlStats, setNhlStats] = useState<{
    gp: number; goals: number; assists: number; points: number; plus_minus: number;
    jersey?: number | null;
    // goalie
    wins?: number; losses?: number; ot_losses?: number; gaa?: number; sv_pct?: number; shutouts?: number;
  } | null>(null);
  // Injury status for this player (from team injuries endpoint)
  const [injuryBadge, setInjuryBadge] = useState<string | null>(null);
  // Bio data from NHL landing endpoint
  const [bio, setBio] = useState<{
    height_cm: number | null;
    weight_kg: number | null;
    birth_date: string | null;
    birthplace: string | null;
    shoots_catches: string | null;
    jersey_number: number | null;
    draft_year: number | null;
    draft_round: number | null;
    draft_pick: number | null;
    draft_overall: number | null;
    draft_team: string | null;
    current_team_abbrev: string | null;
  } | null>(null);
  // Live team override — NHL API is source of truth for current team
  const [liveTeam, setLiveTeam] = useState<string | null>(null);

  // Contract data from CapWages
  const [contract, setContract] = useState<{
    cap_hit: number | null;
    contract_type: string;
    expiry_status: string;
    expiry_year: number | null;
    years_remaining: number | null;
  } | null>(null);

  // Search bar state
  const [searchQ, setSearchQ]       = useState("");
  const [allPlayers, setAllPlayers] = useState<{ name: string; team: string; position: string }[]>([]);
  const [showSugg, setShowSugg]     = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch("/api/phase2/players")
      .then(r => r.json())
      .then(d => setAllPlayers(d.players ?? []))
      .catch(() => {});
  }, []);

  const suggestions = searchQ.trim().length >= 2
    ? allPlayers.filter(p => p.name.toLowerCase().includes(searchQ.trim().toLowerCase())).slice(0, 8)
    : [];

  function goToPlayer(name: string) {
    setSearchQ("");
    setShowSugg(false);
    router.push(`/players/${encodeURIComponent(normalizePlayerName(name))}`);
  }

  useEffect(() => {
    if (!playerName) return;
    setLoading(true);

    fetch(`/api/phase2/player?name=${encodeURIComponent(playerName)}`)
      .then(r => r.json())
      .then((d) => {
        if (!d.not_found) {
          setData({
            ...d,
            shoots_catches: d.shoots ?? null,
          } as ProfileData);
        }
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [playerName]);

  // Apply the player's team theme site-wide as a preview (reverts on navigate away).
  // Re-fires when liveTeam overrides a stale model team (e.g. traded player).
  // Cortex pin takes priority — don't override it.
  useEffect(() => {
    const t = liveTeam ?? data?.team;
    if (!t || cortexPinned) return;
    const primary   = TEAM_COLORS[t]    ?? "#94a3b8";
    const secondary = TEAM_SECONDARY?.[t] ?? primary;
    setPreviewTheme({ abbrev: t, primaryColor: primary, secondaryColor: secondary, logoUrl: logoUrl(t) });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data?.team, liveTeam]);

  // Once we know the team, fetch current-season stats and injury status
  useEffect(() => {
    if (!data?.team) return;
    const team = data.team;
    const fullName = normalizePlayerName(data.player_name ?? "").toLowerCase();

    // Current-season stats + jersey from nhl-team route
    fetch(`/api/nhl-team/${team}`)
      .then(r => r.json())
      .then(d => {
        const all = [...(d.skaters ?? []), ...(d.goalies ?? [])] as Record<string, unknown>[];
        const match = all.find(p => {
          const fn = normalizePlayerName(String(p.first_name ?? "")).toLowerCase();
          const ln = normalizePlayerName(String(p.last_name ?? "")).toLowerCase();
          return fullName.includes(fn) && fullName.includes(ln);
        });
        if (match) {
          const isG = (d.goalies ?? []).includes(match);
          setNhlStats({
            gp:         Number(match.gp ?? 0),
            goals:      Number(match.goals ?? 0),
            assists:    Number(match.assists ?? 0),
            points:     Number(match.points ?? 0),
            plus_minus: Number(match.plus_minus ?? 0),
            jersey:     match.jersey != null ? Number(match.jersey) : null,
            ...(isG ? {
              wins:      Number(match.wins ?? 0),
              losses:    Number(match.losses ?? 0),
              ot_losses: Number(match.ot_losses ?? 0),
              gaa:       Number(match.gaa ?? 0),
              sv_pct:    Number(match.sv_pct ?? 0),
              shutouts:  Number(match.shutouts ?? 0),
            } : {}),
          });
          const inj = String(match.injury_status ?? "");
          if (inj) setInjuryBadge(inj);
        }
      })
      .catch(() => {});

    // Contract data from CapWages via puckpedia proxy
    fetch(`/api/puckpedia/${team}`)
      .then(r => r.json())
      .then(d => {
        const players = (d.players ?? []) as { name: string; cap_hit: number | null; contract_type: string; expiry_status: string; expiry_year: number | null; years_remaining: number | null }[];
        const last = fullName.split(" ").pop() ?? "";
        const first = fullName.split(" ")[0] ?? "";
        const match = players.find(p => {
          const pn = p.name.toLowerCase();
          return pn.includes(last) && pn.includes(first);
        });
        if (match) setContract(match);
      })
      .catch(() => {});

    // Injury report for this team
    fetch(`/api/injuries/${team}`)
      .then(r => r.json())
      .then(d => {
        const injuries = (d.injuries ?? []) as { player_name: string; status: string }[];
        const found = injuries.find(i => {
          const n = (i.player_name ?? "").toLowerCase();
          return fullName.includes(n.split(" ")[0]) && fullName.includes(n.split(" ").pop() ?? "");
        });
        if (found) setInjuryBadge(found.status);
      })
      .catch(() => {});
  }, [data?.team, data?.player_name]);

  // Fetch bio from NHL landing endpoint once we have a player ID
  useEffect(() => {
    if (!data?.player_id) return;
    fetch(`/api/player-nhl/${data.player_id}`)
      .then(r => r.json())
      .then(d => {
        if (!d.error) {
          setBio(d);
          // If NHL API reports a different team than the model parquet, use the live one
          if (d.current_team_abbrev && d.current_team_abbrev !== data?.team) {
            setLiveTeam(d.current_team_abbrev);
          }
        }
      })
      .catch(() => {});
  }, [data?.player_id]);

  // Fetch shot data for shot map visualization
  useEffect(() => {
    if (!data?.player_id) return;
    fetch(`/api/player-shots/${data.player_id}`)
      .then(r => r.json())
      .then(d => { if (d.shots?.length) setShots(d.shots); })
      .catch(() => {});
  }, [data?.player_id]);

  if (loading) {
    return (
      <main className="min-h-screen p-4 sm:p-6 max-w-3xl mx-auto w-full overflow-x-hidden">
        <div className="mb-5 flex items-center gap-3">
          <div className="h-24 w-24 rounded-full bg-white/[0.06] animate-pulse shrink-0" />
          <div className="space-y-2 flex-1">
            <div className="h-7 bg-white/[0.08] rounded w-48 animate-pulse" />
            <div className="h-4 bg-white/[0.05] rounded w-32 animate-pulse" />
          </div>
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          {[0,1,2,3,4,5].map(i => <SkeletonCard key={i} />)}
        </div>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="min-h-screen p-4 sm:p-6 max-w-3xl mx-auto w-full overflow-x-hidden flex items-center justify-center">
        <div className="flex flex-col items-center gap-5 text-center max-w-sm">
          <div className="rounded-2xl border border-white/[0.08] bg-white/[0.02] px-6 py-8 space-y-3">
            <p className="text-[15px] font-bold text-white/60">Player not found</p>
            <p className="text-[12px] text-white/30 leading-relaxed">
              <span className="text-white/50 font-semibold">{playerName}</span> may be in the minors, a prospect with little to no NHL games played, or not yet in our model data.
            </p>
            <p className="text-[10px] text-white/18">Try searching from the team&apos;s Depth Chart tab.</p>
          </div>
          <button
            onClick={() => router.back()}
            className="flex items-center gap-2 px-4 py-2 rounded-full border border-white/[0.12] bg-white/[0.04] text-[11px] font-semibold text-white/50 hover:text-white/80 hover:border-white/[0.22] hover:bg-white/[0.07] transition-all duration-150"
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <path d="M8 2L4 6L8 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Go back
          </button>
        </div>
      </main>
    );
  }

  const isGoalie = data.is_goalie;
  const sy = data.season ?? (new Date().getFullYear() - 1);
  const headshotUrl = !imgErr && data.player_id && data.team
    ? `https://assets.nhle.com/mugs/nhl/${sy}${sy + 1}/${data.team}/${data.player_id}.png`
    : null;
  // Theme priority: player's own team colour by default.
  // If cortexPinned (user explicitly set the Cortex theme), cortex purple wins.
  // A globally pinned team theme does NOT override the player's own team colour —
  // the player page always reflects the player's identity first.
  const CORTEX_PRIMARY   = "#a78bfa";
  const CORTEX_SECONDARY = "#2e1065";
  // liveTeam overrides the stale model team when NHL API reports a trade
  const displayTeam         = liveTeam ?? data.team ?? "";
  const playerTeamColor     = TEAM_COLORS[displayTeam]    ?? "#94a3b8";
  const playerTeamSecondary = TEAM_SECONDARY?.[displayTeam] ?? "#0f0a1e";
  const teamColor     = cortexPinned ? CORTEX_PRIMARY   : playerTeamColor;
  const teamSecondary = cortexPinned ? CORTEX_SECONDARY : playerTeamSecondary;
  // Dark theme colors — same logic as team page
  const teamDarkBg     = darkBlend(darkerOfPP(teamColor, teamSecondary), 0.92);
  const cardStyle: React.CSSProperties = {
    border: `1.5px solid ${teamColor}30`,
    background: `linear-gradient(160deg, ${teamDarkBg} 0%, #060708 70%)`,
    boxShadow: `0 4px 24px rgba(0,0,0,0.5), 0 0 0 1px ${teamColor}08`,
  };
  const headshotRing = {
    boxShadow: `0 0 0 3px ${teamColor}, 0 0 0 5px rgba(255,255,255,0.18), 0 0 30px ${teamColor}cc, 0 0 60px ${teamColor}66, 0 0 100px ${teamColor}33, 0 12px 48px rgba(0,0,0,0.85)`,
  };

  const ewmaAbove  = data.ewma_xgf60 != null ? data.ewma_xgf60 - 4.09 : null;
  const playerType = derivePlayerType(data, nhlStats);

  // "Why hot" blurb
  let formBlurb: string | null = null;
  const gl = data.game_log;
  if (!isGoalie && gl && gl.summary.n_games > 0) {
    const s = gl.summary;
    const score = data.hot_hand_score ?? 0;
    if (score > 0.5 || (data.ewma_form_flag === "rising")) {
      formBlurb = `${s.goals}G ${s.assists}A in last ${s.n_games} games`;
      if (data.hot_hand_xg5 && data.hot_hand_xg5 > 0)
        formBlurb += ` · ${data.hot_hand_xg5.toFixed(1)} xG (hot hand active)`;
    } else if (data.ewma_form_flag === "falling") {
      formBlurb = `${s.goals}G ${s.assists}A in last ${s.n_games} games`;
      if (ewmaAbove != null && ewmaAbove < -1)
        formBlurb += ` · generating ${Math.abs(ewmaAbove).toFixed(1)} below league average`;
    }
  }

  // Derive behavioral style summary from NN
  let playStyle: string | null = null;
  const carry = data.nn_carry_in_pct ?? 0;
  const slot  = data.nn_shoot_slot_pct ?? 0;
  const net   = data.nn_drive_net_pct ?? 0;
  const dump  = data.nn_dump_pct ?? 0;
  if (carry > 0 || slot > 0) {
    const parts: string[] = [];
    if (carry > 35) parts.push("prefers carrying the puck in");
    else if (dump > 35) parts.push("tends to dump and chase");
    if (slot > 30) parts.push("shoots from high-danger areas");
    else if (data.nn_shoot_perimeter_pct && data.nn_shoot_perimeter_pct > 30) parts.push("perimeter shooter");
    if (net > 20) parts.push("drives the net hard");
    if (parts.length > 0) playStyle = parts.join(" · ");
  }

  return (
    <main className="min-h-screen p-4 sm:p-6 max-w-3xl mx-auto w-full overflow-x-hidden w-full overflow-x-hidden">

      {/* ── Search bar — team-colored ── */}
      <div className="flex justify-center mb-5">
        <div className="relative w-full max-w-md">
          <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-sm pointer-events-none select-none"
            style={{ color: `${teamColor}60` }}>⌕</span>
          <input
            ref={searchRef}
            value={searchQ}
            onChange={e => { setSearchQ(e.target.value); setShowSugg(true); }}
            onKeyDown={e => {
              if (e.key === "Enter" && suggestions.length > 0) goToPlayer(suggestions[0].name);
              if (e.key === "Enter" && searchQ.trim() && suggestions.length === 0) goToPlayer(searchQ.trim());
              if (e.key === "Escape") { setShowSugg(false); searchRef.current?.blur(); }
            }}
            onFocus={() => setShowSugg(true)}
            onBlur={() => setTimeout(() => setShowSugg(false), 150)}
            placeholder="Search any NHL player…"
            className="w-full pl-9 pr-9 py-2.5 text-sm rounded-xl text-white/85 placeholder:text-white/20 focus:outline-none transition-all"
            style={{
              background: `${teamDarkBg}cc`,
              border: `1px solid ${teamColor}35`,
            }}
          />
          {searchQ && (
            <button
              onClick={() => { setSearchQ(""); searchRef.current?.focus(); }}
              className="absolute right-3.5 top-1/2 -translate-y-1/2 text-white/20 hover:text-white/50 transition-colors text-xs"
            >✕</button>
          )}
          {showSugg && suggestions.length > 0 && (
            <div className="absolute top-full left-0 right-0 mt-1 rounded-xl overflow-hidden z-50"
              style={{ border: `1px solid ${teamColor}25`, background: "#0f1114", boxShadow: `0 8px_32px rgba(0,0,0,0.8), 0 0 20px ${teamColor}10` }}>
              {suggestions.map((p, i) => (
                <button
                  key={i}
                  className="w-full flex items-center gap-3 px-3.5 py-2.5 text-left hover:bg-white/[0.04] transition-colors border-b border-white/[0.05] last:border-0"
                  onMouseDown={() => goToPlayer(p.name)}
                >
                  {p.team && <TeamLogo team={p.team} size={20} />}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-white/80 truncate">{p.name}</p>
                    <p className="text-[10px] text-white/30">{p.team} · {p.position === "L" ? "LW" : p.position === "R" ? "RW" : p.position}</p>
                  </div>
                  <span className="text-[10px] text-white/15 shrink-0">→</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Hero section — centered NHL.com style ── */}
      <div className="rounded-2xl overflow-hidden mb-4 shadow-[0_16px_60px_rgba(0,0,0,0.75)]"
        style={{ border: `1.5px solid ${teamColor}35`, background: `linear-gradient(175deg, ${teamDarkBg} 0%, #060708 60%)` }}>

        {/* Hero image strip */}
        {data.hero_image && (
          <div className="h-24 overflow-hidden relative">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={data.hero_image} alt="" className="w-full h-full object-cover object-top opacity-25" />
            <div className="absolute inset-0 bg-gradient-to-b from-transparent to-[#0d0f13]" />
          </div>
        )}

        {/* ── Centered hero body ── */}
        <div className={`flex flex-col items-center text-center px-5 pb-6 ${data.hero_image ? "-mt-10 relative z-10" : "pt-8"}`}>

          {/* Circular headshot — glow backdrop behind the ring */}
          <div className="relative mb-4 shrink-0">
            <div
              className="absolute inset-0 rounded-full pointer-events-none"
              style={{
                background: `radial-gradient(circle, ${teamColor}55 0%, ${teamColor}22 45%, transparent 72%)`,
                transform: "scale(1.6)",
                filter: "blur(16px)",
              }}
            />
          <div
            className="relative h-32 w-32 rounded-full overflow-hidden"
            style={{
              background: `radial-gradient(circle at 50% 40%, ${darkBlend(teamSecondary, 0.45)}, ${darkBlend(teamSecondary, 0.82)})`,
              ...headshotRing,
            }}
          >
            {headshotUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={headshotUrl} alt={data.player_name ?? ""} onError={() => setImgErr(true)}
                className="h-full w-full object-cover object-top scale-110 origin-top" />
            ) : (
              <div className="h-full w-full flex items-center justify-center">
                <span className="text-4xl font-bold text-white/30">
                  {(data.player_name ?? "?").split(" ").map((w: string) => w[0]).slice(0, 2).join("")}
                </span>
              </div>
            )}
          </div>
          </div>

          {/* Name */}
          <h1 className="text-3xl font-black text-white leading-tight tracking-tight mb-1">
            {data.player_name}
          </h1>

          {/* Team logo | #jersey | position — centered */}
          <div className="flex items-center justify-center gap-2 mb-2 flex-wrap">
            {displayTeam && (
              <button onClick={() => router.push(`/teams/${displayTeam}`)} className="shrink-0 hover:opacity-80 transition-opacity">
                <TeamLogo team={displayTeam} size={52} />
              </button>
            )}
            {(bio?.jersey_number ?? nhlStats?.jersey ?? data.jersey_number) != null && (
              <>
                <span className="text-white/20 text-xs">|</span>
                <span className="text-sm font-medium text-white/50">
                  #{bio?.jersey_number ?? nhlStats?.jersey ?? data.jersey_number}
                </span>
              </>
            )}
            {data.position && (
              <>
                <span className="text-white/20 text-xs">|</span>
                <span className="text-sm font-semibold text-white/50">
                  {data.position === "L" ? "LW" : data.position === "R" ? "RW" : data.position}
                </span>
              </>
            )}
          </div>

          {/* Archetype badge — centered */}
          {playerType && (
            <div className="mb-2">
              <span
                className="inline-flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest rounded-full px-3 py-1"
                style={{
                  color: "#fcd34d",
                  background: "linear-gradient(135deg, rgba(245,158,11,0.20) 0%, rgba(120,53,15,0.18) 100%)",
                  border: "1px solid rgba(245,158,11,0.45)",
                  boxShadow: "0 0 12px rgba(245,158,11,0.15)",
                }}
              >
                <span style={{ fontSize: 8 }}>★</span>
                {playerType}
              </span>
            </div>
          )}

          {/* Hot hand badge next to name area */}
          {!injuryBadge && (data.hot_hand_score ?? 0) > 0.5 && (
            <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-[#C9A84C] bg-[#C9A84C]/10 border border-[#C9A84C]/30 rounded-full px-2.5 py-0.5 mb-2">
              ⚡ Hot Hand
            </span>
          )}

          {/* Injury / form / rank badges — centered */}
          <div className="flex items-center justify-center gap-2 flex-wrap">
            {injuryBadge ? (
              <span
                className="inline-flex items-center gap-1 text-[10px] font-bold rounded-full px-2.5 py-0.5 uppercase tracking-wide"
                style={{
                  color: injuryBadge === "Out" || injuryBadge.startsWith("IR") ? "#f87171" : "#fbbf24",
                  backgroundColor: injuryBadge === "Out" || injuryBadge.startsWith("IR") ? "rgba(248,113,113,0.10)" : "rgba(251,191,36,0.10)",
                  border: `1px solid ${injuryBadge === "Out" || injuryBadge.startsWith("IR") ? "rgba(248,113,113,0.30)" : "rgba(251,191,36,0.30)"}`,
                }}
              >
                🚑 {injuryBadge}
              </span>
            ) : (
              <FormBadge flag={data.ewma_form_flag} ewmaDelta={ewmaAbove} />
            )}
            {data.war_rank && data.war_rank <= 30 && (
              <span className="text-[10px] font-semibold text-[#a78bfa] bg-[#a78bfa]/10 border border-[#a78bfa]/25 rounded-full px-2.5 py-0.5">
                🏆 #{data.war_rank} in NHL
              </span>
            )}
          </div>
        </div>

        {/* ── 2025-26 Season stats strip ── */}
        {nhlStats && (
          <div className="border-t px-5 py-3 flex items-center justify-center gap-4 flex-wrap" style={{ borderColor: `${teamColor}20` }}>
            <span className="text-[9px] text-white/25 uppercase tracking-widest font-semibold">2025-26</span>
            {isGoalie ? (
              <>
                {[
                  { label: "GP",  value: nhlStats.gp },
                  { label: "W",   value: nhlStats.wins ?? 0 },
                  { label: "L",   value: nhlStats.losses ?? 0 },
                  { label: "OTL", value: nhlStats.ot_losses ?? 0 },
                ].map(s => (
                  <div key={s.label} className="flex flex-col items-center min-w-[36px]">
                    <span className="text-sm font-bold tabular-nums text-white/85">{s.value}</span>
                    <span className="text-[8px] text-white/30 uppercase tracking-wider">{s.label}</span>
                  </div>
                ))}
                <div className="flex flex-col items-center min-w-[36px]">
                  <span className="text-base font-black tabular-nums" style={{ color: teamColor }}>
                    {nhlStats.sv_pct ? `.${Math.round(nhlStats.sv_pct * 1000)}` : "—"}
                  </span>
                  <span className="text-[8px] text-white/30 uppercase tracking-wider">SV%</span>
                </div>
                <div className="flex flex-col items-center min-w-[36px]">
                  <span className="text-sm font-bold tabular-nums text-white/85">
                    {nhlStats.gaa?.toFixed(2) ?? "—"}
                  </span>
                  <span className="text-[8px] text-white/30 uppercase tracking-wider">GAA</span>
                </div>
                <div className="flex flex-col items-center min-w-[36px]">
                  <span className="text-sm font-bold tabular-nums text-white/85">{nhlStats.shutouts ?? 0}</span>
                  <span className="text-[8px] text-white/30 uppercase tracking-wider">SO</span>
                </div>
              </>
            ) : (
              <>
                {[
                  { label: "GP",  value: nhlStats.gp },
                  { label: "G",   value: nhlStats.goals },
                  { label: "A",   value: nhlStats.assists },
                  { label: "PTS", value: nhlStats.points, bold: true },
                ].map(s => (
                  <div key={s.label} className="flex flex-col items-center min-w-[36px]">
                    <span className={`tabular-nums ${s.bold ? "text-base font-black" : "text-sm font-bold"}`}
                      style={{ color: s.bold ? teamColor : "rgba(255,255,255,0.85)" }}>
                      {s.value}
                    </span>
                    <span className="text-[8px] text-white/30 uppercase tracking-wider">{s.label}</span>
                  </div>
                ))}
                <div className="flex flex-col items-center min-w-[36px]">
                  <span className="text-sm font-bold tabular-nums"
                    style={{ color: nhlStats.plus_minus > 0 ? "#4ade80" : nhlStats.plus_minus < 0 ? "#f87171" : "rgba(255,255,255,0.40)" }}>
                    {nhlStats.plus_minus > 0 ? `+${nhlStats.plus_minus}` : nhlStats.plus_minus}
                  </span>
                  <span className="text-[8px] text-white/30 uppercase tracking-wider">+/-</span>
                </div>
              </>
            )}
          </div>
        )}

        {/* Career totals */}
        {(data.nhl_games_played || data.nhl_career_points) && (
          <div className="border-t px-5 py-2.5 flex items-center justify-center gap-4 flex-wrap" style={{ borderColor: `${teamColor}12` }}>
            <span className="text-[9px] text-white/20 uppercase tracking-widest font-semibold">Career</span>
            {data.nhl_games_played && (
              <div className="flex flex-col items-center min-w-[36px]">
                <span className="text-sm font-bold text-white/70 tabular-nums">{data.nhl_games_played}</span>
                <span className="text-[8px] text-white/25 uppercase tracking-wider">GP</span>
              </div>
            )}
            {data.nhl_career_goals && (
              <div className="flex flex-col items-center min-w-[36px]">
                <span className="text-sm font-bold text-white/70 tabular-nums">{data.nhl_career_goals}</span>
                <span className="text-[8px] text-white/25 uppercase tracking-wider">G</span>
              </div>
            )}
            {data.nhl_career_points && (
              <div className="flex flex-col items-center min-w-[36px]">
                <span className="text-sm font-bold tabular-nums" style={{ color: `${teamColor}CC` }}>{data.nhl_career_points}</span>
                <span className="text-[8px] text-white/25 uppercase tracking-wider">PTS</span>
              </div>
            )}
          </div>
        )}

        {/* ── Bio section — uses NHL landing API data (bio state) ── */}
        {bio && (() => {
          const hCm  = bio.height_cm ?? data.height_cm;
          const wKg  = bio.weight_kg ?? data.weight_kg;
          const bd   = bio.birth_date ?? data.birth_date;
          const bp   = bio.birthplace ?? (data.birth_city ? `${data.birth_city}${data.birth_country ? `, ${data.birth_country}` : ""}` : null);
          const sc   = bio.shoots_catches ?? data.shoots_catches;
          const dy   = bio.draft_year  ?? data.draft_year;
          const dr   = bio.draft_round ?? data.draft_round;
          const dp   = bio.draft_pick  ?? data.draft_pick;
          const dov  = bio.draft_overall;
          const dt   = bio.draft_team  ?? data.draft_team;

          const age = calcAge(bd);
          const heightIn = hCm ? Math.round(hCm / 2.54) : null;
          const heightFmt = heightIn ? `${Math.floor(heightIn / 12)}′${heightIn % 12}″` : null;
          const weightLbs = wKg ? Math.round(wKg * 2.205) : null;
          const birthFmt = bd
            ? (() => { const dd = new Date(bd + "T12:00:00"); return `${dd.getMonth()+1}/${dd.getDate()}/${dd.getFullYear()}`; })()
            : null;
          const draftFmt = dy
            ? `${dy}${dt ? `, ${dt}` : ""}${dov != null ? ` (${ordinal(dov)} overall)` : ""}${dr ? `, ${ordinal(dr)} round${dp ? `, ${ordinal(dp)} pick` : ""}` : ""}`
            : null;

          const fmtCapHit = (n: number) => {
            const abs = Math.abs(n);
            if (abs >= 1_000_000) return `$${(abs / 1_000_000).toFixed(2)}M`;
            if (abs >= 1_000) return `$${(abs / 1_000).toFixed(0)}K`;
            return `$${abs}`;
          };
          const contractStr = contract?.cap_hit
            ? [
                fmtCapHit(contract.cap_hit),
                contract.contract_type || null,
                contract.expiry_status || null,
                contract.expiry_year   ? `exp. ${contract.expiry_year}` : null,
              ].filter(Boolean).join(" · ")
            : null;

          const rows: [string, string][] = [
            ...(heightFmt   ? [["Height", heightFmt] as [string,string]] : []),
            ...(weightLbs   ? [["Weight", `${weightLbs} lb`] as [string,string]] : []),
            ...(birthFmt    ? [["Born",   `${birthFmt}${age != null ? ` (Age: ${age})` : ""}`] as [string,string]] : []),
            ...(bp          ? [["Birthplace", bp] as [string,string]] : []),
            ...(sc          ? [[isGoalie ? "Catches" : "Shoots", sc] as [string,string]] : []),
            ...(draftFmt    ? [["Draft", draftFmt] as [string,string]] : []),
            ...(contractStr ? [["Cap Hit", contractStr] as [string,string]] : []),
          ];

          if (rows.length === 0) return null;
          return (
            <div className="border-t px-5 py-4 space-y-1.5" style={{ borderColor: `${teamColor}12` }}>
              {rows.map(([label, value]) => (
                <div key={label} className="flex items-baseline gap-2">
                  <span className="text-[11px] text-white/30 w-24 shrink-0">{label}:</span>
                  <span className="text-[13px] font-medium text-white/75">{value}</span>
                </div>
              ))}
            </div>
          );
        })()}

        {/* Form blurb */}
        {formBlurb && (
          <div className={`border-t border-white/[0.05] px-5 py-2.5 text-[11px] font-medium
            ${data.ewma_form_flag === "rising" || (data.hot_hand_score ?? 0) > 0.5
              ? "text-[#4ade80]/80 bg-[#4ade80]/[0.04]"
              : "text-[#f87171]/80 bg-[#f87171]/[0.04]"}`}
          >
            {data.ewma_form_flag === "rising" || (data.hot_hand_score ?? 0) > 0.5 ? "🔥 " : "🧊 "}
            {formBlurb}
          </div>
        )}
      </div>

      {/* ── Cards grid ── */}
      <div className="mt-5 grid gap-4 sm:grid-cols-2 min-w-0 overflow-x-hidden">

        {/* Tier legend — worst → best */}
        <div className="sm:col-span-2 flex items-center justify-center gap-1 flex-wrap px-3 py-2 rounded-xl" style={{ border: `1px solid ${teamColor}15`, background: `${teamDarkBg}80` }}>
          <span className="text-[8px] text-white/20 uppercase tracking-wider font-semibold shrink-0 mr-0.5">Scale</span>
          {([["Low","Low"],["Below Average","Below Avg"],["Average","Avg"],["Above Average","Above Avg"],["Elite","Elite"]] as [Tier,string][]).map(([t, label], i, arr) => (
            <div key={t} className="flex items-center gap-1.5">
              <span className="text-[8px] font-semibold uppercase tracking-wider rounded border px-1.5 py-0.5 shrink-0 whitespace-nowrap"
                style={{ color: TIER_COLOR[t], borderColor: `${TIER_COLOR[t]}40`, backgroundColor: `${TIER_COLOR[t]}12` }}>
                {label}
              </span>
              {i < arr.length - 1 && <span className="text-[8px] text-white/10 shrink-0">›</span>}
            </div>
          ))}
        </div>

        {/* ── Performance Snapshot charts ── */}
        {!isGoalie && (
          <div className="sm:col-span-2">
            <Card title="Performance Snapshot" icon="📊" style={cardStyle}>
              <div className="flex flex-wrap justify-center gap-6">

                {/* Radar */}
                {(data.xgf_per60 != null || data.cdr != null || data.battle_percentile != null) && (
                  <div className="flex flex-col items-center w-full overflow-x-auto" style={{ scrollbarWidth: "none" }}>
                    <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-white/30 mb-2 text-center">Attribute Radar</p>
                    <PlayerRadarChart data={data} teamColor={teamColor} />
                  </div>
                )}

                {/* Game log bar chart */}
                {gl && gl.games.length >= 3 && (
                  <div className="flex-1 min-w-0">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-white/30 mb-2 text-center">
                      Last {Math.min(gl.games.length, 10)} Games — G+A
                    </p>
                    <GameLogChart games={gl.games} teamColor={teamColor} />
                    <div className="mt-1 flex items-center justify-center gap-3">
                      <span className="flex items-center gap-1 text-[8px] text-white/30">
                        <span className="inline-block w-2.5 h-2 rounded-sm" style={{ background: teamColor, opacity: 0.75 }} /> Goals
                      </span>
                      <span className="flex items-center gap-1 text-[8px] text-white/30">
                        <span className="inline-block w-2.5 h-2 rounded-sm bg-white/20" /> Assists
                      </span>
                    </div>
                  </div>
                )}

              </div>

              {/* EWMA trend */}
              {data.ewma_xgf60 != null && (
                <div className="mt-5 border-t pt-4" style={{ borderColor: `${teamColor}12` }}>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-white/30 mb-1">
                    xGF/60 Momentum Trend
                    <span className="ml-2 font-normal text-white/25 normal-case tracking-normal">
                      (current: {data.ewma_xgf60.toFixed(2)} · avg: 4.09)
                    </span>
                  </p>
                  <EwmaTrendChart xgf60={data.ewma_xgf60} teamColor={teamColor} />
                </div>
              )}
            </Card>
          </div>
        )}

        {/* ── Shot Map card ── */}
        {!isGoalie && (
          <div className="sm:col-span-2">
            <Card title="Shot Map" icon="🎯" style={cardStyle}>
              {shots.length > 0 ? (
                <>
                  <p className="text-[9px] text-white/25 uppercase tracking-wider text-center mb-3">
                    Arena-adjusted shot locations · last 2 seasons · {shots.length} shots · {shots.filter(s => s.goal).length} goals
                  </p>
                  <ShotMapViz shots={shots} teamColor={teamColor} />
                </>
              ) : (
                <div className="flex flex-col items-center justify-center py-10 gap-2">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-white/20">Shot data syncing</p>
                  <p className="text-[9px] text-white/15 text-center max-w-[220px]">
                    MoneyPuck shot locations load after the data sync runs. Check back after the next sync.
                  </p>
                </div>
              )}
            </Card>
          </div>
        )}

        {/* ── Zone tendency + skating zone side by side ── */}
        {!isGoalie && (
          <div className="sm:col-span-2">
            <Card title="Zone Tendencies" icon="🗺️" style={cardStyle}>
              <div className="flex flex-col sm:flex-row gap-6 items-start justify-center">
                <div className="flex-1 min-w-0">
                  <p className="text-[9px] font-semibold uppercase tracking-wider text-white/30 mb-2 text-center">
                    Offensive Zone Tendency
                  </p>
                  {data.nn_shoot_slot_pct != null ? (
                    <>
                      <ZoneTendencyMap data={data} teamColor={teamColor} />
                      <p className="text-[8px] text-white/20 text-center mt-2">% of offensive actions by zone</p>
                    </>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-8 gap-1.5">
                      <p className="text-[9px] text-white/20 uppercase tracking-wider">Model not yet trained</p>
                      <p className="text-[8px] text-white/12 text-center">Available after Phase 2 model run</p>
                    </div>
                  )}
                </div>
                <div className="flex-1 min-w-0 flex flex-col items-center">
                  <p className="text-[9px] font-semibold uppercase tracking-wider text-white/30 mb-2 text-center">
                    Ice Time By Zone
                  </p>
                  {data.skating_zone_time_oz_pct != null ? (
                    <>
                      <SkatingZoneMap data={data} />
                      <p className="text-[8px] text-white/20 text-center mt-2">% of skating time per zone</p>
                    </>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-8 gap-1.5">
                      <p className="text-[9px] text-white/20 uppercase tracking-wider">Model not yet trained</p>
                      <p className="text-[8px] text-white/12 text-center">Available after Phase 2 model run</p>
                    </div>
                  )}
                </div>
              </div>
            </Card>
          </div>
        )}

        {/* Recent Games */}
        {gl && gl.games.length > 0 && (
          <div className="sm:col-span-2">
            <Card title="Recent Games" icon="📅" style={cardStyle}>
              <GameLogTable allGames={gl.games} />
            </Card>
          </div>
        )}

        {/* Offensive Profile */}
        {!isGoalie && (data.finishing != null || data.war != null || data.rapm_ev_off != null) && (
          <Card title="Offensive Profile" icon="⚡" style={cardStyle}>
            <div className="space-y-0">
              {data.finishing != null && (
                <StatRow
                  label="Finishing Ability"
                  value={`${data.finishing > 0 ? "+" : ""}${data.finishing.toFixed(1)} vs xG`}
                  tier={finishingTier(data.finishing)}
                  sub="Goals above what shot quality predicts"
                  tip="Goals minus expected goals based on shot location and type. Positive = scores more than their chances predict. Elite finishers beat the model consistently — it's a real skill."
                />
              )}
              {data.war != null && (
                <StatRow
                  label="Overall Value (WAR)"
                  value={`${data.war > 0 ? "+" : ""}${data.war.toFixed(2)}`}
                  tier={warTier(data.war)}
                  sub={data.war_rank ? `Ranked #${data.war_rank} of ${data.war_total_qualified ?? "?"} qualified skaters` : "Wins above replacement"}
                  tip="Wins Above Replacement — total value a player adds vs a freely available AHL callup. Combines offense, defense, and special teams into one number. The most complete single-number player value."
                />
              )}
              {data.xgf_per60 != null && (
                <StatRow
                  label="xGF / 60 min"
                  value={data.xgf_per60.toFixed(2)}
                  tier={xgf60Tier(data.xgf_per60)}
                  sub="Expected goals for per 60 minutes at 5v5"
                  tip="Expected goals generated per 60 minutes at 5-on-5. The gold standard possession metric. League average is ~4.1 — players above 5.0 are generating elite scoring chances."
                />
              )}
              {data.rapm_ev_off != null && (
                <StatRow
                  label="Offensive Impact (RAPM)"
                  value={`${data.rapm_ev_off > 0 ? "+" : ""}${data.rapm_ev_off.toFixed(2)}`}
                  tier={rapmOffTier(data.rapm_ev_off)}
                  sub="Isolated offensive impact — controls for teammates"
                  tip="Regularized Adjusted Plus-Minus — offensive side. Isolates a player's true offensive impact by controlling for the quality of linemates and opponents faced. Harder to fake than raw point totals."
                />
              )}
              {data.shots_per60 != null && (
                <StatRow label="Shots / 60" value={data.shots_per60.toFixed(2)} tier={shots60Tier(data.shots_per60)}
                  tip="Shot attempts per 60 minutes at even strength. High volume shooters create sustained pressure. League average is around 10–11." />
              )}
              {data.goals_per60 != null && (
                <StatRow label="Goals / 60" value={data.goals_per60.toFixed(2)} tier={goals60Tier(data.goals_per60)}
                  tip="Even-strength goals scored per 60 minutes. League average is roughly 0.5–0.6. Above 1.0 is elite production." />
              )}
              {data.toi_ev != null && (
                <StatRow label="EV Ice Time" value={`${data.toi_ev.toFixed(0)} min`}
                  sub="5v5 even-strength minutes this season"
                  tip="Total even-strength (5v5) minutes played this season. More TOI = more trust from the coaching staff. All other metrics are measured per 60 to account for ice time differences." />
              )}
            </div>
          </Card>
        )}

        {/* Defensive Profile */}
        {!isGoalie && (data.cdr != null || data.rapm_ev_def != null || data.battle_score != null) && (
          <Card title="Defensive Profile" icon="🛡️" style={cardStyle}>
            <div className="space-y-0">
              {(data.cdr != null || data.rapm_ev_def != null) && (
                <StatRow
                  label="Defensive Value"
                  value={`${(data.cdr ?? data.rapm_ev_def ?? 0) > 0 ? "+" : ""}${(data.cdr ?? data.rapm_ev_def ?? 0).toFixed(2)}`}
                  tier={defTier(data.cdr ?? data.rapm_ev_def ?? 0)}
                  sub="Composite defensive rating (CDR + RAPM)"
                  tip="Composite Defensive Rating — combines RAPM defensive, shot suppression, and zone exit metrics. A single number capturing how well a player prevents scoring chances when on ice."
                />
              )}
              {data.rapm_ev_def != null && data.cdr != null && (
                <StatRow label="RAPM Defensive" value={`${data.rapm_ev_def > 0 ? "+" : ""}${data.rapm_ev_def.toFixed(2)}`}
                  tier={rapmDefTier(data.rapm_ev_def)}
                  sub="Regularized adjusted defensive plus-minus"
                  tip="The defensive component of RAPM. Measures how much a player suppresses opponent scoring chances when controlling for their linemates and opponents. Positive = above average defender." />
              )}
              {data.rapm_xga_60 != null && (
                <StatRow label="xGA / 60 (allowed)" value={data.rapm_xga_60.toFixed(2)}
                  tier={xgaAllowedTier(data.rapm_xga_60)}
                  sub="Expected goals against per 60 when on ice — lower is better"
                  tip="Expected goals against per 60 minutes when this player is on the ice. Lower is better — elite defenders keep this number below 2.5. League average is around 3.2–3.5." />
              )}
              {data.battle_score != null && (
                <StatRow
                  label="Puck Battle Rating"
                  value={`${data.battle_percentile?.toFixed(0) ?? "?"}th pct`}
                  tier={battlePctTier(data.battle_percentile ?? 0)}
                  sub="Physical compete score — hits, blocks, zone battles"
                  tip="Percentile rank in physical compete: hits, blocked shots, and contested zone battles combined. 85th percentile = wins more pucks than 85% of skaters."
                />
              )}
              {data.hits_per60 != null && (
                <StatRow label="Hits / 60" value={data.hits_per60.toFixed(2)} sub="Physical hits per 60 minutes"
                  tip="Physical hits delivered per 60 minutes. Context stat — high hitters can disrupt play but excessive hits can also mean getting caught out of position." />
              )}
              {data.blocks_per60 != null && (
                <StatRow label="Blocked Shots / 60" value={data.blocks_per60.toFixed(2)} sub="Blocked shots per 60 minutes"
                  tip="Shots blocked per 60 minutes. Blocking shots is a commitment to defense — but blocking too many can also indicate spending time in defensive situations." />
              )}
            </div>
          </Card>
        )}

        {/* Special Teams */}
        {!isGoalie && (data.special_teams_pp != null || data.special_teams_pk != null) && (
          <Card title="Special Teams" icon="⭐" style={cardStyle}>
            <div className="space-y-0">
              {data.special_teams_pp != null && (
                <StatRow
                  label="PP xGF/60"
                  value={data.special_teams_pp.toFixed(2)}
                  tier={stTier(data.special_teams_pp)}
                  sub="Expected goals generated per 60 min on the power play"
                  tip="Power play expected goals generated per 60 minutes, isolated from team context. League average PP xGF/60 is ~6–8. Elite PP players exceed 12."
                />
              )}
              {data.special_teams_pk != null && (
                <StatRow
                  label="PK xGF/60"
                  value={data.special_teams_pk.toFixed(2)}
                  tier={stTier(data.special_teams_pk)}
                  sub="Expected goals generated per 60 min short-handed"
                  tip="Penalty kill expected goals generated per 60 minutes, isolated from team context. Higher = more dangerous shorthanded player."
                />
              )}
            </div>
          </Card>
        )}

        {/* Current Form */}
        {!isGoalie && (data.ewma_xgf60 != null || data.hot_hand_score != null || data.clutch_index != null) && (
          <Card title="Current Form" icon="📈" style={cardStyle}>
            <div className="space-y-0">
              {data.ewma_xgf60 != null && (
                <StatRow
                  label="Momentum (EWMA xGF/60)"
                  value={`${data.ewma_xgf60.toFixed(2)}`}
                  tier={
                    ewmaAbove != null
                      ? (ewmaAbove > 1.5 ? "Above Average" : ewmaAbove > -1 ? "Average" : "Below Average")
                      : "Average"
                  }
                  sub={ewmaAbove != null
                    ? `${ewmaAbove > 0 ? "+" : ""}${ewmaAbove.toFixed(1)} vs 4.09 league average`
                    : "Exponentially weighted recent scoring chance rate"}
                  tip="Exponentially Weighted Moving Average of recent xGF/60 — recent games carry more weight than older ones. Tracks hot and cold streaks in real time. League average is 4.09."
                />
              )}
              {data.hot_hand_score != null && (
                <StatRow
                  label="Hot Hand Score"
                  value={data.hot_hand_score.toFixed(3)}
                  tier={hotHandTier(data.hot_hand_score)}
                  sub={`${data.hot_hand_goals5 ?? 0} goals, ${((data.hot_hand_xg5 ?? 0) as number).toFixed(1)} xG in last 5 games`}
                  tip="Statistical test for streaks — measured in standard deviations above expected output over the last 5 games. Above +0.7 = meaningfully running hot. Above +1.5 = serious heater."
                />
              )}
              {data.clutch_index != null && (
                <StatRow
                  label="Clutch Index"
                  value={`${data.clutch_index > 0 ? "+" : ""}${data.clutch_index.toFixed(4)}`}
                  tier={clutchTier(data.clutch_index)}
                  sub="Win probability added per 60 in high-leverage situations"
                  tip="Win Probability Added per 60 minutes in high-leverage situations — close games, third period, tight score. Positive = raises their game when it matters."
                />
              )}
              {data.inseason_mu_blend != null && (
                <StatRow
                  label="In-Season Rating Blend"
                  value={`${data.inseason_mu_blend.toFixed(3)}`}
                  tier={warTier(data.inseason_mu_blend)}
                  sub={data.inseason_ci_lower != null
                    ? `95% CI: [${data.inseason_ci_lower.toFixed(2)}, ${data.inseason_ci_upper?.toFixed(2)}] · ${data.inseason_games ?? "?"} games`
                    : "Bayesian blend of current season + career priors"}
                  tip="Current season stats blended with career priors using Bayesian inference. Early in the season, career history pulls the estimate toward the player's true level. Stabilizes after ~30 games."
                />
              )}
            </div>
          </Card>
        )}

        {/* Playoff & Context */}
        {!isGoalie && (data.playoff_delta != null || data.former_team_boost != null || data.bayesian_rating != null) && (
          <Card title="Advanced Context" icon="🔬" style={cardStyle}>
            <div className="space-y-0">
              {data.bayesian_rating != null && (
                <StatRow
                  label="Bayesian Rating"
                  value={`${data.bayesian_rating.toFixed(3)}`}
                  tier={bayesianTier(data.bayesian_rating)}
                  sub={data.bayesian_uncertainty != null
                    ? `± ${data.bayesian_uncertainty.toFixed(3)} uncertainty`
                    : "Posterior mean — shrunk toward position average"}
                  tip="Posterior mean skill estimate — career stats shrunk toward position average to account for sample size. The smaller the sample, the more it pulls toward average. More stable than single-season raw stats."
                />
              )}
              {data.playoff_delta != null && (
                <StatRow
                  label="Playoff Performer"
                  value={`${data.playoff_delta > 0 ? "+" : ""}${data.playoff_delta.toFixed(3)} xGF/60`}
                  tier={Math.abs(data.playoff_delta) < 0.15 ? "Average" : data.playoff_delta > 0.3 ? "Above Average" : data.playoff_delta < -0.3 ? "Below Average" : "Average"}
                  sub="Production shift in playoffs vs regular season (Bayesian shrinkage)"
                  tip="Difference in xGF/60 between playoff and regular season games, shrunk with Bayesian priors to account for small playoff samples. Positive = elevates their game in the playoffs."
                />
              )}
              {data.former_team_boost != null && data.former_team && (
                <StatRow
                  label={`vs. Former Team (${data.former_team})`}
                  value={`+${(data.former_team_boost * 100).toFixed(0)}%`}
                  sub="Historical xGF bump when facing former employer"
                  tip="Historical xGF improvement when this player faces their former team. Some players have a documented tendency to elevate against teams they used to play for."
                />
              )}
              {data.gar != null && (
                <StatRow
                  label="Goals Above Replacement (GAR)"
                  value={`${data.gar > 0 ? "+" : ""}${data.gar.toFixed(2)}`}
                  tier={garTier(data.gar)}
                  sub="Goals added above a replacement-level player"
                  tip="Goals added above a replacement-level player (an AHL callup). Includes offense, defense, and special teams contributions expressed in goal units. +10 GAR is an elite season."
                />
              )}
              {data.contract_efficiency != null && (
                <StatRow
                  label="Contract Efficiency"
                  value={`${data.contract_efficiency.toFixed(2)}x`}
                  tier={contractEffTier(data.contract_efficiency)}
                  sub="WAR per $M AAV relative to league average"
                  tip="WAR generated per million dollars of cap hit, relative to league average. 1.0x = performing to contract. 2.0x+ = team-friendly deal. Below 0.5x = likely overpaid."
                />
              )}
            </div>
          </Card>
        )}

        {/* Behavioral NN + Skating */}
        {!isGoalie && (
          (data.nn_carry_in_pct != null || data.skating_avg_speed_kmh != null) && (
          <Card title="Play Style (Neural Network)" icon="🧠" style={cardStyle}>
            {playStyle && (
              <p className="text-[11px] text-white/50 mb-4 italic">{playStyle}</p>
            )}
            {data.nn_carry_in_pct != null && (
              <div className="space-y-3 mb-4">
                <p className="text-[10px] font-semibold text-white/30 uppercase tracking-wider">Zone Entry</p>
                <BarStat label="Carry-in (controlled)" value={data.nn_carry_in_pct} max={60} color="#4ade80" />
                <BarStat label="Dump and chase" value={data.nn_dump_pct ?? 0} max={60} color="#f87171" />
              </div>
            )}
            {data.nn_shoot_slot_pct != null && (
              <div className="space-y-3 mb-4">
                <p className="text-[10px] font-semibold text-white/30 uppercase tracking-wider">Shot Selection</p>
                <BarStat label="High-danger slot shots" value={data.nn_shoot_slot_pct} max={50} color="#d946ef" />
                <BarStat label="Perimeter shots" value={data.nn_shoot_perimeter_pct ?? 0} max={50} color="#94a3b8" />
                <BarStat label="Net drives" value={data.nn_drive_net_pct ?? 0} max={40} color="#fbbf24" />
              </div>
            )}
            {data.nn_battle_corner_pct != null && (
              <div className="space-y-3">
                <p className="text-[10px] font-semibold text-white/30 uppercase tracking-wider">Puck Battles</p>
                <BarStat label="Corner battles" value={data.nn_battle_corner_pct} max={40} color="#38bdf8" />
                <BarStat label="Corner holds" value={data.nn_hold_corner_pct ?? 0} max={40} color="#38bdf8" />
              </div>
            )}
            {data.skating_avg_speed_kmh != null && (
              <div className="space-y-3 mt-4">
                <p className="text-[10px] font-semibold text-white/30 uppercase tracking-wider">Skating Baseline</p>
                <BarStat label={`Avg speed — ${data.skating_avg_speed_kmh.toFixed(1)} km/h`} value={data.skating_avg_speed_kmh} max={24} color="#fb923c" suffix=" km/h" />
                <BarStat label={`Max speed — ${data.skating_max_speed_kmh?.toFixed(1) ?? "?"} km/h`} value={data.skating_max_speed_kmh ?? 0} max={35} color="#fb923c" suffix=" km/h" />
                {data.skating_distance_per_game_km != null && (
                  <BarStat label={`Distance / game — ${data.skating_distance_per_game_km.toFixed(2)} km`} value={data.skating_distance_per_game_km} max={8} color="#fb923c" suffix=" km" />
                )}
                {data.skating_zone_time_oz_pct != null && (
                  <div className="space-y-1 mt-3">
                    <p className="text-[10px] font-semibold text-white/30 uppercase tracking-wider">Zone Time</p>
                    <BarStat label="Offensive zone" value={data.skating_zone_time_oz_pct} max={60} color="#4ade80" />
                    <BarStat label="Defensive zone" value={data.skating_zone_time_dz_pct ?? 0} max={60} color="#f87171" />
                  </div>
                )}
                {data.skating_games_sample != null && (
                  <p className="text-[9px] text-white/20 mt-1">{data.skating_games_sample} games sampled for baseline</p>
                )}
              </div>
            )}
            {data.carry_entry_pct != null && (
              <div className="space-y-3 mt-4">
                <p className="text-[10px] font-semibold text-white/30 uppercase tracking-wider">Physical</p>
                <BarStat label="Controlled zone entries %" value={data.carry_entry_pct} max={80} color="#38bdf8" />
                {data.net_front_pct != null && <BarStat label="Net front presence %" value={data.net_front_pct} max={40} color="#38bdf8" />}
              </div>
            )}
          </Card>
          )
        )}

        {/* Goalie stats */}
        {/* Goalie Performance Snapshot — above Goalie Profile */}
        {isGoalie && (
          <div className="sm:col-span-2">
            <Card title="Performance Snapshot" icon="📊" style={cardStyle}>
              <div className="flex flex-wrap justify-center gap-6">
                <div className="flex flex-col items-center w-full overflow-x-auto" style={{ scrollbarWidth: "none" }}>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-white/30 mb-2 text-center">Goalie Radar</p>
                  <GoalieRadarChart data={data} teamColor={teamColor} nhlSvPct={nhlStats?.sv_pct} nhlGaa={nhlStats?.gaa} />
                </div>
              </div>
            </Card>
          </div>
        )}

        {isGoalie && (
          <div className="sm:col-span-2">
            <Card title="Goalie Profile" icon="🥅" style={cardStyle}>
              <div className="space-y-0">
                {data.gsax != null && (
                  <StatRow
                    label="Goals Saved Above Expected"
                    value={`${data.gsax > 0 ? "+" : ""}${data.gsax.toFixed(1)}`}
                    tier={gsaxTier(data.gsax)}
                    sub="The gold standard goalie metric — extra saves vs average goalie"
                    tip="Extra saves made compared to an average goalie facing the same shots. The best single goalie metric because it accounts for shot quality. +15 GSAx is an elite season."
                  />
                )}
                {data.hdsv_pct != null && (
                  <StatRow
                    label="High-Danger Save Rate"
                    value={`${((data.hdsv_pct) * 100).toFixed(1)}%`}
                    tier={hdsvTier((data.hdsv_pct) * 100)}
                    sub="Saves on slot / close-range shots — hardest shots to stop"
                    tip="Save percentage on close-range, slot shots — the hardest shots to stop. The most predictive goalie metric for future performance. League average is around 79–81%."
                  />
                )}
                {data.sv_pct != null && (
                  <StatRow
                    label="Overall Save %"
                    value={`.${Math.round(data.sv_pct * 1000)}`}
                    tier={svpctTier(data.sv_pct)}
                    sub="Save percentage across all shot danger levels"
                    tip="Traditional save percentage across all shots. Less predictive than GSAx or HDSV% because it doesn't account for shot quality — a goalie behind a weak team faces harder shots."
                  />
                )}
                {data.xga != null && (
                  <StatRow
                    label="Expected Goals Against"
                    value={data.xga.toFixed(2)}
                    sub="How many goals an average goalie would allow given the same shots"
                    tip="The number of goals an average NHL goalie would be expected to allow given the exact same shots this goalie faced. The difference between xGA and actual goals against = GSAx."
                  />
                )}
                {data.mdsv_pct != null && (
                  <StatRow
                    label="Mid-Danger Save %"
                    value={`${(data.mdsv_pct * 100).toFixed(1)}%`}
                    tier={mdsvTier(data.mdsv_pct)}
                    sub="Saves on medium-danger shots — outside the slot"
                    tip="Save percentage on medium-danger shots — outside the slot but inside the circles. Should be high for all NHL goalies. Below 90% indicates an issue with positioning or reads."
                  />
                )}
                {data.ldsv_pct != null && (
                  <StatRow
                    label="Low-Danger Save %"
                    value={`${(data.ldsv_pct * 100).toFixed(1)}%`}
                    tier={ldsvTier(data.ldsv_pct)}
                    sub="Saves on long-range, low-threat shots"
                    tip="Save percentage on long-range perimeter shots. Should be extremely high for all NHL goalies (97%+). Dips here can indicate concentration lapses or positioning issues."
                  />
                )}
              </div>
            </Card>
          </div>
        )}

        {/* Line Pairs (chemistry) */}
        {!isGoalie && data.line_pairs && data.line_pairs.length > 0 && (
          <div className="sm:col-span-2">
            <Card title="Best Linemates" icon="🤝" style={cardStyle}>
              <p className="text-[10px] text-white/30 mb-3">
                Partners this player generates above-average scoring chances with — based on shared ice time and xGF%.
              </p>
              <div className="space-y-3">
                {data.line_pairs.map((pair, i) => {
                  const sy2 = data.season ?? new Date().getFullYear();
                  const pct = Math.min(100, Math.max(0, (pair.chemistry_delta ?? 0) * 200));
                  return (
                    <div key={pair.partner_id} className="flex items-center gap-3">
                      <span className="text-[11px] font-bold text-white/20 w-4 shrink-0 tabular-nums">{i + 1}</span>
                      {/* Partner headshot (best effort) */}
                      <PartnerHeadshot playerId={pair.partner_id} season={sy2} />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-white/80 truncate">{pair.partner_name}</p>
                        <p className="text-[9px] text-white/35">
                          {pair.games_together != null ? `${pair.games_together} games together` : ""}
                          {pair.model_xgf_pct != null ? ` · ${pair.model_xgf_pct.toFixed(1)}% xGF` : ""}
                          {pair.co_toi_ev != null ? ` · ${pair.co_toi_ev.toFixed(0)} min EV` : ""}
                        </p>
                      </div>
                      <div className="w-20 h-1.5 rounded-full bg-white/[0.07] overflow-hidden shrink-0">
                        <div className="h-full rounded-full bg-[#4ade80]" style={{ width: `${pct}%`, opacity: 0.75 }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </Card>
          </div>
        )}

      </div>

      {/* Footer */}
      <p className="mt-4 text-[9px] text-white/15 font-mono text-center">
        All metrics sourced from NHL API, MoneyPuck, and GRTZKY models. Season {sy}–{sy + 1}. Informational only.
      </p>
    </main>
  );
}
