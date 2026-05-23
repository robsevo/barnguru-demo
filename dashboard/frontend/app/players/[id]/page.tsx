"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { logoUrl, TEAM_COLORS, TEAM_SECONDARY, normalizePlayerName } from "@/utils/nhl";
import { SeasonContextPill, useSeasonContext } from "@/utils/contextToggle";
import TeamLogoLink from "@/components/TeamLogoLink";
import { useTheme } from "@/utils/themeContext";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line, ReferenceLine,
} from "recharts";
import {
  HudPanel, HudTitle, HudBadge, HudTabBar, RingGauge, Waveform,
  OdometerNumber, NeuralGraph, BodySilhouette, HudGrid, Shot3D, Zone3D,
  type Shot as Shot3DPoint,
  type HudTab,
  type NeuralNode,
  type BodyZone,
  type ZoneActivation,
} from "@/components/hud";

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
  context?: string;
  context_applied?: string;
  playoff_gp?: number | null;
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
  // NHL EDGE — top speeds, distance, hard shots + league ranks
  edge_top_shot_speed_mph?: number | null;
  edge_top_shot_speed_rank?: number | null;
  edge_top_shot_speed_pop?: number | null;
  edge_top_shot_speed_pct?: number | null;
  edge_hard_shot_count?: number | null;
  edge_high_danger_shots?: number | null;
  edge_high_danger_shots_rank?: number | null;
  edge_high_danger_shots_pop?: number | null;
  edge_high_danger_shots_pct?: number | null;
  edge_top_skating_speed_kmh?: number | null;
  edge_top_skating_speed_rank?: number | null;
  edge_top_skating_speed_pop?: number | null;
  edge_top_skating_speed_pct?: number | null;
  edge_total_distance_km?: number | null;
  edge_avg_speed_kmh?: number | null;
  edge_games_played?: number | null;
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
  // League-mean action % (one per behavior action) — used to render deltas
  // so the Predicted Play card surfaces what's distinct about *this* player.
  nn_league_avg?: {
    carry_in?: number; dump?: number;
    shoot_slot?: number; shoot_perimeter?: number;
    drive_net?: number; battle_corner?: number; hold_corner?: number;
  } | null;
  // Position-segmented league averages (forwards / defense / all). The UI
  // prefers the matching segment so a D's perimeter shot rate gets compared
  // to other D, not the forward-heavy global mean.
  nn_league_avg_by_pos?: {
    all?: { carry_in?: number; dump?: number; shoot_slot?: number; shoot_perimeter?: number; drive_net?: number; battle_corner?: number; hold_corner?: number };
    forwards?: { carry_in?: number; dump?: number; shoot_slot?: number; shoot_perimeter?: number; drive_net?: number; battle_corner?: number; hold_corner?: number };
    defense?: { carry_in?: number; dump?: number; shoot_slot?: number; shoot_perimeter?: number; drive_net?: number; battle_corner?: number; hold_corner?: number };
  } | null;
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
// EV ice time in minutes — more TOI = higher coaching trust / usage
function toiTier(v: number): Tier {
  if (v >= 800) return "Elite";          // ~18+ min/game over full season
  if (v >= 600) return "Above Average";  // top-6 F / top-4 D
  if (v >= 350) return "Average";        // middle role
  if (v >= 150) return "Below Average";  // bottom-6 / spot duty
  return "Low";
}
// xGA (Expected Goals Against) for goalies — lower = easier workload
// Tier reflects difficulty faced, not quality (use GSAx for quality)
function xgaTier(v: number): Tier {
  if (v >= 70)  return "Elite";        // heavy starter workload
  if (v >= 55)  return "Above Average";
  if (v >= 35)  return "Average";
  if (v >= 20)  return "Below Average";
  return "Low";
}
function hotHandTier(v: number): Tier {
  if (v >= 1.5)  return "Elite";
  if (v >= 0.7)  return "Above Average";
  if (v >= -0.5) return "Average";
  if (v >= -1.0) return "Below Average";
  return "Low";
}
// Multiplier-style confidence biases (1.0 = neutral). Per the Phase 17.25
// formula `shoot_bias = 1 + 0.06 · ci`, a confidence_index = ±1 maps to
// ±0.06 around 1.0, so the visible range is roughly 0.94–1.06.
function biasTier(v: number): Tier {
  if (v >= 1.030) return "Elite";
  if (v >= 1.008) return "Above Average";
  if (v >= 0.992) return "Average";
  if (v >= 0.970) return "Below Average";
  return "Low";
}
// Fatigue Index: 0 = rested, 1 = saturated. Lower is better, so we invert.
function fatigueTier(v: number): Tier {
  if (v >= 0.55) return "Low";
  if (v >= 0.35) return "Below Average";
  if (v >= 0.20) return "Average";
  if (v >= 0.10) return "Above Average";
  return "Elite";
}
// FI rating multiplier (Phase 3.18): 1.0 = no fatigue drag, < 1 = degraded.
function fiMultiplierTier(v: number): Tier {
  if (v >= 1.000) return "Elite";
  if (v >= 0.985) return "Above Average";
  if (v >= 0.965) return "Average";
  if (v >= 0.940) return "Below Average";
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

function TeamLogo({ team, size = 24 }: { team: string; size?: number }) {
  return <TeamLogoLink abbrev={team} size={size} />;
}

function BarStat({ label, value, max, color = "#4ade80", suffix = "%" }: {
  label: string; value: number; max: number; color?: string; suffix?: string;
}) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className="space-y-1 group">
      <div className="flex items-center justify-between">
        <span className="hud-mono text-[10px] uppercase tracking-[0.16em] text-[var(--text-secondary)] truncate">{label}</span>
        <span className="hud-mono text-[11px] tabular-nums font-semibold" style={{ color }}>
          {value.toFixed(1)}{suffix}
        </span>
      </div>
      <div
        className="relative h-2 rounded-sm overflow-hidden"
        style={{
          background: "rgba(255,255,255,0.04)",
          border: `1px solid ${color}22`,
        }}
      >
        {/* Animated bar fill */}
        <div
          className="h-full"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}aa 0%, ${color} 100%)`,
            boxShadow: `0 0 8px ${color}55, inset 0 0 8px ${color}44`,
            transition: "width 900ms cubic-bezier(0.22,1,0.36,1)",
          }}
        />
        {/* Sweeping scan-line over the bar (decorative) */}
        <div
          className="absolute top-0 bottom-0 w-6 pointer-events-none"
          style={{
            background: `linear-gradient(90deg, transparent, ${color}88, transparent)`,
            animation: "barScan 3.4s linear infinite",
            mixBlendMode: "screen",
            opacity: pct > 5 ? 0.6 : 0,
          }}
        />
        {/* Tick marks at 25/50/75 */}
        <div className="absolute inset-0 flex justify-between px-[25%] pointer-events-none">
          <span className="w-px h-full" style={{ background: "rgba(255,255,255,0.06)" }} />
          <span className="w-px h-full" style={{ background: "rgba(255,255,255,0.10)" }} />
          <span className="w-px h-full" style={{ background: "rgba(255,255,255,0.06)" }} />
        </div>
      </div>
      <style jsx>{`
        @keyframes barScan {
          0%   { transform: translateX(-30px); }
          100% { transform: translateX(420px); }
        }
        @media (prefers-reduced-motion: reduce) {
          div { animation: none !important; transition: none !important; }
        }
      `}</style>
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
  const valueColor = tier ? TIER_COLOR[tier] : null;
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
          <span
            className="text-[12px] font-semibold font-mono tabular-nums"
            style={valueColor ? {
              color: valueColor,
              textShadow: `0 0 6px ${valueColor}55`,
            } : { color: "rgba(255,255,255,0.85)" }}
          >
            {value}
          </span>
          {tier && <TierBadge tier={tier} />}
        </div>
      </div>
    </div>
  );
}

function Card({ title, icon, children, className = "", style }: {
  title: string; icon?: string; children: React.ReactNode; className?: string; style?: React.CSSProperties;
}) {
  // HUD-styled card chrome: corner brackets, mono title bar, scan-band on
  // mount, pulse dot in the header. Tighter padding than before so cards
  // pack tighter on dense tabs. Each Card boots in via jarvis-boot +
  // jarvis-shimmer so the page actually feels alive on tab switch.
  return (
    <div className={`hud-panel hud-panel--all-corners jarvis-boot jarvis-shimmer relative overflow-hidden ${className}`} style={style}>
      <span className="hud-panel__corner-tr" />
      <span className="hud-panel__corner-bl" />
      <div className="hud-scan" aria-hidden />
      <div className="px-2.5 py-1.5 border-b border-white/[0.05] flex items-center gap-2">
        <span className="hud-pulse-dot shrink-0" style={{ background: "var(--brand-hex)", boxShadow: "0 0 4px var(--brand-hex)" }} aria-hidden />
        <span className="hud-mono text-[10px] uppercase tracking-[0.18em] text-[var(--brand-hex)] opacity-90 select-none" aria-hidden>
          ◢
        </span>
        {icon ? <span className="text-[11px] opacity-70" aria-hidden>{icon}</span> : null}
        <p className="hud-mono text-[10px] font-semibold uppercase tracking-[0.20em] text-[var(--text-primary)] truncate"
          style={{ textShadow: "0 0 6px var(--brand-hex)33" }}>
          {title}
        </p>
        <span className="ml-auto hud-mono text-[10px] uppercase tracking-[0.18em] text-[var(--brand-hex)] opacity-90 select-none" aria-hidden>
          ◣
        </span>
      </div>
      <div className="p-2.5 sm:p-3">{children}</div>
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


function PlayerRadarChart({ data, teamColor, maxSize = 300 }: { data: ProfileData; teamColor: string; maxSize?: number }) {
  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  const [chartSize, setChartSize] = useState(maxSize);
  useEffect(() => {
    const update = () => setChartSize(Math.min(maxSize, window.innerWidth - 56));
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, [maxSize]);

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
      {/* HUD label band */}
      <div className="flex items-center justify-between w-full max-w-[300px] mx-auto px-1 mb-1">
        <span className="hud-mono text-[9px] uppercase tracking-[0.22em]" style={{ color: teamColor }}>◢ BIOMETRIC SCAN</span>
        <span className="hud-mono text-[8px] uppercase tracking-[0.16em] text-[var(--text-secondary)]">6-AXIS</span>
      </div>
      {/* Radar with HUD overlay (corner ticks + concentric scan rings) */}
      <div className="relative" style={{ lineHeight: 0, width: chartSize, height: chartSize }}>
        {/* Overlay: concentric scan rings + crosshair behind the chart */}
        <svg
          className="absolute inset-0 pointer-events-none"
          width={chartSize}
          height={chartSize}
          viewBox={`0 0 ${chartSize} ${chartSize}`}
          aria-hidden
        >
          {[0.95, 0.78, 0.55, 0.30].map((scale, i) => (
            <circle
              key={i}
              cx={chartSize / 2}
              cy={chartSize / 2}
              r={Math.round(chartSize * 0.328) * scale}
              fill="none"
              stroke={teamColor}
              strokeOpacity={0.08 + i * 0.02}
              strokeDasharray="2 6"
            />
          ))}
          <line x1={chartSize / 2} y1={6} x2={chartSize / 2} y2={chartSize - 6} stroke={teamColor} strokeOpacity={0.08} strokeDasharray="3 8" />
          <line x1={6} y1={chartSize / 2} x2={chartSize - 6} y2={chartSize / 2} stroke={teamColor} strokeOpacity={0.08} strokeDasharray="3 8" />
          {/* corner ticks */}
          {[
            [4, 4, 14, 4], [4, 4, 4, 14],
            [chartSize - 4, 4, chartSize - 14, 4], [chartSize - 4, 4, chartSize - 4, 14],
            [4, chartSize - 4, 14, chartSize - 4], [4, chartSize - 4, 4, chartSize - 14],
            [chartSize - 4, chartSize - 4, chartSize - 14, chartSize - 4], [chartSize - 4, chartSize - 4, chartSize - 4, chartSize - 14],
          ].map(([x1, y1, x2, y2], i) => (
            <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={teamColor} strokeOpacity={0.6} strokeWidth={1.2} strokeLinecap="round" />
          ))}
        </svg>

        <RadarChart
          width={chartSize} height={chartSize}
          cx={chartSize / 2} cy={chartSize / 2}
          outerRadius={Math.round(chartSize * 0.328)}
          data={chartData}
          margin={{ top: Math.round(chartSize * 0.078), right: Math.round(chartSize * 0.1), bottom: Math.round(chartSize * 0.078), left: Math.round(chartSize * 0.1) }}
          onMouseLeave={() => setActiveIdx(null)}
        >
          <PolarGrid stroke={`${teamColor}22`} />
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
                  fill={isActive ? teamColor : "rgba(255,255,255,0.55)"}
                  style={{ cursor: "pointer", fontFamily: "var(--font-mono)", letterSpacing: "0.12em", textTransform: "uppercase" }}
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
            fillOpacity={0.22}
            strokeWidth={1.5}
            dot={(props: { cx: number; cy: number; index: number }) => {
              const isActive = activeIdx === props.index;
              return (
                <circle
                  key={props.index}
                  cx={props.cx} cy={props.cy}
                  r={isActive ? 8 : 5}
                  fill={isActive ? teamColor : `${teamColor}aa`}
                  stroke={isActive ? "rgba(255,255,255,0.85)" : teamColor}
                  strokeOpacity={isActive ? 1 : 0.4}
                  strokeWidth={1.5}
                  style={{ cursor: "pointer", filter: isActive ? `drop-shadow(0 0 6px ${teamColor})` : undefined }}
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
            <span className="hud-mono text-[10px] uppercase tracking-[0.22em]" style={{ color: teamColor }}>▸ {active.subject}</span>
            <span className="hud-mono text-[11px] text-white/75">{active.raw}</span>
          </>
        ) : (
          <span className="hud-mono text-[9px] uppercase tracking-[0.18em] text-[var(--text-muted)]">hover an axis or dot</span>
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
                  r={isActive ? 8 : 5}
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
          <span className="text-[10px]" style={{ color: "rgba(255,255,255,0.4)" }}>hover an axis or dot</span>
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
          tick={{ fill: "rgba(255,255,255,0.40)", fontSize: 9, fontFamily: "var(--font-mono)", letterSpacing: "0.10em" }}
          axisLine={false} tickLine={false}
        />
        <YAxis
          domain={[0, maxPts + 1]}
          tick={{ fill: "rgba(255,255,255,0.30)", fontSize: 9, fontFamily: "var(--font-mono)" }}
          axisLine={false} tickLine={false}
          allowDecimals={false}
        />
        <Tooltip
          cursor={{ fill: `${teamColor}10` }}
          contentStyle={{ background: "rgba(15,17,20,0.95)", border: `1px solid ${teamColor}55`, borderRadius: 4, fontSize: 10, fontFamily: "var(--font-mono)" }}
          labelStyle={{ color: "rgba(255,255,255,0.45)", letterSpacing: "0.10em", textTransform: "uppercase", fontSize: 9 }}
          formatter={(val: number, name: string) => [val, name === "g" ? "G" : name === "a" ? "A" : name]}
        />
        <Bar dataKey="a" stackId="pts" fill="rgba(255,255,255,0.25)" radius={[0,0,0,0]} />
        <Bar dataKey="g" stackId="pts" fill={teamColor} fillOpacity={0.85} radius={[3,3,0,0]}>
          {chartData.map((_, idx) => (
            <Cell key={idx} fill={teamColor} fillOpacity={chartData[idx].pts > 0 ? 0.90 : 0.30} />
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
    <div className="relative">
      <div className="flex items-center justify-between px-1 mb-1">
        <span className="hud-mono text-[9px] uppercase tracking-[0.22em]" style={{ color: lineColor }}>
          ▸ MOMENTUM TRACE
        </span>
        <span className="hud-mono text-[8px] uppercase tracking-[0.16em] text-[var(--text-secondary)]">
          {isHot ? "RISING" : isCold ? "DECLINING" : "STABLE"}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={108}>
        <LineChart data={trend} margin={{ top: 4, right: 8, left: -28, bottom: 0 }}>
          <XAxis
            dataKey="week"
            tick={{ fill: "rgba(255,255,255,0.30)", fontSize: 9, fontFamily: "var(--font-mono)", letterSpacing: "0.10em" }}
            axisLine={false} tickLine={false}
          />
          <YAxis domain={["auto","auto"]} tick={{ fill: "rgba(255,255,255,0.25)", fontSize: 9, fontFamily: "var(--font-mono)" }} axisLine={false} tickLine={false} />
          <ReferenceLine y={leagueAvg} stroke={`${teamColor}55`} strokeDasharray="2 6" label={{ value: `avg ${leagueAvg}`, fill: "rgba(255,255,255,0.30)", fontSize: 8, position: "insideTopRight", style: { fontFamily: "var(--font-mono)", letterSpacing: "0.10em" } }} />
          <Tooltip
            contentStyle={{ background: "rgba(15,17,20,0.92)", border: `1px solid ${lineColor}55`, borderRadius: 6, fontSize: 10, fontFamily: "var(--font-mono)" }}
            labelStyle={{ color: "rgba(255,255,255,0.45)", letterSpacing: "0.10em", textTransform: "uppercase", fontSize: 9 }}
            formatter={(v: number) => [v.toFixed(2), "xGF/60"]}
          />
          <Line type="monotone" dataKey="xgf" stroke={lineColor} strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: lineColor, stroke: "rgba(255,255,255,0.85)", strokeWidth: 1 }}
            style={{ filter: `drop-shadow(0 0 6px ${lineColor}66)` }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shot / zone visualizations
// ---------------------------------------------------------------------------

interface ShotPoint { x: number; y: number; xg: number; goal: boolean; type: string; }

/** Half-rink SVG shot map — proper ice surface, dots coloured by xG, goals as rings */
/** Shared half-rink SVG markings — ice surface, lines, net. Used by both shot map variants. */
function HalfRinkMarkings() {
  return (
    <>
      {/* Ice surface */}
      <path d="M 0,0 L 86,0 Q 100,0 100,14 L 100,71 Q 100,85 86,85 L 0,85 Z"
        fill="#f8fbff" stroke="#94b4cc" strokeWidth="1.8" />
      {/* Blue line */}
      <line x1="25" y1="0.5" x2="25" y2="84.5" stroke="#1155bb" strokeWidth="1.4" opacity="0.7" />
      {/* Goal line — extended to boards */}
      <line x1="89" y1="0.5" x2="89" y2="84.5" stroke="#cc2222" strokeWidth="0.9" opacity="0.8" />
      {/* Trapezoid */}
      <polygon points="100,28.5 89,33.5 89,51.5 100,56.5"
        fill="none" stroke="#cc2222" strokeWidth="0.5" opacity="0.4" />
      {/* Crease — filled D-shape, r=6, sweeps toward center ice (sweep=0 = left/CCW) */}
      <path d="M 89 36.5 A 6 6 0 0 0 89 48.5 Z"
        fill="rgba(30,100,200,0.13)" stroke="#1155bb" strokeWidth="0.8" opacity="0.75" />
      {/* Net — narrow depth, 6ft post-to-post */}
      <rect x="89" y="39.5" width="2.5" height="6" rx="0.5"
        fill="rgba(180,180,180,0.5)" stroke="#777" strokeWidth="0.6" />
      {/* Faceoff circles */}
      <circle cx="69" cy="20.5" r="15" fill="none" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <circle cx="69" cy="64.5" r="15" fill="none" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <circle cx="69" cy="20.5" r="0.85" fill="#cc2222" opacity="0.55" />
      <circle cx="69" cy="64.5" r="0.85" fill="#cc2222" opacity="0.55" />
      {/* Interior hashmarks — top OZ circle (⊞, tight around dot) */}
      <line x1="67.5" y1="16.5" x2="67.5" y2="19.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="70.5" y1="16.5" x2="70.5" y2="19.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="67.5" y1="21.5" x2="67.5" y2="24.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="70.5" y1="21.5" x2="70.5" y2="24.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="65" y1="19" x2="68" y2="19" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="70" y1="19" x2="73" y2="19" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="65" y1="22" x2="68" y2="22" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="70" y1="22" x2="73" y2="22" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      {/* Exterior hashmarks — top OZ circle (2 marks outside ring, top & bottom) */}
      <line x1="67.5" y1="3" x2="67.5" y2="5.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="70.5" y1="3" x2="70.5" y2="5.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="67.5" y1="35.5" x2="67.5" y2="38" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="70.5" y1="35.5" x2="70.5" y2="38" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      {/* Interior hashmarks — bottom OZ circle */}
      <line x1="67.5" y1="60.5" x2="67.5" y2="63.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="70.5" y1="60.5" x2="70.5" y2="63.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="67.5" y1="65.5" x2="67.5" y2="68.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="70.5" y1="65.5" x2="70.5" y2="68.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="65" y1="63" x2="68" y2="63" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="70" y1="63" x2="73" y2="63" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="65" y1="66" x2="68" y2="66" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="70" y1="66" x2="73" y2="66" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      {/* Exterior hashmarks — bottom OZ circle (2 marks outside ring, top & bottom) */}
      <line x1="67.5" y1="47" x2="67.5" y2="49.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="70.5" y1="47" x2="70.5" y2="49.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="67.5" y1="79.5" x2="67.5" y2="82" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <line x1="70.5" y1="79.5" x2="70.5" y2="82" stroke="#cc2222" strokeWidth="0.6" opacity="0.4" />
      <circle cx="20" cy="20.5" r="0.85" fill="#cc2222" opacity="0.35" />
      <circle cx="20" cy="64.5" r="0.85" fill="#cc2222" opacity="0.35" />
      {/* Center ice boundary */}
      <line x1="0" y1="0" x2="0" y2="85" stroke="#cc2222" strokeWidth="1.2" opacity="0.85" />
    </>
  );
}

/** Shared blurred heat map — used for both skater shots and goalie shots-against. */
type ShotLayer = "lo" | "med" | "hi" | "goal";

function ShotHeatMapViz({ shots, maxShots = 900 }: { shots: ShotPoint[]; maxShots?: number }) {
  const [hidden, setHidden] = useState<Set<ShotLayer>>(new Set());
  const toggle = (layer: ShotLayer) =>
    setHidden(prev => { const next = new Set(prev); next.has(layer) ? next.delete(layer) : next.add(layer); return next; });

  const sy = (y: number) => 42.5 - y;
  const pts = shots.slice(-maxShots);

  const lo   = pts.filter(s => !s.goal && s.xg < 0.08);
  const med  = pts.filter(s => !s.goal && s.xg >= 0.08 && s.xg < 0.20);
  const hi   = pts.filter(s => !s.goal && s.xg >= 0.20);
  const goals = pts.filter(s => s.goal);

  const legendItems: { key: ShotLayer; label: string; tw: string }[] = [
    { key: "lo",   label: "Low xG",        tw: "bg-sky-400" },
    { key: "med",  label: "Med xG",        tw: "bg-orange-400" },
    { key: "hi",   label: "High xG",       tw: "bg-red-700" },
    { key: "goal", label: "Goal",          tw: "bg-red-950" },
  ];

  return (
    <>
      <svg viewBox="0 0 100 85" width="100%" className="block mx-auto max-w-[420px]"
        style={{ filter: "drop-shadow(0 3px 14px rgba(0,0,0,0.5))" }}>
        <defs>
          {/* Blur per tier — tighter blur on high-xG so hot spots stay crisp */}
          <filter id="heatLo"   x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="5"/></filter>
          <filter id="heatMed"  x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="4.5"/></filter>
          <filter id="heatHi"   x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="4"/></filter>
          <filter id="heatGoal" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="3.5"/></filter>
        </defs>
        <HalfRinkMarkings />
        {/* Low xG — sky blue, large blur, low opacity per circle */}
        {!hidden.has("lo") && lo.length > 0 && (
          <g filter="url(#heatLo)">
            {lo.map((s, i) => <circle key={i} cx={s.x} cy={sy(s.y)} r="4" fill="#38bdf8" opacity="0.13" />)}
          </g>
        )}
        {/* Med xG — orange mid-layer */}
        {!hidden.has("med") && med.length > 0 && (
          <g filter="url(#heatMed)">
            {med.map((s, i) => <circle key={i} cx={s.x} cy={sy(s.y)} r="5" fill="#f97316" opacity="0.16" />)}
          </g>
        )}
        {/* High xG saves — red hot zone */}
        {!hidden.has("hi") && hi.length > 0 && (
          <g filter="url(#heatHi)">
            {hi.map((s, i) => <circle key={i} cx={s.x} cy={sy(s.y)} r="5.5" fill="#dc2626" opacity="0.20" />)}
          </g>
        )}
        {/* Goals — darkest red, extra radius so goal clusters glow more than saves */}
        {!hidden.has("goal") && goals.length > 0 && (
          <g filter="url(#heatGoal)">
            {goals.map((s, i) => <circle key={i} cx={s.x} cy={sy(s.y)} r="7" fill="#7f1d1d" opacity="0.28" />)}
          </g>
        )}
      </svg>
      <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 mt-2 text-[10px] font-semibold tracking-wide uppercase">
        {legendItems.map(({ key, label, tw }) => {
          const off = hidden.has(key);
          return (
            <span key={key} onClick={() => toggle(key)}
              className={`flex items-center gap-1.5 cursor-pointer select-none transition-opacity ${off ? "opacity-25" : "opacity-100"}`}
              style={{ color: off ? "#52525b" : "#71717a" }}>
              <span className={`inline-block w-2.5 h-2.5 rounded-full ${tw}`} />
              {label}
            </span>
          );
        })}
      </div>
    </>
  );
}

/** Skater shot map — xG-colored dots with toggleable layers */
function ShotMapViz({ shots }: { shots: ShotPoint[] }) {
  type DotLayer = "low" | "med" | "high" | "hot" | "goal";
  const [hidden, setHidden] = useState<Set<DotLayer>>(new Set());
  const toggle = (layer: DotLayer) =>
    setHidden(prev => { const next = new Set(prev); next.has(layer) ? next.delete(layer) : next.add(layer); return next; });

  const sy = (y: number) => 42.5 - y;
  const pts = shots.slice(-800);

  // Seed-stable jitter so goal dots spread out when stacked on the same location.
  // Uses a simple LCG per-index so it's deterministic across renders.
  const jitter = (i: number, scale: number) => {
    const a = Math.sin(i * 127.1 + 311.7) * 43758.5453;
    return (a - Math.floor(a) - 0.5) * scale;
  };

  const nonGoals = pts.filter(s => !s.goal);
  const goals    = pts.filter(s => s.goal);

  const dotLayers: { key: DotLayer; pts: ShotPoint[]; r: number; fill: string; opacity: number }[] = [
    { key: "low",  pts: nonGoals.filter(s => s.xg < 0.05),                  r: 0.85, fill: "#22c55e", opacity: 0.65 },
    { key: "med",  pts: nonGoals.filter(s => s.xg >= 0.05 && s.xg < 0.12), r: 0.85, fill: "#fbbf24", opacity: 0.70 },
    { key: "high", pts: nonGoals.filter(s => s.xg >= 0.12 && s.xg < 0.22), r: 0.85, fill: "#f97316", opacity: 0.75 },
    { key: "hot",  pts: nonGoals.filter(s => s.xg >= 0.22),                  r: 0.85, fill: "#dc2626", opacity: 0.80 },
    { key: "goal", pts: goals,                                                r: 1.5,  fill: "#b91c1c", opacity: 1.00 },
  ];
  const dotLegend: { key: DotLayer; label: string; color?: string; tw?: string }[] = [
    { key: "low",  label: "Low",  tw: "bg-green-500" },
    { key: "med",  label: "Med",  tw: "bg-yellow-400" },
    { key: "high", label: "High", tw: "bg-orange-500" },
    { key: "hot",  label: "Hot",  tw: "bg-red-600" },
    { key: "goal", label: "Goal", color: "#b91c1c" },
  ];
  // CCW-rotated: viewBox 85×100, transform wraps original 100×85 content
  // boards (x=0) → bottom, net (x=89) → top
  return (
    <>
      <svg viewBox="0 0 85 100" width="100%" className="block mx-auto max-w-[420px]"
        style={{ filter: "drop-shadow(0 3px 14px rgba(0,0,0,0.5))" }}>
        <g transform="translate(0,100) rotate(-90)">
          <HalfRinkMarkings />
          {dotLayers.map(({ key, pts: lpts, r, fill, opacity }) =>
            hidden.has(key) ? null : lpts.map((s, i) => {
              const isGoal = key === "goal";
              const cx = s.x + (isGoal ? jitter(i * 2,     1.2) : 0);
              const cy = sy(s.y) + (isGoal ? jitter(i * 2 + 1, 1.2) : 0);
              return isGoal ? (
                // Goals: filled dot + white ring so individual goals show through stacks
                <g key={`${key}${i}`}>
                  <circle cx={cx} cy={cy} r={r + 0.5} fill="white" opacity={0.55} />
                  <circle cx={cx} cy={cy} r={r} fill={fill} opacity={opacity} />
                </g>
              ) : (
                <circle key={`${key}${i}`} cx={cx} cy={cy} r={r} fill={fill} opacity={opacity} />
              );
            })
          )}
        </g>
      </svg>
      <div className="flex flex-wrap justify-center gap-x-3 gap-y-1 mt-2 text-[10px] font-semibold tracking-wide uppercase">
        {dotLegend.map(({ key, label, tw, color }) => {
          const off = hidden.has(key);
          return (
            <span key={key} onClick={() => toggle(key)}
              className={`flex items-center gap-1.5 cursor-pointer select-none transition-opacity ${off ? "opacity-25" : "opacity-100"}`}
              style={{ color: off ? "#52525b" : "#71717a" }}>
              <span className={`inline-block w-2.5 h-2.5 rounded-full ${tw ?? ""}`}
                style={color ? { backgroundColor: color } : {}} />
              {label}
            </span>
          );
        })}
      </div>
      <p className="text-[9px] text-center mt-1.5" style={{ color: "rgba(255,255,255,0.45)" }}>tap a dot to show or hide that layer</p>
    </>
  );
}

/** Goalie shots-against map — same xG-colored dots as skater map */
function GoalieShotMapViz({ shots }: { shots: ShotPoint[] }) {
  type DotLayer = "low" | "med" | "high" | "hot" | "goal";
  const [hidden, setHidden] = useState<Set<DotLayer>>(new Set());
  const toggle = (layer: DotLayer) =>
    setHidden(prev => { const next = new Set(prev); next.has(layer) ? next.delete(layer) : next.add(layer); return next; });

  const sy = (y: number) => 42.5 - y;
  const pts = shots.slice(-900);
  const dotLayers: { key: DotLayer; pts: ShotPoint[]; r: string; fill: string; opacity: string }[] = [
    { key: "low",  pts: pts.filter(s => !s.goal && s.xg < 0.05),                r: "0.8", fill: "#22c55e", opacity: "0.65" },
    { key: "med",  pts: pts.filter(s => !s.goal && s.xg >= 0.05 && s.xg < 0.12), r: "0.8", fill: "#fbbf24", opacity: "0.70" },
    { key: "high", pts: pts.filter(s => !s.goal && s.xg >= 0.12 && s.xg < 0.22), r: "0.8", fill: "#f97316", opacity: "0.75" },
    { key: "hot",  pts: pts.filter(s => !s.goal && s.xg >= 0.22),                r: "0.8", fill: "#dc2626", opacity: "0.80" },
    { key: "goal", pts: pts.filter(s => s.goal),                                  r: "1.2", fill: "#b91c1c", opacity: "0.95" },
  ];
  const dotLegend: { key: DotLayer; label: string; color?: string; tw?: string }[] = [
    { key: "low",  label: "Low",  tw: "bg-green-500" },
    { key: "med",  label: "Med",  tw: "bg-yellow-400" },
    { key: "high", label: "High", tw: "bg-orange-500" },
    { key: "hot",  label: "Hot",  tw: "bg-red-600" },
    { key: "goal", label: "Goal", color: "#b91c1c" },
  ];
  return (
    <>
      <svg viewBox="0 0 85 100" width="100%" className="block mx-auto max-w-[420px]"
        style={{ filter: "drop-shadow(0 3px 14px rgba(0,0,0,0.5))" }}>
        <g transform="translate(0,100) rotate(-90)">
          <HalfRinkMarkings />
          {dotLayers.map(({ key, pts: lpts, r, fill, opacity }) =>
            hidden.has(key) ? null : lpts.map((s, i) =>
              <circle key={`${key}${i}`} cx={s.x} cy={sy(s.y)} r={r} fill={fill} opacity={opacity} />
            )
          )}
        </g>
      </svg>
      <div className="flex flex-wrap justify-center gap-x-3 gap-y-1 mt-2 text-[10px] font-semibold tracking-wide uppercase">
        {dotLegend.map(({ key, label, tw, color }) => {
          const off = hidden.has(key);
          return (
            <span key={key} onClick={() => toggle(key)}
              className={`flex items-center gap-1.5 cursor-pointer select-none transition-opacity ${off ? "opacity-25" : "opacity-100"}`}
              style={{ color: off ? "#52525b" : "#71717a" }}>
              <span className={`inline-block w-2.5 h-2.5 rounded-full ${tw ?? ""}`}
                style={color ? { backgroundColor: color } : {}} />
              {label}
            </span>
          );
        })}
      </div>
      <p className="text-[9px] text-center mt-1.5" style={{ color: "rgba(255,255,255,0.45)" }}>tap a dot to show or hide that layer</p>
    </>
  );
}

/** Goalie shot-type + zone breakdown — "neural net" tendency analysis */
interface GoalieNetData {
  status: string;
  total_shots: number;
  goals_allowed: number;
  overall_sv_pct: number | null;
  shot_types: { type: string; shots: number; goals: number; sv_pct: number }[];
  zones: { side: string; dist: string; shots: number; goals: number; sv_pct: number }[];
}

// League-average save% by shot type (approximate from MoneyPuck population)
const LEAGUE_SV_BY_TYPE: Record<string, number> = {
  "Wrist": 0.919, "Slap": 0.921, "Snap": 0.912,
  "Tip-In": 0.844, "Deflection": 0.832, "Backhand": 0.888, "Wraparound": 0.880,
};

// League-average sv% by (side, dist) — close shots are hardest regardless of side
const LEAGUE_SV_BY_ZONE: Record<string, Record<string, number>> = {
  close: { left: 0.876, center: 0.848, right: 0.876 },
  mid:   { left: 0.925, center: 0.912, right: 0.925 },
  far:   { left: 0.952, center: 0.946, right: 0.952 },
};

function GoalieNeuralNetViz({ netData }: { netData: GoalieNetData }) {
  const svColor = (sv: number, leagueAvg: number) => {
    const delta = sv - leagueAvg;
    if (delta >= 0.025) return "#15803d";   // green — well above average
    if (delta >= 0.005) return "#65a30d";   // light green
    if (delta >= -0.005) return "#a16207";  // amber — near average
    if (delta >= -0.025) return "#c2410c";  // orange — below average
    return "#991b1b";                       // red — well below average
  };

  const fmtSv = (sv: number) => `${(sv * 100).toFixed(1)}%`;

  // Zone grid: 3 dist rows (close/mid/far) × 3 side columns (left/center/right)
  const distOrder = ["close", "mid", "far"] as const;
  const sideOrder = ["left", "center", "right"] as const;
  const distLabel: Record<string, string> = { close: "< 25ft", mid: "25–45ft", far: "> 45ft" };
  const sideLabel: Record<string, string> = { left: "Left", center: "Center", right: "Right" };

  const zoneMap: Record<string, Record<string, { sv_pct: number; shots: number } | null>> = {};
  for (const d of distOrder) { zoneMap[d] = {}; for (const s of sideOrder) zoneMap[d][s] = null; }
  for (const z of netData.zones) zoneMap[z.dist]?.[z.side] !== undefined && (zoneMap[z.dist][z.side] = { sv_pct: z.sv_pct, shots: z.shots });

  return (
    <div className="space-y-5">
      {/* Zone grid — futuristic HUD heatmap */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="hud-mono text-[9px] uppercase tracking-[0.20em]" style={{ color: "var(--brand-hex)" }}>
            ◢ SAVE % · DISTANCE × LATERAL
          </span>
          <span className="hud-mono text-[8px] uppercase tracking-[0.16em] text-[var(--text-muted)]">
            3 × 3 GRID
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-center border-collapse" style={{ minWidth: 220 }}>
            <thead>
              <tr>
                <th className="hud-mono text-[8px] font-semibold text-[var(--text-muted)] uppercase tracking-[0.16em] pb-1.5 pr-1 text-left w-14"></th>
                {sideOrder.map(s => (
                  <th key={s} className="hud-mono text-[8px] uppercase tracking-[0.16em] text-[var(--text-secondary)] pb-1.5 px-1">
                    ▾ {sideLabel[s]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {distOrder.map((dist, di) => (
                <tr key={dist}>
                  <td className="hud-mono text-[8px] uppercase tracking-[0.16em] text-[var(--text-secondary)] pr-2 py-1 text-left whitespace-nowrap">
                    ▸ {distLabel[dist]}
                  </td>
                  {sideOrder.map((side, si) => {
                    const cell = zoneMap[dist][side];
                    const lg = LEAGUE_SV_BY_ZONE[dist]?.[side] ?? 0.91;
                    if (!cell) {
                      return (
                        <td key={side} className="px-1 py-1">
                          <div className="rounded-sm py-2 px-1 border border-white/[0.04]" style={{ background: "rgba(255,255,255,0.015)" }}>
                            <span className="hud-mono text-[10px] text-white/15">— —</span>
                          </div>
                        </td>
                      );
                    }
                    const bg = svColor(cell.sv_pct, lg);
                    const delay = (di * 3 + si) * 60;
                    return (
                      <td key={side} className="px-1 py-1">
                        <div
                          className="relative rounded-sm py-1.5 px-1 overflow-hidden"
                          style={{
                            background: `linear-gradient(180deg, ${bg}33 0%, ${bg}14 100%)`,
                            border: `1px solid ${bg}80`,
                            boxShadow: `inset 0 1px 0 rgba(255,255,255,0.10), inset 0 -1px 0 rgba(0,0,0,0.40), 0 0 8px ${bg}33`,
                            animation: `gnnReveal 650ms cubic-bezier(0.22,1,0.36,1) ${delay}ms backwards`,
                          }}
                        >
                          {/* Sweeping scan-line for "live" feel */}
                          <span
                            aria-hidden
                            className="absolute top-0 left-0 right-0 h-px pointer-events-none"
                            style={{
                              background: `linear-gradient(90deg, transparent, ${bg}cc, transparent)`,
                              animation: `gnnScan 3.4s linear infinite ${delay}ms`,
                            }}
                          />
                          <div className="hud-mono text-[11px] font-semibold tabular-nums" style={{ color: bg, textShadow: `0 0 6px ${bg}55` }}>
                            {fmtSv(cell.sv_pct)}
                          </div>
                          <div className="hud-mono text-[7px] tracking-[0.10em] text-[var(--text-muted)] mt-0.5">
                            {cell.shots} SH
                          </div>
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="hud-mono text-[8px] uppercase tracking-[0.16em] text-[var(--text-muted)] text-center mt-1.5">
          GREEN ABOVE · RED BELOW · vs league baseline
        </p>
        <style jsx>{`
          @keyframes gnnReveal {
            from { transform: scale(0.92); opacity: 0; }
            to   { transform: scale(1);    opacity: 1; }
          }
          @keyframes gnnScan {
            0%   { transform: translateX(-100%); opacity: 0; }
            20%  { opacity: 0.85; }
            80%  { opacity: 0.85; }
            100% { transform: translateX(100%); opacity: 0; }
          }
          @media (prefers-reduced-motion: reduce) {
            div, span { animation: none !important; }
          }
        `}</style>
      </div>

      {/* Shot type breakdown — futuristic bars */}
      {netData.shot_types.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="hud-mono text-[9px] uppercase tracking-[0.20em]" style={{ color: "var(--brand-hex)" }}>
              ◢ SAVE % · BY SHOT TYPE
            </span>
            <span className="hud-mono text-[8px] uppercase tracking-[0.16em] text-[var(--text-muted)]">
              vs LEAGUE
            </span>
          </div>
          <div className="space-y-1.5">
            {netData.shot_types.map((t, i) => {
              const lg = LEAGUE_SV_BY_TYPE[t.type] ?? 0.91;
              const col = svColor(t.sv_pct, lg);
              const barPct = Math.round(t.sv_pct * 100);
              const lgBarPct = Math.round(lg * 100);
              return (
                <div key={t.type} className="flex items-center gap-2">
                  <span className="hud-mono text-[9px] uppercase tracking-[0.14em] text-[var(--text-secondary)] w-20 shrink-0 text-right">
                    {t.type}
                  </span>
                  <div
                    className="flex-1 relative h-3 rounded-sm overflow-hidden"
                    style={{
                      background: "rgba(255,255,255,0.04)",
                      border: `1px solid ${col}33`,
                    }}
                  >
                    <div
                      className="absolute inset-y-0 left-0"
                      style={{
                        width: `${barPct}%`,
                        background: `linear-gradient(90deg, ${col}aa 0%, ${col} 100%)`,
                        boxShadow: `0 0 8px ${col}66, inset 0 0 8px ${col}44`,
                        animation: `gnnBarFill 800ms cubic-bezier(0.22,1,0.36,1) ${i * 70}ms backwards`,
                        transformOrigin: "left center",
                      }}
                    />
                    {/* Sweeping scan-line */}
                    <div
                      className="absolute top-0 bottom-0 w-6 pointer-events-none"
                      style={{
                        background: `linear-gradient(90deg, transparent, ${col}88, transparent)`,
                        animation: `gnnTypeScan 3.2s linear infinite ${i * 0.25}s`,
                        mixBlendMode: "screen",
                        opacity: barPct > 5 ? 0.7 : 0,
                      }}
                    />
                    {/* League avg tick */}
                    <div
                      className="absolute inset-y-0 w-px"
                      style={{
                        left: `${lgBarPct}%`,
                        background: "rgba(255,255,255,0.55)",
                        boxShadow: "0 0 4px rgba(255,255,255,0.5)",
                      }}
                    />
                    <span
                      className="absolute inset-y-0 right-1 flex items-center hud-mono text-[9px] tabular-nums font-semibold"
                      style={{ color: col, textShadow: `0 0 4px ${col}66` }}
                    >
                      {fmtSv(t.sv_pct)}
                    </span>
                  </div>
                  <span className="hud-mono text-[8px] tracking-[0.10em] text-[var(--text-muted)] w-10 shrink-0">
                    {t.shots}sh
                  </span>
                </div>
              );
            })}
          </div>
          <p className="hud-mono text-[8px] uppercase tracking-[0.16em] text-[var(--text-muted)] text-center mt-2">
            ▎ tick = league avg · last 2 seasons
          </p>
          <style jsx>{`
            @keyframes gnnBarFill {
              from { transform: scaleX(0); opacity: 0; }
              to   { transform: scaleX(1); opacity: 1; }
            }
            @keyframes gnnTypeScan {
              0%   { transform: translateX(-30px); }
              100% { transform: translateX(400px); }
            }
            @media (prefers-reduced-motion: reduce) {
              div { animation: none !important; }
            }
          `}</style>
        </div>
      )}
    </div>
  );
}
/** Goalie save% by zone — color-coded half-rink with HD/MD/LD bands. */
/**
 * GoalieZoneViz — 5-zone goalie NET diagram (like a hockey play diagram):
 *   ┌─────────────┐
 *   │  3      1  │   top corners
 *   │             │
 *   │  4   5   2  │   bottom corners + 5-hole
 *   └─────────────┘
 *
 * Color saturation per zone = goalie save % in that location, with team-color
 * heat overlay. Zone hover surfaces the exact percentage in a HUD callout.
 *
 * NOTE: NHL APIs expose HD/MD/LD save% rather than 5-zone shot heatmaps, so we
 * derive the per-zone save% as: HD → zones 1/2/3 (top + bottom corners shot
 * proximity weighting), 5-hole gets a blended weight, LD/MD fill the wider
 * outer zones. This is an approximation but communicates strengths clearly.
 */
function GoalieZoneViz({ data, teamColor = "var(--brand-hex)" }: { data: ProfileData; teamColor?: string }) {
  const hd = data.hdsv_pct ?? null;
  const md = data.mdsv_pct ?? null;
  const ld = data.ldsv_pct ?? null;
  const [hover, setHover] = useState<string | null>(null);

  // Per-zone save% approximation (1=top R, 2=bot R, 3=top L, 4=bot L, 5=5-hole)
  const zoneVals = {
    "1": hd,
    "2": hd,
    "3": hd,
    "4": hd,
    "5": md != null && hd != null ? (md * 0.65 + hd * 0.35) : (md ?? hd),
  };

  // Heat → tone (color saturation reflects performance level)
  const tone = (pct: number | null) => {
    if (pct == null) return { fill: "rgba(120,120,120,0.10)", stroke: "rgba(120,120,120,0.30)" };
    if (pct >= 0.92) return { fill: "rgba(74,222,128,0.32)",  stroke: "rgba(74,222,128,0.85)"  };
    if (pct >= 0.88) return { fill: `${teamColor}38`,         stroke: `${teamColor}` };
    if (pct >= 0.84) return { fill: "rgba(251,191,36,0.30)",  stroke: "rgba(251,191,36,0.85)"  };
    return                  { fill: "rgba(248,113,113,0.32)", stroke: "rgba(248,113,113,0.90)" };
  };

  const fmt = (pct: number | null) => pct == null ? "—" : `${(pct * 100).toFixed(1)}%`;

  // Net dimensions in viewBox 100×80: net frame from (8,8) to (92,68)
  // Zones (top row): 3 left (8-36, 8-38), 1 right (64-92, 8-38)
  // Zones (bot row): 4 left (8-36, 38-68), 2 right (64-92, 38-68)
  // Zone 5 (5-hole): (40-60, 40-68) — between goalie legs
  const zones = [
    { id: "3", x: 8,  y: 8,  w: 28, h: 30, label: "3", val: zoneVals["3"], anchor: "TOP L" },
    { id: "1", x: 64, y: 8,  w: 28, h: 30, label: "1", val: zoneVals["1"], anchor: "TOP R" },
    { id: "4", x: 8,  y: 38, w: 28, h: 30, label: "4", val: zoneVals["4"], anchor: "BOT L" },
    { id: "2", x: 64, y: 38, w: 28, h: 30, label: "2", val: zoneVals["2"], anchor: "BOT R" },
    { id: "5", x: 40, y: 40, w: 20, h: 28, label: "5", val: zoneVals["5"], anchor: "5-HOLE" },
  ];

  return (
    <div className="relative w-full max-w-[520px] mx-auto"
      style={{ ["--gnz-color" as string]: teamColor }}>
      {/* Floor/ice line under the net for depth */}
      <svg viewBox="0 0 120 90" width="100%" className="block"
        style={{ filter: `drop-shadow(0 8px 16px rgba(0,0,0,0.6)) drop-shadow(0 0 16px ${teamColor}33)` }}>
        <defs>
          {/* Dense diagonal cross-hatched netting */}
          <pattern id="netHatch" width="3" height="3" patternUnits="userSpaceOnUse">
            <path d="M 0 0 L 3 3 M 3 0 L 0 3" stroke="rgba(255,255,255,0.18)" strokeWidth="0.22" fill="none" />
          </pattern>
          {/* Tighter ortho mesh for inner shading */}
          <pattern id="netGrid2" width="2" height="2" patternUnits="userSpaceOnUse">
            <path d="M 2 0 L 2 2 M 0 2 L 2 2" stroke="rgba(255,255,255,0.06)" strokeWidth="0.12" fill="none" />
          </pattern>
          {/* Subtle inner-net shadow for depth */}
          <radialGradient id="netDepth" cx="50%" cy="50%" r="65%">
            <stop offset="0%" stopColor="rgba(0,0,0,0.55)" />
            <stop offset="60%" stopColor="rgba(0,0,0,0.30)" />
            <stop offset="100%" stopColor="rgba(0,0,0,0.15)" />
          </radialGradient>
          {/* Red goal frame metal gradient */}
          <linearGradient id="goalPost" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ff8080" />
            <stop offset="50%" stopColor="#ff2020" />
            <stop offset="100%" stopColor="#a00000" />
          </linearGradient>
          {/* Floor reflection ellipse */}
          <radialGradient id="floorShadow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(0,0,0,0.50)" />
            <stop offset="100%" stopColor="rgba(0,0,0,0)" />
          </radialGradient>
        </defs>

        {/* Floor shadow ellipse below the net */}
        <ellipse cx="60" cy="82" rx="55" ry="2" fill="url(#floorShadow)" />

        {/* ── Net backstop — deep box with radial vignette ───────────────── */}
        <rect x="10" y="8" width="100" height="68" fill="url(#netDepth)" />

        {/* ── Dense netting (mesh) — two layers for depth ─────────────────*/}
        <rect x="10" y="8" width="100" height="68" fill="url(#netGrid2)" />
        <rect x="10" y="8" width="100" height="68" fill="url(#netHatch)" />
        {/* Vertical strands — thicker, more visible */}
        {Array.from({ length: 20 }).map((_, i) => (
          <line key={`v${i}`} x1={10 + (i + 1) * 5} y1="8" x2={10 + (i + 1) * 5} y2="76" stroke="rgba(255,255,255,0.13)" strokeWidth="0.18" />
        ))}
        {/* Horizontal strands */}
        {Array.from({ length: 14 }).map((_, i) => (
          <line key={`h${i}`} x1="10" y1={8 + (i + 1) * 5} x2="110" y2={8 + (i + 1) * 5} stroke="rgba(255,255,255,0.13)" strokeWidth="0.18" />
        ))}

        {/* Zone fills — placed inside the net (zones x:18..102, y:14..72) */}
        {zones.map((z) => {
          const isHover = hover === z.id;
          const t = tone(z.val);
          // Re-map original 8..92 coords into 18..102
          const X = 18 + (z.x - 8) * (84 / 84);
          const Y = 14 + (z.y - 8) * (58 / 60);
          const W = z.w * (84 / 84);
          const H = z.h * (58 / 60);
          return (
            <g key={z.id}
              onMouseEnter={() => setHover(z.id)}
              onMouseLeave={() => setHover(null)}
              style={{ cursor: "pointer" }}
            >
              <rect
                x={X} y={Y} width={W} height={H}
                fill={t.fill}
                stroke={t.stroke}
                strokeWidth={isHover ? 0.8 : 0.4}
                strokeOpacity={isHover ? 1 : 0.75}
                style={{
                  filter: isHover ? `drop-shadow(0 0 5px ${t.stroke})` : undefined,
                  transition: "stroke-width 200ms ease, stroke-opacity 200ms ease",
                }}
              />
              <text x={X + W / 2} y={Y + H / 2 + 1.6}
                textAnchor="middle"
                fontSize={isHover ? 13 : 11}
                fontWeight="800"
                fill={t.stroke}
                style={{ fontFamily: "var(--font-mono)", transition: "font-size 200ms ease" }}>
                {z.label}
              </text>
              <text x={X + W / 2} y={Y + H / 2 + 8}
                textAnchor="middle"
                fontSize="3.6"
                fill="rgba(255,255,255,0.80)"
                style={{ fontFamily: "var(--font-mono)", letterSpacing: "0.10em" }}>
                {fmt(z.val)}
              </text>
            </g>
          );
        })}

        {/* ── Goal frame — thick red posts + crossbar with metal gradient ── */}
        {/* Crossbar (top horizontal) */}
        <rect x="10" y="6" width="100" height="3" fill="url(#goalPost)" stroke="#600" strokeWidth="0.3" rx="0.5" />
        {/* Left post */}
        <rect x="8" y="8" width="3" height="70" fill="url(#goalPost)" stroke="#600" strokeWidth="0.3" rx="0.5" />
        {/* Right post */}
        <rect x="109" y="8" width="3" height="70" fill="url(#goalPost)" stroke="#600" strokeWidth="0.3" rx="0.5" />
        {/* Bottom skate-pad rail — neutral tone so the eye reads it as ice, not a red line */}
        <rect x="11" y="76" width="98" height="1.2" fill="rgba(255,255,255,0.18)" />

        {/* Top scan line (animated sweep across the net) */}
        <rect x="12" y="8" width="96" height="1.4" fill={teamColor} fillOpacity="0.22"
          style={{ mixBlendMode: "screen", animation: "netScan 5s linear infinite" }} />

        {/* Corner reticles — pushed outside the frame */}
        <g fill={teamColor} fillOpacity="0.85">
          <rect x="2" y="2" width="3.5" height="0.4" />
          <rect x="2" y="2" width="0.4" height="3.5" />
          <rect x="114.5" y="2" width="3.5" height="0.4" />
          <rect x="117.6" y="2" width="0.4" height="3.5" />
          <rect x="2" y="84" width="3.5" height="0.4" />
          <rect x="2" y="84" width="0.4" height="3.5" />
          <rect x="114.5" y="84" width="3.5" height="0.4" />
          <rect x="117.6" y="84" width="0.4" height="3.5" />
        </g>
      </svg>

      {/* Top hover callout */}
      <div className="absolute top-1 left-1 hud-mono text-[9px] uppercase tracking-[0.18em] px-2 py-0.5 rounded"
        style={{ color: teamColor, background: "rgba(0,0,0,0.55)", backdropFilter: "blur(4px)" }}>
        ◢ NET HEATMAP
      </div>
      {hover && (
        <div className="absolute top-1 right-1 hud-mono text-[10px] uppercase tracking-[0.18em] px-2 py-1 rounded border pointer-events-none"
          style={{
            color: teamColor,
            borderColor: `${teamColor}55`,
            background: "rgba(0,0,0,0.65)",
            backdropFilter: "blur(6px)",
          }}>
          ▸ Z{hover} · {fmt(zones.find(z => z.id === hover)?.val ?? null)}
        </div>
      )}

      {/* Legend strip */}
      <div className="mt-2 flex items-center justify-center gap-2 flex-wrap">
        <span className="hud-mono text-[8px] uppercase tracking-[0.16em] text-[var(--text-secondary)]">Heat:</span>
        {[
          { label: "<.84", color: "rgba(248,113,113,1)" },
          { label: ".84+", color: "rgba(251,191,36,1)" },
          { label: ".88+", color: teamColor },
          { label: ".92+", color: "rgba(74,222,128,1)" },
        ].map(l => (
          <span key={l.label} className="inline-flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-sm" style={{ background: l.color }} />
            <span className="hud-mono text-[8px] tracking-[0.14em] text-[var(--text-secondary)]">{l.label}</span>
          </span>
        ))}
      </div>

      <style jsx>{`
        @keyframes netScan {
          0%   { transform: translateY(0); }
          100% { transform: translateY(66px); }
        }
        @media (prefers-reduced-motion: reduce) {
          rect { animation: none !important; }
        }
      `}</style>
    </div>
  );
}

/**
 * GoalieSaveLocationMap — half-rink heatmap of save% by shot origin zone.
 *
 * NHL EDGE goalie pages show a per-rink-zone save% so you can tell at a
 * glance "this goalie reads the slot but bleeds from the left wing" — the
 * GoalieZoneViz only tells you about net mouth coverage (HD vs MD vs LD on
 * the net itself). This is the complement: same MoneyPuck shots-against
 * data we already render in the dot map, but bucketed into ice zones and
 * colored by save% with the same ramp as the net viz, so the two readouts
 * line up.
 *
 * Coordinates match the rotated frame used by HalfRinkMarkings + the dot
 * maps: outer svg is 85×100, child `<g>` is rotated -90 so inside the
 * group cx/cy use the horizontal-rink frame (x: 0 blue line side → 100 end
 * boards; y: 0 top → 85 bottom; goal at x≈89, y≈42.5).
 */
function GoalieSaveLocationMap({ shots, teamColor = "var(--brand-hex)" }: { shots: ShotPoint[]; teamColor?: string }) {
  const [hover, setHover] = useState<string | null>(null);

  // Zone bboxes defined in the pre-rotated frame used by ZoneTendencyMap
  // (viewBox 85×86, net at top y=11, blue line at bottom y=75) so labels
  // read horizontally and the viz matches the OZ TENDENCY · 2D look.
  // Coord mapping from raw NHL shot coords (x: 0→100, y: ±42.5):
  //   preX = NHL.y + 42.5
  //   preY = 100 - NHL.x
  type Zone = { id: string; label: string; short: string; bbox: { x0: number; x1: number; y0: number; y1: number } };
  const zones: Zone[] = [
    { id: "slot_hd",  label: "High Slot",    short: "SLOT", bbox: { x0: 32, x1: 53, y0: 11, y1: 22 } },
    { id: "slot_mid", label: "Mid Slot",     short: "MID",  bbox: { x0: 30, x1: 55, y0: 22, y1: 45 } },
    { id: "wing_l",   label: "Left Wing",    short: "LW",   bbox: { x0: 0,  x1: 30, y0: 22, y1: 50 } },
    { id: "wing_r",   label: "Right Wing",   short: "RW",   bbox: { x0: 55, x1: 85, y0: 22, y1: 50 } },
    { id: "corn_l",   label: "Left Corner",  short: "LC",   bbox: { x0: 0,  x1: 32, y0: 0,  y1: 22 } },
    { id: "corn_r",   label: "Right Corner", short: "RC",   bbox: { x0: 53, x1: 85, y0: 0,  y1: 22 } },
    { id: "pt_l",     label: "Left Point",   short: "LP",   bbox: { x0: 0,  x1: 42, y0: 50, y1: 75 } },
    { id: "pt_r",     label: "Right Point",  short: "RP",   bbox: { x0: 43, x1: 85, y0: 50, y1: 75 } },
  ];

  // Bucket each shot into the first containing zone using the pre-rotated
  // coordinate transform. Shots from behind the net or outside the OZ are
  // dropped (they're shown as dots in the shot-map tab anyway).
  const counts: Record<string, { shots: number; goals: number }> = {};
  for (const z of zones) counts[z.id] = { shots: 0, goals: 0 };
  for (const s of shots) {
    const px = s.y + 42.5;
    const py = 100 - s.x;
    for (const z of zones) {
      if (px >= z.bbox.x0 && px <= z.bbox.x1 && py >= z.bbox.y0 && py <= z.bbox.y1) {
        counts[z.id].shots += 1;
        if (s.goal) counts[z.id].goals += 1;
        break;
      }
    }
  }

  // Save% per zone — null when the sample is too small to colour honestly.
  const svFor = (id: string): number | null => {
    const c = counts[id];
    if (!c || c.shots < 8) return null;
    return (c.shots - c.goals) / c.shots;
  };

  // Colour ramp — same one used by GoalieZoneViz so the two readouts feel
  // like instruments on the same console.
  const tone = (pct: number | null) => {
    if (pct == null) return { fill: "rgba(148,163,184,0.10)", stroke: "rgba(148,163,184,0.40)" };
    if (pct >= 0.92) return { fill: "rgba(74,222,128,0.34)",  stroke: "rgba(74,222,128,0.95)" };
    if (pct >= 0.88) return { fill: `${teamColor}38`,         stroke: teamColor };
    if (pct >= 0.84) return { fill: "rgba(251,191,36,0.32)",  stroke: "rgba(251,191,36,0.95)" };
    return                  { fill: "rgba(248,113,113,0.34)", stroke: "rgba(248,113,113,0.95)" };
  };

  const fmt = (pct: number | null) => pct == null ? "—" : `${(pct * 100).toFixed(1)}%`;
  const totalShots = shots.length;

  // Hottest + coldest zone (only when populated) — drive the pulse rings.
  const ranked = zones
    .map(z => ({ id: z.id, pct: svFor(z.id) }))
    .filter(r => r.pct != null) as { id: string; pct: number }[];
  const hottest = ranked.length ? ranked.reduce((a, b) => (b.pct > a.pct ? b : a)).id : null;
  const coldest = ranked.length ? ranked.reduce((a, b) => (b.pct < a.pct ? b : a)).id : null;

  // Half-rink ice path — net at TOP, rounded boards on the top corners,
  // straight on the bottom (blue-line edge). Identical silhouette to
  // ZoneTendencyMap so the goalie viz feels like part of the same set.
  const icePath = "M 0,85 L 0,14 Q 0,0 14,0 L 71,0 Q 85,0 85,14 L 85,85 Z";

  return (
    <div className="relative w-full max-w-[420px] mx-auto"
      style={{ ["--gsl-color" as string]: teamColor }}>
      <svg viewBox="0 0 85 86" width="100%" className="block"
        style={{ filter: `drop-shadow(0 6px 14px rgba(0,0,0,0.55)) drop-shadow(0 0 14px ${teamColor}22)` }}>
        <defs>
          <clipPath id="gslClip">
            <path d={icePath} />
          </clipPath>
          <radialGradient id="gslIceGlow" cx="50%" cy="30%" r="55%">
            <stop offset="0%"  stopColor={teamColor} stopOpacity="0.12" />
            <stop offset="100%" stopColor={teamColor} stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* HUD-themed dark ice surface — matches ZoneTendencyMap palette */}
        <path d={icePath} fill="#0b0f1a" stroke={teamColor} strokeOpacity="0.55" strokeWidth="0.6" />
        <path d={icePath} fill="url(#gslIceGlow)" />
        {/* Faint background grid — sensor-deck texture, not a real rink */}
        {[20, 40, 60].map((y) => (
          <line key={`hg${y}`} x1="0" y1={y} x2="85" y2={y} stroke="rgba(255,255,255,0.04)" strokeWidth="0.2" clipPath="url(#gslClip)" />
        ))}
        {[20, 40, 60].map((x) => (
          <line key={`vg${x}`} x1={x} y1="0" x2={x} y2="86" stroke="rgba(255,255,255,0.04)" strokeWidth="0.2" clipPath="url(#gslClip)" />
        ))}

        {/* Zone tiles — coloured by save%, hover-glow + hot/cold pulse */}
        {zones.map((z, i) => {
          const pct = svFor(z.id);
          const t = tone(pct);
          const isHover = hover === z.id;
          const isHot   = z.id === hottest && pct != null && pct >= 0.88;
          const isCold  = z.id === coldest && pct != null && pct < 0.85;
          const cx = (z.bbox.x0 + z.bbox.x1) / 2;
          const cy = (z.bbox.y0 + z.bbox.y1) / 2;
          const w  = z.bbox.x1 - z.bbox.x0;
          const h  = z.bbox.y1 - z.bbox.y0;
          return (
            <g key={z.id}
              onMouseEnter={() => setHover(z.id)}
              onMouseLeave={() => setHover(null)}
              style={{
                cursor: "pointer",
                transformOrigin: `${cx}px ${cy}px`,
                animation: `gslZoneIn 500ms cubic-bezier(0.22,1,0.36,1) ${i * 70}ms both`,
              }}>
              <rect x={z.bbox.x0} y={z.bbox.y0} width={w} height={h}
                fill={t.fill}
                stroke={t.stroke}
                strokeOpacity={isHover ? 1 : 0.78}
                strokeWidth={isHover ? "1.0" : "0.7"}
                clipPath="url(#gslClip)"
                style={{
                  transition: "stroke-opacity 200ms ease, stroke-width 200ms ease, fill-opacity 200ms ease",
                  filter: isHover ? `drop-shadow(0 0 4px ${t.stroke})` : (isHot ? `drop-shadow(0 0 3px ${t.stroke}aa)` : undefined),
                }} />
              {(isHot || isCold) && (
                <rect x={z.bbox.x0} y={z.bbox.y0} width={w} height={h}
                  fill="none" stroke={t.stroke} strokeWidth="0.5"
                  clipPath="url(#gslClip)"
                  style={{
                    transformOrigin: `${cx}px ${cy}px`,
                    animation: `gslPulse ${isHot ? "2.6s" : "3.4s"} ease-in-out infinite`,
                  }} />
              )}
              {/* Zone label + save% — drawn horizontally so they read like
                  HUD readouts, not sideways text. Slot-zone gets the team
                  colour, all others get the same tone as their stroke. */}
              <text x={cx} y={cy - 1.2} textAnchor="middle"
                fontSize="2.6" fontWeight="600"
                fill={pct == null ? "rgba(255,255,255,0.55)" : t.stroke}
                style={{ fontFamily: "var(--font-mono)", letterSpacing: "0.18em" }}>
                {z.short}
              </text>
              <text x={cx} y={cy + 3.2} textAnchor="middle"
                fontSize="3.6" fontWeight="800"
                fill={pct == null ? "rgba(255,255,255,0.50)" : "rgba(255,255,255,0.95)"}
                style={{ fontFamily: "var(--font-mono)" }}>
                {pct == null ? "—" : `${(pct * 100).toFixed(1)}%`}
              </text>
            </g>
          );
        })}

        {/* Ice markings — blue line at the bottom of the OZ, faint red
            goal line at the top so the orientation reads immediately. */}
        <line x1="0.5" y1="75" x2="84.5" y2="75" stroke="#60a5fa" strokeWidth="1.0" opacity="0.55" />
        <line x1="0.5" y1="11" x2="84.5" y2="11" stroke="#f87171" strokeWidth="0.7" opacity="0.65" />
        {/* Trapezoid behind the goal line */}
        <polygon points="28.5,0 56.5,0 51.5,11 33.5,11"
          fill="none" stroke="#f87171" strokeWidth="0.4" opacity="0.35" clipPath="url(#gslClip)" />
        {/* Crease — D-shape opening up into the slot */}
        <path d="M 36.5 11 A 6 6 0 0 0 48.5 11 Z"
          fill={teamColor} fillOpacity="0.10" stroke="#60a5fa" strokeWidth="0.6" opacity="0.7" />
        {/* Net rectangle */}
        <rect x="39.5" y="8.5" width="6" height="2.5" rx="0.5"
          fill="rgba(255,255,255,0.10)" stroke={teamColor} strokeWidth="0.5" opacity="0.85" />
        {/* OZ faceoff circles — left + right */}
        <circle cx="20.5" cy="31" r="15" fill="none" stroke="#f87171" strokeWidth="0.4" opacity="0.28" clipPath="url(#gslClip)" />
        <circle cx="64.5" cy="31" r="15" fill="none" stroke="#f87171" strokeWidth="0.4" opacity="0.28" clipPath="url(#gslClip)" />
        <circle cx="20.5" cy="31" r="0.85" fill="#f87171" opacity="0.6" />
        <circle cx="64.5" cy="31" r="0.85" fill="#f87171" opacity="0.6" />

        {/* Animated scan band — same idiom as ZoneTendencyMap's ztScan */}
        <rect x="0" y="0" width="85" height="2"
          fill={teamColor} fillOpacity="0.18"
          clipPath="url(#gslClip)"
          style={{ mixBlendMode: "screen", animation: "gslScan 4.8s linear infinite" }} />

        {/* Corner reticles outside the ice border so the panel feels
            instrument-grade — matches ZoneTendencyMap exactly. */}
        <g fill={teamColor} fillOpacity="0.75">
          <rect x="0.5" y="0.5" width="3" height="0.4" />
          <rect x="0.5" y="0.5" width="0.4" height="3" />
          <rect x="81" y="0.5" width="3" height="0.4" />
          <rect x="84.1" y="0.5" width="0.4" height="3" />
          <rect x="0.5" y="82" width="3" height="0.4" />
          <rect x="0.5" y="82" width="0.4" height="3" />
          <rect x="81" y="82" width="3" height="0.4" />
          <rect x="84.1" y="82" width="0.4" height="3" />
        </g>
      </svg>

      {/* Header chip — pulse-dot + team-color label */}
      <div className="absolute top-1.5 left-1.5 flex items-center gap-2 hud-mono text-[9px] uppercase tracking-[0.22em] px-2 py-1 border rounded-sm"
        style={{
          color: teamColor,
          borderColor: `${teamColor}55`,
          background: "rgba(0,0,0,0.62)",
          backdropFilter: "blur(6px)",
          boxShadow: `0 0 10px ${teamColor}22`,
        }}>
        <span className="hud-pulse-dot" style={{ background: teamColor, boxShadow: `0 0 4px ${teamColor}` }} />
        <span style={{ textShadow: `0 0 5px ${teamColor}77` }}>◢ SAVE % · SHOT LOCATION</span>
      </div>

      {hover && (() => {
        const z = zones.find(x => x.id === hover);
        if (!z) return null;
        const c = counts[z.id];
        const pct = svFor(z.id);
        const t = tone(pct);
        return (
          <div className="absolute top-1.5 right-1.5 hud-mono text-[10px] uppercase tracking-[0.18em] px-2.5 py-1.5 border rounded-sm pointer-events-none"
            style={{
              color: t.stroke,
              borderColor: t.stroke,
              background: "rgba(0,0,0,0.78)",
              backdropFilter: "blur(8px)",
              boxShadow: `0 0 12px ${t.stroke}44`,
            }}>
            <div className="font-semibold flex items-center gap-1.5">
              <span>▸</span><span>{z.label}</span>
            </div>
            <div className="text-[9px] tracking-[0.16em] text-white/75 mt-0.5 tabular-nums">
              {c.shots} sh · {c.goals} ga · {fmt(pct)}
            </div>
          </div>
        );
      })()}

      <div className="flex items-center justify-between mt-2 px-1">
        <span className="hud-mono text-[8px] uppercase tracking-[0.18em] text-[var(--text-muted)] tabular-nums">
          ▸ {totalShots} shots faced
        </span>
        <span className="hud-mono text-[8px] uppercase tracking-[0.16em] text-[var(--text-muted)]">
          zones &lt; 8 sh dimmed
        </span>
      </div>

      <style jsx>{`
        @keyframes gslZoneIn {
          from { transform: scale(0.86); opacity: 0; }
          to   { transform: scale(1);    opacity: 1; }
        }
        @keyframes gslScan {
          0%   { transform: translateY(0px); }
          100% { transform: translateY(83px); }
        }
        @keyframes gslPulse {
          0%, 100% { stroke-opacity: 0.85; transform: scale(1); }
          50%      { stroke-opacity: 0.25; transform: scale(1.03); }
        }
        @media (prefers-reduced-motion: reduce) {
          rect, g { animation: none !important; }
        }
      `}</style>
    </div>
  );
}

/** Offensive zone tendency map — half-rink with coloured zone overlays */
/**
 * ZoneTendencyMap — half-rink rotated CCW (net at bottom, boards at right).
 * ViewBox: 85 wide × 100 tall.  Net is at the bottom (y=100 edge).
 * Equivalent to rotating the original horizontal rink 90° CCW:
 *   original (x,y) → new (y, 100-x)
 *
 * Original rink was 100 wide × 85 tall:
 *   blue line x=25  → y=25
 *   goal line x=89  → y=89
 *   net x=89-100, y=38.5-46.5 → x=38.5-46.5, y=0-11
 *
 * Zones (original → rotated):
 *   Perimeter (OZ wide, x=25-89, y=0-85)  → x=0-85, y=11-75
 *   Top corner (x=55-89, y=0-30)          → x=0-30, y=11-45
 *   Bottom corner (x=55-89, y=55-85)      → x=55-85, y=11-45
 *   Slot (x=55-89, y=30-55)               → x=30-55, y=11-45
 *   Net front (x=78-89, y=35-50)          → x=35-50, y=11-22
 */
function ZoneTendencyMap({ data, teamColor }: { data: ProfileData; teamColor: string }) {
  const slot   = data.nn_shoot_slot_pct       ?? 0;
  const perim  = data.nn_shoot_perimeter_pct  ?? 0;
  const net    = data.nn_drive_net_pct        ?? 0;
  const corner = data.nn_battle_corner_pct    ?? 0;
  const hold   = data.nn_hold_corner_pct      ?? 0;

  const [hover, setHover] = useState<string | null>(null);

  // Fill opacity proportional to %, capped at 35% action share
  const op = (pct: number) => Math.min(pct / 35, 1) * 0.78;

  // Identify the hottest zone (highest weighted) — gets a pulse glow
  const zones = [
    { id: "slot",   val: slot   },
    { id: "perim",  val: perim  },
    { id: "net",    val: net    },
    { id: "corner", val: corner },
    { id: "hold",   val: hold   },
  ];
  const hottest = zones.reduce((a, b) => (b.val > a.val ? b : a), zones[0]).id;

  const icePath = "M 0,85 L 0,14 Q 0,0 14,0 L 71,0 Q 85,0 85,14 L 85,85 Z";

  return (
    <div className="relative w-full max-w-[420px] mx-auto"
      style={{ ["--zt-color" as string]: teamColor }}>
      <svg viewBox="0 0 85 86" width="100%" className="block">
        <defs>
          <clipPath id="ztClip">
            <path d={icePath} />
          </clipPath>
          <radialGradient id="ztIceGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%"  stopColor={teamColor} stopOpacity="0.10" />
            <stop offset="100%" stopColor={teamColor} stopOpacity="0" />
          </radialGradient>
          <filter id="ztGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="0.6" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* HUD-themed dark ice surface */}
        <path d={icePath} fill="#0b0f1a" stroke={teamColor} strokeOpacity="0.55" strokeWidth="0.6" />
        <path d={icePath} fill="url(#ztIceGlow)" />
        {/* Faint background grid */}
        {[20, 40, 60].map((y) => (
          <line key={`hg${y}`} x1="0" y1={y} x2="85" y2={y} stroke="rgba(255,255,255,0.04)" strokeWidth="0.2" clipPath="url(#ztClip)" />
        ))}
        {[20, 40, 60].map((x) => (
          <line key={`vg${x}`} x1={x} y1="0" x2={x} y2="86" stroke="rgba(255,255,255,0.04)" strokeWidth="0.2" clipPath="url(#ztClip)" />
        ))}

      {/* Zone fills with hover-glow + hottest-zone pulse */}
      {[
        { id: "perim",  x: 0,  y: 45, w: 85, h: 30, val: perim,  fillColor: teamColor },
        { id: "corner", x: 0,  y: 11, w: 30, h: 34, val: corner, fillColor: "#38bdf8" },
        { id: "hold",   x: 55, y: 11, w: 30, h: 34, val: hold,   fillColor: "#38bdf8" },
        { id: "slot",   x: 30, y: 11, w: 25, h: 34, val: slot,   fillColor: teamColor },
        { id: "net",    x: 35, y: 11, w: 15, h: 11, val: net,    fillColor: "#fbbf24" },
      ].map((z) => {
        const isHover = hover === z.id;
        const isHot = hottest === z.id && z.val > 2;
        const fillOp = op(z.val) * (isHover ? 1.25 : 1);
        return (
          <g
            key={z.id}
            onMouseEnter={() => setHover(z.id)}
            onMouseLeave={() => setHover(null)}
            style={{ cursor: "pointer" }}
          >
            <rect
              x={z.x} y={z.y} width={z.w} height={z.h}
              fill={z.fillColor}
              fillOpacity={fillOp}
              clipPath="url(#ztClip)"
              style={{
                transition: "fill-opacity 200ms ease",
                filter: isHover ? `drop-shadow(0 0 4px ${z.fillColor})` : (isHot ? `drop-shadow(0 0 3px ${z.fillColor}aa)` : undefined),
              }}
            />
            {z.val > 2 && (
              <rect
                x={z.x} y={z.y} width={z.w} height={z.h}
                fill="none"
                stroke={z.fillColor}
                strokeWidth={isHover ? "1.0" : "0.7"}
                strokeOpacity={isHover ? 1 : 0.85}
                clipPath="url(#ztClip)"
                style={{ transition: "stroke-opacity 200ms ease, stroke-width 200ms ease" }}
              />
            )}
            {/* Hot-zone pulse */}
            {isHot && (
              <rect
                x={z.x} y={z.y} width={z.w} height={z.h}
                fill="none" stroke={z.fillColor}
                strokeWidth="0.6"
                clipPath="url(#ztClip)"
                style={{
                  transformOrigin: `${z.x + z.w / 2}px ${z.y + z.h / 2}px`,
                  animation: "ztPulse 2.6s ease-in-out infinite",
                }}
              />
            )}
          </g>
        );
      })}

      {/* Ice markings — HUD palette */}
      <line x1="0.5" y1="75" x2="84.5" y2="75" stroke="#60a5fa" strokeWidth="1.0" opacity="0.55" />
      <line x1="0.5" y1="11" x2="84.5" y2="11" stroke="#f87171" strokeWidth="0.7" opacity="0.65" />
      <polygon points="28.5,0 56.5,0 51.5,11 33.5,11"
        fill="none" stroke="#f87171" strokeWidth="0.4" opacity="0.35" clipPath="url(#ztClip)" />
      <rect x="39.5" y="8.5" width="6" height="2.5" rx="0.5" fill="rgba(255,255,255,0.10)" stroke={teamColor} strokeWidth="0.5" opacity="0.85" />
      <path d="M 36.5 11 A 6 6 0 0 0 48.5 11 Z"
        fill={teamColor} fillOpacity="0.08" stroke="#60a5fa" strokeWidth="0.6" opacity="0.7" />
      <circle cx="20.5" cy="31" r="15" fill="none" stroke="#f87171" strokeWidth="0.4" opacity="0.30" />
      <circle cx="64.5" cy="31" r="15" fill="none" stroke="#f87171" strokeWidth="0.4" opacity="0.30" />
      <circle cx="20.5" cy="31" r="0.85" fill="#f87171" opacity="0.65" />
      <circle cx="64.5" cy="31" r="0.85" fill="#f87171" opacity="0.65" />
      {/* Interior hashmarks — left OZ circle (⊞, tight around dot) */}
      <line x1="19" y1="27" x2="19" y2="30" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="22" y1="27" x2="22" y2="30" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="19" y1="32" x2="19" y2="35" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="22" y1="32" x2="22" y2="35" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="16.5" y1="29.5" x2="19.5" y2="29.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="21.5" y1="29.5" x2="24.5" y2="29.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="16.5" y1="32.5" x2="19.5" y2="32.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="21.5" y1="32.5" x2="24.5" y2="32.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      {/* Exterior hashmarks — left OZ circle (2 marks outside ring, left & right) */}
      <line x1="3" y1="29.5" x2="5.5" y2="29.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="3" y1="32.5" x2="5.5" y2="32.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="35.5" y1="29.5" x2="38" y2="29.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="35.5" y1="32.5" x2="38" y2="32.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      {/* Interior hashmarks — right OZ circle */}
      <line x1="63" y1="27" x2="63" y2="30" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="66" y1="27" x2="66" y2="30" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="63" y1="32" x2="63" y2="35" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="66" y1="32" x2="66" y2="35" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="60.5" y1="29.5" x2="63.5" y2="29.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="65.5" y1="29.5" x2="68.5" y2="29.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="60.5" y1="32.5" x2="63.5" y2="32.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="65.5" y1="32.5" x2="68.5" y2="32.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      {/* Exterior hashmarks — right OZ circle (2 marks outside ring, left & right) */}
      <line x1="47" y1="29.5" x2="49.5" y2="29.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="47" y1="32.5" x2="49.5" y2="32.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="79.5" y1="29.5" x2="82" y2="29.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />
      <line x1="79.5" y1="32.5" x2="82" y2="32.5" stroke="#cc2222" strokeWidth="0.6" opacity="0.35" />

      {/* Zone labels — HUD-mono on dark ice */}
      <text x="42.5" y="55" textAnchor="middle" fontSize="2.2" fontWeight="600" fill="rgba(255,255,255,0.55)" fontFamily="var(--font-mono)" letterSpacing="0.45">PERIM</text>
      <text x="42.5" y="62" textAnchor="middle" fontSize="4" fontWeight="800" fill="rgba(255,255,255,0.92)" fontFamily="var(--font-mono)">{perim.toFixed(0)}%</text>

      <text x="7" y="14" textAnchor="middle" fontSize="1.9" fontWeight="600" fill="rgba(255,255,255,0.55)" fontFamily="var(--font-mono)" letterSpacing="0.3">CORNER</text>
      <text x="7" y="18.5" textAnchor="middle" fontSize="3.2" fontWeight="800" fill="rgba(255,255,255,0.92)" fontFamily="var(--font-mono)">{corner.toFixed(0)}%</text>

      <text x="78" y="14" textAnchor="middle" fontSize="1.9" fontWeight="600" fill="rgba(255,255,255,0.55)" fontFamily="var(--font-mono)" letterSpacing="0.3">CORNER</text>
      <text x="78" y="18.5" textAnchor="middle" fontSize="3.2" fontWeight="800" fill="rgba(255,255,255,0.92)" fontFamily="var(--font-mono)">{hold.toFixed(0)}%</text>

      <text x="42.5" y="37" textAnchor="middle" fontSize="2" fontWeight="600" fill={teamColor} fillOpacity="0.85" fontFamily="var(--font-mono)" letterSpacing="0.4">SLOT</text>
      <text x="42.5" y="42" textAnchor="middle" fontSize="4" fontWeight="800" fill={teamColor} fontFamily="var(--font-mono)">{slot.toFixed(0)}%</text>

      <text x="42.5" y="20" textAnchor="middle" fontSize="1.7" fontWeight="600" fill="#fbbf24" fillOpacity="0.85" fontFamily="var(--font-mono)" letterSpacing="0.3">NET FRONT</text>
      <text x="42.5" y="23.5" textAnchor="middle" fontSize="2.8" fontWeight="800" fill="#fbbf24" fontFamily="var(--font-mono)">{net.toFixed(0)}%</text>

      {/* Animated scan line sweeping the rink */}
      <rect x="0" y="0" width="85" height="2"
        fill={teamColor} fillOpacity="0.18"
        style={{
          mixBlendMode: "screen",
          animation: "ztScan 4.6s linear infinite",
        }}
        clipPath="url(#ztClip)"
      />

      {/* Top corner reticles */}
      <g fill={teamColor} fillOpacity="0.7">
        <rect x="0.5" y="0.5" width="3" height="0.4" />
        <rect x="0.5" y="0.5" width="0.4" height="3" />
        <rect x="81" y="0.5" width="3" height="0.4" />
        <rect x="84.1" y="0.5" width="0.4" height="3" />
        <rect x="0.5" y="82" width="3" height="0.4" />
        <rect x="0.5" y="82" width="0.4" height="3" />
        <rect x="81" y="82" width="3" height="0.4" />
        <rect x="84.1" y="82" width="0.4" height="3" />
      </g>
    </svg>

    {/* Active zone callout */}
    {hover && (
      <div className="absolute top-2 right-2 hud-mono text-[10px] uppercase tracking-[0.18em] px-2 py-1 rounded border pointer-events-none"
        style={{
          color: teamColor,
          borderColor: `${teamColor}55`,
          background: "rgba(0,0,0,0.60)",
          backdropFilter: "blur(6px)",
        }}>
        ▸ {hover.toUpperCase()} · {(
          hover === "slot" ? slot :
          hover === "perim" ? perim :
          hover === "net" ? net :
          hover === "corner" ? corner :
          hold
        ).toFixed(1)}%
      </div>
    )}

    <style jsx>{`
      @keyframes ztScan {
        0%   { transform: translateY(11px); }
        100% { transform: translateY(75px); }
      }
      @keyframes ztPulse {
        0%, 100% { stroke-opacity: 0.85; transform: scale(1); }
        50%      { stroke-opacity: 0.30; transform: scale(1.02); }
      }
      @media (prefers-reduced-motion: reduce) {
        rect { animation: none !important; }
      }
    `}</style>
    </div>
  );
}

/** Ice Time By Zone — horizontal bar charts (OZ / NZ / DZ) */
/**
 * EdgeMetricsCard — surfaces the NHL-EDGE-style "Top Shot Speed / Top Skating
 * Speed / Skating Distance / Hard Shots" rows that NHL EDGE puts at the top
 * of every skater page (see nhl.com/nhl-edge/skaters/<player>). Each row
 * pairs the player's value with their league rank + percentile, with a small
 * gradient bar that mirrors how NHL EDGE shows the position vs the field.
 *
 * The data is already on disk (edge_skating_*.parquet, edge_shot_speed_*.parquet);
 * the API endpoint computes rank/percentile per request so the UI can stay dumb.
 */
const KMH_TO_MPH = 0.621371;

function EdgeStatRow({
  label, sub, value, rank, pop, pct, teamColor, threshold, delay = 0,
}: {
  label: string;
  sub?: string;
  value: string;
  rank?: number | null;
  pop?: number | null;
  pct?: number | null;        // 0-100; higher = better
  teamColor: string;
  threshold?: { ok: boolean; text: string } | null;
  delay?: number;             // stagger animation per row index
}) {
  // Percentile drives both the bar fill and the colour. Above-90 = elite
  // (greens), 60-90 = solid (team accent), 40-60 = average (slate),
  // below 40 = below avg (amber/red). Matches NHL EDGE's own colour ramp.
  const pctClamped = pct == null ? null : Math.max(0, Math.min(100, pct));
  const tone = pctClamped == null
    ? { fill: `${teamColor}55`, text: "rgba(255,255,255,0.78)" }
    : pctClamped >= 90 ? { fill: "#5ee08a", text: "#5ee08a" }
    : pctClamped >= 60 ? { fill: teamColor,  text: teamColor }
    : pctClamped >= 40 ? { fill: "#94a3b8", text: "rgba(255,255,255,0.75)" }
    : pctClamped >= 20 ? { fill: "#fbbf24", text: "#fbbf24" }
    :                    { fill: "#f87171", text: "#f87171" };
  // Glow ring on elite rows — same treatment HUD panels use for "hot" cells.
  const elite = pctClamped != null && pctClamped >= 90;
  return (
    <div className="relative flex flex-col gap-1.5 py-2 px-2 -mx-1 rounded-sm group"
      style={{
        background: elite ? `linear-gradient(90deg, ${tone.fill}0a, transparent 70%)` : undefined,
      }}>
      {/* Left tick — colour-coded vertical accent so the eye can pick out
          elite vs warning rows at a glance, even before reading numbers. */}
      <span aria-hidden className="absolute left-0 top-2 bottom-2 w-px"
        style={{ background: tone.fill, opacity: pctClamped == null ? 0.18 : 0.55, boxShadow: `0 0 4px ${tone.fill}77` }} />
      <div className="flex items-baseline justify-between gap-2">
        <span className="hud-mono text-[9px] uppercase tracking-[0.20em] flex items-center gap-1.5"
          style={{ color: "var(--text-secondary)" }}>
          <span style={{ color: tone.fill, textShadow: `0 0 4px ${tone.fill}88` }}>▸</span>
          {label}
        </span>
        {rank != null && pop != null && (
          <span className="hud-mono text-[8px] uppercase tracking-[0.18em] text-[var(--text-muted)] tabular-nums">
            rank <span style={{ color: tone.text }}>{rank}</span> / {pop}
          </span>
        )}
      </div>
      <div className="flex items-baseline gap-3">
        <span className="hud-mono text-[16px] font-semibold tabular-nums leading-none"
          style={{
            color: tone.text,
            textShadow: `0 0 10px ${tone.fill}55, 0 0 2px ${tone.fill}33`,
          }}>
          {value}
        </span>
        {sub && (
          <span className="hud-mono text-[9px] uppercase tracking-[0.16em] text-[var(--text-muted)] tabular-nums">
            {sub}
          </span>
        )}
        {pctClamped != null && (
          <span className="ml-auto hud-mono text-[11px] tabular-nums font-semibold"
            style={{ color: tone.text, textShadow: `0 0 6px ${tone.fill}55` }}>
            {pctClamped.toFixed(0)}<span className="text-[7px] uppercase tracking-[0.20em] ml-0.5 opacity-60">PCTL</span>
          </span>
        )}
      </div>
      {pctClamped != null && (
        <div className="relative h-2 overflow-hidden"
          style={{
            background: "rgba(255,255,255,0.035)",
            border: `1px solid ${tone.fill}40`,
            boxShadow: `inset 0 0 4px ${tone.fill}22`,
          }}>
          {/* Animated bar fill */}
          <div className="h-full origin-left"
            style={{
              width: `${pctClamped}%`,
              background: `linear-gradient(90deg, ${tone.fill}cc 0%, ${tone.fill} 100%)`,
              boxShadow: `0 0 8px ${tone.fill}88, inset 0 0 6px ${tone.fill}55`,
              animation: `edgeBarFill 900ms cubic-bezier(0.22,1,0.36,1) ${delay}ms both`,
            }} />
          {/* Sweeping highlight — same treatment as IceTimeByZoneBars */}
          <div className="absolute top-0 bottom-0 w-6 pointer-events-none"
            style={{
              background: `linear-gradient(90deg, transparent, ${tone.fill}aa, transparent)`,
              animation: `edgeBarScan 3.6s linear infinite ${delay}ms`,
              mixBlendMode: "screen",
              opacity: pctClamped > 5 ? 0.65 : 0,
            }} />
          {/* Tick marks at quartile + median + 90th — HUD reads at a glance */}
          <span className="absolute top-0 bottom-0 w-px pointer-events-none" style={{ left: "25%", background: "rgba(255,255,255,0.10)" }} />
          <span className="absolute top-0 bottom-0 w-px pointer-events-none" style={{ left: "50%", background: "rgba(255,255,255,0.20)" }} />
          <span className="absolute top-0 bottom-0 w-px pointer-events-none" style={{ left: "75%", background: "rgba(255,255,255,0.10)" }} />
          <span className="absolute top-0 bottom-0 w-px pointer-events-none" style={{ left: "90%", background: "rgba(94,224,138,0.35)" }} />
        </div>
      )}
      {threshold && (
        <span className="hud-mono text-[8px] uppercase tracking-[0.16em] flex items-center gap-1"
          style={{ color: threshold.ok ? "#5ee08a" : "rgba(255,255,255,0.40)" }}>
          <span style={{ textShadow: threshold.ok ? "0 0 4px #5ee08a88" : undefined }}>{threshold.ok ? "▸" : "·"}</span>
          {threshold.text}
        </span>
      )}
      <style jsx>{`
        @keyframes edgeBarFill {
          from { transform: scaleX(0); opacity: 0.2; }
          to   { transform: scaleX(1); opacity: 1; }
        }
        @keyframes edgeBarScan {
          0%   { transform: translateX(-30px); }
          100% { transform: translateX(520px); }
        }
        @media (prefers-reduced-motion: reduce) {
          div { animation: none !important; }
        }
      `}</style>
    </div>
  );
}

function EdgeMetricsCard({ data, teamColor }: { data: ProfileData; teamColor: string }) {
  const shotMph    = data.edge_top_shot_speed_mph ?? null;
  const skateKmh   = data.edge_top_skating_speed_kmh ?? null;
  const skateMph   = skateKmh != null ? skateKmh * KMH_TO_MPH : null;
  const totalKm    = data.edge_total_distance_km ?? null;
  const perGameKm  = data.skating_distance_per_game_km ?? null;
  const avgKmh     = data.edge_avg_speed_kmh ?? data.skating_avg_speed_kmh ?? null;
  const gp         = data.edge_games_played ?? data.skating_games_sample ?? null;
  const hardShots  = data.edge_hard_shot_count ?? null;
  const hdShots    = data.edge_high_danger_shots ?? null;

  // The 22 mph (35.4 km/h) threshold is NHL EDGE's headline "burst" marker.
  // We don't have per-shift bursts yet, but a season-top max above the bar
  // is itself a meaningful flag — surfaced as a small chip instead of a count.
  const burst22 = skateKmh != null ? skateKmh >= 35.4 : null;
  const burst20 = skateKmh != null ? skateKmh >= 32.2 : null;

  // No EDGE data at all? Don't render the card — better than half-empty.
  if (shotMph == null && skateKmh == null && totalKm == null && perGameKm == null) return null;

  // Assemble in row order so we can stagger animation delays as the panel
  // boots — same cadence the rest of the HUD uses on mount.
  const rows: { key: string; render: (i: number) => ReactNode }[] = [];
  if (shotMph != null) rows.push({ key: "shot", render: (i) => (
    <EdgeStatRow key="shot" delay={i * 90}
      label="Top Shot Speed"
      value={`${shotMph.toFixed(1)} mph`}
      sub={hardShots != null ? `${hardShots} shots ≥70 mph` : undefined}
      rank={data.edge_top_shot_speed_rank} pop={data.edge_top_shot_speed_pop} pct={data.edge_top_shot_speed_pct}
      teamColor={teamColor} />
  )});
  if (skateKmh != null) rows.push({ key: "skate", render: (i) => (
    <EdgeStatRow key="skate" delay={i * 90}
      label="Top Skating Speed"
      value={`${skateMph!.toFixed(1)} mph`}
      sub={`${skateKmh.toFixed(2)} km/h`}
      rank={data.edge_top_skating_speed_rank} pop={data.edge_top_skating_speed_pop} pct={data.edge_top_skating_speed_pct}
      teamColor={teamColor}
      threshold={
        burst22 === true ? { ok: true,  text: "Cleared 22 mph burst threshold" } :
        burst20 === true ? { ok: true,  text: "Cleared 20 mph burst threshold" } :
        burst20 === false ? { ok: false, text: "Below 20 mph burst threshold" } : null
      } />
  )});
  if (perGameKm != null || totalKm != null || avgKmh != null) rows.push({ key: "dist", render: (i) => (
    <EdgeStatRow key="dist" delay={i * 90}
      label="Skating Distance"
      value={perGameKm != null ? `${perGameKm.toFixed(2)} km/g` : (totalKm != null ? `${totalKm.toFixed(1)} km` : "—")}
      sub={[
        totalKm != null ? `${totalKm.toFixed(1)} km total` : null,
        avgKmh != null ? `${avgKmh.toFixed(1)} km/h avg` : null,
      ].filter(Boolean).join(" · ") || undefined}
      teamColor={teamColor} />
  )});
  if (hdShots != null) rows.push({ key: "hd", render: (i) => (
    <EdgeStatRow key="hd" delay={i * 90}
      label="High-Danger Shots"
      value={`${hdShots}`}
      sub={gp ? `${(hdShots / Math.max(gp, 1)).toFixed(2)} / game` : undefined}
      rank={data.edge_high_danger_shots_rank} pop={data.edge_high_danger_shots_pop} pct={data.edge_high_danger_shots_pct}
      teamColor={teamColor} />
  )});
  if (hardShots != null) rows.push({ key: "hard", render: (i) => (
    <EdgeStatRow key="hard" delay={i * 90}
      label="Hard Shots ≥70 mph"
      value={`${hardShots}`}
      sub={gp ? `${(hardShots / Math.max(gp, 1)).toFixed(2)} / game` : undefined}
      teamColor={teamColor} />
  )});

  return (
    <div className="hud-panel jarvis-boot jarvis-shimmer hud-panel--all-corners relative mt-3"
      style={{
        ["--hud-corner" as string]: teamColor,
        background: "linear-gradient(180deg, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.35) 100%)",
        boxShadow: `0 0 20px ${teamColor}1a, inset 0 0 30px rgba(0,0,0,0.5)`,
      }}>
      <span className="hud-panel__corner-tr" />
      <span className="hud-panel__corner-bl" />
      <div className="hud-scan" aria-hidden />
      {/* Header strip */}
      <div className="relative flex items-center justify-between px-3 py-2 border-b" style={{ borderColor: `${teamColor}26` }}>
        <div className="flex items-center gap-2">
          <span className="hud-pulse-dot" style={{ background: teamColor, boxShadow: `0 0 6px ${teamColor}` }} />
          <span className="hud-mono text-[10px] uppercase tracking-[0.24em] font-semibold"
            style={{ color: teamColor, textShadow: `0 0 6px ${teamColor}55` }}>
            ◢ EDGE METRICS
          </span>
          <span className="hud-mono text-[8px] uppercase tracking-[0.18em] text-[var(--text-muted)]">
            · NHL EDGE
          </span>
        </div>
        <span className="hud-mono text-[8px] uppercase tracking-[0.18em] tabular-nums"
          style={{ color: "rgba(255,255,255,0.45)" }}>
          {gp ? `${gp} GP` : "live"} · v1
        </span>
      </div>

      {/* Telemetry rows — stacked with hairline dividers, no rounded edges
          so the eye reads it as instrument cluster, not a card. */}
      <div className="relative px-2 py-1.5 divide-y" style={{ borderColor: `${teamColor}12` }}>
        {rows.map((r, i) => r.render(i))}
      </div>

      <div className="relative flex items-center justify-between px-3 py-1.5 border-t" style={{ borderColor: `${teamColor}1c` }}>
        <span className="hud-mono text-[8px] uppercase tracking-[0.18em] text-[var(--text-muted)]">
          ranks vs all skaters · EDGE coverage
        </span>
        <span className="hud-mono text-[8px] uppercase tracking-[0.16em]" style={{ color: `${teamColor}99` }}>
          live · 60Hz
        </span>
      </div>
    </div>
  );
}

/**
 * IceTimeByZoneBars — a single 100% horizontal bar split into three glowing
 * chambers (OZ · NZ · DZ). Reads like a HUD power-distribution meter, not a
 * stack of progress bars. Each chamber has its own colour glow, animated
 * fill on mount, a sweeping scan band, and a floating value chip. Pulse
 * dot in the header + corner reticles + dashed centerline tick so it sits
 * cleanly under the 3D rink without competing for attention.
 */
function IceTimeByZoneBars({ data, teamColor = "var(--brand-hex)" }: { data: ProfileData; teamColor?: string }) {
  const oz = data.skating_zone_time_oz_pct ?? 0;
  const dz = data.skating_zone_time_dz_pct ?? 0;
  const nz = Math.max(0, 100 - oz - dz);

  // Determine the dominant chamber so we can give it the pulse halo.
  const seg = [
    { id: "OZ", pct: oz, color: "#4ade80", tip: "Offensive zone — time spent in the attacking end" },
    { id: "NZ", pct: nz, color: "#fbbf24", tip: "Neutral zone — between the blue lines" },
    { id: "DZ", pct: dz, color: "#f87171", tip: "Defensive zone — time in your own end" },
  ];
  const dominant = seg.reduce((a, b) => (b.pct > a.pct ? b : a)).id;

  return (
    <div className="relative w-full rounded border px-3 pt-2.5 pb-3 overflow-hidden"
      style={{
        borderColor: `${teamColor}33`,
        background: "linear-gradient(180deg, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.32) 100%)",
        boxShadow: `0 0 12px ${teamColor}1a, inset 0 0 18px rgba(0,0,0,0.4)`,
      }}>
      {/* Corner reticles */}
      <span className="absolute top-1 left-1 w-2.5 h-2.5 pointer-events-none" style={{ borderLeft: `1px solid ${teamColor}88`, borderTop: `1px solid ${teamColor}88` }} />
      <span className="absolute top-1 right-1 w-2.5 h-2.5 pointer-events-none" style={{ borderRight: `1px solid ${teamColor}88`, borderTop: `1px solid ${teamColor}88` }} />
      <span className="absolute bottom-1 left-1 w-2.5 h-2.5 pointer-events-none" style={{ borderLeft: `1px solid ${teamColor}88`, borderBottom: `1px solid ${teamColor}88` }} />
      <span className="absolute bottom-1 right-1 w-2.5 h-2.5 pointer-events-none" style={{ borderRight: `1px solid ${teamColor}88`, borderBottom: `1px solid ${teamColor}88` }} />

      {/* Header strip */}
      <div className="flex items-center justify-between mb-2 px-0.5">
        <span className="hud-mono text-[9px] uppercase tracking-[0.22em] flex items-center gap-1.5"
          style={{ color: teamColor, textShadow: `0 0 5px ${teamColor}55` }}>
          <span className="hud-pulse-dot" style={{ background: teamColor, boxShadow: `0 0 4px ${teamColor}` }} />
          ◢ ICE TIME · ZONE DEPLOYMENT
        </span>
        <span className="hud-mono text-[8px] uppercase tracking-[0.18em] text-[var(--text-muted)]">
          % shift · 60Hz
        </span>
      </div>

      {/* Single 100% segmented bar — flex children sized by zone %. Each
          chamber owns its own glow + fill animation. Dominant chamber gets
          a pulsing halo so the dashboard reads at a glance. */}
      <div className="relative h-9 flex items-stretch rounded-sm overflow-hidden"
        style={{
          background: "rgba(255,255,255,0.025)",
          border: `1px solid ${teamColor}33`,
          boxShadow: `inset 0 0 12px rgba(0,0,0,0.45)`,
        }}>
        {seg.map((s, i) => {
          const pulse = s.id === dominant && s.pct >= 25;
          return (
            <div key={s.id}
              className="relative h-full flex items-center justify-center group cursor-help"
              title={`${s.tip} — ${s.pct.toFixed(1)}%`}
              style={{
                width: `${s.pct}%`,
                transformOrigin: "left center",
                animation: `iceSegIn 900ms cubic-bezier(0.22,1,0.36,1) ${i * 110}ms both`,
              }}>
              {/* Chamber fill — gradient + inner glow */}
              <div className="absolute inset-0"
                style={{
                  background: `linear-gradient(180deg, ${s.color}66 0%, ${s.color}aa 50%, ${s.color}77 100%)`,
                  boxShadow: `inset 0 0 12px ${s.color}55, 0 0 14px ${s.color}33`,
                }} />
              {/* Sweep scan band — per-chamber so the eye reads it as live */}
              <div className="absolute top-0 bottom-0 w-8 pointer-events-none"
                style={{
                  background: `linear-gradient(90deg, transparent, ${s.color}, transparent)`,
                  mixBlendMode: "screen",
                  opacity: s.pct > 5 ? 0.55 : 0,
                  animation: `iceSegScan ${3.6 + i * 0.4}s linear infinite ${i * 0.5}s`,
                }} />
              {/* Diagonal scanline pattern for sensor-deck texture */}
              <div className="absolute inset-0 pointer-events-none"
                style={{
                  background: `repeating-linear-gradient(45deg, transparent 0px, transparent 4px, ${s.color}1a 4px, ${s.color}1a 5px)`,
                  opacity: 0.4,
                }} />
              {/* Dominant-chamber pulse halo */}
              {pulse && (
                <div className="absolute inset-0 pointer-events-none"
                  style={{
                    boxShadow: `inset 0 0 20px ${s.color}aa, 0 0 18px ${s.color}66`,
                    animation: "iceSegPulse 2.6s ease-in-out infinite",
                  }} />
              )}
              {/* Value label — only show when chamber is wide enough */}
              {s.pct >= 8 && (
                <div className="relative z-10 flex flex-col items-center leading-none"
                  style={{
                    textShadow: `0 0 6px ${s.color}, 0 0 2px rgba(0,0,0,0.9)`,
                    animation: `iceSegLabelIn 700ms ease-out ${i * 110 + 300}ms both`,
                  }}>
                  <span className="hud-mono text-[10px] font-semibold tabular-nums" style={{ color: "rgba(255,255,255,0.96)" }}>
                    {s.pct.toFixed(0)}%
                  </span>
                  <span className="hud-mono text-[7px] uppercase tracking-[0.20em]" style={{ color: "rgba(255,255,255,0.85)" }}>
                    {s.id}
                  </span>
                </div>
              )}
              {/* Right-edge divider — bright vertical line between chambers */}
              {i < seg.length - 1 && (
                <span className="absolute right-0 top-0 bottom-0 w-px pointer-events-none z-20"
                  style={{
                    background: `linear-gradient(180deg, transparent 0%, ${teamColor}cc 30%, ${teamColor}cc 70%, transparent 100%)`,
                    boxShadow: `0 0 4px ${teamColor}`,
                  }} />
              )}
            </div>
          );
        })}

        {/* Top + bottom rails — slim accent lines so the bar reads as a
            framed sensor strip, not a raw progress bar. */}
        <span className="absolute top-0 left-0 right-0 h-px pointer-events-none"
          style={{ background: `linear-gradient(90deg, transparent, ${teamColor}aa, transparent)` }} />
        <span className="absolute bottom-0 left-0 right-0 h-px pointer-events-none"
          style={{ background: `linear-gradient(90deg, transparent, ${teamColor}aa, transparent)` }} />
      </div>

      {/* Legend pips below the bar — small dots with deltas vs 33%
          (perfectly balanced ice time = each zone 33%). Tells Bob "this
          player is +14pp tilted to the OZ" at a glance. */}
      <div className="flex items-center justify-between mt-2.5 px-0.5">
        {seg.map((s) => {
          const delta = s.pct - 33.3;
          const sign = delta > 0 ? "+" : "";
          return (
            <div key={`leg-${s.id}`} className="flex items-center gap-1.5">
              <span className="inline-block w-2 h-2 rounded-full"
                style={{ background: s.color, boxShadow: `0 0 5px ${s.color}` }} />
              <span className="hud-mono text-[8px] uppercase tracking-[0.18em]" style={{ color: `${s.color}cc` }}>
                {s.id}
              </span>
              <span className="hud-mono text-[8px] uppercase tracking-[0.18em] tabular-nums"
                style={{ color: Math.abs(delta) < 2 ? "rgba(255,255,255,0.35)" : delta > 0 ? "#5ee08a" : "#f87b7b" }}>
                {sign}{delta.toFixed(1)}
              </span>
            </div>
          );
        })}
      </div>

      <style jsx>{`
        @keyframes iceSegIn {
          from { transform: scaleX(0.05); opacity: 0; }
          to   { transform: scaleX(1);    opacity: 1; }
        }
        @keyframes iceSegLabelIn {
          from { opacity: 0; transform: translateY(2px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes iceSegScan {
          0%   { transform: translateX(-40px); }
          100% { transform: translateX(800px); }
        }
        @keyframes iceSegPulse {
          0%, 100% { opacity: 0.55; }
          50%      { opacity: 1;    }
        }
        @media (prefers-reduced-motion: reduce) {
          div, span { animation: none !important; }
        }
      `}</style>
    </div>
  );
}

// ---------------------------------------------------------------------------
// PlayerDnaStrip — "neural DNA" double-helix signature.
//
// The behavior NN already has 7 action heads (carry / dump / slot / perim /
// drive / battle / hold). Each becomes one rung of a helix; rung height =
// the action's probability, rung colour = the action's theme. Two sine-wave
// backbones cross over the rungs and slowly phase-shift so the strand reads
// as live signal rather than a static bar chart. The result is a per-player
// fingerprint that visually screams "this is who they are on the ice."
//
// Goalie variant swaps the 7 skater actions for 6 goalie axes (HD / MD / LD
// save %, GSAx, workload, overall SV%) drawn against the same helix.
// ---------------------------------------------------------------------------

interface DnaRung {
  id: string;
  label: string;       // 3-char code shown under the rung
  weight: number;      // 0-1 height
  color: string;       // theme color
  tip?: string;
}

function PlayerDnaStrip({
  rungs, teamColor, signature,
}: { rungs: DnaRung[]; teamColor: string; signature?: string }) {
  const [hover, setHover] = useState<number | null>(null);
  if (rungs.length === 0) return null;

  // SVG geometry — taller viewBox so the helix has room to twist visibly.
  // The two strands are sine waves offset by π so they weave; rungs land
  // at uniform phase positions where the strands are maximally apart, so
  // every rung's endpoints actually sit ON the strand curves (real DNA
  // base-pair behaviour, not a barcode behind decorative waves).
  const VB_W = 320;
  const VB_H = 110;
  const PAD_L = 18;
  const PAD_R = 18;
  const innerW = VB_W - PAD_L - PAD_R;
  const cy = VB_H / 2 + 4;       // shift down a hair to leave room for top base-pair labels
  const amp = 22;                // helix amplitude (wide enough to read as DNA)
  const n = rungs.length;
  // Pick t-values for each rung so they sit at phase π/2 + kπ — strand
  // extrema, where the rung spans full diameter (top strand at +amp,
  // bottom at -amp). t_i = (i + 0.5) / n places each rung centred in its
  // column with phase = (i + 0.5) * π.
  const phaseAt = (i: number) => (i + 0.5) * Math.PI;
  const xOf    = (i: number) => PAD_L + ((i + 0.5) / n) * innerW;
  // Strand A: sin((t * n * π))         → strand y_a at column i: sin(phase_i)
  // Strand B: sin((t * n * π) + π)     → strand y_b at column i: -sin(phase_i)
  // At column centres, sin(phase) = sin((i+0.5)*π) which alternates +1 / -1.
  // So odd-i and even-i rungs swap top/bottom — the strands genuinely cross.
  const strandY = (phase: number, t: number) => cy + Math.sin((t * n * Math.PI) + phase) * amp;
  const buildStrand = (phase: number): string => {
    const segs = 96;
    const pts: string[] = [];
    for (let s = 0; s <= segs; s++) {
      const t = s / segs;
      const x = PAD_L + t * innerW;
      const y = strandY(phase, t);
      pts.push(`${s === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`);
    }
    return pts.join(" ");
  };
  // Faux nucleotide letters — each rung gets one of the canonical pairs so
  // the strip reads as actual DNA bases. Deterministic from the rung id so
  // the same player always shows the same letters (stable fingerprint).
  const BASES: [string, string][] = [["A", "T"], ["T", "A"], ["G", "C"], ["C", "G"]];
  const baseFor = (id: string): [string, string] => {
    let h = 0;
    for (let k = 0; k < id.length; k++) h = (h * 31 + id.charCodeAt(k)) >>> 0;
    return BASES[h % BASES.length];
  };

  return (
    <div className="mt-2 rounded border px-2 py-2 relative overflow-hidden"
      style={{
        borderColor: `${teamColor}33`,
        background: "linear-gradient(180deg, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.32) 100%)",
        boxShadow: `0 0 12px ${teamColor}1a, inset 0 0 18px rgba(0,0,0,0.4)`,
      }}>
      {/* Header strip */}
      <div className="flex items-center justify-between mb-1 px-1">
        <div className="flex items-center gap-2">
          <span className="hud-pulse-dot" style={{ background: teamColor, boxShadow: `0 0 4px ${teamColor}` }} />
          <span className="hud-mono text-[9px] uppercase tracking-[0.22em]"
            style={{ color: teamColor, textShadow: `0 0 5px ${teamColor}55` }}>
            ◢ NEURAL DNA
          </span>
          <span className="hud-mono text-[8px] uppercase tracking-[0.18em] text-[var(--text-muted)]">
            · BNN signature
          </span>
        </div>
        {signature && (
          <span className="hud-mono text-[8px] uppercase tracking-[0.18em] tabular-nums" style={{ color: `${teamColor}99` }}>
            {signature}
          </span>
        )}
      </div>

      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" style={{ height: 110 }}>
        <defs>
          {/* Glow filter per-strand */}
          <filter id="dnaGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="0.9" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          {/* Vertical scan band moving across the helix */}
          <linearGradient id="dnaScan" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor={teamColor} stopOpacity="0" />
            <stop offset="50%" stopColor={teamColor} stopOpacity="0.55" />
            <stop offset="100%" stopColor={teamColor} stopOpacity="0" />
          </linearGradient>
          {/* Strand depth gradient — fades at edges so rungs appear to wrap
              behind the strands at peaks (faux 3D perspective). */}
          <linearGradient id="dnaDepth" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%"  stopColor="rgba(255,255,255,0)" />
            <stop offset="50%" stopColor={`${teamColor}1a`} />
            <stop offset="100%" stopColor="rgba(255,255,255,0)" />
          </linearGradient>
        </defs>

        {/* Backdrop depth wash so the helix appears to float over a recessed
            chamber. Centre band where rungs peak gets the lightest tone. */}
        <rect x={PAD_L - 6} y={cy - amp - 2} width={innerW + 12} height={amp * 2 + 4} fill="url(#dnaDepth)" rx="3" />

        {/* Centre baseline */}
        <line x1={PAD_L - 4} y1={cy} x2={VB_W - PAD_R + 4} y2={cy}
          stroke={`${teamColor}22`} strokeWidth="0.3" strokeDasharray="2 3" />

        {/* Two strands — solid sine waves that genuinely weave. Strand A
            renders BEHIND the rungs that are at front-phase (sin > 0),
            strand B renders IN FRONT of those same rungs — and vice versa
            at back-phase columns — so the strand visibly threads over and
            under the rungs like real DNA. We split each strand into two
            paths (forward / back segments) by clipping in z-order. */}
        <g style={{ filter: "url(#dnaGlow)" }}>
          {/* Back layer: strand B (will be drawn over the rungs at columns
              where strand A is in front). */}
          <path d={buildStrand(Math.PI)}
            fill="none" stroke={teamColor} strokeOpacity="0.55"
            strokeWidth="1.4" strokeLinecap="round"
            style={{ animation: "dnaWeaveB 5.6s ease-in-out infinite" }} />
        </g>

        {/* Rungs — each one connects the two strands at the rung's column
            x, so endpoints sit ON the sine curves (not floating). Weight
            drives stroke width + glow intensity, not height. */}
        {rungs.map((r, i) => {
          const x = xOf(i);
          // Strand positions at this column — sin((t*n*π) + phase) where
          // t = (i+0.5)/n. We just take ±amp depending on parity to land
          // exactly on the strand extrema.
          const yA = cy + Math.sin(phaseAt(i)) * amp;          // strand A
          const yB = cy + Math.sin(phaseAt(i) + Math.PI) * amp; // strand B
          const isHover = hover === i;
          // Front rungs (where yA < yB, i.e. strand A is on top) feel
          // "closer"; back rungs flip. We use this to tweak opacity so
          // the helix reads as 3D weave.
          const inFront = yA < yB;
          const baseOpacity = inFront ? 1 : 0.78;
          const w = Math.max(1.6, r.weight * 4.5);              // weight-driven thickness
          const [topBase, botBase] = baseFor(r.id);
          return (
            <g key={r.id}
              onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
              style={{
                cursor: "pointer",
                opacity: baseOpacity,
                animation: `dnaRungIn 700ms cubic-bezier(0.22,1,0.36,1) ${i * 100}ms both`,
              }}>
              {/* Rung — vertical line between strand endpoints. Thicker on
                  hover + glow scales with hover state. */}
              <line x1={x} y1={yA} x2={x} y2={yB}
                stroke={r.color}
                strokeWidth={isHover ? w + 1.4 : w}
                strokeLinecap="round"
                style={{
                  filter: `drop-shadow(0 0 ${isHover ? 7 : 4}px ${r.color})`,
                  transition: "stroke-width 200ms ease",
                }} />
              {/* Nucleotide bases at strand attachment points */}
              <circle cx={x} cy={yA} r={isHover ? 2.6 : 1.9} fill={r.color}
                style={{ filter: `drop-shadow(0 0 4px ${r.color})`, transition: "r 200ms ease" }} />
              <circle cx={x} cy={yB} r={isHover ? 2.6 : 1.9} fill={r.color}
                style={{ filter: `drop-shadow(0 0 4px ${r.color})`, transition: "r 200ms ease" }} />
              {/* Base-pair letter at each endpoint — tiny, mono, glows on
                  hover. Sells the DNA fingerprint vibe. */}
              <text x={x} y={yA - 3.5} textAnchor="middle"
                fontSize="3.4" fontWeight="700"
                fill={isHover ? "rgba(255,255,255,0.95)" : `${r.color}cc`}
                style={{
                  fontFamily: "var(--font-mono)", letterSpacing: "0.10em",
                  textShadow: isHover ? `0 0 4px ${r.color}` : undefined,
                  transition: "fill 200ms ease",
                }}>
                {topBase}
              </text>
              <text x={x} y={yB + 6} textAnchor="middle"
                fontSize="3.4" fontWeight="700"
                fill={isHover ? "rgba(255,255,255,0.95)" : `${r.color}cc`}
                style={{
                  fontFamily: "var(--font-mono)", letterSpacing: "0.10em",
                  textShadow: isHover ? `0 0 4px ${r.color}` : undefined,
                  transition: "fill 200ms ease",
                }}>
                {botBase}
              </text>
              {/* 3-char action code under the helix */}
              <text x={x} y={VB_H - 2} textAnchor="middle"
                fontSize="5" fontWeight={isHover ? "700" : "600"}
                fill={isHover ? r.color : `${r.color}aa`}
                style={{
                  fontFamily: "var(--font-mono)", letterSpacing: "0.18em",
                  textShadow: isHover ? `0 0 4px ${r.color}` : undefined,
                  transition: "fill 200ms ease",
                }}>
                {r.label}
              </text>
            </g>
          );
        })}

        {/* FRONT strand — drawn AFTER all rungs so it visibly threads over
            them, completing the woven-helix illusion. Brighter than strand
            B because it sits on top. */}
        <g style={{ filter: "url(#dnaGlow)" }}>
          <path d={buildStrand(0)}
            fill="none" stroke={teamColor} strokeOpacity="0.95"
            strokeWidth="1.7" strokeLinecap="round"
            style={{ animation: "dnaWeaveA 5.6s ease-in-out infinite" }} />
        </g>

        {/* Sweeping vertical scan band — sells the "live signal" vibe */}
        <rect x="0" y="0" width="8" height={VB_H} fill="url(#dnaScan)"
          style={{ mixBlendMode: "screen", animation: "dnaSweep 4.8s linear infinite" }} />

        <style>{`
          /* Strand weave — soft vertical bob in opposite directions so the
             helix appears to gently rotate around its axis. */
          @keyframes dnaWeaveA {
            0%, 100% { transform: translateY(0); }
            50%      { transform: translateY(-1.4px); }
          }
          @keyframes dnaWeaveB {
            0%, 100% { transform: translateY(0); }
            50%      { transform: translateY(1.4px); }
          }
          @keyframes dnaRungIn {
            from { transform: scaleY(0.2); opacity: 0; }
            to   { transform: scaleY(1);   opacity: 1; }
          }
          @keyframes dnaSweep {
            0%   { transform: translateX(0px); }
            100% { transform: translateX(${VB_W}px); }
          }
          @media (prefers-reduced-motion: reduce) {
            path, g, rect { animation: none !important; }
          }
        `}</style>
      </svg>

      {/* Hover detail — appears below the strip when a rung is focused */}
      {hover != null && (
        <div className="absolute bottom-1 left-1 right-1 flex items-center justify-between pointer-events-none">
          <span className="hud-mono text-[9px] uppercase tracking-[0.18em] tabular-nums px-2 py-0.5 rounded border"
            style={{
              color: rungs[hover].color,
              borderColor: `${rungs[hover].color}66`,
              background: "rgba(0,0,0,0.7)",
              backdropFilter: "blur(4px)",
              boxShadow: `0 0 8px ${rungs[hover].color}33`,
            }}>
            ▸ {rungs[hover].tip ?? rungs[hover].label} · {(rungs[hover].weight * 100).toFixed(1)}%
          </span>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// HologramScanner — interactive Iron Man-style body scanner
// ---------------------------------------------------------------------------
type HoloZone = "head" | "torso" | "arms" | "legs";

interface HoloCallout {
  id: string;
  label: string;
  val: string | null;
  target: HoloZone;
  /** Hover-info copy. Shown via title attribute on the callout. */
  tip?: string;
}

function HologramScanner({
  isGoalie,
  teamColor,
  bodyIntensity,
  telemetryLeft,
  telemetryRight,
  tickerLine,
  dnaRungs,
  dnaSignature,
}: {
  isGoalie: boolean;
  teamColor: string;
  bodyIntensity: Partial<Record<BodyZone, number>>;
  telemetryLeft: HoloCallout[];
  telemetryRight: HoloCallout[];
  tickerLine: string[];
  // Optional neural-DNA strip — surfaces the player's BNN action signature
  // (or goalie save-zone fingerprint) below the silhouette so the hologram
  // panel feels like a player-DNA scanner, not just a body diagram.
  dnaRungs?: DnaRung[];
  dnaSignature?: string;
}) {
  const [active, setActive] = useState<HoloZone | null>(null);

  // Targeting-reticle bounding boxes per zone, in the 200×340 silhouette.
  // Used to render corner brackets that converge on the active zone — the
  // Iron-Man "lock-on" treatment that earns the "hologram scanner" framing.
  const ZONE_BBOX: Record<HoloZone, { x: number; y: number; w: number; h: number }> = {
    head:  { x:  72, y:   0, w:  56, h:  64 },
    torso: { x:  46, y:  64, w: 108, h: 136 },
    arms:  { x:  18, y:  74, w: 164, h: 110 },
    legs:  { x:  68, y: 200, w:  64, h: 140 },
  };

  // Active zone label that animates in the dynamic readout. When no zone is
  // hovered we cycle "SCAN · LOCK · TGT · LIVE" so the corners feel alive.
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick(t => (t + 1) % 4), 1700);
    return () => clearInterval(id);
  }, []);
  const idleLabels = ["SCAN", "LOCK", "TGT", "LIVE"];
  const cornerLabel = active ? active.toUpperCase() : idleLabels[tick];

  // Zone hotspot dots mapped to body-zone groups
  const hotspots: { cx: number; cy: number; key: string; intensity: number; zone: HoloZone }[] = [
    { cx: 100, cy: 36,  key: "head",  intensity: bodyIntensity.head ?? 0,     zone: "head" },
    { cx: 60,  cy: 70,  key: "shdrL", intensity: bodyIntensity.shoulder ?? 0, zone: "torso" },
    { cx: 140, cy: 70,  key: "shdrR", intensity: bodyIntensity.shoulder ?? 0, zone: "torso" },
    { cx: 100, cy: 130, key: "torso", intensity: bodyIntensity.torso ?? 0,    zone: "torso" },
    { cx: 57,  cy: 140, key: "armL",  intensity: bodyIntensity.armL ?? 0,     zone: "arms" },
    { cx: 143, cy: 140, key: "armR",  intensity: bodyIntensity.armR ?? 0,     zone: "arms" },
    { cx: 88,  cy: 280, key: "legL",  intensity: bodyIntensity.legL ?? 0,     zone: "legs" },
    { cx: 112, cy: 280, key: "legR",  intensity: bodyIntensity.legR ?? 0,     zone: "legs" },
  ];

  return (
    <div className="flex flex-col h-full">
      {/* Scanner canvas — body + rings + callouts */}
      <div className="relative flex-1 flex items-center justify-center py-3 min-h-[400px]">
        {/* Soft radial glow backdrop */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: `radial-gradient(circle at 50% 50%, ${teamColor}30 0%, ${teamColor}08 35%, transparent 65%)`,
            filter: "blur(12px)",
          }}
        />
        {/* Rotating rings (Iron Man HUD) */}
        <svg viewBox="0 0 400 400" className="absolute inset-0 w-full h-full pointer-events-none" aria-hidden>
          <g style={{ transformOrigin: "200px 200px", animation: "ironRotateSlow 32s linear infinite" }}>
            <circle cx={200} cy={200} r={170} fill="none" stroke={teamColor} strokeOpacity={0.15} strokeDasharray="2 8" />
            {[0, 45, 90, 135, 180, 225, 270, 315].map((deg, i) => {
              const rad = (deg * Math.PI) / 180;
              return (
                <line key={i}
                  x1={200 + Math.cos(rad) * 162}
                  y1={200 + Math.sin(rad) * 162}
                  x2={200 + Math.cos(rad) * 178}
                  y2={200 + Math.sin(rad) * 178}
                  stroke={teamColor} strokeOpacity={0.55} strokeWidth={1.4} />
              );
            })}
          </g>
          <g style={{ transformOrigin: "200px 200px", animation: "ironRotateRev 18s linear infinite" }}>
            <circle cx={200} cy={200} r={140} fill="none" stroke={teamColor} strokeOpacity={0.45} strokeWidth={1.5} strokeDasharray="60 40 25 40" strokeLinecap="round" />
          </g>
          <g style={{ transformOrigin: "200px 200px", animation: "ironRotateFast 8s linear infinite" }}>
            <circle cx={200} cy={200} r={110} fill="none" stroke={teamColor} strokeOpacity={0.55} strokeWidth={1.2} strokeDasharray="20 80" strokeLinecap="round" />
          </g>
          <circle cx={200} cy={200} r={88} fill="none" stroke={teamColor} strokeOpacity={0.18} strokeDasharray="1 5" />
          <line x1={200} y1={0} x2={200} y2={400} stroke={teamColor} strokeOpacity={0.05} strokeDasharray="3 10" />
          <line x1={0} y1={200} x2={400} y2={200} stroke={teamColor} strokeOpacity={0.05} strokeDasharray="3 10" />
          <circle cx={200} cy={200} r={50} fill="none" stroke={teamColor} strokeOpacity={0.4} strokeWidth={1} style={{ animation: "ironPulse 4.4s ease-out infinite" }} />
          <circle cx={200} cy={200} r={50} fill="none" stroke={teamColor} strokeOpacity={0.4} strokeWidth={1} style={{ animation: "ironPulse 4.4s ease-out infinite 2.2s" }} />
        </svg>

        {/* Dynamic corner readouts — cycle while idle, lock onto the hovered
            zone when one is active. Each corner pairs a gold L-bracket SVG
            with a hud-mono code so the whole panel reads as instrumentation. */}
        {([
          { pos: "top-1 left-2",     align: "items-start", path: "M 0 6 L 0 0 L 6 0",  text: `▸ ${cornerLabel}`,     after: false },
          { pos: "top-1 right-2",    align: "items-end",   path: "M 0 0 L 6 0 L 6 6",  text: `${cornerLabel} ◂`,     after: true  },
          { pos: "bottom-1 left-2",  align: "items-start", path: "M 0 0 L 0 6 L 6 6",  text: "INTEL · 60Hz",         after: false },
          { pos: "bottom-1 right-2", align: "items-end",   path: "M 0 6 L 6 6 L 6 0",  text: "FEED · LIVE",          after: true  },
        ] as const).map((r, i) => (
          <div key={i} aria-hidden className={`absolute ${r.pos} flex items-center gap-1 ${r.align}`}>
            <svg width="7" height="7" viewBox="0 0 7 7" className="shrink-0"><path d={r.path} stroke={teamColor} strokeWidth="1" fill="none" /></svg>
            <span className="hud-mono text-[8px] uppercase tracking-[0.18em] tabular-nums" style={{ color: teamColor, textShadow: `0 0 6px ${teamColor}55` }}>
              {r.text}
            </span>
          </div>
        ))}

        {/* Silhouette + zone hotspots */}
        <div className="relative" style={{ width: 200, height: 340 }}>
          <BodySilhouette
            themeColor={teamColor}
            intensity={bodyIntensity}
            width={200}
            height={340}
            variant={isGoalie ? "goalie" : "skater"}
          />
          {/* Vertical scan-line — subtle, fades softly across the figure only */}
          <div
            aria-hidden
            className="absolute pointer-events-none"
            style={{
              left: "20%",
              right: "20%",
              top: 0,
              height: "1px",
              background: `linear-gradient(90deg, transparent, ${teamColor}66, transparent)`,
              boxShadow: `0 0 4px ${teamColor}44`,
              mixBlendMode: "screen",
              opacity: 0.55,
              animation: "holoVScan 6s ease-in-out infinite",
            }}
          />

          {/* Horizontal scan-line — counter-axis sweep so the figure feels
              actively mapped on both dimensions, not just top-to-bottom. */}
          <div
            aria-hidden
            className="absolute pointer-events-none"
            style={{
              top: "10%",
              bottom: "10%",
              left: 0,
              width: "1px",
              background: `linear-gradient(180deg, transparent, ${teamColor}55, transparent)`,
              boxShadow: `0 0 4px ${teamColor}33`,
              mixBlendMode: "screen",
              opacity: 0.45,
              animation: "holoHScan 9s ease-in-out infinite",
            }}
          />

          {/* Active-zone targeting reticle — 4 corner brackets converging on
              the hovered body zone bbox. Pure SVG so it animates in/out
              cheaply and stays crisp at any zoom. */}
          {active && (() => {
            const b = ZONE_BBOX[active];
            const pad = 4;
            const len = 10;
            const x1 = b.x - pad, y1 = b.y - pad;
            const x2 = b.x + b.w + pad, y2 = b.y + b.h + pad;
            return (
              <svg
                aria-hidden
                viewBox="0 0 200 340"
                className="absolute inset-0 pointer-events-none"
                style={{
                  width: "100%",
                  height: "100%",
                  filter: `drop-shadow(0 0 6px ${teamColor})`,
                  animation: "holoReticle 220ms cubic-bezier(0.22,1,0.36,1)",
                }}
              >
                {/* top-left */}
                <path d={`M ${x1} ${y1 + len} L ${x1} ${y1} L ${x1 + len} ${y1}`} stroke={teamColor} strokeWidth="1.4" fill="none" />
                {/* top-right */}
                <path d={`M ${x2 - len} ${y1} L ${x2} ${y1} L ${x2} ${y1 + len}`} stroke={teamColor} strokeWidth="1.4" fill="none" />
                {/* bottom-left */}
                <path d={`M ${x1} ${y2 - len} L ${x1} ${y2} L ${x1 + len} ${y2}`} stroke={teamColor} strokeWidth="1.4" fill="none" />
                {/* bottom-right */}
                <path d={`M ${x2 - len} ${y2} L ${x2} ${y2} L ${x2} ${y2 - len}`} stroke={teamColor} strokeWidth="1.4" fill="none" />
                {/* center crosshair */}
                <line x1={b.x + b.w / 2 - 5} y1={b.y + b.h / 2} x2={b.x + b.w / 2 + 5} y2={b.y + b.h / 2} stroke={teamColor} strokeWidth="1" strokeOpacity="0.55" />
                <line x1={b.x + b.w / 2} y1={b.y + b.h / 2 - 5} x2={b.x + b.w / 2} y2={b.y + b.h / 2 + 5} stroke={teamColor} strokeWidth="1" strokeOpacity="0.55" />
              </svg>
            );
          })()}
          {/* Zone hotspot dots — clickable. Active hotspots emit an outward
              ripple ring to give the panel a "live targeting" cadence rather
              than the static dots-on-an-X-ray feel. */}
          {hotspots.map((h) => {
            const isActive = active === h.zone;
            const hotColor = h.intensity >= 0.5 ? "#f87171" : h.intensity >= 0.25 ? "#fbbf24" : teamColor;
            return (
              <div
                key={h.key}
                className="absolute"
                style={{ left: h.cx - 12, top: h.cy - 12, width: 24, height: 24, zIndex: 5 }}
              >
                {isActive && (
                  <>
                    <span aria-hidden className="absolute inset-0 rounded-full pointer-events-none"
                      style={{ border: `1px solid ${hotColor}`, animation: "holoRipple 1.4s ease-out infinite" }} />
                    <span aria-hidden className="absolute inset-0 rounded-full pointer-events-none"
                      style={{ border: `1px solid ${hotColor}`, animation: "holoRipple 1.4s ease-out infinite 0.7s" }} />
                  </>
                )}
                <button
                  type="button"
                  aria-label={`Focus ${h.zone}`}
                  onMouseEnter={() => setActive(h.zone)}
                  onMouseLeave={() => setActive(null)}
                  onClick={() => setActive(a => a === h.zone ? null : h.zone)}
                  className="absolute rounded-full cursor-pointer"
                  style={{
                    left: 6,
                    top:  6,
                    width: 12,
                    height: 12,
                    background: hotColor,
                    border: `1px solid ${hotColor}`,
                    boxShadow: `0 0 ${isActive ? 16 : 8}px ${hotColor}`,
                    transform: isActive ? "scale(1.6)" : undefined,
                    transition: "transform 200ms ease, box-shadow 200ms ease",
                    animation: `holoNode ${1.8 + (h.key.length % 3) * 0.4}s ease-in-out infinite`,
                  }}
                />
              </div>
            );
          })}
        </div>

        {/* LEFT telemetry callouts — 3 only, hoverable, highlights target zone.
            Container is a div (not a button) so the StatInfoTip button inside
            can render properly — nesting <button> in <button> is invalid HTML
            and was why the native `title` was the only thing firing before. */}
        {telemetryLeft.length > 0 && (
          <div className="absolute left-2 top-12 bottom-10 flex flex-col justify-around items-end gap-2 z-10">
            {telemetryLeft.map((c) => {
              const isActive = active === c.target;
              return (
                <div
                  key={c.id}
                  onMouseEnter={() => setActive(c.target)}
                  onMouseLeave={() => setActive(null)}
                  className="flex flex-col items-end gap-0.5 px-2 py-1 rounded backdrop-blur transition-all duration-200"
                  style={{
                    background: isActive ? `${teamColor}1a` : "rgba(0,0,0,0.40)",
                    border: `1px solid ${isActive ? teamColor : `${teamColor}28`}`,
                    boxShadow: isActive ? `0 0 14px ${teamColor}55` : "none",
                  }}
                >
                  <span className="hud-mono text-[8px] uppercase tracking-[0.18em] flex items-center gap-1"
                    style={{ color: isActive ? teamColor : "var(--text-secondary)" }}>
                    ▸ {c.label}
                    {c.tip && <StatInfoTip label={c.label} tip={c.tip} />}
                  </span>
                  <span className="hud-mono text-[11px] tabular-nums font-semibold" style={{ color: teamColor }}>
                    {c.val}
                  </span>
                </div>
              );
            })}
          </div>
        )}

        {/* RIGHT telemetry callouts */}
        {telemetryRight.length > 0 && (
          <div className="absolute right-2 top-12 bottom-10 flex flex-col justify-around items-start gap-2 z-10">
            {telemetryRight.map((c) => {
              const isActive = active === c.target;
              return (
                <div
                  key={c.id}
                  onMouseEnter={() => setActive(c.target)}
                  onMouseLeave={() => setActive(null)}
                  className="flex flex-col items-start gap-0.5 px-2 py-1 rounded backdrop-blur transition-all duration-200"
                  style={{
                    background: isActive ? `${teamColor}1a` : "rgba(0,0,0,0.40)",
                    border: `1px solid ${isActive ? teamColor : `${teamColor}28`}`,
                    boxShadow: isActive ? `0 0 14px ${teamColor}55` : "none",
                  }}
                >
                  <span className="hud-mono text-[8px] uppercase tracking-[0.18em] flex items-center gap-1"
                    style={{ color: isActive ? teamColor : "var(--text-secondary)" }}>
                    {c.tip && <StatInfoTip label={c.label} tip={c.tip} />}
                    {c.label} ◂
                  </span>
                  <span className="hud-mono text-[11px] tabular-nums font-semibold" style={{ color: teamColor }}>
                    {c.val}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Neural DNA strip — player fingerprint between silhouette and ticker */}
      {dnaRungs && dnaRungs.length > 0 && (
        <PlayerDnaStrip rungs={dnaRungs} teamColor={teamColor} signature={dnaSignature} />
      )}

      {/* Bottom ticker — separate row, outside the canvas, no overlap */}
      {tickerLine.length > 0 && (
        <div className="mt-2 h-6 overflow-hidden rounded relative"
          style={{ background: "rgba(0,0,0,0.45)", border: `1px solid ${teamColor}33` }}>
          <div className="absolute inset-y-0 flex items-center gap-6 whitespace-nowrap hud-mono text-[9px] uppercase tracking-[0.18em] px-3"
            style={{ color: `${teamColor}cc`, animation: "holoTicker 28s linear infinite" }}>
            {[...tickerLine, ...tickerLine].map((t, i) => (
              <span key={i} className="inline-flex items-center gap-1">
                <span className="inline-block w-1 h-1 rounded-full" style={{ background: teamColor }} />
                {t}
              </span>
            ))}
          </div>
        </div>
      )}

      <style jsx>{`
        @keyframes ironRotateSlow { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes ironRotateRev  { from { transform: rotate(360deg); } to { transform: rotate(0deg); } }
        @keyframes ironRotateFast { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes ironPulse {
          0%   { r: 50; stroke-opacity: 0.55; }
          100% { r: 165; stroke-opacity: 0; }
        }
        @keyframes holoVScan {
          0%   { transform: translateY(0);   opacity: 0; }
          10%  { opacity: 1; }
          90%  { opacity: 1; }
          100% { transform: translateY(340px); opacity: 0; }
        }
        @keyframes holoHScan {
          0%   { transform: translateX(0);    opacity: 0; }
          10%  { opacity: 1; }
          90%  { opacity: 1; }
          100% { transform: translateX(200px); opacity: 0; }
        }
        @keyframes holoReticle {
          from { transform: scale(1.18); opacity: 0; }
          to   { transform: scale(1);    opacity: 1; }
        }
        @keyframes holoRipple {
          0%   { transform: scale(0.5); opacity: 0.75; }
          100% { transform: scale(2.2); opacity: 0; }
        }
        @keyframes holoNode {
          0%, 100% { opacity: 0.65; }
          50%      { opacity: 1; }
        }
        @keyframes holoTicker {
          from { transform: translateX(0); }
          to   { transform: translateX(-50%); }
        }
        @media (prefers-reduced-motion: reduce) {
          svg g, circle, div, button { animation: none !important; }
        }
      `}</style>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Predicted Play — top-down zone schematic with arrows showing the most
// likely entry + in-zone action sequence given the Behavioral NN weights.
// Adds a "predicted decision" layer on top of the raw matrix so the page
// actually shows what the model expects this player to *do*, not just the
// raw input features.
// ---------------------------------------------------------------------------

type PredPlayLabel =
  | "Slot Shot"
  | "Drive Net"
  | "Perimeter"
  | "Battle Corner"
  | "Hold Corner";

interface PredAction {
  id: string;
  label: string;
  pct: number;            // raw model % (0-100)
}

/** Per-action colour + line style so each arrow on the rink reads as a
 *  distinct kind of play. Shots, skating routes and contested zone plays
 *  each get their own visual treatment.
 *  kind: shot = puck flight to the goal · skate = player carrying the puck
 *  · dump = puck flung into zone · battle = contested possession */
const ACTION_THEME: Record<string, { color: string; dash: string; kind: "shot" | "skate" | "dump" | "battle" | "pass"; legend: string }> = {
  carry:  { color: "#38bdf8", dash: "0",       kind: "skate",  legend: "Carry-in · skate" },
  dump:   { color: "#fb923c", dash: "5 4",     kind: "dump",   legend: "Dump-in · chip" },
  slot:   { color: "#f87171", dash: "0",       kind: "shot",   legend: "Slot · shot" },
  drive:  { color: "#a78bfa", dash: "0",       kind: "skate",  legend: "Drive · net rush" },
  perim:  { color: "#fbbf24", dash: "0",       kind: "shot",   legend: "Perimeter · shot" },
  battle: { color: "#4ade80", dash: "4 3",     kind: "battle", legend: "Battle · corner" },
  hold:   { color: "#2dd4bf", dash: "1 3",     kind: "battle", legend: "Hold · possession" },
  // Pass — not an output of the behavior NN (model has no pass head yet),
  // synthesised at render time from assist/shot mix so playmakers get a
  // visible route instead of looking like average shooters.
  pass:   { color: "#e879f9", dash: "2 2.5",   kind: "pass",   legend: "Pass · setup" },
};

function PredictedPlay({
  carry, dump, slot, perim, drive, battleC, holdC, themeColor, leagueAvg,
  position, shoots, gpg, apg, shotsPer60, rapmOff,
}: {
  carry?: number | null;
  dump?: number | null;
  slot?: number | null;
  perim?: number | null;
  drive?: number | null;
  battleC?: number | null;
  holdC?: number | null;
  themeColor: string;
  leagueAvg?: {
    carry_in?: number; dump?: number;
    shoot_slot?: number; shoot_perimeter?: number;
    drive_net?: number; battle_corner?: number; hold_corner?: number;
  } | null;
  position?: string | null;
  shoots?: string | null;
  // Pass-tendency inputs. The behavior NN has no pass head yet, so we
  // synthesise the playmaking signal from box-score mix: high assists vs.
  // shots = setup man; high RAPM offense with low shots/60 = QB-style D.
  gpg?: number | null;
  apg?: number | null;
  shotsPer60?: number | null;
  rapmOff?: number | null;
}) {
  // A defenseman lives at the offensive blue line — the play does not start
  // above the zone like a forward zone entry. We rewire the schematic so the
  // origin is the L or R point, the in-zone routes read as point shot / walk
  // the line / pinch instead of slot-rush, and carry/dump entries are
  // suppressed (they don't apply to D in the OZ). Drive-to-net only renders
  // when this specific D's nn_drive_net_pct meaningfully clears the league
  // mean for their position — otherwise we drop it so Slavin doesn't get
  // drawn like Makar.
  const pos = (position ?? "").toUpperCase();
  const isD = pos === "D" || pos === "LD" || pos === "RD";
  // Right-hand shots usually play the left point (for the one-timer on the
  // off-wing); LD/RD overrides shoots when explicit. Default to right point
  // when we have no signal at all.
  const dSide: "L" | "R" = (() => {
    if (pos === "LD") return "L";
    if (pos === "RD") return "R";
    const sh = (shoots ?? "").toUpperCase();
    if (sh === "L") return "R";
    if (sh === "R") return "L";
    return "R";
  })();
  // Resolve league averages per action id (matches the ACTION_THEME keys).
  // When the backend hasn't shipped them yet, we leave deltas null and the
  // UI silently falls back to raw-probability ranking.
  const avgFor = (id: string): number | null => {
    if (!leagueAvg) return null;
    const map: Record<string, number | undefined> = {
      carry:  leagueAvg.carry_in,
      dump:   leagueAvg.dump,
      slot:   leagueAvg.shoot_slot,
      perim:  leagueAvg.shoot_perimeter,
      drive:  leagueAvg.drive_net,
      battle: leagueAvg.battle_corner,
      hold:   leagueAvg.hold_corner,
    };
    const v = map[id];
    return typeof v === "number" ? v : null;
  };

  // Pass tendency — synthesised, not modelled (no pass head on behavior NN
  // yet). High assist-share + low shots-per-60 + positive RAPM offense
  // means the player creates more than they finish. Clamped 0-1; > 0.40
  // is enough to surface the pass route. We expose the score upward as a
  // pseudo-action probability so it sorts naturally next to model actions.
  const assistShare = (apg != null && gpg != null && (apg + gpg) > 0.05)
    ? apg / (apg + gpg)
    : null;
  let passTendency = 0;
  if (assistShare != null) passTendency += Math.max(0, (assistShare - 0.50) * 1.8); // 0.50→0, 1.0→0.9
  if (rapmOff != null && rapmOff > 0.4 && (shotsPer60 == null || shotsPer60 < 7)) passTendency += 0.25;
  passTendency = Math.max(0, Math.min(1, passTendency));
  // Convert to a pseudo-percentage on the same scale as the NN action %s
  // so the legend / SEQUENCES strip can rank it alongside slot/perim/etc.
  const passPct = passTendency * 22; // up to ~22% — comparable to model max action share

  // D plays already start in the OZ at the point — carry / dump don't apply.
  // Suppressing them avoids drawing a forward-style entry curve for a D.
  const entryActions: PredAction[] = isD ? [] : [
    { id: "carry", label: "Carry-in", pct: carry ?? 0 },
    { id: "dump",  label: "Dump-in",  pct: dump  ?? 0 },
  ].filter(a => a.pct > 0).sort((a, b) => b.pct - a.pct);
  // In-zone labels are position-aware. For D the same model dimension reads
  // differently on ice: shoot_perimeter → point shot, drive_net → pinch net
  // (only highlighted when this D is meaningfully above forward average),
  // battle/hold corner → cycle pinch / hold the line, shoot_slot → walk-in
  // shot (rare for D — needs the model to actually project it). We also
  // de-prioritize drive_net for D unless their probability tops the league
  // mean by a real margin, so non-pinching D don't get rendered like Makar.
  const inZoneActions: PredAction[] = isD ? [
    { id: "perim",  label: "Point Shot",   pct: perim   ?? 0 },
    { id: "battle", label: "Pinch Corner", pct: battleC ?? 0 },
    { id: "hold",   label: "Hold Line",    pct: holdC   ?? 0 },
    { id: "drive",  label: "Pinch Net",    pct: drive   ?? 0 },
    { id: "slot",   label: "Walk-In Shot", pct: slot    ?? 0 },
    { id: "pass",   label: "Cross-Ice Pass", pct: passPct },
  ].filter(a => a.pct > 0).sort((a, b) => b.pct - a.pct) : [
    { id: "slot",   label: "Slot Shot",     pct: slot    ?? 0 },
    { id: "drive",  label: "Drive Net",     pct: drive   ?? 0 },
    { id: "perim",  label: "Perimeter",     pct: perim   ?? 0 },
    { id: "battle", label: "Battle Corner", pct: battleC ?? 0 },
    { id: "hold",   label: "Hold Corner",   pct: holdC   ?? 0 },
    { id: "pass",   label: "Slot Feed",     pct: passPct },
  ].filter(a => a.pct > 0).sort((a, b) => b.pct - a.pct);

  if (entryActions.length === 0 && inZoneActions.length === 0) return null;

  // Entry: if carry and dump are essentially tied (≤ 5pp gap and both > 0),
  // the model has no real entry signal — usually because carry_entry_pct
  // was missing for this player. Collapse to a "Mixed Entry" indicator
  // rather than picking a fake winner that looks the same for everyone.
  const entryGap = entryActions.length === 2
    ? Math.abs(entryActions[0].pct - entryActions[1].pct)
    : Infinity;
  const entryMixed = entryActions.length === 2 && entryGap < 5;
  const topEntry  = entryMixed ? null : (entryActions[0] ?? null);

  // In-zone primary: when league averages are available, pick the action
  // whose probability deviates *most positively* from league — that's what
  // makes this player distinct. Otherwise fall back to highest raw %.
  // For D this is honest because the caller passes the D-only league mean,
  // so the delta reflects "this D vs other D" — Slavin's drive_net only
  // wins when he's actually pinching more than peer defensemen.
  const primaryByDelta = (() => {
    if (!leagueAvg || inZoneActions.length === 0) return null;
    let best: PredAction | null = null;
    let bestDelta = -Infinity;
    for (const a of inZoneActions) {
      const avg = avgFor(a.id);
      if (avg == null) continue;
      const d = a.pct - avg;
      if (d > bestDelta) { bestDelta = d; best = a; }
    }
    return bestDelta > 1 ? best : null;
  })();
  // Fallback ordering matters too: for D, default the top action to point
  // shot / pinch corner / hold line before drive_net or walk-in slot. We
  // walk the sorted list and prefer the first "D-typical" route present.
  const dPreferred = isD
    ? (inZoneActions.find(a => a.id === "perim")
       ?? inZoneActions.find(a => a.id === "battle")
       ?? inZoneActions.find(a => a.id === "hold")
       ?? inZoneActions[0]
       ?? null)
    : (inZoneActions[0] ?? null);
  const topInZone = primaryByDelta ?? dPreferred;
  const topInZoneDelta = topInZone ? (() => {
    const avg = avgFor(topInZone.id);
    return avg == null ? null : topInZone.pct - avg;
  })() : null;
  // Top 3 in-zone actions for the ranked list under the schematic.
  // Sorted by raw probability so the bars still read most→least likely,
  // but each row carries its own delta-vs-avg chip.
  const top3 = inZoneActions.slice(0, 3);

  // ── SVG geometry ──────────────────────────────────────────────────────
  // Top-down view of the offensive zone. Player enters at the top (above
  // the blue line); net sits at the bottom.
  const VB = { w: 280, h: 180 };
  const NET = { cx: 140, y: 160, w: 36 };
  const BLUE = 20;       // blue-line y
  const SLOT = { cx: 140, cy: 118 };
  const PERIM_L = { cx: 75, cy: 95 };
  const PERIM_R = { cx: 205, cy: 95 };
  const CORNER_L = { cx: 30, cy: 165 };
  const CORNER_R = { cx: 250, cy: 165 };

  // D blue-line position (the "point") — origin for every D route. The
  // off-side point is the dominant deployment (R-handed D on the left point
  // for the one-timer is the modern norm), so we pick the side that matches
  // LD/RD, or invert shoots-hand when only `shoots` is known. The opposite
  // point gets a ghost marker so "walk the line" reads correctly.
  const D_POINT = dSide === "L"
    ? { cx: 90,  cy: 28 }    // left point
    : { cx: 190, cy: 28 };   // right point
  const D_OPP   = dSide === "L"
    ? { cx: 190, cy: 28 }
    : { cx: 90,  cy: 28 };

  // ── Arrow path builder ────────────────────────────────────────────────
  // Returns an SVG path string + endpoint for arrowhead orientation.
  function entryPath(id: string): string {
    if (id === "carry") {
      // Smooth S-curve from above blue line in toward the slot
      return `M 140 -5 Q 110 30 130 60 T 140 90`;
    }
    // dump — straight down to corner then chip to slot
    return `M 140 -5 L 145 50 L 245 150`;
  }
  function inZonePath(id: string): string {
    if (isD) {
      // All D routes originate at the blue-line point on the player's side.
      // Point shot — straight down through traffic, slight off-stick lean so
      // the line reads as a wrist/slap shot not a skating route.
      if (id === "perim") {
        const lean = dSide === "L" ? 5 : -5;
        return `M ${D_POINT.cx} ${D_POINT.cy + 4} L ${D_POINT.cx + lean} ${D_POINT.cy + 38} L ${NET.cx} ${NET.y - 4}`;
      }
      // Pinch net — diagonal skate from point straight down to crease. Only
      // rendered when this D actually does it (drive_net deviates positively).
      if (id === "drive") {
        return `M ${D_POINT.cx} ${D_POINT.cy + 4} Q ${(D_POINT.cx + NET.cx) / 2} ${(D_POINT.cy + NET.y) / 2 - 6} ${NET.cx} ${NET.y - 4}`;
      }
      // Walk-in shot — skate diagonally toward the high slot, shot from there.
      if (id === "slot") {
        return `M ${D_POINT.cx} ${D_POINT.cy + 4} Q ${(D_POINT.cx + SLOT.cx) / 2} ${(D_POINT.cy + SLOT.cy) / 2 - 4} ${SLOT.cx} ${SLOT.cy} L ${NET.cx} ${NET.y - 4}`;
      }
      // Pinch corner — D drops down to the strong-side boards / corner.
      if (id === "battle") {
        const corner = dSide === "L" ? CORNER_L : CORNER_R;
        return `M ${D_POINT.cx} ${D_POINT.cy + 4} Q ${(D_POINT.cx + corner.cx) / 2} ${(D_POINT.cy + corner.cy) / 2 + 6} ${corner.cx + (dSide === "L" ? 6 : -6)} ${corner.cy - 4}`;
      }
      // Cross-ice pass — point-to-point pass that sets up a one-timer on the
      // opposite side. Curve lifts slightly above the blue line so it reads
      // as a tape-to-tape feed, not a shot.
      if (id === "pass") {
        return `M ${D_POINT.cx} ${D_POINT.cy} Q ${(D_POINT.cx + D_OPP.cx) / 2} ${D_POINT.cy - 18} ${D_OPP.cx - (dSide === "L" ? 3 : -3)} ${D_OPP.cy + 1}`;
      }
      // Hold line — D walks the blue line across to the opposite point.
      /* hold */
      return `M ${D_POINT.cx} ${D_POINT.cy} Q ${(D_POINT.cx + D_OPP.cx) / 2} ${D_POINT.cy - 10} ${D_OPP.cx} ${D_OPP.cy}`;
    }
    if (id === "slot")   return `M 140 90  L ${SLOT.cx} ${SLOT.cy} L ${NET.cx} ${NET.y - 4}`;
    if (id === "drive")  return `M 140 80  Q 145 110 ${NET.cx} ${NET.y - 4}`;
    if (id === "perim")  return `M 140 80  L ${PERIM_R.cx} ${PERIM_R.cy} L ${NET.cx + 4} ${NET.y - 4}`;
    if (id === "battle") return `M 140 80  Q 60 120 ${CORNER_L.cx + 5} ${CORNER_L.cy - 5} L ${SLOT.cx - 10} ${SLOT.cy + 5}`;
    // Slot feed — forward enters with the puck then slides a cross-seam
    // pass into the slot for a teammate one-timer. Curves opposite the
    // direct shot route so the two arrows don't overlap.
    if (id === "pass")   return `M 140 80  Q 110 105 ${SLOT.cx - 18} ${SLOT.cy + 4} Q ${SLOT.cx} ${SLOT.cy + 14} ${SLOT.cx + 14} ${SLOT.cy - 2}`;
    /* hold */            return `M 140 80  Q 220 120 ${CORNER_R.cx - 5} ${CORNER_R.cy - 5} Q 245 175 230 165`;
  }

  // Endpoint of a pass route — drives the teammate-receiver marker so the
  // arrow visibly terminates on a person, not a generic spot. Returned in
  // SVG user units of the rink coord space.
  function passEndpoint(): { cx: number; cy: number } {
    if (isD) return { cx: D_OPP.cx - (dSide === "L" ? 3 : -3), cy: D_OPP.cy + 1 };
    return { cx: SLOT.cx + 14, cy: SLOT.cy - 2 };
  }

  return (
    <div className="mt-2 mb-1 rounded border px-2 py-2"
      style={{ borderColor: `${themeColor}33`, background: "rgba(0,0,0,0.30)" }}>
      <div className="flex items-center justify-between mb-1.5">
        <span className="hud-mono text-[9px] uppercase tracking-[0.18em]" style={{ color: themeColor }}>
          ▸ PREDICTED PLAY
        </span>
        <span className="hud-mono text-[8px] uppercase tracking-[0.14em] text-[var(--text-muted)]">
          bnn projection
        </span>
      </div>

      <svg viewBox={`0 0 ${VB.w} ${VB.h}`} className="w-full" style={{ height: 130 }}>
        <defs>
          {/* One arrowhead per action colour so the tip of each arrow matches
              its line. Marker fill cascades from the line stroke via context-stroke. */}
          {Object.entries(ACTION_THEME).map(([id, t]) => (
            <marker key={id} id={`ppArrow-${id}`} viewBox="0 0 10 10" refX="9" refY="5"
              markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill={t.color} />
            </marker>
          ))}
        </defs>

        {/* Boards (rounded rect) */}
        <rect x="2" y="-2" width={VB.w - 4} height={VB.h + 2}
          fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1" rx="14" ry="14" />

        {/* Blue line — bold, blue */}
        <line x1="4" y1={BLUE} x2={VB.w - 4} y2={BLUE} stroke="#3b8fd0" strokeWidth="3" opacity="0.55" />

        {/* Faceoff circles */}
        <circle cx={PERIM_L.cx} cy={PERIM_L.cy} r="22"
          fill="none" stroke="#e63a3a" strokeOpacity="0.20" strokeWidth="1.2" />
        <circle cx={PERIM_R.cx} cy={PERIM_R.cy} r="22"
          fill="none" stroke="#e63a3a" strokeOpacity="0.20" strokeWidth="1.2" />
        <circle cx={PERIM_L.cx} cy={PERIM_L.cy} r="2" fill="#e63a3a" opacity="0.35" />
        <circle cx={PERIM_R.cx} cy={PERIM_R.cy} r="2" fill="#e63a3a" opacity="0.35" />

        {/* Goal crease — half-disc opening upward */}
        <path d={`M ${NET.cx - 22} ${NET.y} A 22 22 0 0 1 ${NET.cx + 22} ${NET.y}`}
          fill="rgba(59,143,208,0.10)" stroke="#3b8fd0" strokeOpacity="0.45" strokeWidth="0.8" />

        {/* Goal line (red) */}
        <line x1="40" y1={NET.y} x2={VB.w - 40} y2={NET.y} stroke="#e63a3a" strokeOpacity="0.55" strokeWidth="0.8" />

        {/* Net rectangle (red posts) */}
        <rect x={NET.cx - NET.w / 2} y={NET.y} width={NET.w} height="14"
          fill="rgba(255,255,255,0.04)" stroke="#ff3030" strokeOpacity="0.85" strokeWidth="1.4" />
        {/* Net mesh diagonals — quick visual texture */}
        {[0.25, 0.5, 0.75].map((t, i) => (
          <line key={i}
            x1={NET.cx - NET.w / 2 + t * NET.w} y1={NET.y}
            x2={NET.cx - NET.w / 2 + t * NET.w} y2={NET.y + 14}
            stroke="#ff8080" strokeOpacity="0.30" strokeWidth="0.6" />
        ))}

        {/* Ghosted secondary in-zone paths (rank 2, 3) — each drawn in its own
            action colour + dash style so the ranked list below ties visually. */}
        {inZoneActions.slice(1, 3).map((a, i) => {
          const t = ACTION_THEME[a.id];
          if (!t) return null;
          return (
            <path key={a.id}
              d={inZonePath(a.id)}
              fill="none"
              stroke={t.color}
              strokeOpacity={0.28 - i * 0.08}
              strokeWidth="1.4"
              strokeDasharray={t.dash !== "0" ? t.dash : "3 5"}
              markerEnd={`url(#ppArrow-${a.id})`}
              style={{ filter: `drop-shadow(0 0 3px ${t.color}55)` }} />
          );
        })}

        {/* Pass overlay — drawn even when pass isn't in the top 3 in-zone
            actions, because the box-score signal (assist-share + RAPM off)
            is independent of the NN action mix. The receiver marker at the
            endpoint makes it read as a feed, not just another shot. */}
        {passTendency >= 0.40 && (() => {
          const t = ACTION_THEME["pass"];
          const isPrimary = topInZone?.id === "pass";
          const end = passEndpoint();
          return (
            <g style={{ animation: `ppDraw 1100ms ease-out ${isPrimary ? 900 : 1300}ms forwards` }}>
              <path
                d={inZonePath("pass")}
                fill="none"
                stroke={t.color}
                strokeOpacity={isPrimary ? 0.95 : 0.70}
                strokeWidth={isPrimary ? "2.2" : "1.6"}
                strokeDasharray={t.dash}
                strokeLinecap="round"
                markerEnd={`url(#ppArrow-pass)`}
                style={{ filter: `drop-shadow(0 0 5px ${t.color}aa)` }} />
              {/* Receiver — small ringed marker at the endpoint so the pass
                  visibly terminates on a teammate, with a soft glow ring. */}
              <circle cx={end.cx} cy={end.cy} r="3.2"
                fill="none" stroke={t.color} strokeOpacity="0.55"
                strokeWidth="0.7" strokeDasharray="1.2 1.2" />
              <circle cx={end.cx} cy={end.cy} r="1.6"
                fill={t.color} opacity="0.85"
                style={{ filter: `drop-shadow(0 0 4px ${t.color})` }} />
            </g>
          );
        })()}

        {/* Primary entry path — action-coloured, kind-styled */}
        {topEntry && (() => {
          const t = ACTION_THEME[topEntry.id];
          if (!t) return null;
          return (
            <path
              d={entryPath(topEntry.id)}
              fill="none"
              stroke={t.color}
              strokeWidth="2.4"
              strokeOpacity="0.92"
              strokeLinecap="round"
              strokeDasharray={t.dash !== "0" ? t.dash : undefined}
              markerEnd={`url(#ppArrow-${topEntry.id})`}
              style={{
                filter: `drop-shadow(0 0 5px ${t.color})`,
                strokeDasharray: t.dash !== "0" ? t.dash : "320",
                strokeDashoffset: t.dash !== "0" ? 0 : 320,
                animation: t.dash !== "0" ? undefined : "ppDraw 1100ms ease-out 80ms forwards",
              }} />
          );
        })()}
        {/* Primary in-zone path — same treatment */}
        {topInZone && (() => {
          const t = ACTION_THEME[topInZone.id];
          if (!t) return null;
          return (
            <path
              d={inZonePath(topInZone.id)}
              fill="none"
              stroke={t.color}
              strokeWidth="2.4"
              strokeOpacity="0.95"
              strokeLinecap="round"
              strokeDasharray={t.dash !== "0" ? t.dash : undefined}
              markerEnd={`url(#ppArrow-${topInZone.id})`}
              style={{
                filter: `drop-shadow(0 0 6px ${t.color})`,
                strokeDasharray: t.dash !== "0" ? t.dash : "240",
                strokeDashoffset: t.dash !== "0" ? 0 : 240,
                animation: t.dash !== "0" ? undefined : "ppDraw 1100ms ease-out 900ms forwards",
              }} />
          );
        })()}

        {/* Origin marker — for forwards we drop an entry-arrow over the blue
            line; for D we plant a player puck at the strong-side point and
            ghost a teammate at the opposite point so "walk the line" reads
            as a real route. The D marker pulses + glows to read as live
            instrument lock-on rather than a static dot. */}
        {isD ? (
          <g style={{ animation: "ppDOrigin 600ms cubic-bezier(0.22,1,0.36,1) 80ms both" }}>
            {/* Sensor ring (slow pulse) at the strong-side point */}
            <circle cx={D_POINT.cx} cy={D_POINT.cy} r="7.5"
              fill="none" stroke={themeColor} strokeOpacity="0.50" strokeWidth="0.5"
              style={{
                transformOrigin: `${D_POINT.cx}px ${D_POINT.cy}px`,
                animation: "ppDLock 2.4s ease-in-out infinite",
              }} />
            {/* Crosshair brackets — sub-pixel lock-on indicator */}
            <g stroke={themeColor} strokeWidth="0.5" strokeLinecap="round" opacity="0.85">
              <line x1={D_POINT.cx - 6}  y1={D_POINT.cy - 6} x2={D_POINT.cx - 3.2} y2={D_POINT.cy - 6} />
              <line x1={D_POINT.cx - 6}  y1={D_POINT.cy - 6} x2={D_POINT.cx - 6}   y2={D_POINT.cy - 3.2} />
              <line x1={D_POINT.cx + 6}  y1={D_POINT.cy - 6} x2={D_POINT.cx + 3.2} y2={D_POINT.cy - 6} />
              <line x1={D_POINT.cx + 6}  y1={D_POINT.cy - 6} x2={D_POINT.cx + 6}   y2={D_POINT.cy - 3.2} />
              <line x1={D_POINT.cx - 6}  y1={D_POINT.cy + 6} x2={D_POINT.cx - 3.2} y2={D_POINT.cy + 6} />
              <line x1={D_POINT.cx - 6}  y1={D_POINT.cy + 6} x2={D_POINT.cx - 6}   y2={D_POINT.cy + 3.2} />
              <line x1={D_POINT.cx + 6}  y1={D_POINT.cy + 6} x2={D_POINT.cx + 3.2} y2={D_POINT.cy + 6} />
              <line x1={D_POINT.cx + 6}  y1={D_POINT.cy + 6} x2={D_POINT.cx + 6}   y2={D_POINT.cy + 3.2} />
            </g>
            {/* Player puck — solid + glow + soft inner highlight */}
            <circle cx={D_POINT.cx} cy={D_POINT.cy} r="4.6" fill={themeColor} opacity="0.96"
              style={{ filter: `drop-shadow(0 0 7px ${themeColor}) drop-shadow(0 0 2px ${themeColor})` }} />
            <circle cx={D_POINT.cx - 1.2} cy={D_POINT.cy - 1.2} r="1.4" fill="rgba(255,255,255,0.35)" />
            <text x={D_POINT.cx} y={D_POINT.cy + 1.6} textAnchor="middle"
              fontSize="4.8" fontWeight="800" fill="rgba(0,0,0,0.82)"
              style={{ fontFamily: "var(--font-mono)", letterSpacing: "0.04em" }}>D</text>
            {/* Ghost teammate at the opposite point — dashed ring + faint dot */}
            <circle cx={D_OPP.cx} cy={D_OPP.cy} r="4.0"
              fill="none" stroke={themeColor} strokeOpacity="0.42" strokeWidth="0.8" strokeDasharray="1.6 1.6" />
            <circle cx={D_OPP.cx} cy={D_OPP.cy} r="1.4" fill={themeColor} opacity="0.32" />
            {/* Side label so the eye binds the strong-side puck to L / R */}
            <text x={D_POINT.cx} y={D_POINT.cy - 9.5} textAnchor="middle"
              fontSize="2.6" fontWeight="700" fill={themeColor} opacity="0.85"
              style={{ fontFamily: "var(--font-mono)", letterSpacing: "0.30em", textShadow: `0 0 4px ${themeColor}` }}>
              {dSide === "L" ? "LP" : "RP"}
            </text>
          </g>
        ) : (
          <polygon points={`${140 - 4},-3 ${140 + 4},-3 140,7`} fill={themeColor} opacity="0.85" />
        )}

        <style>{`
          @keyframes ppDraw { to { stroke-dashoffset: 0; } }
          @keyframes ppDOrigin {
            from { transform: scale(0.4); opacity: 0; }
            to   { transform: scale(1);   opacity: 1; }
          }
          @keyframes ppDLock {
            0%, 100% { stroke-opacity: 0.55; transform: scale(1);    }
            50%      { stroke-opacity: 0.15; transform: scale(1.18); }
          }
          @media (prefers-reduced-motion: reduce) {
            path, g, circle { animation: none !important; stroke-dashoffset: 0 !important; }
          }
        `}</style>
      </svg>

      {(topEntry || topInZone || entryMixed) && (
        <div className="mt-1.5 px-1 flex items-center gap-2 flex-wrap">
          <span className="hud-mono text-[9px] uppercase tracking-[0.18em] text-[var(--text-secondary)]">SEQ ·</span>
          {entryMixed ? (
            <span className="hud-mono text-[10px] uppercase tracking-[0.18em] font-semibold text-white/55">
              Mixed Entry
            </span>
          ) : topEntry && (() => {
            const t = ACTION_THEME[topEntry.id];
            return (
              <span className="hud-mono text-[10px] uppercase tracking-[0.18em] font-semibold"
                style={{ color: t?.color ?? themeColor, textShadow: `0 0 6px ${(t?.color ?? themeColor)}55` }}>
                {topEntry.label}
              </span>
            );
          })()}
          <span className="hud-mono text-[10px] text-[var(--text-muted)]">→</span>
          {topInZone && (() => {
            const t = ACTION_THEME[topInZone.id];
            const c = t?.color ?? themeColor;
            const dPos = topInZoneDelta != null && topInZoneDelta > 0.5;
            const dNeg = topInZoneDelta != null && topInZoneDelta < -0.5;
            return (
              <>
                <span className="hud-mono text-[10px] uppercase tracking-[0.18em] font-semibold"
                  style={{ color: c, textShadow: `0 0 6px ${c}55` }}>
                  {topInZone.label}
                </span>
                {topInZoneDelta != null && (
                  <span className="hud-mono text-[9px] uppercase tracking-[0.14em] tabular-nums px-1.5 py-0.5 rounded border"
                    style={{
                      color: dPos ? "#5ee08a" : dNeg ? "#f87b7b" : "rgba(255,255,255,0.45)",
                      borderColor: dPos ? "#5ee08a55" : dNeg ? "#f87b7b55" : "rgba(255,255,255,0.18)",
                      background: dPos ? "rgba(94,224,138,0.06)" : dNeg ? "rgba(248,123,123,0.06)" : "transparent",
                    }}>
                    {topInZoneDelta > 0 ? "+" : ""}{topInZoneDelta.toFixed(1)}% vs avg
                  </span>
                )}
              </>
            );
          })()}
        </div>
      )}

      {/* Inline legend — one chip per action present in this player's NN
          weights, with the same colour + dash style as the rink arrow so
          Bob can tell which line means what. D legends rename routes to
          their D-specific concept (Point shot / Pinch corner / etc.) so
          the chips match the arrows on the rink. */}
      <div className="mt-2 flex flex-wrap gap-1.5 px-1">
        {[
          ...(entryActions.length ? entryActions.map(a => a.id) : []),
          ...(inZoneActions.slice(0, 3).map(a => a.id)),
        ].filter((id, i, arr) => arr.indexOf(id) === i).map((id) => {
          const t = ACTION_THEME[id];
          if (!t) return null;
          const dLegend: Record<string, string> = {
            slot:   "Walk-in · shot",
            drive:  "Pinch · net",
            perim:  "Point · shot",
            battle: "Pinch · corner",
            hold:   "Walk the line",
            pass:   "D-to-D · feed",
          };
          const fLegend: Record<string, string> = {
            pass:   "Slot · feed",
          };
          const legendText = (isD && dLegend[id]) ? dLegend[id] : (fLegend[id] ?? t.legend);
          return (
            <div key={id} className="flex items-center gap-1.5 px-1.5 py-0.5 rounded border"
              style={{ borderColor: `${t.color}33`, background: `${t.color}0e` }}>
              {/* Mini line swatch — matches stroke style of the SVG path */}
              <svg width="16" height="6" viewBox="0 0 16 6" aria-hidden>
                <line x1="1" y1="3" x2="15" y2="3"
                  stroke={t.color} strokeWidth="1.6" strokeLinecap="round"
                  strokeDasharray={t.dash !== "0" ? t.dash : undefined} />
              </svg>
              <span className="hud-mono text-[8px] uppercase tracking-[0.14em]" style={{ color: `${t.color}cc` }}>
                {legendText}
              </span>
            </div>
          );
        })}
      </div>

      {/* Top-3 ranked in-zone decisions — each bar uses its action colour so
          the row directly matches its arrow on the rink above. Each row also
          carries a small +/- vs-avg chip so identical-looking probabilities
          (e.g. perimeter 19.8%) read very differently for a slot-finisher
          who's *under* the league mean vs a perimeter shooter who's over it. */}
      {top3.length > 0 && (
        <div className="mt-2 pt-2 border-t border-white/[0.05] space-y-1">
          {top3.map((a, i) => {
            const t = ACTION_THEME[a.id];
            const c = t?.color ?? themeColor;
            const avg = avgFor(a.id);
            const delta = avg != null ? a.pct - avg : null;
            const dPos = delta != null && delta > 0.5;
            const dNeg = delta != null && delta < -0.5;
            return (
            <div key={a.id} className="flex items-center gap-2">
              <span className="hud-mono text-[8px] uppercase tracking-[0.16em] w-3 text-right shrink-0"
                style={{ color: i === 0 ? c : "rgba(255,255,255,0.30)" }}>
                {i + 1}
              </span>
              <span className="hud-mono text-[9px] uppercase tracking-[0.14em] text-white/65 w-20 shrink-0 truncate">
                {a.label}
              </span>
              <div className="flex-1 h-1.5 rounded-sm overflow-hidden relative"
                style={{ background: "rgba(255,255,255,0.04)", border: `1px solid ${c}33` }}>
                <div className="h-full" style={{
                  width: `${Math.min(100, a.pct)}%`,
                  background: `linear-gradient(90deg, ${c}aa 0%, ${c} 100%)`,
                  boxShadow: `0 0 6px ${c}55`,
                  animation: `ppBar 900ms cubic-bezier(0.22,1,0.36,1) ${i * 90}ms backwards`,
                  transformOrigin: "left center",
                }} />
                {avg != null && (
                  <div className="absolute top-[-1px] bottom-[-1px] pointer-events-none"
                    style={{ left: `${Math.min(100, avg)}%`, width: 1, background: "rgba(255,255,255,0.35)" }}
                    aria-hidden />
                )}
              </div>
              <span className="hud-mono text-[10px] tabular-nums w-10 text-right font-semibold"
                style={{ color: c }}>
                {a.pct.toFixed(1)}%
              </span>
              {delta != null && (
                <span className="hud-mono text-[8px] uppercase tracking-[0.10em] tabular-nums w-12 text-right shrink-0"
                  style={{ color: dPos ? "#5ee08a" : dNeg ? "#f87b7b" : "rgba(255,255,255,0.30)" }}>
                  {delta > 0 ? "+" : ""}{delta.toFixed(1)}
                </span>
              )}
            </div>
            );
          })}
          <style jsx>{`
            @keyframes ppBar {
              from { transform: scaleX(0); opacity: 0; }
              to   { transform: scaleX(1); opacity: 1; }
            }
            @media (prefers-reduced-motion: reduce) {
              div { animation: none !important; }
            }
          `}</style>
        </div>
      )}

      {/* TOP SEQUENCES — derived 2-step play combinations, scored from the
          action-probability mix + synthesised pass tendency. Each row is a
          named "what the player tends to actually run" pattern: e.g. carry
          + slot = "Carry-in · Slot Shot", high battle + perim = "Cycle ·
          One-Timer". Top 3 surface in their own coloured chips so Bob can
          read intent at a glance without parsing 7 raw percentages. */}
      {(() => {
        // Action lookups (all 0 when missing — combinator handles that).
        const v = {
          carry:  carry   ?? 0,
          dump:   dump    ?? 0,
          slot:   slot    ?? 0,
          drive:  drive   ?? 0,
          perim:  perim   ?? 0,
          battle: battleC ?? 0,
          hold:   holdC   ?? 0,
          pass:   passPct,
        };
        // Geometric-mean weight on the two contributing actions so the
        // sequence only ranks high when BOTH legs are real (not just one
        // huge action with a 0 partner). Then × 100 for display.
        const w = (a: number, b: number) => Math.sqrt(Math.max(0, a) * Math.max(0, b));
        type Combo = { label: string; weight: number; ids: string[]; tip?: string };
        const combos: Combo[] = isD ? [
          { label: "Walk Line · Point Shot",      weight: w(v.hold,   v.perim),  ids: ["hold", "perim"],   tip: "QB at the blue line — walks then snaps it on net." },
          { label: "Pinch Corner · Cycle Out",    weight: w(v.battle, v.hold),   ids: ["battle", "hold"],  tip: "Steps down the boards, cycles back to the point." },
          { label: "Point Shot · Net-Front Tip",  weight: w(v.perim,  v.drive),  ids: ["perim", "drive"],  tip: "Throws the puck on net knowing a forward is screening." },
          { label: "Cross-Ice · One-Timer Setup", weight: w(v.pass,   v.perim),  ids: ["pass", "perim"],   tip: "Slides it to the opposite point for a one-timer." },
          { label: "Pinch Net · Crash",           weight: w(v.drive,  v.battle), ids: ["drive", "battle"], tip: "Activates from the point straight to the crease." },
          { label: "Walk-In Shot · Slot",         weight: w(v.slot,   v.perim),  ids: ["slot", "perim"],   tip: "Sneaks in from the point for a high-slot wrist shot." },
          { label: "D-to-D · Reset",              weight: w(v.hold,   v.pass),   ids: ["hold", "pass"],    tip: "Tape-to-tape D-pair pass to reset the OZ structure." },
        ] : [
          { label: "Carry-in · Slot Shot",        weight: w(v.carry,  v.slot),   ids: ["carry", "slot"],   tip: "Skates the puck in, beats the gap, gets a slot shot." },
          { label: "Carry-in · Slot Feed",        weight: w(v.carry,  v.pass),   ids: ["carry", "pass"],   tip: "Carries then dishes — playmaker's favourite OZ entry." },
          { label: "Carry-in · Perimeter Snap",   weight: w(v.carry,  v.perim),  ids: ["carry", "perim"],  tip: "Off-wing entry, snaps it on net from the half-wall." },
          { label: "Dump · Cycle to Slot",        weight: w(v.dump,   v.battle), ids: ["dump", "battle"],  tip: "Chips it in, wins the wall, works the cycle back to the slot." },
          { label: "Battle · Drive Net",          weight: w(v.battle, v.drive),  ids: ["battle", "drive"], tip: "Possession-first, then power move to the front of the net." },
          { label: "Battle · Slot Feed",          weight: w(v.battle, v.pass),   ids: ["battle", "pass"],  tip: "Wall battle → low-to-high pass to a teammate in the slot." },
          { label: "Hold Corner · One-Timer",     weight: w(v.hold,   v.perim),  ids: ["hold", "perim"],   tip: "Cycles in the corner waiting for the high-slot kick-out shot." },
          { label: "Drive · Rebound Hunt",        weight: w(v.drive,  v.slot),   ids: ["drive", "slot"],   tip: "Net-front presence — drives the lane looking for second chances." },
        ];
        const top = combos
          .filter(c => c.weight > 0.5)
          .sort((a, b) => b.weight - a.weight)
          .slice(0, 3);
        if (top.length === 0) return null;
        // Normalize weights to bar widths relative to the strongest combo.
        const maxW = top[0].weight;
        return (
          <div className="mt-2 pt-2 border-t border-white/[0.05]">
            <div className="flex items-center justify-between mb-1.5 px-1">
              <span className="hud-mono text-[9px] uppercase tracking-[0.20em]" style={{ color: themeColor }}>
                ▸ TOP SEQUENCES
              </span>
              <span className="hud-mono text-[8px] uppercase tracking-[0.16em] text-[var(--text-muted)]">
                2-step combinations
              </span>
            </div>
            <div className="space-y-1.5 px-1">
              {top.map((c, i) => {
                const colors = c.ids.map(id => ACTION_THEME[id]?.color ?? themeColor);
                const grad = `linear-gradient(90deg, ${colors[0]}cc 0%, ${colors[1] ?? colors[0]} 100%)`;
                const widthPct = Math.min(100, (c.weight / maxW) * 100);
                return (
                  <div key={c.label} className="flex items-center gap-2" title={c.tip ?? ""}>
                    <span className="hud-mono text-[8px] uppercase tracking-[0.16em] w-3 text-right shrink-0"
                      style={{ color: i === 0 ? colors[0] : "rgba(255,255,255,0.30)" }}>
                      {i + 1}
                    </span>
                    {/* Combo chip — two coloured dots + label so the eye
                        binds the sequence to the arrows on the rink. */}
                    <div className="flex items-center gap-1 shrink-0">
                      <span className="inline-block w-1.5 h-1.5 rounded-full"
                        style={{ background: colors[0], boxShadow: `0 0 4px ${colors[0]}` }} />
                      <span className="text-[8px] text-white/30">·</span>
                      <span className="inline-block w-1.5 h-1.5 rounded-full"
                        style={{ background: colors[1] ?? colors[0], boxShadow: `0 0 4px ${colors[1] ?? colors[0]}` }} />
                    </div>
                    <span className="hud-mono text-[9px] uppercase tracking-[0.14em] truncate"
                      style={{ color: i === 0 ? "rgba(255,255,255,0.92)" : "rgba(255,255,255,0.65)" }}>
                      {c.label}
                    </span>
                    <div className="flex-1 h-1 rounded-sm overflow-hidden relative ml-1"
                      style={{ background: "rgba(255,255,255,0.04)", border: `1px solid ${colors[0]}22` }}>
                      <div className="h-full"
                        style={{
                          width: `${widthPct}%`,
                          background: grad,
                          boxShadow: `0 0 5px ${colors[0]}66`,
                          animation: `ppCombo 900ms cubic-bezier(0.22,1,0.36,1) ${i * 110}ms backwards`,
                          transformOrigin: "left center",
                        }} />
                    </div>
                  </div>
                );
              })}
              <style jsx>{`
                @keyframes ppCombo {
                  from { transform: scaleX(0); opacity: 0; }
                  to   { transform: scaleX(1); opacity: 1; }
                }
                @media (prefers-reduced-motion: reduce) {
                  div { animation: none !important; }
                }
              `}</style>
            </div>
          </div>
        );
      })()}
    </div>
  );
}

// ---------------------------------------------------------------------------
// KpiBand — the headline tile strip directly under the hero console.
// Skater variant surfaces box-score (GP · G · A · P · TOI · +/-) plus the
// analytical anchor (WAR · FI · CI). Goalie variant flips to GP · W-L ·
// SV% · GSAx · GAA · SO. Each tile is its own HUD pip — compact, glowing
// in team colour, with a tier chip when the value clears a known band.
//
// Sits BETWEEN the hero console and the Ratings panel so the dashboard
// reads top-to-bottom: who the player is → what they did → engine context
// → drill-down telemetry tabs. Without this band the page jumps straight
// from hologram to engine outputs and box-score totals are buried in the
// game log tab.
// ---------------------------------------------------------------------------

interface KpiTile {
  label: string;
  value: string;
  sub?: string;
  tier?: Tier | null;
  tone?: "good" | "warn" | "bad" | "neutral";
  tip?: string;
}

function KpiBand({
  tiles, teamColor,
}: { tiles: KpiTile[]; teamColor: string }) {
  if (tiles.length === 0) return null;
  return (
    <div className="lg:col-span-12 mt-3">
      <div className="hud-panel hud-panel--all-corners jarvis-boot jarvis-shimmer relative overflow-hidden"
        style={{
          ["--hud-corner" as string]: teamColor,
          background: "linear-gradient(180deg, rgba(0,0,0,0.55), rgba(0,0,0,0.32))",
          boxShadow: `0 0 18px ${teamColor}14, inset 0 0 24px rgba(0,0,0,0.4)`,
        }}>
        <span className="hud-panel__corner-tr" />
        <span className="hud-panel__corner-bl" />
        <div className="hud-scan" aria-hidden />
        {/* Header strip — pulse-dot + label, mirrors the Ratings panel
            chrome so the two bands feel like one console layer. */}
        <div className="flex items-center justify-between px-3 py-1.5 border-b" style={{ borderColor: `${teamColor}22` }}>
          <div className="flex items-center gap-2">
            <span className="hud-pulse-dot" style={{ background: teamColor, boxShadow: `0 0 6px ${teamColor}` }} />
            <span className="hud-mono text-[10px] uppercase tracking-[0.24em] font-semibold"
              style={{ color: teamColor, textShadow: `0 0 5px ${teamColor}55` }}>
              ◢ KEY METRICS
            </span>
            <span className="hud-mono text-[8px] uppercase tracking-[0.18em] text-[var(--text-muted)]">
              · season totals
            </span>
          </div>
          <span className="hud-mono text-[8px] uppercase tracking-[0.18em]" style={{ color: `${teamColor}99` }}>
            live · 60Hz
          </span>
        </div>
        {/* Tile row — grid auto-fits 6 across desktop, 3 across tablet,
            2 across phone. Each tile lives in its own column with a
            vertical divider hairline so it reads as instrument cluster. */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 divide-x"
          style={{ borderColor: `${teamColor}14` }}>
          {tiles.map((t, i) => {
            const tierColor = t.tier ? TIER_COLOR[t.tier] : null;
            const toneColor =
              t.tone === "good" ? "#5ee08a" :
              t.tone === "warn" ? "#fbbf24" :
              t.tone === "bad"  ? "#f87171" :
              tierColor ?? teamColor;
            return (
              <div key={i} className="px-3 py-2.5 relative flex flex-col items-center justify-center text-center gap-1"
                title={t.tip ?? ""}
                style={{ borderColor: `${teamColor}14` }}>
                {/* Top mini-tick — matches the Ratings cluster pip */}
                <span aria-hidden className="absolute top-1 left-1/2 -translate-x-1/2 w-6 h-px"
                  style={{ background: `${toneColor}55`, boxShadow: `0 0 4px ${toneColor}77` }} />
                <span className="hud-mono text-[9px] uppercase tracking-[0.22em] text-[var(--text-secondary)]">
                  {t.label}
                </span>
                <span className="hud-mono text-[18px] sm:text-[20px] font-semibold tabular-nums leading-none"
                  style={{
                    color: toneColor,
                    textShadow: `0 0 8px ${toneColor}55, 0 0 2px ${toneColor}33`,
                    animation: `kpiBoot 600ms cubic-bezier(0.22,1,0.36,1) ${i * 70}ms both`,
                  }}>
                  {t.value}
                </span>
                {t.sub && (
                  <span className="hud-mono text-[8px] uppercase tracking-[0.16em] text-[var(--text-muted)] tabular-nums">
                    {t.sub}
                  </span>
                )}
                {t.tier && tierColor && (
                  <span className="hud-mono text-[8px] uppercase tracking-[0.18em] rounded border px-1.5 py-0.5"
                    style={{
                      color: tierColor,
                      borderColor: `${tierColor}55`,
                      backgroundColor: `${tierColor}14`,
                      textShadow: `0 0 4px ${tierColor}55`,
                    }}>
                    {TIER_ABBREV[t.tier]}
                  </span>
                )}
              </div>
            );
          })}
        </div>
        <style jsx>{`
          @keyframes kpiBoot {
            from { transform: translateY(4px); opacity: 0; }
            to   { transform: translateY(0);   opacity: 1; }
          }
          @media (prefers-reduced-motion: reduce) {
            span { animation: none !important; }
          }
        `}</style>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SectionHeader — the "◆ OFFENSE", "◆ DEFENSE" reticle bars that group
// related cards inside a telemetry tab. Same idiom HudPanel uses for its
// title strip, but free-standing so it sits between cards in a flat grid.
// ---------------------------------------------------------------------------

function SectionHeader({
  title, subtitle, teamColor, glyph = "◆",
}: { title: string; subtitle?: string; teamColor: string; glyph?: string }) {
  return (
    <div className="sm:col-span-2 mt-2 px-1">
      <div className="flex items-center gap-2.5 pb-1.5 border-b"
        style={{ borderColor: `${teamColor}26` }}>
        <span className="hud-mono text-[11px]" style={{ color: teamColor, textShadow: `0 0 6px ${teamColor}77` }}>
          {glyph}
        </span>
        <span className="hud-mono text-[10px] uppercase tracking-[0.24em] font-semibold"
          style={{ color: teamColor, textShadow: `0 0 5px ${teamColor}44` }}>
          {title}
        </span>
        {subtitle && (
          <span className="hud-mono text-[8px] uppercase tracking-[0.18em] text-[var(--text-muted)]">
            · {subtitle}
          </span>
        )}
        {/* Right-side stretch hairline so the header reads as a section
            divider, not just a label. */}
        <span className="flex-1 h-px ml-1" style={{ background: `linear-gradient(90deg, ${teamColor}55 0%, transparent 100%)` }} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Profile Page
// ---------------------------------------------------------------------------

export default function PlayerProfilePage() {
  const params = useParams();
  const router = useRouter();
  const { theme, cortexPinned, setPreviewTheme } = useTheme();
  const { ctx: seasonCtx, hydrated: ctxHydrated } = useSeasonContext();
  // The route param is a URL-encoded player name (e.g. "Nathan%20MacKinnon").
  // Normalize accents so old bookmarks / direct URLs with ý, é, etc. still resolve.
  const playerName = normalizePlayerName(decodeURIComponent(params.id as string));

  const [data, setData] = useState<ProfileData | null>(null);
  const [loading, setLoading] = useState(true);
  const [imgErr, setImgErr] = useState(false);
  const [shots, setShots] = useState<ShotPoint[]>([]);
  const [goalieShots, setGoalieShots] = useState<ShotPoint[]>([]);
  const [goalieNetData, setGoalieNetData] = useState<GoalieNetData | null>(null);
  // Current-season stats fetched from nhl-team route
  const [nhlStats, setNhlStats] = useState<{
    gp: number; goals: number; assists: number; points: number; plus_minus: number;
    jersey?: number | null;
    // goalie
    wins?: number; losses?: number; ot_losses?: number; gaa?: number; sv_pct?: number; shutouts?: number;
  } | null>(null);
  // True when the team route silently fell back from the requested context
  // (e.g. user asked for playoffs but the team was eliminated). We surface a
  // hint in the strip instead of pretending the regular-season numbers are
  // playoff numbers.
  const [nhlContextFallback, setNhlContextFallback] = useState(false);
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

  // Phase 3 fatigue data (independent of the main /phase2/player fetch so
  // the rest of the profile still renders even if Phase 3 is empty/null).
  interface Phase3Card {
    fatigue_index:   number | null;
    fi_game_date:    string | null;
    fi_context_fallback: boolean;
    fi_components:   Record<string, number> | null;
    fi_multiplier:   number | null;
    anomaly_z:       number | null;
    is_anomaly:      boolean;
    consecutive_below_n: number | null;
    seasonal_factor: number | null;
    seasonal_month:  number | null;
    edge_load:       number | null;
    edge_speed:      number | null;
    edge_distance:   number | null;
    edge_carry:      number | null;
    edge_burst:      number | null;
    // Goalie fatigue (3.24) — populated for goalies only
    goalie_fi:           number | null;
    goalie_fi_date:      string | null;
    goalie_sv_delta:     number | null;
    goalie_is_b2b:       number | null;
    goalie_rest_days:    number | null;
    goalie_gp_last_7:    number | null;
    goalie_shots_last_7: number | null;
    // Confidence (Phase 17.24)
    confidence_index:     number | null;
    confidence_date:      string | null;
    confidence_player:    number | null;
    confidence_team:      number | null;
    confidence_components: Record<string, number> | null;
    conf_shoot_bias:      number | null;
    conf_risk_bias:       number | null;
    conf_turnover_bias:   number | null;
  }
  const [phase3, setPhase3] = useState<Phase3Card | null>(null);

  // Tab state for the bottom telemetry strip — MUST be declared before any
  // early returns (loading / not-found) so it runs in the same hook order
  // on every render. Rules of Hooks.
  // games + special-teams folded into advanced; not in the union anymore.
  type TelemetryTab = "shot-map" | "zones" | "fatigue" | "advanced" | "neural";
  const [telemetryTab, setTelemetryTab] = useState<TelemetryTab>("neural");

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
    if (!ctxHydrated) return;
    setLoading(true);

    fetch(`/api/phase2/player?name=${encodeURIComponent(playerName)}&context=${seasonCtx}`)
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

    // Phase 3 enrichment — fire-and-forget; missing sub-models render as null.
    // Wait for the context toggle to hydrate so we don't fire two requests
    // (one with the default, then a refetch on the persisted value).
    if (!ctxHydrated) return;
    setPhase3(null);
    fetch(`/api/phase3/player?name=${encodeURIComponent(playerName)}&context=${seasonCtx}`)
      .then(r => r.json())
      .then((p) => {
        if (p?.not_found) return;
        setPhase3({
          fatigue_index:   p?.fi?.fatigue_index ?? null,
          fi_game_date:    p?.fi?.game_date ?? null,
          fi_context_fallback: Boolean(p?.fi?.context_fallback),
          fi_components:   p?.fi?.component_breakdown ?? null,
          fi_multiplier:   p?.fi_multiplier?.multiplier ?? null,
          anomaly_z:       p?.anomaly?.z_score ?? null,
          is_anomaly:      Boolean(p?.anomaly?.is_anomaly),
          consecutive_below_n: p?.anomaly?.consecutive_below_n ?? null,
          seasonal_factor: p?.seasonal?.seasonal_motivation_factor ?? null,
          seasonal_month:  p?.seasonal?.month_of_season ?? null,
          edge_load:       p?.edge_degradation?.predicted_load_factor ?? null,
          edge_speed:      p?.edge_degradation?.speed_vs_baseline ?? null,
          edge_distance:   p?.edge_degradation?.distance_vs_baseline ?? null,
          edge_carry:      p?.edge_degradation?.carry_vs_baseline ?? null,
          edge_burst:      p?.edge_degradation?.burst_vs_baseline ?? null,
          goalie_fi:           p?.goalie_fatigue?.goalie_fi ?? null,
          goalie_fi_date:      p?.goalie_fatigue?.game_date ?? null,
          goalie_sv_delta:     p?.goalie_fatigue?.fatigue_sv_delta ?? null,
          goalie_is_b2b:       p?.goalie_fatigue?.is_b2b ?? null,
          goalie_rest_days:    p?.goalie_fatigue?.rest_days ?? null,
          goalie_gp_last_7:    p?.goalie_fatigue?.gp_last_7 ?? null,
          goalie_shots_last_7: p?.goalie_fatigue?.shots_faced_last_7 ?? null,
          confidence_index:     p?.confidence?.confidence_index ?? null,
          confidence_date:      p?.confidence?.game_date ?? null,
          confidence_player:    p?.confidence?.player_score ?? null,
          confidence_team:      p?.confidence?.team_score ?? null,
          confidence_components: p?.confidence?.component_breakdown ?? null,
          conf_shoot_bias:      p?.confidence_multiplier?.shoot_bias ?? null,
          conf_risk_bias:       p?.confidence_multiplier?.risk_bias ?? null,
          conf_turnover_bias:   p?.confidence_multiplier?.turnover_bias ?? null,
        });
      })
      .catch(() => {});
  }, [playerName, seasonCtx, ctxHydrated]);

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

  // Once we know the team, fetch current-season stats and injury status.
  // Re-fetches when the season/playoffs toggle flips so the stat strip
  // updates in lockstep with the cards below.
  useEffect(() => {
    if (!data?.team) return;
    if (!ctxHydrated) return;
    const team = data.team;
    const fullName = normalizePlayerName(data.player_name ?? "").toLowerCase();

    setNhlStats(null);
    setNhlContextFallback(false);

    // Current-season or playoff stats + jersey from nhl-team route
    fetch(`/api/nhl-team/${team}?context=${seasonCtx}`)
      .then(r => r.json())
      .then(d => {
        setNhlContextFallback(Boolean(d.used_fallback));
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
  }, [data?.team, data?.player_name, seasonCtx, ctxHydrated]);

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

  // Fetch shot data for shot map visualization — context-aware so the heat
  // map reflects regular-season or playoff shots when the pill flips.
  useEffect(() => {
    if (!data?.player_id) return;
    if (!ctxHydrated) return;
    setShots([]);
    fetch(`/api/player-shots/${data.player_id}?context=${seasonCtx}`)
      .then(r => r.json())
      .then(d => { if (d.shots?.length) setShots(d.shots); })
      .catch(() => {});
  }, [data?.player_id, seasonCtx, ctxHydrated]);

  useEffect(() => {
    if (!data?.player_id || !data?.is_goalie) return;
    if (!ctxHydrated) return;
    setGoalieShots([]);
    fetch(`/api/goalie-shots/${data.player_id}?context=${seasonCtx}`)
      .then(r => r.json())
      .then(d => { if (d.shots?.length) setGoalieShots(d.shots); })
      .catch(() => {});
  }, [data?.player_id, data?.is_goalie, seasonCtx, ctxHydrated]);

  useEffect(() => {
    if (!data?.player_id || !data?.is_goalie) return;
    fetch(`/api/goalie-neural-net/${data.player_id}`)
      .then(r => r.json())
      .then(d => { if (d.status === "ok") setGoalieNetData(d); })
      .catch(() => {});
  }, [data?.player_id, data?.is_goalie]);

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

  // ── HUD command-deck derivations ─────────────────────────────────────────
  const fi = phase3?.fatigue_index ?? null;
  const ci = phase3?.confidence_index ?? null;
  const hhs = data.hot_hand_score ?? null;
  const warVal = data.war ?? null;
  const warRankPct = data.war_rank && data.war_total_qualified
    ? 1 - (data.war_rank / data.war_total_qualified)
    : null;

  // Neural graph nodes — render NN activations as radial node weights
  const neuralNodes: NeuralNode[] = isGoalie ? [] : (() => {
    const nodes: NeuralNode[] = [];
    if (data.nn_carry_in_pct != null)        nodes.push({ id: "carry",   label: "Carry-in",    weight: Math.min(1, data.nn_carry_in_pct / 60) });
    if (data.nn_shoot_slot_pct != null)      nodes.push({ id: "slot",    label: "Slot shot",   weight: Math.min(1, data.nn_shoot_slot_pct / 50) });
    if (data.nn_drive_net_pct != null)       nodes.push({ id: "drive",   label: "Net drive",   weight: Math.min(1, data.nn_drive_net_pct / 40) });
    if (data.nn_dump_pct != null)            nodes.push({ id: "dump",    label: "Dump",        weight: Math.min(1, data.nn_dump_pct / 60) });
    if (data.nn_battle_corner_pct != null)   nodes.push({ id: "battle",  label: "Battle",      weight: Math.min(1, data.nn_battle_corner_pct / 40) });
    if (data.nn_shoot_perimeter_pct != null) nodes.push({ id: "perim",   label: "Perimeter",   weight: Math.min(1, data.nn_shoot_perimeter_pct / 50) });
    return nodes;
  })();

  // Hologram body zone tinting from FI components + EDGE degradation
  const bodyIntensity: Partial<Record<BodyZone, number>> = isGoalie ? {} : (() => {
    const out: Partial<Record<BodyZone, number>> = {};
    const comp = phase3?.fi_components ?? {};
    const total = (fi ?? 0);
    const legHit = Math.min(1, (total + Math.abs(phase3?.edge_speed ?? 0) + Math.abs(phase3?.edge_distance ?? 0)) / 1.2);
    const armHit = Math.min(1, (total + Math.abs(phase3?.edge_burst ?? 0)) / 1.0);
    const torsoHit = Math.min(1, (total + ((comp.contact_load ?? 0) + (comp.overtime_load ?? 0))) / 1.0);
    const headHit = Math.min(1, (total + (comp.travel_load ?? 0) + (comp.circadian_load ?? 0)) / 1.2);
    out.legL = legHit;
    out.legR = legHit;
    out.armL = armHit;
    out.armR = armHit;
    out.torso = torsoHit;
    out.head = headHit;
    out.shoulder = armHit * 0.8;
    return out;
  })();

  // EWMA waveform — synthesize a smooth trail toward the current value so
  // the heart-rate-style sparkline always has something meaningful to draw.
  const ewmaWave = (() => {
    const cur = data.ewma_xgf60 ?? null;
    if (cur == null) return [] as number[];
    const len = 20;
    const drift = (Math.random() - 0.5) * 0.4;
    const pts: number[] = [];
    for (let i = 0; i < len; i++) {
      const t = i / (len - 1);
      const base = 4.09 + (cur - 4.09) * t;
      const wobble = Math.sin(t * Math.PI * 3 + drift) * 0.18;
      pts.push(base + wobble);
    }
    return pts;
  })();

  // telemetryTab state is declared above (before the early-return guards).
  // Tab bar — consolidated. "Neural" tab renamed to "Behavior" so it
  // doesn't collide with the Neural Cortex hero panel. "Recent" tab dropped
  // entirely (just a game log); "Special" tab folded into "Advanced"
  // because it only carried 2 stats.
  const telemetryTabs: HudTab[] = isGoalie
    ? [
        { id: "neural",   label: "Behavior" },
        { id: "shot-map", label: "Shots Against" },
        { id: "zones",    label: "Zones" },
        { id: "fatigue",  label: "Fatigue" },
        { id: "advanced", label: "Advanced" },
      ]
    : [
        { id: "neural",   label: "Behavior" },
        { id: "shot-map", label: "Shot Map" },
        { id: "zones",    label: "Zones" },
        { id: "fatigue",  label: "Fatigue" },
        { id: "advanced", label: "Advanced" },
      ];

  // Map ProfileData shots → Shot3D shot type for the 3D rink
  const shots3D: Shot3DPoint[] = (shots ?? []).map((s) => ({
    x: s.x,
    y: s.y,
    goal: !!s.goal,
  }));
  const goalieShots3D: Shot3DPoint[] = (goalieShots ?? []).map((s) => ({
    x: s.x,
    y: s.y,
    goal: !!s.goal,
  }));

  // ── Status flags surfaced as HUD badges
  const statusFlags: { tone: "good" | "warn" | "bad" | "accent" | "neutral"; label: string; pulse?: boolean }[] = (() => {
    const out: { tone: "good" | "warn" | "bad" | "accent" | "neutral"; label: string; pulse?: boolean }[] = [];
    if (injuryBadge) {
      const isHard = injuryBadge === "Out" || injuryBadge.startsWith("IR");
      out.push({ tone: isHard ? "bad" : "warn", label: injuryBadge, pulse: true });
    } else if (data.ewma_form_flag === "rising" || (hhs ?? 0) > 0.7) {
      out.push({ tone: "good", label: "Hot Hand", pulse: true });
    } else if (data.ewma_form_flag === "falling") {
      out.push({ tone: "warn", label: "Cooling", pulse: false });
    } else {
      out.push({ tone: "good", label: "Active", pulse: true });
    }
    if (data.war_rank && data.war_rank <= 30) {
      out.push({ tone: "accent", label: `Rank #${data.war_rank}` });
    }
    if (fi != null) {
      const fatLabel = fi >= 0.45 ? "Gassed" : fi >= 0.25 ? "Tired" : fi >= 0.12 ? "Worked" : "Fresh";
      const tone: "good" | "warn" | "bad" = fi >= 0.45 ? "bad" : fi >= 0.25 ? "warn" : "good";
      out.push({ tone, label: `${fatLabel} · FI ${fi.toFixed(2)}` });
    }
    return out;
  })();

  return (
    <main className="relative min-h-screen p-4 sm:p-6 max-w-3xl lg:max-w-[1400px] mx-auto w-full overflow-x-hidden">
      <HudGrid />

      {/* ── Search bar — team-colored ── */}
      <div className="relative flex justify-center mb-5 gap-2">
        {/* Team page button */}
        {displayTeam && (
          <Link
            href={`/teams/${displayTeam}`}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl shrink-0 transition-all hover:opacity-80"
            style={{ background: `${teamColor}15`, border: `1px solid ${teamColor}30` }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={`https://assets.nhle.com/logos/nhl/svg/${displayTeam}_dark.svg`}
              alt={displayTeam} width={20} height={20} className="object-contain" draggable={false} />
            <span className="text-[11px] font-black uppercase tracking-[0.12em]"
              style={{ color: teamColor }}>Team</span>
          </Link>
        )}
        <div className="relative flex-1 max-w-md">
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


      {/* ── Hero section — HUD-styled identity panel ── */}
      <div className="relative jarvis-shimmer rounded-2xl overflow-hidden mb-4 shadow-[0_16px_60px_rgba(0,0,0,0.75)]"
        style={{ border: `1.5px solid ${teamColor}55`, background: `linear-gradient(175deg, ${teamDarkBg} 0%, #060708 60%)` }}>

        {/* HUD corner brackets — four corners */}
        {([
          [6, 6, "border-l border-t"],
          [6, 6, "border-r border-t right-1.5 left-auto"],
          [6, 6, "border-l border-b bottom-1.5 top-auto"],
          [6, 6, "border-r border-b right-1.5 bottom-1.5 left-auto top-auto"],
        ] as [number, number, string][]).map(([w, h], i) => {
          const pos =
            i === 0 ? { top: 6, left: 6, borderTop: `1px solid ${teamColor}80`, borderLeft: `1px solid ${teamColor}80` } :
            i === 1 ? { top: 6, right: 6, borderTop: `1px solid ${teamColor}80`, borderRight: `1px solid ${teamColor}80` } :
            i === 2 ? { bottom: 6, left: 6, borderBottom: `1px solid ${teamColor}80`, borderLeft: `1px solid ${teamColor}80` } :
                      { bottom: 6, right: 6, borderBottom: `1px solid ${teamColor}80`, borderRight: `1px solid ${teamColor}80` };
          return (
            <span key={i} className="absolute pointer-events-none z-10"
              style={{ width: w + "px", height: h + "px", ...pos } as React.CSSProperties} />
          );
        })}

        {/* Target Profile strip — terminal-style header bar */}
        <div className="relative px-4 py-1.5 flex items-center gap-2 border-b"
          style={{ borderColor: `${teamColor}22`, background: `linear-gradient(90deg, ${teamColor}1a 0%, transparent 60%)` }}>
          <span className="hud-mono text-[9px] uppercase tracking-[0.20em]" style={{ color: teamColor }}>◢</span>
          <span className="hud-mono text-[9px] uppercase tracking-[0.20em]" style={{ color: teamColor }}>
            TARGET PROFILE
          </span>
          <span className="hud-mono text-[8px] uppercase tracking-[0.16em] text-[var(--text-secondary)]">
            · IDX-{data.player_id ?? "—"} · {displayTeam || "—"}
          </span>
          <span className="ml-auto flex items-center gap-1.5">
            <span className="hud-pulse-dot" style={{ background: teamColor }} />
            <span className="hud-mono text-[8px] uppercase tracking-[0.18em] text-[var(--text-secondary)]">
              {injuryBadge ? injuryBadge.toUpperCase() : "ACTIVE"}
            </span>
          </span>
        </div>

        {/* Hero image strip */}
        {data.hero_image && (
          <div className="h-24 overflow-hidden relative">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={data.hero_image} alt="" className="w-full h-full object-cover object-top opacity-25" />
            <div className="absolute inset-0 bg-gradient-to-b from-transparent to-[#0d0f13]" />
            {/* Slow scan-line over the hero image */}
            <div
              className="absolute inset-0 pointer-events-none"
              style={{
                background: `linear-gradient(180deg, transparent 0%, ${teamColor}1a 50%, transparent 100%)`,
                mixBlendMode: "screen",
                animation: "heroScan 6s linear infinite",
              }}
            />
            <style jsx>{`
              @keyframes heroScan {
                0%   { transform: translateY(-100%); }
                100% { transform: translateY(100%); }
              }
            `}</style>
          </div>
        )}

        {/* ── Compact horizontal hero — headshot LEFT, identity CENTER,
            attribute radar RIGHT. Three-column flex so the radar sits in
            line with the player's identity card, not dangling somewhere
            below the bio rows. */}
        <div className={`flex items-start gap-4 sm:gap-5 px-4 sm:px-5 pb-3 ${data.hero_image ? "-mt-12 relative z-10" : "pt-4"}`}>

          {/* Circular headshot — smaller, side-positioned */}
          <div className="relative shrink-0">
            <div
              className="absolute inset-0 rounded-full pointer-events-none"
              style={{
                background: `radial-gradient(circle, ${teamColor}55 0%, ${teamColor}22 45%, transparent 72%)`,
                transform: "scale(1.4)",
                filter: "blur(12px)",
              }}
            />
            <div
              className="jarvis-photo-frame relative h-20 w-20 sm:h-24 sm:w-24 rounded-full overflow-hidden"
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
                  <span className="text-2xl font-bold text-white/30">
                    {(data.player_name ?? "?").split(" ").map((w: string) => w[0]).slice(0, 2).join("")}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Identity stack — name, meta, badges */}
          <div className="flex-1 min-w-0 flex flex-col items-start gap-1">
            <h1 className="text-xl sm:text-2xl font-black text-white leading-tight tracking-tight truncate w-full">
              {data.player_name}
            </h1>

            {/* Team logo + #jersey + position — single line */}
            <div className="flex items-center gap-2 flex-wrap">
              {displayTeam && (
                <button onClick={() => router.push(`/teams/${displayTeam}`)} className="shrink-0 hover:opacity-80 transition-opacity">
                  <TeamLogo team={displayTeam} size={28} />
                </button>
              )}
              {(bio?.jersey_number ?? nhlStats?.jersey ?? data.jersey_number) != null && (
                <span className="hud-mono text-[11px] text-white/55 tabular-nums">
                  #{bio?.jersey_number ?? nhlStats?.jersey ?? data.jersey_number}
                </span>
              )}
              {data.position && (
                <span className="hud-mono text-[10px] uppercase tracking-[0.18em] text-white/55">
                  {data.position === "L" ? "LW" : data.position === "R" ? "RW" : data.position}
                </span>
              )}
              {playerType && (
                <span
                  className="hud-mono inline-flex items-center gap-1 text-[9px] uppercase tracking-[0.16em] rounded px-1.5 py-0.5 ml-1"
                  style={{
                    color: "#fcd34d",
                    background: "rgba(245,158,11,0.10)",
                    border: "1px solid rgba(245,158,11,0.40)",
                  }}
                >
                  ★ {playerType}
                </span>
              )}
            </div>

          {/* Hot hand badge */}
          {!injuryBadge && (data.hot_hand_score ?? 0) > 0.5 && (
            <span className="hud-mono inline-flex items-center gap-1 text-[10px] uppercase tracking-[0.16em] text-[#C9A84C] bg-[#C9A84C]/10 border border-[#C9A84C]/30 rounded px-1.5 py-0.5">
              ⚡ Hot Hand
            </span>
          )}

          {/* Injury / form / rank badges */}
          <div className="flex items-center gap-1.5 flex-wrap">
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
            {phase3?.fatigue_index != null && (() => {
              const v = phase3.fatigue_index;
              const col = v >= 0.45 ? "#f87171" : v >= 0.25 ? "#fb923c" : v >= 0.12 ? "#fbbf24" : "#4ade80";
              const label = v >= 0.45 ? "GASSED" : v >= 0.25 ? "TIRED" : v >= 0.12 ? "WORKED" : "FRESH";
              return (
                <span
                  title={`Composite Fatigue Index (3.17) for ${phase3.fi_game_date ?? "latest game"}. Lower is better — measures schedule, travel, TOI, contact, recovery and seasonal load.`}
                  className="text-[10px] font-semibold rounded-full px-2.5 py-0.5 inline-flex items-center gap-1.5"
                  style={{ color: col, borderColor: `${col}55`, backgroundColor: `${col}1a`, border: "1px solid" }}
                >
                  😮‍💨 {label}
                  <span className="font-mono opacity-80">FI {v.toFixed(2)}</span>
                </span>
              );
            })()}
          </div>
          </div>

          {/* Biometric radar moved back into the Behavior tab — it kept
              clipping / floating awkwardly inline with the identity stack
              and there's no compact size that reads cleanly. The tab has
              the room it needs. */}
        </div>

        {/* ── Season / Playoffs context pill ── */}
        <div className="px-3 sm:px-5 py-2 flex justify-center">
          <SeasonContextPill />
        </div>

        {/* ── 2025-26 stats strip — flips between season + playoffs ── */}
        {nhlStats && (
          <div className="border-t px-3 sm:px-5 py-3 flex items-center justify-center gap-2.5 sm:gap-4 flex-wrap" style={{ borderColor: `${teamColor}20` }}>
            {nhlContextFallback ? (
              <span
                className="hud-mono text-[8px] sm:text-[9px] uppercase tracking-[0.18em] rounded border px-1.5 py-0.5"
                style={{
                  color: "#fbbf24",
                  borderColor: "rgba(251,191,36,0.45)",
                  backgroundColor: "rgba(251,191,36,0.10)",
                  textShadow: "0 0 6px rgba(251,191,36,0.45)",
                }}
                title="No data for the selected context — falling back to regular-season numbers."
              >
                {seasonCtx === "playoffs" ? "2025-26 · NO PO DATA · REG-SZN" : "2025-26"}
              </span>
            ) : (
              <span className="text-[8px] sm:text-[9px] uppercase tracking-widest font-semibold text-white/30">
                {seasonCtx === "playoffs" ? "2025-26 PO" : "2025-26"}
              </span>
            )}
            {isGoalie ? (
              <>
                {[
                  { label: "GP",  value: nhlStats.gp },
                  { label: "W",   value: nhlStats.wins ?? 0 },
                  { label: "L",   value: nhlStats.losses ?? 0 },
                  { label: "OTL", value: nhlStats.ot_losses ?? 0 },
                ].map(s => (
                  <div key={s.label} className="flex flex-col items-center min-w-0">
                    <span className="text-xs sm:text-sm font-bold tabular-nums text-white/85">{s.value}</span>
                    <span className="text-[7px] sm:text-[8px] text-white/30 uppercase tracking-wider">{s.label}</span>
                  </div>
                ))}
                <div className="flex flex-col items-center min-w-0">
                  <span className="text-xs sm:text-base font-black tabular-nums"
                    style={{
                      color: nhlStats.sv_pct ? TIER_COLOR[svpctTier(nhlStats.sv_pct)] : teamColor,
                      textShadow: nhlStats.sv_pct ? `0 0 6px ${TIER_COLOR[svpctTier(nhlStats.sv_pct)]}55` : undefined,
                    }}>
                    {nhlStats.sv_pct ? `.${Math.round(nhlStats.sv_pct * 1000)}` : "—"}
                  </span>
                  <span className="text-[7px] sm:text-[8px] text-white/30 uppercase tracking-wider">SV%</span>
                </div>
                <div className="flex flex-col items-center min-w-0">
                  {(() => {
                    const gaa = nhlStats.gaa ?? null;
                    // Lower GAA is better — derive a tier inline.
                    const gaaTier: Tier | null = gaa == null ? null :
                      gaa <= 2.40 ? "Elite" :
                      gaa <= 2.70 ? "Above Average" :
                      gaa <= 3.00 ? "Average" :
                      gaa <= 3.40 ? "Below Average" : "Low";
                    const col = gaaTier ? TIER_COLOR[gaaTier] : "rgba(255,255,255,0.85)";
                    return (
                      <span className="text-xs sm:text-sm font-bold tabular-nums"
                        style={{ color: col, textShadow: gaaTier ? `0 0 6px ${col}55` : undefined }}>
                        {gaa?.toFixed(2) ?? "—"}
                      </span>
                    );
                  })()}
                  <span className="text-[7px] sm:text-[8px] text-white/30 uppercase tracking-wider">GAA</span>
                </div>
                <div className="flex flex-col items-center min-w-0">
                  <span className="text-xs sm:text-sm font-bold tabular-nums text-white/85">{nhlStats.shutouts ?? 0}</span>
                  <span className="text-[7px] sm:text-[8px] text-white/30 uppercase tracking-wider">SO</span>
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
                {phase3?.fatigue_index != null && (() => {
                  const v = phase3.fatigue_index;
                  const col = TIER_COLOR[fatigueTier(v)];
                  return (
                    <div className="flex flex-col items-center min-w-[36px]"
                         title={`Fatigue Index — composite Phase 3 score for ${phase3.fi_game_date ?? "latest game"}`}>
                      <span className="text-sm font-bold tabular-nums" style={{ color: col }}>
                        {v.toFixed(2)}
                      </span>
                      <span className="text-[8px] text-white/30 uppercase tracking-wider">FI</span>
                    </div>
                  );
                })()}
              </>
            )}
          </div>
        )}

        {/* Career totals — HUD-styled, with derived assists when available */}
        {(data.nhl_games_played != null || data.nhl_career_points != null) && (
          <div
            className="border-t px-5 py-3 flex items-center justify-center gap-4 flex-wrap relative"
            style={{
              borderColor: `${teamColor}22`,
              background: `linear-gradient(90deg, transparent 0%, ${teamColor}07 50%, transparent 100%)`,
            }}
          >
            <span aria-hidden className="absolute left-3 top-1/2 -translate-y-1/2 hud-mono text-[8px] uppercase tracking-[0.18em]" style={{ color: teamColor, opacity: 0.7 }}>◢</span>
            <span aria-hidden className="absolute right-3 top-1/2 -translate-y-1/2 hud-mono text-[8px] uppercase tracking-[0.18em]" style={{ color: teamColor, opacity: 0.7 }}>◣</span>
            <span className="hud-mono text-[9px] uppercase tracking-[0.24em]" style={{ color: teamColor }}>
              CAREER · NHL
            </span>
            {data.nhl_games_played != null && (
              <div className="flex flex-col items-center min-w-[40px]">
                <span className="hud-mono text-sm font-bold text-white/85 tabular-nums">{data.nhl_games_played}</span>
                <span className="hud-mono text-[8px] uppercase tracking-[0.18em]" style={{ color: `${teamColor}AA` }}>GP</span>
              </div>
            )}
            {data.nhl_career_goals != null && (
              <div className="flex flex-col items-center min-w-[40px]">
                <span className="hud-mono text-sm font-bold text-white/85 tabular-nums">{data.nhl_career_goals}</span>
                <span className="hud-mono text-[8px] uppercase tracking-[0.18em]" style={{ color: `${teamColor}AA` }}>G</span>
              </div>
            )}
            {data.nhl_career_goals != null && data.nhl_career_points != null && (
              <div className="flex flex-col items-center min-w-[40px]">
                <span className="hud-mono text-sm font-bold text-white/85 tabular-nums">
                  {Math.max(0, data.nhl_career_points - data.nhl_career_goals)}
                </span>
                <span className="hud-mono text-[8px] uppercase tracking-[0.18em]" style={{ color: `${teamColor}AA` }}>A</span>
              </div>
            )}
            {data.nhl_career_points != null && (
              <div className="flex flex-col items-center min-w-[40px]">
                <span className="hud-mono text-base font-black tabular-nums"
                  style={{ color: teamColor, textShadow: `0 0 6px ${teamColor}66` }}>
                  {data.nhl_career_points}
                </span>
                <span className="hud-mono text-[8px] uppercase tracking-[0.18em]" style={{ color: `${teamColor}AA` }}>PTS</span>
              </div>
            )}
            {data.nhl_games_played != null && data.nhl_career_points != null && data.nhl_games_played > 0 && (
              <div className="flex flex-col items-center min-w-[40px]">
                <span className="hud-mono text-sm font-bold text-white/75 tabular-nums">
                  {(data.nhl_career_points / data.nhl_games_played).toFixed(2)}
                </span>
                <span className="hud-mono text-[8px] uppercase tracking-[0.18em]" style={{ color: `${teamColor}AA` }}>PPG</span>
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
            <div className="border-t px-5 py-3" style={{ borderColor: `${teamColor}12` }}>
              {/* Condensed bio — 2-col on tablet, 3-col on desktop so the
                  data doesn't waste vertical real-estate as a stacked list. */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-5 gap-y-1.5">
                {rows.map(([label, value]) => (
                  <div key={label} className="flex items-baseline gap-2 min-w-0">
                    <span className="text-[11px] text-white/30 w-20 shrink-0">{label}:</span>
                    <span className="text-[12px] font-medium text-white/75 truncate">{value}</span>
                  </div>
                ))}
              </div>
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

      {/* ─────────── HUD COMMAND DECK ─────────── */}
      {/* 3-zone monitoring console: Vitals · Hologram · Neural Cortex.
         Folds the most-glanced metrics into a single surface so the rest of
         the page can hide behind tabs. */}
      <div className="mt-4 grid gap-3 lg:grid-cols-12">

        {/* VITALS column */}
        <div className="lg:col-span-3 flex">
          <HudPanel title="Vitals" themeColor={teamColor} scanline allCorners className="w-full flex flex-col">
            <div className="grid grid-cols-2 gap-3 place-items-center">
              {fi != null && (
                <div className="relative">
                  <RingGauge value={fi} label="Fatigue" sublabel="FI" themeColor={teamColor} invert decimals={2} size={96} />
                  <div className="absolute top-0 right-0">
                    <StatInfoTip label="Fatigue Index (FI)"
                      tip="Composite 0–1 fatigue score for the latest game. Weighs schedule (B2B, 3-in-4), travel + circadian, TOI spikes, contact load, recovery days. Lower is better — green = fresh, red = gassed." />
                  </div>
                </div>
              )}
              {ci != null && (
                <div className="relative">
                  <RingGauge value={Math.min(1, (ci + 0.1) / 0.2)} centerText={`${ci >= 0 ? "+" : ""}${ci.toFixed(2)}`} label="Confidence" sublabel="CI" themeColor={teamColor} size={96} />
                  <div className="absolute top-0 right-0">
                    <StatInfoTip label="Confidence Index (CI)"
                      tip="Signed [-1, +1] decision-bias score. Positive = aggressive (shoots, pinches, takes risks). Negative = passive. Blends player signals (form, hot hand, role usage) with team signals (streak, coach tendencies). Distinct from fatigue — fatigue degrades execution, confidence biases decisions." />
                  </div>
                </div>
              )}
              {hhs != null && (
                <div className="relative">
                  <RingGauge value={Math.min(1, Math.max(0, (hhs + 2) / 4))} centerText={`${hhs >= 0 ? "+" : ""}${hhs.toFixed(1)}`} label="Hot Hand" sublabel="HHS" themeColor={teamColor} size={96} />
                  <div className="absolute top-0 right-0">
                    <StatInfoTip label="Hot Hand Score (HHS)"
                      tip="Standardized streak detector over last 5 games — measures goals + xG above expected output. Above +0.7 = meaningfully running hot. Above +1.5 = serious heater. Negative = ice-cold. Feeds straight into the Rust simulator's shot resolution." />
                  </div>
                </div>
              )}
              {warVal != null && (
                <div className="relative">
                  <RingGauge
                    value={warRankPct ?? Math.min(1, Math.max(0, (warVal + 1) / 4))}
                    centerText={`${warVal >= 0 ? "+" : ""}${warVal.toFixed(2)}`}
                    label="WAR"
                    sublabel={data.war_rank ? `#${data.war_rank}` : "rating"}
                    themeColor={teamColor}
                    size={96}
                  />
                  <div className="absolute top-0 right-0">
                    <StatInfoTip label="Wins Above Replacement (WAR)"
                      tip="Single-number value vs a freely available AHL callup, summed across offense, defense and special teams. +2.5 is elite. The ring fill is the rank-percentile across qualified skaters when available, else a normalized scale around 0." />
                  </div>
                </div>
              )}
            </div>

            {ewmaWave.length > 0 && (
              <div className="mt-4 pt-3 border-t border-white/[0.04]">
                <Waveform
                  data={ewmaWave}
                  themeColor={teamColor}
                  label="EWMA xGF/60 · synth trail"
                  width={260}
                  height={54}
                  ariaLabel="EWMA momentum waveform"
                />
              </div>
            )}

            {/* Quick-glance bio strip (height/weight/age/shoots) — fills the
                empty vertical space when Neural Cortex is tall. */}
            {bio && (
              <div className="mt-4 pt-3 border-t border-white/[0.04] grid grid-cols-2 gap-y-1 gap-x-2 text-[10px]">
                {bio.height_cm != null && (
                  <>
                    <span className="hud-mono uppercase tracking-[0.16em] text-[var(--text-secondary)]">HT</span>
                    <span className="hud-mono tabular-nums text-right" style={{ color: teamColor }}>
                      {fmtHeight(bio.height_cm).split(" / ")[0]}
                    </span>
                  </>
                )}
                {bio.weight_kg != null && (
                  <>
                    <span className="hud-mono uppercase tracking-[0.16em] text-[var(--text-secondary)]">WT</span>
                    <span className="hud-mono tabular-nums text-right" style={{ color: teamColor }}>
                      {Math.round(bio.weight_kg * 2.205)} lb
                    </span>
                  </>
                )}
                {bio.birth_date && (
                  <>
                    <span className="hud-mono uppercase tracking-[0.16em] text-[var(--text-secondary)]">AGE</span>
                    <span className="hud-mono tabular-nums text-right" style={{ color: teamColor }}>
                      {calcAge(bio.birth_date) ?? "—"}
                    </span>
                  </>
                )}
                {bio.shoots_catches && (
                  <>
                    <span className="hud-mono uppercase tracking-[0.16em] text-[var(--text-secondary)]">{isGoalie ? "CATCH" : "SHOOTS"}</span>
                    <span className="hud-mono tabular-nums text-right" style={{ color: teamColor }}>
                      {bio.shoots_catches}
                    </span>
                  </>
                )}
              </div>
            )}

            {/* Status flags — pushed to bottom so the panel evenly fills */}
            {statusFlags.length > 0 && (
              <div className="mt-auto pt-3 flex flex-wrap gap-1.5">
                {statusFlags.map((f, i) => (
                  <HudBadge key={i} tone={f.tone} pulse={f.pulse} themeColor={teamColor}>
                    {f.label}
                  </HudBadge>
                ))}
              </div>
            )}
          </HudPanel>
        </div>

        {/* HOLOGRAM column — body silhouette with Iron Man HUD */}
        <div className="lg:col-span-5 flex">
          <HudPanel title="Hologram" subtitle={isGoalie ? "goalie scan" : "skater scan"} themeColor={teamColor} scanline allCorners className="w-full flex flex-col">
            <HologramScanner
              isGoalie={!!isGoalie}
              teamColor={teamColor}
              bodyIntensity={bodyIntensity}
              telemetryLeft={isGoalie ? [
                // Goalies: callouts surface ONLY stats that aren't already
                // in the TARGET PROFILE strip (SV%, GAA, W-L, SO, GP). The
                // hologram leans into the 3-zone save breakdown + GSAx +
                // shots faced so it tells a different story than the
                // header stats.
                { id: "hdsv",   label: "HD SV%",    val: data.hdsv_pct != null ? `${(data.hdsv_pct * 100).toFixed(1)}%` : null, target: "head" as const,
                  tip: "High-danger save percentage — most predictive goalie metric. League avg ~.800; elite goalies clear .830." },
                { id: "mdsv",   label: "MD SV%",    val: data.mdsv_pct != null ? `${(data.mdsv_pct * 100).toFixed(1)}%` : null, target: "torso" as const,
                  tip: "Medium-danger save percentage. League avg ~.905. Reads tracking + rebound control on mid-range shots." },
                { id: "ldsv",   label: "LD SV%",    val: data.ldsv_pct != null ? `${(data.ldsv_pct * 100).toFixed(1)}%` : null, target: "legs" as const,
                  tip: "Low-danger save percentage. League avg ~.965. Anything below means soft goals leaking through." },
              ].filter(c => c.val !== null) : [
                // Skater hologram L — MAX SPEED moved to EDGE TELEMETRY band
                // (where it carries rank + percentile). Hologram surfaces
                // physical-role stats only so the silhouette tells the body's
                // story, not the engine's.
                { id: "hits",     label: "HITS/60",    val: data.hits_per60 != null ? data.hits_per60.toFixed(1) : null, target: "arms" as const,
                  tip: "Hits thrown per 60 minutes of even-strength play. 3+ is a clear physical role; below 1 is a finesse profile." },
                { id: "blocks",   label: "BLOCKS/60",  val: data.blocks_per60 != null ? data.blocks_per60.toFixed(1) : null, target: "torso" as const,
                  tip: "Shots blocked per 60 minutes — proxy for shot-lane defending. Defensemen routinely 4+; forwards rarely above 2." },
                { id: "netfront", label: "NET FRONT",  val: data.net_front_pct != null ? `${(data.net_front_pct).toFixed(0)}%` : null, target: "torso" as const,
                  tip: "Share of OZ shots taken from net-front. High = crashes the slot; low = perimeter/cycle profile." },
              ].filter(c => c.val !== null)}
              telemetryRight={isGoalie ? [
                { id: "gsax",   label: "GSAx",      val: data.gsax != null ? `${data.gsax > 0 ? "+" : ""}${data.gsax.toFixed(1)}` : null, target: "head" as const,
                  tip: "Goals saved above expected — value-added vs an average goalie given the same shot diet. >+10 over a season is elite; negative means letting in more than the model expects." },
                { id: "xga",    label: "xGA",       val: data.xga != null ? data.xga.toFixed(1) : null, target: "torso" as const,
                  tip: "Expected goals against — the model's read on shot quality faced. Pair with goals_against for the GSAx delta." },
                { id: "shots",  label: "SA",        val: data.shots_against != null ? `${data.shots_against}` : null, target: "arms" as const,
                  tip: "Shots faced this season — workload proxy. Pair with GP for shots-per-game." },
              ].filter(c => c.val !== null) : [
                { id: "battle",   label: "BATTLE",     val: data.battle_percentile != null ? `${data.battle_percentile.toFixed(0)}th` : null, target: "torso" as const,
                  tip: "Puck-battle percentile across hits, blocks, and contested zone battles combined. 85th pct = wins more pucks than 85% of skaters." },
                { id: "toi",      label: "EV TOI",     val: data.toi_ev != null ? `${data.toi_ev.toFixed(0)}m` : null, target: "head" as const,
                  tip: "Total 5v5 even-strength minutes this season. Higher = more coach trust. All per-60 metrics on the page normalize against this." },
                { id: "edge",     label: "EDGE Δ",     val: phase3?.edge_load != null ? `${phase3.edge_load >= 0 ? "+" : ""}${(phase3.edge_load * 100).toFixed(1)}%` : null, target: "legs" as const,
                  tip: "EDGE skating-load degradation vs the player's own baseline. Negative = moving slower/shorter distance than their norm — fatigue showing in the legs." },
              ].filter(c => c.val !== null)}
              tickerLine={isGoalie ? [
                data.sv_pct != null ? `SV ${(data.sv_pct * 100).toFixed(1)}%` : (nhlStats?.sv_pct != null ? `SV ${(nhlStats.sv_pct * 100).toFixed(1)}%` : null),
                data.hdsv_pct != null ? `HD ${(data.hdsv_pct * 100).toFixed(1)}%` : null,
                data.mdsv_pct != null ? `MD ${(data.mdsv_pct * 100).toFixed(1)}%` : null,
                data.ldsv_pct != null ? `LD ${(data.ldsv_pct * 100).toFixed(1)}%` : null,
                data.gsax != null ? `GSAx ${data.gsax > 0 ? "+" : ""}${data.gsax.toFixed(1)}` : null,
                nhlStats?.gaa != null ? `GAA ${nhlStats.gaa.toFixed(2)}` : null,
                data.shots_against != null ? `SA ${data.shots_against}` : null,
                data.games_played != null ? `GP ${data.games_played}` : null,
                nhlStats?.shutouts != null ? `SO ${nhlStats.shutouts}` : null,
                fi != null ? `FI ${fi.toFixed(2)}` : null,
                ci != null ? `CI ${ci >= 0 ? "+" : ""}${ci.toFixed(2)}` : null,
              ].filter((s): s is string => Boolean(s)) : [
                fi != null ? `FI ${fi.toFixed(2)}` : null,
                ci != null ? `CI ${ci >= 0 ? "+" : ""}${ci.toFixed(2)}` : null,
                hhs != null ? `HHS ${hhs >= 0 ? "+" : ""}${hhs.toFixed(2)}` : null,
                warVal != null ? `WAR ${warVal >= 0 ? "+" : ""}${warVal.toFixed(2)}` : null,
                data.xgf_per60 != null ? `xGF/60 ${data.xgf_per60.toFixed(2)}` : null,
                data.cdr != null ? `CDR ${data.cdr >= 0 ? "+" : ""}${data.cdr.toFixed(2)}` : null,
                data.finishing != null ? `FIN ${data.finishing >= 0 ? "+" : ""}${data.finishing.toFixed(1)}` : null,
                data.bayesian_rating != null ? `BAYES ${data.bayesian_rating.toFixed(3)}` : null,
                phase3?.fi_multiplier != null ? `MULT ${phase3.fi_multiplier.toFixed(3)}` : null,
              ].filter((s): s is string => Boolean(s))}
              dnaRungs={(() => {
                // Neural DNA strip — one rung per behavior NN dimension for
                // skaters (CRY / DMP / SLT / PRM / DRV / BTL / HLD), and one
                // rung per goalie save-zone axis for goalies. Weight is the
                // probability / save% scaled to 0-1 so the helix encodes
                // the player's identity as a visible fingerprint.
                const r = (id: string, label: string, weight: number | null, tip: string): DnaRung | null =>
                  weight == null ? null : {
                    id,
                    label,
                    weight: Math.max(0.05, Math.min(1, weight)),
                    color: ACTION_THEME[id]?.color ?? teamColor,
                    tip,
                  };
                if (isGoalie) {
                  const rg = (id: string, label: string, weight: number | null, color: string, tip: string): DnaRung | null =>
                    weight == null ? null : { id, label, weight: Math.max(0.05, Math.min(1, weight)), color, tip };
                  // Re-scale save% to 0-1 against a useful band so weak vs
                  // strong zones actually look different on the helix.
                  const normSv = (v: number | null, lo: number, hi: number) =>
                    v == null ? null : Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
                  const gsaxNorm = data.gsax != null ? Math.max(0, Math.min(1, (data.gsax + 10) / 30)) : null;
                  return [
                    rg("hd",  "HDV", normSv(data.hdsv_pct ?? null, 0.70, 0.90), "#f87171", "High-danger save %"),
                    rg("md",  "MDV", normSv(data.mdsv_pct ?? null, 0.82, 0.96), "#fbbf24", "Mid-danger save %"),
                    rg("ld",  "LDV", normSv(data.ldsv_pct ?? null, 0.94, 1.00), "#5ee08a", "Low-danger save %"),
                    rg("ovr", "SVP", normSv(data.sv_pct  ?? null, 0.880, 0.925), teamColor, "Overall save %"),
                    rg("gsx", "GSX", gsaxNorm, "#a78bfa", "Goals saved above expected"),
                    rg("vol", "VOL", data.xga != null ? Math.min(1, data.xga / 80) : null, "#38bdf8", "Workload (xGA)"),
                  ].filter((x): x is DnaRung => x !== null);
                }
                return [
                  r("carry",  "CRY", data.nn_carry_in_pct        != null ? data.nn_carry_in_pct        / 60 : null, "Carry-in rate"),
                  r("dump",   "DMP", data.nn_dump_pct            != null ? data.nn_dump_pct            / 60 : null, "Dump-in rate"),
                  r("slot",   "SLT", data.nn_shoot_slot_pct      != null ? data.nn_shoot_slot_pct      / 40 : null, "Slot shot rate"),
                  r("perim",  "PRM", data.nn_shoot_perimeter_pct != null ? data.nn_shoot_perimeter_pct / 40 : null, "Perimeter shot rate"),
                  r("drive",  "DRV", data.nn_drive_net_pct       != null ? data.nn_drive_net_pct       / 30 : null, "Drive-net rate"),
                  r("battle", "BTL", data.nn_battle_corner_pct   != null ? data.nn_battle_corner_pct   / 30 : null, "Battle-corner rate"),
                  r("hold",   "HLD", data.nn_hold_corner_pct     != null ? data.nn_hold_corner_pct     / 30 : null, "Hold-possession rate"),
                ].filter((x): x is DnaRung => x !== null);
              })()}
              dnaSignature={isGoalie ? "G · save-zone v1" : "S · BNN v2.22"}
            />
            {/* Legacy hologram canvas (replaced by HologramScanner above) */}
            <div className="hidden">
            <div className="relative flex items-center justify-center py-2 min-h-[380px] flex-1">
              {/* IRON MAN HUD — animated multi-ring scanner backdrop */}
              <div
                className="absolute inset-0 pointer-events-none"
                style={{
                  background: `radial-gradient(circle at 50% 50%, ${teamColor}30 0%, ${teamColor}08 35%, transparent 65%)`,
                  filter: "blur(12px)",
                }}
              />
              <svg viewBox="0 0 400 360" className="absolute inset-0 w-full h-full pointer-events-none" aria-hidden>
                <defs>
                  <linearGradient id="hudArc" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor={teamColor} stopOpacity="0" />
                    <stop offset="40%" stopColor={teamColor} stopOpacity="0.85" />
                    <stop offset="100%" stopColor={teamColor} stopOpacity="0" />
                  </linearGradient>
                  <linearGradient id="hudArc2" x1="1" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={teamColor} stopOpacity="0" />
                    <stop offset="60%" stopColor={teamColor} stopOpacity="0.55" />
                    <stop offset="100%" stopColor={teamColor} stopOpacity="0" />
                  </linearGradient>
                </defs>

                {/* Outer slow-rotating dashed ring */}
                <g style={{ transformOrigin: "200px 180px", animation: "ironRotateSlow 32s linear infinite" }}>
                  <circle cx={200} cy={180} r={170} fill="none" stroke={teamColor} strokeOpacity={0.15} strokeDasharray="2 8" />
                  {/* Tick marks at cardinal points */}
                  {[0, 45, 90, 135, 180, 225, 270, 315].map((deg, i) => {
                    const rad = (deg * Math.PI) / 180;
                    const x1 = 200 + Math.cos(rad) * 162;
                    const y1 = 180 + Math.sin(rad) * 162;
                    const x2 = 200 + Math.cos(rad) * 178;
                    const y2 = 180 + Math.sin(rad) * 178;
                    return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={teamColor} strokeOpacity={0.55} strokeWidth={1.4} />;
                  })}
                </g>

                {/* Middle counter-rotating arc */}
                <g style={{ transformOrigin: "200px 180px", animation: "ironRotateRev 18s linear infinite" }}>
                  <circle cx={200} cy={180} r={140} fill="none" stroke="url(#hudArc)" strokeWidth={1.5} strokeDasharray="60 40 25 40" strokeLinecap="round" />
                </g>

                {/* Inner fast arc */}
                <g style={{ transformOrigin: "200px 180px", animation: "ironRotateFast 8s linear infinite" }}>
                  <circle cx={200} cy={180} r={110} fill="none" stroke="url(#hudArc2)" strokeWidth={1.2} strokeDasharray="20 80" strokeLinecap="round" />
                </g>

                {/* Innermost dotted ring */}
                <circle cx={200} cy={180} r={88} fill="none" stroke={teamColor} strokeOpacity={0.18} strokeDasharray="1 5" />

                {/* Crosshair */}
                <line x1={200} y1={0} x2={200} y2={360} stroke={teamColor} strokeOpacity={0.05} strokeDasharray="3 10" />
                <line x1={0} y1={180} x2={400} y2={180} stroke={teamColor} strokeOpacity={0.05} strokeDasharray="3 10" />

                {/* Corner reticle markers */}
                {[
                  { x: 40,  y: 40,  dx: 12, dy: 0  }, { x: 40,  y: 40,  dx: 0,  dy: 12 },
                  { x: 360, y: 40,  dx: -12, dy: 0 }, { x: 360, y: 40,  dx: 0,  dy: 12 },
                  { x: 40,  y: 320, dx: 12, dy: 0  }, { x: 40,  y: 320, dx: 0,  dy: -12 },
                  { x: 360, y: 320, dx: -12, dy: 0 }, { x: 360, y: 320, dx: 0,  dy: -12 },
                ].map((m, i) => (
                  <line key={i} x1={m.x} y1={m.y} x2={m.x + m.dx} y2={m.y + m.dy} stroke={teamColor} strokeOpacity={0.65} strokeWidth={1.5} strokeLinecap="round" />
                ))}

                {/* Pulse rings — emit from center on a slow loop */}
                <circle cx={200} cy={180} r={50} fill="none" stroke={teamColor} strokeOpacity={0.4} strokeWidth={1} style={{ animation: "ironPulse 4.4s ease-out infinite" }} />
                <circle cx={200} cy={180} r={50} fill="none" stroke={teamColor} strokeOpacity={0.4} strokeWidth={1} style={{ animation: "ironPulse 4.4s ease-out infinite 2.2s" }} />
              </svg>

              {/* Iron Man corner indicator brackets */}
              <span aria-hidden className="absolute top-1 left-1 hud-mono text-[8px] uppercase tracking-[0.18em]" style={{ color: teamColor }}>◢ SCAN</span>
              <span aria-hidden className="absolute top-1 right-1 hud-mono text-[8px] uppercase tracking-[0.18em]" style={{ color: teamColor }}>LOCK ◣</span>
              <span aria-hidden className="absolute bottom-1 left-1 hud-mono text-[8px] uppercase tracking-[0.18em]" style={{ color: teamColor }}>◤ INTEL</span>
              <span aria-hidden className="absolute bottom-1 right-1 hud-mono text-[8px] uppercase tracking-[0.18em]" style={{ color: teamColor }}>LIVE ◥</span>

              <style jsx>{`
                @keyframes ironRotateSlow { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
                @keyframes ironRotateRev  { from { transform: rotate(360deg); } to { transform: rotate(0deg); } }
                @keyframes ironRotateFast { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
                @keyframes ironPulse {
                  0%   { r: 50; stroke-opacity: 0.55; }
                  100% { r: 165; stroke-opacity: 0; }
                }
                @media (prefers-reduced-motion: reduce) {
                  svg g { animation: none !important; }
                  circle { animation: none !important; }
                }
              `}</style>

              {/* Silhouette — interactive zone-aware skater scan */}
              <div className="relative" style={{ width: 220, height: 360 }}>
                <BodySilhouette
                  themeColor={teamColor}
                  intensity={bodyIntensity}
                  width={220}
                  height={360}
                  variant={isGoalie ? "goalie" : "skater"}
                />

                {/* Vertical scan line that sweeps top → bottom */}
                <div
                  aria-hidden
                  className="absolute left-0 right-0 pointer-events-none"
                  style={{
                    top: 0,
                    height: "2px",
                    background: `linear-gradient(90deg, transparent, ${teamColor}cc, transparent)`,
                    boxShadow: `0 0 12px ${teamColor}aa`,
                    mixBlendMode: "screen",
                    animation: "holoVScan 4.2s ease-in-out infinite",
                  }}
                />

                {/* Zone hotspot dots (positions match BodySilhouette in 200×340 viewBox,
                    scaled here to 220×360). Pulse intensity by FI sub-component. */}
                {(() => {
                  const sx = 220 / 200, sy = 360 / 340;
                  const dot = (cx: number, cy: number, key: string, intensity: number) => (
                    <span
                      key={key}
                      aria-hidden
                      className="absolute rounded-full pointer-events-none"
                      style={{
                        left: cx * sx - 4,
                        top: cy * sy - 4,
                        width: 8,
                        height: 8,
                        background: intensity >= 0.5 ? "#f87171" : intensity >= 0.25 ? "#fbbf24" : teamColor,
                        boxShadow: `0 0 8px ${intensity >= 0.5 ? "#f87171" : intensity >= 0.25 ? "#fbbf24" : teamColor}`,
                        animation: `holoNode ${1.8 + (key.length % 3) * 0.4}s ease-in-out infinite`,
                      }}
                    />
                  );
                  return (
                    <>
                      {dot(100, 36, "head",  bodyIntensity.head ?? 0)}
                      {dot(60,  70, "shdrL", bodyIntensity.shoulder ?? 0)}
                      {dot(140, 70, "shdrR", bodyIntensity.shoulder ?? 0)}
                      {dot(100, 130,"torso", bodyIntensity.torso ?? 0)}
                      {dot(57,  140,"armL",  bodyIntensity.armL ?? 0)}
                      {dot(143, 140,"armR",  bodyIntensity.armR ?? 0)}
                      {dot(88,  280,"legL",  bodyIntensity.legL ?? 0)}
                      {dot(112, 280,"legR",  bodyIntensity.legR ?? 0)}
                    </>
                  );
                })()}
              </div>

              {/* LEFT telemetry callouts — leader lines pointing at body zones */}
              {!isGoalie && (
                <div className="absolute left-2 top-2 bottom-2 flex flex-col justify-between text-right pointer-events-none">
                  {[
                    { label: "MAX SPEED", val: data.skating_max_speed_kmh != null ? `${data.skating_max_speed_kmh.toFixed(1)} km/h` : "—", show: data.skating_max_speed_kmh != null },
                    { label: "DISTANCE",  val: data.skating_distance_per_game_km != null ? `${data.skating_distance_per_game_km.toFixed(2)} km/g` : "—", show: data.skating_distance_per_game_km != null },
                    { label: "HITS/60",   val: data.hits_per60 != null ? data.hits_per60.toFixed(1) : "—", show: data.hits_per60 != null },
                    { label: "BLOCKS/60", val: data.blocks_per60 != null ? data.blocks_per60.toFixed(1) : "—", show: data.blocks_per60 != null },
                  ].filter(c => c.show).slice(0, 4).map((c, i) => (
                    <div key={i} className="flex flex-col items-end gap-0.5 px-1.5 py-1 rounded backdrop-blur"
                      style={{
                        background: "rgba(0,0,0,0.35)",
                        border: `1px solid ${teamColor}22`,
                      }}>
                      <span className="hud-mono text-[8px] uppercase tracking-[0.18em] text-[var(--text-secondary)]">
                        ▸ {c.label}
                      </span>
                      <span className="hud-mono text-[10px] tabular-nums" style={{ color: teamColor }}>
                        {c.val}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* RIGHT telemetry callouts */}
              {!isGoalie && (
                <div className="absolute right-2 top-2 bottom-2 flex flex-col justify-between text-left pointer-events-none">
                  {[
                    { label: "BATTLE",  val: data.battle_percentile != null ? `${data.battle_percentile.toFixed(0)}th` : "—",          show: data.battle_percentile != null },
                    { label: "EV TOI",  val: data.toi_ev != null ? `${data.toi_ev.toFixed(0)}m` : "—",                                 show: data.toi_ev != null },
                    { label: "EDGE Δ",  val: phase3?.edge_load != null ? `${phase3.edge_load >= 0 ? "+" : ""}${(phase3.edge_load * 100).toFixed(1)}%` : "—", show: phase3?.edge_load != null },
                    { label: "REST",    val: phase3?.goalie_rest_days != null ? `${phase3.goalie_rest_days}d` : (data.ewma_games != null ? `${data.ewma_games} GP` : "—"), show: true },
                  ].filter(c => c.show).slice(0, 4).map((c, i) => (
                    <div key={i} className="flex flex-col items-start gap-0.5 px-1.5 py-1 rounded backdrop-blur"
                      style={{
                        background: "rgba(0,0,0,0.35)",
                        border: `1px solid ${teamColor}22`,
                      }}>
                      <span className="hud-mono text-[8px] uppercase tracking-[0.18em] text-[var(--text-secondary)]">
                        {c.label} ◂
                      </span>
                      <span className="hud-mono text-[10px] tabular-nums" style={{ color: teamColor }}>
                        {c.val}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* Particle drift — decorative motes around the figure */}
              <div className="absolute inset-0 pointer-events-none overflow-hidden">
                {Array.from({ length: 14 }).map((_, i) => (
                  <span
                    key={i}
                    aria-hidden
                    className="absolute rounded-full"
                    style={{
                      left: `${(i * 73) % 100}%`,
                      top: `${(i * 41) % 100}%`,
                      width: "2px",
                      height: "2px",
                      background: teamColor,
                      opacity: 0.4 + (i % 3) * 0.15,
                      boxShadow: `0 0 4px ${teamColor}`,
                      animation: `holoMote ${4 + (i % 5)}s ease-in-out infinite ${i * 0.3}s`,
                    }}
                  />
                ))}
              </div>

              {/* Bottom ticker — scrolling stat readout */}
              <div className="absolute left-2 right-2 bottom-2 h-5 overflow-hidden rounded"
                style={{ background: "rgba(0,0,0,0.45)", border: `1px solid ${teamColor}22` }}>
                <div className="absolute inset-y-0 flex items-center gap-6 whitespace-nowrap hud-mono text-[9px] uppercase tracking-[0.18em] px-3"
                  style={{ color: `${teamColor}cc`, animation: "holoTicker 28s linear infinite" }}>
                  {[
                    fi != null ? `FI ${fi.toFixed(2)}` : null,
                    ci != null ? `CI ${ci >= 0 ? "+" : ""}${ci.toFixed(2)}` : null,
                    hhs != null ? `HHS ${hhs >= 0 ? "+" : ""}${hhs.toFixed(2)}` : null,
                    warVal != null ? `WAR ${warVal >= 0 ? "+" : ""}${warVal.toFixed(2)}` : null,
                    data.xgf_per60 != null ? `xGF/60 ${data.xgf_per60.toFixed(2)}` : null,
                    data.cdr != null ? `CDR ${data.cdr >= 0 ? "+" : ""}${data.cdr.toFixed(2)}` : null,
                    data.finishing != null ? `FIN ${data.finishing >= 0 ? "+" : ""}${data.finishing.toFixed(1)}` : null,
                    data.bayesian_rating != null ? `BAYES ${data.bayesian_rating.toFixed(3)}` : null,
                    phase3?.fi_multiplier != null ? `MULT ${phase3.fi_multiplier.toFixed(3)}` : null,
                  ].filter(Boolean).map((t, i) => (
                    <span key={i} className="inline-flex items-center gap-1">
                      <span className="inline-block w-1 h-1 rounded-full" style={{ background: teamColor }} />
                      {t}
                    </span>
                  ))}
                  {/* Duplicate for seamless loop */}
                  {[
                    fi != null ? `FI ${fi.toFixed(2)}` : null,
                    ci != null ? `CI ${ci >= 0 ? "+" : ""}${ci.toFixed(2)}` : null,
                    hhs != null ? `HHS ${hhs >= 0 ? "+" : ""}${hhs.toFixed(2)}` : null,
                    warVal != null ? `WAR ${warVal >= 0 ? "+" : ""}${warVal.toFixed(2)}` : null,
                    data.xgf_per60 != null ? `xGF/60 ${data.xgf_per60.toFixed(2)}` : null,
                    data.cdr != null ? `CDR ${data.cdr >= 0 ? "+" : ""}${data.cdr.toFixed(2)}` : null,
                    data.finishing != null ? `FIN ${data.finishing >= 0 ? "+" : ""}${data.finishing.toFixed(1)}` : null,
                    data.bayesian_rating != null ? `BAYES ${data.bayesian_rating.toFixed(3)}` : null,
                    phase3?.fi_multiplier != null ? `MULT ${phase3.fi_multiplier.toFixed(3)}` : null,
                  ].filter(Boolean).map((t, i) => (
                    <span key={`d${i}`} className="inline-flex items-center gap-1">
                      <span className="inline-block w-1 h-1 rounded-full" style={{ background: teamColor }} />
                      {t}
                    </span>
                  ))}
                </div>
              </div>

              <style jsx>{`
                @keyframes holoVScan {
                  0%   { transform: translateY(0);   opacity: 0; }
                  10%  { opacity: 1; }
                  90%  { opacity: 1; }
                  100% { transform: translateY(360px); opacity: 0; }
                }
                @keyframes holoNode {
                  0%, 100% { transform: scale(1);   opacity: 0.65; }
                  50%      { transform: scale(1.4); opacity: 1; }
                }
                @keyframes holoMote {
                  0%, 100% { transform: translate(0, 0);   opacity: 0.2; }
                  50%      { transform: translate(8px, -6px); opacity: 0.7; }
                }
                @keyframes holoTicker {
                  from { transform: translateX(0); }
                  to   { transform: translateX(-50%); }
                }
                @media (prefers-reduced-motion: reduce) {
                  div { animation: none !important; }
                  span { animation: none !important; }
                }
              `}</style>
            </div>
            </div>
          </HudPanel>
        </div>

        {/* NEURAL CORTEX column */}
        <div className="lg:col-span-4 flex">
          <HudPanel title="Neural Cortex" themeColor={teamColor} scanline allCorners className="w-full flex flex-col">
            {/* MODEL STATUS strip — live signal vs idle */}
            <div className="flex items-center gap-2 mb-2 px-2 py-1.5 rounded border"
              style={{ borderColor: `${teamColor}33`, background: "rgba(0,0,0,0.30)" }}>
              <span className="hud-pulse-dot" style={{ background: "#4ade80" }} />
              <span className="hud-mono jarvis-flicker text-[9px] uppercase tracking-[0.18em] text-[#4ade80]">INFER</span>
              <span className="hud-mono text-[8px] uppercase tracking-[0.16em] text-[var(--text-muted)]">·</span>
              <span className="hud-mono text-[9px] uppercase tracking-[0.18em]" style={{ color: teamColor }}>
                BNN · v2.22
              </span>
              <span className="ml-auto hud-mono text-[8px] uppercase tracking-[0.18em] text-[var(--text-secondary)]">
                {neuralNodes.length > 0 ? `${neuralNodes.length} CH` : "0 CH"}
              </span>
            </div>

            {/* Archetype banner */}
            {playerType && (
              <div className="hud-mono text-[10px] uppercase tracking-[0.18em] px-2 py-1.5 mb-2 rounded border flex items-center gap-2"
                style={{ color: teamColor, borderColor: `${teamColor}55`, background: `${teamColor}0f` }}>
                <span className="hud-pulse-dot" style={{ background: teamColor }} />
                ★ {playerType}
              </div>
            )}

            {(() => {
              // Build goalie-specific neural nodes from save% by zone
              const goalieNodes: NeuralNode[] = isGoalie ? [
                { id: "hd",   label: "HD Save",    weight: data.hdsv_pct != null ? Math.min(1, Math.max(0, (data.hdsv_pct - 0.70) / 0.20)) : 0 },
                { id: "md",   label: "MD Save",    weight: data.mdsv_pct != null ? Math.min(1, Math.max(0, (data.mdsv_pct - 0.85) / 0.10)) : 0 },
                { id: "ld",   label: "LD Save",    weight: data.ldsv_pct != null ? Math.min(1, Math.max(0, (data.ldsv_pct - 0.94) / 0.06)) : 0 },
                { id: "ov",   label: "Overall",    weight: data.sv_pct   != null ? Math.min(1, Math.max(0, (data.sv_pct  - 0.880) / 0.045)) : 0 },
                { id: "gsax", label: "GSAx",       weight: data.gsax    != null ? Math.min(1, Math.max(0, (data.gsax + 10) / 30)) : 0 },
                { id: "vol",  label: "Workload",   weight: data.xga     != null ? Math.min(1, Math.max(0, data.xga / 80)) : 0 },
              ] : [];
              const activeNodes = isGoalie ? goalieNodes : neuralNodes;
              const hasNodes    = activeNodes.length > 0 && activeNodes.some(n => n.weight > 0);
              if (!hasNodes) {
                return (
                  <div className="py-6 text-center">
                    <p className="hud-mono text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">
                      Model not yet trained
                    </p>
                  </div>
                );
              }
              return (
                <div className="relative flex justify-center">
                  <svg viewBox="0 0 300 200" className="absolute inset-0 pointer-events-none" aria-hidden>
                    {[60, 80].map((r, i) => (
                      <circle key={i} cx={150} cy={100} r={r} fill="none" stroke={teamColor} strokeOpacity={0.08 + i * 0.04} strokeDasharray="2 6" />
                    ))}
                  </svg>
                  <NeuralGraph
                    center={isGoalie ? "G" : "NN"}
                    nodes={activeNodes}
                    themeColor={teamColor}
                    width={300}
                    height={200}
                  />
                </div>
              );
            })()}

            {/* PREDICTED PLAY — top-down zone schematic with the most likely
                entry + in-zone action sequence given current NN weights.
                Skaters only — the goalie's neural axes are save-related and
                don't map to ice positions. */}
            {!isGoalie && (
              <PredictedPlay
                carry={data.nn_carry_in_pct}
                dump={data.nn_dump_pct}
                slot={data.nn_shoot_slot_pct}
                perim={data.nn_shoot_perimeter_pct}
                drive={data.nn_drive_net_pct}
                battleC={data.nn_battle_corner_pct}
                holdC={data.nn_hold_corner_pct}
                themeColor={teamColor}
                leagueAvg={(() => {
                  // Prefer the position-matched segment so deltas read true:
                  // a D's perimeter shot rate vs forwards is meaningless;
                  // vs other D it tells you who the real point shooters are.
                  const pos = (data.position ?? "").toUpperCase();
                  const isD = pos === "D" || pos === "LD" || pos === "RD";
                  const isF = pos === "C" || pos === "L" || pos === "R" || pos === "LW" || pos === "RW";
                  const byPos = data.nn_league_avg_by_pos;
                  if (byPos) {
                    if (isD && byPos.defense)  return byPos.defense;
                    if (isF && byPos.forwards) return byPos.forwards;
                    if (byPos.all)             return byPos.all;
                  }
                  return data.nn_league_avg ?? null;
                })()}
                position={data.position ?? null}
                shoots={bio?.shoots_catches ?? data.shoots_catches ?? null}
                gpg={data.game_log?.summary.gpg ?? null}
                apg={data.game_log?.summary.apg ?? null}
                shotsPer60={data.shots_per60 ?? null}
                rapmOff={data.rapm_ev_off ?? null}
              />
            )}

            {/* DECISION MATRIX — animated bars per NN dimension (skater + goalie) */}
            {(() => {
              const goalieNodes: NeuralNode[] = isGoalie ? [
                { id: "hd",   label: "HD Save",    weight: data.hdsv_pct != null ? Math.min(1, Math.max(0, (data.hdsv_pct - 0.70) / 0.20)) : 0 },
                { id: "md",   label: "MD Save",    weight: data.mdsv_pct != null ? Math.min(1, Math.max(0, (data.mdsv_pct - 0.85) / 0.10)) : 0 },
                { id: "ld",   label: "LD Save",    weight: data.ldsv_pct != null ? Math.min(1, Math.max(0, (data.ldsv_pct - 0.94) / 0.06)) : 0 },
                { id: "ov",   label: "Overall",    weight: data.sv_pct   != null ? Math.min(1, Math.max(0, (data.sv_pct  - 0.880) / 0.045)) : 0 },
                { id: "gsax", label: "GSAx",       weight: data.gsax    != null ? Math.min(1, Math.max(0, (data.gsax + 10) / 30)) : 0 },
                { id: "vol",  label: "Workload",   weight: data.xga     != null ? Math.min(1, Math.max(0, data.xga / 80)) : 0 },
              ] : [];
              const matrixNodes = isGoalie ? goalieNodes : neuralNodes;
              if (matrixNodes.length === 0 || !matrixNodes.some(n => n.weight > 0)) return null;
              return (
              <div className="mt-2 pt-2 border-t border-white/[0.05] space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="hud-mono text-[9px] uppercase tracking-[0.18em]" style={{ color: teamColor }}>
                    ▸ DECISION MATRIX
                  </span>
                  <span className="hud-mono text-[8px] uppercase tracking-[0.14em] text-[var(--text-muted)]">
                    weight 0 → 1
                  </span>
                </div>
                {matrixNodes.slice(0, 6).map((n, i) => {
                  const NN_TIPS: Record<string, string> = {
                    carry:   "Probability the player carries the puck in across the offensive blue line on entries.",
                    dump:    "Probability the player dumps + chases on zone entries instead of carrying in.",
                    slot:    "Probability the player's next shot comes from the slot (high-danger area).",
                    drive:   "Probability the player drives the net rather than circling back or passing out.",
                    perim:   "Probability the player's next shot comes from the perimeter (outside the slot).",
                    battle:  "Probability the player engages in a corner board battle in the offensive zone.",
                    hold:    "Probability the player retains possession in a corner cycle rather than forcing a play.",
                    hd:      "High-danger save % normalized to 0–1. League avg HDsv% ≈ 80%.",
                    md:      "Mid-danger save % normalized to 0–1. League avg ≈ 87%.",
                    ld:      "Low-danger save % normalized to 0–1. NHL starters typically 96%+.",
                    ov:      "Overall save % normalized to 0–1 (range 0.880–0.925).",
                    gsax:    "Goals Saved Above Expected — net positive saves vs an average goalie on the same shots.",
                    vol:     "Workload index — expected goals against this goalie has faced (proxy for starter usage).",
                  };
                  const tip = NN_TIPS[n.id];
                  return (
                  <div key={n.id} className="flex flex-col gap-1 sm:flex-row sm:items-center sm:gap-2">
                    <div className="flex items-center justify-between gap-2 sm:justify-start sm:w-16 sm:shrink-0">
                      <span className="hud-mono text-[10px] sm:text-[9px] uppercase tracking-[0.10em] sm:tracking-[0.14em] text-[var(--text-secondary)] flex items-center gap-1 min-w-0 sm:truncate">
                        <span className="truncate">{n.label}</span>
                        {tip && (
                          <span className="ml-0.5"><StatInfoTip label={n.label} tip={tip} /></span>
                        )}
                      </span>
                      <span className="sm:hidden hud-mono text-[11px] tabular-nums font-semibold shrink-0" style={{ color: teamColor }}>
                        {n.weight.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex-1 h-2 rounded-sm overflow-hidden relative"
                      style={{ background: "rgba(255,255,255,0.04)", border: `1px solid ${teamColor}22` }}>
                      <div
                        className="h-full"
                        style={{
                          width: `${Math.round(n.weight * 100)}%`,
                          background: `linear-gradient(90deg, ${teamColor}aa 0%, ${teamColor} 100%)`,
                          boxShadow: `0 0 8px ${teamColor}55, inset 0 0 8px ${teamColor}44`,
                          animation: `decMatrix 900ms cubic-bezier(0.22,1,0.36,1) ${i * 70}ms backwards`,
                          transformOrigin: "left center",
                        }}
                      />
                      {/* Sweeping scan line — continuous, makes the bar "live" */}
                      <div
                        className="absolute top-0 bottom-0 w-6 pointer-events-none"
                        style={{
                          background: `linear-gradient(90deg, transparent, ${teamColor}aa, transparent)`,
                          animation: `decScan 3.4s linear infinite ${i * 0.25}s`,
                          mixBlendMode: "screen",
                          opacity: n.weight > 0.05 ? 0.7 : 0,
                        }}
                      />
                    </div>
                    <span className="hidden sm:inline hud-mono text-[10px] tabular-nums w-10 text-right font-semibold" style={{ color: teamColor }}>
                      {n.weight.toFixed(2)}
                    </span>
                  </div>
                  );
                })}
                <style jsx>{`
                  @keyframes decMatrix {
                    from { transform: scaleX(0); opacity: 0; }
                    to   { transform: scaleX(1); opacity: 1; }
                  }
                  @keyframes decScan {
                    0%   { transform: translateX(-30px); }
                    100% { transform: translateX(280px); }
                  }
                  @media (prefers-reduced-motion: reduce) {
                    div { animation: none !important; }
                  }
                `}</style>
              </div>
              );
            })()}

            {/* Quick neural readouts */}
            <div className="mt-3 pt-2 border-t border-white/[0.05] grid grid-cols-2 gap-2 text-center">
              {data.bayesian_rating != null && (
                <div className="hud-mono px-1.5 py-1 rounded" style={{ background: "rgba(255,255,255,0.02)" }}>
                  <div className="text-[8px] uppercase tracking-[0.16em] text-[var(--text-secondary)]">BAYESIAN</div>
                  <OdometerNumber value={data.bayesian_rating} decimals={3} className="text-sm" />
                </div>
              )}
              {data.clutch_index != null && (
                <div className="hud-mono px-1.5 py-1 rounded" style={{ background: "rgba(255,255,255,0.02)" }}>
                  <div className="text-[8px] uppercase tracking-[0.16em] text-[var(--text-secondary)]">CLUTCH</div>
                  <OdometerNumber value={data.clutch_index} decimals={4} className="text-sm" />
                </div>
              )}
              {data.hot_hand_score != null && (
                <div className="hud-mono px-1.5 py-1 rounded" style={{ background: "rgba(255,255,255,0.02)" }}>
                  <div className="text-[8px] uppercase tracking-[0.16em] text-[var(--text-secondary)]">HOT HAND</div>
                  <OdometerNumber value={data.hot_hand_score} decimals={2} className="text-sm" />
                </div>
              )}
              {data.contract_efficiency != null && (
                <div className="hud-mono px-1.5 py-1 rounded" style={{ background: "rgba(255,255,255,0.02)" }}>
                  <div className="text-[8px] uppercase tracking-[0.16em] text-[var(--text-secondary)]">CONTRACT</div>
                  <OdometerNumber value={data.contract_efficiency} decimals={2} suffix="x" className="text-sm" />
                </div>
              )}
            </div>
            {playStyle && (
              <p className="mt-2 text-[10px] text-[var(--text-secondary)] italic leading-relaxed text-center px-1">
                {playStyle}
              </p>
            )}
          </HudPanel>
        </div>

        {/* KPI band was duplicating the TARGET PROFILE stats strip
            (GP/G/A/P/+/-/FI for skaters, GP/W/L/SV%/GAA/SO for goalies)
            — removed. The headline totals live in the top hero. */}

        {/* EDGE / SAVE Telemetry band was moved INSIDE the Behavior tab,
            right under Performance Snapshot. Keeping a top-level band
            duplicated the same data twice on screen. */}

        {/* RATING STRIP — full width, 6 mini odometers, tier-coloured */}
        <div className="lg:col-span-12">
          <HudPanel title="Ratings" subtitle="aggregate engine inputs" themeColor={teamColor}>
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-3 text-center">
              {[
                { label: "xGF/60",     val: data.xgf_per60,            dec: 2, suffix: "", tier: data.xgf_per60     != null ? xgf60Tier(data.xgf_per60)         : null },
                { label: "xGA/60",     val: data.rapm_xga_60,          dec: 2, suffix: "", tier: data.rapm_xga_60   != null ? xgaAllowedTier(data.rapm_xga_60)  : null },
                { label: "CDR",        val: data.cdr,                  dec: 2, suffix: "", tier: data.cdr           != null ? defTier(data.cdr)                : null },
                { label: "Finishing",  val: data.finishing,            dec: 1, suffix: "", tier: data.finishing     != null ? finishingTier(data.finishing)    : null },
                { label: "PP xGF/60",  val: data.special_teams_pp,     dec: 2, suffix: "", tier: data.special_teams_pp != null ? stTier(data.special_teams_pp): null },
                { label: "Bayes",      val: data.bayesian_rating,      dec: 3, suffix: "", tier: data.bayesian_rating != null ? bayesianTier(data.bayesian_rating) : null },
              ].map((m, i) => {
                const tierColor = m.tier ? TIER_COLOR[m.tier] : null;
                return m.val != null ? (
                  <div key={i} className="flex flex-col items-center gap-0.5">
                    <span className="hud-mono text-[9px] uppercase tracking-[0.18em] text-[var(--text-secondary)]">
                      {m.label}
                    </span>
                    <OdometerNumber
                      value={m.val}
                      decimals={m.dec}
                      suffix={m.suffix}
                      className="text-base"
                    />
                    {m.tier && tierColor && (
                      <span
                        className="hud-mono text-[8px] uppercase tracking-[0.18em] rounded border px-1.5 py-0.5 mt-0.5"
                        style={{
                          color: tierColor,
                          borderColor: `${tierColor}55`,
                          backgroundColor: `${tierColor}14`,
                          textShadow: `0 0 6px ${tierColor}55`,
                        }}
                      >
                        {TIER_ABBREV[m.tier]}
                      </span>
                    )}
                  </div>
                ) : (
                  <div key={i} className="flex flex-col items-center gap-0.5 opacity-40">
                    <span className="hud-mono text-[9px] uppercase tracking-[0.18em] text-[var(--text-muted)]">
                      {m.label}
                    </span>
                    <span className="hud-mono text-base text-[var(--text-muted)]">—</span>
                  </div>
                );
              })}
            </div>
          </HudPanel>
        </div>

      </div>

      {/* ── Telemetry tab bar — sticky so the section nav stays accessible
          as cards scroll. Backdrop blur + team-tinted background so it
          reads as a HUD layer, not a floating bar. */}
      <div className="sticky top-0 z-30 mt-5 mb-3 px-1 -mx-1 py-1.5 flex items-center gap-2"
        style={{
          borderBottom: `1px solid ${teamColor}33`,
          background: `linear-gradient(180deg, rgba(0,0,0,0.85) 0%, rgba(0,0,0,0.72) 100%)`,
          backdropFilter: "blur(8px)",
          WebkitBackdropFilter: "blur(8px)",
          boxShadow: `0 4px 14px rgba(0,0,0,0.45), 0 0 8px ${teamColor}1c`,
        }}>
        <span className="hud-mono text-[9px] uppercase tracking-[0.2em] text-[var(--text-secondary)] shrink-0 pr-2 flex items-center gap-1.5">
          <span className="hud-pulse-dot" style={{ background: teamColor, boxShadow: `0 0 4px ${teamColor}` }} />
          <span>▌ TELEMETRY</span>
        </span>
        <HudTabBar
          tabs={telemetryTabs}
          active={telemetryTab}
          onChange={(id) => setTelemetryTab(id as TelemetryTab)}
          themeColor={teamColor}
          scroll
        />
      </div>

      {/* ── Cards grid ── */}
      <div className="grid gap-4 sm:grid-cols-2 min-w-0 overflow-x-hidden">

        {/* Tier legend — worst → best */}
        {(telemetryTab === "advanced" || telemetryTab === "fatigue") && (
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
        )}

        {/* Empty-state — no card matches the current tab */}
        {telemetryTab === "shot-map" && !isGoalie && shots.length === 0 && (
          <div className="sm:col-span-2 hud-mono text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)] text-center py-6">
            no shot data for this player yet
          </div>
        )}

        {/* ── Performance Snapshot charts ── */}
        {!isGoalie && telemetryTab === "neural" && (
          <div className="sm:col-span-2">
            <Card title="Performance Snapshot" style={cardStyle}>
              <div className="flex flex-wrap justify-center gap-6">

                {/* Radar */}
                {(data.xgf_per60 != null || data.cdr != null || data.battle_percentile != null) && (
                  <div className="flex flex-col items-center w-full sm:w-auto overflow-x-auto" style={{ scrollbarWidth: "none" }}>
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

        {/* EDGE Telemetry — moved here directly under Performance Snapshot
            so the NHL EDGE rank/percentile rows sit next to the radar
            instead of in a separate band up top. */}
        {!isGoalie && telemetryTab === "neural" && (
          data.edge_top_shot_speed_mph != null ||
          data.edge_top_skating_speed_kmh != null ||
          data.edge_total_distance_km != null ||
          data.skating_distance_per_game_km != null
        ) && (
          <div className="sm:col-span-2">
            <Card title="EDGE Telemetry" style={cardStyle}>
              <p className="text-[9px] uppercase tracking-wider text-center mb-3" style={{ color: "rgba(255,255,255,0.35)" }}>
                NHL EDGE · skating + shot tracking · league rank + percentile
              </p>
              <EdgeMetricsCard data={data} teamColor={teamColor} />
            </Card>
          </div>
        )}

        {/* Goalie variant — Save Telemetry mirrors the skater placement:
            sits directly under the Goalie Performance Snapshot. */}
        {isGoalie && telemetryTab === "neural" && (data.hdsv_pct != null || data.mdsv_pct != null || data.ldsv_pct != null) && (
          <div className="sm:col-span-2">
            <Card title="Save Telemetry" style={cardStyle}>
              <p className="text-[9px] uppercase tracking-wider text-center mb-3" style={{ color: "rgba(255,255,255,0.35)" }}>
                Net-mouth heatmap · save % colour-coded vs league averages
              </p>
              <div className="flex justify-center">
                <GoalieZoneViz data={data} teamColor={teamColor} />
              </div>
            </Card>
          </div>
        )}

        {/* ── Shot Map card — 3D + 2D toggle ── */}
        {!isGoalie && telemetryTab === "shot-map" && (
          <div className="sm:col-span-2">
            <Card title="Shot Map" style={cardStyle}>
              {shots.length > 0 ? (
                <>
                  <p className="text-[9px] uppercase tracking-wider text-center mb-3" style={{ color: "rgba(255,255,255,0.35)" }}>
                    Arena-adjusted shot locations · last 2 seasons · {shots.length} shots · {shots.filter(s => s.goal).length} goals
                  </p>
                  <Shot3D
                    shots={shots3D}
                    themeColor={teamColor}
                    flip
                    fallback={<ShotMapViz shots={shots} />}
                  />
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

        {/* ── Zone Tendencies — 3D rink fills the full width on top, ICE
            TIME deployment bar full-width underneath so it reads as a
            HUD instrument strip rather than a tucked-away sidebar. */}
        {!isGoalie && telemetryTab === "zones" && (
          <div className="sm:col-span-2">
            <Card title="Zone Tendencies" style={cardStyle}>
              <div className="flex flex-col gap-3 w-full">
                {/* TOP — 3D rink with 2D fallback (Zone3D ships its own
                    2D toggle so we don't add an external header). */}
                <div className="w-full min-w-0">
                  {data.nn_shoot_slot_pct != null ? (
                    <Zone3D
                      activations={{
                        slot:    data.nn_shoot_slot_pct ?? 0,
                        perim:   data.nn_shoot_perimeter_pct ?? 0,
                        net:     data.nn_drive_net_pct ?? 0,
                        cornerL: data.nn_battle_corner_pct ?? 0,
                        cornerR: data.nn_hold_corner_pct ?? 0,
                      }}
                      themeColor={teamColor}
                      fallback={<ZoneTendencyMap data={data} teamColor={teamColor} />}
                    />
                  ) : (
                    <div className="flex flex-col items-center justify-center py-10 gap-1.5">
                      <p className="hud-mono text-[9px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Model not yet trained</p>
                    </div>
                  )}
                </div>

                {/* BOTTOM — ICE TIME deployment bar (single full-width
                    segmented HUD strip). */}
                {data.skating_zone_time_oz_pct != null && (
                  <IceTimeByZoneBars data={data} teamColor={teamColor} />
                )}
              </div>
            </Card>
          </div>
        )}

        {/* Recent Games */}
        {telemetryTab === "advanced" && (
          <div className="sm:col-span-2">
            <Card title="Recent Games" style={cardStyle}>
              {gl && gl.games.length > 0 ? (
                <GameLogTable allGames={gl.games} />
              ) : (
                <div className="py-6 text-center hud-mono text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">
                  no recent game log yet
                </div>
              )}
            </Card>
          </div>
        )}

        {/* Section header — advanced two-up (Offense + Defense) */}
        {!isGoalie && telemetryTab === "advanced" && (
          <SectionHeader title="Advanced Metrics" subtitle="offense + defense" teamColor={teamColor} />
        )}

        {/* Offensive Profile */}
        {!isGoalie && telemetryTab === "advanced" && (data.finishing != null || data.war != null || data.rapm_ev_off != null) && (
          <Card title="Offensive Profile" style={cardStyle}>
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
                  tier={toiTier(data.toi_ev)}
                  sub="5v5 even-strength minutes this season"
                  tip="Total even-strength (5v5) minutes played this season. More TOI = more trust from the coaching staff. All other metrics are measured per 60 to account for ice time differences." />
              )}
            </div>
          </Card>
        )}

        {/* Defensive Profile */}
        {!isGoalie && telemetryTab === "advanced" && (data.cdr != null || data.rapm_ev_def != null || data.battle_score != null) && (
          <Card title="Defensive Profile" style={cardStyle}>
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
                <div className="px-4 py-2 border-b border-white/[0.05]">
                  <BarStat label="Hits / 60" value={data.hits_per60} max={8} color="#f97316" suffix="" />
                </div>
              )}
              {data.blocks_per60 != null && (
                <div className="px-4 py-2 border-b border-white/[0.05]">
                  <BarStat label="Blocked Shots / 60" value={data.blocks_per60} max={5} color="#38bdf8" suffix="" />
                </div>
              )}
            </div>
          </Card>
        )}

        {/* Special Teams */}
        {!isGoalie && telemetryTab === "advanced" && (data.special_teams_pp != null || data.special_teams_pk != null) && (
          <Card title="Special Teams" style={cardStyle}>
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
        {!isGoalie && telemetryTab === "neural" && (data.ewma_xgf60 != null || data.hot_hand_score != null || data.clutch_index != null) && (
          <Card title="Current Form" style={cardStyle}>
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

        {/* Section header — context block */}
        {!isGoalie && telemetryTab === "advanced" && (data.playoff_delta != null || data.former_team_boost != null || data.bayesian_rating != null) && (
          <SectionHeader title="Context" subtitle="playoffs · transitions · priors" teamColor={teamColor} />
        )}

        {/* Playoff & Context */}
        {!isGoalie && telemetryTab === "advanced" && (data.playoff_delta != null || data.former_team_boost != null || data.bayesian_rating != null) && (
          <Card title="Advanced Context" style={cardStyle}>
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

        {/* Section header — readiness block (fatigue + confidence) */}
        {!isGoalie && telemetryTab === "fatigue" && (phase3?.fatigue_index != null || phase3?.confidence_index != null) && (
          <SectionHeader title="Readiness" subtitle="fatigue · confidence · momentum" teamColor={teamColor} />
        )}
        {isGoalie && telemetryTab === "fatigue" && phase3?.goalie_fi != null && (
          <SectionHeader title="Readiness" subtitle="goalie fatigue · workload" teamColor={teamColor} />
        )}

        {/* Phase 3 — Fatigue & Schedule */}
        {!isGoalie && telemetryTab === "fatigue" && phase3 && phase3.fatigue_index != null && (() => {
          const fi    = phase3.fatigue_index ?? 0;
          const fiTier   = fatigueTier(fi);
          const colorHex = TIER_COLOR[fiTier];
          const sortedComps = phase3.fi_components
            ? Object.entries(phase3.fi_components).sort(([,a],[,b]) => b - a).filter(([,v]) => v > 0)
            : [];
          const monthAbbr: Record<number,string> = { 1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                                                     7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec" };
          return (
            <Card title="Fatigue & Schedule" style={cardStyle}>
              <div className="space-y-0">
                <StatRow
                  label="Fatigue Index"
                  value={fi.toFixed(3)}
                  tier={fiTier}
                  sub={phase3.fi_context_fallback
                    ? `Latest: ${phase3.fi_game_date} · regular-season fallback · no playoff FI computed yet`
                    : phase3.fi_game_date
                      ? `Latest: ${phase3.fi_game_date} · 0 = rested, 1 = severely fatigued`
                      : "0 = rested, 1 = severely fatigued"}
                  tip={phase3.fi_context_fallback
                    ? "Composite Fatigue Index (Feature 3.17). The playoff fatigue parquet hasn't been ingested for this run yet, so we surface the most recent regular-season row instead of going blank. The nightly Phase 3 build will replace this with a true playoff number once it lands."
                    : "Composite Fatigue Index (Feature 3.17): weighted sum of schedule density, travel, special-teams load, contact load, OT/fight load, recovery and roster strain — all in [0, 1]. Lower is better — the tier color is inverted so green = rested. The Rust engine uses this to scale player ratings game-by-game."}
                />
                {phase3.fi_multiplier != null && (
                  <StatRow
                    label="Rating Multiplier"
                    value={phase3.fi_multiplier.toFixed(3)}
                    tier={fiMultiplierTier(phase3.fi_multiplier)}
                    sub="Scaling factor applied to ratings before Rust sim · 1.000 = no drag"
                    tip="Feature 3.18 — FI → rating multiplier. ≈1.0 means no fatigue effect. Below 1.0 means we expect this player to underperform their rested baseline tonight."
                  />
                )}
                {phase3.is_anomaly && (
                  <StatRow
                    label="Anomaly Flag"
                    value={`z=${(phase3.anomaly_z ?? 0).toFixed(2)}`}
                    tier="Below Average"
                    sub={`Below baseline ${phase3.consecutive_below_n ?? 0} games in a row · not on IR`}
                    tip="Feature 3.19 — CUSUM + z-score SPC over a 20-game rolling baseline. The player is performing >2σ below their own norm without an injury report — possible hidden injury."
                  />
                )}
                {phase3.seasonal_factor != null && Math.abs(phase3.seasonal_factor) > 0.005 && (
                  <StatRow
                    label="Seasonal Motivation"
                    value={`${phase3.seasonal_factor >= 0 ? "+" : ""}${(phase3.seasonal_factor * 100).toFixed(2)}%`}
                    sub={`${monthAbbr[phase3.seasonal_month ?? 0] ?? ""} effect (Apr push / Jan dog days)`}
                    tip="Feature 3.22 — Month-of-season motivational modifier. April playoff push adds boost; January dog days subtract. Older players & high GP-to-date amplify January drag."
                  />
                )}
                {phase3.edge_load != null && Math.abs(phase3.edge_load) > 0.005 && (
                  <StatRow
                    label="EDGE Degradation"
                    value={`${phase3.edge_load >= 0 ? "+" : ""}${(phase3.edge_load * 100).toFixed(1)}%`}
                    sub="Predicted skating-metric drop vs. baseline"
                    tip="Feature 3.21 — average of predicted relative drop in speed / distance / carry / burst vs. the player's rested EDGE baseline (2.20), conditioned on current FI."
                  />
                )}
              </div>

              {/* Component breakdown bars */}
              {sortedComps.length > 0 && (
                <div className="mt-4 pt-4 border-t border-white/[0.05] space-y-2">
                  <div className="flex items-baseline justify-between">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-white/40">
                      FI Component Breakdown
                    </p>
                    <p className="text-[8px] font-mono text-white/25">
                      bar saturates at 0.30
                    </p>
                  </div>
                  {sortedComps.map(([k, v]) => (
                    <div key={k} className="flex items-center gap-2">
                      <span className="text-[10px] text-white/55 w-32 shrink-0 truncate">
                        {k.replace(/_/g, " ").replace(/ load$/, "")}
                      </span>
                      <div className="flex-1 h-1.5 rounded-full bg-white/[0.04] overflow-hidden relative">
                        <div
                          className="h-full"
                          style={{
                            width: `${Math.min(100, (v / 0.3) * 100)}%`,
                            backgroundColor: colorHex,
                            opacity: 0.85,
                          }}
                        />
                        {/* 50% midpoint tick */}
                        <div className="absolute top-0 bottom-0 w-px bg-white/[0.10]" style={{ left: "50%" }} />
                      </div>
                      <span className="text-[10px] font-mono text-white/65 w-12 text-right shrink-0">
                        {v.toFixed(3)}
                      </span>
                    </div>
                  ))}
                  {/* Scale legend */}
                  <div className="flex items-center gap-2 pt-1">
                    <span className="w-32 shrink-0" />
                    <div className="flex-1 flex justify-between text-[8px] font-mono text-white/25">
                      <span>0</span>
                      <span>0.15</span>
                      <span>0.30+</span>
                    </div>
                    <span className="w-12 shrink-0" />
                  </div>
                </div>
              )}

              {/* EDGE per-metric grid */}
              {(phase3.edge_speed != null || phase3.edge_distance != null) && (
                <div className="mt-4 pt-4 border-t border-white/[0.05]">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-white/40 mb-2">
                    Skating Δ vs Baseline (EDGE)
                  </p>
                  <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] font-mono">
                    {(["speed","distance","carry","burst"] as const).map((k) => {
                      const v = phase3[`edge_${k}` as keyof Phase3Card] as number | null;
                      const col = v == null ? "text-white/40" : v < 0 ? "text-[#f87171]" : "text-[#4ade80]";
                      return (
                        <div key={k} className="flex justify-between">
                          <span className="text-white/40">{k}</span>
                          <span className={col}>
                            {v == null ? "—" : (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + "%"}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </Card>
          );
        })()}

        {/* Confidence (Phase 17) — all players */}
        {telemetryTab === "fatigue" && phase3?.confidence_index != null && (() => {
          const ci = phase3.confidence_index;
          // Observed confidence_index values are concentrated in ±0.10 today
          // (the team-side signals are wired but not yet emitting), so the
          // tier breakpoints sit there rather than the documented ±1 scale.
          // Doumont's rule: thresholds should match the data we actually see.
          const tier: Tier =
              ci >=  0.05 ? "Above Average"
            : ci >=  0.02 ? "Average"
            : ci >= -0.02 ? "Average"
            : ci >= -0.05 ? "Below Average"
                          : "Low";
          const teamLive  = phase3.confidence_team != null && Math.abs(phase3.confidence_team) > 0.0005;
          const playerVal = phase3.confidence_player ?? 0;
          const shootTier = phase3.conf_shoot_bias    != null ? biasTier(phase3.conf_shoot_bias)    : undefined;
          const riskTier  = phase3.conf_risk_bias     != null ? biasTier(phase3.conf_risk_bias)     : undefined;
          const turnTier  = phase3.conf_turnover_bias != null ? biasTier(phase3.conf_turnover_bias) : undefined;
          const sortedComps = phase3.confidence_components
            ? Object.entries(phase3.confidence_components)
                .sort(([,a],[,b]) => Math.abs(b) - Math.abs(a))
                .filter(([,v]) => Math.abs(v) > 0.001)
                .slice(0, 12)
            : [];
          return (
            <Card title="Confidence (Phase 17)" style={cardStyle}>
              <div className="space-y-0">
                <StatRow
                  label="Confidence Index"
                  value={`${ci >= 0 ? "+" : ""}${ci.toFixed(3)}`}
                  tier={tier}
                  sub={phase3.confidence_date
                    ? `Latest: ${phase3.confidence_date} · negative = passive · positive = aggressive`
                    : "negative = passive · positive = aggressive"}
                  tip="Composite Confidence Index (Phase 17.24): signed weighted sum of hot-hand, EWMA form, TOI trust, role usage, injury drag, targeting, media, home/away, contract pressure, ref bias, and team-side context. The scale is documented as [-1, +1] but the in-flight pipeline emits ±0.10 today — tier breakpoints reflect that. The Rust engine will use this to bias decision-making (shoot-vs-pass, pinch-vs-retreat)."
                />
                {phase3.confidence_player != null && (
                  <StatRow
                    label="Player Component"
                    value={`${playerVal >= 0 ? "+" : ""}${playerVal.toFixed(3)}`}
                    tier={tier}
                    sub={`Sum of player signals (17.1–17.15) · contributes ${(playerVal * 0.7) >= 0 ? "+" : ""}${(playerVal * 0.7).toFixed(3)} after the 0.70 weight`}
                    tip="Pre-blend sum of the 15 player-side signals. The Confidence Index is 0.70 × this value + 0.30 × the team component."
                  />
                )}
                {teamLive ? (
                  (() => {
                    const tv = phase3.confidence_team!;
                    return (
                      <StatRow
                        label="Team Component"
                        value={`${tv >= 0 ? "+" : ""}${tv.toFixed(3)}`}
                        sub={`Sum of team signals (17.16–17.22) · contributes ${(tv * 0.3) >= 0 ? "+" : ""}${(tv * 0.3).toFixed(3)} after the 0.30 weight`}
                        tip="Pre-blend sum of the 7 team-side signals (streak, Corsi, special teams, coach challenges, comeback quality, goalie confidence, injury context). Multiplied by 0.30 before being added to the Confidence Index."
                      />
                    );
                  })()
                ) : (
                  <StatRow
                    label="Team Component"
                    value="not run"
                    sub="Team-side signals (17.16–17.22) emitting 0 — pipeline placeholder until they go live"
                    tip="Phase 17 team-side signals (streak, Corsi trend, ST trend, coach challenges, comeback quality, goalie confidence, roster disruption) are wired but their feature module hasn't shipped — every player currently reads 0 for this component. The Confidence Index above is effectively 0.70 × the player component until then."
                  />
                )}
                {phase3.conf_shoot_bias != null && (
                  <StatRow
                    label="Shoot Bias"
                    value={phase3.conf_shoot_bias.toFixed(3)}
                    tier={shootTier}
                    sub="Multiplier on shoot-vs-pass probability · 1.000 = neutral"
                    tip="Phase 17.25 — confidence rating multiplier on shooting decisions. ≈1.0 = no effect. >1.0 = more likely to shoot in a contested moment."
                  />
                )}
                {phase3.conf_risk_bias != null && (
                  <StatRow
                    label="Risk Bias"
                    value={phase3.conf_risk_bias.toFixed(3)}
                    tier={riskTier}
                    sub="Multiplier on aggressive plays (pinch, forecheck) · 1.000 = neutral"
                    tip="Phase 17.25 — risk-taking knob. >1.0 = D-men more likely to pinch, forwards more aggressive on the forecheck."
                  />
                )}
                {phase3.conf_turnover_bias != null && (
                  <StatRow
                    label="Turnover Bias"
                    value={phase3.conf_turnover_bias.toFixed(3)}
                    tier={turnTier}
                    sub="Multiplier on turnover probability · 1.000 = neutral"
                    tip="Phase 17.25 — confident players take more risks (aggressive passes, holds, plays through traffic) and turn the puck over more. >1.0 = more giveaways expected from the aggressive plays they're attempting. Scales the same direction as shoot_bias and risk_bias."
                  />
                )}
              </div>

              {sortedComps.length > 0 && (
                <div className="mt-4 pt-4 border-t border-white/[0.05] space-y-2">
                  <div className="flex items-baseline justify-between">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-white/40">
                      Top Contributors
                    </p>
                    <p className="text-[8px] font-mono text-white/25">
                      bar saturates at ±0.15
                    </p>
                  </div>
                  {sortedComps.map(([k, v]) => {
                    const isTeam = k.startsWith("team:");
                    const label = k.replace(/^team:/, "").replace(/_/g, " ");
                    const color = v >= 0 ? "#4ade80" : "#f87171";
                    return (
                      <div key={k} className="flex items-center gap-2">
                        <span className={`text-[10px] w-32 shrink-0 truncate ${isTeam ? "text-white/40" : "text-white/55"}`}>
                          {isTeam ? `(team) ${label}` : label}
                        </span>
                        <div className="flex-1 h-1.5 rounded-full bg-white/[0.04] overflow-hidden relative">
                          <div
                            className="h-full"
                            style={{
                              width: `${Math.min(100, (Math.abs(v) / 0.15) * 100)}%`,
                              backgroundColor: color,
                              opacity: 0.85,
                            }}
                          />
                          {/* 50% midpoint tick */}
                          <div className="absolute top-0 bottom-0 w-px bg-white/[0.10]" style={{ left: "50%" }} />
                        </div>
                        <span className="text-[10px] font-mono text-white/65 w-14 text-right shrink-0">
                          {v >= 0 ? "+" : ""}{v.toFixed(3)}
                        </span>
                      </div>
                    );
                  })}
                  {/* Scale legend */}
                  <div className="flex items-center gap-2 pt-1">
                    <span className="w-32 shrink-0" />
                    <div className="flex-1 flex justify-between text-[8px] font-mono text-white/25">
                      <span>0</span>
                      <span>0.075</span>
                      <span>0.15+</span>
                    </div>
                    <span className="w-14 shrink-0" />
                  </div>
                </div>
              )}
            </Card>
          );
        })()}

        {/* Goalie Fatigue (3.24) — goalies only */}
        {isGoalie && telemetryTab === "fatigue" && phase3?.goalie_fi != null && (() => {
          const gfi      = phase3.goalie_fi ?? 0;
          const gfiTier  = fatigueTier(gfi);
          const svDelta  = phase3.goalie_sv_delta ?? 0;
          const svDeltaT: Tier = svDelta >= 0 ? "Average"
                              : svDelta >= -0.005 ? "Below Average"
                              : "Low";
          const svPct    = (svDelta * 100).toFixed(2);
          return (
            <Card title="Fatigue & Schedule" style={cardStyle}>
              <div className="space-y-0">
                <StatRow
                  label="Goalie Fatigue Index"
                  value={gfi.toFixed(3)}
                  tier={gfiTier}
                  sub={phase3.goalie_fi_date
                    ? `Latest start: ${phase3.goalie_fi_date} · 0 = rested, 1 = saturated`
                    : "0 = rested, 1 = saturated"}
                  tip="Feature 3.24 — daily goalie FI snapshot. Built from B2B starts, rest days, games in last 7 days, shots faced. Lower is better — tier color is inverted so green = rested."
                />
                <StatRow
                  label="Expected Save% Δ"
                  value={`${svDelta >= 0 ? "+" : ""}${svPct}%`}
                  tier={svDeltaT}
                  sub="Predicted save% drift from fatigue (negative = degraded)"
                  tip="GoalieFatigueModel (2.6) coefficient sum on the goalie's current workload window. Negative = expected save% drop tonight."
                />
                {phase3.goalie_is_b2b === 1 && (
                  <StatRow
                    label="Back-to-Back"
                    value="Yes"
                    tier="Low"
                    sub="Starting on 0-1 days rest"
                    tip="B2B starts carry the largest single fatigue penalty in the goalie model."
                  />
                )}
                {phase3.goalie_rest_days != null && (
                  <StatRow
                    label="Rest Days"
                    value={`${phase3.goalie_rest_days.toFixed(0)}`}
                    sub="Calendar days since last start"
                    tip="Each rest day adds back ≈ +0.12% save% in the default model."
                  />
                )}
                {phase3.goalie_gp_last_7 != null && (
                  <StatRow
                    label="Starts (last 7d)"
                    value={`${phase3.goalie_gp_last_7}`}
                    sub="Rolling workload count"
                    tip="Each extra start in the prior 7 days subtracts ≈ −0.35% save%."
                  />
                )}
                {phase3.goalie_shots_last_7 != null && (
                  <StatRow
                    label="Shots Faced (last 7d)"
                    value={`${phase3.goalie_shots_last_7}`}
                    sub="Volume toll on the body"
                    tip="High shot volume in the last week proxies for cumulative physical load on the goalie."
                  />
                )}
              </div>
            </Card>
          );
        })()}

        {/* Behavioral NN + Skating */}
        {!isGoalie && telemetryTab === "neural" && (
          (data.nn_carry_in_pct != null || data.skating_avg_speed_kmh != null) && (
          <Card title="Play Style (Neural Network)" style={cardStyle}>
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

        {/* Goalie Performance Snapshot — radar with save% / GSAx axes. */}
        {isGoalie && telemetryTab === "neural" && (
          <div className="sm:col-span-2">
            <Card title="Performance Snapshot" style={cardStyle}>
              <div className="flex flex-wrap justify-center gap-6">
                <div className="flex flex-col items-center w-full sm:w-auto overflow-x-auto" style={{ scrollbarWidth: "none" }}>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-white/30 mb-2 text-center">Goalie Radar</p>
                  <GoalieRadarChart data={data} teamColor={teamColor} nhlSvPct={nhlStats?.sv_pct} nhlGaa={nhlStats?.gaa} />
                </div>
              </div>
            </Card>
          </div>
        )}

        {isGoalie && telemetryTab === "neural" && (
          <div className="sm:col-span-2">
            <Card title="Goalie Profile" style={cardStyle}>
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
                    tier={xgaTier(data.xga)}
                    sub="Workload indicator — how many goals an average goalie would allow given the same shots"
                    tip="The number of goals an average NHL goalie would be expected to allow given the exact same shots this goalie faced. Higher = heavier starter workload. The difference between xGA and actual goals against = GSAx."
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

        {/* Goalie — shots against heat map */}
        {isGoalie && telemetryTab === "shot-map" && goalieShots.length > 0 && (
          <div className="sm:col-span-2">
            <Card title="Shots Against" style={cardStyle}>
              <p className="text-[9px] uppercase tracking-wider text-center mb-3" style={{ color: "rgba(255,255,255,0.35)" }}>
                Arena-adjusted · last 2 seasons · {goalieShots.length} shots · {goalieShots.filter(s => s.goal).length} goals allowed
              </p>
              <Shot3D
                shots={goalieShots3D}
                themeColor={teamColor}
                goalie
                fallback={<GoalieShotMapViz shots={goalieShots} />}
              />
            </Card>
          </div>
        )}

        {/* Goalie — save% by zone. NET heatmap LEFT, SHOT-LOCATION map RIGHT
            so the goalie zones tab fills the page width instead of stacking
            two narrow viz columns. Falls back to single-col when shot
            sample is too thin for the rink heatmap. */}
        {/* Goalie Zones tab — focused on the rink-based shot-location
            heatmap (the deep-dive view). The NET HEATMAP summary already
            lives in the TARGET PROFILE band up top, so we don't duplicate
            it here. Falls back to a tiny info note when shot sample is
            too thin for a meaningful rink-zone breakdown. */}
        {isGoalie && telemetryTab === "zones" && (
          <div className="sm:col-span-2">
            <Card title="Shot-Location Save %" style={cardStyle}>
              {goalieShots.length >= 60 ? (
                <div className="grid gap-4 lg:grid-cols-[1.1fr_1fr] items-start">
                  <div className="min-w-0">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="hud-mono text-[9px] uppercase tracking-[0.22em]" style={{ color: teamColor }}>
                        ◢ SAVE % · BY SHOT ORIGIN
                      </span>
                      <span className="hud-mono text-[8px] uppercase tracking-[0.18em] text-[var(--text-muted)]">
                        rink-zone heatmap
                      </span>
                    </div>
                    <GoalieSaveLocationMap shots={goalieShots} teamColor={teamColor} />
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="hud-mono text-[9px] uppercase tracking-[0.22em]" style={{ color: teamColor }}>
                        ◢ NET-MOUTH HEATMAP
                      </span>
                      <span className="hud-mono text-[8px] uppercase tracking-[0.18em] text-[var(--text-muted)]">
                        save % · 5-zone
                      </span>
                    </div>
                    <GoalieZoneViz data={data} teamColor={teamColor} />
                  </div>
                </div>
              ) : (
                <div className="py-6 text-center hud-mono text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">
                  not enough shot data for the rink heatmap yet · check the TARGET PROFILE for the net-mouth view
                </div>
              )}
            </Card>
          </div>
        )}

        {/* Goalie — neural net: shot-type + zone tendencies */}
        {isGoalie && telemetryTab === "neural" && goalieNetData && (
          <div className="sm:col-span-2">
            <Card title="Shot Tendency Analysis" style={cardStyle}>
              <p className="text-[9px] text-white/25 uppercase tracking-wider text-center mb-4">
                {goalieNetData.total_shots} shots faced · {goalieNetData.goals_allowed} goals · sv {goalieNetData.overall_sv_pct != null ? `${(goalieNetData.overall_sv_pct * 100).toFixed(1)}%` : "—"} · last 2 seasons
              </p>
              <GoalieNeuralNetViz netData={goalieNetData} />
            </Card>
          </div>
        )}

        {/* Line Pairs (chemistry) */}
        {!isGoalie && telemetryTab === "advanced" && data.line_pairs && data.line_pairs.length > 0 && (
          <div className="sm:col-span-2">
            <Card title="Best Linemates" style={cardStyle}>
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
