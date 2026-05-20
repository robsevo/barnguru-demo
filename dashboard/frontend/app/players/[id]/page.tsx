"use client";

import { useEffect, useRef, useState } from "react";
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
  // HUD-styled card chrome: corner brackets, mono title bar, optional scan
  // line. Preserves the legacy `style` prop so callers passing team-tinted
  // cardStyle keep their brand glow.
  return (
    <div className={`hud-panel hud-panel--all-corners ${className}`} style={style}>
      <span className="hud-panel__corner-tr" />
      <span className="hud-panel__corner-bl" />
      <div className="px-3 py-2 border-b border-white/[0.05] flex items-center gap-2">
        <span className="hud-mono text-[10px] uppercase tracking-[0.18em] text-[var(--brand-hex)] opacity-80 select-none" aria-hidden>
          ◢
        </span>
        {icon ? <span className="text-[11px] opacity-70" aria-hidden>{icon}</span> : null}
        <p className="hud-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--text-primary)] truncate">
          {title}
        </p>
        <span className="ml-auto hud-mono text-[10px] uppercase tracking-[0.18em] text-[var(--brand-hex)] opacity-80 select-none" aria-hidden>
          ◣
        </span>
      </div>
      <div className="p-3 sm:p-4">{children}</div>
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
        {/* Bottom rail (ice line) */}
        <rect x="10" y="76" width="100" height="2.5" fill="url(#goalPost)" stroke="#600" strokeWidth="0.3" rx="0.5" />

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
function IceTimeByZoneBars({ data }: { data: ProfileData }) {
  const oz = data.skating_zone_time_oz_pct ?? 0;
  const dz = data.skating_zone_time_dz_pct ?? 0;
  const nz = Math.max(0, 100 - oz - dz);

  const bars = [
    { label: "OZ", pct: oz, color: "#4ade80", tip: "Offensive zone" },
    { label: "NZ", pct: nz, color: "#fbbf24", tip: "Neutral zone" },
    { label: "DZ", pct: dz, color: "#f87171", tip: "Defensive zone" },
  ];

  return (
    <div className="w-full flex flex-col gap-2.5 px-1">
      <div className="flex items-center justify-between mb-1">
        <span className="hud-mono text-[9px] uppercase tracking-[0.22em] text-[var(--text-secondary)]">
          ◢ ICE TIME · ZONES
        </span>
        <span className="hud-mono text-[8px] uppercase tracking-[0.18em] text-[var(--text-muted)]">
          % SHIFT
        </span>
      </div>
      {bars.map((b, idx) => (
        <div key={b.label} className="flex items-center gap-2 group" title={b.tip}>
          <span className="hud-mono text-[10px] uppercase tracking-[0.18em] w-7 shrink-0" style={{ color: b.color }}>
            {b.label}
          </span>
          <div
            className="relative flex-1 h-2.5 overflow-hidden rounded-sm"
            style={{
              background: "rgba(255,255,255,0.04)",
              border: `1px solid ${b.color}22`,
            }}
          >
            {/* Animated bar fill */}
            <div
              className="h-full"
              style={{
                width: `${b.pct}%`,
                background: `linear-gradient(90deg, ${b.color}aa 0%, ${b.color} 100%)`,
                boxShadow: `0 0 8px ${b.color}55, inset 0 0 8px ${b.color}44`,
                transformOrigin: "left center",
                animation: `iceBarFill 900ms cubic-bezier(0.22,1,0.36,1) ${idx * 100}ms both`,
              }}
            />
            {/* Sweeping scan-line */}
            <div
              className="absolute top-0 bottom-0 w-6 pointer-events-none"
              style={{
                background: `linear-gradient(90deg, transparent, ${b.color}aa, transparent)`,
                animation: `iceBarScan 3.6s linear infinite ${idx * 0.4}s`,
                mixBlendMode: "screen",
                opacity: b.pct > 5 ? 0.7 : 0,
              }}
            />
            {/* Tick marks at 25/50/75 */}
            <div className="absolute inset-0 flex justify-between px-[25%] pointer-events-none">
              <span className="w-px h-full" style={{ background: "rgba(255,255,255,0.08)" }} />
              <span className="w-px h-full" style={{ background: "rgba(255,255,255,0.12)" }} />
              <span className="w-px h-full" style={{ background: "rgba(255,255,255,0.08)" }} />
            </div>
          </div>
          <span className="hud-mono text-[11px] tabular-nums font-semibold w-10 text-right shrink-0" style={{ color: b.color }}>
            {b.pct.toFixed(0)}%
          </span>
        </div>
      ))}
      <style jsx>{`
        @keyframes iceBarFill {
          from { transform: scaleX(0); opacity: 0; }
          to   { transform: scaleX(1); opacity: 1; }
        }
        @keyframes iceBarScan {
          0%   { transform: translateX(-30px); }
          100% { transform: translateX(420px); }
        }
        @media (prefers-reduced-motion: reduce) {
          div { animation: none !important; }
        }
      `}</style>
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
}

function HologramScanner({
  isGoalie,
  teamColor,
  bodyIntensity,
  telemetryLeft,
  telemetryRight,
  tickerLine,
}: {
  isGoalie: boolean;
  teamColor: string;
  bodyIntensity: Partial<Record<BodyZone, number>>;
  telemetryLeft: HoloCallout[];
  telemetryRight: HoloCallout[];
  tickerLine: string[];
}) {
  const [active, setActive] = useState<HoloZone | null>(null);

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

        {/* Corner reticles */}
        <span aria-hidden className="absolute top-1 left-2 hud-mono text-[8px] uppercase tracking-[0.18em]" style={{ color: teamColor }}>◢ SCAN</span>
        <span aria-hidden className="absolute top-1 right-2 hud-mono text-[8px] uppercase tracking-[0.18em]" style={{ color: teamColor }}>LOCK ◣</span>
        <span aria-hidden className="absolute bottom-1 left-2 hud-mono text-[8px] uppercase tracking-[0.18em]" style={{ color: teamColor }}>◤ INTEL</span>
        <span aria-hidden className="absolute bottom-1 right-2 hud-mono text-[8px] uppercase tracking-[0.18em]" style={{ color: teamColor }}>LIVE ◥</span>

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
          {/* Zone hotspot dots — clickable */}
          {hotspots.map((h) => {
            const isActive = active === h.zone;
            const hotColor = h.intensity >= 0.5 ? "#f87171" : h.intensity >= 0.25 ? "#fbbf24" : teamColor;
            return (
              <button
                key={h.key}
                type="button"
                aria-label={`Focus ${h.zone}`}
                onMouseEnter={() => setActive(h.zone)}
                onMouseLeave={() => setActive(null)}
                onClick={() => setActive(a => a === h.zone ? null : h.zone)}
                className="absolute rounded-full cursor-pointer"
                style={{
                  left: h.cx - 6,
                  top:  h.cy - 6,
                  width: 12,
                  height: 12,
                  background: hotColor,
                  border: `1px solid ${hotColor}`,
                  boxShadow: `0 0 ${isActive ? 16 : 8}px ${hotColor}`,
                  transform: isActive ? "scale(1.6)" : undefined,
                  transition: "transform 200ms ease, box-shadow 200ms ease",
                  animation: `holoNode ${1.8 + (h.key.length % 3) * 0.4}s ease-in-out infinite`,
                  zIndex: 5,
                }}
              />
            );
          })}
        </div>

        {/* LEFT telemetry callouts — 3 only, hoverable, highlights target zone */}
        {telemetryLeft.length > 0 && (
          <div className="absolute left-2 top-12 bottom-10 flex flex-col justify-around items-end gap-2 z-10">
            {telemetryLeft.map((c) => {
              const isActive = active === c.target;
              return (
                <button
                  key={c.id}
                  type="button"
                  onMouseEnter={() => setActive(c.target)}
                  onMouseLeave={() => setActive(null)}
                  className="flex flex-col items-end gap-0.5 px-2 py-1 rounded backdrop-blur transition-all duration-200 group cursor-pointer"
                  style={{
                    background: isActive ? `${teamColor}1a` : "rgba(0,0,0,0.40)",
                    border: `1px solid ${isActive ? teamColor : `${teamColor}28`}`,
                    boxShadow: isActive ? `0 0 14px ${teamColor}55` : "none",
                  }}
                >
                  <span className="hud-mono text-[8px] uppercase tracking-[0.18em]"
                    style={{ color: isActive ? teamColor : "var(--text-secondary)" }}>
                    ▸ {c.label}
                  </span>
                  <span className="hud-mono text-[11px] tabular-nums font-semibold" style={{ color: teamColor }}>
                    {c.val}
                  </span>
                </button>
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
                <button
                  key={c.id}
                  type="button"
                  onMouseEnter={() => setActive(c.target)}
                  onMouseLeave={() => setActive(null)}
                  className="flex flex-col items-start gap-0.5 px-2 py-1 rounded backdrop-blur transition-all duration-200 cursor-pointer"
                  style={{
                    background: isActive ? `${teamColor}1a` : "rgba(0,0,0,0.40)",
                    border: `1px solid ${isActive ? teamColor : `${teamColor}28`}`,
                    boxShadow: isActive ? `0 0 14px ${teamColor}55` : "none",
                  }}
                >
                  <span className="hud-mono text-[8px] uppercase tracking-[0.18em]"
                    style={{ color: isActive ? teamColor : "var(--text-secondary)" }}>
                    {c.label} ◂
                  </span>
                  <span className="hud-mono text-[11px] tabular-nums font-semibold" style={{ color: teamColor }}>
                    {c.val}
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>

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

function PredictedPlay({
  carry, dump, slot, perim, drive, battleC, holdC, themeColor,
}: {
  carry?: number | null;
  dump?: number | null;
  slot?: number | null;
  perim?: number | null;
  drive?: number | null;
  battleC?: number | null;
  holdC?: number | null;
  themeColor: string;
}) {
  const entryActions: PredAction[] = [
    { id: "carry", label: "Carry-in", pct: carry ?? 0 },
    { id: "dump",  label: "Dump-in",  pct: dump  ?? 0 },
  ].filter(a => a.pct > 0).sort((a, b) => b.pct - a.pct);
  const inZoneActions: PredAction[] = [
    { id: "slot",   label: "Slot Shot",     pct: slot    ?? 0 },
    { id: "drive",  label: "Drive Net",     pct: drive   ?? 0 },
    { id: "perim",  label: "Perimeter",     pct: perim   ?? 0 },
    { id: "battle", label: "Battle Corner", pct: battleC ?? 0 },
    { id: "hold",   label: "Hold Corner",   pct: holdC   ?? 0 },
  ].filter(a => a.pct > 0).sort((a, b) => b.pct - a.pct);

  if (entryActions.length === 0 && inZoneActions.length === 0) return null;

  const topEntry  = entryActions[0]  ?? null;
  const topInZone = inZoneActions[0] ?? null;
  // Top 3 in-zone actions for the ranked list under the schematic
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
    if (id === "slot")   return `M 140 90  L ${SLOT.cx} ${SLOT.cy} L ${NET.cx} ${NET.y - 4}`;
    if (id === "drive")  return `M 140 80  Q 145 110 ${NET.cx} ${NET.y - 4}`;
    if (id === "perim")  return `M 140 80  L ${PERIM_R.cx} ${PERIM_R.cy} L ${NET.cx + 4} ${NET.y - 4}`;
    if (id === "battle") return `M 140 80  Q 60 120 ${CORNER_L.cx + 5} ${CORNER_L.cy - 5} L ${SLOT.cx - 10} ${SLOT.cy + 5}`;
    /* hold */            return `M 140 80  Q 220 120 ${CORNER_R.cx - 5} ${CORNER_R.cy - 5} Q 245 175 230 165`;
  }

  // ── Sequence label ───────────────────────────────────────────────────
  const sequenceLabel = [topEntry?.label, topInZone?.label]
    .filter(Boolean).join(" → ");

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
          <marker id="ppArrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill={themeColor} />
          </marker>
          <marker id="ppArrowGhost" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill={themeColor} fillOpacity="0.45" />
          </marker>
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

        {/* Ghosted secondary in-zone paths (rank 2, 3) */}
        {inZoneActions.slice(1, 3).map((a, i) => (
          <path key={a.id}
            d={inZonePath(a.id)}
            fill="none"
            stroke={themeColor}
            strokeOpacity={0.22 - i * 0.06}
            strokeWidth="1.4"
            strokeDasharray="3 5"
            markerEnd="url(#ppArrowGhost)" />
        ))}

        {/* Primary entry path */}
        {topEntry && (
          <path
            d={entryPath(topEntry.id)}
            fill="none"
            stroke={themeColor}
            strokeWidth="2.2"
            strokeOpacity="0.85"
            strokeLinecap="round"
            markerEnd="url(#ppArrow)"
            style={{
              filter: `drop-shadow(0 0 4px ${themeColor})`,
              strokeDasharray: 320,
              strokeDashoffset: 320,
              animation: "ppDraw 1100ms ease-out 80ms forwards",
            }} />
        )}
        {/* Primary in-zone path */}
        {topInZone && (
          <path
            d={inZonePath(topInZone.id)}
            fill="none"
            stroke={themeColor}
            strokeWidth="2.2"
            strokeOpacity="0.95"
            strokeLinecap="round"
            markerEnd="url(#ppArrow)"
            style={{
              filter: `drop-shadow(0 0 5px ${themeColor})`,
              strokeDasharray: 240,
              strokeDashoffset: 240,
              animation: "ppDraw 1100ms ease-out 900ms forwards",
            }} />
        )}

        {/* Origin marker — small triangle at the top of the zone */}
        <polygon points={`${140 - 4},-3 ${140 + 4},-3 140,7`} fill={themeColor} opacity="0.85" />

        <style>{`
          @keyframes ppDraw { to { stroke-dashoffset: 0; } }
          @media (prefers-reduced-motion: reduce) {
            path { animation: none !important; stroke-dashoffset: 0 !important; }
          }
        `}</style>
      </svg>

      {sequenceLabel && (
        <div className="mt-1.5 px-1 flex items-center gap-2 flex-wrap">
          <span className="hud-mono text-[9px] uppercase tracking-[0.18em] text-[var(--text-secondary)]">SEQ ·</span>
          <span className="hud-mono text-[10px] uppercase tracking-[0.18em] font-semibold"
            style={{ color: themeColor, textShadow: `0 0 6px ${themeColor}55` }}>
            {sequenceLabel}
          </span>
        </div>
      )}

      {/* Top-3 ranked in-zone decisions */}
      {top3.length > 0 && (
        <div className="mt-2 pt-2 border-t border-white/[0.05] space-y-1">
          {top3.map((a, i) => (
            <div key={a.id} className="flex items-center gap-2">
              <span className="hud-mono text-[8px] uppercase tracking-[0.16em] w-3 text-right shrink-0"
                style={{ color: i === 0 ? themeColor : "rgba(255,255,255,0.30)" }}>
                {i + 1}
              </span>
              <span className="hud-mono text-[9px] uppercase tracking-[0.14em] text-white/65 w-20 shrink-0 truncate">
                {a.label}
              </span>
              <div className="flex-1 h-1.5 rounded-sm overflow-hidden relative"
                style={{ background: "rgba(255,255,255,0.04)", border: `1px solid ${themeColor}22` }}>
                <div className="h-full" style={{
                  width: `${Math.min(100, a.pct)}%`,
                  background: `linear-gradient(90deg, ${themeColor}aa 0%, ${themeColor} 100%)`,
                  boxShadow: `0 0 6px ${themeColor}55`,
                  animation: `ppBar 900ms cubic-bezier(0.22,1,0.36,1) ${i * 90}ms backwards`,
                  transformOrigin: "left center",
                }} />
              </div>
              <span className="hud-mono text-[10px] tabular-nums w-10 text-right font-semibold"
                style={{ color: themeColor }}>
                {a.pct.toFixed(1)}%
              </span>
            </div>
          ))}
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
  type TelemetryTab = "shot-map" | "zones" | "games" | "special-teams" | "fatigue" | "advanced" | "neural";
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
  const telemetryTabs: HudTab[] = isGoalie
    ? [
        { id: "neural",   label: "Neural" },
        { id: "shot-map", label: "Shots Against" },
        { id: "zones",    label: "Zones" },
        { id: "games",    label: "Recent" },
        { id: "fatigue",  label: "Fatigue" },
      ]
    : [
        { id: "neural",        label: "Neural" },
        { id: "shot-map",      label: "Shot Map" },
        { id: "zones",         label: "Zones" },
        { id: "games",         label: "Recent" },
        { id: "special-teams", label: "Special" },
        { id: "fatigue",       label: "Fatigue" },
        { id: "advanced",      label: "Advanced" },
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

        {/* ── Compact horizontal hero — headshot left, identity stack right ── */}
        <div className={`flex items-center gap-4 sm:gap-5 px-4 sm:px-5 pb-3 ${data.hero_image ? "-mt-12 relative z-10" : "pt-4"}`}>

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
                <RingGauge value={fi} label="Fatigue" sublabel="FI" themeColor={teamColor} invert decimals={2} size={96} />
              )}
              {ci != null && (
                <RingGauge value={Math.min(1, (ci + 0.1) / 0.2)} centerText={`${ci >= 0 ? "+" : ""}${ci.toFixed(2)}`} label="Confidence" sublabel="CI" themeColor={teamColor} size={96} />
              )}
              {hhs != null && (
                <RingGauge value={Math.min(1, Math.max(0, (hhs + 2) / 4))} centerText={`${hhs >= 0 ? "+" : ""}${hhs.toFixed(1)}`} label="Hot Hand" sublabel="HHS" themeColor={teamColor} size={96} />
              )}
              {warVal != null && (
                <RingGauge
                  value={warRankPct ?? Math.min(1, Math.max(0, (warVal + 1) / 4))}
                  centerText={`${warVal >= 0 ? "+" : ""}${warVal.toFixed(2)}`}
                  label="WAR"
                  sublabel={data.war_rank ? `#${data.war_rank}` : "rating"}
                  themeColor={teamColor}
                  size={96}
                />
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
              telemetryLeft={!isGoalie ? [
                { id: "speed",    label: "MAX SPEED",  val: data.skating_max_speed_kmh != null ? `${data.skating_max_speed_kmh.toFixed(1)} km/h` : null, target: "legs" as const },
                { id: "hits",     label: "HITS/60",    val: data.hits_per60 != null ? data.hits_per60.toFixed(1) : null, target: "arms" as const },
                { id: "blocks",   label: "BLOCKS/60",  val: data.blocks_per60 != null ? data.blocks_per60.toFixed(1) : null, target: "torso" as const },
              ].filter(c => c.val !== null) : []}
              telemetryRight={!isGoalie ? [
                { id: "battle",   label: "BATTLE",     val: data.battle_percentile != null ? `${data.battle_percentile.toFixed(0)}th` : null, target: "torso" as const },
                { id: "toi",      label: "EV TOI",     val: data.toi_ev != null ? `${data.toi_ev.toFixed(0)}m` : null, target: "head" as const },
                { id: "edge",     label: "EDGE Δ",     val: phase3?.edge_load != null ? `${phase3.edge_load >= 0 ? "+" : ""}${(phase3.edge_load * 100).toFixed(1)}%` : null, target: "legs" as const },
              ].filter(c => c.val !== null) : []}
              tickerLine={[
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
                    <line x1={150} y1={100} x2={150} y2={20}
                      stroke={teamColor} strokeOpacity={0.55} strokeWidth={1.5}
                      style={{
                        transformOrigin: "150px 100px",
                        animation: "ncRadar 6s linear infinite",
                        filter: `drop-shadow(0 0 4px ${teamColor})`,
                      }} />
                  </svg>
                  <NeuralGraph
                    center={isGoalie ? "G" : "NN"}
                    nodes={activeNodes}
                    themeColor={teamColor}
                    width={300}
                    height={200}
                  />
                  <style jsx>{`
                    @keyframes ncRadar { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
                    @media (prefers-reduced-motion: reduce) { line { animation: none !important; } }
                  `}</style>
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
                {matrixNodes.slice(0, 6).map((n, i) => (
                  <div key={n.id} className="flex items-center gap-2">
                    <span className="hud-mono text-[9px] uppercase tracking-[0.14em] text-[var(--text-secondary)] w-16 shrink-0 truncate">
                      {n.label}
                    </span>
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
                    <span className="hud-mono text-[10px] tabular-nums w-10 text-right font-semibold" style={{ color: teamColor }}>
                      {n.weight.toFixed(2)}
                    </span>
                  </div>
                ))}
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

      {/* ── Telemetry tab bar — switches the cards below ── */}
      <div className="mt-5 mb-2 px-1 flex items-center gap-2"
        style={{ borderBottom: `1px solid ${teamColor}22` }}>
        <span className="hud-mono text-[9px] uppercase tracking-[0.2em] text-[var(--text-secondary)] shrink-0 pr-2">
          ▌ TELEMETRY
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
        {(telemetryTab === "advanced" || telemetryTab === "fatigue" || telemetryTab === "special-teams") && (
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

        {/* ── Shot Map card — 3D + 2D toggle ── */}
        {!isGoalie && telemetryTab === "shot-map" && (
          <div className="sm:col-span-2">
            <Card title="Shot Map" icon="🎯" style={cardStyle}>
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

        {/* ── Zone Tendencies — stacked, centered ── */}
        {!isGoalie && telemetryTab === "zones" && (
          <div className="sm:col-span-2">
            <Card title="Zone Tendencies" style={cardStyle}>
              <div className="flex flex-col items-stretch gap-4 w-full">
                {/* Offensive Zone Tendency — 3D rink with 2D fallback */}
                {data.nn_shoot_slot_pct != null ? (
                  <>
                    <p className="hud-mono text-[9px] uppercase tracking-[0.22em] text-[var(--text-secondary)] text-center">
                      ◢ OFFENSIVE ZONE TENDENCY · 3D
                    </p>
                    <Zone3D
                      activations={{
                        slot:    data.nn_shoot_slot_pct ?? 0,
                        perim:   data.nn_shoot_perimeter_pct ?? 0,
                        net:     data.nn_drive_net_pct ?? 0,
                        cornerL: data.nn_battle_corner_pct ?? 0,
                        cornerR: data.nn_hold_corner_pct ?? 0,
                      }}
                      themeColor={teamColor}
                      fallback={
                        <div className="max-w-md mx-auto">
                          <ZoneTendencyMap data={data} teamColor={teamColor} />
                        </div>
                      }
                    />
                  </>
                ) : (
                  <div className="flex flex-col items-center justify-center py-6 gap-1.5">
                    <p className="hud-mono text-[9px] uppercase tracking-[0.18em] text-[var(--text-muted)]">Model not yet trained</p>
                  </div>
                )}
                {/* Ice Time By Zone bars */}
                {data.skating_zone_time_oz_pct != null ? (
                  <div className="w-full max-w-[420px] mx-auto">
                    <IceTimeByZoneBars data={data} />
                  </div>
                ) : null}
                <p className="hud-mono text-[8px] uppercase tracking-[0.18em] text-center text-[var(--text-muted)]">
                  drag the 3D rink to rotate · scroll to zoom
                </p>
              </div>
            </Card>
          </div>
        )}

        {/* Recent Games */}
        {telemetryTab === "games" && (
          <div className="sm:col-span-2">
            <Card title="Recent Games" icon="📅" style={cardStyle}>
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

        {/* Offensive Profile */}
        {!isGoalie && telemetryTab === "advanced" && (data.finishing != null || data.war != null || data.rapm_ev_off != null) && (
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
                  tier={toiTier(data.toi_ev)}
                  sub="5v5 even-strength minutes this season"
                  tip="Total even-strength (5v5) minutes played this season. More TOI = more trust from the coaching staff. All other metrics are measured per 60 to account for ice time differences." />
              )}
            </div>
          </Card>
        )}

        {/* Defensive Profile */}
        {!isGoalie && telemetryTab === "advanced" && (data.cdr != null || data.rapm_ev_def != null || data.battle_score != null) && (
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
        {!isGoalie && telemetryTab === "special-teams" && (data.special_teams_pp != null || data.special_teams_pk != null) && (
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
        {!isGoalie && telemetryTab === "neural" && (data.ewma_xgf60 != null || data.hot_hand_score != null || data.clutch_index != null) && (
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
        {!isGoalie && telemetryTab === "advanced" && (data.playoff_delta != null || data.former_team_boost != null || data.bayesian_rating != null) && (
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
            <Card title="Fatigue & Schedule" icon="😮‍💨" style={cardStyle}>
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
            <Card title="Confidence (Phase 17)" icon="📈" style={cardStyle}>
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
            <Card title="Fatigue & Schedule" icon="🥅" style={cardStyle}>
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
        {isGoalie && telemetryTab === "neural" && (
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

        {isGoalie && telemetryTab === "neural" && (
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

        {/* Goalie — save% by zone */}
        {isGoalie && telemetryTab === "zones" && (data.hdsv_pct != null || data.mdsv_pct != null || data.ldsv_pct != null) && (
          <div className="sm:col-span-2">
            <Card title="Save % by Zone" style={cardStyle}>
              <p className="text-[9px] text-white/25 uppercase tracking-wider text-center mb-3">
                Color coded vs league averages · green = above avg · red = below avg
              </p>
              <GoalieZoneViz data={data} teamColor={teamColor} />
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
        {!isGoalie && telemetryTab === "games" && data.line_pairs && data.line_pairs.length > 0 && (
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
