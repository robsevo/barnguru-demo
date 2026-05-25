"use client";

import { Fragment, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { logoUrl, TEAM_FULL_NAMES, TEAM_COLORS, TEAM_SECONDARY } from "@/utils/nhl";
import { HudGrid } from "@/components/hud";

// ---------------------------------------------------------------------------
// Types — mirror /coaches/{name} payload from dashboard/api/main.py
// ---------------------------------------------------------------------------

interface LineRow {
  line_type: string; line_rank: number;
  player_ids: number[]; player_names: string[];
  chemistry_toi_secs: number | null; trio_toi_per_game: number | null;
  line_toi_per_game: number | null; cohesion_pct: number | null;
  share_of_team_toi: number | null; team_gp: number;
}
interface MatchingRow {
  own_line_rank: number; opp_line_rank: number;
  venue: string; weighted_share: number; total_toi_secs: number;
}
interface StUnit {
  unit_type: string; player_ids: number[]; player_names: string[];
  unit_toi_secs: number | null; share_of_st_toi: number | null;
  team_st_toi: number | null; team_st_gp: number;
}
interface PullRow {
  deficit: number; n_pulls: number; n_team_games: number;
  mean_pull_time_secs: number | null; median_pull_time_secs: number | null;
  earliest_pull_secs: number | null;
}
interface PenaltyRow {
  n_games: number; n_penalties_taken: number; n_pp_opportunities: number;
  pim_total: number; penalties_taken_per_game: number;
  pp_opps_per_game: number; pim_per_game: number; ref_dim: string | null;
}
interface TimeoutRow {
  period_bucket: string; score_state: string; time_bucket: string;
  n_timeouts: number; n_games: number; rate_per_game: number;
}
interface CoachProfileRow {
  coach_name: string; team: string; first_named_head_coach: string | null;
  season: number; seasons_covered: number[];
  gp_under_coach: number; wins: number; ot_wins: number; losses: number; ot_losses: number;
  points: number; points_pct: number;
  gf_per_game: number; ga_per_game: number;
  pp_pct: number; pk_pct: number;
  sf_per_game: number; sa_per_game: number;
}
interface GoalieCoachRow {
  team: string; season: number; gp: number;
  shots_against: number; goals_against: number;
  season_save_pct: number | null; prior_save_pct: number | null; save_pct_delta: number | null;
  early_split_save_pct: number | null; late_split_save_pct: number | null;
  split_delta: number | null; change_point_detected: boolean;
  rolling_save_pct: number[]; goalie_coach: string;
}
interface PpCoordinatorRow {
  team: string; season: number;
  pp_toi_secs: number; pp_team_gp: number;
  pp_shots: number; pp_goals: number; pp_xg_total: number;
  pp_shots_per_60: number; pp_xg_per_60: number; pp_goals_per_60: number;
  pp_xg_per_shot: number; pp_shot_distance_avg: number;
  pp_carry_pct: number | null;
  pp1_qb_id: number | null; pp1_qb_name: string; pp1_qb_share: number;
  pp_coordinator: string;
}
interface PkCoordinatorRow {
  team: string; season: number; pk_toi_secs: number; pk_team_gp: number;
  pk_sa: number; pk_ga: number; pk_xga_total: number;
  pk_sa_per_60: number; pk_xga_per_60: number; pk_ga_per_60: number;
  pk_save_pct: number; pk_xga_per_shot: number; pk_shot_distance_avg: number;
  sh_shots_for: number; sh_goals_for: number; sh_shots_per_60: number;
  pk1_share: number; pk_coordinator: string;
}
interface CoachingStyleDim { raw: number | null; rank: number | null; }
interface CoachingStyleRow {
  team: string; season: number;
  dimensions: Record<string, CoachingStyleDim>;
}
interface RosterFitRow {
  team: string; season: number; n_skaters: number;
  archetype_top: string; archetypes: string[]; archetype_shares: number[];
  fit_score: number; mismatch_dim: string; mismatch_support: number;
}
interface StaffChangeRow {
  date: string; change_type: string; person_out: string; person_in: string;
  description: string; decay_games: number;
}
interface FoRegimeRow {
  date: string; fo_role: string; person_out: string; person_in: string;
  description: string; decay_games: number;
}
interface BuyerSellerRow {
  team: string; season: number; gp: number; points_pct: number;
  classification: "buyer" | "seller" | "neutral";
  confidence: number; gap: number; threshold: number;
}
interface SellerMotivationRow {
  team: string; seller_drag: number; efficiency_multiplier: number;
  games_since_deadline: number; contextual_flag: string;
}
interface CoachDecisionRow {
  coach_name: string; team: string;
  timeout_aggression: number; pull_aggression: number;
  line_shelter_score: number; st_first_unit_lean: number;
  penalty_discipline: number; matching_intensity: number;
  overall_aggression: number;
}
interface GmFingerprintRow {
  team: string; gm_name: string; action_archetype: string;
  prob_stand_pat: number; prob_add_rental: number; prob_sell_veteran: number;
  prob_rebuild: number; prob_package_deal: number;
  deadline_aggression: number; recent_tx_count: number;
}
interface VenueAtmosphereRow {
  team: string; home_gp: number;
  visiting_sv_delta: number; visiting_fow_delta: number;
  ref_pp_delta: number; visiting_xgf_delta: number;
  scare_factor: number; scare_rank: number;
}
interface PlayoffEliminationRow {
  team: string; playoff_prob: number;
  elimination_drag: number; efficiency_multiplier: number;
  games_remaining: number; points_pct: number;
}

interface CoachProfile {
  status: "ok" | "not_found";
  name?: string;
  meta?: { name: string; team: string; first_named_head_coach: string | null; notes: string };
  line_deployment?:  { rows: LineRow[]; as_of: string | null };
  line_matching?:    { F: MatchingRow[]; D: MatchingRow[]; as_of: string | null };
  st_deployment?:    { units: StUnit[]; as_of: string | null };
  goalie_pull?:      { rows: PullRow[]; as_of: string | null };
  penalty_tendency?: { row: PenaltyRow | null; league_avg: any; as_of: string | null };
  timeout_usage?:    { rows: TimeoutRow[]; as_of: string | null };
  coach_profile?:    { row: CoachProfileRow | null; as_of: string | null };
  goalie_coach?:     { row: GoalieCoachRow | null; as_of: string | null };
  pp_coordinator?:   { row: PpCoordinatorRow | null; as_of: string | null };
  pk_coordinator?:   { row: PkCoordinatorRow | null; as_of: string | null };
  coaching_style?:   { row: CoachingStyleRow | null; league_avg: any; as_of: string | null };
  roster_fit?:       { row: RosterFitRow | null; as_of: string | null };
  staff_changes?:    { rows: StaffChangeRow[]; as_of: string | null };
  fo_regime_changes?:{ rows: FoRegimeRow[]; as_of: string | null };
  buyer_seller?:     { row: BuyerSellerRow | null; as_of: string | null };
  seller_motivation?:{ row: SellerMotivationRow | null; as_of: string | null };
  coach_decision_net?:{ row: CoachDecisionRow | null; as_of: string | null };
  gm_fingerprint?:   { row: GmFingerprintRow | null; as_of: string | null };
  venue_atmosphere?: { row: VenueAtmosphereRow | null; as_of: string | null };
  playoff_elimination?:{ row: PlayoffEliminationRow | null; as_of: string | null };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function darkBlend(hex: string, darkness = 0.82): string {
  const c = hex.replace("#", "");
  const r = parseInt(c.slice(0, 2), 16);
  const g = parseInt(c.slice(2, 4), 16);
  const b = parseInt(c.slice(4, 6), 16);
  const f = 1 - darkness;
  return `rgb(${Math.round(r * f)}, ${Math.round(g * f)}, ${Math.round(b * f)})`;
}

function fmtMin(secs: number | null | undefined): string {
  if (secs == null || !Number.isFinite(secs)) return "—";
  return `${(secs / 60).toFixed(1)}m`;
}
function fmtSec(secs: number | null | undefined): string {
  if (secs == null || !Number.isFinite(secs)) return "—";
  return `${secs.toFixed(0)}s`;
}
function fmtPct(v: number | null | undefined, dec = 1): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(dec)}%`;
}
function fmtNum(v: number | null | undefined, dec = 2): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toFixed(dec);
}

// ---------------------------------------------------------------------------
// HUD primitives
// ---------------------------------------------------------------------------

function HudTitle({ title, subtitle, accent }: { title: string; subtitle?: string; accent: string }) {
  return (
    <div className="flex items-baseline gap-2 min-w-0">
      <span className="hud-mono opacity-80" style={{ color: accent }} aria-hidden>◢</span>
      <span className="hud-mono text-[10px] uppercase tracking-[0.18em] text-white/85">{title}</span>
      {subtitle && <span className="hud-mono text-[9px] uppercase tracking-[0.15em] text-white/40">· {subtitle}</span>}
      <span className="hud-mono opacity-80 ml-auto" style={{ color: accent }} aria-hidden>◣</span>
    </div>
  );
}

function HudCard({
  title, subtitle, right, accent, children, padding = "p-3"
}: {
  title?: string; subtitle?: string; right?: React.ReactNode; accent: string;
  children: React.ReactNode; padding?: string;
}) {
  return (
    <div className="hud-panel hud-panel--all-corners jarvis-boot jarvis-shimmer"
      style={{ ["--hud-corner" as string]: `${accent}aa` }}>
      <span className="hud-panel__corner-tr" />
      <span className="hud-panel__corner-bl" />
      {title && (
        <div className="px-3 pt-2 pb-1.5 flex items-center justify-between gap-2 border-b border-white/[0.06]">
          <HudTitle title={title} subtitle={subtitle} accent={accent} />
          {right}
        </div>
      )}
      <div className={padding}>{children}</div>
    </div>
  );
}

function StatPill({ label, value, color = "text-white" }: {
  label: string; value: React.ReactNode; color?: string;
}) {
  return (
    <div className="rounded border border-white/[0.08] bg-white/[0.02] px-2.5 py-1.5">
      <span className="text-[8px] font-mono uppercase tracking-[0.18em] text-white/30 block">{label}</span>
      <span className={`text-[13px] font-mono font-semibold ${color}`}>{value}</span>
    </div>
  );
}

function SkeletonBadge() {
  return (
    <span className="text-[8px] font-mono px-1.5 py-0.5 rounded border border-[#a78bfa]/30 bg-[#a78bfa]/[0.08] text-[#a78bfa]/80 uppercase tracking-wider">
      NOT TRAINED
    </span>
  );
}

// ---------------------------------------------------------------------------
// Charts (inline SVG)
// ---------------------------------------------------------------------------

function StyleRadar({ dims, accent }: { dims: Record<string, CoachingStyleDim>; accent: string }) {
  const labels = ["Forecheck", "DZ", "Pace", "Physical", "OZ", "NZ", "Match", "ST"];
  const keys = ["forecheck_aggression","dz_structure","pace","physicality","oz_structure","nz_tendency","line_match","st_aggression"];
  const vals = keys.map(k => {
    const v = dims?.[k]?.rank;
    return v != null && !Number.isNaN(v) ? v : 0.5;
  });

  const SIZE = 240;
  const cx = SIZE / 2, cy = SIZE / 2, R = 90;
  const n = labels.length;
  const angleFor = (i: number) => -Math.PI / 2 + (2 * Math.PI * i) / n;
  const points = vals.map((v, i) => {
    const a = angleFor(i);
    const r = v * R;
    return `${cx + Math.cos(a) * r},${cy + Math.sin(a) * r}`;
  }).join(" ");
  const rings = [0.25, 0.5, 0.75, 1.0];

  return (
    <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`} aria-hidden className="block">
      {rings.map((s, i) => (
        <circle key={i} cx={cx} cy={cy} r={R * s} fill="none"
          stroke={accent} strokeOpacity={0.06 + i * 0.03} strokeDasharray="2 6" strokeWidth={1} />
      ))}
      {labels.map((label, i) => {
        const a = angleFor(i);
        const x2 = cx + Math.cos(a) * R;
        const y2 = cy + Math.sin(a) * R;
        const lx = cx + Math.cos(a) * (R + 14);
        const ly = cy + Math.sin(a) * (R + 14);
        return (
          <g key={label}>
            <line x1={cx} y1={cy} x2={x2} y2={y2} stroke="rgba(255,255,255,0.05)" strokeWidth={0.8} />
            <text x={lx} y={ly} fontSize={8.5} fontFamily="monospace"
              fill="rgba(255,255,255,0.5)" textAnchor="middle" dominantBaseline="middle">
              {label}
            </text>
          </g>
        );
      })}
      <polygon points={points} fill={accent} fillOpacity={0.22} stroke={accent} strokeWidth={1.4}
        style={{ filter: `drop-shadow(0 0 6px ${accent}88)` }} />
      {vals.map((v, i) => {
        const a = angleFor(i);
        const r = v * R;
        return (
          <circle key={i} cx={cx + Math.cos(a) * r} cy={cy + Math.sin(a) * r}
            r={3} fill={accent} stroke="rgba(0,0,0,0.5)" strokeWidth={0.6} />
        );
      })}
    </svg>
  );
}

function Sparkline({ pts, accent, w = 320, h = 64, threshold = 0.900 }: {
  pts: number[]; accent: string; w?: number; h?: number; threshold?: number;
}) {
  if (pts.length < 2) return <div className="text-[10px] text-white/30 font-mono">insufficient data</div>;
  const P = 6;
  const lo = Math.min(threshold - 0.02, ...pts);
  const hi = Math.max(threshold + 0.02, ...pts);
  const span = hi - lo || 1;
  const xs = pts.map((_, i) => P + (i * (w - 2 * P)) / (pts.length - 1));
  const ys = pts.map(v => h - P - ((v - lo) / span) * (h - 2 * P));
  const d = xs.map((x, i) => `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${ys[i].toFixed(1)}`).join(" ");
  const dArea = `${d} L ${xs[xs.length - 1]} ${h - P} L ${xs[0]} ${h - P} Z`;
  const thrY = h - P - ((threshold - lo) / span) * (h - 2 * P);
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden className="block">
      <line x1={P} x2={w - P} y1={thrY} y2={thrY} stroke="rgba(255,255,255,0.10)" strokeDasharray="3 4" />
      <path d={dArea} fill={accent} fillOpacity={0.10} />
      <path d={d} fill="none" stroke={accent} strokeWidth={1.6}
        style={{ filter: `drop-shadow(0 0 3px ${accent})` }} />
      {xs.map((x, i) => (
        <circle key={i} cx={x} cy={ys[i]} r={2.2}
          fill={pts[i] >= threshold ? "#4ade80" : "#f87171"}
          stroke="rgba(0,0,0,0.5)" strokeWidth={0.5} />
      ))}
    </svg>
  );
}

function BarRow({ label, value, accent, suffix = "", max = 1 }: {
  label: string; value: number; accent: string; suffix?: string; max?: number;
}) {
  const pct = Math.max(0, Math.min(1, value / max));
  return (
    <div className="flex items-center gap-2 py-1">
      <span className="text-[10px] font-mono text-white/55 w-32 truncate">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-white/[0.04] border border-white/[0.04] overflow-hidden relative">
        <div className="h-full transition-all duration-700"
          style={{
            width: `${pct * 100}%`,
            background: `linear-gradient(90deg, ${accent}aa 0%, ${accent} 100%)`,
            boxShadow: `0 0 6px ${accent}55, inset 0 0 4px ${accent}33`,
          }} />
      </div>
      <span className="text-[10px] font-mono font-semibold w-12 text-right tabular-nums" style={{ color: accent }}>
        {value.toFixed(2)}{suffix}
      </span>
    </div>
  );
}

function Gauge({ value, max = 1, accent, w = 220, h = 90 }: {
  value: number; max?: number; accent: string; w?: number; h?: number;
}) {
  const pct = Math.max(0, Math.min(1, value / max));
  const cx = w / 2, cy = h - 8, r = h - 18;
  const start = Math.PI, end = 0;
  const angle = start + (end - start) * pct;
  const x1 = cx + Math.cos(start) * r;
  const y1 = cy + Math.sin(start) * r;
  const x2 = cx + Math.cos(end) * r;
  const y2 = cy + Math.sin(end) * r;
  const ax = cx + Math.cos(angle) * r;
  const ay = cy + Math.sin(angle) * r;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="block" aria-hidden>
      <path d={`M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`}
        fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={8} strokeLinecap="round" />
      <path d={`M ${x1} ${y1} A ${r} ${r} 0 0 1 ${ax} ${ay}`}
        fill="none" stroke={accent} strokeWidth={8} strokeLinecap="round"
        style={{ filter: `drop-shadow(0 0 6px ${accent})` }} />
      <text x={cx} y={cy - 12} textAnchor="middle" fontSize={20} fontFamily="monospace" fontWeight={700} fill={accent}>
        {value.toFixed(2)}
      </text>
      <text x={cx} y={cy + 6} textAnchor="middle" fontSize={8} fontFamily="monospace" fill="rgba(255,255,255,0.4)">
        / {max.toFixed(1)}
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Hero
// ---------------------------------------------------------------------------

function Hero({ meta, profile, teamColor, teamSecondary }: {
  meta: NonNullable<CoachProfile["meta"]>;
  profile: CoachProfile;
  teamColor: string;
  teamSecondary: string;
}) {
  const cp = profile.coach_profile?.row;
  const bs = profile.buyer_seller?.row;
  return (
    <div className="relative jarvis-shimmer mb-3 rounded-2xl border overflow-hidden hud-panel--all-corners"
      style={{
        ["--hud-corner" as string]: `${teamColor}aa`,
        borderColor: `${teamColor}50`,
        background: `linear-gradient(175deg, ${darkBlend(teamSecondary, 0.78)} 0%, #060708 65%)`,
        boxShadow: `0 4px 24px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.06), 0 0 24px ${teamColor}1a`,
      }}>
      <span className="hud-panel__corner-tr" />
      <span className="hud-panel__corner-bl" />

      {/* Iron Man rings backdrop */}
      <svg viewBox="0 0 700 260" aria-hidden
        className="absolute left-0 top-0 w-full h-full pointer-events-none opacity-55"
        style={{ zIndex: 0 }}>
        <g style={{ transformOrigin: "100px 110px", animation: "coachHeroRingSlow 30s linear infinite" }}>
          <circle cx={100} cy={110} r={92} fill="none" stroke={teamColor} strokeOpacity={0.22} strokeDasharray="2 8" />
          {[0, 45, 90, 135, 180, 225, 270, 315].map((deg, i) => {
            const rad = (deg * Math.PI) / 180;
            return (
              <line key={i}
                x1={100 + Math.cos(rad) * 86} y1={110 + Math.sin(rad) * 86}
                x2={100 + Math.cos(rad) * 100} y2={110 + Math.sin(rad) * 100}
                stroke={teamColor} strokeOpacity={0.55} strokeWidth={1.2} />
            );
          })}
        </g>
        <g style={{ transformOrigin: "100px 110px", animation: "coachHeroRingRev 18s linear infinite" }}>
          <circle cx={100} cy={110} r={74} fill="none" stroke={teamColor} strokeOpacity={0.40}
            strokeDasharray="30 20 10 20" strokeLinecap="round" strokeWidth={1.2} />
        </g>
        <line x1={210} y1={110} x2={690} y2={110} stroke={teamColor} strokeOpacity={0.10} strokeDasharray="3 10" />
      </svg>
      <style jsx>{`
        @keyframes coachHeroRingSlow { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes coachHeroRingRev  { from { transform: rotate(360deg); } to { transform: rotate(0deg); } }
      `}</style>

      {/* TARGET PROFILE bar */}
      <div className="relative px-4 py-1.5 flex items-center gap-2 border-b z-10"
        style={{ borderColor: `${teamColor}22`, background: `linear-gradient(90deg, ${teamColor}1a 0%, transparent 60%)` }}>
        <span className="hud-mono text-[9px] uppercase tracking-[0.20em]" style={{ color: teamColor }}>◢</span>
        <span className="hud-mono text-[9px] uppercase tracking-[0.20em]" style={{ color: teamColor }}>COACH PROFILE</span>
        <span className="hud-mono text-[8px] uppercase tracking-[0.16em] text-white/40">
          · {meta.team} · {TEAM_FULL_NAMES[meta.team] ?? meta.team}
        </span>
        <span className="ml-auto flex items-center gap-1.5">
          <span className="hud-pulse-dot" style={{ background: teamColor }} />
          <span className="hud-mono text-[8px] uppercase tracking-[0.18em] text-white/55">ACTIVE</span>
        </span>
      </div>

      {/* Content */}
      <div className="relative z-10 flex items-center gap-5 px-5 py-5">
        {/* Team logo as avatar with glow */}
        <a href={`/teams/${meta.team}`} className="shrink-0 relative">
          <div className="absolute inset-0 rounded-full pointer-events-none"
            style={{
              background: `radial-gradient(circle, ${teamColor}55 0%, ${teamColor}22 45%, transparent 72%)`,
              transform: "scale(1.5)",
              filter: "blur(14px)",
            }} />
          <div className="relative h-24 w-24 rounded-full overflow-hidden flex items-center justify-center"
            style={{
              background: `radial-gradient(circle at 40% 35%, ${darkBlend(teamSecondary, 0.50)} 0%, ${darkBlend(teamSecondary, 0.88)} 60%, #080a0c 100%)`,
              boxShadow: `0 0 0 3px ${teamColor}, 0 0 0 5px rgba(255,255,255,0.16), 0 0 30px ${teamColor}cc, 0 0 60px ${teamColor}55, 0 8px 32px rgba(0,0,0,0.7)`,
            }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={logoUrl(meta.team)} alt={meta.team} className="w-[72px] h-[72px] object-contain" />
          </div>
        </a>
        <div className="w-px shrink-0 bg-white/[0.10]" style={{ height: 88 }} />

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className="text-[9px] font-mono uppercase tracking-[0.20em]" style={{ color: teamColor }}>HEAD COACH</span>
            {bs && (
              <span className={`px-1.5 py-0.5 rounded text-[8px] font-mono font-semibold uppercase border ${
                bs.classification === "buyer" ? "border-[#4ade80]/40 text-[#4ade80] bg-[#4ade80]/[0.06]"
                : bs.classification === "seller" ? "border-[#f87171]/40 text-[#f87171] bg-[#f87171]/[0.06]"
                : "border-[#fbbf24]/40 text-[#fbbf24] bg-[#fbbf24]/[0.06]"
              }`}>
                {bs.classification} · conf {(bs.confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>
          <h1 className="text-[26px] sm:text-[30px] font-bold text-white tracking-tight leading-tight">
            {meta.name}
          </h1>
          <p className="text-[11px] font-mono text-white/45 mt-1">
            <a href={`/teams/${meta.team}`} className="hover:text-white">
              {TEAM_FULL_NAMES[meta.team] ?? meta.team}
            </a>
            {meta.first_named_head_coach && (
              <>
                <span className="text-white/20 mx-1.5">·</span>
                <span>since {meta.first_named_head_coach}</span>
              </>
            )}
          </p>
          {meta.notes && <p className="text-[10px] text-white/35 mt-1.5 italic max-w-2xl">{meta.notes}</p>}
        </div>

        {/* Right side: career headline stats */}
        {cp && (
          <div className="hidden lg:flex flex-col items-end gap-2 shrink-0">
            <div className="flex items-center gap-3">
              <div className="flex flex-col items-end">
                <span className="text-[8px] uppercase tracking-wider text-white/35 font-mono">Points %</span>
                <span className={`text-[22px] font-mono font-bold leading-none ${
                  cp.points_pct >= 0.55 ? "text-[#4ade80]" : cp.points_pct >= 0.45 ? "text-white" : "text-[#f87171]"
                }`}>
                  {(cp.points_pct * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex flex-col items-end">
                <span className="text-[8px] uppercase tracking-wider text-white/35 font-mono">Record</span>
                <span className="text-[13px] font-mono text-white/85 leading-none mt-1.5">
                  {cp.wins}-{cp.losses}-{cp.ot_losses}
                </span>
              </div>
              <div className="flex flex-col items-end">
                <span className="text-[8px] uppercase tracking-wider text-white/35 font-mono">GP</span>
                <span className="text-[13px] font-mono text-white/85 leading-none mt-1.5">{cp.gp_under_coach}</span>
              </div>
            </div>
            <p className="hud-mono text-[8px] uppercase tracking-[0.16em] text-white/30">
              seasons {cp.seasons_covered.join(", ")}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab bar
// ---------------------------------------------------------------------------

type Tab = "dossier" | "tactics" | "ingame" | "staff" | "identity" | "context";

function TabBar({ active, onChange, teamColor }: {
  active: Tab; onChange: (t: Tab) => void; teamColor: string;
}) {
  const tabs: { id: Tab; label: string; glyph: string; subtitle: string }[] = [
    { id: "dossier",  label: "Dossier",  glyph: "◈", subtitle: "career · highlights" },
    { id: "tactics",  label: "Tactics",  glyph: "▦", subtitle: "lines · matching · ST" },
    { id: "ingame",   label: "In-Game",  glyph: "◭", subtitle: "decisions · pulls · pens" },
    { id: "staff",    label: "Staff",    glyph: "◆", subtitle: "coordinators · changes" },
    { id: "identity", label: "Identity", glyph: "⌖", subtitle: "style · fit · venue" },
    { id: "context",  label: "Context",  glyph: "◉", subtitle: "buyer/seller · GM · playoff" },
  ];
  return (
    <div className="hud-panel hud-panel--all-corners jarvis-boot mb-4 overflow-x-auto"
      style={{ ["--hud-corner" as string]: `${teamColor}aa` }}>
      <span className="hud-panel__corner-tr" />
      <span className="hud-panel__corner-bl" />
      <div className="flex items-stretch min-w-min">
        {tabs.map(tab => {
          const isActive = active === tab.id;
          return (
            <button key={tab.id} onClick={() => onChange(tab.id)}
              className="relative group flex items-center gap-2 lg:gap-3 px-3 lg:px-4 py-2.5 transition-all shrink-0"
              style={{
                background: isActive ? `linear-gradient(90deg, ${teamColor}28 0%, ${teamColor}08 100%)` : "transparent",
                borderLeft: `2px solid ${isActive ? teamColor : "transparent"}`,
                boxShadow: isActive ? `inset 0 0 14px ${teamColor}22` : "none",
              }}>
              {isActive && (
                <span aria-hidden className="absolute inset-y-0 left-0 right-0 pointer-events-none"
                  style={{
                    background: `linear-gradient(90deg, transparent, ${teamColor}33, transparent)`,
                    animation: "coachRailScan 3.6s linear infinite",
                    mixBlendMode: "screen",
                  }} />
              )}
              <span className="hud-mono text-[16px] leading-none shrink-0"
                style={{
                  color: isActive ? teamColor : "rgba(255,255,255,0.55)",
                  textShadow: isActive ? `0 0 8px ${teamColor}` : "none",
                }}>
                {tab.glyph}
              </span>
              <span className="hidden md:flex flex-col items-start min-w-0">
                <span className="hud-mono text-[10px] uppercase tracking-[0.20em] font-semibold leading-tight"
                  style={{
                    color: isActive ? "rgba(255,255,255,0.96)" : "rgba(255,255,255,0.70)",
                    textShadow: isActive ? `0 0 4px ${teamColor}55` : "none",
                  }}>
                  {tab.label}
                </span>
                <span className="hud-mono text-[7px] uppercase tracking-[0.18em] text-white/30 leading-tight">
                  {tab.subtitle}
                </span>
              </span>
              <span className="md:hidden hud-mono text-[11px] uppercase tracking-[0.14em] font-bold"
                style={{ color: isActive ? "white" : "rgba(255,255,255,0.65)" }}>
                {tab.label}
              </span>
            </button>
          );
        })}
      </div>
      <style jsx>{`
        @keyframes coachRailScan {
          0%   { transform: translateX(-100%); }
          100% { transform: translateX(100%);  }
        }
      `}</style>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab content
// ---------------------------------------------------------------------------

function DossierTab({ profile, accent }: { profile: CoachProfile; accent: string }) {
  const cp = profile.coach_profile?.row;
  const dn = profile.coach_decision_net?.row;
  const va = profile.venue_atmosphere?.row;
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <HudCard title="Career with Team" subtitle="4.7" accent={accent}>
        {cp ? (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
              <StatPill label="GP"       value={cp.gp_under_coach} />
              <StatPill label="Pts %"    value={`${(cp.points_pct * 100).toFixed(1)}%`} color={cp.points_pct >= 0.55 ? "text-[#4ade80]" : cp.points_pct >= 0.45 ? "text-white" : "text-[#f87171]"} />
              <StatPill label="Points"   value={cp.points} />
              <StatPill label="Record"   value={`${cp.wins}-${cp.losses}-${cp.ot_losses}`} />
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
              <StatPill label="GF/G"     value={fmtNum(cp.gf_per_game)} />
              <StatPill label="GA/G"     value={fmtNum(cp.ga_per_game)} color={cp.ga_per_game < cp.gf_per_game ? "text-white" : "text-[#f87171]"} />
              <StatPill label="SF/G"     value={fmtNum(cp.sf_per_game, 1)} />
              <StatPill label="SA/G"     value={fmtNum(cp.sa_per_game, 1)} />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <StatPill label="PP %"     value={`${(cp.pp_pct * 100).toFixed(1)}%`} color={cp.pp_pct >= 0.20 ? "text-[#4ade80]" : "text-white"} />
              <StatPill label="PK %"     value={`${(cp.pk_pct * 100).toFixed(1)}%`} color={cp.pk_pct >= 0.80 ? "text-[#4ade80]" : "text-white"} />
            </div>
            <p className="mt-3 text-[9px] font-mono text-white/25">seasons {cp.seasons_covered.join(", ")}</p>
          </>
        ) : (
          <p className="text-[10px] font-mono text-white/40">No coach profile yet.</p>
        )}
      </HudCard>

      <HudCard title="Decision Aggression" subtitle="4.17" accent={accent}>
        {dn ? (
          <>
            <div className="flex items-center justify-center mb-3">
              <Gauge value={dn.overall_aggression} max={1} accent={accent} w={220} h={100} />
            </div>
            <div className="space-y-0.5">
              <BarRow label="Timeout"     value={dn.timeout_aggression} accent={accent} />
              <BarRow label="Pull"        value={dn.pull_aggression}    accent={accent} />
              <BarRow label="Line Shelter" value={dn.line_shelter_score} accent={accent} />
              <BarRow label="ST 1st Unit" value={dn.st_first_unit_lean} accent={accent} />
              <BarRow label="Discipline"  value={dn.penalty_discipline} accent={accent} />
              <BarRow label="Matching"    value={dn.matching_intensity} accent={accent} />
            </div>
          </>
        ) : (
          <p className="text-[10px] font-mono text-white/40">No decision profile yet.</p>
        )}
      </HudCard>

      {va && (
        <HudCard title="Home Venue Scare" subtitle="4.19" accent={accent}>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
            <StatPill label="Scare"      value={(va.scare_factor >= 0 ? "+" : "") + va.scare_factor.toFixed(2)}
              color={va.scare_factor > 0.3 ? "text-[#f87171]" : va.scare_factor > 0 ? "text-[#fbbf24]" : "text-white/55"} />
            <StatPill label="vSV% Δ"     value={(va.visiting_sv_delta >= 0 ? "+" : "") + (va.visiting_sv_delta * 100).toFixed(2) + "%"}
              color={va.visiting_sv_delta < 0 ? "text-[#f87171]" : "text-[#4ade80]"} />
            <StatPill label="vFOW% Δ"    value={(va.visiting_fow_delta >= 0 ? "+" : "") + (va.visiting_fow_delta * 100).toFixed(1) + "%"}
              color={va.visiting_fow_delta < 0 ? "text-[#f87171]" : "text-[#4ade80]"} />
            <StatPill label="Ref PP Δ"   value={(va.ref_pp_delta >= 0 ? "+" : "") + va.ref_pp_delta.toFixed(2)}
              color={va.ref_pp_delta > 0 ? "text-[#4ade80]" : "text-white/55"} />
          </div>
          <p className="text-[9px] font-mono text-white/25">rank {(va.scare_rank * 100).toFixed(0)}th percentile · {va.home_gp} home GP</p>
        </HudCard>
      )}

      {profile.playoff_elimination?.row && profile.playoff_elimination.row.elimination_drag > 0 && (
        <HudCard title="Playoff Elimination" subtitle="4.20" accent="#fbbf24">
          {(() => {
            const pe = profile.playoff_elimination!.row!;
            return (
              <>
                <div className="grid grid-cols-3 gap-2 mb-2">
                  <StatPill label="P(playoff)"  value={(pe.playoff_prob * 100).toFixed(0) + "%"} color="text-[#fbbf24]" />
                  <StatPill label="Drag"        value={(pe.elimination_drag * 100).toFixed(0) + "%"} color="text-[#f87171]" />
                  <StatPill label="Eff Mult"    value={"×" + pe.efficiency_multiplier.toFixed(3)} color="text-[#f87171]" />
                </div>
                <p className="text-[9px] font-mono text-white/30">{pe.games_remaining} games remaining · P% {(pe.points_pct * 100).toFixed(1)}</p>
              </>
            );
          })()}
        </HudCard>
      )}
    </div>
  );
}

function TacticsTab({ profile, accent }: { profile: CoachProfile; accent: string }) {
  const ld = profile.line_deployment?.rows ?? [];
  const lm = profile.line_matching;
  const st = profile.st_deployment?.units ?? [];

  return (
    <div className="space-y-4">
      <HudCard title="Projected Lines · F + D" subtitle="4.1" accent={accent}>
        {ld.length === 0 ? (
          <p className="text-[10px] font-mono text-white/40">No deployment data.</p>
        ) : (
          <div className="space-y-1">
            {ld.map(l => (
              <div key={`${l.line_type}-${l.line_rank}`}
                className="grid grid-cols-[3rem_1fr_4.5rem_4.5rem_4rem_4rem] items-center gap-3 px-3 py-2 rounded bg-white/[0.02] border border-white/[0.05]">
                <span className="text-[10px] font-semibold font-mono text-white/65">{l.line_type}{l.line_rank}</span>
                <div className="flex flex-wrap gap-2 text-[11px] text-white/80">
                  {l.player_names.map((n, i) => (
                    <a key={l.player_ids[i] ?? i} href={`/players/${encodeURIComponent(n)}`}
                      className="hover:text-white transition-colors">{n}</a>
                  ))}
                </div>
                <span className="text-[10px] font-mono text-right" style={{ color: accent }}>{fmtMin(l.line_toi_per_game)}/g</span>
                <span className="text-[10px] font-mono text-white/55 text-right">{fmtMin(l.trio_toi_per_game)}/g</span>
                <span className="text-[10px] font-mono text-white/45 text-right">{fmtPct(l.cohesion_pct)}</span>
                <span className="text-[10px] font-mono text-white/35 text-right">{fmtPct(l.share_of_team_toi)}</span>
              </div>
            ))}
          </div>
        )}
      </HudCard>

      <div className="grid gap-4 md:grid-cols-2">
        <MatchingPanel data={lm} accent={accent} />
        <HudCard title="Special Teams Units" subtitle="4.3" accent={accent}>
          {st.length === 0 ? (
            <p className="text-[10px] font-mono text-white/40">No ST data.</p>
          ) : (
            <div className="grid gap-2">
              {st.map(u => {
                const unitColor =
                  u.unit_type === "PP1" ? "#4ade80" :
                  u.unit_type === "PP2" ? "#fbbf24" :
                  u.unit_type === "PK1" ? "#60a5fa" :
                  u.unit_type === "PK2" ? "#a78bfa" : "rgba(255,255,255,0.2)";
                return (
                  <div key={u.unit_type} className="rounded border bg-white/[0.02] px-3 py-2"
                    style={{ borderColor: `${unitColor}66` }}>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-[11px] font-mono font-semibold" style={{ color: unitColor }}>{u.unit_type}</span>
                      <span className="text-[9px] font-mono text-white/40">
                        {fmtMin(u.unit_toi_secs)} · {fmtPct(u.share_of_st_toi)} · {u.team_st_gp}gp
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-x-2 gap-y-1">
                      {u.player_names.map((n, i) => (
                        <a key={u.player_ids[i] ?? i} href={`/players/${encodeURIComponent(n)}`}
                          className="text-[10px] text-white/75 hover:text-white">{n}</a>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </HudCard>
      </div>
    </div>
  );
}

function MatchingPanel({ data, accent }: {
  data: CoachProfile["line_matching"];
  accent: string;
}) {
  const [lineType, setLineType] = useState<"F" | "D">("F");
  const [venue, setVenue] = useState<"home" | "away" | "all">("all");
  if (!data) return <HudCard title="Matching" subtitle="4.2" accent={accent}><p className="text-[10px] font-mono text-white/40">—</p></HudCard>;
  const rows = lineType === "F" ? data.F : data.D;
  const ranks = lineType === "F" ? [1,2,3,4] : [1,2,3];
  function lookup(own: number, opp: number): number | null {
    const m = rows.filter(r => r.own_line_rank === own && r.opp_line_rank === opp && (venue === "all" || r.venue === venue));
    if (m.length === 0) return null;
    const num = m.reduce((s, r) => s + r.weighted_share * r.total_toi_secs, 0);
    const den = m.reduce((s, r) => s + r.total_toi_secs, 0);
    return den > 0 ? num / den : null;
  }
  function color(v: number | null): string {
    if (v == null) return "rgba(255,255,255,0.04)";
    return `${accent}${Math.round(15 + Math.min(1, v) * 220).toString(16).padStart(2, "0")}`;
  }
  return (
    <HudCard title="Matching Profile" subtitle="4.2" accent={accent}
      right={
        <div className="flex items-center gap-1">
          <div className="flex gap-0.5 border border-white/[0.08] rounded overflow-hidden">
            {(["F", "D"] as const).map(t => (
              <button key={t} onClick={() => setLineType(t)}
                className={`px-1.5 py-0.5 text-[9px] font-mono ${lineType === t ? "bg-white/[0.10] text-white" : "text-white/40"}`}>
                {t}
              </button>
            ))}
          </div>
          <div className="flex gap-0.5 border border-white/[0.08] rounded overflow-hidden">
            {(["all","home","away"] as const).map(v => (
              <button key={v} onClick={() => setVenue(v)}
                className={`px-1.5 py-0.5 text-[8px] font-mono uppercase ${venue === v ? "bg-white/[0.10] text-white" : "text-white/40"}`}>
                {v}
              </button>
            ))}
          </div>
        </div>
      }
    >
      {rows.length === 0 ? <p className="text-[10px] font-mono text-white/40">No data.</p> : (
        <>
          <p className="text-[8px] font-mono text-white/35 mb-2 leading-relaxed">P(own | opp on ice). Brighter = more matched.</p>
          <div className="grid gap-1" style={{ gridTemplateColumns: `2rem repeat(${ranks.length}, 1fr)` }}>
            <div />
            {ranks.map(r => <div key={`h${r}`} className="text-[8px] font-mono text-white/40 text-center">opp{r}</div>)}
            {ranks.map(own => (
              <Fragment key={`r${own}`}>
                <div className="text-[8px] font-mono text-white/40 self-center">own{own}</div>
                {ranks.map(opp => {
                  const v = lookup(own, opp);
                  return (
                    <div key={`${own}-${opp}`} className="rounded border border-white/[0.05] text-center py-1.5"
                      style={{ background: color(v) }}>
                      <span className="text-[9px] font-mono text-white/90">{v == null ? "—" : `${Math.round(v * 100)}`}</span>
                    </div>
                  );
                })}
              </Fragment>
            ))}
          </div>
        </>
      )}
    </HudCard>
  );
}

function InGameTab({ profile, accent }: { profile: CoachProfile; accent: string }) {
  const gp = profile.goalie_pull?.rows ?? [];
  const pt = profile.penalty_tendency?.row;
  const la = profile.penalty_tendency?.league_avg;
  const to = profile.timeout_usage?.rows ?? [];
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <HudCard title="Goalie Pull Timing" subtitle="4.5" accent={accent}>
        {gp.length === 0 ? (
          <p className="text-[10px] font-mono text-white/40">No pulls observed.</p>
        ) : (
          <div className="space-y-1">
            {gp.map(r => (
              <div key={r.deficit}
                className="grid grid-cols-[3rem_1fr_4rem_4rem] items-center gap-2 px-3 py-1.5 rounded bg-white/[0.02] border border-white/[0.05]">
                <span className="text-[11px] font-mono font-semibold text-[#f87171]">−{r.deficit}</span>
                <span className="text-[10px] font-mono text-white/55">
                  {r.n_pulls} pulls / {r.n_team_games}gp
                </span>
                <span className="text-[10px] font-mono text-white/45 text-right">med {fmtSec(r.median_pull_time_secs)}</span>
                <span className="text-[10px] font-mono text-right" style={{ color: accent }}>mean {fmtSec(r.mean_pull_time_secs)}</span>
              </div>
            ))}
          </div>
        )}
      </HudCard>

      <HudCard title="Penalty Tendency" subtitle="4.6" accent={accent}>
        {!pt ? (
          <p className="text-[10px] font-mono text-white/40">No data.</p>
        ) : (
          <>
            <div className="grid grid-cols-3 gap-2 mb-3">
              <StatPill label="Penalties / G" value={fmtNum(pt.penalties_taken_per_game)}
                color={pt.penalties_taken_per_game > (la?.penalties_taken_per_game ?? 0) ? "text-[#f87171]" : "text-[#4ade80]"} />
              <StatPill label="PIM / G" value={fmtNum(pt.pim_per_game)}
                color={pt.pim_per_game > (la?.pim_per_game ?? 0) ? "text-[#f87171]" : "text-[#4ade80]"} />
              <StatPill label="PP Earned / G" value={fmtNum(pt.pp_opps_per_game)} />
            </div>
            <p className="text-[9px] font-mono text-white/25">
              league avg P/G {fmtNum(la?.penalties_taken_per_game)} · PIM/G {fmtNum(la?.pim_per_game)} · n_games {pt.n_games}
            </p>
          </>
        )}
      </HudCard>

      <HudCard title="Timeout Usage" subtitle="4.4" accent={accent} right={to.length === 0 ? <SkeletonBadge /> : undefined}>
        {to.length === 0 ? (
          <p className="text-[10px] font-mono text-white/35">PBP ingester doesn&apos;t yet capture team timeouts. Model is built; will populate once data lands.</p>
        ) : (
          <div className="space-y-1">
            {to.map((r, i) => (
              <div key={i} className="grid grid-cols-[5rem_5rem_5rem_3rem_4rem] items-center gap-2 px-3 py-1.5 rounded bg-white/[0.02] border border-white/[0.05]">
                <span className="text-[10px] font-mono text-white/55">{r.period_bucket}</span>
                <span className="text-[10px] font-mono text-white/55">{r.score_state}</span>
                <span className="text-[10px] font-mono text-white/55">{r.time_bucket}</span>
                <span className="text-[10px] font-mono text-white/70 text-right">{r.n_timeouts}</span>
                <span className="text-[10px] font-mono text-right" style={{ color: accent }}>{fmtNum(r.rate_per_game, 3)}</span>
              </div>
            ))}
          </div>
        )}
      </HudCard>
    </div>
  );
}

function StaffTab({ profile, accent }: { profile: CoachProfile; accent: string }) {
  const pp = profile.pp_coordinator?.row;
  const pk = profile.pk_coordinator?.row;
  const gc = profile.goalie_coach?.row;
  const sc = profile.staff_changes?.rows ?? [];
  const fo = profile.fo_regime_changes?.rows ?? [];

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <HudCard title="PP Coordinator" subtitle="4.9" accent="#4ade80">
          {pp ? (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
                <StatPill label="xG / 60"   value={fmtNum(pp.pp_xg_per_60)} color={pp.pp_xg_per_60 >= 12 ? "text-[#4ade80]" : "text-white"} />
                <StatPill label="Sh / 60"   value={fmtNum(pp.pp_shots_per_60, 1)} />
                <StatPill label="G / 60"    value={fmtNum(pp.pp_goals_per_60)} color={pp.pp_goals_per_60 >= 10 ? "text-[#4ade80]" : "text-white"} />
                <StatPill label="xG / Sh"   value={fmtNum(pp.pp_xg_per_shot, 3)} />
              </div>
              <div className="grid grid-cols-2 gap-2 mb-2">
                <StatPill label="Avg Dist"  value={fmtNum(pp.pp_shot_distance_avg, 1) + " ft"} />
                <StatPill label="PP1 Share" value={(pp.pp1_qb_share * 100).toFixed(1) + "%"} />
              </div>
              <div className="px-3 py-1.5 rounded bg-white/[0.02] border border-white/[0.05] flex items-center gap-2">
                <span className="text-[9px] font-mono uppercase tracking-wider text-[#4ade80] w-20">PP1 QB</span>
                {pp.pp1_qb_name ? (
                  <a href={`/players/${encodeURIComponent(pp.pp1_qb_name)}`} className="text-[11px] text-white/85 hover:text-white">
                    {pp.pp1_qb_name}
                  </a>
                ) : <span className="text-[10px] text-white/30 italic">no D on PP1</span>}
                <span className="ml-auto text-[9px] font-mono text-white/30">{pp.pp_shots} sh · {pp.pp_goals} G · {pp.pp_team_gp}gp</span>
              </div>
            </>
          ) : <p className="text-[10px] font-mono text-white/40">No data.</p>}
        </HudCard>

        <HudCard title="PK Coordinator" subtitle="4.10" accent="#60a5fa">
          {pk ? (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
                <StatPill label="PK SV%"    value={(pk.pk_save_pct * 100).toFixed(1) + "%"} color={pk.pk_save_pct >= 0.82 ? "text-[#4ade80]" : "text-white"} />
                <StatPill label="SA / 60"   value={fmtNum(pk.pk_sa_per_60, 1)} />
                <StatPill label="xGA / 60"  value={fmtNum(pk.pk_xga_per_60)} />
                <StatPill label="GA / 60"   value={fmtNum(pk.pk_ga_per_60)} />
              </div>
              <div className="grid grid-cols-3 gap-2">
                <StatPill label="xGA / Sh"  value={fmtNum(pk.pk_xga_per_shot, 3)} />
                <StatPill label="SH Sh/60"  value={fmtNum(pk.sh_shots_per_60, 1)} color={pk.sh_shots_per_60 >= 20 ? "text-[#4ade80]" : "text-white/55"} />
                <StatPill label="PK1 Share" value={(pk.pk1_share * 100).toFixed(1) + "%"} />
              </div>
              <p className="mt-2 text-[9px] font-mono text-white/25">{pk.pk_sa} SA · {pk.pk_ga} GA · {pk.sh_goals_for} SHG</p>
            </>
          ) : <p className="text-[10px] font-mono text-white/40">No data.</p>}
        </HudCard>
      </div>

      <HudCard title="Goalie Coach Curve" subtitle="4.8" accent={accent}>
        {gc ? (
          <>
            <div className="grid grid-cols-3 gap-2 mb-3">
              <StatPill label="Season SV%" value={gc.season_save_pct != null ? gc.season_save_pct.toFixed(3) : "—"}
                color={gc.season_save_pct != null && gc.season_save_pct >= 0.905 ? "text-[#4ade80]"
                  : gc.season_save_pct != null && gc.season_save_pct >= 0.890 ? "text-white" : "text-[#f87171]"} />
              <StatPill label="vs Prior" value={gc.save_pct_delta != null ? (gc.save_pct_delta >= 0 ? "+" : "") + gc.save_pct_delta.toFixed(3) : "—"}
                color={gc.save_pct_delta != null && gc.save_pct_delta > 0 ? "text-[#4ade80]"
                  : gc.save_pct_delta != null && gc.save_pct_delta < 0 ? "text-[#f87171]" : "text-white/55"} />
              <StatPill label={gc.change_point_detected ? "Split Δ · CHANGE" : "Split Δ"}
                value={gc.split_delta != null ? (gc.split_delta >= 0 ? "+" : "") + gc.split_delta.toFixed(3) : "—"}
                color={gc.change_point_detected ? "text-[#fbbf24]" : "text-white/70"} />
            </div>
            {gc.rolling_save_pct.length >= 2 && (
              <div className="rounded border border-white/[0.05] bg-white/[0.02] p-3">
                <Sparkline pts={gc.rolling_save_pct} accent={accent} w={520} h={68} />
                <div className="flex justify-between text-[8px] font-mono text-white/25 mt-1">
                  <span>early</span><span>league avg .900</span><span>late</span>
                </div>
              </div>
            )}
          </>
        ) : <p className="text-[10px] font-mono text-white/40">No data.</p>}
      </HudCard>

      {(sc.length > 0 || fo.length > 0) && (
        <HudCard title="Staff / FO Changes" subtitle="4.13 + 4.14" accent={accent}>
          <div className="space-y-1.5">
            {sc.map((r, i) => (
              <div key={`s${i}`} className="flex items-center gap-3 px-3 py-2 rounded bg-white/[0.02] border border-[#f87171]/25">
                <span className="text-[9px] font-mono text-white/35 w-20">{r.date}</span>
                <span className="h-1.5 w-1.5 rounded-full bg-[#f87171]" />
                <span className="text-[10px] text-white/75 flex-1">{r.change_type.replace(/_/g, " ")} — {r.person_out || "unknown"}</span>
                <span className="text-[9px] font-mono text-white/30">decay {r.decay_games}g</span>
              </div>
            ))}
            {fo.map((r, i) => (
              <div key={`f${i}`} className="flex items-center gap-3 px-3 py-2 rounded bg-white/[0.02] border border-[#fbbf24]/25">
                <span className="text-[9px] font-mono text-white/35 w-20">{r.date}</span>
                <span className="h-1.5 w-1.5 rounded-full bg-[#fbbf24]" />
                <span className="text-[10px] text-white/75 flex-1">{r.fo_role.replace(/_/g, " ")} — {r.person_out || "unknown"}</span>
                <span className="text-[9px] font-mono text-white/30">decay {r.decay_games}g</span>
              </div>
            ))}
          </div>
        </HudCard>
      )}
    </div>
  );
}

function IdentityTab({ profile, accent }: { profile: CoachProfile; accent: string }) {
  const cs = profile.coaching_style?.row;
  const rf = profile.roster_fit?.row;
  const va = profile.venue_atmosphere?.row;
  return (
    <div className="space-y-4">
      <HudCard title="Coaching Style Vector" subtitle="4.11 · 8-dim radar" accent={accent}>
        {cs ? (
          <div className="flex items-center gap-6 flex-wrap">
            <StyleRadar dims={cs.dimensions} accent={accent} />
            <div className="flex-1 min-w-[200px] space-y-1">
              {["forecheck_aggression","dz_structure","pace","physicality","oz_structure","nz_tendency","line_match","st_aggression"].map(k => {
                const d = cs.dimensions[k];
                const rank = d?.rank;
                const raw  = d?.raw;
                if (rank == null || Number.isNaN(rank)) {
                  return (
                    <div key={k} className="flex items-center gap-2 py-0.5">
                      <span className="text-[10px] font-mono text-white/45 w-28 truncate">{k.replace(/_/g, " ")}</span>
                      <span className="text-[10px] font-mono text-white/25 flex-1">—</span>
                    </div>
                  );
                }
                return <BarRow key={k} label={k.replace(/_/g, " ")} value={rank} accent={accent} />;
              })}
            </div>
          </div>
        ) : <p className="text-[10px] font-mono text-white/40">No data.</p>}
      </HudCard>

      <div className="grid gap-4 md:grid-cols-2">
        <HudCard title="Roster Fit Score" subtitle="4.12" accent={accent}>
          {rf ? (
            <>
              <div className="flex items-center justify-center mb-3">
                <Gauge value={rf.fit_score} max={1} accent={
                  rf.fit_score >= 0.55 ? "#4ade80" : rf.fit_score >= 0.40 ? "#fbbf24" : "#f87171"
                } w={220} h={100} />
              </div>
              <div className="grid grid-cols-2 gap-2 mb-3">
                <StatPill label="Top Archetype" value={rf.archetype_top || "—"} />
                <StatPill label="Weak Dim"      value={rf.mismatch_dim || "—"} color="text-[#f87171]" />
              </div>
              {rf.archetypes.length > 0 && (
                <div className="space-y-1">
                  {rf.archetypes.map((a, i) => (
                    <div key={a} className="flex items-center gap-2">
                      <span className="text-[9px] font-mono text-white/55 w-32 truncate">{a}</span>
                      <div className="flex-1 h-1 rounded-full bg-white/[0.05] overflow-hidden">
                        <div className="h-full" style={{
                          width: `${(rf.archetype_shares[i] || 0) * 100}%`,
                          background: `linear-gradient(90deg, ${accent}aa, ${accent})`,
                        }} />
                      </div>
                      <span className="text-[9px] font-mono text-white/40 w-10 text-right">
                        {((rf.archetype_shares[i] || 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : <p className="text-[10px] font-mono text-white/40">No data.</p>}
        </HudCard>

        <HudCard title="Home Venue Atmosphere" subtitle="4.19" accent={accent}>
          {va ? (
            <>
              <div className="grid grid-cols-2 gap-2 mb-3">
                <StatPill label="Scare Factor"   value={(va.scare_factor >= 0 ? "+" : "") + va.scare_factor.toFixed(2)}
                  color={va.scare_factor > 0.3 ? "text-[#f87171]" : va.scare_factor > 0 ? "text-[#fbbf24]" : "text-white/55"} />
                <StatPill label="League Rank"    value={(va.scare_rank * 100).toFixed(0) + "th"} />
              </div>
              <div className="space-y-0.5">
                <BarRow label="vSV% Suppress" value={Math.max(0, -va.visiting_sv_delta * 10)} accent={accent} max={1} />
                <BarRow label="vFOW% Drop"    value={Math.max(0, -va.visiting_fow_delta * 5)} accent={accent} max={1} />
                <BarRow label="Ref PP Edge"   value={Math.max(0, va.ref_pp_delta)} accent={accent} max={1} />
                <BarRow label="vxGF Suppress" value={Math.max(0, -va.visiting_xgf_delta * 100)} accent={accent} max={1} />
              </div>
              <p className="mt-2 text-[9px] font-mono text-white/25">{va.home_gp} home GP</p>
            </>
          ) : <p className="text-[10px] font-mono text-white/40">No data.</p>}
        </HudCard>
      </div>
    </div>
  );
}

function ContextTab({ profile, accent }: { profile: CoachProfile; accent: string }) {
  const bs = profile.buyer_seller?.row;
  const sm = profile.seller_motivation?.row;
  const pe = profile.playoff_elimination?.row;
  const gm = profile.gm_fingerprint?.row;
  const archs = ["stand_pat","add_rental","sell_veteran","rebuild","package_deal"] as const;
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <HudCard title="Buyer / Seller" subtitle="4.15" accent={accent}>
          {bs ? (
            <>
              <div className={`rounded border px-4 py-3 mb-3 ${
                bs.classification === "buyer"  ? "border-[#4ade80]/30 bg-[#4ade80]/[0.04]"
                : bs.classification === "seller" ? "border-[#f87171]/30 bg-[#f87171]/[0.04]"
                : "border-[#fbbf24]/30 bg-[#fbbf24]/[0.04]"
              }`}>
                <div className="flex items-center gap-3">
                  <span className={`text-[20px] font-mono font-bold uppercase ${
                    bs.classification === "buyer"  ? "text-[#4ade80]"
                    : bs.classification === "seller" ? "text-[#f87171]"
                    : "text-[#fbbf24]"
                  }`}>{bs.classification}</span>
                  <span className="text-[10px] font-mono text-white/40">conf {(bs.confidence * 100).toFixed(0)}%</span>
                </div>
                <p className="text-[9px] font-mono text-white/35 mt-1">
                  P% {(bs.points_pct * 100).toFixed(1)} · threshold {(bs.threshold * 100).toFixed(1)} · gap {bs.gap >= 0 ? "+" : ""}{(bs.gap * 100).toFixed(1)} · {bs.gp} GP
                </p>
              </div>
              {sm && sm.seller_drag > 0 && (
                <div className="rounded border border-[#f87171]/30 bg-[#f87171]/[0.04] px-3 py-2">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono font-semibold text-[#f87171]">SELLER DRAG</span>
                    <span className="text-[13px] font-mono font-bold text-[#f87171]">{(sm.seller_drag * 100).toFixed(0)}%</span>
                    <span className="ml-auto text-[9px] font-mono text-white/30">
                      ×{sm.efficiency_multiplier.toFixed(3)} · ~{sm.games_since_deadline}g post-DL
                    </span>
                  </div>
                </div>
              )}
            </>
          ) : <p className="text-[10px] font-mono text-white/40">No data.</p>}
        </HudCard>

        <HudCard title="Playoff Elimination" subtitle="4.20" accent={accent}>
          {pe ? (
            <>
              <div className="flex items-center justify-center mb-2">
                <Gauge value={pe.playoff_prob} max={1} accent={
                  pe.playoff_prob >= 0.6 ? "#4ade80" : pe.playoff_prob >= 0.25 ? "#fbbf24" : "#f87171"
                } w={220} h={100} />
              </div>
              <p className="text-center text-[9px] font-mono text-white/30 mb-2">P(playoff) · {pe.games_remaining} games remaining</p>
              {pe.elimination_drag > 0 && (
                <div className="rounded border border-[#fbbf24]/30 bg-[#fbbf24]/[0.04] px-3 py-2">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono font-semibold text-[#fbbf24]">ELIMINATION DRAG</span>
                    <span className="text-[13px] font-mono font-bold text-[#fbbf24]">{(pe.elimination_drag * 100).toFixed(0)}%</span>
                    <span className="ml-auto text-[9px] font-mono text-white/30">×{pe.efficiency_multiplier.toFixed(3)}</span>
                  </div>
                </div>
              )}
            </>
          ) : <p className="text-[10px] font-mono text-white/40">No data.</p>}
        </HudCard>
      </div>

      <HudCard title="GM Behavioral Fingerprint" subtitle="4.18" accent={accent}>
        {gm ? (
          <>
            <div className="flex items-center gap-3 mb-3 px-3 py-2 rounded bg-white/[0.02] border border-white/[0.05]">
              <span className="text-[14px] font-mono font-bold uppercase tracking-wider" style={{ color: accent }}>
                {gm.action_archetype.replace(/_/g, " ")}
              </span>
              {gm.gm_name && <span className="text-[10px] text-white/55 ml-auto">{gm.gm_name}</span>}
            </div>
            <div className="space-y-1.5 mb-3">
              {archs.map(a => {
                const prob = (gm as any)[`prob_${a}`] as number;
                return (
                  <BarRow key={a} label={a.replace(/_/g, " ")} value={prob} accent={accent} suffix="" max={1} />
                );
              })}
            </div>
            <div className="grid grid-cols-2 gap-2">
              <StatPill label="DL Aggression" value={gm.deadline_aggression.toFixed(2)} />
              <StatPill label="Recent Tx"     value={gm.recent_tx_count} />
            </div>
          </>
        ) : <p className="text-[10px] font-mono text-white/40">No data.</p>}
      </HudCard>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function CoachPage() {
  const params = useParams<{ name: string }>();
  const rawName = decodeURIComponent(params?.name ?? "");
  const [profile, setProfile] = useState<CoachProfile | null>(null);
  const [tab, setTab] = useState<Tab>("dossier");

  useEffect(() => {
    if (!rawName) return;
    setProfile(null);
    fetch(`/api/coaches/${encodeURIComponent(rawName)}`)
      .then(r => r.json()).then(setProfile)
      .catch(() => setProfile({ status: "not_found", name: rawName }));
  }, [rawName]);

  const today = new Date().toISOString().slice(0, 10);

  if (profile === null) {
    return (
      <main className="relative min-h-screen p-3 sm:p-6">
        <HudGrid />
        <p className="hud-mono text-[10px] uppercase tracking-widest text-white/40 animate-pulse">LOADING…</p>
      </main>
    );
  }

  if (profile.status === "not_found" || !profile.meta) {
    return (
      <main className="relative min-h-screen p-3 sm:p-6">
        <HudGrid />
        <div className="hud-panel hud-panel--all-corners p-6">
          <p className="text-white/55 text-sm">No coach found for &ldquo;{rawName}&rdquo;.</p>
          <p className="text-white/30 text-[10px] mt-1 font-mono">Add them to data/coaches.json.</p>
        </div>
      </main>
    );
  }

  const meta = profile.meta;
  const teamColor = TEAM_COLORS[meta.team] ?? "#fb923c";
  const teamSecondary = TEAM_SECONDARY[meta.team] ?? teamColor;

  return (
    <main className="relative min-h-screen p-3 sm:p-6">
      <HudGrid />

      <div className="relative z-10 mb-3 flex items-center gap-2 flex-wrap">
        <span className="hud-mono text-[10px] uppercase tracking-[0.20em]" style={{ color: teamColor }} aria-hidden>◢</span>
        <span className="hud-mono text-[10px] uppercase tracking-[0.20em]" style={{ color: teamColor }}>
          COACH DOSSIER · {meta.team}
        </span>
        <span className="hud-mono text-[9px] uppercase tracking-[0.16em] text-white/40">
          · phase 4 · all 21 features
        </span>
        <span className="ml-auto text-white/40 text-xs font-mono">{today}</span>
      </div>

      <Hero meta={meta} profile={profile} teamColor={teamColor} teamSecondary={teamSecondary} />

      <TabBar active={tab} onChange={setTab} teamColor={teamColor} />

      <div className="space-y-4">
        {tab === "dossier"  && <DossierTab  profile={profile} accent={teamColor} />}
        {tab === "tactics"  && <TacticsTab  profile={profile} accent={teamColor} />}
        {tab === "ingame"   && <InGameTab   profile={profile} accent={teamColor} />}
        {tab === "staff"    && <StaffTab    profile={profile} accent={teamColor} />}
        {tab === "identity" && <IdentityTab profile={profile} accent={teamColor} />}
        {tab === "context"  && <ContextTab  profile={profile} accent={teamColor} />}
      </div>
    </main>
  );
}
