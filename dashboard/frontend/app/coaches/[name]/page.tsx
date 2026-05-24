"use client";

import { Fragment, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { logoUrl, TEAM_FULL_NAMES } from "@/utils/nhl";
import { HudGrid } from "@/components/hud";

// ---------------------------------------------------------------------------
// Types — mirror /coaches/{name} from dashboard/api/main.py
// ---------------------------------------------------------------------------

interface LineRow {
  line_type:               string;
  line_rank:               number;
  player_ids:              number[];
  player_names:            string[];
  chemistry_toi_secs:      number | null;
  trio_toi_per_game:       number | null;
  line_toi_per_game:       number | null;
  cohesion_pct:            number | null;
  share_of_team_toi:       number | null;
  team_gp:                 number;
}

interface MatchingRow {
  own_line_rank:  number;
  opp_line_rank:  number;
  venue:          "home" | "away" | string;
  weighted_share: number;
  total_toi_secs: number;
}

interface StUnit {
  unit_type:        string;
  player_ids:       number[];
  player_names:     string[];
  unit_toi_secs:    number | null;
  share_of_st_toi:  number | null;
  team_st_toi:      number | null;
  team_st_gp:       number;
}

interface PullRow {
  deficit:               number;
  n_pulls:               number;
  n_team_games:          number;
  mean_pull_time_secs:   number | null;
  median_pull_time_secs: number | null;
  earliest_pull_secs:    number | null;
}

interface PenaltyRow {
  n_games:                  number;
  n_penalties_taken:        number;
  n_pp_opportunities:       number;
  pim_total:                number;
  penalties_taken_per_game: number;
  pp_opps_per_game:         number;
  pim_per_game:             number;
  ref_dim:                  string | null;
}

interface TimeoutRow {
  period_bucket:  string;
  score_state:    string;
  time_bucket:    string;
  n_timeouts:     number;
  n_games:        number;
  rate_per_game:  number;
}

interface CoachProfileRow {
  coach_name:              string;
  team:                    string;
  first_named_head_coach:  string | null;
  season:                  number;
  seasons_covered:         number[];
  gp_under_coach:          number;
  wins:                    number;
  ot_wins:                 number;
  losses:                  number;
  ot_losses:               number;
  points:                  number;
  points_pct:              number;
  gf_per_game:             number;
  ga_per_game:             number;
  pp_pct:                  number;
  pk_pct:                  number;
  sf_per_game:             number;
  sa_per_game:             number;
}

interface GoalieCoachRow {
  team:                    string;
  season:                  number;
  gp:                      number;
  shots_against:           number;
  goals_against:           number;
  season_save_pct:         number | null;
  prior_save_pct:          number | null;
  save_pct_delta:          number | null;
  early_split_save_pct:    number | null;
  late_split_save_pct:     number | null;
  split_delta:             number | null;
  change_point_detected:   boolean;
  rolling_save_pct:        number[];
  goalie_coach:            string;
}

interface PpCoordinatorRow {
  team:                    string;
  season:                  number;
  pp_toi_secs:             number;
  pp_team_gp:              number;
  pp_shots:                number;
  pp_goals:                number;
  pp_xg_total:             number;
  pp_shots_per_60:         number;
  pp_xg_per_60:            number;
  pp_goals_per_60:         number;
  pp_xg_per_shot:          number;
  pp_shot_distance_avg:    number;
  pp_carry_pct:            number | null;
  pp1_qb_id:               number | null;
  pp1_qb_name:             string;
  pp1_qb_share:            number;
  pp_coordinator:          string;
}

interface CoachProfile {
  status: "ok" | "not_found";
  name?: string;
  meta?: {
    name:                       string;
    team:                       string;
    first_named_head_coach:     string | null;
    notes:                      string;
  };
  line_deployment?: { rows: LineRow[]; as_of: string | null };
  line_matching?:   { F: MatchingRow[]; D: MatchingRow[]; as_of: string | null };
  st_deployment?:   { units: StUnit[]; as_of: string | null };
  goalie_pull?:     { rows: PullRow[]; as_of: string | null };
  penalty_tendency?:{
    row: PenaltyRow | null;
    league_avg: {
      penalties_taken_per_game?: number;
      pp_opps_per_game?: number;
      pim_per_game?: number;
    };
    as_of: string | null;
  };
  timeout_usage?:   { rows: TimeoutRow[]; as_of: string | null };
  coach_profile?:   { row: CoachProfileRow   | null; as_of: string | null };
  goalie_coach?:    { row: GoalieCoachRow    | null; as_of: string | null };
  pp_coordinator?:  { row: PpCoordinatorRow  | null; as_of: string | null };
  pk_coordinator?:  { row: PkCoordinatorRow  | null; as_of: string | null };
  coaching_style?:  {
    row: CoachingStyleRow | null;
    league_avg: Record<string, number | null> | null;
    as_of: string | null;
  };
  roster_fit?:      { row: RosterFitRow | null; as_of: string | null };
  staff_changes?:   { rows: StaffChangeRow[]; as_of: string | null };
  fo_regime_changes?: { rows: FoRegimeRow[]; as_of: string | null };
  buyer_seller?:    { row: BuyerSellerRow | null; as_of: string | null };
  seller_motivation?: { row: SellerMotivationRow | null; as_of: string | null };
  coach_decision_net?: { row: CoachDecisionRow | null; as_of: string | null };
  gm_fingerprint?:  { row: GmFingerprintRow | null; as_of: string | null };
  venue_atmosphere?: { row: VenueAtmosphereRow | null; as_of: string | null };
  playoff_elimination?: { row: PlayoffEliminationRow | null; as_of: string | null };
}

interface VenueAtmosphereRow {
  team:               string;
  home_gp:            number;
  visiting_sv_delta:  number;
  visiting_fow_delta: number;
  ref_pp_delta:       number;
  visiting_xgf_delta: number;
  scare_factor:       number;
  scare_rank:         number;
}

interface PlayoffEliminationRow {
  team:                  string;
  playoff_prob:          number;
  elimination_drag:      number;
  efficiency_multiplier: number;
  games_remaining:       number;
  points_pct:            number;
}

interface SellerMotivationRow {
  team:                  string;
  seller_drag:           number;
  efficiency_multiplier: number;
  games_since_deadline:  number;
  contextual_flag:       string;
}

interface CoachDecisionRow {
  coach_name:          string;
  team:                string;
  timeout_aggression:  number;
  pull_aggression:     number;
  line_shelter_score:  number;
  st_first_unit_lean:  number;
  penalty_discipline:  number;
  matching_intensity:  number;
  overall_aggression:  number;
}

interface GmFingerprintRow {
  team:                 string;
  gm_name:              string;
  action_archetype:     string;
  prob_stand_pat:       number;
  prob_add_rental:      number;
  prob_sell_veteran:    number;
  prob_rebuild:         number;
  prob_package_deal:    number;
  deadline_aggression:  number;
  recent_tx_count:      number;
}

interface StaffChangeRow {
  date:          string;
  change_type:   string;
  person_out:    string;
  person_in:     string;
  description:   string;
  decay_games:   number;
}

interface FoRegimeRow {
  date:          string;
  fo_role:       string;
  person_out:    string;
  person_in:     string;
  description:   string;
  decay_games:   number;
}

interface BuyerSellerRow {
  team:            string;
  season:          number;
  gp:              number;
  points_pct:      number;
  classification:  "buyer" | "seller" | "neutral";
  confidence:      number;
  gap:             number;
  threshold:       number;
}

interface PkCoordinatorRow {
  team:                  string;
  season:                number;
  pk_toi_secs:           number;
  pk_team_gp:            number;
  pk_sa:                 number;
  pk_ga:                 number;
  pk_xga_total:          number;
  pk_sa_per_60:          number;
  pk_xga_per_60:         number;
  pk_ga_per_60:          number;
  pk_save_pct:           number;
  pk_xga_per_shot:       number;
  pk_shot_distance_avg:  number;
  sh_shots_for:          number;
  sh_goals_for:          number;
  sh_shots_per_60:       number;
  pk1_share:             number;
  pk_coordinator:        string;
}

interface CoachingStyleDim {
  raw:  number | null;
  rank: number | null;
}

interface CoachingStyleRow {
  team:    string;
  season:  number;
  dimensions: {
    forecheck_aggression: CoachingStyleDim;
    dz_structure:         CoachingStyleDim;
    pace:                 CoachingStyleDim;
    physicality:          CoachingStyleDim;
    oz_structure:         CoachingStyleDim;
    nz_tendency:          CoachingStyleDim;
    line_match:           CoachingStyleDim;
    st_aggression:        CoachingStyleDim;
  };
}

interface RosterFitRow {
  team:              string;
  season:            number;
  n_skaters:         number;
  archetype_top:     string;
  archetypes:        string[];
  archetype_shares:  number[];
  fit_score:         number;
  mismatch_dim:      string;
  mismatch_support:  number;
}

// ---------------------------------------------------------------------------
// Shared primitives — match phase4 / player styling
// ---------------------------------------------------------------------------

function Card({
  title,
  ref,
  headerRight,
  children,
  padding = "p-4",
  accent,
}: {
  title: React.ReactNode;
  ref?: string;
  headerRight?: React.ReactNode;
  children: React.ReactNode;
  padding?: string;
  accent?: string;
}) {
  return (
    <div
      className="hud-panel hud-panel--all-corners jarvis-shimmer"
      style={accent ? { ["--hud-corner" as string]: `${accent}aa` } : undefined}
    >
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-white/[0.10]">
        <div className="flex items-center gap-2">
          {ref && (
            <span className="text-[9px] font-semibold font-mono text-white/30 w-7">{ref}</span>
          )}
          <h2 className="text-[10px] font-semibold uppercase tracking-[0.22em] text-[#94a3b8]">{title}</h2>
        </div>
        {headerRight}
      </div>
      <div className={padding}>{children}</div>
    </div>
  );
}

function Spinner() {
  return (
    <p className="text-[10px] font-semibold uppercase tracking-widest text-[#777] animate-pulse">LOADING…</p>
  );
}

function SkeletonBadge() {
  return (
    <span className="text-[8px] font-semibold tracking-wider px-1.5 py-0.5 rounded border border-[#a78bfa]/30 bg-[#a78bfa]/[0.08] text-[#a78bfa]/80">
      SKELETON · NOT TRAINED
    </span>
  );
}

function TeamLogo({ team, size = 18 }: { team: string; size?: number }) {
  const [err, setErr] = useState(false);
  if (!team || err) {
    return team ? <span className="text-[9px] font-semibold text-white/40 font-mono">{team}</span> : null;
  }
  return (
    <img
      src={logoUrl(team)}
      alt={TEAM_FULL_NAMES[team] ?? team}
      title={TEAM_FULL_NAMES[team] ?? team}
      width={size}
      height={size}
      onError={() => setErr(true)}
      className="object-contain shrink-0"
    />
  );
}

function StatPill({ label, value, color = "text-white" }: {
  label: string; value: React.ReactNode; color?: string;
}) {
  return (
    <div className="rounded-lg border border-white/[0.07] bg-white/[0.02] px-3 py-2 flex flex-col gap-0.5">
      <span className="text-[8px] font-semibold uppercase tracking-[0.18em] text-white/35">{label}</span>
      <span className={`text-[14px] font-mono font-semibold ${color}`}>{value}</span>
    </div>
  );
}

function fmtMin(secs: number | null | undefined): string {
  if (secs == null || !Number.isFinite(secs)) return "—";
  return `${(secs / 60).toFixed(1)}m`;
}

function fmtSec(secs: number | null | undefined): string {
  if (secs == null || !Number.isFinite(secs)) return "—";
  return `${secs.toFixed(0)}s`;
}

function fmtPct(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtNum(v: number | null | undefined, places: number = 2): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toFixed(places);
}

// ---------------------------------------------------------------------------
// Hero — futuristic dossier strip with rings backdrop
// ---------------------------------------------------------------------------

function Hero({ meta }: { meta: NonNullable<CoachProfile["meta"]> }) {
  return (
    <div
      className="relative jarvis-shimmer rounded-2xl border overflow-hidden hud-panel--all-corners"
      style={{
        ["--hud-corner" as string]: "#fb923caa",
        borderColor: "rgba(251,146,60,0.30)",
        background: "linear-gradient(160deg, #0a0c10 0%, #060708 65%)",
        boxShadow: "0 4px 24px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.06), 0 0 24px rgba(251,146,60,0.10)",
      }}
    >
      <span className="hud-panel__corner-tr" />
      <span className="hud-panel__corner-bl" />

      {/* Concentric scan rings backdrop */}
      <svg viewBox="0 0 600 220" aria-hidden
        className="absolute left-0 top-0 w-full h-full pointer-events-none opacity-40"
        style={{ zIndex: 0 }}>
        <g style={{ transformOrigin: "78px 100px", animation: "coachRingSlow 30s linear infinite" }}>
          <circle cx={78} cy={100} r={70} fill="none" stroke="#fb923c" strokeOpacity={0.25} strokeDasharray="2 8" />
          {[0, 60, 120, 180, 240, 300].map((deg, i) => {
            const rad = (deg * Math.PI) / 180;
            return (
              <line key={i}
                x1={78 + Math.cos(rad) * 64}
                y1={100 + Math.sin(rad) * 64}
                x2={78 + Math.cos(rad) * 76}
                y2={100 + Math.sin(rad) * 76}
                stroke="#fb923c" strokeOpacity={0.55} strokeWidth={1.2} />
            );
          })}
        </g>
        <g style={{ transformOrigin: "78px 100px", animation: "coachRingRev 18s linear infinite" }}>
          <circle cx={78} cy={100} r={56} fill="none" stroke="#fb923c" strokeOpacity={0.35}
            strokeDasharray="30 20 10 20" strokeLinecap="round" strokeWidth={1.2} />
        </g>
        <line x1={170} y1={100} x2={580} y2={100} stroke="#fb923c" strokeOpacity={0.10} strokeDasharray="3 10" />
      </svg>
      <style jsx>{`
        @keyframes coachRingSlow { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes coachRingRev  { from { transform: rotate(360deg); } to { transform: rotate(0deg); } }
        @media (prefers-reduced-motion: reduce) {
          svg g { animation: none !important; }
        }
      `}</style>

      <div className="relative z-10 flex items-center gap-5 px-5 py-4">
        <a href={`/teams/${meta.team}`} className="shrink-0">
          <TeamLogo team={meta.team} size={68} />
        </a>
        <div className="w-px shrink-0 bg-white/[0.08]" style={{ height: 56 }} />
        <div className="flex-1 min-w-0">
          <h1 className="text-[22px] sm:text-[26px] font-bold text-white tracking-tight leading-tight">
            {meta.name}
          </h1>
          <div className="mt-1.5 flex items-center gap-1.5 flex-wrap text-[11px] font-mono text-white/45">
            <span className="text-[#fb923c] tracking-wider uppercase text-[9px]">HEAD COACH</span>
            <span className="text-white/20">·</span>
            <a href={`/teams/${meta.team}`} className="hover:text-white transition-colors">
              {TEAM_FULL_NAMES[meta.team] ?? meta.team}
            </a>
            {meta.first_named_head_coach && (
              <>
                <span className="text-white/20">·</span>
                <span>since {meta.first_named_head_coach}</span>
              </>
            )}
          </div>
          {meta.notes && (
            <p className="text-[10px] text-white/35 mt-1.5 italic max-w-2xl">{meta.notes}</p>
          )}
        </div>
        <div className="hidden sm:flex flex-col items-end gap-1">
          <span className="hud-mono text-[9px] uppercase tracking-[0.20em] text-[#fb923c]">PHASE 4 DOSSIER</span>
          <span className="hud-mono text-[8px] uppercase tracking-[0.16em] text-white/30">21 feature slots</span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section header — visually groups built vs. skeleton panels
// ---------------------------------------------------------------------------

function SectionLabel({ children, accent = "#fb923c" }: { children: React.ReactNode; accent?: string }) {
  return (
    <div className="flex items-center gap-2 mt-2 mb-1">
      <span className="hud-mono text-[10px] uppercase tracking-[0.22em]" style={{ color: accent }}>
        ◢ {children}
      </span>
      <span className="flex-1 h-px" style={{ background: `linear-gradient(to right, ${accent}55, transparent)` }} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// 4.1 Line Deployment
// ---------------------------------------------------------------------------

function LineDeploymentPanel({ data }: { data: CoachProfile["line_deployment"] }) {
  if (!data) return null;
  return (
    <Card
      ref="4.1"
      title={<>Projected Lines · Forwards & D-Pairs</>}
      headerRight={data.as_of ? <span className="text-[9px] font-mono text-white/30">{data.as_of}</span> : null}
    >
      {data.rows.length === 0 ? (
        <p className="text-[10px] text-white/40 font-mono">No deployment data for this team yet.</p>
      ) : (
        <div className="space-y-1">
          {data.rows.map(l => (
            <div
              key={`${l.line_type}-${l.line_rank}`}
              className="grid grid-cols-[3rem_1fr_4.5rem_4.5rem_4rem_4rem] items-center gap-3 px-3 py-2 rounded-lg bg-white/[0.02] border border-white/[0.06]"
            >
              <span className="text-[10px] font-semibold font-mono text-white/65">
                {l.line_type}{l.line_rank}
              </span>
              <div className="flex flex-wrap gap-2 text-[11px] font-medium text-white/80">
                {l.player_names.map((n, i) => (
                  <a
                    key={l.player_ids[i] ?? i}
                    href={`/players/${encodeURIComponent(n)}`}
                    className="hover:text-white transition-colors"
                  >
                    {n}
                  </a>
                ))}
              </div>
              <span className="text-[10px] font-mono text-[#fb923c] text-right">{fmtMin(l.line_toi_per_game)}/g</span>
              <span className="text-[10px] font-mono text-white/65 text-right">{fmtMin(l.trio_toi_per_game)}/g</span>
              <span className="text-[10px] font-mono text-white/50 text-right">{fmtPct(l.cohesion_pct)}</span>
              <span className="text-[10px] font-mono text-white/40 text-right">{fmtPct(l.share_of_team_toi)}</span>
            </div>
          ))}
          <p className="mt-2 text-[9px] text-white/30 font-mono">
            line minutes · trio cohesion · cohesion % · share of 5v5 · over {data.rows[0].team_gp} GP
          </p>
        </div>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 4.2 Matching heatmap
// ---------------------------------------------------------------------------

function MatchingPanel({ data, team }: { data: CoachProfile["line_matching"]; team: string }) {
  const [lineType, setLineType] = useState<"F" | "D">("F");
  const [venue, setVenue] = useState<"home" | "away" | "all">("all");
  if (!data) return null;

  const rows = lineType === "F" ? data.F : data.D;
  const ownRanks = lineType === "F" ? [1, 2, 3, 4] : [1, 2, 3];
  const oppRanks = ownRanks;

  function lookup(own: number, opp: number): number | null {
    const matches = rows.filter(r => r.own_line_rank === own && r.opp_line_rank === opp
                                       && (venue === "all" || r.venue === venue));
    if (matches.length === 0) return null;
    if (matches.length === 1) return matches[0].weighted_share;
    const num = matches.reduce((s, r) => s + r.weighted_share * r.total_toi_secs, 0);
    const den = matches.reduce((s, r) => s + r.total_toi_secs, 0);
    return den > 0 ? num / den : null;
  }
  function color(v: number | null): string {
    if (v == null) return "rgba(255,255,255,0.04)";
    const c = Math.min(1, Math.max(0, v));
    return `rgba(251,146,60,${0.10 + c * 0.85})`;
  }

  return (
    <Card
      ref="4.2"
      title={<>{team} Matchup Profile</>}
      headerRight={
        <div className="flex items-center gap-2">
          {data.as_of && <span className="text-[9px] font-mono text-white/30">{data.as_of}</span>}
          <div className="flex gap-0.5 border border-white/[0.08] rounded overflow-hidden">
            {(["F", "D"] as const).map(t => (
              <button key={t} onClick={() => setLineType(t)}
                className={`px-2 py-0.5 text-[9px] font-mono ${lineType === t ? "bg-white/[0.10] text-white" : "text-white/40 hover:text-white/70"}`}>
                {t}
              </button>
            ))}
          </div>
          <div className="flex gap-0.5 border border-white/[0.08] rounded overflow-hidden">
            {(["all", "home", "away"] as const).map(v => (
              <button key={v} onClick={() => setVenue(v)}
                className={`px-2 py-0.5 text-[9px] font-mono uppercase ${venue === v ? "bg-white/[0.10] text-white" : "text-white/40 hover:text-white/70"}`}>
                {v}
              </button>
            ))}
          </div>
        </div>
      }
    >
      {rows.length === 0 ? (
        <p className="text-[10px] text-white/40 font-mono">No matching data for this coach yet.</p>
      ) : (
        <>
          <p className="text-[9px] text-white/40 leading-relaxed mb-3">
            P(own line out | opp line on ice). Rows = own lines, cols = opp lines. Brighter = more matched.
          </p>
          <div className="inline-block">
            <div className="grid gap-1" style={{ gridTemplateColumns: `2.5rem repeat(${oppRanks.length}, 4rem)` }}>
              <div />
              {oppRanks.map(r => (
                <div key={`hdr-${r}`} className="text-[9px] font-mono text-white/50 text-center">opp {lineType}{r}</div>
              ))}
              {ownRanks.map(own => (
                <Fragment key={`row-${own}`}>
                  <div className="text-[9px] font-mono text-white/50 self-center">own {lineType}{own}</div>
                  {oppRanks.map(opp => {
                    const v = lookup(own, opp);
                    return (
                      <div key={`${own}-${opp}`}
                        className="rounded border border-white/[0.06] text-center py-2"
                        style={{ background: color(v) }}
                      >
                        <span className="text-[10px] font-mono text-white/90">
                          {v == null ? "—" : `${Math.round(v * 100)}%`}
                        </span>
                      </div>
                    );
                  })}
                </Fragment>
              ))}
            </div>
          </div>
        </>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 4.3 Special teams units
// ---------------------------------------------------------------------------

function StDeploymentPanel({ data }: { data: CoachProfile["st_deployment"] }) {
  if (!data) return null;
  function unitColor(u: string): string {
    if (u === "PP1") return "border-[#4ade80]/40";
    if (u === "PP2") return "border-[#fbbf24]/40";
    if (u === "PK1") return "border-[#60a5fa]/40";
    if (u === "PK2") return "border-[#a78bfa]/40";
    return "border-white/[0.08]";
  }
  return (
    <Card
      ref="4.3"
      title={<>Special Teams Units</>}
      headerRight={data.as_of ? <span className="text-[9px] font-mono text-white/30">{data.as_of}</span> : null}
    >
      {data.units.length === 0 ? (
        <p className="text-[10px] text-white/40 font-mono">No ST data yet.</p>
      ) : (
        <div className="grid gap-2 md:grid-cols-2">
          {data.units.map(u => (
            <div key={u.unit_type} className={`rounded-lg border bg-white/[0.02] px-3 py-2.5 ${unitColor(u.unit_type)}`}>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[11px] font-mono font-semibold text-white/85">{u.unit_type}</span>
                <span className="text-[9px] font-mono text-white/40">
                  {fmtMin(u.unit_toi_secs)} · {fmtPct(u.share_of_st_toi)} · {u.team_st_gp}gp
                </span>
              </div>
              <div className="flex flex-wrap gap-x-2 gap-y-1">
                {u.player_names.map((n, i) => (
                  <a
                    key={u.player_ids[i] ?? i}
                    href={`/players/${encodeURIComponent(n)}`}
                    className="text-[10px] text-white/75 hover:text-white transition-colors"
                  >
                    {n}
                  </a>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 4.4 Timeout usage
// ---------------------------------------------------------------------------

function TimeoutUsagePanel({ data }: { data: CoachProfile["timeout_usage"] }) {
  if (!data) return null;
  return (
    <Card
      ref="4.4"
      title={<>Timeout Usage</>}
      headerRight={
        data.rows.length === 0
          ? <SkeletonBadge />
          : (data.as_of ? <span className="text-[9px] font-mono text-white/30">{data.as_of}</span> : null)
      }
    >
      {data.rows.length === 0 ? (
        <p className="text-[10px] text-white/40 font-mono">
          PBP ingester doesn&apos;t yet capture team timeouts. The model is built and writes an
          empty parquet; populates as soon as the stoppage-subtype is extended in
          <span className="text-white/60"> data/pbp_parser.py</span>.
        </p>
      ) : (
        <div className="space-y-1">
          {data.rows.map((r, i) => (
            <div key={i}
              className="grid grid-cols-[5rem_5rem_5rem_4rem_4rem] items-center gap-3 px-3 py-2 rounded-lg bg-white/[0.02] border border-white/[0.06]"
            >
              <span className="text-[10px] font-mono text-white/55">{r.period_bucket}</span>
              <span className="text-[10px] font-mono text-white/55">{r.score_state}</span>
              <span className="text-[10px] font-mono text-white/55">{r.time_bucket}</span>
              <span className="text-[10px] font-mono text-white/70 text-right">{r.n_timeouts}</span>
              <span className="text-[10px] font-mono text-[#fb923c] text-right">{fmtNum(r.rate_per_game, 3)}</span>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 4.5 Goalie pull
// ---------------------------------------------------------------------------

function GoaliePullPanel({ data }: { data: CoachProfile["goalie_pull"] }) {
  if (!data) return null;
  return (
    <Card
      ref="4.5"
      title={<>Goalie Pull Tendency</>}
      headerRight={data.as_of ? <span className="text-[9px] font-mono text-white/30">{data.as_of}</span> : null}
    >
      {data.rows.length === 0 ? (
        <p className="text-[10px] text-white/40 font-mono">No qualifying pulls observed.</p>
      ) : (
        <>
          <p className="text-[9px] text-white/40 leading-relaxed mb-3">
            Mean seconds remaining when this coach pulls the goalie, by deficit.
            Higher = earlier hook. Excludes &lt; 5s flickers and pre-3rd-period changes.
          </p>
          <div className="space-y-1">
            {data.rows.map(r => (
              <div key={r.deficit}
                className="grid grid-cols-[3rem_1fr_4rem_4rem_4rem] items-center gap-3 px-3 py-2 rounded-lg bg-white/[0.02] border border-white/[0.06]"
              >
                <span className="text-[10px] font-mono font-semibold text-[#f87171]">−{r.deficit}</span>
                <span className="text-[10px] font-mono text-white/55">
                  {r.n_pulls} pulls / {r.n_team_games} GP
                </span>
                <span className="text-[10px] font-mono text-white/40 text-right" title="median seconds remaining">
                  med {fmtSec(r.median_pull_time_secs)}
                </span>
                <span className="text-[10px] font-mono text-[#fb923c] text-right" title="mean seconds remaining">
                  mean {fmtSec(r.mean_pull_time_secs)}
                </span>
                <span className="text-[10px] font-mono text-white/30 text-right" title="earliest single pull">
                  max {fmtSec(r.earliest_pull_secs)}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 4.6 Penalty tendency
// ---------------------------------------------------------------------------

function PenaltyTendencyPanel({ data }: { data: CoachProfile["penalty_tendency"] }) {
  if (!data || !data.row) {
    return (
      <Card ref="4.6" title={<>Penalty Tendency</>}>
        <p className="text-[10px] text-white/40 font-mono">No penalty data for this team.</p>
      </Card>
    );
  }
  const r = data.row;
  const la = data.league_avg;
  function deltaColor(team: number, league: number | undefined): string {
    if (league == null) return "text-white/40";
    return team > league ? "text-[#f87171]" : "text-[#4ade80]";
  }
  return (
    <Card
      ref="4.6"
      title={<>Penalty Tendency</>}
      headerRight={data.as_of ? <span className="text-[9px] font-mono text-white/30">{data.as_of}</span> : null}
    >
      <p className="text-[9px] text-white/40 leading-relaxed mb-3">
        Per-team baseline ({r.ref_dim ?? "team-only"} — referee crew dimension not yet ingested).
      </p>
      <div className="grid grid-cols-3 gap-2 mb-3">
        <StatPill
          label="Penalties / G"
          value={fmtNum(r.penalties_taken_per_game)}
          color={deltaColor(r.penalties_taken_per_game, la?.penalties_taken_per_game)}
        />
        <StatPill
          label="PIM / G"
          value={fmtNum(r.pim_per_game)}
          color={deltaColor(r.pim_per_game, la?.pim_per_game)}
        />
        <StatPill
          label="PP Earned / G"
          value={fmtNum(r.pp_opps_per_game)}
          color={r.pp_opps_per_game > (la?.pp_opps_per_game ?? 0) ? "text-[#4ade80]" : "text-white/55"}
        />
      </div>
      <div className="grid grid-cols-2 gap-3 text-[10px] font-mono text-white/40">
        <div>league avg P/G: <span className="text-white/65">{fmtNum(la?.penalties_taken_per_game)}</span></div>
        <div>total PIM:      <span className="text-white/65">{r.pim_total}</span></div>
        <div>league avg PIM/G: <span className="text-white/65">{fmtNum(la?.pim_per_game)}</span></div>
        <div>n_games:        <span className="text-white/65">{r.n_games}</span></div>
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 4.7 Coach Profile Database (career-with-current-team aggregates)
// ---------------------------------------------------------------------------

function CoachProfileDbPanel({
  meta,
  data,
}: {
  meta: NonNullable<CoachProfile["meta"]>;
  data: CoachProfile["coach_profile"];
}) {
  const row = data?.row ?? null;
  return (
    <Card
      ref="4.7"
      title={<>Coaching Profile</>}
      headerRight={
        row
          ? <span className="text-[9px] font-mono text-white/30">{data?.as_of ?? ""}</span>
          : <SkeletonBadge />
      }
    >
      {row ? (
        <>
          <p className="text-[9px] text-white/40 leading-relaxed mb-3">
            Career-with-{row.team} aggregates over PBP since
            {" "}<span className="text-white/65">{row.first_named_head_coach || "—"}</span>{" "}
            ({row.seasons_covered.join(", ") || "—"}).
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
            <StatPill label="GP"          value={row.gp_under_coach} />
            <StatPill label="Points %"    value={(row.points_pct * 100).toFixed(1) + "%"}
              color={row.points_pct >= 0.55 ? "text-[#4ade80]" : row.points_pct >= 0.45 ? "text-white" : "text-[#f87171]"} />
            <StatPill label="W / OTW / L / OTL"
              value={`${row.wins} / ${row.ot_wins} / ${row.losses} / ${row.ot_losses}`} />
            <StatPill label="Points"      value={row.points} />
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <StatPill label="GF / G"      value={fmtNum(row.gf_per_game, 2)} />
            <StatPill label="GA / G"      value={fmtNum(row.ga_per_game, 2)}
              color={row.ga_per_game < row.gf_per_game ? "text-white" : "text-[#f87171]"} />
            <StatPill label="SF / G"      value={fmtNum(row.sf_per_game, 1)} />
            <StatPill label="SA / G"      value={fmtNum(row.sa_per_game, 1)} />
          </div>
          <div className="grid grid-cols-2 gap-2 mt-2">
            <StatPill label="PP %"        value={(row.pp_pct * 100).toFixed(1) + "%"}
              color={row.pp_pct >= 0.20 ? "text-[#4ade80]" : "text-white/70"} />
            <StatPill label="PK %"        value={(row.pk_pct * 100).toFixed(1) + "%"}
              color={row.pk_pct >= 0.80 ? "text-[#4ade80]" : "text-white/70"} />
          </div>
          <p className="mt-3 text-[9px] text-white/25 font-mono">
            PP% / PK% computed from raw penalty + goal events; coincidental minors
            net low. Multi-season hockey-reference scrape adds career-pre-team
            history in a follow-up.
          </p>
        </>
      ) : (
        <p className="text-[10px] text-white/40 font-mono">
          No aggregated profile for {meta.name} — run
          <span className="text-white/65"> uv run python scripts/gretzky.py train-coach-profile</span>.
        </p>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 4.8 Goalie Coach Model — team save% trajectory + change-point flag
// ---------------------------------------------------------------------------

function _saveSparkline({ pts, threshold = 0.900 }: { pts: number[]; threshold?: number }) {
  if (pts.length < 2) return null;
  const W = 220, H = 56, P = 4;
  const lo = Math.min(threshold - 0.02, ...pts);
  const hi = Math.max(threshold + 0.02, ...pts);
  const span = hi - lo || 1;
  const xs = pts.map((_, i) => P + (i * (W - 2 * P)) / (pts.length - 1));
  const ys = pts.map(v => H - P - ((v - lo) / span) * (H - 2 * P));
  const d = xs.map((x, i) => `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${ys[i].toFixed(1)}`).join(" ");
  const thrY = H - P - ((threshold - lo) / span) * (H - 2 * P);
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden className="block">
      <line x1={P} x2={W - P} y1={thrY} y2={thrY} stroke="rgba(255,255,255,0.10)" strokeDasharray="3 4" />
      <path d={d} fill="none" stroke="#60a5fa" strokeWidth={1.6} />
      {xs.map((x, i) => (
        <circle key={i} cx={x} cy={ys[i]} r={2.2}
          fill={pts[i] >= threshold ? "#4ade80" : "#f87171"}
          stroke="rgba(0,0,0,0.5)" strokeWidth={0.6} />
      ))}
    </svg>
  );
}

function GoalieCoachPanel({ data }: { data: CoachProfile["goalie_coach"] }) {
  const row = data?.row ?? null;
  return (
    <Card
      ref="4.8"
      title={<>Goalie Coach Curve</>}
      headerRight={
        row
          ? <span className="text-[9px] font-mono text-white/30">{data?.as_of ?? ""}</span>
          : <SkeletonBadge />
      }
    >
      {row ? (
        <>
          <p className="text-[9px] text-white/40 leading-relaxed mb-3">
            Team save% trajectory.  Split Δ = late {(row.gp >= 30 ? 15 : 5)}-game window
            minus early window. Bright = change-point flag → trigger accelerated Bayesian
            update on goalie ratings.
          </p>
          <div className="grid grid-cols-3 gap-2 mb-3">
            <StatPill label="Season SV%"
              value={row.season_save_pct != null ? row.season_save_pct.toFixed(3) : "—"}
              color={row.season_save_pct != null && row.season_save_pct >= 0.905 ? "text-[#4ade80]"
                : row.season_save_pct != null && row.season_save_pct >= 0.890 ? "text-white" : "text-[#f87171]"} />
            <StatPill label="vs Prior Season"
              value={row.save_pct_delta != null ? (row.save_pct_delta >= 0 ? "+" : "") + row.save_pct_delta.toFixed(3) : "—"}
              color={row.save_pct_delta != null && row.save_pct_delta > 0 ? "text-[#4ade80]"
                : row.save_pct_delta != null && row.save_pct_delta < 0 ? "text-[#f87171]" : "text-white/55"} />
            <StatPill label={row.change_point_detected ? "Split Δ · CHANGE" : "Split Δ"}
              value={row.split_delta != null ? (row.split_delta >= 0 ? "+" : "") + row.split_delta.toFixed(3) : "—"}
              color={row.change_point_detected ? "text-[#fbbf24]" : "text-white/70"} />
          </div>
          {row.rolling_save_pct.length >= 2 ? (
            <>
              <div className="mb-1">{_saveSparkline({ pts: row.rolling_save_pct })}</div>
              <div className="flex justify-between text-[8px] font-mono text-white/30">
                <span>early</span><span>league avg ≈ .900</span><span>late</span>
              </div>
            </>
          ) : (
            <p className="text-[9px] text-white/30 font-mono">Not enough games for a rolling trace.</p>
          )}
          <p className="mt-3 text-[9px] text-white/25 font-mono">
            Named goalie coach pending; the
            <span className="text-white/55"> coaches.json </span>
            schema reserves a slot.
          </p>
        </>
      ) : (
        <p className="text-[10px] text-white/40 font-mono">
          No goalie coach curve yet — run
          <span className="text-white/65"> uv run python scripts/gretzky.py train-goalie-coach</span>.
        </p>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 4.9 / 4.10 PP / PK coordinator
// 4.9 → real data; 4.10 → still a skeleton until the PK coordinator model ships.
// ---------------------------------------------------------------------------

function PpCoordinatorPanel({ data }: { data?: CoachProfile["pp_coordinator"] }) {
  const row = data?.row ?? null;
  return (
    <Card
      ref="4.9"
      title={<>PP Coordinator</>}
      headerRight={
        row
          ? <span className="text-[9px] font-mono text-white/30">{data?.as_of ?? ""}</span>
          : <SkeletonBadge />
      }
    >
      <p className="text-[9px] text-white/40 leading-relaxed mb-3">
        System signature: shot quality vs. volume · zone entry · PP1 QB usage.
      </p>
      {row ? (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
            <StatPill label="xG / 60"
              value={fmtNum(row.pp_xg_per_60, 2)}
              color={row.pp_xg_per_60 >= 12 ? "text-[#4ade80]" : "text-white"} />
            <StatPill label="Shots / 60" value={fmtNum(row.pp_shots_per_60, 1)} />
            <StatPill label="G / 60"
              value={fmtNum(row.pp_goals_per_60, 2)}
              color={row.pp_goals_per_60 >= 10 ? "text-[#4ade80]" : "text-white"} />
            <StatPill label="xG / Shot"  value={fmtNum(row.pp_xg_per_shot, 3)} />
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            <StatPill label="Avg Shot Dist (ft)" value={fmtNum(row.pp_shot_distance_avg, 1)} />
            <StatPill label="Carry %"
              value={row.pp_carry_pct == null ? "—" : (row.pp_carry_pct * 100).toFixed(1) + "%"}
              color="text-white/55" />
            <StatPill label="PP1 Share of PP"
              value={(row.pp1_qb_share * 100).toFixed(1) + "%"} />
          </div>
          <div className="mt-3 px-3 py-2 rounded-lg bg-white/[0.02] border border-white/[0.06] flex items-center gap-2">
            <span className="text-[9px] font-mono uppercase tracking-[0.18em] text-white/35 w-20 shrink-0">PP1 QB</span>
            {row.pp1_qb_name ? (
              <a
                href={`/players/${encodeURIComponent(row.pp1_qb_name)}`}
                className="text-[11px] text-white/85 hover:text-white transition-colors"
              >
                {row.pp1_qb_name}
              </a>
            ) : (
              <span className="text-[11px] text-white/30 italic">none (no D on PP1)</span>
            )}
            <span className="ml-auto text-[9px] font-mono text-white/30">
              {row.pp_shots} sh · {row.pp_goals} G · {row.pp_team_gp} GP
            </span>
          </div>
          {row.pp_carry_pct == null && (
            <p className="mt-2 text-[9px] text-white/25 font-mono">
              Carry% pending — the PBP ingester does not yet populate
              <span className="text-white/55"> carry_in </span>
              on zone-entry events.
            </p>
          )}
        </>
      ) : (
        <p className="text-[10px] text-white/40 font-mono">
          No PP coordinator signature — run
          <span className="text-white/65"> uv run python scripts/gretzky.py train-pp-coordinator</span>.
        </p>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 4.11 Coaching Style Radar (skeleton — 8 dimensions)
// ---------------------------------------------------------------------------

const STYLE_DIMS = [
  "Forecheck",   // forecheck aggression
  "DZ Structure",
  "Pace",
  "Physicality",
  "OZ Structure",
  "NZ Tendency",
  "Line Match",  // line shelter vs. match
  "ST Aggression",
];

// ---------------------------------------------------------------------------
// 4.10 PK Coordinator — real data
// ---------------------------------------------------------------------------

function PkCoordinatorPanel({ data }: { data?: CoachProfile["pk_coordinator"] }) {
  const row = data?.row ?? null;
  return (
    <Card
      ref="4.10"
      title={<>PK Coordinator</>}
      headerRight={
        row
          ? <span className="text-[9px] font-mono text-white/30">{data?.as_of ?? ""}</span>
          : <SkeletonBadge />
      }
    >
      <p className="text-[9px] text-white/40 leading-relaxed mb-3">
        PK structure · forecheck pressure · SH rush tendency.
      </p>
      {row ? (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
            <StatPill label="PK SV%"
              value={(row.pk_save_pct * 100).toFixed(1) + "%"}
              color={row.pk_save_pct >= 0.82 ? "text-[#4ade80]" : row.pk_save_pct >= 0.78 ? "text-white" : "text-[#f87171]"} />
            <StatPill label="SA / 60"  value={fmtNum(row.pk_sa_per_60, 1)} />
            <StatPill label="xGA / 60" value={fmtNum(row.pk_xga_per_60, 2)} />
            <StatPill label="GA / 60"  value={fmtNum(row.pk_ga_per_60, 2)} />
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            <StatPill label="xGA / Shot" value={fmtNum(row.pk_xga_per_shot, 3)} />
            <StatPill label="SH Shots / 60"
              value={fmtNum(row.sh_shots_per_60, 1)}
              color={row.sh_shots_per_60 >= 20 ? "text-[#4ade80]" : "text-white/55"} />
            <StatPill label="PK1 Share" value={(row.pk1_share * 100).toFixed(1) + "%"} />
          </div>
          <p className="mt-3 text-[9px] text-white/25 font-mono">
            {row.pk_sa} SA · {row.pk_ga} GA · {row.sh_goals_for} SHG · {row.pk_team_gp} GP
          </p>
        </>
      ) : (
        <p className="text-[10px] text-white/40 font-mono">
          No PK coordinator data — run
          <span className="text-white/65"> uv run python scripts/gretzky.py train-pk-coordinator</span>.
        </p>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 4.11 Coaching Style Vector — real radar
// ---------------------------------------------------------------------------

function StyleRadarPanel({ data }: { data?: CoachProfile["coaching_style"] }) {
  const row = data?.row ?? null;
  const dimKeys: (keyof NonNullable<CoachingStyleRow["dimensions"]>)[] = [
    "forecheck_aggression", "dz_structure", "pace", "physicality",
    "oz_structure", "nz_tendency", "line_match", "st_aggression",
  ];
  const vals = dimKeys.map(k => {
    const v = row?.dimensions?.[k]?.rank;
    return v != null ? v : 0.5;
  });

  const cx = 110, cy = 110, R = 88;
  const n = STYLE_DIMS.length;
  const angleFor = (i: number) => -Math.PI / 2 + (2 * Math.PI * i) / n;
  const polyPoints = vals
    .map((v, i) => {
      const a = angleFor(i);
      const r = v * R;
      return `${cx + Math.cos(a) * r},${cy + Math.sin(a) * r}`;
    })
    .join(" ");
  const ringsR = [R * 0.25, R * 0.5, R * 0.75, R];

  return (
    <Card
      ref="4.11"
      title={<>Coaching Style Vector</>}
      headerRight={
        row
          ? <span className="text-[9px] font-mono text-white/30">{data?.as_of ?? ""}</span>
          : <SkeletonBadge />
      }
    >
      <p className="text-[9px] text-white/40 leading-relaxed mb-3">
        8-dim system vector extracted from play-by-play. Rank ∈ [0, 1] across the league.
        Drives roster fit (4.12) and the per-coach decision net (4.17).
      </p>
      <div className="flex items-start gap-4 flex-wrap">
        <svg width={220} height={220} viewBox="0 0 220 220" aria-hidden className="shrink-0">
          {ringsR.map((r, i) => (
            <circle key={i} cx={cx} cy={cy} r={r} fill="none" stroke="rgba(251,146,60,0.15)" strokeWidth={0.8} />
          ))}
          {STYLE_DIMS.map((label, i) => {
            const a = angleFor(i);
            const x2 = cx + Math.cos(a) * R;
            const y2 = cy + Math.sin(a) * R;
            const lx = cx + Math.cos(a) * (R + 12);
            const ly = cy + Math.sin(a) * (R + 12);
            return (
              <g key={label}>
                <line x1={cx} y1={cy} x2={x2} y2={y2} stroke="rgba(255,255,255,0.06)" strokeWidth={0.8} />
                <text x={lx} y={ly}
                  fontSize={8} fontFamily="monospace"
                  fill="rgba(255,255,255,0.45)"
                  textAnchor="middle" dominantBaseline="middle"
                >{label}</text>
              </g>
            );
          })}
          <polygon points={polyPoints}
            fill={row ? "rgba(251,146,60,0.20)" : "rgba(167,139,250,0.18)"}
            stroke={row ? "rgba(251,146,60,0.65)" : "rgba(167,139,250,0.55)"}
            strokeWidth={1.2}
            strokeDasharray={row ? "none" : "3 3"} />
        </svg>

        <div className="flex-1 grid grid-cols-2 gap-x-3 gap-y-1.5 text-[10px] font-mono min-w-[200px]">
          {STYLE_DIMS.map((label, i) => {
            const rank = vals[i];
            const rawVal = row?.dimensions?.[dimKeys[i]]?.raw;
            return (
              <div key={label} className="flex items-center gap-2">
                <span className="text-white/40 w-24 truncate">{label}</span>
                <div className="flex-1 h-1 rounded-full bg-white/[0.05] overflow-hidden">
                  <div className="h-full bg-[#fb923c]/60" style={{ width: `${rank * 100}%` }} />
                </div>
                <span className="text-white/45 w-8 text-right">{rank.toFixed(2)}</span>
                {rawVal != null && (
                  <span className="text-white/20 w-10 text-right text-[8px]">{typeof rawVal === "number" ? rawVal.toFixed(1) : "—"}</span>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 4.12 Roster Fit Score — real data
// ---------------------------------------------------------------------------

function RosterFitPanel({ data }: { data?: CoachProfile["roster_fit"] }) {
  const row = data?.row ?? null;
  const fitColor = row
    ? row.fit_score >= 0.50 ? "text-[#4ade80]"
    : row.fit_score >= 0.40 ? "text-[#fbbf24]"
    : "text-[#f87171]"
    : "text-white/30";

  return (
    <Card
      ref="4.12"
      title={<>Roster Fit Score</>}
      headerRight={
        row
          ? <span className="text-[9px] font-mono text-white/30">{data?.as_of ?? ""}</span>
          : <SkeletonBadge />
      }
    >
      <p className="text-[9px] text-white/40 leading-relaxed mb-3">
        Style vector (4.11) x roster archetype composition (2.11). Below 0.4 = system/roster
        mismatch → performance drag.
      </p>
      <div className="flex items-center gap-3 mb-3">
        <span className={`text-[28px] font-mono font-bold ${fitColor}`}>
          {row ? row.fit_score.toFixed(3) : "—"}
        </span>
        <div className="flex-1">
          <div className="h-2 rounded-full bg-white/[0.05] overflow-hidden">
            <div className="h-full bg-gradient-to-r from-[#f87171]/50 via-[#fbbf24]/50 to-[#4ade80]/50"
              style={{ width: row ? `${row.fit_score * 100}%` : "50%" }} />
          </div>
          <div className="mt-1 flex justify-between text-[8px] font-mono text-white/30">
            <span>0.0 mismatch</span>
            <span>0.5 neutral</span>
            <span>1.0 aligned</span>
          </div>
        </div>
      </div>
      {row && (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mb-3">
            <StatPill label="Top Archetype" value={row.archetype_top || "—"} />
            <StatPill label="Weakest Dim" value={row.mismatch_dim || "—"} color="text-[#f87171]" />
            <StatPill label="Mismatch Supp" value={fmtNum(row.mismatch_support, 2)} color="text-white/55" />
          </div>
          {row.archetypes.length > 0 && (
            <div className="space-y-1">
              {row.archetypes.map((a, i) => (
                <div key={a} className="flex items-center gap-2 px-3 py-1.5 rounded bg-white/[0.02] border border-white/[0.06]">
                  <span className="text-[10px] font-mono text-white/60 flex-1">{a}</span>
                  <div className="w-24 h-1 rounded-full bg-white/[0.05] overflow-hidden">
                    <div className="h-full bg-[#a78bfa]/50" style={{ width: `${(row.archetype_shares[i] || 0) * 100}%` }} />
                  </div>
                  <span className="text-[9px] font-mono text-white/40 w-10 text-right">
                    {((row.archetype_shares[i] || 0) * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 4.13 Staff Change Detector + 4.14 FO Regime + 4.15 Buyer/Seller
// ---------------------------------------------------------------------------

function StaffChangePanel({ staffData, foData }: {
  staffData?: CoachProfile["staff_changes"];
  foData?: CoachProfile["fo_regime_changes"];
}) {
  const staffRows = staffData?.rows ?? [];
  const foRows    = foData?.rows ?? [];
  const hasData   = staffRows.length > 0 || foRows.length > 0;
  return (
    <Card
      ref="4.13"
      title={<>Staff / FO Change Alerts</>}
      headerRight={
        hasData
          ? <span className="text-[9px] font-mono text-white/30">{staffData?.as_of ?? foData?.as_of ?? ""}</span>
          : <span className="text-[9px] font-mono text-white/30">{staffData?.as_of ?? "scanned"}</span>
      }
    >
      <p className="text-[9px] text-white/40 leading-relaxed mb-3">
        Mid-season coaching (4.13) and front-office (4.14) changes from the transactions feed.
        Each triggers the regime change pipeline (2.14) with the listed decay window.
      </p>
      {!hasData ? (
        <div className="px-3 py-3 rounded-lg bg-white/[0.02] border border-white/[0.06] border-dashed text-center">
          <p className="text-[10px] text-white/40 font-mono">No staff or FO changes detected for this team.</p>
        </div>
      ) : (
        <div className="space-y-1.5">
          {staffRows.map((r, i) => (
            <div key={`s-${i}`} className="flex items-center gap-3 px-3 py-2 rounded-lg bg-white/[0.02] border border-[#f87171]/30">
              <span className="text-[9px] font-mono text-white/35 w-20 shrink-0">{r.date}</span>
              <span className="h-1.5 w-1.5 rounded-full bg-[#f87171] shadow-[0_0_6px_rgba(248,113,113,0.7)]" />
              <span className="text-[10px] text-white/75 flex-1">{r.change_type.replace("_", " ")} — {r.person_out || "unknown"}</span>
              <span className="text-[9px] font-mono text-white/30">decay {r.decay_games}g</span>
            </div>
          ))}
          {foRows.map((r, i) => (
            <div key={`f-${i}`} className="flex items-center gap-3 px-3 py-2 rounded-lg bg-white/[0.02] border border-[#fbbf24]/30">
              <span className="text-[9px] font-mono text-white/35 w-20 shrink-0">{r.date}</span>
              <span className="h-1.5 w-1.5 rounded-full bg-[#fbbf24] shadow-[0_0_6px_rgba(251,191,36,0.7)]" />
              <span className="text-[10px] text-white/75 flex-1">{r.fo_role.replace(/_/g, " ")} — {r.person_out || "unknown"}</span>
              <span className="text-[9px] font-mono text-white/30">decay {r.decay_games}g</span>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

function BuyerSellerPanel({ data }: { data?: CoachProfile["buyer_seller"] }) {
  const row = data?.row ?? null;
  const classColor = row
    ? row.classification === "buyer"  ? "text-[#4ade80]"
    : row.classification === "seller" ? "text-[#f87171]"
    : "text-[#fbbf24]"
    : "text-white/30";
  const classBg = row
    ? row.classification === "buyer"  ? "border-[#4ade80]/30 bg-[#4ade80]/[0.04]"
    : row.classification === "seller" ? "border-[#f87171]/30 bg-[#f87171]/[0.04]"
    : "border-[#fbbf24]/30 bg-[#fbbf24]/[0.04]"
    : "border-white/[0.06]";
  return (
    <Card
      ref="4.15"
      title={<>Buyer / Seller</>}
      headerRight={
        row
          ? <span className="text-[9px] font-mono text-white/30">{data?.as_of ?? ""}</span>
          : <SkeletonBadge />
      }
    >
      {row ? (
        <>
          <div className={`rounded-lg border px-4 py-3 mb-3 ${classBg}`}>
            <div className="flex items-center gap-3">
              <span className={`text-[18px] font-mono font-bold uppercase ${classColor}`}>
                {row.classification}
              </span>
              <span className="text-[10px] font-mono text-white/40">
                conf {(row.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-[9px] text-white/35 mt-1 font-mono">
              P% {(row.points_pct * 100).toFixed(1)} · threshold {(row.threshold * 100).toFixed(1)} · gap {row.gap >= 0 ? "+" : ""}{(row.gap * 100).toFixed(1)} · {row.gp} GP
            </p>
          </div>
          <p className="text-[9px] text-white/25 font-mono">
            V1 uses standings P% only. V2 adds cap space, UFA count, prospect depth.
          </p>
        </>
      ) : (
        <p className="text-[10px] text-white/40 font-mono">
          No buyer/seller data — run
          <span className="text-white/65"> uv run python scripts/gretzky.py classify-buyer-seller</span>.
        </p>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 4.16 Seller Motivation + 4.17 Decision Net + 4.18 GM Fingerprint
// ---------------------------------------------------------------------------

function SellerMotivationPill({ data }: { data?: CoachProfile["seller_motivation"] }) {
  const row = data?.row ?? null;
  if (!row || row.seller_drag <= 0) return null;
  return (
    <div className="rounded-lg border border-[#f87171]/30 bg-[#f87171]/[0.04] px-3 py-2">
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-mono font-semibold text-[#f87171]">SELLER DRAG</span>
        <span className="text-[12px] font-mono font-bold text-[#f87171]">{(row.seller_drag * 100).toFixed(1)}%</span>
        <span className="text-[9px] font-mono text-white/30 ml-auto">eff x{row.efficiency_multiplier.toFixed(3)} · ~{row.games_since_deadline}g post-DL</span>
      </div>
    </div>
  );
}

function DecisionNetPanel({ data }: { data?: CoachProfile["coach_decision_net"] }) {
  const row = data?.row ?? null;
  const dims: { label: string; key: keyof CoachDecisionRow; }[] = [
    { label: "Timeout Aggression", key: "timeout_aggression" },
    { label: "Pull Aggression",    key: "pull_aggression" },
    { label: "Line Shelter",       key: "line_shelter_score" },
    { label: "ST 1st-Unit Lean",   key: "st_first_unit_lean" },
    { label: "Penalty Discipline", key: "penalty_discipline" },
    { label: "Matching Intensity", key: "matching_intensity" },
  ];
  return (
    <Card
      ref="4.17"
      title={<>Decision Profile</>}
      headerRight={
        row
          ? <span className="text-[9px] font-mono text-white/30">{data?.as_of ?? ""}</span>
          : <SkeletonBadge />
      }
    >
      <p className="text-[9px] text-white/40 leading-relaxed mb-3">
        V1: league-percentile-ranked decision profile from Phase 4 aggregates.
        V2 replaces with hierarchical neural net at simulation time.
      </p>
      {row ? (
        <>
          <div className="flex items-center gap-3 mb-3 px-3 py-2 rounded-lg bg-white/[0.02] border border-white/[0.06]">
            <span className="text-[9px] font-mono uppercase tracking-[0.18em] text-white/35">Overall Aggression</span>
            <div className="flex-1 h-1.5 rounded-full bg-white/[0.05] overflow-hidden">
              <div className="h-full bg-[#fb923c]/60" style={{ width: `${row.overall_aggression * 100}%` }} />
            </div>
            <span className="text-[12px] font-mono font-semibold text-[#fb923c]">{row.overall_aggression.toFixed(2)}</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {dims.map(d => {
              const v = row[d.key] as number;
              return (
                <div key={d.label}
                  className="flex items-center gap-3 px-3 py-2 rounded-lg bg-white/[0.02] border border-white/[0.06]"
                >
                  <span className="text-[10px] font-mono text-white/55 flex-1">{d.label}</span>
                  <div className="w-20 h-1 rounded-full bg-white/[0.05] overflow-hidden">
                    <div className="h-full bg-[#a78bfa]/50" style={{ width: `${v * 100}%` }} />
                  </div>
                  <span className="text-[10px] font-mono text-white/65 w-8 text-right">{v.toFixed(2)}</span>
                </div>
              );
            })}
          </div>
        </>
      ) : (
        <p className="text-[10px] text-white/40 font-mono">
          No decision profile — run
          <span className="text-white/65"> uv run python scripts/gretzky.py train-decision-net</span>.
        </p>
      )}
    </Card>
  );
}

function GmFingerprintPanel({ data }: { data?: CoachProfile["gm_fingerprint"] }) {
  const row = data?.row ?? null;
  const archs = ["stand_pat", "add_rental", "sell_veteran", "rebuild", "package_deal"] as const;
  const archLabels: Record<string, string> = {
    stand_pat: "Stand Pat", add_rental: "Add Rental", sell_veteran: "Sell Veteran",
    rebuild: "Rebuild", package_deal: "Package Deal",
  };
  return (
    <Card
      ref="4.18"
      title={<>GM Fingerprint</>}
      headerRight={
        row
          ? <span className="text-[9px] font-mono text-white/30">{data?.as_of ?? ""}</span>
          : <SkeletonBadge />
      }
    >
      {row ? (
        <>
          <div className="flex items-center gap-3 mb-3 px-3 py-2 rounded-lg bg-white/[0.02] border border-white/[0.06]">
            <span className="text-[12px] font-mono font-semibold text-[#fb923c] uppercase">{archLabels[row.action_archetype] ?? row.action_archetype}</span>
            {row.gm_name && <span className="text-[10px] text-white/50 ml-auto">{row.gm_name}</span>}
          </div>
          <div className="space-y-1 mb-3">
            {archs.map(a => {
              const prob = row[`prob_${a}` as keyof GmFingerprintRow] as number;
              return (
                <div key={a} className="flex items-center gap-2 px-3 py-1 rounded bg-white/[0.02] border border-white/[0.06]">
                  <span className="text-[9px] font-mono text-white/50 w-24">{archLabels[a]}</span>
                  <div className="flex-1 h-1 rounded-full bg-white/[0.05] overflow-hidden">
                    <div className="h-full bg-[#a78bfa]/50" style={{ width: `${prob * 100}%` }} />
                  </div>
                  <span className="text-[9px] font-mono text-white/40 w-8 text-right">{(prob * 100).toFixed(0)}%</span>
                </div>
              );
            })}
          </div>
          <div className="grid grid-cols-2 gap-2">
            <StatPill label="DL Aggression" value={row.deadline_aggression.toFixed(2)} />
            <StatPill label="Recent Tx" value={row.recent_tx_count} />
          </div>
        </>
      ) : (
        <p className="text-[10px] text-white/40 font-mono">
          No GM fingerprint — run
          <span className="text-white/65"> uv run python scripts/gretzky.py gm-fingerprint</span>.
        </p>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 4.19 Venue Atmosphere + 4.20 Playoff Elimination
// ---------------------------------------------------------------------------

function VenuePanel({ team, data }: { team: string; data?: CoachProfile["venue_atmosphere"] }) {
  const row = data?.row ?? null;
  const scareColor = row
    ? row.scare_rank >= 0.75 ? "text-[#f87171]" : row.scare_rank >= 0.40 ? "text-[#fbbf24]" : "text-[#4ade80]"
    : "text-white/30";
  return (
    <Card
      ref="4.19"
      title={<>Home Venue Atmosphere</>}
      headerRight={
        row
          ? <span className="text-[9px] font-mono text-white/30">{data?.as_of ?? ""}</span>
          : <SkeletonBadge />
      }
    >
      {row ? (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
            <StatPill label="Scare Factor" value={row.scare_factor >= 0 ? "+" + row.scare_factor.toFixed(2) : row.scare_factor.toFixed(2)} color={scareColor} />
            <StatPill label="vSV% delta" value={(row.visiting_sv_delta >= 0 ? "+" : "") + (row.visiting_sv_delta * 100).toFixed(2) + "%"}
              color={row.visiting_sv_delta < 0 ? "text-[#f87171]" : "text-[#4ade80]"} />
            <StatPill label="vFOW% delta" value={(row.visiting_fow_delta >= 0 ? "+" : "") + (row.visiting_fow_delta * 100).toFixed(1) + "%"}
              color={row.visiting_fow_delta < 0 ? "text-[#f87171]" : "text-[#4ade80]"} />
            <StatPill label="Ref PP delta" value={(row.ref_pp_delta >= 0 ? "+" : "") + row.ref_pp_delta.toFixed(2) + "/g"}
              color={row.ref_pp_delta > 0 ? "text-[#4ade80]" : "text-white/55"} />
          </div>
          <p className="text-[9px] text-white/25 font-mono">
            rank {(row.scare_rank * 100).toFixed(0)}th pctile · {row.home_gp} home GP · {TEAM_FULL_NAMES[team] ?? team}
          </p>
        </>
      ) : (
        <p className="text-[10px] text-white/40 font-mono">
          No venue data — run <span className="text-white/65">uv run python scripts/gretzky.py venue-atmosphere</span>.
        </p>
      )}
    </Card>
  );
}

function PlayoffEliminationPill({ data }: { data?: CoachProfile["playoff_elimination"] }) {
  const row = data?.row ?? null;
  if (!row || row.elimination_drag <= 0) return null;
  return (
    <div className="rounded-lg border border-[#fbbf24]/30 bg-[#fbbf24]/[0.04] px-3 py-2">
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-mono font-semibold text-[#fbbf24]">ELIMINATION DRAG</span>
        <span className="text-[12px] font-mono font-bold text-[#fbbf24]">{(row.elimination_drag * 100).toFixed(0)}%</span>
        <span className="text-[9px] font-mono text-white/30 ml-auto">
          P(playoff) {(row.playoff_prob * 100).toFixed(0)}% · eff x{row.efficiency_multiplier.toFixed(3)} · {row.games_remaining}g left
        </span>
      </div>
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

  useEffect(() => {
    if (!rawName) return;
    setProfile(null);
    fetch(`/api/coaches/${encodeURIComponent(rawName)}`)
      .then(r => r.json())
      .then(setProfile)
      .catch(() => setProfile({ status: "not_found", name: rawName }));
  }, [rawName]);

  const today = new Date().toISOString().slice(0, 10);

  if (profile === null) {
    return (
      <main className="relative min-h-screen p-3 sm:p-6">
        <HudGrid />
        <Spinner />
      </main>
    );
  }

  if (profile.status === "not_found" || !profile.meta) {
    return (
      <main className="relative min-h-screen p-3 sm:p-6">
        <HudGrid />
        <div className="mb-4 hud-panel hud-panel--all-corners p-6">
          <p className="text-white/55 text-sm">No coach found for &ldquo;{rawName}&rdquo;.</p>
          <p className="text-white/30 text-[10px] mt-1 font-mono">Add them to data/coaches.json.</p>
        </div>
      </main>
    );
  }

  const meta = profile.meta;

  return (
    <main className="relative min-h-screen p-3 sm:p-6">
      <HudGrid />

      <div className="relative z-10 mb-3 flex items-center gap-2 flex-wrap">
        <span className="hud-mono text-[10px] uppercase tracking-[0.20em] text-[#fb923c]" aria-hidden>◢</span>
        <span className="hud-mono text-[10px] uppercase tracking-[0.20em] text-[#fb923c]">
          COACH PROFILE · {meta.team}
        </span>
        <span className="hud-mono text-[9px] uppercase tracking-[0.16em] text-[var(--text-secondary)]">
          · phase 4 dossier
        </span>
        <span className="ml-auto text-[#777] text-xs font-mono">{today}</span>
      </div>

      <div className="space-y-4">
        <Hero meta={meta} />

        {/* ──────────────────  TACTICS  ────────────────── */}
        <SectionLabel accent="#fb923c">TACTICS · 4.1 – 4.3</SectionLabel>
        <LineDeploymentPanel data={profile.line_deployment} />
        <div className="grid gap-4 md:grid-cols-2">
          <MatchingPanel data={profile.line_matching} team={meta.team} />
          <StDeploymentPanel data={profile.st_deployment} />
        </div>

        {/* ──────────────────  IN-GAME  ────────────────── */}
        <SectionLabel accent="#fb923c">IN-GAME DECISIONS · 4.4 – 4.6, 4.17</SectionLabel>
        <div className="grid gap-4 md:grid-cols-2">
          <GoaliePullPanel data={profile.goalie_pull} />
          <PenaltyTendencyPanel data={profile.penalty_tendency} />
        </div>
        <TimeoutUsagePanel data={profile.timeout_usage} />
        <DecisionNetPanel data={profile.coach_decision_net} />

        {/* ──────────────────  STAFF  ────────────────── */}
        <SectionLabel accent="#a78bfa">STAFF · 4.7 – 4.10, 4.13</SectionLabel>
        <CoachProfileDbPanel meta={meta} data={profile.coach_profile} />
        <div className="grid gap-4 md:grid-cols-2">
          <PpCoordinatorPanel data={profile.pp_coordinator} />
          <PkCoordinatorPanel data={profile.pk_coordinator} />
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <GoalieCoachPanel data={profile.goalie_coach} />
          <StaffChangePanel staffData={profile.staff_changes} foData={profile.fo_regime_changes} />
        </div>

        {/* ──────────────────  IDENTITY  ────────────────── */}
        <SectionLabel accent="#a78bfa">IDENTITY · 4.11 – 4.12, 4.15 – 4.16, 4.18 – 4.19</SectionLabel>
        <StyleRadarPanel data={profile.coaching_style} />
        <div className="grid gap-4 md:grid-cols-2">
          <RosterFitPanel data={profile.roster_fit} />
          <BuyerSellerPanel data={profile.buyer_seller} />
        </div>
        <SellerMotivationPill data={profile.seller_motivation} />
        <PlayoffEliminationPill data={profile.playoff_elimination} />
        <div className="grid gap-4 md:grid-cols-2">
          <GmFingerprintPanel data={profile.gm_fingerprint} />
          <VenuePanel team={meta.team} data={profile.venue_atmosphere} />
        </div>
      </div>
    </main>
  );
}
