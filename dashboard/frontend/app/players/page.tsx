"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { TEAM_COLORS, TEAM_SECONDARY, normalizePlayerName } from "@/utils/nhl";
import TeamLogoLink from "@/components/TeamLogoLink";
import { useTheme } from "@/utils/themeContext";
import { SeasonContextPill } from "@/utils/contextToggle";

function darkBlend(hex: string, darkness = 0.88): string {
  const base = [9, 10, 12];
  const c = hex.replace("#", "");
  const r = parseInt(c.slice(0, 2), 16) || 0;
  const g = parseInt(c.slice(2, 4), 16) || 0;
  const b = parseInt(c.slice(4, 6), 16) || 0;
  return `#${[r, g, b].map((v, i) => Math.round(v * (1 - darkness) + base[i] * darkness).toString(16).padStart(2, "0")).join("")}`;
}

function toRgbInts(hex: string): [number, number, number] {
  const c = hex.replace("#", "");
  return [parseInt(c.slice(0, 2), 16) || 0, parseInt(c.slice(2, 4), 16) || 0, parseInt(c.slice(4, 6), 16) || 0];
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface PlayerData {
  not_found?: boolean;
  is_goalie?: boolean;
  player_name?: string;
  player_id?: number | null;
  team?: string;
  position?: string | null;
  shoots?: string | null;
  season?: number;
  shots?: number;
  goals?: number;
  xg_sum?: number;
  finishing?: number | null;
  finishing_per60?: number | null;
  cdr?: number | null;
  rapm_ev_off?: number | null;
  rapm_ev_def?: number | null;
  war?: number | null;
  gar?: number | null;
  archetype_name?: string | null;
  ewma_xgf60?: number | null;
  ewma_form_flag?: string | null;
  hot_hand_score?: number | null;
  hot_hand_goals5?: number | null;
  hot_hand_xg5?: number | null;
  clutch_index?: number | null;
  special_teams_pp?: number | null;
  special_teams_pk?: number | null;
  bayesian_rating?: number | null;
  playoff_delta?: number | null;
  former_team_boost?: number | null;
  former_team?: string | null;
  // goalie
  games_played?: number;
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
}

interface RecentGame {
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

interface GameLogSummary {
  n_games: number;
  goals: number;
  assists: number;
  points: number;
  gpg: number;
  apg: number;
  ppg: number;
}

interface GameLog {
  player_id: number;
  games: RecentGame[];
  summary: GameLogSummary;
  status: string;
}

interface EWMAmover {
  player_id: number | null;
  player_name: string;
  team: string | null;
  ewma_xgf60: number | null;
  delta: number | null;
  n_games: number | null;
  form_flag: string;
}

interface WAREntry {
  player_id: number | null;
  player_name: string;
  team: string | null;
  war: number | null;
  gar: number | null;
  rank: number;
}

interface ChemPair {
  player_a: string;
  player_b: string;
  player_a_id: number | null;
  player_b_id: number | null;
  team: string | null;
  chemistry_delta: number | null;
  model_xgf_pct: number | null;
  games_together: number | null;
}

interface RDITeam {
  team: string | null;
  disruption_index: number | null;
  n_moves_30d: number | null;
  n_moves_72h: number | null;
  as_of_date: string | null;
}

interface InjuryRec {
  player_name: string;
  position: string | null;
  team_code: string | null;
  status: string | null;
  return_date_estimate: string | null;
}

interface Transaction {
  date: string;
  event_type: string;
  team: string;
  secondary_team: string | null;
  description: string;
  player_id_espn: string | null;
}

interface SkaterEntry {
  rank: number;
  id: number;
  name: string;
  team: string;
  position: string;
  value: number | null;
}

interface QoTEntry {
  player_id: number | null;
  player_name: string;
  team: string | null;
  qot: number;
  qoc: number | null;
  gp: number | null;
  toi_ev: number | null;
}

interface EdgeEntry {
  rank: number;
  player_id: number | null;
  player_name: string;
  team: string | null;
  value: number | null;
  games_played: number | null;
}

interface GoalieLeaderEntry {
  rank: number;
  player_id: number | null;
  player_name: string;
  team: string | null;
  value: number | null;
  games_played: number | null;
  gsax: number | null;
}

interface ClutchEntry {
  rank: number;
  player_id: number | null;
  player_name: string;
  team: string | null;
  value: number | null;
  wpa_per60: number | null;
}

interface STEntry {
  rank: number;
  player_id: number | null;
  player_name: string;
  team: string | null;
  xgf60: number | null;
  rapm: number | null;
  toi: number | null;
}

interface HotHandEntry {
  rank: number;
  player_id: number | null;
  player_name: string;
  team: string | null;
  value: number | null;
  goals_5g: number | null;
  xg_5g: number | null;
}

interface XGEntry {
  rank: number;
  player_id: number | null;
  player_name: string;
  team: string | null;
  value: number | null;
  games_played: number | null;
}

interface CDREntry {
  rank: number;
  player_id: number | null;
  player_name: string;
  team: string | null;
  value: number | null;
  xga_60: number | null;
  tk_gv_ratio: number | null;
  gp: number | null;
}

interface RAPMEntry {
  rank: number;
  player_id: number | null;
  player_name: string;
  team: string | null;
  value: number | null;
  gp: number | null;
  toi_ev: number | null;
}

interface XGAEntry {
  rank: number;
  player_id: number | null;
  player_name: string;
  team: string | null;
  value: number | null;
  gp: number | null;
  rapm_ev_def: number | null;
}

// ---------------------------------------------------------------------------
// Tier system
// ---------------------------------------------------------------------------

type Tier = "Elite" | "Above Average" | "Average" | "Below Average" | "Developing";

const TIER_COLOR: Record<Tier, string> = {
  "Elite":         "#d946ef",
  "Above Average": "#4ade80",
  "Average":       "#94a3b8",
  "Below Average": "#fbbf24",
  "Developing":    "#f87171",
};

function finishingTier(v: number): Tier {
  if (v >= 6) return "Elite";
  if (v >= 2) return "Above Average";
  if (v >= -1) return "Average";
  if (v >= -4) return "Below Average";
  return "Developing";
}

function warTier(v: number): Tier {
  if (v >= 2.5) return "Elite";
  if (v >= 1.0) return "Above Average";
  if (v >= 0)   return "Average";
  if (v >= -1)  return "Below Average";
  return "Developing";
}

function defTier(v: number): Tier {
  if (v >= 0.6)  return "Elite";
  if (v >= 0.2)  return "Above Average";
  if (v >= -0.1) return "Average";
  if (v >= -0.4) return "Below Average";
  return "Developing";
}

function stTier(v: number): Tier {
  if (v >= 1.0)  return "Elite";
  if (v >= 0.4)  return "Above Average";
  if (v >= -0.1) return "Average";
  if (v >= -0.5) return "Below Average";
  return "Developing";
}

function gsaxTier(v: number): Tier {
  if (v >= 10) return "Elite";
  if (v >= 3)  return "Above Average";
  if (v >= -3) return "Average";
  if (v >= -8) return "Below Average";
  return "Developing";
}

function hdsvTier(v: number): Tier {
  if (v >= 85) return "Elite";
  if (v >= 80) return "Above Average";
  if (v >= 75) return "Average";
  if (v >= 70) return "Below Average";
  return "Developing";
}

function TierBadge({ tier, small }: { tier: Tier; small?: boolean }) {
  const color = TIER_COLOR[tier];
  return (
    <span
      className={`font-semibold uppercase tracking-wider rounded border shrink-0 ${small ? "text-[8px] px-1.5 py-0.5" : "text-[9px] px-2 py-0.5"}`}
      style={{ color, borderColor: `${color}40`, backgroundColor: `${color}12` }}
    >
      {tier}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Team logo helpers
// ---------------------------------------------------------------------------

const NHL_REMAP: Record<string, string> = { LA: "LAK", NJ: "NJD", SJ: "SJS", TB: "TBL" };

const POS_MAP: Record<string, string> = { L: "LW", R: "RW" };
function fmtPos(pos: string | null | undefined): string {
  if (!pos) return "";
  return POS_MAP[pos.toUpperCase()] ?? pos;
}

function TeamLogo({ team, size = 24, className = "", onTeamClick }: { team: string; size?: number; className?: string; onTeamClick?: () => void }) {
  if (onTeamClick) {
    return <button onClick={onTeamClick} className="cursor-pointer"><TeamLogoLink abbrev={team} size={size} className={className} /></button>;
  }
  return <TeamLogoLink abbrev={team} size={size} className={className} />;
}

// ---------------------------------------------------------------------------
// Form badge
// ---------------------------------------------------------------------------

function FormBadge({ flag, delta }: { flag?: string | null; delta?: number | null }) {
  const isHot  = flag === "hot" || flag === "rising"  || (delta ?? 0) > 1.5;
  const isCold = flag === "cold" || flag === "falling" || (delta ?? 0) < -1.5;
  if (!isHot && !isCold) return null;
  return (
    <span
      className="inline-flex items-center gap-1 text-[10px] font-semibold rounded-full px-2.5 py-1"
      style={{
        color: isHot ? "#4ade80" : "#f87171",
        backgroundColor: isHot ? "rgba(74,222,128,0.1)" : "rgba(248,113,113,0.1)",
        border: `1px solid ${isHot ? "rgba(74,222,128,0.3)" : "rgba(248,113,113,0.3)"}`,
      }}
    >
      {isHot ? "Running Hot" : "Running Cold"}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Recent game mini-log
// ---------------------------------------------------------------------------

function RecentGamesTable({ log, loading }: { log: GameLog | null; loading: boolean }) {
  if (loading) return (
    <div className="mt-3 space-y-1.5">
      {[0,1,2].map(i => (
        <div key={i} className="h-8 rounded-lg bg-white/[0.04] animate-pulse" />
      ))}
    </div>
  );
  if (!log || log.status !== "ok" || log.games.length === 0) return null;

  const s = log.summary;
  return (
    <div className="mt-3">
      {/* Summary line */}
      <p className="text-[10px] font-semibold text-white/50 uppercase tracking-wider mb-2">
        Last {s.n_games} games —{" "}
        <span className="text-white/70">
          {s.goals}G {s.assists}A{" "}
        </span>
        <span className="text-white/40">
          ({s.gpg.toFixed(1)} G/gm · {s.ppg.toFixed(1)} pts/gm)
        </span>
      </p>
      {/* Per-game rows */}
      <div className="space-y-1">
        {log.games.map((g, i) => {
          const scored = g.goals > 0 || g.assists > 0;
          return (
            <div key={i}
              className={`flex items-center gap-1.5 sm:gap-2.5 px-2 sm:px-3 py-1.5 rounded-lg text-[10px] transition-colors
                ${scored ? "bg-white/[0.04] border border-white/[0.07]" : "bg-white/[0.02] border border-white/[0.04]"}`}
            >
              <span className="text-white/30 font-mono w-12 shrink-0 tabular-nums">{g.date.slice(5)}</span>
              <span className="text-white/30 font-mono shrink-0">{g.home_road === "H" ? "vs" : "@"}</span>
              <TeamLogo team={g.opponent} size={14} />
              <span className="text-white/40 font-mono shrink-0 w-8">{g.opponent}</span>
              <span className="flex-1" />
              <span className={`font-semibold tabular-nums w-10 text-right shrink-0 ${scored ? "text-white/80" : "text-white/25"}`}>
                {g.goals}G {g.assists}A
              </span>
              {g.pp_points > 0 && (
                <span className="text-[8px] text-[#fbbf24]/60 font-mono shrink-0">PP</span>
              )}
              <span className="hidden sm:inline text-white/25 font-mono w-12 text-right shrink-0">{g.toi}</span>
              <span className="hidden sm:inline text-white/25 font-mono w-8 text-right shrink-0"
                style={{ color: g.plus_minus > 0 ? "rgba(74,222,128,0.6)" : g.plus_minus < 0 ? "rgba(248,113,113,0.6)" : undefined }}>
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
// Full player detail card
// ---------------------------------------------------------------------------

function PlayerDetailCard({
  data,
  log,
  logLoading,
  warRank,
  onClose,
}: {
  data: PlayerData;
  log: GameLog | null;
  logLoading: boolean;
  warRank: number | null;
  onClose: () => void;
}) {
  const [imgErr, setImgErr] = useState(false);
  const sy = data.season ?? (new Date().getFullYear() - 1);
  const headshotUrl = data.player_id && data.team && !imgErr
    ? `https://assets.nhle.com/mugs/nhl/${sy}${sy + 1}/${data.team}/${data.player_id}.png`
    : null;

  // Derive "why hot/cold" explanation
  const isGoalie = data.is_goalie;
  const hotHandGoals = data.hot_hand_goals5 ?? 0;
  const hotHandXG   = data.hot_hand_xg5 ?? 0;
  const hotScore    = data.hot_hand_score ?? 0;
  const delta       = log?.summary ? (log.summary.ppg - 1) : null; // rough above-avg proxy
  const formFlag    = data.ewma_form_flag;
  const ewmaDelta   = data.ewma_xgf60; // absolute, need context from movers — use vs 4.09 league avg
  const ewmaAbove   = ewmaDelta != null ? ewmaDelta - 4.09 : null;

  let whyHot: string | null = null;
  if (!isGoalie && hotScore > 0.5 && log?.summary && log.summary.n_games > 0) {
    const s = log.summary;
    whyHot = `${s.goals}G ${s.assists}A in last ${s.n_games} games`;
    if (hotHandXG > 0) whyHot += ` · ${hotHandXG.toFixed(1)} xG`;
  } else if (!isGoalie && ewmaAbove != null && Math.abs(ewmaAbove) > 1) {
    whyHot = ewmaAbove > 0
      ? `Generating ${data.ewma_xgf60?.toFixed(1)} xGF/60 — +${ewmaAbove.toFixed(1)} above league average`
      : `Generating only ${data.ewma_xgf60?.toFixed(1)} xGF/60 — ${ewmaAbove.toFixed(1)} below league average`;
  }

  const pTeamColor = TEAM_COLORS[data.team ?? ""] ?? "#888888";
  const pTeamSec   = TEAM_SECONDARY[data.team ?? ""] ?? pTeamColor;
  const cardBase   = darkBlend(pTeamSec, 0.82);
  const cardMid    = darkBlend(pTeamSec, 0.93);
  const [cbr, cbg, cbb] = toRgbInts(cardBase);
  const [cmr, cmg, cmb] = toRgbInts(cardMid);
  const [pr, pg, pb]    = toRgbInts(pTeamColor);

  return (
    <div
      className="rounded-2xl overflow-hidden shadow-[0_12px_60px_rgba(0,0,0,0.75),inset_0_1px_0_rgba(255,255,255,0.08)]"
      style={{
        background: `linear-gradient(160deg, rgba(${cbr},${cbg},${cbb},0.95) 0%, rgba(${cmr},${cmg},${cmb},0.98) 100%)`,
        border: `1px solid rgba(${pr},${pg},${pb},0.20)`,
      }}
    >
      {/* Header */}
      <div className="flex items-start gap-3 px-3 sm:px-5 pt-4 pb-3 border-b border-white/[0.07]">
        {/* Headshot */}
        <div className="shrink-0">
          {headshotUrl ? (
            <img
              src={headshotUrl}
              alt={data.player_name ?? ""}
              onError={() => setImgErr(true)}
              className="h-16 w-16 sm:h-20 sm:w-20 rounded-full object-cover object-top"
              style={{ background: "#111" }}
            />
          ) : (
            <div className="h-16 w-16 sm:h-20 sm:w-20 rounded-full bg-white/10 flex items-center justify-center">
              <span className="text-2xl font-bold text-white/30">{(data.player_name ?? "?")[0]}</span>
            </div>
          )}
        </div>

        {/* Name + meta */}
        <div className="flex-1 min-w-0 pt-1">
          <h2 className="text-[16px] sm:text-xl font-bold text-white leading-tight">{data.player_name}</h2>
          <div className="flex items-center gap-2 mt-1 flex-wrap">
            {data.team && <TeamLogo team={data.team} size={40} />}
            {data.team && <span className="text-sm font-semibold text-white/50">{data.team}</span>}
            {data.position && (
              <span className="text-[10px] font-semibold text-white/35 border border-white/[0.12] rounded px-1.5 py-0.5">{fmtPos(data.position)}</span>
            )}
            {data.shoots && (
              <span className="text-[10px] text-white/25 border border-white/[0.08] rounded px-1.5 py-0.5">Shoots {data.shoots}</span>
            )}
            {data.archetype_name && (
              <span className="text-[10px] font-semibold text-[#a78bfa]/80 border border-[#a78bfa]/20 rounded px-1.5 py-0.5 bg-[#a78bfa]/[0.05]">
                {data.archetype_name}
              </span>
            )}
          </div>
          {/* Form status */}
          <div className="flex items-center gap-2 mt-2 flex-wrap">
            <FormBadge flag={formFlag} delta={ewmaAbove} />
            {hotScore > 0.5 && (
              <span className="text-[10px] font-semibold text-[#a78bfa] bg-[#a78bfa]/10 border border-[#a78bfa]/25 rounded-full px-2.5 py-1">
                Hot Hand
              </span>
            )}
            {warRank && warRank <= 20 && (
              <span className="text-[10px] font-semibold text-[#a78bfa] bg-[#a78bfa]/10 border border-[#a78bfa]/25 rounded-full px-2.5 py-1">
                #{warRank} WAR in NHL
              </span>
            )}
          </div>
        </div>

        {/* Close */}
        <button
          onClick={onClose}
          className="shrink-0 text-white/25 hover:text-white/60 transition-colors text-lg leading-none px-1 pt-1"
        >
          ✕
        </button>
      </div>

      {/* Why hot/cold explanation */}
      {whyHot && (
        <div className={`px-5 py-2.5 border-b border-white/[0.05] text-[11px] font-medium text-center
          ${(formFlag === "hot" || formFlag === "rising" || (ewmaAbove ?? 0) > 0)
            ? "text-[#4ade80]/80 bg-[#4ade80]/[0.04]"
            : "text-[#f87171]/80 bg-[#f87171]/[0.04]"}`}
        >
          {whyHot}
        </div>
      )}

      {/* Stats grid */}
      <div className="px-3 sm:px-5 py-3">
        {isGoalie ? (
          <div className="space-y-0">
            {data.gsax != null && (
              <StatRow label="Goals Saved Above Expected"
                value={`${data.gsax > 0 ? "+" : ""}${data.gsax.toFixed(1)}`}
                tier={gsaxTier(data.gsax)}
                sub="How many more saves than an average goalie" />
            )}
            {data.hdsv_pct != null && (
              <StatRow label="High-Danger Save Rate"
                value={`${((data.hdsv_pct) * 100).toFixed(1)}%`}
                tier={hdsvTier((data.hdsv_pct) * 100)}
                sub="Saves on slot / close-range shots" />
            )}
            {data.sv_pct != null && (
              <StatRow label="Save Percentage"
                value={`.${((data.sv_pct) * 1000).toFixed(0)}`} />
            )}
          </div>
        ) : (
          <div className="space-y-0">
            {data.finishing != null && (
              <StatRow label="Finishing Ability"
                value={`${data.finishing > 0 ? "+" : ""}${data.finishing.toFixed(1)} xG`}
                tier={finishingTier(data.finishing)}
                sub="Goals above what shot quality predicts" />
            )}
            {data.war != null && (
              <StatRow label="Overall Value (WAR)"
                value={`${data.war > 0 ? "+" : ""}${data.war.toFixed(2)}`}
                tier={warTier(data.war)}
                sub={warRank ? `Ranked #${warRank} in the NHL` : "Wins added above replacement"} />
            )}
            {(data.cdr != null || data.rapm_ev_def != null) && (
              <StatRow label="Defensive Value"
                value={`${(data.cdr ?? data.rapm_ev_def ?? 0) > 0 ? "+" : ""}${(data.cdr ?? data.rapm_ev_def ?? 0).toFixed(2)}`}
                tier={defTier(data.cdr ?? data.rapm_ev_def ?? 0)}
                sub="Defensive impact rating (CDR + RAPM)" />
            )}
            {(data.special_teams_pp != null || data.special_teams_pk != null) && (
              <div className="flex items-center justify-between py-2 border-b border-white/[0.05] gap-3">
                <div>
                  <p className="text-sm font-medium text-white/60">Special Teams</p>
                </div>
                <div className="flex items-center gap-3">
                  {data.special_teams_pp != null && (
                    <div className="flex items-center gap-1.5">
                      <span className="text-[9px] text-white/30 font-mono">PP</span>
                      <TierBadge tier={stTier(data.special_teams_pp)} small />
                    </div>
                  )}
                  {data.special_teams_pk != null && (
                    <div className="flex items-center gap-1.5">
                      <span className="text-[9px] text-white/30 font-mono">PK</span>
                      <TierBadge tier={stTier(data.special_teams_pk)} small />
                    </div>
                  )}
                </div>
              </div>
            )}
            {data.playoff_delta != null && Math.abs(data.playoff_delta) > 0.15 && (
              <StatRow label="Playoff Performer"
                value={`${data.playoff_delta > 0 ? "+" : ""}${data.playoff_delta.toFixed(2)} xGF/60`}
                tier={data.playoff_delta > 0.3 ? "Above Average" : data.playoff_delta < -0.3 ? "Below Average" : "Average"}
                sub="Production shift in playoffs vs regular season" />
            )}
            {data.former_team_boost != null && data.former_team && (
              <StatRow label={`vs. Former Team (${data.former_team})`}
                value={`+${(data.former_team_boost * 100).toFixed(0)}% boost`}
                sub="Historical bump when facing former employer" />
            )}
          </div>
        )}
      </div>

      {/* Recent games */}
      {data.player_id && (
        <div className="px-3 sm:px-5 pb-4 border-t border-white/[0.05] pt-3">
          <p className="text-[10px] font-semibold text-white/40 uppercase tracking-wider mb-1">Recent Games</p>
          <RecentGamesTable log={log} loading={logLoading} />
        </div>
      )}
    </div>
  );
}

function StatRow({
  label,
  value,
  tier,
  sub,
}: {
  label: string;
  value: string;
  tier?: Tier;
  sub?: string;
}) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-white/[0.05] last:border-0 gap-3">
      <div className="flex-1 min-w-0">
        <p className="text-[12px] font-medium text-white/60">{label}</p>
        {sub && <p className="text-[9px] text-white/30 mt-0.5 leading-snug">{sub}</p>}
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <span className="text-sm font-semibold font-mono text-white/80">{value}</span>
        {tier && <TierBadge tier={tier} />}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Clickable player row (reused in leaderboard sections)
// ---------------------------------------------------------------------------

function PlayerRow({
  name,
  team,
  playerId,
  badge,
  badgeColor,
  sub,
  onClick,
}: {
  name: string;
  team: string | null;
  playerId: number | null;
  badge?: string;
  badgeColor?: string;
  sub?: string;
  onClick: (name: string) => void;
}) {
  const sy = new Date().getFullYear() - 1;
  const [imgErr, setImgErr] = useState(false);
  const headshotUrl = playerId && team && !imgErr
    ? `https://assets.nhle.com/mugs/nhl/${sy}${sy + 1}/${team}/${playerId}.png`
    : null;
  const teamColor     = TEAM_COLORS[team ?? ""]     ?? "#ffffff";
  const secondaryColor = TEAM_SECONDARY[team ?? ""] ?? "#1a1a2e";
  const ringStyle = {
    boxShadow: `0 0 0 2px ${teamColor}, 0 0 0 2.5px rgba(255,255,255,0.25), 0 0 10px ${teamColor}80`,
  };

  return (
    <button
      onClick={() => onClick(name)}
      className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-150 text-left group"
      style={(() => {
        const sec  = TEAM_SECONDARY[team ?? ""] ?? TEAM_COLORS[team ?? ""] ?? "#333";
        const pri  = TEAM_COLORS[team ?? ""] ?? "#888";
        const [br, bg_, bb] = toRgbInts(darkBlend(sec, 0.84));
        const [mr, mg, mb]  = toRgbInts(darkBlend(sec, 0.93));
        const [pr, pg, pb]  = toRgbInts(pri);
        return {
          background: `linear-gradient(135deg, rgba(${br},${bg_},${bb},0.92) 0%, rgba(${mr},${mg},${mb},0.97) 100%)`,
          border: `1px solid rgba(${pr},${pg},${pb},0.20)`,
        };
      })()}
    >
      {/* Mini headshot with team color ring */}
      <div
        className="rounded-full shrink-0 overflow-hidden"
        style={{ width: 36, height: 36, background: `linear-gradient(to bottom, ${secondaryColor} 0%, ${secondaryColor} 72%, #b0b0b0 100%)`, ...ringStyle }}
      >
        {headshotUrl ? (
          <img
            src={headshotUrl}
            alt={name}
            onError={() => setImgErr(true)}
            className="h-full w-full object-cover"
            style={{ objectPosition: "50% 8%" }}
          />
        ) : (
          <div className="h-full w-full flex items-center justify-center">
            <span className="text-[10px] font-black text-white/40">{name.split(" ").map(w => w[0]).slice(0, 2).join("")}</span>
          </div>
        )}
      </div>

      {/* Name + team */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-white/80 truncate group-hover:text-white transition-colors">{name}</p>
        {sub && <p className="text-[9px] text-white/35 mt-0.5 truncate">{sub}</p>}
      </div>

      {/* Team logo */}
      {team && <TeamLogo team={team} size={32} />}

      {/* Badge */}
      {badge && (
        <span className="text-xs font-bold tabular-nums shrink-0" style={{ color: badgeColor ?? "#94a3b8" }}>
          {badge}
        </span>
      )}

      <span className="text-white/15 group-hover:text-white/40 transition-colors text-xs shrink-0">→</span>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Section wrapper
// ---------------------------------------------------------------------------

function Section({
  title,
  icon,
  count,
  children,
  className = "",
  defaultOpen = false,
}: {
  title: string;
  icon?: string;
  count?: number;
  children: React.ReactNode;
  className?: string;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div
      className={`rounded-2xl overflow-hidden w-full shadow-[0_4px_16px_rgba(0,0,0,0.50)] ${className}`}
      style={{
        background: `linear-gradient(160deg, rgba(var(--card-base-r),var(--card-base-g),var(--card-base-b),0.90) 0%, rgba(var(--card-mid-r),var(--card-mid-g),var(--card-mid-b),0.97) 100%)`,
        border: `1px solid rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.14)`,
      }}
    >
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2.5 px-4 py-3 border-b border-white/[0.06] hover:bg-white/[0.03] transition-colors text-left"
      >
        <span className="w-0.5 h-4 rounded-full shrink-0" style={{ background: `rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.50)` }} />
        <p className="text-[11px] font-black uppercase tracking-[0.12em] text-white/70">{title}</p>
        {count !== undefined && (
          <span className="text-[9px] font-mono text-white/20">{count}</span>
        )}
        <span className="ml-auto text-[#a78bfa]/40 text-[10px] transition-transform duration-200" style={{ transform: open ? "rotate(0deg)" : "rotate(-90deg)" }}>▾</span>
      </button>
      {open && <div className="p-3 space-y-1.5">{children}</div>}
    </div>
  );
}

function SectionEmpty({ msg }: { msg: string }) {
  return <p className="text-[11px] text-white/25 py-2 px-1">{msg}</p>;
}

function SectionLoading() {
  return (
    <div className="space-y-2">
      {[0, 1, 2].map(i => (
        <div key={i} className="h-12 rounded-xl bg-white/[0.03] animate-pulse" />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// InfoTip — hover tooltip for stat explanations
// ---------------------------------------------------------------------------
function InfoTip({ tip }: { tip: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative inline-block">
      <button
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 text-[9px] text-[#a78bfa]/40 hover:text-[#a78bfa]/70 transition-colors mb-1"
      >
        <span className="flex items-center justify-center w-4 h-4 rounded-full border border-[#a78bfa]/30 text-[9px] font-medium"
          style={{ fontFamily: "Georgia, serif" }}>i</span>
        <span className="text-[9px] uppercase tracking-widest">What&apos;s this?</span>
      </button>
      {open && (
        <div className="absolute left-0 top-full mt-1 z-50 w-72 rounded-xl border border-[#a78bfa]/20 bg-[#0d0f13]/97 backdrop-blur-xl shadow-[0_8px_40px_rgba(0,0,0,0.85)] p-3">
          <p className="text-[10px] text-white/50 leading-relaxed">{tip}</p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------
// Cortex about pill + DNA strand
// ---------------------------------------------------------------------------

function CortexAbout() {
  const [open, setOpen] = useState(false);
  const { cortexPinned, setCortexPinned } = useTheme();
  return (
    <div className="mb-4">
      <div className="flex items-center justify-center gap-2">
        <button
          onClick={() => setOpen(o => !o)}
          className={`flex items-center gap-1.5 px-4 py-1.5 rounded-lg border transition-all duration-200 ${
            open
              ? "bg-[#a78bfa]/12 border-[#a78bfa]/35 shadow-[0_0_12px_rgba(167,139,250,0.15)]"
              : "bg-[#a78bfa]/[0.05] border-[#a78bfa]/15 hover:bg-[#a78bfa]/10 hover:border-[#a78bfa]/30"
          }`}
        >
          <span className="text-[8px] font-black uppercase tracking-[0.24em] bg-gradient-to-r from-white via-[#c4b5fd] to-[#a78bfa] bg-clip-text text-transparent">
            What is Cortex?
          </span>
          <span className={`text-[#a78bfa]/50 text-[7px] transition-transform duration-200 ${open ? "rotate-180" : ""}`}>▼</span>
        </button>

        {/* Pin cortex theme site-wide */}
        <button
          onClick={() => setCortexPinned(!cortexPinned)}
          title={cortexPinned ? "Cortex theme active site-wide — click to disable" : "Keep Cortex theme across all pages"}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border transition-all duration-200 ${
            cortexPinned
              ? "bg-[#a78bfa]/18 border-[#a78bfa]/55 shadow-[0_0_14px_rgba(167,139,250,0.22)] text-[#c4b5fd]"
              : "bg-[#a78bfa]/[0.04] border-[#a78bfa]/15 text-[#a78bfa]/45 hover:bg-[#a78bfa]/10 hover:border-[#a78bfa]/30 hover:text-[#a78bfa]/70"
          }`}
        >
          {/* Pin icon */}
          <svg width="10" height="10" viewBox="0 0 14 14" fill="none" aria-hidden="true">
            <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.2" strokeOpacity="0.6" fill="none"/>
            <circle cx="7" cy="7" r="2.5" fill="currentColor" fillOpacity={cortexPinned ? "0.95" : "0.5"}/>
            {cortexPinned && <circle cx="7" cy="7" r="1.2" fill="white" fillOpacity="0.65"/>}
          </svg>
          <span className="text-[8px] font-black uppercase tracking-[0.20em]">
            {cortexPinned ? "Theme On" : "Keep Theme"}
          </span>
        </button>
      </div>

      <div className={`overflow-hidden transition-all duration-300 ease-in-out ${open ? "max-h-64 opacity-100 mt-3" : "max-h-0 opacity-0 mt-0"}`}>
        <div className="mx-auto max-w-lg rounded-xl border border-[#a78bfa]/15 bg-[#a78bfa]/[0.04] px-4 py-3">
          <p className="text-[11px] text-white/50 leading-relaxed">
            Every player has a living neural network — built from xG models, RAPM, WAR, behavioral
            patterns, and real-time form tracking. Watch a game and the AI watches with you, feeding
            observations directly into the model. As each phase and feature is completed by the dev,
            new layers of intelligence are added. The deeper the system gets, the sharper the
            simulations become — and the sharper the edge over the oddsmakers.{" "}
            <span className="text-[#c4b5fd]/60">
              Unfamiliar with a stat? Pull up any player and hover the{" "}
              <span className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full border border-[#a78bfa]/50 text-[#a78bfa] text-[8px] font-black leading-none align-middle mx-0.5">i</span>
              {" "}next to it for a plain-language explanation.
            </span>
          </p>
        </div>
      </div>
    </div>
  );
}

// Neural-network / brain icon — radiating nodes connected by axons
function CortexIcon({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none" aria-hidden="true" className="shrink-0">
      {/* Hexagonal badge border */}
      <path d="M24 2 L42 12 L42 36 L24 46 L6 36 L6 12 Z"
            stroke="#a78bfa" strokeWidth="0.9" strokeOpacity="0.28" fill="#a78bfa" fillOpacity="0.03"/>
      {/* Inner subtle ring */}
      <circle cx="24" cy="24" r="17" stroke="#a78bfa" strokeWidth="0.6" strokeOpacity="0.12" fill="none"/>
      {/* LEFT hemisphere */}
      <path d="M24 10 C18 10 11 14 9 20 C7 25 9 31 13 34 C17 37 21 37 24 37"
            stroke="#a78bfa" strokeWidth="1.8" strokeLinecap="round" fill="rgba(167,139,250,0.07)"/>
      {/* RIGHT hemisphere */}
      <path d="M24 10 C30 10 37 14 39 20 C41 25 39 31 35 34 C31 37 27 37 24 37"
            stroke="#a78bfa" strokeWidth="1.8" strokeLinecap="round" fill="rgba(167,139,250,0.07)"/>
      {/* Central sulcus */}
      <line x1="24" y1="10" x2="24" y2="37" stroke="#c4b5fd" strokeWidth="0.9" strokeOpacity="0.18" strokeDasharray="2.5,3.5"/>
      {/* Left gyri (fold lines) */}
      <path d="M11 19 C14 18 17 19 17 22" stroke="#c4b5fd" strokeWidth="1.2" fill="none" strokeOpacity="0.55" strokeLinecap="round"/>
      <path d="M10 27 C13 26 16 27 16 30" stroke="#c4b5fd" strokeWidth="1.1" fill="none" strokeOpacity="0.40" strokeLinecap="round"/>
      <path d="M15 33 C17 32 19 33 19 35" stroke="#c4b5fd" strokeWidth="0.9" fill="none" strokeOpacity="0.30" strokeLinecap="round"/>
      {/* Right gyri (mirror) */}
      <path d="M37 19 C34 18 31 19 31 22" stroke="#c4b5fd" strokeWidth="1.2" fill="none" strokeOpacity="0.55" strokeLinecap="round"/>
      <path d="M38 27 C35 26 32 27 32 30" stroke="#c4b5fd" strokeWidth="1.1" fill="none" strokeOpacity="0.40" strokeLinecap="round"/>
      <path d="M33 33 C31 32 29 33 29 35" stroke="#c4b5fd" strokeWidth="0.9" fill="none" strokeOpacity="0.30" strokeLinecap="round"/>
      {/* Neural spokes to center */}
      <line x1="15" y1="22" x2="24" y2="24" stroke="#a78bfa" strokeWidth="0.8" strokeOpacity="0.28"/>
      <line x1="33" y1="22" x2="24" y2="24" stroke="#a78bfa" strokeWidth="0.8" strokeOpacity="0.28"/>
      <line x1="14" y1="30" x2="24" y2="24" stroke="#a78bfa" strokeWidth="0.7" strokeOpacity="0.20"/>
      <line x1="34" y1="30" x2="24" y2="24" stroke="#a78bfa" strokeWidth="0.7" strokeOpacity="0.20"/>
      {/* Neural nodes */}
      <circle cx="15" cy="22" r="2.2" fill="#a78bfa" fillOpacity="0.88"/>
      <circle cx="33" cy="22" r="2.2" fill="#a78bfa" fillOpacity="0.88"/>
      <circle cx="14" cy="30" r="1.8" fill="#7c3aed" fillOpacity="0.75"/>
      <circle cx="34" cy="30" r="1.8" fill="#7c3aed" fillOpacity="0.75"/>
      <circle cx="18" cy="14" r="1.3" fill="#c4b5fd" fillOpacity="0.60"/>
      <circle cx="30" cy="14" r="1.3" fill="#c4b5fd" fillOpacity="0.60"/>
      {/* Central hub — brightest point */}
      <circle cx="24" cy="24" r="5.0" fill="#a78bfa" fillOpacity="0.10"/>
      <circle cx="24" cy="24" r="3.2" fill="#a78bfa" fillOpacity="0.95"/>
      <circle cx="24" cy="24" r="1.6" fill="white"   fillOpacity="0.72"/>
    </svg>
  );
}

// ---------------------------------------------------------------------------

export default function PlayersPage() {
  const router = useRouter();

  // Search state
  const [query, setQuery]         = useState("");
  const [allPlayers, setAllPlayers] = useState<{ name: string; team: string; position: string; player_id?: number }[]>([]);
  const [showSugg, setShowSugg]   = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Player detail state (inline fallback when no player_id)
  const [selectedPlayer, setSelectedPlayer] = useState<PlayerData | null>(null);
  const [playerLoading, setPlayerLoading]   = useState(false);
  const [playerNotFound, setPlayerNotFound] = useState(false);
  const [gameLog, setGameLog]               = useState<GameLog | null>(null);
  const [logLoading, setLogLoading]         = useState(false);
  const [warRank, setWarRank]               = useState<number | null>(null);
  const cardRef = useRef<HTMLDivElement>(null);

  // Leaderboard data
  const [rising,  setRising]  = useState<EWMAmover[] | null>(null);
  const [falling, setFalling] = useState<EWMAmover[] | null>(null);
  const [warList, setWarList] = useState<WAREntry[] | null>(null);
  const [pairs,   setPairs]   = useState<ChemPair[] | null>(null);
  const [rdi,     setRdi]     = useState<RDITeam[] | null>(null);
  const [injuries,  setInjuries]  = useState<InjuryRec[] | null>(null);
  const [txns,      setTxns]      = useState<Transaction[] | null>(null);
  const [scoringCategories, setScoringCategories] = useState<Record<string, SkaterEntry[]> | null>(null);
  const [scoringSort, setScoringSort] = useState<"points" | "goals" | "assists">("points");
  const [qotList,        setQotList]        = useState<QoTEntry[] | null>(null);

  // Advanced leaderboards
  const [edgeSkating,    setEdgeSkating]    = useState<EdgeEntry[] | null>(null);
  const [edgeShotSpeed,  setEdgeShotSpeed]  = useState<EdgeEntry[] | null>(null);
  const [goalieList,     setGoalieList]     = useState<GoalieLeaderEntry[] | null>(null);
  const [goalieMetric,   setGoalieMetric]   = useState<"gsax" | "hdsv_pct" | "sv_pct">("gsax");
  const [clutchList,     setClutchList]     = useState<ClutchEntry[] | null>(null);
  const [ppList,         setPpList]         = useState<STEntry[] | null>(null);
  const [pkList,         setPkList]         = useState<STEntry[] | null>(null);
  const [hotHandList,    setHotHandList]    = useState<HotHandEntry[] | null>(null);
  const [xgList,         setXgList]         = useState<XGEntry[] | null>(null);
  const [cdrList,        setCdrList]        = useState<CDREntry[] | null>(null);
  const [rapmList,       setRapmList]       = useState<RAPMEntry[] | null>(null);
  const [rapmCategory,   setRapmCategory]   = useState<"ev_off" | "ev_def" | "pp" | "pk">("ev_off");
  const [xgaList,        setXgaList]        = useState<XGAEntry[] | null>(null);
  const [edgeSkateMetric, setEdgeSkateMetric] = useState<"max_speed_kmh" | "avg_speed_kmh" | "distance_per_game_km">("max_speed_kmh");
  const [edgeShotMetric,  setEdgeShotMetric]  = useState<"max_shot_speed_mph" | "avg_shot_speed_mph" | "hard_shot_count">("max_shot_speed_mph");
  const [stSide,          setStSide]          = useState<"PP" | "PK">("PP");

  // Fetch all leaderboard data on mount
  useEffect(() => {
    fetch("/api/phase2/players")
      .then(r => r.json())
      .then(d => setAllPlayers(d.players ?? []))
      .catch(() => {});

    fetch("/api/phase2/ewma-movers?limit=8")
      .then(r => r.json())
      .then(d => { setRising(d.rising ?? []); setFalling(d.falling ?? []); })
      .catch(() => { setRising([]); setFalling([]); });

    fetch("/api/phase2/war-leaderboard?limit=20")
      .then(r => r.json())
      .then(d => setWarList(d.players ?? []))
      .catch(() => setWarList([]));

    fetch("/api/phase2/top-pairs?limit=6")
      .then(r => r.json())
      .then(d => setPairs(d.pairs ?? []))
      .catch(() => setPairs([]));

    fetch("/api/phase2/roster-disruption")
      .then(r => r.json())
      .then(d => setRdi(d.teams ?? []))
      .catch(() => setRdi([]));

    fetch("/api/phase1/recommendations")
      .then(r => r.json())
      .then(d => setInjuries(d.recommendations ?? []))
      .catch(() => setInjuries([]));

    fetch("/api/phase1/transactions")
      .then(r => r.json())
      .then(d => setTxns(d.transactions ?? []))
      .catch(() => setTxns([]));

    fetch("/api/skater-stats")
      .then(r => r.json())
      .then(d => setScoringCategories({
        points:  d.categories?.points  ?? [],
        goals:   d.categories?.goals   ?? [],
        assists: d.categories?.assists ?? [],
      }))
      .catch(() => setScoringCategories({ points: [], goals: [], assists: [] }));

    fetch("/api/phase2/matchup-explorer?limit=20")
      .then(r => r.json())
      .then(d => setQotList(d.qot ?? []))
      .catch(() => setQotList([]));

    // Advanced leaderboards
    fetch("/api/phase2/edge-leaderboard?metric=max_speed_kmh&limit=20")
      .then(r => r.json()).then(d => setEdgeSkating(d.players ?? [])).catch(() => setEdgeSkating([]));
    fetch("/api/phase2/edge-leaderboard?metric=max_shot_speed_mph&limit=20")
      .then(r => r.json()).then(d => setEdgeShotSpeed(d.players ?? [])).catch(() => setEdgeShotSpeed([]));
    fetch("/api/phase2/goalie-leaderboard?metric=gsax&limit=20")
      .then(r => r.json()).then(d => setGoalieList(d.players ?? [])).catch(() => setGoalieList([]));
    fetch("/api/phase2/clutch-leaderboard?limit=20")
      .then(r => r.json()).then(d => setClutchList(d.players ?? [])).catch(() => setClutchList([]));
    fetch("/api/phase2/special-teams-leaderboard?side=pp&limit=20")
      .then(r => r.json()).then(d => setPpList(d.players ?? [])).catch(() => setPpList([]));
    fetch("/api/phase2/special-teams-leaderboard?side=pk&limit=20")
      .then(r => r.json()).then(d => setPkList(d.players ?? [])).catch(() => setPkList([]));
    fetch("/api/phase2/hot-hand-leaderboard?limit=20")
      .then(r => r.json()).then(d => setHotHandList(d.players ?? [])).catch(() => setHotHandList([]));
    fetch("/api/phase2/xg-leaderboard?limit=20")
      .then(r => r.json()).then(d => setXgList(d.players ?? [])).catch(() => setXgList([]));
    fetch("/api/phase2/cdr-leaderboard?limit=20")
      .then(r => r.json()).then(d => setCdrList(d.players ?? [])).catch(() => setCdrList([]));
    fetch("/api/phase2/rapm-leaderboard?category=ev_off&limit=20")
      .then(r => r.json()).then(d => setRapmList(d.players ?? [])).catch(() => setRapmList([]));
    fetch("/api/phase2/xga-leaderboard?limit=20")
      .then(r => r.json()).then(d => setXgaList(d.players ?? [])).catch(() => setXgaList([]));
  }, []);

  // Autocomplete
  const suggestions = query.trim().length >= 2
    ? allPlayers.filter(p => p.name.toLowerCase().includes(query.trim().toLowerCase())).slice(0, 8)
    : [];

  // Navigate to full profile page by player name
  const loadPlayer = useCallback((name: string) => {
    setShowSugg(false);
    router.push(`/players/${encodeURIComponent(normalizePlayerName(name))}`);
  }, [router]);

  const handleRowClick = useCallback((name: string) => {
    router.push(`/players/${encodeURIComponent(normalizePlayerName(name))}`);
  }, [router]);

  // Dynamic metric switch handlers
  const switchEdgeSkateMetric = (m: typeof edgeSkateMetric) => {
    setEdgeSkateMetric(m);
    setEdgeSkating(null);
    fetch(`/api/phase2/edge-leaderboard?metric=${m}&limit=20`)
      .then(r => r.json()).then(d => setEdgeSkating(d.players ?? [])).catch(() => setEdgeSkating([]));
  };
  const switchEdgeShotMetric = (m: typeof edgeShotMetric) => {
    setEdgeShotMetric(m);
    setEdgeShotSpeed(null);
    fetch(`/api/phase2/edge-leaderboard?metric=${m}&limit=20`)
      .then(r => r.json()).then(d => setEdgeShotSpeed(d.players ?? [])).catch(() => setEdgeShotSpeed([]));
  };
  const switchGoalieMetric = (m: typeof goalieMetric) => {
    setGoalieMetric(m);
    setGoalieList(null);
    fetch(`/api/phase2/goalie-leaderboard?metric=${m}&limit=20`)
      .then(r => r.json()).then(d => setGoalieList(d.players ?? [])).catch(() => setGoalieList([]));
  };
  const switchRapmCategory = (c: typeof rapmCategory) => {
    setRapmCategory(c);
    setRapmList(null);
    fetch(`/api/phase2/rapm-leaderboard?category=${c}&limit=20`)
      .then(r => r.json()).then(d => setRapmList(d.players ?? [])).catch(() => setRapmList([]));
  };

  return (
    <main
      className="min-h-screen p-3 sm:p-6 max-w-3xl mx-auto w-full overflow-x-hidden"
      style={{
        /* Force Cortex purple brand vars — overrides any active team theme on this page */
        "--brand-r": "167", "--brand-g": "139", "--brand-b": "250",
        "--brand-hex": "#a78bfa",
        "--card-base-r": "18", "--card-base-g": "14", "--card-base-b": "28",
        "--card-mid-r":  "10", "--card-mid-g":  "8",  "--card-mid-b":  "16",
      } as React.CSSProperties}
    >

      {/* ── Page Header ── */}
      <div
        className="mb-5 rounded-xl backdrop-blur-sm shadow-[inset_0_1px_0_rgba(167,139,250,0.10),0_4px_24px_rgba(0,0,0,0.60)] overflow-hidden"
        style={{
          background: `linear-gradient(160deg, rgba(18,14,28,0.95) 0%, rgba(10,8,16,0.98) 100%)`,
          border: `1px solid rgba(167,139,250,0.14)`,
        }}
      >
        <div className="px-6 py-3 border-b border-[#a78bfa]/[0.08] flex items-center justify-center gap-3">
          <div className="h-px flex-1 bg-gradient-to-r from-transparent to-[#a78bfa]/20" />
          <div className="flex items-center gap-2.5">
            <p
              className="font-black italic tracking-[0.10em] text-[22px] sm:text-[26px] bg-gradient-to-r from-white via-[#c4b5fd] to-[#a78bfa] bg-clip-text text-transparent leading-none"
              style={{ fontFamily: "var(--font-condensed)" }}
            >
              CORTEX
            </p>
            <CortexIcon size={36} />
          </div>
          <div className="h-px flex-1 bg-gradient-to-l from-transparent to-[#a78bfa]/20" />
        </div>
        <div className="px-6 py-2.5 text-center">
          <p className="text-[9px] font-light uppercase tracking-[0.14em] text-[#a78bfa]/40">
            Player Intelligence · Advanced Models · Live Leaderboards
          </p>
        </div>
      </div>

      {/* ── About pill ── */}
      <CortexAbout />

      {/* ── Playoffs / Season context toggle ── */}
      <div className="flex justify-center mb-4">
        <SeasonContextPill />
      </div>

      {/* Search bar */}
      <div className="relative mb-5">
        <div className="absolute left-4 top-1/2 -translate-y-1/2 pointer-events-none z-10"
          style={{ color: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.55)` }}>
          <svg width="18" height="18" viewBox="0 0 20 20" fill="none" aria-hidden="true">
            <circle cx="8.5" cy="8.5" r="5.5" stroke="currentColor" strokeWidth="1.6"/>
            <line x1="12.5" y1="12.5" x2="17" y2="17" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
          </svg>
        </div>
        <input
          ref={inputRef}
          value={query}
          onChange={e => { setQuery(e.target.value); setShowSugg(true); }}
          onKeyDown={e => {
            if (e.key === "Enter" && query.trim()) loadPlayer(query.trim());
            if (e.key === "Escape") setShowSugg(false);
          }}
          onFocus={e => {
            e.currentTarget.style.border = `1px solid rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.55)`;
            e.currentTarget.style.boxShadow = `0 0 22px rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.12)`;
            e.currentTarget.style.background = `rgba(var(--card-base-r), var(--card-base-g), var(--card-base-b), 0.85)`;
            setShowSugg(true);
          }}
          onBlur={e => {
            e.currentTarget.style.border = `1px solid rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.22)`;
            e.currentTarget.style.boxShadow = ``;
            e.currentTarget.style.background = `rgba(var(--card-base-r), var(--card-base-g), var(--card-base-b), 0.60)`;
            setTimeout(() => setShowSugg(false), 150);
          }}
          placeholder="Search any NHL player…"
          className="w-full pl-11 pr-5 py-4 text-base rounded-2xl text-white focus:outline-none transition-[border,box-shadow]"
          style={{
            background: `rgba(var(--card-base-r), var(--card-base-g), var(--card-base-b), 0.60)`,
            border: `1px solid rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.22)`,
            color: "rgba(255,255,255,0.85)",
          }}
        />
        {query && (
          <button
            onClick={() => { setQuery(""); setSelectedPlayer(null); setPlayerNotFound(false); inputRef.current?.focus(); }}
            className="absolute right-4 top-1/2 -translate-y-1/2 transition-colors text-lg"
            style={{ color: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.35)` }}
          >✕</button>
        )}

        {/* Autocomplete dropdown */}
        {showSugg && suggestions.length > 0 && (
          <div
            className="absolute top-full left-0 right-0 mt-1 z-50 rounded-xl backdrop-blur-xl shadow-[0_8px_40px_rgba(0,0,0,0.85)] overflow-hidden"
            style={{
              background: `rgba(var(--card-mid-r), var(--card-mid-g), var(--card-mid-b), 0.98)`,
              border: `1px solid rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.18)`,
            }}
          >
            {suggestions.map(p => (
              <button
                key={p.name}
                className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-white/[0.06] transition-colors"
                onMouseDown={() => {
                  setQuery(p.name);
                  loadPlayer(p.name);
                }}
              >
                {p.team && <TeamLogo team={p.team} size={24} />}
                <span className="text-sm font-medium text-white/80 flex-1">{p.name}</span>
                <span className="text-[10px] font-mono text-white/30">{p.team}</span>
                <span className="text-[10px] text-white/20">{fmtPos(p.position)}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Loading state */}
      {playerLoading && (
        <div className="mb-5 rounded-2xl border border-white/[0.08] bg-white/[0.02] p-6">
          <div className="flex items-center gap-3">
            <div className="h-16 w-16 rounded-full bg-white/[0.06] animate-pulse shrink-0" />
            <div className="flex-1 space-y-2">
              <div className="h-5 bg-white/[0.06] rounded animate-pulse w-40" />
              <div className="h-3 bg-white/[0.04] rounded animate-pulse w-24" />
            </div>
          </div>
        </div>
      )}

      {/* Not found */}
      {playerNotFound && !playerLoading && (
        <div className="mb-5 rounded-2xl border border-white/[0.07] bg-white/[0.02] px-6 py-8 text-center">
          <p className="text-white/50 text-sm">No player found for &ldquo;{query}&rdquo;.</p>
          <p className="text-white/25 text-xs mt-1">Only skaters and goalies with enough ice time this season are available.</p>
        </div>
      )}

      {/* Player detail card */}
      {selectedPlayer && !playerLoading && (
        <div ref={cardRef} className="mb-6">
          <PlayerDetailCard
            data={selectedPlayer}
            log={gameLog}
            logLoading={logLoading}
            warRank={warRank}
            onClose={() => { setSelectedPlayer(null); setQuery(""); }}
          />
        </div>
      )}

      {/* ── Leaderboards header ── */}
      <div className="flex items-center gap-3 mb-1">
        <span className="text-[9px] font-black uppercase tracking-[0.28em] text-[#a78bfa]/40">Leaderboards</span>
        <div className="flex-1 h-px bg-gradient-to-r from-[#a78bfa]/15 to-transparent" />
      </div>

      {/* ── Sections ── */}
      <div className="flex flex-col gap-4">

        {/* Hot Streaks */}
        <Section title="Hot Streak" count={rising?.length}>
          <InfoTip tip="EWMA xGF/60 trending upward over last 5–10 games vs. the player's own season baseline. Delta = how far above their own average they are running right now. Bayesian-shrunk. Source: GRTZKY model 2.13." />
          {rising === null ? <SectionLoading /> :
           rising.length === 0 ? <SectionEmpty msg="Nothing to report yet." /> : (
           <div className="space-y-1">
             {rising.map((p, i) => (
               <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.05] hover:border-white/[0.10] transition-all duration-150 cursor-pointer" onClick={() => handleRowClick(p.player_name)}>
                 {p.team && <TeamLogo team={p.team} size={40} className="shrink-0" onTeamClick={() => router.push(`/teams/${p.team}`)} />}
                 <div className="flex-1 min-w-0">
                   <p className="text-[11px] font-semibold text-white/80 truncate">{p.player_name}</p>
                   {p.delta != null && <p className="text-[9px] text-white/35 mt-0.5">+{p.delta.toFixed(1)} above avg{p.n_games ? ` · ${p.n_games} GP` : ""}</p>}
                 </div>
                 {p.ewma_xgf60 != null && <span className="text-[10px] font-bold tabular-nums shrink-0" style={{ color: "#4ade80" }}>{p.ewma_xgf60.toFixed(1)} xGF/60</span>}
               </div>
             ))}
           </div>
          )}
        </Section>

        {/* Scoring Leaders */}
        <Section title="Scoring Leaders" count={scoringCategories?.[scoringSort]?.length}>
          {scoringCategories === null ? <SectionLoading /> :
           (scoringCategories[scoringSort]?.length ?? 0) === 0 ? <SectionEmpty msg="NHL stats unavailable." /> : (
           <>
             {/* Sort tabs */}
             <div className="flex gap-1.5 mb-2">
               {(["points", "goals", "assists"] as const).map(cat => (
                 <button
                   key={cat}
                   onClick={() => setScoringSort(cat)}
                   className="px-2.5 py-1 rounded-lg text-[10px] font-semibold uppercase tracking-wide border transition-all duration-150"
                   style={scoringSort === cat ? {
                     color: "#a78bfa",
                     borderColor: "rgba(167,139,250,0.40)",
                     backgroundColor: "rgba(167,139,250,0.12)",
                   } : {
                     color: "rgba(255,255,255,0.30)",
                     borderColor: "rgba(255,255,255,0.08)",
                     backgroundColor: "transparent",
                   }}
                 >
                   {cat}
                 </button>
               ))}
             </div>
             <div className="space-y-1 max-h-64 overflow-y-auto pr-1">
               {scoringCategories[scoringSort].map((p, i) => {
                 const unit = scoringSort === "points" ? "pts" : scoringSort === "goals" ? "G" : "A";
                 return (
                   <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.05] hover:border-white/[0.10] transition-all duration-150 cursor-pointer" onClick={() => handleRowClick(p.name)}>
                     <span className="text-[9px] font-mono text-white/25 w-4 shrink-0 text-right">{p.rank}</span>
                     {p.team && <TeamLogo team={p.team} size={40} className="shrink-0" onTeamClick={() => router.push(`/teams/${p.team}`)} />}
                     <div className="flex-1 min-w-0">
                       <p className="text-[11px] font-semibold text-white/80 truncate">{p.name}</p>
                       <p className="text-[9px] text-white/35 mt-0.5">{fmtPos(p.position)} · {p.team}</p>
                     </div>
                     <span className="text-[10px] font-bold tabular-nums shrink-0" style={{ color: "#a78bfa" }}>{p.value} {unit}</span>
                   </div>
                 );
               })}
             </div>
           </>
          )}
        </Section>

        {/* Clutch Index Leaderboard */}
        <Section title="Clutch Index" count={clutchList?.length}>
          <InfoTip tip="Win Probability Added (WPA) above expectation. Clutch Index = actual WPA/60 minus expected WPA/60 given the opportunities faced. Positive = outperforms in high-leverage moments. Bayesian-shrunk (small samples near zero). Source: GRTZKY model 2.29." />
          {clutchList === null ? <SectionLoading /> : clutchList.length === 0 ? <SectionEmpty msg="Run train-clutch to populate." /> : (
            <div className="space-y-1 max-h-72 overflow-y-auto pr-1 mt-2">
              {clutchList.map((p, i) => {
                const color = (p.value ?? 0) >= 0.05 ? "#4ade80" : (p.value ?? 0) >= 0 ? "#fbbf24" : "#f87171";
                return (
                  <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.05] hover:border-white/[0.10] transition-all duration-150 cursor-pointer" onClick={() => handleRowClick(p.player_name)}>
                    <span className="text-[9px] font-mono text-white/25 w-4 shrink-0 text-right">{p.rank}</span>
                    {p.team && <TeamLogo team={p.team} size={40} className="shrink-0" onTeamClick={() => router.push(`/teams/${p.team}`)} />}
                    <div className="flex-1 min-w-0">
                      <p className="text-[11px] font-semibold text-white/80 truncate">{p.player_name}</p>
                      {p.wpa_per60 != null && <p className="text-[9px] text-white/35 mt-0.5">WPA/60: {p.wpa_per60.toFixed(4)}</p>}
                    </div>
                    <span className="text-[10px] font-bold tabular-nums shrink-0" style={{ color }}>{(p.value ?? 0) > 0 ? "+" : ""}{p.value?.toFixed(4) ?? "—"}</span>
                  </div>
                );
              })}
            </div>
          )}
        </Section>

        {/* Hot Hand Leaderboard */}
        <Section title="Hot Hand Score" count={hotHandList?.length}>
          <InfoTip tip="5-game burst signal. Hot Hand Score = z-score of (goals − expected goals) over last 5 games vs. player's own season variance. +2.0 = 2 standard deviations above their own mean — runaway hot streak. Bayesian-shrunk by career GP. Source: GRTZKY model 2.28." />
          {hotHandList === null ? <SectionLoading /> : hotHandList.length === 0 ? <SectionEmpty msg="Run train-hot-hand to populate." /> : (
            <div className="space-y-1 max-h-72 overflow-y-auto pr-1 mt-2">
              {hotHandList.map((p, i) => {
                const color = (p.value ?? 0) >= 1 ? "#4ade80" : (p.value ?? 0) >= 0 ? "#fbbf24" : "#f87171";
                return (
                  <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.05] hover:border-white/[0.10] transition-all duration-150 cursor-pointer" onClick={() => handleRowClick(p.player_name)}>
                    <span className="text-[9px] font-mono text-white/25 w-4 shrink-0 text-right">{p.rank}</span>
                    {p.team && <TeamLogo team={p.team} size={40} className="shrink-0" onTeamClick={() => router.push(`/teams/${p.team}`)} />}
                    <div className="flex-1 min-w-0">
                      <p className="text-[11px] font-semibold text-white/80 truncate">{p.player_name}</p>
                      <p className="text-[9px] text-white/35 mt-0.5">
                        {p.goals_5g != null ? `${p.goals_5g}G` : ""}
                        {p.xg_5g != null ? ` · ${p.xg_5g.toFixed(1)} xG` : ""}
                        {" · last 5 games"}
                      </p>
                    </div>
                    <span className="text-[10px] font-bold tabular-nums shrink-0" style={{ color }}>{(p.value ?? 0) > 0 ? "+" : ""}{p.value?.toFixed(2) ?? "—"}<span className="text-[8px] text-white/30 ml-0.5">σ</span></span>
                  </div>
                );
              })}
            </div>
          )}
        </Section>

        {/* Cold Streaks */}
        <Section title="Cold Streak" count={falling?.length}>
          <InfoTip tip="EWMA xGF/60 trending downward over last 5–10 games vs. the player's own season baseline. Delta = how far below their own average they are running right now. Bayesian-shrunk. Source: GRTZKY model 2.13." />
          {falling === null ? <SectionLoading /> :
           falling.length === 0 ? <SectionEmpty msg="Nothing to report yet." /> : (
           <div className="space-y-1">
             {falling.map((p, i) => (
               <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.05] hover:border-white/[0.10] transition-all duration-150 cursor-pointer" onClick={() => handleRowClick(p.player_name)}>
                 {p.team && <TeamLogo team={p.team} size={40} className="shrink-0" onTeamClick={() => router.push(`/teams/${p.team}`)} />}
                 <div className="flex-1 min-w-0">
                   <p className="text-[11px] font-semibold text-white/80 truncate">{p.player_name}</p>
                   {p.delta != null && <p className="text-[9px] text-white/35 mt-0.5">{p.delta.toFixed(1)} vs avg{p.n_games ? ` · ${p.n_games} GP` : ""}</p>}
                 </div>
                 {p.ewma_xgf60 != null && <span className="text-[10px] font-bold tabular-nums shrink-0" style={{ color: "#f87171" }}>{p.ewma_xgf60.toFixed(1)} xGF/60</span>}
               </div>
             ))}
           </div>
          )}
        </Section>

        {/* Top Performers */}
        <Section title="WAR Rankings" count={warList?.length}>
          <InfoTip tip="Wins Above Replacement. Combines offensive (xGF) and defensive (xGA) contributions relative to a replacement-level player at the same position. League average = 0; elite forwards/D-men typically 3–6 WAR. Source: GRTZKY model 2.25." />
          {warList === null ? <SectionLoading /> :
           warList.length === 0 ? <SectionEmpty msg="Run compute-war to populate." /> : (
           <div className="space-y-1">
             {warList.map((p, i) => {
               const warColor = p.war != null && p.war >= 2 ? "#4ade80" : p.war != null && p.war >= 0 ? "#fbbf24" : "#f87171";
               return (
                 <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.05] hover:border-white/[0.10] transition-all duration-150 cursor-pointer" onClick={() => handleRowClick(p.player_name)}>
                   {p.team && <TeamLogo team={p.team} size={40} className="shrink-0" onTeamClick={() => router.push(`/teams/${p.team}`)} />}
                   <div className="flex-1 min-w-0">
                     <p className="text-[11px] font-semibold text-white/80 truncate">{p.player_name}</p>
                     <p className="text-[9px] text-white/35 mt-0.5">Ranked #{p.rank} in NHL</p>
                   </div>
                   {p.war != null && <span className="text-[10px] font-bold tabular-nums shrink-0" style={{ color: warColor }}>{p.war > 0 ? "+" : ""}{p.war.toFixed(1)} WAR</span>}
                 </div>
               );
             })}
           </div>
          )}
        </Section>

        {/* Injuries */}
        <Section title="Injured / DTD" count={injuries?.length}>
          {injuries === null ? <SectionLoading /> :
           injuries.length === 0 ? <SectionEmpty msg="No injury data synced yet. Run sync." /> : (
           <div className="space-y-1">
             {injuries.slice(0, 10).map((p, i) => {
               const statusColor = p.status === "Out" ? "#f87171" : p.status === "DTD" ? "#fbbf24" : p.status === "Questionable" ? "#fb923c" : "#94a3b8";
               return (
                 <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.05] hover:border-white/[0.10] transition-all duration-150 cursor-pointer" onClick={() => handleRowClick(p.player_name)}>
                   {p.team_code && <TeamLogo team={p.team_code} size={40} className="shrink-0" onTeamClick={() => router.push(`/teams/${p.team_code}`)} />}
                   <div className="flex-1 min-w-0">
                     <p className="text-[11px] font-semibold text-white/80 truncate">{p.player_name}</p>
                     <p className="text-[9px] text-white/35 mt-0.5">{fmtPos(p.position)}{p.return_date_estimate ? ` · ret. ${p.return_date_estimate}` : ""}</p>
                   </div>
                   <span className="text-[8px] font-semibold uppercase shrink-0 px-1.5 py-0.5 rounded border" style={{ color: statusColor, borderColor: `${statusColor}40`, backgroundColor: `${statusColor}14` }}>{p.status ?? "—"}</span>
                 </div>
               );
             })}
           </div>
          )}
        </Section>

        {/* Quality of Teammates */}
        <Section title="Quality of Teammates" count={qotList?.length}>
          <InfoTip tip="Quality of Teammates (QoT): average RAPM of a player's on-ice linemates at even strength, weighted by shared ice time. High QoT = plays with skilled partners. Compare with QoC (Quality of Competition) to understand context. Source: GRTZKY model 2.3." />
          {qotList === null ? <SectionLoading /> :
           qotList.length === 0 ? <SectionEmpty msg="Run train-matchup to populate." /> : (
           <div className="space-y-1 max-h-64 overflow-y-auto pr-1">
             {qotList.map((p, i) => (
               <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.05] hover:border-white/[0.10] transition-all duration-150 cursor-pointer" onClick={() => handleRowClick(p.player_name)}>
                 {p.team && <TeamLogo team={p.team} size={40} className="shrink-0" onTeamClick={() => router.push(`/teams/${p.team}`)} />}
                 <div className="flex-1 min-w-0">
                   <p className="text-[11px] font-semibold text-white/80 truncate">{p.player_name}</p>
                   <p className="text-[9px] text-white/35 mt-0.5">
                     {p.gp != null ? `${p.gp} GP` : ""}
                     {p.toi_ev != null ? ` · ${p.toi_ev.toFixed(1)} EV TOI` : ""}
                     {p.qoc != null ? ` · QoC ${p.qoc.toFixed(2)}` : ""}
                   </p>
                 </div>
                 <span className="text-[10px] font-bold tabular-nums shrink-0" style={{ color: "#4ade80" }}>QoT {p.qot.toFixed(2)}</span>
               </div>
             ))}
           </div>
          )}
        </Section>

        {/* Recent Moves */}
        <Section title="Recent Moves">
          {txns === null ? <SectionLoading /> :
           txns.length === 0 ? <SectionEmpty msg="No transactions in the last 7 days. Run sync." /> : (
           <div className="space-y-1 max-h-72 overflow-y-auto pr-1">
             {txns.map((t, i) => {
               const TYPE_META: Record<string, { label: string; emoji: string; color: string }> = {
                 trade:          { label: "TRADED",       emoji: "",    color: "#a78bfa" },
                 signing:        { label: "SIGNING",      emoji: "",    color: "#4ade80" },
                 call_up:        { label: "CALL UP",      emoji: "",    color: "#4ade80" },
                 activated:      { label: "ACTIVATED",    emoji: "",    color: "#4ade80" },
                 send_down:      { label: "SEND DOWN",    emoji: "",    color: "#f87171" },
                 ltir:           { label: "LTIR",         emoji: "",    color: "#fb923c" },
                 waiver_claim:   { label: "CLAIMED",      emoji: "",    color: "#a78bfa" },
                 waiver_release: { label: "WAIVED",       emoji: "",    color: "#fbbf24" },
                 front_office:   { label: "FRONT OFFICE", emoji: "",    color: "#fbbf24" },
                 other:          { label: "MOVE",         emoji: "",    color: "#94a3b8" },
               };
               const meta = TYPE_META[t.event_type] ?? { label: t.event_type.replace("_", " ").toUpperCase(), emoji: "", color: "#94a3b8" };
               const desc = t.description.length > 50 ? t.description.slice(0, 48) + "…" : t.description;
               return (
                 <div
                   key={i}
                   className="flex items-center gap-3 px-3 py-2 rounded-xl border"
                   style={{ backgroundColor: `${meta.color}08`, borderColor: `${meta.color}20` }}
                 >
                   {t.team && <TeamLogo team={t.team} size={40} className="shrink-0" onTeamClick={() => router.push(`/teams/${t.team}`)} />}
                   <p className="flex-1 text-[11px] text-white/70 truncate min-w-0">{desc}</p>
                   <span
                     className="text-[8px] font-semibold shrink-0 px-1.5 py-0.5 rounded border whitespace-nowrap"
                     style={{ color: meta.color, borderColor: `${meta.color}40`, backgroundColor: `${meta.color}20` }}
                   >
                     {meta.emoji} {meta.label}
                   </span>
                 </div>
               );
             })}
           </div>
          )}
        </Section>

        {/* Team Disruption */}
        <Section title="Team Disruption">
          <InfoTip tip="Roster Disruption Index (RDI): measures how much a team's lineup has been destabilized by trades, injuries, and call-ups over the last 30 days. High RDI = chemistry disrupted, depth tested. Correlates with short-term performance drops. Source: GRTZKY model 2.19." />
          {rdi === null ? <SectionLoading /> :
           rdi.filter(t => (t.n_moves_30d ?? 0) > 0).length === 0 ? (
             <SectionEmpty msg="No significant roster movement in the last 30 days." />
           ) : (
           <div className="space-y-1 max-h-64 overflow-y-auto pr-1">
             {rdi.filter(t => (t.n_moves_30d ?? 0) > 0).map((t, i) => {
               const high = (t.disruption_index ?? 0) > 0.10;
               const med  = (t.disruption_index ?? 0) > 0.05;
               const color = high ? "#f87171" : med ? "#fbbf24" : "#94a3b8";
               return (
                 <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06]">
                   {t.team && <TeamLogo team={t.team} size={40} className="shrink-0" onTeamClick={() => router.push(`/teams/${t.team}`)} />}
                   <div className="flex-1 min-w-0">
                     <p className="text-[11px] font-semibold text-white/80">{t.team}</p>
                     <p className="text-[9px] text-white/35 mt-0.5">
                       {t.n_moves_30d} move{(t.n_moves_30d ?? 0) !== 1 ? "s" : ""} (30d)
                       {(t.n_moves_72h ?? 0) > 0 ? ` · ${t.n_moves_72h} this week` : ""}
                     </p>
                   </div>
                   {med && (
                     <span className="text-[8px] font-semibold uppercase shrink-0 px-1.5 py-0.5 rounded border" style={{ color, borderColor: `${color}40`, backgroundColor: `${color}14` }}>
                       {high ? "HIGH" : "WATCH"}
                     </span>
                   )}
                 </div>
               );
             })}
           </div>
          )}
        </Section>

        {/* ─────────────────────────────────────────────────────────────────── */}
        {/* ADVANCED MODEL LEADERBOARDS */}
        {/* ─────────────────────────────────────────────────────────────────── */}

        {/* EDGE Skating Leaderboard */}
        <Section title="EDGE Skating" count={edgeSkating?.length}>
          {/* Tab bar */}
          <div className="flex gap-1.5 mb-2 flex-wrap">
            {([
              { k: "max_speed_kmh",        label: "Top Speed",      unit: "km/h" },
              { k: "distance_per_game_km",  label: "Dist/Game",      unit: "km"   },
            ] as const).map(({ k, label }) => (
              <button key={k} onClick={() => switchEdgeSkateMetric(k)}
                className="px-2.5 py-1 rounded-lg text-[10px] font-semibold uppercase tracking-wide border transition-all duration-150"
                style={edgeSkateMetric === k ? { color:"#a78bfa", borderColor:"rgba(167,139,250,0.40)", backgroundColor:"rgba(167,139,250,0.10)" }
                  : { color:"rgba(255,255,255,0.30)", borderColor:"rgba(255,255,255,0.08)", backgroundColor:"transparent" }}
              >{label}</button>
            ))}
          </div>
          <InfoTip tip="NHL EDGE player tracking via arena radar sensors. Season aggregates. Top Speed = highest single-game burst recorded; Avg Speed = mean skating speed; Dist/Game = km skated per game." />
          {edgeSkating === null ? <SectionLoading /> : edgeSkating.length === 0 ? <SectionEmpty msg="Run sync to populate EDGE data." /> : (
            <div className="space-y-1 max-h-72 overflow-y-auto pr-1 mt-2">
              {edgeSkating.map((p, i) => {
                const unit = edgeSkateMetric === "distance_per_game_km" ? " km/g" : " km/h";
                return (
                  <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.05] hover:border-white/[0.10] transition-all duration-150 cursor-pointer" onClick={() => handleRowClick(p.player_name)}>
                    <span className="text-[9px] font-mono text-white/25 w-4 shrink-0 text-right">{p.rank}</span>
                    {p.team && <TeamLogo team={p.team} size={40} className="shrink-0" onTeamClick={() => router.push(`/teams/${p.team}`)} />}
                    <div className="flex-1 min-w-0">
                      <p className="text-[11px] font-semibold text-white/80 truncate">{p.player_name}</p>
                      <p className="text-[9px] text-white/35 mt-0.5">{p.games_played != null ? `${p.games_played} GP` : ""}</p>
                    </div>
                    <span className="text-[10px] font-bold tabular-nums shrink-0" style={{ color: "#4ade80" }}>{p.value != null ? `${p.value}${unit}` : "—"}</span>
                  </div>
                );
              })}
            </div>
          )}
        </Section>

        {/* EDGE Shot Speed Leaderboard */}
        <Section title="EDGE Shot Speed" count={edgeShotSpeed?.length}>
          <div className="flex gap-1.5 mb-2 flex-wrap">
            {([
              { k: "max_shot_speed_mph",  label: "Max Shot Speed" },
            ] as const).map(({ k, label }) => (
              <button key={k} onClick={() => switchEdgeShotMetric(k)}
                className="px-2.5 py-1 rounded-lg text-[10px] font-semibold uppercase tracking-wide border transition-all duration-150"
                style={edgeShotMetric === k ? { color:"#a78bfa", borderColor:"rgba(167,139,250,0.40)", backgroundColor:"rgba(167,139,250,0.10)" }
                  : { color:"rgba(255,255,255,0.30)", borderColor:"rgba(255,255,255,0.08)", backgroundColor:"transparent" }}
              >{label}</button>
            ))}
          </div>
          <InfoTip tip="EDGE shot velocity from radar. Max Speed = fastest single shot recorded this season; Avg Speed = season average shot MPH; Hard Shots = count of shots ≥ 70 MPH." />
          {edgeShotSpeed === null ? <SectionLoading /> : edgeShotSpeed.length === 0 ? <SectionEmpty msg="Run sync to populate EDGE shot speed data." /> : (
            <div className="space-y-1 max-h-72 overflow-y-auto pr-1 mt-2">
              {edgeShotSpeed.map((p, i) => {
                const unit = edgeShotMetric === "hard_shot_count" ? "" : " mph";
                return (
                  <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.05] hover:border-white/[0.10] transition-all duration-150 cursor-pointer" onClick={() => handleRowClick(p.player_name)}>
                    <span className="text-[9px] font-mono text-white/25 w-4 shrink-0 text-right">{p.rank}</span>
                    {p.team && <TeamLogo team={p.team} size={40} className="shrink-0" onTeamClick={() => router.push(`/teams/${p.team}`)} />}
                    <div className="flex-1 min-w-0">
                      <p className="text-[11px] font-semibold text-white/80 truncate">{p.player_name}</p>
                      <p className="text-[9px] text-white/35 mt-0.5">{p.games_played != null ? `${p.games_played} GP` : ""}</p>
                    </div>
                    <span className="text-[10px] font-bold tabular-nums shrink-0" style={{ color: "#fb923c" }}>{p.value != null ? `${p.value}${unit}` : "—"}</span>
                  </div>
                );
              })}
            </div>
          )}
        </Section>

        {/* Goalie Leaderboard */}
        <Section title="Goalies" count={goalieList?.length}>
          <div className="flex gap-1.5 mb-2 flex-wrap">
            {([
              { k: "gsax",     label: "GSAx",        tip: "Goals Saved Above Expected" },
              { k: "hdsv_pct", label: "HD Sv%",       tip: "High-Danger Save %" },
              { k: "sv_pct",   label: "Sv%",          tip: "Overall Save %" },
            ] as const).map(({ k, label }) => (
              <button key={k} onClick={() => switchGoalieMetric(k)}
                className="px-2.5 py-1 rounded-lg text-[10px] font-semibold uppercase tracking-wide border transition-all duration-150"
                style={goalieMetric === k ? { color:"#a78bfa", borderColor:"rgba(167,139,250,0.35)", backgroundColor:"rgba(167,139,250,0.10)" }
                  : { color:"rgba(255,255,255,0.30)", borderColor:"rgba(255,255,255,0.08)", backgroundColor:"transparent" }}
              >{label}</button>
            ))}
          </div>
          <InfoTip tip="Source: MoneyPuck. GSAx = Goals Saved Above Expected vs. shot quality — best measure of true goalie impact. HD Sv% = save rate on high-danger shots (slot/close-range). Min 10 GP." />
          {goalieList === null ? <SectionLoading /> : goalieList.length === 0 ? <SectionEmpty msg="Run sync + train-goalie to populate." /> : (
            <div className="space-y-1 max-h-72 overflow-y-auto pr-1 mt-2">
              {goalieList.map((p, i) => {
                const fmt = goalieMetric === "gsax"
                  ? `${(p.value ?? 0) > 0 ? "+" : ""}${p.value?.toFixed(1)}`
                  : goalieMetric === "sv_pct"
                  ? `.${((p.value ?? 0) * 1000).toFixed(0)}`
                  : `${((p.value ?? 0) * 100).toFixed(1)}%`;
                return (
                  <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.05] hover:border-white/[0.10] transition-all duration-150 cursor-pointer" onClick={() => handleRowClick(p.player_name)}>
                    <span className="text-[9px] font-mono text-white/25 w-4 shrink-0 text-right">{p.rank}</span>
                    {p.team && <TeamLogo team={p.team} size={40} className="shrink-0" onTeamClick={() => router.push(`/teams/${p.team}`)} />}
                    <div className="flex-1 min-w-0">
                      <p className="text-[11px] font-semibold text-white/80 truncate">{p.player_name}</p>
                      <p className="text-[9px] text-white/35 mt-0.5">
                        {p.games_played != null ? `${p.games_played} GP` : ""}
                        {goalieMetric !== "gsax" && p.gsax != null ? ` · GSAx ${p.gsax > 0 ? "+" : ""}${p.gsax.toFixed(1)}` : ""}
                      </p>
                    </div>
                    <span className="text-[10px] font-bold tabular-nums shrink-0" style={{ color: "#a78bfa" }}>{fmt}</span>
                  </div>
                );
              })}
            </div>
          )}
        </Section>

        {/* PP & PK xGF/60 */}
        <Section title="Special Teams xGF/60">
          <div className="flex gap-1.5 mb-2">
            {(["PP", "PK"] as const).map(side => (
              <button key={side}
                onClick={() => setStSide(side)}
                className="px-2.5 py-1 rounded-lg text-[10px] font-semibold uppercase tracking-wide border transition-all duration-150"
                style={stSide === side
                  ? side === "PP"
                    ? { color:"#fbbf24", borderColor:"rgba(251,191,36,0.40)", backgroundColor:"rgba(251,191,36,0.10)" }
                    : { color:"#60a5fa", borderColor:"rgba(96,165,250,0.40)", backgroundColor:"rgba(96,165,250,0.10)" }
                  : { color:"rgba(255,255,255,0.25)", borderColor:"rgba(255,255,255,0.08)", backgroundColor:"transparent" }}
              >{side === "PP" ? "Power Play" : "Penalty Kill"}</button>
            ))}
          </div>
          <InfoTip tip="Per-player PP and PK xGF/60 isolated from team context. PP xGF/60 = expected goals generated per 60 min on the power play; PK xGF/60 = expected goals generated while short-handed. Min 50 PP/PK minutes. Source: GRTZKY model 2.7." />
          {stSide === "PP" ? (
            <div className="mt-2">
              {ppList === null ? <SectionLoading /> : ppList.length === 0 ? <SectionEmpty msg="Run train-special-teams." /> : (
                <div className="space-y-1 max-h-72 overflow-y-auto pr-1">
                  {ppList.map((p, i) => (
                    <div key={i} className="flex items-center gap-2.5 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.05] transition-all cursor-pointer" onClick={() => handleRowClick(p.player_name)}>
                      <span className="text-[9px] font-mono text-white/25 w-4 shrink-0">{p.rank}</span>
                      {p.team && <TeamLogo team={p.team} size={34} className="shrink-0" onTeamClick={() => router.push(`/teams/${p.team}`)} />}
                      <p className="text-[11px] font-semibold text-white/80 flex-1 truncate">{p.player_name}</p>
                      <span className="text-[10px] font-bold tabular-nums shrink-0" style={{ color: "#fbbf24" }}>{p.xgf60?.toFixed(1) ?? "—"}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="mt-2">
              {pkList === null ? <SectionLoading /> : pkList.length === 0 ? <SectionEmpty msg="Run train-special-teams." /> : (
                <div className="space-y-1 max-h-72 overflow-y-auto pr-1">
                  {pkList.map((p, i) => (
                    <div key={i} className="flex items-center gap-2.5 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.05] transition-all cursor-pointer" onClick={() => handleRowClick(p.player_name)}>
                      <span className="text-[9px] font-mono text-white/25 w-4 shrink-0">{p.rank}</span>
                      {p.team && <TeamLogo team={p.team} size={34} className="shrink-0" onTeamClick={() => router.push(`/teams/${p.team}`)} />}
                      <p className="text-[11px] font-semibold text-white/80 flex-1 truncate">{p.player_name}</p>
                      <span className="text-[10px] font-bold tabular-nums shrink-0" style={{ color: "#60a5fa" }}>{p.xgf60?.toFixed(1) ?? "—"}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </Section>

        {/* xG Leaderboard */}
        <Section title="xGF/60 Leaders" count={xgList?.length}>
          <InfoTip tip="Expected Goals For per 60 minutes (EWMA — exponentially weighted 20-game rolling average, λ=0.96). Measures how much offense a player drives at even strength, independent of finishing luck. League average ≈ 4.09 xGF/60. Source: GRTZKY model 2.13." />
          {xgList === null ? <SectionLoading /> : xgList.length === 0 ? <SectionEmpty msg="Run compute-ewma to populate." /> : (
            <div className="space-y-1 max-h-72 overflow-y-auto pr-1 mt-2">
              {xgList.map((p, i) => {
                const above = (p.value ?? 0) - 4.09;
                const color = above >= 2 ? "#4ade80" : above >= 0.5 ? "#fbbf24" : "#94a3b8";
                return (
                  <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.05] hover:border-white/[0.10] transition-all duration-150 cursor-pointer" onClick={() => handleRowClick(p.player_name)}>
                    <span className="text-[9px] font-mono text-white/25 w-4 shrink-0 text-right">{p.rank}</span>
                    {p.team && <TeamLogo team={p.team} size={40} className="shrink-0" onTeamClick={() => router.push(`/teams/${p.team}`)} />}
                    <div className="flex-1 min-w-0">
                      <p className="text-[11px] font-semibold text-white/80 truncate">{p.player_name}</p>
                      <p className="text-[9px] text-white/35 mt-0.5">{p.games_played != null ? `${p.games_played} GP · ` : ""}{above >= 0 ? "+" : ""}{above.toFixed(1)} vs avg</p>
                    </div>
                    <span className="text-[10px] font-bold tabular-nums shrink-0" style={{ color }}>{p.value?.toFixed(2) ?? "—"} xGF/60</span>
                  </div>
                );
              })}
            </div>
          )}
        </Section>

        {/* CDR Leaderboard */}
        <Section title="Composite Defensive Rating" count={cdrList?.length}>
          <InfoTip tip="CDR = 0.60×z(EV Def RAPM xGA) + 0.25×z(Takeaway/Giveaway ratio) + 0.15×z(−xGA/60 adj). All components z-scored; DZS% adjusts xGA so heavy defensive deployment is credited. Positive = better than average defender. Source: GRTZKY model 2.26." />
          {cdrList === null ? <SectionLoading /> : cdrList.length === 0 ? <SectionEmpty msg="Run rate-def to populate." /> : (
            <div className="space-y-1 max-h-72 overflow-y-auto pr-1 mt-2">
              {cdrList.map((p, i) => {
                const color = (p.value ?? 0) >= 1 ? "#4ade80" : (p.value ?? 0) >= 0 ? "#fbbf24" : "#f87171";
                return (
                  <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.05] hover:border-white/[0.10] transition-all duration-150 cursor-pointer" onClick={() => handleRowClick(p.player_name)}>
                    <span className="text-[9px] font-mono text-white/25 w-4 shrink-0 text-right">{p.rank}</span>
                    {p.team && <TeamLogo team={p.team} size={40} className="shrink-0" onTeamClick={() => router.push(`/teams/${p.team}`)} />}
                    <div className="flex-1 min-w-0">
                      <p className="text-[11px] font-semibold text-white/80 truncate">{p.player_name}</p>
                      <p className="text-[9px] text-white/35 mt-0.5">
                        {p.gp != null ? `${p.gp} GP` : ""}
                        {p.xga_60 != null ? ` · xGA/60: ${p.xga_60.toFixed(2)}` : ""}
                        {p.tk_gv_ratio != null ? ` · TK/GV: ${p.tk_gv_ratio.toFixed(1)}` : ""}
                      </p>
                    </div>
                    <span className="text-[10px] font-bold tabular-nums shrink-0" style={{ color }}>{(p.value ?? 0) > 0 ? "+" : ""}{p.value?.toFixed(2) ?? "—"}</span>
                  </div>
                );
              })}
            </div>
          )}
        </Section>

        {/* RAPM Leaderboard */}
        <Section title="RAPM" count={rapmList?.length}>
          <div className="flex gap-1.5 mb-2 flex-wrap">
            {([
              { k: "ev_off", label: "EV Off",  tip: "Even Strength Offensive RAPM" },
              { k: "ev_def", label: "EV Def",  tip: "Even Strength Defensive RAPM" },
              { k: "pp",     label: "PP",       tip: "Power Play RAPM" },
              { k: "pk",     label: "PK",       tip: "Penalty Kill RAPM" },
            ] as const).map(({ k, label }) => (
              <button key={k} onClick={() => switchRapmCategory(k)}
                className="px-2.5 py-1 rounded-lg text-[10px] font-semibold uppercase tracking-wide border transition-all duration-150"
                style={rapmCategory === k ? { color:"#a78bfa", borderColor:"rgba(167,139,250,0.40)", backgroundColor:"rgba(167,139,250,0.10)" }
                  : { color:"rgba(255,255,255,0.30)", borderColor:"rgba(255,255,255,0.08)", backgroundColor:"transparent" }}
              >{label}</button>
            ))}
          </div>
          <InfoTip tip="Regularized Adjusted Plus-Minus. Prior-informed ridge regression with daisy-chain across seasons. Isolates a player's true impact from teammates and opponents. EV Off/Def = even-strength offensive/defensive contribution; PP/PK = special teams isolation. Source: GRTZKY model 2.2." />
          {rapmList === null ? <SectionLoading /> : rapmList.length === 0 ? <SectionEmpty msg="Run train-rapm to populate." /> : (
            <div className="space-y-1 max-h-72 overflow-y-auto pr-1 mt-2">
              {rapmList.map((p, i) => {
                const color = (p.value ?? 0) >= 1 ? "#4ade80" : (p.value ?? 0) >= 0 ? "#fbbf24" : "#f87171";
                return (
                  <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.05] hover:border-white/[0.10] transition-all duration-150 cursor-pointer" onClick={() => handleRowClick(p.player_name)}>
                    <span className="text-[9px] font-mono text-white/25 w-4 shrink-0 text-right">{p.rank}</span>
                    {p.team && <TeamLogo team={p.team} size={40} className="shrink-0" onTeamClick={() => router.push(`/teams/${p.team}`)} />}
                    <div className="flex-1 min-w-0">
                      <p className="text-[11px] font-semibold text-white/80 truncate">{p.player_name}</p>
                      <p className="text-[9px] text-white/35 mt-0.5">{p.gp != null ? `${p.gp} GP` : ""}{p.toi_ev != null ? ` · ${p.toi_ev.toFixed(0)} EV TOI` : ""}</p>
                    </div>
                    <span className="text-[10px] font-bold tabular-nums shrink-0" style={{ color }}>{(p.value ?? 0) > 0 ? "+" : ""}{p.value?.toFixed(3) ?? "—"}</span>
                  </div>
                );
              })}
            </div>
          )}
        </Section>

        {/* xGA/60 Leaderboard */}
        <Section title="xGA/60 (Defensive Wall)" count={xgaList?.length}>
          <InfoTip tip="On-ice expected goals against per 60 minutes at even strength. Lower = better — this player's line suppresses shot quality when on the ice. Source: GRTZKY RAPM model. Min 20 GP. Complements CDR (2.26) for identifying elite defensive forwards and D-men." />
          {xgaList === null ? <SectionLoading /> : xgaList.length === 0 ? <SectionEmpty msg="Run train-rapm to populate." /> : (
            <div className="space-y-1 max-h-72 overflow-y-auto pr-1 mt-2">
              {xgaList.map((p, i) => {
                const color = (p.value ?? 99) <= 1.5 ? "#4ade80" : (p.value ?? 99) <= 2.5 ? "#fbbf24" : "#f87171";
                return (
                  <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.05] hover:border-white/[0.10] transition-all duration-150 cursor-pointer" onClick={() => handleRowClick(p.player_name)}>
                    <span className="text-[9px] font-mono text-white/25 w-4 shrink-0 text-right">{p.rank}</span>
                    {p.team && <TeamLogo team={p.team} size={40} className="shrink-0" onTeamClick={() => router.push(`/teams/${p.team}`)} />}
                    <div className="flex-1 min-w-0">
                      <p className="text-[11px] font-semibold text-white/80 truncate">{p.player_name}</p>
                      <p className="text-[9px] text-white/35 mt-0.5">{p.gp != null ? `${p.gp} GP` : ""}{p.rapm_ev_def != null ? ` · Def RAPM ${p.rapm_ev_def > 0 ? "+" : ""}${p.rapm_ev_def.toFixed(2)}` : ""}</p>
                    </div>
                    <span className="text-[10px] font-bold tabular-nums shrink-0" style={{ color }}>{p.value?.toFixed(2) ?? "—"} xGA/60</span>
                  </div>
                );
              })}
            </div>
          )}
        </Section>

      </div>

      <p className="mt-8 text-[9px] text-white/15 font-mono text-center">
        Data sourced from NHL API, MoneyPuck, and GRTZKY models · For informational use only
      </p>
    </main>
  );
}
