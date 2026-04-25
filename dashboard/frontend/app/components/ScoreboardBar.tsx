"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { TEAM_COLORS } from "@/utils/nhl";
import { logoUrl } from "@/utils/nhl";
import { useScoreHidden, useHideAllScores } from "@/utils/scoreVisibility";
import { useTheme } from "@/utils/themeContext";

interface SeriesStatus {
  topSeedTeamAbbrev?: string;
  topSeedWins?: number;
  bottomSeedTeamAbbrev?: string;
  bottomSeedWins?: number;
  gameNumberOfSeries?: number;
}

interface Game {
  game_id: number;
  date: string;
  game_state: string;
  away_team: string;
  home_team: string;
  away_score: number | null;
  home_score: number | null;
  period: number;
  period_label: string | null;
  time_remaining: string | null;
  in_intermission: boolean;
  start_time_utc: string | null;
  outcome_type: string | null;
  away_on_pp?: boolean;
  home_on_pp?: boolean;
  away_skaters?: number;
  home_skaters?: number;
  series_status?: SeriesStatus | null;
}

const NHL_INT_SECS = 18 * 60; // 18-minute intermission
const PRE_GAME_COUNTDOWN_CAP_SECS = 3 * 3600;

function formatSeriesScore(s: SeriesStatus | null | undefined): string | null {
  if (!s) return null;
  const tw = s.topSeedWins ?? 0;
  const bw = s.bottomSeedWins ?? 0;
  const top = s.topSeedTeamAbbrev;
  const bot = s.bottomSeedTeamAbbrev;
  if (!top || !bot) return null;
  if (tw === 0 && bw === 0) return `G${s.gameNumberOfSeries ?? 1}`;
  if (tw === bw) return `TIED ${tw}-${bw}`;
  return tw > bw ? `${top} ${tw}-${bw}` : `${bot} ${bw}-${tw}`;
}

function formatTime(utc: string | null): string {
  if (!utc) return "--:--";
  try { return new Date(utc).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }); }
  catch { return "--:--"; }
}

function fmtCountdown(secs: number): string {
  if (secs <= 0) return "0:00";
  if (secs >= 3600) {
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    return m > 0 ? `${h}h ${m}m` : `${h}h`;
  }
  if (secs > 900) {
    const m = Math.floor(secs / 60);
    return `${m}m`;
  }
  // ≤15 min — show live MM:SS
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function statusLabel(g: Game): string {
  const s = g.game_state;
  if (s === "FINAL" || s === "OFF") return `FINAL${g.outcome_type ? `/${g.outcome_type}` : ""}`;
  if (s === "LIVE" || s === "CRIT") {
    if (g.in_intermission) {
      const n = g.period === 1 ? "1ST" : g.period === 2 ? "2ND" : "";
      return n ? `${n} INT` : "INT";
    }
    return `${g.time_remaining ?? "?:??"}  ${g.period_label ?? ""}`.trim();
  }
  if (s === "PRE") return "PRE-GAME";
  return formatTime(g.start_time_utc);
}

function hoverLabel(g: Game): string {
  const s = g.game_state;
  if (s === "FINAL" || s === "OFF") return g.outcome_type ? `FINAL · ${g.outcome_type}` : "FINAL";
  if (s === "LIVE" || s === "CRIT") {
    if (g.in_intermission) {
      const n = g.period === 1 ? "1ST" : g.period === 2 ? "2ND" : "";
      return n ? `${n} INT` : "INT";
    }
    return `${g.period_label ?? ""} · ${g.time_remaining ?? ""}`.trim().replace(/^·\s*/, "");
  }
  if (s === "PRE") return "PRE-GAME";
  return formatTime(g.start_time_utc);
}

// ---------------------------------------------------------------------------
// Countdown — pre-game uses exact start time; intermission counts 18 min from
// first detection, persisted in localStorage so refresh doesn't reset it.
// ---------------------------------------------------------------------------
function useCountdown(g: Game): { text: string; secs: number } | null {
  const [secs, setSecs] = useState<number | null>(null);
  const prevInt = useRef(false);

  // localStorage key is stable per game+period so it survives refresh
  const intKey = `gretzky_int_start_${g.game_id}_${g.period}`;

  useEffect(() => {
    const tick = () => {
      const s = g.game_state;

      // Pre-game countdown to exact kick-off time
      if ((s === "PRE" || s === "FUT") && g.start_time_utc) {
        const diff = Math.floor((new Date(g.start_time_utc).getTime() - Date.now()) / 1000);
        setSecs(diff > 0 ? diff : null);
        return;
      }

      // Intermission countdown — persist start timestamp across refreshes
      if ((s === "LIVE" || s === "CRIT") && g.in_intermission) {
        if (!prevInt.current) {
          // First detection this mount — check localStorage first
          const stored = localStorage.getItem(intKey);
          if (!stored) {
            localStorage.setItem(intKey, String(Date.now()));
          }
          prevInt.current = true;
        }
        const startMs   = Number(localStorage.getItem(intKey) ?? Date.now());
        const elapsed   = Math.floor((Date.now() - startMs) / 1000);
        const remaining = NHL_INT_SECS - elapsed;
        setSecs(remaining > 0 ? remaining : null);
        return;
      }

      // Leaving intermission — clear stored start so next intermission is fresh
      if (prevInt.current && !g.in_intermission) {
        prevInt.current = false;
        // Don't clear key yet; a different period key will be used next intermission
      }
      setSecs(null);
    };

    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [g.game_state, g.in_intermission, g.start_time_utc, intKey]);

  return secs !== null ? { text: fmtCountdown(secs), secs } : null;
}

// ---------------------------------------------------------------------------
// Team logo — no link wrapper (card is already a link to the game page)
// ---------------------------------------------------------------------------
function Logo({ abbrev, size = 28, smSize }: { abbrev: string; size?: number; smSize?: number }) {
  if (!abbrev) return <span style={{ width: smSize ?? size, height: smSize ?? size }} className="shrink-0 inline-flex" />;
  const img = (s: number) => (
    <img src={logoUrl(abbrev)} alt={abbrev} width={s} height={s}
      className="object-contain shrink-0" style={{ width: s, height: s }} draggable={false} />
  );
  if (!smSize) return img(size);
  return (
    <>
      <span className="sm:hidden inline-flex">{img(size)}</span>
      <span className="hidden sm:inline-flex">{img(smSize)}</span>
    </>
  );
}

// ---------------------------------------------------------------------------
// Game card
// ---------------------------------------------------------------------------
function GameCard({ g }: { g: Game }) {
  const { hidden, toggle: onToggleHide } = useScoreHidden(g.game_id);
  const hideScores = hidden;
  const label        = statusLabel(g);
  const overlayText  = hoverLabel(g);
  const countdown    = useCountdown(g);
  const isFinal      = g.game_state === "FINAL" || g.game_state === "OFF";
  const isLive       = g.game_state === "LIVE"  || g.game_state === "CRIT";
  const isPre        = g.game_state === "PRE"   || g.game_state === "FUT";
  const isInt        = isLive && g.in_intermission;
  const isScored     = g.away_score != null && g.home_score != null;
  const awayWins     = isScored && g.away_score! > g.home_score!;
  const homeWins     = isScored && g.home_score! > g.away_score!;
  const live         = isLive && !isFinal;

  const gcColor  = isLive ? "#4ade80" : isFinal ? "#f87171" : "#fbbf24";
  const gcShadow = isLive ? "rgba(74,222,128,0.35)" : isFinal ? "rgba(248,113,113,0.35)" : "rgba(251,191,36,0.35)";
  const gcBorder = isLive ? "rgba(74,222,128,0.45)" : isFinal ? "rgba(248,113,113,0.45)" : "rgba(251,191,36,0.45)";

  // Glossy glass cards — shiny effect, strong button feel
  const cardCls = live
    ? "border-[1.5px] border-[#4ade80]/[0.38] bg-gradient-to-b from-white/[0.20] via-white/[0.08] to-transparent backdrop-blur-sm cursor-pointer active:scale-[0.97] transition-all duration-150 shadow-[0_10px_40px_rgba(0,0,0,0.90),0_0_24px_rgba(74,222,128,0.18),0_2px_6px_rgba(0,0,0,0.65),inset_0_1px_0_rgba(255,255,255,0.32),inset_0_-1px_0_rgba(0,0,0,0.40),inset_1px_0_0_rgba(255,255,255,0.10),inset_-1px_0_0_rgba(255,255,255,0.10)] hover:shadow-[0_10px_40px_rgba(0,0,0,0.95),0_0_32px_rgba(74,222,128,0.28),inset_0_1px_0_rgba(255,255,255,0.36),inset_0_-1px_0_rgba(0,0,0,0.45)] hover:border-[#4ade80]/[0.55] hover:bg-gradient-to-b hover:from-white/[0.24] hover:via-white/[0.10]"
    : "border-[1.5px] border-[#C9A84C]/[0.30] bg-gradient-to-b from-[#C9A84C]/[0.06] via-white/[0.04] to-transparent backdrop-blur-sm cursor-pointer active:scale-[0.97] transition-all duration-150 shadow-[0_8px_32px_rgba(0,0,0,0.85),0_0_18px_rgba(201,168,76,0.12),0_2px_6px_rgba(0,0,0,0.60),inset_0_1px_0_rgba(255,255,255,0.20),inset_0_-1px_0_rgba(0,0,0,0.40),inset_1px_0_0_rgba(201,168,76,0.04),inset_-1px_0_0_rgba(201,168,76,0.04)] hover:shadow-[0_8px_32px_rgba(0,0,0,0.90),0_0_24px_rgba(201,168,76,0.22),inset_0_1px_0_rgba(255,255,255,0.25)] hover:border-[#C9A84C]/[0.50] hover:from-[#C9A84C]/[0.10] hover:via-white/[0.06]";

  const homeClr = TEAM_COLORS[g.home_team] ?? "#C9A84C";
  const awayClr = TEAM_COLORS[g.away_team] ?? "#94a3b8";

  // Winning/leading team gets their primary color; loser fades out
  const homeBg = (isFinal && !homeWins)
    ? "transparent"
    : homeWins
      ? `${homeClr}44`  // leading/winner: real team color prominent
      : `${homeClr}10`; // neutral tint

  const awayBg = (isFinal && !awayWins)
    ? "transparent"
    : awayWins
      ? `${awayClr}44`  // leading/winner: real team color prominent
      : `${awayClr}08`; // neutral tint

  const row = (abbrev: string, score: number | null, wins: boolean, isHome: boolean) => {
    const clr    = isHome ? homeClr : awayClr;
    const bg     = isHome ? homeBg  : awayBg;
    const isLeading = isHome ? homeWins : awayWins;
    const isLosing  = isFinal && !wins;
    const onPP   = live && (isHome ? g.home_on_pp : g.away_on_pp);

    return (
      <div className="flex items-center gap-1">
        <div
          className="flex items-center gap-0.5 rounded-md px-1 py-0.5 w-[80px] transition-all duration-300"
          style={{
            backgroundColor: bg,
            border: isLeading
              ? `1px solid ${clr}66`
              : "1px solid transparent",
            boxShadow: isLeading
              ? `0 2px 14px rgba(0,0,0,0.65), 0 0 12px ${clr}38, inset 0 1px 0 rgba(255,255,255,0.10), inset 0 0 16px ${clr}0c`
              : undefined,
          }}
        >
          <Logo abbrev={abbrev} size={27} smSize={34} />
          <span className={`text-[9px] font-black tracking-wide ${isLosing ? "text-white/20" : "text-white/80"}`}>
            {abbrev}{onPP && (
              <span className="text-[7px] text-[#f87171] animate-pulse ml-0.5">
                PP
              </span>
            )}
          </span>
        </div>
        <span className={`ml-auto text-[15px] font-black font-mono tabular-nums ${isLosing ? "text-white/20" : "text-white"}`}>
          {hideScores ? <span className="text-[11px] text-white/20">•</span> : (isScored ? score : "–")}
        </span>
      </div>
    );
  };

  // Label colour — yellow for pre-game state label; amber for intermission countdown
  const labelClr     = isPre ? "text-[#fbbf24]" : isInt ? "text-[#fbbf24]/80" : "text-white/30";
  // Countdown colour — grey normally, red when ≤5 min (300s)
  const countdownSecs = countdown?.secs ?? Infinity;
  const countdownClr = countdownSecs <= 300 ? "text-[#f87171]" : "text-white/35";
  // Playoff series chip — replaces the countdown outside the 3-hour window
  // and appends to FINAL so the card carries series context at a glance.
  const seriesText = formatSeriesScore(g.series_status);
  const inFarPreWindow = isPre && countdown != null && countdown.secs > PRE_GAME_COUNTDOWN_CAP_SECS;
  const rightText =
    isFinal && seriesText     ? seriesText :
    inFarPreWindow && seriesText ? seriesText :
    (isPre && !countdown && seriesText) ? seriesText :
    countdown?.text ?? null;
  const rightIsSeries = rightText === seriesText && seriesText != null && (isFinal || inFarPreWindow || (isPre && !countdown));
  const rightClr = rightIsSeries ? "text-[#C9A84C]/75" : countdownClr;

  return (
    <Link href={`/game/${g.game_id}`} className="block shrink-0 w-[132px]">
    <div className={`relative group flex flex-col justify-center gap-1 px-2 py-1.5 w-full rounded-xl transition-all duration-200 border ${cardCls}`}>
      {row(g.away_team, g.away_score, awayWins, false)}
      {row(g.home_team, g.home_score, homeWins, true)}

      {/* Status row */}
      <div className="flex items-center gap-1 mt-0.5">
        {live && !isInt && (
          <span className="w-1.5 h-1.5 rounded-full bg-[#4ade80] shadow-[0_0_6px_rgba(74,222,128,0.9)] animate-pulse shrink-0" />
        )}
        <span className={`text-[11px] font-black tracking-wider uppercase truncate ${
          live && !isInt ? "text-[#4ade80]" : isFinal ? "text-white/20" : labelClr
        }`}>
          {hideScores && (isLive || isFinal) && !isInt ? "—" : label}
        </span>
        {(isInt || !hideScores) && rightText && (
          <span className={`text-[11px] font-black font-mono tabular-nums ml-auto shrink-0 ${rightClr}`}>
            {rightText}
          </span>
        )}
      </div>

      {/* Desktop hover overlay — status info + hide button (live games only) */}
      <div className="absolute inset-0 hidden sm:flex flex-col items-center justify-between opacity-0 group-hover:opacity-100 transition-all duration-200 bg-black/80 backdrop-blur-[6px] rounded-xl py-1.5 px-2 pointer-events-none">
        <div className="flex-1 flex flex-col items-center justify-center gap-1">
          <span
            className="text-[11px] font-black tracking-[0.16em] uppercase px-3 py-1.5 rounded-md border"
            style={{
              color: gcColor,
              borderColor: gcBorder,
              backgroundColor: "rgba(0,0,0,0.6)",
              boxShadow: `0 0 14px ${gcShadow}, inset 0 1px 0 rgba(210,215,220,0.06)`,
            }}
          >
            {overlayText}
          </span>
          {countdown && (
            <span className="text-[13px] font-black font-mono tabular-nums" style={{ color: gcColor }}>
              {countdown.text}
            </span>
          )}
        </div>
        {live && (
          <button
            onClick={e => { e.preventDefault(); e.stopPropagation(); onToggleHide(); }}
            className={`text-[9px] font-black uppercase tracking-[0.14em] px-2 py-0.5 rounded border transition-all duration-150 pointer-events-auto ${
              hidden
                ? "text-[#C9A84C] border-[#C9A84C]/40 bg-[#C9A84C]/10 hover:bg-[#C9A84C]/20"
                : "text-white/60 border-white/[0.22] bg-white/[0.08] hover:text-white hover:border-white/[0.35]"
            }`}
          >
            {hidden ? "show score" : "hide score"}
          </button>
        )}
      </div>

      {/* Mobile button — bottom-right, only when live and no countdown showing */}
      {live && !countdown && (
        <button
          onClick={e => { e.preventDefault(); e.stopPropagation(); onToggleHide(); }}
          className={`absolute bottom-1.5 right-1.5 sm:hidden text-[7px] font-black uppercase tracking-[0.12em] px-1.5 py-0.5 rounded border transition-all duration-150 ${
            hidden
              ? "text-[#C9A84C] border-[#C9A84C]/50 bg-[#C9A84C]/15"
              : "text-white/40 border-white/[0.18] bg-white/[0.06] active:text-white/80"
          }`}
        >
          {hidden ? "show" : "hide"}
        </button>
      )}
    </div>
    </Link>
  );
}

// ---------------------------------------------------------------------------
// Date separator
// ---------------------------------------------------------------------------
function DateSep({ date }: { date: string }) {
  const label = new Date(date + "T12:00:00").toLocaleDateString([], {
    weekday: "short", month: "short", day: "numeric",
  });
  return (
    <div className="flex flex-col items-center justify-center px-2 shrink-0">
      <span className="text-[8px] font-black tracking-[0.2em] uppercase text-white/30 whitespace-nowrap underline underline-offset-2 decoration-white/20">
        {label}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Arrow button
// ---------------------------------------------------------------------------
function Arrow({ dir, onClick, visible }: { dir: "left" | "right"; onClick: () => void; visible: boolean }) {
  return (
    <div className="flex shrink-0 w-7 h-full items-center justify-center">
      <button
        onClick={onClick}
        aria-label={dir === "left" ? "Scroll left" : "Scroll right"}
        className={`w-6 h-7 flex items-center justify-center rounded-lg bg-white/[0.05] hover:bg-white/[0.10] transition-all duration-200 ${visible ? "text-white/50" : "text-white/10 pointer-events-none"}`}
      >
        <span className="text-xs font-black select-none leading-none">{dir === "left" ? "❮" : "❯"}</span>
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Scoreboard bar
// ---------------------------------------------------------------------------
const SCROLL_STEP = 420;

export default function ScoreboardBar() {
  const [games, setGames]       = useState<Game[]>([]);
  const [loaded, setLoaded]     = useState(false);
  const [canLeft, setCanLeft]   = useState(false);
  const [canRight, setCanRight] = useState(false);
  const { hideAll, toggleAll: toggleHideAll } = useHideAllScores();
  const { theme, clearTheme } = useTheme();
  const scrollRef = useRef<HTMLDivElement>(null);

  const updateArrows = () => {
    const el = scrollRef.current;
    if (!el) return;
    setCanLeft(el.scrollLeft > 0);
    setCanRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
  };

  const scroll = (dir: "left" | "right") => {
    scrollRef.current?.scrollBy({ left: dir === "left" ? -SCROLL_STEP : SCROLL_STEP, behavior: "smooth" });
  };

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;

    // Default to today's first game; fall back to first live game; last resort = end
    const todayStr = new Date().toLocaleDateString("en-CA"); // YYYY-MM-DD local
    const todayEl  = el.querySelector(`[data-date='${todayStr}']`) as HTMLElement | null;
    const liveEl   = el.querySelector("[data-live='true']")         as HTMLElement | null;
    if (todayEl) {
      el.scrollLeft = Math.max(0, todayEl.offsetLeft - 8);
    } else if (liveEl) {
      el.scrollLeft = liveEl.offsetLeft;
    } else {
      el.scrollLeft = el.scrollWidth - 120;
    }

    el.addEventListener("scroll", updateArrows, { passive: true });
    updateArrows();
    return () => el.removeEventListener("scroll", updateArrows);
  }, [games]);

  useEffect(() => {
    let id: ReturnType<typeof setInterval>;
    const fetch_ = () =>
      fetch("/api/scoreboard")
        .then(r => r.json())
        .then(d => {
          const incoming: Game[] = d.games ?? [];
          setGames(incoming);
          setLoaded(true);
          // Re-schedule: 10s if any game is live (catches PP expiry quickly), else 30s
          clearInterval(id);
          const hasLiveNow = incoming.some(
            g => g.game_state === "LIVE" || g.game_state === "CRIT"
          );
          id = setInterval(fetch_, hasLiveNow ? 10_000 : 30_000);
        })
        .catch(() => setLoaded(true));
    fetch_();
    id = setInterval(fetch_, 30_000);
    return () => clearInterval(id);
  }, []);

  if (!loaded) return null;

  const hasLive = games.some(g => g.game_state === "LIVE" || g.game_state === "CRIT");
  const seenDates = new Set<string>();

  return (
    <div className="select-none">
      {/* Label strip */}
      <div className="flex items-center justify-center gap-2 border-b border-[#C9A84C]/[0.18] h-8 relative" style={{ background: "linear-gradient(to bottom, rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.04), transparent)" }}>
        {theme && (
          <button
            onClick={clearTheme}
            className="absolute left-2 text-[9px] font-black uppercase tracking-[0.14em] px-2 py-[3px] rounded-md border transition-all duration-200 text-white/30 border-white/[0.10] bg-white/[0.03] hover:text-white/60 hover:border-white/[0.20] hover:bg-white/[0.06]"
          >
            Reset Theme
          </button>
        )}
        <span className={`h-2 w-2 rounded-full shrink-0 border transition-all duration-300 ${
          hasLive
            ? "bg-[#4ade80]/80 border-[#4ade80]/60 shadow-[0_0_7px_rgba(74,222,128,0.75)] animate-pulse"
            : "bg-[#ef4444]/70 border-[#ef4444]/50 shadow-[0_0_5px_rgba(239,68,68,0.45)]"
        }`} />
        <span className="text-[10px] font-semibold uppercase tracking-[0.25em] text-[#c8cdd2]/40">Live Games</span>
        {hasLive && (
          <button
            onClick={toggleHideAll}
            className={`absolute right-2 text-[9px] font-black uppercase tracking-[0.14em] px-2 py-[3px] rounded-md border transition-all duration-200 ${
              hideAll
                ? "text-[#C9A84C] border-[#C9A84C]/50 bg-[#C9A84C]/[0.12] shadow-[0_0_8px_rgba(201,168,76,0.15)]"
                : "text-[#C9A84C]/50 border-[#C9A84C]/20 bg-[#C9A84C]/[0.04] hover:text-[#C9A84C]/80 hover:border-[#C9A84C]/35 hover:bg-[#C9A84C]/[0.08]"
            }`}
          >
            {hideAll ? "show scores" : "hide scores"}
          </button>
        )}
      </div>

      {games.length > 0 && (
        <div className="flex items-center border-b border-[#C9A84C]/[0.14] py-1.5" style={{ background: "linear-gradient(to bottom, rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.025), transparent)" }}>
          <Arrow dir="left" onClick={() => scroll("left")} visible={canLeft} />
          <div
            ref={scrollRef}
            className="flex items-center flex-1 overflow-x-auto justify-center sm:justify-start gap-1.5 px-1"
          >
            {games.map(g => {
              const isNew  = g.date && !seenDates.has(g.date);
              if (g.date) seenDates.add(g.date);
              const isLive = g.game_state === "LIVE" || g.game_state === "CRIT";
              return (
                <div key={g.game_id} className="flex items-center shrink-0 gap-2" data-live={isLive ? "true" : undefined} data-date={g.date ?? undefined}>
                  {isNew && <DateSep date={g.date} />}
                  <GameCard g={g} />
                </div>
              );
            })}
          </div>
          <Arrow dir="right" onClick={() => scroll("right")} visible={canRight} />
        </div>
      )}
    </div>
  );
}
