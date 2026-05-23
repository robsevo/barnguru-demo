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
  /** Wins needed to clinch (4 for best-of-7). NHL ships this on every playoff series. */
  neededToWin?: number;
  /** Set once a team has clinched the series. */
  winningTeamId?: number;
  losingTeamId?: number;
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

/** A scheduled game with number `n` is "if necessary" — i.e. won't be played
 *  if the series ends before it — when the trailing team's wins + needed
 *  wins is less than `n`. Returns false for live/completed games (caller
 *  filters those out separately). */
function isUncertainGame(s: SeriesStatus | null | undefined, n: number | undefined): boolean {
  if (!s || n == null) return false;
  const needed = s.neededToWin ?? 4;
  const minWins = Math.min(s.topSeedWins ?? 0, s.bottomSeedWins ?? 0);
  // G_n requires the leader to have lost enough games that they cannot have
  // clinched before now — so G_n is uncertain iff n > needed + minWins.
  return n > needed + minWins;
}

function formatSeriesScore(s: SeriesStatus | null | undefined): string | null {
  if (!s) return null;
  const tw = s.topSeedWins ?? 0;
  const bw = s.bottomSeedWins ?? 0;
  const top = s.topSeedTeamAbbrev;
  const bot = s.bottomSeedTeamAbbrev;
  if (!top || !bot) return null;
  const needed = s.neededToWin ?? 4;
  const gameNum = s.gameNumberOfSeries;
  const uncertain = isUncertainGame(s, gameNum);
  const star = uncertain ? "*" : "";

  // Series already clinched — every game past the decider is canceled.
  if (s.winningTeamId != null || Math.max(tw, bw) >= needed) {
    return tw > bw ? `${top} WIN ${tw}-${bw}` : `${bot} WIN ${bw}-${tw}`;
  }

  // Pre-series — only emit a G_n chip if NHL actually gave us the game
  // number. Don't fall back to "G1" — that misled callers into showing G1
  // on every chip for series whose later games arrived without a number.
  if (tw === 0 && bw === 0) {
    return gameNum != null ? `G${gameNum}${star}` : null;
  }

  // Series in progress — for "if necessary" games surface that explicitly
  // so the eye sees the asterisk; otherwise show the running series score.
  if (uncertain && gameNum != null) {
    return `G${gameNum}*`;
  }
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

  // Score-change flash — track previous scores and flag the row whose
  // value just incremented so the new number flares briefly.
  const prevAway = useRef<number | null | undefined>(g.away_score);
  const prevHome = useRef<number | null | undefined>(g.home_score);
  const [awayFlash, setAwayFlash] = useState(false);
  const [homeFlash, setHomeFlash] = useState(false);
  useEffect(() => {
    if (prevAway.current != null && g.away_score != null && g.away_score > prevAway.current) {
      setAwayFlash(true);
      const t = setTimeout(() => setAwayFlash(false), 1300);
      prevAway.current = g.away_score;
      return () => clearTimeout(t);
    }
    prevAway.current = g.away_score;
  }, [g.away_score]);
  useEffect(() => {
    if (prevHome.current != null && g.home_score != null && g.home_score > prevHome.current) {
      setHomeFlash(true);
      const t = setTimeout(() => setHomeFlash(false), 1300);
      prevHome.current = g.home_score;
      return () => clearTimeout(t);
    }
    prevHome.current = g.home_score;
  }, [g.home_score]);
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

  // HUD-styled cards — corner brackets via ::before/::after (hud-panel--all-corners helper)
  // No corner brackets on game cards — they read cleaner without.
  // Pre-game (!live && !isFinal) uses the active TEAM theme vars instead of
  // hardcoded amber/yellow so the strip tints with whatever theme is set
  // (CAR red, COL maroon, etc.) — only Live (green) + Final (white) stay
  // status-coloured because those are universal states.
  const cardCls = live
    ? "rounded-md cursor-pointer active:scale-[0.97] transition-all duration-150 hover:shadow-[0_0_24px_rgba(74,222,128,0.35)]"
    : isFinal
    ? "rounded-md cursor-pointer active:scale-[0.97] transition-all duration-150 hover:opacity-90"
    : "rounded-md cursor-pointer active:scale-[0.97] transition-all duration-150 hover:shadow-[0_0_18px_rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.25)]";

  const cardStyle: React.CSSProperties = {
    background: live
      ? "linear-gradient(180deg, rgba(74,222,128,0.10) 0%, rgba(var(--card-base-r),var(--card-base-g),var(--card-base-b),0.95) 30%, rgba(var(--card-mid-r),var(--card-mid-g),var(--card-mid-b),0.99) 100%)"
      : "linear-gradient(180deg, rgba(220,228,240,0.04) 0%, rgba(var(--card-base-r),var(--card-base-g),var(--card-base-b),0.95) 30%, rgba(var(--card-mid-r),var(--card-mid-g),var(--card-mid-b),0.99) 100%)",
    border: `1px solid ${live ? "rgba(74,222,128,0.30)" : isFinal ? "rgba(255,255,255,0.08)" : "rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.30)"}`,
    boxShadow: "inset 0 1px 0 rgba(255,255,255,0.08), inset 0 -1px 0 rgba(0,0,0,0.40), 0 2px 8px rgba(0,0,0,0.50)",
  };

  const cardThemeColor = live
    ? "#4ade80"
    : isFinal
    ? "#f87171"
    : "rgb(var(--brand-r), var(--brand-g), var(--brand-b))";

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
        <span
          className={`ml-auto text-[15px] font-black font-mono tabular-nums ${isLosing ? "text-white/20" : "text-white"} ${
            (isHome ? homeFlash : awayFlash) ? "score-flash" : ""
          }`}
          style={{ color: (isHome ? homeFlash : awayFlash) ? clr : undefined }}
        >
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
    <Link href={`/game/${g.game_id}`} className="block shrink-0 w-[140px] hud-interactive">
    <div className={`relative group flex flex-col justify-center gap-1 px-2 py-1.5 w-full ${cardCls}`}
      style={cardStyle}>
      {row(g.away_team, g.away_score, awayWins, false)}
      {row(g.home_team, g.home_score, homeWins, true)}

      {/* Status row */}
      <div className="flex items-center gap-1 mt-0.5">
        {live && !isInt && (
          <span className="w-1.5 h-1.5 rounded-full bg-[#4ade80] shadow-[0_0_6px_rgba(74,222,128,0.9)] animate-pulse shrink-0" />
        )}
        <span className={`hud-mono text-[10px] tracking-[0.16em] uppercase truncate ${
          live && !isInt ? "text-[#4ade80]" : isFinal ? "text-white/30" : labelClr
        }`}>
          {hideScores && (isLive || isFinal) && !isInt ? "—" : label}
        </span>
        {(isInt || !hideScores) && rightText && (
          <span className={`hud-mono text-[10px] tabular-nums ml-auto shrink-0 ${rightClr}`}>
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
    <div className="flex flex-col items-center justify-center px-2 shrink-0 relative">
      <span className="hud-mono text-[8px] tracking-[0.22em] uppercase whitespace-nowrap"
        style={{ color: "rgba(255,255,255,0.40)", textShadow: "0 0 6px rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.25)" }}>
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
  const didInitialScrollRef = useRef(false);

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
    if (!el || games.length === 0) return;

    // Default: center the card whose date is closest to today (today=0 wins).
    // Run once per mount so subsequent /api/scoreboard polls (every 10–30s)
    // don't snap the user's manual scroll back to default.
    if (!didInitialScrollRef.current) {
      const todayStr = new Date().toLocaleDateString("en-CA"); // YYYY-MM-DD local
      const todayMs  = new Date(todayStr + "T12:00:00").getTime();
      let nearestDate: string | null = null;
      let nearestDelta = Infinity;
      for (const g of games) {
        if (!g.date) continue;
        const d = Math.abs(new Date(g.date + "T12:00:00").getTime() - todayMs);
        if (d < nearestDelta) { nearestDelta = d; nearestDate = g.date; }
      }
      const target = nearestDate
        ? (el.querySelector(`[data-date='${nearestDate}']`) as HTMLElement | null)
        : null;
      if (target) {
        // Use getBoundingClientRect so the math is independent of offsetParent
        // (the scroll container has no `position: relative`, so target.offsetLeft
        // was measuring from an ancestor and inflating to maxSL on mobile —
        // which dumped the user at the far-right of the strip).
        const cRect = el.getBoundingClientRect();
        const tRect = target.getBoundingClientRect();
        const targetLeftInScroll = (tRect.left - cRect.left) + el.scrollLeft;
        const desired = targetLeftInScroll + target.clientWidth / 2 - el.clientWidth / 2;
        const maxSL  = Math.max(0, el.scrollWidth - el.clientWidth);
        el.scrollLeft = Math.max(0, Math.min(maxSL, desired));
        didInitialScrollRef.current = true;
      }
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
      {/* Label strip — border tints with the active team theme instead of
          always rendering gold. Uses --brand-* CSS vars set by ThemeProvider. */}
      <div className="flex items-center justify-center gap-2 h-6 relative"
        style={{
          borderBottom: "1px solid rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.20)",
          background: "linear-gradient(to bottom, rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.04), transparent)",
        }}>
        {theme && (
          <button
            onClick={clearTheme}
            className="absolute left-4 top-1/2 -translate-y-1/2 hud-mono text-[8px] font-black uppercase tracking-[0.10em] px-1.5 py-[1px] leading-none rounded-md border transition-all duration-200 whitespace-nowrap text-white/40 border-white/[0.12] bg-white/[0.03] hover:text-white/70 hover:border-white/[0.22] hover:bg-white/[0.06]"
          >
            reset
          </button>
        )}
        <span className={`h-2 w-2 rounded-full shrink-0 border transition-all duration-300 ${
          hasLive
            ? "bg-[#4ade80]/80 border-[#4ade80]/60 shadow-[0_0_7px_rgba(74,222,128,0.75)] animate-pulse"
            : "bg-[#ef4444]/70 border-[#ef4444]/50 shadow-[0_0_5px_rgba(239,68,68,0.45)]"
        }`} />
        <span className="hud-mono text-[10px] tracking-[0.25em] uppercase"
          style={{ color: hasLive ? "rgba(74,222,128,0.85)" : "rgba(200,205,210,0.50)" }}>
          LIVE GAMES
        </span>
        {hasLive && (
          <button
            onClick={toggleHideAll}
            className="absolute right-4 top-1/2 -translate-y-1/2 hud-mono text-[8px] font-black uppercase tracking-[0.10em] px-1.5 py-[1px] leading-none rounded-md border transition-all duration-200 whitespace-nowrap"
            style={{
              color: hideAll
                ? "rgb(var(--brand-r), var(--brand-g), var(--brand-b))"
                : "rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.6)",
              borderColor: hideAll
                ? "rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.55)"
                : "rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.25)",
              background: hideAll
                ? "rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.14)"
                : "rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.05)",
              boxShadow: hideAll
                ? "0 0 8px rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.20)"
                : "none",
            }}
          >
            {hideAll ? "show" : "hide"}
          </button>
        )}
      </div>

      {games.length > 0 && (
        <div className="flex items-center"
          style={{
            borderBottom: "1px solid rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.16)",
            background: "linear-gradient(to bottom, rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.025), transparent)",
          }}>
          <Arrow dir="left" onClick={() => scroll("left")} visible={canLeft} />
          {/* py-3 here gives the card's own drop shadow + the hud-interactive
              hover lift (-2px) and press scale (0.985) room to render fully
              without overflow-x-auto clipping the top or bottom of the card. */}
          <div
            ref={scrollRef}
            className="scroll-smooth-x flex items-center flex-1 overflow-x-auto gap-1.5 px-1 py-3 [justify-content:safe_center] sm:justify-start"
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
