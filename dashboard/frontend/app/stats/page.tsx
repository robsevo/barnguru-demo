"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { TEAM_COLORS, TEAM_SECONDARY, fmtPos, normalizePlayerName } from "@/utils/nhl";
import TeamLogoLink from "@/components/TeamLogoLink";
import { HudGrid } from "@/components/hud";

interface Leader {
  rank: number;
  id: number;
  name: string;
  first_name: string;
  last_name: string;
  team: string;
  position: string;
  number: number;
  headshot: string;
  value: number | string;
}

interface SkaterCategories {
  goals?: Leader[];
  assists?: Leader[];
  points?: Leader[];
  plusMinus?: Leader[];
  penaltyMinutes?: Leader[];
  powerPlayGoals?: Leader[];
  gameWinningGoals?: Leader[];
  toi?: Leader[];
  shots?: Leader[];
  shootingPctg?: Leader[];
}

interface GoalieCategories {
  wins?: Leader[];
  savePctg?: Leader[];
  goalsAgainstAverage?: Leader[];
  shutouts?: Leader[];
}

interface CalderCategories {
  points?: Leader[];
  goals?: Leader[];
  assists?: Leader[];
}

type PlayerTab  = "skaters" | "goalies" | "calder";
type SeasonType = "regular" | "playoffs";
type SkaterCat  = keyof SkaterCategories;
type GoalieCat  = keyof GoalieCategories;
type CalderCat  = keyof CalderCategories;

function fmtValue(value: number | string, cat: string): string {
  if (value === null || value === undefined) return "—";
  if (cat === "toi") {
    const secs = typeof value === "string" ? parseFloat(value) : value;
    const m = Math.floor(secs / 60);
    const s = Math.round(secs % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  }
  if (cat === "savePctg" || cat === "shootingPctg") {
    const n = typeof value === "string" ? parseFloat(value) : value;
    return (n * 100).toFixed(1) + "%";
  }
  if (cat === "goalsAgainstAverage") {
    const n = typeof value === "string" ? parseFloat(value) : value;
    return n.toFixed(2);
  }
  if (cat === "plusMinus") {
    const n = typeof value === "number" ? value : parseFloat(String(value));
    return n > 0 ? `+${n}` : String(n);
  }
  return String(value);
}

function numericValue(v: number | string): number {
  if (typeof v === "number") return v;
  return parseFloat(v) || 0;
}

function rankColor(rank: number): string {
  if (rank === 1) return "text-[#fbbf24]";
  if (rank === 2) return "text-[#d1d5db]";
  if (rank === 3) return "text-[#fb923c]";
  return "text-white/50";
}

function PlayerHeadshot({ src, name, team, size = 32 }: { src: string | null; name: string; team: string; size?: number }) {
  const [err, setErr] = useState(false);
  const teamColor     = TEAM_COLORS[team]    ?? "#ffffff";
  const secondaryColor = TEAM_SECONDARY[team] ?? "#1a1a2e";
  const ringStyle = {
    boxShadow: `0 0 0 2.5px ${teamColor}, 0 0 0 3px rgba(255,255,255,0.35), 0 0 14px ${teamColor}99, 0 8px 24px rgba(0,0,0,0.8)`,
  };

  if (!err && src) {
    return (
      <div className="jarvis-photo-frame rounded-full shrink-0 overflow-hidden relative"
        style={{ width: size, height: size, background: `linear-gradient(to bottom, ${secondaryColor} 0%, ${secondaryColor} 72%, #b0b0b0 100%)`, ...ringStyle }}>
        <img src={src} alt={name} width={size} height={size}
          className="h-full w-full object-cover"
          style={{ objectPosition: "50% 8%" }}
          onError={() => setErr(true)} />
      </div>
    );
  }
  return (
    <div style={{ width: size, height: size, background: `linear-gradient(to bottom, ${secondaryColor}, #1a1a2e)`, ...ringStyle }}
      className="rounded-full flex items-center justify-center shrink-0">
      <span className="text-[11px] font-black text-white/40">
        {name.split(" ").map(w => w[0]).slice(0, 2).join("")}
      </span>
    </div>
  );
}

function TeamLogo({ abbrev }: { abbrev: string }) {
  return <TeamLogoLink abbrev={abbrev} size={28} />;
}

function LeaderRow({ leader, cat, maxValue, onPlayerClick }: { leader: Leader; cat: string; maxValue: number; onPlayerClick: (name: string) => void }) {
  const teamColor = TEAM_COLORS[leader.team] ?? "#C9A84C";
  const val = numericValue(leader.value);
  const barPct = maxValue > 0 ? (val / maxValue) * 100 : 0;

  return (
    <tr className="border-b border-white/[0.04] hover:bg-white/[0.04] transition-colors group cursor-pointer" onClick={() => onPlayerClick(leader.name)}>
      {/* Rank */}
      <td className="px-3 sm:px-4 py-3 sm:py-2.5 w-10 text-right">
        <span className={`text-[13px] sm:text-[14px] font-black tabular-nums ${rankColor(leader.rank)}`}>{leader.rank}</span>
      </td>
      {/* Player */}
      <td className="px-2 sm:px-3 py-3 sm:py-2.5">
        <div className="flex items-center gap-2 sm:gap-3">
          <div className="w-0.5 h-6 sm:h-7 rounded-full shrink-0" style={{ backgroundColor: `${teamColor}70` }} />
          <span className="hidden sm:block"><PlayerHeadshot src={leader.headshot} name={leader.name} team={leader.team} size={52} /></span>
          <span className="block sm:hidden"><PlayerHeadshot src={leader.headshot} name={leader.name} team={leader.team} size={36} /></span>
          <div className="min-w-0">
            <div className="text-[13px] sm:text-[14px] font-bold text-white/85 whitespace-nowrap">{leader.name}</div>
            <div className="flex items-center gap-1.5 mt-0.5">
              <TeamLogo abbrev={leader.team} />
              <span className="text-[9px] sm:text-[10px] font-black uppercase px-1.5 py-0.5 rounded"
                style={{ backgroundColor: `${teamColor}20`, color: teamColor }}>
                {leader.team}
              </span>
              <span className="text-[9px] sm:text-[10px] text-white/30 hidden sm:inline">{fmtPos(leader.position)}</span>
            </div>
          </div>
        </div>
      </td>
      {/* Value + bar */}
      <td className="px-3 sm:px-5 py-3 sm:py-2.5 text-right">
        <div className="flex flex-col items-end gap-1.5">
          <span className={`text-[16px] sm:text-[18px] font-black tabular-nums leading-none ${rankColor(leader.rank)}`}>
            {fmtValue(leader.value, cat)}
          </span>
          <div className="w-16 sm:w-24 h-1 rounded-full bg-white/[0.06] overflow-hidden">
            <div className="h-full rounded-full" style={{
              width: `${barPct}%`,
              backgroundColor: leader.rank === 1 ? "#fbbf24" : leader.rank === 2 ? "#d1d5db" : leader.rank === 3 ? "#fb923c" : teamColor,
              opacity: leader.rank <= 3 ? 0.9 : 0.5,
            }} />
          </div>
        </div>
      </td>
    </tr>
  );
}

const SKATER_CATS: { key: SkaterCat; label: string }[] = [
  { key: "points",           label: "Points"   },
  { key: "goals",            label: "Goals"    },
  { key: "assists",          label: "Assists"  },
  { key: "plusMinus",        label: "+/-"      },
  { key: "penaltyMinutes",   label: "PIMs"     },
  { key: "powerPlayGoals",   label: "PPG"      },
  { key: "gameWinningGoals", label: "GWG"      },
  { key: "shots",            label: "Shots"    },
  { key: "shootingPctg",     label: "S%"       },
  { key: "toi",              label: "TOI"      },
];

const GOALIE_CATS: { key: GoalieCat; label: string }[] = [
  { key: "wins",                label: "Wins"     },
  { key: "savePctg",            label: "SV%"      },
  { key: "goalsAgainstAverage", label: "GAA"      },
  { key: "shutouts",            label: "Shutouts" },
];

const CALDER_CATS: { key: CalderCat; label: string }[] = [
  { key: "points",  label: "Points"  },
  { key: "goals",   label: "Goals"   },
  { key: "assists", label: "Assists" },
];

export default function StatsPage() {
  const router = useRouter();
  const handlePlayerClick = useCallback((name: string) => {
    router.push(`/players/${encodeURIComponent(normalizePlayerName(name))}`);
  }, [router]);

  const [seasonType, setSeasonType]   = useState<SeasonType>("playoffs");
  const [playerTab, setPlayerTab]     = useState<PlayerTab>("skaters");
  const [skaterCat, setSkaterCat]     = useState<SkaterCat>("points");
  const [goalieCat, setGoalieCat]     = useState<GoalieCat>("wins");
  const [calderCat, setCalderCat]     = useState<CalderCat>("points");
  const [skaterData, setSkaterData]   = useState<SkaterCategories>({});
  const [goalieData, setGoalieData]   = useState<GoalieCategories>({});
  const [calderData, setCalderData]   = useState<CalderCategories>({});
  const [loadingSkaters, setLoadingSkaters] = useState(true);
  const [loadingGoalies, setLoadingGoalies] = useState(true);
  const [loadingCalder,  setLoadingCalder]  = useState(true);
  const [errorSkaters, setErrorSkaters]     = useState<string | null>(null);
  const [errorGoalies, setErrorGoalies]     = useState<string | null>(null);
  const [errorCalder,  setErrorCalder]      = useState<string | null>(null);
  const [posFilter, setPosFilter]     = useState<Set<string>>(new Set());
  const catScrollRef = useRef<HTMLDivElement>(null);
  const [catCanLeft,  setCatCanLeft]  = useState(false);
  const [catCanRight, setCatCanRight] = useState(false);

  // Stale-while-revalidate: serve cached data instantly, then refresh in background
  const SWR_TTL = 5 * 60 * 1000; // 5 min matches server cache
  function swr<T>(key: string, setData: (d: T) => void, setLoading: (b: boolean) => void): T | null {
    try {
      const raw = localStorage.getItem(key);
      if (!raw) return null;
      const { ts, data } = JSON.parse(raw) as { ts: number; data: T };
      if (Date.now() - ts < SWR_TTL) { setData(data); setLoading(false); return data; }
      setData(data); // show stale while refetching
      return null; // still fresh-fetch
    } catch { return null; }
  }
  function swrSet(key: string, data: unknown) {
    try { localStorage.setItem(key, JSON.stringify({ ts: Date.now(), data })); } catch { /* quota */ }
  }

  const fetchSkaters = useCallback(async () => {
    const url = seasonType === "playoffs" ? "/api/skater-stats-playoffs" : "/api/skater-stats";
    const key = seasonType === "playoffs" ? "gretzky_skater_stats_playoffs" : "gretzky_skater_stats";
    const stale = swr<SkaterCategories>(key, setSkaterData, setLoadingSkaters);
    if (stale) return; // fresh cache — skip network
    setLoadingSkaters(true); setErrorSkaters(null);
    try {
      const res = await fetch(url);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      const cats = data.categories ?? {};
      setSkaterData(cats);
      swrSet(key, cats);
    } catch (e: unknown) {
      setErrorSkaters(e instanceof Error ? e.message : "Failed to load skater stats");
    } finally { setLoadingSkaters(false); }
  }, [seasonType]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchGoalies = useCallback(async () => {
    const url = seasonType === "playoffs" ? "/api/goalie-stats-playoffs" : "/api/goalie-stats";
    const key = seasonType === "playoffs" ? "gretzky_goalie_stats_playoffs" : "gretzky_goalie_stats";
    const stale = swr<GoalieCategories>(key, setGoalieData, setLoadingGoalies);
    if (stale) return;
    setLoadingGoalies(true); setErrorGoalies(null);
    try {
      const res = await fetch(url);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      const cats = data.categories ?? {};
      setGoalieData(cats);
      swrSet(key, cats);
    } catch (e: unknown) {
      setErrorGoalies(e instanceof Error ? e.message : "Failed to load goalie stats");
    } finally { setLoadingGoalies(false); }
  }, [seasonType]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchCalder = useCallback(async () => {
    // Calder is regular-season only — no playoff counterpart
    if (seasonType === "playoffs") return;
    const stale = swr<CalderCategories>("gretzky_calder_race", setCalderData, setLoadingCalder);
    if (stale) return;
    setLoadingCalder(true); setErrorCalder(null);
    try {
      const res = await fetch("/api/calder-race");
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      const cats = data.categories ?? {};
      setCalderData(cats);
      swrSet("gretzky_calder_race", cats);
    } catch (e: unknown) {
      setErrorCalder(e instanceof Error ? e.message : "Failed to load Calder race data");
    } finally { setLoadingCalder(false); }
  }, [seasonType]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { fetchSkaters(); }, [fetchSkaters]);
  useEffect(() => { fetchGoalies(); }, [fetchGoalies]);
  useEffect(() => { fetchCalder();  }, [fetchCalder]);

  // If user is on Calder tab and switches to playoffs, bounce back to skaters
  useEffect(() => {
    if (seasonType === "playoffs" && playerTab === "calder") setPlayerTab("skaters");
  }, [seasonType, playerTab]);

  useEffect(() => {
    const el = catScrollRef.current;
    if (!el) return;
    const update = () => {
      setCatCanLeft(el.scrollLeft > 0);
      setCatCanRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
    };
    update();
    el.addEventListener("scroll", update, { passive: true });
    return () => el.removeEventListener("scroll", update);
  }, [playerTab]);

  const isLoading =
    playerTab === "skaters" ? loadingSkaters :
    playerTab === "goalies" ? loadingGoalies :
    loadingCalder;
  const error =
    playerTab === "skaters" ? errorSkaters :
    playerTab === "goalies" ? errorGoalies :
    errorCalder;
  const currentLeaders: Leader[] =
    playerTab === "skaters" ? (skaterData[skaterCat] ?? []) :
    playerTab === "goalies" ? (goalieData[goalieCat] ?? []) :
    (calderData[calderCat] ?? []);
  const currentCat =
    playerTab === "skaters" ? skaterCat :
    playerTab === "goalies" ? goalieCat :
    calderCat;
  const currentLabel =
    playerTab === "skaters" ? (SKATER_CATS.find(c => c.key === skaterCat)?.label ?? skaterCat) :
    playerTab === "goalies" ? (GOALIE_CATS.find(c => c.key === goalieCat)?.label ?? goalieCat) :
    (CALDER_CATS.find(c => c.key === calderCat)?.label ?? calderCat);
  const displayedLeaders = posFilter.size === 0 || playerTab === "goalies"
    ? currentLeaders
    : currentLeaders.filter(l => {
        const p = fmtPos(l.position);
        // "D" pill matches D, LD, and RD since the NHL API doesn't always distinguish
        if (posFilter.has("D") && (p === "D" || p === "LD" || p === "RD")) return true;
        return posFilter.has(p);
      });
  const maxVal = displayedLeaders.length > 0 ? Math.max(...displayedLeaders.map(l => numericValue(l.value))) : 1;

  return (
    <main className="relative min-h-screen flex flex-col px-3 sm:px-8 pt-6 pb-12 max-w-[700px] sm:max-w-[1100px] mx-auto w-full overflow-x-hidden">
      <HudGrid />

      {/* HUD dossier strip */}
      <div className="relative z-10 mb-3 flex items-center gap-2 flex-wrap">
        <span className="hud-mono text-[10px] uppercase tracking-[0.20em] text-[var(--brand-hex)]" aria-hidden>◢</span>
        <span className="hud-mono text-[10px] uppercase tracking-[0.20em] text-[var(--brand-hex)]">
          STATS LEADERS · LEAGUE WIDE
        </span>
        <span className="hud-mono text-[9px] uppercase tracking-[0.16em] text-[var(--text-secondary)]">· official feed</span>
        <span className="ml-auto flex items-center gap-1">
          <span className="hud-pulse-dot" style={{ background: "#4ade80" }} />
          <span className="hud-mono jarvis-flicker text-[9px] uppercase tracking-[0.18em] text-[#4ade80]">LIVE</span>
        </span>
      </div>

      {/* Header — JARVIS dossier */}
      <div className="flex items-end justify-between mb-4 relative z-10">
        <div>
          <Link href="/" className="hud-mono text-[9px] uppercase tracking-[0.28em] text-white/30 hover:text-[var(--brand-hex)] transition-colors mb-1.5 block">
            ← GRTZKY
          </Link>
          <h1 className="text-[19px] sm:text-[34px] font-black tracking-[0.08em] sm:tracking-[0.10em] uppercase leading-none bg-clip-text text-transparent"
            style={{ backgroundImage: "linear-gradient(180deg, #ffffff 0%, #E8D090 38%, #C9A84C 58%, #6a5728 78%, #E8D090 100%)", filter: "drop-shadow(0 1px 0 rgba(0,0,0,0.85))" }}>
            ◢ STATS LEADERS ◣
          </h1>
        </div>
        <button onClick={() => playerTab === "skaters" ? fetchSkaters() : playerTab === "goalies" ? fetchGoalies() : fetchCalder()}
          title="Refresh"
          className="hud-mono text-[8px] sm:text-[10px] uppercase tracking-[0.14em] sm:tracking-[0.18em] px-1.5 sm:px-2.5 py-1 sm:py-1.5 rounded border transition-all"
          style={{
            color: "var(--brand-hex)",
            borderColor: "rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.40)",
            background: "linear-gradient(180deg, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.20) 0%, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.06) 100%)",
            boxShadow: "inset 0 1px 0 rgba(255,228,170,0.18), inset 0 -1px 0 rgba(0,0,0,0.45)",
          }}>
          ↻ SYNC
        </button>
      </div>

      {/* Season-type toggle — segmented metallic */}
      <div className="flex gap-1 mb-3 flex-wrap relative z-10">
        {([
          { id: "playoffs", label: "PLAYOFFS", color: "#fbbf24" },
          { id: "regular",  label: "2025–26 SEASON", color: "#a78bfa" },
        ] as const).map(opt => {
          const active = seasonType === opt.id;
          return (
            <button key={opt.id} onClick={() => { setSeasonType(opt.id); setPosFilter(new Set()); }}
              className="hud-mono px-4 py-1.5 rounded text-[11px] uppercase tracking-[0.18em] transition-all border"
              style={active ? {
                color: opt.color,
                borderColor: `${opt.color}88`,
                background: `linear-gradient(180deg, ${opt.color}30 0%, ${opt.color}12 100%)`,
                boxShadow: `inset 0 1px 0 rgba(255,255,255,0.18), inset 0 -1px 0 rgba(0,0,0,0.45), 0 0 12px ${opt.color}33`,
              } : {
                color: "rgba(255,255,255,0.40)",
                borderColor: "rgba(255,255,255,0.08)",
                background: "linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(0,0,0,0.25) 100%)",
              }}>
              {opt.label}
            </button>
          );
        })}
      </div>

      {/* Player-type tabs — metallic segmented */}
      <div className="flex gap-1 mb-4 flex-wrap relative z-10">
        {([
          { id: "skaters", label: "SKATERS",     color: "#C9A84C" },
          { id: "goalies", label: "GOALIES",      color: "#38bdf8" },
          ...(seasonType === "regular" ? [{ id: "calder" as const, label: "CALDER RACE", color: "#fbbf24" }] : []),
        ] as const).map(opt => {
          const active = playerTab === opt.id;
          return (
            <button key={opt.id} onClick={() => { setPlayerTab(opt.id); setPosFilter(new Set()); }}
              className="hud-mono px-4 py-1.5 rounded text-[11px] uppercase tracking-[0.18em] transition-all border"
              style={active ? {
                color: opt.color,
                borderColor: `${opt.color}88`,
                background: `linear-gradient(180deg, ${opt.color}28 0%, ${opt.color}10 100%)`,
                boxShadow: `inset 0 1px 0 rgba(255,255,255,0.18), inset 0 -1px 0 rgba(0,0,0,0.45), 0 0 10px ${opt.color}33`,
              } : {
                color: "rgba(255,255,255,0.40)",
                borderColor: "rgba(255,255,255,0.08)",
                background: "linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(0,0,0,0.25) 100%)",
              }}>
              {opt.label}
            </button>
          );
        })}
      </div>

      {/* Category tabs */}
      <div className="relative flex items-center mb-4">
        {catCanLeft && (
          <button onClick={() => catScrollRef.current?.scrollBy({ left: -160, behavior: "smooth" })}
            className="absolute left-0 z-10 h-full px-1 flex items-center bg-gradient-to-r from-[#0a0b0d] to-transparent text-white/40 hover:text-white/80 transition-colors">
            ❮
          </button>
        )}
        <div ref={catScrollRef} className="overflow-x-auto pb-1 flex-1" style={{ scrollbarWidth: "none" }}>
          <div className="flex gap-1 min-w-max">
            {(playerTab === "skaters" ? SKATER_CATS : playerTab === "goalies" ? GOALIE_CATS : CALDER_CATS).map(({ key, label }) => (
              <button key={key}
                onClick={() => {
                  if (playerTab === "skaters") setSkaterCat(key as SkaterCat);
                  else if (playerTab === "goalies") setGoalieCat(key as GoalieCat);
                  else setCalderCat(key as CalderCat);
                }}
                className={`px-3 py-1.5 rounded-lg text-[10px] font-black uppercase tracking-wide whitespace-nowrap transition-all ${
                  (playerTab === "skaters" ? skaterCat : playerTab === "goalies" ? goalieCat : calderCat) === key
                    ? "bg-white/[0.08] text-white/85 border border-white/[0.07]"
                    : "text-white/30 hover:text-white/50 border border-transparent"
                }`}
              >{label}</button>
            ))}
          </div>
        </div>
        {catCanRight && (
          <button onClick={() => catScrollRef.current?.scrollBy({ left: 160, behavior: "smooth" })}
            className="absolute right-0 z-10 h-full px-1 flex items-center bg-gradient-to-l from-[#0a0b0d] to-transparent text-white/40 hover:text-white/80 transition-colors">
            ❯
          </button>
        )}
      </div>

      {error && (
        <div className="rounded-xl border border-[#ef4444]/20 bg-[#ef4444]/[0.04] p-4 mb-4 text-[12px] text-[#ef4444]/70">{error}</div>
      )}

      {/* Rotate hint — mobile only */}
      <p className="sm:hidden text-[10px] text-white/25 text-center mb-3 flex items-center justify-center gap-1.5">
        <span>↻</span> Rotate phone for more stats
      </p>

      {/* Table */}
      <div className="rounded-xl border border-[#C9A84C]/[0.15] bg-gradient-to-b from-white/[0.015] via-white/[0.005] to-transparent overflow-hidden shadow-[0_8px_32px_rgba(0,0,0,0.7),0_2px_8px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(220,228,240,0.07),inset_0_-1px_0_rgba(0,0,0,0.35),0_0_0_1px_rgba(190,198,205,0.04)]">
        <div className="px-3 py-2.5 border-b border-white/[0.10] flex items-center justify-between">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[11px] font-black uppercase tracking-[0.22em] text-white/40">{currentLabel}</span>
            {playerTab === "calder" && (
              <span className="text-[8px] font-black uppercase px-1.5 py-0.5 rounded border border-[#fbbf24]/30 bg-[#fbbf24]/10 text-[#fbbf24]">Calder Race · Rookies Only</span>
            )}
            {playerTab !== "goalies" && (
              <div className="flex items-center gap-1 ml-1">
                {(["C","LW","RW","D"] as const).map(pos => {
                  const active = posFilter.has(pos);
                  return (
                    <button
                      key={pos}
                      onClick={() => setPosFilter(prev => {
                        const next = new Set(prev);
                        if (active) next.delete(pos); else next.add(pos);
                        return next;
                      })}
                      className={`text-[8px] font-black uppercase tracking-wide px-1.5 py-0.5 rounded border transition-colors ${
                        active
                          ? "border-white/30 bg-white/10 text-white/80"
                          : "border-white/[0.08] text-white/25 hover:text-white/50 hover:border-white/15"
                      }`}
                    >
                      {pos}
                    </button>
                  );
                })}
                {posFilter.size > 0 && (
                  <button
                    onClick={() => setPosFilter(new Set())}
                    className="text-[8px] text-white/20 hover:text-white/45 ml-0.5 transition-colors"
                    title="Clear filter"
                  >
                    ✕
                  </button>
                )}
              </div>
            )}
          </div>
          {!isLoading && currentLeaders.length > 0 && (
            <span className="text-[10px] text-white/38">
              Leader: <span className="font-bold text-white/50">{fmtValue(currentLeaders[0]?.value, currentCat)}</span>
            </span>
          )}
        </div>
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-white/[0.10]">
              <th className="px-3 py-2 text-right text-[9px] font-black uppercase tracking-wider text-white/38 w-8">#</th>
              <th className="px-2 py-2 text-left text-[9px] font-black uppercase tracking-wider text-white/38">Player</th>
              <th className="px-3 py-2 text-right text-[9px] font-black uppercase tracking-wider text-white/38">{currentLabel}</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              Array.from({ length: 10 }).map((_, i) => (
                <tr key={i} className="border-b border-white/[0.04]">
                  <td className="px-3 py-2.5"><div className="h-3 w-5 rounded bg-white/[0.06] animate-pulse ml-auto" /></td>
                  <td className="px-2 py-2.5">
                    <div className="flex items-center gap-2">
                      <div className="w-7 h-7 rounded-full bg-white/[0.06] animate-pulse shrink-0" />
                      <div className="flex flex-col gap-1.5">
                        <div className="h-3 w-28 rounded bg-white/[0.08] animate-pulse" />
                        <div className="h-2.5 w-16 rounded bg-white/[0.05] animate-pulse" />
                      </div>
                    </div>
                  </td>
                  <td className="px-3 py-2.5"><div className="h-4 w-12 rounded bg-white/[0.06] animate-pulse ml-auto" /></td>
                </tr>
              ))
            ) : displayedLeaders.length === 0 ? (
              <tr>
                <td colSpan={3} className="text-center py-10 text-[12px] text-white/38">No data available</td>
              </tr>
            ) : (
              displayedLeaders.map(leader => (
                <LeaderRow key={leader.id} leader={leader} cat={currentCat} maxValue={maxVal} onPlayerClick={handlePlayerClick} />
              ))
            )}
          </tbody>
        </table>
      </div>
    </main>
  );
}
