"use client";

import { useEffect, useRef, useState } from "react";
import { NHL_CODE, TEAM_COLORS, TEAM_SECONDARY, STRENGTH_STYLE } from "@/utils/nhl";

interface Goal {
  game_id: number;
  away_team: string;
  home_team: string;
  game_state: string;
  period: number;
  period_label: string;
  time: string;
  team: string;
  scorer: string;
  scorer_id: number | null;
  headshot_url: string | null;
  assists: string[];
  away_score: number | null;
  home_score: number | null;
  strength: string;
  scored_at: number | null;
}

function GoalLight({ state }: { state: "off" | "standby" | "on" }) {
  const glow =
    state === "on"      ? "drop-shadow(0 0 5px rgba(239,68,68,1)) drop-shadow(0 0 10px rgba(239,68,68,0.55))"
    : state === "standby" ? "drop-shadow(0 0 3px rgba(239,68,68,0.4))"
    : "none";
  const lens   = state === "on" ? "#ef4444"  : state === "standby" ? "#7f1d1d" : "#1a0505";
  const inner  = state === "on" ? "#ff6666"  : state === "standby" ? "#4d1010" : "#0d0202";
  const rim    = state === "on" ? "#cc2222"  : state === "standby" ? "#5a1111" : "#1a0505";
  const cage   = state === "on" ? "rgba(0,0,0,0.35)" : "rgba(0,0,0,0.55)";
  return (
    <svg width="16" height="16" viewBox="0 0 20 20" style={{ filter: glow, flexShrink: 0 }}>
      <circle cx="10" cy="10" r="9"   fill="#080303" stroke={rim} strokeWidth="0.7" />
      <circle cx="10" cy="10" r="7.5" fill={lens} />
      <circle cx="10" cy="10" r="5"   fill={inner} />
      {state === "on" && <circle cx="10" cy="10" r="2.5" fill="#ffbbbb" opacity="0.65" />}
      <g stroke={cage} strokeWidth="0.9">
        <line x1="2.5" y1="10"  x2="17.5" y2="10"  />
        <line x1="10"  y1="2.5" x2="10"   y2="17.5" />
        <line x1="4.9" y1="4.9" x2="15.1" y2="15.1" />
        <line x1="15.1" y1="4.9" x2="4.9" y2="15.1" />
      </g>
      <circle cx="10" cy="10" r="9" fill="none" stroke={rim} strokeWidth="0.7" />
    </svg>
  );
}

function teamLogoUrl(abbrev: string) {
  return `https://assets.nhle.com/logos/nhl/svg/${NHL_CODE[abbrev] ?? abbrev}_light.svg`;
}

function seasonYear() {
  const now = new Date();
  return now.getMonth() >= 9 ? now.getFullYear() : now.getFullYear() - 1;
}

function TeamLogo({ abbrev, size = 22 }: { abbrev: string; size?: number }) {
  const [err, setErr] = useState(false);
  if (err) return <span style={{ width: size, height: size }} className="text-[9px] font-semibold text-white/20 flex items-center">{abbrev.slice(0, 3)}</span>;
  return (
    <img src={teamLogoUrl(abbrev)} alt={abbrev} width={size} height={size}
      style={{ width: size, height: size }} className="shrink-0 object-contain"
      onError={() => setErr(true)} />
  );
}

function ScorerHeadshot({ playerId, headshotUrl, team, name, size = 48 }: {
  playerId: number | null;
  headshotUrl: string | null;
  team: string;
  name: string;
  size?: number;
}) {
  const [err, setErr] = useState(false);
  const yr = seasonYear();
  const teamColor = TEAM_COLORS[team] ?? "#ffffff";
  const secondaryColor = TEAM_SECONDARY[team] ?? "#1a1a2e";

  const url = !err
    ? (headshotUrl ?? (playerId
        ? `https://assets.nhle.com/mugs/nhl/${yr}${yr + 1}/${NHL_CODE[team] ?? team}/${playerId}.png`
        : null))
    : null;

  const ringStyle = { boxShadow: `0 0 0 2.5px ${teamColor}, 0 0 0 3px rgba(255,255,255,0.40), 0 0 14px ${teamColor}99, 0 8px 24px rgba(0,0,0,0.8)` };

  if (!url) {
    return (
      <div
        className="rounded-full bg-white/20 flex items-center justify-center shrink-0"
        style={{ width: size, height: size, ...ringStyle }}
      >
        <span className="text-sm font-semibold text-white/30">{name[0] ?? "?"}</span>
      </div>
    );
  }
  return (
    <div className="rounded-full shrink-0 overflow-hidden relative" style={{ width: size, height: size, background: `linear-gradient(to bottom, ${secondaryColor} 0%, ${secondaryColor} 72%, #b0b0b0 100%)`, ...ringStyle }}>
      <img src={url} alt={name} width={size} height={size}
        className="h-full w-full object-cover"
        style={{ objectPosition: "50% 8%" }}
        onError={() => setErr(true)} />
    </div>
  );
}


function formatScoredAt(epoch: number | null): string {
  if (!epoch) return "";
  try {
    return new Date(epoch * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  } catch { return ""; }
}

function goalKey(g: Goal) {
  // Use scorer_id + score state as key — stable even if timeInPeriod shifts between API calls
  if (g.scorer_id) return `${g.game_id}_${g.scorer_id}_${g.away_score}_${g.home_score}`;
  return `${g.game_id}_${g.period}_${g.team}_${g.away_score}_${g.home_score}`;
}

const STORAGE_KEY       = "gretzky_feed_enabled";
const STORAGE_NOTIF_KEY = "gretzky_feed_notifs";

export default function GoalFeed() {
  const [open, setOpen]               = useState(false);
  const [goals, setGoals]             = useState<Goal[]>([]);
  const [loaded, setLoaded]           = useState(false);
  const [flashing, setFlashing]       = useState(false);
  const [hasLive, setHasLive]         = useState(false);
  const [newKey, setNewKey]           = useState<string | null>(null);
  const [unread, setUnread]           = useState(0);
  const [unreadKeys, setUnreadKeys]   = useState<Set<string>>(new Set());
  const [feedEnabled, setFeedEnabled] = useState(true);
  const [notifEnabled, setNotifEnabled] = useState(true);
  const [stripInView, setStripInView] = useState(true);

  const seenKeys    = useRef<Set<string>>(new Set());
  const isFirstLoad = useRef(true);
  const stripRef    = useRef<HTMLDivElement>(null);
  const stripAnchorRef = useRef<HTMLDivElement>(null);

  // Load persisted preferences
  useEffect(() => {
    if (localStorage.getItem(STORAGE_KEY) === "false") setFeedEnabled(false);
    if (localStorage.getItem(STORAGE_NOTIF_KEY) === "false") setNotifEnabled(false);
  }, []);

  // Persist feed preference
  const toggleFeed = () => {
    setFeedEnabled(v => {
      const next = !v;
      localStorage.setItem(STORAGE_KEY, String(next));
      if (!next) setOpen(false);
      return next;
    });
  };

  // Track if strip is visible in viewport
  useEffect(() => {
    const el = stripAnchorRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => setStripInView(entry.isIntersecting),
      { threshold: 0.5 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    const fetch_ = () =>
      fetch("/api/goals")
        .then(r => r.json())
        .then(d => {
          const incoming: Goal[] = d.goals ?? [];
          const live = incoming.some(g => g.game_state === "LIVE" || g.game_state === "CRIT");
          setHasLive(live);

          if (isFirstLoad.current) {
            incoming.forEach(g => seenKeys.current.add(goalKey(g)));
            isFirstLoad.current = false;
            setGoals(incoming);
            setLoaded(true);
            return;
          }

          const fresh = incoming.filter(g => !seenKeys.current.has(goalKey(g)));
          if (fresh.length > 0) {
            fresh.forEach(g => seenKeys.current.add(goalKey(g)));
            setNewKey(goalKey(fresh[0]));
            setFlashing(true);
            setUnread(n => n + fresh.length);
            // Persist highlight — only if feed is currently closed (user hasn't seen them)
            setOpen(isOpen => {
              if (!isOpen) {
                setUnreadKeys(prev => new Set([...prev, ...fresh.map(goalKey)]));
              }
              return isOpen;
            });
            setTimeout(() => setFlashing(false), 1800);
            setTimeout(() => setNewKey(null), 3500);
          }

          setGoals(incoming);
          setLoaded(true);
        })
        .catch(() => { setLoaded(true); });

    fetch_();
    const id = setInterval(fetch_, 10_000);
    return () => clearInterval(id);
  }, []);

  // Corner badge only when: feed ON + notifs ON + strip scrolled out of view + something to show
  // When feed is explicitly disabled, never show the badge — it's off for a reason.
  const showCornerBadge = feedEnabled && notifEnabled && !stripInView && (hasLive || unread > 0 || flashing);

  const handleCornerBadgeClick = () => {
    if (!feedEnabled) {
      // Re-enable feed and scroll to top
      setFeedEnabled(true);
      localStorage.setItem(STORAGE_KEY, "true");
      setOpen(true);
      setUnread(0);
      setUnreadKeys(new Set());
      window.scrollTo({ top: 0, behavior: "smooth" });
    } else {
      // Scroll to top and open feed
      setOpen(true);
      setUnread(0);
      setUnreadKeys(new Set());
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  return (
    <>
      <style>{`
        @keyframes goalLight {
          0%   { background-color: transparent;           border-color: rgba(255,255,255,0.06); }
          12%  { background-color: rgba(220,38,38,0.20); border-color: rgba(220,38,38,0.75); }
          28%  { background-color: transparent;           border-color: rgba(255,255,255,0.06); }
          45%  { background-color: rgba(220,38,38,0.24); border-color: rgba(220,38,38,0.85); }
          64%  { background-color: transparent;           border-color: rgba(255,255,255,0.06); }
          78%  { background-color: rgba(220,38,38,0.15); border-color: rgba(220,38,38,0.55); }
          100% { background-color: transparent;           border-color: rgba(255,255,255,0.06); }
        }
        @keyframes goalRow {
          0%   { background-color: rgba(220,38,38,0.15); }
          100% { background-color: transparent; }
        }
        @keyframes badgePulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.7), 0 0 14px rgba(239,68,68,0.4); }
          50%       { box-shadow: 0 0 0 6px rgba(239,68,68,0), 0 0 24px rgba(239,68,68,0.6); }
        }
        @keyframes unreadPulse {
          0%, 100% { background-color: rgba(251,191,36,0.04); border-color: rgba(251,191,36,0.25); }
          50%      { background-color: rgba(251,191,36,0.12); border-color: rgba(251,191,36,0.55); }
        }
        .goal-flash      { animation: goalLight   1.8s ease-out forwards; }
        .goal-row-new    { animation: goalRow     3.5s ease-out forwards; }
        .goal-row-unread { border-left: 2px solid rgba(251,191,36,0.25); animation: unreadPulse 2.4s ease-in-out infinite; }
        .badge-pulse     { animation: badgePulse  1.4s ease-in-out infinite; }
      `}</style>

      {/* Anchor for intersection observer */}
      <div ref={stripAnchorRef} className="h-0 w-0 pointer-events-none" />

      <div ref={stripRef}>
          {/* Toggle strip */}
          <div className={`w-full flex items-center border-b h-7 transition-all duration-200 select-none ${flashing && feedEnabled ? "goal-flash" : "bg-[#0a0b0d]/80 backdrop-blur-sm border-white/[0.10]"}`}>
            {/* Left slot — notification bell toggle */}
            <div className="w-10 shrink-0 flex items-center justify-center">
              <button
                onClick={() => {
                  const next = !notifEnabled;
                  setNotifEnabled(next);
                  localStorage.setItem(STORAGE_NOTIF_KEY, String(next));
                }}
                title={notifEnabled ? "Mute corner notifications" : "Enable corner notifications"}
                className="text-[12px] leading-none transition-all duration-200"
                style={{ opacity: notifEnabled ? 0.55 : 0.2, filter: notifEnabled ? "none" : "grayscale(1)" }}
              >
                {notifEnabled ? "🔔" : "🔕"}
              </button>
            </div>

            {/* Center — GoalLight + text always centered together */}
            <button
              onClick={() => { setOpen(o => !o); setUnread(0); setUnreadKeys(new Set()); }}
              className="flex-1 flex items-center justify-center gap-1.5 h-full hover:bg-white/[0.04] transition-colors duration-200 group"
            >
              <GoalLight state={!feedEnabled ? "off" : flashing ? "on" : hasLive ? "standby" : "off"} />
              <span className={`text-[9px] font-semibold tracking-[0.25em] uppercase transition-colors duration-200 ${feedEnabled ? "text-white/20 group-hover:text-white/40" : "text-white/12"}`}>
                Live Goal Feed
              </span>
              {unread > 0 && (
                <span className={`flex items-center justify-center h-4 min-w-[16px] px-1 rounded-full text-[8px] font-black leading-none transition-all duration-200 ${feedEnabled ? "bg-[#ef4444] text-white shadow-[0_0_8px_rgba(239,68,68,0.7)]" : "bg-white/[0.08] text-white/25"}`}>
                  {unread > 9 ? "9+" : unread}
                </span>
              )}
              <span className={`text-[9px] font-semibold transition-colors duration-200 ${feedEnabled ? "text-white/20 group-hover:text-white/40" : "text-white/10"}`}>
                {open ? "▲" : "▼"}
              </span>
            </button>

            {/* Right slot — on/off toggle, same width as left slot */}
            <div className="w-10 shrink-0 flex items-center justify-center">
              <button
                onClick={toggleFeed}
                title={feedEnabled ? "Turn off feed" : "Turn on feed"}
                className="flex items-center"
              >
                <div className={`relative w-7 h-3.5 rounded-full transition-all duration-200 ${feedEnabled ? "bg-[#ef4444]/60" : "bg-white/10"}`}>
                  <div className={`absolute top-0.5 h-2.5 w-2.5 rounded-full transition-all duration-200 ${feedEnabled ? "right-0.5 bg-white shadow-[0_0_4px_rgba(239,68,68,0.8)]" : "left-0.5 bg-white/30"}`} />
                </div>
              </button>
            </div>
          </div>

          {/* Feed panel */}
          {open && feedEnabled && (
            <div className="bg-[#0a0b0d]/95 backdrop-blur-md border-b border-white/[0.10]">
              {!loaded ? (
                <p className="text-[9px] font-semibold uppercase tracking-widest text-white/20 animate-pulse text-center py-4">
                  Loading…
                </p>
              ) : goals.length === 0 ? (
                <p className="text-[9px] font-semibold uppercase tracking-widest text-white/10 text-center py-4">
                  No goals yet today
                </p>
              ) : (
                <div className="max-h-64 overflow-y-auto">
                  {goals.map((g, i) => {
                    const key         = goalKey(g);
                    const isNew       = key === newKey;
                    const isUnread    = unreadKeys.has(key);
                    const isLive      = g.game_state === "LIVE" || g.game_state === "CRIT";
                    const strengthCls = STRENGTH_STYLE[g.strength];
                    const gcHref = `/game/${g.game_id}`;
                    return (
                      <div
                        key={i}
                        className={`flex items-center justify-center gap-2 sm:gap-3 px-2 sm:px-4 py-2.5 sm:py-3 border-b border-white/[0.04] transition-all duration-200 ${
                          isNew ? "goal-row-new" : isUnread ? "goal-row-unread" : "hover:bg-white/[0.03]"
                        }`}
                      >
                        {/* Who scored */}
                        <a
                          href={gcHref}
                          className="flex flex-col items-center gap-1.5 group/scorer rounded-lg px-1 py-0.5 hover:bg-white/[0.05] transition-colors duration-150"
                          title="Open Live Games"
                        >
                          <div className="flex items-center gap-2">
                            <ScorerHeadshot
                              playerId={g.scorer_id}
                              headshotUrl={g.headshot_url}
                              team={g.team}
                              name={g.scorer}
                              size={36}
                            />
                            <div className="flex flex-col gap-0.5">
                              <div className="flex items-center gap-1.5">
                                <TeamLogo abbrev={g.team} size={24} />
                                <span className="text-[13px] sm:text-[15px] font-semibold text-white leading-tight group-hover/scorer:text-white/80 transition-colors">{g.scorer}</span>
                                {strengthCls && (
                                  <span className={`text-[9px] font-semibold border rounded px-1 py-0.5 shrink-0 ${strengthCls}`}>
                                    {g.strength}
                                  </span>
                                )}
                                {isLive && (
                                  <span className="w-2 h-2 rounded-full bg-[#4ade80] shadow-[0_0_6px_rgba(74,222,128,0.9)] animate-pulse shrink-0" />
                                )}
                              </div>
                              {g.assists.length > 0 && (
                                <span className="text-[11px] font-mono text-white/30 leading-tight">
                                  {g.assists.join(" · ")}
                                </span>
                              )}
                            </div>
                          </div>
                        </a>

                        <span className="self-center text-white/20 text-[14px] font-thin shrink-0">|</span>

                        {/* Game score + time */}
                        <a
                          href={gcHref}
                          className="flex flex-col items-center gap-1.5 group/gc rounded-lg px-2 py-1 hover:bg-white/[0.05] transition-colors duration-150"
                          title="Open Live Games"
                        >
                          <div className="flex items-center gap-1 sm:gap-1.5">
                            <TeamLogo abbrev={g.away_team} size={22} />
                            <span className="text-[13px] sm:text-[15px] font-semibold font-mono text-white tabular-nums">
                              {g.away_score ?? "–"}
                            </span>
                            <span className="text-white/20 font-semibold text-[10px] sm:text-[11px] mx-0.5">—</span>
                            <span className="text-[13px] sm:text-[15px] font-semibold font-mono text-white tabular-nums">
                              {g.home_score ?? "–"}
                            </span>
                            <TeamLogo abbrev={g.home_team} size={22} />
                          </div>
                          <div className="flex items-center gap-1 text-center">
                            <p className="text-[13px] font-semibold font-mono text-[#94a3b8] tabular-nums leading-tight">
                              {g.time}
                            </p>
                            <span className="text-white/20 text-[10px]">·</span>
                            <p className="text-[10px] font-semibold tracking-wider text-white/25 leading-tight">
                              {g.period_label}
                            </p>
                            <span className="text-[8px] text-white/10 group-hover/gc:text-white/30 transition-colors duration-150 ml-0.5">↗</span>
                          </div>
                        </a>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
      </div>

      {/* Fixed corner notification badge */}
      {showCornerBadge && (
        <button
          onClick={handleCornerBadgeClick}
          title={feedEnabled ? "Scroll to goal feed" : "Re-enable goal feed"}
          className={`fixed bottom-5 right-5 z-50 flex items-center gap-2 px-3 py-2 rounded-full
            bg-[#0d1117]/90 backdrop-blur-md border transition-all duration-300 cursor-pointer
            ${flashing
              ? "border-[#ef4444]/70 shadow-[0_0_20px_rgba(239,68,68,0.5)] badge-pulse"
              : "border-[#ef4444]/40 shadow-[0_0_12px_rgba(239,68,68,0.25)]"
            }`}
        >
          <GoalLight state={flashing ? "on" : "standby"} />
          <span className="text-[9px] font-black uppercase tracking-[0.2em] text-white/60">
            {flashing ? "GOAL" : "LIVE"}
          </span>
          {unread > 0 && (
            <span className="flex items-center justify-center h-4 min-w-[16px] px-1 rounded-full bg-[#ef4444] text-[8px] font-black text-white leading-none">
              {unread > 9 ? "9+" : unread}
            </span>
          )}
          {!feedEnabled && (
            <span className="text-[8px] font-semibold text-white/30 ml-0.5">↑ show</span>
          )}
        </button>
      )}
    </>
  );
}
