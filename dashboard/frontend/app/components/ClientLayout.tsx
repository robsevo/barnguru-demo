"use client";

import { useState, useEffect, useRef } from "react";
import { PiPProvider } from "@/utils/pipContext";
import { ThemeProvider, useTheme } from "@/utils/themeContext";
import { SeasonContextProvider } from "@/utils/contextToggle";
import FloatingPlayer   from "./FloatingPlayer";
import ThemeInjector    from "./ThemeInjector";
import TeamThemeLoader  from "./TeamThemeLoader";
import { AnimatePresence, motion } from "framer-motion";

function AboutDisclaimer() {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-white/[0.03] transition-colors duration-150"
      >
        <span className="text-[9px] font-semibold uppercase tracking-[0.22em] text-white/15">About</span>
        <span className="text-[8px] text-white/15">{open ? "▼" : "▶"}</span>
      </button>
      {open && (
        <div className="px-4 pb-3 space-y-2.5">
          <p className="text-[9px] font-mono leading-relaxed text-white/40">
            GRTZKY is a possession-level NHL game simulator built for research and pre-game analysis.
            It is not an automated betting tool — every market position requires human review.
          </p>
          <div className="rounded-lg border border-[#fbbf24]/20 bg-[#fbbf24]/[0.04] px-3 py-2 space-y-1.5">
            <p className="text-[9px] font-semibold uppercase tracking-[0.18em] text-[#fbbf24]/60">
              Ad Notice
            </p>
            <p className="text-[9px] font-mono leading-relaxed text-white/40">
              The Dev is working to eliminate all ads from this platform. In the meantime,
              some may slip through on desktop. Use{" "}
              <span className="text-white/40 font-semibold">Brave Browser</span> or an
              ad blocker extension until this is resolved.
            </p>
          </div>
          <p className="text-[9px] font-mono text-white/40 leading-relaxed">
            Data sourced from NHL API, MoneyPuck, and Polymarket. All analysis is
            for informational purposes only.
          </p>
        </div>
      )}
    </div>
  );
}
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";

// Mini neural-net icon for the mobile Cortex nav button
function MiniCortex() {
  return (
    <svg width="14" height="14" viewBox="0 0 48 48" fill="none" aria-hidden="true" className="shrink-0">
      {/* Hexagonal badge */}
      <path d="M24 2 L42 12 L42 36 L24 46 L6 36 L6 12 Z"
            stroke="#a78bfa" strokeWidth="1.5" strokeOpacity="0.30" fill="#a78bfa" fillOpacity="0.04"/>
      {/* Left hemisphere */}
      <path d="M24 10 C18 10 11 14 9 20 C7 25 9 31 13 34 C17 37 21 37 24 37"
            stroke="#a78bfa" strokeWidth="2.5" strokeLinecap="round" fill="rgba(167,139,250,0.08)"/>
      {/* Right hemisphere */}
      <path d="M24 10 C30 10 37 14 39 20 C41 25 39 31 35 34 C31 37 27 37 24 37"
            stroke="#a78bfa" strokeWidth="2.5" strokeLinecap="round" fill="rgba(167,139,250,0.08)"/>
      {/* Fold lines */}
      <path d="M11 19 C14 18 17 19 17 22" stroke="#c4b5fd" strokeWidth="1.8" fill="none" strokeOpacity="0.55" strokeLinecap="round"/>
      <path d="M37 19 C34 18 31 19 31 22" stroke="#c4b5fd" strokeWidth="1.8" fill="none" strokeOpacity="0.55" strokeLinecap="round"/>
      {/* Neural nodes */}
      <circle cx="15" cy="22" r="3" fill="#a78bfa" fillOpacity="0.88"/>
      <circle cx="33" cy="22" r="3" fill="#a78bfa" fillOpacity="0.88"/>
      {/* Central hub */}
      <circle cx="24" cy="24" r="5.5" fill="#a78bfa" fillOpacity="0.10"/>
      <circle cx="24" cy="24" r="3.8" fill="#a78bfa" fillOpacity="0.95"/>
      <circle cx="24" cy="24" r="1.9" fill="white"   fillOpacity="0.70"/>
    </svg>
  );
}

type PhaseStatus = "complete" | "in_progress" | "not_started";

const PHASES: { num: number; name: string; status: PhaseStatus }[] = [
  { num: 1,  name: "Data Pipeline",             status: "complete"    },
  { num: 2,  name: "Player Rating Models",       status: "complete"    },
  { num: 3,  name: "Fatigue Engine",             status: "complete"    },
  { num: 4,  name: "Coaching Tendency Models",   status: "not_started" },
  { num: 5,  name: "Rust Simulation Engine",     status: "not_started" },
  { num: 6,  name: "Lineup / Roster Forecaster", status: "not_started" },
  { num: 7,  name: "Single-Game Simulation",     status: "not_started" },
  { num: 8,  name: "Polymarket Edge Detection",  status: "not_started" },
  { num: 9,  name: "Historical Backtesting",     status: "not_started" },
  { num: 10, name: "Season Simulator",           status: "not_started" },
  { num: 11, name: "Live In-Game Simulation",    status: "not_started" },
  { num: 12, name: "Research Interface",         status: "not_started" },
  { num: 13, name: "Web Dashboard",              status: "not_started" },
  { num: 14, name: "Reinforcement Learning",     status: "not_started" },
  { num: 15, name: "Living Model",               status: "not_started" },
  { num: 16, name: "CV Tracking Engine",         status: "in_progress" },
  { num: 17, name: "Confidence Layer",           status: "complete"    },
];

const DOT_CLASS: Record<PhaseStatus, string> = {
  complete:    "bg-[#4ade80] shadow-[0_0_6px_rgba(74,222,128,0.9)]",
  in_progress: "bg-[#4ade80] shadow-[0_0_6px_rgba(74,222,128,0.9)]",
  not_started: "bg-white/[0.08]",
};

const NAME_CLASS: Record<PhaseStatus, string> = {
  complete:    "text-white/70",
  in_progress: "text-[#4ade80]/90",
  not_started: "text-white/20",
};

// Nav items — Live Games dot is rendered dynamically based on live status
const NAV_ITEMS = [
  { href: "/",             label: "Home",          dot: "static", dotClass: "bg-[#22d3ee]/75 shadow-[0_0_5px_rgba(34,211,238,0.55)]",               active: (p: string) => p === "/" },
  { href: "/gamecentre",   label: "Games",         dot: "live",   dotClass: "",                                                                      active: (p: string) => p === "/gamecentre" || p.startsWith("/game/") },
  { href: "/league/teams", label: "Teams",         dot: "static", dotClass: "bg-[#fb923c]/75 shadow-[0_0_5px_rgba(251,146,60,0.55)]",               active: (p: string) => p.startsWith("/league") || p.startsWith("/teams/") },
  { href: "/stats",        label: "Stats Leaders", dot: "static", dotClass: "bg-[#f472b6]/70 shadow-[0_0_5px_rgba(244,114,182,0.55)]",              active: (p: string) => p === "/stats" },
  { href: "/standings",    label: "Standings",     dot: "static", dotClass: "bg-[#fbbf24]/70 shadow-[0_0_5px_rgba(251,191,36,0.6)]",                active: (p: string) => p === "/standings" },
  { href: "/players",      label: "Cortex",        dot: "static", dotClass: "bg-[#a78bfa] shadow-[0_0_6px_rgba(167,139,250,0.8)]",                  active: (p: string) => p === "/players" || p.startsWith("/players/") },
  { href: "/barncentre",   label: "BarnCentre",    dot: "static", dotClass: "bg-[#C9A84C]/70 shadow-[0_0_5px_rgba(201,168,76,0.6)]",                active: (p: string) => p === "/barncentre" },
];

function NavIcon({ label }: { label: string }) {
  const common = { width: 13, height: 13, fill: "none" as const };
  if (label === "Home") {
    return (
      <svg {...common} viewBox="0 0 24 24" className="shrink-0" style={{ color: "#22d3ee" }}>
        <path d="M3 11L12 3L21 11V21H14V14H10V21H3V11Z" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    );
  }
  if (label === "Games") {
    return (
      <svg {...common} viewBox="0 0 24 24" className="shrink-0" style={{ color: "#4ade80" }}>
        <ellipse cx="12" cy="12" rx="8" ry="3.5" stroke="currentColor" strokeWidth="1.8" fill="rgba(74,222,128,0.12)"/>
        <path d="M4 12C4 14.2 7.6 16 12 16C16.4 16 20 14.2 20 12" stroke="currentColor" strokeWidth="1.8"/>
      </svg>
    );
  }
  if (label === "Teams") {
    return (
      <svg {...common} viewBox="0 0 24 24" className="shrink-0" style={{ color: "#fb923c" }}>
        <circle cx="9" cy="9" r="3" stroke="currentColor" strokeWidth="1.8"/>
        <circle cx="17" cy="10" r="2.4" stroke="currentColor" strokeWidth="1.6"/>
        <path d="M3 20C3 16.5 6 14.5 9 14.5C12 14.5 15 16.5 15 20" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
        <path d="M15.5 20C15.5 17.5 17 16 19 16C20.5 16 21.5 16.8 21.5 18.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    );
  }
  if (label === "Stats Leaders") {
    return (
      <svg {...common} viewBox="0 0 24 24" className="shrink-0" style={{ color: "#f472b6" }}>
        <circle cx="12" cy="8" r="4" stroke="currentColor" strokeWidth="1.8"/>
        <path d="M8.5 11L7 21L12 18L17 21L15.5 11" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    );
  }
  if (label === "Standings") {
    return (
      <svg {...common} viewBox="0 0 24 24" className="shrink-0" style={{ color: "#fbbf24" }}>
        <rect x="1.5" y="14" width="4" height="7" rx="0.8" stroke="currentColor" strokeWidth="1.6" fill="rgba(251,191,36,0.10)"/>
        <rect x="10" y="8" width="4" height="13" rx="0.8" stroke="currentColor" strokeWidth="1.6" fill="rgba(251,191,36,0.15)"/>
        <rect x="18.5" y="2" width="4" height="19" rx="0.8" stroke="currentColor" strokeWidth="1.6" fill="rgba(251,191,36,0.20)"/>
      </svg>
    );
  }
  if (label === "Cortex") {
    return (
      <svg width="13" height="13" viewBox="0 0 48 48" fill="none" className="shrink-0">
        <path d="M24 10 C18 10 11 14 9 20 C7 25 9 31 13 34 C17 37 21 37 24 37" stroke="#a78bfa" strokeWidth="2.5" strokeLinecap="round"/>
        <path d="M24 10 C30 10 37 14 39 20 C41 25 39 31 35 34 C31 37 27 37 24 37" stroke="#a78bfa" strokeWidth="2.5" strokeLinecap="round"/>
        <circle cx="24" cy="24" r="3.4" fill="#a78bfa" fillOpacity="0.95"/>
      </svg>
    );
  }
  if (label === "BarnCentre") {
    return (
      <svg width="13" height="12" viewBox="0 0 14 13" fill="none" xmlns="http://www.w3.org/2000/svg" className="shrink-0" style={{ color: "#C9A84C" }}>
        <rect x="1" y="2" width="12" height="7.5" rx="1" stroke="currentColor" strokeWidth="1.3"/>
        <line x1="4.5" y1="9.5" x2="3.5" y2="12" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
        <line x1="9.5" y1="9.5" x2="10.5" y2="12" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
        <line x1="3" y1="12" x2="11" y2="12" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
        <line x1="4.5" y1="2" x2="2.5" y2="0.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
        <line x1="9.5" y1="2" x2="11.5" y2="0.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
      </svg>
    );
  }
  return null;
}

function InnerLayout({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const [username, setUsername] = useState<string | null>(null);
  const [hasLive, setHasLive] = useState(false);
  const [statsDropOpen, setStatsDropOpen] = useState(false);
  const statsDropRef = useRef<HTMLDivElement>(null);
  const pathname = usePathname();
  const router = useRouter();
  const { theme, clearTheme, cortexPinned, setCortexPinned } = useTheme();

  async function handleLogout() {
    try {
      await fetch("/api/auth/logout", { method: "POST" });
    } catch {}
    setOpen(false);
    router.push("/login");
    router.refresh();
  }
  const isCortexPage = pathname.startsWith("/players");
  const isCortexMode = !theme && (isCortexPage || cortexPinned);
  const brandHex = theme?.primaryColor ?? (isCortexMode ? "#a78bfa" : "#C9A84C");
  const brandHex2 = theme ? lightenHex(theme.primaryColor) : (isCortexMode ? "#c4b5fd" : "#E8D090");

  useEffect(() => {
    fetch("/api/auth/me")
      .then((r) => r.json())
      .then((d) => setUsername(d.username ?? null))
      .catch(() => {});
  }, []);

  // Close stats dropdown when clicking outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (statsDropRef.current && !statsDropRef.current.contains(e.target as Node)) {
        setStatsDropOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // Close stats dropdown on navigation
  useEffect(() => { setStatsDropOpen(false); }, [pathname]);

  // Poll scoreboard to know if any games are currently live
  useEffect(() => {
    let id: ReturnType<typeof setInterval>;
    const check = () =>
      fetch("/api/scoreboard")
        .then(r => r.json())
        .then(d => {
          const live = (d.games ?? []).some(
            (g: { game_state: string }) => g.game_state === "LIVE" || g.game_state === "CRIT"
          );
          setHasLive(live);
          clearInterval(id);
          id = setInterval(check, live ? 15_000 : 60_000);
        })
        .catch(() => {});
    check();
    id = setInterval(check, 60_000);
    return () => clearInterval(id);
  }, []);

  return (
    <>
      <ThemeInjector />
      <TeamThemeLoader />
      <PiPProvider>
      <FloatingPlayer />
      <div className="relative flex min-h-screen overflow-x-hidden">
        {/* Backdrop */}
        {open && (
          <div
            className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm"
            onClick={() => setOpen(false)}
          />
        )}

        {/* Sidebar */}
        <aside
          className={`fixed top-0 left-0 z-50 h-full w-64 flex flex-col backdrop-blur-2xl border-r border-white/[0.08]
            shadow-[4px_0_40px_rgba(0,0,0,0.85),inset_-1px_0_0_rgba(218,226,236,0.07)]
            transition-transform duration-300 ease-in-out
            ${open ? "translate-x-0" : "-translate-x-full"}`}
          style={{
            background: `linear-gradient(180deg, rgba(var(--header-r), var(--header-g), var(--header-b), 0.99) 0%, rgba(var(--card-mid-r), var(--card-mid-g), var(--card-mid-b), 0.99) 100%)`,
          }}
        >
          {/* Sidebar header */}
          <div className="flex items-center justify-between px-4 py-3.5 border-b border-white/[0.07] shrink-0">
            <Link href="/" onClick={() => setOpen(false)} className="flex items-center gap-2 pr-1">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              {theme ? (
                <img src={theme.logoUrl} alt={theme.abbrev} className="h-8 w-auto object-contain" />
              ) : isCortexMode ? (
                <svg width="30" height="30" viewBox="0 0 48 48" fill="none" aria-hidden="true" className="shrink-0">
                  <path d="M24 2 L42 12 L42 36 L24 46 L6 36 L6 12 Z" stroke="#a78bfa" strokeWidth="1" strokeOpacity="0.30" fill="#a78bfa" fillOpacity="0.04"/>
                  <path d="M24 10 C18 10 11 14 9 20 C7 25 9 31 13 34 C17 37 21 37 24 37" stroke="#a78bfa" strokeWidth="2" strokeLinecap="round" fill="rgba(167,139,250,0.07)"/>
                  <path d="M24 10 C30 10 37 14 39 20 C41 25 39 31 35 34 C31 37 27 37 24 37" stroke="#a78bfa" strokeWidth="2" strokeLinecap="round" fill="rgba(167,139,250,0.07)"/>
                  <path d="M11 19 C14 18 17 19 17 22" stroke="#c4b5fd" strokeWidth="1.3" fill="none" strokeOpacity="0.50" strokeLinecap="round"/>
                  <path d="M37 19 C34 18 31 19 31 22" stroke="#c4b5fd" strokeWidth="1.3" fill="none" strokeOpacity="0.50" strokeLinecap="round"/>
                  <circle cx="15" cy="22" r="2.2" fill="#a78bfa" fillOpacity="0.88"/>
                  <circle cx="33" cy="22" r="2.2" fill="#a78bfa" fillOpacity="0.88"/>
                  <circle cx="24" cy="24" r="3.2" fill="#a78bfa" fillOpacity="0.95"/>
                  <circle cx="24" cy="24" r="1.6" fill="white"   fillOpacity="0.72"/>
                </svg>
              ) : (
                <img src="/logo-circle.png" alt="GRTZKY" className="h-8 w-auto" />
              )}
              <span className="text-white/20 font-thin text-sm">|</span>
              <span
                className="font-black italic tracking-[0.08em] text-base bg-clip-text text-transparent"
                style={{
                  fontFamily: "var(--font-condensed)",
                  backgroundImage: `linear-gradient(to right, #fff, ${brandHex2}, ${brandHex})`,
                }}
              >
                {theme ? theme.abbrev : isCortexMode ? "CORTEX" : "GRTZKY"}
              </span>
            </Link>
            <button
              onClick={() => setOpen(false)}
              className="text-white/25 hover:text-white/60 transition-colors duration-200 text-base leading-none px-1"
            >
              ✕
            </button>
          </div>

          {/* Nav */}
          <nav className="flex-1 overflow-y-auto px-2 py-2 space-y-px">
            <p className="px-3 pt-1 pb-1.5 text-[9px] font-semibold uppercase tracking-[0.22em] text-white/15">
              Navigation
            </p>
            {NAV_ITEMS.map(({ href, label, active, dot, dotClass }) => {
              const isActive = active(pathname);
              const liveDot = dot === "live"
                ? hasLive
                  ? "bg-[#4ade80] shadow-[0_0_6px_rgba(74,222,128,0.9)] animate-pulse"
                  : "bg-[#4ade80]/40 shadow-[0_0_4px_rgba(74,222,128,0.30)]"
                : dotClass;
              return (
                <Link key={href} href={href} onClick={() => setOpen(false)} className="block">
                  <div className={`flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all duration-150 ${isActive ? "bg-white/[0.08]" : "hover:bg-white/[0.04]"}`}>
                    <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${liveDot}`} />
                    <span className="text-[9px] font-semibold font-mono text-white/15 w-5 shrink-0">—</span>
                    <span className={`text-[11px] font-medium leading-tight flex items-center gap-1.5 ${isActive ? "text-white/90" : "text-white/60"}`}>
                      <NavIcon label={label} />
                      {label}
                    </span>
                  </div>
                </Link>
              );
            })}

            {/* Logout */}
            <button onClick={handleLogout} className="block w-full text-left">
              <div className="flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all duration-150 hover:bg-white/[0.04]">
                <span className="h-1.5 w-1.5 rounded-full shrink-0 bg-[#f87171]/70 shadow-[0_0_5px_rgba(248,113,113,0.6)]" />
                <span className="text-[9px] font-semibold font-mono text-white/15 w-5 shrink-0">—</span>
                <span className="text-[11px] font-medium leading-tight text-white/60 flex items-center gap-1.5">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" className="shrink-0" style={{ color: "#f87171" }}>
                    <path d="M14 7V5.5C14 4.7 13.3 4 12.5 4H5.5C4.7 4 4 4.7 4 5.5V18.5C4 19.3 4.7 20 5.5 20H12.5C13.3 20 14 19.3 14 18.5V17" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M10 12H21M21 12L17.5 8.5M21 12L17.5 15.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  Logout
                </span>
              </div>
            </button>

            {/* Team theme section */}
            {theme && (
              <>
                <div className="mx-3 mt-3 mb-1.5 border-t border-white/[0.06]" />
                <p className="px-3 pt-1 pb-1.5 text-[9px] font-semibold uppercase tracking-[0.22em] text-white/15">
                  Theme
                </p>
                <div className="mx-2 px-3 py-2 rounded-lg"
                  style={{ background: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.06)`, border: `1px solid rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.15)` }}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={theme.logoUrl} alt={theme.abbrev} width={20} height={20} className="object-contain" />
                      <span className="text-[10px] font-bold" style={{ color: brandHex }}>{theme.abbrev}</span>
                    </div>
                    <button
                      onClick={() => { clearTheme(); setOpen(false); }}
                      className="text-[8px] font-semibold uppercase tracking-wider px-2 py-1 rounded border text-white/30 border-white/[0.10] hover:text-[#f87171]/70 hover:border-[#f87171]/30 transition-all duration-150"
                    >
                      Reset
                    </button>
                  </div>
                </div>
              </>
            )}

            {/* Cortex theme section — shown when Cortex is pinned and no team theme */}
            {!theme && cortexPinned && (
              <>
                <div className="mx-3 mt-3 mb-1.5 border-t border-white/[0.06]" />
                <p className="px-3 pt-1 pb-1.5 text-[9px] font-semibold uppercase tracking-[0.22em] text-white/15">
                  Theme
                </p>
                <div className="mx-2 px-3 py-2 rounded-lg"
                  style={{ background: `rgba(167,139,250,0.06)`, border: `1px solid rgba(167,139,250,0.18)` }}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <svg width="16" height="16" viewBox="0 0 48 48" fill="none" aria-hidden="true">
                        <path d="M24 10 C18 10 11 14 9 20 C7 25 9 31 13 34 C17 37 21 37 24 37" stroke="#a78bfa" strokeWidth="2" strokeLinecap="round" fill="none"/>
                        <path d="M24 10 C30 10 37 14 39 20 C41 25 39 31 35 34 C31 37 27 37 24 37" stroke="#a78bfa" strokeWidth="2" strokeLinecap="round" fill="none"/>
                        <circle cx="24" cy="24" r="3.2" fill="#a78bfa" fillOpacity="0.95"/>
                      </svg>
                      <span className="text-[10px] font-bold text-[#a78bfa]">CORTEX</span>
                    </div>
                    <button
                      onClick={() => { setCortexPinned(false); setOpen(false); }}
                      className="text-[8px] font-semibold uppercase tracking-wider px-2 py-1 rounded border text-white/30 border-white/[0.10] hover:text-[#f87171]/70 hover:border-[#f87171]/30 transition-all duration-150"
                    >
                      GRTZKY
                    </button>
                  </div>
                </div>
              </>
            )}

            {/* Dev + Phases — rob only */}
            {username === "rob" && (
              <>
                <div className="mx-3 mt-3 mb-1.5 border-t border-white/[0.06]" />
                <p className="px-3 pt-1 pb-1.5 text-[9px] font-semibold uppercase tracking-[0.22em] text-white/15">
                  Dev
                </p>
                <Link
                  href="/dev"
                  onClick={() => setOpen(false)}
                  className="block"
                >
                  <div className={`flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all duration-150
                    ${pathname === "/dev" ? "bg-white/[0.08]" : "hover:bg-white/[0.04]"}`}>
                    <span className="h-1.5 w-1.5 rounded-full shrink-0 bg-white/20" />
                    <span className="text-[9px] font-semibold font-mono text-white/15 w-5 shrink-0">—</span>
                    <span className="text-[11px] font-medium leading-tight text-white/40">Dev Dashboard</span>
                  </div>
                </Link>

                <p className="px-3 pt-3 pb-1.5 text-[9px] font-semibold uppercase tracking-[0.22em] text-white/15">
                  Phases
                </p>
                {PHASES.map((p) => {
                  const href = `/phase${p.num}`;
                  const active = pathname === href;
                  const linkable = p.status !== "not_started";

                  const row = (
                    <div
                      className={`flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all duration-150
                        ${active ? "bg-white/[0.08]" : linkable ? "hover:bg-white/[0.04]" : ""}`}
                    >
                      <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${DOT_CLASS[p.status]}`} />
                      <span className="text-[9px] font-semibold font-mono text-white/15 w-5 shrink-0 tabular-nums">
                        P{p.num}
                      </span>
                      <span className={`text-[11px] font-medium leading-tight ${NAME_CLASS[p.status]}`}>
                        {p.name}
                      </span>
                    </div>
                  );

                  return linkable ? (
                    <Link key={p.num} href={href} onClick={() => setOpen(false)} className="block">
                      {row}
                    </Link>
                  ) : (
                    <div key={p.num} className="cursor-default">
                      {row}
                    </div>
                  );
                })}
              </>
            )}
          </nav>

          {/* About / disclaimer */}
          <div className="border-t border-white/[0.07] shrink-0">
            <AboutDisclaimer />
          </div>

          {/* Sidebar footer */}
          <div className="px-4 py-3 border-t border-white/[0.10] shrink-0 flex items-center justify-center">
            <p className="text-[8px] font-semibold uppercase tracking-[0.2em] text-white/10">
              GRTZKY
            </p>
          </div>
        </aside>

        {/* Main content */}
        <div
          className={`flex-1 flex flex-col min-w-0 overflow-x-hidden transition-all duration-300 ease-in-out
            ${open ? "ml-64" : "ml-0"}`}
        >
          {/* Top bar — solid navy matching the Stats dropdown background */}
          <header
            className="relative sticky top-0 z-30 flex items-center gap-2 px-2 sm:px-3 h-11 backdrop-blur-2xl select-none"
            style={{
              // Always navy regardless of team theme — team color stays on accents only
              background: "rgba(10,12,16,0.97)",
              borderBottom: `1px solid rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.35)`,
              boxShadow: `
                0 1px 0 rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.22),
                0 6px 36px rgba(0,0,0,0.80),
                0 0 60px rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.06),
                inset 0 2px 0 rgba(220,228,240,0.18),
                inset 0 1px 0 rgba(255,255,255,0.10),
                inset 0 -1px 0 rgba(0,0,0,0.45),
                inset 0 -8px 12px -8px rgba(0,0,0,0.50),
                inset 1px 0 0 rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.10),
                inset -1px 0 0 rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.10)
              `,
            }}
          >
            {/* metallic top-edge highlight */}
            <span aria-hidden className="absolute top-0 left-0 right-0 h-px pointer-events-none z-10"
              style={{ background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent)" }} />

            {/* HUD corner brackets — top-left + top-right */}
            <span aria-hidden className="absolute pointer-events-none z-10"
              style={{
                top: 3, left: 3, width: 10, height: 10,
                borderTop: `1px solid rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.65)`,
                borderLeft: `1px solid rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.65)`,
              }} />
            <span aria-hidden className="absolute pointer-events-none z-10"
              style={{
                top: 3, right: 3, width: 10, height: 10,
                borderTop: `1px solid rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.65)`,
                borderRight: `1px solid rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.65)`,
              }} />

            {/* Hamburger — metallic JARVIS button with corner brackets */}
            <button
              onClick={() => setOpen((o) => !o)}
              className="relative flex flex-col justify-center items-center gap-[3px] w-6 h-6 rounded-md shrink-0 transition-all duration-200 active:scale-[0.92] group"
              style={{
                background: open
                  ? `linear-gradient(180deg, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.32) 0%, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.14) 50%, rgba(0,0,0,0.30) 100%)`
                  : `linear-gradient(180deg, rgba(220,228,240,0.08) 0%, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.08) 50%, rgba(0,0,0,0.30) 100%)`,
                border: `1px solid rgba(var(--brand-r),var(--brand-g),var(--brand-b),${open ? 0.55 : 0.30})`,
                boxShadow: open
                  ? `0 0 10px rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.30), inset 0 1px 0 rgba(255,255,255,0.18), inset 0 -1px 0 rgba(0,0,0,0.45)`
                  : `inset 0 1px 0 rgba(255,255,255,0.10), inset 0 -1px 0 rgba(0,0,0,0.40)`,
              }}
              aria-label="Toggle navigation"
            >
              {/* HUD corner brackets — top-left + bottom-right */}
              <span aria-hidden className="absolute pointer-events-none"
                style={{
                  top: 2, left: 2, width: 5, height: 5,
                  borderTop: `1px solid rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.70)`,
                  borderLeft: `1px solid rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.70)`,
                }} />
              <span aria-hidden className="absolute pointer-events-none"
                style={{
                  bottom: 2, right: 2, width: 5, height: 5,
                  borderBottom: `1px solid rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.70)`,
                  borderRight: `1px solid rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.70)`,
                }} />
              {(["rotate(45deg) translateY(5px)", undefined, "rotate(-45deg) translateY(-5px)"] as const).map((transform, i) => (
                <span key={i} className="block h-[1.5px] w-[12px] rounded-full transition-all duration-200 origin-center"
                  style={{
                    background: theme
                      ? "linear-gradient(90deg, rgba(255,255,255,0.40), rgba(255,255,255,0.85) 50%, rgba(255,255,255,0.40))"
                      : `linear-gradient(90deg, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.45), rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.95) 50%, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.45))`,
                    boxShadow: theme
                      ? "0 0 4px rgba(255,255,255,0.30)"
                      : `0 0 4px rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.40)`,
                    transform: open ? transform : undefined,
                    opacity: (i === 1 && open) ? 0 : 1,
                  }} />
              ))}
            </button>

            {/* Logo + name */}
            <Link href="/" className="inline-flex items-center shrink-0 pr-1">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              {theme ? (
                <img src={theme.logoUrl} alt={theme.abbrev} className="h-9 sm:h-8 w-auto object-contain block" />
              ) : isCortexMode ? (
                <svg width="40" height="40" viewBox="0 0 48 48" fill="none" aria-hidden="true" className="shrink-0 sm:!w-9 sm:!h-9 lg:!w-[42px] lg:!h-[42px]">
                  <path d="M24 2 L42 12 L42 36 L24 46 L6 36 L6 12 Z" stroke="#a78bfa" strokeWidth="0.9" strokeOpacity="0.28" fill="#a78bfa" fillOpacity="0.03"/>
                  <circle cx="24" cy="24" r="17" stroke="#a78bfa" strokeWidth="0.6" strokeOpacity="0.12" fill="none"/>
                  <path d="M24 10 C18 10 11 14 9 20 C7 25 9 31 13 34 C17 37 21 37 24 37" stroke="#a78bfa" strokeWidth="1.8" strokeLinecap="round" fill="rgba(167,139,250,0.07)"/>
                  <path d="M24 10 C30 10 37 14 39 20 C41 25 39 31 35 34 C31 37 27 37 24 37" stroke="#a78bfa" strokeWidth="1.8" strokeLinecap="round" fill="rgba(167,139,250,0.07)"/>
                  <line x1="24" y1="10" x2="24" y2="37" stroke="#c4b5fd" strokeWidth="0.9" strokeOpacity="0.18" strokeDasharray="2.5,3.5"/>
                  <path d="M11 19 C14 18 17 19 17 22" stroke="#c4b5fd" strokeWidth="1.2" fill="none" strokeOpacity="0.55" strokeLinecap="round"/>
                  <path d="M10 27 C13 26 16 27 16 30" stroke="#c4b5fd" strokeWidth="1.1" fill="none" strokeOpacity="0.40" strokeLinecap="round"/>
                  <path d="M37 19 C34 18 31 19 31 22" stroke="#c4b5fd" strokeWidth="1.2" fill="none" strokeOpacity="0.55" strokeLinecap="round"/>
                  <path d="M38 27 C35 26 32 27 32 30" stroke="#c4b5fd" strokeWidth="1.1" fill="none" strokeOpacity="0.40" strokeLinecap="round"/>
                  <line x1="15" y1="22" x2="24" y2="24" stroke="#a78bfa" strokeWidth="0.8" strokeOpacity="0.28"/>
                  <line x1="33" y1="22" x2="24" y2="24" stroke="#a78bfa" strokeWidth="0.8" strokeOpacity="0.28"/>
                  <circle cx="15" cy="22" r="2.2" fill="#a78bfa" fillOpacity="0.88"/>
                  <circle cx="33" cy="22" r="2.2" fill="#a78bfa" fillOpacity="0.88"/>
                  <circle cx="14" cy="30" r="1.8" fill="#7c3aed" fillOpacity="0.75"/>
                  <circle cx="34" cy="30" r="1.8" fill="#7c3aed" fillOpacity="0.75"/>
                  <circle cx="18" cy="14" r="1.3" fill="#c4b5fd" fillOpacity="0.60"/>
                  <circle cx="30" cy="14" r="1.3" fill="#c4b5fd" fillOpacity="0.60"/>
                  <circle cx="24" cy="24" r="5.0" fill="#a78bfa" fillOpacity="0.10"/>
                  <circle cx="24" cy="24" r="3.2" fill="#a78bfa" fillOpacity="0.95"/>
                  <circle cx="24" cy="24" r="1.6" fill="white"   fillOpacity="0.72"/>
                </svg>
              ) : (
                <img
                  src="/logo-circle.png"
                  alt="GRTZKY"
                  className="h-9 sm:h-8 w-auto block"
                  style={{
                    filter: `drop-shadow(0 1px 1px rgba(0,0,0,0.65))
                             drop-shadow(0 0 4px rgba(255,255,255,0.20))
                             drop-shadow(0 0 8px rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.30))`,
                  }}
                />
              )}
              <span
                className="h-5 sm:h-5 w-px shrink-0 mx-1.5"
                style={{ background: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.30)` }}
              />
              <span
                className="font-black italic tracking-[0.06em] text-[14px] sm:text-[17px] lg:text-[20px] bg-clip-text text-transparent leading-none pr-1.5"
                style={{
                  fontFamily: "var(--font-condensed)",
                  // Brushed-steel multi-stop gradient with highlight band
                  backgroundImage: isCortexMode
                    ? `linear-gradient(180deg,
                        #ffffff 0%,
                        #ddd6fe 18%,
                        #a78bfa 38%,
                        #7c3aed 58%,
                        #a78bfa 78%,
                        #ddd6fe 100%)`
                    : `linear-gradient(180deg,
                        #ffffff 0%,
                        ${brandHex2} 18%,
                        ${brandHex} 38%,
                        #6a5728 58%,
                        ${brandHex} 78%,
                        ${brandHex2} 100%)`,
                  // Subtle outer + inner glow via text-shadow on the bg-clipped text
                  filter: isCortexMode
                    ? `drop-shadow(0 1px 0 rgba(0,0,0,0.85)) drop-shadow(0 0 6px rgba(167,139,250,0.30))`
                    : `drop-shadow(0 1px 0 rgba(0,0,0,0.85)) drop-shadow(0 0 6px rgba(201,168,76,0.30))`,
                }}
              >
                {theme ? theme.abbrev : isCortexMode ? "CORTEX" : "GRTZKY"}
              </span>
            </Link>

            <span className="hidden lg:block h-4 w-px mx-0.5" style={{ background: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.20)` }} />
            <span
              className="hud-mono text-[9px] tracking-[0.20em] uppercase hidden lg:block"
              style={{ color: theme ? "rgba(255,255,255,0.45)" : `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.45)` }}
            >
              {theme ? "" : isCortexMode ? "◢ NEURAL CORE · v1.0" : "◢ BAYESIAN ANALYTICS · RATING NETWORK"}
            </span>

            {/* Quick-nav buttons */}
            <div className="ml-auto flex items-center gap-0.5 sm:gap-1 shrink-0">
              {(
                [
                  { href: "/gamecentre",   labelSm: "Games",  label: "Games",  active: pathname === "/gamecentre" || pathname.startsWith("/game/") },
                ] as const
              ).map(({ href, labelSm, label, active }) => {
                return (
                  <Link
                    key={href}
                    href={href}
                    className="hud-mono px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-md text-[8px] sm:text-[9px] uppercase tracking-[0.14em] transition-all duration-150 border flex items-center justify-center"
                    style={active ? (theme ? {
                      background: "linear-gradient(180deg, rgba(255,255,255,0.30) 0%, rgba(255,255,255,0.14) 50%, rgba(0,0,0,0.10) 100%)",
                      color: "#ffffff",
                      borderColor: "rgba(255,255,255,0.50)",
                      boxShadow: "0 0 10px rgba(255,255,255,0.18), inset 0 1px 0 rgba(255,255,255,0.30), inset 0 -1px 0 rgba(0,0,0,0.35)",
                    } : {
                      background: `linear-gradient(180deg, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.32) 0%, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.14) 50%, rgba(0,0,0,0.20) 100%)`,
                      color: brandHex,
                      borderColor: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.50)`,
                      boxShadow: `0 0 10px rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.20), inset 0 1px 0 rgba(255,255,255,0.18), inset 0 -1px 0 rgba(0,0,0,0.45)`,
                    }) : (theme ? {
                      color: "rgba(255,255,255,0.60)",
                      borderColor: "rgba(255,255,255,0.18)",
                      background: "linear-gradient(180deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.04) 50%, rgba(0,0,0,0.20) 100%)",
                      boxShadow: "inset 0 1px 0 rgba(255,255,255,0.10), inset 0 -1px 0 rgba(0,0,0,0.35)",
                    } : {
                      color: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.55)`,
                      borderColor: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.22)`,
                      background: `linear-gradient(180deg, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.10) 0%, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.03) 50%, rgba(0,0,0,0.25) 100%)`,
                      boxShadow: `inset 0 1px 0 rgba(220,228,240,0.08), inset 0 -1px 0 rgba(0,0,0,0.40)`,
                    })}
                  >
                    <span className="sm:hidden">{labelSm}</span>
                    <span className="hidden sm:inline">{label}</span>
                  </Link>
                );
              })}

              {/* Stats + Standings split-button dropdown */}
              {(() => {
                const statsActive  = pathname === "/stats";
                const stndActive   = pathname === "/standings";
                const teamsActive  = pathname.startsWith("/league") || pathname.startsWith("/teams/");
                const anyActive    = statsActive || stndActive || teamsActive;
                const activeStyle = anyActive ? (theme ? {
                  background: "linear-gradient(180deg, rgba(255,255,255,0.30) 0%, rgba(255,255,255,0.14) 50%, rgba(0,0,0,0.10) 100%)",
                  color: "#ffffff",
                  borderColor: "rgba(255,255,255,0.50)",
                  boxShadow: "0 0 10px rgba(255,255,255,0.18), inset 0 1px 0 rgba(255,255,255,0.30), inset 0 -1px 0 rgba(0,0,0,0.35)",
                } : {
                  background: `linear-gradient(180deg, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.32) 0%, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.14) 50%, rgba(0,0,0,0.20) 100%)`,
                  color: brandHex,
                  borderColor: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.50)`,
                  boxShadow: `0 0 10px rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.20), inset 0 1px 0 rgba(255,255,255,0.18), inset 0 -1px 0 rgba(0,0,0,0.45)`,
                }) : (theme ? {
                  color: "rgba(255,255,255,0.60)",
                  borderColor: "rgba(255,255,255,0.18)",
                  background: "linear-gradient(180deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.04) 50%, rgba(0,0,0,0.20) 100%)",
                  boxShadow: "inset 0 1px 0 rgba(255,255,255,0.10), inset 0 -1px 0 rgba(0,0,0,0.35)",
                } : {
                  color: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.55)`,
                  borderColor: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.22)`,
                  background: `linear-gradient(180deg, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.10) 0%, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.03) 50%, rgba(0,0,0,0.25) 100%)`,
                  boxShadow: `inset 0 1px 0 rgba(220,228,240,0.08), inset 0 -1px 0 rgba(0,0,0,0.40)`,
                });
                const dropItemStyle = (active: boolean) => active ? {
                  background: theme ? "rgba(255,255,255,0.12)" : `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.12)`,
                  color: theme ? "#fff" : brandHex,
                } : {
                  color: theme ? "rgba(255,255,255,0.65)" : `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.65)`,
                };
                return (
                  <div ref={statsDropRef} className="relative flex items-stretch">
                    {/* Stats button — opens dropdown */}
                    <button
                      onClick={() => setStatsDropOpen(o => !o)}
                      className="hud-mono px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-md text-[8px] sm:text-[9px] uppercase tracking-[0.14em] transition-all duration-150 border flex items-center gap-1"
                      style={activeStyle}
                    >
                      <span className="sm:hidden">Stats</span>
                      <span className="hidden sm:inline">Stats</span>
                      <svg width="8" height="5" viewBox="0 0 8 5" fill="currentColor" className={`transition-transform duration-150 ${statsDropOpen ? "rotate-180" : ""}`}>
                        <path d="M0.5 0.5L4 4L7.5 0.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
                      </svg>
                    </button>
                    {/* Dropdown — JARVIS chrome with corner brackets + scan line */}
                    {statsDropOpen && (
                      <div
                        className="absolute top-full right-0 mt-1.5 z-50 hud-panel hud-panel--all-corners jarvis-shimmer"
                        style={{
                          minWidth: "150px",
                          padding: "4px",
                        }}
                      >
                        <span className="hud-panel__corner-tr" />
                        <span className="hud-panel__corner-bl" />
                        {/* Top mini dossier label */}
                        <div className="px-2 pt-1.5 pb-1 flex items-center gap-1.5 border-b border-white/[0.06]">
                          <span className="hud-mono text-[8px] uppercase tracking-[0.20em] text-[var(--brand-hex)]">◢</span>
                          <span className="hud-mono text-[8px] uppercase tracking-[0.20em] text-[var(--brand-hex)]/80">INDEX</span>
                          <span className="hud-pulse-dot ml-auto" style={{ background: "#4ade80", width: 5, height: 5 }} />
                        </div>
                        <Link
                          href="/league/teams"
                          className="hud-row flex items-center gap-2 px-2.5 py-2 mt-0.5 rounded hud-mono text-[10px] uppercase tracking-[0.16em] transition-all duration-150 hover:translate-x-0.5"
                          style={dropItemStyle(teamsActive)}
                          onClick={() => setStatsDropOpen(false)}
                        >
                          <svg width="11" height="11" viewBox="0 0 10 10" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="3" cy="4" r="1.5" stroke="currentColor" strokeWidth="1.1" fill="none"/>
                            <circle cx="7" cy="4" r="1.5" stroke="currentColor" strokeWidth="1.1" fill="none"/>
                            <path d="M0.8 9 C0.8 7.3 1.8 6.5 3 6.5 C4.2 6.5 5.2 7.3 5.2 9" stroke="currentColor" strokeWidth="1.1" fill="none" strokeLinecap="round"/>
                            <path d="M4.8 9 C4.8 7.3 5.8 6.5 7 6.5 C8.2 6.5 9.2 7.3 9.2 9" stroke="currentColor" strokeWidth="1.1" fill="none" strokeLinecap="round"/>
                          </svg>
                          Teams
                          <span className="ml-auto text-[8px] opacity-50">▸</span>
                        </Link>
                        <Link
                          href="/standings"
                          className="hud-row flex items-center gap-2 px-2.5 py-2 rounded hud-mono text-[10px] uppercase tracking-[0.16em] transition-all duration-150 hover:translate-x-0.5"
                          style={dropItemStyle(stndActive)}
                          onClick={() => setStatsDropOpen(false)}
                        >
                          <svg width="11" height="11" viewBox="0 0 10 10" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <rect x="0.5" y="6.5" width="2" height="3" rx="0.5" fill="currentColor" opacity="0.9"/>
                            <rect x="4" y="3.5" width="2" height="6" rx="0.5" fill="currentColor" opacity="0.9"/>
                            <rect x="7.5" y="0.5" width="2" height="9" rx="0.5" fill="currentColor" opacity="0.9"/>
                          </svg>
                          Standings
                          <span className="ml-auto text-[8px] opacity-50">▸</span>
                        </Link>
                        <Link
                          href="/stats"
                          className="hud-row flex items-center gap-2 px-2.5 py-2 rounded hud-mono text-[10px] uppercase tracking-[0.16em] transition-all duration-150 hover:translate-x-0.5"
                          style={dropItemStyle(statsActive)}
                          onClick={() => setStatsDropOpen(false)}
                        >
                          <svg width="11" height="11" viewBox="0 0 10 10" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="5" cy="5" r="3.5" stroke="currentColor" strokeWidth="1.2"/>
                            <circle cx="5" cy="5" r="1.2" fill="currentColor"/>
                          </svg>
                          Player Stats
                          <span className="ml-auto text-[8px] opacity-50">▸</span>
                        </Link>
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* BarnCentre — TV icon (metallic) */}
              <Link
                href="/barncentre"
                className="px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-md transition-all duration-150 border flex items-center justify-center text-[#C9A84C]/85 border-[#C9A84C]/35 hover:text-[#C9A84C] hover:border-[#C9A84C]/60"
                style={{
                  background: "linear-gradient(180deg, rgba(201,168,76,0.16) 0%, rgba(201,168,76,0.05) 50%, rgba(0,0,0,0.25) 100%)",
                  boxShadow: "inset 0 1px 0 rgba(255,228,170,0.15), inset 0 -1px 0 rgba(0,0,0,0.45)",
                }}
                title="BarnCentre"
              >
                {/* TV icon */}
                <svg width="14" height="13" viewBox="0 0 14 13" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <rect x="1" y="2" width="12" height="7.5" rx="1" stroke="currentColor" strokeWidth="1.3"/>
                  <line x1="4.5" y1="9.5" x2="3.5" y2="12" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                  <line x1="9.5" y1="9.5" x2="10.5" y2="12" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                  <line x1="3" y1="12" x2="11" y2="12" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                  <line x1="4.5" y1="2" x2="2.5" y2="0.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                  <line x1="9.5" y1="2" x2="11.5" y2="0.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                </svg>
              </Link>

              {/* Cortex button — after BarnCentre */}
              {(() => {
                const cortexActive = pathname === "/players" || pathname.startsWith("/players/");
                return (
                  <Link
                    href="/players"
                    className="px-1.5 sm:px-2 py-1.5 sm:py-2 rounded-md text-[8px] sm:text-[9px] transition-all duration-150 border flex items-center justify-center"
                    style={cortexActive ? {
                      color: "#a78bfa",
                      borderColor: "rgba(167,139,250,0.55)",
                      background: "linear-gradient(180deg, rgba(167,139,250,0.32) 0%, rgba(167,139,250,0.14) 50%, rgba(0,0,0,0.20) 100%)",
                      boxShadow: "0 0 10px rgba(167,139,250,0.22), inset 0 1px 0 rgba(220,210,255,0.20), inset 0 -1px 0 rgba(0,0,0,0.40)",
                    } : {
                      color: "rgba(167,139,250,0.65)",
                      borderColor: "rgba(167,139,250,0.25)",
                      background: "linear-gradient(180deg, rgba(167,139,250,0.10) 0%, rgba(167,139,250,0.03) 50%, rgba(0,0,0,0.25) 100%)",
                      boxShadow: "inset 0 1px 0 rgba(220,210,255,0.10), inset 0 -1px 0 rgba(0,0,0,0.40)",
                    }}
                  >
                    <MiniCortex />
                  </Link>
                );
              })()}

              {/* Logout button (metallic red) */}
              <button
                onClick={handleLogout}
                title="Logout"
                aria-label="Logout"
                className="hud-mono px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-md text-[10px] sm:text-[9px] uppercase tracking-[0.14em] transition-all duration-150 border flex items-center justify-center text-[#f87171]/85 border-[#f87171]/35 hover:text-[#f87171] hover:border-[#f87171]/60"
                style={{
                  background: "linear-gradient(180deg, rgba(248,113,113,0.16) 0%, rgba(248,113,113,0.05) 50%, rgba(0,0,0,0.25) 100%)",
                  boxShadow: "inset 0 1px 0 rgba(255,200,200,0.15), inset 0 -1px 0 rgba(0,0,0,0.45)",
                }}
              >
                <span className="sm:hidden leading-none">✕</span>
                <span className="hidden sm:inline">Logout</span>
              </button>
            </div>
            {/* HUD scan-line under the header — slow horizontal sweep, decorative only */}
            <span
              aria-hidden
              className="absolute bottom-0 left-0 right-0 h-px pointer-events-none overflow-hidden"
              style={{ background: `linear-gradient(90deg, transparent, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.55), transparent)`, opacity: 0.7 }}
            />
            <span
              aria-hidden
              className="absolute -bottom-px left-0 h-px w-24 pointer-events-none"
              style={{
                background: `linear-gradient(90deg, transparent, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.85), transparent)`,
                animation: "headerSweep 5.5s linear infinite",
              }}
            />
            <style jsx>{`
              @keyframes headerSweep {
                0%   { transform: translateX(0); }
                100% { transform: translateX(100vw); }
              }
            `}</style>
          </header>

          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={pathname}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
              className="flex-1 flex flex-col min-w-0"
            >
              {children}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
      </PiPProvider>
    </>
  );
}

// Lighten a hex color toward white for gradient endpoint
function lightenHex(hex: string): string {
  const clean = hex.replace("#", "");
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  const lr = Math.round(r + (255 - r) * 0.45);
  const lg = Math.round(g + (255 - g) * 0.45);
  const lb = Math.round(b + (255 - b) * 0.45);
  return `#${lr.toString(16).padStart(2, "0")}${lg.toString(16).padStart(2, "0")}${lb.toString(16).padStart(2, "0")}`;
}

export default function ClientLayout({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider>
      <SeasonContextProvider>
        <InnerLayout>{children}</InnerLayout>
      </SeasonContextProvider>
    </ThemeProvider>
  );
}
