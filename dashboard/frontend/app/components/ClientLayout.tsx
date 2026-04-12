"use client";

import { useState, useEffect } from "react";
import { PiPProvider } from "@/utils/pipContext";
import { ThemeProvider, useTheme } from "@/utils/themeContext";
import FloatingPlayer   from "./FloatingPlayer";
import ThemeInjector    from "./ThemeInjector";
import TeamThemeLoader  from "./TeamThemeLoader";

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
import { usePathname } from "next/navigation";

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
  { num: 3,  name: "Fatigue Engine",             status: "not_started" },
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
  { href: "/",              label: "Home",          shortLabel: "Home",   dot: "static", dotClass: "bg-white/20",                                                              active: (p: string) => p === "/" },
  { href: "/gamecentre",    label: "Live",          shortLabel: "Live",   dot: "live",   dotClass: "",                                                                         active: (p: string) => p === "/gamecentre" || p.startsWith("/game/") },
  { href: "/league/teams",  label: "Teams",         shortLabel: "Teams",  dot: "static", dotClass: "bg-white/20",                                                              active: (p: string) => p.startsWith("/league") || p.startsWith("/teams/") },
  { href: "/standings",     label: "Standings",     shortLabel: "Stnd",   dot: "static", dotClass: "bg-[#fbbf24]/70 shadow-[0_0_5px_rgba(251,191,36,0.6)]",                   active: (p: string) => p === "/standings" },
  { href: "/stats",         label: "Stats Leaders", shortLabel: "Stats",  dot: "static", dotClass: "bg-white/35 shadow-[0_0_5px_rgba(210,215,220,0.5)]",                      active: (p: string) => p === "/stats" },
  { href: "/players",       label: "Cortex",        shortLabel: "Crtx",   dot: "static", dotClass: "bg-[#a78bfa] shadow-[0_0_6px_rgba(167,139,250,0.8)]",                     active: (p: string) => p === "/players" || p.startsWith("/players/") },
];

function InnerLayout({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const [username, setUsername] = useState<string | null>(null);
  const [hasLive, setHasLive] = useState(false);
  const pathname = usePathname();
  const { theme, clearTheme, cortexPinned, setCortexPinned } = useTheme();
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
                  : "bg-white/20"
                : dotClass;
              return (
                <Link key={href} href={href} onClick={() => setOpen(false)} className="block">
                  <div className={`flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all duration-150 ${isActive ? "bg-white/[0.08]" : "hover:bg-white/[0.04]"}`}>
                    <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${liveDot}`} />
                    <span className="text-[9px] font-semibold font-mono text-white/15 w-5 shrink-0">—</span>
                    <span className={`text-[11px] font-medium leading-tight ${isActive ? "text-white/90" : "text-white/60"}`}>
                      {label}
                    </span>
                  </div>
                </Link>
              );
            })}

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
          <div className="px-4 py-3 border-t border-white/[0.10] shrink-0 flex items-center justify-between">
            <p className="text-[8px] font-semibold uppercase tracking-[0.2em] text-white/10">
              GRTZKY
            </p>
            <button
              onClick={async () => {
                await fetch("/api/auth/logout", { method: "POST" });
                window.location.href = "/login";
              }}
              className="text-[8px] font-semibold uppercase tracking-[0.2em] text-white/20 hover:text-[#f87171]/60 transition-colors duration-150"
            >
              Sign out
            </button>
          </div>
        </aside>

        {/* Main content */}
        <div
          className={`flex-1 flex flex-col min-w-0 overflow-x-hidden transition-all duration-300 ease-in-out
            ${open ? "ml-64" : "ml-0"}`}
        >
          {/* Top bar */}
          <header
            className="sticky top-0 z-30 flex items-center gap-3 px-2 sm:px-3 h-12 backdrop-blur-2xl select-none"
            style={{
              background: `linear-gradient(to right, rgba(var(--header-r), var(--header-g), var(--header-b), 0.98), rgba(var(--header-r), var(--header-g), var(--header-b), 0.95), rgba(var(--header-r), var(--header-g), var(--header-b), 0.98))`,
              borderBottom: `1px solid rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.30)`,
              boxShadow: `0 1px 0 rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.18), 0 4px 32px rgba(0,0,0,0.70), 0 0 40px rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.04), inset 0 1px 0 rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.10)`,
            }}
          >
            {/* Hamburger */}
            <button
              onClick={() => setOpen((o) => !o)}
              className="flex flex-col justify-center gap-[5px] w-7 h-7 rounded-md transition-colors duration-200 shrink-0 items-center"
              style={{ background: open ? (theme ? "rgba(255,255,255,0.08)" : `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.08)`) : undefined }}
              aria-label="Toggle navigation"
            >
              {(["rotate(45deg) translateY(7px)", undefined, "rotate(-45deg) translateY(-7px)"] as const).map((transform, i) => (
                <span key={i} className="block h-[2px] w-[16px] rounded-full transition-all duration-200 origin-center"
                  style={{
                    background: theme ? "rgba(255,255,255,0.65)" : `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.60)`,
                    transform: open ? transform : undefined,
                    opacity: (i === 1 && open) ? 0 : 1,
                  }} />
              ))}
            </button>

            {/* Logo + name */}
            <Link href="/" className="inline-flex items-center shrink-0 pr-1">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              {theme ? (
                <img src={theme.logoUrl} alt={theme.abbrev} className="h-7 sm:h-8 w-auto object-contain block" />
              ) : isCortexMode ? (
                <svg width="36" height="36" viewBox="0 0 48 48" fill="none" aria-hidden="true" className="shrink-0 lg:!w-[42px] lg:!h-[42px]">
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
                <img src="/logo-circle.png" alt="GRTZKY" className="h-7 sm:h-8 w-auto block" />
              )}
              <span
                className="h-5 w-px shrink-0 mx-2"
                style={{ background: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.30)` }}
              />
              <span
                className="font-black italic tracking-[0.06em] text-[11px] sm:text-[17px] lg:text-[20px] bg-clip-text text-transparent leading-none pr-1.5"
                style={{
                  fontFamily: "var(--font-condensed)",
                  backgroundImage: isCortexMode
                    ? `linear-gradient(to right, #fff, #c4b5fd, #a78bfa)`
                    : `linear-gradient(to right, #fff, ${brandHex2}, ${brandHex})`,
                }}
              >
                {theme ? theme.abbrev : isCortexMode ? "CORTEX" : "GRTZKY"}
              </span>
            </Link>

            <span className="hidden lg:block h-4 w-px mx-1" style={{ background: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.20)` }} />
            <span
              className="text-[9px] font-medium tracking-[0.16em] uppercase hidden lg:block"
              style={{ color: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.30)` }}
            >
              {theme ? "Team Theme Active" : isCortexMode ? "Player Intelligence · Neural Models" : "Bayesian Analytics and Rating Network"}
            </span>

            {/* Quick-nav buttons */}
            <div className="ml-auto flex items-center gap-0.5 sm:gap-1 shrink-0">
              {(
                [
                  { href: "/gamecentre",   labelSm: "Live",  label: "Live",      active: pathname === "/gamecentre" || pathname.startsWith("/game/") },
                  { href: "/league/teams", labelSm: "Teams", label: "Teams",     active: pathname.startsWith("/league") || pathname.startsWith("/teams/") },
                  { href: "/standings",    labelSm: "Stnd",  label: "Standings", active: pathname === "/standings" },
                  { href: "/stats",        labelSm: "Stats", label: "Stats",     active: pathname === "/stats" },
                  { href: "/players",      labelSm: "Crtx",  label: "Cortex",    active: pathname === "/players" || pathname.startsWith("/players/") },
                ] as const
              ).map(({ href, labelSm, label, active }) => {
                const isCortex = label === "Cortex";
                if (isCortex) {
                  return (
                    <Link
                      key={href}
                      href={href}
                      className={`px-1.5 sm:px-2 py-1.5 sm:py-2 rounded-md text-[8px] sm:text-[9px] font-black uppercase tracking-normal transition-all duration-150 border flex items-center justify-center ${
                        active
                          ? "bg-[#a78bfa]/15 text-[#a78bfa] border-[#a78bfa]/50 shadow-[0_0_8px_rgba(167,139,250,0.20)]"
                          : "text-[#a78bfa]/50 border-[#a78bfa]/20 bg-[#a78bfa]/[0.04] hover:text-[#a78bfa]/80 hover:border-[#a78bfa]/40 hover:bg-[#a78bfa]/10"
                      }`}
                    >
                      <MiniCortex />
                    </Link>
                  );
                }
                return (
                  <Link
                    key={href}
                    href={href}
                    className="px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-md text-[8px] sm:text-[9px] font-black uppercase tracking-normal transition-all duration-150 border flex items-center justify-center"
                    style={active ? (theme ? {
                      background: "rgba(255,255,255,0.18)",
                      color: "#ffffff",
                      borderColor: "rgba(255,255,255,0.45)",
                      boxShadow: "0 0 8px rgba(255,255,255,0.12)",
                    } : {
                      background: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.15)`,
                      color: brandHex,
                      borderColor: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.40)`,
                      boxShadow: `0 0 8px rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.15)`,
                    }) : (theme ? {
                      color: "rgba(255,255,255,0.55)",
                      borderColor: "rgba(255,255,255,0.15)",
                      background: "rgba(255,255,255,0.05)",
                    } : {
                      color: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.45)`,
                      borderColor: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.15)`,
                      background: `rgba(var(--brand-r), var(--brand-g), var(--brand-b), 0.04)`,
                    })}
                  >
                    <span className="sm:hidden">{labelSm}</span>
                    <span className="hidden sm:inline">{label}</span>
                  </Link>
                );
              })}
              <button
                onClick={async () => {
                  await fetch("/api/auth/logout", { method: "POST" });
                  window.location.href = "/login";
                }}
                className="px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-md text-[8px] sm:text-[9px] font-black uppercase tracking-normal transition-all duration-150 border text-[#f87171]/70 border-[#f87171]/25 bg-[#f87171]/[0.05] hover:text-[#f87171] hover:border-[#f87171]/50 hover:bg-[#f87171]/[0.10]"
              >
                <span className="sm:hidden">✕</span>
                <span className="hidden sm:inline">Logout</span>
              </button>
            </div>
          </header>

          {children}
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
      <InnerLayout>{children}</InnerLayout>
    </ThemeProvider>
  );
}
