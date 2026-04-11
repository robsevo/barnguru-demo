"use client";

import { useState, useEffect } from "react";
import { PiPProvider } from "@/utils/pipContext";
import FloatingPlayer  from "./FloatingPlayer";

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

function MiniDNA() {
  return (
    <svg width="11" height="14" viewBox="0 0 26 32" fill="none" aria-hidden="true" className="shrink-0">
      <path d="M21 0 C5 3 3 7 3 10 C3 13 5 17 21 20 C21 20 5 23 3 26 C3 29 21 31 21 32"
        stroke="#a78bfa" strokeWidth="3.5" strokeLinecap="round" fill="none"/>
      <path d="M5 0 C21 3 23 7 23 10 C23 13 21 17 5 20 C5 20 21 23 23 26 C23 29 5 31 5 32"
        stroke="#7c3aed" strokeWidth="3.5" strokeLinecap="round" fill="none"/>
      <line x1="3" y1="10" x2="23" y2="10" stroke="rgba(167,139,250,0.6)" strokeWidth="2" strokeLinecap="round"/>
      <line x1="3" y1="20" x2="23" y2="20" stroke="rgba(167,139,250,0.6)" strokeWidth="2" strokeLinecap="round"/>
      <circle cx="21" cy="0" r="3" fill="#a78bfa" opacity="0.9"/>
      <circle cx="5" cy="0" r="3" fill="#7c3aed" opacity="0.9"/>
      <circle cx="3" cy="10" r="3" fill="#a78bfa" opacity="0.95"/>
      <circle cx="23" cy="10" r="3" fill="#7c3aed" opacity="0.95"/>
      <circle cx="21" cy="20" r="3" fill="#a78bfa" opacity="0.95"/>
      <circle cx="5" cy="20" r="3" fill="#7c3aed" opacity="0.95"/>
      <circle cx="5" cy="32" r="3" fill="#a78bfa" opacity="0.9"/>
      <circle cx="21" cy="32" r="3" fill="#7c3aed" opacity="0.9"/>
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

export default function ClientLayout({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const [username, setUsername] = useState<string | null>(null);
  const pathname = usePathname();

  useEffect(() => {
    fetch("/api/auth/me")
      .then((r) => r.json())
      .then((d) => setUsername(d.username ?? null))
      .catch(() => {});
  }, []);

  return (
    <PiPProvider>
    <FloatingPlayer />
    <div className="relative flex min-h-screen overflow-x-hidden">
      {/* Backdrop — no blur, just a dim overlay */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm"
          onClick={() => setOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed top-0 left-0 z-50 h-full w-64 flex flex-col
          bg-gradient-to-b from-[#0d0f13]/99 to-[#090a0c]/99 backdrop-blur-2xl border-r border-white/[0.08]
          shadow-[4px_0_40px_rgba(0,0,0,0.85),inset_-1px_0_0_rgba(218,226,236,0.07)]
          transition-transform duration-300 ease-in-out
          ${open ? "translate-x-0" : "-translate-x-full"}`}
      >
        {/* Sidebar header */}
        <div className="flex items-center justify-between px-4 py-3.5 border-b border-white/[0.07] shrink-0">
          <Link href="/" onClick={() => setOpen(false)} className="flex items-center gap-2 pr-1">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/logo-circle.png" alt="GRTZKY" className="h-8 w-auto" />
            <span className="text-white/20 font-thin text-sm">|</span>
            <span className="font-black italic tracking-[0.08em] text-base bg-gradient-to-r from-white via-[#E8D090] to-[#C9A84C] bg-clip-text text-transparent" style={{ fontFamily: "var(--font-condensed)" }}>
              GRTZKY
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
          {/* Dev Dashboard link */}
          <p className="px-3 pt-1 pb-1.5 text-[9px] font-semibold uppercase tracking-[0.22em] text-white/15">
            Navigation
          </p>
          <Link
            href="/"
            onClick={() => setOpen(false)}
            className="block"
          >
            <div className={`flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all duration-150
              ${pathname === "/" ? "bg-white/[0.08]" : "hover:bg-white/[0.04]"}`}>
              <span className="h-1.5 w-1.5 rounded-full shrink-0 bg-[#4ade80] shadow-[0_0_6px_rgba(74,222,128,0.9)]" />
              <span className="text-[9px] font-semibold font-mono text-white/15 w-5 shrink-0">—</span>
              <span className="text-[11px] font-medium leading-tight text-white/70">Home</span>
            </div>
          </Link>
          <Link
            href="/gamecentre"
            onClick={() => setOpen(false)}
            className="block"
          >
            <div className={`flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all duration-150
              ${pathname === "/gamecentre" || pathname.startsWith("/game/") ? "bg-white/[0.08]" : "hover:bg-white/[0.04]"}`}>
              <span className="h-1.5 w-1.5 rounded-full shrink-0 bg-white/50 shadow-[0_0_6px_rgba(220,225,230,0.6)]" />
              <span className="text-[9px] font-semibold font-mono text-white/15 w-5 shrink-0">—</span>
              <span className="text-[11px] font-medium leading-tight text-white/70">Live Games</span>
            </div>
          </Link>
          <Link
            href="/standings"
            onClick={() => setOpen(false)}
            className="block"
          >
            <div className={`flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all duration-150
              ${pathname === "/standings" ? "bg-white/[0.08]" : "hover:bg-white/[0.04]"}`}>
              <span className="h-1.5 w-1.5 rounded-full shrink-0 bg-[#fbbf24]/70 shadow-[0_0_5px_rgba(251,191,36,0.6)]" />
              <span className="text-[9px] font-semibold font-mono text-white/15 w-5 shrink-0">—</span>
              <span className="text-[11px] font-medium leading-tight text-white/70">Standings</span>
            </div>
          </Link>
          <Link
            href="/stats"
            onClick={() => setOpen(false)}
            className="block"
          >
            <div className={`flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all duration-150
              ${pathname === "/stats" ? "bg-white/[0.08]" : "hover:bg-white/[0.04]"}`}>
              <span className="h-1.5 w-1.5 rounded-full shrink-0 bg-white/35 shadow-[0_0_5px_rgba(210,215,220,0.5)]" />
              <span className="text-[9px] font-semibold font-mono text-white/15 w-5 shrink-0">—</span>
              <span className="text-[11px] font-medium leading-tight text-white/70">Stats Leaders</span>
            </div>
          </Link>
          <Link
            href="/players"
            onClick={() => setOpen(false)}
            className="block"
          >
            <div className={`flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all duration-150
              ${pathname === "/players" || pathname.startsWith("/players/") ? "bg-white/[0.08]" : "hover:bg-white/[0.04]"}`}>
              <span className="h-1.5 w-1.5 rounded-full shrink-0 bg-[#a78bfa] shadow-[0_0_6px_rgba(167,139,250,0.8)]" />
              <span className="text-[9px] font-semibold font-mono text-white/15 w-5 shrink-0">—</span>
              <span className="text-[11px] font-medium leading-tight text-white/70">Cortex</span>
            </div>
          </Link>

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

              {/* Phase list */}
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

      {/* Main content — shifts right when sidebar is open */}
      <div
        className={`flex-1 flex flex-col min-w-0 transition-all duration-300 ease-in-out
          ${open ? "ml-64" : "ml-0"}`}
      >
        {/* Top bar */}
        <header className="sticky top-0 z-30 flex items-center gap-3 px-2 sm:px-3 h-12
          bg-gradient-to-r from-[#070809]/97 via-[#0a0b0f]/95 to-[#070809]/97 backdrop-blur-2xl
          border-b border-[#C9A84C]/[0.30]
          shadow-[0_1px_0_rgba(201,168,76,0.18),0_4px_32px_rgba(0,0,0,0.70),0_0_40px_rgba(201,168,76,0.04),inset_0_1px_0_rgba(201,168,76,0.10)]
          select-none">

          {/* Hamburger */}
          <button
            onClick={() => setOpen((o) => !o)}
            className="flex flex-col justify-center gap-[5px] w-7 h-7 rounded-md
              hover:bg-[#C9A84C]/[0.08] transition-colors duration-200 shrink-0 items-center"
            aria-label="Toggle navigation"
          >
            <span className={`block h-[2px] w-[16px] bg-[#C9A84C]/60 rounded-full transition-all duration-200 origin-center ${open ? "rotate-45 translate-y-[7px]" : ""}`} />
            <span className={`block h-[2px] w-[16px] bg-[#C9A84C]/60 rounded-full transition-all duration-200 ${open ? "opacity-0 scale-x-0" : ""}`} />
            <span className={`block h-[2px] w-[16px] bg-[#C9A84C]/60 rounded-full transition-all duration-200 origin-center ${open ? "-rotate-45 -translate-y-[7px]" : ""}`} />
          </button>

          {/* Logo + name — always visible, tight to hamburger */}
          <Link href="/" className="inline-flex items-center shrink-0 pr-1">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/logo-circle.png" alt="GRTZKY" className="h-9 sm:h-10 w-auto block" />
            <span className="h-5 w-px bg-[#C9A84C]/30 shrink-0 mx-2" />
            <span
              className="font-black italic tracking-[0.06em] text-[11px] sm:text-[17px] bg-gradient-to-r from-white via-[#E8D090] to-[#C9A84C] bg-clip-text text-transparent leading-none pr-1.5"
              style={{ fontFamily: "var(--font-condensed)" }}
            >
              GRTZKY
            </span>
          </Link>

          <span className="hidden lg:block h-4 w-px bg-[#C9A84C]/20 mx-1" />
          <span className="text-[#C9A84C]/30 text-[9px] font-medium tracking-[0.16em] uppercase hidden lg:block">
            Bayesian Analytics and Rating Network
          </span>

          {/* Quick-nav buttons */}
          <div className="ml-auto flex items-center gap-0.5 sm:gap-1 shrink-0">
            {(
              [
                { href: "/gamecentre", labelSm: "Live",  label: "Games",     active: pathname === "/gamecentre" || pathname.startsWith("/game/") },
                { href: "/standings",  labelSm: "Stnd",  label: "Standings", active: pathname === "/standings" },
                { href: "/stats",      labelSm: "Stats", label: "Stats",     active: pathname === "/stats" },
                { href: "/players",    labelSm: "Crtx",  label: "Cortex",    active: pathname === "/players" || pathname.startsWith("/players/") },
              ] as const
            ).map(({ href, labelSm, label, active }) => {
              const isCortex = label === "Cortex";
              return (
                <Link
                  key={href}
                  href={href}
                  className={`px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-md text-[8px] sm:text-[9px] font-black uppercase tracking-normal transition-all duration-150 border flex items-center justify-center ${
                    isCortex
                      ? active
                        ? "bg-[#a78bfa]/15 text-[#a78bfa] border-[#a78bfa]/50 shadow-[0_0_8px_rgba(167,139,250,0.20)]"
                        : "text-[#a78bfa]/50 border-[#a78bfa]/20 bg-[#a78bfa]/[0.04] hover:text-[#a78bfa]/80 hover:border-[#a78bfa]/40 hover:bg-[#a78bfa]/10"
                      : active
                        ? "bg-[#C9A84C]/15 text-[#C9A84C] border-[#C9A84C]/40 shadow-[0_0_8px_rgba(201,168,76,0.15)]"
                        : "text-[#C9A84C]/45 border-[#C9A84C]/15 bg-[#C9A84C]/[0.04] hover:text-[#C9A84C]/80 hover:border-[#C9A84C]/35 hover:bg-[#C9A84C]/10"
                  }`}
                >
                  <span className="sm:hidden">{isCortex ? <MiniDNA /> : labelSm}</span>
                  <span className="hidden sm:inline">{label}</span>
                </Link>
              );
            })}
            <button
              onClick={async () => {
                await fetch("/api/auth/logout", { method: "POST" });
                window.location.href = "/login";
              }}
              className="px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-md text-[8px] sm:text-[9px] font-black uppercase tracking-normal transition-all duration-150 border text-white/25 border-white/[0.08] hover:text-[#f87171]/70 hover:border-[#f87171]/25 hover:bg-[#f87171]/[0.05]"
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
  );
}
