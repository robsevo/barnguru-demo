"use client";

import { useState } from "react";
import Link from "next/link";

export default function LandingPage() {
  const [aboutOpen, setAboutOpen] = useState(false);

  return (
    <main className="min-h-screen flex flex-col items-center px-6 pt-3 pb-10">

      {/* Divider below goal feed */}
      <div className="w-full max-w-2xl h-px bg-gradient-to-r from-transparent via-white/[0.10] to-transparent mb-3" />

      {/* Hero */}
      <div className="text-center mb-2 max-w-2xl">
        {/* "Welcome to:" — matches header logo text exactly */}
        <p
          className="font-black italic tracking-[0.06em] uppercase text-[17px] sm:text-[20px] mb-0 bg-gradient-to-r from-white via-[#E8D090] to-[#C9A84C] bg-clip-text text-transparent"
          style={{ fontFamily: "var(--font-condensed)" }}
        >
          Welcome to:
        </p>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/hero.png"
          alt="GRTZKY"
          className="mx-auto w-full max-w-[420px] sm:max-w-[620px] lg:max-w-[760px] h-auto drop-shadow-2xl -mt-3"
        />
        {/* Acronym under logo */}
        <p className="text-[10px] sm:text-[13px] font-light uppercase tracking-[0.22em] -mt-1 mb-6 bg-gradient-to-r from-[#C9A84C] via-[#E8D090] to-[#C9A84C] bg-clip-text text-transparent">
          Bayesian&nbsp;·&nbsp;Analytics&nbsp;·&nbsp;Rating&nbsp;·&nbsp;Network
        </p>
        {/* Collapsible About */}
        <div className="mb-1">
          <button
            onClick={() => setAboutOpen((v) => !v)}
            className={`mx-auto flex items-center gap-1.5 px-4 py-1.5 rounded-md border transition-all duration-200 group ${
              aboutOpen
                ? "bg-[#C9A84C]/15 border-[#C9A84C]/40 shadow-[0_0_10px_rgba(201,168,76,0.12)]"
                : "bg-[#C9A84C]/[0.05] border-[#C9A84C]/20 hover:bg-[#C9A84C]/10 hover:border-[#C9A84C]/35"
            }`}
          >
            <span className="text-[9px] font-black uppercase tracking-[0.22em] bg-gradient-to-r from-white via-[#E8D090] to-[#C9A84C] bg-clip-text text-transparent">
              About
            </span>
            <span className={`text-[#C9A84C]/60 text-[8px] transition-all duration-200 ${aboutOpen ? "rotate-180" : ""}`}>
              ▼
            </span>
          </button>

          <div
            className={`overflow-hidden transition-all duration-300 ease-in-out ${
              aboutOpen ? "max-h-[700px] opacity-100 mt-3" : "max-h-0 opacity-0 mt-0"
            }`}
          >
            <div className="w-12 h-px bg-white/[0.12] mx-auto mb-3" />
            <p className="text-[13px] text-white/60 leading-relaxed max-w-lg mx-auto mb-3">
              If the Dev handed you a username, a password, and a secret phrase — count yourself among a very small group with access to something genuinely new. GRTZKY is the first platform of its kind: real AI-native hockey intelligence, built from the ice up, designed to find the edges the market will never see coming. Keep your credentials to yourself — access is tracked, and sharing gets your account pulled. If someone wants in, they ask the Dev.
            </p>
            <div className="w-12 h-px bg-white/[0.12] mx-auto mb-3" />
            <p className="text-[13px] text-white/60 leading-relaxed max-w-lg mx-auto mb-3">
              Every game starts with live rosters, injury reports, morning skate signals, and shift data fed into a multi-layer fatigue engine — travel miles, time zones, back-to-backs, minute spikes, return-from-injury rust. Coaching tendencies, line chemistry, and matchup history layer on top. The output runs through a Rust-powered Markov engine that simulates thousands of complete games before puck drop. The Brier score either moves, or the feature doesn&apos;t ship.
            </p>
            <div className="w-12 h-px bg-white/[0.12] mx-auto mb-3" />
            <p className="text-[13px] text-white/60 leading-relaxed max-w-lg mx-auto mb-3">
              Watch every game with direct streams. Track all 32 teams in real-time standings. Dig into league-wide stats leaders. Use score-hide if you&apos;re on stream delay — the live API posts goals before any broadcast. Or leave it off and you&apos;ll see most goals on your screen seconds before they hit your feed. Follow the full model build in the dev dashboard, from raw ingestion to live Polymarket edge detection.
            </p>
            <div className="w-12 h-px bg-white/[0.12] mx-auto mb-3" />
            <p className="text-[13px] text-white/60 leading-relaxed max-w-lg mx-auto mb-3">
              The live goal feed runs site-wide in real time. Every goal appears as a clickable alert at the top — tap it and you&apos;re at the game instantly. The NHL API registers goals the moment they happen, almost always before your stream catches up. That gap is your window. It&apos;s not a glitch — it&apos;s the fastest way to watch hockey.
            </p>
            <div className="w-12 h-px bg-white/[0.12] mx-auto mb-3" />
            <div className="max-w-lg mx-auto rounded-xl border border-[#fbbf24]/20 bg-[#fbbf24]/[0.04] px-4 py-3 mb-3 text-left">
              <p className="text-[9px] font-semibold uppercase tracking-[0.22em] text-[#fbbf24]/60 mb-1.5">Ad Notice</p>
              <p className="text-[13px] text-white/60 leading-relaxed">
                The Dev is actively working to eliminate ads from this platform entirely. For now, on desktop, some may slip through. Use <span className="font-semibold text-white/80">Brave Browser</span> or an ad blocker extension until that&apos;s resolved.
              </p>
            </div>
          </div>
        </div>
        {/* Divider below about pill */}
        <div className="w-12 h-px bg-white/[0.10] mx-auto mt-3" />
      </div>

      {/* Divider above cards */}
      <div className="w-full max-w-2xl h-px bg-gradient-to-r from-transparent via-white/[0.10] to-transparent mb-4" />

      {/* Choice cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-2xl">

        {/* GameCentre card */}
        <Link href="/gamecentre" className="group">
          <div className="relative h-44 rounded-2xl border border-[#C9A84C]/[0.18] bg-gradient-to-b from-white/[0.015] via-white/[0.005] to-transparent hover:border-white/[0.36] transition-all duration-200 p-6 flex flex-col justify-between overflow-hidden shadow-[0_8px_32px_rgba(0,0,0,0.80),0_2px_6px_rgba(0,0,0,0.55),inset_0_1px_0_rgba(255,255,255,0.06),inset_0_-1px_0_rgba(0,0,0,0.35),inset_1px_0_0_rgba(255,255,255,0.01),inset_-1px_0_0_rgba(255,255,255,0.01)]">
            <div className="absolute -top-8 -right-8 w-40 h-40 rounded-full bg-white/[0.04] blur-2xl pointer-events-none group-hover:bg-white/[0.08] transition-all duration-300" />
            <div>
              <span className="text-[9px] font-black uppercase tracking-[0.3em] text-white/30 block mb-3">Live &amp; Scheduled</span>
              <h2 className="text-[28px] font-black tracking-[0.08em] uppercase text-white leading-none">
                Live<br />Games
              </h2>
            </div>
            <div className="flex items-end justify-between">
              <p className="text-[11px] text-white/35 leading-relaxed max-w-[160px]">
                Live scores, play-by-play, skater stats, and shot maps for every game.
              </p>
              <span className="text-white/30 group-hover:text-white/70 text-xl transition-all duration-200 group-hover:translate-x-1">→</span>
            </div>
          </div>
        </Link>

        {/* Standings card */}
        <Link href="/standings" className="group">
          <div className="relative h-44 rounded-2xl border border-[#fbbf24]/15 bg-[#fbbf24]/[0.02] hover:bg-[#fbbf24]/[0.05] hover:border-[#fbbf24]/30 transition-all duration-200 p-6 flex flex-col justify-between overflow-hidden shadow-[0_4px_24px_rgba(0,0,0,0.6),inset_0_1px_0_rgba(220,225,230,0.08)]">
            <div className="absolute -top-8 -right-8 w-40 h-40 rounded-full bg-[#fbbf24]/8 blur-2xl pointer-events-none group-hover:bg-[#fbbf24]/15 transition-all duration-300" />
            <div>
              <span className="text-[9px] font-black uppercase tracking-[0.3em] text-[#fbbf24]/40 block mb-3">Playoff Race</span>
              <h2 className="text-[28px] font-black tracking-[0.08em] uppercase text-white/80 leading-none">
                Standings
              </h2>
            </div>
            <div className="flex items-end justify-between">
              <p className="text-[11px] text-white/30 leading-relaxed max-w-[160px]">
                Division standings, playoff picture, and in-the-hunt tracker.
              </p>
              <span className="text-[#fbbf24]/30 group-hover:text-[#fbbf24]/70 text-xl transition-all duration-200 group-hover:translate-x-1">→</span>
            </div>
          </div>
        </Link>

        {/* Stats Leaders card */}
        <Link href="/stats" className="group">
          <div className="relative h-44 rounded-2xl border border-[#C9A84C]/[0.18] bg-gradient-to-b from-white/[0.015] via-white/[0.005] to-transparent hover:border-white/[0.36] transition-all duration-200 p-6 flex flex-col justify-between overflow-hidden shadow-[0_8px_32px_rgba(0,0,0,0.80),0_2px_6px_rgba(0,0,0,0.55),inset_0_1px_0_rgba(255,255,255,0.06),inset_0_-1px_0_rgba(0,0,0,0.35),inset_1px_0_0_rgba(255,255,255,0.01),inset_-1px_0_0_rgba(255,255,255,0.01)]">
            <div className="absolute -top-8 -left-8 w-40 h-40 rounded-full bg-white/[0.03] blur-2xl pointer-events-none group-hover:bg-white/[0.07] transition-all duration-300" />
            <div>
              <span className="text-[9px] font-black uppercase tracking-[0.3em] text-white/25 block mb-3">League Leaders</span>
              <h2 className="text-[28px] font-black tracking-[0.08em] uppercase text-white/80 leading-none">
                Stats<br />Leaders
              </h2>
            </div>
            <div className="flex items-end justify-between">
              <p className="text-[11px] text-white/30 leading-relaxed max-w-[160px]">
                Points, goals, assists, +/-, TOI, and goalie leaders updated live.
              </p>
              <span className="text-white/25 group-hover:text-white/65 text-xl transition-all duration-200 group-hover:translate-x-1">→</span>
            </div>
          </div>
        </Link>

        {/* Neural Scout card */}
        <Link href="/players" className="group">
          <div className="relative h-44 rounded-2xl border border-[#a78bfa]/[0.18] bg-gradient-to-b from-[#a78bfa]/[0.03] via-transparent to-transparent hover:border-[#a78bfa]/[0.35] transition-all duration-200 p-6 flex flex-col justify-between overflow-hidden shadow-[0_8px_32px_rgba(0,0,0,0.80),inset_0_1px_0_rgba(167,139,250,0.08)]">
            <div className="absolute -top-8 -right-8 w-40 h-40 rounded-full bg-[#a78bfa]/[0.06] blur-2xl pointer-events-none group-hover:bg-[#a78bfa]/[0.12] transition-all duration-300" />
            <div>
              <span className="text-[9px] font-black uppercase tracking-[0.3em] text-[#a78bfa]/40 block mb-3">Neural Networks · Deep Analysis</span>
              <h2 className="text-[28px] font-black tracking-[0.08em] uppercase text-white/80 leading-none">
                Cortex
              </h2>
            </div>
            <div className="flex items-end justify-between">
              <p className="text-[11px] text-white/30 leading-relaxed max-w-[160px]">
                Model-driven player profiles, form tracking, and deep analytics.
              </p>
              <span className="text-[#a78bfa]/40 group-hover:text-[#a78bfa]/80 text-xl transition-all duration-200 group-hover:translate-x-1">→</span>
            </div>
          </div>
        </Link>
      </div>

      {/* Phase progress strip */}
      <div className="mt-6 w-full max-w-2xl">
        <div className="w-full h-px bg-gradient-to-r from-transparent via-white/[0.10] to-transparent mb-3" />
        <div className="flex flex-col items-center gap-2">
          <div className="flex items-center gap-2">
            {[
              { status: "complete",    n: 2  },
              { status: "in_progress", n: 1  },
              { status: "not_started", n: 12 },
              { status: "complete",    n: 1  },
            ].map(({ status, n }) =>
              Array.from({ length: n }).map((_, i) => (
                <span
                  key={`${status}-${i}`}
                  className={`h-1 rounded-full transition-all ${
                    status === "complete"    ? "w-5 bg-[#4ade80] shadow-[0_0_6px_rgba(74,222,128,0.7)]" :
                    status === "in_progress" ? "w-5 bg-[#fbbf24] shadow-[0_0_6px_rgba(251,191,36,0.7)]" :
                    "w-3 bg-white/[0.08]"
                  }`}
                />
              ))
            )}
          </div>
          <span className="text-[9px] font-semibold uppercase tracking-[0.2em] text-white/20 text-center">
            3 of 16 phases complete<br />CV tracking active
          </span>
        </div>
      </div>

      <p className="mt-3 max-w-xl text-center text-[9px] text-white/12 leading-relaxed">
        GRTZKY is a private research and analytics tool for informational purposes only. Nothing on this platform constitutes financial, betting, or investment advice. All simulation outputs and edge signals are experimental and unverified. Use your own judgement. The Dev is not responsible for any decisions made based on information presented here.
      </p>
    </main>
  );
}
