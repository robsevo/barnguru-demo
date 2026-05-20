"use client";

import { useState } from "react";
import Link from "next/link";
import { HudGrid } from "@/components/hud";

export default function LandingPage() {
  const [aboutOpen, setAboutOpen] = useState(false);

  return (
    <main className="relative min-h-screen flex flex-col items-center px-6 pt-3 pb-10">
      <HudGrid />

      {/* HUD dossier strip */}
      <div className="relative z-10 w-full max-w-2xl mb-3 flex items-center gap-2">
        <span className="hud-mono text-[10px] uppercase tracking-[0.20em] text-[var(--brand-hex)]" aria-hidden>◢</span>
        <span className="hud-mono text-[10px] uppercase tracking-[0.20em] text-[var(--brand-hex)]">
          GRTZKY · HOME
        </span>
        <span className="hud-mono text-[9px] uppercase tracking-[0.16em] text-[var(--text-secondary)]">· welcome</span>
        <span className="ml-auto flex items-center gap-1">
          <span className="hud-pulse-dot" style={{ background: "#4ade80" }} />
          <span className="hud-mono jarvis-flicker text-[9px] uppercase tracking-[0.18em] text-[#4ade80]">ONLINE</span>
        </span>
      </div>

      {/* Divider below goal feed */}
      <div className="w-full max-w-2xl h-px bg-gradient-to-r from-transparent via-white/[0.10] to-transparent mb-3 relative z-10" />

      {/* Hero — animated HUD chrome surrounding the logo */}
      <div className="text-center mb-2 max-w-2xl">
        {/* Inner wrapper anchors the rings to the logo only — the About
            collapsible below sits outside so expanding it doesn't shove
            the animation downward. */}
        <div className="relative">
        {/* Iron Man rings backdrop */}
        <svg
          viewBox="0 0 600 600"
          aria-hidden
          className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none w-[120%] max-w-none h-[120%] opacity-60"
          style={{ zIndex: 0 }}
        >
          <defs>
            <linearGradient id="heroArc" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%"  stopColor="#C9A84C" stopOpacity="0" />
              <stop offset="50%" stopColor="#C9A84C" stopOpacity="0.85" />
              <stop offset="100%" stopColor="#C9A84C" stopOpacity="0" />
            </linearGradient>
            <linearGradient id="heroArc2" x1="1" y1="0" x2="0" y2="1">
              <stop offset="0%"  stopColor="#E8D090" stopOpacity="0" />
              <stop offset="60%" stopColor="#E8D090" stopOpacity="0.55" />
              <stop offset="100%" stopColor="#E8D090" stopOpacity="0" />
            </linearGradient>
          </defs>
          {/* Outer slow ring */}
          <g style={{ transformOrigin: "300px 300px", animation: "heroRotateSlow 40s linear infinite" }}>
            <circle cx={300} cy={300} r={280} fill="none" stroke="#C9A84C" strokeOpacity={0.12} strokeDasharray="2 12" />
            {[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330].map((deg, i) => {
              const rad = (deg * Math.PI) / 180;
              return (
                <line
                  key={i}
                  x1={300 + Math.cos(rad) * 268}
                  y1={300 + Math.sin(rad) * 268}
                  x2={300 + Math.cos(rad) * 292}
                  y2={300 + Math.sin(rad) * 292}
                  stroke="#C9A84C" strokeOpacity={0.5} strokeWidth={1.2}
                />
              );
            })}
          </g>
          {/* Middle counter-rotating arc */}
          <g style={{ transformOrigin: "300px 300px", animation: "heroRotateRev 22s linear infinite" }}>
            <circle cx={300} cy={300} r={235} fill="none" stroke="url(#heroArc)" strokeWidth={1.4} strokeDasharray="100 60 40 60" strokeLinecap="round" />
          </g>
          {/* Inner fast arc */}
          <g style={{ transformOrigin: "300px 300px", animation: "heroRotateFast 12s linear infinite" }}>
            <circle cx={300} cy={300} r={195} fill="none" stroke="url(#heroArc2)" strokeWidth={1.2} strokeDasharray="30 120" strokeLinecap="round" />
          </g>
          {/* Pulse rings */}
          <circle cx={300} cy={300} r={140} fill="none" stroke="#C9A84C" strokeWidth={1}
            style={{ animation: "heroPulse 5.4s ease-out infinite" }} />
          <circle cx={300} cy={300} r={140} fill="none" stroke="#C9A84C" strokeWidth={1}
            style={{ animation: "heroPulse 5.4s ease-out infinite 2.7s" }} />
        </svg>

        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/hero.png"
          alt="GRTZKY"
          className="relative z-10 mx-auto w-full max-w-[420px] sm:max-w-[620px] lg:max-w-[760px] h-auto drop-shadow-2xl -mt-3"
        />

        {/* Scan-line sweeping the logo */}
        <div
          aria-hidden
          className="absolute left-0 right-0 top-0 h-px pointer-events-none"
          style={{
            background: "linear-gradient(90deg, transparent, rgba(232,208,144,0.9), transparent)",
            boxShadow: "0 0 12px rgba(232,208,144,0.6)",
            animation: "heroScan 7s linear infinite",
          }}
        />

        {/* Acronym under logo */}
        <p className="hud-mono text-[10px] sm:text-[13px] uppercase tracking-[0.28em] -mt-1 mb-6 bg-gradient-to-r from-[#C9A84C] via-[#E8D090] to-[#C9A84C] bg-clip-text text-transparent relative z-10">
          ◢ Bayesian&nbsp;·&nbsp;Analytics&nbsp;·&nbsp;Rating&nbsp;·&nbsp;Network ◣
        </p>
        </div>

        <style jsx>{`
          @keyframes heroRotateSlow { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
          @keyframes heroRotateRev  { from { transform: rotate(360deg); } to { transform: rotate(0deg); } }
          @keyframes heroRotateFast { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
          @keyframes heroPulse {
            0%   { r: 140; stroke-opacity: 0.55; }
            100% { r: 280; stroke-opacity: 0; }
          }
          @keyframes heroScan {
            0%   { transform: translateY(0);     opacity: 0; }
            10%  { opacity: 1; }
            90%  { opacity: 1; }
            100% { transform: translateY(380px); opacity: 0; }
          }
          @media (prefers-reduced-motion: reduce) {
            svg g, circle, div { animation: none !important; }
          }
        `}</style>

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
        <Link href="/gamecentre" className="group jarvis-lift jarvis-boot" style={{ animationDelay: "60ms" }}>
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

        {/* Playoff Tree card — opens standings /tree view with live bracket */}
        <Link href="/standings?view=tree" className="group jarvis-lift jarvis-boot" style={{ animationDelay: "120ms" }}>
          <div className="relative h-44 rounded-2xl border border-[#fbbf24]/15 bg-[#fbbf24]/[0.02] hover:bg-[#fbbf24]/[0.05] hover:border-[#fbbf24]/30 transition-all duration-200 p-6 flex flex-col justify-between overflow-hidden shadow-[0_4px_24px_rgba(0,0,0,0.6),inset_0_1px_0_rgba(220,225,230,0.08)]">
            <div className="absolute -top-8 -right-8 w-40 h-40 rounded-full bg-[#fbbf24]/8 blur-2xl pointer-events-none group-hover:bg-[#fbbf24]/15 transition-all duration-300" />
            <div>
              <span className="text-[9px] font-black uppercase tracking-[0.3em] text-[#fbbf24]/40 block mb-3">Live Playoff Bracket</span>
              <h2 className="text-[28px] font-black tracking-[0.08em] uppercase text-white/80 leading-none">
                Playoff<br />Tree
              </h2>
            </div>
            <div className="flex items-end justify-between">
              <p className="text-[11px] text-white/30 leading-relaxed max-w-[160px]">
                Live playoff bracket with series scores, plus race and standings.
              </p>
              <span className="text-[#fbbf24]/30 group-hover:text-[#fbbf24]/70 text-xl transition-all duration-200 group-hover:translate-x-1">→</span>
            </div>
          </div>
        </Link>

        {/* Stats Leaders card */}
        <Link href="/stats" className="group jarvis-lift jarvis-boot" style={{ animationDelay: "180ms" }}>
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
        <Link href="/players" className="group jarvis-lift jarvis-boot" style={{ animationDelay: "240ms" }}>
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
