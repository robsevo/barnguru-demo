"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

export interface GlossaryTerm {
  abbr: string;
  def: string;
}

export const GLOSSARY_MAP: Record<string, string> = {
  // Shooting / scoring
  "xG":           "Expected Goals — probability a shot scores based on location, angle, and type",
  "xGF":          "Expected Goals For — shot quality–weighted scoring chances generated per 60 minutes at 5-on-5",
  "xGA":          "Expected Goals Against — predicted goals conceded based on shot quality allowed",
  "xGF%":         "Expected Goals For % — share of total xG at 5-on-5; primary possession quality proxy",
  "Finishing":    "Goals minus xG — positive means the player is outperforming their shot quality",
  // Goalie
  "GSAx":         "Goals Saved Above Expected — saves above what an average goalie would make on the same shots",
  "SV%":          "Save Percentage — saves divided by total shots faced",
  "HDSV%":        "High-Danger Save % — saves on shots from the slot and in close to the net",
  "MDSV%":        "Medium-Danger Save % — saves on mid-distance shots, typically from the circles",
  "LDSV%":        "Low-Danger Save % — saves on shots from the perimeter and outside",
  "GA":           "Goals Against — total goals scored on the goalie",
  // Player ratings
  "RAPM":         "Regularized Adjusted Plus-Minus — player impact isolated from teammates and opponents using ridge regression",
  "WAR":          "Wins Above Replacement — total player value expressed in wins added over a replacement-level player",
  "GAR":          "Goals Above Replacement — same as WAR but expressed in goals instead of wins",
  "CDR":          "Composite Defensive Rating — weighted blend of defensive RAPM, blocked shots, takeaways, and zone suppression",
  "QoT":          "Quality of Teammates — average RAPM of linemates when a player is on the ice",
  "QoC":          "Quality of Competition — average RAPM of opponents faced; high QoC = tough matchups",
  "Archetype":    "Player cluster from K-means on RAPM + xG/game — groups players by style: Elite Scorer, Defensive D, etc.",
  // Form / trends
  "EWMA":         "Exponentially Weighted Moving Average — recent-form trend weighting recent games more heavily than older ones",
  "EWMA xGF/60":  "EWMA-smoothed Expected Goals For per 60 minutes — tracks offensive form over recent games",
  "Hot":          "Form flag — player's EWMA xGF/60 is significantly above their season baseline",
  "Cold":         "Form flag — player's EWMA xGF/60 is significantly below their season baseline",
  "Regime":       "Detected shift in a player's performance distribution — e.g. slump, breakout, or injury recovery",
  // Special teams
  "PP":           "Power Play — 5-on-4 situation when the opposing team has a penalty",
  "PK":           "Penalty Kill — 4-on-5 shorthanded situation when your team has a penalty",
  "PP%":          "Power Play percentage — goals scored per 100 power play opportunities",
  "PK%":          "Penalty Kill percentage — percentage of opposing power plays successfully killed",
  // Ice time / usage
  "TOI":          "Time On Ice — minutes played per game",
  "GP":           "Games Played",
  "SOG":          "Shots on Goal — shots that were on net and required the goalie to make a save",
  // Shot differentials
  "Corsi":        "Shot attempt differential — includes all shot attempts (on goal, missed, blocked) at 5-on-5",
  "Fenwick":      "Unblocked shot attempt differential — excludes blocked shots; slightly better possession proxy than Corsi",
  "Corsi%":       "CF/(CF+CA) — team shot attempt share at 5-on-5; primary possession proxy",
  // Skating / EDGE
  "EDGE":         "NHL Edge — official NHL puck and player tracking system using radar sensors in arenas",
  "TOP SPEED":    "Maximum recorded skating speed this season in km/h from NHL EDGE data",
  "DIST/GAME":    "Distance skated per game in kilometers from NHL EDGE data",
  "SHOT SPEED":   "Maximum recorded shot speed in mph from NHL EDGE data",
  // Zones
  "OZ":           "Offensive Zone — time spent in the attacking third of the ice",
  "NZ":           "Neutral Zone — time spent in the middle third of the ice",
  "DZ":           "Defensive Zone — time spent in the defending third of the ice",
  "OZS%":         "Offensive Zone Start % — share of faceoffs taken in the offensive zone; higher = sheltered usage",
  // Fatigue / schedule
  "FI":           "Fatigue Index — composite 0.0 (fully rested) to 1.0 (severely fatigued) score per player per game",
  "B2B":          "Back-to-Back — two games on consecutive days; one of the strongest fatigue drivers in the model",
  "SOS":          "Strength of Schedule — difficulty of opponents faced; used to normalise per-game stats",
  // Injury
  "DTD":          "Day-to-Day — player is injured but expected to return within 1–3 days",
  "IR":           "Injured Reserve — player placed on IR, ineligible to play for at least 7 days",
  // Market
  "Kelly":        "Kelly Criterion — bankroll sizing formula: bet size = edge / odds; maximises long-run growth",
  "Edge":         "Model edge — the gap between GRTZKY's win probability and Polymarket's implied probability",
  // Contract
  "AAV":          "Average Annual Value — yearly cap hit of a player's contract",
  "Cap Efficiency": "WAR per $1M of AAV — measures how much value a player delivers relative to their cap cost",
  "NHLe":         "NHL Equivalency — translation factor converting stats from other leagues to NHL level",
}

const TOOLTIP_W = 208; // w-52 = 208px

export function GlossTip({ term, children }: { term: string; children?: React.ReactNode }) {
  const def = GLOSSARY_MAP[term];
  const ref = useRef<HTMLSpanElement>(null);
  const [tipPos, setTipPos] = useState<{ left: number; top: number } | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => { setMounted(true); }, []);

  if (!def) return <>{children ?? term}</>;

  const handleMouseEnter = () => {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const left =
      rect.left + TOOLTIP_W > window.innerWidth - 12
        ? rect.right - TOOLTIP_W
        : rect.left;
    setTipPos({ left, top: rect.top });
  };

  const handleMouseLeave = () => setTipPos(null);

  return (
    <>
      <span
        ref={ref}
        className="cursor-help"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <span className="border-b border-dashed border-white/20 hover:border-white/40 transition-colors duration-150">
          {children ?? term}
        </span>
      </span>
      {mounted && tipPos && createPortal(
        <div
          className="pointer-events-none fixed z-[9999] w-52 rounded-lg border border-white/[0.07] bg-[#080d18] px-3 py-2 shadow-[0_4px_20px_rgba(0,0,0,0.9)] leading-none font-sans"
          style={{ left: tipPos.left, top: tipPos.top, transform: "translateY(calc(-100% - 8px))" }}
        >
          <span className="text-[10px] font-semibold tracking-wider text-[#94a3b8] block mb-0.5">{term}</span>
          <span className="text-[9px] font-mono text-white/50 leading-tight">{def}</span>
        </div>,
        document.body
      )}
    </>
  );
}

export default function Glossary({ terms }: { terms: GlossaryTerm[] }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-xl border border-white/[0.10] bg-white/[0.02] backdrop-blur-sm shadow-[inset_0_1px_0_rgba(255,255,255,0.06),inset_0_-1px_0_rgba(0,0,0,0.35),0_4px_24px_rgba(0,0,0,0.65),0_1px_3px_rgba(0,0,0,0.35)] overflow-hidden mt-4">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-2.5 border-b border-white/[0.10] bg-transparent hover:bg-white/[0.04] transition-all duration-200"
      >
        <div className="flex items-center gap-3">
          <h2 className="text-[13px] font-semibold uppercase tracking-[0.22em] text-[#94a3b8]">
            Glossary
          </h2>
          <span className="text-[11px] font-mono text-white/20">
            {terms.length} terms
          </span>
        </div>
        <span className="text-[10px] font-semibold tracking-widest text-white/25 hover:text-white/50 transition-colors duration-200">
          {open ? "▼ HIDE" : "▶ SHOW"}
        </span>
      </button>

      {open && (
        <div className="p-4">
          <div className="grid gap-x-8 gap-y-1.5 sm:grid-cols-2 lg:grid-cols-3">
            {terms.map((t) => (
              <div key={t.abbr} className="flex items-baseline gap-2 min-w-0">
                <span className="text-[11px] font-semibold tracking-wider text-[#94a3b8] shrink-0 w-16 text-right">
                  {t.abbr}
                </span>
                <span className="text-white/20 font-semibold shrink-0">—</span>
                <span className="text-[11px] font-mono text-[#475569] leading-snug">
                  {t.def}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
