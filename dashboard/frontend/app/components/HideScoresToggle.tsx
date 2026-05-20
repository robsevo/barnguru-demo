"use client";

import { useEffect, useState } from "react";
import { useHideAllScores } from "@/utils/scoreVisibility";

/**
 * Floating hide-scores pill — bottom-left, mirrors the live-goal-feed badge's
 * size and chrome so the two corners feel like a matched set. Brand-gold when
 * scores are hidden (active state pops), neutral dim when scores are visible.
 *
 * Hidden on the /login route via AppShell's existing gate.
 */
export default function HideScoresToggle() {
  const { hideAll, toggleAll } = useHideAllScores();
  const [mounted, setMounted] = useState(false);

  // Avoid SSR/CSR mismatch — localStorage state can only be read after mount.
  useEffect(() => { setMounted(true); }, []);
  if (!mounted) return null;

  const label = hideAll ? "SCORES HIDDEN" : "HIDE SCORES";

  return (
    <button
      onClick={toggleAll}
      title={hideAll ? "Show scores" : "Hide every score on the site"}
      aria-pressed={hideAll}
      className={`fixed bottom-5 left-5 z-50 flex items-center gap-2 px-3 py-2 rounded-full
        bg-[#0d1117]/90 backdrop-blur-md border transition-all duration-300 cursor-pointer
        ${hideAll
          ? "border-[#C9A84C]/70 shadow-[0_0_18px_rgba(201,168,76,0.40)]"
          : "border-[#C9A84C]/30 shadow-[0_0_10px_rgba(201,168,76,0.18)] opacity-75 hover:opacity-100"
        }`}
    >
      {/* Eye / eye-off icon — small SVG to match the goal feed badge size */}
      <svg width="16" height="16" viewBox="0 0 20 20" fill="none" style={{ flexShrink: 0 }}>
        {hideAll ? (
          <>
            <path d="M2.5 2.5 L17.5 17.5" stroke="#C9A84C" strokeWidth="1.4" strokeLinecap="round" />
            <path d="M3.5 10c1.6-2.8 3.8-4.3 6.5-4.3 1.2 0 2.3.3 3.3.8M16.5 10c-.9 1.6-2 2.8-3.4 3.6"
              stroke="#C9A84C" strokeWidth="1.3" strokeLinecap="round" fill="none" />
            <circle cx="10" cy="10" r="2.2" stroke="#C9A84C" strokeWidth="1.2" fill="none" />
          </>
        ) : (
          <>
            <path d="M3.5 10c1.6-2.8 3.8-4.3 6.5-4.3s4.9 1.5 6.5 4.3c-1.6 2.8-3.8 4.3-6.5 4.3S5.1 12.8 3.5 10z"
              stroke="#C9A84C" strokeWidth="1.3" fill="none" strokeLinejoin="round" />
            <circle cx="10" cy="10" r="2.4" fill="#C9A84C" fillOpacity="0.85" />
          </>
        )}
      </svg>
      <span className="hud-mono text-[9px] font-black uppercase tracking-[0.18em]"
        style={{ color: hideAll ? "#E8D090" : "rgba(201,168,76,0.65)" }}>
        {label}
      </span>
    </button>
  );
}
