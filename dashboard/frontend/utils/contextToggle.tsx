"use client";

/**
 * Playoff / Season context toggle.
 *
 * Persisted via localStorage. Drives the ``?context=playoffs|season`` query
 * param on the Phase 3 / Phase 17 endpoints so the player profile + Cortex
 * pages can flip between views without recomputing parquets.
 *
 * Default = "playoffs" while NHL postseason is active (heuristic: April–June).
 * Outside that window the default falls back to "season".
 */

import { createContext, useContext, useEffect, useState } from "react";

export type SeasonContext = "playoffs" | "season";

interface CtxValue {
  ctx: SeasonContext;
  setCtx: (c: SeasonContext) => void;
  /** True until the localStorage hydration has happened (first render only). */
  hydrated: boolean;
}

// Bumped to v2 so any stale "season" value persisted before the playoff
// window opened gets discarded — every visitor is re-evaluated against the
// month-based default below on next load.
const STORAGE_KEY = "grtzky:season-context:v2";

function defaultContext(): SeasonContext {
  if (typeof window === "undefined") return "playoffs";
  const m = new Date().getUTCMonth(); // 0 = Jan; April = 3, June = 5
  return m >= 3 && m <= 6 ? "playoffs" : "season";
}

const ContextToggle = createContext<CtxValue>({
  ctx: "playoffs",
  setCtx: () => {},
  hydrated: false,
});

export function SeasonContextProvider({ children }: { children: React.ReactNode }) {
  const [ctx, setCtxState] = useState<SeasonContext>("playoffs");
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw === "playoffs" || raw === "season") {
        setCtxState(raw);
      } else {
        setCtxState(defaultContext());
      }
    } catch {
      setCtxState(defaultContext());
    }
    setHydrated(true);
  }, []);

  function setCtx(c: SeasonContext) {
    setCtxState(c);
    try {
      window.localStorage.setItem(STORAGE_KEY, c);
    } catch {}
  }

  return (
    <ContextToggle.Provider value={{ ctx, setCtx, hydrated }}>
      {children}
    </ContextToggle.Provider>
  );
}

export function useSeasonContext(): CtxValue {
  return useContext(ContextToggle);
}

/**
 * Reusable two-segment pill. Keep it self-contained so any page can drop it in.
 */
export function SeasonContextPill({
  className = "",
}: { className?: string }) {
  const { ctx, setCtx, hydrated } = useSeasonContext();
  // Render the pill but don't reveal the active state until hydration
  // (avoids SSR/CSR mismatch when the stored value differs from the default).
  const activeP = hydrated && ctx === "playoffs";
  const activeS = hydrated && ctx === "season";

  return (
    <div
      role="group"
      aria-label="Stat context: playoffs or regular season"
      className={`inline-flex items-center rounded-full border border-white/[0.10] bg-white/[0.02] p-0.5 ${className}`}
    >
      <button
        type="button"
        onClick={() => setCtx("playoffs")}
        className={`px-3 py-1 rounded-full text-[10px] font-semibold uppercase tracking-[0.16em] transition-all
          ${activeP
            ? "bg-[#fbbf24] text-black shadow-[0_0_8px_rgba(251,191,36,0.45)]"
            : "text-white/55 hover:text-white/85"}`}
      >
        Playoffs
      </button>
      <button
        type="button"
        onClick={() => setCtx("season")}
        className={`px-3 py-1 rounded-full text-[10px] font-semibold uppercase tracking-[0.16em] transition-all
          ${activeS
            ? "bg-[#94a3b8] text-black shadow-[0_0_8px_rgba(148,163,184,0.30)]"
            : "text-white/55 hover:text-white/85"}`}
      >
        Season
      </button>
    </div>
  );
}
