"use client";

import dynamic from "next/dynamic";
import { ReactNode, useEffect, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";
import type { Shot } from "./Shot3DScene";

export type { Shot } from "./Shot3DScene";

// `Shot3DScene` carries the entire three.js + r3f + drei bundle. We split it
// off so it never touches the initial page bundle. SSR is disabled because
// three.js requires a DOM.
const Shot3DSceneLazy = dynamic(() => import("./Shot3DScene"), {
  ssr: false,
  loading: () => null,
});

export type Shot3DProps = {
  shots: Shot[];
  themeColor?: string;
  flip?: boolean;
  /** Render a goalie figure in each crease + improved net + hide goal lines. */
  goalie?: boolean;
  /** 2D fallback to render before the user opts in (or while bundle loads, or under prefers-reduced-motion). */
  fallback: ReactNode;
  /** Default false: user must opt in via the toggle. Reduces perf cost on first page visit. */
  defaultEnabled?: boolean;
};

/**
 * Lazy-loaded 3D shot map. Renders the 2D fallback until:
 *   (a) the user toggles to 3D, AND
 *   (b) the component scrolls into view, AND
 *   (c) prefers-reduced-motion is not set.
 * Three.js bundle (~150KB gz) only downloads when conditions are met.
 */
export function Shot3D({ shots, themeColor, flip, goalie = false, fallback, defaultEnabled = true }: Shot3DProps) {
  const reduced = useReducedMotion();
  const [enabled, setEnabled] = useState(defaultEnabled && !reduced);
  const [inView, setInView] = useState(false);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            setInView(true);
            io.disconnect();
            break;
          }
        }
      },
      { threshold: 0.25 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const show3D = enabled && inView && !reduced;

  return (
    <div ref={wrapRef} className="relative">
      {show3D ? (
        <Shot3DSceneLazy shots={shots} themeColor={themeColor} flip={flip} goalie={goalie} />
      ) : (
        fallback
      )}

      <button
        type="button"
        onClick={() => setEnabled((v) => !v)}
        disabled={!!reduced}
        className="absolute top-2 left-2 hud-mono text-[9px] uppercase tracking-[0.18em] px-2 py-1 rounded border z-10 disabled:opacity-50"
        style={{
          color: themeColor || "var(--brand-hex)",
          borderColor: "rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.35)",
          background: "rgba(0,0,0,0.45)",
          backdropFilter: "blur(6px)",
        }}
        aria-label={show3D ? "Switch to 2D shot map" : "Switch to 3D shot map"}
      >
        {show3D ? "◐ 2D" : "◑ 3D"}
      </button>
    </div>
  );
}
