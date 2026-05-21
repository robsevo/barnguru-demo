"use client";

import dynamic from "next/dynamic";
import { ReactNode, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
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
  /** Optional per-`shot.type` color map. Used for two-team game shot maps so
   *  away vs home reads at a glance instead of everything in one tint. */
  colorByType?: Record<string, string>;
};

/**
 * Lazy-loaded 3D shot map. Renders the 2D fallback until:
 *   (a) the user toggles to 3D, AND
 *   (b) the component scrolls into view, AND
 *   (c) prefers-reduced-motion is not set.
 * Three.js bundle (~150KB gz) only downloads when conditions are met.
 */
export function Shot3D({ shots, themeColor, flip, goalie = false, fallback, defaultEnabled = true, colorByType }: Shot3DProps) {
  const reduced = useReducedMotion();
  const [enabled, setEnabled] = useState(defaultEnabled && !reduced);
  const [inView, setInView] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [mounted, setMounted] = useState(false);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => { setMounted(true); }, []);

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

  // Esc to close + body scroll lock while expanded. Skipping when expanded is
  // false means the page scrolls normally as usual.
  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setExpanded(false); };
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prevOverflow;
      window.removeEventListener("keydown", onKey);
    };
  }, [expanded]);

  const show3D = enabled && inView && !reduced;

  // Shared body — used both inline and inside the fullscreen portal so the
  // two views render the same scene with the same toggle state.
  const body = show3D
    ? <Shot3DSceneLazy shots={shots} themeColor={themeColor} flip={flip} goalie={goalie} colorByType={colorByType} />
    : fallback;

  // Controls strip (2D/3D toggle + expand/close button). Same buttons in both
  // views so the inline corner and the fullscreen corner stay consistent.
  const controls = (isExpanded: boolean) => (
    <div className="absolute top-2 left-2 z-20 flex items-center gap-1.5">
      <button
        type="button"
        onClick={() => setEnabled((v) => !v)}
        disabled={!!reduced}
        className="hud-mono text-[9px] uppercase tracking-[0.18em] px-2 py-1 rounded border disabled:opacity-50"
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
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="hud-mono text-[9px] uppercase tracking-[0.18em] px-2 py-1 rounded border flex items-center gap-1"
        style={{
          color: themeColor || "var(--brand-hex)",
          borderColor: "rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.35)",
          background: "rgba(0,0,0,0.45)",
          backdropFilter: "blur(6px)",
        }}
        aria-label={isExpanded ? "Close fullscreen shot map" : "Expand shot map"}
      >
        {isExpanded ? (
          <>
            <svg width="9" height="9" viewBox="0 0 10 10" fill="none" aria-hidden>
              <path d="M2 2L8 8M8 2L2 8" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
            </svg>
            CLOSE
          </>
        ) : (
          <>
            <svg width="9" height="9" viewBox="0 0 10 10" fill="none" aria-hidden>
              <path d="M1 4V1H4M9 4V1H6M1 6V9H4M9 6V9H6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
            </svg>
            EXPAND
          </>
        )}
      </button>
    </div>
  );

  return (
    <div ref={wrapRef} className="relative">
      {body}
      {controls(false)}

      {/* Fullscreen portal — backdrop + huge centered scene. We render the
          shared `body` here too so the toggle/legend/controls are all
          re-used; only the surrounding chrome differs. */}
      {mounted && expanded && createPortal(
        <div
          className="fixed inset-0 z-[9998] flex items-center justify-center p-4 sm:p-8"
          style={{ background: "rgba(2,4,8,0.92)", backdropFilter: "blur(8px)" }}
          onClick={(e) => { if (e.target === e.currentTarget) setExpanded(false); }}
        >
          <div
            className="relative w-full max-w-[1600px] rounded-xl border overflow-hidden"
            style={{
              borderColor: `${themeColor || "var(--brand-hex)"}55`,
              background: "rgba(8,10,14,0.85)",
              boxShadow: `0 24px 80px rgba(0,0,0,0.9), 0 0 60px ${themeColor || "var(--brand-hex)"}22`,
            }}
          >
            {body}
            {controls(true)}
          </div>
        </div>,
        document.body,
      )}
    </div>
  );
}
