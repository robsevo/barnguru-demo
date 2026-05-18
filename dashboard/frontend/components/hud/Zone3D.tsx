"use client";

import dynamic from "next/dynamic";
import { ReactNode, useEffect, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";
import type { ZoneActivation } from "./Zone3DScene";

export type { ZoneActivation } from "./Zone3DScene";

const Zone3DSceneLazy = dynamic(() => import("./Zone3DScene"), {
  ssr: false,
  loading: () => null,
});

export type Zone3DProps = {
  activations: ZoneActivation;
  themeColor?: string;
  fallback: ReactNode;
  defaultEnabled?: boolean;
};

export function Zone3D({ activations, themeColor, fallback, defaultEnabled = true }: Zone3DProps) {
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
        <Zone3DSceneLazy activations={activations} themeColor={themeColor} />
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
      >
        {show3D ? "◐ 2D" : "◑ 3D"}
      </button>
    </div>
  );
}
