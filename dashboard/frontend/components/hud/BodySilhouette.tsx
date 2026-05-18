"use client";

import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";

export type BodyZone = "head" | "torso" | "armL" | "armR" | "legL" | "legR" | "shoulder";

export type BodySilhouetteProps = {
  /** Zone -> intensity 0..1 (0 cool / 1 stressed) */
  intensity?: Partial<Record<BodyZone, number>>;
  variant?: "skater" | "goalie";
  themeColor?: string;
  width?: number;
  height?: number;
  className?: string;
  /** Show scanning line sweep */
  scan?: boolean;
};

/**
 * Stylized human silhouette with overlay zones. Intensities map each zone to a
 * red/amber/green tint to communicate fatigue / strain / health by region.
 */
export function BodySilhouette({
  intensity = {},
  themeColor,
  width = 200,
  height = 340,
  className = "",
  scan = true,
}: BodySilhouetteProps) {
  const reduced = useReducedMotion();
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) setInView(e.isIntersecting);
      },
      { threshold: 0.15 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const baseColor = themeColor || "var(--brand-hex)";
  const zoneColor = (lvl?: number) => {
    if (lvl === undefined) return "rgba(255,255,255,0.04)";
    if (lvl >= 0.75) return "rgba(248,113,113,0.40)";
    if (lvl >= 0.45) return "rgba(251,191,36,0.40)";
    return `rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.25)`;
  };

  return (
    <div ref={wrapRef} className={`relative ${className}`} style={{ width, height }}>
      <svg
        viewBox="0 0 200 340"
        width={width}
        height={height}
        style={{ filter: `drop-shadow(0 0 18px rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.18))` }}
      >
        {/* Outline silhouette */}
        <g fill="rgba(255,255,255,0.025)" stroke={baseColor} strokeOpacity={0.32} strokeWidth={1}>
          {/* Head */}
          <circle cx={100} cy={36} r={20} fill={zoneColor(intensity.head)} />
          {/* Neck */}
          <rect x={92} y={54} width={16} height={10} />
          {/* Torso */}
          <path
            d="M 64 64 Q 60 80 64 130 L 70 200 Q 72 210 78 212 L 122 212 Q 128 210 130 200 L 136 130 Q 140 80 136 64 Z"
            fill={zoneColor(intensity.torso)}
          />
          {/* Shoulders */}
          <ellipse cx={60} cy={70} rx={12} ry={8} fill={zoneColor(intensity.shoulder)} />
          <ellipse cx={140} cy={70} rx={12} ry={8} fill={zoneColor(intensity.shoulder)} />
          {/* Arms */}
          <path d="M 56 76 Q 44 130 50 180 L 60 180 Q 64 130 64 76 Z" fill={zoneColor(intensity.armL)} />
          <path d="M 144 76 Q 156 130 150 180 L 140 180 Q 136 130 136 76 Z" fill={zoneColor(intensity.armR)} />
          {/* Hips */}
          <path d="M 74 212 Q 78 230 80 244 L 120 244 Q 122 230 126 212 Z" />
          {/* Legs */}
          <path d="M 80 244 Q 78 290 84 330 L 96 330 Q 98 290 96 244 Z" fill={zoneColor(intensity.legL)} />
          <path d="M 120 244 Q 122 290 116 330 L 104 330 Q 102 290 104 244 Z" fill={zoneColor(intensity.legR)} />
        </g>

        {/* Center axis dotted line */}
        <line
          x1={100}
          y1={56}
          x2={100}
          y2={244}
          stroke={baseColor}
          strokeOpacity={0.18}
          strokeDasharray="2 4"
        />

        {/* Corner ticks (decorative HUD ticks at major joints) */}
        {[
          [36, 36],
          [164, 36],
          [60, 70],
          [140, 70],
          [80, 244],
          [120, 244],
        ].map(([cx, cy], i) => (
          <g key={i}>
            <circle cx={cx} cy={cy} r={2} fill={baseColor} opacity={0.6} />
          </g>
        ))}
      </svg>

      {/* Scan line — slow vertical sweep */}
      {scan && !reduced && inView ? (
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: "linear-gradient(180deg, transparent 0%, rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.22) 50%, transparent 100%)",
            mixBlendMode: "screen",
            animation: "bodyScan 5s linear infinite",
          }}
        />
      ) : null}

      <style jsx>{`
        @keyframes bodyScan {
          0%   { transform: translateY(-100%); opacity: 0; }
          10%  { opacity: 0.8; }
          90%  { opacity: 0.8; }
          100% { transform: translateY(100%); opacity: 0; }
        }
      `}</style>
    </div>
  );
}
