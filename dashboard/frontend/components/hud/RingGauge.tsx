"use client";

import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";

export type RingGaugeProps = {
  /** 0..1 (or 0..100 if `max` is 100) */
  value: number;
  /** Defaults to 1 — meaning value is a fraction. Pass 100 if value is a percent. */
  max?: number;
  label?: string;
  sublabel?: string;
  /** Inner rendered text — defaults to `value` rounded with `decimals` */
  centerText?: string;
  decimals?: number;
  size?: number;
  thickness?: number;
  themeColor?: string;
  /** Higher value = "worse" — color flips amber > 0.66, red > 0.85. */
  invert?: boolean;
  className?: string;
};

export function RingGauge({
  value,
  max = 1,
  label,
  sublabel,
  centerText,
  decimals = 2,
  size = 92,
  thickness = 7,
  themeColor,
  invert = false,
  className = "",
}: RingGaugeProps) {
  const reduced = useReducedMotion();
  const ref = useRef<SVGCircleElement | null>(null);
  const [visible, setVisible] = useState(false);

  const pct = Math.max(0, Math.min(1, value / max));
  const r = (size - thickness) / 2;
  const circ = 2 * Math.PI * r;
  const targetOffset = circ * (1 - pct);

  // Decide stroke color
  const baseColor = themeColor || "var(--brand-hex)";
  const dangerColor = invert ? (pct > 0.85 ? "#f87171" : pct > 0.66 ? "#fbbf24" : baseColor) : baseColor;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (reduced) {
      el.style.strokeDashoffset = String(targetOffset);
      setVisible(true);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            el.style.transition = "stroke-dashoffset 900ms cubic-bezier(0.22, 1, 0.36, 1)";
            el.style.strokeDashoffset = String(targetOffset);
            setVisible(true);
            io.disconnect();
          }
        }
      },
      { threshold: 0.2 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [targetOffset, reduced]);

  const centerStr =
    centerText !== undefined
      ? centerText
      : max === 100
      ? `${(value).toFixed(decimals)}`
      : value.toFixed(decimals);

  return (
    <div className={`inline-flex flex-col items-center justify-center gap-1 ${className}`}>
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90 block">
          {/* Subtle dotted inner reticle behind the gauge for a HUD feel */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r - thickness / 2 - 4}
            stroke={dangerColor}
            strokeOpacity="0.08"
            strokeWidth="0.6"
            strokeDasharray="1 4"
            fill="none"
          />
          {/* Full track ring — cleaner with linecap-butt + lower opacity */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            stroke="rgba(255,255,255,0.10)"
            strokeWidth={thickness}
            fill="none"
          />
          {/* Active arc */}
          <circle
            ref={ref}
            cx={size / 2}
            cy={size / 2}
            r={r}
            stroke={dangerColor}
            strokeWidth={thickness}
            fill="none"
            strokeDasharray={circ}
            strokeDashoffset={circ}
            strokeLinecap="round"
            style={{ filter: visible ? `drop-shadow(0 0 6px ${dangerColor})` : "none" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="hud-mono text-lg leading-none tabular-nums" style={{ color: dangerColor }}>
            {centerStr}
          </span>
          {sublabel ? (
            <span className="hud-mono text-[9px] uppercase tracking-[0.14em] text-[var(--text-secondary)] mt-0.5">
              {sublabel}
            </span>
          ) : null}
        </div>
      </div>
      {label ? (
        <span className="hud-mono text-[9px] uppercase tracking-[0.18em] text-[var(--text-secondary)]">
          {label}
        </span>
      ) : null}
    </div>
  );
}
