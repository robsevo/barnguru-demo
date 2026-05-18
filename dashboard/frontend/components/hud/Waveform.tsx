"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";

export type WaveformProps = {
  data: number[];
  width?: number;
  height?: number;
  themeColor?: string;
  /** Show baseline / midline */
  midline?: boolean;
  /** Show area fill below the line */
  fill?: boolean;
  className?: string;
  label?: string;
  ariaLabel?: string;
};

export function Waveform({
  data,
  width = 220,
  height = 56,
  themeColor,
  midline = true,
  fill = true,
  className = "",
  label,
  ariaLabel,
}: WaveformProps) {
  const reduced = useReducedMotion();
  const pathRef = useRef<SVGPathElement | null>(null);
  const fillRef = useRef<SVGPathElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [animated, setAnimated] = useState(false);

  const { line, area, lengthHint } = useMemo(() => {
    if (!data || data.length === 0) return { line: "", area: "", lengthHint: 0 };
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const step = data.length === 1 ? 0 : width / (data.length - 1);
    const y = (v: number) => height - 4 - ((v - min) / range) * (height - 8);
    const pts = data.map((v, i) => [i * step, y(v)] as const);
    const lineD = pts.map(([x, yv], i) => `${i === 0 ? "M" : "L"}${x.toFixed(2)},${yv.toFixed(2)}`).join(" ");
    const areaD = `${lineD} L${width},${height} L0,${height} Z`;
    return { line: lineD, area: areaD, lengthHint: width * 1.4 };
  }, [data, width, height]);

  useEffect(() => {
    const el = pathRef.current;
    if (!el) return;
    if (reduced) {
      el.style.strokeDasharray = "none";
      el.style.strokeDashoffset = "0";
      setAnimated(true);
      return;
    }
    const len = el.getTotalLength?.() ?? lengthHint;
    el.style.strokeDasharray = String(len);
    el.style.strokeDashoffset = String(len);
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            requestAnimationFrame(() => {
              el.style.transition = "stroke-dashoffset 1100ms cubic-bezier(0.22, 1, 0.36, 1)";
              el.style.strokeDashoffset = "0";
              setAnimated(true);
            });
            obs.disconnect();
          }
        }
      },
      { threshold: 0.15 }
    );
    if (containerRef.current) obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, [line, reduced, lengthHint]);

  const stroke = themeColor || "var(--brand-hex)";
  const fillColor = themeColor
    ? themeColor
    : "rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.18)";

  if (!data || data.length === 0) {
    return (
      <div className={`relative ${className}`} style={{ width, height }}>
        <span className="hud-mono text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">no signal</span>
      </div>
    );
  }

  return (
    <div ref={containerRef} className={`relative ${className}`} style={{ width, height }} aria-label={ariaLabel}>
      {label ? (
        <span className="absolute top-0 left-0 hud-mono text-[9px] uppercase tracking-[0.18em] text-[var(--text-secondary)]">
          {label}
        </span>
      ) : null}
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {midline ? (
          <line
            x1={0}
            y1={height / 2}
            x2={width}
            y2={height / 2}
            stroke="rgba(255,255,255,0.06)"
            strokeDasharray="2 4"
          />
        ) : null}
        {fill ? (
          <path
            ref={fillRef}
            d={area}
            fill={fillColor}
            opacity={animated ? 0.18 : 0}
            style={{ transition: "opacity 700ms ease 200ms" }}
          />
        ) : null}
        <path
          ref={pathRef}
          d={line}
          fill="none"
          stroke={stroke}
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ filter: `drop-shadow(0 0 3px ${stroke})` }}
        />
      </svg>
    </div>
  );
}
