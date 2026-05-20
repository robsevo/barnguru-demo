"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";

export type NeuralNode = {
  id: string;
  label: string;
  weight: number; // 0..1 — drives node size + pulse intensity
};

export type NeuralGraphProps = {
  /** Center node label (e.g., "NN") */
  center?: string;
  nodes: NeuralNode[];
  width?: number;
  height?: number;
  themeColor?: string;
  /** Decorative-only mode: no labels, no center text */
  decorative?: boolean;
  className?: string;
};

/**
 * SVG node graph laid out as a center hub with `nodes.length` satellites arranged
 * radially. Edges flow from center → node with animated dashed flux when in view.
 * Idle-only — IntersectionObserver pauses animation when offscreen.
 */
export function NeuralGraph({
  center = "NN",
  nodes,
  width = 280,
  height = 220,
  themeColor,
  decorative = false,
  className = "",
}: NeuralGraphProps) {
  const reduced = useReducedMotion();
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [inView, setInView] = useState(false);

  const positions = useMemo(() => {
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * 0.36;
    const n = Math.max(1, nodes.length);
    return nodes.map((node, i) => {
      const angle = (-Math.PI / 2) + (i * (2 * Math.PI)) / n;
      return {
        ...node,
        x: cx + Math.cos(angle) * radius,
        y: cy + Math.sin(angle) * radius,
      };
    });
  }, [nodes, width, height]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) setInView(e.isIntersecting);
      },
      { threshold: 0.1 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const accent = themeColor || "var(--brand-hex)";
  const animate = inView && !reduced;

  return (
    <div ref={wrapRef} className={`relative ${className}`} style={{ width, height }}>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden={decorative}>
        <defs>
          <radialGradient id="hud-neural-glow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor={accent} stopOpacity="0.85" />
            <stop offset="60%" stopColor={accent} stopOpacity="0.20" />
            <stop offset="100%" stopColor={accent} stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* Edges — start at the edge of the center hub and stop just outside
            each satellite so the line never visually pierces either node. */}
        {positions.map((p) => {
          const cx = width / 2;
          const cy = height / 2;
          const dx = p.x - cx;
          const dy = p.y - cy;
          const dist = Math.hypot(dx, dy) || 1;
          const ux = dx / dist;
          const uy = dy / dist;
          // Center hub visible radius is 14 (stroked dark disc); start just outside it.
          const CENTER_R = 15;
          // Outer halo of each satellite is 8 + weight*14 → stop short of that.
          const rOuter = 8 + p.weight * 14;
          const x1 = cx + ux * CENTER_R;
          const y1 = cy + uy * CENTER_R;
          const x2 = p.x - ux * (rOuter + 1);
          const y2 = p.y - uy * (rOuter + 1);
          return (
            <line
              key={`e-${p.id}`}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke={accent}
              strokeOpacity={0.35 + p.weight * 0.35}
              strokeWidth={1}
              strokeDasharray="4 6"
              strokeLinecap="round"
              style={{
                animation: animate ? `dashflow ${2.4 + p.weight}s linear infinite` : "none",
              }}
            />
          );
        })}

        {/* Center hub */}
        <circle cx={width / 2} cy={height / 2} r={22} fill="url(#hud-neural-glow)" />
        <circle
          cx={width / 2}
          cy={height / 2}
          r={14}
          fill="rgba(0,0,0,0.55)"
          stroke={accent}
          strokeWidth={1}
        />
        {!decorative ? (
          <text
            x={width / 2}
            y={height / 2 + 4}
            textAnchor="middle"
            fontFamily="var(--font-mono)"
            fontSize={10}
            letterSpacing="0.18em"
            fill={accent}
          >
            {center}
          </text>
        ) : null}

        {/* Nodes — size strongly tracks weight so visual hierarchy reads */}
        {positions.map((p) => {
          // Node ring radius: 8 (weight 0) → 22 (weight 1) — 2.75× spread
          const rOuter = 8 + p.weight * 14;
          // Solid inner disc: 3 (weight 0) → 10 (weight 1) — 3.3× spread
          const rInner = 3 + p.weight * 7;
          return (
            <g key={`n-${p.id}`}>
              {/* Outer halo — opacity also tracks weight for double signal */}
              <circle
                cx={p.x}
                cy={p.y}
                r={rOuter}
                fill={accent}
                opacity={0.08 + p.weight * 0.32}
                style={{
                  animation: animate ? `nodePulse ${1.6 + Math.random() * 0.6}s ease-in-out infinite` : "none",
                  transformOrigin: `${p.x}px ${p.y}px`,
                }}
              />
              {/* Inner solid puck — brightens with weight */}
              <circle
                cx={p.x}
                cy={p.y}
                r={rInner}
                fill={accent}
                opacity={0.45 + p.weight * 0.55}
                stroke={accent}
                strokeWidth={1.2}
              />
              {!decorative ? (
                <text
                  x={p.x}
                  y={p.y + rOuter + 12}
                  textAnchor="middle"
                  fontFamily="var(--font-mono)"
                  fontSize={9}
                  letterSpacing="0.18em"
                  fill="var(--text-secondary)"
                  style={{ textTransform: "uppercase" }}
                >
                  {p.label}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
      <style jsx>{`
        @keyframes dashflow {
          to { stroke-dashoffset: -20; }
        }
        @keyframes nodePulse {
          0%, 100% { transform: scale(1); opacity: 0.5; }
          50%      { transform: scale(1.08); opacity: 0.85; }
        }
      `}</style>
    </div>
  );
}
