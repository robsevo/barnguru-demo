"use client";

import { ReactNode, CSSProperties } from "react";

export type HudBadgeProps = {
  children: ReactNode;
  tone?: "neutral" | "good" | "warn" | "bad" | "accent";
  pulse?: boolean;
  themeColor?: string;
  className?: string;
};

const toneStyles: Record<NonNullable<HudBadgeProps["tone"]>, { color: string; bg: string; border: string }> = {
  neutral: { color: "var(--text-secondary)", bg: "rgba(255,255,255,0.04)",  border: "rgba(255,255,255,0.08)" },
  good:    { color: "#86efac",               bg: "rgba(74,222,128,0.10)",  border: "rgba(74,222,128,0.30)" },
  warn:    { color: "#fcd34d",               bg: "rgba(251,191,36,0.10)",  border: "rgba(251,191,36,0.30)" },
  bad:     { color: "#fca5a5",               bg: "rgba(248,113,113,0.10)", border: "rgba(248,113,113,0.30)" },
  accent:  { color: "var(--brand-hex)",      bg: "rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.10)", border: "rgba(var(--brand-r),var(--brand-g),var(--brand-b),0.32)" },
};

export function HudBadge({
  children,
  tone = "neutral",
  pulse = false,
  themeColor,
  className = "",
}: HudBadgeProps) {
  const styles = toneStyles[tone];
  const dotStyle: CSSProperties = themeColor && tone === "accent"
    ? { background: themeColor }
    : {};
  return (
    <span
      className={`hud-mono inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] uppercase tracking-[0.14em] border ${className}`}
      style={{ color: styles.color, background: styles.bg, borderColor: styles.border }}
    >
      {pulse ? <span className="hud-pulse-dot" style={dotStyle} aria-hidden /> : null}
      {children}
    </span>
  );
}
