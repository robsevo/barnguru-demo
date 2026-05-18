"use client";

export function HudGrid({ className = "" }: { className?: string }) {
  return <div className={`hud-grid-bg ${className}`} aria-hidden />;
}
