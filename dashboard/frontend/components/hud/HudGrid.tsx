"use client";

export function HudGrid({ className = "" }: { className?: string }) {
  return (
    <>
      <div className={`hud-grid-bg ${className}`} aria-hidden />
      {/* Ambient particle drift behind the grid — adds depth without
          piling on DOM nodes. */}
      <div className="hud-particles" aria-hidden />
    </>
  );
}
