"use client";

export function ScanLine({ className = "" }: { className?: string }) {
  return <div className={`hud-scan ${className}`} aria-hidden />;
}
