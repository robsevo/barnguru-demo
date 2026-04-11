"use client";

import { useEffect, useState } from "react";
import { useTheme } from "@/utils/themeContext";

export default function TeamThemeLoader() {
  const { theme, isShowingLoader } = useTheme();
  const [progress, setProgress] = useState(0);
  const [visible, setVisible] = useState(false);
  const [exiting, setExiting] = useState(false);

  useEffect(() => {
    if (!isShowingLoader) {
      if (visible) {
        setExiting(true);
        const t = setTimeout(() => { setVisible(false); setExiting(false); setProgress(0); }, 600);
        return () => clearTimeout(t);
      }
      return;
    }
    setVisible(true);
    setExiting(false);
    setProgress(0);

    const start = Date.now();
    const duration = 5800; // slightly under 6s so progress hits 100% before dismiss

    const tick = () => {
      const elapsed = Date.now() - start;
      const pct = Math.min(100, (elapsed / duration) * 100);
      setProgress(pct);
      if (pct < 100) {
        raf = requestAnimationFrame(tick);
      }
    };
    let raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isShowingLoader, visible]);

  if (!visible || !theme) return null;

  const color = theme.primaryColor;
  const r = parseInt(color.slice(1, 3), 16);
  const g = parseInt(color.slice(3, 5), 16);
  const b = parseInt(color.slice(5, 7), 16);

  return (
    <div
      className="fixed inset-0 z-[9999] flex flex-col items-center justify-center"
      style={{
        background: `radial-gradient(ellipse 80% 60% at 50% 40%, rgba(${r},${g},${b},0.12) 0%, rgba(0,0,0,0) 70%), #090a0c`,
        opacity: exiting ? 0 : 1,
        transition: "opacity 0.6s ease-in-out",
      }}
    >
      {/* Outer glow ring */}
      <div
        className="relative flex items-center justify-center"
        style={{ width: 160, height: 160 }}
      >
        {/* Spinning arc */}
        <svg
          className="absolute inset-0"
          width="160"
          height="160"
          style={{ animation: "spin 1.2s linear infinite" }}
        >
          <defs>
            <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor={color} stopOpacity="0.9" />
              <stop offset="100%" stopColor={color} stopOpacity="0.05" />
            </linearGradient>
          </defs>
          <circle
            cx="80"
            cy="80"
            r="70"
            fill="none"
            stroke="url(#arcGrad)"
            strokeWidth="3"
            strokeDasharray="300 140"
            strokeLinecap="round"
          />
        </svg>

        {/* Static ring base */}
        <svg className="absolute inset-0" width="160" height="160">
          <circle
            cx="80"
            cy="80"
            r="70"
            fill="none"
            stroke={color}
            strokeWidth="1"
            strokeOpacity="0.12"
          />
        </svg>

        {/* Team logo */}
        <div
          className="rounded-full flex items-center justify-center"
          style={{
            width: 96,
            height: 96,
            background: `radial-gradient(circle at 40% 35%, rgba(${r},${g},${b},0.18) 0%, rgba(0,0,0,0.6) 100%)`,
            boxShadow: `0 0 40px rgba(${r},${g},${b},0.25), inset 0 1px 0 rgba(255,255,255,0.08)`,
          }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={theme.logoUrl}
            alt={theme.abbrev}
            width={72}
            height={72}
            className="object-contain"
            style={{ filter: `drop-shadow(0 2px 12px rgba(${r},${g},${b},0.5))` }}
          />
        </div>
      </div>

      {/* Team name */}
      <p
        className="mt-6 text-[13px] font-black uppercase tracking-[0.28em]"
        style={{ color }}
      >
        {theme.abbrev}
      </p>
      <p className="mt-1 text-[10px] font-medium uppercase tracking-[0.2em] text-white/30">
        Theme Activated
      </p>

      {/* Progress bar */}
      <div
        className="mt-6 rounded-full overflow-hidden"
        style={{ width: 160, height: 2, background: `rgba(${r},${g},${b},0.12)` }}
      >
        <div
          className="h-full rounded-full transition-none"
          style={{
            width: `${progress}%`,
            background: `linear-gradient(90deg, ${color}80, ${color})`,
          }}
        />
      </div>

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
