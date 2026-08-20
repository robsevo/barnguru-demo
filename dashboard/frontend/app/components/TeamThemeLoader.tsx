"use client";

import { useEffect, useState } from "react";
import { useTheme } from "@/utils/themeContext";

export default function TeamThemeLoader() {
  const { theme, isShowingLoader, isCortexLoading, isBarnGuruLoading } = useTheme();
  const [progress, setProgress] = useState(0);
  const [visible, setVisible] = useState(false);
  const [exiting, setExiting] = useState(false);

  const activeLoader = isShowingLoader || isCortexLoading || isBarnGuruLoading;

  useEffect(() => {
    if (!activeLoader) {
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
    const duration = 5800;

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
  }, [activeLoader, visible]);

  if (!visible) return null;

  // Cortex loader — takes priority over any active team theme
  if (isCortexLoading) {
    const color = "#a78bfa";
    return (
      <div
        className="fixed inset-0 z-[9999] flex flex-col items-center justify-center"
        style={{
          background: `radial-gradient(ellipse 80% 60% at 50% 40%, rgba(167,139,250,0.14) 0%, rgba(0,0,0,0) 70%), #0a0614`,
          opacity: exiting ? 0 : 1,
          transition: "opacity 0.6s ease-in-out",
        }}
      >
        <div className="relative flex items-center justify-center" style={{ width: 160, height: 160 }}>
          <svg className="absolute inset-0" width="160" height="160" style={{ animation: "spin 1.2s linear infinite" }}>
            <defs>
              <linearGradient id="cortexArcGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor={color} stopOpacity="0.9" />
                <stop offset="100%" stopColor={color} stopOpacity="0.05" />
              </linearGradient>
            </defs>
            <circle cx="80" cy="80" r="70" fill="none" stroke="url(#cortexArcGrad)" strokeWidth="3" strokeDasharray="300 140" strokeLinecap="round"/>
          </svg>
          <svg className="absolute inset-0" width="160" height="160">
            <circle cx="80" cy="80" r="70" fill="none" stroke={color} strokeWidth="1" strokeOpacity="0.12"/>
          </svg>
          {/* Cortex brain icon centered */}
          <div className="rounded-full flex items-center justify-center"
            style={{ width: 96, height: 96, background: `radial-gradient(circle at 40% 35%, rgba(167,139,250,0.20) 0%, rgba(0,0,0,0.6) 100%)`, boxShadow: `0 0 40px rgba(167,139,250,0.28), inset 0 1px 0 rgba(255,255,255,0.08)` }}>
            <svg width="64" height="64" viewBox="0 0 48 48" fill="none">
              <path d="M24 2 L42 12 L42 36 L24 46 L6 36 L6 12 Z" stroke="#a78bfa" strokeWidth="0.9" strokeOpacity="0.30" fill="#a78bfa" fillOpacity="0.04"/>
              <path d="M24 10 C18 10 11 14 9 20 C7 25 9 31 13 34 C17 37 21 37 24 37" stroke="#a78bfa" strokeWidth="1.8" strokeLinecap="round" fill="rgba(167,139,250,0.08)"/>
              <path d="M24 10 C30 10 37 14 39 20 C41 25 39 31 35 34 C31 37 27 37 24 37" stroke="#a78bfa" strokeWidth="1.8" strokeLinecap="round" fill="rgba(167,139,250,0.08)"/>
              <line x1="24" y1="10" x2="24" y2="37" stroke="#c4b5fd" strokeWidth="0.9" strokeOpacity="0.18" strokeDasharray="2.5,3.5"/>
              <path d="M11 19 C14 18 17 19 17 22" stroke="#c4b5fd" strokeWidth="1.2" fill="none" strokeOpacity="0.55" strokeLinecap="round"/>
              <path d="M10 27 C13 26 16 27 16 30" stroke="#c4b5fd" strokeWidth="1.1" fill="none" strokeOpacity="0.40" strokeLinecap="round"/>
              <path d="M37 19 C34 18 31 19 31 22" stroke="#c4b5fd" strokeWidth="1.2" fill="none" strokeOpacity="0.55" strokeLinecap="round"/>
              <path d="M38 27 C35 26 32 27 32 30" stroke="#c4b5fd" strokeWidth="1.1" fill="none" strokeOpacity="0.40" strokeLinecap="round"/>
              <line x1="15" y1="22" x2="24" y2="24" stroke="#a78bfa" strokeWidth="0.8" strokeOpacity="0.28"/>
              <line x1="33" y1="22" x2="24" y2="24" stroke="#a78bfa" strokeWidth="0.8" strokeOpacity="0.28"/>
              <circle cx="15" cy="22" r="2.2" fill="#a78bfa" fillOpacity="0.88"/>
              <circle cx="33" cy="22" r="2.2" fill="#a78bfa" fillOpacity="0.88"/>
              <circle cx="14" cy="30" r="1.8" fill="#7c3aed" fillOpacity="0.75"/>
              <circle cx="34" cy="30" r="1.8" fill="#7c3aed" fillOpacity="0.75"/>
              <circle cx="24" cy="24" r="5.0" fill="#a78bfa" fillOpacity="0.10"/>
              <circle cx="24" cy="24" r="3.2" fill="#a78bfa" fillOpacity="0.95"/>
              <circle cx="24" cy="24" r="1.6" fill="white"   fillOpacity="0.72"/>
            </svg>
          </div>
        </div>
        <p className="mt-6 text-[13px] font-black uppercase tracking-[0.28em] bg-gradient-to-r from-white via-[#c4b5fd] to-[#a78bfa] bg-clip-text text-transparent">
          CORTEX
        </p>
        <p className="mt-1 text-[10px] font-medium uppercase tracking-[0.2em] text-white/30">
          Theme Activated
        </p>
        <div className="mt-6 rounded-full overflow-hidden" style={{ width: 160, height: 2, background: `rgba(167,139,250,0.12)` }}>
          <div className="h-full rounded-full transition-none" style={{ width: `${progress}%`, background: `linear-gradient(90deg, #a78bfa80, #a78bfa)` }}/>
        </div>
        <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  // BarnGuru restore loader — takes priority over any active team theme
  if (isBarnGuruLoading) {
    const gold = "#C9A84C";
    return (
      <div
        className="fixed inset-0 z-[9999] flex flex-col items-center justify-center"
        style={{
          background: `radial-gradient(ellipse 80% 60% at 50% 40%, rgba(201,168,76,0.10) 0%, rgba(0,0,0,0) 70%), #090a0c`,
          opacity: exiting ? 0 : 1,
          transition: "opacity 0.6s ease-in-out",
        }}
      >
        <div className="relative flex items-center justify-center" style={{ width: 160, height: 160 }}>
          {/* Spinning arc */}
          <svg className="absolute inset-0" width="160" height="160" style={{ animation: "spin 1.2s linear infinite" }}>
            <defs>
              <linearGradient id="barnArcGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor={gold} stopOpacity="0.9" />
                <stop offset="100%" stopColor={gold} stopOpacity="0.05" />
              </linearGradient>
            </defs>
            <circle cx="80" cy="80" r="70" fill="none" stroke="url(#barnArcGrad)" strokeWidth="3" strokeDasharray="300 140" strokeLinecap="round"/>
          </svg>
          {/* Static ring */}
          <svg className="absolute inset-0" width="160" height="160">
            <circle cx="80" cy="80" r="70" fill="none" stroke={gold} strokeWidth="1" strokeOpacity="0.12"/>
          </svg>
          {/* Logo */}
          <div className="rounded-full flex items-center justify-center"
            style={{ width: 96, height: 96, background: `radial-gradient(circle at 40% 35%, rgba(201,168,76,0.18) 0%, rgba(0,0,0,0.6) 100%)`, boxShadow: `0 0 40px rgba(201,168,76,0.25), inset 0 1px 0 rgba(255,255,255,0.08)` }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/logo-circle.png" alt="BarnGuru" width={88} height={88} className="object-contain"
              style={{ filter: `drop-shadow(0 2px 12px rgba(201,168,76,0.5))` }} />
          </div>
        </div>
        <p className="mt-6 text-[13px] font-black uppercase tracking-[0.28em]" style={{ color: gold }}>
          BarnGuru
        </p>
        <p className="mt-1 text-[10px] font-medium uppercase tracking-[0.2em] text-white/30">
          Theme Restored
        </p>
        <div className="mt-6 rounded-full overflow-hidden" style={{ width: 160, height: 2, background: `rgba(201,168,76,0.12)` }}>
          <div className="h-full rounded-full transition-none" style={{ width: `${progress}%`, background: `linear-gradient(90deg, ${gold}80, ${gold})` }}/>
        </div>
        <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (!theme) return null;

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
