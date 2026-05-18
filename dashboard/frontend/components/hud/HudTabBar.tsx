"use client";

import { motion } from "framer-motion";

export type HudTab = {
  id: string;
  label: string;
  badge?: string | number;
};

export type HudTabBarProps = {
  tabs: HudTab[];
  active: string;
  onChange: (id: string) => void;
  themeColor?: string;
  size?: "sm" | "md";
  className?: string;
  /** Render scrollable on small screens */
  scroll?: boolean;
};

export function HudTabBar({
  tabs,
  active,
  onChange,
  themeColor,
  size = "md",
  className = "",
  scroll = true,
}: HudTabBarProps) {
  const text = size === "sm" ? "text-[10px]" : "text-[11px]";
  const pad = size === "sm" ? "px-2.5 py-1" : "px-3 py-1.5";
  const accent = themeColor || "var(--brand-hex)";

  return (
    <div
      className={`relative flex items-center gap-1 ${scroll ? "overflow-x-auto" : ""} ${className}`}
      role="tablist"
    >
      {tabs.map((t) => {
        const isActive = t.id === active;
        return (
          <button
            key={t.id}
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(t.id)}
            className={`relative shrink-0 hud-mono ${pad} ${text} uppercase tracking-[0.16em] transition-colors duration-150 ${
              isActive ? "text-[var(--text-primary)]" : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
            }`}
          >
            <span className="relative z-10 flex items-center gap-1.5">
              {t.label}
              {t.badge !== undefined ? (
                <span className="text-[9px] opacity-70">{t.badge}</span>
              ) : null}
            </span>
            {isActive ? (
              <motion.span
                layoutId="hud-tab-underline"
                transition={{ type: "spring", stiffness: 380, damping: 32 }}
                className="absolute inset-x-0 -bottom-px h-px"
                style={{
                  background: accent,
                  boxShadow: `0 0 6px ${accent}, 0 0 12px ${accent}`,
                }}
              />
            ) : null}
          </button>
        );
      })}
    </div>
  );
}
