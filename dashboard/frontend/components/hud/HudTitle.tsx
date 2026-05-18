"use client";

export type HudTitleProps = {
  title: string;
  subtitle?: string;
  size?: "sm" | "md" | "lg";
};

export function HudTitle({ title, subtitle, size = "sm" }: HudTitleProps) {
  const titleSize = size === "lg" ? "text-sm" : size === "md" ? "text-xs" : "text-[10px]";
  return (
    <div className="flex items-baseline gap-2 min-w-0 jarvis-glitch-hover">
      <span className="hud-mono text-[var(--brand-hex)] opacity-80 select-none jarvis-flicker" aria-hidden>
        ◢
      </span>
      <span
        className={`hud-mono ${titleSize} uppercase tracking-[0.18em] text-[var(--text-primary)] truncate`}
      >
        {title}
      </span>
      {subtitle ? (
        <span className="hud-mono text-[9px] uppercase tracking-[0.15em] text-[var(--text-secondary)] truncate">
          · {subtitle}
        </span>
      ) : null}
      <span className="hud-mono text-[var(--brand-hex)] opacity-80 select-none ml-auto" aria-hidden>
        ◣
      </span>
    </div>
  );
}
