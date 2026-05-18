"use client";

import { ReactNode, CSSProperties } from "react";
import { HudTitle } from "./HudTitle";

export type HudPanelProps = {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  themeColor?: string;
  scanline?: boolean;
  padded?: boolean;
  allCorners?: boolean;
  /** Add a one-shot shimmer band sweep on mount. Default true. */
  shimmer?: boolean;
  className?: string;
  style?: CSSProperties;
  right?: ReactNode;
};

export function HudPanel({
  title,
  subtitle,
  children,
  themeColor,
  scanline = false,
  padded = true,
  allCorners = false,
  shimmer = true,
  className = "",
  style,
  right,
}: HudPanelProps) {
  const themed: CSSProperties = themeColor
    ? ({ ["--hud-corner" as string]: themeColor } as CSSProperties)
    : {};
  return (
    <div
      className={`hud-panel ${allCorners ? "hud-panel--all-corners" : ""} ${shimmer ? "jarvis-shimmer" : ""} ${className}`}
      style={{ ...themed, ...style }}
    >
      {allCorners ? (
        <>
          <span className="hud-panel__corner-tr" />
          <span className="hud-panel__corner-bl" />
        </>
      ) : null}
      {scanline ? <div className="hud-scan" aria-hidden /> : null}
      {title ? (
        <div className="px-3 pt-2 pb-1.5 flex items-center justify-between gap-2 border-b border-white/[0.04]">
          <HudTitle title={title} subtitle={subtitle} />
          {right}
        </div>
      ) : null}
      <div className={`${padded ? "p-3" : ""} relative flex-1 min-h-0`}>{children}</div>
    </div>
  );
}
