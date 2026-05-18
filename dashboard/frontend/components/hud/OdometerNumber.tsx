"use client";

import { useEffect, useState } from "react";
import { useMotionValue, useTransform, animate, useReducedMotion } from "framer-motion";

export type OdometerNumberProps = {
  value: number;
  decimals?: number;
  duration?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
};

export function OdometerNumber({
  value,
  decimals = 0,
  duration = 0.9,
  prefix = "",
  suffix = "",
  className = "",
}: OdometerNumberProps) {
  const reduced = useReducedMotion();
  const mv = useMotionValue(reduced ? value : 0);
  const formatted = useTransform(mv, (v) => v.toFixed(decimals));
  const [display, setDisplay] = useState(reduced ? value.toFixed(decimals) : (0).toFixed(decimals));

  useEffect(() => {
    if (reduced) {
      setDisplay(value.toFixed(decimals));
      mv.set(value);
      return;
    }
    const controls = animate(mv, value, { duration, ease: [0.22, 1, 0.36, 1] });
    const unsub = formatted.on("change", (v) => setDisplay(v));
    return () => {
      controls.stop();
      unsub();
    };
  }, [value, decimals, duration, reduced, mv, formatted]);

  return (
    <span className={`hud-mono tabular-nums jarvis-glitch-hover inline-block ${className}`}>
      {prefix}
      {display}
      {suffix}
    </span>
  );
}
