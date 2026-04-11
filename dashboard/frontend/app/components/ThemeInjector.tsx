"use client";

import { useEffect } from "react";
import { useTheme } from "@/utils/themeContext";

function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.replace("#", "");
  const r = parseInt(clean.slice(0, 2), 16) || 0;
  const g = parseInt(clean.slice(2, 4), 16) || 0;
  const b = parseInt(clean.slice(4, 6), 16) || 0;
  return [r, g, b];
}

export default function ThemeInjector() {
  const { theme } = useTheme();

  useEffect(() => {
    const root = document.documentElement;
    if (!theme) {
      root.removeAttribute("data-team-theme");
      root.style.removeProperty("--brand-hex");
      root.style.removeProperty("--brand-r");
      root.style.removeProperty("--brand-g");
      root.style.removeProperty("--brand-b");
      return;
    }
    const [r, g, b] = hexToRgb(theme.primaryColor);
    root.setAttribute("data-team-theme", theme.abbrev);
    root.style.setProperty("--brand-hex", theme.primaryColor);
    root.style.setProperty("--brand-r", String(r));
    root.style.setProperty("--brand-g", String(g));
    root.style.setProperty("--brand-b", String(b));
  }, [theme]);

  return null;
}
