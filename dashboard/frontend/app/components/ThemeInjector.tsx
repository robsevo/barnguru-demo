"use client";

import { useEffect } from "react";
import { usePathname } from "next/navigation";
import { useTheme } from "@/utils/themeContext";

// Cortex auto-theme — applied when on /players pages with no team theme active
const CORTEX = {
  brandHex: "#a78bfa",
  br: 167, bg: 139, bb: 250,
  // dark purple body/card/header values
  bodyR: 22,  bodyG: 14,  bodyB: 39,
  cardR: 27,  cardG: 15,  cardB: 49,
  midR:  16,  midG:  12,  midB:  26,
  headR: 19,  headG: 13,  headB: 32,
};

function hexToRgb(hex: string | undefined | null): [number, number, number] {
  if (!hex) return [9, 10, 12];
  const clean = hex.replace("#", "");
  const r = parseInt(clean.slice(0, 2), 16) || 0;
  const g = parseInt(clean.slice(2, 4), 16) || 0;
  const b = parseInt(clean.slice(4, 6), 16) || 0;
  return [r, g, b];
}

// Blend hex color toward the app's dark base (#090a0c) at given darkness (0–1)
function darkBlend(hex: string, darkness: number): [number, number, number] {
  const base = [9, 10, 12]; // #090a0c
  const [r, g, b] = hexToRgb(hex);
  return [
    Math.round(r * (1 - darkness) + base[0] * darkness),
    Math.round(g * (1 - darkness) + base[1] * darkness),
    Math.round(b * (1 - darkness) + base[2] * darkness),
  ];
}

function luminance([r, g, b]: [number, number, number]): number {
  return 0.2126 * (r / 255) + 0.7152 * (g / 255) + 0.0722 * (b / 255);
}

export default function ThemeInjector() {
  const { theme } = useTheme();
  const pathname = usePathname();
  const isCortexPage = pathname.startsWith("/players");

  useEffect(() => {
    const root = document.documentElement;

    if (!theme) {
      if (isCortexPage) {
        // Auto-apply cortex purple theme on /players pages
        root.setAttribute("data-team-theme", "cortex");
        root.style.setProperty("--brand-hex", CORTEX.brandHex);
        root.style.setProperty("--brand-r", String(CORTEX.br));
        root.style.setProperty("--brand-g", String(CORTEX.bg));
        root.style.setProperty("--brand-b", String(CORTEX.bb));
        root.style.setProperty("--body-base-r", String(CORTEX.bodyR));
        root.style.setProperty("--body-base-g", String(CORTEX.bodyG));
        root.style.setProperty("--body-base-b", String(CORTEX.bodyB));
        root.style.setProperty("--card-base-r", String(CORTEX.cardR));
        root.style.setProperty("--card-base-g", String(CORTEX.cardG));
        root.style.setProperty("--card-base-b", String(CORTEX.cardB));
        root.style.setProperty("--card-mid-r",  String(CORTEX.midR));
        root.style.setProperty("--card-mid-g",  String(CORTEX.midG));
        root.style.setProperty("--card-mid-b",  String(CORTEX.midB));
        root.style.setProperty("--header-r", String(CORTEX.headR));
        root.style.setProperty("--header-g", String(CORTEX.headG));
        root.style.setProperty("--header-b", String(CORTEX.headB));
      } else {
        root.removeAttribute("data-team-theme");
        const vars = [
          "--brand-hex", "--brand-r", "--brand-g", "--brand-b",
          "--body-base-r", "--body-base-g", "--body-base-b",
          "--card-base-r", "--card-base-g", "--card-base-b",
          "--card-mid-r",  "--card-mid-g",  "--card-mid-b",
          "--header-r",    "--header-g",    "--header-b",
        ];
        vars.forEach(v => root.style.removeProperty(v));
      }
      return;
    }

    const [pr, pg, pb] = hexToRgb(theme.primaryColor);
    // secondaryColor may be absent in themes saved before this field was added
    const [sr, sg, sb] = hexToRgb(theme.secondaryColor ?? theme.primaryColor);

    // Pick the darker of primary/secondary as the tint base (for backgrounds)
    const primaryRgb:   [number, number, number] = [pr, pg, pb];
    const secondaryRgb: [number, number, number] = [sr, sg, sb];
    const tintBase = luminance(primaryRgb) <= luminance(secondaryRgb) ? theme.primaryColor : theme.secondaryColor;

    // Body background: darkBlend(tintBase, 0.84) — distinct team colour, still dark
    const [bbr, bbg, bbb] = darkBlend(tintBase, 0.84);

    // Card surface: darker blend — close to pure dark but tinted
    const [cbr, cbg, cbb] = darkBlend(tintBase, 0.78);
    const [cmr, cmg, cmb] = darkBlend(tintBase, 0.92);

    // Header strip: very dark tint of secondary (cooler/deeper)
    const [hr, hg, hb] = darkBlend(theme.secondaryColor, 0.88);

    root.setAttribute("data-team-theme", theme.abbrev);

    // Accent (brand) color
    root.style.setProperty("--brand-hex", theme.primaryColor);
    root.style.setProperty("--brand-r", String(pr));
    root.style.setProperty("--brand-g", String(pg));
    root.style.setProperty("--brand-b", String(pb));

    // Background system
    root.style.setProperty("--body-base-r", String(bbr));
    root.style.setProperty("--body-base-g", String(bbg));
    root.style.setProperty("--body-base-b", String(bbb));

    // Card / glass surface
    root.style.setProperty("--card-base-r", String(cbr));
    root.style.setProperty("--card-base-g", String(cbg));
    root.style.setProperty("--card-base-b", String(cbb));
    root.style.setProperty("--card-mid-r",  String(cmr));
    root.style.setProperty("--card-mid-g",  String(cmg));
    root.style.setProperty("--card-mid-b",  String(cmb));

    // Header strip
    root.style.setProperty("--header-r", String(hr));
    root.style.setProperty("--header-g", String(hg));
    root.style.setProperty("--header-b", String(hb));
  }, [theme, isCortexPage]);

  return null;
}
