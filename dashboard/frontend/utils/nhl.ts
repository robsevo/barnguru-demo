// Shared NHL utilities — used by ScoreboardBar, GoalFeed, and game centre pages.

export const NHL_CODE: Record<string, string> = {
  LA: "LAK",
  NJ: "NJD",
  SJ: "SJS",
  TB: "TBL",
};

export const POS_MAP: Record<string, string> = {
  C: "C", G: "G",
  L: "LW", LW: "LW",
  R: "RW", RW: "RW",
  LD: "LD", RD: "RD", D: "D",
};

export function fmtPos(pos: string | null | undefined): string {
  if (!pos) return "";
  return POS_MAP[pos.toUpperCase()] ?? pos.toUpperCase();
}

// TBL _light is the blue logo; _dark is the white one — use dark on our dark UI
const DARK_VARIANT = new Set(["TBL", "TB"]);

export function logoUrl(abbrev: string): string {
  const code    = NHL_CODE[abbrev] ?? abbrev;
  const variant = DARK_VARIANT.has(abbrev) || DARK_VARIANT.has(code) ? "dark" : "light";
  return `https://assets.nhle.com/logos/nhl/svg/${code}_${variant}.svg`;
}

export const PERIOD_LABELS: Record<number, string> = {
  1: "1ST",
  2: "2ND",
  3: "3RD",
  4: "OT",
  5: "SO",
};

export function periodLabel(period: number): string {
  return PERIOD_LABELS[period] ?? `P${period}`;
}

export const STRENGTH_STYLE: Record<string, string> = {
  PP: "text-[#fbbf24] border-[#fbbf24]/30",
  SH: "text-[#4ade80] border-[#4ade80]/30",
  EN: "text-white/30   border-white/10",
};

export const TEAM_COLORS: Record<string, string> = {
  ANA: "#F47A38", ARI: "#8C2633", BOS: "#FFB81C", BUF: "#003087",
  CGY: "#C8102E", CAR: "#CC0000", CHI: "#CF0A2C", COL: "#6F263D",
  CBJ: "#002654", DAL: "#006847", DET: "#CE1126", EDM: "#FF4C00",
  FLA: "#C8102E", LAK: "#A2AAAD", MIN: "#154734", MTL: "#AF1E2D",
  NSH: "#FFB81C", NJD: "#CE1126", NYI: "#00539B", NYR: "#0038A8",
  OTT: "#C52032", PHI: "#F74902", PIT: "#FCB514", SEA: "#99D9D9",
  SJS: "#006D75", STL: "#002F87", TBL: "#002868", TOR: "#003E7E",
  VAN: "#00843D", VGK: "#B4975A", WSH: "#C8102E", WPG: "#004C97",
  UTH: "#006D3D",
  // short-code aliases
  LA: "#A2AAAD", NJ: "#CE1126", SJ: "#006D75", TB: "#002868",
};

export const TEAM_SECONDARY: Record<string, string> = {
  ANA: "#B09965", ARI: "#E2D200", BOS: "#000000", BUF: "#FCB514",
  CGY: "#F1BE48", CAR: "#000000", CHI: "#000000", COL: "#236192",
  CBJ: "#CE1126", DAL: "#111111", DET: "#FFFFFF", EDM: "#00205B",
  FLA: "#041E42", LAK: "#111111", MIN: "#DDAF38", MTL: "#192168",
  NSH: "#041E42", NJD: "#000000", NYI: "#F47D20", NYR: "#CE1126",
  OTT: "#C2912C", PHI: "#000000", PIT: "#000000", SEA: "#001628",
  SJS: "#EA7200", STL: "#FCB514", TBL: "#FFFFFF", TOR: "#FFFFFF",
  VAN: "#001F5B", VGK: "#333F42", WSH: "#041E42", WPG: "#AC162C",
  UTH: "#010101",
  // short-code aliases
  LA: "#111111", NJ: "#000000", SJ: "#EA7200", TB: "#FFFFFF",
};
