// Shared NHL utilities — used by ScoreboardBar, GoalFeed, and game centre pages.

/**
 * Strip diacritical marks (accents) from a player name before using it in a URL.
 * NHL API returns names like "Slafkovský" but our parquet uses "Slafkovsky".
 * NFD decomposition splits ý→y+combining-acute, then we drop the combining marks.
 */
export function normalizePlayerName(name: string): string {
  return name.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}

export const NHL_CODE: Record<string, string> = {
  LA: "LAK",
  NJ: "NJD",
  SJ: "SJS",
  TB: "TBL",
  UTH: "UTA",
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
// UTA _dark is the black logo — use light (colored) on our dark UI
// Note: TEAM_COLORS.TBL uses #0065B3 (medium blue, visible on dark bg) not #002868 (navy, too dark)
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

export const TEAM_FULL_NAMES: Record<string, string> = {
  ANA: "Anaheim Ducks",       BOS: "Boston Bruins",         BUF: "Buffalo Sabres",
  CGY: "Calgary Flames",      CAR: "Carolina Hurricanes",   CHI: "Chicago Blackhawks",
  COL: "Colorado Avalanche",  CBJ: "Columbus Blue Jackets", DAL: "Dallas Stars",
  DET: "Detroit Red Wings",   EDM: "Edmonton Oilers",       FLA: "Florida Panthers",
  LAK: "Los Angeles Kings",   MIN: "Minnesota Wild",        MTL: "Montréal Canadiens",
  NSH: "Nashville Predators", NJD: "New Jersey Devils",     NYI: "New York Islanders",
  NYR: "New York Rangers",    OTT: "Ottawa Senators",       PHI: "Philadelphia Flyers",
  PIT: "Pittsburgh Penguins", SEA: "Seattle Kraken",        SJS: "San Jose Sharks",
  STL: "St. Louis Blues",     TBL: "Tampa Bay Lightning",   TOR: "Toronto Maple Leafs",
  UTA: "Utah Mammoth",         VAN: "Vancouver Canucks",     VGK: "Vegas Golden Knights",
  WSH: "Washington Capitals", WPG: "Winnipeg Jets",
  // aliases
  LA: "Los Angeles Kings", NJ: "New Jersey Devils", SJ: "San Jose Sharks",
  TB: "Tampa Bay Lightning", UTH: "Utah Mammoth",
};

export const TEAM_COLORS: Record<string, string> = {
  ANA: "#F47A38", ARI: "#8C2633", BOS: "#FFB81C", BUF: "#003087",
  CGY: "#C8102E", CAR: "#CC0000", CHI: "#CF0A2C", COL: "#6F263D",
  CBJ: "#002654", DAL: "#006847", DET: "#CE1126", EDM: "#FF4C00",
  FLA: "#C8102E", LAK: "#A2AAAD", MIN: "#154734", MTL: "#AF1E2D",
  NSH: "#FFB81C", NJD: "#CE1126", NYI: "#00539B", NYR: "#0038A8",
  OTT: "#C52032", PHI: "#F74902", PIT: "#FCB514", SEA: "#99D9D9",
  SJS: "#006D75", STL: "#002F87", TBL: "#0065B3", TOR: "#003E7E",
  VAN: "#00843D", VGK: "#B4975A", WSH: "#C8102E", WPG: "#004C97",
  UTA: "#6CACE4", UTH: "#6CACE4",
  // short-code aliases
  LA: "#A2AAAD", NJ: "#CE1126", SJ: "#006D75", TB: "#0065B3",
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
  UTA: "#010101", UTH: "#010101",
  // short-code aliases
  LA: "#111111", NJ: "#000000", SJ: "#EA7200", TB: "#FFFFFF",
};
