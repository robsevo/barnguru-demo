import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 30;

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── upstream Codes accounts ────────────────────────────────────────────────────
// Fetched from Vercel (not the VPS) so firewall port 8080 is not an issue.
const upstream_ACCOUNTS = [
  { label: "tv14s",    host: "tv14s.xyz",          port: 8080, user: "Serentiy2@ogbtv.com", pass: "0306@1954" },
  { label: "ampztl-a", host: "ampztl.xyz",         port: 8080, user: "arturo",              pass: "YZcm6gw6Ukwt" },
  { label: "ampztl-b", host: "ampztl.xyz",         port: 8080, user: "webtv1847",           pass: "YsAPRy6Jq8TJ" },
  { label: "lunar",    host: "lunar.pm",            port: 8080, user: "JeffOglesby",         pass: "Marriage101" },
];

// NHL-relevant keywords for filtering
const NHL_KEYWORDS = [
  "espn","tnt","tbs","nhl","sportsnet","tsn","msg","fox sport","fs1","fs2",
  "tva sport","rds","nesn","fanduel","bally","victory","nhln",
];

// Channel name aliases — maps normalized upstream names → our BarnCentre names
const NAME_MAP: Record<string, string> = {
  "rds":              "RDS",
  "rds hd":          "RDS",
  "rds info":        "RDS",
  "rds2":            "RDS2",
  "rds2 hd":         "RDS2",
  "tva sports":      "TVA Sports",
  "tva sports 1":    "TVA Sports",
  "tva sport 1":     "TVA Sports",
  "tva":             "TVA Sports",
  "tsn 1":           "TSN1", "tsn1": "TSN1",
  "tsn 2":           "TSN2", "tsn2": "TSN2",
  "tsn 3":           "TSN3", "tsn3": "TSN3",
  "tsn 4":           "TSN4", "tsn4": "TSN4",
  "tsn 5":           "TSN5", "tsn5": "TSN5",
  "sportsnet east":  "Sportsnet East",
  "sportsnet ontario": "Sportsnet Ontario",
  "sportsnet west":  "Sportsnet West",
  "sportsnet pacific": "Sportsnet Pacific",
  "sportsnet 360":   "Sportsnet 360",
  "sportsnet one":   "Sportsnet One",
  "nhl network":     "NHL Network",
  "espn":            "ESPN",
  "espn2":           "ESPN2",
  "espn 2":          "ESPN2",
  "espn+":           "ESPN+",
  "fs1":             "FS1",
  "fs2":             "FS2",
  "nesn":            "NESN",
};

function normalizeupstreamName(raw: string): string {
  let t = raw.trim();
  // Strip special chars/decorators (₮, ƒ, ✤, ✪, etc.)
  t = t.replace(/[^\x00-\x7F]/g, "").trim();
  // Strip upstream quality suffix: HD, FHD, UHD, SD
  t = t.replace(/\s+(fhd|uhd|hd|sd)\s*$/i, "");
  // Strip trailing flag codes: (B), (R), (E), (D)
  t = t.replace(/\s*\([A-Z]\)\s*$/i, "");
  // Strip country/language prefixes: "CA (FR)", "CA:", "CA-FR |", "CA "
  t = t.replace(/^(CA|US|UK)\s*(\([A-Z]{2}\))?\s*[|:\-]?\s*/i, "");
  // Strip group prefix patterns like "CA (FR) " or ". " at start
  t = t.replace(/^\.\s*/, "");
  // Normalize spaces
  t = t.replace(/\s+/g, " ").trim().toLowerCase();
  return t;
}

async function fetchupstreamChannels(): Promise<Map<string, string[]>> {
  // Returns: Map<BarnCentreName, [url, url, ...]>
  const channelMap = new Map<string, string[]>();

  const fetches = upstream_ACCOUNTS.map(async ({ label, host, port, user, pass }) => {
    const url = `http://${host}:${port}/get.php?username=${encodeURIComponent(user)}&password=${encodeURIComponent(pass)}&type=m3u_plus&output=m3u8`;
    try {
      const r = await fetch(url, {
        headers: { "User-Agent": "Mozilla/5.0 (compatible; Grtzky/1.0)" },
        signal: AbortSignal.timeout(15_000),
      });
      if (!r.ok) return;
      const text = await r.text();
      const lines = text.split("\n");
      for (let i = 0; i < lines.length - 1; i++) {
        const line = lines[i].trim();
        if (!line.startsWith("#EXTINF")) continue;
        const nameMatch = line.match(/tvg-name="([^"]+)"/);
        const rawName = nameMatch ? nameMatch[1] : line.split(",").pop()?.trim() ?? "";
        const streamUrl = lines[i + 1]?.trim() ?? "";
        if (!streamUrl.startsWith("http")) { i++; continue; }

        const norm = normalizeupstreamName(rawName);
        if (!NHL_KEYWORDS.some(k => norm.includes(k) || rawName.toLowerCase().includes(k))) { i++; continue; }

        const catalogName = NAME_MAP[norm];
        if (catalogName) {
          const existing = channelMap.get(catalogName) ?? [];
          if (!existing.includes(streamUrl)) existing.push(streamUrl);
          channelMap.set(catalogName, existing);
        }
        i++;
      }
    } catch {
      // Silently skip failed accounts
    }
  });

  await Promise.allSettled(fetches);
  return channelMap;
}

// All BarnCentre channel names in display order — must stay in sync with
// _BARNCENTRE_CHANNEL_NAMES in main.py
const ALL_CHANNEL_NAMES = [
  "TSN1","TSN2","TSN3","TSN4","TSN5",
  "Sportsnet East","Sportsnet Ontario","Sportsnet West","Sportsnet Pacific",
  "Sportsnet 360","Sportsnet One",
  "RDS","RDS2","TVA Sports",
  "ESPN","ESPN2","ESPN+",
  "NHL Network",
  "FS1","FS2",
  "NESN",
  "FanDuel",
];

export async function GET() {
  // Fetch base channel data from Python API + upstream in parallel
  const [baseResp, upstreamMap] = await Promise.allSettled([
    fetch(`${API_URL}/barncentre-channels`, {
      headers: { "ngrok-skip-browser-warning": "skip" },
      signal: AbortSignal.timeout(20_000),
    }),
    fetchupstreamChannels(),
  ]);

  let channels: Record<string, unknown>[] = [];
  if (baseResp.status === "fulfilled" && baseResp.value.ok) {
    const d = await baseResp.value.json();
    channels = d.channels ?? [];
  }

  const upstream = upstreamMap.status === "fulfilled" ? upstreamMap.value : new Map<string, string[]>();

  // Merge upstream URLs into existing channels
  for (const ch of channels) {
    const urls = upstream.get(ch.name as string);
    if (!urls?.length) continue;
    const existing = (ch.backup_urls as string[]) ?? [];
    ch.backup_urls = [...urls, ...existing.filter(u => !urls.includes(u))];
    if (!ch.primary_url) ch.primary_url = urls[0];
    ch.online = true;
  }

  // Add upstream-only channels that the Python API didn't return (e.g. RDS, TVA Sports)
  const existingNames = new Set(channels.map(c => c.name as string));
  for (const name of ALL_CHANNEL_NAMES) {
    if (existingNames.has(name)) continue;
    const urls = upstream.get(name);
    if (!urls?.length) continue;
    channels.push({
      name,
      primary_url:  urls[0],
      backup_urls:  urls.slice(1),
      programs:     [],
      online:       true,
    });
  }

  // Re-sort by ALL_CHANNEL_NAMES order
  const order = Object.fromEntries(ALL_CHANNEL_NAMES.map((n, i) => [n, i]));
  channels.sort((a, b) => (order[a.name as string] ?? 99) - (order[b.name as string] ?? 99));

  return NextResponse.json({ channels });
}
