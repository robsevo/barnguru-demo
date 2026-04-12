import { NextRequest, NextResponse } from "next/server";

const NHL_REMAP: Record<string, string> = {
  UTH: "UTA", LA: "LAK", NJ: "NJD", SJ: "SJS", TB: "TBL",
};

// CapWages team slug = lowercase full team name, spaces → underscores
const SLUGS: Record<string, string> = {
  ANA: "anaheim_ducks",
  BOS: "boston_bruins",
  BUF: "buffalo_sabres",
  CGY: "calgary_flames",
  CAR: "carolina_hurricanes",
  CHI: "chicago_blackhawks",
  COL: "colorado_avalanche",
  CBJ: "columbus_blue_jackets",
  DAL: "dallas_stars",
  DET: "detroit_red_wings",
  EDM: "edmonton_oilers",
  FLA: "florida_panthers",
  LAK: "los_angeles_kings",
  MIN: "minnesota_wild",
  MTL: "montreal_canadiens",
  NSH: "nashville_predators",
  NJD: "new_jersey_devils",
  NYI: "new_york_islanders",
  NYR: "new_york_rangers",
  OTT: "ottawa_senators",
  PHI: "philadelphia_flyers",
  PIT: "pittsburgh_penguins",
  SEA: "seattle_kraken",
  SJS: "san_jose_sharks",
  STL: "st_louis_blues",
  TBL: "tampa_bay_lightning",
  TOR: "toronto_maple_leafs",
  UTA: "utah_hockey_club",
  VAN: "vancouver_canucks",
  VGK: "vegas_golden_knights",
  WSH: "washington_capitals",
  WPG: "winnipeg_jets",
};

const EMPTY = (team: string, status: string) => ({
  team, players: [], cap_ceiling: 95_500_000, total_hit: null, cap_space: null, projected_cap_space: null, status,
});

// "Jan. 2, 2001" → age in years
function calcAge(born: string | undefined): number | null {
  if (!born) return null;
  const d = new Date(born.replace(/\./g, ""));
  if (isNaN(d.getTime())) return null;
  const now = new Date();
  let age = now.getFullYear() - d.getFullYear();
  if (now.getMonth() - d.getMonth() < 0 || (now.getMonth() === d.getMonth() && now.getDate() < d.getDate())) age--;
  return age;
}

// "Standard Contract (Extension)" → "EXT", "Entry Level Contract" → "ELC", etc.
function contractLabel(type: string): string {
  const t = (type ?? "").toLowerCase();
  if (t.includes("entry")) return "ELC";
  if (t.includes("two-way") || t.includes("2-way")) return "2-WAY";
  if (t.includes("extension")) return "EXT";
  if (t.includes("professional tryout") || t.includes("pto")) return "PTO";
  if (t.includes("amateur")) return "ATO";
  if (t.includes("standard")) return "STD";
  return type ? type.slice(0, 6).toUpperCase() : "STD";
}

function parseMoney(val: unknown): number | null {
  if (val == null) return null;
  if (typeof val === "number") return Math.round(val);
  if (typeof val === "string") {
    const n = parseFloat(val.replace(/[$,]/g, ""));
    return isNaN(n) ? null : Math.round(n);
  }
  return null;
}

// Find the current-season contract detail (2025-26 or fall back to first)
function currentDetail(details: Record<string, unknown>[]): Record<string, unknown> | null {
  if (!details?.length) return null;
  const cur = details.find(d => String(d.season ?? "").startsWith("2025")) ?? details[0];
  return cur ?? null;
}

interface CapWagesPlayer {
  name?: string;
  pos?: string;
  status?: string;
  born?: string;
  contracts?: Array<{
    type?: string;
    expiryStatus?: string;
    details?: Record<string, unknown>[];
    buyout?: Record<string, unknown>;
  }>;
  acquisitionDetails?: string;
}

function parseDeadCapEntry(p: CapWagesPlayer, kind: "retained" | "buyout" | "buried") {
  const name = p.name ?? "";
  if (!name || name.length < 2) return null;
  const nameParts = name.includes(",")
    ? name.split(",").map((s: string) => s.trim()).reverse().join(" ")
    : name;

  const contract = p.contracts?.[0] ?? null;
  const detail = currentDetail(contract?.details ?? []);
  const capHit = parseMoney(detail?.capHit);

  // For retained salary, pull the retaining team's share from the retention object
  let retainedHit: number | null = null;
  let retainedPct: string | null = null;
  if (kind === "retained" && detail?.retention && typeof detail.retention === "object") {
    // Find the entry that is NOT the current team (they retained and traded the player away)
    // We want the retaining team's cap charge — the smaller share
    const entries = Object.values(detail.retention as Record<string, { capHit: string; retention: string }>);
    const smallest = entries.reduce((a, b) =>
      parseMoney(a.capHit)! < parseMoney(b.capHit)! ? a : b
    );
    retainedHit = parseMoney(smallest.capHit);
    retainedPct = smallest.retention;
  }

  // For buyouts, get the annual buyout cost
  let buyoutCost: number | null = null;
  if (kind === "buyout" && detail?.buyout && typeof detail.buyout === "object") {
    const bo = detail.buyout as Record<string, string>;
    buyoutCost = parseMoney(bo.cost);
  }

  const expiryYear = detail
    ? Number(String(detail.season ?? "").split("-")[1] ?? 0) + 2000
    : null;

  return {
    name: nameParts,
    position: p.pos ?? "",
    cap_hit: kind === "retained" ? retainedHit : kind === "buyout" ? buyoutCost ?? capHit : capHit,
    full_cap_hit: capHit,
    retained_pct: retainedPct,
    kind,
    expiry_year: expiryYear || null,
    note: p.acquisitionDetails ?? null,
  };
}

function parsePlayer(p: CapWagesPlayer, group: string) {
  const name = p.name ?? "";
  if (!name || name.length < 2) return null;

  // CapWages uses "Last, First" format — flip to "First Last"
  const nameParts = name.includes(",")
    ? name.split(",").map(s => s.trim()).reverse().join(" ")
    : name;

  const contract = p.contracts?.[0] ?? null;
  const detail = currentDetail(contract?.details ?? []);
  const capHit = parseMoney(detail?.capHit);
  const expiryYear = detail
    ? Number(String(detail.season ?? "").split("-")[1] ?? 0) + 2000
    : null;
  const yearsRemaining = (contract?.details?.length ?? 1) - 1;

  return {
    name: nameParts,
    position: p.pos ?? group,
    age: calcAge(p.born),
    cap_hit: capHit,
    contract_type: contractLabel(contract?.type ?? ""),
    expiry_status: contract?.expiryStatus ?? "",   // UFA / RFA
    expiry_year: expiryYear || null,
    years_remaining: yearsRemaining,
    status: p.status ?? null,
  };
}

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ team: string }> }
) {
  const { team: rawTeam } = await params;
  const team = NHL_REMAP[rawTeam.toUpperCase()] ?? rawTeam.toUpperCase();
  const slug = SLUGS[team];
  if (!slug) return NextResponse.json(EMPTY(team, "unknown_team"));

  const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";

  try {
    const res = await fetch(`https://capwages.com/teams/${slug}`, {
      headers: {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
      },
      redirect: "follow",
      cache: "no-store",
    });

    if (!res.ok) return NextResponse.json(EMPTY(team, `http_${res.status}`));

    const html = await res.text();
    const match = html.match(/<script id="__NEXT_DATA__"[^>]*>([\s\S]*?)<\/script>/);
    if (!match) return NextResponse.json(EMPTY(team, "parse_error"));

    const nd = JSON.parse(match[1]);
    const pp = nd?.props?.pageProps ?? {};
    const roster = pp?.data?.roster ?? {};
    const summary = pp?.teamSummary ?? {};

    const groups: [string, string][] = [
      ["forwards", "F"],
      ["defense",  "D"],
      ["goalies",  "G"],
    ];

    const players = groups.flatMap(([key, grp]) =>
      (roster[key] ?? [])
        .map((p: CapWagesPlayer) => parsePlayer(p, grp))
        .filter(Boolean)
    );

    // Also include inactive players (stored as a dict of groups, not a flat array)
    const inactiveRaw = pp?.data?.["inactive"];
    const inactivePlayers: CapWagesPlayer[] = Array.isArray(inactiveRaw)
      ? inactiveRaw
      : inactiveRaw && typeof inactiveRaw === "object"
        ? Object.values(inactiveRaw).flat() as CapWagesPlayer[]
        : [];
    for (const p of inactivePlayers) {
      const parsed = parsePlayer(p, "");
      if (parsed) players.push(parsed);
    }

    const cap_ceiling        = parseMoney(summary?.upperLimit) ?? 95_500_000;
    const total_hit          = parseMoney(summary?.capHit?.total);
    const cap_space          = parseMoney(summary?.capSpace);          // Current Cap Space (right now, may be negative if over cap with LTIR)
    const projected_cap_space = parseMoney(summary?.projectedCapSpace); // Projected Cap Space (end-of-season projection)
    const playoff_cap        = parseMoney(summary?.playoffCap);        // Projected Playoff Cap Space
    const ltir               = parseMoney(summary?.ltir);
    const dead_cap_total     = parseMoney(summary?.capHit?.deadCap);

    // Dead cap: retained salary, buyouts, buried penalties
    const deadCapRaw = pp?.data?.["dead cap"] ?? {};
    const dead_cap = [
      ...(deadCapRaw["retained salary"] ?? []).map((p: CapWagesPlayer) => parseDeadCapEntry(p, "retained")),
      ...(deadCapRaw["buyout history"] ?? []).map((p: CapWagesPlayer) => parseDeadCapEntry(p, "buyout")),
      ...(deadCapRaw["buried penalty"] ?? []).map((p: CapWagesPlayer) => parseDeadCapEntry(p, "buried")),
    ].filter(Boolean);

    // Draft picks — CapWages provides owned picks by year/round
    const draft_picks = (pp?.draftPicks ?? []).map((p: Record<string, unknown>) => ({
      year:           String(p.year ?? ""),
      round:          Number(p.round ?? 0),
      team:           String(p.team ?? ""),
      is_traded_away: Boolean(p.isTradedAway),
      conditions:     (p.conditions as string[]) ?? [],
    }));

    // Cap projections — upcoming free agents from this team and their projected contracts
    const projections = (pp?.projections ?? [])
      .filter((p: Record<string, unknown>) => String(p.team ?? "") === team)
      .map((p: Record<string, unknown>) => ({
        name:            String(p.name ?? "").includes(",")
                           ? String(p.name ?? "").split(",").map((s: string) => s.trim()).reverse().join(" ")
                           : String(p.name ?? ""),
        status:          String(p.status ?? ""),
        arb:             String(p.arb ?? ""),
        proj_length:     Number(p.projectedLength ?? 0),
        proj_cap_hit:    parseMoney(p.projectedCapHit),
        proj_total:      parseMoney(p.projectedTotalValue),
        proj_pct:        p.projectedCapHitPercentage != null ? Number(p.projectedCapHitPercentage) : null,
      }));

    // Reserves / depth chart — unsigned prospects (not yet on an NHL/AHL contract)
    const reserves = (pp?.reserves ?? []).map((p: Record<string, unknown>) => ({
      name:         String(p.name ?? "").includes(",")
                      ? String(p.name ?? "").split(",").map((s: string) => s.trim()).reverse().join(" ")
                      : String(p.name ?? ""),
      slug:         String(p.slug ?? ""),
      position:     String(p.pos ?? ""),
      born:         String(p.born ?? ""),
      drafted_by:   String(p.draftedBy ?? ""),
      draft_year:   Number(p.draftYear ?? 0),
      round:        Number(p.round ?? 0),
      overall:      Number(p.overall ?? 0),
      must_sign_by: String(p.mustSignBy ?? ""),
    }));

    // NHL roster with position info for depth chart formation view
    function parseRosterPlayer(p: CapWagesPlayer & Record<string, unknown>, group: string) {
      const name = String((p as Record<string, unknown>).name ?? "");
      if (!name || name.length < 2) return null;
      const nameParts = name.includes(",")
        ? name.split(",").map((s: string) => s.trim()).reverse().join(" ")
        : name;
      const contract = p.contracts?.[0] ?? null;
      const detail = currentDetail(contract?.details ?? []);
      const capHit = parseMoney(detail?.capHit);
      const posRaw = String((p as Record<string, unknown>).pos ?? group);
      return {
        name: nameParts,
        slug: String((p as Record<string, unknown>).slug ?? ""),
        position: posRaw,
        cap_hit: capHit,
        contract_type: contractLabel(contract?.type ?? ""),
        expiry_status: contract?.expiryStatus ?? "",
        status: String((p as Record<string, unknown>).status ?? ""),
      };
    }

    const rosterRaw = pp?.data?.roster ?? {};
    const nhl_roster = {
      forwards: (rosterRaw.forwards ?? []).map((p: CapWagesPlayer) => parseRosterPlayer(p as CapWagesPlayer & Record<string, unknown>, "F")).filter(Boolean),
      defense:  (rosterRaw.defense  ?? []).map((p: CapWagesPlayer) => parseRosterPlayer(p as CapWagesPlayer & Record<string, unknown>, "D")).filter(Boolean),
      goalies:  (rosterRaw.goalies  ?? []).map((p: CapWagesPlayer) => parseRosterPlayer(p as CapWagesPlayer & Record<string, unknown>, "G")).filter(Boolean),
    };

    // Non-roster = signed players currently in the minors (AHL/ECHL/loans/junior)
    const nonRosterRaw = pp?.data?.["non-roster"] ?? {};
    function parseNonRoster(p: CapWagesPlayer & Record<string, unknown>, group: string) {
      const name = String(p.name ?? "");
      if (!name || name.length < 2) return null;
      const nameParts = name.includes(",")
        ? name.split(",").map((s: string) => s.trim()).reverse().join(" ")
        : name;
      const contract = p.contracts?.[0] ?? null;
      const detail = currentDetail(contract?.details ?? []);
      const capHit = parseMoney(detail?.capHit);
      const minorsSalary = parseMoney(detail?.minorsSalary as unknown);
      return {
        name: nameParts,
        slug: String(p.slug ?? ""),
        position: String(p.pos ?? group),
        cap_hit: capHit,
        minors_salary: minorsSalary,
        contract_type: contractLabel(contract?.type ?? ""),
        expiry_status: contract?.expiryStatus ?? "",
        status: String(p.status ?? ""),           // "Minor" | "Junior" | "Loan" | "NCAA" etc.
        terms: String((p as Record<string, unknown>).terms ?? ""),
        terms_details: String((p as Record<string, unknown>).termsDetails ?? ""),
        draft_year: Number((p as Record<string, unknown>).draft_year ?? 0),
        born: String(p.born ?? ""),
      };
    }

    const non_roster = {
      forwards: ((nonRosterRaw as Record<string, CapWagesPlayer[]>).forwards ?? []).map((p) => parseNonRoster(p as CapWagesPlayer & Record<string, unknown>, "F")).filter(Boolean),
      defense:  ((nonRosterRaw as Record<string, CapWagesPlayer[]>).defense  ?? []).map((p) => parseNonRoster(p as CapWagesPlayer & Record<string, unknown>, "D")).filter(Boolean),
      goalies:  ((nonRosterRaw as Record<string, CapWagesPlayer[]>).goalies  ?? []).map((p) => parseNonRoster(p as CapWagesPlayer & Record<string, unknown>, "G")).filter(Boolean),
    };

    return NextResponse.json({ team, players, cap_ceiling, total_hit, cap_space, projected_cap_space, playoff_cap, ltir, dead_cap, dead_cap_total, draft_picks, projections, reserves, nhl_roster, non_roster, status: "ok" });
  } catch (err) {
    return NextResponse.json(EMPTY(team, `error_${String(err).slice(0, 60)}`));
  }
}
