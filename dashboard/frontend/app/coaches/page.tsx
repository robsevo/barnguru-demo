"use client";

import { useEffect, useState } from "react";
import { logoUrl, TEAM_FULL_NAMES } from "@/utils/nhl";
import { HudGrid } from "@/components/hud";

interface CoachEntry {
  name: string;
  team: string;
  first_named_head_coach?: string | null;
  image_url?: string | null;
}

interface CoachProfileSummary {
  gp_under_coach: number;
  points_pct: number;
  wins: number;
  losses: number;
  ot_losses: number;
}

interface VenueSummary {
  scare_factor: number;
  scare_rank: number;
}

interface CoachCard {
  coach: CoachEntry;
  profile?: CoachProfileSummary | null;
  venue?: VenueSummary | null;
}

function TeamLogo({ team, size = 24 }: { team: string; size?: number }) {
  const [err, setErr] = useState(false);
  if (!team || err) return <span className="text-[9px] font-mono text-white/40">{team}</span>;
  return (
    <img src={logoUrl(team)} alt={TEAM_FULL_NAMES[team] ?? team} width={size} height={size}
      onError={() => setErr(true)} className="object-contain shrink-0" />
  );
}

export default function CoachesPage() {
  const [cards, setCards] = useState<CoachCard[]>([]);
  const [loading, setLoading] = useState(true);
  const [sort, setSort] = useState<"pts" | "team" | "gp">("pts");
  const today = new Date().toISOString().slice(0, 10);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/api/coaches");
        const data = await res.json();
        const coaches: CoachEntry[] = (data.coaches ?? []).map((c: any) => ({
          name: c.name, team: c.team,
          first_named_head_coach: c.first_named_head_coach,
          image_url: c.image_url ?? null,
        }));

        const results: CoachCard[] = [];
        await Promise.all(
          coaches.map(async (coach) => {
            try {
              const pr = await fetch(`/api/coaches/${encodeURIComponent(coach.name)}`);
              const p = await pr.json();
              results.push({
                coach,
                profile: p.coach_profile?.row ?? null,
                venue: p.venue_atmosphere?.row ?? null,
              });
            } catch {
              results.push({ coach });
            }
          })
        );
        setCards(results);
      } catch { /* empty */ }
      setLoading(false);
    })();
  }, []);

  const sorted = [...cards].sort((a, b) => {
    if (sort === "pts") return (b.profile?.points_pct ?? 0) - (a.profile?.points_pct ?? 0);
    if (sort === "gp") return (b.profile?.gp_under_coach ?? 0) - (a.profile?.gp_under_coach ?? 0);
    return a.coach.team.localeCompare(b.coach.team);
  });

  return (
    <main className="relative min-h-screen p-3 sm:p-6">
      <HudGrid />

      <div className="relative z-10 mb-3 flex items-center gap-2 flex-wrap">
        <span className="hud-mono text-[10px] uppercase tracking-[0.20em] text-[#fb923c]" aria-hidden>&#9698;</span>
        <span className="hud-mono text-[10px] uppercase tracking-[0.20em] text-[#fb923c]">
          COACH DIRECTORY
        </span>
        <span className="hud-mono text-[9px] uppercase tracking-[0.16em] text-[var(--text-secondary)]">
          &middot; 32 head coaches
        </span>
        <span className="ml-auto text-[#777] text-xs font-mono">{today}</span>
      </div>

      <div className="relative z-10 mb-4 flex items-center gap-2">
        <span className="text-[9px] font-mono text-white/30 uppercase tracking-wider">Sort:</span>
        {(["pts", "team", "gp"] as const).map(s => (
          <button key={s} onClick={() => setSort(s)}
            className={`px-2 py-0.5 rounded text-[9px] font-mono uppercase ${
              sort === s ? "bg-white/[0.10] text-white" : "text-white/40 hover:text-white/70"
            }`}>
            {s === "pts" ? "Points %" : s === "gp" ? "GP" : "Team"}
          </button>
        ))}
      </div>

      {loading ? (
        <p className="text-[10px] font-semibold uppercase tracking-widest text-[#777] animate-pulse relative z-10">LOADING...</p>
      ) : (
        <div className="relative z-10 grid gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {sorted.map(({ coach, profile, venue }) => (
            <a
              key={coach.team}
              href={`/coaches/${encodeURIComponent(coach.name)}`}
              className="hud-panel hud-panel--all-corners jarvis-shimmer p-3 flex flex-col gap-2 hover:bg-white/[0.04] transition-all group"
            >
              <div className="flex items-center gap-3">
                {/* Coach headshot with small team logo badge */}
                <div className="relative shrink-0">
                  <div className="w-12 h-12 rounded-full overflow-hidden border-2 flex items-center justify-center bg-white/[0.04]"
                    style={{ borderColor: "rgba(251,146,60,0.40)" }}>
                    {coach.image_url ? (
                      /* eslint-disable-next-line @next/next/no-img-element */
                      <img src={coach.image_url} alt={coach.name}
                        className="w-full h-full object-cover object-top scale-110 origin-top"
                        onError={(e) => {
                          const t = e.currentTarget as HTMLImageElement;
                          t.style.display = "none";
                          const fallback = t.nextElementSibling as HTMLElement;
                          if (fallback) fallback.style.display = "flex";
                        }} />
                    ) : null}
                    <span className="text-[9px] font-bold text-[#fb923c] tracking-wider"
                      style={{ display: coach.image_url ? "none" : "flex" }}>HC</span>
                  </div>
                  <div className="absolute -bottom-1 -right-1 w-5 h-5 rounded-full bg-black/80 flex items-center justify-center border border-white/[0.10]">
                    <TeamLogo team={coach.team} size={14} />
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-[12px] font-semibold text-white group-hover:text-[#fb923c] transition-colors truncate">
                    {coach.name}
                  </div>
                  <div className="text-[9px] font-mono text-white/35">
                    <span className="text-[#fb923c] uppercase tracking-wider text-[8px]">HC</span>
                    <span className="mx-1">·</span>
                    {TEAM_FULL_NAMES[coach.team] ?? coach.team}
                  </div>
                </div>
              </div>

              {profile ? (
                <div className="grid grid-cols-3 gap-1.5">
                  <div className="rounded border border-white/[0.06] bg-white/[0.02] px-2 py-1">
                    <span className="text-[7px] uppercase tracking-wider text-white/25 block">P%</span>
                    <span className={`text-[12px] font-mono font-semibold ${
                      profile.points_pct >= 0.55 ? "text-[#4ade80]" : profile.points_pct >= 0.45 ? "text-white" : "text-[#f87171]"
                    }`}>
                      {(profile.points_pct * 100).toFixed(1)}
                    </span>
                  </div>
                  <div className="rounded border border-white/[0.06] bg-white/[0.02] px-2 py-1">
                    <span className="text-[7px] uppercase tracking-wider text-white/25 block">Record</span>
                    <span className="text-[11px] font-mono text-white/70">{profile.wins}-{profile.losses}-{profile.ot_losses}</span>
                  </div>
                  <div className="rounded border border-white/[0.06] bg-white/[0.02] px-2 py-1">
                    <span className="text-[7px] uppercase tracking-wider text-white/25 block">GP</span>
                    <span className="text-[12px] font-mono text-white/70">{profile.gp_under_coach}</span>
                  </div>
                </div>
              ) : (
                <div className="text-[9px] font-mono text-white/25">No profile data</div>
              )}

              {venue && (
                <div className="flex items-center gap-2 text-[8px] font-mono text-white/25">
                  <span>Home Scare: <span className={venue.scare_factor > 0.3 ? "text-[#f87171]/70" : "text-white/40"}>
                    {venue.scare_factor >= 0 ? "+" : ""}{venue.scare_factor.toFixed(2)}
                  </span></span>
                  <span className="text-white/10">·</span>
                  <span>rank {(venue.scare_rank * 100).toFixed(0)}th</span>
                </div>
              )}
            </a>
          ))}
        </div>
      )}
    </main>
  );
}
