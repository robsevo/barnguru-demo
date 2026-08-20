"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { HudPanel, HudBadge, RingGauge } from "@/components/hud";

/**
 * Player profile.
 *
 * This replaced an 8,941-line version. The old page rendered every panel the
 * project had ever built — shot maps, neural-net readouts, fatigue curves,
 * contract tables, injury history — behind a dozen requests, most of which
 * returned nothing for most players. What survives is what the models in this
 * repository actually produce: a season line, a RAPM breakdown, the WAR it
 * converts to, and how both moved across seasons.
 *
 * One request. Everything below comes from `/api/players/{id}`.
 */

interface SeasonLine {
  gp: number; goals: number; assists: number; points: number;
  plusMinus: number; penaltyMinutes: number; powerPlayGoals: number;
  shots: number; shootingPctg: number; toiPerGame: number;
}

interface HistoryRow {
  season: number;
  gp?: number;
  toi_ev?: number; toi_pp?: number; toi_pk?: number;
  rapm_ev_off?: number; rapm_ev_def?: number;
  rapm_pp?: number; rapm_pk?: number;
  xgf_60?: number; xga_60?: number;
  war?: number; gar_total?: number;
}

interface Profile {
  player: { player_id: number; name: string; team: string; position: string; jersey: number; production_rating?: number };
  season: number;
  stats: SeasonLine | null;
  history: HistoryRow[];
  team: { abbrev: string; name: string; conference: string; division: string } | null;
}

/**
 * Map a RAPM coefficient onto the 0..1 a gauge wants.
 *
 * RAPM is goals per 60 minutes and sits roughly within ±1 for real players, so
 * ±1 is the full sweep. `invert` handles the defensive metrics, which are
 * measured as goals ALLOWED — a good defender's coefficient is negative, and a
 * gauge that filled to the right on a bad one would be actively misleading.
 */
function gaugeValue(v: number | undefined | null, invert = false): number {
  if (v == null) return 0;
  const signed = invert ? -v : v;
  return Math.max(0, Math.min(1, (signed + 1) / 2));
}

function fmt(v: number | undefined | null, digits = 2): string {
  return v == null ? "—" : v.toFixed(digits);
}

const RAPM_GAUGES = [
  { key: "rapm_ev_off", label: "EV Off", invert: false },
  { key: "rapm_ev_def", label: "EV Def", invert: true },
  { key: "rapm_pp",     label: "PP",     invert: false },
  { key: "rapm_pk",     label: "PK",     invert: true },
] as const;

const SEASON_CELLS: { label: string; get: (s: SeasonLine) => string }[] = [
  { label: "GP",   get: s => String(s.gp) },
  { label: "G",    get: s => String(s.goals) },
  { label: "A",    get: s => String(s.assists) },
  { label: "P",    get: s => String(s.points) },
  { label: "+/-",  get: s => (s.plusMinus > 0 ? `+${s.plusMinus}` : String(s.plusMinus)) },
  { label: "PIM",  get: s => String(s.penaltyMinutes) },
  { label: "PPG",  get: s => String(s.powerPlayGoals) },
  { label: "SOG",  get: s => String(s.shots) },
  { label: "S%",   get: s => `${(s.shootingPctg * 100).toFixed(1)}%` },
  { label: "TOI",  get: s => `${s.toiPerGame.toFixed(1)}` },
];

export default function PlayerProfilePage() {
  const { id } = useParams<{ id: string }>();
  const [data, setData] = useState<Profile | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetch(`/api/players/${id}`)
      .then(r => (r.ok ? r.json() : Promise.reject(new Error(`profile ${r.status}`))))
      .then(d => { if (alive) setData(d); })
      .catch(e => { if (alive) setError(String(e.message ?? e)); });
    return () => { alive = false; };
  }, [id]);

  if (error) {
    return (
      <main className="mx-auto w-full max-w-5xl px-4 py-8">
        <HudPanel title="Not available">
          <p className="text-[12px] text-[var(--text-secondary)]">{error}</p>
          <Link href="/players" className="mt-3 inline-block text-[12px] text-[var(--brand-hex)]">
            ← All players
          </Link>
        </HudPanel>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="mx-auto w-full max-w-5xl px-4 py-8">
        <p className="text-[12px] text-[var(--text-secondary)]">Loading…</p>
      </main>
    );
  }

  const { player, stats, history, team } = data;
  const current = history.find(h => h.season === data.season) ?? history[history.length - 1];

  return (
    <main className="mx-auto w-full max-w-5xl px-4 py-6 flex flex-col gap-4">
      <Link href="/players" className="text-[11px] text-white/35 hover:text-white/70 transition-colors">
        ← All players
      </Link>

      {/* 1 — identity */}
      <HudPanel scanline>
        <div className="flex flex-wrap items-center gap-4">
          <div
            className="flex items-center justify-center w-14 h-14 rounded-xl border border-[var(--border-default)]
                       bg-white/[0.03] text-[var(--brand-hex)] text-[20px] font-black tabular-nums"
          >
            {player.jersey}
          </div>
          <div className="min-w-0">
            <h1 className="text-[26px] font-black tracking-[0.05em] uppercase text-white leading-none truncate">
              {player.name}
            </h1>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <HudBadge tone="accent">{player.position}</HudBadge>
              {team && (
                <Link
                  href={`/teams/${team.abbrev}`}
                  className="text-[12px] text-white/60 hover:text-[var(--brand-hex)] transition-colors"
                >
                  {team.name}
                </Link>
              )}
              {team && (
                <span className="text-[10px] uppercase tracking-[0.18em] text-white/25">
                  {team.conference} · {team.division}
                </span>
              )}
            </div>
          </div>
        </div>
      </HudPanel>

      {/* 2 — season line */}
      <HudPanel title="Season" subtitle={String(data.season)}>
        {stats ? (
          <div className="grid grid-cols-5 sm:grid-cols-10 gap-y-3">
            {SEASON_CELLS.map(c => (
              <div key={c.label} className="text-center">
                <div className="text-[9px] uppercase tracking-[0.16em] text-white/30">{c.label}</div>
                <div className="mt-0.5 text-[15px] font-mono tabular-nums text-white/90">{c.get(stats)}</div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-[12px] text-[var(--text-secondary)]">No season line for this player.</p>
        )}
      </HudPanel>

      {/* 3 — RAPM */}
      <HudPanel
        title="RAPM"
        subtitle="Regularized adjusted plus-minus · goals per 60, prior-informed ridge"
      >
        {current ? (
          <div className="flex flex-wrap gap-6 justify-around">
            {RAPM_GAUGES.map(g => {
              const raw = current[g.key];
              return (
                <RingGauge
                  key={g.key}
                  value={gaugeValue(raw, g.invert)}
                  label={g.label}
                  centerText={fmt(raw)}
                  sublabel={g.invert ? "lower ↓" : "goals/60"}
                  size={92}
                />
              );
            })}
          </div>
        ) : (
          <p className="text-[12px] text-[var(--text-secondary)]">
            No model output for this player — run{" "}
            <code className="text-[var(--brand-hex)]">scripts/make_demo_data.py</code>.
          </p>
        )}
        {current && (
          <div className="mt-4 pt-3 border-t border-white/[0.05] grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
            {[
              { l: "xGF/60", v: fmt(current.xgf_60) },
              { l: "xGA/60", v: fmt(current.xga_60) },
              { l: "EV TOI", v: current.toi_ev ? `${Math.round(current.toi_ev / 60)}m` : "—" },
              { l: "PP TOI", v: current.toi_pp ? `${Math.round(current.toi_pp / 60)}m` : "—" },
            ].map(x => (
              <div key={x.l}>
                <div className="text-[9px] uppercase tracking-[0.16em] text-white/30">{x.l}</div>
                <div className="mt-0.5 text-[13px] font-mono tabular-nums text-white/80">{x.v}</div>
              </div>
            ))}
          </div>
        )}
      </HudPanel>

      {/* 4 — WAR / GAR */}
      <HudPanel title="Value" subtitle="Goals above replacement, converted to wins">
        {current?.war != null ? (
          <div className="flex flex-wrap items-center gap-8">
            <div>
              <div className="text-[9px] uppercase tracking-[0.16em] text-white/30">WAR</div>
              <div className="text-[34px] font-black tabular-nums leading-none text-[var(--brand-hex)]">
                {current.war > 0 ? "+" : ""}{current.war.toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-[9px] uppercase tracking-[0.16em] text-white/30">GAR</div>
              <div className="text-[22px] font-mono tabular-nums text-white/80">
                {fmt(current.gar_total, 1)}
              </div>
            </div>
            <p className="flex-1 min-w-[220px] text-[11px] text-[var(--text-secondary)] leading-relaxed">
              GAR converts each RAPM component into goals over the ice time actually
              played, against a replacement level measured from this season&rsquo;s
              player pool. WAR divides that by goals-per-win.
            </p>
          </div>
        ) : (
          <p className="text-[12px] text-[var(--text-secondary)]">No value rating for this player.</p>
        )}
      </HudPanel>

      {/* 5 — history */}
      <HudPanel title="By season" padded={false}>
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-[9px] uppercase tracking-[0.18em] text-white/30">
                {["Season", "GP", "EV Off", "EV Def", "PP", "PK", "GAR", "WAR"].map(h => (
                  <th key={h} className={`font-semibold px-3 py-2 ${h === "Season" ? "text-left" : "text-right"}`}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {history.map(h => (
                <tr key={h.season} className="border-t border-white/[0.04]">
                  <td className="px-3 py-1.5 text-white/85 font-semibold">{h.season}</td>
                  <td className="px-3 py-1.5 text-right font-mono tabular-nums text-white/45">{h.gp ?? "—"}</td>
                  <td className="px-3 py-1.5 text-right font-mono tabular-nums text-white/75">{fmt(h.rapm_ev_off)}</td>
                  <td className="px-3 py-1.5 text-right font-mono tabular-nums text-white/75">{fmt(h.rapm_ev_def)}</td>
                  <td className="px-3 py-1.5 text-right font-mono tabular-nums text-white/75">{fmt(h.rapm_pp)}</td>
                  <td className="px-3 py-1.5 text-right font-mono tabular-nums text-white/75">{fmt(h.rapm_pk)}</td>
                  <td className="px-3 py-1.5 text-right font-mono tabular-nums text-white/75">{fmt(h.gar_total, 1)}</td>
                  <td className="px-3 py-1.5 text-right font-mono tabular-nums text-[var(--brand-hex)]">{fmt(h.war)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {history.length === 0 && (
            <p className="px-3 py-6 text-[12px] text-[var(--text-secondary)]">No model history.</p>
          )}
        </div>
      </HudPanel>

      {/* The estimator is fitted on SIMULATED play anchored to this player's real
          production rate. Showing both is the honest way to present a RAPM figure
          that is a demonstration rather than a measurement. */}
      {player.production_rating != null && current?.rapm_ev_off != null && (
        <HudPanel title="Model check" subtitle="RAPM here is fitted on simulated shifts — a demo of the estimator, not a rating of this player">
          <div className="flex flex-wrap items-center gap-8">
            <div>
              <div className="text-[9px] uppercase tracking-[0.16em] text-white/30">Production rating</div>
              <div className="text-[18px] font-mono tabular-nums text-white/80">{fmt(player.production_rating)}</div>
            </div>
            <div>
              <div className="text-[9px] uppercase tracking-[0.16em] text-white/30">RAPM EV Off</div>
              <div className="text-[18px] font-mono tabular-nums text-[var(--brand-hex)]">{fmt(current.rapm_ev_off)}</div>
            </div>
            <p className="flex-1 min-w-[240px] text-[11px] text-[var(--text-secondary)] leading-relaxed">
              Left: this player&rsquo;s real points-per-60, standardised across the
              league — the value the simulator drew his shot outcomes from. Right:
              what RAPM produced seeing only shifts and shots. Across 811 skaters
              the two correlate at about r = 0.45.
            </p>
          </div>
        </HudPanel>
      )}
    </main>
  );
}
