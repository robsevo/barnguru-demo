"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { HudPanel, HudBadge } from "@/components/hud";

/**
 * Player index.
 *
 * This page replaced a 2,177-line version that fired twenty-one requests on
 * mount — a separate leaderboard endpoint per metric, each with its own loading
 * state, stacked into one screen. It answered a question nobody asked ("show me
 * everything at once") and took several seconds to answer it.
 *
 * Two requests now: the season table, and whichever model leaderboard is
 * selected. Sorting is client-side because 768 rows is nothing, and a sort that
 * round-trips to a server feels broken.
 */

interface PlayerRow {
  player_id: number;
  name: string;
  team: string;
  position: string;
  number: number;
  gp: number;
  goals: number;
  assists: number;
  points: number;
  plusMinus: number;
  shots: number;
  shootingPctg: number;
  toiPerGame: number;
  rapm_ev_off: number | null;
  rapm_ev_def: number | null;
  war: number | null;
  gar_total: number | null;
}

interface LeaderRow {
  player_id: number;
  player_name: string;
  team: string;
  [k: string]: unknown;
}

/** Model leaderboards. `ascending` mirrors the API: defensive RAPM is measured
 *  as expected goals ALLOWED, so a good defender's coefficient is negative. */
const METRICS = [
  { key: "war",         label: "WAR",         hint: "Wins above replacement" },
  { key: "gar_total",   label: "GAR",         hint: "Goals above replacement" },
  { key: "rapm_ev_off", label: "RAPM · EV Off", hint: "Even-strength offence, goals/60" },
  { key: "rapm_ev_def", label: "RAPM · EV Def", hint: "Even-strength defence, goals/60 allowed" },
  { key: "rapm_pp",     label: "RAPM · PP",   hint: "Power play, goals/60" },
  { key: "rapm_pk",     label: "RAPM · PK",   hint: "Penalty kill, goals/60 allowed" },
] as const;

type SortKey = "points" | "goals" | "assists" | "plusMinus" | "toiPerGame" | "war" | "rapm_ev_off";

const COLUMNS: { key: SortKey; label: string; fmt: (r: PlayerRow) => string }[] = [
  { key: "points",      label: "P",    fmt: r => String(r.points) },
  { key: "goals",       label: "G",    fmt: r => String(r.goals) },
  { key: "assists",     label: "A",    fmt: r => String(r.assists) },
  { key: "plusMinus",   label: "+/-",  fmt: r => (r.plusMinus > 0 ? `+${r.plusMinus}` : String(r.plusMinus)) },
  { key: "toiPerGame",  label: "TOI",  fmt: r => r.toiPerGame.toFixed(1) },
  { key: "war",         label: "WAR",  fmt: r => (r.war == null ? "—" : r.war.toFixed(2)) },
  { key: "rapm_ev_off", label: "EVO",  fmt: r => (r.rapm_ev_off == null ? "—" : r.rapm_ev_off.toFixed(2)) },
];

function TeamChip({ abbrev }: { abbrev: string }) {
  return (
    <span
      className="inline-flex items-center justify-center w-9 h-5 rounded text-[9px] font-black tracking-wider
                 text-[var(--brand-hex)] border border-[var(--border-default)] bg-white/[0.03]"
    >
      {abbrev}
    </span>
  );
}

export default function PlayersPage() {
  const [rows, setRows] = useState<PlayerRow[]>([]);
  const [leaders, setLeaders] = useState<LeaderRow[]>([]);
  const [metric, setMetric] = useState<string>(METRICS[0].key);
  const [metricBuilt, setMetricBuilt] = useState(true);
  const [sort, setSort] = useState<SortKey>("points");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetch("/api/players?limit=1000")
      .then(r => (r.ok ? r.json() : Promise.reject(new Error(`players ${r.status}`))))
      .then(d => { if (alive) { setRows(d.players ?? []); setLoading(false); } })
      .catch(e => { if (alive) { setError(String(e.message ?? e)); setLoading(false); } });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    let alive = true;
    fetch(`/api/leaders/${metric}?limit=10`)
      .then(r => (r.ok ? r.json() : Promise.reject(new Error(`leaders ${r.status}`))))
      .then(d => { if (alive) { setLeaders(d.players ?? []); setMetricBuilt(d.built !== false); } })
      .catch(() => { if (alive) { setLeaders([]); setMetricBuilt(false); } });
    return () => { alive = false; };
  }, [metric]);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = q
      ? rows.filter(r => r.name.toLowerCase().includes(q) || r.team.toLowerCase().includes(q))
      : rows;
    // Nulls last regardless of direction — a player the model has no rating for
    // is not the best or the worst, he is absent, and sorting him to the top of
    // a WAR table would be a lie.
    return [...filtered].sort((a, b) => {
      const av = a[sort], bv = b[sort];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      return (bv as number) - (av as number);
    });
  }, [rows, query, sort]);

  const active = METRICS.find(m => m.key === metric)!;

  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-6 flex flex-col gap-5">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-[26px] font-black tracking-[0.08em] uppercase text-white leading-none">
            Players
          </h1>
          <p className="mt-1 text-[11px] text-[var(--text-secondary)]">
            {loading ? "Loading…" : `${rows.length} skaters · season stats joined to model output`}
          </p>
        </div>
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Search player or team…"
          className="w-full sm:w-64 px-3 py-1.5 rounded-lg text-[12px] text-white placeholder:text-white/25
                     bg-white/[0.04] border border-[var(--border-dim)] outline-none
                     focus:border-[var(--border-bright)] transition-colors"
        />
      </header>

      {error && (
        <HudPanel title="API unavailable">
          <p className="text-[12px] text-[var(--text-secondary)] leading-relaxed">
            {error}. Start the API with{" "}
            <code className="text-[var(--brand-hex)]">uv run uvicorn dashboard.api.main:app --port 8000</code>
            {" "}and generate data with{" "}
            <code className="text-[var(--brand-hex)]">uv run python scripts/make_demo_data.py</code>.
          </p>
        </HudPanel>
      )}

      {/* Model leaderboard — one metric at a time, chosen deliberately. */}
      <HudPanel
        title="Model leaderboard"
        subtitle={active.hint}
        right={
          <div className="flex flex-wrap gap-1 justify-end">
            {METRICS.map(m => (
              <button
                key={m.key}
                onClick={() => setMetric(m.key)}
                className={`px-2 py-1 rounded text-[10px] font-semibold tracking-wide transition-colors border
                  ${m.key === metric
                    ? "text-[var(--brand-hex)] border-[var(--border-bright)] bg-[var(--brand-hex)]/[0.10]"
                    : "text-white/40 border-transparent hover:text-white/70"}`}
              >
                {m.label}
              </button>
            ))}
          </div>
        }
      >
        {!metricBuilt ? (
          <p className="text-[12px] text-[var(--text-secondary)]">
            This model has not been run. Generate it with{" "}
            <code className="text-[var(--brand-hex)]">uv run python scripts/make_demo_data.py</code>.
          </p>
        ) : (
          <ol className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1">
            {leaders.map((p, i) => {
              const v = p[metric];
              return (
                <li key={p.player_id} className="flex items-center gap-2.5 py-1 border-b border-white/[0.04]">
                  <span className="w-5 text-[10px] font-mono text-white/25 text-right">{i + 1}</span>
                  <TeamChip abbrev={p.team} />
                  <Link
                    href={`/players/${p.player_id}`}
                    className="flex-1 truncate text-[12px] text-white/85 hover:text-[var(--brand-hex)] transition-colors"
                  >
                    {p.player_name}
                  </Link>
                  <span className="text-[12px] font-mono tabular-nums text-[var(--brand-hex)]">
                    {typeof v === "number" ? v.toFixed(2) : "—"}
                  </span>
                </li>
              );
            })}
          </ol>
        )}
      </HudPanel>

      {/* Season table */}
      <HudPanel title="Season" padded={false}>
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-[9px] uppercase tracking-[0.18em] text-white/30">
                <th className="text-left font-semibold px-3 py-2">Player</th>
                <th className="text-left font-semibold px-2 py-2">Team</th>
                <th className="text-right font-semibold px-2 py-2">GP</th>
                {COLUMNS.map(c => (
                  <th key={c.key} className="text-right font-semibold px-2 py-2">
                    <button
                      onClick={() => setSort(c.key)}
                      className={`transition-colors ${sort === c.key ? "text-[var(--brand-hex)]" : "hover:text-white/60"}`}
                    >
                      {c.label}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {visible.slice(0, 100).map(r => (
                <tr key={r.player_id} className="border-t border-white/[0.04] hover:bg-white/[0.02]">
                  <td className="px-3 py-1.5">
                    <Link
                      href={`/players/${r.player_id}`}
                      className="text-white/85 hover:text-[var(--brand-hex)] transition-colors"
                    >
                      {r.name}
                    </Link>
                    <HudBadge className="ml-2" tone="neutral">{r.position}</HudBadge>
                  </td>
                  <td className="px-2 py-1.5"><TeamChip abbrev={r.team} /></td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-white/45">{r.gp}</td>
                  {COLUMNS.map(c => (
                    <td
                      key={c.key}
                      className={`px-2 py-1.5 text-right font-mono tabular-nums ${
                        sort === c.key ? "text-[var(--brand-hex)]" : "text-white/70"
                      }`}
                    >
                      {c.fmt(r)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {!loading && visible.length === 0 && (
            <p className="px-3 py-6 text-[12px] text-[var(--text-secondary)]">
              No players match &ldquo;{query}&rdquo;.
            </p>
          )}
        </div>
      </HudPanel>
    </main>
  );
}
