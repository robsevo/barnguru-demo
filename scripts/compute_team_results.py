"""compute_team_results — derive per-team game results from PBP final scores.

Output: ``~/.gretzky/data/results/results_{season}.parquet`` — one row per
(team, game_id) with columns::

    team        Utf8     3-letter abbrev (NJD, NYI, ...)
    game_id     Int64
    game_date   Utf8     "YYYY-MM-DD"
    is_home     Boolean
    gf          Int64    goals for (this team)
    ga          Int64    goals against
    result      Utf8     "W" | "L" | "OTL"
    season      Int64    ending year (e.g. 2026 = 2025-26 season)

The result is derived from the final ``home_score`` / ``away_score`` rows of
each game in PBP (max per game_id). Period > 3 at any time during the game
identifies OT/SO games; the loser of an OT/SO game gets "OTL". Ties don't
exist in modern NHL.

Consumed by Phase 17 sub-signal 17.16 (team_streak) — without it that
signal returns empty and the team-side composite is missing momentum
context.

Usage::

    uv run python scripts/gretzky.py team-results
    uv run python scripts/gretzky.py team-results -- --seasons 2025 --force
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from data.data_store import DataStore
from data.schedule_sync import latest_schedule_parquet
from models.rapm_model import _NHL_TEAM_IDS


DEFAULT_DATA_DIR = Path(
    os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data"))
)
RESULTS_SUBDIR = "results"

RESULTS_SCHEMA: dict[str, pl.DataType] = {
    "team":      pl.Utf8,
    "game_id":   pl.Int64,
    "game_date": pl.Utf8,
    "is_home":   pl.Boolean,
    "gf":        pl.Int64,
    "ga":        pl.Int64,
    "result":    pl.Utf8,
    "season":    pl.Int64,
}


def _season_ending_year(d: str) -> int:
    """Aug-Dec → year+1 (the spring at end of season); Jan-Jul → current year."""
    dt = datetime.strptime(d, "%Y-%m-%d").date()
    return dt.year + 1 if dt.month >= 8 else dt.year


def _derive_results(pbp: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """One row per (team, game_id) with W/L/OTL."""
    if len(pbp) == 0:
        return pl.DataFrame(schema=RESULTS_SCHEMA)

    needed = {"game_id", "home_score", "away_score", "period", "home_team_id", "away_team_id"}
    missing = needed - set(pbp.columns)
    if missing:
        print(f"[team-results] PBP missing cols: {sorted(missing)}; skipping.")
        return pl.DataFrame(schema=RESULTS_SCHEMA)

    # Final score + whether OT/SO happened.
    finals = (
        pbp.group_by("game_id")
           .agg([
               pl.col("home_score").max().alias("home_final"),
               pl.col("away_score").max().alias("away_final"),
               pl.col("home_team_id").first().alias("home_team_id"),
               pl.col("away_team_id").first().alias("away_team_id"),
               pl.col("period").max().alias("max_period"),
           ])
           .filter(
               pl.col("home_final").is_not_null()
               & pl.col("away_final").is_not_null()
               & pl.col("home_team_id").is_not_null()
               & pl.col("away_team_id").is_not_null()
           )
    )
    if len(finals) == 0:
        return pl.DataFrame(schema=RESULTS_SCHEMA)

    # Join in game_date from schedule
    if "game_date" in schedule.columns:
        finals = finals.join(
            schedule.select(["game_id", "game_date"]),
            on="game_id", how="left",
        )
    else:
        finals = finals.with_columns(pl.lit("").alias("game_date"))

    finals = finals.with_columns(
        pl.col("game_date").map_elements(
            lambda d: _season_ending_year(d) if d else 0,
            return_dtype=pl.Int64,
        ).alias("season")
    )

    # team_id → abbrev mapping (matches rapm_model._NHL_TEAM_IDS)
    abbrev_map = pl.DataFrame(
        {"team_id":   list(_NHL_TEAM_IDS.keys()),
         "team_abbrev": list(_NHL_TEAM_IDS.values())},
        schema={"team_id": pl.Int64, "team_abbrev": pl.Utf8},
    )

    # Build home + away rows separately, then concat.
    rows = finals.to_dicts()
    out: list[dict] = []
    id_to_abbrev = dict(_NHL_TEAM_IDS)
    for r in rows:
        gid       = int(r["game_id"])
        gd        = r.get("game_date") or ""
        season    = int(r.get("season") or 0)
        home_id   = int(r["home_team_id"])
        away_id   = int(r["away_team_id"])
        hf        = int(r["home_final"])
        af        = int(r["away_final"])
        ot_or_so  = (r.get("max_period") or 0) > 3
        home_abbrev = id_to_abbrev.get(home_id)
        away_abbrev = id_to_abbrev.get(away_id)
        if not home_abbrev or not away_abbrev:
            continue
        # Home row
        if hf > af:
            home_result, away_result = "W", "OTL" if ot_or_so else "L"
        elif af > hf:
            home_result, away_result = "OTL" if ot_or_so else "L", "W"
        else:
            # Ties don't exist post-lockout; skip if encountered
            continue
        out.append({
            "team":      home_abbrev,
            "game_id":   gid,
            "game_date": gd,
            "is_home":   True,
            "gf":        hf,
            "ga":        af,
            "result":    home_result,
            "season":    season,
        })
        out.append({
            "team":      away_abbrev,
            "game_id":   gid,
            "game_date": gd,
            "is_home":   False,
            "gf":        af,
            "ga":        hf,
            "result":    away_result,
            "season":    season,
        })

    if not out:
        return pl.DataFrame(schema=RESULTS_SCHEMA)
    return pl.DataFrame(out, schema=RESULTS_SCHEMA)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive per-team-game W/L/OTL results from PBP."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seasons", type=int, nargs="+", default=None,
                        help="Season ending years to write separately (e.g. 2025 2026). "
                             "Default: every season present in PBP.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = args.data_dir / RESULTS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    sched_path = latest_schedule_parquet(args.data_dir / "schedule")
    if sched_path is None:
        print(f"[team-results] No schedule parquet in {args.data_dir / 'schedule'}.")
        sys.exit(1)
    schedule = pl.read_parquet(sched_path)

    with DataStore(args.data_dir) as store:
        pbp = store.pbp()
    if len(pbp) == 0:
        print("[team-results] DataStore has no PBP. Run `ingest` first.")
        sys.exit(1)

    print(f"[team-results] Deriving results from {len(pbp):,} PBP rows…")
    results = _derive_results(pbp, schedule)
    if len(results) == 0:
        print("[team-results] No results derived. Empty output.")
        sys.exit(0)

    # Split per-season and write
    seasons = args.seasons or sorted(results["season"].unique().drop_nulls().to_list())
    for season in seasons:
        sub = results.filter(pl.col("season") == int(season))
        if len(sub) == 0:
            continue
        out_path = out_dir / f"results_{season}.parquet"
        if out_path.exists() and not args.force:
            print(f"[team-results] {out_path.name} exists; use --force to overwrite.")
            continue
        sub.write_parquet(out_path)
        n_games = sub.select("game_id").n_unique()
        n_teams = sub.select("team").n_unique()
        print(f"  Season {season}: {n_games} games × {n_teams} teams → {out_path.name}")

    # Latest-season sanity print
    if seasons:
        latest = max(seasons)
        sub = results.filter(pl.col("season") == int(latest))
        if len(sub) > 0:
            print(f"\n[team-results] Latest-season streak preview ({latest}):")
            streaks = (
                sub.sort(["team", "game_date"])
                   .group_by("team")
                   .tail(8)
                   .group_by("team")
                   .agg(pl.col("result").str.concat("").alias("last8"))
                   .sort("team")
            )
            for r in streaks.head(8).to_dicts():
                print(f"  {r['team']}: {r['last8']}")


if __name__ == "__main__":
    main()
