"""compute_playoff_fatigue — Feature 3.23 driver script.

Derives the per-player-per-playoff-game fatigue inputs from the schedule
parquet (game_type=3 games for the current spring) and PBP appearances, then
writes the playoff-fatigue parquet consumed by composite_fi (3.17).

Inputs needed in DataStore:
- ``schedule_*.parquet`` containing ``game_id``, ``game_date``, ``game_type``,
  ``home_team``, ``away_team`` (built by ``data.schedule_sync``).
- ``store.pbp()`` for distinct (player_id, game_id) playoff appearances.
- Optional: ``travel_distance/*.parquet`` and ``time_zone_crossing/*.parquet``
  computed with ``--game-types 3`` (joined per team-game when present).

The current "spring" is inferred from ``--date`` (or today):
- Aug..Dec → upcoming playoffs in spring (year+1)
- Jan..Jul → playoffs of current year

Usage::

    uv run python scripts/gretzky.py playoff-fatigue
    uv run python scripts/gretzky.py playoff-fatigue -- --date 2026-05-17
    uv run python scripts/gretzky.py playoff-fatigue -- --force
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
from models.playoff_fatigue import (
    PlayoffFatigueModel,
    write_playoff_fatigue,
)


DEFAULT_DATA_DIR = Path(
    os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data"))
)
PLAYOFF_FATIGUE_SUBDIR = "playoff_fatigue"
SCHEDULE_SUBDIR        = "schedule"
TRAVEL_SUBDIR          = "travel_distance"
TZ_SUBDIR              = "time_zone_crossing"

GAME_TYPE_PLAYOFFS = 3


# Same union-of-id-columns trick as compute_prior_playoff_load — keeps modern
# and legacy PBP schemas working without a separate code path.
_PLAYER_ID_COLS: tuple[str, ...] = (
    "event_player_1_id",
    "event_player_2_id",
    "shooter_id",
    "shot_goalie_id",
    "scorer_id",
    "assist1_id",
    "assist2_id",
    "committed_by_id",
    "drawn_by_id",
    "winning_player_id",
    "losing_player_id",
    "turnover_player_id",
    "hitter_id",
    "hittee_id",
    "blocker_id",
)


def _spring_year(d: str) -> int:
    """Map YYYY-MM-DD to the spring (playoff) year."""
    dt = datetime.strptime(d, "%Y-%m-%d").date()
    return dt.year + 1 if dt.month >= 8 else dt.year


def _per_player_per_game(pbp: pl.DataFrame) -> pl.DataFrame:
    pid_cols = [c for c in _PLAYER_ID_COLS if c in pbp.columns]
    if not pid_cols:
        return pl.DataFrame(schema={"player_id": pl.Int64, "game_id": pl.Int64})
    frames = []
    for col in pid_cols:
        frames.append(
            pbp.select(["game_id", pl.col(col).alias("player_id")])
            .filter(pl.col("player_id").is_not_null())
        )
    return pl.concat(frames).unique()


def _build_team_series(playoff_sched: pl.DataFrame) -> pl.DataFrame:
    """Annotate each playoff team-game with series-position metadata.

    Returns one row per (team, game_id) with:
        series_index               1-based ordinal of the team's series this spring
        games_in_current_series    1..N count of games into this series
        rest_days_inside_series    days since team's previous game in this series
                                   (−1 for game 1 of any series)
    """
    if len(playoff_sched) == 0:
        return pl.DataFrame(schema={
            "team": pl.Utf8,
            "game_id": pl.Int64,
            "game_date": pl.Utf8,
            "opponent": pl.Utf8,
            "series_index": pl.Int64,
            "games_in_current_series": pl.Int64,
            "rest_days_inside_series": pl.Int64,
        })

    # Explode to one row per team-side of each game.
    home = playoff_sched.select([
        pl.col("game_id"),
        pl.col("game_date"),
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
    ])
    away = playoff_sched.select([
        pl.col("game_id"),
        pl.col("game_date"),
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
    ])
    team_games = pl.concat([home, away]).sort(["team", "game_date", "game_id"])

    rows = team_games.to_dicts()
    # Single pass per team to assign series_index + per-series ordinal + rest.
    by_team: dict[str, list[dict]] = {}
    for r in rows:
        by_team.setdefault(r["team"], []).append(r)

    out: list[dict] = []
    for team, games in by_team.items():
        series_idx = 0
        prev_opp: str | None = None
        in_series_n = 0
        prev_date_in_series: datetime | None = None
        for g in games:
            opp = g["opponent"]
            d   = datetime.strptime(g["game_date"], "%Y-%m-%d")
            if opp != prev_opp:
                series_idx += 1
                in_series_n = 1
                rest = -1
                prev_date_in_series = d
            else:
                in_series_n += 1
                rest = (d - prev_date_in_series).days - 1 if prev_date_in_series else -1
                prev_date_in_series = d
            prev_opp = opp
            out.append({
                "team":                    team,
                "game_id":                 g["game_id"],
                "game_date":               g["game_date"],
                "opponent":                opp,
                "series_index":            series_idx,
                "games_in_current_series": in_series_n,
                "rest_days_inside_series": rest,
            })

    return pl.DataFrame(out, schema={
        "team": pl.Utf8,
        "game_id": pl.Int64,
        "game_date": pl.Utf8,
        "opponent": pl.Utf8,
        "series_index": pl.Int64,
        "games_in_current_series": pl.Int64,
        "rest_days_inside_series": pl.Int64,
    })


def _latest(dirpath: Path, prefix: str) -> Path | None:
    if not dirpath.exists():
        return None
    files = sorted(dirpath.glob(f"{prefix}_*.parquet"))
    return files[-1] if files else None


def _read_or_empty(dirpath: Path, prefix: str) -> pl.DataFrame:
    p = _latest(dirpath, prefix)
    return pl.read_parquet(p) if p else pl.DataFrame()


def _build_signals(args, as_of_iso: str) -> pl.DataFrame:
    sched_dir  = args.data_dir / SCHEDULE_SUBDIR
    sched_path = latest_schedule_parquet(sched_dir)
    if sched_path is None:
        print(f"[playoff-fatigue] No schedule parquet in {sched_dir}.")
        sys.exit(1)
    schedule = pl.read_parquet(sched_path).select(
        ["game_id", "game_date", "game_type", "home_team", "away_team"]
    )

    spring = _spring_year(as_of_iso)
    playoff_sched = schedule.filter(
        (pl.col("game_type") == GAME_TYPE_PLAYOFFS)
        & (pl.col("game_date").map_elements(_spring_year, return_dtype=pl.Int64) == spring)
        & (pl.col("game_date") <= as_of_iso)
    )
    if len(playoff_sched) == 0:
        print(f"[playoff-fatigue] No playoff games on or before {as_of_iso} for spring {spring}.")
        return pl.DataFrame()

    team_series = _build_team_series(playoff_sched)

    # PBP appearances → which players actually played each playoff game.
    with DataStore(args.data_dir) as store:
        pbp = store.pbp()
    if len(pbp) == 0:
        print("[playoff-fatigue] DataStore has no PBP. Run `ingest` first.")
        sys.exit(1)

    appearances = _per_player_per_game(pbp).join(
        playoff_sched.select(["game_id"]), on="game_id", how="inner"
    )
    if len(appearances) == 0:
        print("[playoff-fatigue] No player appearances joined to playoff schedule.")
        return pl.DataFrame()

    # Per-player cumulative playoff GP this spring (running count by game_date).
    apps_dated = appearances.join(
        playoff_sched.select(["game_id", "game_date"]),
        on="game_id", how="left",
    ).sort(["player_id", "game_date", "game_id"])
    apps_dated = apps_dated.with_columns(
        pl.col("game_id").cum_count().over("player_id").alias("cumulative_playoff_gp_this_spring")
    )

    # Player→team for each playoff appearance, so we can join in the team's
    # series-position metadata. Use event_owner_team_id when available.
    if "event_owner_team_id" in pbp.columns:
        pt_lookup = (
            pbp.select(["game_id", "event_owner_team_id"])
            .filter(pl.col("event_owner_team_id").is_not_null())
            .unique()
        )
        # Map team_id → team abbrev via schedule (home/away abbrevs known per game_id).
        # Each game appears twice — once per team_id — so we union them via the
        # schedule itself.
        # Simpler approach: pull pid → team_abbrev from PBP rows that have both.
        pass  # Fall through — for v1 use team-broadcast: join series metadata to
              # every player on either roster via home/away abbreviations.

    # Broadcast: every player who appeared in a game gets the home OR away
    # team_series row. We don't yet know which side they were on without
    # roster joins, so we join twice (home+away) and take the row whose
    # team matches their PBP team_id when available; for v1 we use the
    # union and de-dup with a coalesce on series fields.
    home_series = team_series.rename({"team": "home_team"}).drop("opponent")
    away_series = team_series.rename({"team": "away_team"}).drop("opponent")

    apps_with_sched = apps_dated.join(
        playoff_sched.select(["game_id", "home_team", "away_team"]),
        on="game_id", how="left",
    )

    # Roster → which side each player is on. Fall back to NHL roster cache file.
    rosters_path = args.data_dir / "rosters_latest.parquet"
    player_team: pl.DataFrame | None = None
    if rosters_path.exists():
        try:
            rosters = pl.read_parquet(rosters_path)
            if {"player_id", "team_abbrev"}.issubset(rosters.columns):
                player_team = rosters.select(["player_id", "team_abbrev"]).unique()
        except Exception:
            player_team = None

    if player_team is not None:
        apps_with_team = apps_with_sched.join(player_team, on="player_id", how="left")
        apps_with_team = apps_with_team.with_columns(
            pl.when(pl.col("team_abbrev") == pl.col("home_team"))
              .then(pl.col("home_team"))
              .when(pl.col("team_abbrev") == pl.col("away_team"))
              .then(pl.col("away_team"))
              .otherwise(None)
              .alias("player_team")
        )
    else:
        # Without roster context we still want every player to receive series
        # metadata — match by home_team and away_team independently and pick
        # the first non-null. v2: use PBP event_owner_team_id for attribution.
        apps_with_team = apps_with_sched.with_columns(
            pl.col("home_team").alias("player_team")
        )

    # Join in series metadata via player_team (matches either home or away row).
    enriched = apps_with_team.join(
        team_series.select([
            pl.col("team").alias("player_team"),
            "game_id",
            "series_index",
            "games_in_current_series",
            "rest_days_inside_series",
        ]),
        on=["player_team", "game_id"], how="left",
    )

    # Optional: travel + TZ loads, per team-game.
    travel = _read_or_empty(args.data_dir / TRAVEL_SUBDIR, "travel_distance")
    tz     = _read_or_empty(args.data_dir / TZ_SUBDIR, "time_zone_crossing")
    if len(travel) > 0 and {"team", "game_id", "miles_last_7d"}.issubset(travel.columns):
        # Normalize miles_last_7d to [0,1] by /2000 like composite_fi does.
        travel = travel.with_columns(
            (pl.col("miles_last_7d").clip(0.0, 2000.0) / 2000.0).alias("miles_load")
        ).select(["team", "game_id", "miles_load"])
        enriched = enriched.join(
            travel.rename({"team": "player_team"}),
            on=["player_team", "game_id"], how="left",
        )
    if len(tz) > 0 and {"team", "game_id", "tz_load_48h"}.issubset(tz.columns):
        tz = tz.with_columns(
            (pl.col("tz_load_48h").clip(0.0, 3.0) / 3.0).alias("tz_load")
        ).select(["team", "game_id", "tz_load"])
        enriched = enriched.join(
            tz.rename({"team": "player_team"}),
            on=["player_team", "game_id"], how="left",
        )

    # Add game_type for the model's no-effect gate.
    enriched = enriched.with_columns(pl.lit(GAME_TYPE_PLAYOFFS).cast(pl.Int64).alias("game_type"))

    # Fill defaults so the model can score even partial rows.
    enriched = enriched.with_columns([
        pl.col("games_in_current_series").fill_null(0),
        pl.col("rest_days_inside_series").fill_null(-1),
    ])
    for col, default in (("miles_load", 0.0), ("tz_load", 0.0)):
        if col not in enriched.columns:
            enriched = enriched.with_columns(pl.lit(default).cast(pl.Float64).alias(col))
        else:
            enriched = enriched.with_columns(pl.col(col).fill_null(default))

    return enriched.select([
        "player_id", "game_id", "game_date", "game_type",
        "games_in_current_series", "cumulative_playoff_gp_this_spring",
        "rest_days_inside_series", "miles_load", "tz_load",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-player-per-game playoff fatigue (Feature 3.23)."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None,
                        help="as-of date. Default: today (UTC).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir  = args.data_dir / PLAYOFF_FATIGUE_SUBDIR
    out_path = out_dir / f"playoff_fatigue_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[playoff-fatigue] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[playoff-fatigue] Building inputs as of {as_of}…")
    signals = _build_signals(args, as_of)
    if len(signals) == 0:
        print("[playoff-fatigue] No playoff signals to score. Writing empty parquet.")
        result = PlayoffFatigueModel().compute(
            pl.DataFrame(schema={
                "player_id": pl.Int64, "game_id": pl.Int64, "game_date": pl.Utf8,
            }),
            as_of_date=as_of,
        )
        path = write_playoff_fatigue(result, out_dir, as_of)
        print(f"[playoff-fatigue] Written: {path}")
        return

    print(f"  {len(signals):,} (player, playoff-game) input rows")

    model = PlayoffFatigueModel()
    result = model.compute(signals, as_of_date=as_of)
    n_rows = len(result)
    print(f"  {n_rows:,} player-game rows produced.")

    if n_rows > 0:
        top = result.sort("playoff_fatigue_score", descending=True).head(10)
        print("\n  Top 10 current playoff-fatigue scores:")
        print(f"  {'Player':<10}  {'Game':<10}  {'Date':<10}  {'Score':>6}  "
              f"{'Intens':>6}  {'CumGP':>6}  {'Compr':>6}  {'Travel':>6}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*6}  "
              f"{'─'*6}  {'─'*6}")
        for r in top.to_dicts():
            print(
                f"  {r['player_id']:<10}  {r['game_id']:<10}  {r['game_date']:<10}  "
                f"{r['playoff_fatigue_score']:>6.3f}  {r['series_intensity']:>6.3f}  "
                f"{r['cumulative_playoff_gp']:>6.3f}  {r['series_compression']:>6.3f}  "
                f"{r['cross_series_travel']:>6.3f}"
            )

    path = write_playoff_fatigue(result, out_dir, as_of)
    print(f"\n[playoff-fatigue] Written: {path}")


if __name__ == "__main__":
    main()
