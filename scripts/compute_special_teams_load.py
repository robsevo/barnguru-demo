"""compute_special_teams_load — Feature 3.8 driver script.

Derives per-(game, player) PP/PK seconds from the PBP + shifts tables in
the DataStore, joins with the schedule for ``game_date``, and computes
rolling 5-game special-teams accumulation + weighted intensity score per
player.

Per-game PPTOI/SHTOI is built using the same shift × strength-event overlap
logic that ``PlayerStatsAggregator._compute_pptoi_shtoi`` uses for season
totals — only here we keep ``game_id`` in the group-by so we get a row per
game.

Usage::

    uv run python scripts/compute_special_teams_load.py
    uv run python scripts/compute_special_teams_load.py --seasons 2023 2024
    uv run python scripts/compute_special_teams_load.py --force
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
from models.special_teams_load import (
    SpecialTeamsLoadTracker,
    write_special_teams_load,
)


DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
SCHEDULE_SUBDIR = "schedule"
ST_LOAD_SUBDIR  = "special_teams_load"


def _per_game_pptoi_shtoi(
    pbp: pl.DataFrame, shifts: pl.DataFrame
) -> pl.DataFrame:
    """Per-(game, player, team) PPTOI and SHTOI in seconds.

    Mirrors PlayerStatsAggregator._compute_pptoi_shtoi but keeps game_id in
    the group-by so callers get one row per game.
    """
    empty = pl.DataFrame(
        schema={
            "game_id":    pl.Int64,
            "player_id":  pl.Int64,
            "team_id":    pl.Int64,
            "pptoi_secs": pl.Int64,
            "shtoi_secs": pl.Int64,
        }
    )
    if "home_team_id" not in pbp.columns or len(pbp) == 0 or len(shifts) == 0:
        return empty

    game_teams = (
        pbp.filter(pl.col("home_team_id").is_not_null())
        .select(["game_id", "home_team_id", "away_team_id"])
        .unique(subset=["game_id"])
    )
    if len(game_teams) == 0:
        return empty

    pp_pk = (
        pbp.filter(pl.col("strength").is_in(["pp", "pk"]))
        .with_columns(
            ((pl.col("period") - 1) * 1200 + pl.col("time_in_period_secs"))
            .alias("event_gsecs")
        )
        .select(["game_id", "event_gsecs", "strength"])
    )
    if len(pp_pk) == 0:
        return empty

    shifts_with_teams = shifts.join(game_teams, on="game_id", how="left")
    overlapping = (
        shifts_with_teams.join(pp_pk, on="game_id", how="inner")
        .filter(
            (pl.col("event_gsecs") >= pl.col("game_seconds_start"))
            & (pl.col("event_gsecs") <= pl.col("game_seconds_end"))
        )
    )
    if len(overlapping) == 0:
        return empty

    overlapping = overlapping.with_columns(
        pl.when(
            ((pl.col("strength") == "pp") & (pl.col("team_id") == pl.col("home_team_id")))
            | ((pl.col("strength") == "pk") & (pl.col("team_id") == pl.col("away_team_id")))
        )
        .then(pl.lit("pp"))
        .when(
            ((pl.col("strength") == "pk") & (pl.col("team_id") == pl.col("home_team_id")))
            | ((pl.col("strength") == "pp") & (pl.col("team_id") == pl.col("away_team_id")))
        )
        .then(pl.lit("pk"))
        .otherwise(pl.lit("ev"))
        .alias("shift_type")
    )

    # One classification per unique shift (sort pp/pk before ev so ST wins).
    shift_classified = (
        overlapping
        .sort(["player_id", "game_id", "game_seconds_start", "shift_type"])
        .unique(
            subset=["player_id", "game_id", "game_seconds_start"],
            keep="first",
        )
        .with_columns(
            (pl.col("game_seconds_end") - pl.col("game_seconds_start"))
            .alias("shift_dur")
        )
    )

    grouped = (
        shift_classified
        .group_by(["game_id", "player_id", "team_id", "shift_type"])
        .agg(pl.col("shift_dur").sum().alias("secs"))
        .pivot(
            index=["game_id", "player_id", "team_id"],
            on="shift_type",
            values="secs",
            aggregate_function="sum",
        )
    )
    # Pivot may omit columns when no rows for that strength existed.
    for col in ("pp", "pk"):
        if col not in grouped.columns:
            grouped = grouped.with_columns(pl.lit(0).cast(pl.Int64).alias(col))
    grouped = grouped.with_columns(
        pl.col("pp").fill_null(0).cast(pl.Int64).alias("pptoi_secs"),
        pl.col("pk").fill_null(0).cast(pl.Int64).alias("shtoi_secs"),
    )
    return grouped.select(["game_id", "player_id", "team_id", "pptoi_secs", "shtoi_secs"])


def _load_st_with_date(args) -> pl.DataFrame:
    sched_dir = args.data_dir / SCHEDULE_SUBDIR
    sched_path = latest_schedule_parquet(sched_dir)
    if sched_path is None:
        print(f"[st-load] No schedule parquet found in {sched_dir}.")
        sys.exit(1)
    print(f"[st-load] Loading schedule: {sched_path}")
    schedule = pl.read_parquet(sched_path).select(["game_id", "game_date"])

    with DataStore(args.data_dir) as store:
        if args.seasons:
            pbp_frames    = [store.pbp(season=s)    for s in args.seasons]
            shift_frames  = [store.shifts(season=s) for s in args.seasons]
            pbp_frames    = [df for df in pbp_frames    if len(df) > 0]
            shift_frames  = [df for df in shift_frames  if len(df) > 0]
            pbp_df    = pl.concat(pbp_frames)   if pbp_frames    else store.pbp()
            shifts_df = pl.concat(shift_frames) if shift_frames  else store.shifts()
        else:
            pbp_df    = store.pbp()
            shifts_df = store.shifts()

    if len(pbp_df) == 0 or len(shifts_df) == 0:
        print("[st-load] DataStore has no PBP or shifts. Run `ingest` first.")
        sys.exit(1)

    st_df = _per_game_pptoi_shtoi(pbp_df, shifts_df)
    joined = st_df.join(schedule, on="game_id", how="inner")
    if len(joined) == 0:
        print("[st-load] No ST rows joined to scheduled games.")
        sys.exit(1)
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-player rolling special-teams load (Feature 3.8)."
    )
    parser.add_argument("--seasons", nargs="*", type=int, default=[], metavar="YEAR")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    st_joined = _load_st_with_date(args)
    print(f"  {len(st_joined):,} (game, player) ST rows")

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir = args.data_dir / ST_LOAD_SUBDIR
    out_path = out_dir / f"special_teams_load_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[st-load] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[st-load] Computing ST load as of {as_of}…")
    result = SpecialTeamsLoadTracker().compute(st_joined)
    n_rows = len(result)
    print(f"  {n_rows:,} player-game rows produced.")

    if n_rows > 0:
        top = (
            result.sort("st_intensity_score", descending=True).head(10)
        )
        print("\n  Top 10 ST-intensity-score rows:")
        print(f"  {'Player':<10}  {'Date':<10}  {'PP5s':>5}  {'PK5s':>5}  {'Score':>8}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*5}  {'─'*5}  {'─'*8}")
        for row in top.to_dicts():
            print(
                f"  {row['player_id']:<10}  {row['game_date']:<10}  "
                f"{row['pptoi_5game_sum_secs']:>5d}  "
                f"{row['shtoi_5game_sum_secs']:>5d}  "
                f"{row['st_intensity_score']:>8.1f}"
            )

    path = write_special_teams_load(result, out_dir, as_of)
    print(f"\n[st-load] Written: {path}")


if __name__ == "__main__":
    main()
