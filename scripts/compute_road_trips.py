"""compute_road_trips — Feature 3.2 driver script.

Loads (or fetches) an NHL schedule parquet, annotates each team-game with
road-trip identifiers + position-within-trip, and writes the result to
parquet.

Usage::

    uv run python scripts/compute_road_trips.py
    uv run python scripts/compute_road_trips.py --seasons 2024 2025
    uv run python scripts/compute_road_trips.py --schedule /path/to/schedule.parquet
    uv run python scripts/compute_road_trips.py --force
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from data.schedule_sync import (
    fetch_schedule,
    latest_schedule_parquet,
    write_schedule,
)
from models.road_trip_tracker import (
    RoadTripTracker,
    write_road_trips,
)


DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
SCHEDULE_SUBDIR = "schedule"
TRIPS_SUBDIR    = "road_trips"


def _load_or_fetch_schedule(args) -> pl.DataFrame:
    if args.schedule is not None:
        path = Path(args.schedule)
        if not path.exists():
            print(f"[road-trips] --schedule path not found: {path}")
            sys.exit(1)
        print(f"[road-trips] Loading schedule from {path}")
        return pl.read_parquet(path)

    sched_dir: Path = args.data_dir / SCHEDULE_SUBDIR
    cached = latest_schedule_parquet(sched_dir)
    if cached is not None and not args.refresh_schedule:
        print(f"[road-trips] Loading cached schedule: {cached}")
        return pl.read_parquet(cached)

    print(f"[road-trips] Fetching schedule live for seasons={args.seasons} "
          f"game_types={args.game_types}…")
    df = asyncio.run(fetch_schedule(sorted(args.seasons), game_types=args.game_types))
    if len(df) == 0:
        print("[road-trips] Schedule fetch returned no rows.")
        sys.exit(1)
    out_sched = write_schedule(df, sched_dir, sorted(args.seasons))
    print(f"  Cached schedule at {out_sched} ({len(df):,} games)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute road-trip annotations per team-game (Feature 3.2)."
    )
    parser.add_argument("--seasons", nargs="+", type=int, default=[2024], metavar="YEAR")
    parser.add_argument("--game-types", nargs="+", type=int, default=[2], metavar="N",
                        help="NHL game types (1=preseason, 2=regular, 3=playoffs, "
                             "4=all-star). Default: [2].")
    parser.add_argument("--schedule", type=Path, default=None)
    parser.add_argument("--refresh-schedule", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    schedule_df = _load_or_fetch_schedule(args)
    print(f"  {len(schedule_df):,} schedule rows")

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    trips_dir = args.data_dir / TRIPS_SUBDIR
    out_path = trips_dir / f"road_trips_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[road-trips] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[road-trips] Computing road-trip rows as of {as_of}…")
    result_df = RoadTripTracker().compute(schedule_df)
    n_team_games = len(result_df)
    n_road = result_df.filter(pl.col("is_road")).height
    n_trips = result_df.filter(pl.col("trip_id") > 0)["trip_id"].n_unique() if n_road else 0
    print(f"  {n_team_games:,} team-game rows | {n_road:,} road games | "
          f"{n_trips:,} distinct trips (per team).")

    if n_road > 0:
        long_trips = (
            result_df
            .filter(pl.col("trip_game_number") == 1)
            .sort("trip_total_length", descending=True)
            .head(10)
        )
        print("\n  Top 10 longest road trips this slate:")
        print(f"  {'Team':<6}  {'Start date':<10}  {'Length':>6}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*6}")
        for row in long_trips.to_dicts():
            print(f"  {row['team']:<6}  {row['game_date']:<10}  {row['trip_total_length']:>6}")

    path = write_road_trips(result_df, trips_dir, as_of)
    print(f"\n[road-trips] Written: {path}")


if __name__ == "__main__":
    main()
