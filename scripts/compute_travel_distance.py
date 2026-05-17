"""compute_travel_distance — Feature 3.3 driver script.

Loads (or fetches) an NHL schedule parquet, computes per-team-per-game
travel mileage and rolling-window miles, and writes the result to parquet.

Usage::

    uv run python scripts/compute_travel_distance.py
    uv run python scripts/compute_travel_distance.py --seasons 2024 2025
    uv run python scripts/compute_travel_distance.py --schedule /path/to/schedule.parquet
    uv run python scripts/compute_travel_distance.py --force
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
from models.travel_distance import (
    TravelDistanceCalculator,
    write_travel_distance,
)


DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
SCHEDULE_SUBDIR = "schedule"
TRAVEL_SUBDIR   = "travel_distance"


def _load_or_fetch_schedule(args) -> pl.DataFrame:
    if args.schedule is not None:
        path = Path(args.schedule)
        if not path.exists():
            print(f"[travel-distance] --schedule path not found: {path}")
            sys.exit(1)
        print(f"[travel-distance] Loading schedule from {path}")
        return pl.read_parquet(path)

    sched_dir: Path = args.data_dir / SCHEDULE_SUBDIR
    cached = latest_schedule_parquet(sched_dir)
    if cached is not None and not args.refresh_schedule:
        print(f"[travel-distance] Loading cached schedule: {cached}")
        return pl.read_parquet(cached)

    print(f"[travel-distance] Fetching schedule live for seasons={args.seasons} "
          f"game_types={args.game_types}…")
    df = asyncio.run(fetch_schedule(sorted(args.seasons), game_types=args.game_types))
    if len(df) == 0:
        print("[travel-distance] Schedule fetch returned no rows.")
        sys.exit(1)
    out_sched = write_schedule(df, sched_dir, sorted(args.seasons))
    print(f"  Cached schedule at {out_sched} ({len(df):,} games)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute travel mileage per team-game (Feature 3.3)."
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
    travel_dir = args.data_dir / TRAVEL_SUBDIR
    out_path = travel_dir / f"travel_distance_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[travel-distance] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[travel-distance] Computing mileage as of {as_of}…")
    result_df = TravelDistanceCalculator().compute(schedule_df)
    n_team_games = len(result_df)
    print(f"  {n_team_games:,} team-game rows produced.")

    if n_team_games > 0:
        # Top weekly mileage per team (any game where rolling 7-day miles peaks)
        top_weeks = (
            result_df
            .sort("miles_last_7d", descending=True)
            .head(10)
        )
        print("\n  Top 10 single-game peak 7-day mileage:")
        print(f"  {'Team':<6}  {'Date':<10}  {'To game':>9}  {'7d miles':>9}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*9}  {'─'*9}")
        for row in top_weeks.to_dicts():
            print(
                f"  {row['team']:<6}  {row['game_date']:<10}  "
                f"{row['miles_to_game']:>9.0f}  {row['miles_last_7d']:>9.0f}"
            )

    path = write_travel_distance(result_df, travel_dir, as_of)
    print(f"\n[travel-distance] Written: {path}")


if __name__ == "__main__":
    main()
