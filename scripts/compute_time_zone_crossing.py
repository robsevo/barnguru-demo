"""compute_time_zone_crossing — Feature 3.4 driver script.

Loads (or fetches) an NHL schedule parquet, annotates each team-game with
time-zone crossing features (signed delta, direction, 48h absolute sum),
and writes the result to parquet.

Usage::

    uv run python scripts/compute_time_zone_crossing.py
    uv run python scripts/compute_time_zone_crossing.py --seasons 2024 2025
    uv run python scripts/compute_time_zone_crossing.py --schedule /path/to/schedule.parquet
    uv run python scripts/compute_time_zone_crossing.py --force
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
from models.time_zone_crossing import (
    TimeZoneCrossingModel,
    write_time_zone_crossing,
)


DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
SCHEDULE_SUBDIR = "schedule"
TZ_SUBDIR       = "time_zone_crossing"


def _load_or_fetch_schedule(args) -> pl.DataFrame:
    if args.schedule is not None:
        path = Path(args.schedule)
        if not path.exists():
            print(f"[tz-crossing] --schedule path not found: {path}")
            sys.exit(1)
        print(f"[tz-crossing] Loading schedule from {path}")
        return pl.read_parquet(path)

    sched_dir: Path = args.data_dir / SCHEDULE_SUBDIR
    cached = latest_schedule_parquet(sched_dir)
    if cached is not None and not args.refresh_schedule:
        print(f"[tz-crossing] Loading cached schedule: {cached}")
        return pl.read_parquet(cached)

    print(f"[tz-crossing] Fetching schedule live for seasons={args.seasons} "
          f"game_types={args.game_types}…")
    df = asyncio.run(fetch_schedule(sorted(args.seasons), game_types=args.game_types))
    if len(df) == 0:
        print("[tz-crossing] Schedule fetch returned no rows.")
        sys.exit(1)
    out_sched = write_schedule(df, sched_dir, sorted(args.seasons))
    print(f"  Cached schedule at {out_sched} ({len(df):,} games)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute time-zone crossings per team-game (Feature 3.4)."
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
    tz_dir = args.data_dir / TZ_SUBDIR
    out_path = tz_dir / f"time_zone_crossing_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[tz-crossing] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[tz-crossing] Computing TZ crossings as of {as_of}…")
    result_df = TimeZoneCrossingModel().compute(schedule_df)
    n_team_games = len(result_df)
    print(f"  {n_team_games:,} team-game rows produced.")

    if n_team_games > 0:
        n_east = result_df.filter(pl.col("direction") == "east").height
        n_west = result_df.filter(pl.col("direction") == "west").height
        print(f"  eastbound legs: {n_east:,} | westbound legs: {n_west:,}")

        worst = (
            result_df
            .sort("abs_tz_crossed_48h", descending=True)
            .head(10)
        )
        print("\n  Top 10 worst 48-hour TZ-crossing loads:")
        print(f"  {'Team':<6}  {'Date':<10}  {'Δprev':>6}  {'|48h|':>6}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*6}  {'─'*6}")
        for row in worst.to_dicts():
            print(
                f"  {row['team']:<6}  {row['game_date']:<10}  "
                f"{row['tz_crossed_from_prev']:>+6.1f}  "
                f"{row['abs_tz_crossed_48h']:>6.1f}"
            )

    path = write_time_zone_crossing(result_df, tz_dir, as_of)
    print(f"\n[tz-crossing] Written: {path}")


if __name__ == "__main__":
    main()
