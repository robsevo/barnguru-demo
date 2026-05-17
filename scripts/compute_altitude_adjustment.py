"""compute_altitude_adjustment — Feature 3.6 driver script.

Loads (or fetches) an NHL schedule parquet, annotates each team-game with
venue elevation, visitor elevation delta, and aerobic-penalty score, and
writes the result to parquet.

Usage::

    uv run python scripts/compute_altitude_adjustment.py
    uv run python scripts/compute_altitude_adjustment.py --seasons 2024 2025
    uv run python scripts/compute_altitude_adjustment.py --schedule /path/to/schedule.parquet
    uv run python scripts/compute_altitude_adjustment.py --force
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
from models.altitude_adjustment import (
    AltitudeAdjustmentModel,
    write_altitude_adjustment,
)


DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
SCHEDULE_SUBDIR = "schedule"
ALT_SUBDIR      = "altitude_adjustment"


def _load_or_fetch_schedule(args) -> pl.DataFrame:
    if args.schedule is not None:
        path = Path(args.schedule)
        if not path.exists():
            print(f"[altitude] --schedule path not found: {path}")
            sys.exit(1)
        print(f"[altitude] Loading schedule from {path}")
        return pl.read_parquet(path)

    sched_dir: Path = args.data_dir / SCHEDULE_SUBDIR
    cached = latest_schedule_parquet(sched_dir)
    if cached is not None and not args.refresh_schedule:
        print(f"[altitude] Loading cached schedule: {cached}")
        return pl.read_parquet(cached)

    print(f"[altitude] Fetching schedule live for seasons={args.seasons} "
          f"game_types={args.game_types}…")
    df = asyncio.run(fetch_schedule(sorted(args.seasons), game_types=args.game_types))
    if len(df) == 0:
        print("[altitude] Schedule fetch returned no rows.")
        sys.exit(1)
    out_sched = write_schedule(df, sched_dir, sorted(args.seasons))
    print(f"  Cached schedule at {out_sched} ({len(df):,} games)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute altitude penalty per team-game (Feature 3.6)."
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
    alt_dir = args.data_dir / ALT_SUBDIR
    out_path = alt_dir / f"altitude_adjustment_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[altitude] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[altitude] Computing altitude features as of {as_of}…")
    result_df = AltitudeAdjustmentModel().compute(schedule_df)
    n_rows = len(result_df)
    print(f"  {n_rows:,} team-game rows produced.")

    if n_rows > 0:
        n_high = result_df.filter(pl.col("is_high_altitude")).height
        print(f"  high-altitude team-games: {n_high:,}")
        worst = (
            result_df
            .sort("altitude_penalty", descending=True)
            .head(10)
        )
        print("\n  Top 10 worst altitude penalties:")
        print(f"  {'Team':<6}  {'Date':<10}  {'Venue':<6}  {'Δft':>6}  {'Penalty':>7}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*7}")
        for row in worst.to_dicts():
            print(
                f"  {row['team']:<6}  {row['game_date']:<10}  "
                f"{row['venue_team']:<6}  "
                f"{row['elevation_delta_ft']:>6d}  "
                f"{row['altitude_penalty']:>7.4f}"
            )

    path = write_altitude_adjustment(result_df, alt_dir, as_of)
    print(f"\n[altitude] Written: {path}")


if __name__ == "__main__":
    main()
