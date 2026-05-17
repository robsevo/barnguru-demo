"""compute_schedule_density — Feature 3.1 driver script.

Loads (or fetches) an NHL schedule parquet, computes per-team-per-game
schedule-density features, and writes the result to parquet.

Usage::

    uv run python scripts/compute_schedule_density.py
    uv run python scripts/compute_schedule_density.py --seasons 2024 2025
    uv run python scripts/compute_schedule_density.py --schedule /path/to/schedule.parquet
    uv run python scripts/compute_schedule_density.py --force
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
from models.schedule_density import (
    ScheduleDensityModel,
    write_schedule_density,
)


DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
SCHEDULE_SUBDIR = "schedule"
DENSITY_SUBDIR  = "schedule_density"


def _load_or_fetch_schedule(args) -> pl.DataFrame:
    """Resolve schedule input from --schedule, cache, or live fetch."""
    if args.schedule is not None:
        path = Path(args.schedule)
        if not path.exists():
            print(f"[schedule-density] --schedule path not found: {path}")
            sys.exit(1)
        print(f"[schedule-density] Loading schedule from {path}")
        return pl.read_parquet(path)

    sched_dir: Path = args.data_dir / SCHEDULE_SUBDIR
    cached = latest_schedule_parquet(sched_dir)
    if cached is not None and not args.refresh_schedule:
        print(f"[schedule-density] Loading cached schedule: {cached}")
        return pl.read_parquet(cached)

    print(f"[schedule-density] Fetching schedule live for seasons={args.seasons} "
          f"game_types={args.game_types}…")
    df = asyncio.run(fetch_schedule(sorted(args.seasons), game_types=args.game_types))
    if len(df) == 0:
        print("[schedule-density] Schedule fetch returned no rows.")
        sys.exit(1)
    out_sched = write_schedule(df, sched_dir, sorted(args.seasons))
    print(f"  Cached schedule at {out_sched} ({len(df):,} games)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute schedule density per team-game (Feature 3.1)."
    )
    parser.add_argument("--seasons", nargs="+", type=int, default=[2024],
                        metavar="YEAR", help="Seasons to fetch when live syncing (default: 2024).")
    parser.add_argument("--game-types", nargs="+", type=int, default=[2],
                        metavar="N",
                        help="NHL game types to include (1=preseason, 2=regular, "
                             "3=playoffs, 4=all-star). Default: [2].")
    parser.add_argument("--schedule", type=Path, default=None,
                        help="Path to a pre-built schedule parquet.")
    parser.add_argument("--refresh-schedule", action="store_true",
                        help="Force a live NHL API fetch even if a cached parquet exists.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"Root data directory (default: {DEFAULT_DATA_DIR}).")
    parser.add_argument("--date", type=str, default=None,
                        help="Tag for the output parquet (default: today UTC).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing density parquet for today.")
    args = parser.parse_args()

    schedule_df = _load_or_fetch_schedule(args)
    print(f"  {len(schedule_df):,} schedule rows")

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    density_dir = args.data_dir / DENSITY_SUBDIR
    out_path = density_dir / f"schedule_density_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[schedule-density] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[schedule-density] Computing density features as of {as_of}…")
    result_df = ScheduleDensityModel().compute(schedule_df)
    n_team_games = len(result_df)
    print(f"  {n_team_games:,} team-game rows produced.")

    if n_team_games > 0:
        n_b2b = result_df.filter(pl.col("is_b2b")).height
        n_3in4 = result_df.filter(pl.col("is_3_in_4")).height
        print(f"  is_b2b: {n_b2b:,} | is_3_in_4: {n_3in4:,}")

    path = write_schedule_density(result_df, density_dir, as_of)
    print(f"\n[schedule-density] Written: {path}")


if __name__ == "__main__":
    main()
