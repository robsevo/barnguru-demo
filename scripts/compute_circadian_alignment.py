"""compute_circadian_alignment — Feature 3.5 driver script.

Loads (or fetches) an NHL schedule parquet, computes per-team-per-game
body-clock misalignment and prime-window deviation, and writes the result
to parquet.

If the schedule does not include a ``start_time_utc`` column, the model
falls back to 19:00 venue-local and emits a DataMissingWarning. Re-run
``gretzky sync`` once the start-time pipeline is wired through Phase 1.

Usage::

    uv run python scripts/compute_circadian_alignment.py
    uv run python scripts/compute_circadian_alignment.py --seasons 2024 2025
    uv run python scripts/compute_circadian_alignment.py --schedule /path/to/schedule.parquet
    uv run python scripts/compute_circadian_alignment.py --force
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
from models.circadian_alignment import (
    CircadianAlignmentScorer,
    write_circadian_alignment,
)


DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
SCHEDULE_SUBDIR   = "schedule"
CIRCADIAN_SUBDIR  = "circadian_alignment"


def _load_or_fetch_schedule(args) -> pl.DataFrame:
    if args.schedule is not None:
        path = Path(args.schedule)
        if not path.exists():
            print(f"[circadian] --schedule path not found: {path}")
            sys.exit(1)
        print(f"[circadian] Loading schedule from {path}")
        return pl.read_parquet(path)

    sched_dir: Path = args.data_dir / SCHEDULE_SUBDIR
    cached = latest_schedule_parquet(sched_dir)
    if cached is not None and not args.refresh_schedule:
        print(f"[circadian] Loading cached schedule: {cached}")
        return pl.read_parquet(cached)

    print(f"[circadian] Fetching schedule live for seasons={args.seasons} "
          f"game_types={args.game_types}…")
    df = asyncio.run(fetch_schedule(sorted(args.seasons), game_types=args.game_types))
    if len(df) == 0:
        print("[circadian] Schedule fetch returned no rows.")
        sys.exit(1)
    out_sched = write_schedule(df, sched_dir, sorted(args.seasons))
    print(f"  Cached schedule at {out_sched} ({len(df):,} games)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute circadian alignment per team-game (Feature 3.5)."
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
    out_dir = args.data_dir / CIRCADIAN_SUBDIR
    out_path = out_dir / f"circadian_alignment_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[circadian] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[circadian] Computing alignment as of {as_of}…")
    result_df = CircadianAlignmentScorer().compute(schedule_df)
    n_rows = len(result_df)
    print(f"  {n_rows:,} team-game rows produced.")

    if n_rows > 0:
        worst = (
            result_df
            .sort("prime_window_deviation", descending=True)
            .head(10)
        )
        print("\n  Top 10 worst body-clock deviations from 19:00:")
        print(f"  {'Team':<6}  {'Date':<10}  {'Venue':<6}  {'Δhrs':>5}  {'BodyClk':>7}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*6}  {'─'*5}  {'─'*7}")
        for row in worst.to_dicts():
            print(
                f"  {row['team']:<6}  {row['game_date']:<10}  "
                f"{row['venue_team']:<6}  "
                f"{row['misalignment_hours']:>+5.1f}  "
                f"{row['body_clock_hours']:>7.2f}"
            )

    path = write_circadian_alignment(result_df, out_dir, as_of)
    print(f"\n[circadian] Written: {path}")


if __name__ == "__main__":
    main()
