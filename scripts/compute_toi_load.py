"""compute_toi_load — Feature 3.7 driver script.

Loads per-game TOI from the DataStore, joins with the schedule for
``game_date``, computes rolling 5-game / season averages and spike flags
per player, and writes the result to parquet.

Usage::

    uv run python scripts/compute_toi_load.py
    uv run python scripts/compute_toi_load.py --seasons 2023 2024
    uv run python scripts/compute_toi_load.py --force
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
from models.toi_load import TOILoadTracker, write_toi_load


DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
SCHEDULE_SUBDIR  = "schedule"
TOI_LOAD_SUBDIR  = "toi_load"


def _load_toi_with_date(args) -> pl.DataFrame:
    """Pull TOI from DataStore and attach game_date from the cached schedule."""
    sched_dir = args.data_dir / SCHEDULE_SUBDIR
    sched_path = latest_schedule_parquet(sched_dir)
    if sched_path is None:
        print(f"[toi-load] No schedule parquet found in {sched_dir}.")
        print("  Run schedule-density (or any other Phase 3 script) first to "
              "produce a schedule cache.")
        sys.exit(1)
    print(f"[toi-load] Loading schedule: {sched_path}")
    schedule = pl.read_parquet(sched_path).select(["game_id", "game_date"])

    with DataStore(args.data_dir) as store:
        if args.seasons:
            toi_frames = [store.toi(season=s) for s in args.seasons]
            toi_frames = [df for df in toi_frames if len(df) > 0]
            toi_df = pl.concat(toi_frames) if toi_frames else store.toi()
        else:
            toi_df = store.toi()

    if len(toi_df) == 0:
        print("[toi-load] DataStore has no TOI rows. Run `ingest` first.")
        sys.exit(1)

    joined = toi_df.join(schedule, on="game_id", how="inner")
    if len(joined) == 0:
        print("[toi-load] TOI rows did not join to any scheduled game_id.")
        sys.exit(1)
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-player rolling TOI load (Feature 3.7)."
    )
    parser.add_argument("--seasons", nargs="*", type=int, default=[], metavar="YEAR",
                        help="Optional season filter. Default: all seasons in DataStore.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    toi_joined = _load_toi_with_date(args)
    print(f"  {len(toi_joined):,} (game, player) TOI rows")

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir = args.data_dir / TOI_LOAD_SUBDIR
    out_path = out_dir / f"toi_load_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[toi-load] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[toi-load] Computing TOI load as of {as_of}…")
    result = TOILoadTracker().compute(toi_joined)
    n_rows = len(result)
    print(f"  {n_rows:,} player-game rows produced.")

    if n_rows > 0:
        n_spike = result.filter(pl.col("is_toi_spike")).height
        print(f"  TOI spikes: {n_spike:,}")
        top = (
            result.sort("toi_spike_z", descending=True).head(10)
        )
        print("\n  Top 10 TOI spikes (most extreme z-scores):")
        print(f"  {'Player':<10}  {'Date':<10}  {'TOI(s)':>7}  {'SeasonAvg':>9}  {'z':>5}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*7}  {'─'*9}  {'─'*5}")
        for row in top.to_dicts():
            print(
                f"  {row['player_id']:<10}  {row['game_date']:<10}  "
                f"{row['toi_secs']:>7d}  {row['toi_season_avg_secs']:>9.1f}  "
                f"{row['toi_spike_z']:>5.2f}"
            )

    path = write_toi_load(result, out_dir, as_of)
    print(f"\n[toi-load] Written: {path}")


if __name__ == "__main__":
    main()
