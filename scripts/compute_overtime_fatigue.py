"""compute_overtime_fatigue — Feature 3.10 driver script.

Joins the TOI table (for per-player ``toi_ot_secs``) with the PBP table
(for the ``period_type == "OT"`` flag at game level) and the schedule
(for ``game_date``), then computes rolling 7-day OT fatigue per player.

Usage::

    uv run python scripts/compute_overtime_fatigue.py
    uv run python scripts/compute_overtime_fatigue.py --seasons 2023 2024
    uv run python scripts/compute_overtime_fatigue.py --force
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
from models.overtime_fatigue import (
    OvertimeFatigueTracker,
    write_overtime_fatigue,
)


DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
SCHEDULE_SUBDIR  = "schedule"
OT_FATIGUE_SUBDIR = "overtime_fatigue"


def _games_with_ot(pbp: pl.DataFrame) -> pl.DataFrame:
    """Return one row per game with ``played_ot`` boolean from PBP."""
    if len(pbp) == 0:
        return pl.DataFrame(schema={"game_id": pl.Int64, "played_ot": pl.Boolean})
    return (
        pbp.select(["game_id", "period_type"])
        .group_by("game_id")
        .agg((pl.col("period_type") == "OT").any().alias("played_ot"))
    )


def _load_ot_with_date(args) -> pl.DataFrame:
    sched_dir  = args.data_dir / SCHEDULE_SUBDIR
    sched_path = latest_schedule_parquet(sched_dir)
    if sched_path is None:
        print(f"[ot-fatigue] No schedule parquet found in {sched_dir}.")
        sys.exit(1)
    print(f"[ot-fatigue] Loading schedule: {sched_path}")
    schedule = pl.read_parquet(sched_path).select(["game_id", "game_date"])

    with DataStore(args.data_dir) as store:
        if args.seasons:
            toi_frames = [store.toi(season=s) for s in args.seasons]
            toi_frames = [df for df in toi_frames if len(df) > 0]
            toi_df = pl.concat(toi_frames) if toi_frames else store.toi()

            pbp_frames = [store.pbp(season=s) for s in args.seasons]
            pbp_frames = [df for df in pbp_frames if len(df) > 0]
            pbp_df = pl.concat(pbp_frames) if pbp_frames else store.pbp()
        else:
            toi_df = store.toi()
            pbp_df = store.pbp()

    if len(toi_df) == 0:
        print("[ot-fatigue] DataStore has no TOI rows. Run `ingest` first.")
        sys.exit(1)
    if len(pbp_df) == 0:
        print("[ot-fatigue] DataStore has no PBP rows. Run `ingest` first.")
        sys.exit(1)

    ot_flags = _games_with_ot(pbp_df)
    base = toi_df.select(["game_id", "player_id", "team_id", "toi_ot_secs"])
    joined = (
        base.join(ot_flags, on="game_id", how="left")
        .join(schedule, on="game_id", how="inner")
        .with_columns(
            pl.col("played_ot").fill_null(False),
            pl.col("toi_ot_secs").fill_null(0).cast(pl.Int64),
        )
    )
    if len(joined) == 0:
        print("[ot-fatigue] No TOI rows joined to scheduled games.")
        sys.exit(1)
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-player rolling 7-day OT fatigue (Feature 3.10)."
    )
    parser.add_argument("--seasons", nargs="*", type=int, default=[], metavar="YEAR")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ot_joined = _load_ot_with_date(args)
    print(f"  {len(ot_joined):,} (game, player) TOI rows")

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir = args.data_dir / OT_FATIGUE_SUBDIR
    out_path = out_dir / f"overtime_fatigue_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[ot-fatigue] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[ot-fatigue] Computing OT fatigue as of {as_of}…")
    result = OvertimeFatigueTracker().compute(ot_joined)
    n_rows = len(result)
    print(f"  {n_rows:,} player-game rows produced.")

    if n_rows > 0:
        top = result.sort("ot_fatigue_score", descending=True).head(10)
        print("\n  Top 10 OT fatigue loads (last 7 days):")
        print(f"  {'Player':<10}  {'Date':<10}  {'OT#':>3}  {'ActSec':>6}  {'Equiv':>6}  {'Score':>7}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*3}  {'─'*6}  {'─'*6}  {'─'*7}")
        for row in top.to_dicts():
            print(
                f"  {row['player_id']:<10}  {row['game_date']:<10}  "
                f"{row['ot_games_7day']:>3d}  {row['ot_secs_actual_7day']:>6d}  "
                f"{row['ot_load_equiv_secs']:>6.0f}  {row['ot_fatigue_score']:>7.0f}"
            )

    path = write_overtime_fatigue(result, out_dir, as_of)
    print(f"\n[ot-fatigue] Written: {path}")


if __name__ == "__main__":
    main()
