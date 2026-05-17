"""compute_fight_fatigue — Feature 3.11 driver script.

Aggregates per-(game, player) fighting majors from the PBP table
(``penalty_type == "MAJ"`` AND penalty_description contains "fight"),
joins with the schedule for ``game_date``, and computes the decayed
adrenal-load score per player.

Usage::

    uv run python scripts/compute_fight_fatigue.py
    uv run python scripts/compute_fight_fatigue.py --seasons 2023 2024
    uv run python scripts/compute_fight_fatigue.py --force
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
from models.fight_fatigue import (
    FightFatigueTracker,
    write_fight_fatigue,
)


DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
SCHEDULE_SUBDIR     = "schedule"
FIGHT_FATIGUE_SUBDIR = "fight_fatigue"


def _aggregate_fights(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per-(game, player, team) fighting-major counts from PBP."""
    empty = pl.DataFrame(
        schema={
            "game_id":          pl.Int64,
            "player_id":        pl.Int64,
            "team_id":          pl.Int64,
            "fights_this_game": pl.Int64,
        }
    )
    if len(pbp) == 0:
        return empty

    needed = {"event_type", "penalty_type", "penalty_description",
              "committed_by_id", "event_owner_team_id"}
    if not needed.issubset(set(pbp.columns)):
        return empty

    fights = (
        pbp.filter(
            (pl.col("event_type") == "penalty")
            & (pl.col("penalty_type") == "MAJ")
            & pl.col("penalty_description").is_not_null()
            & pl.col("penalty_description").str.to_lowercase().str.contains("fight")
            & pl.col("committed_by_id").is_not_null()
        )
        .select([
            "game_id",
            pl.col("committed_by_id").alias("player_id"),
            pl.col("event_owner_team_id").alias("team_id"),
        ])
        .group_by(["game_id", "player_id", "team_id"])
        .agg(pl.len().alias("fights_this_game"))
    )
    if len(fights) == 0:
        return empty
    return fights.with_columns(pl.col("fights_this_game").cast(pl.Int64))


def _load_fights_with_date(args) -> pl.DataFrame:
    sched_dir  = args.data_dir / SCHEDULE_SUBDIR
    sched_path = latest_schedule_parquet(sched_dir)
    if sched_path is None:
        print(f"[fight-fatigue] No schedule parquet found in {sched_dir}.")
        sys.exit(1)
    print(f"[fight-fatigue] Loading schedule: {sched_path}")
    schedule = pl.read_parquet(sched_path).select(["game_id", "game_date"])

    with DataStore(args.data_dir) as store:
        if args.seasons:
            frames = [store.pbp(season=s) for s in args.seasons]
            frames = [df for df in frames if len(df) > 0]
            pbp_df = pl.concat(frames) if frames else store.pbp()
        else:
            pbp_df = store.pbp()

    if len(pbp_df) == 0:
        print("[fight-fatigue] DataStore has no PBP rows. Run `ingest` first.")
        sys.exit(1)

    fights = _aggregate_fights(pbp_df)
    joined = fights.join(schedule, on="game_id", how="inner")
    if len(joined) == 0:
        print("[fight-fatigue] No fighting-major rows joined to scheduled games.")
        # Still write empty so downstream can detect the file.
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-player rolling fight-fatigue load (Feature 3.11)."
    )
    parser.add_argument("--seasons", nargs="*", type=int, default=[], metavar="YEAR")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    fights_joined = _load_fights_with_date(args)
    print(f"  {len(fights_joined):,} (game, player) fight rows")

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir = args.data_dir / FIGHT_FATIGUE_SUBDIR
    out_path = out_dir / f"fight_fatigue_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[fight-fatigue] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[fight-fatigue] Computing fight fatigue as of {as_of}…")
    result = FightFatigueTracker().compute(fights_joined)
    n_rows = len(result)
    print(f"  {n_rows:,} player-game rows produced.")

    if n_rows > 0:
        top = result.sort("fight_load_score", descending=True).head(10)
        print("\n  Top 10 active fight loads:")
        print(f"  {'Player':<10}  {'Date':<10}  {'Fights14':>8}  {'DaysAgo':>7}  {'Score':>6}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*7}  {'─'*6}")
        for row in top.to_dicts():
            print(
                f"  {row['player_id']:<10}  {row['game_date']:<10}  "
                f"{row['fights_14day']:>8d}  {row['days_since_last_fight']:>7d}  "
                f"{row['fight_load_score']:>6.2f}"
            )

    path = write_fight_fatigue(result, out_dir, as_of)
    print(f"\n[fight-fatigue] Written: {path}")


if __name__ == "__main__":
    main()
