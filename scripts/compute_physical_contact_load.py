"""compute_physical_contact_load — Feature 3.9 driver script.

Aggregates per-(game, player) hits-taken and blocked-shots from the PBP
table in the DataStore, joins with the schedule for ``game_date``, and
computes rolling 5-game contact-load metrics.

Hits-taken comes from PBP rows with ``event_type == "hit"`` and a
non-null ``hittee_id``. Blocked-shots come from rows with
``shot_result == "blocked"`` and a non-null ``blocker_id``.

Usage::

    uv run python scripts/compute_physical_contact_load.py
    uv run python scripts/compute_physical_contact_load.py --seasons 2023 2024
    uv run python scripts/compute_physical_contact_load.py --force
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
from models.physical_contact_load import (
    PhysicalContactLoadTracker,
    write_physical_contact_load,
)


DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
SCHEDULE_SUBDIR = "schedule"
CONTACT_SUBDIR  = "physical_contact_load"


def _aggregate_contact(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per-(game, player, team) hits_taken + blocked_shots from PBP.

    The current NHL PBP schema doesn't always carry ``hittee_id`` /
    ``blocker_id`` (those are emitted by a newer parser version; older
    ingests omit them). Each component is computed if and only if its
    source column is present, and the function gracefully returns a
    partial frame — composite-FI degrades the contact-load signal to
    zero when this output is missing.
    """
    empty = pl.DataFrame(
        schema={
            "game_id":       pl.Int64,
            "player_id":     pl.Int64,
            "team_id":       pl.Int64,
            "hits_taken":    pl.Int64,
            "blocked_shots": pl.Int64,
        }
    )
    if len(pbp) == 0:
        return empty

    cols = set(pbp.columns)
    have_hittee  = "hittee_id"  in cols
    have_blocker = "blocker_id" in cols
    if not have_hittee and not have_blocker:
        # Neither component recoverable from the current parquet — emit
        # empty frame so the downstream rolling tracker still writes a
        # zero-row parquet rather than crashing the nightly step.
        return empty

    if have_hittee:
        hits = (
            pbp.filter(
                (pl.col("event_type") == "hit") & pl.col("hittee_id").is_not_null()
            )
            .select([
                "game_id",
                pl.col("hittee_id").alias("player_id"),
                # The hittee is on the team being hit, i.e. NOT the event owner.
                pl.when(pl.col("event_owner_team_id") == pl.col("home_team_id"))
                .then(pl.col("away_team_id"))
                .otherwise(pl.col("home_team_id"))
                .alias("team_id"),
            ])
            .group_by(["game_id", "player_id", "team_id"])
            .agg(pl.len().alias("hits_taken"))
        )
    else:
        hits = pl.DataFrame(schema={
            "game_id": pl.Int64, "player_id": pl.Int64,
            "team_id": pl.Int64, "hits_taken": pl.Int64,
        })

    if have_blocker:
        blocks = (
            pbp.filter(
                (pl.col("shot_result") == "blocked") & pl.col("blocker_id").is_not_null()
            )
            .select([
                "game_id",
                pl.col("blocker_id").alias("player_id"),
                # The blocker is on the defending team — opposite the
                # shooter's event_owner_team_id.
                pl.when(pl.col("event_owner_team_id") == pl.col("home_team_id"))
                .then(pl.col("away_team_id"))
                .otherwise(pl.col("home_team_id"))
                .alias("team_id"),
            ])
            .group_by(["game_id", "player_id", "team_id"])
            .agg(pl.len().alias("blocked_shots"))
        )
    else:
        blocks = pl.DataFrame(schema={
            "game_id": pl.Int64, "player_id": pl.Int64,
            "team_id": pl.Int64, "blocked_shots": pl.Int64,
        })

    if len(hits) == 0 and len(blocks) == 0:
        return empty

    merged = hits.join(
        blocks, on=["game_id", "player_id", "team_id"], how="full", coalesce=True
    )
    merged = merged.with_columns(
        pl.col("hits_taken").fill_null(0).cast(pl.Int64),
        pl.col("blocked_shots").fill_null(0).cast(pl.Int64),
    )
    return merged.select(["game_id", "player_id", "team_id", "hits_taken", "blocked_shots"])


def _load_contact_with_date(args) -> pl.DataFrame:
    sched_dir = args.data_dir / SCHEDULE_SUBDIR
    sched_path = latest_schedule_parquet(sched_dir)
    if sched_path is None:
        print(f"[contact-load] No schedule parquet found in {sched_dir}.")
        sys.exit(1)
    print(f"[contact-load] Loading schedule: {sched_path}")
    schedule = pl.read_parquet(sched_path).select(["game_id", "game_date"])

    with DataStore(args.data_dir) as store:
        if args.seasons:
            frames = [store.pbp(season=s) for s in args.seasons]
            frames = [df for df in frames if len(df) > 0]
            pbp_df = pl.concat(frames) if frames else store.pbp()
        else:
            pbp_df = store.pbp()

    if len(pbp_df) == 0:
        print("[contact-load] DataStore has no PBP rows. Run `ingest` first.")
        sys.exit(1)

    contact_df = _aggregate_contact(pbp_df)
    if len(contact_df) == 0:
        cols = set(pbp_df.columns)
        missing = [c for c in ("hittee_id", "blocker_id") if c not in cols]
        if missing:
            print(
                f"[contact-load] PBP parquet is missing {missing}; "
                "writing an empty contact-load parquet so composite-FI can "
                "still run with the contact signal at its no-effect default."
            )
        else:
            print("[contact-load] PBP yielded no hit/block rows.")
        return contact_df.join(schedule.head(0), on="game_id", how="inner")

    joined = contact_df.join(schedule, on="game_id", how="inner")
    if len(joined) == 0:
        print("[contact-load] No contact rows joined to scheduled games.")
        return joined
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-player rolling physical-contact load (Feature 3.9)."
    )
    parser.add_argument("--seasons", nargs="*", type=int, default=[], metavar="YEAR")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    contact_joined = _load_contact_with_date(args)
    print(f"  {len(contact_joined):,} (game, player) contact rows")

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir = args.data_dir / CONTACT_SUBDIR
    out_path = out_dir / f"physical_contact_load_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[contact-load] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[contact-load] Computing contact load as of {as_of}…")
    result = PhysicalContactLoadTracker().compute(contact_joined)
    n_rows = len(result)
    print(f"  {n_rows:,} player-game rows produced.")

    if n_rows > 0:
        top = (
            result.sort("contact_5game_score", descending=True).head(10)
        )
        print("\n  Top 10 5-game contact loads:")
        print(f"  {'Player':<10}  {'Date':<10}  {'Hits5':>5}  {'Blk5':>5}  {'Score':>7}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*5}  {'─'*5}  {'─'*7}")
        for row in top.to_dicts():
            print(
                f"  {row['player_id']:<10}  {row['game_date']:<10}  "
                f"{row['hits_5game_sum']:>5d}  {row['blocks_5game_sum']:>5d}  "
                f"{row['contact_5game_score']:>7.1f}"
            )

    path = write_physical_contact_load(result, out_dir, as_of)
    print(f"\n[contact-load] Written: {path}")


if __name__ == "__main__":
    main()
