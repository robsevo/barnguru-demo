"""compute_injury_status — Feature 3.13 driver script.

Stitches together the ESPN injury history (every parquet
``injuries_*.parquet`` written by ``data.injury_sync``) plus the player's
real games played (joined via NHL player IDs from the roster cache), and
writes a one-row-per-player status snapshot containing current status,
availability probability, return date, games-since-return, and
return-from-injury rust factor.

Usage::

    uv run python scripts/gretzky.py injury-status
    uv run python scripts/gretzky.py injury-status -- --date 2026-05-17
    uv run python scripts/gretzky.py injury-status -- --force
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
from models.injury_status_integrator import (
    InjuryStatusIntegrator,
    write_injury_status,
)


DEFAULT_DATA_DIR = Path(
    os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data"))
)
INJURY_STATUS_SUBDIR = "injury_status"
SCHEDULE_SUBDIR      = "schedule"


def _injury_history(store: DataStore) -> pl.DataFrame:
    """Load every cached ESPN injury parquet, join to NHL player IDs.

    The injury feed uses ESPN athlete IDs; the rest of the GRTZKY pipeline
    keys off NHL ``player_id``. We bridge via the roster cache, matching
    on ``(team_code, player_name)`` — the ESPN feed always carries both.
    """
    raw = store.injuries(date=None)
    if len(raw) == 0:
        return pl.DataFrame(
            schema={"player_id": pl.Int64, "observed_date": pl.Utf8, "status": pl.Utf8}
        )

    # Pull NHL roster for the name→player_id bridge.
    roster = store.roster()
    if len(roster) == 0 or "player_id" not in roster.columns:
        return pl.DataFrame(
            schema={"player_id": pl.Int64, "observed_date": pl.Utf8, "status": pl.Utf8}
        )

    # Build a (team, full_name) → player_id lookup, lowercase for safety.
    name_col = "full_name" if "full_name" in roster.columns else "player_name"
    if name_col not in roster.columns:
        return pl.DataFrame(
            schema={"player_id": pl.Int64, "observed_date": pl.Utf8, "status": pl.Utf8}
        )

    bridge = (
        roster.select(["player_id", "team_abbrev", name_col])
        .rename({"team_abbrev": "team_code", name_col: "player_name"})
        .with_columns(
            pl.col("team_code").cast(pl.Utf8).str.to_uppercase(),
            pl.col("player_name").cast(pl.Utf8).str.to_lowercase(),
        )
        .unique(subset=["team_code", "player_name"])
    )

    injuries = (
        raw.select(["player_name", "team_code", "status", "fetched_at"])
        .with_columns(
            pl.col("team_code").cast(pl.Utf8).str.to_uppercase(),
            pl.col("player_name").cast(pl.Utf8).str.to_lowercase(),
            pl.col("fetched_at").cast(pl.Utf8).str.slice(0, 10).alias("observed_date"),
        )
        .drop("fetched_at")
        .join(bridge, on=["team_code", "player_name"], how="inner")
        .select(["player_id", "observed_date", "status"])
        .unique()
    )
    return injuries


def _player_games(store: DataStore) -> pl.DataFrame:
    """Per-player game-date rows, derived from PBP + schedule."""
    sched_path = latest_schedule_parquet(store._raw.parent / SCHEDULE_SUBDIR) \
        if hasattr(store, "_raw") else None
    if sched_path is None:
        return pl.DataFrame(schema={"player_id": pl.Int64, "game_date": pl.Utf8})

    schedule = pl.read_parquet(sched_path).select(["game_id", "game_date"])
    pbp = store.pbp()
    if len(pbp) == 0 or "committed_by_id" not in pbp.columns:
        return pl.DataFrame(schema={"player_id": pl.Int64, "game_date": pl.Utf8})

    # Distinct (player, game) appearances. event-owner skater fields vary by
    # event type; committed_by_id catches penalties; for general appearances
    # we use the union of event_player_1_id / event_player_2_id when present.
    pid_cols = [c for c in ("event_player_1_id", "event_player_2_id",
                            "committed_by_id") if c in pbp.columns]
    if not pid_cols:
        return pl.DataFrame(schema={"player_id": pl.Int64, "game_date": pl.Utf8})

    frames = []
    for col in pid_cols:
        frames.append(
            pbp.select(["game_id", pl.col(col).alias("player_id")])
            .filter(pl.col("player_id").is_not_null())
        )
    stacked = pl.concat(frames).unique()
    return (
        stacked.join(schedule, on="game_id", how="inner")
        .select(["player_id", "game_date"])
        .unique()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-player injury status + rust factor (Feature 3.13)."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None,
                        help="as-of date. Default: today (UTC).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir  = args.data_dir / INJURY_STATUS_SUBDIR
    out_path = out_dir / f"injury_status_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[injury-status] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[injury-status] Computing injury status as of {as_of}…")
    with DataStore(args.data_dir) as store:
        history = _injury_history(store)
        games   = _player_games(store)

    print(f"  {len(history):,} injury observation rows")
    print(f"  {len(games):,}    player-game rows")

    result = InjuryStatusIntegrator().compute(history, games, as_of_date=as_of)
    n_rows = len(result)
    print(f"  {n_rows:,} player rows produced.")

    if n_rows > 0:
        rusty = (
            result.filter(pl.col("rust_factor") < 1.0)
            .sort("rust_factor")
            .head(10)
        )
        if len(rusty) > 0:
            print("\n  Players currently rusty (low → high rust_factor):")
            print(f"  {'Player':<10}  {'Status':<8}  {'Returned':<10}  "
                  f"{'GP':>3}  {'Rust':>5}")
            print(f"  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*3}  {'─'*5}")
            for r in rusty.to_dicts():
                print(
                    f"  {r['player_id']:<10}  {r['status']:<8}  "
                    f"{r['return_date']:<10}  {r['games_since_return']:>3d}  "
                    f"{r['rust_factor']:>5.2f}"
                )

    path = write_injury_status(result, out_dir, as_of)
    print(f"\n[injury-status] Written: {path}")


if __name__ == "__main__":
    main()
