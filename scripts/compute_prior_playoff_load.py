"""compute_prior_playoff_load — Feature 3.15 driver script.

Derives per-player ``prior_playoff_games`` (last spring's playoff GP),
``prior_playoff_round`` (rough proxy = ``ceil(team_playoff_games / 7)``,
capped at 4), and ``games_played_this_season`` (current-season GP up to
``as_of_date``) from cached PBP + schedule, then writes the playoff-load
penalty per player.

Inputs needed in DataStore:
- ``store.pbp()`` with skater-id columns
- ``schedule_*.parquet`` containing ``game_id``, ``game_date``, ``game_type``,
  and the season the game belongs to (derived here from the season the
  game_date falls in: Oct..Apr = season ending in Apr-year, etc.).

Usage::

    uv run python scripts/gretzky.py prior-playoff-load
    uv run python scripts/gretzky.py prior-playoff-load -- --date 2026-05-17
    uv run python scripts/gretzky.py prior-playoff-load -- --force
"""

from __future__ import annotations

import argparse
import math
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
from models.prior_playoff_load import (
    PriorPlayoffLoad,
    write_prior_playoff_load,
)


DEFAULT_DATA_DIR = Path(
    os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data"))
)
PRIOR_PLAYOFF_LOAD_SUBDIR = "prior_playoff_load"
SCHEDULE_SUBDIR           = "schedule"

GAME_TYPE_PLAYOFFS = 3
GAME_TYPE_REGULAR  = 2


def _season_ending_year(d: str) -> int:
    """Map a YYYY-MM-DD game date to the season's *ending* year.

    Aug–Dec → season ending year+1 (the upcoming spring).
    Jan–Jul → season ending current year.
    """
    dt = datetime.strptime(d, "%Y-%m-%d").date()
    return dt.year + 1 if dt.month >= 8 else dt.year


def _per_player_per_game(pbp: pl.DataFrame) -> pl.DataFrame:
    """Distinct (player_id, game_id) appearances from PBP."""
    pid_cols = [c for c in ("event_player_1_id", "event_player_2_id",
                            "committed_by_id") if c in pbp.columns]
    if not pid_cols:
        return pl.DataFrame(schema={"player_id": pl.Int64, "game_id": pl.Int64})
    frames = []
    for col in pid_cols:
        frames.append(
            pbp.select(["game_id", pl.col(col).alias("player_id")])
            .filter(pl.col("player_id").is_not_null())
        )
    return pl.concat(frames).unique()


def _build_load_df(store: DataStore, args, as_of_iso: str) -> pl.DataFrame:
    sched_dir  = args.data_dir / SCHEDULE_SUBDIR
    sched_path = latest_schedule_parquet(sched_dir)
    if sched_path is None:
        print(f"[prior-playoff-load] No schedule parquet in {sched_dir}.")
        sys.exit(1)
    schedule = pl.read_parquet(sched_path).select(
        ["game_id", "game_date", "game_type"]
    )

    pbp = store.pbp()
    if len(pbp) == 0:
        print("[prior-playoff-load] DataStore has no PBP. Run `ingest` first.")
        sys.exit(1)

    appearances = _per_player_per_game(pbp).join(
        schedule, on="game_id", how="inner"
    )
    if len(appearances) == 0:
        print("[prior-playoff-load] No player appearances joined to schedule.")
        sys.exit(1)

    # Season-end-year per appearance row
    appearances = appearances.with_columns(
        pl.col("game_date").map_elements(_season_ending_year, return_dtype=pl.Int64)
          .alias("season_end")
    )

    cur_season = _season_ending_year(as_of_iso)
    prev_season = cur_season - 1

    # Prior-season playoff GP per player
    prior_pp = (
        appearances.filter(
            (pl.col("season_end") == prev_season)
            & (pl.col("game_type") == GAME_TYPE_PLAYOFFS)
        )
        .group_by("player_id")
        .agg(pl.len().alias("prior_playoff_games"))
    )

    # Prior-season team playoff GP per player → infer round
    # (max team playoff games among the player's playoff appearances → round)
    if "event_owner_team_id" in pbp.columns:
        team_gp = (
            pbp.select(["game_id", "event_owner_team_id"])
            .filter(pl.col("event_owner_team_id").is_not_null())
            .unique()
            .join(schedule, on="game_id", how="inner")
            .filter(pl.col("game_type") == GAME_TYPE_PLAYOFFS)
            .with_columns(
                pl.col("game_date").map_elements(_season_ending_year,
                                                 return_dtype=pl.Int64)
                  .alias("season_end")
            )
            .filter(pl.col("season_end") == prev_season)
            .group_by("event_owner_team_id")
            .agg(pl.len().alias("team_playoff_games"))
        )
        # Map player → team via prior-season appearances
        player_team = (
            pbp.select([
                pl.col("event_player_1_id").alias("player_id"),
                pl.col("event_owner_team_id").alias("team_id"),
                "game_id",
            ])
            .filter(pl.col("player_id").is_not_null() & pl.col("team_id").is_not_null())
            .unique()
            .join(schedule, on="game_id", how="inner")
            .filter(pl.col("game_type") == GAME_TYPE_PLAYOFFS)
            .with_columns(
                pl.col("game_date").map_elements(_season_ending_year,
                                                 return_dtype=pl.Int64)
                  .alias("season_end")
            )
            .filter(pl.col("season_end") == prev_season)
            .select(["player_id", "team_id"])
            .unique()
        )
        round_lookup = (
            player_team.join(
                team_gp.rename({"event_owner_team_id": "team_id"}),
                on="team_id", how="left",
            )
            .group_by("player_id")
            .agg(pl.col("team_playoff_games").max().alias("team_gp"))
            .with_columns(
                pl.col("team_gp")
                  .map_elements(lambda g: 0 if g is None or g <= 0
                                else min(4, math.ceil(int(g) / 7)),
                                return_dtype=pl.Int64)
                  .alias("prior_playoff_round")
            )
            .select(["player_id", "prior_playoff_round"])
        )
    else:
        round_lookup = pl.DataFrame(
            schema={"player_id": pl.Int64, "prior_playoff_round": pl.Int64}
        )

    # Current-season GP per player (regular season, ≤ as_of)
    cur_gp = (
        appearances.filter(
            (pl.col("season_end") == cur_season)
            & (pl.col("game_type") == GAME_TYPE_REGULAR)
            & (pl.col("game_date") <= as_of_iso)
        )
        .group_by("player_id")
        .agg(pl.len().alias("games_played_this_season"))
    )

    # Union of players to score: anyone with prior playoff GP OR current GP
    all_pids = pl.concat([
        prior_pp.select("player_id"),
        cur_gp.select("player_id"),
    ]).unique()

    return (
        all_pids
        .join(prior_pp, on="player_id", how="left")
        .join(round_lookup, on="player_id", how="left")
        .join(cur_gp, on="player_id", how="left")
        .with_columns(
            pl.col("prior_playoff_games").fill_null(0),
            pl.col("prior_playoff_round").fill_null(0),
            pl.col("games_played_this_season").fill_null(0),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-player prior-playoff-load penalty (Feature 3.15)."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None,
                        help="as-of date. Default: today (UTC).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir  = args.data_dir / PRIOR_PLAYOFF_LOAD_SUBDIR
    out_path = out_dir / f"prior_playoff_load_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[prior-playoff-load] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[prior-playoff-load] Computing playoff load as of {as_of}…")
    with DataStore(args.data_dir) as store:
        load_df = _build_load_df(store, args, as_of)
    print(f"  {len(load_df):,} player rows from PBP × schedule")

    result = PriorPlayoffLoad().compute(load_df, as_of_date=as_of)
    n_rows = len(result)
    print(f"  {n_rows:,} player rows produced.")

    if n_rows > 0:
        top = result.sort("playoff_load_penalty", descending=True).head(10)
        print("\n  Top 10 current playoff-load penalties:")
        print(f"  {'Player':<10}  {'PriorGP':>7}  {'Rd':>2}  {'GP':>3}  "
              f"{'Base':>6}  {'Now':>6}")
        print(f"  {'─'*10}  {'─'*7}  {'─'*2}  {'─'*3}  {'─'*6}  {'─'*6}")
        for r in top.to_dicts():
            print(
                f"  {r['player_id']:<10}  {r['prior_playoff_games']:>7d}  "
                f"{r['prior_playoff_round']:>2d}  "
                f"{r['games_played_this_season']:>3d}  "
                f"{r['base_playoff_penalty']:>6.3f}  "
                f"{r['playoff_load_penalty']:>6.3f}"
            )

    path = write_prior_playoff_load(result, out_dir, as_of)
    print(f"\n[prior-playoff-load] Written: {path}")


if __name__ == "__main__":
    main()
