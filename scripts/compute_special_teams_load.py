"""compute_special_teams_load — Feature 3.8 driver script.

Derives per-(game, player) PP/PK seconds from the PBP + shifts tables in
the DataStore, joins with the schedule for ``game_date``, and computes
rolling 5-game special-teams accumulation + weighted intensity score per
player.

Per-game PPTOI/SHTOI is built using the same shift × strength-event overlap
logic that ``PlayerStatsAggregator._compute_pptoi_shtoi`` uses for season
totals — only here we keep ``game_id`` in the group-by so we get a row per
game.

Usage::

    uv run python scripts/compute_special_teams_load.py
    uv run python scripts/compute_special_teams_load.py --seasons 2023 2024
    uv run python scripts/compute_special_teams_load.py --force
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
from models.special_teams_load import (
    SpecialTeamsLoadTracker,
    write_special_teams_load,
)


DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
SCHEDULE_SUBDIR = "schedule"
ST_LOAD_SUBDIR  = "special_teams_load"


# Batch size for the per-game PP/PK overlap join. The shifts × pp_pk
# cross-join is bounded by *shifts_in_batch × pp_pk_in_batch*, both of
# which scale linearly with the number of games in the chunk. 200 games
# keeps each intermediate frame around 1.5M rows on the 2 GB VPS even
# after the .filter() pre-pass on event timing. Tuning lever, not a
# correctness knob — final aggregate is identical to a single-pass run.
_GAME_BATCH = 200


def _per_game_pptoi_shtoi(
    pbp: pl.DataFrame, shifts: pl.DataFrame
) -> pl.DataFrame:
    """Per-(game, player, team) PPTOI and SHTOI in seconds.

    Mirrors PlayerStatsAggregator._compute_pptoi_shtoi but keeps game_id
    in the group-by so callers get one row per game. Processes games in
    fixed-size chunks (see ``_GAME_BATCH``) to bound peak memory — the
    shifts × pp_pk overlap join is the hottest spot in the whole Phase 3
    pipeline and previously OOM-killed the 2 GB VPS at single-season
    scale.
    """
    empty_cols = {
        "game_id":    pl.Int64,
        "player_id":  pl.Int64,
        "team_id":    pl.Int64,
        "pptoi_secs": pl.Int64,
        "shtoi_secs": pl.Int64,
    }
    empty = pl.DataFrame(schema=empty_cols)
    if "home_team_id" not in pbp.columns or len(pbp) == 0 or len(shifts) == 0:
        return empty

    game_teams_all = (
        pbp.filter(pl.col("home_team_id").is_not_null())
        .select(["game_id", "home_team_id", "away_team_id"])
        .unique(subset=["game_id"])
    )
    if len(game_teams_all) == 0:
        return empty

    pp_pk_all = (
        pbp.filter(pl.col("strength").is_in(["pp", "pk"]))
        .with_columns(
            ((pl.col("period") - 1) * 1200 + pl.col("time_in_period_secs"))
            .alias("event_gsecs")
        )
        .select(["game_id", "event_gsecs", "strength"])
    )
    if len(pp_pk_all) == 0:
        return empty

    # Slim shifts to the 5 columns the join needs.
    shifts_slim_all = shifts.select([
        "game_id", "player_id", "team_id",
        "game_seconds_start", "game_seconds_end",
    ])

    # Process games in batches. Each batch produces its own small per-
    # (game, player, team) aggregate; we concat at the end. Memory is
    # bounded by the size of one batch's cross-join, not the season.
    all_gids = sorted(game_teams_all["game_id"].to_list())
    aggregates: list[pl.DataFrame] = []

    for batch_start in range(0, len(all_gids), _GAME_BATCH):
        batch = all_gids[batch_start: batch_start + _GAME_BATCH]
        if not batch:
            break

        game_teams = game_teams_all.filter(pl.col("game_id").is_in(batch))
        pp_pk      = pp_pk_all.filter(pl.col("game_id").is_in(batch))
        if len(pp_pk) == 0:
            continue
        shifts_slim = shifts_slim_all.filter(pl.col("game_id").is_in(batch))
        if len(shifts_slim) == 0:
            continue

        shifts_with_teams = shifts_slim.join(game_teams, on="game_id", how="left")
        overlapping = (
            shifts_with_teams.join(pp_pk, on="game_id", how="inner")
            .filter(
                (pl.col("event_gsecs") >= pl.col("game_seconds_start"))
                & (pl.col("event_gsecs") <= pl.col("game_seconds_end"))
            )
        )
        if len(overlapping) == 0:
            continue

        overlapping = overlapping.with_columns(
            pl.when(
                ((pl.col("strength") == "pp") & (pl.col("team_id") == pl.col("home_team_id")))
                | ((pl.col("strength") == "pk") & (pl.col("team_id") == pl.col("away_team_id")))
            )
            .then(pl.lit("pp"))
            .when(
                ((pl.col("strength") == "pk") & (pl.col("team_id") == pl.col("home_team_id")))
                | ((pl.col("strength") == "pp") & (pl.col("team_id") == pl.col("away_team_id")))
            )
            .then(pl.lit("pk"))
            .otherwise(pl.lit("ev"))
            .alias("shift_type")
        )

        # One classification per unique shift (sort pp/pk before ev so ST wins).
        shift_classified = (
            overlapping
            .sort(["player_id", "game_id", "game_seconds_start", "shift_type"])
            .unique(
                subset=["player_id", "game_id", "game_seconds_start"],
                keep="first",
            )
            .with_columns(
                (pl.col("game_seconds_end") - pl.col("game_seconds_start"))
                .alias("shift_dur")
            )
        )

        grouped = (
            shift_classified
            .group_by(["game_id", "player_id", "team_id", "shift_type"])
            .agg(pl.col("shift_dur").sum().alias("secs"))
            .pivot(
                index=["game_id", "player_id", "team_id"],
                on="shift_type",
                values="secs",
                aggregate_function="sum",
            )
        )
        # Pivot may omit columns when no rows for that strength existed.
        for col in ("pp", "pk"):
            if col not in grouped.columns:
                grouped = grouped.with_columns(pl.lit(0).cast(pl.Int64).alias(col))
        grouped = grouped.with_columns(
            pl.col("pp").fill_null(0).cast(pl.Int64).alias("pptoi_secs"),
            pl.col("pk").fill_null(0).cast(pl.Int64).alias("shtoi_secs"),
        )
        aggregates.append(
            grouped.select(["game_id", "player_id", "team_id",
                            "pptoi_secs", "shtoi_secs"])
        )

    if not aggregates:
        return empty
    return pl.concat(aggregates)


def _load_st_with_date(args) -> pl.DataFrame:
    sched_dir = args.data_dir / SCHEDULE_SUBDIR
    sched_path = latest_schedule_parquet(sched_dir)
    if sched_path is None:
        print(f"[st-load] No schedule parquet found in {sched_dir}.")
        sys.exit(1)
    print(f"[st-load] Loading schedule: {sched_path}")
    schedule = pl.read_parquet(sched_path).select(["game_id", "game_date"])

    # Lazy + column-projected loads. Avoids materialising ~40 PBP columns and
    # the full shift schema when _per_game_pptoi_shtoi only needs 6 + 5
    # respectively. On the 2 GB VPS this is the single biggest peak-memory
    # reduction in the Phase 3 pipeline (~600-900 MB drop, multi-season).
    raw = args.data_dir / "raw"
    if args.seasons:
        seasons = args.seasons
    else:
        seasons = sorted(int(p.stem.split("_", 1)[1]) for p in raw.glob("pbp_*.parquet"))
    pbp_paths    = [raw / f"pbp_{s}.parquet"    for s in seasons]
    shift_paths  = [raw / f"shifts_{s}.parquet" for s in seasons]
    pbp_paths   = [p for p in pbp_paths   if p.exists()]
    shift_paths = [p for p in shift_paths if p.exists()]
    if not pbp_paths or not shift_paths:
        print("[st-load] DataStore has no PBP or shifts. Run `ingest` first.")
        sys.exit(1)
    pbp_df = (
        pl.scan_parquet(pbp_paths)
        .select(["game_id", "home_team_id", "away_team_id", "strength", "period", "time_in_period_secs"])
        .collect(streaming=True)
    )
    shifts_df = (
        pl.scan_parquet(shift_paths)
        .select(["game_id", "player_id", "team_id", "game_seconds_start", "game_seconds_end"])
        .collect(streaming=True)
    )

    if len(pbp_df) == 0 or len(shifts_df) == 0:
        print("[st-load] DataStore has no PBP or shifts. Run `ingest` first.")
        sys.exit(1)

    st_df = _per_game_pptoi_shtoi(pbp_df, shifts_df)
    joined = st_df.join(schedule, on="game_id", how="inner")
    if len(joined) == 0:
        print("[st-load] No ST rows joined to scheduled games.")
        sys.exit(1)
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-player rolling special-teams load (Feature 3.8)."
    )
    parser.add_argument("--seasons", nargs="*", type=int, default=[], metavar="YEAR")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    st_joined = _load_st_with_date(args)
    print(f"  {len(st_joined):,} (game, player) ST rows")

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir = args.data_dir / ST_LOAD_SUBDIR
    out_path = out_dir / f"special_teams_load_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[st-load] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[st-load] Computing ST load as of {as_of}…")
    result = SpecialTeamsLoadTracker().compute(st_joined)
    n_rows = len(result)
    print(f"  {n_rows:,} player-game rows produced.")

    if n_rows > 0:
        top = (
            result.sort("st_intensity_score", descending=True).head(10)
        )
        print("\n  Top 10 ST-intensity-score rows:")
        print(f"  {'Player':<10}  {'Date':<10}  {'PP5s':>5}  {'PK5s':>5}  {'Score':>8}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*5}  {'─'*5}  {'─'*8}")
        for row in top.to_dicts():
            print(
                f"  {row['player_id']:<10}  {row['game_date']:<10}  "
                f"{row['pptoi_5game_sum_secs']:>5d}  "
                f"{row['shtoi_5game_sum_secs']:>5d}  "
                f"{row['st_intensity_score']:>8.1f}"
            )

    path = write_special_teams_load(result, out_dir, as_of)
    print(f"\n[st-load] Written: {path}")


if __name__ == "__main__":
    main()
