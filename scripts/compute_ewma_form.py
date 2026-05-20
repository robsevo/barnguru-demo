#!/usr/bin/env python3
"""Compute EWMA form metrics per player (Feature 2.13).

Computes exponentially weighted moving average of xGF/60 and Corsi% per player
across the current season, using λ=0.96 (α=0.04) with a ~17-game half-life.
Flags players as "hot", "cold", or "neutral" based on deviation from seasonal mean.

Requires:
    ~/.gretzky/data/shots/shots_{season}.parquet

Usage::

    uv run python scripts/compute_ewma_form.py
    uv run python scripts/compute_ewma_form.py --seasons 2025
    uv run python scripts/compute_ewma_form.py --force

Outputs:
    ~/.gretzky/data/ewma/ewma_form_{season}.parquet   — per-player latest EWMA snapshot
    ~/.gretzky/data/ewma/ewma_games_{season}.parquet  — per-game EWMA time series
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
import os

import polars as pl

_DEFAULT_DATA_DIR   = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))


def _current_nhl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


def _default_seasons() -> list[int]:
    cur = _current_nhl_season()
    return list(range(2023, cur + 1))


def _load_shots(data_dir: Path, season: int, season_type: str = "regular") -> pl.DataFrame:
    from models.rapm_model import DataMissingWarning

    path = data_dir / "shots" / f"shots_{season}.parquet"
    if not path.exists():
        warnings.warn(
            f"Season {season}: shots file not found at {path}.",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame()
    # Streaming + filter pushdown: regular-season runs drop ~10% playoff rows
    # before materialisation; playoff runs drop ~90% regular rows. Either way,
    # peak per-season RSS is meaningfully lower than the eager read_parquet.
    lf = pl.scan_parquet(path)
    if "is_playoff" in lf.collect_schema().names():
        if season_type == "playoffs":
            lf = lf.filter(pl.col("is_playoff") == True)  # noqa: E712
        elif season_type == "regular":
            lf = lf.filter(pl.col("is_playoff") == False)  # noqa: E712
    return lf.collect(streaming=True)


def _build_player_game_df(shots_df: pl.DataFrame, season: int) -> pl.DataFrame:
    """Aggregate shots into per-player-game xGF/60 and Corsi/60 rows.

    Requires columns: shooter_id, game_id, xgoal, isGoal, event (or similar).
    Returns DataFrame with player_id, game_id, xgf_per60, cf_per60, season.
    """
    # Normalise column names — parquets use snake_case; raw CSV uses camelCase
    xgoal_col = "x_goal" if "x_goal" in shots_df.columns else "xgoal"
    id_col = "shooter_id" if (
        "shooter_id" in shots_df.columns and shots_df["shooter_id"].dtype != pl.Null
    ) else None
    required = {"game_id", xgoal_col}
    if id_col:
        required.add(id_col)
    else:
        required.add("shooter_name")
    if not required.issubset(set(shots_df.columns)):
        missing = required - set(shots_df.columns)
        warnings.warn(
            f"shots_df missing columns: {sorted(missing)}. Cannot compute EWMA.",
            stacklevel=2,
        )
        return pl.DataFrame()

    player_col = id_col or "shooter_name"

    # Aggregate: sum xgoal per (shooter, game); derive /60 assuming 1 min TOI per shot (approx)
    per_game = (
        shots_df
        .filter(pl.col(player_col).is_not_null())
        .group_by([player_col, "game_id"])
        .agg([
            pl.col(xgoal_col).sum().alias("xgf"),
            pl.col(xgoal_col).count().alias("n_shots"),
        ])
        .rename({player_col: "_player_raw"})
        .with_columns([
            # Convert player identifier to stable Int64
            pl.col("_player_raw").map_elements(
                lambda v: int(v) if str(v).lstrip("-").isdigit()
                else abs(hash(str(v))) % (2 ** 31),
                return_dtype=pl.Int64,
            ).alias("player_id"),
            # Approximate per-60: scale by (60 / approx toi); use n_shots as proxy
            (pl.col("xgf") * 60.0 / pl.col("n_shots").clip(1)).alias("xgf_per60"),
            pl.lit(1.0).alias("cf_per60"),  # Corsi requires shift data; placeholder
            pl.lit(season).cast(pl.Int64).alias("season"),
        ])
        .select(["player_id", "game_id", "xgf_per60", "cf_per60", "season"])
        .sort(["player_id", "game_id"])
    )
    return per_game


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute EWMA form metrics per player (Feature 2.13).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=_default_seasons(),
        metavar="YEAR",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-compute even if output already exists.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        metavar="PATH",
    )
    parser.add_argument(
        "--season-type",
        choices=["regular", "playoffs"],
        default="regular",
        help="'regular' writes ewma_form_{season}.parquet (default). "
             "'playoffs' filters to playoff games and writes "
             "ewma_form_{season}_playoffs.parquet.",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    output_dir = data_dir / "ewma"
    output_dir.mkdir(parents=True, exist_ok=True)
    season_type = args.season_type
    out_suffix = "_playoffs" if season_type == "playoffs" else ""

    from models.ewma_model import EWMAFormModel

    seasons: list[int] = sorted(args.seasons)
    ewma_model = EWMAFormModel()
    processed = 0

    for season in seasons:
        summary_path = output_dir / f"ewma_form_{season}{out_suffix}.parquet"
        games_path   = output_dir / f"ewma_games_{season}{out_suffix}.parquet"

        if summary_path.exists() and not args.force:
            print(f"  Season {season}: already exists, skipping (use --force)")
            continue

        print(f"\n  Season {season} ({season_type}):")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shots_df = _load_shots(data_dir, season, season_type=season_type)

        for w in caught:
            print(f"    [WARN] {w.message}", file=sys.stderr)

        if len(shots_df) == 0:
            print(f"    No shot data; skipping season {season}")
            continue

        player_game_df = _build_player_game_df(shots_df, season)
        if len(player_game_df) == 0:
            print(f"    Could not build player-game DataFrame; skipping season {season}")
            continue

        print(f"    Player-game rows: {len(player_game_df)}")

        # Compute per-game EWMA time series
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            games_df = ewma_model.compute(player_game_df, season)

        for w in caught:
            print(f"    [WARN] {w.message}", file=sys.stderr)

        games_df.write_parquet(games_path)
        print(f"    EWMA games: {len(games_df)} rows → {games_path}")

        # Compute latest snapshot (one row per player)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            summary_df = ewma_model.summary(player_game_df, season)

        for w in caught:
            print(f"    [WARN] {w.message}", file=sys.stderr)

        summary_df.write_parquet(summary_path)
        print(f"    EWMA summary: {len(summary_df)} players → {summary_path}")

        # Print form flags
        if "form_flag" in summary_df.columns:
            hot  = summary_df.filter(pl.col("form_flag") == "hot")
            cold = summary_df.filter(pl.col("form_flag") == "cold")
            print(f"    Hot players:  {len(hot)}")
            print(f"    Cold players: {len(cold)}")

            if len(hot) > 0 and "player_id" in hot.columns:
                top_hot = hot.sort("ewma_xgf60", descending=True).head(5)
                print("    Top 5 hot:")
                for row in top_hot.to_dicts():
                    pid  = row["player_id"]
                    xgf  = row.get("ewma_xgf60", float("nan"))
                    gp   = row.get("n_games", "?")
                    print(f"      player_id={pid}  ewma_xgf60={xgf:.3f}  n_games={gp}")

        processed += 1

    print(f"\nDone. {processed}/{len(seasons)} seasons processed.")


if __name__ == "__main__":
    main()
