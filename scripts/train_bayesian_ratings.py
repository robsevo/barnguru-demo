"""Train Bayesian Player Rating System — Features 2.9 + 2.10.

Requires (per season):
  - Player game logs / xGF: ~/.gretzky/data/shots/shots_{season}.parquet
    (or a pre-aggregated player_game_xgf_{season}.parquet)
  - RAPM parquets:          ~/.gretzky/data/rapm/rapm_{season}.parquet (optional)

The script:
  1. Aggregates per-player xGF/60 from shot data (or loads pre-agg if present)
  2. Fits BayesianPlayerRating conjugate model per season
  3. Optionally exports SequentialBayesianUpdater state (all game-by-game updates)
  4. Saves parquets to ~/.gretzky/data/bayes_ratings/

Usage::

    uv run python scripts/train_bayesian_ratings.py
    uv run python scripts/train_bayesian_ratings.py --seasons 2024 2025
    uv run python scripts/train_bayesian_ratings.py --seasons 2025 --force
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

_DEFAULT_DATA_DIR = Path.home() / ".gretzky" / "data"


def _current_nhl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


def _default_seasons() -> list[int]:
    cur = _current_nhl_season()
    return list(range(2023, cur + 1))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _aggregate_xgf_from_shots(shots_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-player xGF/60 from shot-level data.

    Args:
        shots_df: MoneyPuck shots parquet with columns:
            shooter_id, shooter_name, xGoal, time (seconds into period),
            is_home (0/1), game_id, situation.

    Returns:
        player_game_df: one row per player with columns:
            player_id, player_name, xgf_per60, n_games, xgf_std.
    """
    required = {"shooter_id", "xGoal", "game_id"}
    missing = required - set(shots_df.columns)
    if missing:
        warnings.warn(
            f"shots_df missing columns {sorted(missing)} — using available data.",
            UserWarning,
            stacklevel=2,
        )

    # Filter to EV only if situation column exists
    if "situation" in shots_df.columns:
        shots_df = shots_df.filter(pl.col("situation") == "5on5")

    # Normalise column names — our parquet uses snake_case; MoneyPuck raw CSV uses camelCase
    xgoal_col = "x_goal" if "x_goal" in shots_df.columns else "xGoal"

    # Player identifier: prefer integer shooter_id; fall back to shooter_name
    if (
        "shooter_id" in shots_df.columns
        and shots_df["shooter_id"].dtype != pl.Null
        and shots_df["shooter_id"].null_count() < len(shots_df)
    ):
        player_col = "shooter_id"
    elif "shooter_name" in shots_df.columns:
        player_col = "shooter_name"
    else:
        return pl.DataFrame()

    optional_name = ["shooter_name"] if (
        "shooter_name" in shots_df.columns and player_col != "shooter_name"
    ) else []

    group_cols = [player_col, "game_id"]

    game_level = (
        shots_df
        .filter(pl.col(player_col).is_not_null())
        .group_by(group_cols + optional_name)
        .agg(pl.col(xgoal_col).sum().alias("xgf_game"))
    )

    # Per-player season summary
    agg_exprs = [
        pl.col("xgf_game").mean().alias("xgf_per60"),   # approx xGF/game ≈ /60 min at EV
        pl.col("xgf_game").std().fill_null(0.0).alias("xgf_std"),
        pl.col("game_id").n_unique().alias("n_games"),
    ]
    if optional_name:
        agg_exprs.append(pl.col("shooter_name").first().alias("player_name"))

    player_agg = (
        game_level
        .group_by(player_col)
        .agg(agg_exprs)
        .rename({player_col: "player_id"})
        .filter(pl.col("n_games") >= 1)
    )

    # When player_col was shooter_name, player_id is a string — convert to stable int hash
    if player_agg["player_id"].dtype in (pl.Utf8, pl.String):
        player_agg = player_agg.with_columns(
            pl.col("player_id").map_elements(
                lambda s: abs(hash(s)) % (2 ** 31), return_dtype=pl.Int64
            ).alias("player_id"),
            pl.col("player_id").alias("player_name"),
        )
    elif "player_name" not in player_agg.columns:
        player_agg = player_agg.with_columns(
            pl.lit("").alias("player_name")
        )

    return player_agg.with_columns(
        pl.col("xgf_per60").fill_nan(0.0),
        pl.col("xgf_std").fill_nan(0.0),
    )


def _load_season(
    data_dir: Path,
    season: int,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load player xGF data and optional RAPM for one season.

    Returns (player_game_df or None, rapm_df or None).
    """
    from models.rapm_model import DataMissingWarning

    # Prefer pre-aggregated
    preagg_path = data_dir / "bayes_ratings" / f"player_game_xgf_{season}.parquet"
    shots_path  = data_dir / "shots"         / f"shots_{season}.parquet"
    rapm_path   = data_dir / "rapm"          / f"rapm_{season}.parquet"

    player_df: pl.DataFrame | None = None

    if preagg_path.exists():
        player_df = pl.read_parquet(preagg_path)
        print(f"  Season {season}: loaded pre-aggregated ({len(player_df)} players)")
    elif shots_path.exists():
        shots_df = pl.read_parquet(shots_path)
        # Regular season only
        if "is_playoff" in shots_df.columns:
            shots_df = shots_df.filter(pl.col("is_playoff") == False)  # noqa: E712
        player_df = _aggregate_xgf_from_shots(shots_df)
        print(f"  Season {season}: aggregated from shots ({len(player_df)} players)")
    else:
        warnings.warn(
            f"Season {season}: no shot data at {shots_path} or pre-agg at {preagg_path}.",
            DataMissingWarning,
            stacklevel=2,
        )

    rapm_df: pl.DataFrame | None = None
    if rapm_path.exists():
        rapm_df = pl.read_parquet(rapm_path)

    return player_df, rapm_df


def _print_top_players(ratings_df: pl.DataFrame, season: int, n: int = 10) -> None:
    """Print top N players by posterior_mean with credible intervals."""
    print(f"\n  ── Season {season} Top Players (posterior_mean xGF/60) ──")
    top = ratings_df.head(n)
    for r in top.to_dicts():
        name = r.get("player_name") or str(r["player_id"])
        print(
            f"    {name:<22}  "
            f"μ={r['posterior_mean']:.3f}  "
            f"σ={r['posterior_sigma']:.3f}  "
            f"CI=[{r['ci_lower_95']:.2f}, {r['ci_upper_95']:.2f}]  "
            f"GP={r['n_games']}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Bayesian Player Rating System (Features 2.9 + 2.10).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=_default_seasons(),
        metavar="YEAR",
        help="Season years to process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output parquets already exist.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        metavar="PATH",
        help="Root data directory.",
    )
    args = parser.parse_args()

    data_dir      = args.data_dir
    target_seasons = sorted(args.seasons)
    bayes_dir     = data_dir / "bayes_ratings"
    bayes_dir.mkdir(parents=True, exist_ok=True)

    from models.bayesian_player_rating import (
        BayesianPlayerRating,
        SequentialBayesianUpdater,
        write_player_ratings,
    )
    from models.rapm_model import DataMissingWarning

    # Check existing outputs
    if not args.force:
        existing = [
            s for s in target_seasons
            if (bayes_dir / f"player_ratings_{s}.parquet").exists()
        ]
        if existing:
            print(f"Outputs already exist for seasons {existing}. Use --force to retrain.")
            if len(existing) == len(target_seasons):
                return

    print(f"\nBayesian Player Rating — seasons {target_seasons}\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for season in target_seasons:
            out_path = bayes_dir / f"player_ratings_{season}.parquet"
            if out_path.exists() and not args.force:
                print(f"  Season {season}: output exists, skipping.")
                continue

            player_df, rapm_df = _load_season(data_dir, season)
            if player_df is None or player_df.is_empty():
                print(f"  Season {season}: skipping (no data).")
                continue

            # Feature 2.9 — batch Bayesian ratings
            model = BayesianPlayerRating()
            ratings = model.fit_season(
                player_game_df = player_df,
                season         = season,
                rapm_df        = rapm_df,
            )
            saved = write_player_ratings(ratings, bayes_dir, season)
            print(f"  Saved player_ratings_{season}.parquet  ({len(ratings)} players) → {saved}")
            _print_top_players(ratings, season)

            # Feature 2.10 — sequential updater state
            updater = SequentialBayesianUpdater()
            # Replay all players' season data game-by-game via batch update
            for row in player_df.to_dicts():
                pid   = int(row["player_id"])
                name  = str(row.get("player_name") or "")
                xgf   = float(row.get("xgf_per60") or 0.0)
                n     = int(row.get("n_games") or 0)
                std   = float(row.get("xgf_std") or 0.0)
                if n < 1:
                    continue
                import numpy as np
                obs = np.full(n, xgf)
                updater.update_batch(
                    player_id    = pid,
                    player_name  = name,
                    observations = obs,
                    obs_sigma    = max(std, 0.5) if std > 0 else None,
                )

            updater_path = bayes_dir / f"updater_state_{season}.pkl"
            updater.save(updater_path)
            print(f"  Saved updater_state_{season}.pkl")

    for w in caught:
        if issubclass(w.category, DataMissingWarning):
            print(f"  [WARN] {w.message}", file=sys.stderr)

    print(f"\nDone.  Outputs in {bayes_dir}")


if __name__ == "__main__":
    main()
