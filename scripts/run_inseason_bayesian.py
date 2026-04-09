#!/usr/bin/env python3
"""Run in-season Bayesian blend pipeline (Feature 2.12).

Blends historical Bayesian posteriors (from Feature 2.9/2.10) with current-
season observations using a linearly growing weight function. The blend weight
grows from 0.25 at game 10 to 0.73 at game 82, ensuring historical context
dominates early in the season while in-season performance takes over by the end.

Requires:
    ~/.gretzky/data/rapm/rapm_{season}.parquet       (historical prior source)
    ~/.gretzky/data/bayesian/bayesian_{season}.parquet (optional pre-fit priors)
    ~/.gretzky/data/shots/shots_{season}.parquet      (current-season observations)

Usage::

    uv run python scripts/run_inseason_bayesian.py
    uv run python scripts/run_inseason_bayesian.py --seasons 2025
    uv run python scripts/run_inseason_bayesian.py --force

Outputs:
    ~/.gretzky/data/inseason/inseason_blend_{season}.parquet
    ~/.gretzky/data/inseason/inseason_pipeline_{season}.pkl
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
import os

import polars as pl

_DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))


def _current_nhl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


def _default_seasons() -> list[int]:
    cur = _current_nhl_season()
    return list(range(2023, cur + 1))


def _load_bayes_priors(data_dir: Path, season: int) -> pl.DataFrame:
    """Load Feature 2.9 Bayesian ratings as historical priors."""
    from models.rapm_model import DataMissingWarning

    path = data_dir / "bayes_ratings" / f"player_ratings_{season}.parquet"
    if not path.exists():
        warnings.warn(
            f"Season {season}: Bayesian ratings not found at {path}. "
            "Run: uv run python scripts/gretzky.py train-bayesian",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame()
    return pl.read_parquet(path)


def _load_shots(data_dir: Path, season: int) -> pl.DataFrame:
    from models.rapm_model import DataMissingWarning

    path = data_dir / "shots" / f"shots_{season}.parquet"
    if not path.exists():
        warnings.warn(
            f"Season {season}: shots file not found at {path}.",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame()
    return pl.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run in-season Bayesian blend pipeline (Feature 2.12).",
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
                        help="Re-run even if output already exists.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        metavar="PATH",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    output_dir = data_dir / "inseason"
    output_dir.mkdir(parents=True, exist_ok=True)

    from models.inseason_bayesian import InSeasonBayesianPipeline
    from models.rapm_model import DataMissingWarning

    seasons: list[int] = sorted(args.seasons)
    processed = 0

    for season in seasons:
        blend_path    = output_dir / f"inseason_blend_{season}.parquet"
        pipeline_path = output_dir / f"inseason_pipeline_{season}.pkl"

        if blend_path.exists() and not args.force:
            print(f"  Season {season}: already exists, skipping (use --force)")
            continue

        print(f"\n  Season {season}:")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bayes_df = _load_bayes_priors(data_dir, season)
            shots_df = _load_shots(data_dir, season)

        for w in caught:
            print(f"    [WARN] {w.message}", file=sys.stderr)

        if len(bayes_df) == 0:
            print(f"    No Bayesian prior data; skipping season {season}")
            continue

        # Build pipeline with historical Bayesian ratings as prior
        pipeline = InSeasonBayesianPipeline(historical_ratings_df=bayes_df)

        # Feed current-season shot observations if available
        n_obs = 0
        if len(shots_df) > 0 and "x_goal" in shots_df.columns:
            # Aggregate shots into per-player xGF to feed as game observations
            agg_df = (
                shots_df
                .filter(pl.col("shooter_name").is_not_null())
                .group_by("shooter_name")
                .agg(pl.col("x_goal").sum().alias("xgf_per60"))
                .rename({"shooter_name": "player_name"})
                .with_columns(pl.lit(0).cast(pl.Int64).alias("player_id"))
            )
            if len(agg_df) > 0:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pipeline.observe_game(agg_df)
                    n_obs = len(agg_df)
                for w in caught:
                    print(f"    [WARN] {w.message}", file=sys.stderr)

        print(f"    Bayesian prior players: {len(bayes_df)}, shot observations: {n_obs}")

        # Compute blended ratings
        blend_df = pipeline.blended_ratings(season=season)
        blend_df.write_parquet(blend_path)
        print(f"    Blended ratings: {len(blend_df)} players → {blend_path}")

        # Save pipeline state
        pipeline.save(pipeline_path)
        print(f"    Pipeline saved → {pipeline_path}")

        # Print top blended performers
        if "mu_blend" in blend_df.columns and "player_name" in blend_df.columns:
            top = (
                blend_df
                .filter(pl.col("games_played") >= 5)
                .sort("mu_blend", descending=True)
                .head(5)
            )
            print(f"    Top blended (mu_blend, min 5 GP):")
            for row in top.to_dicts():
                name = row.get("player_name", f"id_{row.get('player_id', '?')}")
                mu   = row.get("mu_blend", float("nan"))
                gp   = row.get("games_played", 0)
                w    = row.get("blend_weight", float("nan"))
                print(f"      {name:<22} mu={mu:+.3f}  gp={gp}  w={w:.2f}")

        processed += 1

    print(f"\nDone. {processed}/{len(seasons)} seasons processed.")


if __name__ == "__main__":
    main()
