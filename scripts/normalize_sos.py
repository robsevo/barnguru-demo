#!/usr/bin/env python3
"""Compute Strength-of-Schedule weights per player-game (Feature 2.15).

Weights each game observation by opponent quality, preventing easy-schedule
mirages from contaminating Bayesian posteriors. High-quality opponents get up
to 2× weight; weak opponents get down to 0.5× weight.

Requires:
    ~/.gretzky/data/rapm/rapm_{season}.parquet   (opponent quality source)
    ~/.gretzky/data/shots/shots_{season}.parquet  (player-game log)

Usage::

    uv run python scripts/normalize_sos.py
    uv run python scripts/normalize_sos.py --seasons 2025
    uv run python scripts/normalize_sos.py --force

Outputs:
    ~/.gretzky/data/sos/sos_weights_{season}.parquet
    ~/.gretzky/data/sos/sos_normalizer_{season}.pkl
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Strength-of-Schedule weights per player-game.",
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
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    output_dir = data_dir / "sos"
    output_dir.mkdir(parents=True, exist_ok=True)

    from models.schedule_normalizer import StrengthOfScheduleNormalizer, write_sos_weights
    from models.rapm_model import DataMissingWarning

    seasons: list[int] = sorted(args.seasons)
    processed = 0

    for season in seasons:
        sos_path        = output_dir / f"sos_weights_{season}.parquet"
        normalizer_path = output_dir / f"sos_normalizer_{season}.pkl"

        if sos_path.exists() and not args.force:
            print(f"  Season {season}: already exists, skipping (use --force)")
            continue

        print(f"\n  Season {season}:")

        # Load RAPM ratings as opponent quality source
        rapm_path = data_dir / "rapm" / f"rapm_{season}.parquet"
        if not rapm_path.exists():
            warnings.warn(
                f"Season {season}: RAPM file not found at {rapm_path}. "
                "Run: uv run python scripts/gretzky.py train-rapm",
                DataMissingWarning,
                stacklevel=2,
            )
            print(f"    No RAPM data; skipping season {season}")
            continue

        rapm_df = pl.read_parquet(rapm_path)
        print(f"    Loaded {len(rapm_df)} players from RAPM")

        # Build normalizer and load opponent quality
        normalizer = StrengthOfScheduleNormalizer()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            normalizer.update_opponent_ratings(rapm_df)

        for w in caught:
            print(f"    [WARN] {w.message}", file=sys.stderr)

        # Load shot data to build player-game log
        shots_path = data_dir / "shots" / f"shots_{season}.parquet"
        if not shots_path.exists():
            warnings.warn(
                f"Season {season}: shots file not found at {shots_path}.",
                DataMissingWarning,
                stacklevel=2,
            )
            print(f"    No shot data; skipping season {season}")
            continue

        shots_df = pl.read_parquet(shots_path)
        required = {"shooter_id", "game_id"}
        if not required.issubset(set(shots_df.columns)):
            missing = required - set(shots_df.columns)
            print(f"    shots_df missing columns {sorted(missing)}; skipping", file=sys.stderr)
            continue

        # Build minimal player-game log (player_id, game_id)
        player_game_df = (
            shots_df
            .filter(pl.col("shooter_id").is_not_null())
            .select(["shooter_id", "game_id"])
            .unique()
            .rename({"shooter_id": "player_id"})
        )

        # Attempt to attach opponent_id if available
        if "home_team_id" in shots_df.columns and "away_team_id" in shots_df.columns:
            # Rough opponent lookup: if shooter is home team, opponent = away team, and vice versa
            pass  # Leave opponent_id matching to a future integration step

        print(f"    Player-game pairs: {len(player_game_df)}")

        # Compute SoS weights
        sos_df = normalizer.to_parquet_df(player_game_df, season=season)

        # Print summary stats
        if "sos_weight" in sos_df.columns:
            mean_w = float(sos_df["sos_weight"].mean() or 1.0)
            min_w  = float(sos_df["sos_weight"].min() or 1.0)
            max_w  = float(sos_df["sos_weight"].max() or 1.0)
            print(f"    SoS weights — mean={mean_w:.3f}  min={min_w:.3f}  max={max_w:.3f}")

        # Save SoS weights
        write_sos_weights(sos_df, output_dir, season)
        print(f"    SoS weights saved → {sos_path}")

        # Save normalizer state
        normalizer.save(normalizer_path)
        print(f"    Normalizer saved → {normalizer_path}")

        processed += 1

    print(f"\nDone. {processed}/{len(seasons)} seasons processed.")


if __name__ == "__main__":
    main()
