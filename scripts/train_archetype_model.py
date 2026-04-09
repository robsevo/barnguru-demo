#!/usr/bin/env python3
"""Train Player Archetype Clustering model (Feature 2.11).

Uses K-means clustering on per-player RAPM / xG / usage features to assign
each player to one of K archetypes (default K=10). Archetypes are used as
priors for Matchup model gaps when pair-GP < 10 (Feature 2.3).

Requires:
    ~/.gretzky/data/rapm/rapm_{season}.parquet   (run train-rapm first)

Usage::

    uv run python scripts/train_archetype_model.py
    uv run python scripts/train_archetype_model.py --seasons 2023 2024 2025
    uv run python scripts/train_archetype_model.py --k 8 --force

Outputs:
    ~/.gretzky/data/archetypes/archetype_model.pkl
    ~/.gretzky/data/archetypes/archetype_assignments_{season}.parquet
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


def _load_rapm(data_dir: Path, season: int) -> pl.DataFrame:
    from models.rapm_model import DataMissingWarning

    path = data_dir / "rapm" / f"rapm_{season}.parquet"
    if not path.exists():
        warnings.warn(
            f"Season {season}: RAPM file not found at {path}. "
            "Run: uv run python scripts/gretzky.py train-rapm",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame()
    return pl.read_parquet(path)


def _enrich_with_xg(rapm_df: pl.DataFrame, data_dir: Path, season: int) -> pl.DataFrame:
    """Join xG finishing stats into a single-season RAPM frame.

    Computes true per-game rates (xg_sum / gp, shots / gp) using the RAPM
    frame's own gp column so the rates are comparable across seasons of
    different lengths (e.g. injury-shortened or partial current season).
    """
    path = data_dir / "xg_finishing" / f"xg_finishing_{season}.parquet"
    if not path.exists():
        return rapm_df
    try:
        xg = pl.read_parquet(path)
        required = {"shooter_id", "shots", "xg_sum"}
        if not required.issubset(xg.columns):
            return rapm_df

        xg = xg.rename({"shooter_id": "player_id"}).select(["player_id", "shots", "xg_sum"])
        xg = xg.with_columns(pl.col("player_id").cast(pl.Int64))
        rapm_df = rapm_df.with_columns(pl.col("player_id").cast(pl.Int64))

        joined = rapm_df.join(xg, on="player_id", how="left")

        # Use the RAPM row's own gp for normalisation — guarantees per-game
        # rates are comparable regardless of season length
        gp_col = "gp" if "gp" in joined.columns else "games_played"
        if gp_col in joined.columns:
            joined = joined.with_columns([
                (pl.col("xg_sum") / pl.when(pl.col(gp_col) > 0)
                    .then(pl.col(gp_col)).otherwise(1)).alias("xg_per_game"),
                (pl.col("shots") / pl.when(pl.col(gp_col) > 0)
                    .then(pl.col(gp_col)).otherwise(1)).alias("shots_per_game"),
            ])
        else:
            # No GP column — fall back to season totals (less accurate but safe)
            joined = joined.with_columns([
                pl.col("xg_sum").alias("xg_per_game"),
                pl.col("shots").alias("shots_per_game"),
            ])
        return joined.drop(["xg_sum", "shots"])
    except Exception:
        return rapm_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Player Archetype Clustering model (K-means).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=_default_seasons(),
        metavar="YEAR",
        help="Seasons to include in the training pool.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        metavar="K",
        help="Number of archetype clusters.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-train even if output already exists.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        metavar="PATH",
        help="Root data directory.",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    output_dir = data_dir / "archetypes"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "archetype_model.pkl"
    if model_path.exists() and not args.force:
        print(
            f"Archetype model already exists at {model_path}. "
            "Use --force to retrain."
        )
        return

    # Load RAPM data across seasons, enriching each with per-game xG rates
    # before concatenation so rates are comparable regardless of season length.
    seasons: list[int] = sorted(args.seasons)
    frames: list[pl.DataFrame] = []
    n_xg_enriched = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for season in seasons:
            df = _load_rapm(data_dir, season)
            if len(df) == 0:
                continue
            df = df.with_columns(pl.lit(season).alias("season"))
            before = set(df.columns)
            df = _enrich_with_xg(df, data_dir, season)
            if "xg_per_game" in df.columns:
                n_xg_enriched += df["xg_per_game"].drop_nulls().len()
            frames.append(df)

    for w in caught:
        print(f"  [WARN] {w.message}", file=sys.stderr)

    if not frames:
        print(
            "ERROR: No RAPM data found for any target season.\n"
            "  Run: uv run python scripts/gretzky.py train-rapm",
            file=sys.stderr,
        )
        sys.exit(1)

    rapm_df = pl.concat(frames, how="diagonal")
    print(f"Loaded {len(rapm_df)} player-season rows from seasons {seasons}")
    if n_xg_enriched:
        print(f"  xG per-game rates enriched for {n_xg_enriched} player-season rows")
    else:
        print("  [WARN] No xg_finishing data found — xg_per_game will be zero for all players")

    # Drop fringe call-up rows from TRAINING only.
    # Players with < 30 GP have highly regularised RAPM (small n → large
    # uncertainty) and form their own spurious "high RAPM" micro-clusters.
    # We still predict archetypes for ALL players later; this filter only
    # controls which rows the K-means centroid positions are learnt from.
    MIN_TRAIN_GP = 30
    gp_col = "gp" if "gp" in rapm_df.columns else "games_played"
    if gp_col in rapm_df.columns:
        rapm_train = rapm_df.filter(pl.col(gp_col) >= MIN_TRAIN_GP)
        print(f"  Training on {len(rapm_train)} rows with ≥{MIN_TRAIN_GP} GP "
              f"(dropped {len(rapm_df) - len(rapm_train)} fringe rows)")
    else:
        rapm_train = rapm_df
        print("  [WARN] No GP column found — skipping minimum-GP filter")

    from models.archetype_model import PlayerArchetypeModel

    model = PlayerArchetypeModel(k=args.k)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = model.fit(rapm_train, season=max(seasons))

    for w in caught:
        print(f"  [WARN] {w.message}", file=sys.stderr)

    print(
        f"  K={args.k}  inertia={metrics['inertia']:.1f}  "
        f"n_players={metrics['n_players']}  "
        f"silhouette={metrics.get('silhouette', float('nan')):.3f}"
    )

    # Save model
    model.save(model_path)
    print(f"  Model saved → {model_path}")

    # Print centroid summary
    centroids = model.centroids_dataframe()
    print("\n  Archetype centroids (top features):")
    for row in centroids.to_dicts():
        cid  = row["cluster_id"]
        name = row.get("archetype", f"cluster_{cid}")
        n    = row.get("n_players", "?")
        off  = row.get("off_rapm", float("nan"))
        def_ = row.get("def_rapm", float("nan"))
        xgpg = row.get("xg_per_game", float("nan"))
        print(f"    [{cid}] {name:<22} n={n:>4}  off={off:+.3f}  def={def_:+.3f}  xg/gm={xgpg:.1f}")

    # Save per-season assignments
    for season in seasons:
        season_df = rapm_df.filter(pl.col("season") == season) if "season" in rapm_df.columns else rapm_df
        if len(season_df) == 0:
            continue
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assignments = model.predict(season_df, season=season)

        out_path = output_dir / f"archetype_assignments_{season}.parquet"
        assignments.write_parquet(out_path)
        print(f"  Season {season}: {len(assignments)} assignments → {out_path}")

    print(f"\nDone. Archetype model saved to {model_path}")


if __name__ == "__main__":
    main()
