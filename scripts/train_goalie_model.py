"""Train the Goalie Model (Feature 2.5) and Goalie Fatigue Sub-model (Feature 2.6).

Requires (per season):
  - MoneyPuck goalie stats: ~/.gretzky/data/raw/goalie_stats_{season}.parquet
    (run: uv run python scripts/gretzky.py sync first)

Optional (for fatigue model):
  - Goalie game log:  ~/.gretzky/data/raw/goalie_gamelog_{season}.parquet
    (per-goalie-game rows with is_b2b, rest_days, gp_last_7, etc.)

Usage::

    uv run python scripts/train_goalie_model.py
    uv run python scripts/train_goalie_model.py --seasons 2023 2024 2025
    uv run python scripts/train_goalie_model.py --seasons 2025 --force
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
# Helpers
# ---------------------------------------------------------------------------


def _load_goalie_stats(data_dir: Path, season: int) -> pl.DataFrame | None:
    """Load MoneyPuck goalie stats parquet for one season."""
    from models.rapm_model import DataMissingWarning

    # Check both goalie_stats/ (sync output) and raw/ (legacy location)
    path = data_dir / "goalie_stats" / f"goalie_stats_{season}.parquet"
    if not path.exists():
        path = data_dir / "raw" / f"goalie_stats_{season}.parquet"
    if not path.exists():
        warnings.warn(
            f"Season {season}: goalie_stats file not found at {path}.\n"
            "  Run: uv run python scripts/gretzky.py sync",
            DataMissingWarning,
            stacklevel=2,
        )
        return None
    return pl.read_parquet(path)


def _load_goalie_gamelog(data_dir: Path, season: int) -> pl.DataFrame | None:
    """Load per-goalie game log for fatigue model (optional)."""
    path = data_dir / "goalie_stats" / f"goalie_gamelog_{season}.parquet"
    if not path.exists():
        path = data_dir / "raw" / f"goalie_gamelog_{season}.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


def _print_goalie_leaders(ratings_df: pl.DataFrame, season: int) -> None:
    """Print top 10 goalies by model_gsax_per60."""
    print(f"\n  ── Season {season} Top Goalies (model_gsax_per60) ──")
    top = ratings_df.head(10)
    for r in top.to_dicts():
        print(
            f"    {r['player_name']:<22}  {r['team']}  "
            f"GSAx/60: {r['model_gsax_per60']:+.3f}  "
            f"SV%: {r['sv_pct']:.3f}  "
            f"GP: {r['games_played']}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Goalie Model (Feature 2.5) + Goalie Fatigue (Feature 2.6).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=_default_seasons(),
        metavar="YEAR",
        help="Season years to fit and predict.",
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

    data_dir:      Path = args.data_dir
    target_seasons      = sorted(args.seasons)
    goalie_dir          = data_dir / "goalie_ratings"
    goalie_dir.mkdir(parents=True, exist_ok=True)

    from models.goalie_model import GoalieModel, write_goalie_ratings
    from models.goalie_fatigue_model import GoalieFatigueModel
    from models.rapm_model import DataMissingWarning

    # Check existing outputs
    if not args.force:
        existing = [
            s for s in target_seasons
            if (goalie_dir / f"goalie_ratings_{s}.parquet").exists()
        ]
        if existing:
            print(f"Outputs already exist for seasons {existing}. Use --force to retrain.")
            if len(existing) == len(target_seasons):
                return

    # Load data
    print(f"\nLoading goalie stats for seasons {target_seasons}…\n")
    goalie_dfs: list[pl.DataFrame] = []
    valid_seasons: list[int]       = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for season in target_seasons:
            df = _load_goalie_stats(data_dir, season)
            if df is None or df.is_empty():
                print(f"  Season {season}: skipping (no data).")
                continue
            # Inject season column if missing
            if "season" not in df.columns:
                df = df.with_columns(pl.lit(season).cast(pl.Int64).alias("season"))
            print(f"  Season {season}: {len(df):,} goalie rows")
            goalie_dfs.append(df)
            valid_seasons.append(season)

    for w in caught:
        if issubclass(w.category, DataMissingWarning):
            print(f"  [WARN] {w.message}", file=sys.stderr)

    if not valid_seasons:
        print("ERROR: No goalie data found for any target season.", file=sys.stderr)
        sys.exit(1)

    # Train goalie model
    print(f"\nTraining GoalieModel on seasons {valid_seasons}…")
    model = GoalieModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = model.fit(goalie_dfs, valid_seasons)

    print(f"  Trained on {metrics['n_goalies']} goalies  |  RMSE: {metrics['rmse']:.4f}")

    model_path = goalie_dir / "goalie_model.pkl"
    model.save(model_path)
    print(f"  Model saved → {model_path}")

    # Feature importance
    fi = model.feature_importance()
    if fi:
        print("\n  Feature importances (gain):")
        for feat, score in list(fi.items())[:5]:
            print(f"    {feat:<20}  {score:.4f}")

    # Generate ratings per season
    print(f"\nGenerating goalie ratings for seasons {valid_seasons}…\n")
    for goalie_df, season in zip(goalie_dfs, valid_seasons):
        out_path = goalie_dir / f"goalie_ratings_{season}.parquet"
        if out_path.exists() and not args.force:
            print(f"  goalie_ratings_{season}.parquet already exists, skipping.")
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ratings = model.predict_season(goalie_df, season)
        write_goalie_ratings(ratings, goalie_dir, season)
        print(f"  Saved goalie_ratings_{season}.parquet  ({len(ratings)} goalies)")
        _print_goalie_leaders(ratings, season)

    # Goalie fatigue model — train if game logs available
    fatigue_dir = data_dir / "goalie_fatigue"
    fatigue_dir.mkdir(parents=True, exist_ok=True)
    fatigue_model = GoalieFatigueModel()

    gamelog_seasons: list[pl.DataFrame] = []
    for season in valid_seasons:
        gl = _load_goalie_gamelog(data_dir, season)
        if gl is not None:
            gamelog_seasons.append(gl)

    if gamelog_seasons:
        print(f"\nTraining GoalieFatigueModel on {len(gamelog_seasons)} season(s) of game logs…")
        combined = pl.concat(gamelog_seasons, how="diagonal")
        if "sv_delta" in combined.columns:
            try:
                fm = fatigue_model.fit(combined)
                print(f"  RMSE: {fm['rmse']:.4f}  R²: {fm['r2']:.3f}  n_games: {fm['n_games']}")
                fatigue_model_path = fatigue_dir / "goalie_fatigue_model.pkl"
                fatigue_model.save(fatigue_model_path)
                print(f"  Fatigue model saved → {fatigue_model_path}")
                print("\n  Auditable coefficients:")
                for feat, coef in fatigue_model.coefficients().items():
                    print(f"    {feat:<20}  {coef:+.6f}")
            except Exception as exc:
                print(f"  [WARN] Fatigue fit failed: {exc}", file=sys.stderr)
        else:
            print("  Game log found but no 'sv_delta' column — using default coefficients.")
    else:
        print("\nNo goalie game logs found — GoalieFatigueModel using default coefficients.")
        print("  Auditable default coefficients:")
        for feat, coef in fatigue_model.coefficients().items():
            print(f"    {feat:<20}  {coef:+.6f}")

    print(f"\nDone. Goalie ratings saved to {goalie_dir}")


if __name__ == "__main__":
    main()
