"""compute_playoff_delta — Feature 2.23 training script.

Loads historical player stats split by game type (regular season vs. playoff)
from NHL PBP parquets and computes per-player playoff performance deltas with
Bayesian shrinkage.

Usage::

    uv run python scripts/compute_playoff_delta.py
    uv run python scripts/compute_playoff_delta.py --seasons 2022 2023 2024 2025
    uv run python scripts/compute_playoff_delta.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from models.playoff_delta import (
    PlayoffDeltaModel,
    write_playoff_delta,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR  = Path.home() / ".gretzky" / "data"
DEFAULT_SEASONS   = [2022, 2023, 2024, 2025]
STATS_SUBDIR      = "raw"
PBP_SUBDIR        = "pbp"
DELTA_SUBDIR      = "playoff_delta"


def _load_player_stats(stats_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """Load and concatenate player stats parquets for given seasons.

    Accepts both game-type-split files (player_stats_{season}_*.parquet) and
    a single unified file (player_stats_{season}.parquet).
    """
    frames: list[pl.DataFrame] = []
    for season in seasons:
        # Prefer split files (one per game_type) if present
        split_files = sorted(stats_dir.glob(f"player_stats_{season}_*.parquet"))
        if split_files:
            for f in split_files:
                try:
                    frames.append(pl.read_parquet(f))
                    print(f"  Loaded: {f.name}")
                except Exception as e:
                    print(f"  Warning: could not load {f.name}: {e}")
        else:
            # Fall back to unified file
            f = stats_dir / f"player_stats_{season}.parquet"
            if f.exists():
                try:
                    frames.append(pl.read_parquet(f))
                    print(f"  Loaded: {f.name}")
                except Exception as e:
                    print(f"  Warning: could not load {f.name}: {e}")
            else:
                print(f"  No stats file found for season {season}")
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Playoff Performance Delta (Feature 2.23)."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=DEFAULT_SEASONS,
        metavar="YEAR",
        help=f"Seasons to pool for player stats (default: {DEFAULT_SEASONS}).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Root data directory (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output parquet.",
    )
    args = parser.parse_args()

    seasons: list[int] = sorted(args.seasons)
    data_dir: Path = args.data_dir
    target_season = max(seasons)

    # ── Check output ──────────────────────────────────────────────────────
    delta_dir = data_dir / DELTA_SUBDIR
    out_path  = delta_dir / f"playoff_delta_{target_season}.parquet"
    if out_path.exists() and not args.force:
        print(f"[playoff-delta] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    # ── Load player stats ─────────────────────────────────────────────────
    stats_dir = data_dir / STATS_SUBDIR
    if not stats_dir.exists():
        print(f"[playoff-delta] Stats directory not found: {stats_dir}")
        print("  Run 'uv run python scripts/gretzky.py ingest' first.")
        sys.exit(1)

    print(f"[playoff-delta] Loading player stats for seasons: {seasons}")
    stats_df = _load_player_stats(stats_dir, seasons)

    if len(stats_df) == 0:
        print("[playoff-delta] No player stats found.")
        print("  Run 'uv run python scripts/gretzky.py ingest' first.")
        sys.exit(1)
    print(f"  {len(stats_df):,} player-season-gametype rows loaded")

    # ── Validate required columns ─────────────────────────────────────────
    if "player_id" not in stats_df.columns:
        print("[playoff-delta] Missing required column: player_id")
        print("  Ensure player_stats parquets include player_id.")
        sys.exit(1)
    # If game_type is missing, default all rows to regular season (2)
    if "game_type" not in stats_df.columns:
        print("  [warn] game_type column missing — defaulting all rows to regular season (2).")
        stats_df = stats_df.with_columns(pl.lit(2).cast(pl.Int64).alias("game_type"))

    # ── Fit model ─────────────────────────────────────────────────────────
    print(f"\n[playoff-delta] Computing playoff deltas…")
    model = PlayoffDeltaModel()
    delta_df = model.fit(stats_df)

    n_players = len(delta_df)
    print(f"  Deltas computed for {n_players:,} player(s).")

    # ── Summary ───────────────────────────────────────────────────────────
    if n_players > 0 and "shrunk_delta_goals_per60" in delta_df.columns:
        non_null = delta_df.filter(pl.col("shrunk_delta_goals_per60").is_not_null())
        n_positive = (non_null["shrunk_delta_goals_per60"] > 0).sum()
        n_negative = (non_null["shrunk_delta_goals_per60"] < 0).sum()
        print(f"\n  Playoff goal-scoring delta (shrinkage-adjusted):")
        print(f"    Players with positive delta: {n_positive}")
        print(f"    Players with negative delta: {n_negative}")

        top = (
            delta_df
            .filter(pl.col("shrunk_delta_goals_per60").is_not_null())
            .sort("shrunk_delta_goals_per60", descending=True)
            .head(10)
        )
        if len(top) > 0:
            print(f"\n  Top-10 playoff performers (by shrunk goals/60 delta):")
            print(f"  {'Player':<22}  {'Pos':<3}  {'PO GP':>5}  {'Δ goals/60':>10}  {'Shrinkage':>9}")
            print(f"  {'─'*22}  {'─'*3}  {'─'*5}  {'─'*10}  {'─'*9}")
            for row in top.to_dicts():
                delta_val = row.get("shrunk_delta_goals_per60")
                sf = row.get("shrinkage_factor", 0.0)
                print(
                    f"  {row['player_name']:<22}  {row['position']:<3}  "
                    f"{row['playoff_gp']:>5}  "
                    f"{delta_val:>+10.3f}  "
                    f"{sf:>9.1%}"
                )

    # ── Save ──────────────────────────────────────────────────────────────
    path = write_playoff_delta(delta_df, delta_dir, target_season)
    print(f"\n[playoff-delta] Deltas written: {path}")

    model_path = delta_dir / "playoff_delta_model.pkl"
    model.save(model_path)
    print(f"[playoff-delta] Model saved: {model_path}")


if __name__ == "__main__":
    main()
