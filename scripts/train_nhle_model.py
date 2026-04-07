"""train_nhle_model — Feature 2.24 training script.

Loads prospect / player stats split by league from the data directory,
computes NHLe PPG translations, and estimates P(NHLer) / P(star) for
each player using the NHLeModel logistic regression classifiers.

Usage::

    uv run python scripts/train_nhle_model.py
    uv run python scripts/train_nhle_model.py --seasons 2023 2024 2025
    uv run python scripts/train_nhle_model.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from models.nhle_model import (
    NHLeModel,
    write_nhle_projections,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR   = Path.home() / ".gretzky" / "data"
DEFAULT_SEASONS    = [2023, 2024, 2025]
PROSPECTS_SUBDIR   = "prospects"
NHLE_SUBDIR        = "nhle_projections"


def _load_prospects(prospects_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """Load and concatenate prospect parquets for given seasons."""
    frames: list[pl.DataFrame] = []
    for season in seasons:
        files = sorted(prospects_dir.glob(f"prospects_{season}*.parquet"))
        if not files:
            files = sorted(prospects_dir.glob(f"player_stats_{season}*.parquet"))
        for f in files:
            try:
                frames.append(pl.read_parquet(f))
                print(f"  Loaded: {f.name}")
            except Exception as e:
                print(f"  Warning: could not load {f.name}: {e}")
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NHLe/Prospect Projection Model (Feature 2.24)."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=DEFAULT_SEASONS,
        metavar="YEAR",
        help=f"Seasons to pool (default: {DEFAULT_SEASONS}).",
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
    nhle_dir  = data_dir / NHLE_SUBDIR
    out_path  = nhle_dir / f"nhle_projections_{target_season}.parquet"
    if out_path.exists() and not args.force:
        print(f"[nhle] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    # ── Load prospect data ────────────────────────────────────────────────
    prospects_dir = data_dir / PROSPECTS_SUBDIR
    if not prospects_dir.exists():
        # Fall back to raw/player_stats parquets — treat all players as "prospects"
        raw_dir = data_dir / "raw"
        if raw_dir.exists() and any(raw_dir.glob("player_stats_*.parquet")):
            print(f"[nhle] Prospects directory not found; falling back to raw/player_stats parquets.")
            prospects_dir = raw_dir
        else:
            print(f"[nhle] Prospects directory not found: {prospects_dir}")
            print("  Run 'uv run python scripts/gretzky.py ingest' first.")
            sys.exit(1)

    print(f"[nhle] Loading prospect data for seasons: {seasons}")
    prospects_df = _load_prospects(prospects_dir, seasons)

    if len(prospects_df) == 0:
        print("[nhle] No prospect data found.")
        sys.exit(1)
    print(f"  {len(prospects_df):,} prospect-season rows loaded")

    # ── Validate required columns ──────────────────────────────────────────
    nhle_required = {"age", "league", "goals", "assists", "gp"}
    missing_nhle = nhle_required - set(prospects_df.columns)
    if missing_nhle:
        print(f"  [warn] Prospect data missing NHLe columns: {sorted(missing_nhle)}")
        print("  NHLe requires prospect data with age, league, goals, assists, gp.")
        print("  Current player_stats parquets contain only situational stats (faceoffs, etc.).")
        print("  Skipping NHLe computation — output will be empty.")
        nhle_dir.mkdir(parents=True, exist_ok=True)
        proj_df = pl.DataFrame(schema={"player_id": pl.Int64, "season": pl.Int64,
                                        "league": pl.Utf8, "nhle_ppg": pl.Float64,
                                        "p_nhler": pl.Float64, "p_star": pl.Float64})
        proj_df.write_parquet(out_path)
        print(f"\n[nhle] Empty projections written: {out_path}")
        return

    # ── Train model ───────────────────────────────────────────────────────
    print(f"\n[nhle] Computing NHLe translations…")
    model = NHLeModel()
    proj_df = model.fit(prospects_df)

    n_players = len(proj_df)
    print(f"  Projections computed for {n_players:,} player(s).")

    # ── Summary ───────────────────────────────────────────────────────────
    if n_players > 0:
        valid = proj_df.filter(pl.col("nhle_ppg").is_not_null())
        if len(valid) > 0:
            print(f"\n  League breakdown:")
            league_counts = (
                valid
                .group_by("league")
                .agg(pl.len().alias("n"), pl.col("nhle_ppg").mean().alias("mean_nhle"))
                .sort("mean_nhle", descending=True)
            )
            for row in league_counts.to_dicts():
                print(f"    {row['league']:<18s}  n={row['n']:>4}  mean NHLe: {row['mean_nhle']:.3f}")

            top = (
                valid
                .sort("nhle_ppg", descending=True)
                .head(10)
            )
            print(f"\n  Top-10 prospects by NHLe PPG:")
            print(f"  {'Player':<22}  {'League':<8}  {'Age':>3}  {'NHLe PPG':>8}  {'P(NHLer)':>8}  {'P(star)':>7}")
            print(f"  {'─'*22}  {'─'*8}  {'─'*3}  {'─'*8}  {'─'*8}  {'─'*7}")
            for row in top.to_dicts():
                print(
                    f"  {row['player_name']:<22}  {row['league']:<8}  "
                    f"{row['age']:>3.0f}  {row['nhle_ppg']:>8.3f}  "
                    f"{row['p_nhler']:>8.1%}  {row['p_star']:>7.1%}"
                )

    # ── Save ──────────────────────────────────────────────────────────────
    path = write_nhle_projections(proj_df, nhle_dir, target_season)
    print(f"\n[nhle] Projections written: {path}")

    model_path = nhle_dir / "nhle_model.pkl"
    model.save(model_path)
    print(f"[nhle] Model saved: {model_path}")


if __name__ == "__main__":
    main()
