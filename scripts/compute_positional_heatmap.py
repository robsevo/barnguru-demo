"""compute_positional_heatmap — Feature 2.21 training script.

Loads shot data (from MoneyPuck parquets) and optionally EDGE skating data,
computes per-player offensive zone shot distributions and attaches defensive
zone deployment profiles, then writes the result to parquet.

Usage::

    uv run python scripts/compute_positional_heatmap.py
    uv run python scripts/compute_positional_heatmap.py --seasons 2023 2024 2025
    uv run python scripts/compute_positional_heatmap.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import os

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from models.positional_heatmap import (
    PositionalHeatmapModel,
    write_positional_heatmap,
    CONFIDENCE_FULL,
    CONFIDENCE_PARTIAL,
    CONFIDENCE_INSUFFICIENT,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR  = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
DEFAULT_SEASONS   = [2023, 2024, 2025]
SHOTS_SUBDIR      = "shots"
EDGE_SUBDIR       = "edge"
HEATMAP_SUBDIR    = "positional_heatmap"


def _fix_null_columns(frames: list[pl.DataFrame]) -> list[pl.DataFrame]:
    """Cast Null-dtype columns to consistent types to allow diagonal concat."""
    for i, df in enumerate(frames):
        casts = []
        for col in df.columns:
            if df[col].dtype == pl.Null:
                target_dtype = None
                for other in frames:
                    if col in other.columns and other[col].dtype != pl.Null:
                        target_dtype = other[col].dtype
                        break
                casts.append(
                    pl.lit(None).cast(target_dtype or pl.Utf8).alias(col)
                )
        if casts:
            frames[i] = df.with_columns(casts)
    return frames


def _load_shots_parquets(shots_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """Load and concatenate MoneyPuck shots parquets for the requested seasons."""
    frames: list[pl.DataFrame] = []
    for season in seasons:
        files = sorted(shots_dir.glob(f"shots_{season}*.parquet"))
        if not files:
            files = sorted(shots_dir.glob(f"moneypuck_shots_{season}*.parquet"))
        for f in files:
            try:
                frames.append(pl.read_parquet(f))
                print(f"  Loaded shots: {f.name}")
            except Exception as e:
                print(f"  Warning: could not load {f.name}: {e}")
    if not frames:
        return pl.DataFrame()
    return pl.concat(_fix_null_columns(frames), how="diagonal")


def _load_edge_parquets(edge_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """Load and concatenate EDGE skating parquets for the requested seasons."""
    frames: list[pl.DataFrame] = []
    for season in seasons:
        files = sorted(edge_dir.glob(f"edge_skating_{season}*.parquet"))
        if not files:
            files = sorted(edge_dir.glob(f"edge_{season}*.parquet"))
        for f in files:
            try:
                frames.append(pl.read_parquet(f))
                print(f"  Loaded EDGE: {f.name}")
            except Exception as e:
                print(f"  Warning: could not load {f.name}: {e}")
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Player Positional Heat Map from shot data (Feature 2.21)."
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
    heatmap_dir = data_dir / HEATMAP_SUBDIR
    out_path = heatmap_dir / f"positional_heatmap_{target_season}.parquet"
    if out_path.exists() and not args.force:
        print(f"[positional-heatmap] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    # ── Load shots ────────────────────────────────────────────────────────
    shots_dir = data_dir / SHOTS_SUBDIR
    if not shots_dir.exists():
        print(f"[positional-heatmap] Shots directory not found: {shots_dir}")
        print("  Run 'uv run python scripts/gretzky.py ingest' first.")
        sys.exit(1)

    print(f"[positional-heatmap] Loading shot data for seasons: {seasons}")
    shots_df = _load_shots_parquets(shots_dir, seasons)

    if len(shots_df) == 0:
        print("[positional-heatmap] No shot data found.")
        sys.exit(1)
    print(f"  {len(shots_df):,} shot rows loaded")

    # ── Load EDGE (optional) ──────────────────────────────────────────────
    edge_dir = data_dir / EDGE_SUBDIR
    edge_df  = None
    if edge_dir.exists():
        print(f"[positional-heatmap] Loading EDGE skating data…")
        raw = _load_edge_parquets(edge_dir, seasons)
        if len(raw) > 0:
            edge_df = raw
            print(f"  {len(edge_df):,} EDGE rows loaded")
    else:
        print(f"[positional-heatmap] No EDGE directory found — defensive profile will be null.")

    # ── Fit model ─────────────────────────────────────────────────────────
    model = PositionalHeatmapModel()
    print(f"[positional-heatmap] Computing positional profiles (target season: {target_season})…")
    heatmap_df = model.fit(shots_df, season=target_season, edge_df=edge_df)

    n_players = len(heatmap_df)
    print(f"  Profiles computed for {n_players:,} player(s).")

    # ── Summary ───────────────────────────────────────────────────────────
    summary = model.summary(heatmap_df)
    print(f"\n  Confidence breakdown:")
    print(f"    full:         {summary['n_full']:>5}  (≥ {model._min_full} shots)")
    print(f"    partial:      {summary['n_partial']:>5}  ({model._min_partial}–{model._min_full-1} shots)")
    print(f"    insufficient: {summary['n_insufficient']:>5}  (< {model._min_partial} shots)")
    if summary["mean_high_danger_pct"] is not None:
        print(f"\n  League mean high-danger shot%: {summary['mean_high_danger_pct']:.1%}")

    # ── Top net-front players ─────────────────────────────────────────────
    if n_players > 0:
        top = (
            heatmap_df
            .filter(pl.col("confidence").is_in([CONFIDENCE_FULL, CONFIDENCE_PARTIAL]))
            .sort("off_net_front_pct", descending=True)
            .head(10)
        )
        if len(top) > 0:
            print(f"\n  Top-10 net-front shot% players:")
            print(f"  {'Player':<22}  {'Team':<6}  {'Net-Front':>9}  {'Slot':>6}  {'Shots':>5}")
            print(f"  {'─'*22}  {'─'*6}  {'─'*9}  {'─'*6}  {'─'*5}")
            for row in top.to_dicts():
                print(
                    f"  {row['player_name']:<22}  {row['team']:<6}  "
                    f"{row['off_net_front_pct']:>9.1%}  "
                    f"{row['off_slot_pct']:>6.1%}  {row['n_shots']:>5}"
                )

    # ── Save ──────────────────────────────────────────────────────────────
    path = write_positional_heatmap(heatmap_df, heatmap_dir, target_season)
    print(f"\n[positional-heatmap] Written: {path}")

    model_path = heatmap_dir / "positional_heatmap_model.pkl"
    model.save(model_path)
    print(f"[positional-heatmap] Model saved: {model_path}")


if __name__ == "__main__":
    main()
