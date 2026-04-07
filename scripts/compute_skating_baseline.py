"""compute_skating_baseline — Feature 2.20 training script.

Loads EDGE skating data across requested seasons, fits per-player baseline
profiles (rested-conditioned), and writes the result to parquet.

Usage::

    uv run python scripts/compute_skating_baseline.py
    uv run python scripts/compute_skating_baseline.py --seasons 2023 2024 2025
    uv run python scripts/compute_skating_baseline.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from models.skating_baseline import (
    SkatingBaselineModel,
    write_skating_baseline,
    CONFIDENCE_RESTED,
    CONFIDENCE_AGGREGATE,
    CONFIDENCE_INSUFFICIENT,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR  = Path.home() / ".gretzky" / "data"
DEFAULT_SEASONS   = [2023, 2024, 2025]
EDGE_SUBDIR       = "edge"
BASELINE_SUBDIR   = "skating_baseline"


def _load_edge_parquets(edge_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """Load and concatenate EDGE skating parquets for the requested seasons."""
    frames: list[pl.DataFrame] = []
    for season in seasons:
        pattern = f"edge_skating_{season}*.parquet"
        files = sorted(edge_dir.glob(pattern))
        if not files:
            # Try without season-specific pattern
            files = sorted(edge_dir.glob(f"edge_{season}*.parquet"))
        if files:
            for f in files:
                try:
                    frames.append(pl.read_parquet(f))
                    print(f"  Loaded: {f.name}")
                except Exception as e:
                    print(f"  Warning: could not load {f.name}: {e}")
        else:
            print(f"  No EDGE parquet found for season {season} in {edge_dir}")
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Player Skating Baseline profiles from EDGE data (Feature 2.20)."
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
        help="Overwrite existing baseline parquets.",
    )
    args = parser.parse_args()

    seasons: list[int] = sorted(args.seasons)
    data_dir: Path = args.data_dir
    target_season = max(seasons)

    # ── Load EDGE data ────────────────────────────────────────────────────
    edge_dir = data_dir / EDGE_SUBDIR
    if not edge_dir.exists():
        print(f"[skating-baseline] EDGE data directory not found: {edge_dir}")
        print("  Run 'uv run python scripts/gretzky.py sync' first to populate EDGE data.")
        sys.exit(1)

    print(f"[skating-baseline] Loading EDGE data for seasons: {seasons}")
    edge_df = _load_edge_parquets(edge_dir, seasons)

    if len(edge_df) == 0:
        print("[skating-baseline] No EDGE data found. Cannot compute baselines.")
        print("  Run 'uv run python scripts/gretzky.py sync' first.")
        sys.exit(1)

    print(f"  {len(edge_df):,} EDGE rows loaded across {len(seasons)} season(s)")

    # ── Check if output already exists ───────────────────────────────────
    baseline_dir = data_dir / BASELINE_SUBDIR
    out_path = baseline_dir / f"skating_baseline_{target_season}.parquet"
    if out_path.exists() and not args.force:
        print(f"[skating-baseline] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    # ── Fit model ─────────────────────────────────────────────────────────
    model = SkatingBaselineModel()
    print(f"[skating-baseline] Fitting baseline profiles (target season: {target_season})…")
    baseline_df = model.fit(edge_df, season=target_season)

    n_players = len(baseline_df)
    print(f"  Profiles computed for {n_players:,} player(s).")

    # ── Print summary ─────────────────────────────────────────────────────
    summary = model.summary(baseline_df)
    print(f"\n  Confidence breakdown:")
    print(f"    rested:       {summary['n_rested']:>4}  (used rested-game filter)")
    print(f"    aggregate:    {summary['n_aggregate']:>4}  (season-aggregate fallback)")
    print(f"    insufficient: {summary['n_insufficient']:>4}  (< min games, no baseline)")
    if summary["mean_speed"] is not None:
        print(f"\n  League mean avg speed:    {summary['mean_speed']:.2f} km/h")
    if summary["mean_distance"] is not None:
        print(f"  League mean distance/gm:  {summary['mean_distance']:.2f} km")
    if summary["mean_oz_pct"] is not None:
        print(f"  League mean OZ time%:     {summary['mean_oz_pct']:.1%}")

    # ── Top speed players ─────────────────────────────────────────────────
    if n_players > 0:
        top_speed = (
            baseline_df
            .filter(pl.col("baseline_avg_speed_kmh").is_not_null())
            .sort("baseline_avg_speed_kmh", descending=True)
            .head(10)
        )
        if len(top_speed) > 0:
            print(f"\n  Top-10 skating speed baselines:")
            print(f"  {'Player':<22}  {'Team':<6}  {'AvgSpeed':>8}  {'MaxSpeed':>8}  {'Dist/Gm':>8}  {'Conf'}")
            print(f"  {'─'*22}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*4}")
            for row in top_speed.to_dicts():
                avg_s = row.get("baseline_avg_speed_kmh") or 0.0
                max_s = row.get("baseline_max_speed_kmh") or 0.0
                dist  = row.get("baseline_distance_per_game_km") or 0.0
                print(
                    f"  {row['player_name']:<22}  {row['team']:<6}  "
                    f"{avg_s:>8.2f}  {max_s:>8.2f}  {dist:>8.2f}  {row['confidence']}"
                )

    # ── Save ──────────────────────────────────────────────────────────────
    path = write_skating_baseline(baseline_df, baseline_dir, target_season)
    print(f"\n[skating-baseline] Written: {path}")

    # ── Save model ────────────────────────────────────────────────────────
    model_path = baseline_dir / "skating_baseline_model.pkl"
    model.save(model_path)
    print(f"[skating-baseline] Model saved: {model_path}")


if __name__ == "__main__":
    main()
