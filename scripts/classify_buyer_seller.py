#!/usr/bin/env python3
"""Classify Trade Deadline Buyer/Seller — Feature 4.15.

Requires::
    ~/.gretzky/data/raw/team_stats_{season}.parquet

Usage::
    uv run python scripts/classify_buyer_seller.py
    uv run python scripts/classify_buyer_seller.py --season 2025 --force

Outputs::
    ~/.gretzky/data/buyer_seller/buyer_seller_{season}.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

_DEFAULT_DATA_DIR = Path(
    os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data"))
)


def _current_nhl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify Trade Deadline Buyer/Seller (Feature 4.15)")
    parser.add_argument("--season", type=int, default=_current_nhl_season())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    args = parser.parse_args()

    from models.buyer_seller import compute_buyer_seller, write_buyer_seller
    from models.rapm_model   import _NHL_TEAM_IDS

    output_dir = args.data_dir / "buyer_seller"
    out_path = output_dir / f"buyer_seller_{args.season}.parquet"
    if out_path.exists() and not args.force:
        print(f"  Season {args.season}: already exists, skipping (use --force)")
        return

    ts_path = args.data_dir / "raw" / f"team_stats_{args.season}.parquet"
    if not ts_path.exists():
        print(f"[buyer-seller] required file missing: {ts_path}", file=sys.stderr)
        sys.exit(1)

    print(f"  Loading team_stats for {args.season}…")
    team_stats = pl.read_parquet(ts_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = compute_buyer_seller(team_stats, team_lookup=_NHL_TEAM_IDS, season=args.season)
    for w in caught:
        print(f"  [WARN] {w.message}", file=sys.stderr)

    path = write_buyer_seller(df, output_dir, args.season)
    print(f"  Saved {path}  ({len(df)} rows)")

    buyers  = df.filter(pl.col("classification") == "buyer")
    sellers = df.filter(pl.col("classification") == "seller")
    neutral = df.filter(pl.col("classification") == "neutral")
    print(f"\n  Buyers ({len(buyers)}):")
    for r in buyers.sort("points_pct", descending=True).head(10).iter_rows(named=True):
        print(f"    {r['team']:<4}  P% {r['points_pct']:.3f}  gap {r['gap']:+.3f}  conf {r['confidence']:.2f}")
    print(f"\n  Sellers ({len(sellers)}):")
    for r in sellers.sort("points_pct").head(10).iter_rows(named=True):
        print(f"    {r['team']:<4}  P% {r['points_pct']:.3f}  gap {r['gap']:+.3f}  conf {r['confidence']:.2f}")
    print(f"\n  Neutral ({len(neutral)}):")
    for r in neutral.iter_rows(named=True):
        print(f"    {r['team']:<4}  P% {r['points_pct']:.3f}  gap {r['gap']:+.3f}  conf {r['confidence']:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
