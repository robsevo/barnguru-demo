#!/usr/bin/env python3
"""Compute Seller Motivation State — Feature 4.16.

Requires: ~/.gretzky/data/buyer_seller/buyer_seller_{season}.parquet

Outputs:  ~/.gretzky/data/seller_motivation/seller_motivation_{season}.parquet
"""
from __future__ import annotations
import argparse, os, sys, warnings
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import polars as pl

_DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))

def _current_nhl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Seller Motivation (Feature 4.16)")
    parser.add_argument("--season", type=int, default=_current_nhl_season())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    args = parser.parse_args()

    from models.seller_motivation import compute_seller_motivation, write_seller_motivation

    output_dir = args.data_dir / "seller_motivation"
    out_path = output_dir / f"seller_motivation_{args.season}.parquet"
    if out_path.exists() and not args.force:
        print(f"  Season {args.season}: exists, skipping (use --force)")
        return

    bs_path = args.data_dir / "buyer_seller" / f"buyer_seller_{args.season}.parquet"
    if not bs_path.exists():
        print(f"[seller-motivation] buyer_seller not found: {bs_path}", file=sys.stderr)
        sys.exit(1)

    bs_df = pl.read_parquet(bs_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = compute_seller_motivation(bs_df, season=args.season)
    for w in caught:
        print(f"  [WARN] {w.message}", file=sys.stderr)

    path = write_seller_motivation(df, output_dir, args.season)
    print(f"  Saved {path}  ({len(df)} rows)")
    dragged = df.filter(pl.col("seller_drag") > 0)
    if not dragged.is_empty():
        print(f"\n  Active seller drags:")
        for r in dragged.iter_rows(named=True):
            print(f"    {r['team']:<4}  drag {r['seller_drag']:.3f}  eff ×{r['efficiency_multiplier']:.3f}  GP since DL ~{r['games_since_deadline']}")
    else:
        print(f"  No active seller drags (deadline may not have passed yet).")
    print("\nDone.")

if __name__ == "__main__":
    main()
