#!/usr/bin/env python3
"""Compute Per-GM Behavioral Fingerprint — Feature 4.18.

Requires: buyer_seller + transactions parquets

Outputs:  ~/.gretzky/data/gm_fingerprint/gm_fingerprint_{season}.parquet
"""
from __future__ import annotations
import argparse, os, sys, warnings, glob
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

def _load_all_transactions(data_dir: Path) -> pl.DataFrame:
    tx_dir = data_dir / "transactions"
    if not tx_dir.exists():
        return pl.DataFrame()
    files = sorted(tx_dir.glob("transactions_*.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute GM Fingerprint (Feature 4.18)")
    parser.add_argument("--season", type=int, default=_current_nhl_season())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    args = parser.parse_args()

    from models.gm_fingerprint import compute_gm_fingerprint, write_gm_fingerprint

    output_dir = args.data_dir / "gm_fingerprint"
    out_path = output_dir / f"gm_fingerprint_{args.season}.parquet"
    if out_path.exists() and not args.force:
        print(f"  Season {args.season}: exists, skipping (use --force)")
        return

    bs_path = args.data_dir / "buyer_seller" / f"buyer_seller_{args.season}.parquet"
    if not bs_path.exists():
        print(f"[gm-fingerprint] buyer_seller not found: {bs_path}", file=sys.stderr)
        sys.exit(1)

    bs_df = pl.read_parquet(bs_path)
    tx_df = _load_all_transactions(args.data_dir)
    print(f"  Loaded buyer_seller ({len(bs_df)} teams) + transactions ({len(tx_df)} rows)")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = compute_gm_fingerprint(bs_df, tx_df, season=args.season)
    for w in caught:
        print(f"  [WARN] {w.message}", file=sys.stderr)

    path = write_gm_fingerprint(df, output_dir, args.season)
    print(f"  Saved {path}  ({len(df)} rows)")
    print(f"\n  ── Top GM action archetypes ──")
    for r in df.head(10).iter_rows(named=True):
        print(f"    {r['team']:<4}  {r['action_archetype']:<16}  "
              f"agg {r['deadline_aggression']:.2f}  tx {r['recent_tx_count']}  "
              f"GM: {r['gm_name'] or '—'}")
    print("\nDone.")

if __name__ == "__main__":
    main()
