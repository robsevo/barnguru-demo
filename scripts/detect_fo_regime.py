#!/usr/bin/env python3
"""Detect Front Office Regime Changes — Feature 4.14.

Scans transactions for GM / AGM / Pres Hockey Ops changes.

Requires::
    ~/.gretzky/data/transactions/transactions_*.parquet

Usage::
    uv run python scripts/detect_fo_regime.py
    uv run python scripts/detect_fo_regime.py --season 2025 --force

Outputs::
    ~/.gretzky/data/fo_regime_changes/fo_regime_changes_{season}.parquet
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


def _load_all_transactions(data_dir: Path) -> pl.DataFrame:
    tx_dir = data_dir / "transactions"
    if not tx_dir.exists():
        return pl.DataFrame()
    files = sorted(tx_dir.glob("transactions_*.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect Front Office Regime Changes (Feature 4.14)")
    parser.add_argument("--season", type=int, default=_current_nhl_season())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    args = parser.parse_args()

    from models.fo_regime_detector import detect_fo_regime_changes, write_fo_regime_changes

    output_dir = args.data_dir / "fo_regime_changes"
    out_path = output_dir / f"fo_regime_changes_{args.season}.parquet"
    if out_path.exists() and not args.force:
        print(f"  Season {args.season}: already exists, skipping (use --force)")
        return

    tx = _load_all_transactions(args.data_dir)
    print(f"  Loaded {len(tx)} total transaction rows")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = detect_fo_regime_changes(tx, season=args.season)
    for w in caught:
        print(f"  [WARN] {w.message}", file=sys.stderr)

    path = write_fo_regime_changes(df, output_dir, args.season)
    print(f"  Saved {path}  ({len(df)} FO regime changes detected)")
    for r in df.iter_rows(named=True):
        print(f"    {r['date']}  {r['team']:<4}  {r['fo_role']:<24}  decay={r['decay_games']}  {r['person_out'] or '—'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
