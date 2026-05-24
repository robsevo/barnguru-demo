#!/usr/bin/env python3
"""Train PK Coordinator Model — Feature 4.10.

Requires::
    ~/.gretzky/data/shots/shots_{season}.parquet
    ~/.gretzky/data/st_deployment/st_deployment_{season}.parquet

Usage::

    uv run python scripts/train_pk_coordinator.py
    uv run python scripts/train_pk_coordinator.py --season 2025 --force

Outputs::
    ~/.gretzky/data/pk_coordinator/pk_coordinator_{season}.parquet
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


def _print_top(df: pl.DataFrame, n: int = 10) -> None:
    print("\n  ── Top PK systems by SV% ──")
    for r in df.head(n).iter_rows(named=True):
        print(
            f"    {r['team']:<4}  "
            f"SV% {r['pk_save_pct']:.3f}  "
            f"SA/60 {r['pk_sa_per_60']:5.1f}  "
            f"xGA/60 {r['pk_xga_per_60']:5.2f}  "
            f"GA/60 {r['pk_ga_per_60']:5.2f}  "
            f"SH-sh/60 {r['sh_shots_per_60']:4.1f}  "
            f"PK1 {r['pk1_share']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PK Coordinator Model (Feature 4.10)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--season", type=int, default=_current_nhl_season(), metavar="YEAR")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    args = parser.parse_args()

    from models.pk_coordinator import compute_pk_coordinator, write_pk_coordinator

    output_dir = args.data_dir / "pk_coordinator"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"pk_coordinator_{args.season}.parquet"
    if out_path.exists() and not args.force:
        print(f"  Season {args.season}: already exists, skipping (use --force)")
        return

    shots_path = args.data_dir / "shots" / f"shots_{args.season}.parquet"
    st_path    = args.data_dir / "st_deployment" / f"st_deployment_{args.season}.parquet"

    for p in (shots_path, st_path):
        if not p.exists():
            print(f"[pk-coordinator] required file missing: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"  Loading shots + st_deployment for {args.season}…")
    shots_df = pl.read_parquet(shots_path)
    st_df    = pl.read_parquet(st_path)

    print(f"  Computing PK coordinator signature for {args.season}…")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = compute_pk_coordinator(
            shots_df = shots_df,
            st_df    = st_df,
            season   = args.season,
        )
    for w in caught:
        print(f"  [WARN] {w.message}", file=sys.stderr)

    path = write_pk_coordinator(df, output_dir, args.season)
    print(f"  Saved {path}  ({len(df)} rows)")
    _print_top(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
