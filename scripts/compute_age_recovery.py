"""compute_age_recovery — Feature 3.12 driver script.

Reads the cached roster JSONs via the DataStore, computes each player's
age as of today (or ``--date``), maps to a recovery coefficient via the
exponential-decay model, and writes per-player coefficients to parquet.

Usage::

    uv run python scripts/compute_age_recovery.py
    uv run python scripts/compute_age_recovery.py --date 2026-05-17
    uv run python scripts/compute_age_recovery.py --force
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from data.data_store import DataStore
from models.age_recovery import (
    AgeRecoveryCalculator,
    write_age_recovery,
)


DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
AGE_RECOVERY_SUBDIR = "age_recovery"


def _load_roster(args) -> pl.DataFrame:
    with DataStore(args.data_dir) as store:
        roster = store.roster()
    if len(roster) == 0:
        print("[age-recovery] DataStore has no roster JSON. Run `sync` first.")
        sys.exit(1)
    return roster.select(["player_id", "birth_date", "age"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-player age-recovery coefficients (Feature 3.12)."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None,
                        help="Date for age computation. Default: today (UTC).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    roster = _load_roster(args)
    print(f"  {len(roster):,} roster rows")

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir  = args.data_dir / AGE_RECOVERY_SUBDIR
    out_path = out_dir / f"age_recovery_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[age-recovery] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[age-recovery] Computing recovery coefficients as of {as_of}…")
    result = AgeRecoveryCalculator().compute(roster, as_of_date=as_of)
    n_rows = len(result)
    print(f"  {n_rows:,} player rows produced.")

    if n_rows > 0:
        oldest = result.sort("age_years", descending=True).head(10)
        print("\n  Top 10 oldest players (slowest recovery):")
        print(f"  {'Player':<10}  {'Age':>5}  {'Recovery':>8}")
        print(f"  {'─'*10}  {'─'*5}  {'─'*8}")
        for row in oldest.to_dicts():
            print(
                f"  {row['player_id']:<10}  {row['age_years']:>5.1f}  "
                f"{row['recovery_coefficient']:>8.3f}"
            )

    path = write_age_recovery(result, out_dir, as_of)
    print(f"\n[age-recovery] Written: {path}")


if __name__ == "__main__":
    main()
