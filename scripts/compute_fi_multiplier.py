"""compute_fi_multiplier — Feature 3.18 driver script.

Reads the latest composite-FI parquet (Feature 3.17) and writes the
per-(player, game) rating multiplier that the Rust engine consumes.

Usage::

    uv run python scripts/gretzky.py fi-multiplier
    uv run python scripts/gretzky.py fi-multiplier -- --date 2026-05-17
    uv run python scripts/gretzky.py fi-multiplier -- --force
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

from models.fi_rating_multiplier import (
    FIRatingMultiplier,
    write_fi_multiplier,
)


DEFAULT_DATA_DIR = Path(
    os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data"))
)
FI_MULTIPLIER_SUBDIR = "fi_rating_multiplier"
COMPOSITE_FI_SUBDIR  = "composite_fi"


def _latest_composite_fi(args, as_of: str) -> pl.DataFrame:
    fi_dir = args.data_dir / COMPOSITE_FI_SUBDIR
    target = fi_dir / f"composite_fi_{as_of}.parquet"
    if target.exists():
        return pl.read_parquet(target)
    if not fi_dir.exists():
        return pl.DataFrame()
    candidates = sorted(fi_dir.glob("composite_fi_*.parquet"))
    if not candidates:
        return pl.DataFrame()
    return pl.read_parquet(candidates[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-(player, game) FI rating multiplier (Feature 3.18)."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None,
                        help="as-of date. Default: today (UTC).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-degradation", type=float, default=0.10,
                        help="Maximum derating at FI=1.0 (default 0.10).")
    args = parser.parse_args()

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir  = args.data_dir / FI_MULTIPLIER_SUBDIR
    out_path = out_dir / f"fi_rating_multiplier_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[fi-multiplier] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    fi = _latest_composite_fi(args, as_of)
    if len(fi) == 0:
        print("[fi-multiplier] No composite_fi parquet found.")
        print("  Run `composite-fi` first.")
        sys.exit(1)

    print(f"[fi-multiplier] Reading composite FI: {len(fi):,} rows")
    print(f"  max_degradation={args.max_degradation:.3f} "
          f"→ multiplier floor={1.0 - args.max_degradation:.3f}")

    model  = FIRatingMultiplier(max_degradation=args.max_degradation)
    result = model.compute(fi, as_of_date=as_of)
    n_rows = len(result)
    print(f"  {n_rows:,} multiplier rows produced.")

    if n_rows > 0:
        worst = result.sort("rating_multiplier").head(10)
        print("\n  10 most-derated (player, game) rows:")
        print(f"  {'Player':<10}  {'Game':<10}  {'Date':<10}  {'FI':>5}  {'Mult':>5}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*5}  {'─'*5}")
        for r in worst.to_dicts():
            print(
                f"  {r['player_id']:<10}  {r['game_id']:<10}  "
                f"{r['game_date']:<10}  {r['fatigue_index']:>5.2f}  "
                f"{r['rating_multiplier']:>5.3f}"
            )

    path = write_fi_multiplier(result, out_dir, as_of)
    print(f"\n[fi-multiplier] Written: {path}")


if __name__ == "__main__":
    main()
