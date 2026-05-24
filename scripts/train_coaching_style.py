#!/usr/bin/env python3
"""Train Coaching Style Vector — Feature 4.11.

Requires::
    ~/.gretzky/data/raw/pbp_{season}.parquet
    ~/.gretzky/data/line_deployment/line_deployment_{season}.parquet
    ~/.gretzky/data/pp_coordinator/pp_coordinator_{season}.parquet  (optional;
        falls back to zero for st_aggression dim when absent)

Usage::

    uv run python scripts/train_coaching_style.py
    uv run python scripts/train_coaching_style.py --season 2025 --force

Outputs::
    ~/.gretzky/data/coaching_style/coaching_style_{season}.parquet
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
    print("\n  ── Style ranks (1.0 = most extreme on this dim) ──")
    cols = [
        "forecheck_aggression_rank", "dz_structure_rank", "pace_rank",
        "physicality_rank", "oz_structure_rank", "nz_tendency_rank",
        "line_match_rank", "st_aggression_rank",
    ]
    for r in df.head(n).iter_rows(named=True):
        vals = " ".join(
            f"{(r[c] if r[c] == r[c] else float('nan')):>5.2f}"
            if r[c] is not None else "  -- "
            for c in cols
        )
        print(f"    {r['team']:<4}  fore dz pace phys oz nz lm st  =  {vals}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Coaching Style Vector (Feature 4.11)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--season", type=int, default=_current_nhl_season(), metavar="YEAR")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    args = parser.parse_args()

    from models.coaching_style import compute_coaching_style, write_coaching_style
    from models.rapm_model     import _NHL_TEAM_IDS

    output_dir = args.data_dir / "coaching_style"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"coaching_style_{args.season}.parquet"
    if out_path.exists() and not args.force:
        print(f"  Season {args.season}: already exists, skipping (use --force)")
        return

    pbp_path   = args.data_dir / "raw" / f"pbp_{args.season}.parquet"
    lines_path = args.data_dir / "line_deployment" / f"line_deployment_{args.season}.parquet"
    pp_path    = args.data_dir / "pp_coordinator" / f"pp_coordinator_{args.season}.parquet"

    for p in (pbp_path, lines_path):
        if not p.exists():
            print(f"[coaching-style] required file missing: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"  Loading PBP + line_deployment for {args.season}…")
    pbp_df   = pl.read_parquet(pbp_path)
    lines_df = pl.read_parquet(lines_path)
    pp_df    = pl.read_parquet(pp_path) if pp_path.exists() else pl.DataFrame()
    if not pp_path.exists():
        print(f"  [INFO] pp_coordinator parquet absent — st_aggression dim will be zero.")

    print(f"  Computing coaching style for {args.season}…")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = compute_coaching_style(
            pbp_df         = pbp_df,
            lines_df       = lines_df,
            pp_coordinator = pp_df,
            team_lookup    = _NHL_TEAM_IDS,
            season         = args.season,
        )
    for w in caught:
        print(f"  [WARN] {w.message}", file=sys.stderr)

    path = write_coaching_style(df, output_dir, args.season)
    print(f"  Saved {path}  ({len(df)} rows)")
    _print_top(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
