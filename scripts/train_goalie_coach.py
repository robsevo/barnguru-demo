#!/usr/bin/env python3
"""Train Goalie Coach Curve — Feature 4.8.

Requires::
    ~/.gretzky/data/raw/pbp_{season}.parquet     (target season; prior season optional)

Usage::

    uv run python scripts/train_goalie_coach.py
    uv run python scripts/train_goalie_coach.py --season 2025 --force

Outputs::
    ~/.gretzky/data/goalie_coach_curve/goalie_coach_curve_{season}.parquet
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
    print("\n  ── Top teams by season save% ──")
    for r in df.head(n).iter_rows(named=True):
        prior = r["prior_save_pct"]
        prior_s = f"{prior:.3f}" if prior == prior else "—"   # NaN check
        delta = r["save_pct_delta"]
        delta_s = f"{delta:+.3f}" if delta == delta else "—"
        split = r["split_delta"]
        split_s = f"{split:+.3f}" if split == split else "—"
        flag = "*" if r["change_point_detected"] else " "
        print(
            f"    {r['team']:<4}  GP {r['gp']:<3}  "
            f"SV% {r['season_save_pct']:.3f}  "
            f"vs prior {prior_s} ({delta_s})  "
            f"split Δ {split_s} {flag}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Goalie Coach Curve (Feature 4.8)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--season", type=int, default=_current_nhl_season(), metavar="YEAR")
    parser.add_argument("--split-gp", type=int, default=15)
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    args = parser.parse_args()

    from models.goalie_coach import compute_goalie_coach_curve, write_goalie_coach_curve
    from models.rapm_model   import _NHL_TEAM_IDS

    output_dir = args.data_dir / "goalie_coach_curve"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"goalie_coach_curve_{args.season}.parquet"
    if out_path.exists() and not args.force:
        print(f"  Season {args.season}: already exists, skipping (use --force)")
        return

    pbp_by_season: dict[int, pl.DataFrame] = {}
    for s in (args.season, args.season - 1):
        pbp_path = args.data_dir / "raw" / f"pbp_{s}.parquet"
        if not pbp_path.exists():
            warnings.warn(f"Season {s}: pbp not found at {pbp_path}", stacklevel=2)
            continue
        print(f"  Loading season {s} pbp…")
        pbp_by_season[s] = pl.read_parquet(pbp_path)

    if args.season not in pbp_by_season:
        print(f"[goalie-coach] target season {args.season} has no PBP; aborting.")
        sys.exit(1)

    print(f"  Computing goalie coach curve for {args.season}…")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = compute_goalie_coach_curve(
            pbp_by_season    = pbp_by_season,
            season           = args.season,
            team_lookup      = _NHL_TEAM_IDS,
            split_gp         = args.split_gp,
            rolling_window   = args.rolling_window,
        )
    for w in caught:
        print(f"  [WARN] {w.message}", file=sys.stderr)

    path = write_goalie_coach_curve(df, output_dir, args.season)
    print(f"  Saved {path}  ({len(df)} rows)")
    _print_top(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
