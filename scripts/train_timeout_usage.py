#!/usr/bin/env python3
"""Train Timeout Usage Model — Feature 4.4.

Requires::
    ~/.gretzky/data/raw/pbp_{season}.parquet

Usage::

    uv run python scripts/train_timeout_usage.py
    uv run python scripts/train_timeout_usage.py --seasons 2024 2025 --force

Outputs::
    ~/.gretzky/data/timeout_usage/timeout_usage_{season}.parquet
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


def _default_seasons() -> list[int]:
    return list(range(2023, _current_nhl_season() + 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Timeout Usage Model (Feature 4.4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seasons", nargs="+", type=int, default=_default_seasons(), metavar="YEAR")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    args = parser.parse_args()

    from models.rapm_model      import _NHL_TEAM_IDS
    from models.timeout_usage   import compute_timeout_usage, write_timeout_usage

    output_dir = args.data_dir / "timeout_usage"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    target_seasons = sorted(args.seasons)
    for season in target_seasons:
        out_path = output_dir / f"timeout_usage_{season}.parquet"
        if out_path.exists() and not args.force:
            print(f"  Season {season}: already exists, skipping (use --force)")
            continue

        pbp_path = args.data_dir / "raw" / f"pbp_{season}.parquet"
        if not pbp_path.exists():
            warnings.warn(f"Season {season}: pbp not found at {pbp_path}", stacklevel=2)
            continue

        print(f"  Loading season {season} pbp…")
        pbp_df = pl.read_parquet(pbp_path)

        print(f"  Computing timeout usage for {season}…")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = compute_timeout_usage(pbp_df, season=season, team_lookup=_NHL_TEAM_IDS)
        for w in caught:
            print(f"  [WARN] {w.message}", file=sys.stderr)

        path = write_timeout_usage(df, output_dir, season)
        print(f"  Saved {path}  ({len(df)} rows)")
        saved += 1

    print(f"\nDone. {saved}/{len(target_seasons)} seasons saved.")


if __name__ == "__main__":
    main()
