#!/usr/bin/env python3
"""Train Special Teams Deployment Model — Feature 4.3.

Identifies per-team PP1/PP2/PK1/PK2 personnel and the first-unit share
of total special teams TOI.

Requires::
    ~/.gretzky/data/raw/shifts_{season}.parquet
    ~/.gretzky/data/raw/pbp_{season}.parquet

Run prerequisite pipeline first::

    uv run python scripts/gretzky.py ingest

Usage::

    uv run python scripts/train_st_deployment.py
    uv run python scripts/train_st_deployment.py --seasons 2024 2025 --force

Outputs::
    ~/.gretzky/data/st_deployment/st_deployment_{season}.parquet
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
    cur = _current_nhl_season()
    return list(range(2023, cur + 1))


def _print_team(df: pl.DataFrame, team: str) -> None:
    sub = df.filter(pl.col("team") == team)
    if sub.is_empty():
        return
    print(f"\n  ── {team} ST units ──")
    for r in sub.iter_rows(named=True):
        pers = "-".join(str(p) for p in r["personnel"])
        print(
            f"    {r['unit_type']:<4} "
            f"[{pers:<40}]  "
            f"TOI {r['unit_toi_secs']/60:6.1f}m  "
            f"share {r['share_of_st_toi']*100:5.1f}%  "
            f"GP {r['team_st_gp']:3d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Special Teams Deployment Model (Feature 4.3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=_default_seasons(), metavar="YEAR",
        help="Season years to process.",
    )
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if output parquet exists.")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--preview-teams", nargs="*", default=["MTL", "TBL", "EDM"],
                        help="Print ST unit previews for these team abbrevs.")
    args = parser.parse_args()

    from models.st_deployment import (
        compute_st_deployment,
        write_st_deployment,
    )
    from models.rapm_model import _NHL_TEAM_IDS

    output_dir = args.data_dir / "st_deployment"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    target_seasons = sorted(args.seasons)
    for season in target_seasons:
        out_path = output_dir / f"st_deployment_{season}.parquet"
        if out_path.exists() and not args.force:
            print(f"  Season {season}: already exists, skipping (use --force)")
            continue

        shifts_path = args.data_dir / "raw" / f"shifts_{season}.parquet"
        pbp_path    = args.data_dir / "raw" / f"pbp_{season}.parquet"
        if not shifts_path.exists() or not pbp_path.exists():
            warnings.warn(
                f"Season {season}: shifts or pbp parquet missing — skipping.",
                stacklevel=2,
            )
            continue

        print(f"  Loading season {season} shifts/pbp…")
        shifts_df = pl.read_parquet(shifts_path)
        pbp_df    = pl.read_parquet(pbp_path)

        print(f"  Computing ST deployment for {season}…")
        df = compute_st_deployment(
            shifts_df, pbp_df, season=season, team_lookup=_NHL_TEAM_IDS,
        )
        if df.is_empty():
            print(f"  Season {season}: no usable PP/PK windows, skipping.")
            continue

        path = write_st_deployment(df, output_dir, season)
        print(f"  Saved {path}  ({len(df)} rows, {df['team'].n_unique()} teams)")
        for t in args.preview_teams:
            _print_team(df, t)
        saved += 1

    print(f"\nDone. {saved}/{len(target_seasons)} seasons saved.")


if __name__ == "__main__":
    main()
