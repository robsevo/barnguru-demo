#!/usr/bin/env python3
"""Train Line-Matching Model — Feature 4.2.

Predicts defensive counter-deployment matrices per team/coach: which of
your lines goes out when opponent's L1 is on the ice (and the
last-change advantage at home).

Requires::
    ~/.gretzky/data/raw/shifts_{season}.parquet
    ~/.gretzky/data/raw/pbp_{season}.parquet
    ~/.gretzky/data/line_deployment/line_deployment_{season}.parquet

Run prerequisite pipeline first::

    uv run python scripts/gretzky.py train-deployment

Usage::

    uv run python scripts/train_line_matching.py
    uv run python scripts/train_line_matching.py --seasons 2024 2025 --force

Outputs::
    ~/.gretzky/data/line_matching/line_matching_{season}.parquet
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


def _print_team(df: pl.DataFrame, team: str, line_type: str = "F") -> None:
    from models.line_matching import team_matchup_profile

    prof = team_matchup_profile(df, team, line_type=line_type)
    if prof.is_empty():
        return
    print(f"\n  ── {team} matchup profile ({line_type}) ──")
    for r in prof.iter_rows(named=True):
        print(
            f"    {r['venue']:<4}  "
            f"own L{r['own_line_rank']} vs opp L{r['opp_line_rank']}  "
            f"share {r['weighted_share']*100:5.1f}%  "
            f"toi {r['total_toi_secs']/60:6.0f}m"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Line-Matching Model (Feature 4.2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=_default_seasons(), metavar="YEAR",
        help="Season years to process.",
    )
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if output parquet exists.")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--preview-teams", nargs="*", default=["MTL", "TBL"],
                        help="Print matchup previews for these team abbrevs.")
    args = parser.parse_args()

    from models.line_matching import (
        compute_line_matching,
        write_line_matching,
        read_line_matching,  # noqa: F401  (kept for downstream consumers)
    )
    from models.line_deployment import read_line_deployment
    from models.rapm_model import _NHL_TEAM_IDS

    output_dir = args.data_dir / "line_matching"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    target_seasons = sorted(args.seasons)
    for season in target_seasons:
        out_path = output_dir / f"line_matching_{season}.parquet"
        if out_path.exists() and not args.force:
            print(f"  Season {season}: already exists, skipping (use --force)")
            continue

        shifts_path = args.data_dir / "raw" / f"shifts_{season}.parquet"
        pbp_path    = args.data_dir / "raw" / f"pbp_{season}.parquet"
        ldep_path   = args.data_dir / "line_deployment" / f"line_deployment_{season}.parquet"

        if not shifts_path.exists() or not pbp_path.exists():
            warnings.warn(
                f"Season {season}: missing shifts or pbp parquet — skipping.",
                stacklevel=2,
            )
            continue
        if not ldep_path.exists():
            warnings.warn(
                f"Season {season}: line_deployment parquet missing.\n"
                f"  Run: uv run python scripts/gretzky.py train-deployment",
                stacklevel=2,
            )
            continue

        print(f"  Loading season {season} shifts/pbp/line_deployment…")
        shifts_df = pl.read_parquet(shifts_path)
        pbp_df    = pl.read_parquet(pbp_path)
        ldep_df   = pl.read_parquet(ldep_path)

        print(f"  Computing line matching for {season}…")
        df = compute_line_matching(
            shifts_df, pbp_df, ldep_df, season=season, team_lookup=_NHL_TEAM_IDS,
        )
        if df.is_empty():
            print(f"  Season {season}: no usable stints, skipping.")
            continue

        path = write_line_matching(df, output_dir, season)
        print(f"  Saved {path}  ({len(df)} rows, {df['team'].n_unique()} focal teams)")
        for t in args.preview_teams:
            _print_team(df, t, line_type="F")
        saved += 1

    print(f"\nDone. {saved}/{len(target_seasons)} seasons saved.")


if __name__ == "__main__":
    main()
