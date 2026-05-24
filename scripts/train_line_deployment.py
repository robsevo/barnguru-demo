#!/usr/bin/env python3
"""Train Line Deployment Forecaster — Feature 4.1.

Predicts per-coach forward lines + D-pairs + minutes allocation from
shift, play-by-play, and shot data.

Requires::
    ~/.gretzky/data/raw/shifts_{season}.parquet
    ~/.gretzky/data/raw/pbp_{season}.parquet
    ~/.gretzky/data/shots/shots_{season}.parquet

Run prerequisite pipeline first::

    uv run python scripts/gretzky.py ingest
    uv run python scripts/gretzky.py sync

Usage::

    uv run python scripts/train_line_deployment.py
    uv run python scripts/train_line_deployment.py --seasons 2024 2025
    uv run python scripts/train_line_deployment.py --force

Outputs::
    ~/.gretzky/data/line_deployment/line_deployment_{season}.parquet
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


def _load_season(
    data_dir: Path,
    season:   int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    shifts_path = data_dir / "raw"   / f"shifts_{season}.parquet"
    pbp_path    = data_dir / "raw"   / f"pbp_{season}.parquet"
    shots_path  = data_dir / "shots" / f"shots_{season}.parquet"

    empty = pl.DataFrame()
    if not shifts_path.exists():
        warnings.warn(f"Season {season}: shifts not found at {shifts_path}", stacklevel=2)
        return empty, empty, empty
    if not pbp_path.exists():
        warnings.warn(f"Season {season}: pbp not found at {pbp_path}", stacklevel=2)
        return empty, empty, empty
    if not shots_path.exists():
        warnings.warn(f"Season {season}: shots not found at {shots_path}", stacklevel=2)
        return empty, empty, empty

    print(f"  Loading season {season} shifts/pbp/shots…")
    return (
        pl.read_parquet(shifts_path),
        pl.read_parquet(pbp_path),
        pl.read_parquet(shots_path),
    )


def _print_team(df: pl.DataFrame, team: str) -> None:
    sub = df.filter(pl.col("team") == team)
    if sub.is_empty():
        return
    print(f"\n  ── {team} projected lines ──")
    for r in sub.iter_rows(named=True):
        pids = [r["player_1"], r["player_2"], r["player_3"]]
        pids = [str(p) for p in pids if p is not None]
        line_min = (r["line_toi_per_game"] or 0.0) / 60.0
        trio_min = (r["trio_toi_per_game"] or 0.0) / 60.0
        print(
            f"    {r['line_type']}{r['line_rank']}  "
            f"{'-'.join(pids):<35} "
            f"line {line_min:5.2f}m/g  "
            f"trio {trio_min:5.2f}m/g  "
            f"cohesion {(r['cohesion_pct'] or 0)*100:4.0f}%  "
            f"share {r['share_of_team_toi']*100:5.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Line Deployment Forecaster (Feature 4.1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=_default_seasons(), metavar="YEAR",
        help="Season years to process.",
    )
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if output parquet exists.")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--preview-teams", nargs="*", default=["MTL", "TBL", "TOR"],
                        help="Print line previews for these team abbrevs.")
    args = parser.parse_args()

    from models.line_deployment import (
        compute_line_deployment,
        write_line_deployment,
    )
    from models.rapm_model import _NHL_TEAM_IDS

    output_dir = args.data_dir / "line_deployment"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    target_seasons = sorted(args.seasons)
    for season in target_seasons:
        out_path = output_dir / f"line_deployment_{season}.parquet"
        if out_path.exists() and not args.force:
            print(f"  Season {season}: already exists, skipping (use --force)")
            continue

        shifts_df, pbp_df, shots_df = _load_season(args.data_dir, season)
        if shifts_df.is_empty() or pbp_df.is_empty():
            print(f"  Season {season}: missing inputs, skipping.")
            continue

        print(f"  Computing line deployment for {season}…")
        df = compute_line_deployment(
            shifts_df, pbp_df, shots_df, season=season, team_lookup=_NHL_TEAM_IDS,
        )
        if df.is_empty():
            print(f"  Season {season}: no usable stints, skipping.")
            continue

        path = write_line_deployment(df, output_dir, season)
        print(f"  Saved {path}  ({len(df)} rows, {df['team'].n_unique()} teams)")
        for t in args.preview_teams:
            _print_team(df, t)
        saved += 1

    print(f"\nDone. {saved}/{len(target_seasons)} seasons saved.")


if __name__ == "__main__":
    main()
