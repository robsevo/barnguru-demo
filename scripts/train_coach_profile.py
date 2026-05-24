#!/usr/bin/env python3
"""Train Coach Profile Database — Feature 4.7.

Requires::
    ~/.gretzky/data/raw/pbp_{season}.parquet  (current + ideally prior seasons)
    data/coaches.json

Usage::

    uv run python scripts/train_coach_profile.py
    uv run python scripts/train_coach_profile.py --seasons 2024 2025 --force

Outputs::
    ~/.gretzky/data/coach_profiles/coach_profiles_{latest_season}.parquet
"""

from __future__ import annotations

import argparse
import json
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
_COACHES_JSON = _REPO / "data" / "coaches.json"


def _current_nhl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


def _default_seasons() -> list[int]:
    return list(range(2023, _current_nhl_season() + 1))


def _load_coaches() -> list[dict]:
    if not _COACHES_JSON.exists():
        raise FileNotFoundError(f"coaches.json not found at {_COACHES_JSON}")
    return json.loads(_COACHES_JSON.read_text()).get("coaches", []) or []


def _print_top(df: pl.DataFrame, n: int = 10) -> None:
    print("\n  ── Top coaches by points% ──")
    for r in df.head(n).iter_rows(named=True):
        print(
            f"    {r['team']:<4}  {r['coach_name']:<24}  "
            f"GP {r['gp_under_coach']:<3}  "
            f"P% {r['points_pct']:.3f}  "
            f"GF/G {r['gf_per_game']:.2f}  "
            f"PP% {r['pp_pct']:.3f}  "
            f"PK% {r['pk_pct']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Coach Profile Database (Feature 4.7)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seasons", nargs="+", type=int, default=_default_seasons(), metavar="YEAR")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    args = parser.parse_args()

    from models.coach_profile import compute_coach_profiles, write_coach_profiles
    from models.rapm_model    import _NHL_TEAM_IDS

    output_dir = args.data_dir / "coach_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_seasons = sorted(args.seasons)
    if not target_seasons:
        print("[coach-profile] no seasons given — nothing to do.")
        return
    snapshot_season = target_seasons[-1]

    out_path = output_dir / f"coach_profiles_{snapshot_season}.parquet"
    if out_path.exists() and not args.force:
        print(f"  Snapshot {snapshot_season}: already exists, skipping (use --force)")
        return

    pbp_by_season: dict[int, pl.DataFrame] = {}
    for season in target_seasons:
        pbp_path = args.data_dir / "raw" / f"pbp_{season}.parquet"
        if not pbp_path.exists():
            warnings.warn(f"Season {season}: pbp not found at {pbp_path}", stacklevel=2)
            continue
        print(f"  Loading season {season} pbp…")
        pbp_by_season[season] = pl.read_parquet(pbp_path)

    coaches = _load_coaches()
    print(f"  Loaded {len(coaches)} head coaches from data/coaches.json")

    print(f"  Computing coach profiles (snapshot season {snapshot_season})…")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = compute_coach_profiles(
            pbp_by_season   = pbp_by_season,
            coaches         = coaches,
            team_lookup     = _NHL_TEAM_IDS,
            snapshot_season = snapshot_season,
        )
    for w in caught:
        print(f"  [WARN] {w.message}", file=sys.stderr)

    path = write_coach_profiles(df, output_dir, snapshot_season)
    print(f"  Saved {path}  ({len(df)} rows)")
    _print_top(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
