#!/usr/bin/env python3
"""Train Special Teams Ratings — Feature 2.7.

Computes per-player Power Play (PP) and Penalty Kill (PK) ratings from
NHL shift + MoneyPuck shot data.

Requires:
  - Shift data:    ~/.gretzky/data/raw/shifts_{season}.parquet
  - Shot data:     ~/.gretzky/data/shots/shots_{season}.parquet
  - RAPM output:   ~/.gretzky/data/rapm/rapm_{season}.parquet

Run the prerequisite pipeline first::

    uv run python scripts/gretzky.py ingest
    uv run python scripts/gretzky.py train-rapm

Usage::

    uv run python scripts/train_special_teams.py
    uv run python scripts/train_special_teams.py --seasons 2024 2025
    uv run python scripts/train_special_teams.py --force

Outputs:
    ~/.gretzky/data/special_teams/special_teams_{season}.parquet
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))


def _current_nhl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


def _default_seasons() -> list[int]:
    cur = _current_nhl_season()
    return list(range(2023, cur + 1))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_season(
    data_dir: Path,
    season: int,
    season_type: str = "regular",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load shifts, shots, and RAPM for a season.

    season_type filters shifts (by game_id encoded type) and shots (by
    is_playoff column), and reads the matching RAPM parquet variant.

    Returns:
        (shifts_df, shots_df, rapm_df).  Empty DataFrames on any missing file.
    """
    from models.rapm_model import DataMissingWarning

    suffix = "_playoffs" if season_type == "playoffs" else ""
    shifts_path = data_dir / "raw"   / f"shifts_{season}.parquet"
    shots_path  = data_dir / "shots" / f"shots_{season}.parquet"
    rapm_path   = data_dir / "rapm"  / f"rapm_{season}{suffix}.parquet"
    if season_type == "playoffs" and not rapm_path.exists():
        rapm_path = data_dir / "rapm" / f"rapm_{season}.parquet"

    empty = pl.DataFrame()

    if not shifts_path.exists():
        warnings.warn(
            f"Season {season}: shifts not found at {shifts_path}.\n"
            "  Run: uv run python scripts/gretzky.py ingest",
            DataMissingWarning,
            stacklevel=2,
        )
        return empty, empty, empty

    if not shots_path.exists():
        warnings.warn(
            f"Season {season}: shots not found at {shots_path}.\n"
            "  Run: uv run python scripts/gretzky.py sync",
            DataMissingWarning,
            stacklevel=2,
        )
        return empty, empty, empty

    if not rapm_path.exists():
        warnings.warn(
            f"Season {season}: RAPM not found at {rapm_path}.\n"
            "  Run: uv run python scripts/gretzky.py train-rapm",
            DataMissingWarning,
            stacklevel=2,
        )
        return empty, empty, empty

    print(f"  Loading season {season} ({season_type})...")
    shifts_df = pl.read_parquet(shifts_path)
    shots_df  = pl.read_parquet(shots_path)
    rapm_df   = pl.read_parquet(rapm_path)

    if season_type == "playoffs":
        if "game_id" in shifts_df.columns:
            gt = (pl.col("game_id") % 1_000_000) // 10_000
            shifts_df = shifts_df.filter(gt == 3)
        if "is_playoff" in shots_df.columns:
            shots_df = shots_df.filter(pl.col("is_playoff") == True)  # noqa: E712
    elif season_type == "regular":
        if "game_id" in shifts_df.columns:
            gt = (pl.col("game_id") % 1_000_000) // 10_000
            shifts_df = shifts_df.filter(gt == 2)
        if "is_playoff" in shots_df.columns:
            shots_df = shots_df.filter(pl.col("is_playoff") == False)  # noqa: E712
    return shifts_df, shots_df, rapm_df


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------


def _print_leaders(st_df: pl.DataFrame, season: int) -> None:
    """Print top-10 PP and PK leaders for a season."""
    from models.special_teams_model import top_pk_players, top_pp_players

    print(f"\n  ── Season {season} Top-10 PP Players ──")
    pp_top = top_pp_players(st_df, n=10)
    for r in pp_top.to_dicts():
        toi_min = r["toi_pp"] / 60.0
        print(
            f"    {r['player_name']:<22} {r['team']}  "
            f"PP: {r['pp_rating']:+.3f}  "
            f"xGF/60: {r['pp_xgf60']:.2f}  "
            f"TOI: {toi_min:.0f}m"
        )

    print(f"\n  ── Season {season} Top-10 PK Players ──")
    pk_top = top_pk_players(st_df, n=10)
    for r in pk_top.to_dicts():
        toi_min = r["toi_pk"] / 60.0
        print(
            f"    {r['player_name']:<22} {r['team']}  "
            f"PK: {r['pk_rating']:+.3f}  "
            f"xGA/60: {r['pk_xga60']:.2f}  "
            f"TOI: {toi_min:.0f}m"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Special Teams Ratings (PP + PK) from stints + RAPM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=_default_seasons(),
        metavar="YEAR",
        help="Season years to process (oldest → newest).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-compute even if output parquets already exist.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        metavar="PATH",
        help="Root data directory.",
    )
    parser.add_argument(
        "--season-type",
        choices=["regular", "playoffs"],
        default="regular",
        help="'regular' writes special_teams_{year}.parquet (default). "
             "'playoffs' filters shifts+shots to playoff games and writes "
             "special_teams_{year}_playoffs.parquet.",
    )
    args = parser.parse_args()

    data_dir:  Path      = args.data_dir
    output_dir: Path     = data_dir / "special_teams"
    output_dir.mkdir(parents=True, exist_ok=True)
    season_type: str     = args.season_type
    out_suffix           = "_playoffs" if season_type == "playoffs" else ""

    target_seasons: list[int] = sorted(args.seasons)

    # Skip seasons that already have output (unless --force)
    if not args.force:
        all_exist = all(
            (output_dir / f"special_teams_{s}{out_suffix}.parquet").exists()
            for s in target_seasons
        )
        if all_exist:
            print(
                f"All special-teams parquets exist for {target_seasons}. "
                "Use --force to recompute."
            )
            return

    # Verify at least one season has shifts
    has_shifts = any(
        (data_dir / "raw" / f"shifts_{s}.parquet").exists()
        for s in target_seasons
    )
    if not has_shifts:
        print(
            "ERROR: No shift data found for any target season.\n"
            "  Run: uv run python scripts/gretzky.py ingest",
            file=sys.stderr,
        )
        sys.exit(1)

    from models.rapm_model import build_stints
    from models.special_teams_model import compute_special_teams, write_special_teams

    saved = 0
    for season in target_seasons:
        out_path = output_dir / f"special_teams_{season}{out_suffix}.parquet"
        if out_path.exists() and not args.force:
            print(f"  Season {season}: already exists, skipping (use --force)")
            continue

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shifts_df, shots_df, rapm_df = _load_season(data_dir, season, season_type=season_type)

        for w in caught:
            print(f"  [WARN] {w.message}", file=sys.stderr)

        if shifts_df.is_empty() or shots_df.is_empty() or rapm_df.is_empty():
            print(f"  Season {season}: skipping — required data not available.")
            continue

        print(f"  Building stints for {season}...")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stints_df = build_stints(shifts_df, shots_df)

        for w in caught:
            print(f"  [WARN] {w.message}", file=sys.stderr)

        print(f"  Computing special teams ratings for {season}...")
        st_df = compute_special_teams(stints_df, rapm_df, season)

        path = write_special_teams(st_df, output_dir, season, season_type=season_type)
        print(f"  Saved {path}  ({len(st_df)} players)")
        _print_leaders(st_df, season)
        saved += 1

    print(f"\nDone. {saved}/{len(target_seasons)} seasons saved.")


if __name__ == "__main__":
    main()
