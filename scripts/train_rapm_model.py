#!/usr/bin/env python3
"""Train the RAPM model (Feature 2.2) with daisy-chain seasonal priors.

Requires:
  - NHL shift data:  ~/.gretzky/data/raw/shifts_{season}.parquet
                     (run: uv run python scripts/run_phase1_sync.py to populate)
  - MoneyPuck shots: ~/.gretzky/data/shots/shots_{season}.parquet
                     (already populated if phase 1 sync has run)

Usage::

    uv run python scripts/train_rapm_model.py
    uv run python scripts/train_rapm_model.py --seasons 2023 2024 2025
    uv run python scripts/train_rapm_model.py --chain-from 2015 --seasons 2025 --force

Outputs:
    ~/.gretzky/data/rapm/rapm_{season}.parquet   — per-player RAPM estimates
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
import os

from datetime import datetime, timezone

import polars as pl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))


def _current_nhl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


def _default_seasons() -> list[int]:
    cur = _current_nhl_season()
    return list(range(2023, cur + 1))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_season(data_dir: Path, season: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load shifts + shots for a season. Returns (shifts_df, shots_df).

    Raises DataMissingWarning (returns empty DFs) if files are absent.
    """
    from models.rapm_model import DataMissingWarning

    shifts_path = data_dir / "raw" / f"shifts_{season}.parquet"
    shots_path  = data_dir / "shots" / f"shots_{season}.parquet"

    if not shifts_path.exists():
        warnings.warn(
            f"Season {season}: shifts file not found at {shifts_path}.\n"
            "  Run: uv run python scripts/run_phase1_sync.py",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(), pl.DataFrame()

    if not shots_path.exists():
        warnings.warn(
            f"Season {season}: shots file not found at {shots_path}.\n"
            "  Run: uv run python scripts/run_phase1_sync.py",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(), pl.DataFrame()

    print(f"  Loading season {season}: {shifts_path.name}, {shots_path.name}")
    shifts_df = pl.read_parquet(shifts_path)
    shots_df  = pl.read_parquet(shots_path)
    return shifts_df, shots_df


def _print_leaders(result: pl.DataFrame, season: int) -> None:
    """Print top 10 EV offensive and defensive RAPM leaders."""
    print(f"\n  ── Season {season} EV Offensive RAPM leaders ──")
    top_off = (
        result
        .filter(pl.col("toi_ev") >= 3600)  # min 60 min EV TOI
        .sort("rapm_ev_off", descending=True)
        .head(10)
        .select(["player_name", "team", "rapm_ev_off", "toi_ev"])
    )
    for r in top_off.to_dicts():
        toi_min = r["toi_ev"] / 60.0
        print(f"    {r['player_name']:<22} {r['team']}  EV_Off: {r['rapm_ev_off']:+.3f}  TOI: {toi_min:.0f}m")

    print(f"\n  ── Season {season} EV Defensive RAPM leaders ──")
    top_def = (
        result
        .filter(pl.col("toi_ev") >= 3600)
        .sort("rapm_ev_def", descending=False)   # lower xGA = better defender
        .head(10)
        .select(["player_name", "team", "rapm_ev_def", "toi_ev"])
    )
    for r in top_def.to_dicts():
        toi_min = r["toi_ev"] / 60.0
        print(f"    {r['player_name']:<22} {r['team']}  EV_Def: {r['rapm_ev_def']:+.3f}  TOI: {toi_min:.0f}m")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RAPM model with daisy-chain priors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=_default_seasons(),
        metavar="YEAR",
        help="Season years to fit (oldest → newest). "
             "Each season uses the previous season's posterior as its prior. "
             "Defaults to the 5 most recent NHL seasons.",
    )
    parser.add_argument(
        "--chain-from",
        type=int,
        default=None,
        metavar="YEAR",
        help="Extend the daisy chain back to this year to warm up priors "
             "(seasons before --seasons are fit but not saved).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-train even if output parquets already exist.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        metavar="PATH",
        help="Root data directory.",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    output_dir = data_dir / "rapm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if all outputs already exist
    target_seasons: list[int] = sorted(args.seasons)
    if not args.force:
        all_exist = all(
            (output_dir / f"rapm_{s}.parquet").exists()
            for s in target_seasons
        )
        if all_exist:
            print(
                f"All RAPM parquets already exist for {target_seasons}. "
                "Use --force to retrain."
            )
            return

    # Build full season list (warmup + target)
    warmup_seasons: list[int] = []
    if args.chain_from is not None:
        warmup_seasons = sorted(
            s for s in range(args.chain_from, min(target_seasons))
            if s not in target_seasons
        )

    all_seasons = warmup_seasons + target_seasons
    print(f"Daisy chain: {all_seasons}")
    print(f"Saving outputs for: {target_seasons}")
    print(f"Output dir: {output_dir}\n")

    # Verify at least one season has shifts data
    has_shifts = any(
        (data_dir / "raw" / f"shifts_{s}.parquet").exists()
        for s in target_seasons
    )
    if not has_shifts:
        print(
            "ERROR: No shift data found for any target season.\n"
            "  Shift data is required to build the stints matrix.\n\n"
            "  Run the historical ingestor first:\n"
            "    uv run python scripts/gretzky.py ingest\n\n"
            "  This will populate:\n"
            f"    {data_dir / 'raw' / 'shifts_<season>.parquet'}\n"
            f"    {data_dir / 'raw' / 'pbp_<season>.parquet'}",
            file=sys.stderr,
        )
        sys.exit(1)

    from models.rapm_model import RAPMModel

    model = RAPMModel()

    # Build a cross-season player name lookup from seasons that have shooter_id
    # populated (2024+). Used as fallback for 2021-2023 where MoneyPuck shooter_id
    # is null and names would otherwise display as "player_8478007".
    name_fallback: dict[int, str] = {}
    for _ns in (2024, 2025):
        _shots_path = data_dir / "shots" / f"shots_{_ns}.parquet"
        if _shots_path.exists():
            _s = pl.read_parquet(_shots_path, columns=["shooter_id", "shooter_name"])
            for _r in _s.drop_nulls("shooter_id").unique("shooter_id").to_dicts():
                name_fallback[int(_r["shooter_id"])] = str(_r["shooter_name"])
    model.name_fallback = name_fallback

    def loader(season: int) -> tuple[pl.DataFrame, pl.DataFrame]:
        return _load_season(data_dir, season)

    print("Training RAPM daisy chain...\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = model.fit_daisy_chain(
            seasons=all_seasons,
            data_loader_fn=loader,
        )

    for w in caught:
        print(f"  [WARN] {w.message}", file=sys.stderr)

    # Save target seasons only
    saved = 0
    for season in target_seasons:
        if season not in results:
            print(f"  Season {season}: no result (data missing?)")
            continue

        result = results[season]
        out_path = output_dir / f"rapm_{season}.parquet"

        if out_path.exists() and not args.force:
            print(f"  Season {season}: already exists, skipping (use --force)")
            continue

        result.write_parquet(out_path)
        print(f"  Saved {out_path}  ({len(result)} players)")
        _print_leaders(result, season)
        saved += 1

    print(f"\nDone. {saved}/{len(target_seasons)} seasons saved.")

    # Save model object
    model_path = data_dir / "rapm" / "rapm_model.pkl"
    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
