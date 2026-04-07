#!/usr/bin/env python3
"""Save a full GRTZKY system snapshot (Feature 2.16).

Captures the complete in-memory GRTZKY state — Bayesian pipeline, EWMA
form DataFrame, RegimeChangeDetector state, and StrengthOfScheduleNormalizer
— into a single dated .pkl file for reproducible replay and historical
comparison.

Requires:
    ~/.gretzky/data/inseason/inseason_pipeline_{season}.pkl  (run inseason-bayes)
    ~/.gretzky/data/ewma/ewma_form_{season}.parquet          (run compute-ewma)
    ~/.gretzky/data/regime/regime_detector_{season}.pkl      (run detect-regime)
    ~/.gretzky/data/sos/sos_normalizer_{season}.pkl          (run normalize-sos)

Usage::

    uv run python scripts/snapshot_gretzky.py
    uv run python scripts/snapshot_gretzky.py --season 2025
    uv run python scripts/snapshot_gretzky.py --notes "Pre-trade deadline state"
    uv run python scripts/snapshot_gretzky.py --list

Outputs:
    ~/.gretzky/snapshots/GRETZKY_as_of_{date}.pkl
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

_DEFAULT_DATA_DIR   = Path.home() / ".gretzky" / "data"
_DEFAULT_SNAP_DIR   = Path.home() / ".gretzky" / "snapshots"


def _current_nhl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


def _try_load(path: Path, loader):
    """Load from path using loader(), return None on any error."""
    if not path.exists():
        print(f"  [skip] Not found: {path}")
        return None
    try:
        return loader(path)
    except Exception as exc:
        print(f"  [warn] Failed to load {path}: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save a full GRTZKY system snapshot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--season",
        type=int,
        default=_current_nhl_season(),
        metavar="YEAR",
        help="Season to snapshot.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-text note attached to the snapshot.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom snapshot name (default: GRETZKY_as_of_{date}).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_snaps",
        help="List existing snapshots and exit.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("SNAP_A", "SNAP_B"),
        help="Compare two snapshots by name and print diff.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        metavar="PATH",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=_DEFAULT_SNAP_DIR,
        metavar="PATH",
    )
    args = parser.parse_args()

    from models.snapshot_system import GRETZKYSnapshot, SnapshotIndex

    snap_dir = args.snapshot_dir
    index    = SnapshotIndex(snap_dir)

    # ── List mode ──────────────────────────────────────────────────────────
    if args.list_snaps:
        entries = index.list()
        if not entries:
            print("No snapshots found.")
            return
        print(f"{'Name':<35}  {'Season':>6}  {'Games':>5}  {'Created':>24}  Notes")
        print("─" * 90)
        for e in entries:
            name    = e.get("name", "?")[:35]
            season  = e.get("season", "")
            games   = e.get("game_count", "")
            created = e.get("created_at", "")[:24]
            notes   = e.get("notes", "")[:30]
            print(f"{name:<35}  {season:>6}  {games:>5}  {created:>24}  {notes}")
        return

    # ── Compare mode ───────────────────────────────────────────────────────
    if args.compare:
        name_a, name_b = args.compare
        try:
            snap_a = index.load(name_a)
            snap_b = index.load(name_b)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        diff = GRETZKYSnapshot.compare(snap_a, snap_b)
        print(f"Comparing  {diff['name_a']}  ↔  {diff['name_b']}")
        print(f"  Season:          {diff['season_a']} → {diff['season_b']}")
        print(f"  Game count:      {diff['game_count_a']} → {diff['game_count_b']}  (Δ {diff['game_count_delta']:+d})")
        print(f"  Players:         {diff['n_players_a']} → {diff['n_players_b']}")
        print(f"  Active regimes:  {diff['n_active_regimes_a']} → {diff['n_active_regimes_b']}")
        print(f"  EWMA mean xGF60: {diff['ewma_mean_xgf60_a']} → {diff['ewma_mean_xgf60_b']}")
        return

    # ── Capture snapshot ───────────────────────────────────────────────────
    season   = args.season
    data_dir = args.data_dir

    print(f"Capturing GRTZKY snapshot for season {season}...\n")

    # Load in-season pipeline
    from models.inseason_bayesian import InSeasonBayesianPipeline
    pipeline_path = data_dir / "inseason" / f"inseason_pipeline_{season}.pkl"
    pipeline = _try_load(pipeline_path, InSeasonBayesianPipeline.load)

    # Load EWMA form summary
    ewma_path = data_dir / "ewma" / f"ewma_form_{season}.parquet"
    ewma_df = _try_load(ewma_path, pl.read_parquet)

    # Load regime detector
    from models.regime_change_detector import RegimeChangeDetector
    detector_path = data_dir / "regime" / f"regime_detector_{season}.pkl"
    detector = _try_load(detector_path, RegimeChangeDetector.load)

    # Load SoS normalizer
    from models.schedule_normalizer import StrengthOfScheduleNormalizer
    sos_path = data_dir / "sos" / f"sos_normalizer_{season}.pkl"
    sos_norm = _try_load(sos_path, StrengthOfScheduleNormalizer.load)

    # Determine game_count from pipeline or EWMA
    game_count = 0
    if pipeline is not None:
        try:
            game_count = pipeline.game_count
        except AttributeError:
            pass
    if game_count == 0 and ewma_df is not None and "n_games" in ewma_df.columns:
        game_count = int(ewma_df["n_games"].max() or 0)

    # Capture snapshot
    snap = GRETZKYSnapshot.capture(
        pipeline        = pipeline,
        season          = season,
        game_count      = game_count,
        ewma_df         = ewma_df,
        regime_detector = detector,
        sos_normalizer  = sos_norm,
        name            = args.name,
        notes           = args.notes,
    )

    print(snap.summary())
    print()

    # Save
    path = index.save(snap)
    print(f"Snapshot saved → {path}")
    print(f"Index now has {len(index)} snapshot(s) in {snap_dir}")


if __name__ == "__main__":
    main()
