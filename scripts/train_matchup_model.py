#!/usr/bin/env python3
"""Train the Matchup Efficiency Matrix (Feature 2.3).

Requires (per season):
  - Shift data:       ~/.gretzky/data/raw/shifts_{season}.parquet
  - MoneyPuck shots:  ~/.gretzky/data/shots/shots_{season}.parquet
  - RAPM parquets:    ~/.gretzky/data/rapm/rapm_{season}.parquet
                      (run: uv run python scripts/train_rapm_model.py first)
  - vs_team splits:   ~/.gretzky/data/raw/vs_team_{season}.parquet   (optional)
  - player stats:     ~/.gretzky/data/raw/player_stats_{season}.parquet (optional)

Usage::

    uv run python scripts/train_matchup_model.py
    uv run python scripts/train_matchup_model.py --seasons 2023 2024 2025
    uv run python scripts/train_matchup_model.py --seasons 2025 --force

Outputs:
    ~/.gretzky/data/matchup/matchup_model.pkl
    ~/.gretzky/data/qot_qoc/qot_qoc_{season}.parquet
    ~/.gretzky/data/matchup_preds/matchup_preds_{season}.parquet
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
import os

import polars as pl

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


def _load_season(
    data_dir: Path,
    season: int,
    season_type: str = "regular",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None, pl.DataFrame | None]:
    """Load all inputs for one season.

    season_type filters shifts (by game_id type) + shots (by is_playoff)
    and selects the matching RAPM parquet variant.

    Returns:
        (stints_df, rapm_df, vs_team_df, player_stats_df)
        stints_df is None-equivalent (empty DF) if shifts or shots are missing.
    """
    from models.rapm_model import DataMissingWarning, build_stints

    suffix = "_playoffs" if season_type == "playoffs" else ""
    shifts_path      = data_dir / "raw"   / f"shifts_{season}.parquet"
    shots_path       = data_dir / "shots" / f"shots_{season}.parquet"
    rapm_path        = data_dir / "rapm"  / f"rapm_{season}{suffix}.parquet"
    if season_type == "playoffs" and not rapm_path.exists():
        rapm_path = data_dir / "rapm" / f"rapm_{season}.parquet"
    vs_team_path     = data_dir / "raw"   / f"vs_team_{season}.parquet"
    player_stats_path = data_dir / "raw"  / f"player_stats_{season}.parquet"

    for label, path in [("shifts", shifts_path), ("shots", shots_path), ("rapm", rapm_path)]:
        if not path.exists():
            warnings.warn(
                f"Season {season}: {label} file not found at {path}.\n"
                "  Run the historical ingestor + train_rapm_model.py first.",
                DataMissingWarning,
                stacklevel=2,
            )
            return pl.DataFrame(), pl.DataFrame(), None, None

    print(f"  Loading {season} ({season_type}): shifts, shots, rapm", end="", flush=True)

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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stints_df = build_stints(shifts_df, shots_df)

    vs_team_df = None
    if vs_team_path.exists():
        vs_team_df = pl.read_parquet(vs_team_path)
        print(f", vs_team", end="", flush=True)

    player_stats_df = None
    if player_stats_path.exists():
        player_stats_df = pl.read_parquet(player_stats_path)
        print(f", player_stats", end="", flush=True)

    print(f"  →  {len(stints_df):,} stints  |  {len(rapm_df)} RAPM players")
    return stints_df, rapm_df, vs_team_df, player_stats_df


def _print_qot_qoc_leaders(qot_qoc_df: pl.DataFrame, season: int) -> None:
    """Print top 10 QoT and top 10 QoC leaders."""
    min_toi = 3600.0  # 60 min EV TOI

    print(f"\n  ── Season {season} Quality-of-Teammates (QoT) leaders ──")
    top_qot = (
        qot_qoc_df
        .filter(pl.col("toi_ev") >= min_toi)
        .sort("qot", descending=True)
        .head(10)
    )
    for r in top_qot.to_dicts():
        print(
            f"    {r['player_name']:<22} {r['team']}  "
            f"QoT: {r['qot']:+.3f}  QoC: {r['qoc']:+.3f}  "
            f"QoT-TOI: {r['qot_toi']/60:.0f}m"
        )

    print(f"\n  ── Season {season} Quality-of-Competition (QoC) leaders ──")
    top_qoc = (
        qot_qoc_df
        .filter(pl.col("toi_ev") >= min_toi)
        .sort("qoc", descending=True)
        .head(10)
    )
    for r in top_qoc.to_dicts():
        print(
            f"    {r['player_name']:<22} {r['team']}  "
            f"QoC: {r['qoc']:+.3f}  QoT: {r['qot']:+.3f}  "
            f"QoC-TOI: {r['qoc_toi']/60:.0f}m"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Feature 2.3 — Matchup Efficiency Matrix (QoT, QoC, pair xGF%).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=_default_seasons(),
        metavar="YEAR",
        help="Season years to fit and predict (e.g. 2023 2024 2025). "
             "Defaults to the 5 most recent NHL seasons.",
    )
    parser.add_argument(
        "--train-seasons",
        nargs="+",
        type=int,
        default=None,
        metavar="YEAR",
        help="Seasons used ONLY for model training (not saved as predictions). "
             "Defaults to all --seasons except the last.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output parquets already exist.",
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
        help="'regular' writes qot_qoc_{year}.parquet (default). "
             "'playoffs' filters shifts+shots to playoff games and writes "
             "qot_qoc_{year}_playoffs.parquet + matchup_preds_{year}_playoffs.parquet.",
    )
    args = parser.parse_args()

    data_dir: Path    = args.data_dir
    target_seasons    = sorted(args.seasons)
    season_type:  str = args.season_type
    out_suffix        = "_playoffs" if season_type == "playoffs" else ""
    matchup_dir       = data_dir / "matchup"
    qot_qoc_dir       = data_dir / "qot_qoc"
    matchup_preds_dir = data_dir / "matchup_preds"

    for d in (matchup_dir, qot_qoc_dir, matchup_preds_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Check if any output already exists
    if not args.force:
        existing = [
            s for s in target_seasons
            if (qot_qoc_dir / f"qot_qoc_{s}{out_suffix}.parquet").exists()
            and (matchup_preds_dir / f"matchup_preds_{s}{out_suffix}.parquet").exists()
        ]
        if existing:
            print(
                f"Outputs already exist for seasons {existing}. "
                "Use --force to retrain."
            )
            if len(existing) == len(target_seasons):
                return

    # Verify at least one season has shift data
    has_shifts = any(
        (data_dir / "raw" / f"shifts_{s}.parquet").exists()
        for s in target_seasons
    )
    if not has_shifts:
        print(
            "ERROR: No shift data found for any target season.\n"
            "  Run the historical ingestor first:\n"
            "    uv run python scripts/gretzky.py ingest\n\n"
            "  This will populate:\n"
            f"    {data_dir / 'raw' / 'shifts_<season>.parquet'}",
            file=sys.stderr,
        )
        sys.exit(1)

    from models.matchup_model import (
        MatchupModel,
        compute_qot_qoc,
        write_matchup_preds,
        write_qot_qoc,
    )
    from models.rapm_model import DataMissingWarning

    # ── Load all seasons ─────────────────────────────────────────────────────
    print(f"\nLoading data for seasons {target_seasons}…\n")

    stints_list:       list[pl.DataFrame]       = []
    rapm_list:         list[pl.DataFrame]       = []
    vs_team_list:      list[pl.DataFrame | None] = []
    player_stats_list: list[pl.DataFrame | None] = []
    valid_seasons:     list[int]                = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for season in target_seasons:
            stints_df, rapm_df, vs_team_df, stats_df = _load_season(
                data_dir, season, season_type=season_type
            )
            if stints_df.is_empty() or rapm_df.is_empty():
                print(f"  Season {season}: skipping (data missing).")
                continue
            stints_list.append(stints_df)
            rapm_list.append(rapm_df)
            vs_team_list.append(vs_team_df)
            player_stats_list.append(stats_df)
            valid_seasons.append(season)

    for w in caught:
        if issubclass(w.category, DataMissingWarning):
            print(f"  [WARN] {w.message}", file=sys.stderr)

    if not valid_seasons:
        print("ERROR: No valid seasons with complete data.", file=sys.stderr)
        sys.exit(1)

    # ── Train XGBoost matchup model on all available seasons ─────────────────
    print(f"\nTraining MatchupModel on seasons {valid_seasons}…")

    model = MatchupModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = model.fit(
            stints_list=stints_list,
            rapm_list=rapm_list,
            vs_team_list=vs_team_list,
            player_stats_list=player_stats_list,
            seasons=valid_seasons,
        )

    print(
        f"  Trained on {metrics['n_pairs']:,} opposition pairs  |  "
        f"RMSE: {metrics['rmse']:.4f}"
    )

    model_path = matchup_dir / "matchup_model.pkl"
    model.save(model_path)
    print(f"  Model saved → {model_path}")

    # ── Generate QoT/QoC and matchup predictions per season ──────────────────
    print(f"\nGenerating outputs for seasons {valid_seasons}…\n")

    saved_qot = saved_preds = 0

    for stints_df, rapm_df, vs_team_df, stats_df, season in zip(
        stints_list, rapm_list, vs_team_list, player_stats_list, valid_seasons
    ):
        print(f"  Season {season}:")

        # QoT / QoC
        qot_path = qot_qoc_dir / f"qot_qoc_{season}{out_suffix}.parquet"
        if qot_path.exists() and not args.force:
            print(f"    qot_qoc_{season}{out_suffix}.parquet already exists, skipping.")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                qot_qoc_df = compute_qot_qoc(stints_df, rapm_df, season)
            write_qot_qoc(qot_qoc_df, qot_qoc_dir, season, season_type=season_type)
            print(f"    Saved qot_qoc_{season}{out_suffix}.parquet  ({len(qot_qoc_df)} players)")
            _print_qot_qoc_leaders(qot_qoc_df, season)
            saved_qot += 1

        # Matchup predictions
        preds_path = matchup_preds_dir / f"matchup_preds_{season}{out_suffix}.parquet"
        if preds_path.exists() and not args.force:
            print(f"    matchup_preds_{season}{out_suffix}.parquet already exists, skipping.")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds_df = model.predict_season(
                    stints_df=stints_df,
                    rapm_df=rapm_df,
                    vs_team_df=vs_team_df,
                    player_stats_df=stats_df,
                    season=season,
                )
            write_matchup_preds(preds_df, matchup_preds_dir, season, season_type=season_type)
            print(f"    Saved matchup_preds_{season}{out_suffix}.parquet  ({len(preds_df):,} pairs)")
            saved_preds += 1

    print(
        f"\nDone.  "
        f"{saved_qot}/{len(valid_seasons)} QoT/QoC files saved, "
        f"{saved_preds}/{len(valid_seasons)} matchup prediction files saved."
    )


if __name__ == "__main__":
    main()
