#!/usr/bin/env python3
"""Train the Line Chemistry Model (Feature 2.4).

Requires (per season):
  - Shift data:       ~/.gretzky/data/raw/shifts_{season}.parquet
  - MoneyPuck shots:  ~/.gretzky/data/shots/shots_{season}.parquet
  - RAPM parquets:    ~/.gretzky/data/rapm/rapm_{season}.parquet
                      (run: uv run python scripts/gretzky.py train-rapm first)
  - finishing:        ~/.gretzky/data/xg_finishing/xg_finishing_{season}.parquet (optional)
  - player stats:     ~/.gretzky/data/raw/player_stats_{season}.parquet (optional)

Usage::

    uv run python scripts/train_chemistry_model.py
    uv run python scripts/train_chemistry_model.py --seasons 2023 2024 2025
    uv run python scripts/train_chemistry_model.py --seasons 2025 --force

Outputs:
    ~/.gretzky/data/chemistry/chemistry_model.pkl
    ~/.gretzky/data/chemistry/pair_chemistry_{season}.parquet
    ~/.gretzky/data/chemistry/player_chemistry_{season}.parquet
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

    season_type filters shifts (game_id type), shots (is_playoff), and picks
    matching RAPM + xG finishing parquets.

    Returns:
        (stints_df, rapm_df, finishing_df, player_stats_df)
        stints_df and rapm_df are empty DataFrames if required files are missing.
    """
    from models.rapm_model import DataMissingWarning, build_stints

    suffix = "_playoffs" if season_type == "playoffs" else ""
    shifts_path       = data_dir / "raw"          / f"shifts_{season}.parquet"
    shots_path        = data_dir / "shots"         / f"shots_{season}.parquet"
    rapm_path         = data_dir / "rapm"          / f"rapm_{season}{suffix}.parquet"
    if season_type == "playoffs" and not rapm_path.exists():
        rapm_path = data_dir / "rapm" / f"rapm_{season}.parquet"
    finishing_path    = data_dir / "xg_finishing"  / f"xg_finishing_{season}{suffix}.parquet"
    if season_type == "playoffs" and not finishing_path.exists():
        finishing_path = data_dir / "xg_finishing" / f"xg_finishing_{season}.parquet"
    player_stats_path = data_dir / "raw"           / f"player_stats_{season}.parquet"

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

    # Filter by game type so build_stints sees only the playoff/regular cohort.
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

    finishing_df = None
    if finishing_path.exists():
        finishing_df = pl.read_parquet(finishing_path)
        print(f", finishing", end="", flush=True)

    player_stats_df = None
    if player_stats_path.exists():
        player_stats_df = pl.read_parquet(player_stats_path)
        print(f", player_stats", end="", flush=True)

    print(f"  →  {len(stints_df):,} stints  |  {len(rapm_df)} RAPM players")
    return stints_df, rapm_df, finishing_df, player_stats_df


def _print_pair_chemistry_leaders(pair_df: pl.DataFrame, season: int) -> None:
    """Print top 10 pair chemistry leaders (minimum 30 min EV shared TOI)."""
    min_toi = 1800.0  # 30 min EV shared TOI

    print(f"\n  ── Season {season} Top Pair Chemistry (chemistry_delta) ──")
    top_pairs = (
        pair_df
        .filter(pl.col("co_toi_ev") >= min_toi)
        .sort("chemistry_delta", descending=True)
        .head(10)
    )
    for r in top_pairs.to_dicts():
        toi_min = r["co_toi_ev"] / 60.0
        hist_str = f"{r['hist_xgf_pct']:.3f}" if r["hist_xgf_pct"] is not None else " N/A "
        print(
            f"    {r['player_a_id']} + {r['player_b_id']}  "
            f"Δ: {r['chemistry_delta']:+.4f}  "
            f"xGF%: {r['model_xgf_pct']:.3f}  "
            f"hist: {hist_str}  "
            f"EV-TOI: {toi_min:.0f}m"
        )


def _print_player_chemistry_leaders(player_df: pl.DataFrame, season: int) -> None:
    """Print top 10 player chemistry scores."""
    print(f"\n  ── Season {season} Top Player Chemistry Scores ──")
    top_players = (
        player_df
        .sort("chemistry_score", descending=True)
        .head(10)
    )
    for r in top_players.to_dicts():
        print(
            f"    {r['player_name']:<22} {r['team']}  "
            f"Score: {r['chemistry_score']:+.4f}  "
            f"Pairs: {r['chemistry_pairs']}  "
            f"EV-TOI: {r['toi_ev']/60:.0f}m"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Feature 2.4 — Line Chemistry Model (linemate pair xGF%).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=_default_seasons(),
        metavar="YEAR",
        help="Season years to fit and predict (e.g. 2023 2024 2025). "
             "Defaults to the seasons from 2023 through the current NHL season.",
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
        help="'regular' writes pair/player_chemistry_{year}.parquet (default). "
             "'playoffs' filters shifts+shots to playoff games and writes "
             "*_chemistry_{year}_playoffs.parquet.",
    )
    args = parser.parse_args()

    data_dir:      Path = args.data_dir
    target_seasons      = sorted(args.seasons)
    season_type:   str  = args.season_type
    out_suffix          = "_playoffs" if season_type == "playoffs" else ""
    chemistry_dir       = data_dir / "chemistry"

    chemistry_dir.mkdir(parents=True, exist_ok=True)

    # Check if outputs already exist
    if not args.force:
        existing = [
            s for s in target_seasons
            if (chemistry_dir / f"pair_chemistry_{s}{out_suffix}.parquet").exists()
            and (chemistry_dir / f"player_chemistry_{s}{out_suffix}.parquet").exists()
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

    from models.line_chemistry_model import (
        LineChemistryModel,
        compute_player_chemistry,
        write_pair_chemistry,
        write_player_chemistry,
    )
    from models.rapm_model import DataMissingWarning

    # ── Load all seasons ─────────────────────────────────────────────────────
    print(f"\nLoading data for seasons {target_seasons}…\n")

    stints_list:       list[pl.DataFrame]        = []
    rapm_list:         list[pl.DataFrame]        = []
    finishing_list:    list[pl.DataFrame | None] = []
    player_stats_list: list[pl.DataFrame | None] = []
    valid_seasons:     list[int]                 = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for season in target_seasons:
            stints_df, rapm_df, finishing_df, stats_df = _load_season(
                data_dir, season, season_type=season_type
            )
            if stints_df.is_empty() or rapm_df.is_empty():
                print(f"  Season {season}: skipping (data missing).")
                continue
            stints_list.append(stints_df)
            rapm_list.append(rapm_df)
            finishing_list.append(finishing_df)
            player_stats_list.append(stats_df)
            valid_seasons.append(season)

    for w in caught:
        if issubclass(w.category, DataMissingWarning):
            print(f"  [WARN] {w.message}", file=sys.stderr)

    if not valid_seasons:
        print("ERROR: No valid seasons with complete data.", file=sys.stderr)
        # Playoffs variant is a no-op when playoff stints/RAPM haven't been
        # populated yet (pre-playoff season). Don't fail the pipeline.
        if season_type == "playoffs":
            sys.exit(0)
        sys.exit(1)

    # ── Train XGBoost chemistry model on all available seasons ────────────────
    print(f"\nTraining LineChemistryModel on seasons {valid_seasons}…")

    model = LineChemistryModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = model.fit(
            stints_list=stints_list,
            rapm_list=rapm_list,
            finishing_list=finishing_list,
            player_stats_list=player_stats_list,
            seasons=valid_seasons,
        )

    print(
        f"  Trained on {metrics['n_pairs']:,} linemate pairs  |  "
        f"RMSE: {metrics['rmse']:.4f}"
    )

    model_path = chemistry_dir / "chemistry_model.pkl"
    model.save(model_path)
    print(f"  Model saved → {model_path}")

    # ── Feature importance ────────────────────────────────────────────────────
    fi = model.feature_importance()
    if fi:
        print("\n  Feature importances (gain):")
        for feat, score in list(fi.items())[:5]:
            print(f"    {feat:<25}  {score:.1f}")

    # ── Generate pair and player chemistry per season ─────────────────────────
    print(f"\nGenerating outputs for seasons {valid_seasons}…\n")

    saved_pairs = saved_players = 0

    for stints_df, rapm_df, finishing_df, stats_df, season in zip(
        stints_list, rapm_list, finishing_list, player_stats_list, valid_seasons
    ):
        print(f"  Season {season}:")

        # Pair chemistry
        pair_path = chemistry_dir / f"pair_chemistry_{season}{out_suffix}.parquet"
        if pair_path.exists() and not args.force:
            print(f"    pair_chemistry_{season}{out_suffix}.parquet already exists, skipping.")
            pair_df = None
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pair_df = model.predict_season(
                    stints_df=stints_df,
                    rapm_df=rapm_df,
                    finishing_df=finishing_df,
                    player_stats_df=stats_df,
                    season=season,
                )
            write_pair_chemistry(pair_df, chemistry_dir, season, season_type=season_type)
            print(f"    Saved pair_chemistry_{season}{out_suffix}.parquet  ({len(pair_df):,} pairs)")
            _print_pair_chemistry_leaders(pair_df, season)
            saved_pairs += 1

        # Player chemistry
        player_path = chemistry_dir / f"player_chemistry_{season}{out_suffix}.parquet"
        if player_path.exists() and not args.force:
            print(f"    player_chemistry_{season}{out_suffix}.parquet already exists, skipping.")
        else:
            if pair_df is None:
                pair_df = model.predict_season(
                    stints_df=stints_df,
                    rapm_df=rapm_df,
                    finishing_df=finishing_df,
                    player_stats_df=stats_df,
                    season=season,
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                player_df = compute_player_chemistry(pair_df, rapm_df, season)
            write_player_chemistry(player_df, chemistry_dir, season, season_type=season_type)
            print(f"    Saved player_chemistry_{season}.parquet  ({len(player_df)} players)")
            _print_player_chemistry_leaders(player_df, season)
            saved_players += 1

    print(
        f"\nDone.  "
        f"{saved_pairs}/{len(valid_seasons)} pair chemistry files saved, "
        f"{saved_players}/{len(valid_seasons)} player chemistry files saved."
    )


if __name__ == "__main__":
    main()
