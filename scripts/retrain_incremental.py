#!/usr/bin/env python3
"""Weekly warm-start incremental retraining of xG / Goalie models (Feature 2.17).

Adds new boosting rounds to existing XGBoost models using current-season shots,
with season-decayed sample weights (current=1.0, prior=0.7^gap). Accepts the
updated model only if the validation metric improves.

Requires:
    ~/.gretzky/data/shots/shots_{season}.parquet      (MoneyPuck shots)
    ~/.gretzky/data/xg/xg_model.pkl                  (existing xG model)
    ~/.gretzky/data/goalies/goalie_model.pkl          (existing goalie model)
    ~/.gretzky/data/goalies/goalies_{season}.parquet  (MoneyPuck goalie stats)

Usage::

    uv run python scripts/retrain_incremental.py
    uv run python scripts/retrain_incremental.py --type xg --seasons 2024 2025
    uv run python scripts/retrain_incremental.py --type goalie --force
    uv run python scripts/retrain_incremental.py --n-trees 100

Outputs:
    ~/.gretzky/data/xg/xg_model.pkl              (updated if accepted)
    ~/.gretzky/data/goalies/goalie_model.pkl      (updated if accepted)
    ~/.gretzky/data/retrain/retrain_log_xg.parquet
    ~/.gretzky/data/retrain/retrain_log_goalie.parquet
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

_DEFAULT_DATA_DIR = Path.home() / ".gretzky" / "data"


def _current_nhl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


def _default_seasons() -> list[int]:
    cur = _current_nhl_season()
    return list(range(2023, cur + 1))


def _load_shots(data_dir: Path, season: int) -> pl.DataFrame:
    from models.rapm_model import DataMissingWarning

    path = data_dir / "shots" / f"shots_{season}.parquet"
    if not path.exists():
        warnings.warn(
            f"Season {season}: shots not found at {path}.",
            DataMissingWarning, stacklevel=2,
        )
        return pl.DataFrame()
    return pl.read_parquet(path)


def _load_goalies(data_dir: Path, season: int) -> pl.DataFrame:
    from models.rapm_model import DataMissingWarning

    path = data_dir / "goalies" / f"goalies_{season}.parquet"
    if not path.exists():
        warnings.warn(
            f"Season {season}: goalie stats not found at {path}.",
            DataMissingWarning, stacklevel=2,
        )
        return pl.DataFrame()
    return pl.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Warm-start incremental retraining of xG / Goalie models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--type",
        choices=["xg", "goalie", "both"],
        default="both",
        dest="model_type",
        help="Which model(s) to retrain.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=_default_seasons(),
        metavar="YEAR",
        help="Seasons to include in training pool (oldest → newest).",
    )
    parser.add_argument(
        "--n-trees",
        type=int,
        default=50,
        metavar="N",
        help="Number of additional boosting rounds to add.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Accept updated model even if metric regresses.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        metavar="PATH",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    log_dir = data_dir / "retrain"
    log_dir.mkdir(parents=True, exist_ok=True)

    from models.incremental_retrainer import IncrementalRetrainer, write_retrain_log

    retrainer = IncrementalRetrainer(n_new_trees=args.n_trees)
    seasons   = sorted(args.seasons)
    cur       = seasons[-1]

    # ── xG retraining ─────────────────────────────────────────────────────
    # xG models are saved as ~/.gretzky/models/xg_model_{date}.pkl — find latest
    if args.model_type in ("xg", "both"):
        _models_dir = data_dir.parent / "models"
        _xg_candidates = sorted(_models_dir.glob("xg_model_*.pkl")) if _models_dir.exists() else []
        xg_model_path = _xg_candidates[-1] if _xg_candidates else (data_dir / "xg" / "xg_model.pkl")
        if not xg_model_path.exists():
            print(f"  [xG] Model not found at {xg_model_path}. Run train-xg first.", file=sys.stderr)
        else:
            from models.xg_model import XGModel

            print(f"\n  [xG] Loading existing model from {xg_model_path}")
            xg_model = XGModel.load(xg_model_path)

            shots_data: dict[int, pl.DataFrame] = {}
            for s in seasons:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    shots_data[s] = _load_shots(data_dir, s)
                for w in caught:
                    print(f"    [WARN] {w.message}", file=sys.stderr)

            n_total = sum(len(df) for df in shots_data.values())
            print(f"  [xG] Training pool: {len(seasons)} seasons, {n_total:,} shots")
            print(f"  [xG] Adding {args.n_trees} new trees (warm-start)...")

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                new_xg_model, result = retrainer.retrain_xg(
                    model          = xg_model,
                    shots_data     = shots_data,
                    current_season = cur,
                    n_new_trees    = args.n_trees,
                    force          = args.force,
                )
            for w in caught:
                print(f"    [WARN] {w.message}", file=sys.stderr)

            print(f"  [xG] {result}")
            if result.accepted:
                new_xg_model.save(xg_model_path)
                print(f"  [xG] Model updated and saved → {xg_model_path}")
            else:
                print(f"  [xG] Model NOT updated (metric did not improve; use --force to override)")

            log_df = retrainer.log_dataframe()
            log_path = write_retrain_log(
                log_df.filter(pl.col("model_type") == "xg"), log_dir, "xg"
            )
            print(f"  [xG] Retrain log → {log_path}")

    # ── Goalie retraining ──────────────────────────────────────────────────
    if args.model_type in ("goalie", "both"):
        goalie_model_path = data_dir / "goalie_ratings" / "goalie_model.pkl"
        if not goalie_model_path.exists():
            print(f"  [Goalie] Model not found at {goalie_model_path}. Run train-goalie first.", file=sys.stderr)
        else:
            from models.goalie_model import GoalieModel

            print(f"\n  [Goalie] Loading existing model from {goalie_model_path}")
            goalie_model = GoalieModel.load(goalie_model_path)

            goalie_data: dict[int, pl.DataFrame] = {}
            for s in seasons:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    goalie_data[s] = _load_goalies(data_dir, s)
                for w in caught:
                    print(f"    [WARN] {w.message}", file=sys.stderr)

            n_total = sum(len(df) for df in goalie_data.values())
            print(f"  [Goalie] Training pool: {len(seasons)} seasons, {n_total:,} goalie-games")
            print(f"  [Goalie] Adding {args.n_trees} new trees (warm-start)...")

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                new_goalie_model, result = retrainer.retrain_goalie(
                    model          = goalie_model,
                    goalie_data    = goalie_data,
                    current_season = cur,
                    n_new_trees    = args.n_trees,
                    force          = args.force,
                )
            for w in caught:
                print(f"    [WARN] {w.message}", file=sys.stderr)

            print(f"  [Goalie] {result}")
            if result.accepted:
                new_goalie_model.save(goalie_model_path)
                print(f"  [Goalie] Model updated and saved → {goalie_model_path}")
            else:
                print(f"  [Goalie] Model NOT updated (metric did not improve; use --force to override)")

            log_df = retrainer.log_dataframe()
            log_path = write_retrain_log(
                log_df.filter(pl.col("model_type") == "goalie"), log_dir, "goalie"
            )
            print(f"  [Goalie] Retrain log → {log_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
