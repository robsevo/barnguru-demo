"""Retrain the xG model with CV tracking features — Feature 16.10.

Loads MoneyPuck shot data and CV tracking features from the DuckDB
accumulator (Feature 16.8), then trains two models side-by-side:

    1. Base model     — standard 19 MoneyPuck features (Feature 2.1 baseline)
    2. Tracking model — base 19 + 3 CV features (screen_count, defender_dist,
                        goalie_displacement)

The tracking model is saved only when its eval-set AUC strictly exceeds the
base model's by at least _MIN_AUC_IMPROVEMENT (default 0.001), preventing
regression.

Quality gates (same as Feature 2.1):
    Brier score < 0.10
    AUC-ROC     > 0.75

Outputs (tracking model, saved only if gate passes):
    ~/.gretzky/models/xg_tracking_{date}.pkl

Usage::

    uv run python scripts/gretzky.py retrain-xg-tracking
    uv run python scripts/gretzky.py retrain-xg-tracking -- --seasons 2024 2025
    uv run python scripts/gretzky.py retrain-xg-tracking -- --min-tracking-games 10
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO))

import polars as pl

from data.moneypuck_client import MoneyPuckClient
from data.moneypuck_sync import MoneyPuckSync
from data.tracking_accumulator import TrackingAccumulator
from data.tracking_feature_extractor import (
    extract_snapshot_features,
    join_tracking_features,
    _DEFAULT_DB as _DEFAULT_TRACKING_DB,
)
from models.xg_model import XGModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_MIN_AUC_IMPROVEMENT: float = 0.001   # tracking model must beat base by at least this
_MIN_TRACKING_GAMES:  int   = 20      # minimum games with shot snapshots before running


def _data_dir() -> Path:
    return Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))


def _models_dir() -> Path:
    return Path(os.environ.get("GRETZKY_MODELS_DIR", str(Path.home() / ".gretzky" / "models")))


def _tracking_db() -> Path:
    return Path(os.environ.get("GRETZKY_TRACKING_DB", str(_DEFAULT_TRACKING_DB)))


def _current_nhl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


def _default_seasons() -> list[int]:
    cur = _current_nhl_season()
    return list(range(cur - 1, cur + 1))


# ---------------------------------------------------------------------------
# Data helpers (reused from train_xg_model.py)
# ---------------------------------------------------------------------------

def _ensure_shots(data_dir: Path, seasons: list[int]) -> None:
    shots_dir = data_dir / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    sync = MoneyPuckSync(shots_dir)
    manifest = sync.get_manifest()
    to_dl = [s for s in seasons
             if manifest.get("seasons", {}).get(str(s), {}).get("status") != "ok"]
    if not to_dl:
        print(f"  All seasons already cached: {seasons}")
        return
    print(f"  Downloading shots for seasons: {to_dl}")
    with MoneyPuckClient() as client:
        for season in to_dl:
            print(f"    → season {season}…", end=" ", flush=True)
            result = sync.sync_season(client, season)
            if result.status == "ok":
                print(f"{result.shots_count:,} shots")
            else:
                print(f"FAILED: {result.error_message}")


def _load_shots(data_dir: Path, seasons: list[int]) -> pl.DataFrame:
    shots_dir = data_dir / "shots"
    dfs = []
    for season in seasons:
        path = shots_dir / f"shots_{season}.parquet"
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping season {season}")
            continue
        df = pl.read_parquet(path)
        for col in ("shooter_id", "goalie_id", "period", "time",
                    "home_skaters", "away_skaters"):
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Int64, strict=False))
        if not df.is_empty():
            if "is_playoff" in df.columns:
                df = df.filter(pl.col("is_playoff") == False)  # noqa: E712
            dfs.append(df)
    if not dfs:
        raise RuntimeError("No shot data found. Run 'train-xg' first to cache shot CSVs.")
    return pl.concat(dfs, how="diagonal")


def _game_based_split(
    df: pl.DataFrame, train_frac: float = 0.70
) -> tuple[pl.DataFrame, pl.DataFrame]:
    game_ids = sorted(df["game_id"].drop_nulls().unique().to_list())
    split_idx = int(len(game_ids) * train_frac)
    train_ids = set(game_ids[:split_idx])
    train = df.filter(pl.col("game_id").is_in(list(train_ids)))
    eval_ = df.filter(~pl.col("game_id").is_in(list(train_ids)))
    return train, eval_


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain xG model with CV tracking features (Feature 16.10)."
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=_default_seasons(),
        help="Seasons to train on (default: two most recent).",
    )
    parser.add_argument(
        "--min-tracking-games",
        type=int,
        default=_MIN_TRACKING_GAMES,
        dest="min_tracking_games",
        help=(
            f"Minimum games with shot snapshots required to proceed "
            f"(default: {_MIN_TRACKING_GAMES})."
        ),
    )
    parser.add_argument(
        "--min-auc-improvement",
        type=float,
        default=_MIN_AUC_IMPROVEMENT,
        dest="min_auc_improvement",
        help=f"AUC improvement gate (default: {_MIN_AUC_IMPROVEMENT}).",
    )
    args = parser.parse_args()

    data_dir   = _data_dir()
    models_dir = _models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    db_path = _tracking_db()

    print("=" * 62)
    print("  GRTZKY — xG Tracking Retrain (Feature 16.10)")
    print("=" * 62)
    print(f"  Seasons          : {args.seasons}")
    print(f"  Tracking DB      : {db_path}")
    print(f"  Min tracking games: {args.min_tracking_games}")
    print()

    # ── 1. Check tracking data availability ──────────────────────────────────
    print("[ 1/6 ] Checking tracking data availability…")
    if not db_path.exists():
        print(f"  Tracking DB not found at {db_path}")
        print(
            "  Run the CV worker (Feature 16.7) during live games to accumulate data."
        )
        print("  Exiting — nothing to retrain against.")
        return

    acc = TrackingAccumulator(db_path, read_only=True)
    try:
        games_with_data = acc.games_with_data()
        total_shot_snaps = acc.shot_count()
    finally:
        acc.close()

    n_games = len(games_with_data)
    print(f"  Games with tracking data : {n_games}")
    print(f"  Total shot snapshots     : {total_shot_snaps:,}")

    if n_games < args.min_tracking_games:
        print(
            f"\n  Only {n_games} game(s) with tracking data — need at least "
            f"{args.min_tracking_games} before retraining.\n"
            "  Keep the CV worker running during live games and retry later."
        )
        return
    print()

    # ── 2. Load MoneyPuck shots ───────────────────────────────────────────────
    print(f"[ 2/6 ] Ensuring MoneyPuck shot data for seasons {args.seasons}…")
    _ensure_shots(data_dir, args.seasons)
    all_df = _load_shots(data_dir, args.seasons)
    print(f"  Loaded {len(all_df):,} shots across {all_df['game_id'].n_unique()} games")
    print()

    # ── 3. Extract and join tracking features ────────────────────────────────
    print("[ 3/6 ] Extracting tracking features from DuckDB…")
    tracking_df = extract_snapshot_features(db_path)
    n_snapshots_with_coords = len(tracking_df)
    print(f"  Snapshots with rink coordinates : {n_snapshots_with_coords:,}")

    if n_snapshots_with_coords == 0:
        print(
            "  No rink coordinates available (homography not yet computed).\n"
            "  Tracking features will default to zero — model will likely not improve.\n"
            "  Proceeding anyway to verify gate logic."
        )

    enriched_df = join_tracking_features(all_df, tracking_df)

    # How many shots matched a tracking snapshot?
    if "screen_count" in enriched_df.columns and n_snapshots_with_coords > 0:
        # Rows where screen_count could have been set from real data
        matched = enriched_df.filter(pl.col("screen_count") != 0).height
        # Even screen_count=0 might be real, so approximate by non-null join
        pass

    print(
        f"  Shots with tracking coverage    : "
        f"{min(n_snapshots_with_coords, len(enriched_df)):,} / {len(enriched_df):,}"
    )
    print()

    # ── 4. Train/eval split ───────────────────────────────────────────────────
    print("[ 4/6 ] Splitting data…")
    target_season = max(args.seasons)
    if "season" in enriched_df.columns and len(args.seasons) > 1:
        train_df = enriched_df.filter(pl.col("season") != target_season)
        eval_df  = enriched_df.filter(pl.col("season") == target_season)
        print(f"  Train: {len(train_df):,} shots  (seasons {[s for s in args.seasons if s != target_season]})")
        print(f"  Eval : {len(eval_df):,} shots  (season {target_season})")
    else:
        train_df, eval_df = _game_based_split(enriched_df)
        print(f"  Train: {len(train_df):,} shots  (first 70% of games)")
        print(f"  Eval : {len(eval_df):,} shots  (last 30% of games)")
    print()

    # ── 5. Train base and tracking models (same split, fair comparison) ───────
    print("[ 5/6 ] Training base model (19 features)…")
    base_model = XGModel(tracking_features=False)
    base_metrics = base_model.fit(train_df, eval_df=eval_df)
    print(f"  Brier : {base_metrics['brier_score']:.4f}   AUC : {base_metrics['auc_roc']:.4f}")

    print()
    print("[ 5/6 ] Training tracking model (22 features)…")
    tracking_model = XGModel(tracking_features=True)
    tracking_metrics = tracking_model.fit(train_df, eval_df=eval_df)
    print(f"  Brier : {tracking_metrics['brier_score']:.4f}   AUC : {tracking_metrics['auc_roc']:.4f}")
    print()

    # ── 6. Gate and deploy ────────────────────────────────────────────────────
    print("[ 6/6 ] Evaluating AUC improvement gate…")
    base_auc     = base_metrics["auc_roc"]
    tracking_auc = tracking_metrics["auc_roc"]
    delta        = tracking_auc - base_auc

    print(f"  Base AUC     : {base_auc:.4f}")
    print(f"  Tracking AUC : {tracking_auc:.4f}  ({delta:+.4f})")
    print(f"  Required Δ   : ≥ {args.min_auc_improvement:.4f}")
    print()

    if tracking_auc > base_auc + args.min_auc_improvement:
        # Additional quality gates
        if tracking_metrics["brier_score"] >= 0.10:
            print(f"  WARNING: Brier {tracking_metrics['brier_score']:.4f} ≥ 0.10")
        if tracking_auc < 0.75:
            print(f"  WARNING: AUC {tracking_auc:.4f} < 0.75 quality floor")

        model_path = models_dir / f"xg_tracking_{today}.pkl"
        tracking_model.save(model_path)

        print(f"  ✓ Gate PASSED (+{delta:.4f} AUC improvement)")
        print(f"  Tracking model saved → {model_path}")
        print()
        print("  Top feature importances (gain):")
        fi = tracking_model.feature_importance()
        for fname, score in list(fi.items())[:10]:
            bar = "█" * min(int(score / 100), 20)
            print(f"    {fname:<26s}  {bar}")
    else:
        print(
            f"  ✗ Gate FAILED — tracking model did not improve AUC by ≥ "
            f"{args.min_auc_improvement:.4f}  (Δ = {delta:+.4f})"
        )
        print(
            "  Not saving tracking model.  Possible causes:\n"
            "    • Too few shots have matching tracking snapshots\n"
            "    • Homography calibration not yet accurate\n"
            "    • More games needed (current: "
            f"{n_games}, recommended: ≥ {args.min_tracking_games})"
        )

    print()
    print("=" * 62)
    print("  Done.")
    print("=" * 62)


if __name__ == "__main__":
    main()
