"""Train the xG Player Finishing model — Feature 2.1.

Downloads MoneyPuck shot CSVs for requested seasons (if not already cached),
trains an XGBoost binary classifier, evaluates quality gates, and computes
per-player Finishing scores for the target season.

Quality gates (hard warnings, not hard failures):
    Brier score      < 0.10
    AUC-ROC          > 0.75
    MoneyPuck corr   >= 0.85

Outputs:
    ~/.gretzky/models/xg_model_{date}.pkl
    ~/.gretzky/data/xg_finishing/xg_finishing_{target_season}.parquet

Usage:
    uv run python scripts/train_xg_model.py
    uv run python scripts/train_xg_model.py --seasons 2020 2021 2022 2023 2024
    uv run python scripts/train_xg_model.py --force
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# ── repo root importable ──────────────────────────────────────────────────────
_REPO = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO))

import polars as pl

from data.moneypuck_client import MoneyPuckClient
from data.moneypuck_sync import MoneyPuckSync
from models.xg_model import XGModel
from models.player_finishing import PlayerFinishingAggregator, write_finishing


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _default_seasons() -> list[int]:
    cur = _current_nhl_season()
    return list(range(2023, cur + 1))


def _data_dir() -> Path:
    default = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
    return Path(os.environ.get("GRETZKY_DATA_DIR", str(default)))


def _models_dir() -> Path:
    default = Path.home() / ".gretzky" / "models"
    return Path(os.environ.get("GRETZKY_MODELS_DIR", str(default)))


def _current_nhl_season() -> int:
    """NHL season runs Oct–Sep. March 2026 → 2025."""
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _ensure_shots(
    data_dir: Path,
    seasons: list[int],
    force: bool = False,
) -> None:
    """Download MoneyPuck shot CSVs for any seasons not already cached."""
    shots_dir = data_dir / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    sync = MoneyPuckSync(shots_dir)

    if force:
        # Clear only error/empty entries; keep "ok" ones unless force
        for season in seasons:
            path = shots_dir / f"shots_{season}.parquet"
            if path.exists():
                path.unlink()
        manifest = sync.get_manifest()
        for season in seasons:
            manifest.get("seasons", {}).pop(str(season), None)
        (shots_dir / "manifest.json").write_text(
            __import__("json").dumps(manifest, indent=2)
        )

    seasons_to_download = []
    manifest = sync.get_manifest()
    for season in seasons:
        status = manifest.get("seasons", {}).get(str(season), {}).get("status")
        if status != "ok":
            seasons_to_download.append(season)

    if not seasons_to_download:
        print(f"  All seasons already cached: {seasons}")
        return

    print(f"  Downloading shots for seasons: {seasons_to_download}")
    print("  (This may take several minutes for large seasons — ~50–150 MB each)")
    with MoneyPuckClient() as client:
        for season in seasons_to_download:
            print(f"    → season {season}…", end=" ", flush=True)
            result = sync.sync_season(client, season)
            if result.status == "ok":
                print(f"{result.shots_count:,} shots  [{result.warning_count} warnings]")
            elif result.status == "skipped":
                print(f"skipped (already cached)")
            else:
                print(f"FAILED: {result.error_message}")


def _game_based_split(
    df: pl.DataFrame, train_frac: float = 0.70
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split by game_id (chronological order) to prevent data leakage.

    game_id encodes date in the NHL API format, so sorting preserves
    temporal ordering within a season.
    """
    game_ids = sorted(df["game_id"].drop_nulls().unique().to_list())
    split_idx = int(len(game_ids) * train_frac)
    train_ids = set(game_ids[:split_idx])
    train = df.filter(pl.col("game_id").is_in(list(train_ids)))
    eval_ = df.filter(~pl.col("game_id").is_in(list(train_ids)))
    return train, eval_


def _load_shots(data_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """Load shot parquets for given seasons, concatenating into one DataFrame."""
    shots_dir = data_dir / "shots"
    dfs = []
    for season in seasons:
        path = shots_dir / f"shots_{season}.parquet"
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping season {season}")
            continue
        df = pl.read_parquet(path)
        # Ensure nullable int columns are Int64 (older parquets may store them as Null)
        for _col in ("shooter_id", "goalie_id", "period", "time",
                     "home_skaters", "away_skaters"):
            if _col in df.columns:
                df = df.with_columns(pl.col(_col).cast(pl.Int64, strict=False))
        if not df.is_empty():
            # Regular season only — filter out playoff shots
            if "is_playoff" in df.columns:
                df = df.filter(pl.col("is_playoff") == False)  # noqa: E712
            dfs.append(df)
    if not dfs:
        raise RuntimeError("No shot data found. Run without --force or check download errors.")
    return pl.concat(dfs, how="diagonal")


def _load_icetime(data_dir: Path, season: int) -> pl.DataFrame | None:
    """Load EDGE skating parquet for icetime normalisation."""
    edge_dir = data_dir / "edge"
    path = edge_dir / f"edge_skating_{season}.parquet"
    if not path.exists():
        return None
    df = pl.read_parquet(path)
    # EDGE skating has player_id + icetime columns
    if "player_id" in df.columns and "icetime" in df.columns:
        return df.select(["player_id", "icetime"])
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train xG Player Finishing model (Feature 2.1)."
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=_default_seasons(),
        help="Seasons to download and train on. Defaults to the 6 most recent NHL seasons.",
    )
    parser.add_argument(
        "--target-season",
        type=int,
        default=None,
        help="Season for player finishing output (default: most recent in --seasons)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download shots and overwrite cached parquets",
    )
    args = parser.parse_args()

    target_season = args.target_season or max(args.seasons)

    data_dir   = _data_dir()
    models_dir = _models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print("=" * 60)
    print("  GRTZKY — xG Model Training (Feature 2.1)")
    print("=" * 60)
    print(f"  Seasons          : {args.seasons}")
    print(f"  Target season    : {target_season}")
    print(f"  Data dir         : {data_dir}")
    print(f"  Models dir       : {models_dir}")
    print()

    # ── 1. Ensure shot data is cached ────────────────────────────────────────
    print("[ 1/5 ] Ensuring shot data is available…")
    _ensure_shots(data_dir, args.seasons, force=args.force)
    print()

    # ── 2. Load all shots and split ───────────────────────────────────────────
    print(f"[ 2/5 ] Loading shots (seasons {args.seasons}) + splitting…")
    all_df = _load_shots(data_dir, args.seasons)
    train_seasons = [s for s in args.seasons if s != target_season]
    if train_seasons and "season" in all_df.columns:
        # Season-based split: train on all seasons except target, eval on target
        train_df = all_df.filter(pl.col("season") != target_season)
        eval_df  = all_df.filter(pl.col("season") == target_season)
        print(f"  Train: {len(train_df):,} shots  (seasons {train_seasons})")
        print(f"  Eval : {len(eval_df):,} shots  (season {target_season})")
    else:
        # Single season — fall back to game-based split (temporal, no leakage)
        train_df, eval_df = _game_based_split(all_df, train_frac=0.70)
        n_train_games = train_df["game_id"].n_unique() if "game_id" in train_df.columns else "?"
        n_eval_games  = eval_df["game_id"].n_unique()  if "game_id" in eval_df.columns  else "?"
        print(f"  Train: {len(train_df):,} shots  ({n_train_games} games, first 70%)")
        print(f"  Eval : {len(eval_df):,} shots  ({n_eval_games} games, last 30%)")
    print()

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("[ 3/5 ] Training XGBoost xG model…")
    model = XGModel()
    metrics = model.fit(train_df, eval_df=eval_df)

    print(f"  Brier score          : {metrics['brier_score']:.4f}  (target < 0.10)")
    print(f"  AUC-ROC              : {metrics['auc_roc']:.4f}  (target > 0.75)")
    print(f"  MoneyPuck corr       : {metrics['moneypuck_correlation']:.4f}  (target ≥ 0.85)")

    # Quality gate warnings
    if metrics["brier_score"] >= 0.10:
        print(f"  WARNING: Brier {metrics['brier_score']:.4f} ≥ 0.10 — model underperforming")
    if metrics["auc_roc"] < 0.75:
        print(f"  WARNING: AUC {metrics['auc_roc']:.4f} < 0.75 — model underperforming")
    if metrics["moneypuck_correlation"] < 0.85:
        print(f"  WARNING: MoneyPuck corr {metrics['moneypuck_correlation']:.4f} < 0.85")

    # Feature importance
    print()
    print("  Feature importances (gain):")
    for fname, score in list(model.feature_importance().items())[:8]:
        bar = "█" * min(int(score / 100), 20)
        print(f"    {fname:<22s}  {bar}")
    print()

    # Save model
    model_path = models_dir / f"xg_model_{today}.pkl"
    model.save(model_path)
    print(f"  Model saved → {model_path}")
    print()

    # ── 4. Compute player finishing ───────────────────────────────────────────
    print(f"[ 4/5 ] Computing player Finishing for season {target_season}…")
    # Reuse already-loaded data if target season is in the training set
    if target_season in args.seasons:
        target_df = all_df.filter(pl.col("season") == target_season) if "season" in all_df.columns else all_df
    else:
        target_df = _load_shots(data_dir, [target_season])
    icetime_df = _load_icetime(data_dir, target_season)
    if icetime_df is not None:
        print(f"  EDGE icetime loaded ({len(icetime_df)} players)")
    else:
        print("  No EDGE icetime found — finishing_per60 will be null")

    agg = PlayerFinishingAggregator()
    finishing_df = agg.compute(
        target_df,
        model,
        icetime_df=icetime_df,
        season=target_season,
        model_version=f"xg_v1_{today}",
    )

    out_path = write_finishing(finishing_df, data_dir / "xg_finishing", target_season)

    n_players = len(finishing_df)
    top5 = finishing_df.sort("finishing", descending=True).head(5)

    print(f"  Players computed: {n_players}")
    print(f"  Parquet saved  → {out_path}")
    print()
    print("  Top 5 Finishing (goals above expected):")
    for row in top5.to_dicts():
        f60 = f"{row['finishing_per60']:+.2f}/60" if row.get("finishing_per60") else ""
        print(
            f"    {row['shooter_name']:<22s}  {row['finishing']:+.2f}"
            f"  ({row['goals']}G, {row['xg_sum']:.1f} xG, {row['shots']}sh)"
            f"  {f60}"
        )

    print()
    print("=" * 60)
    print("  Done. Refresh /phase2 and search any player to see ratings.")
    print("=" * 60)


if __name__ == "__main__":
    main()
