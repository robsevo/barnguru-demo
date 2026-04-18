"""train_behavior_net — Feature 2.22 training script.

Loads player stats (puck battle output), positional heatmap output (2.21),
and skating baseline output (2.20), trains the hierarchical behavioral
network via imitation learning, and writes predictions for all fitted players.

Usage::

    uv run python scripts/train_behavior_net.py
    uv run python scripts/train_behavior_net.py --seasons 2023 2024 2025
    uv run python scripts/train_behavior_net.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import os

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from models.player_behavior_net import (
    PlayerBehaviorNet,
    write_behavior_predictions,
    ACTION_LABELS,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR      = Path(os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data")))
DEFAULT_SEASONS       = [2023, 2024, 2025]
BATTLE_SUBDIR         = "battles"
HEATMAP_SUBDIR        = "positional_heatmap"
SKATING_SUBDIR        = "skating_baseline"
BEHAVIOR_SUBDIR       = "behavior_net"
CV_OBS_SUBDIR         = "cv_live_observations"
CV_GATE_FILE          = "cv_gate.json"


def _latest_parquet(directory: Path, glob_pattern: str) -> Path | None:
    files = sorted(directory.glob(glob_pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _load_player_battles(battle_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """Load player-level puck battle output for given seasons."""
    frames: list[pl.DataFrame] = []
    for season in seasons:
        f = _latest_parquet(battle_dir, f"puck_battle_{season}*.parquet")
        if f:
            frames.append(pl.read_parquet(f))
            print(f"  Loaded player battles: {f.name}")
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal")


def _load_positional_heatmap(heatmap_dir: Path, season: int) -> pl.DataFrame | None:
    f = _latest_parquet(heatmap_dir, f"positional_heatmap_{season}*.parquet")
    if f:
        print(f"  Loaded positional heatmap: {f.name}")
        return pl.read_parquet(f)
    return None


def _load_skating_baseline(skating_dir: Path, season: int) -> pl.DataFrame | None:
    f = _latest_parquet(skating_dir, f"skating_baseline_{season}*.parquet")
    if f:
        print(f"  Loaded skating baseline: {f.name}")
        return pl.read_parquet(f)
    return None


def _read_cv_gate(data_dir: Path) -> bool:
    """Check whether the rob-only CV→player-NN gate is ON.

    File at ``$GRETZKY_DATA_DIR/cv_gate.json``; absence is treated as OFF.
    Lives next to the data dir (not the repo) so the web API and training
    scripts share one source of truth without import cycles.
    """
    import json
    p = data_dir / CV_GATE_FILE
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("enabled", False))
    except Exception:
        return False


def _load_cv_observations(data_dir: Path) -> pl.DataFrame | None:
    """Concatenate every per-game ``summary.parquet`` written by the aggregator.

    Returns None when there's nothing to feed. Consumers treat None as a
    gate-off / no-data signal.
    """
    obs_root = data_dir / CV_OBS_SUBDIR
    if not obs_root.exists():
        return None
    parquets = list(obs_root.glob("*/summary.parquet"))
    if not parquets:
        return None
    frames = [pl.read_parquet(p) for p in parquets]
    return pl.concat(frames, how="diagonal") if frames else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Per-Player Behavioral Neural Network (Feature 2.22)."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=DEFAULT_SEASONS,
        metavar="YEAR",
        help=f"Seasons to pool for player stats (default: {DEFAULT_SEASONS}).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Root data directory (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing behavior predictions parquet.",
    )
    args = parser.parse_args()

    seasons: list[int] = sorted(args.seasons)
    data_dir: Path = args.data_dir
    target_season = max(seasons)

    # ── Check output ──────────────────────────────────────────────────────
    behavior_dir = data_dir / BEHAVIOR_SUBDIR
    out_path = behavior_dir / f"behavior_predictions_{target_season}.parquet"
    if out_path.exists() and not args.force:
        print(f"[behavior-net] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    # ── Load player battle stats (primary input) ──────────────────────────
    battle_dir = data_dir / BATTLE_SUBDIR
    if not battle_dir.exists():
        print(f"[behavior-net] Puck battle directory not found: {battle_dir}")
        print("  Run 'uv run python scripts/gretzky.py puck-battles' first.")
        sys.exit(1)

    print(f"[behavior-net] Loading player battle stats for seasons: {seasons}")
    player_stats_df = _load_player_battles(battle_dir, seasons)

    if len(player_stats_df) == 0:
        print("[behavior-net] No player battle stats found.")
        print("  Run 'uv run python scripts/gretzky.py puck-battles' first.")
        sys.exit(1)
    print(f"  {len(player_stats_df):,} player-season rows loaded")

    # ── Load positional heatmap (optional) ───────────────────────────────
    print(f"[behavior-net] Loading positional heatmap for season {target_season}…")
    positional_df = _load_positional_heatmap(data_dir / HEATMAP_SUBDIR, target_season)
    if positional_df is None:
        print("  Positional heatmap not available — zone priors will be uniform.")
        print("  Run 'uv run python scripts/gretzky.py compute-heatmap' first for best results.")

    # ── Load skating baseline (optional) ─────────────────────────────────
    print(f"[behavior-net] Loading skating baseline for season {target_season}…")
    skating_df = _load_skating_baseline(data_dir / SKATING_SUBDIR, target_season)
    if skating_df is None:
        print("  Skating baseline not available — speed priors will use defaults.")

    # ── CV → player-NN feed gate (Feature 16.26) ─────────────────────────
    # Detector + per-arena training run always. Player-specific CV features
    # flow into this fit() call only when rob has flipped the gate ON via
    # the /dev toggle. Nightly/weekly non-CV inputs above are untouched.
    cv_df: pl.DataFrame | None = None
    if _read_cv_gate(data_dir):
        cv_df = _load_cv_observations(data_dir)
        if cv_df is None:
            print("[behavior-net] CV feed: ON, but no aggregated observations yet "
                  "(run 'gretzky aggregate-cv-obs' after users watch some games).")
        else:
            print(f"[behavior-net] CV feed: ON — {len(cv_df):,} aggregated CV rows will join training.")
    else:
        print("[behavior-net] CV feed: OFF (rob has not enabled it on /dev — "
              "flip 'CV → Player-NN Feed' when enough player data accrues).")

    # ── Train model ───────────────────────────────────────────────────────
    print(f"\n[behavior-net] Training behavioral network…")
    net = PlayerBehaviorNet()
    net.fit(
        player_stats_df     = player_stats_df,
        positional_df       = positional_df,
        skating_baseline_df = skating_df,
        cv_observations_df  = cv_df,
    )

    n_players = len(net.player_ids())
    print(f"  Base profiles built for {n_players:,} player(s).")
    print(f"  Context adjustment MLP trained (synthetic data + domain rules).")

    # ── Generate predictions (rested baseline) ────────────────────────────
    print(f"[behavior-net] Generating rested-baseline predictions for {target_season}…")
    predictions_df = net.predict_season(season=target_season)

    # ── Print action distribution summary ────────────────────────────────
    if len(predictions_df) > 0:
        print(f"\n  League-wide mean action distribution (FI=0 / rested baseline):")
        for action in ACTION_LABELS:
            mean_val = predictions_df[action].mean()
            print(f"    {action:<18}  {mean_val:.1%}")

    # ── Save ──────────────────────────────────────────────────────────────
    path = write_behavior_predictions(predictions_df, behavior_dir, target_season)
    print(f"\n[behavior-net] Predictions written: {path}")

    model_path = behavior_dir / "behavior_net.pkl"
    net.save(model_path)
    print(f"[behavior-net] Model saved: {model_path}")


if __name__ == "__main__":
    main()
