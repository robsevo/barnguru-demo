#!/usr/bin/env python3
"""Run Regime Change Detector over current-season data (Feature 2.14).

Feeds per-player game observations through the RegimeChangeDetector to flag
sustained statistical drifts. Outputs active alerts and optionally updates
the in-season Bayesian pipeline's sigma adjustments.

Requires:
    ~/.gretzky/data/inseason/inseason_blend_{season}.parquet  (blended posteriors)
    ~/.gretzky/data/inseason/inseason_pipeline_{season}.pkl   (pipeline state)

Usage::

    uv run python scripts/run_regime_detector.py
    uv run python scripts/run_regime_detector.py --seasons 2025
    uv run python scripts/run_regime_detector.py --force

Outputs:
    ~/.gretzky/data/regime/regime_alerts_{season}.parquet
    ~/.gretzky/data/regime/regime_detector_{season}.pkl
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Regime Change Detector over current-season data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=_default_seasons(),
        metavar="YEAR",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output already exists.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        metavar="PATH",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    output_dir = data_dir / "regime"
    output_dir.mkdir(parents=True, exist_ok=True)

    from models.regime_change_detector import RegimeChangeDetector, write_regime_alerts
    from models.rapm_model import DataMissingWarning

    seasons: list[int] = sorted(args.seasons)
    processed = 0

    for season in seasons:
        alerts_path   = output_dir / f"regime_alerts_{season}.parquet"
        detector_path = output_dir / f"regime_detector_{season}.pkl"

        if alerts_path.exists() and not args.force:
            print(f"  Season {season}: already exists, skipping (use --force)")
            continue

        print(f"\n  Season {season}:")

        # Load blended posteriors to get posterior_mean / posterior_sigma per player-game
        blend_path = data_dir / "inseason" / f"inseason_blend_{season}.parquet"
        if not blend_path.exists():
            warnings.warn(
                f"Season {season}: blended ratings not found at {blend_path}. "
                "Run: uv run python scripts/gretzky.py inseason-bayes",
                DataMissingWarning,
                stacklevel=2,
            )
            print(f"    No blended ratings; skipping season {season}")
            continue

        blend_df = pl.read_parquet(blend_path)
        print(f"    Loaded {len(blend_df)} player records from blended ratings")

        # Load per-game observations from EWMA games file (has xgf_per60 per game)
        ewma_games_path = data_dir / "ewma" / f"ewma_games_{season}.parquet"
        if not ewma_games_path.exists():
            warnings.warn(
                f"Season {season}: EWMA games file not found at {ewma_games_path}. "
                "Run: uv run python scripts/gretzky.py compute-ewma",
                DataMissingWarning,
                stacklevel=2,
            )
            print(f"    No EWMA game data; skipping season {season}")
            continue

        ewma_games = pl.read_parquet(ewma_games_path)
        print(f"    Loaded {len(ewma_games)} player-game rows")

        # Build detector and feed observations
        detector = RegimeChangeDetector()
        n_alerts = 0

        # Group by player and process in game order
        # Support both EWMA column naming conventions
        xgf_obs_col = "ewma_xgf60" if "ewma_xgf60" in ewma_games.columns else "xgf_per60"
        if "player_id" not in ewma_games.columns or xgf_obs_col not in ewma_games.columns:
            print(f"    EWMA games missing required columns; skipping season {season}")
            continue

        # Build lookup: player_id → (posterior_mean, posterior_sigma)
        posterior_lookup: dict[int, tuple[float, float]] = {}
        # Support both inseason blend column naming: mu_blend (model) or blended_mu (old)
        mu_col = "mu_blend" if "mu_blend" in blend_df.columns else "blended_mu"
        sigma_col_opt = "sigma_blend" if "sigma_blend" in blend_df.columns else (
            "blended_sigma" if "blended_sigma" in blend_df.columns else None
        )
        if "player_id" in blend_df.columns and mu_col in blend_df.columns:
            for row in blend_df.to_dicts():
                pid = int(row["player_id"])
                mu  = float(row.get(mu_col, 0.0) or 0.0)
                sig = float(row.get(sigma_col_opt, 0.5) or 0.5) if sigma_col_opt else 0.5
                posterior_lookup[pid] = (mu, sig)

        # Feed per-player-game observations
        for pid_df in ewma_games.sort(["player_id", "game_id"]).group_by("player_id"):
            player_id = int(pid_df[0][0])
            rows = pid_df[1].to_dicts()
            player_name = f"player_{player_id}"
            mu, sigma = posterior_lookup.get(player_id, (3.0, 0.5))

            for game_num, row in enumerate(rows, start=1):
                obs   = float(row.get(xgf_obs_col, mu) or mu)
                alert = detector.update(
                    player_id=player_id,
                    player_name=player_name,
                    observation=obs,
                    posterior_mean=mu,
                    posterior_sigma=sigma,
                    game_number=game_num,
                    season=season,
                )
                if alert:
                    n_alerts += 1

        active = detector.active_regimes()
        print(f"    Active regimes: {len(active)}  Total alerts triggered: {n_alerts}")

        if active:
            print("    Active regime players:")
            for state in sorted(active, key=lambda s: s.player_id):
                print(f"      pid={state.player_id:<8} state={state.regime_state:<12} "
                      f"games_processed={state.games_processed}")

        # Save alerts parquet
        alerts_df = detector.alerts_dataframe(season=season)
        write_regime_alerts(alerts_df, output_dir, season)
        print(f"    Alerts saved → {alerts_path}")

        # Save detector state
        detector.save(detector_path)
        print(f"    Detector saved → {detector_path}")

        processed += 1

    print(f"\nDone. {processed}/{len(seasons)} seasons processed.")


if __name__ == "__main__":
    main()
