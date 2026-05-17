"""compute_fi_edge_degradation — Feature 3.21 driver script.

Reads the latest composite-FI parquet (3.17) and predicts the per-EDGE
metric delta (vs. each player's skating baseline, 2.20) for each
(player, game) row. Outputs feed the per-player behavioral net (2.22).

The model is trained on paired ``(FI, observed_EDGE_delta)`` rows when
available — supply a training parquet via ``--training PATH`` with
columns ``fi, delta_speed_vs_baseline, delta_distance_vs_baseline,
delta_carry_vs_baseline, delta_burst_vs_baseline``. Without training
data the model falls back to literature-style default coefficients
(documented in ``models/fi_edge_degradation.py``).

Usage::

    uv run python scripts/gretzky.py fi-edge-degradation
    uv run python scripts/gretzky.py fi-edge-degradation -- --date 2026-05-17
    uv run python scripts/gretzky.py fi-edge-degradation -- --training data/fi_edge_train.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from models.fi_edge_degradation import (
    FIEdgeDegradationModel,
    write_fi_edge_degradation,
)


DEFAULT_DATA_DIR = Path(
    os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data"))
)
FI_SUBDIR        = "composite_fi"
OUT_SUBDIR       = "fi_edge_degradation"


def _latest_fi_parquet(fi_dir: Path) -> Path | None:
    if not fi_dir.exists():
        return None
    parquets = sorted(fi_dir.glob("composite_fi_*.parquet"))
    return parquets[-1] if parquets else None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute per-(player, game) EDGE-metric degradation "
                    "predictions from FI (Feature 3.21)."
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--date", type=str, default=None,
                   help="as-of date. Default: today (UTC).")
    p.add_argument("--training", type=Path, default=None,
                   help="Optional training parquet with paired (FI, deltas).")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir  = args.data_dir / OUT_SUBDIR
    out_path = out_dir / f"fi_edge_degradation_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[fi-edge-degradation] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    fi_dir  = args.data_dir / FI_SUBDIR
    fi_path = fi_dir / f"composite_fi_{as_of}.parquet"
    if not fi_path.exists():
        latest = _latest_fi_parquet(fi_dir)
        if latest is None:
            print(f"[fi-edge-degradation] No composite-FI parquet in {fi_dir}.")
            print("  Run `gretzky composite-fi` first.")
            sys.exit(1)
        fi_path = latest
    print(f"[fi-edge-degradation] Reading FI from {fi_path}")
    fi_df = pl.read_parquet(fi_path)
    # composite_fi schema uses ``fatigue_index``; rename for the model.
    if "fatigue_index" in fi_df.columns and "fi" not in fi_df.columns:
        fi_df = fi_df.rename({"fatigue_index": "fi"})
    cols = ["player_id", "game_id", "game_date", "fi"]
    fi_df = fi_df.select([c for c in cols if c in fi_df.columns])
    print(f"  {len(fi_df):,} (player, game) FI rows")

    model = FIEdgeDegradationModel()
    if args.training is not None:
        if args.training.exists():
            train_df = pl.read_parquet(args.training)
            print(f"[fi-edge-degradation] Fitting on {len(train_df):,} rows "
                  f"from {args.training}")
            model.fit(train_df)
            print(f"  fitted: {model.is_fitted}  n_train: {model.n_train}")
        else:
            print(f"[fi-edge-degradation] Training file not found: "
                  f"{args.training}  (using defaults)")

    print("\n  Per-metric coefficients (α, β):")
    for metric, (alpha, beta) in model.coeffs.items():
        print(f"    {metric:<28s}  α={alpha:+.4f}  β={beta:+.4f}")

    result = model.predict(fi_df, as_of_date=as_of)
    n = len(result)
    print(f"\n  {n:,} prediction rows produced")

    if n > 0:
        print("\n  Worst-10 load factors (most degraded):")
        worst = result.sort("predicted_load_factor").head(10)
        print(f"  {'pid':>10}  {'date':<10}  {'fi':>5}  "
              f"{'speed':>7}  {'dist':>7}  {'carry':>7}  {'burst':>7}  "
              f"{'load':>7}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*5}  "
              f"{'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
        for r in worst.to_dicts():
            print(
                f"  {r['player_id']:>10}  {r['game_date']:<10}  "
                f"{r['fi']:>5.2f}  "
                f"{r['speed_vs_baseline']:>+7.3f}  "
                f"{r['distance_vs_baseline']:>+7.3f}  "
                f"{r['carry_vs_baseline']:>+7.3f}  "
                f"{r['burst_vs_baseline']:>+7.3f}  "
                f"{r['predicted_load_factor']:>+7.3f}"
            )

    path = write_fi_edge_degradation(result, out_dir, as_of)
    print(f"\n[fi-edge-degradation] Written: {path}")

    model_path = out_dir / "fi_edge_degradation_model.pkl"
    model.save(model_path)
    print(f"[fi-edge-degradation] Model saved: {model_path}")


if __name__ == "__main__":
    main()
