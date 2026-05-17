"""compute_performance_anomaly — Feature 3.19 driver script.

Builds a per-(player, game) performance frame from the player_stats
parquet (xGF/60 as the primary metric — adjust ``--metric`` for others),
joins the latest injury-status snapshot for the IR mask, and writes
z-score + CUSUM SPC flags per game.

Usage::

    uv run python scripts/gretzky.py performance-anomaly
    uv run python scripts/gretzky.py performance-anomaly -- --date 2026-05-17
    uv run python scripts/gretzky.py performance-anomaly -- --metric ixg
    uv run python scripts/gretzky.py performance-anomaly -- --force
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

from data.data_store import DataStore
from data.schedule_sync import latest_schedule_parquet
from models.injury_status_integrator import STATUS_OUT
from models.performance_anomaly_detector import (
    PerformanceAnomalyDetector,
    write_performance_anomaly,
)


DEFAULT_DATA_DIR = Path(
    os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data"))
)
PERF_ANOMALY_SUBDIR  = "performance_anomaly"
SCHEDULE_SUBDIR      = "schedule"
INJURY_STATUS_SUBDIR = "injury_status"

DEFAULT_METRIC_COLUMN = "ixg"   # individual xG from player_stats


def _build_perf_df(store: DataStore, args, ir_set: set[int]) -> pl.DataFrame:
    sched_dir  = args.data_dir / SCHEDULE_SUBDIR
    sched_path = latest_schedule_parquet(sched_dir)
    if sched_path is None:
        print(f"[performance-anomaly] No schedule parquet in {sched_dir}.")
        sys.exit(1)
    schedule = pl.read_parquet(sched_path).select(["game_id", "game_date"])

    stats = store.player_stats()
    if len(stats) == 0:
        print("[performance-anomaly] DataStore has no player_stats. "
              "Run `ingest` first.")
        sys.exit(1)

    metric_col = args.metric
    if metric_col not in stats.columns:
        # Fall back: pick first numeric column not in identity set.
        identity = {"player_id", "game_id", "team_id", "season"}
        numeric = [
            c for c, dt in zip(stats.columns, stats.dtypes)
            if c not in identity and dt.is_numeric()
        ]
        if not numeric:
            print(f"[performance-anomaly] No usable numeric metric in player_stats.")
            sys.exit(1)
        metric_col = numeric[0]
        print(f"[performance-anomaly] Falling back to metric={metric_col}")

    df = (
        stats.select(["player_id", "game_id", pl.col(metric_col).alias("metric")])
             .filter(pl.col("player_id").is_not_null())
             .join(schedule, on="game_id", how="inner")
             .with_columns(
                 pl.col("player_id").is_in(list(ir_set)).alias("is_on_ir"),
             )
             .select(["player_id", "game_id", "game_date", "metric", "is_on_ir"])
    )
    return df


def _latest_ir_set(args, as_of: str) -> set[int]:
    inj_dir = args.data_dir / INJURY_STATUS_SUBDIR
    target  = inj_dir / f"injury_status_{as_of}.parquet"
    if target.exists():
        inj = pl.read_parquet(target)
    elif inj_dir.exists():
        candidates = sorted(inj_dir.glob("injury_status_*.parquet"))
        if not candidates:
            return set()
        inj = pl.read_parquet(candidates[-1])
    else:
        return set()
    if "status" not in inj.columns or len(inj) == 0:
        return set()
    return set(
        inj.filter(pl.col("status") == STATUS_OUT)["player_id"].to_list()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-(player, game) performance anomalies (Feature 3.19)."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None,
                        help="as-of date. Default: today (UTC).")
    parser.add_argument("--metric", type=str, default=DEFAULT_METRIC_COLUMN,
                        help=f"player_stats column to monitor "
                             f"(default {DEFAULT_METRIC_COLUMN}).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir  = args.data_dir / PERF_ANOMALY_SUBDIR
    out_path = out_dir / f"performance_anomaly_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[performance-anomaly] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[performance-anomaly] Computing perf anomalies as of {as_of}…")
    ir_set = _latest_ir_set(args, as_of)
    print(f"  {len(ir_set)} players currently on IR")

    with DataStore(args.data_dir) as store:
        perf = _build_perf_df(store, args, ir_set)
    print(f"  {len(perf):,} (player, game) rows from player_stats × schedule")

    result = PerformanceAnomalyDetector().compute(perf, as_of_date=as_of)
    n_rows = len(result)
    n_flag = int(result["is_anomaly"].sum() or 0)
    print(f"  {n_rows:,} rows produced  ({n_flag} flagged as anomalies)")

    if n_flag > 0:
        flagged = (
            result.filter(pl.col("is_anomaly"))
                  .sort("z_score")
                  .head(10)
        )
        print("\n  Top 10 flagged players (most negative z first):")
        print(f"  {'Player':<10}  {'Game':<10}  {'Date':<10}  "
              f"{'z':>5}  {'cusum':>6}  {'streak':>6}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*5}  {'─'*6}  {'─'*6}")
        for r in flagged.to_dicts():
            print(
                f"  {r['player_id']:<10}  {r['game_id']:<10}  "
                f"{r['game_date']:<10}  {r['z_score']:>5.2f}  "
                f"{r['cusum']:>6.3f}  {r['consecutive_below_n']:>6d}"
            )

    path = write_performance_anomaly(result, out_dir, as_of)
    print(f"\n[performance-anomaly] Written: {path}")


if __name__ == "__main__":
    main()
