"""compute_performance_anomaly — Feature 3.19 driver script.

Builds a per-(player, game) performance frame by aggregating scoring
and shot events out of PBP (goal + assist + shot-on-goal weighted into
a single game-score), joins the latest injury-status snapshot for the
IR mask, and writes z-score + CUSUM SPC flags per game.

Usage::

    uv run python scripts/gretzky.py performance-anomaly
    uv run python scripts/gretzky.py performance-anomaly -- --date 2026-05-17
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

DEFAULT_METRIC = "game_score"

# How event-types map to a single-game game-score metric. Goals and shots
# are the primary signal; primary/secondary assists each count partial.
_GAME_SCORE_WEIGHTS = {
    "goal":          1.0,   # finishing
    "primary_assist": 0.7,
    "secondary_assist": 0.4,
    "shot_on_goal":  0.075,   # raw attempt rate
}


def _build_perf_df(store: DataStore, args, ir_set: set[int]) -> pl.DataFrame:
    """Per-(player, game) game-score metric, built directly from PBP.

    The season-aggregate ``player_stats`` parquet has no ``game_id`` —
    it's one row per (player, season). To get a per-game rolling
    baseline we aggregate scoring events out of PBP ourselves:

        goals       — one row per ``scorer_id`` × game_id
        assists     — one row per ``assist1_id`` / ``assist2_id`` × game_id
        shots       — one row per ``shooter_id`` × game_id (event_type=shot_on_goal)

    These get weighted into a per-game game-score metric. Players with
    zero events on a given game contribute a zero-row that the rolling
    SPC baseline counts as a "did-not-impact" game. The metric never
    goes negative — anomalies are detected as drops vs. the player's
    own rolling 20-game mean, not against zero.
    """
    sched_dir  = args.data_dir / SCHEDULE_SUBDIR
    sched_path = latest_schedule_parquet(sched_dir)
    if sched_path is None:
        print(f"[performance-anomaly] No schedule parquet in {sched_dir}.")
        sys.exit(1)
    schedule = pl.read_parquet(sched_path).select(["game_id", "game_date"])

    pbp = store.pbp()
    if len(pbp) == 0:
        print("[performance-anomaly] DataStore has no PBP. Run `ingest` first.")
        sys.exit(1)

    frames: list[pl.DataFrame] = []

    # Goals — one row per (scorer, game_id).
    if "scorer_id" in pbp.columns:
        frames.append(
            pbp.filter(
                (pl.col("event_type") == "goal")
                & pl.col("scorer_id").is_not_null()
            )
            .group_by(["scorer_id", "game_id"])
            .agg(pl.len().alias("count"))
            .rename({"scorer_id": "player_id"})
            .with_columns(
                (pl.col("count") * _GAME_SCORE_WEIGHTS["goal"]).alias("metric")
            )
            .select(["player_id", "game_id", "metric"])
        )

    # Primary assists — assist1_id (note: these rows are duplicated under
    # event_type=goal in NHL PBP, so we filter on goal events only).
    if "assist1_id" in pbp.columns:
        frames.append(
            pbp.filter(
                (pl.col("event_type") == "goal")
                & pl.col("assist1_id").is_not_null()
            )
            .group_by(["assist1_id", "game_id"])
            .agg(pl.len().alias("count"))
            .rename({"assist1_id": "player_id"})
            .with_columns(
                (pl.col("count") * _GAME_SCORE_WEIGHTS["primary_assist"])
                .alias("metric")
            )
            .select(["player_id", "game_id", "metric"])
        )

    # Secondary assists — assist2_id.
    if "assist2_id" in pbp.columns:
        frames.append(
            pbp.filter(
                (pl.col("event_type") == "goal")
                & pl.col("assist2_id").is_not_null()
            )
            .group_by(["assist2_id", "game_id"])
            .agg(pl.len().alias("count"))
            .rename({"assist2_id": "player_id"})
            .with_columns(
                (pl.col("count") * _GAME_SCORE_WEIGHTS["secondary_assist"])
                .alias("metric")
            )
            .select(["player_id", "game_id", "metric"])
        )

    # Shots on goal — event_type=shot_on_goal × shooter_id.
    if "shooter_id" in pbp.columns:
        frames.append(
            pbp.filter(
                (pl.col("event_type") == "shot_on_goal")
                & pl.col("shooter_id").is_not_null()
            )
            .group_by(["shooter_id", "game_id"])
            .agg(pl.len().alias("count"))
            .rename({"shooter_id": "player_id"})
            .with_columns(
                (pl.col("count") * _GAME_SCORE_WEIGHTS["shot_on_goal"])
                .alias("metric")
            )
            .select(["player_id", "game_id", "metric"])
        )

    if not frames:
        print("[performance-anomaly] PBP has no scoring/shooting columns.")
        sys.exit(1)

    perf = (
        pl.concat(frames)
        .group_by(["player_id", "game_id"])
        .agg(pl.col("metric").sum())
    )

    df = (
        perf.filter(pl.col("player_id").is_not_null())
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
    parser.add_argument("--metric", type=str, default=DEFAULT_METRIC,
                        help=f"Per-game metric name (default {DEFAULT_METRIC}). "
                             "Currently only a PBP-derived game-score is "
                             "produced; the flag is kept for future per-metric "
                             "switching once xG-per-shot is wired through.")
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
    print(f"  {len(perf):,} (player, game) game-score rows from PBP × schedule")

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
