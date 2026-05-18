"""compute_goalie_fi — Feature 3.24 driver script.

Builds per-goalie-per-recent-game fatigue features from PBP + schedule,
loads the pickled GoalieFatigueModel (Feature 2.6) — or falls back to its
hand-tuned default coefficients — and writes a daily snapshot parquet that
the dashboard and the Rust engine can read alongside skater composite FI.

The existing GoalieFatigueModel predicts ``fatigue_sv_delta`` (negative =
expected save% drop due to fatigue). We map that into a positive
``goalie_fi ∈ [0, 1]`` so it lines up with the skater composite_fi scale:

    goalie_fi = clamp(−fatigue_sv_delta / GOALIE_FI_SCALE, 0.0, 1.0)

with ``GOALIE_FI_SCALE = 0.05`` (a 5% save% drop saturates at 1.0).

Inputs in DataStore:
- ``store.pbp()`` with ``game_id`` + ``shot_goalie_id``
- ``schedule_*.parquet`` with ``game_id``, ``game_date``, ``home_team``, ``away_team``

Usage::

    uv run python scripts/gretzky.py goalie-fi
    uv run python scripts/gretzky.py goalie-fi -- --date 2026-05-17 --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import polars as pl

from data.data_store import DataStore
from data.schedule_sync import latest_schedule_parquet
from models.goalie_fatigue_model import (
    GoalieFatigueModel,
    FATIGUE_FEATURE_NAMES,
)


DEFAULT_DATA_DIR = Path(
    os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data"))
)
GOALIE_FATIGUE_SUBDIR = "goalie_fatigue"
SCHEDULE_SUBDIR       = "schedule"

GOALIE_FI_SCALE = 0.05    # 5% expected-save% drop saturates goalie_fi at 1.0
LOOKBACK_DAYS   = 30      # only score recent games (rolling fatigue window)

GOALIE_FI_SCHEMA: dict[str, pl.DataType] = {
    "goalie_id":             pl.Int64,
    "game_id":                pl.Int64,
    "game_date":              pl.Utf8,
    "as_of_date":             pl.Utf8,
    "goalie_fi":              pl.Float64,
    "fatigue_sv_delta":       pl.Float64,
    "is_b2b":                 pl.Int8,
    "rest_days":              pl.Float64,
    "gp_last_7":              pl.Int64,
    "shots_faced_last_7":     pl.Int64,
    "road_game_num":          pl.Int64,
    "component_breakdown":    pl.Utf8,
}


def _build_goalie_game_log(
    pbp: pl.DataFrame,
    schedule: pl.DataFrame,
) -> pl.DataFrame:
    """One row per (goalie_id, game_id) with per-game fatigue features.

    Identifies "the goalie of record" for a game as the one with the
    highest shots-faced count in that game. v1 ignores split games (two
    goalies appearing) — only the heavier-workload goalie is logged.
    """
    if "shot_goalie_id" not in pbp.columns:
        return pl.DataFrame(schema={
            "goalie_id": pl.Int64, "game_id": pl.Int64,
            "shots_faced": pl.Int64,
        })

    shots = (
        pbp.select(["game_id", "shot_goalie_id"])
           .filter(pl.col("shot_goalie_id").is_not_null())
           .rename({"shot_goalie_id": "goalie_id"})
    )
    if len(shots) == 0:
        return pl.DataFrame(schema={
            "goalie_id": pl.Int64, "game_id": pl.Int64,
            "shots_faced": pl.Int64,
        })

    per_goalie_game = (
        shots.group_by(["game_id", "goalie_id"])
             .agg(pl.len().alias("shots_faced"))
    )

    # Goalie of record per game = max shots_faced (ties broken by min goalie_id)
    starters = (
        per_goalie_game.sort(
            ["game_id", "shots_faced", "goalie_id"],
            descending=[False, True, False],
        )
        .group_by("game_id")
        .agg(pl.col("goalie_id").first().alias("goalie_id"),
             pl.col("shots_faced").first().alias("shots_faced"))
    )

    log = starters.join(
        schedule.select(["game_id", "game_date", "home_team", "away_team"]),
        on="game_id", how="inner",
    )

    # Rolling fatigue features per goalie (chronological per goalie).
    log = log.sort(["goalie_id", "game_date", "game_id"])

    # rest_days = days since previous start; first start = 7 (treat as well-rested)
    log = log.with_columns(
        pl.col("game_date").cast(pl.Date).alias("_d"),
    )
    log = log.with_columns(
        (pl.col("_d") - pl.col("_d").shift(1).over("goalie_id"))
            .dt.total_days()
            .fill_null(7)
            .clip(0, 7)
            .cast(pl.Float64)
            .alias("rest_days"),
    )
    log = log.with_columns(
        (pl.col("rest_days") <= 1).cast(pl.Int8).alias("is_b2b"),
    )

    # GP / shots faced in last 7 and 14 days (using rolling counts over goalie).
    log = log.with_columns([
        pl.col("game_id").cum_count().over("goalie_id").alias("gp_career"),
    ])
    # Use a per-row lookback by counting prior games within window.
    rows = log.to_dicts()
    by_goalie: dict[int, list[dict]] = {}
    for r in rows:
        by_goalie.setdefault(int(r["goalie_id"]), []).append(r)

    out_rows: list[dict] = []
    for gid, games in by_goalie.items():
        # Already sorted by game_date.
        for i, g in enumerate(games):
            d_i = datetime.strptime(g["game_date"], "%Y-%m-%d").date()
            gp7  = 0
            gp14 = 0
            sh7  = 0
            for prev in games[:i]:
                d_p = datetime.strptime(prev["game_date"], "%Y-%m-%d").date()
                gap = (d_i - d_p).days
                if 0 < gap <= 7:
                    gp7 += 1
                    sh7 += int(prev["shots_faced"])
                if 0 < gap <= 14:
                    gp14 += 1
            # Road game position (consecutive away starts)
            road_n = 0
            j = i
            while j >= 0:
                gg = games[j]
                if gg["away_team"] == gg.get("_team_abbrev"):
                    # We don't carry a goalie-side abbrev in PBP cleanly; fall back to
                    # using away_team membership against the goalie's most-recent team.
                    pass
                j -= 1
            # Use away_team-distinct streak as a rough proxy: how many of the
            # last 5 starts were not at the same home venue.
            road_n = 0
            recent = games[max(0, i-4):i+1]
            home_set = {gg["home_team"] for gg in recent}
            road_n = max(0, len(home_set) - 1)

            out_rows.append({
                "goalie_id":      gid,
                "game_id":        int(g["game_id"]),
                "game_date":      g["game_date"],
                "is_b2b":         int(g["is_b2b"]),
                "rest_days":      float(g["rest_days"]),
                "gp_last_7":      gp7,
                "gp_last_14":     gp14,
                "shots_faced_last_7": sh7,
                "road_game_num":  road_n,
                "saves_last_5":   0,    # not yet available in v1 (no per-game saves table)
                "age":            30.0, # default age — v2: join with rosters
                "toi_last_5_avg": 3600.0,  # default 60min/game baseline
            })

    if not out_rows:
        return pl.DataFrame(schema={
            "goalie_id": pl.Int64, "game_id": pl.Int64,
            "shots_faced_last_7": pl.Int64,
        })
    return pl.DataFrame(out_rows)


def _load_model(args) -> GoalieFatigueModel:
    """Load pickled model from goalie_fatigue/goalie_fatigue_model.pkl or use defaults."""
    pkl = args.data_dir / GOALIE_FATIGUE_SUBDIR / "goalie_fatigue_model.pkl"
    if pkl.exists():
        try:
            return GoalieFatigueModel.load(pkl)
        except Exception as e:
            print(f"[goalie-fi] WARN: failed to load {pkl}: {e}. Using defaults.")
    return GoalieFatigueModel()


def _to_fi(delta: float, scale: float = GOALIE_FI_SCALE) -> float:
    """Map fatigue_sv_delta (negative = drop) → goalie_fi ∈ [0, 1]."""
    if delta is None or not np.isfinite(delta):
        return 0.0
    return float(max(0.0, min(1.0, -float(delta) / scale)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-goalie daily Fatigue Index snapshot (Feature 3.24)."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None,
                        help="as-of date. Default: today (UTC).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()
    out_dir  = args.data_dir / GOALIE_FATIGUE_SUBDIR
    out_path = out_dir / f"goalie_fi_{as_of}.parquet"
    if out_path.exists() and not args.force:
        print(f"[goalie-fi] Output already exists: {out_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    sched_dir  = args.data_dir / SCHEDULE_SUBDIR
    sched_path = latest_schedule_parquet(sched_dir)
    if sched_path is None:
        print(f"[goalie-fi] No schedule parquet under {sched_dir}.")
        sys.exit(1)
    schedule = pl.read_parquet(sched_path)

    print(f"[goalie-fi] Building goalie game log as of {as_of}…")
    with DataStore(args.data_dir) as store:
        pbp = store.pbp()
    if len(pbp) == 0:
        print("[goalie-fi] DataStore has no PBP. Run `ingest` first.")
        sys.exit(1)

    game_log = _build_goalie_game_log(pbp, schedule)
    if len(game_log) == 0:
        print("[goalie-fi] No goalie starts identified — writing empty parquet.")
        empty = pl.DataFrame(
            {c: pl.Series([], dtype=t) for c, t in GOALIE_FI_SCHEMA.items()}
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        empty.write_parquet(out_path)
        print(f"[goalie-fi] Written: {out_path}")
        return

    # Keep only games within LOOKBACK_DAYS of as_of (current fatigue is recent)
    cutoff = (
        datetime.strptime(as_of, "%Y-%m-%d").date()
    )
    game_log = game_log.filter(
        pl.col("game_date").map_elements(
            lambda d: (cutoff - datetime.strptime(d, "%Y-%m-%d").date()).days <= LOOKBACK_DAYS,
            return_dtype=pl.Boolean,
        )
    )
    if len(game_log) == 0:
        print(f"[goalie-fi] No goalie starts in last {LOOKBACK_DAYS} days. Empty output.")
        out_dir.mkdir(parents=True, exist_ok=True)
        empty = pl.DataFrame(
            {c: pl.Series([], dtype=t) for c, t in GOALIE_FI_SCHEMA.items()}
        )
        empty.write_parquet(out_path)
        print(f"[goalie-fi] Written: {out_path}")
        return

    # Schedule features compatible with GoalieFatigueModel.build_fatigue_features
    feature_df = game_log.rename({"goalie_id": "player_id"}).select([
        "player_id", "game_id", "is_b2b", "rest_days", "gp_last_7", "gp_last_14",
        "road_game_num", "saves_last_5", "age", "toi_last_5_avg",
    ])

    model = _load_model(args)
    print(f"  Model version: {model.version if hasattr(model, 'version') else 'default'}")
    print(f"  Coefficients: {model.coefficients()}")

    # Build feature matrix + predict directly (avoid predict_season's strict schema)
    from models.goalie_fatigue_model import build_fatigue_features
    X, _ = build_fatigue_features(feature_df)
    deltas = model.predict(X)

    out_rows: list[dict] = []
    for r, delta in zip(game_log.to_dicts(), deltas):
        fi = _to_fi(float(delta))
        comps = {
            "is_b2b":         model.coefficients()["is_b2b"]        * int(r["is_b2b"]),
            "rest_days":      model.coefficients()["rest_days"]     * float(r["rest_days"]),
            "gp_last_7":      model.coefficients()["gp_last_7"]     * int(r["gp_last_7"]),
            "shots_faced_last_7": float(r["shots_faced_last_7"]),
            "road_game_num":  model.coefficients()["road_game_num"] * int(r["road_game_num"]),
        }
        out_rows.append({
            "goalie_id":            int(r["goalie_id"]),
            "game_id":              int(r["game_id"]),
            "game_date":            r["game_date"],
            "as_of_date":           as_of,
            "goalie_fi":            fi,
            "fatigue_sv_delta":     float(delta),
            "is_b2b":               int(r["is_b2b"]),
            "rest_days":            float(r["rest_days"]),
            "gp_last_7":            int(r["gp_last_7"]),
            "shots_faced_last_7":   int(r["shots_faced_last_7"]),
            "road_game_num":        int(r["road_game_num"]),
            "component_breakdown":  json.dumps(
                {k: round(v, 6) for k, v in comps.items()},
                sort_keys=True,
            ),
        })

    if not out_rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        empty = pl.DataFrame(
            {c: pl.Series([], dtype=t) for c, t in GOALIE_FI_SCHEMA.items()}
        )
        empty.write_parquet(out_path)
        print(f"[goalie-fi] Written: {out_path}")
        return

    df = pl.DataFrame(out_rows, schema=GOALIE_FI_SCHEMA)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"  {len(df):,} goalie-game rows produced.")

    top = df.sort("goalie_fi", descending=True).head(10)
    print("\n  Top 10 most-fatigued goalie-game rows:")
    print(f"  {'Goalie':<10}  {'Game':<10}  {'Date':<10}  {'FI':>5}  "
          f"{'B2B':>3}  {'Rest':>4}  {'GP7':>3}  {'Sh7':>4}")
    for r in top.to_dicts():
        print(
            f"  {r['goalie_id']:<10}  {r['game_id']:<10}  {r['game_date']:<10}  "
            f"{r['goalie_fi']:>5.3f}  {r['is_b2b']:>3d}  "
            f"{r['rest_days']:>4.1f}  {r['gp_last_7']:>3d}  "
            f"{r['shots_faced_last_7']:>4d}"
        )

    print(f"\n[goalie-fi] Written: {out_path}")


if __name__ == "__main__":
    main()
