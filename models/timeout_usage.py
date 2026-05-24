"""Timeout Usage Model — Feature 4.4.

Estimates P(timeout | period, score state, time remaining) per coach.

Data caveat (V1)
----------------
The current PBP ingester (`data/pbp_parser.py`) emits a fixed-set
canonical event_type map that **does not include team timeouts** — the
NHL API exposes them under a stoppage subtype that we drop today.
When this model runs against the live parquets, it therefore finds
zero timeout events.  Rather than fabricating data we fail loudly with
a `DataMissingWarning` and emit an empty parquet so the pipeline still
materializes — when the ingester is extended (one-line change adding
``team-timeout`` to the canonical type map), this model will start
populating immediately without further code changes.

Approach (once timeout events are ingested)
-------------------------------------------
1. From PBP, locate every ``event_type == "timeout"`` (or
   ``event_type_raw in {"team-timeout"}``) event.
2. For each timeout: capture (coach=team, period, time_remaining_secs,
   score_deficit, opponent).
3. Aggregate per (team, period bucket, score-state bucket, time bucket)
   into call rate per opportunity (every game has 3 third-period
   minutes of "timeout pressure").
4. Express the model as a 3-D lookup table per team plus a global
   league baseline; the dashboard renders heatmaps from this.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "timeout_usage_v1"


class DataMissingWarning(UserWarning):
    """Raised when timeout events are absent from PBP."""


# ---------------------------------------------------------------------------
# Output schema — one row per (team, period_bucket, score_state, time_bucket)
# ---------------------------------------------------------------------------

TIMEOUT_USAGE_SCHEMA: dict[str, pl.DataType] = {
    "team":            pl.Utf8,
    "season":          pl.Int64,
    "period_bucket":   pl.Utf8,    # "P1", "P2", "P3_early", "P3_late", "OT"
    "score_state":     pl.Utf8,    # "leading", "tied", "trailing"
    "time_bucket":     pl.Utf8,    # "0-5m", "5-10m", "10-15m", "15-20m"
    "n_timeouts":      pl.Int64,
    "n_games":         pl.Int64,
    "rate_per_game":   pl.Float64, # timeouts taken per game in this bucket
    "model_version":   pl.Utf8,
}


# ---------------------------------------------------------------------------
# Bucketing helpers
# ---------------------------------------------------------------------------


def _period_bucket(period: int, time_in_period_secs: int) -> str:
    if period == 1:
        return "P1"
    if period == 2:
        return "P2"
    if period == 3:
        # Bucket third by half — late half is when coaches actually call timeouts.
        return "P3_late" if time_in_period_secs >= 600 else "P3_early"
    return "OT"


def _score_state(team_score: int, opp_score: int) -> str:
    if team_score > opp_score:
        return "leading"
    if team_score < opp_score:
        return "trailing"
    return "tied"


def _time_bucket(time_in_period_secs: int) -> str:
    m = time_in_period_secs // 60
    if m < 5:
        return "0-5m"
    if m < 10:
        return "5-10m"
    if m < 15:
        return "10-15m"
    return "15-20m"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_timeout_usage(
    pbp_df:      pl.DataFrame,
    season:      int,
    team_lookup: dict[int, str] | None = None,
) -> pl.DataFrame:
    """Compute per-team timeout rate by (period, score_state, time_bucket).

    Args:
        pbp_df:       play-by-play data for the season.  Must include
                      event_type / event_type_raw, home_score, away_score,
                      home_team_id, away_team_id, period, time_in_period_secs,
                      event_owner_team_id.
        season:       e.g. 2025.
        team_lookup:  optional team_id (Int64) → abbrev (str) mapping.

    Returns:
        Polars DataFrame matching TIMEOUT_USAGE_SCHEMA.  Empty + warns
        when no timeout events are present in PBP.
    """
    team_lookup = team_lookup or {}
    known_ids: set[int] | None = set(team_lookup.keys()) if team_lookup else None

    if pbp_df.is_empty():
        return pl.DataFrame(schema=TIMEOUT_USAGE_SCHEMA)

    # Filter to timeout events.  Accept either canonical or raw label.
    cols = pbp_df.columns
    cond = pl.lit(False)
    if "event_type" in cols:
        cond = cond | (pl.col("event_type") == "timeout")
    if "event_type_raw" in cols:
        cond = cond | pl.col("event_type_raw").is_in([
            "team-timeout", "timeout", "tv-timeout",
        ])
    timeouts = pbp_df.filter(cond)

    if timeouts.is_empty():
        warnings.warn(
            "compute_timeout_usage: PBP contains no team-timeout events. "
            "The current ingester (data/pbp_parser.py) drops the stoppage "
            "subtype that carries timeouts. Extend _CANONICAL_TYPE_MAP to "
            "include 'team-timeout' for this model to populate.",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(schema=TIMEOUT_USAGE_SCHEMA)

    # Score columns can be sparse — forward-fill within game/period before bucketing.
    needed = {"period", "time_in_period_secs", "home_score", "away_score",
              "home_team_id", "away_team_id", "event_owner_team_id"}
    missing = needed - set(cols)
    if missing:
        raise ValueError(f"pbp_df missing columns: {sorted(missing)}")

    pbp_sorted = pbp_df.sort(["game_id", "sort_order"]) if "sort_order" in cols else pbp_df.sort(["game_id"])
    pbp_filled = pbp_sorted.with_columns([
        pl.col("home_score").fill_null(strategy="forward").over("game_id"),
        pl.col("away_score").fill_null(strategy="forward").over("game_id"),
    ]).with_columns([
        pl.col("home_score").fill_null(0),
        pl.col("away_score").fill_null(0),
    ])

    # Re-derive timeouts after filling
    t = pbp_filled.filter(cond).with_columns([
        pl.col("period").cast(pl.Int64),
        pl.col("time_in_period_secs").cast(pl.Int64),
        pl.col("home_score").cast(pl.Int64),
        pl.col("away_score").cast(pl.Int64),
        pl.col("home_team_id").cast(pl.Int64),
        pl.col("away_team_id").cast(pl.Int64),
        pl.col("event_owner_team_id").cast(pl.Int64),
    ])

    out_rows: list[dict] = []
    for r in t.iter_rows(named=True):
        owner    = int(r["event_owner_team_id"] or 0)
        home_id  = int(r["home_team_id"] or 0)
        away_id  = int(r["away_team_id"] or 0)
        if owner not in (home_id, away_id):
            continue
        if known_ids is not None and (home_id not in known_ids or away_id not in known_ids):
            continue
        team_id  = owner
        opp_id   = away_id if owner == home_id else home_id
        team_pts = int(r["home_score"]) if team_id == home_id else int(r["away_score"])
        opp_pts  = int(r["away_score"]) if team_id == home_id else int(r["home_score"])
        period   = int(r["period"])
        tip_secs = int(r["time_in_period_secs"])
        out_rows.append({
            "team":   team_lookup.get(team_id, str(team_id)),
            "season": int(season),
            "period_bucket": _period_bucket(period, tip_secs),
            "score_state":   _score_state(team_pts, opp_pts),
            "time_bucket":   _time_bucket(tip_secs),
        })

    if not out_rows:
        return pl.DataFrame(schema=TIMEOUT_USAGE_SCHEMA)

    raw = pl.DataFrame(out_rows)

    # Games-per-team for normalization
    gp = (
        pbp_filled.group_by("game_id")
        .agg([
            pl.col("home_team_id").first().alias("h"),
            pl.col("away_team_id").first().alias("a"),
        ])
    )
    team_gp_counts: dict[str, int] = {}
    for row in gp.iter_rows(named=True):
        ha = [int(row["h"] or 0), int(row["a"] or 0)]
        for tid in ha:
            ab = team_lookup.get(tid, str(tid))
            team_gp_counts[ab] = team_gp_counts.get(ab, 0) + 1

    agg = (
        raw.group_by(["team", "season", "period_bucket", "score_state", "time_bucket"])
        .agg(pl.len().alias("n_timeouts"))
        .with_columns([
            pl.col("team").map_elements(lambda t: team_gp_counts.get(t, 0), return_dtype=pl.Int64).alias("n_games"),
        ])
        .with_columns([
            (pl.col("n_timeouts") / pl.col("n_games").clip(lower_bound=1)).alias("rate_per_game"),
            pl.lit(MODEL_VERSION).alias("model_version"),
        ])
    )

    for col, dtype in TIMEOUT_USAGE_SCHEMA.items():
        if col in agg.columns:
            agg = agg.with_columns(pl.col(col).cast(dtype))
    return agg.select(list(TIMEOUT_USAGE_SCHEMA.keys())).sort(["team", "period_bucket"])


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_timeout_usage(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"timeout_usage_{season}.parquet"
    df.write_parquet(path)
    return path


def read_timeout_usage(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "timeout_usage"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"timeout_usage_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("timeout_usage_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
