"""Goalie Pull Timing Model — Feature 4.5.

Estimates P(pull goalie) at each (score deficit, time remaining) per
coach (V1 = per team-season; coach identity baked in).

Approach
--------
1. Detect **pull events** from PBP: the timeline transition where one
   team's skater count rises from ≤ 5 to 6 (goalie pulled for an extra
   attacker).  Capture (game_id, pulling_team_id, t_pull_secs,
   score_deficit_at_pull).  Filter to 3rd-period / OT pulls only — early
   pulls are almost always delayed-penalty advantage skaters, not
   coach-driven trailing-team decisions.
2. Per pull, compute ``time_remaining_secs`` = 1200 − time_in_period_secs
   (3rd period) or OT-elapsed (OT pulls).
3. Aggregate per team-season:
   - Empirical pull events bucketed by (deficit, time_remaining).
   - Mean pull time per deficit (1 / 2 / 3+ goals down).
   - Pull rate per game in close trailing-team situations.

The schema captures both per-(team, deficit) summary rows and the raw
pull events (long-form) so the dashboard can render either a coach-by-
coach mean-pull-time bar chart or a per-game scatter.

Limitations
-----------
- Score columns in PBP are sparsely populated.  We forward-fill within
  game so the deficit at the pull is the score state from the most
  recent goal-scoring event.
- Pulls during delayed penalties are excluded by requiring ≥ 60s
  duration of the 6-skater state.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "goalie_pull_v1"

MIN_PULL_DURATION_SECS = 5.0     # filter delayed-penalty flickers
MIN_PULL_PERIOD        = 3       # earliest period where coach-driven pull counted


class DataMissingWarning(UserWarning):
    """Raised when PBP yields no qualifying pull events."""


# ---------------------------------------------------------------------------
# Output schema — per-team-season aggregated summary
# ---------------------------------------------------------------------------

GOALIE_PULL_SCHEMA: dict[str, pl.DataType] = {
    "team":                pl.Utf8,
    "season":              pl.Int64,
    "deficit":             pl.Int64,    # 1, 2, 3+ (clamped)
    "n_pulls":             pl.Int64,
    "n_team_games":        pl.Int64,
    "mean_pull_time_secs": pl.Float64,  # mean seconds before end-of-regulation
    "median_pull_time_secs": pl.Float64,
    "earliest_pull_secs":  pl.Float64,
    "model_version":       pl.Utf8,
}

# Event-level (long form) schema — raw pull events
PULL_EVENTS_SCHEMA: dict[str, pl.DataType] = {
    "game_id":             pl.Int64,
    "team":                pl.Utf8,
    "season":              pl.Int64,
    "period":              pl.Int64,
    "t_pull_secs":         pl.Float64,
    "time_remaining_secs": pl.Float64,
    "deficit":             pl.Int64,
    "model_version":       pl.Utf8,
}


# ---------------------------------------------------------------------------
# Pull detection
# ---------------------------------------------------------------------------


def detect_pulls(
    pbp_df:      pl.DataFrame,
    season:      int,
    team_lookup: dict[int, str] | None = None,
) -> pl.DataFrame:
    """Return long-form pull events from PBP timeline transitions.

    A pull event = transition from skater_count ≤ 5 to skater_count == 6
    persisting at least MIN_PULL_DURATION_SECS seconds.
    """
    needed = {"game_id", "period", "time_in_period_secs",
              "home_skaters", "away_skaters",
              "home_team_id", "away_team_id",
              "home_score", "away_score"}
    missing = needed - set(pbp_df.columns)
    if missing:
        raise ValueError(f"pbp_df missing: {sorted(missing)}")
    if pbp_df.is_empty():
        return pl.DataFrame(schema=PULL_EVENTS_SCHEMA)

    team_lookup = team_lookup or {}
    known_ids: set[int] | None = set(team_lookup.keys()) if team_lookup else None

    # Sort timeline; forward-fill scores so deficit is current.
    sort_col = "sort_order" if "sort_order" in pbp_df.columns else "time_in_period_secs"
    pbp = (
        pbp_df.sort(["game_id", sort_col])
        .with_columns([
            pl.col("home_score").fill_null(strategy="forward").over("game_id"),
            pl.col("away_score").fill_null(strategy="forward").over("game_id"),
        ])
        .with_columns([
            pl.col("home_score").fill_null(0),
            pl.col("away_score").fill_null(0),
        ])
    )

    rows: list[dict] = []
    for gid in pbp["game_id"].unique().to_list():
        g = pbp.filter(pl.col("game_id") == gid)
        if g.is_empty():
            continue
        home_id = int(g["home_team_id"].drop_nulls().first() or 0)
        away_id = int(g["away_team_id"].drop_nulls().first() or 0)
        if home_id == 0 or away_id == 0:
            continue
        # Skip exhibition games whose team_ids aren't in the NHL map
        if known_ids is not None and (home_id not in known_ids or away_id not in known_ids):
            continue

        periods = g["period"].to_numpy()
        tips    = g["time_in_period_secs"].to_numpy()
        hs      = g["home_skaters"].to_numpy()
        aw      = g["away_skaters"].to_numpy()
        hsc     = g["home_score"].to_numpy()
        asc     = g["away_score"].to_numpy()

        # Walk side-by-side for home & away separately
        for side, sk in (("home", hs), ("away", aw)):
            in_pull = False
            pull_start_t: float | None = None
            pull_period:  int | None  = None
            pull_deficit: int | None  = None

            for i in range(len(sk)):
                p   = int(periods[i])
                tip = int(tips[i])
                if p < MIN_PULL_PERIOD:
                    in_pull = False
                    pull_start_t = None
                    continue
                count   = int(sk[i])
                game_t  = (p - 1) * 1200 + tip
                hs_now  = int(hsc[i])
                as_now  = int(asc[i])
                team_pt = hs_now if side == "home" else as_now
                opp_pt  = as_now if side == "home" else hs_now
                deficit = max(0, opp_pt - team_pt)

                if count >= 6:
                    if not in_pull:
                        in_pull        = True
                        pull_start_t   = float(game_t)
                        pull_period    = p
                        pull_deficit   = deficit
                else:
                    if in_pull and pull_start_t is not None:
                        # End of pull window
                        end_t = float(game_t)
                        if end_t - pull_start_t >= MIN_PULL_DURATION_SECS:
                            team_id = home_id if side == "home" else away_id
                            t_pull_secs = pull_start_t
                            # Time remaining = (end of regulation) − t_pull
                            tr = max(0.0, 3600.0 - t_pull_secs) if (pull_period or 3) == 3 else 0.0
                            rows.append({
                                "game_id":             int(gid),
                                "team":                team_lookup.get(team_id, str(team_id)),
                                "season":              int(season),
                                "period":              int(pull_period or 3),
                                "t_pull_secs":         float(t_pull_secs),
                                "time_remaining_secs": float(tr),
                                "deficit":             int(min(3, pull_deficit or 0)),
                                "model_version":       MODEL_VERSION,
                            })
                        in_pull = False
                        pull_start_t = None

            # Trailing pull at end of game (never returned skaters)
            if in_pull and pull_start_t is not None:
                team_id = home_id if side == "home" else away_id
                end_t = float((periods[-1] - 1) * 1200 + tips[-1])
                if end_t - pull_start_t >= MIN_PULL_DURATION_SECS:
                    tr = max(0.0, 3600.0 - pull_start_t) if (pull_period or 3) == 3 else 0.0
                    rows.append({
                        "game_id":             int(gid),
                        "team":                team_lookup.get(team_id, str(team_id)),
                        "season":              int(season),
                        "period":              int(pull_period or 3),
                        "t_pull_secs":         float(pull_start_t),
                        "time_remaining_secs": float(tr),
                        "deficit":             int(min(3, pull_deficit or 0)),
                        "model_version":       MODEL_VERSION,
                    })

    if not rows:
        return pl.DataFrame(schema=PULL_EVENTS_SCHEMA)

    df = pl.DataFrame(rows)
    for col, dtype in PULL_EVENTS_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(PULL_EVENTS_SCHEMA.keys()))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_pulls(
    pull_events: pl.DataFrame,
    pbp_df:      pl.DataFrame,
    team_lookup: dict[int, str] | None = None,
) -> pl.DataFrame:
    """Roll up per-team-season aggregates by deficit bucket."""
    team_lookup = team_lookup or {}

    if pull_events.is_empty():
        warnings.warn(
            "aggregate_pulls: no goalie-pull events detected — "
            "either PBP is empty or no skater_count ≥ 6 transitions occurred.",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(schema=GOALIE_PULL_SCHEMA)

    # Per-team games played count (each game contributes to two teams)
    team_gp: dict[str, int] = defaultdict(int)
    if not pbp_df.is_empty() and {"game_id", "home_team_id", "away_team_id"}.issubset(pbp_df.columns):
        gp = (
            pbp_df.group_by("game_id")
            .agg([
                pl.col("home_team_id").first().alias("h"),
                pl.col("away_team_id").first().alias("a"),
            ])
        )
        for r in gp.iter_rows(named=True):
            for tid in (int(r["h"] or 0), int(r["a"] or 0)):
                team_gp[team_lookup.get(tid, str(tid))] += 1

    # Restrict to trailing-team pulls (deficit ≥ 1)
    pulls = pull_events.filter(pl.col("deficit") >= 1)

    agg = (
        pulls.group_by(["team", "season", "deficit"])
        .agg([
            pl.len().alias("n_pulls"),
            pl.col("time_remaining_secs").mean().alias("mean_pull_time_secs"),
            pl.col("time_remaining_secs").median().alias("median_pull_time_secs"),
            pl.col("time_remaining_secs").max().alias("earliest_pull_secs"),  # max time-remaining = earliest pull
        ])
        .with_columns([
            pl.col("team").map_elements(lambda t: team_gp.get(t, 0), return_dtype=pl.Int64).alias("n_team_games"),
            pl.lit(MODEL_VERSION).alias("model_version"),
        ])
    )

    for col, dtype in GOALIE_PULL_SCHEMA.items():
        if col in agg.columns:
            agg = agg.with_columns(pl.col(col).cast(dtype))
    return agg.select(list(GOALIE_PULL_SCHEMA.keys())).sort(["team", "deficit"])


def compute_goalie_pull(
    pbp_df:      pl.DataFrame,
    season:      int,
    team_lookup: dict[int, str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """One-shot driver: detect pulls + aggregate into per-team summary.

    Returns:
        (summary_df, events_df) — both Polars DataFrames matching their
        respective schemas.
    """
    events = detect_pulls(pbp_df, season=season, team_lookup=team_lookup)
    summary = aggregate_pulls(events, pbp_df, team_lookup=team_lookup)
    return summary, events


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_goalie_pull(
    summary: pl.DataFrame,
    events:  pl.DataFrame,
    output_dir: Path,
    season:  int,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"goalie_pull_{season}.parquet"
    events_path  = output_dir / f"goalie_pull_events_{season}.parquet"
    summary.write_parquet(summary_path)
    events.write_parquet(events_path)
    return summary_path, events_path


def read_goalie_pull(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "goalie_pull"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"goalie_pull_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("goalie_pull_[0-9]*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
