"""Goalie Coach Model — Feature 4.8.

Per-coaching-staff save% improvement / regression curve, plus
mid-season change-point detection on team save%.  Designed to
trigger an accelerated Bayesian update on goalie ratings when
the curve breaks (canonical example: MTL 2024–25 mid-season
coaching change).

What it produces
----------------
For each (team, season):

- season_save_pct        — team save% over the season
- prior_save_pct         — team save% in season - 1 (None for first year)
- save_pct_delta         — season - prior (None for first year)
- early_split_save_pct   — first ``split_gp`` games
- late_split_save_pct    — last ``split_gp`` games
- split_delta            — late - early (intra-season trajectory)
- change_point_detected  — bool, True when the absolute split_delta exceeds
                            ``change_threshold``
- rolling_save_pct       — list of save% values over rolling 30-day windows
- goalie_coach           — placeholder (coaches.json does not yet carry the
                            named goalie coach; column reserved for the
                            future v2 ingestion)

V1 caveat: without a per-team named goalie coach, the "delta vs prior
coach" cannot be cleanly partitioned when the team changed coaches
mid-season.  The change-point detector still surfaces the curve break,
which is the actionable signal.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "goalie_coach_v1"

# Number of games at each end of the season used for the split-delta window.
SPLIT_GP_DEFAULT       = 15
# Absolute save% delta between early/late split that flags a change point.
CHANGE_THRESHOLD       = 0.012   # ~1.2 SV%, ≈ 1 goal per 80 shots
# Rolling window size (games) — coarse trace for the dashboard.
ROLLING_WINDOW_DEFAULT = 10


class DataMissingWarning(UserWarning):
    """Raised when goalie coach data is absent or insufficient."""


GOALIE_COACH_SCHEMA: dict[str, pl.DataType] = {
    "team":                  pl.Utf8,
    "season":                pl.Int64,
    "gp":                    pl.Int64,
    "shots_against":         pl.Int64,
    "goals_against":         pl.Int64,
    "season_save_pct":       pl.Float64,
    "prior_save_pct":        pl.Float64,
    "save_pct_delta":        pl.Float64,
    "early_split_save_pct":  pl.Float64,
    "late_split_save_pct":   pl.Float64,
    "split_delta":           pl.Float64,
    "change_point_detected": pl.Boolean,
    "rolling_save_pct":      pl.List(pl.Float64),
    "goalie_coach":          pl.Utf8,
    "model_version":         pl.Utf8,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _per_game_save_stats(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Reduce PBP → per (game_id, team_id) shots-against / goals-against.

    Shots are counted as ``shot_result == 'on_goal'`` plus all ``event_type
    == 'goal'`` events (goals are scored shots and are not duplicated as
    'shot' rows in this PBP schema).
    """
    required = {
        "game_id", "event_type", "event_owner_team_id",
        "home_team_id", "away_team_id", "shot_result",
    }
    missing = required - set(pbp_df.columns)
    if missing:
        raise ValueError(f"pbp_df missing required columns: {sorted(missing)}")

    df = pbp_df.with_columns([
        pl.col("home_team_id").cast(pl.Int64),
        pl.col("away_team_id").cast(pl.Int64),
        pl.col("event_owner_team_id").cast(pl.Int64),
    ])

    # Determine the defending team for each shot/goal event.
    shot_events = (
        df.filter(
            ((pl.col("event_type") == "shot") & (pl.col("shot_result") == "on_goal"))
            | (pl.col("event_type") == "goal")
        )
        .with_columns(
            pl.when(pl.col("event_owner_team_id") == pl.col("home_team_id"))
              .then(pl.col("away_team_id"))
              .when(pl.col("event_owner_team_id") == pl.col("away_team_id"))
              .then(pl.col("home_team_id"))
              .otherwise(None)
              .alias("defending_team_id")
        )
        .filter(pl.col("defending_team_id").is_not_null())
    )

    sa_df = (
        shot_events.group_by(["game_id", "defending_team_id"])
        .agg([
            pl.len().alias("shots_against"),
            (pl.col("event_type") == "goal").sum().cast(pl.Int64).alias("goals_against"),
        ])
        .rename({"defending_team_id": "team_id"})
        .sort(["team_id", "game_id"])
    )
    return sa_df


def _split_save_pct(team_df: pl.DataFrame, split_gp: int) -> tuple[float | None, float | None]:
    """Return (early_save_pct, late_save_pct) using the first/last ``split_gp`` games."""
    if team_df.is_empty():
        return None, None
    n = len(team_df)
    take = min(split_gp, n)
    early = team_df.head(take)
    late  = team_df.tail(take)
    e_sa = int(early["shots_against"].sum() or 0)
    e_ga = int(early["goals_against"].sum() or 0)
    l_sa = int(late["shots_against"].sum() or 0)
    l_ga = int(late["goals_against"].sum() or 0)
    e_sv = 1.0 - (e_ga / e_sa) if e_sa else None
    l_sv = 1.0 - (l_ga / l_sa) if l_sa else None
    return e_sv, l_sv


def _rolling_save_pct(team_df: pl.DataFrame, window_gp: int) -> list[float]:
    """Return a list of rolling save% values across the season.

    Walks forward by ``window_gp`` games at a time so the dashboard sees
    ~5–8 buckets per team (sparse enough to render, dense enough to show
    the curve break).
    """
    if team_df.is_empty():
        return []
    rows = team_df.iter_rows(named=True)
    bucket: list[tuple[int, int]] = []   # (shots_against, goals_against) per game
    for r in rows:
        bucket.append((int(r["shots_against"] or 0), int(r["goals_against"] or 0)))

    out: list[float] = []
    i = 0
    while i < len(bucket):
        chunk = bucket[i:i + window_gp]
        sa = sum(b[0] for b in chunk)
        ga = sum(b[1] for b in chunk)
        if sa:
            out.append(round(1.0 - ga / sa, 4))
        i += window_gp
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_goalie_coach_curve(
    pbp_by_season:   dict[int, pl.DataFrame],
    season:          int,
    team_lookup:     dict[int, str],
    split_gp:        int = SPLIT_GP_DEFAULT,
    rolling_window:  int = ROLLING_WINDOW_DEFAULT,
    change_threshold: float = CHANGE_THRESHOLD,
) -> pl.DataFrame:
    """Build per-team goalie-coach curve rows for the target season.

    Args:
        pbp_by_season:   season-start-year → PBP DataFrame.  Must contain
                         ``season`` (and ideally ``season - 1`` for the
                         year-over-year delta).
        season:          target season-start year.
        team_lookup:     team_id → abbrev.
        split_gp:        games at each end of the season for the intra-season split.
        rolling_window:  game window for the rolling trace.
        change_threshold: absolute split_delta that flags a curve break.
    """
    if season not in pbp_by_season:
        warnings.warn(
            f"compute_goalie_coach_curve: no PBP for season {season} — "
            f"output will be empty.",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(schema=GOALIE_COACH_SCHEMA)

    cur_stats = _per_game_save_stats(pbp_by_season[season])
    prior_stats = (
        _per_game_save_stats(pbp_by_season[season - 1])
        if (season - 1) in pbp_by_season
        else pl.DataFrame(schema={"game_id": pl.Int64, "team_id": pl.Int64,
                                  "shots_against": pl.Int64, "goals_against": pl.Int64})
    )

    rows: list[dict[str, Any]] = []
    for team_id, abbrev in sorted(team_lookup.items(), key=lambda kv: kv[1]):
        t_cur = cur_stats.filter(pl.col("team_id") == team_id)
        if t_cur.is_empty():
            continue

        sa = int(t_cur["shots_against"].sum() or 0)
        ga = int(t_cur["goals_against"].sum() or 0)
        season_sv = 1.0 - (ga / sa) if sa else 0.0

        prior_t = prior_stats.filter(pl.col("team_id") == team_id) if not prior_stats.is_empty() else prior_stats
        p_sa = int(prior_t["shots_against"].sum() or 0) if not prior_t.is_empty() else 0
        p_ga = int(prior_t["goals_against"].sum() or 0) if not prior_t.is_empty() else 0
        prior_sv: float | None = 1.0 - (p_ga / p_sa) if p_sa else None
        sv_delta: float | None = (season_sv - prior_sv) if prior_sv is not None else None

        early_sv, late_sv = _split_save_pct(t_cur, split_gp)
        split_delta: float | None = (
            (late_sv - early_sv) if (early_sv is not None and late_sv is not None) else None
        )
        change_pt = bool(split_delta is not None and abs(split_delta) >= change_threshold)
        rolling = _rolling_save_pct(t_cur, rolling_window)

        # `goalie_coach` not yet populated in coaches.json — column reserved.
        rows.append({
            "team":                  abbrev,
            "season":                int(season),
            "gp":                    int(len(t_cur)),
            "shots_against":         sa,
            "goals_against":         ga,
            "season_save_pct":       round(season_sv, 4) if season_sv else 0.0,
            "prior_save_pct":        round(prior_sv, 4) if prior_sv is not None else float("nan"),
            "save_pct_delta":        round(sv_delta, 4) if sv_delta is not None else float("nan"),
            "early_split_save_pct":  round(early_sv, 4) if early_sv is not None else float("nan"),
            "late_split_save_pct":   round(late_sv, 4)  if late_sv  is not None else float("nan"),
            "split_delta":           round(split_delta, 4) if split_delta is not None else float("nan"),
            "change_point_detected": change_pt,
            "rolling_save_pct":      rolling,
            "goalie_coach":          "",
            "model_version":         MODEL_VERSION,
        })

    if not rows:
        return pl.DataFrame(schema=GOALIE_COACH_SCHEMA)

    df = pl.DataFrame(rows)
    for col, dtype in GOALIE_COACH_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(GOALIE_COACH_SCHEMA.keys())).sort("season_save_pct", descending=True)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_goalie_coach_curve(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"goalie_coach_curve_{season}.parquet"
    df.write_parquet(path)
    return path


def read_goalie_coach_curve(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "goalie_coach_curve"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"goalie_coach_curve_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("goalie_coach_curve_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
