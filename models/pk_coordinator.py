"""PK Coordinator Model — Feature 4.10.

Mirror of the PP coordinator (4.9), but for penalty kill.  Different
*owner* (the PK coordinator) and different system levers:

- **Structure / shots-against**: How leaky is the PK box?
- **Forecheck pressure**: Does the PK pressure the puck (more SH shots-for,
  occasional SH goals) or sit back (zero pressure, all OZ time for opp)?
- **xGA quality**: Are the shots they allow from the slot or the perimeter?

What it produces (one row per team per season)
----------------------------------------------
Volume / pressure:
    pk_toi_secs               — total PK time-on-ice (team-seconds)
    pk_team_gp                — PK team-games (from st_deployment)
    pk_sa                     — shots-against on the PK
    pk_ga                     — goals-against on the PK
    pk_xga_total              — sum of x_goal against on the PK
    pk_sa_per_60              — shots-against / 60 of PK TOI (lower = tighter box)
    pk_xga_per_60             — xGA / 60 of PK TOI
    pk_ga_per_60              — actual GA / 60 of PK TOI
Quality:
    pk_save_pct               — 1 - GA/SA on the PK (team SH save%)
    pk_xga_per_shot           — average shot quality conceded
    pk_shot_distance_avg      — average distance of PK shots-against (ft)
Forecheck pressure:
    sh_shots_for              — shorthanded shots produced
    sh_goals_for              — shorthanded goals scored
    sh_shots_per_60           — SH shots / 60 of PK TOI
PK1 share (signature stability):
    pk1_share                 — PK1 unit TOI / total PK TOI

Data sources
------------
- ``data/raw/pbp_{season}.parquet``                — strength + shot events
- ``data/shots/shots_{season}.parquet``            — MoneyPuck xG values
- ``data/st_deployment/st_deployment_{season}``    — PK1 TOI + PK TOI total

V1 caveat
---------
``coaches.json`` doesn't yet carry the named PK coordinator; the column
is reserved so the dashboard surfaces *who* the system belongs to as
soon as the schema is extended.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "pk_coordinator_v1"


class DataMissingWarning(UserWarning):
    """Raised when PK coordinator inputs are absent or insufficient."""


PK_COORDINATOR_SCHEMA: dict[str, pl.DataType] = {
    "team":                   pl.Utf8,
    "season":                  pl.Int64,
    "pk_toi_secs":             pl.Float64,
    "pk_team_gp":              pl.Int64,
    "pk_sa":                   pl.Int64,
    "pk_ga":                   pl.Int64,
    "pk_xga_total":            pl.Float64,
    "pk_sa_per_60":            pl.Float64,
    "pk_xga_per_60":           pl.Float64,
    "pk_ga_per_60":            pl.Float64,
    "pk_save_pct":             pl.Float64,
    "pk_xga_per_shot":         pl.Float64,
    "pk_shot_distance_avg":    pl.Float64,
    "sh_shots_for":            pl.Int64,
    "sh_goals_for":            pl.Int64,
    "sh_shots_per_60":         pl.Float64,
    "pk1_share":               pl.Float64,
    "pk_coordinator":          pl.Utf8,
    "model_version":           pl.Utf8,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pk_shot_stats(shots_df: pl.DataFrame) -> pl.DataFrame:
    """Per-team PK shots-against + SH shots-for using MoneyPuck shots.

    A PK situation = the *defending* team is short-handed (e.g. home_skaters=4
    while away_skaters=5).  Shots-against on the PK are filtered by the
    defending team's skater count being LESS than the opponent's.

    Returns:
        DataFrame with columns: team, pk_sa, pk_ga, pk_xga_total,
        pk_shot_distance_avg, sh_shots_for, sh_goals_for
    """
    required = {
        "shooting_team", "home_team", "away_team", "home_skaters",
        "away_skaters", "x_goal", "shot_distance", "is_goal",
    }
    missing = required - set(shots_df.columns)
    if missing:
        raise ValueError(f"shots_df missing required columns: {sorted(missing)}")

    if shots_df.is_empty():
        return pl.DataFrame(schema={
            "team":                  pl.Utf8,
            "pk_sa":                 pl.Int64,
            "pk_ga":                 pl.Int64,
            "pk_xga_total":          pl.Float64,
            "pk_shot_distance_avg":  pl.Float64,
            "sh_shots_for":          pl.Int64,
            "sh_goals_for":          pl.Int64,
        })

    with_meta = shots_df.with_columns([
        pl.when(pl.col("shooting_team") == pl.col("home_team"))
          .then(pl.col("home_skaters")).otherwise(pl.col("away_skaters"))
          .alias("skaters_for"),
        pl.when(pl.col("shooting_team") == pl.col("home_team"))
          .then(pl.col("away_skaters")).otherwise(pl.col("home_skaters"))
          .alias("skaters_against"),
        pl.when(pl.col("shooting_team") == pl.col("home_team"))
          .then(pl.col("away_team")).otherwise(pl.col("home_team"))
          .alias("defending_team"),
    ])

    # PK shots-against: the *defending team* is shorthanded (5v4 the wrong way).
    pk_against = (
        with_meta.filter(
            (pl.col("skaters_against") < pl.col("skaters_for"))
            & pl.col("skaters_for").is_in([5, 4])
            & pl.col("skaters_against").is_in([4, 3])
        )
        .group_by("defending_team")
        .agg([
            pl.len().alias("pk_sa"),
            pl.col("is_goal").sum().cast(pl.Int64).alias("pk_ga"),
            pl.col("x_goal").sum().alias("pk_xga_total"),
            pl.col("shot_distance").mean().alias("pk_shot_distance_avg"),
        ])
        .rename({"defending_team": "team"})
    )

    # SH shots-for: shooter is shorthanded.
    sh_for = (
        with_meta.filter(
            (pl.col("skaters_for") < pl.col("skaters_against"))
            & pl.col("skaters_for").is_in([4, 3])
            & pl.col("skaters_against").is_in([5, 4])
        )
        .group_by("shooting_team")
        .agg([
            pl.len().alias("sh_shots_for"),
            pl.col("is_goal").sum().cast(pl.Int64).alias("sh_goals_for"),
        ])
        .rename({"shooting_team": "team"})
    )

    if pk_against.is_empty() and sh_for.is_empty():
        return pl.DataFrame(schema={
            "team":                  pl.Utf8,
            "pk_sa":                 pl.Int64,
            "pk_ga":                 pl.Int64,
            "pk_xga_total":          pl.Float64,
            "pk_shot_distance_avg":  pl.Float64,
            "sh_shots_for":          pl.Int64,
            "sh_goals_for":          pl.Int64,
        })

    return pk_against.join(sh_for, on="team", how="full", coalesce=True).with_columns([
        pl.col("pk_sa").fill_null(0),
        pl.col("pk_ga").fill_null(0),
        pl.col("pk_xga_total").fill_null(0.0),
        pl.col("pk_shot_distance_avg").fill_null(0.0),
        pl.col("sh_shots_for").fill_null(0),
        pl.col("sh_goals_for").fill_null(0),
    ])


def _pk_unit_info(st_df: pl.DataFrame) -> pl.DataFrame:
    """PK TOI + PK1 share per team."""
    if st_df.is_empty():
        return pl.DataFrame(schema={
            "team": pl.Utf8, "pk_toi_secs": pl.Float64,
            "pk_team_gp": pl.Int64, "pk1_share": pl.Float64,
        })

    pk_only = st_df.filter(pl.col("unit_type").is_in(["PK1", "PK2"]))
    if pk_only.is_empty():
        return pl.DataFrame(schema={
            "team": pl.Utf8, "pk_toi_secs": pl.Float64,
            "pk_team_gp": pl.Int64, "pk1_share": pl.Float64,
        })

    rows: list[dict] = []
    for team, group in pk_only.group_by("team"):
        team_abbrev = team[0] if isinstance(team, tuple) else team
        pk_toi   = float(group["team_st_toi"].first() or 0.0)
        pk_gp    = int(group["team_st_gp"].first() or 0)

        pk1 = group.filter(pl.col("unit_type") == "PK1")
        pk1_share = 0.0
        if not pk1.is_empty() and pk_toi > 0:
            pk1_toi = float(pk1.row(0, named=True)["unit_toi_secs"] or 0.0)
            pk1_share = pk1_toi / pk_toi

        rows.append({
            "team":         team_abbrev,
            "pk_toi_secs":  pk_toi,
            "pk_team_gp":   pk_gp,
            "pk1_share":    pk1_share,
        })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_pk_coordinator(
    shots_df:      pl.DataFrame,
    st_df:         pl.DataFrame,
    season:        int,
) -> pl.DataFrame:
    """Build per-team PK coordinator system signature rows."""
    if shots_df.is_empty():
        warnings.warn(
            "compute_pk_coordinator: empty shots_df — PK metrics will be zero.",
            DataMissingWarning,
            stacklevel=2,
        )

    shot_stats = _pk_shot_stats(shots_df)
    pk_info    = _pk_unit_info(st_df)

    if pk_info.is_empty():
        warnings.warn(
            "compute_pk_coordinator: ST deployment frame had no PK units — no rows produced.",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(schema=PK_COORDINATOR_SCHEMA)

    base = pk_info.join(shot_stats, on="team", how="left").with_columns([
        pl.col("pk_sa").fill_null(0),
        pl.col("pk_ga").fill_null(0),
        pl.col("pk_xga_total").fill_null(0.0),
        pl.col("pk_shot_distance_avg").fill_null(0.0),
        pl.col("sh_shots_for").fill_null(0),
        pl.col("sh_goals_for").fill_null(0),
    ])

    df = base.with_columns([
        pl.when(pl.col("pk_toi_secs") > 0)
          .then(pl.col("pk_sa") / pl.col("pk_toi_secs") * 3600.0)
          .otherwise(0.0).alias("pk_sa_per_60"),
        pl.when(pl.col("pk_toi_secs") > 0)
          .then(pl.col("pk_xga_total") / pl.col("pk_toi_secs") * 3600.0)
          .otherwise(0.0).alias("pk_xga_per_60"),
        pl.when(pl.col("pk_toi_secs") > 0)
          .then(pl.col("pk_ga") / pl.col("pk_toi_secs") * 3600.0)
          .otherwise(0.0).alias("pk_ga_per_60"),
        pl.when(pl.col("pk_sa") > 0)
          .then(1.0 - pl.col("pk_ga") / pl.col("pk_sa"))
          .otherwise(0.0).alias("pk_save_pct"),
        pl.when(pl.col("pk_sa") > 0)
          .then(pl.col("pk_xga_total") / pl.col("pk_sa"))
          .otherwise(0.0).alias("pk_xga_per_shot"),
        pl.when(pl.col("pk_toi_secs") > 0)
          .then(pl.col("sh_shots_for") / pl.col("pk_toi_secs") * 3600.0)
          .otherwise(0.0).alias("sh_shots_per_60"),
    ]).with_columns([
        pl.lit(int(season)).alias("season"),
        pl.lit("").alias("pk_coordinator"),
        pl.lit(MODEL_VERSION).alias("model_version"),
    ])

    for col, dtype in PK_COORDINATOR_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(PK_COORDINATOR_SCHEMA.keys())).sort("pk_save_pct", descending=True)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_pk_coordinator(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"pk_coordinator_{season}.parquet"
    df.write_parquet(path)
    return path


def read_pk_coordinator(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "pk_coordinator"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"pk_coordinator_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("pk_coordinator_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
