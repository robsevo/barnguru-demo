"""PP Coordinator Model — Feature 4.9.

Per-team power-play **system signature**.  Different from the head-coach
penalty tendency (4.6) — this is the *style* of the PP unit, owned by
the PP coordinator: how much they shoot, how good the looks are,
whether they prefer to carry the puck in, and who runs the umbrella.

What it produces (one row per team per season)
----------------------------------------------
Volume:
    pp_shots_per_60       — MoneyPuck shots with strength_for=5, against=4
    pp_xg_per_60          — xG (x_goal) per 60 of PP TOI
    pp_goals_per_60       — actual goals per 60 of PP TOI
Quality:
    pp_xg_per_shot        — average x_goal per shot — proxy for shot quality
    pp_shot_distance_avg  — average shot distance (ft from net)
Zone entry:
    pp_carry_pct          — % of PP zone-entry events that were controlled
                             (``carry_in == true``) from PBP
QB usage (PP1 lead D — the umbrella quarterback):
    pp1_qb_id             — the single D on the PP1 unit, if there is one
                             (no D = unknown; multiple D = the first listed)
    pp1_qb_name
    pp1_qb_share          — PP1 unit share of total PP TOI (signature stability)

Data sources
------------
- ``data/raw/pbp_{season}.parquet``                — strength + zone_entry events
- ``data/shots/shots_{season}.parquet``            — MoneyPuck xG + skater counts
- ``data/st_deployment/st_deployment_{season}``    — PP1 personnel + PP TOI
- ``data/raw/shots_{season}.parquet`` (shooter→position) — to identify the D on PP1

Like 4.8 the named PP coordinator isn't yet in ``data/coaches.json`` —
the column is reserved; v1 surfaces the system signature itself.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "pp_coordinator_v1"


class DataMissingWarning(UserWarning):
    """Raised when PP coordinator inputs are absent or insufficient."""


PP_COORDINATOR_SCHEMA: dict[str, pl.DataType] = {
    "team":                  pl.Utf8,
    "season":                pl.Int64,
    "pp_toi_secs":           pl.Float64,
    "pp_team_gp":            pl.Int64,
    "pp_shots":              pl.Int64,
    "pp_goals":              pl.Int64,
    "pp_xg_total":           pl.Float64,
    "pp_shots_per_60":       pl.Float64,
    "pp_xg_per_60":          pl.Float64,
    "pp_goals_per_60":       pl.Float64,
    "pp_xg_per_shot":        pl.Float64,
    "pp_shot_distance_avg":  pl.Float64,
    "pp_carry_pct":          pl.Float64,
    "pp1_qb_id":             pl.Int64,
    "pp1_qb_name":           pl.Utf8,
    "pp1_qb_share":          pl.Float64,
    "pp_coordinator":        pl.Utf8,
    "model_version":         pl.Utf8,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pp_shot_stats(shots_df: pl.DataFrame) -> pl.DataFrame:
    """Filter MoneyPuck shots → power-play-for events; aggregate per team.

    PP-for-shooter is identified by skater asymmetry: shooting team has
    more skaters than the opponent.  Pulled goalies (en) and 4v4 are
    excluded by requiring exactly 5v4 / 5v3 / 4v3.
    """
    required = {
        "shooting_team", "home_team", "away_team", "home_skaters",
        "away_skaters", "x_goal", "shot_distance", "is_goal",
    }
    missing = required - set(shots_df.columns)
    if missing:
        raise ValueError(f"shots_df missing required columns: {sorted(missing)}")

    # Map (home_skaters, away_skaters) → (skaters_for, skaters_against)
    pp = (
        shots_df.with_columns([
            pl.when(pl.col("shooting_team") == pl.col("home_team"))
              .then(pl.col("home_skaters")).otherwise(pl.col("away_skaters"))
              .alias("skaters_for"),
            pl.when(pl.col("shooting_team") == pl.col("home_team"))
              .then(pl.col("away_skaters")).otherwise(pl.col("home_skaters"))
              .alias("skaters_against"),
        ])
        .filter(
            (pl.col("skaters_for") > pl.col("skaters_against"))
            & pl.col("skaters_for").is_in([5, 4])
            & pl.col("skaters_against").is_in([4, 3])
        )
    )
    if pp.is_empty():
        return pl.DataFrame(schema={
            "team": pl.Utf8, "pp_shots": pl.Int64, "pp_goals": pl.Int64,
            "pp_xg_total": pl.Float64, "pp_shot_distance_avg": pl.Float64,
        })

    agg = (
        pp.group_by("shooting_team")
        .agg([
            pl.len().alias("pp_shots"),
            pl.col("is_goal").sum().cast(pl.Int64).alias("pp_goals"),
            pl.col("x_goal").sum().alias("pp_xg_total"),
            pl.col("shot_distance").mean().alias("pp_shot_distance_avg"),
        ])
        .rename({"shooting_team": "team"})
    )
    return agg


def _pp_carry_pct(pbp_df: pl.DataFrame, team_lookup: dict[int, str]) -> pl.DataFrame:
    """% controlled zone-entries on PP, per team.

    Filters PBP zone_entry events where ``strength == 'pp'`` and the
    entering team is the PP team.  ``carry_in`` is the boolean from
    the parser.
    """
    required = {"event_type", "strength", "carry_in", "entering_team_id"}
    missing = required - set(pbp_df.columns)
    if missing:
        return pl.DataFrame(schema={"team": pl.Utf8, "pp_carry_pct": pl.Float64})

    ze = (
        pbp_df.filter(
            (pl.col("event_type") == "zone_entry")
            & (pl.col("strength") == "pp")
            & pl.col("entering_team_id").is_not_null()
            & pl.col("carry_in").is_not_null()
        )
        .with_columns(pl.col("entering_team_id").cast(pl.Int64))
        .group_by("entering_team_id")
        .agg([
            pl.col("carry_in").cast(pl.Int64).sum().alias("carry_count"),
            pl.len().alias("entry_count"),
        ])
        .with_columns(
            (pl.col("carry_count") / pl.col("entry_count")).alias("pp_carry_pct")
        )
    )
    if ze.is_empty():
        return pl.DataFrame(schema={"team": pl.Utf8, "pp_carry_pct": pl.Float64})

    rows: list[dict] = []
    for r in ze.iter_rows(named=True):
        tid = int(r["entering_team_id"] or 0)
        abbrev = team_lookup.get(tid)
        if abbrev is None:
            continue
        rows.append({"team": abbrev, "pp_carry_pct": float(r["pp_carry_pct"] or 0.0)})
    if not rows:
        return pl.DataFrame(schema={"team": pl.Utf8, "pp_carry_pct": pl.Float64})
    return pl.DataFrame(rows)


def _pp_unit_info(
    st_df:        pl.DataFrame,
    position_map: dict[int, str],
    name_lookup:  dict[int, str],
) -> pl.DataFrame:
    """Pull PP1 QB + PP TOI for every team in the ST deployment parquet.

    Returns columns: team, pp_toi_secs, pp_team_gp, pp1_qb_id,
    pp1_qb_name, pp1_qb_share.  ``pp1_qb_share`` = PP1 unit_toi_secs /
    team total PP TOI.
    """
    if st_df.is_empty():
        return pl.DataFrame(schema={
            "team": pl.Utf8, "pp_toi_secs": pl.Float64, "pp_team_gp": pl.Int64,
            "pp1_qb_id": pl.Int64, "pp1_qb_name": pl.Utf8, "pp1_qb_share": pl.Float64,
        })

    pp_only = st_df.filter(pl.col("unit_type").is_in(["PP1", "PP2"]))
    if pp_only.is_empty():
        return pl.DataFrame(schema={
            "team": pl.Utf8, "pp_toi_secs": pl.Float64, "pp_team_gp": pl.Int64,
            "pp1_qb_id": pl.Int64, "pp1_qb_name": pl.Utf8, "pp1_qb_share": pl.Float64,
        })

    rows: list[dict] = []
    for team, group in pp_only.group_by("team"):
        team_abbrev = team[0] if isinstance(team, tuple) else team
        pp_toi   = float(group["team_st_toi"].first() or 0.0)
        pp_gp    = int(group["team_st_gp"].first() or 0)

        pp1 = group.filter(pl.col("unit_type") == "PP1")
        qb_id   = -1
        qb_name = ""
        qb_share = 0.0
        if not pp1.is_empty():
            pp1_row = pp1.row(0, named=True)
            personnel = list(pp1_row["personnel"] or [])
            ds = [pid for pid in personnel if position_map.get(int(pid)) == "D"]
            if ds:
                qb_id = int(ds[0])
                qb_name = name_lookup.get(qb_id, "")
            unit_toi = float(pp1_row["unit_toi_secs"] or 0.0)
            qb_share = (unit_toi / pp_toi) if pp_toi > 0 else 0.0

        rows.append({
            "team":        team_abbrev,
            "pp_toi_secs": pp_toi,
            "pp_team_gp":  pp_gp,
            "pp1_qb_id":   qb_id,
            "pp1_qb_name": qb_name,
            "pp1_qb_share": qb_share,
        })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_pp_coordinator(
    pbp_df:        pl.DataFrame,
    shots_df:      pl.DataFrame,
    st_df:         pl.DataFrame,
    position_map:  dict[int, str],
    name_lookup:   dict[int, str],
    season:        int,
    team_lookup:   dict[int, str],
) -> pl.DataFrame:
    """Build per-team PP coordinator system signature rows."""
    if shots_df.is_empty():
        warnings.warn(
            "compute_pp_coordinator: empty shots_df — PP volume metrics will be zero.",
            DataMissingWarning,
            stacklevel=2,
        )
    if pbp_df.is_empty():
        warnings.warn(
            "compute_pp_coordinator: empty pbp_df — carry% will be missing.",
            DataMissingWarning,
            stacklevel=2,
        )

    shot_stats = _pp_shot_stats(shots_df)
    carry_df   = _pp_carry_pct(pbp_df, team_lookup)
    pp_info    = _pp_unit_info(st_df, position_map, name_lookup)

    if pp_info.is_empty():
        warnings.warn(
            "compute_pp_coordinator: ST deployment frame had no PP units — no rows produced.",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(schema=PP_COORDINATOR_SCHEMA)

    base = pp_info.join(shot_stats, on="team", how="left").join(
        carry_df, on="team", how="left"
    ).with_columns([
        pl.col("pp_shots").fill_null(0),
        pl.col("pp_goals").fill_null(0),
        pl.col("pp_xg_total").fill_null(0.0),
        pl.col("pp_shot_distance_avg").fill_null(0.0),
        # Leave pp_carry_pct as NaN when the parser hasn't populated
        # carry_in yet — the dashboard renders "—" for NaN, never 0.
        pl.col("pp_carry_pct").fill_null(float("nan")),
    ])

    pp_toi_min = (pl.col("pp_toi_secs") / 60.0).alias("pp_toi_min")
    df = base.with_columns([
        pl.when(pl.col("pp_toi_secs") > 0)
          .then(pl.col("pp_shots") / pl.col("pp_toi_secs") * 3600.0)
          .otherwise(0.0).alias("pp_shots_per_60"),
        pl.when(pl.col("pp_toi_secs") > 0)
          .then(pl.col("pp_xg_total") / pl.col("pp_toi_secs") * 3600.0)
          .otherwise(0.0).alias("pp_xg_per_60"),
        pl.when(pl.col("pp_toi_secs") > 0)
          .then(pl.col("pp_goals") / pl.col("pp_toi_secs") * 3600.0)
          .otherwise(0.0).alias("pp_goals_per_60"),
        pl.when(pl.col("pp_shots") > 0)
          .then(pl.col("pp_xg_total") / pl.col("pp_shots"))
          .otherwise(0.0).alias("pp_xg_per_shot"),
    ]).with_columns([
        pl.lit(int(season)).alias("season"),
        pl.lit("").alias("pp_coordinator"),
        pl.lit(MODEL_VERSION).alias("model_version"),
    ])

    for col, dtype in PP_COORDINATOR_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(PP_COORDINATOR_SCHEMA.keys())).sort("pp_xg_per_60", descending=True)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_pp_coordinator(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"pp_coordinator_{season}.parquet"
    df.write_parquet(path)
    return path


def read_pp_coordinator(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "pp_coordinator"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"pp_coordinator_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("pp_coordinator_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
