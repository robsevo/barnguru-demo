"""Penalty Tendency Model — Feature 4.6.

Per-team (and per-referee-crew when available) foul call rate and PP
opportunities per game baseline.

Data caveat (V1)
----------------
Our PBP ingester does not yet capture **referee identities** (the NHL
API exposes them under ``rosterSpots.officials`` but our parser drops
them today).  When/if that field is added, this model can be extended
to bucket by ref crew.  For now, we produce per-team baselines and tag
the parquet with `ref_dim = "team-only"` so downstream consumers can
warn rather than silently assume ref-conditioned rates.

Approach
--------
1. From PBP, extract every ``event_type == "penalty"`` event with
   ``duration_minutes`` (minors = 2, majors = 5, misconduct = 10).  We
   only count minors+majors as PP-generating calls.
2. For each game, identify the team that took the penalty (offender)
   and the team that received PP (opponent).
3. Aggregate per (team, season):
   - n_penalties_taken      (PIM events charged to this team)
   - n_pp_opportunities     (PP events earned by this team — penalties
                              against the opponent)
   - penalties_taken_per_game
   - pp_opps_per_game
   - pim_total              (cumulative penalty minutes)
4. Build a league baseline (means + stds) so per-team values can be
   z-scored.

Output: per-team-season summary + league row.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "penalty_tendency_v1"

PP_GENERATING_TYPES = {"MIN", "MAJ"}  # excludes misconducts that don't yield PP


class DataMissingWarning(UserWarning):
    """Raised when penalty data is absent or insufficient."""


# ---------------------------------------------------------------------------
# Output schema — one row per (team, season)
# ---------------------------------------------------------------------------

PENALTY_TENDENCY_SCHEMA: dict[str, pl.DataType] = {
    "team":                     pl.Utf8,
    "season":                   pl.Int64,
    "n_games":                  pl.Int64,
    "n_penalties_taken":        pl.Int64,
    "n_pp_opportunities":       pl.Int64,
    "pim_total":                pl.Int64,
    "penalties_taken_per_game": pl.Float64,
    "pp_opps_per_game":         pl.Float64,
    "pim_per_game":             pl.Float64,
    "ref_dim":                  pl.Utf8,    # "team-only" until officials are ingested
    "model_version":            pl.Utf8,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_penalty_tendency(
    pbp_df:      pl.DataFrame,
    season:      int,
    team_lookup: dict[int, str] | None = None,
) -> pl.DataFrame:
    """Compute per-team penalty + PP-opportunity baselines for the season.

    Args:
        pbp_df:       play-by-play DataFrame.
        season:       e.g. 2025.
        team_lookup:  team_id (Int64) → abbrev (str).  When provided,
                      only team_ids present in this map appear in the
                      output — exhibition team_ids from events like the
                      4 Nations Face-Off are filtered.

    Returns:
        Polars DataFrame matching PENALTY_TENDENCY_SCHEMA.
    """
    team_lookup = team_lookup or {}
    known_ids = set(team_lookup.keys()) if team_lookup else None

    if pbp_df.is_empty():
        return pl.DataFrame(schema=PENALTY_TENDENCY_SCHEMA)

    required = {"game_id", "event_type", "event_owner_team_id",
                "home_team_id", "away_team_id",
                "penalty_type", "duration_minutes"}
    missing = required - set(pbp_df.columns)
    if missing:
        raise ValueError(f"pbp_df missing: {sorted(missing)}")

    penalties = pbp_df.filter(
        (pl.col("event_type") == "penalty")
        & pl.col("penalty_type").is_in(list(PP_GENERATING_TYPES))
    ).with_columns([
        pl.col("event_owner_team_id").cast(pl.Int64),
        pl.col("home_team_id").cast(pl.Int64),
        pl.col("away_team_id").cast(pl.Int64),
        pl.col("duration_minutes").cast(pl.Int64),
    ])

    if penalties.is_empty():
        warnings.warn(
            "compute_penalty_tendency: no PP-generating penalties found in PBP — "
            "model output will be all-zero rows.",
            DataMissingWarning,
            stacklevel=2,
        )

    # n_games per team — every game contributes to home + away counts
    gp = (
        pbp_df.group_by("game_id")
        .agg([
            pl.col("home_team_id").first().alias("h"),
            pl.col("away_team_id").first().alias("a"),
        ])
    )
    games_per_team: dict[int, int] = {}
    for r in gp.iter_rows(named=True):
        for tid in (int(r["h"] or 0), int(r["a"] or 0)):
            games_per_team[tid] = games_per_team.get(tid, 0) + 1

    # Aggregate taken (charged to event_owner_team_id) and earned PP (opponent)
    taken_counts: dict[int, list[int]] = {}      # tid → [n_penalties, pim]
    earned_counts: dict[int, int]      = {}      # tid → n_pp_opportunities

    for r in penalties.iter_rows(named=True):
        owner   = int(r["event_owner_team_id"] or 0)
        home_id = int(r["home_team_id"] or 0)
        away_id = int(r["away_team_id"] or 0)
        dur     = int(r["duration_minutes"] or 0)
        if owner not in (home_id, away_id):
            continue
        opp = away_id if owner == home_id else home_id
        taken_counts.setdefault(owner, [0, 0])
        taken_counts[owner][0] += 1
        taken_counts[owner][1] += dur
        earned_counts[opp] = earned_counts.get(opp, 0) + 1

    all_team_ids = set(games_per_team.keys()) | set(taken_counts.keys()) | set(earned_counts.keys())
    # Drop exhibition / unmapped team_ids (e.g. 4 Nations event entries).
    if known_ids is not None:
        all_team_ids = {tid for tid in all_team_ids if tid in known_ids}
    rows: list[dict] = []
    for tid in sorted(all_team_ids):
        abbrev = team_lookup.get(tid, str(tid))
        n_games = games_per_team.get(tid, 0)
        n_taken, pim = taken_counts.get(tid, [0, 0])
        n_earned = earned_counts.get(tid, 0)
        rows.append({
            "team":                     abbrev,
            "season":                   int(season),
            "n_games":                  n_games,
            "n_penalties_taken":        n_taken,
            "n_pp_opportunities":       n_earned,
            "pim_total":                pim,
            "penalties_taken_per_game": n_taken / n_games if n_games else 0.0,
            "pp_opps_per_game":         n_earned / n_games if n_games else 0.0,
            "pim_per_game":             pim / n_games if n_games else 0.0,
            "ref_dim":                  "team-only",
            "model_version":            MODEL_VERSION,
        })

    if not rows:
        return pl.DataFrame(schema=PENALTY_TENDENCY_SCHEMA)

    df = pl.DataFrame(rows)
    for col, dtype in PENALTY_TENDENCY_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(PENALTY_TENDENCY_SCHEMA.keys())).sort("penalties_taken_per_game", descending=True)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_penalty_tendency(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"penalty_tendency_{season}.parquet"
    df.write_parquet(path)
    return path


def read_penalty_tendency(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "penalty_tendency"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"penalty_tendency_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("penalty_tendency_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
