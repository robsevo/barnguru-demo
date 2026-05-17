"""Schedule utilities shared across Phase 3 fatigue models (3.1, 3.2, 3.3).

A ``schedule_df`` carries one row per scheduled NHL game with columns:

    game_id     Int64    - NHL game id (or any unique int)
    game_date   Utf8     - ISO date string "YYYY-MM-DD"
    home_team   Utf8     - 3-letter team abbreviation (e.g. "EDM")
    away_team   Utf8     - 3-letter team abbreviation

``explode_to_team_games`` rotates this into a per-team-perspective view (two
rows per game, one per side) which every Phase 3 model needs.
"""

from __future__ import annotations

import warnings

import polars as pl

from models.rapm_model import DataMissingWarning


SCHEDULE_REQUIRED_COLS = ("game_id", "game_date", "home_team", "away_team")


def validate_schedule(schedule_df: pl.DataFrame) -> bool:
    """Return True iff schedule_df has the required columns.

    Emits a DataMissingWarning and returns False when columns are missing,
    matching the convention used elsewhere in the codebase (see
    ``models/roster_disruption.py``).
    """
    missing = [c for c in SCHEDULE_REQUIRED_COLS if c not in schedule_df.columns]
    if missing:
        warnings.warn(
            f"schedule_df missing columns: {missing}. Required: {list(SCHEDULE_REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


def explode_to_team_games(schedule_df: pl.DataFrame) -> pl.DataFrame:
    """Return one row per (team, game) with columns:

        team       Utf8    - the team whose perspective this row represents
        opponent   Utf8    - the other team
        game_id    Int64
        game_date  Utf8
        is_home    Boolean
    """
    if not validate_schedule(schedule_df):
        return pl.DataFrame(
            schema={
                "team": pl.Utf8,
                "opponent": pl.Utf8,
                "game_id": pl.Int64,
                "game_date": pl.Utf8,
                "is_home": pl.Boolean,
            }
        )

    home = schedule_df.select(
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.col("game_id").cast(pl.Int64),
        pl.col("game_date").cast(pl.Utf8),
        pl.lit(True).alias("is_home"),
    )
    away = schedule_df.select(
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.col("game_id").cast(pl.Int64),
        pl.col("game_date").cast(pl.Utf8),
        pl.lit(False).alias("is_home"),
    )
    return pl.concat([home, away]).sort(["team", "game_date", "game_id"])
