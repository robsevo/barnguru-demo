"""Schedule density calculator — Feature 3.1.

Per-team-per-game schedule-load flags that feed the composite Fatigue Index
(3.17). Computed strictly from the schedule (no lookahead).

Inputs
------
``schedule_df`` (Polars) — one row per scheduled NHL game::

    game_id     Int64
    game_date   Utf8    ("YYYY-MM-DD")
    home_team   Utf8    (3-letter abbreviation, e.g. "EDM")
    away_team   Utf8

Outputs
-------
One row per (team, game) with the SCHEDULE_DENSITY_SCHEMA columns:

    team             — Utf8
    game_id          — Int64
    game_date        — Utf8
    is_home          — Boolean
    is_b2b           — Boolean   game played on consecutive calendar days
    is_3_in_4        — Boolean   three games (incl. current) in last 4 days
    rest_days        — Int64     calendar days since previous game (−1 if first)
    games_last_7d    — Int64     count of games incl. current in 7-day window
    games_last_14d   — Int64     count of games incl. current in 14-day window
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from models.schedule_utils import explode_to_team_games, validate_schedule


MODEL_VERSION = "schedule_density_v1"

SCHEDULE_DENSITY_SCHEMA: dict[str, pl.DataType] = {
    "team":           pl.Utf8,
    "game_id":        pl.Int64,
    "game_date":      pl.Utf8,
    "is_home":        pl.Boolean,
    "is_b2b":         pl.Boolean,
    "is_3_in_4":      pl.Boolean,
    "rest_days":      pl.Int64,
    "games_last_7d":  pl.Int64,
    "games_last_14d": pl.Int64,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in SCHEDULE_DENSITY_SCHEMA.items()}
    )


def _compute_team_block(block: pl.DataFrame) -> pl.DataFrame:
    """Compute density features for one team's chronologically-sorted games.

    Uses three left-pointers for sliding windows so the whole pass is O(n).
    Convention: "last N days" means today + (N−1) prior calendar days, so a
    game is in the window iff ``(today − game_date).days < N``. The 3-in-4
    flag uses a 4-day inclusive window (``elapsed <= 3``).
    """
    if len(block) == 0:
        return block

    dates = [date.fromisoformat(d) for d in block["game_date"].to_list()]
    n = len(dates)

    rest_days:      list[int] = [-1] * n
    is_b2b:         list[bool] = [False] * n
    is_3_in_4:      list[bool] = [False] * n
    games_last_7d:  list[int] = [0] * n
    games_last_14d: list[int] = [0] * n

    left_7 = left_14 = left_4 = 0
    for i, d in enumerate(dates):
        if i > 0:
            delta = (d - dates[i - 1]).days
            rest_days[i] = delta
            is_b2b[i] = delta == 1

        while (d - dates[left_14]).days >= 14:
            left_14 += 1
        while (d - dates[left_7]).days >= 7:
            left_7 += 1
        while (d - dates[left_4]).days > 3:
            left_4 += 1

        games_last_14d[i] = i - left_14 + 1
        games_last_7d[i]  = i - left_7  + 1
        is_3_in_4[i]      = (i - left_4 + 1) >= 3

    return block.with_columns(
        pl.Series("rest_days",      rest_days,      dtype=pl.Int64),
        pl.Series("is_b2b",         is_b2b,         dtype=pl.Boolean),
        pl.Series("is_3_in_4",      is_3_in_4,      dtype=pl.Boolean),
        pl.Series("games_last_7d",  games_last_7d,  dtype=pl.Int64),
        pl.Series("games_last_14d", games_last_14d, dtype=pl.Int64),
    )


# ---------------------------------------------------------------------------
# ScheduleDensityModel
# ---------------------------------------------------------------------------

class ScheduleDensityModel:
    """Schedule density calculator (Feature 3.1).

    Stateless transformer. Constructor takes no parameters; call ``compute``
    or ``compute_team`` with a schedule DataFrame.
    """

    def __init__(self) -> None:
        self._version = MODEL_VERSION

    # ------------------------------------------------------------------
    def compute(self, schedule_df: pl.DataFrame) -> pl.DataFrame:
        """Return one row per (team, game) with density features.

        Empty schedule or missing columns → empty result matching schema
        (warning emitted by ``validate_schedule``).
        """
        if not validate_schedule(schedule_df):
            return _empty_output()
        if len(schedule_df) == 0:
            return _empty_output()

        team_games = explode_to_team_games(schedule_df)

        blocks = [
            _compute_team_block(group)
            for _, group in team_games.group_by("team", maintain_order=True)
        ]
        if not blocks:
            return _empty_output()

        result = pl.concat(blocks)
        return result.select(list(SCHEDULE_DENSITY_SCHEMA.keys()))

    def compute_team(self, schedule_df: pl.DataFrame, team: str) -> pl.DataFrame:
        """Return density features for a single team only."""
        full = self.compute(schedule_df)
        return full.filter(pl.col("team") == team)

    def latest_for_team(
        self,
        schedule_df: pl.DataFrame,
        team: str,
        as_of_date: str | date | None = None,
    ) -> dict[str, Any] | None:
        """Return the density row for ``team``'s most recent game on/before
        ``as_of_date`` (default: latest game in the schedule)."""
        rows = self.compute_team(schedule_df, team)
        if len(rows) == 0:
            return None
        if as_of_date is not None:
            as_of = (
                as_of_date if isinstance(as_of_date, str)
                else as_of_date.isoformat()
            )
            rows = rows.filter(pl.col("game_date") <= as_of)
            if len(rows) == 0:
                return None
        return rows.sort(["game_date", "game_id"]).tail(1).to_dicts()[0]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_schedule_density(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write density DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"schedule_density_{as_of_date}.parquet"
    for col, dtype in SCHEDULE_DENSITY_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(SCHEDULE_DENSITY_SCHEMA.keys())).write_parquet(path)
    return path
