"""Road trip tracker — Feature 3.2.

Identifies contiguous road trips per team and assigns each road game its
position within the trip (game 1, 2, 3, 4+). Home games get sentinel values.

A "road trip" is defined as a maximal run of consecutive **away** games for a
team, in chronological order. A single away game flanked by home games is a
1-game trip. Returning home (the next home game) ends the trip.

Inputs
------
``schedule_df`` (Polars) — same schema as Feature 3.1::

    game_id, game_date, home_team, away_team

Outputs
-------
One row per (team, game) with the ROAD_TRIP_SCHEMA columns:

    team               — Utf8
    game_id            — Int64
    game_date          — Utf8
    is_home            — Boolean
    is_road            — Boolean
    trip_id            — Int64    sequential id of the trip this game belongs
                                  to (1-indexed); −1 for home games
    trip_game_number   — Int64    1-indexed position within the trip; 0 for home
    trip_total_length  — Int64    total games in the trip; 0 for home
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from models.schedule_utils import explode_to_team_games, validate_schedule


MODEL_VERSION = "road_trip_v1"

ROAD_TRIP_SCHEMA: dict[str, pl.DataType] = {
    "team":              pl.Utf8,
    "game_id":           pl.Int64,
    "game_date":         pl.Utf8,
    "is_home":           pl.Boolean,
    "is_road":           pl.Boolean,
    "trip_id":           pl.Int64,
    "trip_game_number":  pl.Int64,
    "trip_total_length": pl.Int64,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in ROAD_TRIP_SCHEMA.items()}
    )


def _annotate_team_block(block: pl.DataFrame) -> pl.DataFrame:
    """Assign trip_id / trip_game_number / trip_total_length per row."""
    is_home_list = block["is_home"].to_list()
    n = len(is_home_list)

    trip_id            = [-1] * n
    trip_game_number   = [0]  * n
    trip_total_length  = [0]  * n

    next_trip_id = 1
    i = 0
    while i < n:
        if is_home_list[i]:
            i += 1
            continue
        # Start of a road trip — find its end (last consecutive away game).
        j = i
        while j < n and not is_home_list[j]:
            j += 1
        trip_len = j - i
        for k in range(i, j):
            trip_id[k]           = next_trip_id
            trip_game_number[k]  = k - i + 1
            trip_total_length[k] = trip_len
        next_trip_id += 1
        i = j

    return block.with_columns(
        (~pl.col("is_home")).alias("is_road"),
        pl.Series("trip_id",           trip_id,           dtype=pl.Int64),
        pl.Series("trip_game_number",  trip_game_number,  dtype=pl.Int64),
        pl.Series("trip_total_length", trip_total_length, dtype=pl.Int64),
    )


# ---------------------------------------------------------------------------
# RoadTripTracker
# ---------------------------------------------------------------------------

class RoadTripTracker:
    """Road trip tracker (Feature 3.2). Stateless transformer."""

    def __init__(self) -> None:
        self._version = MODEL_VERSION

    # ------------------------------------------------------------------
    def compute(self, schedule_df: pl.DataFrame) -> pl.DataFrame:
        """Return one row per (team, game) with road-trip features."""
        if not validate_schedule(schedule_df):
            return _empty_output()
        if len(schedule_df) == 0:
            return _empty_output()

        team_games = explode_to_team_games(schedule_df)

        blocks = [
            _annotate_team_block(group)
            for _, group in team_games.group_by("team", maintain_order=True)
        ]
        if not blocks:
            return _empty_output()

        result = pl.concat(blocks)
        return result.select(list(ROAD_TRIP_SCHEMA.keys()))

    def compute_team(self, schedule_df: pl.DataFrame, team: str) -> pl.DataFrame:
        """Return trip rows for a single team."""
        return self.compute(schedule_df).filter(pl.col("team") == team)

    def current_trip(
        self,
        schedule_df: pl.DataFrame,
        team: str,
        as_of_date: str,
    ) -> dict | None:
        """Return the trip row for ``team``'s most recent road game on/before
        ``as_of_date``. Returns ``None`` if the team's last game on/before
        that date was a home game (i.e. no active road trip)."""
        rows = self.compute_team(schedule_df, team).filter(
            pl.col("game_date") <= as_of_date
        )
        if len(rows) == 0:
            return None
        last = rows.sort(["game_date", "game_id"]).tail(1).to_dicts()[0]
        if last["is_home"]:
            return None
        return last


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_road_trips(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write trip DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"road_trips_{as_of_date}.parquet"
    for col, dtype in ROAD_TRIP_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(ROAD_TRIP_SCHEMA.keys())).write_parquet(path)
    return path
