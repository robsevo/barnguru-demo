"""Time zone crossing model — Feature 3.4.

Per-team-per-game tracking of time-zone shifts between consecutive venues plus
a rolling 48-hour absolute zone-crossing total. Captures the "jet lag" channel
that compounds with travel mileage (3.3) and back-to-backs (3.1) when a team's
body clock is dragged east/west across multiple zones in a short window.

Inputs
------
``schedule_df`` (Polars) — same schema as Features 3.1–3.3::

    game_id     Int64
    game_date   Utf8    ("YYYY-MM-DD")
    home_team   Utf8
    away_team   Utf8

Outputs
-------
One row per (team, game) with TIME_ZONE_CROSSING_SCHEMA::

    team              — Utf8
    game_id           — Int64
    game_date         — Utf8
    is_home           — Boolean
    venue_team        — Utf8     team whose arena the game was played at
    venue_tz_offset   — Float64  UTC offset of the venue, in hours, for the
                                 *evening* of game_date (DST-aware)
    tz_crossed_from_prev  — Float64  signed delta from previous game's venue
                                     to this game's venue, in hours:
                                     positive = eastward; negative = westward;
                                     0.0 if first game or same TZ
    direction         — Utf8     "east" | "west" | "none"
    abs_tz_crossed_48h — Float64 sum of |zones crossed| over the last 48 hours
                                 inclusive of the current game

Conventions
-----------
- First game in the schedule for any team → zero crossings.
- DST is handled via IANA zones (``zoneinfo``). The offset is read at the
  game's *evening* local time (~19:00) on ``game_date`` — the moment the
  body-clock impact is felt — so spring-forward / fall-back transitions
  carry through correctly.
- An unknown venue team triggers a ``DataMissingWarning`` and gets treated
  as zero crossings (consistent with travel_distance behavior).
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from models.rapm_model import DataMissingWarning
from models.schedule_utils import explode_to_team_games, validate_schedule


MODEL_VERSION = "time_zone_crossing_v1"


# ---------------------------------------------------------------------------
# Venue IANA time zone map (current 32 NHL markets + legacy ARI).
# ---------------------------------------------------------------------------
# These are deliberately IANA strings — zoneinfo handles DST + historical
# transitions correctly without us hard-coding offsets.

NHL_VENUE_TZ: dict[str, str] = {
    "ANA": "America/Los_Angeles",
    "ARI": "America/Phoenix",        # AZ does not observe DST
    "BOS": "America/New_York",
    "BUF": "America/New_York",
    "CAR": "America/New_York",
    "CBJ": "America/New_York",
    "CGY": "America/Edmonton",       # MT, observes DST
    "CHI": "America/Chicago",
    "COL": "America/Denver",
    "DAL": "America/Chicago",
    "DET": "America/Detroit",
    "EDM": "America/Edmonton",
    "FLA": "America/New_York",
    "LAK": "America/Los_Angeles",
    "MIN": "America/Chicago",
    "MTL": "America/Montreal",
    "NJD": "America/New_York",
    "NSH": "America/Chicago",
    "NYI": "America/New_York",
    "NYR": "America/New_York",
    "OTT": "America/Toronto",
    "PHI": "America/New_York",
    "PIT": "America/New_York",
    "SEA": "America/Los_Angeles",
    "SJS": "America/Los_Angeles",
    "STL": "America/Chicago",
    "TBL": "America/New_York",
    "TOR": "America/Toronto",
    "UTA": "America/Denver",         # Salt Lake City
    "VAN": "America/Vancouver",
    "VGK": "America/Los_Angeles",
    "WPG": "America/Winnipeg",
    "WSH": "America/New_York",
}


TIME_ZONE_CROSSING_SCHEMA: dict[str, pl.DataType] = {
    "team":                 pl.Utf8,
    "game_id":              pl.Int64,
    "game_date":            pl.Utf8,
    "is_home":              pl.Boolean,
    "venue_team":           pl.Utf8,
    "venue_tz_offset":      pl.Float64,
    "tz_crossed_from_prev": pl.Float64,
    "direction":            pl.Utf8,
    "abs_tz_crossed_48h":   pl.Float64,
}


# ---------------------------------------------------------------------------
# Offset helpers
# ---------------------------------------------------------------------------

_EVENING_LOCAL_HOUR = 19  # 7 PM local — typical NHL puck drop.


def venue_utc_offset_hours(
    team: str,
    on_date: date,
    tz_map: dict[str, str] | None = None,
    hour_local: int = _EVENING_LOCAL_HOUR,
) -> float:
    """UTC offset (in hours) for a venue at the evening of ``on_date``.

    Raises KeyError if the team is unknown. We resolve the offset at evening
    local time so DST-aware transitions on the day of the game carry the
    correct value into the model.
    """
    zones = tz_map if tz_map is not None else NHL_VENUE_TZ
    tz = ZoneInfo(zones[team])
    naive = datetime(on_date.year, on_date.month, on_date.day, hour_local, 0)
    aware = naive.replace(tzinfo=tz)
    off = aware.utcoffset()
    if off is None:
        return 0.0
    return off.total_seconds() / 3600.0


def _direction(delta_hours: float) -> str:
    if delta_hours > 0:
        return "east"
    if delta_hours < 0:
        return "west"
    return "none"


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in TIME_ZONE_CROSSING_SCHEMA.items()}
    )


# ---------------------------------------------------------------------------
# Core per-team annotation
# ---------------------------------------------------------------------------

def _annotate_team_block(
    block: pl.DataFrame,
    tz_map: dict[str, str],
    unknown: set[str],
) -> pl.DataFrame:
    """Annotate one team's chronological games with TZ crossing features."""
    rows = block.to_dicts()
    n = len(rows)
    if n == 0:
        return block.with_columns(
            pl.Series("venue_team",           [], dtype=pl.Utf8),
            pl.Series("venue_tz_offset",      [], dtype=pl.Float64),
            pl.Series("tz_crossed_from_prev", [], dtype=pl.Float64),
            pl.Series("direction",            [], dtype=pl.Utf8),
            pl.Series("abs_tz_crossed_48h",   [], dtype=pl.Float64),
        )

    venue_teams:    list[str]   = []
    dates:          list[date]  = []
    offsets:        list[float] = []
    crossings:      list[float] = []
    directions:     list[str]   = []

    for row in rows:
        venue = row["team"] if row["is_home"] else row["opponent"]
        venue_teams.append(venue)
        d = date.fromisoformat(row["game_date"])
        dates.append(d)
        try:
            offsets.append(venue_utc_offset_hours(venue, d, tz_map))
        except KeyError:
            unknown.add(venue)
            offsets.append(0.0)

    prev_off: float | None = None
    for off in offsets:
        if prev_off is None:
            crossings.append(0.0)
            directions.append("none")
        else:
            # Positive offset shift = moving east (towards UTC+0 from
            # the negative-offset North American side). LA (-8) → NY (-5)
            # = +3 hours = eastward. We treat that as a positive number
            # so "+ direction" matches the intuitive east-bound flight.
            delta = off - prev_off
            crossings.append(delta)
            directions.append(_direction(delta))
        prev_off = off

    # Rolling 48-hour absolute crossings (inclusive of current game).
    # Two-pointer on the sorted date list keeps the pass O(n).
    abs_48h: list[float] = [0.0] * n
    left = 0
    running = 0.0
    for i, d in enumerate(dates):
        running += abs(crossings[i])
        while (d - dates[left]).days > 2:  # > 48h strict-ish window
            running -= abs(crossings[left])
            left += 1
        abs_48h[i] = running

    return block.with_columns(
        pl.Series("venue_team",           venue_teams, dtype=pl.Utf8),
        pl.Series("venue_tz_offset",      offsets,     dtype=pl.Float64),
        pl.Series("tz_crossed_from_prev", crossings,   dtype=pl.Float64),
        pl.Series("direction",            directions,  dtype=pl.Utf8),
        pl.Series("abs_tz_crossed_48h",   abs_48h,     dtype=pl.Float64),
    )


# ---------------------------------------------------------------------------
# TimeZoneCrossingModel
# ---------------------------------------------------------------------------

class TimeZoneCrossingModel:
    """Time zone crossing model (Feature 3.4). Stateless transformer."""

    def __init__(self, tz_map: dict[str, str] | None = None) -> None:
        self._tz = dict(tz_map) if tz_map is not None else dict(NHL_VENUE_TZ)
        self._version = MODEL_VERSION

    # ------------------------------------------------------------------
    def known_teams(self) -> set[str]:
        return set(self._tz.keys())

    def venue_offset(self, team: str, on_date: date) -> float:
        return venue_utc_offset_hours(team, on_date, self._tz)

    def compute(self, schedule_df: pl.DataFrame) -> pl.DataFrame:
        """Return one row per (team, game) with time-zone crossing features."""
        if not validate_schedule(schedule_df):
            return _empty_output()
        if len(schedule_df) == 0:
            return _empty_output()

        unknown: set[str] = set()
        team_games = explode_to_team_games(schedule_df)

        blocks = [
            _annotate_team_block(group, self._tz, unknown)
            for _, group in team_games.group_by("team", maintain_order=True)
        ]
        if not blocks:
            return _empty_output()
        if unknown:
            warnings.warn(
                f"Unknown venue teams in schedule (treated as 0 TZ offset): "
                f"{sorted(unknown)}",
                DataMissingWarning,
                stacklevel=2,
            )

        result = pl.concat(blocks)
        return result.select(list(TIME_ZONE_CROSSING_SCHEMA.keys()))

    def compute_team(self, schedule_df: pl.DataFrame, team: str) -> pl.DataFrame:
        return self.compute(schedule_df).filter(pl.col("team") == team)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_time_zone_crossing(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write TZ-crossing DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"time_zone_crossing_{as_of_date}.parquet"
    for col, dtype in TIME_ZONE_CROSSING_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(TIME_ZONE_CROSSING_SCHEMA.keys())).write_parquet(path)
    return path
