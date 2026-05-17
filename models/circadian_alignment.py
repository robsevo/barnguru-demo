"""Circadian alignment scorer — Feature 3.5.

Per-team-per-game body-clock misalignment between game start time at the
venue and the team's own home-city clock. Captures the channel that
compounds 3.4 (TZ crossings) and 3.3 (travel miles): a Toronto team
playing a 7 PM PT game in LA feels it at 10 PM body time even though
the rink clock says 7. The model emits the signed delta in hours,
its absolute value, and the "deviation from prime 19:00 body time"
which is what actually feeds the Fatigue Index.

Inputs
------
``schedule_df`` (Polars) — same schema as Features 3.1–3.4, optionally with
an extra ISO-UTC ``start_time_utc`` column. If absent / null, the model
defaults the puck drop to 19:00 venue-local time on game_date and emits a
``DataMissingWarning`` — the honest fallback so the pipeline doesn't silently
degrade to fake start times.

Outputs
-------
One row per (team, game) with CIRCADIAN_SCHEMA columns::

    team                   — Utf8
    game_id                — Int64
    game_date              — Utf8
    is_home                — Boolean
    venue_team             — Utf8
    home_tz_offset         — Float64  team's home venue UTC offset (DST-aware)
    venue_tz_offset        — Float64  game venue UTC offset
    misalignment_hours     — Float64  signed body-clock delta:
                                      ``home_tz - venue_tz`` (0 for home games)
                                      Positive  = body clock ahead of rink clock
                                                  (team flew west)
                                      Negative  = body clock behind rink clock
                                                  (team flew east)
    abs_misalignment_hours — Float64  unsigned magnitude
    game_start_utc         — Utf8    ISO datetime puck-drop UTC ("" when default)
    body_clock_hours       — Float64  decimal hour [0, 24) on the team's home
                                      clock at puck drop
    venue_local_hours      — Float64  decimal hour [0, 24) on the venue clock
                                      at puck drop
    prime_window_deviation — Float64  |body_clock_hours − 19.0|; how far from
                                      the body's "prime hockey time" 7 PM home

The prime-time deviation is the single number that should feed the composite
Fatigue Index (3.17). Combining a 22:00 body time with three back-to-backs is
materially worse than a 19:00 body time with the same schedule load.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from models.rapm_model import DataMissingWarning
from models.schedule_utils import explode_to_team_games, validate_schedule
from models.time_zone_crossing import NHL_VENUE_TZ, venue_utc_offset_hours


MODEL_VERSION = "circadian_alignment_v1"


CIRCADIAN_SCHEMA: dict[str, pl.DataType] = {
    "team":                   pl.Utf8,
    "game_id":                pl.Int64,
    "game_date":              pl.Utf8,
    "is_home":                pl.Boolean,
    "venue_team":             pl.Utf8,
    "home_tz_offset":         pl.Float64,
    "venue_tz_offset":        pl.Float64,
    "misalignment_hours":     pl.Float64,
    "abs_misalignment_hours": pl.Float64,
    "game_start_utc":         pl.Utf8,
    "body_clock_hours":       pl.Float64,
    "venue_local_hours":      pl.Float64,
    "prime_window_deviation": pl.Float64,
}


DEFAULT_LOCAL_PUCKDROP_HOUR = 19  # 7 PM venue-local → typical NHL slot
PRIME_BODY_CLOCK_HOUR       = 19.0  # body's "prime hockey window"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in CIRCADIAN_SCHEMA.items()}
    )


def _parse_utc(s: str) -> datetime | None:
    """Parse an ISO-UTC string into an aware ``datetime`` in UTC."""
    if s is None or s == "":
        return None
    raw = s.strip()
    # Accept trailing 'Z' as well as ±HH:MM.
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        # NHL API timestamps without TZ are conventionally UTC.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _default_puckdrop_utc(
    venue_team: str,
    on_date: date,
    tz_map: dict[str, str],
    hour_local: int = DEFAULT_LOCAL_PUCKDROP_HOUR,
) -> datetime:
    """Return a 19:00-venue-local default puck-drop time as a UTC datetime."""
    tz = ZoneInfo(tz_map[venue_team])
    naive = datetime(on_date.year, on_date.month, on_date.day, hour_local, 0)
    return naive.replace(tzinfo=tz).astimezone(timezone.utc)


def _decimal_hour(dt_utc: datetime, offset_hours: float) -> float:
    """Return the decimal local hour [0, 24) given a UTC dt and a UTC offset."""
    local = dt_utc + timedelta(hours=offset_hours)
    return (local.hour + local.minute / 60.0 + local.second / 3600.0) % 24.0


# ---------------------------------------------------------------------------
# CircadianAlignmentScorer
# ---------------------------------------------------------------------------

class CircadianAlignmentScorer:
    """Circadian alignment scorer (Feature 3.5). Stateless transformer."""

    def __init__(self, tz_map: dict[str, str] | None = None) -> None:
        self._tz = dict(tz_map) if tz_map is not None else dict(NHL_VENUE_TZ)
        self._version = MODEL_VERSION

    # ------------------------------------------------------------------
    def known_teams(self) -> set[str]:
        return set(self._tz.keys())

    def compute(self, schedule_df: pl.DataFrame) -> pl.DataFrame:
        """Return one row per (team, game) with circadian alignment features."""
        if not validate_schedule(schedule_df):
            return _empty_output()
        if len(schedule_df) == 0:
            return _empty_output()

        has_start = "start_time_utc" in schedule_df.columns
        team_games = explode_to_team_games(schedule_df)

        if has_start:
            start_map = {
                int(row["game_id"]): row["start_time_utc"]
                for row in schedule_df.select(["game_id", "start_time_utc"]).to_dicts()
            }
        else:
            start_map = {}

        unknown_teams:  set[str] = set()
        defaulted_games: set[int] = set()

        rows_out: list[dict] = []
        for row in team_games.to_dicts():
            team = row["team"]
            venue = team if row["is_home"] else row["opponent"]
            d = date.fromisoformat(row["game_date"])

            # Venue / home TZ offsets at evening of game_date — DST-aware.
            try:
                venue_off = venue_utc_offset_hours(venue, d, self._tz)
            except KeyError:
                unknown_teams.add(venue)
                venue_off = 0.0
            try:
                home_off = venue_utc_offset_hours(team, d, self._tz)
            except KeyError:
                unknown_teams.add(team)
                home_off = 0.0

            misalign = home_off - venue_off
            abs_misalign = abs(misalign)

            # Resolve puck drop UTC: real value > default > fail-loud.
            game_start: datetime | None = None
            if has_start:
                game_start = _parse_utc(start_map.get(int(row["game_id"]), ""))
            if game_start is None:
                if venue in self._tz:
                    game_start = _default_puckdrop_utc(venue, d, self._tz)
                    defaulted_games.add(int(row["game_id"]))
                    start_iso = ""
                else:
                    # No TZ for venue and no override → bail with zeros.
                    rows_out.append(
                        dict(
                            team=team,
                            game_id=int(row["game_id"]),
                            game_date=row["game_date"],
                            is_home=bool(row["is_home"]),
                            venue_team=venue,
                            home_tz_offset=home_off,
                            venue_tz_offset=venue_off,
                            misalignment_hours=misalign,
                            abs_misalignment_hours=abs_misalign,
                            game_start_utc="",
                            body_clock_hours=0.0,
                            venue_local_hours=0.0,
                            prime_window_deviation=0.0,
                        )
                    )
                    continue
            else:
                start_iso = game_start.isoformat()

            body_clock   = _decimal_hour(game_start, home_off)
            venue_local  = _decimal_hour(game_start, venue_off)
            # Wrap the deviation so 23:00 → 19:00 = 4h, not 4h-vs-24h confusion.
            raw_delta = body_clock - PRIME_BODY_CLOCK_HOUR
            wrapped   = abs(((raw_delta + 12.0) % 24.0) - 12.0)

            rows_out.append(
                dict(
                    team=team,
                    game_id=int(row["game_id"]),
                    game_date=row["game_date"],
                    is_home=bool(row["is_home"]),
                    venue_team=venue,
                    home_tz_offset=home_off,
                    venue_tz_offset=venue_off,
                    misalignment_hours=misalign,
                    abs_misalignment_hours=abs_misalign,
                    game_start_utc=start_iso,
                    body_clock_hours=body_clock,
                    venue_local_hours=venue_local,
                    prime_window_deviation=wrapped,
                )
            )

        if unknown_teams:
            warnings.warn(
                f"Unknown teams in schedule (treated as UTC offset 0): "
                f"{sorted(unknown_teams)}",
                DataMissingWarning,
                stacklevel=2,
            )
        if defaulted_games:
            n = len(defaulted_games)
            if not has_start:
                detail = "schedule_df missing 'start_time_utc' column"
            else:
                detail = f"'start_time_utc' null for {n} game(s)"
            warnings.warn(
                f"{detail} — defaulted {n} game(s) to 19:00 venue-local. "
                f"Run sync to populate real start times.",
                DataMissingWarning,
                stacklevel=2,
            )

        if not rows_out:
            return _empty_output()

        result = pl.DataFrame(rows_out)
        # Cast to the declared schema and order columns deterministically.
        for col, dtype in CIRCADIAN_SCHEMA.items():
            if col not in result.columns:
                result = result.with_columns(pl.lit(None).cast(dtype).alias(col))
            else:
                result = result.with_columns(pl.col(col).cast(dtype))
        return result.select(list(CIRCADIAN_SCHEMA.keys())).sort(
            ["team", "game_date", "game_id"]
        )

    def compute_team(self, schedule_df: pl.DataFrame, team: str) -> pl.DataFrame:
        return self.compute(schedule_df).filter(pl.col("team") == team)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_circadian_alignment(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write circadian DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"circadian_alignment_{as_of_date}.parquet"
    for col, dtype in CIRCADIAN_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(CIRCADIAN_SCHEMA.keys())).write_parquet(path)
    return path
