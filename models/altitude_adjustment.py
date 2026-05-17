"""Altitude adjustment — Feature 3.6.

Per-team-per-game aerobic penalty applied when a visiting team's home rink
sits at meaningfully lower elevation than the game venue. Captures the
"breathing thinner air" effect: VO₂max drops by about 3% per 1,000 ft above
~3,000 ft of acclimatization. Calgary, Edmonton, Salt Lake City and Denver
are the only NHL venues where this materially bites visiting teams.

Inputs
------
``schedule_df`` (Polars) — same schema as Features 3.1–3.5::

    game_id     Int64
    game_date   Utf8
    home_team   Utf8
    away_team   Utf8

Outputs
-------
One row per (team, game) with ALTITUDE_SCHEMA columns::

    team                   — Utf8
    game_id                — Int64
    game_date              — Utf8
    is_home                — Boolean
    venue_team             — Utf8
    venue_elevation_ft     — Int64    elevation of the game venue
    home_elevation_ft      — Int64    elevation of this team's home rink
    elevation_delta_ft     — Int64    venue − home (positive when visiting up)
    is_high_altitude       — Boolean  venue ≥ HIGH_ALTITUDE_THRESHOLD_FT
    altitude_penalty       — Float64  fractional aerobic-capacity hit
                                      ∈ [0.0, MAX_PENALTY]; positive only
                                      when the visiting team's home is
                                      substantially lower than the venue

Conventions
-----------
- Home team always gets penalty = 0.0 (they're acclimatized).
- Visiting teams from comparably high cities (e.g. CGY at COL) take a much
  smaller hit than visitors from sea level (e.g. NYR at COL).
- Formula: penalty = clip( 0.03 × max(0, (venue − HIGH_THRESHOLD)) / 1000
                           × max(0, min(1, (HIGH_THRESHOLD − home_elev) / HIGH_THRESHOLD)),
                           0.0, MAX_PENALTY )
  where the second factor scales down the penalty linearly as the visiting
  team's home elevation approaches the threshold. A team whose home matches
  the venue elevation receives no penalty.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl

from models.rapm_model import DataMissingWarning
from models.schedule_utils import explode_to_team_games, validate_schedule


MODEL_VERSION = "altitude_adjustment_v1"


# ---------------------------------------------------------------------------
# Arena elevations (statute feet, rounded).
# ---------------------------------------------------------------------------
# These are deliberately coarse — the model only needs to distinguish "this
# venue is materially higher" from "essentially the same elevation." Source:
# arena addresses → published city elevations (rounded to nearest 50 ft).

NHL_ARENA_ELEVATION_FT: dict[str, int] = {
    "ANA":   50,
    "ARI": 1180,
    "BOS":   20,
    "BUF":  600,
    "CAR":  350,
    "CBJ":  850,
    "CGY": 3400,
    "CHI":  600,
    "COL": 5280,
    "DAL":  430,
    "DET":  600,
    "EDM": 2200,
    "FLA":   10,
    "LAK":  290,
    "MIN":  800,
    "MTL":   30,
    "NJD":   30,
    "NSH":  600,
    "NYI":   50,
    "NYR":   50,
    "OTT":  200,
    "PHI":   30,
    "PIT":  750,
    "SEA":  360,
    "SJS":   80,
    "STL":  470,
    "TBL":   50,
    "TOR":  250,
    "UTA": 4200,
    "VAN":   50,
    "VGK": 2000,
    "WPG":  760,
    "WSH":   30,
}


HIGH_ALTITUDE_THRESHOLD_FT = 3000.0     # below this, no aerobic penalty
VO2_DROP_PER_1000FT        = 0.03       # ~3% VO₂max per 1,000 ft above threshold
MAX_PENALTY                = 0.10       # hard ceiling on the modifier


ALTITUDE_SCHEMA: dict[str, pl.DataType] = {
    "team":                pl.Utf8,
    "game_id":             pl.Int64,
    "game_date":           pl.Utf8,
    "is_home":             pl.Boolean,
    "venue_team":          pl.Utf8,
    "venue_elevation_ft":  pl.Int64,
    "home_elevation_ft":   pl.Int64,
    "elevation_delta_ft":  pl.Int64,
    "is_high_altitude":    pl.Boolean,
    "altitude_penalty":    pl.Float64,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in ALTITUDE_SCHEMA.items()}
    )


def altitude_penalty(
    venue_elevation_ft: float,
    home_elevation_ft: float,
    is_home: bool,
    threshold_ft: float = HIGH_ALTITUDE_THRESHOLD_FT,
    drop_per_1000: float = VO2_DROP_PER_1000FT,
    max_penalty: float = MAX_PENALTY,
) -> float:
    """Compute the aerobic penalty for one team-game.

    Returns 0.0 for home teams and for any venue at or below ``threshold_ft``.
    Otherwise scales linearly with how much higher the venue is than the
    threshold AND with how much lower the team's home is than the threshold,
    capped at ``max_penalty``.
    """
    if is_home:
        return 0.0
    if venue_elevation_ft <= threshold_ft:
        return 0.0
    venue_excess = (venue_elevation_ft - threshold_ft) / 1000.0
    raw_penalty = drop_per_1000 * venue_excess
    # Acclimatization factor: a team whose home rink is already at/above the
    # threshold loses no aerobic capacity; sea-level teams take the full hit.
    acclim = max(0.0, min(1.0, (threshold_ft - home_elevation_ft) / threshold_ft))
    return max(0.0, min(max_penalty, raw_penalty * acclim))


# ---------------------------------------------------------------------------
# AltitudeAdjustmentModel
# ---------------------------------------------------------------------------

class AltitudeAdjustmentModel:
    """Altitude adjustment model (Feature 3.6). Stateless transformer."""

    def __init__(
        self,
        elevation_map: dict[str, int] | None = None,
        threshold_ft: float = HIGH_ALTITUDE_THRESHOLD_FT,
        drop_per_1000: float = VO2_DROP_PER_1000FT,
        max_penalty: float = MAX_PENALTY,
    ) -> None:
        self._elev = dict(elevation_map) if elevation_map is not None else dict(NHL_ARENA_ELEVATION_FT)
        self._threshold = float(threshold_ft)
        self._drop = float(drop_per_1000)
        self._max = float(max_penalty)
        self._version = MODEL_VERSION

    # ------------------------------------------------------------------
    def known_teams(self) -> set[str]:
        return set(self._elev.keys())

    def elevation(self, team: str) -> int:
        return self._elev[team]

    def compute(self, schedule_df: pl.DataFrame) -> pl.DataFrame:
        """Return one row per (team, game) with altitude features."""
        if not validate_schedule(schedule_df):
            return _empty_output()
        if len(schedule_df) == 0:
            return _empty_output()

        team_games = explode_to_team_games(schedule_df)
        unknown: set[str] = set()

        rows_out: list[dict] = []
        for row in team_games.to_dicts():
            team = row["team"]
            venue = team if row["is_home"] else row["opponent"]

            try:
                venue_elev = self._elev[venue]
            except KeyError:
                unknown.add(venue)
                venue_elev = 0
            try:
                home_elev = self._elev[team]
            except KeyError:
                unknown.add(team)
                home_elev = 0

            penalty = altitude_penalty(
                venue_elev,
                home_elev,
                bool(row["is_home"]),
                self._threshold,
                self._drop,
                self._max,
            )
            rows_out.append(
                dict(
                    team=team,
                    game_id=int(row["game_id"]),
                    game_date=row["game_date"],
                    is_home=bool(row["is_home"]),
                    venue_team=venue,
                    venue_elevation_ft=int(venue_elev),
                    home_elevation_ft=int(home_elev),
                    elevation_delta_ft=int(venue_elev - home_elev),
                    is_high_altitude=bool(venue_elev >= self._threshold),
                    altitude_penalty=float(penalty),
                )
            )

        if unknown:
            warnings.warn(
                f"Unknown teams in schedule (treated as elevation 0): "
                f"{sorted(unknown)}",
                DataMissingWarning,
                stacklevel=2,
            )
        if not rows_out:
            return _empty_output()

        result = pl.DataFrame(rows_out)
        for col, dtype in ALTITUDE_SCHEMA.items():
            if col not in result.columns:
                result = result.with_columns(pl.lit(None).cast(dtype).alias(col))
            else:
                result = result.with_columns(pl.col(col).cast(dtype))
        return result.select(list(ALTITUDE_SCHEMA.keys())).sort(
            ["team", "game_date", "game_id"]
        )

    def compute_team(self, schedule_df: pl.DataFrame, team: str) -> pl.DataFrame:
        return self.compute(schedule_df).filter(pl.col("team") == team)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_altitude_adjustment(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write altitude DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"altitude_adjustment_{as_of_date}.parquet"
    for col, dtype in ALTITUDE_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(ALTITUDE_SCHEMA.keys())).write_parquet(path)
    return path
