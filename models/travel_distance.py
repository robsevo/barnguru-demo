"""Travel distance calculator — Feature 3.3.

Per-team-per-game travel mileage from the previous game venue to the current
venue, plus rolling 7-day and 14-day mileage windows.

Distance uses the great-circle (haversine) formula between arena coordinates
of the 32 current NHL markets (legacy ARI included for pre-2024 schedules).
Result is in **statute miles** — the unit hockey ops uses for road-trip talk.

Conventions
-----------
- For a team's first game (no prior game in the schedule), travel = 0.
- For subsequent games, travel = distance between the previous game's venue
  city and the current game's venue city. A team going from a road game in
  COL back to a home game in EDM travels ``distance(COL, EDM)``.
- ``miles_last_7d`` includes the current game's travel. Window is 7 calendar
  days back (inclusive of current date).

Inputs / outputs match the conventions in Features 3.1 and 3.2.
"""

from __future__ import annotations

from datetime import date
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import polars as pl

from models.schedule_utils import explode_to_team_games, validate_schedule


MODEL_VERSION = "travel_distance_v1"

EARTH_RADIUS_MILES = 3958.7613


# ---------------------------------------------------------------------------
# NHL arena coordinates (latitude, longitude in decimal degrees)
# ---------------------------------------------------------------------------
# Source: arena addresses → Google Maps. Approximate to ~50 m which is well
# below the per-mile precision we need for fatigue modelling.

NHL_CITY_COORDS: dict[str, tuple[float, float]] = {
    "ANA": (33.8075, -117.8765),  # Honda Center, Anaheim
    "ARI": (33.4324, -111.9402),  # Mullett Arena, Tempe (legacy)
    "BOS": (42.3662,  -71.0621),  # TD Garden
    "BUF": (42.8750,  -78.8760),  # KeyBank Center
    "CAR": (35.8033,  -78.7220),  # Lenovo Center, Raleigh
    "CBJ": (39.9690,  -83.0061),  # Nationwide Arena, Columbus
    "CGY": (51.0374, -114.0519),  # Scotiabank Saddledome (legacy until Scotia Place)
    "CHI": (41.8807,  -87.6742),  # United Center
    "COL": (39.7487, -105.0077),  # Ball Arena, Denver
    "DAL": (32.7906,  -96.8103),  # American Airlines Center
    "DET": (42.3411,  -83.0552),  # Little Caesars Arena
    "EDM": (53.5469, -113.4978),  # Rogers Place
    "FLA": (26.1583,  -80.3256),  # Amerant Bank Arena, Sunrise
    "LAK": (34.0431, -118.2675),  # Crypto.com Arena, Los Angeles
    "MIN": (44.9447,  -93.1011),  # Xcel Energy Center, St. Paul
    "MTL": (45.4965,  -73.5694),  # Bell Centre
    "NJD": (40.7336,  -74.1711),  # Prudential Center, Newark
    "NSH": (36.1592,  -86.7785),  # Bridgestone Arena
    "NYI": (40.7242,  -73.5907),  # UBS Arena, Elmont
    "NYR": (40.7505,  -73.9934),  # Madison Square Garden
    "OTT": (45.2969,  -75.9281),  # Canadian Tire Centre
    "PHI": (39.9012,  -75.1719),  # Wells Fargo Center
    "PIT": (40.4395,  -79.9892),  # PPG Paints Arena
    "SEA": (47.6219, -122.3540),  # Climate Pledge Arena
    "SJS": (37.3328, -121.9012),  # SAP Center
    "STL": (38.6268,  -90.2026),  # Enterprise Center
    "TBL": (27.9427,  -82.4518),  # Amalie Arena
    "TOR": (43.6435,  -79.3791),  # Scotiabank Arena
    "UTA": (40.7683, -111.9011),  # Delta Center, Salt Lake City (Utah Mammoth)
    "VAN": (49.2778, -123.1089),  # Rogers Arena
    "VGK": (36.1029, -115.1782),  # T-Mobile Arena, Las Vegas
    "WPG": (49.8927,  -97.1432),  # Canada Life Centre
    "WSH": (38.8981,  -77.0209),  # Capital One Arena
}


TRAVEL_DISTANCE_SCHEMA: dict[str, pl.DataType] = {
    "team":           pl.Utf8,
    "game_id":        pl.Int64,
    "game_date":      pl.Utf8,
    "is_home":        pl.Boolean,
    "venue_team":     pl.Utf8,     # the team whose arena the game was played at
    "miles_to_game":  pl.Float64,  # distance from previous venue → this venue
    "miles_last_7d":  pl.Float64,
    "miles_last_14d": pl.Float64,
}


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def haversine_miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two (lat, lon) points, in statute miles."""
    lat1, lon1 = a
    lat2, lon2 = b
    rlat1, rlat2 = radians(lat1), radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    h = sin(dlat / 2.0) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_MILES * asin(sqrt(h))


def city_distance(
    team_a: str,
    team_b: str,
    coords: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Distance in miles between two teams' home arenas.

    Raises KeyError if either team is unknown — we'd rather fail loudly than
    silently treat an unknown team as 0 miles (real data policy).
    """
    table = coords if coords is not None else NHL_CITY_COORDS
    if team_a == team_b:
        return 0.0
    return haversine_miles(table[team_a], table[team_b])


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in TRAVEL_DISTANCE_SCHEMA.items()}
    )


def _compute_team_block(
    block: pl.DataFrame,
    coords: dict[str, tuple[float, float]],
    unknown: set[str],
) -> pl.DataFrame:
    """Annotate one team's chronological games with travel mileage."""
    rows = block.to_dicts()
    n = len(rows)
    if n == 0:
        return block.with_columns(
            pl.Series("venue_team",     [], dtype=pl.Utf8),
            pl.Series("miles_to_game",  [], dtype=pl.Float64),
            pl.Series("miles_last_7d",  [], dtype=pl.Float64),
            pl.Series("miles_last_14d", [], dtype=pl.Float64),
        )

    venue_teams:  list[str] = []
    miles_each:   list[float] = []
    dates:        list[date] = []

    for row in rows:
        venue = row["team"] if row["is_home"] else row["opponent"]
        venue_teams.append(venue)
        dates.append(date.fromisoformat(row["game_date"]))

    prev_venue: str | None = None
    for i, venue in enumerate(venue_teams):
        if prev_venue is None:
            miles_each.append(0.0)
        else:
            try:
                miles_each.append(city_distance(prev_venue, venue, coords))
            except KeyError as e:
                unknown.add(str(e).strip("'"))
                miles_each.append(0.0)
        prev_venue = venue

    # Rolling windows include the current game. Conventions match Feature 3.1:
    # "miles in last N days" = today + (N−1) prior calendar days = elapsed < N.
    # Two-pointer sliding sums keep the pass O(n) per team.
    miles_7d:  list[float] = [0.0] * n
    miles_14d: list[float] = [0.0] * n
    left_7 = left_14 = 0
    sum_7  = sum_14  = 0.0
    for i, d in enumerate(dates):
        sum_14 += miles_each[i]
        sum_7  += miles_each[i]
        while (d - dates[left_14]).days >= 14:
            sum_14 -= miles_each[left_14]
            left_14 += 1
        while (d - dates[left_7]).days >= 7:
            sum_7 -= miles_each[left_7]
            left_7 += 1
        miles_7d[i]  = sum_7
        miles_14d[i] = sum_14

    return block.with_columns(
        pl.Series("venue_team",     venue_teams, dtype=pl.Utf8),
        pl.Series("miles_to_game",  miles_each,  dtype=pl.Float64),
        pl.Series("miles_last_7d",  miles_7d,    dtype=pl.Float64),
        pl.Series("miles_last_14d", miles_14d,   dtype=pl.Float64),
    )


# ---------------------------------------------------------------------------
# TravelDistanceCalculator
# ---------------------------------------------------------------------------

class TravelDistanceCalculator:
    """Travel distance calculator (Feature 3.3). Stateless transformer."""

    def __init__(
        self,
        city_coords: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        # Allow injection for tests / custom maps.
        self._coords = dict(city_coords) if city_coords is not None else dict(NHL_CITY_COORDS)
        self._version = MODEL_VERSION

    # ------------------------------------------------------------------
    def known_teams(self) -> set[str]:
        return set(self._coords.keys())

    def distance(self, team_a: str, team_b: str) -> float:
        """City-pair distance using this calculator's coordinate table."""
        if team_a == team_b:
            return 0.0
        return haversine_miles(self._coords[team_a], self._coords[team_b])

    def compute(self, schedule_df: pl.DataFrame) -> pl.DataFrame:
        """Return one row per (team, game) with mileage features."""
        if not validate_schedule(schedule_df):
            return _empty_output()
        if len(schedule_df) == 0:
            return _empty_output()

        unknown: set[str] = set()
        team_games = explode_to_team_games(schedule_df)
        blocks = [
            _compute_team_block(group, self._coords, unknown)
            for _, group in team_games.group_by("team", maintain_order=True)
        ]

        if not blocks:
            return _empty_output()
        if unknown:
            import warnings
            from models.rapm_model import DataMissingWarning
            warnings.warn(
                f"Unknown teams in schedule (treated as 0 miles): {sorted(unknown)}",
                DataMissingWarning,
                stacklevel=2,
            )

        result = pl.concat(blocks)
        return result.select(list(TRAVEL_DISTANCE_SCHEMA.keys()))

    def compute_team(self, schedule_df: pl.DataFrame, team: str) -> pl.DataFrame:
        return self.compute(schedule_df).filter(pl.col("team") == team)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_travel_distance(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write travel-distance DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"travel_distance_{as_of_date}.parquet"
    for col, dtype in TRAVEL_DISTANCE_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(TRAVEL_DISTANCE_SCHEMA.keys())).write_parquet(path)
    return path
