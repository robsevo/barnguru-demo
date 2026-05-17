"""Tests for Feature 3.3 — TravelDistanceCalculator."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from models.rapm_model import DataMissingWarning
from models.travel_distance import (
    NHL_CITY_COORDS,
    TRAVEL_DISTANCE_SCHEMA,
    TravelDistanceCalculator,
    city_distance,
    haversine_miles,
    write_travel_distance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sched(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "game_id":   0,
        "game_date": "2026-03-01",
        "home_team": "EDM",
        "away_team": "TOR",
    }
    filled = [{**defaults, **r} for r in rows]
    for i, r in enumerate(filled):
        if r.get("game_id") in (None, 0):
            r["game_id"] = 1_000 + i
    if not filled:
        return pl.DataFrame(
            schema={
                "game_id":   pl.Int64,
                "game_date": pl.Utf8,
                "home_team": pl.Utf8,
                "away_team": pl.Utf8,
            }
        )
    return pl.DataFrame(filled)


# ---------------------------------------------------------------------------
# Coord table sanity
# ---------------------------------------------------------------------------

class TestCityCoords:
    def test_all_32_current_teams_present(self):
        # Current NHL teams (post-2024 Utah move). ARI is legacy-only.
        current = {
            "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
            "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
            "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
            "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
        }
        missing = current - set(NHL_CITY_COORDS.keys())
        assert missing == set(), f"Missing NHL teams: {missing}"

    def test_legacy_ari_present_for_pre_2024_schedules(self):
        assert "ARI" in NHL_CITY_COORDS

    def test_coords_within_north_america(self):
        for team, (lat, lon) in NHL_CITY_COORDS.items():
            assert 24 < lat < 60, f"{team} lat looks wrong: {lat}"
            assert -125 < lon < -65, f"{team} lon looks wrong: {lon}"


# ---------------------------------------------------------------------------
# Haversine + city_distance
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_same_point_is_zero(self):
        p = NHL_CITY_COORDS["EDM"]
        assert haversine_miles(p, p) == pytest.approx(0.0, abs=0.001)

    def test_known_distance_edm_to_van(self):
        # Edmonton ↔ Vancouver is ~510 miles great-circle.
        d = haversine_miles(NHL_CITY_COORDS["EDM"], NHL_CITY_COORDS["VAN"])
        assert 480 < d < 540

    def test_symmetric(self):
        a = NHL_CITY_COORDS["NYR"]
        b = NHL_CITY_COORDS["LAK"]
        assert haversine_miles(a, b) == pytest.approx(haversine_miles(b, a))

    def test_city_distance_same_team_zero(self):
        assert city_distance("EDM", "EDM") == pytest.approx(0.0)

    def test_city_distance_unknown_team_raises(self):
        with pytest.raises(KeyError):
            city_distance("ZZZ", "EDM")


# ---------------------------------------------------------------------------
# Validation / empty
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_columns_warns(self):
        bad = pl.DataFrame({"foo": ["bar"]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = TravelDistanceCalculator().compute(bad)
        assert len(result) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_returns_empty_schema(self):
        result = TravelDistanceCalculator().compute(_sched([]))
        for col in TRAVEL_DISTANCE_SCHEMA:
            assert col in result.columns
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Travel computation
# ---------------------------------------------------------------------------

class TestTravelComputation:
    def test_first_game_zero_miles(self):
        df = _sched([{"home_team": "EDM", "away_team": "TOR", "game_date": "2026-03-01"}])
        result = TravelDistanceCalculator().compute(df)
        # Both teams' first game → miles_to_game = 0.
        for row in result.to_dicts():
            assert row["miles_to_game"] == pytest.approx(0.0)

    def test_visiting_team_travels_to_venue(self):
        # TOR plays at EDM, then back at TOR. TOR is visiting first.
        rows = [
            {"game_date": "2026-03-01", "home_team": "EDM", "away_team": "TOR"},
            {"game_date": "2026-03-05", "home_team": "TOR", "away_team": "EDM"},
        ]
        result = TravelDistanceCalculator().compute(_sched(rows))
        tor = result.filter(pl.col("team") == "TOR").sort("game_date").to_dicts()
        # First game (in EDM): no prior, miles_to_game = 0.
        assert tor[0]["miles_to_game"] == pytest.approx(0.0)
        # Second game: from EDM back to TOR.
        expected = city_distance("EDM", "TOR")
        assert tor[1]["miles_to_game"] == pytest.approx(expected)
        # EDM plays first game at home, second game in TOR → travels EDM → TOR.
        edm = result.filter(pl.col("team") == "EDM").sort("game_date").to_dicts()
        assert edm[0]["miles_to_game"] == pytest.approx(0.0)
        assert edm[1]["miles_to_game"] == pytest.approx(expected)

    def test_back_to_back_home_zero_travel(self):
        # EDM hosts two games in a row → no travel between them.
        rows = [
            {"game_date": "2026-03-01", "home_team": "EDM", "away_team": "TOR"},
            {"game_date": "2026-03-03", "home_team": "EDM", "away_team": "VAN"},
        ]
        result = TravelDistanceCalculator().compute(_sched(rows))
        edm = result.filter(pl.col("team") == "EDM").sort("game_date").to_dicts()
        assert edm[1]["miles_to_game"] == pytest.approx(0.0)

    def test_rolling_7d_inclusive(self):
        # Three games on consecutive days; final 7d miles = sum of all three.
        rows = [
            {"game_date": "2026-03-01", "home_team": "EDM", "away_team": "TOR"},
            {"game_date": "2026-03-02", "home_team": "VAN", "away_team": "TOR"},
            {"game_date": "2026-03-03", "home_team": "CGY", "away_team": "TOR"},
        ]
        result = TravelDistanceCalculator().compute(_sched(rows))
        tor = result.filter(pl.col("team") == "TOR").sort("game_date").to_dicts()
        expected_total = tor[0]["miles_to_game"] + tor[1]["miles_to_game"] + tor[2]["miles_to_game"]
        assert tor[2]["miles_last_7d"] == pytest.approx(expected_total)

    def test_rolling_7d_window_drops_old_games(self):
        # Game 10 days before the second falls outside the 7d window.
        rows = [
            {"game_date": "2026-03-01", "home_team": "EDM", "away_team": "TOR"},
            {"game_date": "2026-03-11", "home_team": "VAN", "away_team": "TOR"},
        ]
        result = TravelDistanceCalculator().compute(_sched(rows))
        tor = result.filter(pl.col("team") == "TOR").sort("game_date").to_dicts()
        # On day 11, only the current game's miles are in the 7d window.
        assert tor[1]["miles_last_7d"] == pytest.approx(tor[1]["miles_to_game"])

    def test_rolling_14d_keeps_older_games(self):
        rows = [
            {"game_date": "2026-03-01", "home_team": "EDM", "away_team": "TOR"},
            {"game_date": "2026-03-11", "home_team": "VAN", "away_team": "TOR"},
        ]
        result = TravelDistanceCalculator().compute(_sched(rows))
        tor = result.filter(pl.col("team") == "TOR").sort("game_date").to_dicts()
        expected = tor[0]["miles_to_game"] + tor[1]["miles_to_game"]
        assert tor[1]["miles_last_14d"] == pytest.approx(expected)

    def test_rolling_7d_boundary_inclusive(self):
        # 6 days apart — within the 7-day inclusive window.
        rows = [
            {"game_date": "2026-03-01", "home_team": "EDM", "away_team": "TOR"},
            {"game_date": "2026-03-07", "home_team": "VAN", "away_team": "TOR"},
        ]
        result = TravelDistanceCalculator().compute(_sched(rows))
        tor = result.filter(pl.col("team") == "TOR").sort("game_date").to_dicts()
        expected = tor[0]["miles_to_game"] + tor[1]["miles_to_game"]
        assert tor[1]["miles_last_7d"] == pytest.approx(expected)

    def test_rolling_7d_boundary_exclusive_at_7(self):
        # 7 days apart — earliest game falls outside the 7-day window.
        rows = [
            {"game_date": "2026-03-01", "home_team": "EDM", "away_team": "TOR"},
            {"game_date": "2026-03-08", "home_team": "VAN", "away_team": "TOR"},
        ]
        result = TravelDistanceCalculator().compute(_sched(rows))
        tor = result.filter(pl.col("team") == "TOR").sort("game_date").to_dicts()
        assert tor[1]["miles_last_7d"] == pytest.approx(tor[1]["miles_to_game"])

    def test_rolling_14d_boundary_inclusive(self):
        rows = [
            {"game_date": "2026-03-01", "home_team": "EDM", "away_team": "TOR"},
            {"game_date": "2026-03-14", "home_team": "VAN", "away_team": "TOR"},  # 13 days apart
        ]
        result = TravelDistanceCalculator().compute(_sched(rows))
        tor = result.filter(pl.col("team") == "TOR").sort("game_date").to_dicts()
        expected = tor[0]["miles_to_game"] + tor[1]["miles_to_game"]
        assert tor[1]["miles_last_14d"] == pytest.approx(expected)

    def test_rolling_14d_boundary_exclusive_at_14(self):
        rows = [
            {"game_date": "2026-03-01", "home_team": "EDM", "away_team": "TOR"},
            {"game_date": "2026-03-15", "home_team": "VAN", "away_team": "TOR"},  # 14 days apart
        ]
        result = TravelDistanceCalculator().compute(_sched(rows))
        tor = result.filter(pl.col("team") == "TOR").sort("game_date").to_dicts()
        assert tor[1]["miles_last_14d"] == pytest.approx(tor[1]["miles_to_game"])

    def test_unknown_team_warns_and_does_not_crash(self):
        # Use a fake team code; calculator should warn and treat as 0 miles.
        rows = [
            {"game_date": "2026-03-01", "home_team": "EDM", "away_team": "TOR"},
            {"game_date": "2026-03-03", "home_team": "ZZZ", "away_team": "TOR"},
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = TravelDistanceCalculator().compute(_sched(rows))
        assert any(issubclass(x.category, DataMissingWarning) for x in w)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_output_columns(self):
        df = _sched([{"home_team": "EDM", "away_team": "TOR", "game_date": "2026-03-01"}])
        result = TravelDistanceCalculator().compute(df)
        assert set(result.columns) == set(TRAVEL_DISTANCE_SCHEMA.keys())

    def test_venue_team_set_correctly(self):
        rows = [
            {"game_date": "2026-03-01", "home_team": "EDM", "away_team": "TOR"},
        ]
        result = TravelDistanceCalculator().compute(_sched(rows))
        for row in result.to_dicts():
            assert row["venue_team"] == "EDM"  # game was played in EDM


# ---------------------------------------------------------------------------
# Custom coords / injection
# ---------------------------------------------------------------------------

class TestCustomCoords:
    def test_custom_coords_override(self):
        coords = {
            "A": (0.0, 0.0),
            "B": (0.0, 1.0),  # 1 degree of longitude at equator ≈ 69.17 mi
        }
        calc = TravelDistanceCalculator(city_coords=coords)
        rows = [
            {"game_date": "2026-03-01", "home_team": "A", "away_team": "B"},
            {"game_date": "2026-03-02", "home_team": "B", "away_team": "A"},
        ]
        result = calc.compute(_sched(rows))
        a = result.filter(pl.col("team") == "A").sort("game_date").to_dicts()
        # A starts at home, travels to B for game 2 → ~69 miles.
        assert a[1]["miles_to_game"] == pytest.approx(69.17, rel=0.01)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_write_travel_distance(self, tmp_path):
        df = _sched([{"home_team": "EDM", "away_team": "TOR", "game_date": "2026-03-01"}])
        result = TravelDistanceCalculator().compute(df)
        path = write_travel_distance(result, tmp_path, "2026-03-01")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in TRAVEL_DISTANCE_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        df = _sched([{"home_team": "EDM", "away_team": "TOR", "game_date": "2026-03-01"}])
        result = TravelDistanceCalculator().compute(df)
        path = write_travel_distance(result, tmp_path, "2026-04-30")
        assert "2026-04-30" in path.name
