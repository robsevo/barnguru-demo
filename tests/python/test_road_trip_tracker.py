"""Tests for Feature 3.2 — RoadTripTracker."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from models.rapm_model import DataMissingWarning
from models.road_trip_tracker import (
    ROAD_TRIP_SCHEMA,
    RoadTripTracker,
    write_road_trips,
)


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


def _team_schedule(team: str, opp: str, dates: list[str], home: list[bool]) -> pl.DataFrame:
    rows = []
    for d, h in zip(dates, home):
        if h:
            rows.append({"game_date": d, "home_team": team, "away_team": opp})
        else:
            rows.append({"game_date": d, "home_team": opp, "away_team": team})
    return _sched(rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_columns_warns_zero(self):
        bad = pl.DataFrame({"foo": ["bar"]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = RoadTripTracker().compute(bad)
        assert len(result) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_returns_empty_schema(self):
        result = RoadTripTracker().compute(_sched([]))
        for col in ROAD_TRIP_SCHEMA:
            assert col in result.columns
        assert len(result) == 0

    def test_output_schema(self):
        df = _team_schedule("EDM", "TOR", ["2026-03-01"], [True])
        result = RoadTripTracker().compute(df)
        assert set(result.columns) == set(ROAD_TRIP_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Home-only / road-only patterns
# ---------------------------------------------------------------------------

class TestBasicPatterns:
    def test_all_home_games_have_no_trip(self):
        df = _team_schedule(
            "EDM", "TOR",
            ["2026-03-01", "2026-03-03", "2026-03-05"],
            [True, True, True],
        )
        result = RoadTripTracker().compute_team(df, "EDM")
        for row in result.to_dicts():
            assert row["trip_id"] == -1
            assert row["trip_game_number"] == 0
            assert row["trip_total_length"] == 0
            assert row["is_road"] is False

    def test_single_road_game_is_one_game_trip(self):
        df = _team_schedule(
            "EDM", "TOR",
            ["2026-03-01", "2026-03-03"],
            [True, False],
        )
        result = RoadTripTracker().compute_team(df, "EDM").sort("game_date")
        rows = result.to_dicts()
        assert rows[1]["is_road"] is True
        assert rows[1]["trip_id"] == 1
        assert rows[1]["trip_game_number"] == 1
        assert rows[1]["trip_total_length"] == 1

    def test_four_game_road_trip(self):
        df = _team_schedule(
            "EDM", "TOR",
            ["2026-03-01", "2026-03-03", "2026-03-05", "2026-03-07", "2026-03-09", "2026-03-11"],
            [True, False, False, False, False, True],  # 4 road games sandwiched
        )
        result = RoadTripTracker().compute_team(df, "EDM").sort("game_date")
        rows = result.to_dicts()
        # Indices 1..4 are the road trip
        for trip_pos, idx in enumerate(range(1, 5), start=1):
            assert rows[idx]["is_road"] is True
            assert rows[idx]["trip_id"] == 1
            assert rows[idx]["trip_game_number"] == trip_pos
            assert rows[idx]["trip_total_length"] == 4
        # Closing home game ends the trip
        assert rows[5]["trip_id"] == -1
        assert rows[5]["trip_total_length"] == 0

    def test_two_separate_trips_get_distinct_ids(self):
        df = _team_schedule(
            "EDM", "TOR",
            ["2026-03-01", "2026-03-03", "2026-03-05", "2026-03-07", "2026-03-09"],
            [False, True, False, False, True],
        )
        result = RoadTripTracker().compute_team(df, "EDM").sort("game_date")
        rows = result.to_dicts()
        # First road trip: a single game on Mar 1.
        assert rows[0]["trip_id"] == 1
        assert rows[0]["trip_total_length"] == 1
        # Second road trip: Mar 5 & Mar 7.
        assert rows[2]["trip_id"] == 2
        assert rows[2]["trip_game_number"] == 1
        assert rows[3]["trip_id"] == 2
        assert rows[3]["trip_game_number"] == 2
        assert rows[2]["trip_total_length"] == 2
        assert rows[3]["trip_total_length"] == 2


# ---------------------------------------------------------------------------
# Multi-team independence
# ---------------------------------------------------------------------------

class TestMultiTeam:
    def test_both_teams_get_rows(self):
        df = _sched([{"home_team": "EDM", "away_team": "TOR", "game_date": "2026-03-01"}])
        result = RoadTripTracker().compute(df)
        teams = set(result["team"].to_list())
        assert teams == {"EDM", "TOR"}

    def test_trips_indexed_per_team(self):
        # EDM hosts TOR (TOR on road); both should reflect their side.
        rows = [
            {"game_date": "2026-03-01", "home_team": "EDM", "away_team": "TOR"},
            {"game_date": "2026-03-03", "home_team": "EDM", "away_team": "TOR"},
        ]
        result = RoadTripTracker().compute(_sched(rows))
        # TOR plays two consecutive road games → one 2-game trip.
        tor = result.filter(pl.col("team") == "TOR").sort("game_date").to_dicts()
        assert tor[0]["trip_id"] == 1
        assert tor[0]["trip_game_number"] == 1
        assert tor[1]["trip_id"] == 1
        assert tor[1]["trip_game_number"] == 2
        assert tor[1]["trip_total_length"] == 2
        # EDM hosts both → no trip.
        edm = result.filter(pl.col("team") == "EDM").to_dicts()
        for row in edm:
            assert row["trip_id"] == -1


# ---------------------------------------------------------------------------
# current_trip
# ---------------------------------------------------------------------------

class TestCurrentTrip:
    def test_returns_active_trip_for_road_team(self):
        df = _team_schedule(
            "EDM", "TOR",
            ["2026-03-01", "2026-03-03", "2026-03-05"],
            [True, False, False],
        )
        current = RoadTripTracker().current_trip(df, "EDM", as_of_date="2026-03-05")
        assert current is not None
        assert current["is_road"] is True
        assert current["trip_game_number"] == 2

    def test_returns_none_when_last_game_is_home(self):
        df = _team_schedule(
            "EDM", "TOR",
            ["2026-03-01", "2026-03-03"],
            [False, True],  # closed out a trip on Mar 3
        )
        current = RoadTripTracker().current_trip(df, "EDM", as_of_date="2026-03-04")
        assert current is None

    def test_returns_none_when_no_games_yet(self):
        df = _team_schedule("EDM", "TOR", ["2026-03-10"], [False])
        current = RoadTripTracker().current_trip(df, "EDM", as_of_date="2026-03-01")
        assert current is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_parquet_with_schema(self, tmp_path):
        df = _team_schedule("EDM", "TOR", ["2026-03-01"], [False])
        result = RoadTripTracker().compute(df)
        path = write_road_trips(result, tmp_path, "2026-03-01")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in ROAD_TRIP_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        df = _team_schedule("EDM", "TOR", ["2026-03-01"], [False])
        result = RoadTripTracker().compute(df)
        path = write_road_trips(result, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name
