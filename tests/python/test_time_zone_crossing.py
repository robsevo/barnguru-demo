"""Tests for Feature 3.4 — TimeZoneCrossingModel."""

from __future__ import annotations

import warnings
from datetime import date

import polars as pl
import pytest

from models.rapm_model import DataMissingWarning
from models.time_zone_crossing import (
    NHL_VENUE_TZ,
    TIME_ZONE_CROSSING_SCHEMA,
    TimeZoneCrossingModel,
    venue_utc_offset_hours,
    write_time_zone_crossing,
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
# Venue TZ map sanity
# ---------------------------------------------------------------------------

class TestVenueTzMap:
    def test_all_32_current_teams_present(self):
        current = {
            "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
            "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
            "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
            "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
        }
        assert current - set(NHL_VENUE_TZ.keys()) == set()

    def test_legacy_ari_present(self):
        assert "ARI" in NHL_VENUE_TZ


# ---------------------------------------------------------------------------
# Offset helper
# ---------------------------------------------------------------------------

class TestVenueOffsetHelper:
    def test_eastern_winter_is_minus_5(self):
        # January in Boston (EST) → UTC-5
        off = venue_utc_offset_hours("BOS", date(2026, 1, 15))
        assert off == pytest.approx(-5.0)

    def test_pacific_winter_is_minus_8(self):
        off = venue_utc_offset_hours("LAK", date(2026, 1, 15))
        assert off == pytest.approx(-8.0)

    def test_eastern_summer_is_minus_4(self):
        # July in NYC (EDT) → UTC-4
        off = venue_utc_offset_hours("NYR", date(2026, 7, 15))
        assert off == pytest.approx(-4.0)

    def test_phoenix_no_dst(self):
        # AZ stays on MST year-round → UTC-7 always.
        jan = venue_utc_offset_hours("ARI", date(2026, 1, 15))
        jul = venue_utc_offset_hours("ARI", date(2026, 7, 15))
        assert jan == pytest.approx(-7.0)
        assert jul == pytest.approx(-7.0)

    def test_unknown_team_raises(self):
        with pytest.raises(KeyError):
            venue_utc_offset_hours("ZZZ", date(2026, 3, 1))


# ---------------------------------------------------------------------------
# Validation / empty
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_columns_warns(self):
        bad = pl.DataFrame({"foo": ["bar"]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = TimeZoneCrossingModel().compute(bad)
        assert len(result) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_returns_empty_schema(self):
        result = TimeZoneCrossingModel().compute(_sched([]))
        for col in TIME_ZONE_CROSSING_SCHEMA:
            assert col in result.columns
        assert len(result) == 0

    def test_output_columns(self):
        df = _sched([{"home_team": "EDM", "away_team": "TOR", "game_date": "2026-03-01"}])
        result = TimeZoneCrossingModel().compute(df)
        assert set(result.columns) == set(TIME_ZONE_CROSSING_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Crossing logic
# ---------------------------------------------------------------------------

class TestCrossingLogic:
    def test_first_game_zero_crossings(self):
        df = _sched([{"home_team": "EDM", "away_team": "TOR", "game_date": "2026-01-15"}])
        result = TimeZoneCrossingModel().compute(df)
        for row in result.to_dicts():
            assert row["tz_crossed_from_prev"] == pytest.approx(0.0)
            assert row["direction"] == "none"
            assert row["abs_tz_crossed_48h"] == pytest.approx(0.0)

    def test_eastbound_after_westbound_trip(self):
        # TOR: home (ET), road in LAK (PT), back home.
        rows = [
            {"game_date": "2026-01-15", "home_team": "TOR", "away_team": "BOS"},
            {"game_date": "2026-01-17", "home_team": "LAK", "away_team": "TOR"},
            {"game_date": "2026-01-20", "home_team": "TOR", "away_team": "BOS"},
        ]
        result = TimeZoneCrossingModel().compute(_sched(rows))
        tor = result.filter(pl.col("team") == "TOR").sort("game_date").to_dicts()
        assert tor[0]["tz_crossed_from_prev"] == pytest.approx(0.0)
        # ET (-5) → PT (-8) = -3 (westward)
        assert tor[1]["tz_crossed_from_prev"] == pytest.approx(-3.0)
        assert tor[1]["direction"] == "west"
        # PT (-8) → ET (-5) = +3 (eastward)
        assert tor[2]["tz_crossed_from_prev"] == pytest.approx(3.0)
        assert tor[2]["direction"] == "east"

    def test_zero_crossing_same_tz(self):
        # NYR → BOS → NYR — all same TZ (ET) → no crossings.
        rows = [
            {"game_date": "2026-01-15", "home_team": "NYR", "away_team": "TOR"},
            {"game_date": "2026-01-17", "home_team": "BOS", "away_team": "NYR"},
            {"game_date": "2026-01-19", "home_team": "NYR", "away_team": "TOR"},
        ]
        result = TimeZoneCrossingModel().compute(_sched(rows))
        nyr = result.filter(pl.col("team") == "NYR").sort("game_date").to_dicts()
        for row in nyr:
            assert row["tz_crossed_from_prev"] == pytest.approx(0.0)
            assert row["direction"] == "none"

    def test_abs_48h_window_inclusive(self):
        # Three games in 48h: TOR → LAK → COL. Two crossings should sum.
        rows = [
            {"game_date": "2026-01-15", "home_team": "TOR", "away_team": "BOS"},
            {"game_date": "2026-01-16", "home_team": "LAK", "away_team": "TOR"},
            {"game_date": "2026-01-17", "home_team": "COL", "away_team": "TOR"},
        ]
        result = TimeZoneCrossingModel().compute(_sched(rows))
        tor = result.filter(pl.col("team") == "TOR").sort("game_date").to_dicts()
        # ET→PT = -3, PT→MT = +1 → abs sum = 4
        assert tor[2]["abs_tz_crossed_48h"] == pytest.approx(4.0)

    def test_abs_48h_window_drops_older(self):
        # First game then a 5-day gap → only the latest crossing counts.
        rows = [
            {"game_date": "2026-01-15", "home_team": "TOR", "away_team": "BOS"},
            {"game_date": "2026-01-16", "home_team": "LAK", "away_team": "TOR"},
            {"game_date": "2026-01-22", "home_team": "COL", "away_team": "TOR"},
        ]
        result = TimeZoneCrossingModel().compute(_sched(rows))
        tor = result.filter(pl.col("team") == "TOR").sort("game_date").to_dicts()
        # Last game: only the PT→MT crossing (+1) is in the 48h window.
        assert tor[2]["abs_tz_crossed_48h"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# DST handling
# ---------------------------------------------------------------------------

class TestDST:
    def test_phoenix_in_summer_vs_dst_neighbor(self):
        # ARI never observes DST. In summer, ARI (UTC-7) vs DEN/COL (UTC-6) =
        # crossing of +1 (eastward) when leaving ARI for COL.
        rows = [
            {"game_date": "2026-07-15", "home_team": "ARI", "away_team": "TOR"},
            {"game_date": "2026-07-17", "home_team": "COL", "away_team": "TOR"},
        ]
        result = TimeZoneCrossingModel().compute(_sched(rows))
        tor = result.filter(pl.col("team") == "TOR").sort("game_date").to_dicts()
        # ARI (-7) → COL summer DST (-6) = +1 eastward.
        assert tor[1]["tz_crossed_from_prev"] == pytest.approx(1.0)
        assert tor[1]["direction"] == "east"


# ---------------------------------------------------------------------------
# Multi-team independence
# ---------------------------------------------------------------------------

class TestMultiTeam:
    def test_both_teams_emit_rows(self):
        df = _sched([{"home_team": "EDM", "away_team": "TOR", "game_date": "2026-01-15"}])
        result = TimeZoneCrossingModel().compute(df)
        teams = set(result["team"].to_list())
        assert teams == {"EDM", "TOR"}

    def test_unknown_venue_warns_and_does_not_crash(self):
        rows = [
            {"game_date": "2026-01-15", "home_team": "EDM", "away_team": "TOR"},
            {"game_date": "2026-01-17", "home_team": "ZZZ", "away_team": "TOR"},
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = TimeZoneCrossingModel().compute(_sched(rows))
        assert any(issubclass(x.category, DataMissingWarning) for x in w)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_write_parquet(self, tmp_path):
        df = _sched([{"home_team": "EDM", "away_team": "TOR", "game_date": "2026-01-15"}])
        result = TimeZoneCrossingModel().compute(df)
        path = write_time_zone_crossing(result, tmp_path, "2026-01-15")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in TIME_ZONE_CROSSING_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        df = _sched([{"home_team": "EDM", "away_team": "TOR", "game_date": "2026-01-15"}])
        result = TimeZoneCrossingModel().compute(df)
        path = write_time_zone_crossing(result, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name
