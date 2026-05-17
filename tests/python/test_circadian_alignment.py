"""Tests for Feature 3.5 — CircadianAlignmentScorer."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from models.circadian_alignment import (
    CIRCADIAN_SCHEMA,
    CircadianAlignmentScorer,
    write_circadian_alignment,
)
from models.rapm_model import DataMissingWarning


def _sched(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "game_id":   0,
        "game_date": "2026-01-15",
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
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_columns_warns(self):
        bad = pl.DataFrame({"foo": ["bar"]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = CircadianAlignmentScorer().compute(bad)
        assert len(result) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_returns_empty_schema(self):
        out = CircadianAlignmentScorer().compute(_sched([]))
        for col in CIRCADIAN_SCHEMA:
            assert col in out.columns
        assert len(out) == 0

    def test_output_schema(self):
        df = _sched([{"home_team": "EDM", "away_team": "TOR"}])
        out = CircadianAlignmentScorer().compute(df)
        assert set(out.columns) == set(CIRCADIAN_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Misalignment math
# ---------------------------------------------------------------------------

class TestMisalignment:
    def test_home_team_zero_misalignment(self):
        # EDM at home → home_tz == venue_tz → 0 misalignment.
        df = _sched([{"home_team": "EDM", "away_team": "TOR"}])
        out = CircadianAlignmentScorer().compute(df)
        edm = out.filter(pl.col("team") == "EDM").to_dicts()[0]
        assert edm["misalignment_hours"] == pytest.approx(0.0)
        assert edm["abs_misalignment_hours"] == pytest.approx(0.0)

    def test_tor_visiting_lak_west_flight(self):
        # TOR (ET, UTC-5) at LAK (PT, UTC-8) in January.
        # misalignment = home_tz - venue_tz = -5 - (-8) = +3 (body 3h ahead).
        df = _sched([{"home_team": "LAK", "away_team": "TOR", "game_date": "2026-01-15"}])
        out = CircadianAlignmentScorer().compute(df)
        tor = out.filter(pl.col("team") == "TOR").to_dicts()[0]
        assert tor["misalignment_hours"] == pytest.approx(3.0)
        assert tor["abs_misalignment_hours"] == pytest.approx(3.0)
        assert tor["is_home"] is False
        assert tor["venue_team"] == "LAK"

    def test_lak_visiting_tor_east_flight(self):
        # LAK at TOR → home_tz - venue_tz = -8 - (-5) = -3 (body behind).
        df = _sched([{"home_team": "TOR", "away_team": "LAK", "game_date": "2026-01-15"}])
        out = CircadianAlignmentScorer().compute(df)
        lak = out.filter(pl.col("team") == "LAK").to_dicts()[0]
        assert lak["misalignment_hours"] == pytest.approx(-3.0)
        assert lak["abs_misalignment_hours"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Body-clock hours / prime window deviation
# ---------------------------------------------------------------------------

class TestBodyClock:
    def test_explicit_start_time_uses_real_value(self):
        # 7 PM PT in January = 03:00 UTC next day.
        rows = [{
            "home_team": "LAK",
            "away_team": "TOR",
            "game_date": "2026-01-15",
            "start_time_utc": "2026-01-16T03:00:00Z",
        }]
        df = _sched(rows)
        out = CircadianAlignmentScorer().compute(df)
        tor = out.filter(pl.col("team") == "TOR").to_dicts()[0]
        # TOR body clock at puck drop: 03:00 UTC + (-5) = 22:00 (10pm ET).
        assert tor["body_clock_hours"] == pytest.approx(22.0)
        # Venue clock: 03:00 UTC + (-8) = 19:00 (7pm PT).
        assert tor["venue_local_hours"] == pytest.approx(19.0)
        # Prime deviation: |22 - 19| = 3h.
        assert tor["prime_window_deviation"] == pytest.approx(3.0)

    def test_per_row_null_start_time_warns(self):
        # Column exists but one row is null — must still emit DataMissingWarning
        # and not silently fall back to the 19:00 default.
        df = pl.DataFrame([
            {"game_id": 1, "game_date": "2026-01-15",
             "home_team": "LAK", "away_team": "TOR",
             "start_time_utc": "2026-01-16T03:00:00Z"},
            {"game_id": 2, "game_date": "2026-01-16",
             "home_team": "TOR", "away_team": "MTL",
             "start_time_utc": None},
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = CircadianAlignmentScorer().compute(df)
        # game 2 was defaulted; game 1 was not.
        assert any(
            issubclass(x.category, DataMissingWarning) and "null" in str(x.message)
            for x in w
        )
        g2 = out.filter(pl.col("game_id") == 2).to_dicts()
        for row in g2:
            assert row["game_start_utc"] == ""  # marker of defaulted path
        g1 = out.filter(pl.col("game_id") == 1).to_dicts()
        for row in g1:
            assert row["game_start_utc"] != ""

    def test_default_puck_drop_when_start_missing(self):
        df = _sched([{"home_team": "LAK", "away_team": "TOR", "game_date": "2026-01-15"}])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = CircadianAlignmentScorer().compute(df)
        # Default = 19:00 venue local → body clock 22:00 → deviation 3h.
        tor = out.filter(pl.col("team") == "TOR").to_dicts()[0]
        assert tor["venue_local_hours"] == pytest.approx(19.0)
        assert tor["body_clock_hours"] == pytest.approx(22.0)
        assert tor["prime_window_deviation"] == pytest.approx(3.0)
        # The fallback path emits a DataMissingWarning.
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_matinee_with_no_travel_still_deviates(self):
        # MTL home game at 13:00 local = body clock 13:00 → 6h from 19:00.
        rows = [{
            "home_team": "MTL",
            "away_team": "TOR",
            "game_date": "2026-01-15",
            "start_time_utc": "2026-01-15T18:00:00Z",  # 13:00 ET
        }]
        out = CircadianAlignmentScorer().compute(_sched(rows))
        mtl = out.filter(pl.col("team") == "MTL").to_dicts()[0]
        assert mtl["misalignment_hours"] == pytest.approx(0.0)  # home game
        assert mtl["body_clock_hours"] == pytest.approx(13.0)
        assert mtl["prime_window_deviation"] == pytest.approx(6.0)

    def test_wraparound_late_game(self):
        # If body clock = 02:00 (i.e. crazy late), deviation wraps the
        # shorter way around the 24h dial: |02 - 19| → min(17, 24-17) = 7.
        rows = [{
            "home_team": "TOR",
            "away_team": "VAN",
            "game_date": "2026-01-15",
            "start_time_utc": "2026-01-16T07:00:00Z",  # 02:00 ET
        }]
        out = CircadianAlignmentScorer().compute(_sched(rows))
        tor = out.filter(pl.col("team") == "TOR").to_dicts()[0]
        assert tor["body_clock_hours"] == pytest.approx(2.0)
        assert tor["prime_window_deviation"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# DST
# ---------------------------------------------------------------------------

class TestDST:
    def test_summer_offsets(self):
        # July: ET = UTC-4, PT = UTC-7. TOR at LAK → misalignment +3 still.
        df = _sched([{"home_team": "LAK", "away_team": "TOR", "game_date": "2026-07-15"}])
        out = CircadianAlignmentScorer().compute(df)
        tor = out.filter(pl.col("team") == "TOR").to_dicts()[0]
        assert tor["home_tz_offset"] == pytest.approx(-4.0)
        assert tor["venue_tz_offset"] == pytest.approx(-7.0)
        assert tor["misalignment_hours"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_parquet(self, tmp_path):
        df = _sched([{"home_team": "EDM", "away_team": "TOR"}])
        out = CircadianAlignmentScorer().compute(df)
        path = write_circadian_alignment(out, tmp_path, "2026-01-15")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in CIRCADIAN_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        df = _sched([{"home_team": "EDM", "away_team": "TOR"}])
        out = CircadianAlignmentScorer().compute(df)
        path = write_circadian_alignment(out, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name


# ---------------------------------------------------------------------------
# Multi-team
# ---------------------------------------------------------------------------

class TestMultiTeam:
    def test_both_teams_get_rows(self):
        df = _sched([{"home_team": "LAK", "away_team": "TOR"}])
        out = CircadianAlignmentScorer().compute(df)
        teams = set(out["team"].to_list())
        assert teams == {"LAK", "TOR"}

    def test_signs_are_opposite_per_side(self):
        # Visiting team gets +X, home team gets 0 (their venue is their home).
        df = _sched([{"home_team": "LAK", "away_team": "TOR"}])
        out = CircadianAlignmentScorer().compute(df)
        tor = out.filter(pl.col("team") == "TOR").to_dicts()[0]
        lak = out.filter(pl.col("team") == "LAK").to_dicts()[0]
        assert lak["misalignment_hours"] == pytest.approx(0.0)
        assert tor["misalignment_hours"] == pytest.approx(3.0)
