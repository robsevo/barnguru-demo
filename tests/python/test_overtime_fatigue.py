"""Tests for Feature 3.10 — OvertimeFatigueTracker."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from models.overtime_fatigue import (
    OT_FATIGUE_SCHEMA,
    OT_TOI_EQUIVALENT_SECS,
    ROLLING_WINDOW_DAYS,
    OvertimeFatigueTracker,
    write_overtime_fatigue,
)
from models.rapm_model import DataMissingWarning


def _ot_df(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "game_id":     0,
        "game_date":   "2026-03-01",
        "player_id":   1,
        "team_id":     10,
        "toi_ot_secs": 0,
        "played_ot":   False,
    }
    filled = [{**defaults, **r} for r in rows]
    for i, r in enumerate(filled):
        if r.get("game_id") in (None, 0):
            r["game_id"] = 1_000 + i
    if not filled:
        return pl.DataFrame(
            schema={
                "game_id":     pl.Int64,
                "game_date":   pl.Utf8,
                "player_id":   pl.Int64,
                "team_id":     pl.Int64,
                "toi_ot_secs": pl.Int64,
                "played_ot":   pl.Boolean,
            }
        )
    return pl.DataFrame(filled)


def _series(
    player_id: int,
    dates: list[str],
    played_ot: list[bool],
    toi_ot_secs: list[int] | None = None,
) -> pl.DataFrame:
    if toi_ot_secs is None:
        toi_ot_secs = [0] * len(dates)
    assert len(dates) == len(played_ot) == len(toi_ot_secs)
    return _ot_df([
        {
            "player_id":   player_id,
            "game_date":   d,
            "played_ot":   p,
            "toi_ot_secs": s,
        }
        for d, p, s in zip(dates, played_ot, toi_ot_secs)
    ])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_columns_warns(self):
        bad = pl.DataFrame({"foo": ["bar"]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = OvertimeFatigueTracker().compute(bad)
        assert len(result) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_returns_empty_schema(self):
        result = OvertimeFatigueTracker().compute(_ot_df([]))
        for col in OT_FATIGUE_SCHEMA:
            assert col in result.columns
        assert len(result) == 0

    def test_output_schema(self):
        df = _series(1, ["2026-03-01"], [True], [180])
        result = OvertimeFatigueTracker().compute(df)
        assert set(result.columns) == set(OT_FATIGUE_SCHEMA.keys())

    def test_negative_equiv_secs_rejected(self):
        with pytest.raises(ValueError):
            OvertimeFatigueTracker(equivalent_toi_secs=-1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_window_is_seven(self):
        assert ROLLING_WINDOW_DAYS == 7

    def test_equiv_in_five_to_seven_min_range(self):
        # Spec says "~5–7 min equivalent TOI per OT game".
        assert 5 * 60 <= OT_TOI_EQUIVALENT_SECS <= 7 * 60


# ---------------------------------------------------------------------------
# Single-game outputs
# ---------------------------------------------------------------------------

class TestSingleGame:
    def test_no_ot_zero_load(self):
        df = _series(1, ["2026-03-01"], [False], [0])
        row = OvertimeFatigueTracker().compute(df).to_dicts()[0]
        assert row["ot_games_7day"] == 0
        assert row["ot_secs_actual_7day"] == 0
        assert row["ot_load_equiv_secs"] == 0.0
        assert row["ot_fatigue_score"] == 0.0

    def test_single_ot_game_adds_equivalent(self):
        df = _series(1, ["2026-03-01"], [True], [240])
        row = OvertimeFatigueTracker().compute(df).to_dicts()[0]
        assert row["played_ot"] is True
        assert row["toi_ot_secs"] == 240
        assert row["ot_games_7day"] == 1
        assert row["ot_secs_actual_7day"] == 240
        assert row["ot_load_equiv_secs"] == pytest.approx(OT_TOI_EQUIVALENT_SECS)
        assert row["ot_fatigue_score"] == pytest.approx(OT_TOI_EQUIVALENT_SECS + 240)

    def test_team_ot_with_no_player_ot_ice_still_counts(self):
        # A 4th-liner who didn't take an OT shift still inherits the team-level
        # late-night / overlapping schedule burn.
        df = _series(1, ["2026-03-01"], [True], [0])
        row = OvertimeFatigueTracker().compute(df).to_dicts()[0]
        assert row["ot_games_7day"] == 1
        assert row["ot_secs_actual_7day"] == 0
        assert row["ot_load_equiv_secs"] == pytest.approx(OT_TOI_EQUIVALENT_SECS)


# ---------------------------------------------------------------------------
# Rolling 7-day window
# ---------------------------------------------------------------------------

class TestRollingWindow:
    def test_seven_day_window_inclusive(self):
        # Three OT games spaced across exactly seven days — all should count
        # for the last row.
        dates = ["2026-03-01", "2026-03-04", "2026-03-07"]
        df = _series(1, dates, [True, True, True], [120, 180, 240])
        rows = OvertimeFatigueTracker().compute(df).sort("game_date").to_dicts()
        assert rows[0]["ot_games_7day"] == 1
        assert rows[1]["ot_games_7day"] == 2
        assert rows[2]["ot_games_7day"] == 3
        assert rows[2]["ot_secs_actual_7day"] == 120 + 180 + 240

    def test_window_drops_old_games(self):
        # Day 1 OT, day 9 OT — day 9's window does NOT see day 1.
        dates = ["2026-03-01", "2026-03-09"]
        df = _series(1, dates, [True, True], [120, 120])
        rows = OvertimeFatigueTracker().compute(df).sort("game_date").to_dicts()
        assert rows[1]["ot_games_7day"] == 1
        assert rows[1]["ot_secs_actual_7day"] == 120

    def test_non_ot_games_dont_inflate_count(self):
        dates = ["2026-03-01", "2026-03-02", "2026-03-03"]
        df = _series(1, dates, [True, False, False], [120, 0, 0])
        rows = OvertimeFatigueTracker().compute(df).sort("game_date").to_dicts()
        for r in rows:
            assert r["ot_games_7day"] == 1
            assert r["ot_secs_actual_7day"] == 120
            assert r["ot_load_equiv_secs"] == pytest.approx(OT_TOI_EQUIVALENT_SECS)

    def test_equiv_load_scales_linearly_with_count(self):
        dates = ["2026-03-01", "2026-03-02", "2026-03-03"]
        df = _series(1, dates, [True, True, True], [0, 0, 0])
        rows = OvertimeFatigueTracker().compute(df).sort("game_date").to_dicts()
        assert rows[2]["ot_load_equiv_secs"] == pytest.approx(3 * OT_TOI_EQUIVALENT_SECS)


# ---------------------------------------------------------------------------
# Multi-player independence
# ---------------------------------------------------------------------------

class TestMultiPlayer:
    def test_players_independent(self):
        rows = [
            {"player_id": 1, "game_date": "2026-03-01", "played_ot": True,  "toi_ot_secs": 240},
            {"player_id": 1, "game_date": "2026-03-02", "played_ot": True,  "toi_ot_secs": 240},
            {"player_id": 2, "game_date": "2026-03-01", "played_ot": False, "toi_ot_secs": 0},
            {"player_id": 2, "game_date": "2026-03-02", "played_ot": False, "toi_ot_secs": 0},
        ]
        out = OvertimeFatigueTracker().compute(_ot_df(rows))
        p1 = out.filter(pl.col("player_id") == 1).sort("game_date").to_dicts()
        p2 = out.filter(pl.col("player_id") == 2).sort("game_date").to_dicts()
        assert p1[-1]["ot_games_7day"] == 2
        assert p1[-1]["ot_secs_actual_7day"] == 480
        assert p2[-1]["ot_games_7day"] == 0
        assert p2[-1]["ot_secs_actual_7day"] == 0


# ---------------------------------------------------------------------------
# Equivalent-secs override (audit knob)
# ---------------------------------------------------------------------------

class TestEquivalentOverride:
    def test_override_scales_load(self):
        df = _series(1, ["2026-03-01"], [True], [0])
        a = OvertimeFatigueTracker(equivalent_toi_secs=300).compute(df).to_dicts()[0]
        b = OvertimeFatigueTracker(equivalent_toi_secs=420).compute(df).to_dicts()[0]
        assert a["ot_load_equiv_secs"] == pytest.approx(300)
        assert b["ot_load_equiv_secs"] == pytest.approx(420)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_parquet(self, tmp_path):
        df = _series(1, ["2026-03-01"], [True], [180])
        out = OvertimeFatigueTracker().compute(df)
        path = write_overtime_fatigue(out, tmp_path, "2026-03-01")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in OT_FATIGUE_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        df = _series(1, ["2026-03-01"], [True], [180])
        out = OvertimeFatigueTracker().compute(df)
        path = write_overtime_fatigue(out, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name
