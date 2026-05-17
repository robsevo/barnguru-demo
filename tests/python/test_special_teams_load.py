"""Tests for Feature 3.8 — SpecialTeamsLoadTracker."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from models.rapm_model import DataMissingWarning
from models.special_teams_load import (
    PK_INTENSITY_WEIGHT,
    PP_INTENSITY_WEIGHT,
    ROLLING_WINDOW_GAMES,
    ST_LOAD_SCHEMA,
    SpecialTeamsLoadTracker,
    write_special_teams_load,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _st_df(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "game_id":   0,
        "game_date": "2026-03-01",
        "player_id": 1,
        "team_id":   10,
        "pptoi_secs": 0,
        "shtoi_secs": 0,
    }
    filled = [{**defaults, **r} for r in rows]
    for i, r in enumerate(filled):
        if r.get("game_id") in (None, 0):
            r["game_id"] = 1_000 + i
    if not filled:
        return pl.DataFrame(
            schema={
                "game_id":    pl.Int64,
                "game_date":  pl.Utf8,
                "player_id":  pl.Int64,
                "team_id":    pl.Int64,
                "pptoi_secs": pl.Int64,
                "shtoi_secs": pl.Int64,
            }
        )
    return pl.DataFrame(filled)


def _player_series(
    player_id: int,
    dates: list[str],
    pp_secs: list[int],
    pk_secs: list[int],
) -> pl.DataFrame:
    assert len(dates) == len(pp_secs) == len(pk_secs)
    return _st_df([
        {
            "player_id":  player_id,
            "game_date":  d,
            "pptoi_secs": p,
            "shtoi_secs": k,
        }
        for d, p, k in zip(dates, pp_secs, pk_secs)
    ])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_columns_warns(self):
        bad = pl.DataFrame({"foo": ["bar"]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = SpecialTeamsLoadTracker().compute(bad)
        assert len(result) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_returns_empty_schema(self):
        result = SpecialTeamsLoadTracker().compute(_st_df([]))
        for col in ST_LOAD_SCHEMA:
            assert col in result.columns
        assert len(result) == 0

    def test_output_schema(self):
        df = _player_series(1, ["2026-03-01"], [120], [60])
        result = SpecialTeamsLoadTracker().compute(df)
        assert set(result.columns) == set(ST_LOAD_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Rolling-window math
# ---------------------------------------------------------------------------

class TestRollingWindow:
    def test_first_game_sums_to_current(self):
        out = SpecialTeamsLoadTracker().compute(
            _player_series(1, ["2026-03-01"], [120], [60])
        ).to_dicts()[0]
        assert out["pptoi_5game_sum_secs"] == 120
        assert out["shtoi_5game_sum_secs"] == 60
        assert out["st_total_5game_sum_secs"] == 180

    def test_five_game_window_inclusive(self):
        # 6 consecutive games at 60s PP each → 5game sum is 300 from G5 onward.
        dates = [f"2026-03-{i:02d}" for i in range(1, 7)]
        pp = [60] * 6
        pk = [0]  * 6
        out = SpecialTeamsLoadTracker().compute(
            _player_series(1, dates, pp, pk)
        ).sort("game_date").to_dicts()
        assert out[4]["pptoi_5game_sum_secs"] == 300
        # Game 6: drops Game 1 → still 5×60 = 300.
        assert out[5]["pptoi_5game_sum_secs"] == 300

    def test_running_sum_grows_until_window_full(self):
        dates = [f"2026-03-{i:02d}" for i in range(1, 5)]
        out = SpecialTeamsLoadTracker().compute(
            _player_series(1, dates, [60, 60, 60, 60], [0, 0, 0, 0])
        ).sort("game_date").to_dicts()
        assert [r["pptoi_5game_sum_secs"] for r in out] == [60, 120, 180, 240]


# ---------------------------------------------------------------------------
# Intensity score
# ---------------------------------------------------------------------------

class TestIntensityScore:
    def test_pure_pp_uses_pp_weight(self):
        out = SpecialTeamsLoadTracker().compute(
            _player_series(1, ["2026-03-01"], [120], [0])
        ).to_dicts()[0]
        assert out["st_intensity_score"] == pytest.approx(120 * PP_INTENSITY_WEIGHT)

    def test_pure_pk_uses_pk_weight(self):
        out = SpecialTeamsLoadTracker().compute(
            _player_series(1, ["2026-03-01"], [0], [120])
        ).to_dicts()[0]
        assert out["st_intensity_score"] == pytest.approx(120 * PK_INTENSITY_WEIGHT)

    def test_pk_weighted_higher_than_pp(self):
        # Same minutes in PP vs PK → PK score should be higher.
        out = SpecialTeamsLoadTracker().compute(
            _player_series(
                1,
                ["2026-03-01", "2026-03-02"],
                [120, 0],
                [0, 120],
            )
        ).sort("game_date").to_dicts()
        # Game 1: pure PP. Game 2: pure PK. Both within the 5-game window.
        # After game 2: 120 PP + 120 PK in window → PP×1.5 + PK×1.8.
        assert out[1]["st_intensity_score"] > out[0]["st_intensity_score"]

    def test_intensity_weights_overridable(self):
        tracker = SpecialTeamsLoadTracker(pp_intensity_weight=2.0, pk_intensity_weight=3.0)
        out = tracker.compute(
            _player_series(1, ["2026-03-01"], [100], [100])
        ).to_dicts()[0]
        # 100×2.0 + 100×3.0 = 500.
        assert out["st_intensity_score"] == pytest.approx(500.0)

    def test_zero_minutes_zero_score(self):
        out = SpecialTeamsLoadTracker().compute(
            _player_series(1, ["2026-03-01"], [0], [0])
        ).to_dicts()[0]
        assert out["st_intensity_score"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Multi-player independence
# ---------------------------------------------------------------------------

class TestMultiPlayer:
    def test_players_indexed_independently(self):
        rows = []
        for d, p, k in zip(["2026-03-01", "2026-03-02"], [120, 120], [0, 0]):
            rows.append({"player_id": 1, "game_date": d, "pptoi_secs": p, "shtoi_secs": k})
        for d, p, k in zip(["2026-03-01", "2026-03-02"], [0, 0], [120, 120]):
            rows.append({"player_id": 2, "game_date": d, "pptoi_secs": p, "shtoi_secs": k})
        out = SpecialTeamsLoadTracker().compute(_st_df(rows))
        p1 = out.filter(pl.col("player_id") == 1).sort("game_date").to_dicts()
        p2 = out.filter(pl.col("player_id") == 2).sort("game_date").to_dicts()
        assert p1[-1]["pptoi_5game_sum_secs"] == 240
        assert p1[-1]["shtoi_5game_sum_secs"] == 0
        assert p2[-1]["pptoi_5game_sum_secs"] == 0
        assert p2[-1]["shtoi_5game_sum_secs"] == 240


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_parquet(self, tmp_path):
        df = _player_series(1, ["2026-03-01"], [120], [60])
        out = SpecialTeamsLoadTracker().compute(df)
        path = write_special_teams_load(out, tmp_path, "2026-03-01")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in ST_LOAD_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        df = _player_series(1, ["2026-03-01"], [120], [60])
        out = SpecialTeamsLoadTracker().compute(df)
        path = write_special_teams_load(out, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_window_is_five(self):
        assert ROLLING_WINDOW_GAMES == 5

    def test_pk_weight_exceeds_pp_weight(self):
        assert PK_INTENSITY_WEIGHT > PP_INTENSITY_WEIGHT
