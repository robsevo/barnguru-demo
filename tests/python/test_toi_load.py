"""Tests for Feature 3.7 — TOILoadTracker."""

from __future__ import annotations

import math
import warnings

import polars as pl
import pytest

from models.rapm_model import DataMissingWarning
from models.toi_load import (
    ROLLING_WINDOW_GAMES,
    SPIKE_MIN_DELTA_SECS,
    SPIKE_Z_THRESHOLD,
    TOI_LOAD_SCHEMA,
    TOILoadTracker,
    write_toi_load,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _toi_df(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "game_id":   0,
        "game_date": "2026-03-01",
        "player_id": 1,
        "team_id":   10,
        "toi_total_secs": 0,
    }
    filled = [{**defaults, **r} for r in rows]
    for i, r in enumerate(filled):
        if r.get("game_id") in (None, 0):
            r["game_id"] = 1_000 + i
    if not filled:
        return pl.DataFrame(
            schema={
                "game_id":        pl.Int64,
                "game_date":      pl.Utf8,
                "player_id":      pl.Int64,
                "team_id":        pl.Int64,
                "toi_total_secs": pl.Int64,
            }
        )
    return pl.DataFrame(filled)


def _player_series(
    player_id: int,
    dates: list[str],
    toi_secs: list[int],
    team_id: int = 10,
) -> pl.DataFrame:
    assert len(dates) == len(toi_secs)
    rows = [
        {
            "player_id": player_id,
            "team_id":   team_id,
            "game_date": d,
            "toi_total_secs": t,
        }
        for d, t in zip(dates, toi_secs)
    ]
    return _toi_df(rows)


# ---------------------------------------------------------------------------
# Validation / empty
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_columns_warns(self):
        bad = pl.DataFrame({"foo": ["bar"]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = TOILoadTracker().compute(bad)
        assert len(result) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_returns_empty_schema(self):
        result = TOILoadTracker().compute(_toi_df([]))
        for col in TOI_LOAD_SCHEMA:
            assert col in result.columns
        assert len(result) == 0

    def test_output_schema(self):
        df = _player_series(1, ["2026-03-01"], [1000])
        result = TOILoadTracker().compute(df)
        assert set(result.columns) == set(TOI_LOAD_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Single-player behavior
# ---------------------------------------------------------------------------

class TestSinglePlayer:
    def test_first_game_baseline(self):
        # First game: rolling avg = current TOI; season avg = current TOI;
        # delta and z are 0; no spike.
        df = _player_series(1, ["2026-03-01"], [1200])
        out = TOILoadTracker().compute(df)
        row = out.to_dicts()[0]
        assert row["toi_secs"] == 1200
        assert row["games_played_to_date"] == 1
        assert row["toi_5game_avg_secs"] == pytest.approx(1200.0)
        assert row["toi_season_avg_secs"] == pytest.approx(1200.0)
        assert row["toi_spike_delta_secs"] == pytest.approx(0.0)
        assert row["toi_spike_z"] == pytest.approx(0.0)
        assert row["is_toi_spike"] is False

    def test_five_game_window_inclusive(self):
        dates = [f"2026-03-{i:02d}" for i in range(1, 7)]
        toi = [1000, 1100, 1200, 1300, 1400, 1500]
        out = TOILoadTracker().compute(_player_series(1, dates, toi)).sort("game_date")
        rows = out.to_dicts()
        # Game 5: window covers games 1–5 (inclusive), mean = 1200.
        assert rows[4]["toi_5game_avg_secs"] == pytest.approx(1200.0)
        # Game 6: window drops game 1 → covers 2–6, mean = 1300.
        assert rows[5]["toi_5game_avg_secs"] == pytest.approx(1300.0)

    def test_season_mean_is_running(self):
        toi = [600, 800, 1000]
        out = TOILoadTracker().compute(
            _player_series(1, ["2026-03-01", "2026-03-02", "2026-03-03"], toi)
        ).sort("game_date")
        rows = out.to_dicts()
        assert rows[0]["toi_season_avg_secs"] == pytest.approx(600.0)
        assert rows[1]["toi_season_avg_secs"] == pytest.approx(700.0)
        assert rows[2]["toi_season_avg_secs"] == pytest.approx(800.0)

    def test_delta_sign(self):
        toi = [1000, 1000, 1500]  # last game is well above the 1166 mean
        out = TOILoadTracker().compute(
            _player_series(1, ["2026-03-01", "2026-03-02", "2026-03-03"], toi)
        ).sort("game_date")
        last = out.to_dicts()[-1]
        assert last["toi_spike_delta_secs"] > 0
        assert last["toi_spike_z"] > 0

    def test_spike_requires_both_z_and_min_delta(self):
        # Player has tight low variance, jumps by 30s only — z may be high but
        # delta is below the 60s floor → no spike.
        toi = [600, 600, 600, 600, 600, 630]
        out = TOILoadTracker().compute(
            _player_series(
                1,
                [f"2026-03-{i:02d}" for i in range(1, 7)],
                toi,
            )
        ).sort("game_date")
        last = out.to_dicts()[-1]
        # z must be high (variance is zero before the bump → noise threshold)
        # but delta_secs = 30 → below floor.
        assert last["toi_spike_delta_secs"] == pytest.approx(25.0)
        assert last["is_toi_spike"] is False

    def test_spike_fires_on_real_jump(self):
        # Stable 18min (1080s) for 5 games, then 28min (1680s).
        toi = [1080] * 5 + [1680]
        out = TOILoadTracker().compute(
            _player_series(
                1,
                [f"2026-03-{i:02d}" for i in range(1, 7)],
                toi,
            )
        ).sort("game_date")
        last = out.to_dicts()[-1]
        assert last["toi_spike_delta_secs"] > SPIKE_MIN_DELTA_SECS
        assert last["toi_spike_z"] >= SPIKE_Z_THRESHOLD
        assert last["is_toi_spike"] is True

    def test_constant_toi_no_spike_no_nan(self):
        toi = [1000] * 7
        out = TOILoadTracker().compute(
            _player_series(
                1,
                [f"2026-03-{i:02d}" for i in range(1, 8)],
                toi,
            )
        ).sort("game_date")
        for row in out.to_dicts():
            assert row["toi_spike_z"] == pytest.approx(0.0)
            assert row["is_toi_spike"] is False
            assert not math.isnan(row["toi_5game_avg_secs"])


# ---------------------------------------------------------------------------
# Multi-player independence
# ---------------------------------------------------------------------------

class TestMultiPlayer:
    def test_players_indexed_independently(self):
        # Two players with totally different TOI profiles must not leak into
        # each other's rolling statistics.
        rows: list[dict] = []
        for d, t in zip(["2026-03-01", "2026-03-02"], [600, 600]):
            rows.append({"player_id": 1, "game_date": d, "toi_total_secs": t})
        for d, t in zip(["2026-03-01", "2026-03-02"], [1800, 1800]):
            rows.append({"player_id": 2, "game_date": d, "toi_total_secs": t})
        out = TOILoadTracker().compute(_toi_df(rows))
        p1 = out.filter(pl.col("player_id") == 1).sort("game_date").to_dicts()
        p2 = out.filter(pl.col("player_id") == 2).sort("game_date").to_dicts()
        assert p1[-1]["toi_season_avg_secs"] == pytest.approx(600.0)
        assert p2[-1]["toi_season_avg_secs"] == pytest.approx(1800.0)

    def test_games_played_per_player(self):
        rows: list[dict] = []
        for d in ["2026-03-01", "2026-03-03", "2026-03-05"]:
            rows.append({"player_id": 1, "game_date": d, "toi_total_secs": 1000})
        for d in ["2026-03-02"]:
            rows.append({"player_id": 2, "game_date": d, "toi_total_secs": 1000})
        out = TOILoadTracker().compute(_toi_df(rows))
        p1 = out.filter(pl.col("player_id") == 1).sort("game_date").to_dicts()
        p2 = out.filter(pl.col("player_id") == 2).to_dicts()
        assert [r["games_played_to_date"] for r in p1] == [1, 2, 3]
        assert [r["games_played_to_date"] for r in p2] == [1]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_parquet(self, tmp_path):
        df = _player_series(1, ["2026-03-01"], [1000])
        out = TOILoadTracker().compute(df)
        path = write_toi_load(out, tmp_path, "2026-03-01")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in TOI_LOAD_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        df = _player_series(1, ["2026-03-01"], [1000])
        out = TOILoadTracker().compute(df)
        path = write_toi_load(out, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name


# ---------------------------------------------------------------------------
# Sanity check on window constant
# ---------------------------------------------------------------------------

class TestConstants:
    def test_window_is_five(self):
        assert ROLLING_WINDOW_GAMES == 5
