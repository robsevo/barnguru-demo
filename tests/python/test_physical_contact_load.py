"""Tests for Feature 3.9 — PhysicalContactLoadTracker."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from models.physical_contact_load import (
    BLOCK_WEIGHT,
    HIT_WEIGHT,
    PHYSICAL_LOAD_SCHEMA,
    ROLLING_WINDOW_GAMES,
    PhysicalContactLoadTracker,
    contact_score,
    write_physical_contact_load,
)
from models.rapm_model import DataMissingWarning


def _contact_df(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "game_id":   0,
        "game_date": "2026-03-01",
        "player_id": 1,
        "team_id":   10,
        "hits_taken":    0,
        "blocked_shots": 0,
    }
    filled = [{**defaults, **r} for r in rows]
    for i, r in enumerate(filled):
        if r.get("game_id") in (None, 0):
            r["game_id"] = 1_000 + i
    if not filled:
        return pl.DataFrame(
            schema={
                "game_id":       pl.Int64,
                "game_date":     pl.Utf8,
                "player_id":     pl.Int64,
                "team_id":       pl.Int64,
                "hits_taken":    pl.Int64,
                "blocked_shots": pl.Int64,
            }
        )
    return pl.DataFrame(filled)


def _player_series(
    player_id: int,
    dates: list[str],
    hits: list[int],
    blocks: list[int],
) -> pl.DataFrame:
    assert len(dates) == len(hits) == len(blocks)
    return _contact_df([
        {
            "player_id":  player_id,
            "game_date":  d,
            "hits_taken": h,
            "blocked_shots": b,
        }
        for d, h, b in zip(dates, hits, blocks)
    ])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_columns_warns(self):
        bad = pl.DataFrame({"foo": ["bar"]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = PhysicalContactLoadTracker().compute(bad)
        assert len(result) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_returns_empty_schema(self):
        result = PhysicalContactLoadTracker().compute(_contact_df([]))
        for col in PHYSICAL_LOAD_SCHEMA:
            assert col in result.columns
        assert len(result) == 0

    def test_output_schema(self):
        df = _player_series(1, ["2026-03-01"], [3], [2])
        result = PhysicalContactLoadTracker().compute(df)
        assert set(result.columns) == set(PHYSICAL_LOAD_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Contact-score formula
# ---------------------------------------------------------------------------

class TestContactScore:
    def test_pure_hits(self):
        assert contact_score(3, 0) == pytest.approx(3 * HIT_WEIGHT)

    def test_pure_blocks(self):
        assert contact_score(0, 4) == pytest.approx(4 * BLOCK_WEIGHT)

    def test_mixed(self):
        assert contact_score(2, 3) == pytest.approx(2 * HIT_WEIGHT + 3 * BLOCK_WEIGHT)

    def test_block_weight_exceeds_hit_weight(self):
        # The model's whole point: blocks are weighted higher than hits.
        assert BLOCK_WEIGHT > HIT_WEIGHT


# ---------------------------------------------------------------------------
# Rolling-window math
# ---------------------------------------------------------------------------

class TestRollingWindow:
    def test_first_game_sums_to_current(self):
        out = PhysicalContactLoadTracker().compute(
            _player_series(1, ["2026-03-01"], [3], [2])
        ).to_dicts()[0]
        assert out["hits_5game_sum"] == 3
        assert out["blocks_5game_sum"] == 2
        assert out["contact_score"] == pytest.approx(3 * HIT_WEIGHT + 2 * BLOCK_WEIGHT)
        assert out["contact_5game_score"] == pytest.approx(3 * HIT_WEIGHT + 2 * BLOCK_WEIGHT)

    def test_five_game_window_inclusive(self):
        dates = [f"2026-03-{i:02d}" for i in range(1, 7)]
        hits  = [2] * 6
        blks  = [1] * 6
        out = PhysicalContactLoadTracker().compute(
            _player_series(1, dates, hits, blks)
        ).sort("game_date").to_dicts()
        # Game 5: window covers 1–5 → 10 hits, 5 blocks.
        assert out[4]["hits_5game_sum"] == 10
        assert out[4]["blocks_5game_sum"] == 5
        # Game 6: drops Game 1 → still 10 / 5.
        assert out[5]["hits_5game_sum"] == 10
        assert out[5]["blocks_5game_sum"] == 5

    def test_running_sum_grows_until_window_full(self):
        dates = [f"2026-03-{i:02d}" for i in range(1, 5)]
        out = PhysicalContactLoadTracker().compute(
            _player_series(1, dates, [1, 1, 1, 1], [0, 0, 0, 0])
        ).sort("game_date").to_dicts()
        assert [r["hits_5game_sum"] for r in out] == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Score combinations
# ---------------------------------------------------------------------------

class TestScoreCombinations:
    def test_block_heavy_player_gets_higher_score(self):
        # 5 blocks vs 5 hits (same count) → block-heavy must score higher.
        a = PhysicalContactLoadTracker().compute(
            _player_series(1, ["2026-03-01"], [5], [0])
        ).to_dicts()[0]["contact_score"]
        b = PhysicalContactLoadTracker().compute(
            _player_series(1, ["2026-03-01"], [0], [5])
        ).to_dicts()[0]["contact_score"]
        assert b > a

    def test_weight_override(self):
        tracker = PhysicalContactLoadTracker(hit_weight=2.0, block_weight=4.0)
        out = tracker.compute(
            _player_series(1, ["2026-03-01"], [2], [1])
        ).to_dicts()[0]
        assert out["contact_score"] == pytest.approx(2 * 2.0 + 1 * 4.0)


# ---------------------------------------------------------------------------
# Multi-player independence
# ---------------------------------------------------------------------------

class TestMultiPlayer:
    def test_players_independent(self):
        rows = [
            {"player_id": 1, "game_date": "2026-03-01", "hits_taken": 5, "blocked_shots": 0},
            {"player_id": 1, "game_date": "2026-03-02", "hits_taken": 5, "blocked_shots": 0},
            {"player_id": 2, "game_date": "2026-03-01", "hits_taken": 0, "blocked_shots": 5},
            {"player_id": 2, "game_date": "2026-03-02", "hits_taken": 0, "blocked_shots": 5},
        ]
        out = PhysicalContactLoadTracker().compute(_contact_df(rows))
        p1 = out.filter(pl.col("player_id") == 1).sort("game_date").to_dicts()
        p2 = out.filter(pl.col("player_id") == 2).sort("game_date").to_dicts()
        assert p1[-1]["hits_5game_sum"] == 10
        assert p1[-1]["blocks_5game_sum"] == 0
        assert p2[-1]["hits_5game_sum"] == 0
        assert p2[-1]["blocks_5game_sum"] == 10


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_parquet(self, tmp_path):
        df = _player_series(1, ["2026-03-01"], [2], [1])
        out = PhysicalContactLoadTracker().compute(df)
        path = write_physical_contact_load(out, tmp_path, "2026-03-01")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in PHYSICAL_LOAD_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        df = _player_series(1, ["2026-03-01"], [2], [1])
        out = PhysicalContactLoadTracker().compute(df)
        path = write_physical_contact_load(out, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_window_is_five(self):
        assert ROLLING_WINDOW_GAMES == 5
