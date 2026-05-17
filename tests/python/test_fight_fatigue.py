"""Tests for Feature 3.11 — FightFatigueTracker."""

from __future__ import annotations

import math
import warnings

import polars as pl
import pytest

from models.fight_fatigue import (
    FIGHT_FATIGUE_SCHEMA,
    HALF_LIFE_DAYS,
    NO_RECENT_FIGHT,
    ROLLING_WINDOW_DAYS,
    FightFatigueTracker,
    fight_decay_weight,
    write_fight_fatigue,
)
from models.rapm_model import DataMissingWarning


def _fight_df(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "game_id":          0,
        "game_date":        "2026-03-01",
        "player_id":        1,
        "team_id":          10,
        "fights_this_game": 0,
    }
    filled = [{**defaults, **r} for r in rows]
    for i, r in enumerate(filled):
        if r.get("game_id") in (None, 0):
            r["game_id"] = 1_000 + i
    if not filled:
        return pl.DataFrame(
            schema={
                "game_id":          pl.Int64,
                "game_date":        pl.Utf8,
                "player_id":        pl.Int64,
                "team_id":          pl.Int64,
                "fights_this_game": pl.Int64,
            }
        )
    return pl.DataFrame(filled)


def _series(player_id: int, dates: list[str], fights: list[int]) -> pl.DataFrame:
    assert len(dates) == len(fights)
    return _fight_df([
        {"player_id": player_id, "game_date": d, "fights_this_game": f}
        for d, f in zip(dates, fights)
    ])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_columns_warns(self):
        bad = pl.DataFrame({"foo": ["bar"]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = FightFatigueTracker().compute(bad)
        assert len(result) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_returns_empty_schema(self):
        result = FightFatigueTracker().compute(_fight_df([]))
        for col in FIGHT_FATIGUE_SCHEMA:
            assert col in result.columns
        assert len(result) == 0

    def test_output_schema(self):
        df = _series(1, ["2026-03-01"], [1])
        result = FightFatigueTracker().compute(df)
        assert set(result.columns) == set(FIGHT_FATIGUE_SCHEMA.keys())

    def test_bad_half_life_rejected(self):
        with pytest.raises(ValueError):
            FightFatigueTracker(half_life_days=0.0)
        with pytest.raises(ValueError):
            FightFatigueTracker(half_life_days=-1.0)

    def test_bad_window_rejected(self):
        with pytest.raises(ValueError):
            FightFatigueTracker(window_days=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_window_is_two_weeks(self):
        assert ROLLING_WINDOW_DAYS == 14

    def test_half_life_positive(self):
        assert HALF_LIFE_DAYS > 0


# ---------------------------------------------------------------------------
# Decay-weight function
# ---------------------------------------------------------------------------

class TestDecayWeight:
    def test_same_day_full_weight(self):
        assert fight_decay_weight(0) == pytest.approx(1.0)

    def test_half_life_day_is_half(self):
        # By construction: weight at HALF_LIFE_DAYS should equal 0.5.
        assert fight_decay_weight(int(HALF_LIFE_DAYS)) == pytest.approx(
            math.exp(-math.log(2.0))
        )

    def test_weight_decays_monotonically(self):
        ws = [fight_decay_weight(d) for d in range(0, 8)]
        for a, b in zip(ws, ws[1:]):
            assert b < a

    def test_negative_days_clipped(self):
        assert fight_decay_weight(-5) == pytest.approx(1.0)

    def test_custom_half_life(self):
        # half-life of 1 day → 1 day ago == 0.5.
        assert fight_decay_weight(1, half_life=1.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Single-game outputs
# ---------------------------------------------------------------------------

class TestSingleGame:
    def test_no_fight_zero(self):
        df = _series(1, ["2026-03-01"], [0])
        row = FightFatigueTracker().compute(df).to_dicts()[0]
        assert row["fights_this_game"] == 0
        assert row["fights_14day"] == 0
        assert row["days_since_last_fight"] == NO_RECENT_FIGHT
        assert row["fight_load_score"] == 0.0

    def test_one_fight_full_load(self):
        df = _series(1, ["2026-03-01"], [1])
        row = FightFatigueTracker().compute(df).to_dicts()[0]
        assert row["fights_14day"] == 1
        assert row["days_since_last_fight"] == 0
        assert row["fight_load_score"] == pytest.approx(1.0)

    def test_two_fights_same_game(self):
        df = _series(1, ["2026-03-01"], [2])
        row = FightFatigueTracker().compute(df).to_dicts()[0]
        assert row["fights_14day"] == 2
        assert row["fight_load_score"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Decay across games
# ---------------------------------------------------------------------------

class TestDecayAcrossGames:
    def test_load_decays_with_days(self):
        # Fight on day 0; check load on a follow-up no-fight game N days later.
        for gap in (1, 3, 6):
            df = _series(
                1,
                ["2026-03-01", f"2026-03-{1 + gap:02d}"],
                [1, 0],
            )
            rows = FightFatigueTracker().compute(df).sort("game_date").to_dicts()
            expected = fight_decay_weight(gap)
            assert rows[1]["fight_load_score"] == pytest.approx(expected)
            assert rows[1]["fights_14day"] == 1
            assert rows[1]["days_since_last_fight"] == gap

    def test_window_drops_old_fight(self):
        # Fight on day 0; on day 14 the window cutoff is day 1
        # (15-day inclusive span = 14 day gap excluded).
        df = _series(1, ["2026-03-01", "2026-03-15"], [1, 0])
        rows = FightFatigueTracker().compute(df).sort("game_date").to_dicts()
        assert rows[1]["fights_14day"] == 0
        assert rows[1]["days_since_last_fight"] == NO_RECENT_FIGHT
        assert rows[1]["fight_load_score"] == 0.0

    def test_multiple_recent_fights_compound(self):
        df = _series(
            1,
            ["2026-03-01", "2026-03-04", "2026-03-05"],
            [1, 1, 0],
        )
        rows = FightFatigueTracker().compute(df).sort("game_date").to_dicts()
        # On row 2 (no fight today): one fight 4 days ago, one fight 1 day ago.
        expected = fight_decay_weight(4) + fight_decay_weight(1)
        assert rows[2]["fights_14day"] == 2
        assert rows[2]["days_since_last_fight"] == 1
        assert rows[2]["fight_load_score"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Multi-player independence
# ---------------------------------------------------------------------------

class TestMultiPlayer:
    def test_players_independent(self):
        rows = [
            {"player_id": 1, "game_date": "2026-03-01", "fights_this_game": 1},
            {"player_id": 1, "game_date": "2026-03-02", "fights_this_game": 0},
            {"player_id": 2, "game_date": "2026-03-01", "fights_this_game": 0},
            {"player_id": 2, "game_date": "2026-03-02", "fights_this_game": 0},
        ]
        out = FightFatigueTracker().compute(_fight_df(rows))
        p1 = out.filter(pl.col("player_id") == 1).sort("game_date").to_dicts()
        p2 = out.filter(pl.col("player_id") == 2).sort("game_date").to_dicts()
        assert p1[-1]["fights_14day"] == 1
        assert p1[-1]["fight_load_score"] > 0.0
        assert p2[-1]["fights_14day"] == 0
        assert p2[-1]["fight_load_score"] == 0.0


# ---------------------------------------------------------------------------
# Custom half-life
# ---------------------------------------------------------------------------

class TestCustomHalfLife:
    def test_shorter_half_life_decays_faster(self):
        df = _series(1, ["2026-03-01", "2026-03-04"], [1, 0])
        fast = FightFatigueTracker(half_life_days=1.0).compute(df).sort("game_date").to_dicts()
        slow = FightFatigueTracker(half_life_days=7.0).compute(df).sort("game_date").to_dicts()
        # Same evidence, faster decay → smaller load three days later.
        assert fast[1]["fight_load_score"] < slow[1]["fight_load_score"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_parquet(self, tmp_path):
        df = _series(1, ["2026-03-01"], [1])
        out = FightFatigueTracker().compute(df)
        path = write_fight_fatigue(out, tmp_path, "2026-03-01")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in FIGHT_FATIGUE_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        df = _series(1, ["2026-03-01"], [1])
        out = FightFatigueTracker().compute(df)
        path = write_fight_fatigue(out, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name
