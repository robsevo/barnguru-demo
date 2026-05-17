"""Tests for Feature 3.15 — PriorPlayoffLoad."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from models.prior_playoff_load import (
    DECAY_GAMES,
    PENALTY_MAX,
    PENALTY_PER_GAME,
    PRIOR_PLAYOFF_LOAD_SCHEMA,
    PriorPlayoffLoad,
    base_penalty,
    current_penalty,
    decay_weight_for,
    write_prior_playoff_load,
)
from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _load_df(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "player_id":                pl.Int64,
        "prior_playoff_games":      pl.Int64,
        "prior_playoff_round":      pl.Int64,
        "games_played_this_season": pl.Int64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    defaults = {"prior_playoff_round": 0}
    filled = [{**defaults, **r} for r in rows]
    return pl.DataFrame(filled, schema=schema)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_columns_warns(self):
        bad = pl.DataFrame({"foo": [1]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = PriorPlayoffLoad().compute(bad, "2026-05-17")
        assert len(out) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_input_returns_empty_schema(self):
        out = PriorPlayoffLoad().compute(_load_df([]), "2026-05-17")
        for col in PRIOR_PLAYOFF_LOAD_SCHEMA:
            assert col in out.columns
        assert len(out) == 0

    def test_output_schema(self):
        df = _load_df([{"player_id": 1, "prior_playoff_games": 7,
                        "games_played_this_season": 0}])
        out = PriorPlayoffLoad().compute(df, "2026-05-17")
        assert set(out.columns) == set(PRIOR_PLAYOFF_LOAD_SCHEMA.keys())

    def test_bad_as_of_date_raises(self):
        df = _load_df([{"player_id": 1, "prior_playoff_games": 5,
                        "games_played_this_season": 0}])
        with pytest.raises(ValueError):
            PriorPlayoffLoad().compute(df, "not-a-date")

    def test_bad_params_rejected(self):
        with pytest.raises(ValueError):
            PriorPlayoffLoad(penalty_per_game=-0.1)
        with pytest.raises(ValueError):
            PriorPlayoffLoad(penalty_max=-0.1)
        with pytest.raises(ValueError):
            PriorPlayoffLoad(decay_games=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_penalty_per_game_positive(self):
        assert PENALTY_PER_GAME > 0

    def test_penalty_max_positive(self):
        assert PENALTY_MAX > 0

    def test_decay_games_positive(self):
        assert DECAY_GAMES >= 1


# ---------------------------------------------------------------------------
# Formula
# ---------------------------------------------------------------------------

class TestBasePenalty:
    def test_zero_games_zero_penalty(self):
        assert base_penalty(0) == pytest.approx(0.0)

    def test_negative_games_zero_penalty(self):
        assert base_penalty(-10) == pytest.approx(0.0)

    def test_linear_below_cap(self):
        assert base_penalty(10) == pytest.approx(10 * PENALTY_PER_GAME)

    def test_clamped_at_cap(self):
        # 100 games well exceeds any realistic cap → clamped.
        assert base_penalty(1000) == pytest.approx(PENALTY_MAX)


class TestDecayWeight:
    def test_at_opener_is_one(self):
        assert decay_weight_for(0) == pytest.approx(1.0)

    def test_zero_at_decay_games(self):
        assert decay_weight_for(DECAY_GAMES) == pytest.approx(0.0)

    def test_zero_past_decay_games(self):
        assert decay_weight_for(DECAY_GAMES + 50) == pytest.approx(0.0)

    def test_monotone_non_increasing(self):
        prev = decay_weight_for(0)
        for g in range(1, DECAY_GAMES + 3):
            curr = decay_weight_for(g)
            assert curr <= prev
            prev = curr

    def test_negative_games_treated_as_zero(self):
        assert decay_weight_for(-5) == pytest.approx(1.0)

    def test_within_unit_interval(self):
        for g in range(-2, DECAY_GAMES + 5):
            v = decay_weight_for(g)
            assert 0.0 <= v <= 1.0


class TestCurrentPenalty:
    def test_no_prior_no_penalty(self):
        assert current_penalty(0, 0) == pytest.approx(0.0)

    def test_zero_after_decay(self):
        assert current_penalty(20, DECAY_GAMES + 1) == pytest.approx(0.0)

    def test_decays_with_games(self):
        a = current_penalty(20, 0)
        b = current_penalty(20, DECAY_GAMES // 2)
        c = current_penalty(20, DECAY_GAMES - 1)
        assert a > b > c >= 0.0


# ---------------------------------------------------------------------------
# Compute path
# ---------------------------------------------------------------------------

class TestCompute:
    def test_no_playoff_no_penalty(self):
        df = _load_df([{"player_id": 1, "prior_playoff_games": 0,
                        "games_played_this_season": 0}])
        out = PriorPlayoffLoad().compute(df, "2026-05-17").to_dicts()[0]
        assert out["base_playoff_penalty"] == pytest.approx(0.0)
        assert out["playoff_load_penalty"] == pytest.approx(0.0)
        assert out["decay_weight"] == pytest.approx(1.0)

    def test_opener_has_full_base_penalty(self):
        df = _load_df([{"player_id": 1, "prior_playoff_games": 10,
                        "games_played_this_season": 0}])
        out = PriorPlayoffLoad().compute(df, "2026-05-17").to_dicts()[0]
        assert out["playoff_load_penalty"] == pytest.approx(
            base_penalty(10)
        )

    def test_mid_decay_partial(self):
        df = _load_df([{"player_id": 1, "prior_playoff_games": 20,
                        "games_played_this_season": DECAY_GAMES // 2}])
        out = PriorPlayoffLoad().compute(df, "2026-05-17").to_dicts()[0]
        expected = base_penalty(20) * decay_weight_for(DECAY_GAMES // 2)
        assert out["playoff_load_penalty"] == pytest.approx(expected)
        assert out["base_playoff_penalty"] > out["playoff_load_penalty"]

    def test_post_decay_zero(self):
        df = _load_df([{"player_id": 1, "prior_playoff_games": 26,
                        "games_played_this_season": DECAY_GAMES}])
        out = PriorPlayoffLoad().compute(df, "2026-05-17").to_dicts()[0]
        assert out["playoff_load_penalty"] == pytest.approx(0.0)
        assert out["base_playoff_penalty"] == pytest.approx(PENALTY_MAX)

    def test_round_carried_through_and_clamped(self):
        df = _load_df([
            {"player_id": 1, "prior_playoff_games": 7,
             "prior_playoff_round": 1, "games_played_this_season": 0},
            {"player_id": 2, "prior_playoff_games": 28,
             "prior_playoff_round": 9, "games_played_this_season": 0},
        ])
        out = PriorPlayoffLoad().compute(df, "2026-05-17").sort("player_id").to_dicts()
        assert out[0]["prior_playoff_round"] == 1
        assert out[1]["prior_playoff_round"] == 4   # clamped to max round 4

    def test_duplicate_player_deduped(self):
        df = _load_df([
            {"player_id": 1, "prior_playoff_games": 7,
             "games_played_this_season": 0},
            {"player_id": 1, "prior_playoff_games": 14,
             "games_played_this_season": 5},
        ])
        out = PriorPlayoffLoad().compute(df, "2026-05-17")
        assert len(out) == 1

    def test_negative_inputs_skipped(self):
        df = _load_df([
            {"player_id": 1, "prior_playoff_games": -5,
             "games_played_this_season": 0},
            {"player_id": 2, "prior_playoff_games": 5,
             "games_played_this_season": -3},
            {"player_id": 3, "prior_playoff_games": 5,
             "games_played_this_season": 0},
        ])
        out = PriorPlayoffLoad().compute(df, "2026-05-17")
        assert set(out["player_id"].to_list()) == {3}

    def test_penalty_in_valid_range(self):
        df = _load_df([
            {"player_id": i, "prior_playoff_games": gp,
             "games_played_this_season": g}
            for i, (gp, g) in enumerate(
                [(0, 0), (7, 5), (14, 10), (28, 0), (28, DECAY_GAMES)],
                start=1,
            )
        ])
        out = PriorPlayoffLoad().compute(df, "2026-05-17")
        for r in out.to_dicts():
            assert 0.0 <= r["base_playoff_penalty"] <= PENALTY_MAX
            assert 0.0 <= r["playoff_load_penalty"] <= PENALTY_MAX
            assert 0.0 <= r["decay_weight"] <= 1.0

    def test_round_defaults_to_zero_when_missing(self):
        # prior_playoff_round column present but null → treated as 0.
        df = pl.DataFrame(
            [{"player_id": 1, "prior_playoff_games": 7,
              "prior_playoff_round": None, "games_played_this_season": 0}],
            schema={
                "player_id":                pl.Int64,
                "prior_playoff_games":      pl.Int64,
                "prior_playoff_round":      pl.Int64,
                "games_played_this_season": pl.Int64,
            },
        )
        out = PriorPlayoffLoad().compute(df, "2026-05-17").to_dicts()[0]
        assert out["prior_playoff_round"] == 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_parquet(self, tmp_path):
        df = _load_df([{"player_id": 1, "prior_playoff_games": 7,
                        "games_played_this_season": 0}])
        out = PriorPlayoffLoad().compute(df, "2026-05-17")
        path = write_prior_playoff_load(out, tmp_path, "2026-05-17")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in PRIOR_PLAYOFF_LOAD_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        out = PriorPlayoffLoad().compute(_load_df([]), "2026-05-17")
        path = write_prior_playoff_load(out, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name
