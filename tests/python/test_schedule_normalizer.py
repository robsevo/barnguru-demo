"""Tests for Feature 2.15 — StrengthOfScheduleNormalizer."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from models.schedule_normalizer import (
    StrengthOfScheduleNormalizer,
    compute_sos_weights,
    write_sos_weights,
    SOS_W_MIN,
    SOS_W_MAX,
    SOS_WEIGHT_SCHEMA,
)
from models.bayesian_player_rating import OBS_SIGMA_XGF60


# ---------------------------------------------------------------------------
# compute_sos_weights — standalone function
# ---------------------------------------------------------------------------

class TestComputeSosWeights:
    def test_empty_input(self):
        weights, sigmas = compute_sos_weights(np.array([]))
        assert len(weights) == 0
        assert len(sigmas) == 0

    def test_uniform_quality_gives_weight_one(self):
        # If all opponents have the same quality, weights should all be 1.0
        q = np.array([3.0, 3.0, 3.0])
        weights, sigmas = compute_sos_weights(q)
        np.testing.assert_allclose(weights, [1.0, 1.0, 1.0], atol=1e-9)

    def test_high_quality_opponent_gets_higher_weight(self):
        # Median quality = 2.0; opponent at 4.0 → raw_w = 2.0 → clipped at W_MAX
        q = np.array([1.0, 2.0, 2.0, 4.0])
        weights, _ = compute_sos_weights(q)
        assert weights[-1] == SOS_W_MAX

    def test_low_quality_opponent_gets_lower_weight(self):
        # Median = 2.0; opponent at 0.5 → raw_w = 0.25 → clipped at W_MIN
        q = np.array([2.0, 2.0, 2.0, 0.5])
        weights, _ = compute_sos_weights(q)
        assert weights[-1] == SOS_W_MIN

    def test_all_nan_gives_weight_one(self):
        q = np.array([float("nan"), float("nan")])
        weights, _ = compute_sos_weights(q)
        np.testing.assert_allclose(weights, [1.0, 1.0], atol=1e-9)

    def test_nan_entry_in_mixed_array_defaults_to_one(self):
        q = np.array([2.0, 2.0, float("nan")])
        weights, _ = compute_sos_weights(q)
        assert weights[2] == 1.0

    def test_effective_sigma_formula(self):
        q = np.array([2.0, 2.0])
        base = OBS_SIGMA_XGF60
        weights, sigmas = compute_sos_weights(q, base_obs_sigma=base)
        expected = base / np.sqrt(1.0)
        np.testing.assert_allclose(sigmas, [expected, expected], atol=1e-9)

    def test_higher_weight_gives_lower_sigma(self):
        q = np.array([1.0, 2.0, 4.0])   # median = 2.0
        base = 1.0
        weights, sigmas = compute_sos_weights(q, base_obs_sigma=base)
        # sigma decreases as weight increases
        assert sigmas[0] >= sigmas[1] >= sigmas[2]

    def test_zero_league_avg_gives_weight_one(self):
        q = np.array([0.0, 0.0, 0.0])
        weights, _ = compute_sos_weights(q)
        np.testing.assert_allclose(weights, [1.0, 1.0, 1.0], atol=1e-9)

    def test_custom_w_min_w_max(self):
        q = np.array([1.0, 10.0])
        weights, _ = compute_sos_weights(q, w_min=0.8, w_max=1.2)
        assert all(0.8 <= w <= 1.2 for w in weights)

    def test_weights_within_bounds(self):
        rng = np.random.default_rng(42)
        q = rng.uniform(0.1, 5.0, size=50)
        weights, _ = compute_sos_weights(q)
        assert all(SOS_W_MIN <= w <= SOS_W_MAX for w in weights)


# ---------------------------------------------------------------------------
# StrengthOfScheduleNormalizer — construction and update
# ---------------------------------------------------------------------------

class TestNormalizerInit:
    def test_default_construction(self):
        n = StrengthOfScheduleNormalizer()
        assert n._w_min == SOS_W_MIN
        assert n._w_max == SOS_W_MAX

    def test_custom_params(self):
        n = StrengthOfScheduleNormalizer(w_min=0.6, w_max=1.8)
        assert n._w_min == 0.6
        assert n._w_max == 1.8


class TestUpdateOpponentRatings:
    def _ratings_df(self, pids, vals, col="posterior_mean"):
        return pl.DataFrame({"player_id": pids, col: vals})

    def test_basic_update(self):
        n = StrengthOfScheduleNormalizer()
        df = self._ratings_df([1, 2, 3], [2.0, 3.0, 4.0])
        n.update_opponent_ratings(df)
        assert n.opponent_quality(1) == pytest.approx(2.0)
        assert n.opponent_quality(2) == pytest.approx(3.0)

    def test_unknown_opponent_returns_none(self):
        n = StrengthOfScheduleNormalizer()
        assert n.opponent_quality(999) is None

    def test_missing_player_id_column_raises(self):
        n = StrengthOfScheduleNormalizer()
        df = pl.DataFrame({"posterior_mean": [1.0]})
        with pytest.raises(ValueError, match="player_id"):
            n.update_opponent_ratings(df)

    def test_fallback_to_off_rapm(self):
        n = StrengthOfScheduleNormalizer(quality_col="posterior_mean")
        df = pl.DataFrame({"player_id": [1], "off_rapm": [1.5]})
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            n.update_opponent_ratings(df)
        # Should still load using off_rapm (first fallback)
        # The fallback logic tries available columns
        # Since posterior_mean is absent, it picks off_rapm
        assert n.opponent_quality(1) == pytest.approx(1.5)

    def test_no_fallback_emits_warning(self):
        n = StrengthOfScheduleNormalizer()
        df = pl.DataFrame({"player_id": [1], "some_col": [1.0]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n.update_opponent_ratings(df)
        assert any("uniform weights" in str(warning.message) for warning in w)

    def test_nan_quality_skipped(self):
        n = StrengthOfScheduleNormalizer()
        df = pl.DataFrame({"player_id": [1, 2], "posterior_mean": [float("nan"), 2.0]})
        n.update_opponent_ratings(df)
        assert n.opponent_quality(1) is None
        assert n.opponent_quality(2) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# effective_sigma
# ---------------------------------------------------------------------------

class TestEffectiveSigma:
    def test_unknown_opponent_returns_base(self):
        n = StrengthOfScheduleNormalizer(base_obs_sigma=1.0)
        sigma = n.effective_sigma(opponent_id=999)
        assert sigma == pytest.approx(1.0)

    def test_known_opponent(self):
        n = StrengthOfScheduleNormalizer(base_obs_sigma=1.0)
        df = pl.DataFrame({"player_id": [1, 2, 3], "posterior_mean": [2.0, 2.0, 4.0]})
        n.update_opponent_ratings(df)
        sigma = n.effective_sigma(opponent_id=3)   # high quality → weight > 1 → sigma < base
        assert sigma < 1.0

    def test_direct_quality_overrides_lookup(self):
        n = StrengthOfScheduleNormalizer(base_obs_sigma=1.0)
        df = pl.DataFrame({"player_id": [1, 2, 3], "posterior_mean": [2.0, 2.0, 2.0]})
        n.update_opponent_ratings(df)
        sigma = n.effective_sigma(opponent_quality=2.0)
        assert sigma == pytest.approx(1.0)  # exactly league avg → weight 1 → sigma = base


# ---------------------------------------------------------------------------
# add_weights
# ---------------------------------------------------------------------------

class TestAddWeights:
    def _game_df(self, n_rows: int = 5) -> pl.DataFrame:
        return pl.DataFrame({
            "player_id": list(range(n_rows)),
            "game_id":   list(range(100, 100 + n_rows)),
            "opponent_id": [10, 11, 12, 10, 11],
        })

    def test_adds_required_columns(self):
        n = StrengthOfScheduleNormalizer()
        df = pl.DataFrame({"player_id": [1], "posterior_mean": [2.0]})
        n.update_opponent_ratings(df)
        game_df = pl.DataFrame({
            "player_id": [1], "game_id": [100], "opponent_id": [1],
        })
        out = n.add_weights(game_df, season=2025)
        for col in ["sos_weight", "effective_sigma", "opponent_quality"]:
            assert col in out.columns

    def test_missing_required_columns_raises(self):
        n = StrengthOfScheduleNormalizer()
        df = pl.DataFrame({"player_id": [1]})  # missing game_id
        with pytest.raises(ValueError, match="game_id"):
            n.add_weights(df, season=2025)

    def test_unknown_opponent_gets_weight_one(self):
        n = StrengthOfScheduleNormalizer()
        game_df = pl.DataFrame({
            "player_id": [1], "game_id": [100], "opponent_id": [999],
        })
        out = n.add_weights(game_df, season=2025)
        assert out["sos_weight"][0] == pytest.approx(1.0)

    def test_weights_within_bounds(self):
        n = StrengthOfScheduleNormalizer()
        ratings = pl.DataFrame({
            "player_id": [10, 11, 12],
            "posterior_mean": [1.0, 2.0, 5.0],
        })
        n.update_opponent_ratings(ratings)
        game_df = self._game_df(5)
        out = n.add_weights(game_df, season=2025)
        assert out["sos_weight"].min() >= SOS_W_MIN - 1e-9
        assert out["sos_weight"].max() <= SOS_W_MAX + 1e-9

    def test_missing_opponent_col_defaults_neutral(self):
        n = StrengthOfScheduleNormalizer()
        game_df = pl.DataFrame({"player_id": [1], "game_id": [100]})
        out = n.add_weights(game_df, season=2025)
        assert "sos_weight" in out.columns


# ---------------------------------------------------------------------------
# season_sos_summary
# ---------------------------------------------------------------------------

class TestSeasonSosSummary:
    def test_returns_dataframe(self):
        n = StrengthOfScheduleNormalizer()
        ratings = pl.DataFrame({"player_id": [10, 11], "posterior_mean": [2.0, 4.0]})
        n.update_opponent_ratings(ratings)
        game_df = pl.DataFrame({
            "player_id": [1, 1, 2, 2],
            "game_id":   [100, 101, 102, 103],
            "opponent_id": [10, 11, 10, 11],
        })
        summary = n.season_sos_summary(game_df, season=2025)
        assert isinstance(summary, pl.DataFrame)
        for col in ["player_id", "n_games", "mean_sos_weight", "min_sos_weight", "max_sos_weight"]:
            assert col in summary.columns

    def test_summary_row_count(self):
        n = StrengthOfScheduleNormalizer()
        game_df = pl.DataFrame({
            "player_id": [1, 2, 3],
            "game_id": [100, 101, 102],
        })
        summary = n.season_sos_summary(game_df, season=2025)
        assert len(summary) == 3


# ---------------------------------------------------------------------------
# to_parquet_df
# ---------------------------------------------------------------------------

class TestToParquetDf:
    def test_schema_correct(self):
        n = StrengthOfScheduleNormalizer()
        game_df = pl.DataFrame({
            "player_id": [1], "game_id": [100],
        })
        df = n.to_parquet_df(game_df, season=2025)
        for col in SOS_WEIGHT_SCHEMA:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        n = StrengthOfScheduleNormalizer(w_min=0.6, w_max=1.8)
        ratings = pl.DataFrame({"player_id": [1, 2], "posterior_mean": [2.0, 4.0]})
        n.update_opponent_ratings(ratings)

        path = tmp_path / "sos.pkl"
        n.save(path)

        n2 = StrengthOfScheduleNormalizer.load(path)
        assert n2._w_min == pytest.approx(0.6)
        assert n2._w_max == pytest.approx(1.8)
        assert n2.opponent_quality(1) == pytest.approx(2.0)
        assert n2.opponent_quality(2) == pytest.approx(4.0)

    def test_write_sos_weights(self, tmp_path):
        n = StrengthOfScheduleNormalizer()
        game_df = pl.DataFrame({"player_id": [1], "game_id": [100]})
        df = n.to_parquet_df(game_df, season=2025)
        path = write_sos_weights(df, tmp_path, 2025)
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in SOS_WEIGHT_SCHEMA:
            assert col in loaded.columns
