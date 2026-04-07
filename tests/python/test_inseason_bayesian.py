"""Tests for InSeasonBayesianPipeline (Feature 2.12)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from models.inseason_bayesian import (
    BLEND_GAME_HIGH,
    BLEND_GAME_LOW,
    BLEND_W_GAME10,
    BLEND_W_GAME82,
    INSEASON_RATING_SCHEMA,
    InSeasonBayesianPipeline,
    blend_posteriors,
    blend_weight,
    blend_weights_array,
    write_inseason_ratings,
    read_inseason_ratings,
)


# ---------------------------------------------------------------------------
# blend_weight tests
# ---------------------------------------------------------------------------

class TestBlendWeight:

    def test_at_game_10_returns_025(self):
        assert blend_weight(10) == pytest.approx(BLEND_W_GAME10, abs=1e-9)

    def test_at_game_82_returns_073(self):
        assert blend_weight(82) == pytest.approx(BLEND_W_GAME82, abs=1e-9)

    def test_at_game_0_clamped_to_zero(self):
        assert blend_weight(0) >= 0.0

    def test_monotone_increasing(self):
        weights = [blend_weight(g) for g in range(0, 90)]
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1]

    def test_beyond_game_82_capped(self):
        assert blend_weight(100) == pytest.approx(BLEND_W_GAME82, abs=1e-9)

    def test_mid_season_interpolation(self):
        w_mid = blend_weight(46)   # midpoint of [10, 82]
        expected = (BLEND_W_GAME10 + BLEND_W_GAME82) / 2
        assert w_mid == pytest.approx(expected, abs=0.02)

    def test_vectorised_matches_scalar(self):
        games = np.array([0, 10, 20, 40, 60, 82, 100])
        scalar_ws = np.array([blend_weight(g) for g in games])
        vector_ws = blend_weights_array(games)
        np.testing.assert_array_almost_equal(scalar_ws, vector_ws, decimal=9)


# ---------------------------------------------------------------------------
# blend_posteriors tests
# ---------------------------------------------------------------------------

class TestBlendPosteriors:

    def test_at_game_10_75pct_historical(self):
        mu_blend, sigma_blend, w = blend_posteriors(
            mu_hist    = 2.0,
            sigma_hist = 0.6,
            mu_curr    = 4.0,
            sigma_curr = 0.4,
            games_played = 10,
        )
        assert w == pytest.approx(BLEND_W_GAME10, abs=1e-9)
        expected_mu = 0.25 * 4.0 + 0.75 * 2.0
        assert mu_blend == pytest.approx(expected_mu, abs=1e-9)

    def test_at_game_82_73pct_current(self):
        mu_blend, _, w = blend_posteriors(
            mu_hist=2.0, sigma_hist=0.6,
            mu_curr=4.0, sigma_curr=0.4,
            games_played=82,
        )
        assert w == pytest.approx(BLEND_W_GAME82, abs=1e-9)
        expected_mu = 0.73 * 4.0 + 0.27 * 2.0
        assert mu_blend == pytest.approx(expected_mu, abs=1e-9)

    def test_sigma_blend_non_negative(self):
        _, sigma_blend, _ = blend_posteriors(
            mu_hist=2.5, sigma_hist=0.6,
            mu_curr=3.0, sigma_curr=0.3,
            games_played=40,
        )
        assert sigma_blend >= 0

    def test_blend_is_convex_combination_of_means(self):
        mu_h, sigma_h, mu_c, sigma_c = 2.0, 0.6, 5.0, 0.3
        for g in [10, 25, 50, 82]:
            mu_b, _, w = blend_posteriors(mu_h, sigma_h, mu_c, sigma_c, g)
            expected = w * mu_c + (1 - w) * mu_h
            assert mu_b == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# InSeasonBayesianPipeline — initialisation
# ---------------------------------------------------------------------------

def _make_historical_df(n: int = 20, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "player_id":       list(range(1000, 1000 + n)),
        "player_name":     [f"Player{i}" for i in range(n)],
        "posterior_mean":  rng.normal(2.5, 0.6, n).tolist(),
        "posterior_sigma": rng.uniform(0.2, 0.5, n).tolist(),
    })


class TestPipelineInit:

    def test_init_without_historical(self):
        pipeline = InSeasonBayesianPipeline()
        assert pipeline.n_players_observed() == 0

    def test_init_with_historical(self):
        hist = _make_historical_df(20)
        pipeline = InSeasonBayesianPipeline(hist)
        # Historical loaded but no current-season observations yet
        assert pipeline.n_players_observed() == 0

    def test_historical_missing_columns_raises(self):
        bad_df = pl.DataFrame({"player_id": [1, 2]})
        with pytest.raises(ValueError, match="missing columns"):
            InSeasonBayesianPipeline(bad_df)


# ---------------------------------------------------------------------------
# InSeasonBayesianPipeline — observe
# ---------------------------------------------------------------------------

class TestPipelineObserve:

    def test_single_observe(self):
        pipeline = InSeasonBayesianPipeline()
        pipeline.observe(1000, "Matthews", 3.5)
        assert pipeline.n_players_observed() == 1
        assert pipeline.games_played(1000) == 1

    def test_multiple_games_increments(self):
        pipeline = InSeasonBayesianPipeline()
        for _ in range(10):
            pipeline.observe(1000, "A", 3.0)
        assert pipeline.games_played(1000) == 10

    def test_observe_multiple_players(self):
        pipeline = InSeasonBayesianPipeline()
        pipeline.observe(1001, "A", 3.0)
        pipeline.observe(1002, "B", 2.0)
        assert pipeline.n_players_observed() == 2

    def test_observe_game_batch(self):
        pipeline = InSeasonBayesianPipeline()
        game_df = pl.DataFrame({
            "player_id":   [1001, 1002, 1003],
            "player_name": ["A", "B", "C"],
            "xgf_per60":   [3.0, 2.0, 1.5],
        })
        pipeline.observe_game(game_df)
        assert pipeline.n_players_observed() == 3

    def test_observe_game_missing_column_raises(self):
        pipeline = InSeasonBayesianPipeline()
        bad_df = pl.DataFrame({"player_id": [1]})
        with pytest.raises(ValueError, match="missing columns"):
            pipeline.observe_game(bad_df)

    def test_observe_skips_null_observations(self):
        pipeline = InSeasonBayesianPipeline()
        game_df = pl.DataFrame({
            "player_id":   [1001, 1002],
            "player_name": ["A", "B"],
            "xgf_per60":   [None, 2.5],
        })
        pipeline.observe_game(game_df)
        assert pipeline.games_played(1001) == 0
        assert pipeline.games_played(1002) == 1


# ---------------------------------------------------------------------------
# InSeasonBayesianPipeline — blended_ratings
# ---------------------------------------------------------------------------

class TestPipelineBlendedRatings:

    def test_blended_ratings_schema(self):
        hist = _make_historical_df(10)
        pipeline = InSeasonBayesianPipeline(hist)
        for i in range(10):
            pipeline.observe(1000 + i, f"P{i}", 2.5 + i * 0.1)
        result = pipeline.blended_ratings(season=2025)
        for col in INSEASON_RATING_SCHEMA:
            assert col in result.columns, f"Missing: {col}"

    def test_blended_ratings_sorted_descending(self):
        pipeline = InSeasonBayesianPipeline()
        for i in range(5):
            pipeline.observe(1000 + i, f"P{i}", float(i + 1))
        result = pipeline.blended_ratings(season=2025)
        vals = result["mu_blend"].to_list()
        assert vals == sorted(vals, reverse=True)

    def test_blend_weight_reflects_games(self):
        hist = _make_historical_df(5)
        pipeline = InSeasonBayesianPipeline(hist)
        pid = 1000
        # Observe 10 games → weight should be BLEND_W_GAME10
        for _ in range(10):
            pipeline.observe(pid, "P0", 5.0)
        result = pipeline.blended_ratings(season=2025)
        row = result.filter(pl.col("player_id") == pid).row(0, named=True)
        assert row["blend_weight"] == pytest.approx(BLEND_W_GAME10, abs=0.01)
        assert row["games_played"] == 10

    def test_historical_used_as_prior(self):
        hist = pl.DataFrame({
            "player_id":      [1000],
            "player_name":    ["Star"],
            "posterior_mean": [4.5],
            "posterior_sigma": [0.3],
        })
        pipeline = InSeasonBayesianPipeline(hist)
        # Zero games → use historical only
        result = pipeline.player_blended_rating(1000, season=2025)
        assert result is None  # no current-season observations yet

    def test_player_not_in_historical_uses_league_prior(self):
        pipeline = InSeasonBayesianPipeline()
        pipeline.observe(9999, "Unknown", 2.5)
        result = pipeline.blended_ratings(season=2025)
        assert len(result) == 1
        row = result.row(0, named=True)
        assert row["mu_historical"] == pytest.approx(pipeline._league_mean, abs=1e-6)

    def test_player_blended_rating_returns_dict(self):
        pipeline = InSeasonBayesianPipeline()
        pipeline.observe(1001, "A", 3.0)
        result = pipeline.player_blended_rating(1001, season=2025)
        assert result is not None
        assert "mu_blend" in result
        assert "blend_weight" in result

    def test_player_blended_rating_unknown_returns_none(self):
        pipeline = InSeasonBayesianPipeline()
        assert pipeline.player_blended_rating(99999, season=2025) is None

    def test_ci_encompasses_blend_mean(self):
        pipeline = InSeasonBayesianPipeline()
        for _ in range(20):
            pipeline.observe(1000, "A", 3.0)
        result = pipeline.blended_ratings(season=2025)
        row = result.row(0, named=True)
        assert row["ci_lower_95"] < row["mu_blend"] < row["ci_upper_95"]

    def test_empty_when_no_observations(self):
        pipeline = InSeasonBayesianPipeline()
        result = pipeline.blended_ratings(season=2025)
        assert len(result) == 0

    def test_higher_obs_increases_blend_mean(self):
        """Player with high xGF observations should have higher mu_blend than low."""
        hist = _make_historical_df(2)
        pipeline = InSeasonBayesianPipeline(hist)
        pid_high, pid_low = 1000, 1001
        for _ in range(40):
            pipeline.observe(pid_high, "High", 5.0)
            pipeline.observe(pid_low,  "Low",  0.5)
        result = pipeline.blended_ratings(season=2025)
        mu_high = result.filter(pl.col("player_id") == pid_high)["mu_blend"][0]
        mu_low  = result.filter(pl.col("player_id") == pid_low) ["mu_blend"][0]
        assert mu_high > mu_low


# ---------------------------------------------------------------------------
# InSeasonBayesianPipeline — reset
# ---------------------------------------------------------------------------

class TestPipelineReset:

    def test_reset_player(self):
        pipeline = InSeasonBayesianPipeline()
        pipeline.observe(1000, "A", 3.0)
        pipeline.reset_player(1000)
        assert pipeline.games_played(1000) == 0

    def test_reset_all(self):
        pipeline = InSeasonBayesianPipeline()
        pipeline.observe(1000, "A", 3.0)
        pipeline.observe(1001, "B", 2.0)
        pipeline.reset_all()
        assert pipeline.n_players_observed() == 0


# ---------------------------------------------------------------------------
# InSeasonBayesianPipeline — save/load
# ---------------------------------------------------------------------------

class TestPipelineIO:

    def test_save_load_roundtrip(self, tmp_path):
        hist = _make_historical_df(10)
        pipeline = InSeasonBayesianPipeline(hist)
        for i in range(5):
            pipeline.observe(1000 + i, f"P{i}", 2.5)
        path = tmp_path / "pipeline.pkl"
        pipeline.save(path)
        loaded = InSeasonBayesianPipeline.load(path)
        r1 = pipeline.blended_ratings(season=2025)
        r2 = loaded.blended_ratings(season=2025)
        assert len(r1) == len(r2)
        np.testing.assert_array_almost_equal(
            r1.sort("player_id")["mu_blend"].to_numpy(),
            r2.sort("player_id")["mu_blend"].to_numpy(),
            decimal=6,
        )

    def test_write_parquet(self, tmp_path):
        pipeline = InSeasonBayesianPipeline()
        pipeline.observe(1000, "A", 3.0)
        ratings = pipeline.blended_ratings(season=2025)
        path = write_inseason_ratings(ratings, tmp_path / "inseason", 2025, game_number=10)
        assert path.exists()
        loaded = pl.read_parquet(path)
        assert len(loaded) == 1
