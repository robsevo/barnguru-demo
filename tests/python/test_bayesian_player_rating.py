"""Tests for BayesianPlayerRating (Feature 2.9) and SequentialBayesianUpdater (Feature 2.10)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from models.bayesian_player_rating import (
    LEAGUE_MEAN_XGF60,
    LEAGUE_SIGMA_XGF60,
    MODEL_VERSION,
    OBS_SIGMA_XGF60,
    PLAYER_RATING_SCHEMA,
    BayesianPlayerRating,
    PlayerState,
    SequentialBayesianUpdater,
    conjugate_update,
    write_player_ratings,
    read_player_ratings,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_player_game_df(n_players: int = 30, seed: int = 42) -> pl.DataFrame:
    """Synthetic per-player aggregated DataFrame."""
    rng = np.random.default_rng(seed)
    xgf = rng.normal(LEAGUE_MEAN_XGF60, LEAGUE_SIGMA_XGF60, n_players).clip(0.5, 5.0)
    n_games = rng.integers(5, 82, n_players)
    return pl.DataFrame({
        "player_id":    list(range(1000, 1000 + n_players)),
        "player_name":  [f"Player{i}" for i in range(n_players)],
        "xgf_per60":    xgf.tolist(),
        "n_games":      n_games.tolist(),
        "xgf_std":      rng.uniform(0.5, 1.5, n_players).tolist(),
    })


# ===========================================================================
# conjugate_update
# ===========================================================================

class TestConjugateUpdate:

    def test_no_observations_returns_prior(self):
        mu, sigma = conjugate_update(2.5, 0.6, np.array([]), obs_sigma=1.2)
        assert mu == pytest.approx(2.5)
        assert sigma == pytest.approx(0.6)

    def test_posterior_pulls_toward_data(self):
        # Strong signal: 10 obs of 4.0 with informative prior at 2.5
        prior_mu, prior_sigma = 2.5, 0.6
        obs = np.full(10, 4.0)
        post_mu, post_sigma = conjugate_update(prior_mu, prior_sigma, obs, obs_sigma=1.2)
        assert post_mu > prior_mu  # should shift toward obs
        assert post_sigma < prior_sigma  # more data = tighter posterior

    def test_posterior_sigma_decreases_with_more_data(self):
        prior_mu, prior_sigma = 2.5, 0.6
        obs = np.full(1, 3.0)
        _, sigma_1 = conjugate_update(prior_mu, prior_sigma, obs, obs_sigma=1.2)
        obs = np.full(20, 3.0)
        _, sigma_20 = conjugate_update(prior_mu, prior_sigma, obs, obs_sigma=1.2)
        assert sigma_20 < sigma_1

    def test_single_very_noisy_obs_barely_moves_posterior(self):
        prior_mu, prior_sigma = 2.5, 0.1
        obs = np.array([100.0])
        post_mu, _ = conjugate_update(prior_mu, prior_sigma, obs, obs_sigma=100.0)
        # Very noisy observation with tight prior → barely moves
        assert abs(post_mu - prior_mu) < 1.0

    def test_invalid_prior_sigma_raises(self):
        with pytest.raises(ValueError, match="prior_sigma"):
            conjugate_update(2.5, 0.0, np.array([3.0]))

    def test_invalid_obs_sigma_raises(self):
        with pytest.raises(ValueError, match="obs_sigma"):
            conjugate_update(2.5, 0.6, np.array([3.0]), obs_sigma=0.0)

    def test_precision_weighted_average(self):
        """Verify result matches manual precision-weighted formula."""
        prior_mu, prior_sigma, obs_sigma = 2.0, 1.0, 2.0
        obs = np.array([4.0])
        post_mu, post_sigma = conjugate_update(prior_mu, prior_sigma, obs, obs_sigma)

        prior_prec = 1.0 / prior_sigma**2
        obs_prec   = 1.0 / obs_sigma**2
        expected_post_prec = prior_prec + obs_prec
        expected_mu = (prior_mu * prior_prec + 4.0 * obs_prec) / expected_post_prec
        expected_sigma = np.sqrt(1.0 / expected_post_prec)

        assert post_mu    == pytest.approx(expected_mu,    abs=1e-9)
        assert post_sigma == pytest.approx(expected_sigma, abs=1e-9)


# ===========================================================================
# SequentialBayesianUpdater (Feature 2.10)
# ===========================================================================

class TestSequentialBayesianUpdater:

    def test_update_initialises_player(self):
        upd = SequentialBayesianUpdater()
        state = upd.update(player_id=1, player_name="McDavid", observation=3.5)
        assert state.player_id == 1
        assert state.n_games == 1

    def test_update_moves_mean_toward_observation(self):
        upd = SequentialBayesianUpdater(league_mean=2.5)
        # High-scoring observation should push posterior above prior
        state = upd.update(player_id=1, player_name="Sniper", observation=5.0)
        assert state.mu > 2.5

    def test_update_decreases_sigma(self):
        upd = SequentialBayesianUpdater()
        state1 = upd.update(player_id=1, player_name="A", observation=3.0)
        sigma_after_1 = state1.sigma
        state2 = upd.update(player_id=1, player_name="A", observation=3.0)
        assert state2.sigma < sigma_after_1

    def test_n_games_increments(self):
        upd = SequentialBayesianUpdater()
        for i in range(5):
            upd.update(player_id=1, player_name="A", observation=2.5)
        assert upd.get(1).n_games == 5

    def test_multiple_players_independent(self):
        upd = SequentialBayesianUpdater(league_mean=2.5)
        upd.update(player_id=1, player_name="Scorer", observation=5.0)
        upd.update(player_id=2, player_name="Checker", observation=0.5)
        s1 = upd.get(1)
        s2 = upd.get(2)
        assert s1.mu > s2.mu

    def test_get_unknown_player_returns_none(self):
        upd = SequentialBayesianUpdater()
        assert upd.get(999) is None

    def test_reset_single_player(self):
        upd = SequentialBayesianUpdater()
        upd.update(player_id=1, player_name="A", observation=5.0)
        upd.reset(player_id=1)
        assert upd.get(1) is None

    def test_reset_all(self):
        upd = SequentialBayesianUpdater()
        upd.update(player_id=1, player_name="A", observation=5.0)
        upd.update(player_id=2, player_name="B", observation=2.0)
        upd.reset()
        assert len(upd.all_states()) == 0

    def test_update_batch_equivalent_to_sequential(self):
        obs = np.array([3.0, 2.8, 3.2, 2.9])

        upd_seq = SequentialBayesianUpdater(league_mean=2.5, league_sigma=0.6)
        for o in obs:
            upd_seq.update(player_id=1, player_name="A", observation=float(o))

        upd_batch = SequentialBayesianUpdater(league_mean=2.5, league_sigma=0.6)
        upd_batch.update_batch(player_id=1, player_name="A", observations=obs)

        s_seq   = upd_seq.get(1)
        s_batch = upd_batch.get(1)
        assert s_seq.mu    == pytest.approx(s_batch.mu,    abs=1e-6)
        assert s_seq.sigma == pytest.approx(s_batch.sigma, abs=1e-6)
        assert s_seq.n_games == s_batch.n_games

    def test_all_states_sorted_descending(self):
        upd = SequentialBayesianUpdater()
        upd.update(player_id=1, player_name="Low",  observation=0.5)
        upd.update(player_id=2, player_name="High", observation=5.0)
        upd.update(player_id=3, player_name="Mid",  observation=2.5)
        states = upd.all_states()
        mus = [s.mu for s in states]
        assert mus == sorted(mus, reverse=True)

    def test_to_dataframe_schema(self):
        upd = SequentialBayesianUpdater()
        upd.update(player_id=1, player_name="A", observation=3.0)
        upd.update(player_id=2, player_name="B", observation=2.0)
        df = upd.to_dataframe(season=2025)
        for col in PLAYER_RATING_SCHEMA:
            assert col in df.columns

    def test_to_dataframe_empty(self):
        upd = SequentialBayesianUpdater()
        df = upd.to_dataframe(season=2025)
        assert len(df) == 0

    def test_ci_encompasses_posterior_mean(self):
        upd = SequentialBayesianUpdater()
        upd.update(player_id=1, player_name="A", observation=3.0)
        df = upd.to_dataframe(season=2025)
        row = df.row(0, named=True)
        assert row["ci_lower_95"] < row["posterior_mean"] < row["ci_upper_95"]

    def test_save_load_roundtrip(self, tmp_path):
        upd = SequentialBayesianUpdater()
        upd.update(player_id=1, player_name="A", observation=3.5)
        upd.update(player_id=2, player_name="B", observation=1.5)
        path = tmp_path / "updater.pkl"
        upd.save(path)
        loaded = SequentialBayesianUpdater.load(path)
        s1 = loaded.get(1)
        assert s1 is not None
        assert s1.mu == pytest.approx(upd.get(1).mu, abs=1e-9)


# ===========================================================================
# BayesianPlayerRating (Feature 2.9)
# ===========================================================================

class TestBayesianPlayerRating:

    def test_fit_season_returns_dataframe(self):
        df = _make_player_game_df(30)
        model = BayesianPlayerRating()
        result = model.fit_season(df, season=2025)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 30

    def test_output_schema(self):
        df = _make_player_game_df(30)
        model = BayesianPlayerRating()
        result = model.fit_season(df, season=2025)
        for col in PLAYER_RATING_SCHEMA:
            assert col in result.columns, f"Missing column: {col}"

    def test_sorted_descending(self):
        df = _make_player_game_df(30)
        model = BayesianPlayerRating()
        result = model.fit_season(df, season=2025)
        vals = result["posterior_mean"].to_list()
        assert vals == sorted(vals, reverse=True)

    def test_shrinkage_for_low_sample_players(self):
        """Players with fewer games should have posterior closer to prior."""
        df = pl.DataFrame({
            "player_id":   [1, 2],
            "player_name": ["Few", "Many"],
            "xgf_per60":   [5.0, 5.0],   # same observed rate
            "n_games":     [3, 75],        # very different sample sizes
            "xgf_std":     [1.0, 1.0],
        })
        model = BayesianPlayerRating(league_mean=2.5)
        result = model.fit_season(df, season=2025)
        few_row  = result.filter(pl.col("player_id") == 1).row(0, named=True)
        many_row = result.filter(pl.col("player_id") == 2).row(0, named=True)
        # Player with more games should have posterior closer to observed
        assert many_row["posterior_mean"] > few_row["posterior_mean"]

    def test_low_sample_wider_ci(self):
        """Low-GP players should have wider credible intervals."""
        df = pl.DataFrame({
            "player_id":   [1, 2],
            "player_name": ["Few", "Many"],
            "xgf_per60":   [3.0, 3.0],
            "n_games":     [3, 75],
            "xgf_std":     [1.0, 1.0],
        })
        model = BayesianPlayerRating(league_mean=2.5)
        result = model.fit_season(df, season=2025)
        few_width  = result.filter(pl.col("player_id") == 1)["credible_width"][0]
        many_width = result.filter(pl.col("player_id") == 2)["credible_width"][0]
        assert few_width > many_width

    def test_missing_columns_raises(self):
        df = pl.DataFrame({"player_id": [1], "xgf_per60": [2.5]})  # missing n_games
        model = BayesianPlayerRating()
        with pytest.raises(ValueError, match="missing columns"):
            model.fit_season(df, season=2025)

    def test_empty_df_returns_empty(self):
        df = pl.DataFrame({
            "player_id":  pl.Series([], dtype=pl.Int64),
            "player_name": pl.Series([], dtype=pl.Utf8),
            "xgf_per60":  pl.Series([], dtype=pl.Float64),
            "n_games":    pl.Series([], dtype=pl.Int64),
        })
        model = BayesianPlayerRating()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit_season(df, season=2025)
        assert len(result) == 0

    def test_ci_encompasses_posterior_mean(self):
        df = _make_player_game_df(20)
        model = BayesianPlayerRating()
        result = model.fit_season(df, season=2025)
        for row in result.to_dicts():
            assert row["ci_lower_95"] < row["posterior_mean"] < row["ci_upper_95"]

    def test_credible_width_is_2z_sigma(self):
        df = _make_player_game_df(20)
        model = BayesianPlayerRating()
        result = model.fit_season(df, season=2025)
        z95 = 1.959964
        for row in result.to_dicts():
            expected_width = 2 * z95 * row["posterior_sigma"]
            assert row["credible_width"] == pytest.approx(expected_width, abs=1e-6)

    def test_with_rapm_prior_shifts_mean(self):
        """RAPM-informed prior should shift posterior mean."""
        df = _make_player_game_df(10, seed=99)
        pids = df["player_id"].to_list()

        rapm_df = pl.DataFrame({
            "player_id": pids,
            "off_rapm": [2.0] * len(pids),  # strong positive RAPM for all
        })

        model_no_rapm   = BayesianPlayerRating(league_mean=2.5)
        model_with_rapm = BayesianPlayerRating(league_mean=2.5)

        result_no   = model_no_rapm.fit_season(df, season=2025, rapm_df=None)
        result_with = model_with_rapm.fit_season(df, season=2025, rapm_df=rapm_df)

        # With positive RAPM, posterior means should be higher on average
        mean_no   = result_no["posterior_mean"].mean()
        mean_with = result_with["posterior_mean"].mean()
        assert mean_with > mean_no

    def test_model_version_set(self):
        df = _make_player_game_df(10)
        model = BayesianPlayerRating()
        result = model.fit_season(df, season=2025)
        assert (result["model_version"] == MODEL_VERSION).all()

    def test_ratings_property_before_fit_raises(self):
        model = BayesianPlayerRating()
        with pytest.raises(RuntimeError, match="fit_season"):
            model.ratings()

    def test_ratings_property_after_fit(self):
        df = _make_player_game_df(10)
        model = BayesianPlayerRating()
        result = model.fit_season(df, season=2025)
        assert model.ratings().shape == result.shape

    def test_save_load_roundtrip(self, tmp_path):
        df = _make_player_game_df(20)
        model = BayesianPlayerRating()
        model.fit_season(df, season=2025)
        path = tmp_path / "bayes_ratings.pkl"
        model.save(path)
        loaded = BayesianPlayerRating.load(path)
        np.testing.assert_array_almost_equal(
            model.ratings()["posterior_mean"].to_numpy(),
            loaded.ratings()["posterior_mean"].to_numpy(),
            decimal=6,
        )

    def test_write_read_parquet(self, tmp_path):
        df = _make_player_game_df(20)
        model = BayesianPlayerRating()
        ratings = model.fit_season(df, season=2025)
        write_player_ratings(ratings, tmp_path / "bayes_ratings", 2025)
        path = tmp_path / "bayes_ratings" / "player_ratings_2025.parquet"
        assert path.exists()
        loaded = pl.read_parquet(path)
        assert len(loaded) == 20
