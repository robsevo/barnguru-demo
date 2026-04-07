"""Tests for Feature 2.17 — IncrementalRetrainer."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from models.incremental_retrainer import (
    IncrementalRetrainer,
    RetrainResult,
    season_weight,
    _build_weight_array,
    write_retrain_log,
    SEASON_DECAY,
    RETRAIN_LOG_SCHEMA,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic data matching xG / Goalie model expectations
# ---------------------------------------------------------------------------

def _make_shots_df(n: int = 200, seed: int = 42) -> pl.DataFrame:
    """Minimal shot DataFrame matching xG model build_features() requirements."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "shot_angle":          rng.uniform(0, 90, n).tolist(),
        "arena_adj_distance":  rng.uniform(5, 80, n).tolist(),
        "shot_type":           rng.choice(["WRIST", "SLAP", "SNAP", "BACKHAND", "TIP"], n).tolist(),
        "arena_adj_y":         rng.uniform(-42, 42, n).tolist(),
        "last_event_team":     rng.choice(["EDM", "TOR"], n).tolist(),
        "shooting_team":       rng.choice(["EDM", "TOR"], n).tolist(),
        "home_skaters":        rng.integers(4, 7, n).tolist(),
        "away_skaters":        rng.integers(4, 7, n).tolist(),
        "period":              rng.integers(1, 4, n).tolist(),
        "shooter_hand":        rng.choice(["L", "R"], n).tolist(),
        "x_on_goal":           rng.uniform(0, 0.3, n).tolist(),
        "game_id":             rng.integers(1000, 2000, n).tolist(),
        "seconds_since_last":  rng.uniform(0, 60, n).tolist(),
        "is_goal":             rng.integers(0, 2, n).tolist(),
    })


def _make_goalie_df(n: int = 50, seed: int = 0, season: int = 2025) -> pl.DataFrame:
    """Minimal goalie DataFrame matching GoalieModel._build_goalie_features() schema."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "player_id":       rng.integers(100_000, 999_999, n).tolist(),
        "player_name":     [f"goalie_{i}" for i in range(n)],
        "team":            ["TOR"] * n,
        "situation":       ["all"] * n,
        "season":          [season] * n,
        "games_played":    rng.integers(10, 82, n).tolist(),
        "icetime_seconds": rng.uniform(30_000, 200_000, n).tolist(),
        "shots":           rng.integers(500, 2000, n).tolist(),
        "saves":           rng.integers(450, 1900, n).tolist(),
        "xga":             rng.uniform(40.0, 150.0, n).tolist(),
        "gsax":            rng.uniform(-10.0, 10.0, n).tolist(),
        "hd_shots":        rng.integers(50, 300, n).tolist(),
        "md_shots":        rng.integers(100, 600, n).tolist(),
        "hdsv_pct":        rng.uniform(0.75, 0.90, n).tolist(),
        "mdsv_pct":        rng.uniform(0.85, 0.95, n).tolist(),
        "ldsv_pct":        rng.uniform(0.95, 0.99, n).tolist(),
        "sv_pct":          rng.uniform(0.88, 0.94, n).tolist(),
        "gsax_per60":      rng.uniform(-1.0, 1.0, n).tolist(),
    })


def _train_xg_model(shots_df: pl.DataFrame):
    from models.xg_model import XGModel
    m = XGModel()
    m.fit(shots_df)
    return m


def _train_goalie_model(goalie_df: pl.DataFrame):
    from models.goalie_model import GoalieModel
    m = GoalieModel()
    m.fit([goalie_df], [2025])
    return m


# ---------------------------------------------------------------------------
# season_weight
# ---------------------------------------------------------------------------

class TestSeasonWeight:
    def test_current_season_is_one(self):
        assert season_weight(2025, 2025) == pytest.approx(1.0)

    def test_one_season_lag(self):
        assert season_weight(2025, 2024) == pytest.approx(SEASON_DECAY)

    def test_two_season_lag(self):
        assert season_weight(2025, 2023) == pytest.approx(SEASON_DECAY ** 2)

    def test_three_season_lag(self):
        assert season_weight(2025, 2022) == pytest.approx(SEASON_DECAY ** 3)

    def test_future_season_treated_as_current(self):
        # gap = max(0, ...) → 0 if data_season > current_season
        assert season_weight(2025, 2026) == pytest.approx(1.0)

    def test_custom_decay(self):
        assert season_weight(2025, 2024, decay=0.5) == pytest.approx(0.5)


class TestBuildWeightArray:
    def test_shape_preserved(self):
        seasons = np.array([2023, 2024, 2025])
        w = _build_weight_array(seasons, current_season=2025)
        assert len(w) == 3

    def test_current_season_weight_is_one(self):
        seasons = np.array([2025])
        w = _build_weight_array(seasons, current_season=2025)
        assert w[0] == pytest.approx(1.0)

    def test_prior_seasons_decayed(self):
        seasons = np.array([2023, 2024, 2025])
        w = _build_weight_array(seasons, current_season=2025)
        assert w[0] < w[1] < w[2]


# ---------------------------------------------------------------------------
# RetrainResult
# ---------------------------------------------------------------------------

class TestRetrainResult:
    def _make(self, before, after, metric="auc_roc", accepted=True):
        return RetrainResult(
            model_type="xg", current_season=2025,
            metric_name=metric, metric_before=before, metric_after=after,
            n_new_trees=50, n_seasons_used=3, accepted=accepted,
        )

    def test_improvement_auc_positive(self):
        r = self._make(0.75, 0.76, metric="auc_roc")
        assert r.improvement() == pytest.approx(0.01)

    def test_improvement_auc_negative(self):
        r = self._make(0.76, 0.74, metric="auc_roc")
        assert r.improvement() == pytest.approx(-0.02)

    def test_improvement_rmse_lower_is_better(self):
        # RMSE dropped 0.75→0.70 → improvement = 0.05
        r = self._make(0.75, 0.70, metric="rmse")
        assert r.improvement() == pytest.approx(0.05)

    def test_str_contains_key_info(self):
        r = self._make(0.75, 0.76)
        s = str(r)
        assert "xg" in s
        assert "2025" in s
        assert "auc_roc" in s


# ---------------------------------------------------------------------------
# IncrementalRetrainer — xG
# ---------------------------------------------------------------------------

class TestRetrainXG:
    def test_returns_xgmodel_and_result(self):
        from models.xg_model import XGModel

        shots = _make_shots_df(300)
        model = _train_xg_model(shots)
        retrainer = IncrementalRetrainer(n_new_trees=5)

        new_model, result = retrainer.retrain_xg(
            model          = model,
            shots_data     = {2025: shots},
            current_season = 2025,
            force          = True,
        )
        assert isinstance(new_model, XGModel)
        assert isinstance(result, RetrainResult)
        assert result.model_type == "xg"
        assert result.accepted is True

    def test_accepted_when_forced(self):
        shots = _make_shots_df(300)
        model = _train_xg_model(shots)
        retrainer = IncrementalRetrainer(n_new_trees=5)
        _, result = retrainer.retrain_xg(
            model=model, shots_data={2025: shots},
            current_season=2025, force=True,
        )
        assert result.accepted

    def test_multi_season_pool(self):
        shots_2024 = _make_shots_df(150, seed=1)
        shots_2025 = _make_shots_df(150, seed=2)
        model = _train_xg_model(pl.concat([shots_2024, shots_2025]))
        retrainer = IncrementalRetrainer(n_new_trees=5)
        _, result = retrainer.retrain_xg(
            model          = model,
            shots_data     = {2024: shots_2024, 2025: shots_2025},
            current_season = 2025,
            force          = True,
        )
        assert result.n_seasons_used == 2

    def test_empty_shots_data_warns_and_returns_original(self):
        shots = _make_shots_df(200)
        model = _train_xg_model(shots)
        retrainer = IncrementalRetrainer(n_new_trees=5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            returned_model, result = retrainer.retrain_xg(
                model          = model,
                shots_data     = {2025: pl.DataFrame()},
                current_season = 2025,
            )
        assert not result.accepted
        # Original model returned
        assert returned_model is model

    def test_log_populated_after_retrain(self):
        shots = _make_shots_df(200)
        model = _train_xg_model(shots)
        retrainer = IncrementalRetrainer(n_new_trees=5)
        retrainer.retrain_xg(model=model, shots_data={2025: shots},
                              current_season=2025, force=True)
        assert len(retrainer._log) == 1

    def test_metric_before_after_finite(self):
        shots = _make_shots_df(400)
        model = _train_xg_model(shots)
        retrainer = IncrementalRetrainer(n_new_trees=5)
        _, result = retrainer.retrain_xg(
            model=model, shots_data={2025: shots},
            current_season=2025, force=True,
        )
        assert np.isfinite(result.metric_before)
        assert np.isfinite(result.metric_after)


# ---------------------------------------------------------------------------
# IncrementalRetrainer — Goalie
# ---------------------------------------------------------------------------

class TestRetrainGoalie:
    def test_returns_goalie_model_and_result(self):
        from models.goalie_model import GoalieModel

        goalie = _make_goalie_df(50)
        model = _train_goalie_model(goalie)
        retrainer = IncrementalRetrainer(n_new_trees=5)
        new_model, result = retrainer.retrain_goalie(
            model=model, goalie_data={2025: goalie},
            current_season=2025, force=True,
        )
        assert isinstance(new_model, GoalieModel)
        assert result.model_type == "goalie"
        assert result.accepted

    def test_metric_name_is_rmse(self):
        goalie = _make_goalie_df(50)
        model = _train_goalie_model(goalie)
        retrainer = IncrementalRetrainer(n_new_trees=5)
        _, result = retrainer.retrain_goalie(
            model=model, goalie_data={2025: goalie},
            current_season=2025, force=True,
        )
        assert result.metric_name == "rmse"

    def test_empty_goalie_data_returns_original(self):
        goalie = _make_goalie_df(50)
        model = _train_goalie_model(goalie)
        retrainer = IncrementalRetrainer(n_new_trees=5)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            returned, result = retrainer.retrain_goalie(
                model=model, goalie_data={2025: pl.DataFrame()},
                current_season=2025,
            )
        assert returned is model
        assert not result.accepted

    def test_multi_season_weights(self):
        goalie_2024 = _make_goalie_df(50, seed=10, season=2024)
        goalie_2025 = _make_goalie_df(50, seed=11, season=2025)
        model = _train_goalie_model(pl.concat([goalie_2024, goalie_2025]))
        retrainer = IncrementalRetrainer(n_new_trees=5)
        _, result = retrainer.retrain_goalie(
            model=model,
            goalie_data={2024: goalie_2024, 2025: goalie_2025},
            current_season=2025,
            force=True,
        )
        assert result.n_seasons_used == 2


# ---------------------------------------------------------------------------
# log_dataframe
# ---------------------------------------------------------------------------

class TestLogDataframe:
    def test_empty_log(self):
        r = IncrementalRetrainer()
        df = r.log_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        for col in RETRAIN_LOG_SCHEMA:
            assert col in df.columns

    def test_populated_after_retrain(self):
        shots = _make_shots_df(200)
        model = _train_xg_model(shots)
        r = IncrementalRetrainer(n_new_trees=5)
        r.retrain_xg(model=model, shots_data={2025: shots},
                     current_season=2025, force=True)
        df = r.log_dataframe()
        assert len(df) == 1
        assert df["model_type"][0] == "xg"

    def test_schema_correct(self):
        shots = _make_shots_df(200)
        model = _train_xg_model(shots)
        r = IncrementalRetrainer(n_new_trees=5)
        r.retrain_xg(model=model, shots_data={2025: shots},
                     current_season=2025, force=True)
        df = r.log_dataframe()
        for col in RETRAIN_LOG_SCHEMA:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        r = IncrementalRetrainer(decay=0.5, n_new_trees=20, accept_threshold=0.002)
        shots = _make_shots_df(200)
        model = _train_xg_model(shots)
        r.retrain_xg(model=model, shots_data={2025: shots},
                     current_season=2025, force=True)

        path = tmp_path / "retrainer.pkl"
        r.save(path)
        r2 = IncrementalRetrainer.load(path)

        assert r2._decay == pytest.approx(0.5)
        assert r2._n_new_trees == 20
        assert r2._accept_threshold == pytest.approx(0.002)
        assert len(r2._log) == 1

    def test_write_retrain_log(self, tmp_path):
        shots = _make_shots_df(200)
        model = _train_xg_model(shots)
        r = IncrementalRetrainer(n_new_trees=5)
        r.retrain_xg(model=model, shots_data={2025: shots},
                     current_season=2025, force=True)
        df = r.log_dataframe()
        path = write_retrain_log(df, tmp_path, "xg")
        assert path.exists()
        loaded = pl.read_parquet(path)
        assert len(loaded) == 1
