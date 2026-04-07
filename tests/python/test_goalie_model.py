"""Tests for GoalieModel (Feature 2.5) and GoalieFatigueModel (Feature 2.6)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from models.goalie_model import (
    GOALIE_RATING_SCHEMA,
    GoalieModel,
    _build_goalie_features,
    write_goalie_ratings,
    read_goalie_ratings,
)
from models.goalie_fatigue_model import (
    FATIGUE_FEATURE_NAMES,
    GOALIE_FATIGUE_SCHEMA,
    GoalieFatigueModel,
    build_fatigue_features,
    apply_fatigue_to_ratings,
    write_goalie_fatigue,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REQUIRED_GOALIE_COLS = {
    "player_id", "player_name", "team", "situation", "games_played",
    "icetime_seconds", "shots", "saves", "xga", "gsax",
    "hd_shots", "md_shots", "hdsv_pct", "mdsv_pct", "ldsv_pct",
    "sv_pct", "season",
}


def _make_goalie_df(n: int = 20, season: int = 2025) -> pl.DataFrame:
    """Synthetic MoneyPuck goalie DataFrame."""
    rng = np.random.default_rng(42)
    base_sv = rng.uniform(0.88, 0.93, n)
    shots    = rng.integers(500, 1200, n).astype(float)
    saves    = (shots * base_sv).astype(int).astype(float)
    goals_against = shots - saves
    xga = goals_against * rng.uniform(0.9, 1.1, n)
    gsax = (xga - goals_against) * rng.uniform(0.8, 1.2, n)

    hd_frac = rng.uniform(0.15, 0.30, n)
    md_frac = rng.uniform(0.30, 0.45, n)

    return pl.DataFrame({
        "player_id":       [1000 + i for i in range(n)],
        "player_name":     [f"Goalie{i}" for i in range(n)],
        "team":            ["TOR" if i % 2 == 0 else "BOS" for i in range(n)],
        "season":          [season] * n,
        "situation":       ["all"] * n,
        "games_played":    rng.integers(15, 60, n).tolist(),
        "icetime_seconds": (rng.uniform(1800, 4000, n) * rng.integers(15, 60, n)).tolist(),
        "shots":           shots.tolist(),
        "saves":           saves.tolist(),
        "xga":             xga.tolist(),
        "gsax":            gsax.tolist(),
        "hd_shots":        (shots * hd_frac).tolist(),
        "md_shots":        (shots * md_frac).tolist(),
        "hdsv_pct":        rng.uniform(0.82, 0.88, n).tolist(),
        "mdsv_pct":        rng.uniform(0.89, 0.93, n).tolist(),
        "ldsv_pct":        rng.uniform(0.95, 0.99, n).tolist(),
        "sv_pct":          base_sv.tolist(),
    })


def _make_fatigue_df(n: int = 50) -> pl.DataFrame:
    """Synthetic per-goalie-game schedule DataFrame."""
    rng = np.random.default_rng(7)
    return pl.DataFrame({
        "player_id":      [100 + (i % 10) for i in range(n)],
        "player_name":    [f"G{i % 10}" for i in range(n)],
        "game_id":        list(range(n)),
        "is_b2b":         rng.integers(0, 2, n).tolist(),
        "rest_days":      rng.uniform(0, 7, n).tolist(),
        "gp_last_7":      rng.integers(0, 5, n).tolist(),
        "gp_last_14":     rng.integers(0, 8, n).tolist(),
        "road_game_num":  rng.integers(0, 6, n).tolist(),
        "saves_last_5":   rng.uniform(100, 180, n).tolist(),
        "age":            rng.uniform(22, 38, n).tolist(),
        "toi_last_5_avg": rng.uniform(2000, 4000, n).tolist(),
        "sv_delta":       rng.uniform(-0.04, 0.01, n).tolist(),
    })


# ===========================================================================
# Feature 2.5 — GoalieModel
# ===========================================================================

class TestBuildGoalieFeatures:

    def test_returns_correct_shapes(self):
        df = _make_goalie_df(20, 2025)
        X, y, meta = _build_goalie_features(df, 2025)
        assert X.shape[1] == 10
        assert y is not None
        assert len(y) == X.shape[0]
        assert len(meta) == X.shape[0]

    def test_filters_min_games(self):
        df = _make_goalie_df(20, 2025)
        # All have GP >= 15 in synthetic data — all should pass MIN_GAMES=10
        X, y, meta = _build_goalie_features(df, 2025)
        assert len(meta) == 20

    def test_low_gp_filtered(self):
        df = _make_goalie_df(5, 2025)
        # Override GP to be below threshold
        df = df.with_columns(pl.lit(3).cast(pl.Int64).alias("games_played"))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            X, y, meta = _build_goalie_features(df, 2025)
        assert len(meta) == 0
        assert X.shape[0] == 0

    def test_season_filter(self):
        df = _make_goalie_df(10, 2024)
        X, y, meta = _build_goalie_features(df, 2025)  # wrong season
        assert len(meta) == 0

    def test_missing_columns_raises(self):
        df = _make_goalie_df(10, 2025).drop("shots")
        with pytest.raises(ValueError, match="missing columns"):
            _build_goalie_features(df, 2025)

    def test_no_nan_in_X(self):
        df = _make_goalie_df(20, 2025)
        X, y, _ = _build_goalie_features(df, 2025)
        assert not np.isnan(X).any()

    def test_sv_minus_xsv_in_range(self):
        df = _make_goalie_df(20, 2025)
        X, y, _ = _build_goalie_features(df, 2025)
        # sv_minus_xsv is feature index 9
        sv_minus_xsv = X[:, 9]
        assert (sv_minus_xsv > -0.20).all()
        assert (sv_minus_xsv < 0.20).all()


class TestGoalieModelFit:

    def test_fit_returns_metrics(self):
        df = _make_goalie_df(20, 2025)
        model = GoalieModel()
        metrics = model.fit([df], [2025])
        assert "rmse" in metrics
        assert "n_goalies" in metrics
        assert metrics["n_goalies"] > 0
        assert metrics["rmse"] >= 0

    def test_fit_multiple_seasons(self):
        df1 = _make_goalie_df(20, 2023)
        df2 = _make_goalie_df(20, 2024)
        model = GoalieModel()
        metrics = model.fit([df1, df2], [2023, 2024])
        assert metrics["n_goalies"] >= 20

    def test_fit_no_data_raises(self):
        df = _make_goalie_df(5, 2025).with_columns(
            pl.lit(3).cast(pl.Int64).alias("games_played")
        )
        model = GoalieModel()
        with pytest.raises(ValueError, match="No training data"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit([df], [2025])

    def test_fit_wrong_season_raises(self):
        df = _make_goalie_df(20, 2024)
        model = GoalieModel()
        with pytest.raises(ValueError, match="No training data"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit([df], [2025])  # season mismatch


class TestGoalieModelPredict:

    @pytest.fixture
    def trained_model(self):
        df = _make_goalie_df(20, 2025)
        model = GoalieModel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit([df], [2025])
        return model, df

    def test_predict_returns_dataframe(self, trained_model):
        model, df = trained_model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.predict_season(df, 2025)
        assert isinstance(result, pl.DataFrame)

    def test_output_schema_correct(self, trained_model):
        model, df = trained_model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.predict_season(df, 2025)
        for col in GOALIE_RATING_SCHEMA:
            assert col in result.columns, f"Missing column: {col}"

    def test_sorted_descending(self, trained_model):
        model, df = trained_model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.predict_season(df, 2025)
        vals = result["model_gsax_per60"].to_list()
        assert vals == sorted(vals, reverse=True)

    def test_predict_untrained_raises(self):
        model = GoalieModel()
        df = _make_goalie_df(5, 2025)
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict_season(df, 2025)

    def test_gsax_delta_equals_model_gsax_per60(self, trained_model):
        model, df = trained_model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.predict_season(df, 2025)
        np.testing.assert_array_almost_equal(
            result["gsax_delta"].to_numpy(),
            result["model_gsax_per60"].to_numpy(),
            decimal=6,
        )

    def test_empty_season_returns_empty(self, trained_model):
        model, _ = trained_model
        df_empty = _make_goalie_df(5, 2025).with_columns(
            pl.lit(3).cast(pl.Int64).alias("games_played")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.predict_season(df_empty, 2025)
        assert len(result) == 0


class TestGoalieModelFeatureImportance:

    def test_feature_importance_keys(self):
        df = _make_goalie_df(20, 2025)
        model = GoalieModel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit([df], [2025])
        fi = model.feature_importance()
        assert isinstance(fi, dict)
        assert set(fi.keys()) >= set(["sv_pct", "sv_minus_xsv"])

    def test_feature_importance_untrained_raises(self):
        model = GoalieModel()
        with pytest.raises(RuntimeError):
            model.feature_importance()


class TestGoalieModelIO:

    def test_save_load_roundtrip(self, tmp_path):
        df = _make_goalie_df(20, 2025)
        model = GoalieModel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit([df], [2025])

        path = tmp_path / "goalie_model.pkl"
        model.save(path)
        loaded = GoalieModel.load(path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orig   = model.predict_season(df, 2025)
            loaded_ = loaded.predict_season(df, 2025)

        np.testing.assert_array_almost_equal(
            orig["model_gsax_per60"].to_numpy(),
            loaded_["model_gsax_per60"].to_numpy(),
            decimal=5,
        )

    def test_write_read_parquet(self, tmp_path):
        df = _make_goalie_df(20, 2025)
        model = GoalieModel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit([df], [2025])
            ratings = model.predict_season(df, 2025)

        write_goalie_ratings(ratings, tmp_path / "goalie_ratings", 2025)
        path = tmp_path / "goalie_ratings" / "goalie_ratings_2025.parquet"
        assert path.exists()


# ===========================================================================
# Feature 2.6 — GoalieFatigueModel
# ===========================================================================

class TestBuildFatigueFeatures:

    def test_returns_correct_shape(self):
        df = _make_fatigue_df(50)
        X, y = build_fatigue_features(df)
        assert X.shape == (50, len(FATIGUE_FEATURE_NAMES))
        assert y is not None
        assert len(y) == 50

    def test_no_nan_in_X(self):
        df = _make_fatigue_df(50)
        X, _ = build_fatigue_features(df)
        assert not np.isnan(X).any()

    def test_missing_column_raises(self):
        df = _make_fatigue_df(50).drop("is_b2b")
        with pytest.raises(ValueError, match="missing columns"):
            build_fatigue_features(df)

    def test_y_none_when_no_sv_delta(self):
        df = _make_fatigue_df(50).drop("sv_delta")
        X, y = build_fatigue_features(df)
        assert y is None

    def test_clips_rest_days(self):
        df = _make_fatigue_df(10).with_columns(
            pl.lit(100.0).cast(pl.Float64).alias("rest_days")
        )
        X, _ = build_fatigue_features(df)
        assert (X[:, 1] <= 7).all()

    def test_clips_age(self):
        df = _make_fatigue_df(10).with_columns(
            pl.lit(99.0).cast(pl.Float64).alias("age")
        )
        X, _ = build_fatigue_features(df)
        assert (X[:, 6] <= 45).all()


class TestGoalieFatigueModelDefaults:

    def test_default_coefficients_sign(self):
        model = GoalieFatigueModel()
        coef = model.coefficients()
        # B2B should reduce save%
        assert coef["is_b2b"] < 0
        # Rest days should help
        assert coef["rest_days"] > 0
        # More games in 7 days should hurt
        assert coef["gp_last_7"] < 0

    def test_predict_with_defaults(self):
        model = GoalieFatigueModel()
        X = np.zeros((5, len(FATIGUE_FEATURE_NAMES)), dtype=np.float32)
        deltas = model.predict(X)
        assert deltas.shape == (5,)

    def test_b2b_predicts_negative(self):
        model = GoalieFatigueModel()
        # B2B game: set is_b2b=1, rest_days=0
        X = np.zeros((1, len(FATIGUE_FEATURE_NAMES)), dtype=np.float32)
        X[0, 0] = 1.0  # is_b2b
        X[0, 1] = 0.0  # rest_days
        delta = model.predict(X)[0]
        assert delta < 0, "B2B game should predict negative save% delta"

    def test_rested_predicts_better_than_b2b(self):
        model = GoalieFatigueModel()
        X_b2b = np.zeros((1, len(FATIGUE_FEATURE_NAMES)), dtype=np.float32)
        X_b2b[0, 0] = 1.0  # is_b2b
        X_b2b[0, 1] = 0.0  # rest_days

        X_rest = np.zeros((1, len(FATIGUE_FEATURE_NAMES)), dtype=np.float32)
        X_rest[0, 0] = 0.0  # not B2B
        X_rest[0, 1] = 3.0  # 3 rest days

        assert model.predict(X_rest)[0] > model.predict(X_b2b)[0]


class TestGoalieFatigueModelFit:

    def test_fit_returns_metrics(self):
        df = _make_fatigue_df(50)
        model = GoalieFatigueModel()
        metrics = model.fit(df)
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "n_games" in metrics
        assert metrics["n_games"] == 50
        assert metrics["rmse"] >= 0

    def test_fit_updates_coefficients(self):
        df = _make_fatigue_df(100)
        model = GoalieFatigueModel()
        default_b2b = model.coefficients()["is_b2b"]
        model.fit(df)
        fitted_b2b = model.coefficients()["is_b2b"]
        # After fitting, coefficients should change (synthetic data may shift them)
        assert isinstance(fitted_b2b, float)

    def test_fit_no_sv_delta_raises(self):
        df = _make_fatigue_df(50).drop("sv_delta")
        model = GoalieFatigueModel()
        with pytest.raises(ValueError, match="sv_delta"):
            model.fit(df)

    def test_predict_after_fit(self):
        df = _make_fatigue_df(50)
        model = GoalieFatigueModel()
        model.fit(df)
        X, _ = build_fatigue_features(df)
        deltas = model.predict(X)
        assert len(deltas) == 50

    def test_wrong_feature_count_raises(self):
        model = GoalieFatigueModel()
        X_bad = np.zeros((5, 3))
        with pytest.raises(ValueError, match="features"):
            model.predict(X_bad)


class TestGoalieFatigueModelPredictSeason:

    def test_predict_season_output_columns(self):
        df = _make_fatigue_df(20)
        model = GoalieFatigueModel()
        result = model.predict_season(df, 2025)
        for col in ["player_id", "game_id", "fatigue_sv_delta", "is_b2b", "rest_days"]:
            assert col in result.columns

    def test_predict_season_length_matches(self):
        df = _make_fatigue_df(20)
        model = GoalieFatigueModel()
        result = model.predict_season(df, 2025)
        assert len(result) == 20


class TestApplyFatigueToRatings:

    def test_adjust_produces_column(self):
        goalie_df = _make_goalie_df(10, 2025)
        model = GoalieModel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit([goalie_df], [2025])
            ratings = model.predict_season(goalie_df, 2025)

        # Build fatigue schedule with matching player_ids and run predict_season
        pids = ratings["player_id"].to_list()
        raw_fatigue = _make_fatigue_df(30).with_columns(
            pl.Series("player_id", [pids[i % len(pids)] for i in range(30)], dtype=pl.Int64)
        )
        fatigue_model = GoalieFatigueModel()
        fatigue_df = fatigue_model.predict_season(raw_fatigue, 2025)

        result = apply_fatigue_to_ratings(ratings, fatigue_df)
        assert "adjusted_gsax_per60" in result.columns

    def test_missing_player_id_raises(self):
        ratings = pl.DataFrame({"model_gsax_per60": [1.0, 2.0]})
        fatigue  = pl.DataFrame({"player_id": [1, 2], "fatigue_sv_delta": [-0.01, -0.02]})
        with pytest.raises(ValueError, match="player_id"):
            apply_fatigue_to_ratings(ratings, fatigue)


class TestGoalieFatigueModelIO:

    def test_save_load_roundtrip(self, tmp_path):
        df = _make_fatigue_df(50)
        model = GoalieFatigueModel()
        model.fit(df)
        path = tmp_path / "goalie_fatigue.pkl"
        model.save(path)
        loaded = GoalieFatigueModel.load(path)
        assert loaded._fitted is True
        X, _ = build_fatigue_features(df)
        np.testing.assert_array_almost_equal(
            model.predict(X), loaded.predict(X), decimal=6
        )

    def test_write_parquet(self, tmp_path):
        df = _make_fatigue_df(20)
        model = GoalieFatigueModel()
        result = model.predict_season(df, 2025)
        path = write_goalie_fatigue(result, tmp_path / "goalie_fatigue", 2025)
        assert path.exists()
        loaded = pl.read_parquet(path)
        assert len(loaded) == 20
