"""Unit tests for models/xg_model.py — Feature 2.1.

Tests cover feature engineering and model lifecycle without requiring
a full MoneyPuck shot dataset.  XGBoost is trained on a tiny synthetic
DataFrame so tests run fast and offline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from models.xg_model import (
    XGModel,
    _SHOT_TYPES,
    _danger_zone,
    build_features,
    feature_names,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_shot_df(n: int = 100, seed: int = 42) -> pl.DataFrame:
    """Create a minimal synthetic shot DataFrame matching the MoneyPuck schema."""
    rng = np.random.default_rng(seed)
    shot_types = rng.choice(_SHOT_TYPES, size=n)
    home_skaters = rng.integers(3, 6, size=n)
    away_skaters = rng.integers(3, 6, size=n)
    # Mix of home/away teams for rush proxy
    teams = ["EDM", "CGY", "VAN", "TOR"]
    shooting_team = rng.choice(teams, size=n)
    last_event_team = rng.choice(teams + [""], size=n)

    distance = rng.uniform(5.0, 80.0, size=n)
    adj_y = rng.uniform(-30.0, 30.0, size=n)

    return pl.DataFrame({
        "shot_angle":         rng.uniform(-90.0, 90.0, size=n).tolist(),
        "arena_adj_distance": distance.tolist(),
        "arena_adj_y":        adj_y.tolist(),
        "shot_type":          shot_types.tolist(),
        "last_event_team":    last_event_team.tolist(),
        "shooting_team":      shooting_team.tolist(),
        "home_skaters":       home_skaters.tolist(),
        "away_skaters":       away_skaters.tolist(),
        "period":             rng.integers(1, 4, size=n).tolist(),
        "shooter_hand":       rng.choice(["L", "R"], size=n).tolist(),
        "is_goal":            rng.integers(0, 2, size=n).tolist(),
        "x_goal":             rng.uniform(0.0, 0.25, size=n).tolist(),  # MoneyPuck xG
        # Required by schema but not used in feature engineering
        "shooter_id":         rng.integers(1000, 9999, size=n).tolist(),
        "shooter_name":       ["Player A"] * n,
        "shooting_team":      shooting_team.tolist(),
        "season":             [2024] * n,
        "is_playoff":         [False] * n,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_shape(self) -> None:
        df = _make_shot_df(50)
        X, y = build_features(df)
        expected_cols = len(feature_names())
        assert X.shape == (50, expected_cols), f"Expected (50, {expected_cols}), got {X.shape}"
        assert y is not None
        assert y.shape == (50,)

    def test_no_nan(self) -> None:
        df = _make_shot_df(50)
        X, y = build_features(df)
        assert not np.isnan(X).any(), "X contains NaN values"
        assert not np.isnan(y).any(), "y contains NaN values"  # type: ignore[arg-type]

    def test_no_is_goal_returns_none_y(self) -> None:
        df = _make_shot_df(20).drop("is_goal")
        X, y = build_features(df)
        assert y is None

    def test_missing_required_column_raises(self) -> None:
        df = _make_shot_df(10).drop("shot_angle")
        with pytest.raises(ValueError, match="missing required columns"):
            build_features(df)


class TestDangerZone:
    def test_high_danger(self) -> None:
        # In front of net, close distance, between the dots
        dist   = np.array([10.0, 15.0, 5.0])
        adj_y  = np.array([ 0.0,  5.0, 10.0])
        zones  = _danger_zone(dist, adj_y)
        assert all(z == 2 for z in zones), f"Expected all HIGH, got {zones}"

    def test_low_danger(self) -> None:
        # Point shot far from net
        dist   = np.array([60.0, 70.0])
        adj_y  = np.array([ 0.0,  0.0])
        zones  = _danger_zone(dist, adj_y)
        assert all(z == 0 for z in zones), f"Expected all LOW, got {zones}"

    def test_medium_danger(self) -> None:
        dist  = np.array([30.0])
        adj_y = np.array([ 0.0])
        zones = _danger_zone(dist, adj_y)
        assert zones[0] == 1, f"Expected MED (1), got {zones[0]}"


class TestXGModel:
    def test_predict_proba_range(self) -> None:
        df = _make_shot_df(200)
        model = XGModel()
        train = df.head(160)
        test  = df.tail(40)
        model.fit(train)
        proba = model.predict_proba(test)
        assert proba.shape == (40,)
        assert proba.min() >= 0.0, "Probabilities below 0"
        assert proba.max() <= 1.0, "Probabilities above 1"

    def test_fit_returns_metrics(self) -> None:
        df = _make_shot_df(300)
        model = XGModel()
        metrics = model.fit(df.head(240), eval_df=df.tail(60))
        assert "brier_score" in metrics
        assert "auc_roc" in metrics
        assert "moneypuck_correlation" in metrics
        assert 0.0 <= metrics["brier_score"] <= 1.0
        # AUC may be NaN on tiny dataset with no class variation; just check it's a float
        assert isinstance(metrics["auc_roc"], float)

    def test_save_load_roundtrip(self) -> None:
        df = _make_shot_df(200)
        model = XGModel()
        model.fit(df.head(160))
        proba_before = model.predict_proba(df.tail(40))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xg_model_test.pkl"
            model.save(path)
            loaded = XGModel.load(path)

        proba_after = loaded.predict_proba(df.tail(40))
        np.testing.assert_allclose(proba_before, proba_after, rtol=1e-5)

    def test_untrained_predict_raises(self) -> None:
        model = XGModel()
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict_proba(_make_shot_df(10))

    def test_feature_importance_returns_dict(self) -> None:
        df = _make_shot_df(200)
        model = XGModel()
        model.fit(df)
        fi = model.feature_importance()
        assert isinstance(fi, dict)
        assert len(fi) > 0
        # All values should be non-negative
        assert all(v >= 0 for v in fi.values())
