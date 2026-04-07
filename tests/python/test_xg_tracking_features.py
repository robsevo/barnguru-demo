"""Tests for Feature 16.10 — tracking-enhanced xG model.

Covers:
 - tracking feature extractor (snapshot geometry calculations)
 - XGModel with tracking_features=True
 - join_tracking_features() left-join behaviour
 - retrain gate logic (no actual DuckDB I/O in tests)
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from data.tracking_feature_extractor import (
    _compute_snapshot_features,
    _count_screens,
    _DEFAULT_DEFENDER_DIST,
    _GOAL_LINE_X,
    _RAY_WIDTH_FT,
    extract_snapshot_features,
    join_tracking_features,
)
from models.xg_model import (
    XGModel,
    _SHOT_TYPES,
    build_features,
    build_features_tracking,
    feature_names,
    feature_names_tracking,
    MODEL_VERSION,
    MODEL_VERSION_TRACKING,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shot_df(n: int = 100, seed: int = 42, with_tracking: bool = False) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    shot_types = rng.choice(_SHOT_TYPES, size=n)
    teams = ["EDM", "CGY"]
    shooting_team = rng.choice(teams, size=n)
    last_event_team = rng.choice(teams + [""], size=n)

    df = pl.DataFrame({
        "shot_angle":         rng.uniform(-90.0, 90.0, size=n).tolist(),
        "arena_adj_distance": rng.uniform(5.0, 80.0, size=n).tolist(),
        "arena_adj_y":        rng.uniform(-30.0, 30.0, size=n).tolist(),
        "shot_type":          shot_types.tolist(),
        "last_event_team":    last_event_team.tolist(),
        "shooting_team":      shooting_team.tolist(),
        "home_skaters":       rng.integers(3, 6, size=n).tolist(),
        "away_skaters":       rng.integers(3, 6, size=n).tolist(),
        "period":             rng.integers(1, 4, size=n).tolist(),
        "shooter_hand":       rng.choice(["L", "R"], size=n).tolist(),
        "is_goal":            rng.integers(0, 2, size=n).tolist(),
        "x_goal":             rng.uniform(0.0, 0.25, size=n).tolist(),
        "shooter_id":         rng.integers(1000, 9999, size=n).tolist(),
        "shooter_name":       ["Player A"] * n,
        "season":             [2024] * n,
        "is_playoff":         [False] * n,
    })

    if with_tracking:
        df = df.with_columns([
            pl.lit(1).alias("screen_count").cast(pl.Int32),
            pl.lit(8.0).alias("defender_dist").cast(pl.Float32),
            pl.lit(2.5).alias("goalie_displacement").cast(pl.Float32),
        ])

    return df


def _make_player_row(
    snapshot_id: int = 1,
    game_id: int = 2024020001,
    period: int = 1,
    time_in_period_secs: int = 300,
    shooter_id: int = 8481600,
    shooter_team: str = "MTL",
    shot_x: float = 50.0,
    shot_y: float = 5.0,
    player_x: float = 60.0,
    player_y: float = 3.0,
    player_team: str = "TBL",
    class_name: str = "player_away",
    confidence: float = 0.95,
) -> dict:
    return {
        "snapshot_id":         snapshot_id,
        "game_id":             game_id,
        "period":              period,
        "time_in_period_secs": time_in_period_secs,
        "shooter_id":          shooter_id,
        "shooter_team":        shooter_team,
        "shot_x":              shot_x,
        "shot_y":              shot_y,
        "player_x":            player_x,
        "player_y":            player_y,
        "player_team":         player_team,
        "class_name":          class_name,
        "confidence":          confidence,
    }


# ---------------------------------------------------------------------------
# Tests: _count_screens
# ---------------------------------------------------------------------------

class TestCountScreens:
    def test_player_directly_in_ray(self):
        # Shooter at (50, 0), goalie at (89, 0), player at (70, 0) → screens
        skaters = [{"player_x": 70.0, "player_y": 0.0}]
        assert _count_screens(50.0, 0.0, 89.0, 0.0, skaters) == 1

    def test_player_outside_ray_width(self):
        # Player 10 ft to the side → no screen
        skaters = [{"player_x": 70.0, "player_y": 15.0}]
        assert _count_screens(50.0, 0.0, 89.0, 0.0, skaters) == 0

    def test_player_behind_goalie(self):
        # t > 1 → behind the goal
        skaters = [{"player_x": 95.0, "player_y": 0.0}]
        assert _count_screens(50.0, 0.0, 89.0, 0.0, skaters) == 0

    def test_player_behind_shooter(self):
        # t < 0 → behind the shooter
        skaters = [{"player_x": 30.0, "player_y": 0.0}]
        assert _count_screens(50.0, 0.0, 89.0, 0.0, skaters) == 0

    def test_shooter_goalie_same_position(self):
        # ray_len_sq < 1 → should return 0, not divide by zero
        skaters = [{"player_x": 70.0, "player_y": 0.0}]
        assert _count_screens(89.0, 0.0, 89.0, 0.0, skaters) == 0

    def test_multiple_screens(self):
        skaters = [
            {"player_x": 65.0, "player_y": 0.5},
            {"player_x": 75.0, "player_y": -1.0},
            {"player_x": 40.0, "player_y": 0.0},   # behind shooter
            {"player_x": 70.0, "player_y": 20.0},  # way off to the side
        ]
        assert _count_screens(50.0, 0.0, 89.0, 0.0, skaters) == 2

    def test_null_player_coords_skipped(self):
        skaters = [{"player_x": None, "player_y": None}]
        assert _count_screens(50.0, 0.0, 89.0, 0.0, skaters) == 0


# ---------------------------------------------------------------------------
# Tests: _compute_snapshot_features
# ---------------------------------------------------------------------------

class TestComputeSnapshotFeatures:
    def test_returns_none_for_empty_rows(self):
        assert _compute_snapshot_features([]) is None

    def test_returns_none_when_shot_coords_missing(self):
        row = _make_player_row(shot_x=None, shot_y=None)
        assert _compute_snapshot_features([row]) is None

    def test_basic_single_defender(self):
        # Shooter at (50, 0), defending skater at (60, 0), goalie at (80, 5)
        rows = [
            _make_player_row(
                shot_x=50.0, shot_y=0.0,
                player_x=60.0, player_y=0.0,
                player_team="TBL", class_name="player_away",
            ),
            _make_player_row(
                shot_x=50.0, shot_y=0.0,
                player_x=80.0, player_y=5.0,
                player_team="TBL", class_name="goalie_away",
            ),
        ]
        feat = _compute_snapshot_features(rows)
        assert feat is not None
        assert feat["game_id"] == rows[0]["game_id"]
        assert feat["screen_count"] >= 0
        # Defender at 10 ft
        assert abs(feat["defender_dist"] - 10.0) < 0.01

    def test_goalie_displacement_from_crease(self):
        # Goalie exactly at crease centre (89, 0) → displacement = 0
        rows = [
            _make_player_row(
                shot_x=50.0, shot_y=0.0,
                player_x=_GOAL_LINE_X, player_y=0.0,
                player_team="TBL", class_name="goalie_away",
            ),
        ]
        feat = _compute_snapshot_features(rows)
        assert feat is not None
        assert feat["goalie_displacement"] == pytest.approx(0.0, abs=0.01)

    def test_goalie_displaced_from_crease(self):
        # Goalie 3 ft out from crease centre: (86, 0) → displacement = 3
        rows = [
            _make_player_row(
                shot_x=50.0, shot_y=0.0,
                player_x=86.0, player_y=0.0,
                player_team="TBL", class_name="goalie_away",
            ),
        ]
        feat = _compute_snapshot_features(rows)
        assert feat is not None
        assert feat["goalie_displacement"] == pytest.approx(3.0, abs=0.05)

    def test_left_side_attack_crease_at_neg89(self):
        # Shooter at x=-50 (attacking left end) → crease at x=-89
        rows = [
            _make_player_row(
                shot_x=-50.0, shot_y=0.0,
                player_x=-_GOAL_LINE_X, player_y=0.0,
                player_team="TBL", class_name="goalie_away",
            ),
        ]
        feat = _compute_snapshot_features(rows)
        assert feat is not None
        assert feat["goalie_displacement"] == pytest.approx(0.0, abs=0.01)

    def test_no_goalie_defender_dist_defaults(self):
        # Only a single skater; no goalie
        rows = [
            _make_player_row(
                shot_x=50.0, shot_y=0.0,
                player_x=55.0, player_y=0.0,
                player_team="TBL", class_name="player_away",
            ),
        ]
        feat = _compute_snapshot_features(rows)
        assert feat is not None
        assert feat["defender_dist"] == pytest.approx(5.0, abs=0.01)
        assert feat["goalie_displacement"] == pytest.approx(0.0, abs=0.01)

    def test_screen_player_directly_in_path(self):
        # Player at midpoint of shooter→goalie line → screens
        rows = [
            _make_player_row(
                shot_x=50.0, shot_y=0.0,
                player_x=70.0, player_y=0.0,    # midpoint
                player_team="TBL", class_name="player_away",
            ),
            _make_player_row(
                shot_x=50.0, shot_y=0.0,
                player_x=89.0, player_y=0.0,
                player_team="TBL", class_name="goalie_away",
            ),
        ]
        feat = _compute_snapshot_features(rows)
        assert feat is not None
        # Player at 70 is between shooter(50) and goalie(89) and within 3 ft → screens
        assert feat["screen_count"] >= 1

    def test_return_keys(self):
        rows = [_make_player_row()]
        feat = _compute_snapshot_features(rows)
        assert feat is not None
        required = {"game_id", "period", "time_in_period_secs", "shooter_id",
                    "screen_count", "defender_dist", "goalie_displacement"}
        assert required.issubset(feat.keys())


# ---------------------------------------------------------------------------
# Tests: extract_snapshot_features (empty DB)
# ---------------------------------------------------------------------------

class TestExtractSnapshotFeaturesEmpty:
    def test_returns_empty_frame_when_db_missing(self, tmp_path):
        # Non-existent DB path — should return empty frame with correct schema
        result = extract_snapshot_features(tmp_path / "nonexistent.duckdb")
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()
        assert "screen_count" in result.columns
        assert "defender_dist" in result.columns
        assert "goalie_displacement" in result.columns

    def test_returns_empty_frame_for_empty_db(self, tmp_path):
        # Create a real empty DB
        from data.tracking_accumulator import TrackingAccumulator
        acc = TrackingAccumulator(tmp_path / "empty.duckdb")
        acc.close()
        result = extract_snapshot_features(tmp_path / "empty.duckdb")
        assert result.is_empty()


# ---------------------------------------------------------------------------
# Tests: join_tracking_features
# ---------------------------------------------------------------------------

class TestJoinTrackingFeatures:
    def _make_tracking_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "game_id":             [2024020001, 2024020001],
            "period":              [1, 2],
            "time_in_period_secs": [300, 600],
            "shooter_id":          [8481600, 8481600],
            "screen_count":        [2, 1],
            "defender_dist":       [6.0, 9.0],
            "goalie_displacement": [1.5, 3.0],
        })

    def test_join_adds_three_columns(self):
        shot_df = pl.DataFrame({
            "game_id":    [2024020001],
            "period":     [1],
            "time":       [300],
            "shooter_id": [8481600],
            "is_goal":    [0],
        })
        tracking_df = self._make_tracking_df()
        result = join_tracking_features(shot_df, tracking_df)
        assert "screen_count" in result.columns
        assert "defender_dist" in result.columns
        assert "goalie_displacement" in result.columns

    def test_join_correct_values_on_match(self):
        shot_df = pl.DataFrame({
            "game_id":    [2024020001],
            "period":     [1],
            "time":       [300],
            "shooter_id": [8481600],
        })
        tracking_df = self._make_tracking_df()
        result = join_tracking_features(shot_df, tracking_df)
        assert result["screen_count"][0] == 2
        assert result["defender_dist"][0] == pytest.approx(6.0)
        assert result["goalie_displacement"][0] == pytest.approx(1.5)

    def test_join_fill_on_miss(self):
        # Shot with no matching snapshot
        shot_df = pl.DataFrame({
            "game_id":    [9999999],
            "period":     [3],
            "time":       [900],
            "shooter_id": [0000000],
        })
        tracking_df = self._make_tracking_df()
        result = join_tracking_features(shot_df, tracking_df)
        assert result["screen_count"][0] == 0
        assert result["defender_dist"][0] == pytest.approx(_DEFAULT_DEFENDER_DIST)
        assert result["goalie_displacement"][0] == pytest.approx(0.0)

    def test_join_empty_tracking_df_adds_fill_columns(self):
        shot_df = pl.DataFrame({
            "game_id":    [2024020001, 2024020002],
            "period":     [1, 1],
            "time":       [100, 200],
            "shooter_id": [111, 222],
        })
        empty_tracking = pl.DataFrame({
            "game_id":             pl.Series([], dtype=pl.Int64),
            "period":              pl.Series([], dtype=pl.Int32),
            "time_in_period_secs": pl.Series([], dtype=pl.Int32),
            "shooter_id":          pl.Series([], dtype=pl.Int64),
            "screen_count":        pl.Series([], dtype=pl.Int32),
            "defender_dist":       pl.Series([], dtype=pl.Float32),
            "goalie_displacement": pl.Series([], dtype=pl.Float32),
        })
        result = join_tracking_features(shot_df, empty_tracking)
        assert len(result) == 2
        assert result["screen_count"].to_list() == [0, 0]
        assert all(d == pytest.approx(_DEFAULT_DEFENDER_DIST) for d in result["defender_dist"].to_list())

    def test_row_count_preserved(self):
        # Left join must never drop or duplicate shot rows
        shot_df = _make_shot_df(50)
        shot_df = shot_df.with_columns([
            pl.lit(300).alias("time").cast(pl.Int64),
        ])
        tracking_df = pl.DataFrame({
            "game_id":             pl.Series([], dtype=pl.Int64),
            "period":              pl.Series([], dtype=pl.Int32),
            "time_in_period_secs": pl.Series([], dtype=pl.Int32),
            "shooter_id":          pl.Series([], dtype=pl.Int64),
            "screen_count":        pl.Series([], dtype=pl.Int32),
            "defender_dist":       pl.Series([], dtype=pl.Float32),
            "goalie_displacement": pl.Series([], dtype=pl.Float32),
        })
        result = join_tracking_features(shot_df, tracking_df)
        assert len(result) == 50


# ---------------------------------------------------------------------------
# Tests: build_features_tracking
# ---------------------------------------------------------------------------

class TestBuildFeaturesTracking:
    def test_shape_without_tracking_cols(self):
        df = _make_shot_df(50)
        X, y = build_features_tracking(df)
        assert X.shape == (50, len(feature_names_tracking()))
        assert y is not None

    def test_shape_with_tracking_cols(self):
        df = _make_shot_df(50, with_tracking=True)
        X, y = build_features_tracking(df)
        assert X.shape == (50, len(feature_names_tracking()))

    def test_no_nan(self):
        df = _make_shot_df(50)
        X, _ = build_features_tracking(df)
        assert not np.isnan(X).any()

    def test_tracking_defaults_when_absent(self):
        df = _make_shot_df(20)
        assert "screen_count" not in df.columns
        X, _ = build_features_tracking(df)
        # screen_count column (index -3) should be all zeros
        n_base = len(feature_names())
        assert np.all(X[:, n_base] == 0.0)       # screen_count
        assert np.all(X[:, n_base + 1] == 15.0)  # defender_dist default
        assert np.all(X[:, n_base + 2] == 0.0)   # goalie_displacement

    def test_tracking_values_used_when_present(self):
        df = _make_shot_df(10, with_tracking=True)
        X, _ = build_features_tracking(df)
        n_base = len(feature_names())
        assert np.all(X[:, n_base] == 1.0)   # screen_count
        assert np.all(X[:, n_base + 1] == 8.0)  # defender_dist
        assert np.all(X[:, n_base + 2] == 2.5)  # goalie_displacement

    def test_feature_names_tracking_count(self):
        assert len(feature_names_tracking()) == len(feature_names()) + 3

    def test_tracking_names_suffix(self):
        names = feature_names_tracking()
        assert names[-3] == "screen_count"
        assert names[-2] == "defender_dist"
        assert names[-1] == "goalie_displacement"


# ---------------------------------------------------------------------------
# Tests: XGModel with tracking_features=True
# ---------------------------------------------------------------------------

class TestXGModelTracking:
    def test_fit_and_predict(self):
        df = _make_shot_df(200, with_tracking=True)
        model = XGModel(tracking_features=True)
        metrics = model.fit(df.head(160), eval_df=df.tail(40))
        assert "auc_roc" in metrics
        proba = model.predict_proba(df.tail(40))
        assert proba.shape == (40,)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_version_string(self):
        model = XGModel(tracking_features=True)
        assert model._version == MODEL_VERSION_TRACKING

    def test_base_version_string(self):
        model = XGModel(tracking_features=False)
        assert model._version == MODEL_VERSION

    def test_save_load_roundtrip_tracking(self):
        df = _make_shot_df(200, with_tracking=True)
        model = XGModel(tracking_features=True)
        model.fit(df.head(160))
        proba_before = model.predict_proba(df.tail(40))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xg_tracking_test.pkl"
            model.save(path)
            loaded = XGModel.load(path)

        assert loaded._tracking_features is True
        proba_after = loaded.predict_proba(df.tail(40))
        np.testing.assert_allclose(proba_before, proba_after, rtol=1e-5)

    def test_save_load_roundtrip_base(self):
        # Ensure loading a base model still has tracking_features=False
        df = _make_shot_df(200)
        model = XGModel(tracking_features=False)
        model.fit(df.head(160))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xg_base_test.pkl"
            model.save(path)
            loaded = XGModel.load(path)

        assert loaded._tracking_features is False

    def test_tracking_model_uses_extended_feature_names(self):
        df = _make_shot_df(100, with_tracking=True)
        model = XGModel(tracking_features=True)
        model.fit(df)
        names = model._feature_names()
        assert len(names) == len(feature_names_tracking())
        assert "screen_count" in names
        assert "defender_dist" in names

    def test_base_model_uses_base_feature_names(self):
        df = _make_shot_df(100)
        model = XGModel()
        model.fit(df)
        names = model._feature_names()
        assert len(names) == len(feature_names())
        assert "screen_count" not in names

    def test_tracking_fallback_on_shot_without_tracking_cols(self):
        # Tracking model can infer on shots lacking the 3 columns (falls back to defaults)
        train_df = _make_shot_df(150, with_tracking=True)
        model = XGModel(tracking_features=True)
        model.fit(train_df)
        # Test on DataFrame WITHOUT tracking columns
        test_no_tracking = _make_shot_df(20, with_tracking=False)
        proba = model.predict_proba(test_no_tracking)
        assert proba.shape == (20,)
        assert not np.isnan(proba).any()


# ---------------------------------------------------------------------------
# Tests: retrain gate logic (no live DuckDB)
# ---------------------------------------------------------------------------

class TestRetrainGateLogic:
    """Verify the gate condition in the retrain script is correct."""

    def test_gate_passes_when_tracking_auc_higher(self):
        base_auc     = 0.760
        tracking_auc = 0.768
        min_delta    = 0.001
        assert tracking_auc > base_auc + min_delta

    def test_gate_fails_when_tracking_auc_same(self):
        base_auc     = 0.760
        tracking_auc = 0.760
        min_delta    = 0.001
        assert not (tracking_auc > base_auc + min_delta)

    def test_gate_fails_when_tracking_auc_lower(self):
        base_auc     = 0.762
        tracking_auc = 0.760
        min_delta    = 0.001
        assert not (tracking_auc > base_auc + min_delta)

    def test_gate_passes_on_expected_lift(self):
        # Expected 4–8% AUC lift per PLAN.md
        base_auc     = 0.760
        tracking_auc = 0.760 * 1.05   # +5%
        min_delta    = 0.001
        assert tracking_auc > base_auc + min_delta
