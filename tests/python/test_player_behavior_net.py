"""Tests for Feature 2.22 — PlayerBehaviorNet."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from models.player_behavior_net import (
    PlayerBehaviorNet,
    write_behavior_predictions,
    _apply_context_rules,
    _build_base_profile,
    _softmax,
    ACTION_LABELS,
    N_ACTIONS,
    CONTEXT_FEATURE_NAMES,
    N_CONTEXT,
    BEHAVIOR_PREDICTION_SCHEMA,
    MODEL_VERSION,
)
from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _player_stats(n: int = 10, seed: int = 0) -> pl.DataFrame:
    """Minimal player stats DataFrame."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "player_id":       list(range(1, n + 1)),
        "player_name":     [f"Player {i}" for i in range(1, n + 1)],
        "team":            rng.choice(["EDM", "TOR", "BOS"], n).tolist(),
        "carry_entry_pct": rng.uniform(0.3, 0.7, n).tolist(),
        "battle_score":    rng.normal(0, 1, n).tolist(),
    })


def _positional_df(n: int = 10) -> pl.DataFrame:
    """Minimal positional heatmap DataFrame."""
    rng = np.random.default_rng(11)
    raw = np.abs(rng.dirichlet(np.ones(6), n))
    return pl.DataFrame({
        "player_id":         list(range(1, n + 1)),
        "off_net_front_pct": raw[:, 0].tolist(),
        "off_slot_pct":      raw[:, 1].tolist(),
        "off_circle_pct":    raw[:, 2].tolist(),
        "off_half_wall_pct": raw[:, 3].tolist(),
        "off_point_pct":     raw[:, 4].tolist(),
        "off_corner_pct":    raw[:, 5].tolist(),
    })


def _train_net(n_players: int = 10) -> PlayerBehaviorNet:
    """Fit a PlayerBehaviorNet on synthetic data."""
    net = PlayerBehaviorNet(hidden_layer_sizes=(8, 4), max_iter=50)
    net.fit(_player_stats(n_players), _positional_df(n_players))
    return net


# ---------------------------------------------------------------------------
# _softmax
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_sums_to_one(self):
        logits = np.array([1.0, 2.0, 3.0, 0.5, -1.0, 0.0, 0.2])
        probs = _softmax(logits)
        assert probs.sum() == pytest.approx(1.0)

    def test_all_non_negative(self):
        logits = np.array([1.0, -5.0, 0.0, 100.0, -100.0, 0.1, 0.0])
        assert np.all(_softmax(logits) >= 0)

    def test_uniform_logits_give_uniform_probs(self):
        logits = np.zeros(7)
        probs = _softmax(logits)
        assert np.allclose(probs, 1.0 / 7)


# ---------------------------------------------------------------------------
# _apply_context_rules
# ---------------------------------------------------------------------------

class TestApplyContextRules:
    _base = np.ones(7) / 7.0

    def test_returns_probs_sum_one(self):
        p = _apply_context_rules(self._base.copy(), 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0)
        assert p.sum() == pytest.approx(1.0)

    def test_all_non_negative(self):
        p = _apply_context_rules(self._base.copy(), 1.0, 1.0, -1.0, -1.0, 2.0, 0.5, 0.5)
        assert np.all(p >= 0)

    def test_high_fi_increases_dump(self):
        p_low  = _apply_context_rules(self._base.copy(), fi_score=0.0,  period_norm=0.5,
                                       score_diff_norm=0.0, manpower_norm=0.0,
                                       matchup_quality=0.0, speed_ratio=1.0, carry_ratio=1.0)
        p_high = _apply_context_rules(self._base.copy(), fi_score=1.0,  period_norm=0.5,
                                       score_diff_norm=0.0, manpower_norm=0.0,
                                       matchup_quality=0.0, speed_ratio=1.0, carry_ratio=1.0)
        # dump (index 1) should be higher at high FI
        assert p_high[1] > p_low[1]

    def test_high_fi_decreases_carry_in(self):
        p_low  = _apply_context_rules(self._base.copy(), 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0)
        p_high = _apply_context_rules(self._base.copy(), 1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0)
        assert p_high[0] < p_low[0]

    def test_losing_increases_perimeter(self):
        p_win  = _apply_context_rules(self._base.copy(), 0.0, 0.5,  1.0, 0.0, 0.0, 1.0, 1.0)
        p_lose = _apply_context_rules(self._base.copy(), 0.0, 0.5, -1.0, 0.0, 0.0, 1.0, 1.0)
        # shoot_perimeter (index 3) should be higher when losing
        assert p_lose[3] > p_win[3]


# ---------------------------------------------------------------------------
# _build_base_profile
# ---------------------------------------------------------------------------

class TestBuildBaseProfile:
    def test_returns_probability_vector(self):
        row = {"carry_entry_pct": 0.55, "battle_score": 0.2}
        probs = _build_base_profile(row, None)
        assert len(probs) == N_ACTIONS
        assert probs.sum() == pytest.approx(1.0)
        assert np.all(probs >= 0)

    def test_higher_carry_pct_shifts_carry_in_up(self):
        row_low  = {"carry_entry_pct": 0.20, "battle_score": 0.0}
        row_high = {"carry_entry_pct": 0.80, "battle_score": 0.0}
        p_low  = _build_base_profile(row_low, None)
        p_high = _build_base_profile(row_high, None)
        assert p_high[0] > p_low[0]  # carry_in (index 0) higher

    def test_positional_row_shifts_zones(self):
        row = {"carry_entry_pct": 0.50, "battle_score": 0.0}
        pos = {"off_net_front_pct": 0.40, "off_slot_pct": 0.30,
               "off_circle_pct": 0.10, "off_half_wall_pct": 0.10,
               "off_point_pct": 0.05, "off_corner_pct": 0.05}
        p = _build_base_profile(row, pos)
        assert len(p) == N_ACTIONS
        assert p.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# PlayerBehaviorNet.encode_context
# ---------------------------------------------------------------------------

class TestEncodeContext:
    def test_returns_correct_length(self):
        ctx = PlayerBehaviorNet.encode_context()
        assert len(ctx) == N_CONTEXT

    def test_default_context_is_neutral(self):
        ctx = PlayerBehaviorNet.encode_context()
        # fi_score=0, period=2 → period_norm=1/3, score_diff=0, home=1, manpower=0
        assert ctx[0] == pytest.approx(0.0)  # fi_score
        assert ctx[2] == pytest.approx(0.0)  # score_diff_norm
        assert ctx[3] == pytest.approx(1.0)  # home

    def test_fi_clamped(self):
        ctx_lo = PlayerBehaviorNet.encode_context(fi_score=-1.0)
        ctx_hi = PlayerBehaviorNet.encode_context(fi_score=5.0)
        assert ctx_lo[0] == pytest.approx(0.0)
        assert ctx_hi[0] == pytest.approx(1.0)

    def test_score_diff_clamped(self):
        ctx = PlayerBehaviorNet.encode_context(score_diff=-10)
        assert ctx[2] == pytest.approx(-1.0)

    def test_away_flag(self):
        ctx = PlayerBehaviorNet.encode_context(home=False)
        assert ctx[3] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# PlayerBehaviorNet.fit
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_returns_self(self):
        net = PlayerBehaviorNet(hidden_layer_sizes=(4,), max_iter=10)
        result = net.fit(_player_stats(5))
        assert result is net

    def test_fitted_flag(self):
        net = PlayerBehaviorNet(hidden_layer_sizes=(4,), max_iter=10)
        net.fit(_player_stats(5))
        assert net._fitted

    def test_base_profiles_populated(self):
        net = _train_net(10)
        assert len(net._base_profiles) == 10

    def test_player_bias_populated(self):
        net = _train_net(10)
        assert len(net._player_bias) == 10

    def test_missing_player_id_raises(self):
        net = PlayerBehaviorNet()
        with pytest.raises(ValueError, match="player_id"):
            net.fit(pl.DataFrame({"carry_entry_pct": [0.5]}))

    def test_empty_df_warns(self):
        net = PlayerBehaviorNet()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.fit(pl.DataFrame(schema={"player_id": pl.Int64,
                                          "carry_entry_pct": pl.Float64}))
        assert net._fitted
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_fit_with_positional_df(self):
        net = PlayerBehaviorNet(hidden_layer_sizes=(4,), max_iter=10)
        net.fit(_player_stats(5), _positional_df(5))
        assert len(net._base_profiles) == 5


# ---------------------------------------------------------------------------
# PlayerBehaviorNet.predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_returns_all_action_labels(self):
        net = _train_net(5)
        probs = net.predict(1)
        assert set(probs.keys()) == set(ACTION_LABELS)

    def test_probs_sum_to_one(self):
        net = _train_net(5)
        probs = net.predict(1)
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)

    def test_all_probs_non_negative(self):
        net = _train_net(5)
        probs = net.predict(1)
        assert all(v >= 0 for v in probs.values())

    def test_unknown_player_returns_default(self):
        net = _train_net(5)
        probs = net.predict(99999)
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(v >= 0 for v in probs.values())

    def test_high_fi_shifts_toward_dump(self):
        net = _train_net(10)
        p_rested  = net.predict(1, fi_score=0.0)
        p_fatigued = net.predict(1, fi_score=0.9)
        assert p_fatigued["dump"] > p_rested["dump"]

    def test_high_fi_reduces_carry_in(self):
        net = _train_net(10)
        p_rested  = net.predict(1, fi_score=0.0)
        p_fatigued = net.predict(1, fi_score=0.9)
        assert p_fatigued["carry_in"] < p_rested["carry_in"]

    def test_power_play_shifts_toward_slot(self):
        net = _train_net(10)
        p_ev = net.predict(1, manpower=0)
        p_pp = net.predict(1, manpower=1)
        assert p_pp["shoot_slot"] > p_ev["shoot_slot"]

    def test_penalty_kill_shifts_toward_dump(self):
        net = _train_net(10)
        p_ev = net.predict(1, manpower=0)
        p_pk = net.predict(1, manpower=-1)
        assert p_pk["dump"] > p_ev["dump"]

    def test_all_players_different_profiles(self):
        net = _train_net(5)
        p1 = net.predict(1)
        p2 = net.predict(2)
        # Players have different base profiles → probabilities differ
        assert p1 != p2

    def test_neutral_context_rested(self):
        net = _train_net(5)
        p = net.predict(1, fi_score=0.0, period=2, score_diff=0,
                        home=True, manpower=0, matchup_quality=0.0,
                        speed_ratio=1.0, carry_ratio=1.0)
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# predict_batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_returns_dataframe(self):
        net = _train_net(5)
        pids = [1, 2, 3]
        ctxs = np.stack([PlayerBehaviorNet.encode_context() for _ in pids])
        result = net.predict_batch(pids, ctxs)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    def test_batch_columns(self):
        net = _train_net(5)
        pids = [1, 2]
        ctxs = np.stack([PlayerBehaviorNet.encode_context() for _ in pids])
        df = net.predict_batch(pids, ctxs)
        for act in ACTION_LABELS:
            assert act in df.columns

    def test_batch_probs_sum_to_one(self):
        net = _train_net(5)
        pids = [1, 2, 3]
        ctxs = np.stack([PlayerBehaviorNet.encode_context() for _ in pids])
        df = net.predict_batch(pids, ctxs)
        for row in df.to_dicts():
            total = sum(row[a] for a in ACTION_LABELS)
            assert total == pytest.approx(1.0, abs=1e-6)

    def test_empty_batch(self):
        net = _train_net(5)
        df = net.predict_batch([], np.empty((0, N_CONTEXT)))
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# predict_season
# ---------------------------------------------------------------------------

class TestPredictSeason:
    def test_returns_dataframe(self):
        net = _train_net(5)
        df = net.predict_season(season=2025)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5

    def test_schema_columns_present(self):
        net = _train_net(5)
        df = net.predict_season(season=2025)
        for col in BEHAVIOR_PREDICTION_SCHEMA:
            assert col in df.columns

    def test_season_attached(self):
        net = _train_net(5)
        df = net.predict_season(season=2024)
        assert all(df["season"].to_list()[i] == 2024 for i in range(len(df)))

    def test_fi_scores_applied(self):
        net = _train_net(5)
        fi_map = {1: 0.8, 2: 0.0}
        df = net.predict_season(season=2025, fi_scores=fi_map)
        row1 = df.filter(pl.col("player_id") == 1).to_dicts()[0]
        row2 = df.filter(pl.col("player_id") == 2).to_dicts()[0]
        # fatigued player (1) should have more dump than rested (2)
        assert row1["dump"] > row2["dump"]

    def test_empty_net_returns_empty_df(self):
        net = PlayerBehaviorNet()
        net._fitted = True
        df = net.predict_season(season=2025)
        assert len(df) == 0
        for col in BEHAVIOR_PREDICTION_SCHEMA:
            assert col in df.columns

    def test_model_version_present(self):
        net = _train_net(3)
        df = net.predict_season(season=2025)
        assert df["model_version"][0] == MODEL_VERSION


# ---------------------------------------------------------------------------
# action_labels / player_ids
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_action_labels_count(self):
        net = PlayerBehaviorNet()
        assert len(net.action_labels()) == 7

    def test_action_labels_correct(self):
        net = PlayerBehaviorNet()
        assert net.action_labels() == ACTION_LABELS

    def test_player_ids_after_fit(self):
        net = _train_net(8)
        assert len(net.player_ids()) == 8
        assert set(net.player_ids()) == set(range(1, 9))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        net = PlayerBehaviorNet(hidden_layer_sizes=(8, 4), max_iter=30)
        net.fit(_player_stats(5), _positional_df(5))
        path = tmp_path / "behavior.pkl"
        net.save(path)
        net2 = PlayerBehaviorNet.load(path)
        assert net2._hidden == (8, 4)
        assert net2._fitted
        assert len(net2._base_profiles) == 5
        assert len(net2._player_bias) == 5

    def test_loaded_predict_consistent(self, tmp_path):
        net = PlayerBehaviorNet(hidden_layer_sizes=(8,), max_iter=30)
        net.fit(_player_stats(5))
        p_before = net.predict(1, fi_score=0.5)

        path = tmp_path / "beh2.pkl"
        net.save(path)
        net2 = PlayerBehaviorNet.load(path)
        p_after = net2.predict(1, fi_score=0.5)

        for action in ACTION_LABELS:
            assert p_after[action] == pytest.approx(p_before[action], abs=1e-9)

    def test_write_behavior_predictions(self, tmp_path):
        net = _train_net(5)
        df = net.predict_season(season=2025)
        path = write_behavior_predictions(df, tmp_path, 2025)
        assert path.exists()
        loaded = pl.read_parquet(path)
        assert "player_id" in loaded.columns
        assert "carry_in" in loaded.columns

    def test_write_filename_contains_season(self, tmp_path):
        net = _train_net(5)
        df = net.predict_season(season=2023)
        path = write_behavior_predictions(df, tmp_path, 2023)
        assert "2023" in path.name
