"""Tests for Feature 2.14 — RegimeChangeDetector."""

from __future__ import annotations

import pytest
import polars as pl

from models.regime_change_detector import (
    RegimeChangeDetector,
    PlayerRegimeState,
    REGIME_SIGMA_THRESHOLD,
    REGIME_CONSECUTIVE_TRIGGER,
    REGIME_PRIOR_SIGMA_MULT,
    REGIME_OBS_SIGMA_DIVISOR,
    REGIME_CLEAR_CONSECUTIVE,
    write_regime_alerts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feed_games(
    detector: RegimeChangeDetector,
    player_id: int,
    player_name: str,
    z_scores: list[float],
    posterior_sigma: float = 0.5,
    season: int = 2025,
) -> list:
    """Feed a sequence of z-score games to the detector. Returns alerts list."""
    posterior_mean = 3.0
    alerts = []
    for i, z in enumerate(z_scores):
        observation = posterior_mean + z * posterior_sigma
        alert = detector.update(
            player_id=player_id,
            player_name=player_name,
            observation=observation,
            posterior_mean=posterior_mean,
            posterior_sigma=posterior_sigma,
            game_number=i + 1,
            season=season,
        )
        alerts.append(alert)
    return alerts


# ---------------------------------------------------------------------------
# Basic update — no regime
# ---------------------------------------------------------------------------

class TestUpdateNoRegime:
    def test_no_alert_below_threshold(self):
        det = RegimeChangeDetector()
        # Only 9 consecutive above-threshold games — not enough to trigger
        z_scores = [2.0] * 9
        alerts = _feed_games(det, 1, "Player A", z_scores)
        assert all(a is None for a in alerts)

    def test_no_alert_mixed(self):
        det = RegimeChangeDetector()
        # Alternating above/below resets consecutive counters
        z_scores = [2.0, -2.0] * 15
        alerts = _feed_games(det, 1, "Player A", z_scores)
        assert all(a is None for a in alerts)

    def test_no_alert_neutral(self):
        det = RegimeChangeDetector()
        z_scores = [0.5] * 20
        alerts = _feed_games(det, 1, "Player A", z_scores)
        assert all(a is None for a in alerts)


# ---------------------------------------------------------------------------
# up_shift trigger
# ---------------------------------------------------------------------------

class TestUpShiftTrigger:
    def test_up_shift_at_exactly_10(self):
        det = RegimeChangeDetector()
        z_scores = [2.0] * 10
        alerts = _feed_games(det, 1, "Player A", z_scores)
        # Only last alert should fire
        assert alerts[-1] is not None
        assert alerts[-1]["regime_state"] == "up_shift"
        assert all(a is None for a in alerts[:-1])

    def test_up_shift_alert_fields(self):
        det = RegimeChangeDetector()
        z_scores = [2.0] * 10
        alerts = _feed_games(det, 1, "Matthews", z_scores, season=2025)
        alert = alerts[-1]
        assert alert["player_id"] == 1
        assert alert["player_name"] == "Matthews"
        assert alert["season"] == 2025
        assert alert["triggered_at_game"] == 10
        assert alert["consecutive_games"] == 10
        assert alert["sigma_multiplier"] == REGIME_PRIOR_SIGMA_MULT
        assert alert["obs_sigma_divisor"] == REGIME_OBS_SIGMA_DIVISOR
        assert "Statistical" in alert["reason"]

    def test_up_shift_no_further_alerts_when_active(self):
        det = RegimeChangeDetector()
        # Trigger regime then add more above-threshold games
        z_scores = [2.0] * 20
        alerts = _feed_games(det, 1, "Player A", z_scores)
        # Only one alert should fire (at game 10)
        non_none = [a for a in alerts if a is not None]
        assert len(non_none) == 1


# ---------------------------------------------------------------------------
# down_shift trigger
# ---------------------------------------------------------------------------

class TestDownShiftTrigger:
    def test_down_shift_at_exactly_10(self):
        det = RegimeChangeDetector()
        z_scores = [-2.0] * 10
        alerts = _feed_games(det, 1, "Player A", z_scores)
        assert alerts[-1] is not None
        assert alerts[-1]["regime_state"] == "down_shift"

    def test_down_shift_resets_on_neutral(self):
        det = RegimeChangeDetector()
        # 9 below, then 1 neutral, then 9 below — should not trigger
        z_scores = [-2.0] * 9 + [0.0] + [-2.0] * 9
        alerts = _feed_games(det, 1, "Player A", z_scores)
        assert all(a is None for a in alerts)


# ---------------------------------------------------------------------------
# Clearing regime
# ---------------------------------------------------------------------------

class TestClearRegime:
    def test_up_shift_cleared_by_neutral_games(self):
        det = RegimeChangeDetector()
        # Trigger up_shift
        _feed_games(det, 1, "Player A", [2.0] * 10)
        state = det.get_state(1)
        assert state.regime_state == "up_shift"

        # Feed REGIME_CLEAR_CONSECUTIVE neutral games
        for i in range(REGIME_CLEAR_CONSECUTIVE):
            det.update(1, "Player A", observation=3.0, posterior_mean=3.0,
                       posterior_sigma=0.5, game_number=11 + i)

        state = det.get_state(1)
        assert state.regime_state == "none"

    def test_down_shift_cleared_by_neutral_games(self):
        det = RegimeChangeDetector()
        _feed_games(det, 1, "Player A", [-2.0] * 10)
        for i in range(REGIME_CLEAR_CONSECUTIVE):
            det.update(1, "Player A", observation=3.0, posterior_mean=3.0,
                       posterior_sigma=0.5, game_number=11 + i)
        state = det.get_state(1)
        assert state.regime_state == "none"

    def test_partial_neutral_not_enough_to_clear(self):
        det = RegimeChangeDetector()
        _feed_games(det, 1, "Player A", [2.0] * 10)
        for i in range(REGIME_CLEAR_CONSECUTIVE - 1):
            det.update(1, "Player A", observation=3.0, posterior_mean=3.0,
                       posterior_sigma=0.5, game_number=11 + i)
        state = det.get_state(1)
        assert state.regime_state == "up_shift"


# ---------------------------------------------------------------------------
# Manual override
# ---------------------------------------------------------------------------

class TestManualOverride:
    def test_manual_up_override(self):
        det = RegimeChangeDetector()
        alert = det.manual_override(
            player_id=1, player_name="Matthews",
            direction="up", reason="New coach + contract year", season=2025
        )
        assert alert["regime_state"] == "manual_up"
        assert "[MANUAL]" in alert["reason"]
        state = det.get_state(1)
        assert state.regime_state == "manual_up"

    def test_manual_down_override(self):
        det = RegimeChangeDetector()
        alert = det.manual_override(1, "Matthews", "down", "Trade shock")
        assert alert["regime_state"] == "manual_down"

    def test_invalid_direction_raises(self):
        det = RegimeChangeDetector()
        with pytest.raises(ValueError):
            det.manual_override(1, "Matthews", "sideways", "test")

    def test_manual_override_prevents_data_driven(self):
        det = RegimeChangeDetector()
        det.manual_override(1, "Matthews", "up", "Manual reason")
        # Feed below-threshold games — should not change to down_shift
        z_scores = [-2.0] * 10
        alerts = _feed_games(det, 1, "Matthews", z_scores)
        assert all(a is None for a in alerts)
        state = det.get_state(1)
        assert state.regime_state == "manual_up"

    def test_clear_manual_override(self):
        det = RegimeChangeDetector()
        det.manual_override(1, "Matthews", "up", "Test")
        det.clear_manual_override(1)
        state = det.get_state(1)
        assert state.regime_state == "none"

    def test_clear_manual_override_nonexistent_player(self):
        det = RegimeChangeDetector()
        det.clear_manual_override(999)  # Should not raise


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

class TestQueries:
    def test_active_regimes_empty_initially(self):
        det = RegimeChangeDetector()
        assert det.active_regimes() == []

    def test_active_regimes_after_trigger(self):
        det = RegimeChangeDetector()
        _feed_games(det, 1, "Player A", [2.0] * 10)
        active = det.active_regimes()
        assert len(active) == 1
        assert active[0].player_id == 1

    def test_active_regimes_multiple_players(self):
        det = RegimeChangeDetector()
        _feed_games(det, 1, "Player A", [2.0] * 10)
        _feed_games(det, 2, "Player B", [-2.0] * 10)
        active = det.active_regimes()
        assert len(active) == 2

    def test_get_state_returns_none_for_unknown(self):
        det = RegimeChangeDetector()
        assert det.get_state(999) is None

    def test_recommended_sigma_no_regime(self):
        det = RegimeChangeDetector()
        mult, div = det.recommended_sigma_adjustments(999)
        assert mult == 1.0
        assert div == 1.0

    def test_recommended_sigma_active_regime(self):
        det = RegimeChangeDetector()
        _feed_games(det, 1, "Player A", [2.0] * 10)
        mult, div = det.recommended_sigma_adjustments(1)
        assert mult == REGIME_PRIOR_SIGMA_MULT
        assert div == REGIME_OBS_SIGMA_DIVISOR

    def test_recommended_sigma_no_regime_after_clear(self):
        det = RegimeChangeDetector()
        _feed_games(det, 1, "Player A", [2.0] * 10)
        for i in range(REGIME_CLEAR_CONSECUTIVE):
            det.update(1, "Player A", 3.0, 3.0, 0.5, 11 + i)
        mult, div = det.recommended_sigma_adjustments(1)
        assert mult == 1.0
        assert div == 1.0


# ---------------------------------------------------------------------------
# alerts_dataframe
# ---------------------------------------------------------------------------

class TestAlertsDataframe:
    def test_empty_df_when_no_alerts(self):
        det = RegimeChangeDetector()
        df = det.alerts_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_df_has_correct_schema(self):
        det = RegimeChangeDetector()
        df = det.alerts_dataframe()
        from models.regime_change_detector import REGIME_ALERT_SCHEMA
        for col in REGIME_ALERT_SCHEMA:
            assert col in df.columns

    def test_df_populated_after_trigger(self):
        det = RegimeChangeDetector()
        _feed_games(det, 1, "Player A", [2.0] * 10, season=2025)
        df = det.alerts_dataframe()
        assert len(df) == 1
        assert df["regime_state"][0] == "up_shift"

    def test_df_season_filter(self):
        det = RegimeChangeDetector()
        _feed_games(det, 1, "Player A", [2.0] * 10, season=2024)
        _feed_games(det, 2, "Player B", [2.0] * 10, season=2025)
        df_2025 = det.alerts_dataframe(season=2025)
        assert len(df_2025) == 1
        assert int(df_2025["season"][0]) == 2025


# ---------------------------------------------------------------------------
# Zero / edge sigma
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_posterior_sigma_skipped(self):
        det = RegimeChangeDetector()
        alert = det.update(
            player_id=1, player_name="P",
            observation=5.0, posterior_mean=3.0,
            posterior_sigma=0.0, game_number=1
        )
        assert alert is None

    def test_multiple_players_independent(self):
        det = RegimeChangeDetector()
        _feed_games(det, 1, "Player A", [2.0] * 10)
        # Player B has 9 games only — should not trigger
        _feed_games(det, 2, "Player B", [2.0] * 9)
        assert det.get_state(1).regime_state == "up_shift"
        assert det.get_state(2).regime_state == "none"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, tmp_path):
        det = RegimeChangeDetector()
        _feed_games(det, 1, "Player A", [2.0] * 10, season=2025)
        det.manual_override(2, "Player B", "down", "Test", season=2025)

        path = tmp_path / "regime.pkl"
        det.save(path)

        det2 = RegimeChangeDetector.load(path)
        assert det2.get_state(1).regime_state == "up_shift"
        assert det2.get_state(2).regime_state == "manual_down"
        assert len(det2.alerts_dataframe()) == 2

    def test_write_regime_alerts(self, tmp_path):
        det = RegimeChangeDetector()
        _feed_games(det, 1, "Player A", [2.0] * 10, season=2025)
        df = det.alerts_dataframe(season=2025)
        path = write_regime_alerts(df, tmp_path, 2025)
        assert path.exists()
        loaded = pl.read_parquet(path)
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# PlayerRegimeState
# ---------------------------------------------------------------------------

class TestPlayerRegimeState:
    def test_is_active_false_for_none_state(self):
        state = PlayerRegimeState(player_id=1, player_name="P")
        assert not state.is_active()

    def test_is_active_true_for_up_shift(self):
        state = PlayerRegimeState(player_id=1, player_name="P", regime_state="up_shift")
        assert state.is_active()

    def test_mean_residual_empty(self):
        state = PlayerRegimeState(player_id=1, player_name="P")
        assert state.mean_residual() == 0.0

    def test_mean_residual_computed(self):
        state = PlayerRegimeState(player_id=1, player_name="P")
        state.residuals.extend([1.0, 2.0, 3.0])
        assert abs(state.mean_residual() - 2.0) < 1e-9
