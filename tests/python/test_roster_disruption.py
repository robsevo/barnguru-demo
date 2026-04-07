"""Tests for Feature 2.19 — RosterDisruptionModel."""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from models.roster_disruption import (
    RosterDisruptionModel,
    decay_factor,
    event_weight,
    write_roster_disruption,
    DECAY_HALF_LIFE_DAYS,
    CHAOS_MULTIPLIER,
    CHAOS_THRESHOLD_72H,
    MAX_EFFICIENCY_EFFECT,
    MAX_RAW,
    EVENT_TYPE_WEIGHTS,
    ROLE_MULTIPLIERS,
    ROSTER_DISRUPTION_SCHEMA,
)
from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AS_OF = date(2025, 3, 15)


_TX_SCHEMA = {
    "date": pl.Utf8,
    "event_type": pl.Utf8,
    "team": pl.Utf8,
    "secondary_team": pl.Utf8,
    "player_or_executive": pl.Utf8,
    "description": pl.Utf8,
}


def _tx_df(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal transactions DataFrame."""
    defaults = {
        "date": AS_OF.isoformat(),
        "event_type": "trade",
        "team": "EDM",
        "secondary_team": None,
        "player_or_executive": "Player A",
        "description": "",
    }
    filled = [{**defaults, **r} for r in rows]
    if not filled:
        return pl.DataFrame(schema=_TX_SCHEMA)
    return pl.DataFrame(filled)


def _single_trade(days_ago: int = 0, team: str = "EDM") -> pl.DataFrame:
    """One trade for team EDM, N days ago."""
    d = (AS_OF - timedelta(days=days_ago)).isoformat()
    return _tx_df([{"date": d, "event_type": "trade", "team": team}])


# ---------------------------------------------------------------------------
# decay_factor
# ---------------------------------------------------------------------------

class TestDecayFactor:
    def test_day_zero_is_one(self):
        assert decay_factor(0) == pytest.approx(1.0)

    def test_half_life_is_half(self):
        assert decay_factor(DECAY_HALF_LIFE_DAYS) == pytest.approx(0.5)

    def test_double_half_life_is_quarter(self):
        assert decay_factor(DECAY_HALF_LIFE_DAYS * 2) == pytest.approx(0.25)

    def test_negative_days_clamped_to_zero(self):
        # Negative days_elapsed (future event) treated as 0
        assert decay_factor(-5) == pytest.approx(1.0)

    def test_custom_half_life(self):
        assert decay_factor(7, half_life=7) == pytest.approx(0.5)

    def test_very_large_days_near_zero(self):
        assert decay_factor(200) < 0.01

    def test_zero_half_life_does_not_crash(self):
        # half_life clamped to 1e-9 — should not divide by zero
        val = decay_factor(1, half_life=0.0)
        assert np.isfinite(val)


# ---------------------------------------------------------------------------
# event_weight
# ---------------------------------------------------------------------------

class TestEventWeight:
    def test_trade_base_weight(self):
        assert event_weight("trade") == pytest.approx(EVENT_TYPE_WEIGHTS["trade"])

    def test_unknown_type_returns_other(self):
        assert event_weight("mystery_event") == pytest.approx(EVENT_TYPE_WEIGHTS["other"])

    def test_role_multiplier_applied(self):
        base = EVENT_TYPE_WEIGHTS["trade"]
        role_mult = ROLE_MULTIPLIERS["starter_g"]
        expected = min(base * role_mult, 1.0)
        assert event_weight("trade", player_role="starter_g") == pytest.approx(expected)

    def test_role_multiplier_clips_at_one(self):
        # Even top role × high event weight shouldn't exceed 1.0
        w = event_weight("trade", player_role="starter_g")
        assert 0.0 <= w <= 1.0

    def test_no_role_no_penalty_or_bonus(self):
        # Missing role → multiplier = 1.0
        assert event_weight("signing") == pytest.approx(EVENT_TYPE_WEIGHTS["signing"])

    def test_custom_weights_override(self):
        custom = {"trade": 0.99, "other": 0.01}
        assert event_weight("trade", custom_weights=custom) == pytest.approx(0.99)

    def test_bottom_role_reduces_weight(self):
        base = event_weight("trade")
        bottom = event_weight("trade", player_role="backup_g")
        assert bottom < base


# ---------------------------------------------------------------------------
# RosterDisruptionModel.compute — basic
# ---------------------------------------------------------------------------

class TestComputeBasic:
    def test_zero_transactions_zero_disruption(self):
        model = RosterDisruptionModel()
        df = _tx_df([])
        result = model.compute(df, team="EDM", as_of_date=AS_OF)
        assert result["disruption_index"] == pytest.approx(0.0)
        assert result["n_moves_30d"] == 0

    def test_single_trade_today_nonzero(self):
        model = RosterDisruptionModel()
        df = _single_trade(days_ago=0)
        result = model.compute(df, team="EDM", as_of_date=AS_OF)
        assert result["disruption_index"] > 0.0

    def test_disruption_index_in_zero_one(self):
        model = RosterDisruptionModel()
        rows = [
            {"date": AS_OF.isoformat(), "event_type": "trade", "team": "EDM"},
            {"date": AS_OF.isoformat(), "event_type": "ltir", "team": "EDM"},
            {"date": AS_OF.isoformat(), "event_type": "waiver_claim", "team": "EDM"},
        ]
        result = model.compute(_tx_df(rows), team="EDM", as_of_date=AS_OF)
        assert 0.0 <= result["disruption_index"] <= 1.0

    def test_returns_expected_keys(self):
        model = RosterDisruptionModel()
        result = model.compute(_tx_df([]), team="EDM", as_of_date=AS_OF)
        for key in ROSTER_DISRUPTION_SCHEMA:
            assert key in result

    def test_efficiency_modifier_near_one_for_zero_disruption(self):
        model = RosterDisruptionModel()
        result = model.compute(_tx_df([]), team="EDM", as_of_date=AS_OF)
        assert result["efficiency_modifier"] == pytest.approx(1.0)

    def test_efficiency_modifier_min_at_max_disruption(self):
        # Artificially set disruption_index=1.0
        model = RosterDisruptionModel()
        em = model.efficiency_modifier(1.0)
        assert em == pytest.approx(1.0 - MAX_EFFICIENCY_EFFECT)

    def test_as_of_date_string_accepted(self):
        model = RosterDisruptionModel()
        df = _single_trade(days_ago=0)
        result = model.compute(df, team="EDM", as_of_date=AS_OF.isoformat())
        assert "disruption_index" in result

    def test_as_of_date_none_uses_today(self):
        model = RosterDisruptionModel()
        df = _tx_df([])
        result = model.compute(df, team="EDM", as_of_date=None)
        assert "as_of_date" in result

    def test_model_version_present(self):
        model = RosterDisruptionModel()
        result = model.compute(_tx_df([]), team="EDM", as_of_date=AS_OF)
        assert result["model_version"].startswith("rdi")


# ---------------------------------------------------------------------------
# Time decay
# ---------------------------------------------------------------------------

class TestTimeDecay:
    def test_recent_trade_higher_than_old_trade(self):
        model = RosterDisruptionModel()
        recent = model.compute(_single_trade(days_ago=1), team="EDM", as_of_date=AS_OF)
        old = model.compute(_single_trade(days_ago=29), team="EDM", as_of_date=AS_OF)
        assert recent["disruption_index"] > old["disruption_index"]

    def test_event_beyond_lookback_excluded(self):
        model = RosterDisruptionModel()
        # 31 days ago — outside 30-day lookback
        df = _single_trade(days_ago=31)
        result = model.compute(df, team="EDM", as_of_date=AS_OF)
        assert result["disruption_index"] == pytest.approx(0.0)
        assert result["n_moves_30d"] == 0

    def test_event_at_lookback_boundary_included(self):
        model = RosterDisruptionModel()
        df = _single_trade(days_ago=30)
        result = model.compute(df, team="EDM", as_of_date=AS_OF)
        assert result["n_moves_30d"] == 1


# ---------------------------------------------------------------------------
# Chaos multiplier
# ---------------------------------------------------------------------------

class TestChaosMultiplier:
    def test_no_chaos_below_threshold(self):
        model = RosterDisruptionModel()
        # 2 trades in last 72h (below threshold of 3)
        rows = [
            {"date": AS_OF.isoformat(), "event_type": "trade", "team": "EDM"},
            {"date": AS_OF.isoformat(), "event_type": "ltir", "team": "EDM"},
        ]
        result = model.compute(_tx_df(rows), team="EDM", as_of_date=AS_OF)
        assert result["chaos_multiplier"] == pytest.approx(1.0)
        assert result["n_moves_72h"] == 2

    def test_chaos_triggered_at_threshold(self):
        model = RosterDisruptionModel()
        rows = [
            {"date": AS_OF.isoformat(), "event_type": "trade", "team": "EDM"},
            {"date": AS_OF.isoformat(), "event_type": "ltir", "team": "EDM"},
            {"date": AS_OF.isoformat(), "event_type": "waiver_claim", "team": "EDM"},
        ]
        result = model.compute(_tx_df(rows), team="EDM", as_of_date=AS_OF)
        assert result["chaos_multiplier"] == pytest.approx(CHAOS_MULTIPLIER)
        assert result["n_moves_72h"] == 3

    def test_chaos_inflates_disruption_index(self):
        model = RosterDisruptionModel()
        # Without chaos
        rows_2 = [
            {"date": AS_OF.isoformat(), "event_type": "trade", "team": "EDM"},
            {"date": AS_OF.isoformat(), "event_type": "ltir", "team": "EDM"},
        ]
        no_chaos = model.compute(_tx_df(rows_2), team="EDM", as_of_date=AS_OF)
        # With chaos (3 moves)
        rows_3 = rows_2 + [{"date": AS_OF.isoformat(), "event_type": "waiver_claim", "team": "EDM"}]
        with_chaos = model.compute(_tx_df(rows_3), team="EDM", as_of_date=AS_OF)
        # chaos_mult × sum should be larger than without
        assert with_chaos["disruption_index"] > no_chaos["disruption_index"]

    def test_old_moves_not_counted_in_72h(self):
        model = RosterDisruptionModel()
        # 3 moves but all 5+ days ago — outside 72h window
        rows = [
            {"date": (AS_OF - timedelta(days=5)).isoformat(), "event_type": "trade", "team": "EDM"},
            {"date": (AS_OF - timedelta(days=5)).isoformat(), "event_type": "ltir", "team": "EDM"},
            {"date": (AS_OF - timedelta(days=5)).isoformat(), "event_type": "waiver_claim", "team": "EDM"},
        ]
        result = model.compute(_tx_df(rows), team="EDM", as_of_date=AS_OF)
        assert result["chaos_multiplier"] == pytest.approx(1.0)
        assert result["n_moves_72h"] == 0


# ---------------------------------------------------------------------------
# Secondary team (trade receiving side)
# ---------------------------------------------------------------------------

class TestSecondaryTeam:
    def test_secondary_trade_counted_for_receiving_team(self):
        model = RosterDisruptionModel()
        # EDM receives a player from TOR via trade — secondary_team = EDM
        df = _tx_df([{
            "date": AS_OF.isoformat(),
            "event_type": "trade",
            "team": "TOR",
            "secondary_team": "EDM",
            "player_or_executive": "Player B",
        }])
        result = model.compute(df, team="EDM", as_of_date=AS_OF)
        assert result["n_moves_30d"] == 1
        assert result["disruption_index"] > 0.0

    def test_secondary_non_trade_not_counted(self):
        model = RosterDisruptionModel()
        # secondary_team filter only applies to trades
        df = _tx_df([{
            "date": AS_OF.isoformat(),
            "event_type": "signing",
            "team": "TOR",
            "secondary_team": "EDM",
        }])
        result = model.compute(df, team="EDM", as_of_date=AS_OF)
        assert result["n_moves_30d"] == 0


# ---------------------------------------------------------------------------
# Front-office events excluded
# ---------------------------------------------------------------------------

class TestFrontOfficeExclusion:
    def test_front_office_events_excluded(self):
        model = RosterDisruptionModel()
        df = _tx_df([{
            "date": AS_OF.isoformat(),
            "event_type": "front_office",
            "team": "EDM",
        }])
        result = model.compute(df, team="EDM", as_of_date=AS_OF)
        assert result["n_moves_30d"] == 0


# ---------------------------------------------------------------------------
# Missing columns → warning + zero disruption
# ---------------------------------------------------------------------------

class TestMissingColumns:
    def test_missing_required_cols_warns_zero(self):
        model = RosterDisruptionModel()
        bad_df = pl.DataFrame({"description": ["trade"]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model.compute(bad_df, team="EDM", as_of_date=AS_OF)
        assert result["disruption_index"] == pytest.approx(0.0)
        assert any(issubclass(x.category, DataMissingWarning) for x in w)


# ---------------------------------------------------------------------------
# Player role weights
# ---------------------------------------------------------------------------

class TestPlayerRoles:
    def test_starter_g_higher_than_backup_g(self):
        model = RosterDisruptionModel()
        df = _tx_df([{"date": AS_OF.isoformat(), "event_type": "trade", "team": "EDM",
                       "player_or_executive": "Goalie A"}])
        starter = model.compute(df, team="EDM", as_of_date=AS_OF,
                                 player_roles={"Goalie A": "starter_g"})
        backup = model.compute(df, team="EDM", as_of_date=AS_OF,
                                player_roles={"Goalie A": "backup_g"})
        assert starter["disruption_index"] > backup["disruption_index"]

    def test_missing_player_in_roles_uses_default(self):
        model = RosterDisruptionModel()
        df = _tx_df([{"date": AS_OF.isoformat(), "event_type": "trade", "team": "EDM",
                       "player_or_executive": "Unknown Player"}])
        roles = {"Different Player": "top6_f"}
        result = model.compute(df, team="EDM", as_of_date=AS_OF, player_roles=roles)
        expected = model.compute(df, team="EDM", as_of_date=AS_OF, player_roles=None)
        assert result["disruption_index"] == pytest.approx(expected["disruption_index"])


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------

class TestComputeAll:
    def test_returns_dataframe(self):
        model = RosterDisruptionModel()
        df = _tx_df([
            {"date": AS_OF.isoformat(), "event_type": "trade", "team": "EDM"},
            {"date": AS_OF.isoformat(), "event_type": "ltir", "team": "TOR"},
        ])
        result = model.compute_all(df, as_of_date=AS_OF)
        assert isinstance(result, pl.DataFrame)
        assert len(result) >= 2

    def test_all_schema_columns_present(self):
        model = RosterDisruptionModel()
        df = _tx_df([{"date": AS_OF.isoformat(), "event_type": "trade", "team": "EDM"}])
        result = model.compute_all(df, as_of_date=AS_OF)
        for col in ROSTER_DISRUPTION_SCHEMA:
            assert col in result.columns

    def test_explicit_teams_list(self):
        model = RosterDisruptionModel()
        df = _tx_df([
            {"date": AS_OF.isoformat(), "event_type": "trade", "team": "EDM"},
            {"date": AS_OF.isoformat(), "event_type": "ltir", "team": "TOR"},
        ])
        result = model.compute_all(df, as_of_date=AS_OF, teams=["EDM", "BOS"])
        teams = result["team"].to_list()
        assert set(teams) == {"EDM", "BOS"}

    def test_empty_df_returns_empty_schema(self):
        model = RosterDisruptionModel()
        df = pl.DataFrame(schema={"team": pl.Utf8, "date": pl.Utf8,
                                   "event_type": pl.Utf8})
        result = model.compute_all(df, as_of_date=AS_OF)
        for col in ROSTER_DISRUPTION_SCHEMA:
            assert col in result.columns

    def test_no_team_column_returns_empty_schema(self):
        model = RosterDisruptionModel()
        df = pl.DataFrame({"event_type": ["trade"]})
        result = model.compute_all(df, as_of_date=AS_OF)
        for col in ROSTER_DISRUPTION_SCHEMA:
            assert col in result.columns
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Custom constructor parameters
# ---------------------------------------------------------------------------

class TestCustomParameters:
    def test_custom_half_life_changes_score(self):
        default = RosterDisruptionModel(decay_half_life=14)
        fast = RosterDisruptionModel(decay_half_life=3)
        df = _single_trade(days_ago=7)
        r_default = default.compute(df, "EDM", AS_OF)
        r_fast = fast.compute(df, "EDM", AS_OF)
        # Faster decay → 7 days ago already much lower
        assert r_fast["disruption_index"] < r_default["disruption_index"]

    def test_custom_chaos_threshold(self):
        model = RosterDisruptionModel(chaos_threshold_72h=2)
        rows = [
            {"date": AS_OF.isoformat(), "event_type": "trade", "team": "EDM"},
            {"date": AS_OF.isoformat(), "event_type": "ltir", "team": "EDM"},
        ]
        result = model.compute(_tx_df(rows), team="EDM", as_of_date=AS_OF)
        # Should trigger chaos at threshold=2
        assert result["chaos_multiplier"] == pytest.approx(CHAOS_MULTIPLIER)

    def test_custom_event_weights(self):
        custom = {**EVENT_TYPE_WEIGHTS, "trade": 0.10}
        model = RosterDisruptionModel(event_weights=custom)
        df = _single_trade(days_ago=0)
        result = model.compute(df, team="EDM", as_of_date=AS_OF)
        # With trade weight=0.10, score should be much lower than default 0.85
        default = RosterDisruptionModel()
        r_default = default.compute(df, team="EDM", as_of_date=AS_OF)
        assert result["disruption_index"] < r_default["disruption_index"]


# ---------------------------------------------------------------------------
# efficiency_modifier standalone
# ---------------------------------------------------------------------------

class TestEfficiencyModifier:
    def test_zero_disruption_is_one(self):
        model = RosterDisruptionModel()
        assert model.efficiency_modifier(0.0) == pytest.approx(1.0)

    def test_full_disruption_is_min(self):
        model = RosterDisruptionModel()
        assert model.efficiency_modifier(1.0) == pytest.approx(1.0 - MAX_EFFICIENCY_EFFECT)

    def test_half_disruption(self):
        model = RosterDisruptionModel()
        assert model.efficiency_modifier(0.5) == pytest.approx(1.0 - 0.5 * MAX_EFFICIENCY_EFFECT)

    def test_clips_above_one(self):
        model = RosterDisruptionModel()
        assert model.efficiency_modifier(2.0) == pytest.approx(1.0 - MAX_EFFICIENCY_EFFECT)

    def test_clips_below_zero(self):
        model = RosterDisruptionModel()
        assert model.efficiency_modifier(-0.5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        model = RosterDisruptionModel(
            decay_half_life=7,
            lookback_days=21,
            chaos_threshold_72h=2,
            chaos_multiplier=2.0,
            max_efficiency_effect=0.12,
            max_raw=3.0,
        )
        path = tmp_path / "rdi.pkl"
        model.save(path)
        m2 = RosterDisruptionModel.load(path)
        assert m2._half_life == pytest.approx(7)
        assert m2._lookback == 21
        assert m2._chaos_thr == 2
        assert m2._chaos_mult == pytest.approx(2.0)
        assert m2._max_effect == pytest.approx(0.12)
        assert m2._max_raw == pytest.approx(3.0)

    def test_compute_consistent_after_load(self, tmp_path):
        model = RosterDisruptionModel()
        df = _single_trade(days_ago=5)
        original = model.compute(df, "EDM", AS_OF)

        path = tmp_path / "rdi.pkl"
        model.save(path)
        loaded = RosterDisruptionModel.load(path)
        restored = loaded.compute(df, "EDM", AS_OF)

        assert restored["disruption_index"] == pytest.approx(original["disruption_index"])


# ---------------------------------------------------------------------------
# write_roster_disruption
# ---------------------------------------------------------------------------

class TestWriteRosterDisruption:
    def test_writes_parquet(self, tmp_path):
        model = RosterDisruptionModel()
        df_in = _tx_df([{"date": AS_OF.isoformat(), "event_type": "trade", "team": "EDM"}])
        result_df = model.compute_all(df_in, as_of_date=AS_OF)
        path = write_roster_disruption(result_df, tmp_path, AS_OF.isoformat())
        assert path.exists()
        loaded = pl.read_parquet(path)
        assert "team" in loaded.columns
        assert "disruption_index" in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        model = RosterDisruptionModel()
        df_in = _tx_df([])
        result_df = model.compute_all(df_in, as_of_date=AS_OF, teams=["EDM"])
        path = write_roster_disruption(result_df, tmp_path, AS_OF.isoformat())
        assert AS_OF.isoformat() in path.name
