"""Tests for Feature 3.20 — TradeIntegrationModel."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from models.rapm_model import DataMissingWarning
from models.trade_integration import (
    CI_HALF_WIDTH,
    DECAY_GAMES_BY_POSITION,
    DEFAULT_DECAY_GAMES,
    DEFAULT_FIT_SCORE,
    FIT_DELTA_MAX,
    TRADE_INTEGRATION_SCHEMA,
    TRAVEL_PENALTY_GAMES,
    TRAVEL_PENALTY_MAX,
    TradeIntegrationModel,
    _decay_factor,
    _decay_games,
    _fit_delta,
    _normalize_position,
    _travel_penalty,
    write_trade_integration,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _trades(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "player_id":  pl.Int64,
        "trade_date": pl.Utf8,
        "new_team":   pl.Utf8,
        "old_team":   pl.Utf8,
        "position":   pl.Utf8,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    defaults = {"position": "C"}
    return pl.DataFrame([{**defaults, **r} for r in rows], schema=schema)


def _obs(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "player_id": pl.Int64,
        "game_id":   pl.Int64,
        "game_date": pl.Utf8,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    defaults = {"game_id": 0}
    return pl.DataFrame([{**defaults, **r} for r in rows], schema=schema)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_normalize_position_left_to_lw(self):
        assert _normalize_position("L") == "LW"
        assert _normalize_position("R") == "RW"
        assert _normalize_position("C") == "C"
        assert _normalize_position("D") == "D"
        assert _normalize_position(None) == ""
        assert _normalize_position("") == ""

    def test_decay_games_per_position(self):
        assert _decay_games("LW") == DECAY_GAMES_BY_POSITION["LW"]
        assert _decay_games("L")  == DECAY_GAMES_BY_POSITION["L"]
        assert _decay_games("C")  == DECAY_GAMES_BY_POSITION["C"]
        assert _decay_games("D")  == DECAY_GAMES_BY_POSITION["D"]
        assert _decay_games("ZZ") == DEFAULT_DECAY_GAMES
        assert _decay_games(None) == DEFAULT_DECAY_GAMES

    def test_decay_factor_monotone_decreasing(self):
        n = DEFAULT_DECAY_GAMES
        assert _decay_factor(0, n) == pytest.approx(1.0)
        assert _decay_factor(n // 2, n) == pytest.approx(0.5)
        assert _decay_factor(n, n) == pytest.approx(0.0)
        assert _decay_factor(n + 5, n) == pytest.approx(0.0)

    def test_decay_factor_negative_returns_zero(self):
        assert _decay_factor(-1, 10) == 0.0

    def test_travel_penalty_full_at_game_zero(self):
        assert _travel_penalty(0) == pytest.approx(TRAVEL_PENALTY_MAX)

    def test_travel_penalty_decays_to_zero(self):
        # Past TRAVEL_PENALTY_GAMES → zero.
        assert _travel_penalty(TRAVEL_PENALTY_GAMES) == pytest.approx(0.0)
        assert _travel_penalty(TRAVEL_PENALTY_GAMES + 5) == pytest.approx(0.0)

    def test_fit_delta_missing_lookup_zero(self):
        assert _fit_delta(None, 1, "TOR", "MTL") == pytest.approx(0.0)

    def test_fit_delta_basic_swing(self):
        lookup = {(1, "TOR"): 0.8, (1, "MTL"): 0.3}
        assert _fit_delta(lookup, 1, "TOR", "MTL") == pytest.approx(0.5)
        assert _fit_delta(lookup, 1, "MTL", "TOR") == pytest.approx(-0.5)

    def test_fit_delta_missing_one_side_defaults(self):
        lookup = {(1, "TOR"): 0.8}
        delta = _fit_delta(lookup, 1, "TOR", "MTL")
        assert delta == pytest.approx(0.8 - DEFAULT_FIT_SCORE)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construction(self):
        m = TradeIntegrationModel()
        assert m._version  # smoke

    def test_rejects_negative_fit_delta_max(self):
        with pytest.raises(ValueError):
            TradeIntegrationModel(fit_delta_max=-0.01)

    def test_rejects_negative_travel_max(self):
        with pytest.raises(ValueError):
            TradeIntegrationModel(travel_penalty_max=-0.01)

    def test_rejects_negative_travel_games(self):
        with pytest.raises(ValueError):
            TradeIntegrationModel(travel_penalty_games=-1)

    def test_rejects_negative_ci_half(self):
        with pytest.raises(ValueError):
            TradeIntegrationModel(ci_half_width=-0.01)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_trade_columns_warns(self):
        bad = pl.DataFrame({"foo": [1]})
        good = _obs([{"player_id": 1, "game_date": "2026-01-15"}])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = TradeIntegrationModel().compute(bad, good, "2026-05-17")
        assert len(out) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_missing_observation_columns_warns(self):
        trades = _trades([{
            "player_id": 1, "trade_date": "2026-01-01",
            "new_team": "TOR", "old_team": "MTL",
        }])
        bad = pl.DataFrame({"foo": [1]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = TradeIntegrationModel().compute(trades, bad, "2026-05-17")
        assert len(out) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_bad_as_of_date_raises(self):
        trades = _trades([{
            "player_id": 1, "trade_date": "2026-01-01",
            "new_team": "TOR", "old_team": "MTL",
        }])
        obs = _obs([{"player_id": 1, "game_date": "2026-01-15"}])
        with pytest.raises(ValueError):
            TradeIntegrationModel().compute(trades, obs, "not-a-date")

    def test_empty_trades_returns_empty(self):
        out = TradeIntegrationModel().compute(_trades([]), _obs([
            {"player_id": 1, "game_date": "2026-01-15"},
        ]), "2026-05-17")
        for c in TRADE_INTEGRATION_SCHEMA:
            assert c in out.columns
        assert len(out) == 0

    def test_empty_observations_returns_empty(self):
        out = TradeIntegrationModel().compute(_trades([{
            "player_id": 1, "trade_date": "2026-01-01",
            "new_team": "TOR", "old_team": "MTL",
        }]), _obs([]), "2026-05-17")
        assert len(out) == 0

    def test_output_schema(self):
        trades = _trades([{
            "player_id": 1, "trade_date": "2026-01-01",
            "new_team": "TOR", "old_team": "MTL",
        }])
        obs = _obs([{"player_id": 1, "game_date": "2026-01-15"}])
        out = TradeIntegrationModel().compute(trades, obs, "2026-05-17")
        assert set(out.columns) == set(TRADE_INTEGRATION_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Direction and decay behaviour
# ---------------------------------------------------------------------------

class TestDirection:
    def test_positive_fit_delta_is_boost(self):
        trades = _trades([{
            "player_id": 1, "trade_date": "2026-01-01",
            "new_team": "TOR", "old_team": "MTL", "position": "C",
        }])
        obs = _obs([{"player_id": 1, "game_date": "2026-01-02"}])
        fit = {(1, "TOR"): 0.9, (1, "MTL"): 0.2}
        out = TradeIntegrationModel().compute(
            trades, obs, "2026-05-17", fit_scores=fit,
        ).to_dicts()[0]
        # +ve fit_delta → integration_factor net negative (boost), but
        # travel penalty kicks in at game 0; net should still be negative
        # because fit term dominates.
        assert out["fit_delta"] == pytest.approx(0.7)
        assert out["integration_factor"] < 0.0

    def test_negative_fit_delta_is_drag(self):
        trades = _trades([{
            "player_id": 1, "trade_date": "2026-01-01",
            "new_team": "TOR", "old_team": "MTL", "position": "C",
        }])
        obs = _obs([{"player_id": 1, "game_date": "2026-01-02"}])
        fit = {(1, "TOR"): 0.2, (1, "MTL"): 0.9}
        out = TradeIntegrationModel().compute(
            trades, obs, "2026-05-17", fit_scores=fit,
        ).to_dicts()[0]
        assert out["fit_delta"] == pytest.approx(-0.7)
        # negative fit → drag (positive integration_factor) + travel.
        assert out["integration_factor"] > 0.0

    def test_zero_fit_delta_is_travel_only(self):
        trades = _trades([{
            "player_id": 1, "trade_date": "2026-01-01",
            "new_team": "TOR", "old_team": "MTL", "position": "C",
        }])
        obs = _obs([{"player_id": 1, "game_date": "2026-01-02"}])
        # No fit_scores → fit_delta = 0.
        out = TradeIntegrationModel().compute(
            trades, obs, "2026-05-17",
        ).to_dicts()[0]
        assert out["fit_delta"] == pytest.approx(0.0)
        assert out["integration_factor"] > 0.0
        assert out["integration_factor"] <= TRAVEL_PENALTY_MAX + 1e-9

    def test_high_uncertainty_ci(self):
        trades = _trades([{
            "player_id": 1, "trade_date": "2026-01-01",
            "new_team": "TOR", "old_team": "MTL", "position": "C",
        }])
        obs = _obs([{"player_id": 1, "game_date": "2026-01-02"}])
        out = TradeIntegrationModel().compute(
            trades, obs, "2026-05-17",
        ).to_dicts()[0]
        spread = out["ci_high"] - out["ci_low"]
        assert spread == pytest.approx(2 * CI_HALF_WIDTH)


# ---------------------------------------------------------------------------
# Decay and integration window
# ---------------------------------------------------------------------------

class TestDecay:
    def test_winger_decays_faster_than_center(self):
        # Same trade, same fit_delta, two players: LW vs C
        trades = _trades([
            {"player_id": 1, "trade_date": "2026-01-01",
             "new_team": "TOR", "old_team": "MTL", "position": "LW"},
            {"player_id": 2, "trade_date": "2026-01-01",
             "new_team": "TOR", "old_team": "MTL", "position": "C"},
        ])
        # 14 calendar days ≈ 4 games. LW window=15, C window=20.
        obs = _obs([
            {"player_id": 1, "game_date": "2026-01-15"},
            {"player_id": 2, "game_date": "2026-01-15"},
        ])
        fit = {(1, "TOR"): 0.9, (1, "MTL"): 0.2,
               (2, "TOR"): 0.9, (2, "MTL"): 0.2}
        out = TradeIntegrationModel().compute(
            trades, obs, "2026-05-17", fit_scores=fit,
        )
        lw = out.filter(pl.col("player_id") == 1).to_dicts()[0]
        ctr = out.filter(pl.col("player_id") == 2).to_dicts()[0]
        # Same games_since (calendar proxy), LW decays faster → smaller decay.
        assert lw["games_since_trade"] == ctr["games_since_trade"]
        assert lw["decay_factor"] < ctr["decay_factor"]

    def test_past_decay_window_filtered_out(self):
        # 200 calendar days ≈ 57 games, well past any decay window.
        trades = _trades([{
            "player_id": 1, "trade_date": "2025-06-01",
            "new_team": "TOR", "old_team": "MTL", "position": "C",
        }])
        obs = _obs([{"player_id": 1, "game_date": "2026-01-15"}])
        out = TradeIntegrationModel().compute(trades, obs, "2026-05-17")
        assert len(out) == 0

    def test_no_trade_means_no_output(self):
        trades = _trades([{
            "player_id": 1, "trade_date": "2026-02-01",
            "new_team": "TOR", "old_team": "MTL",
        }])
        # game is BEFORE the trade
        obs = _obs([{"player_id": 1, "game_date": "2026-01-15"}])
        out = TradeIntegrationModel().compute(trades, obs, "2026-05-17")
        assert len(out) == 0


# ---------------------------------------------------------------------------
# Output bounds — the spec requires the modifier live in a sane range.
# ---------------------------------------------------------------------------

class TestOutputBounds:
    def test_factor_within_documented_bounds(self):
        # Stress test: a few thousand random-ish trades.
        rows_t = []
        rows_o = []
        for pid in range(1, 50):
            rows_t.append({
                "player_id": pid, "trade_date": "2026-01-01",
                "new_team": "TOR", "old_team": "MTL", "position": "C",
            })
            for d in range(1, 20):
                rows_o.append({
                    "player_id": pid,
                    "game_date": f"2026-01-{d:02d}",
                })
        out = TradeIntegrationModel().compute(
            _trades(rows_t), _obs(rows_o), "2026-05-17",
        )
        factors = out["integration_factor"].to_list()
        # |fit_term| ≤ FIT_DELTA_MAX, travel ≤ TRAVEL_PENALTY_MAX, decay ≤ 1
        max_abs = FIT_DELTA_MAX + TRAVEL_PENALTY_MAX
        assert all(-max_abs - 1e-9 <= f <= max_abs + 1e-9 for f in factors)


# ---------------------------------------------------------------------------
# Games-played lookup integration
# ---------------------------------------------------------------------------

class TestGamesPlayedLookup:
    def test_lookup_overrides_calendar_proxy(self):
        trades = _trades([{
            "player_id": 1, "trade_date": "2026-01-01",
            "new_team": "TOR", "old_team": "MTL", "position": "C",
        }])
        obs = _obs([{"player_id": 1, "game_date": "2026-01-15"}])
        gp = {
            (1, "2026-01-01"): 30,
            (1, "2026-01-15"): 33,
        }
        out = TradeIntegrationModel().compute(
            trades, obs, "2026-05-17", games_played_lookup=gp,
        ).to_dicts()[0]
        # gp diff = 3 (not the ~4 the calendar proxy would yield)
        assert out["games_since_trade"] == 3


# ---------------------------------------------------------------------------
# Multi-player and multi-trade behaviour
# ---------------------------------------------------------------------------

class TestMultiTrade:
    def test_most_recent_trade_wins(self):
        trades = _trades([
            {"player_id": 1, "trade_date": "2025-06-01",
             "new_team": "MTL", "old_team": "EDM", "position": "C"},
            {"player_id": 1, "trade_date": "2026-01-01",
             "new_team": "TOR", "old_team": "MTL", "position": "C"},
        ])
        obs = _obs([{"player_id": 1, "game_date": "2026-01-15"}])
        out = TradeIntegrationModel().compute(
            trades, obs, "2026-05-17",
        ).to_dicts()[0]
        assert out["new_team"] == "TOR"
        assert out["old_team"] == "MTL"

    def test_independent_players(self):
        trades = _trades([
            {"player_id": 1, "trade_date": "2026-01-01",
             "new_team": "TOR", "old_team": "MTL"},
            {"player_id": 2, "trade_date": "2026-01-05",
             "new_team": "MTL", "old_team": "BOS"},
        ])
        obs = _obs([
            {"player_id": 1, "game_date": "2026-01-15"},
            {"player_id": 2, "game_date": "2026-01-15"},
        ])
        out = TradeIntegrationModel().compute(trades, obs, "2026-05-17")
        assert set(out["player_id"].to_list()) == {1, 2}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_invalid_player_id_dropped(self):
        trades = pl.DataFrame(
            [{"player_id": None, "trade_date": "2026-01-01",
              "new_team": "TOR", "old_team": "MTL", "position": "C"}],
            schema={
                "player_id":  pl.Int64,
                "trade_date": pl.Utf8,
                "new_team":   pl.Utf8,
                "old_team":   pl.Utf8,
                "position":   pl.Utf8,
            },
        )
        obs = _obs([{"player_id": 1, "game_date": "2026-01-15"}])
        out = TradeIntegrationModel().compute(trades, obs, "2026-05-17")
        # No usable trades → no rows
        assert len(out) == 0

    def test_invalid_trade_date_dropped(self):
        trades = _trades([
            {"player_id": 1, "trade_date": "nope",
             "new_team": "TOR", "old_team": "MTL", "position": "C"},
            {"player_id": 1, "trade_date": "2026-01-01",
             "new_team": "TOR", "old_team": "MTL", "position": "C"},
        ])
        obs = _obs([{"player_id": 1, "game_date": "2026-01-15"}])
        out = TradeIntegrationModel().compute(trades, obs, "2026-05-17")
        assert len(out) == 1
        assert out["trade_date"].to_list() == ["2026-01-01"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_parquet(self, tmp_path):
        trades = _trades([{
            "player_id": 1, "trade_date": "2026-01-01",
            "new_team": "TOR", "old_team": "MTL", "position": "C",
        }])
        obs = _obs([{"player_id": 1, "game_date": "2026-01-02"}])
        out = TradeIntegrationModel().compute(trades, obs, "2026-05-17")
        path = write_trade_integration(out, tmp_path, "2026-05-17")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in TRADE_INTEGRATION_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        out = TradeIntegrationModel().compute(_trades([]), _obs([]), "2026-05-17")
        path = write_trade_integration(out, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name
