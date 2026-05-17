"""Tests for Feature 3.13 — InjuryStatusIntegrator."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from models.injury_status_integrator import (
    DEFAULT_AVAILABILITY,
    INJURY_STATUS_SCHEMA,
    NO_RETURN_SENTINEL,
    RUST_FLOOR,
    RUST_RECOVERY_GAMES,
    STATUS_DTD,
    STATUS_HEALTHY,
    STATUS_OUT,
    InjuryStatusIntegrator,
    rust_factor,
    write_injury_status,
)
from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _history(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "player_id":     pl.Int64,
                "observed_date": pl.Utf8,
                "status":        pl.Utf8,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "player_id":     pl.Int64,
            "observed_date": pl.Utf8,
            "status":        pl.Utf8,
        },
    )


def _games(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={"player_id": pl.Int64, "game_date": pl.Utf8}
        )
    return pl.DataFrame(
        rows, schema={"player_id": pl.Int64, "game_date": pl.Utf8}
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_history_columns_warns(self):
        bad = pl.DataFrame({"foo": ["bar"]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = InjuryStatusIntegrator().compute(bad, _games([]), "2026-05-17")
        assert len(out) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_missing_games_columns_warns(self):
        hist = _history([{"player_id": 1, "observed_date": "2026-05-01",
                          "status": STATUS_OUT}])
        bad_games = pl.DataFrame({"foo": [1]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = InjuryStatusIntegrator().compute(hist, bad_games, "2026-05-17")
        assert len(out) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_inputs_returns_empty_schema(self):
        out = InjuryStatusIntegrator().compute(_history([]), _games([]),
                                               "2026-05-17")
        for col in INJURY_STATUS_SCHEMA:
            assert col in out.columns
        assert len(out) == 0

    def test_output_schema(self):
        hist = _history([{"player_id": 1, "observed_date": "2026-05-01",
                          "status": STATUS_OUT}])
        out = InjuryStatusIntegrator().compute(hist, _games([]), "2026-05-17")
        assert set(out.columns) == set(INJURY_STATUS_SCHEMA.keys())

    def test_unknown_status_warns_and_skipped(self):
        hist = _history([{"player_id": 1, "observed_date": "2026-05-01",
                          "status": "PROBABLE"}])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = InjuryStatusIntegrator().compute(hist, _games([]),
                                                   "2026-05-17")
        assert any(issubclass(x.category, DataMissingWarning) for x in w)
        # Unknown observation dropped → player effectively unknown → no rows
        assert len(out) == 0

    def test_bad_as_of_date_raises(self):
        hist = _history([{"player_id": 1, "observed_date": "2026-05-01",
                          "status": STATUS_OUT}])
        with pytest.raises(ValueError):
            InjuryStatusIntegrator().compute(hist, _games([]), "not-a-date")

    def test_bad_params_rejected(self):
        with pytest.raises(ValueError):
            InjuryStatusIntegrator(rust_recovery_games=0)
        with pytest.raises(ValueError):
            InjuryStatusIntegrator(rust_floor=-0.1)
        with pytest.raises(ValueError):
            InjuryStatusIntegrator(rust_floor=1.5)
        with pytest.raises(ValueError):
            InjuryStatusIntegrator(availability_table={STATUS_OUT: 1.5})


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_rust_floor_in_unit_interval(self):
        assert 0.0 <= RUST_FLOOR <= 1.0

    def test_recovery_games_positive(self):
        # Plan says "games 1–7 post-return"
        assert RUST_RECOVERY_GAMES >= 1

    def test_no_return_sentinel_is_negative(self):
        assert NO_RETURN_SENTINEL < 0

    def test_default_availability_keys(self):
        assert set(DEFAULT_AVAILABILITY) == {STATUS_HEALTHY, STATUS_DTD,
                                             STATUS_OUT}
        assert DEFAULT_AVAILABILITY[STATUS_HEALTHY] == 1.0
        assert DEFAULT_AVAILABILITY[STATUS_OUT] == 0.0
        assert 0.0 < DEFAULT_AVAILABILITY[STATUS_DTD] < 1.0


# ---------------------------------------------------------------------------
# Rust factor formula
# ---------------------------------------------------------------------------

class TestRustFactor:
    def test_no_return_is_one(self):
        assert rust_factor(0) == pytest.approx(1.0)
        assert rust_factor(-1) == pytest.approx(1.0)

    def test_first_game_is_floor(self):
        assert rust_factor(1) == pytest.approx(RUST_FLOOR)

    def test_full_recovery_at_recovery_games(self):
        assert rust_factor(RUST_RECOVERY_GAMES) == pytest.approx(1.0)

    def test_past_recovery_stays_one(self):
        assert rust_factor(RUST_RECOVERY_GAMES + 10) == pytest.approx(1.0)

    def test_monotone_increasing(self):
        prev = rust_factor(1)
        for g in range(2, RUST_RECOVERY_GAMES + 1):
            curr = rust_factor(g)
            assert curr >= prev
            prev = curr

    def test_custom_floor(self):
        assert rust_factor(1, floor=0.5) == pytest.approx(0.5)
        assert rust_factor(7, floor=0.5) == pytest.approx(1.0)

    def test_output_in_unit_interval(self):
        for g in range(0, 15):
            v = rust_factor(g)
            assert RUST_FLOOR <= v <= 1.0 or v == 1.0


# ---------------------------------------------------------------------------
# Compute path
# ---------------------------------------------------------------------------

class TestCompute:
    def test_currently_out_has_zero_rust(self):
        hist = _history([
            {"player_id": 1, "observed_date": "2026-05-10", "status": STATUS_OUT},
        ])
        out = InjuryStatusIntegrator().compute(hist, _games([]),
                                               "2026-05-17").to_dicts()[0]
        assert out["status"] == STATUS_OUT
        assert out["availability_probability"] == pytest.approx(0.0)
        assert out["rust_factor"] == pytest.approx(0.0)
        assert out["games_since_return"] == NO_RETURN_SENTINEL

    def test_currently_dtd_has_dtd_availability(self):
        hist = _history([
            {"player_id": 1, "observed_date": "2026-05-10", "status": STATUS_DTD},
        ])
        out = InjuryStatusIntegrator().compute(hist, _games([]),
                                               "2026-05-17").to_dicts()[0]
        assert out["status"] == STATUS_DTD
        assert out["availability_probability"] == pytest.approx(
            DEFAULT_AVAILABILITY[STATUS_DTD]
        )

    def test_player_with_no_history_is_healthy(self):
        # Player only shows up via games_played → implicit HEALTHY.
        games = _games([
            {"player_id": 42, "game_date": "2026-05-10"},
        ])
        out = InjuryStatusIntegrator().compute(_history([]), games,
                                               "2026-05-17").to_dicts()[0]
        assert out["player_id"] == 42
        assert out["status"] == STATUS_HEALTHY
        assert out["availability_probability"] == pytest.approx(1.0)
        assert out["games_since_return"] == NO_RETURN_SENTINEL
        assert out["rust_factor"] == pytest.approx(1.0)
        assert out["return_date"] == ""

    def test_return_from_injury_counts_games(self):
        # Out on May 1, came back on May 5; played 3 games since.
        hist = _history([
            {"player_id": 7, "observed_date": "2026-05-01", "status": STATUS_OUT},
            {"player_id": 7, "observed_date": "2026-05-05", "status": STATUS_HEALTHY},
        ])
        games = _games([
            {"player_id": 7, "game_date": "2026-04-20"},   # pre-injury, ignored
            {"player_id": 7, "game_date": "2026-05-05"},   # return game = game 1
            {"player_id": 7, "game_date": "2026-05-07"},   # game 2
            {"player_id": 7, "game_date": "2026-05-10"},   # game 3
        ])
        out = InjuryStatusIntegrator().compute(hist, games,
                                               "2026-05-17").to_dicts()[0]
        assert out["status"] == STATUS_HEALTHY
        assert out["return_date"] == "2026-05-05"
        assert out["games_since_return"] == 3
        assert out["rust_factor"] == pytest.approx(rust_factor(3))
        assert 0.0 < out["rust_factor"] < 1.0

    def test_rust_factor_full_after_recovery_games(self):
        # 7 games after return → full recovery.
        hist = _history([
            {"player_id": 7, "observed_date": "2026-04-01", "status": STATUS_OUT},
            {"player_id": 7, "observed_date": "2026-04-05", "status": STATUS_HEALTHY},
        ])
        games = _games([
            {"player_id": 7, "game_date": f"2026-04-{day:02d}"}
            for day in (5, 7, 9, 11, 13, 15, 17, 19, 21)   # 9 games since return
        ])
        out = InjuryStatusIntegrator().compute(hist, games,
                                               "2026-05-17").to_dicts()[0]
        assert out["games_since_return"] >= RUST_RECOVERY_GAMES
        assert out["rust_factor"] == pytest.approx(1.0)

    def test_back_to_out_drops_rust_to_zero(self):
        # Was out, came back, played one game, then was placed OUT again.
        hist = _history([
            {"player_id": 7, "observed_date": "2026-04-01", "status": STATUS_OUT},
            {"player_id": 7, "observed_date": "2026-04-05", "status": STATUS_HEALTHY},
            {"player_id": 7, "observed_date": "2026-04-10", "status": STATUS_OUT},
        ])
        games = _games([
            {"player_id": 7, "game_date": "2026-04-06"},
        ])
        out = InjuryStatusIntegrator().compute(hist, games,
                                               "2026-05-17").to_dicts()[0]
        assert out["status"] == STATUS_OUT
        assert out["rust_factor"] == pytest.approx(0.0)
        # Return date remains the prior transition for the record
        assert out["return_date"] == "2026-04-05"

    def test_future_observations_ignored(self):
        hist = _history([
            {"player_id": 7, "observed_date": "2026-05-10", "status": STATUS_OUT},
            {"player_id": 7, "observed_date": "2026-06-01", "status": STATUS_HEALTHY},
        ])
        out = InjuryStatusIntegrator().compute(hist, _games([]),
                                               "2026-05-15").to_dicts()[0]
        # As of 2026-05-15 the player is still OUT (future status not used).
        assert out["status"] == STATUS_OUT

    def test_no_return_when_never_out(self):
        hist = _history([
            {"player_id": 7, "observed_date": "2026-05-01", "status": STATUS_DTD},
            {"player_id": 7, "observed_date": "2026-05-10", "status": STATUS_HEALTHY},
        ])
        games = _games([
            {"player_id": 7, "game_date": "2026-05-12"},
        ])
        out = InjuryStatusIntegrator().compute(hist, games,
                                               "2026-05-17").to_dicts()[0]
        assert out["games_since_return"] == NO_RETURN_SENTINEL
        assert out["rust_factor"] == pytest.approx(1.0)
        assert out["return_date"] == ""

    def test_multiple_players_each_row(self):
        hist = _history([
            {"player_id": 1, "observed_date": "2026-05-01", "status": STATUS_OUT},
            {"player_id": 2, "observed_date": "2026-05-02", "status": STATUS_DTD},
        ])
        games = _games([{"player_id": 3, "game_date": "2026-05-05"}])
        out = InjuryStatusIntegrator().compute(hist, games, "2026-05-17")
        assert set(out["player_id"].to_list()) == {1, 2, 3}

    def test_zero_games_since_return_keeps_rust_one(self):
        # Player returned today (as of as_of_date) but hasn't played yet
        # → games_since_return = 0 → rust = 1.0 (no game played to be rusty in).
        hist = _history([
            {"player_id": 7, "observed_date": "2026-05-01", "status": STATUS_OUT},
            {"player_id": 7, "observed_date": "2026-05-17", "status": STATUS_HEALTHY},
        ])
        out = InjuryStatusIntegrator().compute(hist, _games([]),
                                               "2026-05-17").to_dicts()[0]
        assert out["status"] == STATUS_HEALTHY
        assert out["games_since_return"] == 0
        assert out["rust_factor"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_parquet(self, tmp_path):
        hist = _history([{"player_id": 1, "observed_date": "2026-05-01",
                          "status": STATUS_OUT}])
        out = InjuryStatusIntegrator().compute(hist, _games([]), "2026-05-17")
        path = write_injury_status(out, tmp_path, "2026-05-17")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in INJURY_STATUS_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        out = InjuryStatusIntegrator().compute(_history([]), _games([]),
                                               "2026-05-17")
        path = write_injury_status(out, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name
