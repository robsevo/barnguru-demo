"""Tests for models/playoff_delta.py — Feature 2.23."""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from models.playoff_delta import (
    GAME_TYPE_PLAYOFF,
    GAME_TYPE_REG,
    MODEL_VERSION,
    PLAYOFF_DELTA_SCHEMA,
    SHRINK_GAMES,
    PlayoffDeltaModel,
    _normalise_game_type,
    _safe_cf_pct,
    _safe_per60,
    shrinkage_factor,
    write_playoff_delta,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stats(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal stats DataFrame for testing."""
    base = {
        "player_id": pl.Int64,
        "player_name": pl.Utf8,
        "team": pl.Utf8,
        "position": pl.Utf8,
        "game_type": pl.Int64,
        "gp": pl.Int64,
        "toi_seconds": pl.Float64,
        "goals": pl.Float64,
        "xgf": pl.Float64,
        "cf": pl.Float64,
        "ca": pl.Float64,
        "sv_pct": pl.Float64,
        "shots_against": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=base)
    return pl.DataFrame(rows)


def _player_reg(pid=1, gp=60, toi=3600.0, goals=20.0, xgf=18.0, cf=400.0, ca=360.0):
    return {
        "player_id": pid, "player_name": "Test Player", "team": "TOR", "position": "C",
        "game_type": GAME_TYPE_REG, "gp": gp, "toi_seconds": toi,
        "goals": goals, "xgf": xgf, "cf": cf, "ca": ca,
        "sv_pct": None, "shots_against": None,
    }


def _player_po(pid=1, gp=15, toi=900.0, goals=6.0, xgf=5.0, cf=110.0, ca=90.0):
    return {
        "player_id": pid, "player_name": "Test Player", "team": "TOR", "position": "C",
        "game_type": GAME_TYPE_PLAYOFF, "gp": gp, "toi_seconds": toi,
        "goals": goals, "xgf": xgf, "cf": cf, "ca": ca,
        "sv_pct": None, "shots_against": None,
    }


def _goalie_reg(pid=99, gp=40, toi=2400.0, sv_pct=0.910, shots_against=1200.0):
    return {
        "player_id": pid, "player_name": "Goalie Guy", "team": "MTL", "position": "G",
        "game_type": GAME_TYPE_REG, "gp": gp, "toi_seconds": toi,
        "goals": None, "xgf": None, "cf": None, "ca": None,
        "sv_pct": sv_pct, "shots_against": shots_against,
    }


def _goalie_po(pid=99, gp=12, toi=720.0, sv_pct=0.930, shots_against=360.0):
    return {
        "player_id": pid, "player_name": "Goalie Guy", "team": "MTL", "position": "G",
        "game_type": GAME_TYPE_PLAYOFF, "gp": gp, "toi_seconds": toi,
        "goals": None, "xgf": None, "cf": None, "ca": None,
        "sv_pct": sv_pct, "shots_against": shots_against,
    }


# ---------------------------------------------------------------------------
# shrinkage_factor
# ---------------------------------------------------------------------------

class TestShrinkageFactor:
    def test_zero_games_returns_zero(self):
        assert shrinkage_factor(0) == 0.0

    def test_full_games_returns_one(self):
        assert shrinkage_factor(SHRINK_GAMES) == 1.0

    def test_half_games_returns_half(self):
        assert shrinkage_factor(SHRINK_GAMES // 2) == pytest.approx(0.5)

    def test_more_than_shrink_games_clamped(self):
        assert shrinkage_factor(SHRINK_GAMES * 3) == 1.0

    def test_custom_shrink_games(self):
        assert shrinkage_factor(10, shrink_games=20) == pytest.approx(0.5)

    def test_negative_clamped_to_zero(self):
        assert shrinkage_factor(-5) == 0.0

    def test_returns_float(self):
        assert isinstance(shrinkage_factor(10), float)


# ---------------------------------------------------------------------------
# _safe_per60
# ---------------------------------------------------------------------------

class TestSafePer60:
    def test_basic(self):
        assert _safe_per60(20.0, 3600.0) == pytest.approx(20.0)

    def test_none_value_returns_none(self):
        assert _safe_per60(None, 3600.0) is None

    def test_none_toi_returns_none(self):
        assert _safe_per60(20.0, None) is None

    def test_zero_toi_returns_none(self):
        assert _safe_per60(20.0, 0.0) is None

    def test_zero_goals_returns_zero(self):
        assert _safe_per60(0.0, 3600.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _safe_cf_pct
# ---------------------------------------------------------------------------

class TestSafeCfPct:
    def test_basic(self):
        assert _safe_cf_pct(60.0, 40.0) == pytest.approx(0.6)

    def test_zero_total_returns_none(self):
        assert _safe_cf_pct(0.0, 0.0) is None

    def test_none_cf_returns_none(self):
        assert _safe_cf_pct(None, 40.0) is None

    def test_none_ca_returns_none(self):
        assert _safe_cf_pct(60.0, None) is None


# ---------------------------------------------------------------------------
# _normalise_game_type
# ---------------------------------------------------------------------------

class TestNormaliseGameType:
    @pytest.mark.parametrize("val,expected", [
        (2, 2), (3, 3),
        ("2", 2), ("3", 3),
        ("R", 2), ("r", 2), ("REG", 2), ("REGULAR", 2),
        ("P", 3), ("p", 3), ("PO", 3), ("PLAYOFF", 3), ("PLAYOFFS", 3),
        (2.0, 2), (3.0, 3),
    ])
    def test_valid(self, val, expected):
        assert _normalise_game_type(val) == expected

    @pytest.mark.parametrize("val", [None, 99, "X", "bad", 0])
    def test_invalid_returns_none(self, val):
        assert _normalise_game_type(val) is None


# ---------------------------------------------------------------------------
# PlayoffDeltaModel.fit — basic
# ---------------------------------------------------------------------------

class TestPlayoffDeltaFit:
    def test_empty_input_returns_empty_schema(self):
        model = PlayoffDeltaModel()
        df = model.fit(_make_stats([]))
        assert df.is_empty()
        for col in PLAYOFF_DELTA_SCHEMA:
            assert col in df.columns

    def test_single_player_both_types(self):
        stats = _make_stats([_player_reg(), _player_po()])
        model = PlayoffDeltaModel()
        df = model.fit(stats)
        assert len(df) == 1
        row = df.row(0, named=True)
        assert row["player_id"] == 1
        assert row["reg_gp"] == 60
        assert row["playoff_gp"] == 15
        assert row["model_version"] == MODEL_VERSION

    def test_reg_only_player_included_no_playoff_data(self):
        stats = _make_stats([_player_reg()])
        model = PlayoffDeltaModel()
        df = model.fit(stats)
        assert len(df) == 1
        row = df.row(0, named=True)
        assert row["playoff_gp"] == 0
        assert row["delta_goals_per60"] is None
        assert row["shrinkage_factor"] == 0.0

    def test_output_columns_match_schema(self):
        stats = _make_stats([_player_reg(), _player_po()])
        model = PlayoffDeltaModel()
        df = model.fit(stats)
        assert set(df.columns) == set(PLAYOFF_DELTA_SCHEMA.keys())

    def test_min_reg_toi_filters_player(self):
        # toi_seconds = 100 is below default 300s threshold
        stats = _make_stats([_player_reg(toi=100.0), _player_po()])
        model = PlayoffDeltaModel()
        df = model.fit(stats)
        assert df.is_empty()

    def test_multiple_players(self):
        stats = _make_stats([
            _player_reg(pid=1), _player_po(pid=1),
            _player_reg(pid=2, gp=50), _player_po(pid=2, gp=5),
        ])
        model = PlayoffDeltaModel()
        df = model.fit(stats)
        assert len(df) == 2

    def test_delta_calculation(self):
        # reg: 20 goals in 3600s = 20 goals/60
        # po:  6 goals in 900s = 24 goals/60
        # delta = +4.0
        stats = _make_stats([
            _player_reg(goals=20.0, toi=3600.0),
            _player_po(goals=6.0, toi=900.0),
        ])
        model = PlayoffDeltaModel()
        df = model.fit(stats)
        row = df.row(0, named=True)
        assert row["reg_goals_per60"] == pytest.approx(20.0)
        assert row["playoff_goals_per60"] == pytest.approx(24.0)
        assert row["delta_goals_per60"] == pytest.approx(4.0)

    def test_shrinkage_applied_to_delta(self):
        stats = _make_stats([_player_reg(goals=20.0, toi=3600.0), _player_po(gp=15, goals=6.0, toi=900.0)])
        model = PlayoffDeltaModel()
        df = model.fit(stats)
        row = df.row(0, named=True)
        sf = 15 / SHRINK_GAMES  # 0.5
        assert row["shrinkage_factor"] == pytest.approx(sf)
        expected_shrunk = row["delta_goals_per60"] * sf
        assert row["shrunk_delta_goals_per60"] == pytest.approx(expected_shrunk)

    def test_full_shrinkage_at_30_games(self):
        stats = _make_stats([_player_reg(), _player_po(gp=30)])
        model = PlayoffDeltaModel()
        df = model.fit(stats)
        row = df.row(0, named=True)
        assert row["shrinkage_factor"] == pytest.approx(1.0)
        assert row["shrunk_delta_goals_per60"] == pytest.approx(row["delta_goals_per60"])

    def test_goalie_sv_pct_delta(self):
        stats = _make_stats([_goalie_reg(), _goalie_po()])
        model = PlayoffDeltaModel()
        df = model.fit(stats)
        assert len(df) == 1
        row = df.row(0, named=True)
        assert row["reg_sv_pct"] == pytest.approx(0.910)
        assert row["playoff_sv_pct"] == pytest.approx(0.930)
        assert row["delta_sv_pct"] == pytest.approx(0.020)

    def test_cf_pct_computed_correctly(self):
        stats = _make_stats([_player_reg(cf=400.0, ca=360.0), _player_po(cf=110.0, ca=90.0)])
        model = PlayoffDeltaModel()
        df = model.fit(stats)
        row = df.row(0, named=True)
        assert row["reg_cf_pct"] == pytest.approx(400.0 / 760.0)
        assert row["playoff_cf_pct"] == pytest.approx(110.0 / 200.0)

    def test_string_game_type_r_p(self):
        r = dict(_player_reg()); r["game_type"] = "R"
        p = dict(_player_po()); p["game_type"] = "P"
        stats = pl.DataFrame([r, p])
        model = PlayoffDeltaModel()
        df = model.fit(stats)
        assert len(df) == 1

    def test_multi_season_rows_aggregated(self):
        """Multiple reg rows for same player should be summed."""
        r1 = _player_reg(gp=60, toi=3600.0, goals=20.0, xgf=18.0, cf=400.0, ca=360.0)
        r2 = dict(r1); r2["gp"] = 50; r2["toi_seconds"] = 3000.0; r2["goals"] = 15.0
        r2["xgf"] = 14.0; r2["cf"] = 350.0; r2["ca"] = 310.0
        stats = _make_stats([r1, r2, _player_po()])
        model = PlayoffDeltaModel()
        df = model.fit(stats)
        row = df.row(0, named=True)
        # toi summed
        assert row["reg_toi_sec"] == pytest.approx(6600.0)
        # goals summed: 35 / (6600s) × 3600 = 19.09…
        assert row["reg_goals_per60"] == pytest.approx(35.0 / 6600.0 * 3600.0)


# ---------------------------------------------------------------------------
# PlayoffDeltaModel.get_modifier / apply_modifier
# ---------------------------------------------------------------------------

class TestModifierLookup:
    def _trained_model(self):
        stats = _make_stats([_player_reg(goals=20.0, toi=3600.0), _player_po(gp=30, goals=8.0, toi=900.0)])
        model = PlayoffDeltaModel()
        model.fit(stats)
        return model

    def test_get_modifier_returns_dict_keys(self):
        model = self._trained_model()
        mod = model.get_modifier(1)
        assert set(mod.keys()) == {"goals_per60", "xgf_per60", "cf_pct", "sv_pct"}

    def test_get_modifier_unknown_player_returns_zeros(self):
        model = PlayoffDeltaModel()
        mod = model.get_modifier(9999)
        assert all(v == 0.0 for v in mod.values())

    def test_apply_modifier_adds_delta(self):
        model = self._trained_model()
        mod = model.get_modifier(1)
        result = model.apply_modifier(1, base_goals_per60=20.0, base_xgf_per60=18.0)
        assert result["goals_per60"] == pytest.approx(20.0 + mod["goals_per60"])
        assert result["xgf_per60"] == pytest.approx(18.0 + mod["xgf_per60"])

    def test_apply_modifier_no_xgf(self):
        model = self._trained_model()
        result = model.apply_modifier(1, base_goals_per60=20.0)
        assert result["xgf_per60"] is None

    def test_modifier_not_stored_for_insufficient_toi(self):
        stats = _make_stats([_player_reg(toi=100.0), _player_po(gp=30)])
        model = PlayoffDeltaModel()
        model.fit(stats)
        # player was filtered out — should return zeros
        mod = model.get_modifier(1)
        assert all(v == 0.0 for v in mod.values())


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        stats = _make_stats([_player_reg(), _player_po(gp=30)])
        model = PlayoffDeltaModel()
        model.fit(stats)
        path = tmp_path / "playoff_delta.pkl"
        model.save(path)
        loaded = PlayoffDeltaModel.load(path)
        assert loaded.get_modifier(1) == model.get_modifier(1)

    def test_load_preserves_shrink_games(self, tmp_path):
        model = PlayoffDeltaModel(shrink_games=20)
        model.save(tmp_path / "m.pkl")
        loaded = PlayoffDeltaModel.load(tmp_path / "m.pkl")
        assert loaded._shrink == 20


# ---------------------------------------------------------------------------
# write_playoff_delta
# ---------------------------------------------------------------------------

class TestWritePlayoffDelta:
    def test_writes_parquet(self, tmp_path):
        stats = _make_stats([_player_reg(), _player_po()])
        model = PlayoffDeltaModel()
        df = model.fit(stats)
        path = write_playoff_delta(df, tmp_path / "out", season=2025)
        assert path.exists()
        assert path.name == "playoff_delta_2025.parquet"

    def test_written_file_is_readable(self, tmp_path):
        stats = _make_stats([_player_reg(), _player_po()])
        model = PlayoffDeltaModel()
        df = model.fit(stats)
        path = write_playoff_delta(df, tmp_path / "out", season=2025)
        back = pl.read_parquet(path)
        assert len(back) == 1

    def test_creates_output_dir(self, tmp_path):
        df = pl.DataFrame(schema={c: d for c, d in PLAYOFF_DELTA_SCHEMA.items()})
        deep = tmp_path / "a" / "b" / "c"
        write_playoff_delta(df, deep, season=2024)
        assert deep.exists()
