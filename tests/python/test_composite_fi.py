"""Tests for Feature 3.17 — CompositeFatigueIndex."""

from __future__ import annotations

import json
import warnings

import polars as pl
import pytest

from models.composite_fi import (
    COMPOSITE_FI_SCHEMA,
    DEFAULT_WEIGHTS,
    SCALE_MILES_7D,
    SCALE_TOI_SPIKE_Z,
    CompositeFatigueIndex,
    write_composite_fi,
)
from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SIGNALS_SCHEMA = {
    "player_id":                       pl.Int64,
    "game_id":                         pl.Int64,
    "game_date":                       pl.Utf8,
    "is_b2b":                          pl.Boolean,
    "is_3_in_4":                       pl.Boolean,
    "games_last_7d":                   pl.Int64,
    "miles_last_7d":                   pl.Float64,
    "tz_load_48h":                     pl.Float64,
    "circadian_misalignment_hours":    pl.Float64,
    "altitude_penalty":                pl.Float64,
    "toi_spike_z":                     pl.Float64,
    "st_intensity_score":              pl.Float64,
    "contact_load_score":              pl.Float64,
    "ot_equiv_toi_secs":               pl.Float64,
    "fight_score":                     pl.Float64,
    "team_strain_score":               pl.Float64,
    "recovery_coefficient":            pl.Float64,
    "fatigue_sensitivity_multiplier":  pl.Float64,
    "rust_factor":                     pl.Float64,
    "playoff_load_penalty":            pl.Float64,
}


def _signals(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "is_b2b":          False,
        "is_3_in_4":       False,
        "games_last_7d":   0,
        "miles_last_7d":   0.0,
        "tz_load_48h":     0.0,
        "circadian_misalignment_hours": 0.0,
        "altitude_penalty":             0.0,
        "toi_spike_z":     0.0,
        "st_intensity_score":   0.0,
        "contact_load_score":   0.0,
        "ot_equiv_toi_secs":    0.0,
        "fight_score":          0.0,
        "team_strain_score":    0.0,
        "recovery_coefficient": 1.0,
        "fatigue_sensitivity_multiplier": 1.0,
        "rust_factor":          1.0,
        "playoff_load_penalty": 0.0,
        "game_id":              1,
    }
    if not rows:
        return pl.DataFrame(schema=_SIGNALS_SCHEMA)
    return pl.DataFrame([{**defaults, **r} for r in rows], schema=_SIGNALS_SCHEMA)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_columns_warns(self):
        bad = pl.DataFrame({"foo": [1]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = CompositeFatigueIndex().compute(bad, "2026-05-17")
        assert len(out) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_input_returns_empty_schema(self):
        out = CompositeFatigueIndex().compute(_signals([]), "2026-05-17")
        for col in COMPOSITE_FI_SCHEMA:
            assert col in out.columns
        assert len(out) == 0

    def test_output_schema(self):
        df = _signals([{"player_id": 1, "game_date": "2026-05-17"}])
        out = CompositeFatigueIndex().compute(df, "2026-05-17")
        assert set(out.columns) == set(COMPOSITE_FI_SCHEMA.keys())

    def test_bad_as_of_date_raises(self):
        df = _signals([{"player_id": 1, "game_date": "2026-05-17"}])
        with pytest.raises(ValueError):
            CompositeFatigueIndex().compute(df, "not-a-date")

    def test_unknown_weight_key_rejected(self):
        with pytest.raises(ValueError):
            CompositeFatigueIndex(weights={"bogus": 0.1})

    def test_weight_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            CompositeFatigueIndex(weights={"toi_load": 1.5})
        with pytest.raises(ValueError):
            CompositeFatigueIndex(weights={"toi_load": -0.1})

    def test_recovery_floor_validation(self):
        with pytest.raises(ValueError):
            CompositeFatigueIndex(recovery_floor=0.0)
        with pytest.raises(ValueError):
            CompositeFatigueIndex(recovery_floor=1.5)


# ---------------------------------------------------------------------------
# Default weights are auditable
# ---------------------------------------------------------------------------

class TestDefaultWeights:
    def test_named_keys_only(self):
        assert set(DEFAULT_WEIGHTS) == {
            "sched_density_load", "travel_load", "tz_load",
            "circadian_load", "altitude_load", "toi_load",
            "st_load", "contact_load", "ot_load", "fight_load",
            "strain_load",
        }

    def test_all_in_unit_interval(self):
        for v in DEFAULT_WEIGHTS.values():
            assert 0.0 <= v <= 1.0

    def test_total_weight_saturates_to_one(self):
        # When every signal is fully saturated, FI should hit 1.0.
        model = CompositeFatigueIndex()
        total = sum(model.weights.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_weights_property_returns_copy(self):
        model = CompositeFatigueIndex()
        w = model.weights
        w["toi_load"] = 999.0
        assert model.weights["toi_load"] != 999.0


# ---------------------------------------------------------------------------
# Compute path — fundamentals
# ---------------------------------------------------------------------------

class TestComputeFundamentals:
    def test_rested_player_zero_fi(self):
        df = _signals([{"player_id": 1, "game_date": "2026-05-17"}])
        out = CompositeFatigueIndex().compute(df, "2026-05-17").to_dicts()[0]
        assert out["fatigue_index"] == pytest.approx(0.0)
        assert out["raw_load"] == pytest.approx(0.0)
        assert out["rust_load"] == pytest.approx(0.0)

    def test_full_saturation_hits_one(self):
        # Every signal maxed → raw_load ≈ 1.0 → FI clamped to 1.0.
        df = _signals([{
            "player_id": 1, "game_date": "2026-05-17",
            "is_b2b": True, "is_3_in_4": True, "games_last_7d": 5,
            "miles_last_7d": SCALE_MILES_7D * 3,
            "tz_load_48h": 10.0,
            "circadian_misalignment_hours": 8.0,
            "altitude_penalty": 1.5,
            "toi_spike_z": SCALE_TOI_SPIKE_Z + 1.0,
            "st_intensity_score": 2.0,
            "contact_load_score": 2.0,
            "ot_equiv_toi_secs": 1500.0,
            "fight_score": 2.0,
            "team_strain_score": 1.0,
        }])
        out = CompositeFatigueIndex().compute(df, "2026-05-17").to_dicts()[0]
        assert out["fatigue_index"] == pytest.approx(1.0)
        assert out["raw_load"] >= 0.99

    def test_b2b_only_produces_partial_load(self):
        df = _signals([{"player_id": 1, "game_date": "2026-05-17",
                        "is_b2b": True}])
        out = CompositeFatigueIndex().compute(df, "2026-05-17").to_dicts()[0]
        # b2b alone contributes 0.5 of sched_density_load weight.
        expected = 0.50 * DEFAULT_WEIGHTS["sched_density_load"]
        assert out["raw_load"] == pytest.approx(expected, abs=1e-9)
        assert 0.0 < out["fatigue_index"] < 1.0

    def test_rust_factor_adds_to_fi(self):
        df = _signals([{"player_id": 1, "game_date": "2026-05-17",
                        "rust_factor": 0.85}])
        out = CompositeFatigueIndex().compute(df, "2026-05-17").to_dicts()[0]
        assert out["rust_load"] == pytest.approx(0.15, abs=1e-9)
        assert out["fatigue_index"] == pytest.approx(0.15, abs=1e-9)

    def test_playoff_load_adds_to_fi(self):
        df = _signals([{"player_id": 1, "game_date": "2026-05-17",
                        "playoff_load_penalty": 0.05}])
        out = CompositeFatigueIndex().compute(df, "2026-05-17").to_dicts()[0]
        assert out["fatigue_index"] == pytest.approx(0.05, abs=1e-9)

    def test_age_recovery_amplifies(self):
        # Same load with age recovery 0.6 → base / 0.6 = larger FI.
        df_young = _signals([{"player_id": 1, "game_date": "2026-05-17",
                              "is_b2b": True, "recovery_coefficient": 1.0}])
        df_old   = _signals([{"player_id": 2, "game_date": "2026-05-17",
                              "is_b2b": True, "recovery_coefficient": 0.6}])
        y = CompositeFatigueIndex().compute(df_young, "2026-05-17").to_dicts()[0]
        o = CompositeFatigueIndex().compute(df_old,   "2026-05-17").to_dicts()[0]
        assert o["fatigue_index"] > y["fatigue_index"]

    def test_concussion_multiplier_amplifies(self):
        df_norm = _signals([{"player_id": 1, "game_date": "2026-05-17",
                             "is_b2b": True}])
        df_conc = _signals([{"player_id": 2, "game_date": "2026-05-17",
                             "is_b2b": True,
                             "fatigue_sensitivity_multiplier": 1.20}])
        a = CompositeFatigueIndex().compute(df_norm, "2026-05-17").to_dicts()[0]
        b = CompositeFatigueIndex().compute(df_conc, "2026-05-17").to_dicts()[0]
        assert b["fatigue_index"] == pytest.approx(a["fatigue_index"] * 1.20,
                                                   abs=1e-9)

    def test_fi_in_unit_interval(self):
        df = _signals([
            {"player_id": 1, "game_date": "2026-05-17"},
            {"player_id": 2, "game_date": "2026-05-17",
             "is_b2b": True, "is_3_in_4": True, "rust_factor": 0.5,
             "playoff_load_penalty": 0.3, "toi_spike_z": 5.0,
             "miles_last_7d": 5000.0, "fatigue_sensitivity_multiplier": 1.25,
             "recovery_coefficient": 0.6, "team_strain_score": 1.0},
            {"player_id": 3, "game_date": "2026-05-17",
             "fight_score": 1.0, "ot_equiv_toi_secs": 420.0},
        ])
        out = CompositeFatigueIndex().compute(df, "2026-05-17")
        for r in out.to_dicts():
            assert 0.0 <= r["fatigue_index"] <= 1.0


# ---------------------------------------------------------------------------
# Component breakdown auditability
# ---------------------------------------------------------------------------

class TestComponentBreakdown:
    def test_breakdown_is_valid_json(self):
        df = _signals([{"player_id": 1, "game_date": "2026-05-17",
                        "is_b2b": True, "miles_last_7d": 1000.0}])
        out = CompositeFatigueIndex().compute(df, "2026-05-17").to_dicts()[0]
        comps = json.loads(out["component_breakdown"])
        assert set(comps) == set(DEFAULT_WEIGHTS)
        # Sum of components equals raw_load.
        assert sum(comps.values()) == pytest.approx(out["raw_load"], abs=1e-6)

    def test_breakdown_identifies_dominant_signal(self):
        df = _signals([{"player_id": 1, "game_date": "2026-05-17",
                        "toi_spike_z": SCALE_TOI_SPIKE_Z}])
        out = CompositeFatigueIndex().compute(df, "2026-05-17").to_dicts()[0]
        comps = json.loads(out["component_breakdown"])
        # The only non-zero contribution should be toi_load.
        assert comps["toi_load"] == pytest.approx(DEFAULT_WEIGHTS["toi_load"],
                                                  abs=1e-6)
        assert sum(comps.values()) == pytest.approx(DEFAULT_WEIGHTS["toi_load"],
                                                    abs=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_negative_signals_treated_as_zero(self):
        df = _signals([{"player_id": 1, "game_date": "2026-05-17",
                        "miles_last_7d": -100.0,
                        "toi_spike_z": -5.0,
                        "rust_factor": 1.0}])
        out = CompositeFatigueIndex().compute(df, "2026-05-17").to_dicts()[0]
        assert out["fatigue_index"] == pytest.approx(0.0)

    def test_missing_optional_columns_are_no_op(self):
        # Build a minimal-column input frame.
        minimal = pl.DataFrame(
            [{"player_id": 1, "game_date": "2026-05-17"}],
            schema={"player_id": pl.Int64, "game_date": pl.Utf8},
        )
        out = CompositeFatigueIndex().compute(minimal, "2026-05-17").to_dicts()[0]
        assert out["fatigue_index"] == pytest.approx(0.0)

    def test_invalid_game_date_row_dropped(self):
        df = pl.DataFrame(
            [
                {"player_id": 1, "game_date": "nope"},
                {"player_id": 2, "game_date": "2026-05-17"},
            ],
            schema={"player_id": pl.Int64, "game_date": pl.Utf8},
        )
        out = CompositeFatigueIndex().compute(df, "2026-05-17")
        assert set(out["player_id"].to_list()) == {2}

    def test_rust_factor_below_zero_clamped(self):
        df = _signals([{"player_id": 1, "game_date": "2026-05-17",
                        "rust_factor": -0.5}])
        out = CompositeFatigueIndex().compute(df, "2026-05-17").to_dicts()[0]
        # rust_factor clamped to 0 → rust_load = 1.0 → FI = 1.0.
        assert out["rust_load"] == pytest.approx(1.0)
        assert out["fatigue_index"] == pytest.approx(1.0)

    def test_recovery_floor_protects_division(self):
        df = _signals([{"player_id": 1, "game_date": "2026-05-17",
                        "is_b2b": True, "recovery_coefficient": 0.0}])
        out = CompositeFatigueIndex().compute(df, "2026-05-17").to_dicts()[0]
        # Even with recovery_coefficient = 0 the floor saves us.
        assert out["fatigue_index"] <= 1.0
        assert out["recovery_coefficient"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Custom weights
# ---------------------------------------------------------------------------

class TestCustomWeights:
    def test_partial_override_keeps_defaults(self):
        model = CompositeFatigueIndex(weights={"toi_load": 0.50})
        assert model.weights["toi_load"] == pytest.approx(0.50)
        for k, default_v in DEFAULT_WEIGHTS.items():
            if k == "toi_load":
                continue
            assert model.weights[k] == pytest.approx(default_v)

    def test_zero_weight_kills_signal(self):
        zero_toi = CompositeFatigueIndex(weights={"toi_load": 0.0})
        df = _signals([{"player_id": 1, "game_date": "2026-05-17",
                        "toi_spike_z": 10.0}])
        out = zero_toi.compute(df, "2026-05-17").to_dicts()[0]
        assert out["raw_load"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_parquet(self, tmp_path):
        df = _signals([{"player_id": 1, "game_date": "2026-05-17",
                        "is_b2b": True}])
        out = CompositeFatigueIndex().compute(df, "2026-05-17")
        path = write_composite_fi(out, tmp_path, "2026-05-17")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in COMPOSITE_FI_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        out = CompositeFatigueIndex().compute(_signals([]), "2026-05-17")
        path = write_composite_fi(out, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name
