"""Tests for Feature 3.18 — FIRatingMultiplier."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from models.fi_rating_multiplier import (
    FI_MULTIPLIER_SCHEMA,
    MAX_DEGRADATION_DEFAULT,
    FIRatingMultiplier,
    rating_multiplier,
    write_fi_multiplier,
)
from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fi_df(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "player_id":     pl.Int64,
        "game_id":       pl.Int64,
        "game_date":     pl.Utf8,
        "fatigue_index": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    defaults = {"game_id": 1}
    return pl.DataFrame([{**defaults, **r} for r in rows], schema=schema)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_columns_warns(self):
        bad = pl.DataFrame({"foo": [1]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = FIRatingMultiplier().compute(bad, "2026-05-17")
        assert len(out) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_input_returns_empty_schema(self):
        out = FIRatingMultiplier().compute(_fi_df([]), "2026-05-17")
        for col in FI_MULTIPLIER_SCHEMA:
            assert col in out.columns
        assert len(out) == 0

    def test_output_schema(self):
        df = _fi_df([{"player_id": 1, "game_date": "2026-05-17",
                      "fatigue_index": 0.5}])
        out = FIRatingMultiplier().compute(df, "2026-05-17")
        assert set(out.columns) == set(FI_MULTIPLIER_SCHEMA.keys())

    def test_bad_as_of_date_raises(self):
        df = _fi_df([{"player_id": 1, "game_date": "2026-05-17",
                      "fatigue_index": 0.5}])
        with pytest.raises(ValueError):
            FIRatingMultiplier().compute(df, "not-a-date")

    def test_bad_params_rejected(self):
        with pytest.raises(ValueError):
            FIRatingMultiplier(max_degradation=-0.1)
        with pytest.raises(ValueError):
            FIRatingMultiplier(max_degradation=1.0)
        with pytest.raises(ValueError):
            FIRatingMultiplier(slope_floor=-0.1)
        with pytest.raises(ValueError):
            FIRatingMultiplier(slope_floor=1.0)


# ---------------------------------------------------------------------------
# Public formula — rating_multiplier
# ---------------------------------------------------------------------------

class TestRatingMultiplier:
    def test_fi_zero_is_one(self):
        assert rating_multiplier(0.0) == pytest.approx(1.0)

    def test_fi_one_hits_floor(self):
        assert rating_multiplier(1.0) == pytest.approx(
            1.0 - MAX_DEGRADATION_DEFAULT
        )

    def test_half_fi_is_half_degradation(self):
        assert rating_multiplier(0.5) == pytest.approx(
            1.0 - 0.5 * MAX_DEGRADATION_DEFAULT
        )

    def test_monotone_non_increasing(self):
        prev = rating_multiplier(0.0)
        for i in range(1, 21):
            curr = rating_multiplier(i / 20.0)
            assert curr <= prev
            prev = curr

    def test_clamped_below_zero(self):
        assert rating_multiplier(-1.0) == pytest.approx(1.0)

    def test_clamped_above_one(self):
        assert rating_multiplier(5.0) == pytest.approx(
            1.0 - MAX_DEGRADATION_DEFAULT
        )

    def test_nan_safe_default(self):
        assert rating_multiplier(float("nan")) == pytest.approx(1.0)

    def test_custom_max_degradation(self):
        # 25% derating: FI=1 → 0.75
        assert rating_multiplier(1.0, max_degradation=0.25) == pytest.approx(0.75)
        assert rating_multiplier(0.5, max_degradation=0.25) == pytest.approx(0.875)

    def test_slope_floor_creates_free_zone(self):
        # Below the floor, multiplier is exactly 1.0.
        assert rating_multiplier(0.10, slope_floor=0.20) == pytest.approx(1.0)
        # At the floor, still 1.0.
        assert rating_multiplier(0.20, slope_floor=0.20) == pytest.approx(1.0)
        # Above the floor, ramp starts.
        assert rating_multiplier(1.00, slope_floor=0.20) == pytest.approx(
            1.0 - MAX_DEGRADATION_DEFAULT
        )

    def test_output_in_valid_range(self):
        for i in range(-5, 25):
            v = rating_multiplier(i / 20.0)
            assert (1.0 - MAX_DEGRADATION_DEFAULT) <= v <= 1.0


# ---------------------------------------------------------------------------
# Class accessor properties
# ---------------------------------------------------------------------------

class TestClassProperties:
    def test_multiplier_floor_consistent(self):
        m = FIRatingMultiplier(max_degradation=0.20)
        assert m.multiplier_floor == pytest.approx(0.80)

    def test_for_fi_matches_module_function(self):
        m = FIRatingMultiplier(max_degradation=0.15, slope_floor=0.10)
        for fi in (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0):
            assert m.for_fi(fi) == pytest.approx(
                rating_multiplier(fi, max_degradation=0.15, slope_floor=0.10)
            )


# ---------------------------------------------------------------------------
# Compute path
# ---------------------------------------------------------------------------

class TestCompute:
    def test_emits_one_row_per_input(self):
        df = _fi_df([
            {"player_id": 1, "game_date": "2026-05-17", "fatigue_index": 0.0},
            {"player_id": 2, "game_date": "2026-05-17", "fatigue_index": 0.5},
            {"player_id": 3, "game_date": "2026-05-17", "fatigue_index": 1.0},
        ])
        out = FIRatingMultiplier().compute(df, "2026-05-17")
        assert len(out) == 3

    def test_multiplier_matches_formula(self):
        df = _fi_df([
            {"player_id": 1, "game_date": "2026-05-17", "fatigue_index": 0.4},
        ])
        out = FIRatingMultiplier().compute(df, "2026-05-17").to_dicts()[0]
        assert out["rating_multiplier"] == pytest.approx(
            rating_multiplier(0.4)
        )

    def test_fatigue_clamped_into_unit_interval(self):
        df = _fi_df([
            {"player_id": 1, "game_date": "2026-05-17", "fatigue_index": -1.0},
            {"player_id": 2, "game_date": "2026-05-17", "fatigue_index": 5.0},
        ])
        out = FIRatingMultiplier().compute(df, "2026-05-17").to_dicts()
        assert out[0]["fatigue_index"] == pytest.approx(0.0)
        assert out[0]["rating_multiplier"] == pytest.approx(1.0)
        assert out[1]["fatigue_index"] == pytest.approx(1.0)
        assert out[1]["rating_multiplier"] == pytest.approx(
            1.0 - MAX_DEGRADATION_DEFAULT
        )

    def test_invalid_game_date_dropped(self):
        df = pl.DataFrame(
            [
                {"player_id": 1, "game_id": 1, "game_date": "nope",
                 "fatigue_index": 0.5},
                {"player_id": 2, "game_id": 2, "game_date": "2026-05-17",
                 "fatigue_index": 0.5},
            ],
            schema={
                "player_id":     pl.Int64,
                "game_id":       pl.Int64,
                "game_date":     pl.Utf8,
                "fatigue_index": pl.Float64,
            },
        )
        out = FIRatingMultiplier().compute(df, "2026-05-17")
        assert out["player_id"].to_list() == [2]

    def test_null_inputs_dropped(self):
        df = pl.DataFrame(
            [
                {"player_id": None, "game_id": 1, "game_date": "2026-05-17",
                 "fatigue_index": 0.5},
                {"player_id": 1, "game_id": 1, "game_date": None,
                 "fatigue_index": 0.5},
                {"player_id": 2, "game_id": 2, "game_date": "2026-05-17",
                 "fatigue_index": None},
                {"player_id": 3, "game_id": 3, "game_date": "2026-05-17",
                 "fatigue_index": 0.5},
            ],
            schema={
                "player_id":     pl.Int64,
                "game_id":       pl.Int64,
                "game_date":     pl.Utf8,
                "fatigue_index": pl.Float64,
            },
        )
        out = FIRatingMultiplier().compute(df, "2026-05-17")
        assert out["player_id"].to_list() == [3]

    def test_multiplier_in_valid_range(self):
        df = _fi_df([
            {"player_id": i, "game_date": "2026-05-17", "fatigue_index": i / 10.0}
            for i in range(11)
        ])
        out = FIRatingMultiplier().compute(df, "2026-05-17")
        for r in out.to_dicts():
            assert (1.0 - MAX_DEGRADATION_DEFAULT) <= r["rating_multiplier"] <= 1.0

    def test_custom_max_degradation_passes_through(self):
        df = _fi_df([{"player_id": 1, "game_date": "2026-05-17",
                      "fatigue_index": 1.0}])
        out = FIRatingMultiplier(max_degradation=0.25).compute(
            df, "2026-05-17"
        ).to_dicts()[0]
        assert out["rating_multiplier"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_parquet(self, tmp_path):
        df = _fi_df([{"player_id": 1, "game_date": "2026-05-17",
                      "fatigue_index": 0.5}])
        out = FIRatingMultiplier().compute(df, "2026-05-17")
        path = write_fi_multiplier(out, tmp_path, "2026-05-17")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in FI_MULTIPLIER_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        out = FIRatingMultiplier().compute(_fi_df([]), "2026-05-17")
        path = write_fi_multiplier(out, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name
