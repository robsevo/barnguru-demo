"""Tests for Feature 3.12 — AgeRecoveryCalculator."""

from __future__ import annotations

import math
import warnings

import polars as pl
import pytest

from models.age_recovery import (
    AGE_RECOVERY_SCHEMA,
    RECOVERY_DECAY_K,
    RECOVERY_FLOOR,
    RECOVERY_PEAK_AGE,
    AgeRecoveryCalculator,
    recovery_coefficient,
    write_age_recovery,
)
from models.rapm_model import DataMissingWarning


def _roster_df(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "player_id":  pl.Int64,
                "birth_date": pl.Utf8,
                "age":        pl.Float64,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "player_id":  pl.Int64,
            "birth_date": pl.Utf8,
            "age":        pl.Float64,
        },
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_required_columns_warns(self):
        bad = pl.DataFrame({"foo": ["bar"]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = AgeRecoveryCalculator().compute(bad, as_of_date="2026-05-17")
        assert len(result) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_no_age_or_birth_warns(self):
        bad = pl.DataFrame({"player_id": [1, 2]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = AgeRecoveryCalculator().compute(bad, as_of_date="2026-05-17")
        assert len(result) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_returns_empty_schema(self):
        result = AgeRecoveryCalculator().compute(_roster_df([]), as_of_date="2026-05-17")
        for col in AGE_RECOVERY_SCHEMA:
            assert col in result.columns
        assert len(result) == 0

    def test_output_schema(self):
        df = _roster_df([{"player_id": 1, "birth_date": "1995-01-01", "age": None}])
        result = AgeRecoveryCalculator().compute(df, as_of_date="2026-05-17")
        assert set(result.columns) == set(AGE_RECOVERY_SCHEMA.keys())

    def test_bad_params_rejected(self):
        with pytest.raises(ValueError):
            AgeRecoveryCalculator(peak_age=0.0)
        with pytest.raises(ValueError):
            AgeRecoveryCalculator(decay_k=-0.1)
        with pytest.raises(ValueError):
            AgeRecoveryCalculator(recovery_floor=-0.1)
        with pytest.raises(ValueError):
            AgeRecoveryCalculator(recovery_floor=1.5)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_peak_age_is_thirty(self):
        # Spec: "Exponential recovery decay post-30"
        assert RECOVERY_PEAK_AGE == 30.0

    def test_decay_k_positive(self):
        assert RECOVERY_DECAY_K > 0

    def test_floor_in_unit_interval(self):
        assert 0.0 <= RECOVERY_FLOOR <= 1.0


# ---------------------------------------------------------------------------
# Functional form
# ---------------------------------------------------------------------------

class TestRecoveryFunction:
    def test_below_peak_full_recovery(self):
        for age in (18, 22, 25, 28, 29.5):
            assert recovery_coefficient(age) == pytest.approx(1.0)

    def test_at_peak_full_recovery(self):
        assert recovery_coefficient(RECOVERY_PEAK_AGE) == pytest.approx(1.0)

    def test_monotone_decreasing_above_peak(self):
        prev = recovery_coefficient(RECOVERY_PEAK_AGE)
        for age in (31, 33, 35, 37, 39):
            curr = recovery_coefficient(age)
            assert curr < prev or curr == RECOVERY_FLOOR
            prev = curr

    def test_floor_clamps_extreme_age(self):
        assert recovery_coefficient(80.0) == pytest.approx(RECOVERY_FLOOR)
        assert recovery_coefficient(120.0) == pytest.approx(RECOVERY_FLOOR)

    def test_exponential_form_default(self):
        # At default decay_k=0.05, 35 → exp(-0.25) ≈ 0.7788
        expected = math.exp(-RECOVERY_DECAY_K * 5.0)
        assert recovery_coefficient(35.0) == pytest.approx(expected)

    def test_nonfinite_age_returns_floor(self):
        assert recovery_coefficient(float("nan")) == pytest.approx(RECOVERY_FLOOR)
        assert recovery_coefficient(float("inf")) == pytest.approx(RECOVERY_FLOOR)

    def test_custom_decay(self):
        # Faster decay → lower coefficient at the same age.
        slow = recovery_coefficient(35.0, decay_k=0.03)
        fast = recovery_coefficient(35.0, decay_k=0.10)
        assert fast < slow


# ---------------------------------------------------------------------------
# DataFrame compute path
# ---------------------------------------------------------------------------

class TestCompute:
    def test_birth_date_drives_age(self):
        # Born 1995-05-17, as-of 2026-05-17 → exactly 31.0
        df = _roster_df([{"player_id": 7, "birth_date": "1995-05-17", "age": None}])
        out = AgeRecoveryCalculator().compute(df, as_of_date="2026-05-17").to_dicts()[0]
        assert out["age_years"] == pytest.approx(31.0, abs=0.01)
        assert out["recovery_coefficient"] == pytest.approx(
            math.exp(-RECOVERY_DECAY_K * 1.0), abs=1e-3
        )

    def test_age_fallback_when_no_birth_date(self):
        df = _roster_df([{"player_id": 3, "birth_date": None, "age": 35.0}])
        out = AgeRecoveryCalculator().compute(df, as_of_date="2026-05-17").to_dicts()[0]
        assert out["age_years"] == pytest.approx(35.0)
        assert out["recovery_coefficient"] == pytest.approx(
            math.exp(-RECOVERY_DECAY_K * 5.0), abs=1e-6
        )

    def test_birth_date_overrides_age_field(self):
        # Conflicting inputs → birth_date wins (deterministic, exact).
        df = _roster_df([{"player_id": 9, "birth_date": "2000-05-17", "age": 80.0}])
        out = AgeRecoveryCalculator().compute(df, as_of_date="2026-05-17").to_dicts()[0]
        assert out["age_years"] == pytest.approx(26.0, abs=0.01)
        assert out["recovery_coefficient"] == pytest.approx(1.0)

    def test_player_with_no_age_data_is_skipped(self):
        df = _roster_df([
            {"player_id": 1, "birth_date": None, "age": None},
            {"player_id": 2, "birth_date": "1995-01-01", "age": None},
        ])
        out = AgeRecoveryCalculator().compute(df, as_of_date="2026-05-17")
        pids = set(out["player_id"].to_list())
        assert pids == {2}

    def test_duplicate_player_id_deduped(self):
        df = _roster_df([
            {"player_id": 5, "birth_date": "1995-05-17", "age": None},
            {"player_id": 5, "birth_date": "1995-05-17", "age": None},
        ])
        out = AgeRecoveryCalculator().compute(df, as_of_date="2026-05-17")
        assert len(out) == 1

    def test_unparseable_birth_date_falls_back_to_age(self):
        df = _roster_df([{"player_id": 1, "birth_date": "not-a-date", "age": 32.0}])
        out = AgeRecoveryCalculator().compute(df, as_of_date="2026-05-17").to_dicts()[0]
        assert out["age_years"] == pytest.approx(32.0)

    def test_future_birth_date_falls_back(self):
        df = _roster_df([{"player_id": 1, "birth_date": "2050-01-01", "age": 28.0}])
        out = AgeRecoveryCalculator().compute(df, as_of_date="2026-05-17").to_dicts()[0]
        assert out["age_years"] == pytest.approx(28.0)

    def test_output_in_unit_interval(self):
        rows = [{"player_id": i, "birth_date": None, "age": float(a)}
                for i, a in enumerate([20, 25, 30, 33, 36, 40, 45], start=1)]
        out = AgeRecoveryCalculator().compute(_roster_df(rows), as_of_date="2026-05-17")
        for coef in out["recovery_coefficient"].to_list():
            assert RECOVERY_FLOOR <= coef <= 1.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_parquet(self, tmp_path):
        df = _roster_df([{"player_id": 1, "birth_date": "1990-01-01", "age": None}])
        out = AgeRecoveryCalculator().compute(df, as_of_date="2026-05-17")
        path = write_age_recovery(out, tmp_path, "2026-05-17")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in AGE_RECOVERY_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        df = _roster_df([{"player_id": 1, "birth_date": "1990-01-01", "age": None}])
        out = AgeRecoveryCalculator().compute(df, as_of_date="2026-05-17")
        path = write_age_recovery(out, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name
