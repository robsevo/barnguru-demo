"""Tests for Feature 3.21 — FI → EDGE metric degradation model."""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl
import pytest

from models.fi_edge_degradation import (
    DEFAULT_COEFFS,
    FI_EDGE_DEGRADATION_SCHEMA,
    MAX_DELTA,
    METRICS,
    MIN_DELTA,
    MIN_OBS,
    FIEdgeDegradationModel,
    write_fi_edge_degradation,
)
from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Synthetic training fixtures
# ---------------------------------------------------------------------------

def _training_rows(n: int, slope: float = -0.20, alpha: float = 0.01,
                   noise: float = 0.0) -> pl.DataFrame:
    """Build a dataset that bakes in a known slope for every metric."""
    fi_vals = [(i / max(1, n - 1)) for i in range(n)]
    rows = []
    for fi in fi_vals:
        rows.append({
            "fi":                          fi,
            "delta_speed_vs_baseline":     alpha + slope * fi + noise,
            "delta_distance_vs_baseline":  alpha + slope * fi + noise,
            "delta_carry_vs_baseline":     alpha + slope * fi + noise,
            "delta_burst_vs_baseline":     alpha + slope * fi + noise,
        })
    return pl.DataFrame(rows)


def _fi_obs(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "player_id": pl.Int64,
        "game_id":   pl.Int64,
        "game_date": pl.Utf8,
        "fi":        pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    defaults = {"game_id": 0}
    return pl.DataFrame([{**defaults, **r} for r in rows], schema=schema)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_metrics_match_default_coeffs(self):
        assert set(METRICS) == set(DEFAULT_COEFFS.keys())

    def test_default_slopes_are_negative(self):
        # Tired players slow down — that's the falsifiable check.
        for _alpha, beta in DEFAULT_COEFFS.values():
            assert beta < 0.0

    def test_clamp_bounds_make_sense(self):
        assert MIN_DELTA < 0.0
        assert MAX_DELTA > MIN_DELTA


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construct(self):
        m = FIEdgeDegradationModel()
        assert m.coeffs == DEFAULT_COEFFS
        assert m.is_fitted is False

    def test_bad_min_obs_raises(self):
        with pytest.raises(ValueError):
            FIEdgeDegradationModel(min_obs=1)

    def test_bad_clamp_raises(self):
        with pytest.raises(ValueError):
            FIEdgeDegradationModel(min_delta=0.1, max_delta=0.0)


# ---------------------------------------------------------------------------
# Fit behavior
# ---------------------------------------------------------------------------

class TestFit:
    def test_fits_known_slope(self):
        df = _training_rows(n=200, slope=-0.25, alpha=0.02)
        m = FIEdgeDegradationModel().fit(df)
        assert m.is_fitted is True
        for metric in METRICS:
            alpha, beta = m.coeffs[metric]
            assert alpha == pytest.approx(0.02, abs=1e-6)
            assert beta == pytest.approx(-0.25, abs=1e-6)

    def test_too_few_rows_falls_back_to_defaults(self):
        df = _training_rows(n=10)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = FIEdgeDegradationModel().fit(df)
        assert m.is_fitted is False
        assert m.coeffs == DEFAULT_COEFFS
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_missing_columns_falls_back(self):
        df = pl.DataFrame({"fi": [0.1, 0.2, 0.3]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = FIEdgeDegradationModel().fit(df)
        assert m.coeffs == DEFAULT_COEFFS
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_zero_variance_fi_handled(self):
        # All FI = 0.5 — no variance in x. Should not crash, intercept = mean.
        rows = [{
            "fi": 0.5,
            "delta_speed_vs_baseline":    -0.1,
            "delta_distance_vs_baseline": -0.1,
            "delta_carry_vs_baseline":    -0.1,
            "delta_burst_vs_baseline":    -0.1,
        }] * MIN_OBS
        df = pl.DataFrame(rows)
        m = FIEdgeDegradationModel().fit(df)
        for metric in METRICS:
            alpha, beta = m.coeffs[metric]
            assert beta == pytest.approx(0.0)
            assert alpha == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# Prediction behavior
# ---------------------------------------------------------------------------

class TestPredictOne:
    def test_predict_at_zero_fi_returns_intercepts(self):
        m = FIEdgeDegradationModel()
        out = m.predict_one(0.0)
        for metric, value in out.items():
            alpha, _beta = DEFAULT_COEFFS[metric]
            assert value == pytest.approx(alpha)

    def test_predict_clamps_at_lower_bound(self):
        # Crank slope way past the clamp.
        m = FIEdgeDegradationModel()
        m.coeffs = {metric: (0.0, -10.0) for metric in METRICS}
        out = m.predict_one(1.0)
        for v in out.values():
            assert v >= MIN_DELTA - 1e-9
            assert v == pytest.approx(MIN_DELTA)

    def test_predict_clamps_at_upper_bound(self):
        m = FIEdgeDegradationModel()
        m.coeffs = {metric: (10.0, 0.0) for metric in METRICS}
        out = m.predict_one(0.0)
        for v in out.values():
            assert v <= MAX_DELTA + 1e-9
            assert v == pytest.approx(MAX_DELTA)

    def test_negative_fi_treated_as_zero(self):
        m = FIEdgeDegradationModel()
        out_neg = m.predict_one(-0.5)
        out_zero = m.predict_one(0.0)
        for metric in METRICS:
            assert out_neg[metric] == pytest.approx(out_zero[metric])

    def test_fi_above_one_treated_as_one(self):
        m = FIEdgeDegradationModel()
        out_high = m.predict_one(2.0)
        out_one  = m.predict_one(1.0)
        for metric in METRICS:
            assert out_high[metric] == pytest.approx(out_one[metric])

    def test_higher_fi_yields_lower_speed_under_default(self):
        m = FIEdgeDegradationModel()
        low  = m.predict_one(0.1)["speed_vs_baseline"]
        high = m.predict_one(0.9)["speed_vs_baseline"]
        assert high < low


# ---------------------------------------------------------------------------
# Predict batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_predict_schema(self):
        df = _fi_obs([
            {"player_id": 1, "game_date": "2026-01-15", "fi": 0.3},
            {"player_id": 2, "game_date": "2026-01-15", "fi": 0.7},
        ])
        out = FIEdgeDegradationModel().predict(df, "2026-05-17")
        assert set(out.columns) == set(FI_EDGE_DEGRADATION_SCHEMA.keys())
        assert len(out) == 2

    def test_empty_input_returns_empty(self):
        out = FIEdgeDegradationModel().predict(_fi_obs([]), "2026-05-17")
        assert len(out) == 0
        for c in FI_EDGE_DEGRADATION_SCHEMA:
            assert c in out.columns

    def test_missing_required_columns_warns(self):
        bad = pl.DataFrame({"foo": [1, 2]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = FIEdgeDegradationModel().predict(bad, "2026-05-17")
        assert len(out) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_bad_as_of_raises(self):
        df = _fi_obs([{"player_id": 1, "game_date": "2026-01-15", "fi": 0.3}])
        with pytest.raises(ValueError):
            FIEdgeDegradationModel().predict(df, "not-a-date")

    def test_load_factor_within_clamp(self):
        df = _fi_obs([
            {"player_id": i, "game_date": "2026-01-15", "fi": i / 10.0}
            for i in range(11)
        ])
        out = FIEdgeDegradationModel().predict(df, "2026-05-17").to_dicts()
        for r in out:
            assert MIN_DELTA - 1e-9 <= r["predicted_load_factor"] <= MAX_DELTA + 1e-9

    def test_per_metric_deltas_in_clamp(self):
        df = _fi_obs([{"player_id": 1, "game_date": "2026-01-15", "fi": 0.5}])
        out = FIEdgeDegradationModel().predict(df, "2026-05-17").to_dicts()[0]
        for metric in METRICS:
            assert MIN_DELTA - 1e-9 <= out[metric] <= MAX_DELTA + 1e-9


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_round_trip(self, tmp_path):
        df = _training_rows(n=200, slope=-0.30, alpha=0.05)
        m = FIEdgeDegradationModel().fit(df)
        path = tmp_path / "model.pkl"
        m.save(path)
        m2 = FIEdgeDegradationModel.load(path)
        assert m2.is_fitted is True
        for metric in METRICS:
            assert m2.coeffs[metric] == pytest.approx(m.coeffs[metric])

    def test_parquet_write(self, tmp_path):
        df = _fi_obs([{"player_id": 1, "game_date": "2026-01-15", "fi": 0.4}])
        out = FIEdgeDegradationModel().predict(df, "2026-05-17")
        path = write_fi_edge_degradation(out, tmp_path, "2026-05-17")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in FI_EDGE_DEGRADATION_SCHEMA:
            assert col in loaded.columns
        assert "2026-05-17" in path.name


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_non_finite_fi_dropped(self):
        df = pl.DataFrame(
            [
                {"player_id": 1, "game_id": 1, "game_date": "2026-01-15",
                 "fi": float("nan")},
                {"player_id": 1, "game_id": 2, "game_date": "2026-01-16",
                 "fi": 0.5},
            ],
            schema={
                "player_id": pl.Int64, "game_id": pl.Int64,
                "game_date": pl.Utf8,  "fi": pl.Float64,
            },
        )
        out = FIEdgeDegradationModel().predict(df, "2026-05-17")
        assert out["game_id"].to_list() == [2]

    def test_invalid_date_dropped(self):
        df = pl.DataFrame(
            [
                {"player_id": 1, "game_id": 1, "game_date": "nope",
                 "fi": 0.3},
                {"player_id": 1, "game_id": 2, "game_date": "2026-01-16",
                 "fi": 0.3},
            ],
            schema={
                "player_id": pl.Int64, "game_id": pl.Int64,
                "game_date": pl.Utf8,  "fi": pl.Float64,
            },
        )
        out = FIEdgeDegradationModel().predict(df, "2026-05-17")
        assert out["game_id"].to_list() == [2]
