"""Tests for Feature 3.22 — Seasonal Performance Factor."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from models.rapm_model import DataMissingWarning
from models.seasonal_performance import (
    AGE_PIVOT,
    DEFAULT_ALPHAS,
    DEFAULT_BASE_MONTH,
    DOG_DAYS_MONTHS,
    GP_PIVOT,
    MAX_FACTOR,
    MIN_FACTOR,
    PLAYOFF_PUSH_MONTHS,
    SEASONAL_PERFORMANCE_SCHEMA,
    SeasonalPerformanceFactor,
    write_seasonal_performance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _inputs(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "player_id":         pl.Int64,
        "game_id":           pl.Int64,
        "game_date":         pl.Utf8,
        "playoff_prob":      pl.Float64,
        "age":               pl.Float64,
        "games_played_ytd":  pl.Int64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    defaults = {
        "game_id":          0,
        "playoff_prob":     0.5,
        "age":              float(AGE_PIVOT),
        "games_played_ytd": GP_PIVOT,
    }
    return pl.DataFrame([{**defaults, **r} for r in rows], schema=schema)


# ---------------------------------------------------------------------------
# Constants — sanity that the spec ranges hold
# ---------------------------------------------------------------------------

class TestConstants:
    def test_factor_range_matches_spec(self):
        # Spec: [-0.05, +0.03]
        assert MIN_FACTOR == pytest.approx(-0.05)
        assert MAX_FACTOR == pytest.approx(+0.03)

    def test_january_base_in_dog_days_range(self):
        # Spec: Jan −3% to −5%
        assert -0.05 <= DEFAULT_BASE_MONTH[1] <= -0.03

    def test_april_base_in_push_range(self):
        # Spec: Apr +2% to +3%
        assert 0.02 <= DEFAULT_BASE_MONTH[4] <= 0.03

    def test_october_base_in_rust_range(self):
        # Spec: Oct −2% to −3%
        assert -0.03 <= DEFAULT_BASE_MONTH[10] <= -0.02

    def test_off_season_zero_or_near(self):
        assert DEFAULT_BASE_MONTH[7] == pytest.approx(0.0)
        assert DEFAULT_BASE_MONTH[8] == pytest.approx(0.0)

    def test_all_alphas_present(self):
        assert set(DEFAULT_ALPHAS.keys()) >= {"playoff", "age", "gp"}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construct(self):
        m = SeasonalPerformanceFactor()
        assert m.base_month == DEFAULT_BASE_MONTH
        assert m.alphas == DEFAULT_ALPHAS

    def test_rejects_bad_factor_range(self):
        with pytest.raises(ValueError):
            SeasonalPerformanceFactor(min_factor=0.1, max_factor=0.0)

    def test_rejects_bad_pivots(self):
        with pytest.raises(ValueError):
            SeasonalPerformanceFactor(age_pivot=0)
        with pytest.raises(ValueError):
            SeasonalPerformanceFactor(gp_pivot=0)

    def test_rejects_bad_month_key(self):
        with pytest.raises(ValueError):
            SeasonalPerformanceFactor(base_month={13: 0.01})
        with pytest.raises(ValueError):
            SeasonalPerformanceFactor(base_month={0: 0.01})


# ---------------------------------------------------------------------------
# compute_row — pure
# ---------------------------------------------------------------------------

class TestComputeRow:
    def test_july_off_season_returns_zero(self):
        m = SeasonalPerformanceFactor()
        out = m.compute_row(month=7, playoff_prob=0.9, age=30, games_played=20)
        assert out["base_month_effect"] == pytest.approx(0.0)
        # Off-season has no dog-days or push gates active.
        assert out["playoff_adj"] == pytest.approx(0.0)
        assert out["age_adj"] == pytest.approx(0.0)
        assert out["gp_adj"] == pytest.approx(0.0)
        assert out["seasonal_motivation_factor"] == pytest.approx(0.0)

    def test_january_pivot_player_is_base_only(self):
        m = SeasonalPerformanceFactor()
        out = m.compute_row(month=1, playoff_prob=0.5,
                            age=AGE_PIVOT, games_played=GP_PIVOT)
        assert out["base_month_effect"] == DEFAULT_BASE_MONTH[1]
        assert out["seasonal_motivation_factor"] == pytest.approx(
            DEFAULT_BASE_MONTH[1]
        )

    def test_april_high_playoff_prob_boost(self):
        m = SeasonalPerformanceFactor()
        no_push = m.compute_row(4, playoff_prob=0.5,
                                age=AGE_PIVOT, games_played=GP_PIVOT)
        push    = m.compute_row(4, playoff_prob=1.0,
                                age=AGE_PIVOT, games_played=GP_PIVOT)
        assert push["seasonal_motivation_factor"] > no_push["seasonal_motivation_factor"]

    def test_january_older_player_extra_drag(self):
        m = SeasonalPerformanceFactor()
        young = m.compute_row(1, playoff_prob=0.5, age=22, games_played=GP_PIVOT)
        old   = m.compute_row(1, playoff_prob=0.5, age=38, games_played=GP_PIVOT)
        # Older players get more drag (negative direction is "more").
        assert old["seasonal_motivation_factor"] < young["seasonal_motivation_factor"]

    def test_january_high_gp_extra_drag(self):
        m = SeasonalPerformanceFactor()
        fresh  = m.compute_row(1, playoff_prob=0.5, age=AGE_PIVOT, games_played=20)
        slogged = m.compute_row(1, playoff_prob=0.5, age=AGE_PIVOT, games_played=70)
        assert slogged["seasonal_motivation_factor"] < fresh["seasonal_motivation_factor"]

    def test_april_age_does_not_apply(self):
        # In April the age_adj gate is closed (only fires in dog days).
        m = SeasonalPerformanceFactor()
        young = m.compute_row(4, playoff_prob=0.5, age=22, games_played=GP_PIVOT)
        old   = m.compute_row(4, playoff_prob=0.5, age=38, games_played=GP_PIVOT)
        assert young["age_adj"] == pytest.approx(0.0)
        assert old["age_adj"] == pytest.approx(0.0)

    def test_clamp_holds_under_extreme_inputs(self):
        m = SeasonalPerformanceFactor()
        # Force extreme negative
        out = m.compute_row(1, playoff_prob=0.0, age=99, games_played=200)
        assert out["seasonal_motivation_factor"] >= MIN_FACTOR - 1e-9
        # Force extreme positive
        out = m.compute_row(4, playoff_prob=1.0, age=AGE_PIVOT, games_played=GP_PIVOT)
        assert out["seasonal_motivation_factor"] <= MAX_FACTOR + 1e-9


# ---------------------------------------------------------------------------
# predict batch
# ---------------------------------------------------------------------------

class TestPredict:
    def test_schema(self):
        df = _inputs([
            {"player_id": 1, "game_date": "2026-01-15"},
            {"player_id": 2, "game_date": "2026-04-10"},
        ])
        out = SeasonalPerformanceFactor().predict(df, "2026-05-17")
        assert set(out.columns) == set(SEASONAL_PERFORMANCE_SCHEMA.keys())
        assert len(out) == 2

    def test_month_extracted_from_game_date(self):
        df = _inputs([
            {"player_id": 1, "game_date": "2026-01-15"},
            {"player_id": 1, "game_date": "2026-04-15"},
        ])
        out = SeasonalPerformanceFactor().predict(df, "2026-05-17").to_dicts()
        assert out[0]["month_of_season"] == 1
        assert out[1]["month_of_season"] == 4

    def test_january_negative_april_positive(self):
        df = _inputs([
            {"player_id": 1, "game_date": "2026-01-15"},
            {"player_id": 1, "game_date": "2026-04-15"},
        ])
        out = SeasonalPerformanceFactor().predict(df, "2026-05-17").to_dicts()
        assert out[0]["seasonal_motivation_factor"] < 0
        assert out[1]["seasonal_motivation_factor"] > 0

    def test_factor_within_documented_range(self):
        rows = []
        for m in range(1, 13):
            rows.append({"player_id": m,
                         "game_date": f"2026-{m:02d}-15"})
        out = SeasonalPerformanceFactor().predict(_inputs(rows), "2026-05-17")
        for v in out["seasonal_motivation_factor"].to_list():
            assert MIN_FACTOR - 1e-9 <= v <= MAX_FACTOR + 1e-9

    def test_missing_optional_columns_use_defaults(self):
        # Build inputs DF without playoff_prob/age/gp.
        df = pl.DataFrame(
            [{"player_id": 1, "game_id": 1, "game_date": "2026-01-15"}],
            schema={
                "player_id": pl.Int64, "game_id": pl.Int64,
                "game_date": pl.Utf8,
            },
        )
        out = SeasonalPerformanceFactor().predict(df, "2026-05-17").to_dicts()[0]
        # Defaults should land at pivot values.
        assert out["playoff_prob"] == pytest.approx(0.5)
        assert out["age"] == pytest.approx(float(AGE_PIVOT))
        assert out["games_played_ytd"] == GP_PIVOT

    def test_missing_required_warns(self):
        bad = pl.DataFrame({"foo": [1]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = SeasonalPerformanceFactor().predict(bad, "2026-05-17")
        assert len(out) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_input(self):
        out = SeasonalPerformanceFactor().predict(_inputs([]), "2026-05-17")
        assert len(out) == 0
        for c in SEASONAL_PERFORMANCE_SCHEMA:
            assert c in out.columns

    def test_bad_as_of_raises(self):
        df = _inputs([{"player_id": 1, "game_date": "2026-01-15"}])
        with pytest.raises(ValueError):
            SeasonalPerformanceFactor().predict(df, "not-a-date")

    def test_invalid_game_date_dropped(self):
        df = pl.DataFrame(
            [
                {"player_id": 1, "game_id": 1, "game_date": "nope",
                 "playoff_prob": 0.5, "age": 28.0, "games_played_ytd": 30},
                {"player_id": 1, "game_id": 2, "game_date": "2026-01-15",
                 "playoff_prob": 0.5, "age": 28.0, "games_played_ytd": 30},
            ],
            schema={
                "player_id": pl.Int64, "game_id": pl.Int64,
                "game_date": pl.Utf8,  "playoff_prob": pl.Float64,
                "age": pl.Float64, "games_played_ytd": pl.Int64,
            },
        )
        out = SeasonalPerformanceFactor().predict(df, "2026-05-17")
        assert out["game_id"].to_list() == [2]

    def test_playoff_prob_clamped_to_unit(self):
        df = _inputs([
            {"player_id": 1, "game_date": "2026-04-15", "playoff_prob": 5.0},
            {"player_id": 2, "game_date": "2026-04-15", "playoff_prob": -3.0},
        ])
        out = SeasonalPerformanceFactor().predict(df, "2026-05-17").to_dicts()
        assert out[0]["playoff_prob"] == pytest.approx(1.0)
        assert out[1]["playoff_prob"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Fit behavior
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_updates_per_month_mean(self):
        train = pl.DataFrame({
            "month_of_season": [1, 1, 1, 4, 4, 4],
            "residual":        [-0.04, -0.05, -0.03, 0.02, 0.03, 0.025],
        })
        m = SeasonalPerformanceFactor().fit(train)
        assert m.base_month[1] == pytest.approx((-0.04 + -0.05 + -0.03) / 3)
        assert m.base_month[4] == pytest.approx((0.02 + 0.03 + 0.025) / 3)

    def test_fit_clamps_extreme_means(self):
        train = pl.DataFrame({
            "month_of_season": [1, 1],
            "residual":        [-0.20, -0.30],   # absurd
        })
        m = SeasonalPerformanceFactor().fit(train)
        assert m.base_month[1] == pytest.approx(MIN_FACTOR)

    def test_fit_keeps_unobserved_months(self):
        train = pl.DataFrame({"month_of_season": [4], "residual": [0.01]})
        m = SeasonalPerformanceFactor().fit(train)
        assert m.base_month[1] == DEFAULT_BASE_MONTH[1]
        assert m.base_month[10] == DEFAULT_BASE_MONTH[10]

    def test_fit_missing_columns_warns(self):
        bad = pl.DataFrame({"foo": [1]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = SeasonalPerformanceFactor().fit(bad)
        assert m.base_month == DEFAULT_BASE_MONTH
        assert any(issubclass(x.category, DataMissingWarning) for x in w)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_round_trip(self, tmp_path):
        m = SeasonalPerformanceFactor()
        path = tmp_path / "model.pkl"
        m.save(path)
        m2 = SeasonalPerformanceFactor.load(path)
        assert m2.base_month == m.base_month
        assert m2.alphas == m.alphas

    def test_parquet_write(self, tmp_path):
        df = _inputs([{"player_id": 1, "game_date": "2026-01-15"}])
        out = SeasonalPerformanceFactor().predict(df, "2026-05-17")
        path = write_seasonal_performance(out, tmp_path, "2026-05-17")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for c in SEASONAL_PERFORMANCE_SCHEMA:
            assert c in loaded.columns
        assert "2026-05-17" in path.name
