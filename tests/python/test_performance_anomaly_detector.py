"""Tests for Feature 3.19 — PerformanceAnomalyDetector."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from models.performance_anomaly_detector import (
    ANOMALY_Z_THRESHOLD,
    CUSUM_H_SIGMA,
    MIN_BASELINE_GAMES,
    PERFORMANCE_ANOMALY_SCHEMA,
    WINDOW_GAMES,
    PerformanceAnomalyDetector,
    write_performance_anomaly,
)
from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _perf(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "player_id": pl.Int64,
        "game_id":   pl.Int64,
        "game_date": pl.Utf8,
        "metric":    pl.Float64,
        "is_on_ir":  pl.Boolean,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    defaults = {"is_on_ir": False}
    return pl.DataFrame([{**defaults, **r} for r in rows], schema=schema)


def _stable_baseline_rows(
    player_id: int = 1,
    n: int = 10,
    metric: float = 1.0,
    on_ir: bool = False,
) -> list[dict]:
    """N rows with constant metric — produces zero-variance baseline."""
    return [
        {"player_id": player_id, "game_id": i + 1,
         "game_date": f"2026-01-{i + 1:02d}",
         "metric": metric, "is_on_ir": on_ir}
        for i in range(n)
    ]


def _noisy_baseline_rows(
    player_id: int = 1,
    n: int = 10,
    mean: float = 1.0,
    seed_seq: list[float] | None = None,
    on_ir: bool = False,
) -> list[dict]:
    """N rows with a controlled per-game pattern around ``mean``."""
    seq = seed_seq or [-0.1, 0.1, -0.05, 0.05, -0.08, 0.08,
                       -0.12, 0.12, -0.07, 0.07, -0.09, 0.09]
    return [
        {"player_id": player_id, "game_id": i + 1,
         "game_date": f"2026-01-{i + 1:02d}",
         "metric": mean + seq[i % len(seq)], "is_on_ir": on_ir}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_columns_warns(self):
        bad = pl.DataFrame({"foo": [1]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = PerformanceAnomalyDetector().compute(bad, "2026-05-17")
        assert len(out) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_input_returns_empty_schema(self):
        out = PerformanceAnomalyDetector().compute(_perf([]), "2026-05-17")
        for col in PERFORMANCE_ANOMALY_SCHEMA:
            assert col in out.columns
        assert len(out) == 0

    def test_output_schema(self):
        df = _perf(_stable_baseline_rows(n=5))
        out = PerformanceAnomalyDetector().compute(df, "2026-05-17")
        assert set(out.columns) == set(PERFORMANCE_ANOMALY_SCHEMA.keys())

    def test_bad_as_of_date_raises(self):
        df = _perf(_stable_baseline_rows(n=5))
        with pytest.raises(ValueError):
            PerformanceAnomalyDetector().compute(df, "not-a-date")

    def test_bad_params_rejected(self):
        with pytest.raises(ValueError):
            PerformanceAnomalyDetector(window_games=1)
        with pytest.raises(ValueError):
            PerformanceAnomalyDetector(min_baseline_games=1)
        with pytest.raises(ValueError):
            PerformanceAnomalyDetector(min_baseline_games=50, window_games=20)
        with pytest.raises(ValueError):
            PerformanceAnomalyDetector(z_threshold=0.5)
        with pytest.raises(ValueError):
            PerformanceAnomalyDetector(cusum_k_sigma=-0.1)
        with pytest.raises(ValueError):
            PerformanceAnomalyDetector(cusum_h_sigma=0.0)
        with pytest.raises(ValueError):
            PerformanceAnomalyDetector(consecutive_z=1.0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_z_threshold_is_negative(self):
        assert ANOMALY_Z_THRESHOLD < 0

    def test_window_positive(self):
        assert WINDOW_GAMES >= 2

    def test_min_baseline_positive(self):
        assert MIN_BASELINE_GAMES >= 2

    def test_cusum_h_positive(self):
        assert CUSUM_H_SIGMA > 0


# ---------------------------------------------------------------------------
# Compute path — Z-score behaviour
# ---------------------------------------------------------------------------

class TestZScore:
    def test_early_games_have_zero_z(self):
        # First few rows lack enough prior to evaluate.
        df = _perf(_stable_baseline_rows(n=3))
        out = PerformanceAnomalyDetector().compute(df, "2026-05-17").to_dicts()
        for r in out:
            assert r["z_score"] == pytest.approx(0.0)
            assert r["is_z_anomaly"] is False
            assert r["is_anomaly"] is False

    def test_zero_variance_baseline_yields_zero_z(self):
        df = _perf(_stable_baseline_rows(n=8))
        out = PerformanceAnomalyDetector().compute(df, "2026-05-17").to_dicts()
        # All metrics identical → baseline_std = 0 → z forced to 0.
        for r in out:
            assert r["z_score"] == pytest.approx(0.0)

    def test_obvious_negative_outlier_triggers_z_alarm(self):
        rows = _noisy_baseline_rows(n=12, mean=1.0)
        # Drop one massively negative game at game 13.
        rows.append({"player_id": 1, "game_id": 13, "game_date": "2026-01-13",
                     "metric": -5.0, "is_on_ir": False})
        out = PerformanceAnomalyDetector().compute(_perf(rows),
                                                   "2026-05-17").to_dicts()
        last = out[-1]
        assert last["z_score"] <= ANOMALY_Z_THRESHOLD
        assert last["is_z_anomaly"] is True
        assert last["is_anomaly"] is True

    def test_obvious_positive_outlier_does_not_trigger(self):
        # Positive outliers are *good* — we don't alarm them.
        rows = _noisy_baseline_rows(n=12, mean=1.0)
        rows.append({"player_id": 1, "game_id": 13, "game_date": "2026-01-13",
                     "metric": 5.0, "is_on_ir": False})
        out = PerformanceAnomalyDetector().compute(_perf(rows),
                                                   "2026-05-17").to_dicts()
        last = out[-1]
        assert last["is_z_anomaly"] is False
        assert last["is_anomaly"] is False


# ---------------------------------------------------------------------------
# CUSUM behaviour
# ---------------------------------------------------------------------------

class TestCusum:
    def test_cusum_starts_at_zero(self):
        df = _perf(_stable_baseline_rows(n=1))
        out = PerformanceAnomalyDetector().compute(df, "2026-05-17").to_dicts()
        assert out[0]["cusum"] == pytest.approx(0.0)

    def test_sustained_dip_triggers_cusum(self):
        # 12 baseline games at mean 1.0 with ~0.1 noise, then 10 games at 0.6.
        baseline = _noisy_baseline_rows(n=12, mean=1.0)
        for i in range(10):
            baseline.append({
                "player_id": 1, "game_id": 13 + i,
                "game_date": f"2026-02-{i + 1:02d}",
                "metric": 0.6, "is_on_ir": False,
            })
        out = PerformanceAnomalyDetector().compute(_perf(baseline),
                                                   "2026-05-17").to_dicts()
        # Somewhere in the sustained dip the CUSUM alarm should fire.
        assert any(r["is_cusum_anomaly"] for r in out[-10:])
        assert any(r["is_anomaly"] for r in out[-10:])

    def test_cusum_resets_on_normal_play(self):
        # Mild dip then full recovery — cusum should decay back to 0.
        rows = _noisy_baseline_rows(n=12, mean=1.0)
        rows.append({"player_id": 1, "game_id": 13, "game_date": "2026-01-13",
                     "metric": 0.3, "is_on_ir": False})
        # Several strong games afterwards — cusum should drain.
        for i in range(8):
            rows.append({
                "player_id": 1, "game_id": 14 + i,
                "game_date": f"2026-01-{14 + i:02d}",
                "metric": 1.5, "is_on_ir": False,
            })
        out = PerformanceAnomalyDetector().compute(_perf(rows),
                                                   "2026-05-17").to_dicts()
        # Final CUSUM should be much smaller than peak.
        peak = max(r["cusum"] for r in out)
        assert out[-1]["cusum"] < peak
        # After enough recovery, cusum settles back to 0.
        assert out[-1]["cusum"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# IR filter
# ---------------------------------------------------------------------------

class TestIRFilter:
    def test_ir_player_not_flagged(self):
        rows = _noisy_baseline_rows(n=12, mean=1.0)
        rows.append({"player_id": 1, "game_id": 13, "game_date": "2026-01-13",
                     "metric": -5.0, "is_on_ir": True})
        out = PerformanceAnomalyDetector().compute(_perf(rows),
                                                   "2026-05-17").to_dicts()
        last = out[-1]
        # Z-alarm still fires (the raw signal is anomalous), but the
        # human-actionable is_anomaly flag is suppressed.
        assert last["is_z_anomaly"] is True
        assert last["is_anomaly"] is False
        assert last["is_on_ir"] is True

    def test_missing_ir_column_defaults_to_false(self):
        rows = _noisy_baseline_rows(n=12, mean=1.0)
        rows.append({"player_id": 1, "game_id": 13, "game_date": "2026-01-13",
                     "metric": -5.0})
        # Build a DF without is_on_ir.
        df = pl.DataFrame(
            rows,
            schema={
                "player_id": pl.Int64,
                "game_id":   pl.Int64,
                "game_date": pl.Utf8,
                "metric":    pl.Float64,
            },
        )
        out = PerformanceAnomalyDetector().compute(df, "2026-05-17").to_dicts()
        last = out[-1]
        # Missing column treated as False → outlier flagged as anomaly.
        assert last["is_on_ir"] is False
        assert last["is_anomaly"] is True


# ---------------------------------------------------------------------------
# Consecutive-below counter
# ---------------------------------------------------------------------------

class TestConsecutiveBelow:
    def test_counter_increments_on_consecutive_dips(self):
        rows = _noisy_baseline_rows(n=12, mean=1.0)
        # Three games below z=-1 in a row.
        for i, val in enumerate([0.3, 0.2, 0.25]):
            rows.append({
                "player_id": 1, "game_id": 13 + i,
                "game_date": f"2026-01-{13 + i:02d}",
                "metric": val, "is_on_ir": False,
            })
        out = PerformanceAnomalyDetector().compute(_perf(rows),
                                                   "2026-05-17").to_dicts()
        # Tail rows should have a monotonically growing consecutive count.
        tail = out[-3:]
        counts = [r["consecutive_below_n"] for r in tail]
        assert counts == sorted(counts)
        assert counts[-1] >= counts[0]
        assert counts[-1] >= 1

    def test_counter_resets_on_normal_game(self):
        rows = _noisy_baseline_rows(n=12, mean=1.0)
        rows.append({"player_id": 1, "game_id": 13, "game_date": "2026-01-13",
                     "metric": 0.3, "is_on_ir": False})
        rows.append({"player_id": 1, "game_id": 14, "game_date": "2026-01-14",
                     "metric": 1.0, "is_on_ir": False})
        out = PerformanceAnomalyDetector().compute(_perf(rows),
                                                   "2026-05-17").to_dicts()
        assert out[-1]["consecutive_below_n"] == 0


# ---------------------------------------------------------------------------
# Multi-player independence
# ---------------------------------------------------------------------------

class TestMultiPlayer:
    def test_players_independent(self):
        rows = (_noisy_baseline_rows(player_id=1, n=12, mean=1.0)
                + _noisy_baseline_rows(player_id=2, n=12, mean=5.0))
        out = PerformanceAnomalyDetector().compute(_perf(rows), "2026-05-17")
        for pid in (1, 2):
            sub = out.filter(pl.col("player_id") == pid).to_dicts()
            assert len(sub) == 12
            # Baselines reflect each player's own mean.
            assert sub[-1]["baseline_mean"] == pytest.approx(
                1.0 if pid == 1 else 5.0, abs=0.5
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_invalid_metric_dropped(self):
        df = pl.DataFrame(
            [
                {"player_id": 1, "game_id": 1, "game_date": "2026-01-01",
                 "metric": None, "is_on_ir": False},
                {"player_id": 1, "game_id": 2, "game_date": "2026-01-02",
                 "metric": 1.0,  "is_on_ir": False},
            ],
            schema={
                "player_id": pl.Int64, "game_id": pl.Int64,
                "game_date": pl.Utf8, "metric": pl.Float64,
                "is_on_ir":  pl.Boolean,
            },
        )
        out = PerformanceAnomalyDetector().compute(df, "2026-05-17")
        assert out["game_id"].to_list() == [2]

    def test_invalid_date_dropped(self):
        df = pl.DataFrame(
            [
                {"player_id": 1, "game_id": 1, "game_date": "nope",
                 "metric": 1.0, "is_on_ir": False},
                {"player_id": 1, "game_id": 2, "game_date": "2026-01-02",
                 "metric": 1.0, "is_on_ir": False},
            ],
            schema={
                "player_id": pl.Int64, "game_id": pl.Int64,
                "game_date": pl.Utf8, "metric": pl.Float64,
                "is_on_ir":  pl.Boolean,
            },
        )
        out = PerformanceAnomalyDetector().compute(df, "2026-05-17")
        assert out["game_id"].to_list() == [2]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_parquet(self, tmp_path):
        df = _perf(_stable_baseline_rows(n=8))
        out = PerformanceAnomalyDetector().compute(df, "2026-05-17")
        path = write_performance_anomaly(out, tmp_path, "2026-05-17")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in PERFORMANCE_ANOMALY_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        out = PerformanceAnomalyDetector().compute(_perf([]), "2026-05-17")
        path = write_performance_anomaly(out, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name
