"""Tests for Feature 3.14 — ConcussionHistoryFlag."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from models.concussion_history import (
    CONCUSSION_HISTORY_SCHEMA,
    EPISODE_GAP_DAYS,
    MAX_MULTIPLIER,
    PER_EPISODE_INCREMENT,
    ConcussionHistoryFlag,
    concussion_multiplier,
    write_concussion_history,
)
from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _hist(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "player_id":     pl.Int64,
        "observed_date": pl.Utf8,
        "injury_type":   pl.Utf8,
        "injury_detail": pl.Utf8,
        "status_raw":    pl.Utf8,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    defaults = {"injury_type": None, "injury_detail": None, "status_raw": None}
    filled = [{**defaults, **r} for r in rows]
    return pl.DataFrame(filled, schema=schema)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_required_columns_warns(self):
        bad = pl.DataFrame({"foo": ["bar"]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = ConcussionHistoryFlag().compute(bad, "2026-05-17")
        assert len(out) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_no_text_columns_warns(self):
        bad = pl.DataFrame({"player_id": [1], "observed_date": ["2026-05-01"]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = ConcussionHistoryFlag().compute(bad, "2026-05-17")
        assert len(out) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_input_returns_empty_schema(self):
        out = ConcussionHistoryFlag().compute(_hist([]), "2026-05-17")
        for col in CONCUSSION_HISTORY_SCHEMA:
            assert col in out.columns
        assert len(out) == 0

    def test_output_schema(self):
        df = _hist([{"player_id": 1, "observed_date": "2026-05-01",
                     "injury_type": "Concussion"}])
        out = ConcussionHistoryFlag().compute(df, "2026-05-17")
        assert set(out.columns) == set(CONCUSSION_HISTORY_SCHEMA.keys())

    def test_bad_as_of_date_raises(self):
        df = _hist([{"player_id": 1, "observed_date": "2026-05-01",
                     "injury_type": "Concussion"}])
        with pytest.raises(ValueError):
            ConcussionHistoryFlag().compute(df, "not-a-date")

    def test_bad_params_rejected(self):
        with pytest.raises(ValueError):
            ConcussionHistoryFlag(per_episode_increment=-0.1)
        with pytest.raises(ValueError):
            ConcussionHistoryFlag(max_multiplier=0.5)
        with pytest.raises(ValueError):
            ConcussionHistoryFlag(episode_gap_days=0)
        with pytest.raises(ValueError):
            ConcussionHistoryFlag(keywords=frozenset())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_per_episode_increment_positive(self):
        assert PER_EPISODE_INCREMENT > 0

    def test_max_multiplier_above_one(self):
        assert MAX_MULTIPLIER > 1.0

    def test_episode_gap_positive(self):
        assert EPISODE_GAP_DAYS >= 1


# ---------------------------------------------------------------------------
# Multiplier formula
# ---------------------------------------------------------------------------

class TestMultiplier:
    def test_zero_count_is_one(self):
        assert concussion_multiplier(0) == pytest.approx(1.0)

    def test_negative_count_clamped_to_zero(self):
        assert concussion_multiplier(-3) == pytest.approx(1.0)

    def test_linear_below_cap(self):
        assert concussion_multiplier(1) == pytest.approx(1.0 + PER_EPISODE_INCREMENT)
        assert concussion_multiplier(2) == pytest.approx(1.0 + 2 * PER_EPISODE_INCREMENT)

    def test_clamped_at_max(self):
        big = concussion_multiplier(100)
        assert big == pytest.approx(MAX_MULTIPLIER)

    def test_monotone_non_decreasing(self):
        prev = concussion_multiplier(0)
        for c in range(1, 20):
            curr = concussion_multiplier(c)
            assert curr >= prev
            prev = curr


# ---------------------------------------------------------------------------
# Compute path
# ---------------------------------------------------------------------------

class TestCompute:
    def test_no_concussion_keywords_drops_player(self):
        # Player only has an upper-body injury → not counted.
        df = _hist([
            {"player_id": 1, "observed_date": "2026-04-01",
             "injury_type": "Upper Body"},
        ])
        out = ConcussionHistoryFlag().compute(df, "2026-05-17")
        assert len(out) == 0

    def test_single_concussion_episode(self):
        df = _hist([
            {"player_id": 1, "observed_date": "2026-04-01",
             "injury_type": "Concussion"},
            {"player_id": 1, "observed_date": "2026-04-03",
             "injury_type": "Concussion"},
            {"player_id": 1, "observed_date": "2026-04-08",
             "injury_type": "Concussion"},
        ])
        out = ConcussionHistoryFlag().compute(df, "2026-05-17").to_dicts()[0]
        assert out["prior_concussion_count"] == 1
        assert out["has_prior_concussion"] is True
        assert out["last_concussion_date"] == "2026-04-08"
        assert out["fatigue_sensitivity_multiplier"] == pytest.approx(
            1.0 + PER_EPISODE_INCREMENT
        )

    def test_two_episodes_far_apart(self):
        # First episode: April. Second: November (well past gap).
        df = _hist([
            {"player_id": 1, "observed_date": "2025-04-01",
             "injury_type": "Concussion"},
            {"player_id": 1, "observed_date": "2025-04-05",
             "injury_type": "Concussion"},
            {"player_id": 1, "observed_date": "2025-11-10",
             "injury_type": "Concussion"},
            {"player_id": 1, "observed_date": "2025-11-12",
             "injury_type": "Concussion"},
        ])
        out = ConcussionHistoryFlag().compute(df, "2026-05-17").to_dicts()[0]
        assert out["prior_concussion_count"] == 2
        assert out["fatigue_sensitivity_multiplier"] == pytest.approx(
            1.0 + 2 * PER_EPISODE_INCREMENT
        )

    def test_head_keyword_also_matches(self):
        df = _hist([
            {"player_id": 1, "observed_date": "2026-04-01",
             "injury_type": "Head"},
        ])
        out = ConcussionHistoryFlag().compute(df, "2026-05-17")
        assert len(out) == 1

    def test_status_raw_matches(self):
        df = _hist([
            {"player_id": 1, "observed_date": "2026-04-01",
             "status_raw": "Concussion Protocol"},
        ])
        out = ConcussionHistoryFlag().compute(df, "2026-05-17")
        assert len(out) == 1

    def test_case_insensitive_match(self):
        df = _hist([
            {"player_id": 1, "observed_date": "2026-04-01",
             "injury_detail": "CONCUSSION-LIKE SYMPTOMS"},
        ])
        out = ConcussionHistoryFlag().compute(df, "2026-05-17")
        assert len(out) == 1

    def test_days_since_last(self):
        df = _hist([
            {"player_id": 1, "observed_date": "2026-05-10",
             "injury_type": "Concussion"},
        ])
        out = ConcussionHistoryFlag().compute(df, "2026-05-17").to_dicts()[0]
        assert out["days_since_last_concussion"] == 7

    def test_future_observations_ignored(self):
        df = _hist([
            {"player_id": 1, "observed_date": "2026-06-01",
             "injury_type": "Concussion"},
        ])
        out = ConcussionHistoryFlag().compute(df, "2026-05-17")
        assert len(out) == 0

    def test_max_multiplier_cap_observed(self):
        # 20 episodes well-spaced and all in the past → capped at MAX_MULTIPLIER.
        rows = []
        for k in range(20):
            year = 2005 + k    # 2005..2024, all ≤ as_of_date 2026-05-17
            rows.append({
                "player_id": 1,
                "observed_date": f"{year}-04-01",
                "injury_type": "Concussion",
            })
        out = ConcussionHistoryFlag().compute(_hist(rows), "2026-05-17").to_dicts()[0]
        assert out["fatigue_sensitivity_multiplier"] == pytest.approx(MAX_MULTIPLIER)
        assert out["prior_concussion_count"] == 20

    def test_multipliers_in_valid_range(self):
        df = _hist([
            {"player_id": i, "observed_date": "2026-04-01",
             "injury_type": "Concussion"}
            for i in range(1, 6)
        ])
        out = ConcussionHistoryFlag().compute(df, "2026-05-17")
        for mult in out["fatigue_sensitivity_multiplier"].to_list():
            assert 1.0 <= mult <= MAX_MULTIPLIER

    def test_custom_keywords(self):
        # Override keywords: only "neck" qualifies.
        flag = ConcussionHistoryFlag(keywords=frozenset({"neck"}))
        df = _hist([
            {"player_id": 1, "observed_date": "2026-04-01",
             "injury_type": "Concussion"},
            {"player_id": 2, "observed_date": "2026-04-01",
             "injury_type": "Neck Strain"},
        ])
        out = flag.compute(df, "2026-05-17")
        pids = set(out["player_id"].to_list())
        assert pids == {2}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_parquet(self, tmp_path):
        df = _hist([{"player_id": 1, "observed_date": "2026-04-01",
                     "injury_type": "Concussion"}])
        out = ConcussionHistoryFlag().compute(df, "2026-05-17")
        path = write_concussion_history(out, tmp_path, "2026-05-17")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in CONCUSSION_HISTORY_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        out = ConcussionHistoryFlag().compute(_hist([]), "2026-05-17")
        path = write_concussion_history(out, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name
