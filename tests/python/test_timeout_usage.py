"""Unit tests for models/timeout_usage.py — Feature 4.4."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from models.timeout_usage import (
    DataMissingWarning,
    MODEL_VERSION,
    TIMEOUT_USAGE_SCHEMA,
    _period_bucket,
    _score_state,
    _time_bucket,
    compute_timeout_usage,
    read_timeout_usage,
    write_timeout_usage,
)


def test_bucket_helpers() -> None:
    assert _period_bucket(1, 0) == "P1"
    assert _period_bucket(2, 600) == "P2"
    assert _period_bucket(3, 599) == "P3_early"
    assert _period_bucket(3, 1199) == "P3_late"
    assert _period_bucket(4, 0) == "OT"

    assert _score_state(3, 1) == "leading"
    assert _score_state(1, 1) == "tied"
    assert _score_state(0, 1) == "trailing"

    assert _time_bucket(0) == "0-5m"
    assert _time_bucket(299) == "0-5m"
    assert _time_bucket(300) == "5-10m"
    assert _time_bucket(900) == "15-20m"


def test_no_timeouts_emits_warning() -> None:
    pbp = pl.DataFrame({
        "game_id":              [1, 1],
        "sort_order":           [1, 2],
        "event_type":           ["shot", "hit"],
        "event_type_raw":       ["shot-on-goal", "hit"],
        "period":               [1, 1],
        "time_in_period_secs":  [60, 90],
        "home_score":           [0, 0],
        "away_score":           [0, 0],
        "home_team_id":         [8, 8],
        "away_team_id":         [10, 10],
        "event_owner_team_id":  [8, 10],
    })
    with pytest.warns(DataMissingWarning):
        df = compute_timeout_usage(pbp, season=2025, team_lookup={8: "MTL", 10: "TOR"})
    assert df.is_empty()
    assert set(df.columns) == set(TIMEOUT_USAGE_SCHEMA.keys())


def test_compute_timeout_usage_aggregates() -> None:
    """Synthesize 2 games with team-timeout events and verify per-team rates."""
    pbp = pl.DataFrame({
        "game_id":              [1, 1, 1, 2, 2],
        "sort_order":           [1, 2, 3, 1, 2],
        "event_type":           ["shot", "timeout", "shot", "shot", "timeout"],
        "event_type_raw":       ["shot-on-goal", "team-timeout", "shot-on-goal",
                                  "shot-on-goal", "team-timeout"],
        "period":               [3, 3, 3, 3, 3],
        "time_in_period_secs":  [60, 1100, 1150, 200, 1080],
        "home_score":           [1, 2, 2, 0, 1],
        "away_score":           [0, 0, 1, 1, 1],
        "home_team_id":         [8, 8, 8, 10, 10],
        "away_team_id":         [10, 10, 10, 8, 8],
        "event_owner_team_id":  [8, 8, 10, 10, 8],
    })
    df = compute_timeout_usage(pbp, season=2025, team_lookup={8: "MTL", 10: "TOR"})
    assert not df.is_empty()
    assert set(df.columns) == set(TIMEOUT_USAGE_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()

    # Two timeouts in the input — both owned by team 8 (MTL).
    mtl = df.filter(pl.col("team") == "MTL")
    assert not mtl.is_empty()
    assert mtl["n_timeouts"].sum() == 2


def test_compute_timeout_usage_empty_input() -> None:
    df = compute_timeout_usage(pl.DataFrame(), season=2025)
    assert df.is_empty()
    assert set(df.columns) == set(TIMEOUT_USAGE_SCHEMA.keys())


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    pbp = pl.DataFrame({
        "game_id":              [1],
        "sort_order":           [1],
        "event_type":           ["timeout"],
        "event_type_raw":       ["team-timeout"],
        "period":               [3],
        "time_in_period_secs":  [1100],
        "home_score":           [1],
        "away_score":           [2],
        "home_team_id":         [8],
        "away_team_id":         [10],
        "event_owner_team_id":  [8],
    })
    df = compute_timeout_usage(pbp, season=2025, team_lookup={8: "MTL", 10: "TOR"})
    write_timeout_usage(df, tmp_path / "timeout_usage", season=2025)
    rt = read_timeout_usage(tmp_path, season=2025)
    assert rt is not None and len(rt) == len(df)
