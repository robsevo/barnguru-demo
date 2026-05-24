"""Unit tests for models/venue_atmosphere.py — Feature 4.19."""
from __future__ import annotations
from pathlib import Path
import polars as pl
import pytest
from models.venue_atmosphere import (
    VENUE_ATMOSPHERE_SCHEMA, MODEL_VERSION, DataMissingWarning,
    compute_venue_atmosphere, write_venue_atmosphere, read_venue_atmosphere,
)

_BASE = {"game_id": 1, "event_type": "shot", "event_owner_team_id": None,
         "home_team_id": 10, "away_team_id": 8, "shot_result": None,
         "strength": "ev", "penalty_type": None, "duration_minutes": None,
         "winning_player_id": None, "losing_player_id": None,
         "home_score": 0, "away_score": 0, "x_coord": None, "y_coord": None,
         "zone_code": "O", "carry_in": None, "entering_team_id": None,
         "turnover_player_id": None, "hitter_id": None, "hittee_id": None,
         "blocker_id": None, "shot_type": None, "shot_goalie_id": None,
         "period": 1, "period_type": "REG", "event_id": 1, "sort_order": 1,
         "time_in_period": "00:00", "time_in_period_secs": 0,
         "time_remaining_secs": 1200, "event_type_raw": "shot",
         "away_skaters": 5, "home_skaters": 5, "away_goalie": True,
         "home_goalie": True, "scorer_id": None, "assist1_id": None,
         "assist2_id": None, "committed_by_id": None, "drawn_by_id": None,
         "penalty_description": None, "shooter_id": None}

def _pbp(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame([{**_BASE, **r} for r in rows])

def test_schema_and_version() -> None:
    pbp = _pbp([
        {"event_type": "shot", "event_owner_team_id": 10, "shot_result": "on_goal"},
        {"event_type": "goal", "event_owner_team_id": 10},
        {"event_type": "faceoff", "event_owner_team_id": 10, "winning_player_id": 1},
        {"event_type": "penalty", "event_owner_team_id": 8, "penalty_type": "MIN", "duration_minutes": 2},
    ])
    df = compute_venue_atmosphere(pbp, pl.DataFrame(), {10: "TOR", 8: "MTL"}, 2025)
    assert not df.is_empty()
    assert set(df.columns) == set(VENUE_ATMOSPHERE_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()

def test_scare_rank_bounded() -> None:
    pbp = _pbp([
        {"event_type": "shot", "event_owner_team_id": 10, "shot_result": "on_goal"},
        {"event_type": "goal", "event_owner_team_id": 8},
        {"event_type": "faceoff", "event_owner_team_id": 10, "winning_player_id": 1},
    ])
    df = compute_venue_atmosphere(pbp, pl.DataFrame(), {10: "TOR", 8: "MTL"}, 2025)
    for r in df.iter_rows(named=True):
        assert 0.0 <= r["scare_rank"] <= 1.0

def test_empty_pbp_warns() -> None:
    with pytest.warns(DataMissingWarning):
        df = compute_venue_atmosphere(pl.DataFrame(), pl.DataFrame(), {10: "TOR"}, 2025)
    assert df.is_empty()

def test_write_read_roundtrip(tmp_path: Path) -> None:
    pbp = _pbp([{"event_type": "shot", "event_owner_team_id": 10, "shot_result": "on_goal"}])
    df = compute_venue_atmosphere(pbp, pl.DataFrame(), {10: "TOR", 8: "MTL"}, 2025)
    write_venue_atmosphere(df, tmp_path / "venue_atmosphere", 2025)
    rt = read_venue_atmosphere(tmp_path, 2025)
    assert rt is not None and len(rt) == len(df)
