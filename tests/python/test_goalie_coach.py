"""Unit tests for models/goalie_coach.py — Feature 4.8."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from models.goalie_coach import (
    CHANGE_THRESHOLD,
    GOALIE_COACH_SCHEMA,
    MODEL_VERSION,
    SPLIT_GP_DEFAULT,
    _per_game_save_stats,
    _split_save_pct,
    compute_goalie_coach_curve,
    read_goalie_coach_curve,
    write_goalie_coach_curve,
)


_PBP_BASE = {
    "game_id":              0,
    "event_type":           "shot",
    "event_owner_team_id":  None,
    "home_team_id":         10,
    "away_team_id":         8,
    "shot_result":          None,
}

def _pbp(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame([{**_PBP_BASE, **r} for r in rows])


def _two_game_pbp() -> pl.DataFrame:
    """TOR (defending team 10) faces 6 shots over 2 games, allows 1 goal."""
    return _pbp([
        # Game 1: MTL takes 3 shots (1 on_goal saved, 1 missed, 1 goal); TOR takes 1 (missed)
        {"game_id": 1, "event_type": "shot", "event_owner_team_id": 8,  "shot_result": "on_goal"},
        {"game_id": 1, "event_type": "shot", "event_owner_team_id": 8,  "shot_result": "missed"},
        {"game_id": 1, "event_type": "goal", "event_owner_team_id": 8},
        {"game_id": 1, "event_type": "shot", "event_owner_team_id": 10, "shot_result": "missed"},
        # Game 2: MTL takes 2 (1 on_goal, 1 on_goal) — no goals; TOR takes 1 on_goal
        {"game_id": 2, "event_type": "shot", "event_owner_team_id": 8,  "shot_result": "on_goal"},
        {"game_id": 2, "event_type": "shot", "event_owner_team_id": 8,  "shot_result": "on_goal"},
        {"game_id": 2, "event_type": "shot", "event_owner_team_id": 10, "shot_result": "on_goal"},
    ])


def test_schema_and_version() -> None:
    df = compute_goalie_coach_curve(
        pbp_by_season   = {2025: _two_game_pbp()},
        season          = 2025,
        team_lookup     = {8: "MTL", 10: "TOR"},
        split_gp        = 1,
        rolling_window  = 1,
    )
    assert not df.is_empty()
    assert set(df.columns) == set(GOALIE_COACH_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()


def test_save_pct_arithmetic() -> None:
    df = compute_goalie_coach_curve(
        pbp_by_season   = {2025: _two_game_pbp()},
        season          = 2025,
        team_lookup     = {8: "MTL", 10: "TOR"},
        split_gp        = 1,
        rolling_window  = 1,
    )
    tor = df.filter(pl.col("team") == "TOR").row(0, named=True)
    # TOR shots-against = 3 on_goal + 1 goal = 4; goals-against = 1 → SV% = 0.75
    assert tor["shots_against"] == 4
    assert tor["goals_against"] == 1
    assert tor["season_save_pct"] == pytest.approx(0.75, rel=1e-3)
    assert tor["gp"] == 2


def test_save_pct_in_unit_range() -> None:
    df = compute_goalie_coach_curve(
        pbp_by_season   = {2025: _two_game_pbp()},
        season          = 2025,
        team_lookup     = {8: "MTL", 10: "TOR"},
        split_gp        = 1,
    )
    for r in df.iter_rows(named=True):
        if r["shots_against"] > 0:
            assert 0.0 <= r["season_save_pct"] <= 1.0


def test_change_point_detection_flag() -> None:
    """Engineered: early window heavy GA, late window perfect saves
    → split_delta > CHANGE_THRESHOLD → change_point_detected=True."""
    rows: list[dict] = []
    # 3 "early" games — bad save% (1 saved, 1 goal per game = 50% SV)
    for gid in range(1, 4):
        rows += [
            {"game_id": gid, "event_type": "shot", "event_owner_team_id": 8, "shot_result": "on_goal"},
            {"game_id": gid, "event_type": "goal", "event_owner_team_id": 8},
        ]
    # 3 "late" games — perfect saves (5 on_goal each, 0 goals)
    for gid in range(4, 7):
        rows += [
            {"game_id": gid, "event_type": "shot", "event_owner_team_id": 8, "shot_result": "on_goal"}
        ] * 5
    pbp = _pbp(rows)
    df = compute_goalie_coach_curve(
        pbp_by_season   = {2025: pbp},
        season          = 2025,
        team_lookup     = {8: "MTL", 10: "TOR"},
        split_gp        = 3,
        rolling_window  = 3,
        change_threshold = 0.05,
    )
    tor = df.filter(pl.col("team") == "TOR").row(0, named=True)
    assert tor["change_point_detected"] is True
    assert tor["split_delta"] is not None and tor["split_delta"] > 0.4   # huge jump up


def test_no_prior_season_yields_nan_delta() -> None:
    df = compute_goalie_coach_curve(
        pbp_by_season   = {2025: _two_game_pbp()},
        season          = 2025,
        team_lookup     = {8: "MTL", 10: "TOR"},
        split_gp        = 1,
    )
    tor = df.filter(pl.col("team") == "TOR").row(0, named=True)
    # No 2024 PBP supplied → NaN
    assert tor["prior_save_pct"] != tor["prior_save_pct"]   # NaN check
    assert tor["save_pct_delta"] != tor["save_pct_delta"]


def test_missing_pbp_for_target_season_warns_and_empty() -> None:
    with pytest.warns(UserWarning):
        df = compute_goalie_coach_curve(
            pbp_by_season   = {},
            season          = 2025,
            team_lookup     = {8: "MTL", 10: "TOR"},
        )
    assert df.is_empty()
    assert set(df.columns) == set(GOALIE_COACH_SCHEMA.keys())


def test_split_save_pct_helper_empty() -> None:
    e, l = _split_save_pct(pl.DataFrame({"shots_against": [], "goals_against": []}, schema={"shots_against": pl.Int64, "goals_against": pl.Int64}), SPLIT_GP_DEFAULT)
    assert e is None and l is None


def test_per_game_save_stats_raises_on_missing_cols() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        _per_game_save_stats(pl.DataFrame({"game_id": [1], "event_type": ["shot"]}))


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    df = compute_goalie_coach_curve(
        pbp_by_season   = {2025: _two_game_pbp()},
        season          = 2025,
        team_lookup     = {8: "MTL", 10: "TOR"},
        split_gp        = 1,
    )
    write_goalie_coach_curve(df, tmp_path / "goalie_coach_curve", season=2025)
    rt = read_goalie_coach_curve(tmp_path, season=2025)
    assert rt is not None and len(rt) == len(df)
    assert set(rt.columns) == set(GOALIE_COACH_SCHEMA.keys())


def test_change_threshold_default_is_finite() -> None:
    # Sanity guard on constant — protects against accidental 0/inf.
    assert 0.0 < CHANGE_THRESHOLD < 1.0
