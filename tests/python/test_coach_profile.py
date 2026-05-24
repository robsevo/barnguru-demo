"""Unit tests for models/coach_profile.py — Feature 4.7."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from models.coach_profile import (
    COACH_PROFILE_SCHEMA,
    DataMissingWarning,
    MODEL_VERSION,
    _first_named_season,
    _per_team_game_summary,
    compute_coach_profiles,
    read_coach_profiles,
    write_coach_profiles,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PBP_BASE = {
    "game_id":              0,
    "event_id":             0,
    "period":               1,
    "period_type":          "REG",
    "event_type":           "shot",
    "event_owner_team_id":  None,
    "x_coord":              None, "y_coord": None,
    "zone_code":            "O",
    "shot_result":          None,
    "shot_type":            None,
    "strength":             "ev",
    "home_team_id":         10,
    "away_team_id":         8,
    "home_score":           0,
    "away_score":           0,
    "penalty_type":         None,
    "duration_minutes":     None,
    "carry_in":             None,
    "entering_team_id":     None,
}

def _pbp(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame([{**_PBP_BASE, **r} for r in rows])


def _two_game_pbp() -> pl.DataFrame:
    """Two-game season: TOR (10) vs MTL (8).
    Game 1: TOR wins 3-2 in regulation.
    Game 2: TOR loses 1-2 in overtime."""
    return _pbp([
        # Game 1: shots, one goal each + one TOR PP goal off MTL minor
        {"game_id": 1, "event_type": "shot",   "event_owner_team_id": 10, "shot_result": "on_goal", "home_score": 0, "away_score": 0},
        {"game_id": 1, "event_type": "shot",   "event_owner_team_id": 8,  "shot_result": "missed",  "home_score": 0, "away_score": 0},
        {"game_id": 1, "event_type": "penalty","event_owner_team_id": 8,  "penalty_type": "MIN", "duration_minutes": 2},
        {"game_id": 1, "event_type": "goal",   "event_owner_team_id": 10, "strength": "pp",  "home_score": 1, "away_score": 0},
        {"game_id": 1, "event_type": "goal",   "event_owner_team_id": 10, "strength": "ev",  "home_score": 2, "away_score": 0},
        {"game_id": 1, "event_type": "goal",   "event_owner_team_id": 10, "strength": "ev",  "home_score": 3, "away_score": 0},
        {"game_id": 1, "event_type": "goal",   "event_owner_team_id": 8,  "strength": "ev",  "home_score": 3, "away_score": 1},
        {"game_id": 1, "event_type": "goal",   "event_owner_team_id": 8,  "strength": "ev",  "home_score": 3, "away_score": 2},
        {"game_id": 1, "event_type": "game_end","home_score": 3, "away_score": 2, "period_type": "REG"},
        # Game 2: OT thriller — final 1-2, OT period present
        {"game_id": 2, "event_type": "shot",   "event_owner_team_id": 10, "shot_result": "on_goal", "home_score": 0, "away_score": 0},
        {"game_id": 2, "event_type": "goal",   "event_owner_team_id": 10, "strength": "ev",  "home_score": 1, "away_score": 0},
        {"game_id": 2, "event_type": "goal",   "event_owner_team_id": 8,  "strength": "ev",  "home_score": 1, "away_score": 1},
        {"game_id": 2, "event_type": "goal",   "event_owner_team_id": 8,  "strength": "ev",  "home_score": 1, "away_score": 2, "period_type": "OT"},
        {"game_id": 2, "event_type": "game_end","home_score": 1, "away_score": 2, "period_type": "OT"},
    ])


def _coaches_two() -> list[dict]:
    return [
        {"team": "TOR", "name": "Coach Tor",  "first_named_head_coach": "2025-08-01", "notes": ""},
        {"team": "MTL", "name": "Coach Mtl",  "first_named_head_coach": "2025-08-01", "notes": ""},
    ]


# ---------------------------------------------------------------------------
# Schema / version guards
# ---------------------------------------------------------------------------


def test_compute_basic_columns_and_version() -> None:
    df = compute_coach_profiles(
        pbp_by_season   = {2025: _two_game_pbp()},
        coaches         = _coaches_two(),
        team_lookup     = {8: "MTL", 10: "TOR"},
        snapshot_season = 2025,
    )
    assert not df.is_empty()
    assert set(df.columns) == set(COACH_PROFILE_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()
    assert set(df["coach_name"].to_list()) == {"Coach Tor", "Coach Mtl"}


def test_compute_wins_losses_and_points() -> None:
    df = compute_coach_profiles(
        pbp_by_season   = {2025: _two_game_pbp()},
        coaches         = _coaches_two(),
        team_lookup     = {8: "MTL", 10: "TOR"},
        snapshot_season = 2025,
    )
    tor = df.filter(pl.col("team") == "TOR").row(0, named=True)
    mtl = df.filter(pl.col("team") == "MTL").row(0, named=True)

    # TOR: game 1 W, game 2 OTL → wins=1, ot_losses=1
    assert tor["wins"] == 1
    assert tor["ot_losses"] == 1
    assert tor["points"] == 2 * 1 + 1   # W=2 + OTL=1
    assert tor["points_pct"] == pytest.approx(3 / 4, rel=1e-3)
    assert tor["gp_under_coach"] == 2

    # MTL: game 1 L, game 2 OTW → ot_wins=1, losses=1
    assert mtl["ot_wins"] == 1
    assert mtl["losses"] == 1
    assert mtl["points"] == 2

    # GF/GA arithmetic
    assert tor["gf_per_game"] == pytest.approx((3 + 1) / 2)
    assert tor["ga_per_game"] == pytest.approx((2 + 2) / 2)


def test_pp_pct_and_pk_pct_in_range() -> None:
    df = compute_coach_profiles(
        pbp_by_season   = {2025: _two_game_pbp()},
        coaches         = _coaches_two(),
        team_lookup     = {8: "MTL", 10: "TOR"},
        snapshot_season = 2025,
    )
    for r in df.iter_rows(named=True):
        assert 0.0 <= r["pp_pct"] <= 1.0
        assert 0.0 <= r["pk_pct"] <= 1.0
        assert 0.0 <= r["points_pct"] <= 1.0


def test_first_named_season_helper() -> None:
    # Pre-season hire → start year applies (Aug 2024)
    assert _first_named_season("2024-08-12") == 2024
    # Mid-season hire (Nov) → that season's start year
    assert _first_named_season("2024-11-19") == 2024
    # Spring hire → previous season's start year (April 2024 = 2023-24 season)
    assert _first_named_season("2024-04-22") == 2023
    # Malformed / empty
    assert _first_named_season(None) is None
    assert _first_named_season("") is None
    assert _first_named_season("notadate") is None


def test_per_team_game_summary_raises_on_missing_cols() -> None:
    bad = pl.DataFrame({"game_id": [1], "event_type": ["shot"]})
    with pytest.raises(ValueError, match="missing required columns"):
        _per_team_game_summary(bad)


def test_unknown_team_abbrev_produces_stub_row() -> None:
    coaches = [{"team": "ZZZ", "name": "Mystery Coach",
                "first_named_head_coach": "2020-01-01", "notes": "test"}]
    df = compute_coach_profiles(
        pbp_by_season   = {2025: _two_game_pbp()},
        coaches         = coaches,
        team_lookup     = {8: "MTL", 10: "TOR"},
        snapshot_season = 2025,
    )
    # Stub row exists with zeroed counts; never silently drop the coach.
    assert len(df) == 1
    row = df.row(0, named=True)
    assert row["coach_name"] == "Mystery Coach"
    assert row["gp_under_coach"] == 0
    assert row["points"] == 0
    assert row["points_pct"] == 0.0


def test_empty_coaches_warns_and_returns_empty() -> None:
    with pytest.warns(DataMissingWarning):
        df = compute_coach_profiles(
            pbp_by_season   = {2025: _two_game_pbp()},
            coaches         = [],
            team_lookup     = {8: "MTL", 10: "TOR"},
            snapshot_season = 2025,
        )
    assert df.is_empty()
    assert set(df.columns) == set(COACH_PROFILE_SCHEMA.keys())


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    df = compute_coach_profiles(
        pbp_by_season   = {2025: _two_game_pbp()},
        coaches         = _coaches_two(),
        team_lookup     = {8: "MTL", 10: "TOR"},
        snapshot_season = 2025,
    )
    write_coach_profiles(df, tmp_path / "coach_profiles", season=2025)
    rt = read_coach_profiles(tmp_path, season=2025)
    assert rt is not None and len(rt) == len(df)
    # Schema matches end-to-end
    assert set(rt.columns) == set(COACH_PROFILE_SCHEMA.keys())
