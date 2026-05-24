"""Unit tests for models/penalty_tendency.py — Feature 4.6."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from models.penalty_tendency import (
    DataMissingWarning,
    MODEL_VERSION,
    PENALTY_TENDENCY_SCHEMA,
    compute_penalty_tendency,
    read_penalty_tendency,
    write_penalty_tendency,
)


def _pbp(rows: list[dict]) -> pl.DataFrame:
    """Build a PBP frame with all required columns, filling defaults."""
    base = {
        "game_id":              0,
        "event_type":           "shot",
        "event_owner_team_id":  None,
        "home_team_id":         8,
        "away_team_id":         10,
        "penalty_type":         None,
        "duration_minutes":     None,
    }
    return pl.DataFrame([{**base, **r} for r in rows])


def test_compute_basic_two_teams() -> None:
    pbp = _pbp([
        # game 1 — MTL@TOR (home=TOR=10, away=MTL=8). Wait — set explicit.
        {"game_id": 1, "home_team_id": 10, "away_team_id": 8,
         "event_type": "penalty", "event_owner_team_id": 8,
         "penalty_type": "MIN", "duration_minutes": 2},
        {"game_id": 1, "home_team_id": 10, "away_team_id": 8,
         "event_type": "penalty", "event_owner_team_id": 10,
         "penalty_type": "MAJ", "duration_minutes": 5},
        # game 2 — MTL@TOR same teams
        {"game_id": 2, "home_team_id": 10, "away_team_id": 8,
         "event_type": "penalty", "event_owner_team_id": 8,
         "penalty_type": "MIN", "duration_minutes": 2},
    ])
    df = compute_penalty_tendency(pbp, season=2025, team_lookup={8: "MTL", 10: "TOR"})
    assert not df.is_empty()
    assert set(df.columns) == set(PENALTY_TENDENCY_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()

    mtl = df.filter(pl.col("team") == "MTL").row(0, named=True)
    tor = df.filter(pl.col("team") == "TOR").row(0, named=True)

    # 2 games played by each team
    assert mtl["n_games"] == 2
    assert tor["n_games"] == 2
    # MTL took 2 penalties (g1 + g2), PIM = 4
    assert mtl["n_penalties_taken"] == 2
    assert mtl["pim_total"] == 4
    # TOR took 1 penalty, PIM = 5
    assert tor["n_penalties_taken"] == 1
    assert tor["pim_total"] == 5
    # MTL earned 1 PP opp (TOR's penalty), TOR earned 2 PP opps (MTL's penalties)
    assert mtl["n_pp_opportunities"] == 1
    assert tor["n_pp_opportunities"] == 2

    # ref_dim sanity tag
    assert mtl["ref_dim"] == "team-only"


def test_compute_excludes_misconducts() -> None:
    pbp = _pbp([
        {"game_id": 1, "home_team_id": 10, "away_team_id": 8,
         "event_type": "penalty", "event_owner_team_id": 8,
         "penalty_type": "MIS", "duration_minutes": 10},   # 10-minute misconduct
    ])
    df = compute_penalty_tendency(pbp, season=2025, team_lookup={8: "MTL", 10: "TOR"})
    # Misconducts excluded from PP-generating filter → no penalties counted
    mtl = df.filter(pl.col("team") == "MTL")
    if not mtl.is_empty():
        assert mtl.row(0, named=True)["n_penalties_taken"] == 0


def test_empty_input() -> None:
    df = compute_penalty_tendency(pl.DataFrame(), season=2025)
    assert df.is_empty()
    assert set(df.columns) == set(PENALTY_TENDENCY_SCHEMA.keys())


def test_no_penalties_warns() -> None:
    pbp = _pbp([
        {"game_id": 1, "home_team_id": 10, "away_team_id": 8, "event_type": "shot"},
    ])
    with pytest.warns(DataMissingWarning):
        df = compute_penalty_tendency(pbp, season=2025, team_lookup={8: "MTL", 10: "TOR"})
    # We still get baseline rows (both teams) with zero penalties
    assert not df.is_empty()
    assert (df["n_penalties_taken"] == 0).all()


def test_unknown_team_ids_filtered_when_lookup_provided() -> None:
    """Exhibition team_ids (e.g. 4 Nations team 68) must be dropped when
    a non-empty team_lookup is provided — keeps the output table tied
    to the 32 NHL clubs."""
    pbp = _pbp([
        # Real NHL game
        {"game_id": 1, "home_team_id": 10, "away_team_id": 8,
         "event_type": "penalty", "event_owner_team_id": 8,
         "penalty_type": "MIN", "duration_minutes": 2},
        # Exhibition game with team_id 68 (not in lookup)
        {"game_id": 2, "home_team_id": 68, "away_team_id": 67,
         "event_type": "penalty", "event_owner_team_id": 68,
         "penalty_type": "MIN", "duration_minutes": 2},
    ])
    df = compute_penalty_tendency(pbp, season=2025, team_lookup={8: "MTL", 10: "TOR"})
    assert set(df["team"].to_list()) == {"MTL", "TOR"}
    assert (df["team"] != "68").all()


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    pbp = _pbp([
        {"game_id": 1, "home_team_id": 10, "away_team_id": 8,
         "event_type": "penalty", "event_owner_team_id": 8,
         "penalty_type": "MIN", "duration_minutes": 2},
    ])
    df = compute_penalty_tendency(pbp, season=2025, team_lookup={8: "MTL", 10: "TOR"})
    write_penalty_tendency(df, tmp_path / "penalty_tendency", season=2025)
    rt = read_penalty_tendency(tmp_path, season=2025)
    assert rt is not None and len(rt) == len(df)
