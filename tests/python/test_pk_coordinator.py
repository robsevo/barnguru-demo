"""Unit tests for models/pk_coordinator.py — Feature 4.10."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from models.pk_coordinator import (
    DataMissingWarning,
    MODEL_VERSION,
    PK_COORDINATOR_SCHEMA,
    _pk_shot_stats,
    compute_pk_coordinator,
    read_pk_coordinator,
    write_pk_coordinator,
)


_SHOT_BASE = {
    "shot_id":          0,
    "season":           2025,
    "game_id":          0,
    "is_playoff":       False,
    "shooting_team":    "TOR",
    "home_team":        "TOR",
    "away_team":        "MTL",
    "shooter_name":     "x",
    "shooter_id":       0,
    "goalie_id":        0,
    "x_goal":           0.10,
    "shot_distance":    25.0,
    "shot_result":      "missed",
    "is_goal":          False,
    "home_skaters":     5,
    "away_skaters":     4,
    "player_position":  "C",
    "shooter_hand":     "L",
    "period":           1,
    "time":             0,
    "x_on_goal":        0.5,
    "x_cord":           0,    "y_cord":           0,
    "arena_adj_x":      0,    "arena_adj_y":      0,
    "shot_angle":       0,    "arena_adj_distance": 25.0,
    "shot_type":        "wrist",
    "event":            "SHOT",
    "last_event_team":  "TOR",
}


def _shots(rows: list[dict]) -> pl.DataFrame:
    payload = [{**_SHOT_BASE, **r} for r in (rows or [_SHOT_BASE])]
    df = pl.DataFrame(payload[:len(rows)] if rows else payload)
    return df if rows else df.clear()


def _st_deployment() -> pl.DataFrame:
    return pl.DataFrame([
        {"team": "TOR", "season": 2025, "unit_type": "PK1",
         "personnel": [101, 102, 103, 104],
         "unit_toi_secs": 400.0, "share_of_st_toi": 0.4,
         "team_st_toi": 1000.0, "team_st_gp": 10,
         "model_version": "st_deployment_v1"},
        {"team": "TOR", "season": 2025, "unit_type": "PK2",
         "personnel": [105, 106, 107, 108],
         "unit_toi_secs": 200.0, "share_of_st_toi": 0.2,
         "team_st_toi": 1000.0, "team_st_gp": 10,
         "model_version": "st_deployment_v1"},
        {"team": "MTL", "season": 2025, "unit_type": "PK1",
         "personnel": [201, 202, 203, 204],
         "unit_toi_secs": 300.0, "share_of_st_toi": 0.5,
         "team_st_toi": 600.0, "team_st_gp": 10,
         "model_version": "st_deployment_v1"},
    ])


def test_basic_schema() -> None:
    df = compute_pk_coordinator(
        shots_df = _shots([]),
        st_df    = _st_deployment(),
        season   = 2025,
    )
    assert not df.is_empty()
    assert set(df.columns) == set(PK_COORDINATOR_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()


def test_pk_sa_counts_opponent_pp_shots() -> None:
    """TOR defending (away=MTL shoots at 5v4) → TOR gets pk_sa credit."""
    shots = _shots([
        # MTL shoots while TOR is shorthanded (home=TOR has 4, away=MTL has 5)
        {"shooting_team": "MTL", "home_team": "TOR", "away_team": "MTL",
         "home_skaters": 4, "away_skaters": 5, "x_goal": 0.12, "is_goal": False},
        {"shooting_team": "MTL", "home_team": "TOR", "away_team": "MTL",
         "home_skaters": 4, "away_skaters": 5, "x_goal": 0.15, "is_goal": True},
        # TOR SH shot (TOR has 4, MTL has 5 — TOR is shorthanded shooter)
        {"shooting_team": "TOR", "home_team": "TOR", "away_team": "MTL",
         "home_skaters": 4, "away_skaters": 5, "x_goal": 0.05, "is_goal": False},
    ])
    stats = _pk_shot_stats(shots)
    tor = stats.filter(pl.col("team") == "TOR")
    assert not tor.is_empty()
    r = tor.row(0, named=True)
    assert r["pk_sa"] == 2
    assert r["pk_ga"] == 1
    assert r["sh_shots_for"] == 1


def test_pk_save_pct_in_range() -> None:
    shots = _shots([
        {"shooting_team": "MTL", "home_team": "TOR", "away_team": "MTL",
         "home_skaters": 4, "away_skaters": 5, "x_goal": 0.10, "is_goal": False},
        {"shooting_team": "MTL", "home_team": "TOR", "away_team": "MTL",
         "home_skaters": 4, "away_skaters": 5, "x_goal": 0.20, "is_goal": True},
    ])
    df = compute_pk_coordinator(shots_df=shots, st_df=_st_deployment(), season=2025)
    for r in df.iter_rows(named=True):
        assert 0.0 <= r["pk_save_pct"] <= 1.0


def test_empty_st_deployment_warns_and_empty() -> None:
    empty_st = pl.DataFrame(schema={
        "team": pl.Utf8, "season": pl.Int64, "unit_type": pl.Utf8,
        "personnel": pl.List(pl.Int64), "unit_toi_secs": pl.Float64,
        "share_of_st_toi": pl.Float64, "team_st_toi": pl.Float64,
        "team_st_gp": pl.Int64, "model_version": pl.Utf8,
    })
    with pytest.warns(DataMissingWarning):
        df = compute_pk_coordinator(shots_df=_shots([]), st_df=empty_st, season=2025)
    assert df.is_empty()
    assert set(df.columns) == set(PK_COORDINATOR_SCHEMA.keys())


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    shots = _shots([
        {"shooting_team": "MTL", "home_team": "TOR", "away_team": "MTL",
         "home_skaters": 4, "away_skaters": 5, "x_goal": 0.10, "is_goal": False},
    ])
    df = compute_pk_coordinator(shots_df=shots, st_df=_st_deployment(), season=2025)
    write_pk_coordinator(df, tmp_path / "pk_coordinator", season=2025)
    rt = read_pk_coordinator(tmp_path, season=2025)
    assert rt is not None and len(rt) == len(df)
    assert set(rt.columns) == set(PK_COORDINATOR_SCHEMA.keys())
