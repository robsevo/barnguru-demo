"""Unit tests for models/pp_coordinator.py — Feature 4.9."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from models.pp_coordinator import (
    DataMissingWarning,
    MODEL_VERSION,
    PP_COORDINATOR_SCHEMA,
    _pp_shot_stats,
    compute_pp_coordinator,
    read_pp_coordinator,
    write_pp_coordinator,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

_PBP_BASE = {
    "event_type":           "zone_entry",
    "strength":             "pp",
    "carry_in":             None,
    "entering_team_id":     None,
}

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


def _pbp(rows: list[dict]) -> pl.DataFrame:
    # Always include the base row so empty inputs still infer a schema.
    payload = [_PBP_BASE.copy() for _ in (rows or [{}])]
    payload = [{**base, **(rows[i] if i < len(rows) else {})}
               for i, base in enumerate(payload)]
    df = pl.DataFrame(payload)
    return df if rows else df.clear()


def _shots(rows: list[dict]) -> pl.DataFrame:
    payload = [_SHOT_BASE.copy() for _ in (rows or [{}])]
    payload = [{**base, **(rows[i] if i < len(rows) else {})}
               for i, base in enumerate(payload)]
    df = pl.DataFrame(payload)
    return df if rows else df.clear()


def _st_deployment() -> pl.DataFrame:
    """Two-team PP1/PP2 fixture. TOR PP1 has 1 D (id=200);
    MTL PP1 has no D → QB resolves to None."""
    return pl.DataFrame([
        {"team": "TOR", "season": 2025, "unit_type": "PP1",
         "personnel": [101, 102, 103, 104, 200],
         "unit_toi_secs": 600.0, "share_of_st_toi": 0.5,
         "team_st_toi": 1200.0, "team_st_gp": 10,
         "model_version": "st_deployment_v1"},
        {"team": "TOR", "season": 2025, "unit_type": "PP2",
         "personnel": [105, 106, 107, 108, 201],
         "unit_toi_secs": 200.0, "share_of_st_toi": 0.17,
         "team_st_toi": 1200.0, "team_st_gp": 10,
         "model_version": "st_deployment_v1"},
        {"team": "MTL", "season": 2025, "unit_type": "PP1",
         "personnel": [301, 302, 303, 304, 305],
         "unit_toi_secs": 500.0, "share_of_st_toi": 0.5,
         "team_st_toi": 1000.0, "team_st_gp": 10,
         "model_version": "st_deployment_v1"},
    ])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_schema_and_version() -> None:
    df = compute_pp_coordinator(
        pbp_df       = _pbp([]),
        shots_df     = _shots([]),
        st_df        = _st_deployment(),
        position_map = {200: "D", 201: "D", 301: "C"},
        name_lookup  = {200: "Top D"},
        season       = 2025,
        team_lookup  = {},
    )
    assert not df.is_empty()
    assert set(df.columns) == set(PP_COORDINATOR_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()
    assert (df["season"] == 2025).all()


def test_pp_shot_filter_excludes_even_strength() -> None:
    # 1 PP shot (5v4) + 1 EV (5v5) + 1 SH-against (4v5 — shooter shorthanded)
    shots = _shots([
        {"home_skaters": 5, "away_skaters": 4, "shooting_team": "TOR", "x_goal": 0.30},
        {"home_skaters": 5, "away_skaters": 5, "shooting_team": "TOR", "x_goal": 0.10},
        {"home_skaters": 4, "away_skaters": 5, "shooting_team": "TOR", "x_goal": 0.05},
    ])
    out = _pp_shot_stats(shots)
    assert len(out) == 1
    r = out.row(0, named=True)
    assert r["team"] == "TOR"
    assert r["pp_shots"] == 1
    assert r["pp_xg_total"] == pytest.approx(0.30)


def test_pp_xg_per_shot_and_xg_per_60() -> None:
    shots = _shots([
        {"home_skaters": 5, "away_skaters": 4, "shooting_team": "TOR", "x_goal": 0.10},
        {"home_skaters": 5, "away_skaters": 4, "shooting_team": "TOR", "x_goal": 0.20},
    ])
    df = compute_pp_coordinator(
        pbp_df       = _pbp([]),
        shots_df     = shots,
        st_df        = _st_deployment(),
        position_map = {200: "D"},
        name_lookup  = {200: "Top D"},
        season       = 2025,
        team_lookup  = {},
    )
    tor = df.filter(pl.col("team") == "TOR").row(0, named=True)
    assert tor["pp_shots"] == 2
    assert tor["pp_xg_total"] == pytest.approx(0.30)
    assert tor["pp_xg_per_shot"] == pytest.approx(0.15)
    # PP TOI = 1200 sec; xG total = 0.30 → xG/60 = 0.30 * 3600 / 1200 = 0.9
    assert tor["pp_xg_per_60"] == pytest.approx(0.30 * 3600 / 1200)


def test_pp1_qb_identifies_only_defenseman_on_unit() -> None:
    df = compute_pp_coordinator(
        pbp_df       = _pbp([]),
        shots_df     = _shots([]),
        st_df        = _st_deployment(),
        position_map = {200: "D", 201: "D", 301: "C", 302: "C", 303: "C", 304: "C", 305: "C"},
        name_lookup  = {200: "Top D", 201: "Backup D"},
        season       = 2025,
        team_lookup  = {},
    )
    tor = df.filter(pl.col("team") == "TOR").row(0, named=True)
    assert tor["pp1_qb_id"] == 200
    assert tor["pp1_qb_name"] == "Top D"
    # PP1 unit TOI 600 / team PP TOI 1200 = 0.5
    assert tor["pp1_qb_share"] == pytest.approx(0.5)
    mtl = df.filter(pl.col("team") == "MTL").row(0, named=True)
    assert mtl["pp1_qb_id"] == -1   # no D on PP1
    assert mtl["pp1_qb_name"] == ""


def test_pp_carry_pct_when_carry_data_present() -> None:
    pbp = _pbp([
        {"event_type": "zone_entry", "strength": "pp", "carry_in": True,  "entering_team_id": 10},
        {"event_type": "zone_entry", "strength": "pp", "carry_in": False, "entering_team_id": 10},
        {"event_type": "zone_entry", "strength": "pp", "carry_in": True,  "entering_team_id": 10},
        # MTL: 1 of 2 controlled = 50%
        {"event_type": "zone_entry", "strength": "pp", "carry_in": True,  "entering_team_id": 8},
        {"event_type": "zone_entry", "strength": "pp", "carry_in": False, "entering_team_id": 8},
        # EV entries shouldn't count
        {"event_type": "zone_entry", "strength": "ev", "carry_in": False, "entering_team_id": 10},
    ])
    df = compute_pp_coordinator(
        pbp_df       = pbp,
        shots_df     = _shots([]),
        st_df        = _st_deployment(),
        position_map = {200: "D"},
        name_lookup  = {200: "Top D"},
        season       = 2025,
        team_lookup  = {10: "TOR", 8: "MTL"},
    )
    tor = df.filter(pl.col("team") == "TOR").row(0, named=True)
    mtl = df.filter(pl.col("team") == "MTL").row(0, named=True)
    assert tor["pp_carry_pct"] == pytest.approx(2 / 3, rel=1e-3)
    assert mtl["pp_carry_pct"] == pytest.approx(0.5, rel=1e-3)


def test_pp_carry_pct_nan_when_no_carry_data() -> None:
    df = compute_pp_coordinator(
        pbp_df       = _pbp([]),
        shots_df     = _shots([]),
        st_df        = _st_deployment(),
        position_map = {200: "D"},
        name_lookup  = {200: "Top D"},
        season       = 2025,
        team_lookup  = {},
    )
    for r in df.iter_rows(named=True):
        assert math.isnan(r["pp_carry_pct"])


def test_empty_st_deployment_warns_and_returns_empty() -> None:
    empty_st = pl.DataFrame(schema={
        "team": pl.Utf8, "season": pl.Int64, "unit_type": pl.Utf8,
        "personnel": pl.List(pl.Int64), "unit_toi_secs": pl.Float64,
        "share_of_st_toi": pl.Float64, "team_st_toi": pl.Float64,
        "team_st_gp": pl.Int64, "model_version": pl.Utf8,
    })
    with pytest.warns(DataMissingWarning):
        df = compute_pp_coordinator(
            pbp_df       = _pbp([]),
            shots_df     = _shots([]),
            st_df        = empty_st,
            position_map = {},
            name_lookup  = {},
            season       = 2025,
            team_lookup  = {},
        )
    assert df.is_empty()
    assert set(df.columns) == set(PP_COORDINATOR_SCHEMA.keys())


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    df = compute_pp_coordinator(
        pbp_df       = _pbp([]),
        shots_df     = _shots([]),
        st_df        = _st_deployment(),
        position_map = {200: "D"},
        name_lookup  = {200: "Top D"},
        season       = 2025,
        team_lookup  = {},
    )
    write_pp_coordinator(df, tmp_path / "pp_coordinator", season=2025)
    rt = read_pp_coordinator(tmp_path, season=2025)
    assert rt is not None and len(rt) == len(df)
    assert set(rt.columns) == set(PP_COORDINATOR_SCHEMA.keys())
