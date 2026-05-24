"""Unit tests for models/coaching_style.py — Feature 4.11."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from models.coaching_style import (
    COACHING_STYLE_SCHEMA,
    DataMissingWarning,
    MODEL_VERSION,
    STYLE_DIMENSIONS,
    _line_match_score,
    _percentile_rank,
    compute_coaching_style,
    read_coaching_style,
    write_coaching_style,
)


_PBP_BASE = {
    "event_type":           "shot",
    "event_owner_team_id":  None,
    "home_team_id":         10,
    "away_team_id":         8,
    "strength":             "ev",
    "zone_code":            "O",
    "shot_result":          "on_goal",
    "shot_distance":        None,
    "x_coord":              None,
    "carry_in":             None,
    "entering_team_id":     None,
    "turnover_player_id":   None,
    "hitter_id":            None,
    "game_id":              1,
}


def _pbp(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame([{**_PBP_BASE, **r} for r in rows])


def _lines() -> pl.DataFrame:
    return pl.DataFrame([
        {"team": "TOR", "line_type": "F", "line_rank": 1, "share_of_team_toi": 0.40, "season": 2025},
        {"team": "TOR", "line_type": "F", "line_rank": 2, "share_of_team_toi": 0.25, "season": 2025},
        {"team": "TOR", "line_type": "F", "line_rank": 3, "share_of_team_toi": 0.20, "season": 2025},
        {"team": "TOR", "line_type": "F", "line_rank": 4, "share_of_team_toi": 0.15, "season": 2025},
    ])


def test_schema_and_version() -> None:
    pbp = _pbp([
        {"event_type": "takeaway", "event_owner_team_id": 10, "zone_code": "O"},
        {"event_type": "shot",     "event_owner_team_id": 10, "shot_result": "on_goal", "x_coord": 90},
        {"event_type": "hit",      "event_owner_team_id": 10},
        {"event_type": "giveaway", "event_owner_team_id": 10, "zone_code": "D"},
    ])
    df = compute_coaching_style(
        pbp_df         = pbp,
        lines_df       = _lines(),
        pp_coordinator = pl.DataFrame(),
        team_lookup    = {10: "TOR", 8: "MTL"},
        season         = 2025,
    )
    assert not df.is_empty()
    assert set(df.columns) == set(COACHING_STYLE_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()


def test_all_rank_columns_bounded_01() -> None:
    pbp = _pbp([
        {"event_type": "takeaway", "event_owner_team_id": 10, "zone_code": "O"},
        {"event_type": "hit",      "event_owner_team_id": 10},
        {"event_type": "hit",      "event_owner_team_id": 8},
        {"event_type": "shot",     "event_owner_team_id": 10, "x_coord": 95},
        {"event_type": "shot",     "event_owner_team_id": 8,  "x_coord": 30},
    ])
    df = compute_coaching_style(
        pbp_df=pbp, lines_df=_lines(), pp_coordinator=pl.DataFrame(),
        team_lookup={10: "TOR", 8: "MTL"}, season=2025,
    )
    for dim in STYLE_DIMENSIONS:
        col = f"{dim}_rank"
        for v in df[col].to_list():
            if v is not None and v == v:  # skip NaN
                assert 0.0 <= v <= 1.0, f"{col}={v} out of [0,1]"


def test_nz_tendency_nan_when_no_carry_data() -> None:
    pbp = _pbp([
        {"event_type": "zone_entry", "entering_team_id": 10, "carry_in": None},
        {"event_type": "shot",       "event_owner_team_id": 10, "x_coord": 90},
    ])
    df = compute_coaching_style(
        pbp_df=pbp, lines_df=_lines(), pp_coordinator=pl.DataFrame(),
        team_lookup={10: "TOR", 8: "MTL"}, season=2025,
    )
    tor = df.filter(pl.col("team") == "TOR").row(0, named=True)
    assert tor["nz_tendency_raw"] != tor["nz_tendency_raw"]  # NaN


def test_line_match_score_balanced_vs_concentrated() -> None:
    balanced = _line_match_score(pl.DataFrame([
        {"team": "A", "line_type": "F", "line_rank": 1, "share_of_team_toi": 0.25},
        {"team": "A", "line_type": "F", "line_rank": 2, "share_of_team_toi": 0.25},
        {"team": "A", "line_type": "F", "line_rank": 3, "share_of_team_toi": 0.25},
        {"team": "A", "line_type": "F", "line_rank": 4, "share_of_team_toi": 0.25},
    ]))
    concentrated = _line_match_score(pl.DataFrame([
        {"team": "B", "line_type": "F", "line_rank": 1, "share_of_team_toi": 0.60},
        {"team": "B", "line_type": "F", "line_rank": 2, "share_of_team_toi": 0.30},
        {"team": "B", "line_type": "F", "line_rank": 3, "share_of_team_toi": 0.09},
        {"team": "B", "line_type": "F", "line_rank": 4, "share_of_team_toi": 0.01},
    ]))
    assert balanced["A"] < concentrated["B"]
    # Perfectly balanced → near 0
    assert balanced["A"] == pytest.approx(0.0, abs=0.01)


def test_percentile_rank_simple() -> None:
    ranks = _percentile_rank([1.0, 2.0, 3.0])
    assert ranks == [0.0, 0.5, 1.0]


def test_percentile_rank_with_nan() -> None:
    ranks = _percentile_rank([1.0, float("nan"), 3.0])
    assert ranks[0] == 0.0
    assert math.isnan(ranks[1])
    assert ranks[2] == 1.0


def test_empty_pbp_warns_and_empty() -> None:
    with pytest.warns(DataMissingWarning):
        df = compute_coaching_style(
            pbp_df=pl.DataFrame(schema={
                "event_type": pl.Utf8, "event_owner_team_id": pl.Int64,
                "home_team_id": pl.Int64, "away_team_id": pl.Int64,
                "strength": pl.Utf8, "zone_code": pl.Utf8,
                "shot_result": pl.Utf8, "shot_distance": pl.Float64,
                "x_coord": pl.Float64, "carry_in": pl.Boolean,
                "entering_team_id": pl.Int64, "turnover_player_id": pl.Int64,
                "hitter_id": pl.Int64, "game_id": pl.Int64,
            }),
            lines_df=_lines(),
            pp_coordinator=pl.DataFrame(),
            team_lookup={10: "TOR"},
            season=2025,
        )
    assert df.is_empty()


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    pbp = _pbp([
        {"event_type": "shot", "event_owner_team_id": 10, "x_coord": 90},
        {"event_type": "hit",  "event_owner_team_id": 10},
    ])
    df = compute_coaching_style(
        pbp_df=pbp, lines_df=_lines(), pp_coordinator=pl.DataFrame(),
        team_lookup={10: "TOR", 8: "MTL"}, season=2025,
    )
    write_coaching_style(df, tmp_path / "coaching_style", season=2025)
    rt = read_coaching_style(tmp_path, season=2025)
    assert rt is not None and len(rt) == len(df)
