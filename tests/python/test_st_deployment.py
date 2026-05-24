"""Unit tests for models/st_deployment.py — Feature 4.3.

All tests use synthetic data — no live API calls or on-disk parquet files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from models.st_deployment import (
    MIN_UNIT_SECS,
    MODEL_VERSION,
    PK_UNIT_SIZE,
    PP_UNIT_SIZE,
    ST_DEPLOYMENT_SCHEMA,
    _extract_units,
    _st_windows,
    compute_st_deployment,
    read_st_deployment,
    write_st_deployment,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic data factories
# ---------------------------------------------------------------------------


def _build_synth_inputs(
    n_games: int = 20,
    seed:    int = 11,
    pp1_share_target: float = 0.75,
):
    """Build shifts + pbp for two teams (HOME=8, AWAY=10) with explicit
    PP1/PP2 + PK1/PK2 units and `pp1_share_target` deployment bias.

    Units (per team):
        HOME PP1 = {1,2,3,4,5}     PP2 = {6,7,8,9,10}
        HOME PK1 = {11,12,13,14}   PK2 = {15,16,17,18}
    Away team players offset by 100.
    """
    rng = np.random.default_rng(seed)

    home_id = 8
    away_id = 10

    HOME_PP1 = [1, 2, 3, 4, 5]
    HOME_PP2 = [6, 7, 8, 9, 10]
    HOME_PK1 = [11, 12, 13, 14]
    HOME_PK2 = [15, 16, 17, 18]
    AWAY_PP1 = [p + 100 for p in HOME_PP1]
    AWAY_PP2 = [p + 100 for p in HOME_PP2]
    AWAY_PK1 = [p + 100 for p in HOME_PK1]
    AWAY_PK2 = [p + 100 for p in HOME_PK2]

    shifts_rows: list[dict] = []
    pbp_rows:    list[dict] = []

    # Per game: 6 PP windows total — 3 home_pp + 3 away_pp, each 90s long
    PP_WINDOW_DUR = 90.0
    PP_WINDOWS_PER_GAME = 3

    for g_idx in range(n_games):
        game_id = 2025020000 + g_idx + 1
        cur_t = 0.0

        # First event marks 5v5 EV from t=0
        pbp_rows.append({
            "game_id":             game_id,
            "period":              1,
            "time_in_period_secs": 0,
            "home_skaters":        5,
            "away_skaters":        5,
            "home_team_id":        home_id,
            "away_team_id":        away_id,
        })

        # Layout 6 windows spaced out across period 1 (cur_t < 1200)
        # Alternate home_pp, away_pp, home_pp, away_pp, home_pp, away_pp
        spacing = 1200.0 / (PP_WINDOWS_PER_GAME * 2 + 1)
        for w_idx in range(PP_WINDOWS_PER_GAME * 2):
            window_start = (w_idx + 1) * spacing
            window_end   = window_start + PP_WINDOW_DUR
            is_home_pp = (w_idx % 2 == 0)

            # Transition into PP
            pbp_rows.append({
                "game_id":             game_id,
                "period":              1,
                "time_in_period_secs": int(window_start),
                "home_skaters":        5 if is_home_pp else 4,
                "away_skaters":        4 if is_home_pp else 5,
                "home_team_id":        home_id,
                "away_team_id":        away_id,
            })
            # Transition back to EV
            pbp_rows.append({
                "game_id":             game_id,
                "period":              1,
                "time_in_period_secs": int(window_end),
                "home_skaters":        5,
                "away_skaters":        5,
                "home_team_id":        home_id,
                "away_team_id":        away_id,
            })

            # Choose which units are on for THIS window — biased toward PP1/PK1
            if is_home_pp:
                pp_unit = HOME_PP1 if rng.random() < pp1_share_target else HOME_PP2
                pk_unit = AWAY_PK1 if rng.random() < pp1_share_target else AWAY_PK2
            else:
                pp_unit = AWAY_PP1 if rng.random() < pp1_share_target else AWAY_PP2
                pk_unit = HOME_PK1 if rng.random() < pp1_share_target else HOME_PK2

            # Add shift rows for PP unit (full window)
            pp_team_id = home_id if is_home_pp else away_id
            pk_team_id = away_id if is_home_pp else home_id
            for p in pp_unit:
                shifts_rows.append({
                    "game_id":            game_id,
                    "player_id":          p,
                    "team_id":            pp_team_id,
                    "period":             1,
                    "shift_number":       w_idx,
                    "duration_secs":      PP_WINDOW_DUR,
                    "game_seconds_start": window_start,
                    "game_seconds_end":   window_end,
                })
            for p in pk_unit:
                shifts_rows.append({
                    "game_id":            game_id,
                    "player_id":          p,
                    "team_id":            pk_team_id,
                    "period":             1,
                    "shift_number":       w_idx,
                    "duration_secs":      PP_WINDOW_DUR,
                    "game_seconds_start": window_start,
                    "game_seconds_end":   window_end,
                })

    shifts_df = pl.DataFrame(shifts_rows)
    pbp_df    = pl.DataFrame(pbp_rows)
    return shifts_df, pbp_df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_st_windows_identifies_home_and_away_pp() -> None:
    pbp = pl.DataFrame({
        "period":              [1, 1, 1, 1],
        "time_in_period_secs": [0, 120, 240, 600],
        "home_skaters":        [5, 5, 5, 5],
        "away_skaters":        [5, 4, 5, 5],
        "home_team_id":        [8, 8, 8, 8],
        "away_team_id":        [10, 10, 10, 10],
    })
    windows = _st_windows(pbp)
    # One home_pp window between 120s and 240s
    assert len(windows) == 1
    t0, t1, lab = windows[0]
    assert t0 == 120.0
    assert t1 == 240.0
    assert lab == "home_pp"


def test_st_windows_skips_pulled_goalie() -> None:
    """When one side has 6 skaters (goalie pulled), the situation should
    NOT be tagged as a PP window."""
    pbp = pl.DataFrame({
        "period":              [1, 1, 1],
        "time_in_period_secs": [0, 60, 120],
        "home_skaters":        [5, 6, 5],
        "away_skaters":        [5, 5, 5],
        "home_team_id":        [8, 8, 8],
        "away_team_id":        [10, 10, 10],
    })
    windows = _st_windows(pbp)
    assert windows == []


def test_extract_units_picks_top_two_disjoint() -> None:
    snapshots = [
        (90.0, frozenset({1, 2, 3, 4, 5})),
        (90.0, frozenset({1, 2, 3, 4, 5})),
        (90.0, frozenset({1, 2, 3, 4, 5})),
        (90.0, frozenset({6, 7, 8, 9, 10})),
        (90.0, frozenset({6, 7, 8, 9, 10})),
    ]
    player_toi = {p: 270.0 if p <= 5 else 180.0 for p in range(1, 11)}
    (u1, u1_secs), (u2, u2_secs) = _extract_units(snapshots, player_toi, PP_UNIT_SIZE)
    assert u1 == frozenset({1, 2, 3, 4, 5})
    assert u2 == frozenset({6, 7, 8, 9, 10})
    assert u1_secs == 270.0
    assert u2_secs == 180.0


def test_extract_units_empty_below_threshold() -> None:
    snapshots = [(10.0, frozenset({1, 2, 3, 4, 5}))]
    player_toi = {p: 10.0 for p in range(1, 6)}
    (u1, _), (u2, _) = _extract_units(snapshots, player_toi, PP_UNIT_SIZE)
    assert u1 is None and u2 is None


def test_compute_st_deployment_recovers_units() -> None:
    shifts_df, pbp_df = _build_synth_inputs(n_games=30, seed=42, pp1_share_target=0.80)
    df = compute_st_deployment(
        shifts_df, pbp_df, season=2025,
        team_lookup={8: "MTL", 10: "TOR"},
    )
    assert not df.is_empty()
    assert set(df.columns) == set(ST_DEPLOYMENT_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()

    # MTL PP1 should be {1..5}, PP2 = {6..10}, PK1 = {11..14}, PK2 = {15..18}
    mtl = df.filter(pl.col("team") == "MTL")
    units = {r["unit_type"]: set(r["personnel"]) for r in mtl.iter_rows(named=True)}
    assert units["PP1"] == {1, 2, 3, 4, 5}
    assert units["PP2"] == {6, 7, 8, 9, 10}
    assert units["PK1"] == {11, 12, 13, 14}
    assert units["PK2"] == {15, 16, 17, 18}

    # PP1 share should exceed PP2 share
    pp1_share = mtl.filter(pl.col("unit_type") == "PP1")["share_of_st_toi"][0]
    pp2_share = mtl.filter(pl.col("unit_type") == "PP2")["share_of_st_toi"][0]
    assert pp1_share > pp2_share


def test_compute_st_deployment_empty_inputs() -> None:
    df = compute_st_deployment(
        pl.DataFrame(), pl.DataFrame(), season=2025,
    )
    assert df.is_empty()
    assert set(df.columns) == set(ST_DEPLOYMENT_SCHEMA.keys())


def test_share_lies_in_zero_one() -> None:
    shifts_df, pbp_df = _build_synth_inputs(n_games=20, seed=99, pp1_share_target=0.65)
    df = compute_st_deployment(
        shifts_df, pbp_df, season=2025,
        team_lookup={8: "MTL", 10: "TOR"},
    )
    if df.is_empty():
        pytest.skip("synth produced no usable windows")
    assert (df["share_of_st_toi"] >= 0.0).all()
    assert (df["share_of_st_toi"] <= 1.0 + 1e-9).all()


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    shifts_df, pbp_df = _build_synth_inputs(n_games=15, seed=3, pp1_share_target=0.7)
    df = compute_st_deployment(
        shifts_df, pbp_df, season=2025,
        team_lookup={8: "MTL", 10: "TOR"},
    )
    if df.is_empty():
        pytest.skip("synth produced no usable windows")
    write_st_deployment(df, tmp_path / "st_deployment", season=2025)
    rt = read_st_deployment(tmp_path, season=2025)
    assert rt is not None and len(rt) == len(df)
