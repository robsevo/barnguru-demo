"""Tests for Feature 2.3 — Matchup Efficiency Matrix.

All tests use synthetic data; no disk I/O or live API calls required.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from models.matchup_model import (
    CO_TOI_SCHEMA,
    LEAGUE_AVG_XGF_PCT,
    MATCHUP_PRED_SCHEMA,
    MIN_PAIR_GP,
    MIN_SHARED_TOI_SECS,
    QOT_QOC_SCHEMA,
    MatchupModel,
    build_cotoi_from_stints,
    compute_qot_qoc,
    read_matchup_preds,
    read_qot_qoc,
    write_matchup_preds,
    write_qot_qoc,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_stints(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal stints DataFrame from a list of dicts."""
    schema = {
        "game_id":          pl.Utf8,
        "situation":        pl.Utf8,
        "duration_secs":    pl.Float64,
        "home_player_ids":  pl.List(pl.Int64),
        "away_player_ids":  pl.List(pl.Int64),
        "xgf_home":         pl.Float64,
        "xgf_away":         pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def _make_rapm(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "player_id":    pl.Int64,
        "player_name":  pl.Utf8,
        "team":         pl.Utf8,
        "season":       pl.Int64,
        "gp":           pl.Int64,
        "toi_ev":       pl.Float64,
        "rapm_ev_off":  pl.Float64,
        "rapm_ev_def":  pl.Float64,
        "rapm_pp":      pl.Float64,
        "rapm_pk":      pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def _make_player_stats(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "player_id":         pl.Int64,
        "season":            pl.Int64,
        "zone_start_oz_pct": pl.Float64,
        "zone_start_dz_pct": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def _make_vs_team(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "player_id":        pl.Int64,
        "opponent_team_id": pl.Int64,
        "season":           pl.Int64,
        "shots_for":        pl.Int64,
        "goals_for":        pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SEASON = 20232024

@pytest.fixture()
def simple_stints() -> pl.DataFrame:
    """One game, two EV stints.

    Stint 1 (120s):  home=[1,2,3]  away=[6,7,8]  xgf_home=0.30  xgf_away=0.20
    Stint 2 (60s):   home=[1,2,4]  away=[6,7,9]  xgf_home=0.12  xgf_away=0.08
    """
    return _make_stints([
        {
            "game_id": "G1", "situation": "EV", "duration_secs": 120.0,
            "home_player_ids": [1, 2, 3], "away_player_ids": [6, 7, 8],
            "xgf_home": 0.30, "xgf_away": 0.20,
        },
        {
            "game_id": "G1", "situation": "EV", "duration_secs": 60.0,
            "home_player_ids": [1, 2, 4], "away_player_ids": [6, 7, 9],
            "xgf_home": 0.12, "xgf_away": 0.08,
        },
    ])


@pytest.fixture()
def simple_rapm() -> pl.DataFrame:
    # player 1..4 home; 6..9 away
    return _make_rapm([
        {"player_id": 1, "player_name": "P1", "team": "TOR", "season": SEASON, "gp": 60, "toi_ev": 900.0,  "rapm_ev_off": 0.50, "rapm_ev_def": 0.20, "rapm_pp": 0.1, "rapm_pk": 0.0},
        {"player_id": 2, "player_name": "P2", "team": "TOR", "season": SEASON, "gp": 58, "toi_ev": 800.0,  "rapm_ev_off": 1.00, "rapm_ev_def": 0.10, "rapm_pp": 0.2, "rapm_pk": 0.0},
        {"player_id": 3, "player_name": "P3", "team": "TOR", "season": SEASON, "gp": 55, "toi_ev": 750.0,  "rapm_ev_off":-0.50, "rapm_ev_def":-0.30, "rapm_pp": 0.0, "rapm_pk": 0.0},
        {"player_id": 4, "player_name": "P4", "team": "TOR", "season": SEASON, "gp": 50, "toi_ev": 700.0,  "rapm_ev_off": 0.20, "rapm_ev_def": 0.10, "rapm_pp": 0.0, "rapm_pk": 0.0},
        {"player_id": 6, "player_name": "P6", "team": "BOS", "season": SEASON, "gp": 62, "toi_ev": 920.0,  "rapm_ev_off": 0.30, "rapm_ev_def":-0.10, "rapm_pp": 0.1, "rapm_pk": 0.0},
        {"player_id": 7, "player_name": "P7", "team": "BOS", "season": SEASON, "gp": 60, "toi_ev": 880.0,  "rapm_ev_off":-0.20, "rapm_ev_def": 0.05, "rapm_pp": 0.0, "rapm_pk": 0.0},
        {"player_id": 8, "player_name": "P8", "team": "BOS", "season": SEASON, "gp": 55, "toi_ev": 800.0,  "rapm_ev_off": 0.10, "rapm_ev_def": 0.00, "rapm_pp": 0.0, "rapm_pk": 0.0},
        {"player_id": 9, "player_name": "P9", "team": "BOS", "season": SEASON, "gp": 50, "toi_ev": 750.0,  "rapm_ev_off": 0.40, "rapm_ev_def":-0.20, "rapm_pp": 0.0, "rapm_pk": 0.0},
    ])


# ---------------------------------------------------------------------------
# build_cotoi_from_stints — basic sanity
# ---------------------------------------------------------------------------

def test_cotoi_empty_stints_returns_empty_dfs():
    """No EV stints → empty linemate and opponent DataFrames."""
    stints = _make_stints([{
        "game_id": "G1", "situation": "PP", "duration_secs": 60.0,
        "home_player_ids": [1, 2], "away_player_ids": [6, 7],
        "xgf_home": 0.10, "xgf_away": 0.02,
    }])
    import warnings
    from models.matchup_model import DataMissingWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataMissingWarning)
        lm, opp = build_cotoi_from_stints(stints, SEASON)
    assert lm.is_empty()
    assert opp.is_empty()


def test_cotoi_missing_column_raises():
    stints = pl.DataFrame({"game_id": ["G1"], "situation": ["EV"]})
    with pytest.raises(ValueError, match="missing columns"):
        build_cotoi_from_stints(stints, SEASON)


def test_cotoi_linemate_schema(simple_stints):
    lm, _ = build_cotoi_from_stints(simple_stints, SEASON)
    for col, dtype in CO_TOI_SCHEMA.items():
        assert col in lm.columns, f"missing column: {col}"


def test_cotoi_opponent_schema(simple_stints):
    _, opp = build_cotoi_from_stints(simple_stints, SEASON)
    for col, dtype in CO_TOI_SCHEMA.items():
        assert col in opp.columns, f"missing column: {col}"


def test_cotoi_linemate_co_toi_values(simple_stints):
    """Players 1 and 2 share both stints: co_toi = 120 + 60 = 180s."""
    lm, _ = build_cotoi_from_stints(simple_stints, SEASON)
    row = lm.filter(
        (pl.col("player_a_id") == 1) & (pl.col("player_b_id") == 2)
    )
    assert len(row) == 1
    assert row["co_toi_ev"][0] == pytest.approx(180.0)
    assert row["games_together"][0] == 1


def test_cotoi_linemate_partial_co_toi(simple_stints):
    """Player 1 and 3 only share stint 1: co_toi = 120s."""
    lm, _ = build_cotoi_from_stints(simple_stints, SEASON)
    row = lm.filter(
        (pl.col("player_a_id") == 1) & (pl.col("player_b_id") == 3)
    )
    assert len(row) == 1
    assert row["co_toi_ev"][0] == pytest.approx(120.0)


def test_cotoi_canonical_ordering(simple_stints):
    """Linemate pairs must have player_a_id < player_b_id."""
    lm, _ = build_cotoi_from_stints(simple_stints, SEASON)
    assert (lm["player_a_id"] < lm["player_b_id"]).all()


def test_cotoi_opponent_xgf_pct(simple_stints):
    """Player 1 (home) vs Player 6 (away) share both stints.

    xgf_for = 0.30 + 0.12 = 0.42  (home xG)
    xgf_against = 0.20 + 0.08 = 0.28  (away xG)
    xgf_pct = 0.42 / (0.42 + 0.28) = 0.60
    """
    _, opp = build_cotoi_from_stints(simple_stints, SEASON)
    row = opp.filter(
        (pl.col("player_a_id") == 1) & (pl.col("player_b_id") == 6)
    )
    assert len(row) == 1
    assert row["xgf_pct"][0] == pytest.approx(0.60, abs=1e-6)


def test_cotoi_opponent_both_orientations(simple_stints):
    """Opponent df must contain both (1 vs 6) and (6 vs 1) orientations."""
    _, opp = build_cotoi_from_stints(simple_stints, SEASON)
    has_1_vs_6 = len(opp.filter((pl.col("player_a_id") == 1) & (pl.col("player_b_id") == 6))) == 1
    has_6_vs_1 = len(opp.filter((pl.col("player_a_id") == 6) & (pl.col("player_b_id") == 1))) == 1
    assert has_1_vs_6
    assert has_6_vs_1


def test_cotoi_opponent_inverted_xgf(simple_stints):
    """Row (6 vs 1) has xgf_pct = 1 - xgf_pct(1 vs 6)."""
    _, opp = build_cotoi_from_stints(simple_stints, SEASON)
    pct_1_6 = opp.filter((pl.col("player_a_id") == 1) & (pl.col("player_b_id") == 6))["xgf_pct"][0]
    pct_6_1 = opp.filter((pl.col("player_a_id") == 6) & (pl.col("player_b_id") == 1))["xgf_pct"][0]
    assert pct_1_6 + pct_6_1 == pytest.approx(1.0, abs=1e-6)


def test_cotoi_traded_player_no_duplicate_pairs():
    """Regression: player traded mid-season appears as both home AND away.

    Player 1 starts on TOR (home team in G1) then is traded to BOS
    (away team in G2).  In both games they face player 6 on the other team.
    The opposition pair (1 vs 6) must appear exactly ONCE in opp_df with
    co_toi and xGF aggregated across both stints — not as two rows.
    """
    stints = _make_stints([
        # G1: player 1 on home team vs player 6 on away team
        {
            "game_id": "G1", "situation": "EV", "duration_secs": 60.0,
            "home_player_ids": [1, 2], "away_player_ids": [6, 7],
            "xgf_home": 0.30, "xgf_away": 0.20,
        },
        # G2: player 1 is now traded, on away team vs player 6 on home team
        {
            "game_id": "G2", "situation": "EV", "duration_secs": 60.0,
            "home_player_ids": [6, 7], "away_player_ids": [1, 2],
            "xgf_home": 0.10, "xgf_away": 0.15,
        },
    ])
    _, opp = build_cotoi_from_stints(stints, SEASON)

    rows_1_vs_6 = opp.filter(
        (pl.col("player_a_id") == 1) & (pl.col("player_b_id") == 6)
    )
    # Must be exactly one row — no duplicates from the trade
    assert len(rows_1_vs_6) == 1

    # co_toi_ev should be summed across both stints (60 + 60 = 120s)
    assert rows_1_vs_6["co_toi_ev"][0] == pytest.approx(120.0)

    # xGF for player 1's team:
    #   G1: player 1 is home → xgf_for=0.30, xgf_against=0.20
    #   G2: player 1 is away → xgf_for=0.15, xgf_against=0.10
    # Total: xgf_for=0.45, xgf_against=0.30 → xgf_pct = 0.45/0.75 = 0.60
    assert rows_1_vs_6["xgf_pct"][0] == pytest.approx(0.60, abs=1e-6)


def test_cotoi_traded_player_predict_no_crash():
    """Regression: predict_season must not crash when a traded player creates
    duplicate (player_a_id, player_b_id) pairs before the dedup fix."""
    stints = _make_stints([
        {"game_id": f"G{g}", "situation": "EV", "duration_secs": 60.0,
         "home_player_ids": [1, 2], "away_player_ids": [6, 7],
         "xgf_home": 0.25, "xgf_away": 0.18}
        for g in range(8)
    ] + [
        {"game_id": f"H{g}", "situation": "EV", "duration_secs": 60.0,
         "home_player_ids": [6, 7], "away_player_ids": [1, 2],
         "xgf_home": 0.20, "xgf_away": 0.22}
        for g in range(8)
    ])
    rapm = _make_rapm([
        {"player_id": pid, "player_name": f"P{pid}", "team": t, "season": SEASON,
         "gp": 50, "toi_ev": 700.0, "rapm_ev_off": 0.3, "rapm_ev_def": 0.1,
         "rapm_pp": 0.0, "rapm_pk": 0.0}
        for pid, t in [(1, "TOR"), (2, "TOR"), (6, "BOS"), (7, "BOS")]
    ])

    model = MatchupModel()
    model.fit(
        stints_list=[stints],
        rapm_list=[rapm],
        vs_team_list=[None],
        player_stats_list=[None],
        seasons=[SEASON],
    )
    # This must not raise a Series length mismatch error
    preds = model.predict_season(
        stints_df=stints,
        rapm_df=rapm,
        vs_team_df=None,
        player_stats_df=None,
        season=SEASON,
    )
    assert not preds.is_empty()
    # Each (player_a_id, player_b_id) pair appears exactly once
    n_unique = preds.select(["player_a_id", "player_b_id"]).unique().height
    assert n_unique == preds.height


def test_cotoi_min_shared_toi_filter():
    """Pairs below MIN_SHARED_TOI_SECS must not appear in output."""
    stints = _make_stints([{
        "game_id": "G1", "situation": "EV",
        "duration_secs": MIN_SHARED_TOI_SECS - 1.0,
        "home_player_ids": [1, 2], "away_player_ids": [6],
        "xgf_home": 0.05, "xgf_away": 0.03,
    }])
    lm, opp = build_cotoi_from_stints(stints, SEASON)
    assert lm.is_empty()
    assert opp.is_empty()


# ---------------------------------------------------------------------------
# compute_qot_qoc
# ---------------------------------------------------------------------------

def test_qot_qoc_schema(simple_stints, simple_rapm):
    result = compute_qot_qoc(simple_stints, simple_rapm, SEASON)
    for col in QOT_QOC_SCHEMA:
        assert col in result.columns, f"missing column: {col}"


def test_qot_qoc_rapm_missing_columns(simple_stints):
    bad_rapm = pl.DataFrame({"player_id": [1], "player_name": ["P1"]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_qot_qoc(simple_stints, bad_rapm, SEASON)


def test_qot_value_numerical(simple_stints, simple_rapm):
    """Player 1's QoT should be weighted avg of linemates P2, P3, P4.

    Linemate co-TOI weights:
        P2 (rapm_composite=1.10): co_toi = 180s
        P3 (rapm_composite=-0.80): co_toi = 120s
        P4 (rapm_composite=0.30): co_toi = 60s

    QoT_P1 = (180×1.10 + 120×(-0.80) + 60×0.30) / (180+120+60)
           = (198 - 96 + 18) / 360
           = 120 / 360
           = 0.3333...
    """
    result = compute_qot_qoc(simple_stints, simple_rapm, SEASON)
    p1 = result.filter(pl.col("player_id") == 1)
    assert len(p1) == 1
    assert p1["qot"][0] == pytest.approx(120.0 / 360.0, abs=1e-4)


def test_qoc_value_numerical(simple_stints, simple_rapm):
    """Player 1's QoC: opponents P6, P7, P8, P9.

    Opposition co-TOI weights:
        P6 (0.20): 180s   P7 (-0.15): 180s
        P8 (0.10): 120s   P9 (0.20): 60s

    QoC_P1 = (180×0.20 + 180×(-0.15) + 120×0.10 + 60×0.20) / (180+180+120+60)
           = (36 - 27 + 12 + 12) / 540
           = 33 / 540 ≈ 0.06111
    """
    result = compute_qot_qoc(simple_stints, simple_rapm, SEASON)
    p1 = result.filter(pl.col("player_id") == 1)
    assert p1["qoc"][0] == pytest.approx(33.0 / 540.0, abs=1e-4)


def test_qot_qoc_no_nulls(simple_stints, simple_rapm):
    result = compute_qot_qoc(simple_stints, simple_rapm, SEASON)
    for col in ("qot", "qoc", "qot_toi", "qoc_toi"):
        assert result[col].is_not_null().all(), f"{col} has nulls"


def test_qot_qoc_sorted_descending(simple_stints, simple_rapm):
    result = compute_qot_qoc(simple_stints, simple_rapm, SEASON)
    qot_vals = result["qot"].to_list()
    assert qot_vals == sorted(qot_vals, reverse=True)


def test_qot_qoc_season_filter():
    """compute_qot_qoc only uses RAPM rows matching the requested season."""
    stints = _make_stints([{
        "game_id": "G1", "situation": "EV", "duration_secs": 60.0,
        "home_player_ids": [1, 2], "away_player_ids": [6],
        "xgf_home": 0.10, "xgf_away": 0.08,
    }])
    # Provide RAPM for two seasons; only SEASON data should be used
    rapm = _make_rapm([
        {"player_id": 1, "player_name": "P1", "team": "TOR", "season": SEASON,
         "gp": 60, "toi_ev": 900.0, "rapm_ev_off": 1.0, "rapm_ev_def": 0.0,
         "rapm_pp": 0.0, "rapm_pk": 0.0},
        {"player_id": 2, "player_name": "P2", "team": "TOR", "season": SEASON,
         "gp": 58, "toi_ev": 800.0, "rapm_ev_off": 0.5, "rapm_ev_def": 0.0,
         "rapm_pp": 0.0, "rapm_pk": 0.0},
        {"player_id": 6, "player_name": "P6", "team": "BOS", "season": SEASON,
         "gp": 60, "toi_ev": 900.0, "rapm_ev_off": 0.3, "rapm_ev_def": 0.0,
         "rapm_pp": 0.0, "rapm_pk": 0.0},
        # Previous season — should be ignored
        {"player_id": 1, "player_name": "P1_old", "team": "TOR", "season": 20222023,
         "gp": 60, "toi_ev": 900.0, "rapm_ev_off": 99.0, "rapm_ev_def": 0.0,
         "rapm_pp": 0.0, "rapm_pk": 0.0},
    ])
    result = compute_qot_qoc(stints, rapm, SEASON)
    p1 = result.filter(pl.col("player_id") == 1)
    # QoT for P1 should be based on rapm_composite of P2 (=0.5), not P1_old (=99.0)
    assert p1["qot"][0] == pytest.approx(0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# MatchupModel — fit / predict
# ---------------------------------------------------------------------------

def _make_multi_game_stints(n_games: int = 15) -> pl.DataFrame:
    """Create enough games to exceed MIN_PAIR_GP for pair (1 vs 6)."""
    rows = []
    for g in range(n_games):
        rows.append({
            "game_id": f"G{g}", "situation": "EV", "duration_secs": 90.0,
            "home_player_ids": [1, 2, 3], "away_player_ids": [6, 7, 8],
            "xgf_home": 0.25, "xgf_away": 0.15,
        })
    return _make_stints(rows)


def _make_rapm_for_fit() -> pl.DataFrame:
    rows = []
    for season in (20212022, 20222023, 20232024):
        for pid, name, team, off, deff in [
            (1, "P1", "TOR", 0.5, 0.2),
            (2, "P2", "TOR", 1.0, 0.1),
            (3, "P3", "TOR", -0.5, -0.3),
            (6, "P6", "BOS", 0.3, -0.1),
            (7, "P7", "BOS", -0.2, 0.05),
            (8, "P8", "BOS", 0.1, 0.0),
        ]:
            rows.append({
                "player_id": pid, "player_name": name, "team": team,
                "season": season, "gp": 60, "toi_ev": 800.0,
                "rapm_ev_off": off, "rapm_ev_def": deff,
                "rapm_pp": 0.0, "rapm_pk": 0.0,
            })
    return _make_rapm(rows)


def test_matchup_model_fit_returns_metrics():
    stints = _make_multi_game_stints(20)
    rapm   = _make_rapm_for_fit()

    model = MatchupModel()
    metrics = model.fit(
        stints_list=[stints, stints],
        rapm_list=[
            rapm.filter(pl.col("season") == 20212022),
            rapm.filter(pl.col("season") == 20222023),
        ],
        vs_team_list=[None, None],
        player_stats_list=[None, None],
        seasons=[20212022, 20222023],
    )
    assert "rmse" in metrics
    assert "n_pairs" in metrics
    assert metrics["rmse"] >= 0.0
    assert metrics["n_pairs"] > 0


def test_matchup_model_predict_schema():
    stints = _make_multi_game_stints(20)
    rapm   = _make_rapm_for_fit()

    model = MatchupModel()
    model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        vs_team_list=[None],
        player_stats_list=[None],
        seasons=[20212022],
    )
    preds = model.predict_season(
        stints_df=stints,
        rapm_df=rapm.filter(pl.col("season") == 20232024),
        vs_team_df=None,
        player_stats_df=None,
        season=20232024,
    )
    for col in MATCHUP_PRED_SCHEMA:
        assert col in preds.columns, f"missing column: {col}"


def test_matchup_model_predictions_in_range():
    """Blended xgf_pct must lie in [0, 1]."""
    stints = _make_multi_game_stints(20)
    rapm   = _make_rapm_for_fit()

    model = MatchupModel()
    model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        vs_team_list=[None],
        player_stats_list=[None],
        seasons=[20212022],
    )
    preds = model.predict_season(
        stints_df=stints,
        rapm_df=rapm.filter(pl.col("season") == 20212022),
        vs_team_df=None,
        player_stats_df=None,
        season=20212022,
    )
    xgf = preds["xgf_pct"].to_numpy()
    assert (xgf >= 0.0).all() and (xgf <= 1.0).all()


def test_matchup_model_untrained_raises():
    model = MatchupModel()
    with pytest.raises(RuntimeError, match="not trained"):
        model.predict_season(
            stints_df=_make_multi_game_stints(5),
            rapm_df=_make_rapm_for_fit().filter(pl.col("season") == 20212022),
            vs_team_df=None,
            player_stats_df=None,
            season=20212022,
        )


def test_matchup_model_save_load(tmp_path):
    stints = _make_multi_game_stints(20)
    rapm   = _make_rapm_for_fit()

    model = MatchupModel()
    model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        vs_team_list=[None],
        player_stats_list=[None],
        seasons=[20212022],
    )
    pkl = tmp_path / "matchup.pkl"
    model.save(pkl)

    loaded = MatchupModel.load(pkl)
    preds_orig   = model.predict_season(stints, rapm.filter(pl.col("season") == 20212022), None, None, 20212022)
    preds_loaded = loaded.predict_season(stints, rapm.filter(pl.col("season") == 20212022), None, None, 20212022)

    np.testing.assert_allclose(
        preds_orig["xgf_pct"].to_numpy(),
        preds_loaded["xgf_pct"].to_numpy(),
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Blending logic
# ---------------------------------------------------------------------------

def test_blending_low_gp_uses_model_only():
    """Pair with < MIN_PAIR_GP games → weight_hist = 0 → xgf_pct == model_xgf_pct."""
    stints = _make_multi_game_stints(n_games=MIN_PAIR_GP - 1)
    rapm   = _make_rapm_for_fit()

    model = MatchupModel()
    model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        vs_team_list=[None],
        player_stats_list=[None],
        seasons=[20212022],
    )
    preds = model.predict_season(
        stints_df=stints,
        rapm_df=rapm.filter(pl.col("season") == 20212022),
        vs_team_df=None,
        player_stats_df=None,
        season=20212022,
    )
    low_gp = preds.filter(pl.col("games_together") < MIN_PAIR_GP)
    if not low_gp.is_empty():
        np.testing.assert_allclose(
            low_gp["xgf_pct"].to_numpy(),
            low_gp["model_xgf_pct"].to_numpy(),
            rtol=1e-5,
        )


def test_blending_high_gp_blends_historical():
    """Pair with >= MIN_PAIR_GP games → xgf_pct blends model and historical."""
    stints = _make_multi_game_stints(n_games=MIN_PAIR_GP + 5)
    rapm   = _make_rapm_for_fit()

    model = MatchupModel()
    model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        vs_team_list=[None],
        player_stats_list=[None],
        seasons=[20212022],
    )
    preds = model.predict_season(
        stints_df=stints,
        rapm_df=rapm.filter(pl.col("season") == 20212022),
        vs_team_df=None,
        player_stats_df=None,
        season=20212022,
    )
    high_gp = preds.filter(pl.col("games_together") >= MIN_PAIR_GP)
    if not high_gp.is_empty():
        expected = (
            0.70 * high_gp["model_xgf_pct"].to_numpy()
            + 0.30 * high_gp["hist_xgf_pct"].to_numpy()
        )
        np.testing.assert_allclose(
            high_gp["xgf_pct"].to_numpy(),
            expected,
            rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def test_write_read_qot_qoc(tmp_path, simple_stints, simple_rapm):
    df = compute_qot_qoc(simple_stints, simple_rapm, SEASON)
    write_qot_qoc(df, tmp_path / "qot_qoc", SEASON)

    loaded = read_qot_qoc(tmp_path, SEASON)
    assert loaded is not None
    assert len(loaded) == len(df)


def test_read_qot_qoc_missing_returns_none(tmp_path):
    assert read_qot_qoc(tmp_path, SEASON) is None


def test_read_qot_qoc_latest(tmp_path, simple_stints, simple_rapm):
    df = compute_qot_qoc(simple_stints, simple_rapm, SEASON)
    write_qot_qoc(df, tmp_path / "qot_qoc", SEASON)
    loaded = read_qot_qoc(tmp_path)  # no season arg → latest
    assert loaded is not None


def test_write_read_matchup_preds(tmp_path):
    stints = _make_multi_game_stints(20)
    rapm   = _make_rapm_for_fit()

    model = MatchupModel()
    model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        vs_team_list=[None],
        player_stats_list=[None],
        seasons=[20212022],
    )
    preds = model.predict_season(
        stints_df=stints,
        rapm_df=rapm.filter(pl.col("season") == 20212022),
        vs_team_df=None,
        player_stats_df=None,
        season=20212022,
    )
    write_matchup_preds(preds, tmp_path / "matchup_preds", 20212022)
    loaded = read_matchup_preds(tmp_path, 20212022)
    assert loaded is not None
    assert len(loaded) == len(preds)


def test_read_matchup_preds_missing_returns_none(tmp_path):
    assert read_matchup_preds(tmp_path, SEASON) is None


# ---------------------------------------------------------------------------
# Zone starts and vs_team integration
# ---------------------------------------------------------------------------

def test_player_stats_zone_starts_used(simple_stints, simple_rapm):
    """Model runs end-to-end with player_stats providing zone start data."""
    stats = _make_player_stats([
        {"player_id": pid, "season": SEASON,
         "zone_start_oz_pct": 0.55, "zone_start_dz_pct": 0.30}
        for pid in [1, 2, 3, 4, 6, 7, 8, 9]
    ])
    stints = _make_multi_game_stints(20)
    rapm   = _make_rapm_for_fit()

    model = MatchupModel()
    model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        vs_team_list=[None],
        player_stats_list=[stats.with_columns(pl.lit(20212022).alias("season"))],
        seasons=[20212022],
    )
    preds = model.predict_season(
        stints_df=stints,
        rapm_df=rapm.filter(pl.col("season") == 20212022),
        vs_team_df=None,
        player_stats_df=stats.with_columns(pl.lit(20212022).alias("season")),
        season=20212022,
    )
    assert not preds.is_empty()


def test_vs_team_feature_used(simple_stints, simple_rapm):
    """Model runs end-to-end with vs_team historical splits provided."""
    # TOR players (1,2,3,4) vs BOS (team_id=6)
    vs_team = _make_vs_team([
        {"player_id": pid, "opponent_team_id": 6,
         "season": SEASON, "shots_for": 100, "goals_for": 12}
        for pid in [1, 2, 3, 4]
    ])
    stints = _make_multi_game_stints(20)
    rapm   = _make_rapm_for_fit()

    model = MatchupModel()
    model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        vs_team_list=[vs_team.with_columns(pl.lit(20212022).alias("season"))],
        player_stats_list=[None],
        seasons=[20212022],
    )
    preds = model.predict_season(
        stints_df=stints,
        rapm_df=rapm.filter(pl.col("season") == 20212022),
        vs_team_df=vs_team.with_columns(pl.lit(20212022).alias("season")),
        player_stats_df=None,
        season=20212022,
    )
    assert not preds.is_empty()
