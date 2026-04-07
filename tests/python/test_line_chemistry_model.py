"""Tests for Feature 2.4 — Line Chemistry Model.

All tests use synthetic data; no disk I/O or live API calls required
(save/load tests use tempfile.TemporaryDirectory()).
"""

from __future__ import annotations

import math
import tempfile
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from models.line_chemistry_model import (
    LEAGUE_AVG_XGF_PCT,
    MIN_PAIR_GP,
    MIN_SHARED_TOI_SECS,
    MODEL_VERSION,
    PAIR_CHEMISTRY_SCHEMA,
    PLAYER_CHEMISTRY_SCHEMA,
    LineChemistryModel,
    build_linemate_pairs_with_xgf,
    compute_player_chemistry,
    read_pair_chemistry,
    read_player_chemistry,
    write_pair_chemistry,
    write_player_chemistry,
)
from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_stints(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal stints DataFrame from a list of dicts."""
    schema = {
        "game_id":         pl.Utf8,
        "situation":       pl.Utf8,
        "duration_secs":   pl.Float64,
        "home_player_ids": pl.List(pl.Int64),
        "away_player_ids": pl.List(pl.Int64),
        "xgf_home":        pl.Float64,
        "xgf_away":        pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def _make_rapm(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "player_id":   pl.Int64,
        "player_name": pl.Utf8,
        "team":        pl.Utf8,
        "season":      pl.Int64,
        "gp":          pl.Int64,
        "toi_ev":      pl.Float64,
        "rapm_ev_off": pl.Float64,
        "rapm_ev_def": pl.Float64,
        "rapm_pp":     pl.Float64,
        "rapm_pk":     pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def _make_finishing(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "shooter_id":      pl.Int64,
        "shooter_name":    pl.Utf8,
        "team":            pl.Utf8,
        "season":          pl.Int64,
        "shots":           pl.Int64,
        "goals":           pl.Int64,
        "xg_sum":          pl.Float64,
        "finishing":       pl.Float64,
        "finishing_per60": pl.Float64,
        "model_version":   pl.Utf8,
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


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SEASON = 20232024


@pytest.fixture()
def simple_stints() -> pl.DataFrame:
    """One game, two EV stints.

    Stint 1 (120s): home=[1,2,3]  away=[6,7,8]  xgf_home=0.30  xgf_away=0.20
    Stint 2 (60s):  home=[1,2,4]  away=[6,7,9]  xgf_home=0.12  xgf_away=0.08
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
    return _make_rapm([
        {"player_id": 1, "player_name": "P1", "team": "TOR", "season": SEASON,
         "gp": 60, "toi_ev": 900.0,  "rapm_ev_off":  0.50, "rapm_ev_def":  0.20,
         "rapm_pp": 0.1, "rapm_pk": 0.0},
        {"player_id": 2, "player_name": "P2", "team": "TOR", "season": SEASON,
         "gp": 58, "toi_ev": 800.0,  "rapm_ev_off":  1.00, "rapm_ev_def":  0.10,
         "rapm_pp": 0.2, "rapm_pk": 0.0},
        {"player_id": 3, "player_name": "P3", "team": "TOR", "season": SEASON,
         "gp": 55, "toi_ev": 750.0,  "rapm_ev_off": -0.50, "rapm_ev_def": -0.30,
         "rapm_pp": 0.0, "rapm_pk": 0.0},
        {"player_id": 4, "player_name": "P4", "team": "TOR", "season": SEASON,
         "gp": 50, "toi_ev": 700.0,  "rapm_ev_off":  0.20, "rapm_ev_def":  0.10,
         "rapm_pp": 0.0, "rapm_pk": 0.0},
        {"player_id": 6, "player_name": "P6", "team": "BOS", "season": SEASON,
         "gp": 62, "toi_ev": 920.0,  "rapm_ev_off":  0.30, "rapm_ev_def": -0.10,
         "rapm_pp": 0.1, "rapm_pk": 0.0},
        {"player_id": 7, "player_name": "P7", "team": "BOS", "season": SEASON,
         "gp": 60, "toi_ev": 880.0,  "rapm_ev_off": -0.20, "rapm_ev_def":  0.05,
         "rapm_pp": 0.0, "rapm_pk": 0.0},
        {"player_id": 8, "player_name": "P8", "team": "BOS", "season": SEASON,
         "gp": 55, "toi_ev": 800.0,  "rapm_ev_off":  0.10, "rapm_ev_def":  0.00,
         "rapm_pp": 0.0, "rapm_pk": 0.0},
        {"player_id": 9, "player_name": "P9", "team": "BOS", "season": SEASON,
         "gp": 50, "toi_ev": 750.0,  "rapm_ev_off":  0.40, "rapm_ev_def": -0.20,
         "rapm_pp": 0.0, "rapm_pk": 0.0},
    ])


# ---------------------------------------------------------------------------
# Helper: stints with many games (for model fit/predict tests)
# ---------------------------------------------------------------------------

def _make_multi_game_stints(n_games: int = 15) -> pl.DataFrame:
    """Create enough games to exceed MIN_PAIR_GP for pair (1, 2)."""
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
            (1, "P1", "TOR",  0.5,  0.2),
            (2, "P2", "TOR",  1.0,  0.1),
            (3, "P3", "TOR", -0.5, -0.3),
            (6, "P6", "BOS",  0.3, -0.1),
            (7, "P7", "BOS", -0.2,  0.05),
            (8, "P8", "BOS",  0.1,  0.0),
        ]:
            rows.append({
                "player_id": pid, "player_name": name, "team": team,
                "season": season, "gp": 60, "toi_ev": 800.0,
                "rapm_ev_off": off, "rapm_ev_def": deff,
                "rapm_pp": 0.0, "rapm_pk": 0.0,
            })
    return _make_rapm(rows)


# ---------------------------------------------------------------------------
# 1. test_build_linemate_pairs_schema
# ---------------------------------------------------------------------------

def test_build_linemate_pairs_schema(simple_stints):
    """Output has the expected columns and integer season label."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataMissingWarning)
        lm = build_linemate_pairs_with_xgf(simple_stints, SEASON)

    expected_cols = {
        "player_a_id", "player_b_id", "season", "co_toi_ev",
        "games_together", "xgf_for", "xgf_against", "xgf_pct",
    }
    assert expected_cols.issubset(set(lm.columns)), (
        f"missing columns: {expected_cols - set(lm.columns)}"
    )
    assert (lm["season"] == SEASON).all()


# ---------------------------------------------------------------------------
# 2. test_build_linemate_pairs_canonical_ordering
# ---------------------------------------------------------------------------

def test_build_linemate_pairs_canonical_ordering(simple_stints):
    """player_a_id < player_b_id must hold for every row."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataMissingWarning)
        lm = build_linemate_pairs_with_xgf(simple_stints, SEASON)

    assert not lm.is_empty(), "expected at least one pair"
    assert (lm["player_a_id"] < lm["player_b_id"]).all(), (
        "canonical ordering violated: player_a_id must be < player_b_id"
    )


# ---------------------------------------------------------------------------
# 3. test_build_linemate_pairs_xgf_accumulation
# ---------------------------------------------------------------------------

def test_build_linemate_pairs_xgf_accumulation(simple_stints):
    """Players 1 and 2 share both stints as home linemates.

    xgf_for = 0.30 + 0.12 = 0.42 (xgf_home summed)
    xgf_against = 0.20 + 0.08 = 0.28 (xgf_away summed)
    xgf_pct = 0.42 / (0.42 + 0.28) = 0.60
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataMissingWarning)
        lm = build_linemate_pairs_with_xgf(simple_stints, SEASON)

    row = lm.filter(
        (pl.col("player_a_id") == 1) & (pl.col("player_b_id") == 2)
    )
    assert len(row) == 1
    assert row["xgf_for"][0] == pytest.approx(0.42, abs=1e-6)
    assert row["xgf_against"][0] == pytest.approx(0.28, abs=1e-6)
    assert row["xgf_pct"][0] == pytest.approx(0.60, abs=1e-6)
    assert row["co_toi_ev"][0] == pytest.approx(180.0, abs=1e-6)
    assert row["games_together"][0] == 1


# ---------------------------------------------------------------------------
# 4. test_build_linemate_pairs_min_toi_filter
# ---------------------------------------------------------------------------

def test_build_linemate_pairs_min_toi_filter():
    """Pairs below MIN_SHARED_TOI_SECS must be excluded."""
    stints = _make_stints([{
        "game_id": "G1", "situation": "EV",
        "duration_secs": MIN_SHARED_TOI_SECS - 1.0,
        "home_player_ids": [1, 2],
        "away_player_ids": [6],
        "xgf_home": 0.05, "xgf_away": 0.03,
    }])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataMissingWarning)
        lm = build_linemate_pairs_with_xgf(stints, SEASON)
    assert lm.is_empty(), (
        f"expected empty DataFrame for pair below MIN_SHARED_TOI_SECS "
        f"({MIN_SHARED_TOI_SECS}s), got {len(lm)} rows"
    )


# ---------------------------------------------------------------------------
# 5. test_build_linemate_pairs_empty_ev
# ---------------------------------------------------------------------------

def test_build_linemate_pairs_empty_ev():
    """No EV stints → empty DataFrame + DataMissingWarning."""
    stints = _make_stints([{
        "game_id": "G1", "situation": "PP", "duration_secs": 60.0,
        "home_player_ids": [1, 2], "away_player_ids": [6, 7],
        "xgf_home": 0.10, "xgf_away": 0.02,
    }])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lm = build_linemate_pairs_with_xgf(stints, SEASON)

    assert lm.is_empty(), "expected empty DataFrame for no EV stints"
    warning_types = [w.category for w in caught]
    assert DataMissingWarning in warning_types, (
        "expected DataMissingWarning when no EV stints present"
    )


# ---------------------------------------------------------------------------
# 6. test_build_linemate_pairs_missing_columns
# ---------------------------------------------------------------------------

def test_build_linemate_pairs_missing_columns():
    """Missing required columns raises ValueError."""
    bad = pl.DataFrame({"game_id": ["G1"], "situation": ["EV"]})
    with pytest.raises(ValueError, match="missing columns"):
        build_linemate_pairs_with_xgf(bad, SEASON)


# ---------------------------------------------------------------------------
# 7. test_chemistry_model_fit_predict
# ---------------------------------------------------------------------------

def test_chemistry_model_fit_predict():
    """fit on synthetic data, predict returns PAIR_CHEMISTRY_SCHEMA."""
    stints = _make_multi_game_stints(20)
    rapm   = _make_rapm_for_fit()

    model = LineChemistryModel()
    metrics = model.fit(
        stints_list=[stints, stints],
        rapm_list=[
            rapm.filter(pl.col("season") == 20212022),
            rapm.filter(pl.col("season") == 20222023),
        ],
        finishing_list=[None, None],
        player_stats_list=[None, None],
        seasons=[20212022, 20222023],
    )

    assert "rmse" in metrics
    assert "n_pairs" in metrics
    assert metrics["rmse"] >= 0.0
    assert metrics["n_pairs"] > 0

    preds = model.predict_season(
        stints_df=stints,
        rapm_df=rapm.filter(pl.col("season") == 20232024),
        finishing_df=None,
        player_stats_df=None,
        season=20232024,
    )

    for col in PAIR_CHEMISTRY_SCHEMA:
        assert col in preds.columns, f"missing column: {col}"

    # chemistry_delta must be defined as model_xgf_pct - LEAGUE_AVG_XGF_PCT
    np.testing.assert_allclose(
        preds["chemistry_delta"].to_numpy(),
        preds["model_xgf_pct"].to_numpy() - LEAGUE_AVG_XGF_PCT,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 8. test_chemistry_delta_at_league_avg
# ---------------------------------------------------------------------------

def test_chemistry_delta_at_league_avg():
    """A pair whose model prediction is at league avg gets chemistry_delta ≈ 0."""
    stints = _make_multi_game_stints(20)
    rapm   = _make_rapm_for_fit()

    model = LineChemistryModel()
    model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        finishing_list=[None],
        player_stats_list=[None],
        seasons=[20212022],
    )

    preds = model.predict_season(
        stints_df=stints,
        rapm_df=rapm.filter(pl.col("season") == 20212022),
        finishing_df=None,
        player_stats_df=None,
        season=20212022,
    )
    assert not preds.is_empty()

    # chemistry_delta = model_xgf_pct - LEAGUE_AVG_XGF_PCT for all rows
    deltas = preds["chemistry_delta"].to_numpy()
    model_pcts = preds["model_xgf_pct"].to_numpy()
    np.testing.assert_allclose(deltas, model_pcts - LEAGUE_AVG_XGF_PCT, atol=1e-6)


# ---------------------------------------------------------------------------
# 9. test_blending_below_min_gp
# ---------------------------------------------------------------------------

def test_blending_below_min_gp():
    """Pairs with games_together < MIN_PAIR_GP use model prediction only."""
    # Use fewer games than MIN_PAIR_GP
    stints = _make_multi_game_stints(n_games=MIN_PAIR_GP - 1)
    rapm   = _make_rapm_for_fit()

    model = LineChemistryModel()
    model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        finishing_list=[None],
        player_stats_list=[None],
        seasons=[20212022],
    )
    preds = model.predict_season(
        stints_df=stints,
        rapm_df=rapm.filter(pl.col("season") == 20212022),
        finishing_df=None,
        player_stats_df=None,
        season=20212022,
    )

    # All pairs should have games_together < MIN_PAIR_GP
    assert (preds["games_together"] < MIN_PAIR_GP).all(), (
        "expected all pairs to be below MIN_PAIR_GP for this test"
    )

    # model_xgf_pct == chemistry_delta + LEAGUE_AVG_XGF_PCT
    # and since no blending occurs, chemistry_delta derives directly from model output
    np.testing.assert_allclose(
        preds["chemistry_delta"].to_numpy(),
        preds["model_xgf_pct"].to_numpy() - LEAGUE_AVG_XGF_PCT,
        atol=1e-6,
    )

    # hist_xgf_pct must be null (no history below MIN_PAIR_GP)
    assert preds["hist_xgf_pct"].is_null().all(), (
        "hist_xgf_pct should be null for pairs below MIN_PAIR_GP"
    )


# ---------------------------------------------------------------------------
# 10. test_blending_above_min_gp
# ---------------------------------------------------------------------------

def test_blending_above_min_gp():
    """Pairs with games_together >= MIN_PAIR_GP blend in historical data."""
    stints = _make_multi_game_stints(n_games=MIN_PAIR_GP + 5)
    rapm   = _make_rapm_for_fit()

    model = LineChemistryModel()
    model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        finishing_list=[None],
        player_stats_list=[None],
        seasons=[20212022],
    )
    preds = model.predict_season(
        stints_df=stints,
        rapm_df=rapm.filter(pl.col("season") == 20212022),
        finishing_df=None,
        player_stats_df=None,
        season=20212022,
    )

    above = preds.filter(pl.col("games_together") >= MIN_PAIR_GP)
    assert not above.is_empty(), "expected some pairs above MIN_PAIR_GP"

    # hist_xgf_pct should be non-null for blended pairs
    assert above["hist_xgf_pct"].is_not_null().all(), (
        "hist_xgf_pct should be non-null for pairs at or above MIN_PAIR_GP"
    )

    # model_xgf_pct is the blended value: 0.70*model + 0.30*hist
    # chemistry_delta = model_xgf_pct - LEAGUE_AVG_XGF_PCT
    np.testing.assert_allclose(
        above["chemistry_delta"].to_numpy(),
        above["model_xgf_pct"].to_numpy() - LEAGUE_AVG_XGF_PCT,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 11. test_model_save_load
# ---------------------------------------------------------------------------

def test_model_save_load():
    """save, load, predict again → same output."""
    stints = _make_multi_game_stints(20)
    rapm   = _make_rapm_for_fit()

    model = LineChemistryModel()
    model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        finishing_list=[None],
        player_stats_list=[None],
        seasons=[20212022],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        pkl = Path(tmpdir) / "chemistry.pkl"
        model.save(pkl)

        loaded = LineChemistryModel.load(pkl)

    preds_orig   = model.predict_season(
        stints, rapm.filter(pl.col("season") == 20212022), None, None, 20212022
    )
    preds_loaded = loaded.predict_season(
        stints, rapm.filter(pl.col("season") == 20212022), None, None, 20212022
    )

    np.testing.assert_allclose(
        preds_orig["model_xgf_pct"].to_numpy(),
        preds_loaded["model_xgf_pct"].to_numpy(),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        preds_orig["chemistry_delta"].to_numpy(),
        preds_loaded["chemistry_delta"].to_numpy(),
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# 12. test_compute_player_chemistry_schema
# ---------------------------------------------------------------------------

def test_compute_player_chemistry_schema():
    """compute_player_chemistry output matches PLAYER_CHEMISTRY_SCHEMA."""
    stints = _make_multi_game_stints(20)
    rapm   = _make_rapm_for_fit()

    model = LineChemistryModel()
    model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        finishing_list=[None],
        player_stats_list=[None],
        seasons=[20212022],
    )
    preds = model.predict_season(
        stints, rapm.filter(pl.col("season") == 20212022), None, None, 20212022
    )
    player_df = compute_player_chemistry(
        preds, rapm.filter(pl.col("season") == 20212022), 20212022
    )

    for col in PLAYER_CHEMISTRY_SCHEMA:
        assert col in player_df.columns, f"missing column: {col}"


# ---------------------------------------------------------------------------
# 13. test_compute_player_chemistry_weighted_avg
# ---------------------------------------------------------------------------

def test_compute_player_chemistry_weighted_avg():
    """Verify weighted avg chemistry_delta calculation numerically.

    Construct a synthetic pair_df with known chemistry_delta values and
    known co_toi_ev weights, then verify the player scores match hand-calc.

    Player 1 appears in pairs:
        (1, 2): chemistry_delta=+0.10, co_toi_ev=200
        (1, 3): chemistry_delta=-0.05, co_toi_ev=100

    Expected chemistry_score for player 1:
        (200 * 0.10 + 100 * (-0.05)) / (200 + 100) = (20 - 5) / 300 = 0.05
    """
    pair_df = pl.DataFrame({
        "player_a_id":    pl.Series([1, 1],  dtype=pl.Int64),
        "player_b_id":    pl.Series([2, 3],  dtype=pl.Int64),
        "season":         pl.Series([20212022, 20212022], dtype=pl.Int64),
        "co_toi_ev":      pl.Series([200.0, 100.0], dtype=pl.Float64),
        "games_together": pl.Series([15, 12], dtype=pl.Int64),
        "hist_xgf_pct":   pl.Series([0.60, 0.45], dtype=pl.Float64),
        "model_xgf_pct":  pl.Series([0.60, 0.45], dtype=pl.Float64),
        "chemistry_delta": pl.Series([0.10, -0.05], dtype=pl.Float64),
        "model_version":  pl.Series([MODEL_VERSION, MODEL_VERSION], dtype=pl.Utf8),
    })

    rapm = _make_rapm([
        {"player_id": 1, "player_name": "P1", "team": "TOR", "season": 20212022,
         "gp": 60, "toi_ev": 900.0, "rapm_ev_off": 0.5, "rapm_ev_def": 0.2,
         "rapm_pp": 0.0, "rapm_pk": 0.0},
        {"player_id": 2, "player_name": "P2", "team": "TOR", "season": 20212022,
         "gp": 58, "toi_ev": 800.0, "rapm_ev_off": 1.0, "rapm_ev_def": 0.1,
         "rapm_pp": 0.0, "rapm_pk": 0.0},
        {"player_id": 3, "player_name": "P3", "team": "TOR", "season": 20212022,
         "gp": 55, "toi_ev": 750.0, "rapm_ev_off": -0.5, "rapm_ev_def": -0.3,
         "rapm_pp": 0.0, "rapm_pk": 0.0},
    ])

    player_df = compute_player_chemistry(pair_df, rapm, 20212022)

    p1 = player_df.filter(pl.col("player_id") == 1)
    assert len(p1) == 1
    expected_score = (200.0 * 0.10 + 100.0 * (-0.05)) / (200.0 + 100.0)
    assert p1["chemistry_score"][0] == pytest.approx(expected_score, abs=1e-6)
    assert p1["chemistry_pairs"][0] == 2


# ---------------------------------------------------------------------------
# 14. test_traded_player_no_duplicate_pairs
# ---------------------------------------------------------------------------

def test_traded_player_no_duplicate_pairs():
    """Player traded mid-season appears as both home AND away on same team.

    Player 1 starts on TOR (home team in G1) then is traded to BOS
    (away team in G2).  In both games they play alongside player 2.
    The linemate pair (1, 2) must appear exactly ONCE with co_toi and
    xGF aggregated across both stints — not as two rows.
    """
    stints = _make_stints([
        # G1: players 1 and 2 both on home team (TOR)
        {
            "game_id": "G1", "situation": "EV", "duration_secs": 70.0,
            "home_player_ids": [1, 2], "away_player_ids": [6, 7],
            "xgf_home": 0.20, "xgf_away": 0.15,
        },
        # G2: players 1 and 2 are now both on away team (after trade to BOS)
        {
            "game_id": "G2", "situation": "EV", "duration_secs": 70.0,
            "home_player_ids": [6, 7], "away_player_ids": [1, 2],
            "xgf_home": 0.10, "xgf_away": 0.18,
        },
    ])

    lm = build_linemate_pairs_with_xgf(stints, SEASON)

    rows_1_2 = lm.filter(
        (pl.col("player_a_id") == 1) & (pl.col("player_b_id") == 2)
    )
    # Must be exactly one row — no duplicates from the trade
    assert len(rows_1_2) == 1, (
        f"expected exactly 1 row for pair (1,2), got {len(rows_1_2)}"
    )

    # co_toi_ev should be summed across both stints (70 + 70 = 140s)
    assert rows_1_2["co_toi_ev"][0] == pytest.approx(140.0, abs=1e-6)

    # xgf_for for pair (1,2):
    #   G1: both home → xgf_for=0.20
    #   G2: both away → xgf_for=0.18
    # Total xgf_for = 0.38, xgf_against = 0.15 + 0.10 = 0.25
    assert rows_1_2["xgf_for"][0] == pytest.approx(0.38, abs=1e-6)
    assert rows_1_2["xgf_against"][0] == pytest.approx(0.25, abs=1e-6)

    # Canonical ordering: player_a_id < player_b_id
    assert rows_1_2["player_a_id"][0] < rows_1_2["player_b_id"][0]


# ---------------------------------------------------------------------------
# 15. test_no_crash_missing_optional_inputs
# ---------------------------------------------------------------------------

def test_no_crash_missing_optional_inputs():
    """fit and predict must not crash when finishing_df=None, player_stats_df=None."""
    stints = _make_multi_game_stints(20)
    rapm   = _make_rapm_for_fit()

    model = LineChemistryModel()
    # Should not raise
    metrics = model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        finishing_list=[None],
        player_stats_list=[None],
        seasons=[20212022],
    )
    assert metrics["n_pairs"] > 0

    preds = model.predict_season(
        stints_df=stints,
        rapm_df=rapm.filter(pl.col("season") == 20212022),
        finishing_df=None,
        player_stats_df=None,
        season=20212022,
    )
    assert not preds.is_empty()
    for col in PAIR_CHEMISTRY_SCHEMA:
        assert col in preds.columns


# ---------------------------------------------------------------------------
# Extra: untrained model raises RuntimeError
# ---------------------------------------------------------------------------

def test_untrained_model_raises():
    """predict_season on untrained model raises RuntimeError."""
    model = LineChemistryModel()
    stints = _make_multi_game_stints(5)
    rapm   = _make_rapm_for_fit().filter(pl.col("season") == 20212022)
    with pytest.raises(RuntimeError, match="not trained"):
        model.predict_season(stints, rapm, None, None, 20212022)


# ---------------------------------------------------------------------------
# Extra: predictions in range [0, 1]
# ---------------------------------------------------------------------------

def test_model_xgf_pct_in_range():
    """model_xgf_pct must lie in [0, 1]."""
    stints = _make_multi_game_stints(20)
    rapm   = _make_rapm_for_fit()

    model = LineChemistryModel()
    model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        finishing_list=[None],
        player_stats_list=[None],
        seasons=[20212022],
    )
    preds = model.predict_season(
        stints, rapm.filter(pl.col("season") == 20212022), None, None, 20212022
    )
    vals = preds["model_xgf_pct"].to_numpy()
    assert (vals >= 0.0).all() and (vals <= 1.0).all()


# ---------------------------------------------------------------------------
# Extra: write/read roundtrip
# ---------------------------------------------------------------------------

def test_write_read_pair_chemistry():
    """write_pair_chemistry and read_pair_chemistry roundtrip correctly."""
    stints = _make_multi_game_stints(20)
    rapm   = _make_rapm_for_fit()

    model = LineChemistryModel()
    model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        finishing_list=[None],
        player_stats_list=[None],
        seasons=[20212022],
    )
    preds = model.predict_season(
        stints, rapm.filter(pl.col("season") == 20212022), None, None, 20212022
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        path = write_pair_chemistry(preds, out_dir, 20212022)
        assert path.exists()

        # read_pair_chemistry expects data_dir/chemistry/pair_chemistry_*.parquet
        (out_dir / "chemistry").mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(path, out_dir / "chemistry" / "pair_chemistry_20212022.parquet")

        loaded = read_pair_chemistry(out_dir, 20212022)

    assert set(loaded.columns) == set(PAIR_CHEMISTRY_SCHEMA.keys())
    assert len(loaded) == len(preds)


def test_write_read_player_chemistry():
    """write_player_chemistry and read_player_chemistry roundtrip correctly."""
    stints = _make_multi_game_stints(20)
    rapm   = _make_rapm_for_fit()

    model = LineChemistryModel()
    model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        finishing_list=[None],
        player_stats_list=[None],
        seasons=[20212022],
    )
    preds = model.predict_season(
        stints, rapm.filter(pl.col("season") == 20212022), None, None, 20212022
    )
    player_df = compute_player_chemistry(
        preds, rapm.filter(pl.col("season") == 20212022), 20212022
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        path = write_player_chemistry(player_df, out_dir, 20212022)
        assert path.exists()

        (out_dir / "chemistry").mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(path, out_dir / "chemistry" / "player_chemistry_20212022.parquet")

        loaded = read_player_chemistry(out_dir, 20212022)

    assert set(loaded.columns) == set(PLAYER_CHEMISTRY_SCHEMA.keys())
    assert len(loaded) == len(player_df)


# ---------------------------------------------------------------------------
# Extra: finishing and player_stats are used when provided
# ---------------------------------------------------------------------------

def test_chemistry_model_with_finishing_and_stats():
    """fit and predict with finishing_df and player_stats_df do not crash."""
    stints = _make_multi_game_stints(20)
    rapm   = _make_rapm_for_fit()

    finishing = _make_finishing([
        {"shooter_id": pid, "shooter_name": f"P{pid}", "team": t, "season": 20212022,
         "shots": 100, "goals": 10, "xg_sum": 8.0,
         "finishing": 2.0, "finishing_per60": 0.5, "model_version": "xg_v1"}
        for pid, t in [(1, "TOR"), (2, "TOR"), (3, "TOR"), (6, "BOS"), (7, "BOS"), (8, "BOS")]
    ])

    player_stats = _make_player_stats([
        {"player_id": pid, "season": 20212022, "zone_start_oz_pct": 0.52, "zone_start_dz_pct": 0.48}
        for pid in [1, 2, 3, 6, 7, 8]
    ])

    model = LineChemistryModel()
    metrics = model.fit(
        stints_list=[stints],
        rapm_list=[rapm.filter(pl.col("season") == 20212022)],
        finishing_list=[finishing],
        player_stats_list=[player_stats],
        seasons=[20212022],
    )
    assert metrics["n_pairs"] > 0

    preds = model.predict_season(
        stints_df=stints,
        rapm_df=rapm.filter(pl.col("season") == 20212022),
        finishing_df=finishing,
        player_stats_df=player_stats,
        season=20212022,
    )
    assert not preds.is_empty()
    for col in PAIR_CHEMISTRY_SCHEMA:
        assert col in preds.columns
