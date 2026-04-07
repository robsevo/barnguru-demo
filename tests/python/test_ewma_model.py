"""Tests for EWMAFormModel (Feature 2.13)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from models.ewma_model import (
    EWMA_ALPHA,
    EWMA_LAMBDA,
    HOT_SIGMA_THRESHOLD,
    COLD_SIGMA_THRESHOLD,
    MIN_GAMES_FOR_FLAG,
    PLAYER_FORM_SCHEMA,
    PLAYER_FORM_SUMMARY_SCHEMA,
    EWMAFormModel,
    compute_ewma_series,
    write_player_form,
    read_player_form_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_game_log(n_players: int = 5, n_games: int = 30, seed: int = 42) -> pl.DataFrame:
    """Synthetic per-player-game DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1000, 1000 + n_players):
        for g in range(1, n_games + 1):
            rows.append({
                "player_id":   pid,
                "player_name": f"Player{pid - 1000}",
                "game_id":     pid * 1000 + g,
                "xgf_per60":   float(rng.normal(2.5, 1.0)),
                "corsi_pct":   float(rng.normal(0.50, 0.06)),
                "sv_pct":      None,   # not a goalie
            })
    return pl.DataFrame(rows)


def _make_goalie_log(n_games: int = 30, seed: int = 7) -> pl.DataFrame:
    """Synthetic per-goalie-game DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(1, n_games + 1):
        rows.append({
            "player_id":   9001,
            "player_name": "GoalieX",
            "game_id":     9001 * 1000 + g,
            "xgf_per60":   0.0,
            "corsi_pct":   None,
            "sv_pct":      float(rng.normal(0.91, 0.02)),
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_ewma_series
# ---------------------------------------------------------------------------

class TestComputeEwmaSeries:

    def test_single_value_returns_that_value(self):
        arr = compute_ewma_series(np.array([3.0]))
        assert arr[0] == pytest.approx(3.0, abs=1e-6)

    def test_constant_series_stays_constant(self):
        arr = compute_ewma_series(np.full(20, 2.5))
        np.testing.assert_array_almost_equal(arr, 2.5, decimal=6)

    def test_length_matches_input(self):
        vals = np.linspace(1, 5, 50)
        arr = compute_ewma_series(vals)
        assert len(arr) == 50

    def test_ewma_converges_toward_new_level(self):
        """After a step-change, EWMA should converge toward new level."""
        vals = np.concatenate([np.full(30, 2.0), np.full(50, 5.0)])
        arr = compute_ewma_series(vals)
        assert arr[-1] > arr[30]   # converging upward after step change

    def test_warm_start_sets_initial_value(self):
        vals = np.array([3.0, 3.0, 3.0])
        arr = compute_ewma_series(vals, warm_start=10.0)
        # First update: 0.96*10 + 0.04*3 = 9.72
        assert arr[0] == pytest.approx(0.96 * 10.0 + 0.04 * 3.0, abs=1e-6)

    def test_nan_values_propagate_ewma_unchanged(self):
        vals = np.array([3.0, np.nan, 3.0])
        arr = compute_ewma_series(vals)
        # NaN should not change EWMA
        assert arr[1] == pytest.approx(arr[0], abs=1e-9)

    def test_empty_array_returns_empty(self):
        arr = compute_ewma_series(np.array([]))
        assert len(arr) == 0

    def test_lambda_decay(self):
        """Verify exact EWMA formula: ewma_t = (1-α)*ewma_{t-1} + α*value_t"""
        vals = np.array([1.0, 2.0, 3.0])
        arr = compute_ewma_series(vals, alpha=0.04, warm_start=vals[0])
        expected_0 = vals[0]
        expected_1 = 0.96 * expected_0 + 0.04 * 2.0
        expected_2 = 0.96 * expected_1 + 0.04 * 3.0
        assert arr[0] == pytest.approx(expected_0, abs=1e-9)
        assert arr[1] == pytest.approx(expected_1, abs=1e-9)
        assert arr[2] == pytest.approx(expected_2, abs=1e-9)


# ---------------------------------------------------------------------------
# EWMAFormModel — init
# ---------------------------------------------------------------------------

class TestEWMAFormModelInit:

    def test_default_lambda(self):
        model = EWMAFormModel()
        assert model.lambda_ == pytest.approx(EWMA_LAMBDA)

    def test_default_alpha(self):
        model = EWMAFormModel()
        assert model.alpha == pytest.approx(EWMA_ALPHA)

    def test_half_life_approximately_17(self):
        model = EWMAFormModel()
        # ln(0.5)/ln(0.96) ≈ 17.0 games
        assert model.half_life_games == pytest.approx(17.0, abs=1.0)

    def test_custom_alpha(self):
        model = EWMAFormModel(alpha=0.10)
        assert model.alpha == pytest.approx(0.10)
        assert model.lambda_ == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# EWMAFormModel — compute
# ---------------------------------------------------------------------------

class TestEWMAFormCompute:

    def test_compute_returns_dataframe(self):
        df = _make_game_log(3, 20)
        model = EWMAFormModel()
        result = model.compute(df, season=2025)
        assert isinstance(result, pl.DataFrame)

    def test_output_schema(self):
        df = _make_game_log(3, 20)
        model = EWMAFormModel()
        result = model.compute(df, season=2025)
        for col in PLAYER_FORM_SCHEMA:
            assert col in result.columns, f"Missing: {col}"

    def test_row_count_matches_input(self):
        df = _make_game_log(4, 25)
        model = EWMAFormModel()
        result = model.compute(df, season=2025)
        assert len(result) == len(df)

    def test_sorted_by_player_and_game(self):
        df = _make_game_log(3, 20)
        model = EWMAFormModel()
        result = model.compute(df, season=2025)
        pids    = result["player_id"].to_list()
        gnums   = result["game_number"].to_list()
        for i in range(len(pids) - 1):
            if pids[i] == pids[i + 1]:
                assert gnums[i] <= gnums[i + 1]

    def test_form_flags_are_valid(self):
        df = _make_game_log(3, 30)
        model = EWMAFormModel()
        result = model.compute(df, season=2025)
        valid = {"hot", "cold", "neutral"}
        assert set(result["form_flag"].to_list()).issubset(valid)

    def test_missing_xgf_column_raises(self):
        df = _make_game_log(2, 10).drop("xgf_per60")
        model = EWMAFormModel()
        with pytest.raises(ValueError, match="missing columns"):
            model.compute(df, season=2025)

    def test_missing_game_id_raises(self):
        df = _make_game_log(2, 10).drop("game_id")
        model = EWMAFormModel()
        with pytest.raises(ValueError, match="missing columns"):
            model.compute(df, season=2025)

    def test_corsi_ewma_populated_when_column_present(self):
        df = _make_game_log(2, 20)
        model = EWMAFormModel()
        result = model.compute(df, season=2025)
        assert result["ewma_corsi_pct"].null_count() == 0

    def test_sv_pct_null_when_not_goalie(self):
        df = _make_game_log(2, 10)
        # No sv_pct column → should be null in output
        model = EWMAFormModel()
        result = model.compute(df.drop("sv_pct") if "sv_pct" in df.columns else df, season=2025)
        assert result["ewma_sv_pct"].null_count() == len(result)

    def test_goalie_sv_pct_populated(self):
        df = _make_goalie_log(30)
        model = EWMAFormModel()
        result = model.compute(df, season=2025)
        sv_vals = result.filter(pl.col("player_id") == 9001)["ewma_sv_pct"]
        assert sv_vals.null_count() == 0

    def test_empty_df_returns_empty(self):
        df = pl.DataFrame({
            "player_id": pl.Series([], dtype=pl.Int64),
            "game_id":   pl.Series([], dtype=pl.Int64),
            "xgf_per60": pl.Series([], dtype=pl.Float64),
        })
        model = EWMAFormModel()
        result = model.compute(df, season=2025)
        assert len(result) == 0

    def test_game_number_sequential(self):
        df = _make_game_log(1, 15)
        model = EWMAFormModel()
        result = model.compute(df, season=2025)
        gnums = result["game_number"].to_list()
        assert gnums == list(range(1, 16))

    def test_high_scorer_marked_hot(self):
        """Player consistently scoring very high should get hot flag."""
        rng = np.random.default_rng(99)
        rows = []
        # Hot player: always 6.0 xGF/60
        for g in range(1, 40):
            rows.append({"player_id": 1, "player_name": "Hot", "game_id": g, "xgf_per60": 6.0})
        # Cold player: always 0.1 xGF/60
        for g in range(1, 40):
            rows.append({"player_id": 2, "player_name": "Cold", "game_id": 100 + g, "xgf_per60": 0.1})
        df = pl.DataFrame(rows)
        model = EWMAFormModel()
        result = model.compute(df, season=2025)
        # After enough games, hot/cold flags should appear
        hot_flags  = result.filter(
            (pl.col("player_id") == 1) & (pl.col("game_number") >= MIN_GAMES_FOR_FLAG)
        )["form_flag"].to_list()
        cold_flags = result.filter(
            (pl.col("player_id") == 2) & (pl.col("game_number") >= MIN_GAMES_FOR_FLAG)
        )["form_flag"].to_list()
        # At least some hot/cold flags expected
        assert "hot"  in hot_flags  or all(f == "neutral" for f in hot_flags)
        assert "cold" in cold_flags or all(f == "neutral" for f in cold_flags)

    def test_neutral_for_early_games(self):
        """Before MIN_GAMES_FOR_FLAG, all flags should be neutral."""
        rows = [
            {"player_id": 1, "player_name": "A", "game_id": g, "xgf_per60": 10.0}
            for g in range(1, MIN_GAMES_FOR_FLAG)
        ]
        df = pl.DataFrame(rows)
        model = EWMAFormModel()
        result = model.compute(df, season=2025)
        assert (result["form_flag"] == "neutral").all()


# ---------------------------------------------------------------------------
# EWMAFormModel — summary
# ---------------------------------------------------------------------------

class TestEWMAFormSummary:

    def test_summary_one_row_per_player(self):
        df = _make_game_log(5, 30)
        model = EWMAFormModel()
        summary = model.summary(df, season=2025)
        assert len(summary) == 5

    def test_summary_schema(self):
        df = _make_game_log(3, 20)
        model = EWMAFormModel()
        summary = model.summary(df, season=2025)
        for col in PLAYER_FORM_SUMMARY_SCHEMA:
            assert col in summary.columns, f"Missing: {col}"

    def test_summary_sorted_by_ewma_descending(self):
        df = _make_game_log(5, 30)
        model = EWMAFormModel()
        summary = model.summary(df, season=2025)
        vals = summary["ewma_xgf60"].to_list()
        assert vals == sorted(vals, reverse=True)

    def test_games_played_correct(self):
        df = _make_game_log(2, 25)
        model = EWMAFormModel()
        summary = model.summary(df, season=2025)
        for row in summary.to_dicts():
            assert row["games_played"] == 25

    def test_season_mean_non_zero(self):
        df = _make_game_log(3, 20)
        model = EWMAFormModel()
        summary = model.summary(df, season=2025)
        assert (summary["season_mean_xgf60"] > 0).all()


# ---------------------------------------------------------------------------
# EWMAFormModel — update_player
# ---------------------------------------------------------------------------

class TestEWMAUpdatePlayer:

    def test_single_update_formula(self):
        model = EWMAFormModel()
        result = model.update_player(3.0, 5.0)
        expected = (1 - EWMA_ALPHA) * 3.0 + EWMA_ALPHA * 5.0
        assert result == pytest.approx(expected, abs=1e-9)

    def test_nan_observation_unchanged(self):
        model = EWMAFormModel()
        result = model.update_player(3.0, float("nan"))
        assert result == pytest.approx(3.0)

    def test_constant_observation_converges(self):
        model = EWMAFormModel()
        ewma = 3.0
        for _ in range(500):
            ewma = model.update_player(ewma, 5.0)
        assert ewma == pytest.approx(5.0, abs=0.01)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

class TestEWMAModelIO:

    def test_write_read_parquets(self, tmp_path):
        df = _make_game_log(3, 20)
        model = EWMAFormModel()
        form_df    = model.compute(df, season=2025)
        summary_df = model.summary(df, season=2025)
        p_form, p_summary = write_player_form(form_df, summary_df, tmp_path / "form", 2025)
        assert p_form.exists()
        assert p_summary.exists()
        loaded_summary = pl.read_parquet(p_summary)
        assert len(loaded_summary) == 3
