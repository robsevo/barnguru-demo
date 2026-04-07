"""Unit tests for models/special_teams_model.py — Feature 2.7.

All tests use synthetic data — no live API calls or on-disk parquet files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from models.special_teams_model import (
    MIN_PK_TOI_SECS,
    MIN_PP_TOI_SECS,
    MODEL_VERSION,
    SPECIAL_TEAMS_SCHEMA,
    _compute_situation_rates,
    _zscore,
    compute_special_teams,
    read_special_teams,
    top_pk_players,
    top_pp_players,
    write_special_teams,
)


# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------


def _make_stints(
    n_pp: int = 30,
    n_pk: int = 30,
    n_ev: int = 40,
    seed: int = 42,
) -> pl.DataFrame:
    """Build a synthetic stints DataFrame with PP, PK, and EV stints.

    Player IDs are in the range 1–10 for home and 11–20 for away.
    Each stint has 5 home and 5 away players (matching real on-ice counts).
    """
    rng = np.random.default_rng(seed)
    home_pool = list(range(1, 11))     # 10 home players
    away_pool = list(range(11, 21))    # 10 away players

    rows = []
    # PP stints: home team on power play (more home skaters)
    for _ in range(n_pp):
        rows.append({
            "game_id":         "G001",
            "situation":       "PP",
            "duration_secs":   float(rng.integers(30, 120)),
            "home_player_ids": rng.choice(home_pool, size=5, replace=False).tolist(),
            "away_player_ids": rng.choice(away_pool, size=4, replace=False).tolist(),
            "xgf_home":        float(rng.uniform(0.0, 0.3)),
            "xgf_away":        float(rng.uniform(0.0, 0.05)),
        })
    # PK stints: home team on penalty kill (fewer home skaters)
    for _ in range(n_pk):
        rows.append({
            "game_id":         "G001",
            "situation":       "PK",
            "duration_secs":   float(rng.integers(30, 120)),
            "home_player_ids": rng.choice(home_pool, size=4, replace=False).tolist(),
            "away_player_ids": rng.choice(away_pool, size=5, replace=False).tolist(),
            "xgf_home":        float(rng.uniform(0.0, 0.05)),
            "xgf_away":        float(rng.uniform(0.0, 0.3)),
        })
    # EV stints
    for _ in range(n_ev):
        rows.append({
            "game_id":         "G001",
            "situation":       "EV",
            "duration_secs":   float(rng.integers(30, 120)),
            "home_player_ids": rng.choice(home_pool, size=5, replace=False).tolist(),
            "away_player_ids": rng.choice(away_pool, size=5, replace=False).tolist(),
            "xgf_home":        float(rng.uniform(0.0, 0.2)),
            "xgf_away":        float(rng.uniform(0.0, 0.2)),
        })

    return pl.DataFrame(rows).with_columns([
        pl.col("home_player_ids").cast(pl.List(pl.Int64)),
        pl.col("away_player_ids").cast(pl.List(pl.Int64)),
    ])


def _make_rapm(n: int = 20, seed: int = 42) -> pl.DataFrame:
    """Build a synthetic RAPM DataFrame for players 1–n."""
    rng = np.random.default_rng(seed)
    teams = ["EDM", "CGY", "TOR", "BOS", "NYR"]
    return pl.DataFrame({
        "player_id":   list(range(1, n + 1)),
        "player_name": [f"Player {i}" for i in range(1, n + 1)],
        "team":        [teams[i % len(teams)] for i in range(n)],
        "season":      [2025] * n,
        "gp":          rng.integers(20, 82, size=n).tolist(),
        "toi_ev":      rng.uniform(500.0, 2000.0, size=n).tolist(),
        "toi_pp":      rng.uniform(60.0, 400.0, size=n).tolist(),
        "toi_pk":      rng.uniform(60.0, 400.0, size=n).tolist(),
        "rapm_ev_off": rng.uniform(-0.5, 0.5, size=n).tolist(),
        "rapm_ev_def": rng.uniform(-0.5, 0.5, size=n).tolist(),
        "rapm_pp":     rng.uniform(-0.5, 0.5, size=n).tolist(),
        "rapm_pk":     rng.uniform(-0.5, 0.5, size=n).tolist(),
        "xga_60":      rng.uniform(1.5, 3.5, size=n).tolist(),
        "model_version": ["rapm_v1"] * n,
    })


# ---------------------------------------------------------------------------
# _zscore
# ---------------------------------------------------------------------------


class TestZscore:
    def test_mean_zero(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = _zscore(arr)
        assert abs(z.mean()) < 1e-10

    def test_std_one(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = _zscore(arr)
        assert abs(z.std() - 1.0) < 1e-10

    def test_constant_returns_zeros(self) -> None:
        arr = np.array([7.0, 7.0, 7.0])
        z = _zscore(arr)
        assert (z == 0.0).all()

    def test_nan_imputed_before_scoring(self) -> None:
        arr = np.array([1.0, 2.0, float("nan"), 4.0])
        z = _zscore(arr)
        assert not np.isnan(z).any()

    def test_single_element(self) -> None:
        arr = np.array([5.0])
        z = _zscore(arr)
        assert z[0] == 0.0


# ---------------------------------------------------------------------------
# _compute_situation_rates
# ---------------------------------------------------------------------------


class TestComputeSituationRates:
    def test_pp_home_returns_dataframe(self) -> None:
        stints = _make_stints(n_pp=10, n_pk=0, n_ev=0)
        result = _compute_situation_rates(
            stints, "PP", "home_player_ids", "xgf_home", "xgf_away"
        )
        assert isinstance(result, pl.DataFrame)
        assert "player_id" in result.columns
        assert "duration_secs" in result.columns
        assert "xgf" in result.columns
        assert "xga" in result.columns

    def test_no_matching_situation_returns_empty(self) -> None:
        stints = _make_stints(n_pp=0, n_pk=0, n_ev=10)
        result = _compute_situation_rates(
            stints, "PP", "home_player_ids", "xgf_home", "xgf_away"
        )
        assert result.is_empty()

    def test_player_ids_are_int64(self) -> None:
        stints = _make_stints(n_pp=10, n_pk=0, n_ev=0)
        result = _compute_situation_rates(
            stints, "PP", "home_player_ids", "xgf_home", "xgf_away"
        )
        assert result["player_id"].dtype == pl.Int64

    def test_duration_sums_correctly(self) -> None:
        """Total duration across all players must sum to ≥ stint total (overlap expected)."""
        stints = _make_stints(n_pp=20, n_pk=0, n_ev=0)
        result = _compute_situation_rates(
            stints, "PP", "home_player_ids", "xgf_home", "xgf_away"
        )
        stint_total = stints.filter(pl.col("situation") == "PP")["duration_secs"].sum()
        # Each stint contributes to 5 players → player sum ≥ stint total
        assert result["duration_secs"].sum() >= stint_total

    def test_xgf_non_negative(self) -> None:
        stints = _make_stints(n_pp=20, n_pk=0, n_ev=0)
        result = _compute_situation_rates(
            stints, "PP", "home_player_ids", "xgf_home", "xgf_away"
        )
        assert (result["xgf"] >= 0.0).all()


# ---------------------------------------------------------------------------
# compute_special_teams
# ---------------------------------------------------------------------------


class TestComputeSpecialTeams:
    def test_returns_dataframe(self) -> None:
        stints = _make_stints()
        rapm   = _make_rapm()
        result = compute_special_teams(stints, rapm, season=2025)
        assert isinstance(result, pl.DataFrame)

    def test_schema_columns_present(self) -> None:
        stints = _make_stints()
        rapm   = _make_rapm()
        result = compute_special_teams(stints, rapm, season=2025)
        for col in SPECIAL_TEAMS_SCHEMA:
            assert col in result.columns, f"Missing column: {col}"

    def test_model_version_correct(self) -> None:
        stints = _make_stints()
        rapm   = _make_rapm()
        result = compute_special_teams(stints, rapm, season=2025)
        assert (result["model_version"] == MODEL_VERSION).all()

    def test_season_column_correct(self) -> None:
        stints = _make_stints()
        rapm   = _make_rapm()
        result = compute_special_teams(stints, rapm, season=2025)
        assert (result["season"] == 2025).all()

    def test_toi_non_negative(self) -> None:
        stints = _make_stints()
        rapm   = _make_rapm()
        result = compute_special_teams(stints, rapm, season=2025)
        assert (result["toi_pp"] >= 0.0).all()
        assert (result["toi_pk"] >= 0.0).all()

    def test_pp_rating_nan_for_low_toi(self) -> None:
        """Players with < MIN_PP_TOI_SECS of PP time should have NaN pp_rating."""
        # Build stints with ONLY EV and very short PP/PK so nobody crosses the threshold
        stints = _make_stints(n_pp=0, n_pk=0, n_ev=40)
        rapm   = _make_rapm()
        result = compute_special_teams(stints, rapm, season=2025)
        # With no PP or PK stints, all ratings should be NaN
        pp_rated = result.filter(pl.col("pp_rating").is_not_nan())
        assert len(pp_rated) == 0

    def test_sorted_descending_by_pp_rating(self) -> None:
        stints = _make_stints(n_pp=50, n_pk=50, n_ev=50)
        rapm   = _make_rapm()
        result = compute_special_teams(stints, rapm, season=2025)
        rated = result.filter(pl.col("pp_rating").is_not_nan())
        vals = rated["pp_rating"].to_list()
        assert vals == sorted(vals, reverse=True)

    def test_missing_stints_column_raises(self) -> None:
        stints = _make_stints().drop("situation")
        rapm   = _make_rapm()
        with pytest.raises(ValueError, match="stints_df missing required columns"):
            compute_special_teams(stints, rapm, season=2025)

    def test_missing_rapm_column_raises(self) -> None:
        stints = _make_stints()
        rapm   = _make_rapm().drop("rapm_pp")
        with pytest.raises(ValueError, match="rapm_df missing required columns"):
            compute_special_teams(stints, rapm, season=2025)

    def test_season_filter_applied(self) -> None:
        """When rapm_df has multiple seasons, only the requested season is used."""
        rapm_2024 = _make_rapm(10, seed=1).with_columns(pl.lit(2024).alias("season"))
        rapm_2025 = _make_rapm(10, seed=2).with_columns(pl.lit(2025).alias("season"))
        combined  = pl.concat([rapm_2024, rapm_2025], how="diagonal")
        stints    = _make_stints()
        result    = compute_special_teams(stints, combined, season=2025)
        assert (result["season"] == 2025).all()
        assert len(result) == 10

    def test_no_nan_in_toi_columns(self) -> None:
        stints = _make_stints()
        rapm   = _make_rapm()
        result = compute_special_teams(stints, rapm, season=2025)
        assert not result["toi_pp"].is_nan().any()
        assert not result["toi_pk"].is_nan().any()

    def test_per60_rates_computed(self) -> None:
        stints = _make_stints(n_pp=50, n_pk=50, n_ev=50)
        rapm   = _make_rapm()
        result = compute_special_teams(stints, rapm, season=2025)
        # Per-60 rates should be non-negative for players with TOI
        pp_players = result.filter(pl.col("toi_pp") > 0)
        assert (pp_players["pp_xgf60"] >= 0.0).all()

    def test_empty_stints_returns_rapm_players_with_nan_ratings(self) -> None:
        """Stints with only EV data → all players get NaN PP and PK ratings."""
        stints = _make_stints(n_pp=0, n_pk=0, n_ev=20)
        rapm   = _make_rapm()
        result = compute_special_teams(stints, rapm, season=2025)
        # Should return the RAPM universe, all with NaN ratings
        assert len(result) == len(rapm)
        assert result["pp_rating"].is_nan().all()
        assert result["pk_rating"].is_nan().all()


# ---------------------------------------------------------------------------
# top_pp_players / top_pk_players
# ---------------------------------------------------------------------------


class TestTopPlayers:
    def _sample_st(self) -> pl.DataFrame:
        stints = _make_stints(n_pp=50, n_pk=50, n_ev=50)
        rapm   = _make_rapm()
        return compute_special_teams(stints, rapm, season=2025)

    def test_top_pp_returns_n(self) -> None:
        st = self._sample_st()
        top = top_pp_players(st, n=5)
        assert len(top) <= 5

    def test_top_pk_returns_n(self) -> None:
        st = self._sample_st()
        top = top_pk_players(st, n=5)
        assert len(top) <= 5

    def test_top_pp_sorted_descending(self) -> None:
        st  = self._sample_st()
        top = top_pp_players(st, n=10)
        vals = top["pp_rating"].to_list()
        assert vals == sorted(vals, reverse=True)

    def test_top_pk_sorted_descending(self) -> None:
        st  = self._sample_st()
        top = top_pk_players(st, n=10)
        vals = top["pk_rating"].to_list()
        assert vals == sorted(vals, reverse=True)

    def test_top_pp_no_nan_ratings(self) -> None:
        st  = self._sample_st()
        top = top_pp_players(st, n=20)
        assert not top["pp_rating"].is_nan().any()

    def test_top_pk_no_nan_ratings(self) -> None:
        st  = self._sample_st()
        top = top_pk_players(st, n=20)
        assert not top["pk_rating"].is_nan().any()


# ---------------------------------------------------------------------------
# write_special_teams / read_special_teams
# ---------------------------------------------------------------------------


class TestIO:
    def _sample_st(self) -> pl.DataFrame:
        stints = _make_stints(n_pp=50, n_pk=50, n_ev=50)
        rapm   = _make_rapm()
        return compute_special_teams(stints, rapm, season=2025)

    def test_write_creates_file(self) -> None:
        st = self._sample_st()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "special_teams"
            path = write_special_teams(st, out_dir, season=2025)
            assert path.exists()
            assert path.name == "special_teams_2025.parquet"

    def test_write_read_roundtrip(self) -> None:
        st = self._sample_st()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "special_teams"
            write_special_teams(st, out_dir, season=2025)
            loaded = read_special_teams(Path(tmp), season=2025)
        assert loaded is not None
        assert len(loaded) == len(st)
        np.testing.assert_allclose(
            st["toi_pp"].to_numpy(),
            loaded["toi_pp"].to_numpy(),
            rtol=1e-5,
        )

    def test_read_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = read_special_teams(Path(tmp), season=2025)
        assert result is None

    def test_read_latest_when_no_season(self) -> None:
        st = self._sample_st()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "special_teams"
            write_special_teams(st, out_dir, season=2025)
            loaded = read_special_teams(Path(tmp))
        assert loaded is not None
        assert len(loaded) == len(st)

    def test_write_creates_output_dir(self) -> None:
        st = self._sample_st()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "nested" / "special_teams"
            assert not out_dir.exists()
            write_special_teams(st, out_dir, season=2025)
            assert out_dir.exists()
