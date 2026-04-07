"""Unit tests for models/defensive_rating.py — Feature 2.26.

Tests cover CDR computation, edge cases, and I/O without requiring
real Evolving Hockey or historical ingestor data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from models.defensive_rating import (
    CDR_SCHEMA,
    MIN_GP,
    CompositeDefensiveRating,
    _zscore,
    lookup_player,
    read_cdr,
    write_cdr,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_rapm_df(n: int = 50, seed: int = 42) -> pl.DataFrame:
    """Create a synthetic RAPM DataFrame matching the EH schema."""
    rng = np.random.default_rng(seed)
    teams = ["EDM", "CGY", "TOR", "BOS", "NYR"]
    return pl.DataFrame({
        "player_name": [f"Player {i}" for i in range(n)],
        "player_id":   rng.integers(8_000_000, 9_000_000, size=n).tolist(),
        "team":        [teams[i % len(teams)] for i in range(n)],
        "season":      [2025] * n,
        "gp":          rng.integers(15, 82, size=n).tolist(),
        "toi":         rng.uniform(200.0, 1500.0, size=n).tolist(),
        "rapm_xga":    rng.uniform(-0.5, 0.5, size=n).tolist(),
        "xga_60":      rng.uniform(1.5, 3.5, size=n).tolist(),
        "dzs_pct":     rng.uniform(0.35, 0.65, size=n).tolist(),
    })


def _make_pbp_stats_df(rapm_df: pl.DataFrame, seed: int = 42) -> pl.DataFrame:
    """Create matching PBP player stats (T/G counts)."""
    rng = np.random.default_rng(seed)
    n = len(rapm_df)
    return pl.DataFrame({
        "player_id":      rapm_df["player_id"].to_list(),
        "player_name":    rapm_df["player_name"].to_list(),
        "takeaway_count": rng.integers(10, 80, size=n).tolist(),
        "giveaway_count": rng.integers(10, 60, size=n).tolist(),
    })


# ---------------------------------------------------------------------------
# _zscore
# ---------------------------------------------------------------------------


class TestZscore:
    def test_mean_zero_std_one(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = _zscore(arr)
        assert abs(z.mean()) < 1e-10
        assert abs(z.std() - 1.0) < 1e-10

    def test_constant_array_returns_zeros(self) -> None:
        arr = np.array([3.0, 3.0, 3.0])
        z = _zscore(arr)
        assert (z == 0.0).all()

    def test_nan_replaced_with_mean(self) -> None:
        arr = np.array([1.0, 2.0, float("nan"), 4.0])
        z = _zscore(arr)
        assert not np.isnan(z).any()


# ---------------------------------------------------------------------------
# CompositeDefensiveRating.compute
# ---------------------------------------------------------------------------


class TestComputeCDR:
    def test_shape(self) -> None:
        rapm = _make_rapm_df(50)
        pbp  = _make_pbp_stats_df(rapm)
        cdr  = CompositeDefensiveRating().compute(rapm, pbp, season=2025)
        assert len(cdr) == 50
        assert set(CDR_SCHEMA.keys()).issubset(set(cdr.columns))

    def test_cdr_range(self) -> None:
        rapm = _make_rapm_df(100)
        pbp  = _make_pbp_stats_df(rapm)
        cdr  = CompositeDefensiveRating().compute(rapm, pbp, season=2025)
        vals = cdr["cdr"].to_numpy()
        assert vals.min() > -5.0, "CDR too negative — check weights"
        assert vals.max() <  5.0, "CDR too positive — check weights"

    def test_sorted_descending(self) -> None:
        rapm = _make_rapm_df(30)
        pbp  = _make_pbp_stats_df(rapm)
        cdr  = CompositeDefensiveRating().compute(rapm, pbp, season=2025)
        vals = cdr["cdr"].to_list()
        assert vals == sorted(vals, reverse=True)

    def test_minimum_gp_filter(self) -> None:
        rapm = _make_rapm_df(40)
        # Force half the players below MIN_GP
        gp_vals = [MIN_GP - 1 if i % 2 == 0 else 25 for i in range(40)]
        rapm = rapm.with_columns(pl.Series("gp", gp_vals))
        pbp  = _make_pbp_stats_df(rapm)
        cdr  = CompositeDefensiveRating().compute(rapm, pbp, season=2025)
        assert len(cdr) == 20, f"Expected 20 qualified players, got {len(cdr)}"
        assert cdr["gp"].min() >= MIN_GP

    def test_dzs_adjustment_applied(self) -> None:
        """Player with 65% DZS gets lower xga_60_adj than raw xga_60."""
        rapm = _make_rapm_df(20)
        high_dzs = rapm.with_columns(pl.lit(0.65).alias("dzs_pct"))
        low_dzs  = rapm.with_columns(pl.lit(0.35).alias("dzs_pct"))
        pbp = _make_pbp_stats_df(rapm)
        cdr_high = CompositeDefensiveRating().compute(high_dzs, pbp, season=2025)
        cdr_low  = CompositeDefensiveRating().compute(low_dzs, pbp, season=2025)
        # High DZS → lower xga_60_adj (credit for tough deployment)
        assert cdr_high["xga_60_adj"].mean() < cdr_low["xga_60_adj"].mean()

    def test_zero_giveaways_no_division_error(self) -> None:
        rapm = _make_rapm_df(10)
        pbp  = _make_pbp_stats_df(rapm).with_columns(pl.lit(0).alias("giveaway_count"))
        # Should not raise ZeroDivisionError
        cdr = CompositeDefensiveRating().compute(rapm, pbp, season=2025)
        assert not cdr["cdr"].is_nan().any()

    def test_missing_pbp_stats_uses_medians(self) -> None:
        """Empty pbp_stats_df triggers median fallback — should not raise."""
        rapm = _make_rapm_df(20)
        cdr  = CompositeDefensiveRating().compute(rapm, pl.DataFrame(), season=2025)
        assert len(cdr) == 20

    def test_missing_required_column_raises(self) -> None:
        rapm = _make_rapm_df(10).drop("rapm_xga")
        with pytest.raises(ValueError, match="missing required columns"):
            CompositeDefensiveRating().compute(rapm, pl.DataFrame(), season=2025)

    def test_no_qualified_players_raises(self) -> None:
        rapm = _make_rapm_df(10).with_columns(pl.lit(MIN_GP - 1).alias("gp"))
        with pytest.raises(ValueError, match="No qualified players"):
            CompositeDefensiveRating().compute(rapm, pl.DataFrame(), season=2025)

    def test_season_filter(self) -> None:
        """When season column present, only the requested season is used."""
        rapm_2024 = _make_rapm_df(20, seed=1).with_columns(pl.lit(2024).alias("season"))
        rapm_2025 = _make_rapm_df(30, seed=2).with_columns(pl.lit(2025).alias("season"))
        combined  = pl.concat([rapm_2024, rapm_2025], how="diagonal")
        pbp = _make_pbp_stats_df(combined)
        cdr = CompositeDefensiveRating().compute(combined, pbp, season=2025)
        assert len(cdr) == 30

    def test_no_nan_in_output(self) -> None:
        rapm = _make_rapm_df(40)
        pbp  = _make_pbp_stats_df(rapm)
        cdr  = CompositeDefensiveRating().compute(rapm, pbp, season=2025)
        for col in ("cdr", "rapm_xga_z", "tk_gv_z", "xga_adj_z"):
            assert not cdr[col].is_nan().any(), f"NaN found in column '{col}'"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


class TestIO:
    def test_write_read_roundtrip(self) -> None:
        rapm = _make_rapm_df(30)
        pbp  = _make_pbp_stats_df(rapm)
        cdr  = CompositeDefensiveRating().compute(rapm, pbp, season=2025)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "cdr"
            path = write_cdr(cdr, out_dir, 2025)
            assert path.exists()
            loaded = read_cdr(Path(tmp), season=2025)
        assert loaded is not None
        assert len(loaded) == len(cdr)
        np.testing.assert_allclose(
            cdr["cdr"].to_numpy(), loaded["cdr"].to_numpy(), rtol=1e-5
        )

    def test_read_cdr_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = read_cdr(Path(tmp), season=2025)
        assert result is None

    def test_read_cdr_latest_when_no_season(self) -> None:
        rapm = _make_rapm_df(10)
        pbp  = _make_pbp_stats_df(rapm)
        cdr  = CompositeDefensiveRating().compute(rapm, pbp, season=2025)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "cdr"
            write_cdr(cdr, out_dir, 2025)
            loaded = read_cdr(Path(tmp))
        assert loaded is not None
        assert len(loaded) == 10


# ---------------------------------------------------------------------------
# lookup_player
# ---------------------------------------------------------------------------


class TestLookupPlayer:
    def _sample_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "player_name": ["Connor McDavid", "Leon Draisaitl", "Sidney Crosby"],
            "cdr":         [0.5, 0.3, 0.7],
        })

    def test_exact_match(self) -> None:
        df = self._sample_df()
        result = lookup_player(df, "Connor McDavid")
        assert len(result) == 1
        assert result["player_name"][0] == "Connor McDavid"

    def test_case_insensitive(self) -> None:
        df = self._sample_df()
        assert len(lookup_player(df, "connor mcdavid")) == 1
        assert len(lookup_player(df, "LEON DRAISAITL")) == 1
        assert len(lookup_player(df, "Sidney Crosby")) == 1

    def test_not_found_returns_empty(self) -> None:
        df = self._sample_df()
        result = lookup_player(df, "Nobody Here")
        assert len(result) == 0

    def test_strips_whitespace(self) -> None:
        df = self._sample_df()
        result = lookup_player(df, "  Connor McDavid  ")
        assert len(result) == 1
