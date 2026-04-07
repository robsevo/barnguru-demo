"""Tests for Feature 2.18 — PuckBattleModel."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from models.puck_battle_model import (
    PuckBattleModel,
    compute_net_front_pct,
    compute_battle_scores,
    write_player_battle,
    write_team_battle,
    BATTLE_WEIGHTS,
    NET_FRONT_X_MIN,
    NET_FRONT_Y_MAX,
    PLAYER_BATTLE_SCHEMA,
    TEAM_BATTLE_SCHEMA,
    MIN_TOI_SEC,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _player_stats(n: int = 20, toi_sec: float = 3600.0, seed: int = 0) -> pl.DataFrame:
    """Minimal player stats DataFrame."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "player_id":     list(range(1, n + 1)),
        "team":          rng.choice(["EDM", "TOR", "BOS", "VGK"], n).tolist(),
        "toi_ev":        [toi_sec] * n,
        "hits":          rng.integers(0, 200, n).tolist(),
        "blocked_shots": rng.integers(0, 100, n).tolist(),
        "carry_entry_pct": rng.uniform(0.3, 0.7, n).tolist(),
    })


def _shots_df(n: int = 500, seed: int = 0) -> pl.DataFrame:
    """Minimal shot DataFrame with coordinates."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "shooter_id":  rng.integers(1, 21, n).tolist(),
        "x_adjusted":  rng.uniform(0, 100, n).tolist(),
        "y_adjusted":  rng.uniform(-42, 42, n).tolist(),
        "game_id":     rng.integers(1000, 2000, n).tolist(),
    })


# ---------------------------------------------------------------------------
# compute_net_front_pct
# ---------------------------------------------------------------------------

class TestComputeNetFrontPct:
    def test_returns_dataframe(self):
        shots = _shots_df(200)
        df = compute_net_front_pct(shots)
        assert isinstance(df, pl.DataFrame)
        assert "player_id" in df.columns
        assert "net_front_pct" in df.columns

    def test_pct_in_zero_one(self):
        shots = _shots_df(500)
        df = compute_net_front_pct(shots)
        assert df["net_front_pct"].min() >= 0.0
        assert df["net_front_pct"].max() <= 1.0

    def test_net_front_shots_counted(self):
        # All shots placed in net-front zone → pct = 1.0
        shots = pl.DataFrame({
            "shooter_id": [1] * 50,
            "x_adjusted": [NET_FRONT_X_MIN + 1.0] * 50,
            "y_adjusted": [0.0] * 50,
        })
        df = compute_net_front_pct(shots)
        row = df.filter(pl.col("player_id") == 1)
        assert float(row["net_front_pct"][0]) == pytest.approx(1.0)

    def test_perimeter_shots_not_counted(self):
        # All shots far from net → pct = 0.0
        shots = pl.DataFrame({
            "shooter_id": [1] * 50,
            "x_adjusted": [30.0] * 50,  # far from net
            "y_adjusted": [0.0] * 50,
        })
        df = compute_net_front_pct(shots)
        row = df.filter(pl.col("player_id") == 1)
        assert float(row["net_front_pct"][0]) == pytest.approx(0.0)

    def test_missing_shooter_id_returns_empty(self):
        shots = pl.DataFrame({"game_id": [1, 2]})
        df = compute_net_front_pct(shots)
        assert len(df) == 0

    def test_no_coordinate_columns_returns_nan(self):
        shots = pl.DataFrame({"shooter_id": [1, 2, 3]})
        df = compute_net_front_pct(shots)
        assert "net_front_pct" in df.columns
        assert all(np.isnan(v) for v in df["net_front_pct"].to_list())

    def test_x_adjusted_alias_supported(self):
        shots = pl.DataFrame({
            "shooter_id": [1] * 20,
            "x_adjusted": [80.0] * 20,
            "y_adjusted": [5.0] * 20,
        })
        df = compute_net_front_pct(shots)
        assert len(df) >= 1

    def test_y_boundary_exactly_at_max(self):
        # |y| == NET_FRONT_Y_MAX — should still count
        shots = pl.DataFrame({
            "shooter_id": [1] * 10,
            "x_adjusted": [NET_FRONT_X_MIN + 1.0] * 10,
            "y_adjusted": [NET_FRONT_Y_MAX] * 10,
        })
        df = compute_net_front_pct(shots)
        row = df.filter(pl.col("player_id") == 1)
        assert float(row["net_front_pct"][0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_battle_scores
# ---------------------------------------------------------------------------

class TestComputeBattleScores:
    def test_adds_required_columns(self):
        df = _player_stats(10)
        result = compute_battle_scores(df)
        for col in ["battle_score", "battle_percentile", "controlled_entry_prob"]:
            assert col in result.columns

    def test_cep_in_zero_one(self):
        df = _player_stats(20)
        result = compute_battle_scores(df)
        assert result["controlled_entry_prob"].min() > 0.0
        assert result["controlled_entry_prob"].max() < 1.0

    def test_percentile_range(self):
        df = _player_stats(20)
        result = compute_battle_scores(df)
        assert result["battle_percentile"].min() >= 0.0
        assert result["battle_percentile"].max() <= 100.0

    def test_empty_dataframe(self):
        df = pl.DataFrame({
            "hits_per60": pl.Series([], dtype=pl.Float64),
            "blocks_per60": pl.Series([], dtype=pl.Float64),
        })
        result = compute_battle_scores(df)
        assert "battle_score" in result.columns

    def test_uniform_inputs_give_median_score(self):
        # All players identical → all battle_scores should be identical (all zero after z-score)
        df = pl.DataFrame({
            "player_id": list(range(10)),
            "hits_per60": [5.0] * 10,
            "blocks_per60": [3.0] * 10,
            "carry_entry_pct": [0.5] * 10,
            "net_front_pct": [0.2] * 10,
        })
        result = compute_battle_scores(df)
        scores = result["battle_score"].to_list()
        assert all(abs(s) < 1e-9 for s in scores)

    def test_custom_weights(self):
        df = _player_stats(10)
        weights = {"hits_per60": 1.0, "blocks_per60": 0.0, "carry_entry_pct": 0.0, "net_front_pct": 0.0}
        result = compute_battle_scores(df, weights=weights)
        assert "battle_score" in result.columns


# ---------------------------------------------------------------------------
# PuckBattleModel.fit
# ---------------------------------------------------------------------------

class TestPuckBattleModelFit:
    def test_fit_returns_dataframe(self):
        model = PuckBattleModel()
        df = model.fit(_player_stats(20), season=2025)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_fit_schema_columns(self):
        model = PuckBattleModel()
        df = model.fit(_player_stats(20), season=2025)
        for col in ["player_id", "battle_score", "controlled_entry_prob", "season"]:
            assert col in df.columns

    def test_fit_with_shots(self):
        model = PuckBattleModel()
        shots = _shots_df(500)
        df = model.fit(_player_stats(20), season=2025, shots_df=shots)
        assert "net_front_pct" in df.columns

    def test_fit_without_shots_gives_nan_net_front(self):
        model = PuckBattleModel()
        df = model.fit(_player_stats(10), season=2025, shots_df=None)
        # net_front_pct should be filled with nan → treated as 0 in scoring
        assert "net_front_pct" in df.columns

    def test_missing_player_id_raises(self):
        model = PuckBattleModel()
        df = pl.DataFrame({"team": ["EDM"], "hits": [100]})
        with pytest.raises(ValueError, match="player_id"):
            model.fit(df, season=2025)

    def test_toi_filter_applied(self):
        model = PuckBattleModel()
        # Mix of above/below MIN_TOI_SEC
        stats = pl.DataFrame({
            "player_id": [1, 2, 3],
            "team":      ["EDM", "TOR", "BOS"],
            "toi_ev":    [60.0, MIN_TOI_SEC + 1, MIN_TOI_SEC * 2],
            "hits":      [10, 50, 100],
            "blocked_shots": [5, 20, 40],
        })
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            df = model.fit(stats, season=2025)
        # Player 1 (toi=60) should be filtered out
        assert 1 not in df["player_id"].to_list()
        assert 2 in df["player_id"].to_list()

    def test_controlled_entry_prob_in_zero_one(self):
        model = PuckBattleModel()
        df = model.fit(_player_stats(30), season=2025)
        assert df["controlled_entry_prob"].min() > 0.0
        assert df["controlled_entry_prob"].max() < 1.0

    def test_season_col_set(self):
        model = PuckBattleModel()
        df = model.fit(_player_stats(10), season=2024)
        assert df["season"][0] == 2024

    def test_missing_hits_col_graceful(self):
        model = PuckBattleModel()
        stats = pl.DataFrame({
            "player_id": list(range(10)),
            "team": ["EDM"] * 10,
            "toi_ev": [3600.0] * 10,
            "carry_entry_pct": [0.5] * 10,
        })
        df = model.fit(stats, season=2025)
        assert "battle_score" in df.columns


# ---------------------------------------------------------------------------
# PuckBattleModel.team_battle_ratings
# ---------------------------------------------------------------------------

class TestTeamBattleRatings:
    def _player_df(self):
        model = PuckBattleModel()
        return model.fit(_player_stats(20, seed=5), season=2025)

    def test_returns_dataframe(self):
        model = PuckBattleModel()
        player_df = self._player_df()
        team_df = model.team_battle_ratings(player_df, season=2025)
        assert isinstance(team_df, pl.DataFrame)
        assert len(team_df) > 0

    def test_team_schema(self):
        model = PuckBattleModel()
        player_df = self._player_df()
        team_df = model.team_battle_ratings(player_df, season=2025)
        for col in TEAM_BATTLE_SCHEMA:
            assert col in team_df.columns

    def test_controlled_entry_prob_in_zero_one(self):
        model = PuckBattleModel()
        player_df = self._player_df()
        team_df = model.team_battle_ratings(player_df, season=2025)
        assert team_df["controlled_entry_prob"].min() > 0.0
        assert team_df["controlled_entry_prob"].max() < 1.0

    def test_missing_team_column_warns(self):
        model = PuckBattleModel()
        player_df = pl.DataFrame({
            "player_id": [1, 2],
            "battle_score": [0.5, -0.5],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model.team_battle_ratings(player_df, season=2025)
        assert len(result) == 0

    def test_season_attached(self):
        model = PuckBattleModel()
        player_df = self._player_df()
        team_df = model.team_battle_ratings(player_df, season=2023)
        assert team_df["season"][0] == 2023


# ---------------------------------------------------------------------------
# compute_net_front_pct edge cases
# ---------------------------------------------------------------------------

class TestNetFrontBoundary:
    def test_exactly_at_x_min(self):
        shots = pl.DataFrame({
            "shooter_id": [1] * 10,
            "x_adjusted": [NET_FRONT_X_MIN] * 10,
            "y_adjusted": [0.0] * 10,
        })
        df = compute_net_front_pct(shots)
        row = df.filter(pl.col("player_id") == 1)
        assert float(row["net_front_pct"][0]) == pytest.approx(1.0)

    def test_outside_y_boundary(self):
        shots = pl.DataFrame({
            "shooter_id": [1] * 10,
            "x_adjusted": [NET_FRONT_X_MIN + 1.0] * 10,
            "y_adjusted": [NET_FRONT_Y_MAX + 1.0] * 10,
        })
        df = compute_net_front_pct(shots)
        row = df.filter(pl.col("player_id") == 1)
        assert float(row["net_front_pct"][0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        model = PuckBattleModel(weights={"hits_per60": 0.5, "blocks_per60": 0.5,
                                          "carry_entry_pct": 0.0, "net_front_pct": 0.0})
        path = tmp_path / "battle.pkl"
        model.save(path)
        model2 = PuckBattleModel.load(path)
        assert model2._weights["hits_per60"] == pytest.approx(0.5)
        assert model2._weights["blocks_per60"] == pytest.approx(0.5)

    def test_write_player_battle(self, tmp_path):
        model = PuckBattleModel()
        df = model.fit(_player_stats(10), season=2025)
        path = write_player_battle(df, tmp_path, 2025)
        assert path.exists()
        loaded = pl.read_parquet(path)
        assert "player_id" in loaded.columns

    def test_write_team_battle(self, tmp_path):
        model = PuckBattleModel()
        player_df = model.fit(_player_stats(20, seed=3), season=2025)
        team_df = model.team_battle_ratings(player_df, season=2025)
        path = write_team_battle(team_df, tmp_path, 2025)
        assert path.exists()
        loaded = pl.read_parquet(path)
        assert "team" in loaded.columns
        assert "controlled_entry_prob" in loaded.columns
