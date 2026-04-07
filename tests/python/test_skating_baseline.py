"""Tests for Feature 2.20 — SkatingBaselineModel."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from models.skating_baseline import (
    SkatingBaselineModel,
    write_skating_baseline,
    SKATING_BASELINE_SCHEMA,
    MIN_REST_DAYS,
    MIN_GAMES_RESTED,
    MIN_GAMES_TOTAL,
    MAX_SPEED_KMH,
    MAX_DIST_KM,
    CONFIDENCE_RESTED,
    CONFIDENCE_AGGREGATE,
    CONFIDENCE_INSUFFICIENT,
    MODEL_VERSION,
    _clip_edge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_df(
    n: int = 20,
    seed: int = 0,
    season: int = 2025,
    include_rest: bool = False,
    rest_days_val: int = 2,
) -> pl.DataFrame:
    """Minimal EDGE skating DataFrame."""
    rng = np.random.default_rng(seed)
    data = {
        "player_id":            list(range(1, n + 1)),
        "player_name":          [f"player_{i}" for i in range(1, n + 1)],
        "team":                 rng.choice(["EDM", "TOR", "BOS", "VGK"], n).tolist(),
        "season":               [season] * n,
        "game_type":            [2] * n,
        "games_played":         rng.integers(20, 82, n).tolist(),
        "avg_speed_kmh":        rng.uniform(18.0, 22.0, n).tolist(),
        "max_speed_kmh":        rng.uniform(28.0, 35.0, n).tolist(),
        "total_distance_km":    rng.uniform(200.0, 500.0, n).tolist(),
        "distance_per_game_km": rng.uniform(3.5, 7.0, n).tolist(),
        "zone_time_pct_oz":     rng.uniform(0.25, 0.45, n).tolist(),
        "zone_time_pct_dz":     rng.uniform(0.25, 0.45, n).tolist(),
        "zone_time_pct_nz":     rng.uniform(0.15, 0.30, n).tolist(),
    }
    if include_rest:
        data["rest_days"] = [rest_days_val] * n
    return pl.DataFrame(data)


def _game_level_df(
    player_id: int = 1,
    n_games: int = 10,
    n_rested: int = 6,
    seed: int = 0,
    season: int = 2025,
) -> pl.DataFrame:
    """Game-level EDGE rows for a single player with rest_days column."""
    rng = np.random.default_rng(seed)
    rest = [2] * n_rested + [0] * (n_games - n_rested)
    return pl.DataFrame({
        "player_id":            [player_id] * n_games,
        "player_name":          ["Test Player"] * n_games,
        "team":                 ["EDM"] * n_games,
        "season":               [season] * n_games,
        "game_type":            [2] * n_games,
        "games_played":         [n_games] * n_games,
        "avg_speed_kmh":        rng.uniform(19.0, 21.0, n_games).tolist(),
        "max_speed_kmh":        rng.uniform(29.0, 33.0, n_games).tolist(),
        "total_distance_km":    rng.uniform(20.0, 30.0, n_games).tolist(),
        "distance_per_game_km": rng.uniform(4.0, 6.0, n_games).tolist(),
        "zone_time_pct_oz":     rng.uniform(0.30, 0.40, n_games).tolist(),
        "zone_time_pct_dz":     rng.uniform(0.25, 0.35, n_games).tolist(),
        "zone_time_pct_nz":     rng.uniform(0.20, 0.30, n_games).tolist(),
        "rest_days":            rest,
    })


# ---------------------------------------------------------------------------
# _clip_edge
# ---------------------------------------------------------------------------

class TestClipEdge:
    def test_clips_speed_above_max(self):
        df = pl.DataFrame({"avg_speed_kmh": [50.0], "max_speed_kmh": [60.0],
                            "distance_per_game_km": [5.0]})
        result = _clip_edge(df)
        assert result["avg_speed_kmh"][0] == pytest.approx(MAX_SPEED_KMH)
        assert result["max_speed_kmh"][0] == pytest.approx(MAX_SPEED_KMH)

    def test_clips_distance_above_max(self):
        df = pl.DataFrame({"distance_per_game_km": [15.0]})
        result = _clip_edge(df)
        assert result["distance_per_game_km"][0] == pytest.approx(MAX_DIST_KM)

    def test_clips_zone_pct_above_one(self):
        df = pl.DataFrame({"zone_time_pct_oz": [1.5], "zone_time_pct_dz": [-0.1],
                            "zone_time_pct_nz": [0.3]})
        result = _clip_edge(df)
        assert result["zone_time_pct_oz"][0] == pytest.approx(1.0)
        assert result["zone_time_pct_dz"][0] == pytest.approx(0.0)

    def test_valid_values_unchanged(self):
        df = pl.DataFrame({"avg_speed_kmh": [20.0], "distance_per_game_km": [5.0]})
        result = _clip_edge(df)
        assert result["avg_speed_kmh"][0] == pytest.approx(20.0)

    def test_missing_columns_ignored(self):
        df = pl.DataFrame({"player_id": [1]})
        result = _clip_edge(df)
        assert "player_id" in result.columns


# ---------------------------------------------------------------------------
# SkatingBaselineModel.fit — basic
# ---------------------------------------------------------------------------

class TestFitBasic:
    def test_returns_dataframe(self):
        model = SkatingBaselineModel()
        df = model.fit(_edge_df(10), season=2025)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 10

    def test_schema_columns_present(self):
        model = SkatingBaselineModel()
        df = model.fit(_edge_df(5), season=2025)
        for col in SKATING_BASELINE_SCHEMA:
            assert col in df.columns

    def test_season_column_set(self):
        model = SkatingBaselineModel()
        df = model.fit(_edge_df(5), season=2024)
        assert all(df["season"].to_list()[i] == 2024 for i in range(len(df)))

    def test_model_version_set(self):
        model = SkatingBaselineModel()
        df = model.fit(_edge_df(5), season=2025)
        assert df["model_version"][0] == MODEL_VERSION

    def test_empty_df_returns_empty_schema(self):
        model = SkatingBaselineModel()
        empty = pl.DataFrame(schema={
            "player_id": pl.Int64, "player_name": pl.Utf8, "team": pl.Utf8,
            "season": pl.Int64, "game_type": pl.Int64, "games_played": pl.Int64,
            "avg_speed_kmh": pl.Float64, "max_speed_kmh": pl.Float64,
            "total_distance_km": pl.Float64, "distance_per_game_km": pl.Float64,
            "zone_time_pct_oz": pl.Float64, "zone_time_pct_dz": pl.Float64,
            "zone_time_pct_nz": pl.Float64,
        })
        df = model.fit(empty, season=2025)
        for col in SKATING_BASELINE_SCHEMA:
            assert col in df.columns
        assert len(df) == 0

    def test_missing_player_id_raises(self):
        model = SkatingBaselineModel()
        bad = pl.DataFrame({"avg_speed_kmh": [20.0]})
        with pytest.raises(ValueError, match="player_id"):
            model.fit(bad, season=2025)

    def test_baseline_speed_in_plausible_range(self):
        model = SkatingBaselineModel()
        df = model.fit(_edge_df(20), season=2025)
        speeds = df["baseline_avg_speed_kmh"].drop_nulls().to_list()
        assert all(0 <= s <= MAX_SPEED_KMH for s in speeds)

    def test_baseline_distance_in_plausible_range(self):
        model = SkatingBaselineModel()
        df = model.fit(_edge_df(20), season=2025)
        dists = df["baseline_distance_per_game_km"].drop_nulls().to_list()
        assert all(0 <= d <= MAX_DIST_KM for d in dists)

    def test_zone_pcts_in_zero_one(self):
        model = SkatingBaselineModel()
        df = model.fit(_edge_df(20), season=2025)
        for col in ["baseline_zone_time_pct_oz", "baseline_zone_time_pct_dz",
                    "baseline_zone_time_pct_nz"]:
            vals = df[col].drop_nulls().to_list()
            assert all(0.0 <= v <= 1.0 for v in vals)


# ---------------------------------------------------------------------------
# Confidence levels
# ---------------------------------------------------------------------------

class TestConfidenceLevels:
    def test_rested_confidence_when_enough_rested_games(self):
        # n_rested=8 ≥ MIN_GAMES_RESTED=5 → confidence=rested
        df = _game_level_df(player_id=1, n_games=15, n_rested=8)
        model = SkatingBaselineModel(min_games_rested=5, min_games_total=3)
        result = model.fit(df, season=2025)
        row = result.filter(pl.col("player_id") == 1)
        assert row["confidence"][0] == CONFIDENCE_RESTED

    def test_aggregate_confidence_when_too_few_rested(self):
        # n_rested=2 < MIN_GAMES_RESTED=5, but n_total=15 ≥ MIN_GAMES_TOTAL=3
        df = _game_level_df(player_id=1, n_games=15, n_rested=2)
        model = SkatingBaselineModel(min_games_rested=5, min_games_total=3)
        result = model.fit(df, season=2025)
        row = result.filter(pl.col("player_id") == 1)
        assert row["confidence"][0] == CONFIDENCE_AGGREGATE

    def test_insufficient_when_too_few_total_games(self):
        # Only 2 rows < min_games_total=3
        df = _game_level_df(player_id=1, n_games=2, n_rested=2)
        model = SkatingBaselineModel(min_games_rested=5, min_games_total=3)
        result = model.fit(df, season=2025)
        row = result.filter(pl.col("player_id") == 1)
        assert row["confidence"][0] == CONFIDENCE_INSUFFICIENT

    def test_no_rest_column_uses_aggregate(self):
        # Without rest_days column → rested bucket is empty → aggregate
        df = _edge_df(10, include_rest=False)
        model = SkatingBaselineModel(min_games_total=3)
        result = model.fit(df, season=2025)
        # Without rest column, all must be aggregate (since rested=0 < min_games_rested)
        assert all(c in (CONFIDENCE_AGGREGATE, CONFIDENCE_INSUFFICIENT)
                   for c in result["confidence"].to_list())

    def test_n_games_rested_counted_correctly(self):
        df = _game_level_df(player_id=1, n_games=10, n_rested=6)
        model = SkatingBaselineModel(min_games_rested=5)
        result = model.fit(df, season=2025)
        row = result.filter(pl.col("player_id") == 1)
        assert row["n_games_rested"][0] == 6
        assert row["n_games_total"][0] == 10


# ---------------------------------------------------------------------------
# Rest filter threshold
# ---------------------------------------------------------------------------

class TestRestFilter:
    def test_rest_days_zero_excluded(self):
        # rest_days=0 (back-to-back) should not count as rested
        df = pl.DataFrame({
            "player_id":            [1] * 10,
            "player_name":          ["P"] * 10,
            "team":                 ["EDM"] * 10,
            "season":               [2025] * 10,
            "game_type":            [2] * 10,
            "games_played":         [10] * 10,
            "avg_speed_kmh":        [20.0] * 10,
            "max_speed_kmh":        [30.0] * 10,
            "total_distance_km":    [50.0] * 10,
            "distance_per_game_km": [5.0] * 10,
            "zone_time_pct_oz":     [0.35] * 10,
            "zone_time_pct_dz":     [0.30] * 10,
            "zone_time_pct_nz":     [0.25] * 10,
            "rest_days":            [0] * 10,  # all B2B
        })
        model = SkatingBaselineModel(min_rest_days=1, min_games_rested=5, min_games_total=3)
        result = model.fit(df, season=2025)
        row = result.filter(pl.col("player_id") == 1)
        # 0 rested days < threshold=1 → no rested rows → aggregate
        assert row["confidence"][0] == CONFIDENCE_AGGREGATE
        assert row["n_games_rested"][0] == 0

    def test_rest_days_exactly_min_excluded(self):
        # min_rest_days=1: rest_days must be STRICTLY > 1 to count
        df = _game_level_df(player_id=1, n_games=10, n_rested=0)
        df = df.with_columns(pl.lit(1).alias("rest_days"))
        model = SkatingBaselineModel(min_rest_days=1, min_games_rested=5, min_games_total=3)
        result = model.fit(df, season=2025)
        row = result.filter(pl.col("player_id") == 1)
        assert row["n_games_rested"][0] == 0


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

class TestLookupHelpers:
    def _trained_model(self):
        model = SkatingBaselineModel()
        df = _edge_df(5, seed=42)
        model.fit(df, season=2025)
        return model

    def test_get_player_baseline_returns_dict(self):
        model = self._trained_model()
        b = model.get_player_baseline(1)
        assert isinstance(b, dict)
        assert "baseline_avg_speed_kmh" in b

    def test_get_player_baseline_unknown_returns_none(self):
        model = self._trained_model()
        assert model.get_player_baseline(99999) is None

    def test_speed_ceiling_returns_float(self):
        model = self._trained_model()
        s = model.speed_ceiling(1)
        assert s is None or isinstance(s, float)

    def test_distance_ceiling_returns_float(self):
        model = self._trained_model()
        d = model.distance_ceiling(1)
        assert d is None or isinstance(d, float)

    def test_zone_profile_returns_dict(self):
        model = self._trained_model()
        z = model.zone_profile(1)
        assert z is not None
        assert set(z.keys()) == {"oz", "dz", "nz"}

    def test_zone_profile_unknown_returns_none(self):
        model = self._trained_model()
        assert model.zone_profile(99999) is None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_returns_expected_keys(self):
        model = SkatingBaselineModel()
        df = model.fit(_edge_df(10), season=2025)
        s = model.summary(df)
        for key in ["n_players", "n_rested", "n_aggregate", "n_insufficient",
                    "mean_speed", "mean_distance", "mean_oz_pct"]:
            assert key in s

    def test_n_players_correct(self):
        model = SkatingBaselineModel()
        df = model.fit(_edge_df(10), season=2025)
        s = model.summary(df)
        assert s["n_players"] == 10

    def test_empty_df_summary(self):
        model = SkatingBaselineModel()
        empty = pl.DataFrame(schema={col: dt for col, dt in SKATING_BASELINE_SCHEMA.items()})
        s = model.summary(empty)
        assert s["n_players"] == 0
        assert s["mean_speed"] is None

    def test_confidence_counts_sum_to_n_players(self):
        model = SkatingBaselineModel()
        df = model.fit(_edge_df(15), season=2025)
        s = model.summary(df)
        total = s["n_rested"] + s["n_aggregate"] + s["n_insufficient"]
        assert total == s["n_players"]

    def test_mean_speed_in_plausible_range(self):
        model = SkatingBaselineModel()
        df = model.fit(_edge_df(20), season=2025)
        s = model.summary(df)
        if s["mean_speed"] is not None:
            assert 0 < s["mean_speed"] <= MAX_SPEED_KMH


# ---------------------------------------------------------------------------
# Multi-season data
# ---------------------------------------------------------------------------

class TestMultiSeason:
    def test_multi_season_concat_produces_per_player_row(self):
        # Same player in 2024 and 2025 → model.fit on concat should use all rows
        df_2024 = _game_level_df(player_id=1, n_games=8, n_rested=5, season=2024)
        df_2025 = _game_level_df(player_id=1, n_games=10, n_rested=6, season=2025)
        combined = pl.concat([df_2024, df_2025])
        model = SkatingBaselineModel(min_games_rested=5, min_games_total=3)
        result = model.fit(combined, season=2025)
        # One row per player_id (player 1)
        assert len(result.filter(pl.col("player_id") == 1)) == 1

    def test_multi_season_total_games_accumulated(self):
        df_2024 = _game_level_df(player_id=1, n_games=8, n_rested=5, season=2024)
        df_2025 = _game_level_df(player_id=1, n_games=10, n_rested=6, season=2025)
        combined = pl.concat([df_2024, df_2025])
        model = SkatingBaselineModel(min_games_rested=5, min_games_total=3)
        result = model.fit(combined, season=2025)
        row = result.filter(pl.col("player_id") == 1)
        assert row["n_games_total"][0] == 18  # 8 + 10
        assert row["n_games_rested"][0] == 11  # 5 + 6


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        model = SkatingBaselineModel(
            min_rest_days=2, min_games_rested=8, min_games_total=5,
            max_speed_kmh=38.0, max_dist_km=9.0,
        )
        df = _edge_df(10)
        model.fit(df, season=2025)
        path = tmp_path / "skate.pkl"
        model.save(path)
        m2 = SkatingBaselineModel.load(path)
        assert m2._min_rest == 2
        assert m2._min_rested == 8
        assert m2._min_total == 5
        assert m2._max_speed == pytest.approx(38.0)
        assert m2._max_dist == pytest.approx(9.0)
        assert len(m2._baselines) == 10

    def test_loaded_model_lookup_works(self, tmp_path):
        model = SkatingBaselineModel()
        df = _edge_df(5, seed=7)
        model.fit(df, season=2025)
        path = tmp_path / "skate2.pkl"
        model.save(path)
        m2 = SkatingBaselineModel.load(path)
        b = m2.get_player_baseline(1)
        assert b is not None
        assert "baseline_avg_speed_kmh" in b

    def test_write_skating_baseline(self, tmp_path):
        model = SkatingBaselineModel()
        df = model.fit(_edge_df(10), season=2025)
        path = write_skating_baseline(df, tmp_path, 2025)
        assert path.exists()
        loaded = pl.read_parquet(path)
        assert "player_id" in loaded.columns
        assert "baseline_avg_speed_kmh" in loaded.columns

    def test_write_filename_contains_season(self, tmp_path):
        model = SkatingBaselineModel()
        df = model.fit(_edge_df(5), season=2024)
        path = write_skating_baseline(df, tmp_path, 2024)
        assert "2024" in path.name
