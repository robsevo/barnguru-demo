"""Tests for Feature 2.21 — PositionalHeatmapModel."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from models.positional_heatmap import (
    PositionalHeatmapModel,
    classify_shot_zone,
    classify_shots_batch,
    write_positional_heatmap,
    POSITIONAL_HEATMAP_SCHEMA,
    ZONE_X_NET_FRONT,
    ZONE_Y_NET_FRONT,
    ZONE_X_INNER,
    ZONE_Y_SLOT,
    ZONE_Y_CIRCLE,
    ZONE_X_BLUE_LINE,
    MIN_SHOTS_FULL,
    MIN_SHOTS_PARTIAL,
    CONFIDENCE_FULL,
    CONFIDENCE_PARTIAL,
    CONFIDENCE_INSUFFICIENT,
    MODEL_VERSION,
)
from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shots_df(
    n: int = 100,
    player_id: int = 1,
    seed: int = 0,
    x_col: str = "arena_adj_x",
    y_col: str = "arena_adj_y",
    player_col: str = "shooter_id",
) -> pl.DataFrame:
    """Minimal shots DataFrame with arena-adjusted coordinates."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        player_col:    [player_id] * n,
        "player_name": [f"Player {player_id}"] * n,
        "team":        ["EDM"] * n,
        x_col:         rng.uniform(20.0, 89.0, n).tolist(),
        y_col:         rng.uniform(-42.0, 42.0, n).tolist(),
    })


def _net_front_shots(n: int = 30, player_id: int = 1) -> pl.DataFrame:
    """All shots from net-front zone."""
    return pl.DataFrame({
        "shooter_id":  [player_id] * n,
        "arena_adj_x": [80.0] * n,
        "arena_adj_y": [0.0] * n,
    })


def _point_shots(n: int = 30, player_id: int = 1) -> pl.DataFrame:
    """All shots from the point (defenseman shots)."""
    return pl.DataFrame({
        "shooter_id":  [player_id] * n,
        "arena_adj_x": [20.0] * n,
        "arena_adj_y": [0.0] * n,
    })


def _multi_player_shots(n_per_player: int = 30) -> pl.DataFrame:
    """Three players, each with n shots in distinct zones."""
    rng = np.random.default_rng(99)
    rows = []
    for pid, (x, y) in [(1, (80.0, 0.0)), (2, (20.0, 0.0)), (3, (60.0, 25.0))]:
        rows.extend([{
            "shooter_id":  pid,
            "player_name": f"Player {pid}",
            "team":        "EDM",
            "arena_adj_x": x + rng.uniform(-1, 1),
            "arena_adj_y": y + rng.uniform(-1, 1),
        } for _ in range(n_per_player)])
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# classify_shot_zone
# ---------------------------------------------------------------------------

class TestClassifyShotZone:
    def test_net_front(self):
        assert classify_shot_zone(ZONE_X_NET_FRONT, 0.0) == "net_front"
        assert classify_shot_zone(85.0, 5.0) == "net_front"

    def test_slot(self):
        # In slot: x ≥ 54, |y| ≤ 22, but not net_front
        assert classify_shot_zone(60.0, 10.0) == "slot"
        assert classify_shot_zone(ZONE_X_INNER, ZONE_Y_SLOT) == "slot"

    def test_circle(self):
        # x ≥ 54, 22 < |y| ≤ 36
        assert classify_shot_zone(60.0, 25.0) == "circle"
        assert classify_shot_zone(ZONE_X_INNER, ZONE_Y_CIRCLE) == "circle"

    def test_corner(self):
        # x ≥ 54, |y| > 36
        assert classify_shot_zone(60.0, 40.0) == "corner"
        assert classify_shot_zone(89.0, 38.0) == "corner"

    def test_half_wall(self):
        # 25 ≤ x < 54, any y
        assert classify_shot_zone(40.0, 0.0) == "half_wall"
        assert classify_shot_zone(ZONE_X_BLUE_LINE, 20.0) == "half_wall"

    def test_point(self):
        # x < 25
        assert classify_shot_zone(20.0, 0.0) == "point"
        assert classify_shot_zone(0.0, 0.0) == "point"

    def test_negative_x_is_point_fallback(self):
        assert classify_shot_zone(-10.0, 0.0) == "point"

    def test_boundary_net_front_y(self):
        # |y| == ZONE_Y_NET_FRONT exactly → net_front
        assert classify_shot_zone(80.0, ZONE_Y_NET_FRONT) == "net_front"

    def test_boundary_net_front_x(self):
        # x == ZONE_X_NET_FRONT, |y| ≤ threshold → net_front
        assert classify_shot_zone(ZONE_X_NET_FRONT, 5.0) == "net_front"

    def test_just_outside_net_front_y(self):
        # |y| > ZONE_Y_NET_FRONT → slot (if x still in slot range)
        assert classify_shot_zone(80.0, ZONE_Y_NET_FRONT + 1) == "slot"


# ---------------------------------------------------------------------------
# classify_shots_batch
# ---------------------------------------------------------------------------

class TestClassifyShotsBatch:
    def test_same_result_as_scalar(self):
        xs = np.array([80.0, 60.0, 40.0, 20.0])
        ys = np.array([0.0, 25.0, 0.0, 0.0])
        batch = classify_shots_batch(xs, ys)
        scalar = [classify_shot_zone(x, y) for x, y in zip(xs, ys)]
        assert list(batch) == scalar

    def test_all_zones_represented(self):
        xs = np.array([80.0, 60.0, 60.0, 60.0, 40.0, 20.0])
        ys = np.array([5.0,  10.0, 25.0, 38.0, 0.0,  0.0])
        result = classify_shots_batch(xs, ys)
        assert "net_front" in result
        assert "slot"      in result
        assert "circle"    in result
        assert "corner"    in result
        assert "half_wall" in result
        assert "point"     in result

    def test_nan_coords_do_not_crash(self):
        xs = np.array([float("nan"), 80.0])
        ys = np.array([0.0, float("nan")])
        # NaN coords treated as (0, 0) → point
        xs_safe = np.where(np.isnan(xs), 0.0, xs)
        ys_safe = np.where(np.isnan(ys), 0.0, ys)
        result = classify_shots_batch(xs_safe, ys_safe)
        assert result[0] == "point"


# ---------------------------------------------------------------------------
# PositionalHeatmapModel.fit — basic
# ---------------------------------------------------------------------------

class TestFitBasic:
    def test_returns_dataframe(self):
        model = PositionalHeatmapModel()
        df = model.fit(_shots_df(50), season=2025)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1

    def test_schema_columns_present(self):
        model = PositionalHeatmapModel()
        df = model.fit(_shots_df(50), season=2025)
        for col in POSITIONAL_HEATMAP_SCHEMA:
            assert col in df.columns

    def test_season_attached(self):
        model = PositionalHeatmapModel()
        df = model.fit(_shots_df(30), season=2024)
        assert df["season"][0] == 2024

    def test_model_version_present(self):
        model = PositionalHeatmapModel()
        df = model.fit(_shots_df(30), season=2025)
        assert df["model_version"][0] == MODEL_VERSION

    def test_zone_pcts_sum_to_one(self):
        model = PositionalHeatmapModel()
        df = model.fit(_shots_df(50), season=2025)
        zone_cols = ["off_net_front_pct", "off_slot_pct", "off_circle_pct",
                     "off_half_wall_pct", "off_point_pct", "off_corner_pct"]
        total = sum(df[c][0] for c in zone_cols)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_net_front_only_shots(self):
        model = PositionalHeatmapModel()
        df = model.fit(_net_front_shots(30), season=2025)
        assert df["off_net_front_pct"][0] == pytest.approx(1.0)
        assert df["off_slot_pct"][0] == pytest.approx(0.0)

    def test_point_only_shots(self):
        model = PositionalHeatmapModel()
        df = model.fit(_point_shots(30), season=2025)
        assert df["off_point_pct"][0] == pytest.approx(1.0)
        assert df["off_net_front_pct"][0] == pytest.approx(0.0)

    def test_n_shots_correct(self):
        model = PositionalHeatmapModel()
        df = model.fit(_shots_df(47), season=2025)
        assert df["n_shots"][0] == 47

    def test_multiple_players(self):
        model = PositionalHeatmapModel()
        df = model.fit(_multi_player_shots(30), season=2025)
        assert len(df) == 3

    def test_player_id_column_alias(self):
        model = PositionalHeatmapModel()
        shots = _shots_df(30, player_col="player_id")
        df = model.fit(shots, season=2025)
        assert len(df) == 1

    def test_x_column_alias_x_adjusted(self):
        model = PositionalHeatmapModel()
        shots = _shots_df(30, x_col="x_adjusted", y_col="y_adjusted")
        df = model.fit(shots, season=2025)
        assert len(df) == 1

    def test_empty_shots_returns_empty_schema(self):
        model = PositionalHeatmapModel()
        empty = pl.DataFrame({
            "shooter_id":  pl.Series([], dtype=pl.Int64),
            "arena_adj_x": pl.Series([], dtype=pl.Float64),
            "arena_adj_y": pl.Series([], dtype=pl.Float64),
        })
        df = model.fit(empty, season=2025)
        for col in POSITIONAL_HEATMAP_SCHEMA:
            assert col in df.columns
        assert len(df) == 0

    def test_missing_player_col_raises(self):
        model = PositionalHeatmapModel()
        bad = pl.DataFrame({"arena_adj_x": [80.0], "arena_adj_y": [0.0]})
        with pytest.raises(ValueError, match="player identifier"):
            model.fit(bad, season=2025)

    def test_missing_coord_cols_warns_empty(self):
        model = PositionalHeatmapModel()
        bad = pl.DataFrame({"shooter_id": [1, 2]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = model.fit(bad, season=2025)
        assert len(df) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)


# ---------------------------------------------------------------------------
# Confidence levels
# ---------------------------------------------------------------------------

class TestConfidenceLevels:
    def test_full_confidence_at_min_shots_full(self):
        model = PositionalHeatmapModel(min_shots_full=20, min_shots_partial=5)
        df = model.fit(_net_front_shots(20), season=2025)
        assert df["confidence"][0] == CONFIDENCE_FULL

    def test_partial_confidence_below_full(self):
        model = PositionalHeatmapModel(min_shots_full=20, min_shots_partial=5)
        df = model.fit(_net_front_shots(10), season=2025)
        assert df["confidence"][0] == CONFIDENCE_PARTIAL

    def test_insufficient_below_partial(self):
        model = PositionalHeatmapModel(min_shots_full=20, min_shots_partial=5)
        df = model.fit(_net_front_shots(3), season=2025)
        assert df["confidence"][0] == CONFIDENCE_INSUFFICIENT

    def test_confidence_full_exactly(self):
        model = PositionalHeatmapModel(min_shots_full=20, min_shots_partial=5)
        # Exactly 20 shots → full
        df = model.fit(_net_front_shots(MIN_SHOTS_FULL), season=2025)
        assert df["confidence"][0] == CONFIDENCE_FULL

    def test_confidence_partial_exactly(self):
        model = PositionalHeatmapModel(min_shots_full=20, min_shots_partial=5)
        df = model.fit(_net_front_shots(MIN_SHOTS_PARTIAL), season=2025)
        assert df["confidence"][0] == CONFIDENCE_PARTIAL


# ---------------------------------------------------------------------------
# EDGE zone time attachment
# ---------------------------------------------------------------------------

class TestEdgeZoneTime:
    def _edge_df(self, player_id: int = 1) -> pl.DataFrame:
        return pl.DataFrame({
            "player_id":        [player_id],
            "zone_time_pct_oz": [0.40],
            "zone_time_pct_dz": [0.35],
            "zone_time_pct_nz": [0.25],
        })

    def test_zone_time_attached(self):
        model = PositionalHeatmapModel()
        shots = _net_front_shots(30, player_id=1)
        df = model.fit(shots, season=2025, edge_df=self._edge_df(1))
        assert df["zone_time_oz"][0] == pytest.approx(0.40)
        assert df["zone_time_dz"][0] == pytest.approx(0.35)
        assert df["zone_time_nz"][0] == pytest.approx(0.25)

    def test_no_edge_df_zone_time_null(self):
        model = PositionalHeatmapModel()
        df = model.fit(_net_front_shots(30, player_id=1), season=2025, edge_df=None)
        assert df["zone_time_oz"][0] is None or df["zone_time_oz"].is_null().all()

    def test_edge_player_id_mismatch_zone_null(self):
        model = PositionalHeatmapModel()
        shots = _net_front_shots(30, player_id=1)
        # Edge data is for player 99, not player 1
        edge = self._edge_df(player_id=99)
        df = model.fit(shots, season=2025, edge_df=edge)
        assert df["zone_time_oz"][0] is None or df["zone_time_oz"].is_null().any()


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

class TestLookupHelpers:
    def _trained(self):
        model = PositionalHeatmapModel()
        model.fit(_net_front_shots(30, player_id=1), season=2025)
        return model

    def test_get_player_profile_returns_dict(self):
        model = self._trained()
        profile = model.get_player_profile(1)
        assert isinstance(profile, dict)
        assert "off_net_front_pct" in profile

    def test_get_player_profile_unknown_returns_none(self):
        model = self._trained()
        assert model.get_player_profile(99999) is None

    def test_offensive_zone_distribution(self):
        model = self._trained()
        dist = model.offensive_zone_distribution(1)
        assert isinstance(dist, dict)
        assert set(dist.keys()) == {"net_front", "slot", "circle",
                                     "half_wall", "point", "corner"}
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_offensive_zone_distribution_unknown_player(self):
        model = self._trained()
        assert model.offensive_zone_distribution(99999) is None

    def test_high_danger_pct_net_front_player(self):
        model = self._trained()
        # All shots from net-front → high_danger_pct = 1.0
        hd = model.high_danger_pct(1)
        assert hd == pytest.approx(1.0)

    def test_high_danger_pct_point_player(self):
        model = PositionalHeatmapModel()
        model.fit(_point_shots(30, player_id=5), season=2025)
        hd = model.high_danger_pct(5)
        assert hd == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_keys(self):
        model = PositionalHeatmapModel()
        df = model.fit(_shots_df(50), season=2025)
        s = model.summary(df)
        for key in ["n_players", "n_full", "n_partial",
                    "n_insufficient", "mean_high_danger_pct"]:
            assert key in s

    def test_n_players_correct(self):
        model = PositionalHeatmapModel()
        df = model.fit(_multi_player_shots(30), season=2025)
        s = model.summary(df)
        assert s["n_players"] == 3

    def test_empty_summary(self):
        model = PositionalHeatmapModel()
        empty = pl.DataFrame(schema={col: dt for col, dt in POSITIONAL_HEATMAP_SCHEMA.items()})
        s = model.summary(empty)
        assert s["n_players"] == 0
        assert s["mean_high_danger_pct"] is None

    def test_confidence_counts_sum_to_n_players(self):
        model = PositionalHeatmapModel()
        df = model.fit(_multi_player_shots(30), season=2025)
        s = model.summary(df)
        assert s["n_full"] + s["n_partial"] + s["n_insufficient"] == s["n_players"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        model = PositionalHeatmapModel(min_shots_full=15, min_shots_partial=3)
        model.fit(_net_front_shots(30, player_id=1), season=2025)
        path = tmp_path / "heatmap.pkl"
        model.save(path)
        m2 = PositionalHeatmapModel.load(path)
        assert m2._min_full == 15
        assert m2._min_partial == 3
        assert 1 in m2._profiles

    def test_loaded_lookup_works(self, tmp_path):
        model = PositionalHeatmapModel()
        model.fit(_net_front_shots(30, player_id=42), season=2025)
        path = tmp_path / "hm2.pkl"
        model.save(path)
        m2 = PositionalHeatmapModel.load(path)
        profile = m2.get_player_profile(42)
        assert profile is not None
        assert profile["off_net_front_pct"] == pytest.approx(1.0)

    def test_write_positional_heatmap(self, tmp_path):
        model = PositionalHeatmapModel()
        df = model.fit(_shots_df(50), season=2025)
        path = write_positional_heatmap(df, tmp_path, 2025)
        assert path.exists()
        loaded = pl.read_parquet(path)
        assert "player_id" in loaded.columns
        assert "off_net_front_pct" in loaded.columns

    def test_write_filename_contains_season(self, tmp_path):
        model = PositionalHeatmapModel()
        df = model.fit(_shots_df(30), season=2023)
        path = write_positional_heatmap(df, tmp_path, 2023)
        assert "2023" in path.name
