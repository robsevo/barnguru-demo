"""Tests for PlayerArchetypeModel (Feature 2.11)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from models.archetype_model import (
    DEFAULT_K,
    MIN_K,
    MAX_K,
    PLAYER_ARCHETYPE_SCHEMA,
    PlayerArchetypeModel,
    _assign_archetype_labels,
    write_archetypes,
    read_archetypes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rapm_df(n: int = 50, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "player_id":    list(range(1000, 1000 + n)),
        "player_name":  [f"Player{i}" for i in range(n)],
        "season":       [2025] * n,
        "games_played": rng.integers(30, 82, n).tolist(),
        "off_rapm":     rng.normal(0.0, 1.0, n).tolist(),
        "def_rapm":     rng.normal(0.0, 0.7, n).tolist(),
        "pp_rapm":      rng.normal(0.0, 0.5, n).tolist(),
        "pk_rapm":      rng.normal(0.0, 0.5, n).tolist(),
        "xgf_per60":    rng.normal(2.5, 0.6, n).clip(0.5, 5.5).tolist(),
        "shots_per60":  rng.normal(8.0, 2.0, n).clip(1, 20).tolist(),
    })


@pytest.fixture
def rapm_df():
    return _make_rapm_df(50)


@pytest.fixture
def trained_model(rapm_df):
    model = PlayerArchetypeModel(k=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(rapm_df, season=2025)
    return model, rapm_df


# ---------------------------------------------------------------------------
# Auto-labelling
# ---------------------------------------------------------------------------

class TestAutoLabel:

    def test_returns_k_labels(self):
        k = 10
        centroids = np.random.default_rng(0).normal(0, 1, (k, 7))
        centroids[:, 0] = np.linspace(-2, 2, k)  # off_rapm spread
        centroids[:, 1] = np.linspace(-1, 1, k)  # def_rapm spread
        names = _assign_archetype_labels(centroids, ["off_rapm", "def_rapm", "pp_rapm",
                                                      "pk_rapm", "xgf_per60", "shots_per60",
                                                      "log1p_gp"], k)
        assert len(names) == k
        assert all(isinstance(n, str) and len(n) > 0 for n in names)

    def test_labels_are_unique_or_sensible(self):
        """Labels don't have to be unique but shouldn't be empty."""
        k = 8
        rng = np.random.default_rng(1)
        centroids = rng.normal(0, 1, (k, 7))
        names = _assign_archetype_labels(
            centroids,
            ["off_rapm", "def_rapm", "pp_rapm", "pk_rapm", "xgf_per60", "shots_per60", "log1p_gp"],
            k,
        )
        assert all(n for n in names)

    def test_without_rapm_columns_falls_back(self):
        k = 5
        centroids = np.ones((k, 2))
        names = _assign_archetype_labels(centroids, ["x", "y"], k)
        assert len(names) == k
        assert all("Archetype_" in n or len(n) > 0 for n in names)


# ---------------------------------------------------------------------------
# PlayerArchetypeModel — init
# ---------------------------------------------------------------------------

class TestArchetypeModelInit:

    def test_default_k(self):
        model = PlayerArchetypeModel()
        assert model._k == DEFAULT_K

    def test_custom_k(self):
        model = PlayerArchetypeModel(k=8)
        assert model._k == 8

    def test_k_too_small_raises(self):
        with pytest.raises(ValueError, match="k must be"):
            PlayerArchetypeModel(k=MIN_K - 1)

    def test_k_too_large_raises(self):
        with pytest.raises(ValueError, match="k must be"):
            PlayerArchetypeModel(k=MAX_K + 1)


# ---------------------------------------------------------------------------
# PlayerArchetypeModel — fit
# ---------------------------------------------------------------------------

class TestArchetypeModelFit:

    def test_fit_returns_metrics(self, rapm_df):
        model = PlayerArchetypeModel(k=8)
        metrics = model.fit(rapm_df, season=2025)
        assert "inertia" in metrics
        assert "n_players" in metrics
        assert metrics["n_players"] == 50
        assert metrics["inertia"] > 0

    def test_fit_too_few_players_raises(self):
        df = _make_rapm_df(5)
        model = PlayerArchetypeModel(k=8)
        with pytest.raises(ValueError, match="Cannot fit"):
            model.fit(df, season=2025)

    def test_fit_missing_player_id_raises(self):
        df = _make_rapm_df(20).drop("player_id")
        model = PlayerArchetypeModel(k=8)
        with pytest.raises(ValueError, match="missing columns"):
            model.fit(df, season=2025)

    def test_fit_works_without_optional_columns(self):
        df = _make_rapm_df(50).select(["player_id", "player_name", "off_rapm", "def_rapm"])
        model = PlayerArchetypeModel(k=8)
        metrics = model.fit(df, season=2025)
        assert metrics["n_players"] == 50

    def test_archetype_names_assigned_after_fit(self, rapm_df):
        model = PlayerArchetypeModel(k=8)
        model.fit(rapm_df, season=2025)
        names = model.archetype_names()
        assert len(names) == 8
        assert all(isinstance(v, str) for v in names.values())

    def test_silhouette_none_or_float(self, rapm_df):
        model = PlayerArchetypeModel(k=8)
        metrics = model.fit(rapm_df, season=2025)
        s = metrics.get("silhouette")
        assert s is None or isinstance(s, float)


# ---------------------------------------------------------------------------
# PlayerArchetypeModel — predict
# ---------------------------------------------------------------------------

class TestArchetypeModelPredict:

    def test_predict_returns_dataframe(self, trained_model):
        model, rapm_df = trained_model
        result = model.predict(rapm_df, season=2025)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(rapm_df)

    def test_output_schema(self, trained_model):
        model, rapm_df = trained_model
        result = model.predict(rapm_df, season=2025)
        for col in PLAYER_ARCHETYPE_SCHEMA:
            assert col in result.columns, f"Missing: {col}"

    def test_cluster_ids_in_range(self, trained_model):
        model, rapm_df = trained_model
        result = model.predict(rapm_df, season=2025)
        ids = result["cluster_id"].to_list()
        assert all(0 <= c < model._k for c in ids)

    def test_archetypes_all_non_empty(self, trained_model):
        model, rapm_df = trained_model
        result = model.predict(rapm_df, season=2025)
        assert (result["archetype"].str.len_chars() > 0).all()

    def test_distances_non_negative(self, trained_model):
        model, rapm_df = trained_model
        result = model.predict(rapm_df, season=2025)
        assert (result["distance"] >= 0).all()

    def test_predict_unfitted_raises(self):
        model = PlayerArchetypeModel()
        df = _make_rapm_df(50)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(df, season=2025)


# ---------------------------------------------------------------------------
# PlayerArchetypeModel — centroids
# ---------------------------------------------------------------------------

class TestArchetypeCentroids:

    def test_centroids_dataframe_shape(self, trained_model):
        model, _ = trained_model
        df = model.centroids_dataframe()
        assert len(df) == model._k

    def test_centroids_has_feature_columns(self, trained_model):
        model, _ = trained_model
        df = model.centroids_dataframe()
        assert "off_rapm" in df.columns
        assert "def_rapm" in df.columns
        assert "archetype" in df.columns
        assert "n_players" in df.columns

    def test_centroids_n_players_sums_to_total(self, trained_model):
        model, rapm_df = trained_model
        df = model.centroids_dataframe()
        assert df["n_players"].sum() == len(rapm_df)

    def test_centroids_unfitted_raises(self):
        model = PlayerArchetypeModel()
        with pytest.raises(RuntimeError):
            model.centroids_dataframe()

    def test_rename_archetype(self, trained_model):
        model, _ = trained_model
        model.rename_archetype(0, "My Custom Archetype")
        names = model.archetype_names()
        assert names[0] == "My Custom Archetype"

    def test_rename_invalid_cluster_raises(self, trained_model):
        model, _ = trained_model
        with pytest.raises(ValueError, match="cluster_id"):
            model.rename_archetype(999, "Bad")

    def test_rename_unfitted_raises(self):
        model = PlayerArchetypeModel()
        with pytest.raises(RuntimeError):
            model.rename_archetype(0, "X")


# ---------------------------------------------------------------------------
# PlayerArchetypeModel — archetype_matchup_prior
# ---------------------------------------------------------------------------

class TestArchetypeMatchupPrior:

    def test_returns_dataframe(self, trained_model):
        model, rapm_df = trained_model
        assignments = model.predict(rapm_df, 2025)
        pairs = pl.DataFrame({
            "player_a_id":    rapm_df["player_id"][:20].to_list(),
            "player_b_id":    rapm_df["player_id"][20:40].to_list(),
            "model_xgf_pct":  [0.52, 0.48, 0.51, 0.50, 0.53] * 4,
        })
        result = model.archetype_matchup_prior(assignments, pairs)
        assert isinstance(result, pl.DataFrame)
        assert "mean_xgf_pct" in result.columns

    def test_empty_pairs_returns_empty(self, trained_model):
        model, rapm_df = trained_model
        assignments = model.predict(rapm_df, 2025)
        empty_pairs = pl.DataFrame(schema={
            "player_a_id": pl.Int64, "player_b_id": pl.Int64, "model_xgf_pct": pl.Float64
        })
        result = model.archetype_matchup_prior(assignments, empty_pairs)
        assert len(result) == 0

    def test_no_player_a_id_returns_empty(self, trained_model):
        model, rapm_df = trained_model
        assignments = model.predict(rapm_df, 2025)
        bad_df = pl.DataFrame({"x": [1, 2]})
        result = model.archetype_matchup_prior(assignments, bad_df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# PlayerArchetypeModel — save/load
# ---------------------------------------------------------------------------

class TestArchetypeModelIO:

    def test_save_load_roundtrip(self, tmp_path, rapm_df):
        model = PlayerArchetypeModel(k=8)
        model.fit(rapm_df, 2025)
        path = tmp_path / "archetype_model.pkl"
        model.save(path)
        loaded = PlayerArchetypeModel.load(path)
        r1 = model.predict(rapm_df, 2025)
        r2 = loaded.predict(rapm_df, 2025)
        assert r1["cluster_id"].to_list() == r2["cluster_id"].to_list()

    def test_save_unfitted_raises(self, tmp_path):
        model = PlayerArchetypeModel()
        with pytest.raises(RuntimeError):
            model.save(tmp_path / "model.pkl")

    def test_write_read_parquets(self, tmp_path, rapm_df):
        model = PlayerArchetypeModel(k=8)
        model.fit(rapm_df, 2025)
        assignments = model.predict(rapm_df, 2025)
        centroids   = model.centroids_dataframe()
        out_dir = tmp_path / "archetypes"
        p_a, p_c = write_archetypes(assignments, centroids, out_dir, 2025)
        assert p_a.exists()
        assert p_c.exists()
        loaded = pl.read_parquet(p_a)
        assert len(loaded) == len(rapm_df)
