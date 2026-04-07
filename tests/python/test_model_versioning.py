"""Tests for ModelRegistry (Feature 2.8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.model_versioning import ModelRegistry, SnapshotMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry(tmp_path) -> ModelRegistry:
    """Fresh registry backed by a temp directory."""
    return ModelRegistry(registry_path=tmp_path / "registry.json")


@pytest.fixture
def fake_pkl(tmp_path) -> Path:
    """A dummy .pkl file to use as model path."""
    p = tmp_path / "model.pkl"
    p.write_bytes(b"dummy")
    return p


# ---------------------------------------------------------------------------
# SnapshotMetadata tests
# ---------------------------------------------------------------------------

class TestSnapshotMetadata:

    def test_to_from_dict_roundtrip(self, fake_pkl):
        meta = SnapshotMetadata(
            slug="xg_2025_2026-01-01",
            model_type="xg",
            season=2025,
            created_at="2026-01-01T00:00:00+00:00",
            path=str(fake_pkl),
            metrics={"rmse": 0.12, "auc": 0.81},
            feature_names=["shot_type", "distance"],
            params={"n_estimators": 300},
            notes="test snapshot",
        )
        d = meta.to_dict()
        restored = SnapshotMetadata.from_dict(d)
        assert restored.slug == meta.slug
        assert restored.metrics == meta.metrics
        assert restored.feature_names == meta.feature_names

    def test_exists_true_for_real_file(self, fake_pkl):
        meta = SnapshotMetadata(
            slug="test_2025",
            model_type="test",
            season=2025,
            created_at="2026-01-01T00:00:00+00:00",
            path=str(fake_pkl),
        )
        assert meta.exists() is True

    def test_exists_false_for_missing_file(self, tmp_path):
        meta = SnapshotMetadata(
            slug="test_2025",
            model_type="test",
            season=2025,
            created_at="2026-01-01T00:00:00+00:00",
            path=str(tmp_path / "nonexistent.pkl"),
        )
        assert meta.exists() is False

    def test_default_fields(self, fake_pkl):
        meta = SnapshotMetadata(
            slug="x",
            model_type="xg",
            season=2025,
            created_at="2026-01-01T00:00:00+00:00",
            path=str(fake_pkl),
        )
        assert meta.metrics == {}
        assert meta.feature_names == []
        assert meta.params == {}
        assert meta.notes == ""


# ---------------------------------------------------------------------------
# ModelRegistry — registration
# ---------------------------------------------------------------------------

class TestRegistryRegister:

    def test_register_returns_metadata(self, registry, fake_pkl):
        meta = registry.register(
            model_type="xg",
            season=2025,
            path=fake_pkl,
            metrics={"rmse": 0.09},
        )
        assert isinstance(meta, SnapshotMetadata)
        assert meta.model_type == "xg"
        assert meta.season == 2025
        assert "rmse" in meta.metrics

    def test_slug_auto_generated(self, registry, fake_pkl):
        meta = registry.register(model_type="rapm", season=2025, path=fake_pkl)
        assert meta.slug.startswith("rapm_2025_")

    def test_slug_override(self, registry, fake_pkl):
        meta = registry.register(
            model_type="xg", season=2025, path=fake_pkl, slug="my_custom_slug"
        )
        assert meta.slug == "my_custom_slug"

    def test_duplicate_slug_incremented(self, registry, fake_pkl):
        meta1 = registry.register(model_type="xg", season=2025, path=fake_pkl)
        meta2 = registry.register(model_type="xg", season=2025, path=fake_pkl)
        assert meta1.slug != meta2.slug

    def test_registry_persisted_to_disk(self, registry, fake_pkl):
        registry.register(model_type="xg", season=2025, path=fake_pkl)
        assert registry._path.exists()
        raw = json.loads(registry._path.read_text())
        assert "snapshots" in raw
        assert len(raw["snapshots"]) == 1

    def test_feature_names_stored(self, registry, fake_pkl):
        meta = registry.register(
            model_type="xg",
            season=2025,
            path=fake_pkl,
            feature_names=["a", "b", "c"],
        )
        assert meta.feature_names == ["a", "b", "c"]

    def test_params_stored(self, registry, fake_pkl):
        meta = registry.register(
            model_type="xg",
            season=2025,
            path=fake_pkl,
            params={"n_estimators": 300, "max_depth": 4},
        )
        assert meta.params["n_estimators"] == 300

    def test_notes_stored(self, registry, fake_pkl):
        meta = registry.register(
            model_type="xg", season=2025, path=fake_pkl, notes="production run"
        )
        assert meta.notes == "production run"


# ---------------------------------------------------------------------------
# ModelRegistry — querying
# ---------------------------------------------------------------------------

class TestRegistryQuery:

    @pytest.fixture
    def populated(self, registry, fake_pkl):
        registry.register(model_type="xg",   season=2024, path=fake_pkl, metrics={"rmse": 0.09})
        registry.register(model_type="xg",   season=2025, path=fake_pkl, metrics={"rmse": 0.08})
        registry.register(model_type="rapm",  season=2024, path=fake_pkl, metrics={"rmse": 0.40})
        registry.register(model_type="rapm",  season=2025, path=fake_pkl, metrics={"rmse": 0.38})
        registry.register(model_type="goalie", season=2025, path=fake_pkl, metrics={"rmse": 0.32})
        return registry

    def test_list_all(self, populated):
        all_snaps = populated.list()
        assert len(all_snaps) == 5

    def test_list_by_model_type(self, populated):
        xg_snaps = populated.list(model_type="xg")
        assert all(s.model_type == "xg" for s in xg_snaps)
        assert len(xg_snaps) == 2

    def test_list_by_season(self, populated):
        s25 = populated.list(season=2025)
        assert all(s.season == 2025 for s in s25)
        assert len(s25) == 3

    def test_list_by_both(self, populated):
        snaps = populated.list(model_type="rapm", season=2025)
        assert len(snaps) == 1
        assert snaps[0].model_type == "rapm"
        assert snaps[0].season == 2025

    def test_list_sorted_newest_first(self, populated):
        snaps = populated.list()
        for i in range(len(snaps) - 1):
            assert snaps[i].created_at >= snaps[i + 1].created_at

    def test_get_existing_slug(self, populated):
        all_snaps = populated.list()
        slug = all_snaps[0].slug
        meta = populated.get(slug)
        assert meta is not None
        assert meta.slug == slug

    def test_get_missing_slug_returns_none(self, populated):
        assert populated.get("nonexistent_slug") is None

    def test_latest_returns_most_recent(self, populated):
        latest = populated.latest("xg", 2025)
        assert latest is not None
        assert latest.model_type == "xg"
        assert latest.season == 2025

    def test_latest_missing_returns_none(self, populated):
        assert populated.latest("goalie", 2024) is None

    def test_len(self, populated):
        assert len(populated) == 5


# ---------------------------------------------------------------------------
# ModelRegistry — comparison
# ---------------------------------------------------------------------------

class TestRegistryCompare:

    def test_compare_rmse_lower_is_better(self, registry, fake_pkl):
        m1 = registry.register(
            model_type="xg", season=2024, path=fake_pkl,
            metrics={"rmse": 0.09}, slug="xg_2024_a"
        )
        m2 = registry.register(
            model_type="xg", season=2025, path=fake_pkl,
            metrics={"rmse": 0.07}, slug="xg_2025_b"
        )
        result = registry.compare("xg_2024_a", "xg_2025_b", metric="rmse")
        assert result["better"] == "xg_2025_b"
        assert abs(result["delta"] - (-0.02)) < 1e-9

    def test_compare_auc_higher_is_better(self, registry, fake_pkl):
        registry.register(
            model_type="xg", season=2024, path=fake_pkl,
            metrics={"auc": 0.78}, slug="xg_2024_a"
        )
        registry.register(
            model_type="xg", season=2025, path=fake_pkl,
            metrics={"auc": 0.81}, slug="xg_2025_b"
        )
        result = registry.compare("xg_2024_a", "xg_2025_b", metric="auc")
        assert result["better"] == "xg_2025_b"

    def test_compare_missing_metric_returns_none(self, registry, fake_pkl):
        registry.register(
            model_type="xg", season=2024, path=fake_pkl, slug="xg_2024_a"
        )
        registry.register(
            model_type="xg", season=2025, path=fake_pkl, slug="xg_2025_b"
        )
        result = registry.compare("xg_2024_a", "xg_2025_b", metric="rmse")
        assert result["value_a"] is None
        assert result["better"] is None

    def test_compare_missing_slug(self, registry, fake_pkl):
        registry.register(
            model_type="xg", season=2025, path=fake_pkl, slug="xg_2025"
        )
        result = registry.compare("xg_2025", "nonexistent", metric="rmse")
        assert result["value_b"] is None


# ---------------------------------------------------------------------------
# ModelRegistry — deletion
# ---------------------------------------------------------------------------

class TestRegistryDelete:

    def test_delete_removes_entry(self, registry, fake_pkl):
        meta = registry.register(model_type="xg", season=2025, path=fake_pkl)
        assert len(registry) == 1
        deleted = registry.delete(meta.slug)
        assert deleted is True
        assert len(registry) == 0

    def test_delete_missing_returns_false(self, registry):
        result = registry.delete("nonexistent")
        assert result is False

    def test_delete_file_true_removes_file(self, registry, tmp_path):
        pkl = tmp_path / "to_delete.pkl"
        pkl.write_bytes(b"data")
        meta = registry.register(model_type="xg", season=2025, path=pkl)
        registry.delete(meta.slug, delete_file=True)
        assert not pkl.exists()

    def test_delete_persisted_after_reload(self, tmp_path, fake_pkl):
        reg_path = tmp_path / "registry.json"
        reg1 = ModelRegistry(registry_path=reg_path)
        meta = reg1.register(model_type="xg", season=2025, path=fake_pkl)
        reg1.delete(meta.slug)

        reg2 = ModelRegistry(registry_path=reg_path)
        assert len(reg2) == 0


# ---------------------------------------------------------------------------
# ModelRegistry — persistence
# ---------------------------------------------------------------------------

class TestRegistryPersistence:

    def test_reload_preserves_data(self, tmp_path, fake_pkl):
        reg_path = tmp_path / "registry.json"
        reg1 = ModelRegistry(registry_path=reg_path)
        meta1 = reg1.register(
            model_type="xg", season=2025, path=fake_pkl,
            metrics={"rmse": 0.08}, feature_names=["f1", "f2"]
        )

        reg2 = ModelRegistry(registry_path=reg_path)
        meta2 = reg2.get(meta1.slug)
        assert meta2 is not None
        assert meta2.metrics["rmse"] == pytest.approx(0.08)
        assert meta2.feature_names == ["f1", "f2"]

    def test_empty_registry_starts_clean(self, tmp_path):
        reg = ModelRegistry(registry_path=tmp_path / "new_registry.json")
        assert len(reg) == 0

    def test_corrupted_json_starts_empty(self, tmp_path):
        reg_path = tmp_path / "registry.json"
        reg_path.write_text("{corrupt json{{")
        reg = ModelRegistry(registry_path=reg_path)
        assert len(reg) == 0

    def test_summary_non_empty(self, tmp_path, fake_pkl):
        reg = ModelRegistry(registry_path=tmp_path / "r.json")
        reg.register(model_type="xg", season=2025, path=fake_pkl, metrics={"rmse": 0.08})
        summary = reg.summary()
        assert "xg" in summary
        assert "2025" in summary

    def test_summary_empty(self, registry):
        assert "empty" in registry.summary().lower()
