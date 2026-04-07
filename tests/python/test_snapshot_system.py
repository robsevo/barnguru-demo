"""Tests for Feature 2.16 — GRETZKYSnapshot and SnapshotIndex."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from models.snapshot_system import (
    GRETZKYSnapshot,
    SnapshotIndex,
    SNAPSHOT_VERSION,
)


# ---------------------------------------------------------------------------
# Minimal stub objects (mimic pipeline / detector / normalizer interface)
# ---------------------------------------------------------------------------

class _StubPipeline:
    def n_players_observed(self):
        return 42


class _StubDetector:
    def active_regimes(self):
        return ["x", "y"]  # 2 active regimes


class _StubNormalizer:
    pass


def _make_snap(**kwargs) -> GRETZKYSnapshot:
    defaults = dict(
        pipeline=_StubPipeline(),
        season=2025,
        game_count=30,
        ewma_df=pl.DataFrame({"player_id": [1, 2], "ewma_xgf60": [3.5, 2.8]}),
        regime_detector=_StubDetector(),
        sos_normalizer=_StubNormalizer(),
        notes="Test snapshot",
    )
    defaults.update(kwargs)
    return GRETZKYSnapshot.capture(**defaults)


# ---------------------------------------------------------------------------
# GRETZKYSnapshot.capture
# ---------------------------------------------------------------------------

class TestCapture:
    def test_auto_name_contains_date(self):
        snap = _make_snap()
        assert snap.name.startswith("GRETZKY_as_of_")

    def test_custom_name(self):
        snap = _make_snap(name="my_custom_snap")
        assert snap.name == "my_custom_snap"

    def test_created_at_is_set(self):
        snap = _make_snap()
        assert "T" in snap.created_at  # ISO 8601

    def test_season_set(self):
        snap = _make_snap(season=2024)
        assert snap.season == 2024

    def test_game_count_set(self):
        snap = _make_snap(game_count=42)
        assert snap.game_count == 42

    def test_notes_set(self):
        snap = _make_snap(notes="Pre-deadline state")
        assert snap.notes == "Pre-deadline state"

    def test_extra_stored(self):
        snap = _make_snap(extra={"foo": "bar"})
        assert snap.extra["foo"] == "bar"

    def test_extra_defaults_empty_dict(self):
        snap = _make_snap()
        assert snap.extra == {}

    def test_none_components_allowed(self):
        snap = GRETZKYSnapshot.capture(
            pipeline=None,
            season=2025,
            ewma_df=None,
            regime_detector=None,
            sos_normalizer=None,
        )
        assert snap.pipeline is None
        assert snap.ewma_df is None


# ---------------------------------------------------------------------------
# GRETZKYSnapshot.save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_returns_path(self, tmp_path):
        snap = _make_snap()
        path = snap.save(tmp_path)
        assert isinstance(path, Path)
        assert path.exists()
        assert path.suffix == ".pkl"

    def test_save_creates_dir(self, tmp_path):
        subdir = tmp_path / "nested" / "dir"
        snap = _make_snap()
        path = snap.save(subdir)
        assert path.exists()

    def test_roundtrip_fields(self, tmp_path):
        snap = _make_snap(season=2024, game_count=55, notes="test note")
        path = snap.save(tmp_path)
        snap2 = GRETZKYSnapshot.load(path)
        assert snap2.season == 2024
        assert snap2.game_count == 55
        assert snap2.notes == "test note"
        assert snap2.name == snap.name
        assert snap2.created_at == snap.created_at

    def test_roundtrip_ewma_df(self, tmp_path):
        ewma = pl.DataFrame({"player_id": [1, 2, 3], "ewma_xgf60": [3.0, 2.5, 4.1]})
        snap = _make_snap(ewma_df=ewma)
        path = snap.save(tmp_path)
        snap2 = GRETZKYSnapshot.load(path)
        assert snap2.ewma_df is not None
        assert len(snap2.ewma_df) == 3

    def test_roundtrip_none_ewma(self, tmp_path):
        snap = GRETZKYSnapshot.capture(pipeline=None, season=2025, ewma_df=None)
        path = snap.save(tmp_path)
        snap2 = GRETZKYSnapshot.load(path)
        assert snap2.ewma_df is None

    def test_filename_sanitises_spaces(self, tmp_path):
        snap = GRETZKYSnapshot.capture(
            pipeline=None, season=2025, name="my snap with spaces"
        )
        path = snap.save(tmp_path)
        assert " " not in path.name

    def test_roundtrip_extra(self, tmp_path):
        snap = _make_snap(extra={"key": 123})
        path = snap.save(tmp_path)
        snap2 = GRETZKYSnapshot.load(path)
        assert snap2.extra["key"] == 123


# ---------------------------------------------------------------------------
# GRETZKYSnapshot.compare
# ---------------------------------------------------------------------------

class TestCompare:
    def test_compare_returns_dict(self):
        snap_a = _make_snap(game_count=10)
        snap_b = _make_snap(game_count=20)
        diff = GRETZKYSnapshot.compare(snap_a, snap_b)
        assert isinstance(diff, dict)

    def test_compare_keys(self):
        snap_a = _make_snap()
        snap_b = _make_snap()
        diff = GRETZKYSnapshot.compare(snap_a, snap_b)
        for key in [
            "name_a", "name_b", "game_count_a", "game_count_b",
            "game_count_delta", "n_players_a", "n_players_b",
            "n_active_regimes_a", "n_active_regimes_b",
            "ewma_mean_xgf60_a", "ewma_mean_xgf60_b",
        ]:
            assert key in diff

    def test_game_count_delta(self):
        snap_a = _make_snap(game_count=10)
        snap_b = _make_snap(game_count=30)
        diff = GRETZKYSnapshot.compare(snap_a, snap_b)
        assert diff["game_count_delta"] == 20

    def test_n_players_from_stub(self):
        snap_a = _make_snap()
        snap_b = _make_snap()
        diff = GRETZKYSnapshot.compare(snap_a, snap_b)
        assert diff["n_players_a"] == 42

    def test_n_active_regimes_from_stub(self):
        snap_a = _make_snap()
        snap_b = _make_snap()
        diff = GRETZKYSnapshot.compare(snap_a, snap_b)
        assert diff["n_active_regimes_a"] == 2

    def test_ewma_mean_computed(self):
        snap_a = _make_snap()
        snap_b = _make_snap()
        diff = GRETZKYSnapshot.compare(snap_a, snap_b)
        assert diff["ewma_mean_xgf60_a"] == pytest.approx(3.15)

    def test_none_pipeline_gives_zero_players(self):
        snap = GRETZKYSnapshot.capture(pipeline=None, season=2025)
        diff = GRETZKYSnapshot.compare(snap, snap)
        assert diff["n_players_a"] == 0

    def test_none_ewma_gives_none_mean(self):
        snap = GRETZKYSnapshot.capture(pipeline=None, season=2025, ewma_df=None)
        diff = GRETZKYSnapshot.compare(snap, snap)
        assert diff["ewma_mean_xgf60_a"] is None


# ---------------------------------------------------------------------------
# GRETZKYSnapshot.summary / __repr__
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_is_string(self):
        snap = _make_snap()
        s = snap.summary()
        assert isinstance(s, str)

    def test_summary_contains_name(self):
        snap = _make_snap(name="test_snap")
        assert "test_snap" in snap.summary()

    def test_summary_contains_season(self):
        snap = _make_snap(season=2024)
        assert "2024" in snap.summary()

    def test_summary_contains_notes(self):
        snap = _make_snap(notes="Pre-deadline")
        assert "Pre-deadline" in snap.summary()

    def test_summary_no_notes_no_notes_line(self):
        snap = _make_snap(notes="")
        assert "Notes" not in snap.summary()

    def test_repr(self):
        snap = _make_snap(name="snap_x", season=2025, game_count=10)
        r = repr(snap)
        assert "snap_x" in r
        assert "2025" in r


# ---------------------------------------------------------------------------
# SnapshotIndex
# ---------------------------------------------------------------------------

class TestSnapshotIndex:
    def test_empty_index_list(self, tmp_path):
        idx = SnapshotIndex(tmp_path)
        assert idx.list() == []

    def test_len_empty(self, tmp_path):
        idx = SnapshotIndex(tmp_path)
        assert len(idx) == 0

    def test_save_and_len(self, tmp_path):
        idx = SnapshotIndex(tmp_path)
        snap = _make_snap()
        idx.save(snap)
        assert len(idx) == 1

    def test_list_returns_entries(self, tmp_path):
        idx = SnapshotIndex(tmp_path)
        snap = _make_snap(notes="my_note")
        idx.save(snap)
        entries = idx.list()
        assert len(entries) == 1
        entry = entries[0]
        assert "name" in entry
        assert "path" in entry
        assert "created_at" in entry

    def test_list_sorted_newest_first(self, tmp_path):
        idx = SnapshotIndex(tmp_path)
        import time
        snap1 = _make_snap(name="snap_2024", season=2024)
        idx.save(snap1)
        time.sleep(0.01)
        snap2 = _make_snap(name="snap_2025", season=2025)
        idx.save(snap2)
        entries = idx.list()
        # Newest (larger created_at) first
        assert entries[0]["name"] == "snap_2025"

    def test_load_by_name(self, tmp_path):
        idx = SnapshotIndex(tmp_path)
        snap = _make_snap(name="my_snap", season=2025)
        idx.save(snap)
        loaded = idx.load("my_snap")
        assert loaded.season == 2025

    def test_load_not_found_raises(self, tmp_path):
        idx = SnapshotIndex(tmp_path)
        with pytest.raises(FileNotFoundError):
            idx.load("nonexistent")

    def test_delete_existing(self, tmp_path):
        idx = SnapshotIndex(tmp_path)
        snap = _make_snap(name="to_delete")
        idx.save(snap)
        assert len(idx) == 1
        result = idx.delete("to_delete")
        assert result is True
        assert len(idx) == 0

    def test_delete_nonexistent(self, tmp_path):
        idx = SnapshotIndex(tmp_path)
        result = idx.delete("ghost")
        assert result is False

    def test_repr(self, tmp_path):
        idx = SnapshotIndex(tmp_path)
        r = repr(idx)
        assert "SnapshotIndex" in r

    def test_default_dir_created(self, tmp_path, monkeypatch):
        """SnapshotIndex should create its directory if it doesn't exist."""
        new_dir = tmp_path / "snapshots"
        assert not new_dir.exists()
        idx = SnapshotIndex(new_dir)
        assert new_dir.exists()

    def test_multiple_snapshots(self, tmp_path):
        idx = SnapshotIndex(tmp_path)
        for i in range(3):
            snap = _make_snap(name=f"snap_{i}", season=2025 + i)
            idx.save(snap)
        assert len(idx) == 3
        entries = idx.list()
        assert len(entries) == 3
