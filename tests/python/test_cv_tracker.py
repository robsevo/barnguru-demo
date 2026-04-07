"""Tests for Feature 16.7 — CV worker service (data/cv_tracker.py).

These tests exercise the public API of CvWorkerManager and CvWorker without
requiring trained model weights, ffmpeg, or a live HLS stream.  All heavy
dependencies are mocked at the subprocess / model-loading boundary.
"""

from __future__ import annotations

import time
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from data.cv_tracker import (
    CvWorkerManager,
    CvWorker,
    TrackPoint,
    WorkerStatus,
    _nearest_crop,
    CLASS_NAMES,
    _FRAME_W,
    _FRAME_H,
    _FRAME_BYTES,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _dummy_status(game_id: int = 1, state: str = "running") -> WorkerStatus:
    return WorkerStatus(
        game_id=game_id, hls_url="http://example.com/test.m3u8",
        fps=6.0, arena=None, state=state, error=None,
        frames_processed=0, tracks_active=0,
        started_at=time.time(), last_frame_at=None,
    )


def _blank_frame() -> np.ndarray:
    return np.zeros((_FRAME_H, _FRAME_W, 3), dtype=np.uint8)


# ── TrackPoint ────────────────────────────────────────────────────────────────

class TestTrackPoint:
    def test_to_dict_contains_required_keys(self):
        tp = TrackPoint(
            track_id=1, class_id=0, class_name="player_home",
            cx=640.0, cy=360.0, vx=1.0, vy=-0.5,
            w=40.0, h=80.0, rink_x=10.0, rink_y=-5.0,
            jersey_number=88, jersey_locked=True,
            team="CHI", frame_ts=1.0,
        )
        d = tp.to_dict()
        for key in ("track_id", "class_id", "class_name", "cx", "cy",
                    "vx", "vy", "w", "h", "rink_x", "rink_y",
                    "jersey_number", "jersey_locked", "team", "frame_ts"):
            assert key in d, f"missing key: {key}"

    def test_to_dict_none_rink_coords(self):
        tp = TrackPoint(
            track_id=2, class_id=1, class_name="player_away",
            cx=100.0, cy=200.0, vx=0.0, vy=0.0,
            w=30.0, h=60.0, rink_x=None, rink_y=None,
            jersey_number=None, jersey_locked=False,
            team=None, frame_ts=0.0,
        )
        d = tp.to_dict()
        assert d["rink_x"] is None
        assert d["rink_y"] is None
        assert d["team"] is None


# ── WorkerStatus ──────────────────────────────────────────────────────────────

class TestWorkerStatus:
    def test_to_dict_contains_required_keys(self):
        s = _dummy_status()
        d = s.to_dict()
        for key in ("game_id", "hls_url", "fps", "arena", "state",
                    "error", "frames_processed", "tracks_active",
                    "started_at", "last_frame_at"):
            assert key in d

    def test_state_default_starting(self):
        w = CvWorker(game_id=99, hls_url="http://x.com/s.m3u8")
        assert w.status.state == "starting"


# ── _nearest_crop ─────────────────────────────────────────────────────────────

class TestNearestCrop:
    def test_returns_none_on_zero_bbox(self):
        # tcx right at edge + zero size → x1 == x2 → None
        frame = _blank_frame()
        crop = _nearest_crop(frame, [], tcx=0, tcy=0, tw=0.0, th=0.0)
        assert crop is None or crop.size == 0

    def test_uses_nearest_detection(self):
        frame = np.ones((_FRAME_H, _FRAME_W, 3), dtype=np.uint8) * 128

        # Two fake detections
        det_close = MagicMock()
        det_close.cx, det_close.cy = 640.0, 360.0
        det_close.x1, det_close.y1 = 610.0, 320.0
        det_close.x2, det_close.y2 = 670.0, 400.0

        det_far = MagicMock()
        det_far.cx, det_far.cy = 100.0, 100.0
        det_far.x1, det_far.y1 =  80.0,  80.0
        det_far.x2, det_far.y2 = 120.0, 120.0

        crop = _nearest_crop(frame, [det_close, det_far],
                             tcx=640, tcy=360, tw=60, th=80)
        assert crop is not None
        assert crop.shape[0] == 80   # y2-y1 = 400-320
        assert crop.shape[1] == 60   # x2-x1 = 670-610

    def test_fallback_to_tracker_bbox_when_no_close_det(self):
        frame = np.ones((_FRAME_H, _FRAME_W, 3), dtype=np.uint8) * 200

        # Detection is very far from the track centre
        det_far = MagicMock()
        det_far.cx, det_far.cy = 50.0, 50.0
        det_far.x1, det_far.y1 = 30.0, 30.0
        det_far.x2, det_far.y2 = 70.0, 70.0

        # Track at centre with 80×40 bbox
        crop = _nearest_crop(frame, [det_far], tcx=640, tcy=360, tw=80, th=40)
        assert crop is not None
        # Crop height should be from tracker bbox ≈ 40 px (may vary by 1 due to int rounding)
        assert 30 <= crop.shape[0] <= 42

    def test_fallback_when_no_detections(self):
        frame = np.ones((_FRAME_H, _FRAME_W, 3), dtype=np.uint8) * 100
        crop = _nearest_crop(frame, [], tcx=640, tcy=360, tw=60, th=40)
        assert crop is not None
        assert crop.shape == (40, 60, 3)


# ── CvWorkerManager — unit tests (no ffmpeg / models) ────────────────────────

class TestCvWorkerManager:
    def test_no_workers_at_init(self):
        mgr = CvWorkerManager()
        assert mgr.all_statuses() == []
        assert mgr.active_game_ids() == []

    def test_get_positions_unknown_game_returns_none(self):
        mgr = CvWorkerManager()
        assert mgr.get_positions(9999) is None

    def test_get_status_unknown_game_returns_none(self):
        mgr = CvWorkerManager()
        assert mgr.get_status(9999) is None

    def test_stop_unknown_game_returns_false(self):
        mgr = CvWorkerManager()
        assert mgr.stop(9999) is False

    def test_start_registers_worker(self):
        """start() should register a worker even before it loads models."""
        mgr = CvWorkerManager()
        # Patch CvWorker.start so no real thread is spawned
        with patch.object(CvWorker, "start", return_value=None):
            status = mgr.start(game_id=42, hls_url="http://x.com/s.m3u8")
        assert status.game_id == 42
        assert status.state == "starting"
        assert mgr.get_status(42) is not None

    def test_stop_removes_worker(self):
        mgr = CvWorkerManager()
        with patch.object(CvWorker, "start", return_value=None):
            with patch.object(CvWorker, "stop", return_value=None):
                mgr.start(game_id=7, hls_url="http://x.com/s.m3u8")
                removed = mgr.stop(7)
        assert removed is True
        assert mgr.get_status(7) is None

    def test_all_statuses_lists_all_workers(self):
        mgr = CvWorkerManager()
        with patch.object(CvWorker, "start", return_value=None):
            mgr.start(game_id=1, hls_url="http://x.com/1.m3u8")
            mgr.start(game_id=2, hls_url="http://x.com/2.m3u8")
        statuses = mgr.all_statuses()
        assert len(statuses) == 2
        ids = {s.game_id for s in statuses}
        assert ids == {1, 2}

    def test_start_restarts_existing_worker(self):
        """Re-starting a game_id should replace the old worker."""
        mgr = CvWorkerManager()
        with patch.object(CvWorker, "start", return_value=None):
            with patch.object(CvWorker, "stop", return_value=None):
                mgr.start(game_id=5, hls_url="http://x.com/old.m3u8")
                mgr.start(game_id=5, hls_url="http://x.com/new.m3u8")
        assert mgr.get_status(5).hls_url == "http://x.com/new.m3u8"
        # Only one worker should be registered
        assert len(mgr.all_statuses()) == 1

    def test_shutdown_stops_all_workers(self):
        mgr = CvWorkerManager()
        stop_calls = []
        def _fake_stop(timeout=5.0):
            stop_calls.append(True)
        with patch.object(CvWorker, "start", return_value=None):
            with patch.object(CvWorker, "stop", side_effect=_fake_stop):
                mgr.start(game_id=10, hls_url="http://x.com/a.m3u8")
                mgr.start(game_id=11, hls_url="http://x.com/b.m3u8")
                mgr.shutdown()
        assert len(stop_calls) == 2
        assert mgr.all_statuses() == []

    def test_latest_positions_empty_before_frame(self):
        mgr = CvWorkerManager()
        with patch.object(CvWorker, "start", return_value=None):
            mgr.start(game_id=3, hls_url="http://x.com/s.m3u8")
        positions = mgr.get_positions(3)
        assert positions == []


# ── CvWorker — error path tests ───────────────────────────────────────────────

class TestCvWorkerErrorPaths:
    def test_worker_errors_when_ffmpeg_missing(self):
        """Worker should set state='error' when ffmpeg is not on PATH."""
        worker = CvWorker(game_id=1, hls_url="http://x.com/s.m3u8")

        # Patch _load_models to succeed, but Popen to raise FileNotFoundError
        mock_models = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

        # Also patch gretzky_engine import and homography import
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = []

        def fake_open_ffmpeg():
            raise FileNotFoundError("ffmpeg not found")

        with patch.object(CvWorker, "_load_models", return_value=mock_models):
            with patch("builtins.__import__", side_effect=_patched_import_factory(mock_tracker)):
                with patch.object(CvWorker, "_open_ffmpeg", side_effect=fake_open_ffmpeg):
                    worker._run()

        assert worker.status.state == "error"
        assert "ffmpeg" in (worker.status.error or "")

    def test_worker_errors_when_model_missing(self):
        """Worker should set state='error' when model files are absent."""
        worker = CvWorker(game_id=2, hls_url="http://x.com/s.m3u8")

        def fake_load():
            raise FileNotFoundError("player_detector.pt not found")

        with patch.object(CvWorker, "_load_models", side_effect=fake_load):
            worker._run()

        assert worker.status.state == "error"
        assert worker.status.error is not None


# ── CLASS_NAMES sanity ────────────────────────────────────────────────────────

def test_class_names_match_detector_registry():
    assert CLASS_NAMES[0] == "player_home"
    assert CLASS_NAMES[1] == "player_away"
    assert CLASS_NAMES[2] == "goalie_home"
    assert CLASS_NAMES[3] == "goalie_away"
    assert CLASS_NAMES[4] == "referee"
    assert CLASS_NAMES[5] == "puck"
    assert len(CLASS_NAMES) == 6


def test_frame_constants_are_sane():
    assert _FRAME_W == 1280
    assert _FRAME_H == 720
    assert _FRAME_BYTES == _FRAME_W * _FRAME_H * 3


# ── Patch helper ──────────────────────────────────────────────────────────────

def _patched_import_factory(mock_tracker):
    """Return an import side_effect that fakes gretzky_engine and homography."""
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "gretzky_engine":
            mod = MagicMock()
            mod.Tracker.return_value = mock_tracker
            return mod
        if "homography" in name:
            mod = MagicMock()
            mod.compute_homography.return_value = None
            mod.project_to_rink.return_value = (0.0, 0.0)
            mod.save_homography.return_value = None
            mod.load_homography.side_effect = Exception("no cache")
            return mod
        return real_import(name, *args, **kwargs)

    return _fake_import
