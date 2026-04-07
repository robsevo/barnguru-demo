import pytest

gretzky_engine = pytest.importorskip(
    "gretzky_engine",
    reason="Run: uv run maturin develop --manifest-path engine/pyproject.toml",
)


def test_imports():
    assert gretzky_engine is not None


def test_version():
    v = gretzky_engine.gretzky_version()
    assert isinstance(v, str) and v == "0.1.0"


# ── Tracker (Feature 16.6) ───────────────────────────────────────────────────

def test_tracker_importable():
    assert hasattr(gretzky_engine, "Tracker")


def test_tracker_default_construction():
    t = gretzky_engine.Tracker()
    assert t.confirmed_count() == 0
    assert t.active_count() == 0
    assert t.frame_count() == 0


def test_tracker_empty_frame():
    t = gretzky_engine.Tracker()
    results = t.update([])
    assert results == []


def test_tracker_single_det_creates_tentative():
    t = gretzky_engine.Tracker(n_init=3)
    t.update([[100.0, 100.0, 50.0, 50.0, 1.0, 0]])
    assert t.active_count() == 1
    assert t.confirmed_count() == 0


def test_tracker_confirmed_after_n_init():
    t = gretzky_engine.Tracker(n_init=3, iou_threshold=0.1)
    for i in range(3):
        t.update([[100.0 + i, 100.0, 50.0, 50.0, 1.0, 0]])
    assert t.confirmed_count() == 1


def test_tracker_result_dict_fields():
    t = gretzky_engine.Tracker(n_init=3, iou_threshold=0.1)
    for i in range(3):
        results = t.update([[100.0 + i, 100.0, 50.0, 50.0, 1.0, 0]])
    assert len(results) == 1
    r = results[0]
    for key in ("id", "cx", "cy", "w", "h", "vx", "vy", "state", "class_id"):
        assert key in r, f"missing key: {key}"
    assert r["state"] == "confirmed"
    assert isinstance(r["id"], int)


def test_tracker_reset():
    t = gretzky_engine.Tracker(n_init=2, iou_threshold=0.1)
    for i in range(2):
        t.update([[100.0 + i, 100.0, 50.0, 50.0, 1.0, 0]])
    assert t.confirmed_count() == 1
    t.reset()
    assert t.confirmed_count() == 0
    assert t.frame_count() == 0
