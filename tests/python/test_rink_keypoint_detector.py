"""Tests for Phase 16.5 — Rink Keypoint Detector + Homography.

Covers:
  - RinkKeypointNet: forward pass shape, output range
  - _preprocess: tensor shape and normalisation
  - _heatmaps_to_keypoints: coord scaling, confidence threshold
  - _render_gaussian: shape, peak location, normalised values
  - _build_heatmaps: visible/invisible keypoints, coordinate scaling
  - RinkKeypointDetector: missing model → FileNotFoundError
  - RinkKeypointDetector: detect returns correct key set, conf in [0,1]
  - RinkKeypointDetector: detect_batch length matches input
  - homography: RINK_KEYPOINTS_WORLD has exactly 7 entries matching KEYPOINT_NAMES
  - compute_homography: returns None for < 4 points
  - compute_homography: valid H for ≥ 4 perfect points
  - project_to_rink: centre-ice identity (H from perfect keypoints → (0,0))
  - project_to_rink: output clipped to rink bounds
  - project_batch: length matches input, result is list
  - save_homography / load_homography: round-trip
  - list_cached_arenas: returns slugs
  - _arena_slug: sanitises arena name
  - Training smoke: RinkKeypointDataset loads synthetic annotation
  - check_quality_gate: passed/failed thresholds
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from models.cv.rink_keypoint_detector import (
    CONF_THR,
    HMAP_H,
    HMAP_W,
    INPUT_H,
    INPUT_W,
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    PCK_GATE,
    RinkKeypointNet,
    _heatmaps_to_keypoints,
    _preprocess,
    check_quality_gate,
)
from models.cv.homography import (
    RINK_KEYPOINTS_WORLD,
    RINK_X_MAX,
    RINK_X_MIN,
    RINK_Y_MAX,
    RINK_Y_MIN,
    _arena_slug,
    compute_homography,
    list_cached_arenas,
    load_homography,
    project_batch,
    project_to_rink,
    save_homography,
)
from scripts.train_rink_keypoints import (
    _build_heatmaps,
    _render_gaussian,
    RinkKeypointDataset,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _random_frame(h: int = 720, w: int = 1280) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def _perfect_detected_kps() -> dict[str, tuple[float, float, float]]:
    """Return all 7 keypoints at their ground-truth world positions as pixel coords.

    Using a simple affine-like fake mapping: pixel = world * 5 + (640, 360).
    Confidence is fixed at 0.99.
    """
    result = {}
    for name, (wx, wy) in RINK_KEYPOINTS_WORLD.items():
        px = wx * 5.0 + 640.0
        py = wy * 5.0 + 360.0
        result[name] = (px, py, 0.99)
    return result


# ===========================================================================
# RinkKeypointNet — architecture
# ===========================================================================

class TestRinkKeypointNet:
    def test_forward_output_shape(self):
        net = RinkKeypointNet()
        net.eval()
        x = torch.zeros(1, 3, INPUT_H, INPUT_W)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1, NUM_KEYPOINTS, HMAP_H, HMAP_W)

    def test_batch_output_shape(self):
        net = RinkKeypointNet()
        net.eval()
        x = torch.zeros(4, 3, INPUT_H, INPUT_W)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (4, NUM_KEYPOINTS, HMAP_H, HMAP_W)

    def test_output_in_0_1(self):
        """Sigmoid activation → output must be in [0, 1]."""
        net = RinkKeypointNet()
        net.eval()
        x = torch.randn(2, 3, INPUT_H, INPUT_W)
        with torch.no_grad():
            out = net(x)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_num_parameters_reasonable(self):
        """Model should be lightweight — under 500K parameters."""
        net = RinkKeypointNet()
        n = sum(p.numel() for p in net.parameters())
        assert n < 500_000, f"Model has {n:,} params — too heavy for real-time use"

    def test_heatmap_resolution(self):
        """Heatmap must be exactly INPUT // 4."""
        assert HMAP_H == INPUT_H // 4
        assert HMAP_W == INPUT_W // 4


# ===========================================================================
# _preprocess
# ===========================================================================

class TestPreprocess:
    def test_output_shape(self):
        frame = _random_frame()
        t = _preprocess(frame)
        assert t.shape == (1, 3, INPUT_H, INPUT_W)

    def test_output_dtype_float(self):
        frame = _random_frame()
        t = _preprocess(frame)
        assert t.dtype == torch.float32

    def test_output_range_after_normalisation(self):
        """After ImageNet normalisation, values can be outside [0,1]."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        t = _preprocess(frame)
        # A black image normalised → negative values (subtracting mean)
        assert float(t.min()) < 0.0

    def test_non_square_frame(self):
        frame = _random_frame(1080, 1920)
        t = _preprocess(frame)
        assert t.shape == (1, 3, INPUT_H, INPUT_W)


# ===========================================================================
# _heatmaps_to_keypoints
# ===========================================================================

class TestHeatmapsToKeypoints:
    def _zeros(self) -> np.ndarray:
        return np.zeros((NUM_KEYPOINTS, HMAP_H, HMAP_W), dtype=np.float32)

    def test_all_below_threshold_returns_none(self):
        hm = self._zeros()
        result = _heatmaps_to_keypoints(hm, 720, 1280, conf_thr=0.30)
        for name in KEYPOINT_NAMES:
            assert result[name] is None

    def test_correct_keys(self):
        hm = self._zeros()
        result = _heatmaps_to_keypoints(hm, 720, 1280)
        assert set(result.keys()) == set(KEYPOINT_NAMES)

    def test_peak_at_origin_gives_low_coords(self):
        hm = self._zeros()
        hm[0, 0, 0] = 1.0   # top-left corner, channel 0
        result = _heatmaps_to_keypoints(hm, 720, 1280, conf_thr=0.5)
        assert result[KEYPOINT_NAMES[0]] is not None
        x, y, conf = result[KEYPOINT_NAMES[0]]
        # Scaled from heatmap (0,0) to image space: x ≈ scale_x * 0.5, y ≈ scale_y * 0.5
        scale_x = 1280 / HMAP_W
        scale_y = 720 / HMAP_H
        assert abs(x - 0.5 * scale_x) < 1.0
        assert abs(y - 0.5 * scale_y) < 1.0

    def test_confidence_returned(self):
        hm = self._zeros()
        hm[1, HMAP_H // 2, HMAP_W // 2] = 0.75
        result = _heatmaps_to_keypoints(hm, 720, 1280, conf_thr=0.5)
        assert result[KEYPOINT_NAMES[1]] is not None
        _, _, conf = result[KEYPOINT_NAMES[1]]
        assert abs(conf - 0.75) < 0.01

    def test_coord_scaling(self):
        """Peak at heatmap centre → pixel roughly at image centre."""
        hm = self._zeros()
        hm[0, HMAP_H // 2, HMAP_W // 2] = 1.0
        orig_h, orig_w = 720, 1280
        result = _heatmaps_to_keypoints(hm, orig_h, orig_w, conf_thr=0.5)
        x, y, _ = result[KEYPOINT_NAMES[0]]
        assert abs(x - orig_w / 2) < orig_w / HMAP_W + 1
        assert abs(y - orig_h / 2) < orig_h / HMAP_H + 1


# ===========================================================================
# _render_gaussian (training helper)
# ===========================================================================

class TestRenderGaussian:
    def test_output_shape(self):
        hm = _render_gaussian((20.0, 15.0), HMAP_H, HMAP_W)
        assert hm.shape == (HMAP_H, HMAP_W)

    def test_peak_at_centre(self):
        hm = _render_gaussian((HMAP_W / 2, HMAP_H / 2), HMAP_H, HMAP_W)
        flat_max = int(hm.argmax())
        py, px = flat_max // HMAP_W, flat_max % HMAP_W
        assert abs(px - HMAP_W // 2) <= 1
        assert abs(py - HMAP_H // 2) <= 1

    def test_values_in_0_1(self):
        hm = _render_gaussian((30.0, 20.0), HMAP_H, HMAP_W)
        assert float(hm.min()) >= 0.0
        assert float(hm.max()) <= 1.0 + 1e-5

    def test_dtype_float32(self):
        hm = _render_gaussian((10.0, 10.0), HMAP_H, HMAP_W)
        assert hm.dtype == np.float32


# ===========================================================================
# _build_heatmaps (training helper)
# ===========================================================================

class TestBuildHeatmaps:
    def _ann(self) -> dict:
        return {
            "keypoints": {
                name: {"x": 640.0, "y": 360.0, "visible": True}
                for name in KEYPOINT_NAMES
            }
        }

    def test_output_shape(self):
        ann = self._ann()
        hm = _build_heatmaps(ann, 720, 1280)
        assert hm.shape == (NUM_KEYPOINTS, HMAP_H, HMAP_W)

    def test_visible_keypoint_nonzero(self):
        ann = self._ann()
        hm = _build_heatmaps(ann, 720, 1280)
        for i in range(NUM_KEYPOINTS):
            assert hm[i].max() > 0.0, f"Channel {i} ({KEYPOINT_NAMES[i]}) is zero"

    def test_invisible_keypoint_zero(self):
        ann = self._ann()
        ann["keypoints"]["center_ice_dot"]["visible"] = False
        hm = _build_heatmaps(ann, 720, 1280)
        assert hm[0].max() == 0.0   # index 0 = center_ice_dot

    def test_dtype_float32(self):
        ann = self._ann()
        hm = _build_heatmaps(ann, 720, 1280)
        assert hm.dtype == np.float32


# ===========================================================================
# RinkKeypointDetector — init
# ===========================================================================

class TestDetectorInit:
    def test_raises_if_model_missing(self, tmp_path):
        from models.cv.rink_keypoint_detector import RinkKeypointDetector
        with pytest.raises(FileNotFoundError, match="(?i)rink keypoint detector model not found"):
            RinkKeypointDetector(model_path=tmp_path / "missing.pt")

    def test_loads_model(self, tmp_path):
        from models.cv.rink_keypoint_detector import RinkKeypointDetector
        net = RinkKeypointNet()
        pt = tmp_path / "kp.pt"
        torch.save(net.state_dict(), str(pt))
        det = RinkKeypointDetector(model_path=pt, device="cpu")
        assert det is not None

    def test_repr(self, tmp_path):
        from models.cv.rink_keypoint_detector import RinkKeypointDetector
        net = RinkKeypointNet()
        pt = tmp_path / "kp.pt"
        torch.save(net.state_dict(), str(pt))
        det = RinkKeypointDetector(model_path=pt, device="cpu")
        r = repr(det)
        assert "RinkKeypointDetector" in r
        assert str(NUM_KEYPOINTS) in r


# ===========================================================================
# RinkKeypointDetector — detect
# ===========================================================================

class TestDetectorDetect:
    @pytest.fixture
    def det(self, tmp_path):
        from models.cv.rink_keypoint_detector import RinkKeypointDetector
        net = RinkKeypointNet()
        pt = tmp_path / "kp.pt"
        torch.save(net.state_dict(), str(pt))
        return RinkKeypointDetector(model_path=pt, device="cpu", conf_thr=0.0)

    def test_returns_dict_with_all_keys(self, det):
        frame = _random_frame()
        result = det.detect(frame)
        assert set(result.keys()) == set(KEYPOINT_NAMES)

    def test_values_none_or_3tuple(self, det):
        frame = _random_frame()
        result = det.detect(frame)
        for name, val in result.items():
            if val is not None:
                x, y, c = val
                assert 0.0 <= c <= 1.0, f"{name}: conf {c} out of range"

    def test_batch_length(self, det):
        frames = [_random_frame() for _ in range(4)]
        results = det.detect_batch(frames)
        assert len(results) == 4

    def test_keypoint_names_property(self, det):
        assert det.keypoint_names == KEYPOINT_NAMES


# ===========================================================================
# RINK_KEYPOINTS_WORLD
# ===========================================================================

class TestWorldCoords:
    def test_exactly_7_keypoints(self):
        assert len(RINK_KEYPOINTS_WORLD) == NUM_KEYPOINTS

    def test_names_match_keypoint_names(self):
        assert set(RINK_KEYPOINTS_WORLD.keys()) == set(KEYPOINT_NAMES)

    def test_center_is_origin(self):
        wx, wy = RINK_KEYPOINTS_WORLD["center_ice_dot"]
        assert wx == 0.0
        assert wy == 0.0

    def test_blue_lines_symmetric(self):
        lx, _ = RINK_KEYPOINTS_WORLD["blue_line_left"]
        rx, _ = RINK_KEYPOINTS_WORLD["blue_line_right"]
        assert abs(lx + rx) < 1e-9   # symmetric about centre
        assert lx < 0 < rx

    def test_faceoff_circles_symmetric(self):
        hl = RINK_KEYPOINTS_WORLD["faceoff_home_left"]
        hr = RINK_KEYPOINTS_WORLD["faceoff_home_right"]
        al = RINK_KEYPOINTS_WORLD["faceoff_away_left"]
        ar = RINK_KEYPOINTS_WORLD["faceoff_away_right"]
        # Home and away symmetric in x
        assert abs(hl[0] + al[0]) < 1e-9
        # Left/right symmetric in y
        assert abs(hl[1] + hr[1]) < 1e-9
        assert abs(al[1] + ar[1]) < 1e-9

    def test_all_within_rink_bounds(self):
        for name, (wx, wy) in RINK_KEYPOINTS_WORLD.items():
            assert RINK_X_MIN <= wx <= RINK_X_MAX, f"{name} x={wx} out of bounds"
            assert RINK_Y_MIN <= wy <= RINK_Y_MAX, f"{name} y={wy} out of bounds"


# ===========================================================================
# compute_homography
# ===========================================================================

class TestComputeHomography:
    def test_none_for_zero_points(self):
        kps = {name: None for name in KEYPOINT_NAMES}
        H = compute_homography(kps)
        assert H is None

    def test_none_for_three_points(self):
        kps = {name: None for name in KEYPOINT_NAMES}
        perfect = _perfect_detected_kps()
        for name in list(perfect.keys())[:3]:
            kps[name] = perfect[name]
        H = compute_homography(kps, min_pts=4)
        assert H is None

    def test_valid_h_for_all_points(self):
        kps = _perfect_detected_kps()
        H = compute_homography(kps)
        assert H is not None
        assert H.shape == (3, 3)

    def test_h_dtype_float64(self):
        kps = _perfect_detected_kps()
        H = compute_homography(kps)
        assert H is not None
        assert H.dtype == np.float64

    def test_low_confidence_excluded(self):
        """If all keypoints have conf below threshold, returns None."""
        kps = {name: (100.0, 100.0, 0.05) for name in KEYPOINT_NAMES}
        H = compute_homography(kps, conf_thr=0.3)
        assert H is None

    def test_partial_points_min4(self):
        """4 visible points (≥ min_pts=4) should succeed."""
        kps: dict = {name: None for name in KEYPOINT_NAMES}
        perfect = _perfect_detected_kps()
        for name in list(perfect.keys())[:4]:
            kps[name] = perfect[name]
        H = compute_homography(kps, min_pts=4)
        assert H is not None


# ===========================================================================
# project_to_rink
# ===========================================================================

class TestProjectToRink:
    def test_centre_maps_near_origin(self):
        """With H derived from perfect keypoints, centre pixel → near (0,0)."""
        kps = _perfect_detected_kps()
        H = compute_homography(kps)
        assert H is not None
        # The mapping is pixel = world * 5 + (640, 360), so inverse: world = (pixel - 640) / 5
        cx, cy = 640.0, 360.0
        rx, ry = project_to_rink((cx, cy), H, clip=False)
        assert abs(rx) < 2.0, f"Expected rx≈0, got {rx:.2f}"
        assert abs(ry) < 2.0, f"Expected ry≈0, got {ry:.2f}"

    def test_clipping_enforced(self):
        """Points outside the rink get clipped to bounds."""
        kps = _perfect_detected_kps()
        H = compute_homography(kps)
        assert H is not None
        # Extreme pixel that should project outside rink
        rx, ry = project_to_rink((0.0, 0.0), H, clip=True)
        assert RINK_X_MIN <= rx <= RINK_X_MAX
        assert RINK_Y_MIN <= ry <= RINK_Y_MAX

    def test_no_clipping_option(self):
        """clip=False allows out-of-bounds results."""
        kps = _perfect_detected_kps()
        H = compute_homography(kps)
        assert H is not None
        # With clip=False results might exceed bounds for out-of-rink pixels
        # Just verify the function runs without error
        result = project_to_rink((0.0, 0.0), H, clip=False)
        assert len(result) == 2


# ===========================================================================
# project_batch
# ===========================================================================

class TestProjectBatch:
    def test_empty_input(self):
        kps = _perfect_detected_kps()
        H = compute_homography(kps)
        assert H is not None
        result = project_batch([], H)
        assert result == []

    def test_length_matches_input(self):
        kps = _perfect_detected_kps()
        H = compute_homography(kps)
        assert H is not None
        pixels = [(100.0 * i, 50.0 * i) for i in range(5)]
        result = project_batch(pixels, H)
        assert len(result) == 5

    def test_values_clipped(self):
        kps = _perfect_detected_kps()
        H = compute_homography(kps)
        assert H is not None
        pixels = [(0.0, 0.0), (1280.0, 720.0)]
        result = project_batch(pixels, H, clip=True)
        for rx, ry in result:
            assert RINK_X_MIN <= rx <= RINK_X_MAX
            assert RINK_Y_MIN <= ry <= RINK_Y_MAX


# ===========================================================================
# Arena homography cache
# ===========================================================================

class TestArenaCache:
    def test_save_and_load_roundtrip(self, tmp_path):
        H = np.eye(3, dtype=np.float64) * 2.0
        save_homography("Bell Centre", H, cache_dir=tmp_path)
        H2 = load_homography("Bell Centre", cache_dir=tmp_path)
        assert H2 is not None
        np.testing.assert_array_almost_equal(H, H2)

    def test_load_missing_returns_none(self, tmp_path):
        H = load_homography("Nonexistent Arena", cache_dir=tmp_path)
        assert H is None

    def test_list_cached_arenas(self, tmp_path):
        H = np.eye(3)
        save_homography("Rogers Place",     H, cache_dir=tmp_path)
        save_homography("Scotiabank Arena", H, cache_dir=tmp_path)
        arenas = list_cached_arenas(tmp_path)
        assert "rogers_place" in arenas
        assert "scotiabank_arena" in arenas

    def test_list_empty_dir(self, tmp_path):
        arenas = list_cached_arenas(tmp_path)
        assert arenas == []

    def test_list_nonexistent_dir(self, tmp_path):
        arenas = list_cached_arenas(tmp_path / "does_not_exist")
        assert arenas == []


# ===========================================================================
# _arena_slug
# ===========================================================================

class TestArenaSlug:
    def test_lowercase_and_underscore(self):
        assert _arena_slug("Bell Centre") == "bell_centre"

    def test_special_characters_removed(self):
        slug = _arena_slug("TD Garden (Boston)")
        assert all(c.isalnum() or c == "_" for c in slug)

    def test_leading_trailing_stripped(self):
        slug = _arena_slug("  Rogers Place  ")
        assert not slug.startswith("_")
        assert not slug.endswith("_")

    def test_empty_string(self):
        slug = _arena_slug("")
        assert slug == "unknown"


# ===========================================================================
# RinkKeypointDataset
# ===========================================================================

class TestRinkKeypointDataset:
    def _synthetic_ann(self, tmp_path: Path) -> dict:
        from PIL import Image
        frame_path = tmp_path / "frame.jpg"
        Image.new("RGB", (1280, 720), color=(150, 200, 220)).save(str(frame_path))
        return {
            "frame_path": str(frame_path),
            "arena": "Test",
            "image_w": 1280,
            "image_h": 720,
            "keypoints": {
                name: {"x": 640.0, "y": 360.0, "visible": True}
                for name in KEYPOINT_NAMES
            },
        }

    def test_len(self, tmp_path):
        ann = self._synthetic_ann(tmp_path)
        ds = RinkKeypointDataset([ann])
        assert len(ds) == 1

    def test_item_shapes(self, tmp_path):
        ann = self._synthetic_ann(tmp_path)
        ds = RinkKeypointDataset([ann])
        img, hm = ds[0]
        assert img.shape == (3, INPUT_H, INPUT_W)
        assert hm.shape == (NUM_KEYPOINTS, HMAP_H, HMAP_W)

    def test_item_dtypes(self, tmp_path):
        ann = self._synthetic_ann(tmp_path)
        ds = RinkKeypointDataset([ann])
        img, hm = ds[0]
        assert img.dtype == torch.float32
        assert hm.dtype == torch.float32

    def test_heatmap_nonzero_for_visible(self, tmp_path):
        ann = self._synthetic_ann(tmp_path)
        ds = RinkKeypointDataset([ann])
        _, hm = ds[0]
        for i in range(NUM_KEYPOINTS):
            assert hm[i].max() > 0.0, f"Channel {i} ({KEYPOINT_NAMES[i]}) is zero"

    def test_missing_frame_falls_back(self, tmp_path):
        """Missing frame file should not crash — returns black image."""
        ann = {
            "frame_path": "data/cv_training/keypoints/frames/does_not_exist.jpg",
            "arena": "Test",
            "image_w": 1280,
            "image_h": 720,
            "keypoints": {name: {"x": 0.0, "y": 0.0, "visible": False}
                          for name in KEYPOINT_NAMES},
        }
        ds = RinkKeypointDataset([ann])
        img, hm = ds[0]
        assert img.shape == (3, INPUT_H, INPUT_W)


# ===========================================================================
# check_quality_gate
# ===========================================================================

class TestCheckQualityGate:
    def test_passes_above_threshold(self):
        passed, pck = check_quality_gate({"pck_at_10": PCK_GATE + 0.01})
        assert passed is True
        assert abs(pck - (PCK_GATE + 0.01)) < 1e-6

    def test_fails_below_threshold(self):
        passed, _ = check_quality_gate({"pck_at_10": PCK_GATE - 0.01})
        assert passed is False

    def test_exactly_at_threshold(self):
        passed, _ = check_quality_gate({"pck_at_10": PCK_GATE})
        assert passed is True

    def test_missing_key_fails(self):
        passed, pck = check_quality_gate({})
        assert passed is False
        assert pck == 0.0
