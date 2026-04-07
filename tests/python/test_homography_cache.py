"""Tests for Feature 16.11 — Per-arena homography library.

Covers:
  - compute_reprojection_error
  - compute_homography_with_error
  - save_homography metadata round-trip
  - save_homography_if_better quality gate
  - get_cache_entry
  - get_cache_coverage against NHL_ARENAS
  - API endpoint GET /api/cv/homography/cache
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from models.cv.homography import (
    NHL_ARENAS,
    _arena_slug,
    compute_reprojection_error,
    get_cache_coverage,
    get_cache_entry,
    list_cached_arenas,
    load_homography,
    save_homography,
    save_homography_if_better,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_H() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def _perfect_kps():
    """7 perfectly-placed keypoints (pixel = world * 5 + (640, 360))."""
    from models.cv.homography import RINK_KEYPOINTS_WORLD
    result = {}
    for name, (wx, wy) in RINK_KEYPOINTS_WORLD.items():
        px = wx * 5.0 + 640.0
        py = wy * 5.0 + 360.0
        result[name] = (px, py, 0.99)
    return result


# ---------------------------------------------------------------------------
# Tests: NHL_ARENAS roster
# ---------------------------------------------------------------------------

class TestNHLArenasRoster:
    def test_exactly_32_arenas(self):
        assert len(NHL_ARENAS) == 32

    def test_all_values_are_strings(self):
        for team, arena in NHL_ARENAS.items():
            assert isinstance(team, str), f"team {team!r} not a string"
            assert isinstance(arena, str), f"arena {arena!r} not a string"

    def test_known_arenas_present(self):
        assert NHL_ARENAS["MTL"] == "Bell Centre"
        assert NHL_ARENAS["EDM"] == "Rogers Place"
        assert NHL_ARENAS["TOR"] == "Scotiabank Arena"
        assert NHL_ARENAS["NYR"] == "Madison Square Garden"

    def test_team_codes_are_three_chars(self):
        for team in NHL_ARENAS:
            assert len(team) == 3, f"team code {team!r} is not 3 chars"

    def test_no_duplicate_arenas(self):
        names = list(NHL_ARENAS.values())
        assert len(names) == len(set(names)), "duplicate arena names found"


# ---------------------------------------------------------------------------
# Tests: compute_reprojection_error
# ---------------------------------------------------------------------------

class TestComputeReprojectionError:
    def test_identity_mapping_zero_error(self):
        import cv2
        # Simple case: src and dst are same points, H = identity
        src = [(1.0, 2.0), (3.0, 4.0)]
        dst = [(1.0, 2.0), (3.0, 4.0)]
        H = np.eye(3, dtype=np.float64)
        err = compute_reprojection_error(src, dst, H)
        assert err == pytest.approx(0.0, abs=0.01)

    def test_empty_returns_inf(self):
        H = np.eye(3, dtype=np.float64)
        err = compute_reprojection_error([], [], H)
        assert math.isinf(err)

    def test_non_negative(self):
        from models.cv.homography import compute_homography
        kps = _perfect_kps()
        H = compute_homography(kps)
        assert H is not None
        from models.cv.homography import RINK_KEYPOINTS_WORLD
        src_pts = [(v[0], v[1]) for v in kps.values() if v is not None]
        dst_pts = [RINK_KEYPOINTS_WORLD[k] for k in kps if kps[k] is not None]
        err = compute_reprojection_error(src_pts, dst_pts, H)
        assert err >= 0.0
        assert math.isfinite(err)

    def test_perfect_keypoints_small_error(self):
        from models.cv.homography import compute_homography_with_error
        kps = _perfect_kps()
        H, err = compute_homography_with_error(kps)
        assert H is not None
        assert math.isfinite(err)
        # Perfect keypoints should have near-zero reprojection error
        assert err < 2.0, f"Expected small error for perfect kps, got {err:.3f} ft"


# ---------------------------------------------------------------------------
# Tests: compute_homography_with_error
# ---------------------------------------------------------------------------

class TestComputeHomographyWithError:
    def test_returns_none_inf_for_too_few_points(self):
        from models.cv.homography import compute_homography_with_error
        H, err = compute_homography_with_error({"center_ice_dot": (640.0, 360.0, 0.99)})
        assert H is None
        assert math.isinf(err)

    def test_returns_h_and_finite_error_for_enough_points(self):
        from models.cv.homography import compute_homography_with_error
        kps = _perfect_kps()
        H, err = compute_homography_with_error(kps)
        assert H is not None
        assert H.shape == (3, 3)
        assert math.isfinite(err)
        assert err >= 0.0

    def test_low_confidence_excluded(self):
        from models.cv.homography import compute_homography_with_error
        # All keypoints below the default threshold (0.30)
        kps = _perfect_kps()
        low_conf = {k: (v[0], v[1], 0.10) for k, v in kps.items()}
        H, err = compute_homography_with_error(low_conf, conf_thr=0.30)
        assert H is None


# ---------------------------------------------------------------------------
# Tests: save_homography / get_cache_entry metadata round-trip
# ---------------------------------------------------------------------------

class TestMetadataRoundTrip:
    def test_save_stores_metadata(self, tmp_path):
        H = _identity_H() * 3.0
        save_homography(
            "Bell Centre", H,
            cache_dir=tmp_path,
            reprojection_error_ft=1.23,
            game_id=2024021000,
            keypoints_used=6,
        )
        entry = get_cache_entry("Bell Centre", cache_dir=tmp_path)
        assert entry is not None
        np.testing.assert_array_almost_equal(entry["H"], H)
        assert entry["game_id"] == 2024021000
        assert entry["reprojection_error_ft"] == pytest.approx(1.23, abs=1e-5)
        assert entry["keypoints_used"] == 6
        assert entry["calibration_count"] == 1

    def test_calibration_count_increments(self, tmp_path):
        H = _identity_H()
        save_homography("Rogers Place", H, cache_dir=tmp_path)
        save_homography("Rogers Place", H, cache_dir=tmp_path)
        save_homography("Rogers Place", H, cache_dir=tmp_path)
        entry = get_cache_entry("Rogers Place", cache_dir=tmp_path)
        assert entry is not None
        assert entry["calibration_count"] == 3

    def test_calibrated_at_is_iso_string(self, tmp_path):
        save_homography("Scotiabank Arena", _identity_H(), cache_dir=tmp_path)
        entry = get_cache_entry("Scotiabank Arena", cache_dir=tmp_path)
        assert entry is not None
        cal_at = entry["calibrated_at"]
        assert isinstance(cal_at, str)
        assert "T" in cal_at    # ISO-8601 includes a T separator

    def test_load_homography_backward_compat(self, tmp_path):
        H = _identity_H() * 2.0
        save_homography("TD Garden", H, cache_dir=tmp_path)
        H2 = load_homography("TD Garden", cache_dir=tmp_path)
        assert H2 is not None
        np.testing.assert_array_almost_equal(H, H2)

    def test_get_cache_entry_returns_none_when_missing(self, tmp_path):
        entry = get_cache_entry("Nonexistent Arena", cache_dir=tmp_path)
        assert entry is None

    def test_slug_normalisation(self, tmp_path):
        H = _identity_H()
        # Save under one name, load under equivalent slug form
        save_homography("Amalie Arena", H, cache_dir=tmp_path)
        entry = get_cache_entry("Amalie Arena", cache_dir=tmp_path)
        assert entry is not None
        assert entry["arena_slug"] == "amalie_arena"

    def test_infinite_error_stored_and_loaded(self, tmp_path):
        H = _identity_H()
        save_homography("KeyBank Center", H, cache_dir=tmp_path,
                        reprojection_error_ft=math.inf)
        entry = get_cache_entry("KeyBank Center", cache_dir=tmp_path)
        assert entry is not None
        # Infinity can't be stored in float64 npz cleanly, but should survive
        # as a very large number or inf
        assert entry["reprojection_error_ft"] > 1e6 or math.isinf(entry["reprojection_error_ft"])


# ---------------------------------------------------------------------------
# Tests: save_homography_if_better
# ---------------------------------------------------------------------------

class TestSaveHomographyIfBetter:
    def test_saves_when_no_existing_entry(self, tmp_path):
        H = _identity_H()
        saved, reason = save_homography_if_better(
            H, "Ball Arena", reprojection_error_ft=2.0,
            cache_dir=tmp_path, game_id=111,
        )
        assert saved is True
        assert "saved" in reason.lower()
        assert get_cache_entry("Ball Arena", cache_dir=tmp_path) is not None

    def test_saves_when_new_error_lower(self, tmp_path):
        H = _identity_H()
        save_homography("Bell Centre", H, cache_dir=tmp_path,
                        reprojection_error_ft=3.5)
        H2 = _identity_H() * 1.1
        saved, _ = save_homography_if_better(
            H2, "Bell Centre", reprojection_error_ft=2.0, cache_dir=tmp_path
        )
        assert saved is True
        # New entry should reflect updated error
        entry = get_cache_entry("Bell Centre", cache_dir=tmp_path)
        assert entry["reprojection_error_ft"] == pytest.approx(2.0, abs=1e-4)

    def test_does_not_save_when_error_same_or_higher(self, tmp_path):
        H = _identity_H()
        save_homography("Wells Fargo Center", H, cache_dir=tmp_path,
                        reprojection_error_ft=1.5)
        # Try to overwrite with worse calibration
        saved, reason = save_homography_if_better(
            H, "Wells Fargo Center", reprojection_error_ft=1.5,
            cache_dir=tmp_path,
        )
        assert saved is False

        saved2, _ = save_homography_if_better(
            H, "Wells Fargo Center", reprojection_error_ft=2.0,
            cache_dir=tmp_path,
        )
        assert saved2 is False

    def test_rejects_invalid_error_values(self, tmp_path):
        H = _identity_H()
        saved, _ = save_homography_if_better(
            H, "SAP Center", reprojection_error_ft=math.inf, cache_dir=tmp_path
        )
        assert saved is False

        saved2, _ = save_homography_if_better(
            H, "SAP Center", reprojection_error_ft=-1.0, cache_dir=tmp_path
        )
        assert saved2 is False

    def test_overwrites_inf_cached_error(self, tmp_path):
        H = _identity_H()
        # Cache entry with unknown (inf) error → should always accept finite
        save_homography("United Center", H, cache_dir=tmp_path,
                        reprojection_error_ft=math.inf)
        saved, _ = save_homography_if_better(
            H, "United Center", reprojection_error_ft=5.0, cache_dir=tmp_path
        )
        assert saved is True


# ---------------------------------------------------------------------------
# Tests: get_cache_coverage
# ---------------------------------------------------------------------------

class TestGetCacheCoverage:
    def test_empty_cache_returns_all_missing(self, tmp_path):
        cov = get_cache_coverage(cache_dir=tmp_path)
        assert cov["total"] == 32
        assert cov["calibrated"] == 0
        assert cov["missing"] == 32
        assert cov["coverage_pct"] == 0.0
        assert len(cov["arenas"]) == 32

    def test_one_arena_calibrated(self, tmp_path):
        save_homography("Bell Centre", _identity_H(), cache_dir=tmp_path,
                        reprojection_error_ft=1.5)
        cov = get_cache_coverage(cache_dir=tmp_path)
        assert cov["calibrated"] == 1
        assert cov["missing"] == 31
        assert cov["coverage_pct"] == pytest.approx(1 / 32 * 100, abs=0.2)

    def test_all_arenas_calibrated(self, tmp_path):
        H = _identity_H()
        for arena in NHL_ARENAS.values():
            save_homography(arena, H, cache_dir=tmp_path,
                            reprojection_error_ft=2.0)
        cov = get_cache_coverage(cache_dir=tmp_path)
        assert cov["calibrated"] == 32
        assert cov["missing"] == 0
        assert cov["coverage_pct"] == 100.0

    def test_arenas_list_has_correct_shape(self, tmp_path):
        cov = get_cache_coverage(cache_dir=tmp_path)
        required_keys = {
            "team", "arena", "slug", "calibrated",
            "calibrated_at", "game_id", "reprojection_error_ft",
            "keypoints_used", "calibration_count",
        }
        for arena_entry in cov["arenas"]:
            missing = required_keys - set(arena_entry.keys())
            assert not missing, f"Missing keys in arena entry: {missing}"

    def test_uncalibrated_arena_has_none_error(self, tmp_path):
        cov = get_cache_coverage(cache_dir=tmp_path)
        uncal = [a for a in cov["arenas"] if not a["calibrated"]]
        for a in uncal:
            assert a["reprojection_error_ft"] is None
            assert a["calibrated_at"] is None

    def test_calibrated_arena_has_metadata(self, tmp_path):
        save_homography("Rogers Place", _identity_H(), cache_dir=tmp_path,
                        reprojection_error_ft=0.8, game_id=2024020500,
                        keypoints_used=7)
        cov = get_cache_coverage(cache_dir=tmp_path)
        edm = next(a for a in cov["arenas"] if a["team"] == "EDM")
        assert edm["calibrated"] is True
        assert edm["arena"] == "Rogers Place"
        assert edm["reprojection_error_ft"] == pytest.approx(0.8, abs=1e-4)
        assert edm["game_id"] == 2024020500
        assert edm["keypoints_used"] == 7


# ---------------------------------------------------------------------------
# Tests: API endpoint /api/cv/homography/cache
# ---------------------------------------------------------------------------

class TestHomographyCacheEndpoint:
    def test_returns_200_with_correct_schema(self):
        from starlette.testclient import TestClient
        from dashboard.api.main import app
        client = TestClient(app)
        res = client.get("/api/cv/homography/cache")
        assert res.status_code == 200
        data = res.json()
        assert "total" in data
        assert "calibrated" in data
        assert "missing" in data
        assert "coverage_pct" in data
        assert "arenas" in data
        assert data["total"] == 32

    def test_arenas_list_length(self):
        from starlette.testclient import TestClient
        from dashboard.api.main import app
        client = TestClient(app)
        res = client.get("/api/cv/homography/cache")
        assert res.status_code == 200
        data = res.json()
        assert len(data["arenas"]) == 32

    def test_coverage_pct_is_float(self):
        from starlette.testclient import TestClient
        from dashboard.api.main import app
        client = TestClient(app)
        res = client.get("/api/cv/homography/cache")
        assert res.status_code == 200
        assert isinstance(res.json()["coverage_pct"], (int, float))
