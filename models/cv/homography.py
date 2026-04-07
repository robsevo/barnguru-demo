"""Phase 16.5 + 16.11 — Rink Homography & Per-Arena Calibration Library.

Computes the perspective transform (homography) that maps broadcast-frame pixel
coordinates to real-world rink coordinates (feet, centre-ice origin).

NHL rink standard dimensions used for world keypoint coordinates:
  - Rink length: 200 ft  → x ∈ [-100, +100]
  - Rink width:   85 ft  → y ∈ [-42.5, +42.5]
  - Centre-ice origin: (0, 0)
  - Blue lines at x = ±25 ft
  - Goal lines at x = ±89 ft
  - Defensive-zone faceoff dots: x = ±69 ft, y = ±22 ft

Algorithm (per frame):
  1. Collect detected keypoints with confidence ≥ conf_thr.
  2. If ≥ min_pts (default 4) are visible, call cv2.findHomography with RANSAC.
  3. Cache the computed homography per arena to
     ``data/cv_training/homography_cache/{arena_slug}.npz``.
  4. Recompute every N frames (recommended: 30). Between recomputations use the
     cached H.

Feature 16.11 adds:
  - Reprojection-error measurement (RMS feet) to every calibration.
  - Quality-aware cache updates: ``save_homography_if_better()`` replaces the
    cached calibration only when the new one has lower reprojection error.
  - Full 32-arena NHL roster ``NHL_ARENAS`` so coverage can be tracked.
  - ``get_cache_coverage()`` — returns calibration status for all 32 arenas.

Usage::

    from models.cv.homography import (
        compute_homography,          # → H | None  (16.5)
        compute_homography_with_error, # → (H, error_ft) | (None, inf)  (16.11)
        project_to_rink,             # pixel → feet  (16.5)
        save_homography,             # persist H + metadata  (16.5/16.11)
        save_homography_if_better,   # quality-gated save  (16.11)
        load_homography,             # load H (backward-compat)  (16.5)
        get_cache_entry,             # load H + metadata  (16.11)
        get_cache_coverage,          # 32-arena status  (16.11)
        list_cached_arenas,          # slug list  (16.5)
        NHL_ARENAS,                  # all 32 arenas  (16.11)
    )

    # Basic usage (same as 16.5)
    detected = kp_det.detect(frame)
    H = compute_homography(detected)
    if H is not None:
        rink_xy = project_to_rink((px, py), H)

    # Quality-aware usage (16.11)
    H, err_ft = compute_homography_with_error(detected)
    if H is not None:
        save_homography_if_better(H, arena_name, err_ft, game_id=game_id)
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO      = Path(__file__).parents[2]
CACHE_DIR  = _REPO / "data" / "cv_training" / "homography_cache"

# ---------------------------------------------------------------------------
# NHL rink world coordinates (feet, centre-ice origin)
# ---------------------------------------------------------------------------

RINK_KEYPOINTS_WORLD: dict[str, tuple[float, float]] = {
    "center_ice_dot":     (   0.0,   0.0),
    "blue_line_left":     ( -25.0,   0.0),
    "blue_line_right":    ( +25.0,   0.0),
    "faceoff_home_left":  ( -69.0, -22.0),
    "faceoff_home_right": ( -69.0, +22.0),
    "faceoff_away_left":  ( +69.0, -22.0),
    "faceoff_away_right": ( +69.0, +22.0),
}

# Rink bounds in world feet (for clipping projected coordinates)
RINK_X_MIN, RINK_X_MAX = -100.0, +100.0
RINK_Y_MIN, RINK_Y_MAX =  -42.5,  +42.5

# ---------------------------------------------------------------------------
# Feature 16.11 — All 32 NHL arena names (2025-26 season)
#
# Keys are the canonical NHL API arena names.
# The cache file for each arena is at: CACHE_DIR / f"{_arena_slug(name)}.npz"
# ---------------------------------------------------------------------------

NHL_ARENAS: dict[str, str] = {
    # Team abbreviation → official arena name
    "ANA": "Honda Center",
    "BOS": "TD Garden",
    "BUF": "KeyBank Center",
    "CGY": "Scotiabank Saddledome",
    "CAR": "PNC Arena",
    "CHI": "United Center",
    "COL": "Ball Arena",
    "CBJ": "Nationwide Arena",
    "DAL": "American Airlines Center",
    "DET": "Little Caesars Arena",
    "EDM": "Rogers Place",
    "FLA": "Amerant Bank Arena",
    "VGK": "T-Mobile Arena",
    "LAK": "Crypto.com Arena",
    "MIN": "Xcel Energy Center",
    "MTL": "Bell Centre",
    "NSH": "Bridgestone Arena",
    "NJD": "Prudential Center",
    "NYI": "UBS Arena",
    "NYR": "Madison Square Garden",
    "OTT": "Canadian Tire Centre",
    "PHI": "Wells Fargo Center",
    "PIT": "PPG Paints Arena",
    "SJS": "SAP Center",
    "SEA": "Climate Pledge Arena",
    "STL": "Enterprise Center",
    "TBL": "Amalie Arena",
    "TOR": "Scotiabank Arena",
    "UTA": "Delta Center",
    "VAN": "Rogers Arena",
    "WSH": "Capital One Arena",
    "WPG": "Canada Life Centre",
}


# ---------------------------------------------------------------------------
# Core homography functions
# ---------------------------------------------------------------------------

def compute_homography(
    detected_kps: dict[str, tuple[float, float, float] | None],
    conf_thr: float = 0.30,
    min_pts: int = 4,
) -> Optional[np.ndarray]:
    """Compute a 3×3 homography from detected rink keypoints.

    Args:
        detected_kps: Output of ``RinkKeypointDetector.detect()`` — dict mapping
                      keypoint name to ``(x_px, y_px, conf)`` or ``None``.
        conf_thr:     Minimum confidence to include a keypoint as a source point.
        min_pts:      Minimum number of visible keypoints required; returns
                      ``None`` if fewer are available.

    Returns:
        A (3, 3) float64 homography matrix mapping pixel → rink feet, or ``None``
        if too few keypoints are visible.
    """
    import cv2

    src_pts: list[tuple[float, float]] = []
    dst_pts: list[tuple[float, float]] = []

    for name, val in detected_kps.items():
        if val is None:
            continue
        x_px, y_px, conf = val
        if conf < conf_thr:
            continue
        if name not in RINK_KEYPOINTS_WORLD:
            continue
        src_pts.append((x_px, y_px))
        dst_pts.append(RINK_KEYPOINTS_WORLD[name])

    if len(src_pts) < min_pts:
        return None

    src = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=3.0)

    if H is None:
        return None
    return H.astype(np.float64)


def compute_reprojection_error(
    src_pts: list[tuple[float, float]],
    dst_pts: list[tuple[float, float]],
    H: np.ndarray,
) -> float:
    """Compute the RMS reprojection error in rink feet.

    Projects each source pixel coordinate through H → world, then measures the
    Euclidean distance to the expected world coordinate.

    Args:
        src_pts: Pixel (x, y) coordinates of the matched keypoints.
        dst_pts: Corresponding world (x_ft, y_ft) rink coordinates.
        H:       3×3 homography mapping pixels → rink feet.

    Returns:
        Root-mean-square reprojection error in feet.  Returns ``inf`` if
        ``src_pts`` is empty or the transform fails.
    """
    if not src_pts:
        return math.inf

    try:
        import cv2
        src = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(src, H.astype(np.float32)).reshape(-1, 2)
        dst = np.array(dst_pts, dtype=np.float32)
        sq_errors = np.sum((projected - dst) ** 2, axis=1)
        return float(math.sqrt(float(np.mean(sq_errors))))
    except Exception:
        return math.inf


def compute_homography_with_error(
    detected_kps: dict[str, tuple[float, float, float] | None],
    conf_thr: float = 0.30,
    min_pts: int = 4,
) -> tuple[Optional[np.ndarray], float]:
    """Compute homography and its RMS reprojection error in rink feet.

    Extends ``compute_homography()`` by also returning the calibration quality.
    A lower error means a better fit between detected keypoints and the NHL rink
    model.

    Args:
        detected_kps: Same as ``compute_homography()``.
        conf_thr:     Minimum keypoint confidence threshold.
        min_pts:      Minimum keypoints required.

    Returns:
        ``(H, error_ft)`` where:
          - H is the 3×3 homography or ``None`` if too few keypoints.
          - error_ft is the RMS reprojection error in feet, or ``math.inf``
            when H is ``None``.
    """
    import cv2

    src_pts: list[tuple[float, float]] = []
    dst_pts: list[tuple[float, float]] = []

    for name, val in detected_kps.items():
        if val is None:
            continue
        x_px, y_px, conf = val
        if conf < conf_thr:
            continue
        if name not in RINK_KEYPOINTS_WORLD:
            continue
        src_pts.append((x_px, y_px))
        dst_pts.append(RINK_KEYPOINTS_WORLD[name])

    if len(src_pts) < min_pts:
        return None, math.inf

    src = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=3.0)

    if H is None:
        return None, math.inf

    H = H.astype(np.float64)
    error_ft = compute_reprojection_error(src_pts, dst_pts, H)
    return H, error_ft


def project_to_rink(
    pixel_xy: tuple[float, float],
    H: np.ndarray,
    clip: bool = True,
) -> tuple[float, float]:
    """Project a pixel coordinate to rink world coordinates using homography H.

    Args:
        pixel_xy: (x_px, y_px) in the broadcast frame.
        H:        3×3 homography (pixel → feet).
        clip:     If True, clamp result to rink bounds.

    Returns:
        (x_feet, y_feet) in NHL rink coordinate system.
    """
    px, py = pixel_xy
    pt = np.array([[[px, py]]], dtype=np.float64)

    import cv2
    result = cv2.perspectiveTransform(pt.astype(np.float32), H.astype(np.float32))
    x, y = float(result[0, 0, 0]), float(result[0, 0, 1])

    if clip:
        x = float(np.clip(x, RINK_X_MIN, RINK_X_MAX))
        y = float(np.clip(y, RINK_Y_MIN, RINK_Y_MAX))
    return x, y


def project_batch(
    pixel_coords: list[tuple[float, float]],
    H: np.ndarray,
    clip: bool = True,
) -> list[tuple[float, float]]:
    """Project a list of pixel coordinates to rink world coordinates."""
    if not pixel_coords:
        return []

    import cv2
    pts = np.array(pixel_coords, dtype=np.float32).reshape(-1, 1, 2)
    result = cv2.perspectiveTransform(pts, H.astype(np.float32))
    out: list[tuple[float, float]] = []
    for i in range(len(pixel_coords)):
        x, y = float(result[i, 0, 0]), float(result[i, 0, 1])
        if clip:
            x = float(np.clip(x, RINK_X_MIN, RINK_X_MAX))
            y = float(np.clip(y, RINK_Y_MIN, RINK_Y_MAX))
        out.append((x, y))
    return out


# ---------------------------------------------------------------------------
# Per-arena calibration cache
# ---------------------------------------------------------------------------

def _arena_slug(arena_name: str) -> str:
    """Convert arena name to filesystem-safe slug."""
    slug = arena_name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_") or "unknown"


def save_homography(
    arena_name: str,
    H: np.ndarray,
    cache_dir: Path = CACHE_DIR,
    reprojection_error_ft: float = math.inf,
    game_id: int = 0,
    keypoints_used: int = 0,
) -> Path:
    """Save a homography matrix + metadata for an arena to the cache directory.

    The .npz file stores the matrix H plus calibration metadata:
      ``calibrated_at``      — ISO-8601 UTC timestamp
      ``game_id``            — NHL game_id (0 if unknown)
      ``reprojection_error_ft`` — RMS projection error in rink feet
      ``keypoints_used``     — number of keypoints used for calibration
      ``calibration_count``  — total times this arena has been calibrated

    Existing files are overwritten unconditionally.  Use
    ``save_homography_if_better()`` for quality-gated updates.

    Returns the path of the saved file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_arena_slug(arena_name)}.npz"

    # Increment calibration count from the existing entry if present
    prev_count = 0
    if path.exists():
        try:
            with np.load(str(path), allow_pickle=False) as prev:
                if "calibration_count" in prev:
                    prev_count = int(prev["calibration_count"].flat[0])
        except Exception:
            pass

    calibrated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    np.savez(
        str(path),
        H=H,
        calibrated_at=np.array([calibrated_at], dtype="U32"),
        game_id=np.array([game_id], dtype=np.int64),
        reprojection_error_ft=np.array([reprojection_error_ft], dtype=np.float64),
        keypoints_used=np.array([keypoints_used], dtype=np.int32),
        calibration_count=np.array([prev_count + 1], dtype=np.int32),
    )
    return path


def save_homography_if_better(
    H: np.ndarray,
    arena_name: str,
    reprojection_error_ft: float,
    cache_dir: Path = CACHE_DIR,
    game_id: int = 0,
    keypoints_used: int = 0,
) -> tuple[bool, str]:
    """Save the homography only when it improves upon the cached calibration.

    Compares ``reprojection_error_ft`` against the existing cached error.
    Saves unconditionally when no cache entry exists for the arena.

    Args:
        H:                      New homography matrix.
        arena_name:             NHL arena name (used as cache key).
        reprojection_error_ft:  RMS reprojection error of the new calibration.
        cache_dir:              Cache directory (default: ``CACHE_DIR``).
        game_id:                NHL game_id for provenance tracking.
        keypoints_used:         Number of keypoints used to compute H.

    Returns:
        ``(saved, reason)`` — ``saved`` is True when the file was written;
        ``reason`` is a human-readable explanation.
    """
    if math.isinf(reprojection_error_ft) or reprojection_error_ft < 0:
        return False, f"invalid error value {reprojection_error_ft}"

    existing = get_cache_entry(arena_name, cache_dir=cache_dir)
    if existing is not None:
        existing_error = existing.get("reprojection_error_ft", math.inf)
        if math.isfinite(existing_error) and reprojection_error_ft >= existing_error:
            return False, (
                f"cached error {existing_error:.3f} ft ≤ new error "
                f"{reprojection_error_ft:.3f} ft — not replacing"
            )

    save_homography(
        arena_name, H,
        cache_dir=cache_dir,
        reprojection_error_ft=reprojection_error_ft,
        game_id=game_id,
        keypoints_used=keypoints_used,
    )
    return True, f"saved (error {reprojection_error_ft:.3f} ft)"


def load_homography(
    arena_name: str,
    cache_dir: Path = CACHE_DIR,
) -> Optional[np.ndarray]:
    """Load a cached homography matrix for an arena.

    Backward-compatible — returns only the 3×3 H matrix.
    Use ``get_cache_entry()`` to retrieve the full metadata.

    Returns ``None`` if no cache entry exists for the given arena.
    """
    path = cache_dir / f"{_arena_slug(arena_name)}.npz"
    if not path.exists():
        return None
    data = np.load(str(path), allow_pickle=False)
    return data["H"]


def get_cache_entry(
    arena_name: str,
    cache_dir: Path = CACHE_DIR,
) -> Optional[dict]:
    """Load the full cache entry for an arena, including metadata.

    Returns a dict with keys:
        ``H``                     — (3, 3) float64 homography matrix
        ``calibrated_at``         — ISO-8601 UTC string
        ``game_id``               — NHL game_id (int)
        ``reprojection_error_ft`` — RMS error in feet (float)
        ``keypoints_used``        — number of keypoints (int)
        ``calibration_count``     — total calibrations (int)
        ``arena_slug``            — filesystem slug

    Returns ``None`` when no entry exists.
    """
    slug = _arena_slug(arena_name)
    path = cache_dir / f"{slug}.npz"
    if not path.exists():
        return None

    try:
        data = np.load(str(path), allow_pickle=False)
    except Exception:
        return None

    entry: dict = {"H": data["H"], "arena_slug": slug}

    def _scalar(key: str, default):
        if key in data:
            v = data[key]
            return v.flat[0] if hasattr(v, "flat") else v
        return default

    entry["calibrated_at"]         = str(_scalar("calibrated_at", ""))
    entry["game_id"]               = int(_scalar("game_id", 0))
    entry["reprojection_error_ft"] = float(_scalar("reprojection_error_ft", math.inf))
    entry["keypoints_used"]        = int(_scalar("keypoints_used", 0))
    entry["calibration_count"]     = int(_scalar("calibration_count", 0))

    return entry


def list_cached_arenas(cache_dir: Path = CACHE_DIR) -> list[str]:
    """Return the slugs of all arenas with cached homographies."""
    if not cache_dir.exists():
        return []
    return sorted(p.stem for p in cache_dir.glob("*.npz"))


def get_cache_coverage(
    cache_dir: Path = CACHE_DIR,
) -> dict:
    """Return calibration coverage across all 32 NHL arenas.

    Returns a dict with:
        ``total``       — 32 (total arenas in the league)
        ``calibrated``  — number with a cached homography
        ``missing``     — number with no cache entry
        ``coverage_pct``— percentage calibrated
        ``arenas``      — list of per-arena dicts (one per NHL team/arena)

    Each per-arena dict has:
        ``team``                  — 3-letter team abbreviation
        ``arena``                 — official arena name
        ``slug``                  — cache file stem
        ``calibrated``            — bool
        ``calibrated_at``         — ISO string or ``None``
        ``game_id``               — int or 0
        ``reprojection_error_ft`` — float or ``None``
        ``keypoints_used``        — int or 0
        ``calibration_count``     — int or 0
    """
    arenas_list = []
    calibrated_count = 0

    for team, arena_name in sorted(NHL_ARENAS.items()):
        entry = get_cache_entry(arena_name, cache_dir=cache_dir)
        if entry is not None:
            calibrated_count += 1
            arenas_list.append({
                "team":                  team,
                "arena":                 arena_name,
                "slug":                  entry["arena_slug"],
                "calibrated":            True,
                "calibrated_at":         entry["calibrated_at"] or None,
                "game_id":               entry["game_id"],
                "reprojection_error_ft": entry["reprojection_error_ft"]
                                         if math.isfinite(entry["reprojection_error_ft"])
                                         else None,
                "keypoints_used":        entry["keypoints_used"],
                "calibration_count":     entry["calibration_count"],
            })
        else:
            arenas_list.append({
                "team":                  team,
                "arena":                 arena_name,
                "slug":                  _arena_slug(arena_name),
                "calibrated":            False,
                "calibrated_at":         None,
                "game_id":               0,
                "reprojection_error_ft": None,
                "keypoints_used":        0,
                "calibration_count":     0,
            })

    total = len(NHL_ARENAS)
    missing = total - calibrated_count
    coverage_pct = round(calibrated_count / total * 100, 1) if total else 0.0

    return {
        "total":        total,
        "calibrated":   calibrated_count,
        "missing":      missing,
        "coverage_pct": coverage_pct,
        "arenas":       arenas_list,
    }
