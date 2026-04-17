"""Runtime homography for the live CV worker, built on AdaptiveRinkLUT.

Drop-in replacement for the named-keypoint path in ``models/cv/homography.py``
that uses the 56-indexed SHL keypoint model + self-learning LUT.

Pipeline per frame:
  1. Run YOLO pose → 56 keypoints with confidences.
  2. Match detected kps to AdaptiveRinkLUT (trusted kps only).
  3. RANSAC homography: frame pixels → canonical LUT space.
  4. If ``canonical_to_nhl`` anchor is loaded, compose to get pixel → NHL feet.
  5. Feed the detection back into the LUT (online learning).

The anchor is a second 3×3 homography that maps canonical LUT pixels to NHL
feet. It is built by manually placing 4 known NHL landmarks on the canonical
frame (see ``set_anchor()``). Without it, we still get pixel→canonical which
is enough for cross-frame tracking; NHL feet are only required for speed
calculations and the RinkAnimation SVG overlay.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from models.cv.adaptive_rink_lut import AdaptiveRinkLUT, LUT_PATH

_REPO       = Path(__file__).resolve().parents[2]
ANCHOR_PATH = _REPO / "models" / "cv" / "rink_anchor.json"


@dataclass
class HomographyResult:
    """Result of compute_homography_adaptive()."""

    H_pixel_to_canonical: Optional[np.ndarray]
    H_pixel_to_nhl:       Optional[np.ndarray]   # None if anchor not set
    matched:              int
    inliers:              int
    rms_px:               float

    @property
    def ok(self) -> bool:
        return self.H_pixel_to_canonical is not None


def load_anchor(path: Path = ANCHOR_PATH) -> Optional[np.ndarray]:
    """Load canonical→NHL feet homography (the one-time manual calibration)."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if "H_canonical_to_nhl" not in d:
        return None
    return np.array(d["H_canonical_to_nhl"], dtype=np.float64)


def set_anchor(
    canonical_points: list[tuple[float, float]],
    nhl_feet_points:  list[tuple[float, float]],
    path: Path = ANCHOR_PATH,
) -> np.ndarray:
    """One-time calibration: match ≥4 LUT canonical points to their NHL feet coords.

    Args:
        canonical_points: pixel coords in the LUT's canonical frame
        nhl_feet_points:  corresponding (x_ft, y_ft) in NHL rink system
                          (center ice origin; x ∈ [-100, 100], y ∈ [-42.5, 42.5])

    Returns the 3×3 H mapping canonical → NHL feet.
    """
    import cv2
    assert len(canonical_points) == len(nhl_feet_points) >= 4, \
        "need ≥4 matched points to solve homography"
    src = np.array(canonical_points, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array(nhl_feet_points,  dtype=np.float32).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=1.0)
    if H is None:
        raise ValueError("findHomography returned None — points may be collinear or too few")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "H_canonical_to_nhl": H.tolist(),
        "canonical_points":   canonical_points,
        "nhl_feet_points":    nhl_feet_points,
    }, indent=2))
    return H


def compute_homography_adaptive(
    detected_kps: np.ndarray,               # shape (56, 3): [x_px, y_px, conf]
    lut:          AdaptiveRinkLUT,
    anchor:       Optional[np.ndarray] = None,
    conf_min:     float = 0.35,
    min_pairs:    int   = 4,
    ransac_thr:   float = 8.0,
    trusted_n:    int   = 20,                # require LUT entries seen ≥ this many frames
    trusted_std:  float = 15.0,
    update_lut:   bool  = True,
) -> HomographyResult:
    """Solve frame→canonical homography via the adaptive LUT.

    If ``update_lut`` is True, this also refines the LUT with the observation
    (the CV worker should leave this on during live use so the LUT keeps
    improving game-over-game).
    """
    import cv2

    trusted = lut.trusted_kps(min_n=trusted_n, max_std=trusted_std)
    src, dst = [], []
    for i, (x, y, c) in enumerate(detected_kps):
        if c < conf_min or i not in trusted:
            continue
        src.append([x, y])
        dst.append(list(trusted[i]))

    if len(src) < min_pairs:
        return HomographyResult(None, None, len(src), 0, math.inf)

    src_np = np.array(src, dtype=np.float32).reshape(-1, 1, 2)
    dst_np = np.array(dst, dtype=np.float32).reshape(-1, 1, 2)
    H_pc, mask = cv2.findHomography(src_np, dst_np, cv2.RANSAC, ransacReprojThreshold=ransac_thr)
    if H_pc is None:
        return HomographyResult(None, None, len(src), 0, math.inf)

    inliers = int(mask.sum())
    projected = cv2.perspectiveTransform(src_np, H_pc).reshape(-1, 2)
    diffs = projected - np.array(dst, dtype=np.float32)
    rms_px = float(np.sqrt(np.mean(np.sum(diffs[mask.ravel().astype(bool)] ** 2, axis=1))))

    H_pn: Optional[np.ndarray] = None
    if anchor is not None:
        H_pn = anchor @ H_pc

    if update_lut:
        lut.update(detected_kps, conf_min=conf_min, min_pairs=min_pairs, ransac_thr=ransac_thr)

    return HomographyResult(H_pc, H_pn, len(src), inliers, rms_px)


def project_pixels_to_nhl(
    pixel_points: np.ndarray,               # shape (N, 2)
    H_pixel_to_nhl: np.ndarray,
) -> np.ndarray:
    """Map pixel coordinates to NHL rink feet via a pre-composed homography."""
    import cv2
    if len(pixel_points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = pixel_points.astype(np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H_pixel_to_nhl.astype(np.float32)).reshape(-1, 2)
    return out
