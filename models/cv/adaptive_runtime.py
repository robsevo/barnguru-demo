"""Live runtime wrapper around the 56-kp SHL detector + AdaptiveRinkLUT.

This is the integration shim used by ``data/cv_tracker.py``. It hides the YOLO
model, the LUT, and the optional canonical→NHL anchor behind a tiny API:

    runtime = AdaptiveRinkRuntime()
    H_pixel_to_nhl = runtime.update(frame)      # None until enough trusted kps match
    xy = runtime.project(cx, cy, H_pixel_to_nhl)  # (rink_x_ft, rink_y_ft) or None

The LUT keeps self-learning every call (``update_lut=True``) so both halves of
the rink get populated as the broadcast pans. The composed H_pixel_to_nhl is
only valid when the anchor file exists; without it, ``update()`` still returns
None for NHL-feet projection but the LUT continues to refine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from models.cv.adaptive_homography import (
    compute_homography_adaptive,
    load_anchor,
    project_pixels_to_nhl,
)
from models.cv.adaptive_rink_lut import AdaptiveRinkLUT, LUT_PATH

log = logging.getLogger(__name__)

_REPO    = Path(__file__).resolve().parents[2]
_WEIGHTS = _REPO / "models" / "cv" / "HockeyRink.pt"


class AdaptiveRinkRuntime:
    """56-kp SHL detector + self-learning LUT + optional NHL-feet anchor."""

    def __init__(
        self,
        weights_path: Path = _WEIGHTS,
        lut_path:     Path = LUT_PATH,
        conf_min:     float = 0.35,
        trusted_n:    int   = 20,
        trusted_std:  float = 15.0,
    ) -> None:
        from ultralytics import YOLO  # heavy import kept local

        if not weights_path.exists():
            raise FileNotFoundError(f"HockeyRink weights not found: {weights_path}")

        self._model       = YOLO(str(weights_path))
        self._lut         = AdaptiveRinkLUT.load(lut_path)
        self._anchor      = load_anchor()
        self._conf_min    = conf_min
        self._trusted_n   = trusted_n
        self._trusted_std = trusted_std
        self._lut_path    = lut_path

        log.info(
            "[adaptive-runtime] lut: %d kps, %d frames_seen, anchor=%s",
            len(self._lut.stats),
            self._lut.frames_seen,
            self._anchor is not None,
        )

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Run the pose model and return a (56, 3) array of [x, y, conf]."""
        res = self._model.predict(frame, verbose=False, conf=0.01)[0]
        if res.keypoints is None or res.keypoints.data.numel() == 0:
            return np.zeros((56, 3), dtype=np.float32)
        return res.keypoints.data[0].cpu().numpy().astype(np.float32)

    def update(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Solve homography for this frame and refine the LUT.

        Returns the 3×3 H mapping pixel → NHL feet, or None if the frame could
        not be solved (too few trusted matches) or if no anchor is calibrated.
        """
        kps = self.detect(frame)
        r = compute_homography_adaptive(
            kps,
            self._lut,
            anchor=self._anchor,
            conf_min=self._conf_min,
            trusted_n=self._trusted_n,
            trusted_std=self._trusted_std,
            update_lut=True,
        )
        return r.H_pixel_to_nhl

    def project(
        self,
        cx: float,
        cy: float,
        H_pixel_to_nhl: Optional[np.ndarray],
    ) -> Optional[tuple[float, float]]:
        """Map one pixel → (rink_x_ft, rink_y_ft) via a pre-composed H."""
        if H_pixel_to_nhl is None:
            return None
        out = project_pixels_to_nhl(
            np.array([[cx, cy]], dtype=np.float32),
            H_pixel_to_nhl,
        )
        return float(out[0, 0]), float(out[0, 1])

    def save_lut(self) -> None:
        """Persist the learning LUT (call periodically from the worker)."""
        self._lut.save(self._lut_path)

    def quality(self) -> dict:
        return self._lut.quality_report()

    @property
    def has_anchor(self) -> bool:
        return self._anchor is not None
