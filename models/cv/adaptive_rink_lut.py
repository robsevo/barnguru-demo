"""Self-learning LUT for SHL-trained 56-keypoint rink detector applied to NHL frames.

The pretrained HockeyRink model outputs 56 indexed keypoints per frame. We don't
know the semantic identity of each index (which rink landmark it corresponds to)
without hand-annotation, but we know the identities are *stable* across frames —
kp_N is always the same landmark.

This module builds a canonical coordinate for each kp_idx by consensus across
many frames:

  1. **Bootstrap.** Pick the first frame with ≥ N high-confidence keypoints
     spread across the image. Use those pixel positions as the canonical
     coords for each kp_idx.

  2. **Refine.** For each subsequent frame:
        - Detect kps.
        - Solve a RANSAC homography from detected pixels → canonical coords
          using kp indices already in the LUT.
        - Back-project every detected kp through H → canonical space.
        - Update the LUT entry for that kp_idx (EMA mean + variance).

     Over time, the LUT's per-keypoint variance shrinks. Keypoints that keep
     landing consistently end up tightly clustered; unstable keypoints (model
     noise, short-lived labels) get wide variance and are dropped from future
     homography fits.

  3. **Persist.** LUT is saved to disk as JSON so both offline tooling and the
     live CV worker share the same calibration.

NHL-feet mapping (RinkAnimation display, shot speed in ft/s, etc.) is a separate
one-time calibration: label any 4 kp indices with their known NHL-feet coords
(center-ice dot, blue-line midpoints, faceoff dots) and a final H_canonical→NHL
is applied to every LUT entry.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_REPO     = Path(__file__).resolve().parents[2]
LUT_PATH  = _REPO / "models" / "cv" / "rink_adaptive_lut.json"


@dataclass
class KeypointStats:
    """Running estimate of one keypoint's canonical coord."""
    x:        float                  # EMA mean, canonical x
    y:        float                  # EMA mean, canonical y
    var_x:    float = 0.0            # EMA variance
    var_y:    float = 0.0
    n:        int   = 1              # observations
    conf_sum: float = 1.0            # cumulative confidence (for weighting)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var_x, self.var_y))

    def to_dict(self) -> dict:
        return {
            "x":        float(self.x),    "y":        float(self.y),
            "var_x":    float(self.var_x), "var_y":    float(self.var_y),
            "n":        int(self.n),       "conf_sum": float(self.conf_sum),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KeypointStats":
        return cls(**d)


@dataclass
class AdaptiveRinkLUT:
    """Running per-keypoint canonical-coord estimate.

    Attributes:
        stats:             dict of kp_index → KeypointStats
        frames_seen:       count of frames fed to update()
        bootstrap_frame:   kp layout used to seed canonical coords (pixel size)
        canonical_size:    (width, height) of the canonical space, in pixels
    """

    stats:           dict[int, KeypointStats] = field(default_factory=dict)
    frames_seen:     int                      = 0
    canonical_size:  tuple[int, int]          = (1280, 720)
    bootstrap_hash:  str                      = ""

    # ---- bootstrap ----------------------------------------------------------

    def bootstrap_from_frame(
        self,
        detected_kps: np.ndarray,          # shape (56, 3) [x, y, conf]
        frame_wh:     tuple[int, int],
        conf_min:     float = 0.5,
        min_kps:      int   = 8,
    ) -> bool:
        """Seed canonical coords from one good frame's detected kps.

        Returns True when the frame passes the quality gate and the LUT is
        initialized. False when the frame is too poor to bootstrap from.
        """
        hi_conf = [(i, x, y, c)
                   for i, (x, y, c) in enumerate(detected_kps)
                   if c >= conf_min]
        if len(hi_conf) < min_kps:
            return False

        # Sanity: kps should span a decent fraction of the image, not cluster tightly
        xs = np.array([x for _, x, _, _ in hi_conf])
        ys = np.array([y for _, _, y, _ in hi_conf])
        w, h = frame_wh
        if (xs.max() - xs.min()) < 0.30 * w or (ys.max() - ys.min()) < 0.15 * h:
            return False

        self.stats.clear()
        for i, x, y, c in hi_conf:
            self.stats[i] = KeypointStats(x=float(x), y=float(y), conf_sum=float(c))
        self.canonical_size = (int(w), int(h))
        self.frames_seen = 1
        return True

    # ---- update -------------------------------------------------------------

    def _match_pairs(
        self,
        detected_kps: np.ndarray,
        conf_min:     float,
        require_n:    int,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Return (src_pixels, dst_canonical, kp_indices) for RANSAC.

        Only uses kp indices currently present in the LUT that were also detected
        in this frame with confidence ≥ conf_min.
        """
        src, dst, idx = [], [], []
        for i, (x, y, c) in enumerate(detected_kps):
            if c < conf_min or i not in self.stats:
                continue
            s = self.stats[i]
            src.append([x, y])
            dst.append([s.x, s.y])
            idx.append(i)
        return np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32), idx

    def update(
        self,
        detected_kps: np.ndarray,
        conf_min:     float = 0.3,
        ransac_thr:   float = 8.0,
        min_pairs:    int   = 4,
        learn_rate:   float = 0.05,
    ) -> Optional[dict]:
        """Incorporate one frame's detections into the running LUT.

        Args:
            detected_kps: (56, 3) array of [x_px, y_px, conf].
            conf_min:     skip kps with confidence below this.
            ransac_thr:   RANSAC pixel threshold in canonical space.
            min_pairs:    minimum matching kps needed to compute H.
            learn_rate:   cap on EMA learning rate per observation.

        Returns metrics dict with `inliers`, `matched`, `rms_px`, `n_lut` — or
        None if the frame didn't produce a valid H.
        """
        import cv2

        src, dst, idxs = self._match_pairs(detected_kps, conf_min, min_pairs)
        if len(src) < min_pairs:
            return None

        H, mask = cv2.findHomography(
            src.reshape(-1, 1, 2),
            dst.reshape(-1, 1, 2),
            cv2.RANSAC,
            ransacReprojThreshold=ransac_thr,
        )
        if H is None:
            return None
        inliers = mask.ravel().astype(bool)
        if inliers.sum() < min_pairs:
            return None

        # Reprojection RMS (pixels in canonical space)
        projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        rms_px = float(np.sqrt(np.mean(np.sum((projected[inliers] - dst[inliers]) ** 2, axis=1))))

        # Back-project EVERY detected kp through H to update the LUT
        all_det = [(i, x, y, c)
                   for i, (x, y, c) in enumerate(detected_kps)
                   if c >= conf_min]
        if not all_det:
            return {"inliers": int(inliers.sum()), "matched": len(src),
                    "rms_px": rms_px, "n_lut": len(self.stats)}

        pts_src = np.array([[x, y] for _, x, y, _ in all_det], dtype=np.float32).reshape(-1, 1, 2)
        pts_can = cv2.perspectiveTransform(pts_src, H).reshape(-1, 2)

        for (i, _, _, c), (cx, cy) in zip(all_det, pts_can):
            # Reject wildly out-of-plane points (perspective transform can blow up)
            if not (0 <= cx <= self.canonical_size[0] * 1.5 and
                    -self.canonical_size[1] * 0.2 <= cy <= self.canonical_size[1] * 1.5):
                continue
            self._update_kp(i, cx, cy, c, learn_rate)

        self.frames_seen += 1
        return {"inliers": int(inliers.sum()), "matched": len(src),
                "rms_px": rms_px, "n_lut": len(self.stats)}

    def _update_kp(
        self,
        idx: int,
        x:   float,
        y:   float,
        conf: float,
        learn_rate: float,
    ) -> None:
        """EMA update for a single keypoint."""
        if idx not in self.stats:
            self.stats[idx] = KeypointStats(x=float(x), y=float(y), conf_sum=float(conf))
            return
        s = self.stats[idx]
        s.conf_sum += conf
        # learning rate decays as we accumulate confidence
        a = min(learn_rate, conf / s.conf_sum)
        # Welford-esque weighted mean+variance
        dx, dy = x - s.x, y - s.y
        s.x += a * dx
        s.y += a * dy
        new_dx, new_dy = x - s.x, y - s.y
        s.var_x = (1 - a) * s.var_x + a * new_dx * dx
        s.var_y = (1 - a) * s.var_y + a * new_dy * dy
        s.n   += 1

    # ---- queries ------------------------------------------------------------

    def trusted_kps(
        self,
        min_n:   int   = 5,
        max_std: float = 15.0,
    ) -> dict[int, tuple[float, float]]:
        """Return {kp_idx: (x, y)} for kps stable enough to use for homography."""
        out: dict[int, tuple[float, float]] = {}
        for i, s in self.stats.items():
            if s.n >= min_n and s.std <= max_std:
                out[i] = (s.x, s.y)
        return out

    def quality_report(self) -> dict:
        if not self.stats:
            return {"n_kps": 0, "frames_seen": self.frames_seen}
        stds = [s.std for s in self.stats.values()]
        return {
            "n_kps":       len(self.stats),
            "frames_seen": self.frames_seen,
            "mean_std_px": float(np.mean(stds)),
            "median_std":  float(np.median(stds)),
            "trusted":     len(self.trusted_kps()),
        }

    # ---- persist ------------------------------------------------------------

    def save(self, path: Path = LUT_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "canonical_size": list(self.canonical_size),
            "frames_seen":    self.frames_seen,
            "stats":          {str(i): s.to_dict() for i, s in self.stats.items()},
        }, indent=2))
        return path

    @classmethod
    def load(cls, path: Path = LUT_PATH) -> "AdaptiveRinkLUT":
        if not path.exists():
            return cls()
        d = json.loads(path.read_text())
        lut = cls(
            frames_seen=int(d.get("frames_seen", 0)),
            canonical_size=tuple(d.get("canonical_size", (1280, 720))),
            stats={int(i): KeypointStats.from_dict(v) for i, v in d.get("stats", {}).items()},
        )
        return lut
