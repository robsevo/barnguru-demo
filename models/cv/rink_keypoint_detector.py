"""Phase 16.5 — Rink Keypoint Detector.

Heatmap regression CNN that detects 7 rink landmarks in a broadcast frame:
  - center_ice_dot          — centre-ice faceoff dot
  - blue_line_left          — midpoint of left/home blue line
  - blue_line_right         — midpoint of right/away blue line
  - faceoff_home_left       — home defensive-zone faceoff circle, left side
  - faceoff_home_right      — home defensive-zone faceoff circle, right side
  - faceoff_away_left       — away defensive-zone faceoff circle, left side
  - faceoff_away_right      — away defensive-zone faceoff circle, right side

Architecture::

    Conv(3→32, 3×3) + BN + ReLU + MaxPool(2×2)   → 32 × 128 × 224
    Conv(32→64, 3×3) + BN + ReLU + MaxPool(2×2)  → 64 ×  64 × 112
    Conv(64→128, 3×3) + BN + ReLU                 → 128 ×  64 × 112
    Conv(128→7, 1×1) + Sigmoid                    →   7 ×  64 × 112  (heatmaps)

    Input: (3, 256, 448) normalised float tensor (ImageNet mean/std)
    Stride 4 encoder → heatmap resolution (64, 112)

At inference each heatmap channel is reduced to a single (x, y) pixel coordinate
via argmax, then scaled back to the original image size.
Confidence = peak value in the heatmap channel (already ∈ [0, 1] after Sigmoid).

Quality gate: PCK@10 ≥ 0.80 on held-out annotated frames (80% of keypoints
within 10 px of ground truth at original resolution).

Model saved to models/cv/rink_keypoint_detector.pt

Usage::

    from models.cv.rink_keypoint_detector import RinkKeypointDetector

    det = RinkKeypointDetector()           # loads default model
    kps = det.detect(frame_bgr)
    # kps["center_ice_dot"] → (x_px, y_px, confidence) or None (not visible)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO        = Path(__file__).parents[2]
DEFAULT_PT   = _REPO / "models" / "cv" / "rink_keypoint_detector.pt"
METRICS_JSON = _REPO / "models" / "cv" / "rink_keypoint_detector_metrics.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_KEYPOINTS = 7
INPUT_H       = 256    # model input height (pixels)
INPUT_W       = 448    # model input width  (pixels, 16:9 ≈ 448/256)
HMAP_H        = 64     # heatmap height (INPUT_H // 4)
HMAP_W        = 112    # heatmap width  (INPUT_W // 4)
CONF_THR      = 0.30   # minimum heatmap peak to count as visible
PCK_GATE      = 0.80   # quality gate: PCK@10 ≥ 80%
PCK_RADIUS    = 10     # pixels at original image resolution

# ImageNet normalisation (standard for fine-tuned models)
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)

KEYPOINT_NAMES: list[str] = [
    "center_ice_dot",
    "blue_line_left",
    "blue_line_right",
    "faceoff_home_left",
    "faceoff_home_right",
    "faceoff_away_left",
    "faceoff_away_right",
]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RinkKeypointNet(nn.Module):
    """Lightweight heatmap regression CNN for rink keypoint detection.

    Stride-4 encoder outputs one heatmap per keypoint (Sigmoid activation).
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1 — stride 2
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 × 224
            # Block 2 — stride 4
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 × 112
            # Block 3 — no stride
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(128, NUM_KEYPOINTS, 1),  # 1×1 projection
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 256, 448) → heatmaps: (B, 7, 64, 112)"""
        feats = self.encoder(x)
        return self.heatmap_head(feats)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _preprocess(bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 (H, W, 3) → normalised float tensor (1, 3, INPUT_H, INPUT_W)."""
    import cv2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    for c, (m, s) in enumerate(zip(_MEAN, _STD)):
        arr[:, :, c] = (arr[:, :, c] - m) / s
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def _heatmaps_to_keypoints(
    heatmaps: np.ndarray,
    orig_h: int,
    orig_w: int,
    conf_thr: float = CONF_THR,
) -> dict[str, tuple[float, float, float] | None]:
    """Convert (7, HMAP_H, HMAP_W) heatmaps to pixel coords in original image.

    Returns ``{name: (x, y, conf)}`` or ``{name: None}`` if below threshold.
    """
    scale_x = orig_w / HMAP_W
    scale_y = orig_h / HMAP_H

    result: dict[str, tuple[float, float, float] | None] = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        hm = heatmaps[i]                     # (HMAP_H, HMAP_W)
        conf = float(hm.max())
        if conf < conf_thr:
            result[name] = None
        else:
            flat_idx = int(hm.argmax())
            hy = flat_idx // HMAP_W
            hx = flat_idx % HMAP_W
            # Scale from heatmap space to original image space (+0.5 for centre of cell)
            px = (hx + 0.5) * scale_x
            py = (hy + 0.5) * scale_y
            result[name] = (px, py, conf)
    return result


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class RinkKeypointDetector:
    """Rule-based rink keypoint detector using heatmap regression.

    Args:
        model_path: Path to ``rink_keypoint_detector.pt``.
                    Defaults to ``models/cv/rink_keypoint_detector.pt``.
        device:     ``"cuda"`` | ``"cpu"`` | ``None`` (auto-detect).
        conf_thr:   Minimum heatmap confidence to count a keypoint visible.
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_PT,
        device: str | None = None,
        conf_thr: float = CONF_THR,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Rink keypoint detector model not found: {model_path}\n"
                "Run `uv run python scripts/gretzky.py train-rink-kp` to train it."
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device   = torch.device(device)
        self._conf_thr = conf_thr

        self._net = RinkKeypointNet().to(self._device)
        state = torch.load(model_path, map_location=self._device, weights_only=True)
        self._net.load_state_dict(state)
        self._net.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        frame_bgr: np.ndarray,
    ) -> dict[str, tuple[float, float, float] | None]:
        """Detect rink keypoints in a broadcast frame.

        Args:
            frame_bgr: Full broadcast frame (H, W, 3) BGR uint8.

        Returns:
            Dict mapping keypoint name to ``(x_px, y_px, confidence)`` at the
            *original* frame resolution, or ``None`` if keypoint not visible
            (confidence below threshold).
        """
        orig_h, orig_w = frame_bgr.shape[:2]
        tensor = _preprocess(frame_bgr).to(self._device)
        with torch.no_grad():
            heatmaps = self._net(tensor)[0].cpu().numpy()   # (7, 64, 112)
        return _heatmaps_to_keypoints(heatmaps, orig_h, orig_w, self._conf_thr)

    def detect_batch(
        self,
        frames: list[np.ndarray],
    ) -> list[dict[str, tuple[float, float, float] | None]]:
        """Detect keypoints in a batch of frames."""
        return [self.detect(f) for f in frames]

    @property
    def keypoint_names(self) -> list[str]:
        return list(KEYPOINT_NAMES)

    def __repr__(self) -> str:
        return (
            f"RinkKeypointDetector(keypoints={NUM_KEYPOINTS}, "
            f"input=({INPUT_H},{INPUT_W}), heatmap=({HMAP_H},{HMAP_W}))"
        )


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def check_quality_gate(
    metrics: dict,
) -> tuple[bool, float]:
    """Return ``(passed, pck_at_10)`` for a metrics dict from training.

    Metrics dict must contain ``"pck_at_10"`` key (float 0–1).
    """
    pck = float(metrics.get("pck_at_10", 0.0))
    return pck >= PCK_GATE, pck
