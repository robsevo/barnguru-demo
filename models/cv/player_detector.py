"""Phase 16.2 — Player Detection Model (inference wrapper).

Loads the fine-tuned YOLO model from ``models/cv/player_detector.pt`` and
exposes a clean ``detect(frame)`` API used by the CV worker (16.7).

Classes (must match training config):
    0  player_home
    1  player_away
    2  goalie_home
    3  goalie_away
    4  referee
    5  puck

Usage::

    from models.cv.player_detector import PlayerDetector

    detector = PlayerDetector()           # loads from default path
    detections = detector.detect(frame)   # frame: np.ndarray BGR (H, W, 3)

    for d in detections:
        print(d.class_name, d.confidence, d.cx, d.cy)

The model runs on GPU if available; falls back to CPU automatically.
Inference is thread-safe — each ``detect()`` call is independent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO            = Path(__file__).parents[2]
DEFAULT_PT       = _REPO / "models" / "cv" / "player_detector.pt"
DEFAULT_ONNX     = _REPO / "models" / "cv" / "player_detector.onnx"
DEFAULT_ONNX_INT8 = _REPO / "models" / "cv" / "player_detector.int8.onnx"
METRICS_JSON     = _REPO / "models" / "cv" / "detector_metrics.json"

# Browser runtime loads at this size — keep ONNX export in sync with
# `dashboard/frontend/app/game/[gameId]/cvWorker.ts` (IN_SIZE).
RUNTIME_IMGSZ = 320

# ---------------------------------------------------------------------------
# Class registry — index must match training dataset.yaml
# ---------------------------------------------------------------------------

CLASS_NAMES: list[str] = [
    "player",        # 0  — home/away decided post-detection by torso saturation
    "goalie",        # 1  — home/away decided post-detection by torso saturation
    "referee",       # 2
    "puck",          # 3
]

NUM_CLASSES = len(CLASS_NAMES)

# Quality gate from PLAN.md §16.2
MAP50_GATE = 0.82


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    class_id:   int
    class_name: str
    confidence: float
    x1: float       # pixel coords (top-left)
    y1: float
    x2: float       # pixel coords (bottom-right)
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    def to_dict(self) -> dict:
        return {
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "cx": round(self.cx, 1),
            "cy": round(self.cy, 1),
            "w":  round(self.w, 1),
            "h":  round(self.h, 1),
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class PlayerDetector:
    """YOLO-based player + puck detector for NHL broadcast frames.

    Args:
        model_path:  Path to the fine-tuned ``.pt`` file.
                     Defaults to ``models/cv/player_detector.pt``.
        conf:        Minimum confidence threshold (default 0.25).
        iou:         NMS IoU threshold (default 0.45).
        device:      ``"cuda"``, ``"cpu"``, or ``None`` for auto-select.
        half:        Use FP16 inference on GPU (faster, requires CUDA).
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_PT,
        conf: float = 0.25,
        iou:  float = 0.45,
        device: str | None = None,
        half: bool = False,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Player detector model not found: {model_path}\n"
                "Run `uv run python scripts/gretzky.py train-cv-detector` to build it."
            )

        from ultralytics import YOLO
        import torch

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._conf   = conf
        self._iou    = iou
        self._half   = half and self._device == "cuda"

        self._model = YOLO(str(model_path))
        self._model.to(self._device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        conf: float | None = None,
        iou:  float | None = None,
    ) -> list[Detection]:
        """Run detection on a single BGR frame (OpenCV format).

        Args:
            frame: ``np.ndarray`` of shape ``(H, W, 3)``, dtype uint8, BGR.
            conf:  Override confidence threshold for this call.
            iou:   Override NMS IoU threshold for this call.

        Returns:
            List of :class:`Detection` objects sorted by confidence (desc).
        """
        results = self._model.predict(
            source=frame,
            conf=conf or self._conf,
            iou=iou or self._iou,
            half=self._half,
            device=self._device,
            verbose=False,
            imgsz=640,
        )

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id  = int(boxes.cls[i].item())
                conf_v  = float(boxes.conf[i].item())
                xyxy    = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                cls_name = CLASS_NAMES[cls_id] if cls_id < NUM_CLASSES else f"class_{cls_id}"
                detections.append(Detection(cls_id, cls_name, conf_v, x1, y1, x2, y2))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_batch(
        self,
        frames: list[np.ndarray],
        conf: float | None = None,
        iou:  float | None = None,
    ) -> list[list[Detection]]:
        """Detect on a batch of frames. Returns one list per frame."""
        results = self._model.predict(
            source=frames,
            conf=conf or self._conf,
            iou=iou or self._iou,
            half=self._half,
            device=self._device,
            verbose=False,
            imgsz=640,
        )
        output = []
        for result in results:
            dets: list[Detection] = []
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    cls_id  = int(boxes.cls[i].item())
                    conf_v  = float(boxes.conf[i].item())
                    xyxy    = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                    cls_name = CLASS_NAMES[cls_id] if cls_id < NUM_CLASSES else f"class_{cls_id}"
                    dets.append(Detection(cls_id, cls_name, conf_v, x1, y1, x2, y2))
            dets.sort(key=lambda d: d.confidence, reverse=True)
            output.append(dets)
        return output

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        return self._device

    def warm_up(self) -> None:
        """Run one dummy inference to load CUDA kernels / JIT compile."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy)

    def __repr__(self) -> str:
        return f"PlayerDetector(device={self._device!r}, conf={self._conf}, iou={self._iou})"


# ---------------------------------------------------------------------------
# Saved metrics loader
# ---------------------------------------------------------------------------

def load_metrics() -> dict | None:
    """Load the training metrics saved by train_cv_detector.py, or None."""
    if not METRICS_JSON.exists():
        return None
    with open(METRICS_JSON) as fh:
        return json.load(fh)


def check_quality_gate(metrics: dict | None = None) -> tuple[bool, float]:
    """Return (passed, map50) checking the mAP@0.5 ≥ 0.82 gate.

    If no metrics file exists, returns (False, 0.0).
    """
    if metrics is None:
        metrics = load_metrics()
    if metrics is None:
        return False, 0.0
    map50 = float(metrics.get("map50", 0.0))
    return map50 >= MAP50_GATE, map50
