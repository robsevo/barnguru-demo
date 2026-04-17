"""Puck pseudo-label cache for weekly detector retraining.

Caches full broadcast frames + multi-class YOLO labels whenever the runtime
detector produces a high-confidence puck detection that survives the Kalman
consistency gate. Weekly retrain merges these frames with the HockeyAI base
set to lift the puck mAP over time without human labeling.

Gate (all must hold):
    1. raw puck confidence >= ``min_conf``
    2. Kalman ``PuckState`` is active, not occluded, not just-reset (``lost``)
    3. puck track has been active for >= ``min_track_frames``
    4. since last cached sample for this game:
       moved >= ``min_dist_px`` OR elapsed >= ``min_time_gap_s``
    5. weekly sample count < ``max_per_week``

Layout::

    data/cv_training/pseudo_labels/<YYYY>-W<WW>/images/g<game>_f<frame>.jpg
    data/cv_training/pseudo_labels/<YYYY>-W<WW>/labels/g<game>_f<frame>.txt

Label file is full-frame multi-class YOLO normalized (``class cx cy w h``).
Class ids match the GRTZKY 6-class schema in ``models/cv/player_detector.py``.
Only detections with confidence >= ``min_label_conf`` are written — low-conf
players/refs are dropped so they don't poison the next model.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = _REPO / "data" / "cv_training" / "pseudo_labels"


@dataclass
class _GameState:
    last_cx: float = -1e9
    last_cy: float = -1e9
    last_ts: float = 0.0


class PuckPseudoCache:
    """Selectively writes broadcast frames as YOLO training samples."""

    def __init__(
        self,
        root: Path | str = DEFAULT_ROOT,
        min_conf:         float = 0.85,
        min_label_conf:   float = 0.5,
        min_track_frames: int   = 3,
        min_dist_px:      float = 20.0,
        min_time_gap_s:   float = 2.0,
        max_per_week:     int   = 2000,
    ) -> None:
        self._root             = Path(root)
        self._min_conf         = min_conf
        self._min_label_conf   = min_label_conf
        self._min_track_frames = min_track_frames
        self._min_dist_px      = min_dist_px
        self._min_time_gap_s   = min_time_gap_s
        self._max_per_week     = max_per_week

        self._games: dict[int, _GameState] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _iso_week_key(ts: float) -> str:
        iso = datetime.fromtimestamp(ts, tz=timezone.utc).isocalendar()
        return f"{iso[0]:04d}-W{iso[1]:02d}"

    def _week_dirs(self, ts: float) -> tuple[Path, Path]:
        wk = self._iso_week_key(ts)
        return self._root / wk / "images", self._root / wk / "labels"

    def _week_count(self, ts: float) -> int:
        img_dir, _ = self._week_dirs(ts)
        if not img_dir.is_dir():
            return 0
        return sum(1 for _ in img_dir.glob("*.jpg"))

    def consider(
        self,
        frame:              np.ndarray,
        detections:         list,                                  # list[Detection]
        puck_raw:           Optional[tuple[float, float, float]],  # (cx, cy, conf)
        puck_state,                                                 # PuckState | None
        puck_frames_active: int,
        game_id:            int,
        frame_n:            int,
        timestamp:          float,
    ) -> bool:
        """Return True iff this frame was written to the cache."""
        if puck_raw is None:
            return False
        raw_cx, raw_cy, raw_conf = puck_raw
        if raw_conf < self._min_conf:
            return False

        if puck_state is None or puck_state.occluded or puck_state.lost:
            return False
        if puck_frames_active < self._min_track_frames:
            return False

        with self._lock:
            gs = self._games.setdefault(game_id, _GameState())
            dx = raw_cx - gs.last_cx
            dy = raw_cy - gs.last_cy
            dist = math.sqrt(dx * dx + dy * dy)
            dt = timestamp - gs.last_ts
            if dist < self._min_dist_px and dt < self._min_time_gap_s:
                return False

            if self._week_count(timestamp) >= self._max_per_week:
                return False

            gs.last_cx = raw_cx
            gs.last_cy = raw_cy
            gs.last_ts = timestamp

        try:
            self._write(frame, detections, game_id, frame_n, timestamp)
            return True
        except Exception as exc:
            log.debug("[puck-cache] write failed g=%d f=%d: %s", game_id, frame_n, exc)
            return False

    def _write(
        self,
        frame:     np.ndarray,
        detections: list,
        game_id:   int,
        frame_n:   int,
        timestamp: float,
    ) -> None:
        import cv2

        img_dir, lbl_dir = self._week_dirs(timestamp)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        stem = f"g{game_id}_f{frame_n:06d}"
        img_path = img_dir / f"{stem}.jpg"
        lbl_path = lbl_dir / f"{stem}.txt"

        H, W = frame.shape[:2]
        lines: list[str] = []
        for d in detections:
            if d.confidence < self._min_label_conf:
                continue
            cx = (d.x1 + d.x2) / 2 / W
            cy = (d.y1 + d.y2) / 2 / H
            w  = (d.x2 - d.x1) / W
            h  = (d.y2 - d.y1) / H
            if w <= 0 or h <= 0:
                continue
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w  = max(0.0, min(1.0, w))
            h  = max(0.0, min(1.0, h))
            lines.append(f"{d.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not lines:
            return

        lbl_tmp = lbl_path.with_suffix(".txt.tmp")
        lbl_tmp.write_text("\n".join(lines) + "\n")
        os.replace(lbl_tmp, lbl_path)

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        img_tmp = img_path.with_name(img_path.name + ".tmp")
        img_tmp.write_bytes(buf.tobytes())
        os.replace(img_tmp, img_path)

    def stats(self) -> dict[str, int]:
        """Return {week: n_samples} for every non-empty week under root."""
        out: dict[str, int] = {}
        if not self._root.is_dir():
            return out
        for wk in sorted(self._root.iterdir()):
            if not wk.is_dir():
                continue
            img_dir = wk / "images"
            if not img_dir.is_dir():
                continue
            n = sum(1 for _ in img_dir.glob("*.jpg"))
            if n:
                out[wk.name] = n
        return out
