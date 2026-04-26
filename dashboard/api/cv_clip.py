"""One-shot CV processing for a Brightcove highlight clip.

Given a resolved HLS URL, runs the full detect → track → homography pipeline
to completion and writes a cache file at ``data/cv_clip_cache/<clip>.json``:

    {
      "clip_id":      "<bc_id>",
      "fps":          6.0,
      "total_frames": N,
      "duration_s":   float,
      "frames": [
        {
          "frame_seq":    i,
          "timestamp_ms": int,
          "players": [
            {
              "track_id":     int,
              "class_id":     int,
              "class_name":   str,
              "cx_n":  float, "cy_n":  float,
              "w_n":   float, "h_n":   float,
              "speed_kmh":    float,
              "jersey_number": null,
              "x":            float | null,   # NHL rink feet
              "y":            float | null,
            },
            ...
          ]
        },
        ...
      ]
    }

Cut-driven re-init
------------------
Highlight clips jump-cut every ~3-5 s. That's fine for bounding boxes but kills
ID persistence — when a new scene appears the old tracks are stale. We detect a
large drop in active track count (≥ 30% decrease) and reset the Rust tracker so
the next frame starts fresh instead of wasting slots on stale IDs.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]
CLIP_CACHE_DIR = _REPO / "data" / "cv_clip_cache"

# Frame dimensions + FPS — must match detector training
_FRAME_W = 1280
_FRAME_H = 720
_FRAME_BYTES = _FRAME_W * _FRAME_H * 3
_FPS = 6.0

# Speed conversion: px/frame → km/h
# 1 px ≈ 0.03048 ft on a 1280×720 broadcast (≈ 1 m), then ×fps ×3.6 → km/h
_PX_FRAME_TO_KMH = 0.03048 * _FPS * 3.6

# Homography refresh cadence (frames)
_HOMOGRAPHY_EVERY = 30

# Jump-cut reset heuristic
_CUT_RESET_THRESHOLD = 0.30

# Minimum confidence for a puck detection to be fed into the Kalman tracker.
# A 5-px puck is the hardest class for the detector, so the top-1 detection
# can be a logo edge, a linesman's arm signal, or a stick blade. Requiring
# at least 0.35 confidence kills most of the garbage before it gets the
# chance to yank the tracker across the ice.
_PUCK_MIN_CONF = 0.35


class ClipJobStatus:
    """Thread-safe status object read by the status endpoint.

    Holds partial results while the detect loop runs so the frames endpoint
    can serve overlays progressively — the frontend doesn't have to wait for
    the full clip to finish before seeing bounding boxes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status: str = "processing"    # processing | ready | error
        self.frames_done: int = 0
        self.total_frames_hint: int = 0    # best-effort; 0 if unknown
        self.message: Optional[str] = None
        self.started_at: float = time.time()
        self.finished_at: Optional[float] = None
        self.frames: list[dict] = []       # appended as each frame is processed

    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def append_frame(self, frame: dict) -> None:
        with self._lock:
            self.frames.append(frame)
            self.frames_done = len(self.frames)

    def get_frames_slice(self, from_seq: int, limit: int) -> list[dict]:
        """Return frames with frame_seq >= from_seq, up to *limit* entries."""
        with self._lock:
            if from_seq >= len(self.frames):
                return []
            return list(self.frames[from_seq : from_seq + limit])

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "status":       self.status,
                "frames_done":  self.frames_done,
                "total_frames": self.total_frames_hint or self.frames_done,
                "message":      self.message,
                "started_at":   self.started_at,
                "finished_at":  self.finished_at,
            }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_clip(
    clip_id: str,
    hls_url: str,
    status:  ClipJobStatus,
    game_id: Optional[int] = None,
) -> Path:
    """Process one clip end-to-end. Returns the path to the written cache file.

    Raises on ffmpeg / model / IO failure — the caller (FastAPI background
    task) is responsible for catching and recording the error on ``status``.
    """
    from models.cv.player_detector import PlayerDetector, CLASS_NAMES
    from models.cv.adaptive_runtime import AdaptiveRinkRuntime
    from data.puck_tracker import PuckTracker

    try:
        from gretzky_engine import Tracker  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(f"gretzky_engine not built: {exc}. Run: make build-engine") from exc

    # Detector class indices (single source of truth = player_detector.py).
    _PUCK_CLASS_ID    = CLASS_NAMES.index("puck")
    _REFEREE_CLASS_ID = CLASS_NAMES.index("referee")
    _GOALIE_CLASS_ID  = CLASS_NAMES.index("goalie")
    _PLAYER_CLASS_ID  = CLASS_NAMES.index("player")

    CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    log.info("[cv-clip:%s] starting (hls=%s)", clip_id, hls_url[:80])

    detector   = PlayerDetector()
    rink_rt    = AdaptiveRinkRuntime()
    tracker    = Tracker(fps=_FPS, n_init=3, max_lost=15, iou_threshold=0.25, reid_iou=0.30)
    puck_tr    = PuckTracker()

    # Team classifier — needed to split detector "player" / "goalie" into the
    # legacy player_home / player_away buckets the highlight overlay expects.
    # Best-effort: if the LUT is missing or the NHL landing fetch fails the
    # clip still processes, just without home/away colour separation.
    color_clf = None
    home_team: str | None = None
    away_team: str | None = None
    try:
        from models.cv.team_color_classifier import TeamColorClassifier
        color_clf = TeamColorClassifier()
    except Exception as exc:
        log.warning("[cv-clip:%s] team color classifier unavailable: %s", clip_id, exc)
    if game_id is not None:
        try:
            import asyncio as _aio_clip
            from data.nhl_client import NHLClient
            async def _fetch_teams():
                async with NHLClient() as cl:
                    return await cl.get_landing(game_id)
            landing = _aio_clip.run(_fetch_teams())
            if isinstance(landing, dict):
                home_team = (landing.get("homeTeam", {}) or {}).get("abbrev")
                away_team = (landing.get("awayTeam", {}) or {}).get("abbrev")
                home_team = home_team.upper() if home_team else None
                away_team = away_team.upper() if away_team else None
        except Exception as exc:
            log.warning("[cv-clip:%s] could not fetch team codes for game_id=%s: %s",
                        clip_id, game_id, exc)

    def _resolve_class_name(cls_id: int, team: str | None) -> str:
        """Translate detector 4-class output to legacy 6-class label.

        Mirrors data/cv_tracker.py:_resolve_class_name. Falls back to home
        when the team classifier is unsure or unconfigured — better than
        dropping the player from the overlay.
        """
        if cls_id == _PUCK_CLASS_ID:
            return "puck"
        if cls_id == _REFEREE_CLASS_ID:
            return "referee"
        if cls_id == _PLAYER_CLASS_ID:
            base = "player"
        elif cls_id == _GOALIE_CLASS_ID:
            base = "goalie"
        else:
            return "unknown"
        team_u = (team or "").upper()
        if away_team and team_u == away_team:
            return f"{base}_away"
        return f"{base}_home"

    # Probe duration so the UI can show an ETA + progress denominator.
    # Best-effort — if ffprobe fails we just leave the hint at 0.
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                hls_url,
            ],
            capture_output=True, timeout=20, text=True,
        )
        dur_s = float(probe.stdout.strip()) if probe.returncode == 0 else 0.0
        if dur_s > 0:
            status.update(total_frames_hint=int(dur_s * _FPS))
    except Exception as exc:
        log.debug("[cv-clip:%s] ffprobe failed: %s", clip_id, exc)

    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel", "error",
            "-i", hls_url,
            "-vf", f"fps={_FPS},scale={_FRAME_W}:{_FRAME_H}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=_FRAME_BYTES * 2,
    )

    frames_out: list[dict] = []
    homography: Optional[np.ndarray] = None
    frame_n  = 0
    last_track_count = 0

    try:
        while True:
            raw = proc.stdout.read(_FRAME_BYTES)
            if len(raw) < _FRAME_BYTES:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(_FRAME_H, _FRAME_W, 3)

            # Homography refresh
            if frame_n % _HOMOGRAPHY_EVERY == 0:
                try:
                    new_H = rink_rt.update(frame)
                    if new_H is not None:
                        homography = new_H
                except Exception:
                    pass

            try:
                all_dets = detector.detect(frame)
            except Exception as exc:
                log.debug("[cv-clip:%s] detect error f=%d: %s", clip_id, frame_n, exc)
                frame_n += 1
                continue

            puck_dets   = [d for d in all_dets
                           if d.class_name == "puck" and d.confidence >= _PUCK_MIN_CONF]
            player_dets = [d for d in all_dets if d.class_name != "puck"]

            tracker_input = [
                [d.x1, d.y1, d.w, d.h, d.confidence, float(d.class_id)]
                for d in player_dets
            ]
            try:
                tracks = tracker.update(tracker_input)
            except Exception as exc:
                log.debug("[cv-clip:%s] tracker error f=%d: %s", clip_id, frame_n, exc)
                tracks = []

            # Jump-cut detection: large drop in active track count → reset
            if last_track_count >= 6 and len(tracks) < last_track_count * (1 - _CUT_RESET_THRESHOLD):
                # Rebuild tracker to clear stale tracks
                tracker = Tracker(fps=_FPS, n_init=3, max_lost=15, iou_threshold=0.25, reid_iou=0.30)
                try:
                    tracks = tracker.update(tracker_input)
                except Exception:
                    tracks = []
            last_track_count = len(tracks)

            players_out: list[dict] = []

            for t in tracks:
                tcx, tcy = float(t["cx"]), float(t["cy"])
                tw,  th  = float(t["w"]),  float(t["h"])
                tvx, tvy = float(t["vx"]), float(t["vy"])
                tcls     = int(t["class_id"])
                tid      = int(t["id"])

                rink_x = rink_y = None
                if homography is not None:
                    try:
                        xy = rink_rt.project(tcx, tcy, homography)
                        if xy is not None:
                            rink_x, rink_y = xy
                    except Exception:
                        pass

                # Team classification — only meaningful for player/goalie.
                team_code: str | None = None
                if (
                    color_clf is not None
                    and tcls in (_PLAYER_CLASS_ID, _GOALIE_CLASS_ID)
                ):
                    x1 = max(0, int(tcx - tw / 2))
                    y1 = max(0, int(tcy - th / 2))
                    x2 = min(_FRAME_W, int(tcx + tw / 2))
                    y2 = min(_FRAME_H, int(tcy + th / 2))
                    if x2 > x1 and y2 > y1:
                        try:
                            team_code, _ = color_clf.classify(frame[y1:y2, x1:x2])
                        except Exception:
                            team_code = None

                speed_px = math.sqrt(tvx * tvx + tvy * tvy)
                players_out.append({
                    "track_id":      tid,
                    "class_id":      tcls,
                    "class_name":    _resolve_class_name(tcls, team_code),
                    "team":          team_code,
                    "cx_n":          round(tcx / _FRAME_W, 4),
                    "cy_n":          round(tcy / _FRAME_H, 4),
                    "w_n":           round(tw  / _FRAME_W, 4),
                    "h_n":           round(th  / _FRAME_H, 4),
                    "speed_kmh":     round(speed_px * _PX_FRAME_TO_KMH, 1),
                    "jersey_number": None,
                    "x":             round(rink_x, 2) if rink_x is not None else None,
                    "y":             round(rink_y, 2) if rink_y is not None else None,
                })

            # Puck
            puck_meas = None
            if puck_dets:
                best = max(puck_dets, key=lambda d: d.confidence)
                puck_meas = (float(best.cx), float(best.cy), float(best.confidence))
            puck_state = puck_tr.update(puck_meas)
            if puck_state is not None:
                rink_x = rink_y = None
                if homography is not None:
                    try:
                        xy = rink_rt.project(puck_state.cx, puck_state.cy, homography)
                        if xy is not None:
                            rink_x, rink_y = xy
                    except Exception:
                        pass
                players_out.append({
                    "track_id":      -1,
                    "class_id":      _PUCK_CLASS_ID,
                    "class_name":    "puck",
                    "cx_n":          round(puck_state.cx / _FRAME_W, 4),
                    "cy_n":          round(puck_state.cy / _FRAME_H, 4),
                    "w_n":           0.005,
                    "h_n":           0.008,
                    "speed_kmh":     round(puck_state.speed_px * _PX_FRAME_TO_KMH, 1),
                    "jersey_number": None,
                    "x":             round(rink_x, 2) if rink_x is not None else None,
                    "y":             round(rink_y, 2) if rink_y is not None else None,
                })

            one_frame = {
                "frame_seq":    frame_n,
                "timestamp_ms": int(frame_n * 1000 / _FPS),
                "players":      players_out,
            }
            frames_out.append(one_frame)
            # Publish progressively so the frames endpoint can stream
            # partial results to the overlay as detection runs.
            status.append_frame(one_frame)

            frame_n += 1
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    ret = proc.returncode
    if ret not in (0, None) and not frames_out:
        err_tail = (proc.stderr.read(2000).decode("utf-8", errors="replace")
                    if proc.stderr is not None else "")
        raise RuntimeError(f"ffmpeg failed (code {ret}): {err_tail.strip()[-500:]}")

    # Persist to JSON cache
    cache_path = CLIP_CACHE_DIR / f"{clip_id}.json"
    duration_s = frames_out[-1]["timestamp_ms"] / 1000.0 if frames_out else 0.0
    payload = {
        "clip_id":      clip_id,
        "game_id":      game_id,
        "fps":          _FPS,
        "total_frames": len(frames_out),
        "duration_s":   round(duration_s, 2),
        "frames":       frames_out,
    }
    tmp = cache_path.with_name(cache_path.name + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(cache_path)

    log.info("[cv-clip:%s] done — %d frames, %.2fs duration", clip_id, len(frames_out), duration_s)
    status.update(status="ready", frames_done=len(frames_out), finished_at=time.time())
    return cache_path


# ---------------------------------------------------------------------------
# Cache lookup helpers (used by the GET endpoints)
# ---------------------------------------------------------------------------

def cache_path_for(clip_id: str) -> Path:
    return CLIP_CACHE_DIR / f"{clip_id}.json"


def load_cache(clip_id: str) -> Optional[dict]:
    """Return the parsed cache JSON or None if missing / corrupt."""
    p = cache_path_for(clip_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None
